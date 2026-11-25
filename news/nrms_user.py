'''
NRMS-style user encoder on frozen MiniLM article vectors (news study, step 1 of the neural plan).

The article vectors (output/minilm_title_abstract_entities_all) stay fixed. A small projection of them and an
attention-based user encoder (multi-head self-attention over the last 50 history articles, then additive
pooling) are trained end to end on the clicks of Nov 9-13: each click against NEG non-clicked articles from the
same impression (softmax over 1 + NEG). The score of a candidate is the dot product of user and article vector.
Settings are chosen on validation (Nov 14) only; dev and test are not touched here.

Variants (every one is a separate selection on Nov 14 and is reported, whatever its result):
    base               the first run (doc 14): content only, 4 negatives, dropout 0.2, 3 epochs
    pop                + popularity: score = u.c + w[group] * standardised log cum_ctr, one learned weight per
                         history group (0 / 1-3 / 4-10 / >10); Nov 9 rows have no cum_ctr and get 0 (no effect)
    pop_pos            + recency: a learned vector per history slot (slot 49 = the latest click)
    pop_pos_neg8       + 8 negatives per click instead of 4
    pop_pos_neg8_long  + dropout 0.1 and 5 epochs
    pop_pos_neg8_no9   pop_pos_neg8 trained on Nov 10-13 only (Nov 9 has no cum_ctr, so there the empty-history
                         vector may learn to stand in for popularity)
    pop_pos_neg8_cscale  pop_pos_neg8 + a learned content weight per history group: score = v[group] * u.c +
                         w[group] * pop, so new users can lean on popularity

Command line (repository root, GPU build of PyTorch):
    python news/nrms_user.py check [variant]   # checks on small data + untrained AUC + overfit test; writes nothing
    python news/nrms_user.py val <variant>     # train on Nov 9-13, pick the epoch on a val sample, score all of val
    python news/nrms_user.py val <variant> <seed>   # seed check: same variant, another seed (initial weights,
                                                    # shuffling, negatives); output <variant>/seed<seed>/
    python news/nrms_user.py switch <variant>  # empty histories -> cum_ctr, others -> the variant's val scores

Output: output/news/nrms_user/<variant>/ (base: output/news/nrms_user/); results via mind_eval.evaluate.
'''
import json
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import mind_eval as me
import news_protocol as npr
from hybrid_e import load_embeddings, group_means

OUT = me.ROOT / "output" / "news" / "nrms_user"
FEATURES = me.ROOT / "output" / "news" / "features"
MAX_HIST, BATCH, LR, SEED = 50, 512, 1e-3, 42
VAL_SAMPLE_SEED = 43
DIM, HEADS = 256, 8
VAL_SAMPLE = 50_000          # impressions of val scored after every epoch (fixed sample) to pick the epoch
GROUP_EDGES = np.array([0, 3, 10])   # history length -> group id 0: 0, 1: 1-3, 2: 4-10, 3: >10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BASE = {"pop": False, "pos": False, "neg": 4, "dropout": 0.2, "epochs": 3, "no9": False, "cscale": False}
BEST = {**BASE, "pop": True, "pos": True, "neg": 8}          # best of the first four on val (doc 14)
VARIANTS = {
    "base": BASE,
    "pop": {**BASE, "pop": True},
    "pop_pos": {**BASE, "pop": True, "pos": True},
    "pop_pos_neg8": BEST,
    "pop_pos_neg8_long": {**BEST, "dropout": 0.1, "epochs": 5},
    "pop_pos_neg8_no9": {**BEST, "no9": True},
    "pop_pos_neg8_cscale": {**BEST, "cscale": True},
}
FIRST_FIT_DAY_NO9 = np.datetime64("2019-11-10")


def run_name(variant, seed=SEED):
    """Result name; runs with another seed than 42 get the suffix _seed<k> (seed check, doc 14)."""
    name = "nrms_user_minilm" if variant == "base" else f"nrms_user_{variant}"
    return name if seed == SEED else f"{name}_seed{seed}"


def out_dir(variant, seed=SEED):
    folder = OUT if variant == "base" else OUT / variant
    return folder if seed == SEED else folder / f"seed{seed}"


# ---------------------------------------------------------------------------------------------------- data
def article_table(vectors):
    """Frozen article vectors on the GPU; row 0 = padding, article row k -> k + 1."""
    D = vectors.shape[1]
    pad = np.zeros((1, D), dtype=np.float32)
    table = np.vstack([pad, vectors])
    return torch.from_numpy(table).to(torch.float32).to(DEVICE)


def click_samples(data):
    """One sample per click in the fit split -> (pos, imp): flat click position, its impression index."""
    pos = np.flatnonzero(data.labels == 1)
    imp = np.searchsorted(data.offsets, pos, side="right") - 1
    clicks_per_imp = np.add.reduceat(data.labels, data.offsets[:-1], dtype=np.int64)
    negs_per_imp = data.sizes - clicks_per_imp
    keep = negs_per_imp[imp] > 0
    pos, imp = pos[keep], imp[keep]
    print(f"click_samples: {len(pos)} clicks with at least one negative candidate")
    print(f"click kept: {keep.mean():.1%} of all clicks, {negs_per_imp[imp].mean():.1f} negatives per kept click")
    return pos, imp


def history_rows(data, lookup, imp):
    """Table rows of the last MAX_HIST history articles of each impression in imp, left-padded with 0."""
    start = data.history_offsets[imp]
    end = data.history_offsets[imp + 1]
    indx = end[:, None] - MAX_HIST + np.arange(MAX_HIST)[None, :]
    valid = indx >= start[:, None]
    idx = np.where(valid, indx, 0)
    return np.where(valid, lookup[data.history[idx]], 0)


def history_group(data, imp):
    """History group id (0: empty, 1: 1-3, 2: 4-10, 3: >10) of each impression in imp."""
    lengths = data.history_offsets[imp + 1] - data.history_offsets[imp]
    return np.searchsorted(GROUP_EDGES, lengths, side="left")


def make_batch(data, lookup, pos, imp, rng, neg, pop=None):
    """Batch of click samples -> hist (B, MAX_HIST), cand (B, 1 + neg): table rows; column 0 of cand = the click;
    group (B,); pop (B, 1 + neg) popularity input of each candidate, or None."""
    B = len(pos)
    hist = history_rows(data, lookup, imp)

    cand_pos = np.empty((B, 1 + neg), dtype=np.int64)
    cand_pos[:, 0] = pos
    for k in range(B):
        a, b = data.offsets[imp[k]], data.offsets[imp[k] + 1]
        negs = a + np.flatnonzero(data.labels[a:b] == 0)
        cand_pos[k, 1:] = rng.choice(negs, size=neg, replace=len(negs) < neg)
    cand = lookup[data.candidates[cand_pos]]

    to = lambda x: torch.from_numpy(np.ascontiguousarray(x)).to(DEVICE)
    return to(hist), to(cand), to(history_group(data, imp)), None if pop is None else to(pop[cand_pos])


def popularity_input(split_values, stats=None):
    """cum_ctr -> standardised log; NaN (Nov 9, no earlier day) -> 0. Mean and sd come from the fit split."""
    logs = np.log(split_values.astype(np.float64))
    if stats is None:
        stats = {"mean": float(np.nanmean(logs)), "sd": float(np.nanstd(logs))}
    z = (logs - stats["mean"]) / stats["sd"]
    return np.nan_to_num(z, nan=0.0).astype(np.float32), stats


def load_splits(cfg):
    """fit (Nov 9-13, or Nov 10-13 with no9), val (Nov 14), a fixed val sample, and their popularity inputs
    (None without pop)."""
    train_all = me.load_impressions("train")
    mask = npr.val_mask(train_all)
    fit_mask = ~mask & (train_all.times >= FIRST_FIT_DAY_NO9) if cfg["no9"] else ~mask
    sizes = train_all.sizes
    fit = npr.subset(train_all, fit_mask, "fit")
    val = npr.subset(train_all, mask, "val")
    del train_all
    if cfg["no9"]:
        print(f"no9: fit starts {fit.times.min()} ({fit.n_impressions} impressions)")
        assert fit.times.min() >= FIRST_FIT_DAY_NO9
    keep = np.zeros(val.n_impressions, dtype=bool)
    # the epoch-choice sample is the same for every training seed (seed 43 = SEED + 1 of the first runs)
    keep[np.random.default_rng(VAL_SAMPLE_SEED).choice(val.n_impressions, VAL_SAMPLE, replace=False)] = True
    val_sample = npr.subset(val, keep, "val_sample")
    print(f"fit {fit.n_impressions} impressions, val {val.n_impressions}, val sample {val_sample.n_impressions}")
    pops = {"fit": None, "val": None, "val_sample": None, "stats": None}
    if cfg["pop"]:
        raw = np.load(FEATURES / "cum_ctr_train.npy", mmap_mode="r")
        slot_val, slot_fit = np.repeat(mask, sizes), np.repeat(fit_mask, sizes)
        pops["fit"], stats = popularity_input(raw[slot_fit])
        pops["val"], _ = popularity_input(raw[slot_val], stats)
        pops["val_sample"] = pops["val"][np.repeat(keep, val.sizes)]
        pops["stats"] = stats
        nan_fit = float(np.isnan(raw[slot_fit]).mean())
        nan_val = float(np.isnan(raw[slot_val]).mean())
        stats.update({"nan_share_fit": nan_fit, "nan_share_val": nan_val})
        print(f"popularity: log cum_ctr mean {stats['mean']:.3f} sd {stats['sd']:.3f}; "
              f"NaN share fit {nan_fit:.1%} (Nov 9 rows), val {nan_val:.1%}")
        assert len(pops["fit"]) == fit.n_candidates and len(pops["val"]) == val.n_candidates
    return fit, val, val_sample, pops


# ---------------------------------------------------------------------------------------------------- model
class UserEncoder(nn.Module):
    def __init__(self, dim_in=384, dim=DIM, heads=HEADS, dropout=0.2, use_pop=False, use_pos=False,
                 use_cscale=False):
        super().__init__()
        self.news = nn.Sequential(nn.Linear(dim_in, dim), nn.Tanh())       # projection on the frozen vectors
        self.self_att = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.pool = nn.Sequential(nn.Linear(dim, 128), nn.Tanh(), nn.Linear(128, 1))
        self.empty_user = nn.Parameter(torch.zeros(dim))                   # user vector for an empty history
        self.drop = nn.Dropout(dropout)
        self.position = nn.Parameter(torch.zeros(MAX_HIST, dim)) if use_pos else None   # slot 49 = latest click
        self.pop_weight = nn.Parameter(torch.zeros(4)) if use_pop else None             # one per history group
        self.content_weight = nn.Parameter(torch.ones(4)) if use_cscale else None       # one per history group

    def encode_news(self, table, rows):
        """Table rows (any shape) -> article vectors (same shape + dim)."""
        return self.news(self.drop(table[rows]))

    def encode_user(self, table, hist):
        """hist (B, MAX_HIST) table rows, 0 = padding -> user vectors (B, dim)."""
        pad = hist == 0
        empty = pad.all(dim=1)
        # a row that is all padding would give NaN in attention and softmax: open one (zero) position for it;
        # its result is replaced by empty_user below
        pad = pad.clone()
        pad[empty, -1] = False
        h = self.encode_news(table, hist)
        if self.position is not None:
            h = h + self.position
        h, _ = self.self_att(h, h, h, key_padding_mask=pad, need_weights=False)
        h = self.drop(h)
        weights = self.pool(h).squeeze(-1).masked_fill(pad, float("-inf"))
        weights = torch.softmax(weights, dim=1)
        user = (weights.unsqueeze(-1) * h).sum(dim=1)
        return torch.where(empty[:, None], self.empty_user.expand_as(user), user)

    def popularity_term(self, group, pop):
        """w[group] * pop, broadcast over the candidates of each row (0 when the variant has no popularity)."""
        if self.pop_weight is None:
            return 0.0
        w = self.pop_weight[group]
        return w[:, None] * pop if pop.dim() == 2 else w * pop

    def content_term(self, group, dots):
        """v[group] * u.c (u.c unchanged when the variant has no content weights)."""
        if self.content_weight is None:
            return dots
        v = self.content_weight[group]
        return v[:, None] * dots if dots.dim() == 2 else v * dots

    def forward(self, table, hist, cand, group=None, pop=None):
        """-> (B, 1 + neg) logits: dot product of the user vector with each candidate (+ popularity term)."""
        user = self.encode_user(table, hist)
        c = self.encode_news(table, cand)
        return self.content_term(group, (user[:, None, :] * c).sum(-1)) + self.popularity_term(group, pop)


@torch.no_grad()
def score(model, table, lookup, data, pop=None, chunk=4096):
    """Score of every candidate of data, in the flat order of data.candidates."""
    model.eval()
    out = np.empty(data.n_candidates, dtype=np.float32)
    for a in range(0, data.n_impressions, chunk):
        b = min(a + chunk, data.n_impressions)
        imp = np.arange(a, b)
        hist = torch.from_numpy(history_rows(data, lookup, imp)).to(DEVICE)
        user = model.encode_user(table, hist)
        c0, c1 = data.offsets[a], data.offsets[b]
        cand = model.encode_news(table, torch.from_numpy(lookup[data.candidates[c0:c1]]).to(DEVICE))
        sizes = torch.from_numpy(data.sizes[a:b]).to(DEVICE)
        s = (torch.repeat_interleave(user, sizes, dim=0) * cand).sum(-1)
        group = torch.repeat_interleave(torch.from_numpy(history_group(data, imp)).to(DEVICE), sizes)
        s = model.content_term(group, s)
        if model.pop_weight is not None:
            s = s + model.popularity_term(group, torch.from_numpy(pop[c0:c1]).to(DEVICE))
        out[c0:c1] = s.cpu().numpy()
    return out


def batch_loss(model, table, hist, cand, group, pop):
    logits = model(table, hist, cand, group, pop)
    return F.cross_entropy(logits, torch.zeros(len(logits), dtype=torch.long, device=DEVICE))


def new_model(table, cfg, seed=SEED):
    torch.manual_seed(seed)
    return UserEncoder(dim_in=table.shape[1], dropout=cfg["dropout"], use_pop=cfg["pop"],
                       use_pos=cfg["pos"], use_cscale=cfg["cscale"]).to(DEVICE)


def train(model, table, lookup, fit, pos, imp, val_sample, cfg, pops, folder=None, seed=SEED):
    """AdamW on the click softmax loss; after each epoch: metrics on the fixed val sample (and weights saved).
    The seed sets the shuffling and the negatives (the initial weights are set in new_model)."""
    rng = np.random.default_rng(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    history = []
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        order = rng.permutation(len(pos))
        n_batches = (len(order) + BATCH - 1) // BATCH
        total, start = torch.zeros((), device=DEVICE), time.time()
        for step, s in enumerate(range(0, len(order), BATCH), 1):
            b = order[s:s + BATCH]
            batch = make_batch(fit, lookup, pos[b], imp[b], rng, cfg["neg"], pops["fit"])
            loss = batch_loss(model, table, *batch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.detach() * len(b)
            if step % 1000 == 0 or step == n_batches:
                print(f"  epoch {epoch} step {step}/{n_batches}  mean loss {total.item() / min(s + BATCH, len(order)):.4f}"
                      f"  {time.time() - start:.0f}s", flush=True)
        res = me.evaluate(val_sample, score(model, table, lookup, val_sample, pops["val_sample"]),
                          save=False, verbose=False)
        row = {"epoch": epoch, "train_loss": total.item() / len(order), "seconds": round(time.time() - start),
               **{f"val_sample_{m}": res[m] for m in me.METRICS}}
        if model.pop_weight is not None:
            row["pop_weight"] = [round(w, 4) for w in model.pop_weight.detach().cpu().tolist()]
        if model.content_weight is not None:
            row["content_weight"] = [round(w, 4) for w in model.content_weight.detach().cpu().tolist()]
        history.append(row)
        print(f"epoch {epoch}: loss {row['train_loss']:.4f}, val sample AUC {res['AUC']:.4f} "
              f"nDCG@10 {res['nDCG@10']:.4f} ({row['seconds']}s)"
              + (f", popularity weight by group {row['pop_weight']}" if "pop_weight" in row else "")
              + (f", content weight by group {row['content_weight']}" if "content_weight" in row else ""), flush=True)
        if folder is not None:
            torch.save(model.state_dict(), folder / f"epoch{epoch}.pt")
    return history


# ---------------------------------------------------------------------------------------------------- runs
def run_check(variant):
    """Checks on small data; nothing is written."""
    cfg = VARIANTS[variant]
    print(f"device {DEVICE}, torch {torch.__version__}, variant {variant}: {cfg}")
    fit, val, val_sample, pops = load_splits(cfg)
    vectors, row_of = load_embeddings()
    table, lookup = article_table(vectors), row_of + 1
    neg = cfg["neg"]

    # article_table
    n = int(np.flatnonzero(row_of >= 0)[0])
    assert table.shape == (len(vectors) + 1, vectors.shape[1]) and table[0].abs().sum().item() == 0
    assert np.allclose(table[lookup[n]].cpu().numpy(), vectors[row_of[n]])
    # click_samples
    pos, imp = click_samples(fit)
    assert (fit.labels[pos] == 1).all()
    assert ((fit.offsets[imp] <= pos) & (pos < fit.offsets[imp + 1])).all()
    # make_batch
    rng = np.random.default_rng(SEED)
    sel = np.arange(0, len(pos), len(pos) // 64)[:64]
    hist, cand, group, pop = make_batch(fit, lookup, pos[sel], imp[sel], rng, neg, pops["fit"])
    assert hist.shape == (len(sel), MAX_HIST) and cand.shape == (len(sel), 1 + neg)
    for k, (p, i) in enumerate(zip(pos[sel], imp[sel])):
        h = fit.history_of(i)
        if len(h):
            assert hist[k, -1].item() == lookup[h[-1]]
        assert (hist[k] != 0).sum().item() == min(len(h), MAX_HIST)
        assert group[k].item() == list(npr.GROUPS).index(npr.history_group([len(h)])[0])   # same groups as npr
        assert cand[k, 0].item() == lookup[fit.candidates[p]]
        clicked_rows = set(lookup[fit.candidates[fit.offsets[i]:fit.offsets[i + 1]][fit.labels_of(i) == 1]].tolist())
        assert not clicked_rows & set(cand[k, 1:].tolist()), "a clicked article was drawn as a negative"
        if pop is not None:
            assert np.isclose(pop[k, 0].item(), pops["fit"][p])
    assert (cand != 0).all()
    print("article_table, click_samples, make_batch: all checks passed")

    # untrained model: loss about ln(1 + neg), val sample AUC about 0.5, no NaN for empty histories
    model = new_model(table, cfg)
    empty = torch.zeros((2, MAX_HIST), dtype=torch.long, device=DEVICE)
    two_pop = None if pop is None else pop[:2]
    assert torch.isfinite(model(table, empty, cand[:2], group[:2], two_pop)).all()
    with torch.no_grad():
        model.eval()
        print(f"untrained loss {batch_loss(model, table, hist, cand, group, pop).item():.4f} "
              f"(ln {1 + neg} = {np.log(1 + neg):.4f})")
    me.evaluate(val_sample, score(model, table, lookup, val_sample, pops["val_sample"]), name="untrained", save=False)

    # overfit test: the same 2,000 clicks with the same negatives every pass -> loss should fall far below ln(1+neg)
    model = new_model(table, cfg)
    model.drop.p = 0.0
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    small = np.arange(2000)
    for rep in range(60):
        r = np.random.default_rng(0)
        model.train()
        for s in range(0, len(small), BATCH):
            b = small[s:s + BATCH]
            loss = batch_loss(model, table, *make_batch(fit, lookup, pos[b], imp[b], r, neg, pops["fit"]))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        if rep % 10 == 9:
            print(f"  overfit pass {rep + 1}: last batch loss {loss.item():.4f}")
    print(f"check done: run 'val {variant}' if the overfit loss fell well below {np.log(1 + neg):.2f}")


def run_val(variant, seed=SEED):
    cfg = VARIANTS[variant]
    folder, name = out_dir(variant, seed), run_name(variant, seed)
    if (folder / "info.json").exists():
        sys.exit(f"{name} was already run ({folder / 'info.json'}); not overwritten")
    folder.mkdir(parents=True, exist_ok=True)
    start = time.time()
    print(f"variant {variant}, seed {seed}: {cfg}")
    fit, val, val_sample, pops = load_splits(cfg)
    vectors, row_of = load_embeddings()
    table, lookup = article_table(vectors), row_of + 1
    pos, imp = click_samples(fit)
    model = new_model(table, cfg, seed)
    history = train(model, table, lookup, fit, pos, imp, val_sample, cfg, pops, folder, seed)

    best = max(history, key=lambda r: r["val_sample_AUC"])["epoch"]
    model.load_state_dict(torch.load(folder / f"epoch{best}.pt", map_location=DEVICE))
    scores = score(model, table, lookup, val, pops["val"])
    np.save(folder / "scores_val.npy", scores)
    parts = ["NRMS-style user encoder on frozen MiniLM vectors", f"proj {DIM}, {HEADS} heads, last {MAX_HIST} clicks",
             f"{cfg['neg']} negatives, dropout {cfg['dropout']}"]
    if cfg["pop"]:
        parts.append("+ popularity term w[history group] * standardised log cum_ctr")
    if cfg["pos"]:
        parts.append("+ learned history-slot (recency) vectors")
    if cfg["cscale"]:
        parts.append("+ content weight v[history group] on u.c")
    days = "Nov 10-13" if cfg["no9"] else "Nov 9-13"
    note = "; ".join(parts) + f"; trained on {days}, epoch {best} of {cfg['epochs']} chosen on a {VAL_SAMPLE} val sample"
    if seed != SEED:
        note += f"; seed {seed}"
    result = me.evaluate(val, scores, name=name, notes=note)
    groups = group_means(val, scores)
    print(groups.round(4))
    others = ["lgbm_minilm_cumctr_mono", "hybrid_e_content_cumctr", "nrms_user_minilm"]
    if seed != SEED:
        others.append(run_name(variant))            # the same variant with seed 42
    for other in others:
        if other != name:
            npr.compare("val", name, other, "AUC").to_csv(folder / f"compare_vs_{other}_AUC.csv", index=False)
    info = {"variant": variant, "name": name, "notes": note, "settings": {**cfg, "MAX_HIST": MAX_HIST, "BATCH": BATCH,
            "LR": LR, "SEED": seed, "DIM": DIM, "HEADS": HEADS, "VAL_SAMPLE": VAL_SAMPLE,
            "VAL_SAMPLE_SEED": VAL_SAMPLE_SEED},
            "best_epoch": best, "history": history, "popularity_input": pops["stats"], "clicks": int(len(pos)),
            "torch": torch.__version__, "device": torch.cuda.get_device_name(0) if DEVICE == "cuda" else "cpu",
            "val": result, "val_groups": groups.reset_index().to_dict("records"),
            "minutes": round((time.time() - start) / 60, 1)}
    (folder / "info.json").write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
    print(f"done in {info['minutes']} min; M2 content alone 0.6319, base 0.6531, Hybrid E 0.6603, M6 0.6750 (val AUC)")


def run_switch(variant):
    """Empty history -> popularity (cum_ctr), every other impression -> the variant's saved val scores.

    The same rule as the group switch of doc 12 for group 0 (popularity was best there); no new setting is chosen.
    Metrics are computed within impressions, so mixing two score scales across impressions is harmless; every
    impression must get exactly the result of its source run, which is asserted."""
    name = f"nrms_switch_g0_cumctr__{variant}"
    folder = OUT / "switch"
    folder.mkdir(parents=True, exist_ok=True)
    train_all = me.load_impressions("train")
    mask = npr.val_mask(train_all)
    slot_val = np.repeat(mask, train_all.sizes)
    val = npr.subset(train_all, mask, "val")
    del train_all
    neural = np.load(out_dir(variant) / "scores_val.npy").astype(np.float64)
    popularity = np.load(FEATURES / "cum_ctr_train.npy", mmap_mode="r")[slot_val].astype(np.float64)
    assert len(neural) == len(popularity) == val.n_candidates and np.isfinite(popularity).all()
    empty = np.repeat(np.diff(val.history_offsets) == 0, val.sizes)
    switch = np.where(empty, popularity, neural)
    note = f"group 0 (empty history): cum_ctr; groups 1-3, 4-10, >10: {run_name(variant)} (rule of doc 12)"
    me.evaluate(val, switch, name=name, notes=note)

    mine = npr.per_impression("val", name).set_index("impression_id")
    for source, groups in (("single_cum_ctr", ["0"]), (run_name(variant), ["1-3", "4-10", ">10"])):
        other = npr.per_impression("val", source).set_index("impression_id")
        ids = mine.index[mine["group"].isin(groups)]
        for m in me.METRICS:
            assert np.allclose(mine.loc[ids, m], other.loc[ids, m], atol=1e-12), (source, m)
    print("every impression equals its source run")
    print(group_means(val, switch).round(4))
    for other in (run_name(variant), "lgbm_minilm_cumctr_mono", "group_switch_auc", "single_cum_ctr"):
        for metric in ("AUC", "nDCG@10"):
            npr.compare("val", name, other, metric).to_csv(
                folder / f"compare_vs_{other}_{metric.replace('@', '')}.csv", index=False)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    variant = sys.argv[2] if len(sys.argv) > 2 else ("base" if mode == "check" else "")
    if mode not in ("check", "val", "switch") or variant not in VARIANTS:
        sys.exit(__doc__)
    if mode == "val" and len(sys.argv) > 3:
        run_val(variant, int(sys.argv[3]))
    else:
        {"check": run_check, "val": run_val, "switch": run_switch}[mode](variant)
