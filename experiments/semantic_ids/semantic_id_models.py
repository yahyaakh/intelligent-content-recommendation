'''
Semantic IDs as the ITEM REPRESENTATION of a model (doc 28), the way the literature uses them.
Two models, each with a switchable item representation; everything else identical.

Model 1, "two-tower" (NCF family, the Singh et al. 2024 comparison):
  user vector = MLP(mean of the seed films' vectors); score of film i = user vector . vector(i);
  softmax over the 1,549 pool films.
  vector(i) =  atomic : one learned embedding per film id
               sid    : sum of the learned embeddings of its 3 semantic-ID tokens (one table per level;
                        films with the same 3 tokens get the same vector)
               both   : atomic + sid (Singh et al.'s recommended combination)
Model 2, "generative" (TIGER, Rajput et al. 2023, use 3 of doc 18):
  a GRU encoder reads the seeds in time order;
               atomic : the decoder predicts the next film id in one step (softmax over 1,549 films)
               sid    : input = the seeds' 4 ID tokens each (3 levels + collision token), and a GRU decoder
                        generates the target's 4 tokens one by one; recommendations by beam search
                        restricted to the IDs of pool films (TIGER's generative retrieval)

Semantic IDs: Semantic_IDs from recomm.py, enriched embedding, 32 codes, 3 levels + collision token.
Training data: the warm users' liked films (rating >= own mean) in the pool, oldest first. One example =
3-10 consecutive liked films (input) and one liked film after them (target), 10 examples per user and epoch.
New users are scored from their seeds alone (no fold-in), exactly like the other cold-start methods.

Protocol (fixed before running):
  tune   epochs per (model, representation): trained on warm half A, scored every 5 epochs on 2,000
         simulated users of half B (3 and 10 seeds averaged). Validation users never used.
  val    3 training runs per variant (torch seed r, k-means seed 42 + r), trained on all warm users with
         the chosen epochs, scored on the 203 validation users (3 and 10 seeds), NDCG@10 averaged over the
         runs per user; paired Wilcoxon vs the atomic version of the same model.
         Also: recall@10 on tail films only (the least-liked third of the pool), where semantic IDs are
         expected to help most.
Fixed settings: dimension 64 (two-tower) / 128 (GRU), Adam lr 1e-3, batch 512, beam 30.

Usage:
    python experiments/semantic_ids/semantic_id_models.py pilot    1 epoch each, 300 users: checks that everything runs
    python experiments/semantic_ids/semantic_id_models.py tune     choose epochs on the simulated users
    python experiments/semantic_ids/semantic_id_models.py val      validation run with the chosen epochs
Outputs: output/experiments/semantic_id_models/
'''

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))  # shared modules live in src/
import sys

MODE = sys.argv[1] if len(sys.argv) > 1 else "pilot"
sys.argv = [sys.argv[0]]
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import wilcoxon

import build_movie_texts as bmt
import cold_start_experiment as cs
import LightGBM_XBoost_rerank_experiment as ltr
from recomm import Semantic_IDs

OUT = Path("output") / "experiments" / "semantic_id_models"
EMBEDDING = "movies_enriched_embeddinggemma_v2"
CODES = 32
K = bmt.K
SEED_COUNTS = (10, 3)
VARIANTS = [("two-tower", "atomic"), ("two-tower", "sid"), ("two-tower", "both"),
            ("generative", "atomic"), ("generative", "sid")]
# Grid history: a first tune run with 30 epochs left both atomic variants still improving at epoch 30 (the
# edge), which would have favoured the semantic-ID variants (peaks at 15-25), so the grid was extended to 60.
MAX_EPOCHS, EVAL_EVERY, TUNE_USERS = 60, 5, 2000
EXAMPLES_PER_USER, BATCH, LR, BEAM = 10, 512, 1e-3, 30
PAD = -1


# ───────────────────────── data ─────────────────────────
def load():
    movies = bmt.load_movies()
    ratings = bmt.prepare_ratings(movies)
    train, cold = cs.split_users(ratings)
    pool_ids = sorted(set(int(m) for m in train["movie_id"]))
    cold_users = cs.build_cold_user_data(cold, set(pool_ids), max(cs.SEED_COUNTS), cs.MIN_TEST_RELEVANT)
    val_users, _ = cs.split_validation(cold_users, cs.VAL_FRACTION)
    movie_ids = movies["movie_id"].to_numpy()
    position_of = {m: i for i, m in enumerate(movie_ids)}
    data = {"movies": movies, "movie_ids": movie_ids, "position_of": position_of,
            "pool": sorted(position_of[m] for m in pool_ids)}
    return data, train, pool_ids, val_users


def liked_sequences(ratings, col):
    """Each user's liked pool films (rating >= own mean), oldest first, as pool column numbers."""
    r = ratings[ratings["rating"] >= ratings.groupby("user_id")["rating"].transform("mean")]
    r = r[r["movie_id"].isin(col)].sort_values(["user_id", "timestamp", "movie_id"])
    seqs = [g["movie_id"].map(col).to_numpy() for _, g in r.groupby("user_id")]
    return [s for s in seqs if len(s) >= 4]


def make_examples(seqs, rng):
    """Input = 3-10 consecutive liked films, target = one liked film after them."""
    inputs, targets = [], []
    for s in seqs:
        for _ in range(EXAMPLES_PER_USER):
            n = int(rng.integers(3, min(10, len(s) - 1) + 1))
            start = int(rng.integers(0, len(s) - n))
            inputs.append(np.pad(s[start:start + n], (0, 10 - n), constant_values=PAD))
            targets.append(s[int(rng.integers(start + n, len(s)))])
    return torch.tensor(np.array(inputs)), torch.tensor(np.array(targets))


def sid_codes(movies, pool_ids, kmeans_seed):
    """(pool films, 4) codes: 3 levels + collision token, from recomm.py's Semantic_IDs."""
    ids = Semantic_IDs(movies, n_levels=3, codebook_size=CODES, seed=kmeans_seed)
    ids.load_embeddings(name=EMBEDDING)
    ids.fit()
    return torch.tensor(np.array([ids.codes[ids.row_of_movie[m]] for m in pool_ids]), dtype=torch.long)


# ───────────────────────── model 1: two-tower ─────────────────────────
class TwoTower(nn.Module):
    def __init__(self, n_items, codes, rep, dim=64):
        super().__init__()
        self.rep, self.codes = rep, codes[:, :3]
        self.atomic = nn.Embedding(n_items, dim)
        self.levels = nn.ModuleList(nn.Embedding(CODES, dim) for _ in range(3))
        for e in [self.atomic, *self.levels]:
            nn.init.normal_(e.weight, std=0.05)
        self.user = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))

    def item_vectors(self):
        sid = sum(self.levels[l](self.codes[:, l]) for l in range(3))
        return {"atomic": self.atomic.weight, "sid": sid, "both": self.atomic.weight + sid}[self.rep]

    def forward(self, inputs):
        items = self.item_vectors()
        mask = (inputs != PAD).float().unsqueeze(-1)
        seeds = items[inputs.clamp(min=0)] * mask
        user = self.user(seeds.sum(1) / mask.sum(1))
        return user @ items.T                                           # (batch, pool films)

    def loss(self, inputs, targets):
        return nn.functional.cross_entropy(self.forward(inputs), targets)

    @torch.no_grad()
    def recommend(self, inputs, seen_masks, allowed=None):
        scores = self.forward(inputs).masked_fill(seen_masks, -np.inf)
        return torch.topk(scores, K, dim=1).indices


# ───────────────────────── model 2: generative ─────────────────────────
class Generative(nn.Module):
    def __init__(self, n_items, codes, rep, dim=128):
        super().__init__()
        self.rep, self.codes, self.n_items = rep, codes, n_items
        self.sizes = [CODES] * 3 + [int(codes[:, 3].max()) + 1]
        if rep == "atomic":
            self.embed = nn.Embedding(n_items, dim)
            self.out = nn.Linear(dim, n_items)
        else:
            self.tok = nn.ModuleList(nn.Embedding(s, dim) for s in self.sizes)
            self.start = nn.Parameter(torch.zeros(dim))
            self.cell = nn.GRUCell(dim, dim)
            self.heads = nn.ModuleList(nn.Linear(dim, s) for s in self.sizes)
            self._build_trie()
        self.encoder = nn.GRU(dim, dim, batch_first=True)

    def _build_trie(self):
        """allowed[l][prefix] = boolean mask of the next tokens that lead to a pool film."""
        self.allowed = [dict() for _ in range(4)]
        for row in self.codes.tolist():
            for l in range(4):
                prefix = tuple(row[:l])
                if prefix not in self.allowed[l]:
                    self.allowed[l][prefix] = torch.zeros(self.sizes[l], dtype=torch.bool)
                self.allowed[l][prefix][row[l]] = True
        self.item_of = {tuple(row): i for i, row in enumerate(self.codes.tolist())}

    def encode(self, inputs):
        valid = inputs != PAD
        lengths = valid.sum(1)
        if self.rep == "atomic":
            x = self.embed(inputs.clamp(min=0))
        else:                                                           # each seed -> its 4 tokens
            c = self.codes[inputs.clamp(min=0)]                         # (batch, 10, 4)
            x = torch.stack([self.tok[l](c[..., l]) for l in range(4)], dim=2).flatten(1, 2)   # (batch, 40, dim)
            lengths = lengths * 4
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.encoder(packed)
        return h[0]                                                     # (batch, dim)

    def loss(self, inputs, targets):
        h = self.encode(inputs)
        if self.rep == "atomic":
            return nn.functional.cross_entropy(self.out(h), targets)
        tgt = self.codes[targets]
        x, total = self.start.expand(len(h), -1), 0
        for l in range(4):
            h = self.cell(x, h)
            total = total + nn.functional.cross_entropy(self.heads[l](h), tgt[:, l])
            x = self.tok[l](tgt[:, l])
        return total

    @torch.no_grad()
    def recommend(self, inputs, seen_masks, allowed=None):
        h = self.encode(inputs)
        if self.rep == "atomic":
            return torch.topk(self.out(h).masked_fill(seen_masks, -np.inf), K, dim=1).indices
        results = []
        for u in range(len(h)):                                         # constrained beam search
            beams = [((), 0.0, h[u], self.start)]
            for l in range(4):
                states = torch.stack([b[2] for b in beams])
                xs = torch.stack([b[3] for b in beams])
                hs = self.cell(xs, states)
                logp = torch.log_softmax(self.heads[l](hs), dim=1)
                masks = torch.stack([self.allowed[l][b[0]] for b in beams])
                logp = logp.masked_fill(~masks, -np.inf)
                total = logp + torch.tensor([b[1] for b in beams]).unsqueeze(1)
                flat = total.flatten()
                width = BEAM if l < 3 else BEAM + 10
                top = torch.topk(flat, min(width, int(torch.isfinite(flat).sum()))).indices
                size = self.sizes[l]
                beams = [(beams[int(i) // size][0] + (int(i) % size,), float(flat[i]), hs[int(i) // size],
                          self.tok[l](torch.tensor(int(i) % size))) for i in top]
            items = [self.item_of[b[0]] for b in beams]
            items = [i for i in items if not seen_masks[u, i]][:K]
            results.append(items + [items[-1]] * (K - len(items)))      # (never shorter in practice)
        return torch.tensor(results)


# ───────────────────────── training and scoring ─────────────────────────
def build(model_name, rep, n_items, codes):
    return (TwoTower if model_name == "two-tower" else Generative)(n_items, codes, rep)


def train_epochs(model, seqs, epochs, seed, callback=None):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    for epoch in range(1, epochs + 1):
        model.train()
        inputs, targets = make_examples(seqs, rng)
        order = torch.randperm(len(inputs))
        for b in range(0, len(order), BATCH):
            idx = order[b:b + BATCH]
            loss = model.loss(inputs[idx], targets[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        if callback:
            callback(epoch)


def user_tensors(users, col, n_items):
    inputs = torch.tensor(np.array([np.pad([col[m] for m in u["liked"]], (0, 10 - len(u["liked"])),
                                           constant_values=PAD) for u in users]))
    seen = torch.zeros(len(users), n_items, dtype=torch.bool)
    for i, u in enumerate(users):
        seen[i, [col[m] for m in u["seen"]]] = True
    return inputs, seen


def score(model, users, col, pool_ids, tail):
    model.eval()
    inputs, seen = user_tensors(users, col, len(pool_ids))
    rows = []
    for b in range(0, len(users), 256):
        top = model.recommend(inputs[b:b + 256], seen[b:b + 256])
        for u, items in zip(users[b:b + 256], top.tolist()):
            rec = [pool_ids[i] for i in items]
            rel_tail = u["relevant"] & tail
            rows.append({"user_id": u["user_id"], **bmt.user_metrics(rec, set(u["relevant"]), K),
                         "tail_recall": (len(set(rec) & rel_tail) / len(rel_tail)) if rel_tail else np.nan,
                         "tail_share": np.mean([m in tail for m in rec])})
    return pd.DataFrame(rows)


def tail_films(train, pool_ids):
    counts = cs.popularity_scores(train).reindex(pool_ids).fillna(0)
    return set(counts.sort_values(kind="stable").index[:len(pool_ids) // 3].astype(int))


# ───────────────────────── modes ─────────────────────────
def run_tune(pilot=False):
    data, _, pool_ids, _ = load()
    col = {m: i for i, m in enumerate(pool_ids)}
    users = {}
    for n in SEED_COUNTS:
        train, in_A, users_by_half = ltr.warm_halves(data, n)
        users[n] = users_by_half["B"][:300 if pilot else TUNE_USERS]
    half_A = train[in_A]
    seqs = liked_sequences(half_A, col)
    tail = tail_films(half_A, pool_ids)
    codes = sid_codes(data["movies"], pool_ids, 42)
    print(f"training users (half A): {len(seqs):,}; tune users (half B): {len(users[10])}; "
          f"tail films: {len(tail)}")

    rows = []
    for model_name, rep in VARIANTS:
        model = build(model_name, rep, len(pool_ids), codes)
        start = time.time()
        def evaluate(epoch):
            if epoch % EVAL_EVERY and not pilot:
                return
            for n in SEED_COUNTS:
                s = score(model, users[n], col, pool_ids, tail)
                rows.append({"model": model_name, "rep": rep, "epoch": epoch, "seeds": n, "ndcg": s["ndcg"].mean(),
                             "tail_recall": s["tail_recall"].mean()})
            last = [r for r in rows if r["epoch"] == epoch and r["model"] == model_name and r["rep"] == rep]
            print(f"  {model_name:<10} {rep:<6} epoch {epoch:>2}: NDCG@10 "
                  + " | ".join(f"{r['seeds']} seeds {r['ndcg']:.4f}" for r in last)
                  + f"  ({time.time() - start:.0f} s)", flush=True)
        train_epochs(model, seqs, 1 if pilot else MAX_EPOCHS, seed=0, callback=evaluate)

    table = pd.DataFrame(rows)
    if pilot:
        print(table.round(4).to_string(index=False))
        return
    table.to_csv(OUT / "tune.csv", index=False)
    mean = table.groupby(["model", "rep", "epoch"])["ndcg"].mean().reset_index()
    chosen = {f"{m}|{r}": int(g.loc[g["ndcg"].idxmax(), "epoch"]) for (m, r), g in mean.groupby(["model", "rep"])}
    with open(OUT / "chosen_epochs.json", "w") as f:
        json.dump(chosen, f, indent=1)
    print("\nMean NDCG@10 over 3 and 10 seeds, by epoch (simulated users of half B):")
    print(mean.pivot_table(index=["model", "rep"], columns="epoch", values="ndcg").round(4).to_string())
    print(f"chosen epochs: {chosen}")


def run_val():
    with open(OUT / "chosen_epochs.json") as f:
        chosen = json.load(f)
    data, train, pool_ids, val_users = load()
    col = {m: i for i, m in enumerate(pool_ids)}
    seqs = liked_sequences(train, col)
    tail = tail_films(train, pool_ids)
    users = {n: cs.users_with_n_seeds(val_users, n) for n in SEED_COUNTS}
    print(f"training users (all warm): {len(seqs):,}; validation users: {len(val_users)}; tail films: {len(tail)}")

    # Each finished (run, model, representation) is saved at once, so a stopped run resumes where it stopped
    partial = OUT / ("val_metrics_partial" if MODE == "metrics" else "val_partial")
    partial.mkdir(exist_ok=True)
    for r in range(3):
        codes = None
        for model_name, rep in VARIANTS:
            path = partial / f"run{r}_{model_name}_{rep}.csv"
            if path.exists():
                print(f"run {r}: {model_name} {rep} already done (loaded)", flush=True)
                continue
            if codes is None:
                codes = sid_codes(data["movies"], pool_ids, 42 + r)
            start = time.time()
            model = build(model_name, rep, len(pool_ids), codes)
            train_epochs(model, seqs, chosen[f"{model_name}|{rep}"], seed=r)
            pd.concat([score(model, users[n], col, pool_ids, tail).assign(model=model_name, rep=rep, run=r, seeds=n)
                       for n in SEED_COUNTS]).to_csv(path, index=False)
            print(f"run {r}: {model_name} {rep} done ({time.time() - start:.0f} s)", flush=True)

    per_run = pd.concat(pd.read_csv(p) for p in sorted(partial.glob("run*.csv")))
    assert len(per_run.groupby(["run", "model", "rep"])) == 3 * len(VARIANTS), "some runs are missing"
    if MODE == "metrics":
        metrics_summary(per_run)
        return
    per_run.to_csv(OUT / "val_per_user_per_run.csv", index=False)
    per_user = per_run.groupby(["seeds", "model", "rep", "user_id"])[["ndcg", "tail_recall", "tail_share"]].mean()
    runs = per_run.groupby(["seeds", "model", "rep", "run"])["ndcg"].mean().unstack("run")
    summary = per_user.groupby(["seeds", "model", "rep"]).mean()
    summary["ndcg_run_min"], summary["ndcg_run_max"] = runs.min(axis=1), runs.max(axis=1)

    tests = []
    for (n, model_name), g in per_user.reset_index().groupby(["seeds", "model"]):
        table = g.pivot(index="user_id", columns="rep", values="ndcg")
        for rep in [c for c in table.columns if c != "atomic"]:
            diff = table[rep] - table["atomic"]
            tests.append({"seeds": n, "model": model_name, "comparison": f"{rep} vs atomic",
                          "difference": diff.mean(), "better": int((diff > 0).sum()), "worse": int((diff < 0).sum()),
                          "p_value": wilcoxon(diff).pvalue if diff.any() else 1.0})
    tests = pd.DataFrame(tests)
    summary.to_csv(OUT / "val_summary.csv")
    tests.to_csv(OUT / "val_tests.csv", index=False)

    pd.set_option("display.width", 250)
    print("\nValidation users (203), mean over 3 training runs:")
    print(summary.round(4).to_string())
    print("\nPaired Wilcoxon vs the atomic version of the same model (uncorrected):")
    print(tests.round(4).to_string(index=False))


def metrics_summary(per_run):
    """Precision@10 / recall@10 next to NDCG@10, after checking NDCG against the original validation run."""
    old = pd.read_csv(OUT / "val_per_user_per_run.csv")
    keys = ["user_id", "model", "rep", "run", "seeds"]
    merged = old.merge(per_run, on=keys, suffixes=("_old", ""))
    assert len(merged) == len(old) == len(per_run), "users differ from the original run"
    gap = (merged["ndcg_old"] - merged["ndcg"]).abs()
    print(f"NDCG check vs original run: max per-user difference {gap.max():.2e}, "
          f"mean NDCG {merged['ndcg_old'].mean():.6f} -> {merged['ndcg'].mean():.6f}")
    if gap.max() > 1e-9:
        print("WARNING: training did not reproduce exactly; the new P/R come from slightly different models")
    per_run = per_run.assign(hit=(per_run["precision"] > 0).astype(float))
    cols = ["ndcg", "precision", "recall", "hit", "tail_recall", "tail_share"]
    per_user = per_run.groupby(["seeds", "model", "rep", "user_id"])[cols].mean()
    summary = per_user.groupby(["seeds", "model", "rep"]).mean()
    summary.to_csv(OUT / "val_metrics_summary.csv")
    pd.set_option("display.width", 250)
    print(summary.round(4).to_string())


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    if MODE in ("pilot", "tune"):
        run_tune(pilot=(MODE == "pilot"))
    elif MODE in ("val", "metrics"):
        run_val()
    else:
        sys.exit(__doc__)
