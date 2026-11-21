'''
The protocol of the news study (thesis/news_documentation/06_protocol.md): one place for the validation
split, the history groups, the fixed LLM samples and the paired tests, so every news experiment uses them
the same way.

    Validation   train impressions of Nov 14 ("val"); models are trained on Nov 9-13
    Final test   the MIND dev set (Nov 15, "dev"), pre-registered, run once per method
    Groups       history length 0 / 1-3 / 4-10 / >10 clicks (real new users)
    LLM samples  fixed, stratified by history group: val_llm (500 per group), dev_llm (2,500 per group)

Use from a model script:

    import mind_eval as me, news_protocol as npr
    val = npr.load_val()                          # Impressions with split "val" (results go to results/val/)
    pop = pf.load_feature_matrix("train", names, impression_mask=npr.val_mask())   # val rows of a feature
    me.evaluate(val, scores, name="my_model")
    llm = npr.load_sample("val_llm")              # the 2,000 sampled val impressions, split "val_llm"

Command line (repository root):
    python news/news_protocol.py samples                 build and save the two LLM samples (once)
    python news/news_protocol.py groups dev              every saved run of a split, by history group
    python news/news_protocol.py compare dev A B [AUC]   paired test of run A against run B, overall and by group
    python news/news_protocol.py facts                   group sizes of val and dev
'''
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import mind_eval as me

VAL_START = np.datetime64("2019-11-14T00:00:00")
PROTOCOL = me.ROOT / "output" / "news" / "protocol"
GROUPS = ("0", "1-3", "4-10", ">10")
SAMPLE_SEED = 42
SAMPLE_SIZES = {"val_llm": ("val", 500), "dev_llm": ("dev", 2_500)}     # name -> (source split, per group)


# ---------------------------------------------------------------------------------------------------- groups
def history_group(lengths):
    """History length (clicks before the impression) -> group label."""
    lengths = np.asarray(lengths)
    return np.select([lengths == 0, lengths <= 3, lengths <= 10], GROUPS[:3], GROUPS[3])


# ---------------------------------------------------------------------------------------------------- subsets
def subset(data, keep, split):
    """A new Impressions with only the impressions where keep is True (file order kept), named `split`."""
    keep = np.asarray(keep, dtype=bool)
    assert len(keep) == data.n_impressions
    cand_keep = np.repeat(keep, data.sizes)
    hist_keep = np.repeat(keep, np.diff(data.history_offsets))
    offsets = np.concatenate([[0], np.cumsum(data.sizes[keep])])
    history_offsets = np.concatenate([[0], np.cumsum(np.diff(data.history_offsets)[keep])])
    return me.Impressions(split, data.ids[keep], data.users[keep], data.times[keep],
                          data.history[hist_keep], history_offsets, data.candidates[cand_keep], offsets,
                          None if data.labels is None else data.labels[cand_keep])


def val_mask(train=None):
    """Impression mask over the train split: True for Nov 14 (validation)."""
    train = me.load_impressions("train") if train is None else train
    return train.times >= VAL_START


def fit_mask(train=None):
    """Impression mask over the train split: True for Nov 9-13 (what models are trained on)."""
    return ~val_mask(train)


def load_val():
    train = me.load_impressions("train")
    return subset(train, val_mask(train), "val")


def load_split(split):
    """'val' or any mind_eval split."""
    return load_val() if split == "val" else me.load_impressions(split)


# ---------------------------------------------------------------------------------------------------- LLM samples
def _evaluable(data):
    """Impressions with at least one click and one non-click (others have no AUC and are left out)."""
    clicks = np.add.reduceat(data.labels.astype(np.int64), data.offsets[:-1])
    return (clicks > 0) & (clicks < data.sizes)


def build_samples():
    """Draw the fixed LLM samples once: per history group the same number of evaluable impressions."""
    PROTOCOL.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SAMPLE_SEED)
    for name, (source, per_group) in SAMPLE_SIZES.items():
        path = PROTOCOL / f"sample_{name}.csv"
        if path.exists():
            print(f"{path.relative_to(me.ROOT)} exists; delete it by hand to draw again")
            continue
        data = load_split(source)
        group = history_group(np.diff(data.history_offsets))
        ok = _evaluable(data)
        rows = []
        for g in GROUPS:
            pool = np.flatnonzero(ok & (group == g))
            chosen = np.sort(rng.choice(pool, per_group, replace=False))
            rows.append(pd.DataFrame({"impression_id": data.ids[chosen], "group": g,
                                      "n_candidates": data.sizes[chosen],
                                      "history_len": np.diff(data.history_offsets)[chosen]}))
        sample = pd.concat(rows).sort_values("impression_id")
        sample.to_csv(path, index=False)
        print(f"{name}: {len(sample):,} impressions from {source} -> {path.relative_to(me.ROOT)}")
        print(sample.groupby("group", sort=False)["n_candidates"].describe()[["mean", "50%", "max"]]
              .reindex(GROUPS).to_string(float_format="%.1f"))


def load_sample(name):
    """The sampled impressions as an Impressions object with split = name (results go to results/<name>/)."""
    source = SAMPLE_SIZES[name][0]
    ids = pd.read_csv(PROTOCOL / f"sample_{name}.csv")["impression_id"].to_numpy()
    data = load_split(source)
    keep = np.isin(data.ids, ids)
    assert keep.sum() == len(ids)
    return subset(data, keep, name)


def group_shares(split):
    """Share of evaluable impressions per history group in a full split: the weights that turn per-group
    means of a stratified sample back into an estimate for the whole split."""
    data = load_split(split)
    group = history_group(np.diff(data.history_offsets))[_evaluable(data)]
    return pd.Series(group).value_counts(normalize=True).reindex(GROUPS)


# ---------------------------------------------------------------------------------------------------- reports
def per_impression(split, name):
    path = me.RESULTS / split / name / "per_impression.csv.gz"
    frame = pd.read_csv(path)
    frame["group"] = history_group(frame["history_len"])
    return frame


def by_group(split, name, metric="AUC"):
    """Mean of one metric per history group, plus all impressions and the group-weighted mean for samples."""
    frame = per_impression(split, name)
    row = frame.groupby("group")[metric].mean().reindex(GROUPS)
    row["all"] = frame[metric].mean()
    return row


def groups_table(split, metric="AUC"):
    runs = sorted(p.parent.name for p in (me.RESULTS / split).glob("*/per_impression.csv.gz"))
    table = pd.DataFrame({name: by_group(split, name, metric) for name in runs}).T
    counts = per_impression(split, runs[0])["group"].value_counts().reindex(GROUPS)
    table = table.sort_values("all")
    path = PROTOCOL / f"groups_{split}_{metric.replace('@', '')}.csv"
    PROTOCOL.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, float_format="%.4f")
    print(f"{metric} by history group on {split} (impressions per group: "
          + ", ".join(f"{g}: {counts[g]:,}" for g in GROUPS) + ")")
    print(table.to_string(float_format="%.4f"))
    print(f"saved {path.relative_to(me.ROOT)}")
    return table


def compare(split, a, b, metric="AUC"):
    """Paired comparison of run a against run b on the same impressions: mean difference, 95% interval
    (normal approximation, n is large) and a two-sided Wilcoxon signed-rank test, overall and per group."""
    fa, fb = per_impression(split, a), per_impression(split, b)
    both = fa[["impression_id", "group", metric]].merge(fb[["impression_id", metric]], on="impression_id",
                                                         suffixes=("_a", "_b"))
    assert len(both) == len(fa) == len(fb), "the two runs were not evaluated on the same impressions"
    rows = []
    for g in (*GROUPS, "all"):
        part = both if g == "all" else both[both["group"] == g]
        d = (part[f"{metric}_a"] - part[f"{metric}_b"]).to_numpy()
        se = d.std(ddof=1) / np.sqrt(len(d))
        p = wilcoxon(d[d != 0]).pvalue if (d != 0).sum() > 0 else 1.0
        rows.append({"group": g, "n": len(d), f"{a}": part[f"{metric}_a"].mean(), f"{b}": part[f"{metric}_b"].mean(),
                     "diff": d.mean(), "ci_low": d.mean() - 1.96 * se, "ci_high": d.mean() + 1.96 * se,
                     "a_better": (d > 0).mean(), "b_better": (d < 0).mean(), "p_wilcoxon": p})
    table = pd.DataFrame(rows)
    print(f"{metric} on {split}: {a} minus {b}")
    print(table.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    return table


def facts():
    for split in ("val", "dev"):
        data = load_split(split)
        group = history_group(np.diff(data.history_offsets))
        ok = _evaluable(data)
        counts = pd.Series(group).value_counts().reindex(GROUPS)
        print(f"{split}: {data.n_impressions:,} impressions ({int(ok.sum()):,} evaluable), "
              f"{len(np.unique(data.users)):,} users, {data.n_candidates:,} candidates")
        for g in GROUPS:
            m = group == g
            print(f"   history {g:>4}: {counts[g]:>7,} impressions ({counts[g] / len(group):.1%}), "
                  f"candidates median {np.median(data.sizes[m]):.0f}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "samples":
        build_samples()
    elif mode == "groups":
        groups_table(sys.argv[2] if len(sys.argv) > 2 else "dev", *(sys.argv[3:4]))
    elif mode == "compare" and len(sys.argv) >= 5:
        compare(sys.argv[2], sys.argv[3], sys.argv[4], *(sys.argv[5:6]))
    elif mode == "facts":
        facts()
    else:
        sys.exit(__doc__)
