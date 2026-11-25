'''
N2 of the news plan (thesis/news_documentation/05_plan_news_study.md): Hybrid E for news.

The movie study's Hybrid E mixes a personal score with popularity:

    hybrid = alpha * personal + (1 - alpha) * popularity,   both rescaled to 0..1 within each impression

For news:
    personal     content similarity: cosine between the mean MiniLM vector of the user's history and the
                 candidate (vectors from output/minilm_title_abstract_entities_all, already unit length);
                 0 for an empty history, so that group falls back to popularity, as for movies
    popularity   cum_ctr, the smoothed click rate over all earlier days (doc 03)
    alpha        chosen separately per history group (0 / 1-3 / 4-10 / >10) on the validation day, by AUC
                 (the primary metric, doc 06); a tie keeps the smaller alpha, as in the movie study

Validation only (Nov 14). The dev and test articles that are not in the train news file were embedded
later by news/embed_missing_news.py (doc 10); the train vectors are unchanged.

Command line (repository root):
    python news/hybrid_e.py features     content score for every train slot -> features/content_mean_train.npy
    python news/hybrid_e.py val          content alone, the alpha grid, Hybrid E, paired test vs cum_ctr
'''
import json
import sys
import time

import numpy as np
import pandas as pd

import mind_eval as me
import news_protocol as npr

# train vectors from SemanticContentModel.py plus dev/test articles added by embed_missing_news.py
EMB_DIR = me.ROOT / "output" / "minilm_title_abstract_entities_all"
FEATURES = me.ROOT / "output" / "news" / "features"
OUT = me.ROOT / "output" / "news" / "hybrid_e"
CHUNK = 20_000                                   # impressions per chunk (memory)
ALPHAS = [0.0, 0.02, 0.05, 0.08, 0.1, 0.12, 0.15, 0.2, 0.25, 0.3, 0.4,
          0.5, 0.6, 0.7, 0.8, 0.9, 1.0]          # the movie study's grid (cold_start_experiment.ALPHAS)
HYBRID_NAME = "hybrid_e_content_cumctr"
CONTENT_NAME = "single_content_mean"


# ---------------------------------------------------------------------------------------------------- content
def load_embeddings():
    """Embedding table indexed by the integer news id (row 0 for ids without a vector) and a 'known' mask."""
    vectors = np.load(EMB_DIR / "embeddings.npy")
    ids = np.array([int(s[1:]) for s in json.loads((EMB_DIR / "article_ids.json").read_text("utf-8"))])
    row_of = np.full(ids.max() + 1, -1, dtype=np.int64)
    row_of[ids] = np.arange(len(ids))
    return vectors, row_of


def content_scores(data, vectors, row_of):
    """Cosine of each candidate with the mean vector of its impression's history (0 if no history)."""
    scores = np.zeros(data.n_candidates, dtype=np.float32)
    hist_len = np.diff(data.history_offsets)
    start = time.time()
    for a in range(0, data.n_impressions, CHUNK):
        b = min(a + CHUNK, data.n_impressions)
        h0, h1 = data.history_offsets[a], data.history_offsets[b]
        c0, c1 = data.offsets[a], data.offsets[b]
        lens = hist_len[a:b]
        hist_rows = row_of[data.history[h0:h1]]
        assert (hist_rows >= 0).all(), "history article without a vector"
        # mean history vector per impression; impressions without history keep a zero vector
        sums = np.zeros((b - a, vectors.shape[1]), dtype=np.float32)
        has = lens > 0
        if has.any():
            starts = (data.history_offsets[a:b] - h0)[has]
            sums[has] = np.add.reduceat(vectors[hist_rows], starts, axis=0)
        means = sums / np.maximum(lens, 1)[:, None]
        norms = np.linalg.norm(means, axis=1, keepdims=True)
        means = np.divide(means, norms, out=np.zeros_like(means), where=norms > 0)
        cand_rows = row_of[data.candidates[c0:c1]]
        known = cand_rows >= 0
        imp = np.repeat(np.arange(b - a), data.sizes[a:b])
        part = np.zeros(c1 - c0, dtype=np.float32)
        part[known] = np.einsum("ij,ij->i", vectors[cand_rows[known]], means[imp[known]])
        scores[c0:c1] = part
        if (a // CHUNK) % 20 == 0:
            print(f"   {b:,}/{data.n_impressions:,} impressions, {time.time() - start:.0f}s", flush=True)
    return scores


def build_features():
    vectors, row_of = load_embeddings()
    train = me.load_impressions("train")
    print(f"train: {train.n_impressions:,} impressions, {train.n_candidates:,} slots")
    scores = content_scores(train, vectors, row_of)
    FEATURES.mkdir(parents=True, exist_ok=True)
    np.save(FEATURES / "content_mean_train.npy", scores)
    print(f"saved {(FEATURES / 'content_mean_train.npy').relative_to(me.ROOT)}")


# ---------------------------------------------------------------------------------------------------- hybrid
def rescale_within(data, values):
    """Min-max rescaling to 0..1 inside each impression (all equal -> 0), as cold_start_experiment.rescale."""
    starts = data.offsets[:-1]
    low = np.repeat(np.minimum.reduceat(values, starts), data.sizes)
    high = np.repeat(np.maximum.reduceat(values, starts), data.sizes)
    span = high - low
    return np.divide(values - low, span, out=np.zeros_like(values), where=span > 0)


def group_means(data, scores):
    frame = me.per_impression_metrics(data, scores)
    frame = frame[frame["AUC"].notna()]
    frame["group"] = npr.history_group(frame["history_len"])
    table = frame.groupby("group")[["AUC", "nDCG@10"]].mean().reindex(npr.GROUPS)
    table.loc["all"] = frame[["AUC", "nDCG@10"]].mean()
    return table


def run_val():
    train = me.load_impressions("train")
    mask = npr.val_mask(train)
    slot_mask = np.repeat(mask, train.sizes)
    val = npr.subset(train, mask, "val")
    del train
    content = np.load(FEATURES / "content_mean_train.npy", mmap_mode="r")[slot_mask].astype(np.float64)
    popularity = np.load(FEATURES / "cum_ctr_train.npy", mmap_mode="r")[slot_mask].astype(np.float64)
    assert len(content) == len(popularity) == val.n_candidates
    print(f"val: {val.n_impressions:,} impressions, {val.n_candidates:,} slots")

    # 1. content alone (empty histories score 0 everywhere -> random order there)
    me.evaluate(val, content, name=CONTENT_NAME,
                notes="cosine of candidate with mean MiniLM history vector; 0 for empty history")

    # 2. alpha grid per history group
    personal, prior = rescale_within(val, content), rescale_within(val, popularity)
    curve = []
    for alpha in ALPHAS:
        table = group_means(val, alpha * personal + (1 - alpha) * prior)
        for g, row in table.iterrows():
            curve.append({"alpha": alpha, "group": g, "AUC": row["AUC"], "nDCG@10": row["nDCG@10"]})
        print(f"alpha {alpha:<4}: " + "  ".join(f"{g} {table.loc[g, 'AUC']:.4f}" for g in table.index), flush=True)
    curve = pd.DataFrame(curve)
    OUT.mkdir(parents=True, exist_ok=True)
    curve.to_csv(OUT / "alpha_curve_val.csv", index=False, float_format="%.5f")

    chosen = {}
    for g in npr.GROUPS:
        part = curve[curve["group"] == g].reset_index(drop=True)
        best = part.loc[part["AUC"].idxmax()]          # idxmax keeps the first (smallest alpha) on a tie
        chosen[g] = 0.0 if g == "0" else float(best["alpha"])   # group 0: personal is all 0, alpha is moot
    print("chosen alpha per group:", chosen)
    (OUT / "chosen_alpha.json").write_text(json.dumps(chosen, indent=2), encoding="utf-8")

    # 3. Hybrid E with the chosen alpha per group
    slot_alpha = pd.Series(npr.history_group(np.diff(val.history_offsets))).map(chosen).to_numpy()
    slot_alpha = np.repeat(slot_alpha, val.sizes)
    me.evaluate(val, slot_alpha * personal + (1 - slot_alpha) * prior, name=HYBRID_NAME,
                notes="content (MiniLM mean history cosine) + cum_ctr, alpha per history group "
                      + json.dumps(chosen))

    # 4. paired tests against popularity
    for name in (HYBRID_NAME, CONTENT_NAME):
        for metric in ("AUC", "nDCG@10"):
            npr.compare("val", name, "single_cum_ctr", metric).to_csv(
                OUT / f"compare_{name}_vs_cum_ctr_{metric.replace('@', '')}.csv", index=False)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "features":
        build_features()
    elif mode == "val":
        run_val()
    else:
        sys.exit(__doc__)
