'''
The best dev model of doc 04 (LightGBM lambdarank, 5 MiniLM + 10 popularity features) on the validation day.

The saved ranker output/news/lgbm_minilm_pop15.txt (news/SemanticContentModel.py) was trained on the train
impressions before Nov 14, so the validation day (Nov 14) was not used to fit it. The model is not retrained.

Features, in the ranker's column order:
    5 MiniLM features   the same definitions as SemanticContentModel.build_features, computed here in a vectorised
                        loop (the original runs one pandas frame per impression and cannot be imported, because the
                        script runs at import time):
                          top5_similarity         mean of the 5 highest cosines between the candidate and the
                                                  distinct history articles (fewer if the history is shorter)
                          max_similarity          the highest such cosine
                          avg_profile_similarity  cosine with the mean vector of the distinct history articles
                          category_preference     share of the distinct history articles in the candidate's category
                          history_length          number of distinct history articles
                        a candidate or history article without a vector gets 0 (none since doc 10)
    10 popularity       popularity_features.load_feature_matrix (doc 03)

`check` recomputes the features of a random sample of training impressions and compares them with the cached
training matrix output/lightgbm_full_features_entities_v1/X_train.npy, which the original code wrote.

Command line (repository root):
    python news/lgbm_val.py check
    python news/lgbm_val.py val
'''
import json
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

import mind_eval as me
import news_protocol as npr
import popularity_features as pf
from hybrid_e import load_embeddings

MODEL = me.ROOT / "output" / "news" / "lgbm_minilm_pop15.txt"
TRAIN_CACHE = me.ROOT / "output" / "lightgbm_full_features_entities_v1"
OUT = me.ROOT / "output" / "news" / "lgbm_val"
MINILM_FEATURES = ["top5_similarity", "max_similarity", "avg_profile_similarity", "category_preference",
                   "history_length"]
POP_FEATURES = ["shown_1h", "shown_6h", "shown_24h", "shown_all", "prev_shows", "prev_clicks",
                "prev_ctr", "cum_shows", "cum_clicks", "cum_ctr"]
RUN_NAME = "lgbm_minilm_pop15"
CHECK_N = 5_000
SEED = 42


def load_categories():
    """Category code per integer news id (-1 if unknown), from the train, dev and test news files."""
    news = pd.concat([me.load_news(s)[["category"]] for s in ("train", "dev", "test")])
    news = news[~news.index.duplicated(keep="first")]
    codes, _ = pd.factorize(news["category"])
    category_of = np.full(news.index.max() + 1, -1, dtype=np.int64)
    category_of[news.index.to_numpy()] = codes
    category_of[news.index.to_numpy()[news["category"].to_numpy() == ""]] = -1
    return category_of


def minilm_features(data, vectors, row_of, category_of, out=None):
    """out: an optional zero-filled (n_candidates, 5) float32 array to write into (e.g. a disk memmap)."""
    X = np.zeros((data.n_candidates, 5), dtype=np.float32) if out is None else out
    n_cat = category_of.max() + 1
    start = time.time()
    for i in range(data.n_impressions):
        c0, c1 = data.offsets[i], data.offsets[i + 1]
        history = np.unique(data.history[data.history_offsets[i]:data.history_offsets[i + 1]])
        block = X[c0:c1]
        block[:, 4] = len(history)
        if len(history) == 0:
            continue
        cands = data.candidates[c0:c1]
        h_rows = row_of[history]
        h_rows = h_rows[h_rows >= 0]
        c_rows = row_of[cands]
        known = c_rows >= 0
        if len(h_rows) and known.any():
            H, C = vectors[h_rows], vectors[c_rows[known]]
            sims = C @ H.T                                          # unit vectors -> cosine
            k = min(5, len(h_rows))
            top = np.partition(sims, sims.shape[1] - k, axis=1)[:, -k:]
            mean = H.mean(axis=0)
            mean /= np.linalg.norm(mean)
            block[known, 0] = top.mean(axis=1)
            block[known, 1] = sims.max(axis=1)
            block[known, 2] = C @ mean
        h_cat = category_of[history]
        h_cat = h_cat[h_cat >= 0]
        if len(h_cat):
            share = np.bincount(h_cat, minlength=n_cat) / len(h_cat)
            c_cat = category_of[cands]
            block[:, 3] = np.where(c_cat >= 0, share[np.maximum(c_cat, 0)], 0.0)
        if i % 100_000 == 0:
            print(f"   {i:,}/{data.n_impressions:,} impressions, {time.time() - start:.0f}s", flush=True)
    return X


# ---------------------------------------------------------------------------------------------------- check
def run_check():
    train = me.load_impressions("train")
    fit = npr.subset(train, npr.fit_mask(train), "fit")
    del train
    groups = np.load(TRAIN_CACHE / "groups.npy")
    assert np.array_equal(groups, fit.sizes), "training cache is not in the order of the Nov 9-13 impressions"
    cache = np.load(TRAIN_CACHE / "X_train.npy", mmap_mode="r")
    pick = np.zeros(fit.n_impressions, dtype=bool)
    pick[np.random.default_rng(SEED).choice(fit.n_impressions, CHECK_N, replace=False)] = True
    sample = npr.subset(fit, pick, "check")
    rows = np.repeat(pick, fit.sizes)
    expected = np.asarray(cache[rows])
    vectors, row_of = load_embeddings()
    got = minilm_features(sample, vectors, row_of, load_categories())
    diff = np.abs(got - expected).max(axis=0)
    report = {"impressions": CHECK_N, "candidates": int(rows.sum()),
              "max_abs_diff": dict(zip(MINILM_FEATURES, map(float, diff)))}
    print(json.dumps(report, indent=2))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "feature_check.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    assert diff.max() < 1e-4, "features differ from the training cache"


# ---------------------------------------------------------------------------------------------------- val
def run_val():
    booster = lgb.Booster(model_file=str(MODEL))
    assert booster.feature_name() == MINILM_FEATURES + POP_FEATURES, booster.feature_name()
    train = me.load_impressions("train")
    mask = npr.val_mask(train)
    val = npr.subset(train, mask, "val")
    del train
    print(f"val: {val.n_impressions:,} impressions, {val.n_candidates:,} slots")

    vectors, row_of = load_embeddings()
    X_minilm = minilm_features(val, vectors, row_of, load_categories())
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / "minilm5_val.npy", X_minilm)
    X = np.hstack([X_minilm, pf.load_feature_matrix("train", POP_FEATURES, impression_mask=mask)])
    assert X.shape == (val.n_candidates, 15)

    scores = booster.predict(X)
    me.evaluate(val, scores, name=RUN_NAME,
                notes="doc 04 ranker (trained Nov 9-13, not retrained), 5 MiniLM + 10 popularity features")

    gain = pd.Series(booster.feature_importance("gain"), index=booster.feature_name())
    (gain / gain.sum()).sort_values(ascending=False).rename("gain_share").to_csv(OUT / "feature_importance.csv")
    print((gain / gain.sum()).sort_values(ascending=False).round(4).to_string())

    for other in ("hybrid_e_content_cumctr", "single_cum_ctr"):
        for metric in ("AUC", "nDCG@10"):
            npr.compare("val", RUN_NAME, other, metric).to_csv(
                OUT / f"compare_{RUN_NAME}_vs_{other}_{metric.replace('@', '')}.csv", index=False)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "check":
        run_check()
    elif mode == "val":
        run_val()
    else:
        sys.exit(__doc__)
