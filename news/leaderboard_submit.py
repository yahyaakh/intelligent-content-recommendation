'''
Leaderboard files for the hidden MIND test set (pre-registration §6; log: output/news/submissions/leaderboard_log.md).

    m6   LightGBM, 5 MiniLM features + monotone cum_ctr (doc 13), the model as trained on Nov 9-13
         (output/news/lgbm_newuser/minilm_cumctr_mono.txt; not refit)

The 5 MiniLM features of the 93 million test slots are computed once with lgbm_val.minilm_features (checked against
the training cache in doc 11) and kept on disk in output/news/features/minilm5_test.npy (memory). Scores are
predicted in chunks and written with mind_eval.write_submission, which checks the format.

Command line (repository root):
    python news/leaderboard_submit.py m6
'''
import json
import sys
import time

import lightgbm as lgb
import numpy as np

import mind_eval as me
import lgbm_val as lv
from hybrid_e import load_embeddings

FEATURES = me.ROOT / "output" / "news" / "features"
MODELS = {"m6": {"file": me.ROOT / "output" / "news" / "lgbm_newuser" / "minilm_cumctr_mono.txt",
                 "pop": ["cum_ctr"], "name": "m6_lgbm_minilm_cumctr_mono",
                 "notes": "M6 of the news pre-registration: LightGBM, 5 MiniLM + monotone cum_ctr, trained on Nov 9-13 "
                          "(not refit)"}}
CHUNK = 5_000_000


def minilm_test(test):
    path = FEATURES / "minilm5_test.npy"
    if path.exists():
        X = np.load(path, mmap_mode="r")
        assert X.shape == (test.n_candidates, 5)
        return X
    tmp = FEATURES / "minilm5_test.partial.npy"
    X = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float32, shape=(test.n_candidates, 5))
    vectors, row_of = load_embeddings()
    start = time.time()
    lv.minilm_features(test, vectors, row_of, lv.load_categories(), out=X)
    X.flush()
    del X
    tmp.replace(path)
    print(f"MiniLM features for test written in {time.time() - start:.0f}s: {path.relative_to(me.ROOT)}")
    return np.load(path, mmap_mode="r")


def build(key):
    spec = MODELS[key]
    booster = lgb.Booster(model_file=str(spec["file"]))
    assert booster.feature_name() == lv.MINILM_FEATURES + spec["pop"], booster.feature_name()
    test = me.load_impressions("test")
    X_minilm = minilm_test(test)
    pop = [np.load(FEATURES / f"{name}_test.npy", mmap_mode="r") for name in spec["pop"]]
    scores = np.empty(test.n_candidates)
    for a in range(0, test.n_candidates, CHUNK):
        b = min(a + CHUNK, test.n_candidates)
        X = np.hstack([np.asarray(X_minilm[a:b])] + [np.asarray(p[a:b], dtype=np.float32)[:, None] for p in pop])
        scores[a:b] = booster.predict(X)
    empty = np.repeat(np.diff(test.history_offsets) == 0, test.sizes)
    facts = {"slots": int(test.n_candidates), "score_min": float(scores.min()), "score_max": float(scores.max()),
             "share_slots_history_length_0": float((X_minilm[:, 4] == 0).mean()),
             "share_slots_empty_history_from_offsets": float(empty.mean())}
    print(json.dumps(facts, indent=2))
    me.write_submission(test, scores, name=spec["name"], notes=spec["notes"])


if __name__ == "__main__":
    key = sys.argv[1] if len(sys.argv) > 1 else ""
    if key not in MODELS:
        sys.exit(__doc__)
    build(key)
