'''
A LightGBM ranker that does not lose to popularity for new users (doc 13).

Doc 11: the ranker lgbm_minilm_pop15 is the best model overall on val, but for an empty history it is 0.020 AUC
below popularity. Only 2% of the training impressions have an empty history, so the ranker learns little about
that case. Here the same ranker (same 15 features, same settings) is retrained on Nov 9-13 with weights per
training impression that give short histories more say:

    base      no weights: the doc 04 ranker retrained, to check that this pipeline reproduces the saved model
    w_sqrt    impression weight proportional to 1 / sqrt(share of its history group in the training impressions)
    w_equal   impression weight proportional to 1 / share: every history group has the same total weight

Weights are scaled to mean 1 over the training impressions; every candidate of an impression gets its weight.
LightGBM's lambdarank multiplies each row's gradient by its weight.

Memory: the 66 million training rows x 15 features do not fit next to the rest, so LightGBM reads them in batches
(lgb.Sequence) from the cached MiniLM features (output/lightgbm_full_features_entities_v1/X_train.npy) and the
popularity feature files (output/news/features/<name>_train.npy).

Command line (repository root):
    python news/lgbm_newuser.py <variant>      train on Nov 9-13, save the model, score val, paired tests
'''
import json
import sys
import time

import lightgbm as lgb
import numpy as np

import mind_eval as me
import news_protocol as npr
import popularity_features as pf
import lgbm_val as lv

OUT = me.ROOT / "output" / "news" / "lgbm_newuser"
FEATURES = me.ROOT / "output" / "news" / "features"
# the saved ranker's settings (parameters section of output/news/lgbm_minilm_pop15.txt); the rest are defaults
PARAMS = {"objective": "lambdarank", "metric": "ndcg", "num_iterations": 100, "learning_rate": 0.05,
          "num_leaves": 15, "num_threads": 8, "seed": 42, "verbosity": -1}
VARIANTS = {"base": None, "w_sqrt": 0.5, "w_equal": 1.0}       # weight = share ** -power


class FitRows(lgb.Sequence):
    """The 15 feature columns of the Nov 9-13 candidate slots, read in batches."""
    batch_size = 1_000_000

    def __init__(self, slot_index):
        self.minilm = np.load(lv.TRAIN_CACHE / "X_train.npy", mmap_mode="r")
        self.pop = [np.load(FEATURES / f"{name}_train.npy", mmap_mode="r") for name in lv.POP_FEATURES]
        self.slot_index = slot_index
        assert len(self.minilm) == len(slot_index)

    def __len__(self):
        return len(self.slot_index)

    def __getitem__(self, key):
        if isinstance(key, (int, np.integer)):
            j = self.slot_index[key]
            return np.array([*self.minilm[key], *(p[j] for p in self.pop)], dtype=np.float32).astype(np.float64)
        rows = self.slot_index[key]
        return np.hstack([np.asarray(self.minilm[key], dtype=np.float32),
                          np.column_stack([p[rows] for p in self.pop]).astype(np.float32)]).astype(np.float64)


def impression_weights(lengths, power):
    group = npr.history_group(lengths)
    share = {g: float((group == g).mean()) for g in npr.GROUPS}
    w = np.array([share[g] ** -power for g in group])
    return w / w.mean(), share


def train(variant):
    power = VARIANTS[variant]
    train_imp = me.load_impressions("train")
    fit_mask = npr.fit_mask(train_imp)
    sizes = train_imp.sizes[fit_mask]
    lengths = np.diff(train_imp.history_offsets)[fit_mask]
    slot_index = np.flatnonzero(np.repeat(fit_mask, train_imp.sizes))
    del train_imp
    assert np.array_equal(np.load(lv.TRAIN_CACHE / "groups.npy"), sizes)
    labels = np.load(lv.TRAIN_CACHE / "y_train.npy")

    info = {"variant": variant, "params": PARAMS, "impressions": int(len(sizes)), "rows": int(len(labels))}
    weight = None
    if power is not None:
        w_imp, share = impression_weights(lengths, power)
        weight = np.repeat(w_imp, sizes).astype(np.float32)
        group = npr.history_group(lengths)
        info.update({"power": power, "group_share": share,
                     "weight_per_impression": {g: float(w_imp[group == g][0]) for g in npr.GROUPS},
                     "weight_share": {g: float(w_imp[group == g].sum() / w_imp.sum()) for g in npr.GROUPS}})
    print(json.dumps(info, indent=2), flush=True)

    start = time.time()
    data = lgb.Dataset(FitRows(slot_index), label=labels, group=sizes, weight=weight,
                       feature_name=lv.MINILM_FEATURES + lv.POP_FEATURES, free_raw_data=True)
    booster = lgb.train(PARAMS, data)
    info["train_seconds"] = round(time.time() - start)
    print(f"trained in {info['train_seconds']}s", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(OUT / f"{variant}.txt"))
    del data, labels, weight, slot_index
    return booster, info


def score_val(booster, variant, info):
    train_imp = me.load_impressions("train")
    mask = npr.val_mask(train_imp)
    val = npr.subset(train_imp, mask, "val")
    del train_imp
    X = np.hstack([np.load(lv.OUT / "minilm5_val.npy"),
                   pf.load_feature_matrix("train", lv.POP_FEATURES, impression_mask=mask)])
    scores = booster.predict(X)
    if variant == "base":
        saved = lgb.Booster(model_file=str(lv.MODEL)).predict(X)
        info["max_abs_diff_vs_saved_model"] = float(np.abs(scores - saved).max())
        print(f"base vs saved model, largest score difference on val: {info['max_abs_diff_vs_saved_model']:.3g}")
    name = f"lgbm_pop15_{variant}"
    me.evaluate(val, scores, name=name,
                notes=f"doc 04 features, retrained on Nov 9-13, impression weights: {variant}")
    (OUT / f"{variant}_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    for other in ("lgbm_minilm_pop15", "group_switch_auc", "single_cum_ctr"):
        for metric in ("AUC", "nDCG@10"):
            npr.compare("val", name, other, metric).to_csv(
                OUT / f"compare_{name}_vs_{other}_{metric.replace('@', '')}.csv", index=False)


if __name__ == "__main__":
    v = sys.argv[1] if len(sys.argv) > 1 else ""
    if v not in VARIANTS:
        sys.exit(__doc__)
    booster, info = train(v)
    score_val(booster, v, info)
