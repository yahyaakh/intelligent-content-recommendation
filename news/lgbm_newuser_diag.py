'''
Diagnostic for doc 13: can a learned ranker on the 10 popularity features beat cum_ctr alone for an empty history?

In news/lgbm_newuser.py, giving empty-history impressions more weight made the ranker WORSE for them on val. Two
popularity-only rankers (same LightGBM settings) separate "too few examples" from "the learned popularity mix
does not carry over from Nov 9-13 to Nov 14":

    pop_g0         trained on the empty-history impressions of Nov 9-13 only
    pop_all        trained on all impressions of Nov 9-13

Both are scored on val; the group-0 result is compared with single_cum_ctr. The script also prints how the
empty-history training impressions spread over the five training days.

Two follow-ups, after pop_all turned out worse than cum_ctr in every group on val (but better on its training days):

    pop_all_to13   the same, trained on Nov 9-12 and scored on Nov 13 (not saved as a val run): is the drop on the
                   next day general, or specific to Nov 14?
    minilm_cumctr  the doc 04 ranker with only the stable popularity feature: 5 MiniLM features + cum_ctr, trained on
                   Nov 9-13, scored on val (the candidate new-user ranker)
    minilm_cumctr_mono  the same with a monotone constraint: the score may only rise with cum_ctr. For an empty
                   history the MiniLM features are all 0, so the score is then a rising function of cum_ctr alone

Command line (repository root):
    python news/lgbm_newuser_diag.py <variant>
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
import lgbm_val as lv
from lgbm_newuser import PARAMS, FEATURES, OUT

VARIANTS = ("pop_g0", "pop_all", "pop_all_to13", "minilm_cumctr", "minilm_cumctr_mono")


class Rows(lgb.Sequence):
    """Feature rows read in batches: optional MiniLM columns from the training cache, then popularity columns."""
    batch_size = 1_000_000

    def __init__(self, slot_index, pop_names, cache_rows=None):
        self.pop = [np.load(FEATURES / f"{name}_train.npy", mmap_mode="r") for name in pop_names]
        self.minilm = None if cache_rows is None else np.load(lv.TRAIN_CACHE / "X_train.npy", mmap_mode="r")
        self.slot_index, self.cache_rows = slot_index, cache_rows

    def __len__(self):
        return len(self.slot_index)

    def _rows(self, key):
        slots = np.atleast_1d(self.slot_index[key])
        cols = [np.column_stack([p[slots] for p in self.pop])]
        if self.minilm is not None:
            cols.insert(0, np.asarray(self.minilm[np.atleast_1d(self.cache_rows[key])]))
        return np.hstack(cols).astype(np.float32).astype(np.float64)

    def __getitem__(self, key):
        if isinstance(key, (int, np.integer)):
            return self._rows(key)[0]
        return self._rows(key)


def run(variant):
    train = me.load_impressions("train")
    fit = npr.fit_mask(train)
    lengths = np.diff(train.history_offsets)
    empty = lengths == 0
    day = train.times.astype("datetime64[D]")
    days = pd.Series(day[fit & empty]).value_counts().sort_index()
    info = {"variant": variant, "params": PARAMS,
            "empty_history_fit_impressions_per_day": {str(k)[:10]: int(v) for k, v in days.items()},
            "empty_history_share_of_fit": float(empty[fit].mean())}
    with_minilm = variant.startswith("minilm_cumctr")
    pop_names = ["cum_ctr"] if with_minilm else lv.POP_FEATURES
    names = (lv.MINILM_FEATURES if with_minilm else []) + pop_names
    params = dict(PARAMS)
    if variant == "minilm_cumctr_mono":
        params["monotone_constraints"] = [0] * len(lv.MINILM_FEATURES) + [1]
    keep = {"pop_g0": fit & empty, "pop_all": fit, "minilm_cumctr": fit, "minilm_cumctr_mono": fit,
            "pop_all_to13": day < np.datetime64("2019-11-13")}[variant]
    sizes = train.sizes[keep]
    slot_keep = np.repeat(keep, train.sizes)
    slot_index = np.flatnonzero(slot_keep)
    labels = train.labels[slot_keep].astype(np.int32)
    cache_rows = np.arange(len(slot_index)) if with_minilm else None   # keep == fit: cache order
    test_mask = day == np.datetime64("2019-11-13") if variant == "pop_all_to13" else npr.val_mask(train)
    test = npr.subset(train, test_mask, "nov13" if variant == "pop_all_to13" else "val")
    del train
    info.update({"params": params, "features": names, "impressions": int(len(sizes)), "rows": int(len(labels))})
    print(json.dumps(info, indent=2), flush=True)

    start = time.time()
    data = lgb.Dataset(Rows(slot_index, pop_names, cache_rows), label=labels, group=sizes, feature_name=names,
                       free_raw_data=True)
    booster = lgb.train(params, data)
    info["train_seconds"] = round(time.time() - start)
    OUT.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(OUT / f"{variant}.txt"))
    del data, slot_index, labels
    gain = pd.Series(booster.feature_importance("gain"), index=booster.feature_name())
    info["gain_share"] = (gain / gain.sum()).sort_values(ascending=False).round(4).to_dict()
    print(json.dumps(info["gain_share"], indent=2))

    X = pf.load_feature_matrix("train", pop_names, impression_mask=test_mask)
    if with_minilm:
        X = np.hstack([np.load(lv.OUT / "minilm5_val.npy"), X])
    scores = booster.predict(X)

    if variant == "pop_all_to13":           # next-day check on Nov 13, by history group, not saved as a run
        rows = []
        cum = np.nan_to_num(X[:, lv.POP_FEATURES.index("cum_ctr")].astype(np.float64), nan=0.0)
        for label, s in (("cum_ctr", cum), ("lgbm_pop_all_to13", scores)):
            frame = me.per_impression_metrics(test, s)
            frame = frame[frame["AUC"].notna()]
            frame["group"] = npr.history_group(frame["history_len"])
            row = frame.groupby("group")["AUC"].mean().reindex(npr.GROUPS)
            row["all"] = frame["AUC"].mean()
            rows.append(row.rename(label))
        table = pd.DataFrame(rows)
        print("AUC on Nov 13 (trained on Nov 9-12):")
        print(table.to_string(float_format=lambda x: f"{x:.4f}"))
        table.to_csv(OUT / "pop_all_to13_auc_nov13.csv", float_format="%.5f")
        info["auc_nov13"] = table.round(5).to_dict()
        (OUT / f"{variant}_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
        return

    name = f"lgbm_{variant}"
    note = {"pop_g0": "10 popularity features only, trained on Nov 9-13 empty histories only",
            "pop_all": "10 popularity features only, trained on Nov 9-13",
            "minilm_cumctr": "5 MiniLM features + cum_ctr, trained on Nov 9-13",
            "minilm_cumctr_mono": "5 MiniLM features + cum_ctr (monotone rising), trained on Nov 9-13"}[variant]
    me.evaluate(test, scores, name=name, notes=note)
    (OUT / f"{variant}_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    others = ["single_cum_ctr"] + (["lgbm_pop15_base", "group_switch_auc"] if with_minilm else [])         + (["lgbm_minilm_cumctr"] if variant == "minilm_cumctr_mono" else [])
    for other in others:
        for metric in ("AUC", "nDCG@10"):
            npr.compare("val", name, other, metric).to_csv(
                OUT / f"compare_{name}_vs_{other}_{metric.replace('@', '')}.csv", index=False)


if __name__ == "__main__":
    v = sys.argv[1] if len(sys.argv) > 1 else ""
    if v not in VARIANTS:
        sys.exit(__doc__)
    run(v)
