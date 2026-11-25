'''
Per-group switch: for each history group, use the method with the best validation AUC.

Candidates, simplest first (a tie keeps the simpler method):
    single_cum_ctr            popularity, cum_ctr (doc 03)
    hybrid_e_content_cumctr   Hybrid E, MiniLM content + cum_ctr, alpha per group (doc 09)
    lgbm_minilm_pop15         LightGBM, 5 MiniLM + 10 popularity features (docs 04, 11)

The choice is made from the per-impression results already saved for val (output/news/results/val/<run>/).
The switch is then evaluated as its own run: each impression is ranked by the scores of its group's method.
Metrics are computed within impressions, so mixing score scales across impressions is harmless, and the
random tie-break of mind_eval depends only on the slot, so every impression must get exactly the result of its
source run. The script asserts this.

The switch is chosen on val, so its val result is in-sample (an upper bound for val). The honest test is the
pre-registered final test on dev.

Command line (repository root):
    python news/group_switch.py val
'''
import json
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

import mind_eval as me
import news_protocol as npr
import popularity_features as pf
import hybrid_e as he
import lgbm_val as lv

CANDIDATES = ["single_cum_ctr", "hybrid_e_content_cumctr", "lgbm_minilm_pop15"]   # simplest first
RUN_NAME = "group_switch_auc"
OUT = me.ROOT / "output" / "news" / "group_switch"
METRIC = "AUC"


def choose(split="val"):
    table = pd.DataFrame({name: npr.by_group(split, name, METRIC) for name in CANDIDATES}).loc[list(npr.GROUPS)]
    chosen = {}
    for g in npr.GROUPS:
        row = table.loc[g, CANDIDATES].to_numpy()
        chosen[g] = CANDIDATES[int(np.argmax(row))]       # argmax keeps the first (simplest) on a tie
    return table, chosen


def val_scores():
    """Scores of the three candidates for every val slot, rebuilt from the saved features and models."""
    train = me.load_impressions("train")
    mask = npr.val_mask(train)
    slot_mask = np.repeat(mask, train.sizes)
    val = npr.subset(train, mask, "val")
    del train
    popularity = np.load(he.FEATURES / "cum_ctr_train.npy", mmap_mode="r")[slot_mask].astype(np.float64)
    content = np.load(he.FEATURES / "content_mean_train.npy", mmap_mode="r")[slot_mask].astype(np.float64)

    alpha = json.loads((he.OUT / "chosen_alpha.json").read_text("utf-8"))
    slot_alpha = pd.Series(npr.history_group(np.diff(val.history_offsets))).map(alpha).to_numpy()
    slot_alpha = np.repeat(slot_alpha, val.sizes)
    hybrid = slot_alpha * he.rescale_within(val, content) + (1 - slot_alpha) * he.rescale_within(val, popularity)

    booster = lgb.Booster(model_file=str(lv.MODEL))
    X = np.hstack([np.load(lv.OUT / "minilm5_val.npy"),
                   pf.load_feature_matrix("train", lv.POP_FEATURES, impression_mask=mask)])
    lgbm = booster.predict(X)
    return val, {"single_cum_ctr": popularity, "hybrid_e_content_cumctr": hybrid, "lgbm_minilm_pop15": lgbm}


def run_val():
    table, chosen = choose("val")
    print(f"{METRIC} by group on val:\n{table.to_string(float_format=lambda x: f'{x:.4f}')}")
    print("chosen:", chosen)
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / "candidates_by_group_val.csv", float_format="%.5f")
    (OUT / "chosen_method.json").write_text(json.dumps(chosen, indent=2), encoding="utf-8")

    val, scores = val_scores()
    group = npr.history_group(np.diff(val.history_offsets))
    slot_group = np.repeat(group, val.sizes)
    switch = np.empty(val.n_candidates)
    for g, name in chosen.items():
        switch[slot_group == g] = scores[name][slot_group == g]
    me.evaluate(val, switch, name=RUN_NAME, notes="per history group the best val-AUC method: " + json.dumps(chosen))

    # every impression must get exactly the result of its source run
    mine = npr.per_impression("val", RUN_NAME).set_index("impression_id")
    for g, name in chosen.items():
        source = npr.per_impression("val", name).set_index("impression_id")
        ids = mine.index[mine["group"] == g]
        for m in me.METRICS:
            assert np.allclose(mine.loc[ids, m], source.loc[ids, m], atol=1e-12), (g, name, m)
    print("check passed: each impression equals its source run")

    for other in ("lgbm_minilm_pop15", "hybrid_e_content_cumctr", "single_cum_ctr"):
        for metric in ("AUC", "nDCG@10"):
            npr.compare("val", RUN_NAME, other, metric).to_csv(
                OUT / f"compare_{RUN_NAME}_vs_{other}_{metric.replace('@', '')}.csv", index=False)


if __name__ == "__main__":
    if (sys.argv[1] if len(sys.argv) > 1 else "") == "val":
        run_val()
    else:
        sys.exit(__doc__)
