'''
Cheap ensembles on validation (Nov 14), from saved scores; no training (doc 15; on cat + fresh: doc 16).

    A  seed average: the 3 runs (seeds 42, 1, 2) of the base neural variant:
         cscale     nrms_user_pop_pos_neg8_cscale (doc 14 seed check; doc 15)
         cat_fresh  nrms_user_pop_pos_neg8_cscale_cat_fresh (doc 16 seed check)
       Each run's scores are standardised within every impression (mean 0, sd 1), then averaged.
    B  blend: 0.5 * A + 0.5 * M6 (lgbm_minilm_cumctr_mono, doc 13), both standardised within impressions.
       The weight 0.5 is fixed in advance. A grid of weights (0, 0.1, ..., 1) is written as a diagnostic only:
       it is chosen on val, so the best grid value is in-sample.
    Both also as a switch: empty history -> cum_ctr (the group-0 rule of docs 12 and 14).

M6's val scores were not saved in doc 13; they are rebuilt from the saved model and the saved val features
(lgbm_val/minilm5_val.npy + cum_ctr) and must reproduce the saved run per impression (asserted).

Command line (repository root):
    python news/ensemble_val.py [cscale|cat_fresh]      # default cscale (doc 15)
Each base writes to its own folder and refuses to run again over an existing results.json.
'''
import json
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

import mind_eval as me
import news_protocol as npr
import lgbm_val as lv
from hybrid_e import group_means

ENSEMBLE = me.ROOT / "output" / "news" / "ensemble"
FEATURES = me.ROOT / "output" / "news" / "features"
# base -> (neural variant, short tag for run names, output folder); cscale keeps the doc 15 names and folder
BASES = {"cscale": ("pop_pos_neg8_cscale", "cscale", ENSEMBLE),
         "cat_fresh": ("pop_pos_neg8_cscale_cat_fresh", "catfresh", ENSEMBLE / "cat_fresh")}
M6_MODEL = me.ROOT / "output" / "news" / "lgbm_newuser" / "minilm_cumctr_mono.txt"
M6_RUN = "lgbm_minilm_cumctr_mono"
WEIGHT = 0.5


def standardise(data, scores):
    """(s - mean) / sd within every impression; impressions whose scores are all equal get 0."""
    starts, sizes = data.offsets[:-1], data.sizes
    s = scores.astype(np.float64)
    mean = np.add.reduceat(s, starts) / sizes
    centred = s - np.repeat(mean, sizes)
    sd = np.sqrt(np.add.reduceat(centred ** 2, starts) / sizes)
    sd = np.repeat(sd, sizes)
    return np.divide(centred, sd, out=np.zeros_like(centred), where=sd > 0)


def m6_scores(cum_ctr_val):
    booster = lgb.Booster(model_file=str(M6_MODEL))
    assert booster.feature_name() == lv.MINILM_FEATURES + ["cum_ctr"], booster.feature_name()
    X = np.hstack([np.load(lv.OUT / "minilm5_val.npy"), cum_ctr_val.astype(np.float32)[:, None]])
    return booster.predict(X)


def same_as_saved(data, scores, saved):
    """Per-impression metrics of these scores equal those of a saved run (nothing is written)."""
    a = me.per_impression_metrics(data, scores)
    a = a[a["AUC"].notna()].set_index("impression_id")
    b = npr.per_impression("val", saved).set_index("impression_id")
    assert a.index.equals(b.index), saved
    for m in me.METRICS:
        assert np.allclose(a[m], b[m], atol=1e-12, equal_nan=True), (saved, m)


def main(base):
    variant, tag, OUT = BASES[base]
    if (OUT / "results.json").exists():
        sys.exit(f"{OUT / 'results.json'} exists; not overwritten")
    OUT.mkdir(parents=True, exist_ok=True)
    nrms = me.ROOT / "output" / "news" / "nrms_user" / variant
    seed_runs = {42: nrms / "scores_val.npy", 1: nrms / "seed1" / "scores_val.npy", 2: nrms / "seed2" / "scores_val.npy"}
    single, single_switch = f"nrms_user_{variant}", f"nrms_switch_g0_cumctr__{variant}"
    avg = f"nrms_{tag}_seedavg3"
    blend = "blend_seedavg3_m6_equal" if base == "cscale" else f"blend_{tag}_seedavg3_m6_equal"
    train_all = me.load_impressions("train")
    mask = npr.val_mask(train_all)
    slot_val = np.repeat(mask, train_all.sizes)
    val = npr.subset(train_all, mask, "val")
    del train_all
    cum_ctr = np.load(FEATURES / "cum_ctr_train.npy", mmap_mode="r")[slot_val].astype(np.float64)
    assert np.isfinite(cum_ctr).all() and len(cum_ctr) == val.n_candidates
    empty = np.repeat(np.diff(val.history_offsets) == 0, val.sizes)

    # M6 rebuilt: must equal the saved doc 13 run, impression by impression
    m6 = m6_scores(cum_ctr)
    np.save(OUT / "m6_scores_val.npy", m6)
    me.evaluate(val, m6, save=False)
    same_as_saved(val, m6, M6_RUN)
    print("M6 rebuilt = saved run, per impression")

    # A: seed average
    seeds = {s: np.load(p).astype(np.float64) for s, p in seed_runs.items()}
    assert all(len(v) == val.n_candidates for v in seeds.values())
    a = np.mean([standardise(val, v) for v in seeds.values()], axis=0)
    np.save(OUT / "seedavg3_scores_val.npy", a)
    runs = {avg: (a, f"mean of the within-impression standardised scores of {single}, seeds 42, 1, 2")}

    # B: equal blend with M6
    za, zm = standardise(val, a), standardise(val, m6)
    b = WEIGHT * za + (1 - WEIGHT) * zm
    runs[blend] = (b, f"{WEIGHT} * seed average + {1 - WEIGHT} * M6 "
                                          "(both standardised within impressions; weight fixed in advance)")
    for name in list(runs):
        s, note = runs[name]
        runs[name + "_switch"] = (np.where(empty, cum_ctr, s), note + "; empty history: cum_ctr")

    results = {}
    for name, (s, note) in runs.items():
        results[name] = me.evaluate(val, s, name=name, notes=note)
        print(group_means(val, s).round(4))

    comparisons = [(avg, single), (avg + "_switch", single_switch), (blend, avg), (blend, M6_RUN),
                   (blend + "_switch", avg + "_switch"), (blend + "_switch", M6_RUN)]
    if base != "cscale":                                   # against the doc 15 ensemble
        comparisons += [(avg + "_switch", "blend_seedavg3_m6_equal_switch"),
                        (blend + "_switch", "blend_seedavg3_m6_equal_switch")]
    for x, y in comparisons:
        for metric in ("AUC", "nDCG@10"):
            npr.compare("val", x, y, metric).to_csv(OUT / f"compare_{x}_vs_{y}_{metric.replace('@', '')}.csv",
                                                    index=False)

    # diagnostic: weight grid (in-sample, not saved as runs)
    rows = []
    for w in np.round(np.arange(0, 1.01, 0.1), 1):
        s = np.where(empty, cum_ctr, w * za + (1 - w) * zm)
        frame = me.per_impression_metrics(val, s)
        frame = frame[frame["AUC"].notna()]
        frame["group"] = npr.history_group(frame["history_len"])
        row = {"weight_neural": w, "AUC": frame["AUC"].mean(), "nDCG@10": frame["nDCG@10"].mean()}
        row.update({f"AUC {g}": v for g, v in frame.groupby("group")["AUC"].mean().items()})
        rows.append(row)
    grid = pd.DataFrame(rows)
    grid.to_csv(OUT / "weight_grid_switch_val.csv", index=False)
    print("\nweight grid (switch, in-sample diagnostic):")
    print(grid.round(4).to_string(index=False))
    (OUT / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    choice = sys.argv[1] if len(sys.argv) > 1 else "cscale"
    if choice not in BASES:
        sys.exit(__doc__)
    main(choice)
