# 12 — Per-group switch: the best method for each history group

**Status:** done 2026-11-23 (validation only).
- **Code:** `news/group_switch.py val`.
- **Outputs:**
  - `output/news/group_switch/`: `candidates_by_group_val.csv`, `chosen_method.json` and the paired tests `compare_*.csv`;
  - run log `output/news/group_switch_log.txt`;
  - run `group_switch_auc` in `output/news/results/val/`.

---

## 1. Question

Doc 11 found that no single method is best for every history group:
- popularity is best for an empty history;
- Hybrid E is best on AUC for 1–10 history articles;
- LightGBM is best for more than 10.

How much does it gain to use the best method in each group?

## 2. Set-up

- **Candidates**, simplest first. A tie keeps the simpler method.
  1. `single_cum_ctr`: popularity (doc 03).
  2. `hybrid_e_content_cumctr`: Hybrid E (doc 09).
  3. `lgbm_minilm_pop15`: LightGBM (docs 04, 11).
- **Rule:** in each history group, take the candidate with the highest mean AUC on val (AUC is the primary metric, doc 06). The choice uses the per-impression results already saved for val.
- **The switch as a run:** each impression is ranked by the scores of its group's method.
  - The scores are rebuilt from the saved features and models.
  - Metrics are computed within each impression, so mixing score scales across impressions does not matter.
- **Check:** every impression gets exactly the metrics of its source run (asserted for all metrics). This also confirms that the rebuilt Hybrid E and LightGBM scores reproduce the saved runs.
- **This is an in-sample result.** The choice is made on the same day it is evaluated on. It is the upper end of what the switch can give on val. The honest test is the pre-registered final test on dev.

## 3. Results

### 3.1 Choice (AUC by group on val)

| Group | Popularity | Hybrid E | LightGBM | Chosen |
|---|---|---|---|---|
| 0 | **0.6049** | 0.6049 (same ranking) | 0.5853 | popularity |
| 1-3 | 0.6118 | **0.6347** | 0.6248 | Hybrid E |
| 4-10 | 0.6127 | **0.6559** | 0.6516 | Hybrid E |
| >10 | 0.6056 | 0.6664 | **0.6771** | LightGBM |

### 3.2 Overall

| Run | AUC | MRR | nDCG@5 | nDCG@10 |
|---|---|---|---|---|
| `single_cum_ctr` | 0.6076 | 0.2571 | 0.2692 | 0.3309 |
| `hybrid_e_content_cumctr` | 0.6603 | 0.2960 | 0.3201 | 0.3827 |
| `lgbm_minilm_pop15` | 0.6655 | **0.3045** | **0.3316** | **0.3933** |
| `group_switch_auc` | **0.6676** | 0.3030 | 0.3298 | 0.3917 |

### 3.3 Paired tests, all impressions

| Comparison | AUC difference (95% CI) | Wilcoxon p | nDCG@10 difference (95% CI) | Wilcoxon p |
|---|---|---|---|---|
| switch − LightGBM | +0.0021 (+0.0018, +0.0025) | 2 × 10⁻⁵ | −0.0017 (−0.0020, −0.0013) | 1 × 10⁻¹⁷ |
| switch − Hybrid E | +0.0074 (+0.0069, +0.0078) | < 10⁻³⁰⁰ | +0.0089 (+0.0085, +0.0094) | < 10⁻³⁰⁰ |
| switch − popularity | +0.0601 (+0.0592, +0.0609) | < 10⁻³⁰⁰ | +0.0607 (+0.0600, +0.0615) | < 10⁻³⁰⁰ |

Against LightGBM, the AUC gain comes from the three groups where LightGBM is replaced:

| Group | AUC difference |
|---|---|
| 0 | +0.0196 |
| 1-3 | +0.0099 |
| 4-10 | +0.0043 |
| >10 | 0 (LightGBM is kept) |

In nDCG@10 the same switch loses:

| Group | nDCG@10 difference |
|---|---|
| 1-3 | −0.0038 |
| 4-10 | −0.0076 |

Only group 0 gains in nDCG@10 (+0.0095).

## 4. What this means

1. **The switch is the best method on val by the primary metric, but only by a little:** AUC +0.002 over LightGBM. Groups 0–10 hold only 31% of the impressions, so their gains are diluted.
2. **The part that is certain to matter is the empty history.** For new users, popularity beats LightGBM by 0.020 AUC and 0.010 nDCG@10. The same was found on dev (doc 06: 0.623 vs 0.604). This is the news version of the movie finding that popularity is the bar for new users.
3. **For 1–10 history articles the choice depends on the metric.** Hybrid E is better on AUC and LightGBM on nDCG@10 (doc 11). A switch chosen by nDCG@10 would keep LightGBM there. The pre-registration must fix one metric for the choice; the protocol says AUC.
4. **Practical reading:** a learned ranker that handles new users itself would make the switch unnecessary. Retraining LightGBM with the history group or the Hybrid E score as features (doc 11, next steps) is the cleaner alternative. The switch is the simple baseline that such a ranker has to beat.

## 5. For the pre-registration

The final test can include:
- popularity (M1);
- Hybrid E with α per group (0 / 0.3 / 0.4 / 0.5);
- LightGBM `lgbm_minilm_pop15`;
- this switch, with its method per group fixed as above.

A natural hypothesis is "switch > LightGBM in group 0" (seen on dev and val). The overall difference of +0.002 AUC is close to any equivalence margin and should be stated as such.
