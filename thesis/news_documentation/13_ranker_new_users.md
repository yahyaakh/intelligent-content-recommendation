# 13 — A ranker for new users: weighting fails; fewer popularity features and a monotone click rate work

**Status:** done 2026-11-23 (validation only).
- **Code:**
  - `news/lgbm_newuser.py`: variants `base`, `w_sqrt`, `w_equal`;
  - `news/lgbm_newuser_diag.py`: variants `pop_g0`, `pop_all`, `pop_all_to13`, `minilm_cumctr`, `minilm_cumctr_mono`.
- **Outputs:**
  - `output/news/lgbm_newuser/`:
    - models `<variant>.txt` and `<variant>_info.json` (settings, weights, gain shares);
    - paired tests `compare_*.csv`;
    - `groups_val_AUC.csv`, `groups_val_nDCG10.csv`, `summary.txt`;
    - `insample_check_nov13.txt`, `pop_all_to13_auc_nov13.csv`;
  - run logs `output/news/lgbm_newuser_<variant>_log.txt`;
  - runs `lgbm_pop15_*`, `lgbm_pop_*` and `lgbm_minilm_cumctr*` in `output/news/results/val/`.

---

## 1. Question

Docs 11 and 12 found that LightGBM (5 MiniLM + 10 popularity features) is the best model overall on val. For an empty history, however, it is 0.020 AUC below popularity, so a per-group switch was needed. Can one learned ranker handle new users itself?

**Target:** a single ranker that is not below popularity for an empty history, and not below the switch overall.

## 2. Set-up

- **Training and test:** every ranker is trained on Nov 9–13 and scored on val (Nov 14, 431,517 impressions, protocol of doc 06).
- **Settings:** those of the doc 04 ranker (lambdarank, 100 trees, learning rate 0.05, 15 leaves, seed 42, 8 threads, other LightGBM defaults). They are read from the saved model file.
- **Features:**
  - the MiniLM features are the cached training matrix of doc 04, checked in doc 11;
  - the popularity features are those of doc 03.
- **Memory:** LightGBM reads the 66 million training rows in batches (`lgb.Sequence`).
- **Retraining check (`base`):** the doc 04 ranker retrained with this pipeline.
  - It differs from the saved model only in the histogram bin boundaries: the batched input samples a different set of 200,000 rows to place the bins.
  - Its first tree splits on the same features in the same order, with thresholds that agree to 3–4 digits.
  - On val it is within 0.0005 AUC and 0.001 nDCG@10 of the saved model (overall AUC difference not significant, p = 0.68).
  - All variants below are compared with `base`, which was built with the same pipeline.

| Variant | What changes compared with `base` |
|---|---|
| `w_sqrt` | each training impression weighted ∝ 1/√(share of its history group). Weights per impression: 4.2 / 2.2 / 1.2 / 0.71 for groups 0 / 1-3 / 4-10 / >10 |
| `w_equal` | weighted ∝ 1/share, so every group has the same total weight. Weights per impression: 13.0 / 3.5 / 1.07 / 0.37 |
| `pop_g0` | only the 10 popularity features, trained on the 34,672 empty-history impressions only |
| `pop_all` | only the 10 popularity features, trained on all impressions |
| `pop_all_to13` | as `pop_all`, trained on Nov 9–12 and scored on Nov 13 (a next-day check, not a val run) |
| `minilm_cumctr` | the 5 MiniLM features + `cum_ctr` only (9 popularity features dropped) |
| `minilm_cumctr_mono` | as `minilm_cumctr`, with a monotone constraint: the score may only rise with `cum_ctr` |

Empty histories are 1.9% of the training impressions. By training day they number 2,042 on Nov 9, 3,104 on Nov 10, 8,125 on Nov 11, 10,393 on Nov 12 and 11,008 on Nov 13.

## 3. Results

### 3.1 Overall (val)

| Run | AUC | MRR | nDCG@5 | nDCG@10 |
|---|---|---|---|---|
| `single_cum_ctr` (popularity) | 0.6076 | 0.2571 | 0.2692 | 0.3309 |
| `hybrid_e_content_cumctr` (doc 09) | 0.6603 | 0.2960 | 0.3201 | 0.3827 |
| `lgbm_minilm_pop15` (saved doc 04 ranker) | 0.6655 | 0.3045 | 0.3316 | 0.3933 |
| `group_switch_auc` (doc 12) | 0.6676 | 0.3030 | 0.3298 | 0.3917 |
| `lgbm_pop15_base` | 0.6650 | 0.3031 | 0.3302 | 0.3923 |
| `lgbm_pop15_w_sqrt` | 0.6660 | 0.3019 | 0.3292 | 0.3912 |
| `lgbm_pop15_w_equal` | 0.6666 | 0.3019 | 0.3295 | 0.3913 |
| `lgbm_pop_g0` | 0.5530 | 0.2253 | 0.2323 | 0.2983 |
| `lgbm_pop_all` | 0.5606 | 0.2266 | 0.2347 | 0.3011 |
| `lgbm_minilm_cumctr` | 0.6733 | 0.3040 | 0.3320 | 0.3928 |
| `lgbm_minilm_cumctr_mono` | **0.6750** | **0.3047** | **0.3324** | **0.3935** |

### 3.2 AUC by history group (val)

| Run | 0 | 1-3 | 4-10 | >10 | all |
|---|---|---|---|---|---|
| popularity | **0.6049** | 0.6118 | 0.6127 | 0.6056 | 0.6076 |
| group switch | **0.6049** | 0.6347 | 0.6559 | 0.6771 | 0.6676 |
| `lgbm_pop15_base` | 0.5832 | 0.6219 | 0.6508 | 0.6770 | 0.6650 |
| `lgbm_pop15_w_sqrt` | 0.5688 | 0.6200 | 0.6529 | 0.6785 | 0.6660 |
| `lgbm_pop15_w_equal` | 0.5611 | 0.6228 | 0.6546 | 0.6789 | 0.6666 |
| `lgbm_pop_g0` | 0.5481 | 0.5470 | 0.5505 | 0.5546 | 0.5530 |
| `lgbm_pop_all` | 0.5572 | 0.5549 | 0.5577 | 0.5623 | 0.5606 |
| `lgbm_minilm_cumctr` | 0.5913 | 0.6349 | 0.6647 | 0.6831 | 0.6733 |
| `lgbm_minilm_cumctr_mono` | 0.6021 | **0.6390** | **0.6661** | **0.6842** | **0.6750** |

nDCG@10 by group is in `groups_val_nDCG10.csv`. For `lgbm_minilm_cumctr_mono` it is 0.3257 / 0.3968 / 0.4099 / 0.3905. The switch scores 0.3288 / 0.3885 / 0.3970 / 0.3927.

### 3.3 Weighting short histories makes new users worse

Compared with `base`, in AUC:

| Group | `w_sqrt` | `w_equal` |
|---|---|---|
| 0 | −0.014 (95% CI −0.018 to −0.011) | −0.022 (−0.026 to −0.018) |
| >10 | +0.002 | +0.002 |

The small overall AUC gains (+0.001 and +0.002) come from long histories. nDCG@10 falls overall in both variants.

### 3.4 A learned mix of popularity features does not carry over to the next day

On val, both popularity-only rankers are far below `cum_ctr` alone, in every history group:
- `pop_all`: −0.047 AUC;
- `pop_g0`: −0.055 AUC.

On days the ranker was trained on, the same `pop_all` model beats `cum_ctr` clearly:

| Day | `cum_ctr` | `lgbm_pop_all` |
|---|---|---|
| Nov 12 (training) | 0.5579 | 0.6299 |
| Nov 13 (training) | 0.5986 | 0.6362 |
| Nov 14 (val, unseen) | 0.6076 | 0.5606 |

The next-day check trains on Nov 9–12 and scores Nov 13. There the learned mix is level with `cum_ctr`: 0.5964 vs 0.5986 overall, and 0.5883 vs 0.5885 for an empty history.

On an unseen day the mix of 10 popularity features never beats the single smoothed click rate. On Nov 14 it is clearly worse. The model is small (100 trees of 15 leaves on 66 million rows), so this is not memorisation. The relation between counts such as `shown_1h` or `prev_shows` and clicks changes from day to day, while the smoothed click rate stays useful.

### 3.5 Keeping only `cum_ctr`, with a monotone constraint

- **`minilm_cumctr`** (6 features) beats `base` in every group on AUC: +0.008 overall, and +0.008 to +0.014 in groups 0–10.
- **The monotone constraint** adds another +0.0017 AUC overall, mostly in group 0 (+0.011).
  - With an empty history all MiniLM features are 0, so the score becomes a rising function of `cum_ctr` alone.
  - Group 0 reaches 0.6021, against 0.6049 for popularity: still 0.003 lower (95% CI −0.0054 to −0.0002).
  - The remaining gap is likely due to ties: a tree model's score changes in steps, so close `cum_ctr` values share one score, and ties are broken at random.
- **Against the switch,** `minilm_cumctr_mono` is ahead overall: AUC +0.0073 (95% CI +0.0069 to +0.0078) and nDCG@10 +0.0018 (+0.0014 to +0.0022). By group:

| Group | AUC difference | nDCG@10 difference |
|---|---|---|
| 0 | −0.003 | −0.003 |
| 1-3 | +0.004 | +0.008 |
| 4-10 | +0.010 | +0.013 |
| >10 | +0.007 | −0.002 |

- **Gain shares** of the monotone ranker:

| Feature | Share |
|---|---|
| `cum_ctr` | 0.378 |
| `avg_profile_similarity` | 0.291 |
| `max_similarity` | 0.145 |
| `top5_similarity` | 0.064 |
| `history_length` | 0.063 |
| `category_preference` | 0.059 |

## 4. What this means

1. **The new-user problem was not a lack of examples.** It came from the popularity features themselves. Giving empty histories more weight makes the ranker lean more on popularity features whose learned use does not carry over to a new day, so new users get worse.
2. **The robust popularity signal is the smoothed click rate.** This is the news version of the movie lesson that popularity is a hard bar: here it is hard to *learn* a better popularity score than the simple `cum_ctr`.
3. **One ranker now does what the switch did, and better.** `lgbm_minilm_cumctr_mono` (5 MiniLM features + monotone `cum_ctr`):
   - is the best run on val by every metric (AUC 0.675, nDCG@10 0.394);
   - is ahead of the switch by +0.007 AUC;
   - is within 0.003 AUC of popularity for an empty history.

   It is also simpler: 6 features instead of 15.
4. **Caveat: these choices were made on val.** Dropping 9 features and adding the monotone constraint were both decided after looking at val, among 7 variants. The val numbers are therefore somewhat optimistic. The pre-registered test on dev (Nov 15) decides.
   - Doc 04's 15-feature ranker did well on dev (0.682), so the Nov 14 drop of the 10 popularity features may be larger than on Nov 15.
   - The final test should therefore include both rankers.

## 5. Next steps

- **Pre-registration (news):** fill in the methods with the settings fixed above:
  - popularity;
  - Hybrid E;
  - the 15-feature LightGBM;
  - the switch;
  - `lgbm_minilm_cumctr_mono`.

  Then add hypotheses, for example "`lgbm_minilm_cumctr_mono` > LightGBM 15 overall" and "`lgbm_minilm_cumctr_mono` ≥ popularity − margin for an empty history".
- **Training on all of train:** for the final test, decide whether the rankers are retrained on Nov 9–14 (all of train) before scoring dev. That must be fixed in the pre-registration.
- **Still open from doc 09:** earlier-day clicks as history, which would reduce the number of empty histories, and a rate-based audience feature.
