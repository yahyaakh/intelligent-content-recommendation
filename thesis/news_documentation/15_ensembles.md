# 15. Cheap ensembles: seed average and neural + LightGBM blend

**Date:** 2026-11-25 · **Split:** val (Nov 14, 431,517 impressions) · **Code:** `news/ensemble_val.py` · **Outputs:** `output/news/ensemble/`

## 1. Question

Doc 14's best model, the learned user encoder with a content weight per history group (`nrms_user_pop_pos_neg8_cscale`), was trained with 3 seeds. The best LightGBM, M6 (`lgbm_minilm_cumctr_mono`, doc 13), is a different kind of model on partly different inputs. Two questions, neither of which needs new training:

- **A.** Does averaging the 3 seeds beat one seed?
- **B.** Does blending the neural model with M6 beat either one alone?

## 2. Set-up

- **Standardising.** Each model's scores are put on one scale within every impression: (score − impression mean) / impression sd, and 0 when all scores in the impression are equal. The metrics are computed within impressions, so this only matters for mixing.
- **A, seed average** (`nrms_cscale_seedavg3`): the mean of the standardised scores of seeds 42, 1 and 2.
- **B, blend** (`blend_seedavg3_m6_equal`): 0.5 × standardised A + 0.5 × standardised M6. **The weight 0.5 was fixed before the run.**
- **M6 scores rebuilt.** Doc 13 did not save M6's val scores. They were predicted again from the saved model (`lgbm_newuser/minilm_cumctr_mono.txt`) and the saved val features (`lgbm_val/minilm5_val.npy` + `cum_ctr`). The script asserts that every impression's four metrics equal the saved doc 13 run, and that assertion passed.
- **Switch.** Both models are also run with the group-0 rule of docs 12 and 14: an empty history gets `cum_ctr`. The names end in `_switch`.
- **Weight grid (diagnostic only).** The weights 0, 0.1, …, 1 were tried for the switch blend. This is chosen on val, so the best value in the grid is in-sample and is not a result.

## 3. Results

### 3.1 Overall (AUC / MRR / nDCG@5 / nDCG@10)

| Run | AUC | MRR | nDCG@5 | nDCG@10 |
|---|---|---|---|---|
| M6 `lgbm_minilm_cumctr_mono` (doc 13) | 0.6750 | 0.3047 | 0.3324 | 0.3935 |
| `nrms_user_pop_pos_neg8_cscale`, seed 42 (doc 14) | 0.6834 | 0.3192 | 0.3469 | 0.4070 |
| doc 14 switch, seed 42 | 0.6840 | 0.3195 | 0.3474 | 0.4074 |
| A `nrms_cscale_seedavg3` | 0.6855 | 0.3209 | 0.3493 | 0.4094 |
| A + switch `nrms_cscale_seedavg3_switch` | 0.6859 | 0.3212 | 0.3496 | 0.4098 |
| B `blend_seedavg3_m6_equal` | 0.6919 | 0.3195 | 0.3489 | 0.4105 |
| **B + switch `blend_seedavg3_m6_equal_switch`** | **0.6921** | 0.3197 | 0.3491 | **0.4107** |

### 3.2 By history group (AUC; nDCG@10 in brackets)

| Run | 0 | 1–3 | 4–10 | >10 |
|---|---|---|---|---|
| M6 | 0.6021 (0.3257) | 0.6390 (0.3968) | 0.6661 (0.4099) | 0.6842 (0.3905) |
| A seed average | 0.5887 | 0.6361 (0.3960) | 0.6678 (0.4163) | 0.6999 (0.4122) |
| B blend | 0.5964 (0.3187) | 0.6453 (0.4012) | 0.6770 (0.4195) | 0.7050 (0.4120) |
| B + switch | 0.6049 (0.3288) | 0.6453 (0.4012) | 0.6770 (0.4195) | 0.7050 (0.4120) |

Group sizes: 11,393 / 29,744 / 94,697 / 295,683 impressions.

### 3.3 Paired comparisons (difference, 95% interval, Wilcoxon p)

| Comparison | AUC | nDCG@10 |
|---|---|---|
| A − seed 42 | +0.0021 (0.0018 to 0.0023), p 3e-166 | +0.0024 (0.0021 to 0.0027), p 5e-79 |
| A + switch − doc 14 switch | +0.0020 (0.0017 to 0.0022), p 1e-156 | +0.0023 (0.0020 to 0.0026), p 3e-73 |
| B − A | +0.0064 (0.0060 to 0.0068), p 4e-218 | +0.0011 (0.0006 to 0.0016), p 0.07 |
| B − M6 | +0.0169 (0.0165 to 0.0174) | +0.0170 (0.0165 to 0.0174) |
| B + switch − A + switch | +0.0062 (0.0058 to 0.0066), p 1e-197 | +0.0010 (0.0005 to 0.0014), p 0.046 |
| B + switch − M6 | +0.0172 (0.0167 to 0.0176) | +0.0173 (0.0168 to 0.0177) |

All values are from `output/news/ensemble/compare_*.csv`. A p shown as 0 in the CSV is below floating-point range; it is left out here.

**By group, B − A (AUC):** +0.0077 (group 0), +0.0093 (1–3), +0.0092 (4–10), +0.0051 (>10). **nDCG@10:** gains for 1–3 (+0.0052) and 4–10 (+0.0032); none for >10 (−0.0001, interval −0.0007 to +0.0005).

### 3.4 Weight grid (switch blend, in-sample diagnostic)

| Neural weight | 0 | 0.3 | 0.4 | **0.5 (fixed)** | 0.6 | 0.7 | 1.0 |
|---|---|---|---|---|---|---|---|
| AUC | 0.6750 | 0.6880 | 0.6905 | **0.6921** | 0.6928 | 0.6924 | 0.6859 |
| nDCG@10 | 0.3936 | 0.4049 | 0.4079 | **0.4107** | 0.4130 | 0.4143 | 0.4098 |

Full grid: `weight_grid_switch_val.csv`. Short histories prefer more LightGBM (1–3 peaks at 0.4, 0.6468), long ones more neural (>10 peaks at 0.6, 0.7064).

## 4. What it means

- **The seed average is a small, safe gain:** +0.002 AUC and +0.002 nDCG@10 over seed 42. By group (AUC) it is +0.0036 / +0.0033 / +0.0000 / +0.0026 (the 4–10 interval, −0.0006 to +0.0006, includes 0); on nDCG@10 every group gains. The gain is about as large as the seed spread in doc 14 (0.0015–0.0028). It also argues for the mean over seeds in the M8 OPEN item: the mean is better than any single seed and does not depend on picking one.
- **The blend is the larger gain: +0.006 AUC over the seed average.** LightGBM and the neural model make different mistakes. LightGBM sees hand-made similarity features and the monotone `cum_ctr`; the neural model sees the whole ordered history. The AUC gain is largest for short and medium histories (1–10). On nDCG@10 the blend helps only for 1–10 history articles. For long histories the neural model alone is as good at the top of the list.
- **The best val run is now B + switch: AUC 0.6921, nDCG@10 0.4107**, +0.017 / +0.017 over M6.
- **New users are still popularity's.** Even the blend scores 0.5964 for empty histories, below `cum_ctr` (0.6049) and below M6 (0.6021). The switch remains needed.
- **The fixed weight was close to the in-sample best.** 0.5 is 0.0007 AUC below the grid's best (0.6, in-sample). So little is lost by not tuning it, and there is no reason to report a tuned weight.
- **Compared with the leaderboard:** the gap to published NRMS (test AUC 67.76) and the 2020 4th place (70.44, 13 models) is about method, not about combining. The blend is a two-model ensemble, but val (Nov 14) is not the leaderboard setting (doc `leaderboard_results.md`), so the numbers are not directly comparable.

## 5. Limits

- Everything is chosen and measured on the same day (Nov 14). The design was fixed in advance (equal weights, the seeds that already existed), but the choice to try a blend came after seeing that both models are strong. The pre-registered dev test is where this is tested properly.
- The M6 rebuild used the val features. On dev, both models would need their own dev scoring code (M8 OPEN item).

## 6. Next

- Pre-registration: replace the M8 candidate with B + switch (seed mean of 3 neural models + M6, equal weight, `cum_ctr` for empty histories), or keep both as separate candidates. This needs dev scoring for the neural model.
- Stronger single models (options C–E: category and entity vectors, a candidate-aware user vector, fine-tuned MiniLM). The blend is a fixed recipe, so any stronger neural model can be put back into it.
