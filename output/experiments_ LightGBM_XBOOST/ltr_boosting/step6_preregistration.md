# Pre-registration: final test of the LTR v2 ranker

**Fixed on:** 2026-10-25, before any LTR v2 model has been scored on the test users.
**Script:** `LightGBM_XBoost_rerank_experiment.py final` (runs once; it refuses to run again if results exist)
**Related:** doc 15 (LTR v1, models trained on 203 validation users); LTR v2 steps 1–5 (validation only)

---

## 1. What has been decided before the test

All choices below were made using **only** the 9,084 simulated training users (warm users, cross-fitted) and the 203 validation users. The 814 test users have **not** been used for any LTR v2 decision. They were used once before, in doc 15, to test models trained on the 203 validation users. Those models are only a comparator here (H2).

| Decision | Choice | Based on |
|---|---|---|
| Training data | All 9,084 simulated new users from the warm users, A/B cross-fitted (features of each half from the other half's ratings) | Step 1: +0.017 to +0.033 over training on 203 users (validation) |
| Features | **R7** = G1 collaborative + G2 content + G3 LLM labels (DeepSeek V4.1 Flash, v5) + G5 latent (SVD, co-visitation, recent seeds); 25 features | Step 2 |
| Candidates | **C30**: Hybrid E's top 30 for the given number of seeds | Step 3: C50 and C30+20 not reliably better |
| Model | **50/50 average of within-user percentile ranks of tuned XGBoost and tuned LightGBM** (`ensemble_scores`) | Step 4, chosen **by rule** (independent of which library wins), not by validation score |
| Parameters | `step4_chosen_params.json`: XGBoost depth 6, 500 trees, learning rate 0.05, min_child_weight 10; LightGBM 63 leaves, 200 trees, learning rate 0.05, min_child_samples 20; all other settings as `PARAMS` / `XGB_PARAMS` (deterministic) | Step 4 grid on 1,816 held-out simulated users |
| Seeds | 10 (main) and 3 (additional), the same pipeline; Hybrid E's α per seed count from `alphas.csv` (0.25 and 0.12) | Steps 1–5 |

Validation results of the final model, for reference (not confirmatory): 10 seeds **0.6686** (Hybrid E 0.6286); 3 seeds **0.6227** (Hybrid E 0.5837).

## 2. Test users and ground truth

- **Users:** the 814 cold-start test users of doc 06. 814 have at least 10 liked films; the same users are used with 3 seeds.
- **Seeds:** the first 10 or 3 liked films in time order. All 10 seed films are excluded from recommendations in both settings.
- **Answer key:** all other liked films (rating ≥ the user's own mean) in the pool. It is never a model input.
- **Features:** computed from the full warm users' ratings (not the half tables), as for the validation users.
- **Metric:** NDCG@10 per user, averaged over users.

## 3. Hypotheses and tests

| # | Seeds | Comparison | Hypothesis |
|---|---|---|---|
| **H1** | 10 | final model vs **Hybrid E** | the final model ranks better than Hybrid E |
| **H2** | 10 | final model vs **XGBoost R2 trained on the 203 validation users** (doc 15 setup, default settings; test NDCG 0.6446, the best LTR v1 result) | LTR v2 improves on LTR v1 |
| **H3** | 3 | final model vs **Hybrid E** | the improvement also holds with 3 seeds |

- **Test:** paired two-sided Wilcoxon signed-rank test on per-user NDCG@10 differences.
- **Correction:** Bonferroni over the 3 tests; a hypothesis is **confirmed** if the corrected p < 0.05 and the mean difference is positive.
- **Also reported, descriptively, without tests:**
  - tuned XGBoost and tuned LightGBM separately;
  - the oracle@30;
  - users better / worse;
  - a 95% bootstrap confidence interval of each mean difference (10,000 resamples of users, seed 42).

## 4. Sanity checks (must pass before the results are reported)

1. Sorting the test candidates by `hybrid_rank` reproduces Hybrid E: **0.6311** (10 seeds) and **0.5970** (3 seeds).
2. XGBoost R2 trained on `features_val.csv` reproduces its doc-15 test result, **0.6446**.
3. The 10-seed test table with G5 has the same users, films and order as `features_test.csv`. No missing values; 30 rows per user.
4. No test user appears in the training tables.

If a check fails, the run stops, and the cause is fixed and documented before any result is used.

## 5. Reporting commitments

- All three tests are reported whatever their outcome, with the numbers above. No other model, feature set, candidate list or setting is substituted after seeing the test results.
- The validation-only steps (1–5) are reported as exploratory, including the variants that were not chosen (C50, C30+20, single tuned models, R6).
- If H2 fails while H1 holds, the conclusion is that the LTR v2 changes did not significantly improve on LTR v1 on the test users, even though both beat Hybrid E.
