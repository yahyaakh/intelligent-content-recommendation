# Pre-registration: final test on the 814 test users (doc 36)

**Status:** DRAFT, written 2026-10-29. It is frozen when it is **committed to git**: the commit date proves it was fixed before the run (doc 18 §9, step 2). After that commit nothing in §2–§6 changes. Any later deviation is listed in doc 36 as a deviation.
## Amendment 1 (2026-10-29, after commit a400efe, before any run on the test users)

**Correction to §1.** §1 says the test users have never been used. That is wrong. They were used before:

| Doc | Measured on the 814 test users |
|---|---|
| 13 | DeepSeek V4.1 Flash p2 reranking Hybrid E, 10 seeds: 0.6169 (answers cached); also the popularity-candidate and gemma runs |
| 15 | LTR v1 (XGBoost R2): 0.6446 |
| 16 | **pre-registered:** LTR v2 0.6609 vs Hybrid E 0.6311 (10 seeds); 0.6257 vs 0.5970 (3 seeds); popularity 0.5598 |

**Changes, which replace the matching parts of §2–§5:**

1. **Already-known results are checks, not hypotheses.** These must be reproduced to within 5 × 10⁻⁴, or the run stops and the mismatch is reported:
   - Hybrid E (both N);
   - LTR v2 (both N);
   - DeepSeek p2 (N = 10);
   - popularity.

   A1, A2 and B7 are removed from the families.
2. **The DeepSeek p2 test answers of doc 13 are reused**, not requested again, so the numbers stay consistent with doc 13 and the fusion uses the same answers. The new calls are DeepSeek p5, Kimi K3 p2 and Kimi K3 p5 (N = 10, 814 users each).
3. **Family A (primary), Bonferroni over 5:**

   | # | Hypothesis | Note |
   |---|---|---|
   | A3 | EASE > Hybrid E, N = 10 | |
   | A4 | EASE > Hybrid E, N = 3 | |
   | A5 | LLM fusion > Hybrid E, N = 10 | |
   | A6 | LTR v2 > Hybrid E, S = 50 | |
   | A7 | Kimi K3 p2 equivalent to Hybrid E, N = 10 | equivalent if the 99% CI (1 − 0.05/5) lies inside ±0.02 |

4. **Family B (secondary), Bonferroni over 7:** B1–B6 and B8 as in §5.
5. **Bootstrap:** 10,000 resamples (seed 42) instead of 1,000, so the 99% interval of A7 is stable.
6. **Code check before the test run.** Both boards are first recomputed on the validation users with the same code and must equal the saved validation results (`final_comparison/summary_val.csv`, `warm_benchmark/summary_val.csv`).
7. **The test run uses its own driver script, `final_test.py`.** It calls the existing functions of `final_comparison.py`, `warm_benchmark.py`, `LightGBM_XBoost_rerank_experiment.py` and `cold_start_baselines.py` with the fixed settings. The validation scripts are left unchanged.

**Plan it implements:** doc 18 §9.
**Scripts:** `final_comparison.py` (cold-start board, N = 10 and N = 3) and `warm_benchmark.py` (history S = 50). Both currently refuse the test split. The only code change allowed before the run is a `test` option that loads the test users instead of the validation users, with no other change.
**Outputs:** `output/experiments/final_test/` (own folder and own LLM caches, never mixed with the validation runs).

## 1. Purpose

- **Every choice in this work was made on the 203 validation users:** prompts, models, α, k, λ, LTR features and parameters.
- **The 814 test users have never been used.** This test measures the chosen methods **once**, on those users, so the reported numbers are not inflated by those choices.
- **Everything is reported, including null results.**

## 2. Users and protocol (unchanged from the validation runs)

- **Users:** the 814 test users of `cs.split_validation` (cold users, seed 42).
- **Seeds:** each user's first N liked films in time order, N = 10 and N = 3. The answer key is the same at both N: their other liked pool films. All 10 seed films are excluded at both N.
- **History S = 50 (doc 23 design):**
  - test users with ≥ 50 liked pool films + 5 more;
  - the first 50 are excluded at every S, and the answer key = the rest.
  - If fewer than 120 test users qualify, S = 25 is used instead (doc 18 §7b rule).
- **Training data:** the 9,591 warm users only, as in every validation run. No model is retrained on test users.
- **Metric:** NDCG@10 decides. Precision@10, Recall@10 and coverage are reported.
- **Tests:**
  - two-sided paired Wilcoxon signed-rank over users;
  - 95% user-bootstrap CI of the mean difference (1,000 resamples, seed 42);
  - Bonferroni within each family (§5).

## 3. Variants fixed in advance (rules applied to the validation boards)

| Slot (doc 18 §9) | Rule | Chosen | Validation NDCG@10 (N = 10 / N = 3) |
|---|---|---|---|
| Combined LTR model | final model of docs 16 and 24 (no idea adopted since, incl. EASE as a feature) | **LTR v2** (R7 features, XGBoost + LightGBM, doc 16 step-4 parameters) | 0.6686 / 0.6227 |
| Best LLM variant (doc 17) | highest validation NDCG@10 at N = 10 among the LLM rows of the final comparison | **LLM fusion: Hybrid E + DeepSeek V4.1 Flash p2, w = 0.4** | 0.6395 / – |
| Best pure LLM reranker | the same rule, LLM rerankers only | **Kimi K3, p2** (Hybrid E top 30) | 0.6274 / – |
| Best NCF / SVD model | higher validation NDCG@10 at both N | **NCF (option 10 NeuMF, fold-in, doc 20 settings)** | 0.5980 / 0.5537 (SVD: 0.5236 / 0.5323) |
| EASE | λ chosen on the simulated warm users (`ease_tuning.py`), never on validation | **EASE**, λ 20,000 (N = 10), 120,000 (N = 3) | 0.6514 / 0.6055 |
| Semantic-ID method | a method enters only if it beat its own baseline on validation | **none.** No semantic-ID ranking method passed (docs 27, 28, 30–33). Doc 34 (rating prediction, warm users) is its own completed pre-registered test. | – |
| Warm history S = 50 | settings fixed in `warm_benchmark/settings.json` and `settings_ease.json` | Hybrid E α 0.5, user-kNN k 100, EASE λ 3,000, NCF fold-in 300 / 0.05, LTR v2 retrained as doc 23 | see §4 |

**Reference method:** Hybrid E (α from doc 06: N = 10 → 0.25, N = 3 → 0.12).

**Also run** (descriptive, for the complete board, as in `final_comparison.py`):
- popularity, content, item-CF, user-kNN (k 200), SVD;
- the option 10 methods;
- DeepSeek V4.1 Flash p2, DeepSeek p5 and Kimi K3 p5 at N = 10.

## 4. LLM calls needed (new, test users only)

| Run | Users | Prompt size | Cost |
|---|---|---|---|
| DeepSeek V4.1 Flash p2, N = 10 (also the input of the fusion) | 814 | ~1,600 tokens | ≈ $0.50 |
| DeepSeek V4.1 Flash p5, N = 10 | 814 | ~2,900 tokens (doc 17) | ≈ $0.95 |
| Kimi K3 p2 and p5, N = 10 | 814 each | ~1,600 / ~2,900 | Ollama Pro credit |

- **Total:** about $1.50 (an estimate at the doc 13 rate) plus Ollama Pro credit.
- **No LLM runs at S = 50.** At S = 50 the LLMs were not better than Hybrid E on validation (doc 23), so they are left out for cost.
- **Unparsable answers:** handled exactly as in doc 13 (`lr.parse_ranking`).
  - An answer without a usable ranking keeps Hybrid E's order.
  - Films left out of a ranking are added at the end in Hybrid E's order.
  - No answer is requested again. Every such user is counted and reported.

## 5. Hypotheses and decision rules

### Family A: primary (Bonferroni over 7; "confirmed" = p × 7 < 0.05 and the difference has the stated sign)

| # | Comparison | Validation evidence | Claim if confirmed |
|---|---|---|---|
| A1 | LTR v2 > Hybrid E, N = 10 | +0.040, p_Bonf 0.009 | the learned ranker beats the tuned hybrid |
| A2 | LTR v2 > Hybrid E, N = 3 | +0.039, p_Bonf < 0.001 | the same with 3 liked films |
| A3 | EASE > Hybrid E, N = 10 | +0.023, n.s. | EASE beats the hybrid for new users |
| A4 | EASE > Hybrid E, N = 3 | +0.022, p_Bonf 0.013 | the same with 3 liked films |
| A5 | LLM fusion > Hybrid E, N = 10 | +0.011, n.s. | combining the LLM with Hybrid E helps |
| A6 | LTR v2 > Hybrid E, S = 50 | +0.038, p_Bonf 0.005 | the ranking holds for long histories |
| A7 | **Kimi K3 p2 equivalent to Hybrid E, N = 10** | −0.001, n.s. | **"equivalent"** if the Bonferroni-adjusted CI (99.29%) of the difference lies inside [−0.02, +0.02]; "not shown equivalent" otherwise |

### Family B: secondary (Bonferroni over 8)

| # | Comparison |
|---|---|
| B1 | LTR v2 vs EASE, N = 10 |
| B2 | LTR v2 vs EASE, N = 3 |
| B3 | user-kNN vs Hybrid E, N = 10 |
| B4 | user-kNN vs Hybrid E, N = 3 |
| B5 | NCF vs Hybrid E, N = 10 |
| B6 | NCF vs Hybrid E, N = 3 |
| B7 | DeepSeek V4.1 Flash p2 vs Hybrid E, N = 10 |
| B8 | EASE vs Hybrid E, S = 50 |

### Descriptive (no claim of significance)

- Every other method and pair.
- Precision@10, Recall@10, coverage.
- The validation vs test difference of every method (overfitting to validation is expected to be small for methods with few settings and larger for LTR v2 and the LLM prompt choices).

## 6. What will be reported

- One table per board (N = 10, N = 3, S = 50): every method with NDCG@10, P@10, R@10, coverage, the difference to Hybrid E with its CI, and better/worse counts.
- Family A and B results exactly as defined in §5, whether confirmed or not.
- Validation vs test table for the methods of §3.
- **The main TDK numbers are the test numbers.** Validation numbers are labelled as such.
- LLM tokens, unparsable answers and cost.

## 7. Limits stated in advance

- **One random user split (seed 42).** The test users come from the same MovieLens population as the validation users.
- **LTR v2's features and parameters and the LLM prompts were chosen after many validation runs.** Their test results are the honest estimate, and a drop from validation is possible.
- **Hosted LLMs can change behind the same model name.** The answers are cached with the date of the run.
- **The S = 50 board has fewer users than N = 3 / 10** (validation: 185 of 203).
