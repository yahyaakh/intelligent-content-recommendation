# 36 — Final pre-registered test on the 814 test users

**Status:** complete (2026-10-29). Run once.

**Pre-registration:** `output/experiments/final_test/preregistration.md`.
- Committed in `a400efe`.
- **Amendment 1** was written after that commit but before any test-user run, and is **not committed**. It corrects the claim that the test users were unused (docs 13, 15 and 16 used them) and turns the already-known results into checks.

**Script:** `final_test.py`. It calls the existing functions of `final_comparison.py`, `warm_benchmark.py`, `LightGBM_XBoost_rerank_experiment.py` and `cold_start_baselines.py`, with every setting fixed in advance.

**Outputs:** `output/experiments/final_test/`:
- `per_user_test.csv`, `summary_test.csv`, `tests_test.csv`;
- `checks_known_results.csv`, `code_check_validation.csv`;
- `llm_usage_test.csv`, `features_test_S50.csv`.

## 1. Before the run

### 1.1 Code check on the validation users

Both boards were recomputed on the 203 validation users with `final_test.py` and compared with the saved validation results.
- **Every method matched to 5 decimals, except one:** LTR v2 at S = 50 (0.4957 now vs 0.4926 in doc 23).
- **Cause:** the doc 23 run scored LTR v2 with the feature tables still in memory, straight after building them. The saved CSV gives slightly different values: up to 5 × 10⁻⁷ for the validation table (shown to change nothing) and probably float32 vs float64 differences in the training table.
  - Re-scoring from the saved tables reproduces doc 23 exactly at S = 3 but not at S = 10, 25 or 50 (−0.0045 to +0.0030).
  - The tree models react to these tiny differences.
- **Decision:** the test scores LTR v2 from saved files, the only reproducible option. The deviation is recorded here. It changes no doc 23 conclusion.

### 1.2 LLM answers for the test users (10 seeds)

| Run | Calls | Unparsable | Minor issues | Input tokens (mean / total) | Output tokens (total) | s per answer |
|---|---|---|---|---|---|---|
| DeepSeek V4.1 Flash p2 | reused from doc 13 | 0 | – | – | – | – |
| DeepSeek V4.1 Flash p5 | 814 | 0 | 0 | 2,932 / 2.39M | 76,516 | 1.2 |
| Kimi K3 p2 | 814 | 0 | 0 | 1,712 / 1.39M | 82,214 | 1.9 |
| Kimi K3 p5 | 814 | 0 | 3 | 2,999 / 2.44M | 82,220 | 1.9 |

**Cost:**
- DeepSeek p5: about $1.0 at the doc 13 rate (an estimate, not a bill).
- Kimi K3: Ollama Pro subscription credit.

### 1.3 Checks against the results already known on the test users (Amendment 1)

All 7 were reproduced exactly: Hybrid E 0.6311 / 0.5970, LTR v2 0.6609 / 0.6257, DeepSeek p2 0.6169, popularity 0.5598 (10 / 3 seeds).

## 2. Results (NDCG@10, 814 test users; S = 50: 758 test users)

### 2.1 Cold start, 10 seeds

| Method | NDCG@10 | P@10 | R@10 | vs Hybrid E | Validation |
|---|---|---|---|---|---|
| **LTR v2** | **0.6609** | **0.6403** | 0.0610 | +0.030 | 0.6686 |
| EASE (λ 20,000) | 0.6508 | 0.6323 | **0.0623** | +0.020 | 0.6514 |
| User-kNN (k 200) | 0.6458 | 0.6316 | 0.0606 | +0.015 | 0.6417 |
| LLM fusion (Hybrid E + DeepSeek p2, w 0.4) | 0.6317 | 0.6128 | 0.0582 | +0.001 | 0.6395 |
| **Hybrid E** (reference) | 0.6311 | 0.6155 | 0.0579 | – | 0.6286 |
| LLM: DeepSeek p5 | 0.6237 | 0.6093 | 0.0567 | −0.008 | 0.6259 |
| LLM: Kimi K3 p5 | 0.6227 | 0.6074 | 0.0561 | −0.008 | 0.6207 |
| LLM: Kimi K3 p2 | 0.6179 | 0.6007 | 0.0574 | −0.013 | 0.6274 |
| LLM: DeepSeek p2 | 0.6169 | 0.6017 | 0.0571 | −0.014 | 0.6212 |
| Item-CF | 0.6166 | 0.6015 | 0.0570 | −0.015 | 0.5990 |
| NCF (fold-in) | 0.5984 | 0.5850 | 0.0565 | −0.033 | 0.5980 |
| Popularity | 0.5598 | 0.5477 | 0.0490 | −0.071 | 0.5626 |
| SVD (fold-in) | 0.5260 | 0.4999 | 0.0453 | −0.105 | 0.5236 |
| Content (embedding + LLM labels) | 0.4051 | 0.3698 | 0.0347 | −0.226 | 0.4004 |
| Hybrid (CF + SVD) | 0.3498 | 0.3478 | 0.0330 | −0.281 | 0.3580 |
| Content (TF-IDF) | 0.3192 | 0.2771 | 0.0254 | −0.312 | 0.3211 |
| Collaborative (5 neighbours) | 0.3151 | 0.3135 | 0.0299 | −0.316 | 0.3191 |

### 2.2 Cold start, 3 seeds

| Method | NDCG@10 | P@10 | R@10 | vs Hybrid E | Validation |
|---|---|---|---|---|---|
| **LTR v2** | **0.6257** | **0.6087** | 0.0564 | +0.029 | 0.6227 |
| EASE (λ 120,000) | 0.6141 | 0.5999 | **0.0569** | +0.017 | 0.6055 |
| User-kNN | 0.6087 | 0.5951 | 0.0555 | +0.012 | 0.6053 |
| **Hybrid E** | 0.5970 | 0.5851 | 0.0542 | – | 0.5837 |
| Popularity | 0.5598 | 0.5477 | 0.0490 | −0.037 | 0.5626 |
| NCF | 0.5466 | 0.5284 | 0.0474 | −0.050 | 0.5537 |
| Item-CF | 0.5388 | 0.5244 | 0.0487 | −0.058 | 0.5132 |
| SVD | 0.5324 | 0.5095 | 0.0465 | −0.065 | 0.5323 |
| Hybrid (CF + SVD) | 0.3587 | 0.3550 | 0.0330 | −0.238 | 0.3301 |
| Content (embedding + LLM labels) | 0.3471 | 0.3236 | 0.0304 | −0.250 | 0.3458 |
| Collaborative (5 neighbours) | 0.2860 | 0.2787 | 0.0250 | −0.311 | 0.2773 |
| Content (TF-IDF) | 0.2296 | 0.2005 | 0.0178 | −0.367 | 0.2470 |

### 2.3 Longer history, S = 50 liked films (758 test users)

| Method | NDCG@10 | P@10 | R@10 | vs Hybrid E | Validation (185 users) |
|---|---|---|---|---|---|
| **LTR v2** | **0.5057** | **0.4859** | **0.0651** | +0.044 | 0.4957 (doc 23: 0.4926) |
| EASE (λ 3,000) | 0.4829 | 0.4714 | 0.0645 | +0.021 | 0.4737 |
| User-kNN (k 100) | 0.4627 | 0.4517 | 0.0596 | +0.001 | 0.4746 |
| **Hybrid E** (α 0.5) | 0.4616 | 0.4526 | 0.0605 | – | 0.4549 |
| Item-CF | 0.4584 | 0.4488 | 0.0602 | −0.003 | 0.4493 |
| NCF (fold-in 300 / 0.05) | 0.4067 | 0.3970 | 0.0514 | −0.055 | 0.4277 |
| Popularity | 0.3774 | 0.3704 | 0.0466 | −0.084 | 0.3768 |
| Content | 0.3092 | 0.2844 | 0.0368 | −0.152 | 0.3104 |
| SVD (fold-in) | 0.3071 | 0.2865 | 0.0334 | −0.155 | 0.2897 |

### 2.4 Pre-registered tests

Two-sided paired Wilcoxon. CIs are bootstrap over users (10,000 resamples).

**Family A (primary), Bonferroni ×5:**

| # | Comparison | Difference | 95% CI | Bonferroni CI (99%) | Better / worse | p_Bonf | Verdict |
|---|---|---|---|---|---|---|---|
| A3 | EASE > Hybrid E, 10 seeds | **+0.020** | [+0.011, +0.029] | [+0.008, +0.032] | 431 / 336 | 0.0001 | ✅ **confirmed** |
| A4 | EASE > Hybrid E, 3 seeds | **+0.017** | [+0.008, +0.026] | [+0.006, +0.028] | 421 / 326 | 0.0001 | ✅ **confirmed** |
| A5 | LLM fusion > Hybrid E, 10 seeds | +0.001 | [−0.008, +0.009] | [−0.010, +0.011] | 372 / 394 | 1.0 | ❌ not confirmed |
| A6 | LTR v2 > Hybrid E, S = 50 | **+0.044** | [+0.032, +0.055] | [+0.029, +0.059] | 447 / 272 | < 0.0001 | ✅ **confirmed** |
| A7 | Kimi K3 p2 ≡ Hybrid E (±0.02), 10 seeds | −0.013 | [−0.025, −0.001] | [−0.029, +0.003] | 365 / 417 | 0.036 | ❌ **not shown equivalent** |

**Family B (secondary), Bonferroni ×7:**

| # | Comparison | Difference | 95% CI | p_Bonf | Result |
|---|---|---|---|---|---|
| B1 | LTR v2 vs EASE, 10 seeds | +0.010 | [+0.000, +0.020] | 0.40 | not significant |
| B2 | LTR v2 vs EASE, 3 seeds | +0.012 | [+0.003, +0.020] | 0.26 | not significant |
| B3 | User-kNN vs Hybrid E, 10 seeds | +0.015 | [+0.008, +0.022] | 0.0009 | **higher** |
| B4 | User-kNN vs Hybrid E, 3 seeds | +0.012 | [+0.003, +0.021] | 0.23 | not significant |
| B5 | NCF vs Hybrid E, 10 seeds | −0.033 | [−0.048, −0.017] | 0.0001 | **lower** |
| B6 | NCF vs Hybrid E, 3 seeds | −0.050 | [−0.065, −0.036] | < 0.0001 | **lower** |
| B8 | EASE vs Hybrid E, S = 50 | +0.021 | [+0.010, +0.033] | 0.0012 | **higher** |

## 3. Findings

1. **LTR v2 is the best method for new users at every history length tested** (3, 10 and 50 liked films).
   - Its lead over Hybrid E at 3 and 10 seeds was already confirmed in doc 16 (+0.030 / +0.029).
   - It is now also confirmed with 50 liked films: +0.044 (A6).
2. **EASE is confirmed above Hybrid E** with 10 seeds (+0.020) and 3 seeds (+0.017), and is also higher at S = 50 (+0.021).
   - LTR v2 is not significantly above EASE (+0.010 / +0.012; B1, B2).
   - So a closed-form linear model with one setting comes within about 0.01 of the tuned learned ranker.
3. **User-kNN is above Hybrid E with 10 seeds (+0.015).** With 3 seeds, and at S = 50, it is not significantly different.
4. **LLM reranking did not improve on Hybrid E for new users.**
   - All four LLM rerankers are 0.008–0.014 below Hybrid E.
   - The fusion with Hybrid E, which was +0.011 on validation, is +0.001 on test (A5 not confirmed).
   - Kimi K3 p2 was not shown equivalent within ±0.02 (A7). Its 95% CI lies entirely below zero: as a two-sided comparison it is **significantly below** Hybrid E (−0.013, p_Bonf 0.036).
   - The safe statement: the LLMs rerank about as well as, or slightly worse than, the classical candidate list they start from.
5. **NCF with fold-in is below Hybrid E** at both seed counts (B5, B6). Of the six classical baselines, only NCF beats popularity, and only with 10 seeds (Chapter 4 of the report).
6. **Validation-to-test shrinkage was small.**
   - The largest drops were LTR v2 (−0.008, 10 seeds), the fusion (−0.008) and Kimi K3 p2 (−0.010). These are the methods whose choices used the most validation runs, as expected.
   - The classical methods moved by −0.017 to +0.029. Most of the larger rises are at 3 seeds: Hybrid E +0.013, item-CF +0.026, the hybrid +0.029.

## 4. Deviations and limitations

- **Amendment 1 is not committed.** Its date cannot be proven by git. It was written before any test-user run in this session, and all its checks passed.
- **LTR v2 at S = 50 is scored from the saved CSV tables** (§1.1). Doc 23's validation value cannot be reproduced exactly (0.4926 vs 0.4957).
- **The test users had been used before** (docs 13, 15, 16). Those results were reproduced, not retested.
- **LLM answers come from hosted models on 2026-10-29.** The same model names may behave differently later.
- **The S = 50 board has 758 of the 814 test users** (those with ≥ 55 liked films).

## 5. Safe statements for the TDK

- "On 814 held-out new users, the learned ranker (LTR v2) was the best method with 3, 10 and 50 liked films. Its lead over the tuned hybrid (Hybrid E) was +0.029 to +0.044 NDCG@10, confirmed in pre-registered tests."
- "EASE, a closed-form linear model, was significantly better than Hybrid E (+0.020 with 10 seeds, +0.017 with 3; pre-registered) and not significantly different from the learned ranker."
- "Reranking Hybrid E's top 30 with an LLM (DeepSeek V4.1 Flash or Kimi K3, with or without collaborative evidence in the prompt) did not improve NDCG@10 (−0.008 to −0.014). Combining the LLM's ranking with Hybrid E's gave no gain on the test users (+0.001)."
- Avoid:
  - "LLMs match Hybrid E" (the pre-registered equivalence was not shown);
  - "EASE equals LTR v2" (not significantly different is not equal).
