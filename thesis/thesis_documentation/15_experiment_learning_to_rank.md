# Experiment: Learning-to-Rank with LLM Features (LightGBM and XGBoost)

**Status:** finished (run date: 2026-10-25)
**Script:** `LightGBM_XBoost_rerank_experiment.py` (full run: no argument; feature importance only: `importance`)
**Results folder:** `output/experiments_ LightGBM_XBOOST/ltr_boosting/`
**Related documents:** `06_experiment_cold_start.md` (Hybrid E, the cold-start split), `13_experiment_llm_as_recommender.md` (LLM reranking, ideas 1 and 3), `14_experiment_label_quality_and_vocabulary_v5.md` (DeepSeek v5 labels), `16_experiment_learning_to_rank_v2.md` (LTR v2, the follow-up)

> **Note (added 2026-10-25):** two LLM-label features (`theme_overlap`, `mood_overlap`) changed slightly from run to run because of a bug, since fixed. This affects the R3/R4 numbers and the feature importance below by about ±0.002; R1, R2, R5, Hybrid E and DeepSeek alone are exact. No conclusion changes. See section 6a.

---

## 1. Question

Experiment 13 let an LLM rerank Hybrid E's top 30 on its own (idea 1), and no model beat Hybrid E: DeepSeek V4.1 Flash came closest (test 0.6169 vs 0.6311). An LLM reranking replaces the collaborative ranking completely, though. Its signal might still help as **one input among several**.

This experiment asks:

> **When collaborative, content and LLM-derived signals are combined by a learned ranker, do the LLM signals add information that the classical signals do not already carry?**

A learned ranker settles this fairly. It weighs every feature by how much the feature helps on held-out users. If an LLM feature carries new information, a model with it must rank better than the same model without it.

---

## 2. Method

### 2.1 Learning-to-rank in brief

- **The setting:** each user has a fixed list of 30 candidate films and the model scores every candidate; the user's top 10 is the 10 highest scores.
- **The models:** both are gradient-boosted decision trees, i.e. hundreds of small trees, each correcting the errors of the earlier ones.
  - **LightGBM** `LGBMRanker`, objective `lambdarank`;
  - **XGBoost** `XGBRanker`, objective `rank:ndcg`.
- **What they optimise:** both optimise NDCG directly. Every pair of candidates within a user is weighted by how much swapping them would change NDCG. The trees learn non-linear rules and interactions, such as "trust the LLM rank only when the collaborative score is weak", that a fixed linear blend like Hybrid E cannot express.
- **Groups:** a ranker learns from whole users (a "group" or "query"), never from single rows, so all 30 rows of a user stay together in training and in cross-validation.

### 2.2 Data

| | Validation | Test |
|---|---|---|
| Users (N = 10 seeds) | 203 | 814 |
| Rows (30 candidates per user) | 6,090 | 24,420 |
| Relevant candidates | 53.7% | 54.7% |

- **Candidates:** Hybrid E's top 30, with α per N from `alphas.csv` (N = 10: α = 0.25). These are the same candidates as idea 1 of experiment 13; the ceiling is oracle@30 ≈ 0.95.
- **Label:** 1 if the candidate is in the user's answer key (later liked films), else 0. It is never used as a feature.
- **Features:** computed only from the 10 seed films, never from the answer key.
- **Checks in `build_table`:**
  - 30 rows per user and no missing values;
  - sorting by `hybrid_rank` reproduces Hybrid E (test 0.6311, validation 0.6286);
  - sorting by `rank_deepseek` reproduces DeepSeek's idea-1 NDCG (test 0.6169, validation 0.6212).

### 2.3 Features (21, in four groups)

| Group | Features | Source |
|---|---|---|
| **G1 collaborative and popularity** (8) | `cf_mean`, `cf_max` (item-CF similarity to the seeds), `hybrid_score`, `hybrid_rank`, `log_popularity`, `vote_count`, `vote_average`, `year` | ratings, TMDB metadata |
| **G2 content** (3) | `emb_mean`, `emb_max` (embeddinggemma similarity of the plain movie texts to the seeds), `content_max` (TF-IDF/CountVectorizer similarity) | movie texts, no LLM |
| **G3 LLM labels and profile** (8) | `emb_v5_mean`, `emb_v5_max` (similarity of the DeepSeek-v5-enriched texts), `theme_overlap`, `mood_overlap` (share of the user's repeated seed themes/moods that the film has), `genre_overlap`, `director_match`, `actor_overlap`, `decade_gap` | DeepSeek V4.1 Flash v5 labels (doc 14), seed profile |
| **G4 LLM rankings** (2) | `rank_deepseek`, `rank_gemma` (position 1–30 in the model's idea-1 ranking, prompt p2, shuffle 0) | cached answers from experiment 13, no new calls |

`genre_overlap`, `director_match`, `actor_overlap` and `decade_gap` come from metadata, not from an LLM. They sit in G3 because they belong to the same seed-profile idea as the LLM themes and moods. G3 therefore slightly *overstates* the LLM's share, which makes "the LLM adds nothing" a conservative conclusion.

### 2.4 Feature sets (ablation)

| Model | Groups | Features | Question |
|---|---|---|---|
| R1 | G1 | 8 | Can a learned ranker beat the fixed Hybrid E blend with the same signals? |
| R2 | G1 + G2 | 11 | Does non-LLM content help? |
| R3 | G1 + G2 + G3 | 19 | Do LLM labels help beyond content? |
| R4 | all | 21 | Do LLM rankings help on top of everything? |
| R5 | G1 + G4 | 10 | Do LLM rankings help without content? |

### 2.5 Settings

The hyperparameters are fixed in advance, with no tuning, so that the validation users stay unused for selection:

```python
# LightGBM
objective="lambdarank", n_estimators=200, learning_rate=0.05, num_leaves=15,
min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
random_state=42, deterministic=True, force_row_wise=True, n_jobs=1
# XGBoost
objective="rank:ndcg", n_estimators=200, learning_rate=0.05, max_depth=4,
min_child_weight=1, subsample=0.8, colsample_bytree=0.8, tree_method="hist",
random_state=42, n_jobs=1
```

`max_depth=4` gives at most 16 leaves, comparable to `num_leaves=15`. Both libraries are run deterministically, so every rerun gives identical numbers. An earlier draft defined `PARAMS` twice and overrode the deterministic settings, so its results changed between runs. The files without a library suffix (`cv_per_user_val.csv`, `test_tests.csv`, ...) come from that draft and are superseded.

### 2.6 Evaluation protocol

1. **Validation (exploratory):** 5-fold cross-validation **by user** on the 203 validation users. Every user is scored by a model that never saw them.
2. **Test (pre-registered, fixed 2026-10-25 before the test run):** R1–R5 are trained once on all 203 validation users and score the 814 test users once.
   - The pre-registered model is **LightGBM**, with two tests: **R4 vs R1** (do the LLM signals help a learned ranker?) and **R4 vs Hybrid E** (does the full model beat the best classical method?). Both use the Wilcoxon signed-rank test and are Bonferroni-corrected over the 2 tests.
   - **XGBoost** follows the identical procedure as a **robustness check** of the choice of library.
3. **Breakdown (descriptive):** R2 vs R1, R3 vs R2, R4 vs R2 and similar on the test users, without correction, to locate where any gain comes from.
4. **Feature importance:** of the final R4 models (section 5).

---

## 3. Results

### 3.1 NDCG@10

| Method | Validation CV, LightGBM | Validation CV, XGBoost | **Test, LightGBM** | **Test, XGBoost** |
|---|---|---|---|---|
| Hybrid E | 0.6286 | 0.6286 | 0.6311 | 0.6311 |
| DeepSeek alone (idea 1) | 0.6212 | 0.6212 | 0.6169 | 0.6169 |
| R1 collaborative | 0.6180 | 0.6262 | 0.6286 | 0.6369 |
| R2 + content | 0.6216 | 0.6345 | 0.6342 | **0.6446** |
| R3 + LLM labels | 0.6230 | 0.6386 | **0.6364** | 0.6432 |
| R4 + LLM rankings (all) | **0.6408** | **0.6447** | 0.6355 | 0.6430 |
| R5 collaborative + LLM rankings | 0.6282 | 0.6334 | 0.6295 | 0.6378 |

All differences are small, at most about 0.016 NDCG on test, compared with the +0.040 that idea 3 of experiment 13 gained over popularity.

### 3.2 Pre-registered tests (814 test users)

| Library | Comparison | Difference | Better / worse users | p | p (Bonferroni, ×2) |
|---|---|---|---|---|---|
| **LightGBM** (pre-registered) | R4 vs R1 | +0.0070 | 399 / 362 | 0.064 | 0.13 |
| **LightGBM** (pre-registered) | R4 vs Hybrid E | +0.0044 | 393 / 386 | 0.61 | 1.0 |
| XGBoost (robustness) | R4 vs R1 | +0.0061 | 400 / 348 | 0.039 | 0.078 |
| XGBoost (robustness) | R4 vs Hybrid E | **+0.0119** | 404 / 357 | 0.019 | **0.039** |

- **The pre-registered LightGBM hypotheses are not confirmed.** R4 is not significantly better than R1 or Hybrid E.
- **XGBoost R4 beats Hybrid E** (+0.012, corrected p = 0.039), the only significant result. It's a secondary check, not the pre-registered model, so it is evidence that a learned ranker *can* beat Hybrid E rather than a confirmed finding.

### 3.3 Where the gain comes from (test, descriptive)

| Comparison | What is added | LightGBM | XGBoost |
|---|---|---|---|
| R1 vs Hybrid E | learning instead of a fixed blend | −0.0025 (p = 0.38) | +0.0058 (p = 0.53) |
| R2 vs R1 | non-LLM content | +0.0057 (p = 0.16) | **+0.0076 (p = 0.019)** |
| R3 vs R2 | LLM labels | +0.0022 (p = 0.33) | −0.0014 (p = 0.80) |
| R4 vs R3 | LLM rankings | −0.0009 (p = 0.96) | −0.0002 (p = 0.61) |
| **R4 vs R2** | **all LLM signals** | **+0.0013 (p = 0.34)** | **−0.0016 (p = 0.86)** |
| R5 vs R1 | LLM rankings without content | +0.0009 (p = 0.59) | +0.0009 (p = 0.62) |
| R2 vs Hybrid E | collaborative + content, learned | +0.0031 (p = 0.93) | **+0.0134 (p = 0.009)** |
| R4 vs DeepSeek alone | | +0.0187 (p = 0.0004) | +0.0261 (p < 0.0001) |

- **The gain over Hybrid E comes from content (G2), not from the LLM.** With XGBoost, R2 alone already beats Hybrid E by +0.013, and adding all LLM features afterwards changes nothing (R4 vs R2: −0.002).
- **The LLM rankings add nothing in either combination.** That holds on top of content (R4 vs R3) and without content (R5 vs R1).
- **The combination beats the LLM alone** by +0.019 to +0.026: the collaborative signal is what DeepSeek's idea-1 ranking lacked.

### 3.4 Validation vs test

On validation, R4 looked clearly best (LightGBM: R4 vs R1 +0.023, p = 0.010; R4 vs R3 +0.018, p = 0.009, both above 0.05 after Bonferroni over 7 tests). On test, R4 ≈ R3 ≈ R2. With 203 users and 7 comparisons, the validation advantage of R4 was noise, or overfit by chance to the validation folds. This is the reason for the one-shot pre-registered test on 814 users. Without it, the validation numbers would have suggested that LLM rankings help.

---

## 4. Answer to the question

**No: the LLM signals add no measurable information to a learned ranker.** Neither the DeepSeek v5 labels nor the DeepSeek and gemma rankings improve on collaborative + content features, with either library (R4 vs R2: +0.001 / −0.002). What *does* help is learning the combination. XGBoost with collaborative and plain content features (R2) beats the fixed Hybrid E blend, and that model needs no LLM at all.

---

## 5. Feature importance of the final R4 models

The ablation shows the LLM features add nothing. Feature importance shows **how the models use them**. The final R4 models are retrained on all 203 validation users exactly as in `final_test`. Their test NDCG reproduces section 3 (0.6355 and 0.6430), so they are the same models.

### 5.1 Three measures

| Measure | What it says | Data |
|---|---|---|
| **Gain** | how much the training objective improved through splits on the feature | validation (training) |
| **Mean \|SHAP\|** | average size of the feature's contribution to a candidate's score (TreeSHAP, built into both libraries: `pred_contrib` / `pred_contribs`). **Centred within each user**, because a ranking depends only on differences between one user's candidates. | test users |
| **Permutation drop** | loss of test NDCG@10 when the feature, or a whole group, is shuffled **within each user** (5 repeats, averaged) | test users |

Shuffling a whole group at once matters because many features are near duplicates (section 5.4), and shuffling one of a pair barely hurts when its twin is still present.

### 5.2 Per group

| Group | Gain share (LGBM / XGB) | SHAP share (LGBM / XGB) | NDCG drop if group shuffled (LGBM / XGB) |
|---|---|---|---|
| G1 collaborative | 48% / 50% | 44% / 46% | 0.045 / 0.045 |
| G2 content | 20% / 16% | 18% / 17% | 0.009 / 0.009 |
| G3 LLM labels | 19% / 19% | 22% / 21% | 0.006 / 0.003 |
| G4 LLM rankings | 13% / 15% | 15% / 16% | **0.019 / 0.014** |

### 5.3 Top features (SHAP share)

| Rank | LightGBM | XGBoost |
|---|---|---|
| 1 | `hybrid_rank` 10.6% | `hybrid_rank` 15.5% |
| 2 | **`rank_deepseek` 9.9%** | **`rank_deepseek` 10.4%** |
| 3 | `emb_v5_max` 9.8% | `emb_v5_max` 10.2% |
| 4 | `emb_max` 7.8% | `emb_max` 7.8% |
| 5 | `vote_average` 7.4% | `vote_average` 7.4% |
| 6 | `cf_mean` 6.4% | `cf_mean` 6.1% |
| 7 | `emb_mean` 5.7% | `rank_gemma` 5.7% |
| 8 | `rank_gemma` 5.3% | `emb_mean` 5.7% |

The profile features (`theme_overlap`, `mood_overlap`, `director_match`, `actor_overlap`, `decade_gap`) carry 0.4–3.3% each and have permutation drops near 0. Full tables: `importance_features_{lightgbm,xgboost}.csv`.

**Direction:** the mean centred SHAP of `rank_deepseek` by DeepSeek's own rank shows whether the model trusts the LLM's choices.

| DeepSeek rank | LightGBM | XGBoost |
|---|---|---|
| 1–10 | +0.138 | +0.117 |
| 11–20 | +0.049 | +0.019 |
| 21–30 | −0.187 | −0.136 |

The models push the LLM's favourites up and its rejects down, a sensible use. Gemma's ranking has the same pattern at about a third of the strength.

### 5.4 Why the LLM features are used heavily yet add nothing: redundancy

The result looks contradictory at first:
- **the LLM features get about 37% of the SHAP importance,** and `rank_deepseek` is the second most important feature;
- **shuffling G4 costs more NDCG than shuffling content** (0.019 vs 0.009 with LightGBM);
- **yet removing them costs nothing** (R4 vs R2: +0.001 / −0.002).

The two kinds of measures answer different questions:
- **Feature importance** measures what *this trained model* leans on. Shuffling `rank_deepseek` hurts because the model expects it and is fed wrong values.
- **The ablation** retrains the model *without* the feature. If the same information is available elsewhere, the retrained model takes it from there, and nothing is lost.

The LLM features carry information the classical features already have. Measured within each user's 30 candidates on the test users:

| Pair | Within-user Spearman |
|---|---|
| `rank_deepseek` vs `emb_mean` (plain-text embedding) | −0.55 |
| `rank_deepseek` vs `emb_max` | −0.49 |
| `rank_deepseek` vs `cf_mean` (item-CF) | −0.47 |
| `rank_deepseek` vs `hybrid_rank` | +0.37 |
| `rank_deepseek` vs `rank_gemma` | +0.56 |
| **`emb_v5_max` vs `emb_max`** (LLM-enriched vs plain text) | **+0.98** |

A low rank means a good film, so the negative signs mean agreement. A linear regression on the 10 non-LLM features from G1 and G2 alone explains **41%** of DeepSeek's ranking within users; the trees can capture more.

- **G3 is almost entirely a duplicate of G2.** `emb_v5_max` and `emb_max` correlate at 0.98, so the model splits the credit between them more or less arbitrarily. The same happened in experiments 05 and 14, where the LLM-enriched embeddings ranked like the plain ones.
- **G4 largely re-expresses similarity to the seeds.** DeepSeek ranks a candidate high when it resembles the seeds in content and when it is collaboratively close. The part of its ranking that is *not* explained by those features doesn't predict the answer key. Otherwise R4 would beat R2.

---

## 6. Conclusions

1. **A learned ranker can beat Hybrid E, without an LLM.** XGBoost on collaborative and plain content features (R2) beats Hybrid E on test by +0.013 (p = 0.009, descriptive); the full R4 does by +0.012 (Bonferroni p = 0.039, robustness check). LightGBM's gains are smaller and not significant, so the size of the gain depends on the library.
2. **The LLM signals add no measurable information.** That covers the DeepSeek v5 labels, the profile overlaps and the DeepSeek and gemma rankings: R4 vs R2 is +0.001 / −0.002, and R5 vs R1 is +0.001 for both libraries. This agrees with experiments 05, 13 and 14: LLM signals help when the alternative is popularity alone (idea 3), not when collaborative and content similarity are already available.
3. **The models do use the LLM features, as substitutes.** `rank_deepseek` is the second most important feature and is used in the sensible direction. But it is correlated with embedding and item-CF similarity, and `emb_v5_max` is nearly identical to `emb_max`. Retraining without them recovers the same accuracy. Feature importance and ablation together show **redundancy**, not uselessness of the signal itself.
4. **Validation results on 203 users were misleading.** R4's apparent advantage on validation (+0.023 over R1 with LightGBM) did not hold on 814 test users. The pre-registered one-shot test prevented a false positive.
5. **Practical consequence:** the LLM rankings cost one model call per user, plus latency. Since they bring no accuracy gain over free features, a production ranker would leave them out.

---

## 6a. Correction: non-deterministic label features (found 2026-10-25)

**The bug:**
- `theme_overlap` and `mood_overlap` depend on the user's top seed themes and moods, chosen by `lr.keep_repeated`.
- When several themes are tied, `keep_repeated` keeps them in insertion order, and that order came from looping over a Python `set` of strings.
- Python randomises string hashing per process, so the set order, and with it the chosen themes, changed from one run of the script to the next.

**The fix:** `candidate_rows` now loops over the themes and moods in sorted order. Two separate runs were checked to give identical features. All feature tables were rebuilt; the old ones are kept in `old_unsorted_features/`.

**Effect on this document:**
- **Unaffected:** R1, R2 and R5, Hybrid E and "DeepSeek alone"; they don't use these two features.
- **Affected:** R3, R4 and the feature importance of R4 (section 5). Their numbers come from one run before the fix.
- **How much a rerun moved them:** a second run (also before the fix) gave these values on the test users:

| | This document | Second run |
|---|---|---|
| LightGBM R3 / R4 | 0.6364 / 0.6355 | 0.6367 / 0.6361 |
| XGBoost R3 / R4 | 0.6432 / 0.6430 | 0.6410 / 0.6423 |
| LightGBM R4 vs R1 (p Bonf.) | +0.0070 (0.13) | +0.0075 (0.24) |
| LightGBM R4 vs Hybrid E (p Bonf.) | +0.0044 (1.0) | +0.0049 (1.0) |
| XGBoost R4 vs R1 (p Bonf.) | +0.0061 (0.078) | +0.0054 (0.17) |
| XGBoost R4 vs Hybrid E (p Bonf.) | +0.0119 (0.039) | +0.0112 (0.033) |

**Conclusion:** the variation is about ±0.002 NDCG, and no conclusion changes:
- the pre-registered LightGBM tests remain non-significant;
- XGBoost R4 still beats Hybrid E after correction;
- the LLM features still add nothing beyond R2.

The doc-15 pipeline was not rerun with the fixed features; LTR v2 (doc 16) uses only fixed features.

---

## 7. Limitations

- **One setting:** N = 10 seeds and Hybrid E's top 30. With fewer seeds (N = 3), the collaborative signal is weaker and the LLM features might matter more. LTR v2 (doc 16) later tested this: with 3 seeds, the labels showed a borderline effect only without co-visitation features.
- **Fixed, untuned hyperparameters.** Tuning might change the size of the gains, but it would need a separate validation split. LTR v2 (doc 16) tuned them on simulated users; the effect was about ±0.01.
- **Non-deterministic label features** in the reported R3/R4 runs (section 6a).
- **Only two LLM rankers** (DeepSeek V4.1 Flash and gemma4:e4b, prompt p2, shuffle 0 only). Each carries the position bias from experiment 13; averaging shuffles there didn't improve the ranking.
- **The XGBoost result is a robustness check, not the pre-registered model.** Both libraries were run under the identical procedure and no XGBoost setting was chosen after seeing test results.
- **Permutation importance includes the cost of feeding the model wrong values,** not just the loss of information, so it overstates what the feature *adds*. That's the reason for reporting it alongside the ablation.
- **The candidates are fixed.** A reranker can't recover relevant films outside Hybrid E's top 30 (oracle@30 ≈ 0.95).

---

## 8. Files

| File | Contents |
|---|---|
| `features_val.csv`, `features_test.csv` | the feature tables (30 rows per user) |
| `cv_per_user_val_{lightgbm,xgboost}.csv`, `cv_tests_val_{...}.csv` | validation cross-validation, per user and tests |
| `test_per_user_{lightgbm,xgboost}.csv`, `test_tests_{...}.csv` | test results, per user and pre-registered tests |
| `importance_features_{lightgbm,xgboost}.csv` | gain, SHAP and permutation importance per feature |
| `importance_groups_{lightgbm,xgboost}.csv` | the same per group (permutation: whole group shuffled) |
| `importance_llm_direction_{lightgbm,xgboost}.csv` | mean centred SHAP of the LLM rank features by DeepSeek's rank |

**Reproduce:**
```bash
python LightGBM_XBoost_rerank_experiment.py              # features, ablation, test (both libraries)
python LightGBM_XBoost_rerank_experiment.py importance   # feature importance of the final R4 models
```
