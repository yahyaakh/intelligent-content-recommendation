# Experiment: Learning-to-Rank v2 (Simulated Training Users, Co-visitation, Ensemble)

**Status:** finished (run date: 2026-10-25); final test pre-registered and run once
**Script:** `LightGBM_XBoost_rerank_experiment.py`, modes `simulated` (step 1), `g5` (step 2), `candidates` (step 3), `tuning` (step 4), `seeds3` (step 5), `final` (step 6)
**Results folder:** `output/experiments_ LightGBM_XBOOST/ltr_boosting/` (files `step1_*` … `step6_*`, `features_*`)
**Pre-registration:** `output/experiments_ LightGBM_XBOOST/ltr_boosting/step6_preregistration.md`
**Related documents:** `06_experiment_cold_start.md` (Hybrid E, the cold-start split), `13_experiment_llm_as_recommender.md` (LLM reranking), `14_experiment_label_quality_and_vocabulary_v5.md` (DeepSeek v5 labels), `15_experiment_learning_to_rank.md` (LTR v1)

---

## 1. Question

In experiment 15 (LTR v1), a learned ranker trained on the 203 validation users beat Hybrid E only slightly: about +0.012 on the test users with XGBoost, and not significantly with the pre-registered LightGBM model. The LLM features added nothing. Its validation results were also noisy, and the gain shrank on the test users. Training on 203 users (6,090 rows) is very little data for a boosted tree model.

This experiment asks:

> **How far can a learned ranker get beyond Hybrid E for new users, and what drives the gain: more training data, better collaborative features, wider candidate lists, tuning, or the LLM signals?**

The work was done in six steps. Steps 1–5 used only simulated training users and the 203 validation users. Step 6 was a single pre-registered test on the 814 test users.

---

## 2. Method

### 2.1 Simulated new users from the warm users (step 1)

- **The idea:** about 9,600 warm users have full rating histories. Each one can be treated like a new user:
  - **seeds:** the first 10 liked films (in time order, rating ≥ the user's own mean);
  - **answer key:** all their other liked films;
  - **the same rules as for the cold users:** `cs.build_cold_user_data`, a seed pool of 10, at least 5 relevant films left.
- **The size:** this gives **9,084 training users** (272,520 rows), compared with 203 users (6,090 rows) in experiment 15.
- **The leakage problem:** Hybrid E's candidates and the collaborative features come from a CF table and popularity built from the warm users' ratings. A warm user's own ratings would then be inside the table that scores them.
- **Cross-fitting:**
  - the warm users are split at random (seed 42) into halves **A** (4,540 simulated users) and **B** (4,544);
  - the users of half A get their CF table, popularity, SVD and co-visitation from **half B's ratings only**, and vice versa;
  - no user's own ratings are ever in a table that scores them;
  - validation and test users get their features from the **full** warm tables, as before.
- **The check:** Hybrid E on the simulated users with half tables scores 0.6282, almost identical to the validation users (0.6286). Training and evaluation users therefore look alike.
- **LLM rankings (G4) are not used:** the simulated users have no cached LLM answers, and experiment 15 showed G4 to be redundant. The DeepSeek v5 labels (G3) are available for every user and stay in.

### 2.2 New collaborative features, G5 (step 2)

All computed from the ratings of the relevant table (half or full warm users):

| Feature | Definition |
|---|---|
| `svd_mean`, `svd_max` | cosine similarity between the candidate's and the seeds' SVD film vectors (rank 64, user-centred rating matrix), mean and max over the seeds |
| `svd_score` | PureSVD fold-in: the score of a new user who liked the seeds, Σ over seeds of (U Uᵀ)[seed, candidate] |
| `covis_mean`, `covis_max` | P(candidate liked \| seed liked): users who liked both / users who liked the seed, mean and max over the seeds |
| `cf_recent` | item-CF similarity to the **last 3** seeds only (taste drift); with 3 seeds it equals `cf_mean` |

SciPy's sparse SVD (`svds`, ARPACK) crashes in this environment with a Windows DLL error (`0xc06d007f`). The truncated SVD is therefore computed exactly from the eigenvectors of the 1,957 × 1,957 matrix M Mᵀ (NumPy), which also makes it deterministic.

### 2.3 Feature sets

The groups G1–G4 are those of experiment 15. New models:

| Model | Groups | Features |
|---|---|---|
| R1 | G1 collaborative | 8 |
| R2 | G1 + G2 content | 11 |
| R3 | G1 + G2 + G3 LLM labels | 19 |
| **R6** | G1 + G2 + G5 | 17 |
| **R7** | G1 + G2 + G3 + G5 | 25 |

### 2.4 Candidates, settings, ensemble (steps 3 and 4)

- **Candidate lists (step 3):**
  - **C30:** Hybrid E's top 30, as in experiments 13 and 15;
  - **C50:** Hybrid E's top 50;
  - **C30+20:** the top 30 plus the 20 films most similar in content (embedding `emb_max`) among the rest.
  - Every candidate keeps its rank in Hybrid E's full ranking as `hybrid_rank`.
- **Tuning (step 4):**
  - a grid of 24 settings per library, fixed in advance:
    - XGBoost: depth {3, 4, 6} × trees {200, 500} × learning rate {0.05, 0.1} × min_child_weight {1, 10};
    - LightGBM: leaves {15, 31, 63} × trees {200, 500} × learning rate {0.05, 0.1} × min_child_samples {20, 100};
  - each setting is trained on 80% of the simulated users and scored on the other 20% (1,816 users), so the validation users are not used for tuning.
- **Ensemble:** the 50/50 average of the within-user percentile ranks of the tuned XGBoost and tuned LightGBM models. The weights are fixed, so there is nothing more to tune.

### 2.5 Evaluation

- **Steps 1–5:** the 203 validation users, NDCG@10, paired Wilcoxon tests, Bonferroni-corrected within each step. These steps are exploratory.
- **Step 6:** the pre-registered test on the 814 test users (section 4).
- **All settings are deterministic** (fixed seeds, `deterministic=True`, one thread), so every rerun gives identical numbers.

### 2.6 Bug found and fixed during step 1

- **The symptom:** `theme_overlap` and `mood_overlap` (G3) changed from one run to the next.
- **The cause:** the user's top seed themes are chosen by `keep_repeated`, which breaks ties by insertion order. That order came from looping over a Python `set` of strings, and Python's string hash is randomised per process, so the order differed in every run.
- **The fix:** the loops now go through the themes in sorted order. Two separate runs were checked to give identical features.
- **The rebuild:** all feature tables (validation, test, simulated) were rebuilt with the fix before steps 1–6 were reported; the old tables are kept in `old_unsorted_features/`.
- **Scope:** only G3, and therefore R3, R4 and R7, are affected. R1 and R2 were always exact. See the note in experiment 15.

---

## 3. Results on the validation users (exploratory)

### 3.1 Step 1: more training users

| Model | Trained on 203 users (5-fold CV, doc 15) | **Trained on 9,084 simulated users** | Gain (p Bonf.) | vs Hybrid E 0.6286 (p Bonf.) |
|---|---|---|---|---|
| XGBoost R1 | 0.6262 | 0.6435 | +0.017 (0.027) | +0.015 (0.045) |
| XGBoost R2 | 0.6345 | 0.6532 | +0.019 (0.026) | +0.025 (0.017) |
| XGBoost R3 | 0.6432* | 0.6536 | +0.010 (0.33) | +0.025 (0.014) |
| LightGBM R1 | 0.6180 | 0.6485 | +0.031 (0.006) | +0.020 (0.018) |
| LightGBM R2 | 0.6216 | 0.6544 | +0.033 (0.004) | +0.026 (0.017) |
| LightGBM R3 | 0.6244* | 0.6571 | +0.033 (0.001) | +0.029 (0.008) |

\* From the doc-15 CV files of a run before the fix in §2.6.

- **Training data was the bottleneck.** Every model gains about +0.02 to +0.03, LightGBM the most: it was the more data-hungry library and lost to XGBoost in experiment 15 for that reason.
- **The learned ranker now beats Hybrid E with Hybrid E's own signals** (R1, +0.015 to +0.020): the trees combine item-CF, popularity and TMDB metadata better than the fixed blend.
- **Content adds about +0.006 to +0.01** (R2 vs R1). **The LLM labels add +0.000 to +0.003** (R3 vs R2).

### 3.2 Step 2: G5 features

| Model | XGBoost | LightGBM |
|---|---|---|
| R2 | 0.6532 | 0.6544 |
| R3 | 0.6536 | 0.6571 |
| R6 = R2 + G5 | 0.6645 | 0.6603 |
| R7 = R3 + G5 | **0.6670** | 0.6583 |

| Test (Bonferroni over 4) | XGBoost | LightGBM |
|---|---|---|
| R6 vs R2 | +0.011 (p = 0.53) | +0.006 (p = 1.0) |
| R7 vs R3 | +0.013 (p = 0.34) | +0.001 (p = 1.0) |
| R7 vs Hybrid E | **+0.039 (p = 0.001)** | +0.030 (p = 0.014) |

**Each G5 feature added alone to R2 (descriptive):**

| Feature | XGBoost | LightGBM |
|---|---|---|
| `covis_max` | **+0.010** | **+0.010** |
| `covis_mean` | +0.006 | −0.006 |
| `svd_mean` | +0.001 | +0.004 |
| `svd_max` | −0.003 | +0.001 |
| `svd_score` | −0.003 | +0.001 |
| `cf_recent` | −0.002 | +0.001 |

- **Co-visitation is the useful new signal.** `covis_max` adds about +0.010 with both libraries. It measures how many of the users who liked a seed also liked the candidate, i.e. "people who liked X also liked Y".
- **SVD mostly repeats item-CF.**
- **The G5 gain over R2/R3 is not significant on 203 users,** but it raises the best model to +0.039 over Hybrid E.
- **The G5 feature means are almost identical** for the validation and simulated users (e.g. `covis_max` 0.675 vs 0.677), so cross-fitting causes no shift.

### 3.3 Step 3: wider candidate lists

| List | Candidates | Oracle | XGB R6 | XGB R7 | LGBM R6 | LGBM R7 |
|---|---|---|---|---|---|---|
| **C30** | 30 | 0.953 | 0.6645 | 0.6670 | 0.6603 | 0.6583 |
| C50 | 50 | 0.984 | 0.6635 | 0.6717 | 0.6606 | 0.6633 |
| C30+20 | 50 | 0.977 | 0.6712 | 0.6707 | 0.6672 | 0.6663 |

- **No difference is significant** (all p_Bonf = 1.0).
- **C30+20 is +0.004 to +0.008 on average, but helps fewer users than it hurts** (e.g. 70 better vs 87 worse). Its mean comes from a few users with large gains.
- **Hybrid E's top 30 already contains almost everything** a perfect reranker needs (oracle 0.953), and the models do not use the extra room.
- **C30 was kept:** it is the simplest list, and the same list as Hybrid E and the LLM rerankers used.
- A check confirmed that C30 reproduces the step-2 tables exactly (largest difference 5 × 10⁻⁷).

### 3.4 Step 4: tuning and ensemble (R7, C30)

| | Default setting | Best setting | Held-out NDCG (1,816 simulated users) |
|---|---|---|---|
| XGBoost | depth 4, 200 trees, lr 0.05, mcw 1 | depth 6, 500 trees, lr 0.05, mcw 10 | 0.6678 → 0.6762 |
| LightGBM | 15 leaves, 200 trees, lr 0.05, mcs 20 | 63 leaves, 200 trees, lr 0.05, mcs 20 | 0.6716 → 0.6755 |

The top of each grid is flat: the five best settings are within 0.002 of each other.

| Validation (203 users) | NDCG@10 | Test (Bonferroni over 4) |
|---|---|---|
| XGBoost default / tuned | 0.6670 / 0.6637 | −0.003 (p = 1.0) |
| LightGBM default / tuned | 0.6583 / 0.6693 | +0.011 (p = 0.12) |
| Ensemble of defaults / of tuned models | 0.6637 / **0.6686** | tuned ensemble vs tuned LightGBM −0.001 (p = 1.0); **vs Hybrid E +0.040 (p = 0.002)** |

Tuning and ensembling move the score by about ±0.01, within the noise of 203 users.

### 3.5 Step 5: three seeds

Hybrid E with 3 seeds: 0.5837, which reproduces docs 06 and 13.

| Model | XGBoost | LightGBM |
|---|---|---|
| R2 | 0.6061 | 0.6020 |
| R3 | 0.6079 | 0.6086 |
| R6 | 0.6220 | 0.6253 |
| R7 | 0.6183 | 0.6243 |
| R7 tuned | 0.6199 | 0.6199 |
| **Tuned ensemble, R7** | **0.6227** | |

| Test (Bonferroni over 5) | Difference | Better / worse | p (Bonf.) |
|---|---|---|---|
| Ensemble vs Hybrid E | **+0.039** | 120 / 63 | < 0.0001 |
| XGBoost R3 vs R2 | +0.002 | 79 / 71 | 1.0 |
| LightGBM R3 vs R2 | +0.007 | 99 / 62 | 0.070 |
| XGBoost R7 vs R6 | −0.004 | 44 / 65 | 0.16 |
| LightGBM R7 vs R6 | −0.001 | 71 / 87 | 1.0 |

- **The gain over Hybrid E is as large with 3 seeds as with 10.**
- **G5 matters more with few seeds:** R6 vs R2 is +0.016 / +0.023, about double the 10-seed gain.
- **The LLM labels show their only (borderline) effect here: without G5,** LightGBM R3 vs R2 is +0.007 (p = 0.014 uncorrected, 0.070 corrected). **With G5 the effect disappears** (R7 vs R6: −0.004 / −0.001). Co-visitation carries the same signal, and more.

---

## 4. Pre-registered final test (814 test users)

### 4.1 What was fixed in advance

The pre-registration (`step6_preregistration.md`, 2026-10-25) fixed everything before the test:
- **the final model,** chosen **by rule, not by validation score:** the tuned XGBoost + LightGBM ensemble on R7, C30, trained on all 9,084 simulated users with the parameters in `step4_chosen_params.json`. The ensemble was chosen because it does not depend on which library happens to win; in experiment 15, the winner flipped between validation and test.
- **three hypotheses,** Bonferroni over 3:
  - **H1 (10 seeds):** the final model beats Hybrid E;
  - **H2 (10 seeds):** it beats LTR v1, XGBoost R2 trained on the 203 validation users (doc 15's best test result, 0.6446);
  - **H3 (3 seeds):** it beats Hybrid E.
- **four sanity checks** that had to pass before any final-model score was computed;
- **reporting commitments:** all results are reported whatever their outcome.

The script runs the test only once: it writes `step6_tests_test.csv` last and refuses to run if the file exists.

**Sanity checks, all passed:**
- Hybrid E reproduced 0.6311 (10 seeds) and 0.5970 (3 seeds);
- LTR v1's XGBoost R2 reproduced 0.6446;
- the test tables matched `features_test.csv`, with 30 rows per user and no missing values;
- no test user was in the training data.

### 4.2 Results

| | Seeds | Final model | Baseline | Difference [95% bootstrap CI] | Better / worse users | p | p (Bonferroni) |
|---|---|---|---|---|---|---|---|
| **H1** vs Hybrid E | 10 | **0.6609** | 0.6311 | **+0.030** [+0.022, +0.038] | 454 / 312 | 6.6 × 10⁻¹⁰ | **2.0 × 10⁻⁹** |
| **H2** vs LTR v1 (XGBoost R2, 203 users) | 10 | **0.6609** | 0.6446 | **+0.016** [+0.008, +0.025] | 412 / 359 | 6.0 × 10⁻⁴ | **0.0018** |
| **H3** vs Hybrid E | 3 | **0.6257** | 0.5970 | **+0.029** [+0.020, +0.037] | 435 / 324 | 2.7 × 10⁻¹⁰ | **8.1 × 10⁻¹⁰** |

**All three hypotheses are confirmed.**

**Descriptive (not tested):**

| Seeds | Oracle@30 | Tuned XGBoost R7 | Tuned LightGBM R7 | Ensemble (final) |
|---|---|---|---|---|
| 10 | 0.9645 | 0.6564 | 0.6622 | 0.6609 |
| 3 | 0.9516 | 0.6262 | 0.6250 | 0.6257 |

### 4.3 Validation vs test

The final model scored 0.6686 on validation and 0.6609 on test (−0.008, 10 seeds); with 3 seeds it scored 0.6227 and 0.6257 (+0.003). That is much less shrinkage than in experiment 15, where R4's validation advantage disappeared on test. The model was chosen with 1,816 held-out simulated users and a fixed rule, not with the 203 validation users.

---

## 5. All test-user results in context (NDCG@10, 814 users)

| Method | 10 seeds | 3 seeds |
|---|---|---|
| **LTR v2 final model (this experiment)** | **0.6609** | **0.6257** |
| LTR v1: XGBoost R2 trained on 203 users (doc 15) | 0.6446 | — |
| Hybrid E (doc 06) | 0.6311 | 0.5970 |
| DeepSeek V4.1 Flash reranking Hybrid E (doc 13) | 0.6169 | — |
| DeepSeek V4.1 Flash reranking popularity (doc 13) | 0.5999 | — |
| Popularity | 0.5598 | 0.5598 |

With only 3 seeds, the final model (0.6257) almost reaches Hybrid E with 10 seeds (0.6311).

---

## 6. Conclusions

1. **A learned ranker clearly beats the best classical method for new users:** +0.030 NDCG@10 (+4.7%) with 10 seeds and +0.029 (+4.8%) with 3 seeds, pre-registered and confirmed on 814 test users.
2. **The main driver is training data.** Training on 9,084 simulated new users instead of 203 real ones gives +0.02 to +0.03 on its own. Cross-fitting makes this possible without leakage, and the simulated users behave like real new users (Hybrid E 0.6282 vs 0.6286).
3. **Co-visitation is the most useful new feature** ("users who liked a seed also liked the candidate"), especially with few seeds. SVD mostly repeats item-CF.
4. **Wider candidate lists, tuning and ensembling change little** (about ±0.01, none significant). Hybrid E's top 30 is a good candidate set, and the ranker is near the limit of its features.
5. **The LLM signals contribute nothing measurable,** in line with experiments 05, 13, 14 and 15.
   - The DeepSeek v5 labels are part of the final model (R7) only because the model was fixed by rule; every ablation shows them to add 0 to +0.003.
   - Their only borderline effect (3 seeds, without co-visitation: +0.007, corrected p = 0.07) disappears once co-visitation is included.
   - A lean model without LLM features (R6) would be expected to perform the same.
6. **LTR v2 improves on LTR v1** (+0.016, p = 0.0018). The improvement over experiment 15 comes from the training-data and feature changes, not from the LLM.

---

## 7. Limitations

- **Simulated users are warm users in disguise:** they rated more films in total, and their later history plays the role of the answer key. The cold users' Hybrid E score is nearly identical, which suggests they are comparable, but they are not real new users.
- **One dataset (MovieLens 25M, 1,957 films), one definition of "liked",** and offline evaluation that favours popular films (doc 06).
- **R7 includes the LLM labels by rule; R6 was not tested on the test users.** The claim that R6 performs the same rests on the validation ablations.
- **The test users were used once before** (experiment 15) to test the LTR v1 models. No LTR v2 decision used them.
- **The 3-seed setting reuses the 10-seed tuned parameters,** without retuning.
- **G4 (LLM rankings) was not tested in LTR v2,** because the simulated users have no LLM answers; about 9,000 DeepSeek calls (about $6) would be needed.
- **Commit timing:** the pre-registration file is dated 2026-10-25 and was written before the test run. It was not committed to git before the run.

---

## 8. Files

All in `output/experiments_ LightGBM_XBOOST/ltr_boosting/`:

| File | Contents |
|---|---|
| `features_simulated.csv`, `features_val.csv`, `features_test.csv` | G1–G4 feature tables (C30, 10 seeds), rebuilt after the fix; old ones in `old_unsorted_features/` |
| `features_{val,simulated,test}_g5.csv` | the same + G5 (10 seeds) |
| `features_{val,simulated}_{C30,C50,C30plus20}.csv` | step-3 candidate lists |
| `features_{val,simulated,test}_n3.csv` | 3-seed tables |
| `step1_*`, `step2_*`, `step3_*`, `step4_*`, `step5_*` | validation per-user results, tests, summaries (step 4 also `step4_grid_*.csv`, `step4_chosen_params.json`) |
| `step6_preregistration.md` | the pre-registration |
| `step6_tests_test.csv`, `step6_summary_test.csv`, `step6_per_user_test_n{10,3}.csv` | the final test |
| `step2_run.log`, `step3_run.log`, `step4_run.log` | run logs |

**Reproduce** (in this order; each later step uses earlier outputs):
```bash
python LightGBM_XBoost_rerank_experiment.py simulated
python LightGBM_XBoost_rerank_experiment.py g5
python LightGBM_XBoost_rerank_experiment.py candidates
python LightGBM_XBoost_rerank_experiment.py tuning
python LightGBM_XBoost_rerank_experiment.py seeds3
python LightGBM_XBoost_rerank_experiment.py final        # runs once
```
