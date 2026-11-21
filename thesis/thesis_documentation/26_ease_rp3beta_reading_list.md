# 26 — EASE and RP3beta: papers, formulas, code

Status (2026-10-28): **EASE implemented and evaluated on validation (section 7)**; RP3beta not implemented. Every reference was checked on arXiv,
the ACM listing or the repository itself. Formulas were checked against the EASE paper and the
reference code of Ferrari Dacrema et al.

## 1. Why these two

- Independent comparisons find simple item-to-item models at or near the top, ahead of neural models such
  as NCF: Ferrari Dacrema et al. 2019 (most neural methods beaten by nearest-neighbour or graph
  heuristics) and Anelli et al. 2022 (RP3beta and EASE at the top of the table we could see, NeuMF
  behind; check the per-dataset tables before citing).
- Both learn a **film × film** matrix and no user vectors. A new user's scores are just the sum of the
  rows of their seed films: no fold-in, nothing to fit per user. That suits our cold-start set-up
  better than SVD, NCF or iALS.
- Our item-CF (the base of Hybrid E) is the same kind of model with fixed cosine weights. EASE and
  RP3beta are drop-in alternatives for it.

## 2. EASE (Embarrassingly Shallow Autoencoder)

**Paper:** Harald Steck (Netflix), "Embarrassingly Shallow Autoencoders for Sparse Data", WWW 2019.
[arXiv 1905.03375](https://arxiv.org/abs/1905.03375) · DOI 10.1145/3308558.3313710

**Idea:** predict each item from all *other* items with one linear layer. X = users × items, 0/1.

- Objective (paper Eq. 2–3): minimise ‖X − XB‖²_F + λ‖B‖²_F subject to diag(B) = 0.
  The zero diagonal stops the trivial answer "each item predicts itself".
- Closed form (Eq. 4, 7, 8):
  - P = (XᵀX + λI)⁻¹
  - B_ij = −P_ij / P_jj for i ≠ j, and B_ii = 0
- Scores for a user: x · B (x = the user's 0/1 row). For us: x = the seed films.
- One hyperparameter, λ. Values reported in the paper: about 500 (ML-20M), 1,000 (Netflix), 200 (MSD),
  3,000 (ML-10M). Ours must be tuned; a smaller data set usually needs relatively different λ.
- Datasets in the paper: ML-20M, Netflix, Million Song Data (protocol of Liang et al. 2018), and ML-10M.
- **Official code:** the paper itself gives the full training routine (Algorithm 1, a few lines of
  numpy). This is the author's own code and the version to follow. No separate official repository
  from the author was found (searched 2026-10-28; the paper links only the baselines' code).
- Cost for us: one inversion of a ~1,549 × 1,549 matrix: well under a second.

**Third-party implementation (not official):** `EASE_R/EASE_R_Recommender.py` in
[MaurizioFD/RecSys2019_DeepLearning_Evaluation](https://github.com/MaurizioFD/RecSys2019_DeepLearning_Evaluation)
(AGPL-3.0). Its `fit()` builds the Gram matrix, adds `l2_norm` to the diagonal (default 1,000),
inverts, sets `B = P / (−diag(P))` and zeroes the diagonal. Optional `topK` sparsification (default none).

## 3. RP3beta (and P3alpha)

**P3alpha paper:** Cooper, Lee, Radzik, Siantos (King's College London), "Random Walks in Recommender
Systems: Exact Computation and Simulations", WWW 2014 Companion. DOI 10.1145/2567948.2579244 ·
[PDF](https://archives.iw3c2.org/www2014/proceedings/companion/p811.pdf). Experiments on MovieLens.
- P3: score items by a 3-step random walk on the user–item graph: user → item → user → item.
- P3alpha: raise the transition probabilities to a power α before multiplying.

**RP3beta papers:**
- Christoffel, Paudel, Newell, Bernstein, "Blockbusters and Wallflowers: Accurate, Diverse, and
  Scalable Recommendations with Random Walks", RecSys 2015. DOI 10.1145/2792838.2800180.
- Journal version: Paudel, Christoffel, Newell, Bernstein, "Updatable, Accurate, Diverse, and Scalable
  Recommendations for Interactive Applications", ACM TiiS 7, 2016. DOI 10.1145/2955101.
- Idea: P3alpha, then divide each item's score by its popularity to the power β. This pushes long-tail
  films up (the papers' main selling point: accuracy *and* diversity).

**How the similarity is computed** (third-party code, not from the RP3beta authors:
`GraphBased/RP3betaRecommender.py` in the same repository; defaults α = 1, β = 0.6, topK = 100;
no official RP3beta code was found, so check the steps against the TiiS paper before relying on them):
1. P_ui = user → item transition: each user's row of X divided by its sum.
2. P_iu = item → user transition: each item's column of X (as 0/1) divided by its number of users.
3. If α ≠ 1: raise every non-zero entry of both to the power α (element-wise, no renormalisation).
4. W = P_iu · P_ui (item → user → item); for a user, the third step is their own row.
5. Popularity penalty: multiply column j by popularity_j^(−β) (popularity = number of users of item j).
6. Zero the diagonal, keep the topK largest per row, L1-normalise each row.
- Scores for a user: x · W.
- P3alpha is the same without step 5 (β = 0).
- Hyperparameters to tune: α, β, topK.

## 4. Benchmarks and context (for the related-work chapter)

- Ferrari Dacrema, Cremonesi, Jannach, "Are We Really Making Much Progress? A Worrying Analysis of Recent
  Neural Recommendation Approaches", RecSys 2019. [arXiv 1907.06902](https://arxiv.org/abs/1907.06902)
  · code: the repository above. 18 neural methods; 7 reproducible; 6 of those often beaten by simple
  nearest-neighbour or graph-based heuristics (RP3beta is one of their baselines).
- Anelli, Bellogín, Di Noia, Jannach, Pomo, "Top-N Recommendation Algorithms: A Quest for the
  State-of-the-Art", UMAP 2022. [arXiv 2203.01155](https://arxiv.org/abs/2203.01155). Ten algorithms,
  three datasets, reproducible.
- Rendle, Krichene, Zhang, Anderson, "Neural Collaborative Filtering vs. Matrix Factorization
  Revisited", RecSys 2020. [arXiv 2005.09683](https://arxiv.org/abs/2005.09683): a tuned dot product
  beats NCF's learned MLP similarity.
- Rendle, Krichene, Zhang, Koren, "Revisiting the Performance of iALS on Item Recommendation
  Benchmarks", RecSys 2022. [arXiv 2110.14037](https://arxiv.org/abs/2110.14037): tuned iALS is
  competitive. (Needs per-user fold-in like SVD, so lower priority for us.)

## 5. Code to read

| Where | What | Notes |
|---|---|---|
| EASE paper, Algorithm 1 | complete EASE training | **official** (by the author, in the paper); start here |
| [MaurizioFD/RecSys2019_DeepLearning_Evaluation](https://github.com/MaurizioFD/RecSys2019_DeepLearning_Evaluation) | `EASE_R/`, `GraphBased/` (P3alpha, RP3beta), `KNN/` (item/user-kNN), `SLIM_ElasticNet/` | **third-party**, written by the benchmark authors (Ferrari Dacrema et al.) for their reproducibility study, not by the EASE or RP3beta authors; AGPL-3.0, so read and reimplement rather than copy into the thesis code |

## 6. How it would fit our set-up (plan, not done)

- **Training data:** the warm users' **liked** films (rating ≥ the user's own mean), as 0/1, on the
  1,549-film candidate pool, the same input item-CF gets. For LTR features: the cross-fitted halves
  (half A scored with a model trained on half B), exactly as for item-CF.
- **Scoring a new user:** x = their seeds (0/1), score = x · B (EASE) or x · W (RP3beta); seeds and
  seen films removed.
- **Tuning:** λ (EASE) and α, β, topK (RP3beta) on the simulated warm users, never on the 203
  validation users; then one evaluation on validation with 3 and 10 seeds, in the final-comparison
  table.
- **Uses:**
  1. as recommenders on their own (vs item-CF, user-kNN, Hybrid E);
  2. in place of item-CF inside Hybrid E;
  3. as extra features in LTR v2 (decision rule as in doc 24: +0.005 on both user groups).
- **Expectation:** unknown. User-kNN (0.642) and LTR v2 (0.669) are already strong; EASE beating item-CF
  (0.599) would not be surprising given the literature, beating LTR v2 would be.
- **Cost:** CPU, minutes, no LLM calls.

## 7. EASE results (2026-10-28)

**Implementation:** `EASE_R_Recommende` in recomm.py (written by the author, following Algorithm 1 of the
paper): X = users × films, 1 = liked (rating ≥ the user's own mean); G = XᵀX + λI; P = G⁻¹;
B = −P / diag(P), diagonal 0. A new user's scores = sum of B's rows for the seed films; seeds and all
10 seed-pool films are never recommended (same rule as every other method). Sanity check (Toy Story →
Toy Story 2 and 3, Pixar/Disney; Godfather + Pulp Fiction + Shawshank → Godfather II, Reservoir Dogs,
GoodFellas) passed.

**λ tuning** (`ease_tuning.py`, `output/experiments/ease/tuning.csv`): the 9,084 simulated warm users
(halves A/B, seed 42, the LTR v2 users), cross-fitted (half A scored by an EASE fitted on half B and the
other way round). Validation users not used. Item-CF scored with the same code on the same users as a
reference.

| | 10 seeds | 3 seeds |
|---|---|---|
| Item-CF | 0.614 | 0.529 |
| EASE, λ = 500 (paper's ML-20M value) | 0.606 | 0.510 |
| EASE, best λ | **0.650** (λ = 20,000) | **0.609** (λ = 120,000) |

- Both optima are flat (10 seeds: 15,000–30,000 within 0.001; 3 seeds: 80,000–120,000 within 0.001);
  halves A and B agree. Grid 500–200,000.
- The best λ is much larger than in the paper, and larger with fewer seeds. A large λ moves B towards
  scaled co-occurrence counts, i.e. more weight on popular films; with 3 seeds there is less user signal.
  (Interpretation, not tested separately.)
- A first 300-user pilot suggested a peak near 5,000–10,000; the full set did not confirm it.

**Validation** (`final_comparison.py`, 203 users, fitted once on all warm users, NDCG@10; p values paired
Wilcoxon, Bonferroni within each table as printed by the script):

| | 10 seeds | 3 seeds |
|---|---|---|
| LTR v2 | 0.6686 | 0.6227 |
| **EASE** | **0.6514** | **0.6055** |
| User-kNN | 0.6417 | 0.6053 |
| Hybrid E | 0.6286 | 0.5837 |
| Item-CF | 0.5990 | 0.5132 |
| EASE vs Hybrid E | +0.023 (p_Bonf 1.0) | +0.022 (p_Bonf 0.013) |

Extra paired Wilcoxon tests (uncorrected, run once after the table):
- EASE vs item-CF: +0.052 (10 seeds) and +0.092 (3 seeds), both p < 0.0001.
- EASE vs user-kNN: +0.010 (p = 0.47) and +0.000 (p = 0.96): no difference.
- EASE vs LTR v2: −0.017 (p = 0.19) and −0.017 (p = 0.11): lower, not significant.

Adding EASE to the table did not change any other method's NDCG, and no other method's significance
verdict changed (one more test per table in the Bonferroni factor).

**Safe statements:**
- "EASE, a one-parameter linear item-to-item model, clearly outperformed item-based CF in cold start
  (+0.05 NDCG@10 with 10 seeds, +0.09 with 3)."
- "It was the strongest single model without a learned ranker, on par with user-kNN, and second only to
  the LTR v2 ensemble (difference not significant on 203 users)."
- Do **not** say EASE beats Hybrid E with 10 seeds (not significant after correction), or that LTR v2
  beats EASE (not significant).

**Open:** EASE inside Hybrid E in place of item-CF; EASE score as an LTR v2 feature (cross-fitted, same
decision rule as doc 24); RP3beta; test users only in the pre-registered final test.

## 8. EASE as an LTR v2 feature (2026-10-28) — not adopted

**Script:** `ltr_ease_experiment.py`; outputs in `output/experiments/ltr_ease/`. The L1 harness of doc 20
§10.1 / doc 24, unchanged: new group G7 = `ease_score` (sum of B's rows for the seeds) and `ease_rank`
(rank among the 30 candidates); R9 = R7 + G7; tuned XGBoost + LightGBM (step 4 parameters, not retuned)
and their rank ensemble.

- **Cross-fitting:** simulated users of half A get EASE fitted on half B and the other way round;
  validation users get EASE fitted on all warm users.
- **λ (fixed before running):** the tuned value is a half-size λ (section 7 tuned on half models), so
  the full model uses λ × warm users / half users ≈ 2× (40,000 and 240,000). Check: `ease_score` has
  the same distribution on validation and simulated users (10 seeds mean 0.059 vs 0.061).
- **Decision rule (doc 24):** adopt only if R9 gains ≥ +0.005 on both user groups. R7 reproduced
  LTR v2 exactly (0.6686, 0.6227).

| NDCG@10, ensemble | 10 seeds held-out (1,816) | 10 seeds validation (203) | 3 seeds held-out | 3 seeds validation |
|---|---|---|---|---|
| R7 (LTR v2) | 0.6759 | 0.6686 | 0.6265 | 0.6227 |
| R9 = R7 + EASE | 0.6763 | 0.6722 | 0.6277 | 0.6262 |
| Gain (p, Wilcoxon) | +0.0003 (0.94) | +0.0036 (0.39) | +0.0012 (0.46) | +0.0035 (0.43) |
| *EASE alone, reranking the same 30* | 0.6508 | 0.6504 | 0.6077 | 0.6075 |
| *Hybrid E* | 0.6283 | 0.6286 | 0.5880 | 0.5837 |

**Decision:** not adopted with 3 or 10 seeds; LTR v2 stays the final model.

**Findings:**
- EASE adds almost nothing to the ranker. Within the 30 candidates its order correlates 0.72–0.80
  (Spearman) with `hybrid_score`, and with 3 seeds 0.80–0.83 with co-visitation. The trees already
  have most of this signal.
- On its own, EASE is a much better ranker of the 30 candidates than Hybrid E (+0.022 / +0.024). It
  is also about as good as a recommender with its own candidates (0.651 / 0.606, section 7). Still,
  0.017–0.018 below LTR v2.
- Same pattern as NCF and user-kNN (doc 20 §10): a strong stand-alone model is not automatically a
  useful extra feature.

**Caveat for section 7's validation numbers:** `final_comparison.py` fits EASE on all warm users with
the half-size λ (20,000 / 120,000), not the scaled λ used here. This was not changed afterwards, to avoid
choosing on the validation users. Scaling λ is the principled choice for the final test; decide before
running it.

**Safe statement:** "Adding the EASE score as a feature to the learned ranker did not meet the
pre-set threshold (+0.0003 / +0.0036 with 10 seeds, +0.0012 / +0.0035 with 3; none significant). EASE is
a strong stand-alone model but largely repeats the collaborative signal the ranker already has."
