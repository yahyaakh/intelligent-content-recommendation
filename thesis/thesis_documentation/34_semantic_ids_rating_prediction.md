# 34 — Semantic IDs for rating prediction (option 9 time split)

**Status:** complete (2026-10-29). **Pre-registered** before any model ran.
**Script:** `sid_rating_experiment.py` (modes `check`, `tune`, `test`, `summarise`).
**Pre-registration:** `output/experiments/sid_rating/preregistration.md`.
**Outputs:** `output/experiments/sid_rating/` (`tune.csv`, `chosen.json`, `predictions/`, `summary.csv`, `tests.json`).

## 1. Question

Do semantic IDs improve rating prediction for films with few or no training ratings? This is the setting where the literature reports their main benefit (Singh et al., RecSys 2024: new and long-tail items).

**Motivation:** on option 9, Surprise SVD's predictions for films with 1–20 training ratings were worse than simply predicting each user's mean rating (0.832 vs 0.809 RMSE).

## 2. Set-up

- **Data:** the option 9 time split, **warm users only.** All 1,065 cold users are excluded, including the 203 validation and 814 test users.
  - Train: ratings before 2017 (1.84M ratings, 8,339 users).
  - Test: ratings from 2017 on (107,187 ratings, 2,076 users, 1,528 films).
- **Tuning:** on an inner split only: train before 2016, validate on 2016 ratings. The test ratings were not used for any choice.
- **Film slices**, by the film's number of training ratings:

  | Slice | Test ratings | Share | Films |
  |---|---|---|---|
  | 0 (new) | 17,924 | 17% | 193 |
  | 1–20 | 5,988 | 6% | 98 |
  | 21–100 | 5,544 | 5% | 142 |
  | >100 | 77,731 | 73% | 1,095 |

  **The primary slice is "low"** = 0–20 ratings (23,912 test ratings).
- **IDs:** content IDs (S2): residual k-means on the enriched embeddinggemma vectors, 3 levels × 32 codes, for all 1,957 films. They are used as prefix tokens (level 1: 32 tokens; level 2: 659 used; level 3: 1,654 used). k-means seed 42 + run.
- **Models** (PyTorch: dimension 64, Adam, MSE + λ‖·‖², deterministic, 5 runs):

  | Model | Film representation |
  |---|---|
  | **MF atomic** | own vector and bias per film (the control) |
  | **MF sid** | sum of its 3 prefix-token vectors and biases only |
  | **MF both** | own + token vectors and biases |
  | NCF | NeuMF regression (GMF 32 + MLP 64-32-16) |
  | SVD tuned / SVD option 9 | Surprise SVD, tuned on the inner split / as in option 9 |
  | Baseline / User mean | Surprise BaselineOnly (global mean + user and film bias) / user's mean rating |

- **Chosen settings (inner split):**

  | Model | Setting | Validation RMSE |
  |---|---|---|
  | MF atomic | λ 0.03, 19 epochs | 0.7703 |
  | MF sid | λ 0.03, 13 epochs | 0.7733 |
  | MF both | λ 0.03, 9 epochs | 0.7682 |
  | NCF | λ 0.01, 8 epochs | 0.7815 |
  | SVD | 50 factors, regularisation 0.02, 30 epochs | 0.7717 |

## 3. Results (test ratings from 2017 on, mean of 5 runs)

### 3.1 RMSE by film slice

| Model | All | **Low (0–20)** | 0 (new) | 1–20 | 21–100 | >100 |
|---|---|---|---|---|---|---|
| **MF both** | **0.7720** | 0.7766 | **0.7754** | 0.7803 | **0.7920** | 0.7692 |
| MF sid | 0.7784 | **0.7746** | 0.7760 | **0.7705** | 0.7979 | 0.7782 |
| MF atomic | 0.7770 | 0.7975 | 0.7916 | 0.8151 | 0.7993 | **0.7689** |
| SVD option 9 | 0.7803 | 0.7953 | 0.7867 | 0.8205 | 0.8055 | 0.7737 |
| SVD tuned | 0.7805 | 0.8015 | 0.7936 | 0.8246 | 0.7992 | 0.7725 |
| NCF | 0.7872 | 0.7973 | 0.8015 | 0.7845 | 0.7930 | 0.7836 |
| Baseline (biases only) | 0.8134 | 0.7973 | 0.7857 | 0.8313 | 0.8125 | 0.8183 |
| User mean | 0.8646 | 0.8045 | 0.8023 | 0.8109 | 0.8311 | 0.8846 |

The standard deviation over runs is ≤ 0.003 on the low slice and ≤ 0.002 overall. MAE gives the same order (low slice: MF sid 0.573, MF both 0.576, MF atomic 0.586, SVD tuned 0.591).

### 3.2 Pre-registered tests

| Test | Difference (mean of 5 runs) | Runs better | CI | Result |
|---|---|---|---|---|
| **H1:** MF both − MF atomic, low slice | **−0.0209** | 5 / 5 | 95% [−0.0248, −0.0174] | ✅ **holds** (needed ≤ −0.010, 5/5, CI < 0) |
| **H2:** MF both − MF atomic, all ratings | **−0.0049** | 5 / 5 | 95% [−0.0067, −0.0032] | ✅ **holds** (needed ≤ +0.005); in fact **better** overall |
| S1: MF both − SVD tuned, low slice | −0.0249 | 5 / 5 | 98.3% [−0.0334, −0.0165] | lower (better) |
| S2: MF both − NCF, low slice | −0.0207 | 5 / 5 | 98.3% [−0.0307, −0.0115] | lower (better) |
| S3: MF sid − MF atomic, low slice | −0.0229 | 5 / 5 | 98.3% [−0.0289, −0.0172] | lower (better) |

CIs come from a bootstrap over users (1,000 resamples). The secondary CIs are Bonferroni-corrected over 3.

## 4. Findings

1. **Semantic IDs improve rating prediction for films with few or no ratings.** Adding content ID tokens to matrix factorisation lowered RMSE on films with 0–20 training ratings by 0.021 (2.6% relative), in every run. This is the first pre-registered positive result for semantic IDs in this work.
2. **Where the gain comes from:**

   | Slice | MF atomic → MF both |
   |---|---|
   | New films (0 ratings) | 0.792 → 0.775 |
   | Films with 1–20 ratings | 0.815 → 0.780 |

   - **New films:** MF atomic has nothing to learn from for them; their own vectors stay at their random start. With IDs, they inherit the bias and vector learned for their group, a content-based prior.
   - **Films with 1–20 ratings:** MF atomic overfits a few ratings, while the shared group tokens are stable.
3. **No cost on well-known films, and a small overall gain.** On films with >100 ratings, MF both equals MF atomic (0.7692 vs 0.7689). Overall it is the best model (0.7720), ahead of both SVD variants (0.780) and NCF (0.787).
4. **IDs only (MF sid) are best on the rarest films but lose on popular ones.**
   - 1–20 ratings: 0.7705, the best of all models.
   - >100 ratings: 0.7782, worse than atomic.
   - So the "both" design is the right default, as Singh et al. recommend: per-film parameters for films with data, shared tokens for films without.
5. **Without IDs, matrix factorisation adds nothing for rare films.**
   - On the low slice, MF atomic (0.7975), SVD (0.795–0.802) and NCF (0.797) are no better than the biases-only Baseline (0.7973).
   - On films with 1–20 ratings, SVD is even worse than predicting the user's mean (0.82 vs 0.81), as in the motivation.
   - Only the ID models beat these simple baselines on rare films.
6. **Tuning note:** SVD tuned on 2016 ratings was no better than option 9's default SVD on the 2017+ test (0.7805 vs 0.7803 overall). Settings chosen on one year did not transfer exactly to the next. This applies to all models equally, since all were tuned the same way.

## 5. Limitations

- **IDs vs raw content features not tested.** The comparison is "with ID tokens" vs "without any content".
  - A model fed the content embedding directly (e.g. a learned projection of the 768-d vector) was not tested.
  - Docs 27 and 31 found the full embedding at least as good as its IDs for similarity. The safe claim is that content information via semantic IDs helps rare films, not that IDs beat other ways of adding content.
- **Rating prediction (RMSE), warm users only.** It says nothing about ranking for new users, the main task of the thesis.
- **The effect is modest in absolute terms** (0.02 RMSE on 22% of test ratings; 0.005 overall). It is consistent in all runs and its CI is narrow.
- **"New" films are films first rated in 2017 or later** in this data. That is close to real new releases, but each is still rated by enough users to be in the test set.
- **One ID source (content), one codebook (3 × 32).** Rating-based IDs cannot cover new films. Larger codebooks (doc 33, 1d) were not tried here.
- **Each model was tuned once** (run 0, k-means seed 42), with the same grid for all MF variants.

## 6. Safe statements for the TDK

- "In a pre-registered rating-prediction test on a time split (ratings from 2017 onwards, 9,591 warm users), adding content-based semantic ID tokens to matrix factorisation lowered RMSE on films with at most 20 training ratings by 0.021 (95% CI [0.017, 0.025], 5 of 5 runs). It did not hurt films with many ratings and slightly improved overall RMSE (−0.005)."
- "On these rarely rated films, SVD, NCF and matrix factorisation without IDs were no better than a biases-only baseline; only the models with semantic ID tokens improved on it."
- Avoid: "semantic IDs beat content features" (not tested), and any claim about cold-start users or ranking.

## 7. Where the semantic-ID work stands (update of doc 31 §7)

| Use | Result |
|---|---|
| Use 1: similarity, known films (docs 30, 33) | rating-based IDs > content IDs (+0.11 NDCG); all below popularity |
| Use 1: new films (doc 31) | content IDs 0.264 vs random 0.088; below the raw embedding |
| Use 2: LTR features (docs 27, 30) | no gain |
| Use 3: item representation in ranking models (doc 28) | atomic + sid ≈ atomic |
| **Use 3 for rating prediction (this doc)** | **positive: −0.021 RMSE on rarely rated films, −0.005 overall** |
