# Pre-registration: semantic IDs for rating prediction (option 9 split)

**Written:** 2026-10-29, before any model in this experiment was trained.
**Script:** `sid_rating_experiment.py` (modes `check`, `tune`, `test`, `summarise`).
**Outputs:** `output/experiments/sid_rating/`.

## 1. Question

Do semantic IDs improve rating prediction for films with few or no training ratings, the setting where the literature reports their main benefit (Singh et al., RecSys 2024)?

**Motivation (measured before this pre-registration, option 9 as it is in `recomm.py`, all users):**
- Surprise SVD reaches RMSE 0.7745 overall.
- On films with 1–20 training ratings it scores 0.832, which is worse than each user's mean rating (0.809).
- On films with 0 training ratings it scores 0.782, about the same as the user mean (0.796).
- Together these films carry 27% of the test ratings.

## 2. Data and splits (fixed)

- **Ratings:** `bmt.prepare_ratings(bmt.load_movies())`, which gives 2,547,690 ratings on the 1,957-film catalogue.
- **Users:** the **warm users only** (`cs.split_users`, seed 42). All cold users are excluded, including the 203 validation and 814 test users, so the final cold-start test stays untouched.
- **Outer split (as option 9):**
  - train = ratings before 2017-01-01;
  - test = ratings from 2017-01-01 on, by users who have training ratings.
- **Inner split (tuning only):**
  - train = ratings before 2016-01-01;
  - validation = ratings in 2016, by users with inner training ratings.
  - The outer test ratings are never used for any choice.
- **Film slices**, by the film's number of ratings in the respective training set: `0 (new)`, `1-20`, `21-100`, `>100`.
- **The primary slice is "low"** = `0 (new)` + `1-20`.

## 3. Semantic IDs

- **Source:** S2 content IDs: residual k-means on `movies_enriched_embeddinggemma_v2` via `Semantic_IDs` from `recomm.py`, 3 levels × 32 codes, for all 1,957 films. New films need IDs without ratings, so this is the only source that covers them.
- **Not tested here:** S3, S4 and S5 have no IDs for unrated films. S7's rating part would be imputed from content, which did not help on new films (doc 31).
- **Tokens are prefixes,** as in Singh et al.'s prefix n-grams:
  - level 1 = c1 (32 values);
  - level 2 = (c1, c2) (1,024 values);
  - level 3 = (c1, c2, c3) (32,768 values).
- **k-means seed** 42 + run (runs 0–4). Tuning uses seed 42.

## 4. Models

All PyTorch models:
- dimension 64 (NCF: 32 per tower);
- Adam, lr 1e-3, batch 4,096;
- loss = MSE + λ × the squared norm of the embedding rows used in the batch;
- predictions clipped to [0.5, 5];
- `torch.use_deterministic_algorithms(True)` and a fixed seed per run.

| Name | Prediction |
|---|---|
| **MF atomic** | μ + b_u + b_i + p_u · q_i (one learned vector and bias per film) |
| **MF sid** | as MF atomic, but the film vector and bias are the sums of its 3 prefix-token vectors and biases (no per-film parameters) |
| **MF both** | film vector = q_i + Σ token vectors; film bias = b_i + Σ token biases |
| NCF | NeuMF regression: GMF (32) and MLP (64 → 32 → 16), linear output, plus μ, b_u and b_i |
| SVD tuned | Surprise SVD, tuned on the inner split |
| SVD option 9 | Surprise SVD as in option 9 (n_epochs 15, other settings default) |
| Baseline | Surprise BaselineOnly (reference) |
| User mean | the user's mean training rating (reference) |

**MF atomic is the control for H1:** the same code and training as MF both, without the IDs.

## 5. Tuning (inner split, run 0 only)

- **PyTorch models:** λ ∈ {0.01, 0.03, 0.1}.
  - Up to 40 epochs, with early stopping (patience 3) on inner validation RMSE (all films).
  - Keep the λ with the lowest validation RMSE and its best epoch count.
  - Each variant is tuned separately with the same grid.
- **SVD:** n_factors ∈ {50, 100, 200}, reg_all ∈ {0.02, 0.05}, n_epochs ∈ {15, 30}, lr 0.005. Lowest inner validation RMSE.
- **Selection uses overall RMSE, not the low slice,** so that nothing is tuned for the hypothesis.

## 6. Test (outer split, 5 runs)

- **Each run r = 0–4:** refit every model on the outer training set with the chosen settings (fixed epochs, no early stopping).
  - Seeds: torch seed r, k-means seed 42 + r, Surprise random_state 42 + r.
- **Saved:** per-rating predictions of every model and run.

## 7. Hypotheses and decision rules

**H1 (primary):** MF both has lower RMSE than MF atomic on the low slice. Confirmed only if all three hold:
1. the mean over 5 runs of RMSE(both) − RMSE(atomic) is ≤ −0.010;
2. both is better in 5 of 5 runs;
3. the 95% bootstrap CI of that mean difference is entirely below 0. The bootstrap resamples users (1,000 resamples, seed 42) and recomputes each run's RMSE.

**H2 (no overall cost):** the mean over runs of RMSE(both) − RMSE(atomic) on all test ratings is ≤ +0.005.

**Secondary** (low slice; Bonferroni over 3, so 98.33% CIs):
- S1: MF both vs SVD tuned;
- S2: MF both vs NCF;
- S3: MF sid vs MF atomic.

A secondary comparison counts as a difference only if its CI excludes 0.

**Descriptive only:** every slice separately, MAE, SVD option 9, Baseline, user mean, and tuned settings.

**How results will be reported:**
- H1 and H2 hold → "content-based semantic IDs improve rating prediction for films with few ratings without overall cost".
- H1 fails → reported as a negative result.
- Secondary results never replace the primary one.

## 8. Known limits (stated in advance)

- **RMSE is a rating-accuracy measure,** not the ranking measure of the main thesis.
- **"New" films here are films first rated after 2016** (real new releases), not random films.
- **Warm users only.** The result says nothing about cold-start users.
- **Each model is tuned once** (run 0, k-means seed 42).
