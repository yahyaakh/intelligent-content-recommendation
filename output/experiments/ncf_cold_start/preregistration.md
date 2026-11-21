# Pre-registration: NCF (NeuMF, fold-in) on the 814 test users

Written 2026-10-26, **before** NCF was scored on any test user. The test (`python ncf_experiments.py test`)
refuses to run unless this file is committed to git unchanged, and it runs only once.

## 1. What is tested

NCF as in recomm.py (NeuMF, n_factors 4, layers [16, 8, 4], 30 epochs, batch 256, lr 1e-3, every rating a
positive), trained on the 9,591 warm users only, training seed 42.

- Model file: `output/experiments/final_comparison/cold_start_models/ncf_warm.npz`
- SHA-256: `1f9c7c38141f6dfca3c4094a537606636f3a175c622a2090fc11f80e785d8959` (checked before scoring)

New users by fold-in: every network weight and item embedding frozen; only the user's GMF (4) and MLP (8)
embeddings are learned on the seeds as positives, 4 random negatives per positive, Adam, binary cross-entropy,
random start N(0, 0.01), random generator seeded with 42 + user_id.

| Seeds | Adam steps | Learning rate | Validation NDCG@10 (chosen on these 203 users) |
|---|---|---|---|
| 10 | 1,000 | 0.05 | 0.5980 |
| 3 | 300 | 0.01 | 0.5537 |

Nothing about the model or fold-in changes after this file is committed.

## 2. Setup (the same as every other test result in the thesis)

814 test users (`cs.split_validation`), seeds = each user's first 3 or 10 liked films in time order, answer key
= their other liked films (rating >= their own mean) in the pool, the same key for 3 and 10 seeds; candidates
= every pool film except the 10 seed films; NDCG@10 (precision and recall reported, not tested).

## 3. Hypotheses (two-sided paired Wilcoxon over the 814 users, Bonferroni over 4, alpha 0.05)

| | Seeds | Comparison | Validation difference |
|---|---|---|---|
| H1 | 10 | NCF vs Hybrid E | -0.0305 |
| H2 | 3 | NCF vs Hybrid E | -0.0300 |
| H3 | 10 | NCF vs Popularity | +0.0354 |
| H4 | 3 | NCF vs Popularity | -0.0089 |

A hypothesis is **confirmed** when p_Bonferroni < 0.05; its direction is the sign of the mean difference
(reported with a 95% bootstrap interval, 10,000 resamples of users, seed 42). Not significant = "no
difference shown", never "equal". Every result is reported, whichever way it goes.

## 4. Sanity checks (before any NCF test score is computed; the run stops if one fails)

1. Hybrid E on the test users = 0.6311 (10 seeds) and 0.5970 (3 seeds), as in docs 06 and 16.
2. Popularity on the test users = 0.5598 (both seed counts).
3. The NCF file's SHA-256 matches section 1.
4. NCF reproduces its validation numbers (0.5980 and 0.5537).
5. No test user is among the warm training users.

## 5. Not part of this test

User-kNN, SVD and the other option-10 methods are not scored on the test users here (user-kNN would need its
own pre-registration). The training-seed study (`python ncf_experiments.py seeds`) uses validation users only.
