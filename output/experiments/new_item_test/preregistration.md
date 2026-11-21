# Pre-registration: new-item test for semantic IDs (doc 31)

Written 2026-10-29, **before** any run of `new_item_test.py`. Not to be changed after the first run;
later additions go in a new dated section below.

## Question
Doc 30: collaborative IDs (S3 SVD, S4 EASE) beat content IDs (S2) by about 0.10 NDCG@10 in use 1, but
they cannot give an ID to a film without ratings. The hybrid S7 can (its SVD block is predicted from
content). **Can S7 recommend films that have no ratings yet, and better than content-only IDs?**

## New films
- 10% of the 1,549 pool films (155), chosen at random (rng 42) **within popularity deciles** (deciles of the
  number of warm users who liked the film), so the new films are not only rare ones.
- Their ratings are removed from **everything that is built from ratings**: SVD, EASE, the S7 ridge map,
  popularity, item-CF. The users' seeds and answer keys are unchanged (a user may have liked a new film:
  it then counts in the answer key, and as a seed it has a content vector but no collaborative one).
- Limitation: real new films are recent; random new films are a simplification (stated in doc 31).

## Methods (all cross-fitted with halves A/B, seed 42; 9,084 simulated warm users, 10 and 3 seeds)
- ID methods (use 1, `Semantic_IDs.recommend`, flat weights, 3 x 32 codes, k-means seeds 42-46):
  S2 content, S3 SVD, S4 EASE (lambda 20,000), S7 hybrid. Built exactly as in doc 30, from the ratings
  without the new films. S3/S4 have no ID for the new films.
- Direct references (no IDs): content embedding (max), S7 vectors (cosine, max), popularity, item-CF
  (mean), random (task N).

## Tasks
- **N, new films only (main):** candidates = the new films not among the user's 10 seeds; relevant =
  the user's answer key ∩ new films; users with >= 1 relevant new film. NDCG@10. Ties: random (rng 0).
  S3, S4, popularity and item-CF cannot rank new films (no vector) and are reported as such, not run.
- **F, full pool:** candidates = pool films that have an ID / vector; answer key unchanged. NDCG@10,
  new-film recall@10 (hits among relevant new films / relevant new films, users with >= 1) and
  new-film share of the top 10. Ties: popularity of the other half (new films have popularity 0).

## Decision rule
Main: task N, 10 seeds. **S7 is better than S2 on new films** if its mean NDCG@10 over the 5 k-means
seeds is higher AND it wins >= 4 of 5 seeds. 3 seeds and task F are reported and do not decide.
Descriptive comparisons: S7 vs content embedding directly and vs random on task N; S7 vs S3/S4 on task F
(does the coverage cost accuracy on the whole pool?).

Validation and test users are not used.

## Outputs (`output/experiments/new_item_test/`)
`new_films.csv`, `runs/` (one file per finished run), `results.csv`, `summary.csv`.
