# Pre-registration: robustness of the two positive semantic-ID results (doc 33)

Written 2026-10-29, **before** any run of `sid_robustness.py`. Not to be changed after the first run;
later additions go in a new dated section below.

Result 1 (doc 30): IDs from ratings (S3 SVD, S4 EASE) beat IDs from content (S2) by about 0.10 NDCG@10 in
use 1 (9,084 simulated users, 5 of 5 k-means seeds). Result 2 (docs 27, 30): films sharing an ID prefix
share genre / director / decade / co-liking far more often than random pairs.

Common settings unless stated: 3 levels x 32 codes, k-means seeds 42-46, flat prefix weights, use 1 as in
doc 30 (candidates = pool films with an ID, seeds and all 10 seed-pool films excluded, ties by popularity of
the other half), cross-fitted halves A/B (seed 42), N = 10 (decides) and 3 (reported). Main metric NDCG@10;
precision and recall @10 always reported. Test users are not used.

## 1a. Where does the gain come from? (simulated users)
Rerun use 1 for S2, S3, S4, S7 with the top-10 lists kept. Pool films are split into thirds by the number
of warm users who liked them (low / mid / high). Per user: NDCG@10 against the relevant films of one third
only (users with >= 1 relevant film in that third), share of the top 10 in the low third, novelty
(mean -log2 of the share of warm users who liked the film). Per run: catalogue coverage (share of pool films
recommended to >= 1 user) and Gini of recommendation counts.
**Rule:** the rating-based gain is "not only popular films" if S4 - S2 AND S3 - S2 are positive on the LOW and
on the MID third (10 seeds, popularity ties; mean over k-means seeds, and >= 4 of 5 seeds). Random ties reported.

## 1b. The 203 validation users
Use 1 with IDs fitted on all warm users (S2 content; S3, S4, S7 from `sid_vector_sources/vectors/*_all.npy`),
popularity ties from all warm users, 5 k-means seeds, per-user NDCG = mean over the seeds.
**Rule:** confirmed if S3 > S2 and S4 > S2 with paired Wilcoxon p < 0.05 after Bonferroni over the 2 tests
(10 seeds). 3 seeds and S7 reported. Popularity on the same users reported for context.

## 1c. Per-user significance on the simulated users (existing doc 30 runs, `runs_metrics/`)
Per-user NDCG / precision / recall = mean over the 5 k-means seeds (popularity ties). Paired Wilcoxon and 95%
bootstrap CI for S3 - S2, S4 - S2, S7 - S2 at 10 and 3 seeds; Bonferroni over these 6 (NDCG).

## 1d. Other ID settings (simulated users)
S2, S3, S4 with (codes, levels) = (16, 3), (64, 3), (128, 3), (32, 2), (32, 4); popularity ties only.
**Rule:** result 1 "holds across settings" if in EVERY setting S3 and S4 each have a higher mean than S2 and
win >= 4 of 5 k-means seeds (10 seeds).

## 1e. Another rating-based source (simulated users)
S5 item-CF: row i of the item-CF similarity table built from the other half's ratings (pool columns; films
with an all-zero row dropped). Same screening as doc 30.
**Rule:** S5 better than S2 if higher mean and >= 4 of 5 seeds (10 seeds). (NCF is not tested: it has no
per-half film vectors without new training; stated as a limitation.)

## 2a. Lifts vs random IDs (permutation test)
IDs of test A (doc 30: `ids/{S2,S3,S4,S7}_all-pool_k32_s42.npz`). Null: the same codes randomly reassigned to
the films (1,000 permutations, rng 42), which keeps every group size. For genre (first genre), director,
decade and co-liked at L1-L3: p = (number of null lifts >= observed + 1) / 1,001; Bonferroni over 48.

## 2b. Odd-one-out test with the LLM
Groups = first token (L1) of `S2_all-pool_k32_s42` (content) and `S4_all-pool_k32_s42` (EASE); control =
the S4 groups with labels randomly permuted (rng 42). Only well-known films (top half of the pool by number
of warm users who liked them); groups with >= 5 such films. 100 trials per condition (rng 42): 5 films from
one group + 1 film from another group, in random order, shown as "title (year)". The LLM (deepseek-v4.1-flash,
ollama_cloud, temperature 0) is told that five were grouped together by a recommender and asked for the odd one.
**Rule:** a condition's groups "make sense to the LLM" if accuracy > 1/6 with a one-sided binomial p < 0.05
after Bonferroni over 3. Content vs EASE compared descriptively (Fisher exact test).

## 2c. Does the residual hierarchy matter? (simulated users)
For S2 and S4: our residual IDs vs FLAT IDs: at each level L an independent k-means on the vectors with as many
clusters as the residual IDs have distinct prefixes of length L. Flat score = mean over seeds of the number of
levels at which the candidate shares the seed's cluster (no prefix condition).
**Rule:** the hierarchy helps if residual > flat in mean and in >= 4 of 5 seeds (10 seeds), per source.

## Outputs (`output/experiments/sid_robustness/`)
One CSV per test (`t1a_*.csv`, ...), `runs/` for finished runs, and a summary printed per test.
