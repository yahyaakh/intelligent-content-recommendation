# 33 — Robustness of the two positive semantic-ID results

**Status:** complete (2026-10-29), 8 tests.
**Script:** `sid_robustness.py` (one mode per test).
**Pre-registration:** `output/experiments/sid_robustness/preregistration.md` (written before any run, unchanged).
**Outputs:** `output/experiments/sid_robustness/` (`t1a_*` … `t2c_*`, `runs/`).

**Check before any test:** the script's own scoring reproduces doc 30 exactly (S2 0.405814, S3 0.494076 at k-means seed 42). ✔

**The two results under test:**
1. IDs built from ratings (SVD S3, EASE S4) beat IDs built from content (S2) by about 0.10 NDCG@10 (doc 30).
2. Films sharing an ID prefix share genre, director, decade and co-liking far more often than random pairs (docs 27, 30).

## Summary

| Test | Question | Result |
|---|---|---|
| **1c** | Is result 1 significant per user? | ✅ +0.10 to +0.11, 95% CI ±0.003, p ≈ 0 (9,084 users) |
| **1b** | Does it hold on the 203 validation users? | ✅ **Confirmed** (+0.104 / +0.110, p < 0.001 after Bonferroni) |
| **1d** | Does it hold for other ID settings? | ✅ **Holds in all 5 settings** (+0.09 to +0.12, 5 of 5 seeds each) |
| **1e** | Does another rating-based source also beat content? | ✅ item-CF IDs +0.091, 5 of 5 seeds |
| **1a** | Is the gain more than "popular films"? | ❌ **No.** The gain is on the popular third; content IDs are slightly better on the rest |
| **2a** | Are the lifts beyond chance? | ✅ All 48 lifts far above 1,000 random reassignments |
| **2b** | Do the groups make sense to an outside judge (LLM odd-one-out)? | ✅ 74% (content) and 77% (EASE) vs 17% chance; random groups 19% |
| **2c** | Does the residual hierarchy help? | ❌ **No.** Flat clusters do better (+0.022 content, +0.012 EASE) |

## 1c. Per-user significance (9,084 simulated users; mean over 5 k-means seeds; popularity ties)

| Seeds | Comparison | NDCG@10 | Difference | 95% CI | Users better / worse | Precision difference |
|---|---|---|---|---|---|---|
| 10 | SVD − content | 0.498 vs 0.396 | **+0.102** | [0.099, 0.106] | 6,758 / 2,317 | +0.105 |
| 10 | EASE − content | 0.507 vs 0.396 | **+0.111** | [0.108, 0.115] | 6,921 / 2,152 | +0.116 |
| 10 | hybrid − content | 0.502 vs 0.396 | **+0.107** | [0.104, 0.110] | 6,970 / 2,103 | +0.108 |
| 3 | SVD − content | 0.441 vs 0.353 | +0.088 | [0.085, 0.091] | 6,480 / 2,585 | +0.094 |
| 3 | EASE − content | 0.460 vs 0.353 | +0.107 | [0.104, 0.110] | 6,727 / 2,343 | +0.112 |
| 3 | hybrid − content | 0.440 vs 0.353 | +0.087 | [0.084, 0.090] | 6,582 / 2,489 | +0.089 |

All p values are below 10⁻³⁰⁰, including after Bonferroni over 6.

## 1b. The 203 validation users (IDs from all warm users; mean over 5 k-means seeds)

| Seeds | Content | SVD | EASE | Hybrid | Popularity |
|---|---|---|---|---|---|
| 10 | 0.395 | 0.499 (+0.104, CI [0.078, 0.129]) | **0.505** (+0.110, CI [0.086, 0.134]) | 0.500 (+0.105) | 0.563 |
| 3 | 0.358 | 0.443 (+0.086) | **0.467** (+0.109) | 0.447 (+0.090) | 0.563 |

- Users better / worse (10 seeds): SVD 158 / 45, EASE 155 / 48.
- p_Bonf < 0.001. Precision and recall agree: EASE +0.119 precision and +0.012 recall.
- **Rule: confirmed.** The validation numbers are almost identical to the simulated users'.

## 1d. Other ID settings (simulated users, 10 seeds, popularity ties)

| Codes × levels | Content | SVD | EASE | SVD − content | EASE − content |
|---|---|---|---|---|---|
| 16 × 3 | 0.383 | 0.485 | 0.497 | +0.103 (5/5) | +0.115 (5/5) |
| 32 × 3 (doc 30) | 0.396 | 0.498 | 0.507 | +0.102 (5/5) | +0.111 (5/5) |
| 64 × 3 | 0.403 | 0.520 | **0.527** | +0.117 (5/5) | +0.124 (5/5) |
| 128 × 3 | 0.411 | 0.523 | 0.523 | +0.111 (5/5) | +0.112 (5/5) |
| 32 × 2 | 0.409 | 0.497 | 0.511 | +0.088 (5/5) | +0.102 (5/5) |
| 32 × 4 | 0.397 | 0.501 | 0.507 | +0.105 (5/5) | +0.111 (5/5) |

- **Rule: holds in every setting.** The gap is 0.09–0.12 whatever the codebook.
- **Larger codebooks help all sources:** 64–128 codes beat 32 by about +0.02. EASE 64 × 3 (0.527) is the best semantic-ID result in this work, still below popularity (0.559).

## 1e. Item-CF as a further rating-based source (simulated users)

| | Content | **Item-CF IDs** | SVD | EASE | Hybrid |
|---|---|---|---|---|---|
| NDCG@10, 10 seeds | 0.396 | **0.487** | 0.498 | 0.507 | 0.502 |
| NDCG@10, 3 seeds | 0.353 | 0.431 | 0.441 | 0.460 | 0.440 |

- **Rule:** item-CF IDs beat content IDs (+0.091, 5 of 5 seeds).
- Three different rating-based sources (SVD, EASE, item-CF) all beat content. NCF was not tested: it would need new training for each half.

## 1a. Where does the gain come from? (simulated users, mean over 5 k-means seeds)

Pool films are split into thirds by how many warm users liked them. "NDCG, third" scores the same top 10 against only the liked films of that third.

| 10 seeds, popularity ties | Content | SVD | EASE | Hybrid |
|---|---|---|---|---|
| NDCG@10 (all) | 0.396 | 0.498 | **0.507** | 0.502 |
| NDCG, popular third | 0.382 | 0.488 | **0.495** | 0.492 |
| NDCG, middle third | **0.014** | 0.010 | 0.012 | 0.010 |
| NDCG, least popular third | **0.0018** | 0.0010 | 0.0015 | 0.0008 |
| Share of top 10 from least popular third | 2.6% | 3.2% | 0.4% | 1.6% |
| Novelty (higher = less mainstream) | **2.32** | 1.90 | 1.69 | 1.75 |
| Catalogue coverage | 67% | **73%** | 70% | 72% |
| Gini (higher = more concentrated) | **0.865** | 0.878 | 0.874 | 0.878 |

With random ties the pattern is the same. Content has the best middle third (0.023) and the highest novelty (2.98). SVD has the highest coverage (78%).

**Rule failed:** SVD and EASE are *below* content on the middle and least popular thirds (0 to 2 of 5 seeds).

**Findings:**
1. **The +0.10 gain is entirely on popular films.** Rating-based IDs are better at telling which popular films a user will like.
2. **No ID method finds liked films outside the popular third** (NDCG ≤ 0.023 there for all). The users' liked films are mostly popular, so this is where any method can gain.
3. **Content IDs are less mainstream** (novelty 2.32 vs 1.69–1.90) and slightly better on less popular films. But those films rarely match what users liked.
4. **Rating-based IDs still reach more of the catalogue** (70–73% vs 67%). They are not simply recommending the same few blockbusters.

## 2a. Lifts vs random IDs (permutation test, 1,000 reassignments, IDs of doc 30 test A)

| Level 1 lift | Genre | Director | Decade | Co-liked |
|---|---|---|---|---|
| Content | 1.76 | 14.3 | 1.89 | 1.62 |
| SVD | 1.52 | 9.4 | 2.06 | 1.70 |
| EASE | 1.50 | 10.2 | **3.12** | 1.66 |
| Hybrid | 1.74 | 10.3 | 2.18 | 1.72 |
| *Random IDs (99th percentile)* | *≤ 1.05* | *≤ 1.34* | *≤ 1.05* | *≤ 1.03* |

- No random reassignment reached any observed lift, at any level, for any source.
- p = 1/1,001 for each test, so p_Bonf = 0.048 over 48 tests. That is the smallest value 1,000 permutations can give; more permutations would give smaller p.

## 2b. LLM odd-one-out test (deepseek-v4.1-flash, 100 trials per condition)

| Groups | Correct | Accuracy | Chance | p_Bonf (one-sided) |
|---|---|---|---|---|
| Content IDs (level 1) | 74 | **74%** | 16.7% | < 10⁻⁴ |
| EASE IDs (level 1) | 77 | **77%** | 16.7% | < 10⁻⁴ |
| Random groups (control) | 19 | 19% | 16.7% | 0.91 |

- Content vs EASE: Fisher exact p = 0.74, no difference. All answers were parsed.
- **Method:** the word-intrusion test of Chang et al. (2009), with an LLM as the judge as in Stammbach et al. (2023) for topic models.
- **Novelty of the method:** to our knowledge (literature search, 2026-10-29) it has not been applied to semantic IDs. Prior SID evaluation uses accuracy, codebook utilisation and collision rates.
- **Findings:**
  - Both kinds of groups are recognisable to an outside judge, about 4.5× chance.
  - EASE groups are built only from who liked what, yet they are as coherent as content groups.
  - The random control shows the test is fair.
- **Limitation:** no human check of the LLM's judgements.

## 2c. Does the residual hierarchy matter? (simulated users)

Flat IDs use an independent k-means at each level, with as many clusters as the residual IDs have distinct prefixes. They are scored by matches at each level, without a prefix condition.

| NDCG@10 | Residual (ours) | Flat | Residual − flat (wins) |
|---|---|---|---|
| Content, 10 seeds | 0.396 | **0.418** | −0.022 (1 of 5) |
| EASE, 10 seeds | 0.507 | **0.519** | −0.012 (0 of 5) |
| Content, 3 seeds | 0.353 | **0.404** | −0.050 |
| EASE, 3 seeds | 0.460 | **0.477** | −0.017 |

- **Rule failed:** the residual hierarchy does not help; plain clusters do better.
- **Likely reason:** in a residual ID, a film that differs from a seed at level 1 gets no credit for any deeper match. Flat clusters give partial credit at each level independently.
- This matches Wang et al. (2026, arXiv 2607.24995): semantic IDs keep the coarse organisation but lose fine local structure.

## Overall conclusions

**Result 1 (ratings > content) is robust:**
- significant per user;
- confirmed on held-out validation users;
- holds for every codebook setting and for three rating-based sources.

**But it is a popular-film effect:** rating-based IDs better identify which popular films a user likes. Content IDs are less mainstream and slightly better on less popular films, which rarely match.

**Result 2 (meaningful structure) is robust:**
- far beyond chance in a permutation test;
- recognisable to an LLM judge in about 3 of 4 trials, for both content and EASE groups.

**Two qualifications:**
- **No method beats popularity.** The best semantic-ID result is EASE flat 0.519 and EASE 64 × 3 0.527, against popularity 0.559 and item-CF 0.614.
- **The residual hierarchy is not what makes the IDs useful.** Flat clusters do at least as well.

## Safe statements for the TDK

- "Semantic IDs built from collaborative vectors outperformed content-based IDs by 0.10–0.11 NDCG@10 (95% CI ±0.003, 9,084 simulated users; confirmed on 203 validation users, p < 0.001). The gap held across five codebook configurations and three collaborative sources (SVD, EASE, item-CF)."
- "The gain was concentrated on popular films: on the least popular two thirds of the catalogue, content-based IDs were marginally better and recommended less mainstream films."
- "Films sharing an ID prefix agreed on genre, director, decade and co-liking far above chance (permutation test). An LLM identified an intruder film among five same-group films in 74–77% of trials (chance 17%; random groups 19%)."
- "A flat clustering with the same number of groups per level outperformed the residual hierarchy (−0.012 to −0.022 NDCG@10)."
