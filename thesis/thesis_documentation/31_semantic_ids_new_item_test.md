# 31 — New-item test: can semantic IDs recommend films without ratings?

**Status:** complete (2026-10-29).
**Script:** `new_item_test.py` (modes `pilot`, `full`).
**Pre-registration:** `output/experiments/new_item_test/preregistration.md` (written before the first run, unchanged).
**Outputs:** `output/experiments/new_item_test/`. This folder holds:
- `new_films.csv`;
- `runs/`;
- `results.csv` and `summary.csv`.

**Related:**
- doc 30 (vector sources);
- doc 28 §6.1 (the new-item test was planned there);
- doc 29 §4 (literature: new and long-tail items are where semantic IDs are said to help).

## 1. Question

Doc 30 showed that IDs built from ratings (SVD, EASE) beat content IDs by about 0.10. But they cannot give an ID to a film without ratings. The hybrid S7 can: for such a film, its rating block is predicted from its content by ridge regression.

**Question:** on films without ratings, are S7 IDs better than plain content IDs (S2)?

## 2. Set-up (as pre-registered)

- **New films:** 157 of the 1,547 pool films (10%), chosen at random (rng 42) within popularity deciles.
  - Median likes: 317 for the new films and 344 for the rest, so the new films are not only rare ones.
  - Their ratings are removed from everything built from ratings: SVD, EASE, the S7 ridge map, popularity and item-CF.
  - Users are unchanged: the 9,084 simulated warm users, cross-fitted halves A/B, 10 and 3 seeds.
  - 9,037 users have at least one new film in their answer key.
- **Check:** SVD and EASE have no vector for any new film (asserted in the script). ✔
- **Task N (main):** rank only the new films. NDCG@10 against the user's liked new films; random ties.
- **Task F:** rank the whole pool. Measured with:
  - NDCG@10;
  - new-film recall@10;
  - the share of new films in the top 10.
  Ties are broken by popularity; new films have popularity 0.
- **Decision rule:** S7 counts as better than S2 if, on task N with 10 seeds, it has the higher mean over 5 k-means seeds **and** wins at least 4 of the 5.

## 3. Results

### 3.1 Task N: ranking only the new films (NDCG@10)

| Method | 10 seeds | 3 seeds |
|---|---|---|
| **Content embedding, directly (max)** | **0.309** | **0.278** |
| S2 content IDs | 0.264 ± 0.019 | 0.231 ± 0.015 |
| S7 hybrid vectors, directly (max) | 0.256 | 0.244 |
| S7 hybrid IDs | 0.240 ± 0.025 | 0.199 ± 0.017 |
| Random | 0.088 | 0.088 |
| S3 SVD / S4 EASE IDs, popularity, item-CF | cannot rank new films | cannot rank new films |

**Decision rule** (10 seeds, S7 − S2 per k-means seed):

| Seed | 42 | 43 | 44 | 45 | 46 | Mean |
|---|---|---|---|---|---|---|
| S7 − S2 | −0.036 | +0.035 | −0.031 | −0.042 | −0.049 | **−0.025** |

S7 wins 1 of 5, so **S7 is not better than S2 on new films.**

### 3.2 Task F: the whole pool (10 seeds)

| Method | NDCG@10 | New-film recall@10 | New films in top 10 |
|---|---|---|---|
| Item-CF (mean) | **0.604** | 0 | 0% |
| Popularity | 0.558 | 0 | 0% |
| S4 EASE IDs | 0.497 | 0 | 0% |
| S3 SVD IDs | 0.485 | 0 | 0% |
| S7 hybrid IDs | 0.484 | 0.006 | 2.0% |
| S7 hybrid vectors (max) | 0.471 | 0.023 | 8.5% |
| Content embedding (max) | 0.400 | **0.032** | 10.8% |
| S2 content IDs | 0.386 | 0.012 | 4.9% |

New films are 10% of the pool. A method that ignored whether a film is new would give them about 10% of its top 10.

## 4. Findings

1. **The hybrid did not help on new films.** S7 IDs were 0.025 below plain content IDs (1 of 5 seeds), and S7 vectors were 0.053 below the content embedding.
   - For a film without ratings, the rating block is only a guess from content, and the guess adds noise rather than information.
   - Adding a noisy collaborative block is worse than using content alone.
2. **For new films, the best method is the content embedding itself, without IDs** (0.309, 3.5× random). Compressing it into IDs costs 0.045 (0.309 → 0.264). This is the same loss seen in docs 27–28.
3. **There is a trade-off between accuracy and new films on the whole pool.**
   - The rating-based methods are the most accurate (item-CF 0.604, EASE IDs 0.497), but they never recommend a new film.
   - Content methods do recommend new films, at about their share of the pool, but are less accurate overall.
   - S7 IDs sit in between: nearly the accuracy of S3 (0.484 vs 0.485), but only 2% new films in the top 10.
4. **So, in this set-up, semantic IDs have no structural advantage for new items.** Content IDs can represent new films, but the raw content embedding does it better. Collaborative IDs are better for known films but cannot represent new ones. The hybrid does not combine the two strengths.
5. **The literature's claim is not refuted.** The new-item benefit reported for semantic IDs (TIGER and the papers in doc 29) comes from *generative* models that share token embeddings between known and new items. Here the IDs are used as a similarity measure only (use 1), and the doc 28 generative model was below popularity. The test answers "do the IDs themselves carry the information?", not "can a strong generative model use them?".

## 5. Limitations

- The new films are random, not recent. Real new releases have different content and no ratings at all; here, a random sample of older films had their ratings removed.
- Only use 1 was tested. The ridge map from content to SVD is a simple linear map; a learned mapping might work better (as in LETTER and similar papers).
- Simulated users only. Validation and test users were not used.

## 6. Safe statements for the TDK report

- "On films without ratings, semantic IDs built from content (NDCG@10 0.264) outperformed hybrid IDs whose collaborative part was predicted from content (0.240, 1 of 5 k-means seeds better). Both were below the raw content embedding (0.309) and well above random (0.088)."
- "Rating-based methods were the most accurate on the whole pool but never recommended a new film. Content-based methods recommended new films at about their share of the catalogue."
- Avoid: "semantic IDs do not help with cold-start items" in general (only the similarity use was tested), and any claim about test users.

## 7. Where the semantic-ID work stands

| Use | Best result | vs the strongest baseline |
|---|---|---|
| Use 1, known films (doc 30) | EASE IDs 0.507 | below popularity (0.559) and item-CF (0.614) |
| Use 1, new films (this doc) | content IDs 0.264 | below the content embedding (0.309) |
| Use 2, LTR features (docs 27, 30) | no gain from any source | LTR v2 unchanged |
| Use 3, item representation (doc 28) | atomic + sid ≈ atomic | all below popularity |

**Consistent picture:** on this data (1,549 films), a semantic ID is a compressed copy of a vector. Whatever the source, the full vector or a dedicated model does at least as well. The one robust positive finding is the source comparison: rating-based IDs beat content-based IDs by about 0.10 (doc 30).
