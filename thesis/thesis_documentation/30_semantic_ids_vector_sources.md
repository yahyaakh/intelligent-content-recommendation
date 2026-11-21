# 30 — Semantic IDs from different vector sources (stage 1: content vs SVD)

**Status:** stages 1, 2a, 2b and 2c complete (2026-10-29). The source comparison is closed as pre-registered.
**Script:** `sid_vector_sources.py` (modes `pilot`, `screen`).
**Pre-registration:** `output/experiments/sid_vector_sources/preregistration.md` (written before the first run, unchanged).
**Outputs:** `output/experiments/sid_vector_sources/`. This folder holds:
- `vectors/`, `ids/` and `runs/`;
- `results.csv` and `summary.csv`.

**Related:**
- doc 27 (building the IDs, uses 1 and 2);
- doc 28 (use 3);
- doc 29 (literature; Penha et al. found CF-based IDs far better on MovieLens 25M).

## 1. Question

Docs 27–28 built every semantic ID from **content** vectors (embeddinggemma on the LLM-enriched text). Doc 29 found that the literature's strongest MovieLens result uses **collaborative** vectors instead.

**Question:** do IDs built from SVD film vectors recommend better than IDs built from content vectors, with everything else fixed?

## 2. Set-up (as pre-registered)

| Source | Vectors | Uses ratings? |
|---|---|---|
| **S2 content** | `movies_enriched_embeddinggemma_v2` (the IDs of docs 27–28) | no |
| **S3 SVD** | film vectors U·s, rank 64, of the user-centred rating matrix (same SVD as `latent_tables`) | yes → cross-fitted |

- **Same for both:** 3 levels × 32 codes + collision token; faiss k-means (niter 25, nredo 10); k-means seeds 42–46; vectors normalised to length 1 (`Semantic_IDs.set_vectors`); flat prefix weights.
- **Cross-fitting:** simulated users of half A are scored with IDs built from half B's ratings, and vice versa.
- **Coverage:** half A's SVD covers 1,540 of the 1,549 pool films, half B's covers 1,530. The missing films have no ratings in that half. They get no ID and are left out of the candidates for the other half's users. Test A uses IDs built from all warm users, which cover all 1,549 films.
- **Test A (descriptive):** lift of genre, director, decade and **co-liked** agreement among films that share the first L tokens.
- **Test B (decides):** use 1 (prefix similarity, `Semantic_IDs.recommend`) on the 9,084 simulated warm users; NDCG@10; 10 and 3 seeds; ties broken by popularity (main) or at random.
- **Check:** S2 with k-means seed 42 reproduced doc 27 §4a exactly (0.4058 / 0.3614). ✔
- **Decision rule:** S3 counts as better if it has the higher mean at 10 seeds with popularity ties **and** wins at least 4 of the 5 k-means seeds.

## 3. Results

### 3.1 Test B: use 1, NDCG@10 (9,084 simulated warm users)

**Per k-means seed** (10 seeds, popularity ties; this decides):

| k-means seed | S2 content | S3 SVD | S3 − S2 |
|---|---|---|---|
| 42 | 0.4058 | 0.4941 | +0.088 |
| 43 | 0.3824 | 0.5018 | +0.119 |
| 44 | 0.4167 | 0.4992 | +0.083 |
| 45 | 0.3851 | 0.4819 | +0.097 |
| 46 | 0.3885 | 0.5126 | +0.124 |
| **mean** | **0.3957** | **0.4979** | **+0.102** (S3 wins 5 of 5) |

**Summary** (mean ± std over the 5 k-means seeds):

| Seeds | Ties | S2 content | S3 SVD |
|---|---|---|---|
| 10 | popularity | 0.396 ± 0.015 | **0.498 ± 0.011** |
| 10 | random | 0.311 ± 0.015 | **0.444 ± 0.017** |
| 3 | popularity | 0.353 ± 0.007 | **0.441 ± 0.005** |
| 3 | random | 0.258 ± 0.008 | **0.360 ± 0.009** |

**References on the same users** (no IDs):

| Method | 10 seeds | 3 seeds |
|---|---|---|
| Item-CF (mean) | 0.614 | 0.529 |
| Popularity | 0.559 | 0.559 |
| SVD vectors directly (cosine, max) | 0.472 | 0.412 |
| Content embedding directly (max) | 0.400 | 0.338 |

### 3.2 Test A: what the IDs group together (lift, mean over k-means seeds)

| Attribute | Prefix | S2 content | S3 SVD |
|---|---|---|---|
| co-liked | L1 / L2 / L3 | 1.62 / 2.04 / 2.80 | 1.68 / 2.15 / 2.78 |
| genre | L1 / L2 / L3 | 1.82 / 2.26 / 2.60 | 1.54 / 1.92 / 2.32 |
| director | L1 / L2 / L3 | 13.5 / 52.3 / 150.7 | 9.8 / 39.0 / 110.5 |
| decade | L1 / L2 / L3 | 1.93 / 2.59 / 3.06 | 2.04 / 2.50 / 2.94 |

## 4. Findings

1. **SVD-based IDs beat content-based IDs clearly** by the pre-registered rule: +0.102 NDCG@10 (10 seeds, popularity ties), 5 of 5 k-means seeds. The lead holds with 3 seeds (+0.088) and with random ties (+0.13 / +0.10). This agrees with Penha et al. (doc 29).
2. **The source matters more than any ID setting tried before.** The gap between sources (+0.10) is larger than the effect of any codebook or weighting choice in doc 27, and S3's spread over k-means seeds (std 0.011) is smaller than S2's (0.015).
3. **The IDs from SVD did better than the SVD vectors they were built from** (0.498 vs 0.472 at 10 seeds). Two differences between the methods may explain this: prefix similarity scores the seed films together rather than taking the max over single seeds, and ties go to popular films. This is an observation, not a tested claim.
4. **Test A hardly separates the sources.** Their co-liked lift is about the same (1.68 vs 1.62 at L1, equal at L3), and content IDs group genre and director more strongly, as expected. So the descriptive lifts do not predict the use-1 result. The NDCG gap must come from finer structure than "films liked together", e.g. the ordering inside the shared groups.
5. **Use 1 is still weak in absolute terms.** Even S3 (0.498) is below popularity (0.559) and item-CF (0.614) on the same users. The value of S3 has to be shown inside a model (use 2 in LTR, or use 3), not as a recommender of its own.
6. **Coverage:** S3 has no ID for films without ratings (9–19 films per half here, and the 408 catalog films with no ratings at all). Content IDs cover everything. A content + SVD hybrid (S7) is the planned way to get both.

## 5. Safe statements for the TDK report

- "Semantic IDs built from SVD film vectors outperformed IDs built from content embeddings by 0.10 NDCG@10 when used as a similarity measure (5 of 5 k-means seeds, 9,084 simulated users), in line with Penha et al. (2025)."
- "Used on their own, neither kind of ID reached the popularity baseline."
- Avoid: "collaborative semantic IDs work" (only use 1 has been tested so far), or any claim about validation or test users (none were used).

## 6. Next (pre-registered consequence: S3 is better)

1. **Stage 2:** S3 IDs as LTR v2 features (use 2), using the L1 harness and the +0.005 rule on both user groups, the same as doc 27 §7.
2. **Add to the screening:** S4 EASE item vectors and S7 content + SVD hybrid (which also covers films without ratings).
3. Later: S3 IDs in the doc 28 models (use 3), and the new-item test (doc 29 §4), where S7 matters most.

Each of these gets a dated addition to the pre-registration before it runs.

---

## 7. Stage 2b: two more sources (EASE and hybrid), 2026-10-29

Pre-registered in the stage 2 addition of `preregistration.md` before running. The screening is the same as in stage 1 (test A + test B, 5 k-means seeds, same 9,084 users).

| Source | Vectors | Films with an ID (half A / B) |
|---|---|---|
| **S4 EASE** | column j of the EASE weight matrix B (lambda 20,000), cross-fitted | 1,509 / 1,494 (films never liked in a half are dropped) |
| **S7 hybrid** | [content ; SVD], each block of length 1, / √2. Films without an SVD vector get their SVD block predicted from content (ridge, alpha 1). | 1,957 / 1,957 (all) |

### 7.1 Test B: use 1, NDCG@10 (10 seeds, popularity ties)

| k-means seed | S2 content | S3 SVD | S4 EASE | S7 hybrid |
|---|---|---|---|---|
| 42 | 0.4058 | 0.4941 | **0.5132** | 0.5030 |
| 43 | 0.3824 | 0.5018 | **0.5081** | 0.4947 |
| 44 | 0.4167 | 0.4992 | 0.5005 | **0.5192** |
| 45 | 0.3851 | 0.4819 | **0.5021** | 0.4842 |
| 46 | 0.3885 | **0.5126** | 0.5104 | 0.5108 |
| **mean ± std** | 0.396 ± 0.015 | 0.498 ± 0.011 | **0.507 ± 0.005** | 0.502 ± 0.014 |

| Seeds / ties | S3 SVD | S4 EASE | S7 hybrid |
|---|---|---|---|
| 10 / random | 0.444 | **0.451** | 0.446 |
| 3 / popularity | 0.441 | **0.460** | 0.440 |
| 3 / random | 0.360 | **0.381** | 0.361 |

**Decision rule** (the new source must have a higher mean than S3 and win ≥ 4 of 5 seeds):
- **S4 EASE replaces S3:** +0.0089, wins 4 of 5.
- **S7 hybrid does not:** +0.0045, wins 3 of 5.

### 7.2 Test A (lift, mean over k-means seeds)

| Metric | S2 | S3 | S4 | S7 |
|---|---|---|---|---|
| co-liked L1 / L3 | 1.62 / 2.80 | 1.68 / 2.78 | 1.71 / 2.65 | 1.69 / 2.84 |
| genre L1 / L3 | 1.82 / 2.60 | 1.54 / 2.32 | 1.55 / 2.44 | 1.70 / 2.41 |
| director L1 / L3 | 13.5 / 150.7 | 9.8 / 110.5 | 10.5 / 95.7 | 10.9 / 130.1 |
| decade L1 / L3 | 1.93 / 3.06 | 2.04 / 2.94 | **2.86 / 3.26** | 2.21 / 3.00 |

### 7.3 Findings

1. **All three collaborative sources are far ahead of content** (+0.10 to +0.11). The differences among them are small: S4 − S3 = +0.009, which is less than a tenth of the stage-1 gap. "Collaborative beats content" is the robust finding. "EASE beats SVD" is a narrow pass of the rule.
2. **EASE IDs are the most stable**: std 0.005 over k-means seeds, against 0.011–0.015 for the others. They are also ahead in every other setting (3 seeds, random ties).
3. **The hybrid did not add accuracy** on these users, but it gives an ID to all 1,957 films, including the 408 with no ratings. That matters for the new-item test, not for this one, where unrated films are never in the answer key.
4. **EASE groups films by era most strongly** (decade lift 2.86 at L1, against about 2 for the others). This is plausible: people who like films of the same era often like them together, and EASE weights co-liking directly.
5. **Consequence (pre-registered):** S4 gets the LTR test (stage 2c), alongside the S3 test (2a).

---

## 8. Stage 2a: S3 SVD IDs as LTR v2 features (use 2), 2026-10-29

**Script:** `ltr_sid_sources.py 10|3 S3`; outputs in `output/experiments/ltr_sid_sources/S3/`.

**Set-up:**
- the harness and features of doc 27 §7, with S3 IDs;
- G9 = share of seeds that share the candidate's first 1/2/3 tokens, mean over k-means seeds 42–46;
- cross-fitted: the simulated users of each half use IDs from the other half, and validation users use IDs from all warm users;
- R7 reproduced LTR v2 exactly (0.6686 / 0.6227). ✔

**Pilot:**
- the features have the same distribution on validation and simulated users (`sid_share_l1` non-zero for 80–81% of candidates with 3 seeds);
- within-user Spearman of `sid_share_l1` is 0.35–0.44 with R7's `svd_mean` and 0.40–0.47 with `cf_mean`.

| NDCG@10, ensemble | 10 seeds, held-out (1,816) | 10 seeds, validation (203) | 3 seeds, held-out | 3 seeds, validation |
|---|---|---|---|---|
| R7 (LTR v2) | 0.6759 | 0.6686 | 0.6265 | 0.6227 |
| R11 = R7 + S3 IDs | 0.6751 | 0.6644 | 0.6267 | 0.6171 |
| Gain (p) | −0.0008 (0.45) | −0.0042 (0.52) | +0.0001 (0.43) | −0.0056 (0.17) |

**Decision:** not adopted with 3 or 10 seeds.

**Findings:**
- This is the same outcome as the content IDs in doc 27 §7 (−0.001 / −0.006), and it was the expectation written down beforehand.
- R7 already has the full SVD similarity (`svd_mean`, `svd_max`, `svd_score`) and item-CF. The IDs are a compressed copy of that.
- So the stage-1 gain of collaborative IDs is a gain **in use 1** (IDs as the only signal). It does not carry over to a ranker that already has the underlying vectors.

---

## 9. Stage 2c: S4 EASE IDs as LTR v2 features, 2026-10-29

Pre-registered in addition (2) of `preregistration.md` before running. Same as 2a, with S4 IDs (`ltr_sid_sources.py 10|3 S4`; outputs in `output/experiments/ltr_sid_sources/S4/`).

- R7 reproduced LTR v2 exactly. ✔
- The features have the same distribution on both user groups (with 10 seeds, `sid_share_l1` is non-zero for 96% of simulated and 94% of validation candidates).

| NDCG@10, ensemble | 10 seeds, held-out | 10 seeds, validation | 3 seeds, held-out | 3 seeds, validation |
|---|---|---|---|---|
| R7 (LTR v2) | 0.6759 | 0.6686 | 0.6265 | 0.6227 |
| R11 = R7 + S4 IDs | 0.6753 | 0.6594 | 0.6265 | 0.6226 |
| Gain (p) | −0.0007 (0.59) | −0.0093 (0.040) | −0.0000 (0.57) | −0.0001 (0.70) |

**Decision:** not adopted with 3 or 10 seeds.

- With 10 seeds the validation users are slightly worse (67 better, 92 worse, p = 0.04 uncorrected, descriptive only).
- This repeats doc 27 §7: extra, partly redundant features let the trees fit noise that does not transfer to validation users.

## 10. Overall conclusion of the source comparison

| Use | Content IDs (S2) | Collaborative IDs (S3 SVD / S4 EASE) |
|---|---|---|
| Use 1: IDs as the only signal (simulated users, 10 seeds) | 0.396 | 0.498 / **0.507** (+0.10 to +0.11, all seeds) |
| Use 2: added to LTR v2 (validation, 10 / 3 seeds) | −0.006 / +0.004 (doc 27) | S3: −0.004 / −0.006; S4: −0.009 / −0.000 |

1. **The source of the vectors matters a lot for the IDs themselves.** Collaborative IDs are about 25% better than content IDs as a similarity measure. This replicates Penha et al. (2025) on our data.
2. **It does not matter for the final system.** No kind of semantic ID improves LTR v2, because the ranker already has the full vectors the IDs are compressed from.
3. **Semantic IDs alone stay below popularity (0.559) and item-CF (0.614)** on the same users, even from the best source.
4. **The remaining case for semantic IDs is coverage:** films with no or few ratings. The hybrid (S7) gives every film an ID at no loss in use 1 (0.502 vs S3's 0.498). This is what the new-item test should measure (doc 29 §4).

### Safe statements for the TDK report
- "Semantic IDs built from collaborative vectors (SVD or EASE) outperformed IDs built from LLM-enriched content embeddings by 0.10–0.11 NDCG@10 when used as a similarity measure (5 of 5 k-means seeds, 9,084 simulated users), consistent with Penha et al. (2025)."
- "Adding either kind of semantic-ID feature to the learned ranker gave no improvement (all changes between −0.009 and +0.004 NDCG@10, none meeting the pre-registered +0.005 threshold)."
- "A content–collaborative hybrid gave every film an ID, including the 408 without ratings, at no loss in accuracy."
- Avoid: "EASE IDs are better than SVD IDs" as a strong claim (+0.009, a narrow pass), and any claim about test users.
