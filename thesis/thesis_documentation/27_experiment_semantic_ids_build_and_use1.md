# 27 — Semantic IDs: building, checking, and use 1 (prefix similarity)

Status (2026-10-28): steps 1–3 of doc 18 §3 done on the **simulated warm users only**. Validation users
not used yet. Background and references: doc 19.

## 1. Building the IDs

- **Class:** `Semantic_IDs` in recomm.py (written by the author). Residual k-means, as in OneRec
  (doc 19, paper 10).
  - Input: the content embedding (embeddinggemma, 768-d, normalised).
  - 3 levels: k-means on the vectors, subtract each film's cluster centre, then k-means on what is left.
  - A 4th token separates films with identical 3-token codes.
  - `faiss.Kmeans` (25 iterations, 10 restarts, seed 42). Reproducible: two runs give identical IDs.
- **Implementation note:** sklearn's KMeans crashes on Windows whenever faiss is loaded in the same
  process (threadpoolctl error 0xc06d007f, all import orders tested). faiss's own k-means avoids it.
- **Two embeddings** (both already used in doc 05):
  - base = TMDB text only;
  - enriched = TMDB text + LLM labels.
- **Two codebook sizes:** 16 and 32 codes per level.

## 2. Checks (step 2): `report()` and `meaningfulness()`

| | base 16 | enriched 16 | base 32 | enriched 32 |
|---|---|---|---|---|
| Codes used per level | 16/16 | 16/16 | 32/32 | 32/32 |
| Unique 3-token codes (of 1,957) | 1,190 | 1,209 | 1,683 | 1,644 |
| Largest collision group | 14 | 10 | 6 | 6 |
| Mean residual length after 3 levels (start 1.0) | 0.76 | 0.74 | 0.73 | 0.72 |

**Meaningfulness** = among film pairs sharing the first L tokens, the share with the same main genre,
director or decade, divided by the same share among all pairs ("lift"; 1 = random). Lift at L = 1 / 2 / 3:

| | base 16 | enriched 16 | base 32 | enriched 32 |
|---|---|---|---|---|
| Genre | 2.0 / 2.4 / 2.8 | 1.9 / 2.4 / 2.6 | 1.9 / 2.5 / 2.9 | 2.1 / 2.4 / 2.8 |
| Director | 8 / 33 / 110 | 7 / 26 / 72 | 14 / 59 / 155 | 14 / 45 / 189 |
| Decade | 1.9 / 2.4 / 2.7 | 1.8 / 2.4 / 3.2 | 1.9 / 2.6 / 3.2 | 1.8 / 2.5 / 2.9 |

- **The IDs are meaningful:** lift grows with every level for all three attributes, and franchises stay
  together (Toy Story, Godfather and LOTR share 2–3 tokens). This meets doc 18's success criterion.
- **Base vs enriched:** no consistent difference on these attributes. Expected: genre, director and year
  are already in the TMDB text, and the LLM adds themes, moods and audience, which this check does not
  measure.
- **Caveat:** genre and director are part of the embedded text, so the lift shows the IDs *keep* the
  embedding's information. It is not independent evidence.
- **The code keeps only the broad structure.** Three levels remove only about a quarter of the
  embedding's length (1.0 → 0.72–0.76).

## 3. Use 1: recommending by shared prefix

**Script:** `semantic_id_eval.py`; outputs in `output/experiments/semantic_ids/`.

**Scoring:** for each candidate, the shared prefix length with each seed (0–3, or weighted 1/2/3 per
level), averaged over the seeds.

**Set-up:**
- Candidates: the 1,549-film pool; all 10 seed-pool films never recommended.
- Users: the 9,084 simulated warm users (halves A/B).
- Popularity for tie-breaking and for the reference comes from the other half.

**References**, on the same users with the same code: content embedding (max over seeds, as
"Content" in final_comparison.py), item-CF, popularity.

| NDCG@10 | 10 seeds | 3 seeds |
|---|---|---|
| Item-CF | 0.614 | 0.529 |
| Popularity | 0.559 | 0.559 |
| Content embedding, enriched / base | 0.400 / 0.398 | 0.338 / 0.341 |
| **IDs, enriched, 32, flat, popularity ties (best)** | **0.406** | **0.361** |
| IDs, base, 32, flat, popularity ties | 0.374 | 0.341 |
| IDs, enriched, 16, flat, popularity ties | 0.374 | 0.361 |
| IDs, deep weights (1,2,3), any setting | 0.345–0.400 | 0.331–0.351 |

**Tie-break check** (enriched / base, 32 codes, flat; run after the table above):

| Ties broken by | 10 seeds | 3 seeds |
|---|---|---|
| popularity (other half) | 0.406 / 0.374 | 0.361 / 0.341 |
| catalog order | 0.365 / 0.342 | 0.319 / 0.302 |
| random | **0.309 / 0.284** | **0.265 / 0.245** |

(Catalog order is not random: it correlates with popularity, hence its middle position.)

## 4. Findings

1. **On their own, the IDs are a weaker content signal than the full embedding.**
   - Without popularity (random ties) the best IDs score 0.309, against 0.400 for the embedding they
     came from (10 seeds). The 3-token code throws away detail that matters for recommendation.
   - The apparent edge of the IDs over content (0.406 vs 0.400) comes from the popularity tie-break.
     Many films tie on prefix score, so the tie-break decides much of the top 10.
2. **Flat weights beat deep weights** in every setting. Rewarding a full 3-token match more does not
   help; the level-3 matches are too rare and too specific.
3. **32 codes ≥ 16 codes**, clearly with the enriched embedding at 10 seeds (0.406 vs 0.374).
4. **Enriched vs base IDs: no reliable difference** (see section 4a). With k-means seed 42, enriched beat
   base by +0.02 to +0.03 in every variant, but over 5 k-means seeds the mean gain is only +0.007 and
   enriched wins in 2–3 of 5 seeds. Seed 42 happened to favour enriched.
5. **Like every content method here, the IDs stay far below the collaborative ones** (item-CF 0.614,
   EASE 0.650 on the same users).

## 4a. k-means seed check (`python semantic_id_eval.py seeds`)

32 codes, flat weights, the same 9,084 simulated users. k-means seeds 42–46, for both embeddings. Ties
broken by popularity (other half) or at random (fixed generator), so the comparison is paired.

| Mean over 5 k-means seeds (sd) | base | enriched | enriched − base (min … max) | enriched wins |
|---|---|---|---|---|
| 10 seeds, popularity ties | 0.388 (0.018) | 0.396 (0.015) | +0.008 (−0.034 … +0.045) | 2 / 5 |
| 10 seeds, random ties | 0.304 (0.017) | 0.311 (0.015) | +0.007 (−0.031 … +0.045) | 3 / 5 |
| 3 seeds, popularity ties | 0.346 (0.005) | 0.353 (0.007) | +0.007 (−0.008 … +0.021) | 3 / 5 |
| 3 seeds, random ties | 0.251 (0.011) | 0.258 (0.008) | +0.007 (−0.020 … +0.021) | 3 / 5 |

- **The k-means seed matters more than the embedding.** One seed moves NDCG@10 by up to 0.04 (10 seeds:
  base 0.372–0.417), several times the average enriched − base gap.
- **Enriched vs base is not established.** The sign is positive on average in all four rows, but enriched
  wins only 2–3 of 5 seeds. Treat it as "no reliable difference".
- **The main finding stands:** even the best seed with random ties (0.336) is below the full content
  embedding (0.400). The IDs lose information whichever seed is used.
- **Consequence for later steps:** IDs from a single k-means run are noisy. Results with IDs (use 2,
  validation) should be averaged over several k-means seeds, or the seed must be fixed in advance and
  this variance reported. The "best setting" of section 3 was chosen with seed 42 and is slightly
  optimistic for enriched.

## 5. Safe statements

- "Semantic IDs built by residual k-means preserve the content structure (shared-prefix films are 2–3×
  more likely to share genre or decade and up to 190× more likely to share a director than random
  pairs), but as a stand-alone recommender they lose information relative to the embedding they
  compress (NDCG@10 0.31 vs 0.40 without a popularity tie-break)."
- "IDs from the LLM-enriched and the plain embedding performed alike; the variation between k-means seeds
  (up to 0.04 NDCG@10) was larger than the difference between embeddings (+0.007 on average)."
- Do **not** claim that semantic IDs beat the content embedding (only with a popularity tie-break the
  embedding does not get), nor that the LLM labels improve the IDs (not stable over k-means seeds).

## 6. Next

- ~~Several k-means seeds for enriched vs base~~: done (section 4a). Not stable.
- **Use 2:** prefix-match features in LTR v2. Use 1 suggests a small effect: LTR already has the full
  content similarity.
- **Hybrid IDs** (content + collaborative embedding, doc 19 paper 9): the collaborative signal is what
  content-only IDs miss.
- **Use 3** (generative) only after use 2 and the hybrid IDs.
- **LLM cluster names** and `similar_to` in the constraint layer (doc 25): useful even if the IDs
  are weak as a recommender.
- No validation run yet. Pick one setting on the simulated users (enriched, 32, flat) before running it.

## 7. Use 2: semantic-ID features in LTR v2 (2026-10-28) — not adopted

**Script:** `ltr_semantic_id_experiment.py`; outputs in `output/experiments/ltr_semantic_ids/`.

**Set-up:** the L1 harness of doc 20 §10.1 / doc 24, unchanged; tuned XGBoost + LightGBM, step 4
parameters. Fixed before running:
- **Features (G8):** for each candidate, the share of seeds that share its first 1, 2 and 3 ID tokens.
  IDs from the enriched embedding, 32 codes, 3 levels.
- Each feature is the **mean over 5 k-means seeds** (42–46). The seed-42-only version is descriptive.
- No cross-fitting needed: the IDs use no ratings.
- **Decision rule (doc 24):** R10 = R7 + G8 needs ≥ +0.005 on both user groups. R7 reproduced LTR v2
  exactly.

**Pilot checks:**
- Same feature distribution on validation and simulated users.
- `sid_share_l1` is non-zero for 89% of candidates with 10 seeds (61% with 3).
- `sid_share_l1` correlates 0.45–0.53 (within-user Spearman) with R7's embedding features, 0.36–0.43
  with genre overlap, and about 0.1–0.2 with director match.

| NDCG@10, ensemble | 10 seeds, held-out (1,816) | 10 seeds, validation (203) | 3 seeds, held-out | 3 seeds, validation |
|---|---|---|---|---|
| R7 (LTR v2) | 0.6759 | 0.6686 | 0.6265 | 0.6227 |
| R10 = R7 + SID (5-seed mean) | 0.6749 | 0.6624 | 0.6261 | 0.6270 |
| Gain (p Bonf.) | −0.0011 (0.69) | −0.0062 (0.11) | −0.0004 (1.0) | +0.0044 (0.94) |
| *R10, seed 42 only* | *0.6761 (+0.0002)* | *0.6579 (−0.0108, p 0.014)* | *0.6283 (+0.0018)* | *0.6220 (−0.0007)* |

**Decision:** not adopted with 3 or 10 seeds. LTR v2 stays the final model.

**Findings:**
- The semantic-ID features add nothing the ranker can use. On the 1,816 held-out users the change is
  −0.001 to +0.002, with users helped and users hurt almost equal (e.g. 685 vs 714).
- With 10 seeds the validation users are slightly worse: −0.006 for the averaged features, and −0.011
  for the single-seed version (68 users better, 97 worse; p = 0.014 after Bonferroni, descriptive only).
  The extra, partly redundant features seem to let the trees fit noise that does not transfer.
- Averaging over k-means seeds was the safer choice: the single-seed features were worse on validation
  in both seed counts.
- This fits use 1: the IDs are a compressed copy of the content embedding, and R7 already has the full
  embedding. It also matches EASE, NCF and user-kNN as features (docs 20, 26): LTR v2 is hard to improve
  with one more signal of a kind it already has.

**Safe statement:** "Adding semantic-ID prefix features to the learned ranker did not help (−0.001 /
−0.006 NDCG@10 with 10 seeds, −0.000 / +0.004 with 3; held-out simulated / validation users; none
significant). The features largely repeat the content similarity the ranker already uses."

**Remaining semantic-ID uses**, where the IDs replace an item representation rather than add a feature:
- **NCF with ID-token embeddings** (atomic id vs semantic ID vs both, as Singh et al.). Report also
  by film popularity, since the literature's gains are on sparse items.
- **Use 3 (generative):** the same sequence model with film ids vs ID tokens as the vocabulary (TIGER).
