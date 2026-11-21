# Pre-registration: semantic IDs from different vector sources (stage 1, screening)

Written 2026-10-29, **before** any run of `sid_vector_sources.py`. Not to be changed after the first run;
later additions go in a new dated section below.

## Question
Do semantic IDs built from **collaborative** vectors (SVD of the warm users' ratings) recommend better than
IDs built from **content** vectors (embeddinggemma + LLM labels, the IDs of docs 27-28)?
Motivation: Penha et al. (RecSys 2025, MovieLens 25M) found CF-based IDs far better for recommendation (doc 29).

## Sources (stage 1)
- **S2 content:** `movies_enriched_embeddinggemma_v2` (as in docs 27-28). Uses no ratings.
- **S3 SVD:** film vectors U·s (rank 64) of the user-centred rating matrix, exactly the SVD of
  `LightGBM_XBoost_rerank_experiment.latent_tables` (eigh of M Mᵀ), restricted to films rated in that set
  of ratings. Uses ratings, so it is **cross-fitted**: simulated users of half A use IDs built from half B's
  ratings and vice versa; test A (no users) uses IDs built from all warm users.

## Fixed settings
3 levels × 32 codes + collision token, faiss k-means (niter 25, nredo 10), k-means seeds 42, 43, 44, 45, 46.
Vectors normalised to length 1 (`Semantic_IDs.set_vectors`). Flat prefix weights (1, 1, 1).

## Tests
- **A, meaningfulness** (IDs from all warm users, both sources on the same 1,549 pool films):
  genre / director / decade lift (`meaningfulness()`), and **co-liked lift**: mean P(j liked | i liked) over
  film pairs sharing the first L tokens, divided by the mean over all pairs (co-visitation from all warm users).
  Descriptive only, no decision.
- **B, use 1** (prefix similarity, `Semantic_IDs.recommend`) on the 9,084 simulated warm users (halves A/B,
  seed 42), 10 and 3 seeds, NDCG@10. Candidates = pool films that have an ID; ties broken by popularity of
  the other half (main) and at random (rng 0, secondary).
  References on the same users: popularity, item-CF (mean), content embedding (max), SVD vectors directly
  (cosine, max over seeds; cross-fitted).

## Check before trusting anything
S2 with k-means seed 42, popularity ties must reproduce doc 27 §4a: 0.4058 (10 seeds) and 0.3614 (3 seeds),
within 1e-4. If not, stop.

## Decision rule (stage 1)
S3 is called **better than S2** if, on test B with 10 seeds and popularity ties, its mean NDCG@10 over the 5
k-means seeds is higher **and** it beats S2 in at least 4 of the 5 seeds (paired by seed).
- If S3 is better: stage 2 = S3 in LTR v2 (use 2, L1 harness, +0.005 rule on both user groups), and add
  EASE (S4) and content+SVD hybrid (S7) to the screening.
- If not: still screen S4 and S7 (one more round), then stop the source comparison.
3 seeds and random ties are reported but do not decide.

The validation and test users are not used in stage 1.

## Outputs (`output/experiments/sid_vector_sources/`)
`vectors/`, `ids/`, `runs/` (one file per finished run; finished runs are skipped on restart),
`results.csv` (one row per run and metric), `summary.csv` (mean / std / min / max over k-means seeds).

---

## Addition 2026-10-29: stage 2 (written after stage 1, before any stage-2 run)

Stage 1 result: S3 SVD better than S2 content (+0.102, 5 of 5 seeds; doc 30). Per the rule above, stage 2 is:

### 2a. S3 IDs as LTR v2 features (use 2) — script `ltr_sid_sources.py`
- Harness: the L1 harness of doc 27 §7 / `ltr_semantic_id_experiment.py`, unchanged (tuned XGBoost + LightGBM
  ensemble, step-4 parameters, Hybrid E top 30, held-out simulated users = 20% seed 42, then the 203
  validation users with the model trained on all simulated users).
- New group G9 = `sid3_share_l1/l2/l3`: share of the user's seeds that share the candidate's first 1/2/3
  S3 tokens, **mean over k-means seeds 42-46**. Exactly the G8 features of doc 27 §7, with S3 IDs.
- Cross-fitted: simulated users of half A use the S3 IDs fitted on half B (`ids/S3_B_*`) and vice versa;
  validation users use the S3 IDs fitted on all warm users (`ids/S3_all-pool_*`, 1,547 films).
  A seed or candidate without an ID matches nothing (counts as 0, stays in the denominator).
- Feature sets: R7 (LTR v2) and **R11 = R7 + G9** (decides). Check first: R7 reproduces 0.6686 / 0.6227.
- **Decision rule (doc 24):** adopt only if R11 gains >= +0.005 over R7 on BOTH user groups (held-out
  and validation), for that seed count. 10 and 3 seeds are separate decisions.
- Expectation written down beforehand: R7 already has `svd_mean/svd_max/svd_score` from the same SVD,
  so a small or zero gain is the likely outcome (as for G8 in doc 27 §7).

### 2b. Two more sources, screened exactly as in stage 1 (test A + test B, 5 k-means seeds)
- **S4 EASE:** column j of the EASE weight matrix B (lambda 20,000, the half-size value tuned for 10 seeds,
  `ease_tuning.py`), fitted on the other half's ratings (cross-fitted) or all warm users (test A).
  Films with an all-zero column (never liked in that part) are dropped and get no ID.
- **S7 hybrid:** [content / |content| ; svd / |svd|] / sqrt(2) (content block = S2 vectors, SVD block =
  S3 vectors of the same part). For films with no SVD vector in that part, the SVD block is predicted from
  the content vector by ridge regression (alpha 1.0) fitted on the films that have both. So S7 covers
  all pool films. This coverage advantage is part of the method and is reported, not removed.
- **Decision:** a new source replaces S3 as the best source if, on test B with 10 seeds and popularity
  ties, its mean is higher than S3's AND it beats S3 in >= 4 of 5 k-means seeds. If it does, it gets
  the same LTR test as 2a (stage 2c, a new dated addition first). If neither does, the source comparison
  stops here, as fixed above.

Validation users are used only in 2a (and 2c), as in doc 27 §7. Test users are not used.

---

## Addition 2026-10-29 (2): stage 2c (written after the stage-2b screening, before any S4 LTR run)

Stage 2b result: S4 EASE replaces S3 by the rule (+0.0089 over S3, 4 of 5 k-means seeds); S7 does not
(+0.0045, 3 of 5). Note the margin is small (stage 1's S3 - S2 gap was +0.102).

**2c = 2a with S4 IDs**, nothing else changed: `python ltr_sid_sources.py {10,3} S4`.
G9 from `ids/S4_{A,B,all-pool}_k32_s42-46.npz` (cross-fitted as in 2a; validation users use the IDs
fitted on all warm users, EASE lambda 20,000 as in the screening, not rescaled for the larger data).
Same decision rule: adopt R11 only if it gains >= +0.005 over R7 on BOTH user groups.
2a (S3) and 2c (S4) are separate decisions; if both pass, the larger mean gain on validation users is
reported as the result, and the other as descriptive.
