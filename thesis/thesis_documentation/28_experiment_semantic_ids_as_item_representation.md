# 28 — Semantic IDs as the item representation (use 3: two-tower and generative)

**Status:** validation run complete (2026-10-29), 15 of 15 trainings.
**Script:** `semantic_id_models.py` (modes `pilot`, `tune`, `val`).
**Outputs:** `output/experiments/semantic_id_models/`. This folder holds:
- `tune.csv` and `tune_30epochs.csv`;
- `chosen_epochs.json`;
- `val_partial/`, `val_summary.csv`, `val_tests.csv` and `val_per_user_per_run.csv`.

**Related:**
- doc 19 (reading list);
- doc 27 (building the IDs, uses 1 and 2);
- doc 20 (NCF/SVD for new users).

## 1. Question

Docs 27 §3 and §7 used semantic IDs as a **similarity measure** (use 1) and as **ranker features** (use 2). Neither use helped.

The literature mostly uses semantic IDs differently: as the **item representation inside a learned model**, in place of an arbitrary ID. Two set-ups are standard:
- Singh et al. 2024 use them as the item embedding of a ranking model;
- TIGER (Rajput et al. 2023) uses them as the output vocabulary of a generative model.

**Question:** if everything else in a model is kept fixed, does replacing the atomic film ID with the semantic ID (or adding it alongside) improve recommendations for new users?

## 2. Set-up

### 2.1 Models

Each model has a switchable item representation. Everything else is identical between the variants.

| Model | Representation | How a film is represented |
|---|---|---|
| **Two-tower** (NCF family, Singh et al.) | atomic | one learned embedding per film |
| | sid | the sum of the learned embeddings of its 3 semantic-ID tokens (one table per level). Films with the same 3 tokens get the same vector. |
| | both | atomic + sid (the combination Singh et al. recommend) |
| **Generative** (TIGER-style) | atomic | a GRU reads the seeds in time order; a softmax over the 1,549 pool films predicts the next film |
| | sid | the input is the seeds' 4 ID tokens each; a GRU decoder generates the target's 4 tokens one at a time; beam search (width 30) is restricted to the IDs of pool films |

- In the two-tower model, the user vector is an MLP applied to the mean of the seed films' vectors.
- A film's score is the dot product of the user vector and the film vector, with a softmax over the pool.
- New users are scored from their seeds alone (no fold-in), like every other cold-start method in this work.

**Semantic IDs:** `Semantic_IDs` from recomm.py:
- enriched embedding (embeddinggemma, v2 text);
- 32 codes per level, 3 levels, plus a collision token;
- the setting chosen in doc 27.

**Training data:** the warm users' liked films in the pool, oldest first.
- One example is 3–10 consecutive liked films (the input) and one liked film after them (the target).
- Each user gives 10 examples per epoch.

**Fixed settings:** dimension 64 (two-tower) and 128 (GRU), Adam with lr 1e-3, batch 512.

### 2.2 Protocol (fixed before running)

1. **Epochs**, chosen on simulated users, never on validation users:
   - each variant was trained on warm half A and scored every 5 epochs on 2,000 simulated users of half B (3 and 10 seeds averaged);
   - the first grid went to 30 epochs, but both atomic variants were still improving at epoch 30 (the edge of the grid). That would have favoured the sid variants, which peaked at 15–25 epochs, so the grid was extended to 60 before validation (comment in the script).
2. **Validation:** 3 training runs per variant.
   - run r uses torch seed r and k-means seed 42 + r, so the IDs change between runs as well;
   - each run is trained on all warm users with the chosen epochs and scored on the 203 validation users;
   - per-user NDCG@10 is averaged over the 3 runs;
   - each variant is compared with the atomic version of the same model using a paired Wilcoxon test.
3. **Tail recall:** Recall@10 counted on the least-liked third of the pool only (515 films). This is where semantic IDs are expected to help most.

### 2.3 Chosen epochs and tuning scores (simulated users, half B)

| Model | Representation | Epochs | NDCG@10 (mean over 3 and 10 seeds) |
|---|---|---|---|
| two-tower | atomic | 30 | 0.502 |
| two-tower | sid | 15 | 0.455 |
| two-tower | both | 15 | 0.508 |
| generative | atomic | 30 | 0.469 |
| generative | sid | 50 | 0.477 |

## 3. Results (203 validation users, mean of 3 training runs)

| Seeds | Model | Repr. | NDCG@10 | Lowest / highest run | Tail recall | Tail share |
|---|---|---|---|---|---|---|
| 10 | two-tower | atomic | 0.5258 | 0.516 / 0.532 | 0.000 | 0.000 |
| 10 | two-tower | sid | 0.3932 | 0.347 / 0.469 | 0.002 | 0.007 |
| 10 | two-tower | both | 0.5269 | 0.504 / 0.543 | 0.000 | 0.000 |
| 10 | generative | atomic | 0.5231 | 0.510 / 0.531 | 0.000 | 0.002 |
| 10 | generative | sid | 0.4791 | 0.455 / 0.492 | 0.000 | 0.000 |
| 3 | two-tower | atomic | 0.4718 | 0.459 / 0.482 | 0.000 | 0.001 |
| 3 | two-tower | sid | 0.3601 | 0.298 / 0.435 | 0.001 | 0.007 |
| 3 | two-tower | both | 0.4800 | 0.462 / 0.493 | 0.000 | 0.000 |
| 3 | generative | atomic | 0.4703 | 0.464 / 0.473 | 0.000 | 0.002 |
| 3 | generative | sid | 0.4500 | 0.421 / 0.471 | 0.000 | 0.000 |

Tail share is the share of recommended films that are tail films.

**Paired Wilcoxon against the atomic version of the same model.** Six tests. The Bonferroni-corrected p is 6 × the raw p.

| Seeds | Model | Comparison | Difference | Users better / worse | p (raw) | p (Bonf. ×6) |
|---|---|---|---|---|---|---|
| 10 | two-tower | sid vs atomic | **−0.133** | 55 / 148 | 3e-16 | 2e-15 |
| 10 | two-tower | both vs atomic | +0.001 | 93 / 109 | 0.71 | 1 |
| 10 | generative | sid vs atomic | **−0.044** | 88 / 114 | 0.0005 | 0.003 |
| 3 | two-tower | sid vs atomic | **−0.112** | 59 / 143 | 1e-12 | 8e-12 |
| 3 | two-tower | both vs atomic | +0.008 | 103 / 99 | 0.36 | 1 |
| 3 | generative | sid vs atomic | −0.020 | 92 / 111 | 0.065 | 0.39 |

**In context** (same 203 users, `output/experiments/final_comparison/summary_val.csv`):

| Method | NDCG@10, 10 seeds | NDCG@10, 3 seeds |
|---|---|---|
| LTR v2 | 0.669 | 0.623 |
| EASE | 0.651 | 0.605 |
| Hybrid E | 0.629 | 0.584 |
| NCF (NeuMF, fold-in) | 0.598 | 0.554 |
| Popularity | 0.563 | 0.563 |
| best model of this doc (two-tower both) | 0.527 | 0.480 |

### 3.1 Precision, recall and hit rate (added 2026-10-29, from retrained models)

**Script:** `semantic_id_models.py metrics`.
**Outputs:** `output/experiments/semantic_id_models/val_metrics_summary.csv` (per run in `val_metrics_partial/`).

**Reproducibility note:** the original runs did not save their recommendation lists, so all 30 models were retrained with the same seeds, epochs and data. PyTorch training on CPU is not bit-reproducible:
- **generative atomic** reproduced exactly;
- **two-tower and generative sid** came out slightly different, within 0.007 NDCG of the originals and inside the original lowest/highest-run range.

**The NDCG table and tests above remain the official results.** The table below is for comparing the extra metrics. Future runs should set `torch.use_deterministic_algorithms(True)`.

| Seeds | Model | Repr. | NDCG@10 (retrained) | Precision@10 | Recall@10 | Hit rate@10 |
|---|---|---|---|---|---|---|
| 10 | two-tower | atomic | 0.525 | 0.515 | 0.0514 | 0.972 |
| 10 | two-tower | sid | 0.392 | 0.382 | 0.0375 | 0.962 |
| 10 | two-tower | both | **0.529** | **0.523** | 0.0513 | 0.966 |
| 10 | generative | atomic | 0.523 | 0.513 | **0.0519** | 0.966 |
| 10 | generative | sid | 0.483 | 0.475 | 0.0485 | 0.982 |
| 3 | two-tower | atomic | 0.470 | 0.465 | 0.0450 | 0.966 |
| 3 | two-tower | sid | 0.359 | 0.351 | 0.0339 | 0.931 |
| 3 | two-tower | both | **0.480** | **0.475** | **0.0454** | 0.967 |
| 3 | generative | atomic | 0.470 | 0.467 | 0.0445 | 0.969 |
| 3 | generative | sid | 0.457 | 0.444 | 0.0425 | 0.964 |

**Findings:**
- **Precision and recall rank the models exactly as NDCG does.** In both two-tower and generative models, sid < atomic ≈ both, so the extra metrics show no hidden strength of the semantic-ID versions.
- **The sid gap is the same size in precision as in NDCG**, for example two-tower 10 seeds: −0.133 precision vs −0.133 NDCG. The sid models pick fewer liked films; the loss is not just in their ordering.
- **Hit rate is saturated** (0.93–0.98: almost every user gets at least one liked film) and does not separate the models.
- **Recall@10 is capped near 0.08** because users have about 125 liked films. Its differences are small in absolute size but follow precision.

## 4. Findings

1. **Semantic IDs alone are worse than atomic IDs in both models.**
   - Two-tower: −0.13 (10 seeds) and −0.11 (3 seeds), both significant after correction.
   - Generative: −0.044 (10 seeds, significant) and −0.020 (3 seeds, not significant).
   - This is consistent with doc 27: the ID loses information. Films that share all 3 tokens get the same vector in the two-tower sid variant, so the model cannot tell them apart.
2. **Adding semantic IDs to atomic IDs ("both") changes nothing measurable:** +0.001 and +0.008, both p > 0.3. Singh et al.'s recommended combination does not help here.
3. **The generative model loses less from semantic IDs than the two-tower model** (−0.04 vs −0.13 at 10 seeds). This is probably because:
   - its decoder gets the collision token, so every pool film has a distinct ID;
   - beam search over a trie restricted to the pool can still reach any film.
4. **The tuning lead did not hold.** On the simulated users, generative sid was slightly ahead of generative atomic (0.477 vs 0.469). On the validation users it is behind (−0.044 at 10 seeds). This is the same pattern as earlier pilots: small leads during tuning do not survive.
5. **Semantic IDs make training less stable.** The two-tower sid variant varies by up to 0.12 NDCG between runs (0.347 to 0.469 at 10 seeds), against about 0.02 for atomic. Part of this comes from the k-means seed, which changes the IDs themselves (doc 27 §4a).
6. **The long-tail effect could not be tested here.** Every variant recommends almost no tail films (tail share ≤ 0.7%), so tail recall is about 0 for all. The advantage the literature reports for semantic IDs (new and long-tail items) needs a set-up where such items can be recommended at all: see §6, the new-item test.
7. **All of these models are weak recommenders in absolute terms.** The best (0.527) is below popularity (0.563) and far below Hybrid E, EASE and LTR v2.
   - The comparison between representations is still fair, because model, data, training and protocol are identical apart from the item representation.
   - But the result is "semantic IDs do not help *these* models". It is not "semantic IDs cannot help a strong model".
   - A likely reason the models are weak: each user gives only 10 next-film examples per epoch, and the models have no popularity term. Popularity alone scores 0.563 on these users.

## 5. Safe statements for the TDK report

- "Using semantic IDs instead of atomic film IDs as the item representation lowered NDCG@10 in both a two-tower model (by 0.11–0.13, p < 10⁻¹¹ after Bonferroni correction) and a TIGER-style generative model (by 0.02–0.04; significant with 10 seeds only), on 203 validation users."
- "Combining semantic and atomic IDs, as recommended by Singh et al., gave no measurable change (+0.001 and +0.008, p > 0.3)."
- "Both models were below the popularity baseline, so these results show that semantic IDs did not help these models. They do not show that semantic IDs cannot help a stronger model."
- "The long-tail benefit reported in the literature could not be tested in this set-up, because none of the models recommended tail films."
- Avoid: "semantic IDs don't work", "generative retrieval fails", or any comparison with TIGER's numbers (different data and scale).

## 6. Next (to decide)

The user plans to test more semantic-ID methods and models, including generative retrieval, before chapter 8 is written. Options, cheapest first:

1. **New-item test** (planned). Hide about 10% of pool films from training. Check whether the sid variants can recommend them and whether they hit the answer key. Atomic IDs cannot represent unseen films at all, so this is where semantic IDs have a structural advantage.
2. **Stronger generative baseline before comparing representations.** For example:
   - a Transformer encoder (SASRec-style) in place of the GRU;
   - more training examples per user;
   - a popularity prior or bias term.

   The aim is to bring the atomic model at least up to popularity level, so the comparison is between serious models.
3. **IDs built differently:**
   - from collaborative embeddings (e.g. EASE or SVD item vectors) in place of content embeddings, or a hybrid of the two (LETTER-style);
   - smaller codebooks (8 or 16 codes) to reduce data sparsity per token.
4. **Semantic IDs for an LLM** (the prompt ladder idea): show the LLM a film's ID group or a cluster name. This needs no training.

Any of these should keep the same protocol: settings chosen on simulated users, 3 runs, validation users, and the decision rules fixed before running.
