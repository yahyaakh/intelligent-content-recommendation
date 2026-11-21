# 19 — Semantic IDs: reading list and background

Status (2026-10-28): reading list only; nothing implemented yet. The plan is in doc 18 §3.
Every entry was checked on arXiv or the publisher page (title, authors, venue). The "what it says" notes
come from the abstracts and, where stated, the paper itself. Read the paper before citing any number.

## 1. The idea in five sentences

- A normal recommender gives every film a random ID (`movie_id = 278`). The ID says nothing about the film,
  so a model can only learn about it from interactions, and new films with few ratings stay unknown.
- A **semantic ID** replaces it with a short code of a few tokens, e.g. `(12, 5, 33)`, built from the
  film's content embedding, so that **similar films share a prefix**: all films starting with `12` are
  close in embedding space; `(12, 5, …)` are closer still.
- The code is built by **residual quantisation**: cluster the embeddings (level 1), subtract each film's
  cluster centre, cluster what is left (level 2), and so on. Each level refines the previous one.
- A sequence model can then **generate** the next film's ID token by token (generative retrieval, TIGER),
  or a ranking model can use the IDs as features (YouTube).
- Two ways to build the code: **RQ-VAE** (a small neural network learns the codebooks) or **residual
  k-means** (plain k-means at each level, no training). Doc 18 planned residual k-means.

## 2. Reading order

### Step 1: the basic method (read fully)

1. **Rajput et al., "Recommender Systems with Generative Retrieval" (TIGER), NeurIPS 2023.**
   [arXiv 2305.05065](https://arxiv.org/abs/2305.05065) · [HTML](https://ar5iv.labs.arxiv.org/html/2305.05065)
   - The paper that made semantic IDs standard in recommendation. Read §3 (semantic ID generation) and
     the ablations.
   - Set-up (checked in the paper): Sentence-T5 embeddings (768-d) of title, price, brand and category;
     RQ-VAE with **3 levels × 256 codes**; a **4th token** separates films that share the first three;
     a Transformer encoder-decoder predicts the next item's ID from the history.
   - Datasets: Amazon Beauty, Sports and Outdoors, Toys and Games (**not MovieLens**).
   - Baselines: GRU4Rec, Caser, HGN, SASRec, BERT4Rec, FDSA, S3-Rec, P5.
   - Also reports that RQ-VAE IDs beat random IDs and hashing (LSH) IDs in its ablation, and that the
     model can recommend items not seen in training.

2. **Singh et al., "Better Generalization with Semantic IDs: A Case Study in Ranking for
   Recommendations", Google, RecSys 2024.** [arXiv 2306.08121](https://arxiv.org/abs/2306.08121)
   - Semantic IDs used **inside a ranking model** (YouTube) instead of random IDs, not for generation.
   - Main claim: better results on new and long-tail items without hurting the overall model.
   - Closest to our "use 2" (semantic IDs as features in LTR v2) and to our cold-start question.

### Step 2: where it comes from (skim)

3. **Tay et al., "Transformer Memory as a Differentiable Search Index" (DSI), NeurIPS 2022.**
   [arXiv 2202.06991](https://arxiv.org/abs/2202.06991)
   - Origin of "generate the ID instead of retrieving it", for documents. Its "semantic docids" are built
     with **hierarchical k-means** (10 clusters per level, recurse until ≤ 100 documents): the simplest
     version of the idea and a good baseline for us.
4. **van den Oord et al., "Neural Discrete Representation Learning" (VQ-VAE), NeurIPS 2017.**
   [arXiv 1711.00937](https://arxiv.org/abs/1711.00937) — the vector-quantised autoencoder behind RQ-VAE.
5. **Lee et al., "Autoregressive Image Generation using Residual Quantization" (RQ-VAE), CVPR 2022.**
   [arXiv 2203.01941](https://arxiv.org/abs/2203.01941) — introduces residual quantisation with a VAE.
   Only the method section is needed.

### Step 3: semantic IDs with LLMs (our thesis angle)

6. **Hua et al., "How to Index Item IDs for Recommendation Foundation Models", SIGIR-AP 2023.**
   [arXiv 2305.06569](https://arxiv.org/abs/2305.06569) · code: [Wenyueh/LLM-RecSys-ID](https://github.com/Wenyueh/LLM-RecSys-ID)
   - Compares ways to name items for an LLM recommender (P5): random, title, sequential, collaborative,
     semantic and hybrid indexing. Says the choice of ID has a large effect. Good for the related-work
     argument "why not just use titles".
7. **Zheng et al., "Adapting Large Language Models by Integrating Collaborative Semantics for
   Recommendation" (LC-Rec), ICDE 2024.** [arXiv 2311.09049](https://arxiv.org/abs/2311.09049) ·
   code: [RUCAIBox/LC-Rec](https://github.com/RUCAIBox/LC-Rec)
   - Fine-tunes an LLM (LLaMA-7B) to read and generate quantised item IDs. Shows what "the LLM with
     semantic IDs" (doc 18 use 4) looks like at full scale; far beyond our compute, cite as context.
8. **Eugene Yan, "Training an LLM-RecSys Hybrid for Steerable Recs with Semantic IDs", blog, Sep 2025.**
   [eugeneyan.com](https://www.eugeneyan.com) · code: [eugeneyan/semantic-ids-llm](https://github.com/eugeneyan/semantic-ids-llm)
   - Readable, practical walk-through: semantic ID tokens added to an LLM vocabulary (Qwen3-8B), Amazon
     video games. Best "intuition first" source. A blog, not peer-reviewed: don't cite numbers from it.

### Step 4: design choices and what goes wrong (read before implementing)

9. **Wang et al., "Learnable Item Tokenization for Generative Recommendation" (LETTER), CIKM 2024.**
   [arXiv 2405.07314](https://arxiv.org/abs/2405.07314)
   - Problems with plain RQ-VAE IDs: they ignore collaborative signals and use the codes unevenly
     ("code assignment bias"). Adds a collaborative alignment loss and a diversity loss.
10. **Zhou et al., "OneRec Technical Report", Kuaishou, 2025.** [arXiv 2506.13695](https://arxiv.org/abs/2506.13695)
    - Industrial system that uses **residual k-means (RQ-Kmeans), not RQ-VAE**, with 3 coarse-to-fine
      levels. Reason given (§2.1.2, Table 11): better reconstruction, codebook use and balance than
      RQ-VAE in their setting. This is the strongest published support for our residual k-means choice.
11. **Chen et al., "What Makes a Good Semantic ID for Generative Recommendation? A Reproducibility
    Study", SIGIR-AP 2026.** [arXiv 2609.24430](https://arxiv.org/abs/2609.24430)
    - Compares semantic ID designs on one backbone (TIGER). Abstract: **no single design is best
      everywhere**; RQ-VAE- and OPQ-based IDs behave inconsistently across datasets; even codebook use is
      a useful check but not enough for good recommendations; longer codes do not always help.
    - Useful for the thesis: supports reporting the ID-quality checks and not over-claiming.
12. **Ju et al., "Generative Recommendation with Semantic IDs: A Practitioner's Handbook" (GRID), Snap,
    CIKM 2025.** [arXiv 2507.22224](https://arxiv.org/abs/2507.22224) · code: [snap-research/GRID](https://github.com/snap-research/GRID)
    - Open-source framework and ablations; says many small design choices change the results a lot.

### Step 5: overview (optional)

13. **"A Survey on Generative Recommendation: Data, Model, and Tasks", 2025.**
    [arXiv 2510.27157](https://arxiv.org/abs/2510.27157) — groups item tokenisation into ID-based,
    text-based and codebook-based (semantic IDs). Use for the related-work chapter. (Only the abstract
    was checked.)

## 3. Code to look at

| Repository | What | Notes |
|---|---|---|
| [EdoardoBotta/RQ-VAE-Recommender](https://github.com/EdoardoBotta/RQ-VAE-Recommender) | PyTorch TIGER: RQ-VAE tokenizer + Transformer | MIT licence. Includes **MovieLens 1M and 32M, but only for training the tokenizer**; the generative part is set up for Amazon. Good reference for the RQ-VAE code (k-means initialisation). |
| [snap-research/GRID](https://github.com/snap-research/GRID) | Modular framework for semantic-ID generative recommendation | Paper 12. |
| [eugeneyan/semantic-ids-llm](https://github.com/eugeneyan/semantic-ids-llm) | Semantic IDs in an LLM vocabulary | Paper/blog 8. |
| [RUCAIBox/LC-Rec](https://github.com/RUCAIBox/LC-Rec) | LLM + quantised item IDs | Paper 7. |

## 4. What this means for our plan (doc 18 §3)

- **Residual k-means is a defensible choice.** It needs no training, is reproducible (fixed seed), and
  OneRec (paper 10) chose it over RQ-VAE. The reproducibility study (paper 11) says no design wins
  everywhere, so we should not claim ours is optimal; RQ-VAE can be an optional comparison.
- **Our catalog is tiny compared with the papers.** TIGER uses 3 × 256 codes for tens of thousands of
  items. With 1,957 films, 256 codes at level 1 would leave about 8 films per code, and levels 2–3 would
  be almost empty. We need small codebooks (for example 3 levels × 8–32 codes) and should report codebook
  use and collisions (paper 11's checks). The exact sizes are a design decision to test, not taken from
  the papers.
- **Embedding to quantise:** doc 18 says the best content embedding (embeddinggemma + labels, doc 05).
  TIGER used Sentence-T5 on product text; the principle is the same.
- **Collisions:** TIGER's 4th token (0, 1, … among films with the same first three tokens) is the
  standard fix.
- **Uses in order of cost:** (1) prefix similarity in cold start, (2) prefix-match features in LTR v2
  (like paper 2), (3) small generative model (like paper 1, stretch), (4) LLM with IDs (papers 7–8,
  optional).
- **Honest framing:** semantic IDs are mainly known to help **new and long-tail items** (papers 1–2).
  In our cold-start set-up the *users* are new, the *films* are not, so a gain is not guaranteed. A
  negative result is still a result (doc 18 success criterion).
