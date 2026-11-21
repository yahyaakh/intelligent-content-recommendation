# Related Work: LLM Enrichment, Embeddings and Evaluation

**Status:** literature notes (compiled 2026-10-17)
**Related documents:** `01_llm_movie_enrichment.md` (enrichment run), `02_…` (texts, embeddings and evaluation, planned)

This document summarises published work on the questions this thesis studies, and compares it with the results obtained so far. The datasets in these papers are not limited to movies.

**Verification note:** for KAR and LLM-Rec, the numbers below were taken from the papers' result tables. For the other papers, only the abstract or a summary was read; their result tables should be checked before specific numbers are cited in the thesis.

---

## 1. Does LLM enrichment improve recommendations?

| Paper | What the LLM adds | Data | Reported result |
|---|---|---|---|
| **KAR** (Xi et al., RecSys 2024; Huawei) | Factual knowledge about items and reasoning about user preferences, encoded as vectors and fed to 9 CTR models | MovieLens-1M, Amazon-Books | AUC +1.22% to +1.49% (MovieLens-1M) and +0.94% to +1.38% (Amazon-Books), significant for every backbone. Online A/B tests: +7% recall (news), +1.7% song plays (music). Reasoning about user preferences helped more than factual item knowledge; both together helped most. |
| **LLM-Rec** (Lyu et al., NAACL Findings 2024) | LLM-augmented item descriptions (four prompting strategies), combined with the original description | MovieLens-1M, Recipe | NDCG@10 +8.54% (MovieLens-1M, GPT-3) and +21.72% (Recipe, GPT-3). Llama-2-7B gave smaller gains than GPT-3. KAR-style knowledge augmentation **hurt** on Recipe (NDCG@10 0.0525 vs 0.0580), and duplicating the original text hurt on MovieLens-1M (0.3567 vs 0.3640). |
| **ONCE / GENRE** (Liu et al., WSDM 2024) | GPT-3.5-generated content summaries and user profiles; LLaMA layers as a fine-tuned content encoder | MIND (news), Goodreads (books) | Up to +19.32% relative improvement over the previous best model |
| **LLMRec** (Wei et al., WSDM 2024) | LLM-generated item attributes, user profiles and extra interaction edges, with a denoising step | Netflix, MovieLens | Improvements reported; a reproducibility study of LLMRec appeared at RecSys 2025 |
| **LLMs as data augmenters for cold-start items** (Wang et al., Google DeepMind, 2024) | The LLM judges which of two new items a user would prefer; used as an auxiliary training signal | Amazon | Improvements for cold-start items across several recommendation models |

### Takeaways

1. **Gains are usually modest and context-dependent** (roughly 1–20%), and depend on the dataset, the LLM and how its output is used. LLM-Rec reports cases where augmentation **reduced** performance.
2. **Combining original and generated content** (LLM-Rec), or **learning how much weight the LLM output receives** (KAR), works better than simply adding generated text.
3. **The choice of LLM matters:** GPT-3 outperformed Llama-2 in LLM-Rec.
4. **Cold start** is where several papers make the strongest case for LLM enrichment.

---

## 2. Do larger or newer embedding models give better recommendations?

| Source | Setup | Finding |
|---|---|---|
| **Gorse text-embedding benchmark** (2026; practitioner benchmark, not peer reviewed) | GitHub repositories; rank 100 candidates by embedding distance to the user's previous item; NDCG@10; Qwen3-Embedding 0.6B/4B/8B and a commercial model | The 4B model outperformed the 8B model: "larger model size is not always better". The commercial model scored best in most settings. |
| **BLaIR** (Hou et al., 2024; ACL 2026) | Benchmark of LLMs as semantic encoders for recommendation and retrieval | General embedding benchmarks such as MTEB do not capture what recommendation tasks require |
| **MoRec** (Yuan et al., SIGIR 2023) | Item representations from text and images vs learned item-ID embeddings | Modality-based models can match ID-based models, including on popular items |
| **AlphaRec** (Sheng et al., 2024) | Item text embeddings from strong language models, linearly mapped, plus graph convolution and contrastive loss | Outperforms leading ID-based collaborative filtering models on several datasets |
| **Upper limits of text-based CF** (Li et al., 2023) | Item encoders scaled from 100M to 100B parameters | Reports "positive results" but also "unexpected negative outcomes" |

### Takeaways

1. Benchmark rankings of embedding models do not reliably predict recommendation quality.
2. Larger models are not consistently better; results differ between datasets (the Gorse benchmark found 4B best, this thesis found the opposite ranking within the Qwen3 family).
3. Text-based item representations are competitive with ID-based methods.

---

## 3. Popularity bias in MovieLens evaluation

| Paper | Finding |
|---|---|
| **Extending MovieLens-32M** (2025) | In standard offline evaluation, the popularity baseline ranks in the middle of 22 methods. When users judge recommendations by their interest in watching, popularity becomes one of the worst-performing methods. |
| **A Re-visit of the Popularity Baseline** (Ji et al., SIGIR 2020) | The popularity baseline is often defined inconsistently; on MovieLens its HR and NDCG can improve by 70% or more when popularity is computed relative to the time of each user's interactions. |

### Takeaway

A high offline score for popularity does not mean popularity is the best recommender. Offline MovieLens evaluation overrates it.

---

## 4. Reproducibility of LLM-based recommenders

| Paper | Finding |
|---|---|
| **RecRankerEval** (ECIR 2026) | Reproducing the LLM-based recommender RecRanker showed that its unusually high performance came from **data leakage**: ground-truth information was included in the prompts. |
| Reproducibility audits of LLM-based recommender research | LLMs add randomness and architectural complexity, which makes results harder to reproduce. |

---

## 5. Comparison with this thesis

### 5.1 Results so far

Warm-start evaluation on 10,490 MovieLens-25M users and 1,957 movies (original half-star ratings, 75/25 random split, seed 42, k = 10, mean scoring). Enrichment = LLM themes and mood from gemma4:e4b appended to the text before embedding. Numbers from the rerun in `05_experiment_scoring_comparison.md` (2026-10-20, corrected ratings).

| Embedding model | NDCG@10 base | NDCG@10 enriched | Change | p (Wilcoxon) |
|---|---|---|---|---|
| qwen3-embedding:0.6b | 0.198 | 0.228 | +15.1% | 10⁻²¹² |
| embeddinggemma | 0.273 | 0.289 | +5.8% | 10⁻¹³⁴ |
| qwen3-embedding:4b | 0.176 | 0.175 | −0.8% | 0.12 |
| bge-m3 | 0.160 | 0.154 | −4.0% | 10⁻³⁷ |
| qwen3-embedding:8b | 0.269 | 0.258 | −3.8% | 10⁻¹⁰ |
| embeddinggemma-2 | 0.259 | 0.244 | −6.0% | 10⁻⁵³ |

Reference: classic content-based (CountVectorizer + TF-IDF) 0.063 base / 0.062 enriched (no significant difference); popularity 0.421.

### 5.2 How the findings relate

| Thesis finding | Literature |
|---|---|
| The effect of enrichment ranges from +15.1% to −6.0% depending on the embedding model | Gains in the literature are typically 1–20% and context-dependent; LLM-Rec also reports cases where augmentation reduces performance |
| Enrichment had no effect on the classic content-based recommender | Simple addition of generated content is weaker than combining or weighting it (LLM-Rec, KAR) |
| Larger or newer embedding models did not perform better (embeddinggemma 300M tied with qwen3-embedding 8B; embeddinggemma-2 below embeddinggemma; qwen3 4B below 0.6B) | Gorse benchmark: 4B above 8B; BLaIR: general embedding benchmarks do not predict recommendation quality |
| Embedding-based recommendation clearly outperformed classic content-based under mean scoring (0.273 vs 0.063); under max scoring the gap is about 14% (`05_experiment_scoring_comparison.md`) | MoRec and AlphaRec: text-based representations are competitive even with ID-based methods |
| Popularity scored highest offline (0.421) | Extending MovieLens-32M: popularity is overrated by offline evaluation and among the worst when judged by users |

### 5.3 Differences in method

- Most papers **train** a model on the LLM output (CTR models, MLPs, graph models, fine-tuned encoders). This thesis uses LLM enrichment **without training**, through fixed similarity between items. This is simpler and cheaper, but a weaker way of using the labels, which may explain the smaller and less consistent gains.
- Several papers use large hosted LLMs (GPT-3, GPT-3.5); this thesis uses a local 4B model (gemma4:e4b) for enrichment.

### 5.4 Implications for the next experiments

1. **Weighted combination:** keep the LLM labels as a separate similarity and combine it with the base similarity using a weight tuned on a validation split, instead of appending them to the text (supported by LLM-Rec and KAR).
2. **Cold start:** evaluate users excluded from training, where several papers report the clearest benefit of enrichment.
3. **Enrichment model:** test whether labels from a larger LLM (e.g. gpt-oss-120b) change the recommendation gain, following the GPT-3 vs Llama-2 difference in LLM-Rec.
4. **Popularity:** keep reporting popularity of recommendations and results without the most popular items.
5. **Reproducibility:** keep the fixed seed, the split before any user features, and saved per-user results (avoids the leakage problems reported for LLM-based recommenders).

---

## 6. Sources

- Xi et al. — [Towards Open-World Recommendation with Knowledge Augmentation from LLMs (KAR), arXiv 2306.10933](https://arxiv.org/html/2306.10933v4)
- Lyu et al. — [LLM-Rec: Personalized Recommendation via Prompting LLMs, arXiv 2307.15780](https://arxiv.org/html/2307.15780v3)
- Liu et al. — [ONCE: Boosting Content-based Recommendation with Open- and Closed-source LLMs, arXiv 2305.06566](https://arxiv.org/pdf/2305.06566)
- Wei et al. — [LLMRec: LLMs with Graph Augmentation for Recommendation, arXiv 2311.00423](https://arxiv.org/pdf/2311.00423) · [code](https://github.com/HKUDS/LLMRec)
- Wang et al. — [Large Language Models as Data Augmenters for Cold-Start Item Recommendation, arXiv 2402.11724](https://arxiv.org/pdf/2402.11724)
- Gorse — [Benchmark Text Embedding Models for RecSys in 2026](https://gorse.io/posts/embedding-benchmark-2026)
- Hou et al. — [BLaIR: Bridging Language and Items for Retrieval and Recommendation, arXiv 2403.03952](https://arxiv.org/pdf/2403.03952)
- [RecBench: Can LLMs Outshine Conventional Recommenders?, arXiv 2503.05493](https://arxiv.org/html/2503.05493v2)
- Yuan et al. — [Where to Go Next for Recommender Systems? ID- vs. Modality-based Recommender Models Revisited, arXiv 2303.13835](https://arxiv.org/pdf/2303.13835)
- Sheng et al. — [Language Models Encode Collaborative Signals in Recommendation (AlphaRec), arXiv 2407.05441](https://arxiv.org/abs/2407.05441v1)
- Li et al. — [Exploring the Upper Limits of Text-Based Collaborative Filtering Using LLMs, arXiv 2305.11700](https://arxiv.org/abs/2305.11700)
- [Extending MovieLens-32M to Provide New Evaluation Objectives, arXiv 2504.01863](https://arxiv.org/abs/2504.01863)
- Ji et al. — [A Re-visit of the Popularity Baseline in Recommender Systems, arXiv 2005.13829](https://arxiv.org/abs/2005.13829)
- [RecRankerEval: Reproducible Framework for LLM-based Top-k Recommenders, arXiv 2507.05880](https://arxiv.org/pdf/2507.05880)
