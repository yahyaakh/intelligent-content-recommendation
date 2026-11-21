# Experiment: User-Profile Scoring (Mean vs Max Similarity)

**Status:** finished (first run 2026-10-18; rerun 2026-10-20 on corrected ratings, see section 2)
**Script:** `build_movie_texts.py` → `run_scoring_comparison()`
**Results folder:** `output/experiments/scoring_comparison/`
**Related documents:** `01_llm_movie_enrichment.md`, `03_related_work.md`, `04_experiment_larger_enrichment_model.md`

---

## 1. Question

All recommenders in this thesis score a candidate movie by its similarity to the movies the user liked. A user has about 100 liked movies, so these similarities must be combined into one score. Two ways were compared:

| Scoring | Score of a candidate | Interpretation |
|---|---|---|
| **Mean** | average similarity to all liked movies | similar to the user's taste overall |
| **Max** | similarity to the single most similar liked movie | very similar to at least one movie the user liked |

**Example.** A user likes *The Matrix*, *Blade Runner*, *Notting Hill* and *When Harry Met Sally*. The candidate *Ex Machina* has similarities 0.80, 0.75, 0.10 and 0.08 to these. Mean scoring gives 0.43, because the romantic comedies pull the score down; max scoring gives 0.80.

**Motivation.** The early prototype in `LLMS_test.py` used max-like scoring (closest liked movie among each liked movie's 30 nearest neighbours) and reached NDCG@10 0.3208, higher than the main pipeline (mean scoring, 0.2977), both measured on the earlier, rounded ratings. A controlled check showed that the difference came from the scoring, not the text representation. This experiment re-evaluates every existing case under both scorings.

**Note on method choice.** Max scoring was first observed to perform better on the test split. To avoid selecting a method after seeing test results, both scorings are reported for every case.

---

## 2. Method

Everything except the scoring is identical to all previous evaluations:

| Setting | Value |
|---|---|
| Ratings | MovieLens 25M (`processed_30k`), restricted to the 1,957 movies: 2,547,690 ratings, original half stars (0.5–5) |
| Split | random 75/25, seed 42 |
| Users evaluated | 10,490 (about 99 liked and 33 relevant movies each) |
| Candidate pool | 1,547 movies seen in training (minus each user's seen movies) |
| Relevance | test rating ≥ the user's training mean |
| Metrics | NDCG@10, Precision@10, Recall@10; Wilcoxon signed-rank test per user |

### 2.1 Cases (19 similarity tables, each evaluated with both scorings)

| Group | Cases |
|---|---|
| Content-based (CountVectorizer 75% + TF-IDF 25%) | no labels (A); with gemma4:e4b, gemma4:31b and gpt-oss-120b labels (B) |
| Embedding-based, six models | without labels (C) and with gemma4:e4b labels (D): embeddinggemma, embeddinggemma-2, bge-m3, qwen3-embedding 0.6b / 4b / 8b |
| Embedding-based, other label sets | embeddinggemma with gemma4:31b labels and with gpt-oss-120b labels |
| Old text representation | embeddinggemma vectors of the early `textual_representation` from `LLMS_test.py` |

No embeddings or labels were recomputed; all saved indexes and label files were reused.

### 2.2 Implementation

- `recommend` and `evaluate_method` received an optional `scoring` parameter (`"mean"` or `"max"`, default `"mean"`, so all earlier runs are unchanged).
- `embedding_similarity` received an optional `folder` parameter (to load experiment indexes and the old index) and normalises vectors before computing cosine similarity.
- All tables are built once; 38 evaluations ran in about 5 minutes.

**Data correction (2026-10-20).** The first run used ratings in which ML-25M's half stars had been rounded with round-half-to-even (3.5 → 4, 4.5 → 4, 2.5 → 2, 0.5 → 1), which changed 35% of the ratings. The data preparation (`prepare_movielens25m.py`) now keeps the original half stars, and all numbers in this document come from the rerun on the corrected ratings. The embeddings, labels, split procedure and code were unchanged; the conclusions are the same as in the first run, with values about 0.01–0.02 lower.

**Validity checks:** the embeddinggemma results (max scoring, with and without e4b labels) were reproduced exactly by an independent re-implementation of the evaluation, and five stored vectors per index were re-embedded with cosine similarity 1.0000 to the saved ones.

---

## 3. Results

### 3.1 NDCG@10 for every case

Sorted by NDCG@10 under max scoring.

| Case | Mean | Max | Max − mean |
|---|---|---|---|
| embeddinggemma + e4b labels | 0.2892 | 0.3186 | +0.0294 |
| embeddinggemma + 31b labels | 0.2901 | 0.3182 | +0.0280 |
| embeddinggemma + gpt-oss labels | 0.2870 | 0.3171 | +0.0301 |
| embeddinggemma | 0.2732 | 0.3152 | +0.0420 |
| old text (embeddinggemma) | 0.2859 | 0.3073 | +0.0214 |
| qwen3-embedding:8b + e4b labels | 0.2584 | 0.3049 | +0.0465 |
| bge-m3 + e4b labels | 0.1537 | 0.3030 | +0.1493 |
| qwen3-embedding:4b | 0.1760 | 0.3017 | +0.1257 |
| bge-m3 | 0.1601 | 0.3015 | +0.1414 |
| qwen3-embedding:8b | 0.2686 | 0.2964 | +0.0277 |
| embeddinggemma-2 + e4b labels | 0.2436 | 0.2955 | +0.0519 |
| embeddinggemma-2 | 0.2592 | 0.2923 | +0.0331 |
| qwen3-embedding:4b + e4b labels | 0.1746 | 0.2892 | +0.1146 |
| content + 31b | 0.0603 | 0.2872 | +0.2269 |
| content + e4b | 0.0622 | 0.2824 | +0.2201 |
| qwen3-embedding:0.6b | 0.1979 | 0.2809 | +0.0830 |
| content, no labels | 0.0625 | 0.2789 | +0.2163 |
| qwen3-embedding:0.6b + e4b labels | 0.2279 | 0.2772 | +0.0494 |
| content + gpt-oss | 0.0629 | 0.2771 | +0.2142 |

Reference: popularity baseline 0.4214 (Precision@10 0.3906, Recall@10 0.1250), independent of the scoring. Popularity = number of positive training ratings per movie (rating ≥ that user's mean); the same definition is used in every experiment.

### 3.2 Precision@10 and Recall@10

| Case | Precision mean | Precision max | Recall mean | Recall max |
|---|---|---|---|---|
| content, no labels | 0.0635 | 0.2314 | 0.0232 | 0.0766 |
| content + e4b | 0.0629 | 0.2368 | 0.0221 | 0.0778 |
| content + 31b | 0.0644 | 0.2415 | 0.0235 | 0.0798 |
| content + gpt-oss | 0.0638 | 0.2284 | 0.0227 | 0.0754 |
| embeddinggemma | 0.2623 | 0.2712 | 0.0912 | 0.0912 |
| embeddinggemma + e4b | 0.2815 | 0.2727 | 0.0964 | 0.0915 |
| embeddinggemma + 31b | 0.2794 | 0.2719 | 0.0959 | 0.0914 |
| embeddinggemma + gpt-oss | 0.2742 | 0.2730 | 0.0942 | 0.0920 |
| embeddinggemma-2 | 0.2384 | 0.2715 | 0.0812 | 0.0912 |
| embeddinggemma-2 + e4b | 0.2271 | 0.2696 | 0.0769 | 0.0905 |
| bge-m3 | 0.1552 | 0.2679 | 0.0515 | 0.0888 |
| bge-m3 + e4b | 0.1477 | 0.2695 | 0.0496 | 0.0893 |
| qwen3-embedding:0.6b | 0.1875 | 0.2457 | 0.0610 | 0.0805 |
| qwen3-embedding:0.6b + e4b | 0.1996 | 0.2450 | 0.0649 | 0.0803 |
| qwen3-embedding:4b | 0.1712 | 0.2625 | 0.0571 | 0.0868 |
| qwen3-embedding:4b + e4b | 0.1674 | 0.2533 | 0.0545 | 0.0828 |
| qwen3-embedding:8b | 0.2349 | 0.2661 | 0.0787 | 0.0890 |
| qwen3-embedding:8b + e4b | 0.2312 | 0.2665 | 0.0790 | 0.0885 |
| old text (embeddinggemma) | 0.2610 | 0.2639 | 0.0904 | 0.0888 |

### 3.3 Max vs mean (Wilcoxon, NDCG@10 per user)

| Case | Max − mean | p-value | Users better / worse |
|---|---|---|---|
| content, no labels | +0.2163 | < 10⁻³⁰⁰ | 8,174 / 905 |
| content + e4b | +0.2201 | < 10⁻³⁰⁰ | 8,310 / 777 |
| content + 31b | +0.2269 | < 10⁻³⁰⁰ | 8,419 / 725 |
| content + gpt-oss | +0.2142 | < 10⁻³⁰⁰ | 8,190 / 828 |
| embeddinggemma | +0.0420 | 1 × 10⁻⁷⁰ | 5,546 / 4,300 |
| embeddinggemma + e4b | +0.0294 | 5 × 10⁻³⁶ | 5,375 / 4,466 |
| embeddinggemma + 31b | +0.0280 | 1 × 10⁻³³ | 5,332 / 4,490 |
| embeddinggemma + gpt-oss | +0.0301 | 5 × 10⁻³⁸ | 5,386 / 4,448 |
| embeddinggemma-2 | +0.0331 | 6 × 10⁻⁴⁵ | 5,494 / 4,396 |
| embeddinggemma-2 + e4b | +0.0519 | 2 × 10⁻¹⁰⁹ | 5,815 / 4,045 |
| bge-m3 | +0.1414 | < 10⁻³⁰⁰ | 7,269 / 2,305 |
| bge-m3 + e4b | +0.1493 | < 10⁻³⁰⁰ | 7,397 / 2,181 |
| qwen3-embedding:0.6b | +0.0830 | 9 × 10⁻²⁹¹ | 6,315 / 3,317 |
| qwen3-embedding:0.6b + e4b | +0.0494 | 1 × 10⁻¹¹⁴ | 5,786 / 3,853 |
| qwen3-embedding:4b | +0.1257 | < 10⁻³⁰⁰ | 7,008 / 2,747 |
| qwen3-embedding:4b + e4b | +0.1146 | < 10⁻³⁰⁰ | 6,906 / 2,723 |
| qwen3-embedding:8b | +0.0277 | 5 × 10⁻³¹ | 5,346 / 4,500 |
| qwen3-embedding:8b + e4b | +0.0465 | 2 × 10⁻⁸⁸ | 5,737 / 4,108 |
| old text (embeddinggemma) | +0.0214 | 3 × 10⁻¹⁹ | 5,269 / 4,574 |

### 3.4 Effect of LLM enrichment under each scoring (NDCG@10)

| Comparison | Mean: difference | p | Max: difference | p | Max: users better / worse |
|---|---|---|---|---|---|
| content + e4b vs content | −0.0003 | 0.57 | **+0.0035** | 4 × 10⁻¹¹ | 4,097 / 3,272 |
| content + 31b vs content | −0.0022 | 0.0038 | **+0.0083** | 1 × 10⁻⁴⁵ | 4,387 / 3,255 |
| content + gpt-oss vs content | +0.0004 | 0.84 | **−0.0018** | 0.0031 | 3,523 / 3,874 |
| embeddinggemma + e4b | +0.0160 | 1 × 10⁻¹³⁴ | +0.0034 | 7 × 10⁻⁷⁵ | 2,932 / 1,508 |
| embeddinggemma + 31b | +0.0169 | 3 × 10⁻¹⁵¹ | +0.0030 | 2 × 10⁻⁵⁶ | 3,114 / 1,790 |
| embeddinggemma + gpt-oss | +0.0138 | 8 × 10⁻¹⁰⁹ | +0.0020 | 2 × 10⁻¹⁹ | 2,514 / 1,826 |
| embeddinggemma-2 + e4b | −0.0155 | 2 × 10⁻⁵³ | **+0.0032** | 5 × 10⁻²⁵ | 3,933 / 2,995 |
| bge-m3 + e4b | −0.0064 | 1 × 10⁻³⁷ | **+0.0015** | 7 × 10⁻⁹ | 1,853 / 1,593 |
| qwen3-embedding:0.6b + e4b | +0.0299 | 1 × 10⁻²¹² | **−0.0037** | 7 × 10⁻¹⁸ | 3,140 / 3,860 |
| qwen3-embedding:4b + e4b | −0.0015 | 0.12 | −0.0125 | 3 × 10⁻⁵⁰ | 3,818 / 4,949 |
| qwen3-embedding:8b + e4b | −0.0103 | 3 × 10⁻¹⁰ | **+0.0085** | 4 × 10⁻²³ | 4,743 / 3,977 |

Bold marks a change of sign compared with mean scoring.

---

## 4. Findings

1. **Max scoring improves NDCG@10 for all 19 cases,** by +0.021 to +0.227, all highly significant. The weaker a method was under mean scoring, the more it gains (content-based ×4.5; bge-m3 ×1.9 and qwen3-embedding:4b ×1.7; embeddinggemma +15%). Interpretation: with about 100 liked movies spanning several tastes, averaging blurs those tastes together; max scoring lets each taste find its own matches.
2. **The large gap between content-based and embedding-based recommendation was mostly caused by mean scoring.** Under mean scoring the best embedding case was 4.6× better than content-based (0.290 vs 0.063); under max scoring the gap is about 14% (0.319 vs 0.279), and content-based performs about as well as several embedding models.
3. **For the strong embedding models, max scoring mainly improves the ordering at the top of the list.** For embeddinggemma, NDCG@10 rises clearly, but Precision@10 and Recall@10 stay about the same (with e4b labels they are even slightly lower: precision 0.282 → 0.273, recall 0.096 → 0.092). For content-based and the weaker embedding models, all three metrics rise strongly.
4. **The effect of LLM enrichment is small, and its direction depends on the scoring.** Under max scoring, enrichment improves 8 of 11 comparisons (5 of 11 under mean), but the largest gain is +0.0085 NDCG@10. Seven comparisons change sign between the two scorings. For embeddinggemma, the gain falls from +5.8% to about +1%. Under max scoring, content-based recommendation benefits slightly from gemma4:e4b and gemma4:31b labels (+0.004, +0.008), which it did not under mean scoring.
5. **The enrichment LLM matters even less under max scoring.** The three label sets with embeddinggemma lie within 0.0015 NDCG@10 (e4b 0.3186, 31b 0.3182, gpt-oss 0.3171).
6. **The best configuration overall** is embeddinggemma with gemma4:e4b labels and max scoring (NDCG@10 0.3186), using only local models.
7. **Ranking of design choices by effect size (NDCG@10):**

   | Factor | Effect |
   |---|---|
   | Scoring (mean → max) | +0.021 to +0.227 |
   | Representation (content vs best embedding, max scoring) | about +0.03 to +0.04 |
   | LLM enrichment (max scoring) | −0.013 to +0.009 |
   | Enrichment LLM (three models, max scoring) | ±0.001 |

8. **Popularity remains higher offline** (0.4214, against 0.3186 for the best embedding case; Wilcoxon p < 10⁻³⁰⁰). This is consistent with the known overrating of popularity in offline MovieLens evaluation (`03_related_work.md`). The value of content- and embedding-based methods is expected mainly in cold-start and natural-language settings, where popularity cannot personalise.

---

## 5. Consequences for the thesis

- **Max scoring is used as the main scoring method** for the remaining experiments (cold start, reranking). Results for existing cases are reported under both scorings.
- **Revised conclusions** compared with earlier mean-scoring results:
  - embedding-based recommendation outperforms classic content-based recommendation modestly (about 14%), not by a factor of four;
  - LLM enrichment gives small gains (about +1% for the best embedding model) whose direction depends on the embedding model and scoring;
  - the choice of enrichment LLM has a negligible effect.
- The user-profile aggregation has a larger effect than any representation or LLM choice tested so far, an aspect often given little attention in comparisons of recommendation methods.

---

## 6. Limitations

- **One split** (seed 42); robustness across seeds is not yet tested.
- **Two aggregation methods only.** Intermediate options (e.g. mean of the top-k most similar liked movies, or clustering a user's liked movies into several taste profiles) were not tested.
- **Warm-start users only** (about 100 liked movies each). With few liked movies (cold start), the two scorings are expected to differ less; this will be measured in the cold-start experiment.
- **Max scoring was first observed on the test split;** both scorings are therefore reported for all cases.

---

## 7. Files

| File | Content |
|---|---|
| `output/experiments/scoring_comparison/per_user_results.csv` | NDCG, precision and recall per user, case and scoring |
| `output/experiments/scoring_comparison/summary.csv` | averages per case and scoring |
| `output/experiments/scoring_comparison/significance_tests.csv` | max vs mean per case; enrichment effects under both scorings; best cases vs popularity |
| `output/experiments/scoring_comparison/popularity_per_user.csv` | popularity baseline per user |
| `output/experiments/scoring_comparison/settings.json` | split, models, label sets, cases |

---

## 8. Reproducing the run

In `build_movie_texts.py`:

```python
if __name__ == "__main__":
    RUN_EMBEDDINGS = False
    RUN_EVALUATION = False
    RUN_EXPERIMENT = False
    RUN_SCORING = True
    ...
    if RUN_SCORING:
        run_scoring_comparison()
```

Requirements: the base and enriched indexes of the six embedding models in `output/embeddings/`, the experiment indexes and label files in `output/experiments/gemma4_31b-cloud/` and `output/experiments/gpt-oss_120b-cloud/`, and the old index `movie_embeddinggemma.faiss` with its ID file in the project root.
