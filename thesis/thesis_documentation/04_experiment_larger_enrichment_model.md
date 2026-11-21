# Experiment: Enrichment Model Comparison (gemma4:e4b vs gemma4:31b vs gpt-oss-120b)

**Status:** finished (run dates: 2026-10-18; recommendation results rerun 2026-10-20 on corrected ratings, see section 2)
**Script:** `build_movie_texts.py` → `run_enrichment_experiment(enrich_model, provider, reasoning_effort)`
**Results folders:** `output/experiments/gemma4_31b-cloud/`, `output/experiments/gpt-oss_120b-cloud/`
**Related documents:** `01_llm_movie_enrichment.md` (original enrichment with gemma4:e4b), `03_related_work.md`

---

## 1. Question

Does a larger LLM produce better themes and mood labels, and do better labels lead to better recommendations?

Three enrichment models were compared, with everything else identical:

| Model | Parameters | Family | Where it runs | Reasoning |
|---|---|---|---|---|
| `gemma4:e4b` | about 4B effective | Gemma 4 | local (RTX 4050 Laptop, 6 GB) | off |
| `gemma4:31b-cloud` | 31B | Gemma 4 | hosted (Ollama cloud) | off |
| `gpt-oss:120b-cloud` | 120B (mixture of experts) | gpt-oss (OpenAI) | hosted (Ollama cloud) | low |

- **gemma4:e4b vs gemma4:31b** isolates **model size** within one family.
- **gpt-oss-120b** adds a much larger model from a **different family**.

This follows the observation in LLM-Rec (Lyu et al., 2024) that GPT-3 augmentation outperformed Llama-2 (see `03_related_work.md`).

The two hosted models do not fit on the local GPU (gemma4:31b alone needs about 19–20 GB). Requests go through the local Ollama installation to Ollama's servers; only public movie metadata is sent. DeepSeek-V4 and Kimi-K3 were also considered but are not available on the free Ollama plan.

---

## 2. Method

Everything except the enrichment model is identical across the three runs.

### 2.1 Enrichment

- Same prompt v2, label vocabularies (44 themes, 18 moods), audience labels and intensity scale as in `01_llm_movie_enrichment.md`.
- Temperature 0, seed 42. Thinking off for the Gemma models; reasoning effort "low" for gpt-oss (which does not support switching reasoning off).
- One resumable cache per model: `output/enrichment/<model>_v2.jsonl`.
- **Hosted models ignore the JSON schema** (`format`) and wrap their answer in a Markdown code block. A subclass of the enricher (`FenceStrippingEnricher`) removes the code block and all raw line breaks before parsing. Raw line breaks are only allowed between JSON elements, never inside text values, so removing them cannot change a valid answer; it repaired two gpt-oss answers that contained a line break inside a word (`"isolatio⏎⏎n and loneliness"`).
- Because the schema is not enforced for hosted models, all answers go through the same validation (`validate_output`), which removes invalid or duplicate labels and records each repair.

### 2.2 Texts and embeddings

- Base texts: unchanged (they do not depend on the enrichment model).
- Enriched texts: base text + "Themes: … Mood: …" from each model's labels.
- Embedding model and prefix are read from the saved base index (`embeddinggemma`, `title: none | text: `), so all vectors are made the same way.

### 2.3 Evaluation

Same users, split and metrics as all other results:

| Setting | Value |
|---|---|
| Ratings | MovieLens 25M (`processed_30k`), restricted to the 1,957 movies: 2,547,690 ratings |
| Split | random 75/25, seed 42 |
| Users evaluated | 10,490 (about 99 liked and 33 relevant movies each) |
| Candidate pool | 1,547 movies seen in training |
| Relevance | test rating ≥ the user's training mean |
| Scoring | mean similarity to the user's liked movies |
| Metrics | NDCG@10, Precision@10, Recall@10; Wilcoxon signed-rank test per user |

Each run evaluated six cases: content-based and embedding-based, each without labels, with gemma4:e4b labels, and with the new model's labels.

**Data correction (2026-10-20).** The first runs used ratings in which ML-25M's half stars had been rounded with round-half-to-even (3.5 → 4, 4.5 → 4, 2.5 → 2), which changed 35% of the ratings. The ratings now keep the original half stars. The recommendation results in section 5 come from the rerun in `05_experiment_scoring_comparison.md` (mean scoring), which evaluates all label sets on the corrected ratings with identical labels, embeddings and split; the label analysis in sections 3–4 does not depend on the ratings and is unchanged. The conclusions are the same as in the first runs.

---

## 3. Enrichment runs

| Measure | gemma4:e4b | gemma4:31b | gpt-oss-120b |
|---|---|---|---|
| Movies enriched | 1,957 | 1,957 | 1,957 |
| Answers repaired by validation | 0 | 8 | 139 |
| Answers with broken JSON | 0 | 0 | 2 (fixed: one by retry, one by line-break removal) |
| Run time | about 1 hour (local) | about 30 minutes (hosted) | about 38 minutes (hosted) |

Breakdown of the 139 gpt-oss repairs: 129 audience values "adult" instead of "adults" (set to unknown), 8 invented theme labels (removed), 1 invented mood label (removed), 1 missing intensity score.

**Observation:** schema enforcement mattered more for reliability than model size. The local 4B model with an enforced schema needed no repairs; the 120B hosted model without schema enforcement needed 139.

Hosted run times depend on the provider's servers and are not comparable with local run times.

---

## 4. Label comparison

### 4.1 Distribution

| Measure | gemma4:e4b | gemma4:31b | gpt-oss-120b |
|---|---|---|---|
| Themes per movie | 4.36 | 4.17 | 4.25 |
| Themes used (of 44) | 44 | 44 | 44 |
| Most common theme | identity and self-discovery (46%) | family relationships (25%) | identity and self-discovery (36%) |
| "good versus evil" | 26% | 6% | 16% |
| Moods per movie | 2.99 | 2.97 | 2.99 |

Ten most common themes:

| gemma4:31b | gpt-oss-120b |
|---|---|
| family relationships 25% | identity and self-discovery 36% |
| friendship 24% | friendship 26% |
| love and romance 23% | family relationships 25% |
| identity and self-discovery 21% | love and romance 23% |
| survival 17% | loss and grief 19% |
| crime and criminals 17% | redemption 18% |
| social class and inequality 14% | survival 18% |
| deception and conspiracy 14% | coming of age 17% |
| historical events 13% | power and corruption 16% |
| justice and the law 13% | good versus evil 16% |

Largest changes from gemma4:e4b to gemma4:31b (number of movies): identity and self-discovery −502, good versus evil −392, loss and grief −159; friendship +168, rebellion against authority +116, family relationships +110.

### 4.2 Agreement between models

| Pair | Theme overlap (Jaccard, mean per movie) | Audience agreement |
|---|---|---|
| e4b – 31b | 0.57 | 75% |
| e4b – gpt-oss | 0.59 | 74% |
| 31b – gpt-oss | 0.59 | 75% |

Only 9% of movies received identical theme sets from e4b and 31b.

### 4.3 Audience and content intensity

| Measure | gemma4:e4b | gemma4:31b | gpt-oss-120b |
|---|---|---|---|
| Audience adults / teens / family | 1,258 / 389 / 299 | 1,033 / 338 / 579 | 1,015 / 337 / 473 (129 unknown) |
| Violence (mean, 0–3) | 1.06 | 1.42 | 1.44 |
| Language | 0.60 | 1.24 | 1.35 |
| Sexual content | 0.18 | 0.75 | 0.34 |

### 4.4 Interpretation of the labels

- **Theme assignment is subjective.** Any two models share only about 58% of their themes per movie; even large models do not converge on the same labels.
- **gemma4:31b reduced the overuse of broad themes the most** (identity and self-discovery 46% → 21%, good versus evil 26% → 6%); gpt-oss reduced it partly (36% and 16%).
- **Both large models assign much higher intensity scores,** about double for language. Small models underestimating language was the main intensity problem in the model comparison (`01_llm_movie_enrichment.md`, section 3).
- Both large models label fewer films "adults" and more "family", which likely reduces e4b's tendency to label any serious film "adults".
- These observations are based on label distributions, not on ground truth (see section 7).

---

## 5. Recommendation results

### 5.1 Embedding-based (embeddinggemma)

| Labels from | NDCG@10 | Precision@10 | Recall@10 | Gain over no labels |
|---|---|---|---|---|
| none | 0.2732 | 0.2623 | 0.0912 | — |
| gemma4:e4b | 0.2892 | 0.2815 | 0.0964 | +5.8% |
| gemma4:31b | **0.2901** | 0.2794 | 0.0959 | **+6.2%** |
| gpt-oss-120b | 0.2870 | 0.2742 | 0.0942 | +5.0% |

### 5.2 Content-based (CountVectorizer 75% + TF-IDF 25%)

| Labels from | NDCG@10 | Precision@10 | Recall@10 |
|---|---|---|---|
| none | 0.0625 | 0.0635 | 0.0232 |
| gemma4:e4b | 0.0622 | 0.0629 | 0.0221 |
| gemma4:31b | 0.0603 | 0.0644 | 0.0235 |
| gpt-oss-120b | 0.0629 | 0.0638 | 0.0227 |

### 5.3 Significance tests (Wilcoxon signed-rank, NDCG@10 per user)

| Comparison | Mean difference | p-value | Users better / worse |
|---|---|---|---|
| Embedding: 31b vs e4b labels | +0.0010 (+0.3%) | 0.0085 | 4,196 / 3,960 |
| Embedding: gpt-oss vs e4b labels | −0.0022 (−0.8%) | 2 × 10⁻⁵ | 3,878 / 4,218 |
| Embedding: 31b labels vs none | +0.0169 | 3 × 10⁻¹⁵¹ | 5,402 / 3,378 |
| Embedding: gpt-oss labels vs none | +0.0138 | 8 × 10⁻¹⁰⁹ | 5,216 / 3,426 |
| Content: 31b vs e4b labels | −0.0019 | 0.056 (n.s.) | 2,870 / 2,977 |
| Content: 31b labels vs none | −0.0022 | 0.0038 | 2,804 / 2,997 |
| Content: gpt-oss vs e4b labels | +0.0007 | 0.30 (n.s.) | 2,977 / 2,898 |
| Content: gpt-oss labels vs none | +0.0004 | 0.84 (n.s.) | 3,056 / 3,031 |

---

## 6. Findings

1. **Model size does not predict the recommendation gain.** The ranking is 31B > 4B > 120B, and all three lie within ±0.002 NDCG@10 of each other. The largest model was slightly but significantly worse than the small local one.
2. **All three label sets improve embedding-based recommendation by a similar amount** (+5.0% to +6.2% NDCG@10 over no labels), although the models agree on only about 58% of themes. The exact labels matter much less than the fact that category labels are added at all.
3. **Fewer generic themes may help slightly.** The model that avoided broad themes most (gemma4:31b) scored best, and the one that overused them most among the large models (gpt-oss) scored lowest. The effect is small.
4. **Content-based recommendation does not benefit** from any label set (differences of −0.002 to +0.0004 NDCG@10, mostly not significant).
5. **The enrichment model is the least influential factor tested so far:**

   | Factor | Effect on NDCG@10 |
   |---|---|
   | Scoring method (mean vs closest liked movie) | +0.042 |
   | Embedding model (six models) | enrichment effect from −0.016 to +0.030 |
   | Enrichment LLM (three models) | 0.2870 – 0.2901 (±0.002) |

6. **Practical conclusion:** a local 4B model performs within 0.001 NDCG@10 of the best hosted model, at no cost, with full reproducibility and with no repairs needed. A larger or hosted model is not justified for this purpose.
7. **Where the larger models may still matter:** the constraint layer, which uses audience and intensity directly (e.g. child safety). The large models' intensity and audience labels differ substantially from e4b's (section 4.3).

These results are consistent with the interpretation that label quality is not the bottleneck in this setup. The limiting factor appears to be how the labels are used: appended as one sentence to a text of about 100 words and compared with a fixed similarity, without training. In the literature, gains from larger LLMs were reported mainly when a model was trained on the generated content (`03_related_work.md`).

---

## 7. Limitations and next steps

- **Label accuracy not measured against ground truth.** An audit on a random sample (IMDb Parents Guide, AllMovie, MovieLens Tag Genome) is needed to confirm which labels are more accurate.
- **One embedding model.** Appending e4b labels reduced quality for qwen3-embedding:8b, bge-m3 and embeddinggemma-2; the larger models' labels should be evaluated with these models as well.
- **One scoring method.** With "closest liked movie" (max) scoring, the e4b enrichment gain fell from +5.8% to about +1%.
- **One split** (seed 42).
- **gpt-oss ran at "low" reasoning;** default reasoning might label differently.
- **129 gpt-oss audience values are unknown** ("adult" instead of "adults"); this does not affect the recommendation results (only themes and mood are used) but matters for the constraint layer.
- **Hosted models** did not enforce the schema, their run times are not comparable with local models, and they may change over time.
- **Possible follow-ups:** a non-LLM baseline (BART zero-shot classification with the same label lists) to test whether an LLM is needed at all; a revised theme vocabulary (prompt v3) filling documented gaps such as "superheroes"; evaluation of the large models' audience and intensity labels in the constraint layer.

---

## 8. Files

| File | Content |
|---|---|
| `output/enrichment/gemma4_31b-cloud_v2.jsonl` | raw gemma4:31b enrichment records |
| `output/enrichment/gpt-oss_120b-cloud_v2.jsonl` | raw gpt-oss-120b enrichment records |
| `output/experiments/<model>/movies_enriched.csv` | flattened labels, including the `repaired` flag |
| `output/experiments/<model>/movie_texts.csv` | base and enriched texts with that model's labels |
| `output/experiments/<model>/movies_enriched_index.*` | embeddinggemma index of the enriched texts |
| `output/experiments/<model>/per_user_results.csv` | NDCG, precision and recall per user and case |
| `output/experiments/<model>/summary.csv` | averages per case |
| `output/experiments/<model>/significance_tests.csv` | Wilcoxon tests |
| `output/experiments/<model>/settings.json` | models, prompt version, split, scoring, repairs, mean shift |

`<model>` is `gemma4_31b-cloud` or `gpt-oss_120b-cloud`. The original e4b files (`output/enrichment/movies_enriched_v2.csv`, `movie_texts_v2.csv`, `output/embeddings/…`, `output/evaluation/…`) were read but not modified.

---

## 9. Reproducing the runs

In `build_movie_texts.py`:

```python
if __name__ == "__main__":
    RUN_EMBEDDINGS = False
    RUN_EVALUATION = False
    RUN_EXPERIMENT = True
    ...
    if RUN_EXPERIMENT:
        run_enrichment_experiment("gemma4:31b-cloud")
        # run_enrichment_experiment("gpt-oss:120b-cloud", reasoning_effort="low")
```

Requirements: `ollama signin` (for cloud models), the existing embeddinggemma base and enriched indexes in `output/embeddings/`, and `output/enrichment/movies_enriched_v2.csv`. Enrichment is resumable; rerunning continues from the cache, and the embedding step reuses the saved index if it exists.
