# Experiment: Larger Label Vocabulary from the MovieLens Tag Genome

**Status:** finished (run date: 2026-10-21, on the corrected half-star ratings)
**Scripts:** `build_vocabulary_v4.py` (vocabulary), prompt v4 in `llm_enricher.py`, `run_enrichment_experiment("gemma4:e4b", prompt_version="v4")` in `build_movie_texts.py` (labels and embeddings), `vocabulary_experiment.py` (evaluation)
**Results folders:** `vocabularies/v4_genome.json`, `output/experiments/gemma4_e4b_v4/`, `output/experiments/vocabulary/`
**Related documents:** `01_llm_movie_enrichment.md`, `05_experiment_scoring_comparison.md`, `06_experiment_cold_start.md`, `07_experiment_outside_knowledge.md`

---

## 1. Question

The v2 labels use 44 themes and 18 moods. Several are broad and very frequent: "identity and self-discovery" was assigned to 46% of all movies, and the mood "tense" to 44%. Such labels apply to many films and separate them poorly. Experiment 07 also showed that a fixed coarse vocabulary limits how much of the model's knowledge can be expressed.

**Question:** does a larger, more specific and externally validated vocabulary produce better labels, and do better labels improve recommendation?

**Expectations, written down before the full evaluation:**

- v4 labels are more specific (fewer movies per label) and mostly confirmed by the genome;
- warm start, max scoring: embedding-based recommendation with v4 labels within about ±0.005 NDCG@10 of v2; content-based recommendation may change most, because it matches label words exactly;
- cold-start hybrids: no significant difference.

---

## 2. Vocabulary

### 2.1 Source

The **MovieLens Tag Genome** (MovieLens 25M, `genome-tags.csv`; Vig et al., 2012): 1,128 tags curated by MovieLens, with a relevance score from 0 to 1 for each tag and about 13,800 movies (`genome-scores.csv`). The tags are words real users use to describe films ("atmospheric", "slow paced", "twist ending", "dystopia", "moral ambiguity"). Because every v4 label is a genome tag, the labels can be checked against the genome scores.

The files were extracted from the official archive (`files.grouplens.org/datasets/movielens/ml-25m.zip`, MD5 `6b51fb2759a8657d3bfcbfc42b592ada`, matching the published checksum) into `data/ml-25m/genome/`.

Two alternatives were considered and rejected: clustering the TMDB keywords of the catalog (the keywords are already part of the LLM input, so labels would mostly repeat them), and an LLM-designed list (least principled).

### 2.2 Filtering

Tags that are not themes or moods were removed by hand:

| Removed | Examples | Reason |
|---|---|---|
| People, franchises, studios | "tarantino", "007", "pixar" | not themes |
| Awards and lists | "oscar (best picture)", "imdb top 250" | quality signals, not content |
| Quality judgements | "great acting", "boring", "overrated" | opinions |
| Production and source | "based on a book", "remake", "cgi", "3d" | not themes |
| Places, years and eras | "new york city", "1980s" | setting, a different kind of attribute |
| Genres | "comedy", "horror", "sci-fi" | already in the LLM input |
| Nudity and age ratings | "nudity (topless)", "pg-13" | covered by the intensity fields |
| Spelling duplicates | "distopia" next to "dystopia" | one form kept |

The remaining tags were split into **themes** (what the film is about) and **moods** (how the film feels and is told: tone, style, pacing, narrative structure). A tag was kept only if the genome rates it as clearly relevant (score ≥ 0.5) for at least 5 of the 1,957 catalog movies, so that the LLM is not offered labels that fit almost nothing. This removed 19 candidates (e.g. "mountain climbing" with 0 movies, "abortion" and "baseball" with 1). The genome was used only to choose which words enter the vocabulary, not to label any movie.

| | v2 | v4 |
|---|---|---|
| Themes | 44 | **246** |
| Moods | 18 | **103** |

1,389 of the 1,957 catalog movies have genome scores; most of the others were released after MovieLens 25M (2019).

---

## 3. Prompt v4

v4 differs from v2 only in the vocabulary. The system prompt, the audience and intensity fields, the maximum number of labels (5 themes, 3 moods), the input text, the model (gemma4:e4b) and all settings (temperature 0, seed 42, JSON schema) are identical. The user prompt changes only where v2 named labels that do not exist in the new lists: the two rules about specific v2 themes are replaced by one general rule ("Prefer the most specific themes that fit. Use broad themes such as 'friendship', 'death', 'love story' or 'good versus evil' only when they are a main focus of the story"), and the mood instruction adds "and how it is told". The JSON schema is built from the v4 lists, so the model can only answer with genome tags.

The vocabulary is part of the prompt version in the code (`PROMPTS["v4"]`); v2 was verified to be unchanged (identical schema and prompt).

### 3.1 Pilot (20 movies)

| Measure | Result |
|---|---|
| Valid answers | 20 / 20, no repairs |
| Labels confirmed by the genome (16 movies with genome scores) | 103 / 126 = **82%** |
| Seconds per movie | 1.99 (v2: about 1.3; the prompt is about twice as long, 1,513 tokens) |

Example, *The Godfather*: v2 "crime and criminals, family relationships, power and corruption, revenge, betrayal"; v4 "mafia, organized crime, crime gone awry, corruption, revenge". Other v4 labels: *Schindler's List* "holocaust, genocide, nazis"; *The Notebook* "love story, memory loss, aging"; *Brokeback Mountain* "homophobia, love story, sexuality"; *Spirited Away* moods "dreamlike, atmospheric, whimsical".

### 3.2 Full run

All 1,957 movies, 1 h 08 min (2.08 s per movie), 2 answers repaired by validation. Appending the v4 labels changes the embeddings as little as the v2 labels did (cosine between base and enriched vectors: mean 0.989).

---

## 4. Label quality

### 4.1 Specificity

| | v2 themes | v4 themes | v2 moods | v4 moods |
|---|---|---|---|---|
| Vocabulary used | 44 | 239 | 18 | 91 |
| Labels per movie | 4.36 | 4.82 | 2.99 | 3.00 |
| Most common label | identity and self-discovery (46%) | friendship (22%) | tense (44%) | dark (27%) |
| Average share of the catalog carrying an assigned label | 16.9% | **6.2%** | 22.6% | **10.2%** |

v4 labels are 2.7 times (themes) and 2.2 times (moods) more specific: an assigned v4 theme is shared on average by 6% of the catalog instead of 17%.

### 4.2 Accuracy against the Tag Genome

Share of assigned v4 labels with genome relevance ≥ 0.5 (1,388 movies with genome scores), compared with the share expected for a label drawn at random from the same list:

| | Labels assigned | Confirmed | Random label | Ratio |
|---|---|---|---|---|
| Themes | 6,724 | **62.9%** | 6.4% | about 10× |
| Moods | 4,164 | **69.2%** | 17.1% | about 4× |

These are the first ground-truth accuracy figures for the LLM labels in this thesis. The full-catalog figure is lower than in the pilot (82%), which contained mostly famous films. The v2 labels are not genome tags and cannot be checked in the same way. A genome relevance below 0.5 does not prove a label wrong (the genome itself is estimated from user tags, ratings and reviews), so the confirmed share is a conservative accuracy estimate.

---

## 5. Recommendation results

Same users, split, candidate pool and metrics as experiments 05 (warm) and 06 (cold). Wilcoxon signed-rank tests per user, Bonferroni-corrected over all 42 tests.

### 5.1 Warm start (10,490 users, NDCG@10)

| Representation | Max scoring | Mean scoring |
|---|---|---|
| Content, no labels | 0.2789 | 0.0625 |
| Content + v2 labels | 0.2824 | 0.0622 |
| **Content + v4 labels** | **0.2918** | **0.1027** |
| Embedding, no labels | 0.3152 | 0.2732 |
| **Embedding + v2 labels** | **0.3186** | **0.2892** |
| Embedding + v4 labels | 0.3169 | 0.2853 |
| Popularity | 0.4214 | |

| Comparison (b vs a) | Max scoring | Mean scoring |
|---|---|---|
| Content: v4 vs v2 | **+0.0094** (+3.3%) | **+0.0404** (+65%) |
| Content: v4 vs no labels | +0.0129 | +0.0402 |
| Content: v2 vs no labels | +0.0035 | −0.0003 (n.s.) |
| Embedding: v4 vs v2 | **−0.0017** | **−0.0039** |
| Embedding: v4 vs no labels | +0.0018 | +0.0121 |
| Embedding: v2 vs no labels | +0.0034 | +0.0160 |
| Content v4 vs embedding v2 | −0.0268 | −0.1865 |

All differences are significant after correction except content v2 vs no labels under mean scoring. Precision@10 and Recall@10 follow the same pattern (content max: precision 0.2314 → 0.2368 → 0.2431, recall 0.0766 → 0.0778 → 0.0803 for no labels, v2, v4).

### 5.2 Cold start (814 test users, NDCG@10)

| Method | N = 1 | N = 3 | N = 5 | N = 10 |
|---|---|---|---|---|
| Content, no labels (max) | 0.1820 | 0.2296 | 0.2553 | 0.3192 |
| Content + v2 (max) | 0.1551 | 0.2187 | 0.2502 | 0.3095 |
| Content + v4 (max) | 0.1674 | 0.2214 | 0.2503 | 0.3097 |
| Embedding, no labels (mean) | 0.2922 | 0.3798 | 0.4124 | 0.4540 |
| Embedding + v2 (mean) | 0.2922 | 0.3820 | 0.4147 | 0.4576 |
| Embedding + v4 (mean) | 0.2924 | 0.3819 | 0.4114 | 0.4581 |
| Hybrid embedding: none / v2 / v4 (same α) | 0.5697 / 0.5711 / 0.5709 | 0.5778 / 0.5768 / 0.5777 | 0.6001 / 0.5993 / 0.5992 | 0.6128 / 0.6133 / 0.6140 |
| Hybrid content: none / v2 / v4 (same α) | 0.5675 / 0.5668 / 0.5666 | 0.5698 / 0.5716 / 0.5703 | 0.5814 / 0.5830 / 0.5833 | 0.5857 / 0.5872 / 0.5917 |
| Popularity | 0.5598 | 0.5598 | 0.5598 | 0.5598 |

α was chosen on the 203 validation users for the v4 hybrid of each family (embedding: 0.05, 0.05, 0.15, 0.25; content: 0.02, 0.02, 0.05, 0.08 for N = 1, 3, 5, 10) and used for the v2 and no-label hybrids of the same family.

- **Hybrids:** v4 vs v2 and v4 vs no labels differ by at most 0.006 at every N; none is significant after correction.
- **Pure content, max scoring:** with one seed, v4 beats v2 (+0.012, significant after correction), but both are below content without labels (v4 vs none −0.015, significant). From N = 3 on, the differences are not significant.

---

## 6. Findings

1. **The larger vocabulary produces clearly better labels.** v4 labels are 2.2–2.7 times more specific and are confirmed by the genome for 63% (themes) and 69% (moods) of assignments, about 10× and 4× the rate of a random label. The local 4B model handles a list of 349 labels reliably (2 repairs in 1,957 answers).
2. **Content-based recommendation benefits most** (warm: +3.3% NDCG@10 under max scoring, +65% under mean scoring over v2 labels). This is the largest enrichment effect found in the thesis. A word-matching model needs specific, shared tokens: "mafia" links mafia films, whereas "family relationships" links a quarter of the catalog.
3. **Embedding-based recommendation does not benefit.** Embeddings with v4 labels score slightly below embeddings with v2 labels (−0.0017 max, −0.0039 mean), though above no labels. The embedding model already captures specific topics from the overview and keywords; the broad v2 categories appear to add a smoother grouping signal that suits embeddings better than specific tags.
4. **The best configurations are unchanged.** Warm: embeddinggemma with v2 labels and max scoring (0.3186); the best content model with v4 labels (0.2918) remains 0.027 below it. Cold: no hybrid changes significantly. Popularity remains far ahead offline (0.4214).
5. **The expectations were met:** labels more specific and mostly confirmed, embeddings within ±0.005, content-based changing most, no change in the cold-start hybrids.

**Interpretation.** Label quality can be improved substantially with an external, validated vocabulary, and the improvement reaches the recommender that uses the labels most directly. It does not reach the embedding-based recommender or the hybrids. Together with experiments 05–07, this supports the conclusion that in these configurations ranking accuracy is limited by the preference signal (popularity and what users co-like), not by the quality of the film descriptions.

---

## 7. Limitations

- **Genome accuracy is a conservative estimate.** The genome's relevance scores are themselves estimated, and relevant tags can have scores below 0.5; 568 catalog movies (mostly after 2019) have no genome scores.
- **Vocabulary choices were made by hand** (which tags count as themes or moods); a different filtering would give a somewhat different list. The filtering rules and the full list are saved for reproducibility.
- **The vocabulary contains narrative and style tags** ("twist ending", "non-linear", "slow paced") in the mood list, a different mix of attributes from v2's moods.
- **One enrichment model** (gemma4:e4b) and one embedding model. Whether a large model with v4 makes more accurate labels was not tested (an optional run with `minimax-m3.1-flash:cloud`, free until 22 October 2026, was prepared and tested on two movies but is not part of these results; it showed different labels between two identical calls at temperature 0).
- **The v4 prompt is about twice as long,** so enrichment takes about 60% longer per movie.
- **Offline evaluation on MovieLens** rewards popularity; the more specific labels may matter more for typed requests and explanations, which NDCG does not measure.

---

## 8. Consequences

- **Use v4 labels for content-based recommendation and for interpretable outputs** (explanations, filtering by specific topics), and keep v2 labels for the embedding-based recommender, where they score slightly higher.
- The genome-tag vocabulary makes the labels **checkable**: the same accuracy measure can be used for future label sets and models.
- The specific vocabulary is a natural fit for the planned **typed-request experiment**, where a request such as "a bleak, atmospheric film about memory loss" can be matched directly against labels.

---

## 9. Files

| File | Content |
|---|---|
| `data/ml-25m/genome/genome-tags.csv`, `genome-scores.csv`, `tags.csv` | Tag Genome files from MovieLens 25M |
| `vocabularies/v4_genome.json` | v4 themes and moods, filter settings, catalog movies per tag |
| `output/enrichment/gemma4_e4b_v4.jsonl` | v4 enrichment cache (one record per movie, with timings and validation issues) |
| `output/experiments/gemma4_e4b_v4/movies_enriched.csv`, `movie_texts.csv`, `movies_enriched_index.*` | v4 labels, texts and embeddings |
| `output/experiments/vocabulary/label_statistics.csv` | specificity of v2 and v4 labels |
| `output/experiments/vocabulary/genome_accuracy.csv` | genome accuracy of v4 labels and the random baseline |
| `output/experiments/vocabulary/warm_*`, `cold_*`, `settings.json` | recommendation results, tests and chosen α |

---

## 10. Reproducing the runs

```bash
python build_vocabulary_v4.py
```

```bash
python -c "from build_movie_texts import run_enrichment_experiment; run_enrichment_experiment('gemma4:e4b', prompt_version='v4')"
```

```bash
python vocabulary_experiment.py
```

Requirements: the genome files in `data/ml-25m/genome/` (from the MovieLens 25M archive), the v2 labels and indexes from the earlier experiments. The enrichment takes about 1 hour on the local GPU; the evaluation about 15 minutes. All random choices use seed 42.
