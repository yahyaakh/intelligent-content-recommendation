# Experiment: Label Quality against the Tag Genome, Vocabulary v5, and Its Effect on Recommendations

**Status:** finished (run dates: 2026-10-25)
**Scripts:** `label_quality_experiment.py` (label quality), `build_vocabulary_v5.py` (vocabulary v5), `enrich_catalog_deepseek_v5.py` (full catalogue), `label_set_experiment.py` (recommendation rerun); prompt `v5` in `llm_enricher.py`, alias mapping in `enrich_catalog.validate_output`
**Results folders:** `output/experiments/label_quality/`, `output/experiments/deepseek_v4.1_flash_v5/`, `output/experiments/label_sets/`
**Related documents:** `04_experiment_larger_enrichment_model.md`, `05_experiment_scoring_comparison.md`, `06_experiment_cold_start.md`, `08_experiment_larger_vocabulary.md`, `10_enrichment_vocabulary_improvement_plan.md`

---

## 1. Questions

Experiment 04 found that larger enrichment models (gemma4:31b, gpt-oss-120b) did not improve recommendations over the local gemma4:e4b, but it could not show whether their labels were actually better: there was no ground truth. Experiment 08 introduced the v4 vocabulary built from MovieLens Tag Genome tags, which makes labels checkable. This experiment asks:

1. **Does a stronger model assign more accurate labels?** DeepSeek V4.1 Flash vs gemma4:e4b, with the identical v4 prompt, checked against the Tag Genome.
2. **Does the v4 vocabulary miss concepts the models need?** If so, can a revised vocabulary (v5) fix this without losing verifiability?
3. **Do measurably better labels give better recommendations?** Warm and cold start, as in experiments 05, 06 and 08.

---

## 2. Label quality: DeepSeek V4.1 Flash vs gemma4:e4b (prompt v4)

### 2.1 Method

- **Models:** gemma4:e4b (the existing experiment-08 labels, local, schema enforced) and DeepSeek V4.1 Flash (Ollama cloud through the local installation, thinking off, temperature 0, seed 42). Identical prompt v4, vocabulary (246 themes, 103 moods), JSON schema, input text (title, year, genres, keywords, overview) and limits (5 themes, 3 moods).
- **Films:** all **1,388** catalogue films with genome scores (a 100-film stratified pilot first). Popularity thirds by TMDB vote count among these films (463 / 462 / 463).
- **Ground truth:** the genome relevance (0–1) of every vocabulary tag for every film.
- **Metrics per film and field (themes, moods), paired between the models:**

| Metric | Definition |
|---|---|
| Precision | share of assigned labels with genome relevance ≥ 0.5 (as in experiment 08) |
| Mean relevance | mean genome relevance of the assigned labels (continuous) |
| Recall | share of the genome's top 5 vocabulary themes (top 3 moods) that the model assigned |
| Commonness | mean share of genome films for which an assigned label has relevance ≥ 0.5 (lower = more specific) |
| Random precision | precision of a label drawn at random from the vocabulary |

Wilcoxon signed-rank tests per film. **Validity check:** gemma's precision (themes 0.625, moods 0.692) matches experiment 08's 62.9% and 69.2% (per-film vs pooled averaging).

### 2.2 Results (1,388 films)

| | Themes: gemma4:e4b | Themes: DeepSeek | Moods: gemma4:e4b | Moods: DeepSeek |
|---|---|---|---|---|
| **Precision** | 0.625 | **0.673** (+0.048) | 0.692 | **0.767** (+0.075) |
| **Mean relevance** | 0.601 | **0.634** (+0.033) | 0.628 | **0.678** (+0.050) |
| **Recall** | 0.379 | **0.397** (+0.018) | 0.278 | **0.344** (+0.066) |
| **Commonness** | 0.069 | **0.063** | 0.104 | **0.094** |
| Random precision | 0.064 | | 0.171 | |

All differences p < 0.001 (themes, recall: p = 0.0005). Films better / worse with DeepSeek: theme precision 581 / 359, mood precision 405 / 167. Agreement between the models (Jaccard): themes 0.42, moods 0.32.

**By popularity third (precision, themes / moods):**

| Third | gemma4:e4b | DeepSeek | Gain |
|---|---|---|---|
| Low (less-known films) | 0.539 / 0.634 | 0.596 / 0.718 | +0.057 / +0.084 |
| Mid | 0.609 / 0.688 | 0.668 / 0.770 | +0.060 / +0.083 |
| High (famous films) | 0.729 / 0.754 | 0.754 / 0.813 | +0.026 / +0.059 |

All significant in every third. The gain is largest for less-known films, where the small model is weakest.

### 2.3 Labels outside the vocabulary

The cloud endpoint does not strictly enforce the JSON schema (also seen in experiments 04 and 11): **404 of 1,388 answers (29%) contained 473 labels outside v4**, which validation removed (gemma, local with enforced schema: 2 repairs in 1,957 films). The reported precision is for the kept labels.

**Were the dropped labels wrong?** 112 of the 473 are genome tags outside the v4 vocabulary; **79.5% of these are confirmed by the genome** (mean relevance 0.73), more than DeepSeek's in-vocabulary labels. Dropping them therefore did not flatter DeepSeek, and they point to gaps in v4.

---

## 3. Vocabulary v5

### 3.1 The 473 dropped labels

| Group | Uses | Examples |
|---|---|---|
| Not a genome tag | 361 (198 labels) | grief, tragic, faith, jealousy, class conflict, ambition, loyalty, first love |
| Genome tag not among the v4 candidates | 85 (29 labels) | coming of age (20), satire (13), romance (7), surrealism, nostalgia, infidelity |
| Right label, wrong field | 23 | "tragedy" as a theme (a v4 mood), dark humor, inspirational |
| v4 candidate removed by the 5-film rule | 4 | movie business, competition |

"Coming of age" was an omission in v4 (adolescence and high school were included).

### 3.2 What v5 contains (`build_vocabulary_v5.py`)

Rules unchanged from v4: every label is an exact genome tag (verifiable), genres stay out (already in the input), and a tag is kept only if the genome rates it ≥ 0.5 for at least 5 catalogue films.

- **All 349 v4 labels**, unchanged.
- **9 new genome themes:** coming of age (146 catalogue films), infidelity (33), morality (151), humanity (94), voyeurism (21), poetry (14), undercover cop (27), video game (7), food (5) → **255 themes, 103 moods**.
- **11 aliases**, mapped by the answer validation instead of dropped: tragedy/tragic → *tragedy* (mood), dark humor, inspirational, surreal → mood; satire → *satirical*, surrealism → *surreal*, nostalgia → *nostalgic*, melancholy → *melancholic*, coming-of-age → *coming of age*, undercover → *undercover cop*. Labels moved from the other field are appended last, so they never displace the model's own choices.
- **Not added:** genres (romance, adventure, mystery, crime, suspense, dark comedy, parody, neo-noir), "idealism" (fits 341 catalogue films, too broad), and the 198 non-genome concepts (would make part of every answer unverifiable).

Prompt v5 uses the v4 prompt text with the v5 vocabulary.

### 3.3 v4 vs v5 (DeepSeek V4.1 Flash, 1,388 genome films; all runs scored against the v5 vocabulary)

| | v4 | v5 |
|---|---|---|
| Labels dropped | 473 (399 films) | **372 (326 films)** |
| Labels saved by an alias | — | 63 |
| Theme precision / mean relevance / recall | 0.673 / 0.634 / 0.394 | 0.675 / 0.637 / 0.400 |
| Mood precision / mean relevance / recall | 0.767 / 0.678 / 0.344 | 0.772 / 0.680 / 0.344 |

v5 − v4: theme mean relevance +0.003 (p = 0.04), theme recall +0.006 (p = 0.02); everything else not significant. **v5 reduces lost labels by 21% at equal or slightly better accuracy.**

**New themes in use** (genome-confirmed share): coming of age 109× (72%), morality 41× (**34%**), infidelity 25× (48%), humanity 24× (75%), undercover cop 11× (64%), voyeurism 6×, poetry, food, video game 3–4×; together 63%, similar to the rest of the vocabulary. "Morality" is the one weak addition; it was kept in v5 (decision 2026-10-25) and is noted here as a candidate for removal.

### 3.4 Full catalogue

All 1,957 films were labelled with DeepSeek Flash and prompt v5 (`output/experiments/deepseek_v4.1_flash_v5/movies_enriched.csv`, same format as the gemma tables): no film missing, every film with themes and moods; 512 labels dropped and 81 mapped over the whole catalogue; total cost ≈ $1.14.

DeepSeek's audience and intensity labels differ strongly from gemma's (as the larger models in experiment 04):

| | gemma4:e4b v4 | DeepSeek v5 |
|---|---|---|
| Audience adults / teens / family / children | 1,308 / 352 / 291 / 6 | 1,011 / 573 / 357 / 16 |
| Mean violence / language / sexual content (0–3) | 1.06 / 0.55 / 0.19 | 1.46 / 1.23 / 0.97 |

These are not checked against ground truth here (the genome has no comparable scale); a sample check against an external source (e.g. IMDb Parents Guide) is needed before using them as more accurate in the constraint layer.

---

## 4. Do better labels give better recommendations?

### 4.1 Method (`label_set_experiment.py`)

Same users, split, candidate pool and metrics as experiments 05/08 (warm: 10,490 users, random 75/25 split) and 06/08 (cold: 814 test users, N = 1, 3, 5, 10 seeds). Three label sets: gemma v2, gemma v4, DeepSeek v5. Content-based (A = no labels, B = with labels; labels as tokens) and embeddings (C = no labels, D = with labels; embeddinggemma of the base text plus "Themes … Mood …"), each with max and mean scoring. Cold hybrids: α chosen on the validation users for the v5 hybrid of each family and used unchanged for v2, v4 and no labels. Wilcoxon per user, Bonferroni over all 48 tests.

**Pre-registered expectations** (written in the script before running): warm embeddings v5 within ±0.005 of v4; content may gain most; cold hybrids no significant difference.

**Validity check:** all six warm results of experiment 08 (A, B_v2, B_v4, C, D_v2, D_v4; max and mean) were reproduced exactly.

### 4.2 Warm start (10,490 users, NDCG@10; popularity 0.4214)

| Method | No labels | gemma v2 | gemma v4 | DeepSeek v5 |
|---|---|---|---|---|
| Embedding, max | 0.3152 | **0.3186** | 0.3169 | 0.3184 |
| Embedding, mean | 0.2732 | **0.2892** | 0.2853 | 0.2835 |
| Content, max | 0.2789 | 0.2824 | **0.2918** | 0.2885 |
| Content, mean | 0.0625 | 0.0622 | **0.1027** | 0.0892 |

| Comparison | Max scoring | Mean scoring |
|---|---|---|
| D_v5 vs D_v4 | +0.0015 (sig.) | −0.0018 (sig.) |
| D_v5 vs D_v2 | −0.0002 (n.s.) | −0.0057 (sig.) |
| B_v5 vs B_v4 | **−0.0033** (sig.) | **−0.0135** (sig.) |
| B_v5 vs no labels (A) | +0.0096 (sig.) | +0.0266 (sig.) |

### 4.3 Cold start (814 test users, NDCG@10)

| Method | N = 1 | N = 3 | N = 5 | N = 10 |
|---|---|---|---|---|
| Hybrid embedding: none / v2 / v4 / v5 | 0.5697 / 0.5711 / 0.5709 / 0.5707 | 0.5778 / 0.5768 / 0.5777 / 0.5780 | 0.6001 / 0.5993 / 0.5992 / 0.5963 | 0.6128 / 0.6133 / 0.6140 / 0.6118 |
| Hybrid content: none / v2 / v4 / v5 | 0.5598 (all) | 0.5698 / 0.5716 / 0.5703 / 0.5702 | 0.5814 / 0.5830 / 0.5833 / 0.5833 | 0.5857 / 0.5872 / 0.5917 / 0.5890 |
| Embedding (mean): v4 / v5 | 0.2924 / 0.2859 | 0.3819 / 0.3766 | 0.4114 / 0.4091 | 0.4581 / 0.4523 |
| Content (max): v4 / v5 | 0.1674 / 0.1677 | 0.2214 / 0.2205 | 0.2503 / 0.2526 | 0.3097 / 0.3165 |
| Popularity | 0.5598 | 0.5598 | 0.5598 | 0.5598 |

At N = 1 the content hybrids select α = 0 (pure popularity). After correction, no hybrid comparison is significant (largest gap −0.003); the only significant cold difference is pure embedding v5 vs v4 at N = 1 (−0.0065).

---

## 5. Findings

1. **A stronger model labels films more accurately.** Against the Tag Genome, DeepSeek V4.1 Flash beats gemma4:e4b on every measure (theme precision +0.048, mood precision +0.075, higher recall, more specific labels), most of all on less-known films. This is the first ground-truth evidence for the enrichment model's label quality in this thesis.
2. **The v4 vocabulary had real gaps.** About a quarter of DeepSeek's off-list genome tags were missing concepts (above all "coming of age") or the right label in the wrong field. Vocabulary v5 (+9 genome themes, 11 aliases) cuts lost labels by 21% at equal or slightly better accuracy, while staying fully checkable against the genome.
3. **Better labels do not give better recommendations.** As pre-registered, DeepSeek v5 labels tie with the best previous label set for embeddings (warm 0.3184 vs v2 0.3186), are worse than gemma v4 for content-based matching (−0.003 / −0.014), and make no difference to the cold-start hybrids. Together with experiment 04, this shows directly that **label quality is not the bottleneck** for recommendation accuracy in this setup.
4. **More specific labels can hurt token matching.** DeepSeek's labels are shared by fewer films, so as content tokens they connect fewer films; accuracy against the genome and usefulness as matching features are different properties.
5. **The value of better labels lies elsewhere:** explanations shown to users, the constraint layer (audience and content intensity), and breadth (experiment 12: labels push content-based lists towards the long tail).

---

## 6. Limitations

- **The genome is an estimate** (from user tags, ratings and reviews), not perfect truth; precision is a conservative accuracy measure applied equally to both models. Only 1,388 of 1,957 films have genome scores.
- **One stronger model** (DeepSeek V4.1 Flash) and one small model; the cloud endpoint does not enforce the schema, so DeepSeek's answers were cleaned (29% with dropped labels under v4, 23% under v5).
- **198 non-genome concepts** (grief, faith, jealousy …) remain unavailable by design.
- **Audience and intensity labels** are not validated against ground truth.
- **Recommendation evaluation** uses one embedding model (embeddinggemma), labels appended as one sentence or as tokens, and the offline ground truth of experiments 05–08.

---

## 7. Files

| File | Content |
|---|---|
| `output/experiments/label_quality/labels_deepseek-v4.1-flash_cloud_v4.jsonl`, `…_v5.jsonl` | raw DeepSeek answers (prompt v4: 1,388 films; v5: all 1,957), with issues and token counts |
| `output/experiments/label_quality/per_film_all.csv`, `means_all.csv`, `tests_all.csv` | v4 comparison (gemma vs DeepSeek) |
| `output/experiments/label_quality/per_film_all_v5.csv`, `means_all_v5.csv`, `tests_all_v5.csv` | v5 comparison (gemma v4, DeepSeek v4, DeepSeek v5) |
| `output/experiments/label_quality/*_sample100.csv` | 100-film pilot |
| `vocabularies/v5_genome.json` | vocabulary v5 with aliases and catalogue counts |
| `output/experiments/deepseek_v4.1_flash_v5/movies_enriched.csv`, `movie_texts.csv`, `movies_enriched_index.*` | DeepSeek v5 labels for the whole catalogue, enriched texts, embedding index |
| `output/experiments/label_sets/` | warm and cold per-user results, significance tests, settings |

---

## 8. Reproducing the runs

```bash
python label_quality_experiment.py all
```

```bash
python build_vocabulary_v5.py
```

```bash
python label_quality_experiment.py all v5
```

```bash
python enrich_catalog_deepseek_v5.py
```

```bash
python label_set_experiment.py
```

DeepSeek is called as `deepseek-v4.1-flash:cloud` through a signed-in local Ollama installation (Ollama Pro). Answers are cached, so reruns make no new calls. The genome files are in `data/ml-25m/genome/`. Total API cost of this experiment at peak prices: about $1.94 (prompt v4 on the 1,388 genome films, including the 100-film pilot: $0.80; prompt v5 on all 1,957 films: $1.14).
