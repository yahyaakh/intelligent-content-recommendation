# Reproducing the report

This guide maps every table and figure of the TDK report (`thesis/thesis_latex/`) to the script, data and saved
outputs behind it. It separates two levels:

- **Recompute** — rebuild a table's numbers from saved outputs and cached LLM answers. Needs no API key,
  calls no model and never uses the test users again. This is what a reviewer should run.
- **Regenerate** — rerun an experiment from the raw data, including new LLM calls. Needs API keys, costs
  money (a few dollars in total), and gives slightly different LLM answers, because hosted models are not
  deterministic.

## 1. Version and environment

| Item | Value |
|---|---|
| Code | git commit `<fill in the hash of the submission commit>` (the last commit before this guide is `a400efe`, which froze the pre-registrations) |
| Python | 3.11.15 (conda env `recomm_env`) |
| Packages | `requirements.txt`, exact versions |
| OS used | Windows 11, NVIDIA RTX 4050 (6 GB) for the local models |
| Report | `thesis/thesis_latex/thesis.tex`, XeLaTeX + BibTeX (MiKTeX) |

```bash
conda create -n recomm_env python=3.11.15
conda activate recomm_env
pip install -r requirements.txt
```

All scripts are run from the repository root, for example `python experiments/evaluation/final_test.py check`.
Shared modules are in `src/`; every script in `experiments/` and `scripts/` adds `src/` to its import path.

## 2. Data

`data/` and `output/` are not in git (1.6 GB and 2.8 GB). They must be shipped separately, for example as an
archive next to the submission.

| Path | Content | Made by |
|---|---|---|
| `data/ml-25m/` | MovieLens 25M (`ratings.csv`, `links.csv`, `movies.csv`, `genome/`) | download from GroupLens |
| `data/data/processed_ml100k_v2/` | MovieLens 100K ratings mapped to TMDB ids (users 1–943) | `data/prepare_movielens.py` (run from inside `data/`; it writes to the relative path `data/processed_ml100k_v2`) |
| `data/data/processed/movies.csv` | the 1,957-film catalog with TMDB metadata | `data/fetch_tmdb.py`, `data/process_tmdb.py`, `scripts/fetch_credits_full.py` |
| `data/data/processed_30k/ratings.csv` | final ratings: ML-25M users with ≥ 500 ratings (ids + 943) and the 943 ML-100K users | `scripts/prepare_movielens25m.py` |
| `output/embeddings/` | FAISS indexes of the film embeddings (EmbeddingGemma) | `src/build_movie_texts.py` |
| `output/experiments/*/answers_*.jsonl`, `labels_*.jsonl` | cached LLM answers | the experiment scripts |

Check the data with `python scripts/data_statistics.py`. It prints Table 3.1 and the MovieLens 100K / 25M split
(943 / 28,378 and 9,713 / 2,519,312), and confirms the candidate pool of 1,547 films rated by warm users.

**User split.** `cold_start_experiment.split_users` (seed 42, 10 % cold users), `build_cold_user_data`
(≥ 10 liked + 5 more), and `split_validation` (20 % of the cold users are validation users: 203; test: 814).
`llm_rerank_experiment.load_data()` builds all of this, and almost every script uses it.

## 3. Models and keys

| Model | Used for | Access |
|---|---|---|
| gemma4:e4b | labels (Ch 5), typed requests, re-ranking | local Ollama |
| EmbeddingGemma (`embeddinggemma`) | film and request embeddings | local Ollama |
| DeepSeek V4.1 Flash | labels, re-ranking, safety labels, request parsing | Ollama cloud (`OLLAMA_API_KEY`) |
| Kimi K3, GLM-5.3 Flash, Mistral Large 4, DeepSeek V4 Pro, Nemotron 3 Super | re-ranking comparison (Table 6.2) | Ollama cloud |
| gpt-oss:20b and others | structured preferences (Table 6.1) | Ollama cloud |
| TMDB API | film metadata and US certifications | `TMDB_API_KEY` |

Keys go in a `.env` file in the repository root, which is git-ignored. Temperature 0 and seed 42 everywhere
unless a table says otherwise.

**Caution.** `src/build_movie_texts.py` currently defaults to `EMBED_PROVIDER = "gemini"`, which was a later
experiment. To regenerate the report's embeddings, switch it to the commented `ollama` / `embeddinggemma`
lines at the top of the file. The existing files in `output/embeddings/movies_*_embeddinggemma_v2.*` are the
ones the report uses.

## 4. Table and figure map

"Doc" refers to `thesis/thesis_documentation/NN_*.md`, which records each experiment's settings, decisions and
full results.

### Chapter 3 – Data and protocol

| Item | Script | Output | Doc |
|---|---|---|---|
| Table 3.1 | `scripts/data_statistics.py` | `output/experiments/data_statistics/table_3_1.csv` | – |

### Chapter 4 – Classical methods

| Item | Script | Output | Doc |
|---|---|---|---|
| Table 4.1 | `experiments/evaluation/final_test.py report` | `output/experiments/final_test/summary_test.csv` | 36 |
| Figure 4.1 | `scripts/make_chart_ch4.py` | `thesis/thesis_latex/figures/classical_ndcg.pdf` | 36 |
| EASE λ | `src/ease_tuning.py` | `output/experiments/ease/` | 26 |
| NCF / SVD fold-in | `src/cold_start_baselines.py` | `output/experiments/ncf_cold_start/` | 20 |

### Chapter 5 – LLM enrichment

| Item | Script | Output | Doc |
|---|---|---|---|
| Table A.6 (models) | `src/enrich_catalog.py`, `scripts/enrich_catalog_deepseek_v5.py` | `output/experiments/label_quality/labels_*.jsonl` | 14 |
| Table A.7 (label quality) | `experiments/enrichment/label_quality_experiment.py` | `output/experiments/label_quality/means_all*.csv` | 14 |
| Table 5.1 (new users) | `experiments/enrichment/label_set_validation.py` | `output/experiments/label_sets/validation/` | 14 |
| Test-user and warm-start label runs | `src/label_set_experiment.py` | `output/experiments/label_sets/` | 14 |
| Max vs mean scoring | `src/build_movie_texts.py` | `output/experiments/scoring_comparison/` | 05 |

### Chapter 6 – LLMs for new users

| Item | Script | Output | Doc |
|---|---|---|---|
| Table 6.1 (typed requests) | `src/typed_request_experiment.py`, `experiments/llm/structured_preferences_experiment.py`, `experiments/llm/request_rerank_experiment.py` | `output/experiments/typed_requests/`, `structured_preferences/`, `llm_rerank/request_rerank_*` | 09, 11 |
| Table 6.2 (models) | `src/llm_rerank_experiment.py`, `experiments/llm/kimi_experiments.py` | `output/experiments/llm_rerank/`, `kimi_k3/` | 13 |
| Prompt versions p1–p8 (Table A.3) | `src/llm_rerank_experiment.py` | `output/experiments/llm_rerank/` | 13, 17 |
| Order sensitivity | `experiments/llm/shuffle_average_experiment.py` | `output/experiments/shuffle_average/` | 13 |
| Fusion | `src/llm_fusion_experiment.py` | `output/experiments/llm_fusion/` | 13 |
| Table 6.3 (test) | `experiments/evaluation/final_test.py report` | `output/experiments/final_test/` | 36 |
| Re-ranking popularity (pre-registered) | `src/llm_rerank_experiment.py` | `output/experiments/llm_rerank/compare_per_user_test_814users_p2_n10_cpop.csv` | 13 §4.3 |
| Free generation | `experiments/llm/llm_generate_experiment.py` | `output/experiments/llm_generate/` | 13 |
| Figure 6.2 | `scripts/make_chart_ch6.py` | `thesis/thesis_latex/figures/llm_roles_ndcg.pdf` | 13 |
| Table 6.4 (beyond accuracy) | `src/beyond_accuracy.py`, `experiments/llm/beyond_accuracy_llm.py` | `output/experiments/beyond_accuracy/`, `beyond_accuracy_llm/` | 12 |

### Chapter 7 – Learning to rank

| Item | Script | Output | Doc |
|---|---|---|---|
| Tables 7.2, 7.3, A.8 | `src/LightGBM_XBoost_rerank_experiment.py` | `output/experiments_ LightGBM_XBOOST/ltr_boosting/` | 15, 16 |
| Co-visitation features (G5) | `src/LightGBM_XBoost_rerank_experiment.py` (function computing `covis`) | – | 16 |
| Table 7.4 (pre-registered test) | `src/LightGBM_XBoost_rerank_experiment.py` (step 6) | `ltr_boosting/step6_*` (incl. `step6_preregistration.md`) | 16 |
| Attempts (Table A.4) | `experiments/ltr/ltr_improvements_experiment.py`, `src/ltr_ncf_knn_experiment.py`, `experiments/ltr/ltr_ease_experiment.py`, `experiments/semantic_ids/ltr_semantic_id_experiment.py` | `output/experiments/ltr_*/` | 24, 26, 27 |

### Chapter 8 – Semantic IDs

| Item | Script | Output | Doc |
|---|---|---|---|
| Algorithm 1 | `src/recomm.py` (class `Semantic_IDs`) | – | 27 |
| Figure 8.1 | `experiments/semantic_ids/sid_tree_examples.py` | printed | 27 |
| Table 8.1 (lift) | `experiments/semantic_ids/semantic_id_eval.py`, `experiments/semantic_ids/sid_robustness.py` | `output/experiments/semantic_ids/`, `sid_robustness/` | 27, 33 |
| Table 8.2 (vector sources) | `src/sid_vector_sources.py`, `experiments/semantic_ids/sid_robustness.py` | `output/experiments/sid_vector_sources/`, `sid_robustness/` | 30, 33 |
| Models (Table A.5) | `experiments/semantic_ids/semantic_id_models.py` | `output/experiments/semantic_id_models/` | 28 |
| New films | `experiments/semantic_ids/new_item_test.py` | `output/experiments/new_item_test/` | 31 |
| Table A.9 (rating prediction) | `experiments/semantic_ids/sid_rating_experiment.py` | `output/experiments/sid_rating/` | 34 |

### Chapter 9 – Constraints

| Item | Script | Output | Doc |
|---|---|---|---|
| Child-safety signals | `experiments/constraints/safety_labels_experiment.py` | `output/experiments/safety_labels/` | 21 |
| Tables 9.2, A.10 | `experiments/constraints/constraint_parsing_experiment.py run` | `output/experiments/constraint_parsing/` | 37 |
| Web application | `web/` (Django), methods in `web/recommender/thesis_engine.py` | – | 25 |

### Chapter 10 – Overall evaluation

| Item | Script | Output | Doc |
|---|---|---|---|
| Table 10.1 | `experiments/evaluation/final_test.py report` | `output/experiments/final_test/summary_test.csv`, `tests_test.csv` | 36 |
| Figure 10.1 | `scripts/example_user.py` | `output/experiments/example_user/example_users.json` | – |
| Table A.11 (N = 50) | `experiments/evaluation/final_test.py report` | `output/experiments/final_test/` | 36 |
| Standard warm start | `experiments/evaluation/standard_warm_benchmark.py` | `output/experiments/standard_warm/` | 35 |
| Table A.12 (cost and time) | `scripts/timing_experiment.py` | `output/experiments/timing/`, `final_test/llm_usage_test.csv` | 13, 37 |

## 5. Recompute the headline results (no LLM calls, no new test-user run)

```bash
python scripts/data_statistics.py            # Table 3.1
python experiments/evaluation/final_test.py check           # both boards on the validation users, compared with the saved results
python experiments/evaluation/final_test.py report          # tests of Tables 4.1, 6.3, 10.1, A.11 from the saved test files
python scripts/example_user.py               # Figure 10.1
python scripts/make_chart_ch4.py             # Figure 4.1
python scripts/make_chart_ch6.py             # Figure 6.2
```

`output/experiments/final_test/per_user_test.csv` holds every per-user score of the final test. The means
in `summary_test.csv` can be recomputed from it directly.

## 6. Known deviations

These are also stated in the report or in the documentation.

- **Test users were used in several pre-registered tests**, not once: docs 13 (LLM re-ranking), 15 and 16
  (learned ranker), and 36 (final test). The final test re-ran the earlier results as checks, and all 7 were
  reproduced exactly. A few exploratory test-user runs (docs 05/14, 09/11, 12) are marked as such in the
  report. Amendment 1 of the final pre-registration is not in git, so its date cannot be proven.
- **Learned ranker, S = 50:** scoring from the saved feature tables gives 0.4957 on validation, against 0.4926
  when scored from memory (doc 36 §1.1). The likely cause is float precision in the saved tables. The test
  uses the saved-file version.
- **Embedding row of Table 10.1** uses max scoring and the `v2` label embeddings, not the mean scoring
  selected in Chapter 5. That setting was not part of the final test.
- **Hosted LLMs change over time.** Regenerating answers will not give identical numbers. The cached
  `.jsonl` answers are the reference.
