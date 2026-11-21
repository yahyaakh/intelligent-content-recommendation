# Intelligent Content Recommendation System

**Where can a large language model help a recommender system with a new user?**

This repository holds the code, experiment write-ups and pre-registrations of my BSc research at the
Budapest University of Technology and Economics (BME). It compares classical recommenders, a learned
ranker, LLMs and semantic IDs for users who have given only a few liked films.

**2nd place, BME TDK Conference 2026 · qualified for the National Student Research Conference (OTDK) 2027.**
[Paper and talk on the TDK site](https://tdk.bme.hu/conference/VIK/2026/sessions/uzleti/paper/Intelligent-Content-Recommendation-System)

---

## Results: pre-registered final test (814 held-out new users)

The data is MovieLens 25M joined with TMDB metadata: 10,656 users, 2.55 million ratings and a catalog of
1,957 films. Each new user names 10 liked films, and the methods rank the films they rated later
(NDCG@10, relevance = rating ≥ the user's mean).

| Method | NDCG@10 | vs Hybrid E |
|---|---|---|
| **Learning-to-rank (XGBoost + LightGBM ensemble)** | **0.661** | **+0.030** |
| EASE | 0.651 | +0.020 |
| User-kNN | 0.646 | +0.015 |
| Hybrid E (item-CF + popularity, the reference) | 0.631 | – |
| Best LLM reranker (DeepSeek V4.1 Flash) | 0.624 | −0.007 |
| Item-CF | 0.617 | −0.015 |
| Neural CF (fold-in) | 0.598 | −0.033 |
| Popularity | 0.560 | −0.071 |
| SVD (fold-in) | 0.526 | −0.105 |
| Content (embeddings + LLM labels) | 0.405 | −0.226 |

Full table, 3-seed and 50-seed boards: [`36_final_test.md`](thesis/thesis_documentation/36_final_test.md) and
[`summary_test.csv`](output/experiments/final_test/summary_test.csv).

**Main findings**

1. **The learned ranker is the best method.** It beats Hybrid E by +0.030 NDCG@10 (95% CI [+0.022, +0.038],
   p = 2 × 10⁻⁹ after Bonferroni correction). With 50 liked films the gap grows to +0.044.
2. **LLMs do not beat collaborative filtering as rerankers.** Twelve models from eight families were tried;
   on the test users every LLM reranker scored 0.007–0.014 below Hybrid E, and fusing an LLM with Hybrid E
   added nothing (+0.001, not significant).
3. **An LLM can personalise without rating data.** Reranking a popularity list from the 10 named films gives
   +0.040 over popularity (p = 1.5 × 10⁻⁶), 56% of the gain that collaborative filtering achieves.
4. **LLM labels are useful for filtering.** LLM audience labels found films unsuitable for children with
   F1 0.905, against 0.696 for the rule-based filter they replaced.
5. **Semantic IDs mostly do not help ranking**, but codes built from ratings beat codes built from content.
   Adding semantic-ID tokens to matrix factorisation lowered RMSE on rarely rated films by 0.021 (5 of 5 runs).

**How the evaluation was kept honest**

- Every setting was chosen on 203 validation users or on simulated users. The 814 test users were scored once,
  after the test plan was written down ([pre-registrations](output/experiments/)).
- Paired Wilcoxon tests, bootstrap confidence intervals over users, Bonferroni correction per test family.
- Negative results are reported with the same detail as positive ones. Each experiment has its own write-up
  in [`thesis/thesis_documentation/`](thesis/thesis_documentation/) (docs 01–38).

## In progress: news recommendation (Microsoft MIND)

The BSc thesis repeats the study on a second dataset, MIND-large (2.2 million training impressions,
104,151 articles). It has its own evaluator with the official metrics, time-aware popularity features built
so that no label from the same day leaks into a feature, and a fixed validation protocol. Best so far:
a LightGBM ranker with MiniLM content features and popularity features, AUC 0.682 on MIND dev (random: 0.500).
Write-ups: [`thesis/news_documentation/`](thesis/news_documentation/).

---

## Layout

| Folder | Content |
|---|---|
| `src/` | Shared modules: data loading and user split (`cold_start_experiment.py`, `llm_rerank_experiment.load_data`), film texts and metrics (`build_movie_texts.py`), classical methods (`recomm.py`, `cold_start_baselines.py`, `ease_tuning.py`), the learned ranker (`LightGBM_XBoost_rerank_experiment.py`), LLM labelling (`llm_enricher.py`, `enrich_catalog.py`) |
| `experiments/enrichment/` | LLM labels: quality against the MovieLens Tag Genome, effect on recommendations, vocabularies, model comparisons |
| `experiments/llm/` | LLMs for new users: typed requests, reranking variants, free generation, verification |
| `experiments/ltr/` | Attempts to improve the learned ranker, NCF experiments |
| `experiments/semantic_ids/` | Semantic-ID construction, evaluation, models, new films, rating prediction |
| `experiments/constraints/` | Request parsing into constraints, child-safety labels |
| `experiments/evaluation/` | Final pre-registered test, standard warm-start benchmark |
| `scripts/` | Data preparation, vocabulary building, figures, timing |
| `news/` | MIND news recommendation: `mind_eval.py` is the one evaluator for every news model; `news_protocol.py` holds the validation protocol |
| `web/` | Django web application (`web/recommender/thesis_engine.py` holds the report's methods) |
| `thesis/` | Experiment write-ups for movies (`thesis_documentation/`) and news (`news_documentation/`) |
| `output/` | Only the pre-registrations and final summary tables are in git; the rest (cached LLM answers, embeddings, models) is about 15 GB |
| `data/vocabularies/` | The label vocabularies; the datasets themselves are not in git |
| `archive/` | Earlier scripts kept for reference |

## Running

```bash
conda create -n recomm_env python=3.11.15
conda activate recomm_env
pip install -r requirements.txt
cp .env.example .env    # add the API keys you need (TMDB for data; LLM providers only for new LLM calls)
```

Run every script from the repository root. Scripts in `experiments/` and `scripts/` add `src/` to their
import path themselves. For example:

```bash
python scripts/data_statistics.py
python experiments/evaluation/final_test.py check
python web/check_thesis_engine.py
```

[`REPRODUCE.md`](REPRODUCE.md) maps every table and figure of the report to the script and output behind it.

## Data and licences

The datasets are not redistributed here. Download them from their sources:

- [MovieLens 25M](https://grouplens.org/datasets/movielens/25m/) (GroupLens) and MovieLens 100K
- [TMDB API](https://developer.themoviedb.org/) for film metadata (needs a free API key)
- [MIND](https://msnews.github.io/) (Microsoft News Dataset): `python scripts/download_mind.py`
