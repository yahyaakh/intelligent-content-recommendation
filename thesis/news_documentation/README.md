# News study documentation (MIND-large)

This folder documents the second dataset of the BSc thesis: **news recommendation on Microsoft MIND-large**. The news study repeats the movie study's main questions so that every idea is tested on two datasets. The movie study's documents are in `../thesis_documentation/` (docs 01–38).

**Main question (the same for both datasets):** *Where can a large language model help a recommender system with a new user?*

## Documents

| # | Document | Content |
|---|---|---|
| 01 | [01_dataset_mind.md](01_dataset_mind.md) | MIND-large: files, size, history lengths, new articles, how news differs from movies |
| 02 | [02_evaluation_protocol.md](02_evaluation_protocol.md) | `news/mind_eval.py`, official metrics, splits, the rule against label leakage |
| 03 | [03_popularity_features.md](03_popularity_features.md) | Time-aware popularity features (showings, earlier-day clicks, click rates) |
| 04 | [04_models_and_results_so_far.md](04_models_and_results_so_far.md) | Every news model built so far, how it works, its dev result and what it taught |
| 05 | [05_plan_news_study.md](05_plan_news_study.md) | Plan: each movie experiment mapped to a news version (N1–N11), protocol, cost, order of work |
| 06 | [06_protocol.md](06_protocol.md) | Protocol: Nov 14 validation, history groups, fixed LLM samples, paired tests; the MIND history field is fixed; existing models by history group |
| 07 | [07_candidate_coverage_and_article_age.md](07_candidate_coverage_and_article_age.md) | Do candidates have click history (two thirds do), does age matter (click rate 4.4% → 2.4%); age feature; popularity on validation |
| 08 | [08_covisitation_diagnosis.md](08_covisitation_diagnosis.md) | Why co-visitation is random: top-50 lists miss 97% of candidates; without them the score tracks read volume (ρ 0.86), not clicks |
| — | [results_log.md](results_log.md) | One row per run: every news result in one place, newest last |

New experiments get the next number (09, 10, …), one document each.

## Conventions

- **One document per experiment.** It contains the question, set-up, results and what the results mean, in the same format as the movie docs.
- **Every result also goes into [results_log.md](results_log.md).** Its numbers are copied from `output/news/results/<split>_summary.csv`, which `mind_eval.evaluate()` writes on every run. Never type a number by hand that is not in a results file.
- **Metrics:** the official MIND metrics (AUC, MRR, nDCG@5, nDCG@10, mean over impressions), always computed by `news/mind_eval.py`.
- **Results on disk:** `output/news/results/<split>/<name>/metrics.json` and `per_impression.csv.gz`. The per-impression scores allow paired tests between any two runs.
- **Code:** `news/`. Run Python with `C:\Users\yahya\miniconda3\envs\recomm_env\python.exe`.

## Status (2026-11-19)

- Done: data, evaluation module, popularity features, MiniLM embeddings, five model families on the full dev set (12 runs).
- Best so far: **LightGBM with 5 MiniLM/category features and 10 popularity features**, dev AUC 0.682 and nDCG@10 0.431 (random: 0.500 and 0.287).
- Protocol done (doc 06): `news/news_protocol.py`, LLM samples drawn, pre-registration skeleton in `output/news/final_test/preregistration.md`.
- First group result: for empty histories, popularity alone (`cum_ctr`, AUC 0.623) beats the best model (0.604).
- Doc 07: two thirds of candidates have earlier clicks; fresh articles are clicked more; popularity on validation AUC 0.608.
- Doc 08: item co-visitation/EASE dropped (they measure read volume, not click rate).
- Next: Hybrid E on validation, then a rate-based audience feature.
