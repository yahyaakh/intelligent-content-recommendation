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
| 09 | [09_hybrid_e.md](09_hybrid_e.md) | Hybrid E (MiniLM content + click rate, α per history group): val AUC 0.660, +0.053 over popularity; content alone beats popularity from 4 history articles |
| 10 | [10_embeddings_dev_test.md](10_embeddings_dev_test.md) | MiniLM vectors for all 28,852 dev/test articles missing from train; same text convention (CSV quote quirk), stored vectors reproduced exactly, 100% coverage |
| 11 | [11_lightgbm_val.md](11_lightgbm_val.md) | LightGBM (MiniLM + popularity) on val: AUC 0.666, best overall, but only from >10-article histories; Hybrid E better on AUC for 1–10, popularity for 0 |
| 12 | [12_group_switch.md](12_group_switch.md) | Per-group switch (popularity / Hybrid E / Hybrid E / LightGBM by val AUC): val AUC 0.668, +0.002 over LightGBM, but −0.002 nDCG@10; the empty-history fix is the robust part |
| 13 | [13_ranker_new_users.md](13_ranker_new_users.md) | Ranker for new users: weighting short histories makes them worse; learned popularity mixes don't carry to a new day; 5 MiniLM + monotone cum_ctr = best val run (AUC 0.675), within 0.003 of popularity for empty histories |
| 14 | [14_neural_user_encoder.md](14_neural_user_encoder.md) | Learned user encoder (NRMS-style, frozen MiniLM) + popularity + content weight per group: best val run (AUC 0.683, nDCG@10 0.407; +0.009 / +0.014 over M6 as a switch with popularity for empty histories); new users still below popularity in all 7 variants |
| 15 | [15_ensembles.md](15_ensembles.md) | Seed average of 3 neural runs (+0.002 AUC) and an equal blend with M6 LightGBM (+0.006 more): best val run AUC 0.692, nDCG@10 0.411 with popularity for empty histories; new users still below popularity |
| 16 | [16_category_and_freshness.md](16_category_and_freshness.md) | Neural encoder + learned category/subcategory vectors: +0.0073 AUC (0.6907), one model as good as the doc 15 ensemble on AUC and better on nDCG@10 (0.4134 as a switch); age and recent showings next to cum_ctr: +0.0020; both together: AUC 0.6937 / nDCG@10 0.4159, beats the ensemble, and matches popularity for empty histories (0.6046 vs 0.6049) |
| — | [leaderboard_results.md](leaderboard_results.md) | Every submission to the official MIND leaderboard (hidden test set): ID, model, score; why it is a different setting from dev |
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
- Doc 09 (2026-11-23): Hybrid E on val AUC 0.660 (+0.053 over popularity, α 0.3/0.4/0.5 by history group); empty histories stay at popularity.
- Doc 10 (2026-11-23): all 28,852 dev/test articles missing from train embedded (`output/minilm_title_abstract_entities_all/`); every train/dev/test candidate and history article has a vector.
- Doc 11 (2026-11-23): LightGBM on val AUC 0.666 / nDCG@10 0.393, +0.005 / +0.011 over Hybrid E overall; wins >10 history articles only; for 0 history it is 0.020 AUC below popularity.
- Doc 12 (2026-11-23): per-group switch on val AUC 0.668 (in-sample), +0.002 over LightGBM, −0.002 nDCG@10.
- Doc 13 (2026-11-23): best val run is LightGBM with 5 MiniLM features + monotone `cum_ctr` (AUC 0.675, nDCG@10 0.394; +0.007 AUC over the switch). The 9 other popularity features hurt on a new day; up-weighting new users makes them worse.
- Pre-registration drafted (2026-11-23, not frozen): `output/news/final_test/preregistration.md`. It covers methods M1–M6 with fixed settings, hypotheses H1–H6 (Bonferroni, ±0.005 AUC margin), and §6, the external check on the hidden MIND test set (M1, M4 and M6 submitted once each). Open decisions are marked OPEN.
- Doc 14 (2026-11-24): a learned user encoder on frozen MiniLM vectors (7 variants, GPU) with popularity and a content weight per history group is the best val model (AUC 0.6834). As a switch with popularity for empty histories: AUC 0.6840, nDCG@10 0.4074 (+0.0090 / +0.0140 over M6). No variant reaches popularity for empty histories; the learned content weight rises with history length, as Hybrid E's α.
- Doc 14 seed check (2026-11-25): the content weight per group beats `pop_pos_neg8` for each of 3 seeds (mean AUC 0.6826 vs 0.6795, nDCG@10 0.4067 vs 0.4035); seed spread 0.0015–0.0028 AUC; the seed-42 run is 0.0008 above the mean. The new-user gap holds for every seed.
- Doc 15 (2026-11-25): averaging the 3 neural seeds gives AUC 0.6855 (+0.0021 over seed 42); an equal-weight blend of that average with M6 gives 0.6919, and 0.6921 / nDCG@10 0.4107 with `cum_ctr` for empty histories (+0.017 / +0.017 over M6). The fixed weight 0.5 is within 0.0007 AUC of the in-sample best.
- Doc 16 (2026-11-25): learned category and subcategory vectors in the neural encoder give val AUC 0.6907 / nDCG@10 0.4131 (+0.0073 / +0.0062 over cscale, every group); as a switch 0.6909 / 0.4134. Age and recent showings next to cum_ctr: +0.0020 / +0.0020. One seed each; cat was still improving at its last epoch. Both in one model: AUC 0.6937 / nDCG@10 0.4159 (+0.0030 over cat), better than the doc 15 ensemble; for empty histories 0.6046 vs popularity 0.6049, the first model to reach popularity there. Seed check (2026-11-26): seeds 42 / 1 / 2 give 0.6937 / 0.6973 / 0.6940 (mean 0.6950, nDCG@10 0.4165), +0.010 to +0.015 AUC over cscale for each seed; no seed differs significantly from popularity for empty histories. Doc 15 recipe on it: seed average AUC 0.6996 / nDCG@10 0.4208 (best nDCG@10; non-inferior to popularity for empty histories, 0.6072 vs 0.6049); blend with M6 0.7005 / 0.4174 (best AUC; beats popularity for empty histories, +0.0077). M6 now adds only +0.0009 AUC and costs 0.0034 nDCG@10.
- Next: settle the OPEN items, including whether M8 (now the cat + fresh seed average of doc 16, in the pre-registration draft) is included (needs dev scoring code for the neural model and a refit decision); earlier-day clicks as history; rate-based audience feature; LLM methods on val (or leave them out of the final test).
