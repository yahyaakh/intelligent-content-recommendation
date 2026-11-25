# MIND leaderboard results (hidden test set)

This file holds every submission to the official MIND leaderboard, one row per upload, newest last. Scores are copied by hand from the Codabench result page exactly as shown. Validation numbers are copied from `results_log.md`.

- **Leaderboard:** Codabench competition 13967, "MIND News Recommendation Competition", phase **Official Test**.
  - Limits: 10 submissions per day, 999 in total. The phase never ends.
- **Test set:** MIND-large test, Nov 16–22, 2,370,727 impressions, labels hidden (`output/news/final_test/test_split_facts.json`).
- **Plan:** pre-registration §6 (`output/news/final_test/preregistration.md`). Each model is submitted once, and every score is reported, whatever it is.
- **Files:** `output/news/submissions/<name>/prediction.zip`, written and format-checked by `mind_eval.write_submission()`. The rankers are built by `news/leaderboard_submit.py`.

## Submissions

| # | Date (local) | Codabench ID | Model | Training data | Val AUC | Score (leaderboard) | AUC | MRR | nDCG@5 | nDCG@10 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 2026-11-23 20:13 | 1065488 | M1 popularity (`cum_ctr`) | none (no training) | 0.6076 | 0.5139 | 0.5139 | | | |
| 2 | 2026-11-23 20:52 | 1065503 | M6 LightGBM, 5 MiniLM + monotone `cum_ctr` | Nov 9–13 (not refit) | 0.6750 | **0.6679** | 0.6679 | 0.3306 | 0.3589 | 0.4147 |

- **Where the full metrics come from:** the official scoring output of each submission ("Output from scoring step" in the submission's details), stored as `output/news/submissions/<name>/scoring_result_<ID>.zip` and `scores_<ID>.json`.
- **The Score column is AUC:** for submission 1065503 the Score equals `scores.json` AUC 0.6679353515.
- **Still missing:** the scoring output of M1 (1065488). Its MRR and nDCG columns stay empty until it is downloaded.

## Notes per submission

1. **M1:** uploaded before pre-registration §6 was committed. M1 has no trained parameters and no open settings, so the order makes no difference. It is near random for the reason below.
2. **M6:** the model as trained on Nov 9–13, **not refit**. The refit proposal in the pre-registration was still open, and the author chose to see the result now.
   - Built by `news/leaderboard_submit.py m6` (log: `output/news/submissions/m6_build_log.txt`).
   - Test MiniLM features: `output/news/features/minilm5_test.npy`.
   - The score is close to the validation AUC (0.675) although the test set gives almost no click history, so the content features carry the model.

## Why the leaderboard is a different setting from dev

Test labels are hidden, so no clicks are known after Nov 15.
- **83.3% of test candidate slots have no earlier showing at all.** These are articles first seen in the test week. Their `cum_ctr` is the prior, a single constant, so their order inside an impression comes from the random tie-break.
- **On dev, only 5.4% of slots** are in that state.
- The numbers are in `output/news/final_test/popularity_coverage_dev_test.json`, computed from the features alone, without labels.

The leaderboard therefore measures recommendation **without click feedback** during the test week:
- click-rate popularity has almost nothing to work with;
- content and showing counts (known up to the impression time) still work.

Its scores are not comparable with the dev or validation results, and they do not test the hypotheses of the pre-registration. This check needs no labels and should have been run before the first upload.

Only 1.2% of test impressions have an empty history (dev: 3.0%), so the leaderboard mostly measures users with a long history.
