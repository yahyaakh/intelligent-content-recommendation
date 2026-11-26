# Pre-registration: news final test (MIND-large dev) and external leaderboard check

**Status: DRAFT (2026-11-23), not frozen.**
- The parts marked **OPEN** must be decided before the freeze.
- Methods still being built on validation (NRMS, LLM methods, semantic IDs) are added to §3 before the freeze, or are left out of the final test.
- **The freeze:** commit this file with the status changed to FROZEN. After the freeze, nothing in this file changes.
- **The runs:** every method listed is run **once** on dev, and every leaderboard model is submitted **once**.

Protocol details: `thesis/news_documentation/06_protocol.md`. Code: `news/news_protocol.py`. Validation evidence: news docs 09–13 and `thesis/news_documentation/results_log.md`.

---

## 1. Data

| Role | Data | Impressions |
|---|---|---|
| Training (validation stage) | train impressions of Nov 9–13 | 1,801,231 |
| Validation (all choices) | train impressions of Nov 14 (`val`) | 431,517 |
| **Final test** | MIND dev, Nov 15 (`dev`) | 376,471 |
| Final test, LLM methods | fixed sample `dev_llm`: 2,500 per history group | 10,000 |
| External check (§6) | MIND test, Nov 16–22, labels hidden | 2,370,727 (`output/news/final_test/test_split_facts.json`) |

- **Final models.** Every setting is fixed on validation (§3). **Proposed:** refit each learned model on Nov 9–14 (all of train) with exactly those settings before scoring dev.
  - Reason: doc 13 found that the relation between popularity features and clicks changes from day to day, so the day closest to the test should be in the training data.
  - Hybrid E and the switch have no trained parameters besides α and the per-group choice, which stay as fixed below.
  - **OPEN: confirm (a) refit on Nov 9–14, or (b) keep the Nov 9–13 models.**
- **Features:** clicks from earlier days only (doc 02 §4); showings up to the impression time. MiniLM vectors for every article: doc 10.
- **User profile:** the MIND history field. **OPEN:** whether the user's own clicks in earlier impressions are added (doc 06 §4). If this is to be tested, it must first be run on validation.

**Earlier use of dev.** 12 development runs were scored on dev before this protocol (doc 04, results_log.md). This includes the 15-feature LightGBM (M4) at AUC 0.682. Doc 06 also reported M4 against popularity by history group on dev. Those numbers are not final results. Every method below is run again under this protocol. Where a hypothesis was already seen on dev, §4 says so.

## 2. Metrics and groups

- **Primary metric: AUC.** It is the main metric of the MIND leaderboard.
- **Secondary metrics:** MRR, nDCG@5, nDCG@10. nDCG@10 is the one to compare with the movie chapter.
- **Report:** every result for all impressions and for history groups 0 / 1–3 / 4–10 / >10.
- **LLM sample:** per-group means, plus an estimate for the whole dev set weighted by the dev group shares (3.0% / 7.2% / 22.6% / 67.2%).

## 3. Methods

Each method's settings and validation result (AUC / nDCG@10, Nov 14) are shown below.

| # | Method | Settings fixed on validation | Val AUC / nDCG@10 | Source doc |
|---|---|---|---|---|
| M1 | Popularity: `cum_ctr` | smoothed click rate over all earlier days, m = 10 | 0.6076 / 0.3309 | 03, 07 |
| M2 | Content alone | cosine between the candidate and the mean MiniLM vector of the history; 0 for an empty history | 0.6319 / 0.3729 | 09 |
| M3 | Hybrid E | α·content + (1 − α)·`cum_ctr`, both min-max rescaled per impression; α per history group 0 / 0.3 / 0.4 / 0.5 (groups 0 / 1–3 / 4–10 / >10) | 0.6603 / 0.3827 | 09 |
| M4 | LightGBM, 15 features | lambdarank; 5 MiniLM + 10 popularity features; 100 trees, learning rate 0.05, 15 leaves, seed 42, other LightGBM defaults | 0.6655 / 0.3933 | 04, 11 |
| M5 | Per-group switch | group 0: M1; groups 1–3 and 4–10: M3; group >10: M4 | 0.6676 / 0.3917 | 12 |
| M6 | LightGBM, 5 MiniLM + monotone `cum_ctr` | as M4, but the only popularity feature is `cum_ctr`, with a rising monotone constraint | **0.6750 / 0.3935** | 13 |
| M7 | *optional:* XGBoost copy of M6 | **OPEN:** include as a library robustness check? It must first be run on validation with the same features and constraint | — | — |
| M8 | *candidate:* neural seed average (docs 14, 16) | The mean of the within-impression standardised scores ((score − mean) / sd, 0 if all equal) of `nrms_user_pop_pos_neg8_cscale_cat_fresh` trained with seeds 42, 1 and 2, for every impression, including empty histories (no switch). Model: frozen MiniLM vector + learned category and subcategory vectors (32 each) per article, projected to 256; self-attention + pooling over the last 50 clicks; 8 negatives; content weight per history group; popularity term with 4 inputs (log `cum_ctr`, log(1 + age in hours), log(1 + showings in the last hour), `older_than_log`; standardised on the training days) and one weight per input and group; AdamW 0.001, batch 512, dropout 0.2, 3 epochs, epoch per seed chosen on a 50,000-impression val sample (epoch 3 for all three seeds). No weight or setting of the average is tuned. Empty histories on val: 0.6072 vs M1 0.6049 (+0.0023, 95% CI −0.0027 to +0.0074). **OPEN:** include? It needs dev scoring code for the neural model and the refit decision of §1 (for a refit: each seed is trained for its chosen number of epochs, 3). **OPEN:** whether M8 gets hypotheses of its own (e.g. M8 vs M6 overall; M8 vs M1 for group 0, non-inferior), which would change the Bonferroni count. Earlier candidates, not chosen: the doc 15 blend + switch (0.6921 / 0.4107); the blend of this seed average with M6 (0.7005 / 0.4174, doc 16 §6b) | 0.6996 / 0.4208 | 14, 15, 16 |
| M9+ | LLM methods (N3–N5), semantic IDs (N7) | **OPEN:** add each with its fixed settings once its validation run exists; otherwise leave it out | — | 05 |

Validation selection was heavy: 17 α values per group for M3, 3 candidates per group for M5, 7 LightGBM variants for M6, and 10 neural variants plus 2 seed averages and 2 blends for M8. The cat + fresh variant and the choice of the seed average over the blend were made after the results were seen on val. The validation numbers above are therefore optimistic. That is why they are tested again here.

## 4. Hypotheses and tests

- **Test:** paired two-sided Wilcoxon signed-rank test per impression, plus the 95% interval of the mean difference.
- **Correction:** Bonferroni over H1–H6, so each test uses α = 0.05 / 6 ≈ 0.0083 (and a 99.17% interval for H3).
- **Equivalence margin:** ±0.005 AUC. **OPEN: confirm the margin.**

| # | Comparison | Group(s) | Expected direction | Validation evidence |
|---|---|---|---|---|
| H1 | M3 Hybrid E vs M1 popularity | all impressions | M3 > M1 | +0.053 AUC (doc 09) |
| H2 | M6 vs M4 | all impressions | M6 > M4 | +0.0095 vs the saved M4 (0.6750 vs 0.6655, doc 13) |
| H3 | M6 vs M1 | group 0 (empty history) | non-inferior: the lower bound of the 99.17% interval of M6 − M1 is above −0.005 | −0.0028 (95% CI −0.0054 to −0.0002), so borderline (doc 13) |
| H4 | M4 vs M1 | group 0 | M4 < M1 | −0.020 on val (doc 11). **Already seen on dev** in development (doc 06: 0.604 vs 0.623), so this is a confirmation, not a fresh test |
| H5 | M6 vs M5 switch | all impressions | M6 > M5 | +0.0073 (doc 13) |
| H6 | M2 content vs M1 popularity | groups 4–10 and >10, each tested | M2 > M1 | content beats popularity from 4 history articles (doc 09) |

H6 has two comparisons. **OPEN:** count them as one hypothesis with both groups required, or as two tests (Bonferroni over 7).

**Reported without a test:** every method in every group, on all four metrics. This includes M6 vs M3 and M6 vs M2.

## 5. What would change the conclusions

| # | Confirmed if | Rejected or weakened if | What it would mean |
|---|---|---|---|
| H1 | M3 − M1 > 0, p < 0.0083 | not significant, or M3 < M1 | content similarity does not add to popularity for news, against doc 09 |
| H2 | M6 − M4 > 0, p < 0.0083 | M6 ≤ M4 | the extra 9 popularity features fail on Nov 14 but not on Nov 15. The doc 13 finding is then specific to that day, and the thesis must say the feature choice does not carry over |
| H3 | lower bound > −0.005 | lower bound ≤ −0.005 | one learned ranker can (or cannot) replace popularity for new users; if not, the switch (M5) stays the practical answer for group 0 |
| H4 | M4 − M1 < 0, p < 0.0083 | not significant | the "popularity is the bar for new users" result does not hold for the 15-feature ranker after refitting |
| H5 | M6 − M5 > 0, p < 0.0083 | M6 ≤ M5 | a group switch is as good as or better than one ranker; the thesis recommends the switch |
| H6 | M2 − M1 > 0 in both groups, p < 0.0083 | either group fails | content alone needs a longer history on Nov 15 than on Nov 14 |

All results are reported, whichever way they go. No method is added, removed or changed after the freeze.

## 6. External check: MIND leaderboard (hidden test set)

- **Purpose:** an untouched test whose labels nobody can see, and a score that can be compared with published MIND results. It does **not** test H1–H6. The leaderboard returns only the means over all impressions, so there are no per-group results and no paired tests.
- **Where:** the official MIND leaderboard. `news/mind_eval.py` (`write_submission`) points to Codabench competition 13967, phase "Official Test". The old CodaLab page (competition 420) lists the Official Test phase with "Competition Ends: Never".
  - **Checked by the author on 2026-11-23** (registered, "My Submissions" tab, phase "Official Test"): the limit is 10 submissions per day and 999 in total. The phase never ends.
  - **OPEN:** whether a submission appears on the public leaderboard automatically or only when chosen (check after the first upload).
- **File:** `prediction.zip` containing only `prediction.txt`. Each line is `ImpressionID [r1,r2,...]`: the rank of each candidate in file order, where 1 is the top. It is written and format-checked by `mind_eval.write_submission()` into `output/news/submissions/<name>/`.
- **Submitted models, once each:**
  - M1 popularity;
  - M4 LightGBM 15;
  - M6 LightGBM 5 MiniLM + monotone `cum_ctr`.

  **Proposed:** refit M4 and M6 on Nov 9–15 (train + dev) with the settings of §3, because test starts on Nov 16. **OPEN: confirm.**
- **Test-week features:** popularity uses clicks up to Nov 15 only, because test labels are hidden; showings are known up to the impression time. Popularity therefore gets staler over the 7 test days. This is a harder setting than dev, so the scores are not comparable with the dev results.
- **No click feedback in the test week (found 2026-11-23 after the M1 upload, from features alone):**
  - 83.3% of test candidate slots have no earlier showing, so their `cum_ctr` is the prior (dev: 5.4%; `popularity_coverage_dev_test.json`).
  - M1 scored 0.5139 (`thesis/news_documentation/leaderboard_results.md`).
  - On the leaderboard, popularity is near random, and rankers can only use content and showing counts. The leaderboard therefore tests a different setting from dev and cannot be compared with it.
  - **OPEN:** which further models to submit, given this.
- **Composition:** only 1.2% of test impressions have an empty history (dev: 3.0%), and 74.8% have more than 10 history articles (`test_split_facts.json`). The test set mostly measures users with a long history.
- **Who uploads:** the files are prepared in this project; the upload is made by the author from their own leaderboard account.
- **Reporting:** all four metrics are reported as returned, for every submitted model, whatever their rank. No further submissions are made after the results are seen. If an upload is rejected for its format, it is fixed and uploaded again; that does not count as a second submission.
