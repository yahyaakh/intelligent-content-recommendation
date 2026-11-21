# News results log

One row per run, newest last. The numbers are copied from `output/news/results/<split>_summary.csv` (written by `mind_eval.evaluate()`). Per-impression results are in `output/news/results/<split>/<name>/per_impression.csv.gz`.

**Split** says where the numbers come from:
- **dev (development):** runs before the protocol of doc 05;
- **val:** Nov 14, the validation day;
- **final:** the pre-registered final test.

| Date | Run name | Split | AUC | MRR | nDCG@5 | nDCG@10 | Doc | Notes |
|---|---|---|---|---|---|---|---|---|
| 2026-11-06 | random | dev (development) | 0.5001 | 0.2193 | 0.2241 | 0.2867 | 02 | uniform random scores, seed 1 |
| 2026-11-06 | file_order | dev (development) | 0.5001 | 0.2187 | 0.2233 | 0.2864 | 02 | candidates in `behaviors.tsv` order |
| 2026-11-06 | news_mlp_3feat | dev (development) | 0.6175 | 0.2974 | 0.3236 | 0.3837 | 04 | MLP: history size, category and subcategory match; trained on Nov 9–13 |
| 2026-11-06 | covisitation_top50 | dev (development) | 0.5040 | 0.2242 | 0.2300 | 0.2925 | 04 | item co-visitation, cosine-normalised shared readers, top 50 neighbours, fit on all of train |
| 2026-11-06 | lgbm_minilm_5feat | dev (development) | 0.6492 | 0.3184 | 0.3492 | 0.4081 | 04 | LightGBM lambdarank, 5 MiniLM/category features; trained on Nov 9–13 |
| 2026-11-11 | single_shown_1h | dev (development) | 0.5152 | 0.2224 | 0.2352 | 0.3019 | 03 | showings in the last hour |
| 2026-11-11 | single_shown_6h | dev (development) | 0.5135 | 0.2312 | 0.2500 | 0.3063 | 03 | showings in the last 6 hours |
| 2026-11-11 | single_shown_24h | dev (development) | 0.5137 | 0.2299 | 0.2437 | 0.3042 | 03 | showings in the last 24 hours |
| 2026-11-11 | single_shown_all | dev (development) | 0.5149 | 0.2270 | 0.2383 | 0.2999 | 03 | showings since Nov 9 |
| 2026-11-11 | single_prev_ctr | dev (development) | 0.6157 | 0.2812 | 0.3081 | 0.3661 | 03 | smoothed click rate of the day before (m = 10) |
| 2026-11-11 | single_cum_ctr | dev (development) | 0.6294 | 0.2870 | 0.3137 | 0.3730 | 03 | smoothed click rate over all earlier days (m = 10) |
| 2026-11-12 | lgbm_minilm_pop15 | dev (development) | **0.6821** | **0.3353** | **0.3723** | **0.4313** | 04 | LightGBM lambdarank, 5 MiniLM + 10 popularity features; trained on Nov 9–13 |
| 2026-11-19 | single_age | val | 0.5400 | 0.2144 | 0.2178 | 0.2781 | 07 | −age_hours (newest first); age = hours since first showing |
| 2026-11-19 | single_cum_ctr | val | 0.6076 | 0.2571 | 0.2692 | 0.3309 | 07 | popularity reference on validation |

**Internal validation (Nov 14, 431,517 impressions; scikit-learn nDCG, not `mind_eval`; not comparable with the table above):**

| Run | nDCG@5 | nDCG@10 | File |
|---|---|---|---|
| news_mlp_3feat | 0.2811 | 0.3411 | `output/news/news_mlp_3feat_internal_val.json` |
| lgbm_minilm_5feat | 0.3250 | 0.3832 | `output/news/lgbm_minilm_5feat_internal_val.json` |
