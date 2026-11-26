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
| 2026-11-23 | single_content_mean | val | 0.6319 | 0.2928 | 0.3143 | 0.3729 | 09 | cosine of candidate with mean MiniLM history vector; 0 for empty history |
| 2026-11-23 | hybrid_e_content_cumctr | val | 0.6603 | 0.2960 | 0.3201 | 0.3827 | 09 | content + cum_ctr, α per history group (0 / 0.3 / 0.4 / 0.5) chosen on val |
| 2026-11-23 | lgbm_minilm_pop15 | val | 0.6655 | 0.3045 | 0.3316 | 0.3933 | 11 | doc 04 ranker (trained Nov 9–13, not retrained), 5 MiniLM + 10 popularity features |
| 2026-11-23 | group_switch_auc | val | 0.6676 | 0.3030 | 0.3298 | 0.3917 | 12 | best val-AUC method per history group: popularity / Hybrid E / Hybrid E / LightGBM; chosen on val (in-sample) |
| 2026-11-23 | lgbm_pop15_base | val | 0.6650 | 0.3031 | 0.3302 | 0.3923 | 13 | doc 04 ranker retrained (batched input), no weights |
| 2026-11-23 | lgbm_pop15_w_sqrt | val | 0.6660 | 0.3019 | 0.3292 | 0.3912 | 13 | as base, impression weight ∝ 1/√(group share) |
| 2026-11-23 | lgbm_pop15_w_equal | val | 0.6666 | 0.3019 | 0.3295 | 0.3913 | 13 | as base, equal total weight per history group |
| 2026-11-23 | lgbm_pop_g0 | val | 0.5530 | 0.2253 | 0.2323 | 0.2983 | 13 | 10 popularity features only, trained on empty histories only |
| 2026-11-23 | lgbm_pop_all | val | 0.5606 | 0.2266 | 0.2347 | 0.3011 | 13 | 10 popularity features only, trained on all Nov 9–13 impressions |
| 2026-11-23 | lgbm_minilm_cumctr | val | 0.6733 | 0.3040 | 0.3320 | 0.3928 | 13 | 5 MiniLM features + cum_ctr |
| 2026-11-23 | lgbm_minilm_cumctr_mono | val | 0.6750 | 0.3047 | 0.3324 | 0.3935 | 13 | 5 MiniLM features + cum_ctr, monotone rising in cum_ctr |
| 2026-11-24 | nrms_user_minilm | val | 0.6531 | 0.3004 | 0.3256 | 0.3854 | 14 | learned user encoder (self-attention + pooling) on frozen MiniLM vectors; 4 negatives, 3 epochs; no popularity |
| 2026-11-24 | nrms_user_pop | val | 0.6786 | 0.3156 | 0.3431 | 0.4033 | 14 | + popularity term w[history group] · standardised log cum_ctr |
| 2026-11-24 | nrms_user_pop_pos | val | 0.6788 | 0.3151 | 0.3425 | 0.4029 | 14 | + learned recency vector per history slot |
| 2026-11-24 | nrms_user_pop_pos_neg8 | val | 0.6796 | 0.3167 | 0.3435 | 0.4038 | 14 | + 8 negatives per click |
| 2026-11-24 | nrms_user_pop_pos_neg8_long | val | 0.6758 | 0.3141 | 0.3407 | 0.4010 | 14 | as pop_pos_neg8, dropout 0.1, 5 epochs (epoch 2 chosen) |
| 2026-11-24 | nrms_user_pop_pos_neg8_no9 | val | 0.6765 | 0.3135 | 0.3406 | 0.4011 | 14 | as pop_pos_neg8, trained on Nov 10–13 only |
| 2026-11-24 | nrms_user_pop_pos_neg8_cscale | val | 0.6834 | 0.3192 | 0.3469 | 0.4070 | 14 | as pop_pos_neg8, + content weight per history group |
| 2026-11-24 | nrms_switch_g0_cumctr__pop_pos_neg8_cscale | val | 0.6840 | 0.3195 | 0.3474 | 0.4074 | 14 | empty history: cum_ctr; otherwise nrms_user_pop_pos_neg8_cscale (group-0 rule of doc 12) |
| 2026-11-25 | nrms_user_pop_pos_neg8_seed1 | val | 0.6809 | 0.3175 | 0.3453 | 0.4054 | 14 | seed check: pop_pos_neg8 with seed 1 |
| 2026-11-25 | nrms_user_pop_pos_neg8_cscale_seed1 | val | 0.6819 | 0.3179 | 0.3460 | 0.4063 | 14 | seed check: pop_pos_neg8_cscale with seed 1 |
| 2026-11-25 | nrms_user_pop_pos_neg8_seed2 | val | 0.6781 | 0.3140 | 0.3404 | 0.4012 | 14 | seed check: pop_pos_neg8 with seed 2 |
| 2026-11-25 | nrms_user_pop_pos_neg8_cscale_seed2 | val | 0.6825 | 0.3190 | 0.3464 | 0.4068 | 14 | seed check: pop_pos_neg8_cscale with seed 2; cscale > pop_pos_neg8 for all 3 seeds (mean 0.6826 vs 0.6795) |
| 2026-11-25 | nrms_cscale_seedavg3 | val | 0.6855 | 0.3209 | 0.3493 | 0.4094 | 15 | mean of within-impression standardised scores of pop_pos_neg8_cscale, seeds 42, 1, 2 |
| 2026-11-25 | nrms_cscale_seedavg3_switch | val | 0.6859 | 0.3212 | 0.3496 | 0.4098 | 15 | empty history: cum_ctr; otherwise nrms_cscale_seedavg3 |
| 2026-11-25 | blend_seedavg3_m6_equal | val | 0.6919 | 0.3195 | 0.3489 | 0.4105 | 15 | 0.5 × seed average + 0.5 × M6 (lgbm_minilm_cumctr_mono), both standardised within impressions; weight fixed in advance |
| 2026-11-25 | blend_seedavg3_m6_equal_switch | val | 0.6921 | 0.3197 | 0.3491 | 0.4107 | 15 | empty history: cum_ctr; otherwise blend_seedavg3_m6_equal |
| 2026-11-25 | nrms_user_pop_pos_neg8_cscale_cat | val | 0.6907 | 0.3230 | 0.3522 | 0.4131 | 16 | as pop_pos_neg8_cscale, + learned category and subcategory vectors (32 each) on the article input; epoch 3 of 3 |
| 2026-11-25 | nrms_user_pop_pos_neg8_cscale_fresh | val | 0.6854 | 0.3208 | 0.3489 | 0.4090 | 16 | as pop_pos_neg8_cscale, + age, showings in the last hour and older_than_log next to cum_ctr (weight per input and group); epoch 2 of 3 |
| 2026-11-25 | nrms_switch_g0_cumctr__pop_pos_neg8_cscale_cat | val | 0.6909 | 0.3233 | 0.3526 | 0.4134 | 16 | empty history: cum_ctr; otherwise nrms_user_pop_pos_neg8_cscale_cat |
| 2026-11-25 | nrms_user_pop_pos_neg8_cscale_cat_fresh | val | 0.6937 | 0.3255 | 0.3549 | 0.4159 | 16 | cat + fresh in one model; epoch 3 of 3; empty histories 0.6046 vs popularity 0.6049 |
| 2026-11-25 | nrms_switch_g0_cumctr__pop_pos_neg8_cscale_cat_fresh | val | 0.6937 | 0.3256 | 0.3551 | 0.4159 | 16 | empty history: cum_ctr; otherwise nrms_user_pop_pos_neg8_cscale_cat_fresh |
| 2026-11-26 | nrms_user_pop_pos_neg8_cscale_cat_fresh_seed1 | val | 0.6973 | 0.3271 | 0.3567 | 0.4179 | 16 | seed check: cat + fresh with seed 1; +0.0154 AUC over cscale seed 1 |
| 2026-11-26 | nrms_user_pop_pos_neg8_cscale_cat_fresh_seed2 | val | 0.6940 | 0.3251 | 0.3541 | 0.4156 | 16 | seed check: cat + fresh with seed 2; +0.0114 AUC over cscale seed 2; mean over 3 seeds 0.6950 / 0.4165 |
| 2026-11-26 | nrms_catfresh_seedavg3 | val | 0.6996 | 0.3293 | 0.3596 | **0.4208** | 16 | mean of within-impression standardised scores of cat + fresh, seeds 42, 1, 2; empty histories 0.6072 (popularity 0.6049) |
| 2026-11-26 | nrms_catfresh_seedavg3_switch | val | 0.6996 | **0.3294** | **0.3597** | **0.4208** | 16 | empty history: cum_ctr; otherwise nrms_catfresh_seedavg3 |
| 2026-11-26 | blend_catfresh_seedavg3_m6_equal | val | **0.7005** | 0.3248 | 0.3554 | 0.4174 | 16 | 0.5 × cat + fresh seed average + 0.5 × M6, both standardised within impressions; weight fixed in advance (doc 15 recipe); empty histories 0.6125 |
| 2026-11-26 | blend_catfresh_seedavg3_m6_equal_switch | val | 0.7003 | 0.3249 | 0.3555 | 0.4174 | 16 | empty history: cum_ctr; otherwise blend_catfresh_seedavg3_m6_equal |

**Internal validation (Nov 14, 431,517 impressions; scikit-learn nDCG, not `mind_eval`; not comparable with the table above):**

| Run | nDCG@5 | nDCG@10 | File |
|---|---|---|---|
| news_mlp_3feat | 0.2811 | 0.3411 | `output/news/news_mlp_3feat_internal_val.json` |
| lgbm_minilm_5feat | 0.3250 | 0.3832 | `output/news/lgbm_minilm_5feat_internal_val.json` |
