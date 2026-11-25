# 11 — LightGBM (MiniLM + popularity) on the validation day

**Status:** done 2026-11-23 (validation only).
- **Code:** `news/lgbm_val.py` (`check`, then `val`).
- **Outputs:**
  - `output/news/lgbm_val/`:
    - `feature_check.json`;
    - `minilm5_val.npy` (the 5 MiniLM features for every val slot);
    - `feature_importance.csv`;
    - paired tests `compare_*.csv`;
  - run log `output/news/lgbm_val_log.txt`;
  - run `lgbm_minilm_pop15` in `output/news/results/val/`.

---

## 1. Question

Doc 04's best model is LightGBM lambdarank with 5 MiniLM features and 10 popularity features (dev AUC 0.682). Doc 09's Hybrid E was only measured on validation (AUC 0.660). Doc 06 measured LightGBM on dev only. Two questions:

1. How does LightGBM score on the validation day, next to Hybrid E and popularity?
2. For which history groups is it better?

## 2. Set-up

- **Model:** the saved ranker `output/news/lgbm_minilm_pop15.txt`, which `news/SemanticContentModel.py` trained on the train impressions before Nov 14.
  - The validation day was not used to fit it. **It is not retrained or tuned here.**
  - Settings: 100 trees, learning rate 0.05, 15 leaves.
- **Features:**
  - The 5 MiniLM features have the definitions of doc 04 §2.3: `top5_similarity`, `max_similarity`, `avg_profile_similarity`, `category_preference` and `history_length`.
    - They are computed in a vectorised loop in `lgbm_val.py`, because the original script runs at import time.
    - The vectors come from doc 10 and cover every article.
  - The 10 popularity features of doc 03 complete the 15 columns.
- **Check of the features:**
  - On 5,000 random training impressions (182,842 candidates), the recomputed features match the training matrix that the original code saved (`output/lightgbm_full_features_entities_v1/X_train.npy`).
  - Largest difference: 7 × 10⁻⁷ for the similarity features, 0 for `category_preference` and `history_length`.
- **Data and tests:**
  - Validation day: Nov 14, 431,517 impressions, 17.4 million slots, protocol of doc 06.
  - Each comparison is paired per impression (`npr.compare`: mean difference, 95% interval, Wilcoxon).
- **Advantage to Hybrid E:** Hybrid E's α values were chosen on this same day. The comparison therefore slightly favours Hybrid E, not LightGBM.

## 3. Results

### 3.1 Overall

| Run | AUC | MRR | nDCG@5 | nDCG@10 |
|---|---|---|---|---|
| `single_cum_ctr` (popularity) | 0.6076 | 0.2571 | 0.2692 | 0.3309 |
| `single_content_mean` | 0.6319 | 0.2928 | 0.3143 | 0.3729 |
| `hybrid_e_content_cumctr` | 0.6603 | 0.2960 | 0.3201 | 0.3827 |
| `lgbm_minilm_pop15` | **0.6655** | **0.3045** | **0.3316** | **0.3933** |

### 3.2 By history group: AUC

In group 0, Hybrid E is the same as popularity (α = 0).

| Group | Impressions | Popularity | Hybrid E | LightGBM | LightGBM − Hybrid E (95% CI) | Wilcoxon p |
|---|---|---|---|---|---|---|
| 0 | 11,393 | 0.6049 | **0.6049** | 0.5853 | −0.0196 (−0.0236, −0.0156) | 3 × 10⁻²⁷ |
| 1-3 | 29,744 | 0.6118 | **0.6347** | 0.6248 | −0.0099 (−0.0127, −0.0071) | 4 × 10⁻⁹ |
| 4-10 | 94,697 | 0.6127 | **0.6559** | 0.6516 | −0.0043 (−0.0057, −0.0028) | 0.005 |
| >10 | 295,683 | 0.6056 | 0.6664 | **0.6771** | +0.0108 (+0.0101, +0.0114) | < 10⁻³⁰⁰ |
| all | 431,517 | 0.6076 | 0.6603 | **0.6655** | +0.0052 (+0.0046, +0.0058) | < 10⁻³⁰⁰ |

### 3.3 By history group: nDCG@10

| Group | Popularity | Hybrid E | LightGBM | LightGBM − Hybrid E (95% CI) | Wilcoxon p |
|---|---|---|---|---|---|
| 0 | 0.3288 | **0.3288** | 0.3193 | −0.0095 (−0.0129, −0.0061) | 2 × 10⁻¹⁰ |
| 1-3 | 0.3698 | 0.3885 | 0.3923 | +0.0038 (+0.0014, +0.0062) | 0.07 |
| 4-10 | 0.3584 | 0.3970 | **0.4045** | +0.0076 (+0.0063, +0.0088) | 1 × 10⁻³⁰ |
| >10 | 0.3183 | 0.3797 | **0.3927** | +0.0130 (+0.0123, +0.0137) | < 10⁻³⁰⁰ |
| all | 0.3309 | 0.3827 | **0.3933** | +0.0106 (+0.0100, +0.0112) | < 10⁻³⁰⁰ |

LightGBM against popularity (AUC): −0.020 in group 0, then +0.013, +0.039 and +0.072 in groups 1-3, 4-10 and >10, and +0.058 overall. These figures are from `compare_lgbm_minilm_pop15_vs_single_cum_ctr_AUC.csv`.

### 3.4 Feature importance (share of total gain)

| Feature | Share |
|---|---|
| `prev_ctr` | 0.222 |
| `avg_profile_similarity` | 0.217 |
| `cum_ctr` | 0.131 |
| `max_similarity` | 0.116 |
| `cum_shows` | 0.056 |
| `prev_shows` | 0.056 |
| `shown_1h` | 0.051 |
| `top5_similarity` | 0.045 |
| `category_preference` | 0.037 |
| `history_length` | 0.030 |
| the other five popularity features | 0.039 together |

## 4. What this means

1. **LightGBM is the best model on validation overall**: AUC +0.005 and nDCG@10 +0.011 over Hybrid E, both significant. Hybrid E was tuned on this day and LightGBM was not, so the lead is not an artefact of tuning.
2. **The lead comes only from long histories.** The >10 group holds 69% of the validation impressions, and LightGBM wins it clearly. For 1–10 history articles, Hybrid E is better on AUC. For an empty history, LightGBM is below popularity by 0.020 AUC. This is the same result as on dev (doc 06, 0.604 vs 0.623) and the same as in the movie study: for new users, the learned ranker does not beat the simple popularity-based method. A likely cause: only about 2% of training impressions have an empty history (doc 06).
3. **AUC and nDCG@10 disagree for groups 1-3 and 4-10.** LightGBM is trained with lambdarank, which optimises nDCG, so it places the clicked article higher in the top 10. Hybrid E orders the whole list slightly better, which is what AUC measures. AUC is the primary metric (doc 06), so in these groups Hybrid E is the better method by the protocol.
4. **The ranker mostly relies on the same two signals as Hybrid E.** The click rates (`prev_ctr`, `cum_ctr`) and the cosine with the mean history vector (`avg_profile_similarity`) carry 57% of the gain. The extra 13 features add +0.005 AUC overall.

## 5. Next steps

- **Per-group switch:** popularity for an empty history, Hybrid E for 1–10 articles and LightGBM for >10, chosen on validation by AUC. On validation this would give each group its best value from the tables above. It must be listed in the pre-registration as a method chosen on val.
- **A ranker for new users:** add the Hybrid E score and the history group as features, or give more weight to short-history impressions, and retrain on Nov 9–13. The goal is one model that is not worse than popularity for an empty history.
- **Still open from doc 09:** a rate-based audience feature, and earlier-day clicks as history (which would make many empty histories non-empty).
