# 09 — Hybrid E for news: content similarity + click rate

**Status:** done 2026-11-23 (validation only).
- **Code:** `news/hybrid_e.py` (`features`, then `val`).
- **Outputs:**
  - `output/news/features/content_mean_train.npy` (content score for every train slot);
  - `output/news/hybrid_e/` (`alpha_curve_val.csv`, `chosen_alpha.json`, paired tests `compare_*.csv`, `run_log.txt`);
  - runs `single_content_mean` and `hybrid_e_content_cumctr` in `output/news/results/val/`.
- **Plan item:** N2 (doc 05).

---

## 1. Question

In the movie study, Hybrid E (item-CF + popularity, α chosen per number of seed films) was the reference method for new users. Doc 08 showed that item co-visitation does not work on MIND, so the news version uses **content similarity** as the personal score. Two questions:

1. Does a personal content score beat popularity on news, where popularity was the hard bar for new users in the movie study?
2. How should the weight α depend on the length of the user's history?

## 2. Set-up

- **Data:** the validation day (Nov 14, 431,517 impressions, 17.4 million candidate slots), protocol of doc 06.
- **Personal score (`content_mean`):** cosine between the candidate's MiniLM vector and the mean MiniLM vector of the user's history. The vectors (`output/minilm_title_abstract_entities_mine`, all-MiniLM-L6-v2 on title + abstract + entity names) cover every train article. An empty history gets 0 for every candidate.
- **Popularity:** `cum_ctr`, the smoothed click rate over all earlier days (doc 03).
- **Hybrid E:** `α × personal + (1 − α) × popularity`, both min-max rescaled within each impression (the movie study's `rescale`).
- **α:** the movie grid (0, 0.02, …, 1.0), chosen **per history group** by mean AUC (the primary metric, doc 06); a tie keeps the smaller α. Group 0 has no personal score, so its α is set to 0 (any α < 1 gives the same ranking).
- **Tests:** paired per impression against `single_cum_ctr` (`npr.compare`: mean difference, 95% interval, Wilcoxon).
- **Check:** on 300 validation impressions the vectorised score equals a direct calculation (largest difference 6 × 10⁻⁸); empty histories score 0.

## 3. Results

### 3.1 Overall (val, all 431,517 impressions)

| Run | AUC | MRR | nDCG@5 | nDCG@10 |
|---|---|---|---|---|
| `single_cum_ctr` (popularity) | 0.6076 | 0.2571 | 0.2692 | 0.3309 |
| `single_content_mean` | 0.6319 | 0.2928 | 0.3143 | 0.3729 |
| `hybrid_e_content_cumctr` | **0.6603** | **0.2960** | **0.3201** | **0.3827** |

### 3.2 By history group (AUC; difference to popularity with 95% interval)

| Group | Impressions | Popularity | Content alone | Hybrid E | Hybrid E − popularity |
|---|---|---|---|---|---|
| 0 | 11,393 | 0.6049 | 0.5055 | 0.6049 | 0 (same ranking) |
| 1–3 | 29,744 | 0.6118 | 0.5858 | 0.6347 | +0.023 [0.021, 0.025] |
| 4–10 | 94,697 | 0.6127 | 0.6223 | 0.6559 | +0.043 [0.042, 0.045] |
| >10 | 295,683 | 0.6056 | 0.6444 | 0.6664 | +0.061 [0.060, 0.062] |
| all | 431,517 | 0.6076 | 0.6319 | **0.6603** | **+0.053 [0.052, 0.053]** |

All Hybrid E gains are significant (Wilcoxon p < 10⁻⁷⁸). On nDCG@10 the picture is the same: +0.052 overall, +0.019 / +0.039 / +0.061 in groups 1–3 / 4–10 / >10.

**Content alone against popularity:** worse with 0–3 history articles (AUC −0.099 and −0.026), better with 4–10 (+0.010) and >10 (+0.039).

### 3.3 The chosen α

| Group | Chosen α | AUC at α = 0 → chosen α → 1 |
|---|---|---|
| 1–3 | 0.3 | 0.612 → 0.635 → 0.586 |
| 4–10 | 0.4 | 0.613 → 0.656 → 0.622 |
| >10 | 0.5 | 0.606 → 0.666 → 0.644 |

The curves are flat near the top (for >10, α = 0.4–0.6 lies within 0.002 AUC), so the exact choice matters little. The full curve is in `alpha_curve_val.csv`.

## 4. What the results mean

1. **Mixing beats both parts in every group with history.** The two scores carry different information: content says what the user reads, popularity says what is being clicked today. Hybrid E gains +0.053 AUC over popularity overall.
2. **The weight on the personal score grows with history**, as in the movie study (more seed films → larger α). It is larger on news (0.3–0.5) than on movies (about 0.05–0.25). On movies, popularity was a strong signal for new users; on news, today's click rate is a weaker signal (AUC about 0.61), so the personal score is given more weight.
3. **A difference from the movie study:** content alone beats popularity for users with more than three history articles. On movies, content-based recommendation was far below popularity for new users (doc 06 of the movie study). A likely reason is that news interests are topical (sports, finance, a region), and MiniLM captures topics well.
4. **Empty histories stay at the popularity level (0.605).** Hybrid E cannot help them by construction. This is where an LLM or the user's own clicks from earlier days (doc 06, possible extra experiment) would have to add something.

**Caveats:**
- α was chosen on the same day it is reported on. The curves are flat near the optimum, so the optimism is small, but the clean number comes from the final test on dev.
- The existing LightGBM ranker (`lgbm_minilm_pop15`, dev AUC 0.682) has not been run on validation yet, so the two cannot be compared here.
- **For the final test:** 5.4% of dev candidate slots are articles without a MiniLM vector (they are not in the train news file). Their content score would be 0. Every dev and test article must be embedded before the final test. **Done in doc 10:** every train, dev and test article now has a vector (`output/minilm_title_abstract_entities_all/`), and `hybrid_e.py` uses it. The earlier dev results of doc 04 are not affected, because that run embedded its new dev articles on the fly.

## 5. Next

1. Embed the dev and test articles that have no vector (same model and text fields).
2. Run the LightGBM ranker on validation, so it can be compared with Hybrid E on the same day.
3. The rate-based audience feature of doc 08 §3.
4. The extra experiment of doc 06: add the user's own clicks from earlier days to the history, which would give most empty-history impressions a personal score.
