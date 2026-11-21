# 03 — Popularity features

**Status:** done (2026-11-11/12). Code: `news/popularity_features.py`. Saved features: `output/news/features/<name>_<split>.npy`, one value per candidate in `mind_eval` order.

---

## 1. Why

News popularity changes by the hour. An all-time count is meaningless for an article published this morning. So popularity is measured **at the moment of the impression**, in two ways:
1. how often the article was **shown** recently;
2. how often it was **clicked** when shown on earlier days.

All features are built from one **stream**: every (impression, candidate) row of train, dev and test, in time order (`build_stream`).

---

## 2. Features

| Feature | Definition | Uses labels? |
|---|---|---|
| `shown_1h`, `shown_6h`, `shown_24h` | showings of this article in the last 1 / 6 / 24 hours before the impression | no |
| `shown_all` | showings since Nov 9 before the impression | no |
| `prev_shows`, `prev_clicks` | showings and clicks on the **source day** | yes (earlier day only) |
| `prev_ctr` | smoothed click rate on the source day | yes |
| `cum_shows`, `cum_clicks` | showings and clicks on all labelled days up to the source day | yes |
| `cum_ctr` | smoothed click rate over those days | yes |

**Source day** = the day before the impression, capped at Nov 15, the last labelled day: `source day = min(day − 1, 6)`, where day 0 = Nov 9. Rows of Nov 9 have no earlier day, so their click features are missing (NaN).

**Smoothing:** click rate = (clicks + m · p) / (showings + m), with m = 10 and p the overall click rate of that day. An article shown 3 times with 1 click is therefore not rated as a 33% article.

**Implementation:**
- Showings are counted by sorting the stream by (article, time) and binary-searching each window, which takes minutes for about 100 million rows.
- Click tables are article × day matrices built with `bincount` from labelled rows only.

---

## 3. Each feature alone on dev

Each feature was used directly as the score (higher = ranked first), with no model:

| Feature alone | AUC | MRR | nDCG@5 | nDCG@10 |
|---|---|---|---|---|
| `shown_1h` | 0.515 | 0.222 | 0.235 | 0.302 |
| `shown_6h` | 0.513 | 0.231 | 0.250 | 0.306 |
| `shown_24h` | 0.514 | 0.230 | 0.244 | 0.304 |
| `shown_all` | 0.515 | 0.227 | 0.238 | 0.300 |
| `prev_ctr` | 0.616 | 0.281 | 0.308 | 0.366 |
| **`cum_ctr`** | **0.629** | **0.287** | **0.314** | **0.373** |
| random (reference) | 0.500 | 0.219 | 0.224 | 0.287 |

**What it shows:**
- **How often an article was shown says almost nothing** (AUC about 0.51). Articles are shown a lot whether or not people click them.
- **The click rate on earlier days is strong:** `cum_ctr` alone reaches AUC 0.629, close to the LightGBM content model with 5 features (0.649). On news, "what other people clicked" is a hard baseline, just as popularity was for new users in the movie study.
- `cum_ctr` will be the **popularity baseline** of the news study (doc 05, N1).
