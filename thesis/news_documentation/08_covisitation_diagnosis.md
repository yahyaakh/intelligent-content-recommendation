# 08 — Why item co-visitation fails on MIND

**Status:** done 2026-11-19.
- **Code:** `news/covisitation_diagnosis.py`.
- **Outputs:** `output/news/analysis/covisitation_*_val_sample.csv`.
- **Data:** a random sample of **10,000 validation impressions** (Nov 14, seed 42; 392,164 candidate slots). The reader data comes from Nov 9–13 only.

---

## 1. Question

Item co-visitation (doc 04, `news/co_visitation.py`) scored AUC 0.504 on dev, close to random. Doc 07 showed that missing data is not the cause: two thirds of candidates had been clicked before. So what goes wrong?

**The model, as in `co_visitation.py`:**
- **Readers of an article:** users who have it in their history or clicked it in an impression. Here: train histories plus clicks of Nov 9–13; 709,864 users and 15.9 million (user, article) pairs.
- **Similarity of two articles:** shared readers / √(readers of a × readers of b).
- **The old model** keeps the 50 strongest neighbours of each article. A candidate's score is the sum of its weights in the neighbour lists of the user's history articles.

**Scores compared on the same slots:**

| Score | Definition |
|---|---|
| `covis_top50` | the old model (top-50 neighbours) |
| `covis_full` | the same similarity, with every shared reader counted (no top-50 limit) |
| `shared_raw` | raw shared readers, no normalisation |
| `shared_cond` | shared readers / readers of the history article, i.e. P(candidate read \| history article read) |
| `cum_ctr` | popularity reference (doc 03) |

---

## 2. Results

### 2.1 Most candidates never reach a top-50 list

| | All slots | Clicked slots |
|---|---|---|
| Linked to the user's history (at least one shared reader) | 47.6% | 44.2% |
| In the top-50 list of at least one history article | **2.6%** | **4.2%** |

- A history article has a median of **3,508** neighbours, and the 50th neighbour's weight has a median of 0.044.
- For linked slots, the candidate's strongest link reaches a median of only **0.28 × the 50th neighbour's weight**. The top-50 lists are filled by other (mostly older) articles.

**So the old model gives 97% of candidates a score of 0.** The order then comes from the random tie-break. This alone explains the near-random dev result.

### 2.2 Without the limit, it is still almost random

AUC on the 10,000 sampled validation impressions:

| Score | Group 0 | 1–3 | 4–10 | >10 | All | nDCG@10 |
|---|---|---|---|---|---|---|
| `covis_top50` (old model) | 0.490 | 0.504 | 0.518 | 0.506 | 0.508 | 0.284 |
| `covis_full` (no limit) | 0.490 | 0.518 | 0.517 | 0.506 | **0.509** | 0.289 |
| `shared_raw` | 0.490 | 0.516 | 0.512 | 0.499 | 0.503 | 0.279 |
| `shared_cond` | 0.490 | 0.518 | 0.514 | 0.503 | 0.506 | 0.283 |
| `cum_ctr` (popularity) | 0.615 | 0.589 | 0.617 | 0.608 | **0.609** | 0.335 |

Group 0 has no history, so every co-visitation score is 0 and the result is the random tie-break. On this sample, `cum_ctr` gives 0.609, the same as on all of val (0.608, doc 07), so the sample is representative.

### 2.3 Why: the score measures how widely an article is read, not whether this user will click it

Spearman correlation over the 392,164 slots:

| | `covis_full` | `shared_raw` | `shared_cond` | candidate's readers | `cum_ctr` |
|---|---|---|---|---|---|
| `covis_full` | 1.00 | 0.99 | 0.99 | **0.86** | 0.11 |
| candidate's readers | 0.86 | 0.89 | 0.88 | 1.00 | 0.24 |

- **All three normalisations rank candidates almost identically** (ρ = 0.99), and that ranking is mostly the candidate's **number of readers** (ρ = 0.86–0.89).
- **How many people read an article does not predict clicks on MIND.** The showing counts behaved the same way (AUC about 0.51, doc 03). What predicts clicks is the click **rate** (`cum_ctr`), and co-visitation hardly correlates with it (ρ = 0.11).
- **A likely reason (not measured here):** an article collects readers over time, and older articles are clicked less (doc 07: click rate 4.4% under a day, about 2.4% after two days). So a high reader count partly means "older".

### 2.4 There is a small real signal where the history is well linked

`covis_full` AUC by the share of an impression's candidates that are linked to the history:

| Linked candidates | Impressions | AUC |
|---|---|---|
| ≤ 50% | 6,449 | 0.498 |
| 50–80% | 2,809 | 0.520 |
| 80–95% | 588 | 0.551 |
| > 95% | 154 | 0.560 |

Where almost every candidate shares readers with the history, co-visitation does carry some collaborative signal (AUC 0.55–0.56). But that applies to only 7% of impressions, and even there it stays below popularity.

### 2.5 The top-ranked candidate

| Score | Top-1 is a pre-log article | Top-1 clicked |
|---|---|---|
| `covis_full` | 33.0% | 10.6% |
| `cum_ctr` | 38.4% | 14.2% |

Pre-log articles are 9.1% of slots and 5.6% of clicked slots. Both scores put such an article first about a third of the time, so the age effect is not specific to co-visitation. When `cum_ctr` ranks an article first, it is clicked more often (14.2% vs 10.6%).

---

## 3. Conclusions

1. **The old model was random because of the top-50 limit:** only 2.6% of candidates appear in any history article's top-50 list.
2. **Removing the limit does not fix it** (AUC 0.509). Item-to-item co-occurrence on MIND mostly measures how widely an article is read, and that volume does not predict clicks. No normalisation of the co-occurrence counts changes this.
3. **Consequences for the classical models:**
   - **Item co-visitation is dropped** as a stand-alone baseline. It is reported in the thesis as a negative result with this explanation, which is itself a difference from the movie study, where item-CF was strong.
   - **EASE is not built for news.** It learns from the same co-occurrence counts and would face the same volume problem. This is a reasoned decision, not a measured one; say so in the thesis.
   - **What should work instead** is a collaborative signal built on the click *rate*, not the read count. Among users who share history articles with this user and **were shown** the candidate on earlier days, what share clicked it, compared with everyone? This is the "audience" idea of the plan, now with a clear target: the rate, corrected for exposure.
   - `covis_full` may still be tried as one learned-ranker feature (the small signal in §2.4), but with low priority.

---

## 4. Next

1. Hybrid E on validation (content + `cum_ctr`, α per history group).
2. A rate-based audience feature: personalised click rate from earlier days.
