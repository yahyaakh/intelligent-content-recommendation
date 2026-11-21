# 07 — Do candidates have click history, and does article age matter?

**Status:** done 2026-11-19. This is step 1 of the classical-model work (doc 05, N1): a read-only analysis that tells which classical model families can work on news.
- **Code:** `news/coverage_and_age.py`.
- **Outputs:**
  - tables in `output/news/analysis/` (`coverage_*.csv`, `earlier_clicks_*.csv`, `age_*.csv`);
  - new features `output/news/features/age_hours_<split>.npy` and `older_than_log_<split>.npy`;
  - validation runs `single_age` and `single_cum_ctr` in `output/news/results/val/`.

---

## 1. Questions

- **A.** How many shown articles (candidate slots) already have click data that a collaborative model could use?
- **B.** Does the age of an article change how often it is clicked, and is age a useful ranking signal?

Both questions are answered for **val** (Nov 14) and **dev** (Nov 15). Only val is used for model results. Dev is the final test, so for dev only data descriptions are shown, no model scores.

---

## 2. Definitions

**Coverage levels.** Each slot gets the first level that applies:

| Level | Meaning | Source |
|---|---|---|
| `in_history` | the article is in someone's MIND history: clicked before the log | train histories (for dev: train + dev) |
| `clicked_earlier` | clicked in an impression on an earlier labelled day | `cum_clicks > 0` (doc 03) |
| `shown_never_clicked` | shown on an earlier labelled day, never clicked | `cum_shows > 0`, `cum_clicks = 0` |
| `never_seen` | no information from earlier days at all | the rest |

"Earlier" means earlier **days**, as in the leakage rule (doc 02 §4). An article first shown earlier on the same day still counts as `never_seen`.

**Article age** = the impression time minus the article's first showing in any impression of train, dev or test. A showing needs no label, and the first showing is never later than the slot itself, so there is no leakage.
- Articles that are in a **train** history were published before the log. Their real age is unknown, so they form their own bucket, `older_than_log`. Train histories never change during Nov 9–14 (doc 06 §4), so they hold only pre-log clicks.
- Ages near the start of the log (Nov 9) are lower bounds. This does not affect val or dev, which are 5–6 days later.

**Counts:**
- Article ids go up to 130,380.
- 54,581 articles are shown at least once in train, dev or test.
- 79,546 distinct articles appear in train histories; 79,937 in train + dev histories.

---

## 3. Results A: coverage

**Share of candidate slots per level** (each column sums to 100%):

| Level | Val: all slots | Val: clicked slots | Dev: all slots | Dev: clicked slots |
|---|---|---|---|---|
| in history | 9.1% | 5.4% | 7.7% | 5.1% |
| clicked on an earlier day | 56.6% | 60.8% | 62.9% | 67.8% |
| shown earlier, never clicked | 28.6% | 26.5% | 24.0% | 18.4% |
| never seen | 5.7% | 7.3% | 5.4% | 8.7% |

**By history group,** the shares hardly change. For example, `clicked_earlier` on val is 55.6% / 54.9% / 55.5% / 57.1% for groups 0 / 1–3 / 4–10 / >10. So users with short histories are shown the same kind of articles as everyone else. The full table is in `output/news/analysis/coverage_*.csv`.

**Earlier clicks of the candidate** (`cum_clicks`, all earlier labelled days):

| Earlier clicks | Val: all slots | Val: clicked slots | Dev: all slots | Dev: clicked slots |
|---|---|---|---|---|
| 0 | 36.5% | 34.9% | 31.0% | 29.2% |
| 1–4 | 23.1% | 30.5% | 22.7% | 25.1% |
| 5–19 | 6.5% | 6.5% | 6.1% | 6.6% |
| 20+ | 33.8% | 28.1% | 40.2% | 39.1% |

**What this shows:**
1. **Most candidates do have click data.** About two thirds of slots, and two thirds of clicked slots, are articles that someone clicked before: 66% / 66% on val and 71% / 73% on dev, counting `in_history` and `clicked_earlier` together. About a third already have 20 or more earlier clicks.
2. **So missing data does not explain co-visitation's failure** (AUC 0.504, doc 04). Doc 04 guessed that it did; this result rules that out. The likelier cause is in how the model is built:
   - **Top-50 truncation:** an old history article's 50 strongest neighbours are probably other old articles, not today's candidates;
   - **Old history vs new candidates:** the user's history is from before the log, while the candidates are from the log.

   This needs a direct check (§5) before item-CF, EASE or user-kNN are dropped.
3. **New articles get more than their share of clicks.** `never_seen` articles are 5.4% of dev slots but **8.7% of dev clicks**, the same numbers as doc 01. On val: 5.7% and 7.3%. Only content (text, LLM labels, content-based semantic IDs) can score these articles.
4. **Articles from before the log are rarely clicked.** `in_history` articles are 9.1% of val slots but 5.4% of clicks.

---

## 4. Results B: article age

| Age at the impression | Val: share of slots | Val: share of clicks | Val: click rate | Dev: share of slots | Dev: share of clicks | Dev: click rate |
|---|---|---|---|---|---|---|
| 0–1 h | 0.3% | 0.3% | 4.1% | 0.2% | 0.2% | 5.3% |
| 1–6 h | 2.6% | 3.0% | 4.3% | 1.4% | 1.7% | 4.8% |
| 6–24 h | 60.3% | 68.7% | 4.4% | 64.7% | 71.3% | 4.5% |
| 1–2 days | 19.5% | 17.6% | 3.5% | 16.4% | 13.5% | 3.4% |
| more than 2 days | 8.1% | 5.0% | 2.4% | 9.6% | 8.3% | 3.5% |
| older than the log | 9.1% | 5.4% | 2.3% | 7.7% | 5.1% | 2.7% |

- Median age of shown slots: 20.7 h on val and 19.8 h on dev. Of clicked slots: 19.5 h and 18.6 h.
- **Most shown articles are less than a day old** (63% of val slots), and they get most of the clicks (72% on val).
- **The click rate falls with age:** about 4.4% under a day, 3.5% at 1–2 days, about 2.4% after two days (val). Fresh articles are clicked almost twice as often as old ones.

**Age as a ranking signal on its own (validation):**

| Score | Group 0 | 1–3 | 4–10 | >10 | All: AUC | MRR | nDCG@5 | nDCG@10 |
|---|---|---|---|---|---|---|---|---|
| −age (newest first) | 0.533 | 0.533 | 0.531 | 0.544 | 0.540 | 0.214 | 0.218 | 0.278 |
| `cum_ctr` (popularity) | 0.605 | 0.612 | 0.613 | 0.606 | **0.608** | 0.257 | 0.269 | 0.331 |

Age alone is weak (AUC 0.540): within one impression, most candidates are of a similar age (6–24 h). It is still a cheap, leak-free feature that the click rate does not contain, so it goes into the learned ranker (step 3 of the plan).

**Popularity on validation:** `cum_ctr` gives **AUC 0.608** on val, against 0.629 on dev (doc 03). The two days differ, which is one more reason to compare models only within the same split. This is now the popularity reference on validation.

---

## 5. What this means for the classical models

| Model family | Verdict | Reason |
|---|---|---|
| Popularity (`cum_ctr`) | keep: baseline | val AUC 0.608, flat across history groups |
| Freshness (age, older-than-log flag) | add as features | the click rate falls from 4.4% to 2.4% with age; weak alone |
| Item-CF / co-visitation / EASE | **dropped** after doc 08 | the data exists, but co-occurrence measures read volume, not click rate (AUC 0.509 without the top-50 limit) |
| Audience similarity (who clicked the candidate before) | promising | 63% of val slots and 69% of dev slots have at least one earlier click |
| Content (MiniLM, NRMS, LLM labels) | needed in any case | it is the only signal for the 7–9% of clicks on never-seen articles |

**Next steps** (step 1 done in doc 08: the top-50 limit, plus a score that measures read volume, not click rate):
1. **Diagnose co-visitation** on val. For each slot, check whether the candidate is among the neighbours of any history article (with top 50, and with no limit), and get the AUC on just the slots where it is.
2. **Hybrid E** (doc 05, N2) on val, with the popularity reference AUC 0.608.
