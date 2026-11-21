# 01 — The dataset: MIND-large

**Status:** done. The numbers were computed on 2026-11-19 from the raw files with a read-only script (history lengths, candidates per impression, new articles).
**Source:** Microsoft News Dataset (MIND), large version: Wu et al., *MIND: A Large-scale Dataset for News Recommendation*, ACL 2020.

---

## 1. Files

`data/Mind_Large_Data/MINDlarge_{train,dev,test}/MINDlarge_{split}/`, downloaded with `scripts/download_mind.py`. CSV copies are in `data/Mind_Large_Data/mind_large_dataset_csv/`.

| File | Columns | Use |
|---|---|---|
| `news.tsv` | news id, category, subcategory, title, abstract, url, title entities, abstract entities | article text and labels |
| `behaviors.tsv` | impression id, user id, time, history, impressions | one row per page view |
| `entity_embedding.vec`, `relation_embedding.vec` | Wikidata entity and relation vectors (100 dimensions) | not used yet |

**One impression** is one time a user opened the news page:
- **history:** the articles the user clicked *before* this impression (space-separated ids, oldest first);
- **impressions:** the articles shown on this page, each with `-1` (clicked) or `-0` (not clicked), e.g. `N55689-1 N35729-0`.

The task: for each impression, rank its shown articles so that the clicked ones come first.

The entity columns are JSON lists with each entity's label, its Wikidata id, its type and where it occurs in the text.

---

## 2. Size

| | Train | Dev | Test |
|---|---|---|---|
| Days | Nov 9–14, 2019 | Nov 15, 2019 | Nov 16–22, 2019 |
| Impressions | 2,232,748 | 376,471 | (labels hidden) |
| Users | 711,222 | 255,990 | |
| Clicks per impression | mean 1.52; exactly one click 72.3% | mean 1.53; exactly one click 71.3% | |
| Candidates per impression | mean 37.4, median 25, p90 92, max 300 | mean 37.4, median 23, p90 92, max 299 | |

- **Articles:** 101,527 in train's `news.tsv`, 72,023 in dev's, 104,151 in the two together.
- **Categories:** 18 categories, 285 subcategories.
- **Test labels are hidden.** Test results can only be obtained by uploading a prediction file to the official leaderboard (Codabench). `mind_eval.write_submission()` writes that file.

---

## 3. History length: real new users

Share of impressions by the number of clicks in the history:

| History clicks | Train | Dev |
|---|---|---|
| 0 (empty) | 2.1% | 3.0% |
| 1–3 | 7.1% | 7.2% |
| 4–10 | 23.0% | 22.6% |
| more than 10 | 67.8% | 67.2% |
| median | 19 | 19 |

About 10% of dev impressions (≈ 38,000) come from users with 0–3 clicks. These are **real** new users, not simulated ones as in the movie study. Results will be reported for all four groups.

87.1% of dev impressions come from users who also appear in train's impressions.

---

## 4. New articles

News goes out of date in hours, so many shown articles are new:
- **5.4%** of dev candidate slots are articles never seen in train (in any history or impression);
- **8.7%** of dev **clicks** go to such articles, so new articles are clicked more often than average;
- 2,624 dev articles are missing from train's `news.tsv`. They must be embedded separately (doc 04).

A model based only on clicks has no information about these articles. A content model (text) or a popularity count from the same day still has information.

---

## 5. How news differs from the movie data

| | Movies (MovieLens 25M + TMDB) | News (MIND-large) |
|---|---|---|
| Feedback | explicit ratings; "liked" = at or above the user's own average | implicit: clicked or not, inside one impression |
| What is ranked | the whole catalog (1,547 candidates) | only the impression's own candidates (median 23–25) |
| New user | simulated: first 3 or 10 liked films | real: 10% of dev impressions have 0–3 clicks |
| New items | none | 8.7% of dev clicks |
| Time | ratings over many years | 6 + 1 + 7 days; popularity changes by the hour |
| Labels to check LLM output against | Tag Genome, US age ratings | MIND category and subcategory, Wikidata entities |
| Catalog size | 1,957 films | 104,151 articles |
| Metrics | NDCG@10 over the catalog | AUC, MRR, nDCG@5, nDCG@10 per impression |
