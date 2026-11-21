# 05 — Plan: the news study (MIND-large), mirroring the movie study

**Status:** plan, written 2026-11-19 (after the TDK, 2nd place). Nothing run yet.
**Goal:** repeat the movie study's main questions on a second, very different dataset, so the BSc thesis evaluates every idea on **two datasets** (answers review 1: "one dataset, one protocol, small catalog"; movie doc 38).
**Main question, unchanged:** *Where can a large language model help a recommender system with a new user?* Rank, describe, understand, plus semantic IDs, the learned ranker and cost.

---

## 1. What already exists for news

| Item | Where | State |
|---|---|---|
| MIND-large raw data (train, dev, test) | `data/Mind_Large_Data/` (≈4.6 GB) | downloaded (`scripts/download_mind.py`) |
| Evaluation module: official AUC, MRR, nDCG@5, nDCG@10, Codabench submission | `news/mind_eval.py` | done, self-tested against the official script |
| Popularity features (showings 1h/6h/24h/all, earlier-day clicks and click rates) | `news/popularity_features.py`, `output/news/features/` | done |
| MiniLM article embeddings (title + abstract + entities) | `output/minilm_title_abstract_entities_mine/` | done |
| Co-visitation (item-item) | `news/co_visitation.py` | done, weak |
| LightGBM lambdarank, content + popularity features | `news/SemanticContentModel.py` | done, best so far |
| Old small pipeline (5,000 articles, TF-IDF cold-start handler) | `news/prepare_mind_data.py`, `cold_start_handler.py` | superseded; keep only for the history chapter |

**Dev results so far** (`output/news/results/dev_summary.csv`, 376,471 impressions):

| Model | AUC | MRR | nDCG@5 | nDCG@10 |
|---|---|---|---|---|
| Random | 0.500 | 0.219 | 0.224 | 0.287 |
| Co-visitation, top 50 | 0.504 | 0.224 | 0.230 | 0.293 |
| Single feature: cumulative click rate | 0.629 | 0.287 | 0.314 | 0.373 |
| MLP, 3 category features | 0.617 | 0.297 | 0.324 | 0.384 |
| LightGBM, 5 MiniLM/category features | 0.649 | 0.318 | 0.349 | 0.408 |
| **LightGBM, 5 MiniLM + 10 popularity features** | **0.682** | **0.335** | **0.372** | **0.431** |

All models above were trained on train impressions before 2019-11-14. Published MIND-large numbers for NRMS / LSTUR / NAML (official test set) still have to be looked up in the papers before any comparison.

---

## 2. Facts that shape the design

Computed 2026-11-19 from `MINDlarge_{train,dev}/behaviors.tsv` and `news.tsv`:

| Fact | Train (Nov 9–14) | Dev (Nov 15) |
|---|---|---|
| Impressions | 2,232,748 | 376,471 |
| Users | 711,222 | 255,990 |
| Candidates per impression | mean 37.4, median 25, p90 92, max 300 | mean 37.4, median 23, p90 92, max 299 |
| Clicks per impression | mean 1.52; exactly 1 click 72% | mean 1.53; exactly 1 click 71% |
| History empty / 1–3 / 4–10 / >10 clicks | 2.1% / 7.1% / 23.0% / 67.8% | 3.0% / 7.2% / 22.6% / 67.2% |
| Median history length | 19 | 19 |

- Articles: 104,151 in train ∪ dev; 18 categories, 285 subcategories.
- **New articles:** 5.4% of dev candidate slots and **8.7% of dev clicks** are articles never seen in train.
- 87.1% of dev impressions come from users who also appear in train.

### Movies vs news

| | Movies (TDK) | News (MIND) |
|---|---|---|
| Feedback | explicit ratings, "liked" = at or above the user's average | implicit clicks in an impression |
| What is ranked | the whole catalog (1,547 candidates) | the impression's own candidates (mean 37) |
| New user | simulated: first 3 or 10 liked films | **real**: 10% of dev impressions have 0–3 history clicks; can also be simulated by truncating histories |
| New items | none (all films had ratings) | yes: 8.7% of clicks go to articles unseen in train |
| Time | ratings years apart | everything happens in 6 days; popularity changes by the hour |
| Metrics | NDCG@10 (+P/R) | official AUC, MRR, nDCG@5, nDCG@10 per impression |
| Ground truth for LLM labels | Tag Genome, US age ratings | MIND category / subcategory, Wikidata entities in `news.tsv` |
| Catalog size | 1,957 | 104,151 |

**What this means:**
1. Impressions are natural candidate lists, so the LLM reranks the **impression itself**, not another model's top 30. Their length varies (2–300), which tests the reviewer's question on **shorter lists** for free (movie doc 38, F2).
2. The new-user group is real here, not only simulated: report every result for **history 0, 1–3, 4–10, >10**.
3. Popularity is time-dependent: the "popularity" baseline is the click rate so far (`cum_ctr`), not an all-time count.
4. EASE on 104k articles needs a 104k × 104k matrix (≈ 43 GB), so it must be restricted to recent articles with enough clicks (for example the 20k most clicked in the last 2 days, ≈ 1.6 GB).

---

## 3. Protocol (to fix before any LLM run)

- **Validation:** the last train day (Nov 14), models trained on Nov 9–13. All choices (α, features, prompts, list length) are made here only.
- **Final test:** MIND **dev** (Nov 15, labels public), run once with a pre-registration as in movie doc 36. Optional: one Codabench submission on the hidden test set for the best classical model.
- **LLM samples:** LLM runs use a fixed random sample of impressions, stratified by history group: about 2,000 validation and 5,000–10,000 test impressions, the same sample for every LLM method.
- **Statistics:** paired Wilcoxon per impression, Bonferroni, bootstrap intervals for equivalence; the same rules as the movie study.

---

## 4. Experiment map: each movie experiment and its news version

| # | Movie experiment (movie docs) | News version | Notes |
|---|---|---|---|
| N1 | Classical baselines (06, 20, 26) | Popularity (`cum_ctr`, recent click counts), content (MiniLM similarity to history), item-CF / co-visitation, user-kNN, EASE on recent articles, **NRMS** as the standard neural news model | NRMS is in the `recommenders` library (installed in `recomm_env`; check TensorFlow + GPU) |
| N2 | Hybrid E: personal score + popularity, α by grid search (06) | content (or item-CF) + click rate, α per history group | expect a smaller α for empty histories, as with 3 seeds |
| N3 | Learned ranker (15, 16, 24) | extend LightGBM pop15 with co-visitation, user-kNN, EASE and LLM features | the question again: do LLM features add anything? |
| N4 | LLM enrichment (01, 04, 14) | LLM labels each article: topic, subtopic, entities, sentiment/tone, audience, sensitive content | **verification:** predict MIND's subcategory blind (accuracy, 285 classes) and compare entities with MIND's Wikidata entities, like the Tag Genome check |
| N5 | LLM reranking (13, 17) | LLM reorders the impression's candidates from the user's clicked titles. Prompts: titles only / titles + abstracts + category / + click statistics (like p5) | also by list length (≤10, 11–30, >30) and history group |
| N6 | Free generation (13) | **skip**: the LLM cannot know articles published after its training | say so in the thesis, as a property of news |
| N7 | Semantic IDs (27–34) | content IDs from MiniLM vectors; click-based IDs for articles with clicks | **strongest case for news:** new articles get an ID from content at once. Main test: new articles (8.7% of clicks), like the movie new-item and rare-film tests |
| N8 | Constraints + typed requests (21, 25, 37) | "no politics", "only sports", "nothing violent or sad", "short reads"; LLM labels topic and sensitive content; 50 news requests written before testing | no age ratings exist, so a small hand-labelled set (e.g. 300 articles, labelled before testing) is the ground truth for sensitive content |
| N9 | Cost and latency (ch. 10) | the same table for news | news needs answers in milliseconds, so the same "once per article, not per list" conclusion should hold more strongly |
| N10 | Pre-registered final test (36) | one run on dev with a pre-registration | |
| N11 | Future work from the TDK (movie doc 38) | F2 (shorter lists) falls out of N5; F3 (agentic, tool access to click data) can be tried on news first | optional |

---

## 5. Cost and time estimates (DeepSeek V4.1 Flash, from the movie runs)

| Job | Movie reference | News estimate |
|---|---|---|
| Label all articles | 1,957 films: $1.14, 0.85 s each | 104,151 articles: about **$60** and about 25 h sequential. Start with only the articles in the validation and test samples (a few thousand). |
| Rerank one list | about $0.0006 and 1 s per user | 2,000 validation impressions ≈ $1.3; 10,000 test impressions ≈ $6.5 and ≈ 3 h sequential |
| Parse a typed request | 0.67 s, under $0.001 | the same |

The costs are scaled from the movie runs. Abstracts make news prompts shorter than film-detail prompts, but list lengths vary, so measure on a 20-impression pilot first.

---

## 6. Order of work

1. **Protocol and data** (N10 set-up): the train/validation split by day, history groups, fixed LLM samples, and a pre-registration skeleton. Reuse `mind_eval.py`.
2. **Classical baselines** (N1, N2): popularity, content, item-CF, user-kNN, EASE-recent, hybrid with α; then NRMS. **Checkpoint:** is popularity again a hard bar for empty histories?
3. **Learned ranker** (N3): add the new classical scores as features.
4. **LLM enrichment and its verification** (N4) on the sampled articles.
5. **LLM reranking** (N5), with list length and history group.
6. **Semantic IDs** (N7), new-article test first.
7. **Constraints and typed requests** (N8).
8. **Cost table** (N9), **pre-registered final test** (N10), thesis chapter.

---

## 7. Open decisions

1. **Final test set:** MIND dev (labels known, the plan above) or the hidden Codabench test (official, but every run needs an upload)? Recommendation: dev, plus one Codabench check.
2. **Scale of LLM runs:** stratified samples (recommended) or all dev impressions (≈ 376k calls, not affordable).
3. **NRMS:** train it (the standard MIND baseline; reviewers will expect it) or cite published numbers only. Recommendation: train it, if TensorFlow runs on the GPU.
4. **Old small news pipeline** (`prepare_mind_data.py`, `cold_start_handler.py`): archive it, as was done for the old movie code.
