# 38 — TDK reviews, talk questions and future work

**Status:** written 2026-11-18, after the TDK talk.
**Purpose:** keep every reviewer criticism and every question from the talk, with the answer backed by results, and turn the open points into future work for the BSc thesis.
**Related:** doc 13 (LLM reranking), doc 14 (label quality), doc 16 (learning to rank v2), doc 21 and 37 (constraints), doc 33 (semantic-ID robustness), doc 36 (final test).

---

## 1. Written reviews

### 1.1 Review 1: 30 / 35

| Part | Score |
|---|---|
| A. Style | 3 / 5 |
| B. Figures and references | 4 / 4 |
| C. Topic | 8 / 8 |
| D. Novelty / soundness | 4 / 5 + 5 / 5 |
| E. Results | 6 / 8 |

**Positive:** excellent English, a strong introduction with five research questions, current references.

**Criticisms and answers**

| Criticism | Answer / what to change in the thesis |
|---|---|
| Tries to do too much; the main text is 43 pages (limit 30–35) and chapters are compressed. | Fewer research questions in the thesis, or group them under one main question; move detail to the appendix. |
| No new algorithm: the contribution is empirical. | Agreed and stated openly: the contribution is a controlled comparison under one protocol, with pre-registered final tests. |
| Conclusions hold for one dataset (MovieLens 25M + TMDB), one cold-start protocol and a small catalog (1,957 films). | Stated as a limitation. Future work F6 (second dataset, larger catalog). |
| Results (E) need scalability and robustness evidence to score 7–8. | Partly there: warm-user benchmarks (doc 23, doc 35), semantic-ID robustness (doc 33). Missing: a second dataset and a larger catalog (F6). |

### 1.2 Review 2: 29.5 / 35

| Part | Score |
|---|---|
| A. Style | 3 / 5 |
| B. Figures and references | 3 + 4 |
| C. Topic | 7 / 8 |
| D. Novelty / soundness | 4 / 5 + 5 / 5 |
| E. Results | 7 / 8 |

**Criticisms and answers**

| Criticism | Answer |
|---|---|
| Main text over 35 pages. | As review 1. |
| Too many research questions, somewhat scattered. | The talk used one main question ("Where can an LLM help a new user?") with three roles (rank, describe, understand). Use the same structure in the thesis. |
| Table 8.1 (semantic-ID lift) is unclear: the reviewer expected the random value to be about 1, but the brackets showed up to 5.37 for director. | The brackets were the **99th percentile** of 1,000 random shuffles, not the mean. The **random mean is about 1.00** for every attribute (the test is fair); the 99th percentile is the bar a real lift must clear for p < 0.01. Director's bar is higher because two random films share a director only 0.15% of the time and most level-3 groups hold one film, so one chance match moves the random lift a lot. Every real value is above all 1,000 shuffles (p = 1/1,001; 0.048 after Bonferroni over 48 tests). **Fix:** show the random mean and the 99th percentile as two columns, as on the talk's extra slide. |
| Why is Hybrid E the reference when EASE is best? | The LLMs and the learned ranker rerank the **same** Hybrid E top 30, so they are directly comparable; a perfect reorder of those 30 reaches NDCG@10 0.95, so there is plenty of room. The LLM was also tested on top of user-kNN, the second-best method: 0.631 vs user-kNN's own 0.642, the same pattern. Reranking EASE's top 30 directly has not been run (F1). |
| Typical α range? | α was chosen by grid search (17 values, finer near 0) on the 203 validation users only, then fixed for the test users. Typical values are 0.05 to 0.25: the more seed films a user gives, the more weight the personal score gets (Hybrid E: 0.12 with 3 seeds, 0.25 with 10). Structured preferences: about 0. |

---

## 2. Questions at the talk (2026-11-18)

### Q1. "Did you think of agentic ideas? This is basically a naive RAG."

**Answered at the talk:** a good idea, not tested; future work.

**Prepared answer:** Yes, the set-up is retrieve-then-rerank: a classical model retrieves 30 candidates and the LLM reads them in one call, like a one-step RAG. It was kept to one call on purpose, since one call already costs about 1 s per user; an agent would need several. The results hint at where an agent's gain would come from: rating statistics in the prompt (p5) halved the gap to Hybrid E (−0.014 → −0.007), but the learned ranker, which uses collaborative data directly, wins in 12 ms. **Future work F3, F4.**

### Q2. "Do your LLMs have access to the ratings, the dataset, the keywords?"

**Answered at the talk:** only the seed films and Hybrid E's top 30; giving the whole dataset and all ratings would cost too much money and time.

**Complete answer (doc 13):**
- **p2 (film details):** title, year, genres, director and cast of each candidate, plus a profile of the liked films.
- **p5 (details + ratings data):** the same, plus for each candidate the share of fans of the liked films who also liked it, and its popularity.
- No raw ratings, scores or labels in p1–p3 prompts (checked by `verify_llm_rerank.py`).
- Result: p5 halved the gap to Hybrid E (DeepSeek −0.014 → −0.007, test users) but did not close it.

### Q3. "Why rerank Hybrid E's top 30? Did you test the top 10? Would a shorter list give the LLM better results?"

**Answered at the talk:** a long list gives the LLM many options; probably not better with 10, but not sure, and worth testing.

**Evidence for 30:**
- Reranking only the top 10 cannot bring in new films; it can only reorder them.
- With 30 candidates a perfect reorder reaches **NDCG@10 0.95** (on average 16 of the 30 candidates are relevant), against about 0.63 for Hybrid E's own order (doc 13 §3).
- For the learned ranker, **50 candidates gave no gain** over 30 (0.6635 vs 0.6645; oracle 0.984 vs 0.953; doc 16).

**What supports the reviewer's idea:** the same 30 candidates in a different order change about **a quarter of the LLM's top 10** (doc 13 §6.3). The LLM struggles with long lists, so a shorter list may suit it better. **Not tested for the LLM: future work F2.**

### Q4. "How did you verify the dataset enrichment?"

**Answered at the talk:** against the Tag Genome (labels from real users); themes and moods are subjective.

**Complete answer (doc 14, report ch. 5 and appendix):**
- A label counts as confirmed if its Tag Genome relevance for that film is at least 0.5; checked on the **1,388 films** with genome scores.
- DeepSeek V4.1 Flash: **67.3%** of themes and **76.7%** of moods confirmed, against **6.4%** and **17.1%** for a random label. DeepSeek beats gemma4:e4b on every measure (p < 0.001), most of all on less-known films.
- The audience label was checked against official US age ratings: F1 **0.905** vs 0.696 for a genre + keyword rule (doc 21).
- The genome is itself estimated from user tags, so these numbers are a conservative check. Subjectivity is a limitation, not the main answer.

---

## 3. Future work for the BSc thesis

| # | Idea | Why (source) | Design sketch |
|---|---|---|---|
| F1 | LLM reranks **EASE's** top 30 | Review 2; open since the final test | Same protocol as doc 13 (validation first, pre-registered test). Compare with EASE's own order and the oracle@30. |
| F2 | **Shorter candidate lists** for the LLM (10, 20, 30) | Talk Q3 | Same users, prompts p2 and p5. Report NDCG@10, oracle per list size and shuffle stability (top-10 overlap between shuffles). Hypothesis: shorter lists are more stable but have a lower ceiling. |
| F3 | **Agentic LLM** with tools | Talk Q1 | The LLM may call tools: fans of the liked films who also liked X, film details, similar users' favourites. Compare with the one-call p5 prompt on NDCG@10, number of calls, time and cost per user. |
| F4 | **Richer RAG** context | Talk Q1 | Retrieve evidence per candidate (co-liked films, viewer tags, short review snippets) into one prompt. Tests whether better retrieved context closes the remaining gap without an agent. |
| F5 | **Real new users** | Talk conclusion; review 1 | Typed requests and seed films from real people instead of simulated users; check the 98.3% constraint satisfaction and 43/50 parsing on their own wording. |
| F6 | **Second dataset, larger catalog** | Review 1 (one dataset, small catalog) | Repeat the main comparison (popularity, Hybrid E, EASE, learned ranker, one LLM reranker) on another public dataset, or on more MovieLens films. |
| F7 | **Human check of the LLM labels** | Talk Q4 | A small panel rates whether themes and moods fit, giving agreement between people and LLM-vs-human agreement beyond the Tag Genome. |
| F8 | **Constraint parser fixes** | Doc 37 §6 | Treat several genres as "all of" when the request names one kind of film; no extra excluded moods when an intensity limit already covers the request; retry once when validation rejects a value. |
