# 32 — Semantic IDs for LLM prompts and for LTR candidates

**Status:** complete (2026-10-29). Two independent experiments.
**Related:**
- doc 30 (EASE semantic IDs);
- docs 17–18 (LLM reranking, prompts p1–p8);
- doc 16 §3.3 and doc 24 option C (wider LTR candidate lists).

---

## Part B. Taste-group names in the LLM reranking prompt

**Script:** `llm_sid_prompt_experiment.py`.
**Pre-registration:** `output/experiments/llm_sid_prompt/preregistration.md` (written before the run; one fact corrected before the full run).
**Outputs:** `output/experiments/llm_sid_prompt/`.

### B.1 Question

Can a short prompt that adds one semantic-ID "taste group" per film give the LLM useful information at fewer tokens than the best prompt (p2)?

### B.2 Set-up

- **Model and users:** deepseek-v4.1-flash (ollama_cloud, temperature 0); the 203 validation users; N = 10; Hybrid E's top 30; the LLM's own order is scored.
- **Groups:** the 32 first-level groups of the S4 EASE IDs (all warm users, k-means seed 42). These are films liked by the same people. 1,527 films have a group.
- **Group names:** written once by the same LLM from each group's 15 most popular films, e.g.:
  - "Anime Film Fans"
  - "Unsettling Cult Horror Thrillers"
  - "Acclaimed 90s Drama Thrillers"
  - "Classic Guy Movies"

  All 32 are in `group_names.json`.
- **Prompts:**
  - p1: title, year and genres;
  - **p9:** p1 + "taste group: <name>" per film;
  - p2: + director, cast and a seed profile (cached).
- **Candidate orders:** 3 per user (shuffles 0/1/2), averaged per user.

### B.3 Results (mean over 3 shuffles, 203 users)

| Prompt | Input tokens | NDCG@10 | Precision@10 | Recall@10 |
|---|---|---|---|---|
| p1 (titles + genres) | 757 | 0.600 | 0.583 | 0.056 |
| **p9 (p1 + taste group)** | 1,167 | 0.601 | 0.593 | 0.057 |
| p2 (+ director, cast, profile) | 1,612 | **0.619** | **0.602** | **0.058** |
| *Hybrid E (no LLM), for reference* | 0 | *0.629* | | |

| Hypothesis | Difference in NDCG | 95% CI | Users better / worse | p | Result |
|---|---|---|---|---|---|
| H1: p9 vs p1 | +0.001 | [−0.020, +0.021] | 105 / 95 | 0.89 | not shown |
| H2: p9 vs p2 (margin −0.005) | −0.018 | [−0.039, +0.003] | 91 / 106 | 0.09 | not shown |

### B.4 Findings

1. **The taste-group names did not help the LLM.** p9 equals p1 in NDCG (+0.001). They cost 54% more tokens than p1.
2. **p9 does not match p2:** it is 0.018 lower, which is outside the 0.005 margin. p2's extra tokens (director, cast, profile) are worth more than a group name.
3. **Precision moves slightly more than NDCG** (p9 vs p1: +0.010 vs +0.001). The names may help the LLM pick which films to include, but not how to order them. This is descriptive only: not tested, and within the shuffle noise.
4. **Every LLM order is still below Hybrid E (0.629)**, as in docs 17–18.
5. **Likely reason:** the LLM already knows these famous films from their titles. A coarse group of about 48 films adds little to what "The Godfather (1972)" already tells it.

**Safe statement:** "Adding a semantic-ID taste-group name to each film in the reranking prompt did not change the LLM's accuracy (+0.001 NDCG@10 over titles and genres, p = 0.89) and did not reach the richer prompt (−0.018)."

---

## Part A. Do semantic-ID picks raise LTR v2's candidate ceiling?

**Script:** `ltr_candidate_ceiling.py`. Descriptive, with no model fits.
**Outputs:** `output/experiments/ltr_candidate_ceiling/`.

### A.1 Set-up

- **Users:** the 9,084 cross-fitted simulated users.
- **Lists:** C30 (Hybrid E's top 30) plus the best 20 films outside it according to:
  - content (the doc 16 list);
  - EASE IDs (doc 30, k-means seed 42);
  - EASE itself.

  Everything is built from the other half's ratings.
- **Oracle:** the NDCG@10 a perfect reranker would reach within the list.
- **Rule written before running:** the full LTR test runs only if the EASE-ID picks add clearly more relevant films than the content picks, which gave no validation gain in doc 16.

### A.2 Results (10 seeds; 3 seeds in the CSV, same pattern)

| List | Oracle NDCG@10 | Liked films in list | Liked films added by the 20 picks |
|---|---|---|---|
| C30 | 0.961 | 16.3 | – |
| + 20 by content (doc 16) | 0.982 | 21.5 | 5.1 |
| **+ 20 by EASE IDs** | 0.984 | 23.6 | **7.2** |
| + 20 by EASE | **0.990** | **26.0** | **9.6** |

### A.3 Findings

1. **The EASE-ID picks add 41% more liked films than the content picks** (7.2 vs 5.1). By count, this meets the pre-run rule.
2. **But the ceiling barely moves:** 0.982 → 0.984. Every list is already at 0.96 or higher, while LTR v2 reaches 0.67. The ceiling is not what limits LTR v2. Ordering the films it already has is.
3. **EASE itself adds even more liked films (9.6) than its IDs (7.2).** As in docs 27–31, the full model beats its compressed IDs. If a wider list is tested again, EASE is the stronger choice. That is close to doc 24 option C (user-kNN picks), which was not adopted.

**Decision for the user:** the rule allows a full LTR test with EASE-ID picks. Given findings 2 and 3, the expected gain is small, and the doc 16 and doc 24 tests both failed on validation users.
