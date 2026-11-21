# Experiment: Typed Requests for New Users (Preference Interpretation)

**Status:** finished (run date: 2026-10-21, on the corrected half-star ratings)
**Script:** `typed_request_experiment.py`
**Results folder:** `output/experiments/typed_requests/`
**Related documents:** `06_experiment_cold_start.md`, `07_experiment_outside_knowledge.md`, `08_experiment_larger_vocabulary.md`

---

## 1. Questions

The TDK abstract asks whether new users who express their preferences in natural language receive more relevant recommendations from LLM-assisted methods than from content-based and popularity-based alternatives. This experiment simulates such users and asks:

1. **Can a typed request replace the seed movies** a new user names?
2. **Does a typed request add anything** when the seed movies are known as well?
3. **Does LLM enrichment help match typed requests?** Users describe what they want with words such as "bleak" or "atmospheric"; LLM labels (v2, v4) and knowledge-rich profiles (gemma4:31b P2, experiment 07) describe films in the same kind of language, so this is where enrichment was expected to help most.
4. **Does specific wording help** compared with general wording?

---

## 2. Method

### 2.1 Users and ground truth

The users, seeds, answer key and validation/test split are identical to experiment 06: 1,017 simulated new users (cold users excluded from training), each with 10 seed movies (their first liked movies in time order) and about 128 relevant movies (all their other liked movies in the candidate pool); 203 validation users and 814 test users. Seeds are set aside at every N, so the candidates and answer key do not change with N. Requests were written for N = 3 and N = 10 seed movies.

### 2.2 Simulated typed requests

For every user and N, gemma4:e4b (temperature 0, seed 42, thinking off, JSON schema `{"request": "..."}`) wrote the request the user could type into the system, based on their N seed movies. For each seed, the model saw the title, year, genres and the first 200 characters of the overview, i.e. what a user would know about films they liked. The request had to be 1–2 sentences (at most 40 words), in the first person, describing the kinds of films, themes, moods and styles the person likes, **without naming any film, actor, director, character or franchise**.

Two request styles differ in one instruction:

| Style | Additional instruction | Simulates |
|---|---|---|
| **r1 (general)** | none | a quick, casual request |
| **r2 (specific)** | "Be specific: name the particular themes, moods, settings and kinds of stories these films share. Avoid vague words such as 'intense', 'high stakes' or 'gripping' unless they are the main point." | a user who describes their taste carefully |

r2 was added after a pilot on 20 validation users showed that r1 requests often used the same generic phrases ("intense", "high stakes", "keep me on the edge of my seat") regardless of the seeds.

**Examples (pilot, validation users):**

| Seeds | r1 | r2 |
|---|---|---|
| *The Usual Suspects, The Godfather, The Godfather Part II* | "I really like intense stories about organized crime and morally complex characters…" | "I like stories about powerful, organized crime families and the moral compromises they make…" |
| *The Exorcist, The Birds, Jaws* | "I really like intense, scary movies with a strong sense of dread…" | "I like stories with a pervasive sense of dread, often set in isolated or ordinary locations. I'm drawn to films featuring supernatural threats or primal dangers that attack communities." |
| *The Iron Giant, E.T., Close Encounters* | "I really like heartwarming stories about extraordinary beings visiting our world…" | "…gentle, heartwarming encounters with extraordinary beings… often set in small-town or natural settings." |

| Run | Requests | Mean words | Seconds per request | Flagged as naming a seed |
|---|---|---|---|---|
| r1 | 2,034 | 26.4 | 1.17 | 6 (0.3%) |
| r2 | 2,034 | 27.5 | 1.27 | 13 (0.6%) |

The seed check flags a request if it contains a seed title of at least 5 characters; inspected cases were false positives caused by titles that are common words.

### 2.3 Matching and methods

Requests were embedded with embeddinggemma using its query prompt (`task: search result | query: `), and movies with the document vectors used in earlier experiments. Four movie representations were compared: **base text**, **+ v2 labels**, **+ v4 labels** (experiment 08) and **+ gemma4:31b P2 profiles** (experiment 07).

Per user, three signals were computed over the candidates and each rescaled to 0..1: request similarity (cosine of request and movie), seed similarity (mean similarity to the N seeds), and popularity (log number of warm users who liked the movie).

| Method | Score |
|---|---|
| Popularity | popularity |
| Seeds | seed similarity |
| Request | request similarity |
| Request + seeds | w × request + (1 − w) × seeds |
| Hybrid seeds | α × seeds + (1 − α) × popularity (experiment 06) |
| Hybrid request | α × request + (1 − α) × popularity |
| Hybrid request + seeds | α × (w × request + (1 − w) × seeds) + (1 − α) × popularity |

w ∈ {1.0, 0.9, …, 0.0} and α (the grid of experiment 06) were chosen on the 203 validation users, separately for every style, N, representation and method; all results are on the 814 test users. Wilcoxon signed-rank tests per user, Bonferroni-corrected over all 42 tests.

**Validity check:** hybrid seeds with the base text reproduces hybrid C of experiment 06 exactly (N = 3: 0.5778; N = 10: 0.6135), and popularity is identical (0.5598).

---

## 3. Results

### 3.1 Main comparison (base text, NDCG@10)

| Method | r1, N = 3 | r1, N = 10 | r2, N = 3 | r2, N = 10 |
|---|---|---|---|---|
| Popularity | 0.5598 | 0.5598 | 0.5598 | 0.5598 |
| Request | 0.1262 | 0.1531 | 0.1281 | 0.1439 |
| Seeds | 0.3798 | 0.4540 | 0.3798 | 0.4540 |
| Request + seeds | 0.3798 | 0.4540 | 0.3798 | 0.4540 |
| **Hybrid request** | 0.5747 | 0.5784 | 0.5770 | 0.5837 |
| **Hybrid seeds** | 0.5778 | **0.6135** | 0.5778 | **0.6135** |
| Hybrid request + seeds | **0.5809** | 0.6103 | **0.5829** | 0.6060 |

Precision@10 and Recall@10 follow the same order (e.g. r1, N = 3: request 0.127 / 0.011, hybrid request 0.562 / 0.051, hybrid seeds 0.566 / 0.051; popularity 0.548 / 0.049).

| Comparison (base text) | r1, N = 3 | r1, N = 10 | r2, N = 3 | r2, N = 10 |
|---|---|---|---|---|
| Hybrid request vs popularity | **+0.0149** (p = 0.002) | +0.0186 (p = 0.065) | **+0.0172** (p < 0.001) | **+0.0239** (p < 0.001) |
| Hybrid request vs hybrid seeds | −0.0031 (n.s.) | **−0.0351** (p < 0.001) | −0.0008 (n.s.) | **−0.0298** (p < 0.001) |
| Hybrid request + seeds vs hybrid seeds | +0.0031 (n.s.) | −0.0032 (n.s.) | +0.0051 (n.s.) | −0.0075 (n.s.) |

p-values Bonferroni-corrected; bold = significant.

**Chosen weights.** Without popularity, the validation users chose **w = 0** for request + seeds in every case (the request is ignored, so the method equals seeds). With popularity, the request received weight w = 0.2–0.7.

### 3.2 Movie representations for request matching

Request only (NDCG@10):

| Representation | r1, N = 3 | r1, N = 10 | r2, N = 3 | r2, N = 10 |
|---|---|---|---|---|
| Base text | 0.1262 | 0.1531 | 0.1281 | 0.1439 |
| + v2 labels | 0.1259 | 0.1500 | 0.1270 | 0.1398 |
| + v4 labels | 0.1304 | 0.1576 | 0.1298 | 0.1464 |
| **+ 31b P2 profiles** | **0.1343** | **0.1599** | **0.1318** | **0.1516** |

| Comparison | r1, N = 3 | r1, N = 10 | r2, N = 3 | r2, N = 10 |
|---|---|---|---|---|
| 31b P2 vs base | **+0.0081** (p = 0.008) | +0.0068 (p = 0.15) | +0.0037 (n.s.) | **+0.0077** (p = 0.007) |
| v4 vs base | +0.0042 (n.s.) | +0.0045 (n.s.) | +0.0017 (n.s.) | +0.0025 (n.s.) |
| v2 vs base | −0.0003 (n.s.) | −0.0031 (n.s.) | −0.0011 (n.s.) | −0.0041 (n.s.) |

In the hybrid request + seeds, representation differences are mostly not significant; the only significant case is v4 at r2, N = 10 (+0.0053, p = 0.029). Hybrid differences between representations at N = 3 depend strongly on the chosen α (as in experiment 06) and should not be interpreted as effects of the representation.

### 3.3 Request style

r2 vs r1 (Bonferroni-corrected): request only, base text: +0.0019 (N = 3) and −0.0092 (N = 10); hybrid request + seeds: +0.0020 and −0.0043; request only with v4 labels: −0.0007 and −0.0112. None is significant.

---

## 4. Findings

1. **A typed request alone is a weak recommender** (NDCG@10 0.13–0.16), far below the seed movies (0.38–0.45) and popularity (0.56).
2. **Combined with popularity, a typed request beats popularity** by +0.015 to +0.024 NDCG@10 (+2.7% to +4.3%), significant after correction in 3 of 4 settings.
3. **A typed request is worth about three seed movies, but not ten.** With popularity, the request performs as well as three named films (no significant difference) and clearly worse than ten (−0.030 to −0.035, significant).
4. **The request adds nothing when the seeds are known.** Without popularity, the validation users ignore the request (w = 0); with popularity, request + seeds does not differ significantly from seeds. The requests were written from the same seed films and therefore contain no information beyond them.
5. **Enrichment helps request matching slightly, mainly through knowledge-rich profiles.** gemma4:31b P2 profiles improve request-only matching by +3% to +6% (significant in 2 of 4 settings); v4 labels by up to +3% (not significant after correction); v2 labels not at all. This is the setting in which free-text profiles written with the model's own knowledge show their clearest benefit, consistent with experiment 07. In the hybrids, the effect largely disappears.
6. **Specific wording does not help.** r2 requests are more concrete than r1 requests, but do not improve any method.

**Answer to the abstract's question.** For new users without any rated films, a natural-language request combined with popularity gives more relevant recommendations than popularity alone. It does not outperform simply naming a few films, and LLM enrichment of the movie descriptions improves request matching only slightly. This agrees with Sanner et al. (RecSys 2023), who found language-based preferences competitive with item-based methods in near cold start, and specifies the conditions: competitive with about three named films, and useful mainly in combination with popularity.

---

## 5. Limitations

- **Simulated requests.** The requests were written by an LLM from the user's seed films, not by real users. They can therefore not contain information beyond the seeds; real requests may add preferences the seeds do not show (e.g. "nothing too violent", "something short"), which is the constraint-handling setting of the abstract and is not evaluated here. Real users' requests may also be shorter, vaguer or more specific.
- **One request writer** (gemma4:e4b) and one embedding model; the request is matched by plain embedding similarity, without an LLM reranking or interpreting it.
- **Ground truth is the user's later history,** which on MovieLens favours popular films; this explains part of the strength of popularity and of the hybrids.
- **Weights tuned on 203 validation users,** so some small negative differences (e.g. hybrid request + seeds vs hybrid seeds at N = 10) reflect overfitting of the w and α choice rather than a harmful request.
- **The seed-naming check is a heuristic** (substring match of titles of at least 5 characters).

---

## 6. Next steps

- **Constraint parsing:** let real-style requests contain explicit restrictions (genre, era, length, age suitability) and test whether an LLM can turn them into structured constraints that the constraint layer enforces; measure constraint satisfaction.
- **LLM interpretation and reranking of requests:** instead of embedding the request directly, let an LLM rerank the hybrid's candidates against the request (Hou et al., ECIR 2024, report popularity and position biases to control for).
- **Topic queries with genome ground truth** (evaluation B): requests built from genome tags inside and outside the v4 vocabulary, to test whether v4's advantage is real or only word overlap.

---

## 7. Files

| File | Content |
|---|---|
| `output/experiments/typed_requests/requests_r1.jsonl`, `requests_r2.jsonl` | all requests (user, N, seeds, text, words, seed check, timing) |
| `output/experiments/typed_requests/pilot_requests_r1.jsonl`, `pilot_requests_r2.jsonl` | pilot requests (20 validation users) |
| `output/experiments/typed_requests/per_user.csv` | NDCG, precision and recall per test user, style, N, representation and method |
| `output/experiments/typed_requests/summary.csv` | averages |
| `output/experiments/typed_requests/tuned_parameters.csv` | chosen w and α |
| `output/experiments/typed_requests/significance_tests.csv` | Wilcoxon tests with uncorrected and Bonferroni-corrected p-values |

---

## 8. Reproducing the run

```bash
python typed_request_experiment.py pilot r1
```

```bash
python typed_request_experiment.py
```

The pilot prints example requests for 20 validation users. The full run writes about 4,070 requests (about 1.4 hours with gemma4:e4b on the local GPU; cached and resumable) and then evaluates all methods (a few minutes). It requires the indexes from experiments 05 (base, v2), 07 (31b P2 profiles) and 08 (v4 labels). All random choices use seed 42.
