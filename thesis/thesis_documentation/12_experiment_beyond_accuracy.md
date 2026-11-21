# Experiment: Beyond-Accuracy Evaluation of the Cold-Start Methods

**Status:** finished (run date: 2026-10-23)
**Script:** `beyond_accuracy.py`
**Results folder:** `output/experiments/beyond_accuracy/`
**Related documents:** `06_experiment_cold_start.md`, `07_experiment_outside_knowledge.md`, `09_experiment_typed_requests.md`, `11_experiment_structured_preferences.md`

---

## 1. Questions

All earlier experiments measured accuracy against the user's later ratings (NDCG@10, Precision@10, Recall@10). On this ground truth popularity is very strong, and the best cold-start methods were hybrids with popularity. Accuracy does not show *which* movies a method recommends. This experiment asks:

1. **What do the hybrids give up for their accuracy?** Do they still recommend anything beyond the most popular films?
2. **Do content-based and LLM-assisted methods show value on other criteria:** catalogue coverage, novelty, long-tail recommendations and diversity?
3. **Does LLM enrichment change these properties,** even where it does not change accuracy?

No new model runs were needed. The top-10 lists were rebuilt from the cached embeddings, labels, requests and extracted preferences with the parameters chosen in the earlier experiments.

---

## 2. Method

### 2.1 Users and methods

The users, seeds, candidates and answer key are identical to experiments 06, 07, 09 and 11. All results are on the **814 test users**; seed-based methods use N = 3 and N = 10 seeds, request methods use the r1 requests for N = 3.

| Group | Methods (source) | Parameters |
|---|---|---|
| Baseline | popularity (06) | — |
| Seed-based, pure | content (A), content + v2 labels (B), both with max scoring; embedding (C), embedding + v2 labels (D), embedding + 31b P2 profiles (07), item-CF (E), all with mean scoring | — |
| Seed-based, hybrid | hybrid C, hybrid D, hybrid E (06) | α chosen on validation users in 06 |
| Typed request | request embedding, hybrid request, hybrid request + seeds (09) | w, α chosen on validation users in 09 |
| Structured request | structured preferences from gpt-oss:20b, gemma4:31b and gemma4:e4b (11) | — |

**Validity check:** the NDCG@10 of all 26 rebuilt method settings equals the saved results of the earlier experiments to within 10⁻⁴.

### 2.2 Metrics

| Metric | Definition | Better |
|---|---|---|
| NDCG@10 | as before | higher |
| NDCG@10 low / mid / high | the same list, scored only against the user's relevant movies in one popularity third of the candidate pool (TMDB vote count: low ≤ 845 < mid ≤ 3,172 < high), as in experiment 07. Users without a relevant movie in a third are left out (684 users have one in the low third, 807 in the mid third) | higher |
| Coverage | share of the 1,547 candidate movies that appear in at least one user's top 10 | higher |
| Gini | Gini index of how often each candidate movie is recommended (0 = all equally often, 1 = always the same movie) | lower |
| Novelty | mean −log₂(share of warm users who liked the movie), in bits | higher |
| Long-tail share | share of recommended movies in the low or mid third | higher |
| Diversity | intra-list diversity: mean pairwise genre distance (1 − Jaccard of the TMDB genre sets) within a list | higher |

Genres are used for diversity instead of embeddings so that no method is favoured by being measured with its own representation.

### 2.3 Significance tests

Thirteen pairs of methods were compared (§3.3). Per-user metrics (NDCG, NDCG per third, long-tail share, novelty, diversity) were tested with the Wilcoxon signed-rank test. Coverage and Gini describe the whole set of lists, not a user, so they were tested with a **paired permutation test**: under the null hypothesis a user's two lists are exchangeable, so each user's two lists were swapped at random (10,000 permutations). A bootstrap over users was tried first and rejected because it is biased for coverage: resampling duplicates users, which lowers coverage more for the broader method. All 104 tests were Bonferroni-corrected together.

---

## 3. Results

### 3.1 All methods, N = 3 seeds (814 test users)

| Method | NDCG@10 | NDCG low | NDCG mid | Coverage | Gini | Novelty | Long-tail share | Diversity |
|---|---|---|---|---|---|---|---|---|
| Popularity | 0.5598 | 0 | 0 | 1.3% | 0.992 | 0.77 | 0% | **0.746** |
| **Hybrid E (item-CF)** | **0.5970** | 0 | 0 | 6.1% | 0.984 | 0.87 | 0% | 0.741 |
| Hybrid D (embedding + labels) | 0.5920 | 0 | 0 | 3.9% | 0.987 | 0.84 | 0% | 0.742 |
| Hybrid request + seeds | 0.5809 | 0 | 0 | 3.0% | 0.988 | 0.82 | 0% | 0.724 |
| Hybrid C (embedding) | 0.5778 | 0 | 0 | 2.3% | 0.991 | 0.79 | 0% | 0.764 |
| Hybrid request | 0.5747 | 0 | 0 | 2.3% | 0.991 | 0.79 | 0% | 0.745 |
| Item-CF | 0.5388 | 0.0021 | 0.0348 | 36.7% | 0.914 | 1.51 | 9% | 0.707 |
| Embedding + 31b P2 | 0.3849 | 0.0067 | 0.0382 | 50.3% | 0.839 | 2.54 | 22% | 0.621 |
| Embedding + labels | 0.3820 | 0.0044 | 0.0369 | 48.0% | 0.849 | 2.50 | 22% | 0.620 |
| Embedding | 0.3798 | 0.0046 | 0.0371 | 48.4% | 0.849 | 2.51 | 22% | 0.632 |
| Structured (gpt-oss:20b) | 0.2505 | 0.0058 | 0.0193 | **62.9%** | **0.760** | 3.54 | 35% | 0.638 |
| Structured (gemma4:31b) | 0.2478 | 0.0036 | 0.0160 | 58.7% | 0.777 | 3.40 | 32% | 0.644 |
| Content | 0.2296 | 0.0053 | 0.0354 | 54.8% | 0.802 | 4.05 | 42% | 0.539 |
| Content + labels | 0.2187 | **0.0076** | 0.0315 | 56.8% | 0.786 | 4.33 | **45%** | 0.506 |
| Structured (gemma4:e4b) | 0.2000 | 0.0057 | 0.0230 | 54.8% | 0.796 | 4.00 | 43% | 0.642 |
| Request embedding | 0.1262 | 0.0040 | 0.0161 | 39.1% | 0.893 | **4.89** | 42% | 0.618 |

NDCG high (the relevant movies in the high third) equals the overall NDCG for popularity and all hybrids, and is 0.11–0.50 for the pure methods.

### 3.2 N = 10 seeds (seed-based methods)

| Method | NDCG@10 | Coverage | Gini | Novelty | Long-tail share | Diversity |
|---|---|---|---|---|---|---|
| Popularity | 0.5598 | 1.3% | 0.992 | 0.77 | 0% | 0.746 |
| Hybrid E | **0.6311** | 11.0% | 0.977 | 0.96 | 0.2% | 0.735 |
| Hybrid D | 0.6138 | 5.2% | 0.985 | 0.89 | 0% | 0.727 |
| Hybrid C | 0.6135 | 5.4% | 0.985 | 0.90 | 0% | 0.733 |
| Item-CF | 0.6166 | 25.8% | 0.950 | 1.27 | 5% | 0.719 |
| Embedding + 31b P2 | 0.4633 | 33.3% | 0.914 | 2.21 | 14% | 0.622 |
| Embedding + labels | 0.4576 | 30.3% | 0.922 | 2.19 | 14% | 0.615 |
| Embedding | 0.4540 | 30.9% | 0.922 | 2.20 | 14% | 0.626 |
| Content | 0.3192 | 33.8% | 0.895 | 3.52 | 29% | 0.633 |
| Content + labels | 0.3095 | 37.0% | 0.892 | 3.57 | 35% | 0.602 |

With ten seeds, all pure methods become more accurate and less broad: the mean of ten seed similarities favours movies that resemble many seeds, which are mostly popular.

### 3.3 Significance tests (differences b − a; Bonferroni over 104 tests)

**LLM enrichment, pure methods:**

| Comparison | N | NDCG | NDCG low | Long-tail share | Novelty | Coverage | Gini | Diversity |
|---|---|---|---|---|---|---|---|---|
| Content + labels vs content | 3 | −0.011 (n.s.) | +0.002 (n.s.) | **+3.9 pp** | **+0.28** | +1.9 pp (n.s.) | **−0.016** | **−0.033** |
| | 10 | −0.010 (n.s.) | +0.001 (n.s.) | **+6.7 pp** | +0.06 (n.s.) | **+3.2 pp** | −0.003 (n.s.) | **−0.030** |
| Embedding + labels vs embedding | 3 | +0.002 (n.s.) | 0.000 (n.s.) | −0.2 pp (n.s.) | −0.01 (n.s.) | −0.4 pp (n.s.) | +0.001 (n.s.) | **−0.012** |
| | 10 | +0.004 (n.s.) | 0.000 (n.s.) | −0.7 pp (n.s.) | 0.00 (n.s.) | −0.6 pp (n.s.) | 0.000 (n.s.) | **−0.011** |
| Embedding + 31b P2 vs embedding | 3 | +0.005 (n.s.) | +0.002 (n.s.) | +0.1 pp (n.s.) | +0.03 (n.s.) | +1.9 pp (n.s., p = 0.11) | **−0.010** | **−0.011** |
| | 10 | +0.009 (n.s.) | +0.001 (n.s.) | +0.1 pp (n.s.) | +0.01 (n.s.) | **+2.4 pp** | **−0.008** | −0.004 (n.s.) |

**LLM interpretation of the request (N = 3):**

| Comparison | NDCG | NDCG low | Long-tail share | Novelty | Coverage | Gini | Diversity |
|---|---|---|---|---|---|---|---|
| Structured (gpt-oss:20b) vs request | **+0.124** | +0.002 (n.s.) | **−6.9 pp** | **−1.35** | **+23.8 pp** | **−0.133** | +0.019 (n.s.) |
| Structured (gemma4:31b) vs request | **+0.122** | 0.000 (n.s.) | **−9.9 pp** | **−1.49** | **+19.6 pp** | **−0.116** | **+0.026** |
| Structured gpt-oss:20b vs gemma4:e4b | **+0.051** | 0.000 (n.s.) | **−7.7 pp** | **−0.46** | **+8.1 pp** | **−0.036** | 0.000 (n.s.) |

**The accuracy / breadth trade-off (N = 3):**

| Comparison | NDCG | NDCG low | NDCG mid | Long-tail share | Novelty | Coverage | Gini | Diversity |
|---|---|---|---|---|---|---|---|---|
| Hybrid E vs popularity | **+0.037** | 0 | 0 | 0 | **+0.10** | **+4.9 pp** | **−0.009** | −0.005 (n.s.) |
| Item-CF vs hybrid E | **−0.058** | +0.002 (n.s.) | **+0.035** | **+9.2 pp** | **+0.64** | **+30.6 pp** | **−0.070** | **−0.034** |
| Structured (gpt-oss:20b) vs hybrid request | **−0.324** | **+0.006** | **+0.019** | **+35.2 pp** | **+2.75** | **+60.6 pp** | **−0.231** | **−0.107** |
| Hybrid D vs hybrid C (own α) | **+0.014** | 0 | 0 | 0 | **+0.05** | **+1.6 pp** | **−0.004** | **−0.022** |

Bold = significant after correction. pp = percentage points.

The hybrid D vs hybrid C comparison uses each hybrid's own α (0.12 vs 0.05 at N = 3), so the differences come mainly from the larger weight of the personal signal in D, not from the labels. At the same α, experiment 06 found no significant NDCG difference.

---

## 4. Findings

1. **The hybrids are popularity, slightly re-ordered.** With the α chosen for accuracy (0.05–0.25), every hybrid recommends **only movies from the most popular third**: the long-tail share is 0%, NDCG in the low and mid thirds is exactly 0, and coverage is 2–6% at N = 3 (popularity: 1.3%). Their accuracy gain over popularity (+0.015 to +0.037) is earned entirely by choosing better blockbusters. Hybrid E still adds significant novelty and coverage compared with popularity, but on a very small scale (6% of the catalogue).
2. **Pure content-based and LLM-assisted methods use the catalogue.** They cover 39–63% of the candidate movies (30–50 times popularity at N = 3), with novelty of 2.5–4.9 bits instead of 0.8, and 22–45% of their recommendations in the less popular two thirds. The price is accuracy: their NDCG is 0.13–0.38 at N = 3, against 0.56–0.60 for popularity and the hybrids.
3. **Structured preferences give the broadest recommendations of all methods.** gpt-oss:20b reaches 62.9% coverage and the most even spread of recommendations (Gini 0.76), and it does so with twice the accuracy of the embedded request (+0.124 NDCG). Embedding the request recommends more obscure movies (novelty 4.9 vs 3.5 bits) but fewer different ones: similar requests retrieve the same obscure films. Among the extraction models, gpt-oss:20b covers more of the catalogue than gemma4:31b (62.9% vs 58.7%) at equal accuracy (experiment 11).
4. **LLM labels move bag-of-words content towards the long tail.** With v2 labels, content-based lists contain significantly more long-tail movies (+3.9 pp at N = 3, +6.7 pp at N = 10), are more novel (N = 3) and spread more evenly (Gini at N = 3, coverage at N = 10), without a significant change in NDCG. The labels connect less popular films that share a theme or mood with the seeds.
5. **Knowledge-rich profiles (31b P2) widen embedding-based recommendations slightly.** They increase coverage (+1.9 to +2.4 pp; significant at N = 10) and spread recommendations more evenly (Gini significant at both N), without changing the long-tail share. Labels on embeddings (D vs C) change none of the breadth metrics.
6. **Long-tail accuracy is low for every method, and no LLM effect on it is significant.** The best NDCG on the low third is 0.0076 (content + labels). Its increase over content (+0.002 at N = 3) and that of 31b P2 over the embedding (+0.002) are not significant: only 28–60 users differ at all. On this ground truth, a correct long-tail recommendation is rare for any method.
7. **Popularity does not lose on genre diversity.** Its top 10 spans many genres (0.746), and the hybrids keep this. Pure similarity methods recommend movies resembling the seeds, so their lists are less varied in genre (0.51–0.71). LLM labels and profiles reduce genre diversity further (−0.01 to −0.03, significant), consistent with lists that follow the seeds' themes more closely. Diversity is the one beyond-accuracy metric on which popularity and the hybrids are best.

**Answer to the questions.** The accuracy advantage of popularity and the hybrids is bought with almost no catalogue coverage and no long-tail recommendations. Content-based and LLM-assisted methods are less accurate on this ground truth but recommend 20–50 times as many different movies, many outside the blockbusters. LLM enrichment and interpretation contribute on this axis rather than on accuracy: v2 labels push content-based lists towards the long tail, knowledge-rich profiles widen embedding-based lists, and structured preferences from a typed request give the broadest recommendations of all methods while being twice as accurate as the embedded request. These effects are significant but moderate, and none of them yields measurably more correct long-tail recommendations.

---

## 5. Limitations

- **The hybrids were tuned for NDCG only.** A different α would trade some accuracy for breadth; this experiment shows the two ends of that trade-off, not the curve between them.
- **Beyond-accuracy metrics do not measure usefulness.** A novel recommendation is only valuable if the user would like it; with the user's later history as ground truth, long-tail hits are too rare to show this.
- **Diversity uses TMDB genres only,** a coarse description. Other definitions (embeddings, themes) would rank the methods differently.
- **Popularity is defined by TMDB vote count for the thirds and by warm-user likes for novelty;** both are popularity proxies from different sources.
- **Cold start only** (814 test users, N = 3 and N = 10). Warm start was not evaluated.
- **Coverage and Gini depend on the number of users;** they are compared only between methods on the same 814 users.

---

## 6. Next steps

- **Trade-off curves:** report NDCG against coverage and long-tail share for every α of the hybrids, and choose α with a combined criterion (e.g. the highest coverage within 0.01 NDCG of the best), so that the hybrids can be compared at equal breadth.
- **Re-ranking for the long tail:** apply a calibrated or popularity-penalised re-ranking to the hybrids and test whether LLM signals (labels, structured preferences) help choose *which* long-tail movies to include.
- **Warm start:** repeat the evaluation on the 10,490 warm users of experiments 04–08.

---

## 7. Files

| File | Content |
|---|---|
| `output/experiments/beyond_accuracy/per_user.csv` | top-10 list and all metrics per user, method and N |
| `output/experiments/beyond_accuracy/summary.csv` | averages and set-level metrics per method and N, with the saved NDCG for the validity check |
| `output/experiments/beyond_accuracy/significance_tests.csv` | Wilcoxon and permutation tests with uncorrected and Bonferroni-corrected p-values |
| `output/experiments/beyond_accuracy/pool.csv` | the 1,547 candidate movies (for coverage and Gini) |

---

## 8. Reproducing the run

```bash
python beyond_accuracy.py
```

```bash
python beyond_accuracy.py tests
```

The first command rebuilds all lists, checks them against the saved NDCG, writes the summary and runs the tests (a few minutes; it embeds the 814 requests with embeddinggemma on the local Ollama server, no LLM calls). The second reruns only the tests on the saved lists (about 10 seconds). It requires the results of experiments 06 (α), 07 (31b P2 index), 09 (requests and tuned parameters) and 11 (extracted preferences and test results for all three models). All random choices use seed 42.
