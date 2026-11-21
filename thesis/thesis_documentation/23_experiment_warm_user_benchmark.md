# Experiment 23: Warm-User Benchmark — Do the Conclusions Hold With a Long History?

**Status:** validation done (2026-10-26). Test users not used (they join the final pre-registered test, doc 18 §9).
**Script:** `warm_benchmark.py` (modes `check`, `base`, `ltr`, `llm [full]`, `report`)
**Results folder:** `output/experiments/warm_benchmark/`
**Related documents:** 06 (cold start, Hybrid E), 13 and 17 (LLM reranking), 16 (LTR v2), 18 §7b (the plan), 20 (NCF, SVD, user-kNN)

---

## 1. Question

Every main result since doc 06 is for **new users** with 3 or 10 seed films. An examiner may ask: *does LTR v2 still win, and does the LLM still trail, when the user has a long history?* This experiment answers it with a history-length curve.

Four questions, fixed in the plan (doc 18 §7b):
1. Does the ranking of methods at 10 seeds still hold at S = 50?
2. Does each method's gap to Hybrid E grow or shrink from S = 10 to S = 50?
3. Does the LLM's deficit to Hybrid E shrink with more history?
4. At what S does every personalised method clearly leave popularity behind?

## 2. Setup

| | |
|---|---|
| Users | the **185 of 203 validation users** with ≥ 50 liked pool films and ≥ 5 more to evaluate (14 had too few liked films, 4 too few left) |
| History | the first **S liked films in time order**, S = 3, 10, 25, 50 |
| Answer key | each user's liked films (rating ≥ their own mean) **minus the first 50**: the same key at every S (96.2 films per user on average, minimum 5) |
| Never recommended | all of the first 50 liked films, at every S |
| Training data | the 9,591 warm users, unchanged; no model sees these users' ratings |
| Metric | NDCG@10 (+ precision, recall); paired Wilcoxon vs Hybrid E and vs popularity, Bonferroni within each S (9 comparisons) |

**What "warm" means here:** a long list of liked films, not a full rating history with dislikes. SVD is the exception: as in doc 20 it gets the seeds' actual ratings.

**This is its own board.** Because the first 50 liked films are excluded and removed from the key at every S, the key is smaller and harder than in docs 06–20 (it lacks the films a user liked first, often the most popular ones). So **S = 10 here (Hybrid E 0.419) is not comparable with the cold-start 10 seeds (0.629)**, and these numbers never replace the earlier ones. For the same reason nothing could be reused from the cold-start runs: Hybrid E's top 30 is different, so the LLM was called again and LTR v2 retrained at every S.

## 3. Methods and settings

| Method | At each S | Chosen on these 185 users |
|---|---|---|
| Popularity | the most-liked films among warm users (the same list at every S) | – |
| Content | embeddinggemma + LLM labels, max over the S films (doc 05) | – |
| Item-CF | mean item-CF similarity to the S films (doc 06) | – |
| Hybrid E | α × item-CF + (1 − α) × popularity | α from the doc 06 grid |
| User-kNN | similarity-weighted sum over the k most similar warm users (doc 20) | k from {10, 20, 50, 100, 200, 400, 1000} |
| SVD | warm-user model, fold-in on the S films' ratings (doc 20) | – |
| NCF | warm-user NeuMF (model 42), fold-in on the S films (doc 20) | S = 3, 10: doc 20's settings; S = 25, 50: from {300, 1000, 2000 steps} × lr 0.05 |
| LTR v2 | R7 features, the step-4 parameters of doc 16 (no re-tuning), 50/50 XGBoost + LightGBM, reranks Hybrid E's top 30; **retrained per S** on 8,535 simulated warm users built by the same 50-film rule, cross-fitted halves A/B as doc 16 | – |
| LLM | DeepSeek V4.1 Flash and Kimi K3, prompt p2, rerank Hybrid E's top 30, no thinking | – |

**Chosen settings** (`settings.json`; curves in `settings_curves.csv`):

| S | Hybrid E α | User-kNN k | NCF fold-in (steps / lr) |
|---|---|---|---|
| 3 | 0.20 | 200 | 300 / 0.01 |
| 10 | 0.25 | 400 | 1000 / 0.05 |
| 25 | 0.40 | 400 | 1000 / 0.05 (300: 0.3949, 1000: 0.4008, 2000: 0.4005) |
| 50 | 0.50 | 100 | 300 / 0.05 (300: 0.4277, 1000: 0.4173, 2000: 0.4179) |

- **Hybrid E's α rises with history** (0.2 → 0.5): the more films the user has liked, the more weight goes to personal similarity and the less to popularity.
- All settings were fixed (saved) before the LTR and LLM runs.

**Checks in the code:**
- Hybrid E from the LLM code equals Hybrid E from `cold_start_experiment` at every S.
- The LTR tables' Hybrid E order gives the base run's Hybrid E score at every S.
- No benchmark user is among the simulated training users.
- Hybrid E on the simulated warm users (0.394, 0.410, 0.424, 0.448) is close to the validation users (0.409, 0.419, 0.426, 0.455), so training and validation users look alike.

## 4. Results (185 validation users)

### 4.1 NDCG@10 by history length

| Method | S = 3 | S = 10 | S = 25 | S = 50 |
|---|---|---|---|---|
| **LTR v2** | **0.4352** | **0.4452** | **0.4616** | **0.4926** |
| User-kNN | 0.4167 | 0.4204 | 0.4436 | 0.4746 |
| Hybrid E (reference) | 0.4093 | 0.4187 | 0.4262 | 0.4549 |
| LLM: Kimi K3, p2 | 0.3960 | 0.4036 | 0.4307 | 0.4540 |
| LLM: DeepSeek V4.1 Flash, p2 | 0.3967 | 0.4064 | 0.4357 | 0.4372 |
| Item-CF | 0.3608 | 0.3931 | 0.4153 | 0.4493 |
| NCF (fold-in) | 0.3808 | 0.3861 | 0.4008 | 0.4277 |
| Popularity | 0.3768 | 0.3768 | 0.3768 | 0.3768 |
| Content (embedding + labels) | 0.2349 | 0.2607 | 0.2730 | 0.3104 |
| SVD (fold-in) | 0.3619 | 0.3445 | 0.3185 | 0.2897 |

![NDCG@10 against history length](../output/experiments/warm_benchmark/curve_ndcg_vs_history.png)

*`curve_ndcg_vs_history.png`: left, the recommenders; right, LTR v2 and the two LLMs with Hybrid E as the reference (LLM lines dashed).*

### 4.2 Difference to Hybrid E (Bonferroni within each S)

| Method | S = 3 | S = 10 | S = 25 | S = 50 |
|---|---|---|---|---|
| LTR v2 | +0.026 (p 0.19) | +0.027 (p 0.20) | +0.035 (p 0.088) | **+0.038 (p 0.004)** |
| User-kNN | +0.007 (1.0) | +0.002 (1.0) | +0.017 (0.098) | **+0.020 (p 0.031)** |
| Kimi K3 | −0.013 (1.0) | −0.015 (1.0) | +0.004 (1.0) | −0.001 (1.0) |
| DeepSeek Flash | −0.013 (1.0) | −0.012 (1.0) | +0.010 (1.0) | −0.018 (1.0) |
| Item-CF | **−0.049 (p < 0.001)** | −0.026 (0.26) | −0.011 (1.0) | −0.006 (1.0) |
| NCF | −0.029 (0.37) | −0.033 (0.29) | −0.025 (0.83) | −0.027 (0.56) |
| SVD | **−0.047 (0.029)** | **−0.074 (p < 0.001)** | **−0.108 (p < 0.001)** | **−0.165 (p < 0.001)** |
| Content | **−0.174** | **−0.158** | **−0.153** | **−0.145** (all p < 0.001) |

**LTR v2 vs user-kNN** (direct, descriptive; Bonferroni over the 4 S): +0.019 (p 0.44), +0.025 (p 0.13), +0.018 (p 0.46), +0.018 (p 0.41). LTR v2 is about 0.02 above user-kNN at every S, never significantly.

### 4.3 Difference to popularity (Bonferroni within each S)

Significantly above popularity (Bonferroni p < 0.05):

| S | Methods |
|---|---|
| 3 | LTR v2 only (+0.059, p 0.001) |
| 10 | LTR v2 (+0.068), user-kNN (+0.044, p 0.043) |
| 25 | LTR v2, user-kNN, Hybrid E (+0.050, p 0.010), DeepSeek (+0.059, p 0.013) |
| 50 | LTR v2, user-kNN, Hybrid E, Kimi K3, item-CF, DeepSeek, NCF |

Content and SVD are significantly **below** popularity at S = 50 (and content at every S).

### 4.4 Does the gap to Hybrid E change from S = 10 to S = 50? (per user, Bonferroni over 9)

| Method | Gap at S = 10 | Gap at S = 50 | Change | p (raw) | p (Bonf.) |
|---|---|---|---|---|---|
| Item-CF | −0.026 | −0.006 | +0.020 | 0.026 | 0.23 |
| User-kNN | +0.002 | +0.020 | +0.018 | 0.017 | 0.15 |
| Kimi K3 | −0.015 | −0.001 | +0.014 | 0.16 | 1.0 |
| Content | −0.158 | −0.145 | +0.014 | 0.40 | 1.0 |
| LTR v2 | +0.027 | +0.038 | +0.011 | 0.32 | 1.0 |
| NCF | −0.033 | −0.027 | +0.005 | 0.78 | 1.0 |
| DeepSeek Flash | −0.012 | −0.018 | −0.005 | 0.75 | 1.0 |
| Popularity | −0.042 | −0.078 | −0.036 | 0.001 | **0.011** |
| SVD | −0.074 | −0.165 | −0.091 | < 0.001 | **< 0.001** |

### 4.5 LLM cost with long histories

| S | DeepSeek input tokens | Kimi input tokens | Seconds per answer (DeepSeek / Kimi) | Answers with issues |
|---|---|---|---|---|
| 3 | 1,356 | 1,454 | 0.84 / 1.35 | 0 |
| 10 | 1,610 | 1,713 | 0.93 / 1.14 | 0 |
| 25 | 2,153 | 2,265 | 0.87 / 1.43 | 0 |
| 50 | 3,039 | 3,165 | 0.99 / 1.86 | 0 |

1,480 answers in total, none repaired or unparsable. Output stays at 94 / 101 tokens.

### 4.6 EASE added (2026-10-29)

`python warm_benchmark.py ease`.
- **Model:** EASE fitted on the 9,591 warm users.
- **λ per S:** chosen on these 185 users from the grid of `ease_tuning.py` (500–200,000), the same way as α and k above. The chosen values are slightly optimistic; the curve is in `settings_curves_ease.csv`.
- **Chosen λ:** S = 3: 80,000; S = 10: 30,000; S = 25: 30,000; S = 50: 3,000. The S = 50 curve is flat (0.465–0.474 for λ 500–30,000), so its exact λ is not meaningful.

| | S = 3 | S = 10 | S = 25 | S = 50 |
|---|---|---|---|---|
| EASE NDCG@10 | 0.4193 | 0.4367 | 0.4421 | 0.4737 |
| Rank among 11 methods | 2 | 2 | 3 | 3 |
| vs Hybrid E (Bonferroni over 10) | +0.010 (1.0) | +0.018 (0.32) | +0.016 (0.34) | +0.019 (0.91) |
| vs LTR v2 (descriptive) | −0.016 | −0.009 | −0.020 | −0.019 |
| vs popularity (Bonferroni) | +0.043 (0.12) | **+0.060 (0.002)** | **+0.065 (< 0.001)** | **+0.097 (< 0.001)** |
| Recall@10 | 0.0504 | **0.0559** | 0.0545 | **0.0621** |

**What EASE adds:**
- **EASE is the strongest single model after LTR v2 at S = 3 and 10,** and about equal with user-kNN at S = 25 and 50 (within 0.002). This matches the cold-start board (doc 26), where EASE was second to LTR v2.
- **Its lead over Hybrid E stays the same as history grows** (gap +0.018 at S = 10 and +0.019 at S = 50, change p 0.81). It is never significant on these 185 users.
- **Highest recall at S = 10 and 50**, slightly above LTR v2, as on the cold-start board. LTR v2 orders the top better; EASE finds about as many liked films.

**Note:** adding EASE makes 10 comparisons per S instead of 9, so every Bonferroni-corrected p in `summary_val.csv` is now ×10/9 of the values in 4.2. No conclusion changes:
- LTR v2 at S = 50: p 0.0046, was 0.004;
- user-kNN at S = 50: p 0.034, was 0.031.

## 5. Findings

1. **The cold-start ranking holds for long histories (question 1).** At every S the order at the top is LTR v2 > user-kNN ≥ Hybrid E, with NCF and SVD below Hybrid E and content last. LTR v2 is first at every S, **significantly above Hybrid E only at S = 50** (+0.038, p 0.004); at S = 3–25 it is +0.026 to +0.035, not significant on this smaller, harder board (in the cold-start test it was significant at 3 and 10 seeds, doc 16).
2. **No method's advantage over Hybrid E changes significantly with history (question 2).** User-kNN (+0.018) and item-CF (+0.020) gain more from history than Hybrid E (raw p 0.017 / 0.026, not significant after correction): a trend, not a result. LTR v2's lead stays roughly constant (+0.011, p 0.32).
   - User-kNN becomes significantly better than Hybrid E at S = 50 (+0.020, p 0.031), as it was at 3 seeds in doc 20.
   - Item-CF catches up with Hybrid E: with long histories, Hybrid E's popularity part matters less (its α rises to 0.5).
3. **The LLM's deficit does not significantly change with history (question 3).** Both LLMs are within 0.018 of Hybrid E at every S, never significantly different.
   - Kimi K3 closes its gap from −0.015 to −0.001 (change +0.014, p 0.16); DeepSeek's gap stays (−0.012 → −0.018). At S = 25 both are slightly above Hybrid E (n.s.).
   - DeepSeek gains nothing from 25 to 50 films (0.4357 → 0.4372) while Hybrid E gains +0.029. Longer seed lists do not hurt the answers' format (0 issues), but they do not help DeepSeek's ranking either; whether this is a "lost in the middle" effect was not tested.
   - Both LLMs are always below LTR v2 and user-kNN.
4. **Personalisation needs history to beat popularity clearly (question 4).** With 3 films only LTR v2 is significantly above popularity; with 10, LTR v2 and user-kNN; with 25, also Hybrid E and DeepSeek; with 50, every collaborative and LLM method. Popularity's gap to Hybrid E grows significantly from S = 10 to 50 (−0.042 → −0.078).
5. **SVD gets worse with every added film** (0.362 → 0.290), significantly, and falls below popularity. A likely reason (not tested): its fold-in sees only liked films, all rated above the user's mean, so more of them pull the user vector towards films rated highly in general rather than towards the user's taste (doc 20 already found SVD dominated by item bias). Report it as a property of fold-in on liked-only histories, not of SVD in general.
6. **NCF stays about 0.03 below Hybrid E at every S.** More history does not close the gap (+0.005).
7. **Content improves with history** (0.235 → 0.310) but stays last, as in docs 05–06.

## 6. Statements for the thesis

- (added 2026-10-29) "EASE was the second-best method at 3 and 10 liked films and level with user-kNN at 25 and 50, about 0.01–0.02 below LTR v2 and 0.01–0.02 above Hybrid E at every history length (not significant on 185 users)."

- "On a warm-user benchmark (185 validation users, histories of 3 to 50 liked films, a fixed answer key), the ranking of methods found for new users held at every history length: LTR v2 first, then user-kNN and Hybrid E; NCF, SVD and content below."
- "LTR v2 was first at every history length and significantly better than Hybrid E with 50 films (+0.038, p = 0.004 after Bonferroni)."
- "The LLM rerankers stayed within 0.018 of Hybrid E at every history length (none significant); more history did not significantly change their gap."
- "SVD with fold-in on liked films only became worse as more films were added."

## 7. Limitations

- **Settings chosen on the reported users** (α, k, NCF fold-in), as in docs 06 and 20; the chosen rows are slightly optimistic. LTR v2 and the LLMs use no settings chosen here.
- **Liked films only.** No dislikes, no full rating history; not the standard random-split warm start (the optional robustness check of doc 18 §7b was not run).
- **A smaller, harder answer key** than the cold-start experiments; numbers are not comparable across boards.
- **185 users.** Differences around 0.02 are near the limit of what this board can detect (LTR vs Hybrid E at S = 3–25, LTR vs user-kNN).
- **LTR v2 not re-tuned per S** (doc 16 step-4 parameters).
- **One candidate order per LLM answer.** Doc 17 §7.5 measured about 0.01 spread between orders, the size of the LLM differences here.
- **Validation only.** The S = 50 comparisons join the final pre-registered test (doc 18 §9).

## 8. Files

| File (`output/experiments/warm_benchmark/`) | Content |
|---|---|
| `settings.json`, `settings_curves.csv` | chosen α, k, NCF fold-in per S, and every value tried |
| `per_user_base.csv` | popularity, content, item-CF, Hybrid E, user-kNN, SVD, NCF per user and S |
| `per_user_ltr.csv`, `ltr/features_{val,simulated}_S{3,10,25,50}.csv` | LTR v2 per user and S; its feature tables |
| `llm/answers_*.jsonl`, `llm/compare_*.csv` | LLM answers (own caches, separate from `llm_rerank/`) and per-run summaries |
| `per_user_all.csv`, `summary_val.csv`, `gap_change_S10_S50.csv` | the report |
| `curve_ndcg_vs_history.png` | the curve |

## 9. Reproduce

```
python warm_benchmark.py check        # counts and prompt sizes, nothing saved
python warm_benchmark.py base         # settings + non-LLM methods (about 20 min, mostly NCF)
python warm_benchmark.py ltr          # LTR v2 at all four S (a few minutes; tables cached)
python warm_benchmark.py llm          # LLM pilot, 20 users
python warm_benchmark.py llm full     # all 185 users (about 45 min; answers cached)
python warm_benchmark.py report       # tables, tests, curve (no model calls)
```
