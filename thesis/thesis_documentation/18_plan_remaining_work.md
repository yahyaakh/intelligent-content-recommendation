# Plan: Remaining Work for the Thesis (TDK Abstract Gaps)

**Status:** plan (written 2026-10-25; section 7b added 2026-10-26; 7b done on validation 2026-10-26, doc 23)
**Goal:** close every promise in the TDK abstract ("Large Language Models and Semantic IDs for Personalized Movie Recommendation") that has no experiment yet, with the same protocol as before.
**Related documents:** 06 (cold start, Hybrid E), 13 (LLM as recommender), 16 (LTR v2), 17 (improving the LLM results)

---

## 1. Where we stand

| Question | Answer so far | Document |
|---|---|---|
| Does LLM enrichment help? | only for pure content matching; ≈ 0 once rating data is used | 05–08, 15, 16 |
| Can an LLM recommend for new users? | at best ties Hybrid E (Kimi K3 0.6274 vs 0.6286, validation); reranking or fusing with user-kNN adds nothing over user-kNN (doc 17 §7.6) | 13, 17 |
| Can an LLM personalise without rating data? | yes, +0.040 over popularity (pre-registered test) | 13 |
| Do typed requests help? | only when there are no seed films | 09, 11 |
| Best method for new users | LTR v2: 0.6609 vs Hybrid E 0.6311 on test (+0.030, pre-registered) | 16 |
| Do the conclusions hold with a long history? | yes (validation, 185 users, 3–50 liked films): LTR v2 first at every S (+0.038 vs Hybrid E at S = 50, p 0.004); LLMs within 0.018 of Hybrid E; SVD gets worse with history | 23 |
| Do NCF / SVD help new users? | no: both below Hybrid E (validation); a standard user-kNN beats Hybrid E (0.6417 vs 0.6286); NCF and user-kNN as LTR features: no gain, not adopted (doc 20 §10.1) | 20 |

**LTR improvements (doc 24, 2026-10-27):** user-level features, seed ratings, C30 + 20 user-kNN films, and a blend with user-kNN: none passes the +0.005-on-both-groups rule; LTR v2 stays final.

**Open:** semantic IDs, natural-language constraints, NCF/SVD as models, the rest of doc 17 (GLM-5.3 run, D, F, G), a cost and latency table, and the final pre-registered test. The warm-user benchmark (section 7b) is done on validation (doc 23): the new-user ranking holds for histories up to 50 films.

## 2. The same protocol for everything

- **Data:** MovieLens 25M subset, 1,957 films, the same cold-start split: 9,591 warm users, 203 validation users, 814 test users.
- **Answer key:** liked = rating ≥ the user's own mean. **Main metric:** NDCG@10 (plus precision, recall, coverage where useful).
- **Method choice:** validation users only, with paired Wilcoxon tests and Bonferroni correction within each experiment.
- **Test users:** used once, in one pre-registered final test at the end (section 9).
- **Outputs:**
  - every experiment writes to `output/experiments/<name>/` and never overwrites earlier results;
  - every experiment gets its own `thesis_documentation/NN_*.md`.
- **Pilots:** 20 users before every paid or long run.

## 3. Semantic IDs (highest priority: in the title, not started)

**Status (2026-10-28):** reading list with checked references in doc 19 (TIGER, YouTube semantic IDs, DSI, RQ-VAE, OneRec's residual k-means, a 2026 reproducibility study). Implementation not started.

**What:** each film gets a short code of discrete tokens (e.g. `<12><5><33>`), built from its content embedding, so that similar films share prefixes. The model then reasons over IDs instead of titles (TIGER, Rajput et al. 2023).

**Steps:**
1. **Build the IDs:**
   - start from the best content embedding (embeddinggemma + labels, doc 05);
   - quantise it with residual k-means (3 levels × 64–256 codes), a simple, reproducible stand-in for RQ-VAE;
   - add a 4th token to break collisions.
2. **Check the IDs (free):**
   - collision rate;
   - do films with the same first token share genre, director or decade? (compare with random);
   - codebook usage.
3. **Use 1, semantic-ID similarity:** score candidates by shared prefix length with the seeds; compare with the plain embedding in the cold-start setting.
4. **Use 2, semantic IDs as features:** add prefix-match features (level-1 and level-2 overlap with the seeds) to the LTR v2 feature set; measure on validation.
5. **Use 3, generative retrieval (stretch):**
   - train a small sequence model (GRU or small transformer) on warm users' liked-film sequences, written as semantic-ID tokens;
   - it predicts the next film's ID for cold users from their seeds;
   - compare with item-CF and Hybrid E.
6. **Use 4, the LLM with semantic IDs (optional):** show the LLM the IDs next to the titles, or rerank with them. This is only worth doing if uses 1–3 show a signal.

**Success criterion:** semantic IDs are meaningful (clusters match genre and director well above random) and are compared fairly with embeddings and CF. A negative result is still a result.
**Time:** 3–4 days (steps 1–4 in 2 days; step 5 in 1–2 days). **Cost:** free (local).
**Doc:** 19.

## 4. NCF and SVD as models, and combinations

**Status (2026-10-26), see doc 20:**
- **Steps 1–3 done.** NCF and SVD were retrained on the warm users and evaluated with fold-in on validation:
  - NCF 0.598 (3-model mean 0.589) and SVD 0.524 at 10 seeds, both below Hybrid E;
  - SVD is below popularity.
- **New finding:** a standard user-kNN (k = 200) beats Hybrid E (0.6417 / 0.6053).
- **NCF test:** pre-registered, not yet committed or run.
- **Step 4 dropped:** NCF/SVD alone are already below Hybrid E.
- **Step 5 is now doc 20 §10 (L1–L6):** NCF and user-kNN as LTR features, cross-fitted. **L1 + L2 at 10 seeds done (2026-10-26):** no feature group passed the decision rule (held-out +0.001 to +0.002, validation −0.004 to −0.008); LTR v2 (R7) stays the final model. L3–L6 open, low priority.

**What:** collaborative models that learn from all warm users' ratings, compared with item-CF and combined with the other signals.

**Steps:**
1. **Retrain NCF** (`recomm.py` `NCFRecommender`) on the corrected half-star data. The saved models were trained on rounded ratings, so they can't be reused.
2. **Cold-start use:** NCF and SVD need an embedding for the new user.
   - Use **fold-in**: fit the user vector from the seed films only, with the item factors fixed.
   - This is the fair version; no retraining with test ratings.
3. **Alone:** NCF fold-in and SVD fold-in vs item-CF and Hybrid E (validation, N = 3 and 10).
4. **Hybrids:** NCF/SVD + popularity with α tuned on validation (as Hybrid E).
5. **Combinations:** the NCF score as a feature in LTR v2, together with the semantic-ID features and the LLM rank, to show whether they add anything beyond G1–G5.

**Success criterion:** a clear ranking of CF families for new users, and the answer to "does NCF + LLM + XGBoost beat LTR v2?".
**Time:** 2–3 days (NCF training is the slow part). **Cost:** free (local GPU/CPU).
**Doc:** 20.

## 5. Natural-language constraints

**Status (2026-10-26):** E1 done (doc 21): against the US age certification of 1,360 films, the LLM audience label finds films unsuitable for under-13s with F1 0.905 vs 0.696 for the recomm.py genre + keyword rule (recall 0.968 vs 0.590). Next: the age-rating filter on the audience label; E2 (NDCG cost of hard constraints), E3 (recomm.py soft layer on top of Hybrid E / LTR v2), E4 (parsing typed constraints, below).

**Status (2026-10-28):** constraint layer rebuilt (doc 25): LLM audience label as child/teen filter, request dict (genres, moods, themes, actors, directors, intensity limits) from menus or a typed sentence parsed by DeepSeek; soft layer switched off. Open points and E2–E4 are listed in doc 25 §4–5.

**What:** a new user types a request with hard rules ("no horror, under 2 hours, something from the 90s"). An LLM turns it into structured constraints, and the system must obey them.

**Steps:**
1. **Constraint schema:** genre include/exclude, runtime max/min, decade or year range, language, age rating / child-safe, director or actor include/exclude. The existing `ConstraintLayer` in `recomm.py` covers part of this.
2. **Test set:**
   - about 200 requests generated from validation users' seeds, with 1–3 constraints each;
   - the true constraints are known, because they are generated from a template first and then paraphrased by an LLM;
   - about 30 hand-written requests, to check that it works on human wording.
3. **Parsing accuracy:** LLM output vs the true constraints (exact match, per-field precision and recall). Compare:
   - a small local model (gemma4:e4b);
   - DeepSeek Flash;
   - a rule-based baseline: keyword rules;
   - optionally the BART zero-shot baseline from `LLMS_test.py`.
4. **Constraint satisfaction:**
   - share of top-10 films that break a constraint, before vs after filtering (Hybrid E, LTR v2, LLM rerank);
   - the accuracy cost of filtering (NDCG@10 on the films that satisfy the constraints).
5. **LLM reranking with constraints in the prompt vs hard filtering:** does the LLM obey them by itself?

**Success criterion:** parsing accuracy per model; a violation rate close to 0 with filtering; the accuracy cost measured.
**Time:** 2–3 days. **Cost:** about $1 (DeepSeek Flash).
**Doc:** 21.

## 6. Finish doc 17 (improving the LLM results)

| Item | What | Time | Cost |
|---|---|---|---|
| GLM-5.3 p5, 203 users | the only open lead (+0.012 vs Hybrid E on the 20-user pilot); run overnight | about 8 h (overnight) | Ollama Pro credit |
| DeepSeek Flash p5 + thinking | control: is a GLM gain due to reasoning? | about 1.5 h | about $1 |
| DeepSeek V4 Pro p5 | p2 is done (0.6055); p5 not yet | about 15 min | small |
| minimax-m3, nemotron-3-ultra | 20-user p2 pilots, to close "largest models" | about 30 min | small |
| D: pairwise / sliding window | RankGPT-style windows of 10 with step 5, or pairwise comparisons; removes position bias | 0.5 day code + about 2 h run | $2–5 |
| G: negative feedback | add 3–5 films the user rated low; a separate information setting | 0.5 day | about $0.30 |
| F: fine-tuning | LoRA on a 4–8B model (Qwen/gemma) with the 9,084 simulated warm users; Colab/Kaggle GPU | 1–2 days | free GPU or about $10 |

**Order:** start the GLM-5.3 run overnight first. Then the DeepSeek-thinking control, V4 Pro p5, the pilots, D, G. F last, or a stated limitation if time runs out.
**Doc:** 17 (sections 7.5 onwards).

## 7. Cost and latency table

**What:** one table for the thesis, with per-user cost and time for every method.

**Columns:** method, NDCG@10 (validation / test), input tokens, output tokens, seconds per user, $ per 1,000 users, needs a GPU (yes/no), runs offline (yes/no).

**Sources:**
- the LLM caches already store tokens and seconds for every answer;
- Hybrid E, LTR, NCF and semantic IDs need timing only: time 1,000 users, CPU only.

**Time:** 0.5 day. **Cost:** free.
**Doc:** a section in the final summary (or doc 22).

## 7a. One cold-start comparison of every method (`final_comparison.py` + CLI option 11)

**What:** one script that ranks every method of the thesis in the **same cold-start setup**, and a CLI option 11 in `recomm.py` that runs it. Options 9 and 10 stay as they are, as warm-start checks:
- option 9: SVD rating prediction, RMSE/MAE only;
- option 10: a random 75/25 warm-user split.

**The same setup for every method (as docs 06–17):**
- **Training data:** the 9,591 warm users only.
- **Evaluation users:** the 203 validation users by default, or the 814 test users for the final test (section 9).
- **Input per user:** the first 3 or 10 liked films (seeds). The answer key is their later liked films (rating ≥ their own mean).
- **Candidates and metrics:** the same candidate pool, the same metrics (NDCG@10, precision, recall, coverage).
- **Statistics:** per-user results saved; paired Wilcoxon vs Hybrid E with Bonferroni correction.

**Methods in the table:**

| Group | Method | How it gets a new user's scores | Source |
|---|---|---|---|
| Baselines | Popularity | count of positive ratings (the thesis definition) | doc 06 |
| | Content | embeddinggemma + labels, max over the seeds | doc 05 |
| Collaborative | Item-CF | mean similarity to the seeds | doc 06 |
| | **SVD** | fold-in: user vector fitted from the seeds, item factors fixed | doc 20 (new) |
| | **NCF** | fold-in from the seeds (or the average of similar warm users' embeddings) | doc 20 (new) |
| Hybrids | Hybrid E | item-CF + popularity, α from validation | doc 06 |
| | SVD / NCF + popularity | same recipe, α from validation | doc 20 (new) |
| Semantic IDs | prefix similarity, and generative retrieval if built | from the seeds' IDs | doc 19 (new) |
| LLM | best reranker (DeepSeek / Kimi, p2 / p5), fused version (method A) | **cached answers only, no new model calls** | docs 13, 17 |
| Learned ranker | LTR v2 (XGBoost + LightGBM ensemble), + NCF/semantic-ID features if they help | saved models | docs 16, 20 |

**Rules:**
- **No new LLM calls.** If a cached answer is missing for a user, the script reports it; it never calls a model.
- **One code path.** Option 11 imports `final_comparison.py` and runs the validation table. The test split can only be run through the pre-registered final test (section 9), which is guarded by a run-once file as in doc 16.
- **Output:** `output/experiments/final_comparison/` (per-user CSV, summary, tests), printed as one table in the CLI.

**When:** built step by step. The skeleton (baselines, item-CF, Hybrid E, LTR v2, cached LLM) can be built now. SVD/NCF are added after doc 20, and semantic IDs after doc 19. **Time:** about 1 day in total.

## 7b. Warm-user benchmark: do the conclusions hold for users with a long history?

**Done on validation (2026-10-26): results in doc 23.** Steps 1–4 below are complete; step 5 (test) waits for the final pre-registered test.

**Why:** since doc 06 every main result is for **new users** (3 or 10 seed films). The main methods (Hybrid E, user-kNN, LTR v2, NCF, SVD, LLM reranking) have never been compared for users the system already knows well. The warm results that exist (docs 04–05, CLI options 9–10) cover other methods and other splits, so they cannot answer this. An examiner may ask: *does LTR v2 still win, and does the LLM still trail, when the user has a long history?*

### Design (recommended): a history-length curve on the same held-out users

Keep the cold-start split, but give each held-out user **more history**: the first S liked films in time order, for **S = 3, 10, 25, 50**.

| | Choice | Reason |
|---|---|---|
| Users | the 203 validation / 814 test users with **at least 55 liked pool films** (50 seeds + `MIN_TEST_RELEVANT` 5) | the same users at every S |
| Answer key | their liked films **minus the first 50**, the **same key for every S** | differences between S then come from the history alone |
| Never recommended | all 50 first liked films, at every S | as the 10 seeds now |
| Training data | the 9,591 warm users, unchanged | no model is retrained for the users; no leakage |
| Metrics and statistics | NDCG@10 (+ precision, recall); paired Wilcoxon vs Hybrid E, Bonferroni within each S | as everywhere |

**What it does not cover:** the history is **liked films only**, as for the seeds. There are no dislikes and no full rating history; SVD is the exception, and still gets the seeds' own ratings. So "warm" here means "much more history", not "the standard random-split warm start". This is stated in the doc.

**Note:** the 3- and 10-seed numbers on this board will differ from doc 06, because of fewer users and a smaller key. They form their own board and never replace the doc 06 numbers.

### What each method needs at S = 25 and 50

| Method | Change | Chosen on validation | New model calls / training |
|---|---|---|---|
| Popularity | none | – | – |
| Item-CF | mean over S seeds | – | – |
| Hybrid E | α per S | α grid of doc 06 | – |
| User-kNN (standard) | k per S | k ∈ {10, 20, 50, 100, 200, 400, 1000} | – |
| SVD fold-in | ridge fit to the S seeds' ratings (λ = reg × S) | – | – |
| NCF fold-in | Adam steps and learning rate per S | the grid of `cold_start_baselines.FOLD` | – (warm NCF reused) |
| LTR v2 | retrain on the simulated warm users with S seeds: same cross-fitting, same features, the tuned parameters of doc 16 step 4 (no retuning) | – | about 1–2 h of feature building per S |
| LLM reranking (DeepSeek Flash p2, Kimi K3 p2) | prompt with S seed films | – | **new calls** at S = 25/50 (the caches hold the 3/10-seed prompts only) |

**~~Free part~~ (corrected 2026-10-26, while writing the code):** the cached 3- and 10-seed LLM answers **cannot** be reused. Here the first 50 liked films are excluded at every S (in the cold-start runs only the first 10 were), so Hybrid E's top 30, and with it the prompt, differs for most users even at S = 3 and 10. The LLM therefore needs new calls at all four S, and LTR v2 is retrained at all four S (with simulated warm users built by the same 50-film rule). Both are still cheap: about 185 × 4 calls per model, about 1–2 h of feature building per S.

**LLM cost:** the p2 prompt is about 1,600 input tokens with 10 seeds. With 50 seeds it is about 3,200. DeepSeek Flash on 203 users is about 5 minutes and well under $0.50; Kimi K3 about 10 minutes of Ollama Pro credit. Pilot on 20 users first. Long seed lists may also expose a "lost in the middle" effect, which is reported if seen.

### Questions answered (validation first)

1. **Ranking per S:** does the validation order at 10 seeds (LTR v2 > user-kNN ≥ Hybrid E ≥ best LLM > NCF ≈ item-CF > SVD) still hold at S = 50?
2. **Gap vs history:** does each method's difference from Hybrid E grow or shrink from S = 10 to S = 50? This is tested per user as a difference of differences (Wilcoxon).
3. **The LLM question:** does the LLM's deficit to Hybrid E shrink with more history, or does more history help CF more than the LLM?
4. **Personalisation vs popularity:** at what S does every personalised method clearly leave popularity behind?

### Steps

1. **Pre-check (free, minutes):** count the validation and test users with ≥ 55 liked pool films. If fewer than about 120 validation users remain, use S ≤ 25 (≥ 30 liked) instead and say so. **Done 2026-10-26: 185 of 203 validation users qualify** (answer key 96 liked films per user on average, minimum 5); test users about 758 of 814. S = 50 is kept.

**Code (written 2026-10-26): `warm_benchmark.py`**, outputs in `output/experiments/warm_benchmark/`:
- `check`: counts and prompt sizes;
- `base`: chooses the settings per S on the 185 users (Hybrid E α from the doc 06 grid; user-kNN k from {10, …, 1000}; NCF fold-in as doc 20 for S = 3/10 and from {300, 1000, 2000 steps} × lr 0.05 for S = 25/50), saves them to `settings.json`, then scores popularity, content, item-CF, Hybrid E, user-kNN, SVD and NCF;
- `ltr [S …]`: LTR v2 per S (R7, step-4 parameters, no tuning, cross-fitted halves), feature tables cached in `ltr/`;
- `llm [full] [S …]`: DeepSeek V4.1 Flash and Kimi K3, p2, 20-user pilot or all users; caches in `warm_benchmark/llm/` (never mixed with the cold-start caches);
- `report`: one table per S (vs Hybrid E and vs popularity, Bonferroni within S), the S = 10 → 50 gap change per method, and `curve_ndcg_vs_history.png`. Runs on part of the users (pilots) are shown separately.
- The leaderboard boards "history S = …" are not added yet.
2. **Code:** `warm_benchmark.py`, reusing `final_comparison.py` and `cold_start_baselines.py`:
   - a user builder with a 50-film seed pool and the fixed key;
   - per-S settings chosen on validation;
   - outputs in `output/experiments/warm_benchmark/`;
   - new leaderboard boards "history S = …".
3. **Validation runs:** non-LLM methods (minutes, NCF fold-in about 10 minutes); LTR v2 retraining per S; LLM pilots (20 users), then the full 203.
4. **Doc 23:** results, the curve plot (NDCG@10 vs S per method), and the limits above.
5. **Test:** the chosen S = 50 comparisons join the final pre-registered test (section 9). There is no separate test run.

**Optional robustness check (only if the curve changes a conclusion, or an examiner asks): standard warm start.**
- Hold out each warm user's last 20% of ratings in time order.
- Rebuild item-CF, popularity, user-kNN, SVD, NCF (about 40 minutes) and the LTR training table from the remaining 80% only.
- Evaluate on the held-out 20%.

It is heavier, and the LTR simulated users must also be built from the 80% only, so it is not part of the main plan.

**Success criterion:** one table per S and one curve. Each cold-start conclusion is either confirmed for long histories or scoped to new users.
**Time:** about 1.5–2 days (0.5 code, about 0.5 runs incl. LTR retraining, 0.5 doc). **Cost:** under $1 plus Kimi credit.
**Doc:** 23.

## 8. Order (days are working days from 2026-10-26)

| Day | Work |
|---|---|
| 1 | Start the GLM-5.3 p5 run (overnight). Run the DeepSeek-thinking control, V4 Pro p5 and the large-model pilots. Update doc 17. |
| 2–3 | Semantic IDs: build, check, similarity and LTR features (steps 1–4). |
| 4 | Semantic IDs: generative retrieval (step 5). Write doc 19. |
| 5–6 | NCF/SVD: retrain, fold-in, hybrids, LTR combination. Write doc 20. |
| 7–8 | Natural-language constraints: test set, parsing, satisfaction. Write doc 21. |
| 9 | Doc 17 methods D and G. |
| 10 | `final_comparison.py` + CLI option 11 (all methods, cold-start setup). Cost and latency table. |
| 11–12 | Warm-user benchmark (section 7b): pre-check, code, validation runs incl. LTR retraining and LLM at S = 25/50. Write doc 23. Choose the final variants. |
| 13 | **Final pre-registered test** (section 9). |
| later | F (fine-tuning) if time allows; otherwise a limitation. |

## 9. Final pre-registered test (once, at the end)

1. **Fix the variants in advance,** chosen on validation by rules written before the run, in a pre-registration file:
   - the best LLM variant from doc 17;
   - the best semantic-ID method;
   - the best NCF/SVD model;
   - the best combined LTR model;
   - the warm-user (S = 50) comparisons of section 7b, with their per-S settings fixed.
2. **Commit the pre-registration file before the run,** so its date is provable (this was not done for doc 16).
3. **Run once on the 814 test users** with `final_comparison.py` (section 7a), for N = 10 and N = 3. Compare each with Hybrid E and with LTR v2 (0.6609). Use Bonferroni over all planned comparisons.
4. **Report everything,** including the null results.

## 10. Risks

- **Semantic IDs could take longer than planned.** Fall-back: residual k-means IDs with use 1 + use 2 only (the generative step becomes future work).
- **NCF retraining may be slow on this machine.** Fall-back: SVD fold-in only (already proven fast via numpy eigh).
- **GLM-5.3 may stop overnight.** The cache resumes it.
- **Warm benchmark: too few users with 55+ liked films.** Fall-back: S ≤ 25 (section 7b, step 1).
- **Scope:** every claim stays scoped to what was tested (see doc 17 §5). Until section 7b is done, every main result is stated as a new-user (cold-start) result.

---

## 11. Gaps against the submitted TDK abstract (recorded 2026-10-29)

The TDK abstract is submitted and cannot change, so the report must deliver what it promises. Three promises have no evidence yet:

1. **Constraint satisfaction (E4, doc 25 §5). Most important: the abstract names it twice.**
   - About 50 typed requests, each with the correct structured answer.
   - Measure the LLM parser's accuracy per field.
   - Measure how often the hard constraints are satisfied in the returned lists.
   - Also record validation rejections, time and tokens, with the menu as the baseline.
2. **Cost of constraints (E2, doc 25 §5).** How much NDCG@10 drops when a constraint filters the list (validation users, simulated constraints). Small and quick.
3. **Web app (`web/`, Django).** Check what it can do now. Add constraint input and a method comparison if they are missing. The abstract promises a web app where users, including new users, express preferences and constraints, explore recommendations and compare the evaluated methods.

Also to collect: one table of computational cost and response time (tokens and seconds per answer) from the LLM docs. The abstract names both.
