# Plan: Can the LLM Results Be Improved? (Methods Not Yet Tested)

**Status:** plan (written 2026-10-25); methods A, B, C and E (Kimi K3) done (validation); 2026-10-26: two more models (DeepSeek V4.1 Flash `1017`, MiniMax M3) and LLM reranking / fusion with user-kNN done (§7.6); GLM-5.3 p5 full run open
**Goal:** leave no reasonable method untested before stating in the thesis that LLMs do not beat collaborative filtering for new users. Every claim must be scoped to what was actually tested.
**Related documents:** `13_experiment_llm_as_recommender.md` (LLM reranking and generation), `15_experiment_learning_to_rank.md`, `16_experiment_learning_to_rank_v2.md`

---

## 1. Starting point

Test users (814), 10 seeds, NDCG@10:

| Method | NDCG@10 |
|---|---|
| LTR v2 final model (doc 16) | 0.6609 |
| Hybrid E (item-CF + popularity) | 0.6311 |
| **Best LLM: DeepSeek V4.1 Flash reranking Hybrid E's top 30 (p2)** | **0.6169** |
| DeepSeek V4.1 Flash reranking popularity (no rating data) | 0.5999 |
| Popularity | 0.5598 |

## 2. Why the LLM falls short of Hybrid E

1. **No collaborative data.** The LLM sees titles, directors, cast and a seed profile, but not who liked what together. Hybrid E and the learned rankers rely on exactly that signal; co-visitation was the strongest new feature in doc 16.
2. **The answer key favours popular films.** The LLM picks less mainstream films (novelty and coverage up, doc 13 §6.2), so part of the gap is the metric's preference, not worse recommendations.
3. **Listwise ranking of 30 films is unstable.** Position bias is about 0.2, and a quarter of the top 10 changes with the candidate order (doc 13 §6.3).
4. **Zero-shot only.** The model never sees how MovieLens users behave.

## 3. Already tested (no open question)

| Direction | Tested | Result |
|---|---|---|
| Model strength | 12 models, 8 families, 4B to frontier class (DeepSeek V4 Pro, GLM, Mistral Large, gpt-oss 120B) | bigger is not better; V4 Pro < V4.1 Flash |
| Reasoning | GLM, gpt-oss "high", DeepSeek thinking | helps only in idea 3 (reranking popularity) |
| Prompt content | p1–p4 (profile, LLM labels, plot, keywords) | p2 best; more information hurts |
| Seeds | 3 vs 10 | 10 needed |
| Position bias | 3 shuffles averaged | no gain |
| Role | rerank Hybrid E, rerank popularity, free generation | reranking Hybrid E best |
| LLM rank as a feature of a learned ranker | doc 15 (trained on 203 users) | redundant |

## 4. Not yet tested: the gaps a reviewer could raise

| # | Method | Idea | Cost | Could it change the conclusion? |
|---|---|---|---|---|
| **A** | **Score fusion** | blend the cached LLM rank with Hybrid E's rank, with one weight tuned on validation (or reciprocal rank fusion) | free (cached answers), about 30 min | low to medium. Doc 15's learned ranker (R5) said no, but it was trained on 203 users; a one-weight fusion is more robust |
| **B** | **CF evidence in the prompt (p5)** | show the LLM, per candidate, what it lacks: "liked by X% of users who liked your seeds" (co-visitation), popularity, Hybrid E rank. The LLM becomes a reasoner over the signals | about $0.15 per 203 users (DeepSeek Flash) | **high.** The direct test of "the LLM only lacks data". If it still cannot beat Hybrid E, that is a strong statement |
| **C** | **Few-shot examples** | 2–3 warm users in the prompt: "these were their seeds, these they liked later" | about $0.30 | medium. Gives the model the dataset's behaviour |
| **D** | **Pointwise / pairwise ranking** | score each candidate separately, or compare pairs (PRP, sliding window as in RankGPT), instead of ordering 30 at once; removes position bias | 10–30× more calls, $2–5 | medium. The literature shows these beat listwise ranking for small and mid-size models |
| **E** | **One or two frontier models** | e.g. Kimi K2 / Qwen3-Max class on Ollama Pro, or GPT-5 / Claude / Gemini via API, with p2 and with the best of B/C | $1–10 | low (bigger has not helped so far), but closes the "you did not try the strongest model" objection |
| **F** | **Fine-tuning a small LLM** | LoRA on gemma/Qwen 4–8B with the 9,084 simulated users (TALLRec / LlamaRec style), the method in the literature where LLMs reach or beat CF | GPU (Colab/Kaggle T4 or cloud), 1–2 days of work | **high.** Without it, every claim is only about off-the-shelf, prompted LLMs |
| **G** | **Negative feedback** | also show films the user rated low ("did not like X") | about $0.15 | low to medium. A new information setting, so it must be reported separately and fairly |

**Also cheap:**
- reranking only the **top 15** instead of 30 (an easier task);
- **p2 + thinking in idea 1** with DeepSeek, since thinking helped in idea 3.

## 5. Protocol

1. **Claims stay scoped.** For example: "prompted off-the-shelf LLMs (12 models, zero-shot, listwise, with and without reasoning) did not beat Hybrid E", never "LLMs cannot beat CF".
2. **Methods A–E (and G) run on the 203 validation users only,** with the same protocol as doc 13: paired Wilcoxon tests, Bonferroni within each method.
3. **The best LLM variant is chosen by a rule fixed in advance.**
4. **One pre-registered test** of that variant on the 814 test users: vs Hybrid E, and vs the current best LLM (0.6169).
5. **F (fine-tuning) is a separate experiment,** or an explicit limitation if time runs out. Its score can later become a feature of the learned ranker (together with NCF and semantic IDs).

## 6. Order

1. **A:** score fusion (free).
2. **B:** CF evidence in the prompt.
3. **C:** few-shot, and **E:** one frontier model, both on the best prompt from B.
4. **D:** pairwise or sliding window, if B/C show that ranking, not information, is the bottleneck.
5. **F:** fine-tuning.
6. **G:** negative feedback (separate information setting).

## 7. Results

*(filled in as the methods are run)*

| # | Method | Validation NDCG@10 | vs Hybrid E (0.6286) | Status |
|---|---|---|---|---|
| A | Score fusion, best: DeepSeek V4.1 Flash rank blend (w = 0.4 in every fold) | 0.6395 | +0.011 (p = 0.33, Bonf. 1.0) | done: no LLM-specific gain |
| B | CF evidence in the prompt, DeepSeek V4.1 Flash p5 (co-visitation + popularity) | 0.6259 | −0.003 (p = 0.96) | done: the LLM follows the data but does not beat Hybrid E |
| B | p6 = p5 + Hybrid E's rank | 0.6266 | −0.002 (p = 0.66) | done: mostly copies Hybrid E |
| C | Few-shot p7: p5 + 3 fixed random warm users as worked examples | 0.6231 | −0.005 (p = 0.85) | done: no gain over p5 |
| C | Few-shot p8: p5 + the 3 most similar warm users as examples | 0.6221 | −0.007 (p = 0.62) | done: no gain over p5 |
| E | Frontier model Kimi K3, p2 | 0.6274 (mean of 3 candidate orders: 0.6207) | −0.001 (p = 0.86); 3-order average −0.007 (p = 0.36) | done: just below Hybrid E, = DeepSeek |
| A+E | Fusion 0.4 × Hybrid E + 0.6 × Kimi K3 p2 | 0.6408 (CV weight: 0.6325) | +0.012 (p = 0.32) | done: = DeepSeek fusion, n.s. |
| E | Frontier model Kimi K3, p5 | 0.6207 | −0.008 (p = 0.73) | done: no gain from the CF data |
| E | GLM-5.3 (always reasons), p5 | 20-user pilot only: 0.5518 vs Hybrid E 0.5395 | +0.012 (p = 0.71) | open: full run (about 8 h) |
| E | DeepSeek V4.1 Flash, 17 Oct snapshot (`1017`), p2 / p5 | 0.6233 / 0.6248 | −0.005 (p = 0.54) / −0.004 (p = 0.95) | done (§7.6): = the old snapshot |
| E | MiniMax M3, p2 / p5 | 20-user pilot only: 0.5172 / 0.5278 | −0.022 / −0.012 | dropped after the pilot (§7.6) |
| H | LLM reranks **user-kNN's** top 30 (p5): DeepSeek Flash / Kimi K3 | 0.6314 / 0.6220 | +0.003 / −0.007; vs user-kNN (0.6417): −0.010 (p = 0.23) / −0.020 (p = 0.055) | done (§7.6): no gain over user-kNN |
| A+H | Fusion of user-kNN with the LLM ranking, best: DeepSeek score blend (CV) | 0.6489 | vs user-kNN +0.007 (p = 0.051, Bonf. 0.76) | done (§7.6): content similarity fuses better (0.6548) |

### 7.1 Method A: score fusion (2026-10-25, `llm_fusion_experiment.py`, `output/experiments/llm_fusion/`)

**Setup:**
- **Inputs:** the cached idea-1 rankings (p2, 10 seeds, Hybrid E's top 30) of 6 LLMs, so no new model calls.
- **Users:** the 203 validation users.
- **Fusion methods:**
  - rank blend: w × Hybrid E rank + (1 − w) × LLM rank;
  - score blend: Hybrid E score with the LLM position score;
  - **w** chosen by 5-fold cross-validation over users, from 0, 0.05, …, 1;
  - reciprocal rank fusion (RRF, k = 60, no tuning).
- **Controls:** the same fusions with three non-LLM orders:
  - random (averaged over 10 orders);
  - popularity;
  - content similarity (`emb_max`, plain-text embeddings, no LLM).

**Results (NDCG@10; Hybrid E 0.6286):**

| Order fused with Hybrid E | Alone | Rank blend (CV) | Score blend (CV) | RRF |
|---|---|---|---|---|
| DeepSeek V4.1 Flash | 0.6212 | **0.6395** | 0.6283 | 0.6382 |
| GLM-5.3-Flash | 0.6201 | 0.6265 | 0.6279 | 0.6332 |
| gemma4:e4b | 0.6176 | 0.6332 | 0.6306 | 0.6268 |
| Mistral Large 4 | 0.6135 | 0.6260 | 0.6274 | 0.6241 |
| DeepSeek V4 Pro | 0.6055 | 0.6292 | 0.6349 | 0.6301 |
| Nemotron 3 Super | 0.5965 | 0.6364 | 0.6334 | 0.6324 |
| *Control: random* | *0.5350* | *0.6283* | *0.6278* | *0.5926* |
| *Control: popularity* | *0.5762* | *0.6278* | *0.6251* | *0.6079* |
| *Control: content similarity (no LLM)* | *0.6189* | *0.6296* | *0.6317* | *0.6356* |

**Findings:**
- **Fusion lifts every LLM to Hybrid E's level or slightly above** (−0.004 to +0.011). Its LLM-alone score was 0.006 to 0.032 below Hybrid E. **No fusion is significantly better than Hybrid E** (all Bonferroni p = 1.0 over 27 tests).
- **The LLM orders carry real information beyond Hybrid E:** fusing with a random or popularity order gains nothing, or loses.
- **That information is not LLM-specific.** Fusing with plain content similarity (an embedding, no LLM) gives about the same result. DeepSeek fusion vs content fusion: rank blend +0.010 (p = 0.34), RRF +0.003 (p = 0.84), score blend −0.004 (p = 0.54).
  - This agrees with doc 15: DeepSeek's ranking correlates about 0.5 with content similarity.
- **Conclusion for A:** fusing an LLM reranking with collaborative filtering does not produce a significant gain. Whatever it adds, a content embedding adds just as well.
- **Candidate for the final LLM test:** DeepSeek V4.1 Flash rank blend, w = 0.4 (chosen in all 5 folds).

### 7.2 Method B: CF evidence in the prompt (2026-10-25, `llm_rerank_experiment.py`, prompts p5/p6)

**Setup:**
- **Model and setting:** DeepSeek V4.1 Flash, no thinking, idea 1 (rerank Hybrid E's top 30), 10 seeds, the 203 validation users, the same shuffled candidate order as p2.
- **p5** = p2 + one data line under each candidate, computed from the 9,591 warm users only (liked = rating ≥ the user's own mean, as everywhere):
  - "liked by X% of the fans of the user's films": co-visitation P(candidate | seed), averaged over the 10 seeds, plus the highest seed with its title;
  - "liked by X% of all users": popularity.
  - Example: `[data: liked by 61% of the fans of the user's films (most: 68% of the fans of GoodFellas); liked by 47% of all users]`
- **p6** = p5 + "our system's rank: k of 30" (Hybrid E's rank), to see whether the model only copies the system when it sees its answer.
- **Cost:** about 2,900 (p5) and 3,300 (p6) input tokens per user vs 1,600 for p2; about 1 s per answer; no unparsable answers.

**Results (NDCG@10, 203 validation users):**

| Ranking of Hybrid E's top 30 | NDCG@10 | vs Hybrid E | Paired test |
|---|---|---|---|
| Hybrid E | 0.6286 | – | – |
| DeepSeek p2 (no CF data) | 0.6212 | −0.007 (p = 0.37) | – |
| **DeepSeek p5 (+ co-visitation, popularity)** | **0.6259** | −0.003 (p = 0.96) | p5 − p2: +0.005 (better 102, worse 93, p = 0.30) |
| DeepSeek p6 (+ Hybrid E rank) | 0.6266 | −0.002 (p = 0.66) | p6 − p5: +0.001 (p = 0.97) |
| *Reference: co-visitation mean alone (no LLM)* | *0.6066* | *−0.022* | – |

**Did the LLM use the data?** Mean Spearman correlation per user between the LLM's ranking and each signal, and the overlap of the top 10:

| Prompt | with Hybrid E rank | with co-visitation | with popularity | top-10 overlap with Hybrid E | top-10 overlap with co-visitation |
|---|---|---|---|---|---|
| p2 | 0.37 | 0.07 | −0.08 | 5.3 / 10 | 3.8 / 10 |
| p5 | 0.61 | **0.78** | 0.66 | 6.6 / 10 | **7.8 / 10** |
| p6 | **0.79** | 0.70 | 0.56 | **8.1 / 10** | 7.2 / 10 |

**Findings:**
- **The LLM reads and follows the collaborative data.** Without it (p2) its order is almost unrelated to co-visitation (0.07); with it (p5) the order follows co-visitation closely (0.78), and when Hybrid E's rank is shown (p6) it mostly copies that rank (0.79, 8 of 10 top films shared).
- **Following the data brings the LLM up to Hybrid E, not past it.** p5 and p6 are within 0.003 of Hybrid E (both n.s.), and the gain over p2 (+0.005) is not significant.
- **The LLM adds something to the raw co-visitation numbers** (0.6259 vs 0.6066 for ranking by co-visitation alone), but this only recovers what Hybrid E already gets by mixing co-visitation-like item-CF with popularity.
- **Conclusion for B:** the gap between the LLM and Hybrid E is not only missing data. Given the same collaborative evidence, the prompted LLM becomes an approximate copy of collaborative filtering: it reaches Hybrid E's level but does not combine the data with its film knowledge into a better ranking.
- **Not tested here:** fusing p5 with Hybrid E (method A on p5), and other models with p5. Method C (few-shot) and E (frontier model) can use p5 as the base prompt.

### 7.3 Method C: few-shot examples (2026-10-25, `llm_rerank_experiment.py`, prompts p7/p8)

**Setup:**
- **Model and setting:** as in B (DeepSeek V4.1 Flash, no thinking, Hybrid E's top 30, 10 seeds, 203 validation users). Built on p5, so the model sees the same collaborative data.
- **Examples:** 3 warm users per prompt, built exactly like the cold users (seeds = their first 10 liked films in time order; outcome = their later liked films). Warm users only: no overlap with the cold users, and the target user's hidden ratings never appear.
- **Each example shows:**
  - the 10 films the warm user enjoyed first;
  - 10 films from their own Hybrid E top 30 (a fixed random pick, **not chosen by outcome**), each with the same data line as the real candidates;
  - whether they liked each film later ("LIKED" / "not liked or not watched").
- **p7:** the same 3 random warm users (seed 42) for every user.
- **p8:** the 3 warm users most similar to this user (cosine between the user's seeds and the warm user's liked films).
  - A raw count of shared seeds was tried first and rejected before any model call: it picks heavy raters, who liked 83% of their shown candidates vs 54% for real new users. Cosine gives 43%.
- **Cost:** about 5,000 input tokens per user (p5: 2,900), about 1.5 s per answer, no unparsable answers.

**Results (NDCG@10, 203 validation users):**

| Prompt | NDCG@10 | vs Hybrid E (0.6286) | vs p5 (0.6259) |
|---|---|---|---|
| p5 (CF data, no examples) | 0.6259 | −0.003 (p = 0.96) | – |
| p7 (+ 3 fixed examples) | 0.6231 | −0.005 (p = 0.85) | −0.003 (better 82, worse 81, p = 0.88) |
| p8 (+ 3 similar-user examples) | 0.6221 | −0.007 (p = 0.62) | −0.004 (better 69, worse 90, p = 0.12) |

**How closely the ranking follows the signals** (mean Spearman per user; top-10 overlap):

| Prompt | with Hybrid E rank | with co-visitation | with popularity | top-10 overlap with co-visitation |
|---|---|---|---|---|
| p5 | 0.61 | 0.78 | 0.66 | 7.8 / 10 |
| p7 | 0.63 | 0.82 | 0.70 | 8.2 / 10 |
| p8 | 0.64 | 0.79 | 0.66 | 7.9 / 10 |

**Findings:**
- **Few-shot examples do not help.** Both variants are slightly below p5 (n.s.) and below Hybrid E (n.s.).
- **The examples push the model even further towards the data:** p7 follows co-visitation and popularity a little more than p5 (0.82 vs 0.78), without a better result.
- **Examples from similar users (p8) are not better than random ones (p7).** Showing what similar people liked is user-based collaborative filtering in text form, and the LLM does not turn it into a better ranking.
- **Conclusion for C:** showing the model how the dataset's users behave does not close the gap. Together with B, the LLM's ranking ends up close to the collaborative signal it is given, at Hybrid E's level or slightly below.

### 7.4 Method E: frontier models (2026-10-25, `llm_rerank_experiment.py`)

**Setup:**
- **Models:** two large models on Ollama Pro not tested before:
  - **Kimi K3** (Moonshot, a new family);
  - **GLM-5.3** (the full GLM flagship; only GLM-5.2 and 5.3-Flash were tested before).
- **Setting:** idea 1 (Hybrid E's top 30), 10 seeds, validation users, prompts p2 (no CF data) and p5 (CF data). DeepSeek V4.1 Flash from cache as the reference.
- **Reasoning:**
  - Kimi K3 answered with about 100 output tokens (no reasoning), about 2.4 s per answer.
  - **GLM-5.3 ignores "think: off"** and writes its reasoning into the answer: about 4,500 output tokens on p2 and 17,000 on p5, 36 s and 140 s per user. Its answers still parse (0 unparsable), but it must be reported as a reasoning model, and a full p5 run takes about 8 hours.
- No Qwen flagship is reachable on the current accounts (the largest reachable Qwen is Qwen3.8-27B on Groq; only qwen3.5:9b was tested, doc 13).

**Kimi K3, all 203 validation users:**

| Prompt | DeepSeek Flash | Kimi K3 | Kimi vs Hybrid E (0.6286) | Kimi vs DeepSeek |
|---|---|---|---|---|
| p2 | 0.6212 | **0.6274** | −0.001 (p = 0.86) | +0.006 (better 93, worse 86, p = 0.36) |
| p5 | 0.6259 | 0.6207 | −0.008 (p = 0.73) | −0.005 (p = 0.47) |
| p5 − p2 | +0.005 (p = 0.30) | −0.007 (p = 0.92) | | |

**GLM-5.3, 20-user pilot only** (Hybrid E on these users: 0.5395):

| Prompt | DeepSeek Flash | Kimi K3 | GLM-5.3 |
|---|---|---|---|
| p2 | 0.5262 | 0.5328 | 0.5219 |
| p5 | 0.5229 | 0.5244 | **0.5518** (+0.012 vs Hybrid E, p = 0.71; vs DeepSeek +0.029, better 11, worse 5, p = 0.088) |

**Findings:**
- **Kimi K3 ties Hybrid E on p2** (0.6274, the best LLM-alone validation score so far, −0.001) and is not significantly better than DeepSeek Flash. A stronger, newer model family does not beat collaborative filtering.
- **The collaborative data does not help Kimi** (p5 − p2: −0.007). Only DeepSeek showed a (non-significant) gain from it.
- **GLM-5.3 with p5 is the only open lead:** on 20 users it is above Hybrid E and DeepSeek, but not significantly, and 20-user pilots have moved by ±0.02 before (DeepSeek p5: −0.017 on 20 users, −0.003 on 203).
  - Its p5 gain (+0.030 over its own p2) comes together with very long reasoning, so model and reasoning cannot be separated. A full run should be paired with **DeepSeek Flash p5 with thinking on**, as the control for reasoning.
- **Conclusion for E (so far):** the frontier non-reasoning model tested (Kimi K3) reaches Hybrid E's level but does not exceed it, with or without collaborative data. Open: GLM-5.3 p5 on all 203 users, and the reasoning control.

### 7.5 Kimi K3 follow-ups (2026-10-26, `kimi_experiments.py`, `output/experiments/kimi_k3/`)

Four experiments with Kimi K3 (no thinking), 203 validation users. The fusion uses cached answers; the other three are new runs (about 2 s per answer, no unparsable answers).

**1. Fusion with Hybrid E (method A with Kimi; p2, 10 seeds; Hybrid E 0.6286):**

| | Kimi K3 | DeepSeek Flash |
|---|---|---|
| alone | 0.6274 | 0.6212 |
| rank blend, w = 0.4 (fixed from method A) | **0.6408** (+0.012, p = 0.32) | 0.6395 (+0.011, p = 0.33) |
| reciprocal rank fusion | 0.6389 | 0.6382 |
| rank blend, w by 5-fold CV | 0.6325 (w per fold 0.35–0.55) | 0.6395 (w = 0.4 in every fold) |

- Kimi fusion − DeepSeek fusion (w = 0.4): +0.001 (p = 0.46).
- No fusion is significantly above Hybrid E (Bonferroni 1.0).
- With a weight chosen by cross-validation, Kimi's fusion falls to 0.6325, because its best weight is unstable across folds. The w = 0.4 figure borrows DeepSeek's weight and is optimistic.

**2. Idea 3, reranking popularity's top 30 (no rating data; p2, 10 seeds; popularity 0.5626):**

| Model | NDCG@10 | vs popularity | p (Bonferroni over 4) |
|---|---|---|---|
| DeepSeek Flash + thinking | **0.6063** | +0.044 | 0.008 |
| **Kimi K3** | 0.5977 | +0.035 | **0.044** |
| DeepSeek Flash | 0.5942 | +0.032 | 0.057 |
| gemma4:e4b | 0.5591 | −0.004 | 1.0 |

- Kimi − DeepSeek: +0.004 (p = 0.55).
- Kimi − DeepSeek with thinking: −0.009 (p = 0.20).

**3. Candidate order (Kimi p2, 10 seeds, the same 30 candidates shown in 3 random orders):**

| | Kimi K3 | DeepSeek Flash |
|---|---|---|
| order 0 (the original run) | 0.6274 | 0.6212 |
| order 1 | 0.6193 | 0.6218 |
| order 2 | 0.6154 | 0.6127 |
| mean over the 3 orders | **0.6207** | 0.6186 |
| spread between orders | 0.012 | 0.009 |
| rank averaged over the 3 orders | 0.6217 (vs Hybrid E −0.007, p = 0.36) | 0.6183 (−0.010, p = 0.25) |
| top-10 films shared between two orders | 8.3 / 10 | 7.6 / 10 |

**4. Three seeds (Kimi p2, Hybrid E's top 30 for 3 seeds; Hybrid E 0.5837, popularity 0.5626):**

| Model | NDCG@10 | vs Hybrid E | vs popularity |
|---|---|---|---|
| Kimi K3 | 0.5586 | −0.025 (p = 0.10) | −0.004 (n.s.) |
| gemma4:e4b | 0.5476 | −0.036 (p = 0.01) | −0.015 (n.s.) |

**Findings:**
- **Kimi's tie with Hybrid E was partly luck of the candidate order.** Its first run (0.6274) was its best of three orders. The mean over orders is 0.6207, about 0.008 below Hybrid E (n.s.), the same as DeepSeek (0.6186). The thesis should report the 3-order mean, not the best order.
- **Kimi is slightly more stable than DeepSeek** (8.3 vs 7.6 shared top-10 films between orders), but the spread between orders (0.012) is as large as the differences between models. Single-order comparisons below about 0.01 are not meaningful (as in doc 13 §6.3).
- **Fusion does not separate the models.** Fused with Hybrid E, Kimi and DeepSeek give the same result (0.6408 vs 0.6395), still not significantly above Hybrid E.
- **In idea 3, Kimi works without rating data** (+0.035 over popularity, significant after correction), at DeepSeek's level. Without thinking it does not reach DeepSeek with thinking. Kimi + thinking in idea 3 is the remaining open run (2–3 h).
- **With 3 seeds Kimi falls clearly below Hybrid E** (−0.025) and is no better than popularity, the same pattern as for the other models (doc 13 §3.2): with few seeds the LLM has too little to go on.
- **Conclusion:** a frontier model without reasoning does not change any conclusion of methods A–E.

### 7.6 Two more models, and the LLM with user-kNN (2026-10-26, `llm_new_models_pilot.py`)

All on the validation users, 10 seeds, no thinking. Every answer is cached in `output/experiments/llm_rerank/`. Test users untouched.

**1. Two models not tested before: 20-user pilot** (idea 1, Hybrid E's top 30, p2 and p5; Hybrid E on these users 0.5395). DeepSeek Flash, Kimi K3 and GLM-5.3 come from the cache, on the same users.

| Model | p2 | p5 | Seconds per answer | Output tokens |
|---|---|---|---|---|
| DeepSeek V4.1 Flash, 17 Oct snapshot (`deepseek-v4.1-flash:1017`) | 0.5377 | 0.5351 | 0.9 | 94 |
| MiniMax M3 (`minimax-m3`, a new family) | 0.5172 | 0.5278 | 1.9–2.3 | 65 |
| DeepSeek V4.1 Flash (the snapshot used so far) | 0.5262 | 0.5229 | 0.8–1.0 | 94 |
| Kimi K3 | 0.5328 | 0.5244 | 3.7–4.2 | 101 |
| GLM-5.3 (reasons) | 0.5219 | **0.5518** | 36 / 140 | 4,500 / 17,000 |

- **MiniMax M3** was below Hybrid E with both prompts (−0.022 / −0.012) and needed repairs in 3 of 20 p2 answers. It answered without reasoning (65 output tokens). Dropped.
- **DeepSeek `1017`** was closest to Hybrid E (−0.002 / −0.004) and +0.012 above the old snapshot on both prompts (p ≈ 0.5). It went to a full run.

**2. DeepSeek `1017`, all 203 validation users** (Hybrid E 0.6286, popularity 0.5626):

| Model | p2 | p5 |
|---|---|---|
| DeepSeek `1017` | 0.6233 (−0.005, better 92 / worse 100, p = 0.54) | 0.6248 (−0.004, 100 / 89, p = 0.95) |
| DeepSeek, old snapshot | 0.6212 | 0.6259 |
| Kimi K3 | 0.6274 | 0.6207 |

- **New vs old snapshot:** +0.002 (p = 0.95) on p2, −0.001 (p = 0.80) on p5. They are the same model for this task.
- p5 − p2: +0.0015 (p = 0.73). All three are significantly above popularity (Bonferroni p ≤ 0.013). 2 of 203 p2 answers needed repairs, none unparsable.
- **The pilot's +0.012 over the old snapshot disappeared on 203 users.** The best of a 20-user pilot was chosen, so its lead was partly luck (as with DeepSeek p5 and NCF at 3 seeds, doc 20 §6.3). Pilots only choose what gets a full run.

**3. Method H: the LLM reranks user-kNN's top 30 instead of Hybrid E's** (p5, 203 users).

User-kNN (doc 20) is the strongest method without a learned ranker (0.6417). The same reranking code was used with a third candidate source, `candidates="uknn"`. Its top 10 reproduce option 11's user-kNN score exactly (0.6417), and its answers are cached separately (`_cuknn`). The two candidate lists share 23.3 of 30 films on average. As in every run, the candidates are shown in a fixed random order, so the LLM sees user-kNN's films but not its order.

| LLM reranks user-kNN's top 30 | NDCG@10 | vs user-kNN (0.6417) | vs Hybrid E (0.6286) | Same LLM on Hybrid E's top 30 |
|---|---|---|---|---|
| DeepSeek V4.1 Flash | 0.6314 | −0.010 (86 / 101, p = 0.23) | +0.003 (p = 0.68) | 0.6259 |
| Kimi K3 | 0.6220 | −0.020 (82 / 105, p = 0.055; Bonf. 0.11) | −0.007 (p = 0.74) | 0.6207 |

- **The LLM does not improve user-kNN.** Both are below it (not significant after correction; the 20-user pilot pointed the same way, −0.028 / −0.042).
- **The candidate list barely matters to the LLM:** user-kNN's candidates instead of Hybrid E's gives +0.005 (DeepSeek, p = 0.92) and +0.001 (Kimi, p = 0.41). The LLM's own ordering stays at 0.62–0.63 whatever list it gets.
- DeepSeek on user-kNN's list (0.6314) is the highest LLM-alone score on 203 users so far, but only +0.003 over Hybrid E (n.s.).

**4. Method A with user-kNN: fusing user-kNN's order with the LLM ranking** (`python llm_new_models_pilot.py blend`; cached answers, no calls). The same three fusions, the same 5-fold cross-validation over users and the same three controls as §7.1, with user-kNN in place of Hybrid E (w = 1 is user-kNN alone). Checks in the code: w = 1 gives 0.6417, w = 0 gives each LLM's score above.

| Order fused with user-kNN | Alone | Rank blend (CV) | Score blend (CV) | RRF |
|---|---|---|---|---|
| DeepSeek V4.1 Flash p5 | 0.6314 | 0.6427 | **0.6489** | 0.6435 |
| Kimi K3 p5 | 0.6220 | 0.6363 | 0.6452 | 0.6395 |
| *Control: random (10 orders)* | *0.5478* | *0.6405* | *0.6420* | *0.6099* |
| *Control: popularity* | *0.5889* | *0.6417* | *0.6445* | *0.6245* |
| *Control: content similarity (`emb_max`, no LLM)* | *0.6175* | *0.6506* | ***0.6548*** | *0.6501* |

- Weights chosen by CV (score blend): 0.7–0.8 on user-kNN for both LLMs, 0.45 with content similarity (which gets more weight than either LLM).
- **Best LLM fusion vs user-kNN:** DeepSeek score blend +0.007 (better 93 / worse 68, p = 0.051, Bonferroni over 15: 0.76). Kimi: +0.003 (p = 0.24). None of the 15 fusions is significantly above user-kNN after correction.
- **Content similarity fuses best** (+0.013 vs user-kNN, p = 0.32), better than either LLM. A random order in a score blend already gives 0.6420, so part of every blend's small gain is tie-breaking.
- **The same pattern as §7.1:** whatever the LLM adds to collaborative filtering, a plain content embedding adds at least as well.

**Findings of 7.6:**
- **Two more models (one of them a new family, MiniMax) do not change the picture.** A newer DeepSeek snapshot equals the old one; MiniMax M3 is weaker.
- **On 203 validation users, every LLM-alone reranking so far is within 0.008 of Hybrid E** (0.6207–0.6314 across Hybrid E's and user-kNN's lists), none significantly different. Not significant ≠ equal: an equivalence claim would need a confidence-interval test (not done).
- **The LLM does not improve the stronger user-kNN list, alone or fused.** Its contribution in fusion is no larger than that of a content embedding.

**Statements for the thesis** (validation users, 10 seeds):
- "Reranking user-kNN's top 30 with an LLM did not improve on user-kNN (DeepSeek −0.010, Kimi −0.020; neither significant after Bonferroni)."
- "Across candidate lists, prompts and models, the LLM's own ordering reached 0.62–0.63 NDCG@10; better candidates added at most +0.005."
- "Fusing an LLM ranking into user-kNN's order raised NDCG@10 by at most 0.007 (n.s.); a content-similarity order fused the same way did at least as well, so the gain is not LLM-specific."

**Files** (`output/experiments/`):
- `llm_rerank/new_models_pilot_20users_n10.csv`, `new_models_full_203users_n10_p2_p5.csv`;
- `llm_rerank/uknn_rerank_{20,203}users_n10_p5.csv`, `compare_summary_val_{20,203}users_p5_n10_cuknn.csv`;
- `llm_fusion/*_uknn_203users_p5_n10.csv` (per user, summary, tests, weight curves).

**Reproduce:**
```
python llm_new_models_pilot.py check            # no calls: prompt sizes, user-kNN = 0.6417
python llm_new_models_pilot.py pilot            # 2 new models, 20 users, p2 + p5
python llm_new_models_pilot.py pilot full       # DeepSeek 1017, 203 users
python llm_new_models_pilot.py uknn full        # LLM reranks user-kNN's top 30, 203 users, p5
python llm_new_models_pilot.py blend            # fusion with user-kNN, cached answers only
```
