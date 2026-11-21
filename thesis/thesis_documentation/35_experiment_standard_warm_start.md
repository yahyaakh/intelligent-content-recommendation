# 35 — Standard warm-start benchmark (warm users, held-out later ratings)

**Status:** complete (2026-10-29). **Pre-registered** (frozen before any outer run).
**Script:** `standard_warm_benchmark.py` (modes `check`, `fast`, `llm`, `llm full`, `report`).
**Pre-registration:** `output/experiments/standard_warm/preregistration.md`.
**Outputs:** `output/experiments/standard_warm/`:
- `settings.json`, `settings_curves_T.csv`, `settings_curves_R.csv`;
- `per_user_T.csv`, `per_user_R.csv`, `per_user_llm.csv`, `llm_usage.csv`;
- `summary.csv`, `tests.csv`, `coverage.json`;
- `llm/` (answer caches).

**Related:** doc 18 §7b (the plan); doc 23 (new users with 3–50 liked films); doc 06 (Hybrid E); docs 13–17 (LLM reranking); doc 26 (EASE).

## 1. Question

The main evaluation of this work is **cold start**: new users who give a few liked films. This benchmark checks the usual task in the recommender-system literature instead.
- The system knows a user's **earlier ratings, likes and dislikes**.
- The user is part of the training data.
- The system must predict which films the user will like **later**.

| | Cold start (main, docs 06–20) | Doc 23 | **This benchmark** |
|---|---|---|---|
| Users | held-out new users | held-out new users | warm users, in the training data |
| History | 3 or 10 liked films | 3–50 liked films | all earlier ratings (about 90–112 liked films on average) |
| Models trained with the user | no | no | **yes** |

## 2. Set-up

- **Users:** the 9,591 warm users only. The cold users (incl. the 203 validation and 814 test users) are never loaded.
- **Two splits:**

  | Split | How | Training | Held out | Evaluated users | Pool |
  |---|---|---|---|---|---|
  | **T (time, primary)** | each user's first ⌈80%⌉ ratings in time order train, the rest held out | 1,837,727 | 454,636 | **8,697** | 1,509 films |
  | R (random, secondary) | random 75/25 of all warm ratings (seed 42), as docs 04–08 | 1,719,272 | 573,091 | **8,983** | 1,546 films |

- **Liked** = training rating ≥ the user's mean training rating. **Relevant** = held-out rating ≥ that same mean.
- **Evaluated users:** ≥ 5 relevant held-out films and ≥ 10 liked training films.
  - Split T: 24.1 relevant films per user on average; 112 liked training films.
  - Split R: 34.5 relevant films per user on average; 103 liked training films.
- **Films the user rated in training are never recommended.**
- **Tuning:** every setting was chosen on an **inner split** of the training part (T: the last 20% again; R: random 25%, seed 43). The outer held-out ratings were used once.

  | Setting | Split T | Split R | Grid |
  |---|---|---|---|
  | Hybrid E α | 0.7 | 0.6 | doc 06 grid |
  | User-kNN k | 50 | 200 | 10–1000 |
  | EASE λ | **500** (smallest in the grid) | 1,000 | 500–200,000 |
  | SVD factors / regularisation | 50 / 0.02 | 50 / 0.02 | {50,100,200} × {0.02,0.05} |

- **Methods:**
  - Popularity.
  - Content (doc 05 embedding + labels, max over liked films).
  - Item-CF (mean over liked films).
  - Hybrid E.
  - User-kNN (the user's own centred ratings; the user excluded from the neighbours).
  - EASE.
  - SVD (Surprise, trained on all training ratings including the user's).
  - Two LLM rerankers on **300 sampled users of split T**.
- **Not included (decided before running):** NCF and LTR v2.
- **Check:** on both splits, Hybrid E's lists were identical to `cs.recommend_hybrid` for the first 50 users.

## 3. Results

### 3.1 Split T (time, primary): 8,697 users

| Rank | Method | NDCG@10 | Precision@10 | Recall@10 | Coverage | Δ vs Hybrid E [95% CI] | p (Bonf. ×6) |
|---|---|---|---|---|---|---|---|
| 1 | **EASE** | **0.3165** | **0.2847** | **0.1342** | 51.7% | **+0.0580** [+0.055, +0.061] | < 0.001 |
| 2 | Hybrid E (reference) | 0.2585 | 0.2335 | 0.1050 | 42.2% | – | – |
| 3 | Item-CF | 0.2571 | 0.2323 | 0.1051 | 51.4% | −0.0014 [−0.002, −0.000] | 0.098 (n.s.) |
| 4 | User-kNN | 0.2455 | 0.2215 | 0.0997 | 52.6% | −0.0129 [−0.016, −0.010] | < 0.001 |
| 5 | Popularity | 0.1831 | 0.1677 | 0.0731 | 17.2% | −0.0754 [−0.079, −0.072] | < 0.001 |
| 6 | SVD (user in training) | 0.1645 | 0.1465 | 0.0628 | 67.1% | −0.0940 [−0.098, −0.090] | < 0.001 |
| 7 | Content (embedding + labels, max) | 0.1556 | 0.1372 | 0.0602 | 28.2% | −0.1029 [−0.108, −0.098] | < 0.001 |

**EASE is significantly above every other method** (Q2; all p_Bonf < 0.001). For example, it is +0.058 vs Hybrid E and better for 5,290 users vs worse for 2,610.

**Recall@10 ceiling:** the mean of min(10, |relevant|)/|relevant| is **0.534** on split T. So EASE's recall of 0.134 is about a quarter of the best possible.

### 3.2 Split R (random, secondary): 8,983 users

| Rank | Method | NDCG@10 | Precision@10 | Recall@10 | Δ vs Hybrid E [95% CI] |
|---|---|---|---|---|---|
| 1 | **EASE** | **0.7037** | **0.6622** | **0.2208** | **+0.0991** [+0.096, +0.102] |
| 2 | Hybrid E | 0.6046 | 0.5589 | 0.1810 | – |
| 3 | Item-CF | 0.5936 | 0.5466 | 0.1775 | −0.0110 [−0.012, −0.010] |
| 4 | User-kNN | 0.5579 | 0.5140 | 0.1629 | −0.0467 [−0.050, −0.044] |
| 5 | Popularity | 0.4366 | 0.4076 | 0.1260 | −0.1680 [−0.172, −0.164] |
| 6 | SVD (user in training) | 0.3485 | 0.3093 | 0.0959 | −0.2561 [−0.261, −0.252] |
| 7 | Content | 0.3336 | 0.2868 | 0.0917 | −0.2710 [−0.276, −0.266] |

- **The order is the same as on split T.** All differences vs Hybrid E are significant (p_Bonf < 0.001); here item-CF is also significantly below Hybrid E.
- **The numbers are much higher than on split T** (0.70 vs 0.32 for EASE). A random split hides ratings from the middle of a user's history, which films rated in the same sessions predict easily. A time split asks for the future. This is why split T is the primary split.
- **Content at 0.334** is close to the docs 04–05 random-split numbers (about 0.32). The data and answer key differ slightly (warm users only here), so this is a rough match, not a reproduction.

### 3.3 LLM rerankers: 300 sampled users of split T

The LLMs rerank Hybrid E's top 30. The prompt (p2, unchanged) lists the user's 50 most recent liked training films.

| Method (same 300 users) | NDCG@10 | Precision@10 | Recall@10 | Δ vs Hybrid E [95% CI] | Better / worse | p (Bonf. ×2) |
|---|---|---|---|---|---|---|
| EASE | 0.3291 | 0.2980 | 0.1405 | – | – | – |
| Hybrid E (reference) | 0.2558 | 0.2303 | 0.1045 | – | – | – |
| Item-CF | 0.2553 | 0.2280 | 0.1045 | – | – | – |
| User-kNN | 0.2447 | 0.2200 | 0.1001 | – | – | – |
| **LLM: DeepSeek V4.1 Flash, p2** | 0.2422 | 0.2227 | 0.1010 | −0.0136 [−0.032, +0.005] | 117 / 143 | 0.23 (n.s.) |
| **LLM: Kimi K3, p2** | 0.2383 | 0.2193 | 0.0998 | −0.0176 [−0.037, +0.003] | 116 / 149 | 0.15 (n.s.) |
| Popularity | 0.1873 | 0.1673 | 0.0759 | – | – | – |
| SVD | 0.1649 | 0.1440 | 0.0660 | – | – | – |
| Content | 0.1622 | 0.1410 | 0.0616 | – | – | – |

- **Q4 (pre-registered rule):** "within ±0.02" needs the whole CI inside [−0.02, +0.02].
  - Both CIs reach just below −0.02 (−0.032 and −0.037), so **the rule is not met**.
  - Neither LLM is significantly different from Hybrid E either.
  - So the result is **inconclusive at 300 users**. The point estimates (−0.014, −0.018) match the earlier pattern: about equal to Hybrid E, slightly below.
- **Answers:** 600 calls, 0 unparsable, 1 answer with a minor issue (repaired by the parser as in doc 13).
- **Tokens:**
  - DeepSeek: 3,011 input / 94 output per user; 0.90M input in total.
  - Kimi K3: 3,139 / 101 per user; 0.94M in total.
- **Cost:**
  - DeepSeek: about $0.35–0.40 at the doc 13 rate (estimate, not a bill).
  - Kimi K3: Ollama Pro subscription credit.
  - Time: 4.6 min (DeepSeek), 9.7 min (Kimi K3).

## 4. Findings

1. **For warm users, EASE is clearly the best method, on both splits.**
   - It beats Hybrid E by +0.058 (time split) and +0.099 (random split), and every other method significantly.
   - For new users EASE was already second, behind only LTR v2:

     | Validation users | EASE | Hybrid E | LTR v2 |
     |---|---|---|---|
     | 3 seeds | 0.606 | 0.584 | 0.623 |
     | 10 seeds | 0.651 | 0.629 | 0.669 |

     In doc 23 (3–50 liked films) EASE was +0.010 to +0.019 above Hybrid E, not significant.
   - With the full history its lead over Hybrid E grows to +0.058–0.099 and becomes clearly significant. LTR v2 was not run here, so whether EASE also passes LTR v2 for warm users is open.
   - This matches the literature, where EASE is a strong warm-start baseline (Steck 2019).
2. **Below EASE, the cold-start order holds** (Q1):
   - Hybrid E ≈ item-CF > user-kNN > popularity > SVD and content, on both splits.
   - Hybrid E's popularity part adds little here (α = 0.6–0.7, against 0.08–0.25 for new users). With a long history, the personal signal dominates.
3. **SVD is weak at top-10 ranking even when trained with the user** (Q3, descriptive). It is below popularity on both splits, and it recommends the widest range of films (coverage 67%).
   - SVD orders films by predicted rating, which favours rarely rated, highly rated films over films the user is likely to watch and like.
   - So doc 23's weak SVD was not only a fold-in problem.
   - This agrees with doc 34, where SVD predicted ratings well (RMSE about 0.78), and with the literature: good rating prediction does not give good top-N lists.
4. **The content method is last for warm users too.** This matches every earlier board; it remains useful for new films (doc 31), not for ranking known films.
5. **LLM reranking is close to Hybrid E but not shown to be within ±0.02** (Q4; inconclusive at 300 users). The LLMs do not close the gap to EASE: EASE is +0.07 to +0.09 above them on the same 300 users (descriptive).

## 5. Limitations

- **NCF and LTR v2 were not run** (decided before running, for time). Whether LTR v2 would beat EASE for warm users is unknown.
- **EASE's λ on split T is the smallest value in the grid (500).** The inner curve is flat there (0.2651 at 500, 0.2649 at 1,000), so a smaller λ is unlikely to change the conclusion, but the optimum may lie below the grid.
- **MovieLens time order is rating order**, not viewing order. Users often rate many films in one session, so even split T is easier than real future prediction.
- **The LLM prompt was written for new users** ("A new user told us…") and shows only 50 liked films, not dislikes. It was used unchanged, so it is an adaptation, not a warm-start design.
- **The LLMs ran on 300 users** of split T only. The CI is about ±0.02 wide.
- **Hybrid E and item-CF use only liked films**, while EASE uses liked films and SVD/user-kNN use all ratings. That is each method's standard form, not a matched input.
- **Split R is not comparable in level with split T** (§3.2).

## 6. Safe statements for the TDK

- "In a standard warm-start test (8,697 MovieLens users, each user's last 20% of ratings in time order held out), EASE was the best method (NDCG@10 0.317), significantly above Hybrid E (0.259, +0.058, 95% CI [0.055, 0.061]) and every other method. The same order held on a random 75/25 split."
- "Below EASE, the order of the cold-start methods held for warm users: Hybrid E ≈ item-CF > user-kNN > popularity > SVD, content."
- "SVD trained with the user's own ratings was below popularity for top-10 ranking, although it is a strong rating predictor (doc 34)."
- "On 300 warm users, LLM reranking of Hybrid E's top 30 was 0.014–0.018 below Hybrid E, not significantly different."
- Avoid:
  - "EASE is the best method overall" (LTR v2 is above it for new users, and LTR v2 and NCF were not tested here);
  - "the LLMs are equivalent to Hybrid E for warm users" (the pre-registered ±0.02 rule was not met).

## 7. Where this leaves the method boards

| Setting | Best method | Hybrid E | LLM rerank vs Hybrid E |
|---|---|---|---|
| New users, 3 / 10 liked films (docs 06–20) | LTR v2 | reference | about −0.01 |
| New users, 3–50 liked films (doc 23) | LTR v2; EASE rank 2–3 | reference | about −0.01 to +0.01 |
| **Warm users, full history (this doc)** | **EASE** (+0.058 vs Hybrid E); LTR v2 not run | rank 2 | −0.014 / −0.018 (n.s., 300 users) |

(First row: EASE was second for new users too, +0.02 above Hybrid E on the validation users.)
