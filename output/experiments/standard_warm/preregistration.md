# Pre-registration: standard warm-start benchmark (doc 35)

**Status:** FROZEN 2026-10-29, before any model was scored on the outer held-out ratings. Tuning on the inner split (§3) is allowed before freezing; nothing in this file changes after the first outer run.
**Scope change (2026-10-29, before freezing):** NCF and LTR v2 are left out (cost and time). The methods are the fast methods plus the two LLM rerankers on 300 users.
**Script:** `standard_warm_benchmark.py`.
**Outputs:** `output/experiments/standard_warm/`.
**Related:**
- doc 18 §7b ("optional robustness check");
- doc 23 (history-length benchmark, liked films only, new users);
- docs 04–08 (random 75/25 split, content methods only).

## 1. Question

Do the thesis's conclusions hold for the **standard warm-start task**? In this task the system knows each user's earlier ratings (likes and dislikes) and must predict the films they will like later.

| | Cold start (main) | Doc 23 (history curve) | **This benchmark** |
|---|---|---|---|
| Users | held-out new users | held-out new users | warm users, in the training data |
| History | 3 or 10 liked films | 3–50 liked films | all earlier ratings, incl. dislikes |
| Answer key | later liked films | later liked films | liked films in the held-out ratings |
| Models see the user while training | no | no | **yes** (standard) |

## 2. Data and splits

- **Users:** the 9,591 warm users only (`cs.split_users`, seed 42).
  - All 1,065 cold users are excluded.
  - So the 203 validation and 814 test users stay untouched for the final test (doc 18 §9).
- **Primary split, T (time):**
  - each warm user's ratings are sorted by time (ties by movie id);
  - the first ⌈0.8·n⌉ ratings are training and the rest (the last ~20%) are held out.
  - So the held-out ratings are never earlier than the same user's training ratings, as in a real service.
- **Secondary split, R (random):** a random 75/25 split of all warm-user ratings (`python_random_split`, seed 42), as in docs 04–08, for continuity with those numbers.
- **Liked (input):** a training rating ≥ the user's mean training rating.
- **Relevant (answer key):** a held-out rating ≥ the user's mean **training** rating, for a pool film.
- **Pool (candidates):** every film with at least one training rating in that split.
- **Evaluated users:** warm users with ≥ 5 relevant held-out films and ≥ 10 liked training films.
- **Never recommended:** every film the user rated in training.

## 3. Tuning (no outer held-out ratings used)

- **Inner split:** for each split (T and R), the training part is split again the same way:
  - T: last 20% of each user's training ratings;
  - R: random 25%, seed 43.
  - The same eligibility rules (§2) are applied on the inner split.
- **Chosen on the inner split** (by mean NDCG@10 over all inner evaluated users; a tie goes to the first value in the grid):

  | Setting | Grid |
  |---|---|
  | Hybrid E α | `cs.ALPHAS` (doc 06) |
  | User-kNN k | {10, 20, 50, 100, 200, 400, 1000} (doc 23) |
  | EASE λ | `ease_tuning.LAMBDAS` (500 … 200,000) |
  | SVD | factors {50, 100, 200} × regularisation {0.02, 0.05}; 20 epochs, learning rate 0.005, seed 42 |

- **Then:** every model is refitted on the full training part and evaluated once on the held-out part.

## 4. Methods

| Method | Warm-start version |
|---|---|
| Popularity | count of liked training ratings (`bmt.popularity_ranking`) |
| Content (embedding + labels, max) | the doc 05 embedding table; max similarity to the user's liked training films |
| Item-CF | item-CF table from the training ratings; mean similarity to the user's liked training films |
| Hybrid E | item-CF (mean) + popularity, both rescaled over the user's candidates; α tuned |
| User-kNN | the user's own mean-centred training ratings; cosine to every other user (the user excluded); score = similarity-weighted sum of centred ratings over the k most similar users |
| EASE | fitted on all training likes; the user's row = their liked training films |
| SVD | Surprise SVD trained on all training ratings, including the user's (the user's own factor, no fold-in); score = predicted rating |
| LLM rerankers: DeepSeek V4.1 Flash p2, Kimi K3 p2 | see below |

**LLM rerankers:**
- They rerank Hybrid E's top 30, with the same p2 prompt text as docs 13–23, and no thinking.
- The prompt lists the user's **50 most recent liked training films** (all of them if fewer), in time order, as in doc 23 at S = 50.
- They run on a **fixed random sample of 300 evaluated users** (`np.random.default_rng(42)`, from the sorted user ids), **split T only** (cost).
- They have their own answer cache (`output/experiments/standard_warm/llm/`).
- The first 20 sample users are a pilot (format and cost check). The pilot answers are kept and count in the 300.

## 5. Questions and decision rules (split T decides; split R is reported alongside)

- **Q1 (main): does the cold-start order of the fast methods hold?**
  - Mean NDCG@10 of every method.
  - Paired Wilcoxon of each method vs Hybrid E, Bonferroni over 6 comparisons (popularity, content, item-CF, user-kNN, EASE, SVD).
  - 95% user-bootstrap CI (1,000 resamples, seed 42) of each difference.
  - "Holds" is reported per pair as in the cold-start board (better / not different / worse at p_Bonf < 0.05).
- **Q2: which method is best for warm users?** The highest mean NDCG@10. It is called "best" only if it is significantly above every other method (Bonferroni over those comparisons).
- **Q3: SVD with the user in training vs with fold-in (doc 23).** Descriptive only, since the users and answer keys differ.
- **Q4 (LLM, 300 users): do the LLM rerankers stay within ±0.02 NDCG@10 of Hybrid E, as in docs 13–23?**
  - Paired Wilcoxon on the 300 users, Bonferroni over 2.
  - 95% bootstrap CI of the difference.
  - "Within ±0.02" if the whole CI lies inside [−0.02, +0.02].
- **Metrics:**
  - NDCG@10 (decides);
  - Precision@10;
  - Recall@10, with its ceiling (mean of min(10, |relevant|) / |relevant|) reported;
  - coverage (share of pool films recommended to at least one user).

## 6. Limits stated in advance

- **MovieLens time order is rating order, not viewing order.** Users often rate many films in one session.
- **The p2 prompt was written for new users** ("A new user told us…"). It is used unchanged; the warm version is an adaptation, not a re-design.
- **The LLMs are run on 300 users only.**
- **NCF and LTR v2 are not in this benchmark.** Their warm-start rank is unknown.
- **Settings are tuned on the inner split.** The outer held-out ratings are used once.
