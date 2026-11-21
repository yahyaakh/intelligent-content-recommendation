# Experiment: User Cold Start (New Users with a Few Seed Movies)

**Status:** finished (run date: 2026-10-20, on the corrected half-star ratings)
**Script:** `cold_start_experiment.py`
**Results folder:** `output/experiments/cold_start/`
**Related documents:** `01_llm_movie_enrichment.md`, `03_related_work.md`, `04_experiment_larger_enrichment_model.md`, `05_experiment_scoring_comparison.md`

---

## 1. Questions

All earlier evaluations used warm-start users with about 100 liked training movies each. This experiment evaluates **new users**, who are excluded from training and tell the system only a few movies they like.

1. **Which method recommends best for new users?**
2. **From how many seed movies does personalisation beat popularity?**
3. **Does LLM enrichment help more for new users than for existing users?** The literature reports the clearest benefit of content enrichment in cold-start settings (`03_related_work.md`), and the TDK abstract states this as a hypothesis.

---

## 2. Method

### 2.1 Simulating new users

New users cannot be evaluated directly, because their future preferences are unknown. They are therefore **simulated**: real users with long histories are removed from training, and only their first few liked movies are revealed.

| Step | Setting |
|---|---|
| Data | the same 1,957 movies and 2,547,690 ratings as experiments 04–05 (original half stars, 0.5–5) |
| User split | all 10,656 users sorted by ID, shuffled with seed 42; the first 10% (1,065) become **cold users**, the other 9,591 **warm users** |
| Training data | **all ratings of warm users only** (2,292,363). No cold user's rating is used for training, popularity or similarities. |
| Candidate pool | the 1,549 movies rated by warm users |
| Liked | rating ≥ the user's own mean rating (computed over all their ratings; used only to define the ground truth, never as model input) |
| Seed pool | per cold user, the **first 10 liked movies in time order** (ties broken by movie ID), restricted to the candidate pool. This simulates a user naming a few movies they like when signing up. |
| Answer key | all other liked movies in the candidate pool. **All 10 seed-pool movies are excluded** from the answer key and from the recommendations at every N. |
| Seeds revealed | N = 1, 3, 5, 10: the system is told the first N movies of the seed pool |
| Users kept | 1,017 of 1,065 (38 with fewer than 10 liked movies, 10 with fewer than 5 relevant movies left) |
| Per kept user | about 251 ratings and 128 relevant (answer-key) movies |
| Validation / test | the kept cold users shuffled with seed 42: **203 validation users** (only for tuning the hybrids' weight) and **814 test users** (all reported results) |
| Metrics | NDCG@10, Precision@10, Recall@10; Wilcoxon signed-rank test per user, Bonferroni-corrected over all 32 tests |

**Why the seeds are set aside at every N.** If only the N revealed seeds were excluded, the other 10 − N seed movies (liked early, and therefore often popular) could still be recommended but would always count as misses. That would make every method, including popularity, look better as N grows for a reason unrelated to the method. With all 10 set aside, the candidates and the answer key are identical for every N, and only the information given to the system changes.

**Validity checks.**

- No user appears in both the warm and the cold group; together they contain every rating exactly once.
- The two groups have nearly identical activity (median about 225 ratings per user) and mean rating, as expected from a random split.
- For every kept cold user: 10 distinct, liked, in-pool seeds; the seeds are the earliest liked movies; no seed is in the answer key.
- A hand-made example with known answer reproduced the expected seeds and answer key.
- Popularity gives exactly the same score at every N, which confirms that candidates and answer key do not change with N.
- On the validation users, the hybrid with α = 0 reproduces popularity, and with α = 1 the pure similarity table, to every decimal.

### 2.2 Methods

| # | Method | Uses | Notes |
|---|---|---|---|
| 0 | **Popularity** | number of warm users who liked each movie | same list for everyone (the shared definition from experiment 05) |
| A / B | **Content-based** (CountVectorizer 75% + TF-IDF 25%), without / with LLM labels | seeds + movie metadata | as in experiments 04–05 |
| C / D | **Embedding-based** (embeddinggemma), without / with LLM labels | seeds + movie texts | best embedding model in experiment 05 |
| E | **Item-based collaborative filtering** | seeds + movie-to-movie cosine similarity of mean-centred warm-user ratings | works for a new user without retraining |
| Hybrid C / D / E | **Similarity + popularity** | α × personal similarity + (1 − α) × popularity | see 2.3 |

Methods A–E are evaluated with **mean** and **max** scoring (`05_experiment_scoring_comparison.md`). With N = 1 both scorings are identical.

SVD and NCF learn one vector per training user and cannot score a user who was not in training without retraining or folding-in; they are not included in this experiment (see section 6).

### 2.3 Hybrids

For each candidate movie:

> **score = α × personal + (1 − α) × popularity**

- **personal:** mean similarity of the candidate to the N seeds, from table C, D or E;
- **popularity:** log(1 + number of warm users who liked the movie);
- both are **rescaled to 0..1 over each user's candidates**, so that the two parts are on the same scale;
- **α** is chosen **on the 203 validation users only**, separately for each table and each N, from {0, 0.02, 0.05, 0.08, 0.1, 0.12, 0.15, 0.2, 0.25, 0.3, 0.4, …, 1.0}; on a tie the smaller α is kept.

| Hybrid | Personal part | "Similar" means |
|---|---|---|
| Hybrid C | embeddings without labels | about similar things (story, themes) |
| Hybrid D | embeddings with LLM labels | about similar things, plus the LLM's themes and mood |
| Hybrid E | item-based CF | liked by the same people |

**Control for the enrichment question.** Hybrid C and hybrid D choose their α independently, and on 203 validation users the chosen values can differ by chance. To compare "with vs without labels" fairly, hybrid C is additionally evaluated **at hybrid D's α** ("hybrid C at D's α").

Chosen α:

| Table | N = 1 | N = 3 | N = 5 | N = 10 |
|---|---|---|---|---|
| C (embedding) | 0.05 | 0.05 | 0.15 | 0.20 |
| D (embedding + LLM labels) | 0.05 | 0.12 | 0.15 | 0.20 |
| E (item-CF) | 0.08 | 0.12 | 0.20 | 0.25 |

The weight of the personal part grows with the number of seeds. Because both parts are rescaled per user but have different spreads (popularity varies strongly between candidates, similarities are often close together), α is a tuned setting and should not be read as "the share of personalisation".

---

## 3. Results

### 3.1 NDCG@10 (814 test users)

| Method | N = 1 | N = 3 | N = 5 | N = 10 |
|---|---|---|---|---|
| **Hybrid E** (item-CF + popularity) | **0.5779** | **0.5970** | **0.6081** | **0.6311** |
| Hybrid D (embedding + LLM labels + popularity) | 0.5711 | 0.5920 | 0.5993 | 0.6138 |
| Hybrid C at D's α | 0.5697 | 0.5916 | 0.6001 | 0.6135 |
| Hybrid C (own α) | 0.5697 | 0.5778 | 0.6001 | 0.6135 |
| **Popularity** | 0.5598 | 0.5598 | 0.5598 | 0.5598 |
| E: item-CF (mean) | 0.4503 | 0.5388 | 0.5690 | 0.6166 |
| E: item-CF (max) | 0.4503 | 0.5181 | 0.5440 | 0.5790 |
| D: embedding + LLM labels (mean) | 0.2922 | 0.3820 | 0.4147 | 0.4576 |
| C: embedding (mean) | 0.2922 | 0.3798 | 0.4124 | 0.4540 |
| D: embedding + LLM labels (max) | 0.2922 | 0.3471 | 0.3653 | 0.4051 |
| C: embedding (max) | 0.2922 | 0.3474 | 0.3653 | 0.4041 |
| A: content (max) | 0.1820 | 0.2296 | 0.2553 | 0.3192 |
| B: content + LLM labels (max) | 0.1551 | 0.2187 | 0.2502 | 0.3095 |
| A: content (mean) | 0.1820 | 0.2184 | 0.2338 | 0.2460 |
| B: content + LLM labels (mean) | 0.1551 | 0.2010 | 0.2096 | 0.2230 |

Gain over popularity (NDCG@10):

| Method | N = 1 | N = 3 | N = 5 | N = 10 |
|---|---|---|---|---|
| Hybrid E | +3.2% | +6.6% | +8.6% | +12.7% |
| Hybrid D | +2.0% | +5.8% | +7.1% | +9.6% |
| Hybrid C at D's α | +1.8% | +5.7% | +7.2% | +9.6% |
| Item-CF alone (mean) | −19.6% | −3.8% | +1.6% | +10.1% |

### 3.2 Precision@10 and Recall@10

| Method | P@10 N=1 | P@10 N=10 | R@10 N=1 | R@10 N=10 |
|---|---|---|---|---|
| Hybrid E | 0.5654 | 0.6155 | 0.0522 | 0.0579 |
| Hybrid D | 0.5604 | 0.5956 | 0.0509 | 0.0560 |
| Hybrid C at D's α | 0.5590 | 0.5951 | 0.0507 | 0.0556 |
| Popularity | 0.5477 | 0.5477 | 0.0490 | 0.0490 |
| Item-CF (mean) | 0.4364 | 0.6015 | 0.0405 | 0.0570 |
| Embedding + LLM labels (mean) | 0.2725 | 0.4440 | 0.0242 | 0.0417 |
| Embedding (mean) | 0.2730 | 0.4382 | 0.0243 | 0.0413 |
| Content (max) | 0.1650 | 0.2771 | 0.0144 | 0.0254 |
| Content + LLM labels (max) | 0.1407 | 0.2654 | 0.0121 | 0.0242 |

Precision and recall follow the same ranking as NDCG. Recall is low for every method because each test user has about 128 relevant movies and only 10 are recommended.

### 3.3 Significance tests (Wilcoxon signed-rank, NDCG@10 per user, Bonferroni over 32 tests)

| Comparison (b vs a) | N = 1 | N = 3 | N = 5 | N = 10 |
|---|---|---|---|---|
| Hybrid E vs popularity | **+0.018** (p = 0.002) | **+0.037** (p < 10⁻⁸) | **+0.048** (p < 10⁻¹⁰) | **+0.071** (p < 10⁻¹⁹) |
| Hybrid D vs popularity | **+0.011** (p = 0.007) | **+0.032** (p < 10⁻⁷) | **+0.040** (p < 10⁻¹⁰) | **+0.054** (p < 10⁻¹⁷) |
| Hybrid C vs popularity (own α) | +0.010 (p = 0.056) | **+0.018** (p < 10⁻⁷) | **+0.040** (p < 10⁻¹¹) | **+0.054** (p < 10⁻¹⁷) |
| Item-CF (mean) vs popularity | **−0.109** (p < 10⁻¹⁸) | −0.021 (p = 1.0) | +0.009 (p = 1.0) | **+0.057** (p < 10⁻⁸) |
| **Hybrid D vs hybrid C at the same α** (LLM labels) | +0.001 (p = 1.0) | +0.000 (p = 1.0) | −0.001 (p = 1.0) | +0.000 (p = 1.0) |
| Embedding D vs C (mean; LLM labels) | −0.000 (p = 1.0) | +0.002 (p = 1.0) | +0.002 (p = 1.0) | +0.004 (p = 1.0) |
| Content B vs A (max; LLM labels) | **−0.027** (p < 10⁻⁸) | −0.011 (p = 0.28) | −0.005 (p = 1.0) | −0.010 (p = 0.094) |
| Hybrid E vs hybrid D | +0.007 (p = 1.0) | +0.005 (p = 1.0) | +0.009 (p = 1.0) | +0.017 (p = 0.10) |

Bold: significant after Bonferroni correction (p < 0.05). Uncorrected p-values are in `significance_tests.csv`; for hybrid D vs C at the same α they are 0.34–0.63 at every N.

---

## 4. Findings

1. **Popularity is a hard baseline for new users.** No pure similarity method beats it with fewer than about 10 seed movies. Content-based (0.18–0.32) and embedding-based recommendation (0.29–0.46) stay far below popularity (0.56) even with 10 seeds.
2. **Item-based CF is by far the best pure personal method** (0.45 → 0.62), clearly ahead of embeddings and content. It crosses popularity between 5 and 10 seeds; the difference is significant at N = 10 (+10.1%), not at N = 3 or 5. Its neighbours are films the same audience likes, so it carries part of popularity's quality signal (e.g. *The Matrix* → *Fight Club*, *The Shawshank Redemption*, *The Lord of the Rings*).
3. **Combining a few seeds with popularity beats popularity from the first seed.** Hybrid E improves NDCG@10 by +3.2% with one seed and +12.7% with ten, significant at every N; the embedding hybrids by +2% to +10%. The answer to question 2 is therefore: **pure personalisation needs about 10 seeds; personalisation on top of popularity helps from 1 seed.**
4. **The collaborative signal is the most useful personal signal.** Hybrid E is the best method at every N. Its lead over hybrid D (+0.005 to +0.017) is consistent but not significant after correction.
5. **LLM enrichment does not help new users.** With the same α, hybrid D and hybrid C differ by at most 0.0014 NDCG@10 (uncorrected p = 0.34–0.63 at every N). Pure embeddings with and without labels differ by at most 0.004 (not significant). For content-based recommendation the labels **reduce** quality at every N (−0.027 with one seed, significant). This **contradicts the hypothesis** that enrichment matters most when the user history is short. Hybrid D's apparent lead over hybrid C at N = 3 with their own α (0.592 vs 0.578) is caused by the different chosen α (0.12 vs 0.05), not by the labels.
6. **The best aggregation depends on the length of the history.** For new users, **mean** scoring beats max for embeddings and item-CF (e.g. embeddings with 10 seeds: 0.454 vs 0.404), the reverse of the warm-start result, where max scoring was better for all 19 cases (experiment 05). With a few seeds, max scoring follows a single seed's nearest neighbours; averaging gives a more balanced picture of the user's taste. For content-based recommendation, max remains better.
7. **The personal part earns more weight as the number of seeds grows** (chosen α 0.05–0.08 at N = 1, 0.20–0.25 at N = 10).

**Robustness.** The same experiment on the earlier, rounded ratings gave the same conclusions: hybrid E +2.5% (N = 1) to +10.4% (N = 10) over popularity, hybrid D vs C within ±0.0006, and content-based recommendation worse with labels.

---

## 5. Consequences for the thesis

- **Answer to the enrichment question across all experiments:** LLM-generated themes and moods improve ranking accuracy only marginally for existing users (about +1% NDCG@10 for the best embedding model under max scoring, with a direction that depends on the embedding model; experiment 05) and **not at all for new users** (this experiment). A plausible reason is that the labels restate information that is already present in the overview and keywords used by the content and embedding methods, and that ranking accuracy on MovieLens is dominated by popularity and collaborative signals.
- **Recommended cold-start strategy for the system:** item-based CF combined with popularity, with the personal weight increasing with the number of movies the user has rated.
- **Where enrichment may still matter** (not yet tested): typed natural-language requests (moods and themes are close to how people describe what they want), movies with poor metadata, explanations, and beyond-accuracy metrics (diversity, novelty, coverage).

---

## 6. Limitations

- **Simulated new users.** Cold users are heavy raters (about 251 ratings each) whose first liked movies are revealed. Real new users may name different, possibly less popular, movies.
- **Ground truth is the user's later history,** which on MovieLens is itself biased towards popular films. Offline evaluation is known to overrate popularity (`03_related_work.md`); the strength of the popularity baseline here should be read with this in mind.
- **Not comparable with warm-start numbers.** Cold users have about 128 relevant movies each (everything except the seeds), warm users about 33, which raises precision and NDCG.
- **Mixed data sources.** 943 users come from MovieLens 100K (rated 1997–98, movies mapped to TMDB by title), the rest from MovieLens 25M. The two groups behave differently; the random split mixes both into validation and test.
- **One split** (seed 42) and **α tuned on 203 validation users,** so the chosen α is somewhat noisy (see finding 5).
- **SVD and NCF are not included,** because they cannot score a new user without retraining. SVD with folding-in (fitting only the new user's vector from the seeds) would be a fair matrix-factorisation baseline.
- **Typed preferences are not yet evaluated**; this experiment uses seed movies only.

---

## 7. Next steps

1. **Typed preferences:** an LLM writes a short request from the user's seed movies; the request is embedded and used for retrieval, compared with seed-based scoring at the same N. This tests preference interpretation and is the most promising place for LLM enrichment.
2. **Unified benchmark** with SVD (folding-in for new users) and NCF, on the same data, split and candidate pool, for warm and cold users.
3. **Learned ranker** (LightGBM with a lambdarank objective) over popularity, similarities, item-CF and LLM-label features, trained on simulated cold users from the warm group; feature importance as a second measure of the contribution of the LLM labels.
4. **Beyond-accuracy metrics:** coverage, novelty and diversity of the recommendation lists.

---

## 8. Files

| File | Content |
|---|---|
| `output/experiments/cold_start/per_user.csv` | NDCG, precision and recall per test user, method, scoring and N (chosen α for the hybrids) |
| `output/experiments/cold_start/summary.csv` | averages per method, scoring and N |
| `output/experiments/cold_start/alpha_validation.csv` | validation NDCG@10 for every α, table and N |
| `output/experiments/cold_start/alphas.csv` | chosen α per table and N |
| `output/experiments/cold_start/significance_tests.csv` | Wilcoxon tests with uncorrected and Bonferroni-corrected p-values |

---

## 9. Reproducing the run

```bash
python cold_start_experiment.py
```

Requirements: `data/data/processed_30k/ratings.csv` with half-star ratings (regenerated by `prepare_movielens25m.py`), `output/enrichment/movies_enriched_v2.csv`, and the indexes `movies_base_embeddinggemma_v2` and `movies_enriched_embeddinggemma_v2` in `output/embeddings/`. The run takes a few minutes; all random choices use seed 42.
