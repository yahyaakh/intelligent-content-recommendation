# Experiment 20: NCF, SVD and the recomm.py Recommenders for New Users

**Status:** validation done (2026-10-26); L1 + L2 (NCF and user-kNN as LTR features, 10 seeds) done: not adopted (§10.1); pre-registered NCF test written, not yet committed or run
**Scripts:**
- `cold_start_baselines.py`: the methods;
- `final_comparison.py`: CLI option 11;
- `ncf_experiments.py`: pilot, seed study and test;
- `leaderboard.py`: CLI option 12.

**Results folders:** `output/experiments/final_comparison/`, `output/experiments/ncf_cold_start/`
**Saved models:** `output/experiments/final_comparison/cold_start_models/`. These are trained on the warm users only:
- `svd_warm.pkl`;
- `ncf_warm.npz` (seed 42), `ncf_warm_seed43.npz`, `ncf_warm_seed44.npz`.

**Related documents:** 06 (cold start, Hybrid E), 16 (LTR v2), 18 (plan, section 4)

---

## 1. Question

The CLI's original recommenders (`recomm.py`) are content-based, user-based CF, SVD, NCF (NeuMF) and a hybrid. They were only evaluated in option 10, a warm-start test. **How good are they for new users, in the same controlled setup as every other method of the thesis?** In particular:
- can NCF and SVD, the two learned collaborative models, compete with item-CF, Hybrid E and LTR v2?
- is the NCF result robust to chance?

## 2. Why option 10's numbers cannot be used for this

Option 10 (`evaluate_movie_models`) splits **all** ratings of all 10,656 users at random, 75/25. On that split:

| Method | NDCG@10 | Precision@10 | Recall@10 |
|---|---|---|---|
| NCF | 0.556 | 0.532 | 0.064 |
| Popularity | 0.495 | 0.468 | 0.051 |
| SVD | 0.334 | 0.298 | 0.034 |
| Hybrid | 0.159 | 0.147 | 0.030 |
| Collaborative | 0.147 | 0.141 | 0.028 |
| Content | 0.057 | 0.047 | 0.005 |

This is a **different question**: every user has about 600 training ratings, and NCF and SVD learned each user's vector during training.

It also cannot be reused for new users:
- **The models saw the cold users' ratings.** Option 10's SVD and NCF were trained on 75% of everyone's ratings, including the 1,017 users that the cold-start experiments hold out. Using them would leak the answer key.
- **Its popularity score uses synthetic columns** (`view_count`, `completion_rate`).
- **Its content method starts from one favourite film.**

So all five methods were **rebuilt for new users**, trained on the 9,591 warm users only.

## 3. Setup (identical to docs 06–17)

| | |
|---|---|
| Training data | ratings of the 9,591 warm users only (2,292,363 ratings) |
| Evaluation | 203 validation users; the 814 test users only in a pre-registered test |
| Input per new user | the first 3 or 10 liked films in time order (seeds) |
| Answer key | their other liked films (rating ≥ their own mean), the same for 3 and 10 seeds |
| Candidates | the 1,549 pool films minus all 10 seed films |
| Metric | NDCG@10 (precision, recall); paired Wilcoxon vs Hybrid E, Bonferroni within each seed count (15 comparisons at 10 seeds, 10 at 3) |

## 4. Methods

| Method | How it scores films for a new user | Settings |
|---|---|---|
| **Content** (option 10) | 0.75 × CountVectorizer(metadata) + 0.25 × TF-IDF(text) cosine; similarity to the closest seed (max) | as recomm.py |
| **Collaborative** (option 10) | cosine between mean-centred rating vectors; the 5 most similar warm users; score = **mean** over the neighbours who liked the film of (centred rating × similarity). The new user's vector is +1 on each seed. | 5 neighbours, as recomm.py |
| **User-kNN** (standard, new) | the textbook version of the same idea: score = **sum** over the k most similar warm users of similarity × centred rating | k = 200, chosen on validation (§5) |
| **SVD** | Surprise SVD as recomm.py (100 factors, 15 epochs, seed 42) trained on the warm users. **Fold-in:** item biases and factors frozen; the user's bias and factors are the exact minimum of SVD's own loss on the seeds (ridge, λ = 0.02 × number of seeds). SVD gets the seeds' **actual ratings**: more information than any other method. | – |
| **NCF** | NeuMF as recomm.py (4 GMF factors, MLP [16, 8, 4], 30 epochs, batch 256, lr 0.001, every rating = a positive), trained on the warm users (seed 42, 38 minutes). **Fold-in:** every weight and item embedding frozen; only the user's 12 embedding numbers (GMF 4 + MLP 8) are learned. The seeds are positives, with 4 random negatives each (the training ratio), Adam, binary cross-entropy. | steps and learning rate per seed count, chosen on validation (§5) |
| **Hybrid** (option 10) | the Collaborative top 20, rescored 0.4 × SVD rating (0.5–5 → 0–1) + 0.6 × Collaborative rank; popularity fills gaps | as recomm.py |

Every setting that was chosen favours SVD and NCF: SVD gets more input, and NCF's fold-in was tuned on the evaluation users. Conclusions of the form "X beats NCF/SVD" are therefore conservative.

## 5. Settings chosen on the validation users

These are chosen on the same 203 users that are reported, like Hybrid E's α in doc 06. So the chosen rows are slightly optimistic.

**NCF fold-in (Adam steps / learning rate):**

| Seeds | 300 / 0.01 | 1,000 / 0.01 | 300 / 0.05 | 1,000 / 0.05 | 2,000 / 0.05 | 300 / 0.1 |
|---|---|---|---|---|---|---|
| 10 | 0.5703 | 0.5782 | 0.5860 | **0.5980** | 0.5969 | 0.5812 |
| 3 | **0.5537** | – | 0.5448 | 0.5310 | – | – |

- With 10 seeds, more fitting helps up to a plateau.
- With 3 seeds, more fitting overfits the three films.
- Chosen: 1,000 / 0.05 for 10 seeds, 300 / 0.01 for 3 seeds.

**User-kNN neighbours k:**

| k | 10 | 20 | 50 | 100 | **200** | 400 | 1,000 |
|---|---|---|---|---|---|---|---|
| 10 seeds | 0.5738 | 0.6098 | 0.6363 | 0.6345 | **0.6417** | 0.6377 | 0.6302 |
| 3 seeds | 0.4962 | 0.5364 | 0.5792 | 0.6027 | **0.6053** | 0.6013 | 0.5966 |

The curve is flat between 50 and 400, so the choice is not a lucky single value.

**Option 10's Collaborative with more neighbours** (10 seeds, a check, not a choice): 5 → 0.3191, 20 → 0.1659, 50 → 0.0691, 200 → 0.0207. It gets **worse** with more neighbours, which shows the problem is its formula, not k (§7, finding 4).

## 6. Results (203 validation users)

### 6.1 10 seeds

| Method | NDCG@10 | Precision | Recall | vs Hybrid E | Better / worse users | p (Bonf.) |
|---|---|---|---|---|---|---|
| LTR v2 (doc 16) | **0.6686** | 0.6399 | 0.0628 | +0.0401 | 109 / 80 | **0.009** |
| **User-kNN (standard)** | **0.6417** | 0.6217 | 0.0613 | +0.0132 | 104 / 84 | 1.0 |
| LLM fusion, DeepSeek p2 (doc 17) | 0.6395 | 0.6177 | 0.0607 | +0.0110 | 98 / 89 | 1.0 |
| **Hybrid E** (reference) | 0.6286 | 0.6153 | 0.0596 | – | – | – |
| LLM: Kimi K3 p2 (doc 17) | 0.6274 | 0.6133 | 0.0599 | −0.0012 | 98 / 95 | 1.0 |
| Item-CF | 0.5990 | 0.5798 | 0.0572 | −0.0296 | 81 / 110 | 0.018 |
| **NCF (fold-in)** | **0.5980** | 0.5793 | 0.0581 | −0.0305 | 81 / 118 | 0.25 |
| Popularity | 0.5626 | 0.5458 | 0.0495 | −0.0660 | 73 / 119 | < 0.001 |
| **SVD (fold-in)** | 0.5236 | 0.5015 | 0.0469 | −0.1049 | 62 / 137 | < 0.001 |
| Content (embedding + labels, doc 05) | 0.4004 | 0.3704 | 0.0358 | −0.2281 | 36 / 165 | < 0.001 |
| Hybrid (option 10) | 0.3580 | 0.3586 | 0.0335 | −0.2706 | 36 / 164 | < 0.001 |
| Content (option 10) | 0.3211 | 0.2833 | 0.0277 | −0.3075 | 29 / 173 | < 0.001 |
| Collaborative (option 10) | 0.3191 | 0.3177 | 0.0299 | −0.3094 | 31 / 170 | < 0.001 |

### 6.2 3 seeds

| Method | NDCG@10 | Precision | Recall | vs Hybrid E | Better / worse users | p (Bonf.) |
|---|---|---|---|---|---|---|
| LTR v2 | **0.6227** | 0.6025 | 0.0560 | +0.0389 | 120 / 63 | < 0.001 |
| **User-kNN (standard)** | **0.6053** | 0.5931 | 0.0550 | +0.0215 | 117 / 73 | **0.031** |
| **Hybrid E** (reference) | 0.5837 | 0.5734 | 0.0512 | – | – | – |
| Popularity | 0.5626 | 0.5458 | 0.0495 | −0.0212 | 91 / 96 | 1.0 |
| **NCF (fold-in)** | 0.5537 | 0.5360 | 0.0483 | −0.0300 | 87 / 109 | 0.31 |
| **SVD (fold-in)** | 0.5323 | 0.5167 | 0.0471 | −0.0514 | 82 / 113 | 0.017 |
| Item-CF | 0.5132 | 0.5025 | 0.0454 | −0.0706 | 75 / 119 | < 0.001 |
| Content (embedding + labels) | 0.3458 | 0.3207 | 0.0292 | −0.2379 | 37 / 162 | < 0.001 |
| Hybrid (option 10) | 0.3301 | 0.3276 | 0.0314 | −0.2537 | 35 / 167 | < 0.001 |
| Collaborative (option 10) | 0.2773 | 0.2700 | 0.0247 | −0.3064 | 30 / 171 | < 0.001 |
| Content (option 10) | 0.2470 | 0.2118 | 0.0183 | −0.3367 | 24 / 173 | < 0.001 |

### 6.3 NCF robustness

#### 20-user pilot

The first 20 validation users: a subset of the 203, not a new sample.

| Seeds | NCF | Hybrid E | Popularity | NCF − Hybrid E (p) |
|---|---|---|---|---|
| 10 | 0.4954 | 0.5395 | 0.4642 | −0.044 (0.31) |
| 3 | 0.5051 | 0.4280 | 0.4642 | **+0.077 (0.048)** |

At 3 seeds the pilot points the **opposite way** from the 203 users (−0.030). On these 20 users Hybrid E is unusually weak (0.428 vs 0.584 overall, below popularity). This is a direct example of why 20-user pilots only check that the code works and never decide a result.

#### Training and fold-in seeds

NCF was retrained with seeds 43 and 44 (the same data and settings), and model 42 was also folded in with seeds 43 and 44.

| Seeds | Training seed | Fold-in seed | NDCG@10 | vs Hybrid E | p (raw) |
|---|---|---|---|---|---|
| 10 | **42** | 42 | **0.5980** | −0.0305 | 0.016 |
| 10 | 43 | 43 | 0.5888 | −0.0398 | 0.011 |
| 10 | 44 | 44 | 0.5793 | −0.0492 | 0.002 |
| 10 | 42 | 43 | 0.5889 | −0.0396 | 0.003 |
| 10 | 42 | 44 | 0.5975 | −0.0311 | 0.025 |
| 3 | **42** | 42 | **0.5537** | −0.0300 | 0.031 |
| 3 | 43 | 43 | 0.5181 | −0.0656 | 0.002 |
| 3 | 44 | 44 | 0.5355 | −0.0483 | 0.002 |
| 3 | 42 | 43 | 0.5555 | −0.0283 | 0.040 |
| 3 | 42 | 44 | 0.5596 | −0.0241 | 0.073 |

| Seeds | What varies | Mean | Std | Range |
|---|---|---|---|---|
| 10 | training seed (3 models) | **0.5887** | 0.0093 | 0.5793–0.5980 |
| 10 | fold-in seed (model 42) | 0.5948 | 0.0051 | 0.5889–0.5980 |
| 3 | training seed (3 models) | **0.5358** | 0.0178 | 0.5181–0.5537 |
| 3 | fold-in seed (model 42) | 0.5563 | 0.0030 | 0.5537–0.5596 |

- **Model 42 (used in option 11) is the best of the three models** at both seed counts. One reason: the fold-in settings were tuned on it. The fair validation estimate is the 3-model mean: **0.589 ± 0.009** (10 seeds) and **0.536 ± 0.018** (3 seeds).
- **Training randomness matters more than fold-in randomness:** about 2× (10 seeds) and 6× (3 seeds) the spread.
- **The direction is stable:** NCF is below Hybrid E in all 10 runs (raw p < 0.05 in 9). It is above popularity in all 5 runs with 10 seeds and below popularity in all 5 runs with 3 seeds.

## 7. Findings

1. **No option-10 model beats Hybrid E for new users.** The best, NCF, is 0.031 below Hybrid E (model 42) and about 0.040 below on average over three trained models.
2. **NCF ≈ item-CF with 10 seeds, below popularity with 3.**
   - Model 42 ties item-CF (0.5980 vs 0.5990), but the 3-model mean is about 0.01 lower.
   - With 3 seeds NCF beats item-CF (0.554 vs 0.513) but not popularity.
   - A neural model trained on 2.3 M ratings does not add a better new-user signal than a cosine table.
3. **SVD is below popularity at both seed counts,** even though it gets the seeds' actual ratings. Its ranking is dominated by item bias, which favours highly rated niche films. Doc 16 found the same: SVD features mostly repeat item-CF.
4. **Option 10's Collaborative fails because of its formula, not because user-based CF fails.**
   - It averages over the neighbours who liked a film, so one close neighbour's niche film outranks a film all five liked. More neighbours make it worse.
   - **The textbook version (a weighted sum over 200 neighbours) is the second-best method of the thesis:** above Hybrid E at both seed counts, significantly with 3 seeds (+0.022, p_Bonf 0.031), with k chosen on these users.
   - The option-10 Hybrid inherits the weakness.
   - **Write "this implementation fails", never "user-based CF fails".**
5. **Option 10's content model is weaker than the embedding content model** (0.321 vs 0.400 at 10 seeds): the CountVectorizer/TF-IDF tables vs embeddinggemma + LLM labels, as in doc 05.
6. **LTR v2 stays first** (0.6686 / 0.6227), and user-kNN is now the strongest single signal below it.

## 8. Limitations

- **NCF fold-in and user-kNN k were chosen on the reported users,** so their validation numbers are slightly optimistic. Both are fixed before any test run.
- **NCF architecture:** recomm.py's small NeuMF only (4 factors). A larger NCF was not tried. Fold-in is one way to handle new users; others (e.g. the mean embedding of similar warm users) were not tried.
- **Validation only.** No option-10 method or user-kNN has been scored on the test users.
- **NCF's training randomness:** a single model's number can be off by about ±0.01–0.02. Report the 3-model mean.

## 9. Pre-registered test (written, not run)

`output/experiments/ncf_cold_start/preregistration.md` fixes the following.
- **Model:** model 42, checked by its SHA-256 fingerprint.
- **Fold-in settings:** as in §5.
- **Four two-sided hypotheses** (Bonferroni over 4):
  - NCF vs Hybrid E at 10 and at 3 seeds;
  - NCF vs popularity at 10 and at 3 seeds.
- **Sanity checks before scoring.**

`python ncf_experiments.py test` refuses to run until that file is committed unchanged, and runs only once. **Before committing:** add the seed-study results of §6.3 as "known before the test", so the expected differences become the 3-model means (−0.040 at 10 seeds, −0.048 at 3) rather than model 42's.

## 10. Next: NCF and user-kNN inside the learned ranker (XGBoost + LightGBM)

### What the results above suggest

- **NCF alone is weak, but that does not decide whether it helps LTR v2.** A ranker can use a weak signal if it carries information the others lack. Doc 16 is the guide:
  - SVD features (G5) mostly repeated item-CF (each +0.001 to +0.004 or negative);
  - co-visitation added about +0.010, because it measures something different ("people who liked X also liked Y").
- **NCF may repeat item-CF and SVD.** It is a collaborative model trained on the same ratings, and it ties item-CF alone.
- **User-kNN is the most promising new feature.** No LTR v2 feature is user-based: G1–G5 all compare candidates with the seeds film by film. User-kNN compares the new user with whole warm users, and alone it already beats Hybrid E.
- **The candidate list is not the bottleneck:** oracle@30 is 0.953–0.965 (doc 16). Better reranking matters more than more candidates.

### Proposed experiments (validation users and held-out simulated users; L1 + L2 at 10 seeds run 2026-10-26, results in §10.1)

| # | Experiment | Why | Cost |
|---|---|---|---|
| L1 | **G6 features**, added to R7: `ncf_score`, `ncf_rank` (within the 30), `uknn_score`, `uknn_rank`. Models: R8a = R7 + NCF, R8b = R7 + user-kNN, R8 = R7 + both. The tuned XGBoost + LightGBM ensemble with the step-4 parameters. | does NCF or user-kNN add information beyond G1–G5? | see the leakage row |
| L1-leak | **Cross-fitting, as in doc 16:** the simulated warm users of half A get NCF and user-kNN from **half B only**, and vice versa. NCF must be trained twice on half the warm users (about 20 min each). Validation and test users use the full warm model (model 42) and all warm users as neighbours. | a warm user's own ratings must never be in the model that scores them | 2 × 20 min training; NCF fold-in for 9,084 users × 2 seed counts (about 1–2 h) |
| L2 | **Ablation and importance:** gain and SHAP share of the G6 features in R8, and "each feature alone added to R7" (as doc 16 §3.2) | shows which signal is used, and whether NCF is redundant with SVD/item-CF (correlation of `ncf_score` with `cf_mean`, `svd_score`) | minutes |
| L3 | **NCF averaged over 3 models** as the feature (`ncf_score` = mean of models 42/43/44) | the seed study shows a single NCF model is noisy; averaging may make the feature more useful | 4 more half-trainings (about 1.5 h) |
| L4 | **Simple fusion baseline:** rank blend of Hybrid E + user-kNN (+ NCF), weight by 5-fold CV on validation | if a blend matches R8, the trees are not needed for this gain; an honest control for L1 | minutes |
| L5 | **Candidates C30 ∪ user-kNN top 20** (oracle first) | only if L1 shows user-kNN is the main gain: films Hybrid E ranks low but kNN likes | about 1 h |
| L6 | **3 seeds:** repeat L1 with the 3-seed tables (`features_*_n3.csv`, fold-in 300 / 0.01) | user-kNN's clearest gain was with 3 seeds | as L1 |

**Code for L1 + L2 (and L6):** `ltr_ncf_knn_experiment.py` (10 seeds) and `ltr_ncf_knn_experiment.py 3`; outputs in `output/experiments/ltr_ncf_knn/`. Written 2026-10-26, not yet run. It differs from the table above in two details, both fixed in advance:
- `uknn_score` is the kNN sum divided by the neighbours' total similarity (the same order within a user, comparable across users);
- the halves use k = 200 scaled to the half's share of users (about 100).

**Decision rule** (fixed before running, as doc 16):
- choose with the 1,816 held-out simulated users, and report the 203 validation users alongside;
- adopt a new feature group only if it improves both, by at least +0.005;
- compare R8 vs R7 with a paired Wilcoxon on validation, Bonferroni over the variants tried.

**The winning model joins the final pre-registered test** (doc 18 §9), compared with LTR v2 (0.6609 on test).

**Expected outcome (stated before running):**
- user-kNN features: small positive (+0.005 to +0.015), because the signal is new;
- NCF features: about 0, because doc 16 found that SVD-type signals repeat item-CF.

A null result for NCF still answers the abstract's question "does NCF + XGBoost/LightGBM beat LTR v2?".

### 10.1 Results of L1 and L2 (10 seeds, 2026-10-26)

**Run:** `python ltr_ncf_knn_experiment.py` (outputs in `output/experiments/ltr_ncf_knn/`).
- Two half-user NCF models were trained for the cross-fitting (`ncf_halfA.npz`, `ncf_halfB.npz` in `cold_start_models/`, seed 42, 30 epochs, about 15–20 min each). They never saw 15 and 7 pool films. No candidate row lacked an NCF score.
- **Check passed:** the R7 ensemble on validation = 0.6686, LTR v2 of doc 16.

**Feature shift between training and validation users:**
- `uknn_score` (normalised) has the same distribution in both groups: mean 0.309 vs 0.309.
- `ncf_score` does not: mean 0.548 on validation (full model 42) vs 0.334 on the simulated users (half models). The trees are trained on the half-model scale and applied to the full-model scale, which could hurt the NCF feature specifically.

**L1: tuned XGBoost + LightGBM ensemble, NDCG@10** (Hybrid E: 0.6283 held-out, 0.6286 validation):

| Features | Held-out simulated users (1,816) | vs R7 | Validation users (203) | vs R7 |
|---|---|---|---|---|
| R7 (LTR v2) | 0.6759 | – | **0.6686** | – |
| R8a = R7 + NCF | 0.6745 | −0.0014 (p = 0.21) | 0.6651 | −0.0035 (p = 0.52) |
| R8b = R7 + user-kNN | 0.6773 | +0.0013 (p = 0.27) | 0.6636 | −0.0051 (p = 0.34) |
| R8 = R7 + both | 0.6780 | +0.0021 (p = 0.22) | 0.6607 | −0.0079 (p = 0.16) |

- p: paired Wilcoxon, raw; Bonferroni over 3 within each group gives 0.48–1.0.
- The single models show the same: on validation every G6 variant of XGBoost (0.6621–0.6633) and LightGBM (0.6616–0.6656) is below its R7 version (0.6637 / 0.6693).

**Decision (rule fixed before the run: ≥ +0.005 on both groups): no feature group adopted.** LTR v2 (R7) stays the final model; the pre-registered test result of doc 16 is unchanged.

**L2: importance and redundancy:**

| | Gain share of G6 | SHAP share of G6 |
|---|---|---|
| XGBoost (R8) | 0.406 | 0.211 |
| LightGBM (R8) | 0.381 | 0.248 |

- The most used G6 feature is `uknn_rank` (gain share 0.25 / 0.22, SHAP share about 0.10 in both models).
- **The models use the new features a lot but gain nothing:** user-kNN replaces signals R7 already has. Mean within-user Spearman correlation of `uknn_score` (validation): 0.74 with `hybrid_score`, 0.61 with `cf_mean`, 0.58 with `covis_max`.
- `ncf_score` overlaps less (0.34–0.39 with `cf_mean`, `hybrid_score`, `covis_max`, `uknn_score`), but NCF is a weak signal on its own (0.598 alone, §6.1).
- **Each feature added alone to R7 with default settings** (descriptive): XGBoost −0.002 to +0.004; LightGBM +0.002 to +0.017 (`ncf_score`). These are measured against default-setting R7 models (0.6670 / 0.6583) that are weaker than the tuned ones. With the tuned ensemble the gains disappear, so they are not reported as results.

**Expected vs found:** user-kNN was expected to give +0.005 to +0.015 and gave +0.001 (held-out) / −0.005 (validation); NCF was expected to give about 0 and did.

**Limitations of L1:**
- The XGBoost / LightGBM settings were tuned for R7 (doc 16 step 4) and not re-tuned for R8. Re-tuning could move R8 slightly; given held-out gains ≤ +0.002 on 1,816 users, a gain of +0.005 is unlikely.
- The NCF feature shift above.
- 10 seeds only; L6 (3 seeds, `python ltr_ncf_knn_experiment.py 3`) not run. L3–L5 not run (L4 partly answered by doc 17 §7.6: blending user-kNN with content similarity gives 0.6548, below LTR v2).

**Statement for the thesis:** "Adding NCF and user-kNN scores as cross-fitted features to the learned ranker did not improve it (held-out simulated users +0.001 to +0.002, validation users −0.004 to −0.008; none significant). The models used the features heavily, but user-kNN largely repeats the item-CF and Hybrid E signal (within-user Spearman 0.74 with the Hybrid E score), and NCF is a weak signal."

## 11. Files

| File | Content |
|---|---|
| `final_comparison/per_user_val.csv`, `summary_val.csv` | option 11: every method, per user and summary |
| `final_comparison/cold_start_models/` | `svd_warm.pkl`, `ncf_warm.npz` (seed 42), `ncf_warm_seed43.npz`, `ncf_warm_seed44.npz` |
| `ncf_cold_start/pilot_val20.csv` | 20-user pilot |
| `ncf_cold_start/seeds_per_user_val.csv`, `seeds_summary_val.csv`, `seeds_spread_val.csv` | seed study |
| `ncf_cold_start/preregistration.md` | the test plan (not yet committed) |
| `output/leaderboard/leaderboard.csv` | every result, by board (option 12) |
| `final_comparison/cold_start_models/ncf_halfA.npz`, `ncf_halfB.npz` | NCF trained on half A / half B of the warm users (cross-fitting for L1) |
| `ltr_ncf_knn/features_{val,simulated}_g6_n10.csv` | R7 + G6 feature tables |
| `ltr_ncf_knn/l1_summary_n10.csv`, `l1_tests_n10.csv`, `l1_decision_n10.csv`, `l1_per_user_{holdout,val}_n10.csv`, `l1_feature_shift_n10.csv` | L1 |
| `ltr_ncf_knn/l2_importance_n10.csv`, `l2_single_feature_n10.csv`, `l2_correlation_n10.csv` | L2 |

## 12. Reproduce

```
python final_comparison.py          # option 11; trains SVD/NCF once if the saved models are missing
python ncf_experiments.py pilot     # seconds
python ncf_experiments.py seeds     # trains seeds 43 and 44 once (about 40 min each)
python leaderboard.py rebuild       # option 12
python ltr_ncf_knn_experiment.py    # L1 + L2, 10 seeds; trains the two half NCF models once (about 40 min)
python ltr_ncf_knn_experiment.py 3  # L6, 3 seeds (not yet run)
```
