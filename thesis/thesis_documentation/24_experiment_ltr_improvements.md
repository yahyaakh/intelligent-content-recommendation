# 24 — Four attempts to improve LTR v2 (ideas A, C, E, F)

Status (2026-10-27): done with 10 seeds, validation + held-out simulated users. **No idea adopted; LTR v2 stays the final model.**
Script: `ltr_improvements_experiment.py`; outputs in `output/experiments/ltr_improvements/`.

## 1. Question

LTR v2 (doc 16) reranks Hybrid E's top 30 (C30) using 25 features (R7). It is trained on cross-fitted
simulated warm users, as a tuned XGBoost + LightGBM ensemble. Doc 20 §10.1 showed that adding NCF and
user-kNN *scores as features* does not help. This experiment tries four other ideas from the improvement
list. Each is a change on top of LTR v2:

| Idea | What changes |
|---|---|
| **A**, user-level features (9) | Features that describe the user as a whole: mean and spread of the seeds' log popularity and year; how similar the seeds are to each other (item-CF and content embeddings); genre spread of the seeds; candidate minus seed mean for popularity and year. The idea: the trees could learn "this user likes niche films, trust popularity less". |
| **E**, seed ratings (4) | The stars the user gave each seed: their mean and spread, item-CF similarity weighted by `rating − min + 0.5`, and similarity to the highest-rated seeds. **A different information setting**: the model gets ratings, not only "liked". The no-training check is Hybrid E-w: Hybrid E with the weighted item-CF mean, same α. |
| **C**, wider candidates | C30 plus the 20 best user-kNN films outside it (50 per user; doc 20 L5). **C** = R7 on the wider list; **C+** = R7 + user-kNN score and rank, so the trees can see why a film was added. |
| **F**, blend | `(1 − w) × LTR v2 rank + w × user-kNN rank` on C30, with w from {0, 0.1, …, 1}. For the held-out users w is chosen by 2-fold cross-fitting; for the validation users it is chosen on all held-out users. |

## 2. Set-up (the L1 harness of doc 20 §10.1, unchanged)

- **Cross-fitting:** a simulated user of half A gets item-CF, popularity, the latent tables and user-kNN
  neighbours from half B only, and the other way round. Validation users get tables from all warm users.
  User-kNN uses k = 200 for all warm users, scaled to the size of a half.
- **Two evaluations:**
  1. 1,816 held-out simulated users (train on the other 80%; doc 16 split, seed 42);
  2. the 203 validation users (train on all 9,084 simulated users).
- **Models:** tuned XGBoost, tuned LightGBM (step4_chosen_params.json, not retuned) and their 50/50 rank ensemble.
- **Checks (all passed):**
  - on validation, LTR v2 = 0.6686, Hybrid E = 0.6286 and user-kNN = 0.6417, exactly as in docs 16 and 20;
  - the C30 rows of the wider lists reproduce the cached LTR v2 features;
  - cf_mean and log_popularity recomputed in this script equal the cached tables, so each user got the right half's tables.
- **Tests:** each idea's ensemble vs LTR v2, paired Wilcoxon, Bonferroni over 5 within each user group.
- **Decision rule (fixed before running, as in L1):** an idea is adopted only if it gains at least +0.005
  on **both** user groups.
- The test users were not touched. No LLM was called. The run took about 20 minutes.

## 3. Results (10 seeds, NDCG@10, ensemble)

| Method | Held-out simulated (1,816) | Δ vs LTR v2 | p (Bonf.) | Validation (203) | Δ vs LTR v2 | p (Bonf.) |
|---|---|---|---|---|---|---|
| **LTR v2 (R7, C30)** | 0.6759 | — | — | 0.6686 | — | — |
| A: + user-level features | 0.6762 | +0.0003 | 1.00 | 0.6737 | +0.0051 | 1.00 |
| E: + seed ratings | 0.6804 | +0.0045 | 0.11 | 0.6725 | +0.0039 | 1.00 |
| C: C30 + 20 user-kNN films | 0.6837 | +0.0078 | 0.0005 | 0.6655 | −0.0031 | 0.25 |
| C+: C, + kNN features | **0.6862** | **+0.0103** | **< 0.0001** | 0.6718 | +0.0032 | 1.00 |
| F: blend with user-kNN | 0.6767 | +0.0008 | 0.84 | 0.6677 | −0.0009 | 1.00 |
| *Hybrid E* | 0.6283 | | | 0.6286 | | |
| *Hybrid E-w (seed ratings)* | 0.6357 | | | 0.6344 | | |
| *User-kNN* | 0.6449 | | | 0.6417 | | |
| *Ceiling, C30* | 0.9592 | | | 0.9534 | | |
| *Ceiling, C30 + 20 user-kNN* | 0.9869 | | | 0.9864 | | |

(The ceiling is the NDCG reached if every relevant candidate were ranked first. Single-library results are in `summary_n10.csv`.)

**Decision:** no idea gains +0.005 on both groups. A: +0.0003 / +0.0051. E: +0.0045 / +0.0039.
C: +0.0078 / −0.0031. C+: +0.0103 / +0.0032. F: +0.0008 / −0.0009.

**Hybrid E-w vs Hybrid E** (one separate test): +0.0074 on the held-out users (683 better vs 530 worse,
p < 0.0001) and +0.0059 on validation (p 0.25).

**Blend weight (F):** cross-fitting chose w = 0.1 and 0.2, and all held-out users chose w = 0.1. On
validation, any w > 0 lowers NDCG; w = 1, i.e. user-kNN alone on C30, gives 0.6383.

## 4. Findings

1. **User-level features (A) add nothing on the simulated users** (+0.0003). The +0.005 on validation
   comes with 83 users better vs 79 worse (p 0.41): it is noise, not a signal that only shows up on cold users.
2. **Seed ratings (E) give a small, consistent gain below the threshold.** It is +0.0045 and +0.0039,
   pointing the same way on both groups but significant on neither after correction. For the simple
   method the gain is clearer: Hybrid E-w beats Hybrid E by +0.007 on 1,816 held-out users
   (p < 0.0001), with the same sign on validation (+0.006, n.s.). Knowing *how much* a user liked each
   seed helps a little. Because this needs a different input (ratings), it can only be reported as a
   separate setting, like SVD in doc 20.
3. **Wider candidates (C, C+) help the simulated users but not the validation users.** C+ is the
   largest and only strongly significant gain on held-out users (+0.010, p < 0.0001), but on validation
   it is +0.003 (93 better vs 89 worse). Without the kNN features (C) it is negative on validation.
   The extra 20 films raise the ceiling from 0.95 to 0.99, so they do contain relevant films, but the
   ranker can't reliably move the right ones into the top 10 for the cold users. This repeats doc 16
   step 3 (C30+20 by content similarity: positive on average, helped fewer users than it hurt) and the
   pattern of doc 20 §10.1 (gains on simulated users that do not carry over to validation users).
4. **Blending in user-kNN (F) does not help a ranker that has already seen the same signals.** The
   chosen weight is small (0.1), and on validation every weight above 0 is worse. This matches
   doc 20 §10.1, where user-kNN as a feature didn't help either.
5. **Overall:** LTR v2 stays the final model. Four cheap ideas, each tested on 1,816 + 203 users with a
   rule fixed in advance, gave no gain that holds on both groups.

## 5. Thesis statements (safe to write)

- "Four extensions of the learned ranker (user-level features, seed ratings, wider candidate lists
  with user-kNN, and a score blend with user-kNN) were tested on 1,816 held-out simulated users and
  203 validation users. None reached the pre-set adoption threshold (+0.005 NDCG@10 on both groups);
  LTR v2 remained the final model."
- "Wider candidate lists raised the attainable NDCG@10 from 0.95 to 0.99, and on the simulated users
  the ranker gained +0.010 (p < 0.0001), but on the validation users the gain was +0.003 (n.s.)."
- "Weighting item-CF by the user's seed ratings improved Hybrid E by +0.007 on the held-out simulated
  users (p < 0.0001); this uses rating information that the other methods are not given."
- Do **not** write that user-level features or wider candidates improve the model for new users.

## 6. Limitations

- 10 seeds only. A 3-seed run (`python ltr_improvements_experiment.py 3`, about 20 minutes) was not done.
- The decision rule needs both groups. With 203 validation users the per-user differences vary a lot
  (standard deviation 0.056 for A, 0.063 for E, 0.094 for C+), so only gains of roughly +0.011 to +0.019
  would be detectable there (80% power, 5% level, rough estimate). The held-out group is the more precise one.
- Feature choices (the 9 user features, the rating weight `rating − min + 0.5`, 20 extra films) were set
  once, before running, and not tuned.

## 7. Reproduce

```
python ltr_improvements_experiment.py           # 10 seeds (about 20 min; tables cached afterwards)
python ltr_improvements_experiment.py 3         # 3 seeds
python ltr_improvements_experiment.py 10 dry    # code test on 40 + 300 users (results meaningless)
```

Files: `features_{val,simulated}_{c30,union}_n10.csv`, `baselines_n10.csv`, `summary_n10.csv`,
`tests_n10.csv`, `decision_n10.csv`, `blend_weights_n10.csv`, `blend_curve_n10.csv`,
`feature_shift_n10.csv`, `per_user_{holdout,val}_n10.csv`, `run_n10.log`.
