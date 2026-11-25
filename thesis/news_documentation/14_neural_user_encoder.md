# 14 — A learned user encoder: best overall ranker, but still not a substitute for popularity for new users

**Status:** done 2026-11-24 (validation only).
- **Code:** `news/nrms_user.py`, modes `check`, `val` (optionally with a seed) and `switch`. Variants: `base`, `pop`, `pop_pos`, `pop_pos_neg8`, `pop_pos_neg8_long`, `pop_pos_neg8_no9`, `pop_pos_neg8_cscale`.
- **Outputs:**
  - `output/news/nrms_user/<variant>/` (`base`: `output/news/nrms_user/`):
    - `epoch<k>.pt` (weights after each epoch);
    - `scores_val.npy`;
    - `info.json` (settings, epochs, popularity and content weights, results by group, run time);
    - paired tests `compare_vs_*_AUC.csv`;
  - `output/news/nrms_user/compare/`: paired tests between consecutive variants (AUC and nDCG@10);
  - `output/news/nrms_user/switch/`: paired tests of the switch;
  - `output/news/nrms_user/seed_check/`: seed check (§3.7): `seeds.csv`, `pairs.csv`, `mean_over_seeds.csv` and paired tests per seed; seed runs in `<variant>/seed<k>/`;
  - runs `nrms_user_*` and `nrms_switch_g0_cumctr__pop_pos_neg8_cscale` in `output/news/results/val/`.
- **Software:** torch 2.11.0+cu128 on an NVIDIA GeForce RTX 4050 Laptop GPU (6 GB). The movie study's neural model ran on torch 2.12 (CPU).

---

## 1. Question

Every model so far uses the user's history in a fixed way: the mean MiniLM vector (M2, Hybrid E) or 5 hand-made similarity features (LightGBM, docs 04 and 11). Published neural news recommenders instead **learn** how to turn a history into a user vector. The best of them in the MIND paper, NRMS, reaches test AUC 0.6776. M6 reached 0.6679 on the leaderboard (`leaderboard_results.md`).

Questions:
1. Does a learned user encoder beat the fixed history features on val?
2. Can it, with popularity added, replace popularity for new users (empty history)? Doc 13 found that no LightGBM ranker could.

This is step 1 of the neural plan: the user side is learned, while the article vectors stay the frozen MiniLM vectors of doc 10. Steps 2 and 3 (learning the news encoder from words, or fine-tuning MiniLM) are not done.

## 2. Set-up

### 2.1 Model

| Part | What it does |
|---|---|
| Article vectors | frozen MiniLM vectors (`output/minilm_title_abstract_entities_all`, 384 numbers each), held on the GPU; row 0 = padding |
| News projection | learned: linear 384 → 256, then tanh. Applied to history articles and candidates alike |
| Self-attention | learned: 8 heads over the user's last 50 clicked articles (`key_padding_mask` hides the padding) |
| Additive pooling | learned: 256 → 128 → 1 per history article, softmax over the history, weighted sum = user vector u |
| Empty history | learned: one vector `empty_user` is the user vector of every impression without history |
| Score | u · c (dot product with the projected candidate vector), plus the variant terms below |

The model has about 395,000 trainable numbers. MiniLM itself (22 million) is not trained.

### 2.2 Training and selection

- **Training data:** the clicks of Nov 9–13 (the `fit` part of train, 1,801,231 impressions), one sample per click: 2,712,612 samples. Clicks in impressions without any non-clicked candidate are dropped; the log shows 100.0% kept (rounded).
- **Each sample:** the click plus `neg` non-clicked articles drawn from the same impression, with a softmax over these 1 + `neg` scores (cross-entropy, the click is the target). Negatives are drawn again in every epoch, with replacement only when an impression has fewer than `neg`.
- **Optimiser:** AdamW, learning rate 0.001, batch 512, seed 42. Dropout 0.2 on the article vectors and in the attention.
- **Epoch choice:** after each epoch, AUC on a fixed sample of 50,000 val impressions (seed 43). The epoch with the best sample AUC is scored on all of val. This sample is part of val, so the epoch choice is one more selection on Nov 14.
- **Scoring:** every val candidate gets u · c (+ terms), evaluated by `mind_eval.evaluate` (official metrics, the usual tie-break).
- **Run time:** about 3 minutes per epoch and 9–17 minutes per variant (`info.json`).

**Checks** (`nrms_user.py check`):
- the batch builder was compared with the raw data on 64 samples: history order, padding, the click in column 0, no clicked article drawn as a negative, the popularity input of each candidate;
- an untrained model gives a loss close to ln 5 = 1.609 and an AUC near 0.5 on the val sample (printed by the check run: 1.6083 and 0.5299; not saved to a file);
- training on 2,000 samples with fixed negatives drives the loss down to about 0.35, so the model can learn;
- on fake data, `score()` gives the same scores as the training forward pass, for every variant.

### 2.3 Variants

Every variant was defined before it was run, and every one is reported below. Each one is a further selection on Nov 14.

| Variant | Change |
|---|---|
| `base` | the model above, 4 negatives, 3 epochs |
| `pop` | + popularity term w[g] · p, one learned weight per history group g (0 / 1–3 / 4–10 / >10). p = log `cum_ctr`, standardised with the fit-split mean −3.461 and sd 0.750. Nov 9 rows have no `cum_ctr` (11.4% of fit slots); they get p = 0, which is equal for every candidate of the impression and has no effect on the softmax |
| `pop_pos` | + a learned vector per history slot (slot 49 = the latest click), added before the self-attention: recency |
| `pop_pos_neg8` | + 8 negatives per click instead of 4 |
| `pop_pos_neg8_long` | as `pop_pos_neg8`, with dropout 0.1 and 5 epochs |
| `pop_pos_neg8_no9` | as `pop_pos_neg8`, trained on Nov 10–13 only (1,608,679 impressions, 2,422,558 clicks). Tests whether the Nov 9 rows without popularity teach `empty_user` to stand in for popularity |
| `pop_pos_neg8_cscale` | as `pop_pos_neg8`, + a learned content weight per history group: score = v[g] · (u · c) + w[g] · p (v starts at 1) |

Finally, the switch `nrms_switch_g0_cumctr__pop_pos_neg8_cscale` uses `cum_ctr` for empty histories and `pop_pos_neg8_cscale` for every other impression. This is the rule doc 12 chose for group 0 (popularity), so no new setting is chosen. Every impression gets exactly the result of its source run (asserted).

## 3. Results

### 3.1 Overall (val, 431,517 impressions)

| Run | AUC | MRR | nDCG@5 | nDCG@10 |
|---|---|---|---|---|
| `single_cum_ctr` (M1 popularity) | 0.6076 | 0.2571 | 0.2692 | 0.3309 |
| `single_content_mean` (M2) | 0.6319 | 0.2928 | 0.3143 | 0.3729 |
| `hybrid_e_content_cumctr` (M3) | 0.6603 | 0.2960 | 0.3201 | 0.3827 |
| `group_switch_auc` (M5) | 0.6676 | 0.3030 | 0.3298 | 0.3917 |
| `lgbm_minilm_cumctr_mono` (M6) | 0.6750 | 0.3047 | 0.3324 | 0.3935 |
| `nrms_user_minilm` (`base`) | 0.6531 | 0.3004 | 0.3256 | 0.3854 |
| `nrms_user_pop` | 0.6786 | 0.3156 | 0.3431 | 0.4033 |
| `nrms_user_pop_pos` | 0.6788 | 0.3151 | 0.3425 | 0.4029 |
| `nrms_user_pop_pos_neg8` | 0.6796 | 0.3167 | 0.3435 | 0.4038 |
| `nrms_user_pop_pos_neg8_long` | 0.6758 | 0.3141 | 0.3407 | 0.4010 |
| `nrms_user_pop_pos_neg8_no9` | 0.6765 | 0.3135 | 0.3406 | 0.4011 |
| `nrms_user_pop_pos_neg8_cscale` | 0.6834 | 0.3192 | 0.3469 | 0.4070 |
| **switch: `cum_ctr` for group 0, `cscale` otherwise** | **0.6840** | **0.3195** | **0.3474** | **0.4074** |

### 3.2 By history group (val)

Impressions per group: 11,393 / 29,744 / 94,697 / 295,683.

| Run | AUC 0 | AUC 1–3 | AUC 4–10 | AUC >10 | nDCG@10 0 | nDCG@10 1–3 | nDCG@10 4–10 | nDCG@10 >10 |
|---|---|---|---|---|---|---|---|---|
| M1 popularity | **0.6049** | 0.6118 | 0.6127 | 0.6056 | **0.3288** | 0.3698 | 0.3584 | 0.3183 |
| M6 LightGBM | 0.6021 | **0.6390** | 0.6661 | 0.6842 | 0.3257 | **0.3968** | 0.4099 | 0.3905 |
| `base` | 0.5526 | 0.5977 | 0.6373 | 0.6676 | 0.2905 | 0.3752 | 0.3948 | 0.3871 |
| `pop` | 0.5894 | 0.6291 | 0.6657 | 0.6911 | 0.3092 | 0.3950 | 0.4146 | 0.4041 |
| `pop_pos` | 0.5889 | 0.6272 | 0.6645 | 0.6920 | 0.3075 | 0.3921 | 0.4131 | 0.4044 |
| `pop_pos_neg8` | 0.5832 | 0.6274 | 0.6654 | 0.6931 | 0.3085 | 0.3904 | 0.4132 | 0.4058 |
| `pop_pos_neg8_long` | 0.5830 | 0.6217 | 0.6597 | 0.6900 | 0.3080 | 0.3873 | 0.4093 | 0.4033 |
| `pop_pos_neg8_no9` | 0.5840 | 0.6238 | 0.6625 | 0.6899 | 0.3067 | 0.3917 | 0.4118 | 0.4022 |
| `pop_pos_neg8_cscale` | 0.5851 | 0.6328 | **0.6678** | **0.6973** | 0.3107 | 0.3931 | **0.4151** | **0.4094** |
| switch | **0.6049** | 0.6328 | **0.6678** | **0.6973** | **0.3288** | 0.3931 | **0.4151** | **0.4094** |

### 3.3 What each change did (paired, all impressions)

Mean difference per impression and its 95% interval (`output/news/nrms_user/compare/`).

| Change | Compared with | ΔAUC | 95% interval | ΔnDCG@10 | 95% interval |
|---|---|---|---|---|---|
| + popularity (`pop`) | `base` | **+0.0255** | +0.0250 to +0.0260 | **+0.0179** | +0.0174 to +0.0184 |
| + recency (`pop_pos`) | `pop` | +0.0002 | −0.0001 to +0.0005 | −0.0004 | −0.0007 to −0.0000 |
| + 8 negatives (`pop_pos_neg8`) | `pop_pos` | +0.0008 | +0.0004 to +0.0013 | +0.0009 | +0.0004 to +0.0014 |
| dropout 0.1, 5 epochs (`_long`) | `pop_pos_neg8` | −0.0038 | −0.0041 to −0.0035 | −0.0028 | −0.0031 to −0.0025 |
| without Nov 9 (`_no9`) | `pop_pos_neg8` | −0.0031 | −0.0035 to −0.0027 | −0.0027 | −0.0032 to −0.0023 |
| + content weight per group (`_cscale`) | `pop_pos_neg8` | **+0.0038** | +0.0035 to +0.0041 | **+0.0031** | +0.0028 to +0.0035 |

- **The learned encoder beats fixed history features.** Without popularity, `base` (0.6531) is above M2, which ranks by cosine with the mean history vector (0.6319), in every group. For users with more than 10 clicks it reaches Hybrid E (0.6676 against 0.6664, doc 09) without any popularity.
- **Popularity is the largest single gain** (+0.026 AUC). The learned popularity weights stay close to 0.25–0.30 in every group and every variant (`info.json`).
- **Recency adds nothing.** The 50 latest clicks are already a recent window. The overall Wilcoxon p is small (9 × 10⁻⁵), but the mean difference is 0.0002 and its interval includes 0.
- **More negatives help slightly** for users with a history (>10: +0.0011), but cost new users −0.0058 (95% interval −0.0074 to −0.0042).
- **Longer training overfits to the training days.** In `_long`, the training loss falls in every epoch (1.9045 → 1.8428), while the val sample AUC peaks at epoch 2 (0.6751) and then drops to 0.6705. Doc 13 found the same kind of day-to-day change for LightGBM.
- **Nov 9 is useful training data.** Leaving it out costs about 0.003 AUC in every group with a history and does not help new users (group 0: +0.0008, interval −0.0013 to +0.0030). The hypothesis behind `_no9` is rejected.
- **A content weight per group is the second largest gain** and improves every group (0: +0.0020; 1–3: +0.0054; 4–10: +0.0024; >10: +0.0042; all intervals above 0).

### 3.4 What the content weights learned

Weights at the chosen epoch (2) of `pop_pos_neg8_cscale`:

| | group 0 | 1–3 | 4–10 | >10 |
|---|---|---|---|---|
| content weight v | 0.8016 | 0.1018 | 0.1793 | 0.2898 |
| popularity weight w | 0.2550 | 0.2699 | 0.2548 | 0.2772 |

- **Groups 1–3, 4–10 and >10 share the user encoder, so their v can be compared.** The weight on content rises with history length (0.10 → 0.18 → 0.29), while the weight on popularity stays flat. The longer the history, the more the ranking leans on content.
- **Hybrid E found the same pattern by grid search on val** (α = 0.3 / 0.4 / 0.5, doc 09). Here the model learned it on the training days. Two methods, one tuned and one learned, agree.
- **The v of group 0 cannot be compared with the others.** It multiplies the separate `empty_user` vector, whose length is also learned.
- **The weights had not settled:** at epoch 3, v was 0.72 / 0.05 / 0.10 / 0.16, and the val sample AUC fell from 0.6828 to 0.6785. With seeds 1 and 2, epoch 3 was chosen, with v = 0.69 / 0.05 / 0.10 / 0.16 and 0.67 / 0.05 / 0.09 / 0.15 (§3.7). The size of v depends on the stopping epoch, but its order over the groups is the same in all three seeds.

### 3.5 New users: no variant reaches popularity

- **For empty histories, every neural variant stays below popularity:** 0.5830–0.5894 against 0.6049. That includes the two variants built for this group.
- **The gap is like M6's, only larger:** M6 is 0.0028 below popularity on group 0, the best neural variant (`pop`) is 0.0155 below.
- **The training days favour a different mix than Nov 14.** On Nov 9–13, a mix of `empty_user` content and a modest popularity weight is the best fit for new users. On Nov 14, popularity alone is better. Neither dropping Nov 9 nor a free content weight changes this. It matches doc 13 (§3.4): what a ranker learns about popularity does not carry over to the next day.

### 3.6 The switch

`cum_ctr` for empty histories and `pop_pos_neg8_cscale` for the rest (`output/news/nrms_user/switch/`):

| Compared with | ΔAUC | 95% interval | ΔnDCG@10 | 95% interval |
|---|---|---|---|---|
| `pop_pos_neg8_cscale` | +0.0005 | +0.0004 to +0.0007 | +0.0005 | +0.0004 to +0.0006 |
| M6 `lgbm_minilm_cumctr_mono` | **+0.0090** | +0.0083 to +0.0097 | **+0.0140** | +0.0133 to +0.0147 |
| M5 `group_switch_auc` | +0.0163 | +0.0156 to +0.0170 | +0.0158 | +0.0151 to +0.0165 |

By group, against M6 (AUC):

| Group | ΔAUC | 95% interval |
|---|---|---|
| 0 | +0.0028 | +0.0002 to +0.0054 |
| 1–3 | **−0.0062** | −0.0092 to −0.0032 |
| 4–10 | +0.0017 | +0.0001 to +0.0032 |
| >10 | +0.0131 | +0.0123 to +0.0139 |

- **The switch is the best val result of the news study,** on all four metrics.
- **The win over M6 comes from users with a history.** With more than 10 history articles the switch gains +0.0131 AUC and +0.0190 nDCG@10 (`compare_vs_lgbm_minilm_cumctr_mono_*.csv`).
- **For 1–3 history articles, M6 is still better.** Against M5, which uses Hybrid E there, the switch's −0.0020 is not significant (interval −0.0051 to +0.0012).

### 3.7 Seed check

`pop_pos_neg8` and `pop_pos_neg8_cscale` were each run with two more seeds (1 and 2). The seed sets the initial weights, the shuffling and the negatives. The epoch-choice sample stays the same (seed 43).

**Rule, fixed before the runs:** the content weight counts as a real gain only if `cscale` beats `pop_pos_neg8` on val AUC for each of the three seeds (42, 1, 2), comparing runs with the same seed.

| Seed | `pop_pos_neg8` AUC | `cscale` AUC | ΔAUC (95% interval) | ΔnDCG@10 (95% interval) | chosen epoch (`pop_pos_neg8` / `cscale`) |
|---|---|---|---|---|---|
| 42 | 0.6796 | 0.6834 | +0.0038 (+0.0035 to +0.0041) | +0.0031 (+0.0028 to +0.0035) | 2 / 2 |
| 1 | 0.6809 | 0.6819 | +0.0010 (+0.0007 to +0.0013) | +0.0009 (+0.0006 to +0.0013) | 3 / 3 |
| 2 | 0.6781 | 0.6825 | +0.0045 (+0.0040 to +0.0049) | +0.0056 (+0.0052 to +0.0061) | 2 / 3 |

Means over the three seeds (`mean_over_seeds.csv`; range in brackets):

| | AUC | nDCG@10 | AUC 0 | AUC 1–3 | AUC 4–10 | AUC >10 |
|---|---|---|---|---|---|---|
| `pop_pos_neg8` | 0.6795 (0.6781–0.6809) | 0.4035 (0.4012–0.4054) | 0.5856 | 0.6258 | 0.6638 | 0.6936 |
| `pop_pos_neg8_cscale` | **0.6826** (0.6819–0.6834) | **0.4067** (0.4063–0.4070) | 0.5858 | **0.6339** | 0.6653 | **0.6968** |

- **The rule is met:** `cscale` is better for every seed, on AUC and on nDCG@10. The mean gain is +0.0031 AUC and +0.0032 nDCG@10.
- **The size of the gain varies with the seed** (+0.0010 to +0.0045 AUC). The seed-42 run (+0.0038) is close to the mean.
- **The seed spread of one model is 0.0015 (`cscale`) to 0.0028 (`pop_pos_neg8`) AUC, from the lowest to the highest seed.** Differences between variants smaller than that (recency +0.0002, 8 negatives +0.0008 in §3.3) cannot be told apart from seed noise with one seed each.
- **Seed 42 was the best of the three `cscale` runs:** the reported 0.6834 is 0.0008 above the mean over seeds.
- **The content weight helps most for 1–3 history articles:** +0.0081 AUC in the mean.
- **The new-user gap holds for every seed:** group 0 AUC is 0.5832–0.5871 in all six runs, against 0.6049 for popularity. The content weight does not change group 0 (−0.0012 to +0.0020 by seed).
- **All six runs are above M6** (0.6750) on val AUC.

## 4. What this means

1. **A learned user encoder is the strongest way found so far to use a history.** With popularity and a content weight per group, it beats every earlier model on val by a clear margin. On nDCG@10 the margin over M6 (+0.014) is larger than any earlier step.
2. **It does not change the answer for new users.** The thesis question was where a new user is best served. Seven neural variants, including two built for new users, all stay below plain popularity for an empty history. The practical system remains a switch: popularity for an empty history, a learned ranker otherwise.
3. **Hybrid E's pattern is confirmed by learning:** the more history a user has, the more weight content gets.
4. **These val numbers are optimistic.** Seven variants, the epoch per variant and the switch rule were all chosen on Nov 14. The seed check (§3.7) shows that the content-weight gain is not seed noise. The seed-42 number is 0.0008 above the mean over seeds, and small variant differences (recency, 8 negatives) are within seed noise. The honest test is the pre-registered final test on dev (Nov 15).

## 5. Next steps

- **Pre-registration (draft):** the switch with `pop_pos_neg8_cscale` is the candidate for M8. It is not yet added: for the final test it needs scoring code for dev (Nov 15; `cum_ctr_dev.npy` exists) and a decision on refitting (Nov 9–14), as for M4 and M6. If it is added, the hypotheses H2 and H5 (which compare M6 with M4 and with the switch) need a matching hypothesis for M8, for example "M8 > M6 overall".
- **Seed check:** done (§3.7). The content weight is kept. For the final test, M8 can be fixed as the seed-42 model, or as the mean of the three seeds' scores; that must be decided in the pre-registration.
- **Leaderboard:** the test week has almost no click history (83.3% of slots at the `cum_ctr` prior, `leaderboard_results.md`). The popularity term will contribute little there, so the content side carries the score, as for M6.
- **Steps 2 and 3 of the neural plan:** learn the news encoder from words (NRMS with GloVe), or fine-tune MiniLM inside the model. These are larger jobs. Step 1 shows that the user side pays off.
