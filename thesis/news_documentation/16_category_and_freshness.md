# 16. Neural user encoder: category vectors and freshness inputs

**Date:** 2026-11-25 · **Split:** val (Nov 14, 431,517 impressions) · **Code:** `news/nrms_user.py` (variants `pop_pos_neg8_cscale_cat`, `pop_pos_neg8_cscale_fresh`, `pop_pos_neg8_cscale_cat_fresh`) · **Outputs:** `output/news/nrms_user/<variant>/`, `output/news/nrms_user/switch/pop_pos_neg8_cscale_cat/`

## 1. Question

Doc 14's best model (`nrms_user_pop_pos_neg8_cscale`, val AUC 0.6834) sees each article only through its frozen MiniLM vector, and popularity only through `cum_ctr`. Two cheap additions, each one training run on top of that model:

- **cat:** does a learned category and subcategory vector help the article representation?
- **fresh:** do the article's age and recent showings help next to `cum_ctr`? Doc 07 found that fresh articles are clicked more.

## 2. Set-up

Everything is as in doc 14 (`pop_pos_neg8_cscale`, seed 42, 3 epochs, epoch chosen on the fixed 50,000-impression val sample). Only one part changes per variant.

- **cat:** every article gets a learned category vector and a learned subcategory vector (32 numbers each). They are joined to its 384 MiniLM numbers before the projection to 256. There are 18 categories and 293 subcategories, from the train, dev and test news files. Every article in the vector table has both.
- **fresh:** the popularity term gets 4 inputs instead of 1, each standardised with the mean and sd of the fit split:
  - log `cum_ctr` (as before; Nov 9 has none and gets 0);
  - log(1 + age in hours), where age = hours since the article's first showing;
  - log(1 + showings in the last hour);
  - `older_than_log` (0/1: first shown before the log starts; 26.1% of fit slots).

  All of them are known at impression time. The model learns one weight per input and history group (16 weights).

## 3. Results

### 3.1 Overall

| Run | AUC | MRR | nDCG@5 | nDCG@10 | Chosen epoch |
|---|---|---|---|---|---|
| `nrms_user_pop_pos_neg8_cscale` (doc 14) | 0.6834 | 0.3192 | 0.3469 | 0.4070 | 2 |
| **`nrms_user_pop_pos_neg8_cscale_cat`** | **0.6907** | 0.3230 | 0.3522 | **0.4131** | 3 of 3 |
| `nrms_user_pop_pos_neg8_cscale_fresh` | 0.6854 | 0.3208 | 0.3489 | 0.4090 | 2 of 3 |
| `nrms_switch_g0_cumctr__pop_pos_neg8_cscale_cat` | 0.6909 | 0.3233 | 0.3526 | 0.4134 | — |
| for reference: doc 15 blend + switch (3 seeds + M6) | 0.6921 | 0.3197 | 0.3491 | 0.4107 | — |

### 3.2 Paired against `pop_pos_neg8_cscale` (difference, 95% interval)

| Group | n | cat AUC | cat nDCG@10 | fresh AUC | fresh nDCG@10 |
|---|---|---|---|---|---|
| 0 | 11,393 | +0.0111 (0.0074 to 0.0148) | +0.0071 | +0.0059 (0.0042 to 0.0075) | +0.0086 |
| 1–3 | 29,744 | +0.0074 (0.0051 to 0.0098) | +0.0066 | +0.0054 (0.0042 to 0.0065) | +0.0055 |
| 4–10 | 94,697 | +0.0032 (0.0019 to 0.0046) | +0.0034 | +0.0015 (0.0010 to 0.0021) | +0.0026 |
| >10 | 295,683 | +0.0084 (0.0077 to 0.0091) | +0.0070 | +0.0016 (0.0014 to 0.0019) | +0.0012 |
| **all** | 431,517 | **+0.0073 (0.0067 to 0.0078)** | **+0.0062 (0.0056 to 0.0068)** | +0.0020 (0.0018 to 0.0022) | +0.0020 (0.0017 to 0.0023) |

Every interval in the table is above 0. Files: `<variant>/compare_vs_nrms_user_pop_pos_neg8_cscale_{AUC,nDCG10}.csv`.

### 3.3 By group (AUC / nDCG@10)

| Run | 0 | 1–3 | 4–10 | >10 |
|---|---|---|---|---|
| popularity `cum_ctr` | 0.6049 / 0.3288 | | | |
| M6 | 0.6021 / 0.3257 | 0.6390 / 0.3968 | 0.6661 / 0.4099 | 0.6842 / 0.3905 |
| cscale (doc 14) | 0.5851 / 0.3107 | 0.6328 / 0.3931 | 0.6678 / 0.4151 | 0.6973 / 0.4094 |
| cat | 0.5962 / 0.3178 | 0.6402 / 0.3997 | 0.6710 / 0.4185 | 0.7057 / 0.4164 |
| fresh | 0.5910 / 0.3193 | 0.6381 / 0.3986 | 0.6694 / 0.4178 | 0.6990 / 0.4106 |

The cat switch (empty history: `cum_ctr`) beats M6 in every group on AUC, but within noise for 1–3 (+0.0012, interval −0.0020 to +0.0044). Overall it is +0.0160 AUC and +0.0199 nDCG@10 over M6 (`switch/pop_pos_neg8_cscale_cat/`).

### 3.4 Training

- **cat:** the val-sample AUC was still rising at the last epoch (0.6822 → 0.6876 → 0.6902). 3 epochs may be too few for this variant.
- **fresh:** the val-sample AUC peaked at epoch 2 (0.6781 → 0.6847 → 0.6808).
- **Learned fresh weights (epoch 2), in the order `cum_ctr` / age / showings last hour / older than log:**
  - group 0: 0.263 / −0.046 / −0.169 / 0.019;
  - group >10: 0.275 / −0.116 / −0.025 / −0.008.

  `cum_ctr` stays the main popularity input. Newer articles score higher, as doc 07 found. Articles shown very often in the last hour score lower, mostly for new users.

## 4. What it means

- **Category vectors are the biggest single-model gain since popularity was added (doc 14):** +0.0073 AUC and +0.0062 nDCG@10. That is 3 to 5 times the seed spread of doc 14 (0.0015–0.0028), so it is not seed luck, although it is one seed. The MiniLM text already contains the title, abstract and entities, but not the category label. A learned category vector lets the model learn how much each topic is clicked, instead of only how close it is to the history.
- **One cat model is now as good as the doc 15 ensemble:** 0.6909 vs 0.6921 AUC for the switches, and higher on nDCG@10 (0.4134 vs 0.4107). The ensemble used 4 models.
- **The freshness inputs help a little:** +0.0020 AUC, the size of one seed's spread. The gain is largest for short histories (group 0: +0.0059 AUC, +0.0086 nDCG@10), where popularity carries the most weight. Doc 13 found that LightGBM's extra popularity features did not carry to a new day. Here, with linear weights per group, they do, if only slightly.
- **New users:** both variants narrow the gap to popularity for empty histories, but neither closes it (cat 0.5962, fresh 0.5910, popularity 0.6049). The switch is still needed.

## 5. cat + fresh in one model (`pop_pos_neg8_cscale_cat_fresh`)

The two gains come from different parts of the model (article input vs popularity term), so both were put into one model. Everything else is as above. Epoch 3 of 3 was chosen; the val-sample AUC was still rising (0.6840 → 0.6893 → 0.6929).

| Run | AUC | MRR | nDCG@5 | nDCG@10 |
|---|---|---|---|---|
| cat | 0.6907 | 0.3230 | 0.3522 | 0.4131 |
| **cat + fresh** | **0.6937** | 0.3255 | 0.3549 | **0.4159** |
| cat + fresh, switch | 0.6937 | 0.3256 | 0.3551 | 0.4159 |
| doc 15 blend + switch (3 seeds + M6) | 0.6921 | 0.3197 | 0.3491 | 0.4107 |

**Paired (AUC difference, 95% interval; nDCG@10 difference):**

| Group | vs cat | vs fresh | vs cscale (doc 14) | vs popularity `cum_ctr` |
|---|---|---|---|---|
| 0 | +0.0084 (0.0064 to 0.0105); +0.0077 | +0.0136; +0.0062 | +0.0195; +0.0148 | **−0.0002 (−0.0054 to +0.0049)**; −0.0033 (−0.0078 to +0.0012) |
| 1–3 | +0.0056 (0.0043 to 0.0069); +0.0063 | +0.0077; +0.0074 | +0.0130; +0.0129 | +0.0340; +0.0362 |
| 4–10 | +0.0031 (0.0025 to 0.0037); +0.0035 | +0.0048; +0.0042 | +0.0063; +0.0069 | +0.0615; +0.0636 |
| >10 | +0.0025 (0.0022 to 0.0028); +0.0019 | +0.0092; +0.0077 | +0.0109; +0.0089 | +0.1026; +0.1000 |
| **all** | **+0.0030 (0.0027 to 0.0032); +0.0027** | +0.0083; +0.0069 | **+0.0102; +0.0089** | +0.0861; +0.0849 |

Files: `pop_pos_neg8_cscale_cat_fresh/compare_vs_*.csv` and `switch/pop_pos_neg8_cscale_cat_fresh/`. Against M6: +0.0187 AUC, +0.0225 nDCG@10 (as a switch).

**Group 0 by model (AUC):** cscale 0.5851 → fresh 0.5910 → cat 0.5962 → **cat + fresh 0.6046**, against popularity 0.6049.

## 6. What the combined model means

- **The gains add up.** cat + fresh is +0.0030 AUC over cat, about what fresh added to cscale (+0.0020). Over doc 14's model the total is +0.0102 AUC and +0.0089 nDCG@10, in every group.
- **One model now beats the doc 15 ensemble** on every metric (0.6937 vs 0.6921 AUC, 0.4159 vs 0.4107 nDCG@10).
- **First model in the study to match popularity for new users.** For empty histories it is 0.0002 AUC below `cum_ctr`. The 95% interval (−0.0054 to +0.0049) includes 0 but reaches just past the pre-registered ±0.005 margin, as M6 did on val (doc 13, H3). The switch therefore changes almost nothing (+0.0000 AUC overall). The cat vectors give the empty-history user a topic prior, and the freshness inputs give a better popularity prior. Together they reach what popularity alone does.
- **Caveats.** One seed. The model was still improving at its last epoch. The variant was chosen after the cat and fresh results were seen (3 variants in this doc).

## 6a. Seed check (2026-11-26)

cat + fresh was trained again with seeds 1 and 2 (as in doc 14 §3.7: the seed sets the initial weights, the shuffling and the negatives; the epoch-choice sample stays fixed). Each seed is compared with `pop_pos_neg8_cscale` trained with the same seed. Outputs: `output/news/nrms_user/pop_pos_neg8_cscale_cat_fresh/seed{1,2}/`, `output/news/nrms_user/seed_check_cat_fresh/`.

| Seed | cat + fresh AUC / nDCG@10 | cscale AUC / nDCG@10 | AUC difference (95% interval) | nDCG@10 difference (95% interval) | Chosen epoch |
|---|---|---|---|---|---|
| 42 | 0.6937 / 0.4159 | 0.6834 / 0.4070 | +0.0102 (0.0097 to 0.0108) | +0.0089 (0.0083 to 0.0095) | 3 |
| 1 | 0.6973 / 0.4179 | 0.6819 / 0.4063 | +0.0154 (0.0149 to 0.0159) | +0.0116 (0.0110 to 0.0121) | 3 |
| 2 | 0.6940 / 0.4156 | 0.6825 / 0.4068 | +0.0114 (0.0109 to 0.0120) | +0.0088 (0.0082 to 0.0094) | 3 |
| **mean** | **0.6950 / 0.4165** | 0.6826 / 0.4067 | +0.0124 | +0.0098 | |

**Mean over seeds by group (AUC):** 0.6037 / 0.6487 / 0.6772 / 0.7089 (groups 0 / 1–3 / 4–10 / >10), against cscale 0.5858 / 0.6339 / 0.6653 / 0.6968. Every group gains on both metrics (`mean_over_seeds.csv`).

**Empty histories vs popularity (`cum_ctr`, 0.6049), AUC difference (95% interval):**

| Seed | Group-0 AUC | Difference |
|---|---|---|
| 42 | 0.6046 | −0.0002 (−0.0054 to +0.0049) |
| 1 | 0.6059 | +0.0010 (−0.0042 to +0.0062) |
| 2 | 0.6005 | −0.0044 (−0.0095 to +0.0008) |

**What it means:**

- **The gain is not seed luck.** cat + fresh beats cscale for every seed, by +0.0102 to +0.0154 AUC. That is 3 to 10 times the seed range of either model (cat + fresh: 0.6937–0.6973; cscale: 0.6819–0.6834).
- **Seed 42 is the lowest of the three seeds** (0.6937, mean 0.6950). The single-seed numbers in §5 are therefore not optimistic.
- **New users: no seed is significantly below popularity**, and no seed is significantly above it. The group-0 interval includes 0 for all three. The seed-to-seed range (0.6005–0.6059) is as wide as the interval itself, so "about equal to popularity" is the honest reading. The switch is no longer needed for the AUC, but is harmless.
- All 3 seeds chose the last epoch (3). For seed 2 the val sample barely moved from epoch 2 to 3 (0.6921 → 0.6925).

## 6b. The doc 15 recipe on cat + fresh (2026-11-26)

The same fixed recipe as doc 15, with no new choice: the seed average of seeds 42, 1, 2 (within-impression standardised scores, averaged), the equal blend 0.5 × seed average + 0.5 × M6, and both with the group-0 switch. Code: `news/ensemble_val.py cat_fresh`; outputs: `output/news/ensemble/cat_fresh/`. The M6 val scores were rebuilt again and equal the saved doc 13 run for every impression (asserted).

| Run | AUC | MRR | nDCG@5 | nDCG@10 |
|---|---|---|---|---|
| cat + fresh, seed 42 | 0.6937 | 0.3255 | 0.3549 | 0.4159 |
| **seed average `nrms_catfresh_seedavg3`** | 0.6996 | **0.3293** | **0.3596** | **0.4208** |
| seed average + switch | 0.6996 | 0.3294 | 0.3597 | 0.4208 |
| **blend `blend_catfresh_seedavg3_m6_equal`** | **0.7005** | 0.3248 | 0.3554 | 0.4174 |
| blend + switch | 0.7003 | 0.3249 | 0.3555 | 0.4174 |
| doc 15 blend + switch (cscale) | 0.6921 | 0.3197 | 0.3491 | 0.4107 |

**By group (AUC / nDCG@10):**

| Run | 0 | 1–3 | 4–10 | >10 |
|---|---|---|---|---|
| seed average | 0.6072 / 0.3280 | 0.6518 / 0.4105 | 0.6812 / 0.4282 | 0.7139 / 0.4230 |
| blend | 0.6125 / 0.3298 | 0.6561 / 0.4094 | 0.6853 / 0.4261 | 0.7133 / 0.4188 |
| popularity `cum_ctr` | 0.6049 / 0.3288 | | | |

**Paired (difference, 95% interval):**

| Comparison | AUC | nDCG@10 |
|---|---|---|
| seed average − seed 42 | +0.0059 (0.0056 to 0.0062) | +0.0049 (0.0046 to 0.0053) |
| blend − seed average | +0.0009 (0.0005 to 0.0013), Wilcoxon p 0.13 | −0.0034 (−0.0039 to −0.0028) |
| blend − M6 | +0.0256 (0.0251 to 0.0260) | +0.0240 (0.0235 to 0.0244) |
| seed average + switch − doc 15 blend + switch | +0.0075 (0.0069 to 0.0080) | +0.0101 (0.0095 to 0.0107) |
| **empty histories:** seed average − popularity | +0.0023 (−0.0027 to +0.0074) | −0.0008 (−0.0052 to +0.0036) |
| **empty histories:** blend − popularity | **+0.0077 (0.0036 to 0.0117)**, p 5.5e-05 | +0.0010 (−0.0025 to +0.0045) |

**Weight grid (switch, in-sample diagnostic, `weight_grid_switch_val.csv`):** AUC 0.6750 at neural weight 0, 0.7003 at 0.5, highest at 0.7 (0.7035, nDCG@10 0.4235), 0.6996 at 1. The best weight moved from 0.6 (doc 15) to 0.7: the stronger neural model needs less of M6.

**What it means:**

- **The seed average is the clear gain:** +0.0059 AUC and +0.0049 nDCG@10 over one seed, more than for cscale in doc 15 (+0.0021). The three cat + fresh seeds differ more from each other (doc 16 §6a), so averaging them helps more.
- **The seed average is the best model on nDCG@10, MRR and nDCG@5** (0.4208 / 0.3293 / 0.3596), and beats the doc 15 ensemble by +0.0075 AUC and +0.0101 nDCG@10.
- **M6 now adds little.** The blend gains only +0.0009 AUC over the seed average (the paired interval is above 0, but the Wilcoxon test is not significant), and it loses −0.0034 on nDCG@10. In doc 15 the blend added +0.0064 AUC. The stronger neural model already has what LightGBM contributed.
- **New users: the seed average is non-inferior to popularity** for empty histories: +0.0023 AUC, interval −0.0027 to +0.0074. The lower bound is above the pre-registered −0.005 margin, so the switch is no longer needed (it changes the result by less than 0.0001).
- **The blend is the first model in the study that beats popularity for new users** (+0.0077 AUC, interval 0.0036 to 0.0117). For the blend the switch *hurts* group 0 (0.6125 → 0.6049).
- **Choice for M8 (pre-registration).** The seed average is simpler (one model type), best on every top-of-list metric, and non-inferior for new users. The blend is +0.0009 AUC better overall and better for new users, but worse on nDCG@10 and needs M6 as well.

## 7. Next

- **Pre-registration:** M8 is now the seed average without a switch (decided 2026-11-26; draft, not frozen).
- **More epochs** for cat + fresh: all 3 seeds chose the last epoch. This is a separate selection on val and must be reported as one.
