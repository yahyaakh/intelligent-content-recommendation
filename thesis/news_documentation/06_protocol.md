# 06 — Protocol: validation day, history groups, LLM samples, paired tests

**Status:** done 2026-11-19 (step 1 of the plan in doc 05).
- **Code:** `news/news_protocol.py`.
- **Outputs:** `output/news/protocol/` (samples and group tables) and `output/news/final_test/preregistration.md` (skeleton, not frozen).

---

## 1. What was fixed

| Part | Decision |
|---|---|
| Training data | train impressions of **Nov 9–13**: 1,801,231 impressions |
| Validation | train impressions of **Nov 14**: 431,517 impressions, split name `val`. Every choice (α, features, prompts, list length) is made here. |
| Final test | MIND **dev** (Nov 15): 376,471 impressions, run once per method under a frozen pre-registration |
| History groups | **0 / 1–3 / 4–10 / >10** clicks in the MIND history field |
| LLM samples | fixed and stratified: `val_llm` = 500 impressions per group (2,000); `dev_llm` = 2,500 per group (10,000); seed 42 |
| Statistics | paired over impressions: mean difference, 95% interval, two-sided Wilcoxon signed-rank test. Bonferroni correction over the comparisons listed in the pre-registration. |

**Why equal numbers per group:** the new-user groups are small (3% and 7% of dev). A random sample of 2,000 would hold only about 60 empty-history impressions. With 500 or 2,500 per group, each group can be tested on its own. An estimate for the whole split is the group means weighted by the group shares (`group_shares()`).

**All impressions can be evaluated:** every impression in val and dev has at least one click and one non-click, so no impression is dropped from AUC.

---

## 2. The module

`news/news_protocol.py` works on top of `mind_eval.py`:

| Function | What it does |
|---|---|
| `load_val()` | Nov 14 as an `Impressions` object with split `val`. `mind_eval.evaluate()` then saves to `output/news/results/val/`. |
| `val_mask()`, `fit_mask()` | impression masks over the train split, e.g. for `popularity_features.load_feature_matrix(..., impression_mask=...)` |
| `load_sample("val_llm" / "dev_llm")` | the sampled impressions as their own split; results go to `results/val_llm/` and `results/dev_llm/` |
| `history_group()`, `group_shares()` | group labels and group weights |
| `groups_table(split, metric)` | every saved run of a split, by history group |
| `compare(split, a, b, metric)` | paired test of two runs, overall and per group |

Command line: `python news/news_protocol.py samples | facts | groups dev [metric] | compare dev A B [metric]`.

---

## 3. Group sizes

| History clicks | Val (Nov 14) | Dev (Nov 15) |
|---|---|---|
| 0 | 11,393 (2.6%), median 26 candidates | 11,270 (3.0%), median 23 candidates |
| 1–3 | 29,744 (6.9%), median 22 | 27,173 (7.2%), median 18 |
| 4–10 | 94,697 (21.9%), median 23 | 85,048 (22.6%), median 20 |
| >10 | 295,683 (68.5%), median 28 | 252,980 (67.2%), median 25 |
| Users | 286,814 | 255,990 |

Mean candidates per impression in the LLM samples:

| Sample | Group 0 | 1–3 | 4–10 | >10 |
|---|---|---|---|---|
| `val_llm` | 41.6 | 36.6 | 32.6 | 46.1 |
| `dev_llm` | 38.8 | 31.8 | 33.8 | 39.2 |

The longest lists in both samples have 263–290 candidates.

---

## 4. Finding: the MIND history field is fixed

The MIND history field does **not** grow during the log. It holds the clicks made *before* the log period, and it is the same in every impression of a user:
- In train, the history never changes between a user's impressions (0 of 1,521,526 repeat impressions differ).
- In dev, it never changes either (0 of 120,481).
- For the 216,778 users who appear in both train and dev, the history is **identical** in train and dev.

So clicks made inside the log (in the impressions of Nov 9–14) never reach the history. **80.5% of the "empty history" dev impressions come from users who clicked in train impressions**, with a median of 4 clicks:

| Dev history group | User clicked in train impressions | Median train clicks (of those) |
|---|---|---|
| 0 | 80.5% | 4 |
| 1–3 | 75.5% | 2 |
| 4–10 | 71.2% | 2 |
| >10 | 94.0% | 7 |

**Consequences:**
- **Main protocol:** the user profile is the MIND history field, as in the standard MIND set-up and in the published models. The groups therefore mean "new user as the model sees them".
- **Truly new users:** a stricter group is empty history **and** no earlier impressions. In dev that is the 19.5% of the empty-history group whose user is not in train, about 2,200 impressions.
- **Possible extra experiment:** add the user's own clicks from earlier days to the history. This is allowed by the leakage rule (earlier days only) and would turn about 80% of empty-history impressions into short histories. It is a news-specific way to fight cold start. Decide before the freeze whether to run it.

---

## 5. First result: the existing models by history group (dev, development runs)

These are the development runs of doc 04, split by group from their saved per-impression results. No model was retrained.

**AUC:**

| Model | 0 | 1–3 | 4–10 | >10 | All |
|---|---|---|---|---|---|
| Random | 0.505 | 0.500 | 0.499 | 0.500 | 0.500 |
| Co-visitation, top 50 | 0.500 | 0.501 | 0.502 | 0.505 | 0.504 |
| MLP, 3 category features | 0.500 | 0.547 | 0.596 | 0.637 | 0.617 |
| `cum_ctr` alone (popularity) | **0.623** | 0.634 | 0.636 | 0.627 | 0.629 |
| LightGBM, 5 MiniLM features | 0.500 | 0.581 | 0.630 | 0.670 | 0.649 |
| LightGBM, MiniLM + popularity | 0.604 | **0.639** | **0.669** | **0.695** | **0.682** |

**nDCG@10:**

| Model | 0 | 1–3 | 4–10 | >10 | All |
|---|---|---|---|---|---|
| Random | 0.292 | 0.327 | 0.314 | 0.273 | 0.287 |
| `cum_ctr` alone | **0.369** | 0.416 | 0.406 | 0.357 | 0.373 |
| LightGBM, 5 MiniLM features | 0.289 | 0.397 | 0.419 | 0.411 | 0.408 |
| LightGBM, MiniLM + popularity | 0.363 | **0.431** | **0.445** | **0.430** | **0.431** |

nDCG differs between groups even for random scores, because list length and the number of clicks differ. Compare models within a group, not groups with each other.

**Paired tests (AUC, dev, all 376,471 impressions):**

| Comparison | Group 0 | 1–3 | 4–10 | >10 | All |
|---|---|---|---|---|---|
| MiniLM + popularity minus `cum_ctr` | **−0.020** [−0.025, −0.014], p = 7e-6 | +0.005 [+0.001, +0.009], p = 1e-7 | +0.033, p ≈ 0 | +0.068, p ≈ 0 | +0.053, p ≈ 0 |
| MiniLM + popularity minus MiniLM only | +0.103, p = 1e-164 | +0.058 | +0.039 | +0.025 | +0.033 |

The brackets are 95% intervals; "p ≈ 0" means p is below the computer's precision.

**What this shows:**
1. **For users with no history, popularity alone beats the best model.** `cum_ctr` reaches 0.623 against 0.604 for LightGBM with MiniLM + popularity, a significant difference. This is the same result as in the movie study, where popularity was the hard bar for new users. A likely reason: only about 2% of training impressions have an empty history, so the ranker learns little about that case. A Hybrid-E-style mix with α chosen per group (doc 05, N2) or a group-aware ranker should fix it.
2. **Content alone is useless without a history.** MiniLM and category features give 0.500 for group 0, because there is nothing to compare with: co-visitation, the MLP and MiniLM-only LightGBM all give every candidate the same score (identical AUC 0.5005), so the order comes from the random tie-break. They become useful as the history grows (0.581 → 0.630 → 0.670).
3. **The gain of the learned ranker grows with history length:** −0.020, then +0.005, +0.033 and +0.068 over popularity. This is the news version of the movie finding that personal signals help more with more seed films.
4. **Item co-visitation is random in every group,** including users with long histories.

These are development numbers on dev. The same comparisons will be repeated on validation and then in the pre-registered final test.

---

## 6. Open decisions (to settle before the freeze)

1. **Refit the final models on Nov 9–14,** or keep the Nov 9–13 models. Recommended: refit, since news goes out of date fast.
2. **Primary metric:** AUC (the MIND leaderboard metric) in the skeleton; nDCG@10 is the link to the movie chapter.
3. **Add earlier-day clicks to the history** (§4): as an extra experiment only, or not at all.
4. **Equivalence margin** for "no difference" claims (e.g. ±0.005 AUC).
