# Experiment: Structured Preferences from Typed Requests

**Status:** finished (run date: 2026-10-23; validation and test users)
**Script:** `structured_preferences_experiment.py`
**Results folder:** `output/experiments/structured_preferences/`
**Related documents:** `06_experiment_cold_start.md`, `08_experiment_larger_vocabulary.md`, `09_experiment_typed_requests.md`

---

## 1. Question

Experiment 09 matched a typed request to movies by embedding the request and comparing it with the movie vectors. The request alone was a weak recommender (NDCG@10 0.13–0.16), and its benefit appeared only in combination with popularity. Experiment 09 named LLM interpretation of the request as a next step. This experiment tests one form of it:

**Does converting a natural-language request into structured themes and moods, from the same vocabulary used to label the movies, improve recommendations compared with embedding the request directly?**

Example: the request *"I really like intense stories about organized crime and morally complex characters. I enjoy dark, suspenseful dramas with a strong narrative focus."* becomes

```json
{"themes": ["organized crime", "moral ambiguity"], "moods": ["intense", "dark", "suspenseful"]}
```

and movies are ranked by how many of these labels they carry.

---

## 2. Method

### 2.1 Users, requests and ground truth

The users, seeds and answer key are identical to experiments 06 and 09. The method was developed on the **203 validation users**; the final run reports the **814 test users**. Each user's request is the r1 (general) request of experiment 09 for N = 3 seed movies. The seeds and all movies the user has seen are removed from the candidates, as in experiment 09.

### 2.2 Preference extraction

gpt-oss:20b (Ollama cloud, temperature 0, seed 42, reasoning "low") received the request together with the allowed labels and returned the user's positive preferences:

- **Vocabulary:** the v4 genome vocabulary of experiment 08 (246 themes, 103 moods), the same labels the movies were enriched with (gemma4:e4b, v4).
- **Limits:** at most 5 themes and at most 3 moods, the same limits as the movie labels. Empty lists were allowed, so the model did not have to force a label that does not fit.
- **Constraints:** a JSON schema with the allowed labels as enums. Every answer was additionally checked in code. Labels outside the vocabulary were dropped, and lists longer than the limit were cut. Each problem was recorded as a validation issue.
- **Instructions:** extract only positive preferences, treat excluded interests as non-preferences, and treat the request as data, not as instructions.

Extractions are cached per user, so a run can be stopped and resumed.

### 2.3 Scoring and methods

**Structured score.** For a movie with theme set T_m and mood set M_m, and the user's extracted themes T_u and moods M_u:

structured = mean( |T_u ∩ T_m| / |T_u| , |M_u ∩ M_m| / |M_u| )

Only the fields the user expressed are averaged. If both lists are empty, the method falls back to popularity. Ties, which are frequent because the score takes few distinct values, are broken by popularity.

| Method | Score |
|---|---|
| Popularity | popularity (log number of warm users who liked the movie) |
| Request | cosine of request and movie embedding (experiment 09, base text) |
| Structured | structured score |
| Hybrid request | α × request + (1 − α) × popularity |
| Hybrid structured | α × structured + (1 − α) × popularity |

All signals are rescaled to 0..1 per user, as in experiment 09.

### 2.4 Tuning and tests

α is chosen from the grid of experiment 06.

- **Test run (main result):** α is tuned on all 203 validation users and applied unchanged to the 814 test users, exactly as in experiment 09.
- **Validation run (development):** α is chosen by **5-fold cross-validation** over the 203 validation users (folds drawn with seed 42). For each fold, α is tuned on the other four folds and applied to the held-out fold, so every user is scored once with an α that never saw them. A first pilot tuned α on 101 users and reported on the other 102. It gave the same picture and was replaced by the cross-validated run.

Differences are tested per user with the Wilcoxon signed-rank test, Bonferroni-corrected over the 6 comparisons in §3.3. 95% confidence intervals of the mean difference come from 5,000 bootstrap resamples of users.

**Validity check:** on the test users, popularity (0.5598), request (0.1262) and hybrid request (0.5747) reproduce experiment 09 exactly.

---

## 3. Results

### 3.1 Extraction

| Statistic | 814 test users | 203 validation users |
|---|---|---|
| Users with no theme | 231 (28%) | 52 (26%) |
| Users with neither theme nor mood (popularity fallback) | 4 | 1 |
| Mean themes / moods per user | 2.21 / 2.87 | 2.24 / 2.89 |
| Users with 3 moods | 730 (90%) | 185 (91%) |
| Users with validation issues | 43 (5%): 16 with more than 3 moods, 27 with a theme outside the vocabulary | 11 (5%): 5 with more than 3 moods, 6 with a theme outside the vocabulary (e.g. "crime") |
| Time and output tokens per user | 2.1 s, 165 tokens | 1.9 s, 164 tokens |

The most frequent moods are generic: "intense" was extracted for **480 of 814 test users (59%)**, followed by "dramatic" (238), "suspenseful" (175) and "dark" (144). On the validation users, the most frequent themes are "war" (31), "organized crime" (22) and "family bonds" (20).

Examples:

| Request (shortened) | Themes | Moods |
|---|---|---|
| "…heartwarming stories about extraordinary beings visiting our world…" | first contact, aliens, supernatural | heartwarming, emotional, feel-good |
| "…intense, scary movies with a strong sense of dread…" | paranoia, psychology, murder, serial killer, terrorism | intense, scary, tense |
| "…thrilling adventures with high stakes… ancient mysteries or futuristic battles…" | — | exciting, intense, suspenseful |
| "…thought-provoking and have a dark edge… high-concept adventures with a good sense of humor." | — | thought-provoking, dark, humorous |

When no theme fits, the request is described by moods alone, which are shared by many movies.

On the validation users, 6 answers contained themes outside the vocabulary although the schema listed only allowed labels; on the test users, 27 did. The cloud endpoint therefore does not seem to enforce the schema strictly. The check in code removed these labels.

### 3.2 Recommendation quality

**814 test users (main result; α tuned on the 203 validation users):**

| Method | NDCG@10 | Precision@10 | Recall@10 | α |
|---|---|---|---|---|
| Popularity | 0.5598 | 0.5477 | 0.0490 | — |
| Request | 0.1262 | 0.1271 | 0.0107 | — |
| Structured | 0.2505 | 0.2364 | 0.0208 | — |
| Hybrid structured | 0.5598 | 0.5477 | 0.0490 | 0 |
| **Hybrid request** | **0.5747** | **0.5623** | **0.0505** | 0.05 |

**203 validation users (5-fold CV):**

| Method | NDCG@10 | Precision@10 | Recall@10 | α chosen per fold |
|---|---|---|---|---|
| Popularity | 0.5626 | 0.5458 | 0.0495 | — |
| Request | 0.1428 | 0.1483 | 0.0121 | — |
| Structured | 0.2332 | 0.2232 | 0.0196 | — |
| Hybrid structured | 0.5599 | 0.5438 | 0.0491 | 0, 0, 0, 0, 0.02 |
| **Hybrid request** | **0.5720** | **0.5527** | **0.0500** | 0.08, 0.05, 0.05, 0.08, 0.08 |

On the test users, the validation users chose α = 0 for the structured hybrid, so it is identical to popularity.

### 3.3 Significance tests (NDCG@10 per user)

| Comparison | Test: mean diff. | Test: wins / losses | Test: p (Bonf.) | Test: 95% CI | Validation: mean diff. (p Bonf.) |
|---|---|---|---|---|---|
| **Structured vs request** | **+0.1243** | 521 / 207 | **< 0.001** | [+0.110, +0.140] | +0.0903 (< 0.001) |
| **Hybrid request vs popularity** | **+0.0149** | 407 / 331 | **< 0.001** | [+0.009, +0.021] | +0.0094 (0.74) |
| **Hybrid request vs hybrid structured** | **+0.0149** | 407 / 331 | **< 0.001** | [+0.009, +0.021] | +0.0121 (0.35) |
| Hybrid structured vs popularity | 0 | 0 / 0 | 1.0 (identical) | — | −0.0027 (1.0) |
| Structured vs popularity | −0.3092 | 115 / 681 | < 0.001 | [−0.329, −0.289] | −0.3294 (< 0.001) |
| Request vs popularity | −0.4336 | 64 / 735 | < 0.001 | [−0.454, −0.414] | −0.4197 (< 0.001) |

The validation results hold on the test users, and with four times as many users the request hybrid's gain over popularity becomes significant.

### 3.4 Comparison with experiment 09 (814 test users)

| | Exp. 09 | Exp. 11 |
|---|---|---|
| Popularity | 0.5598 | 0.5598 |
| Request only, base text | 0.1262 | 0.1262 |
| Request only, best enriched representation (31b P2) | 0.1343 | — |
| **Structured only** | — | **0.2505** |
| Hybrid request | 0.5747 | 0.5747 |
| Hybrid structured | — | 0.5598 (= popularity) |
| Hybrid seeds (N = 3) | 0.5778 | — |
| Hybrid request + seeds (N = 3) | 0.5809 | — |

- **Request matching.** In experiment 09, the best enrichment (31b P2 profiles) raised request-only matching by +0.008 (+6%). Extracting labels from the request raises it by +0.124 (**+98%**, almost double), about fifteen times as much.
- **Hybrids.** The request hybrid is the same as in experiment 09 (+0.015 over popularity, significant). The structured hybrid does not reach it: the validation users chose α = 0, so it becomes popularity.

### 3.5 Model comparison: does a stronger extraction model help?

Setup:
- **Models:** the extraction was repeated with **gemma4:31b** (Ollama cloud), **gemma4:e4b** (local) and **DeepSeek V4.1 Flash** (Ollama cloud, no reasoning), plus **gpt-oss:120b** (Ollama cloud) on the validation users only.
- **Unchanged:** prompt, vocabulary, limits, schema and scoring.
- **What the model affects:** popularity, request and hybrid request don't use the extraction, so they're identical for every model. Only *structured* and *hybrid structured* depend on it.

**Extraction per model (814 test users; gpt-oss:120b: 203 validation users):**

| Model | Users with no theme | Mean themes / moods | Answers with validation issues | Most frequent moods | Time per user |
|---|---|---|---|---|---|
| gpt-oss:20b (cloud) | 231 (28%) | 2.21 / 2.87 | 43 | intense 480, dramatic 238, suspenseful 175 | 2.1 s |
| gpt-oss:120b (cloud, val.) | 47 (23%) | 2.08 / 2.92 | 11 | — | 1.4 s |
| gemma4:31b (cloud) | 220 (27%) | 1.35 / 2.84 | 36 | intense 414, dramatic 312, epic 184 | 0.8 s |
| gemma4:e4b (local) | **0 (0%)** | **4.56** / 3.00 | 0 | intense 470, dramatic 381, epic 205 | 2.1 s |
| DeepSeek V4.1 Flash (cloud) | 130 (16%) | 2.95 (4.10 before cleaning) / 2.84 | **451** | intense 421, dramatic 248, exciting 188 | 1.0 s |

- **The local gemma4:e4b never leaves the themes empty.** It nearly always fills all 5 slots (mean 4.56), even for requests that name no topic, such as "thrilling adventures with high stakes".
- **The larger models use the empty list** that the prompt allows, in about a quarter of the cases.
- **gemma4:31b is the most cautious:** 1.35 themes on average.
- **The local schema is enforced strictly,** so e4b has no validation issues; the cloud endpoints aren't strict (section 3.1).
- **DeepSeek ignores the vocabulary most often.** More than half of its answers (451 of 814) contain labels outside the list, almost all **genre words used as themes**: "adventure" 125 times, "action" 87, "high stakes" 57, "mystery" 35, "drama" 31. It wants to fill the theme slots like e4b (4.1 themes before cleaning). The invalid labels are removed in code, leaving 2.95 themes and 16% of users with none.

**Structured NDCG@10 per model:**

| Model | Validation (5-fold CV, 203) | **Test (814)** | Structured vs request on test (p Bonf.) | Hybrid structured on test (α) |
|---|---|---|---|---|
| gpt-oss:20b | 0.2332 | **0.2505** | +0.124 (< 0.001), 521 / 207 users | 0.5598 = popularity (α = 0) |
| gemma4:31b | **0.2603** | 0.2478 | +0.122 (< 0.001), 528 / 204 users | 0.5598 = popularity (α = 0) |
| gemma4:e4b | 0.2054 | 0.2000 | +0.074 (< 0.001), 472 / 236 users | 0.5598 = popularity (α = 0) |
| DeepSeek V4.1 Flash | 0.2192 | 0.2352 | +0.109 (< 0.001), 522 / 205 users | 0.5645 (α = 0.02) |
| gpt-oss:120b | 0.2417 | — | — | — (CV: α = 0 in all folds) |
| *Request only (no extraction)* | *0.1428* | *0.1262* | | |
| *Popularity* | *0.5626* | *0.5598* | | |

**Paired tests between the models (structured, test, Wilcoxon):**

| Comparison | Difference | p |
|---|---|---|
| gpt-oss:20b vs gemma4:31b | +0.003 | 0.92 |
| gpt-oss:20b vs gemma4:e4b | **+0.051** | 4.9e-12 |
| gemma4:31b vs gemma4:e4b | **+0.048** | 1.9e-10 |
| DeepSeek vs gpt-oss:20b | −0.015 | 0.083 |
| DeepSeek vs gemma4:31b | −0.013 | 0.084 |
| DeepSeek vs gemma4:e4b | **+0.035** | 8.5e-8 |

**Findings:**
- **The two large cloud models are equally good.** They differ by 0.003 on test, and their order flips between validation and test. gpt-oss:120b on validation (0.2417) sits between them, so a model larger than about 20B parameters brings no further gain.
- **The small local model is clearly worse,** by about 0.05 (−20%), but still far better than embedding the request (+0.074, p < 0.001).
- **DeepSeek V4.1 Flash sits in between:** about 0.015 below the two large models (not significant, p ≈ 0.08) and 0.035 above e4b (p < 0.001). Its tendency to fill the theme slots with genre words costs accuracy. Cleaning removes those words, but the remaining themes are less selective than those of gpt-oss:20b and gemma4:31b. In experiment 14, DeepSeek assigned the most accurate *film* labels, but it is not the best at reading *requests*.
- **The cause is over-assignment, not wrong labels.**
  - e4b gives a generic request 4–5 themes it doesn't support. A film then scores on themes the user never asked for, and the ranking becomes noise.
  - The larger models leave the themes empty instead. Their users are ranked by moods only, which is coarse but not misleading.
  - This is the same pattern as the label dilution in experiments 08 and 14: more labels per item is not more information.
- **The hybrid is almost unaffected.** For gpt-oss:20b, gemma4:31b and gemma4:e4b, the validation users choose α = 0 for the structured hybrid, so on test it is identical to popularity. Only DeepSeek gets a small weight (α = 0.02): +0.005 over popularity on test (p = 0.012, 0.07 after Bonferroni), still below the request hybrid (−0.010, p = 0.013 corrected). A better extraction model improves request matching alone but does not change the final recommendation.

---

## 4. Findings

1. **Structured preferences match requests much better than embeddings do.** Ranking by the share of extracted labels a movie carries improves request-only NDCG@10 on the test users from 0.126 to 0.251 (+98%, significant, better for 521 of 814 users; +63% on the validation users). This is by far the largest effect of LLM processing on requests seen so far. The movie labels and the user preferences come from the same vocabulary, so they meet in the same terms, whereas an embedding compares a casual sentence with a long movie description.
2. **Alone, neither request method is a usable recommender.** Both remain far below popularity (−0.31 and −0.43 on the test users).
3. **Structured preferences add nothing to popularity.** The validation users chose α = 0 (in four of five cross-validation folds, and when tuned on all of them), so on the test users the hybrid is identical to popularity. The structured score is coarse: most movies share none of the user's labels, and many share the same 1–2 labels. Mixing it into the continuous popularity score changes the order only near the top, where popular films already are.
4. **The embedded request remains the better partner for popularity.** On the test users, hybrid request beats popularity, and therefore also hybrid structured, by +0.015 (+2.7%, significant; 407 wins, 331 losses). On the 203 validation users the same direction was too small to be significant. The embedding gives a continuous score that can shift popular films slightly, which the label overlap cannot.
5. **The extraction is limited by generic requests, not by the vocabulary size.** The vocabulary has 246 themes and 103 moods, yet 28% of test users get no theme. Their requests ("thrilling adventures with high stakes") name no topic. The extracted moods are dominated by "intense" (59% of test users), which repeats the generic wording of the r1 requests noted in experiment 09.
6. **A large model is needed for extraction, but not a very large one.** gpt-oss:20b and gemma4:31b are equal (0.251 vs 0.248 on test, p = 0.92), and gpt-oss:120b brings nothing more on validation. DeepSeek V4.1 Flash (0.235) is not better either: it fills themes with genre words outside the vocabulary. The small local gemma4:e4b is about 0.05 worse because it fills in themes for generic requests instead of leaving them empty (section 3.5).

**Answer to the question.** Converting a request into structured themes and moods is clearly better than embedding it when the request is the only signal. Once popularity is combined with it, the structured form loses its advantage, and the embedded request remains the better choice for the hybrid. For the TDK abstract, LLM interpretation of the request helps preference matching substantially, but the final recommendation quality for new users is still dominated by popularity.

---

## 5. Limitations

- **Simulated requests.** As in experiment 09, the requests were written by an LLM from three seed films and contain no information beyond them. Many are generic, which caps what any extraction can recover.
- **Four extraction models** (section 3.5), but only one prompt, one request style (r1), and one seed count (N = 3). gpt-oss:120b was run on the validation users only.
- **The schema was not strictly enforced** by the cloud endpoint. Out-of-vocabulary labels were removed in code, affecting 27 test and 6 validation answers.
- **Simple scoring.** The structured score counts exact label matches, with themes and moods weighted equally. It ignores label specificity (a rare theme such as "first contact" counts as much as "war") and related labels ("mafia" vs "organized crime").
- **Ground truth is the user's later history,** which favours popular films (see experiments 06 and 09).

---

## 6. Next steps

- **Weight labels by specificity** (e.g. IDF of the label over the catalog), so rare, informative labels count more than "intense" or "war".
- **Combine both request signals,** e.g. α × (w × structured + (1 − w) × request) + (1 − α) × popularity, since the structured score is better alone and the embedding blends better with popularity.
- **Use the structured form for constraints,** where it is most natural: extract explicit restrictions (era, length, audience, content intensity) and enforce them with the constraint layer.
- **Try r2 (specific) requests,** which name more concrete themes and might reduce the share of users without a theme.

---

## 7. Files

| File | Content |
|---|---|
| `output/experiments/structured_preferences/preferences_gpt-oss_20b-cloud.jsonl` | extracted themes and moods per user (1,017 users), raw answer, validation issues, timing |
| `output/experiments/structured_preferences/test_results_gpt-oss_20b-cloud.csv` | test users: NDCG, precision and recall per user and method, with α |
| `output/experiments/structured_preferences/test_tests_gpt-oss_20b-cloud.csv` | test users: Wilcoxon tests, Bonferroni-corrected p-values, bootstrap CIs |
| `output/experiments/structured_preferences/cv_results_gpt-oss_20b-cloud.csv` | validation users: NDCG, precision and recall per user and method, with fold and chosen α |
| `output/experiments/structured_preferences/cv_tests_gpt-oss_20b-cloud.csv` | validation users: Wilcoxon tests |
| `output/experiments/structured_preferences/pilot_results_gpt-oss_20b-cloud.csv` | first pilot (α tuned on 101 users, reported on 102); superseded |
| `..._gemma4_31b-cloud.*`, `..._gemma4_e4b.*` | the same files for the model comparison (section 3.5): preferences, test and CV results and tests |
| `..._deepseek-v4.1-flash.*` | the same files for DeepSeek V4.1 Flash (validation and test) |
| `..._gpt-oss_120b-cloud.*` | preferences (203 validation users) and CV results and tests only; no test run |

---

## 8. Reproducing the run

```bash
python structured_preferences_experiment.py
```

```bash
python structured_preferences_experiment.py gpt-oss:20b-cloud test
```

The first command runs gpt-oss:20b-cloud on the 203 validation users with cross-validated α. The second tunes α on the validation users and reports the 814 test users (about 30 minutes of extraction). A different model or number of validation users can be given as `python structured_preferences_experiment.py <model> <users>`. Cloud models need `OLLAMA_API_KEY` in `.env`; other model names use the local Ollama server. Extraction takes about 2 s per user and is cached; evaluation takes a few minutes. The script requires the requests of experiment 09, the v4 vocabulary and movie labels of experiment 08, and the base embedding index of experiment 05. All random choices use seed 42.
