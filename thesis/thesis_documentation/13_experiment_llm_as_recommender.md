# Experiment: The LLM as a Recommender for New Users (Reranking and Generation)

**Status:** finished for this stage (run dates: 2026-10-23 to 2026-10-25); one further confirmation planned (§8)
**Scripts:** `llm_rerank_experiment.py` (ideas 1 and 3), `llm_generate_experiment.py` (idea 2), `verify_llm_rerank.py` (verification), `beyond_accuracy_llm.py` (§6.2), `shuffle_average_experiment.py` (§6.3)
**Results folders:** `output/experiments/llm_rerank/`, `output/experiments/llm_generate/`, `output/experiments/beyond_accuracy_llm/`, `output/experiments/shuffle_average/`
**Related documents:** `06_experiment_cold_start.md` (Hybrid E, users, split), `07_experiment_outside_knowledge.md`, `08_experiment_larger_vocabulary.md` (v4 labels), `11_experiment_structured_preferences.md`, `12_experiment_beyond_accuracy.md`

---

## 1. Questions

Experiments 04–12 used LLMs to **describe items** (labels, profiles) or to **interpret requests**. This experiment lets the LLM **recommend** to a new user directly, in three roles:

| Role | The LLM receives | Question |
|---|---|---|
| **Idea 1: rerank collaborative filtering** | the user's seed films + Hybrid E's top 30 (shuffled) | Can an LLM improve the best method's ranking? |
| **Idea 3: rerank popularity** | the seed films + the 30 most popular unseen films (shuffled) | Can an LLM personalise a non-personal list, without any rating data about the user's taste? |
| **Idea 2: recommend freely** | only the seed films | Can an LLM recommend well on its own, from its own knowledge? |

Within these, the experiment varies the model (12 models, 8 families, 4B to frontier), reasoning, the prompt (four levels of film information) and the number of seed films.

---

## 2. Method

### 2.1 Users and ground truth

Identical to experiments 06–12: 1,017 cold users, each with a pool of 10 seed films (their first liked films in time order) and an answer key of about 128 later liked films; **203 validation users** for every choice of model, prompt and setting, **814 test users** used only for pre-specified final comparisons. The first N seeds (N = 3, 5 or 10) are shown to the LLM; all 10 are excluded from recommendation, so the answer key is the same for every N. Metric: NDCG@10 per user (also precision and recall), averaged over users.

### 2.2 Reranking (ideas 1 and 3)

1. **Candidates:** Hybrid E's top 30 (item-CF + popularity, α from experiment 06 for the given N) or popularity's top 30 (the same list for everyone, minus seen films).
2. **Shuffling:** the 30 candidates are shuffled with the user id as seed and numbered 1–30, so the model cannot copy the original order.
3. **Prompt:** the seed films, then the numbered candidates; the model returns `{"ranking": [numbers]}`. Prompt versions:

| Version | Information per film | Profile of the seeds |
|---|---|---|
| p1 | title, year, genres | — |
| p2 | + director, cast | most common genres, directors, actors, decades |
| p3 | p2 + v4 themes and moods (experiment 08) | + themes, moods |
| p4 | p3 + runtime, 5 TMDB keywords, one-line plot | as p3 |

4. **Parsing:** the last JSON object with a `ranking` is used (code fences, extra text and corrected second answers are handled); invalid or repeated numbers are dropped and missing films appended in the candidates' original order. Every repair is counted.
5. **Settings:** temperature 0, seed 42, thinking off unless stated; local models with a 16,384-token context window (longer prompts are otherwise silently cut). Answers are cached per model and setting.
6. **Ceiling:** oracle@30 = the best NDCG@10 any reranking of the 30 candidates can reach (all relevant candidates first).

### 2.3 Generation (idea 2)

The model receives only the seeds (g1: as p1; g2: as p2) and is asked for 15 films as `{"title", "year"}`. Each answer is matched to the catalogue (normalised title and year ±1, then the title without subtitle, then without spaces, then fuzzy matching) and classified as **valid** (in the 1,547-film candidate pool, not seen, not repeated), **seen**, **not in pool**, **unmatched** (not in the 1,957-film catalogue) or **duplicate**. Scores: **filled** = the first 10 valid answers; **strict** = the first 10 answers with every invalid one counted as a miss.

**Matcher validation (no model calls):** 1,800 catalogue titles rewritten as an LLM might (lowercase, no punctuation, year +1, no year, "The" added or removed) were matched to the right film in 99.7–100% of cases; 10 hand-written variants ("Godfather Part 2", "Star Wars: Episode IV – A New Hope", "Seven" for *Se7en*) all correctly; invented titles were never matched. All 18 inspected "unmatched" answers of real runs (e.g. *Dark City*, *Soylent Green*, *Signs*) are genuinely absent from the catalogue.

### 2.4 Statistics and verification

Paired Wilcoxon signed-rank tests per user, Bonferroni-corrected over the models in a table. Final claims rest on **pre-specified test runs** on the 814 test users, fixed before running (§3.6, §4.3).

`verify_llm_rerank.py` rebuilds every reranking run from scratch and checks it independently: NDCG recomputed with a separate implementation (largest difference to the saved results 1 × 10⁻¹⁶ in every checked run); candidates without seen films or duplicates; the LLM ranking a permutation of exactly the 30 candidates; the oracle never below any method; no ratings, scores or labels in any prompt. All checks passed for every idea-1 run (validation and test) and every idea-3 run (§6.1). The cache file and label of a run come from one shared function (`run_name`) in both scripts, so the verification always reads the same answers the experiment used. The script also writes side-by-side top-10 lists (candidates' top 10, LLM, oracle) with every hit marked.

---

## 3. Results: idea 1, reranking Hybrid E

Oracle@30 = 0.95 (validation and test, N = 10): on average 16 of the 30 candidates are relevant, so a perfect reranker could raise NDCG@10 from about 0.63 to 0.95.

### 3.1 Eight models, prompt p1, N = 3 (validation; Hybrid E = 0.5837)

| Model | LLM | vs Hybrid E | p | Answers repaired | Position bias |
|---|---|---|---|---|---|
| gemma4:e4b (local, 4B) | 0.5528 | −0.031 | 0.036 | 61 | 0.00 |
| gemma4:31b | 0.5528 | −0.031 | 0.069 | 0 | 0.04 |
| glm-5.3-flash (reasons in answer) | 0.5485 | −0.035 | 0.028 | 0 | 0.03 |
| deepseek-v4.1-flash | 0.5459 | −0.038 | 0.023 | 0 | 0.16 |
| qwen3.5:9b (local) | 0.5448 | −0.039 | 0.007 | 73 | 0.09 |
| gpt-oss:120b | 0.5377 | −0.046 | 0.008 | 12 | 0.23 |
| llama3.1:8b (local) | 0.5266 | −0.057 | < 0.001 | 6 | **0.52** |
| gpt-oss:20b | 0.5208 | −0.063 | < 0.001 | 52 | 0.21 |

*Position bias:* Spearman correlation between the (random) shown number and the model's rank; 0 = none, 1 = keeps the shown order. Every model is below Hybrid E. Mixing the LLM order with Hybrid E's score (β = 0.1…0.7) gained at most +0.005, never significantly. A 20-user pilot had shown all three local models *above* Hybrid E; on 203 users all were below, which is why every pilot here is treated as a format check only.

### 3.2 Prompt × number of seeds (gemma4:e4b, validation)

| | N = 3 | N = 5 | N = 10 |
|---|---|---|---|
| Hybrid E | 0.5837 | 0.6063 | 0.6286 |
| p1 | 0.5528 (−0.031) | 0.5768 (−0.030) | 0.5904 (−0.038) |
| p2 | 0.5476 (−0.036) | 0.5873 (−0.019) | 0.6176 (−0.011, n.s.) |
| **p2 − p1** | −0.005 (p = 0.89) | +0.011 (p = 0.31) | **+0.027 (p = 0.002)** |

The value of the detailed prompt grows with the number of seeds (interaction N = 10 vs N = 3: +0.032, p = 0.04): with three films the "profile" only restates them; with ten it summarises a real taste, and the LLM reaches Hybrid E's level (not significantly different).

### 3.3 More film information hurts (p2 → p3 → p4, N = 10, validation)

| Prompt | gemma4:e4b | DeepSeek Flash |
|---|---|---|
| p2 | **0.6176** | **0.6212** |
| p3 (+ v4 themes and moods) | 0.5951 | 0.5958 |
| p4 (+ runtime, keywords, plot) | 0.5820 | 0.6018 |
| **p3 − p2** | **−0.023 (p = 0.028)** | **−0.025 (p = 0.002)** |
| p4 − p3 | −0.013 (p = 0.21) | +0.006 (p = 0.28) |

Adding the LLM-generated themes and moods makes both models significantly worse. The popularity of the chosen films does not change (log-popularity 8.38 → 8.38), but the theme overlap between the user's profile and the chosen films rises (gemma 0.89 → 0.96, DeepSeek 0.92 → 1.04 shared themes per film), and the chosen films come from further down Hybrid E's list (mean rank 12.3 → 13.2 and 11.7 → 12.6). The labels turn the LLM into a theme matcher; theme similarity is not what predicts a user's later choices.

### 3.4 Model strength and reasoning (p2, N = 10, validation; Hybrid E = 0.6286)

| Model | LLM | vs Hybrid E | p | Cost per 203 users |
|---|---|---|---|---|
| deepseek-v4.1-flash | **0.6212** | −0.007 | 0.37 | ≈ $0.13 |
| glm-5.3-flash (reasoning) | 0.6201 | −0.009 | 0.44 | ≈ $0.40 |
| gemma4:e4b (local) | 0.6176 | −0.011 | 0.28 | free |
| mistral-large-4 | 0.6135 | −0.015 | 0.13 | ≈ $0.27 |
| deepseek-v4-pro | 0.6055 | −0.023 | 0.045 | ≈ $0.25 (off-peak) |
| nemotron-3-super | 0.5965 | −0.032 | 0.012 | ≈ $0.02 |

The larger DeepSeek V4 Pro is significantly worse than V4.1 Flash (−0.016, p = 0.005). In side-by-side lists, Pro follows the seeds' theme more strictly (e.g. 80s action films for a user who went on to like *Star Wars* and *Saving Private Ryan*). Reasoning (GLM, gpt-oss "high") did not help in this role. The public Agent Arena leaderboard (2026-10-23) also ranks V4.1 Flash (rank 13 of 55) above V4 Pro (31), so the finding is not specific to this task.

### 3.5 Test run (pre-specified: p2, N = 10, each model vs Hybrid E, Bonferroni over 2; 814 test users)

| Method | NDCG@10 | vs Hybrid E | Better / worse | p (Bonferroni) | Hits in top 10 |
|---|---|---|---|---|---|
| **Hybrid E** | **0.6311** | — | — | — | 6.15 |
| DeepSeek V4.1 Flash | 0.6169 | −0.014 | 356 / 426 | 0.012 | 6.02 |
| gemma4:e4b | 0.6001 | −0.031 | 324 / 469 | < 0.001 | 5.87 |
| Oracle@30 | 0.9645 | | | | |

---

## 4. Results: idea 3, reranking popularity

Popularity's and Hybrid E's top-30 lists share 64–76% of their films; the ceiling is almost as high (oracle@30 0.92 validation, 0.94 test).

### 4.1 Six models, p2, N = 10 (validation; popularity = 0.5626, Hybrid E = 0.6286)

| Model | LLM | vs popularity | p | p (Bonferroni over 6) | vs Hybrid E |
|---|---|---|---|---|---|
| glm-5.3-flash (reasoning) | 0.5999 | +0.037 | 0.012 | 0.074 | −0.029 |
| deepseek-v4.1-flash | 0.5942 | +0.032 | 0.014 | 0.086 | −0.034 |
| deepseek-v4-pro | 0.5893 | +0.027 | 0.046 | 0.27 | −0.039 |
| mistral-large-4 | 0.5891 | +0.027 | 0.058 | 0.35 | −0.040 |
| gemma4:e4b (4B) | 0.5591 | −0.004 | 0.84 | 1.0 | −0.070 |
| nemotron-3-super | 0.5553 | −0.007 | 0.65 | 1.0 | −0.073 |

GLM-5.2 (run later, no reasoning): 0.5972, +0.035. All five large models raise popularity by +0.027 to +0.037; the two weak models do not. With N = 3 seeds, neither DeepSeek (−0.022) nor gemma (−0.025) improves popularity. In contrast to idea 1, where gemma4:e4b was as good as the large models, personalising a non-personal list requires a strong model.

### 4.2 Reasoning helps in this role (validation, 203 users)

| DeepSeek V4.1 Flash | LLM | vs popularity | p (Bonferroni over 4) | Output tokens | Time per user |
|---|---|---|---|---|---|
| no thinking | 0.5942 | +0.032 | 0.057 | 94 | 0.9 s |
| **thinking** | **0.6063** | **+0.044** | **0.008** | 10,285 | 28 s |
| thinking − no thinking | **+0.012** | 107 better / 72 worse | p = 0.025 (paired) | | |

With thinking, the remaining gap to Hybrid E is −0.022 (p = 0.15, not significant on 203 users). The cost is about 110 times higher (≈ $1.30 off-peak per 203 users instead of ≈ $0.07). In idea 1 reasoning did not help; when the LLM itself must infer the user's taste from the seeds, it does.

### 4.3 Pre-registered test (fixed 2026-10-24, before running)

DeepSeek V4.1 Flash without thinking, idea 3, p2, N = 10; one test: LLM vs popularity, two-sided Wilcoxon, α = 0.05; 814 test users never used for idea 3.

| | NDCG@10 |
|---|---|
| Popularity | 0.5598 |
| **DeepSeek Flash reranking popularity** | **0.5999** |
| Hybrid E (descriptive) | 0.6311 |

**LLM vs popularity: +0.040, 441 users better / 349 worse, p = 1.5 × 10⁻⁶ — confirmed.** Against Hybrid E (descriptive): −0.031, p = 4 × 10⁻⁷. 1 of 814 answers needed repair; cost ≈ $0.24. The LLM achieves 56% of collaborative filtering's improvement over popularity (+0.040 of +0.071) without any rating data about the user beyond the 10 named films.

### 4.4 More film information in idea 3 (p2 → p3 → p4, DeepSeek Flash, N = 10, validation)

In §3.3, more film information hurt the reranking of Hybrid E. The same ladder was repeated for idea 3, where the LLM must do the personalising itself: richer film descriptions might help it there.

| Prompt | NDCG@10 | vs popularity | Better / worse users | p | Input tokens per user |
|---|---|---|---|---|---|
| **p2** | **0.5942** | **+0.032** | 114 / 78 | **0.014** | 1,623 |
| p3 (+ v4 themes and moods) | 0.5798 | +0.017 | 102 / 90 | 0.25 | 2,586 |
| p4 (+ runtime, keywords, plot) | 0.5879 | +0.025 | 103 / 88 | 0.067 | 4,643 |

| Paired step | Difference | Better / worse users | p |
|---|---|---|---|
| p3 − p2 | −0.014 | 77 / 106 | 0.061 |
| p4 − p3 | +0.008 | 95 / 78 | 0.13 |
| p4 − p2 | −0.006 | 83 / 99 | 0.35 |

**The pattern of idea 1 repeats, more weakly.**
- **The labels hurt.** Adding the LLM-generated themes and moods (p3) lowers NDCG by 0.014 and halves the gain over popularity, which is no longer significant. The drop itself is just above the 5% level (p = 0.061), whereas in idea 1 it was significant (−0.025, p = 0.002).
- **The plot partly makes up for it.** Adding runtime, keywords and the plot (p4) recovers part of the loss, but p4 stays below p2 and costs almost three times the input tokens.
- **No richer prompt beats p2,** in either role.

The differences in the second table are of the size that §6.3 attributes to the candidate order alone (up to about 0.009). Only the direction, consistent across both ideas, supports the conclusion. p2 was kept for the pre-registered test (§4.3), which had been fixed before this ladder was run.

---

## 5. Results: idea 2, the LLM recommends freely (validation, 203 users)

| | g1, N = 3 | g2, N = 10 |
|---|---|---|
| Popularity | 0.563 | 0.563 |
| Hybrid E | 0.584 | 0.629 |
| Idea 1 (same model reranking Hybrid E) | DeepSeek 0.546 / gemma 0.553 | DeepSeek 0.621 / gemma 0.618 |
| **DeepSeek Flash, filled** | 0.508 | **0.574** |
| gemma4:e4b, filled | 0.447 | 0.487 |
| DeepSeek Flash, strict | 0.433 | 0.542 |

| Answers (share of all suggestions) | Valid | Outside the catalogue | A seed film | Duplicate |
|---|---|---|---|---|
| DeepSeek, g2 N = 10 | 92% | 7% | 1% | 0.1% |
| DeepSeek, g1 N = 3 | 82% | 10% | 8% | 0.1% |
| gemma, g2 N = 10 | 80% | 15% | 4% | 1% |
| gemma, g1 N = 3 | 77% | 16% | 7% | 1% |

DeepSeek with g2 and 10 seeds equals popularity (+0.011, n.s. after correction) and is significantly below Hybrid E (−0.055) and below its own reranking result (−0.047). gemma is significantly below popularity in both settings. The free recommendations are mainstream: catalogue coverage 17–23% (hybrids 2–6%, content methods 48–63%, experiment 12), long-tail share 8–11%.

---

## 6. Further analyses

### 6.1 Verification of the idea-3 runs

All 11 idea-3 runs (10 validation runs of §4.1–4.2 and the pre-registered test run of §4.3) were rebuilt and checked as in §2.4. In every run the LLM NDCG and the popularity baseline recomputed independently equal the saved results (difference 0), and all integrity and leakage checks passed. Position bias per run: 0.20 for DeepSeek Flash without thinking, **0.04 with thinking**, 0.04–0.10 for GLM and Mistral, about 0 for gemma and Nemotron. Thinking makes DeepSeek rely much less on the order in which the candidates are shown, which may explain part of its gain in §4.2.

### 6.2 Beyond-accuracy metrics (definitions of experiment 12)

Top-10 lists rebuilt from the cached answers; baseline = the candidate list's own top 10; Wilcoxon per user, Bonferroni over 42 tests.

| Test users (p2, N = 10) | NDCG@10 | Coverage | Gini | Novelty | Long-tail share | Genre diversity |
|---|---|---|---|---|---|---|
| Hybrid E | 0.6311 | 11.0% | 0.977 | 0.96 | 0.2% | 0.735 |
| Idea 1: DeepSeek Flash | 0.6169 | **14.4%** | 0.963 | **1.13** | **1.0%** | 0.669 |
| Idea 1: gemma4:e4b | 0.6001 | 14.2% | 0.965 | 1.13 | 0.7% | 0.637 |
| Popularity | 0.5598 | 1.3% | 0.992 | 0.77 | 0% | 0.746 |
| Idea 3: DeepSeek Flash | **0.5999** | **2.6%** | 0.982 | **0.93** | 0% | 0.684 |

- **Within the same 30 candidates the LLM picks less mainstream films:** novelty rises significantly (idea 3: higher for all 814 users; idea 1: 759 of 814), coverage by about 30% (idea 1) or doubles (idea 3, from a very low base).
- **It cannot reach the long tail:** the candidate lists consist almost entirely of popular films, so the long-tail share stays at 0–1%. A long-tail effect requires different candidates, not a better reranker.
- **Genre diversity falls** (−0.06 to −0.10, significant): the LLM's lists follow the genres of the user's seeds more closely, as the similarity methods in experiment 12.
- The same pattern holds on the validation users for DeepSeek with thinking and for GLM-5.3-Flash.

### 6.3 Position bias: averaging three shuffles

DeepSeek Flash (p2, N = 10, 203 validation users) ranked the same 30 candidates in three random orders (shuffle 0 = the order of every earlier run; 812 new answers, all parsable). The three rankings were combined by mean rank.

| | Idea 1 (Hybrid E candidates) | Idea 3 (popularity candidates) |
|---|---|---|
| Candidates' own top 10 | 0.6286 | 0.5626 |
| Shuffle 0 / 1 / 2 | 0.6212 / 0.6218 / 0.6127 | 0.5942 / 0.5934 / 0.5922 |
| **Average of the 3 rankings** | 0.6176 | 0.5988 |
| Average vs shuffle 0 | −0.004 (p = 0.52) | +0.005 (p = 0.20) |
| Position bias (mean of the 3 shuffles) | 0.21 | 0.20 |
| Kendall tau between shuffles | 0.59 | 0.60 |
| Top-10 overlap between shuffles | 76% | 78% |

- **Averaging does not improve accuracy significantly** in either role; the order effect adds noise in both directions rather than a systematic loss.
- **About a quarter of a user's top 10 changes only because the candidates are listed in a different order,** and the mean NDCG of a single shuffle varies by up to 0.009. Differences below about 0.01 between two single-shuffle runs (e.g. the +0.012 thinking gain in §4.2) are therefore within this noise and need confirmation on the test users.
- The conclusions are unchanged: every shuffle is below Hybrid E in idea 1 and above popularity in idea 3 (+0.030 to +0.036).

---

## 7. Findings

1. **No LLM beats collaborative filtering** in any role, with any of 12 models (8 families, 4B to frontier), with or without reasoning, with any prompt or number of seeds. Confirmed on the test users for the best setting (DeepSeek Flash −0.014, gemma −0.031 vs Hybrid E).
2. **An LLM can personalise without rating data.** Reranking a popularity list with 10 seed films, five large models improve on popularity (+0.027 to +0.044 on validation), confirmed on the test users by a pre-registered test (+0.040, p = 1.5 × 10⁻⁶). This reaches 56% of the improvement collaborative filtering achieves, which is relevant for systems without rating data.
3. **The LLM is most useful as a reranker of a good list:** reranking Hybrid E (0.621) > reranking popularity (0.594) > recommending freely (0.574) for the same model and prompt (validation). Collaborative filtering's contribution is choosing which films the LLM sees.
4. **The number of seeds matters more than the model.** With 3 seeds no setup helps; with 10 seeds and the detailed prompt the LLM reaches Hybrid E's level in idea 1 and significantly lifts popularity in idea 3.
5. **The right information helps, more information does not.** Director, cast and a seed profile (p2) help with 10 seeds (+0.027); adding the LLM-generated themes and moods hurts significantly (−0.023 to −0.025), because the model starts matching themes instead of using its broader knowledge of what such users like. The same holds, more weakly, when the LLM reranks popularity (idea 3: −0.014, p = 0.061; the plot recovers part of it, but no richer prompt beats p2, §4.4).
6. **Bigger is not better.** DeepSeek V4 Pro is significantly worse than V4.1 Flash; gpt-oss 120B is not better than 20B; gemma 31B equals 4B. A capable model is needed only when the LLM must personalise on its own (idea 3).
7. **Reasoning helps only when the LLM must infer the taste itself** (idea 3: +0.012 for the same model, p = 0.025), at about 110 times the cost; it does not help when reordering collaborative filtering's list.
8. **Practical behaviour:** position bias explains part of the differences between models in idea 1 (llama3.1:8b keeps 52% of the shown order and is the weakest local model); reasoning and large cloud models produce near-perfect JSON, small local models need repairs in a quarter to half of their answers (always repairable).
9. **LLM reranking makes lists less mainstream but not long-tail,** and less varied in genre: novelty and coverage rise significantly within the candidate list, the long-tail share stays near 0 because the candidates are popular films, and genre diversity falls (§6.2).
10. **A single LLM ranking depends noticeably on the order of the candidates:** a quarter of the top 10 changes between shuffles, but averaging three shuffles does not improve accuracy (§6.3). Differences below about 0.01 between single runs are within this noise.

---

## 8. Limitations and next steps

- **Offline ground truth** (the user's later history) favours popular films and counts unseen good films as misses. In idea 2, 7–16% of suggestions are real films outside the 1,957-film catalogue and can never count as hits.
- **Validation decisions:** models and prompts were chosen on 203 users; only the pre-specified comparisons in §3.5 and §4.3 are confirmatory. The thinking result (§4.2) is exploratory until confirmed.
- **One shuffle per setting** in all runs except §6.3; the shuffle experiment shows that single runs vary by up to about 0.009 NDCG because of the candidate order.
- **Hosted models** may change over time; model names, dates and providers are recorded in the caches.
- **Next:** (1) a second pre-registered test: DeepSeek Flash *with thinking*, idea 3, p2, N = 10, LLM vs popularity on the 814 test users (≈ $5, ≈ 6 hours); (2) learning-to-rank with the LLM rankings as features, now done in experiment 15, where the LLM signals added nothing beyond content features; (3) candidates that include less popular films, if the long-tail share should change (§6.2).

---

## 9. Files

| File | Content |
|---|---|
| `output/experiments/llm_rerank/answers_<provider>_<model>[_think-…][_pN][_nN][_cpop][_sK].jsonl` | cached raw answers per model and setting (prompt version, seeds, popularity candidates, shuffle) |
| `output/experiments/llm_rerank/compare_per_user_<split>_<users>users_<prompt>_n<N>[_cpop].csv` | NDCG, precision, recall per user and model, with Hybrid E and popularity |
| `output/experiments/llm_rerank/compare_summary_…csv` | averages and Wilcoxon tests |
| `output/experiments/llm_rerank/verification_summary.csv`, `verification_summary_idea3.csv`, `verification_*.md/.csv` | independent checks (idea 1 and idea 3); side-by-side top-10 lists |
| `output/experiments/beyond_accuracy_llm/` | beyond-accuracy metrics per user, summary and tests (§6.2) |
| `output/experiments/shuffle_average/` | single-shuffle and averaged NDCG, stability between shuffles, tests (§6.3) |
| `output/experiments/llm_rerank/side_by_side_hybridE_flash_pro_oracle_p2_n10.md` | Hybrid E vs DeepSeek Flash vs V4 Pro vs oracle |
| `output/experiments/llm_generate/answers_*.jsonl`, `per_user_*.csv`, `summary_*.csv`, `tests_*.csv` | idea 2 answers, per-user results, summaries and tests |

---

## 10. Reproducing the runs

```bash
python llm_rerank_experiment.py
```

```bash
python llm_generate_experiment.py test
```

```bash
python verify_llm_rerank.py all
```

```bash
python verify_llm_rerank.py idea3
```

```bash
python beyond_accuracy_llm.py
```

```bash
python shuffle_average_experiment.py
```

The main block of `llm_rerank_experiment.py` selects the run (`compare_models(models, data, split, version, n_seeds, candidates, shuffle)`); cached answers are reused, so rerunning a finished setting makes no model calls. Cloud models need `OLLAMA_API_KEY` in `.env` (Ollama Pro for the larger models); local models need Ollama with the model pulled. `llm_generate_experiment.py test` runs the title-matcher check only; without the argument it runs the configured idea-2 runs.
