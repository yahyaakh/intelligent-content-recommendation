# Experiment: Outside Knowledge, Free-Text Profiles and Dilution

**Status:** finished (run date: 2026-10-20/21, on the corrected half-star ratings)
**Scripts:** `enrich_profiles.py` (profiles), `profile_experiment.py` (evaluation), `dilution_experiment.py` (dilution test); prompt v3 in `llm_enricher.py`
**Results folders:** `output/enrichment/profiles/`, `output/experiments/profiles/`, `output/experiments/dilution/`
**Related documents:** `01_llm_movie_enrichment.md`, `04_experiment_larger_enrichment_model.md`, `05_experiment_scoring_comparison.md`, `06_experiment_cold_start.md`

---

## 1. Questions

Experiments 05 and 06 found that LLM labels (themes and moods) improve ranking only marginally for existing users and not at all for new users. Three explanations were considered:

1. **Redundant information:** the LLM reads the overview and keywords that the recommenders already use, so its labels add little new information.
2. **Limited format:** 44 themes and 18 moods are too coarse to express what the model knows.
3. **Dilution:** the labels are appended as one sentence to a text of about 100 words, so they barely change the embedding.

This experiment tests these explanations with three steps:

| Step | Question |
|---|---|
| A. Prompt v3 pilot | Does allowing the LLM to use its own knowledge change the **labels**? |
| B. Free-text profiles | Does outside knowledge help when the output is a **free-text profile** that can carry it? |
| C. Dilution test | Do the labels or profiles help when used as a **separate, weighted similarity** instead of being appended? |

**Hypotheses for step B, written down before the run:**

- **H1:** profiles written with outside knowledge (P2) beat profiles written from the given text only (P1).
- **H2:** the P2 gain is larger for well-known films (high TMDB vote count).
- **H3:** the P2 gain is larger with the large model (gemma4:31b) than with the small one (gemma4:e4b).
- Secondary: P1 profiles vs the existing labels (free text vs fixed labels; format and kind of information change together).

---

## 2. Step A: outside knowledge with fixed labels (prompt v3 pilot)

**Design.** Prompt v3 differs from v2 in one sentence of the system prompt: "Base your labels on the movie information provided AND on your own knowledge of this film … If you do not recognise the film, use only the information provided." Lists, schema, input and settings are identical. Prompt versions are now a parameter of `LLM_MovieEnricher` (default v2), so all earlier code is unchanged.

**Pilot:** 20 movies (13 well-known, 7 with about 350–400 TMDB votes). For each model, v2 was re-run to measure run-to-run noise, and v3 was compared with the saved v2 labels.

| | gemma4:e4b | gemma4:31b-cloud |
|---|---|---|
| v2 re-run identical to the saved v2 labels | 19 of 20 | 15 of 20 |
| Theme overlap v3 vs v2 (Jaccard, mean) | 0.93 | 0.90 |
| Mood overlap v3 vs v2 | 0.92 | 0.95 |
| Movies with any theme change | 5 of 20 | 7 of 20 |

**Result.** Allowing outside knowledge changes the labels about as much as re-running the same prompt (for 31b, 5 of 20 movies already differ between two identical v2 runs). The models seem to use their knowledge of well-known films already, and the fixed vocabulary cannot express more of it. No full v3 run was made, because its labels would be about 90% identical to v2 by construction.

---

## 3. Step B: free-text profiles with and without outside knowledge

### 3.1 Design

| Setting | Value |
|---|---|
| Output | one field `{"profile": "..."}`: 2–3 sentences, at most 60 words, on what makes the film distinctive (tone and atmosphere, style and pacing, how the story is told); no plot retelling, no other film titles |
| **P1** | "Use ONLY the information given below. Do not add facts that are not stated in it or directly implied by it." |
| **P2** | "Use the information given below AND your own knowledge of this film, for example its full story, its style and how it is generally received. If you do not recognise the film, use only the information given." |
| Input | identical to the labels: title, year, genres, keywords, overview |
| Models | gemma4:e4b (local) and gemma4:31b-cloud; temperature 0, seed 42, thinking off, JSON schema |
| Use | base text + " Profile: …", embedded with embeddinggemma exactly like the labels |
| Evaluation | the same users, split, pool and metrics as experiments 05 (warm) and 06 (cold); cold hybrids with α chosen on the validation users for P1, and P2 and the baselines evaluated at the **same α** |

A pilot on 20 movies led to one design change before the full run: an instruction "describe what kind of viewer it appeals to" was removed, because it produced the same formulaic final sentence in every profile.

**Pilot observations (20 movies).** With P2, gemma4:31b added accurate information that is not in the input, e.g. *Whiplash*: "a visceral editing style that mirrors the intensity of a jazz performance"; *Requiem for a Dream*: "frenetic, rhythmic editing style and surreal imagery"; *Faust* (1926): "expressionist visuals and haunting, atmospheric lighting". gemma4:e4b P2 was more fluent but mostly generic ("deliberate, stately pacing"). For the low-vote films, P2 added plausible but unverifiable generic phrases ("slow, intimate pacing", "understated style"). P1 profiles mainly paraphrased the keywords and overview.

### 3.2 Runs

| Model / variant | Valid | Mean words | Seconds per movie |
|---|---|---|---|
| gemma4:e4b P1 | 1,957 / 1,957 | 42.8 | 1.32 |
| gemma4:e4b P2 | 1,957 / 1,957 | 43.9 | 1.45 |
| gemma4:31b P1 | 1,957 / 1,957 | 36.9 | 1.26 |
| gemma4:31b P2 | 1,957 / 1,957 | 39.3 | 1.75 |

Appending a profile changes a movie's embedding very little: the cosine between base and appended vectors is 0.990 (e4b P1), 0.987 (e4b P2), 0.991 (31b P1) and 0.986 (31b P2). Profile lengths differ slightly between conditions (31b shorter than e4b; P2 1–2 words longer than P1).

### 3.3 Warm start (10,490 users, NDCG@10)

| Representation | Max scoring | Mean scoring |
|---|---|---|
| No enrichment | 0.3152 | 0.2732 |
| e4b labels (experiment 05) | **0.3186** | 0.2892 |
| e4b profile P1 | 0.3151 | 0.2847 |
| e4b profile P2 | 0.3130 | 0.2979 |
| 31b profile P1 | 0.3148 | 0.2898 |
| 31b profile P2 | 0.3155 | **0.3078** |
| Popularity | 0.4214 | |

Wilcoxon signed-rank tests (NDCG@10 per user, b − a; Bonferroni over 32 tests):

| Comparison | Max scoring | Mean scoring |
|---|---|---|
| **H1** e4b: P2 vs P1 | −0.0021 (sig.) | **+0.0132** (sig.) |
| **H1** 31b: P2 vs P1 | +0.0007 (sig.) | **+0.0180** (sig.) |
| **H3** (31b P2−P1) vs (e4b P2−P1) | +0.0028 (sig.) | +0.0048 (sig.) |
| 31b P2 vs no enrichment | +0.0003 (n.s.) | +0.0346 (sig.) |
| e4b P2 vs no enrichment | −0.0022 (sig.) | +0.0247 (sig.) |
| Secondary: e4b P1 profile vs e4b labels | −0.0035 (sig.) | −0.0045 (sig.) |
| 31b P2 profile vs e4b labels | −0.0031 (sig.) | +0.0186 (sig.) |

**H2: P2 − P1 by popularity of the relevant movies.** The same top-10 lists are scored only against the relevant movies in one third of the candidate pool by TMDB vote count (low ≤ 845 < mid ≤ 3,172 < high):

| | Third | e4b P2 − P1 | 31b P2 − P1 |
|---|---|---|---|
| Mean scoring | low | +0.0001 (n.s.) | −0.0001 (n.s.) |
| | mid | −0.0013 (sig.) | +0.0014 (sig.) |
| | high | **+0.0142** (sig.) | **+0.0170** (sig.) |
| Max scoring | low | −0.0000 (n.s.) | +0.0010 (sig.) |
| | mid | −0.0005 (sig.) | +0.0012 (sig.) |
| | high | −0.0017 (sig.) | −0.0006 (n.s.) |

Relevant movies in the low third are rarely recommended by any method (NDCG@10 about 0.004–0.011), so a gain there would be hard to detect.

### 3.4 Cold start (814 test users, NDCG@10)

| Method | N = 1 | N = 3 | N = 5 | N = 10 |
|---|---|---|---|---|
| Pure, no enrichment (mean) | 0.2922 | 0.3798 | 0.4124 | 0.4540 |
| Pure, e4b labels | 0.2922 | 0.3820 | 0.4147 | 0.4576 |
| Pure, e4b P1 / P2 | 0.2867 / 0.2886 | 0.3773 / 0.3783 | 0.4072 / 0.4105 | 0.4516 / 0.4554 |
| Pure, 31b P1 / P2 | 0.2865 / **0.2932** | 0.3726 / **0.3849** | 0.4048 / **0.4166** | 0.4520 / **0.4633** |
| Hybrid, no enrichment (31b's α) | 0.5697 | 0.5778 | 0.6001 | 0.6135 |
| Hybrid, 31b P1 / P2 (same α) | 0.5686 / 0.5708 | 0.5767 / 0.5772 | 0.5958 / 0.5974 | 0.6103 / 0.6122 |
| Hybrid, e4b P1 / P2 (same α) | 0.5692 / 0.5710 | 0.5765 / 0.5767 | 0.5976 / 0.5985 | 0.6091 / 0.6121 |
| Popularity | 0.5598 | 0.5598 | 0.5598 | 0.5598 |

- **Pure embeddings, 31b P2 vs P1:** +0.007 to +0.012 at every N, significant after correction.
- **Pure embeddings, e4b P2 vs P1:** +0.001 to +0.004, not significant.
- **Hybrids (same α):** P2 vs P1 and P2 vs no enrichment differ by at most 0.003 at every N; none is significant after correction.

---

## 4. Step C: dilution test (appended vs separate weighted similarity)

### 4.1 Design

- The extra text (the e4b label sentence, or one of the four profiles) is embedded **on its own** with embeddinggemma.
- Similarity = **w × base + (1 − w) × extra**, w ∈ {1.0, 0.9, …, 0.0}.
- **Warm:** w chosen on a random 20% of the warm users (2,098; seed 42), separately per extra and scoring; all methods reported on the other 80% (8,392 users).
- **Cold:** w and the hybrid α chosen together on the 203 validation users, per extra and N; reported on the 814 test users, compared with the base hybrid and the appended hybrid at the same α.
- Ties keep the larger w (closer to the base similarity).

### 4.2 Are the extras different information?

| Extra | Correlation with base similarities: extra embedded alone | appended to the base text |
|---|---|---|
| e4b labels | 0.36 | 0.99 |
| e4b P1 | 0.48 | 0.98 |
| e4b P2 | 0.43 | 0.98 |
| 31b P1 | 0.49 | 0.99 |
| 31b P2 | 0.40 | 0.98 |

Embedded alone, the extras rank movie pairs quite differently from the base text; appended, they hardly change it. Dilution therefore exists at the level of the similarities.

### 4.3 Warm start (8,392 test users, NDCG@10)

| Extra | Max: appended | Max: blended (chosen w) | Mean: appended | Mean: blended (chosen w) |
|---|---|---|---|---|
| none (base) | 0.3138 | — | 0.2742 | — |
| e4b labels | **0.3171** | 0.3159 (0.9) | 0.2902 | 0.2795 (0.8) |
| e4b P1 | 0.3136 | 0.3162 (0.6) | 0.2855 | 0.2766 (0.8) |
| e4b P2 | 0.3114 | 0.3138 (1.0) | 0.2988 | 0.2797 (0.8) |
| 31b P1 | 0.3135 | 0.3138 (1.0) | 0.2908 | 0.2837 (0.9) |
| 31b P2 | 0.3140 | 0.3138 (1.0) | **0.3086** | 0.2874 (0.8) |
| Popularity | 0.4218 | | | |

- **Max scoring:** for e4b P2, 31b P1 and 31b P2 the validation users chose **w = 1.0**, i.e. ignoring the profile. Blending gained over the base only for the labels (+0.0021) and e4b P1 (+0.0024), both significant, and neither beats the appended labels (0.3171).
- **Mean scoring:** blending beats the base (+0.003 to +0.013), but **appending is better than blending** for every extra (by 0.007 to 0.021, significant).

### 4.4 Cold start (814 test users)

Blended hybrids differ from the base hybrid and from the appended hybrid at the same α by at most 0.008 NDCG@10 at every N; none of the 40 comparisons is significant after correction. In several cases the validation users chose w = 1.0.

---

## 5. Findings

1. **Outside knowledge does not change fixed labels.** With prompt v3, the labels change about as much as between two identical runs. The vocabulary limits how much of the model's knowledge can be expressed.
2. **In free text, outside knowledge adds real information (H1 supported where the setup is weak).** P2 profiles beat P1 profiles under mean scoring (+0.013 e4b, +0.018 31b) and, for gemma4:31b, in pure cold-start embeddings at every N (+0.007 to +0.012). Under max scoring and in the hybrids, the effect is at most about 0.002.
3. **The large model contributes more knowledge (H3 supported).** The P2 gain is larger for gemma4:31b in every setting. With gemma4:e4b, "use your own knowledge" even reduces quality under max scoring (−0.0021): the small model adds generic description rather than knowledge.
4. **The knowledge gain comes from well-known films (H2 supported, with a limitation).** Under mean scoring, the P2 gain comes almost entirely from relevant movies in the most popular third. The least popular third is rarely recommended by any method, so a gain there cannot be measured well.
5. **Free-text paraphrase is worse than fixed labels.** Without new knowledge, P1 profiles score below the e4b labels (−0.0035 max, −0.0045 mean).
6. **Dilution exists, but is not the bottleneck.** Embedded alone, profiles and labels carry information that differs from the base text (correlation 0.36–0.49). But using them as a separate weighted signal does not improve the best configurations: under max scoring the validation users mostly chose to ignore the profiles, in cold start no blended hybrid differs significantly, and under mean scoring appending works better than blending.
7. **None of the variants beats the best configurations.** The best warm result remains embeddinggemma with appended e4b labels and max scoring (0.3186 on all users); in cold start, item-CF + popularity (experiment 06) remains best. Popularity remains far ahead offline (0.4214).

**Overall interpretation.** The LLM can add genuinely new, accurate information about well-known films (style, tone, reception), and this information is different from the movie text. It still does not predict which movies a user will rate highly, beyond what the movie text and popularity already capture. The weak effect of enrichment is therefore not mainly a problem of redundant input, label format or dilution: on MovieLens, ranking accuracy is driven by what people like (popularity and collaborative signals) more than by finer descriptions of the films.

---

## 6. Limitations

- **One embedding model** (embeddinggemma) and **one way of combining** (append or linear blend); a trained model using the extras as features (e.g. a learned ranker) was not tested.
- **Profile quality was not checked against ground truth.** The pilot suggests accurate additions for well-known films and unverifiable generic phrases for obscure ones; hallucination was not measured systematically.
- **Profile length differs slightly between conditions,** which can affect embeddings independently of content.
- **Run-to-run variation of the LLMs** (up to 5 of 20 pilot movies for gemma4:31b) means part of any label difference is noise.
- **The candidate pool is the popular part of TMDB** (at least 301 votes), which favours models that know well-known films; on a less popular catalog, outside knowledge may help less and hallucinate more.
- **Offline evaluation on MovieLens** rewards popularity (`03_related_work.md`); benefits of richer descriptions for users (explanations, typed requests, discovery of less popular films) are not captured by NDCG.
- **Many comparisons** were made; Bonferroni correction was applied within each experiment, and all variants are reported, including those without an effect.

---

## 7. Consequences and next steps

- For **ranking accuracy**, further variations of the enrichment text are unlikely to change the conclusions of experiments 05–07.
- The most promising remaining uses of LLM-generated descriptions are those that **offline NDCG does not capture**: typed natural-language requests (matching "a slow-burn, atmospheric drama" against P2-style profiles), explanations, and constraint handling.
- A **learned ranker** (LightGBM) with the extras as features is the remaining way to test whether the different information can be exploited by a trained model.

---

## 8. Files

| File | Content |
|---|---|
| `output/enrichment/profiles/gemma4_<model>_<P1/P2>.jsonl` / `.csv` | profiles per movie (cache with timings and tokens; one row per movie) |
| `output/experiments/profiles/movie_texts_profiles.csv` | base texts and the four appended texts |
| `output/experiments/profiles/movies_profile_*_embeddinggemma.*` | indexes of the appended texts |
| `output/experiments/profiles/warm_per_user.csv`, `warm_summary.csv`, `warm_significance_tests.csv` | warm results (including the popularity-third scores) |
| `output/experiments/profiles/cold_per_user.csv`, `cold_summary.csv`, `cold_significance_tests.csv` | cold results |
| `output/experiments/dilution/extra_only_*` | extra-only texts and indexes |
| `output/experiments/dilution/warm_*`, `cold_*`, `settings.json` | dilution results, the weight validation curve and the chosen w and α |

---

## 9. Reproducing the runs

```bash
python enrich_profiles.py gemma4:31b-cloud
```

```bash
python enrich_profiles.py gemma4:e4b
```

```bash
python profile_experiment.py
```

```bash
python dilution_experiment.py
```

The profile runs take about 1.5 hours per model (both variants); the two evaluation scripts take about 15–30 minutes each. All random choices use seed 42. The prompt v3 pilot can be reproduced with `LLM_MovieEnricher(..., prompt_version="v3")`.
