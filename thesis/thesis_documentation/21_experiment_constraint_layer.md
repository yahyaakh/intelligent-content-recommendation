# 21 — Constraint layer: E1, quality of the child-safety signals

Status (2026-10-26): E1 done (all films, no users involved). E2–E4 planned (§8).
Script: `safety_labels_experiment.py`; outputs in `output/experiments/safety_labels/`.

## 1. Why

The `ConstraintLayer` in `recomm.py` (line 1637) removes a film for a user under 13 when:

- the film has a genre in {Horror, Thriller, Crime, War}, or
- one of the 172 `CHILD_UNSAFE_KEYWORDS` appears **as a substring** of the film's TMDB keyword text
  (`recomm.py:1712`, `kw in movie_keywords`).

Since then every film has LLM labels: an audience label (children / family / teens / adults; teens =
"13 and up", adults = "17 and up") and five content intensities (violence, sexual content, language,
substance use, frightening scenes; 0 = none, 1 = mild, 2 = moderate, 3 = strong). E1 asks which signal
best finds the films that are unsuitable for a child, before any constraint is built on top of it.

## 2. Ground truths

**A (main) — US certification (MPAA) from TMDB** (`/movie/{id}/release_dates`, US entry, theatrical
release first). 1,360 of the 1,957 films have one: G 116, PG 302, PG-13 369, R 567, NC-17 6; 597 have
none (left out).
- child (under 13): PG-13, R, NC-17 = unsuitable (942 films); G, PG = suitable (418).
- teen (under 17): R, NC-17 = unsuitable (573); G, PG, PG-13 = suitable (787).

**B (second) — MovieLens Tag Genome.** Unsuitable for a child when any content tag has relevance ≥ 0.5:
violence (violence, violent, gratuitous violence, gore, goretastic, bloody, brutal, brutality, torture),
sexual content (11 nudity tags, sex, sexual, sexuality, erotic, rape, sexual abuse, sexualized violence),
language (foul language), substances (drugs, drug abuse, drug addiction), frightening (scary,
disturbing, suicide, suicide attempt). The tag "horror" is deliberately not used, so B does not favour
the genre rule. 1,389 films have genome scores; 763 are unsuitable at 0.5.

A and B agree on only 70.9% of the 985 films that have both. B measures what users tagged, not an
age rating, so it is the weaker ground truth.

## 3. Rules compared (fixed in advance, nothing tuned on the ground truth)

| Rule | Unsuitable for a child when | Unsuitable for a teen when |
|---|---|---|
| genre (recomm.py) | genre in {Horror, Thriller, Crime, War} | same (recomm.py has no teen rule) |
| keywords, substring (recomm.py) | an unsafe keyword inside the keyword text | same |
| keywords, exact | an unsafe keyword equal to one TMDB keyword (substring bug fixed) | same |
| **current** (recomm.py) | genre or keywords-substring | same |
| audience (LLM) | audience = teens or adults | audience = adults |
| intensity (LLM) | any intensity ≥ 2 | any intensity = 3 |
| audience or intensity (LLM) | either | either |

LLM labels: DeepSeek V4.1 Flash (prompt v5, the main label set), gemma4:e4b (v4), gemma4:31b (v2).
gpt-oss:120b (v2) is left out: 129 of its films have no labels, so it can't be compared on the same films.

Metrics for the class "unsuitable": precision, recall, F1. 95% intervals from 2,000 bootstrap
resamples of films (seed 42); the difference to *current* is a paired bootstrap on the same resamples.
"Wrongly blocked" = a suitable film the rule removes; "missed" = an unsuitable film it lets through.

## 4. Results, ground truth A (TMDB certification)

### Child (under 13), 1,360 films, 942 unsuitable

| Rule | Flagged | Precision | Recall | F1 [95% CI] | ΔF1 vs current [95% CI] | Wrongly blocked | Missed |
|---|---|---|---|---|---|---|---|
| genre | 449 | 0.882 | 0.420 | 0.569 [0.537, 0.601] | −0.127 [−0.150, −0.105] | 53 | 546 |
| keywords, substring | 487 | 0.852 | 0.441 | 0.581 [0.550, 0.612] | −0.115 [−0.137, −0.095] | 72 | 527 |
| keywords, exact | 457 | 0.858 | 0.416 | 0.560 [0.529, 0.592] | −0.136 [−0.159, −0.114] | 65 | 550 |
| **current (recomm.py)** | 655 | 0.849 | 0.590 | 0.696 [0.670, 0.724] | — | 99 | 386 |
| **audience — DeepSeek** | 1,073 | 0.850 | **0.968** | **0.905** [0.892, 0.918] | **+0.209** [+0.182, +0.236] | 161 | **30** |
| intensity — DeepSeek | 984 | 0.816 | 0.852 | 0.834 [0.815, 0.852] | +0.138 [+0.112, +0.163] | 181 | 139 |
| audience or intensity — DeepSeek | 1,162 | 0.792 | 0.977 | 0.875 [0.860, 0.890] | +0.178 [+0.151, +0.206] | 242 | 22 |
| audience — gemma4:e4b | 1,122 | 0.817 | 0.973 | 0.889 [0.874, 0.902] | +0.192 [+0.164, +0.218] | 205 | 25 |
| intensity — gemma4:e4b | 613 | 0.896 | 0.583 | 0.706 [0.679, 0.731] | +0.010 [−0.018, +0.035] | 64 | 393 |
| audience or intensity — gemma4:e4b | 1,128 | 0.814 | 0.975 | 0.887 [0.872, 0.901] | +0.191 [+0.163, +0.217] | 210 | 24 |
| audience — gemma4:31b | 925 | 0.897 | 0.881 | 0.889 [0.874, 0.904] | +0.193 [+0.169, +0.218] | 95 | 112 |
| intensity — gemma4:31b | 851 | 0.878 | 0.793 | 0.833 [0.814, 0.852] | +0.137 [+0.112, +0.162] | 104 | 195 |
| audience or intensity — gemma4:31b | 1,001 | 0.865 | 0.919 | 0.891 [0.877, 0.906] | +0.195 [+0.170, +0.221] | 135 | 76 |

### Teen (under 17), 1,360 films, 573 unsuitable

| Rule | Precision | Recall | F1 [95% CI] | ΔF1 vs current [95% CI] | Wrongly blocked | Missed |
|---|---|---|---|---|---|---|
| current (recomm.py) | 0.646 | 0.738 | 0.689 [0.660, 0.719] | — | 232 | 150 |
| **audience — DeepSeek** | 0.798 | 0.899 | **0.846** [0.823, 0.866] | **+0.157** [+0.129, +0.186] | 130 | 58 |
| intensity — DeepSeek | 0.870 | 0.560 | 0.682 [0.646, 0.715] | −0.007 [−0.045, +0.028] | 48 | 252 |
| audience or intensity — DeepSeek | 0.765 | 0.918 | 0.834 [0.811, 0.855] | +0.145 [+0.118, +0.175] | 162 | 47 |
| audience — gemma4:e4b | 0.605 | 0.894 | 0.722 [0.696, 0.747] | +0.033 [+0.006, +0.060] | 334 | 61 |
| audience — gemma4:31b | 0.768 | 0.888 | 0.824 [0.801, 0.846] | +0.135 [+0.107, +0.164] | 154 | 64 |

(Full table, including all rules: `summary.csv`.)

### By popularity third (child, F1; thirds of TMDB vote count, about 453 films each)

| Rule | Least known (median 676 votes) | Middle (2,375) | Most known (10,126) |
|---|---|---|---|
| current | 0.584 | 0.717 | 0.758 |
| genre | 0.449 | 0.575 | 0.649 |
| keywords, exact | 0.409 | 0.558 | 0.663 |
| audience — DeepSeek | **0.842** | **0.909** | **0.954** |
| intensity — DeepSeek | 0.757 | 0.851 | 0.872 |
| audience or intensity — DeepSeek | 0.818 | 0.894 | 0.901 |

Recall of *current* is 0.475 / 0.625 / 0.643; recall of the DeepSeek audience rule is 0.946 / 0.967 / 0.986.

## 5. Results, ground truth B (Tag Genome, child, 1,389 films, 763 unsuitable)

| Rule | Precision | Recall | F1 [95% CI] | ΔF1 vs current [95% CI] |
|---|---|---|---|---|
| current (recomm.py) | 0.737 | 0.727 | 0.732 [0.707, 0.756] | — |
| audience — DeepSeek | 0.635 | 0.971 | 0.768 [0.747, 0.788] | +0.036 [+0.012, +0.061] |
| intensity — DeepSeek | 0.685 | 0.911 | 0.782 [0.761, 0.802] | +0.050 [+0.027, +0.073] |
| audience or intensity — DeepSeek | 0.609 | 0.987 | 0.753 [0.732, 0.774] | +0.021 [−0.003, +0.047] |
| intensity — gemma4:31b | 0.755 | 0.883 | **0.814** [0.794, 0.833] | +0.082 [+0.058, +0.106] |

Sensitivity to the genome threshold (F1):

| Threshold | current | audience — DeepSeek | intensity — DeepSeek |
|---|---|---|---|
| 0.4 | 0.741 | 0.856 | 0.831 |
| 0.5 | 0.732 | 0.768 | 0.782 |
| 0.6 | 0.693 | 0.674 | 0.713 |

Under B the LLM rules are better at 0.4 and 0.5 but not at 0.6, where the genome calls fewer films
unsuitable and the LLMs' high recall turns into low precision. The ranking under B depends on the
threshold; the ranking under A does not depend on any threshold we chose.

### Each intensity against its own genome tags (Spearman, 1,389 films)

| LLM | Violence | Sexual content | Language | Substances | Frightening |
|---|---|---|---|---|---|
| DeepSeek V4.1 Flash (v5) | **0.774** | **0.521** | **0.556** | 0.505 | **0.435** |
| gemma4:31b (v2) | 0.756 | 0.519 | 0.428 | **0.514** | 0.421 |
| gemma4:e4b (v4) | 0.737 | 0.366 | 0.409 | 0.478 | 0.384 |

## 6. Examples (ground truth A, child, most known films first)

- *current* blocks G/PG films: Coco, The Truman Show, Monsters, Inc., The Lion King, Toy Story 3,
  Raiders of the Lost Ark, Tangled, Jaws. The substring bug adds cases like Paperman (G), blocked for
  "rape" because its keyword "skyscraper" contains those letters. 47 films have a substring match
  without any exact keyword match.
- *current* lets PG-13 films through: Interstellar, The Avengers, Avatar, Forrest Gump, Guardians of the
  Galaxy, Avengers: Endgame, The Fellowship of the Ring, Titanic. None has an "adult" genre or keyword.
- The LLM audience-or-intensity rule blocks PG films: the Harry Potter films, Star Wars, Up, Coco,
  Finding Nemo (G). The audience label calls these family or children films; the intensity part of the
  rule blocks them (Philosopher's Stone, Up and Finding Nemo have frightening scenes = 2, Star Wars
  violence and frightening = 2). The intensity part adds 81 G/PG films that the audience rule alone keeps.
- The LLM rule lets PG-13 dramas through: The Terminal, The Blind Side, Gifted, The Artist, King
  Richard, The Chorus. (The reasons behind their ratings were not checked.)

`examples.csv` lists 15 films per case.

## 7. Findings

1. **The LLM audience label finds unsuitable films far better than the recomm.py rule.** Against the
   US certification, DeepSeek's audience label reaches F1 0.905 vs 0.696 for *current* (+0.209,
   95% CI +0.182 to +0.236) at the same precision (0.850 vs 0.849). The difference is recall: *current*
   lets 386 of 942 PG-13/R films through (41%), the audience rule 30 (3%). The same holds for the teen
   threshold (0.846 vs 0.689) and for all three LLMs at the child threshold (0.889–0.905).
2. **It is not only memory of famous films.** An LLM may remember a film's MPAA rating, and F1 is indeed
   highest on the most-known third (0.954). But on the least-known third (median 676 TMDB votes) the
   audience rule still reaches 0.842 vs 0.584 for *current*. These are still films with hundreds of
   votes; for truly obscure films we have no evidence.
3. **The audience label is better than the intensities alone** (0.905 vs 0.834 for DeepSeek, child).
   Adding the intensities to the audience rule raises recall slightly (0.968 → 0.977) and costs
   precision (0.850 → 0.792), so the F1 falls. The small model's intensities (gemma4:e4b) are much
   weaker than its audience label (0.706 vs 0.889).
4. **Fixing the substring bug alone does not help.** Exact keyword matching has slightly lower F1
   than substring matching (0.560 vs 0.581): the substring rule catches some real cases by accident
   ("police brutality" contains "brutality"). The keyword list is the problem, not only the matching.
   The genre rule and the keyword rule each find under half of the unsuitable films.
5. **Under the Tag Genome the picture is weaker and threshold-dependent** (§5). B is a noisier ground
   truth (it agrees with A on 71% of films) and the LLM rules are tuned to the age-rating question.
   Per dimension, the LLM violence intensity tracks the genome well (Spearman 0.77); frightening
   scenes least (0.44).

## 8. What this means for the constraint layer, and what comes next

- Use the **LLM audience label as the hard child/teen filter** (child: remove teens and adults;
  teen: remove adults), with the reason shown. Keep the intensities for user-chosen limits
  ("no strong violence"), not as the default age filter.
- Remove the genre-based age rule. Keep the keyword list only as an extra warning signal, matched exactly.
- E2 (planned): cost of hard constraints in NDCG (validation users with a simulated constraint).
- E3 (planned): the recomm.py soft layer on top of Hybrid E and LTR v2.
- E4 (planned): typed constraints parsed by an LLM into a fixed schema; parse accuracy, satisfaction, NDCG.

## 9. Thesis statements (safe to write)

- "Against the US age certification of 1,360 films, the LLM audience label (DeepSeek V4.1 Flash)
  identified films unsuitable for children under 13 with F1 0.905 (precision 0.850, recall 0.968),
  versus 0.696 (recall 0.590) for the genre-and-keyword rule of the original constraint layer
  (paired bootstrap 95% CI of the difference +0.18 to +0.24)."
- "The advantage held for the least-known third of the films (F1 0.842 vs 0.584), so it is not only
  explained by the LLM remembering ratings of famous films."
- Do **not** write that the LLM labels are correct age ratings: 161 suitable films (12% of flagged) are
  still blocked and 30 unsuitable ones pass; and under the Tag Genome ground truth the advantage depends
  on the threshold.

## 10. Limitations

- 597 films (31%) have no US certification and are not in ground truth A. They are older (median year
  1988 vs 2008) and mostly not in English (70% vs 25%), so the results may not carry over to them.
- MPAA ratings are themselves one country's convention (PG-13 is often given for language alone).
- The LLMs saw the TMDB keywords and overview, the same input as the keyword rule, plus their own knowledge.
- Ground truth B uses a hand-picked tag list and a threshold; §5 shows the result depends on the threshold.

## 11. Reproduce

```
python safety_labels_experiment.py fetch   # TMDB US certifications (free, ~2,000 calls, resumable)
python safety_labels_experiment.py run     # evaluation, no API calls
```

Files: `tmdb_release_dates_us.jsonl` (raw US release entries), `tmdb_certifications.csv`, `summary.csv`,
`by_popularity_third.csv`, `dimension_correlations.csv`, `examples.csv`, `per_film.csv`.
