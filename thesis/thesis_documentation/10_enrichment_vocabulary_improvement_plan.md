# Movie enrichment: vocabulary and input improvement plan

Date: 2026-10-21

Status: proposed experiments; not implemented or evaluated.

## Context

The qualitative review of the first 20 movies in `output/experiments/gemma4_e4b_v4/movies_enriched.csv` identified plausible labels, but also missing central themes, redundant selections, and a mixture of themes, topics, moods, and styles. This was an exploratory review, not a representative accuracy measurement.

The goal is to improve enrichment quality and investigate whether that improves recommendation quality. Better-looking labels alone do not establish better recommendations.

## Recommendation

Keep final assignments restricted to an approved vocabulary. Allow the model to suggest missing concepts in a separate field, subject to human review. Do not automatically add unrestricted suggestions to movie labels or recommendation features.

Preserve the current v4 vocabulary, enriched catalogue, and results as the baseline. Save any revised vocabulary and outputs under a new version.

## Three different problems

| Problem | Example from the review | Proposed response |
|---|---|---|
| A central concept is absent from the vocabulary | Hope is unavailable for The Shawshank Redemption, although the overview mentions it | Review and expand the vocabulary |
| The input lacks important story information | A short Interstellar overview omits the father–daughter relationship | Test a fuller, reliable description |
| An appropriate label and supporting evidence exist, but selection misses it | Inspirational is available but was not selected for Shawshank | Improve selection instructions |

Increasing input length cannot fix an unavailable label. Increasing vocabulary size cannot supply missing story evidence.

## Experiment A: missing-label suggestions

Ask for two separate outputs:

```text
selected_themes: only labels from the approved vocabulary
suggested_missing_themes: concepts supported by the input but absent from that vocabulary
```

Illustrative example, not a measured model response:

```text
selected_themes: friendship, freedom, redemption
suggested_missing_themes: hope
```

For each suggestion, retain a short supporting excerpt or explanation tied to the supplied metadata. Permit an empty suggestions list; do not require the model to invent additions.

Review suggestions across a varied development sample. Merge synonyms such as hope and hopefulness, reject unsupported concepts, and approve useful, clearly defined additions. Frequency can inform review, but should not be the sole criterion: a rare concept may still matter for explicit user requests.

Do not select additions based only on famous films or the first 20 rows. Freeze the revised vocabulary before held-out evaluation.

## Experiment B: better selection instructions

Test revised instructions while keeping the vocabulary, model, and input unchanged:

- Select central story themes, not every topic mentioned.
- Prefer informative selections over closely overlapping labels.
- Return up to five themes, not necessarily five.
- Choose moods that describe the viewing experience.
- Assign labels only when supported by the supplied information.
- Allow an empty list when no available label is supported.

Example issue: mafia, organized crime, and gangsters can consume three slots while providing little additional coverage. They are not universally interchangeable, but selecting all three should require a useful distinction.

The current mood vocabulary also includes style or structure labels such as non-linear. Decide whether to retain a combined mood/style field or separate these categories. Treat that schema change as a separate experiment rather than silently changing the baseline.

## Experiment C: richer input descriptions

Compare the current input with a fuller, reliable description plus relevant keywords and genres. Add meaningful information, not repeated fields or unrelated metadata.

Record the source and version of added descriptions. Keep the same input for all compared models within an experimental condition.

For Shawshank, the existing overview already provides evidence of hope, so richer input is not the primary fix. For Interstellar, additional story information may help identify its family relationship themes.

Allowing a model to use its own knowledge is a different condition from supplying richer metadata. If tested, label it explicitly as knowledge-assisted enrichment and audit factual errors; do not mix it into the metadata-only comparison.

## Suggested controlled comparisons

| Condition | Vocabulary | Selection instructions | Input |
|---|---|---|---|
| Baseline | Existing v4 | Existing | Existing |
| Prompt-only | Existing v4 | Revised | Existing |
| Vocabulary-only | Reviewed revision | Existing, except updated label list | Existing |
| Input-only | Existing v4 | Existing | Richer descriptions |
| Combined, optional | Reviewed revision | Revised | Richer descriptions |

Use the same model and generation settings initially. This makes it easier to identify which change matters. The combined condition alone cannot isolate individual contributions.

## Evaluation plan

### Enrichment quality

Use a varied development sample to refine the approach, then a separate held-out audit sample. Include different genres, popularity levels, and overview lengths. Where practical, assess outputs without showing the condition name.

Record:

- Missing central themes.
- Unsupported or misleading labels.
- Redundant selections.
- Mood appropriateness.
- Invalid labels and empty outputs.

Use a written annotation rubric. Distinguish a label that is plausible from one that is central. Agreement with Tag Genome can provide additional evidence, but is not complete or independent human ground truth for every label.

### Recommendation quality

Keep user splits, candidate pools, relevance definitions, scoring, and evaluation users fixed. Tune any parameters on validation data only. Compare NDCG@10 and other already established metrics on the same users, reporting paired differences and uncertainty where feasible.

Rebuild enrichment-dependent representations for each condition without overwriting baseline caches. Keep embedding and scoring models unchanged for this comparison.

For structured preference matching, examine requests containing newly added concepts. Missing labels can limit exact label matching even when those concepts appear in the original overview. Include independently prepared requests, not only requests designed to favor the additions.

Also record generation time, failures, and cost where applicable.

## Implementation checklist for later

- [ ] Preserve v4 baseline files and results.
- [ ] Select and document a varied development sample.
- [ ] Generate separate missing-label suggestions with supporting evidence.
- [ ] Review, merge, and define approved additions.
- [ ] Freeze a new vocabulary version.
- [ ] Run prompt-only, vocabulary-only, and input-only comparisons.
- [ ] Audit enrichment quality on held-out movies.
- [ ] Run the unchanged recommendation evaluation protocol.
- [ ] Document positive, negative, and inconclusive findings.

Do not manually patch individual evaluation movies and present the result as an automatic enrichment improvement. All suggestions in this document remain hypotheses until tested.
