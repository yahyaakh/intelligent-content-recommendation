# E4 — Typed requests to constraints: parser accuracy and constraint satisfaction

Written 2026-10-30, before any request was sent to the LLM. Descriptive experiment: no hypotheses,
the numbers are reported as they come out. Script: `constraint_parsing_experiment.py`.

## Question
RQ5: can an LLM turn a typed request into constraints that are then enforced reliably?
The filtering is code (`ConstraintLayer.apply_hard_constraints`), so the question splits in two:
1. **Parsing:** does the LLM fill in the right settings?
2. **Satisfaction:** do the films the user finally sees meet what they asked for?

## Requests
`requests.json`: 50 requests written for this test, in 8 groups: audience (6), genre (8), mood (8),
theme (6), people (6), intensity (8), combined (5), hard cases (3: "like The Godfather", "only Oscar
winners", "something good"). Two have no constraint at all (6, 50) and one cannot be expressed (49).

Each request has a gold answer (checked by the author before the run):
- `audience`: "children", "teens" or absent.
- include genres / moods / themes: `any_of` = the values that are acceptable. `optional: true` = leaving
  the field empty is also correct (e.g. a genre the request only implies).
- exclude genres / moods: `all_of` = values that must be excluded, `ok` = values that may also be excluded.
- actors / directors: the exact names.
- `max_intensity`: for each dimension the request mentions, the acceptable limits.

## Scoring (fixed)
**Parser** (model and prompt as in `recomm.LLMRequestParser`: DeepSeek V4.1 Flash, temperature 0, seed 42,
JSON schema; one call per request):
- include field correct: (parsed non-empty and parsed ⊆ any_of) or (optional and parsed empty);
  with no gold spec: parsed empty.
- exclude field correct: all_of ⊆ parsed ⊆ all_of ∪ ok; with no gold spec: parsed empty.
- audience, actors, directors: exact match (empty = absent).
- max_intensity correct: every gold dimension has an acceptable value and no other dimension is set.
- A request is correct when all 9 fields are correct. A request the parser fails on or that
  `validate_request` rejects counts as wrong in every field ("rejected").
- Reported: share of correct requests (all and per group); accuracy per field over the requests whose
  gold has that field; **invented constraints** = fields the parser filled although the gold has no spec
  for them (and they are not optional); rejected requests; time per call (mean, median, max) and tokens.

**Constraint satisfaction:**
- Users: the first 20 validation users (10 seeds, the order of `cs.users_with_n_seeds`). Recommender:
  Hybrid E's whole ranked pool (`thesis_engine.rank_hybrid_e`), so a filter never runs out of films.
- Requests: those whose gold has at least one required (non-optional) constraint.
- Three lists per request and user, top 10 after filtering:
  - **none**: no constraints;
  - **LLM**: the parsed request;
  - **menu**: the gold request as a user would set it in the menu (include = any_of, exclude = all_of,
    intensity = the largest acceptable limit; optional fields left out).
- A film satisfies the request when it meets every required gold constraint. This check is written
  separately from `ConstraintLayer`, so the menu list also tests the layer: it must reach 100 %.
- Reported: satisfaction = satisfying films / films shown, for each list type (all requests and per
  group); share of lists with fewer than 10 films; rejected requests are reported separately (no list).

## Not changed after the run
The requests, gold answers, users and scoring above. If a gold answer turns out to be wrong, it is
corrected in a dated note below and both results are reported.
