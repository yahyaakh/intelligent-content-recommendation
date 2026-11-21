# 37 — E4: typed requests to constraints (parser accuracy and constraint satisfaction)

**Status:** complete (2026-10-30). Rules fixed before the run: `output/experiments/constraint_parsing/preregistration.md`
(gold answers checked by the author before the run; nothing changed after it).
**Script:** `constraint_parsing_experiment.py` (modes `check`, `run`).
**Outputs:** `output/experiments/constraint_parsing/` — `requests.json`, `answers.jsonl` (raw LLM answers),
`per_request.csv`, `per_field.csv`, `satisfaction_*.csv`, `run.log`.
**Related:** doc 21 (E1 safety labels), doc 25 (constraint layer, open points).

## 1. Set-up (short)
- 50 typed requests in 8 groups, each with a gold answer. Parser = `recomm.LLMRequestParser`
  (DeepSeek V4.1 Flash via Ollama, temperature 0, JSON schema with the catalog's values), one call each.
- Constraint satisfaction: 20 validation users (10 seeds), Hybrid E's whole ranked pool, top 10 after
  filtering, for the 46 requests with at least one required constraint. Three lists: no constraints,
  LLM-parsed, and the gold request as set in the menu. A separate checker (not `ConstraintLayer`)
  decides whether each shown film meets the request.
- Check: the menu lists satisfy 100 % → the constraint layer and the checker agree. ✔

## 2. Results

### 2.1 Parser accuracy
**43 of 50 requests fully correct (86 %)**; 2 rejected.

| Group | Correct |
|---|---|
| audience | 5 / 6 |
| genre | 8 / 8 |
| mood | 6 / 8 |
| theme | 6 / 6 |
| people | 6 / 6 |
| intensity | 6 / 8 |
| combined | 3 / 5 |
| hard cases | 3 / 3 |

| Field | Requests with this field | Accuracy | Invented (set although not asked) |
|---|---|---|---|
| audience | 8 | 1.00 | 0 |
| include genres | 20 | 0.85 | 1 |
| include moods | 10 | 0.90 | 0 |
| include themes | 10 | 0.90 | 0 |
| exclude genres | 2 | 1.00 | 0 |
| exclude moods | 3 | 0.67 | 3 |
| actors | 3 | 1.00 | 0 |
| directors | 4 | 1.00 | 0 |
| max intensity | 11 | 0.91 | 0 |

(Accuracies include the two rejected requests as wrong.)

**Time and cost:** 0.67 s per request on average (median 0.64 s, max 1.37 s); 1,314 input and 65 output tokens.

### 2.2 The 7 wrong requests

| # | Request | What the LLM did | Effect |
|---|---|---|---|
| 5 | "An animated film for the whole family, my youngest is 5" | genres Animation **or Family** | broader: non-animated family films pass |
| 15 | "Something funny" | mood funny **and genre Comedy** | stricter than asked (doc 25 point 3) |
| 43 | "A funny animated film for my 7 year old, nothing scary" | genres Animation **or Comedy**; also excluded moods scary, frightening | broader on genre; stricter on mood |
| 38 | "Nothing scary, I get nightmares" | correct limit, plus 6 excluded moods | stricter than asked |
| 40 | "No gore and nothing too frightening" | correct limits, plus excluded moods gory, frightening | stricter than asked |
| 20 | "Something light, nothing heavy" | mood "heavy" (not in the list) | **rejected** by `validate_request` |
| 46 | "A war film about survival, not too gory" | intensity dimension "gore" (does not exist) | **rejected** by `validate_request` |

- The JSON schema restricts the values, but the hosted model did not always keep to it. Both invented
  values were caught by `validate_request`, so no wrong filter reached the user: the request is refused
  with a message naming the bad value.
- Three of the five other errors add extra excluded moods: stricter, never wrong films.
- The errors that let wrong films through come from the **"any of" rule for genres**: an extra genre
  widens the filter (5, 43). This is doc 25's open point 2.

### 2.3 Constraint satisfaction (46 requests × 20 users)

| List | Films meeting the request | Lists with fewer than 10 films |
|---|---|---|
| No constraints | 17.5 % | 0 % |
| **LLM-parsed request** | **98.3 %** | 11.4 % |
| Gold request (menu) | 100 % | 6.5 % |

By group (LLM): audience 94.3 %, combined 85.9 %, every other group 100 %.
- Only two requests (5 and 43) produce films that break the request: 71.5 % and 59.0 % of their films meet it.
- Without constraints, 17.5 % of the top-10 films would meet these requests, so the layer is what makes
  the lists fit the request.
- The LLM lists are more often short (11.4 % vs 6.5 %), because of the extra excluded moods.
- Two requests (20, 46) give no list (rejected); they are left out of the 98.3 %.

## 3. Findings
1. **The LLM turns most typed requests into the right constraints:** 86 % fully correct; audience,
   people, themes, genres-only and intensity-only requests almost always right.
2. **The films shown meet the request 98.3 % of the time**, against 17.5 % without constraints and
   100 % with the menu. The filtering itself is exact; every violation comes from a parsing choice.
3. **Errors are mostly on the safe side:** extra exclusions (fewer films, not wrong ones) and invented
   values that validation refuses. The unsafe errors are extra genres under the "any of" rule.
4. **Cheap and fast:** under 1 s and about 1,400 tokens per request, once per request (not per film).

## 4. Limitations
- 50 requests written by the author, in English, one phrasing each; real users write differently.
- One model, one run (temperature 0).
- The gold answers allow several acceptable values for moods and themes; a stricter gold would lower
  the accuracy for those fields.
- Satisfaction uses one recommender (Hybrid E) and 20 validation users; it measures the filter, not
  ranking quality.

## 5. Safe statements for the TDK
- "On 50 typed requests, the LLM parser produced exactly the intended constraints for 43 (86 %); two
  requests were rejected by validation because the model invented a value outside the vocabulary."
- "Across 46 requests and 20 users, 98.3 % of the recommended films met the user's request, against
  17.5 % without constraints and 100 % when the same constraints were set in the menu."
- "The remaining violations came from an extra genre widening an 'any of' genre filter."
- "Parsing took 0.67 s and about 1,400 tokens per request."
- Avoid: "the parser is accurate in general" (50 author-written requests), any claim about real users.

## 6. Possible fixes (not done)
- Genres: treat several genres as "all of" when the request names one kind of film ("animated" →
  Animation only), or ask the model to add a genre only when the request names it.
- Moods: do not add excluded moods when an intensity limit already covers the request ("nothing scary").
- Retry once with the validation message when a value is rejected, instead of refusing the request.
