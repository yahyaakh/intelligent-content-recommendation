# 25 — Constraint layer redesign: what is built, what the LLM adds, what is open

Status (2026-10-28): hard layer rebuilt and working in the CLI (option 13). Soft layer switched off.
Not evaluated yet: E2, E3, E4 (see §5). Code: `recomm.py` (`ConstraintLayer`, `LLMRequestParser`,
`MovieRecommenderCLI.ask_preferences`, option 13); `fetch_credits_full.py`. E1 evidence: doc 21.

## 1. What changed

| Part | Before (original recomm.py) | Now |
|---|---|---|
| Catalog | 10,000 TMDB films | the 1,957 enriched films (same catalog as all experiments) |
| Child filter (under 13) | genre list {Horror, Thriller, Crime, War} + 172 keywords matched as substrings | LLM audience label: keep `children` and `family` |
| Teen filter (13–16) | none | LLM audience label: remove `adults` |
| Unknown age | default 18 ("safest") | default 17 = adult, written down as a choice (18 was not safe: it switches the filter off) |
| "Shares a watched genre" | hard rule | removed (a preference, not a constraint; the ranker already uses genre overlap) |
| Runtime ≤ 1.5 × user average | hard rule | removed (a preference) |
| User requests | none | request dict: audience, include/exclude genres, include/exclude moods, themes, actors, directors, max intensity per dimension |
| Request input | none | menus (code) or one typed sentence (LLM parser); both give the same dict |
| Validation | none | `validate_request`: unknown fields, unknown genre/mood/theme/actor/director, intensity outside 0–3 |
| Cut to top n | before the filter (often fewer than n left) | after the filter, over the whole ranked catalog |
| Removal reasons | printed | returned and printed, e.g. "intensity: violence 1 > 0", "teens audience (children requested)" |
| Actors / directors | top 3 billed / first director | top 10 billed / all directors (new columns `cast_full`, `directors_full`; the models still use the old columns, so earlier results stay valid) |
| Soft layer | re-sorted by its own score, overwriting the model order | switched off (commented), to be redesigned |

Rule: the hard layer only removes films and never reorders them, so the ranking of the model
(Hybrid E, LTR v2, SVD, …) is kept.

## 2. Where the LLM contributes, and how strong the evidence is

| # | Contribution | Evidence | Can be claimed? |
|---|---|---|---|
| 1 | **Audience label as the child/teen filter** | E1 (doc 21): child F1 0.905 vs 0.696 for the original genre + keyword rule, recall 0.968 vs 0.590, CI of the difference +0.18 to +0.24; teen F1 0.846 vs 0.689; holds for the least-known third of films (0.842 vs 0.584) | **Yes**, measured |
| 2 | **Graded content intensities (0–3, five dimensions)** make user-set limits possible ("no sexual content", "not too violent"). The original data had no graded content information, only keywords. | Agreement with Tag Genome tags (Spearman): violence 0.77, language 0.56, sexual content 0.52, substances 0.51, frightening scenes 0.44 (doc 21 §5) | **Partly**: agreement is measured; correctness as a filter is not. Say violence is the most reliable dimension |
| 3 | **Moods and themes** make new request types possible ("something funny", "about friendship") | none for filtering | Only as a capability, not as accuracy |
| 4 | **Free-text requests** turned into the request dict by the LLM (DeepSeek V4.1 Flash, same model as the labels, JSON schema with the allowed values as enums) | working examples only: about 0.7 s and ~1,300 prompt tokens per request | Only as implemented and demonstrated; accuracy needs E4 |
| 5 | **Division of work:** the LLM labels films once, offline, and parses the request; deterministic code does all filtering | design | Yes, as a design choice: filtering is reproducible, free at request time, and every removal has a reason |

Spot check (one user, hybrid, request "children", 20 results, 2026-10-27): 19 of 20 are rated G or PG
in the US; the exception is Big Fish (PG-13), labeled `family`. Anecdotal; the measured number is #1.

## 3. Thesis statements (safe to write)

- "The rule-based child filter of the original system (genre list and keyword matching) was replaced by
  the LLM audience label, which identified films unsuitable for children under 13 with F1 0.905 versus
  0.696 against the US age certification of 1,360 films (doc 21)."
- "LLM content-intensity scores enable user-defined limits that the original metadata could not express;
  the violence score agrees well with the MovieLens Tag Genome (Spearman 0.77), the other dimensions less
  (0.44–0.56)."
- "Requests can be given as free text; an LLM maps them to a fixed schema whose allowed values come from
  the catalog, and the filtering itself is rule-based." (Add parse accuracy only after E4.)
- Do **not** write that the constraints improve recommendation quality (NDCG), user satisfaction, or that
  the parser is accurate: none of these is measured yet.

## 4. Open improvement points (to come back to)

1. **"Similar to X" requests.** "A movie close to The Godfather" is a ranking request, not a filter. The
   parser turned it into genres (Crime or Drama) and themes (mafia, organized crime, family bonds); the
   broad theme let It's a Wonderful Life and The Lion King through, the order stayed the user's usual
   one, and The Godfather itself was recommended. Fix: a `similar_to` field (the LLM extracts the
   title, code matches it to a catalog film, candidates are ranked by content similarity, the film itself
   is excluded, hard filters still apply).
2. **"Any of" vs "all of".** `include_genres` means "at least one", so "crime thriller" becomes Crime *or*
   Thriller. Needs a rule or a field for "all of".
3. **Mood repeated as genre.** "Something funny" became genre Comedy *and* mood funny, which is stricter
   than meant. Prompt rule or post-processing.
4. **Soft layer.** Switched off. It re-sorted by hand-set boosts and overwrote the model order; LTR v2
   already learns director, actor, genre, decade, theme and mood overlap. Options: remove, use as a
   tie-breaker, or turn it into user-requested *preferences* that reorder (like `similar_to`). Test with E3.
5. **Collaborative recommender** only proposes films liked by the 5 most similar users (often ~30 films);
   it is now filled up with popular films. The CLI "hybrid" likewise ranks only 20 films by CF + SVD.
   Neither is the thesis Hybrid E / LTR v2; the CLI should offer those.
6. **NCF** in the CLI still loads the model trained on the old 10,000-film data; retrain on the 1,957 films.
7. **Names** must match TMDB spelling; the parser fixes case only. A "did you mean" hint
   (`difflib.get_close_matches`) would help.
8. **Fields not done:** runtime limit, year range ("recent films"), language. Plain number/code checks;
   the LLM only has to fill them.
9. **Requests the data cannot answer:** "only Oscar winners" (no awards column). Asking the LLM per film
   is not checkable; would need an external source (e.g. Wikidata).
10. **Keywords as warnings:** the old keyword list could stay as an exact-match warning shown next to a
    recommendation, not as a filter (doc 21 §8). Not implemented.

## 5. Evaluations still to run

- **E2:** NDCG@10 cost of each constraint type on the 203 validation users (how often the user's real
  liked film is filtered out).
- **E3:** the old soft layer on top of Hybrid E and LTR v2 (does it help or hurt NDCG?).
- **E4:** about 50 written requests with the correct dict; LLM parser accuracy per field, validation
  rejections, time and tokens; menu as baseline. Include "like X", "any of / all of" and vague requests
  ("something light for date night").
