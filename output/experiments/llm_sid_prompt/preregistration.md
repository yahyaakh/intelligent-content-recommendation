# Pre-registration: semantic-ID group names in the LLM reranking prompt (doc 32)

Written 2026-10-29, **before** any run of `llm_sid_prompt_experiment.py`. Not to be changed after the
first run; later additions go in a new dated section below.

## Question
The LLM reranker's best prompt (p2: title, year, genres, director, cast + a seed profile) costs about
1,612 input tokens per user. Richer prompts (p3, p4: 2,600-4,600 tokens) were not better. Can a SHORT
prompt that adds one semantic-ID group name per film carry useful information at fewer tokens?

## Set-up (as in the p2 runs, doc 17/18)
- Model: deepseek-v4.1-flash via ollama_cloud, temperature 0. Validation users (203), N = 10 seeds,
  Hybrid E's top 30 as candidates, the LLM's own order scored (NDCG@10, plus precision and recall @10).
- **Groups:** the 32 first-level groups of the S4 EASE semantic IDs fitted on all warm users
  (`sid_vector_sources/ids/S4_all-pool_k32_s42.npz`, k-means seed 42; doc 30). EASE groups = films liked
  by the same people, i.e. collaborative information the LLM does not get from titles.
- **Group names:** written once by the same LLM: for each group, its 15 most popular films (title, year)
  and the request for a 2-5 word name of what they share. 32 calls, cached in `group_names.json`.
  Films without an EASE ID (2) get no group.
- **Prompts:**
  - p1 (existing code, unchanged): title, year, genres.
  - **p9 (new):** p1 + "taste group: <name>" after every seed and candidate, and one sentence explaining
    that films in the same taste group tend to be liked by the same people.
  - p2: existing cached answers (3 shuffles), not rerun.
- **Shuffles:** p1 and p9 each with candidate orders 0, 1, 2 (as p2). Per-user NDCG = mean over the 3.
  Single-shuffle differences of about 0.01 are noise (p2: 0.6212 / 0.6218 / 0.6127).

## Hypotheses and rules
- **H1 (does the group name help?):** p9 vs p1, paired Wilcoxon on the 203 per-user means.
  Positive if p9's mean is higher and p < 0.05.
- **H2 (same accuracy for fewer tokens?):** p9 vs p2. Positive if p9 uses fewer input tokens per user AND
  its mean NDCG@10 is at least p2's minus 0.005. Reported with a 95% bootstrap CI of the difference; this
  is a margin check, not a proof of equivalence (203 users).
- Reported, not decided: precision and recall @10, tokens, and the comparison with Hybrid E (0.6286),
  which every LLM order so far has been below.

Test users are not used.

## Outputs (`output/experiments/llm_sid_prompt/`)
`group_names.json`, `per_user.csv`, `summary.csv`, `tests.csv`; answers cached in
`output/experiments/llm_rerank/answers_*` like every other LLM run.

---

## Addition 2026-10-29 (after the group names and the 20-user pilot, before the full run)
- Correction of a fact, not of a rule: 1,527 films have an S4 EASE ID (films never liked by the warm users
  get no EASE vector), not "all but 2". The other films get no taste group in p9.
- Pilot (20 users, shuffle 0, descriptive): p1 0.518, p9 0.489; tokens in 755 / 1,167 (p2: 1,612).
  Nothing was changed after the pilot.
