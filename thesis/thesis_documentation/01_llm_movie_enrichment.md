# LLM Movie Enrichment

**Status:** finished (run date: 2026-10-15)
**Scripts:** `enrich_catalog.py`, `llm_enricher.py`
**Output:** `output/enrichment/gemma4_e4b_v2.jsonl`

---

## 1. Purpose

The thesis asks whether LLM enrichment improves recommendations. This step produces the enriched data that the later experiments compare against the non-enriched baseline.

For every movie in the catalog, a local LLM reads the existing metadata and assigns:

| Field | Type | Allowed values |
|---|---|---|
| `themes` | 1–5 labels, most important first | fixed list of 44 themes |
| `mood` | 1–3 labels | fixed list of 18 moods |
| `audience` | 1 label | children, family, teens, adults |
| `content_intensity` | 5 integer scores, 0–3 | violence, sexual_content, language, substance_use, frightening_scenes |

This covers the task description items **content-feature enrichment** and **text classification**. Themes and mood are meant for the recommenders (content-based and embedding-based). Audience and intensity are meant for the constraint layer.

---

## 2. Data

| Item | Value |
|---|---|
| Catalog file | `data/data/processed/movies.csv` |
| Movies | 1,957 (unique `movie_id`) |
| Relation to the 10k catalog | identical to the first 1,957 rows of `data/data/processed_30k/movies.csv` |
| Movies with ratings in `processed_30k/ratings.csv` | 1,549 (about 2.55 M ratings); 408 have no ratings |
| Missing overview / keywords | 0 / 50 |
| Popularity | TMDB vote count ≥ 301, median 1,318 (full 10k catalog: median 670) |

**Why 1,957 movies and not 10,000:** comparing several enrichment models and settings under identical conditions had to fit a laptop GPU and free API tiers. The chosen model enriches 1,957 movies in about 1 hour; the full catalog with a reasoning-enabled or larger model would take 25–35 hours locally, and exceeds free hosted-API limits. The existing embeddinggemma index also covers exactly these 1,957 movies.

**Limitation:** this subset contains more popular, highly rated films (many classics and anime). LLMs know popular films better, so label quality here may be higher than for an average catalog.

---

## 3. Model selection

### 3.1 Hardware

| Component | Value |
|---|---|
| GPU | NVIDIA RTX 4050 Laptop, 6 GB VRAM |
| RAM | 15 GB |
| Runtime | Ollama 0.40.2 |

Models up to about 4–5 GB fit fully in GPU memory; 7–9B models run partly on the CPU and are 3–6× slower.

### 3.2 Comparison setup

- **20 test movies**: 15 well-known films (crime, war, horror, family, romance, drugs, music, comedy) and 5 obscure films.
- **Same prompt (v1), schema and input text** for every model; temperature 0, seed 42; thinking disabled unless stated.
- **Scoring:** reference answers written by hand for the 15 well-known films (key-theme recall, family-film safety, R-rated → adults, 12 clear intensity cases, average language score on R-rated films). These references are the author's judgement, not ground truth; a proper audit against external sources is still planned (section 8).
- Result files: `output/llm_model_comparison_all.csv`, `output/llm_model_comparison_think_off_rerun.csv`, `output/llm_model_comparison_hosted.csv`.

### 3.3 Results

| Model | Where | Sec/movie | Key themes (of 25) | Family safe (of 4) | R → adults (of 10) | Strong intensity (of 12) | Avg. language (R films) |
|---|---|---|---|---|---|---|---|
| llama3.2:3b | local | 1.6 | 18 | 3 | 10 | 7 | 0.1 |
| phi4-mini | local | 2.0 | 18 | 4 | 10 | 8 | 0.6 |
| gemma3:4b | local | 2.0 | 24 | 4 | 10 | 8 | 0.9 |
| **gemma4:e4b** | **local** | **1.5** | **24** | **4** | **10** | **9** | **1.0** |
| gemma4:e4b + thinking | local | 12.5 | 24 | 4 | 10 | 11 | 1.9 |
| qwen3.5:4b | local | 2.2 | 23 | 3 | 9 | 7 | 1.5 |
| qwen2.5:7b | local | 4.6 | 20 | 4 | 10 | 7 | 0.2 |
| llama3.1:8b | local | 5.5 | 23 | 3 | 10 | 11 | 0.9 |
| granite4.2:8b | local | 7.1 | 19 | 4 | 10 | 4 | 0.6 |
| ministral-3:8b | local | 8.4 | 24 | 4 | 10 | 8 | 1.3 |
| qwen3.5:9b | local | 9.0 | 23 | 3 | 10 | 11 | 1.7 |
| gpt-oss-20b | Groq | 0.7* | 25 | 4 | 10 | 9 | 1.5 |
| qwen3.8-27b | Groq | 0.5* | 23 | 3 | 9 | 12 | 1.8 |
| gpt-oss-120b | Groq | 1.1* | 23 | 3 | 10 | 11 | 2.4 |
| gpt-oss-120b | Ollama cloud | 3.6* | 24 | 4 | 10 | 11 | 2.3 |

\* Hosted times depend on the provider's hardware and network and are not comparable with local times.

All models returned 20/20 valid answers with thinking disabled. Gemini (2.5 Flash retired; 3.8 Flash overloaded and over its free quota after 3 movies; 3.1 Pro not on the free tier) and Llama 3.3 70B (no longer offered on Groq) could not be evaluated.

### 3.4 Findings

1. **Theme quality plateaus at small model sizes.** 4B models (gemma3, gemma4) reach the same key-theme recall (24/25) as gpt-oss-120b. A larger model gave no gain on themes.
2. **Intensity improves with model size or reasoning.** gpt-oss 20B → 120B: intensity 9 → 11, language 1.5 → 2.4. gemma4 without → with thinking: 9 → 11, language 1.0 → 1.9.
3. **Thinking mode is costly.** gemma4:e4b with thinking: 8× slower (12.5 s, 915 output tokens vs 91). qwen3.5:4b with thinking: 13 of 20 answers broke (the reasoning filled the 4,096-token context), 61 s per movie.
4. **Audience labels are unreliable across models.** qwen3.8-27b had the best intensity (12/12) but labelled R-rated films "teens"; most local models labelled *The Notebook* (PG-13) "adults".
5. **All models underestimate language**, because plot summaries rarely mention swearing.

### 3.5 Choice

**gemma4:e4b, thinking off**: best theme quality together with the fastest local speed, fully on the GPU, free and reproducible. Audience and intensity from the LLM are treated as a fallback; the plan is to take audience from official TMDB certifications.

---

## 4. Prompt (v2)

### 4.1 Changes from v1

v1 was used in the model comparison. v2 was written from the comparison's errors and used for the full run.

| Change | Reason |
|---|---|
| "crime and investigation" split into "crime and criminals" and "detective investigation" | The old label was used for *The Godfather* and *Pulp Fiction*, which contain no investigation |
| `uniqueItems` and `minItems: 1` in the schema | Repeated themes (llama3.2) and empty lists |
| "Most movies need 2 to 4 themes", "most important first" | Padding to 5 themes (qwen, ministral) |
| Broad themes only when central | "family relationships" was the most used theme for nearly every model |
| "Precision matters more than coverage" (system prompt) | Same padding problem |
| Audience rule tied to strong content | R-rated films labelled "teens" |
| Hint that summaries rarely mention swearing, nudity or drugs | All models underestimated language |

### 4.2 System prompt

```text
You are a film metadata annotator. You label movies using ONLY the allowed values you are given, spelled exactly as written. You base your labels on the movie information provided. Precision matters more than coverage: a short list of clearly correct labels is better than a long list with weak ones. You always return only valid JSON, with no explanation or extra text.
```

### 4.3 User prompt

`{input_text}`, `{themes}` and `{moods}` are filled in per movie.

```text
Annotate the movie below.

MOVIE
{input_text}

FIELDS TO PRODUCE

1. themes: choose 1 to 5 values from THEME LIST, most important first.
   - Choose only themes that drive the main story. Ignore minor details and subplots.
   - Most movies need 2 to 4 themes. Use 5 only when all five are clearly central.
   - Do not use a theme just because it matches the genre.
   - Use the broad themes "family relationships", "good versus evil", "loss and grief" and "identity and self-discovery" only when they are a main focus of the story, not when they are merely present.
   - "crime and criminals" is for stories about committing crimes or living a criminal life. "detective investigation" is only for stories centred on solving a crime or mystery.
   - Never repeat a theme.

2. mood: choose 1 to 3 values from MOOD LIST that describe how the movie feels to watch. Never repeat a mood.

3. audience: choose exactly one of: children, family, teens, adults.
   - children: made for young children
   - family: suitable for all ages, enjoyable for adults too
   - teens: suitable for ages 13 and up
   - adults: intended for ages 17 and up
   If a movie contains strong violence, sexual content, drug use or frequent strong language, choose adults.

4. content_intensity: give an integer from 0 to 3 for each of: violence, sexual_content, language, substance_use, frightening_scenes.
   - 0 = none
   - 1 = mild or brief
   - 2 = moderate or repeated
   - 3 = strong, graphic, or central to the story
   Plot summaries rarely mention swearing, nudity or drug use. Estimate them from the kind of story described: for example, war, crime and gangster stories usually contain frequent strong language. If unclear, give a cautious estimate (when in doubt, choose the higher value).

THEME LIST
{themes}

MOOD LIST
{moods}

OUTPUT FORMAT
{"themes": [...], "mood": [...], "audience": "...", "content_intensity": {"violence": 0, "sexual_content": 0, "language": 0, "substance_use": 0, "frightening_scenes": 0}}
```

### 4.4 Input text per movie

Built by `LLM_MovieEnricher.build_input_text`. Empty genres or keywords are left out.

```text
Title: The Matrix (1999)
Genres: Action|Sci-Fi
Keywords: man vs machine, martial arts, kung fu, dreams, ...
Overview: Set in the 22nd century, The Matrix tells the story of ...
```

### 4.5 Vocabularies

**Themes (44):** love and romance, friendship, family relationships, coming of age, revenge, betrayal, redemption, survival, identity and self-discovery, loss and grief, good versus evil, power and corruption, justice and the law, crime and criminals, detective investigation, heist, war and its consequences, political conflict, rebellion against authority, freedom and oppression, artificial intelligence and technology, space exploration, time travel, dystopian society, supernatural forces, magic, monsters and creatures, adventure and quest, sports and competition, music and performance, art and creativity, money and ambition, social class and inequality, racism and discrimination, mental illness, addiction, faith and religion, historical events, nature and environment, school life, journey and road trip, isolation and loneliness, sacrifice, deception and conspiracy

**Moods (18):** dark, tense, suspenseful, scary, funny, lighthearted, heartwarming, uplifting, romantic, sad, emotional, thought-provoking, inspiring, action-packed, whimsical, nostalgic, gritty, epic

### 4.6 Output schema

The answer is constrained by a JSON schema passed to Ollama's `format` field: allowed values as enums, `minItems`/`maxItems`/`uniqueItems` for themes (1–5) and mood (1–3), audience as an enum, and each intensity score an integer from 0 to 3. All four fields are required.

---

## 5. Run parameters

| Parameter | Value |
|---|---|
| Model | `gemma4:e4b` (Ollama) |
| Prompt version | v2 |
| Thinking | off (`"think": false`) |
| Temperature | 0 |
| Seed | 42 |
| Output format | JSON schema (section 4.6) |
| Context window | 4,096 tokens (Ollama default) |
| `keep_alive` | 10 min |
| Timeout / retries | 300 s / 2 retries |
| Ollama address | `http://127.0.0.1:11434` |

**Note on the address:** on Windows, `localhost` first tries IPv6 and adds about 2 s to every request. Using `127.0.0.1` reduced the real time per movie from 3.8 s to about 1.0–1.5 s.

---

## 6. Script design

### 6.1 `llm_enricher.py`

Holds the prompt, vocabularies, schema and `LLM_MovieEnricher`:
- `get_movie`, `build_input_text`, `build_prompt`
- `call_llm`, which sends the request to local Ollama (`_call_ollama`) or to an OpenAI-compatible hosted API (`_call_hosted`: Groq, Gemini) and returns status, content, time and token counts

All scripts import the prompt from this one file, so the prompt the model sees, the version saved with the results and the lists used for validation always match.

### 6.2 `enrich_catalog.py`

| Function | Job |
|---|---|
| `cache_path(model_name, prompt_version)` | Builds `output/enrichment/<model>_<version>.jsonl`; `:` and `/` replaced with `_` |
| `load_finished(path)` | Reads the cache; returns the IDs whose last record has `status == "ok"`; skips broken lines |
| `append_record(path, record)` | Appends one JSON line (UTF-8) per movie, immediately after it is processed |
| `clean_labels(...)` | Keeps only allowed labels, removes duplicates, cuts to the maximum count, records each fix |
| `validate_output(output)` | Cleans themes and mood, checks audience, converts and clamps intensity to 0–3 (missing → `None`, never 0) |
| `enrich_movies(enricher, movies_df, path, limit)` | Main loop: skip finished movies, call the LLM, parse, validate, save |

**Status rules:** `failed` (call failed), `invalid_json` (unparseable answer), `invalid` (no theme left after cleaning), otherwise `ok`. Repairs set `repaired = true` and keep `status = ok`.

**Resume:** an interrupted run continues where it stopped; failed and invalid movies are retried on the next run.

### 6.3 Record format (one line per movie)

An actual record from the run (records written after the `wall_seconds` change also contain `"wall_seconds"`):

```json
{"movie_id": 238, "title": "The Godfather", "model": "gemma4:e4b", "prompt_version": "v2",
 "seconds": 1.65, "prompt_tokens": 912, "output_tokens": 91,
 "themes": ["crime and criminals", "family relationships", "power and corruption", "revenge", "betrayal"],
 "mood": ["dark", "tense", "gritty"], "audience": "adults",
 "content_intensity": {"violence": 3, "sexual_content": 0, "language": 2, "substance_use": 2, "frightening_scenes": 0},
 "issues": [], "repaired": false, "status": "ok"}
```

---

## 7. Results

### 7.1 Run

| Measure | Value |
|---|---|
| Movies enriched | 1,957 / 1,957 |
| Status | 1,957 ok · 0 failed · 0 invalid |
| Repairs by validation | 0 |
| Total time | 1.0 hour |
| Time per movie | 1.71 s model · 1.84 s real (first ~100 movies ran before the `127.0.0.1` fix) |
| Tokens | 1,747,641 in · 177,570 out (about 893 in / 91 out per movie) |

### 7.2 Label distribution

**Themes per movie:** 2: 7 · 3: 95 · 4: 1,046 · 5: 809 (average 4.36). All 44 themes were used.

| Most common themes | Share of movies |
|---|---|
| identity and self-discovery | 46% |
| good versus evil | 26% |
| survival | 22% |
| crime and criminals | 20% |
| family relationships | 19% |
| deception and conspiracy | 19% |
| loss and grief | 18% |
| love and romance | 18% |

Rarest: space exploration (27 movies), heist (30), time travel (44), addiction (58), magic (58).

**Moods per movie:** 3 for 1,940 movies, 2 for 17. Most common: tense 44%, dark 36%, thought-provoking 24%, emotional 23%, gritty 20%. Rarest: scary 3%.

**Audience:** adults 1,258 (64%) · teens 389 · family 299 · children 11

**Content intensity:**

| Dimension | 0 | 1 | 2 | 3 | Mean |
|---|---|---|---|---|---|
| violence | 853 | 370 | 488 | 246 | 1.06 |
| sexual_content | 1,700 | 172 | 66 | 19 | 0.18 |
| language | 1,118 | 522 | 308 | 9 | 0.60 |
| substance_use | 1,523 | 306 | 96 | 32 | 0.30 |
| frightening_scenes | 1,072 | 536 | 281 | 68 | 0.67 |

### 7.3 Automatic consistency checks

| Check | Flagged | Interpretation |
|---|---|---|
| family/children with any intensity = 3 | 0 | No severe safety errors |
| family/children with violence ≥ 2 | 4 | Mild; e.g. *The Hobbit: The Desolation of Smaug* (PG-13) would be "teens" |
| Horror labelled family/children | 2 of 86 | Correct (*Scooby-Doo on Zombie Island*, *The Skeleton Dance*) |
| Horror with frightening ≤ 1 | 7 of 86 | Mostly horror comedies (*Rocky Horror*, *What We Do in the Shadows*) |
| War genre without the war theme | 3 of 117 | Consistent |
| Animation + Family labelled adults | 1 of 165 | Consistent |
| Comedy without a funny/light mood | 139 of 496 | Mostly dramedies (*Parasite*, *Life Is Beautiful*, *Forrest Gump*) |
| Sci-fi without a sci-fi theme | 58 of 194 | Vocabulary gap (see 7.4) |
| Adults with all intensity scores 0 | 278 (14%) | Over-restriction (see 7.4) |

### 7.4 Known weaknesses

1. **Audience means "serious film", not "unsuitable for children".** 278 films are labelled adults with every intensity score 0 (e.g. *City Lights*, *Modern Times*, *Cinema Paradiso*, *Ikiru*). The error is in the safe direction, but it would block many suitable films for children. Audience will be taken from TMDB certifications, with the LLM value as a fallback.
2. **Vocabulary gaps.** 58 sci-fi films have no sci-fi theme because no fitting label exists: no "superheroes" (*Avengers*, *Spider-Verse*), nothing for dreams/reality (*Inception*), and space opera is not "space exploration" (*The Empire Strikes Back*).
3. **Overused broad themes.** identity and self-discovery (46%) and good versus evil (26%) remain frequent despite the v2 rule. TF-IDF gives such common labels low weight, so their effect on similarity is limited.
4. **Padding.** 41% of movies received the maximum of 5 themes, and almost all received 3 moods.
5. **Missed central themes** in some films, e.g. *Once Upon a Time in the West* without "revenge".
6. **Underestimated content.** Language is rarely scored 3 (9 films), as plot summaries seldom mention it.

### 7.5 Spot check (random sample of 22, prompt v2)

Accurate examples: *Pulp Fiction* (crime and criminals, redemption, money and ambition; violence 3, language 3, drugs 3), *City of God* (crime and criminals, coming of age, social class, survival; drugs 3), *Green Book* (friendship, racism, road trip, historical events, social class), *Princess Mononoke* (nature and environment), *The Pianist*, *Paths of Glory*, *Oldboy*, *Harakiri*.

Weaker examples: *Once Upon a Time in the West* (missing revenge), *The Green Mile* ("mental illness" is a stretch).

---

## 8. Data handling rules

- The JSONL file is the raw record of what the model produced and is never edited by hand.
- Wrong labels are not corrected movie by movie; that would measure manual corrections instead of LLM enrichment. Errors are measured (audit) and fixed only systematically (new prompt version applied to all movies).
- Values from other sources (e.g. TMDB certifications) are kept in separate files and combined later with a source column.

---

## 9. Next steps

1. **Flatten to CSV** (`output/enrichment/movies_enriched_v2.csv`): one row per movie with `llm_themes`, `llm_mood`, `llm_audience`, five intensity columns, model and prompt version.
2. **Audit:** a random sample of 50–100 movies checked against external sources (IMDb Parents Guide, AllMovie, MovieLens Tag Genome) to measure label accuracy.
3. **TMDB certifications** for all 1,957 movies, to replace the LLM audience.
4. **Text representations:** a clean base text and an enriched text (base + themes + mood) per movie.
5. **Embeddings:** embeddinggemma indexes for both texts.
6. **Evaluation:** content-based and embedding-based recommenders with and without enrichment; warm and cold-start users; NDCG@10, Precision@10, Recall@10 with significance tests.
7. **Optional:** enrich with gpt-oss-120b (Groq) to test whether better labels give better recommendations.

---

## 10. Reproducing the run

```bash
ollama pull gemma4:e4b
python enrich_catalog.py
```

With `limit=None` in `enrich_catalog.py`, the script processes every movie not yet in the cache. Running it again on a finished cache reports `1957 already finished, 0 to process`.
