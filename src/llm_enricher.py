"""LLM-based movie enrichment: themes, mood, audience and content intensity."""

import json
import os
import time

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

HOSTED_PROVIDERS = {
    "groq": {
        "url": "https://api.groq.com/openai/v1/chat/completions",
        "key_env": "GROQ_API_KEY",
    },
    "gemini": {
        "url": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        "key_env": "GEMINI_API_KEY",
    },
    "deepseek": {
        "url": "https://api.deepseek.com/chat/completions",
        "key_env": "DEEPSEEK_API_KEY",
    },
    # Paid models only: OpenCode's free tier is refused outside the OpenCode app.
    "opencode": {
        "url": "https://opencode.ai/inference/openai/v1/chat/completions",
        "key_env": "OPENCODE_API_KEY",
    },
}

THEMES = [
    "love and romance", "friendship", "family relationships", "coming of age",
    "revenge", "betrayal", "redemption", "survival",
    "identity and self-discovery", "loss and grief", "good versus evil",
    "power and corruption", "justice and the law", "crime and criminals",
    "detective investigation", "heist", "war and its consequences",
    "political conflict", "rebellion against authority", "freedom and oppression",
    "artificial intelligence and technology", "space exploration",
    "time travel", "dystopian society", "supernatural forces", "magic",
    "monsters and creatures", "adventure and quest", "sports and competition",
    "music and performance", "art and creativity", "money and ambition",
    "social class and inequality", "racism and discrimination",
    "mental illness", "addiction", "faith and religion", "historical events",
    "nature and environment", "school life", "journey and road trip",
    "isolation and loneliness", "sacrifice", "deception and conspiracy",
]

MOODS = [
    "dark", "tense", "suspenseful", "scary", "funny", "lighthearted",
    "heartwarming", "uplifting", "romantic", "sad", "emotional",
    "thought-provoking", "inspiring", "action-packed", "whimsical",
    "nostalgic", "gritty", "epic",
]

AUDIENCES = ["children", "family", "teens", "adults"]

INTENSITY_DIMENSIONS = [
    "violence", "sexual_content", "language",
    "substance_use", "frightening_scenes",
]

def make_schema(themes, moods):
    """JSON schema that forces the answer into the allowed lists."""
    return {
        "type": "object",
        "properties": {
            "themes": {"type": "array", "items": {"type": "string", "enum": themes},
                       "minItems": 1, "maxItems": 5, "uniqueItems": True},
            "mood": {"type": "array", "items": {"type": "string", "enum": moods},
                     "minItems": 1, "maxItems": 3, "uniqueItems": True},
            "audience": {"type": "string", "enum": AUDIENCES},
            "content_intensity": {
                "type": "object",
                "properties": {d: {"type": "integer", "minimum": 0, "maximum": 3}
                               for d in INTENSITY_DIMENSIONS},
                "required": INTENSITY_DIMENSIONS,
            },
        },
        "required": ["themes", "mood", "audience", "content_intensity"],
    }


# Ollama constrains the model's output to this structure.
OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "themes": {
            "type": "array",
            "items": {"type": "string", "enum": THEMES},
            "minItems": 1,
            "maxItems": 5,
            "uniqueItems": True,
        },
        "mood": {
            "type": "array",
            "items": {"type": "string", "enum": MOODS},
            "minItems": 1,
            "maxItems": 3,
            "uniqueItems": True,
        },
        "audience": {"type": "string", "enum": AUDIENCES},
        "content_intensity": {
            "type": "object",
            "properties": {
                dimension: {"type": "integer", "minimum": 0, "maximum": 3}
                for dimension in INTENSITY_DIMENSIONS
            },
            "required": INTENSITY_DIMENSIONS,
        },
    },
    "required": ["themes", "mood", "audience", "content_intensity"],
}

PROMPT_VERSION = "v2"

SYSTEM_PROMPT = (
    "You are a film metadata annotator. You label movies using ONLY the "
    "allowed values you are given, spelled exactly as written. You base your "
    "labels on the movie information provided. Precision matters more than "
    "coverage: a short list of clearly correct labels is better than a long "
    "list with weak ones. You always return only valid JSON, with no "
    "explanation or extra text."
)

USER_PROMPT = """Annotate the movie below.

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
{{"themes": [...], "mood": [...], "audience": "...", "content_intensity": {{"violence": 0, "sexual_content": 0, "language": 0, "substance_use": 0, "frightening_scenes": 0}}}}
"""

# v3 changes ONE thing compared with v2: the model may also use its own knowledge of
# the film, not only the text it is given. Lists, schema, user prompt and settings are
# identical, so any difference in the results comes from that instruction.
SYSTEM_PROMPT_V3 = (
    "You are a film metadata annotator. You label movies using ONLY the "
    "allowed values you are given, spelled exactly as written. Base your labels "
    "on the movie information provided AND on your own knowledge of this film, "
    "for example its full story, tone and style. If you do not recognise the "
    "film, use only the information provided. Precision matters more than "
    "coverage: a short list of clearly correct labels is better than a long "
    "list with weak ones. You always return only valid JSON, with no "
    "explanation or extra text."
)

# v4 changes the VOCABULARY compared with v2: 246 themes and 103 moods taken from the
# MovieLens Tag Genome (data/vocabularies/v4_genome.json, built by build_vocabulary_v4.py) instead
# of 44 themes and 18 moods. The system prompt, the audience and intensity fields, the
# maximum number of labels and all settings are identical. The user prompt differs only
# where v2 named labels that do not exist in the new lists (the two rules about specific
# v2 themes are replaced by one general rule about broad themes).
_V4_VOCABULARY = json.loads(
    open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "vocabularies", "v4_genome.json"),
         encoding="utf-8").read())
THEMES_V4, MOODS_V4 = _V4_VOCABULARY["themes"], _V4_VOCABULARY["moods"]

USER_PROMPT_V4 = USER_PROMPT.replace(
    """   - Use the broad themes "family relationships", "good versus evil", "loss and grief" and "identity and self-discovery" only when they are a main focus of the story, not when they are merely present.
   - "crime and criminals" is for stories about committing crimes or living a criminal life. "detective investigation" is only for stories centred on solving a crime or mystery.
""",
    """   - Prefer the most specific themes that fit. Use broad themes such as "friendship", "death", "love story" or "good versus evil" only when they are a main focus of the story, not when they are merely present.
""").replace(
    "2. mood: choose 1 to 3 values from MOOD LIST that describe how the movie feels to watch.",
    "2. mood: choose 1 to 3 values from MOOD LIST that describe how the movie feels to watch and how it is told.")
assert USER_PROMPT_V4.count("Prefer the most specific themes") == 1, "v4 theme rule was not inserted"
assert USER_PROMPT_V4.count("and how it is told") == 1, "v4 mood rule was not inserted"

# v5 changes only the VOCABULARY compared with v4: v4 plus 9 genome themes that the label-quality
# experiment showed missing (data/vocabularies/v5_genome.json, built by build_vocabulary_v5.py), and a
# list of aliases that the answer validation maps instead of dropping (e.g. "satire" -> mood
# "satirical", "tragic" -> mood "tragedy"). The prompt text is identical to v4.
_V5_VOCABULARY = json.loads(
    open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "vocabularies", "v5_genome.json"),
         encoding="utf-8").read())
THEMES_V5, MOODS_V5 = _V5_VOCABULARY["themes"], _V5_VOCABULARY["moods"]
ALIASES_V5 = {alias: tuple(target) for alias, target in _V5_VOCABULARY["aliases"].items()}

# prompt version -> system prompt, user prompt, the allowed lists and (v5) the aliases
PROMPTS = {
    "v2": {"system": SYSTEM_PROMPT, "user": USER_PROMPT, "themes": THEMES, "moods": MOODS},
    "v3": {"system": SYSTEM_PROMPT_V3, "user": USER_PROMPT, "themes": THEMES, "moods": MOODS},
    "v4": {"system": SYSTEM_PROMPT, "user": USER_PROMPT_V4, "themes": THEMES_V4, "moods": MOODS_V4},
    "v5": {"system": SYSTEM_PROMPT, "user": USER_PROMPT_V4, "themes": THEMES_V5, "moods": MOODS_V5,
           "aliases": ALIASES_V5},
}


class LLM_MovieEnricher:
    def __init__(self, movie_df, model_name, provider="ollama",
                 reasoning_effort=None,
                 # 127.0.0.1, not localhost: on Windows "localhost" tries IPv6
                 # first and adds about 2 s to every request.
                 ollama_url="http://127.0.0.1:11434/api/chat",
                 timeout=300, max_retries=2, seed=42, prompt_version=PROMPT_VERSION):
        self.movie_df = movie_df
        self.model_name = model_name
        self.provider = provider
        self.reasoning_effort = reasoning_effort
        self.ollama_url = ollama_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.seed = seed
        # Which prompt to use; the default keeps every existing script on v2.
        self.prompt_version = prompt_version
        prompt = PROMPTS[prompt_version]
        self.system_prompt, self.user_prompt = prompt["system"], prompt["user"]
        self.themes, self.moods = prompt["themes"], prompt["moods"]
        self.aliases = prompt.get("aliases", {})        # v5 only; earlier versions map nothing
        self.schema = make_schema(self.themes, self.moods)

    def get_movie(self, movie_id):
        match = self.movie_df.loc[self.movie_df["movie_id"] == movie_id]

        if match.empty:
            raise ValueError(f"Movie {movie_id} not found.")

        # Return the row itself, not a one-row DataFrame.
        return match.iloc[0]

    def build_input_text(self, movie):
        def field(name):
            value = movie.get(name)
            return "" if pd.isna(value) else str(value).strip()

        lines = [f"Title: {field('title')} ({field('year')})"]

        if field("genres"):
            lines.append(f"Genres: {field('genres')}")
        if field("keywords"):
            lines.append(f"Keywords: {field('keywords').replace('|', ', ')}")

        lines.append(f"Overview: {field('overview') or 'Not available.'}")

        return "\n".join(lines)

    def build_prompt(self, input_text):
        return self.user_prompt.format(
            input_text=input_text,
            themes=", ".join(self.themes),
            moods=", ".join(self.moods),
        )

    def _call_ollama(self, user_prompt):
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            # False switches thinking off; gpt-oss needs a level ("low") instead.
            "think": self.reasoning_effort or False,
            "format": self.schema,
            "options": {"temperature": 0, "seed": self.seed},
            "keep_alive": "10m",
        }

        last_error = None

        for attempt in range(1, self.max_retries + 2):
            start = time.perf_counter()

            try:
                res = requests.post(
                    self.ollama_url, json=payload, timeout=self.timeout
                )
                res.raise_for_status()
                body = res.json()
            except (requests.RequestException, ValueError) as error:
                last_error = str(error)

                if attempt <= self.max_retries:
                    time.sleep(2 * attempt)
                continue

            wall_seconds = time.perf_counter() - start

            # Ollama reports durations in nanoseconds.
            total_seconds = body.get("total_duration", 0) / 1e9
            load_seconds = body.get("load_duration", 0) / 1e9

            return {
                "status": "ok",
                "content": body["message"]["content"],
                "attempts": attempt,
                "wall_seconds": wall_seconds,
                "llm_seconds": total_seconds - load_seconds,
                "load_seconds": load_seconds,
                "prompt_tokens": body.get("prompt_eval_count", 0),
                "output_tokens": body.get("eval_count", 0),
            }

        return {
            "status": "failed",
            "content": None,
            "attempts": self.max_retries + 1,
            "error": last_error,
        }

    def call_llm(self, user_prompt):
        if self.provider == "ollama":
            return self._call_ollama(user_prompt)
        return self._call_hosted(user_prompt)

    def _call_hosted(self, user_prompt):
        provider = HOSTED_PROVIDERS[self.provider]
        api_key = os.environ[provider["key_env"]]

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0,
            # Same schema as the local runs.
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "movie_enrichment", "schema": self.schema},
            },
        }

        # Keep reasoning as low as the provider allows, as with think=False locally.
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort

        # Gemini rejects the seed field.
        if self.provider != "gemini":
            payload["seed"] = self.seed

        headers = {"Authorization": f"Bearer {api_key}"}
        last_error = None

        for attempt in range(1, self.max_retries + 2):
            start = time.perf_counter()

            try:
                res = requests.post(
                    provider["url"], json=payload, headers=headers,
                    timeout=self.timeout,
                )
            except requests.RequestException as error:
                last_error = str(error)
                time.sleep(2 * attempt)
                continue

            # Rate limited or temporarily overloaded: wait, then retry.
            if res.status_code == 429 or res.status_code >= 500:
                last_error = f"HTTP {res.status_code}: {res.text[:200]}"
                time.sleep(float(res.headers.get("retry-after", 15 * attempt)))
                continue

            # Other client errors (bad model name, unsupported field) will not
            # fix themselves, so return the message instead of retrying.
            if res.status_code >= 400:
                return {
                    "status": "failed",
                    "content": None,
                    "attempts": attempt,
                    "error": f"HTTP {res.status_code}: {res.text[:500]}",
                }

            body = res.json()
            usage = body.get("usage", {})

            return {
                "status": "ok",
                "content": body["choices"][0]["message"]["content"],
                "attempts": attempt,
                "wall_seconds": time.perf_counter() - start,
                # No model loading for hosted APIs; this includes network time.
                "llm_seconds": time.perf_counter() - start,
                "load_seconds": 0.0,
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
            }

        return {
            "status": "failed",
            "content": None,
            "attempts": self.max_retries + 1,
            "error": last_error,
        }
