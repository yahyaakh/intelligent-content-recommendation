
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))  # shared modules live in src/
import json
import time
from pathlib import Path

import pandas as pd

from llm_enricher import LLM_MovieEnricher, PROMPT_VERSION

# provider, model name (copy the exact name from the provider's model list),
# and reasoning setting (None = model has no reasoning mode)
MODELS = [
    ("groq", "openai/gpt-oss-120b", "low"),
    ("groq", "qwen/qwen3.8-27b", "none"),
    ("groq", "openai/gpt-oss-20b", "low"),
    ("gemini", "gemini-3.8-flash", "none"),
    # Already finished with Ollama's default reasoning level; delete its rows
    # from the CSV to rerun it at "low" like the Groq gpt-oss runs.
    ("ollama", "gpt-oss:120b-cloud", "low"),  # only after `ollama signin`
    # Paid (needs OpenCode credits); a 20-movie run costs a few cents.
    ("opencode", "deepseek-v4.1-flash", None),
    ("opencode", "deepseek-v4-pro", None),
]

TEST_MOVIES = [
    238, 424, 129, 244786, 857, 694, 641, 14160, 11036, 12,
    142, 9552, 18785, 419430, 116149,
    36773, 23283, 3486, 259557, 30950,  # obscure films
]

OUTPUT_PATH = Path("output/llm_model_comparison_hosted.csv")
DELAY_SECONDS = 3  # stay under free-tier per-minute limits

COLUMNS = [
    "provider", "model", "prompt_version", "movie_id", "title", "status",
    "seconds", "prompt_tokens", "output_tokens",
    "themes", "mood", "audience", "intensity", "error",
]

movies_df = pd.read_csv("data/data/processed_30k/movies.csv")

if OUTPUT_PATH.exists():
    done = pd.read_csv(OUTPUT_PATH)
    # Only successful rows count as finished, so failed movies are retried.
    done = done[done["status"] == "ok"]
    finished = set(zip(done["model"], done["movie_id"]))
else:
    finished = set()

for provider, model_name, reasoning_effort in MODELS:
    enricher = LLM_MovieEnricher(
        movies_df, model_name=model_name, provider=provider,
        reasoning_effort=reasoning_effort,
        max_retries=4,  # hosted models can be briefly overloaded (HTTP 503)
    )
    print(f"\n{provider} / {model_name}")

    for movie_id in TEST_MOVIES:
        if (model_name, movie_id) in finished:
            continue

        movie = enricher.get_movie(movie_id)
        prompt = enricher.build_prompt(enricher.build_input_text(movie))
        result = enricher.call_llm(prompt)

        row = {
            "provider": provider,
            "model": model_name,
            "prompt_version": PROMPT_VERSION,
            "movie_id": movie_id,
            "title": movie["title"],
            "status": result["status"],
        }

        if result["status"] == "ok":
            row.update({
                "seconds": round(result["llm_seconds"], 2),
                "prompt_tokens": result["prompt_tokens"],
                "output_tokens": result["output_tokens"],
            })

            try:
                output = json.loads(result["content"])
                row.update({
                    "themes": output.get("themes"),
                    "mood": output.get("mood"),
                    "audience": output.get("audience"),
                    "intensity": output.get("content_intensity"),
                })
            except json.JSONDecodeError:
                row["status"] = "invalid_json"
                row["error"] = result["content"]
        else:
            row["error"] = result["error"]

        pd.DataFrame([row]).reindex(columns=COLUMNS).to_csv(
            OUTPUT_PATH, mode="a", index=False,
            header=not OUTPUT_PATH.exists(),
        )
        print(f"  {movie['title']}: {row['status']} "
              f"({row.get('seconds', '-')}s)", flush=True)

        # A failed first call usually means a setup problem; stop this model.
        if row["status"] == "failed" and movie_id == TEST_MOVIES[0]:
            print(f"  Stopping {model_name}: {row['error']}")
            break

        time.sleep(DELAY_SECONDS)

print(f"\nSaved to {OUTPUT_PATH}")