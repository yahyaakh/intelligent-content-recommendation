
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))  # shared modules live in src/
import json
from pathlib import Path

import pandas as pd
import requests

from llm_enricher import LLM_MovieEnricher, PROMPT_VERSION

OLLAMA = "http://localhost:11434"

HOSTED_PROVIDERS = {
    "groq": {
        "url": "https://api.groq.com/openai/v1/chat/completions",
        "key_env": "GROQ_API_KEY",
    },
    "gemini": {
        "url": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        "key_env": "GEMINI_API_KEY",
    },
}

MODELS = [
    # previous run (reference points)
    "llama3.2:3b", "gemma3:4b", "qwen2.5:7b", "llama3.1:8b",
    # new models
    "gemma4:e4b", "qwen3.5:4b", "qwen3.5:9b",
    "granite4.2:8b", "ministral-3:8b", "phi4-mini",
]

TEST_MOVIES = [
    238, 424, 129, 244786, 857, 694, 641, 14160, 11036, 12,
    142, 9552, 18785, 419430, 116149,
    36773, 23283, 3486, 259557, 30950,  # obscure films
]

OUTPUT_PATH = Path("output/llm_model_comparison_all.csv")

COLUMNS = [
    "model", "prompt_version", "movie_id", "title", "gpu_share", "status",
    "seconds", "prompt_tokens", "output_tokens",
    "themes", "mood", "audience", "intensity", "error",
]


def gpu_share(model_name):
    """Fraction of the loaded model that sits in GPU memory (1.0 = fully on GPU)."""
    loaded = requests.get(f"{OLLAMA}/api/ps", timeout=30).json()["models"]

    for model in loaded:
        if model["name"] == model_name and model["size"]:
            return round(model["size_vram"] / model["size"], 2)

    return None


def unload(model_name):
    """Free GPU memory so the next model is not squeezed onto the CPU."""
    requests.post(
        f"{OLLAMA}/api/generate",
        json={"model": model_name, "keep_alive": 0},
        timeout=60,
    )


movies_df = pd.read_csv("data/data/processed_30k/movies.csv")

# Resume: skip model/movie pairs that already finished in an earlier run.
if OUTPUT_PATH.exists():
    done = pd.read_csv(OUTPUT_PATH)
    finished = set(zip(done["model"], done["movie_id"]))
else:
    finished = set()

for model_name in MODELS:
    if all((model_name, movie_id) in finished for movie_id in TEST_MOVIES):
        print(f"Skipping {model_name}: already finished.")
        continue

    enricher = LLM_MovieEnricher(movies_df, model_name=model_name)

    # Load the model once so its load time does not affect the movie timings.
    warmup = enricher.call_llm("Reply with an empty JSON object.")
    if warmup["status"] != "ok":
        print(f"Skipping {model_name}: {warmup['error']}")
        continue

    on_gpu = gpu_share(model_name)
    gpu_text = "unknown" if on_gpu is None else f"{on_gpu:.0%}"
    print(f"\n{model_name} loaded ({gpu_text} on GPU)")

    for movie_id in TEST_MOVIES:
        if (model_name, movie_id) in finished:
            continue

        movie = enricher.get_movie(movie_id)
        prompt = enricher.build_prompt(enricher.build_input_text(movie))
        result = enricher.call_llm(prompt)

        row = {
            "model": model_name,
            "prompt_version": PROMPT_VERSION,
            "movie_id": movie_id,
            "title": movie["title"],
            "gpu_share": on_gpu,
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

        # Append each result immediately, so a crash loses nothing.
        pd.DataFrame([row]).reindex(columns=COLUMNS).to_csv(
            OUTPUT_PATH, mode="a", index=False,
            header=not OUTPUT_PATH.exists(),
        )
        print(f"  {movie['title']}: {row['status']} "
              f"({row.get('seconds', '-')}s)", flush=True)

    unload(model_name)

print(f"\nSaved to {OUTPUT_PATH}")