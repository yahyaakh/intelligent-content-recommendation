"""Free-text film profiles for the outside-knowledge experiment (thesis doc 07).

P1 and P2 ask for the same short profile and differ in ONE sentence:
    P1: use only the information given (title, year, genres, keywords, overview)
    P2: use the information given AND the model's own knowledge of the film

Usage (one model, both variants; safe to stop and restart, finished movies are skipped):
    python experiments/enrichment/enrich_profiles.py gemma4:31b-cloud
    python experiments/enrichment/enrich_profiles.py gemma4:e4b

Output: output/enrichment/profiles/<model>_<variant>.jsonl (cache) and .csv (one row per movie).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))  # shared modules live in src/
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

from llm_enricher import LLM_MovieEnricher   # build_input_text: the same input as the labels

PROFILE_VERSION = "p1"     # bump if the prompt text below changes
FOLDER = Path("output") / "enrichment" / "profiles"
OLLAMA_URL = "http://127.0.0.1:11434/api/chat"   # local models and Ollama cloud models

SYSTEM_PROMPT = ("You are a film analyst who writes short descriptive profiles of movies for a "
                 "recommendation system. You always return only valid JSON, with no explanation "
                 "or extra text.")

TASK = """Write a profile of the movie below in 2 to 3 sentences (at most 60 words).
Describe what makes it distinctive: its tone and atmosphere, its style and pacing, and how the story is told.
Do not retell the plot and do not name other films.
{source}

MOVIE
{input_text}

OUTPUT FORMAT
{{"profile": "..."}}"""

# The ONLY difference between the two variants.
SOURCE = {
    "P1": "Use ONLY the information given below. Do not add facts that are not stated in it "
          "or directly implied by it.",
    "P2": "Use the information given below AND your own knowledge of this film, for example its "
          "full story, its style and how it is generally received. If you do not recognise the "
          "film, use only the information given.",
}

SCHEMA = {"type": "object", "properties": {"profile": {"type": "string"}}, "required": ["profile"]}
MIN_WORDS, MAX_WORDS = 10, 120


def cache_file(model, variant, suffix="jsonl"):
    safe = model.replace(":", "_").replace("/", "_")
    return FOLDER / f"{safe}_{variant}.{suffix}"


def load_cache(path):
    records = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                records[record["movie_id"]] = record     # the latest record per movie wins
    return records


def call_llm(model, prompt, retries=4):
    payload = {"model": model, "stream": False, "think": False, "format": SCHEMA,
               "options": {"temperature": 0, "seed": 42},
               "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": prompt}]}
    last_error = None
    for attempt in range(retries + 1):
        start = time.perf_counter()
        try:
            res = requests.post(OLLAMA_URL, json=payload, timeout=300)
            res.raise_for_status()
            body = res.json()
            return {"status": "ok", "content": body["message"]["content"],
                    "seconds": time.perf_counter() - start,
                    "prompt_tokens": body.get("prompt_eval_count", 0),
                    "output_tokens": body.get("eval_count", 0)}
        except (requests.RequestException, ValueError, KeyError) as error:
            last_error = str(error)
            time.sleep(5 * (attempt + 1))        # cloud models can be briefly busy
    return {"status": "failed", "error": last_error}


def parse_profile(content):
    """Return (profile, problem). Cloud models sometimes wrap the JSON in ``` fences."""
    text = re.sub(r"^```(?:json)?|```$", "", content.strip()).strip()
    try:
        profile = json.loads(text)["profile"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return None, "invalid_json"
    profile = " ".join(str(profile).split())    # one line, single spaces
    words = len(profile.split())
    if not MIN_WORDS <= words <= MAX_WORDS:
        return profile, f"length {words} words"
    return profile, None


def enrich(model, variant, movies):
    path = cache_file(model, variant)
    path.parent.mkdir(parents=True, exist_ok=True)
    done = {m for m, r in load_cache(path).items() if r["status"] == "ok"}
    todo = movies[~movies["movie_id"].isin(done)]
    print(f"\n{model} {variant}: {len(done)} done, {len(todo)} to process")

    builder = LLM_MovieEnricher(movies, model_name=model)
    with open(path, "a", encoding="utf-8") as f:
        for _, movie in tqdm(todo.iterrows(), total=len(todo), desc=f"{model} {variant}"):
            prompt = TASK.format(source=SOURCE[variant], input_text=builder.build_input_text(movie))
            result = call_llm(model, prompt)
            record = {"movie_id": int(movie["movie_id"]), "title": movie["title"], "model": model,
                      "variant": variant, "profile_version": PROFILE_VERSION}
            if result["status"] != "ok":
                record.update(status="failed", error=result["error"])
            else:
                profile, problem = parse_profile(result["content"])
                record.update(profile=profile, seconds=round(result["seconds"], 2),
                              prompt_tokens=result["prompt_tokens"],
                              output_tokens=result["output_tokens"],
                              status="ok" if problem is None else "invalid", problem=problem)
                if problem == "invalid_json":
                    record["raw"] = result["content"]
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()


def report_and_save(model, variant, movies):
    records = load_cache(cache_file(model, variant))
    ok = {m: r for m, r in records.items() if r["status"] == "ok"}
    missing = set(movies["movie_id"]) - set(ok)
    seconds = [r["seconds"] for r in ok.values()]
    words = [len(r["profile"].split()) for r in ok.values()]
    print(f"{model} {variant}: ok {len(ok)} / {len(movies)} | not ok: {len(missing)} | "
          f"mean {sum(words) / max(1, len(words)):.1f} words | "
          f"mean {sum(seconds) / max(1, len(seconds)):.2f} s per movie")
    if missing:
        print(f"  Run the same command again to retry the {len(missing)} missing movies.")
        return
    rows = [{"movie_id": m, "profile": ok[m]["profile"], "model": model, "variant": variant,
             "profile_version": PROFILE_VERSION} for m in movies["movie_id"]]
    out = cache_file(model, variant, "csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"  Saved {out}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python experiments/enrichment/enrich_profiles.py <model>   e.g. gemma4:31b-cloud or gemma4:e4b")
    model = sys.argv[1]
    variants = sys.argv[2:] or ["P1", "P2"]
    movies = pd.read_csv("data/data/processed/movies.csv")

    for variant in variants:
        enrich(model, variant, movies)
        enrich(model, variant, movies)        # second pass retries anything that failed
    print("\nSummary:")
    for variant in variants:
        report_and_save(model, variant, movies)
