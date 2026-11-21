'''
Label the whole catalogue (1,957 films) with DeepSeek V4.1 Flash and prompt v5, and flatten the
labels into the same table format as the gemma4:e4b v4 labels
(output/experiments/gemma4_e4b_v4/movies_enriched.csv), so later experiments can swap them in.

The 1,388 genome films were already labelled by label_quality_experiment.py; they come from the
same cache file, so only the remaining films are sent to the model.

Usage:
    python scripts/enrich_catalog_deepseek_v5.py
'''

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # shared modules live in src/
from pathlib import Path

import pandas as pd

import build_movie_texts as bmt
from enrich_catalog import enrich_movies, load_records

MODEL = "deepseek-v4.1-flash:cloud"      # through the local Ollama installation (signed in), thinking off
VERSION = "v5"
CACHE = Path("output/experiments/label_quality") / f"labels_{MODEL.replace(':', '_')}_{VERSION}.jsonl"
OUT = Path("output/experiments/deepseek_v4.1_flash_v5")


if __name__ == "__main__":
    movies = pd.read_csv("data/data/processed/movies.csv")
    enricher = bmt.FenceStrippingEnricher(movies, model_name=MODEL, provider="ollama",
                                          max_retries=4, prompt_version=VERSION)
    enrich_movies(enricher, movies, CACHE)

    records = {m: r for m, r in load_records(CACHE).items() if r.get("status") == "ok"}
    missing = set(movies["movie_id"]) - set(records)
    rows = []
    for movie_id, r in records.items():
        intensity = r["content_intensity"]
        rows.append({"movie_id": movie_id,
                     "llm_themes": "|".join(r["themes"]), "llm_mood": "|".join(r["mood"]),
                     "llm_audience": r["audience"],
                     **{f"intensity_{d}": intensity.get(d) for d in
                        ("violence", "sexual_content", "language", "substance_use", "frightening_scenes")},
                     "enrich_model": r["model"], "prompt_version": r["prompt_version"],
                     "repaired": r.get("repaired", False)})
    table = pd.DataFrame(rows).sort_values("movie_id")
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / "movies_enriched.csv", index=False)

    issues = [i for r in records.values() for i in r.get("issues", [])]
    print(f"\n{len(records)} of {len(movies)} films labelled; missing: {len(missing)}")
    print(f"labels dropped (outside v5): {sum('unknown label' in i for i in issues)} | mapped by alias: "
          f"{sum('mapped' in i for i in issues)} | audience unknown: {table['llm_audience'].isna().sum()}")
    tokens_in = sum(r.get("prompt_tokens", 0) for r in records.values())
    tokens_out = sum(r.get("output_tokens", 0) for r in records.values())
    print(f"all films: tokens in {tokens_in:,}, out {tokens_out:,}, "
          f"about ${(tokens_in * 0.30 + tokens_out * 1.20) / 1e6:.2f} at peak price")
    print(f"Saved {OUT / 'movies_enriched.csv'}")
