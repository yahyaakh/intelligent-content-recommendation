"""
Longer cast and all directors for the 1,957 catalog films, from TMDB /movie/{id}/credits.

Why: data/fetch_tmdb.py kept only the first 3 cast members and the first director. The models
(content, LTR v2 actor_overlap / director_match) use those columns, so they are NOT changed here.
This script writes two NEW columns for the constraint layer (request filters on actors/directors):
    cast_full       top CAST_TOP_N actors in TMDB billing order ("order" field), '|'-separated
    directors_full  every crew member with job "Director", in TMDB crew order, '|'-separated

Usage:
    python scripts/fetch_credits_full.py fetch [N]   # fetch credits (free, resumable); N = only the first N films (test)
    python scripts/fetch_credits_full.py build       # raw jsonl -> credits_full.csv
    python scripts/fetch_credits_full.py verify      # is the current `cast` the first 3 billed actors?
"""
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
MOVIES = PROJECT / "data/data/processed_30k/movies.csv"
LABELS = PROJECT / "output/experiments/deepseek_v4.1_flash_v5/movies_enriched.csv"
OUT = PROJECT / "data/data/credits_full"
RAW = OUT / "tmdb_credits_raw.jsonl"     # slimmed raw answer, one line per film
CSV = OUT / "credits_full.csv"
CAST_TOP_N = 10


def catalog():
    """The 1,957 enriched films (movie_id = TMDB id), with the current cast and director."""
    ids = set(pd.read_csv(LABELS)["movie_id"])
    movies = pd.read_csv(MOVIES)
    return movies[movies["movie_id"].isin(ids)].drop_duplicates("movie_id").reset_index(drop=True)


def slim(credits):
    """Keep only what we need from the credits answer (the full answer is large)."""
    cast = [{"id": c.get("id"), "name": c.get("name"), "order": c.get("order"),
             "character": c.get("character")} for c in credits.get("cast", [])]
    directors = [{"id": c.get("id"), "name": c.get("name")}
                 for c in credits.get("crew", []) if c.get("job") == "Director"]
    return {"cast": cast, "directors": directors}


def fetch(limit=None):
    import requests
    from dotenv import load_dotenv
    load_dotenv(PROJECT / ".env")
    key = os.getenv("TMDB_API_KEY")
    if not key:
        raise RuntimeError("TMDB_API_KEY is missing from .env")
    OUT.mkdir(parents=True, exist_ok=True)
    done = set()
    if RAW.exists():
        done = {json.loads(line)["movie_id"] for line in RAW.open(encoding="utf-8")}
    films = [m for m in catalog()["movie_id"] if m not in done]
    if limit:
        films = films[:limit]
    print(f"{len(done)} films cached, {len(films)} to fetch", flush=True)
    session = requests.Session()
    with RAW.open("a", encoding="utf-8") as out:
        for i, movie_id in enumerate(films, 1):
            for attempt in range(5):
                resp = session.get(f"https://api.themoviedb.org/3/movie/{movie_id}/credits",
                                   params={"api_key": key}, timeout=20)
                if resp.status_code == 429 or resp.status_code >= 500:
                    time.sleep(2 ** attempt)
                    continue
                break
            if resp.status_code == 404:
                record = {"movie_id": int(movie_id), "found": False, "cast": [], "directors": []}
            else:
                resp.raise_for_status()
                record = {"movie_id": int(movie_id), "found": True, **slim(resp.json())}
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            if i % 200 == 0:
                out.flush()
                print(f"  {i}/{len(films)}", flush=True)
            time.sleep(0.03)


def load_raw():
    records = {}
    for line in RAW.open(encoding="utf-8"):
        r = json.loads(line)
        records[r["movie_id"]] = r          # last line wins if a film was fetched twice
    return records


def billed(cast):
    """Cast sorted by TMDB billing order (the 'order' field)."""
    return sorted(cast, key=lambda c: (c["order"] is None, c["order"]))


def build():
    records = load_raw()
    films = catalog()
    missing = set(films["movie_id"]) - set(records)
    if missing:
        raise RuntimeError(f"{len(missing)} films not fetched yet; run `fetch` first")
    rows = []
    for movie_id in films["movie_id"]:
        r = records[movie_id]
        cast = billed(r["cast"])
        rows.append({
            "movie_id": movie_id,
            "cast_full": "|".join(c["name"] for c in cast[:CAST_TOP_N]),
            "directors_full": "|".join(dict.fromkeys(d["name"] for d in r["directors"])),
            "n_cast_tmdb": len(cast),
            "n_directors": len(dict.fromkeys(d["name"] for d in r["directors"])),
        })
    table = pd.DataFrame(rows)
    table.to_csv(CSV, index=False)
    print(f"Wrote {CSV.relative_to(PROJECT)}: {len(table)} films")
    print(f"  films with 0 cast on TMDB: {(table.n_cast_tmdb == 0).sum()}")
    print(f"  films with < {CAST_TOP_N} cast: {(table.n_cast_tmdb < CAST_TOP_N).sum()}")
    print(f"  cast size, median: {table.n_cast_tmdb.median():.0f}")
    print(f"  films with 0 directors: {(table.n_directors == 0).sum()}, "
          f"with 2+ directors: {(table.n_directors >= 2).sum()}")


def verify():
    """Is the current `cast` column the first 3 actors by TMDB billing order?"""
    records = load_raw()
    films = catalog()
    films = films[films["movie_id"].isin(records)]
    api_sorted = same_as_first3_api = same_as_first3_billed = same_set = compared = 0
    director_same = director_in_list = 0
    mismatches = []
    for _, film in films.iterrows():
        r = records[film["movie_id"]]
        api_order = [c["order"] for c in r["cast"]]
        api_sorted += api_order == sorted(o for o in api_order if o is not None)
        current = [x for x in str(film["cast"]).split("|") if x and x != "nan"]
        first3_api = [c["name"] for c in r["cast"][:3]]
        first3_billed = [c["name"] for c in billed(r["cast"])[:3]]
        if current or r["cast"]:
            compared += 1
            same_as_first3_api += current == first3_api
            same_as_first3_billed += current == first3_billed
            same_set += set(current) == set(first3_billed)
            if current != first3_billed and len(mismatches) < 10:
                mismatches.append((film["title"], current, first3_billed))
        names = [d["name"] for d in r["directors"]]
        director_same += bool(names) and film["director"] == names[0]
        director_in_list += film["director"] in names
    n = len(films)
    print(f"Films checked: {n}")
    print(f"  TMDB cast list already in billing order: {api_sorted} of {n}")
    print(f"  current cast == first 3 in the API list:    {same_as_first3_api} of {compared}")
    print(f"  current cast == first 3 by billing order:   {same_as_first3_billed} of {compared}")
    print(f"  same 3 names (any order):                   {same_set} of {compared}")
    print(f"  current director == first TMDB director:    {director_same} of {n}")
    print(f"  current director among TMDB directors:      {director_in_list} of {n}")
    if mismatches:
        print("\nFirst mismatches (title | current cast | first 3 billed now):")
        for title, cur, now in mismatches:
            print(f"  {title} | {cur} | {now}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "fetch":
        fetch(int(sys.argv[2]) if len(sys.argv) > 2 else None)
    elif mode == "build":
        build()
    elif mode == "verify":
        verify()
    else:
        print(__doc__)
