import json
import os
import time
from pathlib import Path

import numpy as np
import requests
from tqdm import tqdm
import pandas as pd
from llm_enricher import (
    LLM_MovieEnricher,
    THEMES,
    MOODS,
    AUDIENCES,
    INTENSITY_DIMENSIONS,
    PROMPT_VERSION,
)

movies_df = pd.read_csv("data/data/processed/movies.csv")



def cache_path(model_name, prompt_version) :
    safe_name = model_name.replace(":", "_").replace("/", "_")
    folder = Path("output") / "enrichment"
    folder.mkdir(parents=True, exist_ok=True)
    file_name = safe_name + "_" + prompt_version + ".jsonl"
    return folder / file_name


def load_finished(path):
    if not path.exists():
        return set()

    latest_status = {}

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line == "":
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(f"Skipping broken line: {line[:80]}")
                continue

            latest_status[record["movie_id"]] = record["status"]

    finished = set()

    for movie in latest_status:
        if latest_status[movie] == "ok":
            finished.add(movie)

    return finished


def append_record(path, record):

    line = json.dumps(record,ensure_ascii=False) # Line 1: convert the record to a JSON string
    # record is a Python dict, e.g. {"movie_id": 603, "status": "ok", "themes": [...], ...}. json.dumps(record, ensure_ascii=False) turns it into one line of text.
    #
    # ensure_ascii=False keeps characters like é readable in the file

    with open(path, 'a',encoding="utf-8") as f: # open in append mode "a" and write the line plus a new line
        f.write(line + "\n")


def validate_output (output, themes_allowed=THEMES, moods_allowed=MOODS, aliases=None) :
    issues = []
    theme_values, mood_values = output.get("themes"), output.get("mood")
    if aliases:
        theme_values, mood_values = map_aliases(theme_values, mood_values, themes_allowed,
                                                moods_allowed, aliases, issues)
    themes = clean_labels(theme_values, themes_allowed, 5, "themes", issues)
    mood = clean_labels(mood_values, moods_allowed, 3, "mood", issues)

    audience = str (output.get("audience") or "").strip().lower()
    if audience not in AUDIENCES :
        issues.append(f"audience: invalid value '{audience}'")
        audience = None

    raw_intensity = output.get("content_intensity")
    if not isinstance(raw_intensity, dict):
        issues.append("content_intensity: not a dict")
        raw_intensity = {}
    intensity = {}
    for dimension in INTENSITY_DIMENSIONS :
        value = raw_intensity.get(dimension)
        try:
            value = int(value)
        except (TypeError, ValueError):
            issues.append(f"{dimension}: missing")
            intensity[dimension] = None
            continue

        if value < 0 or value > 3:
            issues.append(f"{dimension}: out of range ({value})")
        intensity[dimension] = max(0, min(3, value))

    clean = {
        "themes": themes,
        "mood": mood,
        "audience": audience,
        "content_intensity": intensity,
    }
    return clean, issues

def map_aliases(theme_values, mood_values, themes_allowed, moods_allowed, aliases, issues):
    """Prompt v5: a label outside the vocabulary that is a known alias is moved to its vocabulary
    label instead of being dropped ("satire" in themes -> "satirical" in mood). Labels that are
    already valid in their own field are left alone. Each mapping is recorded as an issue."""
    allowed = {"themes": set(themes_allowed), "moods": set(moods_allowed)}
    out = {"themes": [], "moods": []}
    moved = {"themes": [], "moods": []}      # labels moved over from the other field go last,
    for field, values in (("themes", theme_values), ("moods", mood_values)):   # so they never push
        if type(values) is not list:                                           # out the model's own
            continue                                                           # choices
        for value in values:
            label = str(value).lower().strip()
            if label not in allowed[field] and label in aliases:
                target_field, target = aliases[label]
                issues.append(f"{field}: mapped '{label}' -> {target_field} '{target}'")
                (out if target_field == field else moved)[target_field].append(target)
            else:
                out[field].append(value)
    # a field that was not a list stays as it was (clean_labels reports "not a list")
    themes = out["themes"] + moved["themes"] if type(theme_values) is list else theme_values
    moods = out["moods"] + moved["moods"] if type(mood_values) is list else mood_values
    return themes, moods


def clean_labels(values , allowed , max_count , field , issues) :
    if type(values) is not list :
        issues.append(f"{field}: not a list")
        return []

    kept = []
    for value in values:
        label = str(value).lower().strip()
        if label not in allowed:
            issues.append(f"{field}: unknown label '{label}'")
        elif label in kept:
            issues.append(f"{field}: duplicate '{label}'")
        else:
            kept.append(label)

    if len(kept) > max_count :
        issues.append(f"{field}: too many labels")
        kept = kept[:max_count]

    return kept



# Main loop: call the LLM per movie, validate, save
# Retry failed movies
# Summary report (coverage, time, top themes)
# Flatten JSONL → CSV
# Text representations (base + enriched)
# Embed both and build the FAISS indexes

def enrich_movies (enricher , movies_df , path , limit= None) :
    finished = load_finished(path)
    todo = []
    for _, movie in movies_df.iterrows():
        if int(movie["movie_id"]) not in finished :
            todo.append(movie)

    if limit is not None:
        todo = todo[:limit]

    print(f"{len(finished)} already finished, {len(todo)} to process")

    for movie in tqdm(todo, desc="Enriching") :
        movie_id = int(movie["movie_id"])
        prompt = enricher.build_prompt(enricher.build_input_text(movie))
        result = enricher.call_llm(prompt)

        record = {
            "movie_id": movie_id,
            "title": movie["title"],
            "model": enricher.model_name,
            "prompt_version": enricher.prompt_version,   # the prompt actually used
        }

        if result["status"] != "ok" :
            record["status"] = "failed"
            record["error"] = result["error"]

        else :
            record["seconds"] = round(result["llm_seconds"], 2)
            record["prompt_tokens"] = result["prompt_tokens"]
            record["output_tokens"] = result["output_tokens"]
            record["wall_seconds"] = round(result["wall_seconds"], 2)

            try :
                output = json.loads(result["content"])

            except json.JSONDecodeError:
                record["status"] = "invalid_json"
                record["error"] = result["content"]
            else :
                  clean , issues = validate_output(output, enricher.themes, enricher.moods,
                                                   getattr(enricher, "aliases", None))
                  record.update(clean)
                  record["issues"] = issues
                  record["repaired"] = len(issues) > 0
                  record["status"] = "ok" if clean["themes"] else "invalid"



        append_record(path, record)



def load_records(path):
    records = {}
    with open(path, "r" , encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line :
                continue
            try :
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            records[record["movie_id"]] = record

    return records


def flatten_to_csv(path, movies_df):
    records = load_records(path)
    rows = []
    for record in records.values() :
        if record["status"] != "ok" :
            continue

        intensity = record["content_intensity"]
        row = {
            "movie_id": record["movie_id"],
            "llm_themes": "|".join(record["themes"]),
            "llm_mood": "|".join(record["mood"]),
            "llm_audience": record["audience"],
            "intensity_violence": intensity["violence"],
            "intensity_sexual_content": intensity["sexual_content"],
            "intensity_language": intensity["language"],
            "intensity_substance_use": intensity["substance_use"],
            "intensity_frightening_scenes": intensity["frightening_scenes"],
            "enrich_model": record["model"],
            "prompt_version": record["prompt_version"],
        }
        rows.append(row)


    enriched = pd.DataFrame(rows)

    enriched_ids = set(enriched["movie_id"])
    catalog_ids = set(movies_df["movie_id"])

    missing = catalog_ids - enriched_ids      # in the catalog, but not enriched
    unknown = enriched_ids - catalog_ids      # enriched, but not in this catalog

    assert enriched["movie_id"].is_unique, "Duplicate movie ids in enrichment"
    assert not unknown, f"{len(unknown)} enriched movies are not in the catalog"

    if missing:
        print(f"Warning: {len(missing)} catalog movies are not enriched")



    out_path = path.parent / f"movies_enriched_{PROMPT_VERSION}.csv"
    enriched.to_csv(out_path, index=False)

    print(f"Saved {len(enriched)} rows to {out_path}")
    return enriched






if __name__ == "__main__":
    model_name = "gemma4:e4b"
    enricher = LLM_MovieEnricher(movies_df, model_name=model_name)
    path = cache_path(model_name, PROMPT_VERSION)

    enrich_movies(enricher, movies_df, path, limit=None)

    flatten_to_csv(path, movies_df)