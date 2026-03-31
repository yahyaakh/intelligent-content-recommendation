
# reads the JSON data from fetch_tmdb.py and produces movies.csv

# process_tmdb.py
# Job: read raw JSON, clean it, build content_features,
#      save final movies.csv
# Run this AFTER fetch_tmdb.py
# ─────────────────────────────────────────────────────

import json
import pandas as pd
import os

#  open and load "data/raw/tmdb_raw.json"
with open("data/raw/tmdb_raw.json", "r", encoding="utf-8") as f:
    raw = json.load(f)

df = pd.DataFrame(raw)

df['year'] = df['year'].astype(int)


df['overview'] = df['overview'].fillna('')
df['keywords'] = df['keywords'].fillna('')
df['runtime']  = df['runtime'].fillna(100)


def build_content_features(row):
    genres = row['genres'].lower().replace(' ', '_').replace('|', ' ')
    keywords = row['keywords'].lower().replace(' ', '_').replace('|', ' ')
    director = row['director'].lower().replace(' ', '_').replace('|', ' ')
    cast = row['cast'].lower().replace(' ', '_').replace('|', ' ')
    language = row['original_language']
    overview = row['overview'].lower()
    decade = str((row['year'] // 10) * 10) + 's'

    # hard constraints
    meta_string = f"{genres}  {cast} {director} {keywords}"

    # soft
    text_string = f" {overview} {language} {decade} "

    return pd.Series([meta_string , text_string])


df[['metadata_features', 'text_features']] = df.apply(build_content_features, axis=1)

df = df.drop_duplicates(subset=['title'])

os.makedirs("data/processed", exist_ok=True)

df.to_csv("data/processed/movies.csv")

print(f"\nSaved {len(df)} movies to data/processed/movies.csv")