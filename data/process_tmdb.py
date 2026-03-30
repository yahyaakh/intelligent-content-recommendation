
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
    # hint: get genres, replace '|' with spaces, lowercase
    genres = row['genres'].replace('|', ' ').lower()
    keywords = row['keywords'].replace('|', ' ').lower()
    director = row['director'].lower().replace(' ', '_')
    cast = row['cast'].replace('|', ' ').lower()
    language = row['original_language']
    overview = row['overview'].lower()
    decade = str((row['year'] // 10) * 10) + 's'

    return f"{genres} {genres} {cast} {director} {keywords} {language} {decade} {overview}"

df['content_features'] = df.apply(build_content_features, axis=1)


print("\n── Shape:", df.shape)
print("── Nulls:\n", df.isnull().sum())
print("\n── Sample content_features:\n", df['content_features'].iloc[0])


os.makedirs("data/processed", exist_ok=True)

df.to_csv("data/processed/movies.csv")

print(f"\nSaved {len(df)} movies to data/processed/movies.csv")