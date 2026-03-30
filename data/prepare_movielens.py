# prepare_movielens.py
# ─────────────────────────────────────────────────────
# Job: download MovieLens 100K, match with TMDB movies,
#      produce ratings.csv and users.csv
# ─────────────────────────────────────────────────────

import pandas as pd
import numpy as np
import urllib.request
import zipfile
import os

from difflib import SequenceMatcher


ML_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
ML_DIR = "data/raw/ml-100k"

# ── Step 1: Download MovieLens 100K ──────────────────

if not os.path.isdir(ML_DIR):
    print("Downloading MovieLens 100K...")
    urllib.request.urlretrieve(ML_URL, "data/raw/ml-100k.zip")

    with zipfile.ZipFile("data/raw/ml-100k.zip", "r") as z:
        z.extractall("data/raw/")

    os.remove("data/raw/ml-100k.zip")
    print("Done.")
else:
    print("MovieLens 100K already exists, skipping download.")


# ── Step 2: Load MovieLens files ──────────────────────

ratings_raw = pd.read_csv(
    "data/raw/ml-100k/u.data",
    sep='\t', header=None,
    names=['user_id', 'movie_id', 'rating', 'timestamp']
)

ml_movies = pd.read_csv(
    "data/raw/ml-100k/u.item",
    sep='|', encoding='latin-1', header=None,
    usecols=[0, 1],
    names=['ml_movie_id', 'ml_title']
)

ml_users = pd.read_csv(
    "data/raw/ml-100k/u.user",
    sep='|', header=None,
    names=['user_id', 'age', 'gender', 'occupation', 'zip_code']
)

print(f"Loaded {len(ratings_raw)} ratings")
print(f"Loaded {len(ml_movies)} MovieLens movies")
print(f"Loaded {len(ml_users)} users")


# ── Step 3: Parse title and year from MovieLens ───────

ml_movies['year']  = ml_movies['ml_title'].str.extract(r'\((\d{4})\)').astype(float)
ml_movies['title'] = ml_movies['ml_title'].str.replace(r'\s*\(\d{4}\)', '', regex=True).str.strip()


# ── Step 4: Load your TMDB movies ────────────────────

tmdb_movies = pd.read_csv("data/processed/movies.csv")

ml_movies['year']   = ml_movies['year'].fillna(0).astype(int)
tmdb_movies['year'] = tmdb_movies['year'].astype(int)


# ── Step 5: Match MovieLens movies to TMDB movies ────

# ── Step 5: Fuzzy title matching ─────────────────────



def similarity(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


# build a lookup from TMDB: normalized_title → movie_id
tmdb_lookup = {
    row['title'].lower(): row['movie_id']
    for _, row in tmdb_movies.iterrows()
}
tmdb_titles = list(tmdb_lookup.keys())

matches = []
for _, ml_row in ml_movies.iterrows():
    ml_title = ml_row['title'].lower()
    ml_year = ml_row['year']

    # find best matching TMDB title
    best_score = 0
    best_title = None
    for tmdb_title in tmdb_titles:
        score = similarity(ml_title, tmdb_title)
        if score > best_score:
            best_score = score
            best_title = tmdb_title

    # only accept if similarity is high enough
    if best_score >= 0.85:
        tmdb_year = tmdb_movies[tmdb_movies['title'].str.lower() == best_title]['year'].values

        if len(tmdb_year) > 0:
            year_diff = abs(int(ml_year) - int(tmdb_year[0]))

            if year_diff <= 1:
                tmdb_id = tmdb_lookup[best_title]
                matches.append({
                    'ml_movie_id': ml_row['ml_movie_id'],
                    'movie_id': tmdb_id,
                    'ml_title': ml_row['ml_title'],
                    'tmdb_title': best_title,
                    'score': round(best_score, 3)
                })


matched = pd.DataFrame(matches)

print(f"\nMovieLens movies       : {len(ml_movies)}")
print(f"TMDB movies            : {len(tmdb_movies)}")
print(f"Matched (overlap)      : {len(matched)}")
print(f"\nAll matched titles:")
for _, row in matched.iterrows():
    print(f"  {row['score']:.3f}  '{row['ml_title']}'  →  '{row['tmdb_title']}'")


# ── Step 6: Filter ratings to matched movies only ────

matched_ml_ids   = set(matched['ml_movie_id'])
ratings_filtered = ratings_raw[ratings_raw['movie_id'].isin(matched_ml_ids)]

ratings_filtered = pd.merge(
    ratings_filtered,
    matched[['ml_movie_id', 'movie_id']],
    left_on='movie_id',
    right_on='ml_movie_id'
)

ratings_filtered = ratings_filtered.drop(columns=['movie_id_x', 'ml_movie_id'])
ratings_filtered = ratings_filtered.rename(columns={'movie_id_y': 'movie_id'})

# add watch_duration and completion_rate
np.random.seed(42)
n              = len(ratings_filtered)
base_completion = (ratings_filtered['rating'] - 1) / 4 * 0.60 + 0.35
ratings_filtered['completion_rate'] = (base_completion + np.random.normal(0, 0.05, n)).clip(0.1, 1.0).round(2)
ratings_filtered['watch_duration']  = (ratings_filtered['completion_rate'] * 100).astype(int)
ratings_filtered['view_count']      = 1


# ── Step 7: Build users.csv ───────────────────────────

bins   = [0,   25,   35,   45,   55,  100]
labels = ['Youth_15-24', 'Adult_25-34', 'Adult_35-44', 'Adult_45-54', 'Senior_55+']

ml_users['age_group'] = pd.cut(ml_users['age'], bins=bins, labels=labels, right=False)
ml_users['country']   = 'US'


#Quality check

print(f"\nFinal ratings          : {len(ratings_filtered)}")
print(f"Unique users           : {ratings_filtered['user_id'].nunique()}")
print(f"Unique movies          : {ratings_filtered['movie_id'].nunique()}")
print(f"Rating mean            : {ratings_filtered['rating'].mean():.2f}")
density = len(ratings_filtered) / (ratings_filtered['user_id'].nunique() * ratings_filtered['movie_id'].nunique()) * 100
print(f"Matrix density         : {density:.2f}%")


# Save

os.makedirs("data/processed", exist_ok=True)

ratings_filtered.to_csv("data/processed/ratings.csv", index=False)
ml_users.to_csv("data/processed/users.csv", index=False)

print("\nSaved ratings.csv and users.csv to data/processed/")
