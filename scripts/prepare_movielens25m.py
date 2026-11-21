import pandas as pd
import numpy as np
import os

print("Loading ML-25M files...")
ratings_raw = pd.read_csv('data/ml-25m/ratings.csv')
links       = pd.read_csv('data/ml-25m/links.csv')
your_movies = pd.read_csv("data/data/processed_30k/movies.csv")


print(f"Raw ratings: {len(ratings_raw):,}")
print(f"Links:       {len(links):,}")
print(f"Your movies: {len(your_movies):,}")

# ── STEP 1: FILTER TO ACTIVE USERS AND POPULAR MOVIES ────────────────────────
print("\nFiltering...")


user_counts  = ratings_raw['userId'].value_counts()
active_users = user_counts[user_counts >= 500].index

ratings_filtered = ratings_raw[
    ratings_raw["userId"].isin(active_users)
].copy()

print(f"After filter: {len(ratings_filtered):,} ratings")
print(f"Users:        {ratings_filtered['userId'].nunique():,}")
print(f"Movies:       {ratings_filtered['movieId'].nunique():,}")

# ── STEP 2: TRANSLATE MOVIELENS IDs → TMDB IDs VIA LINKS ─────────────────────
print("\nJoining with links...")

links_clean          = links[['movieId', 'tmdbId']].dropna()
links_clean['tmdbId'] = links_clean['tmdbId'].astype(int)

ratings_with_tmdb = ratings_filtered.merge(
    links_clean,
    on='movieId',
    how='inner'
)

print(f"After links join: {len(ratings_with_tmdb):,} ratings")

# ── STEP 3: KEEP ONLY MOVIES IN YOUR movies.csv ──────────────────────────────
print("\nMatching to your movies...")

your_movie_ids = set(your_movies['movie_id'].unique())

ratings_matched = ratings_with_tmdb[
    ratings_with_tmdb['tmdbId'].isin(your_movie_ids)
].copy()

print(f"After matching your movies: {len(ratings_matched):,} ratings")
print(f"Movies matched:             {ratings_matched['tmdbId'].nunique():,}")

matched_movie_ids = ratings_matched['tmdbId'].unique()
matched_titles = your_movies[
    your_movies['movie_id'].isin(matched_movie_ids)
][['movie_id', 'title', 'year']].sort_values('year', ascending=False)

print(f"\nSample of matched movies (newest first):")
print(matched_titles.head(20).to_string(index=False))

# ── STEP 4: BUILD FINAL DATAFRAME IN YOUR FORMAT ─────────────────────────────
print("\nBuilding final ratings dataframe...")

np.random.seed(42)
n = len(ratings_matched)

runtime_map = your_movies.set_index('movie_id')['runtime'].to_dict()

final_ratings = pd.DataFrame({
    'user_id':  ratings_matched['userId'].values + 943,
    'movie_id': ratings_matched['tmdbId'].values,
    'rating':   ratings_matched['rating'].values,
    'timestamp': ratings_matched['timestamp'].values,
})

# generate synthetic columns
runtimes = final_ratings['movie_id'].map(runtime_map).fillna(120).astype(int)
final_ratings['watch_duration']  = (runtimes * np.random.uniform(0.5, 1.0, n)).astype(int)
final_ratings['view_count']      = np.random.randint(1, 5, n)
final_ratings['completion_rate'] = np.minimum(
    1.0,
    final_ratings['watch_duration'] / runtimes
).round(2)

# Keep ML-25M's original half-star ratings (0.5–5.0).
# They used to be rounded with .round(0), which rounds halves to the nearest EVEN
# number (3.5 -> 4, 4.5 -> 4, 2.5 -> 2, 0.5 -> 0 -> clipped to 1) and distorted 34%
# of the ratings. Surprise SVD handles half stars with Reader(rating_scale=(0.5, 5)).
final_ratings['rating'] = final_ratings['rating'].astype(float)

# deduplicate — keep last rating if user rated same movie twice
final_ratings = final_ratings.drop_duplicates(
    subset=['user_id', 'movie_id'],
    keep='last'
)

print(f"\nML-25M final ratings: {len(final_ratings):,}")
print(f"Users:                {final_ratings['user_id'].nunique():,}")
print(f"Movies:               {final_ratings['movie_id'].nunique():,}")
print(f"Rating distribution:\n{final_ratings['rating'].value_counts().sort_index()}")

# ── STEP 5: COMBINE ML-100K + ML-25M THEN SAVE ───────────────────────────────
print("\nCombining with original ML-100K ratings...")

output_dir = "data/data/processed_30k"
output_path = os.path.join(output_dir, 'ratings.csv')

ml100k_ratings = pd.read_csv(
    "data/data/processed_ml100k_v2/ratings.csv"
)

print(f"ML-100K ratings loaded: {len(ml100k_ratings):,}")
print(f"ML-25M ratings:         {len(final_ratings):,}")

# combine both datasets
combined = pd.concat([ml100k_ratings, final_ratings], ignore_index=True)
combined = combined.drop_duplicates(subset=['user_id', 'movie_id'], keep='last')

print(f"Combined total:         {len(combined):,}")
print(f"Users 1-943 present:    {combined[combined['user_id'] <= 943]['user_id'].nunique()}")
print(f"Avg ratings per user:   {len(combined) / combined['user_id'].nunique():.1f}")

os.makedirs(output_dir, exist_ok=True)
combined.to_csv(output_path, index=False)
print(f"Saved to {output_path}")


# ── STEP 6: UPDATE USERS.CSV ─────────────────────────────────────────────────
print("\nUpdating users.csv...")

# reset to original 943 ML-100K users only
# prevents duplicates if this script is run multiple times
old_users = pd.read_csv(
    "data/data/processed_ml100k_v2/users.csv"
)
old_users = old_users[old_users['user_id'] <= 943].copy()
print(f"Reset to original users: {len(old_users)}")

old_users_ids_set = set(old_users['user_id'].unique())
new_users_ids_set = set(combined['user_id'].unique())

# users in ML-25M that are NOT in the original 943
truly_new_users_ids = sorted(new_users_ids_set - old_users_ids_set)

print(f"Truly new ML-25M users: {len(truly_new_users_ids)}")

# ML-25M users lack demographic data
# age defaults to 25 for constraint evaluation purposes
# child safety constraints apply only to original 943 ML-100K users
# who have verified demographic profiles
synthetic_users = pd.DataFrame({
    'user_id':    truly_new_users_ids,
    'age':        25,
    'gender':     'Unknown',
    'occupation': 'Unknown',
    'zip_code':   'Unknown',
    'age_group':  'Adult_25-34',
    'country':    'Unknown'
})

users_updated = pd.concat(
    [old_users, synthetic_users],
    ignore_index=True
)

users_updated.to_csv(os.path.join(output_dir, 'users.csv'), index=False)


print(f"\nVerification:")
print(f"  ML-100K users (real demographics): {len(old_users_ids_set)}")
print(f"  ML-25M users  (age=25, synthetic): {len(truly_new_users_ids)}")
print(f"  Total:                             {len(users_updated)}")

# ── FINAL SUMMARY ─────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("PIPELINE COMPLETE")
print(f"  movies.csv  : {len(your_movies):,} movies")
print(f"  ratings.csv : {len(combined):,} ratings")
print(f"  users.csv   : {len(users_updated):,} users")
print(f"  Year range  : {your_movies['year'].min()} – {your_movies['year'].max()}")
print("=" * 55)