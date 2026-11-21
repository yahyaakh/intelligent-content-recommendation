# Purpose of this file:
# first : read the news.tsv and behaviors.tsv from the mind data
# output : clean articles.csv, users.csv, interactions.csv
# this script is one time only, run once then done

import pandas as pd
import os

# ─── CONFIGURATION ───────────────────────────────────────────
OUTPUT_FOLDER = "data/Mind_data"
MAX_ARTICLES = 5000                    # how many articles to keep
MAX_USERS = 2000                          # how many users to keep

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ─── STEP 1: PARSE NEWS.TSV ──────────────────────────────────
print("Reading news.tsv...")

col_names = ['article_id', 'category', 'subcategory', 'title', 'abstract', 'url', 'entities', 'relations']

articles_df = pd.read_csv(
    "data/Mind_data/raw/news.tsv",
    sep="\t",
    names=col_names,
    header=None
)

# Keep only the 4 columns you need
articles_df = articles_df[['article_id', 'category', 'title', 'abstract']]

# Drop rows with any null value
articles_df = articles_df.dropna()

# Remove duplicate articles
articles_df = articles_df.drop_duplicates(subset='article_id')

# Build content_features — mirrors your movies content_features column
# category repeated twice to boost its weight in TF-IDF (same trick as movies)
articles_df['content_features'] = (
    articles_df['category'] + ' ' +
    articles_df['category'] + ' ' +
    articles_df['title'] + ' ' +
    articles_df['abstract']
)

# Sample down to MAX_ARTICLES
articles_df = articles_df.head(MAX_ARTICLES).reset_index(drop=True)

articles_df.to_csv(f"{OUTPUT_FOLDER}/articles.csv", index=False)
print(f"Articles saved: {len(articles_df)} rows")
print(articles_df.head(3))


# ─── STEP 2: PARSE BEHAVIORS.TSV ─────────────────────────────
print("\nReading behaviors.tsv...")

# IMPORTANT: correct column order is impression_id, user_id, timestamp, history, impressions
behaviors_col_names = ['impression_id', 'user_id', 'timestamp', 'history', 'impressions']

behaviors_df = pd.read_csv(
    "data/Mind_data/raw/behaviors.tsv",
    sep="\t",
    names=behaviors_col_names,
    header=None
)

# Sample down to MAX_USERS unique users
unique_users = behaviors_df['user_id'].unique()[:MAX_USERS]
behaviors_df = behaviors_df[behaviors_df['user_id'].isin(unique_users)]

print(f"Behaviors loaded: {len(behaviors_df)} sessions for {len(unique_users)} users")


# ─── STEP 3: EXTRACT INTERACTIONS ────────────────────────────
print("\nExtracting interactions...")

# Keep only articles that exist in our sampled articles_df
valid_article_ids = set(articles_df['article_id'].tolist())

interactions = []

for _, row in behaviors_df.iterrows():
    user_id = row['user_id']
    timestamp = row['timestamp']

    # --- Source 1: History column ---
    # These are all past clicks — every article listed = clicked
    if pd.notna(row['history']) and row['history'] != '':
        history_articles = row['history'].strip().split(' ')
        for article_id in history_articles:
            article_id = article_id.strip()
            if article_id in valid_article_ids:
                interactions.append({
                    'user_id': user_id,
                    'article_id': article_id,
                    'clicked': 1,
                    'timestamp': timestamp
                })

    # --- Source 2: Impressions column ---
    # Format is "N12345-1 N67890-0 ..." where -1 = clicked, -0 = ignored
    if pd.notna(row['impressions']) and row['impressions'] != '':
        impression_items = row['impressions'].strip().split(' ')
        for item in impression_items:
            if '-' not in item:
                continue
            article_id, click_label = item.rsplit('-', 1)
            if click_label == '1' and article_id in valid_article_ids:
                interactions.append({
                    'user_id': user_id,
                    'article_id': article_id,
                    'clicked': 1,
                    'timestamp': timestamp
                })
            # click_label == '0' → ignored, we skip it

interactions_df = pd.DataFrame(interactions)

# Remove duplicate user-article pairs (keep first occurrence)
interactions_df = interactions_df.drop_duplicates(subset=['user_id', 'article_id'])

interactions_df.to_csv(f"{OUTPUT_FOLDER}/interactions.csv", index=False)
print(f"Interactions saved: {len(interactions_df)} rows")
print(interactions_df.head(3))


# ─── STEP 4: BUILD USERS.CSV ─────────────────────────────────
print("\nBuilding users.csv...")

# Count interactions per user
user_counts = interactions_df.groupby('user_id')['article_id'].count().reset_index()
user_counts.columns = ['user_id', 'n_interactions']

# Assign cold start mode based on interaction count
def assign_mode(n):
    if n < 3:
        return 'COLD'
    elif n < 10:
        return 'WARM'
    else:
        return 'HOT'

user_counts['mode'] = user_counts['n_interactions'].apply(assign_mode)

user_counts.to_csv(f"{OUTPUT_FOLDER}/users.csv", index=False)
print(f"Users saved: {len(user_counts)} rows")
print(user_counts['mode'].value_counts())
print(user_counts.head(3))


# ─── SUMMARY ─────────────────────────────────────────────────
print("\n" + "="*50)
print("MIND DATA PIPELINE COMPLETE")
print(f"  Articles:     {len(articles_df)}")
print(f"  Users:        {len(user_counts)}")
print(f"  Interactions: {len(interactions_df)}")
print(f"  Avg interactions per user: {interactions_df.groupby('user_id').size().mean():.1f}")
print(f"\nCold start distribution:")
print(user_counts['mode'].value_counts().to_string())
print(f"\nSaved to: {OUTPUT_FOLDER}/")