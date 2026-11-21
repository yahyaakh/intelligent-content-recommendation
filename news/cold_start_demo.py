# =============================================================================
# cold_start_demo.py
# =============================================================================
# Purpose  : Demonstrate the Cold Start transition in a CMS recommendation system
# Dataset  : Microsoft MIND (real news articles + real user click behavior)
# Research : How and when does a system transition from cold start to
#            personalized recommendations?
# Author   : Yahya Akhrikhar
# =============================================================================

import pandas as pd
from cold_start_handler import cold_start_recommendation

# ─── CONFIGURATION ───────────────────────────────────────────────────────────
N_RECOMMENDATIONS = 5
DATA_PATH = 'data/Mind_data'

# ─── HELPERS ─────────────────────────────────────────────────────────────────

def print_header():
    print("\n" + "═" * 70)
    print("   COLD START DEMO — CMS Article Recommender")
    print("   Dataset: Microsoft MIND (Real News Articles + Real User Behavior)")
    print("═" * 70)

def print_section(title):
    print("\n" + "─" * 70)
    print(f"  {title}")
    print("─" * 70)

def print_recommendations(recs, strategy, category=None):
    tag = f"  Strategy : {strategy}"
    if category:
        tag += f"  (dominant interest: {category.upper()})"
    print(tag)
    print()
    if not recs:
        print("  ⚠  No recommendations found — dataset too sparse for this user.")
        return
    for i, article in enumerate(recs, 1):
        cat = f"[{article['category']:<15}]"
        title = article['title'][:55] + '...' if len(article['title']) > 55 else article['title']
        print(f"  {i}.  {cat}  {title}")

def get_dominant_category(user_id, interactions_df, articles_df):
    """Find the most clicked category for a user."""
    user_clicks = interactions_df[interactions_df['user_id'] == user_id]
    merged = user_clicks.merge(articles_df[['article_id', 'category']], on='article_id')
    if merged.empty:
        return None
    return merged['category'].mode().iloc[0]

def print_user_history(user_id, interactions_df, articles_df, max_show=3):
    """Print a short sample of what the user has already clicked."""
    user_clicks = interactions_df[interactions_df['user_id'] == user_id]
    merged = user_clicks.merge(
        articles_df[['article_id', 'title', 'category']], on='article_id'
    ).head(max_show)
    if merged.empty:
        print("  Click history : (none — brand new user)")
        return
    print(f"  Click history : (showing {len(merged)} of {len(user_clicks)} clicks)")
    for _, row in merged.iterrows():
        title = row['title'][:50] + '...' if len(row['title']) > 50 else row['title']
        print(f"    ✓  [{row['category']:<15}]  {title}")

def print_transition_table():
    print("\n" + "═" * 70)
    print("  TRANSITION THRESHOLDS — When Does Personalization Kick In?")
    print("═" * 70)
    print()
    print(f"  {'Interactions':<18} {'Mode':<10} {'Algorithm':<30} {'Why'}")
    print(f"  {'─'*16:<18} {'─'*8:<10} {'─'*28:<30} {'─'*20}")
    print(f"  {'0 – 2':<18} {'COLD':<10} {'Popularity-based':<30} No personal signal available")
    print(f"  {'3 – 9':<18} {'WARM':<10} {'Content-based (TF-IDF)':<30} Topic interest detected")
    print(f"  {'10+':<18} {'HOT':<10} {'Collaborative filtering':<30} Behavioral patterns usable")
    print()
    print("  ℹ  Mean-centering was evaluated but not applied for binary click")
    print("     data — when all signals are 0/1, subtracting the mean collapses")
    print("     every score to 0. Raw cosine similarity is used instead.")
    print()
    print("═" * 70)

def print_dataset_summary(articles_df, users_df, interactions_df):
    mode_counts = users_df['mode'].value_counts()
    print_section("DATASET OVERVIEW")
    print(f"  Source        : Microsoft MIND-small (real news + real user behavior)")
    print(f"  Articles      : {len(articles_df):,} news articles across {articles_df['category'].nunique()} categories")
    print(f"  Users         : {len(users_df):,} users")
    print(f"  Interactions  : {len(interactions_df):,} article clicks")
    print(f"  Avg clicks/user : {interactions_df.groupby('user_id').size().mean():.1f}")
    print()
    print(f"  Cold start distribution:")
    for mode in ['COLD', 'WARM', 'HOT']:
        count = mode_counts.get(mode, 0)
        bar = '█' * int(count / len(users_df) * 40)
        pct = count / len(users_df) * 100
        print(f"    {mode:<6}  {bar:<42}  {count:>4} users ({pct:.1f}%)")
    print()
    print(f"  Article categories:")
    for cat, count in articles_df['category'].value_counts().head(6).items():
        bar = '░' * int(count / len(articles_df) * 30)
        print(f"    {cat:<20}  {bar:<32}  {count} articles")


# ─── MAIN DEMO ───────────────────────────────────────────────────────────────

def main():
    # Load data
    print("\nLoading MIND dataset...")
    articles_df     = pd.read_csv(f'{DATA_PATH}/articles.csv')
    users_df        = pd.read_csv(f'{DATA_PATH}/users.csv')
    interactions_df = pd.read_csv(f'{DATA_PATH}/interactions.csv')

    # Initialize handler
    handler = cold_start_recommendation(articles_df, users_df, interactions_df)

    # Print header
    print_header()

    # Print dataset summary
    print_dataset_summary(articles_df, users_df, interactions_df)

    # ── Pick one real user from each mode ────────────────────────────────────
    cold_user = users_df[users_df['mode'] == 'COLD'].iloc[0]
    warm_user = users_df[users_df['mode'] == 'WARM'].iloc[4]  # skip a few for variety
    hot_user  = users_df[users_df['mode'] == 'HOT'].iloc[0]

    # ── COLD USER ────────────────────────────────────────────────────────────
    print_section(
        f"STAGE 1 — COLD USER  |  {cold_user['user_id']}  "
        f"|  {cold_user['n_interactions']} interaction(s)"
    )
    print_user_history(cold_user['user_id'], interactions_df, articles_df)
    print()
    recs = handler.recommend(cold_user['user_id'], N_RECOMMENDATIONS)
    print_recommendations(
        recs,
        strategy="Popularity-based  —  showing what most users read"
    )

    # ── WARM USER ────────────────────────────────────────────────────────────
    dominant_cat = get_dominant_category(
        warm_user['user_id'], interactions_df, articles_df
    )
    print_section(
        f"STAGE 2 — WARM USER  |  {warm_user['user_id']}  "
        f"|  {warm_user['n_interactions']} interactions"
    )
    print_user_history(warm_user['user_id'], interactions_df, articles_df)
    print()
    recs = handler.recommend(warm_user['user_id'], N_RECOMMENDATIONS)
    print_recommendations(
        recs,
        strategy="Content-based TF-IDF  —  matching article topics to user taste",
        category=dominant_cat
    )

    # ── HOT USER ─────────────────────────────────────────────────────────────
    dominant_cat_hot = get_dominant_category(
        hot_user['user_id'], interactions_df, articles_df
    )
    print_section(
        f"STAGE 3 — HOT USER   |  {hot_user['user_id']}  "
        f"|  {hot_user['n_interactions']} interactions"
    )
    print_user_history(hot_user['user_id'], interactions_df, articles_df)
    print()
    recs = handler.recommend(hot_user['user_id'], N_RECOMMENDATIONS)
    print_recommendations(
        recs,
        strategy="Collaborative filtering  —  based on users with similar behavior",
        category=dominant_cat_hot
    )

    # ── TRANSITION TABLE ─────────────────────────────────────────────────────
    print_transition_table()

    print("\n  Demo complete.\n")


if __name__ == "__main__":
    main()