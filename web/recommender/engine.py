import os
import sys
import pandas as pd

# add project root to path so recomm.py is importable
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))  # shared modules live in src/

# change working directory so data/ paths in recomm.py resolve correctly
os.chdir(ROOT)

# Why this matters for Django
# Without this pattern:
# User 1 makes request → load 2.5M ratings → get recommendations (30 sec)
# User 2 makes request → load 2.5M ratings → get recommendations (30 sec)
# User 3 makes request → load 2.5M ratings → get recommendations (30 sec)
# With this pattern:
# Django starts        → load 2.5M ratings once (30 sec)
# User 1 makes request → get recommendations (instant)
# User 2 makes request → get recommendations (instant)
# User 3 makes request → get recommendations (instant)


_recommender = None

def get_recommender():
    global _recommender
    if _recommender is None:
        _recommender = _load()
    return _recommender


# First time this runs — _recommender is None, so it calls _load() which reads your CSV files, builds the collaborative matrix, trains SVD, and returns the CLI object. This takes ~30 seconds.
# Every time after that — _recommender already holds the loaded object, so it skips _load() entirely and returns immediately



def _load():
    from recomm import (
        ContentBasedRecommender,
        CollaborativeRecommender,
        SurpriseRecommender,
        HybridRecommenderEnhanced,
        MovieRecommendationSystem,
        RecommendationEvaluator,
        MovieRecommenderCLI,
        ConstraintLayer,
        NCFRecommender,
    )

    print("Loading datasets...")
    # The same data as recomm.main(): the 1,957 films with their LLM labels and full credits (the
    # ConstraintLayer needs llm_themes, llm_mood, cast_full, ...) and the half-star ratings.
    movies_df = pd.read_csv('output/experiments/deepseek_v4.1_flash_v5/movies_enriched.csv').merge(
        pd.read_csv('data/data/processed_30k/movies.csv'), on='movie_id', how='inner')
    movies_df = movies_df.drop(columns=['Unnamed: 0', 'enrich_model', 'prompt_version', 'repaired'])
    movies_df = movies_df.merge(
        pd.read_csv('data/data/credits_full/credits_full.csv', usecols=['movie_id', 'cast_full', 'directors_full']),
        on='movie_id', how='inner')
    users_df       = pd.read_csv('data/data/processed_30k/users.csv')
    interactions_df = pd.read_csv('data/data/processed_30k/ratings.csv')
    interactions_df = interactions_df[interactions_df['movie_id'].isin(movies_df['movie_id'])]

    # This line removes duplicate rows from the ratings data where the same user rated the same movie more than once.
    interactions_df = (interactions_df.sort_values('timestamp')
                       .drop_duplicates(subset=['user_id', 'movie_id'], keep='last')
                       .reset_index(drop=True))

    print("Initialising recommenders...")
    content_rec  = ContentBasedRecommender(movies_df)
    collab_rec   = CollaborativeRecommender(interactions_df)
    rec_system   = MovieRecommendationSystem(movies_df, users_df, interactions_df)
    surprise_rec = SurpriseRecommender(interactions_df)
    hybrid_rec   = HybridRecommenderEnhanced(
        content_rec, collab_rec, movies_df, users_df,
        interactions_df, rec_system, surprise_rec
    )
    evaluator      = RecommendationEvaluator(interactions_df, movies_df)
    constraint_layer = ConstraintLayer(movies_df)

    # the saved model trained on the half-star ratings, as in recomm.main()
    ncf_rec = NCFRecommender(interactions_df, movies_df, epochs=30,
                             model_dir="output/models/ncf_saved_10k_halfstar_random75_25")

    cli = MovieRecommenderCLI(
        hybrid_rec, movies_df, users_df, interactions_df,
        evaluator, surprise_rec, constraint_layer, ncf_rec
    )

    print("Engine ready.")
    return cli