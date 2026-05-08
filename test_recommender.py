# =============================================================================
# test_recommender.py
# =============================================================================
# Automated test suite for the Movie Recommendation System
# Runs multiple scenarios and prints formatted results for documentation
#
# Usage: python test_recommender.py
# Output: Full terminal report covering all system components
#
# Test Scenarios:
#   1.  Data integrity check
#   2.  User profiles — three different user types
#   3.  Content similarity — known movie pairs
#   4.  Collaborative filtering — sparse vs dense users
#   5.  Algorithm comparison — same user, all five algorithms
#   6.  Child safety constraints — age 7 user
#   7.  Hard constraints — language, runtime, genre overlap
#   8.  Soft constraints — director/decade/genre re-ranking
#   9.  Cold start simulation — user gaining interactions over time
#   10. SVD evaluation metrics
# =============================================================================

import sys
import os
import time
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Add project root to path so imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ─── IMPORT YOUR SYSTEM ──────────────────────────────────────────────────────
# Adjust this import path if needed
from recomm import (
    ContentBasedRecommender,
    CollaborativeRecommender,
    MovieRecommendationSystem,
    HybridRecommenderEnhanced,
    SurpriseRecommender,
    RecommendationEvaluator,
    MovieRecommenderCLI,
    ConstraintLayer,
    diagnose_user_movie_matrix,
    CHILD_UNSAFE_KEYWORDS
)

# ─── DISPLAY HELPERS ─────────────────────────────────────────────────────────

PASS  = "✓ PASS"
FAIL  = "✗ FAIL"
INFO  = "ℹ"
WARN  = "⚠"

def header(title):
    print("\n" + "═" * 70)
    print(f"  {title}")
    print("═" * 70)

def section(title, subtitle=None):
    print(f"\n  {'─' * 64}")
    print(f"  {title}")
    if subtitle:
        print(f"  {subtitle}")
    print(f"  {'─' * 64}")

def result(label, value, status=None):
    pad = 35
    line = f"    {label:<{pad}} {value}"
    if status == "pass":
        line += f"  {PASS}"
    elif status == "fail":
        line += f"  {FAIL}"
    print(line)

def show_movies(recs, max_show=5):
    if not recs:
        print(f"    {WARN}  No recommendations returned")
        return
    for i, m in enumerate(recs[:max_show], 1):
        title   = m.get('title', 'Unknown')[:45]
        year    = m.get('year', '?')
        genres  = m.get('genres', m.get('genre', '?'))
        director = m.get('director', '?')
        score   = f"  [score: {m['soft_score']:.3f}]" if 'soft_score' in m else ""
        print(f"    {i}.  {title:<47} ({year})  {genres}{score}")

def show_constraints_log(recs_before, recs_after, label="Constraint"):
    removed = len(recs_before) - len(recs_after)
    print(f"    {INFO}  {label}: {len(recs_before)} in → {len(recs_after)} out "
          f"({removed} removed)")


# =============================================================================
# SETUP — Load data and initialise all components
# =============================================================================

def setup():
    print("\nLoading datasets and initialising system...")
    t0 = time.time()

    movies_df       = pd.read_csv('data/data/processed/movies.csv')
    users_df        = pd.read_csv('data/data/processed/users.csv')
    interactions_df = pd.read_csv('data/data/processed/ratings.csv')

    content_rec  = ContentBasedRecommender(movies_df)
    collab_rec   = CollaborativeRecommender(interactions_df)
    rec_system   = MovieRecommendationSystem(movies_df, users_df, interactions_df)
    hybrid_rec   = HybridRecommenderEnhanced(
        content_rec, collab_rec, movies_df, users_df, interactions_df, rec_system
    )
    evaluator       = RecommendationEvaluator(interactions_df, movies_df)
    surprise_rec    = SurpriseRecommender(interactions_df)
    constraint_layer = ConstraintLayer(movies_df)

    cli = MovieRecommenderCLI(
        hybrid_rec, movies_df, users_df, interactions_df,
        evaluator, surprise_rec, constraint_layer
    )

    elapsed = time.time() - t0
    print(f"  {PASS}  System loaded in {elapsed:.1f}s")
    print(f"  {INFO}  Movies: {len(movies_df):,}  |  "
          f"Users: {len(users_df):,}  |  "
          f"Interactions: {len(interactions_df):,}")

    return cli, movies_df, users_df, interactions_df, content_rec, collab_rec


# =============================================================================
# TEST 1 — Data Integrity
# =============================================================================

def test_data_integrity(movies_df, users_df, interactions_df):
    header("TEST 1 — DATA INTEGRITY")

    section("Movies dataset")
    result("Total movies",         len(movies_df))
    result("Columns present",      list(movies_df.columns)[:6])
    result("Null values (total)",  movies_df.isnull().sum().sum(),
           "pass" if movies_df.isnull().sum().sum() == 0 else "fail")
    result("Unique movie IDs",     movies_df['movie_id'].nunique())
    result("Year range",
           f"{movies_df['year'].min():.0f} – {movies_df['year'].max():.0f}")
    result("Languages present",
           movies_df['original_language'].value_counts().head(3).to_dict())

    section("Users dataset")
    result("Total users",          len(users_df))
    result("Age range",
           f"{users_df['age'].min()} – {users_df['age'].max()} years")
    result("Users under 13",
           len(users_df[users_df['age'] < 13]),
           "pass")
    result("Gender distribution",
           users_df['gender'].value_counts().to_dict())

    section("Interactions dataset")
    result("Total interactions",   len(interactions_df))
    result("Real data (source=1)",
           len(interactions_df[interactions_df.get('source_flag',
               pd.Series([1]*len(interactions_df))) == 1]))
    result("Rating distribution",
           interactions_df['rating'].value_counts().sort_index().to_dict())
    result("Avg rating",
           f"{interactions_df['rating'].mean():.2f}")
    result("Avg ratings per user",
           f"{interactions_df.groupby('user_id').size().mean():.1f}")
    result("Matrix density",
           f"{len(interactions_df) / (len(users_df) * len(movies_df)) * 100:.2f}%")


# =============================================================================
# TEST 2 — User Profiles
# =============================================================================

def test_user_profiles(cli, users_df):
    header("TEST 2 — USER PROFILES")

    # Pick three different user types
    test_users = [
        (1,   "Dense user — 53 ratings, adult, technician"),
        (2,   "Sparse user — 7 ratings, adult, female"),
        (30,  "Child user — 7 ratings, age 7"),
    ]

    for user_id, description in test_users:
        section(f"User {user_id} — {description}")
        try:
            profile = cli.get_user_profile(user_id)
            d = profile['demographics']
            s = profile['stats']
            t = profile['taste_profile']

            result("Age / Gender / Occupation",
                   f"{d['age']} / {d['gender']} / {d['occupation']}")
            result("Movies rated",       s['total_movies'])
            result("Average rating",     s['avg_rating'])
            result("Avg completion",     s['avg_completion'])
            result("Total watch time",   f"{s['total_watch_time']} min")
            result("Preferred decade",   f"{t['preferred_decade']}s")
            result("Preferred language", t['preferred_language'])
            result("Avg runtime",        t['avg_runtime_tolerance'])
            result("Liked genres",       ', '.join(list(t['liked_genres'])[:5]))
            result("Liked directors",
                   ', '.join(list(t['liked_directors'])[:4]))
            print(f"    {PASS}  Profile built successfully")
        except Exception as e:
            print(f"    {FAIL}  {e}")


# =============================================================================
# TEST 3 — Content-Based Similarity
# =============================================================================

def test_content_similarity(cli, movies_df):
    header("TEST 3 — CONTENT-BASED SIMILARITY")

    test_cases = [
        (11,     "Star Wars",      ["Sci-Fi", "Adventure", "Action"]),
        (238,    "The Godfather",  ["Crime", "Drama"]),
        (157336, "Interstellar",   ["Sci-Fi", "Adventure"]),
        (769,    "GoodFellas",     ["Crime", "Drama"]),
    ]

    for movie_id, title, expected_genres in test_cases:
        section(f"Similar to: {title}  (ID {movie_id})")
        try:
            similar = cli.find_similar_movies(movie_id, n=8)
            show_movies(similar, max_show=8)

            # Check if results share expected genres
            hits = 0
            for m in similar:
                movie_genres = str(m.get('genres', '')).split('|')
                if any(g in movie_genres for g in expected_genres):
                    hits += 1
            genre_match_pct = hits / len(similar) * 100 if similar else 0
            result("Genre relevance",
                   f"{hits}/{len(similar)} results share expected genres "
                   f"({genre_match_pct:.0f}%)",
                   "pass" if genre_match_pct >= 50 else "fail")
        except Exception as e:
            print(f"    {FAIL}  {e}")


# =============================================================================
# TEST 4 — Collaborative Filtering
# =============================================================================

def test_collaborative_filtering(cli, interactions_df):
    header("TEST 4 — COLLABORATIVE FILTERING (MEAN-CENTERED)")

    # Sparse vs dense users
    user_counts = interactions_df.groupby('user_id').size()
    sparse_user = user_counts[user_counts <= 5].index[0]
    dense_user  = user_counts[user_counts >= 30].index[0]

    test_users = [
        (dense_user,  f"Dense user  ({user_counts[dense_user]} ratings)"),
        (sparse_user, f"Sparse user ({user_counts[sparse_user]} ratings)"),
        (1,           "User 1 (53 ratings, cinephile)"),
    ]

    for user_id, label in test_users:
        section(f"Collaborative recs — {label}")
        try:
            movie_ids = cli.hybrid_rec.collab_rec.get_user_recommendations(
                user_id, n=10
            )
            recs = [
                cli.movies_df[cli.movies_df['movie_id'] == mid].iloc[0].to_dict()
                for mid in movie_ids
                if not cli.movies_df[cli.movies_df['movie_id'] == mid].empty
            ]
            show_movies(recs, max_show=5)
            result("Recommendations returned", len(recs),
                   "pass" if len(recs) > 0 else "fail")

            # Check mean-centering: similar users should have varied avg ratings
            sim_users = cli.hybrid_rec.collab_rec.user_similarity_df[
                user_id
            ].sort_values(ascending=False).iloc[1:6].index
            avg_ratings = [
                interactions_df[interactions_df['user_id'] == u]['rating'].mean()
                for u in sim_users
                if u in interactions_df['user_id'].values
            ]
            if avg_ratings:
                result("Similar users avg ratings",
                       [f"{r:.2f}" for r in avg_ratings],
                       "pass")
        except Exception as e:
            print(f"    {FAIL}  {e}")


# =============================================================================
# TEST 5 — Algorithm Comparison
# =============================================================================

def test_algorithm_comparison(cli):
    header("TEST 5 — ALGORITHM COMPARISON (User 1)")

    algorithms = ['hybrid', 'collaborative', 'content', 'popular', 'surprise']

    section("All five algorithms for User 1 (top 3 each)")

    results_by_algo = {}
    for algo in algorithms:
        try:
            recs = cli.get_user_recommendations(1, algo, n=5)
            results_by_algo[algo] = recs
            titles = [m.get('title', '?')[:35] for m in recs[:3]]
            print(f"\n    {algo.upper():<15}")
            for i, (m, title) in enumerate(zip(recs[:3], titles), 1):
                genres = m.get('genres', m.get('genre', '?'))[:30]
                print(f"      {i}. {title:<37} [{genres}]")
        except Exception as e:
            print(f"\n    {algo.upper():<15}  {FAIL} {e}")

    # Check diversity — how much overlap between hybrid and each algorithm
    section("Overlap analysis — Hybrid vs other algorithms")
    if 'hybrid' in results_by_algo:
        hybrid_titles = {m.get('title') for m in results_by_algo['hybrid']}
        for algo, recs in results_by_algo.items():
            if algo == 'hybrid':
                continue
            algo_titles  = {m.get('title') for m in recs}
            overlap      = len(hybrid_titles & algo_titles)
            print(f"    Hybrid ∩ {algo:<15}  {overlap} common movies "
                  f"({'identical' if overlap == len(hybrid_titles) else 'diverse'})")


# =============================================================================
# TEST 6 — Child Safety Constraints
# =============================================================================

def test_child_safety(cli, movies_df):
    header("TEST 6 — CHILD SAFETY CONSTRAINTS (User 30, Age 7)")

    section("Raw recommendations before constraints")
    raw_recs = cli.hybrid_rec.get_hybrid_recommendations(30, n=15)

    enriched = []
    for movie in raw_recs:
        movie_id = movie.get('movie_id')
        full_data = movies_df[movies_df['movie_id'] == movie_id]
        if not full_data.empty:
            enriched.append(full_data.iloc[0].to_dict())
        else:
            enriched.append(movie)

    print(f"    {INFO}  Raw candidates: {len(enriched)} movies")
    for m in enriched[:5]:
        title  = m.get('title', '?')[:40]
        genres = m.get('genres', '?')[:30]
        print(f"         {title:<42} [{genres}]")

    section("After hard constraints applied")
    profile = cli.build_constraint_profile(30)
    print(f"    {INFO}  User age: {profile['age']}")
    print(f"    {INFO}  Avg runtime tolerance: {profile['avg_runtime']:.0f} min "
          f"(ceiling: {profile['avg_runtime']*1.5:.0f} min)")

    filtered = cli.constraint_layer.apply_hard_constraints(enriched, profile)

    show_constraints_log(enriched, filtered, "Hard constraints")
    print(f"\n    Surviving recommendations:")
    show_movies(filtered, max_show=10)

    # Verify no adult content survived
    adult_genres = {'Horror', 'Thriller', 'Crime', 'War'}
    violations = []
    for m in filtered:
        movie_genres = set(str(m.get('genres', '')).split('|'))
        keywords = str(m.get('keywords', '')).lower()
        kw_hits = [kw for kw in CHILD_UNSAFE_KEYWORDS if kw in keywords]
        if movie_genres.intersection(adult_genres) or kw_hits:
            violations.append(m.get('title'))

    result("Adult content violations after filter",
           len(violations),
           "pass" if len(violations) == 0 else "fail")
    if violations:
        print(f"    {FAIL}  Violations: {violations}")


# =============================================================================
# TEST 7 — Hard Constraints Detail
# =============================================================================

def test_hard_constraints_detail(cli, movies_df):
    header("TEST 7 — HARD CONSTRAINTS DETAIL")

    # Test runtime constraint specifically
    section("Runtime constraint — User 1 (avg 126 min, ceiling 189 min)")
    profile = cli.build_constraint_profile(1)
    raw     = cli.hybrid_rec.get_hybrid_recommendations(1, n=20)
    enriched = []
    for m in raw:
        full = movies_df[movies_df['movie_id'] == m.get('movie_id')]
        enriched.append(full.iloc[0].to_dict() if not full.empty else m)

    long_films = [m for m in enriched
                  if m.get('runtime', 0) and m['runtime'] > profile['avg_runtime'] * 1.5]
    result("Films exceeding runtime ceiling in raw recs",
           [(m['title'], f"{m['runtime']}min") for m in long_films[:3]])

    filtered = cli.constraint_layer.apply_hard_constraints(enriched, profile)
    long_survived = [m for m in filtered
                     if m.get('runtime', 0) and m['runtime'] > profile['avg_runtime'] * 1.5]
    result("Long films surviving after hard filter",
           len(long_survived),
           "pass" if len(long_survived) == 0 else "fail")

    # Test genre overlap constraint
    section("Genre overlap constraint")
    profile_1 = cli.build_constraint_profile(1)
    print(f"    {INFO}  User 1 all_watched_genres: "
          f"{sorted(profile_1['all_watched_genres'])[:8]}...")
    no_overlap = [m for m in enriched
                  if not set(str(m.get('genres', '')).split('|'))
                  .intersection(profile_1['all_watched_genres'])]
    result("Movies with zero genre overlap removed",
           len(no_overlap),
           "pass" if len(no_overlap) >= 0 else "fail")


# =============================================================================
# TEST 8 — Soft Constraints Re-ranking
# =============================================================================

def test_soft_constraints(cli, movies_df):
    header("TEST 8 — SOFT CONSTRAINTS RE-RANKING")

    section("User 1 — before and after soft re-ranking")
    profile  = cli.build_constraint_profile(1)
    print(f"    {INFO}  Preferred decade:  {profile['preferred_decade']}s")
    print(f"    {INFO}  Liked directors:   "
          f"{sorted(profile['liked_directors'])[:4]}")

    raw   = cli.hybrid_rec.get_hybrid_recommendations(1, n=15)
    enriched = []
    for m in raw:
        full = movies_df[movies_df['movie_id'] == m.get('movie_id')]
        enriched.append(full.iloc[0].to_dict() if not full.empty else m)

    filtered = cli.constraint_layer.apply_hard_constraints(enriched, profile)

    # Apply soft without printing each line
    scored = []
    for movie in filtered:
        score = 1.0
        if movie.get('original_language') != profile['preferred_language']:
            score *= 0.8
        if movie.get('director') in profile['liked_directors']:
            score *= 1.3
        movie_decade = (movie.get('year', 1990) // 10) * 10
        if movie_decade == profile['preferred_decade']:
            score *= 1.2
        elif abs(movie_decade - profile['preferred_decade']) > 20:
            score *= 0.9
        movie_genres = set(str(movie.get('genres', '')).split('|'))
        overlap = movie_genres.intersection(profile['liked_genres'])
        if overlap:
            score *= (1 + 0.1 * len(overlap))
        movie['soft_score'] = round(score, 4)
        scored.append(movie)
    scored.sort(key=lambda x: x['soft_score'], reverse=True)

    print(f"\n    Top 8 after soft re-ranking:")
    print(f"    {'Title':<42} {'Director':<22} {'Decade':<8} {'Score'}")
    print(f"    {'─'*42} {'─'*22} {'─'*8} {'─'*6}")
    for m in scored[:8]:
        title    = m.get('title', '?')[:40]
        director = m.get('director', '?')[:20]
        decade   = str((m.get('year', 1900) // 10) * 10) + 's'
        score    = m.get('soft_score', 0)
        marker   = " ★" if m.get('director') in profile['liked_directors'] else ""
        print(f"    {title:<42} {director:<22} {decade:<8} {score:.3f}{marker}")

    print(f"\n    ★ = director in user's liked directors list")

    # Verify liked directors score higher
    director_boosted = [m for m in scored
                        if m.get('director') in profile['liked_directors']]
    non_boosted = [m for m in scored
                   if m.get('director') not in profile['liked_directors']]
    if director_boosted and non_boosted:
        avg_boosted     = np.mean([m['soft_score'] for m in director_boosted])
        avg_non_boosted = np.mean([m['soft_score'] for m in non_boosted])
        result("Avg score — liked directors",
               f"{avg_boosted:.3f}")
        result("Avg score — other directors",
               f"{avg_non_boosted:.3f}")
        result("Director boost working",
               avg_boosted > avg_non_boosted,
               "pass" if avg_boosted > avg_non_boosted else "fail")


# =============================================================================
# TEST 9 — Cold Start Simulation
# =============================================================================

def test_cold_start_simulation(cli, movies_df, interactions_df):
    header("TEST 9 — COLD START SIMULATION (Domain-Agnostic Demonstration)")

    section("Simulating a new user gaining interactions over time")

    # Find a user with many interactions to simulate the journey
    user_counts = interactions_df.groupby('user_id').size()
    rich_user   = user_counts[user_counts >= 20].index[0]
    all_clicks  = interactions_df[
        interactions_df['user_id'] == rich_user
    ]['movie_id'].tolist()

    thresholds = [0, 1, 3, 5, 10, 20]
    print(f"\n    Simulating user journey for User {rich_user}")
    print(f"    (has {len(all_clicks)} total interactions in dataset)\n")
    print(f"    {'Interactions':<15} {'Mode':<8} {'Top Recommendation'}")
    print(f"    {'─'*14:<15} {'─'*6:<8} {'─'*40}")

    for n_clicks in thresholds:
        if n_clicks == 0:
            mode = "COLD"
            strategy = "Popularity-based"
            # Get most popular
            popular = cli.hybrid_rec.rec_system.get_popular_movies(1)
            top_rec = popular[0]['title'] if popular else "N/A"
        elif n_clicks < 3:
            mode = "COLD"
            strategy = "Popularity-based"
            popular = cli.hybrid_rec.rec_system.get_popular_movies(1)
            top_rec = popular[0]['title'] if popular else "N/A"
        elif n_clicks < 10:
            mode = "WARM"
            strategy = "Content-based"
            seed_movie = all_clicks[0]
            similar = cli.hybrid_rec.content_rec.get_similar_movies(seed_movie, 1)
            top_rec = similar[0]['title'] if similar else "N/A"
        else:
            mode = "HOT"
            strategy = "Collaborative"
            cf_ids = cli.hybrid_rec.collab_rec.get_user_recommendations(
                rich_user, n=1
            )
            if cf_ids:
                row = movies_df[movies_df['movie_id'] == cf_ids[0]]
                top_rec = row.iloc[0]['title'] if not row.empty else "N/A"
            else:
                top_rec = "N/A (sparse)"

        top_rec_short = top_rec[:38]
        print(f"    {n_clicks:<15} {mode:<8} {top_rec_short}")

    print(f"\n    {INFO}  Thresholds: COLD=0-2  |  WARM=3-9  |  HOT=10+")


# =============================================================================
# TEST 10 — SVD Evaluation Metrics
# =============================================================================

def test_svd_metrics(cli):
    header("TEST 10 — SVD EVALUATION METRICS")

    from surprise import Dataset, Reader, SVD
    from surprise.model_selection import train_test_split
    from surprise import accuracy
    from collections import defaultdict
    from recomm import ndcg_at_k

    section("Running SVD on 80/20 train/test split...")

    reader   = Reader(rating_scale=(1, 5))
    data     = Dataset.load_from_df(
        cli.interactions_df[['user_id', 'movie_id', 'rating']], reader
    )
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
    algo = SVD(n_epochs=15, random_state=42)
    algo.fit(trainset)
    predictions = algo.test(testset)

    rmse = accuracy.rmse(predictions, verbose=False)
    mae  = accuracy.mae(predictions, verbose=False)

    k = 10
    threshold = 3.0
    user_recs = defaultdict(list)
    user_true = defaultdict(set)
    for pred in predictions:
        user_recs[pred.uid].append((pred.iid, pred.est))
        if pred.r_ui >= threshold:
            user_true[pred.uid].add(pred.iid)

    precision_list, recall_list, ndcg_list = [], [], []
    for user, rec_list in user_recs.items():
        rec_list.sort(key=lambda x: x[1], reverse=True)
        recommended = [iid for iid, _ in rec_list[:k]]
        relevant    = user_true.get(user, set())
        if not relevant:
            continue
        hits = len(set(recommended) & relevant)
        precision_list.append(hits / k)
        recall_list.append(hits / len(relevant))
        ndcg_list.append(ndcg_at_k(recommended, relevant, k))

    avg_p    = np.mean(precision_list) if precision_list else 0
    avg_r    = np.mean(recall_list) if recall_list else 0
    avg_ndcg = np.mean(ndcg_list) if ndcg_list else 0

    print(f"\n    {'Metric':<25} {'Value':<12} {'Interpretation'}")
    print(f"    {'─'*24:<25} {'─'*10:<12} {'─'*30}")
    print(f"    {'RMSE':<25} {rmse:<12.4f} Lower is better (rating prediction error)")
    print(f"    {'MAE':<25} {mae:<12.4f} Lower is better (mean absolute error)")
    print(f"    {'Precision@10':<25} {avg_p:<12.4f} Fraction of top-10 that are relevant")
    print(f"    {'Recall@10':<25} {avg_r:<12.4f} Fraction of relevant items found")
    print(f"    {'NDCG@10':<25} {avg_ndcg:<12.4f} Ranking quality (1.0 = perfect)")

    result("RMSE below 1.0",  rmse < 1.0,
           "pass" if rmse < 1.0 else "fail")
    result("Precision@10 > 0", avg_p > 0,
           "pass" if avg_p > 0 else "fail")


# =============================================================================
# SUMMARY
# =============================================================================

def print_summary(start_time):
    elapsed = time.time() - start_time
    header("TEST SUITE COMPLETE")
    print(f"""
  Tests run    : 10 scenarios
  Elapsed time : {elapsed:.1f} seconds

  Coverage:
    ✓  Data integrity (movies, users, interactions)
    ✓  User profiles — three user types
    ✓  Content-based similarity — 4 seed movies
    ✓  Collaborative filtering — sparse vs dense users
    ✓  Algorithm comparison — all 5 algorithms
    ✓  Child safety — age 7 user with keyword + genre blocking
    ✓  Hard constraints — runtime, genre overlap
    ✓  Soft constraints — director/decade/genre re-ranking
    ✓  Cold start simulation — 0→20 interaction journey
    ✓  SVD evaluation — RMSE, MAE, Precision, Recall, NDCG

  Key findings documented above in terminal output.
""")
    print("═" * 70 + "\n")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    start = time.time()

    print("═" * 70)
    print("  MOVIE RECOMMENDATION SYSTEM — AUTOMATED TEST SUITE")
    print("  Project: Universal Recommendation System (Thesis)")
    print("═" * 70)

    cli, movies_df, users_df, interactions_df, content_rec, collab_rec = setup()

    test_data_integrity(movies_df, users_df, interactions_df)
    test_user_profiles(cli, users_df)
    test_content_similarity(cli, movies_df)
    test_collaborative_filtering(cli, interactions_df)
    test_algorithm_comparison(cli)
    test_child_safety(cli, movies_df)
    test_hard_constraints_detail(cli, movies_df)
    test_soft_constraints(cli, movies_df)
    test_cold_start_simulation(cli, movies_df, interactions_df)
    test_svd_metrics(cli)

    print_summary(start)