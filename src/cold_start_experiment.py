'''

 The research question
For new users with little or no history, which method recommends best,
and at how many seed movies does personalisation start to beat popularity?
Does LLM enrichment help more for new users than for existing ones?


3. Methods to compare
#	Method	Uses	Why include it
0	Popularity	training ratings only	The standard cold-start baseline. Same list for everyone. Usually hard to beat with few seeds.
1	Content A / B	seeds + CountVectorizer/TF-IDF, without / with LLM labels	Your existing content tables
2	Embedding C / D	seeds + embeddinggemma, without / with LLM labels	Your best warm method
3	Item-based CF	seeds + movie-to-movie similarity from warm users’ ratings	The collaborative baseline that works for new users without retraining (SVD and NCF can’t use a new user)
4	Hybrid: embedding + popularity	blend of 2 and 0	Common in practice: personal taste plus a prior towards well-liked movies



'''
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize

from build_movie_texts import (load_movies, prepare_ratings, user_metrics, K,
                               content_similarity, embedding_similarity, evaluate_method,
                               popularity_ranking, wilcoxon_row)
from llm_enricher import PROMPT_VERSION

# ───────────── cold-start experiment ─────────────
COLD_FRACTION = 0.10
SEED_COUNTS   = [1, 3, 5, 10]
MIN_TEST_RELEVANT = 5          # a cold user needs enough left to evaluate
EMBED_NAME = "embeddinggemma"   # best embedding model in experiments 04-05
RESULTS = Path("output") / "experiments" / "cold_start"


def split_users(ratings):

    users = ratings["user_id"].unique()
    # Sort first so the starting order never depends on the CSV row order
    sorted_users = np.sort(users)
    rng = np.random.default_rng(42)  # the tool
    shuffled_users = rng.permutation(sorted_users)  # the result: users in random order

    # The first COLD_FRACTION of the shuffled users become cold users
    count = max(1, int(len(users) * COLD_FRACTION))

    cold_users = shuffled_users[:count]  # positions 0 up to count (not including count)
    warm_users = shuffled_users[count:]  # positions count to the end

    train = ratings[ ratings["user_id"].isin(warm_users)    ]  # all their ratings
    cold  =  ratings[ ratings["user_id"].isin(cold_users)    ]   # never used for training

    # train = rows whose user_id is in the warm list (all their ratings)
    # cold = rows whose user_id is in the cold list (all their ratings)

    return train, cold


def build_cold_user_data(cold, pool_movie_ids, max_seeds, min_relevant):
    # cold           : ratings of the cold users (from split_users)
    # pool_movie_ids : set of movies rated by warm users (the candidate pool)
    # max_seeds      : max(SEED_COUNTS) = 10
    # min_relevant   : MIN_TEST_RELEVANT = 5

    users = []
    too_few_liked = 0
    too_few_relevant = 0
    for user_id, user_rows in cold.groupby ("user_id"):
        # 1. Oldest rating first: the order in which the user "joined"
        user_rows = user_rows.sort_values(by=["timestamp", "movie_id"])
        # 2. The user's own "liked" threshold (ground truth only, never model input)
        mean =  user_rows.rating.mean()

        # 3. Liked movies, in time order, limited to movies the system can use
        # Returns a filtered DataFrame
        liked_in_order = user_rows[
            (user_rows["rating"] >= mean) &
            (user_rows["movie_id"].isin(pool_movie_ids))
            ]

        # 4. Seed pool: the first max_seeds liked movies
        if len(liked_in_order) < max_seeds:
            too_few_liked += 1
            continue
        seed_pool = liked_in_order["movie_id"].iloc[:max_seeds].tolist()

        # 5. Answer key: liked movies that are NOT seeds and can be recommended
        # 5.1. Filter rows using proper Pandas boolean operators (&) and .isin()
        filtered_rows = user_rows[
            (user_rows["rating"] >= mean) &
            (user_rows["movie_id"].isin(pool_movie_ids)) &
            (~user_rows["movie_id"].isin(seed_pool))
            ]

        # 5.2. Extract the 'movie_id' column and convert to a set
        relevant = set(filtered_rows["movie_id"])

        # 6. Enough left to evaluate?
        if len(relevant) < min_relevant:
            too_few_relevant += 1
            continue

        row = {
            "user_id": user_id,
            "seed_pool" : seed_pool,
            "relevant" : relevant,
        }
        users.append(row)

    kept_ids = {user["user_id"] for user in users}
    avg_relevant = sum(len(user["relevant"]) for user in users) / len(users)
    avg_ratings = cold[cold["user_id"].isin(kept_ids)].groupby("user_id").size().mean()

    print(f"Cold users kept: {len(users):,} | skipped: {too_few_liked} too few liked, "
          f"{too_few_relevant} too few relevant")
    print(f"Average relevant movies per user: {avg_relevant:.1f}")
    print(f"Average ratings per kept user: {avg_ratings:.1f}")
    return users

def users_with_n_seeds(users, n):
    result = []
    for user in users:
        seeds = user["seed_pool"][:n]
        result.append({
            "user_id":  user["user_id"],
            "liked":    seeds,              # what recommend() scores against
            "seen":     set(user["seed_pool"]),   # all 10 seeds for every n, never recommended
            "relevant": user["relevant"],   # the same answer key for every n
        })
    return result

def popularity_scores(train):
    # Number of warm users who liked each movie (rating >= their own mean), most liked first.
    # Uses the shared definition from build_movie_texts so warm and cold start agree exactly
    # (including the tie-break by movie_id).
    return popularity_ranking(train)

def recommend_popular(popular_movie_ids, user, k):
    # take popular_movie_ids in order

    # skip movies in user["seen"]

    movie_list = []
    for movie_id in popular_movie_ids:
        if movie_id not in user["seen"]:
            movie_list.append(movie_id)

    return movie_list[:k]


'''
Part 1: item-based CF

Idea: two movies are similar if the same warm users rated them in a similar way. 
Unlike content and embeddings, this uses people’s behaviour, not the movie text. 
It works for a new user without retraining, because the similarity table is computed from warm users only,
and the new user’s seeds are simply looked up in it.
'''

def item_cf_similarity(train, movies, position_of):
    # 1. Centre each rating on its user's own mean (generous vs harsh raters)
    user_mean = train.groupby("user_id")["rating"].transform("mean")
    centred   = train.rating - user_mean

    # 2. Matrix: one row per movie (in the SAME order as the other tables),
    #    one column per warm user
    row_indices = train["movie_id"].map(position_of).values
    # one column per warm user: turn user ids into positions 0..n_users-1
    column_indices, warm_user_ids = pd.factorize(train["user_id"])
    n_users = len(warm_user_ids)

    M = csr_matrix(
        (centred.values, (row_indices, column_indices)),
        shape=(len(movies), n_users),
    )

    # 3. Cosine similarity between movie rows
    M = normalize(M)  # rows with no ratings stay all-zero
    similarity = M @ M.T                       # movies x movies
    # In Python, @ is matrix multiplication

    similarity = similarity.toarray().astype(np.float32)
    np.fill_diagonal(similarity, 0)   # a movie is not its own neighbour
    return similarity

'''
The idea in one line
hybrid score = alpha × personal similarity + (1 − alpha) × popularity, both rescaled to 0..1

alpha = 1 → pure similarity (table D or E)
alpha = 0 → pure popularity
in between → “similar to your seeds and well liked by many people”

1. Personal similarity: “how much does this movie resemble the user’s seeds?”

What it is: a number per candidate saying how close it is to the movies the user told you they like.

Where it comes from: one of your similarity tables, D (embeddings + LLM labels) or E (item-CF), exactly as in recommend():

Take the user’s seed movies (user["liked"], the first N).
Look up the rows of the table for those seeds, and the columns for the candidates (table[np.ix_(liked, candidates)]). 
This gives an N × candidates block: the similarity of every seed to every candidate.

Mean scoring: average each column, giving one number per candidate, its average similarity to all seeds.


 Two hybrids, same formula, different personal part:

   hybrid score = alpha * personal similarity + (1 - alpha) * popularity
                 (both rescaled to 0..1 over the user's candidates)

# ┌──────────────────────┬────────────────────────────────────┬────────────────────────────────┐
# │                      │ Hybrid D                           │ Hybrid E                       │
# ├──────────────────────┼────────────────────────────────────┼────────────────────────────────┤
# │ Personal part from   │ Table D: embeddings + LLM labels   │ Table E: item-based CF         │
# │ "Similar" means      │ about similar things               │ liked by the same people       │
# │                      │ (story, themes, mood)              │ (warm users' ratings)          │
# │ Popularity part      │ the same (log count of warm users' │ the same                       │
# │                      │ positive ratings)                  │                                │
# │ alpha                │ chosen separately on validation    │ chosen separately on validation│
# │                      │ users, per number of seeds N       │ users, per number of seeds N   │
# └──────────────────────┴────────────────────────────────────┴────────────────────────────────┘

 Hybrid D asks: does the LLM-enriched content signal become useful with a popularity prior?
 Hybrid E asks: can collaborative signal + popularity beat popularity with fewer seeds?

2. Popularity: “how many people liked this movie?”
What it is: a number per candidate saying how widely liked it is, the same for every user.

Where it comes from: popularity_scores(train), the number of warm users who rated the movie at or above their own mean, 
then log(1 + count) via popularity_vector, so blockbusters don’t dwarf everything.

It answers: is this movie good in general? It’s the quality signal that pure similarity lacks.

'''

def popularity_vector(popular, movies, position_of):
    # popular: the Series from popularity_scores (index = movie_id, value = count)
    vector = np.zeros(len(movies), dtype=np.float32)
    for movie_id, count in popular.items():
        vector[position_of[movie_id]] = np.log1p( count) #  ln(count + 1)
    return vector

def rescale(values):
    low  = values.min()
    high = values.max()
    if high == low: return np.zeros_like(values)     # avoid dividing by 0
    return (values - low) / (high - low)            # now 0..1

def recommend_hybrid(table, pop_vector, user, pool_positions,
                          position_of, movie_ids, k, alpha, scoring):
    # same first steps as recommend() in build_movie_texts.py

    seen = {position_of[movie_id] for movie_id in user["seen"]}
    candidates = np.array([p for p in pool_positions if p not in seen])
    liked = [position_of[movie_id] for movie_id in user["liked"]]
    block = table[np.ix_(liked, candidates)]   # rows: seeds, columns: candidates
    if scoring == "max":
        personal = block.max(axis=0)
    else:
        personal = block.mean(axis=0)

    personal = rescale(personal)  # per user, over THEIR candidates
    prior = rescale(pop_vector[candidates])
    score = alpha * personal + (1 - alpha) * prior
    top = np.argsort(-score, kind="stable")[:k]

    return movie_ids[candidates[top]].tolist()

def evaluate_hybrid(name, table, pop_vector, users, pool_positions,
                         position_of, movie_ids, k, alpha, scoring):
    rows = []
    for user in users:
        recommended = recommend_hybrid(table, pop_vector, user, pool_positions,
                                       position_of, movie_ids, k, alpha, scoring)
        rows.append({"method": name, "scoring": scoring, "user_id": user["user_id"],
                     **user_metrics(recommended, user["relevant"], k)})
    return pd.DataFrame(rows)



VAL_FRACTION = 0.20      # constant at the top, next to COLD_FRACTION

def split_validation(cold_users, val_fraction):
    rng   = np.random.default_rng(42)              # same seed as everywhere
    order = rng.permutation(len(cold_users))       # shuffled POSITIONS: e.g. [512, 3, 877, ...]

    count = int(len(cold_users) * val_fraction)    # about 202 of 1,013

    val_users  = [cold_users[i] for i in order[:count]]    # first 20% of the shuffle
    test_users = [cold_users[i] for i in order[count:]]    # the remaining 80%

    print(f"Validation users: {len(val_users)} | test users: {len(test_users)}")
    return val_users, test_users

# Finer steps near 0, where the best alpha was found on the coarse grid
ALPHAS = [0.0, 0.02, 0.05, 0.08, 0.1, 0.12, 0.15, 0.2, 0.25, 0.3, 0.4,
          0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

def choose_alphas(table_name, table, pop_vector, val_users,
                       pool_positions, position_of, movie_ids, k):

    best_alpha = {}         # n → chosen alpha
    curve      = []         # every (n, alpha, ndcg) tried, for the thesis

    for n in SEED_COUNTS:
        group = users_with_n_seeds(val_users, n)

        best_ndcg = -1
        for alpha in ALPHAS:                                   # tried from small to large
            scores = evaluate_hybrid(table_name, table, pop_vector, group,
                                     pool_positions, position_of, movie_ids,
                                     k, alpha, "mean")
            ndcg = scores["ndcg"].mean()
            curve.append({"table": table_name, "n": n, "alpha": alpha, "ndcg": ndcg})

            if ndcg > best_ndcg:              # strict ">" keeps the SMALLER alpha on a tie
                best_ndcg = ndcg
                best_alpha[n] = alpha

        print(f"{table_name} | n={n}: best alpha {best_alpha[n]} (val NDCG {best_ndcg:.4f})")

    return best_alpha, pd.DataFrame(curve)


# (a, b): the test asks whether b differs from a; mean_difference = b - a.
TEST_PAIRS = [
    (("popularity", "-"), ("hybrid_E_item_cf", "mean")),
    (("popularity", "-"), ("hybrid_D_embedding_enriched", "mean")),
    (("popularity", "-"), ("hybrid_C_embedding_base", "mean")),
    (("popularity", "-"), ("E_item_cf", "mean")),
    # Enrichment: same hybrid, same alpha, with vs without LLM labels
    (("hybrid_C_embedding_base_at_D_alpha", "mean"), ("hybrid_D_embedding_enriched", "mean")),
    (("C_embedding_base", "mean"), ("D_embedding_enriched", "mean")),
    (("A_content_base", "max"), ("B_content_enriched", "max")),
    # Collaborative vs content signal inside the hybrid
    (("hybrid_D_embedding_enriched", "mean"), ("hybrid_E_item_cf", "mean")),
]


def significance_tests(results):
    rows = []
    for n, part in results.groupby("n"):
        by_user = part.pivot_table(index="user_id", columns=["method", "scoring"], values="ndcg")
        for a, b in TEST_PAIRS:
            name = f"{b[0]} ({b[1]}) vs {a[0]} ({a[1]})"
            rows.append({"n": n, **wilcoxon_row(name, by_user[a], by_user[b])})
    tests = pd.DataFrame(rows)
    # Bonferroni correction over all tests in this table
    tests["p_bonferroni"] = (tests["p_value"] * len(tests)).clip(upper=1.0)
    return tests


if __name__ == "__main__":
    # Data: the same 1,957 movies and ratings as experiments 04 and 05
    movies = load_movies()
    ratings = prepare_ratings(movies)

    # Split by user: warm users train, cold users are simulated new users
    train, cold = split_users(ratings)
    pool_movie_ids = set(train["movie_id"])

    cold_users = build_cold_user_data(cold, pool_movie_ids,
                                      max(SEED_COUNTS), MIN_TEST_RELEVANT)

    # Positions used by recommend(): row i of every table = i-th movie in `movies`
    movie_ids = movies["movie_id"].to_numpy()
    position_of = {movie_id: i for i, movie_id in enumerate(movie_ids)}
    pool_positions = np.array(sorted(position_of[m] for m in pool_movie_ids))

    # Similarity tables, built once (same as run_evaluation, plus item-based CF)
    tables = {
        "A_content_base": content_similarity(movies, with_enrichment=False),
        "B_content_enriched": content_similarity(movies, with_enrichment=True),
        "C_embedding_base": embedding_similarity(f"movies_base_{EMBED_NAME}_{PROMPT_VERSION}", movies),
        "D_embedding_enriched": embedding_similarity(f"movies_enriched_{EMBED_NAME}_{PROMPT_VERSION}", movies),
        "E_item_cf": item_cf_similarity(train, movies, position_of),
    }

    popular = popularity_scores(train)
    pop_vector = popularity_vector(popular, movies, position_of)

    # ── Choose alpha on validation users only ──
    val_users, test_users = split_validation(cold_users, VAL_FRACTION)

    # C is the control for D: same hybrid without LLM labels,
    # so any D-over-C gain can be attributed to the enrichment
    HYBRID_TABLES = ["C_embedding_base", "D_embedding_enriched", "E_item_cf"]
    alphas = {}
    curves = []
    for table_name in HYBRID_TABLES:
        best, curve = choose_alphas(table_name, tables[table_name], pop_vector, val_users,
                                    pool_positions, position_of, movie_ids, K)
        alphas[table_name] = best
        curves.append(curve)

    RESULTS.mkdir(parents=True, exist_ok=True)
    curves = pd.concat(curves, ignore_index=True)
    curves.to_csv(RESULTS / "alpha_validation.csv", index=False)
    print("\nValidation NDCG@10 per alpha:")
    print(curves.pivot_table(index=["table", "alpha"], columns="n", values="ndcg").round(4))

    pd.DataFrame([{"table": t, "n": n, "alpha": a}
                  for t, best in alphas.items() for n, a in best.items()]
                 ).to_csv(RESULTS / "alphas.csv", index=False)

    # ── Final evaluation: ALL methods on the test users only ──
    results = []
    for n in SEED_COUNTS:
        group = users_with_n_seeds(test_users, n)

        # Method 0: popularity (same list for everyone)
        rows = []
        for user in group:
            recommended = recommend_popular(popular.index, user, K)
            rows.append({"method": "popularity", "scoring": "-", "user_id": user["user_id"],
                         **user_metrics(recommended, user["relevant"], K)})
        results.append(pd.DataFrame(rows).assign(n=n))

        # Methods 1-3: content, embedding and item-CF tables, with both scorings
        for name, table in tables.items():
            for scoring in ["max", "mean"]:
                scores = evaluate_method(name, table, group, pool_positions,
                                         position_of, movie_ids, K, scoring)
                results.append(scores.assign(n=n))

        # Method 4: hybrids, with the alpha chosen on validation for this n
        for table_name in HYBRID_TABLES:
            alpha = alphas[table_name][n]
            scores = evaluate_hybrid(f"hybrid_{table_name}", tables[table_name], pop_vector,
                                     group, pool_positions, position_of, movie_ids,
                                     K, alpha, "mean")
            results.append(scores.assign(n=n, alpha=alpha))

        # Control for the enrichment question: hybrid C at the SAME alpha as hybrid D,
        # so a D-vs-C difference cannot come from the two hybrids choosing different alphas.
        alpha = alphas["D_embedding_enriched"][n]
        scores = evaluate_hybrid("hybrid_C_embedding_base_at_D_alpha", tables["C_embedding_base"],
                                 pop_vector, group, pool_positions, position_of, movie_ids,
                                 K, alpha, "mean")
        results.append(scores.assign(n=n, alpha=alpha))

    results = pd.concat(results, ignore_index=True)
    results.to_csv(RESULTS / "per_user.csv", index=False)

    summary = (results.groupby(["method", "scoring", "n"])[["ndcg", "precision", "recall"]]
               .mean().round(4).reset_index())
    summary.to_csv(RESULTS / "summary.csv", index=False)

    print(f"\nTest users: {len(test_users)} | NDCG@10:")
    print(summary.pivot_table(index=["method", "scoring"], columns="n", values="ndcg"))

    tests = significance_tests(results)
    tests.to_csv(RESULTS / "significance_tests.csv", index=False)
    print("\nWilcoxon signed-rank tests (NDCG@10 per user, b - a):")
    print(tests[["n", "comparison", "mean_difference", "p_value", "p_bonferroni",
                 "users_better", "users_worse"]].round(4).to_string(index=False))