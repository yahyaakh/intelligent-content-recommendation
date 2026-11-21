'''
Learning-to-rank on Hybrid E's top 30 candidates (N = 10 seeds).
Question: do LLM-derived signals add information beyond collaborative filtering and
popularity, when all signals are combined by a learned ranker (LightGBM lambdarank)?
'''


import json
import sys
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import wilcoxon
from tqdm import tqdm

# Read the mode before the imports below parse sys.argv:
#   (none)      features, ablation and test (doc 15)
#   importance  feature importance of the final R4 models (doc 15, section 5)
#   simulated   LTR v2 step 1: train on simulated new users made from the warm users
#   g5          LTR v2 step 2: SVD, co-visitation and recent-seed features
#   candidates  LTR v2 step 3: candidate lists C30, C50, C30+20
#   tuning      LTR v2 step 4: grid search on simulated users, XGBoost + LightGBM ensemble
#   seeds3      LTR v2 step 5: the same pipeline with 3 seeds (needs step 4's chosen parameters)
#   final       LTR v2 step 6: the pre-registered test on the 814 test users (runs once)
MODE = sys.argv[1] if len(sys.argv) > 1 else "full"
IMPORTANCE_ONLY = MODE == "importance"
sys.argv = [sys.argv[0]]
import build_movie_texts as bmt           # load_movies, content_similarity, embedding_similarity, ndcg_at_k
import cold_start_experiment as cs        # users_with_n_seeds, popularity_scores, split_users
import llm_rerank_experiment as lr        # load_data, hybrid_e_candidates, build_prompt, parse_ranking, run_name


FOLDER = Path("output") / "experiments_ LightGBM_XBOOST" / "ltr_boosting"
DEEPSEEK_V5 = Path("output") / "experiments" / "deepseek_v4.1_flash_v5"
FOLDER.mkdir(parents=True, exist_ok=True)

N_SEEDS = 10
TOP_M   = lr.TOP_M                       # 30 candidates per user

# Where the LLM rankings come from (cached answers of idea 1, prompt p2, N = 10, shuffle 0)
LLM_RANKERS = {
    "rank_deepseek": ("ollama_cloud", "deepseek-v4.1-flash"),
    "rank_gemma":    ("ollama", "gemma4:e4b"),
}

# Feature groups for the ablation (filled in by the names the feature function produces)
GROUPS = {
    "G1_collaborative": ["cf_mean", "cf_max", "hybrid_score", "hybrid_rank", "log_popularity",
                         "vote_count", "vote_average", "year"],
    "G2_content":       ["emb_mean", "emb_max", "content_max"],
    "G3_llm_labels":    ["emb_v5_mean", "emb_v5_max", "theme_overlap", "mood_overlap",
                         "genre_overlap", "director_match", "actor_overlap", "decade_gap"],
    "G4_llm_rankings":  ["rank_deepseek", "rank_gemma"],
    # LTR v2 step 2 (added as extra columns by add_g5, not by candidate_rows)
    "G5_latent":        ["svd_mean", "svd_max", "svd_score", "covis_mean", "covis_max", "cf_recent"],
}
MODELS = {
    "R1": ["G1_collaborative"],
    "R2": ["G1_collaborative", "G2_content"],
    "R3": ["G1_collaborative", "G2_content", "G3_llm_labels"],
    "R4": ["G1_collaborative", "G2_content", "G3_llm_labels", "G4_llm_rankings"],
    "R5": ["G1_collaborative", "G4_llm_rankings"],
    "R6": ["G1_collaborative", "G2_content", "G5_latent"],
    "R7": ["G1_collaborative", "G2_content", "G3_llm_labels", "G5_latent"],
}


import xgboost as xgb

PARAMS = dict(objective="lambdarank", n_estimators=200, learning_rate=0.05,
              num_leaves=15, min_child_samples=20, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.8, random_state=42, verbose=-1,
              deterministic=True, force_row_wise=True, n_jobs=1)   # identical numbers on every rerun

XGB_PARAMS = dict(objective="rank:ndcg", n_estimators=200, learning_rate=0.05,
                  max_depth=4,               # at most 16 leaves, like num_leaves=15
                  min_child_weight=1, subsample=0.8, colsample_bytree=0.8,
                  tree_method="hist", random_state=42, n_jobs=1)




def load_input() :
    data = lr.load_data()  # movies, users, split, pool, item_cf, pop_vector, alpha_E, movie_by_id

    # Similarity tables, rows and columns in the same movie order as data["movie_ids"]
    data["emb_base"] = bmt.embedding_similarity("movies_base_embeddinggemma_v2", data["movies"])
    data["emb_v5"] = bmt.embedding_similarity("movies_enriched_index", data["movies"],
                                              folder=DEEPSEEK_V5)
    data["content"] = bmt.content_similarity(data["movies"], with_enrichment=False)

    labels = pd.read_csv(DEEPSEEK_V5 / "movies_enriched.csv")
    data["v5_themes"] = {int(m): set(lr.split_field(t)) for m, t in zip(labels["movie_id"], labels["llm_themes"])}
    data["v5_moods"] = {int(m): set(lr.split_field(t)) for m, t in zip(labels["movie_id"], labels["llm_mood"])}

    # Cached LLM answers for every ranker (validation + test users are in the same cache file)
    data["llm_answers"] = {}
    for name, (provider, model) in LLM_RANKERS.items():
        cache_path = lr.run_name(provider, model, N_SEEDS, None, "p2", "hybrid")[0]
        answers = {}
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            answers[record["user_id"]] = record
        data["llm_answers"][name] = answers
    return data



def candidate_rows(user, data, with_llm_ranks=True, candidates=None):
    """The 30 Hybrid E candidates of one user, each as one row of features + label.
    with_llm_ranks=False leaves out G4 (the simulated warm users have no cached LLM answers).
    candidates: optional (ids, Hybrid E scores, Hybrid E ranks) from wide_candidates (step 3)."""
    if candidates is None:
        top_ids, top_scores = lr.hybrid_e_candidates(user, data)  # same 30 films as idea 1
        hybrid_ranks = list(range(1, len(top_ids) + 1))
    else:
        top_ids, top_scores, hybrid_ranks = candidates

    position_of = data["position_of"]
    seeds = [position_of[m] for m in user["liked"]]  # the 10 seed films (rows)
    cands = [position_of[m] for m in top_ids]  # the 30 candidates (columns)

    ''' Similarity of every candidate to the seeds: a 10 x 30 block per table '''

    cf_block = data["item_cf"][np.ix_(seeds, cands)]  # 10 x 30: seed rows, candidate columns
    emb_block = data["emb_base"][np.ix_(seeds, cands)]
    emb5_block = data["emb_v5"][np.ix_(seeds, cands)]
    cnt_block = data["content"][np.ix_(seeds, cands)]

    '''  The user's seed profile (only from the seeds, never from the answer key)  '''


    # Look up the themes of each seed. For each of the 10 seed films, get its set of themes from data["v5_themes"].
    # That gives 10 small sets, about 4–5 themes each, so roughly 45 themes in total, many of them repeated.
    # Count how often each theme appears. For example, “organized crime” in 4 of the 10 seeds, “betrayal” in 3, “mafia” in 2, and “space travel” in 1.
    # Keep the most frequent ones: the themes that recur across several seeds are the user’s real interests. A theme in only one seed may be a coincidence.

    profile = lr.profile_from_seeds(user, data)  # genres, directors, actors, decades, themes, moods

    # sorted(): a set of strings has a different order in every Python run (hash randomisation),
    # and keep_repeated breaks ties by insertion order, so without sorting the profile changes per run
    theme_counts = Counter()
    for movie_id in user["liked"]:
        for theme in sorted(data["v5_themes"][movie_id]):
            theme_counts[theme] += 1
    seed_themes = set(lr.keep_repeated(theme_counts, 5))

    mood_counts = Counter()
    for movie_id in user["liked"]:
        for mood in sorted(data["v5_moods"][movie_id]):
            mood_counts[mood] += 1
    seed_moods = set(lr.keep_repeated(mood_counts, 3))      # 3: films have at most 3 moods

    # The LLM rankings: position (1 = best) of every candidate in each model's ranking
    llm_rank = {}
    if with_llm_ranks:
        _, number_to_id = lr.build_prompt(user, top_ids, data, "p2")  # same shuffle as idea 1
        for name in LLM_RANKERS:
            content = data["llm_answers"][name][user["user_id"]]["content"]
            order, _ = lr.parse_ranking(content, number_to_id, top_ids)
            llm_rank[name] = {movie_id: position + 1 for position, movie_id in enumerate(order)}

    # The user's favourite genres, directors, actors and decades (from the seeds only)
    profile_genres = set(profile["genres"])
    profile_directors = set(profile["directors"])
    profile_actors = set(profile["actors"])
    seed_decades = profile["decades"]  # e.g. [1970, 1990]

    rows = []
    for j, movie_id in enumerate(top_ids):
        movie = data["movie_by_id"].loc[movie_id]
        movie_genres = set(lr.split_field(movie["genres"]))
        movie_cast = set(lr.split_field(movie["cast"]))
        movie_decade = (int(movie["year"]) // 10) * 10

        row = {
            "user_id": user["user_id"],
            "movie_id": movie_id,
            "label": 1 if movie_id in user["relevant"] else 0,  # the target, never a feature
            # G1 collaborative + popularity
            "cf_mean": cf_block[:, j].mean(),
            "cf_max": cf_block[:, j].max(),
            "hybrid_score": top_scores[j],
            "hybrid_rank": hybrid_ranks[j],
            "log_popularity": data["pop_vector"][cands[j]],
            "vote_count": movie["vote_count"],
            "vote_average": movie["vote_average"],
            "year": movie["year"],
            # G2 content
            "emb_mean": emb_block[:, j].mean(),
            "emb_max": emb_block[:, j].max(),
            "content_max": cnt_block[:, j].max(),
            # G3 LLM labels / profile
            "emb_v5_mean": emb5_block[:, j].mean(),
            "emb_v5_max": emb5_block[:, j].max(),
            "theme_overlap": len(seed_themes & data["v5_themes"][movie_id]) / len(seed_themes) if seed_themes else 0.0,
            "mood_overlap": len(seed_moods & data["v5_moods"][movie_id]) / len(seed_moods) if seed_moods else 0.0,
            "genre_overlap": len(profile_genres & movie_genres) / len(profile_genres) if profile_genres else 0.0,
            "director_match": int(movie["director"] in profile_directors),
            "actor_overlap": len(profile_actors & movie_cast),
            "decade_gap": min(abs(movie_decade - d) for d in seed_decades),
        }
        # G4 LLM rankings
        for name in llm_rank:
            row[name] = llm_rank[name][movie_id]
        rows.append(row)
    return rows


def ndcg_by_score(table, column, relevant_of, ascending):
    """Mean NDCG@10 when each user's 30 rows are sorted by `column`.
    ascending=True for ranks (1 = best), False for scores (higher = better)."""
    values = []
    for user_id, group in table.groupby("user_id"):
        top10 = group.sort_values(column, ascending=ascending, kind="stable")["movie_id"].tolist()[:bmt.K]
        values.append(bmt.ndcg_at_k(top10, relevant_of[user_id], bmt.K))
    return float(np.mean(values))


def build_table(data, split):
    """One DataFrame with 30 rows per user (all users of the split), saved to FOLDER, with checks."""
    users = cs.users_with_n_seeds(data[f"{split}_users"], N_SEEDS)     # 203 (val) or 814 (test)
    rows = []
    for user in tqdm(users, desc=f"features ({split})"):
        rows.extend(candidate_rows(user, data))          # extend: add the 30 rows of this user
    table = pd.DataFrame(rows)

    # Checks
    assert len(table) == len(users) * TOP_M, "not 30 rows per user"
    assert table.isna().sum().sum() == 0, f"missing values:\n{table.isna().sum()[table.isna().sum() > 0]}"
    feature_columns = [c for name, group in GROUPS.items() if name != "G5_latent" for c in group]  # G5: add_g5
    missing = [c for c in feature_columns if c not in table.columns]
    assert not missing, f"feature columns not in the table: {missing}"

    # Check 1: sorting by hybrid_rank must give Hybrid E's NDCG (val 0.6286, test 0.6311 at N = 10)
    # Check 2: sorting by rank_deepseek must give DeepSeek's idea-1 NDCG (val 0.6212, test 0.6169)
    relevant_of = {user["user_id"]: user["relevant"] for user in users}
    print(f"{split}: {len(users)} users, {len(table)} rows, {table['label'].mean():.1%} relevant")
    print(f"   Hybrid E (sorted by hybrid_rank):      {ndcg_by_score(table, 'hybrid_rank', relevant_of, True):.4f}")
    print(f"   DeepSeek (sorted by rank_deepseek):    {ndcg_by_score(table, 'rank_deepseek', relevant_of, True):.4f}")

    table.to_csv(FOLDER / f"features_{split}.csv", index=False)
    return table


def ndcg_per_user(table, column, relevant_of, ascending):
    """NDCG@10 per user when each user's 30 rows are sorted by `column` (a Series by user_id)."""
    values = {}
    for user_id, group in table.groupby("user_id", sort=False):
        top10 = group.sort_values(column, ascending=ascending, kind="stable")["movie_id"].tolist()[:bmt.K]
        values[user_id] = bmt.ndcg_at_k(top10, relevant_of[user_id], bmt.K)
    return pd.Series(values)


def cross_validate(table, relevant_of, columns, folds=5 , library="lightgbm"):
    """Every user is scored by a model that never saw them: 5 folds split by USER, not by row."""
    user_ids = np.sort(table["user_id"].unique())
    shuffled = np.random.default_rng(42).permutation(user_ids)
    fold_of = {user_id: i % folds for i, user_id in enumerate(shuffled)}
    fold = table["user_id"].map(fold_of)

    scores = pd.Series(np.nan, index=table.index)
    for k in range(folds):
        train, held_out = table[fold != k], table[fold == k]
        model = fit_ranker(train, columns, library)
        scores[held_out.index] = model.predict(held_out[columns])

    scored = table.assign(score=scores)
    return ndcg_per_user(scored, "score", relevant_of, ascending=False)

def run_ablation(table, relevant_of, library="lightgbm"):
    """Cross-validated NDCG@10 per user for R1-R5, plus Hybrid E and DeepSeek alone, and paired tests."""
    results = {"Hybrid E": ndcg_per_user(table, "hybrid_rank", relevant_of, ascending=True),
               "DeepSeek alone": ndcg_per_user(table, "rank_deepseek", relevant_of, ascending=True)}
    for name, groups in MODELS.items():
        columns = [c for group in groups for c in GROUPS[group]]
        results[name] = cross_validate(table, relevant_of, columns, library=library)
        print(f"{name} ({len(columns)} features): {results[name].mean():.4f}")
    per_user = pd.DataFrame(results)

    tests = []
    for a, b in [("R1", "R2"), ("R1", "R3"), ("R1", "R4"), ("R1", "R5"), ("R3", "R4"),
                 ("Hybrid E", "R1"), ("Hybrid E", "R4")]:
        diff = per_user[b] - per_user[a]
        tests.append({"comparison": f"{b} vs {a}", "difference": diff.mean(),
                      "better": int((diff > 0).sum()), "worse": int((diff < 0).sum()),
                      "p_value": wilcoxon(diff).pvalue if diff.any() else 1.0})
    tests = pd.DataFrame(tests)
    tests["p_bonferroni"] = (tests["p_value"] * len(tests)).clip(upper=1.0)
    per_user.to_csv(FOLDER / f"cv_per_user_val_{library}.csv")
    tests.to_csv(FOLDER / f"cv_tests_val_{library}.csv", index=False)
    print(per_user.mean().round(4).to_string())
    print(tests.round(4).to_string(index=False))
    return per_user, tests



def train_on(table, columns, library="lightgbm"):
    return fit_ranker(table, columns, library)

def final_test(val, test, relevant_of_test, library="lightgbm"):
    """PRE-REGISTERED (fixed 2026-10-25 before running, LightGBM): train R1-R5 on all 203 validation
    users, score the 814 test users once. Tests: R4 vs R1 and R4 vs Hybrid E (Bonferroni over 2).
    XGBoost: the same procedure as a robustness check."""
    results = {"Hybrid E": ndcg_per_user(test, "hybrid_rank", relevant_of_test, ascending=True),
               "DeepSeek alone": ndcg_per_user(test, "rank_deepseek", relevant_of_test, ascending=True)}
    models = {}
    for name, groups in MODELS.items():
        columns = [c for group in groups for c in GROUPS[group]]
        models[name] = train_on(val, columns, library)
        scored = test.assign(score=models[name].predict(test[columns]))
        results[name] = ndcg_per_user(scored, "score", relevant_of_test, ascending=False)
    per_user = pd.DataFrame(results)

    tests = []
    for a, b in [("R1", "R4"), ("Hybrid E", "R4")]:
        diff = per_user[b] - per_user[a]
        p = wilcoxon(diff).pvalue
        tests.append({"comparison": f"{b} vs {a}", "difference": diff.mean(), "better": int((diff > 0).sum()),
                      "worse": int((diff < 0).sum()), "p_value": p, "p_bonferroni": min(1.0, 2 * p)})
    per_user.to_csv(FOLDER / f"test_per_user_{library}.csv")
    pd.DataFrame(tests).to_csv(FOLDER / f"test_tests_{library}.csv", index=False)
    print(per_user.mean().round(4).to_string())
    print(pd.DataFrame(tests).round(4).to_string(index=False))
    return models, per_user


def fit_ranker(train, columns, library="lightgbm", params=None):
    """Train a ranking model on whole users (all 30 rows of a user stay together).
    params: settings that replace the defaults in PARAMS / XGB_PARAMS (step 4)."""
    if library == "lightgbm":
        model = lgb.LGBMRanker(**{**PARAMS, **(params or {})})
        model.fit(train[columns], train["label"],
                  group=train.groupby("user_id", sort=False).size().to_numpy())
    else:
        # XGBoost needs the query ids (users) in increasing order; a stable sort keeps each
        # user's 30 rows together and in their original order
        train = train.sort_values("user_id", kind="stable")
        model = xgb.XGBRanker(**{**XGB_PARAMS, **(params or {})})
        model.fit(train[columns], train["label"], qid=train["user_id"])
    return model


def shap_values(model, X, library):
    """One contribution per row and feature (TreeSHAP, built into both libraries); bias column dropped."""
    if library == "lightgbm":
        contrib = model.predict(X, pred_contrib=True)
    else:
        contrib = model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)
    return pd.DataFrame(contrib[:, :-1], columns=X.columns, index=X.index)


def shuffled_within_users(table, columns, rng):
    """A copy of `table` where `columns` are shuffled together inside each user's 30 rows.
    The link between those features and the label is broken; everything else stays."""
    n_users = len(table) // TOP_M
    rows = rng.permuted(np.arange(len(table)).reshape(n_users, TOP_M), axis=1).ravel()
    out = table.copy()
    out[columns] = table[columns].to_numpy()[rows]
    return out


def feature_importance(model, val, test, relevant_of_test, library, repeats=5):
    """How much the final R4 model relies on each feature and group, three ways:
       gain        - improvement of the training objective from the splits on a feature (validation users)
       shap        - mean |SHAP| on the test users, centred per user (a ranking only uses within-user differences)
       permutation - drop in test NDCG@10 when the feature (or whole group) is shuffled within each user"""
    columns = [c for group in MODELS["R4"] for c in GROUPS[group]]
    group_of = {c: g for g, cs_ in GROUPS.items() for c in cs_}
    assert (test["user_id"].to_numpy().reshape(-1, TOP_M) == test["user_id"].to_numpy()[::TOP_M, None]).all(), \
        "test rows are not 30 consecutive rows per user"

    # 1. Gain (from training)
    if library == "lightgbm":
        gain = pd.Series(model.booster_.feature_importance(importance_type="gain"), index=columns)
    else:
        gain = pd.Series(model.get_booster().get_score(importance_type="total_gain")).reindex(columns, fill_value=0.0)

    # 2. SHAP on the test users, centred within each user
    shap = shap_values(model, test[columns], library)
    shap_centred = shap - shap.groupby(test["user_id"]).transform("mean")
    mean_abs_shap = shap_centred.abs().mean()

    # 3. Permutation importance on the test users
    def ndcg(table):
        scored = table.assign(score=model.predict(table[columns]))
        return ndcg_per_user(scored, "score", relevant_of_test, ascending=False).mean()

    base = ndcg(test)
    rng = np.random.default_rng(42)
    perm = {c: np.mean([base - ndcg(shuffled_within_users(test, [c], rng)) for _ in range(repeats)])
            for c in tqdm(columns, desc=f"permutation per feature ({library})")}
    perm_group = {g: np.mean([base - ndcg(shuffled_within_users(test, GROUPS[g], rng)) for _ in range(repeats)])
                  for g in MODELS["R4"]}

    per_feature = pd.DataFrame({"group": [group_of[c] for c in columns],
                                "gain_share": gain / gain.sum(),
                                "shap_mean_abs": mean_abs_shap,
                                "shap_share": mean_abs_shap / mean_abs_shap.sum(),
                                "permutation_drop": pd.Series(perm)}, index=columns)
    per_feature.index.name = "feature"
    per_group = per_feature.groupby("group", sort=False)[["gain_share", "shap_share"]].sum()
    per_group["permutation_drop_group"] = pd.Series(perm_group)

    # Direction check: does the model trust the LLM's top choices? Mean centred SHAP by LLM rank bucket
    buckets = pd.cut(test["rank_deepseek"], [0, 10, 20, 30], labels=["1-10", "11-20", "21-30"])
    direction = pd.DataFrame({name: shap_centred[name].groupby(buckets, observed=True).mean()
                              for name in LLM_RANKERS})

    per_feature.sort_values("shap_share", ascending=False).to_csv(FOLDER / f"importance_features_{library}.csv")
    per_group.to_csv(FOLDER / f"importance_groups_{library}.csv")
    direction.to_csv(FOLDER / f"importance_llm_direction_{library}.csv")
    print(f"R4 test NDCG@10 (should match final_test): {base:.4f}")
    print(per_feature.sort_values("shap_share", ascending=False).round(4).to_string())
    print(per_group.round(4).to_string())
    print("Mean centred SHAP of the LLM rank features, by DeepSeek's rank (negative = pushed down):")
    print(direction.round(4).to_string())
    return per_feature, per_group


# ───────────────────────── LTR v2, step 1: simulated training users ─────────────────────────
STEP1_MODELS = ("R1", "R2", "R3")        # G4 needs LLM answers, which the warm users do not have


def half_view(data, train_half):
    """A copy of `data` whose item-CF table and popularity come from `train_half` only.
    Movies, embeddings and labels are shared (they use no ratings)."""
    view = dict(data)                                        # shallow copy: the big tables are shared
    view["item_cf"] = cs.item_cf_similarity(train_half, data["movies"], data["position_of"])
    view["pop_vector"] = cs.popularity_vector(cs.popularity_scores(train_half),
                                              data["movies"], data["position_of"])
    return view


def warm_halves(data, n_seeds=N_SEEDS):
    """The warm users' ratings, the A/B split (seed 42), and the simulated users of each half.
    As for the cold users, the seed pool is always the first 10 liked films (same users, same answer
    key, all 10 never recommended); n_seeds of them are given to the model."""
    ratings = bmt.prepare_ratings(data["movies"])
    train, _ = cs.split_users(ratings)                       # the same warm users as everywhere
    warm_ids = np.sort(train["user_id"].unique())
    half_of = dict(zip(warm_ids, np.random.default_rng(42).permutation(len(warm_ids)) % 2))
    in_A = train["user_id"].map(half_of) == 0
    pool_ids = {int(data["movie_ids"][p]) for p in data["pool"]}

    users_by_half = {}
    for half, this_half in (("A", train[in_A]), ("B", train[~in_A])):
        users_by_half[half] = cs.users_with_n_seeds(
            cs.build_cold_user_data(this_half, pool_ids, max(cs.SEED_COUNTS), cs.MIN_TEST_RELEVANT), n_seeds)
    return train, in_A, users_by_half


def build_training_table(data):
    """Warm users turned into simulated new users (first 10 liked films = seeds, the rest = answer
    key), with cross-fitting: the warm users are split into halves A and B, and the users of one
    half get their features from the CF table and popularity of the OTHER half. No user's own
    ratings are ever in the table that scores them. Cached in features_simulated.csv."""
    path = FOLDER / "features_simulated.csv"
    train, in_A, users_by_half = warm_halves(data)
    relevant_of = {u["user_id"]: u["relevant"] for users in users_by_half.values() for u in users}

    if path.exists():
        table = pd.read_csv(path)
        print(f"Loaded {path} ({table['user_id'].nunique()} users)")
    else:
        rows = []
        for half, other_half in (("A", train[~in_A]), ("B", train[in_A])):
            view = half_view(data, other_half)               # tables built WITHOUT this half's users
            for user in tqdm(users_by_half[half], desc=f"features (simulated, half {half})"):
                for row in candidate_rows(user, view, with_llm_ranks=False):
                    row["half"] = half                       # bookkeeping only, never a feature
                    rows.append(row)
        table = pd.DataFrame(rows)
        table.to_csv(path, index=False)

    # Checks
    n_users = sum(len(u) for u in users_by_half.values())
    assert table["user_id"].nunique() == n_users and len(table) == n_users * TOP_M, "not 30 rows per user"
    assert table.isna().sum().sum() == 0, f"missing values:\n{table.isna().sum()[table.isna().sum() > 0]}"
    assert not (set(table["user_id"]) & set(relevant_of_val_test(data))), "a cold user is in the training table"
    print(f"simulated: {n_users} users (A {len(users_by_half['A'])}, B {len(users_by_half['B'])}), "
          f"{len(table)} rows, {table['label'].mean():.1%} relevant")
    print(f"   Hybrid E on the simulated users (half tables): "
          f"{ndcg_by_score(table, 'hybrid_rank', relevant_of, True):.4f}")
    return table


def relevant_of_val_test(data):
    """User ids of the cold users (validation + test), to check they never enter training."""
    return {u["user_id"] for u in data["val_users"] + data["test_users"]}


def compare_training_size(simulated, val, relevant_of, library):
    """Step 1 question: does R1-R3 trained on the simulated users beat the same model trained on
    203 users? Both are scored on the 203 validation users: the 203-user model by its 5-fold
    cross-validation (cv_per_user_val_{library}.csv, doc 15), the new model trained once on the
    simulated users. Paired Wilcoxon tests, Bonferroni over all tests of this library."""
    old = pd.read_csv(FOLDER / f"cv_per_user_val_{library}.csv", index_col=0)
    results = {"Hybrid E": ndcg_per_user(val, "hybrid_rank", relevant_of, ascending=True)}
    for name in STEP1_MODELS:
        columns = [c for group in MODELS[name] for c in GROUPS[group]]
        model = fit_ranker(simulated, columns, library)
        results[f"{name} (203 users, CV)"] = old[name]
        results[f"{name} (simulated)"] = ndcg_per_user(val.assign(score=model.predict(val[columns])),
                                                       "score", relevant_of, ascending=False)
    per_user = pd.DataFrame(results)
    assert per_user.notna().all().all(), "validation users do not match the doc-15 CV file"

    tests = []
    for name in STEP1_MODELS:
        for a, b in [(f"{name} (203 users, CV)", f"{name} (simulated)"), ("Hybrid E", f"{name} (simulated)")]:
            diff = per_user[b] - per_user[a]
            tests.append({"comparison": f"{b} vs {a}", "difference": diff.mean(),
                          "better": int((diff > 0).sum()), "worse": int((diff < 0).sum()),
                          "p_value": wilcoxon(diff).pvalue if diff.any() else 1.0})
    tests = pd.DataFrame(tests)
    tests["p_bonferroni"] = (tests["p_value"] * len(tests)).clip(upper=1.0)
    per_user.to_csv(FOLDER / f"step1_per_user_val_{library}.csv")
    tests.to_csv(FOLDER / f"step1_tests_val_{library}.csv", index=False)
    print(per_user.mean().round(4).to_string())
    print(tests.round(4).to_string(index=False))
    return per_user, tests


# ───────────────────────── LTR v2, step 2: latent-factor and co-visitation features (G5) ─────────────────────────
SVD_RANK = 64        # fixed in advance, not tuned
RECENT_SEEDS = 3     # cf_recent: item-CF similarity to the last 3 seeds only


def latent_tables(train_part, data):
    """Three film x film tables from the ratings of `train_part` only (rows and columns in the order
    of data["movie_ids"]):
       svd_sim    cosine similarity of the SVD film vectors (U x S) of the user-centred rating matrix
       svd_proj   U U^T: summed over the seeds, the PureSVD fold-in score of a new user who liked them
       covis      P(candidate liked | seed liked): co-visitation count / number of users who liked the seed
    'Liked' = rating >= that user's own mean, the definition used everywhere."""
    from scipy.sparse import csr_matrix

    n_movies = len(data["movies"])
    rows = train_part["movie_id"].map(data["position_of"]).to_numpy()
    cols, user_ids = pd.factorize(train_part["user_id"])
    user_mean = train_part.groupby("user_id")["rating"].transform("mean")

    centred = csr_matrix(((train_part["rating"] - user_mean).to_numpy(np.float64), (rows, cols)),
                         shape=(n_movies, len(user_ids)))
    # Truncated SVD via the small film x film matrix M M^T: its top eigenvectors are U and its
    # eigenvalues are s^2. (scipy's svds/ARPACK crashes in this environment; this is exact and deterministic.)
    gram = (centred @ centred.T).toarray()
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    top = np.argsort(eigenvalues)[::-1][:SVD_RANK]
    U = eigenvectors[:, top]
    s = np.sqrt(np.maximum(eigenvalues[top], 0.0))
    vectors = U * s
    unit = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    svd_sim = (unit @ unit.T).astype(np.float32)
    svd_proj = (U @ U.T).astype(np.float32)

    liked = (train_part["rating"] >= user_mean).to_numpy()
    L = csr_matrix((np.ones(liked.sum(), np.float32), (rows[liked], cols[liked])), shape=(n_movies, len(user_ids)))
    together = (L @ L.T).toarray().astype(np.float32)
    covis = together / np.maximum(np.diag(together), 1.0)[:, None]     # row = seed, column = candidate

    for table in (svd_sim, svd_proj, covis):
        np.fill_diagonal(table, 0)                                     # a film is not its own neighbour
    return {"svd_sim": svd_sim, "svd_proj": svd_proj, "covis": covis}


def add_g5(table, users, view):
    """Add the G5 columns to an existing feature table (30 rows per user, in Hybrid E order).
    `view` must hold item_cf and the latent tables built WITHOUT these users' ratings."""
    user_of = {u["user_id"]: u for u in users}
    position_of = view["position_of"]
    out = {c: np.zeros(len(table), np.float32) for c in GROUPS["G5_latent"]}
    for user_id, group in table.groupby("user_id", sort=False):
        user = user_of[user_id]
        seeds = [position_of[m] for m in user["liked"]]
        recent = seeds[-RECENT_SEEDS:]                               # the seeds are in time order
        cands = [position_of[m] for m in group["movie_id"]]
        idx = table.index.get_indexer(group.index)
        svd_block = view["svd_sim"][np.ix_(seeds, cands)]
        covis_block = view["covis"][np.ix_(seeds, cands)]
        out["svd_mean"][idx] = svd_block.mean(axis=0)
        out["svd_max"][idx] = svd_block.max(axis=0)
        out["svd_score"][idx] = view["svd_proj"][np.ix_(seeds, cands)].sum(axis=0)
        out["covis_mean"][idx] = covis_block.mean(axis=0)
        out["covis_max"][idx] = covis_block.max(axis=0)
        out["cf_recent"][idx] = view["item_cf"][np.ix_(recent, cands)].mean(axis=0)
    return table.assign(**out)


def build_g5_tables(data):
    """features_val_g5.csv (full warm tables) and features_simulated_g5.csv (cross-fitted: each half
    gets G5 from the other half's tables, like the rest of its features). Cached."""
    val_path, sim_path = FOLDER / "features_val_g5.csv", FOLDER / "features_simulated_g5.csv"
    if val_path.exists() and sim_path.exists():
        print(f"Loaded {val_path.name} and {sim_path.name}")
        return pd.read_csv(val_path), pd.read_csv(sim_path)

    train, in_A, users_by_half = warm_halves(data)
    print("latent tables: full warm users")
    full = dict(data, **latent_tables(train, data))
    val = add_g5(pd.read_csv(FOLDER / "features_val.csv"),
                 cs.users_with_n_seeds(data["val_users"], N_SEEDS), full)

    simulated = build_training_table(data)
    parts = []
    for half, other_half in (("A", train[~in_A]), ("B", train[in_A])):
        print(f"latent tables: built without half {half}")
        view = half_view(data, other_half)
        view.update(latent_tables(other_half, data))
        parts.append(add_g5(simulated[simulated["half"] == half], users_by_half[half], view))
    simulated = pd.concat(parts).loc[simulated.index]                 # original row order

    for name, t in (("val", val), ("simulated", simulated)):
        assert t[GROUPS["G5_latent"]].notna().all().all(), f"missing G5 values ({name})"
        print(f"{name}: G5 means " + ", ".join(f"{c} {t[c].mean():.3f}" for c in GROUPS["G5_latent"]))
    val.to_csv(val_path, index=False)
    simulated.to_csv(sim_path, index=False)
    return val, simulated


def compare_g5(simulated, val, relevant_of, library):
    """Step 2 question: do the G5 features add to R2 and R3, all trained on the simulated users?
    Validation users only. Tests: R6 vs R2, R7 vs R3, R6 vs Hybrid E, R7 vs Hybrid E (Bonferroni over 4).
    Descriptive: each G5 feature alone added to R2."""
    results = {"Hybrid E": ndcg_per_user(val, "hybrid_rank", relevant_of, ascending=True)}
    for name in ("R2", "R3", "R6", "R7"):
        columns = [c for group in MODELS[name] for c in GROUPS[group]]
        model = fit_ranker(simulated, columns, library)
        results[name] = ndcg_per_user(val.assign(score=model.predict(val[columns])), "score", relevant_of, False)
    per_user = pd.DataFrame(results)

    tests = []
    for a, b in [("R2", "R6"), ("R3", "R7"), ("Hybrid E", "R6"), ("Hybrid E", "R7")]:
        diff = per_user[b] - per_user[a]
        tests.append({"comparison": f"{b} vs {a}", "difference": diff.mean(),
                      "better": int((diff > 0).sum()), "worse": int((diff < 0).sum()),
                      "p_value": wilcoxon(diff).pvalue if diff.any() else 1.0})
    tests = pd.DataFrame(tests)
    tests["p_bonferroni"] = (tests["p_value"] * len(tests)).clip(upper=1.0)

    r2_columns = [c for group in MODELS["R2"] for c in GROUPS[group]]
    single = {}
    for feature in GROUPS["G5_latent"]:
        model = fit_ranker(simulated, r2_columns + [feature], library)
        scores = ndcg_per_user(val.assign(score=model.predict(val[r2_columns + [feature]])), "score", relevant_of, False)
        single[f"R2 + {feature}"] = scores.mean() - per_user["R2"].mean()

    per_user.to_csv(FOLDER / f"step2_per_user_val_{library}.csv")
    tests.to_csv(FOLDER / f"step2_tests_val_{library}.csv", index=False)
    pd.Series(single, name="difference_vs_R2").to_csv(FOLDER / f"step2_single_features_val_{library}.csv")
    print(per_user.mean().round(4).to_string())
    print(tests.round(4).to_string(index=False))
    print("Each G5 feature alone added to R2 (validation NDCG difference, descriptive):")
    print(pd.Series(single).round(4).to_string())
    return per_user, tests


# ───────────────────────── LTR v2, step 3: wider candidate lists ─────────────────────────
VARIANTS = ("C30", "C50", "C30+20")
STEP3_MODELS = ("R6", "R7")


def wide_candidates(user, data, variant):
    """Candidates for one user, with their Hybrid E score and their rank in Hybrid E's FULL ranking.
       C30     Hybrid E's top 30 (identical to lr.hybrid_e_candidates)
       C50     Hybrid E's top 50
       C30+20  Hybrid E's top 30 + the 20 films with the highest content similarity (emb_max to the
               seeds) among the rest: films Hybrid E ranks low but that resemble the seeds"""
    position_of = data["position_of"]
    seen = {position_of[m] for m in user["seen"]}
    cand = np.array([p for p in data["pool"] if p not in seen])
    liked = [position_of[m] for m in user["liked"]]
    personal = cs.rescale(data["item_cf"][np.ix_(liked, cand)].mean(axis=0))
    popularity = cs.rescale(data["pop_vector"][cand])
    alpha = data["alpha_E"][len(user["liked"])]
    score = alpha * personal + (1 - alpha) * popularity
    order = np.argsort(-score, kind="stable")                     # Hybrid E's full ranking
    rank = np.empty(len(order), int)
    rank[order] = np.arange(1, len(order) + 1)

    if variant == "C30":
        chosen = order[:30]
    elif variant == "C50":
        chosen = order[:50]
    else:
        rest = order[30:]
        content = data["emb_base"][np.ix_(liked, cand[rest])].max(axis=0)
        chosen = np.concatenate([order[:30], rest[np.argsort(-content, kind="stable")[:20]]])
    ids = [int(m) for m in data["movie_ids"][cand[chosen]]]
    return ids, score[chosen], rank[chosen].tolist()


def oracle_ndcg(table, relevant_of):
    """Best possible NDCG@10 within each user's candidates (all relevant candidates first)."""
    return ndcg_per_user(table, "label", relevant_of, ascending=False).mean()


def build_wide_tables(data):
    """Validation (full warm tables) and simulated (cross-fitted) feature tables for every variant,
    G1-G3 + G5 (no G4). Cached as features_{val|simulated}_{variant}.csv."""
    paths = {(split, v): FOLDER / f"features_{split}_{v.replace('+', 'plus')}.csv"
             for split in ("val", "simulated") for v in VARIANTS}
    if all(p.exists() for p in paths.values()):
        print("Loaded the step-3 tables")
        return {key: pd.read_csv(p) for key, p in paths.items()}

    train, in_A, users_by_half = warm_halves(data)
    print("latent tables: full warm users")
    full = dict(data, **latent_tables(train, data))
    views = {}
    for half, other_half in (("A", train[~in_A]), ("B", train[in_A])):
        print(f"latent tables: built without half {half}")
        views[half] = half_view(data, other_half)
        views[half].update(latent_tables(other_half, data))
    val_users = cs.users_with_n_seeds(data["val_users"], N_SEEDS)

    tables = {}
    for v in VARIANTS:
        rows = [r for u in tqdm(val_users, desc=f"val {v}")
                for r in candidate_rows(u, full, False, wide_candidates(u, full, v))]
        tables[("val", v)] = add_g5(pd.DataFrame(rows), val_users, full)
        parts = []
        for half in ("A", "B"):
            rows = []
            for u in tqdm(users_by_half[half], desc=f"simulated {v}, half {half}"):
                for r in candidate_rows(u, views[half], False, wide_candidates(u, views[half], v)):
                    r["half"] = half
                    rows.append(r)
            parts.append(add_g5(pd.DataFrame(rows), users_by_half[half], views[half]))
        tables[("simulated", v)] = pd.concat(parts, ignore_index=True)

    # Check: C30 must reproduce the step-2 tables exactly
    for split in ("val", "simulated"):
        old = pd.read_csv(FOLDER / f"features_{split}_g5.csv")
        new = tables[(split, "C30")]
        columns = [c for name in MODELS["R7"] for c in GROUPS[name]] + ["user_id", "movie_id", "label"]
        worst = float(np.abs(new[columns].to_numpy(float) - old[columns].to_numpy(float)).max())
        print(f"check C30 = step-2 table ({split}): largest difference {worst:.2e}")
        assert worst < 1e-4, "C30 does not reproduce the step-2 features"
    for key, p in paths.items():
        assert tables[key].notna().all().all(), f"missing values in {key}"
        tables[key].to_csv(p, index=False)
    return tables


def compare_candidates(tables, relevant_of, library):
    """Step 3 question: do wider candidate lists help the best models (R6, R7)? Each variant is
    trained on its own simulated table and scored on its own validation table. Tests: C50 and
    C30+20 vs C30, for R6 and R7 (Bonferroni over 4). Oracle@M is the ceiling of each list."""
    hybrid = ndcg_per_user(tables[("val", "C30")], "hybrid_rank", relevant_of, ascending=True)
    results, summary = {"Hybrid E": hybrid}, []
    for v in VARIANTS:
        train, val = tables[("simulated", v)], tables[("val", v)]
        for name in STEP3_MODELS:
            columns = [c for group in MODELS[name] for c in GROUPS[group]]
            model = fit_ranker(train, columns, library)
            results[f"{name} {v}"] = ndcg_per_user(val.assign(score=model.predict(val[columns])),
                                                   "score", relevant_of, ascending=False)
        summary.append({"variant": v, "candidates": len(val) // val["user_id"].nunique(),
                        "relevant_share": val["label"].mean(), "oracle": oracle_ndcg(val, relevant_of),
                        **{name: results[f"{name} {v}"].mean() for name in STEP3_MODELS}})
    per_user = pd.DataFrame(results)

    tests = []
    for name in STEP3_MODELS:
        for v in VARIANTS[1:]:
            diff = per_user[f"{name} {v}"] - per_user[f"{name} C30"]
            tests.append({"comparison": f"{name} {v} vs {name} C30", "difference": diff.mean(),
                          "better": int((diff > 0).sum()), "worse": int((diff < 0).sum()),
                          "p_value": wilcoxon(diff).pvalue if diff.any() else 1.0})
    tests = pd.DataFrame(tests)
    tests["p_bonferroni"] = (tests["p_value"] * len(tests)).clip(upper=1.0)
    summary = pd.DataFrame(summary)
    per_user.to_csv(FOLDER / f"step3_per_user_val_{library}.csv")
    tests.to_csv(FOLDER / f"step3_tests_val_{library}.csv", index=False)
    summary.to_csv(FOLDER / f"step3_summary_val_{library}.csv", index=False)
    print(f"Hybrid E: {hybrid.mean():.4f}")
    print(summary.round(4).to_string(index=False))
    print(tests.round(4).to_string(index=False))
    return per_user, tests


# ───────────────────────── LTR v2, step 4: tuning and ensemble ─────────────────────────
from itertools import product

GRIDS = {   # 24 settings each, fixed in advance
    "xgboost":  {"max_depth": [3, 4, 6], "n_estimators": [200, 500], "learning_rate": [0.05, 0.1],
                 "min_child_weight": [1, 10]},
    "lightgbm": {"num_leaves": [15, 31, 63], "n_estimators": [200, 500], "learning_rate": [0.05, 0.1],
                 "min_child_samples": [20, 100]},
}
TUNING_HOLDOUT = 0.2      # share of the simulated users that scores each setting


def tune(simulated, relevant_sim, columns, library):
    """Grid search on the simulated users only: 80% train, 20% held out (split by user, seed 42).
    Returns the best setting by held-out NDCG@10, and the full grid table."""
    user_ids = np.sort(simulated["user_id"].unique())
    held = set(np.random.default_rng(42).permutation(user_ids)[:int(len(user_ids) * TUNING_HOLDOUT)])
    is_held = simulated["user_id"].isin(held)
    train, holdout = simulated[~is_held], simulated[is_held]

    grid = GRIDS[library]
    rows = []
    for values in tqdm(list(product(*grid.values())), desc=f"grid ({library})"):
        params = dict(zip(grid, values))
        model = fit_ranker(train, columns, library, params)
        ndcg = ndcg_per_user(holdout.assign(score=model.predict(holdout[columns])), "score",
                             relevant_sim, ascending=False).mean()
        rows.append({**params, "holdout_ndcg": ndcg})
    table = pd.DataFrame(rows).sort_values("holdout_ndcg", ascending=False, kind="stable")
    best = {k: (int(v) if float(v).is_integer() and k != "learning_rate" else float(v))
            for k, v in table.iloc[0][list(grid)].items()}
    default = {k: (PARAMS if library == "lightgbm" else XGB_PARAMS)[k] for k in grid}
    default_row = table[(table[list(grid)] == pd.Series(default)).all(axis=1)]
    print(f"{library}: {len(held)} held-out simulated users; best {best} "
          f"({table['holdout_ndcg'].iloc[0]:.4f}); default {default} "
          f"({default_row['holdout_ndcg'].iloc[0] if len(default_row) else float('nan'):.4f})")
    table.to_csv(FOLDER / f"step4_grid_{library}.csv", index=False)
    return best, table


def within_user_rank(table, scores):
    """Scores turned into within-user percentile ranks (0-1], so two models' scores are comparable."""
    return pd.Series(scores, index=table.index).groupby(table["user_id"]).rank(pct=True)


def compare_tuning(simulated, val, relevant_sim, relevant_of):
    """Step 4 question: do tuned settings and a 50/50 XGBoost + LightGBM ensemble beat the default
    models (R7, C30, trained on all simulated users)? Validation users. Tests (Bonferroni over 4):
    tuned vs default for each library, ensemble vs the better tuned single model, ensemble vs Hybrid E."""
    columns = [c for group in MODELS["R7"] for c in GROUPS[group]]
    results = {"Hybrid E": ndcg_per_user(val, "hybrid_rank", relevant_of, ascending=True)}
    ranks, chosen = {}, {}
    for library in ("xgboost", "lightgbm"):
        best, _ = tune(simulated, relevant_sim, columns, library)
        chosen[library] = best
        for label, params in (("default", None), ("tuned", best)):
            model = fit_ranker(simulated, columns, library, params)
            scores = model.predict(val[columns])
            results[f"{library} {label}"] = ndcg_per_user(val.assign(score=scores), "score", relevant_of, False)
            ranks[f"{library} {label}"] = within_user_rank(val, scores)
    for label in ("default", "tuned"):
        mixed = (ranks[f"xgboost {label}"] + ranks[f"lightgbm {label}"]) / 2
        results[f"ensemble {label}"] = ndcg_per_user(val.assign(score=mixed), "score", relevant_of, False)
    per_user = pd.DataFrame(results)

    best_single = max(("xgboost tuned", "lightgbm tuned"), key=lambda c: per_user[c].mean())
    tests = []
    for a, b in [("xgboost default", "xgboost tuned"), ("lightgbm default", "lightgbm tuned"),
                 (best_single, "ensemble tuned"), ("Hybrid E", "ensemble tuned")]:
        diff = per_user[b] - per_user[a]
        tests.append({"comparison": f"{b} vs {a}", "difference": diff.mean(),
                      "better": int((diff > 0).sum()), "worse": int((diff < 0).sum()),
                      "p_value": wilcoxon(diff).pvalue if diff.any() else 1.0})
    tests = pd.DataFrame(tests)
    tests["p_bonferroni"] = (tests["p_value"] * len(tests)).clip(upper=1.0)
    per_user.to_csv(FOLDER / "step4_per_user_val.csv")
    tests.to_csv(FOLDER / "step4_tests_val.csv", index=False)
    with open(FOLDER / "step4_chosen_params.json", "w", encoding="utf-8") as f:
        json.dump(chosen, f, indent=2)
    print(per_user.mean().round(4).to_string())
    print(tests.round(4).to_string(index=False))
    return per_user, tests


# ───────────────────────── Final model for the pre-registered test (chosen 2026-10-25) ─────────────────────────
# Chosen by rule after step 4, NOT by validation score: the tuned ensemble does not depend on which
# library happens to win (doc 15: the winner flipped between validation and test).
FINAL_MODEL = {
    "features": "R7",                    # G1 + G2 + G3 + G5
    "candidates": "C30",                 # Hybrid E's top 30
    "training": "all simulated users (cross-fitted)",
    "model": "50/50 average of within-user ranks of tuned XGBoost and tuned LightGBM",
    "params": "step4_chosen_params.json",
}


def ensemble_scores(train, table, columns, chosen):
    """The final model: tuned XGBoost and tuned LightGBM, each trained on `train`, combined by the
    mean of their within-user percentile ranks on `table`."""
    ranks = [within_user_rank(table, fit_ranker(train, columns, library, chosen[library]).predict(table[columns]))
             for library in ("xgboost", "lightgbm")]
    return (ranks[0] + ranks[1]) / 2


# ───────────────────────── LTR v2, step 5: three seeds ─────────────────────────
STEP5_SEEDS = 3
STEP5_MODELS = ("R2", "R3", "R6", "R7")


def build_seed_tables(data, n_seeds):
    """Validation (full warm tables) and simulated (cross-fitted) tables for n_seeds seeds, Hybrid E's
    top 30 for that seed count (its own alpha), G1-G3 + G5, no G4. The answer key and the excluded
    films are the same as with 10 seeds. Cached as features_{val|simulated}_n{n_seeds}.csv.
    Note: with 3 seeds cf_recent (last 3 seeds) equals cf_mean."""
    paths = {split: FOLDER / f"features_{split}_n{n_seeds}.csv" for split in ("val", "simulated")}
    if all(p.exists() for p in paths.values()):
        print(f"Loaded the {n_seeds}-seed tables")
        return {split: pd.read_csv(p) for split, p in paths.items()}

    train, in_A, users_by_half = warm_halves(data, n_seeds)
    print("latent tables: full warm users")
    full = dict(data, **latent_tables(train, data))
    val_users = cs.users_with_n_seeds(data["val_users"], n_seeds)
    rows = [r for u in tqdm(val_users, desc=f"val, {n_seeds} seeds")
            for r in candidate_rows(u, full, False, wide_candidates(u, full, "C30"))]
    tables = {"val": add_g5(pd.DataFrame(rows), val_users, full)}

    parts = []
    for half, other_half in (("A", train[~in_A]), ("B", train[in_A])):
        print(f"latent tables: built without half {half}")
        view = half_view(data, other_half)
        view.update(latent_tables(other_half, data))
        rows = []
        for u in tqdm(users_by_half[half], desc=f"simulated, {n_seeds} seeds, half {half}"):
            for r in candidate_rows(u, view, False, wide_candidates(u, view, "C30")):
                r["half"] = half
                rows.append(r)
        parts.append(add_g5(pd.DataFrame(rows), users_by_half[half], view))
    tables["simulated"] = pd.concat(parts, ignore_index=True)

    for split, t in tables.items():
        assert t.notna().all().all(), f"missing values ({split})"
        assert (t.groupby("user_id").size() == TOP_M).all(), f"not 30 rows per user ({split})"
        t.to_csv(paths[split], index=False)
    return tables


def compare_seeds(tables, relevant_of, n_seeds):
    """Step 5 questions, validation users, n_seeds seeds:
       1. does the final model (tuned ensemble, R7) still beat Hybrid E?
       2. do the LLM labels matter more with few seeds? R3 vs R2 and R7 vs R6, both libraries (defaults)
    Tests: Bonferroni over 5. Tuned single models are reported descriptively."""
    with open(FOLDER / "step4_chosen_params.json", encoding="utf-8") as f:
        chosen = json.load(f)
    train, val = tables["simulated"], tables["val"]
    hybrid = ndcg_per_user(val, "hybrid_rank", relevant_of, ascending=True)
    results = {"Hybrid E": hybrid}
    print(f"Hybrid E, {n_seeds} seeds (validation; doc 06 / 13: 0.5837 for 3 seeds): {hybrid.mean():.4f}")
    for library in ("xgboost", "lightgbm"):
        for name in STEP5_MODELS:
            columns = [c for group in MODELS[name] for c in GROUPS[group]]
            model = fit_ranker(train, columns, library)
            results[f"{library} {name}"] = ndcg_per_user(val.assign(score=model.predict(val[columns])),
                                                         "score", relevant_of, ascending=False)
        r7 = [c for group in MODELS["R7"] for c in GROUPS[group]]
        model = fit_ranker(train, r7, library, chosen[library])
        results[f"{library} R7 tuned"] = ndcg_per_user(val.assign(score=model.predict(val[r7])),
                                                       "score", relevant_of, ascending=False)
    r7 = [c for group in MODELS["R7"] for c in GROUPS[group]]
    results["ensemble R7 tuned (final model)"] = ndcg_per_user(
        val.assign(score=ensemble_scores(train, val, r7, chosen)), "score", relevant_of, ascending=False)
    per_user = pd.DataFrame(results)

    tests = []
    pairs = [("Hybrid E", "ensemble R7 tuned (final model)")]
    for library in ("xgboost", "lightgbm"):
        pairs += [(f"{library} R2", f"{library} R3"), (f"{library} R6", f"{library} R7")]
    for a, b in pairs:
        diff = per_user[b] - per_user[a]
        tests.append({"comparison": f"{b} vs {a}", "difference": diff.mean(),
                      "better": int((diff > 0).sum()), "worse": int((diff < 0).sum()),
                      "p_value": wilcoxon(diff).pvalue if diff.any() else 1.0})
    tests = pd.DataFrame(tests)
    tests["p_bonferroni"] = (tests["p_value"] * len(tests)).clip(upper=1.0)
    per_user.to_csv(FOLDER / f"step5_per_user_val_n{n_seeds}.csv")
    tests.to_csv(FOLDER / f"step5_tests_val_n{n_seeds}.csv", index=False)
    print(per_user.mean().round(4).to_string())
    print(tests.round(4).to_string(index=False))
    return per_user, tests


# ───────────────────────── LTR v2, step 6: pre-registered final test (step6_preregistration.md) ─────────────────────────
STEP6_DONE = FOLDER / "step6_tests_test.csv"        # exists after the one allowed run


def build_test_tables(data):
    """Test tables from the FULL warm users' ratings: 10 seeds = features_test.csv + G5 (same rows,
    same order); 3 seeds = Hybrid E's top 30 for 3 seeds, G1-G3 + G5. Cached."""
    paths = {10: FOLDER / "features_test_g5.csv", 3: FOLDER / f"features_test_n{STEP5_SEEDS}.csv"}
    if all(p.exists() for p in paths.values()):
        return {n: pd.read_csv(p) for n, p in paths.items()}

    train, _ = cs.split_users(bmt.prepare_ratings(data["movies"]))
    print("latent tables: full warm users")
    full = dict(data, **latent_tables(train, data))
    tables = {}
    users10 = cs.users_with_n_seeds(data["test_users"], 10)
    tables[10] = add_g5(pd.read_csv(FOLDER / "features_test.csv"), users10, full)
    users3 = cs.users_with_n_seeds(data["test_users"], STEP5_SEEDS)
    rows = [r for u in tqdm(users3, desc=f"test, {STEP5_SEEDS} seeds")
            for r in candidate_rows(u, full, False, wide_candidates(u, full, "C30"))]
    tables[3] = add_g5(pd.DataFrame(rows), users3, full)
    for n, t in tables.items():
        t.to_csv(paths[n], index=False)
    return tables


def bootstrap_ci(diff, resamples=10_000):
    """95% bootstrap confidence interval of a mean per-user difference (users resampled, seed 42)."""
    values = diff.to_numpy()
    idx = np.random.default_rng(42).integers(0, len(values), size=(resamples, len(values)))
    means = values[idx].mean(axis=1)
    return np.percentile(means, 2.5), np.percentile(means, 97.5)


def final_test_v2(data):
    """The pre-registered test (step6_preregistration.md). Runs once: stops if results exist."""
    if STEP6_DONE.exists():
        sys.exit(f"The final test has already been run ({STEP6_DONE}). It is run only once.")
    with open(FOLDER / "step4_chosen_params.json", encoding="utf-8") as f:
        chosen = json.load(f)
    r7 = [c for group in MODELS["R7"] for c in GROUPS[group]]
    r2 = [c for group in MODELS["R2"] for c in GROUPS[group]]
    tests_by_n = build_test_tables(data)
    train_by_n = {10: pd.read_csv(FOLDER / "features_simulated_g5.csv"),
                  3: pd.read_csv(FOLDER / f"features_simulated_n{STEP5_SEEDS}.csv")}
    relevant_of = {u["user_id"]: u["relevant"] for u in data["test_users"]}     # the same for 3 and 10 seeds

    # Sanity checks (section 4 of the pre-registration), before any final-model score is computed
    expected_hybrid = {10: 0.6311, 3: 0.5970}
    test_ids = {u["user_id"] for u in data["test_users"]}
    old10 = pd.read_csv(FOLDER / "features_test.csv")
    assert (tests_by_n[10][["user_id", "movie_id"]].to_numpy() == old10[["user_id", "movie_id"]].to_numpy()).all(), \
        "check 3: the 10-seed G5 table does not match features_test.csv"
    for n in (10, 3):
        t = tests_by_n[n]
        assert t[r7].notna().all().all(), f"check 3: missing values ({n} seeds)"
        assert (t.groupby("user_id").size() == TOP_M).all() and set(t["user_id"]) == test_ids, f"check 3 ({n} seeds)"
        assert not (set(train_by_n[n]["user_id"]) & test_ids), f"check 4: a test user is in training ({n} seeds)"
        hybrid = ndcg_per_user(t, "hybrid_rank", relevant_of, ascending=True).mean()
        print(f"check 1: Hybrid E, {n} seeds: {hybrid:.4f} (expected {expected_hybrid[n]})")
        assert abs(hybrid - expected_hybrid[n]) < 5e-5, "check 1 failed"
    v1 = fit_ranker(pd.read_csv(FOLDER / "features_val.csv"), r2, "xgboost")
    v1_scores = ndcg_per_user(old10.assign(score=v1.predict(old10[r2])), "score", relevant_of, ascending=False)
    print(f"check 2: XGBoost R2 trained on 203 users: {v1_scores.mean():.4f} (expected 0.6446)")
    assert abs(v1_scores.mean() - 0.6446) < 5e-5, "check 2 failed"

    # Scores
    per_user, summary = {}, []
    for n in (10, 3):
        t, train = tests_by_n[n], train_by_n[n]
        results = {"Hybrid E": ndcg_per_user(t, "hybrid_rank", relevant_of, ascending=True),
                   "final model": ndcg_per_user(t.assign(score=ensemble_scores(train, t, r7, chosen)),
                                                "score", relevant_of, ascending=False)}
        for library in ("xgboost", "lightgbm"):                         # descriptive only
            model = fit_ranker(train, r7, library, chosen[library])
            results[f"{library} R7 tuned"] = ndcg_per_user(t.assign(score=model.predict(t[r7])),
                                                            "score", relevant_of, ascending=False)
        if n == 10:
            results["XGBoost R2, 203 users (LTR v1)"] = v1_scores
        per_user[n] = pd.DataFrame(results)
        summary.append({"seeds": n, "oracle@30": oracle_ndcg(t, relevant_of),
                        **per_user[n].mean().to_dict()})

    # The three pre-registered tests, Bonferroni over 3
    tests = []
    for label, n, baseline in [("H1", 10, "Hybrid E"), ("H2", 10, "XGBoost R2, 203 users (LTR v1)"),
                               ("H3", 3, "Hybrid E")]:
        diff = per_user[n]["final model"] - per_user[n][baseline]
        low, high = bootstrap_ci(diff)
        p = wilcoxon(diff).pvalue
        tests.append({"hypothesis": label, "seeds": n, "comparison": f"final model vs {baseline}",
                      "final_model": per_user[n]["final model"].mean(), "baseline": per_user[n][baseline].mean(),
                      "difference": diff.mean(), "ci_low": low, "ci_high": high,
                      "better": int((diff > 0).sum()), "worse": int((diff < 0).sum()),
                      "p_value": p, "p_bonferroni": min(1.0, 3 * p)})
    tests = pd.DataFrame(tests)
    tests["confirmed"] = (tests["p_bonferroni"] < 0.05) & (tests["difference"] > 0)

    for n in (10, 3):
        per_user[n].to_csv(FOLDER / f"step6_per_user_test_n{n}.csv")
    pd.DataFrame(summary).to_csv(FOLDER / "step6_summary_test.csv", index=False)
    tests.to_csv(STEP6_DONE, index=False)                               # written last: marks the test as run
    pd.set_option("display.width", 250)
    print(pd.DataFrame(summary).round(4).to_string(index=False))
    print(tests.round(4).to_string(index=False))
    return tests


if __name__ == "__main__":
    # python src/LightGBM_XBoost_rerank_experiment.py final
    #   -> LTR v2 step 6: the pre-registered test on the 814 test users (step6_preregistration.md). Runs ONCE.
    #      Needs the step-2, step-4 and step-5 outputs.
    if MODE == "final":
        final_test_v2(load_input())
        sys.exit()

    # python src/LightGBM_XBoost_rerank_experiment.py seeds3
    #   -> LTR v2 step 5: the same pipeline with 3 seeds, 203 validation users only (test users not touched)
    #      Needs step4_chosen_params.json (run the "tuning" mode first).
    if MODE == "seeds3":
        data = load_input()
        relevant_of = {u["user_id"]: u["relevant"] for u in cs.users_with_n_seeds(data["val_users"], STEP5_SEEDS)}
        tables = build_seed_tables(data, STEP5_SEEDS)
        compare_seeds(tables, relevant_of, STEP5_SEEDS)
        sys.exit()

    # python src/LightGBM_XBoost_rerank_experiment.py tuning
    #   -> LTR v2 step 4 (R7, C30): grid search on simulated users, compared on the 203 validation users
    if MODE == "tuning":
        data = load_input()
        relevant_of = {u["user_id"]: u["relevant"] for u in cs.users_with_n_seeds(data["val_users"], N_SEEDS)}
        _, _, users_by_half = warm_halves(data)
        relevant_sim = {u["user_id"]: u["relevant"] for users in users_by_half.values() for u in users}
        val, simulated = build_g5_tables(data)                     # loads the cached step-2 (C30) tables
        compare_tuning(simulated, val, relevant_sim, relevant_of)
        sys.exit()

    # python src/LightGBM_XBoost_rerank_experiment.py candidates
    #   -> LTR v2 step 3 on the 203 validation users only (the test users are not touched)
    if MODE == "candidates":
        data = load_input()
        relevant_of = {u["user_id"]: u["relevant"] for u in cs.users_with_n_seeds(data["val_users"], N_SEEDS)}
        tables = build_wide_tables(data)
        for library in ("xgboost", "lightgbm"):
            print(f"\n===== {library}: candidate lists, trained on simulated users =====")
            compare_candidates(tables, relevant_of, library)
        sys.exit()

    # python src/LightGBM_XBoost_rerank_experiment.py g5
    #   -> LTR v2 step 2 on the 203 validation users only (the test users are not touched)
    if MODE == "g5":
        data = load_input()
        relevant_of = {u["user_id"]: u["relevant"] for u in cs.users_with_n_seeds(data["val_users"], N_SEEDS)}
        val, simulated = build_g5_tables(data)
        for library in ("xgboost", "lightgbm"):
            print(f"\n===== {library}: G5 latent features, trained on simulated users =====")
            compare_g5(simulated, val, relevant_of, library)
        sys.exit()

    # python src/LightGBM_XBoost_rerank_experiment.py simulated
    #   -> LTR v2 step 1 on the 203 validation users only (the test users are not touched)
    if MODE == "simulated":
        data = load_input()
        val = pd.read_csv(FOLDER / "features_val.csv")       # full warm tables, as in doc 15
        relevant_of = {u["user_id"]: u["relevant"] for u in cs.users_with_n_seeds(data["val_users"], N_SEEDS)}
        simulated = build_training_table(data)
        for library in ("xgboost", "lightgbm"):
            print(f"\n===== {library}: trained on simulated users vs on 203 users =====")
            compare_training_size(simulated, val, relevant_of, library)
        sys.exit()

    data = load_input() if not IMPORTANCE_ONLY else lr.load_data()
    users = cs.users_with_n_seeds(data["val_users"], N_SEEDS)
    relevant_of = {u["user_id"]: u["relevant"] for u in users}

    test = pd.read_csv(FOLDER / "features_test.csv")
    test_users = cs.users_with_n_seeds(data["test_users"], N_SEEDS)
    relevant_of_test = {u["user_id"]: u["relevant"] for u in test_users}

    # python src/LightGBM_XBoost_rerank_experiment.py importance
    #   -> feature importance of the final R4 models only (no ablation rerun, no LLM calls)
    if IMPORTANCE_ONLY:
        val = pd.read_csv(FOLDER / "features_val.csv")
        r4_columns = [c for group in MODELS["R4"] for c in GROUPS[group]]
        for library in ("lightgbm", "xgboost"):
            print(f"\n===== {library}: feature importance of R4 =====")
            feature_importance(fit_ranker(val, r4_columns, library), val, test, relevant_of_test, library)
        sys.exit()

    val = build_table(data, "val")
    for library in ("lightgbm", "xgboost"):
        print(f"\n===== {library} =====")
        run_ablation(val, relevant_of, library)
        final_test(val, test, relevant_of_test, library)