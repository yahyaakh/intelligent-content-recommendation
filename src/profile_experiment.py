"""Outside-knowledge experiment (thesis doc 07): evaluate free-text film profiles.

Profiles come from enrich_profiles.py (P1 = given text only, P2 = given text + the model's
own knowledge), for gemma4:e4b and gemma4:31b-cloud. Each profile is appended to the base
text, embedded exactly like the labels, and evaluated with the same users, split, candidate
pool and metrics as experiments 05 (warm start) and 06 (cold start).

Pre-registered hypotheses
    H1  P2 profiles beat P1 profiles (warm NDCG@10, max scoring; cold-start hybrid at the same alpha)
    H2  the P2 gain is larger for well-known films (high TMDB vote count) than for obscure ones
    H3  the P2 gain is larger with gemma4:31b than with gemma4:e4b
    Secondary: P1 profiles vs the existing gemma4:e4b labels (free text vs fixed labels;
               format and kind of information change together)

Usage:  python src/profile_experiment.py
"""
import json
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from tqdm import tqdm

import build_movie_texts as bmt
import cold_start_experiment as cs

FOLDER = Path("output") / "experiments" / "profiles"
PROFILES = Path("output") / "enrichment" / "profiles"
CONDITIONS = {                       # short name -> profile file
    "e4b_P1": "gemma4_e4b_P1.csv",
    "e4b_P2": "gemma4_e4b_P2.csv",
    "31b_P1": "gemma4_31b-cloud_P1.csv",
    "31b_P2": "gemma4_31b-cloud_P2.csv",
}
BASE_INDEX = "movies_base_embeddinggemma_v2"
LABEL_INDEX = "movies_enriched_embeddinggemma_v2"


# ───────────────────────── texts and embeddings ─────────────────────────
def build_profile_texts(movies):
    base = pd.read_csv("output/enrichment/movie_texts_v2.csv")[["movie_id", "text_base"]]
    assert list(base["movie_id"]) == list(movies["movie_id"]), "base texts are in a different order"
    texts = base.copy()
    for name, file in CONDITIONS.items():
        profiles = pd.read_csv(PROFILES / file)
        assert profiles["movie_id"].is_unique and set(profiles["movie_id"]) == set(movies["movie_id"]), \
            f"{file}: not exactly one profile per catalog movie"
        assert profiles["profile"].notna().all() and (profiles["profile"].str.len() > 0).all(), f"{file}: empty profile"
        profile_of = dict(zip(profiles["movie_id"], profiles["profile"]))
        # The only difference to the base text: one appended sentence group "Profile: ...".
        texts[f"text_{name}"] = [f"{b} Profile: {profile_of[m]}" for m, b in zip(texts["movie_id"], texts["text_base"])]
    texts.to_csv(FOLDER / "movie_texts_profiles.csv", index=False)
    return texts


def profile_vectors(texts):
    with open(bmt.EMBEDDINGS_FOLDER / f"{BASE_INDEX}.json", encoding="utf-8") as f:
        base_info = json.load(f)
    vectors = {}
    for name in CONDITIONS:
        index_name = f"movies_profile_{name}_embeddinggemma"
        if (FOLDER / f"{index_name}.faiss").exists():
            print(f"Reusing saved embeddings for {name}")
        else:
            matrix = bmt.embed_texts(texts[f"text_{name}"].tolist(), model=base_info["model"],
                                     prefix=base_info["prefix"])
            info = {"model": base_info["model"], "prefix": base_info["prefix"],
                    "text_column": f"text_{name}", "profile_file": CONDITIONS[name],
                    "dimension": int(matrix.shape[1]), "movies": int(matrix.shape[0]),
                    "normalized": True, "index_type": "IndexFlatIP"}
            bmt.save_index(bmt.build_index(matrix), texts["movie_id"].to_numpy(), index_name, info, folder=FOLDER)
        vectors[name] = index_name
    return vectors


# ───────────────────────── warm start ─────────────────────────
def popularity_thirds(movies, pool_movie_ids):
    """Movie id -> 'low' / 'mid' / 'high' by TMDB vote count, thirds over the candidate pool."""
    pool = movies[movies["movie_id"].isin(pool_movie_ids)]
    cuts = pool["vote_count"].quantile([1 / 3, 2 / 3]).to_numpy()
    third = np.where(pool["vote_count"] <= cuts[0], "low", np.where(pool["vote_count"] <= cuts[1], "mid", "high"))
    return dict(zip(pool["movie_id"], third)), cuts


def evaluate_warm(tables, users, pool_positions, position_of, movie_ids, third_of):
    rows = []
    for name, table in tables.items():
        for scoring in ("max", "mean"):
            for user in tqdm(users, desc=f"warm {name} ({scoring})"):
                rec = bmt.recommend(table, user, pool_positions, position_of, movie_ids, bmt.K, scoring)
                row = {"method": name, "scoring": scoring, "user_id": user["user_id"],
                       **bmt.user_metrics(rec, user["relevant"], bmt.K)}
                # H2: the same list, scored only against relevant movies of one popularity third
                for t in ("low", "mid", "high"):
                    rel_t = {m for m in user["relevant"] if third_of.get(m) == t}
                    row[f"ndcg_{t}"] = bmt.ndcg_at_k(rec, rel_t, bmt.K) if rel_t else np.nan
                rows.append(row)
    return pd.DataFrame(rows)


def warm_tests(results, popularity):
    tests = []
    for scoring in ("max", "mean"):
        part = results[results["scoring"] == scoring]
        nd = part.pivot(index="user_id", columns="method", values="ndcg")
        pairs = [("e4b_P1", "e4b_P2", "H1 e4b: P2 vs P1"), ("31b_P1", "31b_P2", "H1 31b: P2 vs P1"),
                 ("C_base", "e4b_P1", "e4b P1 vs no enrichment"), ("C_base", "e4b_P2", "e4b P2 vs no enrichment"),
                 ("C_base", "31b_P1", "31b P1 vs no enrichment"), ("C_base", "31b_P2", "31b P2 vs no enrichment"),
                 ("D_labels_e4b", "e4b_P1", "secondary: e4b P1 profile vs e4b labels"),
                 ("D_labels_e4b", "31b_P2", "31b P2 profile vs e4b labels"),
                 ("e4b_P2", "31b_P2", "31b P2 vs e4b P2")]
        for a, b, label in pairs:
            tests.append({"scoring": scoring, **bmt.wilcoxon_row(label, nd[a], nd[b])})
        # H3: is the P2 - P1 gain larger with 31b than with e4b?
        gain_31b = nd["31b_P2"] - nd["31b_P1"]
        gain_e4b = nd["e4b_P2"] - nd["e4b_P1"]
        tests.append({"scoring": scoring, **bmt.wilcoxon_row("H3: (31b P2-P1) vs (e4b P2-P1)", gain_e4b, gain_31b)})
        # H2: P2 - P1 within each popularity third
        for t in ("low", "mid", "high"):
            nt = part.pivot(index="user_id", columns="method", values=f"ndcg_{t}")
            for model in ("e4b", "31b"):
                both = nt[[f"{model}_P1", f"{model}_P2"]].dropna()
                tests.append({"scoring": scoring, **bmt.wilcoxon_row(
                    f"H2 {model}: P2 vs P1, relevant movies in {t} third",
                    both[f"{model}_P1"], both[f"{model}_P2"])})
    tests = pd.DataFrame(tests)
    tests["p_bonferroni"] = (tests["p_value"] * len(tests)).clip(upper=1.0)
    return tests


# ───────────────────────── cold start ─────────────────────────
def evaluate_cold(movies, ratings, tables):
    train, cold = cs.split_users(ratings)
    pool_movie_ids = set(train["movie_id"])
    cold_users = cs.build_cold_user_data(cold, pool_movie_ids, max(cs.SEED_COUNTS), cs.MIN_TEST_RELEVANT)
    movie_ids = movies["movie_id"].to_numpy()
    position_of = {m: i for i, m in enumerate(movie_ids)}
    pool_positions = np.array(sorted(position_of[m] for m in pool_movie_ids))
    popular = cs.popularity_scores(train)
    pop_vector = cs.popularity_vector(popular, movies, position_of)
    val_users, test_users = cs.split_validation(cold_users, cs.VAL_FRACTION)

    # alpha chosen on validation users for each model's P1 hybrid; P2 and the
    # no-enrichment baseline are evaluated at the SAME alpha, so only the profile differs.
    alphas = {}
    for model in ("e4b", "31b"):
        alphas[model], _ = cs.choose_alphas(f"{model}_P1", tables[f"{model}_P1"], pop_vector, val_users,
                                            pool_positions, position_of, movie_ids, bmt.K)

    results = []
    for n in cs.SEED_COUNTS:
        group = cs.users_with_n_seeds(test_users, n)
        rows = [{"method": "popularity", "scoring": "-", "user_id": u["user_id"],
                 **bmt.user_metrics(cs.recommend_popular(popular.index, u, bmt.K), u["relevant"], bmt.K)}
                for u in group]
        results.append(pd.DataFrame(rows).assign(n=n))
        for name, table in tables.items():
            results.append(bmt.evaluate_method(name, table, group, pool_positions, position_of,
                                               movie_ids, bmt.K, "mean").assign(n=n))
        for model in ("e4b", "31b"):
            alpha = alphas[model][n]
            for name in (f"{model}_P1", f"{model}_P2", "C_base", "D_labels_e4b"):
                results.append(cs.evaluate_hybrid(f"hybrid_{name}@{model}_alpha", tables[name], pop_vector,
                                                  group, pool_positions, position_of, movie_ids,
                                                  bmt.K, alpha, "mean").assign(n=n, alpha=alpha))
    results = pd.concat(results, ignore_index=True)

    tests = []
    for n, part in results.groupby("n"):
        nd = part.pivot_table(index="user_id", columns="method", values="ndcg")
        for model in ("e4b", "31b"):
            h = lambda name: f"hybrid_{name}@{model}_alpha"
            for a, b, label in [(h(f"{model}_P1"), h(f"{model}_P2"), f"H1 {model}: hybrid P2 vs P1"),
                                (h("C_base"), h(f"{model}_P2"), f"{model}: hybrid P2 vs no enrichment"),
                                (h("D_labels_e4b"), h(f"{model}_P1"), f"{model}: hybrid P1 vs e4b labels"),
                                (f"{model}_P1", f"{model}_P2", f"{model}: pure P2 vs P1 (mean)")]:
                tests.append({"n": n, **bmt.wilcoxon_row(label, nd[a], nd[b])})
    tests = pd.DataFrame(tests)
    tests["p_bonferroni"] = (tests["p_value"] * len(tests)).clip(upper=1.0)
    return results, tests, alphas, len(test_users)


if __name__ == "__main__":
    FOLDER.mkdir(parents=True, exist_ok=True)
    movies = bmt.load_movies()
    texts = build_profile_texts(movies)
    index_names = profile_vectors(texts)

    tables = {"C_base": bmt.embedding_similarity(BASE_INDEX, movies),
              "D_labels_e4b": bmt.embedding_similarity(LABEL_INDEX, movies)}
    for name, index_name in index_names.items():
        tables[name] = bmt.embedding_similarity(index_name, movies, folder=FOLDER)

    # How much did each profile move the movies? (cosine between base and profile vectors)
    movie_ids = movies["movie_id"].to_numpy()
    for name, index_name in index_names.items():
        a = faiss.read_index(str(bmt.EMBEDDINGS_FOLDER / f"{BASE_INDEX}.faiss")).reconstruct_n(0, len(movies))
        b = faiss.read_index(str(FOLDER / f"{index_name}.faiss")).reconstruct_n(0, len(movies))
        print(f"{name}: base vs profile vector cosine, mean {(a * b).sum(axis=1).mean():.3f}")

    ratings = bmt.prepare_ratings(movies)

    # Warm start: the same split, users and pool as experiment 05
    train, test = bmt.split_ratings(ratings)
    users = bmt.build_user_data(train, test)
    position_of = {m: i for i, m in enumerate(movie_ids)}
    pool_movie_ids = set(train["movie_id"])
    pool_positions = np.array(sorted(position_of[m] for m in pool_movie_ids))
    third_of, cuts = popularity_thirds(movies, pool_movie_ids)
    print(f"Popularity thirds by TMDB vote count: low <= {cuts[0]:.0f} < mid <= {cuts[1]:.0f} < high")

    warm = evaluate_warm(tables, users, pool_positions, position_of, movie_ids, third_of)
    popularity = bmt.evaluate_popularity(bmt.popularity_ranking(train), users, bmt.K)
    warm.to_csv(FOLDER / "warm_per_user.csv", index=False)
    warm_summary = warm.groupby(["method", "scoring"])[["ndcg", "precision", "recall",
                                                        "ndcg_low", "ndcg_mid", "ndcg_high"]].mean()
    warm_summary.round(4).to_csv(FOLDER / "warm_summary.csv")
    wt = warm_tests(warm, popularity)
    wt.to_csv(FOLDER / "warm_significance_tests.csv", index=False)

    # Cold start: the same users, seeds and validation/test split as experiment 06
    cold, ct, alphas, n_test = evaluate_cold(movies, ratings, tables)
    cold.to_csv(FOLDER / "cold_per_user.csv", index=False)
    cold_summary = cold.groupby(["method", "scoring", "n"])[["ndcg", "precision", "recall"]].mean().round(4).reset_index()
    cold_summary.to_csv(FOLDER / "cold_summary.csv", index=False)
    ct.to_csv(FOLDER / "cold_significance_tests.csv", index=False)

    with open(FOLDER / "settings.json", "w", encoding="utf-8") as f:
        json.dump({"conditions": CONDITIONS, "base_index": BASE_INDEX, "label_index": LABEL_INDEX,
                   "popularity_third_cuts": [float(c) for c in cuts], "k": bmt.K, "seed": bmt.SEED,
                   "warm_users": len(users), "cold_test_users": n_test,
                   "cold_alphas": {m: {str(n): a for n, a in v.items()} for m, v in alphas.items()}}, f, indent=2)

    pd.set_option("display.width", 220)
    print("\nWARM START (NDCG@10; ndcg_low/mid/high = only relevant movies of that popularity third)")
    print(warm_summary.round(4).to_string())
    print(f"Popularity: {popularity['ndcg'].mean():.4f}")
    print("\nWarm tests:")
    print(wt[["scoring", "comparison", "mean_difference", "p_value", "p_bonferroni",
              "users_better", "users_worse"]].round(4).to_string(index=False))
    print(f"\nCOLD START (NDCG@10, {n_test} test users)")
    print(cold_summary.pivot_table(index=["method", "scoring"], columns="n", values="ndcg").round(4).to_string())
    print("\nCold tests:")
    print(ct[["n", "comparison", "mean_difference", "p_value", "p_bonferroni",
              "users_better", "users_worse"]].round(4).to_string(index=False))
    print(f"\nSaved results to {FOLDER}")
