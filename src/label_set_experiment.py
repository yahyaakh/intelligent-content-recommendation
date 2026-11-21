"""Do measurably better labels give better recommendations?

Three label sets for the same 1,957 films, compared on the same users, split, candidate pool and
metrics as experiments 05/08 (warm start) and 06/08 (cold start):

    v2   gemma4:e4b, prompt v2 (44 themes, 18 moods)                  experiment 05
    v4   gemma4:e4b, prompt v4 (Tag Genome vocabulary)                experiment 08
    v5   DeepSeek V4.1 Flash, prompt v5 (v4 + 9 themes + aliases)     label-quality experiment

The label-quality experiment showed that the v5 DeepSeek labels match the Tag Genome clearly better
than the gemma v4 labels (theme precision 0.675 vs 0.625, mood precision 0.772 vs 0.692 on 1,388
films). This experiment asks whether that translates into recommendation quality.

Methods (as in experiment 08): content-based (CountVectorizer metadata + TF-IDF text; the labels
added as tokens) and embeddings (embeddinggemma of the base text plus "Themes: ... Mood: ...").
A = content without labels, B_x = content with label set x; C = embedding without labels,
D_x = embedding with label set x. Cold-start hybrids: alpha chosen on the validation users for the
v5 hybrid of each family and used unchanged for the other label sets of that family.

Pre-registered expectations (written before the evaluation, 2026-10-25):
    - warm: v5 within about +-0.005 NDCG@10 of v4 for embeddings; content may gain most
    - cold hybrids: no significant difference between the label sets

Usage:  python src/label_set_experiment.py
"""
import json
from pathlib import Path

import faiss
import numpy as np
import pandas as pd

import build_movie_texts as bmt
import cold_start_experiment as cs

FOLDER = Path("output") / "experiments" / "label_sets"
V4 = Path("output") / "experiments" / "gemma4_e4b_v4"
V5 = Path("output") / "experiments" / "deepseek_v4.1_flash_v5"
INDEX = "movies_enriched_index"
LABEL_SETS = ("v2", "v4", "v5")

# Experiment 08's warm results (max / mean scoring) that this script must reproduce
EXPECTED_WARM = {"A_base": (0.2789, 0.0625), "B_v2": (0.2824, 0.0622), "B_v4": (0.2918, 0.1027),
                 "C_base": (0.3152, 0.2732), "D_v2": (0.3186, 0.2892), "D_v4": (0.3169, 0.2853)}


def build_v5_index(movies_v5):
    """Enriched texts and embeddinggemma index for the v5 labels (reused if they already exist)."""
    texts = bmt.build_texts(movies_v5, out_path=V5 / "movie_texts.csv")
    base_info = json.loads((bmt.EMBEDDINGS_FOLDER / "movies_base_embeddinggemma_v2.json").read_text(encoding="utf-8"))
    if (V5 / f"{INDEX}.faiss").exists():
        ids = np.load(V5 / f"{INDEX}_ids.npy")
        assert (ids == texts["movie_id"].to_numpy()).all(), "saved v5 index is in a different movie order"
        print("Reusing the saved v5 embedding index")
        return
    matrix = bmt.embed_texts(texts["text_enriched"].tolist(), model=base_info["model"], prefix=base_info["prefix"])
    assert matrix.shape[0] == len(texts) and np.allclose(np.linalg.norm(matrix, axis=1), 1, atol=1e-3)
    bmt.save_index(bmt.build_index(matrix), texts["movie_id"].to_numpy(), INDEX,
                   {"model": base_info["model"], "prefix": base_info["prefix"], "text_column": "text_enriched",
                    "labels": str(V5 / "movies_enriched.csv"), "prompt_version": "v5", "dimension": int(matrix.shape[1]),
                    "movies": int(matrix.shape[0]), "normalized": True, "index_type": "IndexFlatIP"}, folder=V5)


def warm(tables, movies, ratings):
    train, test = bmt.split_ratings(ratings)
    users = bmt.build_user_data(train, test)
    movie_ids = movies["movie_id"].to_numpy()
    position_of = {m: i for i, m in enumerate(movie_ids)}
    pool = np.array(sorted(position_of[m] for m in set(train["movie_id"])))
    results = pd.concat([bmt.evaluate_method(n, t, users, pool, position_of, movie_ids, bmt.K, s)
                         for n, t in tables.items() for s in ("max", "mean")])
    popularity = bmt.evaluate_popularity(bmt.popularity_ranking(train), users, bmt.K)
    tests = []
    for s in ("max", "mean"):
        nd = results[results["scoring"] == s].pivot(index="user_id", columns="method", values="ndcg")
        for a, b in [("D_v4", "D_v5"), ("D_v2", "D_v5"), ("C_base", "D_v5"),
                     ("B_v4", "B_v5"), ("B_v2", "B_v5"), ("A_base", "B_v5")]:
            tests.append({"scoring": s, **bmt.wilcoxon_row(f"{b} vs {a}", nd[a], nd[b])})
    return results, popularity, pd.DataFrame(tests), len(users)


def cold(tables, movies, ratings):
    train, cold_ratings = cs.split_users(ratings)
    pool_ids = set(train["movie_id"])
    cold_users = cs.build_cold_user_data(cold_ratings, pool_ids, max(cs.SEED_COUNTS), cs.MIN_TEST_RELEVANT)
    movie_ids = movies["movie_id"].to_numpy()
    position_of = {m: i for i, m in enumerate(movie_ids)}
    pool = np.array(sorted(position_of[m] for m in pool_ids))
    popular = cs.popularity_scores(train)
    pop_vector = cs.popularity_vector(popular, movies, position_of)
    val_users, test_users = cs.split_validation(cold_users, cs.VAL_FRACTION)

    # alpha chosen on the validation users for the v5 hybrid of each family; the other label
    # sets of the same family use the SAME alpha, so differences come from the labels only
    families = {"embedding": ("D_v5", "D_v4", "D_v2", "C_base"), "content": ("B_v5", "B_v4", "B_v2", "A_base")}
    alphas = {f: cs.choose_alphas(names[0], tables[names[0]], pop_vector, val_users, pool,
                                  position_of, movie_ids, bmt.K)[0] for f, names in families.items()}

    results = []
    for n in cs.SEED_COUNTS:
        group = cs.users_with_n_seeds(test_users, n)
        rows = [{"method": "popularity", "scoring": "-", "user_id": u["user_id"],
                 **bmt.user_metrics(cs.recommend_popular(popular.index, u, bmt.K), u["relevant"], bmt.K)}
                for u in group]
        results.append(pd.DataFrame(rows).assign(n=n))
        for name, table in tables.items():
            for s in ("max", "mean"):
                results.append(bmt.evaluate_method(name, table, group, pool, position_of, movie_ids,
                                                   bmt.K, s).assign(n=n))
        for family, names in families.items():
            alpha = alphas[family][n]
            for name in names:
                results.append(cs.evaluate_hybrid(f"hybrid_{name}", tables[name], pop_vector, group, pool,
                                                  position_of, movie_ids, bmt.K, alpha, "mean")
                               .assign(n=n, alpha=alpha))
    results = pd.concat(results, ignore_index=True)

    tests = []
    for n, part in results.groupby("n"):
        nd = part.pivot_table(index="user_id", columns=["method", "scoring"], values="ndcg")
        for a, b in [(("hybrid_D_v4", "mean"), ("hybrid_D_v5", "mean")),
                     (("hybrid_C_base", "mean"), ("hybrid_D_v5", "mean")),
                     (("hybrid_B_v4", "mean"), ("hybrid_B_v5", "mean")),
                     (("hybrid_A_base", "mean"), ("hybrid_B_v5", "mean")),
                     (("D_v4", "mean"), ("D_v5", "mean")),
                     (("B_v4", "max"), ("B_v5", "max"))]:
            tests.append({"n": n, **bmt.wilcoxon_row(f"{b[0]} ({b[1]}) vs {a[0]} ({a[1]})", nd[a], nd[b])})
    return results, pd.DataFrame(tests), alphas, len(test_users)


if __name__ == "__main__":
    FOLDER.mkdir(parents=True, exist_ok=True)
    movies = bmt.load_movies()                                   # catalog + v2 labels
    for path, version in ((V4, "v4"), (V5, "v5")):
        labels = pd.read_csv(path / "movies_enriched.csv")
        assert labels["movie_id"].is_unique and set(labels["movie_id"]) == set(movies["movie_id"]), path
        assert (labels["prompt_version"] == version).all(), f"{path}: not prompt {version}"
    movies_v4 = bmt.with_labels_from(movies, V4 / "movies_enriched.csv")
    movies_v5 = bmt.with_labels_from(movies, V5 / "movies_enriched.csv")
    build_v5_index(movies_v5)

    tables = {
        "A_base": bmt.content_similarity(movies, with_enrichment=False),
        "B_v2": bmt.content_similarity(movies, with_enrichment=True),
        "B_v4": bmt.content_similarity(movies_v4, with_enrichment=True),
        "B_v5": bmt.content_similarity(movies_v5, with_enrichment=True),
        "C_base": bmt.embedding_similarity("movies_base_embeddinggemma_v2", movies),
        "D_v2": bmt.embedding_similarity("movies_enriched_embeddinggemma_v2", movies),
        "D_v4": bmt.embedding_similarity(INDEX, movies, folder=V4),
        "D_v5": bmt.embedding_similarity(INDEX, movies, folder=V5),
    }
    ratings = bmt.prepare_ratings(movies)
    w_results, w_pop, w_tests, n_warm = warm(tables, movies, ratings)
    c_results, c_tests, alphas, n_cold = cold(tables, movies, ratings)
    total = len(w_tests) + len(c_tests)
    for df in (w_tests, c_tests):
        df["p_bonferroni"] = (df["p_value"] * total).clip(upper=1.0)

    w_results.to_csv(FOLDER / "warm_per_user.csv", index=False)
    w_tests.to_csv(FOLDER / "warm_significance_tests.csv", index=False)
    c_results.to_csv(FOLDER / "cold_per_user.csv", index=False)
    c_tests.to_csv(FOLDER / "cold_significance_tests.csv", index=False)
    with open(FOLDER / "settings.json", "w", encoding="utf-8") as f:
        json.dump({"labels": {"v2": "output/enrichment/movies_enriched_v2.csv", "v4": str(V4 / "movies_enriched.csv"),
                              "v5": str(V5 / "movies_enriched.csv")},
                   "warm_users": n_warm, "cold_test_users": n_cold,
                   "cold_alphas_chosen_on_v5": {f: {str(n): a for n, a in v.items()} for f, v in alphas.items()}},
                  f, indent=2)

    pd.set_option("display.width", 220)
    warm_means = w_results.groupby(["method", "scoring"])["ndcg"].mean().unstack("scoring")
    print(f"\nWARM ({n_warm} users), NDCG@10; check against experiment 08")
    for method, row in warm_means.iterrows():
        expected = EXPECTED_WARM.get(method)
        check = "" if expected is None else ("  ok" if abs(row["max"] - expected[0]) < 5e-4 and
                                             abs(row["mean"] - expected[1]) < 5e-4 else f"  DIFFERS from {expected}")
        print(f"   {method:7} max {row['max']:.4f} | mean {row['mean']:.4f}{check}")
    print(f"   popularity {w_pop['ndcg'].mean():.4f}")
    print(w_tests[["scoring", "comparison", "mean_difference", "p_value", "p_bonferroni",
                   "users_better", "users_worse"]].round(4).to_string(index=False))
    print(f"\nCOLD ({n_cold} test users), NDCG@10")
    print(c_results.groupby(["method", "scoring", "n"])["ndcg"].mean().unstack("n").round(4).to_string())
    print(c_tests[["n", "comparison", "mean_difference", "p_value", "p_bonferroni",
                   "users_better", "users_worse"]].round(4).to_string(index=False))
    print(f"\nSaved results to {FOLDER}")
