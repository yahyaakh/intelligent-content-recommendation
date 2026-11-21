"""Larger vocabulary experiment (thesis doc 08): v2 labels (44 themes, 18 moods) vs
v4 labels (246 themes, 103 moods from the MovieLens Tag Genome), both from gemma4:e4b.

Labels come from run_enrichment_experiment("gemma4:e4b", prompt_version="v4")
(output/experiments/gemma4_e4b_v4/). Evaluation uses the same users, split, pool and
metrics as experiments 05 (warm) and 06 (cold).

Pre-registered expectations (written before the full evaluation)
    - v4 labels are more specific (fewer movies per label) and mostly confirmed by the genome
    - warm max scoring: v4 within about +-0.005 NDCG of v2; content-based may change most
    - cold hybrid: no significant difference

Usage:  python experiments/enrichment/vocabulary_experiment.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))  # shared modules live in src/
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import build_movie_texts as bmt
import cold_start_experiment as cs

FOLDER = Path("output") / "experiments" / "vocabulary"
V4 = Path("output") / "experiments" / "gemma4_e4b_v4"


def split_labels(value):
    return [] if pd.isna(value) or not str(value).strip() else str(value).split("|")


# ───────────────────────── label quality ─────────────────────────
def label_statistics(v2, v4):
    rows = []
    for name, df in [("v2", v2), ("v4", v4)]:
        for field in ("llm_themes", "llm_mood"):
            labels = df[field].map(split_labels)
            freq = Counter(l for ls in labels for l in ls)
            share = {l: c / len(df) for l, c in freq.items()}
            rows.append({"labels": name, "field": field, "vocabulary_used": len(freq),
                         "labels_per_movie": labels.map(len).mean(),
                         "most_common": f"{freq.most_common(1)[0][0]} ({freq.most_common(1)[0][1] / len(df):.0%})",
                         # specificity: average share of the catalog that carries an assigned label
                         "mean_share_of_assigned_label": np.mean([share[l] for ls in labels for l in ls])})
    return pd.DataFrame(rows)


def genome_accuracy(v4, vocabulary):
    """Share of assigned v4 labels with genome relevance >= 0.5, against a random-label baseline."""
    links = pd.read_csv("data/ml-25m/links.csv").dropna(subset=["tmdbId"])
    links["tmdbId"] = links["tmdbId"].astype(int)
    ml_of = dict(zip(links["tmdbId"], links["movieId"]))
    tags = pd.read_csv("data/ml-25m/genome/genome-tags.csv")
    vocab_ids = {t: i for t, i in zip(tags["tag"], tags["tagId"])
                 if t in set(vocabulary["themes"]) | set(vocabulary["moods"])}
    wanted = {ml_of[m] for m in v4["movie_id"] if m in ml_of}
    rel = {}
    for chunk in pd.read_csv("data/ml-25m/genome/genome-scores.csv", chunksize=2_000_000):
        chunk = chunk[chunk["movieId"].isin(wanted) & chunk["tagId"].isin(vocab_ids.values())]
        rel.update({(r.movieId, r.tagId): r.relevance for r in chunk.itertuples()})
    have = {m for m, _ in rel}

    out = []
    for field, allowed in [("llm_themes", vocabulary["themes"]), ("llm_mood", vocabulary["moods"])]:
        assigned, confirmed, base = 0, 0, []
        for movie_id, value in zip(v4["movie_id"], v4[field]):
            ml = ml_of.get(movie_id)
            if ml not in have:
                continue
            labels = split_labels(value)
            assigned += len(labels)
            confirmed += sum(rel[(ml, vocab_ids[l])] >= 0.5 for l in labels)
            base.append(np.mean([rel[(ml, vocab_ids[t])] >= 0.5 for t in allowed]))
        out.append({"field": field, "movies_with_genome": len(base), "labels_assigned": assigned,
                    "confirmed_share": confirmed / assigned, "random_label_share": float(np.mean(base))})
    return pd.DataFrame(out)


# ───────────────────────── evaluation ─────────────────────────
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
        for a, b in [("D_v2", "D_v4"), ("C_base", "D_v4"), ("C_base", "D_v2"),
                     ("B_v2", "B_v4"), ("A_base", "B_v4"), ("A_base", "B_v2"), ("D_v2", "B_v4")]:
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

    # alpha chosen on validation users for the v4 hybrids; the v2 and no-label hybrids
    # of the same family are evaluated at the SAME alpha.
    families = {"embedding": ("D_v4", "D_v2", "C_base"), "content": ("B_v4", "B_v2", "A_base")}
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
        for a, b in [(("hybrid_D_v2", "mean"), ("hybrid_D_v4", "mean")),
                     (("hybrid_C_base", "mean"), ("hybrid_D_v4", "mean")),
                     (("hybrid_B_v2", "mean"), ("hybrid_B_v4", "mean")),
                     (("hybrid_A_base", "mean"), ("hybrid_B_v4", "mean")),
                     (("D_v2", "mean"), ("D_v4", "mean")),
                     (("B_v2", "max"), ("B_v4", "max")),
                     (("A_base", "max"), ("B_v4", "max"))]:
            tests.append({"n": n, **bmt.wilcoxon_row(f"{b[0]} ({b[1]}) vs {a[0]} ({a[1]})", nd[a], nd[b])})
    return results, pd.DataFrame(tests), alphas, len(test_users)


if __name__ == "__main__":
    FOLDER.mkdir(parents=True, exist_ok=True)
    movies = bmt.load_movies()                                   # catalog + v2 labels
    v4_labels = pd.read_csv(V4 / "movies_enriched.csv")
    assert v4_labels["movie_id"].is_unique and set(v4_labels["movie_id"]) == set(movies["movie_id"])
    assert (v4_labels["prompt_version"] == "v4").all(), "labels were not made with prompt v4"
    movies_v4 = bmt.with_labels_from(movies, V4 / "movies_enriched.csv")
    vocabulary = json.loads(Path("data/vocabularies/v4_genome.json").read_text(encoding="utf-8"))

    stats = label_statistics(movies, movies_v4)
    genome = genome_accuracy(movies_v4, vocabulary)
    stats.to_csv(FOLDER / "label_statistics.csv", index=False)
    genome.to_csv(FOLDER / "genome_accuracy.csv", index=False)

    tables = {
        "A_base": bmt.content_similarity(movies, with_enrichment=False),
        "B_v2": bmt.content_similarity(movies, with_enrichment=True),
        "B_v4": bmt.content_similarity(movies_v4, with_enrichment=True),
        "C_base": bmt.embedding_similarity("movies_base_embeddinggemma_v2", movies),
        "D_v2": bmt.embedding_similarity("movies_enriched_embeddinggemma_v2", movies),
        "D_v4": bmt.embedding_similarity("movies_enriched_index", movies, folder=V4),
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
        json.dump({"v4_labels": str(V4 / "movies_enriched.csv"), "vocabulary": "data/vocabularies/v4_genome.json",
                   "warm_users": n_warm, "cold_test_users": n_cold,
                   "cold_alphas": {f: {str(n): a for n, a in v.items()} for f, v in alphas.items()}}, f, indent=2)

    pd.set_option("display.width", 220)
    print("\nLABEL STATISTICS"); print(stats.round(3).to_string(index=False))
    print("\nGENOME ACCURACY (v4 labels, relevance >= 0.5)"); print(genome.round(3).to_string(index=False))
    print(f"\nWARM ({n_warm} users), NDCG@10")
    print(w_results.groupby(["method", "scoring"])[["ndcg", "precision", "recall"]].mean().round(4).to_string())
    print(f"Popularity: {w_pop['ndcg'].mean():.4f}")
    print(w_tests[["scoring", "comparison", "mean_difference", "p_value", "p_bonferroni",
                   "users_better", "users_worse"]].round(4).to_string(index=False))
    print(f"\nCOLD ({n_cold} test users), NDCG@10")
    summary = c_results.groupby(["method", "scoring", "n"])["ndcg"].mean().unstack("n").round(4)
    print(summary.to_string())
    print(c_tests[["n", "comparison", "mean_difference", "p_value", "p_bonferroni",
                   "users_better", "users_worse"]].round(4).to_string(index=False))
    print(f"\nSaved results to {FOLDER}")
