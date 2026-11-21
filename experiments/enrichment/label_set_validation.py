"""
Cold-start label comparison on the 203 VALIDATION users (Chapter 5, test 2).

label_set_experiment.py reported the cold-start label comparison on the 814 test users, before the
final pre-registered test. The report's protocol (Chapter 3) keeps test users for pre-registered
tests only, so this script repeats the same comparison on the validation users.

Everything is identical to label_set_experiment.cold(): the same similarity tables, candidate pool,
seeds, metrics and hybrids; alpha chosen on the validation users for the DeepSeek (v5) hybrid of each
family and used unchanged for the other label sets.

Caveat, reported with the results: alpha is chosen on these same 203 users, with the v5 hybrid.
This can favour the v5 hybrids slightly; the pure methods have no tuned setting.

Checks: the chosen alphas must equal those in output/experiments/label_sets/settings.json, and the
v5 hybrid's validation NDCG@10 must equal the best value of the alpha curve.

    python experiments/enrichment/label_set_validation.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))  # shared modules live in src/
import json
from pathlib import Path

import numpy as np
import pandas as pd

import build_movie_texts as bmt
import cold_start_experiment as cs
import label_equivalence as le
import label_set_experiment as lse

OUT = lse.FOLDER / "validation"


def build_tables():
    movies = bmt.load_movies()
    movies_v4 = bmt.with_labels_from(movies, lse.V4 / "movies_enriched.csv")
    movies_v5 = bmt.with_labels_from(movies, lse.V5 / "movies_enriched.csv")
    tables = {
        "A_base": bmt.content_similarity(movies, with_enrichment=False),
        "B_v4": bmt.content_similarity(movies_v4, with_enrichment=True),
        "B_v5": bmt.content_similarity(movies_v5, with_enrichment=True),
        "C_base": bmt.embedding_similarity("movies_base_embeddinggemma_v2", movies),
        "D_v4": bmt.embedding_similarity(lse.INDEX, movies, folder=lse.V4),
        "D_v5": bmt.embedding_similarity(lse.INDEX, movies, folder=lse.V5),
    }
    return movies, tables


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    movies, tables = build_tables()
    ratings = bmt.prepare_ratings(movies)

    train, cold_ratings = cs.split_users(ratings)
    pool_ids = set(train["movie_id"])
    cold_users = cs.build_cold_user_data(cold_ratings, pool_ids, max(cs.SEED_COUNTS), cs.MIN_TEST_RELEVANT)
    movie_ids = movies["movie_id"].to_numpy()
    position_of = {m: i for i, m in enumerate(movie_ids)}
    pool = np.array(sorted(position_of[m] for m in pool_ids))
    popular = cs.popularity_scores(train)
    pop_vector = cs.popularity_vector(popular, movies, position_of)
    val_users, test_users = cs.split_validation(cold_users, cs.VAL_FRACTION)
    print(f"validation users: {len(val_users)}, test users: {len(test_users)}")

    families = {"embedding": ("D_v5", "D_v4", "C_base"), "content": ("B_v5", "B_v4", "A_base")}
    alphas, curves = {}, []
    for family, names in families.items():
        alphas[family], curve = cs.choose_alphas(names[0], tables[names[0]], pop_vector, val_users, pool,
                                                 position_of, movie_ids, bmt.K)
        curves.append(curve)
    curves = pd.concat(curves, ignore_index=True)

    # Check 1: same alphas as the original run
    saved = json.loads((lse.FOLDER / "settings.json").read_text(encoding="utf-8"))["cold_alphas_chosen_on_v5"]
    for family in families:
        assert {str(n): a for n, a in alphas[family].items()} == saved[family], \
            f"{family}: alphas {alphas[family]} differ from the saved {saved[family]}"
    print("check: chosen alphas equal the saved ones")

    results = []
    for n in cs.SEED_COUNTS:
        group = cs.users_with_n_seeds(val_users, n)
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

    # Check 2: the v5 hybrid at its chosen alpha reproduces the best point of its alpha curve
    for family, names in families.items():
        for n in cs.SEED_COUNTS:
            best = curves[(curves["table"] == names[0]) & (curves["n"] == n)]["ndcg"].max()
            got = results[(results["method"] == f"hybrid_{names[0]}") & (results["n"] == n)]["ndcg"].mean()
            assert abs(best - got) < 1e-12, f"{family} n={n}: {got} vs curve {best}"
    print("check: v5 hybrids reproduce the alpha curve")

    results.to_csv(OUT / "cold_per_user.csv", index=False)
    curves.to_csv(OUT / "alpha_curve.csv", index=False)

    # Wilcoxon, same comparisons as the equivalence family, Bonferroni over all of them
    tests = []
    for n in le.COLD_N:
        part = results[results["n"] == n]
        nd = part.pivot_table(index="user_id", columns=["method", "scoring"], values="ndcg")
        for family, a, b, name in le.COLD:
            tests.append({"n": n, **bmt.wilcoxon_row(name, nd[a], nd[b])})
    tests = pd.DataFrame(tests)
    tests["p_bonferroni"] = (tests["p_value"] * len(tests)).clip(upper=1.0)
    tests.to_csv(OUT / "significance_tests.csv", index=False)

    # Equivalence (bootstrap CI, Bonferroni-adjusted), same settings as label_equivalence.py
    rng = np.random.default_rng(le.SEED)
    m = len(le.COLD) * len(le.COLD_N)
    eq = []
    for n in le.COLD_N:
        eq += le.analyse(le.COLD, results[results["n"] == n], "cold (203 validation users)", n, m, rng)
    eq = pd.DataFrame(eq).merge(tests[["n", "comparison", "p_value", "p_bonferroni"]], on=["n", "comparison"])
    eq.to_csv(OUT / "equivalence.csv", index=False)

    with open(OUT / "settings.json", "w", encoding="utf-8") as f:
        json.dump({"users": "validation", "validation_users": len(val_users),
                   "alphas_chosen_on_v5": {fam: {str(n): a for n, a in v.items()} for fam, v in alphas.items()},
                   "caveat": "alpha chosen on these same validation users with the v5 hybrid",
                   "bootstrap": le.B, "seed": le.SEED, "margins": list(le.MARGINS), "family_size": m},
                  f, indent=2)

    pd.set_option("display.width", 220)
    summary = results.groupby(["method", "scoring", "n"])[["ndcg", "precision", "recall"]].mean()
    print(f"\nCOLD, {len(val_users)} validation users")
    print(summary[summary.index.get_level_values("n").isin(le.COLD_N)].round(4).to_string())
    show = eq.copy()
    for c in ["mean_diff", "ci_bonf_low", "ci_bonf_high"]:
        show[c] = show[c].map(lambda x: f"{x:+.4f}")
    show["p_bonferroni"] = show["p_bonferroni"].round(3)
    print(show[["n", "comparison", "mean_diff", "ci_bonf_low", "ci_bonf_high", "p_bonferroni",
                "equivalent_0.01", "equivalent_0.02"]].to_string(index=False))
    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    main()
