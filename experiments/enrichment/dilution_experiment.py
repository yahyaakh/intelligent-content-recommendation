"""Dilution test (thesis doc 07): LLM extras appended to the text vs used as a separate weighted similarity.

Appended (experiments 05-07): base text + extra embedded as ONE text. The extra moves the
embedding by only ~1% (cosine 0.986-0.991), so its information may be drowned out.
Blended (this script): base text and extra embedded SEPARATELY and combined as
    similarity = w * base + (1 - w) * extra
with w tuned on validation users only.

Extras: the e4b labels ("Themes: ... Mood: ..."), and the four free-text profiles
(e4b / 31b, P1 = given text only, P2 = + the model's own knowledge).

Warm start: w tuned on 20% of the warm users, reported on the other 80% (every method on the
same 80%). Cold start: w and the hybrid alpha tuned together on the 203 validation users,
reported on the 814 test users; each blended hybrid is compared with the base hybrid and
the appended hybrid at the SAME alpha.

Usage:  python experiments/enrichment/dilution_experiment.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))  # shared modules live in src/
import json
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from tqdm import tqdm

import build_movie_texts as bmt
import cold_start_experiment as cs

FOLDER = Path("output") / "experiments" / "dilution"
PROFILE_FOLDER = Path("output") / "experiments" / "profiles"
WEIGHTS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
EXTRAS = ["labels_e4b", "e4b_P1", "e4b_P2", "31b_P1", "31b_P2"]
VAL_FRACTION_WARM = 0.20


# ───────────────────────── tables ─────────────────────────
def extra_only_texts(movies):
    """The extra on its own: the label sentence, or 'Profile: ...'."""
    labelled = pd.read_csv("output/enrichment/movie_texts_v2.csv")
    profiles = pd.read_csv(PROFILE_FOLDER / "movie_texts_profiles.csv")
    assert list(labelled.movie_id) == list(movies.movie_id) == list(profiles.movie_id)
    texts = pd.DataFrame({"movie_id": movies.movie_id})
    texts["labels_e4b"] = [e[len(b):].strip() for b, e in zip(labelled.text_base, labelled.text_enriched)]
    for name in EXTRAS[1:]:
        texts[name] = [p[len(b):].strip() for b, p in zip(profiles.text_base, profiles[f"text_{name}"])]
    for name in EXTRAS:
        assert texts[name].str.len().gt(0).all(), name
        assert texts[name].str.startswith("Profile:" if name != "labels_e4b" else ("Themes:", "Mood:")).all(), name
    texts.to_csv(FOLDER / "extra_only_texts.csv", index=False)
    return texts


def extra_only_tables(movies, texts):
    with open(bmt.EMBEDDINGS_FOLDER / "movies_base_embeddinggemma_v2.json", encoding="utf-8") as f:
        info = json.load(f)
    tables = {}
    for name in EXTRAS:
        index_name = f"extra_only_{name}_embeddinggemma"
        if not (FOLDER / f"{index_name}.faiss").exists():
            matrix = bmt.embed_texts(texts[name].tolist(), model=info["model"], prefix=info["prefix"])
            bmt.save_index(bmt.build_index(matrix), texts.movie_id.to_numpy(), index_name,
                           {"model": info["model"], "prefix": info["prefix"], "text": f"{name} only",
                            "dimension": int(matrix.shape[1]), "movies": int(matrix.shape[0]),
                            "normalized": True, "index_type": "IndexFlatIP"}, folder=FOLDER)
        tables[name] = bmt.embedding_similarity(index_name, movies, folder=FOLDER)
    return tables


def blend(base, extra, w):
    return (w * base + (1 - w) * extra).astype(np.float32)


# ───────────────────────── warm start ─────────────────────────
def mean_ndcg(table, users, pool, position_of, movie_ids, scoring):
    return float(np.mean([bmt.user_metrics(bmt.recommend(table, u, pool, position_of, movie_ids, bmt.K, scoring),
                                           u["relevant"], bmt.K)["ndcg"] for u in users]))


def warm(movies, ratings, base, appended, extra):
    train, test = bmt.split_ratings(ratings)
    users = bmt.build_user_data(train, test)
    rng = np.random.default_rng(42)
    order = rng.permutation(len(users))
    n_val = int(len(users) * VAL_FRACTION_WARM)
    val = [users[i] for i in order[:n_val]]
    test_users = [users[i] for i in order[n_val:]]
    print(f"Warm: {len(val)} validation users, {len(test_users)} test users")

    movie_ids = movies.movie_id.to_numpy()
    position_of = {m: i for i, m in enumerate(movie_ids)}
    pool = np.array(sorted(position_of[m] for m in set(train.movie_id)))

    chosen, curve = {}, []
    for scoring in ("max", "mean"):
        for name in EXTRAS:
            best = (-1, None)
            for w in WEIGHTS:
                score = mean_ndcg(blend(base, extra[name], w), val, pool, position_of, movie_ids, scoring)
                curve.append({"scoring": scoring, "extra": name, "w": w, "val_ndcg": score})
                if score > best[0]:                      # strict ">": ties keep the larger w (closer to base)
                    best = (score, w)
            chosen[(scoring, name)] = best[1]
            print(f"  warm {scoring} {name}: best w {best[1]} (val NDCG {best[0]:.4f})")

    rows = []
    for scoring in ("max", "mean"):
        methods = {"base": base, **{f"appended_{n}": appended[n] for n in EXTRAS},
                   **{f"blended_{n}": blend(base, extra[n], chosen[(scoring, n)]) for n in EXTRAS}}
        for method, table in methods.items():
            for u in tqdm(test_users, desc=f"warm test {method} ({scoring})"):
                rec = bmt.recommend(table, u, pool, position_of, movie_ids, bmt.K, scoring)
                rows.append({"method": method, "scoring": scoring, "user_id": u["user_id"],
                             **bmt.user_metrics(rec, u["relevant"], bmt.K)})
    results = pd.DataFrame(rows)
    pop = bmt.evaluate_popularity(bmt.popularity_ranking(train), test_users, bmt.K)

    tests = []
    for scoring in ("max", "mean"):
        nd = results[results.scoring == scoring].pivot(index="user_id", columns="method", values="ndcg")
        for n in EXTRAS:
            tests.append({"scoring": scoring, **bmt.wilcoxon_row(f"blended {n} vs base", nd["base"], nd[f"blended_{n}"])})
            tests.append({"scoring": scoring, **bmt.wilcoxon_row(f"blended {n} vs appended {n}",
                                                                 nd[f"appended_{n}"], nd[f"blended_{n}"])})
        tests.append({"scoring": scoring, **bmt.wilcoxon_row("blended 31b_P2 vs blended 31b_P1",
                                                             nd["blended_31b_P1"], nd["blended_31b_P2"])})
        tests.append({"scoring": scoring, **bmt.wilcoxon_row("blended 31b_P2 vs blended labels_e4b",
                                                             nd["blended_labels_e4b"], nd["blended_31b_P2"])})
    tests = pd.DataFrame(tests)
    return results, pop, tests, pd.DataFrame(curve), chosen, len(test_users)


# ───────────────────────── cold start ─────────────────────────
def cold(movies, ratings, base, appended, extra):
    train, cold_ratings = cs.split_users(ratings)
    pool_ids = set(train.movie_id)
    cold_users = cs.build_cold_user_data(cold_ratings, pool_ids, max(cs.SEED_COUNTS), cs.MIN_TEST_RELEVANT)
    movie_ids = movies.movie_id.to_numpy()
    position_of = {m: i for i, m in enumerate(movie_ids)}
    pool = np.array(sorted(position_of[m] for m in pool_ids))
    popular = cs.popularity_scores(train)
    pop_vector = cs.popularity_vector(popular, movies, position_of)
    val_users, test_users = cs.split_validation(cold_users, cs.VAL_FRACTION)

    def hybrid_val_ndcg(table, group, alpha):
        return float(np.mean([bmt.user_metrics(cs.recommend_hybrid(table, pop_vector, u, pool, position_of,
                                                                   movie_ids, bmt.K, alpha, "mean"),
                                               u["relevant"], bmt.K)["ndcg"] for u in group]))

    chosen = {}
    for n in cs.SEED_COUNTS:
        group = cs.users_with_n_seeds(val_users, n)
        for name in EXTRAS:
            best = (-1, None, None)
            for w in tqdm(WEIGHTS, desc=f"cold tune {name} n={n}", leave=False):
                table = blend(base, extra[name], w)
                for alpha in cs.ALPHAS:
                    score = hybrid_val_ndcg(table, group, alpha)
                    if score > best[0]:
                        best = (score, w, alpha)
            chosen[(n, name)] = best[1:]
            print(f"  cold n={n} {name}: best w {best[1]}, alpha {best[2]} (val NDCG {best[0]:.4f})")

    results = []
    for n in cs.SEED_COUNTS:
        group = cs.users_with_n_seeds(test_users, n)
        for name in EXTRAS:
            w, alpha = chosen[(n, name)]
            for method, table in [(f"hybrid_blended_{name}", blend(base, extra[name], w)),
                                  (f"hybrid_base@{name}", base),
                                  (f"hybrid_appended_{name}@{name}", appended[name])]:
                results.append(cs.evaluate_hybrid(method, table, pop_vector, group, pool, position_of,
                                                  movie_ids, bmt.K, alpha, "mean").assign(n=n, w=w, alpha=alpha))
        rows = [{"method": "popularity", "scoring": "-", "user_id": u["user_id"],
                 **bmt.user_metrics(cs.recommend_popular(popular.index, u, bmt.K), u["relevant"], bmt.K)}
                for u in group]
        results.append(pd.DataFrame(rows).assign(n=n))
    results = pd.concat(results, ignore_index=True)

    tests = []
    for n, part in results.groupby("n"):
        nd = part.pivot_table(index="user_id", columns="method", values="ndcg")
        for name in EXTRAS:
            tests.append({"n": n, **bmt.wilcoxon_row(f"hybrid blended {name} vs base (same alpha)",
                                                     nd[f"hybrid_base@{name}"], nd[f"hybrid_blended_{name}"])})
            tests.append({"n": n, **bmt.wilcoxon_row(f"hybrid blended {name} vs appended (same alpha)",
                                                     nd[f"hybrid_appended_{name}@{name}"], nd[f"hybrid_blended_{name}"])})
    return results, pd.DataFrame(tests), chosen, len(test_users)


if __name__ == "__main__":
    FOLDER.mkdir(parents=True, exist_ok=True)
    movies = bmt.load_movies()
    texts = extra_only_texts(movies)
    extra = extra_only_tables(movies, texts)
    base = bmt.embedding_similarity("movies_base_embeddinggemma_v2", movies)
    appended = {"labels_e4b": bmt.embedding_similarity("movies_enriched_embeddinggemma_v2", movies)}
    for name in EXTRAS[1:]:
        appended[name] = bmt.embedding_similarity(f"movies_profile_{name}_embeddinggemma", movies, folder=PROFILE_FOLDER)

    # How different are the extra-only similarities from the base similarities?
    upper = np.triu_indices(len(movies), 1)
    for name in EXTRAS:
        r = np.corrcoef(base[upper], extra[name][upper])[0, 1]
        print(f"{name}: correlation of extra-only vs base similarities {r:.3f} "
              f"(appended vs base {np.corrcoef(base[upper], appended[name][upper])[0, 1]:.3f})")

    ratings = bmt.prepare_ratings(movies)
    w_results, w_pop, w_tests, w_curve, w_chosen, n_warm = warm(movies, ratings, base, appended, extra)
    c_results, c_tests, c_chosen, n_cold = cold(movies, ratings, base, appended, extra)

    for df in (w_tests, c_tests):
        df["p_bonferroni"] = (df["p_value"] * (len(w_tests) + len(c_tests))).clip(upper=1.0)
    w_results.to_csv(FOLDER / "warm_per_user.csv", index=False)
    w_curve.to_csv(FOLDER / "warm_weight_validation.csv", index=False)
    w_tests.to_csv(FOLDER / "warm_significance_tests.csv", index=False)
    c_results.to_csv(FOLDER / "cold_per_user.csv", index=False)
    c_tests.to_csv(FOLDER / "cold_significance_tests.csv", index=False)
    with open(FOLDER / "settings.json", "w", encoding="utf-8") as f:
        json.dump({"weights": WEIGHTS, "extras": EXTRAS, "warm_test_users": n_warm, "cold_test_users": n_cold,
                   "warm_chosen_w": {f"{s}|{n}": w for (s, n), w in w_chosen.items()},
                   "cold_chosen_w_alpha": {f"{n}|{e}": list(v) for (n, e), v in c_chosen.items()}}, f, indent=2)

    pd.set_option("display.width", 220)
    print(f"\nWARM ({n_warm} test users), NDCG@10")
    print(w_results.groupby(["method", "scoring"]).ndcg.mean().unstack("scoring").round(4).to_string())
    print(f"Popularity: {w_pop.ndcg.mean():.4f}")
    print(w_tests[["scoring", "comparison", "mean_difference", "p_value", "p_bonferroni",
                   "users_better", "users_worse"]].round(4).to_string(index=False))
    print(f"\nCOLD ({n_cold} test users), NDCG@10")
    print(c_results.groupby(["method", "n"]).ndcg.mean().unstack("n").round(4).to_string())
    print(c_tests[["n", "comparison", "mean_difference", "p_value", "p_bonferroni",
                   "users_better", "users_worse"]].round(4).to_string(index=False))
    print(f"\nSaved results to {FOLDER}")
