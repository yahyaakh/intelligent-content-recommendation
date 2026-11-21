'''
Semantic IDs from different VECTOR SOURCES (doc 30; pre-registration in
output/experiments/sid_vector_sources/preregistration.md, written before the first run).

Stage 1 (screening): S2 content (embeddinggemma + LLM labels, the IDs of docs 27-28) vs S3 SVD
(film vectors U*s, rank 64, of the user-centred warm rating matrix; the SVD of
LightGBM_XBoost_rerank_experiment.latent_tables).

  test A  meaningfulness: genre / director / decade lift + co-liked lift, IDs fitted on all warm users,
          both sources on the same pool films. Descriptive.
  test B  use 1 (shared-prefix recommendation) on the 9,084 simulated warm users, 10 and 3 seeds.
          S3 is cross-fitted: users of half A get IDs built from half B's ratings and vice versa.
          Ties: popularity of the other half (main) and random (rng 0).
          References on the same users: popularity, item-CF, content embedding, SVD vectors directly.

Stage 2b (pre-registration addition of 2026-10-29): two more sources, screened the same way
  S4 EASE    column j of the EASE weight matrix B (lambda 20,000), cross-fitted like S3
  S7 hybrid  [content ; SVD] blocks of length 1, / sqrt(2); films without an SVD vector get their SVD
             block predicted from content (ridge, alpha 1), so S7 covers every pool film
Decision: a new source replaces S3 if it has the higher mean (10 seeds, popularity ties) and wins >= 4 of 5.

Check first: S2, k-means seed 42, popularity ties reproduces doc 27 sec. 4a (0.4058 / 0.3614).
Validation and test users are not used.

Saving: every finished run is one file in runs/ and is skipped when the script is restarted;
results.csv and summary.csv are rebuilt from runs/ at the end.

Usage:
    python src/sid_vector_sources.py pilot     300 users per half, k-means seed 42 only, nothing saved
    python src/sid_vector_sources.py           stage 1 screening
    python src/sid_vector_sources.py metrics   rerun test B + references into runs_metrics/ with precision@10 and
                                           recall@10 added; checks that every user's NDCG equals runs/
'''
import sys
import time
from pathlib import Path

MODE = sys.argv[1] if len(sys.argv) > 1 else "screen"
sys.argv = [sys.argv[0]]                 # the imported experiment scripts read sys.argv when imported
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

import build_movie_texts as bmt
import cold_start_experiment as cs
import LightGBM_XBoost_rerank_experiment as ltr
import ease_tuning as et
from recomm import Semantic_IDs

OUT = Path("output") / "experiments" / "sid_vector_sources"
CONTENT_EMBEDDING = "movies_enriched_embeddinggemma_v2"
SOURCES = ["S2 content", "S3 SVD", "S4 EASE", "S7 hybrid"]
EASE_LAMBDA = 20000                                  # half-size value tuned for 10 seeds (ease_tuning.py)
RIDGE_ALPHA = 1.0
N_LEVELS, CODES = 3, 32
KMEANS_SEEDS = [42, 43, 44, 45, 46]
SEED_COUNTS = (10, 3)
SVD_RANK = ltr.SVD_RANK                                  # 64, as in LTR v2
K = bmt.K
EXPECTED_S2 = {10: 0.40581436961289197, 3: 0.36138562419477266}   # doc 27 sec. 4a, seed 42, popularity ties
OTHER = {"A": "B", "B": "A"}
RUNS = OUT / ("runs_metrics" if MODE == "metrics" else "runs")


# ───────────────────────── vectors ─────────────────────────
def svd_vectors(part):
    """Film vectors U*s (rank 64) of the user-centred rating matrix of `part`, for the films rated in `part`.
    Same computation as ltr.latent_tables (eigh of M M^T, exact and deterministic)."""
    films = np.sort(part["movie_id"].unique())
    row_of = {m: i for i, m in enumerate(films)}
    rows = part["movie_id"].map(row_of).to_numpy()
    cols, users = pd.factorize(part["user_id"])
    user_mean = part.groupby("user_id")["rating"].transform("mean")
    centred = csr_matrix(((part["rating"] - user_mean).to_numpy(np.float64), (rows, cols)),
                         shape=(len(films), len(users)))
    eigenvalues, eigenvectors = np.linalg.eigh((centred @ centred.T).toarray())
    top = np.argsort(eigenvalues)[::-1][:SVD_RANK]
    vectors = eigenvectors[:, top] * np.sqrt(np.maximum(eigenvalues[top], 0.0))
    keep = np.linalg.norm(vectors, axis=1) > 1e-9      # a film whose ratings all equal the raters' means
    if (~keep).any():
        print(f"  SVD: dropped {(~keep).sum()} films with an all-zero vector")
    return vectors[keep], films[keep]


def content_vectors(movies):
    model = Semantic_IDs(movies, n_levels=N_LEVELS, codebook_size=CODES, seed=42)
    model.load_embeddings(name=CONTENT_EMBEDDING)
    return model.vectors, model.movie_ids


def ease_vectors(part):
    """Column j of the EASE weight matrix B fitted on `part`: how strongly every film predicts film j."""
    model = et.fit_on(part, EASE_LAMBDA)
    vectors = np.ascontiguousarray(model.B.T)                  # row j = column j of B
    films = np.asarray(model.col_to_film)
    keep = np.linalg.norm(vectors, axis=1) > 1e-12
    if (~keep).any():
        print(f"  EASE: dropped {(~keep).sum()} films with an all-zero column")
    return vectors[keep], films[keep]


def unit(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def hybrid_vectors(content, svd):
    """[content ; svd] / sqrt(2), each block of length 1. Films without an SVD vector: SVD block
    predicted from the content vector by ridge regression fitted on the films that have both."""
    c_vec, c_ids = content
    s_vec, s_ids = svd
    C = unit(c_vec.astype(np.float64))
    s_row = {m: i for i, m in enumerate(s_ids)}
    has = np.array([m in s_row for m in c_ids])
    S_known = unit(s_vec[[s_row[m] for m in c_ids[has]]].astype(np.float64))
    X = C[has]
    W = np.linalg.solve(X.T @ X + RIDGE_ALPHA * np.eye(X.shape[1]), X.T @ S_known)
    S = np.zeros((len(c_ids), S_known.shape[1]))
    S[has] = S_known
    if (~has).any():
        S[~has] = unit(C[~has] @ W)
        print(f"  S7: SVD block predicted from content for {(~has).sum()} films")
    return (np.hstack([C, S]) / np.sqrt(2)).astype(np.float32), np.asarray(c_ids)


def cached_vectors(source, fit_on, build):
    """vectors/{source}_{fit_on}.npy (+ _ids.npy): built once, then loaded."""
    folder = OUT / "vectors"
    folder.mkdir(parents=True, exist_ok=True)
    stem = f"{source.split()[0]}_{fit_on}"
    v_path, i_path = folder / f"{stem}.npy", folder / f"{stem}_ids.npy"
    if v_path.exists() and i_path.exists():
        return np.load(v_path), np.load(i_path)
    vectors, ids = build()
    np.save(v_path, vectors)
    np.save(i_path, ids)
    return vectors, ids


def fit_ids(movies, vectors, ids, source, fit_on, kmeans_seed, save=True):
    """Semantic_IDs fitted on these vectors; codes saved to ids/ so later stages reuse the exact IDs."""
    model = Semantic_IDs(movies, n_levels=N_LEVELS, codebook_size=CODES, seed=kmeans_seed)
    model.set_vectors(vectors, ids)
    model.fit()
    if save:
        folder = OUT / "ids"
        folder.mkdir(parents=True, exist_ok=True)
        np.savez(folder / f"{source.split()[0]}_{fit_on}_k{CODES}_s{kmeans_seed}.npz",
                 codes=model.codes, movie_ids=model.movie_ids)
    return model


# ───────────────────────── test A: meaningfulness ─────────────────────────
def coliked_lift(model, covis, position_of):
    """Mean P(j liked | i liked) over pairs sharing the first L tokens / mean over all pairs (of these films)."""
    pos = np.array([position_of[m] for m in model.movie_ids])
    table = covis[np.ix_(pos, pos)]
    off = ~np.eye(len(pos), dtype=bool)
    base = table[off].mean()
    rows = []
    for L in range(1, model.n_levels + 1):
        key = pd.Series([tuple(c) for c in model.codes[:, :L]]).factorize()[0]
        same = (key[:, None] == key[None, :]) & off
        rows.append({"attribute": "co-liked", "prefix": L, "pairs": int(same.sum() // 2),
                     "match_rate": table[same].mean(), "random_rate": base, "lift": table[same].mean() / base})
    return pd.DataFrame(rows)


def test_a(movies, train, data, pool_ids):
    covis = ltr.latent_tables(train, data)["covis"]
    s3_vec, s3_ids = cached_vectors("S3 SVD", "all", lambda: svd_vectors(train))
    c_vec, c_ids = content_vectors(movies)
    s4_vec, s4_ids = cached_vectors("S4 EASE", "all", lambda: ease_vectors(train))
    s7_vec, s7_ids = cached_vectors("S7 hybrid", "all", lambda: hybrid_vectors((c_vec, c_ids), (s3_vec, s3_ids)))
    films = [m for m in s3_ids if m in set(c_ids)]                       # the same films for S2 and S3
    full = {"S2 content": (c_vec, c_ids), "S3 SVD": (s3_vec, s3_ids),
            "S4 EASE": (s4_vec, s4_ids), "S7 hybrid": (s7_vec, s7_ids)}
    inputs = {}
    for source, (vec, ids) in full.items():
        row = {m: i for i, m in enumerate(ids)}
        these = [m for m in films if m in row]                           # S4 may lack a few films
        inputs[source] = (vec[[row[m] for m in these]], np.array(these))
    for kmeans_seed in KMEANS_SEEDS:
        for source, (vectors, ids) in inputs.items():
            path = OUT / "runs" / f"meaning_{source.split()[0]}_k{CODES}_s{kmeans_seed}.csv"
            if path.exists():
                continue
            model = fit_ids(movies, vectors, ids, source, "all-pool", kmeans_seed)
            result = pd.concat([model.meaningfulness(), coliked_lift(model, covis, data["position_of"])])
            result.assign(source=source, kmeans_seed=kmeans_seed, films=len(ids)).to_csv(path, index=False)
        print(f"test A: k-means seed {kmeans_seed} done", flush=True)


# ───────────────────────── test B: use 1 ─────────────────────────
def score(model, users, n, half, pool_ids, popularity, method, kmeans_seed):
    allowed = pool_ids & set(int(m) for m in model.movie_ids)
    ties = {"popularity": popularity.reindex(model.movie_ids).fillna(0).to_numpy(),
            "random": np.random.default_rng(0).random(len(model.movie_ids))}
    rows = []
    for tie_name, tie in ties.items():
        for user in users:
            recommended = model.recommend(user["liked"], K, exclude=user["seen"], allowed=allowed,
                                          weights=[1, 1, 1], tie_break=tie)
            rows.append({"method": method, "kmeans_seed": kmeans_seed, "ties": tie_name, "seeds": n,
                         "half": half, "user_id": user["user_id"],
                         **bmt.user_metrics([int(m) for m in recommended], user["relevant"], K)})
    return rows


def test_b(movies, data, halves, ratings_of, pool_ids, pilot):
    popular = {h: cs.popularity_scores(ratings_of[OTHER[h]]) for h in ("A", "B")}
    c_vec, c_ids = content_vectors(movies)
    svd = {h: cached_vectors("S3 SVD", h, lambda h=h: svd_vectors(ratings_of[h])) for h in ("A", "B")}
    per_half = {"S3 SVD": svd,
                "S4 EASE": {h: cached_vectors("S4 EASE", h, lambda h=h: ease_vectors(ratings_of[h])) for h in ("A", "B")},
                "S7 hybrid": {h: cached_vectors("S7 hybrid", h, lambda h=h: hybrid_vectors((c_vec, c_ids), svd[h]))
                              for h in ("A", "B")}}
    for source, by_half in per_half.items():
        for h in ("A", "B"):
            print(f"{source} vectors from half {h}: {len(by_half[h][1])} films")

    for kmeans_seed in ([42] if pilot else KMEANS_SEEDS):
        for source in SOURCES:
            path = RUNS / f"use1_{source.split()[0]}_k{CODES}_s{kmeans_seed}.csv"
            if path.exists() and not pilot:
                continue
            start = time.time()
            rows = []
            for half in ("A", "B"):
                if source == "S2 content":                                  # no ratings: one model
                    model = Semantic_IDs(movies, n_levels=N_LEVELS, codebook_size=CODES, seed=kmeans_seed)
                    model.load_embeddings(name=CONTENT_EMBEDDING)
                    model.fit()
                else:                                                       # IDs from the OTHER half
                    vectors, ids = per_half[source][OTHER[half]]
                    model = fit_ids(movies, vectors, ids, source, OTHER[half], kmeans_seed, save=not pilot)
                for n in SEED_COUNTS:
                    rows += score(model, halves[n][half], n, half, pool_ids, popular[half], source, kmeans_seed)
            frame = pd.DataFrame(rows)
            print(f"use 1 | {source} | k-means seed {kmeans_seed}: "
                  + ", ".join(f"{n} seeds {frame[(frame.seeds == n) & (frame.ties == 'popularity')].ndcg.mean():.4f}"
                              for n in SEED_COUNTS) + f" ({time.time() - start:.0f} s)", flush=True)
            if not pilot:
                frame.to_csv(path, index=False)
            if source == "S2 content" and kmeans_seed == 42 and not pilot:
                for n in SEED_COUNTS:
                    got = frame[(frame.seeds == n) & (frame.ties == "popularity")].ndcg.mean()
                    assert abs(got - EXPECTED_S2[n]) < 1e-4, f"S2 does not reproduce doc 27 ({n} seeds: {got})"
                print("check passed: S2 reproduces doc 27 sec. 4a")


def references(movies, data, halves, ratings_of, pilot):
    path = RUNS / "use1_references.csv"
    if path.exists() and not pilot:
        return
    movie_ids, position_of = data["movie_ids"], data["position_of"]
    pool_positions = np.array(data["pool"])
    content = bmt.embedding_similarity(CONTENT_EMBEDDING, movies)
    rows = []
    for half in ("A", "B"):
        other = ratings_of[OTHER[half]]
        popular = cs.popularity_scores(other)
        item_cf = cs.item_cf_similarity(other, movies, position_of)
        svd_sim = ltr.latent_tables(other, data)["svd_sim"]
        for n in SEED_COUNTS:
            for user in halves[n][half]:
                for method, recommended in (
                        ("Popularity", cs.recommend_popular(popular.index, user, K)),
                        ("Item-CF (mean)", bmt.recommend(item_cf, user, pool_positions, position_of, movie_ids, K, "mean")),
                        ("Content embedding (max)", bmt.recommend(content, user, pool_positions, position_of, movie_ids, K, "max")),
                        ("SVD vectors (max)", bmt.recommend(svd_sim, user, pool_positions, position_of, movie_ids, K, "max"))):
                    rows.append({"method": method, "kmeans_seed": np.nan, "ties": "", "seeds": n, "half": half,
                                 "user_id": user["user_id"],
                                 **bmt.user_metrics([int(m) for m in recommended], user["relevant"], K)})
        print(f"references: half {half} done", flush=True)
    frame = pd.DataFrame(rows)
    print(frame.groupby(["seeds", "method"]).ndcg.mean().round(4).to_string())
    if not pilot:
        frame.to_csv(path, index=False)


# ───────────────────────── summary ─────────────────────────
def summarise():
    runs = OUT / "runs"
    use1 = pd.concat([pd.read_csv(p) for p in sorted(runs.glob("use1_*.csv"))], ignore_index=True)
    results = (use1.groupby(["method", "kmeans_seed", "ties", "seeds"], dropna=False)["ndcg"].mean()
               .reset_index().rename(columns={"ndcg": "value"}).assign(test="B use 1", metric="ndcg@10"))
    meaning = [pd.read_csv(p) for p in sorted(runs.glob("meaning_*.csv"))]
    if meaning:
        m = pd.concat(meaning, ignore_index=True)
        m = m.rename(columns={"source": "method"}).assign(test="A meaningfulness",
                                                          metric=lambda d: "lift " + d["attribute"] + " L" + d["prefix"].astype(str))
        results = pd.concat([results, m[["method", "kmeans_seed", "metric", "test"]].assign(value=m["lift"])],
                            ignore_index=True)
    results.to_csv(OUT / "results.csv", index=False)

    summary = (results.groupby(["test", "metric", "method", "ties", "seeds"], dropna=False)["value"]
               .agg(["mean", "std", "min", "max", "count"]).reset_index())
    summary.to_csv(OUT / "summary.csv", index=False)

    pd.set_option("display.width", 250)
    b = results[results.test == "B use 1"]
    print("\nTest B (use 1), NDCG@10, simulated warm users, mean over k-means seeds:")
    print(b.groupby(["seeds", "ties", "method"], dropna=False)["value"].agg(["mean", "std", "min", "max", "count"])
          .round(4).to_string())
    main = b[(b.seeds == 10) & (b.ties == "popularity")].pivot(index="kmeans_seed", columns="method", values="value")
    if {"S2 content", "S3 SVD"} <= set(main.columns):
        diff = main["S3 SVD"] - main["S2 content"]
        better = diff.mean() > 0 and (diff > 0).sum() >= 4
        print("\nDecision rule (10 seeds, popularity ties): S3 - S2 per k-means seed")
        print(diff.round(4).to_string())
        print(f"mean {diff.mean():+.4f}, S3 wins {(diff > 0).sum()} of {len(diff)} -> "
              f"{'S3 BETTER' if better else 'S3 not better'}")
    for new in ("S4 EASE", "S7 hybrid"):
        if {"S3 SVD", new} <= set(main.columns):
            diff = (main[new] - main["S3 SVD"]).dropna()
            better = len(diff) == 5 and diff.mean() > 0 and (diff > 0).sum() >= 4
            print(f"\nStage 2b rule: {new} - S3 per k-means seed: mean {diff.mean():+.4f}, "
                  f"{new} wins {(diff > 0).sum()} of {len(diff)} -> {'REPLACES S3' if better else 'does not replace S3'}")
    a = results[results.test == "A meaningfulness"]
    if len(a):
        print("\nTest A, lift (mean over k-means seeds):")
        print(a.pivot_table(index="metric", columns="method", values="value").round(2).to_string())
    print(f"\nSaved to {OUT}")


def summarise_metrics():
    """Precision@10 and recall@10 next to NDCG@10; first checks that NDCG is unchanged per user."""
    keys = ["method", "kmeans_seed", "ties", "seeds", "half", "user_id"]
    for path in sorted(RUNS.glob("use1_*.csv")):
        new = pd.read_csv(path, keep_default_na=False, na_values=[""])
        old = pd.read_csv(OUT / "runs" / path.name, keep_default_na=False, na_values=[""])
        merged = old.merge(new, on=keys, suffixes=("_old", ""))
        assert len(merged) == len(old) == len(new), f"{path.name}: users differ"
        assert np.allclose(merged["ndcg_old"], merged["ndcg"], atol=1e-12), f"{path.name}: NDCG changed"
    print("check passed: every user's NDCG@10 equals the original runs")
    use1 = pd.concat([pd.read_csv(p) for p in sorted(RUNS.glob("use1_*.csv"))], ignore_index=True)
    use1["hit"] = (use1["precision"] > 0).astype(float)                    # hit rate@10, for reference
    per_run = (use1.groupby(["seeds", "ties", "method", "kmeans_seed"], dropna=False)[["ndcg", "precision", "recall", "hit"]]
               .mean().reset_index())
    summary = (per_run.groupby(["seeds", "ties", "method"], dropna=False)[["ndcg", "precision", "recall", "hit"]]
               .agg(["mean", "std"]).round(4))
    per_run.to_csv(OUT / "metrics_per_run.csv", index=False)
    summary.to_csv(OUT / "metrics_summary.csv")
    pd.set_option("display.width", 250)
    print(summary.to_string())


def main():
    pilot = MODE == "pilot"
    RUNS.mkdir(parents=True, exist_ok=True)
    data = et.minimal_data()
    movies = data["movies"]
    pool_ids = set(int(data["movie_ids"][p]) for p in data["pool"])
    halves = {}
    for n in SEED_COUNTS:
        train, in_A, users_by_half = ltr.warm_halves(data, n)
        halves[n] = users_by_half
    ratings_of = {"A": train[in_A], "B": train[~in_A]}
    if pilot:
        halves = {n: {h: u[:300] for h, u in by.items()} for n, by in halves.items()}
    print({n: {h: len(u) for h, u in by.items()} for n, by in halves.items()}, "simulated users per half")

    if MODE == "screen":
        test_a(movies, train, data, pool_ids)
    test_b(movies, data, halves, ratings_of, pool_ids, pilot)
    references(movies, data, halves, ratings_of, pilot)
    if MODE == "metrics":
        summarise_metrics()
    elif not pilot:
        summarise()


if __name__ == "__main__":
    main()
