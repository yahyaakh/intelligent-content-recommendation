'''
New-item test for semantic IDs (doc 31; pre-registration in output/experiments/new_item_test/preregistration.md,
written before the first run).

10% of the pool films (rng 42, within popularity deciles) become "new": their ratings are removed from
everything built from ratings (SVD, EASE, the S7 ridge map, popularity, item-CF). Users are unchanged.

  task N (main)  rank only the new films; relevant = answer key ∩ new films; NDCG@10, random ties
  task F         the full pool; NDCG@10, new-film recall@10, new-film share of the top 10; popularity ties

Methods: IDs from S2 content / S3 SVD / S4 EASE / S7 hybrid (as in doc 30, 5 k-means seeds, cross-fitted),
and direct references: content embedding, S7 vectors, popularity, item-CF, random.
Decision (task N, 10 seeds): S7 better than S2 if higher mean and >= 4 of 5 k-means seeds.
Validation and test users are not used.

Usage:
    python experiments/semantic_ids/new_item_test.py pilot     300 users per half, k-means seed 42 only, nothing saved but new_films.csv
    python experiments/semantic_ids/new_item_test.py           full run (finished runs in runs/ are skipped)
    python experiments/semantic_ids/new_item_test.py metrics   rerun into runs_metrics/ with precision@10 and recall@10 added;
                                      checks that every user's NDCG equals runs/
'''

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))  # shared modules live in src/
import sys
import time
from pathlib import Path

MODE = sys.argv[1] if len(sys.argv) > 1 else "full"
sys.argv = [sys.argv[0]]                 # the imported experiment scripts read sys.argv when imported
import numpy as np
import pandas as pd

import build_movie_texts as bmt
import cold_start_experiment as cs
import LightGBM_XBoost_rerank_experiment as ltr
import ease_tuning as et
import sid_vector_sources as svs
from recomm import Semantic_IDs

OUT = Path("output") / "experiments" / "new_item_test"
ID_SOURCES = ["S2 content", "S3 SVD", "S4 EASE", "S7 hybrid"]
KMEANS_SEEDS = [42, 43, 44, 45, 46]
SEED_COUNTS = (10, 3)
NEW_SHARE = 0.10
K = bmt.K
OTHER = {"A": "B", "B": "A"}
RUNS = OUT / ("runs_metrics" if MODE == "metrics" else "runs")


# ───────────────────────── set-up ─────────────────────────
def choose_new_films(train, pool_ids):
    """10% of the pool, at random (rng 42) within deciles of the number of warm users who liked the film."""
    path = OUT / "new_films.csv"
    user_mean = train.groupby("user_id")["rating"].transform("mean")
    likes = train[train["rating"] >= user_mean]["movie_id"].value_counts()
    films = pd.DataFrame({"movie_id": sorted(pool_ids)})
    films["likes"] = films["movie_id"].map(likes).fillna(0).astype(int)
    films["decile"] = pd.qcut(films["likes"].rank(method="first"), 10, labels=False)
    rng = np.random.default_rng(42)
    chosen = []
    for _, group in films.groupby("decile"):
        chosen += list(rng.choice(group["movie_id"].to_numpy(), size=round(NEW_SHARE * len(group)), replace=False))
    films["new"] = films["movie_id"].isin(chosen)
    films.to_csv(path, index=False)
    return set(int(m) for m in chosen), films


def similarity_from(vectors, ids, movie_ids):
    """Cosine similarity over the whole catalog (rows = data["movie_ids"] order); films without a vector: 0."""
    row = {int(m): i for i, m in enumerate(ids)}
    full = np.zeros((len(movie_ids), vectors.shape[1]), np.float32)
    for p, m in enumerate(movie_ids):
        if int(m) in row:
            full[p] = vectors[row[int(m)]]
    norms = np.linalg.norm(full, axis=1, keepdims=True)
    full = full / np.maximum(norms, 1e-12)
    sim = full @ full.T
    np.fill_diagonal(sim, 0)
    return sim


def metrics(recommended, user, new_ids):
    recommended = [int(m) for m in recommended]
    relevant_new = set(user["relevant"]) & new_ids
    return {**bmt.user_metrics(recommended, set(user["relevant"]), K),
            "new_recall": len(set(recommended[:K]) & relevant_new) / len(relevant_new) if relevant_new else np.nan,
            "new_share": np.mean([m in new_ids for m in recommended[:K]]) if recommended else 0.0}


def task_n_users(users, new_ids):
    return [u for u in users if set(u["relevant"]) & new_ids]


def new_only(user, new_ids):
    """The user restricted to task N: answer key = relevant new films."""
    return {**user, "relevant": [m for m in user["relevant"] if m in new_ids]}


# ───────────────────────── one ID model ─────────────────────────
def score_ids(model, halves, half, pool_ids, new_ids, popularity, method, kmeans_seed):
    have = set(int(m) for m in model.movie_ids)
    pop_tie = popularity.reindex(model.movie_ids).fillna(0).to_numpy()
    rand_tie = np.random.default_rng(0).random(len(model.movie_ids))
    rows = []
    for n in SEED_COUNTS:
        users = halves[n][half]
        for user in users:                                                  # task F
            rec = model.recommend(user["liked"], K, exclude=user["seen"], allowed=pool_ids & have,
                                  weights=[1, 1, 1], tie_break=pop_tie)
            rows.append({"task": "F", "method": method, "kmeans_seed": kmeans_seed, "seeds": n, "half": half,
                         "user_id": user["user_id"], **metrics(rec, user, new_ids)})
        allowed_new = new_ids & have
        if allowed_new:                                                     # task N (S3/S4: no new film has an ID)
            for user in task_n_users(users, new_ids):
                u = new_only(user, new_ids)
                rec = model.recommend(u["liked"], K, exclude=u["seen"], allowed=allowed_new,
                                      weights=[1, 1, 1], tie_break=rand_tie)
                rows.append({"task": "N", "method": method, "kmeans_seed": kmeans_seed, "seeds": n, "half": half,
                             "user_id": user["user_id"], **bmt.user_metrics([int(m) for m in rec], set(u["relevant"]), K)})
    return rows


def id_runs(movies, halves, parts, vectors, pool_ids, new_ids, popular, pilot):
    for kmeans_seed in ([42] if pilot else KMEANS_SEEDS):
        for source in ID_SOURCES:
            path = RUNS / f"ids_{source.split()[0]}_s{kmeans_seed}.csv"
            if path.exists() and not pilot:
                continue
            start, rows = time.time(), []
            content_model = None
            for half in ("A", "B"):
                if source == "S2 content":                                  # uses no ratings: one model
                    if content_model is None:
                        content_model = Semantic_IDs(movies, n_levels=svs.N_LEVELS, codebook_size=svs.CODES,
                                                     seed=kmeans_seed)
                        content_model.load_embeddings(name=svs.CONTENT_EMBEDDING)
                        content_model.fit()
                    model = content_model
                else:
                    vec, ids = vectors[source][OTHER[half]]
                    model = svs.fit_ids(movies, vec, ids, source, OTHER[half], kmeans_seed, save=False)
                rows += score_ids(model, halves, half, pool_ids, new_ids, popular[half], source, kmeans_seed)
            frame = pd.DataFrame(rows)
            msg = ", ".join(f"{t}{n} {frame[(frame.task == t) & (frame.seeds == n)].ndcg.mean():.4f}"
                            for t in ("N", "F") for n in SEED_COUNTS if len(frame[(frame.task == t) & (frame.seeds == n)]))
            print(f"{source} | k-means seed {kmeans_seed}: {msg} ({time.time() - start:.0f} s)", flush=True)
            if not pilot:
                frame.to_csv(path, index=False)


# ───────────────────────── direct references ─────────────────────────
def references(data, halves, parts, vectors, pool_ids, new_ids, popular, pilot):
    path = RUNS / "references.csv"
    if path.exists() and not pilot:
        return
    movies, movie_ids, position_of = data["movies"], data["movie_ids"], data["position_of"]
    old_positions = np.array([position_of[m] for m in sorted(pool_ids - new_ids)])
    all_positions = np.array([position_of[m] for m in sorted(pool_ids)])
    new_positions = np.array([position_of[m] for m in sorted(new_ids)])
    content = bmt.embedding_similarity(svs.CONTENT_EMBEDDING, movies)
    rows = []
    for half in ("A", "B"):
        item_cf = cs.item_cf_similarity(parts[OTHER[half]], movies, position_of)
        s7 = similarity_from(*vectors["S7 hybrid"][OTHER[half]], movie_ids)
        for n in SEED_COUNTS:
            for user in halves[n][half]:
                for method, rec in (
                        ("Popularity", cs.recommend_popular(popular[half].index, user, K)),
                        ("Item-CF (mean)", bmt.recommend(item_cf, user, old_positions, position_of, movie_ids, K, "mean")),
                        ("Content embedding (max)", bmt.recommend(content, user, all_positions, position_of, movie_ids, K, "max")),
                        ("S7 vectors (max)", bmt.recommend(s7, user, all_positions, position_of, movie_ids, K, "max"))):
                    rows.append({"task": "F", "method": method, "seeds": n, "half": half, "user_id": user["user_id"],
                                 **metrics(rec, user, new_ids)})
            rng = np.random.default_rng(0)
            for user in task_n_users(halves[n][half], new_ids):
                u = new_only(user, new_ids)
                candidates = [int(m) for m in sorted(new_ids) if m not in set(u["seen"])]
                for method, rec in (
                        ("Random", list(rng.permutation(candidates)[:K])),
                        ("Content embedding (max)", bmt.recommend(content, u, new_positions, position_of, movie_ids, K, "max")),
                        ("S7 vectors (max)", bmt.recommend(s7, u, new_positions, position_of, movie_ids, K, "max"))):
                    rows.append({"task": "N", "method": method, "seeds": n, "half": half, "user_id": user["user_id"],
                                 **bmt.user_metrics([int(m) for m in rec], set(u["relevant"]), K)})
        print(f"references: half {half} done", flush=True)
    frame = pd.DataFrame(rows)
    print(frame.groupby(["task", "seeds", "method"])[["ndcg", "new_recall", "new_share"]].mean().round(4).to_string())
    if not pilot:
        frame.to_csv(path, index=False)


# ───────────────────────── summary ─────────────────────────
def summarise():
    runs = OUT / "runs"
    ids = pd.concat([pd.read_csv(p) for p in sorted(runs.glob("ids_*.csv"))], ignore_index=True)
    refs = pd.read_csv(runs / "references.csv").assign(kmeans_seed=np.nan)
    per_run = (pd.concat([ids, refs], ignore_index=True)
               .groupby(["task", "seeds", "method", "kmeans_seed"], dropna=False)[["ndcg", "new_recall", "new_share"]]
               .mean().reset_index())
    per_run.to_csv(OUT / "results.csv", index=False)
    summary = (per_run.groupby(["task", "seeds", "method"])[["ndcg", "new_recall", "new_share"]]
               .agg(["mean", "std", "count"]).round(4))
    summary.to_csv(OUT / "summary.csv")
    pd.set_option("display.width", 250)
    print("\nMean over k-means seeds (references: one run):")
    print(summary.to_string())

    main = per_run[(per_run.task == "N") & (per_run.seeds == 10)].pivot(index="kmeans_seed", columns="method",
                                                                         values="ndcg")
    diff = (main["S7 hybrid"] - main["S2 content"]).dropna()
    better = len(diff) == 5 and diff.mean() > 0 and (diff > 0).sum() >= 4
    print("\nDecision rule (task N, 10 seeds): S7 - S2 per k-means seed")
    print(diff.round(4).to_string())
    print(f"mean {diff.mean():+.4f}, S7 wins {(diff > 0).sum()} of {len(diff)} -> "
          f"{'S7 BETTER on new films' if better else 'S7 not better on new films'}")
    print(f"\nSaved to {OUT}")


def summarise_metrics():
    """Precision@10 and recall@10 next to NDCG@10; first checks that NDCG is unchanged per user."""
    keys = ["task", "method", "seeds", "half", "user_id"]
    for path in sorted(RUNS.glob("*.csv")):
        new, old = pd.read_csv(path), pd.read_csv(OUT / "runs" / path.name)
        k = keys + (["kmeans_seed"] if "kmeans_seed" in old else [])
        merged = old.merge(new, on=k, suffixes=("_old", ""))
        assert len(merged) == len(old) == len(new), f"{path.name}: users differ"
        assert np.allclose(merged["ndcg_old"], merged["ndcg"], atol=1e-12), f"{path.name}: NDCG changed"
    print("check passed: every user's NDCG@10 equals the original runs")
    ids = pd.concat([pd.read_csv(p) for p in sorted(RUNS.glob("ids_*.csv"))], ignore_index=True)
    refs = pd.read_csv(RUNS / "references.csv").assign(kmeans_seed=np.nan)
    rows = pd.concat([ids, refs], ignore_index=True)
    rows["hit"] = (rows["precision"] > 0).astype(float)
    cols = ["ndcg", "precision", "recall", "hit", "new_recall", "new_share"]
    per_run = rows.groupby(["task", "seeds", "method", "kmeans_seed"], dropna=False)[cols].mean().reset_index()
    summary = per_run.groupby(["task", "seeds", "method"])[cols].agg(["mean", "std"]).round(4)
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
    if pilot:
        halves = {n: {h: u[:300] for h, u in by.items()} for n, by in halves.items()}

    new_ids, films = choose_new_films(train, pool_ids)
    print(f"New films: {len(new_ids)} of {len(pool_ids)} pool films; "
          f"median likes {films[films.new]['likes'].median():.0f} (new) vs {films[~films.new]['likes'].median():.0f} (rest)")
    parts = {"A": train[in_A & ~train["movie_id"].isin(new_ids)], "B": train[~in_A & ~train["movie_id"].isin(new_ids)]}
    popular = {h: cs.popularity_scores(parts[OTHER[h]]) for h in ("A", "B")}
    for n in SEED_COUNTS:
        n_users = sum(len(task_n_users(halves[n][h], new_ids)) for h in ("A", "B"))
        print(f"{n} seeds: {sum(len(u) for u in halves[n].values())} users, {n_users} with >= 1 relevant new film")

    c_vec, c_ids = svs.content_vectors(movies)
    vectors = {"S3 SVD": {h: svs.svd_vectors(parts[h]) for h in ("A", "B")},
               "S4 EASE": {h: svs.ease_vectors(parts[h]) for h in ("A", "B")}}
    vectors["S7 hybrid"] = {h: svs.hybrid_vectors((c_vec, c_ids), vectors["S3 SVD"][h]) for h in ("A", "B")}
    for source, by_half in vectors.items():
        for h in ("A", "B"):
            ids = set(int(m) for m in by_half[h][1])
            assert source == "S7 hybrid" or not (ids & new_ids), f"{source} has a vector for a new film"
            print(f"{source} from half {h}: {len(ids & pool_ids)} pool films, {len(ids & new_ids)} new films")

    id_runs(movies, halves, parts, vectors, pool_ids, new_ids, popular, pilot)
    references(data, halves, parts, vectors, pool_ids, new_ids, popular, pilot)
    if MODE == "metrics":
        summarise_metrics()
    elif not pilot:
        summarise()


if __name__ == "__main__":
    main()
