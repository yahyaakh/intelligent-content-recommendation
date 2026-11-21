'''
EASE: choose lambda on the simulated warm users (doc 26, section 6). The validation and test users are
never touched here.

Set-up (the same as LTR v2's training users, doc 16):
  - the 9,591 warm users are split into halves A and B (warm_halves, seed 42)
  - each warm user becomes a simulated new user: first 10 liked films = seed pool,
    the other liked films = answer key; 3 or 10 of the seeds are given to the model,
    all 10 are never recommended
  - cross-fitting: half A's users are scored by an EASE fitted on half B's ratings and the other way
    round, so no user's own ratings are inside the B that scores them
  - metric: NDCG@10 (bmt.user_metrics, the same function as every other experiment)

Usage:
    python src/ease_tuning.py pilot     one lambda, 300 users per half: checks that everything runs
    python src/ease_tuning.py           the full grid; writes output/experiments/ease/tuning.csv
'''
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

MODE = sys.argv[1] if len(sys.argv) > 1 else "full"
sys.argv = [sys.argv[0]]                 # the imported experiment scripts read sys.argv when imported
import build_movie_texts as bmt
import cold_start_experiment as cs
import LightGBM_XBoost_rerank_experiment as ltr
from recomm import EASE_R_Recommende

FOLDER = Path("output") / "experiments" / "ease"
# Grid history: the paper reports ~500 for ML-20M. A first full run over 500-20,000 still improved at
# 20,000 (3 seeds clearly, 10 seeds flattening), so the grid was extended upwards. (A 300-user pilot
# had suggested a peak near 5,000-10,000; the full set of users did not confirm it.)
LAMBDAS = [500, 1000, 2000, 3000, 5000, 7000, 10000, 15000, 20000, 30000, 50000, 80000, 120000, 200000]
SEED_COUNTS = (10, 3)
K = bmt.K


def minimal_data():
    """Only what warm_halves() needs, built exactly as in llm_rerank_experiment.load_data()."""
    movies = bmt.load_movies()
    ratings = bmt.prepare_ratings(movies)
    train, _ = cs.split_users(ratings)
    movie_ids = movies["movie_id"].to_numpy()
    position_of = {movie_id: i for i, movie_id in enumerate(movie_ids)}
    pool = sorted(position_of[m] for m in set(train["movie_id"]))
    return {"movies": movies, "movie_ids": movie_ids, "position_of": position_of, "pool": pool}


def fit_on(ratings, lambda_):
    """EASE fitted on one half's ratings, with the duplicate check from build_X."""
    model = EASE_R_Recommende(ratings, pd.DataFrame(), lambda_)     # movies_df is not used by fit
    X = model.build_X(ratings)
    if X.max() > 1:
        raise ValueError(f"X has values above 1 ({X.max():.0f}): duplicate ratings in the training half")
    model.fit(ratings)
    return model


def score_users(model, users):
    """Mean NDCG@10 (and per-user rows) of one model on a list of simulated users."""
    rows = []
    for user in users:
        recommended = [int(m) for m in model.recommend(user["liked"], K, exclude=user["seen"])]
        rows.append({"user_id": user["user_id"], **bmt.user_metrics(recommended, user["relevant"], K)})
    return rows


def run(pilot=False):
    FOLDER.mkdir(parents=True, exist_ok=True)
    data = minimal_data()

    # Simulated users per seed count; the same users and answer keys for 3 and 10 seeds
    halves = {}
    for n in SEED_COUNTS:
        train, in_A, users_by_half = ltr.warm_halves(data, n)
        halves[n] = users_by_half
    ratings_of = {"A": train[in_A], "B": train[~in_A]}
    other = {"A": "B", "B": "A"}

    if pilot:
        lambdas = [500]
        halves = {n: {h: u[:300] for h, u in by_half.items()} for n, by_half in halves.items()}
    else:
        lambdas = LAMBDAS

    for n in SEED_COUNTS:
        print(f"{n} seeds: {len(halves[n]['A'])} users in half A, {len(halves[n]['B'])} in half B")

    # Reference: item-CF (doc 06) on the same users, cross-fitted the same way, same metric code
    rows = []
    pool_positions = np.array(data["pool"])
    for half in ("A", "B"):
        item_cf = cs.item_cf_similarity(ratings_of[other[half]], data["movies"], data["position_of"])
        for n in SEED_COUNTS:
            for user in halves[n][half]:
                recommended = bmt.recommend(item_cf, user, pool_positions, data["position_of"],
                                            data["movie_ids"], K, "mean")
                rows.append({"method": "Item-CF", "lambda": np.nan, "seeds": n, "half": half,
                             "user_id": user["user_id"], **bmt.user_metrics(recommended, user["relevant"], K)})
    reference = pd.DataFrame(rows).groupby("seeds")["ndcg"].mean()
    print(f"Item-CF reference: NDCG@10 10 seeds {reference[10]:.4f} | 3 seeds {reference[3]:.4f}")

    for lambda_ in lambdas:
        start = time.time()
        for half in ("A", "B"):
            model = fit_on(ratings_of[other[half]], lambda_)          # fitted WITHOUT this half's users
            for n in SEED_COUNTS:
                for row in score_users(model, halves[n][half]):
                    rows.append({"method": "EASE", "lambda": lambda_, "seeds": n, "half": half, **row})
        done = pd.DataFrame([r for r in rows if r["lambda"] == lambda_])
        means = done.groupby("seeds")["ndcg"].mean()
        print(f"lambda {lambda_:>5}: NDCG@10 10 seeds {means[10]:.4f} | 3 seeds {means[3]:.4f} "
              f"({time.time() - start:.0f} s)")

    per_user = pd.DataFrame(rows)
    keys = ["method", "seeds", "lambda"]
    summary = (per_user.groupby(keys, dropna=False)[["ndcg", "precision", "recall"]].mean().reset_index())
    by_half = per_user.groupby(keys + ["half"], dropna=False)["ndcg"].mean().unstack("half")
    summary = summary.merge(by_half.add_prefix("ndcg_half_").reset_index(), on=keys)
    summary = summary.sort_values(["seeds", "method", "lambda"], ascending=[False, True, True])

    print("\nSimulated warm users, NDCG@10 (EASE by lambda, item-CF as reference):")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    for n in SEED_COUNTS:
        part = summary[(summary["seeds"] == n) & (summary["method"] == "EASE")]
        best = part.loc[part["ndcg"].idxmax()]
        edge = " (at the edge of the grid: extend it)" if best["lambda"] in (min(lambdas), max(lambdas)) \
            and len(lambdas) > 1 else ""
        print(f"best lambda, {n} seeds: {best['lambda']:.0f} (NDCG@10 {best['ndcg']:.4f}, "
              f"item-CF {reference[n]:.4f}){edge}")

    if not pilot:
        summary.to_csv(FOLDER / "tuning.csv", index=False)
        per_user.to_csv(FOLDER / "tuning_per_user.csv", index=False)
        print(f"\nSaved {FOLDER / 'tuning.csv'} and tuning_per_user.csv")


if __name__ == "__main__":
    run(pilot=(MODE == "pilot"))
