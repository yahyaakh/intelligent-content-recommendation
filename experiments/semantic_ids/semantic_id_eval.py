'''
Semantic IDs, use 1 (doc 18 section 3, step 3): recommend by shared ID prefix with the seeds.
Settings are chosen on the simulated warm users only; the validation and test users are never touched.

Set-up (the same users as ease_tuning.py and LTR v2):
  - the 9,084 simulated warm users of halves A and B (warm_halves, seed 42): first 10 liked films = seed
    pool, the other liked films = answer key; 3 or 10 seeds given, all 10 never recommended
  - candidates: the 1,549-film pool (films rated by warm users), as for every other method
  - the IDs use no ratings, so they need no cross-fitting; the popularity tie-break does: half A's users
    get the popularity (liked counts) of half B and the other way round
  - metric: NDCG@10 (bmt.user_metrics)

Compared settings (all crossed): embedding base / enriched, 16 / 32 codes per level (3 levels),
prefix weights flat (1, 1, 1) / deep (1, 2, 3). Ties broken by popularity in every setting.
References on the same users: the full content embeddings (max over the seeds, as "Content" in
final_comparison.py), item-CF (mean) and popularity.

Usage:
    python experiments/semantic_ids/semantic_id_eval.py pilot     300 users per half, one ID setting
    python experiments/semantic_ids/semantic_id_eval.py           everything; writes output/experiments/semantic_ids/
    python experiments/semantic_ids/semantic_id_eval.py seeds     enriched vs base IDs over 5 k-means seeds (32 codes, flat)
'''

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))  # shared modules live in src/
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
import ease_tuning as et                  # minimal_data(): only what warm_halves() needs
from recomm import Semantic_IDs

FOLDER = Path("output") / "experiments" / "semantic_ids"
EMBEDDINGS = {"base": "movies_base_embeddinggemma_v2", "enriched": "movies_enriched_embeddinggemma_v2"}
SIZES = [16, 32]
WEIGHTS = {"flat (1,1,1)": [1, 1, 1], "deep (1,2,3)": [1, 2, 3]}
SEED_COUNTS = (10, 3)
K = bmt.K


def run(pilot=False):
    FOLDER.mkdir(parents=True, exist_ok=True)
    data = et.minimal_data()
    movies, position_of, movie_ids = data["movies"], data["position_of"], data["movie_ids"]
    pool_positions = np.array(data["pool"])
    pool_ids = set(int(movie_ids[p]) for p in data["pool"])

    halves = {}
    for n in SEED_COUNTS:
        train, in_A, users_by_half = ltr.warm_halves(data, n)
        halves[n] = users_by_half
    ratings_of = {"A": train[in_A], "B": train[~in_A]}
    other = {"A": "B", "B": "A"}
    if pilot:
        halves = {n: {h: u[:300] for h, u in by_half.items()} for n, by_half in halves.items()}

    # popularity of the OTHER half: tie-break for the IDs, and the popularity reference
    popular = {half: cs.popularity_scores(ratings_of[other[half]]) for half in ("A", "B")}

    # ID models (no ratings used)
    id_models = {}
    for emb, name in EMBEDDINGS.items():
        for size in ([16] if pilot else SIZES):
            model = Semantic_IDs(movies, n_levels=3, codebook_size=size, seed=42)
            model.load_embeddings(name=name)
            model.fit()
            id_models[(emb, size)] = model
    weights = {"flat (1,1,1)": WEIGHTS["flat (1,1,1)"]} if pilot else WEIGHTS

    content = {emb: bmt.embedding_similarity(name, movies) for emb, name in EMBEDDINGS.items()}

    rows = []
    for half in ("A", "B"):
        start = time.time()
        item_cf = cs.item_cf_similarity(ratings_of[other[half]], movies, position_of)
        tie = {key: popular[half].reindex(model.movie_ids).fillna(0).to_numpy()
               for key, model in id_models.items()}
        for n in SEED_COUNTS:
            for user in halves[n][half]:
                base = {"seeds": n, "half": half, "user_id": user["user_id"]}

                def add(method, recommended):
                    rows.append({**base, "method": method,
                                 **bmt.user_metrics([int(m) for m in recommended], user["relevant"], K)})

                add("Popularity", cs.recommend_popular(popular[half].index, user, K))
                add("Item-CF", bmt.recommend(item_cf, user, pool_positions, position_of, movie_ids, K, "mean"))
                for emb, table in content.items():
                    add(f"Content embedding ({emb}, max)",
                        bmt.recommend(table, user, pool_positions, position_of, movie_ids, K, "max"))
                for (emb, size), model in id_models.items():
                    for w_name, w in weights.items():
                        add(f"Semantic IDs ({emb}, {size} codes, {w_name})",
                            model.recommend(user["liked"], K, exclude=user["seen"], allowed=pool_ids,
                                            weights=w, tie_break=tie[(emb, size)]))
        print(f"half {half} scored ({time.time() - start:.0f} s)")

    per_user = pd.DataFrame(rows)
    summary = (per_user.groupby(["seeds", "method"])[["ndcg", "precision", "recall"]].mean().reset_index())
    by_half = per_user.groupby(["seeds", "method", "half"])["ndcg"].mean().unstack("half")
    summary = summary.merge(by_half.add_prefix("ndcg_half_").reset_index(), on=["seeds", "method"])
    summary = summary.sort_values(["seeds", "ndcg"], ascending=[False, False])

    pd.set_option("display.width", 250)
    for n in SEED_COUNTS:
        part = summary[summary["seeds"] == n]
        users = per_user[per_user["seeds"] == n]["user_id"].nunique()
        print(f"\nSimulated warm users, {n} seeds ({users} users), NDCG@10:")
        print(part.drop(columns="seeds").to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    if not pilot:
        summary.to_csv(FOLDER / "use1_simulated.csv", index=False)
        per_user.to_csv(FOLDER / "use1_simulated_per_user.csv", index=False)
        print(f"\nSaved to {FOLDER}")


KMEANS_SEEDS = [42, 43, 44, 45, 46]


def seed_check():
    """Is 'enriched IDs > base IDs' stable across k-means seeds? 32 codes, flat weights (the setting chosen
    above), ties broken by popularity (other half) and at random (fixed rng 0). Same users for every seed,
    so the enriched - base difference is paired."""
    data = et.minimal_data()
    movies = data["movies"]
    pool_ids = set(int(data["movie_ids"][p]) for p in data["pool"])
    rows = []
    for n in SEED_COUNTS:
        train, in_A, users_by_half = ltr.warm_halves(data, n)
        ratings_of = {"A": train[in_A], "B": train[~in_A]}
        for kmeans_seed in KMEANS_SEEDS:
            for emb, name in EMBEDDINGS.items():
                model = Semantic_IDs(movies, n_levels=3, codebook_size=32, seed=kmeans_seed)
                model.load_embeddings(name=name)
                model.fit()
                for half, other in (("A", "B"), ("B", "A")):
                    ties = {"popularity": cs.popularity_scores(ratings_of[other]).reindex(model.movie_ids)
                            .fillna(0).to_numpy(),
                            "random": np.random.default_rng(0).random(len(model.movie_ids))}
                    for tie_name, tie in ties.items():
                        for user in users_by_half[half]:
                            recommended = model.recommend(user["liked"], K, exclude=user["seen"], allowed=pool_ids,
                                                          weights=[1, 1, 1], tie_break=tie)
                            rows.append({"seeds": n, "kmeans_seed": kmeans_seed, "embedding": emb, "ties": tie_name,
                                         "user_id": user["user_id"],
                                         "ndcg": bmt.ndcg_at_k([int(m) for m in recommended], user["relevant"], K)})
            print(f"{n} seeds, k-means seed {kmeans_seed} done", flush=True)

    per_user = pd.DataFrame(rows)
    means = per_user.groupby(["seeds", "ties", "kmeans_seed", "embedding"])["ndcg"].mean().unstack("embedding")
    means["enriched - base"] = means["enriched"] - means["base"]
    print("\nNDCG@10 per k-means seed (32 codes, flat weights, simulated warm users):")
    print(means.round(4).to_string())

    summary = means.groupby(["seeds", "ties"]).agg(["mean", "std", "min", "max"])
    print("\nAcross the 5 k-means seeds:")
    print(summary.round(4).to_string())
    wins = (means["enriched - base"] > 0).groupby(["seeds", "ties"]).sum()
    print("\nSeeds where enriched beats base (of 5):")
    print(wins.to_string())
    means.to_csv(FOLDER / "seed_check_means.csv")
    summary.to_csv(FOLDER / "seed_check_summary.csv")
    print(f"\nSaved to {FOLDER}")


if __name__ == "__main__":
    if MODE == "seeds":
        FOLDER.mkdir(parents=True, exist_ok=True)
        seed_check()
    else:
        run(pilot=(MODE == "pilot"))
