'''
Semantic IDs, use 2 (doc 18 section 3, step 4; doc 27): do prefix-match features help the learned ranker?

LTR v2 (doc 16) reranks Hybrid E's top 30 with 25 features (R7), as a tuned XGBoost + LightGBM ensemble
trained on cross-fitted simulated warm users. R7 already contains content features (emb_mean, emb_max,
content_max, emb_v5_*, genre_overlap, director_match, decade_gap).

New group G8 (one value per user and candidate), fixed before running:
  sid_share_l1   share of the user's seeds whose semantic ID shares the candidate's first token
  sid_share_l2   ... first two tokens
  sid_share_l3   ... all three tokens
IDs: enriched embedding, 32 codes, 3 levels, flat (the setting chosen in use 1 on the simulated users).
Because single k-means runs are noisy (doc 27 section 4a), each feature is the MEAN over 5 k-means seeds
(42-46). The IDs use no ratings, so no cross-fitting is needed.

Feature sets:
  R7                     LTR v2
  R10 = R7 + SID         the decision is made on this one
  R10 (seed 42 only)     descriptive: the same features from one k-means run

Evaluation: the L1 harness of doc 20 section 10.1 / doc 24 / ltr_ease_experiment.py, unchanged.
  1. held-out simulated users (train on the other 80%), 2. the 203 validation users (train on all).
  Check first: R7 reproduces LTR v2 (0.6686 with 10 seeds, 0.6227 with 3).
  Tests: ensembles vs R7, paired Wilcoxon, Bonferroni over 2 within each user group.
  Decision rule (doc 24): adopt only if R10 gains >= +0.005 on BOTH user groups.
Also: mean within-user Spearman correlation of the G8 features with R7's content features.

The test users are never touched. Nothing calls an LLM.

Usage:
    python experiments/semantic_ids/ltr_semantic_id_experiment.py pilot    features only (both seed counts), checks, no model fits
    python experiments/semantic_ids/ltr_semantic_id_experiment.py 10       10 seeds
    python experiments/semantic_ids/ltr_semantic_id_experiment.py 3        3 seeds
Outputs: output/experiments/ltr_semantic_ids/
'''

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))  # shared modules live in src/
import sys

ARG = sys.argv[1] if len(sys.argv) > 1 else "10"
sys.argv = [sys.argv[0]]                 # the imported experiment scripts read sys.argv when imported
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import cold_start_experiment as cs
import llm_rerank_experiment as lr
import LightGBM_XBoost_rerank_experiment as ltr
import ltr_ncf_knn_experiment as l1                 # the L1 harness: holdout split, fits, tests
from recomm import Semantic_IDs

OUT = Path("output") / "experiments" / "ltr_semantic_ids"
EMBEDDING = "movies_enriched_embeddinggemma_v2"
CODEBOOK_SIZE = 32
KMEANS_SEEDS = [42, 43, 44, 45, 46]
G8 = ["sid_share_l1", "sid_share_l2", "sid_share_l3"]
G8_SINGLE = [f"{c}_seed42" for c in G8]
R7 = l1.R7
FEATURE_SETS = {"R7": R7, "R10 = R7 + SID": R7 + G8, "R10 (seed 42 only)": R7 + G8_SINGLE}
INPUT_TABLES = l1.INPUT_TABLES
EXPECTED_R7 = l1.EXPECTED_R7
MIN_GAIN = 0.005
COMPARE_WITH = ["emb_mean", "emb_max", "content_max", "emb_v5_mean", "emb_v5_max", "genre_overlap",
                "director_match", "decade_gap", "hybrid_score"]


def build_ids(movies):
    models = []
    for seed in KMEANS_SEEDS:
        model = Semantic_IDs(movies, n_levels=3, codebook_size=CODEBOOK_SIZE, seed=seed)
        model.load_embeddings(name=EMBEDDING)
        model.fit()
        models.append(model)
    return models


def shares(model, seed_ids, candidate_ids):
    """(candidates, 3): share of seeds sharing the first 1, 2, 3 tokens with each candidate."""
    tokens = model.codes[:, :model.n_levels]
    seeds = tokens[[model.row_of_movie[m] for m in seed_ids]]                 # (seeds, 3)
    cands = tokens[[model.row_of_movie[m] for m in candidate_ids]]            # (candidates, 3)
    same = cands[:, None, :] == seeds[None, :, :]                             # (candidates, seeds, 3)
    prefix = np.cumprod(same, axis=2)                                         # 1 while all earlier levels match
    return prefix.mean(axis=1)                                                # mean over the seeds


def add_g8(table, users, models):
    user_of = {u["user_id"]: u for u in users}
    averaged = np.zeros((len(table), 3))
    single = np.zeros((len(table), 3))
    for user_id, group in tqdm(table.groupby("user_id", sort=False), desc="G8 features", leave=False):
        user = user_of[user_id]
        candidates = [int(m) for m in group["movie_id"]]
        idx = table.index.get_indexer(group.index)
        per_seed = [shares(model, user["liked"], candidates) for model in models]
        averaged[idx] = np.mean(per_seed, axis=0)
        single[idx] = per_seed[0]                                             # k-means seed 42
    out = table.copy()
    out[G8] = averaged
    out[G8_SINGLE] = single
    return out


def build_tables(data, n_seeds, models):
    paths = {split: OUT / f"features_{split}_g8_n{n_seeds}.csv" for split in ("val", "simulated")}
    if all(p.exists() for p in paths.values()):
        print(f"Loaded {paths['val'].name} and {paths['simulated'].name}")
        return {split: pd.read_csv(p) for split, p in paths.items()}

    val_name, sim_name = INPUT_TABLES[n_seeds]
    val = pd.read_csv(ltr.FOLDER / val_name)
    simulated = pd.read_csv(ltr.FOLDER / sim_name)
    _, _, users_by_half = ltr.warm_halves(data, n_seeds)
    sim_users = [u for users in users_by_half.values() for u in users]
    val_users = cs.users_with_n_seeds(data["val_users"], n_seeds)

    tables = {"val": add_g8(val, val_users, models), "simulated": add_g8(simulated, sim_users, models)}
    for split, t in tables.items():
        assert (t.groupby("user_id").size() == ltr.TOP_M).all(), f"not 30 rows per user ({split})"
        assert t[G8 + G8_SINGLE].notna().all().all(), f"missing G8 values ({split})"
        t.to_csv(paths[split], index=False)
    return tables


def describe(tables, n_seeds):
    rows = []
    for split, t in tables.items():
        for c in G8:
            rows.append({"users": split, "feature": c, "mean": t[c].mean(), "share_nonzero": (t[c] > 0).mean()})
    shift = pd.DataFrame(rows)
    shift.to_csv(OUT / f"feature_shift_n{n_seeds}.csv", index=False)
    print("\nG8 features (validation vs simulated users):")
    print(shift.pivot_table(index="feature", columns="users", values=["mean", "share_nonzero"]).round(4).to_string())

    corr = []
    for split, t in tables.items():
        cols = ["sid_share_l1", "sid_share_l2"] + [c for c in COMPARE_WITH if c in t]
        ranks = t[["user_id"] + cols].copy()
        ranks[cols] = ranks.groupby("user_id")[cols].rank()
        for feature in ["sid_share_l1", "sid_share_l2"]:
            for other in cols[2:]:
                per_user = ranks.groupby("user_id")[[feature, other]].apply(
                    lambda g: g[feature].corr(g[other]) if g[feature].nunique() > 1 and g[other].nunique() > 1
                    else np.nan)
                corr.append({"users": split, "feature": feature, "with": other, "spearman": per_user.mean()})
    corr = pd.DataFrame(corr)
    corr.to_csv(OUT / f"correlation_n{n_seeds}.csv", index=False)
    print("\nMean within-user Spearman correlation (high = repeats what R7 already has):")
    print(corr.pivot_table(index=["feature", "with"], columns="users", values="spearman", sort=False)
          .round(3).to_string())


def run(tables, relevant_val, relevant_sim, chosen, n_seeds):
    val, simulated = tables["val"], tables["simulated"]
    is_held = l1.holdout_split(simulated)
    settings = {"held-out simulated users": (simulated[~is_held], simulated[is_held], relevant_sim),
                "validation users": (simulated, val, relevant_val)}
    per_user, summary = {}, []
    for where, (train, table, relevant) in settings.items():
        per_user[where] = {"Hybrid E": ltr.ndcg_per_user(table, "hybrid_rank", relevant, ascending=True)}
        for name, columns in FEATURE_SETS.items():
            print(f"{where} | {name}: fitting tuned XGBoost + LightGBM...", flush=True)
            result, _ = l1.fit_and_score(train, table, columns, chosen, relevant)
            for library, values in result.items():
                per_user[where][f"{name} | {library}"] = values
                summary.append({"users": where, "n_users": len(values), "features": name, "model": library,
                                "ndcg": values.mean()})
        if where == "validation users":
            got = per_user[where]["R7 | ensemble"].mean()
            print(f"check: R7 ensemble on validation = {got:.4f} (LTR v2, expected {EXPECTED_R7[n_seeds]})")
            assert abs(got - EXPECTED_R7[n_seeds]) < 5e-5, "R7 does not reproduce LTR v2: stop"

    others = [f"{n} | ensemble" for n in FEATURE_SETS if n != "R7"]
    tests = []
    for where, values in per_user.items():
        frame = pd.DataFrame(values)
        frame.to_csv(OUT / f"per_user_{'holdout' if 'held' in where else 'val'}_n{n_seeds}.csv")
        tests.append(l1.paired_tests(frame, "R7 | ensemble", others).assign(users=where, n_users=len(frame)))
    tests = pd.concat(tests, ignore_index=True)
    main = tests[tests["comparison"].str.startswith("R10 = R7 + SID")].set_index("users")["difference"]
    decision = pd.DataFrame([{"features": "R10 = R7 + SID", "gain_held_out": main["held-out simulated users"],
                              "gain_validation": main["validation users"],
                              "adopted": bool((main >= MIN_GAIN).all())}])
    summary = pd.DataFrame(summary)
    summary.to_csv(OUT / f"summary_n{n_seeds}.csv", index=False)
    tests.to_csv(OUT / f"tests_n{n_seeds}.csv", index=False)
    decision.to_csv(OUT / f"decision_n{n_seeds}.csv", index=False)

    pd.set_option("display.width", 250)
    print(f"\n=== {n_seeds} seeds: NDCG@10 ===")
    print(summary.pivot_table(index="features", columns=["users", "model"], values="ndcg", sort=False)
          .round(4).to_string())
    print("\nEnsembles vs R7 (paired Wilcoxon, Bonferroni over 2 within each user group):")
    print(tests.round(4).to_string(index=False))
    print(f"\nDecision rule: adopt only if R10 gains >= +{MIN_GAIN} on BOTH user groups")
    print(decision.round(4).to_string(index=False))


def main():
    pilot = ARG == "pilot"
    seed_counts = (10, 3) if pilot else (int(ARG),)
    if not pilot and seed_counts[0] not in INPUT_TABLES:
        sys.exit(f"Use pilot, 10 or 3, not {ARG}.\n{__doc__}")
    OUT.mkdir(parents=True, exist_ok=True)
    data = lr.load_data()
    test_ids = {u["user_id"] for u in data["test_users"]}
    with open(ltr.FOLDER / "step4_chosen_params.json", encoding="utf-8") as f:
        chosen = json.load(f)
    models = build_ids(data["movies"])

    for n_seeds in seed_counts:
        print(f"\n######## {n_seeds} seeds ########")
        relevant_val = {u["user_id"]: u["relevant"] for u in cs.users_with_n_seeds(data["val_users"], n_seeds)}
        _, _, users_by_half = ltr.warm_halves(data, n_seeds)
        relevant_sim = {u["user_id"]: u["relevant"] for users in users_by_half.values() for u in users}
        assert not (test_ids & (set(relevant_val) | set(relevant_sim))), "a test user is in this experiment"
        tables = build_tables(data, n_seeds, models)
        describe(tables, n_seeds)
        if not pilot:
            run(tables, relevant_val, relevant_sim, chosen, n_seeds)
    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    main()
