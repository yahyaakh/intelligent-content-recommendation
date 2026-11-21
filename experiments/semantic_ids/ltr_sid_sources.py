'''
Semantic IDs from a collaborative source as LTR v2 features (use 2), doc 30 stage 2a/2c.
Pre-registration: output/experiments/sid_vector_sources/preregistration.md, addition of 2026-10-29.

Same harness and features as ltr_semantic_id_experiment.py (doc 27 sec. 7), with other IDs:
  G9 = sid_share_l1/l2/l3  share of the user's seeds that share the candidate's first 1/2/3 tokens,
                           MEAN over k-means seeds 42-46. IDs come from sid_vector_sources.py (ids/).
Cross-fitted: simulated users of half A use the IDs fitted on half B and vice versa; validation users
use the IDs fitted on all warm users. A seed or candidate without an ID matches nothing.

Feature sets: R7 (LTR v2) and R11 = R7 + G9 (decides).
Check first: R7 reproduces LTR v2 (0.6686 with 10 seeds, 0.6227 with 3).
Decision rule (doc 24): adopt only if R11 gains >= +0.005 on BOTH held-out and validation users.
Test users are never touched. Nothing calls an LLM.

Usage:
    python experiments/semantic_ids/ltr_sid_sources.py pilot [S3]   features only (both seed counts), checks, no model fits
    python experiments/semantic_ids/ltr_sid_sources.py 10 [S3]      10 seeds
    python experiments/semantic_ids/ltr_sid_sources.py 3 [S3]       3 seeds
Outputs: output/experiments/ltr_sid_sources/<source>/
'''

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))  # shared modules live in src/
import sys

ARG = sys.argv[1] if len(sys.argv) > 1 else "10"
SOURCE = sys.argv[2] if len(sys.argv) > 2 else "S3"
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

IDS = Path("output") / "experiments" / "sid_vector_sources" / "ids"
OUT = Path("output") / "experiments" / "ltr_sid_sources" / SOURCE
KMEANS_SEEDS = [42, 43, 44, 45, 46]
CODES = 32
N_LEVELS = 3
G9 = ["sid_share_l1", "sid_share_l2", "sid_share_l3"]
R7 = l1.R7
FEATURE_SETS = {"R7": R7, "R11 = R7 + G9": R7 + G9}
MAIN = "R11 = R7 + G9"
INPUT_TABLES = l1.INPUT_TABLES
EXPECTED_R7 = l1.EXPECTED_R7
MIN_GAIN = 0.005
OTHER = {"A": "B", "B": "A"}
COMPARE_WITH = ["svd_mean", "svd_max", "svd_score", "covis_mean", "cf_mean", "emb_mean", "hybrid_score"]


def load_ids(fit_on):
    """One (row_of_movie, tokens) pair per k-means seed, from the saved IDs."""
    out = []
    for seed in KMEANS_SEEDS:
        z = np.load(IDS / f"{SOURCE}_{fit_on}_k{CODES}_s{seed}.npz")
        out.append(({int(m): i for i, m in enumerate(z["movie_ids"])}, z["codes"][:, :N_LEVELS]))
    return out


def tokens_of(ids, movie_ids, missing):
    row_of, tokens = ids
    return np.array([tokens[row_of[m]] if m in row_of else [missing] * N_LEVELS for m in movie_ids])


def shares(ids, seed_ids, candidate_ids):
    """(candidates, 3): share of seeds sharing the first 1, 2, 3 tokens. Missing IDs never match
    (seeds get -1, candidates -2)."""
    seeds = tokens_of(ids, seed_ids, -1)
    cands = tokens_of(ids, candidate_ids, -2)
    same = cands[:, None, :] == seeds[None, :, :]
    return np.cumprod(same, axis=2).mean(axis=1)


def add_g9(table, users, ids_of_user):
    user_of = {u["user_id"]: u for u in users}
    values = np.zeros((len(table), N_LEVELS))
    for user_id, group in tqdm(table.groupby("user_id", sort=False), desc="G9 features", leave=False):
        user = user_of[user_id]
        candidates = [int(m) for m in group["movie_id"]]
        idx = table.index.get_indexer(group.index)
        values[idx] = np.mean([shares(ids, [int(m) for m in user["liked"]], candidates)
                               for ids in ids_of_user(user_id)], axis=0)
    out = table.copy()
    out[G9] = values
    return out


def build_tables(data, n_seeds, ids):
    paths = {split: OUT / f"features_{split}_g9_n{n_seeds}.csv" for split in ("val", "simulated")}
    if all(p.exists() for p in paths.values()):
        print(f"Loaded {paths['val'].name} and {paths['simulated'].name}")
        return {split: pd.read_csv(p) for split, p in paths.items()}

    val_name, sim_name = INPUT_TABLES[n_seeds]
    val = pd.read_csv(ltr.FOLDER / val_name)
    simulated = pd.read_csv(ltr.FOLDER / sim_name)
    _, _, users_by_half = ltr.warm_halves(data, n_seeds)
    half_of = {u["user_id"]: h for h, users in users_by_half.items() for u in users}
    sim_users = [u for users in users_by_half.values() for u in users]
    val_users = cs.users_with_n_seeds(data["val_users"], n_seeds)
    assert set(simulated["user_id"]) <= set(half_of), "simulated user without a half"

    tables = {"val": add_g9(val, val_users, lambda _: ids["all-pool"]),
              "simulated": add_g9(simulated, sim_users, lambda u: ids[OTHER[half_of[u]]])}
    for split, t in tables.items():
        assert (t.groupby("user_id").size() == ltr.TOP_M).all(), f"not 30 rows per user ({split})"
        assert t[G9].notna().all().all(), f"missing G9 values ({split})"
        t.to_csv(paths[split], index=False)
    return tables


def describe(tables, n_seeds):
    rows = []
    for split, t in tables.items():
        for c in G9:
            rows.append({"users": split, "feature": c, "mean": t[c].mean(), "share_nonzero": (t[c] > 0).mean()})
    shift = pd.DataFrame(rows)
    shift.to_csv(OUT / f"feature_shift_n{n_seeds}.csv", index=False)
    print("\nG9 features (validation vs simulated users):")
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

    tests = []
    for where, values in per_user.items():
        frame = pd.DataFrame(values)
        frame.to_csv(OUT / f"per_user_{'holdout' if 'held' in where else 'val'}_n{n_seeds}.csv")
        tests.append(l1.paired_tests(frame, "R7 | ensemble", [f"{MAIN} | ensemble"])
                     .assign(users=where, n_users=len(frame)))
    tests = pd.concat(tests, ignore_index=True)
    gain = tests.set_index("users")["difference"]
    decision = pd.DataFrame([{"source": SOURCE, "features": MAIN, "seeds": n_seeds,
                              "gain_held_out": gain["held-out simulated users"],
                              "gain_validation": gain["validation users"],
                              "adopted": bool((gain >= MIN_GAIN).all())}])
    summary = pd.DataFrame(summary)
    summary.to_csv(OUT / f"summary_n{n_seeds}.csv", index=False)
    tests.to_csv(OUT / f"tests_n{n_seeds}.csv", index=False)
    decision.to_csv(OUT / f"decision_n{n_seeds}.csv", index=False)

    pd.set_option("display.width", 250)
    print(f"\n=== {SOURCE}, {n_seeds} seeds: NDCG@10 ===")
    print(summary.pivot_table(index="features", columns=["users", "model"], values="ndcg", sort=False)
          .round(4).to_string())
    print("\nR11 ensemble vs R7 (paired Wilcoxon):")
    print(tests.round(4).to_string(index=False))
    print(f"\nDecision rule: adopt only if R11 gains >= +{MIN_GAIN} on BOTH user groups")
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
    ids = {fit_on: load_ids(fit_on) for fit_on in ("A", "B", "all-pool")}
    print({k: len(v[0][0]) for k, v in ids.items()}, "films with an ID (k-means seed 42)")

    for n_seeds in seed_counts:
        print(f"\n######## {SOURCE}, {n_seeds} seeds ########")
        relevant_val = {u["user_id"]: u["relevant"] for u in cs.users_with_n_seeds(data["val_users"], n_seeds)}
        _, _, users_by_half = ltr.warm_halves(data, n_seeds)
        relevant_sim = {u["user_id"]: u["relevant"] for users in users_by_half.values() for u in users}
        assert not (test_ids & (set(relevant_val) | set(relevant_sim))), "a test user is in this experiment"
        tables = build_tables(data, n_seeds, ids)
        describe(tables, n_seeds)
        if not pilot:
            run(tables, relevant_val, relevant_sim, chosen, n_seeds)
    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    main()
