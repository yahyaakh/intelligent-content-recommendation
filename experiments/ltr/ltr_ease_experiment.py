'''
Does EASE help the learned ranker? (doc 26, section 6, use 3)

LTR v2 (doc 16) reranks Hybrid E's top 30 with 25 features (R7), as a tuned XGBoost + LightGBM ensemble
trained on cross-fitted simulated warm users. This script adds one feature group, G7:
  ease_score   EASE score of the candidate: sum of B's rows for the user's seed films
  ease_rank    its rank among the user's 30 candidates (1 = best)
Feature sets (fixed in advance):  R7 (LTR v2) | R9 = R7 + EASE

Cross-fitting (as docs 16, 20 and 24; no user's own ratings are ever inside the B that scores them):
  simulated warm users of half A  ->  EASE fitted on half B only (and the other way round)
  validation users                ->  EASE fitted on all warm users
Lambda (fixed before running): the value chosen on the simulated warm users (ease_tuning.py), which were
scored by half models, so it is a half-size lambda. G = X^T X grows with the number of users, so the
full-data model uses lambda x (warm users / users in the half), the same idea as user-kNN's k being
scaled to a half in doc 20. This keeps B on the same scale for the training and validation features.

Evaluation (the L1 harness of doc 20 section 10.1 and doc 24, unchanged):
  1. held-out simulated users: train on 80% of the simulated users, score the other 20% (doc 16 split)
  2. the 203 validation users: train on all simulated users
  Check first: R7 on validation must reproduce LTR v2 (0.6686 with 10 seeds, 0.6227 with 3).
  Test: R9 ensemble vs R7 ensemble, paired Wilcoxon, per user group.
  Decision rule (doc 24, fixed before running): adopt only if R9 gains >= +0.005 on BOTH user groups.
Also reported (descriptive): mean within-user Spearman correlation of ease_score with the existing
collaborative features (does EASE repeat what R7 already has?).

The test users are never touched. Nothing calls an LLM.

Usage:
    python experiments/ltr/ltr_ease_experiment.py pilot      features only (both seed counts), checks, no model fits
    python experiments/ltr/ltr_ease_experiment.py 10         10 seeds
    python experiments/ltr/ltr_ease_experiment.py 3          3 seeds
Outputs: output/experiments/ltr_ease/
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
from recomm import EASE_R_Recommende

OUT = Path("output") / "experiments" / "ltr_ease"
G7 = ["ease_score", "ease_rank"]
R7 = l1.R7
FEATURE_SETS = {"R7": R7, "R9 = R7 + EASE": R7 + G7}
INPUT_TABLES = l1.INPUT_TABLES
EXPECTED_R7 = l1.EXPECTED_R7
EASE_LAMBDA_HALF = {10: 20000, 3: 120000}           # ease_tuning.py (tuning.csv), half-size models
MIN_GAIN = 0.005
COMPARE_WITH = ["cf_mean", "cf_max", "hybrid_score", "log_popularity", "svd_score", "covis_max", "covis_mean"]


def fit_ease(ratings, lambda_):
    model = EASE_R_Recommende(ratings, pd.DataFrame(), lambda_)
    model.fit(ratings)
    return model


def add_g7(table, users, model):
    """G7 columns for a feature table (30 rows per user), from an EASE fitted WITHOUT these users."""
    user_of = {u["user_id"]: u for u in users}
    score = np.full(len(table), np.nan)
    for user_id, group in tqdm(table.groupby("user_id", sort=False), desc="G7 features", leave=False):
        user = user_of[user_id]
        seed_cols = [model.film_to_col[m] for m in user["liked"] if m in model.film_to_col]
        all_scores = model.B[seed_cols].sum(axis=0)          # = x . B for the 0/1 seed vector x
        cand = [model.film_to_col.get(int(m), -1) for m in group["movie_id"]]
        idx = table.index.get_indexer(group.index)
        score[idx] = [all_scores[c] if c >= 0 else np.nan for c in cand]
    out = table.assign(ease_score=score)
    out["ease_rank"] = out.groupby("user_id", sort=False)["ease_score"].rank(
        ascending=False, method="first", na_option="bottom")
    return out


def build_tables(data, n_seeds):
    paths = {split: OUT / f"features_{split}_g7_n{n_seeds}.csv" for split in ("val", "simulated")}
    if all(p.exists() for p in paths.values()):
        print(f"Loaded {paths['val'].name} and {paths['simulated'].name}")
        return {split: pd.read_csv(p) for split, p in paths.items()}

    val_name, sim_name = INPUT_TABLES[n_seeds]
    val = pd.read_csv(ltr.FOLDER / val_name)
    simulated = pd.read_csv(ltr.FOLDER / sim_name)
    train, in_A, users_by_half = ltr.warm_halves(data, n_seeds)
    assert len(train) == len(data["train"]), "the warm users differ from the cold-start split"
    n_warm = train["user_id"].nunique()

    parts = []
    for half, other in (("A", train[~in_A]), ("B", train[in_A])):
        lambda_ = EASE_LAMBDA_HALF[n_seeds]
        print(f"Simulated users of half {half}: EASE fitted on the other half "
              f"({other['user_id'].nunique():,} users, lambda {lambda_:,})")
        parts.append(add_g7(simulated[simulated["half"] == half], users_by_half[half], fit_ease(other, lambda_)))
    tables = {"simulated": pd.concat(parts).loc[simulated.index]}            # original row order

    n_half = np.mean([train[in_A]["user_id"].nunique(), train[~in_A]["user_id"].nunique()])
    lambda_full = EASE_LAMBDA_HALF[n_seeds] * n_warm / n_half
    print(f"Validation users: EASE fitted on all {n_warm:,} warm users, lambda {lambda_full:,.0f} "
          f"(= {EASE_LAMBDA_HALF[n_seeds]:,} x {n_warm:,} / {n_half:,.0f})")
    val_users = cs.users_with_n_seeds(data["val_users"], n_seeds)
    tables["val"] = add_g7(val, val_users, fit_ease(data["train"], lambda_full))

    for split, t in tables.items():
        assert (t.groupby("user_id").size() == ltr.TOP_M).all(), f"not 30 rows per user ({split})"
        print(f"{split}: {t['ease_score'].isna().sum()} candidate rows without an EASE score")
        t.to_csv(paths[split], index=False)
    return tables


def describe(tables, n_seeds):
    """Shift check (validation vs cross-fitted simulated) and redundancy with R7's collaborative features."""
    rows = []
    for split, t in tables.items():
        q = t["ease_score"].quantile([0.25, 0.5, 0.75])
        rows.append({"users": split, "mean": t["ease_score"].mean(), "q25": q[0.25], "median": q[0.5],
                     "q75": q[0.75]})
    shift = pd.DataFrame(rows)
    shift.to_csv(OUT / f"feature_shift_n{n_seeds}.csv", index=False)
    print("\nease_score distribution (validation vs cross-fitted simulated users):")
    print(shift.round(4).to_string(index=False))

    corr = []
    for split, t in tables.items():
        cols = ["ease_score"] + [c for c in COMPARE_WITH if c in t]
        ranks = t[["user_id"] + cols].copy()
        ranks[cols] = ranks.groupby("user_id")[cols].rank()
        for other in cols[1:]:
            per_user = ranks.groupby("user_id")[["ease_score", other]].apply(
                lambda g: g["ease_score"].corr(g[other])
                if g["ease_score"].nunique() > 1 and g[other].nunique() > 1 else np.nan)
            corr.append({"users": split, "with": other, "spearman": per_user.mean()})
    corr = pd.DataFrame(corr)
    corr.to_csv(OUT / f"correlation_n{n_seeds}.csv", index=False)
    print("\nMean within-user Spearman correlation of ease_score (high = repeats what R7 has):")
    print(corr.pivot_table(index="with", columns="users", values="spearman", sort=False).round(3).to_string())

    for split, t in tables.items():      # EASE alone, reranking the same 30 candidates (descriptive)
        top = t.sort_values(["user_id", "ease_rank"]).groupby("user_id").head(10)
        print(f"{split}: share of labelled-relevant rows in EASE's top 10 of the 30 = {top['label'].mean():.4f} "
              f"(Hybrid E's top 10: {t[t['hybrid_rank'] <= 10]['label'].mean():.4f})")


def run(tables, relevant_val, relevant_sim, chosen, n_seeds):
    val, simulated = tables["val"], tables["simulated"]
    is_held = l1.holdout_split(simulated)
    settings = {"held-out simulated users": (simulated[~is_held], simulated[is_held], relevant_sim),
                "validation users": (simulated, val, relevant_val)}
    per_user, summary = {}, []
    for where, (train, table, relevant) in settings.items():
        per_user[where] = {"Hybrid E": ltr.ndcg_per_user(table, "hybrid_rank", relevant, ascending=True),
                           "EASE on C30": ltr.ndcg_per_user(table, "ease_rank", relevant, ascending=True)}
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
        t = l1.paired_tests(frame, "R7 | ensemble", ["R9 = R7 + EASE | ensemble"])
        tests.append(t.assign(users=where, n_users=len(frame)))
    tests = pd.concat(tests, ignore_index=True)
    gains = tests.set_index("users")["difference"]
    decision = pd.DataFrame([{"features": "R9 = R7 + EASE", "gain_held_out": gains["held-out simulated users"],
                              "gain_validation": gains["validation users"],
                              "adopted": bool((gains >= MIN_GAIN).all())}])
    summary = pd.DataFrame(summary)
    summary.to_csv(OUT / f"summary_n{n_seeds}.csv", index=False)
    tests.to_csv(OUT / f"tests_n{n_seeds}.csv", index=False)
    decision.to_csv(OUT / f"decision_n{n_seeds}.csv", index=False)

    pd.set_option("display.width", 250)
    print(f"\n=== {n_seeds} seeds: NDCG@10 ===")
    print(summary.pivot_table(index="features", columns=["users", "model"], values="ndcg", sort=False)
          .round(4).to_string())
    for where, values in per_user.items():
        print(f"Hybrid E, {where}: {values['Hybrid E'].mean():.4f} | EASE alone on the 30 candidates: "
              f"{values['EASE on C30'].mean():.4f}")
    print("\nR9 ensemble vs R7 ensemble (paired Wilcoxon; one test per user group):")
    print(tests.round(4).to_string(index=False))
    print(f"\nDecision rule: adopt only if R9 gains >= +{MIN_GAIN} on BOTH user groups")
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

    for n_seeds in seed_counts:
        print(f"\n######## {n_seeds} seeds ########")
        relevant_val = {u["user_id"]: u["relevant"] for u in cs.users_with_n_seeds(data["val_users"], n_seeds)}
        _, _, users_by_half = ltr.warm_halves(data, n_seeds)
        relevant_sim = {u["user_id"]: u["relevant"] for users in users_by_half.values() for u in users}
        assert not (test_ids & (set(relevant_val) | set(relevant_sim))), "a test user is in this experiment"
        tables = build_tables(data, n_seeds)
        describe(tables, n_seeds)
        if not pilot:
            run(tables, relevant_val, relevant_sim, chosen, n_seeds)
    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    main()
