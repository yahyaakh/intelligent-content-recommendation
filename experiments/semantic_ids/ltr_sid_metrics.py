'''
Precision@10 and recall@10 for the semantic-ID LTR experiments (doc 27 sec. 7, doc 30 stages 2a and 2c),
which saved only NDCG@10.

Refits exactly the same models (tuned XGBoost + LightGBM and their rank ensemble, L1 harness) on the
saved feature tables and scores every user with NDCG, precision, recall and hit rate@10.
Check first: every user's NDCG@10 equals the saved per-user results. No new decision is made here.

Usage:
    python experiments/semantic_ids/ltr_sid_metrics.py 10      10 seeds
    python experiments/semantic_ids/ltr_sid_metrics.py 3       3 seeds
Outputs: output/experiments/ltr_sid_metrics/
'''

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))  # shared modules live in src/
import sys

N_SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
sys.argv = [sys.argv[0]]                 # the imported experiment scripts read sys.argv when imported
import json
from pathlib import Path

import numpy as np
import pandas as pd

import build_movie_texts as bmt
import cold_start_experiment as cs
import llm_rerank_experiment as lr
import LightGBM_XBoost_rerank_experiment as ltr
import ltr_ncf_knn_experiment as l1

EXP = Path("output") / "experiments"
OUT = EXP / "ltr_sid_metrics"
G = ["sid_share_l1", "sid_share_l2", "sid_share_l3"]
R7 = l1.R7
# name: (folder, feature-table stem, saved column of the per-user files, feature columns)
EXPERIMENTS = {
    "R7 (LTR v2)":             (EXP / "ltr_semantic_ids", "g8", "R7", R7),
    "R7 + content IDs (doc 27)": (EXP / "ltr_semantic_ids", "g8", "R10 = R7 + SID", R7 + G),
    "R7 + SVD IDs (doc 30, 2a)": (EXP / "ltr_sid_sources" / "S3", "g9", "R11 = R7 + G9", R7 + G),
    "R7 + EASE IDs (doc 30, 2c)": (EXP / "ltr_sid_sources" / "S4", "g9", "R11 = R7 + G9", R7 + G),
}


def metrics_per_user(table, score, relevant_of):
    """NDCG, precision, recall and hit @10 per user, each user's 30 rows sorted by score (high first)."""
    rows = []
    for user_id, group in table.assign(score=score).groupby("user_id", sort=False):
        top10 = group.sort_values("score", ascending=False, kind="stable")["movie_id"].tolist()[:bmt.K]
        m = bmt.user_metrics(top10, set(relevant_of[user_id]), bmt.K)
        rows.append({"user_id": user_id, **m, "hit": float(m["precision"] > 0)})
    return pd.DataFrame(rows).set_index("user_id")


def fit_all(train, table, columns, chosen, relevant_of):
    ranks, out = {}, {}
    for library in ("xgboost", "lightgbm"):
        model = ltr.fit_ranker(train, columns, library, chosen[library])
        scores = model.predict(table[columns])
        ranks[library] = ltr.within_user_rank(table, scores)
        out[library] = metrics_per_user(table, scores, relevant_of)
    out["ensemble"] = metrics_per_user(table, (ranks["xgboost"] + ranks["lightgbm"]) / 2, relevant_of)
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = lr.load_data()
    test_ids = {u["user_id"] for u in data["test_users"]}
    with open(ltr.FOLDER / "step4_chosen_params.json", encoding="utf-8") as f:
        chosen = json.load(f)
    relevant_val = {u["user_id"]: u["relevant"] for u in cs.users_with_n_seeds(data["val_users"], N_SEEDS)}
    _, _, users_by_half = ltr.warm_halves(data, N_SEEDS)
    relevant_sim = {u["user_id"]: u["relevant"] for users in users_by_half.values() for u in users}
    assert not (test_ids & (set(relevant_val) | set(relevant_sim))), "a test user is in this experiment"

    rows, per_user = [], []
    for name, (folder, stem, saved_name, columns) in EXPERIMENTS.items():
        val = pd.read_csv(folder / f"features_val_{stem}_n{N_SEEDS}.csv")
        simulated = pd.read_csv(folder / f"features_simulated_{stem}_n{N_SEEDS}.csv")
        is_held = l1.holdout_split(simulated)
        settings = {"held-out": (simulated[~is_held], simulated[is_held], relevant_sim, "holdout"),
                    "validation": (simulated, val, relevant_val, "val")}
        for where, (train, table, relevant, file_tag) in settings.items():
            print(f"{name} | {where}: fitting...", flush=True)
            result = fit_all(train, table, columns, chosen, relevant)
            saved = pd.read_csv(folder / f"per_user_{file_tag}_n{N_SEEDS}.csv", index_col=0)
            for library, frame in result.items():
                old = saved[f"{saved_name} | {library}"].reindex(frame.index)
                assert old.notna().all(), f"{name} {where}: users differ from the saved run"
                gap = (old - frame["ndcg"]).abs().max()
                assert gap < 1e-9, f"{name} | {where} | {library}: NDCG differs from the saved run ({gap:.2e})"
                rows.append({"features": name, "users": where, "model": library, "n_users": len(frame),
                             **frame[["ndcg", "precision", "recall", "hit"]].mean().to_dict()})
                per_user.append(frame.assign(features=name, users=where, model=library).reset_index())
    print("check passed: every user's NDCG@10 equals the saved runs")

    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / f"summary_n{N_SEEDS}.csv", index=False)
    pd.concat(per_user).to_csv(OUT / f"per_user_n{N_SEEDS}.csv", index=False)
    pd.set_option("display.width", 250)
    ens = summary[summary.model == "ensemble"]
    print(f"\n=== {N_SEEDS} seeds, ensemble ===")
    print(ens.pivot_table(index="features", columns="users", values=["ndcg", "precision", "recall", "hit"],
                          sort=False).round(4).to_string())
    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    main()
