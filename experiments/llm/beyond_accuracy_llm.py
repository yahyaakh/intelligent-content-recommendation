'''
Beyond-accuracy metrics for the LLM reranking (doc 13, ideas 1 and 3), with the definitions of
experiment 12 (beyond_accuracy.py): NDCG per popularity third, catalogue coverage, Gini, novelty,
long-tail share (low + mid popularity thirds) and intra-list genre diversity.

The top-10 lists are rebuilt from the cached LLM answers (no model calls). Baselines are the
candidate lists' own top 10: Hybrid E (idea 1) and popularity (idea 3). A reranking can only
reorder its 30 candidates, so these metrics show how the LLM's choice WITHIN those 30 differs.

Tests: per-user metrics (long-tail share, novelty, diversity, NDCG) with the Wilcoxon signed-rank
test, LLM vs the candidate list's own top 10; coverage and Gini are reported descriptively.

Usage:
    python experiments/llm/beyond_accuracy_llm.py
'''

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))  # shared modules live in src/
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.argv = [sys.argv[0]]
import beyond_accuracy as ba
import build_movie_texts as bmt
import cold_start_experiment as cs
import llm_rerank_experiment as lr
import profile_experiment as pe

FOLDER = Path("output") / "experiments" / "beyond_accuracy_llm"
FOLDER.mkdir(parents=True, exist_ok=True)

# (split, idea label, provider, model, version, N, candidates, think)
RUNS = [
    ("test", "idea 1", "ollama_cloud", "deepseek-v4.1-flash", "p2", 10, "hybrid", None),
    ("test", "idea 1", "ollama", "gemma4:e4b", "p2", 10, "hybrid", None),
    ("test", "idea 3", "ollama_cloud", "deepseek-v4.1-flash", "p2", 10, "popularity", None),
    ("val", "idea 1", "ollama_cloud", "deepseek-v4.1-flash", "p2", 10, "hybrid", None),
    ("val", "idea 3", "ollama_cloud", "deepseek-v4.1-flash", "p2", 10, "popularity", None),
    ("val", "idea 3", "ollama_cloud", "deepseek-v4.1-flash", "p2", 10, "popularity", True),
    ("val", "idea 3", "ollama_cloud", "glm-5.3-flash", "p2", 10, "popularity", None),
]
PER_USER = ["ndcg", "share_lowmid", "novelty", "diversity", "ndcg_low", "ndcg_mid"]


def load_answers(provider, model, n, version, candidates, think):
    path = lr.run_name(provider, model, n, think, version, candidates)[0]
    return {r["user_id"]: r for r in map(json.loads, path.read_text(encoding="utf-8").splitlines())}


if __name__ == "__main__":
    data = lr.load_data()
    movies = data["movies"]
    ratings = bmt.prepare_ratings(movies)
    train, _ = cs.split_users(ratings)
    pool_ids = set(train["movie_id"])
    popular = cs.popularity_scores(train)
    n_warm = train["user_id"].nunique()
    novelty_of = {m: -np.log2(max(popular.get(m, 0), 1) / n_warm) for m in pool_ids}
    third_of, _ = pe.popularity_thirds(movies, pool_ids)
    genres_of = {m: {g.strip() for g in re.split(r"[|,]", str(gs)) if g.strip() and str(gs) != "nan"}
                 for m, gs in zip(movies["movie_id"], movies["genres"])}

    rows = []
    for split, idea, provider, model, version, n, candidates, think in RUNS:
        users = cs.users_with_n_seeds(data[f"{split}_users"], n)
        answers = load_answers(provider, model, n, version, candidates, think)
        name = f"{model}{' (thinking)' if think else ''}"
        base_name = "Hybrid E" if candidates == "hybrid" else "popularity"
        for user in users:
            top_ids, _ = lr.get_candidates(user, data, candidates)
            _, number_to_id = lr.build_prompt(user, top_ids, data, version)
            order, _ = lr.parse_ranking(answers[user["user_id"]]["content"], number_to_id, top_ids)
            for method, rec in ((base_name, top_ids[:bmt.K]), (name, order[:bmt.K])):
                rows.append({"split": split, "idea": idea, "candidates": candidates, "run": name,
                             "method": method, "user_id": user["user_id"], "rec": rec,
                             **ba.list_metrics(rec, user, third_of, genres_of, novelty_of)})
    lists = pd.DataFrame(rows)
    lists["share_lowmid"] = lists["share_low"] + lists["share_mid"]

    summary, tests = [], []
    for (split, idea, run), part in lists.groupby(["split", "idea", "run"], sort=False):
        methods = part["method"].unique()                    # [baseline, LLM]
        for method in methods:
            sub = part[part["method"] == method]
            counts = pd.Series([m for rec in sub["rec"] for m in rec]).value_counts().reindex(sorted(pool_ids), fill_value=0)
            summary.append({"split": split, "idea": idea, "run": run, "method": method, "users": len(sub),
                            "ndcg": sub["ndcg"].mean(), "ndcg_low": sub["ndcg_low"].mean(), "ndcg_mid": sub["ndcg_mid"].mean(),
                            "coverage": (counts > 0).mean(), "gini": ba.gini(counts.to_numpy()),
                            "novelty": sub["novelty"].mean(), "long_tail_share": sub["share_lowmid"].mean(),
                            "diversity": sub["diversity"].mean()})
        a = part[part["method"] == methods[0]].set_index("user_id")
        b = part[part["method"] == methods[1]].set_index("user_id").loc[a.index]
        for metric in PER_USER:
            both = a[metric].notna() & b[metric].notna()
            diff = (b.loc[both, metric] - a.loc[both, metric])
            p = wilcoxon(diff).pvalue if diff.any() else 1.0
            tests.append({"split": split, "idea": idea, "comparison": f"{methods[1]} vs {methods[0]}", "metric": metric,
                          "users": int(both.sum()), "difference": diff.mean(),
                          "better": int((diff > 0).sum()), "worse": int((diff < 0).sum()), "p_value": p})
    summary, tests = pd.DataFrame(summary), pd.DataFrame(tests)
    tests["p_bonferroni"] = (tests["p_value"] * len(tests)).clip(upper=1.0)

    lists.assign(rec=lists["rec"].map(lambda r: "|".join(map(str, r)))).to_csv(FOLDER / "per_user.csv", index=False)
    summary.to_csv(FOLDER / "summary.csv", index=False)
    tests.to_csv(FOLDER / "significance_tests.csv", index=False)
    pd.set_option("display.width", 250)
    print(summary.round(4).to_string(index=False))
    print(tests.round(4).to_string(index=False))
