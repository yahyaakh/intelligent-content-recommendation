'''
Position bias: rank the same candidates in three different random orders and average.

DeepSeek V4.1 Flash keeps part of the (random) order in which the 30 candidates are shown
(position bias 0.19-0.24, doc 13). Here the same user and candidates are shown in shuffles 0
(the order of every earlier run), 1 and 2, and the three rankings are combined by their mean
rank (a Borda count; ties broken by the shuffle-0 ranking). The average cancels most of the
order effect (Hou et al., ECIR 2024).

Setting: DeepSeek V4.1 Flash, prompt p2, N = 10, validation users; idea 1 (Hybrid E candidates)
and idea 3 (popularity candidates). Shuffle 0 is cached from the earlier runs.

Reported: NDCG@10 of each single shuffle, of the 3-shuffle average, and their paired tests;
how stable a single ranking is across shuffles (Kendall tau between shuffles); position bias.

Usage:
    python experiments/llm/shuffle_average_experiment.py
'''

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))  # shared modules live in src/
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr, wilcoxon

sys.argv = [sys.argv[0]]
import build_movie_texts as bmt
import cold_start_experiment as cs
import llm_rerank_experiment as lr

FOLDER = Path("output") / "experiments" / "shuffle_average"
FOLDER.mkdir(parents=True, exist_ok=True)
MODEL = ("ollama_cloud", "deepseek-v4.1-flash")
SHUFFLES = (0, 1, 2)
VERSION, N_SEEDS, SPLIT = "p2", 10, "val"


def rankings(user, data, candidates):
    """The model's full ranking of the candidates for every shuffle, plus the candidates."""
    top_ids, _ = lr.get_candidates(user, data, candidates)
    out = {}
    for k in SHUFFLES:
        path = lr.run_name(*MODEL, N_SEEDS, None, VERSION, candidates, k)[0]
        answers = CACHES.setdefault(path, {r["user_id"]: r for r in map(json.loads, path.read_text(encoding="utf-8").splitlines())})
        _, number_to_id = lr.build_prompt(user, top_ids, data, VERSION, k)
        order, _ = lr.parse_ranking(answers[user["user_id"]]["content"], number_to_id, top_ids)
        out[k] = (order, {m: n for n, m in number_to_id.items()})
    return top_ids, out


CACHES = {}

if __name__ == "__main__":
    data = lr.load_data()
    # 1. The two new shuffles (shuffle 0 is cached); cached answers are reused on reruns
    for candidates in ("hybrid", "popularity"):
        for k in SHUFFLES[1:]:
            lr.compare_models([MODEL], data, split=SPLIT, version=VERSION, n_seeds=N_SEEDS,
                              candidates=candidates, shuffle=k)

    # 2. Single shuffles vs their average
    users = cs.users_with_n_seeds(data[f"{SPLIT}_users"], N_SEEDS)
    rows, stability = [], []
    for candidates in ("hybrid", "popularity"):
        for user in users:
            top_ids, ranks = rankings(user, data, candidates)
            position = {k: {m: i for i, m in enumerate(order)} for k, (order, _) in ranks.items()}
            mean_rank = {m: np.mean([position[k][m] for k in SHUFFLES]) for m in top_ids}
            average = sorted(top_ids, key=lambda m: (mean_rank[m], position[0][m]))
            row = {"candidates": candidates, "user_id": user["user_id"],
                   "baseline": bmt.ndcg_at_k(top_ids[:bmt.K], user["relevant"], bmt.K),
                   "average_3": bmt.ndcg_at_k(average[:bmt.K], user["relevant"], bmt.K)}
            for k, (order, shown) in ranks.items():
                row[f"shuffle_{k}"] = bmt.ndcg_at_k(order[:bmt.K], user["relevant"], bmt.K)
                row[f"bias_{k}"] = spearmanr([shown[m] for m in order], range(len(order))).correlation
            rows.append(row)
            stability.append({"candidates": candidates, "user_id": user["user_id"],
                              "tau_0_1": kendalltau([position[0][m] for m in top_ids], [position[1][m] for m in top_ids]).statistic,
                              "tau_0_2": kendalltau([position[0][m] for m in top_ids], [position[2][m] for m in top_ids]).statistic,
                              "top10_overlap_0_1": len(set(ranks[0][0][:bmt.K]) & set(ranks[1][0][:bmt.K])) / bmt.K})
    per_user, stability = pd.DataFrame(rows), pd.DataFrame(stability)

    summary, tests = [], []
    for candidates, part in per_user.groupby("candidates", sort=False):
        st = stability[stability["candidates"] == candidates]
        summary.append({"candidates": candidates, "users": len(part), "baseline": part["baseline"].mean(),
                        **{f"shuffle_{k}": part[f"shuffle_{k}"].mean() for k in SHUFFLES},
                        "mean_single_shuffle": part[[f"shuffle_{k}" for k in SHUFFLES]].mean(axis=1).mean(),
                        "average_3": part["average_3"].mean(),
                        "position_bias": part[[f"bias_{k}" for k in SHUFFLES]].mean().mean(),
                        "kendall_tau_between_shuffles": st[["tau_0_1", "tau_0_2"]].mean().mean(),
                        "top10_overlap_between_shuffles": st["top10_overlap_0_1"].mean()})
        for a, b in (("shuffle_0", "average_3"), ("baseline", "average_3"), ("baseline", "shuffle_0")):
            diff = part[b] - part[a]
            tests.append({"candidates": candidates, "comparison": f"{b} vs {a}", "difference": diff.mean(),
                          "better": int((diff > 0).sum()), "worse": int((diff < 0).sum()),
                          "p_value": wilcoxon(diff).pvalue if diff.any() else 1.0})
    summary, tests = pd.DataFrame(summary), pd.DataFrame(tests)
    tests["p_bonferroni"] = (tests["p_value"] * len(tests)).clip(upper=1.0)
    per_user.to_csv(FOLDER / "per_user.csv", index=False)
    stability.to_csv(FOLDER / "stability.csv", index=False)
    summary.to_csv(FOLDER / "summary.csv", index=False)
    tests.to_csv(FOLDER / "significance_tests.csv", index=False)
    pd.set_option("display.width", 250)
    print(summary.round(4).to_string(index=False))
    print(tests.round(4).to_string(index=False))
