'''
Doc 17, method E follow-ups with Kimi K3 (no thinking), 203 validation users. Four experiments:

  1. Fusion   0.4 x Hybrid E rank + 0.6 x Kimi p2 rank (method A with Kimi). Cached answers only.
              Also: the weight chosen by 5-fold cross-validation, and reciprocal rank fusion (RRF).
  2. Idea 3   Kimi reranks the 30 most popular films (no rating data), p2, 10 seeds.
              Cached references: DeepSeek Flash with and without thinking, gemma4:e4b.
  3. Shuffles Kimi p2 (Hybrid E's top 30) with 2 more candidate orders (shuffle 1, 2), to measure
              how stable its tie with Hybrid E is. Cached reference: DeepSeek Flash shuffles 0-2.
  4. 3 seeds  Kimi p2 with 3 seed films (Hybrid E's top 30 for 3 seeds). Cached reference: gemma4:e4b.

Answers are cached per setting; a rerun costs nothing. Outputs: output/experiments/kimi_k3/

Usage:
    python experiments/llm/kimi_experiments.py            (all four)
    python experiments/llm/kimi_experiments.py fusion     (only experiment 1, no model calls)
'''

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))  # shared modules live in src/
import sys

ONLY = sys.argv[1] if len(sys.argv) > 1 else "all"
sys.argv = [sys.argv[0]]
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import cold_start_experiment as cs
import llm_rerank_experiment as lr
import llm_fusion_experiment as fusion

FOLDER = Path("output") / "experiments" / "kimi_k3"
FOLDER.mkdir(parents=True, exist_ok=True)
KIMI = ("ollama_cloud", "kimi-k3")
DEEPSEEK = ("ollama_cloud", "deepseek-v4.1-flash")
FIXED_W = 0.4          # the weight of method A (DeepSeek), fixed in advance for Kimi


def paired(a, b):
    """Mean of a - b, better / worse users, Wilcoxon p."""
    diff = (a - b).dropna()
    return {"difference": diff.mean(), "better": int((diff > 0).sum()), "worse": int((diff < 0).sum()),
            "p_value": wilcoxon(diff).pvalue if diff.any() else 1.0}


def run_fusion(data):
    """Experiment 1: method A with Kimi K3 p2 (cached), next to DeepSeek for comparison."""
    fusion.LLMS = [KIMI, DEEPSEEK]                   # user_inputs() reads this list
    users = cs.users_with_n_seeds(data["val_users"], fusion.N_SEEDS)
    inputs = fusion.user_inputs(users, data)
    ids = [u["user_id"] for u in inputs]
    hybrid = pd.Series([fusion.ndcg_of(u, u["hybrid_rank"], True) for u in inputs], index=ids)
    per_user, rows = {"Hybrid E": hybrid}, []
    for _, model in fusion.LLMS:
        methods = {
            "alone": pd.Series([fusion.ndcg_of(u, u["llm_rank"][model], True) for u in inputs], index=ids),
            f"rank blend, w = {FIXED_W} (fixed)": pd.Series([fusion.fused(u, model, "rank blend", FIXED_W)
                                                           for u in inputs], index=ids),
            "RRF": pd.Series([fusion.fused(u, model, "RRF") for u in inputs], index=ids),
        }
        cv_scores, chosen, _ = fusion.cross_validated(inputs, model, "rank blend")
        methods["rank blend, w by 5-fold CV"] = cv_scores
        for method, scores in methods.items():
            per_user[f"{model} {method}"] = scores
            rows.append({"model": model, "method": method, "ndcg": scores.mean(),
                         "w per fold": chosen if "CV" in method else "", **paired(scores, hybrid)})
    table = pd.DataFrame(rows)
    table["p_bonferroni"] = (table["p_value"] * len(table)).clip(upper=1.0)
    kimi_vs_ds = paired(per_user[f"kimi-k3 rank blend, w = {FIXED_W} (fixed)"],
                        per_user[f"deepseek-v4.1-flash rank blend, w = {FIXED_W} (fixed)"])
    pd.DataFrame(per_user).to_csv(FOLDER / "fusion_per_user_val.csv")
    table.to_csv(FOLDER / "fusion_summary_val.csv", index=False)
    print(f"\n=== 1. Fusion with Hybrid E (validation, 203 users; Hybrid E {hybrid.mean():.4f}) ===")
    print(table.round(4).to_string(index=False))
    print(f"Kimi fusion - DeepSeek fusion (w = {FIXED_W}): {kimi_vs_ds['difference']:+.4f} "
          f"(better {kimi_vs_ds['better']}, worse {kimi_vs_ds['worse']}, p = {kimi_vs_ds['p_value']:.3f})")


def ndcg_series(per_user, label_start):
    part = per_user[per_user["model"].str.startswith(label_start)]
    return part.set_index("user_id")["llm_ndcg"]


def run_idea3(data):
    """Experiment 2: Kimi reranks popularity's top 30 (p2, 10 seeds); references from cache."""
    models = [KIMI, DEEPSEEK, (*DEEPSEEK, True), ("ollama", "gemma4:e4b")]
    per_user, summary = lr.compare_models(models, data, split="val", version="p2", n_seeds=10,
                                          candidates="popularity")
    kimi = ndcg_series(per_user, "kimi-k3")
    print("\n=== 2. Idea 3: rerank popularity (validation) ===")
    for name, ref in [("DeepSeek Flash", ndcg_series(per_user, "deepseek-v4.1-flash [")),
                      ("DeepSeek Flash + thinking", ndcg_series(per_user, "deepseek-v4.1-flash (think"))]:
        r = paired(kimi, ref)
        print(f"Kimi K3 - {name}: {r['difference']:+.4f} (better {r['better']}, worse {r['worse']}, "
              f"p = {r['p_value']:.3f})")
    per_user.to_csv(FOLDER / "idea3_per_user_val.csv", index=False)
    summary.to_csv(FOLDER / "idea3_summary_val.csv", index=False)


def run_shuffles(data):
    """Experiment 3: Kimi p2 with the candidate orders 0, 1, 2 (10 seeds, Hybrid E's top 30).
    Per model: NDCG per order, the mean over orders, the NDCG of the rank averaged over the 3 orders,
    and how many of the top 10 films stay the same between two orders."""
    users = cs.users_with_n_seeds(data["val_users"], 10)
    rows, stats = [], []
    for provider, model in (KIMI, DEEPSEEK):
        orders = {}
        for shuffle in (0, 1, 2):
            frame = lr.run_model(provider, model, users, data, version="p2", candidates="hybrid", shuffle=shuffle)
            frame["shuffle"] = shuffle
            rows.append(frame)
            # the model's full order per user, to average ranks and compare top 10s
            path = lr.run_name(provider, model, 10, None, "p2", "hybrid", shuffle)[0]
            cache = {r["user_id"]: r for r in map(lr.json.loads, path.read_text(encoding="utf-8").splitlines())}
            orders[shuffle] = {}
            for user in users:
                top_ids, _ = lr.hybrid_e_candidates(user, data)
                _, number_to_id = lr.build_prompt(user, top_ids, data, "p2", shuffle)
                orders[shuffle][user["user_id"]] = lr.parse_ranking(cache[user["user_id"]]["content"],
                                                                    number_to_id, top_ids)[0]
        frames = pd.concat([r for r in rows if r["model"].iloc[0].startswith(model)])
        by_shuffle = frames.pivot_table(index="user_id", columns="shuffle", values="llm_ndcg")
        averaged, overlap = [], []
        for user in users:
            uid = user["user_id"]
            rank = {}
            for s in (0, 1, 2):
                for i, m in enumerate(orders[s][uid]):
                    rank[m] = rank.get(m, 0) + i + 1
            top_ids, _ = lr.hybrid_e_candidates(user, data)
            avg_order = sorted(top_ids, key=lambda m: (rank[m], top_ids.index(m)))   # ties: Hybrid E order
            averaged.append(lr.bmt.ndcg_at_k(avg_order[:10], user["relevant"], 10))
            overlap += [len(set(orders[a][uid][:10]) & set(orders[b][uid][:10])) for a, b in ((0, 1), (0, 2), (1, 2))]
        hybrid = frames[frames["shuffle"] == 0].set_index("user_id")["hybrid_ndcg"]
        avg = pd.Series(averaged, index=[u["user_id"] for u in users])
        stats.append({"model": model, **{f"order {s}": by_shuffle[s].mean() for s in (0, 1, 2)},
                      "mean of 3 orders": by_shuffle.mean(axis=1).mean(),
                      "spread (max - min order)": by_shuffle.mean().max() - by_shuffle.mean().min(),
                      "rank averaged over 3 orders": avg.mean(),
                      "averaged vs Hybrid E": paired(avg, hybrid)["difference"],
                      "p (averaged vs Hybrid E)": paired(avg, hybrid)["p_value"],
                      "top-10 films shared between two orders": np.mean(overlap)})
    stats = pd.DataFrame(stats)
    pd.concat(rows).to_csv(FOLDER / "shuffles_per_user_val.csv", index=False)
    stats.to_csv(FOLDER / "shuffles_summary_val.csv", index=False)
    print(f"\n=== 3. Candidate order (validation, 10 seeds; Hybrid E {hybrid.mean():.4f}) ===")
    print(stats.round(4).T.to_string())


def run_three_seeds(data):
    """Experiment 4: Kimi p2 with 3 seeds (Hybrid E's top 30 for 3 seeds); gemma4:e4b from cache."""
    per_user, summary = lr.compare_models([KIMI, ("ollama", "gemma4:e4b")], data, split="val",
                                          version="p2", n_seeds=3, candidates="hybrid")
    per_user.to_csv(FOLDER / "three_seeds_per_user_val.csv", index=False)
    summary.to_csv(FOLDER / "three_seeds_summary_val.csv", index=False)
    print("\n=== 4. Three seeds (validation; see the table above for Hybrid E and popularity) ===")


if __name__ == "__main__":
    data = lr.load_data()
    run_fusion(data)
    if ONLY == "fusion":
        sys.exit()
    run_idea3(data)
    run_shuffles(data)
    run_three_seeds(data)
