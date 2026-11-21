'''
Method A of doc 17: score fusion of an LLM's reranking with Hybrid E.

The LLM reranking of idea 1 (doc 13, prompt p2, N = 10, Hybrid E's top 30) replaces Hybrid E's order
completely. Here the two orders are combined instead, so the LLM can only move films up or down
relative to Hybrid E. Uses the cached answers only (no model calls). Validation users only.

Methods per LLM:
  rank blend (CV)   combined = w x Hybrid E rank + (1 - w) x LLM rank (lower = better), w in 0, 0.05, ..., 1
                    chosen by 5-fold cross-validation over users (every user is scored with a w that
                    never saw them); w = 1 is Hybrid E, w = 0 is the LLM alone
  score blend (CV)  combined = w x Hybrid E score (rescaled 0-1 per user) + (1 - w) x LLM position score
                    (1 for the LLM's first film, 0 for its last); w by the same cross-validation
  RRF               reciprocal rank fusion, 1/(60 + Hybrid E rank) + 1/(60 + LLM rank), no tuning
Ties are broken by Hybrid E's order.

Controls: the same fusions with three non-LLM orders (random, averaged over 10 orders; popularity;
content similarity emb_max). If they gain as much, the gain is not specific to the LLM.

Tests: each fusion vs Hybrid E (paired Wilcoxon, Bonferroni over all models, controls and methods).

Usage:
    python src/llm_fusion_experiment.py
'''
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.argv = [sys.argv[0]]
import build_movie_texts as bmt
import cold_start_experiment as cs
import llm_rerank_experiment as lr

FOLDER = Path("output") / "experiments" / "llm_fusion"
FOLDER.mkdir(parents=True, exist_ok=True)
N_SEEDS, VERSION = 10, "p2"
LLMS = [("ollama_cloud", "deepseek-v4.1-flash"), ("ollama_cloud", "glm-5.3-flash"),
        ("ollama", "gemma4:e4b"), ("ollama_cloud", "mistral-large-4"),
        ("ollama_cloud", "deepseek-v4-pro:0813"), ("ollama_cloud", "nemotron-3-super")]
WEIGHTS = np.round(np.arange(0, 1.0001, 0.05), 2)
FOLDS, RRF_K = 5, 60


def load_answers(provider, model):
    path = lr.run_name(provider, model, N_SEEDS, None, VERSION, "hybrid")[0]
    return {r["user_id"]: r for r in map(json.loads, path.read_text(encoding="utf-8").splitlines())}


CONTROLS = ["control: random", "control: popularity", "control: content (emb_max)"]
RANDOM_REPEATS = 10       # the random control is averaged over 10 random orders


def ranks_of(values, higher_is_better=True):
    """1 = best; ties by Hybrid E's order."""
    order = np.argsort(-values if higher_is_better else values, kind="stable")
    rank = np.empty(len(values))
    rank[order] = np.arange(1, len(values) + 1)
    return rank


def user_inputs(users, data):
    """Per user: Hybrid E's 30 candidates (best first), their rescaled Hybrid E scores, each LLM's rank,
    and the ranks of three non-LLM control orders (does fusing with ANY other order help?)."""
    answers = {model: load_answers(provider, model) for provider, model in LLMS}
    emb = bmt.embedding_similarity("movies_base_embeddinggemma_v2", data["movies"])
    position_of = data["position_of"]
    rng = np.random.default_rng(42)
    out = []
    for user in users:
        top_ids, scores = lr.hybrid_e_candidates(user, data)
        _, number_to_id = lr.build_prompt(user, top_ids, data, VERSION)      # the shuffle of the cached run
        llm_rank = {}
        for _, model in LLMS:
            order, _ = lr.parse_ranking(answers[model][user["user_id"]]["content"], number_to_id, top_ids)
            position = {m: i + 1 for i, m in enumerate(order)}
            llm_rank[model] = np.array([position[m] for m in top_ids], float)
        cands = [position_of[m] for m in top_ids]
        seeds = [position_of[m] for m in user["liked"]]
        llm_rank["control: popularity"] = ranks_of(data["pop_vector"][cands])
        llm_rank["control: content (emb_max)"] = ranks_of(emb[np.ix_(seeds, cands)].max(axis=0))
        for r in range(RANDOM_REPEATS):
            llm_rank[f"control: random {r}"] = rng.permutation(len(top_ids)).astype(float) + 1
        out.append({"user_id": user["user_id"], "ids": np.array(top_ids), "relevant": user["relevant"],
                    "hybrid_rank": np.arange(1, len(top_ids) + 1, dtype=float),
                    "hybrid_score": cs.rescale(np.asarray(scores, float)), "llm_rank": llm_rank})
    return out


def ndcg_of(u, combined, lower_is_better):
    """NDCG@10 of one user's candidates sorted by `combined`; ties keep Hybrid E's order (stable sort)."""
    order = np.argsort(combined if lower_is_better else -combined, kind="stable")
    return bmt.ndcg_at_k(u["ids"][order[:bmt.K]].tolist(), u["relevant"], bmt.K)


def fused(u, model, method, w=None):
    llm = u["llm_rank"][model]
    if method == "rank blend":
        return ndcg_of(u, w * u["hybrid_rank"] + (1 - w) * llm, True)
    if method == "score blend":
        llm_score = 1 - (llm - 1) / (len(llm) - 1)
        return ndcg_of(u, w * u["hybrid_score"] + (1 - w) * llm_score, False)
    return ndcg_of(u, 1 / (RRF_K + u["hybrid_rank"]) + 1 / (RRF_K + llm), False)      # RRF


def cross_validated(inputs, model, method):
    """Per-user NDCG with w chosen on the other folds; also the chosen w per fold."""
    ids = np.array([u["user_id"] for u in inputs])
    fold_of = dict(zip(np.random.default_rng(42).permutation(np.sort(ids)), np.arange(len(ids)) % FOLDS))
    table = np.array([[fused(u, model, method, w) for w in WEIGHTS] for u in inputs])   # users x weights
    folds = np.array([fold_of[i] for i in ids])
    scores, chosen = np.empty(len(inputs)), []
    for k in range(FOLDS):
        best = WEIGHTS[np.argmax(table[folds != k].mean(axis=0))]
        chosen.append(float(best))
        scores[folds == k] = table[folds == k, list(WEIGHTS).index(best)]
    return pd.Series(scores, index=ids), chosen, pd.Series(table.mean(axis=0), index=WEIGHTS)


if __name__ == "__main__":
    data = lr.load_data()
    users = cs.users_with_n_seeds(data["val_users"], N_SEEDS)
    inputs = user_inputs(users, data)
    ids = [u["user_id"] for u in inputs]
    hybrid = pd.Series([ndcg_of(u, u["hybrid_rank"], True) for u in inputs], index=ids)
    print(f"Hybrid E (validation, {len(ids)} users): {hybrid.mean():.4f}  (expected 0.6286)")

    per_user, summary, curves, tests = {"Hybrid E": hybrid}, [], {}, []
    for model in [m for _, m in LLMS] + CONTROLS:
        names = [f"control: random {r}" for r in range(RANDOM_REPEATS)] if model == "control: random" else [model]

        def averaged(fn):
            """Per-user mean over the random repeats (or the single order)."""
            return pd.concat([fn(name) for name in names], axis=1).mean(axis=1)

        alone = averaged(lambda n: pd.Series([ndcg_of(u, u["llm_rank"][n], True) for u in inputs], index=ids))
        per_user[f"{model} alone"] = alone
        row = {"model": model, "LLM alone": alone.mean()}
        for method in ("rank blend", "score blend", "RRF"):
            if method == "RRF":
                scores = averaged(lambda n: pd.Series([fused(u, n, "RRF") for u in inputs], index=ids))
                chosen = None
            else:
                runs = [cross_validated(inputs, n, method) for n in names]
                scores = pd.concat([r[0] for r in runs], axis=1).mean(axis=1)
                chosen = runs[0][1] if len(runs) == 1 else "varies"
                curves[f"{model} | {method}"] = pd.concat([r[2] for r in runs], axis=1).mean(axis=1)
            per_user[f"{model} {method}"] = scores
            row[method] = scores.mean()
            if chosen is not None:
                row[f"{method} w per fold"] = chosen
            diff = scores - hybrid
            tests.append({"model": model, "method": method, "difference_vs_hybrid": diff.mean(),
                          "better": int((diff > 0).sum()), "worse": int((diff < 0).sum()),
                          "p_value": wilcoxon(diff).pvalue if diff.any() else 1.0})
        summary.append(row)

    tests = pd.DataFrame(tests)
    tests["p_bonferroni"] = (tests["p_value"] * len(tests)).clip(upper=1.0)
    summary = pd.DataFrame(summary)
    pd.DataFrame(per_user).to_csv(FOLDER / "per_user_val.csv")
    summary.to_csv(FOLDER / "summary_val.csv", index=False)
    tests.to_csv(FOLDER / "tests_val.csv", index=False)
    pd.DataFrame(curves).to_csv(FOLDER / "weight_curves_val.csv", index_label="w (1 = Hybrid E)")
    pd.set_option("display.width", 250)
    print(summary.round(4).to_string(index=False))
    print(tests.round(4).to_string(index=False))
    print("NDCG@10 by weight on all validation users (descriptive; w = 1 is Hybrid E):")
    print(pd.DataFrame(curves).loc[[0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]].round(4).to_string())
