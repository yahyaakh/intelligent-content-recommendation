'''
Verification of the LLM reranking results (llm_rerank_experiment.py).

For each model and setting, every number in the results table is rebuilt from scratch and
checked independently:

1. NDCG@10 is recomputed with a separate implementation (written out below, not bmt.ndcg_at_k)
   for Hybrid E, the LLM ranking and the oracle, and compared with the saved per-user results.
2. Integrity of every user's lists:
   - the 30 candidates contain no seed / seen film and no duplicates;
   - the LLM ranking is a permutation of exactly those 30 candidates;
   - the oracle is >= both Hybrid E and the LLM for every user.
3. The prompt contains nothing from the answer key except as an unlabelled candidate, and no
   ratings or scores.
4. Position bias: does the LLM simply keep the (shuffled) order it was shown?
5. Side-by-side top 10 of Hybrid E, the LLM and the oracle for a few users, with every hit
   marked, written to a Markdown report.

Usage:
    python experiments/llm/verify_llm_rerank.py                                   # DeepSeek Flash, p2, N = 10
    python experiments/llm/verify_llm_rerank.py ollama_cloud deepseek-v4.1-flash p1 3
    python experiments/llm/verify_llm_rerank.py all                               # every complete idea-1 cache, one table
    python experiments/llm/verify_llm_rerank.py idea3                             # every idea-3 run (popularity candidates)
'''

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))  # shared modules live in src/

import json
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys_args = sys.argv[1:]
sys.argv = [sys.argv[0]]                    # llm_rerank_experiment reads sys.argv at import
import cold_start_experiment as cs
import llm_rerank_experiment as lr

EXAMPLE_USERS = 5
K = 10

# Every (provider, model, prompt version, N) with answers for all 203 validation users
ALL_RUNS = [
    *[(p, m, "p1", 3) for p, m in [
        ("ollama", "gemma4:e4b"), ("ollama", "qwen3.5:9b"), ("ollama", "llama3.1:8b"),
        ("ollama_cloud", "gemma4:31b-cloud"), ("ollama_cloud", "gpt-oss:20b-cloud"),
        ("ollama_cloud", "gpt-oss:120b-cloud"), ("ollama_cloud", "glm-5.3-flash"),
        ("ollama_cloud", "deepseek-v4.1-flash")]],
    ("ollama", "gemma4:e4b", "p1", 5), ("ollama", "gemma4:e4b", "p1", 10),
    ("ollama", "gemma4:e4b", "p2", 3), ("ollama", "gemma4:e4b", "p2", 5), ("ollama", "gemma4:e4b", "p2", 10),
    ("ollama_cloud", "glm-5.3-flash", "p2", 10), ("ollama_cloud", "deepseek-v4.1-flash", "p2", 10),
]


# Idea 3 (popularity candidates): (provider, model, version, N, split, think)
IDEA3_RUNS = [
    *[(p, m, "p2", 10, "val", None) for p, m in [
        ("ollama", "gemma4:e4b"), ("ollama_cloud", "deepseek-v4.1-flash"), ("ollama_cloud", "mistral-large-4"),
        ("ollama_cloud", "deepseek-v4-pro:0813"), ("ollama_cloud", "glm-5.3-flash"),
        ("ollama_cloud", "nemotron-3-super"), ("ollama_cloud", "glm-5.2")]],
    ("ollama_cloud", "deepseek-v4.1-flash", "p2", 10, "val", True),
    ("ollama", "gemma4:e4b", "p1", 3, "val", None), ("ollama_cloud", "deepseek-v4.1-flash", "p1", 3, "val", None),
    ("ollama_cloud", "deepseek-v4.1-flash", "p2", 10, "test", None),
]


def my_ndcg(recommended, relevant, k=K):
    """Independent NDCG@k with binary relevance: sum of 1/log2(position + 1) over hits,
    divided by the same sum for min(|relevant|, k) hits at the top."""
    dcg = sum(1.0 / np.log2(position + 1)
              for position, movie_id in enumerate(recommended[:k], start=1) if movie_id in relevant)
    ideal = sum(1.0 / np.log2(position + 1) for position in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal else 0.0


def cache_file(provider, model, version, n_seeds, candidates="hybrid", think=None):
    return lr.run_name(provider, model, n_seeds, think, version, candidates)[0]


def verify(data, provider, model, version, n_seeds, write_examples=True, verbose=True, split="val",
           candidates="hybrid", think=None):
    """Rebuild and check one model and setting on the validation or test users; returns one summary row."""
    users = cs.users_with_n_seeds(data[f"{split}_users"], n_seeds)
    cache_path, label = lr.run_name(provider, model, n_seeds, think, version, candidates)
    answers = {r["user_id"]: r for r in map(json.loads, cache_path.read_text(encoding="utf-8").splitlines())}
    title = lambda m: f"{data['movie_by_id'].loc[m, 'title']} ({data['movie_by_id'].loc[m, 'year']})"

    rows, problems, examples = [], [], []
    for index, user in enumerate(users):
        uid, relevant = user["user_id"], user["relevant"]
        if uid not in answers:
            problems.append(f"user {uid}: no cached answer")
            continue
        top_ids, _ = lr.get_candidates(user, data, candidates)
        messages, number_to_id = lr.build_prompt(user, top_ids, data, version)
        llm_order, issues = lr.parse_ranking(answers[uid]["content"], number_to_id, top_ids)
        oracle = [m for m in top_ids if m in relevant] + [m for m in top_ids if m not in relevant]

        # 2. Integrity
        if set(top_ids) & set(user["seen"]):
            problems.append(f"user {uid}: a seen film is among the candidates")
        if len(set(top_ids)) != lr.TOP_M:
            problems.append(f"user {uid}: duplicate candidates")
        if sorted(llm_order) != sorted(top_ids):
            problems.append(f"user {uid}: the LLM ranking is not a permutation of the candidates")

        # 3. The prompt shows candidates without any label, rating or score
        prompt = messages[1]["content"].lower()
        for word in ("rating", "rated", "relevant", "score", "popular"):
            if word in prompt:
                problems.append(f"user {uid}: the prompt contains '{word}'")

        # 4. Position bias: shown number (1..30) vs position in the LLM ranking
        shown_number = {movie_id: number for number, movie_id in number_to_id.items()}
        rho = spearmanr([shown_number[m] for m in llm_order], range(lr.TOP_M)).correlation

        h, l, o = my_ndcg(top_ids, relevant), my_ndcg(llm_order, relevant), my_ndcg(oracle, relevant)
        if o + 1e-12 < max(h, l):
            problems.append(f"user {uid}: oracle below another method")
        rows.append({"user_id": uid, "hybrid": h, "llm": l, "oracle": o,
                     "hits_in_30": sum(m in relevant for m in top_ids),
                     "hybrid_hits": sum(m in relevant for m in top_ids[:K]),
                     "llm_hits": sum(m in relevant for m in llm_order[:K]),
                     "issues": len(issues), "rho_shown_order": rho})
        if index < EXAMPLE_USERS:
            examples.append((user, top_ids, llm_order, oracle, h, l, o))

    res = pd.DataFrame(rows).set_index("user_id")

    # 1. Compare with the saved results of llm_rerank_experiment.py
    saved = pd.read_csv(lr.FOLDER / f"compare_per_user_{split}_{len(users)}users_{version}_n{n_seeds}"
                                    f"{'' if candidates == 'hybrid' else '_cpop'}.csv")
    saved = saved[saved["model"] == label].set_index("user_id")
    if len(saved) == len(res):
        saved = saved.loc[res.index]
        # "hybrid" here is the candidate list's own top 10: Hybrid E (idea 1) or popularity (idea 3)
        baseline_column = "hybrid_ndcg" if candidates == "hybrid" else "popularity_ndcg"
        diff = max((saved["llm_ndcg"] - res["llm"]).abs().max(), (saved[baseline_column] - res["hybrid"]).abs().max())
        saved_llm = saved["llm_ndcg"].mean()
    else:
        diff, saved_llm = np.nan, np.nan
        problems.append(f"saved per-user file has {len(saved)} rows for '{label}', expected {len(res)}")

    summary = {"split": split, "candidates": candidates, "model": label, "version": version, "n_seeds": n_seeds, "users": len(res),
               "candidates_top10": res["hybrid"].mean(), "llm": res["llm"].mean(), "saved_llm": saved_llm,
               "max_ndcg_diff": diff, "oracle": res["oracle"].mean(),
               "hits_in_30": res["hits_in_30"].mean(), "hits_top10_hybrid": res["hybrid_hits"].mean(),
               "hits_top10_llm": res["llm_hits"].mean(),
               "better": int((res["llm"] > res["hybrid"]).sum()), "worse": int((res["llm"] < res["hybrid"]).sum()),
               "repaired": int((res["issues"] > 0).sum()), "position_bias": res["rho_shown_order"].mean(),
               "problems": len(problems)}

    if verbose:
        print(f"\n{provider} / {model}, prompt {version}, N = {n_seeds}: {len(res)} users")
        print(f"   NDCG candidates' top 10 {summary['candidates_top10']:.4f} | LLM {summary['llm']:.4f} (saved {saved_llm:.4f}) | "
              f"oracle {summary['oracle']:.4f} | largest difference to saved {diff:.1e}")
        print(f"   integrity / prompt checks: {'all passed' if not problems else f'{len(problems)} problems'}; "
              f"position bias {summary['position_bias']:.3f}")
        for p in problems[:5]:
            print("     ", p)

    if write_examples:
        report = [f"# Reranking verification: {model}, prompt {version}, N = {n_seeds}\n",
                  f"First {EXAMPLE_USERS} validation users. ✓ = in the user's answer key (a hit).\n"]
        for user, top_ids, llm_order, oracle, h, l, o in examples:
            report.append(f"\n## User {user['user_id']}\n")
            report.append("Seeds: " + "; ".join(title(m) for m in user["liked"]) + "\n")
            report.append(f"\n| # | Hybrid E (NDCG {h:.3f}) | {model} (NDCG {l:.3f}) | Oracle (NDCG {o:.3f}) |")
            report.append("|---|---|---|---|")
            for i in range(K):
                cell = lambda m: f"{'✓ ' if m in user['relevant'] else ''}{title(m)}"
                report.append(f"| {i + 1} | {cell(top_ids[i])} | {cell(llm_order[i])} | {cell(oracle[i])} |")
        out = lr.FOLDER / (f"verification_{split}_{model.replace(':', '_')}{'' if think is None else f'_think-{think}'}"
                           f"_{version}_n{n_seeds}{'' if candidates == 'hybrid' else '_cpop'}.md")
        out.write_text("\n".join(report), encoding="utf-8")
        res.to_csv(out.with_suffix(".csv"))
    return summary


if __name__ == "__main__":
    data = lr.load_data()
    if sys_args and sys_args[0] == "idea3":
        # Rebuild the saved per-user tables from the caches first (no model calls), so every run
        # of a setting is in one file, then verify each run against it
        groups = {}
        for provider, model, version, n, split, think in IDEA3_RUNS:
            entry = (provider, model) if think is None else (provider, model, think)
            groups.setdefault((version, n, split), []).append(entry)
        for (version, n, split), models in groups.items():
            lr.compare_models(models, data, split=split, version=version, n_seeds=n, candidates="popularity")
        table = pd.DataFrame([verify(data, p, m, v, n, split=sp, candidates="popularity", think=t)
                              for p, m, v, n, sp, t in IDEA3_RUNS])
        table.to_csv(lr.FOLDER / "verification_summary_idea3.csv", index=False)
        pd.set_option("display.width", 250)
        print("\nVerification summary, idea 3 (popularity candidates):")
        print(table.drop(columns=["candidates"]).round(4).to_string(index=False))
    elif sys_args and sys_args[0] == "all":
        table = pd.DataFrame([verify(data, *run) for run in ALL_RUNS])
        table.to_csv(lr.FOLDER / "verification_summary.csv", index=False)
        pd.set_option("display.width", 250)
        print("\nVerification summary (203 validation users per row):")
        print(table.round(4).to_string(index=False))
        print(f"\nSaved {lr.FOLDER / 'verification_summary.csv'}")
    else:
        verify(data,
               sys_args[0] if len(sys_args) > 0 else "ollama_cloud",
               sys_args[1] if len(sys_args) > 1 else "deepseek-v4.1-flash",
               sys_args[2] if len(sys_args) > 2 else "p2",
               int(sys_args[3]) if len(sys_args) > 3 else 10)
