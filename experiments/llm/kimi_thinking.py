'''
Doc 17, the open lead: does reasoning over the collaborative data (prompt p5) let an LLM beat Hybrid E?
Kimi K3 with thinking on, idea 1 (Hybrid E's top 30), p5, 10 seeds, validation users.
References from cache (no new calls): Kimi K3 p5 without thinking, DeepSeek Flash p5, Kimi K3 p2.

Answers are cached (answers_ollama_cloud_kimi-k3_think-True_p5_n10.jsonl): a stopped run resumes.

Usage:
    python experiments/llm/kimi_thinking.py          pilot: the first 20 validation users
    python experiments/llm/kimi_thinking.py full     all 203 validation users
'''

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))  # shared modules live in src/
import sys

FULL = len(sys.argv) > 1 and sys.argv[1] == "full"
sys.argv = [sys.argv[0]]
import pandas as pd
from scipy.stats import wilcoxon

import cold_start_experiment as cs
import llm_rerank_experiment as lr

THINKING = ("ollama_cloud", "kimi-k3", True)
CACHED = [("ollama_cloud", "kimi-k3"), ("ollama_cloud", "deepseek-v4.1-flash")]
SETTING = dict(split="val", n_seeds=10, candidates="hybrid")


def by_user(per_user, start):
    part = per_user[per_user["model"].str.startswith(start)]
    return part.set_index("user_id")["llm_ndcg"]


if __name__ == "__main__":
    data = lr.load_data()
    n_users = None if FULL else 20
    p5, summary = lr.compare_models([THINKING] + CACHED, data, n_users=n_users, version="p5", **SETTING)
    p2, _ = lr.compare_models([CACHED[0]], data, n_users=n_users, version="p2", **SETTING)   # cached

    think = by_user(p5, "kimi-k3 (think")
    print()
    for name, ref in [("Kimi K3 p5, no thinking", by_user(p5, "kimi-k3 [")),
                      ("DeepSeek Flash p5", by_user(p5, "deepseek-v4.1-flash")),
                      ("Kimi K3 p2, no thinking", by_user(p2, "kimi-k3"))]:
        diff = think - ref.loc[think.index]
        p = wilcoxon(diff).pvalue if diff.any() else 1.0
        print(f"Kimi K3 p5 + thinking - {name}: {diff.mean():+.4f} "
              f"(better {(diff > 0).sum()}, worse {(diff < 0).sum()}, p = {p:.3f})")
    row = summary[summary["model"].str.startswith("kimi-k3 (think")].iloc[0]
    print(f"\nThinking: {row['tokens_out']:.0f} output tokens and {row['seconds']:.0f} s per user "
          f"-> about {203 * row['seconds'] / 3600:.1f} h for all 203 users")
