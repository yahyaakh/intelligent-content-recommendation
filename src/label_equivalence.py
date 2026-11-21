"""
Equivalence check for the LLM-label effect (Chapter 5).

Reads the existing per-user results of label_set_experiment.py (no new recommendations, no LLM calls)
and asks: is the effect of adding labels small enough to call it "no meaningful effect"?

For each comparison (labels vs no labels, DeepSeek vs gemma labels) and each N:
  - mean paired difference in NDCG@10 (b - a)
  - percentile bootstrap CI over users (10,000 resamples, seed 42)
  - the CI at 95% and the Bonferroni-adjusted level (1 - 0.05/m, m = comparisons in the family)
  - equivalent within +-0.01 (and +-0.02) if the Bonferroni-adjusted CI lies inside the margin

Equivalence through a two-sided (1 - alpha) CI inside the margin is conservative (TOST needs only 1 - 2*alpha).
The +-0.02 margin is the one pre-registered for the LLM rerankers; +-0.01 is the stricter one.

    python src/label_equivalence.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

FOLDER = Path("output/experiments/label_sets")
B = 10_000
SEED = 42
MARGINS = (0.01, 0.02)

# (family, a = reference, b = with labels); hybrids and pure methods use the scorings of Chapter 5
COLD = [
    ("hybrid", ("hybrid_C_base", "mean"), ("hybrid_D_v4", "mean"), "Embedding hybrid: gemma vs none"),
    ("hybrid", ("hybrid_C_base", "mean"), ("hybrid_D_v5", "mean"), "Embedding hybrid: DeepSeek vs none"),
    ("hybrid", ("hybrid_D_v4", "mean"), ("hybrid_D_v5", "mean"), "Embedding hybrid: DeepSeek vs gemma"),
    ("hybrid", ("hybrid_A_base", "mean"), ("hybrid_B_v4", "mean"), "Content hybrid: gemma vs none"),
    ("hybrid", ("hybrid_A_base", "mean"), ("hybrid_B_v5", "mean"), "Content hybrid: DeepSeek vs none"),
    ("hybrid", ("hybrid_B_v4", "mean"), ("hybrid_B_v5", "mean"), "Content hybrid: DeepSeek vs gemma"),
    ("pure", ("C_base", "mean"), ("D_v4", "mean"), "Embedding: gemma vs none"),
    ("pure", ("C_base", "mean"), ("D_v5", "mean"), "Embedding: DeepSeek vs none"),
    ("pure", ("D_v4", "mean"), ("D_v5", "mean"), "Embedding: DeepSeek vs gemma"),
    ("pure", ("A_base", "max"), ("B_v4", "max"), "Content: gemma vs none"),
    ("pure", ("A_base", "max"), ("B_v5", "max"), "Content: DeepSeek vs none"),
    ("pure", ("B_v4", "max"), ("B_v5", "max"), "Content: DeepSeek vs gemma"),
]
COLD_N = (3, 10)

WARM = [
    ("warm", ("C_base", "max"), ("D_v4", "max"), "Embedding: gemma vs none"),
    ("warm", ("C_base", "max"), ("D_v5", "max"), "Embedding: DeepSeek vs none"),
    ("warm", ("D_v4", "max"), ("D_v5", "max"), "Embedding: DeepSeek vs gemma"),
    ("warm", ("A_base", "max"), ("B_v4", "max"), "Content: gemma vs none"),
    ("warm", ("A_base", "max"), ("B_v5", "max"), "Content: DeepSeek vs none"),
    ("warm", ("B_v4", "max"), ("B_v5", "max"), "Content: DeepSeek vs gemma"),
]


def paired_diff(df, a, b):
    """Per-user NDCG@10 of b minus a, aligned on user_id."""
    left = df[(df["method"] == a[0]) & (df["scoring"] == a[1])].set_index("user_id")["ndcg"]
    right = df[(df["method"] == b[0]) & (df["scoring"] == b[1])].set_index("user_id")["ndcg"]
    assert left.index.is_unique and right.index.is_unique
    assert set(left.index) == set(right.index), f"{a} and {b} have different users"
    return (right - left.loc[right.index]).to_numpy()


def bootstrap_means(diff, rng):
    idx = rng.integers(0, len(diff), size=(B, len(diff)))
    return diff[idx].mean(axis=1)


def analyse(rows, df, setting, n, m, rng):
    out = []
    for family, a, b, name in rows:
        d = paired_diff(df, a, b)
        boot = bootstrap_means(d, rng)
        adj = 0.05 / m
        lo95, hi95 = np.percentile(boot, [2.5, 97.5])
        lo_adj, hi_adj = np.percentile(boot, [100 * adj / 2, 100 * (1 - adj / 2)])
        rec = {"setting": setting, "n": n, "family": family, "comparison": name,
               "a": f"{a[0]} ({a[1]})", "b": f"{b[0]} ({b[1]})", "users": len(d),
               "mean_diff": d.mean(), "ci95_low": lo95, "ci95_high": hi95,
               "m": m, "ci_bonf_low": lo_adj, "ci_bonf_high": hi_adj}
        for margin in MARGINS:
            rec[f"equivalent_{margin}"] = bool(lo_adj > -margin and hi_adj < margin)
        out.append(rec)
    return out


def main():
    rng = np.random.default_rng(SEED)
    cold = pd.read_csv(FOLDER / "cold_per_user.csv")
    warm = pd.read_csv(FOLDER / "warm_per_user.csv")

    results = []
    m_cold = len(COLD) * len(COLD_N)            # one family: every cold comparison at every N
    for n in COLD_N:
        results += analyse(COLD, cold[cold["n"] == n], "cold (814 test users)", n, m_cold, rng)
    results += analyse(WARM, warm, "warm (75/25 split)", "-", len(WARM), rng)

    table = pd.DataFrame(results)
    table.to_csv(FOLDER / "equivalence.csv", index=False)

    show = table.copy()
    for c in ["mean_diff", "ci95_low", "ci95_high", "ci_bonf_low", "ci_bonf_high"]:
        show[c] = show[c].map(lambda x: f"{x:+.4f}")
    cols = ["setting", "n", "comparison", "mean_diff", "ci_bonf_low", "ci_bonf_high",
            "equivalent_0.01", "equivalent_0.02"]
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(show[cols].to_string(index=False))
    print(f"\nSaved {FOLDER / 'equivalence.csv'}  (cold family m = {m_cold}, warm m = {len(WARM)})")


if __name__ == "__main__":
    main()
