"""Reviewer check: does the EASE comparison change if lambda is chosen on the 203 validation users
instead of the simulated warm users? Same data, users, seeds and metric code as final_comparison.py;
only lambda varies. Validation users only (test users are not touched).
Run from the project folder."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))  # shared modules live in src/
import sys
import time
from pathlib import Path

sys.argv = [sys.argv[0]]
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import final_comparison as fc
import cold_start_experiment as cs
import build_movie_texts as bmt
from recomm import EASE_R_Recommende

LAMBDAS = [500, 1000, 2000, 3000, 5000, 7000, 10000, 15000, 20000, 30000, 50000, 80000, 120000, 200000]
OUT = Path("output") / "experiments" / "ease"
K = bmt.K

data = fc.load()
saved = pd.read_csv(Path("output") / "experiments" / "final_comparison" / "per_user_val.csv")

rows = []
for lam in LAMBDAS:
    start = time.time()
    ease = EASE_R_Recommende(data["train"], data["movies"], lam)
    ease.fit(data["train"])
    for n in (10, 3):
        for user in cs.users_with_n_seeds(data["val_users"], n):
            rec = [int(m) for m in ease.recommend(user["liked"], K, exclude=user["seen"])]
            rows.append({"lambda": lam, "seeds": n, "user_id": user["user_id"],
                         **bmt.user_metrics(rec, user["relevant"], K)})
    part = pd.DataFrame([r for r in rows if r["lambda"] == lam]).groupby("seeds")["ndcg"].mean()
    print(f"lambda {lam:>6}: 10 seeds {part[10]:.4f} | 3 seeds {part[3]:.4f} ({time.time() - start:.0f} s)",
          flush=True)

per_user = pd.DataFrame(rows)
curve = per_user.groupby(["seeds", "lambda"])["ndcg"].mean().unstack("seeds")
print("\nValidation NDCG@10 by lambda:\n", curve.round(4).to_string())

# reproduce the leaderboard value at the lambda chosen on simulated users
for n, lam in fc.EASE_LAMBDAS.items():
    mine = curve.loc[lam, n]
    lead = saved[(saved.seeds == n) & (saved.method == f"EASE (lambda {lam:,})")]["ndcg"].mean()
    print(f"check {n} seeds, lambda {lam}: {mine:.4f} vs leaderboard {lead:.4f}",
          "ok" if abs(mine - lead) < 5e-4 else "DIFFERENT")

# best lambda on validation users vs the reference methods (same users, paired)
results = []
for n in (10, 3):
    best = curve[n].idxmax()
    chosen = fc.EASE_LAMBDAS[n]
    ease_best = per_user[(per_user.seeds == n) & (per_user["lambda"] == best)].set_index("user_id")["ndcg"]
    ease_chosen = per_user[(per_user.seeds == n) & (per_user["lambda"] == chosen)].set_index("user_id")["ndcg"]
    ref = saved[saved.seeds == n].pivot(index="user_id", columns="method", values="ndcg")
    for label, scores in (("EASE, lambda chosen on simulated users", ease_chosen),
                          ("EASE, lambda chosen on validation users", ease_best)):
        for other in ("Hybrid E", "User-kNN (standard: similarity-weighted sum)", "LTR v2 (XGBoost + LightGBM)"):
            diff = (scores - ref[other]).dropna()
            results.append({"seeds": n, "ease": label, "lambda": chosen if "simulated" in label else best,
                            "ease_ndcg": scores.mean(), "vs": other, "other_ndcg": ref[other].mean(),
                            "difference": diff.mean(), "p": wilcoxon(diff).pvalue if diff.any() else 1.0})
results = pd.DataFrame(results)
pd.set_option("display.width", 250)
print("\n", results.round(4).to_string(index=False))

OUT.mkdir(parents=True, exist_ok=True)
curve.to_csv(OUT / "val_lambda_check_curve.csv")
results.to_csv(OUT / "val_lambda_check_comparison.csv", index=False)
print(f"\nSaved {OUT / 'val_lambda_check_curve.csv'} and val_lambda_check_comparison.csv")
