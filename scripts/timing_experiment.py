'''
Response time of the ranking methods and the constraint layer (Chapter 10, cost and response time).
No LLM calls. Same code as the web app (web/recommender/thesis_engine.py), one machine, CPU only.

    python scripts/timing_experiment.py

One-time costs: loading the data and similarity tables, fitting EASE, training the learned ranker.
Per request: each method's ranking of the whole pool for the first 20 validation users
(10 and 3 seeds), plus the constraint layer with a typical request. Each user is timed 5 times;
the first (warm-up) run is dropped. Output: output/experiments/timing/.
'''

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # shared modules live in src/
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "web"))
FOLDER = ROOT / "output" / "experiments" / "timing"
N_USERS = 20
REPEATS = 5
REQUEST = {"audience": "children", "max_intensity": {"violence": 1}}   # a typical constraint

t0 = time.perf_counter()
from recommender.thesis_engine import get_engine, METHODS  # noqa: E402
engine = get_engine()
load_seconds = time.perf_counter() - t0
cs, ltr = engine.cs, engine.ltr

# one-time: EASE fit (both lambdas) and learned-ranker training (10 seeds), timed again on their own
from recomm import EASE_R_Recommende  # noqa: E402
one_time = [{"step": "load data and similarity tables (whole engine start)", "seconds": load_seconds}]
for lam in (20000, 120000):
    start = time.perf_counter()
    m = EASE_R_Recommende(engine.data["train"], engine.data["movies"], lam)
    m.fit(engine.data["train"])
    one_time.append({"step": f"fit EASE (lambda {lam})", "seconds": time.perf_counter() - start})

train = pd.read_csv(ltr.FOLDER / "features_simulated_g5.csv")
with open(ltr.FOLDER / "step4_chosen_params.json", encoding="utf-8") as f:
    chosen = json.load(f)
for library in ("xgboost", "lightgbm"):
    start = time.perf_counter()
    ltr.fit_ranker(train, engine.ltr_columns, library, chosen[library])
    one_time.append({"step": f"train learned ranker, {library} ({len(train):,} rows)",
                     "seconds": time.perf_counter() - start})
one_time = pd.DataFrame(one_time)
print(one_time.round(2).to_string(index=False), flush=True)

# per request
rows = []
for seeds in (10, 3):
    users = cs.users_with_n_seeds(engine.data["val_users"], seeds)[:N_USERS]
    for user in users:
        user = {**user, "liked": list(user["liked"])[:seeds]}
        engine._with_alpha(user)
        for method in METHODS:
            for r in range(REPEATS):
                start = time.perf_counter()
                ranked = getattr(engine, f"rank_{method}")(user)
                rank_s = time.perf_counter() - start
                start = time.perf_counter()
                films = [{"movie_id": m, **engine.catalog.loc[m].to_dict()} for m in ranked]
                kept, removed = engine.apply_constraints(films, REQUEST)
                filter_s = time.perf_counter() - start
                if r > 0:
                    rows.append({"seeds": seeds, "user_id": user["user_id"], "method": method,
                                 "rank_s": rank_s, "filter_s": filter_s, "ranked": len(ranked)})
    print(f"{seeds} seeds done", flush=True)

per = pd.DataFrame(rows)
summary = (per.groupby(["method", "seeds"])
           .agg(rank_median_ms=("rank_s", lambda s: 1000 * s.median()),
                rank_p95_ms=("rank_s", lambda s: 1000 * s.quantile(0.95)),
                filter_median_ms=("filter_s", lambda s: 1000 * s.median()),
                films_ranked=("ranked", "median"))
           .reset_index())
FOLDER.mkdir(parents=True, exist_ok=True)
one_time.to_csv(FOLDER / "one_time.csv", index=False)
per.to_csv(FOLDER / "per_request.csv", index=False)
summary.to_csv(FOLDER / "summary.csv", index=False)
print(summary.round(1).to_string(index=False))
import platform  # noqa: E402
print("machine:", platform.processor(), "| python", platform.python_version())
