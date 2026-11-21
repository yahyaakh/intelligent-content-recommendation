'''
Checks that the web app's thesis engine gives the report's validation numbers (NDCG@10, 203 users).
No LLM calls. Run from the project root:
    python web/check_thesis_engine.py
'''
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from recommender.thesis_engine import get_engine

EXPECTED = {(10, "hybrid_e"): 0.6286, (3, "hybrid_e"): 0.5837, (10, "ltr"): 0.6686, (3, "ltr"): 0.6227,
            (10, "ease"): 0.6514, (3, "ease"): 0.6055, (10, "popularity"): 0.5626}

engine = get_engine()
import build_movie_texts as bmt

ok = True
for n in (10, 3):
    users = engine.cs.users_with_n_seeds(engine.data["val_users"], n)
    for method in ("popularity", "hybrid_e", "ease", "ltr"):
        if (n, method) not in EXPECTED:
            continue
        scores = []
        for user in users:
            engine._with_alpha(user)
            ranked = getattr(engine, f"rank_{method}")(user)[:bmt.K]
            scores.append(bmt.user_metrics(ranked, user["relevant"], bmt.K)["ndcg"])
        got = sum(scores) / len(scores)
        expected = EXPECTED[(n, method)]
        status = "ok" if abs(got - expected) < 5e-4 else "DIFFERENT"
        ok &= status == "ok"
        print(f"{n:2d} seeds  {method:10s}  {got:.4f}  (report {expected})  {status}", flush=True)
print("all checks passed" if ok else "SOME CHECKS FAILED")
