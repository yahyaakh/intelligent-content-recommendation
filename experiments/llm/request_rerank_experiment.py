"""
Typed request -> LLM reranks the popularity top 30.

Section 6.1 used the request only through embeddings or extracted labels. Here the LLM itself
reads the request and ranks the 30 most popular unseen films, the setting that works best in
Sanner et al. 2023, He et al. 2023 and Eberhard et al. 2024.

Same users, candidates, shuffling, answer parsing and scoring as the seed-based popularity
reranking of Section 6.4 (llm_rerank_experiment.py, candidates="popularity"); only the user part
of the prompt changes. The requests are the existing simulated ones (typed_requests/requests_r1.jsonl,
general style, written by gemma4:e4b from the first 3 or 10 seed films).

Runs (203 validation users, DeepSeek V4.1 Flash, no reasoning):
  request, 3 seeds      -> the request written from 3 seeds
  request, 10 seeds     -> the request written from 10 seeds
  seeds p2, 3 seeds     -> the seed-based p2 prompt with 3 seeds (fair partner, same candidate detail)
The seed-based p2 run with 10 seeds already exists (cached).

Usage:  python experiments/llm/request_rerank_experiment.py [pilot|full]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))  # shared modules live in src/
import json
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from tqdm import tqdm

sys_argv = sys.argv[:]
sys.argv = [sys.argv[0], "ollama_cloud", "deepseek-v4.1-flash", "full"]   # module-level settings of lr
import llm_rerank_experiment as lr                                      # noqa: E402
import build_movie_texts as bmt                                         # noqa: E402
import cold_start_experiment as cs                                      # noqa: E402

MODE = sys_argv[1] if len(sys_argv) > 1 else "pilot"
PROVIDER, MODEL = "ollama_cloud", "deepseek-v4.1-flash"
REQUESTS = lr.Path("output/experiments/typed_requests/requests_r1.jsonl")
OUT = lr.FOLDER


def load_requests():
    """{(user_id, n_seeds): record} for the general-style requests."""
    result = {}
    for line in REQUESTS.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        result[(r["user_id"], r["n"])] = r
    return result


def request_prompt(user, request, top_ids, data):
    """Same candidates, shuffle and candidate lines as p2; the seed list and profile are replaced
    by the user's typed request."""
    rng = np.random.default_rng(user["user_id"])          # the same order as every shuffle-0 run
    shuffled = rng.permutation(top_ids)
    number_to_id = {i + 1: int(m) for i, m in enumerate(shuffled)}
    candidate_text = "\n".join(f"{n}. {lr.movie_line(m, data, detailed=True)}" for n, m in number_to_id.items())
    user_prompt = f"""A new user described what they like to watch:
"{request}"

Rank ALL {lr.TOP_M} of the following films from the one this user is most likely to enjoy
to the one they are least likely to enjoy. Consider what they asked for, and your own
knowledge of the films:
{candidate_text}

Use every number from 1 to {lr.TOP_M} exactly once.
Return JSON in exactly this form: {{"ranking": [numbers, best first]}}"""
    messages = [{"role": "system", "content": lr.SYSTEM_PROMPT}, {"role": "user", "content": user_prompt}]
    return messages, number_to_id


def run_request(users, data, requests, n_seeds):
    cache_path = OUT / f"answers_{PROVIDER}_{MODEL}_request_r1_n{n_seeds}_cpop.jsonl"
    cache = {}
    if cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            cache[rec["user_id"]] = rec
    rows = []
    with open(cache_path, "a", encoding="utf-8") as f:
        for user in tqdm(users, desc=f"request, {n_seeds} seeds"):
            req = requests[(user["user_id"], n_seeds)]
            assert list(req["seeds"]) == list(user["liked"]), f"seed mismatch for user {user['user_id']}"
            top_ids, _ = lr.popularity_candidates(user, data)
            messages, number_to_id = request_prompt(user, req["request"], top_ids, data)
            rec = cache.get(user["user_id"])
            if rec is None:
                result = lr.call_model(messages, PROVIDER, MODEL)
                rec = {"user_id": int(user["user_id"]), "n_seeds": n_seeds, "request": req["request"],
                       "content": result["content"], "tokens_in": result["tokens_in"],
                       "tokens_out": result["tokens_out"], "seconds": round(result["seconds"], 2)}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
            order, issues = lr.parse_ranking(rec["content"], number_to_id, top_ids)
            rows.append({"user_id": user["user_id"],
                         "popularity_ndcg": bmt.user_metrics(top_ids[:bmt.K], user["relevant"], bmt.K)["ndcg"],
                         "ndcg": bmt.user_metrics(order[:bmt.K], user["relevant"], bmt.K)["ndcg"],
                         "issues": len(issues), "tokens_in": rec["tokens_in"] or 0,
                         "tokens_out": rec["tokens_out"] or 0, "seconds": rec["seconds"]})
    return pd.DataFrame(rows)


def seed_run(users, data, n_seeds):
    """The seed-based p2 reranking of popularity (cached for 10 seeds, new for 3)."""
    df = lr.run_model(PROVIDER, MODEL, users, data, version="p2", candidates="popularity")
    return df.rename(columns={"llm_ndcg": "ndcg"})[["user_id", "popularity_ndcg", "ndcg", "issues",
                                                     "tokens_in", "tokens_out", "seconds"]]


def paired(a, b):
    d = a["ndcg"].to_numpy() - b["ndcg"].to_numpy()
    return d.mean(), int((d > 0).sum()), int((d < 0).sum()), (wilcoxon(d).pvalue if d.any() else 1.0)


def main():
    data = lr.load_data()
    requests = load_requests()
    n_users = 10 if MODE == "pilot" else None
    rows, per_user = [], []
    for n in (3, 10):
        users = cs.users_with_n_seeds(data["val_users"], n)[:n_users]
        req = run_request(users, data, requests, n)
        seeds = seed_run(users, data, n)
        assert (req["user_id"].to_numpy() == seeds["user_id"].to_numpy()).all()
        pop = req[["user_id", "popularity_ndcg"]].rename(columns={"popularity_ndcg": "ndcg"})
        for name, df in (("request", req), ("seeds p2", seeds)):
            diff_pop, better, worse, p_pop = paired(df, pop)
            rows.append({"n_seeds": n, "input": name, "users": len(df), "popularity": pop["ndcg"].mean(),
                         "ndcg": df["ndcg"].mean(), "diff_vs_popularity": diff_pop,
                         "better": better, "worse": worse, "p_vs_popularity": p_pop,
                         "answers_with_issues": int((df["issues"] > 0).sum()),
                         "tokens_in": df["tokens_in"].mean(), "tokens_out": df["tokens_out"].mean(),
                         "seconds": df["seconds"].mean()})
            per_user.append(df.assign(n_seeds=n, input=name))
        d, b, w, p = paired(req, seeds)
        rows.append({"n_seeds": n, "input": "request - seeds p2", "users": len(req), "ndcg": d,
                     "better": b, "worse": w, "p_vs_popularity": p})
    summary = pd.DataFrame(rows)
    tag = "pilot" if MODE == "pilot" else "val_203users"
    summary.to_csv(OUT / f"request_rerank_summary_{tag}.csv", index=False)
    pd.concat(per_user).to_csv(OUT / f"request_rerank_per_user_{tag}.csv", index=False)
    pd.set_option("display.width", 250)
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
