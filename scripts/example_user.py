"""Figure 10.1 of the report: the top 10 of Hybrid E, the cached DeepSeek V4.1 Flash p2 re-ranking and
the learned ranker for a typical validation user (10 seeds), with answer-key hits marked.

A user is "typical" when every method finds exactly its rounded mean number of liked films; among those,
the user whose seeds are best known is shown (validation user 142206 in the report). Never calls an LLM:
the DeepSeek order comes from the cached answers.

Usage:
    python scripts/example_user.py
Output:
    output/experiments/example_user/example_users.json (the three most typical users)
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "web"))
from recommender.thesis_engine import ThesisEngine

FOLDER = Path("output") / "experiments" / "example_user"
METHODS = ("hybrid_e", "llm", "ltr")


def main():
    eng = ThesisEngine()
    d, lr = eng.data, eng.lr
    answers = d["llm_answers"]["rank_deepseek"]
    users = eng.cs.users_with_n_seeds(d["val_users"], 10)

    def title(m):
        row = eng.catalog.loc[m]
        return f"{row['title']} ({int(row['year'])})"

    rows = []
    for user in users:
        if user["user_id"] not in answers:
            continue
        eng._with_alpha(user)
        top30, _ = lr.hybrid_e_candidates(user, d)
        _, number_to_id = lr.build_prompt(user, top30, d, "p2")      # the order the model saw
        llm, _ = lr.parse_ranking(answers[user["user_id"]]["content"], number_to_id, top30)
        lists = {"hybrid_e": top30[:10], "llm": llm[:10], "ltr": eng.rank_ltr(user)[:10]}
        hits = {k: sum(m in user["relevant"] for m in v) for k, v in lists.items()}
        votes = sum(eng.catalog.loc[m, "vote_count"] for m in user["liked"])
        rows.append((user, lists, hits, votes))

    means = {k: round(sum(r[2][k] for r in rows) / len(rows)) for k in METHODS}
    typical = sorted((r for r in rows if r[2] == means), key=lambda r: -r[3])
    print(f"{len(rows)} users; rounded mean hits {means}; {len(typical)} users match exactly")

    out = []
    for user, lists, hits, _ in typical[:3]:
        print(f"\n=== user {user['user_id']} {hits}")
        print("seeds:", "; ".join(title(m) for m in user["liked"]))
        for k, v in lists.items():
            print(f"  {k}:")
            for i, m in enumerate(v, 1):
                print(f"    {i:2d}. {'HIT ' if m in user['relevant'] else '    '}{title(m)}")
        out.append({"user_id": user["user_id"], "hits": hits,
                    "seeds": [title(m) for m in user["liked"]],
                    "lists": {k: [[title(m), m in user["relevant"]] for m in v] for k, v in lists.items()}})

    FOLDER.mkdir(parents=True, exist_ok=True)
    with open(FOLDER / "example_users.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
