'''
Semantic-ID group names in the LLM reranking prompt (doc 32). Pre-registration:
output/experiments/llm_sid_prompt/preregistration.md (written before the first run).

p1  title, year, genres                                     (existing prompt, unchanged)
p9  p1 + "taste group: <name>" per film; groups = first token of the S4 EASE semantic IDs (doc 30),
    names written once by the same LLM (32 calls, cached)
p2  existing cached answers (title, year, genres, director, cast + seed profile)

deepseek-v4.1-flash via ollama_cloud, 203 validation users, N = 10, Hybrid E top 30, shuffles 0/1/2.
H1: p9 vs p1 (Wilcoxon). H2: p9 fewer tokens than p2 and NDCG >= p2 - 0.005.

Usage:
    python experiments/semantic_ids/llm_sid_prompt_experiment.py names    only write the 32 group names (prints them)
    python experiments/semantic_ids/llm_sid_prompt_experiment.py pilot    names + p1 and p9 on the first 20 users, shuffle 0
    python experiments/semantic_ids/llm_sid_prompt_experiment.py full     everything
'''

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))  # shared modules live in src/
import sys

MODE = sys.argv[1] if len(sys.argv) > 1 else "pilot"
sys.argv = [sys.argv[0]]                 # llm_rerank_experiment reads sys.argv when imported
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import cold_start_experiment as cs
import llm_rerank_experiment as lr

OUT = Path("output") / "experiments" / "llm_sid_prompt"
IDS = Path("output") / "experiments" / "sid_vector_sources" / "ids" / "S4_all-pool_k32_s42.npz"
PROVIDER, MODEL = "ollama_cloud", "deepseek-v4.1-flash"
N_SEEDS = 10
SHUFFLES = (0, 1, 2)
FILMS_PER_NAME = 15
MARGIN = 0.005

ORIGINAL_BUILD_PROMPT = lr.build_prompt
GROUP_NAME_OF = {}                       # movie id -> group name, filled in main()


def group_of_film():
    z = np.load(IDS)
    return {int(m): int(c[0]) for m, c in zip(z["movie_ids"], z["codes"])}


def name_groups(data, group_of):
    path = OUT / "group_names.json"
    names = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    for g in sorted(set(group_of.values())):
        if str(g) in names:
            continue
        films = [m for m, gg in group_of.items() if gg == g]
        films.sort(key=lambda m: -data["pop_vector"][data["position_of"][m]])
        lines = "\n".join(f"- {data['movie_by_id'].loc[m, 'title']} ({data['movie_by_id'].loc[m, 'year']})"
                          for m in films[:FILMS_PER_NAME])
        prompt = (f"These films form one group: people who like one of them tend to like the others.\n{lines}\n\n"
                  "Give this group a short name (2-5 words) that says what the films have in common or what kind "
                  'of viewer likes them. Return JSON in exactly this form: {"name": "..."}')
        messages = [{"role": "system", "content": lr.SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
        result = lr.call_model(messages, PROVIDER, MODEL)
        names[str(g)] = json.loads(result["content"])["name"].strip()
        path.write_text(json.dumps(names, indent=1, ensure_ascii=False), encoding="utf-8")
    return {int(k): v for k, v in names.items()}


def build_prompt(user, top_ids, data, version="p1", shuffle=0):
    """p9 here; every other version goes to the original builder unchanged."""
    if version != "p9":
        return ORIGINAL_BUILD_PROMPT(user, top_ids, data, version, shuffle)
    rng = np.random.default_rng(user["user_id"] if not shuffle else [user["user_id"], shuffle])
    number_to_id = {i + 1: int(m) for i, m in enumerate(rng.permutation(top_ids))}   # same order as p1/p2

    def line(m):
        text = lr.movie_line(m, data)
        return text + (f" – taste group: {GROUP_NAME_OF[m]}" if m in GROUP_NAME_OF else "")

    seed_text = "\n".join("- " + line(m) for m in user["liked"])
    candidate_text = "\n".join(f"{n}. {line(m)}" for n, m in number_to_id.items())
    user_prompt = f"""A new user told us they enjoyed these films. Each film is followed by its taste group:
films in the same taste group tend to be liked by the same people.
{seed_text}

Rank ALL {lr.TOP_M} of the following films from the one this user is most likely to enjoy
to the one they are least likely to enjoy:
{candidate_text}

Use every number from 1 to {lr.TOP_M} exactly once.
Return JSON in exactly this form: {{"ranking": [numbers, best first]}}"""
    return [{"role": "system", "content": lr.SYSTEM_PROMPT}, {"role": "user", "content": user_prompt}], number_to_id


def bootstrap_ci(diff, n=10_000):
    rng = np.random.default_rng(42)
    means = rng.choice(diff, size=(n, len(diff)), replace=True).mean(axis=1)
    return np.percentile(means, [2.5, 97.5])


def analyse(per_user):
    per_user.to_csv(OUT / "per_user.csv", index=False)
    by_user = per_user.groupby(["version", "user_id"])[["llm_ndcg", "llm_precision", "llm_recall", "tokens_in"]].mean()
    summary = by_user.groupby("version").mean()
    summary["hybrid_E"] = per_user.groupby("version")["hybrid_ndcg"].mean()
    summary.to_csv(OUT / "summary.csv")
    wide = by_user["llm_ndcg"].unstack("version")

    tests = []
    d1 = wide["p9"] - wide["p1"]
    tests.append({"hypothesis": "H1 p9 vs p1", "difference": d1.mean(), "ci_low": bootstrap_ci(d1.values)[0],
                  "ci_high": bootstrap_ci(d1.values)[1], "better": int((d1 > 0).sum()), "worse": int((d1 < 0).sum()),
                  "p_value": wilcoxon(d1).pvalue if d1.any() else 1.0})
    d2 = wide["p9"] - wide["p2"]
    tests.append({"hypothesis": "H2 p9 vs p2", "difference": d2.mean(), "ci_low": bootstrap_ci(d2.values)[0],
                  "ci_high": bootstrap_ci(d2.values)[1], "better": int((d2 > 0).sum()), "worse": int((d2 < 0).sum()),
                  "p_value": wilcoxon(d2).pvalue if d2.any() else 1.0})
    tests = pd.DataFrame(tests)
    tests.to_csv(OUT / "tests.csv", index=False)

    pd.set_option("display.width", 250)
    print("\nMean over 3 shuffles, 203 validation users:")
    print(summary.round(4).to_string())
    print(tests.round(4).to_string(index=False))
    h1 = d1.mean() > 0 and tests.loc[0, "p_value"] < 0.05
    h2 = summary.loc["p9", "tokens_in"] < summary.loc["p2", "tokens_in"] and d2.mean() >= -MARGIN
    print(f"\nH1 (group names help vs p1): {'POSITIVE' if h1 else 'not shown'}")
    print(f"H2 (p2 accuracy at fewer tokens): {'POSITIVE' if h2 else 'not shown'} "
          f"(tokens {summary.loc['p9', 'tokens_in']:.0f} vs {summary.loc['p2', 'tokens_in']:.0f}, "
          f"NDCG difference {d2.mean():+.4f}, margin -{MARGIN})")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lr.build_prompt = build_prompt
    data = lr.load_data()
    group_of = group_of_film()
    names = name_groups(data, group_of)
    GROUP_NAME_OF.update({m: names[g] for m, g in group_of.items()})
    print(f"{len(names)} group names, {len(GROUP_NAME_OF)} films with a group")
    for g, name in sorted(names.items()):
        print(f"  {g:>2}: {name}  ({sum(1 for x in group_of.values() if x == g)} films)")
    if MODE == "names":
        return

    users = cs.users_with_n_seeds(data["val_users"], N_SEEDS)
    if MODE == "pilot":
        users = users[:20]
        runs = [("p1", 0), ("p9", 0)]
    else:
        runs = [(v, s) for v in ("p1", "p9", "p2") for s in SHUFFLES]
    frames = []
    for version, shuffle in runs:
        frame = lr.run_model(PROVIDER, MODEL, users, data, version=version, shuffle=shuffle)
        frames.append(frame.assign(shuffle=shuffle))
        print(f"{version} shuffle {shuffle}: LLM NDCG@10 {frame['llm_ndcg'].mean():.4f}, "
              f"tokens in {frame['tokens_in'].mean():.0f}", flush=True)
    per_user = pd.concat(frames, ignore_index=True)
    if MODE == "pilot":
        print(per_user.groupby("version")[["llm_ndcg", "hybrid_ndcg", "tokens_in", "issues"]].mean().round(4))
        print("\nExample p9 prompt (first user, shuffle 0):")
        top_ids, _ = lr.hybrid_e_candidates(users[0], data)
        print(build_prompt(users[0], top_ids, data, "p9")[0][1]["content"][:1500])
        return
    analyse(per_user)


if __name__ == "__main__":
    main()
