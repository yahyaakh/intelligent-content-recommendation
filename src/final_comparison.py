'''
One cold-start comparison of every method in the thesis (doc 18, section 7a; CLI option 11).

The same setup for every method (as docs 06-17):
  - training data: the 9,591 warm users only
  - evaluation:    the 203 validation users (the 814 test users only through the pre-registered final test)
  - input:         each user's first 3 or 10 liked films (seeds)
  - answer key:    their other liked films (rating >= their own mean), the same for 3 and 10 seeds
  - metric:        NDCG@10 (plus precision and recall), paired Wilcoxon vs Hybrid E, Bonferroni per seed count

Methods (reusing the code of the experiments, so the numbers match the documents):
  Popularity                  count of positive ratings                                (doc 06)
  Content                     embeddinggemma + LLM labels, max over the seeds          (docs 05, 06)
  Item-CF                     mean item-CF similarity to the seeds                     (doc 06)
  Hybrid E                    item-CF + popularity, alpha from validation              (doc 06)
  LTR v2                      tuned XGBoost + LightGBM ensemble on Hybrid E's top 30   (doc 16)
  LLM (10 seeds only)         cached reranking answers of Hybrid E's top 30, NO model calls (docs 13, 17)
  LLM fusion (10 seeds only)  0.4 x Hybrid E rank + 0.6 x DeepSeek rank (doc 17, method A; w = 0.4 was
                              chosen by cross-validation on these validation users, so it is optimistic)
  Option 10 methods          recomm.py's Content, Collaborative, SVD, NCF and Hybrid, retrained on the warm
                              users only; SVD and NCF get the new user by fold-in (cold_start_baselines.py)
  EASE                        Steck 2019, recomm.py's EASE_R_Recommende fitted on the warm users' liked films;
                              lambda per seed count from the simulated warm users (ease_tuning.py, doc 26)
Still to add: semantic IDs (doc 19).

Usage:
    python src/final_comparison.py           (or option 11 in recomm.py)
'''
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.argv = [sys.argv[0]]                 # the imported experiment scripts read sys.argv when imported
import build_movie_texts as bmt
import cold_start_experiment as cs
import llm_rerank_experiment as lr
import LightGBM_XBoost_rerank_experiment as ltr
import cold_start_baselines as cbl
from recomm import EASE_R_Recommende

FOLDER = Path("output") / "experiments" / "final_comparison"
# EASE lambda per seed count, chosen on the simulated warm users (ease_tuning.py, tuning.csv), never on
# the validation users: 10 seeds 20,000 (NDCG@10 0.650), 3 seeds 120,000 (0.609)
EASE_LAMBDAS = {10: 20000, 3: 120000}
SEED_COUNTS = (10, 3)
K = bmt.K

# Cached LLM runs: (label, provider, model, prompt version). Validation users, 10 seeds, Hybrid E's top 30.
LLM_RUNS = [
    ("LLM: DeepSeek V4.1 Flash, p2", "ollama_cloud", "deepseek-v4.1-flash", "p2"),
    ("LLM: DeepSeek V4.1 Flash, p5 (+ CF data)", "ollama_cloud", "deepseek-v4.1-flash", "p5"),
    ("LLM: Kimi K3, p2", "ollama_cloud", "kimi-k3", "p2"),
    ("LLM: Kimi K3, p5 (+ CF data)", "ollama_cloud", "kimi-k3", "p5"),
]
FUSION = ("LLM fusion: Hybrid E + DeepSeek p2 (w = 0.4)", "ollama_cloud", "deepseek-v4.1-flash", "p2", 0.4)

# Expected validation values from the documents, checked on every run
EXPECTED = {(10, "Hybrid E"): 0.6286, (3, "Hybrid E"): 0.5837,
            (10, "LTR v2 (XGBoost + LightGBM)"): 0.6686, (3, "LTR v2 (XGBoost + LightGBM)"): 0.6227,
            (10, "LLM: DeepSeek V4.1 Flash, p2"): 0.6212}


def load():
    """The shared data of the cold-start experiments, plus the content table."""
    data = lr.load_data()
    data["content_table"] = bmt.embedding_similarity(
        f"movies_enriched_{cs.EMBED_NAME}_{cs.PROMPT_VERSION}", data["movies"])
    data["pool_positions"] = np.array(data["pool"])
    data["popular"] = cs.popularity_scores(data["train"])
    return data


def from_table(table, user, data, scoring):
    """Top K of a similarity table, as in experiments 05-06 (seen films excluded)."""
    return bmt.recommend(table, user, data["pool_positions"], data["position_of"], data["movie_ids"], K, scoring)


def cached_llm_order(user, data, provider, model, version, cache):
    """The LLM's order of Hybrid E's top 30 from its cache. Never calls a model."""
    key = (provider, model, version)
    if key not in cache:
        path = lr.run_name(provider, model, 10, None, version, "hybrid")[0]
        cache[key] = ({} if not path.exists() else
                      {r["user_id"]: r for r in map(lr.json.loads, path.read_text(encoding="utf-8").splitlines())})
    record = cache[key].get(user["user_id"])
    if record is None:
        return None
    top_ids, _ = lr.hybrid_e_candidates(user, data)
    _, number_to_id = lr.build_prompt(user, top_ids, data, version)      # the order the model saw
    order, _ = lr.parse_ranking(record["content"], number_to_id, top_ids)
    return order


def ltr_top10(n_seeds):
    """LTR v2 final model (doc 16) on the validation users: tuned XGBoost + LightGBM, 50/50 rank average,
    trained on the simulated warm users; scored on the saved validation feature tables."""
    suffix = "g5" if n_seeds == 10 else f"n{n_seeds}"
    train = pd.read_csv(ltr.FOLDER / f"features_simulated_{suffix}.csv")
    val = pd.read_csv(ltr.FOLDER / f"features_val_{suffix}.csv")
    with open(ltr.FOLDER / "step4_chosen_params.json", encoding="utf-8") as f:
        chosen = ltr.json.load(f)
    columns = [c for group in ltr.MODELS[ltr.FINAL_MODEL["features"]] for c in ltr.GROUPS[group]]
    val = val.assign(score=ltr.ensemble_scores(train, val, columns, chosen))
    return {int(u): g.sort_values("score", ascending=False, kind="stable")["movie_id"].astype(int).tolist()[:K]
            for u, g in val.groupby("user_id")}


def run(split="val"):
    if split != "val":
        raise ValueError("Option 11 uses the validation users only. The 814 test users are used once, "
                         "in the pre-registered final test (doc 18, section 9).")
    FOLDER.mkdir(parents=True, exist_ok=True)
    data = load()
    rows, missing_llm, cache = [], {}, {}
    option10 = cbl.ColdStartBaselines(data)

    for n in SEED_COUNTS:
        users = cs.users_with_n_seeds(data["val_users"], n)
        print(f"\n{len(users)} validation users, {n} seeds: scoring the methods...")
        ltr_lists = ltr_top10(n)
        ease = EASE_R_Recommende(data["train"], data["movies"], EASE_LAMBDAS[n])
        ease.fit(data["train"])                                  # warm users only
        for user in users:
            hybrid_top30, _ = lr.hybrid_e_candidates(user, data)
            lists = {
                "Popularity": cs.recommend_popular(data["popular"].index, user, K),
                "Content (embedding + LLM labels, max)": from_table(data["content_table"], user, data, "max"),
                "Item-CF": from_table(data["item_cf"], user, data, "mean"),
                "Hybrid E": hybrid_top30[:K],
                "LTR v2 (XGBoost + LightGBM)": ltr_lists[int(user["user_id"])],
                f"EASE (lambda {EASE_LAMBDAS[n]:,})": [int(m) for m in
                                                       ease.recommend(user["liked"], K, exclude=user["seen"])],
                **option10.lists(user),
            }
            if n == 10:
                for label, provider, model, version in LLM_RUNS:
                    order = cached_llm_order(user, data, provider, model, version, cache)
                    if order is None:
                        missing_llm[label] = missing_llm.get(label, 0) + 1
                    else:
                        lists[label] = order[:K]
                label, provider, model, version, w = FUSION
                order = cached_llm_order(user, data, provider, model, version, cache)
                if order is not None:
                    llm_rank = {m: i + 1 for i, m in enumerate(order)}
                    combined = [w * (i + 1) + (1 - w) * llm_rank[m] for i, m in enumerate(hybrid_top30)]
                    best = np.argsort(combined, kind="stable")[:K]      # ties: Hybrid E's order
                    lists[label] = [hybrid_top30[i] for i in best]
            for method, recommended in lists.items():
                rows.append({"seeds": n, "method": method, "user_id": user["user_id"],
                             **bmt.user_metrics(recommended, user["relevant"], K)})

    per_user = pd.DataFrame(rows)
    order = list(dict.fromkeys(per_user["method"]))
    summary = (per_user.groupby(["seeds", "method"], sort=False)[["ndcg", "precision", "recall"]]
               .mean().reset_index())
    summary["users"] = per_user.groupby(["seeds", "method"], sort=False).size().to_numpy()

    tests = []
    for n, part in per_user.groupby("seeds", sort=False):
        table = part.pivot(index="user_id", columns="method", values="ndcg")
        for method in [m for m in order if m in table.columns and m != "Hybrid E"]:
            both = table[[method, "Hybrid E"]].dropna()
            diff = both[method] - both["Hybrid E"]
            tests.append({"seeds": n, "method": method, "vs_hybrid_E": diff.mean(),
                          "better": int((diff > 0).sum()), "worse": int((diff < 0).sum()),
                          "p_value": wilcoxon(diff).pvalue if diff.any() else 1.0})
    tests = pd.DataFrame(tests)
    tests["p_bonferroni"] = (tests["p_value"] * tests.groupby("seeds")["p_value"].transform("size")).clip(upper=1.0)
    summary = summary.merge(tests, on=["seeds", "method"], how="left")

    per_user.to_csv(FOLDER / "per_user_val.csv", index=False)
    summary.to_csv(FOLDER / "summary_val.csv", index=False)

    # Checks against the documents
    print()
    for (n, method), expected in EXPECTED.items():
        got = summary.loc[(summary["seeds"] == n) & (summary["method"] == method), "ndcg"]
        status = "ok" if len(got) and abs(got.iloc[0] - expected) < 5e-4 else "DIFFERENT"
        print(f"check: {method}, {n} seeds = {got.iloc[0] if len(got) else float('nan'):.4f} "
              f"(document: {expected}) -> {status}")
    for label, count in missing_llm.items():
        print(f"note: {label} has no cached answer for {count} users (left out; never called)")

    pd.set_option("display.width", 250)
    for n in SEED_COUNTS:
        part = summary[summary["seeds"] == n].sort_values("ndcg", ascending=False)
        print(f"\nCold start, {n} seeds, validation users - sorted by NDCG@10 "
              "(vs Hybrid E: paired Wilcoxon, Bonferroni within this table)")
        print(part[["method", "users", "ndcg", "precision", "recall", "vs_hybrid_E", "better", "worse",
                    "p_bonferroni"]].round(4).to_string(index=False))
    print(f"\nSaved to {FOLDER}")
    return summary


if __name__ == "__main__":
    run()
