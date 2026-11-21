'''
Two LLM runs on the validation users, 10 seed films, nothing on the test users.

  pilot   Two models not tested before rerank Hybrid E's top 30 (idea 1) on the 20-user pilot,
          with prompts p2 and p5:
              deepseek-v4.1-flash:1017   the 17 October snapshot of DeepSeek V4.1 Flash
              minimax-m3                 a new model family (MiniMax)
          References from the cache (no new calls): DeepSeek V4.1 Flash, Kimi K3 and GLM-5.3,
          all on the same 20 users and prompts, plus Hybrid E and popularity.

  uknn    The LLM reranks user-kNN's top 30 instead of Hybrid E's (idea 4). User-kNN (doc 20) is the
          strongest single method without a learned ranker (validation 0.6417 vs Hybrid E 0.6286 at
          10 seeds), so the question is: does an LLM improve on user-kNN's own order?
          Models: DeepSeek V4.1 Flash and Kimi K3, prompt p5 (p2 with "uknn p2").
          The comparison is LLM vs user-kNN itself on the same users (paired Wilcoxon).

Usage (each answer is cached, so a stopped run continues and nothing is paid for twice):
    python experiments/llm/llm_new_models_pilot.py pilot            20 users, about 160 new calls
    python experiments/llm/llm_new_models_pilot.py pilot full       DeepSeek 1017 on all 203 users, p2 and p5 (about 370 new calls)
    python experiments/llm/llm_new_models_pilot.py pilot full p2    the same, p2 only
    python experiments/llm/llm_new_models_pilot.py uknn             20 users, p5, about 40 new calls
    python experiments/llm/llm_new_models_pilot.py uknn full        all 203 validation users, p5
    python experiments/llm/llm_new_models_pilot.py uknn full p2     the same with prompt p2
    python experiments/llm/llm_new_models_pilot.py blend            user-kNN + LLM fusion (doc 17 method A), cached answers, no calls
    python experiments/llm/llm_new_models_pilot.py check            no calls: prompt sizes and the user-kNN check
Results: output/experiments/llm_rerank/compare_summary_*.csv and new_models_*.csv.
'''

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))  # shared modules live in src/
import json
import sys
from pathlib import Path

ARGS = sys.argv[1:]
sys.argv = [sys.argv[0]]       # llm_rerank_experiment reads sys.argv when imported
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import build_movie_texts as bmt
import cold_start_experiment as cs
import llm_rerank_experiment as lr

MODE = ARGS[0] if ARGS else "check"
FULL = "full" in ARGS
VERSIONS = [a for a in ARGS if a in ("p1", "p2", "p3", "p4", "p5")]

N_SEEDS = 10
PILOT_USERS = 20
NEW_MODELS = [("ollama_cloud", "deepseek-v4.1-flash:1017"), ("ollama_cloud", "minimax-m3")]
FULL_MODELS = [("ollama_cloud", "deepseek-v4.1-flash:1017")]     # "pilot full": all 203 validation users
CACHED_REFERENCES = [("ollama_cloud", "deepseek-v4.1-flash"), ("ollama_cloud", "kimi-k3"),
                     ("ollama_cloud", "glm-5.3")]
UKNN_MODELS = [("ollama_cloud", "deepseek-v4.1-flash"), ("ollama_cloud", "kimi-k3")]
USER_KNN_VAL = 0.6417          # doc 20: user-kNN, 203 validation users, 10 seeds (option 11)


def cached_for(provider, model, version, users, candidates="hybrid"):
    """True if every user already has an answer in this model's cache (so using it costs nothing)."""
    path, _ = lr.run_name(provider, model, N_SEEDS, None, version, candidates)
    if not path.exists():
        return False
    done = {json.loads(line)["user_id"] for line in path.read_text(encoding="utf-8").splitlines()}
    return all(u["user_id"] in done for u in users)


def paired(a, b, text):
    diff = (a - b).dropna()
    p = wilcoxon(diff).pvalue if diff.any() else 1.0
    print(f"  {text:<58} {diff.mean():+.4f}  (better {(diff > 0).sum():>3}, worse {(diff < 0).sum():>3}, p = {p:.3f})")
    return {"comparison": text, "difference": diff.mean(), "better": int((diff > 0).sum()),
            "worse": int((diff < 0).sum()), "p_value": p, "users": len(diff)}


def run_pilot(data, users, new_models=NEW_MODELS):
    versions = VERSIONS or ["p2", "p5"]
    results, ndcg = [], {}
    NEW_MODELS = new_models
    for version in versions:
        references = [m for m in CACHED_REFERENCES if cached_for(*m, version, users)]
        skipped = [m[1] for m in CACHED_REFERENCES if m not in references]
        if skipped:
            print(f"[{version}] not cached for these users, left out (no new calls): {skipped}")
        per_user, _ = lr.compare_models(references + NEW_MODELS, data, n_users=len(users), version=version,
                                        n_seeds=N_SEEDS, candidates="hybrid")
        for (provider, model) in references + NEW_MODELS:
            label = lr.run_name(provider, model, N_SEEDS, None, version, "hybrid")[1]
            ndcg[(model, version)] = per_user[per_user["model"] == label].set_index("user_id")["llm_ndcg"]
        ndcg[("Hybrid E", version)] = per_user.groupby("user_id")["hybrid_ndcg"].first()

    if len(users) <= PILOT_USERS:
        print(f"\nPilot, {len(users)} validation users, {N_SEEDS} seeds (a pilot only picks models for the full run;"
              " 20 users cannot show a difference of about 0.01)")
    else:
        print(f"\nAll {len(users)} validation users, {N_SEEDS} seeds (raw p-values; Bonferroni = p x comparisons "
              "you report; validation only, the test users are untouched)")
    for version in versions:
        print(f"[{version}]")
        for _, model in NEW_MODELS:
            results.append(paired(ndcg[(model, version)], ndcg[("Hybrid E", version)], f"{model} - Hybrid E [{version}]"))
            for _, ref in CACHED_REFERENCES:
                if (ref, version) in ndcg:
                    results.append(paired(ndcg[(model, version)], ndcg[(ref, version)], f"{model} - {ref} [{version}]"))
    if {"p2", "p5"} <= set(versions):
        print("[p5 - p2]")
        for _, model in NEW_MODELS:
            results.append(paired(ndcg[(model, "p5")], ndcg[(model, "p2")], f"{model}: p5 - p2"))
    out = lr.FOLDER / (f"new_models_pilot_{len(users)}users_n{N_SEEDS}.csv" if len(users) <= PILOT_USERS
                       else f"new_models_full_{len(users)}users_n{N_SEEDS}_{'_'.join(versions)}.csv")
    pd.DataFrame(results).to_csv(out, index=False)
    print(f"Saved -> {out}")


def check_uknn(data, users):
    """No model calls: user-kNN's top 10 from the candidate lists must give option 11's NDCG."""
    values = [bmt.user_metrics(lr.uknn_candidates(u, data)[0][:bmt.K], u["relevant"], bmt.K)["ndcg"] for u in users]
    print(f"user-kNN from the candidate lists: {np.mean(values):.4f} on {len(users)} users", end="")
    if len(users) == len(cs.users_with_n_seeds(data["val_users"], N_SEEDS)):
        assert abs(np.mean(values) - USER_KNN_VAL) < 5e-5, f"expected {USER_KNN_VAL} (option 11)"
        print(f" = option 11 ({USER_KNN_VAL}), OK")
    else:
        print()


def run_uknn(data, users):
    versions = VERSIONS or ["p5"]
    results = []
    for version in versions:
        per_user, _ = lr.compare_models(UKNN_MODELS, data, n_users=len(users), version=version,
                                        n_seeds=N_SEEDS, candidates="uknn")
        knn = per_user.groupby("user_id")["candidates_ndcg"].first()
        hybrid = per_user.groupby("user_id")["hybrid_ndcg"].first()
        print(f"\nLLM reranking user-kNN's top 30, {len(users)} validation users, {N_SEEDS} seeds, prompt {version}")
        print(f"  user-kNN {knn.mean():.4f}   Hybrid E {hybrid.mean():.4f}")
        results.append(paired(knn, hybrid, f"user-kNN - Hybrid E [{version}]"))
        for provider, model in UKNN_MODELS:
            label = lr.run_name(provider, model, N_SEEDS, None, version, "uknn")[1]
            llm = per_user[per_user["model"] == label].set_index("user_id")["llm_ndcg"]
            print(f"  {model}: {llm.mean():.4f}")
            results.append(paired(llm, knn, f"{model} reranking user-kNN - user-kNN [{version}]"))
            results.append(paired(llm, hybrid, f"{model} reranking user-kNN - Hybrid E [{version}]"))
            # the same model reranking Hybrid E's top 30, when cached: which candidate list is better for the LLM?
            if cached_for(provider, model, version, users, "hybrid"):
                path, _ = lr.run_name(provider, model, N_SEEDS, None, version, "hybrid")
                records = {json.loads(l)["user_id"]: json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()}
                on_hybrid = {}
                for u in users:
                    top_ids = lr.hybrid_e_candidates(u, data)[0]
                    _, number_to_id = lr.build_prompt(u, top_ids, data, version)
                    order, _ = lr.parse_ranking(records[u["user_id"]]["content"], number_to_id, top_ids)
                    on_hybrid[u["user_id"]] = bmt.user_metrics(order[:bmt.K], u["relevant"], bmt.K)["ndcg"]
                results.append(paired(llm, pd.Series(on_hybrid),
                                      f"{model}: user-kNN candidates - Hybrid E candidates [{version}]"))
    out = lr.FOLDER / f"uknn_rerank_{len(users)}users_n{N_SEEDS}_{'_'.join(versions)}.csv"
    pd.DataFrame(results).to_csv(out, index=False)
    print(f"Saved -> {out}  (raw p-values; Bonferroni = p x number of LLM rows)")


def run_blend(data, users, version="p5"):
    """Method A of doc 17 with user-kNN in place of Hybrid E: user-kNN's order blended with the LLM's
    reranking of user-kNN's top 30 (cached answers, no model calls). The same three fusions and the same
    5-fold cross-validation over users as llm_fusion_experiment.py, and the same three non-LLM controls.
    w = 1 is user-kNN alone, w = 0 the LLM alone. Ties keep user-kNN's order."""
    import llm_fusion_experiment as lf
    answers = {}
    for provider, model in UKNN_MODELS:
        if not cached_for(provider, model, version, users, "uknn"):
            raise SystemExit(f"{model} has no cached {version} answers on user-kNN candidates for these users; "
                             f"run 'uknn full' first")
        path = lr.run_name(provider, model, N_SEEDS, None, version, "uknn")[0]
        answers[model] = {r["user_id"]: r for r in map(json.loads, path.read_text(encoding="utf-8").splitlines())}

    emb = bmt.embedding_similarity("movies_base_embeddinggemma_v2", data["movies"])
    position_of = data["position_of"]
    rng = np.random.default_rng(42)
    inputs = []
    for u in users:
        top_ids, scores = lr.uknn_candidates(u, data)
        _, number_to_id = lr.build_prompt(u, top_ids, data, version)      # the shuffle of the cached run
        rank = {}
        for model, records in answers.items():
            order, _ = lr.parse_ranking(records[u["user_id"]]["content"], number_to_id, top_ids)
            position = {m: i + 1 for i, m in enumerate(order)}
            rank[model] = np.array([position[m] for m in top_ids], float)
        cands, seeds = [position_of[m] for m in top_ids], [position_of[m] for m in u["liked"]]
        rank["control: popularity"] = lf.ranks_of(data["pop_vector"][cands])
        rank["control: content (emb_max)"] = lf.ranks_of(emb[np.ix_(seeds, cands)].max(axis=0))
        for r in range(lf.RANDOM_REPEATS):
            rank[f"control: random {r}"] = rng.permutation(len(top_ids)).astype(float) + 1
        # lf.fused reads the base order from "hybrid_rank"/"hybrid_score": here they hold user-kNN's
        inputs.append({"user_id": u["user_id"], "ids": np.array(top_ids), "relevant": u["relevant"],
                       "hybrid_rank": np.arange(1, len(top_ids) + 1, dtype=float),
                       "hybrid_score": cs.rescale(np.asarray(scores, float)), "llm_rank": rank,
                       "hybrid_e": bmt.user_metrics(lr.hybrid_e_candidates(u, data)[0][:bmt.K], u["relevant"], bmt.K)["ndcg"]})
    ids = [u["user_id"] for u in inputs]
    knn = pd.Series([lf.ndcg_of(u, u["hybrid_rank"], True) for u in inputs], index=ids)
    hybrid = pd.Series([u["hybrid_e"] for u in inputs], index=ids)
    print(f"\nuser-kNN {knn.mean():.4f}   Hybrid E {hybrid.mean():.4f}   ({len(ids)} validation users, {N_SEEDS} seeds)")
    if len(ids) == len(cs.users_with_n_seeds(data["val_users"], N_SEEDS)):
        assert abs(knn.mean() - USER_KNN_VAL) < 5e-5, "user-kNN must equal option 11"

    per_user, summary, tests, curves = {"user-kNN": knn, "Hybrid E": hybrid}, [], [], {}
    for model in list(answers) + lf.CONTROLS:
        names = [f"control: random {r}" for r in range(lf.RANDOM_REPEATS)] if model == "control: random" else [model]
        averaged = lambda fn: pd.concat([fn(n) for n in names], axis=1).mean(axis=1)
        alone = averaged(lambda n: pd.Series([lf.ndcg_of(u, u["llm_rank"][n], True) for u in inputs], index=ids))
        per_user[f"{model} alone"] = alone
        row = {"model": model, "alone": alone.mean()}
        for method in ("rank blend", "score blend", "RRF"):
            if method == "RRF":
                scores = averaged(lambda n: pd.Series([lf.fused(u, n, "RRF") for u in inputs], index=ids))
            else:
                runs = [lf.cross_validated(inputs, n, method) for n in names]
                scores = pd.concat([r[0] for r in runs], axis=1).mean(axis=1)
                row[f"{method} w per fold"] = runs[0][1] if len(runs) == 1 else "varies"
                curves[f"{model} | {method}"] = pd.concat([r[2] for r in runs], axis=1).mean(axis=1)
            per_user[f"{model} {method}"] = scores
            row[method] = scores.mean()
            for ref_name, ref in (("user-kNN", knn), ("Hybrid E", hybrid)):
                diff = scores - ref
                tests.append({"model": model, "method": method, "vs": ref_name, "difference": diff.mean(),
                              "better": int((diff > 0).sum()), "worse": int((diff < 0).sum()),
                              "p_value": wilcoxon(diff).pvalue if diff.any() else 1.0})
        summary.append(row)

    # sanity: the LLM alone must equal the uknn run (llm_ndcg), w = 1 must equal user-kNN
    for model in answers:
        assert abs(curves[f"{model} | rank blend"].loc[1.0] - knn.mean()) < 1e-9
        assert abs(curves[f"{model} | rank blend"].loc[0.0] - per_user[f"{model} alone"].mean()) < 1e-9

    tests = pd.DataFrame(tests)
    for ref_name in ("user-kNN", "Hybrid E"):      # Bonferroni over every model, control and method
        part = tests["vs"] == ref_name
        tests.loc[part, "p_bonferroni"] = (tests.loc[part, "p_value"] * part.sum()).clip(upper=1.0)
    folder = Path("output") / "experiments" / "llm_fusion"
    folder.mkdir(parents=True, exist_ok=True)
    tag = f"uknn_{len(ids)}users_{version}_n{N_SEEDS}"
    pd.DataFrame(per_user).to_csv(folder / f"per_user_val_{tag}.csv")
    pd.DataFrame(summary).to_csv(folder / f"summary_val_{tag}.csv", index=False)
    tests.to_csv(folder / f"tests_val_{tag}.csv", index=False)
    pd.DataFrame(curves).to_csv(folder / f"weight_curves_val_{tag}.csv", index_label="w (1 = user-kNN)")
    pd.set_option("display.width", 250)
    print(pd.DataFrame(summary).round(4).to_string(index=False))
    print("\nTests (paired Wilcoxon; Bonferroni within each reference):")
    print(tests.round(4).to_string(index=False))
    print("\nNDCG@10 by weight on all users (descriptive; w = 1 is user-kNN, w = 0 the other order alone):")
    print(pd.DataFrame(curves).loc[[0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]].round(4).to_string())
    print(f"Saved -> {folder}/*_{tag}.csv")


def check(data, users_all):
    """No model calls: build every prompt once and report its size, and check the user-kNN lists."""
    pilot = users_all[:PILOT_USERS]
    for candidates, versions in (("hybrid", ("p2", "p5")), ("uknn", ("p2", "p5"))):
        for version in versions:
            lengths = [sum(len(m["content"]) for m in lr.build_prompt(u, lr.get_candidates(u, data, candidates)[0],
                                                                      data, version)[0]) for u in pilot]
            print(f"{candidates:>6} {version}: prompts of {np.mean(lengths):,.0f} characters (~{np.mean(lengths) / 4:,.0f} tokens)")
    overlap = [len(set(lr.uknn_candidates(u, data)[0]) & set(lr.hybrid_e_candidates(u, data)[0])) for u in users_all]
    print(f"user-kNN and Hybrid E top 30 share {np.mean(overlap):.1f} films on average (of 30)")
    for _, model in NEW_MODELS:
        print(f"cache path for {model}: {lr.run_name('ollama_cloud', model, N_SEEDS, None, 'p2')[0].name}")
    check_uknn(data, users_all)


if __name__ == "__main__":
    if MODE not in ("pilot", "uknn", "blend", "check"):
        raise SystemExit(__doc__)
    data = lr.load_data()
    users_all = cs.users_with_n_seeds(data["val_users"], N_SEEDS)     # validation only, never test
    users = users_all if FULL else users_all[:PILOT_USERS]
    if MODE == "check":
        check(data, users_all)
    elif MODE == "blend":
        run_blend(data, users_all, (VERSIONS or ["p5"])[0])      # always all 203 users (cached answers)
    elif MODE == "pilot":
        # full: only the model chosen from the pilot (MiniMax M3 dropped: below Hybrid E, 3 repaired answers)
        run_pilot(data, users, FULL_MODELS if FULL else NEW_MODELS)
    else:
        check_uknn(data, users_all)      # stop before any call if the candidate lists are not option 11's
        run_uknn(data, users)
