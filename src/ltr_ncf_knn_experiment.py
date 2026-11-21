'''
Doc 20 section 10, experiments L1 + L2 (and L6 with 3 seeds): do NCF and user-kNN help the learned
ranker (LTR v2: tuned XGBoost + LightGBM ensemble, R7 features, Hybrid E's top 30)?

New feature group G6 (one value per user and candidate):
  ncf_score    NCF (NeuMF) fold-in logit of the candidate, from the user's seeds
  ncf_rank     its rank among the user's 30 candidates (1 = best)
  uknn_score   user-kNN score: similarity-weighted mean of the neighbours' centred ratings
               (the standard sum divided by the neighbours' total similarity: same order within a user,
               values comparable across users and across neighbour pools of different size)
  uknn_rank    its rank among the 30

Feature sets (fixed in advance):  R7 (LTR v2) | R8a = R7 + NCF | R8b = R7 + user-kNN | R8 = R7 + both

Cross-fitting (as doc 16; no user's own ratings ever score them):
  simulated warm users of half A  ->  NCF trained on half B only, user-kNN neighbours from half B only
  simulated warm users of half B  ->  NCF trained on half A only, neighbours from half A only
  validation users                ->  NCF trained on all warm users (model 42), neighbours = all warm users
  Neighbours k: 200 for all warm users (doc 20), scaled to the same share of a half (about 100).
  NCF fold-in: the doc-20 settings (10 seeds 1,000 steps / lr 0.05; 3 seeds 300 / 0.01).

L1 evaluation (validation users never used for any choice):
  1. held-out simulated users: train on 80% of the simulated users, score the other 20% (the split of
     doc 16 step 4, seed 42; about 1,816 users)
  2. the 203 validation users: train on all simulated users
  Models: tuned XGBoost, tuned LightGBM (step4_chosen_params.json, no retuning) and their 50/50 rank
  ensemble (the LTR v2 recipe). Check before anything else: R7 ensemble on validation reproduces LTR v2
  (0.6686 with 10 seeds, 0.6227 with 3).
  Tests: ensemble R8a / R8b / R8 vs R7, paired Wilcoxon, Bonferroni over 3, separately on held-out and
  validation users. Decision rule (doc 20 section 10, fixed before running): a feature group is adopted
  only if its ensemble gains at least +0.005 over R7 on BOTH the held-out and the validation users.

L2 importance (descriptive):
  gain share and mean |SHAP| of every R8 feature (both libraries, validation users); each G6 feature added
  alone to R7 (default settings, validation users, as doc 16 section 3.2); within-user Spearman correlation
  of ncf_score and uknn_score with the existing collaborative features (is NCF redundant?).

The test users are never touched. Nothing calls an LLM.

Cost (first run, then cached): two half-NCF trainings (about 20 minutes each, shared by 10 and 3 seeds),
NCF fold-in for about 9,300 users (roughly 1-2 h with 10 seeds, less with 3), then about 16 model fits for
L1 and 8 for L2 (roughly 30-60 minutes).

Usage:
    python src/ltr_ncf_knn_experiment.py          10 seeds (L1 + L2)
    python src/ltr_ncf_knn_experiment.py 3        3 seeds (L6)
Outputs: output/experiments/ltr_ncf_knn/






New user's seeds (3 or 10 liked films)
        │
        ▼
Hybrid E → top 30 candidate films            (unchanged)
        │
        ▼
For each of the 30 films, build a feature row:
   25 old features (R7)                      (unchanged: item-CF, popularity, content,
                                               LLM labels, SVD, co-visitation…)
 + 4 NEW features (G6):
     ncf_score, ncf_rank                     ← "what does NCF think of this film?"
     uknn_score, uknn_rank                   ← "what do similar users think of it?"
        │
        ▼
XGBoost + LightGBM (tuned, as LTR v2) → score each film
        │
        ▼
Average the two models' ranks → final order → top 10 recommended





'''
import sys

N_SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
sys.argv = [sys.argv[0]]                 # the imported experiment scripts read sys.argv when imported
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from tqdm import tqdm

import cold_start_experiment as cs
import llm_rerank_experiment as lr
import LightGBM_XBoost_rerank_experiment as ltr
import cold_start_baselines as cbl

OUT = Path("output") / "experiments" / "ltr_ncf_knn"
G6 = {"ncf": ["ncf_score", "ncf_rank"], "knn": ["uknn_score", "uknn_rank"]}
R7 = [c for group in ltr.MODELS["R7"] for c in ltr.GROUPS[group]]
FEATURE_SETS = {"R7": R7,
                "R8a = R7 + NCF": R7 + G6["ncf"],
                "R8b = R7 + user-kNN": R7 + G6["knn"],
                "R8 = R7 + both": R7 + G6["ncf"] + G6["knn"]}
INPUT_TABLES = {10: ("features_val_g5.csv", "features_simulated_g5.csv"),
                3: ("features_val_n3.csv", "features_simulated_n3.csv")}
EXPECTED_R7 = {10: 0.6686, 3: 0.6227}           # LTR v2 on validation (docs 16 and 20)
MIN_GAIN = 0.005                                # decision rule, fixed before running
COMPARE_WITH = ["cf_mean", "cf_max", "hybrid_score", "log_popularity", "svd_score", "svd_mean",
                "covis_max", "covis_mean"]


# ───────────────────────── G6 features ─────────────────────────
def add_g6(table, users, ncf, knn, k, column):
    """G6 columns for a feature table (30 rows per user). ncf: NCFFoldIn; knn: UserKNN; both built
    WITHOUT these users' ratings."""
    user_of = {u["user_id"]: u for u in users}
    n_pool = len(column)
    ncf_score = np.full(len(table), np.nan)
    knn_score = np.full(len(table), np.nan)
    for user_id, group in tqdm(table.groupby("user_id", sort=False), desc="G6 features", leave=False):
        user = user_of[user_id]
        seed_cols = [column[m] for m in user["liked"]]
        allowed = np.ones(n_pool, dtype=bool)
        allowed[[column[m] for m in user["seen"]]] = False
        cand = [column[int(m)] for m in group["movie_id"]]
        idx = table.index.get_indexer(group.index)
        ncf_score[idx] = ncf.scores(user_id, seed_cols, allowed)[cand]
        knn_score[idx] = knn.knn_scores(seed_cols, k, normalise=True)[cand]
    out = table.assign(ncf_score=ncf_score, uknn_score=knn_score)
    by_user = out.groupby("user_id", sort=False)
    out["ncf_rank"] = by_user["ncf_score"].rank(ascending=False, method="first", na_option="bottom")
    out["uknn_rank"] = by_user["uknn_score"].rank(ascending=False, method="first")
    return out


def build_tables(data):
    """Validation and simulated tables of LTR v2 (cached, unchanged) + G6, cross-fitted. Cached."""
    paths = {split: OUT / f"features_{split}_g6_n{N_SEEDS}.csv" for split in ("val", "simulated")}
    if all(p.exists() for p in paths.values()):
        print(f"Loaded {paths['val'].name} and {paths['simulated'].name}")
        return {split: pd.read_csv(p) for split, p in paths.items()}

    val_name, sim_name = INPUT_TABLES[N_SEEDS]
    val = pd.read_csv(ltr.FOLDER / val_name)
    simulated = pd.read_csv(ltr.FOLDER / sim_name)
    train, in_A, users_by_half = ltr.warm_halves(data, N_SEEDS)
    assert len(train) == len(data["train"]), "the warm users differ from the cold-start split"

    pool_ids = [int(data["movie_ids"][p]) for p in data["pool"]]
    column = {m: i for i, m in enumerate(pool_ids)}
    k_full = cbl.KNN_NEIGHBOURS[N_SEEDS]
    n_warm = train["user_id"].nunique()

    print(f"Validation users ({N_SEEDS} seeds): NCF of all warm users (model 42), all {n_warm:,} warm users "
          f"as neighbours (k = {k_full})")
    val_users = cs.users_with_n_seeds(data["val_users"], N_SEEDS)
    full_ncf = cbl.NCFFoldIn(cbl.train_ncf(data["train"]), pool_ids)
    full_knn = cbl.UserKNN(data["train"], column, len(pool_ids))
    tables = {"val": add_g6(val, val_users, full_ncf, full_knn, k_full, column)}

    parts = []
    for half, other in (("A", train[~in_A]), ("B", train[in_A])):
        other_name = "B" if half == "A" else "A"
        ncf = cbl.NCFFoldIn(cbl.train_ncf(other, path=cbl.FOLDER / f"ncf_half{other_name}.npz"), pool_ids)
        knn = cbl.UserKNN(other, column, len(pool_ids))
        k = max(1, round(k_full * knn.n_users / n_warm))
        print(f"Simulated users of half {half}: NCF trained on half {other_name} only "
              f"({(~ncf.known).sum()} pool films it never saw), {knn.n_users:,} neighbours' pool, k = {k}")
        part = simulated[simulated["half"] == half]
        parts.append(add_g6(part, users_by_half[half], ncf, knn, k, column))
    tables["simulated"] = pd.concat(parts).loc[simulated.index]          # original row order

    for split, t in tables.items():
        assert (t.groupby("user_id").size() == ltr.TOP_M).all(), f"not 30 rows per user ({split})"
        assert t["uknn_score"].notna().all(), f"missing user-kNN values ({split})"
        print(f"{split}: {t['ncf_score'].isna().sum()} candidate rows without an NCF score (films unseen "
              f"by the half model; left missing, both libraries handle NaN)")
        t.to_csv(paths[split], index=False)
    return tables


def shift_check(tables):
    """Feature means and quartiles, validation vs simulated users (cross-fitting must not shift them)."""
    rows = []
    for c in G6["ncf"][:1] + G6["knn"][:1]:
        for split, t in tables.items():
            q = t[c].quantile([0.25, 0.5, 0.75])
            rows.append({"feature": c, "users": split, "mean": t[c].mean(), "q25": q[0.25], "median": q[0.5],
                         "q75": q[0.75]})
    table = pd.DataFrame(rows)
    table.to_csv(OUT / f"l1_feature_shift_n{N_SEEDS}.csv", index=False)
    print("\nG6 feature distributions (validation vs cross-fitted simulated users):")
    print(table.round(4).to_string(index=False))


# ───────────────────────── L1 ─────────────────────────
def holdout_split(simulated):
    """The held-out simulated users of doc 16 step 4 (20%, seed 42)."""
    user_ids = np.sort(simulated["user_id"].unique())
    held = set(np.random.default_rng(42).permutation(user_ids)[:int(len(user_ids) * ltr.TUNING_HOLDOUT)])
    return simulated["user_id"].isin(held)


def fit_and_score(train, table, columns, chosen, relevant_of):
    """Tuned XGBoost, tuned LightGBM and their 50/50 rank ensemble: NDCG@10 per user, plus the models."""
    models, ranks, result = {}, {}, {}
    for library in ("xgboost", "lightgbm"):
        model = ltr.fit_ranker(train, columns, library, chosen[library])
        scores = model.predict(table[columns])
        models[library] = model
        ranks[library] = ltr.within_user_rank(table, scores)
        result[library] = ltr.ndcg_per_user(table.assign(score=scores), "score", relevant_of, ascending=False)
    mixed = (ranks["xgboost"] + ranks["lightgbm"]) / 2
    result["ensemble"] = ltr.ndcg_per_user(table.assign(score=mixed), "score", relevant_of, ascending=False)
    return result, models


def paired_tests(per_user, reference, others):
    rows = []
    for name in others:
        diff = (per_user[name] - per_user[reference]).dropna()
        rows.append({"comparison": f"{name} vs {reference}", "difference": diff.mean(),
                     "better": int((diff > 0).sum()), "worse": int((diff < 0).sum()),
                     "p_value": wilcoxon(diff).pvalue if diff.any() else 1.0})
    tests = pd.DataFrame(rows)
    tests["p_bonferroni"] = (tests["p_value"] * len(tests)).clip(upper=1.0)
    return tests


def run_l1(tables, relevant_val, relevant_sim, chosen):
    val, simulated = tables["val"], tables["simulated"]
    is_held = holdout_split(simulated)
    settings = {"held-out simulated users": (simulated[~is_held], simulated[is_held], relevant_sim),
                "validation users": (simulated, val, relevant_val)}
    per_user, summary, models = {}, [], {}
    for where, (train, table, relevant) in settings.items():
        per_user[where] = {"Hybrid E": ltr.ndcg_per_user(table, "hybrid_rank", relevant, ascending=True)}
        for name, columns in FEATURE_SETS.items():
            print(f"L1 | {where} | {name}: fitting tuned XGBoost + LightGBM...", flush=True)
            result, fitted = fit_and_score(train, table, columns, chosen, relevant)
            for library, values in result.items():
                per_user[where][f"{name} | {library}"] = values
                summary.append({"users": where, "n_users": len(values), "features": name, "model": library,
                                "ndcg": values.mean()})
            if where == "validation users":
                models[name] = fitted
        if where == "validation users":
            got = per_user[where]["R7 | ensemble"].mean()
            print(f"check: R7 ensemble on validation = {got:.4f} (LTR v2, expected {EXPECTED_R7[N_SEEDS]})")
            assert abs(got - EXPECTED_R7[N_SEEDS]) < 5e-5, "R7 does not reproduce LTR v2: stop"

    tests, decision = [], []
    for where, values in per_user.items():
        frame = pd.DataFrame(values)
        frame.to_csv(OUT / f"l1_per_user_{'holdout' if 'held' in where else 'val'}_n{N_SEEDS}.csv")
        t = paired_tests(frame, "R7 | ensemble", [f"{n} | ensemble" for n in FEATURE_SETS if n != "R7"])
        tests.append(t.assign(users=where, n_users=len(frame)))
    tests = pd.concat(tests, ignore_index=True)
    for name in [n for n in FEATURE_SETS if n != "R7"]:
        gains = tests[tests["comparison"].str.startswith(name)].set_index("users")["difference"]
        decision.append({"features": name, "gain_held_out": gains["held-out simulated users"],
                         "gain_validation": gains["validation users"],
                         "adopted": bool((gains >= MIN_GAIN).all())})
    summary, decision = pd.DataFrame(summary), pd.DataFrame(decision)
    summary.to_csv(OUT / f"l1_summary_n{N_SEEDS}.csv", index=False)
    tests.to_csv(OUT / f"l1_tests_n{N_SEEDS}.csv", index=False)
    decision.to_csv(OUT / f"l1_decision_n{N_SEEDS}.csv", index=False)

    pd.set_option("display.width", 250)
    print(f"\n=== L1, {N_SEEDS} seeds: NDCG@10 ===")
    print(summary.pivot_table(index="features", columns=["users", "model"], values="ndcg", sort=False)
          .round(4).to_string())
    for where, values in per_user.items():
        print(f"Hybrid E, {where}: {values['Hybrid E'].mean():.4f}")
    print("\nEnsemble vs R7 (paired Wilcoxon, Bonferroni over 3 within each user group):")
    print(tests.round(4).to_string(index=False))
    print(f"\nDecision rule: adopt only if the ensemble gains >= +{MIN_GAIN} on BOTH user groups")
    print(decision.round(4).to_string(index=False))
    return models


# ───────────────────────── L2 ─────────────────────────
def run_l2(tables, relevant_val, models, chosen):
    val, simulated = tables["val"], tables["simulated"]
    columns = FEATURE_SETS["R8 = R7 + both"]
    rows = []
    for library, model in models["R8 = R7 + both"].items():
        if library == "lightgbm":
            gain = pd.Series(model.booster_.feature_importance("gain"), index=columns)
        else:
            gain = pd.Series(model.get_booster().get_score(importance_type="total_gain")).reindex(columns).fillna(0)
        shap = ltr.shap_values(model, val[columns], library).abs().mean()
        for c in columns:
            rows.append({"model": library, "feature": c, "group": "G6" if c in G6["ncf"] + G6["knn"] else "R7",
                         "gain_share": gain[c] / gain.sum(), "mean_abs_shap": shap[c],
                         "shap_share": shap[c] / shap.sum()})
    importance = pd.DataFrame(rows)
    importance.to_csv(OUT / f"l2_importance_n{N_SEEDS}.csv", index=False)

    # each G6 feature alone added to R7 (default settings, as doc 16 section 3.2; descriptive)
    single = []
    for library in ("xgboost", "lightgbm"):
        base = ltr.fit_ranker(simulated, R7, library)
        base_ndcg = ltr.ndcg_per_user(val.assign(score=base.predict(val[R7])), "score", relevant_val, False).mean()
        for c in G6["ncf"] + G6["knn"]:
            model = ltr.fit_ranker(simulated, R7 + [c], library)
            ndcg = ltr.ndcg_per_user(val.assign(score=model.predict(val[R7 + [c]])), "score", relevant_val,
                                     False).mean()
            single.append({"model": f"{library} (default settings)", "added to R7": c, "ndcg": ndcg,
                           "R7": base_ndcg, "gain": ndcg - base_ndcg})
    single = pd.DataFrame(single)
    single.to_csv(OUT / f"l2_single_feature_n{N_SEEDS}.csv", index=False)

    # redundancy: mean within-user Spearman correlation with the existing collaborative features
    corr = []
    for split, t in tables.items():
        ranks = t[["user_id", "ncf_score", "uknn_score"] + COMPARE_WITH].copy()
        cols = ["ncf_score", "uknn_score"] + COMPARE_WITH
        ranks[cols] = ranks.groupby("user_id")[cols].rank()
        for feature in ("ncf_score", "uknn_score"):
            for other in COMPARE_WITH + (["uknn_score"] if feature == "ncf_score" else []):
                per_user = ranks.groupby("user_id")[[feature, other]].apply(
                    lambda g: g[feature].corr(g[other]) if g[feature].nunique() > 1 and g[other].nunique() > 1
                    else np.nan)
                corr.append({"users": split, "feature": feature, "with": other, "spearman": per_user.mean()})
    corr = pd.DataFrame(corr)
    corr.to_csv(OUT / f"l2_correlation_n{N_SEEDS}.csv", index=False)

    print(f"\n=== L2, {N_SEEDS} seeds ===")
    print("Share of the R8 models' importance that goes to the G6 features:")
    print(importance.groupby(["model", "group"])[["gain_share", "shap_share"]].sum().round(3).to_string())
    print("\nG6 features (validation users):")
    print(importance[importance["group"] == "G6"].round(4).to_string(index=False))
    print("\nEach G6 feature added alone to R7 (default settings, validation, descriptive):")
    print(single.round(4).to_string(index=False))
    print("\nMean within-user Spearman correlation (high = the feature repeats what R7 already has):")
    print(corr.pivot_table(index=["feature", "with"], columns="users", values="spearman", sort=False)
          .round(3).to_string())


if __name__ == "__main__":
    if N_SEEDS not in INPUT_TABLES:
        sys.exit(f"Seeds must be 10 or 3, not {N_SEEDS}.\n{__doc__}")
    OUT.mkdir(parents=True, exist_ok=True)
    data = lr.load_data()
    test_ids = {u["user_id"] for u in data["test_users"]}
    relevant_val = {u["user_id"]: u["relevant"] for u in cs.users_with_n_seeds(data["val_users"], N_SEEDS)}
    _, _, users_by_half = ltr.warm_halves(data, N_SEEDS)
    relevant_sim = {u["user_id"]: u["relevant"] for users in users_by_half.values() for u in users}
    assert not (test_ids & (set(relevant_val) | set(relevant_sim))), "a test user is in this experiment"
    with open(ltr.FOLDER / "step4_chosen_params.json", encoding="utf-8") as f:
        chosen = json.load(f)

    tables = build_tables(data)
    shift_check(tables)
    models = run_l1(tables, relevant_val, relevant_sim, chosen)
    run_l2(tables, relevant_val, models, chosen)
    print(f"\nSaved to {OUT}")
