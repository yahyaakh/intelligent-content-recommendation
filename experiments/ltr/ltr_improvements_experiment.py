'''
Four attempts to improve LTR v2 (doc 16), ideas A, C, E and F of the improvement list (2026-10-27).
LTR v2 = tuned XGBoost + tuned LightGBM, 50/50 within-user rank ensemble, R7 features (25), Hybrid E's
top 30 (C30), trained on the cross-fitted simulated warm users.

  A  user-level features (G7, 9 columns). Describe the user as a whole, so the trees can learn
     "this user likes niche films, trust popularity less":
       seed_pop_mean, seed_pop_std     log popularity of the seeds
       seed_year_mean, seed_year_std   release year of the seeds
       seed_cf_cohesion                mean item-CF similarity between pairs of seeds
       seed_emb_cohesion               mean content-embedding similarity between pairs of seeds
       seed_genre_spread               distinct genres among the seeds / number of seeds
       pop_gap, year_gap               candidate minus seed mean (log popularity, year)
  E  seed ratings (G8, 4 columns). A DIFFERENT INFORMATION SETTING: the model is given the stars the
     user gave each seed, not only "liked". Reported separately, like SVD in doc 20.
       seed_rating_mean, seed_rating_spread (max - min)
       cf_wmean      item-CF similarity to the seeds weighted by w = rating - min(seed ratings) + 0.5
       cf_top_mean   mean item-CF similarity to the seeds with the user's highest seed rating
     Plus a no-training check: Hybrid E-w = Hybrid E with cf_wmean in place of the plain mean (same alpha).
  C  candidates = C30 + the 20 best user-kNN films outside C30 (up to 50 per user; doc 20 L5).
       C   R7 on the wider list
       C+  R7 + uknn_score + uknn_rank on the wider list (else the trees cannot see why a film is there)
  F  score blend on C30: (1 - w) x LTR v2 rank + w x user-kNN rank (within-user percentile ranks).
     w from {0, 0.1, ..., 1}: for the held-out users by 2-fold cross-fitting (chosen on one half of
     them, applied to the other), for the validation users chosen on all held-out users.

Cross-fitting as everywhere in docs 16/20: a simulated user of half A gets item-CF, popularity, latent
tables and user-kNN neighbours from half B only, and the other way round. Validation users get tables
from all warm users. user-kNN: k = 200 for all warm users (doc 20), scaled to the share of a half.
Seed ratings: the user's own ratings of the seeds (validation: cold ratings; simulated: their own).

Evaluation (the L1 harness of doc 20 section 10.1, unchanged):
  1. held-out simulated users: train on 80% of the simulated users, score the other 20% (doc 16 split,
     seed 42; about 1,816 users)
  2. the 203 validation users: train on all simulated users
  Models: tuned XGBoost, tuned LightGBM (step4_chosen_params.json, no retuning) and their ensemble.
  Checks before any result: the R7/C30 ensemble reproduces LTR v2 on validation (0.6686 / 0.6227);
  Hybrid E and user-kNN reproduce 0.6286 / 0.6417 (10 seeds); the first 30 rows of the C tables
  reproduce the cached LTR v2 features; cf_mean and log_popularity recomputed here equal the cache.
  Tests: ensemble A, E, C, C+, F vs the R7/C30 ensemble, paired Wilcoxon, Bonferroni over 5 within each
  user group. Decision rule (fixed before running, as L1): an idea is adopted only if it gains at
  least +0.005 over LTR v2 on BOTH the held-out and the validation users. An adopted idea still needs
  a pre-registered test-set run; the test users are never touched here. Nothing calls an LLM.

Usage:
    python experiments/ltr/ltr_improvements_experiment.py           10 seeds
    python experiments/ltr/ltr_improvements_experiment.py 3         3 seeds
    python experiments/ltr/ltr_improvements_experiment.py 10 dry    small run (40 validation + 300 simulated users) to
                                                    test the code; results meaningless, written to dry/
Outputs: output/experiments/ltr_improvements/
'''

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))  # shared modules live in src/
import sys

N_SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
DRY = len(sys.argv) > 2 and sys.argv[2] == "dry"
sys.argv = [sys.argv[0]]                 # the imported experiment scripts read sys.argv when imported
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from tqdm import tqdm

import build_movie_texts as bmt
import cold_start_experiment as cs
import llm_rerank_experiment as lr
import LightGBM_XBoost_rerank_experiment as ltr
import cold_start_baselines as cbl

OUT = Path("output") / "experiments" / "ltr_improvements" / ("dry" if DRY else "")
R7 = [c for group in ltr.MODELS["R7"] for c in ltr.GROUPS[group]]
G7_USER = ["seed_pop_mean", "seed_pop_std", "seed_year_mean", "seed_year_std", "seed_cf_cohesion",
           "seed_emb_cohesion", "seed_genre_spread", "pop_gap", "year_gap"]
G8_RATING = ["seed_rating_mean", "seed_rating_spread", "cf_wmean", "cf_top_mean"]
G_KNN = ["uknn_score", "uknn_rank"]
EXTRA_KNN = 20                                   # C: user-kNN films added to C30
BLEND_GRID = np.round(np.arange(0, 1.01, 0.1), 1)
INPUT_TABLES = {10: ("features_val_g5.csv", "features_simulated_g5.csv"),
                3: ("features_val_n3.csv", "features_simulated_n3.csv")}
EXPECTED = {10: {"LTR v2": 0.6686, "Hybrid E": 0.6286, "user-kNN": 0.6417},
            3: {"LTR v2": 0.6227, "Hybrid E": 0.5837, "user-kNN": 0.6053}}
MIN_GAIN = 0.005                                 # decision rule, fixed before running
DRY_USERS = {"val": 40, "simulated": 300}
# name -> (candidate list, feature columns); F is a blend, not a feature set
FEATURE_SETS = {"LTR v2 (R7, C30)": ("c30", R7),
                "A: + user-level features": ("c30", R7 + G7_USER),
                "E: + seed ratings": ("c30", R7 + G8_RATING),
                "C: C30 + 20 user-kNN films": ("union", R7),
                "C+: C30 + 20 user-kNN films, + kNN features": ("union", R7 + G_KNN)}
REFERENCE = "LTR v2 (R7, C30)"
BLEND = "F: blend LTR v2 + user-kNN"


# ───────────────────────── per-user helpers ─────────────────────────
def hybrid_full(user, view, weights=None):
    """Hybrid E over every unseen pool film (as ltr.wide_candidates). weights: one per seed for the
    rating-weighted item-CF mean (Hybrid E-w); None = the plain mean (Hybrid E).
    Returns candidate positions, scores and their order (best first)."""
    position_of = view["position_of"]
    seen = {position_of[m] for m in user["seen"]}
    cand = np.array([p for p in view["pool"] if p not in seen])
    liked = [position_of[m] for m in user["liked"]]
    block = view["item_cf"][np.ix_(liked, cand)]
    personal = block.mean(axis=0) if weights is None else (weights[:, None] * block).sum(axis=0) / weights.sum()
    alpha = view["alpha_E"][len(user["liked"])]
    score = alpha * cs.rescale(personal) + (1 - alpha) * cs.rescale(view["pop_vector"][cand])
    return cand, score, np.argsort(-score, kind="stable")


def rating_weights(ratings):
    return ratings - ratings.min() + 0.5


def pairwise_mean(block):
    """Mean of the off-diagonal entries of a seeds x seeds similarity block."""
    n = len(block)
    return float((block.sum() - np.trace(block)) / (n * (n - 1))) if n > 1 else 0.0


def ndcg_of(ids, relevant):
    return bmt.ndcg_at_k(list(ids)[:bmt.K], relevant, bmt.K)


# ───────────────────────── views (tables built without the scored users) ─────────────────────────
def build_views(data, train, in_A, with_latent):
    """'full' = all warm users (validation); 'A' / 'B' = tables for the simulated users of that half,
    built from the OTHER half. Each view also holds the user-kNN model and its k."""
    pool_ids = [int(data["movie_ids"][p]) for p in data["pool"]]
    column = {m: i for i, m in enumerate(pool_ids)}
    n_warm = train["user_id"].nunique()
    k_full = cbl.KNN_NEIGHBOURS[N_SEEDS]
    views = {"full": dict(data)}
    views["full"]["knn"] = cbl.UserKNN(train, column, len(pool_ids))
    views["full"]["k"] = k_full
    if with_latent:
        print("latent tables: full warm users", flush=True)
        views["full"].update(ltr.latent_tables(train, data))
    for half, other in (("A", train[~in_A]), ("B", train[in_A])):
        print(f"tables for half {half}: built from the other half only", flush=True)
        view = ltr.half_view(data, other)
        view["knn"] = cbl.UserKNN(other, column, len(pool_ids))
        view["k"] = max(1, round(k_full * view["knn"].n_users / n_warm))
        if with_latent:
            view.update(ltr.latent_tables(other, data))
        views[half] = view
    for v in views.values():
        v["column"] = column
        v["pool_col"] = {p: i for i, p in enumerate(data["pool"])}
    return views


def knn_vector(user, view):
    """Normalised user-kNN score of every pool film (pool order)."""
    return view["knn"].knn_scores([view["column"][m] for m in user["liked"]], view["k"], normalise=True)


# ───────────────────────── new columns ─────────────────────────
def add_columns(table, users, view, rating_of):
    """G7 (user-level), G8 (seed ratings) and user-kNN columns for an existing feature table.
    Also recomputes cf_mean and log_popularity as a check that `view` is the view the table came from."""
    user_of = {u["user_id"]: u for u in users}
    pos, movies = view["position_of"], view["movie_by_id"]
    year_of = movies["year"].astype(float)
    genres_of = movies["genres"]
    new = {c: np.zeros(len(table)) for c in G7_USER + G8_RATING + ["uknn_score", "check_cf_mean", "check_pop"]}
    for user_id, group in tqdm(table.groupby("user_id", sort=False), desc="new columns", leave=False):
        user = user_of[user_id]
        seeds = [pos[m] for m in user["liked"]]
        cands = [pos[int(m)] for m in group["movie_id"]]
        idx = table.index.get_indexer(group.index)
        cf = view["item_cf"][np.ix_(seeds, cands)]
        seed_pop = view["pop_vector"][seeds]
        seed_year = year_of.loc[user["liked"]].to_numpy()
        genres = {g for m in user["liked"] for g in lr.split_field(genres_of.loc[m])}
        values = {
            "seed_pop_mean": seed_pop.mean(), "seed_pop_std": seed_pop.std(),
            "seed_year_mean": seed_year.mean(), "seed_year_std": seed_year.std(),
            "seed_cf_cohesion": pairwise_mean(view["item_cf"][np.ix_(seeds, seeds)]),
            "seed_emb_cohesion": pairwise_mean(view["emb_base"][np.ix_(seeds, seeds)]),
            "seed_genre_spread": len(genres) / len(seeds),
        }
        for c, v in values.items():
            new[c][idx] = v
        new["pop_gap"][idx] = view["pop_vector"][cands] - values["seed_pop_mean"]
        new["year_gap"][idx] = group["year"].to_numpy(float) - values["seed_year_mean"]

        stars = np.array([rating_of[(user_id, m)] for m in user["liked"]], dtype=float)
        w = rating_weights(stars)
        top = stars == stars.max()
        new["seed_rating_mean"][idx] = stars.mean()
        new["seed_rating_spread"][idx] = stars.max() - stars.min()
        new["cf_wmean"][idx] = (w[:, None] * cf).sum(axis=0) / w.sum()
        new["cf_top_mean"][idx] = cf[top].mean(axis=0)

        knn = knn_vector(user, view)
        new["uknn_score"][idx] = knn[[view["pool_col"][p] for p in cands]]
        new["check_cf_mean"][idx] = cf.mean(axis=0)
        new["check_pop"][idx] = view["pop_vector"][cands]
    out = table.assign(**new)
    out["uknn_rank"] = out.groupby("user_id", sort=False)["uknn_score"].rank(ascending=False, method="first")
    for check, original in (("check_cf_mean", "cf_mean"), ("check_pop", "log_popularity")):
        worst = float(np.abs(out[check] - out[original]).max())
        assert worst < 1e-4, f"{original} recomputed differs from the cached table by {worst}: wrong view"
    return out.drop(columns=["check_cf_mean", "check_pop"])


def union_rows(user, view):
    """C: Hybrid E's top 30 + the 20 best user-kNN films outside them, as candidate_rows input."""
    cand, score, order = hybrid_full(user, view)
    rank = np.empty(len(order), int)
    rank[order] = np.arange(1, len(order) + 1)
    knn = knn_vector(user, view)[[view["pool_col"][p] for p in cand]]
    rest = order[ltr.TOP_M:]
    extra = rest[np.argsort(-knn[rest], kind="stable")[:EXTRA_KNN]]
    chosen = np.concatenate([order[:ltr.TOP_M], extra])
    ids = [int(m) for m in view["movie_ids"][cand[chosen]]]
    return ids, score[chosen], rank[chosen].tolist()


# ───────────────────────── tables ─────────────────────────
def select_users(users, n):
    return users[:n] if DRY else users


def build_tables(data):
    paths = {(split, cands): OUT / f"features_{split}_{cands}_n{N_SEEDS}.csv"
             for split in ("val", "simulated") for cands in ("c30", "union")}
    train, in_A, users_by_half = ltr.warm_halves(data, N_SEEDS)
    assert len(train) == len(data["train"]), "the warm users differ from the cold-start split"
    val_users = select_users(cs.users_with_n_seeds(data["val_users"], N_SEEDS), DRY_USERS["val"])
    users_by_half = {h: select_users(u, DRY_USERS["simulated"] // 2) for h, u in users_by_half.items()}
    users = {"val": val_users, "simulated": users_by_half["A"] + users_by_half["B"]}
    if all(p.exists() for p in paths.values()):
        print("Loaded the cached tables")
        return {k: pd.read_csv(p) for k, p in paths.items()}, users, None

    # the users' own ratings of their seeds (E)
    ratings = bmt.prepare_ratings(data["movies"]).sort_values("timestamp", kind="stable")
    ids = {u["user_id"] for group in users.values() for u in group}
    ratings = ratings[ratings["user_id"].isin(ids)].drop_duplicates(["user_id", "movie_id"], keep="last")
    rating_of = dict(zip(zip(ratings["user_id"], ratings["movie_id"]), ratings["rating"]))

    views = build_views(data, train, in_A, with_latent=True)
    val_name, sim_name = INPUT_TABLES[N_SEEDS]
    cached = {"val": pd.read_csv(ltr.FOLDER / val_name), "simulated": pd.read_csv(ltr.FOLDER / sim_name)}
    for split in cached:
        keep = {u["user_id"] for u in users[split]}
        cached[split] = cached[split][cached[split]["user_id"].isin(keep)]
        assert cached[split]["user_id"].nunique() == len(keep), f"cached {split} table lacks users"

    tables = {("val", "c30"): add_columns(cached["val"], val_users, views["full"], rating_of)}
    parts = [add_columns(cached["simulated"][cached["simulated"]["half"] == h], users_by_half[h], views[h],
                         rating_of) for h in ("A", "B")]
    tables[("simulated", "c30")] = pd.concat(parts).loc[cached["simulated"].index]

    # C: wider candidate lists, every feature rebuilt for them
    def union_table(group_users, view, half=None):
        rows = []
        for u in tqdm(group_users, desc=f"C tables ({half or 'val'})"):
            for r in ltr.candidate_rows(u, view, False, union_rows(u, view)):
                if half:
                    r["half"] = half
                rows.append(r)
        table = ltr.add_g5(pd.DataFrame(rows), group_users, view)
        return add_columns(table, group_users, view, rating_of)

    tables[("val", "union")] = union_table(val_users, views["full"])
    tables[("simulated", "union")] = pd.concat(
        [union_table(users_by_half[h], views[h], h) for h in ("A", "B")], ignore_index=True)

    # Check: the C30 part of the wider lists equals the cached LTR v2 features
    for split in ("val", "simulated"):
        key = ["user_id", "movie_id"]
        merged = tables[(split, "union")].merge(cached[split][key + R7 + ["label"]], on=key, suffixes=("", "_old"))
        merged = merged[merged["hybrid_rank"] <= ltr.TOP_M]
        assert len(merged) == len(cached[split]), f"C30 is not inside the wider lists ({split})"
        worst = max(float(np.abs(merged[c] - merged[f"{c}_old"]).max()) for c in R7 + ["label"])
        print(f"check: C30 rows of the wider lists = cached LTR v2 features ({split}), largest difference {worst:.1e}")
        assert worst < 1e-4, "the wider lists do not reproduce the LTR v2 features"

    for key, path in paths.items():
        t = tables[key]
        assert t[R7 + G7_USER + G8_RATING + G_KNN].notna().all().all(), f"missing values in {key}"
        t.to_csv(path, index=False)
    return tables, users, views


# ───────────────────────── baselines without training ─────────────────────────
def full_ranking_baselines(data, users, views, holdout_ids, half_of):
    """Hybrid E, Hybrid E-w (seed ratings) and user-kNN over every unseen pool film, per user.
    Simulated users: held-out ones only, with their half's view. Cached."""
    path = OUT / f"baselines_n{N_SEEDS}.csv"
    if path.exists():
        return pd.read_csv(path, index_col=["users", "user_id"])
    if views is None:
        train, in_A, _ = ltr.warm_halves(data, N_SEEDS)
        views = build_views(data, train, in_A, with_latent=False)
    ratings = bmt.prepare_ratings(data["movies"]).sort_values("timestamp", kind="stable")
    ratings = ratings.drop_duplicates(["user_id", "movie_id"], keep="last")
    rating_of = ratings.set_index(["user_id", "movie_id"])["rating"]
    rows = []
    groups = {"validation users": users["val"],
              "held-out simulated users": [u for u in users["simulated"] if u["user_id"] in holdout_ids]}
    for where, group in groups.items():
        for u in tqdm(group, desc=f"baselines ({where})"):
            view = views["full"] if where == "validation users" else views[half_of[u["user_id"]]]
            stars = rating_of.loc[[(u["user_id"], m) for m in u["liked"]]].to_numpy(float)
            cand, _, order = hybrid_full(u, view)
            _, _, order_w = hybrid_full(u, view, rating_weights(stars))
            knn = knn_vector(u, view)[[view["pool_col"][p] for p in cand]]
            ids = view["movie_ids"][cand]
            rows.append({"users": where, "user_id": u["user_id"],
                         "Hybrid E": ndcg_of(ids[order], u["relevant"]),
                         "Hybrid E-w (seed ratings)": ndcg_of(ids[order_w], u["relevant"]),
                         "user-kNN": ndcg_of(ids[np.argsort(-knn, kind="stable")], u["relevant"])})
    table = pd.DataFrame(rows).set_index(["users", "user_id"])
    table.to_csv(path)
    return table


# ───────────────────────── evaluation ─────────────────────────
def holdout_ids(simulated):
    """The held-out simulated users of doc 16 step 4 (20%, seed 42), from the FULL simulated table."""
    user_ids = np.sort(simulated["user_id"].unique())
    return set(np.random.default_rng(42).permutation(user_ids)[:int(len(user_ids) * ltr.TUNING_HOLDOUT)])


def fit_and_score(train, table, columns, chosen, relevant_of):
    """Tuned XGBoost, tuned LightGBM and their 50/50 rank ensemble: NDCG@10 per user + ensemble ranks."""
    ranks, result = {}, {}
    for library in ("xgboost", "lightgbm"):
        scores = ltr.fit_ranker(train, columns, library, chosen[library]).predict(table[columns])
        ranks[library] = ltr.within_user_rank(table, scores)
        result[library] = ltr.ndcg_per_user(table.assign(score=scores), "score", relevant_of, ascending=False)
    mixed = (ranks["xgboost"] + ranks["lightgbm"]) / 2
    result["ensemble"] = ltr.ndcg_per_user(table.assign(score=mixed), "score", relevant_of, ascending=False)
    return result, mixed


def blend_ndcg(table, ltr_rank, w, relevant_of):
    knn_rank = ltr.within_user_rank(table, table["uknn_score"].to_numpy())
    return ltr.ndcg_per_user(table.assign(score=(1 - w) * ltr_rank + w * knn_rank), "score", relevant_of, False)


def choose_w(table, ltr_rank, relevant_of):
    curve = {w: blend_ndcg(table, ltr_rank, w, relevant_of).mean() for w in BLEND_GRID}
    return max(curve, key=lambda w: (curve[w], -w)), curve        # ties: the smaller weight


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


def run(tables, users, baselines, held, chosen):
    relevant_of = {u["user_id"]: u["relevant"] for group in users.values() for u in group}
    per_user, summary, blend_info = {}, [], []
    for where in ("held-out simulated users", "validation users"):
        per_user[where] = {}
        for name, (cands, columns) in FEATURE_SETS.items():
            sim, val = tables[("simulated", cands)], tables[("val", cands)]
            is_held = sim["user_id"].isin(held)
            train, table = (sim[~is_held], sim[is_held]) if where.startswith("held") else (sim, val)
            print(f"{where} | {name}: fitting tuned XGBoost + LightGBM...", flush=True)
            result, mixed = fit_and_score(train, table, columns, chosen, relevant_of)
            for library, values in result.items():
                per_user[where][f"{name} | {library}"] = values
                summary.append({"users": where, "n_users": len(values), "method": name, "model": library,
                                "ndcg": values.mean()})
            if name == REFERENCE:
                reference_table, reference_rank = table, mixed
            if name == REFERENCE and where.startswith("held"):
                held_rank = mixed

        # F: blend of LTR v2 with user-kNN on C30
        if where.startswith("held"):
            ids = np.sort(reference_table["user_id"].unique())
            fold = dict(zip(ids, np.random.default_rng(42).permutation(len(ids)) % 2))
            in_fold = reference_table["user_id"].map(fold)
            values = []
            for f in (0, 1):
                fit_part, apply_part = in_fold != f, in_fold == f
                w, _ = choose_w(reference_table[fit_part], reference_rank[fit_part], relevant_of)
                values.append(blend_ndcg(reference_table[apply_part], reference_rank[apply_part], w, relevant_of))
                blend_info.append({"users": where, "fold": f, "w": w})
            blended = pd.concat(values)
            w_val, curve = choose_w(reference_table, held_rank, relevant_of)   # for the validation users
            blend_info.append({"users": "all held-out (used for validation)", "fold": "-", "w": w_val})
        else:
            blended = blend_ndcg(reference_table, reference_rank, w_val, relevant_of)
            val_curve = {w: blend_ndcg(reference_table, reference_rank, w, relevant_of).mean() for w in BLEND_GRID}
        per_user[where][f"{BLEND} | ensemble"] = blended
        summary.append({"users": where, "n_users": len(blended), "method": BLEND, "model": "ensemble",
                        "ndcg": blended.mean()})
        for name in baselines.columns:
            values = baselines.loc[where][name]
            per_user[where][name] = values
            summary.append({"users": where, "n_users": len(values), "method": name, "model": "-",
                            "ndcg": values.mean()})
        oracle = {"C30": ltr.oracle_ndcg(reference_table, relevant_of)}
        union_part = tables[("val" if where.startswith("val") else "simulated", "union")]
        union_part = union_part[union_part["user_id"].isin(reference_table["user_id"])]
        oracle["C30 + 20 user-kNN"] = ltr.oracle_ndcg(union_part, relevant_of)
        for name, value in oracle.items():
            summary.append({"users": where, "n_users": reference_table["user_id"].nunique(),
                            "method": f"ceiling: all relevant candidates first ({name})", "model": "-", "ndcg": value})

    summary = pd.DataFrame(summary)
    if not DRY:
        got = {"LTR v2": summary.query("users == 'validation users' and method == @REFERENCE and model == 'ensemble'")["ndcg"].item(),
               "Hybrid E": summary.query("users == 'validation users' and method == 'Hybrid E'")["ndcg"].item(),
               "user-kNN": summary.query("users == 'validation users' and method == 'user-kNN'")["ndcg"].item()}
        for name, value in got.items():
            print(f"check: {name} on validation = {value:.4f} (expected {EXPECTED[N_SEEDS][name]})")
            assert abs(value - EXPECTED[N_SEEDS][name]) < 5e-5, f"{name} does not reproduce: stop"

    candidates = [f"{n} | ensemble" for n in FEATURE_SETS if n != REFERENCE] + [f"{BLEND} | ensemble"]
    tests, decision = [], []
    for where, values in per_user.items():
        frame = pd.DataFrame(values)
        frame.to_csv(OUT / f"per_user_{'holdout' if where.startswith('held') else 'val'}_n{N_SEEDS}.csv")
        tests.append(paired_tests(frame, f"{REFERENCE} | ensemble", candidates).assign(users=where, n_users=len(frame)))
        extra = paired_tests(frame, "Hybrid E", ["Hybrid E-w (seed ratings)"]).assign(users=where, n_users=len(frame))
        extra["p_bonferroni"] = extra["p_value"]          # separate question: one test
        tests.append(extra)
    tests = pd.concat(tests, ignore_index=True)
    for name in candidates:
        gains = tests[tests["comparison"].str.startswith(name)].set_index("users")["difference"]
        decision.append({"idea": name.replace(" | ensemble", ""), "gain_held_out": gains["held-out simulated users"],
                         "gain_validation": gains["validation users"], "adopted": bool((gains >= MIN_GAIN).all())})
    decision = pd.DataFrame(decision)
    blend_info = pd.DataFrame(blend_info)
    curves = pd.DataFrame({"w": BLEND_GRID, "held-out (all, in-sample)": [curve[w] for w in BLEND_GRID],
                           "validation (descriptive)": [val_curve[w] for w in BLEND_GRID]})
    summary.to_csv(OUT / f"summary_n{N_SEEDS}.csv", index=False)
    tests.to_csv(OUT / f"tests_n{N_SEEDS}.csv", index=False)
    decision.to_csv(OUT / f"decision_n{N_SEEDS}.csv", index=False)
    blend_info.to_csv(OUT / f"blend_weights_n{N_SEEDS}.csv", index=False)
    curves.to_csv(OUT / f"blend_curve_n{N_SEEDS}.csv", index=False)

    pd.set_option("display.width", 250)
    print(f"\n=== {N_SEEDS} seeds: NDCG@10 ===")
    print(summary.pivot_table(index="method", columns=["users", "model"], values="ndcg", sort=False).round(4).to_string())
    print("\nBlend weight w for user-kNN (F):")
    print(blend_info.to_string(index=False))
    print(curves.round(4).to_string(index=False))
    print("\nTests (paired Wilcoxon; Bonferroni over 5 within each user group; Hybrid E-w vs Hybrid E is one test):")
    print(tests.round(4).to_string(index=False))
    print(f"\nDecision rule: adopt only if the ensemble gains >= +{MIN_GAIN} over LTR v2 on BOTH user groups")
    print(decision.round(4).to_string(index=False))


def describe_features(tables):
    """Means of the new columns, validation vs simulated (cross-fitting must not shift them much)."""
    rows = []
    for split in ("val", "simulated"):
        t = tables[(split, "c30")]
        for c in G7_USER + G8_RATING + G_KNN[:1]:
            rows.append({"feature": c, "users": split, "mean": t[c].mean(), "std": t[c].std()})
    frame = pd.DataFrame(rows).pivot(index="feature", columns="users", values=["mean", "std"])
    frame.to_csv(OUT / f"feature_shift_n{N_SEEDS}.csv")
    print("\nNew columns, validation vs simulated users:")
    print(frame.round(4).to_string())
    users_c = tables[("val", "union")].groupby("user_id").size()
    print(f"C lists: {users_c.mean():.1f} films per validation user (min {users_c.min()}, max {users_c.max()})")


if __name__ == "__main__":
    if N_SEEDS not in INPUT_TABLES:
        sys.exit(f"Seeds must be 10 or 3, not {N_SEEDS}.\n{__doc__}")
    OUT.mkdir(parents=True, exist_ok=True)
    data = ltr.load_input()
    test_ids = {u["user_id"] for u in data["test_users"]}
    with open(ltr.FOLDER / "step4_chosen_params.json", encoding="utf-8") as f:
        chosen = json.load(f)

    tables, users, views = build_tables(data)
    assert not (test_ids & {u["user_id"] for g in users.values() for u in g}), "a test user is in this experiment"
    held = holdout_ids(pd.read_csv(ltr.FOLDER / INPUT_TABLES[N_SEEDS][1], usecols=["user_id"]))
    held &= set(tables[("simulated", "c30")]["user_id"])
    describe_features(tables)
    half_of = tables[("simulated", "c30")].groupby("user_id")["half"].first().to_dict()
    baselines = full_ranking_baselines(data, users, views, held, half_of)
    run(tables, users, baselines, held, chosen)
    print(f"\nSaved to {OUT}")
