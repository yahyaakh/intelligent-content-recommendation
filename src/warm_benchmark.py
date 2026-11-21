'''
Warm-user benchmark (doc 18, section 7b; results in doc 23): do the cold-start conclusions hold for
users with a long history?

Design: the same held-out users, given more history. Each user gets the first S liked films in time
order, for S = 3, 10, 25, 50:
  users        the validation users with at least 50 liked pool films + 5 more to evaluate
               (the same users at every S)
  answer key   their liked films MINUS the first 50: the same key at every S, so differences between
               S come from the history alone
  never shown  all of the first 50 liked films, at every S
  training     the 9,591 warm users, unchanged; no model sees these users' ratings
  metric       NDCG@10 (+ precision, recall); paired Wilcoxon vs Hybrid E, Bonferroni within each S

Because the first 50 films are excluded at EVERY S, Hybrid E's top 30 differs from the cold-start
experiments even at S = 3 and 10. So nothing is reused from earlier runs except trained models:
the LLM answers and the LTR tables are new (own folder, never mixed with the cold-start caches).
The S = 3 and 10 numbers here form their own board and never replace the doc 06-20 numbers.

Settings chosen on these validation users, per S, before any LLM or LTR run (saved in settings.json):
  Hybrid E    alpha from the doc 06 grid
  User-kNN    k from {10, 20, 50, 100, 200, 400, 1000}
  NCF         fold-in steps / learning rate: S = 3, 10 as doc 20; S = 25, 50 from a small grid
  LTR v2      no tuning: R7 features, the step-4 parameters of doc 16, retrained per S on simulated warm
              users (cross-fitted halves A/B as doc 16) whose first 50 liked films are excluded the same way
  LLM         DeepSeek V4.1 Flash and Kimi K3, prompt p2, rerank Hybrid E's top 30, no thinking

Modes:
    python src/warm_benchmark.py check              counts, prompt sizes; no models, nothing saved
    python src/warm_benchmark.py base               settings + popularity, content, item-CF, Hybrid E,
                                                user-kNN, SVD, NCF (minutes; NCF fold-in is the slow part)
    python src/warm_benchmark.py ltr [S ...]        LTR v2 per S (feature tables about 1-2 h per S, cached)
    python src/warm_benchmark.py llm [S ...]        LLM reranking, 20-user pilot (new model calls)
    python src/warm_benchmark.py llm full [S ...]   the same on all users
    python src/warm_benchmark.py ease               EASE per S (added 2026-10-29): lambda per S chosen on these
                                                users from ease_tuning's grid, as alpha and k above
    python src/warm_benchmark.py report             tables, tests and the curve from everything saved
Every mode except check ends with the report. Validation users only: the test users are used once,
in the final pre-registered test (doc 18 section 9).
'''
import json
import sys
from pathlib import Path

ARGS = sys.argv[1:]
sys.argv = [sys.argv[0]]                  # the imported experiment scripts read sys.argv when imported
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import build_movie_texts as bmt
import cold_start_experiment as cs
import llm_rerank_experiment as lr

OUT = Path("output") / "experiments" / "warm_benchmark"
SETTINGS = OUT / "settings.json"
HISTORY = (3, 10, 25, 50)
MAX_HISTORY = max(HISTORY)
K = bmt.K
MIN_VAL_USERS = 120                       # doc 18 7b: below this, use S <= 25 instead
KNN_GRID = [10, 20, 50, 100, 200, 400, 1000]
NCF_GRID = [(300, 0.05), (1000, 0.05), (2000, 0.05)]   # for S = 25 and 50 (fixed in advance)
LLM_MODELS = [("ollama_cloud", "deepseek-v4.1-flash"), ("ollama_cloud", "kimi-k3")]
LLM_VERSION = "p2"
PILOT_USERS = 20
LLM_NAMES = {"deepseek-v4.1-flash": "LLM: DeepSeek V4.1 Flash, p2", "kimi-k3": "LLM: Kimi K3, p2"}
LTR_NAME = "LTR v2 (XGBoost + LightGBM)"

MODE = ARGS[0] if ARGS else "check"
FULL = "full" in ARGS
S_ARGS = [int(a) for a in ARGS if a.isdigit()]
S_LIST = tuple(S_ARGS) if S_ARGS else HISTORY
if any(s not in HISTORY for s in S_LIST):
    raise SystemExit(f"S must be one of {HISTORY}")


# ───────────────────────── users ─────────────────────────
def benchmark_users(data, split="val"):
    """The split's users with >= 50 liked pool films and >= 5 more (seed_pool = first 50 liked,
    relevant = the rest), in the split's original order; plus every rating of the cold users
    (SVD's fold-in needs the seeds' own ratings)."""
    if split != "val":
        raise SystemExit("Validation users only. The test users are used once, in the final pre-registered test.")
    _, cold = cs.split_users(bmt.prepare_ratings(data["movies"]))
    ids = [u["user_id"] for u in data[f"{split}_users"]]
    built = cs.build_cold_user_data(cold[cold["user_id"].isin(ids)], data["pool_ids"], MAX_HISTORY,
                                    cs.MIN_TEST_RELEVANT)
    by_id = {u["user_id"]: u for u in built}
    users = [by_id[i] for i in ids if i in by_id]
    ratings_of = {(u, m): r for u, m, r in zip(cold["user_id"], cold["movie_id"], cold["rating"])}
    print(f"Warm benchmark: {len(users)} of {len(ids)} validation users have >= {MAX_HISTORY} liked films "
          f"+ {cs.MIN_TEST_RELEVANT} to evaluate")
    if len(users) < MIN_VAL_USERS:
        raise SystemExit(f"Fewer than {MIN_VAL_USERS} users: use S <= 25 instead (doc 18 7b) and change HISTORY.")
    return users, ratings_of


def load_settings():
    if not SETTINGS.exists():
        raise SystemExit("settings.json missing: run 'python src/warm_benchmark.py base' first "
                         "(the settings are fixed before any LLM or LTR run).")
    s = json.loads(SETTINGS.read_text(encoding="utf-8"))
    return {"alpha": {int(k): v for k, v in s["alpha"].items()}, "k": {int(k): v for k, v in s["k"].items()},
            "ncf": {int(k): tuple(v) for k, v in s["ncf"].items()}}


def ndcg_list(recommended, user):
    return bmt.user_metrics(recommended, user["relevant"], K)


# ───────────────────────── base: settings + methods without LLM / LTR ─────────────────────────
def run_base():
    import final_comparison as fc
    import cold_start_baselines as cbl
    OUT.mkdir(parents=True, exist_ok=True)
    data = fc.load()
    users, ratings_of = benchmark_users(data)
    position_of, movie_ids = data["position_of"], data["movie_ids"]
    option10 = cbl.ColdStartBaselines(data)
    option10.seed_rating = ratings_of               # SVD fold-in: the seeds' own ratings, for any S
    column, pool_ids = option10.column, option10.pool_ids

    def allowed_of(user):
        allowed = np.ones(len(pool_ids), dtype=bool)
        allowed[[column[m] for m in user["seen"]]] = False
        return allowed

    def top(scores, allowed):
        return pool_ids[np.argsort(-np.where(allowed, scores, -np.inf), kind="stable")[:K]].tolist()

    # 1. Settings per S, on these users (the curves are saved for the doc)
    curves, alpha, k_of, ncf_of = [], {}, {}, {}
    for S in HISTORY:
        group = cs.users_with_n_seeds(users, S)
        best = -1
        for a in cs.ALPHAS:
            v = cs.evaluate_hybrid("E", data["item_cf"], data["pop_vector"], group, data["pool_positions"],
                                   position_of, movie_ids, K, a, "mean")["ndcg"].mean()
            curves.append({"S": S, "setting": "Hybrid E alpha", "value": a, "ndcg": v})
            if v > best:                                   # strict: the smaller alpha wins a tie (doc 06)
                best, alpha[S] = v, a
        best = -1
        for k in KNN_GRID:
            v = np.mean([ndcg_list(top(option10.users.knn_scores([column[m] for m in u["liked"]], k),
                                       allowed_of(u)), u)["ndcg"] for u in group])
            curves.append({"S": S, "setting": "user-kNN k", "value": k, "ndcg": v})
            if v > best:
                best, k_of[S] = v, k
        grid = [cbl.FOLD[S]] if S in cbl.FOLD else NCF_GRID          # S = 3, 10: doc 20's choice
        best = -1
        for steps, rate in grid:
            cbl.FOLD[S] = (steps, rate)
            values = []
            for u in group:
                allowed = allowed_of(u)
                values.append(ndcg_list(top(option10.ncf_scores(u, [column[m] for m in u["liked"]], allowed),
                                            allowed), u)["ndcg"])
            v = float(np.mean(values))
            curves.append({"S": S, "setting": "NCF fold-in (steps, lr)", "value": f"{steps} / {rate}", "ndcg": v})
            print(f"  S = {S}: NCF fold-in {steps} / {rate} -> {v:.4f}")
            if v > best:
                best, ncf_of[S] = v, (steps, rate)
        cbl.FOLD[S] = ncf_of[S]
        print(f"S = {S}: alpha {alpha[S]}, user-kNN k {k_of[S]}, NCF {ncf_of[S]}")
    pd.DataFrame(curves).to_csv(OUT / "settings_curves.csv", index=False)
    SETTINGS.write_text(json.dumps({"alpha": alpha, "k": k_of, "ncf": ncf_of,
                                    "users": len(users), "chosen_on": "validation users of the benchmark"},
                                   indent=2), encoding="utf-8")

    # 2. Every method at every S
    data["alpha_E"] = alpha                                   # Hybrid E (and its top 30) with this board's alpha
    rows = []
    for S in HISTORY:
        group = cs.users_with_n_seeds(users, S)
        for u in group:
            cols, allowed = [column[m] for m in u["liked"]], allowed_of(u)
            hybrid = lr.hybrid_e_candidates(u, data)[0][:K]
            lists = {
                "Popularity": cs.recommend_popular(data["popular"].index, u, K),
                "Content (embedding + LLM labels, max)": fc.from_table(data["content_table"], u, data, "max"),
                "Item-CF": fc.from_table(data["item_cf"], u, data, "mean"),
                "Hybrid E": hybrid,
                cbl.NAMES["knn"]: top(option10.users.knn_scores(cols, k_of[S]), allowed),
                cbl.NAMES["svd"]: top(option10.svd_scores(u, cols), allowed),
                cbl.NAMES["ncf"]: top(option10.ncf_scores(u, cols, allowed), allowed),
            }
            for method, recommended in lists.items():
                rows.append({"S": S, "method": method, "user_id": u["user_id"], **ndcg_list(recommended, u)})
        # check: lr's Hybrid E equals cold_start_experiment's with the chosen alpha
        check = cs.evaluate_hybrid("E", data["item_cf"], data["pop_vector"], group, data["pool_positions"],
                                   position_of, movie_ids, K, alpha[S], "mean")["ndcg"].mean()
        got = np.mean([r["ndcg"] for r in rows if r["S"] == S and r["method"] == "Hybrid E"])
        assert abs(check - got) < 1e-9, f"Hybrid E differs between the two implementations at S = {S}"
    pd.DataFrame(rows).to_csv(OUT / "per_user_base.csv", index=False)
    print(f"Saved -> {OUT / 'per_user_base.csv'}")


# ───────────────────────── EASE per S (added 2026-10-29) ─────────────────────────
EASE_NAME = "EASE"


def run_ease():
    """EASE fitted on the warm users; lambda per S chosen on these users (like alpha and k in 'base'),
    from ease_tuning's grid. Settings go to settings_ease.json so settings.json stays as LTR / LLM used it."""
    from ease_tuning import LAMBDAS
    from recomm import EASE_R_Recommende
    data = lr.load_data()
    users, _ = benchmark_users(data)
    groups = {S: cs.users_with_n_seeds(users, S) for S in HISTORY}
    scores, lists = {}, {}
    for lam in LAMBDAS:
        model = EASE_R_Recommende(data["train"], data["movies"], lam)
        model.fit(data["train"])                                     # warm users only
        for S, group in groups.items():
            recs = [[int(m) for m in model.recommend(u["liked"], K, exclude=u["seen"])] for u in group]
            scores[(S, lam)] = float(np.mean([ndcg_list(r, u)["ndcg"] for r, u in zip(recs, group)]))
            lists[(S, lam)] = recs
        print(f"  lambda {lam:>7,}: " + "  ".join(f"S={S} {scores[(S, lam)]:.4f}" for S in HISTORY), flush=True)
    chosen = {}
    for S in HISTORY:
        best = max(LAMBDAS, key=lambda lam: (scores[(S, lam)], -lam))    # tie -> the smaller lambda
        chosen[S] = best
        edge = " (edge of the grid)" if best in (min(LAMBDAS), max(LAMBDAS)) else ""
        print(f"S = {S}: lambda {best:,} -> {scores[(S, best)]:.4f}{edge}")
    pd.DataFrame([{"S": S, "lambda": lam, "ndcg": v} for (S, lam), v in scores.items()]).to_csv(
        OUT / "settings_curves_ease.csv", index=False)
    (OUT / "settings_ease.json").write_text(json.dumps({"lambda": chosen, "users": len(users),
                                                        "chosen_on": "validation users of the benchmark"},
                                                       indent=2), encoding="utf-8")
    rows = [{"S": S, "method": EASE_NAME, "user_id": u["user_id"], **ndcg_list(r, u)}
            for S in HISTORY for r, u in zip(lists[(S, chosen[S])], groups[S])]
    pd.DataFrame(rows).to_csv(OUT / "per_user_ease.csv", index=False)
    print(f"Saved -> {OUT / 'per_user_ease.csv'}")


# ───────────────────────── LTR v2 per S ─────────────────────────
def run_ltr():
    import LightGBM_XBoost_rerank_experiment as ltr
    from tqdm import tqdm
    settings = load_settings()
    data = ltr.load_input()                                   # + embeddings, content table, v5 labels
    data["alpha_E"] = settings["alpha"]                       # set BEFORE the half views are copied
    users, _ = benchmark_users(data)
    with open(ltr.FOLDER / "step4_chosen_params.json", encoding="utf-8") as f:
        chosen = json.load(f)
    r7 = [c for group in ltr.MODELS["R7"] for c in ltr.GROUPS[group]]

    # Simulated warm users, the halves of doc 16 (seed 42), with the same 50-film rule
    train, _ = cs.split_users(bmt.prepare_ratings(data["movies"]))
    warm_ids = np.sort(train["user_id"].unique())
    half_of = dict(zip(warm_ids, np.random.default_rng(42).permutation(len(warm_ids)) % 2))
    in_A = train["user_id"].map(half_of) == 0
    pool_ids = {int(data["movie_ids"][p]) for p in data["pool"]}
    sim_users = {h: cs.build_cold_user_data(part, pool_ids, MAX_HISTORY, cs.MIN_TEST_RELEVANT)
                 for h, part in (("A", train[in_A]), ("B", train[~in_A]))}
    sim_relevant = {u["user_id"]: u["relevant"] for p in sim_users.values() for u in p}
    assert not (set(sim_relevant) & {u["user_id"] for u in users}), "a benchmark user is in the training data"

    views = None
    base = pd.read_csv(OUT / "per_user_base.csv") if (OUT / "per_user_base.csv").exists() else None
    rows = []
    for S in S_LIST:
        paths = {split: OUT / "ltr" / f"features_{split}_S{S}.csv" for split in ("val", "simulated")}
        if all(p.exists() for p in paths.values()):
            tables = {split: pd.read_csv(p) for split, p in paths.items()}
            print(f"S = {S}: loaded the feature tables")
        else:
            if views is None:                                 # the latent tables, once per run
                print("latent tables: full warm users, and without each half (a few minutes)")
                views = {"full": dict(data, **ltr.latent_tables(train, data))}
                for half, other in (("A", train[~in_A]), ("B", train[in_A])):
                    views[half] = ltr.half_view(data, other)
                    views[half].update(ltr.latent_tables(other, data))
            group = cs.users_with_n_seeds(users, S)
            val_rows = [r for u in tqdm(group, desc=f"val, S = {S}")
                        for r in ltr.candidate_rows(u, views["full"], False, ltr.wide_candidates(u, views["full"], "C30"))]
            tables = {"val": ltr.add_g5(pd.DataFrame(val_rows), group, views["full"])}
            parts = []
            for half in ("A", "B"):
                sim = cs.users_with_n_seeds(sim_users[half], S)
                sim_rows = []
                for u in tqdm(sim, desc=f"simulated, S = {S}, half {half}"):
                    for r in ltr.candidate_rows(u, views[half], False, ltr.wide_candidates(u, views[half], "C30")):
                        r["half"] = half
                        sim_rows.append(r)
                parts.append(ltr.add_g5(pd.DataFrame(sim_rows), sim, views[half]))
            tables["simulated"] = pd.concat(parts, ignore_index=True)
            paths["val"].parent.mkdir(parents=True, exist_ok=True)
            for split, t in tables.items():
                assert t[r7].notna().all().all(), f"missing values ({split}, S = {S})"
                assert (t.groupby("user_id").size() == lr.TOP_M).all(), f"not 30 rows per user ({split})"
                t.to_csv(paths[split], index=False)

        group = cs.users_with_n_seeds(users, S)
        relevant_of = {u["user_id"]: u["relevant"] for u in group}
        val = tables["val"]
        hybrid = ltr.ndcg_per_user(val, "hybrid_rank", relevant_of, ascending=True)
        if base is not None:                                  # the table's Hybrid E = the base run's
            expected = base[(base["S"] == S) & (base["method"] == "Hybrid E")]["ndcg"].mean()
            assert abs(hybrid.mean() - expected) < 1e-9, f"Hybrid E in the LTR table differs at S = {S}"
        sim_hybrid = ltr.ndcg_by_score(tables["simulated"], "hybrid_rank", sim_relevant, True)
        print(f"S = {S}: training on {tables['simulated']['user_id'].nunique():,} simulated users "
              f"(Hybrid E on them: {sim_hybrid:.4f})")
        scores = ltr.ensemble_scores(tables["simulated"], val, r7, chosen)
        val = val.assign(score=scores)
        for user_id, g in val.groupby("user_id"):
            top10 = g.sort_values("score", ascending=False, kind="stable")["movie_id"].astype(int).tolist()[:K]
            rows.append({"S": S, "method": LTR_NAME, "user_id": user_id, **bmt.user_metrics(top10, relevant_of[user_id], K)})
        print(f"S = {S}: LTR v2 {np.mean([r['ndcg'] for r in rows if r['S'] == S]):.4f} vs Hybrid E {hybrid.mean():.4f}")

    path = OUT / "per_user_ltr.csv"
    new = pd.DataFrame(rows)
    if path.exists():                                         # keep the other S values from earlier runs
        old = pd.read_csv(path)
        new = pd.concat([old[~old["S"].isin(S_LIST)], new], ignore_index=True)
    new.to_csv(path, index=False)
    print(f"Saved -> {path}")


# ───────────────────────── LLM reranking per S ─────────────────────────
def llm_folder():
    lr.FOLDER = OUT / "llm"                   # own caches: never mixed with the cold-start answers
    lr.FOLDER.mkdir(parents=True, exist_ok=True)


def run_llm():
    settings = load_settings()
    llm_folder()
    data = lr.load_data()
    data["alpha_E"] = settings["alpha"]
    users, _ = benchmark_users(data)
    group = users if FULL else users[:PILOT_USERS]
    for S in S_LIST:
        lr.compare_models(LLM_MODELS, dict(data, val_users=group), split="val", version=LLM_VERSION,
                          n_seeds=S, candidates="hybrid")


def llm_rows(data, users):
    """Per-user NDCG of every cached LLM run (no model calls), and the users each run covers."""
    rows = []
    for S in HISTORY:
        group = cs.users_with_n_seeds(users, S)
        for provider, model in LLM_MODELS:
            path = lr.run_name(provider, model, S, None, LLM_VERSION, "hybrid")[0]
            if not path.exists():
                continue
            records = {r["user_id"]: r for r in map(json.loads, path.read_text(encoding="utf-8").splitlines())}
            for u in group:
                if u["user_id"] in records:
                    top_ids = lr.hybrid_e_candidates(u, data)[0]
                    _, number_to_id = lr.build_prompt(u, top_ids, data, LLM_VERSION)
                    order, _ = lr.parse_ranking(records[u["user_id"]]["content"], number_to_id, top_ids)
                    rows.append({"S": S, "method": LLM_NAMES[model], "user_id": u["user_id"], **ndcg_list(order[:K], u)})
    return rows


# ───────────────────────── report ─────────────────────────
def paired(a, b):
    diff = (a - b).dropna()
    return diff.mean(), int((diff > 0).sum()), int((diff < 0).sum()), (wilcoxon(diff).pvalue if diff.any() else 1.0), len(diff)


def report():
    settings = load_settings()
    if not (OUT / "per_user_base.csv").exists():
        raise SystemExit("per_user_base.csv missing: run 'python src/warm_benchmark.py base' first.")
    parts = [pd.read_csv(OUT / "per_user_base.csv")]
    for extra in ("per_user_ltr.csv", "per_user_ease.csv"):
        if (OUT / extra).exists():
            parts.append(pd.read_csv(OUT / extra))
    llm_folder()
    data = lr.load_data()
    data["alpha_E"] = settings["alpha"]
    users, _ = benchmark_users(data)
    llm = pd.DataFrame(llm_rows(data, users))
    if len(llm):
        parts.append(llm)
    per_user = pd.concat(parts, ignore_index=True)
    n_users = len(users)

    complete = per_user.groupby(["S", "method"])["user_id"].transform("nunique") == n_users
    main, partial = per_user[complete], per_user[~complete]

    summary, tests = [], []
    for S, part in main.groupby("S"):
        table = part.pivot(index="user_id", columns="method", values="ndcg")
        for method in table.columns:
            row = {"S": S, "method": method, "users": int(table[method].notna().sum()),
                   **part[part["method"] == method][["ndcg", "precision", "recall"]].mean().to_dict()}
            if method != "Hybrid E":
                d, better, worse, p, _ = paired(table[method], table["Hybrid E"])
                row.update({"vs_hybrid_E": d, "better": better, "worse": worse, "p_value": p})
            if method != "Popularity":
                d, _, _, p, _ = paired(table[method], table["Popularity"])
                row.update({"vs_popularity": d, "p_vs_popularity": p})
            summary.append(row)
    summary = pd.DataFrame(summary)
    tested = summary["p_value"].notna()
    summary.loc[tested, "p_bonferroni"] = (summary.loc[tested, "p_value"]
                                           * summary[tested].groupby("S")["p_value"].transform("size")).clip(upper=1.0)
    has_pop = summary["p_vs_popularity"].notna()
    summary.loc[has_pop, "p_vs_popularity_bonferroni"] = (
        summary.loc[has_pop, "p_vs_popularity"] * summary[has_pop].groupby("S")["p_vs_popularity"].transform("size")).clip(upper=1.0)

    # Question 2: does the gap to Hybrid E change from S = 10 to S = 50? (difference of differences per user)
    gap = []
    wide = main.pivot_table(index="user_id", columns=["S", "method"], values="ndcg")
    for method in sorted(main["method"].unique()):
        if method == "Hybrid E" or not all((S, method) in wide.columns for S in (10, 50)):
            continue
        d10 = wide[(10, method)] - wide[(10, "Hybrid E")]
        d50 = wide[(50, method)] - wide[(50, "Hybrid E")]
        d, better, worse, p, _ = paired(d50, d10)
        gap.append({"method": method, "gap_S10": d10.mean(), "gap_S50": d50.mean(), "change": d,
                    "grew": better, "shrank": worse, "p_value": p})
    gap = pd.DataFrame(gap)
    if len(gap):
        gap["p_bonferroni"] = (gap["p_value"] * len(gap)).clip(upper=1.0)

    per_user.to_csv(OUT / "per_user_all.csv", index=False)
    summary.to_csv(OUT / "summary_val.csv", index=False)
    gap.to_csv(OUT / "gap_change_S10_S50.csv", index=False)

    pd.set_option("display.width", 250)
    print(f"\nWarm benchmark, {n_users} validation users (same users and answer key at every S; "
          f"settings: alpha {settings['alpha']}, k {settings['k']}, NCF {settings['ncf']})")
    for S in sorted(summary["S"].unique()):
        part = summary[summary["S"] == S].sort_values("ndcg", ascending=False)
        print(f"\nHistory S = {S} (vs Hybrid E: paired Wilcoxon, Bonferroni within S)")
        print(part[["method", "users", "ndcg", "precision", "recall", "vs_hybrid_E", "better", "worse",
                    "p_bonferroni", "vs_popularity", "p_vs_popularity_bonferroni"]].round(4).to_string(index=False))
    if len(gap):
        print("\nGap to Hybrid E, S = 50 minus S = 10 (per user; positive = the method gains more from history)")
        print(gap.round(4).to_string(index=False))
    if len(partial):
        print("\nRuns on part of the users only (pilot; not in the tables above), vs Hybrid E on the same users:")
        hyb = per_user[per_user["method"] == "Hybrid E"].set_index(["S", "user_id"])["ndcg"]
        for (S, method), part in partial.groupby(["S", "method"]):
            mine = part.set_index(["S", "user_id"])["ndcg"]
            d, better, worse, p, n = paired(mine, hyb.reindex(mine.index))
            print(f"  S = {S:>2} {method:<34} {mine.mean():.4f}  vs Hybrid E {d:+.4f} "
                  f"(better {better}, worse {worse}, p = {p:.3f}, {n} users)")
    plot_curve(summary)
    print(f"\nSaved to {OUT}")


def plot_curve(summary):
    """NDCG@10 against history length: recommenders (left) and the learned ranker vs the LLMs (right),
    one y-axis, direct labels at the line ends (the categorical palette is validated, light mode)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import cold_start_baselines as cbl
    colours = {"Popularity": "#2a78d6", "Item-CF": "#eb6834", "Content (embedding + LLM labels, max)": "#1baf7a",
               "Hybrid E": "#eda100", cbl.NAMES["knn"]: "#e87ba4", cbl.NAMES["svd"]: "#008300",
               cbl.NAMES["ncf"]: "#6250d6", EASE_NAME: "#8a5a2b", LTR_NAME: "#e34948",
               LLM_NAMES["deepseek-v4.1-flash"]: "#2a78d6", LLM_NAMES["kimi-k3"]: "#1baf7a"}
    short = {"Content (embedding + LLM labels, max)": "Content", cbl.NAMES["knn"]: "User-kNN",
             cbl.NAMES["svd"]: "SVD", cbl.NAMES["ncf"]: "NCF", LTR_NAME: "LTR v2",
             LLM_NAMES["deepseek-v4.1-flash"]: "DeepSeek p2", LLM_NAMES["kimi-k3"]: "Kimi K3 p2"}
    panels = [("Recommenders", [m for m in colours if m not in LLM_NAMES.values() and m != LTR_NAME]),
              ("Learned ranker and LLM reranking", ["Hybrid E", LTR_NAME, *LLM_NAMES.values()])]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, (title, methods) in zip(axes, panels):
        ends = []
        for method in methods:
            part = summary[summary["method"] == method].sort_values("S")
            if part.empty:
                continue
            dashed = method in LLM_NAMES.values()           # LLM lines dashed: their hues repeat the left panel's
            ax.plot(part["S"], part["ndcg"], color=colours[method], linewidth=2, marker="o", markersize=5,
                    linestyle="--" if dashed else "-")
            ends.append([part["ndcg"].iloc[-1], part["S"].iloc[-1], short.get(method, method)])
        # end labels, pushed apart so none overlap (at least GAP apart, in NDCG units)
        GAP = 0.009
        ends.sort()
        for i, end in enumerate(ends):
            end.append(end[0] if i == 0 else max(end[0], ends[i - 1][3] + GAP))
        for y, x, label, y_label in ends:
            ax.annotate(label, (x, y), xytext=(x * 1.07, y_label), textcoords="data", va="center",
                        fontsize=8, color="#333333")
        ax.set_title(title, fontsize=10, color="#333333", loc="left")
        ax.set_xscale("log")
        ax.set_xticks(HISTORY)
        ax.set_xticklabels([str(s) for s in HISTORY])
        ax.set_xlim(2.5, 90)
        ax.set_xlabel("history: liked films given to the method (S)", color="#555555")
        ax.grid(axis="y", color="#e6e6e6", linewidth=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("NDCG@10 (validation users)", color="#555555")
    fig.tight_layout()
    fig.savefig(OUT / "curve_ndcg_vs_history.png", dpi=160)
    plt.close(fig)


# ───────────────────────── check ─────────────────────────
def check():
    data = lr.load_data()
    if SETTINGS.exists():
        data["alpha_E"] = load_settings()["alpha"]
    users, _ = benchmark_users(data)
    print(f"answer key: {np.mean([len(u['relevant']) for u in users]):.1f} liked films per user on average "
          f"(min {min(len(u['relevant']) for u in users)})")
    for S in HISTORY:
        u = cs.users_with_n_seeds(users, S)[0]
        top_ids = lr.hybrid_e_candidates(u, data)[0] if S in data["alpha_E"] else None
        if top_ids:
            messages, _ = lr.build_prompt(u, top_ids, data, LLM_VERSION)
            size = sum(len(m["content"]) for m in messages)
            print(f"S = {S}: p2 prompt about {size:,} characters (~{size / 4:,.0f} tokens) for the first user")
        else:
            print(f"S = {S}: no alpha yet (chosen in 'base'); prompt size shown after that")


if __name__ == "__main__":
    if MODE == "check":
        check()
    elif MODE == "base":
        run_base()
        report()
    elif MODE == "ltr":
        run_ltr()
        report()
    elif MODE == "ease":
        run_ease()
        report()
    elif MODE == "llm":
        run_llm()
        report()
    elif MODE == "report":
        report()
    else:
        raise SystemExit(__doc__)
