'''
Final pre-registered test on the 814 test users (doc 36).
Pre-registration: output/experiments/final_test/preregistration.md (incl. Amendment 1).

Two boards, every setting fixed in advance, warm users as training data (as every validation run):
  cold board   first 10 and first 3 liked films (final_comparison.py's methods)
  S = 50 board first 50 liked films, users with >= 50 liked + 5 more (warm_benchmark.py's methods)

Modes:
    python experiments/evaluation/final_test.py check     recompute both boards on the VALIDATION users with this code and compare
                                   with the saved validation results (code check; no test users)
    python experiments/evaluation/final_test.py llm       new LLM answers for the test users, 10 seeds: DeepSeek p5, Kimi K3 p2 and p5
                                   (DeepSeek p2 is reused from doc 13 and must be complete in the cache)
    python experiments/evaluation/final_test.py run       both boards on the test users, once; then the checks and tests
    python experiments/evaluation/final_test.py report    checks and tests from the saved test files
'''

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))  # shared modules live in src/
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
import LightGBM_XBoost_rerank_experiment as ltr
import cold_start_baselines as cbl
import final_comparison as fc
import warm_benchmark as wb
from recomm import EASE_R_Recommende

OUT = Path("output") / "experiments" / "final_test"
K = bmt.K
S_LONG = 50
MIN_S50_USERS = 120
LTR = "LTR v2 (XGBoost + LightGBM)"
KNN, SVD, NCF = cbl.NAMES["knn"], cbl.NAMES["svd"], cbl.NAMES["ncf"]
KIMI_P2, DEEPSEEK_P2 = "LLM: Kimi K3, p2", "LLM: DeepSeek V4.1 Flash, p2"
FUSION = fc.FUSION[0]
EASE_COLD = {n: f"EASE (lambda {lam:,})" for n, lam in fc.EASE_LAMBDAS.items()}
NEW_LLM = [("deepseek-v4.1-flash", "p5"), ("kimi-k3", "p2"), ("kimi-k3", "p5")]

# Amendment 1: results already known on the test users (docs 13, 16) - reproduced as checks
EXPECTED_TEST = {(10, "Hybrid E"): 0.6311, (3, "Hybrid E"): 0.5970, (10, LTR): 0.6609, (3, LTR): 0.6257,
                 (10, DEEPSEEK_P2): 0.6169, (10, "Popularity"): 0.5598, (3, "Popularity"): 0.5598}

# (board, setting, method, reference, kind); kind "greater" = confirmed if p_Bonf < 0.05 and the difference > 0
FAMILY_A = [("cold", 10, EASE_COLD[10], "Hybrid E", "greater", "A3"),
            ("cold", 3, EASE_COLD[3], "Hybrid E", "greater", "A4"),
            ("cold", 10, FUSION, "Hybrid E", "greater", "A5"),
            ("s50", S_LONG, LTR, "Hybrid E", "greater", "A6"),
            ("cold", 10, KIMI_P2, "Hybrid E", "equivalent", "A7")]
FAMILY_B = [("cold", 10, LTR, EASE_COLD[10], "two-sided", "B1"),
            ("cold", 3, LTR, EASE_COLD[3], "two-sided", "B2"),
            ("cold", 10, KNN, "Hybrid E", "two-sided", "B3"),
            ("cold", 3, KNN, "Hybrid E", "two-sided", "B4"),
            ("cold", 10, NCF, "Hybrid E", "two-sided", "B5"),
            ("cold", 3, NCF, "Hybrid E", "two-sided", "B6"),
            ("s50", S_LONG, "EASE", "Hybrid E", "two-sided", "B8")]
EQUIVALENCE_MARGIN = 0.02
BOOTSTRAP = 10_000


# ───────────────────────── cold board (final_comparison.run with a split) ─────────────────────────
def ltr_top10(n, split):
    """LTR v2 final model: trained on the simulated warm users, scored on the split's saved feature table."""
    suffix = "g5" if n == 10 else f"n{n}"
    train = pd.read_csv(ltr.FOLDER / f"features_simulated_{suffix}.csv")
    table = pd.read_csv(ltr.FOLDER / f"features_{split}_{suffix}.csv")
    with open(ltr.FOLDER / "step4_chosen_params.json", encoding="utf-8") as f:
        chosen = json.load(f)
    columns = [c for group in ltr.MODELS[ltr.FINAL_MODEL["features"]] for c in ltr.GROUPS[group]]
    table = table.assign(score=ltr.ensemble_scores(train, table, columns, chosen))
    return {int(u): g.sort_values("score", ascending=False, kind="stable")["movie_id"].astype(int).tolist()[:K]
            for u, g in table.groupby("user_id")}


def cold_board(split):
    data = fc.load()
    option10 = cbl.ColdStartBaselines(data)
    rows, missing, cache = [], {}, {}
    for n in fc.SEED_COUNTS:
        users = cs.users_with_n_seeds(data[f"{split}_users"], n)
        print(f"\n{len(users)} {split} users, {n} seeds")
        ltr_lists = ltr_top10(n, split)
        assert set(ltr_lists) == {u["user_id"] for u in users}, "LTR table users differ from the split's users"
        ease = EASE_R_Recommende(data["train"], data["movies"], fc.EASE_LAMBDAS[n])
        ease.fit(data["train"])
        for user in users:
            hybrid_top30, _ = lr.hybrid_e_candidates(user, data)
            lists = {
                "Popularity": cs.recommend_popular(data["popular"].index, user, K),
                "Content (embedding + LLM labels, max)": fc.from_table(data["content_table"], user, data, "max"),
                "Item-CF": fc.from_table(data["item_cf"], user, data, "mean"),
                "Hybrid E": hybrid_top30[:K],
                LTR: ltr_lists[int(user["user_id"])],
                EASE_COLD[n]: [int(m) for m in ease.recommend(user["liked"], K, exclude=user["seen"])],
                **option10.lists(user),
            }
            if n == 10:
                for label, provider, model, version in fc.LLM_RUNS:
                    order = fc.cached_llm_order(user, data, provider, model, version, cache)
                    if order is None:
                        missing[label] = missing.get(label, 0) + 1
                    else:
                        lists[label] = order[:K]
                label, provider, model, version, w = fc.FUSION
                order = fc.cached_llm_order(user, data, provider, model, version, cache)
                if order is not None:
                    llm_rank = {m: i + 1 for i, m in enumerate(order)}
                    combined = [w * (i + 1) + (1 - w) * llm_rank[m] for i, m in enumerate(hybrid_top30)]
                    best = np.argsort(combined, kind="stable")[:K]
                    lists[label] = [hybrid_top30[i] for i in best]
            for method, recommended in lists.items():
                rows.append({"board": "cold", "setting": n, "method": method, "user_id": user["user_id"],
                             **bmt.user_metrics(recommended, user["relevant"], K)})
    if missing:
        print(f"missing LLM answers: {missing}")
    return pd.DataFrame(rows), missing


# ───────────────────────── S = 50 board (warm_benchmark with fixed settings) ─────────────────────────
def long_history_users(data, split):
    """warm_benchmark.benchmark_users for any split: >= 50 liked pool films + 5 more; plus the cold ratings."""
    _, cold = cs.split_users(bmt.prepare_ratings(data["movies"]))
    ids = [u["user_id"] for u in data[f"{split}_users"]]
    built = cs.build_cold_user_data(cold[cold["user_id"].isin(ids)], data["pool_ids"], S_LONG, cs.MIN_TEST_RELEVANT)
    by_id = {u["user_id"]: u for u in built}
    users = [by_id[i] for i in ids if i in by_id]
    ratings_of = {(u, m): r for u, m, r in zip(cold["user_id"], cold["movie_id"], cold["rating"])}
    print(f"S = {S_LONG} board: {len(users)} of {len(ids)} {split} users qualify")
    if len(users) < MIN_S50_USERS:
        raise SystemExit(f"Fewer than {MIN_S50_USERS} users: the pre-registration says use S = 25; stopping to apply it.")
    return users, ratings_of


def ltr_s50_table(split, users):
    """LTR v2 feature table at S = 50: validation = warm_benchmark's cached table; test = built the same way."""
    if split == "val":
        return pd.read_csv(wb.OUT / "ltr" / f"features_val_S{S_LONG}.csv")
    path = OUT / f"features_test_S{S_LONG}.csv"
    if path.exists():
        return pd.read_csv(path)
    from tqdm import tqdm
    data = ltr.load_input()
    data["alpha_E"] = wb.load_settings()["alpha"]
    train, _ = cs.split_users(bmt.prepare_ratings(data["movies"]))
    print("latent tables: full warm users")
    full = dict(data, **ltr.latent_tables(train, data))
    group = cs.users_with_n_seeds(users, S_LONG)
    rows = [r for u in tqdm(group, desc=f"test, S = {S_LONG}")
            for r in ltr.candidate_rows(u, full, False, ltr.wide_candidates(u, full, "C30"))]
    table = ltr.add_g5(pd.DataFrame(rows), group, full)
    r7 = [c for g in ltr.MODELS["R7"] for c in ltr.GROUPS[g]]
    assert table[r7].notna().all().all() and (table.groupby("user_id").size() == lr.TOP_M).all()
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    return pd.read_csv(path)        # always scored from the saved file, so the result can be reproduced from it


def s50_board(split):
    settings = wb.load_settings()
    ease_lambda = json.loads((wb.OUT / "settings_ease.json").read_text())["lambda"][str(S_LONG)]
    data = fc.load()
    users, ratings_of = long_history_users(data, split)
    table = ltr_s50_table(split, users)              # built before the other models (own data load)
    data["alpha_E"] = settings["alpha"]
    option10 = cbl.ColdStartBaselines(data)
    option10.seed_rating = ratings_of
    cbl.FOLD[S_LONG] = settings["ncf"][S_LONG]
    column, pool_ids = option10.column, option10.pool_ids
    k = settings["k"][S_LONG]
    ease = EASE_R_Recommende(data["train"], data["movies"], ease_lambda)
    ease.fit(data["train"])

    # LTR v2: trained on warm_benchmark's simulated users at S = 50 (doc 23), scored on this split's table
    with open(ltr.FOLDER / "step4_chosen_params.json", encoding="utf-8") as f:
        chosen = json.load(f)
    r7 = [c for g in ltr.MODELS["R7"] for c in ltr.GROUPS[g]]
    simulated = pd.read_csv(wb.OUT / "ltr" / f"features_simulated_S{S_LONG}.csv")
    assert not (set(simulated["user_id"]) & {u["user_id"] for u in users}), "a board user is in the LTR training data"
    table = table.assign(score=ltr.ensemble_scores(simulated, table, r7, chosen))
    ltr_lists = {int(u): g.sort_values("score", ascending=False, kind="stable")["movie_id"].astype(int).tolist()[:K]
                 for u, g in table.groupby("user_id")}

    def top(scores, allowed):
        return pool_ids[np.argsort(-np.where(allowed, scores, -np.inf), kind="stable")[:K]].tolist()

    rows = []
    group = cs.users_with_n_seeds(users, S_LONG)
    print(f"S = {S_LONG}: alpha {settings['alpha'][S_LONG]}, k {k}, NCF {settings['ncf'][S_LONG]}, "
          f"EASE lambda {ease_lambda}, {len(group)} users")
    for u in group:
        cols = [column[m] for m in u["liked"]]
        allowed = np.ones(len(pool_ids), dtype=bool)
        allowed[[column[m] for m in u["seen"]]] = False
        lists = {
            "Popularity": cs.recommend_popular(data["popular"].index, u, K),
            "Content (embedding + LLM labels, max)": fc.from_table(data["content_table"], u, data, "max"),
            "Item-CF": fc.from_table(data["item_cf"], u, data, "mean"),
            "Hybrid E": lr.hybrid_e_candidates(u, data)[0][:K],
            KNN: top(option10.users.knn_scores(cols, k), allowed),
            SVD: top(option10.svd_scores(u, cols), allowed),
            NCF: top(option10.ncf_scores(u, cols, allowed), allowed),
            "EASE": [int(m) for m in ease.recommend(u["liked"], K, exclude=u["seen"])],
            LTR: ltr_lists[int(u["user_id"])],
        }
        for method, recommended in lists.items():
            rows.append({"board": "s50", "setting": S_LONG, "method": method, "user_id": u["user_id"],
                         **bmt.user_metrics(recommended, u["relevant"], K)})
    per_user = pd.DataFrame(rows)
    # the LTR table's own Hybrid E order must be the board's Hybrid E (as warm_benchmark checks)
    relevant_of = {u["user_id"]: u["relevant"] for u in group}
    table_hybrid = ltr.ndcg_per_user(table, "hybrid_rank", relevant_of, ascending=True).mean()
    board_hybrid = per_user[per_user["method"] == "Hybrid E"]["ndcg"].mean()
    assert abs(table_hybrid - board_hybrid) < 1e-9, f"Hybrid E differs: table {table_hybrid}, board {board_hybrid}"
    return per_user


# ───────────────────────── modes ─────────────────────────
def check():
    """Both boards on the validation users with this code vs the saved validation results."""
    cold, _ = cold_board("val")
    saved = pd.read_csv(fc.FOLDER / "per_user_val.csv")
    a = cold.groupby(["setting", "method"])["ndcg"].mean()
    b = saved.groupby(["seeds", "method"])["ndcg"].mean()
    b.index = b.index.set_names(["setting", "method"])
    long = s50_board("val")
    saved50 = pd.read_csv(wb.OUT / "per_user_all.csv")
    saved50 = saved50[(saved50["S"] == S_LONG) & saved50["method"].isin(long["method"].unique())]
    c = long.groupby(["setting", "method"])["ndcg"].mean()
    d = saved50.groupby(["S", "method"])["ndcg"].mean()
    d.index = d.index.set_names(["setting", "method"])
    both = pd.concat([pd.concat([a, c]).rename("now"), pd.concat([b, d]).rename("saved")], axis=1).dropna()
    both["difference"] = both["now"] - both["saved"]
    pd.set_option("display.width", 250)
    print(both.round(5).to_string())
    bad = both[both["difference"].abs() > 5e-4]
    print("\nCODE CHECK: " + ("all validation results reproduced" if bad.empty else f"{len(bad)} DIFFERENT"))
    OUT.mkdir(parents=True, exist_ok=True)
    both.to_csv(OUT / "code_check_validation.csv")


def run_llm():
    data = lr.load_data()
    users = cs.users_with_n_seeds(data["test_users"], 10)
    path = lr.run_name("ollama_cloud", "deepseek-v4.1-flash", 10, None, "p2", "hybrid")[0]
    cached = {json.loads(l)["user_id"] for l in path.read_text(encoding="utf-8").splitlines()}
    missing = [u["user_id"] for u in users if u["user_id"] not in cached]
    assert not missing, f"DeepSeek p2 test answers missing for {len(missing)} users (doc 13 cache)"
    print("DeepSeek p2: all 814 test answers cached (doc 13); reused, no calls")
    usage = []
    for model, version in NEW_LLM:
        part = lr.run_model("ollama_cloud", model, users, data, version=version, candidates="hybrid")
        usage.append({"model": model, "version": version, "users": len(part),
                      "unparsable": int(part["unparsable"].sum()), "with_issues": int((part["issues"] > 0).sum()),
                      "tokens_in_mean": part["tokens_in"].mean(), "tokens_in_total": int(part["tokens_in"].sum()),
                      "tokens_out_total": int(part["tokens_out"].sum()), "seconds_mean": part["seconds"].mean()})
    usage = pd.DataFrame(usage)
    OUT.mkdir(parents=True, exist_ok=True)
    usage.to_csv(OUT / "llm_usage_test.csv", index=False)
    print(usage.round(1).to_string(index=False))


def run():
    done = OUT / "per_user_test.csv"
    if done.exists():
        raise SystemExit(f"The final test has already been run ({done}). It is run only once; use 'report'.")
    cold, missing = cold_board("test")
    if missing:
        raise SystemExit(f"LLM answers missing on the test users: {missing}. Run 'python experiments/evaluation/final_test.py llm' first.")
    long = s50_board("test")
    OUT.mkdir(parents=True, exist_ok=True)
    pd.concat([cold, long], ignore_index=True).to_csv(done, index=False)
    report()


def boot_ci(diff, level, seed=42):
    rng = np.random.default_rng(seed)
    values = diff.to_numpy()
    means = values[rng.integers(0, len(values), size=(BOOTSTRAP, len(values)))].mean(axis=1)
    tail = (1 - level) / 2 * 100
    return np.percentile(means, tail), np.percentile(means, 100 - tail)


def family(per_user, plan, name):
    m = len(plan)
    out = []
    for board, setting, method, ref, kind, tag in plan:
        part = per_user[(per_user["board"] == board) & (per_user["setting"] == setting)]
        wide = part.pivot(index="user_id", columns="method", values="ndcg")
        diff = (wide[method] - wide[ref]).dropna()
        p = wilcoxon(diff).pvalue if diff.any() else 1.0
        lo, hi = boot_ci(diff, 0.95)
        lo_adj, hi_adj = boot_ci(diff, 1 - 0.05 / m)
        p_bonf = min(1.0, p * m)
        if kind == "greater":
            verdict = "CONFIRMED" if (p_bonf < 0.05 and diff.mean() > 0) else "not confirmed"
        elif kind == "equivalent":
            verdict = ("EQUIVALENT" if (lo_adj >= -EQUIVALENCE_MARGIN and hi_adj <= EQUIVALENCE_MARGIN)
                       else "not shown equivalent")
        else:
            verdict = ("significant" if p_bonf < 0.05 else "not significant")
        out.append({"family": name, "id": tag, "board": board, "setting": setting, "method": method, "vs": ref,
                    "users": len(diff), "method_ndcg": wide[method].mean(), "ref_ndcg": wide[ref].mean(),
                    "difference": diff.mean(), "ci95_low": lo, "ci95_high": hi,
                    f"ci_bonf_low": lo_adj, f"ci_bonf_high": hi_adj,
                    "better": int((diff > 0).sum()), "worse": int((diff < 0).sum()),
                    "p": p, "p_bonf": p_bonf, "verdict": verdict})
    return pd.DataFrame(out)


def report():
    per_user = pd.read_csv(OUT / "per_user_test.csv")
    pd.set_option("display.width", 260)
    pd.set_option("display.max_colwidth", 55)

    # Checks: the results already known on the test users (Amendment 1)
    means = per_user.groupby(["board", "setting", "method"])["ndcg"].mean()
    checks = []
    for (n, method), expected in EXPECTED_TEST.items():
        got = means.get(("cold", n, method), np.nan)
        checks.append({"setting": n, "method": method, "now": got, "documented": expected,
                       "ok": abs(got - expected) < 5e-4})
    checks = pd.DataFrame(checks)
    print("Checks against the documented test results (docs 13, 16):")
    print(checks.round(4).to_string(index=False))
    checks.to_csv(OUT / "checks_known_results.csv", index=False)
    if not checks["ok"].all():
        raise SystemExit("A documented test result was NOT reproduced: stop and report (Amendment 1, point 1).")

    # Boards
    boards = []
    for (board, setting), part in per_user.groupby(["board", "setting"], sort=False):
        b = part.groupby("method")[["ndcg", "precision", "recall"]].mean()
        b["users"] = part.groupby("method")["user_id"].nunique()
        b = b.sort_values("ndcg", ascending=False)
        wide = part.pivot(index="user_id", columns="method", values="ndcg")
        b["vs_hybrid_E"] = [(wide[m] - wide["Hybrid E"]).mean() for m in b.index]
        b.insert(0, "setting", setting)
        b.insert(0, "board", board)
        boards.append(b)
        print(f"\nBoard {board}, setting {setting} (test users)")
        print(b.round(4).to_string())
    pd.concat(boards).to_csv(OUT / "summary_test.csv")

    tests = pd.concat([family(per_user, FAMILY_A, "A (primary, Bonferroni x5)"),
                       family(per_user, FAMILY_B, "B (secondary, Bonferroni x7)")], ignore_index=True)
    tests.to_csv(OUT / "tests_test.csv", index=False)
    print("\nPre-registered tests")
    print(tests[["id", "board", "setting", "method", "vs", "users", "difference", "ci95_low", "ci95_high",
                 "ci_bonf_low", "ci_bonf_high", "better", "worse", "p_bonf", "verdict"]].round(4).to_string(index=False))
    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    mode = ARGS[0] if ARGS else ""
    {"check": check, "llm": run_llm, "run": run, "report": report}.get(mode, lambda: print(__doc__))()
