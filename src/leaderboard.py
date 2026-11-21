'''
Leaderboard of every result in the thesis (CLI option 12).

Results are only ranked against results that are comparable, so there is one board per
  user group  20-user pilot (the first 20 validation users) | 203 validation users | 814 test users
  seeds       3 or 10 seed films per new user
  idea        recommenders and LLM reranking of Hybrid E's top 30 (idea 1)
              LLM reranking of the 30 most popular films, no rating data (idea 3)
              LLM recommending freely (idea 2)
              LLM reranking user-kNN's top 30 (idea 4; only once such runs exist)
Every board has a reference line (Hybrid E; popularity for ideas 2 and 3; user-kNN for idea 4) and a paired Wilcoxon
test of each method against it (Bonferroni within the board).

Where the numbers come from (nothing is called, nothing is trained):
  - LLM runs: recomputed from the cached answers (output/experiments/llm_rerank/answers_*.jsonl),
    not from the compare_summary CSVs, which are overwritten when compare_models runs a new model set
  - popularity, content, item-CF, Hybrid E: recomputed with the experiments' own functions
  - LTR v2: final_comparison (validation) and the pre-registered test (doc 16)
  - fusion (doc 17 method A), candidate-order means (doc 17 7.5), free generation (doc 13 idea 2): saved per-user files

Usage:
    python src/leaderboard.py                 build (if needed) and choose a board
    python src/leaderboard.py rebuild         rebuild from the saved results, then choose
'''
import json
import re
import sys
from pathlib import Path

REBUILD = len(sys.argv) > 1 and sys.argv[1] == "rebuild"
sys.argv = [sys.argv[0]]
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

OUT = Path("output") / "leaderboard"
BOARD_FILE = OUT / "leaderboard.csv"
EXP = Path("output") / "experiments"
LTR = Path("output") / "experiments_ LightGBM_XBOOST" / "ltr_boosting"
GROUPS = {"pilot": "20-user pilot", "val": "203 validation users", "test": "814 test users"}
IDEAS = {1: "Recommenders + LLM reranking Hybrid E's top 30",
         3: "LLM reranking the 30 most popular films (no rating data)",
         2: "LLM recommending freely",
         4: "LLM reranking user-kNN's top 30"}
REFERENCES = {1: "Hybrid E", 2: "Popularity", 3: "Popularity",
              4: "User-kNN (standard: similarity-weighted sum)"}   # = cold_start_baselines.NAMES["knn"]
CANDIDATE_IDEAS = (("_cuknn", 4), ("_cpop", 3))   # cache-file suffix -> idea (none = idea 1)

PROMPTS = {
    "p1": "titles, years and genres",
    "p2": "+ director, cast and a profile of the seed films",
    "p3": "p2 + LLM themes and moods",
    "p4": "p3 + runtime, keywords and a one-line plot",
    "p5": "p2 + collaborative data per film (co-visitation and popularity shares)",
    "p6": "p5 + Hybrid E's rank of each film",
    "p7": "p5 + 3 random warm users as worked examples",
    "p8": "p5 + 3 similar warm users as worked examples",
}
NICE = {"deepseek-v4.1-flash": "DeepSeek V4.1 Flash", "deepseek-v4.1-flash:1017": "DeepSeek V4.1 Flash (17 Oct snapshot)",
        "minimax-m3": "MiniMax M3","deepseek-v4-pro:0813": "DeepSeek V4 Pro",
        "kimi-k3": "Kimi K3", "glm-5.3": "GLM-5.3", "glm-5.3-flash": "GLM-5.3-Flash", "glm-5.2": "GLM-5.2",
        "gemma4:e4b": "gemma4:e4b (local)", "gemma4:31b-cloud": "gemma4:31b", "mistral-large-4": "Mistral Large 4",
        "nemotron-3-super": "Nemotron 3 Super", "gpt-oss:120b-cloud": "gpt-oss:120b",
        "gpt-oss:20b-cloud": "gpt-oss:20b", "llama3.1:8b": "llama3.1:8b (local)", "qwen3.5:9b": "qwen3.5:9b (local)"}
# Results confirmed in a pre-registered test (docs 13 and 16)
CONFIRMED = {("test", 10, 1, "LTR v2 (XGBoost + LightGBM)"), ("test", 3, 1, "LTR v2 (XGBoost + LightGBM)"),
             ("test", 10, 3, "DeepSeek V4.1 Flash, p2")}


# ───────────────────────── building ─────────────────────────
def cache_settings(path, record):
    """Settings of one answer cache, read from its file name (see llm_rerank_experiment.run_name)."""
    safe = record["model"].replace(":", "_").replace("/", "_")
    suffix = path.stem[len(f"answers_{record['provider']}_{safe}"):]
    get = lambda pattern, default: (re.search(pattern, suffix) or [None, default])[1]
    return {"provider": record["provider"], "model": record["model"],
            "think": get(r"_think-([^_]+)", None), "version": get(r"_(p\d)", "p1"),
            "n_seeds": int(get(r"_n(\d+)", 3)),
            "idea": next((idea for tag, idea in CANDIDATE_IDEAS if tag in suffix), 1),
            "shuffle": int(get(r"_s(\d+)", 0))}


def llm_label(s):
    label = f"{NICE.get(s['model'], s['model'])}, {s['version']}"
    if s["think"] and s["think"] not in ("False", "None"):
        label += " + thinking" if s["think"] == "True" else f" + thinking ({s['think']})"
    if s["shuffle"]:
        label += f" (candidate order {s['shuffle']})"
    return label


def llm_explanation(s):
    task = {1: "reranks Hybrid E's top 30", 3: "reranks the 30 most popular films (no rating data)",
            4: "reranks user-kNN's top 30"}[s["idea"]]
    text = f"{NICE.get(s['model'], s['model'])} {task}; prompt {s['version']}: {PROMPTS.get(s['version'], '')}"
    if s["think"] and s["think"] not in ("False", "None"):
        text += "; reasoning on"
    if s["shuffle"]:
        text += f"; candidates shown in random order {s['shuffle']} (order 0 = the main run)"
    return text


def rows_for(method, family, explanation, source, per_user, group, n, idea):
    """per_user: DataFrame indexed by user_id with ndcg (and precision, recall if known)."""
    return [{"group": group, "seeds": n, "idea": idea, "method": method, "family": family,
             "user_id": uid, "ndcg": r["ndcg"], "precision": r.get("precision", np.nan),
             "recall": r.get("recall", np.nan), "explanation": explanation, "source": source}
            for uid, r in per_user.iterrows()]


def build():
    import build_movie_texts as bmt
    import cold_start_experiment as cs
    import llm_rerank_experiment as lr
    import final_comparison as fc

    data = fc.load()
    K = bmt.K
    users = {(g, n): cs.users_with_n_seeds(data["val_users"] if g != "test" else data["test_users"], n)
             for g in ("val", "test") for n in (3, 10)}       # the 5-seed pilot runs have no board
    for n in (3, 10):
        users[("pilot", n)] = users[("val", n)][:20]
    rows = []

    def metrics(lst, user):
        return bmt.user_metrics(lst, user["relevant"], K)

    # 1. Recommenders without an LLM (recomputed), on every user group, idea 1 boards (+ references for ideas 3, 4)
    print("Scoring the recommenders without an LLM...")
    has_uknn = any((EXP / "llm_rerank").glob("answers_*_cuknn*.jsonl"))    # idea-4 boards only once runs exist
    for (g, n), group_users in users.items():
        per ={name: {} for name in ("Popularity", "Content (embedding + LLM labels, max)", "Item-CF", "Hybrid E")}
        for u in group_users:
            per["Popularity"][u["user_id"]] = metrics(cs.recommend_popular(data["popular"].index, u, K), u)
            per["Content (embedding + LLM labels, max)"][u["user_id"]] = metrics(
                fc.from_table(data["content_table"], u, data, "max"), u)
            per["Item-CF"][u["user_id"]] = metrics(fc.from_table(data["item_cf"], u, data, "mean"), u)
            per["Hybrid E"][u["user_id"]] = metrics(lr.hybrid_e_candidates(u, data)[0][:K], u)
        explain = {"Popularity": ("baseline", "the most-liked films among warm users, the same list for everyone"),
                   "Content (embedding + LLM labels, max)": ("content", "embeddinggemma similarity to the closest seed film, film texts with LLM labels (doc 05)"),
                   "Item-CF": ("collaborative", "mean item-based CF similarity to the seed films (doc 06)"),
                   "Hybrid E": ("hybrid", "item-CF + popularity, alpha chosen on validation users (doc 06)")}
        for name, values in per.items():
            frame = pd.DataFrame(values).T
            rows += rows_for(name, explain[name][0], explain[name][1], "recomputed", frame, g, n, 1)
            if name in ("Popularity", "Hybrid E"):      # references on the idea-3 (and idea-4) boards
                rows += rows_for(name, explain[name][0], explain[name][1], "recomputed", frame, g, n, 3)
                if has_uknn:
                    rows += rows_for(name, explain[name][0], explain[name][1], "recomputed", frame, g, n, 4)
        if has_uknn:    # idea-4 reference: user-kNN's own top 10 (the same list as option 11's user-kNN row)
            frame = pd.DataFrame({u["user_id"]: metrics(lr.uknn_candidates(u, data)[0][:K], u) for u in group_users}).T
            rows += rows_for(REFERENCES[4], "collaborative", "textbook user-based kNN (k = 200, doc 20); "
                             "its top 30 are the candidates the LLM reranks", "recomputed", frame, g, n, 4)

    # 2. LLM reranking runs, recomputed from every answer cache (no model calls)
    print("Scoring every cached LLM run...")
    for path in sorted((EXP / "llm_rerank").glob("answers_*.jsonl")):
        records = {r["user_id"]: r for r in map(json.loads, path.read_text(encoding="utf-8").splitlines())}
        if not records:
            continue
        s = cache_settings(path, next(iter(records.values())))
        for g in ("pilot", "val", "test"):
            group_users = users.get((g, s["n_seeds"]), [])
            if not group_users or not all(u["user_id"] in records for u in group_users):
                continue
            values = {}
            for u in group_users:
                top_ids = lr.get_candidates(u, data, {1: "hybrid", 3: "popularity", 4: "uknn"}[s["idea"]])[0]
                _, number_to_id = lr.build_prompt(u, top_ids, data, s["version"], s["shuffle"])
                order, _ = lr.parse_ranking(records[u["user_id"]]["content"], number_to_id, top_ids)
                values[u["user_id"]] = metrics(order[:K], u)
            rows += rows_for(llm_label(s), "LLM", llm_explanation(s), path.name, pd.DataFrame(values).T,
                             g, s["n_seeds"], s["idea"])

    # 3. LTR v2 (validation: final_comparison; test: the pre-registered test of doc 16)
    print("Adding LTR, fusion, candidate-order means and free generation...")
    fc_file = EXP / "final_comparison" / "per_user_val.csv"
    if fc_file.exists():
        import cold_start_baselines as cbl
        texts = {"LTR v2 (XGBoost + LightGBM)": ("learned ranker", "tuned XGBoost + LightGBM ensemble reranking "
                                                 "Hybrid E's top 30, trained on simulated warm users (doc 16)"),
                 cbl.NAMES["content"]: ("content", "recomm.py's content model (0.75 CountVectorizer + 0.25 TF-IDF), "
                                        "similarity to the closest seed"),
                 cbl.NAMES["collaborative"]: ("collaborative", "recomm.py's user-based CF (5 most similar warm users); "
                                              "the new user's vector = +1 on each seed"),
                 cbl.NAMES["knn"]: ("collaborative", "textbook user-based kNN: similarity-weighted sum of the "
                                    "centred ratings of the 200 most similar warm users (k chosen on validation)"),
                 cbl.NAMES["svd"]: ("matrix factorisation", "Surprise SVD trained on the warm users; the new user's "
                                    "factors fitted to the seeds' actual ratings (more input than the other methods)"),
                 cbl.NAMES["ncf"]: ("neural CF", "NeuMF trained on the warm users; only the new user's embeddings "
                                    "are learned from the seeds (fold-in)"),
                 cbl.NAMES["hybrid"]: ("hybrid", "recomm.py's hybrid: user-CF top 20 rescored 0.4 x SVD + 0.6 x CF rank")}
        saved = pd.read_csv(fc_file)
        for (method, n), part in saved[saved["method"].isin(texts)].groupby(["method", "seeds"]):
            frame = part.set_index("user_id")[["ndcg", "precision", "recall"]]
            family, text = texts[method]
            rows += rows_for(method, family, text, fc_file.name, frame, "val", n, 1)
            pilot_ids = [u["user_id"] for u in users[("pilot", n)]]
            rows += rows_for(method, family, text, fc_file.name, frame.loc[pilot_ids], "pilot", n, 1)
    for n in (10, 3):
        f = LTR / f"step6_per_user_test_n{n}.csv"
        if f.exists():
            t = pd.read_csv(f, index_col=0)
            rows += rows_for("LTR v2 (XGBoost + LightGBM)", "learned ranker",
                             "the pre-registered final model of doc 16, run once on the test users",
                             f.name, t[["final model"]].rename(columns={"final model": "ndcg"}), "test", n, 1)
            if "XGBoost R2, 203 users (LTR v1)" in t:
                rows += rows_for("LTR v1 (XGBoost R2, trained on 203 users)", "learned ranker",
                                 "XGBoost on collaborative + content features, trained on the validation users (doc 15)",
                                 f.name, t[["XGBoost R2, 203 users (LTR v1)"]].rename(
                                     columns={"XGBoost R2, 203 users (LTR v1)": "ndcg"}), "test", n, 1)

    # 4. Fusion with Hybrid E (doc 17 method A), validation users, 10 seeds
    fusion_files = [(EXP / "llm_fusion" / "per_user_val.csv", {" rank blend": "rank blend (w by CV)", " RRF": "reciprocal rank fusion"}),
                    (EXP / "kimi_k3" / "fusion_per_user_val.csv", {"kimi-k3 rank blend, w = 0.4 (fixed)": "rank blend (w = 0.4, borrowed from DeepSeek)",
                                                                    "kimi-k3 rank blend, w by 5-fold CV": "rank blend (w by CV)",
                                                                    "kimi-k3 RRF": "reciprocal rank fusion"})]
    seen = set()
    for f, wanted in fusion_files:
        if not f.exists():
            continue
        t = pd.read_csv(f, index_col=0)
        for col in t.columns:
            for key, kind in wanted.items():
                if (col.endswith(key) if key.startswith(" ") else col == key):
                    model = col[: -len(key)] if key.startswith(" ") else "kimi-k3"
                    control = model.startswith("control:")
                    name = model.split(":", 1)[1].strip() if control else NICE.get(model, model)
                    method = (f"Fusion control: Hybrid E + {name} order, {kind}" if control
                              else f"Fusion: Hybrid E + {name} p2, {kind}")
                    if method in seen:
                        continue
                    seen.add(method)
                    frame = t[[col]].rename(columns={col: "ndcg"})
                    text = ((f"control for method A: Hybrid E's order blended with a {name} order instead of an LLM"
                             if control else f"Hybrid E's order blended with {name}'s reranking (doc 17, method A)")
                            + ("; the weight was chosen on these users (optimistic)" if "CV" in kind or "0.4" in kind
                               else "; no tuning"))
                    rows += rows_for(method, "control" if control else "LLM fusion", text, f.name, frame, "val", 10, 1)

    # 5. Mean over three candidate orders (doc 17 7.5), validation users, 10 seeds
    f = EXP / "kimi_k3" / "shuffles_per_user_val.csv"
    if f.exists():
        t = pd.read_csv(f)
        for model, part in t.groupby(t["model"].str.split(" ").str[0]):
            frame = part.groupby("user_id")[["llm_ndcg", "llm_precision", "llm_recall"]].mean()
            frame.columns = ["ndcg", "precision", "recall"]
            rows += rows_for(f"{NICE.get(model, model)}, p2 (mean of 3 candidate orders)", "LLM",
                             f"{NICE.get(model, model)} reranks Hybrid E's top 30 with prompt p2; NDCG averaged over "
                             "3 random candidate orders (the fairest single number for this model)", f.name, frame, "val", 10, 1)

    # 6. Free generation (idea 2), validation users
    for f in sorted((EXP / "llm_generate").glob("per_user_val_203users_g*_n*.csv")):
        t = pd.read_csv(f)
        for (model, version, n), part in t.groupby(["model", "version", "n_seeds"]):
            frame = part.set_index("user_id")
            base = {"g1": "3 seeds", "g2": "10 seeds"}.get(version, version)
            rows += rows_for(f"{NICE.get(model, model)}, {version} (filled)", "LLM",
                             f"{NICE.get(model, model)} names 10 films freely from the seeds ({base}); invalid names replaced by popular films",
                             f.name, frame[["ndcg_filled", "precision_filled", "recall_filled"]].set_axis(
                                 ["ndcg", "precision", "recall"], axis=1), "val", int(n), 2)
            rows += rows_for(f"{NICE.get(model, model)}, {version} (strict)", "LLM",
                             f"as filled, but invalid names count as misses", f.name,
                             frame[["ndcg_strict"]].set_axis(["ndcg"], axis=1), "val", int(n), 2)
            if model == t["model"].iloc[0]:
                for ref, col in (("Popularity", "ndcg_popularity"), ("Hybrid E", "ndcg_hybrid_E")):
                    rows += rows_for(ref, "baseline" if ref == "Popularity" else "hybrid", "reference",
                                     f.name, frame[[col]].set_axis(["ndcg"], axis=1), "val", int(n), 2)

    per_user = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    per_user.to_csv(OUT / "leaderboard_per_user.csv", index=False)

    # Summaries and tests against each board's reference
    board_rows = []
    for (g, n, idea), part in per_user.groupby(["group", "seeds", "idea"]):
        reference = REFERENCES[idea]
        ref = part[part["method"] == reference].set_index("user_id")["ndcg"]
        methods = part.groupby("method", sort=False)
        tests = {}
        for method, m in methods:
            if method == reference:
                continue
            values = m.set_index("user_id")["ndcg"]
            diff = (values - ref.reindex(values.index)).dropna()
            tests[method] = (diff.mean(), wilcoxon(diff).pvalue if diff.any() else 1.0,
                             int((diff > 0).sum()), int((diff < 0).sum()))
        for method, m in methods:
            first = m.iloc[0]
            d, p, better, worse = tests.get(method, (np.nan, np.nan, np.nan, np.nan))
            status = {"pilot": "pilot", "val": "validation", "test": "test"}[g]
            if (g, n, idea, method) in CONFIRMED:
                status = "confirmed (pre-registered)"
            board_rows.append({"board": f"{GROUPS[g]} | {n} seeds | {IDEAS[idea]}", "group": g, "seeds": n,
                               "idea": idea, "method": method, "family": first["family"],
                               "users": len(m), "ndcg": m["ndcg"].mean(), "precision": m["precision"].mean(),
                               "recall": m["recall"].mean(), "reference": reference, "vs_reference": d,
                               "better": better, "worse": worse, "p_value": p,
                               "p_bonferroni": min(1.0, p * len(tests)) if p == p else np.nan,
                               "status": status, "explanation": first["explanation"], "source": first["source"]})
    board = pd.DataFrame(board_rows)
    board.to_csv(BOARD_FILE, index=False)
    print(f"Leaderboard built: {len(board)} entries on {board['board'].nunique()} boards -> {BOARD_FILE}")
    return board


# ───────────────────────── printing ─────────────────────────
def show(board, name, k=10):
    part = board[board["board"] == name].sort_values("ndcg", ascending=False).reset_index(drop=True)
    reference = part["reference"].iloc[0]
    ref_row = part[part["method"] == reference]
    everyone = part[part["method"] != reference]
    others = everyone.head(k).reset_index(drop=True)
    print(f"\nLeaderboard - {name}")
    print(f"Top {len(others)} of {len(everyone)} methods by NDCG@10; reference: {reference} "
          f"{ref_row['ndcg'].iloc[0]:.4f}")
    print("p: paired Wilcoxon vs the reference; Bonf.: Bonferroni over every method on this board "
          "(stricter than the per-experiment correction in the documents)\n")
    print(f"{'#':>3}  {'Method':76} {'NDCG@10':>8}  {'vs ' + reference:>16}  {'p':>7}  {'p (Bonf.)':>9}  Status")
    shown_ref = False
    for i, r in others.iterrows():
        if not shown_ref and r["ndcg"] < ref_row["ndcg"].iloc[0]:
            print(f"{'-':>3}  {reference + ' (reference)':76} {ref_row['ndcg'].iloc[0]:8.4f}")
            shown_ref = True
        fmt = lambda v: "" if v != v else ("<0.001" if v < 0.001 else f"{v:.3f}")
        print(f"{i + 1:>3}  {r['method'][:76]:76} {r['ndcg']:8.4f}  {r['vs_reference']:+16.4f}  "
              f"{fmt(r['p_value']):>7}  {fmt(r['p_bonferroni']):>9}  {r['status']}")
    if not shown_ref:
        print(f"{'-':>3}  {reference + ' (reference)':76} {ref_row['ndcg'].iloc[0]:8.4f}")
    return others


def details(row):
    print(f"\n{row['method']}  [{row['family']}]")
    print(f"  {row['explanation']}")
    print(f"  NDCG@10 {row['ndcg']:.4f} | precision {row['precision']:.4f} | recall {row['recall']:.4f} | {row['users']} users")
    print(f"  vs {row['reference']}: {row['vs_reference']:+.4f} (better {row['better']}, worse {row['worse']}, "
          f"p = {row['p_value']:.4g}, Bonferroni {row['p_bonferroni']:.4g}) | status: {row['status']}")
    print(f"  source: {row['source']}")


def load_board(rebuild=False):
    if rebuild or not BOARD_FILE.exists():
        return build()
    return pd.read_csv(BOARD_FILE)


def menu(rebuild=False):
    board = load_board(rebuild)
    order = (board[["board", "group", "seeds", "idea"]].drop_duplicates()
             .assign(g=lambda d: d["group"].map({"test": 0, "val": 1, "pilot": 2}))
             .sort_values(["g", "idea", "seeds"], ascending=[True, True, False])["board"].tolist())
    while True:
        print("\nBoards:")
        for i, name in enumerate(order, 1):
            print(f"  {i:>2}. {name}")
        print("   r. rebuild from the saved results    q. back")
        choice = input("Board: ").strip().lower()
        if choice in ("q", ""):
            return
        if choice == "r":
            board = load_board(True)
            continue
        if not choice.isdigit() or not 1 <= int(choice) <= len(order):
            print("Invalid choice.")
            continue
        k = input("How many (10 / 20 / 30) [10]: ").strip() or "10"
        shown = show(board, order[int(choice) - 1], int(k) if k.isdigit() else 10)
        while True:
            pick = input("\nRow number for details (Enter = back to boards): ").strip()
            if not pick:
                break
            if pick.isdigit() and 1 <= int(pick) <= len(shown):
                details(shown.iloc[int(pick) - 1])


if __name__ == "__main__":
    menu(REBUILD)
