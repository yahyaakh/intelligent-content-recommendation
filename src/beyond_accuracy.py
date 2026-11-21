'''
Beyond-accuracy evaluation of the cold-start methods (experiments 06, 07, 09 and 11).

The earlier experiments saved only NDCG, precision and recall per user. This script rebuilds
the top-10 lists of the main methods on the 814 test users, with the parameters chosen there
(no LLM calls; only the cached embeddings, labels, requests and extracted preferences), and
measures what accuracy does not show:

- NDCG@10 per popularity third: the same list, scored only against the relevant movies in the
  low / mid / high third of the candidate pool by TMDB vote count (as in experiment 07).
- Catalogue coverage: share of the candidate pool that appears in at least one top-10 list.
- Gini index of how often each pool movie is recommended (0 = all equally, 1 = one movie only).
- Intra-list diversity: mean pairwise genre distance (1 - Jaccard) inside a list. Genres are used
  instead of embeddings so that no method is favoured by its own representation.
- Novelty: mean -log2(share of warm users who liked the movie).
- Long-tail share: share of recommended movies in the low / mid third.

Validity check: the NDCG@10 of every rebuilt method is compared with the saved per-user results.

Significance: per-user metrics with the Wilcoxon signed-rank test; coverage and Gini, which
describe the whole set of lists, with a paired permutation test over users. Bonferroni over all tests.

Usage:
    python src/beyond_accuracy.py           # rebuild the lists, summarise and test
    python src/beyond_accuracy.py tests     # tests only, on the saved lists
'''

import json
import re
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

import build_movie_texts as bmt
import cold_start_experiment as cs
import profile_experiment as pe
import typed_request_experiment as tr
from llm_enricher import PROMPT_VERSION
from pathlib import Path

FOLDER = Path("output") / "experiments" / "beyond_accuracy"
FOLDER.mkdir(parents=True, exist_ok=True)
SEED_COUNTS = [3, 10]
STRUCTURED_MODELS = ["gpt-oss_20b-cloud", "gemma4_31b-cloud", "gemma4_e4b"]


# ───────────────────────── list builders ─────────────────────────
def personal_scores(table, user, cand, position_of, scoring):
    block = table[np.ix_([position_of[m] for m in user["liked"]], cand)]
    return block.max(axis=0) if scoring == "max" else block.mean(axis=0)


def top_ids(cand, score, movie_ids):
    return movie_ids[cand[np.argsort(-score, kind="stable")[:bmt.K]]].tolist()


def candidates(user, pool, position_of):
    seen = {position_of[m] for m in user["seen"]}
    return np.array([p for p in pool if p not in seen])


# ───────────────────────── metrics ─────────────────────────
def gini(counts):
    x = np.sort(np.asarray(counts, dtype=float))
    n = len(x)
    if x.sum() == 0:
        return 0.0
    return float((2 * np.arange(1, n + 1) - n - 1) @ x / (n * x.sum()))


def list_metrics(rec, user, third_of, genres_of, novelty_of):
    row = bmt.user_metrics(rec, user["relevant"], bmt.K)
    for t in ("low", "mid", "high"):
        rel_t = {m for m in user["relevant"] if third_of.get(m) == t}
        row[f"ndcg_{t}"] = bmt.ndcg_at_k(rec, rel_t, bmt.K) if rel_t else np.nan
        row[f"share_{t}"] = np.mean([third_of.get(m) == t for m in rec])
    distances = []
    for i in range(len(rec)):
        for j in range(i + 1, len(rec)):
            a, b = genres_of[rec[i]], genres_of[rec[j]]
            union = a | b
            distances.append(1 - len(a & b) / len(union) if union else 0.0)
    row["diversity"] = float(np.mean(distances))
    row["novelty"] = float(np.mean([novelty_of[m] for m in rec]))
    return row


def summarise(lists, pool_ids):
    """lists: DataFrame with method, n, user_id, rec and per-user metrics."""
    rows = []
    for (method, n), part in lists.groupby(["method", "n"], sort=False):
        counts = pd.Series([m for rec in part["rec"] for m in rec]).value_counts()
        counts = counts.reindex(sorted(pool_ids), fill_value=0)
        rows.append({
            "method": method, "n": n,
            "ndcg": part["ndcg"].mean(),
            "ndcg_low": part["ndcg_low"].mean(), "ndcg_mid": part["ndcg_mid"].mean(),
            "ndcg_high": part["ndcg_high"].mean(),
            "coverage": (counts > 0).mean(),
            "gini": gini(counts.to_numpy()),
            "diversity": part["diversity"].mean(),
            "novelty": part["novelty"].mean(),
            "share_low": part["share_low"].mean(), "share_mid": part["share_mid"].mean(),
            "share_high": part["share_high"].mean(),
        })
    return pd.DataFrame(rows)


# ───────────────────────── significance tests ─────────────────────────
# (b, a, n): the test asks whether b differs from a; differences are b - a.
TEST_PAIRS = [
    # Does LLM enrichment change the lists?
    ("B_content+labels (max)", "A_content (max)", 3), ("B_content+labels (max)", "A_content (max)", 10),
    ("D_embedding+labels (mean)", "C_embedding (mean)", 3), ("D_embedding+labels (mean)", "C_embedding (mean)", 10),
    ("embedding+31b_P2 (mean)", "C_embedding (mean)", 3), ("embedding+31b_P2 (mean)", "C_embedding (mean)", 10),
    ("hybrid_D_embedding+labels", "hybrid_C_embedding", 3),
    # LLM interpretation of the request
    ("structured (gpt-oss_20b-cloud)", "request", 3), ("structured (gemma4_31b-cloud)", "request", 3),
    ("structured (gpt-oss_20b-cloud)", "structured (gemma4_e4b)", 3),
    # The accuracy / breadth trade-off
    ("hybrid_E_item_cf", "popularity", 3), ("E_item_cf (mean)", "hybrid_E_item_cf", 3),
    ("structured (gpt-oss_20b-cloud)", "hybrid_request", 3),
]
USER_METRICS = ["ndcg", "ndcg_low", "ndcg_mid", "longtail_share", "novelty", "diversity"]


def significance_tests(lists, pool_ids, n_perm=10000):
    """Per-user metrics: Wilcoxon signed-rank. Coverage and Gini are properties of the whole set
    of lists: paired permutation test over users. Bonferroni over all tests."""
    from scipy.stats import wilcoxon
    lists = lists.assign(longtail_share=lists["share_low"] + lists["share_mid"])
    pool = sorted(pool_ids)
    index_of = {m: i for i, m in enumerate(pool)}
    rng = np.random.default_rng(42)
    rows = []
    for b, a, n in TEST_PAIRS:
        pa = lists[(lists["method"] == a) & (lists["n"] == n)].set_index("user_id")
        pb = lists[(lists["method"] == b) & (lists["n"] == n)].set_index("user_id").loc[pa.index]
        for metric in USER_METRICS:
            both = pa[metric].notna() & pb[metric].notna()
            diff = (pb.loc[both, metric] - pa.loc[both, metric]).to_numpy()
            p = wilcoxon(diff).pvalue if np.any(diff) else 1.0
            rows.append({"b": b, "a": a, "n": n, "metric": metric, "users": int(both.sum()),
                         "a_value": pa.loc[both, metric].mean(), "b_value": pb.loc[both, metric].mean(),
                         "difference": diff.mean(), "users_better": int((diff > 0).sum()),
                         "users_worse": int((diff < 0).sum()), "p_value": p})
        # Recommendation counts per user as a users x pool matrix, for fast resampling
        counts = {}
        for name, part in (("a", pa), ("b", pb)):
            matrix = np.zeros((len(part), len(pool)), dtype=np.int16)
            for row, rec in enumerate(part["rec"]):
                matrix[row, [index_of[m] for m in rec]] += 1
            counts[name] = matrix
        # Paired permutation test: under the null hypothesis, a user's two lists are exchangeable,
        # so each user's lists are swapped at random. (A bootstrap is biased for coverage:
        # duplicated users lower the coverage, and more so for the broader method.)
        total_a, total_b = counts["a"].sum(axis=0), counts["b"].sum(axis=0)
        delta = (counts["a"] - counts["b"]).astype(np.float32)     # what swapping a user moves from b to a
        perm_a, perm_b = [], []
        for start in range(0, n_perm, 1000):                        # batches of swap patterns
            swaps = (rng.random((min(1000, n_perm - start), len(pa))) < 0.5).astype(np.float32)
            moved = swaps @ delta
            perm_b.append(total_b + moved)
            perm_a.append(total_a - moved)
        perm_a, perm_b = np.vstack(perm_a), np.vstack(perm_b)
        for metric, fn in (("coverage", lambda c: (np.asarray(c) > 0.5).mean()), ("gini", gini)):
            observed = fn(total_b) - fn(total_a)
            null = np.array([fn(pb) - fn(pa_) for pa_, pb in zip(perm_a, perm_b)])
            extreme = int((np.abs(null) >= abs(observed) - 1e-12).sum())
            rows.append({"b": b, "a": a, "n": n, "metric": metric, "users": len(pa),
                         "a_value": fn(total_a), "b_value": fn(total_b),
                         "difference": observed, "p_value": (extreme + 1) / (n_perm + 1)})
    tests = pd.DataFrame(rows)
    tests["p_bonferroni"] = (tests["p_value"] * len(tests)).clip(upper=1.0)
    return tests


def run_tests():
    lists = pd.read_csv(FOLDER / "per_user.csv")
    lists["rec"] = lists["rec"].map(lambda r: [int(m) for m in r.split("|")])
    pool_ids = set(pd.read_csv(FOLDER / "pool.csv")["movie_id"])
    tests = significance_tests(lists, pool_ids)
    tests.to_csv(FOLDER / "significance_tests.csv", index=False)
    pd.set_option("display.width", 250)
    print(tests.round(4).to_string(index=False))
    print(f"\nSaved {FOLDER / 'significance_tests.csv'}")


# ───────────────────────── main ─────────────────────────
if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "tests":
    run_tests()                 # tests only, on the saved lists
elif __name__ == "__main__":
    movies = bmt.load_movies()
    ratings = bmt.prepare_ratings(movies)
    train, cold = cs.split_users(ratings)
    pool_ids = set(train["movie_id"])
    cold_users = cs.build_cold_user_data(cold, pool_ids, max(cs.SEED_COUNTS), cs.MIN_TEST_RELEVANT)
    val_users, test_users = cs.split_validation(cold_users, cs.VAL_FRACTION)

    movie_ids = movies["movie_id"].to_numpy()
    position_of = {m: i for i, m in enumerate(movie_ids)}
    pool = np.array(sorted(position_of[m] for m in pool_ids))

    popular = cs.popularity_scores(train)
    pop_vector = cs.popularity_vector(popular, movies, position_of)
    n_warm = train["user_id"].nunique()
    novelty_of = {m: -np.log2(max(popular.get(m, 0), 1) / n_warm) for m in pool_ids}
    third_of, cuts = pe.popularity_thirds(movies, pool_ids)
    genres_of = {m: {g.strip() for g in re.split(r"[|,]", str(gs)) if g.strip() and str(gs) != "nan"}
                 for m, gs in zip(movies["movie_id"], movies["genres"])}

    print("Building similarity tables")
    tables = {
        "A_content (max)": (bmt.content_similarity(movies, with_enrichment=False), "max"),
        "B_content+labels (max)": (bmt.content_similarity(movies, with_enrichment=True), "max"),
        "C_embedding (mean)": (bmt.embedding_similarity(f"movies_base_{cs.EMBED_NAME}_{PROMPT_VERSION}", movies), "mean"),
        "D_embedding+labels (mean)": (bmt.embedding_similarity(f"movies_enriched_{cs.EMBED_NAME}_{PROMPT_VERSION}", movies), "mean"),
        "embedding+31b_P2 (mean)": (bmt.embedding_similarity("movies_profile_31b_P2_embeddinggemma", movies,
                                                             folder=Path("output") / "experiments" / "profiles"), "mean"),
        "E_item_cf (mean)": (cs.item_cf_similarity(train, movies, position_of), "mean"),
    }
    alphas = pd.read_csv("output/experiments/cold_start/alphas.csv").set_index(["table", "n"])["alpha"]
    hybrids = {"hybrid_C_embedding": ("C_embedding (mean)", "C_embedding_base"),
               "hybrid_D_embedding+labels": ("D_embedding+labels (mean)", "D_embedding_enriched"),
               "hybrid_E_item_cf": ("E_item_cf (mean)", "E_item_cf")}

    rows = []

    def add(method, n, user, rec):
        rows.append({"method": method, "n": n, "user_id": user["user_id"], "rec": rec,
                     **list_metrics(rec, user, third_of, genres_of, novelty_of)})

    # Seed-based methods (experiments 06 and 07)
    for n in SEED_COUNTS:
        group = cs.users_with_n_seeds(test_users, n)
        for user in tqdm(group, desc=f"seed methods N={n}"):
            cand = candidates(user, pool, position_of)
            add("popularity", n, user, cs.recommend_popular(popular.index, user, bmt.K))
            pop = cs.rescale(pop_vector[cand])
            personal = {}
            for name, (table, scoring) in tables.items():
                personal[name] = personal_scores(table, user, cand, position_of, scoring)
                add(name, n, user, top_ids(cand, personal[name], movie_ids))
            for name, (source, table_name) in hybrids.items():
                a = alphas[(table_name, n)]
                add(name, n, user, top_ids(cand, a * cs.rescale(personal[source]) + (1 - a) * pop, movie_ids))

    # Typed-request methods, r1 requests at N = 3 (experiments 09 and 11)
    params = pd.read_csv("output/experiments/typed_requests/tuned_parameters.csv")
    params = params[(params["style"] == "r1") & (params["n"] == 3) & (params["representation"] == "base")]
    params = params.drop_duplicates("method").set_index("method")
    requests_df = pd.read_json("output/experiments/typed_requests/requests_r1.jsonl", lines=True)
    request_of = requests_df[requests_df["n"] == 3].set_index("user_id")["request"].to_dict()
    group = cs.users_with_n_seeds(test_users, 3)
    queries = tr.embed_requests([request_of[u["user_id"]] for u in group])
    base_vec = tr.movie_vectors(movies)["base"]

    movie_labels = pd.read_csv("output/experiments/gemma4_e4b_v4/movies_enriched.csv").set_index("movie_id")
    movie_themes = [set(str(v).split("|")) - {"nan", ""} for v in movie_labels.reindex(movie_ids)["llm_themes"]]
    movie_moods = [set(str(v).split("|")) - {"nan", ""} for v in movie_labels.reindex(movie_ids)["llm_mood"]]
    extracted = {}
    for model in STRUCTURED_MODELS:
        path = Path("output") / "experiments" / "structured_preferences" / f"preferences_{model}.jsonl"
        extracted[model] = {r["user_id"]: r for r in map(json.loads, path.read_text(encoding="utf-8").splitlines())}

    def structured(prefs, cand):
        themes, moods = set(prefs["themes"]), set(prefs["moods"])
        parts = []
        if themes:
            parts.append(np.array([len(themes & movie_themes[p]) / len(themes) for p in cand]))
        if moods:
            parts.append(np.array([len(moods & movie_moods[p]) / len(moods) for p in cand]))
        return np.mean(parts, axis=0) if parts else None

    for user, q in tqdm(list(zip(group, queries)), desc="request methods N=3"):
        cand = candidates(user, pool, position_of)
        liked = [position_of[m] for m in user["liked"]]
        pop = cs.rescale(pop_vector[cand])
        req = cs.rescale(base_vec[cand] @ q)
        seed = cs.rescale((base_vec[cand] @ base_vec[liked].T).mean(axis=1))
        add("request", 3, user, top_ids(cand, req, movie_ids))
        a = params.loc["hybrid_request", "alpha"]
        add("hybrid_request", 3, user, top_ids(cand, a * req + (1 - a) * pop, movie_ids))
        a, w = params.loc["hybrid_request+seeds", "alpha"], params.loc["hybrid_request+seeds", "w"]
        add("hybrid_request+seeds", 3, user,
            top_ids(cand, a * cs.rescale(w * req + (1 - w) * seed) + (1 - a) * pop, movie_ids))
        for model in STRUCTURED_MODELS:
            s = structured(extracted[model][user["user_id"]], cand)
            primary = pop if s is None else s
            order = np.lexsort((cand, -pop, -primary))          # ties broken by popularity (experiment 11)
            add(f"structured ({model})", 3, user, movie_ids[cand[order[:bmt.K]]].tolist())

    lists = pd.DataFrame(rows)

    # ── Validity check: NDCG@10 against the saved results ──
    saved = []
    cold = pd.read_csv("output/experiments/cold_start/per_user.csv")
    for method, (key, scoring) in {"popularity": ("popularity", "-"),
                                   "A_content (max)": ("A_content_base", "max"),
                                   "B_content+labels (max)": ("B_content_enriched", "max"),
                                   "C_embedding (mean)": ("C_embedding_base", "mean"),
                                   "D_embedding+labels (mean)": ("D_embedding_enriched", "mean"),
                                   "E_item_cf (mean)": ("E_item_cf", "mean"),
                                   "hybrid_C_embedding": ("hybrid_C_embedding_base", "mean"),
                                   "hybrid_D_embedding+labels": ("hybrid_D_embedding_enriched", "mean"),
                                   "hybrid_E_item_cf": ("hybrid_E_item_cf", "mean")}.items():
        part = cold[(cold["method"] == key) & (cold["scoring"] == scoring)]
        for n, value in part.groupby("n")["ndcg"].mean().items():
            saved.append({"method": method, "n": n, "saved_ndcg": value})
    profiles = pd.read_csv("output/experiments/profiles/cold_per_user.csv")
    part = profiles[(profiles["method"] == "31b_P2") & (profiles["scoring"] == "mean")]
    for n, value in part.groupby("n")["ndcg"].mean().items():
        saved.append({"method": "embedding+31b_P2 (mean)", "n": n, "saved_ndcg": value})
    typed = pd.read_csv("output/experiments/typed_requests/per_user.csv")
    typed = typed[(typed["style"] == "r1") & (typed["n"] == 3) & (typed["representation"] == "base")]
    for method in ("request", "hybrid_request", "hybrid_request+seeds"):
        saved.append({"method": method, "n": 3, "saved_ndcg": typed[typed["method"] == method]["ndcg"].mean()})
    for model in STRUCTURED_MODELS:
        part = pd.read_csv(Path("output") / "experiments" / "structured_preferences" / f"test_results_{model}.csv")
        saved.append({"method": f"structured ({model})", "n": 3,
                      "saved_ndcg": part[part["method"] == "structured"]["ndcg"].mean()})

    summary = summarise(lists, pool_ids).merge(pd.DataFrame(saved), on=["method", "n"], how="left")
    summary["ndcg_matches_saved"] = (summary["ndcg"] - summary["saved_ndcg"]).abs() < 1e-4

    lists.assign(rec=lists["rec"].map(lambda r: "|".join(map(str, r)))).to_csv(FOLDER / "per_user.csv", index=False)
    summary.to_csv(FOLDER / "summary.csv", index=False)
    pd.DataFrame({"movie_id": sorted(pool_ids)}).to_csv(FOLDER / "pool.csv", index=False)

    pd.set_option("display.width", 250)
    print(f"\nPopularity thirds by TMDB vote count: low <= {cuts[0]:.0f} < mid <= {cuts[1]:.0f} < high")
    print(f"Test users: {len(test_users)} | candidate pool: {len(pool_ids)} movies\n")
    print(summary.drop(columns=["saved_ndcg"]).round(4).to_string(index=False))
    if not summary["ndcg_matches_saved"].all():
        print("\nWARNING: rebuilt NDCG differs from the saved results for:")
        print(summary.loc[~summary["ndcg_matches_saved"], ["method", "n", "ndcg", "saved_ndcg"]].to_string(index=False))
    print(f"\nSaved {FOLDER / 'summary.csv'}")
    run_tests()
