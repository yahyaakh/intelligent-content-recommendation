'''
Robustness of the two positive semantic-ID results (doc 33). Pre-registration:
output/experiments/sid_robustness/preregistration.md (written before the first run).

Tests (one mode each):
  check   reproduce doc 30 (S2 / S3, k-means seed 42) with this script's own scoring, then stop
  1c      per-user significance on the simulated users (existing doc 30 runs, no recomputation)
  2a      permutation test of the lifts
  1b      the 203 validation users
  1a      where the gain comes from: popularity thirds, coverage, novelty
  1d      other codebook settings
  1e      item-CF as another rating-based source
  2c      residual (hierarchical) vs flat IDs
  2b      LLM odd-one-out test (API calls)
Usage: python experiments/semantic_ids/sid_robustness.py <mode>
'''

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))  # shared modules live in src/
import sys

MODE = sys.argv[1] if len(sys.argv) > 1 else "check"
sys.argv = [sys.argv[0]]
import json
import time
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.stats import binomtest, fisher_exact, wilcoxon

import build_movie_texts as bmt
import cold_start_experiment as cs
import llm_rerank_experiment as lr
import LightGBM_XBoost_rerank_experiment as ltr
import sid_vector_sources as svs
from recomm import Semantic_IDs

OUT = Path("output") / "experiments" / "sid_robustness"
RUNS = OUT / "runs"
SVS = svs.OUT
KMEANS_SEEDS = [42, 43, 44, 45, 46]
SEED_COUNTS = (10, 3)
K = bmt.K
OTHER = {"A": "B", "B": "A"}
SOURCE_TAG = {"S2 content": "S2", "S3 SVD": "S3", "S4 EASE": "S4", "S5 item-CF": "S5", "S7 hybrid": "S7"}


# ───────────────────────── IDs and scoring ─────────────────────────
class Ids:
    """Codes per film. nested=True: residual IDs scored by shared PREFIX (as Semantic_IDs.recommend);
    nested=False: flat IDs scored by the number of levels with the same cluster."""
    def __init__(self, codes, movie_ids, nested=True):
        self.tokens = np.asarray(codes)
        self.movie_ids = np.asarray(movie_ids).astype(int)
        self.row = {int(m): i for i, m in enumerate(self.movie_ids)}
        self.nested = nested

    def recommend(self, seeds, exclude, allowed_mask, tie):
        rows = [self.row[m] for m in seeds if m in self.row]
        scores = np.zeros(len(self.movie_ids))
        if rows:
            same = self.tokens[None, :, :] == self.tokens[rows][:, None, :]          # (seeds, films, levels)
            if self.nested:
                same = np.cumprod(same, axis=2)
            scores = same.sum(axis=2).mean(axis=0).astype(float)
        hidden = [self.row[m] for m in list(seeds) + list(exclude) if m in self.row]
        scores[hidden] = -np.inf
        scores[~allowed_mask] = -np.inf
        order = np.lexsort((-tie, -scores))
        return [int(m) for m in self.movie_ids[order[:K]]]


def residual_ids(vectors, ids, movies, codes, levels, seed):
    model = Semantic_IDs(movies, n_levels=levels, codebook_size=codes, seed=seed)
    model.set_vectors(vectors, ids)
    model.fit()
    return Ids(model.codes[:, :levels], model.movie_ids)


def content_ids(movies, codes, levels, seed):
    model = Semantic_IDs(movies, n_levels=levels, codebook_size=codes, seed=seed)
    model.load_embeddings(name=svs.CONTENT_EMBEDDING)
    model.fit()
    return Ids(model.codes[:, :levels], model.movie_ids)


def saved_ids(source, fit_on, seed):
    z = np.load(SVS / "ids" / f"{SOURCE_TAG[source]}_{fit_on}_k32_s{seed}.npz")
    return Ids(z["codes"][:, :3], z["movie_ids"])


def vectors_of(source, part_name, parts, movies, data):
    """Film vectors of a rated source, built from one part (A, B or all); cached in sid_vector_sources/vectors."""
    if source == "S5 item-CF":
        return svs.cached_vectors("S5 item-CF", part_name, lambda: item_cf_vectors(parts[part_name], movies, data))
    path = SVS / "vectors" / f"{SOURCE_TAG[source]}_{part_name}.npy"
    return np.load(path), np.load(str(path).replace(".npy", "_ids.npy"))


def item_cf_vectors(part, movies, data):
    sim = cs.item_cf_similarity(part, movies, data["position_of"])
    pool = np.array(data["pool"])
    rated = np.sort(part["movie_id"].unique())
    rows = np.array([data["position_of"][m] for m in rated])
    vectors = np.asarray(sim[np.ix_(rows, pool)], dtype=np.float32)
    keep = np.linalg.norm(vectors, axis=1) > 1e-12
    print(f"  item-CF vectors: {keep.sum()} films ({(~keep).sum()} all-zero rows dropped)")
    return vectors[keep], rated[keep]


# ───────────────────────── shared set-up ─────────────────────────
def setup():
    data = lr.load_data()
    pool_ids = set(int(data["movie_ids"][p]) for p in data["pool"])
    halves = {}
    for n in SEED_COUNTS:
        train, in_A, users_by_half = ltr.warm_halves(data, n)
        halves[n] = users_by_half
    parts = {"A": train[in_A], "B": train[~in_A], "all": train}
    popular = {h: cs.popularity_scores(parts[OTHER[h]]) for h in ("A", "B")}
    popular["all"] = cs.popularity_scores(train)
    user_mean = train.groupby("user_id")["rating"].transform("mean")
    likes = train[train["rating"] >= user_mean]["movie_id"].value_counts()
    film_likes = pd.Series({m: int(likes.get(m, 0)) for m in pool_ids})
    third = pd.qcut(film_likes.rank(method="first"), 3, labels=["low", "mid", "high"])
    n_users = train["user_id"].nunique()
    novelty = -np.log2(np.maximum(film_likes, 1) / n_users)
    return dict(data=data, movies=data["movies"], pool_ids=pool_ids, halves=halves, parts=parts, popular=popular,
                third=third.to_dict(), novelty=novelty.to_dict())


def run_use1(ids_of_half, S, method, extra, ties=("popularity",), keep_lists=False, users=None):
    """ids_of_half: half -> Ids. Returns per-user rows (and lists). users: {n: {half: [users]}} (default simulated)."""
    users = users or S["halves"]
    rows = []
    for half, ids in ids_of_half.items():
        allowed = np.isin(ids.movie_ids, list(S["pool_ids"]))
        tie_vals = {"popularity": S["popular"][half].reindex(ids.movie_ids).fillna(0).to_numpy(),
                    "random": np.random.default_rng(0).random(len(ids.movie_ids))}
        for tie_name in ties:
            for n in users:
                for user in users[n][half]:
                    rec = ids.recommend(user["liked"], user["seen"], allowed, tie_vals[tie_name])
                    row = {"method": method, **extra, "ties": tie_name, "seeds": n, "half": half,
                           "user_id": user["user_id"], **bmt.user_metrics(rec, set(user["relevant"]), K)}
                    if keep_lists:
                        row["list"] = " ".join(map(str, rec))
                    rows.append(row)
    return rows


def cached_run(name, build):
    path = RUNS / f"{name}.csv"
    if path.exists():
        return pd.read_csv(path)
    start = time.time()
    frame = pd.DataFrame(build())
    frame.to_csv(path, index=False)
    print(f"  {name}: done ({time.time() - start:.0f} s) " + ", ".join(
        f"{n} seeds {frame[(frame.seeds == n) & (frame.ties == 'popularity')].ndcg.mean():.4f}"
        for n in sorted(frame.seeds.unique(), reverse=True)), flush=True)
    return frame


def per_half_ids(source, S, seed, codes=32, levels=3):
    if source == "S2 content":
        ids = content_ids(S["movies"], codes, levels, seed)
        return {"A": ids, "B": ids}
    if (codes, levels) == (32, 3) and source != "S5 item-CF":
        return {h: saved_ids(source, OTHER[h], seed) for h in ("A", "B")}       # the exact doc 30 IDs
    return {h: residual_ids(*vectors_of(source, OTHER[h], S["parts"], S["movies"], S["data"]), S["movies"],
                            codes, levels, seed) for h in ("A", "B")}


def seed_rule(frame, a, b, label):
    """frame: per-run means with columns method, kmeans_seed, ndcg (10 seeds, popularity ties)."""
    wide = frame.pivot_table(index="kmeans_seed", columns="method", values="ndcg")
    diff = (wide[a] - wide[b]).dropna()
    ok = len(diff) == 5 and diff.mean() > 0 and (diff > 0).sum() >= 4
    return {"comparison": f"{a} - {b}", "setting": label, "mean_diff": diff.mean(), "wins": int((diff > 0).sum()),
            "passes": ok}


def bootstrap_ci(diff, n=10_000):
    rng = np.random.default_rng(42)
    return np.percentile(rng.choice(diff, size=(n, len(diff))).mean(axis=1), [2.5, 97.5])


def paired(per_user, a, b, metric):
    d = (per_user[a] - per_user[b]).dropna()
    lo, hi = bootstrap_ci(d.to_numpy())
    return {"comparison": f"{a} - {b}", "metric": metric, "users": len(d), "mean_a": per_user[a].mean(),
            "mean_b": per_user[b].mean(), "difference": d.mean(), "ci_low": lo, "ci_high": hi,
            "better": int((d > 0).sum()), "worse": int((d < 0).sum()), "p_value": wilcoxon(d).pvalue}


# ───────────────────────── tests ─────────────────────────
def t_check(S):
    for source, expected in (("S2 content", 0.40581436961289197), ("S3 SVD", 0.4941)):
        frame = pd.DataFrame(run_use1(per_half_ids(source, S, 42), S, source, {"kmeans_seed": 42}))
        got = frame[frame.seeds == 10].ndcg.mean()
        print(f"check {source}, seed 42, 10 seeds: {got:.6f} (doc 30: {expected})")
        assert abs(got - expected) < 1e-4, "this script's scoring does not reproduce doc 30"
    print("check passed")


def t_1c(S):
    runs = pd.concat([pd.read_csv(p) for p in sorted((SVS / "runs_metrics").glob("use1_S*.csv"))], ignore_index=True)
    runs = runs[runs.ties == "popularity"]
    rows = []
    for n in SEED_COUNTS:
        part = runs[runs.seeds == n]
        for metric in ("ndcg", "precision", "recall"):
            per_user = part.groupby(["method", "user_id"])[metric].mean().unstack("method")
            for a in ("S3 SVD", "S4 EASE", "S7 hybrid"):
                rows.append({"seeds": n, **paired(per_user, a, "S2 content", metric)})
    t = pd.DataFrame(rows)
    nd = t.metric == "ndcg"
    t.loc[nd, "p_bonferroni"] = (t.loc[nd, "p_value"] * nd.sum()).clip(upper=1)
    t.to_csv(OUT / "t1c_significance.csv", index=False)
    print(t.round(5).to_string(index=False))


def coliked_sum(cov, labels):
    """Sum and count of covis over ordered off-diagonal pairs with the same label."""
    _, inv = np.unique(labels, return_inverse=True)
    M = csr_matrix((np.ones(len(inv)), (np.arange(len(inv)), inv)))
    total = float((M.multiply(cov @ M)).sum())                 # diagonal of cov is 0
    sizes = np.bincount(inv)
    return total, float((sizes * (sizes - 1)).sum())


def lifts(codes, attrs, cov, base_cov):
    out = {}
    n = len(codes)
    for L in (1, 2, 3):
        key = pd.Series([tuple(c) for c in codes[:, :L]]).factorize()[0]
        g = pd.Series(key)
        prefix_pairs = (g.value_counts() * (g.value_counts() - 1) / 2).sum()
        for name, values in attrs.items():
            both = pd.Series(list(zip(key, values))).value_counts()
            out[(name, L)] = ((both * (both - 1) / 2).sum() / prefix_pairs) / values_random(values, n)
        total, count = coliked_sum(cov, key)
        out[("co-liked", L)] = (total / count) / base_cov
    return out


_RANDOM = {}
def values_random(values, n):
    key = id(values)
    if key not in _RANDOM:
        c = pd.Series(values).value_counts()
        _RANDOM[key] = (c * (c - 1) / 2).sum() / (n * (n - 1) / 2)
    return _RANDOM[key]


def t_2a(S):
    data, movies = S["data"], S["movies"].set_index("movie_id")
    covis_all = ltr.latent_tables(S["parts"]["all"], data)["covis"]
    rng = np.random.default_rng(42)
    rows = []
    for source in ("S2 content", "S3 SVD", "S4 EASE", "S7 hybrid"):
        z = np.load(SVS / "ids" / f"{SOURCE_TAG[source]}_all-pool_k32_s42.npz")
        films, codes = z["movie_ids"].astype(int), z["codes"][:, :3]
        pos = np.array([data["position_of"][m] for m in films])
        cov = covis_all[np.ix_(pos, pos)].astype(np.float64)
        base_cov = cov[~np.eye(len(films), dtype=bool)].mean()
        attrs = {"genre": movies.loc[films, "genres"].str.split("|").str[0].to_numpy(),
                 "director": movies.loc[films, "director"].fillna("?").to_numpy(),
                 "decade": (movies.loc[films, "year"] // 10 * 10).to_numpy()}
        _RANDOM.clear()
        observed = lifts(codes, attrs, cov, base_cov)
        null = {k: [] for k in observed}
        for i in range(1000):
            perm = codes[rng.permutation(len(codes))]
            for k, v in lifts(perm, attrs, cov, base_cov).items():
                null[k].append(v)
        for (attr, L), v in observed.items():
            nv = np.array(null[(attr, L)])
            rows.append({"source": source, "attribute": attr, "level": L, "lift": v, "null_mean": nv.mean(),
                         "null_p99": np.percentile(nv, 99), "p_value": ((nv >= v).sum() + 1) / 1001})
        print(f"  2a: {source} done", flush=True)
    t = pd.DataFrame(rows)
    t["p_bonferroni"] = (t["p_value"] * len(t)).clip(upper=1)
    t.to_csv(OUT / "t2a_permutation.csv", index=False)
    print(t.round(4).to_string(index=False))


def t_1b(S):
    data = S["data"]
    users = {n: {"all": cs.users_with_n_seeds(data["val_users"], n)} for n in SEED_COUNTS}
    S = dict(S, popular=dict(S["popular"]))
    frames = []
    for seed in KMEANS_SEEDS:
        for source in ("S2 content", "S3 SVD", "S4 EASE", "S7 hybrid"):
            def build(source=source, seed=seed):
                if source == "S2 content":
                    ids = content_ids(S["movies"], 32, 3, seed)
                else:
                    ids = residual_ids(*vectors_of(source, "all", S["parts"], S["movies"], data), S["movies"], 32, 3, seed)
                return run_use1({"all": ids}, S, source, {"kmeans_seed": seed}, users=users)
            frames.append(cached_run(f"t1b_{SOURCE_TAG[source]}_s{seed}", build))
    runs = pd.concat(frames, ignore_index=True)
    pop = []
    for n in SEED_COUNTS:
        for user in users[n]["all"]:
            rec = cs.recommend_popular(S["popular"]["all"].index, user, K)
            pop.append({"seeds": n, "user_id": user["user_id"], **bmt.user_metrics([int(m) for m in rec], set(user["relevant"]), K)})
    pop = pd.DataFrame(pop)
    rows = []
    for n in SEED_COUNTS:
        part = runs[runs.seeds == n]
        for metric in ("ndcg", "precision", "recall"):
            per_user = part.groupby(["method", "user_id"])[metric].mean().unstack("method")
            for a in ("S3 SVD", "S4 EASE", "S7 hybrid"):
                rows.append({"seeds": n, **paired(per_user, a, "S2 content", metric),
                             "popularity": pop[pop.seeds == n][metric].mean()})
    t = pd.DataFrame(rows)
    main = (t.metric == "ndcg") & t.comparison.str.match("S[34]")
    t.loc[main, "p_bonferroni"] = (t.loc[main, "p_value"] * 2).clip(upper=1)
    t.to_csv(OUT / "t1b_validation.csv", index=False)
    print(t.round(5).to_string(index=False))
    m = t[(t.seeds == 10) & main]
    print("1b rule (10 seeds): " + ("CONFIRMED" if ((m.difference > 0) & (m.p_bonferroni < 0.05)).all() else "not confirmed"))


def t_1a(S):
    frames = []
    for seed in KMEANS_SEEDS:
        for source in ("S2 content", "S3 SVD", "S4 EASE", "S7 hybrid"):
            frames.append(cached_run(f"t1a_{SOURCE_TAG[source]}_s{seed}", lambda source=source, seed=seed: run_use1(
                per_half_ids(source, S, seed), S, source, {"kmeans_seed": seed}, ties=("popularity", "random"),
                keep_lists=True)))
    runs = pd.concat(frames, ignore_index=True)
    third, novelty = S["third"], S["novelty"]
    relevant = {(n, u["user_id"]): set(u["relevant"]) for n in S["halves"] for h in S["halves"][n] for u in S["halves"][n][h]}
    per_user = []
    for r in runs.itertuples():
        rec = [int(m) for m in r.list.split()]
        rel = relevant[(r.seeds, r.user_id)]
        row = {"method": r.method, "kmeans_seed": r.kmeans_seed, "ties": r.ties, "seeds": r.seeds, "user_id": r.user_id,
               "ndcg": r.ndcg, "precision": r.precision, "recall": r.recall,
               "low_share": np.mean([third[m] == "low" for m in rec]), "novelty": np.mean([novelty[m] for m in rec])}
        for t in ("low", "mid", "high"):
            rt = {m for m in rel if third.get(m) == t}
            row[f"ndcg_{t}"] = bmt.ndcg_at_k(rec, rt, K) if rt else np.nan
        per_user.append(row)
    per_user = pd.DataFrame(per_user)
    cols = ["ndcg", "precision", "recall", "ndcg_low", "ndcg_mid", "ndcg_high", "low_share", "novelty"]
    per_run = per_user.groupby(["seeds", "ties", "method", "kmeans_seed"])[cols].mean()
    cover = runs.groupby(["seeds", "ties", "method", "kmeans_seed"])["list"].apply(
        lambda s: pd.Series(" ".join(s).split()).astype(int))
    stats = []
    for key, films in cover.groupby(level=[0, 1, 2, 3]):
        counts = films.value_counts().reindex(sorted(S["pool_ids"]), fill_value=0).sort_values().to_numpy()
        n = len(counts)
        gini = (2 * np.arange(1, n + 1) - n - 1) @ counts / (n * counts.sum())
        stats.append({"seeds": key[0], "ties": key[1], "method": key[2], "kmeans_seed": key[3],
                      "coverage": (counts > 0).mean(), "gini": gini})
    per_run = per_run.reset_index().merge(pd.DataFrame(stats), on=["seeds", "ties", "method", "kmeans_seed"])
    per_run.to_csv(OUT / "t1a_per_run.csv", index=False)
    summary = per_run.groupby(["seeds", "ties", "method"])[cols + ["coverage", "gini"]].mean()
    summary.to_csv(OUT / "t1a_summary.csv")
    pd.set_option("display.width", 250)
    print(summary.round(4).to_string())
    rules = []
    main = per_run[(per_run.seeds == 10) & (per_run.ties == "popularity")]
    for t in ("low", "mid"):
        for a in ("S3 SVD", "S4 EASE"):
            rules.append(seed_rule(main.rename(columns={f"ndcg_{t}": "ndcg_t"}).assign(ndcg=lambda d: d["ndcg_t"]),
                                   a, "S2 content", f"ndcg_{t}"))
    rules = pd.DataFrame(rules)
    rules.to_csv(OUT / "t1a_rule.csv", index=False)
    print(rules.round(4).to_string(index=False))
    print("1a rule: gain is " + ("NOT ONLY POPULAR FILMS" if rules.passes.all() else "not shown beyond popular films"))


def t_1d(S):
    rows = []
    for codes, levels in ((16, 3), (64, 3), (128, 3), (32, 2), (32, 4)):
        frames = []
        for seed in KMEANS_SEEDS:
            for source in ("S2 content", "S3 SVD", "S4 EASE"):
                frames.append(cached_run(f"t1d_{SOURCE_TAG[source]}_k{codes}_L{levels}_s{seed}",
                                         lambda source=source, seed=seed: run_use1(
                                             per_half_ids(source, S, seed, codes, levels), S, source,
                                             {"kmeans_seed": seed, "codes": codes, "levels": levels})))
        runs = pd.concat(frames, ignore_index=True)
        per_run = runs.groupby(["seeds", "method", "kmeans_seed"])[["ndcg", "precision", "recall"]].mean().reset_index()
        per_run.assign(codes=codes, levels=levels).to_csv(OUT / f"t1d_k{codes}_L{levels}.csv", index=False)
        main = per_run[per_run.seeds == 10]
        for a in ("S3 SVD", "S4 EASE"):
            rows.append({**seed_rule(main, a, "S2 content", f"{codes} x {levels}"),
                         "mean_a": main[main.method == a].ndcg.mean(),
                         "mean_S2": main[main.method == "S2 content"].ndcg.mean()})
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "t1d_rule.csv", index=False)
    print(t.round(4).to_string(index=False))
    print("1d rule: " + ("HOLDS IN EVERY SETTING" if t.passes.all() else "does not hold in every setting"))


def t_1e(S):
    frames = []
    for seed in KMEANS_SEEDS:
        frames.append(cached_run(f"t1e_S5_s{seed}", lambda seed=seed: run_use1(
            per_half_ids("S5 item-CF", S, seed), S, "S5 item-CF", {"kmeans_seed": seed})))
    s5 = pd.concat(frames, ignore_index=True)
    old = pd.concat([pd.read_csv(p) for p in sorted((SVS / "runs_metrics").glob("use1_S*.csv"))], ignore_index=True)
    old = old[old.ties == "popularity"]
    runs = pd.concat([s5, old], ignore_index=True)
    per_run = runs.groupby(["seeds", "method", "kmeans_seed"])[["ndcg", "precision", "recall"]].mean().reset_index()
    per_run.to_csv(OUT / "t1e_per_run.csv", index=False)
    print(per_run.groupby(["seeds", "method"])[["ndcg", "precision", "recall"]].mean().round(4).to_string())
    rule = seed_rule(per_run[per_run.seeds == 10], "S5 item-CF", "S2 content", "32 x 3")
    print(rule)
    print("1e rule: S5 " + ("BETTER than S2" if rule["passes"] else "not better than S2"))


def flat_ids(vectors, ids, ref, seed):
    """Independent k-means per level with as many clusters as `ref` (residual IDs) has distinct prefixes."""
    v = np.ascontiguousarray((vectors / np.linalg.norm(vectors, axis=1, keepdims=True)).astype(np.float32))
    row = {int(m): i for i, m in enumerate(ids)}
    v = v[[row[m] for m in ref.movie_ids]]
    codes = []
    for L in (1, 2, 3):
        k = len({tuple(c) for c in ref.tokens[:, :L]})
        km = faiss.Kmeans(v.shape[1], k, niter=25, nredo=10, seed=seed, verbose=False, min_points_per_centroid=1)
        km.train(v)
        codes.append(km.index.search(v, 1)[1][:, 0])
    return Ids(np.stack(codes, axis=1), ref.movie_ids, nested=False)


def t_2c(S):
    frames = []
    c_vec, c_ids = svs.content_vectors(S["movies"])
    for seed in KMEANS_SEEDS:
        for source in ("S2 content", "S4 EASE"):
            def build(source=source, seed=seed):
                residual = per_half_ids(source, S, seed)
                flat = {}
                for h in ("A", "B"):
                    vec = (c_vec, c_ids) if source == "S2 content" else vectors_of(source, OTHER[h], S["parts"], S["movies"], S["data"])
                    flat[h] = flat_ids(*vec, residual[h], seed)
                return run_use1(flat, S, f"{source} flat", {"kmeans_seed": seed})
            frames.append(cached_run(f"t2c_{SOURCE_TAG[source]}_flat_s{seed}", build))
    flat = pd.concat(frames, ignore_index=True)
    old = pd.concat([pd.read_csv(p) for p in sorted((SVS / "runs_metrics").glob("use1_S[24]*.csv"))], ignore_index=True)
    old = old[old.ties == "popularity"]
    per_run = pd.concat([flat, old]).groupby(["seeds", "method", "kmeans_seed"])[["ndcg", "precision", "recall"]].mean().reset_index()
    per_run.to_csv(OUT / "t2c_per_run.csv", index=False)
    print(per_run.groupby(["seeds", "method"])[["ndcg", "precision", "recall"]].mean().round(4).to_string())
    rules = pd.DataFrame([seed_rule(per_run[per_run.seeds == 10], s, f"{s} flat", "residual vs flat")
                          for s in ("S2 content", "S4 EASE")])
    rules.to_csv(OUT / "t2c_rule.csv", index=False)
    print(rules.round(4).to_string(index=False))


def t_2b(S):
    data = S["data"]
    movies = S["movies"].set_index("movie_id")
    likes = pd.Series({m: -S["novelty"][m] for m in S["pool_ids"]})          # higher = more liked
    known = set(likes[likes >= likes.median()].index)
    conditions = {}
    for source in ("S2 content", "S4 EASE"):
        z = np.load(SVS / "ids" / f"{SOURCE_TAG[source]}_all-pool_k32_s42.npz")
        conditions[source] = dict(zip(z["movie_ids"].astype(int), z["codes"][:, 0].astype(int)))
    s4 = conditions["S4 EASE"]
    films = list(s4)
    conditions["random groups"] = dict(zip(films, np.random.default_rng(42).permutation([s4[m] for m in films])))
    cache_path = OUT / "t2b_answers.jsonl"
    cache = {}
    if cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            cache[(r["condition"], r["trial"])] = r
    rows = []
    with open(cache_path, "a", encoding="utf-8") as f:
        for condition, group_of in conditions.items():
            rng = np.random.default_rng(42)
            members = {}
            for m, g in group_of.items():
                if m in known:
                    members.setdefault(g, []).append(m)
            groups = sorted(g for g, ms in members.items() if len(ms) >= 5)
            for trial in range(100):
                g = groups[rng.integers(len(groups))]
                five = list(rng.choice(sorted(members[g]), 5, replace=False))
                others = sorted(m for gg, ms in members.items() if gg != g for m in ms)
                intruder = int(rng.choice(others))
                shown = [int(x) for x in rng.permutation(five + [intruder])]
                answer = shown.index(intruder) + 1
                record = cache.get((condition, trial))
                if record is None:
                    lines = "\n".join(f"{i + 1}. {movies.loc[m, 'title']} ({movies.loc[m, 'year']})" for i, m in enumerate(shown))
                    prompt = ("Below are six films. Five of them were put in the same group by a movie recommender "
                              f"system; one comes from a different group.\n{lines}\n\nWhich film is the odd one out? "
                              'Return JSON in exactly this form: {"odd": number}')
                    result = lr.call_model([{"role": "system", "content": lr.SYSTEM_PROMPT},
                                            {"role": "user", "content": prompt}], "ollama_cloud", "deepseek-v4.1-flash")
                    record = {"condition": condition, "trial": trial, "shown": shown, "answer": answer,
                              "content": result["content"], "tokens_in": result["tokens_in"]}
                    f.write(json.dumps(record) + "\n")
                    f.flush()
                try:
                    guess = int(json.loads(record["content"])["odd"])
                except (ValueError, KeyError, TypeError):
                    guess = -1
                rows.append({"condition": condition, "trial": trial, "correct": guess == record["answer"],
                             "parsed": guess != -1})
            print(f"  2b: {condition} done", flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "t2b_trials.csv", index=False)
    summary = []
    for condition, g in t.groupby("condition", sort=False):
        k = int(g.correct.sum())
        p = binomtest(k, len(g), 1 / 6, alternative="greater").pvalue
        summary.append({"condition": condition, "trials": len(g), "correct": k, "accuracy": k / len(g),
                        "chance": 1 / 6, "p_value": p, "p_bonferroni": min(1.0, 3 * p), "unparsed": int((~g.parsed).sum())})
    summary = pd.DataFrame(summary)
    a, b = summary.set_index("condition").loc[["S2 content", "S4 EASE"], "correct"]
    summary.to_csv(OUT / "t2b_summary.csv", index=False)
    print(summary.round(5).to_string(index=False))
    print(f"content vs EASE, Fisher exact p = {fisher_exact([[a, 100 - a], [b, 100 - b]]).pvalue:.4f}")


def main():
    RUNS.mkdir(parents=True, exist_ok=True)
    S = setup()
    tests = {"check": t_check, "1c": t_1c, "2a": t_2a, "1b": t_1b, "1a": t_1a, "1d": t_1d, "1e": t_1e,
             "2c": t_2c, "2b": t_2b}
    for mode in MODE.split(","):
        print(f"\n######## {mode} ########", flush=True)
        tests[mode](S)


if __name__ == "__main__":
    main()
