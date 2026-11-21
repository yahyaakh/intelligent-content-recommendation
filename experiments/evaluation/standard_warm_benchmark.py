'''
Standard warm-start benchmark (doc 35; pre-registration: output/experiments/standard_warm/preregistration.md).

The usual recommender task: the system knows a user's earlier ratings (likes AND dislikes) and must predict
the films they will like later. Warm users only (the 9,591 of cs.split_users); the cold users, including the
203 validation and 814 test users, are never loaded here.

  split T (primary)  each user's ratings in time order: first ceil(0.8 n) = training, the rest held out
  split R            random 75/25 of all warm ratings (python_random_split, seed 42), as docs 04-08
  liked              training rating >= the user's mean training rating
  relevant           held-out rating >= the user's mean TRAINING rating, pool films only
  pool               films with a training rating in that split
  users              >= 5 relevant held-out films and >= 10 liked training films
  never shown        every film the user rated in training

Settings (Hybrid E alpha, user-kNN k, EASE lambda, SVD) are chosen on an inner split of the training part
(T: last 20% again; R: random 25%, seed 43), then every model is refitted on the full training part.

Modes:
    python experiments/evaluation/standard_warm_benchmark.py check        user counts per split; no models
    python experiments/evaluation/standard_warm_benchmark.py fast         tuning + popularity, content, item-CF, Hybrid E,
                                                   user-kNN, EASE, SVD on both splits
    python experiments/evaluation/standard_warm_benchmark.py llm          LLM reranking on the first 20 of the 300-user sample (pilot)
    python experiments/evaluation/standard_warm_benchmark.py llm full     the same on all 300
    python experiments/evaluation/standard_warm_benchmark.py report       tables and tests from everything saved
'''

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))  # shared modules live in src/
import json
import math
import sys
from pathlib import Path

ARGS = sys.argv[1:]
sys.argv = [sys.argv[0]]                  # the imported experiment scripts read sys.argv when imported
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.stats import wilcoxon
from sklearn.preprocessing import normalize

import build_movie_texts as bmt
import cold_start_experiment as cs

OUT = Path("output") / "experiments" / "standard_warm"
K = bmt.K
TRAIN_SHARE_T = 0.8
MIN_RELEVANT, MIN_LIKED = 5, 10
KNN_GRID = [10, 20, 50, 100, 200, 400, 1000]
SVD_GRID = [(f, r) for f in (50, 100, 200) for r in (0.02, 0.05)]
SVD_FIXED = dict(n_epochs=20, lr_all=0.005, random_state=42)
TOP_M = 30                                # Hybrid E candidates the LLM reranks
LLM_USERS, LLM_PILOT, LLM_HISTORY = 300, 20, 50
LLM_MODELS = {"deepseek-v4.1-flash": "LLM: DeepSeek V4.1 Flash, p2", "kimi-k3": "LLM: Kimi K3, p2"}
LLM_PROVIDER, LLM_VERSION = "ollama_cloud", "p2"

NAMES = ["Popularity", "Content (embedding + LLM labels, max)", "Item-CF", "Hybrid E",
         "User-kNN", "EASE", "SVD (user in training)"]
REFERENCE = "Hybrid E"


# ───────────────────────── data and splits ─────────────────────────
def load_base():
    movies = bmt.load_movies()
    warm, _ = cs.split_users(bmt.prepare_ratings(movies))
    warm = warm.sort_values(["user_id", "timestamp", "movie_id"], kind="stable").reset_index(drop=True)
    movie_ids = movies["movie_id"].to_numpy()
    position_of = {m: i for i, m in enumerate(movie_ids)}
    return movies, warm, movie_ids, position_of


def split_time(ratings):
    """Per user, in time order (ratings are already sorted by user, time, movie): first ceil(0.8 n) train."""
    rank = ratings.groupby("user_id").cumcount()
    n = ratings.groupby("user_id")["movie_id"].transform("size")
    is_train = rank < np.ceil(TRAIN_SHARE_T * n)
    return ratings[is_train], ratings[~is_train]


def split_random(ratings, seed):
    from recommenders.datasets.python_splitters import python_random_split
    train, test = python_random_split(ratings, ratio=0.75, seed=seed)
    order = ["user_id", "timestamp", "movie_id"]
    return (train.sort_values(order, kind="stable").reset_index(drop=True),
            test.sort_values(order, kind="stable").reset_index(drop=True))


def make_split(warm, name, inner):
    """(train, held-out) of split 'T' or 'R'; inner=True splits the training part again (for tuning)."""
    train, test = split_time(warm) if name == "T" else split_random(warm, 42)
    if inner:
        train, test = split_time(train) if name == "T" else split_random(train, 43)
    return train, test


def build_users(train, test):
    """Evaluated users: liked training films in time order, seen = all training films, relevant set."""
    pool = set(train["movie_id"])
    mean = train.groupby("user_id")["rating"].mean()
    tr = train.assign(mean=train["user_id"].map(mean))
    liked = tr[tr["rating"] >= tr["mean"]].groupby("user_id")["movie_id"].apply(list)
    seen = train.groupby("user_id")["movie_id"].apply(set)
    te = test[test["user_id"].isin(mean.index) & test["movie_id"].isin(pool)]
    te = te[te["rating"] >= te["user_id"].map(mean)]
    relevant = te.groupby("user_id")["movie_id"].apply(set)
    users = []
    for u in sorted(relevant.index):
        if len(relevant[u]) >= MIN_RELEVANT and len(liked.get(u, [])) >= MIN_LIKED:
            users.append({"user_id": int(u), "liked": [int(m) for m in liked[u]],
                          "seen": seen[u], "relevant": relevant[u] - seen[u]})
    return users


# ───────────────────────── the models of one split ─────────────────────────
class Split:
    """Everything fitted on one training part: tables, popularity, user-kNN matrix, EASE, SVD."""
    def __init__(self, train, movies, movie_ids, position_of, content_table):
        self.train, self.movies, self.movie_ids, self.position_of = train, movies, movie_ids, position_of
        self.pool = np.array(sorted(position_of[m] for m in set(train["movie_id"])))  # positions, sorted
        self.pool_ids = movie_ids[self.pool].astype(int)
        self.column = {int(m): i for i, m in enumerate(self.pool_ids)}
        self.content = content_table[np.ix_(self.pool, self.pool)]   # pool x pool, same order everywhere
        self.item_cf = cs.item_cf_similarity(train, movies, position_of)[np.ix_(self.pool, self.pool)]
        self.popular = bmt.popularity_ranking(train)
        self.pop = cs.popularity_vector(self.popular, movies, position_of)[self.pool]
        # user-kNN: users x pool films, mean-centred, as cbl.UserKNN
        rows, ids = pd.factorize(train["user_id"])
        centred = (train["rating"] - train.groupby("user_id")["rating"].transform("mean")).to_numpy()
        self.centred = csr_matrix((centred, (rows, train["movie_id"].map(self.column).to_numpy())),
                                  shape=(len(ids), len(self.pool_ids)))
        self.centred.eliminate_zeros()
        self.unit = normalize(self.centred)
        self.row_of = {int(u): i for i, u in enumerate(ids)}
        self.ease = {}
        self.svd = {}

    # per-user pieces
    def cols(self, films):
        return [self.column[m] for m in films]

    def allowed(self, user):
        a = np.ones(len(self.pool_ids), dtype=bool)
        a[self.cols(user["seen"])] = False
        return a

    def top(self, scores, allowed, k=K):
        """Best k allowed pool films; ties keep pool order (stable), as cs and bmt."""
        return self.pool_ids[np.argsort(-np.where(allowed, scores, -np.inf), kind="stable")[:k]].tolist()

    def hybrid_parts(self, user, allowed):
        """Hybrid E's personal and popularity parts, rescaled over the user's candidates (cs.recommend_hybrid)."""
        liked = self.cols(user["liked"])
        personal = np.zeros(len(self.pool_ids), dtype=np.float32)
        prior = np.zeros(len(self.pool_ids), dtype=np.float32)
        personal[allowed] = cs.rescale(self.item_cf[np.ix_(liked, np.flatnonzero(allowed))].mean(axis=0))
        prior[allowed] = cs.rescale(self.pop[allowed])
        return personal, prior

    def hybrid(self, user, allowed, alpha, k=K):
        personal, prior = self.hybrid_parts(user, allowed)
        return self.top(alpha * personal + (1 - alpha) * prior, allowed, k)

    def knn_sims(self, users):
        """Cosine of each user's centred row to every user; the user's own similarity set to -inf."""
        rows = [self.row_of[u["user_id"]] for u in users]
        sims = (self.unit[rows] @ self.unit.T).toarray()
        sims[np.arange(len(rows)), rows] = -np.inf
        return sims

    def knn(self, sims, k):
        top = np.argsort(-sims, kind="stable")[:k]
        top = top[sims[top] > 0]
        return np.asarray(self.centred[top].T @ sims[top]).ravel()

    def fit_ease(self, lam):
        if lam not in self.ease:
            from recomm import EASE_R_Recommende
            model = EASE_R_Recommende(self.train, self.movies, lam)
            model.fit(self.train)
            # B in pool order (EASE's own columns = films of train = the pool)
            order = [model.film_to_col[m] for m in self.pool_ids]
            self.ease[lam] = model.B[np.ix_(order, order)]
        return self.ease[lam]

    def fit_svd(self, factors, reg):
        key = (factors, reg)
        if key not in self.svd:
            from surprise import SVD, Dataset, Reader
            trainset = Dataset.load_from_df(self.train[["user_id", "movie_id", "rating"]],
                                            Reader(rating_scale=(0.5, 5))).build_full_trainset()
            algo = SVD(n_factors=factors, reg_all=reg, **SVD_FIXED)
            algo.fit(trainset)
            items = [trainset.to_inner_iid(m) for m in self.pool_ids]
            self.svd[key] = {"mu": trainset.global_mean, "bu": algo.bu, "pu": algo.pu,
                             "bi": algo.bi[items], "qi": algo.qi[items],
                             "user": {int(trainset.to_raw_uid(i)): i for i in range(trainset.n_users)}}
        return self.svd[key]

    @staticmethod
    def svd_scores(model, user):
        i = model["user"][user["user_id"]]
        return model["mu"] + model["bu"][i] + model["bi"] + model["qi"] @ model["pu"][i]


def metrics(recommended, user):
    return bmt.user_metrics(recommended, user["relevant"], K)


def batches(users, size=500):
    for start in range(0, len(users), size):
        yield users[start:start + size]


# ───────────────────────── tuning on the inner split ─────────────────────────
def tune(name, base, content_table):
    movies, warm, movie_ids, position_of = base
    from ease_tuning import LAMBDAS
    train, test = make_split(warm, name, inner=True)
    users = build_users(train, test)
    print(f"[{name} inner] train {len(train):,} | held out {len(test):,} | evaluated users {len(users):,}")
    m = Split(train, movies, movie_ids, position_of, content_table)
    curves = []

    def add(setting, value, ndcg):
        curves.append({"split": name, "setting": setting, "value": value, "ndcg": float(ndcg)})
        print(f"  {setting} {value}: {ndcg:.4f}")

    # Hybrid E alpha and user-kNN k: one pass over the users, every grid value
    hyb = {a: [] for a in cs.ALPHAS}
    knn = {k: [] for k in KNN_GRID}
    ease_users = {lam: [] for lam in LAMBDAS}
    for group in batches(users):
        sims = m.knn_sims(group)
        X = np.zeros((len(group), len(m.pool_ids)))
        for j, u in enumerate(group):
            X[j, m.cols(u["liked"])] = 1
        for j, u in enumerate(group):
            allowed = m.allowed(u)
            personal, prior = m.hybrid_parts(u, allowed)
            for a in cs.ALPHAS:
                hyb[a].append(metrics(m.top(a * personal + (1 - a) * prior, allowed), u)["ndcg"])
            for k in KNN_GRID:
                knn[k].append(metrics(m.top(m.knn(sims[j], k), allowed), u)["ndcg"])
        for lam in LAMBDAS:
            S = X @ m.fit_ease(lam)
            for j, u in enumerate(group):
                ease_users[lam].append(metrics(m.top(S[j], m.allowed(u)), u)["ndcg"])
    for a in cs.ALPHAS:
        add("Hybrid E alpha", a, np.mean(hyb[a]))
    for k in KNN_GRID:
        add("user-kNN k", k, np.mean(knn[k]))
    for lam in LAMBDAS:
        add("EASE lambda", lam, np.mean(ease_users[lam]))
    m.ease.clear()
    for factors, reg in SVD_GRID:
        model = m.fit_svd(factors, reg)
        add("SVD factors/reg", f"{factors}/{reg}",
            np.mean([metrics(m.top(m.svd_scores(model, u), m.allowed(u)), u)["ndcg"] for u in users]))
        m.svd.clear()

    curves = pd.DataFrame(curves)

    def best(setting):
        part = curves[curves["setting"] == setting]
        return part.loc[part["ndcg"].idxmax(), "value"]           # first maximum = first in the grid
    f, r = best("SVD factors/reg").split("/")
    chosen = {"alpha": float(best("Hybrid E alpha")), "k": int(best("user-kNN k")),
              "lambda": int(best("EASE lambda")), "svd": [int(f), float(r)],
              "inner_users": len(users)}
    print(f"[{name}] chosen: {chosen}")
    return curves, chosen


# ───────────────────────── evaluation on the outer split ─────────────────────────
def evaluate(name, base, content_table, chosen):
    movies, warm, movie_ids, position_of = base
    train, test = make_split(warm, name, inner=False)
    users = build_users(train, test)
    print(f"[{name}] train {len(train):,} | held out {len(test):,} | evaluated users {len(users):,}")
    m = Split(train, movies, movie_ids, position_of, content_table)
    B = m.fit_ease(chosen["lambda"])
    svd = m.fit_svd(*chosen["svd"])
    rows, lists = [], {}
    for group in batches(users):
        sims = m.knn_sims(group)
        X = np.zeros((len(group), len(m.pool_ids)))
        for j, u in enumerate(group):
            X[j, m.cols(u["liked"])] = 1
        S_ease = X @ B
        for j, u in enumerate(group):
            allowed = m.allowed(u)
            liked = m.cols(u["liked"])
            recs = {
                "Popularity": cs.recommend_popular(m.popular.index, u, K),
                "Content (embedding + LLM labels, max)": m.top(m.content[liked].max(axis=0), allowed),
                "Item-CF": m.top(m.item_cf[liked].mean(axis=0), allowed),
                "Hybrid E": m.hybrid(u, allowed, chosen["alpha"]),
                "User-kNN": m.top(m.knn(sims[j], chosen["k"]), allowed),
                "EASE": m.top(S_ease[j], allowed),
                "SVD (user in training)": m.top(m.svd_scores(svd, u), allowed),
            }
            for method, rec in recs.items():
                rows.append({"split": name, "method": method, "user_id": u["user_id"],
                             "n_relevant": len(u["relevant"]), **metrics(rec, u)})
                lists.setdefault(method, set()).update(rec)

    # check: our Hybrid E and the cold-start implementation agree on the first 50 users
    full_cf = cs.item_cf_similarity(train, movies, position_of)
    full_pop = cs.popularity_vector(m.popular, movies, position_of)
    for u in users[:50]:
        mine = m.hybrid(u, m.allowed(u), chosen["alpha"])
        ref = cs.recommend_hybrid(full_cf, full_pop, u, m.pool, position_of, movie_ids, K, chosen["alpha"], "mean")
        assert mine == [int(x) for x in ref], f"Hybrid E differs for user {u['user_id']}"
    print("Check: Hybrid E equals cs.recommend_hybrid on the first 50 users")
    coverage = {method: len(films) / len(m.pool_ids) for method, films in lists.items()}
    return pd.DataFrame(rows), coverage, len(m.pool_ids)


def content_table_of(movies):
    return bmt.embedding_similarity(f"movies_enriched_{cs.EMBED_NAME}_{cs.PROMPT_VERSION}", movies)


def run_fast():
    OUT.mkdir(parents=True, exist_ok=True)
    base = load_base()
    content_table = content_table_of(base[0])
    settings = json.loads((OUT / "settings.json").read_text()) if (OUT / "settings.json").exists() else {}
    all_curves, all_rows, info = [], [], {}
    for name in ("T", "R"):
        if name not in settings:
            curves, chosen = tune(name, base, content_table)
            curves.to_csv(OUT / f"settings_curves_{name}.csv", index=False)
            settings[name] = chosen
            (OUT / "settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
        rows, coverage, n_pool = evaluate(name, base, content_table, settings[name])
        rows.to_csv(OUT / f"per_user_{name}.csv", index=False)
        info[name] = {"coverage": coverage, "pool": n_pool}
        print(rows.groupby("method", sort=False)[["ndcg", "precision", "recall"]].mean().round(4).to_string())
    (OUT / "coverage.json").write_text(json.dumps(info, indent=2), encoding="utf-8")


# ───────────────────────── LLM reranking (split T, 300 users) ─────────────────────────
def llm_sample(users):
    """300 evaluated users of split T, drawn once (seed 42) from the sorted ids; kept in the draw's order,
    so the pilot is the first 20."""
    ids = sorted(u["user_id"] for u in users)
    by_id = {u["user_id"]: u for u in users}
    return [by_id[int(i)] for i in np.random.default_rng(42).choice(ids, LLM_USERS, replace=False)]


def llm_setup():
    import llm_rerank_experiment as lr
    settings = json.loads((OUT / "settings.json").read_text())
    base = load_base()
    movies, warm, movie_ids, position_of = base
    train, test = make_split(warm, "T", inner=False)
    users = build_users(train, test)
    m = Split(train, movies, movie_ids, position_of, content_table_of(movies))
    movie_by_id = movies.set_index("movie_id")
    v4 = pd.read_csv("output/experiments/gemma4_e4b_v4/movies_enriched.csv").set_index("movie_id")
    movie_by_id["v4_themes"] = v4["llm_themes"].reindex(movie_by_id.index)   # as lr.load_data (the profile
    movie_by_id["v4_moods"] = v4["llm_mood"].reindex(movie_by_id.index)      # reads them; p2 does not print them)
    data = {"movie_by_id": movie_by_id}
    return lr, settings["T"]["alpha"], llm_sample(users), m, data


def llm_cache(model):
    return OUT / "llm" / f"answers_{LLM_PROVIDER}_{model.replace(':', '_')}_{LLM_VERSION}.jsonl"


def prompt_user(u):
    return {"user_id": u["user_id"], "liked": u["liked"][-LLM_HISTORY:]}  # 50 most recent, in time order


def run_llm(full):
    lr, alpha, sample, m, data = llm_setup()
    (OUT / "llm").mkdir(parents=True, exist_ok=True)
    group = sample if full else sample[:LLM_PILOT]
    for model in LLM_MODELS:
        path = llm_cache(model)
        done = ({json.loads(l)["user_id"] for l in path.read_text(encoding="utf-8").splitlines()}
                if path.exists() else set())
        todo = [u for u in group if u["user_id"] not in done]
        print(f"{model}: {len(done)} cached, {len(todo)} to call")
        with open(path, "a", encoding="utf-8") as f:
            for i, u in enumerate(todo):
                top_ids = m.hybrid(u, m.allowed(u), alpha, TOP_M)
                messages, _ = lr.build_prompt(prompt_user(u), top_ids, data, LLM_VERSION)
                result = lr.call_model(messages, LLM_PROVIDER, model)
                f.write(json.dumps({"user_id": u["user_id"], "model": model, "version": LLM_VERSION,
                                    "content": result["content"], "tokens_in": result["tokens_in"],
                                    "tokens_out": result["tokens_out"],
                                    "seconds": round(result["seconds"], 2)}, ensure_ascii=False) + "\n")
                f.flush()
                if (i + 1) % 20 == 0:
                    print(f"  {i + 1} / {len(todo)}")
    llm_rows(lr, alpha, sample, m, data, save=True)


def llm_rows(lr, alpha, sample, m, data, save=False):
    rows, usage = [], []
    for model, label in LLM_MODELS.items():
        path = llm_cache(model)
        if not path.exists():
            continue
        records = {r["user_id"]: r for r in map(json.loads, path.read_text(encoding="utf-8").splitlines())}
        for u in sample:
            if u["user_id"] not in records:
                continue
            top_ids = m.hybrid(u, m.allowed(u), alpha, TOP_M)
            _, number_to_id = lr.build_prompt(prompt_user(u), top_ids, data, LLM_VERSION)
            order, issues = lr.parse_ranking(records[u["user_id"]]["content"], number_to_id, top_ids)
            r = records[u["user_id"]]
            rows.append({"split": "T", "method": label, "user_id": u["user_id"],
                         "n_relevant": len(u["relevant"]), **metrics(order[:K], u)})
            usage.append({"model": model, "user_id": u["user_id"], "issues": len(issues),
                          "unparsable": "unparsable" in issues, "tokens_in": r["tokens_in"] or 0,
                          "tokens_out": r["tokens_out"] or 0, "seconds": r["seconds"]})
    if save and rows:
        pd.DataFrame(rows).to_csv(OUT / "per_user_llm.csv", index=False)
        u = pd.DataFrame(usage)
        u.to_csv(OUT / "llm_usage.csv", index=False)
        print(u.groupby("model")[["issues", "unparsable", "tokens_in", "tokens_out", "seconds"]]
              .agg(["mean", "sum"]).round(1).to_string())
    return rows


# ───────────────────────── report ─────────────────────────
def boot_ci(diff, seed=42, n=1000):
    rng = np.random.default_rng(seed)
    d = np.asarray(diff)
    means = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return np.percentile(means, [2.5, 97.5])


def compare_to(per_user, reference, methods, n_tests):
    wide = per_user.pivot(index="user_id", columns="method", values="ndcg")
    out = []
    for method in methods:
        if method == reference or method not in wide:
            continue
        diff = (wide[method] - wide[reference]).dropna()
        p = wilcoxon(diff).pvalue if diff.any() else 1.0
        lo, hi = boot_ci(diff)
        out.append({"method": method, "vs": reference, "users": len(diff), "difference": diff.mean(),
                    "ci_low": lo, "ci_high": hi, "better": int((diff > 0).sum()), "worse": int((diff < 0).sum()),
                    "p": p, "p_bonf": min(1.0, p * n_tests)})
    return pd.DataFrame(out)


def report():
    coverage = json.loads((OUT / "coverage.json").read_text())
    settings = json.loads((OUT / "settings.json").read_text())
    pd.set_option("display.width", 250)
    tables, tests = [], []
    for name in ("T", "R"):
        per_user = pd.read_csv(OUT / f"per_user_{name}.csv")
        ceiling = (np.minimum(K, per_user["n_relevant"]) / per_user["n_relevant"]).groupby(per_user["method"]).mean()
        board = per_user.groupby("method", sort=False)[["ndcg", "precision", "recall"]].mean()
        board["recall_ceiling"] = ceiling
        board["coverage"] = pd.Series(coverage[name]["coverage"])
        board["users"] = per_user.groupby("method", sort=False)["user_id"].nunique()
        board = board.sort_values("ndcg", ascending=False)
        board.insert(0, "split", name)
        board.insert(1, "rank", range(1, len(board) + 1))
        tables.append(board)
        t = compare_to(per_user, REFERENCE, NAMES, n_tests=len(NAMES) - 1)
        t.insert(0, "split", name)
        tests.append(t)
        # Q2: the best method vs every other
        best = board.index[0]
        t2 = compare_to(per_user, best, NAMES, n_tests=len(NAMES) - 1)
        t2.insert(0, "split", name)
        tests.append(t2.assign(question="Q2"))
        print(f"\nSplit {name} (settings {settings[name]}), pool {coverage[name]['pool']} films")
        print(board.round(4).to_string())
        print(t.round(4).to_string(index=False))
        print(f"Q2: best = {best}")
        print(t2.round(4).to_string(index=False))

    llm_path = OUT / "per_user_llm.csv"
    if llm_path.exists():
        llm = pd.read_csv(llm_path)
        base_t = pd.read_csv(OUT / "per_user_T.csv")
        sample = llm["user_id"].unique()
        both = pd.concat([base_t[base_t["user_id"].isin(sample)], llm], ignore_index=True)
        board = both.groupby("method", sort=False)[["ndcg", "precision", "recall"]].mean()
        board["users"] = both.groupby("method", sort=False)["user_id"].nunique()
        board = board.sort_values("ndcg", ascending=False)
        board.insert(0, "split", "T (LLM sample)")
        tables.append(board)
        t = compare_to(both, REFERENCE, list(LLM_MODELS.values()), n_tests=len(LLM_MODELS))
        t["within_0.02"] = (t["ci_low"] >= -0.02) & (t["ci_high"] <= 0.02)
        t.insert(0, "split", "T (LLM sample)")
        tests.append(t.assign(question="Q4"))
        print(f"\nLLM sample ({len(sample)} users with an answer), split T")
        print(board.round(4).to_string())
        print(t.round(4).to_string(index=False))
    pd.concat(tables).to_csv(OUT / "summary.csv")
    pd.concat(tests, ignore_index=True).to_csv(OUT / "tests.csv", index=False)
    print(f"\nSaved -> {OUT / 'summary.csv'}, {OUT / 'tests.csv'}")


def check():
    movies, warm, movie_ids, position_of = load_base()
    print(f"Warm ratings {len(warm):,}, users {warm['user_id'].nunique():,}")
    for name in ("T", "R"):
        for inner in (True, False):
            train, test = make_split(warm, name, inner)
            users = build_users(train, test)
            rel = np.mean([len(u["relevant"]) for u in users])
            liked = np.mean([len(u["liked"]) for u in users])
            print(f"{name} {'inner' if inner else 'outer'}: train {len(train):,} | held out {len(test):,} | "
                  f"users {len(users):,} | mean relevant {rel:.1f} | mean liked {liked:.1f}")


if __name__ == "__main__":
    mode = ARGS[0] if ARGS else "check"
    if mode == "check":
        check()
    elif mode == "fast":
        run_fast()
        report()
    elif mode == "llm":
        run_llm("full" in ARGS)
        report()
    elif mode == "report":
        report()
    else:
        raise SystemExit(__doc__)
