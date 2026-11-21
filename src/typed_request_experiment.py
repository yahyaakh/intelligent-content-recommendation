"""Typed requests for new users (thesis doc 09, evaluation A: preference interpretation).

For every cold-start user (experiment 06) an LLM writes the short request the user could type
into the recommender, based on their N seed movies ("I like dark, slow-burning crime dramas
with moral ambiguity"); no film may be named. The request is embedded with embeddinggemma's
query prefix and matched against the movie vectors. Relevant movies are the user's later liked
movies, exactly as in experiment 06.

Compared on the same users and seeds:
    popularity | seeds only | request only | request + seeds | + popularity (hybrids)
and, for every request method, four movie representations:
    base text | + v2 labels | + v4 labels | + gemma4:31b P2 profiles

Usage:
    python src/typed_request_experiment.py pilot     # 20 validation users, prints the requests
    python src/typed_request_experiment.py           # all users (requests are cached and resumable)
"""
import json
import re
import sys
import time
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

import build_movie_texts as bmt
import cold_start_experiment as cs

FOLDER = Path("output") / "experiments" / "typed_requests"
OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
REQUEST_MODEL = "gemma4:e4b"
QUERY_PREFIX = "task: search result | query: "   # embeddinggemma's prompt for retrieval queries
REQUEST_SEEDS = [3, 10]

REPRESENTATIONS = {                         # name -> (index name, folder)
    "base": ("movies_base_embeddinggemma_v2", bmt.EMBEDDINGS_FOLDER),
    "v2_labels": ("movies_enriched_embeddinggemma_v2", bmt.EMBEDDINGS_FOLDER),
    "v4_labels": ("movies_enriched_index", Path("output") / "experiments" / "gemma4_e4b_v4"),
    "31b_P2": ("movies_profile_31b_P2_embeddinggemma", Path("output") / "experiments" / "profiles"),
}

SYSTEM_PROMPT = ("You help simulate users of a movie recommendation system. You always return "
                 "only valid JSON, with no explanation or extra text.")

TASK = """A person has just signed up to a movie recommendation system. These are movies they told the system they enjoyed:

{seeds}

Write the short request this person could type into the system to describe what kind of movies they want, in their own words (1 to 2 sentences, at most 40 words).
Describe the kinds of films, themes, moods and styles they seem to like. Write as the person, in the first person.
Do not name any film, actor, director, character or franchise.

OUTPUT FORMAT
{{"request": "..."}}"""

# r2 = r1 plus ONE instruction asking for specific wording (simulates a user who describes
# their taste carefully); everything else is identical.
TASK_R2 = TASK.replace(
    "Do not name any film, actor, director, character or franchise.",
    "Be specific: name the particular themes, moods, settings and kinds of stories these films share. "
    "Avoid vague words such as \"intense\", \"high stakes\" or \"gripping\" unless they are the main point.\n"
    "Do not name any film, actor, director, character or franchise.")
assert TASK_R2 != TASK
STYLES = {"r1": TASK, "r2": TASK_R2}      # request style -> prompt

SCHEMA = {"type": "object", "properties": {"request": {"type": "string"}}, "required": ["request"]}


# ───────────────────────── requests ─────────────────────────
def seed_lines(seed_ids, movie_by_id):
    lines = []
    for m in seed_ids:
        movie = movie_by_id[m]
        overview = str(movie["overview"]) if pd.notna(movie["overview"]) else ""
        overview = overview if len(overview) <= 200 else overview[:200].rsplit(" ", 1)[0] + "..."
        genres = str(movie["genres"]).replace("|", ", ") if pd.notna(movie["genres"]) else ""
        lines.append(f"- {movie['title']} ({int(movie['year'])}). {genres}. {overview}")
    return "\n".join(lines)


def call_llm(prompt, retries=3):
    payload = {"model": REQUEST_MODEL, "stream": False, "think": False, "format": SCHEMA,
               "options": {"temperature": 0, "seed": 42},
               "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]}
    for attempt in range(retries + 1):
        try:
            res = requests.post(OLLAMA_URL, json=payload, timeout=300)
            res.raise_for_status()
            text = re.sub(r"^```(?:json)?|```$", "", res.json()["message"]["content"].strip()).strip()
            return " ".join(str(json.loads(text)["request"]).split())
        except (requests.RequestException, ValueError, KeyError) as error:
            last = str(error)
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"request generation failed: {last}")


def names_a_seed(request, seed_ids, movie_by_id):
    """True if the request contains a seed title (titles of at least 5 characters)."""
    text = request.lower()
    return any(len(t) >= 5 and t in text for t in (str(movie_by_id[m]["title"]).lower() for m in seed_ids))


def generate_requests(users, movie_by_id, cache, style):
    done = {}
    if cache.exists():
        for line in cache.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            done[(r["user_id"], r["n"])] = r
    todo = [(u, n) for u in users for n in REQUEST_SEEDS if (u["user_id"], n) not in done]
    with open(cache, "a", encoding="utf-8") as f:
        for user, n in tqdm(todo, desc="writing requests"):
            seeds = user["seed_pool"][:n]
            start = time.perf_counter()
            text = call_llm(STYLES[style].format(seeds=seed_lines(seeds, movie_by_id)))
            record = {"user_id": int(user["user_id"]), "n": n, "seeds": [int(s) for s in seeds],
                      "request": text, "words": len(text.split()),
                      "names_a_seed": names_a_seed(text, seeds, movie_by_id),
                      "seconds": round(time.perf_counter() - start, 2),
                      "model": REQUEST_MODEL, "style": style}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            done[(record["user_id"], n)] = record
    return done


def embed_requests(texts):
    vectors = []
    for start in range(0, len(texts), 32):
        res = requests.post("http://127.0.0.1:11434/api/embed", timeout=120,
                            json={"model": "embeddinggemma", "input": [QUERY_PREFIX + t for t in texts[start:start + 32]]})
        res.raise_for_status()
        vectors.extend(res.json()["embeddings"])
    matrix = np.asarray(vectors, dtype=np.float32)
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


# ───────────────────────── movie vectors ─────────────────────────
def movie_vectors(movies):
    out = {}
    for name, (index_name, folder) in REPRESENTATIONS.items():
        index = faiss.read_index(str(folder / f"{index_name}.faiss"))
        ids = np.load(folder / f"{index_name}_ids.npy")
        row_of = {m: r for r, m in enumerate(ids)}
        vectors = index.reconstruct_n(0, index.ntotal)[[row_of[m] for m in movies["movie_id"]]]
        out[name] = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    return out


# ───────────────────────── evaluation ─────────────────────────
WEIGHTS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]   # weight of the request vs the seeds


def components(group, queries, vec, pool, position_of, pop_vector):
    """Per user: candidates and the three rescaled signals (request, seeds, popularity)."""
    out = []
    for q, u in zip(queries, group):
        seen = {position_of[m] for m in u["seen"]}
        cand = np.array([p for p in pool if p not in seen])
        liked = [position_of[m] for m in u["liked"]]
        out.append({"user": u, "cand": cand,
                    "req": cs.rescale(vec[cand] @ q),
                    "seed": cs.rescale((vec[cand] @ vec[liked].T).mean(axis=1)),
                    "pop": cs.rescale(pop_vector[cand])})
    return out


def score(c, method, w=None, alpha=None):
    if method == "seeds":
        return c["seed"]
    if method == "request":
        return c["req"]
    if method == "request+seeds":
        return w * c["req"] + (1 - w) * c["seed"]
    personal = {"hybrid_seeds": c["seed"], "hybrid_request": c["req"],
                "hybrid_request+seeds": cs.rescale(w * c["req"] + (1 - w) * c["seed"])
                if method == "hybrid_request+seeds" else None}[method]
    return alpha * personal + (1 - alpha) * c["pop"]


def ndcgs(comps, movie_ids, method, w=None, alpha=None):
    out = []
    for c in comps:
        top = c["cand"][np.argsort(-score(c, method, w, alpha), kind="stable")[:bmt.K]]
        out.append(bmt.user_metrics(movie_ids[top].tolist(), c["user"]["relevant"], bmt.K))
    return out


def tune(comps, movie_ids, method):
    """Best (w, alpha) on validation users; ties keep the first (larger w, smaller alpha)."""
    grid_w = WEIGHTS if method in ("request+seeds", "hybrid_request+seeds") else [None]
    grid_a = cs.ALPHAS if method.startswith("hybrid") else [None]
    best = (-1.0, None, None)
    for w in grid_w:
        for a in grid_a:
            value = float(np.mean([m["ndcg"] for m in ndcgs(comps, movie_ids, method, w, a)]))
            if value > best[0]:
                best = (value, w, a)
    return best[1], best[2]


def evaluate(movies, train, pool_ids, val_users, test_users, requests_by_style):
    movie_ids = movies["movie_id"].to_numpy()
    position_of = {m: i for i, m in enumerate(movie_ids)}
    pool = np.array(sorted(position_of[m] for m in pool_ids))
    popular = cs.popularity_scores(train)
    pop_vector = cs.popularity_vector(popular, movies, position_of)
    vectors = movie_vectors(movies)
    methods = ["seeds", "request", "request+seeds", "hybrid_seeds", "hybrid_request", "hybrid_request+seeds"]

    rows, params = [], []
    for style, records in requests_by_style.items():
        for n in REQUEST_SEEDS:
            groups = {"val": cs.users_with_n_seeds(val_users, n), "test": cs.users_with_n_seeds(test_users, n)}
            queries = {k: embed_requests([records[(u["user_id"], n)]["request"] for u in g]) for k, g in groups.items()}
            for u in groups["test"]:
                rows.append({"style": style, "n": n, "representation": "-", "method": "popularity",
                             "user_id": u["user_id"],
                             **bmt.user_metrics(cs.recommend_popular(popular.index, u, bmt.K), u["relevant"], bmt.K)})
            for rep, vec in vectors.items():
                comps = {k: components(groups[k], queries[k], vec, pool, position_of, pop_vector) for k in groups}
                for method in tqdm(methods, desc=f"{style} N={n} {rep}"):
                    w, alpha = tune(comps["val"], movie_ids, method)
                    params.append({"style": style, "n": n, "representation": rep, "method": method, "w": w, "alpha": alpha})
                    for c, m in zip(comps["test"], ndcgs(comps["test"], movie_ids, method, w, alpha)):
                        rows.append({"style": style, "n": n, "representation": rep, "method": method,
                                     "user_id": c["user"]["user_id"], "w": w, "alpha": alpha, **m})
    results = pd.DataFrame(rows)
    results.to_csv(FOLDER / "per_user.csv", index=False)
    pd.DataFrame(params).to_csv(FOLDER / "tuned_parameters.csv", index=False)

    # Significance tests (Wilcoxon per user, Bonferroni over all tests)
    key = results["representation"] + "|" + results["method"]
    results = results.assign(key=key)
    tests = []
    for (style, n), part in results.groupby(["style", "n"]):
        nd = part.pivot(index="user_id", columns="key", values="ndcg")
        pairs = [("base|request", "v2_labels|request"), ("base|request", "v4_labels|request"),
                 ("base|request", "31b_P2|request"),
                 ("base|hybrid_seeds", "base|hybrid_request+seeds"),        # does the request add to the seeds?
                 ("base|hybrid_seeds", "base|hybrid_request"),              # request instead of seeds
                 ("-|popularity", "base|hybrid_request"),
                 ("base|hybrid_request+seeds", "v4_labels|hybrid_request+seeds"),
                 ("base|hybrid_request+seeds", "31b_P2|hybrid_request+seeds"),
                 ("base|hybrid_request+seeds", "v2_labels|hybrid_request+seeds")]
        for a, b in pairs:
            tests.append({"style": style, "n": n, **bmt.wilcoxon_row(f"{b} vs {a}", nd[a], nd[b])})
    if len(requests_by_style) > 1:
        for n in REQUEST_SEEDS:
            part = results[results["n"] == n]
            for k in ("base|request", "base|hybrid_request+seeds", "v4_labels|request"):
                piv = part[part["key"] == k].pivot(index="user_id", columns="style", values="ndcg")
                tests.append({"style": "r2 vs r1", "n": n, **bmt.wilcoxon_row(f"{k}: r2 vs r1", piv["r1"], piv["r2"])})
    tests = pd.DataFrame(tests)
    tests["p_bonferroni"] = (tests["p_value"] * len(tests)).clip(upper=1.0)
    tests.to_csv(FOLDER / "significance_tests.csv", index=False)

    summary = results.groupby(["style", "n", "representation", "method"])[["ndcg", "precision", "recall"]].mean()
    summary.round(4).to_csv(FOLDER / "summary.csv")
    pd.set_option("display.width", 220)
    print("\nNDCG@10 (test users)")
    print(results.pivot_table(index=["method", "representation"], columns=["style", "n"], values="ndcg").round(4).to_string())
    print("\nTests:")
    print(tests[["style", "n", "comparison", "mean_difference", "p_value", "p_bonferroni",
                 "users_better", "users_worse"]].round(4).to_string(index=False))
    print(f"\nSaved results to {FOLDER}")


if __name__ == "__main__":
    pilot = len(sys.argv) > 1 and sys.argv[1] == "pilot"
    style = sys.argv[2] if len(sys.argv) > 2 else "r1"
    FOLDER.mkdir(parents=True, exist_ok=True)
    movies = bmt.load_movies()
    movie_by_id = {m["movie_id"]: m for m in movies.to_dict("records")}
    ratings = bmt.prepare_ratings(movies)
    train, cold = cs.split_users(ratings)
    pool_ids = set(train["movie_id"])
    cold_users = cs.build_cold_user_data(cold, pool_ids, max(cs.SEED_COUNTS), cs.MIN_TEST_RELEVANT)
    val_users, test_users = cs.split_validation(cold_users, cs.VAL_FRACTION)

    if pilot:
        users = val_users[:20]
        cache = FOLDER / f"pilot_requests_{style}.jsonl"
        if cache.exists():
            cache.unlink()
        records = generate_requests(users, movie_by_id, cache, style)
        for user in users[:8]:
            for n in REQUEST_SEEDS:
                r = records[(user["user_id"], n)]
                print(f"\nuser {r['user_id']}, N={n}: " + "; ".join(movie_by_id[s]["title"] for s in r["seeds"]))
                print(f"  -> {r['request']}")
        rec = pd.DataFrame(records.values())
        print(f"\n{len(rec)} requests | mean {rec.words.mean():.1f} words | names a seed: {rec.names_a_seed.sum()} | "
              f"mean {rec.seconds.mean():.2f} s per request")

        # Quick sanity numbers on these 20 users (far too few for conclusions).
        movie_ids = movies["movie_id"].to_numpy()
        position_of = {m: i for i, m in enumerate(movie_ids)}
        pool = np.array(sorted(position_of[m] for m in pool_ids))
        vectors = movie_vectors(movies)
        for n in REQUEST_SEEDS:
            group = cs.users_with_n_seeds(users, n)
            queries = embed_requests([records[(u["user_id"], n)]["request"] for u in group])
            line = []
            for name, vec in vectors.items():
                scores = []
                for q, u in zip(queries, group):
                    seen = {position_of[m] for m in u["seen"]}
                    cand = np.array([p for p in pool if p not in seen])
                    top = cand[np.argsort(-(vec[cand] @ q), kind="stable")[:bmt.K]]
                    scores.append(bmt.user_metrics(movie_ids[top].tolist(), u["relevant"], bmt.K)["ndcg"])
                line.append(f"{name} {np.mean(scores):.3f}")
            print(f"N={n} request-only NDCG@10 on 20 pilot users: " + " | ".join(line))
        sys.exit(0)

    # ───────────── full run: requests for every validation and test user, both styles ─────────────
    styles = sys.argv[1:] or list(STYLES)
    requests_by_style = {}
    for style in styles:
        cache = FOLDER / f"requests_{style}.jsonl"
        requests_by_style[style] = generate_requests(val_users + test_users, movie_by_id, cache, style)
        rec = pd.DataFrame(requests_by_style[style].values())
        print(f"{style}: {len(rec)} requests | mean {rec.words.mean():.1f} words | "
              f"names a seed: {int(rec.names_a_seed.sum())}")

    evaluate(movies, train, pool_ids, val_users, test_users, requests_by_style)
