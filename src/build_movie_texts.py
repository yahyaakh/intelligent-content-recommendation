import json
import os
import time
from collections import Counter

import numpy as np
import pandas as pd

from pathlib import Path

import requests
import faiss
from sympy.codegen.fnodes import dimension

from enrich_catalog import cache_path, enrich_movies, load_records
from llm_enricher import PROMPT_VERSION, LLM_MovieEnricher

from recommenders.datasets.python_splitters import python_random_split
from scipy.stats import wilcoxon

from tqdm import tqdm

from recommenders.datasets.python_splitters import python_random_split

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

EMBED_PROVIDER = "gemini"                 # "ollama" or "gemini"

EMBEDDINGS_FOLDER = Path("output") / "embeddings"
EMBED_MODEL = "gemini-embedding-001"      # then "gemini-embedding-2"
DOC_PREFIX = ""                           # not used for Gemini



# EMBED_PROVIDER = "ollama"   # "ollama" or "gemini"
#
# EMBEDDINGS_FOLDER = Path("output") / "embeddings"
# EMBED_MODEL = "embeddinggemma"
# DOC_PREFIX = "title: none | text: "
# Filename-safe model name: Windows does not allow ":" (e.g. "qwen3-embedding:4b").
SAFE_MODEL = EMBED_MODEL.replace(":", "_").replace("/", "_")
K = 10
TRAIN_RATIO = 0.75
SEED = 42
EMBEDDINGS_FOLDER = Path("output") / "embeddings"
# One results folder per embedding model, so each run keeps its own results.
RESULTS_FOLDER = Path("output") / "evaluation" / SAFE_MODEL


def split_field(value):
    if pd.isna(value) or str(value).strip() == "":
        return []
    parts = value.split("|")
    return [part.strip() for part in parts if part.strip()]

def load_movies():
    movies_df = pd.read_csv("data/data/processed/movies.csv")
    enriched_df = pd.read_csv(f"output/enrichment/movies_enriched_{PROMPT_VERSION}.csv")
    return movies_df.merge(enriched_df, on="movie_id", how="left", validate="one_to_one")


def build_base_text(movie):
    parts = [f"{movie['title']} ({int(movie['year'])})."]

    genres = split_field(movie["genres"])
    if genres:
        parts.append(f"Genres: {', '.join(genres)}.")

    director = movie["director"]
    if not pd.isna(director) and str(director).strip():
        parts.append(f"Directed by {str(director).strip()}.")

    cast = split_field(movie["cast"])
    if cast:
        parts.append(f"Starring {', '.join(cast)}.")

    keywords = split_field(movie["keywords"])
    if keywords:
        parts.append(f"Keywords: {', '.join(keywords)}.")

    overview = movie["overview"]
    if not pd.isna(overview) and str(overview).strip():
        parts.append(f"Plot: {str(overview).strip()}")

    return " ".join(parts)



# print(build_base_text(movies_df.loc[0]))


def build_enriched_text(base_text, movie):
    extra = []

    themes = split_field(movie["llm_themes"])

    if themes:
        sentence = f"Themes: {', '.join(themes)}."
        extra.append(sentence)

    mood = split_field(movie["llm_mood"])

    if mood:
        sentence = f"Mood: {', '.join(mood)}."
        extra.append(sentence)

    if not extra:
        return base_text

    return base_text + " " + " ".join(extra)


def build_texts(merged_df , out_path=None):

    if out_path is None:
        out_path = Path("output") / "enrichment" / f"movie_texts_{PROMPT_VERSION}.csv"

    not_enriched = merged_df[merged_df["llm_themes"].isna()]
    assert not_enriched.empty, f"{len(not_enriched)} movies have no enrichment"

    rows = []
    for _, movie in merged_df.iterrows():
        base = build_base_text(movie)
        enriched = build_enriched_text(base, movie)
        rows.append({
            "movie_id": int(movie["movie_id"]),
            "title": movie["title"],
            "text_base": base,
            "text_enriched": enriched,
        })

    texts = pd.DataFrame(rows)

    # Checks: the only difference between the two texts must be the added part.
    assert len(texts) == len(merged_df), "Some movies were lost"
    assert all(e.startswith(b) for b, e in zip(texts["text_base"], texts["text_enriched"])), \
        "An enriched text does not start with its base text"
    assert all(len(e) > len(b) for b, e in zip(texts["text_base"], texts["text_enriched"])), \
        "An enriched text adds nothing to its base text"

    # Formatting mistakes: a missing value written as "nan", or an unsplit "|" list.
    for column in ["text_base", "text_enriched"]:
        assert not texts[column].str.contains(r"\bnan\b", regex=True).any(), f"'nan' found in {column}"
        without_title = [text[len(title):] for title, text in zip(texts["title"], texts[column])]
        assert not any("|" in text for text in without_title), f"'|' found in {column}"

    texts.to_csv(out_path, index=False)
    print(f"Saved {len(texts)} rows to {out_path}")

    return texts


def embed_texts(texts, model="embeddinggemma", prefix="title: none | text: ", batch_size=32):
    all_vectors = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        input = [prefix + text for text in batch ]

        res = requests.post(
            "http://127.0.0.1:11434/api/embed",
            json={
                "model": model,
                "input": input,
            },
            timeout=120,
        )
        res.raise_for_status()
        vectors = res.json()["embeddings"]

        assert len(vectors) == len(input)
        all_vectors.extend(vectors)

    matrix = np.array(all_vectors, dtype=np.float32)

    assert np.isfinite(matrix).all(), "Embeddings contain NaN or infinity"
    matrix = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)

    return matrix


def build_index(matrix) :
    dimension = matrix.shape[1]

    index = faiss.IndexFlatIP(dimension)
    index.add(matrix)
    assert index.ntotal == len(matrix)

    return index



def save_index(index, movie_ids, name, info , folder=None):
    if folder is None:
        folder = Path("output") / "embeddings"

    folder.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(folder / f"{name}.faiss"))

    np.save(folder / f"{name}_ids.npy", np.asarray(movie_ids, dtype=np.int64), allow_pickle=False)

    with open(folder / f"{name}.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    print(f"Saved {index.ntotal} vectors to {folder / name}.*")



# ── Step 10: evaluation ───────────────────────
# load_ratings, split_ratings, build_user_data,
# theme_and_mood_tokens, content_similarity, embedding_similarity,
# recommend, dcg_at_k / idcg_at_k / ndcg_at_k, user_metrics,
# evaluate_method, compare

def prepare_ratings(movies):
    ratings = pd.read_csv(
        "data/data/processed_30k/ratings.csv",
        usecols=["user_id", "movie_id", "rating", "timestamp"],
    )

    ratings = ratings[ratings["movie_id"].isin(movies["movie_id"])]

    print(f"Ratings: {len(ratings):,} | users: {ratings['user_id'].nunique():,} | "
          f"movies with ratings: {ratings['movie_id'].nunique():,}")
    return ratings

# TRAIN_RATIO = 0.75: how much data goes to training
# It means 75% of the ratings go to training and 25% to testing.

# “Ratings were randomly split into 75% training and 25% test data (random seed 42),
# following the same procedure as the baseline evaluation.”

TRAIN_RATIO = 0.75
SEED = 42


def split_ratings(ratings):
    train, test = python_random_split(ratings, ratio=TRAIN_RATIO, seed=SEED)

    # Keep only test rows whose user and movie were seen in training.
    train_users = set(train["user_id"])
    train_movies = set(train["movie_id"])

    before = len(test)
    test = test[test["user_id"].isin(train_users) & test["movie_id"].isin(train_movies)]

    print(f"Train: {len(train):,} | Test: {len(test):,} "
          f"(removed {before - len(test):,}) | test users: {test['user_id'].nunique():,}")

    return train, test


def build_user_data(train, test):
    user_means = train.groupby("user_id")["rating"].mean()
    train_by_user = {user_id: rows for user_id, rows in train.groupby("user_id")}

    users = []
    skipped = 0

    for user_id, user_test in test.groupby("user_id"):
        user_train = train_by_user[user_id]
        mean = user_means[user_id]

        seen = set(user_train["movie_id"])
        liked = list(user_train.loc[user_train["rating"] >= mean, "movie_id"])

        relevant = set(user_test.loc[user_test["rating"] >= mean, "movie_id"])
        relevant -= seen

        if not liked or not relevant:
            skipped += 1
            continue

        users.append({
            "user_id": user_id,
            "seen": seen,
            "liked": liked,
            "relevant": relevant,
        })

    print(f"Users evaluated: {len(users):,} (skipped {skipped:,}) | "
          f"avg liked: {sum(len(u['liked']) for u in users) / len(users):.0f} | "
          f"avg relevant: {sum(len(u['relevant']) for u in users) / len(users):.0f}")
    return users


def popularity_ranking(train):
    """Movies sorted by number of positive training ratings (rating >= that user's own mean).

    The single popularity definition used by every experiment. Ties are broken by
    movie_id so the order is reproducible.
    """
    user_mean = train.groupby("user_id")["rating"].transform("mean")
    liked = train[train["rating"] >= user_mean]
    counts = liked.groupby("movie_id").size().rename("count").reset_index()
    counts = counts.sort_values(["count", "movie_id"], ascending=[False, True])
    return counts.set_index("movie_id")["count"]


def evaluate_popularity(ranking, users, k):
    """Popularity baseline: the same ranked list for everyone, minus each user's seen movies."""
    rows = []
    for user in users:
        recommended = [m for m in ranking.index if m not in user["seen"]][:k]
        rows.append({"method": "popularity", "scoring": "-", "user_id": user["user_id"],
                     **user_metrics(recommended, user["relevant"], k)})
    return pd.DataFrame(rows)


def theme_and_mood_tokens(movie):
    themes = ["theme_" + label.replace(" ", "_").replace("-", "_")
              for label in split_field(movie["llm_themes"])]
    moods = ["mood_" + label.replace(" ", "_").replace("-", "_")
             for label in split_field(movie["llm_mood"])]
    return " ".join(themes + moods)


def content_similarity(movies, with_enrichment):
    meta = movies["metadata_features"].fillna("")
    if with_enrichment:
        meta = meta + " " + movies.apply(theme_and_mood_tokens, axis=1)

    meta_matrix = CountVectorizer().fit_transform(meta)
    text_matrix = TfidfVectorizer(stop_words="english").fit_transform(movies["text_features"].fillna(""))

    similarity = 0.75 * cosine_similarity(meta_matrix) + 0.25 * cosine_similarity(text_matrix)
    similarity = similarity.astype(np.float32)
    np.fill_diagonal(similarity, 0)
    return similarity


def embedding_similarity(name, movies, folder=None):
    if folder is None:
        folder = EMBEDDINGS_FOLDER
    index = faiss.read_index(str(folder / f"{name}.faiss"))
    index_ids = np.load(folder / f"{name}_ids.npy")
    vectors = index.reconstruct_n(0, index.ntotal)

    # Reorder the vectors so row i belongs to the i-th movie in `movies`.
    row_of = {movie_id: row for row, movie_id in enumerate(index_ids)}
    missing = [movie_id for movie_id in movies["movie_id"] if movie_id not in row_of]
    assert not missing, f"{len(missing)} movies have no vector in {name}"
    vectors = vectors[[row_of[movie_id] for movie_id in movies["movie_id"]]]
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    similarity = (vectors @ vectors.T).astype(np.float32)
    np.fill_diagonal(similarity, 0)
    return similarity

def recommend(similarity, user, pool_positions, position_of, movie_ids, k, scoring="mean"):
    seen = {position_of[movie_id] for movie_id in user["seen"]}
    candidates = np.array([p for p in pool_positions if p not in seen])
    liked = [position_of[movie_id] for movie_id in user["liked"]]

    block = similarity[np.ix_(liked, candidates)]   # rows: liked movies, columns: candidates
    if scoring == "max":
        scores = block.max(axis=0)       # similarity to the closest liked movie
    elif scoring == "mean":
        scores = block.mean(axis=0)      # average similarity to all liked movies
    else:
        raise ValueError(f"Unknown scoring: {scoring}")

    top = np.argsort(-scores, kind="stable")[:k]
    return [movie_ids[p] for p in candidates[top]]


def dcg_at_k(recommended, relevant, k):
    return sum((1 if movie_id in relevant else 0) / np.log2(i + 2)
               for i, movie_id in enumerate(recommended[:k]))


def ndcg_at_k(recommended, relevant, k):
    ideal = sum(1 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg_at_k(recommended, relevant, k) / ideal if ideal > 0 else 0.0


def user_metrics(recommended, relevant, k):
    hits = len(set(recommended[:k]) & relevant)
    return {
        "ndcg": ndcg_at_k(recommended, relevant, k),
        "precision": hits / k,
        "recall": hits / len(relevant),
    }


def evaluate_method(name, similarity, users, pool_positions, position_of, movie_ids, k, scoring="mean"):
    rows = []
    for user in tqdm(users, desc=f"{name} ({scoring})"):
        recommended = recommend(similarity, user, pool_positions, position_of, movie_ids, k, scoring)
        rows.append({"method": name, "scoring": scoring, "user_id": user["user_id"],
                     **user_metrics(recommended, user["relevant"], k)})
    return pd.DataFrame(rows)


def run_texts_and_embeddings():
    merged_df = load_movies()
    texts = build_texts(merged_df)

    movie_ids = texts["movie_id"].to_numpy()
    titles = dict(zip(texts["movie_id"], texts["title"]))

    # Step 9: embed both versions the same way (same model, prefix, normalisation, order)
    if EMBED_PROVIDER == "gemini":
        base_matrix = embed_texts_gemini(texts["text_base"].tolist(), model=EMBED_MODEL,
                                         cache_name=f"{SAFE_MODEL}_base_{PROMPT_VERSION}")
        enriched_matrix = embed_texts_gemini(texts["text_enriched"].tolist(), model=EMBED_MODEL,
                                             cache_name=f"{SAFE_MODEL}_enriched_{PROMPT_VERSION}")
    else:
        base_matrix = embed_texts(texts["text_base"].tolist(), model=EMBED_MODEL, prefix=DOC_PREFIX)
        enriched_matrix = embed_texts(texts["text_enriched"].tolist(), model=EMBED_MODEL, prefix=DOC_PREFIX)

    assert base_matrix.shape == enriched_matrix.shape == (len(texts), base_matrix.shape[1])
    assert np.allclose(np.linalg.norm(base_matrix, axis=1), 1, atol=1e-3)
    assert np.allclose(np.linalg.norm(enriched_matrix, axis=1), 1, atol=1e-3)

    # How much did the enrichment move each movie? (cosine between its two vectors)
    shift = (base_matrix * enriched_matrix).sum(axis=1)
    print(f"Base vs enriched similarity: mean {shift.mean():.3f}, "
          f"min {shift.min():.3f}, max {shift.max():.3f}")

    # Build and save both indexes
    for kind, matrix in [("base", base_matrix), ("enriched", enriched_matrix)]:
        info = {
            "model": EMBED_MODEL,
            "prefix": DOC_PREFIX,
            "text_column": f"text_{kind}",
            "prompt_version": PROMPT_VERSION,
            "dimension": int(matrix.shape[1]),
            "movies": int(matrix.shape[0]),
            "normalized": True,
            "index_type": "IndexFlatIP",
        }
        save_index(build_index(matrix), movie_ids, f"movies_{kind}_{SAFE_MODEL}_{PROMPT_VERSION}", info)

    # Sanity check: The Matrix's 10 nearest movies in both indexes
    position = int(np.where(movie_ids == 603)[0][0])
    for kind, matrix in [("base", base_matrix), ("enriched", enriched_matrix)]:
        scores, positions = build_index(matrix).search(matrix[position].reshape(1, -1), 11)
        neighbours = [f"{titles[movie_ids[p]]} ({s:.2f})"
                      for s, p in zip(scores[0], positions[0]) if p != position][:10]
        print(f"\nThe Matrix, {kind}:")
        for line in neighbours:
            print("  ", line)

def run_evaluation():
    movies = load_movies()
    ratings = prepare_ratings(movies)
    train, test = split_ratings(ratings)
    users = build_user_data(train, test)

    tables = {
        "A_content_base": content_similarity(movies, with_enrichment=False),
        "B_content_enriched": content_similarity(movies, with_enrichment=True),
        "C_embedding_base": embedding_similarity(f"movies_base_{SAFE_MODEL}_{PROMPT_VERSION}", movies),
        "D_embedding_enriched": embedding_similarity(f"movies_enriched_{SAFE_MODEL}_{PROMPT_VERSION}", movies),
    }

    movie_ids = movies["movie_id"].to_numpy()
    position_of = {movie_id: i for i, movie_id in enumerate(movie_ids)}
    pool_positions = np.array(sorted(position_of[m] for m in set(train["movie_id"])))

    results = pd.concat([
        evaluate_method(name, table, users, pool_positions, position_of, movie_ids, K)
        for name, table in tables.items()
    ])

    compare(results)

def compare(results):
    RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)

    summary = results.groupby("method")[["ndcg", "precision", "recall"]].mean()
    print(summary.round(4))

    ndcg = results.pivot(index="user_id", columns="method", values="ndcg")
    tests = []
    for a, b in [("A_content_base", "B_content_enriched"),
                 ("C_embedding_base", "D_embedding_enriched"),
                 ("A_content_base", "C_embedding_base")]:
        diff = ndcg[b] - ndcg[a]
        test = {
            "comparison": f"{b} vs {a}",
            "mean_difference": diff.mean(),
            "p_value": wilcoxon(ndcg[a], ndcg[b]).pvalue,
            "users_better": int((diff > 0).sum()),
            "users_worse": int((diff < 0).sum()),
            "users_same": int((diff == 0).sum()),
        }
        tests.append(test)
        print(f"{test['comparison']}: diff {test['mean_difference']:+.4f}, p={test['p_value']:.2g}, "
              f"better {test['users_better']}, worse {test['users_worse']}, same {test['users_same']}")

    results.to_csv(RESULTS_FOLDER / "per_user_results.csv", index=False)
    summary.to_csv(RESULTS_FOLDER / "summary.csv")
    pd.DataFrame(tests).to_csv(RESULTS_FOLDER / "significance_tests.csv", index=False)

    settings = {"k": K, "train_ratio": TRAIN_RATIO, "seed": SEED, "prompt_version": PROMPT_VERSION,
                "embed_model": EMBED_MODEL, "doc_prefix": DOC_PREFIX,
                "scoring": "mean similarity to liked movies", "users": int(ndcg.shape[0])}
    with open(RESULTS_FOLDER / "settings.json", "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2)



GEMINI_TASK_TYPE = "SEMANTIC_SIMILARITY"
def embed_texts_gemini(texts, model, cache_name, task_type=GEMINI_TASK_TYPE,
                       batch_size=20, pause_seconds=15, max_retries=6):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:batchEmbedContents"
    headers = {"x-goog-api-key": os.environ["GEMINI_API_KEY"]}

    # Each finished batch is saved here, so a stopped run can continue later.
    cache_folder = EMBEDDINGS_FOLDER / "gemini_cache" / cache_name
    cache_folder.mkdir(parents=True, exist_ok=True)

    for start in tqdm(range(0, len(texts), batch_size), desc=f"Embedding {cache_name}"):
        batch_file = cache_folder / f"batch_{start:05d}.npy"
        if batch_file.exists():
            continue

        batch = texts[start:start + batch_size]
        body = {"requests": [
            {"model": f"models/{model}", "content": {"parts": [{"text": text}]}, "taskType": task_type}
            for text in batch
        ]}

        for attempt in range(1, max_retries + 1):
            res = requests.post(url, json=body, headers=headers, timeout=120)
            # Free tier: wait and retry when rate limited (429) or overloaded (5xx).
            if res.status_code == 429 or res.status_code >= 500:
                time.sleep(30 * attempt)
                continue
            res.raise_for_status()
            break
        else:
            raise RuntimeError(f"Gemini limit reached at text {start}. Progress is saved; "
                               f"run again later to continue. Last error: {res.text[:200]}")

        vectors = np.array([item["values"] for item in res.json()["embeddings"]], dtype=np.float32)
        assert len(vectors) == len(batch), "Gemini returned a different number of vectors"
        np.save(batch_file, vectors)
        time.sleep(pause_seconds)

    # All batches done: load them in order and combine.
    matrix = np.concatenate([np.load(cache_folder / f"batch_{start:05d}.npy")
                             for start in range(0, len(texts), batch_size)])
    assert len(matrix) == len(texts), "Some batches are missing"
    assert np.isfinite(matrix).all(), "Embeddings contain NaN or infinity"
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)



def strip_code_fence(text):
    text = text.strip()
    # What it does: Removes all leading and trailing whitespace characters
    # (such as spaces, tabs, and newline characters \n) from both ends of the string text.
    # Check if the string starts with 3 backticks
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""  # drop the "```json" line
        if text.endswith("```"):
            text = text[:-3]

    return text.strip()


class FenceStrippingEnricher(LLM_MovieEnricher):
    def call_llm(self, user_prompt):
        result = super().call_llm(user_prompt)
        if result["status"] == "ok":
            # Raw line breaks are only allowed between JSON elements, never inside a
            # text value, so removing them cannot change a valid answer. It repairs
            # answers like "isolatio\n\nn and loneliness" (seen with gpt-oss).
            result["content"] = strip_code_fence(result["content"]).replace("\n", "").replace("\r", "")
        return result


def run_enrichment_experiment(enrich_model, provider="ollama", reasoning_effort=None,
                              prompt_version=PROMPT_VERSION):
    safe_name = enrich_model.replace(":", "_").replace("/", "_") # "gemma4_31b-cloud"
    # v2 runs keep their original folder names; other prompt versions get their own folder,
    # so a new run can never overwrite an earlier one.
    if prompt_version != PROMPT_VERSION:
        safe_name = f"{safe_name}_{prompt_version}"         # "gemma4_e4b_v3"
    folder = Path("output") / "experiments" / safe_name
    folder.mkdir(parents=True, exist_ok=True)
    movies_df = pd.read_csv("data/data/processed/movies.csv")

    # reasoning_effort: e.g. "low" for gpt-oss, which ignores think=False.
    enricher = FenceStrippingEnricher(movies_df, model_name=enrich_model, provider=provider,
                                      reasoning_effort=reasoning_effort, max_retries=4,
                                      prompt_version=prompt_version)

    cache = cache_path(enrich_model, prompt_version)  # from enrich_catalog
    # → output/enrichment/gemma4_31b-cloud_v2.jsonl (its own file)

    enrich_movies(enricher, movies_df, cache)

    records = load_records(cache)

    rows = []
    for record in records.values():
        if record["status"] != "ok":
            continue

        intensity = record["content_intensity"]
        rows.append({
            "movie_id": record["movie_id"],
            "llm_themes": "|".join(record["themes"]),
            "llm_mood": "|".join(record["mood"]),
            "llm_audience": record["audience"],
            "intensity_violence": intensity["violence"],
            "intensity_sexual_content": intensity["sexual_content"],
            "intensity_language": intensity["language"],
            "intensity_substance_use": intensity["substance_use"],
            "intensity_frightening_scenes": intensity["frightening_scenes"],
            "enrich_model": record["model"],
            "prompt_version": record["prompt_version"],
            "repaired": record.get("repaired", False),
        })

    enriched_df = pd.DataFrame(rows)

    # One row per movie, and every catalog movie enriched.
    assert enriched_df["movie_id"].is_unique, "Duplicate movie ids in the enrichment"
    missing = set(movies_df["movie_id"]) - set(enriched_df["movie_id"])
    assert not missing, f"{len(missing)} movies are not enriched yet: run the enrichment again"

    # Summary: movies, repairs, most common themes.
    theme_counts = Counter(theme for themes in enriched_df["llm_themes"] for theme in themes.split("|"))
    print(f"Enriched movies: {len(enriched_df)} | repaired: {int(enriched_df['repaired'].sum())}")
    print("Top 10 themes:")
    for theme, count in theme_counts.most_common(10):
        print(f"   {theme}: {count} ({count / len(enriched_df):.0%})")

    enriched_df.to_csv(folder / "movies_enriched.csv", index=False)

    merged=  movies_df.merge(enriched_df, on="movie_id", how="left", validate="one_to_one")

    texts = build_texts(merged, out_path=folder / "movie_texts.csv")

    base_name = f"movies_base_embeddinggemma_{PROMPT_VERSION}"

    with open(EMBEDDINGS_FOLDER / f"{base_name}.json", encoding="utf-8") as f:
        base_info = json.load(f)

    embed_model = base_info["model"]  # "embeddinggemma"
    embed_prefix = base_info["prefix"]  # "title: none | text: "

    # ── B: reuse the saved index if this step already ran ──────────────
    index_name = "movies_enriched_index"
    index_file = folder / f"{index_name}.faiss"

    if index_file.exists():
        index = faiss.read_index(str(index_file))
        enriched_matrix = index.reconstruct_n(0, index.ntotal)
        saved_ids = np.load(folder / f"{index_name}_ids.npy")
        assert (saved_ids == texts["movie_id"].to_numpy()).all(), "Saved index is in a different movie order"
        print("Reusing saved enriched embeddings")
    else:
        enriched_matrix = embed_texts(texts["text_enriched"].tolist(), model=embed_model, prefix=embed_prefix)
        info = {
            "model": embed_model,
            "prefix": embed_prefix,
            "text_column": "text_enriched",
            "enrich_model": enrich_model,
            "prompt_version": prompt_version,
            "dimension": int(enriched_matrix.shape[1]),
            "movies": int(enriched_matrix.shape[0]),
            "normalized": True,
            "index_type": "IndexFlatIP",
        }
        save_index(build_index(enriched_matrix), texts["movie_id"].to_numpy(), index_name, info, folder=folder)

    # ── C: checks ──────────────────────────────────────────────────────
    assert enriched_matrix.shape[0] == len(texts), "One vector per movie expected"
    assert np.allclose(np.linalg.norm(enriched_matrix, axis=1), 1, atol=1e-3), "Vectors are not normalised"

    # ── D: how much did the new labels move each movie? ────────────────
    base_index = faiss.read_index(str(EMBEDDINGS_FOLDER / f"{base_name}.faiss"))
    base_ids = np.load(EMBEDDINGS_FOLDER / f"{base_name}_ids.npy")
    base_vectors = base_index.reconstruct_n(0, base_index.ntotal)

    row_of = {movie_id: row for row, movie_id in enumerate(base_ids)}
    base_matrix = base_vectors[[row_of[movie_id] for movie_id in texts["movie_id"]]]

    shift = (base_matrix * enriched_matrix).sum(axis=1)
    print(f"Base vs enriched similarity ({enrich_model}): mean {shift.mean():.3f}, "
          f"min {shift.min():.3f}, max {shift.max():.3f}")

    # ── Step 5: evaluate on the same users, split and metrics ──────────
    movies = load_movies()                      # catalog + gemma4:e4b labels (llm_themes, llm_mood)
    ratings = prepare_ratings(movies)
    train, test = split_ratings(ratings)
    users = build_user_data(train, test)

    movie_ids = movies["movie_id"].to_numpy()
    position_of = {movie_id: i for i, movie_id in enumerate(movie_ids)}
    pool_positions = np.array(sorted(position_of[m] for m in set(train["movie_id"])))

    # New enriched vectors, reordered from `texts` order to `movies` order.
    text_row = {movie_id: row for row, movie_id in enumerate(texts["movie_id"])}
    new_vectors = enriched_matrix[[text_row[movie_id] for movie_id in movie_ids]]
    new_similarity = (new_vectors @ new_vectors.T).astype(np.float32)
    np.fill_diagonal(new_similarity, 0)

    # Same catalog, but with the new model's themes and mood (for content-based).
    movies_new_labels = movies.drop(columns=["llm_themes", "llm_mood"]).merge(
        enriched_df[["movie_id", "llm_themes", "llm_mood"]],
        on="movie_id", how="left", validate="one_to_one",
    )

    tables = {
        "A_content_base": content_similarity(movies, with_enrichment=False),
        "B_content_e4b": content_similarity(movies, with_enrichment=True),
        "B_content_new": content_similarity(movies_new_labels, with_enrichment=True),
        "C_embedding_base": embedding_similarity(base_name, movies),
        "D_embedding_e4b": embedding_similarity(f"movies_enriched_embeddinggemma_{PROMPT_VERSION}", movies),
        "D_embedding_new": new_similarity,
    }

    results = pd.concat([
        evaluate_method(name, table, users, pool_positions, position_of, movie_ids, K)
        for name, table in tables.items()
    ])

    summary = results.groupby("method")[["ndcg", "precision", "recall"]].mean()
    print(f"\nResults with {enrich_model} labels:")
    print(summary.round(4))

    ndcg = results.pivot(index="user_id", columns="method", values="ndcg")
    tests = []
    for a, b in [("D_embedding_e4b", "D_embedding_new"),
                 ("C_embedding_base", "D_embedding_new"),
                 ("B_content_e4b", "B_content_new"),
                 ("A_content_base", "B_content_new")]:
        diff = ndcg[b] - ndcg[a]
        test = {
            "comparison": f"{b} vs {a}",
            "mean_difference": diff.mean(),
            "p_value": wilcoxon(ndcg[a], ndcg[b]).pvalue,
            "users_better": int((diff > 0).sum()),
            "users_worse": int((diff < 0).sum()),
            "users_same": int((diff == 0).sum()),
        }
        tests.append(test)
        print(f"{test['comparison']}: diff {test['mean_difference']:+.4f}, p={test['p_value']:.2g}, "
              f"better {test['users_better']}, worse {test['users_worse']}")

    results.to_csv(folder / "per_user_results.csv", index=False)
    summary.to_csv(folder / "summary.csv")
    pd.DataFrame(tests).to_csv(folder / "significance_tests.csv", index=False)

    settings = {
        "enrich_model": enrich_model, "embed_model": embed_model, "embed_prefix": embed_prefix,
        "prompt_version": prompt_version, "k": K, "train_ratio": TRAIN_RATIO, "seed": SEED,
        "scoring": "mean similarity to liked movies", "users": int(ndcg.shape[0]),
        "repaired_labels": int(enriched_df["repaired"].sum()),
        "shift_mean": float(shift.mean()),
    }
    with open(folder / "settings.json", "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2)
    print(f"\nSaved results to {folder}")






EMBEDDING_MODELS = ["embeddinggemma", "embeddinggemma-2", "bge-m3",
                    "qwen3-embedding:0.6b", "qwen3-embedding:4b", "qwen3-embedding:8b"]
LABEL_EXPERIMENTS = {"31b": "gemma4_31b-cloud", "gptoss": "gpt-oss_120b-cloud"}
EXPERIMENTS_FOLDER = Path("output") / "experiments"


def with_labels_from(movies, labels_csv):
    """The catalog with themes and mood replaced by another model's labels."""
    new_labels = pd.read_csv(labels_csv)[["movie_id", "llm_themes", "llm_mood"]]
    return movies.drop(columns=["llm_themes", "llm_mood"]).merge(
        new_labels, on="movie_id", how="left", validate="one_to_one")


def wilcoxon_row(name, a_scores, b_scores):
    diff = b_scores - a_scores
    try:
        p_value = wilcoxon(a_scores, b_scores).pvalue
    except ValueError:              # all differences are zero
        p_value = 1.0
    return {"comparison": name, "mean_difference": diff.mean(), "p_value": p_value,
            "users_better": int((diff > 0).sum()), "users_worse": int((diff < 0).sum()),
            "users_same": int((diff == 0).sum())}


def run_scoring_comparison():
    folder = EXPERIMENTS_FOLDER / "scoring_comparison"
    folder.mkdir(parents=True, exist_ok=True)

    # Same users, split and pool as every other evaluation.
    movies = load_movies()
    ratings = prepare_ratings(movies)
    train, test = split_ratings(ratings)
    users = build_user_data(train, test)
    movie_ids = movies["movie_id"].to_numpy()
    position_of = {movie_id: i for i, movie_id in enumerate(movie_ids)}
    pool_positions = np.array(sorted(position_of[m] for m in set(train["movie_id"])))

    # Every similarity table, built once.
    tables = {
        "A_content_base": content_similarity(movies, with_enrichment=False),
        "B_content_e4b": content_similarity(movies, with_enrichment=True),
    }
    for short, name in LABEL_EXPERIMENTS.items():
        labelled = with_labels_from(movies, EXPERIMENTS_FOLDER / name / "movies_enriched.csv")
        tables[f"B_content_{short}"] = content_similarity(labelled, with_enrichment=True)

    for model in EMBEDDING_MODELS:
        safe = model.replace(":", "_").replace("/", "_")
        tables[f"C_{safe}"] = embedding_similarity(f"movies_base_{safe}_{PROMPT_VERSION}", movies)
        tables[f"D_{safe}_e4b"] = embedding_similarity(f"movies_enriched_{safe}_{PROMPT_VERSION}", movies)

    for short, name in LABEL_EXPERIMENTS.items():
        tables[f"D_embeddinggemma_{short}"] = embedding_similarity(
            "movies_enriched_index", movies, folder=EXPERIMENTS_FOLDER / name)

    tables["old_text_embeddinggemma"] = embedding_similarity("movie_embeddinggemma", movies, folder=Path("output/embeddings"))

    # Evaluate every table with both scorings.
    results = pd.concat([
        evaluate_method(name, table, users, pool_positions, position_of, movie_ids, K, scoring)
        for scoring in ("mean", "max")
        for name, table in tables.items()
    ])
    results.to_csv(folder / "per_user_results.csv", index=False)

    # Popularity reference, same users and pool (shared definition: positive ratings).
    popularity = evaluate_popularity(popularity_ranking(train), users, K)
    popularity.to_csv(folder / "popularity_per_user.csv", index=False)
    popularity_summary = popularity[["ndcg", "precision", "recall"]].mean()
    print("Popularity: " + " | ".join(f"{m} {v:.4f}" for m, v in popularity_summary.items()))

    # Summary: NDCG, precision and recall per case and scoring.
    summary = results.groupby(["method", "scoring"])[["ndcg", "precision", "recall"]].mean().unstack("scoring")
    summary.to_csv(folder / "summary.csv")
    ndcg_table = results.pivot_table(index="method", columns="scoring", values="ndcg", aggfunc="mean")
    ndcg_table["max_minus_mean"] = ndcg_table["max"] - ndcg_table["mean"]
    print(ndcg_table.sort_values("max", ascending=False).round(4).to_string())

    # Tests: max vs mean for every case, and enrichment under both scorings.
    tests = []
    by_user = {s: results[results["scoring"] == s].pivot(index="user_id", columns="method", values="ndcg")
               for s in ("mean", "max")}
    for name in tables:
        tests.append(wilcoxon_row(f"{name}: max vs mean", by_user["mean"][name], by_user["max"][name]))

    pairs = [("A_content_base", f"B_content_{label}") for label in ("e4b", "31b", "gptoss")]
    for model in EMBEDDING_MODELS:
        safe = model.replace(":", "_").replace("/", "_")
        pairs.append((f"C_{safe}", f"D_{safe}_e4b"))
    pairs += [("C_embeddinggemma", "D_embeddinggemma_31b"), ("C_embeddinggemma", "D_embeddinggemma_gptoss")]

    for scoring in ("mean", "max"):
        for a, b in pairs:
            tests.append(wilcoxon_row(f"[{scoring}] {b} vs {a}", by_user[scoring][a], by_user[scoring][b]))

    # Best embedding cases against popularity (max scoring).
    pop_by_user = popularity.set_index("user_id")["ndcg"]
    for name in ("C_embeddinggemma", "D_embeddinggemma_e4b"):
        tests.append(wilcoxon_row(f"[max] {name} vs popularity",
                                  pop_by_user, by_user["max"][name].reindex(pop_by_user.index)))

    tests = pd.DataFrame(tests)
    tests.to_csv(folder / "significance_tests.csv", index=False)
    print("\nEnrichment effect under each scoring:")
    print(tests[tests["comparison"].str.startswith("[")][["comparison", "mean_difference", "p_value"]]
          .round(4).to_string(index=False))

    settings = {"ratings": "processed_30k/ratings.csv, original ML-25M half stars (0.5-5)",
                "popularity": "positive training ratings per movie (rating >= user's mean)",
                "popularity_ndcg": float(popularity_summary["ndcg"]),
                "k": K, "train_ratio": TRAIN_RATIO, "seed": SEED, "prompt_version": PROMPT_VERSION,
                "scorings": ["mean", "max"], "embedding_models": EMBEDDING_MODELS,
                "label_experiments": LABEL_EXPERIMENTS, "users": int(len(users)), "cases": list(tables)}
    with open(folder / "settings.json", "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2)
    print(f"\nSaved results to {folder}")













if __name__ == "__main__":
    RUN_EMBEDDINGS = False
    RUN_EVALUATION = False
    RUN_EXPERIMENT = False
    RUN_SCORING = True

    if RUN_EMBEDDINGS:
        run_texts_and_embeddings()
    if RUN_EVALUATION:
        run_evaluation()
    if RUN_EXPERIMENT:
        # Finished: run_enrichment_experiment("gemma4:31b-cloud")
        run_enrichment_experiment("gpt-oss:120b-cloud", reasoning_effort="low")
    if RUN_SCORING:
        run_scoring_comparison()



