import operator
from collections import Counter
from unittest import skip

import pandas as pd

import faiss
import requests
import numpy as np
from recommenders.datasets.python_splitters import python_random_split
from torch.nn.functional import embedding
import os
from dotenv import load_dotenv



# Evaluation metrics
def dcg_at_k(recommended_items, relevant_set, k):
    dcg = 0.0
    for i, item in enumerate(recommended_items[:k]):
        rel = 1 if item in relevant_set else 0
        if i == 0:
            dcg += rel
        else:
            dcg += rel / np.log2(i + 1 + 1)
    return dcg

def idcg_at_k(relevant_set, k):
    rel_count = min(len(relevant_set), k)
    idcg = 0.0
    for i in range(rel_count):
        if i == 0:
            idcg += 1
        else:
            idcg += 1 / np.log2(i + 1 + 1)
    return idcg

def ndcg_at_k(recommended_items, relevant_set, k):
    dcg = dcg_at_k(recommended_items, relevant_set, k)
    idcg = idcg_at_k(relevant_set, k)
    return dcg / idcg if idcg > 0 else 0.0




from tqdm import tqdm
from contextlib import redirect_stdout
import io


def evaluate_all_users(index, movie_ids, train, test, k=10):
    if k < 1:
        raise ValueError("k must be at least 1.")

    train_by_user = {
        user_id: rows
        for user_id, rows in train.groupby("user_id")
    }

    # Use training-known movies, matching the SVD/NCF candidate pool.
    pool_ids = np.asarray(sorted(train["movie_id"].unique()))

    movie_to_index = {
        movie_id: position
        for position, movie_id in enumerate(movie_ids)
    }

    if any(movie_id not in movie_to_index for movie_id in pool_ids):
        raise ValueError("Some training movies have no saved embedding.")

    pool_vectors = np.asarray(
        [
            index.reconstruct(movie_to_index[movie_id])
            for movie_id in pool_ids
        ],
        dtype=np.float32,
    )

    evaluation_index = faiss.IndexFlatL2(index.d)
    evaluation_index.add(pool_vectors)

    # Only test users and movies known during training.
    eligible_test = test[
        test["user_id"].isin(train_by_user)
        & test["movie_id"].isin(pool_ids)
    ]

    test_groups = eligible_test.groupby("user_id")

    total_ndcg = 0.0
    evaluated_users = 0

    for user_id, user_test in tqdm(
        test_groups,
        total=test_groups.ngroups,
        desc="Evaluating all users",
        unit="user",
    ):
        watched_movies = train_by_user[user_id]
        user_mean = watched_movies["rating"].mean()

        liked_movies = watched_movies[
            watched_movies["rating"] >= user_mean
        ]

        relevant_set = set(
            user_test.loc[
                user_test["rating"] >= user_mean,
                "movie_id",
            ]
        )

        relevant_set -= set(watched_movies["movie_id"])

        if not relevant_set:
            continue

        # Hide individual recommendations; keep the progress bar.
        with redirect_stdout(io.StringIO()):
            recommendations = personalized_recommendations(
                evaluation_index,
                pool_ids,
                watched_movies,
                liked_movies,
                user_id,
                k,
            )

        recommended_items = [
            movie_id for movie_id, distance in recommendations
        ]

        total_ndcg += ndcg_at_k(
            recommended_items, relevant_set, k
        )
        evaluated_users += 1

    if evaluated_users == 0:
        raise ValueError("No eligible users to evaluate.")

    average_ndcg = total_ndcg / evaluated_users

    print(f"\nEvaluated users: {evaluated_users:,}")
    print(f"NDCG@{k}: {average_ndcg:.6f}")

    return average_ndcg



df = pd.read_csv("data/data/processed/movies.csv")

def create_textual_representation(row) :
    textual_representation = f"""  title : {row["title"]} , genres: {row["genres"]} , director : {row["director"]} , cast : {row["cast"]}
    , year : {row["year"]} , runtime : {row["runtime"]} , original_language : {row["original_language"]} , overview : {row["overview"]}
    , keywords : {row["keywords"]} , metadata_features : {row["metadata_features"]} , text_features: {row["text_features"]}
    
       """
    return textual_representation

df["textual_representation"] = df.apply(
    create_textual_representation, axis=1
)

dim = 768
index = faiss.IndexFlatL2(dim)

x = np.zeros( (len(df['textual_representation']), dim) , dtype=np.float32 )

movies_df = pd.read_csv('data/data/processed_30k/movies.csv')
users_df = pd.read_csv('data/data/processed_30k/users.csv')
interactions_df = pd.read_csv('data/data/processed_30k/ratings.csv')


def prepare (df) :
    for i, representation in enumerate(df['textual_representation']):
        res = requests.post(
            "http://localhost:11434/api/embed",
            json={
                "model": "embeddinggemma",
                "input": representation,
            },
            timeout=120,
        )

        if not res.ok:
            raise RuntimeError(f"Movie position {i}: {res.text}")

        embedding = np.asarray(
            res.json()["embeddings"][0],
            dtype=np.float32,
        )

        assert embedding.shape == (dim,)
        assert np.isfinite(embedding).all()

        x[i] = embedding

        if (i + 1) % 200 == 0:
            print(f"Processed {i + 1}/{len(df)} movies", flush=True)

    index.add(x)

    faiss.write_index(index, "movie_embeddinggemma.faiss")

    # Preserve which movie belongs to each vector.
    np.save(
        "movie_embeddinggemma_ids.npy",
        df["movie_id"].to_numpy(),
        allow_pickle=False,
    )

    print(f"Saved {index.ntotal} movie vectors.")





def get_similar_movie (movie , movie_ids , index ):


    res = requests.post(
        "http://localhost:11434/api/embed",
        json={
            "model": "embeddinggemma",
            "input": movie["textual_representation"],
        },
        timeout=120,
    )
    res.raise_for_status()

    e = np.asarray(
        res.json()["embeddings"][0],
        dtype=np.float32,
    ).reshape(1, -1)

    assert e.shape[1] == index.d

    # Retrieve 30 movies similar to ours
    D, I = index.search(e, 30)

    movies_by_id = df.set_index("movie_id", verify_integrity=True)

    print("Similar movies to:", movie["title"])

    for distance, position in zip(D[0], I[0]):
        if position < 0:
            continue

        matched_id = movie_ids[position]

        if matched_id == movie["movie_id"]:
            continue

        title = movies_by_id.loc[matched_id, "title"]
        print(f"{distance:.4f} | {title}")



def load ():
    # Load the saved vectors and their movie IDs.
    index = faiss.read_index("movie_embeddinggemma.faiss")
    movie_ids = np.load(
        "movie_embeddinggemma_ids.npy",
        allow_pickle=False,
    )

    assert index.ntotal == len(movie_ids)
    assert index.ntotal > 0

    return  index , movie_ids


def build_constraint_profile(interactions_df, user_id ,movies_df , users_df):
            user_ratings = interactions_df[interactions_df['user_id'] == user_id]

            watched_movies = user_ratings.merge(movies_df, on='movie_id', how='inner')


            # get liked movies using the mean of each user
            user_mean = watched_movies['rating'].mean()
            liked_movies = watched_movies[watched_movies["rating"] >= user_mean]

            # all genres user watched
            all_watched_genres = set()
            for genre_string in watched_movies['genres'].dropna():
                for g in genre_string.split('|'):
                    all_watched_genres.add(g.strip())

            # favourite genres — top 5 most frequent across liked movies
            genre_counts = Counter()
            for genre_string in liked_movies['genres'].dropna():
                for g in genre_string.split('|'):
                    genre_counts[g.strip()] += 1
            liked_genres = set(g for g, _ in genre_counts.most_common(5))

            # favourite directors — top 3 most frequent across liked movies
            director_counts = Counter()
            for director in liked_movies['director'].dropna():
                director_counts[director.strip()] += 1
            liked_directors = set(d for d, _ in director_counts.most_common(3))

            actor_counts = Counter()
            for cast_string in liked_movies['cast'].dropna():
                for actor in cast_string.split('|'):
                    actor_counts[actor.strip()] += 1

            top_actors = [actor for actor, _ in actor_counts.most_common(8)]

            liked_movies['decade'] = (liked_movies['year'] // 10) * 10
            preferred_decade = liked_movies['decade'].mode().iloc[0] if not liked_movies.empty else 1990

            # Preferred language
            preferred_language = watched_movies['original_language'].mode().iloc[
                0] if not watched_movies.empty else 'en'

            # Average runtime they tolerate
            avg_runtime = watched_movies['runtime'].mean()

            user_row = users_df[users_df['user_id'] == user_id].iloc[0]
            user_age = int(user_row['age'])

            user_ratings = interactions_df[interactions_df['user_id'] == user_id]

            watched_movies = user_ratings.merge(movies_df, on='movie_id', how='inner')

            # get liked movies using the mean of each user
            user_mean = watched_movies['rating'].mean()
            liked_movies = watched_movies[watched_movies["rating"] >= user_mean]



            return {
                'user_id': user_id,
                'age': user_age,
                'all_watched_genres': all_watched_genres,  # for hard constraint
                'liked_genres': liked_genres,  # for soft constraint
                'liked_directors': liked_directors,  # for soft constraint
                'preferred_decade': preferred_decade,  # for soft constraint
                'preferred_language': preferred_language,  # for hard constraint
                'avg_runtime': avg_runtime,  # for contextual constraint
                'n_ratings': len(user_ratings),  # for sparse user detection
                'liked_actors': top_actors, # new , added to experiment with the constraint layer and improve accuracy
                'watched_movies' : watched_movies,
                'liked_movies' : liked_movies,

            }

def personalized_recommendations (index, movie_ids , watched_movies , liked_movies   , user_id , n):
    # Map each movie ID to its saved vector position.
    movie_to_index = {
        movie_id: position
        for position, movie_id in enumerate(movie_ids)
    }

    liked_indices = [
        movie_to_index[movie_id]
        for movie_id in liked_movies["movie_id"].unique()
        if movie_id in movie_to_index
    ]

    # Retrieve one saved embedding per liked movie.
    liked_embeddings = np.zeros(
        (len(liked_indices), index.d),
        dtype=np.float32,
    )

    for row, position in enumerate(liked_indices):
        liked_embeddings[row] = index.reconstruct(position)

    print("Liked embeddings shape:", liked_embeddings.shape)

    # Retrieve 30 movies similar to ours
    D, I = index.search(liked_embeddings, 30)

    movies_by_id = df.set_index("movie_id", verify_integrity=True)

    print("Similar movies to user:", user_id)

    watched_ids = set(watched_movies["movie_id"])

    matches ={}

    for distances, positions in zip(D, I):
        for distance, position in zip(distances, positions):
            if position < 0:
                continue

            matched_id = movie_ids[position]

            if matched_id in watched_ids:
                continue

            if matched_id not in matches or distance < matches[matched_id]:
                matches[matched_id] = float(distance)


    sorted_matches = sorted(
        matches.items(),
        key=operator.itemgetter(1),
    )
    # select the first 30
    top_matches = sorted_matches[:n]

    for movie_id, distance in top_matches:
        title = movies_by_id.loc[movie_id, "title"]
        print(f"{distance:.4f} | {title}")

    return top_matches



# test get similar movies
# load()
index , movie_ids = load()
# movie = df.iloc[2]
# get_similar_movie(movie , movie_ids , index)

# test user profile def
# user_id = 1
# print(build_constraint_profile(interactions_df , user_id , movies_df , users_df ))

# test personilized recs


data = interactions_df[interactions_df["movie_id"].isin(movie_ids)]
train, test = python_random_split(
            data,
            ratio=0.75,
            seed=42,
        )

user_id = 9


average_ndcg = evaluate_all_users(
    index,
    movie_ids,
    train,
    test,
    k=10,
)

# test movies recs using a query
def recommendations_using_query (query :str , k : int):
    res = requests.post(
        "http://localhost:11434/api/embed",
        json={"model": "embeddinggemma", "input": query},
        timeout=120,
    )

    res.raise_for_status()

    query_embedding = np.asarray(
        res.json()["embeddings"][0],
        dtype=np.float32,
    ).reshape(1, -1)

    D, I = index.search(query_embedding, k)
    movies_by_id = df.set_index("movie_id", verify_integrity=True)

    for distance, position in zip(D[0], I[0]):
        if position < 0:
            continue

        matched_id = movie_ids[position]
        title = movies_by_id.loc[matched_id, "title"]

        print(f"{distance:.4f} | {title}")


# query = input("Enter query: ")
# recommendations_using_query(query , 30)

# Text classification

# zero shot classification

# we take a pre trained LLM , and without any further training and no further work
# we gonna get the model to assign text to a category

# we provide the model with a prompt and the text that we want it to classify , optionaly we might provides
# categories that we want to classify as part of the prompt

from transformers import pipeline
pipe = pipeline("zero-shot-classification" , model = "facebook/bart-large-mnli")

def zero_shot_classification(movie_overview, themes):
    return pipe(
        movie_overview,
        candidate_labels=themes,
        multi_label=True,
    )


overview = (
    "A computer hacker discovers that his world is a simulated reality "
    "and joins a rebellion against the machines controlling it."
)

themes = [
    "artificial intelligence",
    "rebellion",
    "romance and relationships",
    "war and military",
    "crime and organised crime",
]

# result = zero_shot_classification(overview, themes)
#
# for theme, score in zip(result["labels"], result["scores"]):
#     print(f"{theme}: {score:.4f}")

from collections import Counter

import json
import requests

def generate_themes_for_all_movies (movies_df):
    themes = []
    keywords = {}
    for row_index, movie in movies_df.dropna(subset=["keywords"]).iterrows():
        # set the current movied id from the df to a var movie_id
        movie_id = movie["movie_id"]
        # store the keywords to that specific movie id
        movie_keywords = movie["keywords"].split("|")
        keywords[movie_id] = movie_keywords

    keyword_counts = Counter()

    for movie_keywords in keywords.values():
        unique_keywords = {
            keyword.strip().lower()
            for keyword in movie_keywords
            if keyword.strip()
        }
        keyword_counts.update(unique_keywords)

    print(keyword_counts.most_common(20))

    keyword_sample = [
        keyword
        for keyword, count in keyword_counts.most_common(100)
    ]

    prompt = (
        "Group these movie keywords into clear movie themes. "
        "Merge related keywords, but keep distinct topics such as "
        "World War I and World War II separate. "
        "Use only the supplied keywords in the groups. "
        "Leave ambiguous or irrelevant keywords ungrouped. "
        'Return only JSON with two fields: '
        '"theme_groups": an object mapping theme names to keyword lists, '
        'and "unmapped_keywords": a list of ungrouped keywords.\n'
        + json.dumps(keyword_sample)
    )

    res = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": "llama3.2:3b",
            "messages": [{"role": "user", "content": prompt}],
            "format": "json",
            "stream": False,
            "options": {"temperature": 0},
        },
        timeout=300,
    )
    res.raise_for_status()

    result = json.loads(res.json()["message"]["content"])
    theme_groups = result["theme_groups"]
    themes = list(theme_groups.keys())

    # Inspect the proposed groups and ungrouped keywords.
    print(json.dumps(result, indent=2))

    return themes, theme_groups

# generate_themes_for_all_movies(movies_df)


from llm_enricher import LLM_MovieEnricher


if __name__ == "__main__":
    movies_df = pd.read_csv("data/data/processed_30k/movies.csv")
    enricher = LLM_MovieEnricher(movies_df, model_name="llama3.2:3b")

    movie = enricher.get_movie(603)          # The Matrix; Toy Story is 862
    input_text = enricher.build_input_text(movie)
    print(input_text)

    result = enricher.call_llm(enricher.build_prompt(input_text))

    print("Status:", result["status"])
    print(json.dumps(json.loads(result["content"]), indent=2))
    print(f"LLM time: {result['llm_seconds']:.2f}s | "
          f"tokens in/out: {result['prompt_tokens']}/{result['output_tokens']}")

    import build_movie_texts as bmt

    # Same data, split, users and pool as build_movie_texts.py.
    movies = bmt.load_movies()
    ratings = bmt.prepare_ratings(movies)
    train, test = bmt.split_ratings(ratings)
    users = bmt.build_user_data(train, test)

    eval_movie_ids = movies["movie_id"].to_numpy()
    position_of = {m: i for i, m in enumerate(eval_movie_ids)}
    pool_positions = np.array(sorted(position_of[m] for m in set(train["movie_id"])))

    # Old index → similarity table in the same movie order.
    vectors = index.reconstruct_n(0, index.ntotal)
    row_of = {m: r for r, m in enumerate(movie_ids)}
    vectors = vectors[[row_of[m] for m in eval_movie_ids]]
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    old_similarity = (vectors @ vectors.T).astype(np.float32)
    np.fill_diagonal(old_similarity, 0)

    results = bmt.evaluate_method("old_textual_representation", old_similarity, users,
                                  pool_positions, position_of, eval_movie_ids, bmt.K)
    print(results[["ndcg", "precision", "recall"]].mean().round(4))
