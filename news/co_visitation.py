
import json
from datetime import datetime
from scipy.sparse import coo_matrix

from pathlib import Path
from time import perf_counter
from sklearn.metrics import ndcg_score

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import ndcg_score

from sklearn.metrics.pairwise import cosine_similarity
# import cudf
# import cupy as cp
from sklearn.feature_extraction.text import TfidfVectorizer
from math import ceil
from collections import Counter

from sentence_transformers import SentenceTransformer

base_dir = Path(__file__).resolve().parents[1]  # project root (this file is in news/)


news_path = (
    base_dir
    / "data"
    / "Mind_Large_Data"
    / "mind_large_dataset_csv"
    / "news.csv"
)

news = pd.read_csv(
    news_path,
    usecols=["article_id", "title", "abstract", "category"],
    dtype=str,
    keep_default_na=False,
)

news_by_id = news.set_index(
    "article_id",
    verify_integrity=True,
)

news_by_id["text"] = (
    news_by_id["title"].fillna("")
    + " "
    + news_by_id["abstract"].fillna("")
)



training_path = (
    base_dir
    / "data"
    / "Mind_Large_Data"
    / "mind_large_dataset_csv"
    / "training_behaviors.csv"
)

training_data = pd.read_csv(
    training_path,
    dtype=str,
    keep_default_na=False,
)


validation_path = (
    base_dir
    / "data"
    / "Mind_Large_Data"
    / "mind_large_dataset_csv"
    / "validation_behaviors.csv"
)

validation_data = pd.read_csv(
    validation_path,
    dtype=str,
    keep_default_na=False,
)





# Stage 1: Prepare interactions once

article_id = news_by_id.index[0]

print(news_by_id.loc[article_id, "title"])
print(news_by_id.loc[article_id, "category"])

#
article_to_category = news_by_id["category"].to_dict()




row = validation_data.iloc[0]
history_ids = list(dict.fromkeys(row["history"].split()))

history_categories = []

for article_id in history_ids:
    category = article_to_category.get(article_id)

    if category:
        history_categories.append(category)

category_counts = Counter(history_categories)
matched_history_size = len(history_categories)

print(category_counts)
print("History articles with categories:", matched_history_size)
candidate_ids = [
    entry.rsplit("-", 1)[0]
    for entry in row["impressions"].split()
]

category_scores = np.zeros(len(candidate_ids), dtype=np.float32)

if matched_history_size > 0:
    for position, article_id in enumerate(candidate_ids):
        category = article_to_category.get(article_id)

        category_scores[position] = (
            category_counts.get(category, 0)
            / matched_history_size
        )




category_results = pd.DataFrame({
    "article_id": candidate_ids,
    "category": [
        article_to_category.get(article_id, "")
        for article_id in candidate_ids
    ],
    "category_score": category_scores,
})

print(category_results.head(10).to_string(index=False))

article_to_category = news_by_id["category"].to_dict()

def score_categories(history_ids, candidate_ids):
    unique_history_ids = list(dict.fromkeys(history_ids))

    history_categories = [
        article_to_category[article_id]
        for article_id in unique_history_ids
        if article_to_category.get(article_id)
    ]

    scores = np.zeros(len(candidate_ids), dtype=np.float32)

    if not history_categories:
        return scores

    category_counts = Counter(history_categories)
    history_size = len(history_categories)

    for position, article_id in enumerate(candidate_ids):
        category = article_to_category.get(article_id)

        scores[position] = (
            category_counts.get(category, 0) / history_size
        )

    return scores





interactions = set()


for row in tqdm(training_data.itertuples(index=False), total=len(training_data)):
    for article_id in row.history.split():
        interactions.add((row.user_id, article_id))

    # we gonna add articles ids in impressions but only the ones who have 1

    for entry  in row.impressions.split():
        article , labels = entry.rsplit("-" , 1 )

        if labels == "1":
            interactions.add((row.user_id , article))


interactions_df = pd.DataFrame(
    interactions,
    columns=["user_id", "article_id"],
)

# print(interactions_df.head())

# to see all articles clicked by a specific user
# print(
#     interactions_df[
#         interactions_df["user_id"] == "U674703"
#     ]
# )
#
articles_per_user = interactions_df.groupby("user_id").size()

print("Unique interactions:", len(interactions_df))
print("Unique users:", interactions_df["user_id"].nunique())
print("Unique articles:", interactions_df["article_id"].nunique())

print("\nArticles per user:")
print(articles_per_user.describe(percentiles=[0.90, 0.99]))

counts = articles_per_user.astype("int64")
pair_contributions = (counts * (counts - 1)).sum()

print("\nDirected pair contributions:", pair_contributions)

# Stop here temporarily so the expensive matrix loop does not run.

# Stage 2: Build the matrix from those interactions



class CoVisitationRecommender:
    def __init__(self, top_neighbors=50):
        self.top_neighbors = top_neighbors
        self.article_to_index = {}
        self.index_to_article = []

        self.neighbors = {}

    def fit(self,  interactions_df):

        if interactions_df.empty:
            raise ValueError("No interactions provided.")


        unique_user_ids = sorted(interactions_df["user_id"].unique())

        self.index_to_article = sorted(
            interactions_df["article_id"].unique()
        )

        user_to_index = {
            user_id: index
            for index, user_id in enumerate(unique_user_ids)
        }

        self.article_to_index = {
            article_id: index
            for index, article_id in enumerate(self.index_to_article)
        }

        user_indices = (
            interactions_df["user_id"]
            .map(user_to_index)
            .to_numpy(dtype=np.int32)
        )

        article_indices = (
            interactions_df["article_id"]
            .map(self.article_to_index)
            .to_numpy(dtype=np.int32)
        )

        # Count the distinct readers of each article using the user–article matrix.

        number_of_users = len(unique_user_ids)
        number_of_articles = len(self.index_to_article)



        # Each unique user–article interaction gets a value of 1.
        values = np.ones(len(interactions_df), dtype=np.int32)

        X = coo_matrix(
            (values, (user_indices, article_indices)),
            shape=(number_of_users, number_of_articles),
            dtype=np.int32,
        ).tocsr()

        # Ensure each user–article pair represents one recorded click.
        X.sum_duplicates()
        X.data.fill(1)

        # Transpose: article × user.
        article_user_matrix = X.T.tocsr()

        # .sum(axis=1) adds the values across each article’s user columns.
        # np.asarray(...).ravel() converts the result into a one-dimensional NumPy array.

        # Count the distinct readers of each article using the user–article matrix.
        article_reader_counts = np.asarray(
            article_user_matrix.sum(axis=1)
        ).ravel().astype(np.float64)

        print("User–article matrix shape:", X.shape)
        print("Stored interactions:", X.nnz)

        PARTS = 128
        partition_size = max(1, ceil(number_of_articles / PARTS))

        # Clear relationships from any previous fit.
        self.neighbors = {}

        for article_start in tqdm(
                range(0, number_of_articles, partition_size),
                desc="Building neighbours",
        ):
            article_end = min(
                article_start + partition_size,
                number_of_articles,
            )

            # Shared-user counts for this block of source articles.
            block = (
                    article_user_matrix[article_start:article_end] @ X
            ).tocsr()

            block.sum_duplicates()
            block.eliminate_zeros()

            for local_row in range(article_end - article_start):
                article_index = article_start + local_row

                start = block.indptr[local_row]
                end = block.indptr[local_row + 1]

                neighbor_indices = block.indices[start:end]
                counts = block.data[start:end]

                # Remove the article's relationship with itself.
                keep = neighbor_indices != article_index
                neighbor_indices = neighbor_indices[keep]
                counts = counts[keep]

                # Cosine-normalised shared-reader counts.
                denominator = np.sqrt(
                    article_reader_counts[article_index]
                    * article_reader_counts[neighbor_indices]
                )

                weights = counts / denominator

                # Rank by normalised weight, then article index for ties.
                order = np.lexsort((neighbor_indices, -weights))
                selected = order[:self.top_neighbors]

                self.neighbors[article_index] = {
                    int(neighbor_indices[position]): float(weights[position])
                    for position in selected
                }

            del block

        return self

    def save(self, output_dir):
        if not self.index_to_article or not self.neighbors:
            raise ValueError("Fit or load a model before saving.")

        output_dir = Path(output_dir)

        # Require a new directory to avoid overwriting an existing model.
        output_dir.mkdir(parents=True, exist_ok=False)

        # Save index → article ID mapping.
        with (output_dir / "article_ids.json").open(
                "w", encoding="utf-8"
        ) as file:
            json.dump(self.index_to_article, file, ensure_ascii=False)

        articles_per_file = 1000
        number_of_articles = len(self.index_to_article)
        number_of_parts = (
                                  number_of_articles + articles_per_file - 1
                          ) // articles_per_file

        metadata = {
            "method": "binary_user_item_cooccurrence",
            "articles": number_of_articles,
            "top_neighbors": self.top_neighbors,
            "parts": number_of_parts,
            "completed_parts": [],
            "complete": False,
            "score": "cosine_normalized_shared_user_count",
        }

        metadata_path = output_dir / "metadata.json"

        with metadata_path.open("w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=2)

        for part, start in enumerate(
                range(0, number_of_articles, articles_per_file)
        ):
            end = min(start + articles_per_file, number_of_articles)

            rows = []

            for article_index in range(start, end):
                for neighbor_index, count in self.neighbors.get(
                        article_index, {}
                ).items():
                    rows.append(
                        (article_index, neighbor_index, count)
                    )

            part_matrix = pd.DataFrame(
                rows,
                columns=[
                    "article_index_x",
                    "article_index_y",
                    "count",
                ],
            ).astype({
                "article_index_x": "int32",
                "article_index_y": "int32",
                "count": "float64",
            })

            output_path = output_dir / f"part_{part:03d}.parquet"
            part_matrix.to_parquet(output_path, index=False)

            metadata["completed_parts"].append(part)

            with metadata_path.open("w", encoding="utf-8") as file:
                json.dump(metadata, file, indent=2)

            del rows, part_matrix

        metadata["complete"] = True

        with metadata_path.open("w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=2)

        print(f"Model saved to: {output_dir}")

        return output_dir

    def load(self, model_dir):
        model_dir = Path(model_dir)

        # Load settings.
        with (model_dir / "metadata.json").open(
                "r", encoding="utf-8"
        ) as file:
            metadata = json.load(file)

        if not metadata.get("complete", False):
            raise ValueError("The saved model is incomplete.")

        self.top_neighbors = metadata["top_neighbors"]

        # Load index → article ID.
        with (model_dir / "article_ids.json").open(
                "r", encoding="utf-8"
        ) as file:
            self.index_to_article = json.load(file)

        # Rebuild article ID → index.
        self.article_to_index = {
            article_id: index
            for index, article_id in enumerate(self.index_to_article)
        }

        # We still need to load the neighbour partitions here.

        return self

    def score(self, history_ids, candidate_ids):
        unique_history_ids = list(dict.fromkeys(history_ids))

        history_indices = []

        for article_id in unique_history_ids:
            if article_id in self.article_to_index:
                index = self.article_to_index[article_id]
                history_indices.append(index)

        scores = []

        for candidate_id in candidate_ids:
            candidate_index = self.article_to_index.get(candidate_id)
            total_score = 0

            if candidate_index is not None:
                for history_index in history_indices:
                    neighbors = self.neighbors.get(history_index, {})
                    total_score += neighbors.get(candidate_index, 0)

            scores.append(total_score)

        return np.asarray(scores, dtype=np.float32)

    def recommend(self, history_ids, n=10):
        # Collect candidates from history articles' neighbours.
        candidate_indices = set()

        for article_id in history_ids:
            history_index = self.article_to_index.get(article_id)

            if history_index is not None:
                neighbors = self.neighbors.get(history_index, {})
                candidate_indices.update(neighbors.keys())

        history_set = set(history_ids)

        candidate_ids = [
            self.index_to_article[index]
            for index in sorted(candidate_indices)
            if self.index_to_article[index] not in history_set
        ]

        if n <= 0:
            raise ValueError("n must be greater than zero.")

        scores = self.score(history_ids, candidate_ids)

        recommendations = pd.DataFrame({
            "article_id": candidate_ids,
            "score": scores,
        })

        recommendations = recommendations.sort_values(
            ["score", "article_id"],
            ascending=[False, True],
        )

        return recommendations.head(n).reset_index(drop=True)



def evaluate_scores(validation_data, score_fn):
    total_k5 = 0.0
    total_k10 = 0.0
    count = 0
    for row in tqdm(
            validation_data.itertuples(index=False),
            total=len(validation_data),
            desc="Evaluating",
    ):
        history_ids = row.history.split()

        candidate_ids = []
        click_labels = []

        for entry in row.impressions.split():
            article_id, label = entry.rsplit("-", 1)

            candidate_ids.append(article_id)
            click_labels.append(int(label))

        scores = score_fn(history_ids, candidate_ids)

        y_true = np.asarray(click_labels, dtype=np.float32).reshape(1, -1)
        y_score = np.asarray(scores, dtype=np.float32).reshape(1, -1)

        value_k5 = ndcg_score(y_true, y_score, k=5)
        value_k10 = ndcg_score(y_true, y_score, k=10)

        total_k5 += value_k5
        total_k10 += value_k10
        count += 1

    if count == 0:
        raise ValueError("No validation impressions to evaluate.")


    return {
        "ndcg@5": total_k5 / count,
        "ndcg@10": total_k10 / count,
    }



def show_recommendations(recommendations, news):
    # Attach titles/categories and display the results.
    pass

training_article_ids = sorted(
    interactions_df["article_id"].unique()
)

training_news = news_by_id.loc[
    news_by_id.index.isin(training_article_ids)
]

training_titles = training_news["text"]

print("Training articles with metadata:", len(training_news))
print(training_titles.head())

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    max_features=50_000,
    ngram_range=(1, 2),
    dtype=np.float32,
)

training_title_vectors = vectorizer.fit_transform(training_titles)

print("TF-IDF matrix shape:", training_title_vectors.shape)


article_ids = news_by_id.index.to_list()

article_to_title_index = {
    article_id: index
    for index, article_id in enumerate(article_ids)
}

all_title_vectors = vectorizer.transform(news_by_id["text"])

print("All article title vectors:", all_title_vectors.shape)


row = validation_data.iloc[0]
history_ids = list(dict.fromkeys(row["history"].split()))

history_title_indices = [
    article_to_title_index[article_id]
    for article_id in history_ids
    if article_id in article_to_title_index
]

if not history_title_indices:
    raise ValueError("This history has no articles with title metadata.")

history_vectors = all_title_vectors[history_title_indices]

user_title_profile = np.asarray(
    history_vectors.mean(axis=0)
)

print("User title profile shape:", user_title_profile.shape)

candidate_ids = [
    entry.rsplit("-", 1)[0]
    for entry in row["impressions"].split()
]

# Preserve candidate order; missing article metadata gets score zero.
title_scores = np.zeros(len(candidate_ids), dtype=np.float32)

known_positions = [
    position
    for position, article_id in enumerate(candidate_ids)
    if article_id in article_to_title_index
]

if known_positions:
    candidate_title_indices = [
        article_to_title_index[candidate_ids[position]]
        for position in known_positions
    ]

    candidate_vectors = all_title_vectors[candidate_title_indices]

    title_scores[known_positions] = cosine_similarity(
        user_title_profile,
        candidate_vectors,
    ).ravel()


title_recommendations = pd.DataFrame({
    "article_id": candidate_ids,
    "title_score": title_scores,
})

title_recommendations["title"] = (
    title_recommendations["article_id"].map(news_by_id["title"])
)

print(
    title_recommendations
    .sort_values(
        ["title_score", "article_id"],
        ascending=[False, True],
    )
    .head(10)
    .to_string(index=False)
)
def score_titles(history_ids, candidate_ids):
    scores = np.zeros(len(candidate_ids), dtype=np.float32)

    history_indices = [
        article_to_title_index[article_id]
        for article_id in dict.fromkeys(history_ids)
        if article_id in article_to_title_index
    ]

    known_positions = [
        position
        for position, article_id in enumerate(candidate_ids)
        if article_id in article_to_title_index
    ]

    if not history_indices or not known_positions:
        return scores

    history_vectors = all_title_vectors[history_indices]

    user_profile = np.asarray(
        history_vectors.mean(axis=0)
    )

    candidate_indices = [
        article_to_title_index[candidate_ids[position]]
        for position in known_positions
    ]

    scores[known_positions] = cosine_similarity(
        user_profile,
        all_title_vectors[candidate_indices],
    ).ravel()

    return scores


def score_individual_articles(history_ids, candidate_ids, top_k=3):
    title_scores = np.zeros(len(candidate_ids), dtype=np.float32)

    unique_history_ids = list(dict.fromkeys(history_ids))

    history_title_indices = [
        article_to_title_index[article_id]
        for article_id in unique_history_ids
        if article_id in article_to_title_index
    ]

    if not history_title_indices or not candidate_ids:
        return title_scores

    history_vectors = all_title_vectors[history_title_indices]



    known_positions = [
        position
        for position, article_id in enumerate(candidate_ids)
        if article_id in article_to_title_index
    ]

    if known_positions:
        candidate_title_indices = [
            article_to_title_index[candidate_ids[position]]
            for position in known_positions
        ]

        candidate_vectors = all_title_vectors[candidate_title_indices]

        similarities = cosine_similarity(
            history_vectors,
            candidate_vectors,
        )
        # Number of history matches to use.
        k = min(top_k, similarities.shape[0])

        # Sort each candidate's similarities and keep its strongest k matches.
        strongest_matches = np.sort(
            similarities,
            axis=0,
        )[-k:]

        # Average those matches to get one score per known candidate.
        title_scores[known_positions] = strongest_matches.mean(axis=0)

    return title_scores

def score_content(history_ids, candidate_ids, alpha=0.90):
    title_scores = score_titles(history_ids, candidate_ids)
    category_scores = score_categories(history_ids, candidate_ids)

    return (
        alpha * title_scores
        + (1 - alpha) * category_scores
    )


def scale_scores(scores):
    scores = np.asarray(scores, dtype=np.float32)

    if scores.size == 0:
        return scores

    maximum = scores.max()

    if maximum > 0:
        return scores / maximum

    return scores

def score_hybrid(history_ids, candidate_ids, beta=0.2):
    content_scores = score_content(
        history_ids, candidate_ids, alpha=0.9
    )

    covisitation_scores = model.score(
        history_ids, candidate_ids
    )

    # No collaborative evidence: keep content-only scoring.
    if not np.any(covisitation_scores > 0):
        return content_scores

    content_scaled = scale_scores(content_scores)
    covisitation_scaled = scale_scores(covisitation_scores)

    return (
        (1 - beta) * content_scaled
        + beta * covisitation_scaled
    )





row = validation_data.iloc[0]

history_ids = row["history"].split()
candidate_ids = []
click_labels = []

for entry in row["impressions"].split():
    article_id, label = entry.rsplit("-", 1)
    candidate_ids.append(article_id)
    click_labels.append(int(label))

scores = score_titles(history_ids, candidate_ids)

assert scores.shape == (len(candidate_ids),)
assert np.isfinite(scores).all()

# Evaluate all candidates in their original order.
y_true = np.asarray(click_labels).reshape(1, -1)
y_score = scores.reshape(1, -1)

print("NDCG@5:", ndcg_score(y_true, y_score, k=5))
print("NDCG@10:", ndcg_score(y_true, y_score, k=10))

# Display the highest-ranked candidates.
results = pd.DataFrame({
    "article_id": candidate_ids,
    "title_score": scores,
    "clicked": click_labels,
})

results["title"] = results["article_id"].map(news_by_id["title"])

ranked = results.sort_values(
    ["title_score", "article_id"],
    ascending=[False, True],
).reset_index(drop=True)

ranked["rank"] = np.arange(1, len(ranked) + 1)

print("Total clicked:", results["clicked"].sum())

print(
    ranked.loc[
        ranked["clicked"] == 1,
        ["rank", "article_id", "title_score", "title"],
    ].to_string(index=False)
)

print(
    results.sort_values(
        ["title_score", "article_id"],
        ascending=[False, True],
    )
    .head(10)
    .to_string(index=False)
)

# title_metrics = evaluate_titles(validation_data)

# content_metrics = evaluate_scores(validation_data, score_content)
#
#
# print("Titles + categories, alpha=0.90")
# print(f"NDCG@5:  {content_metrics['ndcg@5']:.6f}")
# print(f"NDCG@10: {content_metrics['ndcg@10']:.6f}")
validation_sample = validation_data.sample(
    n=10_000,
    random_state=42,
)

baseline_results = evaluate_scores(
    validation_sample,
    score_titles,
)

individual_results = evaluate_scores(
    validation_sample,
    score_individual_articles,
)

print("\nTitle + abstract: averaged history profile")
print(f"NDCG@5:  {baseline_results['ndcg@5']:.6f}")
print(f"NDCG@10: {baseline_results['ndcg@10']:.6f}")

print("\nTitle + abstract: individual matching, top_k=3")
print(f"NDCG@5:  {individual_results['ndcg@5']:.6f}")
print(f"NDCG@10: {individual_results['ndcg@10']:.6f}")


def score_max_match(history_ids, candidate_ids):
    return score_individual_articles(
        history_ids,
        candidate_ids,
        top_k=1,
    )


max_results = evaluate_scores(
    validation_sample,
    score_max_match,
)

print("\nTitle + abstract: strongest individual match")
print(f"NDCG@5:  {max_results['ndcg@5']:.6f}")
print(f"NDCG@10: {max_results['ndcg@10']:.6f}")



# ---- co-visitation on the official dev set ----
import mind_eval as me

train_imp = me.load_impressions("train")          # all of train: dev is the next day
hist_users = np.repeat(train_imp.users, np.diff(train_imp.history_offsets))
cand_users = np.repeat(train_imp.users, train_imp.sizes)
clicked = train_imp.labels == 1

users = np.concatenate([hist_users, cand_users[clicked]]).astype(np.int64)
items = np.concatenate([train_imp.history, train_imp.candidates[clicked]]).astype(np.int64)
pairs = np.unique((users << 20) | items)          # one row per (user, article)
pairs_df = pd.DataFrame({"user_id": pairs >> 20, "article_id": pairs & ((1 << 20) - 1)})
print("Unique user-article pairs:", len(pairs_df))

covis = CoVisitationRecommender(top_neighbors=50).fit(pairs_df)

dev = me.load_impressions("dev")
dev_scores = [
    covis.score(dev.history_of(i), dev.candidates_of(i))
    for i in tqdm(range(dev.n_impressions), desc="Scoring dev")
]
me.evaluate(dev, dev_scores, name="covisitation_top50",
            notes="item co-visitation, cosine-normalised shared readers, top 50 neighbours, fit on all of train")