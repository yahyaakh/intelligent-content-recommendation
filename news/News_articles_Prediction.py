import pandas as pd
from pathlib import Path
import tensorflow as tf
from sklearn.metrics import ndcg_score
import numpy as np

import json
from collections import Counter
import mind_eval as me
PROJECT = Path(__file__).resolve().parents[1]  # project root (this file is in news/)
base_dir = PROJECT / "data" / "Mind_Large_Data"

train_dir = base_dir / "MINDlarge_train" / "MINDlarge_train"
dev_dir = base_dir / "MINDlarge_dev" / "MINDlarge_dev"
test_dir = base_dir / "MINDlarge_test" / "MINDlarge_test"

raw_dir = PROJECT / "data" / "Mind_data" / "raw"

news_columns = [
    "article_id",
    "category",
    "subcategory",
    "title",
    "abstract",
    "url",
    "title_entities",
    "abstract_entities",
]

behavior_columns = [
    "impression_id",
    "user_id",
    "timestamp",
    "history",
    "impressions",
]

news = pd.read_csv(
    train_dir / "news.tsv",
    sep="\t",
    header=None,
    names=news_columns,
    dtype=str,
    keep_default_na=False,
)

behaviors = pd.read_csv(
    train_dir / "behaviors.tsv",
    sep="\t",
    header=None,
    names=behavior_columns,
    dtype=str,
    keep_default_na=False,
)

article_lookup = (
    news.set_index("article_id", verify_integrity=True)[
        ["category", "subcategory"]
    ]
    .to_dict("index")
)


output_dir = PROJECT / "output" / "news"  # trained model (news_recommender.keras)
output_dir.mkdir(parents=True, exist_ok=True)
#
# news.to_csv(output_dir / "news.csv", index=False, encoding="utf-8")
# behaviors.to_csv(output_dir / "behaviors.csv", index=False, encoding="utf-8")

# print("CSV files saved to:", output_dir)


def user_profile(user_id: str, impression_id: str):
    user_behaviors = behaviors[
        (behaviors["user_id"] == user_id)
        & (behaviors["impression_id"] == impression_id)
        ]

    if user_behaviors.empty:
        raise ValueError(
            f"No impression {impression_id} found for user {user_id}"
        )

    if len(user_behaviors) > 1:
        raise ValueError(
            f"Multiple rows found for impression {impression_id}"
        )

    user_demo = user_behaviors.iloc[0]
    clicked_articles_ids = user_demo['history'].split()

    number_of_articles_in_history = len(clicked_articles_ids)

    unique_history_count = len(set(clicked_articles_ids))



    matching_articles = news[news['article_id'].isin(clicked_articles_ids)]

    all_clicked_categories = set(matching_articles['category'])

    matched_unique_count = matching_articles["article_id"].nunique()

    missing_article_ids = sorted(
        set(clicked_articles_ids)
        - set(matching_articles["article_id"])
    )


    clicked_titles = set(matching_articles["title"])
    clicked_subcategories = set(matching_articles["subcategory"])


    candidate_article_ids = []
    click_labels = []

    for entry in user_demo["impressions"].split():
        article_id, label = entry.rsplit("-", 1)

        candidate_article_ids.append(article_id)
        click_labels.append(int(label))

    # Count each article once, even if news contains duplicate IDs
    matched_unique_articles = matching_articles.drop_duplicates("article_id")

    category_counts = matched_unique_articles["category"].value_counts()
    subcategory_counts = matched_unique_articles["subcategory"].value_counts()

    category_proportions = (
        category_counts / matched_unique_count
        if matched_unique_count > 0
        else category_counts.astype(float)
    )


    profile = {
        "user_id": user_id,
        "impression_id": user_demo["impression_id"],
        "timestamp": user_demo["timestamp"],
        "history_count": number_of_articles_in_history,
        "unique_history_count": unique_history_count,
        "matched_unique_count": matched_unique_count,
        "missing_article_ids": missing_article_ids,
        "clicked_titles": clicked_titles,
        "clicked_subcategories": clicked_subcategories,
        "all_clicked_categories": all_clicked_categories,
        "category_counts": category_counts.to_dict(),
        "subcategory_counts": subcategory_counts.to_dict(),
        "category_proportions": category_proportions.to_dict(),
    }

    return {
        "profile": profile,
        "candidate_article_ids": candidate_article_ids,
        "click_labels": click_labels,
    }



timestamps = pd.to_datetime(
    behaviors["timestamp"],
    format="%m/%d/%Y %I:%M:%S %p",
    errors="raise",
)

# Before November 14, 2019: training.
# On November 14, 2019: validation.
# Leave official dev separate for final local evaluation.

cutoff = pd.Timestamp("2019-11-14 00:00:00")

training_data = behaviors.loc[timestamps < cutoff]
validation_rows = behaviors.loc[timestamps >= cutoff]


class NewsRecommender:
    def __init__(self):
        self.model = None
        self.normalizer = None
    def build_model(self):
        self.normalizer = tf.keras.layers.Normalization(axis=-1)

        self.model = tf.keras.Sequential([
            tf.keras.Input(shape=(3,)),
            self.normalizer,
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(8, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ])

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=0.001
            ),
            loss="binary_crossentropy",
        )

    def fit(self, train_dataset, validation_dataset):
        if self.model is None:
            self.build_model()

        print("Calculating normalization from training features...")

        training_features = train_dataset.map(
            lambda features, labels: features
        )

        self.normalizer.adapt(training_features)

        print("Starting training...")

        return self.model.fit(
            train_dataset,
            validation_data=validation_dataset,
            epochs=10,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=2,
                    restore_best_weights=True,
                )
            ],
        )

    def score(self, candidate_features):
        if self.model is None:
            raise ValueError("Train or load a model first.")

        return self.model.predict(
            candidate_features,
            verbose=0,
        ).reshape(-1)



def generate_batches (behavior_data, batch_size=4096 ) :
    feature_buffer  = []
    label_buffer = []

    for row in behavior_data.itertuples(index=False):
        history_ids = set(row.history.split())

        history_articles = [
            article_lookup[article_id]
            for article_id in history_ids
            if article_id in article_lookup
        ]

        history_size = len(history_articles)

        category_counts = Counter(
            article["category"] for article in history_articles
        )

        subcategory_counts = Counter(
            article["subcategory"] for article in history_articles
        )

        # Build one example per candidate.
        for entry in row.impressions.split():
            article_id, label_text = entry.rsplit("-", 1)

            if label_text not in ("0", "1"):
                raise ValueError(f"Invalid click label: {entry}")

            if article_id not in article_lookup:
                raise ValueError(
                    f"Missing candidate metadata: {article_id}"
                )

            candidate = article_lookup[article_id]

            if history_size > 0:
                category_match = (
                        category_counts[candidate["category"]]
                        / history_size
                )

                subcategory_match = (
                        subcategory_counts[candidate["subcategory"]]
                        / history_size
                )
            else:
                category_match = 0.0
                subcategory_match = 0.0

            feature_buffer.append([
                history_size,
                category_match,
                subcategory_match,
            ])
            label_buffer.append(int(label_text))

            if len(feature_buffer) == batch_size:
                yield (
                    np.asarray(feature_buffer, dtype=np.float32),
                    np.asarray(
                        label_buffer, dtype=np.float32
                    ).reshape(-1, 1),
                )

                feature_buffer = []
                label_buffer = []

        # Include the final batch even if it is smaller.
    if feature_buffer:
        yield (
            np.asarray(feature_buffer, dtype=np.float32),
            np.asarray(
                label_buffer, dtype=np.float32
            ).reshape(-1, 1),
        )



def make_dataset(behavior_data, batch_size=4096):
    return tf.data.Dataset.from_generator(
        lambda: generate_batches(behavior_data, batch_size),
        output_signature=(
            tf.TensorSpec(shape=(None, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
        ),
    ).prefetch(1)

def dcg_score(y_true, y_score, k=10):
    # 1. Convert labels and scores to 1D NumPy arrays.


    # 2. Check that both arrays have the same length.
    ...

    # 3. Sort candidate positions by descending predicted score.
    ...

    # 4. Select the true labels at the top-k positions.
    ...

    # 5. Calculate gains: 2 ** label - 1.
    ...

    # 6. Calculate discounts: log2(rank + 1).
    #    Ranks start at 1.
    ...

    # 7. Return the sum of gains / discounts.
    return ...



def get_rec (user_id ,impression_id  , model , n) :

    result = user_profile(user_id, impression_id)
    candidate_ids = result["candidate_article_ids"]


    selected_row = behaviors[ (user_id == behaviors["user_id"]) & (behaviors["impression_id"] == impression_id) ]

    if not candidate_ids:
        raise ValueError("This impression has no candidates.")

    score_parts = []

    for features, labels in generate_batches(selected_row):
        # features contains rows of:
        # [history_size, category_match, subcategory_match]
        batch_scores = model.score(features)
        score_parts.append(batch_scores)

    scores = np.concatenate(score_parts)

    print(
        "NDCG@5:",
        calculate_NDCG(
            user_id=user_id,
            impression_id=impression_id,
            y_score=scores,
            k=5,
        ),
    )

    if len(scores) != len(candidate_ids):
        raise ValueError("Candidate and score counts do not match.")

    recommendations = pd.DataFrame({
        "article_id": candidate_ids,
        "score": scores,
        "clicked": result["click_labels"],
    })

    # Inspect ALL candidates.
    print("Total candidates:", len(recommendations))
    print("Total clicked:", recommendations["clicked"].sum())

    print("\nActually clicked articles:")
    print(
        recommendations.loc[recommendations["clicked"] == 1]
        .sort_values("score", ascending=False)
        .to_string(index=False)
    )

    # Article ID provides a consistent tie-breaker.
    recommendations = recommendations.sort_values(
        ["score", "article_id"],
        ascending=[False, True],

    ).head(n)



    # Attach article information without changing ranking order.
    news_by_id = news.set_index("article_id", verify_integrity=True)

    for column in ["title", "category", "subcategory"]:
        recommendations[column] = recommendations["article_id"].map(
            news_by_id[column]
        )

    return recommendations.reset_index(drop=True)


def calculate_NDCG (y_score ,user_id , impression_id , k=5) :
    result = user_profile(user_id, impression_id)

    labels = np.asarray(
        result["click_labels"],
        dtype=np.float32,
    ).reshape(1, -1)

    scores = np.asarray(
        y_score,
        dtype=np.float32,
    ).reshape(1, -1)


    return ndcg_score(labels, scores, k=k)


def evaluate_model(model, validation_data, batch_size=4096):
    if validation_data.empty:
        raise ValueError("No validation impressions to evaluate.")

    if model.model is None:
        raise ValueError("Train or load the model first.")

    # Generator preserves row order and candidate order.
    # These lengths tell us where each impression ends.
    impression_sizes = (
        len(entries.split())
        for entries in validation_data["impressions"]
    )

    current_size = next(impression_sizes)

    if current_size == 0:
        raise ValueError("An impression has no candidates.")

    total_k5 = 0.0
    total_k10 = 0.0
    count = 0
    candidates_seen = 0

    pending_scores = []
    pending_labels = []
    collected = 0

    for features, labels in generate_batches(
        validation_data,
        batch_size=batch_size,
    ):
        # Direct inference avoids predict() setup for every impression.
        scores = model.model(
            features,
            training=False,
        ).numpy().reshape(-1)

        labels = labels.reshape(-1)

        if len(scores) != len(labels):
            raise ValueError("Score and label counts do not match.")

        candidates_seen += len(scores)
        position = 0

        while position < len(scores):
            if current_size is None:
                raise ValueError("More candidates than expected.")

            # Take only the portion belonging to this impression.
            needed = current_size - collected
            take = min(needed, len(scores) - position)
            end = position + take

            pending_scores.append(scores[position:end])
            pending_labels.append(labels[position:end])

            collected += take
            position = end

            if collected == current_size:
                impression_scores = np.concatenate(pending_scores)
                impression_labels = np.concatenate(pending_labels)

                if not np.isfinite(impression_scores).all():
                    raise ValueError("Non-finite predicted scores found.")

                if current_size == 1:
                    value_k5 = value_k10 = float(
                        impression_labels[0] == 1
                    )
                else:
                    value_k5 = ndcg_score(
                        impression_labels.reshape(1, -1),
                        impression_scores.reshape(1, -1),
                        k=5,
                        ignore_ties=False,
                    )

                    value_k10 = ndcg_score(
                        impression_labels.reshape(1, -1),
                        impression_scores.reshape(1, -1),
                        k=10,
                        ignore_ties=False,
                    )

                total_k5 += value_k5
                total_k10 += value_k10
                count += 1

                if count % 5_000 == 0:
                    print(
                        f"Evaluated {count:,}/{len(validation_data):,} | "
                        f"NDCG@5: {total_k5 / count:.4f} | "
                        f"NDCG@10: {total_k10 / count:.4f}",
                        flush=True,
                    )

                # Start collecting the next impression.
                pending_scores = []
                pending_labels = []
                collected = 0

                current_size = next(impression_sizes, None)

                if current_size == 0:
                    raise ValueError("An impression has no candidates.")

    if collected != 0 or current_size is not None:
        raise ValueError("Evaluation ended before all candidates were scored.")

    if count != len(validation_data):
        raise ValueError("Not all validation impressions were evaluated.")

    return {
        "ndcg@5": total_k5 / count,
        "ndcg@10": total_k10 / count,
        "impressions_evaluated": count,
        "candidates_evaluated": candidates_seen,
    }



tf.keras.utils.set_random_seed(42)

train_dataset = make_dataset(training_data)
validation_dataset = make_dataset(validation_rows)

# create csv files from the data frames above
export_dir = base_dir / "mind_large_dataset_csv"
export_dir.mkdir(parents=True, exist_ok=True)

dataframes = {
    "news.csv": news,
    "training_behaviors.csv": training_data,
    "validation_behaviors.csv": validation_rows,
}

#
for filename, df in dataframes.items():
    path = export_dir / filename
    if path.is_file():
        print("Already exists:", path)
        continue
    df.to_csv(path, index=False, encoding="utf-8")
    print("Saved:", path)


print("CSV directory:", export_dir)

model_path = output_dir / "news_recommender.keras"

recommender = NewsRecommender()

if model_path.is_file():
    print("Loading saved model...")

    recommender.model = tf.keras.models.load_model(
        model_path,
        compile=False,
    )
else:
    print("No saved model found. Training...")

    history = recommender.fit(
        train_dataset,
        validation_dataset,
    )

    recommender.model.save(model_path)
    print("Model saved:", model_path)


print("Training impressions:", len(training_data))
print("Validation impressions:", len(validation_rows))

assert len(training_data) == 1_801_231
assert len(validation_rows) == 431_517


row = validation_rows.iloc[0]

recommendations = get_rec(
    user_id=row["user_id"],
    impression_id=row["impression_id"],
    model=recommender,
    n=15,
)

print(recommendations.to_string(index=False))

metrics = evaluate_model(
    model=recommender,
    validation_data=validation_rows,
    batch_size=4096,
)

print("\nFinal validation results:")
print(metrics)

(output_dir / "news_mlp_3feat_internal_val.json").write_text(json.dumps(metrics, indent=2))


# Dev has articles that are not in train's news.tsv: add them to the lookup.
dev_news = pd.read_csv(dev_dir / "news.tsv", sep="\t", header=None, names=news_columns,
                       dtype=str, keep_default_na=False)
for r in dev_news.itertuples(index=False):
    article_lookup.setdefault(r.article_id, {"category": r.category, "subcategory": r.subcategory})

dev_behaviors = pd.read_csv(dev_dir / "behaviors.tsv", sep="\t", header=None, names=behavior_columns,
                            dtype=str, keep_default_na=False)


def score_behaviors(model, behavior_data, batch_size=4096):
    """One score per candidate, in file order (the order mind_eval expects)."""
    parts = []
    for features, _ in generate_batches(behavior_data, batch_size):
        parts.append(model.model(features, training=False).numpy().reshape(-1))
    return np.concatenate(parts)


dev = me.load_impressions("dev")
dev_scores = score_behaviors(recommender, dev_behaviors)
assert len(dev_scores) == dev.n_candidates
me.evaluate(dev, dev_scores, name="news_mlp_3feat",
            notes="MLP on history_size, category_match, subcategory_match; trained on train before 2019-11-14")