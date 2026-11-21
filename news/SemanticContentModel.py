
from pathlib import Path
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import semantic_search

from sklearn.metrics import ndcg_score
from tqdm import tqdm
from lightgbm import LGBMRanker
base_dir = Path(__file__).resolve().parents[1]  # project root (this file is in news/)

import json
from functools import partial
from collections import Counter

import mind_eval as me
import popularity_features as pf




POP_FEATURES = ["shown_1h", "shown_6h", "shown_24h", "shown_all", "prev_shows", "prev_clicks",
                "prev_ctr", "cum_shows", "cum_clicks", "cum_ctr"]
# The 5 columns build_features() makes, in its column order (same as the saved X_train).
MINILM_FEATURES = ["top5_similarity", "max_similarity", "avg_profile_similarity",
                   "category_preference", "history_length"]


news_path = (
    base_dir
    / "data"
    / "Mind_Large_Data"
    / "mind_large_dataset_csv"
    / "news.csv"
)

news = pd.read_csv(
    news_path,
    usecols=[
        "article_id", "title", "abstract", "category",
        "title_entities", "abstract_entities",
    ],
    dtype=str,
    keep_default_na=False,
)

def get_entity_names(title_entities, abstract_entities):
    names = []

    for entity_json in (title_entities, abstract_entities):
        entities = json.loads(entity_json or "[]") or []

        for entity in entities:
            name = (entity.get("Label") or "").strip()

            if name:
                names.append(name)

    # Remove duplicate names, preserving their order.
    return "; ".join(dict.fromkeys(names))


news["entity_text"] = [
    get_entity_names(title_entities, abstract_entities)
    for title_entities, abstract_entities in zip(
        news["title_entities"],
        news["abstract_entities"],
    )
]

news["text"] = news["title"] + " " + news["abstract"]

news["text"] += news["entity_text"].map(
    lambda names: " Entities: " + names if names else ""
)

news_by_id = news.set_index(
    "article_id",
    verify_integrity=True,
)

# Inspect the actual text that will be encoded.
print(news["text"].iloc[0])


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


model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device="cpu",
)
news_texts = news["text"].tolist()

# corpus_embeddings = model.encode(
#     news_texts,
#     batch_size=16,
#     show_progress_bar=True,
#     convert_to_tensor=True,
# )
#
# print("Embedding shape:", corpus_embeddings.shape)

def save () :
    output_dir = (
            base_dir / "output" / "minilm_title_abstract_entities_mine"
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    # Save embeddings as a NumPy array.
    np.save(
        output_dir / "embeddings.npy",
        corpus_embeddings.cpu().numpy(),
        allow_pickle=False,
    )

    # Save article IDs in the same order as the embeddings.
    article_ids = news["article_id"].tolist()

    with (output_dir / "article_ids.json").open(
            "w", encoding="utf-8"
    ) as file:
        json.dump(article_ids, file, ensure_ascii=False)

    # Save the encoding settings.
    metadata = {
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "text_fields": [
            "title", "abstract",
            "title_entities.Label", "abstract_entities.Label",
        ],
        "entity_prefix": " Entities: ",
        "entity_separator": "; ",
    }

    with (output_dir / "metadata.json").open(
            "w", encoding="utf-8"
    ) as file:
        json.dump(metadata, file, indent=2)

    print("Saved to:", output_dir)

def load():
    output_dir = (
            base_dir / "output" / "minilm_title_abstract_entities_mine"
    )

    # Load the embedding matrix.
    embeddings = np.load(
        output_dir / "embeddings.npy",
        allow_pickle=False,
    )

    # Load article IDs in their saved order.
    with (output_dir / "article_ids.json").open(
        "r", encoding="utf-8"
    ) as file:
        article_ids = json.load(file)

    # Load encoding settings.
    with (output_dir / "metadata.json").open(
        "r", encoding="utf-8"
    ) as file:
        metadata = json.load(file)

    if embeddings.shape[0] != len(article_ids):
        raise ValueError("Embedding rows and article IDs do not match.")

    # Rebuild article ID → embedding row.
    article_to_index = {}

    for index, article_id in enumerate(article_ids):
        article_to_index[article_id] = index

    return embeddings, article_ids, article_to_index, metadata





def score(history_ids, candidate_ids , article_to_index):
    scores = np.zeros(len(candidate_ids), dtype=np.float32)

    if not candidate_ids:
        return scores

    history_indices = []

    for article_id in dict.fromkeys(history_ids):
        if article_id in article_to_index:
            history_indices.append(article_to_index[article_id])

    if not history_indices:
        return scores

    history_embeddings = corpus_embeddings[history_indices]

    candidate_indexes = []
    known_positions = []

    for position, article_id in enumerate(candidate_ids):
        if article_id in article_to_index:
            candidate_indexes.append(article_to_index[article_id])
            known_positions.append(position)

    if not candidate_indexes:
        return scores

    candidate_embeddings = corpus_embeddings[candidate_indexes]

    avg_history_embeddings = np.mean(
        history_embeddings,
        axis=0,
        keepdims=True,
    )

    similarity = model.similarity(
        avg_history_embeddings,
        candidate_embeddings,
    )

    scores[known_positions] = similarity.cpu().numpy().ravel()

    return scores





def evaluate_semantic(validation_rows, article_to_index, score_fn=score):
    total_k5 = 0.0
    total_k10 = 0.0
    count = 0

    for row in tqdm(
        validation_rows.itertuples(index=False),
        total=len(validation_rows),
        desc="Evaluating LightGBM",
    ):
        history_ids = row.history.split()

        candidate_ids = []
        click_labels = []

        for entry in row.impressions.split():
            article_id, label = entry.rsplit("-", 1)
            candidate_ids.append(article_id)
            click_labels.append(int(label))

        scores = score_fn(history_ids, candidate_ids, article_to_index)

        if scores.shape != (len(candidate_ids),):
            raise ValueError("Incorrect score shape.")

        if not np.isfinite(scores).all():
            raise ValueError("Scores contain NaN or infinity.")

        y_true = np.asarray(
            click_labels, dtype=np.float32
        ).reshape(1, -1)

        y_score = np.asarray(
            scores, dtype=np.float32
        ).reshape(1, -1)

        total_k5 += ndcg_score(y_true, y_score, k=5)
        total_k10 += ndcg_score(y_true, y_score, k=10)
        count += 1

    if count == 0:
        raise ValueError("No validation impressions to evaluate.")

    return {
        "ndcg@5": total_k5 / count,
        "ndcg@10": total_k10 / count,
    }

def score_history_by_candidate_comparison (history_ids, candidate_ids , article_to_index , top_k = 3 ) :
    scores = np.zeros(len(candidate_ids), dtype=np.float32)

    if not candidate_ids:
        return scores

    history_indices = []

    for article_id in dict.fromkeys(history_ids):
        if article_id in article_to_index:
            history_indices.append(article_to_index[article_id])

    if not history_indices:
        return scores

    history_embeddings = corpus_embeddings[history_indices]

    candidate_indexes = []
    known_positions = []

    for position, article_id in enumerate(candidate_ids):
        if article_id in article_to_index:
            candidate_indexes.append(article_to_index[article_id])
            known_positions.append(position)

    if not candidate_indexes:
        return scores

    candidate_embeddings = corpus_embeddings[candidate_indexes]

    similarities = model.similarity(
        history_embeddings,
        candidate_embeddings,
    )

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    similarities_array = similarities.cpu().numpy()

    # Use fewer matches if the history contains fewer than top_k articles.
    k = min(top_k, len(history_indices))

    # Select the strongest k matches without sorting the entire history.
    top_similarities = np.partition(
        similarities_array,
        kth=len(history_indices) - k,
        axis=0,
    )[-k:, :]

    # Sort only these selected matches to preserve the averaging order.
    top_similarities.sort(axis=0)

    # Produce one score per known candidate.
    candidate_scores = top_similarities.mean(axis=0)

    # Preserve candidate order; unknown articles keep score zero.
    scores[known_positions] = candidate_scores

    return scores



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


def score_semantic_with_categories(history_ids,candidate_ids,article_to_index,category_weight=0.1) :
    semantic_scores = score_history_by_candidate_comparison(history_ids, candidate_ids , article_to_index , top_k = 5 )
    category_scores = score_categories(history_ids, candidate_ids)

    combined_scores = (1 - category_weight) * semantic_scores + (category_weight * category_scores )

    return combined_scores


def build_features(history_ids, candidate_ids, article_to_index):
    top5_scores = score_history_by_candidate_comparison(
        history_ids,
        candidate_ids,
        article_to_index,
    top_k=5,)

    max_scores = score_history_by_candidate_comparison(
        history_ids,
        candidate_ids,
        article_to_index,
    top_k=1,)

    avg_profile_scores = score(
        history_ids,
        candidate_ids,
        article_to_index, )

    category_scores = score_categories(history_ids, candidate_ids)

    features_df = pd.DataFrame({
        "article_id": candidate_ids,
        "top5_similarity": top5_scores,
        "max_similarity": max_scores,
        "avg_profile_similarity": avg_profile_scores,
        "category_preference": category_scores,
        "history_length": len(dict.fromkeys(history_ids)),
    })

    return features_df
"""Replacement for prepare_training_data in SemanticContentModel.py.

Copy the function into that script; it uses its existing build_features and
base_dir. Change the cache directory/version whenever inputs or features change.
"""



def prepare_training_data(training_df, article_to_index, output_dir=None):
    feature_names = [
        "top5_similarity",
        "max_similarity",
        "avg_profile_similarity",
        "category_preference",
        "history_length",
    ]
    if training_df.empty:
        raise ValueError("No training impressions to prepare.")

    if output_dir is None:
        output_dir = base_dir / "output" / "lightgbm_full_features_entities_v1"
    output_dir = Path(output_dir)
    complete_path = output_dir / "complete.json"

    # Reuse a FINISHED cache. Inputs and feature definitions must be unchanged.
    if complete_path.exists():
        with complete_path.open("r", encoding="utf-8") as file:
            cache_info = json.load(file)
        if (
            cache_info["impressions"] != len(training_df)
            or cache_info["feature_names"] != feature_names
        ):
            raise ValueError("Cache does not match. Use a new output directory.")

        print("Loading saved training features...", flush=True)
        X_train = np.load(output_dir / "X_train.npy", mmap_mode="r", allow_pickle=False)
        y_train = np.load(output_dir / "y_train.npy", mmap_mode="r", allow_pickle=False)
        groups = np.load(output_dir / "groups.npy", allow_pickle=False)
        if (
            X_train.shape != (cache_info["candidates"], len(feature_names))
            or y_train.shape != (cache_info["candidates"],)
            or groups.shape != (len(training_df),)
            or int(groups.sum(dtype=np.int64)) != len(y_train)
        ):
            raise ValueError("Saved training arrays do not match.")
        return X_train, y_train, groups

    # Do not overwrite an existing or interrupted cache.
    if output_dir.exists():
        raise FileExistsError(
            "This cache folder is incomplete. Use a new output directory; "
            "this version does not resume interrupted preparation."
        )
    output_dir.mkdir(parents=True, exist_ok=False)

    groups = np.empty(len(training_df), dtype=np.int32)
    for position, entries in enumerate(tqdm(
        training_df["impressions"],
        total=len(training_df),
        desc="Counting candidates",
        unit="impression",
    )):
        groups[position] = len(entries.split())
    if np.any(groups == 0):
        raise ValueError("Training data contains an empty impression.")
    total_candidates = int(groups.sum(dtype=np.int64))
    print(f"Creating disk-backed arrays for {total_candidates:,} candidates...", flush=True)

    X_train = np.lib.format.open_memmap(
        output_dir / "X_train.npy",
        mode="w+",
        dtype=np.float32,
        shape=(total_candidates, len(feature_names)),
    )
    y_train = np.lib.format.open_memmap(
        output_dir / "y_train.npy",
        mode="w+",
        dtype=np.int32,
        shape=(total_candidates,),
    )
    np.save(output_dir / "groups.npy", groups, allow_pickle=False)

    offset = 0
    for row in tqdm(
        training_df.itertuples(index=False),
        total=len(training_df),
        desc="Preparing training features",
        unit="impression",
    ):
        history_ids = row.history.split()
        candidate_ids = []
        click_labels = []
        for entry in row.impressions.split():
            article_id, label = entry.rsplit("-", 1)
            candidate_ids.append(article_id)
            click_labels.append(int(label))

        features_df = build_features(history_ids, candidate_ids, article_to_index)
        X_impression = features_df[feature_names].to_numpy(dtype=np.float32)
        if X_impression.shape != (len(candidate_ids), len(feature_names)):
            raise ValueError("Incorrect feature shape.")
        if not np.isfinite(X_impression).all():
            raise ValueError("Features contain NaN or infinity.")
        if features_df["article_id"].tolist() != candidate_ids:
            raise ValueError("Features do not preserve candidate order.")

        end = offset + len(candidate_ids)
        X_train[offset:end] = X_impression
        y_train[offset:end] = click_labels
        offset = end

    assert offset == total_candidates
    print("Finishing feature files...", flush=True)
    X_train.flush()
    y_train.flush()

    # Mark the cache reusable only after all rows have been successfully written.
    cache_info = {
        "impressions": len(training_df),
        "candidates": total_candidates,
        "feature_names": feature_names,
    }
    with complete_path.open("w", encoding="utf-8") as file:
        json.dump(cache_info, file, indent=2)

    print("Training features saved to:", output_dir, flush=True)
    return X_train, y_train, groups

def train_ranker(X_train, y_train, groups, feature_names="auto"):
    ranker = LGBMRanker(
        objective="lambdarank",
        n_estimators=100,
        learning_rate=0.05,
        num_leaves=15,
        random_state=42,
    )

    with tqdm(
            total=ranker.n_estimators,
            desc="Training LightGBM",
            unit="tree",
    ) as progress:
        def update_progress(env):
            progress.update(1)

        ranker.fit(
            X_train,
            y_train,
            group=groups,
            feature_name=feature_names,
            callbacks=[update_progress],
        )

    return ranker

def score_lightgbm(
    history_ids, candidate_ids, article_to_index, ranker, extra_features=None
):
    """extra_features: this impression's popularity rows (len(candidate_ids) x len(POP_FEATURES)), or None."""
    if not candidate_ids:
        return np.zeros(0, dtype=np.float32)

    features_df = build_features(
        history_ids,
        candidate_ids,
        article_to_index,
    )

    X = features_df.drop(
        columns=["article_id"]
    ).to_numpy(dtype=np.float32)

    if extra_features is not None:
        X = np.hstack([X, extra_features])

    X = pd.DataFrame(X, columns=ranker.feature_name_)

    return ranker.predict(
        X,
        validate_features=True,
    ).astype(np.float32)
emb_file = base_dir / "output" / "minilm_title_abstract_entities_mine" / "embeddings.npy"
if not emb_file.exists():
    corpus_embeddings = model.encode(news_texts, batch_size=64, show_progress_bar=True,
                                     convert_to_tensor=True)
    save()

corpus_embeddings, article_ids, article_to_index, metadata = load()

# Select one validation impression.
row = validation_data.iloc[0]
history_ids = row["history"].split()

candidate_ids = []
click_labels = []

for entry in row["impressions"].split():
    article_id, label = entry.rsplit("-", 1)
    candidate_ids.append(article_id)
    click_labels.append(int(label))

scores = score_history_by_candidate_comparison(
    history_ids,
    candidate_ids,
    article_to_index,
    top_k=5,
)

# Check that scores match the candidates and contain valid numbers.
assert scores.shape == (len(candidate_ids),)
assert np.isfinite(scores).all()

print("Candidates:", len(candidate_ids))
print("Returned scores:", len(scores))
print("Checks passed.")

results = pd.DataFrame({
    "article_id": candidate_ids,
    "score": scores,
    "clicked": click_labels,
})

results["title"] = results["article_id"].map(news_by_id["title"])

print(
    results.sort_values("score", ascending=False)
    .head(10)
    .to_string(index=False)
)

evaluation_sample = validation_data.sample(
    n=10_000,
    random_state=42,
)

# for weight in [0.05]:
#     scoring_function = partial(
#         score_semantic_with_categories,
#         category_weight=weight,
#     )
#
#     metrics = evaluate_semantic(
#         validation_data,
#         article_to_index,
#         score_fn=scoring_function,
#     )
#
#     print(
#         f"Category weight={weight:.2f} | "
#         f"NDCG@5={metrics['ndcg@5']:.6f} | "
#         f"NDCG@10={metrics['ndcg@10']:.6f}"
#     )

features_df = build_features(
    history_ids,
    candidate_ids,
    article_to_index,
)

print(features_df.head().to_string(index=False))

X = features_df.drop(
    columns=["article_id"]
).to_numpy(dtype=np.float32)

assert X.shape == (len(candidate_ids), 5)
assert np.isfinite(X).all()
assert features_df["article_id"].tolist() == candidate_ids

print("Feature checks passed:", X.shape)


training_sample = training_data
X_train, y_train, groups = prepare_training_data(
    training_sample,
    article_to_index,
)


# build the mask : load the train impression mask and compare its times with NOV 14 00:00 then delete the impression obj

train_impressions = me.load_impressions("train")
train_mask = train_impressions.times < np.datetime64("2019-11-14T00:00:00")

del train_impressions  # free memory

pop_train = pf.load_feature_matrix("train", POP_FEATURES, impression_mask=train_mask)
assert len(pop_train) == len(X_train)

# 5 MiniLM columns + 10 popularity columns, side by side (same rows, same order).
X_all = np.hstack([X_train, pop_train])
del pop_train

ranker = train_ranker(X_all, y_train, groups, feature_names=MINILM_FEATURES + POP_FEATURES)

print("LightGBM training finished.")
print("Features:", X_all.shape)
print("Labels:", y_train.shape)
print("Impression groups:", len(groups))
del X_all

model_path = base_dir / "output" / "news" / "lgbm_minilm_pop15.txt"
ranker.booster_.save_model(str(model_path))
print("Ranker saved:", model_path)

importance = pd.Series(ranker.booster_.feature_importance("gain"), index=ranker.feature_name_)
print("\nFeature importance (gain share):")
print((importance / importance.sum()).sort_values(ascending=False).round(4).to_string())


# Internal Nov-14 split, full run on 2026-11-06 (sklearn nDCG), kept for reference:
(base_dir / "output" / "news" / "lgbm_minilm_5feat_internal_val.json").write_text(
    json.dumps({"ndcg@5": 0.325031, "ndcg@10": 0.383189, "impressions": 431517}, indent=2))

# ---- official dev set ----
dev_dir = base_dir / "data" / "Mind_Large_Data" / "MINDlarge_dev" / "MINDlarge_dev"
news_cols = ["article_id", "category", "subcategory", "title", "abstract", "url",
             "title_entities", "abstract_entities"]
dev_news = pd.read_csv(dev_dir / "news.tsv", sep="\t", header=None, names=news_cols,
                       dtype=str, keep_default_na=False)

# 2,624 dev articles are not in train's news: embed them and add their categories.
new = dev_news[~dev_news["article_id"].isin(article_to_index)].copy()
new["text"] = (new["title"] + " " + new["abstract"]
               + [(" Entities: " + n if n else "")
                  for n in map(get_entity_names, new["title_entities"], new["abstract_entities"])])
new_emb = model.encode(new["text"].tolist(), batch_size=64, show_progress_bar=True)
start = len(corpus_embeddings)
corpus_embeddings = np.vstack([corpus_embeddings, new_emb])
for k, aid in enumerate(new["article_id"]):
    article_to_index[aid] = start + k
article_to_category.update(zip(new["article_id"], new["category"]))

dev_behaviors = pd.read_csv(dev_dir / "behaviors.tsv", sep="\t", header=None,
                            names=["impression_id", "user_id", "timestamp", "history", "impressions"],
                            dtype=str, keep_default_na=False)
dev = me.load_impressions("dev")

# Check the order: dev_behaviors and dev (from mind_eval) are both in file order, so row i of
# dev_behaviors is impression i, and its popularity rows are dev.offsets[i]:dev.offsets[i + 1].
assert len(dev_behaviors) == dev.n_impressions
assert np.array_equal(dev_behaviors["impression_id"].astype(np.int64).to_numpy(), dev.ids)
assert np.array_equal(dev_behaviors["impressions"].str.split().str.len().to_numpy(), dev.sizes)

pop_dev = pf.load_feature_matrix("dev", POP_FEATURES)
assert len(pop_dev) == dev.n_candidates

dev_scores = [
    score_lightgbm(row.history.split(),
                   [e.rsplit("-", 1)[0] for e in row.impressions.split()],
                   article_to_index, ranker,
                   extra_features=pop_dev[dev.offsets[i]:dev.offsets[i + 1]])
    for i, row in enumerate(tqdm(dev_behaviors.itertuples(index=False), total=len(dev_behaviors),
                                 desc="Scoring dev"))
]
me.evaluate(dev, dev_scores, name="lgbm_minilm_pop15",
            notes="LightGBM lambdarank on 5 MiniLM/category features + 10 popularity features "
                  "(showings, earlier-day clicks and click rates), trained on train before 2019-11-14")
