import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'     # suppress TensorFlow C++ logs
warnings.filterwarnings('ignore')              # suppress Python warnings

import tensorflow as tf
tf.get_logger().setLevel('ERROR')             # suppress TF Python logs

from recommenders.utils.timer import Timer
from recommenders.models.ncf.ncf_singlenode import NCF
from recommenders.models.ncf.dataset import Dataset as NCFDataset
from recommenders.datasets import movielens
from recommenders.datasets.python_splitters import python_chrono_split, python_random_split
from recommenders.evaluation.python_evaluation import (
    map, ndcg_at_k, precision_at_k, recall_at_k
)
from recommenders.utils.constants import SEED as DEFAULT_SEED
from recommenders.utils.notebook_utils import store_metadata

# top k items to recommend
TOP_K = 10

data = pd.read_csv("data/data/processed/ratings.csv")
# data = data.sample(n=100000, random_state=42)  # to lower the number of rows , the time of training and evaluation will be much lower now


# rename to NCFDataset expected format
data = data.rename(columns={
    'user_id':  'userID',
    'movie_id': 'itemID'
})

EPOCHS     = 30    # start low , the lowest this number the less is the time it gonna take
BATCH_SIZE = 256
SEED       = DEFAULT_SEED

train, test = python_random_split(data, ratio=0.75, seed=SEED)

# filter test to only users AND items the model was trained on
train_users = set(train['userID'].unique())
train_items = set(train['itemID'].unique())
test = test[
    test['userID'].isin(train_users) &
    test['itemID'].isin(train_items)
]

# save splits to temp files
train.sort_values('userID').to_csv('tmp_train.csv', index=False)
test.sort_values('userID').to_csv('tmp_test.csv',   index=False)


# wrap in NCFDataset
# now pass file paths — not DataFrames
dataset = NCFDataset(
    train_file='tmp_train.csv',
    test_file='tmp_test.csv',
    seed=SEED
)


model = NCF(
    n_users=dataset.n_users,
    n_items=dataset.n_items,
    model_type="NeuMF",
    n_factors=4,
    layer_sizes=[16, 8, 4],
    n_epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    learning_rate=1e-3,
    verbose=1, # print every epoch
    seed=SEED
)

n_users=dataset.n_users
n_items=dataset.n_items

print(f"Users: {n_users}, Items: {n_items}")


model.fit(dataset)

# method 1 slow
# predictions = []
# for _ , row in test.iterrows():
#     user = row['user_id']
#     movie = row['item_id']
#
#     prediction = model.predict(user ,movie )
#     predictions.append([user, movie, prediction])

# method 2 faster
predictions = model.predict(
    test['userID'].tolist(),
    test['itemID'].tolist(),
    is_list=True
)

predictions = pd.DataFrame({
    'userID':     test['userID'].values,
    'itemID':     test['itemID'].values,
    'prediction': predictions
})
print('predictions:')
print(predictions.head())


# Generic Evaluation
# We remove rated movies in the top k recommendations To compute ranking metrics, we need predictions on all user
# item pairs. We remove though the items already watched by the user, since we choose not to recommend them again.

with Timer() as test_time:
    # to avoid MemoryError
    # You do not need all 10,096 users to get meaningful metrics. Sample 500
    # sample users for evaluation — representative but memory safe
    # 500 users × 1,289 items = 644,500 rows : manageable
    eval_users = np.random.choice(train.userID.unique(), size=500, replace=False)

    users, items, preds = [], [], []
    item = list(train.itemID.unique())
    for user in eval_users:
        user = [user] * len(item)
        users.extend(user)
        items.extend(item)
        preds.extend(list(model.predict(user, item, is_list=True)))

    all_predictions = pd.DataFrame(data={"userID": users, "itemID":items, "prediction":preds})

    merged = pd.merge(train, all_predictions, on=["userID", "itemID"], how="outer")
    all_predictions = merged[merged.rating.isnull()].drop('rating', axis=1)

print("Took {} seconds for prediction.".format(test_time.interval))

# Also filter test to only those sampled users so the evaluation matches:
test_eval = test[test['userID'].isin(eval_users)]

eval_map = map(test_eval, all_predictions, col_prediction='prediction', k=TOP_K)
eval_ndcg = ndcg_at_k(test_eval, all_predictions, col_prediction='prediction', k=TOP_K)
eval_precision = precision_at_k(test_eval, all_predictions, col_prediction='prediction', k=TOP_K)
eval_recall = recall_at_k(test_eval, all_predictions, col_prediction='prediction', k=TOP_K)

print("MAP:\t%f" % eval_map,
      "NDCG:\t%f" % eval_ndcg,
      "Precision@K:\t%f" % eval_precision,
      "Recall@K:\t%f" % eval_recall, sep='\n')


for f in ['tmp_train.csv', 'tmp_test.csv', 'tmp_test_full.csv']:
    if os.path.exists(f): os.remove(f)