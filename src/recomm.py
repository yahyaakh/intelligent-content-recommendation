import json
import sys
import time

import faiss
import requests
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Surprise library for advanced collaborative filtering
from surprise import Dataset, Reader, SVD
from surprise import PredictionImpossible
from surprise.model_selection import train_test_split
from surprise import accuracy
from collections import defaultdict
from collections import Counter

import pickle

from pathlib import Path

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


def diagnose_user_movie_matrix(collab_rec):
    user_movie_matrix = collab_rec.user_movie_matrix
    if user_movie_matrix.empty:
        print("User-movie rating matrix is empty.")
        return
    print(f"User-movie matrix shape: {user_movie_matrix.shape}")
    rated_counts = (user_movie_matrix > 0).sum(axis=1)
    print(f"Avg movies rated per user: {rated_counts.mean():.2f}")
    print(f"Min movies rated by any user: {rated_counts.min()}")
    print(f"Max movies rated by any user: {rated_counts.max()}")
    print(f"User similarity matrix shape: {collab_rec.user_similarity_df.shape}")
    print(collab_rec.user_similarity_df.describe())

class MovieRecommendationSystem:
    def __init__(self, movies_df, users_df, interactions_df):
        self.movies_df = movies_df.copy()
        self.users_df = users_df.copy()
        self.interactions_df = interactions_df.copy()
        self.movie_popularity = self._calculate_popularity()
    def _calculate_popularity(self):
        pop = self.interactions_df.groupby('movie_id').agg({'rating': ['mean', 'count'], 'view_count': 'sum', 'completion_rate': 'mean'}).fillna(0)
        pop.columns = ['avg_rating', 'rating_count', 'total_views', 'avg_completion']
        pop['score'] = pop['avg_rating']*0.3 + np.log1p(pop['rating_count'])*0.3 + np.log1p(pop['total_views'])*0.2 + pop['avg_completion']*0.2
        return pop.sort_values('score', ascending=False)
    def get_popular_movies(self, n=10):
        top_ids = self.movie_popularity.head(n).index
        top = self.movies_df[self.movies_df['movie_id'].isin(top_ids)].drop_duplicates('movie_id').set_index('movie_id', drop=False)
        # keep the popularity order (isin alone returns the films in file order)
        ordered = [mid for mid in top_ids if mid in top.index]
        return top.loc[ordered, ['movie_id','title','genres','director','year']].to_dict('records')

class ContentBasedRecommender:
    def __init__(self, movies_df):
        self.movies_df = movies_df.copy()
        # --- PIPELINE 1: HARD METADATA ---
        # Initialize CountVectorizer()

        self.count_vec = CountVectorizer()
        # count_matrix = fit and transform the 'metadata_features' column
        self.meta_matrix = self.count_vec.fit_transform(self.movies_df['metadata_features'])
        # meta_sim = calculate cosine_similarity(count_matrix, count_matrix)

        # self.meta_sim = cosine_similarity(self.meta_matrix, self.meta_matrix)


        # --- PIPELINE 2: SOFT TEXT ---
        self.tfidf = TfidfVectorizer(stop_words='english')

        # tfidf_matrix = fit and transform the 'text_features' column

        self.text_matrix = self.tfidf.fit_transform(self.movies_df['text_features'].fillna(''))
        # text_sim = calculate cosine_similarity(tfidf_matrix, tfidf_matrix)

        # self.text_sim = cosine_similarity(self.text_matrix, self.text_matrix)

        # --- THE BLEND ---
        # Give Metadata 75% of the voting power, and Plot text 25%
        # self.cosine_sim = (self.meta_sim * 0.75) + (self.text_sim * 0.25)


        ### explain this function that is responsible for finding the top n movies most similar to the movie we choose
    def get_similar_movies(self, movie_id, n=10):

        ## find the movie : search the pandas df movies_df for the row containing the movie_id passed in.
        idxs = self.movies_df[self.movies_df['movie_id'] == movie_id].index
        if len(idxs) == 0: return []
        idx = idxs[0]

        similarities = (
                0.75 * cosine_similarity(self.meta_matrix[idx], self.meta_matrix).ravel()
                + 0.25 * cosine_similarity(self.text_matrix[idx], self.text_matrix).ravel()
        )

        sim_scores = [
            (i, value)
            for i, value in enumerate(similarities)
            if i != idx
        ]
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[:n]
        movie_idxs = [i[0] for i in sim_scores]
        return self.movies_df.iloc[movie_idxs][['movie_id','title','genres','director','year']].to_dict('records')


# The collaborative recommender core idea:
#
# "Who else has similar taste to this user, and what did they like that this user hasn't seen yet?"
#
# It never looks at what a movie is about. It doesn't care about genres, directors, or plot. It only looks at the pattern of ratings across users.

class CollaborativeRecommender:
    def __init__(self, interactions_df):
        self.interactions_df = interactions_df.copy()

        # Takes the ratings dataframe and makes a copy of it.
        # The .copy() is important — it prevents any changes inside this class from accidentally modifying the original dataframe outside.


        # A safety check. If there are no ratings at all, create empty placeholders and exit early.
        # Without this, the pivot table below would crash on empty data.

        if len(interactions_df) == 0:
            self.user_movie_matrix = pd.DataFrame()
            self.user_similarity_df = pd.DataFrame()
            return


        self.user_movie_matrix = self.interactions_df.pivot(index='user_id',columns='movie_id',values='rating').fillna(0)

        rated_mask = self.user_movie_matrix > 0
        self.user_means = self.user_movie_matrix.replace(0, np.nan).mean(axis=1)
        self.centered_matrix = self.user_movie_matrix.copy().astype(float)
        self.centered_matrix[rated_mask] = self.centered_matrix[rated_mask].sub(
            self.user_means, axis=0
        )

        self.user_similarity = cosine_similarity(self.centered_matrix)


        self.user_similarity_df = pd.DataFrame(self.user_similarity,index=self.user_movie_matrix.index,columns=self.user_movie_matrix.index)


    def get_user_recommendations(self, user_id, n=10):
        if user_id not in self.user_movie_matrix.index: return []
        user_similarities = self.user_similarity_df[user_id].sort_values(ascending=False)
        # print(f"DEBUG user_similarities are {user_similarities}")


        similar_users = user_similarities.iloc[1:6].index
        # print(f"DEBUG similar_users are {similar_users}")

        user_movies = set(self.user_movie_matrix.loc[user_id][self.user_movie_matrix.loc[user_id]>0].index)
        # print(f"DEBUG user_movies are {user_movies}")


        recommendations = {}
        for user in similar_users:
            for movie_id, centered_rating in self.centered_matrix.loc[user].items():

                if centered_rating > 0 and movie_id not in user_movies:
                    recommendations.setdefault(movie_id,[]).append(centered_rating * user_similarities[user])
        avg_recs = {mid:np.mean(rates) for mid,rates in recommendations.items()}
        sorted_recs = sorted(avg_recs.items(),key=lambda x:x[1],reverse=True)
        return [m for m,s in sorted_recs[:n]]

class SurpriseRecommender:
    def __init__(self, interactions_df):
        self.interactions_df = interactions_df.copy()
        reader = Reader(rating_scale=(0.5, 5))  # ML-25M half stars
        data = Dataset.load_from_df(interactions_df[['user_id', 'movie_id', 'rating']], reader)
        trainset = data.build_full_trainset()
        self.algo = SVD(n_epochs=15, random_state=42)
        self.algo.fit(trainset)
        self.user_ids = interactions_df['user_id'].unique()
        self.movie_ids = interactions_df['movie_id'].unique()
    def get_user_recommendations(self, user_id, n=10):
        rated_movies = set(self.interactions_df[self.interactions_df['user_id']==user_id]['movie_id'])
        candidates = [mid for mid in self.movie_ids if mid not in rated_movies]
        predictions = []
        for mid in candidates:
            try:
                pred = self.algo.predict(int(user_id), int(mid))
                predictions.append((mid, pred.est))
            except PredictionImpossible:
                continue
        predictions.sort(key=lambda x: x[1], reverse=True)
        top_movie_ids = [mid for mid,_ in predictions[:n]]
        return top_movie_ids


class NCFRecommender:
    """
    Neural Collaborative Filtering (NeuMF) recommender.
    Trains once and saves to disk. Loads the saved model on subsequent runs
    to avoid the ~50 min retraining cost.
    """
    def __init__(self, interactions_df, movies_df, model_dir='output/models/ncf_saved',
                 epochs=30, force_retrain=False):
        self.interactions_df = interactions_df.copy()
        self.movies_df = movies_df.copy()
        self.model_dir = model_dir
        self.epochs = epochs
        self.model = None
        self.user2id = {}
        self.item2id = {}
        self.n_users = 0
        self.n_items = 0

        mappings_path = os.path.join(model_dir, 'mappings.pkl')

        if os.path.exists(mappings_path) and not force_retrain:
            print("Loading saved NCF model...")
            self._load()
        else:
            print("Training NCF model (first run, may take ~50 min)...")
            self._train_and_save()

    def _prepare_data(self):
        return self.interactions_df.rename(
            columns={'user_id': 'userID', 'movie_id': 'itemID'}
        )

    def _train_and_save(self):
        from recommenders.models.ncf.ncf_singlenode import NCF
        from recommenders.models.ncf.dataset import Dataset as NCFDataset
        from recommenders.datasets.python_splitters import python_random_split
        from recommenders.utils.constants import SEED as DEFAULT_SEED


        data = self._prepare_data()

        # Convert Unix timestamps into dates.
        data["date"] = pd.to_datetime(
            data["timestamp"],
            unit="s",
            errors="coerce",
        )
        data = data.dropna(subset=["date"])

        train, test = python_random_split(
            data,
            ratio=0.75,
            seed=42,
        )

        train_users = set(train["userID"].unique())
        train_items = set(train["itemID"].unique())

        test = test.loc[
            test["userID"].isin(train_users)
            & test["itemID"].isin(train_items)
            ].copy()

        print(f"Train ratings: {len(train):,}")
        print(f"Test ratings (known users/items): {len(test):,}")

        if train.empty or test.empty:
            raise ValueError("Training or test data is empty.")



        for f in ['ncf_train.csv', 'ncf_test.csv', 'ncf_test_full.csv']:
            if os.path.exists(f):
                os.remove(f)

        train.sort_values('userID').to_csv('ncf_train.csv', index=False)
        test.sort_values('userID').to_csv('ncf_test.csv', index=False)

        dataset = NCFDataset(train_file='ncf_train.csv', test_file='ncf_test.csv',
                             seed=DEFAULT_SEED)

        self.model = NCF(
            n_users=dataset.n_users, n_items=dataset.n_items,
            model_type="NeuMF", n_factors=4, layer_sizes=[16, 8, 4],
            n_epochs=self.epochs, batch_size=256, learning_rate=1e-3,
            verbose=1, seed=DEFAULT_SEED
        )
        self.model.fit(dataset)

        self.user2id = self.model.user2id
        self.item2id = self.model.item2id
        self.n_users = dataset.n_users
        self.n_items = dataset.n_items

        os.makedirs(self.model_dir, exist_ok=True)
        self.model.save(dir_name=self.model_dir)

        # save mappings and the held-out test set (for evaluation)
        with open(os.path.join(self.model_dir, 'mappings.pkl'), 'wb') as f:
            pickle.dump({
                'user2id': self.user2id, 'item2id': self.item2id,
                'n_users': self.n_users, 'n_items': self.n_items,
            }, f)
        test.to_csv(os.path.join(self.model_dir, 'test.csv'), index=False)

        print(f"Train ratings: {len(train):,}")
        print(f"Test ratings: {len(test):,}")

        if train.empty or test.empty:
            raise ValueError("Training or test data is empty.")

        for f in ['ncf_train.csv', 'ncf_test.csv', 'ncf_test_full.csv']:
            if os.path.exists(f):
                os.remove(f)

        print("NCF model trained and saved.")

    def _load(self):
        from recommenders.models.ncf.ncf_singlenode import NCF

        with open(os.path.join(self.model_dir, 'mappings.pkl'), 'rb') as f:
            m = pickle.load(f)

        self.user2id = m['user2id']
        self.item2id = m['item2id']
        self.n_users = m['n_users']
        self.n_items = m['n_items']

        self.model = NCF(
            n_users=self.n_users, n_items=self.n_items,
            model_type="NeuMF", n_factors=4, layer_sizes=[16, 8, 4],
            n_epochs=self.epochs, batch_size=256, learning_rate=1e-3,
            verbose=0, seed=42
        )
        self.model.load(neumf_dir=self.model_dir)

        # restore the id mappings onto the model so predict() works
        self.model.user2id = self.user2id
        self.model.item2id = self.item2id
        self.model.id2user = {v: k for k, v in self.user2id.items()}
        self.model.id2item = {v: k for k, v in self.item2id.items()}

        print("NCF model loaded.")

    def get_user_recommendations(self, user_id, n=10):
        if user_id not in self.user2id:
            return []

        all_items = list(self.item2id.keys())
        watched = set(self.interactions_df[
            self.interactions_df['user_id'] == user_id]['movie_id'])
        candidates = [item for item in all_items if item not in watched]

        if not candidates:
            return []

        user_list = [user_id] * len(candidates)
        preds = self.model.predict(user_list, candidates, is_list=True)

        scored = sorted(zip(candidates, preds), key=lambda x: x[1], reverse=True)
        return [int(mid) for mid, _ in scored[:n]]




class HybridRecommenderEnhanced:
    def __init__(self, content_rec, collab_rec, movies_df, users_df, interactions_df, rec_system , surprise_rec):
        self.content_rec = content_rec
        self.collab_rec = collab_rec
        self.movies_df = movies_df
        self.users_df = users_df
        self.interactions_df = interactions_df
        self.rec_system = rec_system

        self.surprise_rec = surprise_rec

    def get_hybrid_recommendations(self, user_id, n=10):
        recommendations = {}
        user_movies = self.interactions_df[self.interactions_df['user_id'] == user_id]

        # cold start — new user
        if len(user_movies) == 0:
            return self.rec_system.get_popular_movies(n)

        # personal mean threshold instead of fixed >= 4
        user_mean = user_movies['rating'].mean()
        user_fav = user_movies[user_movies['rating'] >= user_mean]['movie_id'].tolist()

        # fallback if still empty
        if not user_fav:
            user_fav = user_movies.nlargest(3, 'rating')['movie_id'].tolist()

        # get 20 candidates from collaborative
        collab_recs = self.collab_rec.get_user_recommendations(user_id, 20)

        # ONE loop — both signals calculated and stored together
        for i, movie_id in enumerate(collab_recs):

            # collab signal — normalized 0.0 to 1.0
            collab_score = 20 - i
            collab_normalized = collab_score / 20

            # svd signal — normalized 0.0 to 1.0
            pred = self.surprise_rec.algo.predict(int(user_id), int(movie_id))
            svd_normalized = (pred.est - 0.5) / 4.5  # 0.5-5 scale -> 0-1

            # blend — both signals on equal footing
            hybrid_score = 0.4 * svd_normalized + 0.6 * collab_normalized

            # store everything
            movie_info = self.movies_df[self.movies_df['movie_id'] == movie_id]
            if not movie_info.empty:
                recommendations[movie_id] = {
                    'hybrid_score': hybrid_score,
                    'svd_normalized': svd_normalized,
                    'collab_normalized': collab_normalized,
                    'movie_info': movie_info.iloc[0].to_dict()
                }

        # sort by hybrid score descending
        sorted_recs = sorted(
            recommendations.items(),
            key=lambda x: x[1]['hybrid_score'],
            reverse=True
        )

        seen = set(user_movies['movie_id'].tolist())
        final = []

        for mid, data in sorted_recs:
            if mid not in seen and len(final) < n:
                final.append(data['movie_info'])

        # fallback to popular if not enough results
        if len(final) < n:
            needed = n - len(final)
            popular = self.rec_system.get_popular_movies(needed)
            for pm in popular:
                if pm['movie_id'] not in {m['movie_id'] for m in final}:
                    final.append(pm)
                    if len(final) == n:
                        break

        return final

class RecommendationEvaluator:
    def __init__(self, interactions_df, movies_df):
        self.interactions_df = interactions_df
        self.movies_df = movies_df
    def ndcg_at_k(self, recommended_items, relevant_set, k=10):
        return ndcg_at_k(recommended_items, relevant_set, k)
    def mean_ndcg(self, recommendations_dict, ground_truth_dict, k=10):
        total_ndcg = 0.0
        count = 0
        for user, recs in recommendations_dict.items():
            relevant = set(ground_truth_dict.get(user, []))
            total_ndcg += self.ndcg_at_k(recs, relevant, k)
            count += 1
        return total_ndcg / count if count > 0 else 0
    def average_precision_at_k(self, ranked_list, relevant_set, k=10):
        hits = 0
        sum_precisions = 0.
        for i in range(min(k,len(ranked_list))):
            if ranked_list[i] in relevant_set:
                hits += 1
                sum_precisions += hits/(i+1)
        return sum_precisions/hits if hits>0 else 0.
    def mean_average_precision(self, recommendations_dict, ground_truth_dict, k=10):
        ap_sum = 0.
        num_users = 0
        for user, recs in recommendations_dict.items():
            relevant = ground_truth_dict.get(user, set())
            ap_sum += self.average_precision_at_k(recs,set(relevant),k)
            num_users += 1
        return ap_sum/num_users if num_users>0 else 0.
    def evaluate_user_recommendations(self, user_id, recommended_movies, threshold=3.0, max_k=20):

        # threshold parameter kept for API compatibility
        # personal mean is used instead of global threshold

        # The threshold : is the minimum rating a movie needs to count as something the user actually liked.
        # Every movie the user rated at or above the threshold becomes a relevant item the ground truth of what we should be recommending.
        user_interactions = self.interactions_df[self.interactions_df['user_id']==user_id]

        # global threshold — same for everyone: which is an issue
        # relevant_items = user_interactions[user_interactions['rating']>=threshold]['movie_id'].tolist()

        # solution

        # personal threshold — relative to each user's rating style
        user_mean = user_interactions['rating'].mean()
        relevant_items = user_interactions[
            user_interactions['rating'] >= user_mean
            ]['movie_id'].tolist()

        if not recommended_movies:
            return {f'precision@{k}':0 for k in (5,10,max_k)} | {f'recall@{k}':0 for k in (5,10,max_k)} | {'relevant_items_count':len(relevant_items)}
        recommended_items = [int(m['movie_id']) if isinstance(m,dict) else int(m) for m in recommended_movies]
        relevant_items = [int(i) for i in relevant_items]
        def precision_at_k(k):
            topk = recommended_items[:k]
            hits = len(set(topk)&set(relevant_items))
            return hits/k if k>0 else 0
        def recall_at_k(k):
            if not relevant_items:
                return 0
            topk = recommended_items[:k]
            hits = len(set(topk)&set(relevant_items))
            return hits/len(relevant_items)
        metrics={}
        for k in (5,10,max_k):
            metrics[f'precision@{k}'] = precision_at_k(k)
            metrics[f'recall@{k}'] = recall_at_k(k)
        metrics['relevant_items_count'] = len(relevant_items)
        return metrics


class MovieRecommenderCLI:
    def __init__(self, hybrid_rec, movies_df, users_df, interactions_df, evaluator,
                 surprise_rec=None, constraint_layer=None, ncf_rec=None, request_parser=None):
        self.hybrid_rec = hybrid_rec
        self.request_parser = request_parser   # LLMRequestParser: free text -> request dict (option 13)
        self.movies_df = movies_df
        self.users_df = users_df
        self.interactions_df = interactions_df
        self.evaluator = evaluator
        self.surprise_rec = surprise_rec
        self.constraint_layer = constraint_layer
        self.ncf_rec = ncf_rec

    def get_recommendations_from_movies(self, movie_ratings, n=10):
        """
        Cold-start recommendations from a list of (movie_id, rating) pairs.
        Uses content-based similarity, weighted by the user's rating of each seed movie.
        movie_ratings: list of dicts [{'movie_id': 123, 'rating': 5}, ...]
        """
        seed_ids = {int(m['movie_id']) for m in movie_ratings}
        scores = {}

        for entry in movie_ratings:
            mid = int(entry['movie_id'])
            rating = float(entry.get('rating', 5))
            weight = rating / 5.0  # normalise rating to 0-1 as a weight

            similar = self.hybrid_rec.content_rec.get_similar_movies(mid, 20)
            for rank, sim in enumerate(similar):
                sim_id = sim['movie_id']
                if sim_id in seed_ids:
                    continue
                # closer matches (higher rank) score more, scaled by the seed's rating
                rank_score = (20 - rank) / 20
                scores[sim_id] = scores.get(sim_id, 0) + rank_score * weight

        # sort by aggregated score
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for sim_id, _ in ranked[:n]:
            row = self.movies_df[self.movies_df['movie_id'] == sim_id]
            if not row.empty:
                results.append(row.iloc[0].to_dict())
        return results

    def simulate_watching(self, user_id, movie_id, rating=None, watch_duration=None):
        if rating is None:
            rating = np.random.choice([3, 4, 5], p=[0.2, 0.5, 0.3])
        if watch_duration is None:
            movie_runtime = self.movies_df[self.movies_df['movie_id'] == movie_id]['runtime'].iloc[0]
            watch_duration = np.random.randint(movie_runtime // 2, movie_runtime)
        completion_rate = min(1.0, watch_duration / 120)
        existing = self.interactions_df[
            (self.interactions_df['user_id'] == user_id) &
            (self.interactions_df['movie_id'] == movie_id)
        ]
        if len(existing) > 0:
            idx = existing.index[0]
            self.interactions_df.loc[idx, 'rating'] = rating
            self.interactions_df.loc[idx, 'watch_duration'] += watch_duration
            self.interactions_df.loc[idx, 'view_count'] += 1
            self.interactions_df.loc[idx, 'completion_rate'] = completion_rate
            action = "Updated"
        else:
            new_interaction = pd.DataFrame({
                'user_id': [user_id],
                'movie_id': [movie_id],
                'rating': [rating],
                'watch_duration': [watch_duration],
                'view_count': [1],
                'completion_rate': [completion_rate]
            })
            self.interactions_df = pd.concat([self.interactions_df, new_interaction], ignore_index=True)
            action = "Added"
        self.hybrid_rec.collab_rec = CollaborativeRecommender(self.interactions_df)
        self.hybrid_rec.rec_system = MovieRecommendationSystem(self.movies_df, self.users_df, self.interactions_df)
        movie_title = self.movies_df[self.movies_df['movie_id'] == movie_id]['title'].iloc[0]
        print(f"{action} interaction: User {user_id} watched '{movie_title}' (Rating: {rating}, Duration: {watch_duration}min)")

        # method to build richer user profile data
    def build_constraint_profile(self, user_id):
            user_ratings = self.interactions_df[self.interactions_df['user_id'] == user_id]

            watched_movies = user_ratings.merge(self.movies_df, on='movie_id', how='inner')


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

            user_row = self.users_df[self.users_df['user_id'] == user_id].iloc[0]
            user_age = int(user_row['age'])

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
            }

    def evaluate_movie_models(self, k=10):
        from functools import lru_cache
        from scipy.sparse import csr_matrix
        from sklearn.preprocessing import normalize
        from recommenders.datasets.python_splitters import python_random_split
        from tqdm import tqdm

        if k < 1:
            raise ValueError("k must be at least 1.")

        if self.ncf_rec is None or self.ncf_rec.model is None:
            raise ValueError("Load the trained NCF model first.")

        # Use the original NCF data, not ratings added during the CLI session.
        data = self.ncf_rec.interactions_df.copy()
        data["date"] = pd.to_datetime(
            data["timestamp"], unit="s", errors="coerce"
        )
        data = data.dropna(subset=["date"])

        train, test = python_random_split(
            data, ratio=0.75, seed=42
        )

        train_users = set(train["user_id"])
        train_items = set(train["movie_id"])

        if (
                train_users != set(self.ncf_rec.user2id)
                or train_items != set(self.ncf_rec.item2id)
        ):
            raise ValueError(
                "NCF mappings do not match this training split."
            )

        test = test.loc[
            test["user_id"].isin(train_users)
            & test["movie_id"].isin(train_items)
            ].copy()

        # Check that the saved NCF model uses the same held-out data.
        saved_test = pd.read_csv(
            os.path.join(self.ncf_rec.model_dir, "test.csv"),
            usecols=["userID", "itemID", "rating"],
        ).sort_values(["userID", "itemID"])

        saved_test = saved_test[["userID", "itemID", "rating"]]

        expected_test = test[
            ["user_id", "movie_id", "rating"]
        ].rename(
            columns={"user_id": "userID", "movie_id": "itemID"}
        ).sort_values(["userID", "itemID"])

        if not np.array_equal(
                expected_test.to_numpy(), saved_test.to_numpy()
        ):
            raise ValueError(
                f"The saved NCF test set in {self.ncf_rec.model_dir} does not match this split "
                "(same users and films, different ratings): the model was trained on an older "
                "ratings file. Retrain it: use a new model_dir in main(), or force_retrain=True."
            )

        del saved_test, expected_test

        user_means = train.groupby("user_id")["rating"].mean()

        user_ids = sorted(train_users)
        item_ids = np.asarray(sorted(train_items))

        user_to_row = {
            uid: i for i, uid in enumerate(user_ids)
        }
        item_to_column = {
            mid: i for i, mid in enumerate(item_ids)
        }

        rows = train["user_id"].map(
            user_to_row
        ).to_numpy(dtype=np.int32)

        columns = train["movie_id"].map(
            item_to_column
        ).to_numpy(dtype=np.int32)

        matrix_shape = (len(user_ids), len(item_ids))

        # Keep watched movies separate from mean-centered ratings.
        seen_matrix = csr_matrix(
            (
                np.ones(len(train), dtype=bool),
                (rows, columns),
            ),
            shape=matrix_shape,
        )

        centered_matrix = csr_matrix(
            (
                train["rating"].to_numpy(dtype=np.float64)
                - train["user_id"].map(user_means).to_numpy(),
                (rows, columns),
            ),
            shape=matrix_shape,
        )
        centered_matrix.eliminate_zeros()

        user_vectors = normalize(centered_matrix)
        user_vectors_transposed = user_vectors.T.tocsr()

        del rows, columns

        # Relevant movies: held-out rating >= the user's training mean.
        positive_test = test.loc[
            test["rating"] >= test["user_id"].map(user_means)
            ]

        relevant_by_user = {}

        for uid, group in positive_test.groupby("user_id"):
            row = user_to_row[uid]
            seen_columns = seen_matrix.getrow(row).indices

            relevant = (
                    set(group["movie_id"])
                    - set(item_ids[seen_columns])
            )

            if relevant:
                relevant_by_user[uid] = relevant

        eval_users = sorted(relevant_by_user)

        if not eval_users:
            raise ValueError(
                "No eligible users have relevant test movies."
            )

        print(f"Training ratings: {len(train):,}")
        print(f"Warm-start test ratings: {len(test):,}")
        print(f"Evaluation users: {len(eval_users):,}")
        print("Preparing training-only baselines...", flush=True)

        # Popularity: preserve your formula, using training data only.
        popularity = train.groupby("movie_id").agg(
            avg_rating=("rating", "mean"),
            rating_count=("rating", "count"),
            total_views=("view_count", "sum"),
            avg_completion=("completion_rate", "mean"),
        ).fillna(0)

        popularity_scores = (
                0.3 * popularity["avg_rating"]
                + 0.3 * np.log1p(popularity["rating_count"])
                + 0.2 * np.log1p(popularity["total_views"])
                + 0.2 * popularity["avg_completion"]
        ).reindex(item_ids).to_numpy()

        # Content: use the user's highest-rated training movie.
        favourite_rows = train.groupby(
            "user_id"
        )["rating"].idxmax()

        favourites = train.loc[
            favourite_rows
        ].set_index("user_id")["movie_id"]

        content = self.hybrid_rec.content_rec

        movie_positions = {
            mid: i
            for i, mid in enumerate(content.movies_df["movie_id"])
        }

        if any(mid not in movie_positions for mid in item_ids):
            raise ValueError(
                "Some training movies are missing content metadata."
            )

        pool_positions = [
            movie_positions[mid] for mid in item_ids
        ]

        meta_vectors = normalize(content.meta_matrix)
        text_vectors = normalize(content.text_matrix)

        meta_pool_transposed = meta_vectors[
            pool_positions
        ].T.tocsr()

        text_pool_transposed = text_vectors[
            pool_positions
        ].T.tocsr()

        @lru_cache(maxsize=128)
        def content_scores_for_movie(movie_id):
            position = movie_positions[movie_id]

            return (
                    0.75 * (
                    meta_vectors[position] @ meta_pool_transposed
            ).toarray().ravel()
                    + 0.25 * (
                            text_vectors[position] @ text_pool_transposed
                    ).toarray().ravel()
            )

        print(
            "Training evaluation SVD on the 75% training split...",
            flush=True,
        )

        trainset = Dataset.load_from_df(
            train[["user_id", "movie_id", "rating"]],
            Reader(rating_scale=(0.5, 5)),
        ).build_full_trainset()

        svd = SVD(
            n_epochs=15, random_state=42, verbose=True
        )
        svd.fit(trainset)

        names = [
            "Popularity",
            "Content",
            "Collaborative",
            "Hybrid",
            "SVD",
            "NCF",
        ]

        totals = {
            name: np.zeros(3, dtype=np.float64)
            for name in names
        }

        for uid in tqdm(
                eval_users,
                desc="Comparing all methods",
                unit="user",
        ):
            user_row = user_to_row[uid]

            # Same candidate pool for every method.
            allowed = np.ones(len(item_ids), dtype=bool)
            allowed[
                seen_matrix.getrow(user_row).indices
            ] = False

            candidate_columns = np.flatnonzero(allowed)
            candidate_ids = item_ids[candidate_columns]
            relevant = relevant_by_user[uid]

            popular_scores = popularity_scores[
                candidate_columns
            ]

            content_scores = content_scores_for_movie(
                favourites.loc[uid]
            )[candidate_columns]

            svd_scores = np.asarray([
                svd.predict(uid, mid).est
                for mid in candidate_ids
            ])

            ncf_scores = np.asarray(
                self.ncf_rec.model.predict(
                    [uid] * len(candidate_ids),
                    candidate_ids.tolist(),
                    is_list=True,
                )
            ).ravel()

            # Collaborative: your five-neighbour scoring method.
            similarities = (
                    user_vectors[user_row]
                    @ user_vectors_transposed
            ).toarray().ravel()

            similarities[user_row] = -np.inf

            neighbours = np.argsort(
                -similarities, kind="stable"
            )[:min(5, len(user_ids) - 1)]

            neighbour_ratings = centered_matrix[
                neighbours
            ][:, candidate_columns].toarray()

            positive = neighbour_ratings > 0

            contributions = (
                    neighbour_ratings
                    * similarities[neighbours, None]
            )

            counts = positive.sum(axis=0)

            collaborative_scores = np.divide(
                np.where(
                    positive, contributions, 0
                ).sum(axis=0),
                counts,
                out=np.full(len(candidate_ids), -np.inf),
                where=counts > 0,
            )

            # Your current hybrid reranks 20 collaborative candidates.
            collaborative_order = np.argsort(
                -collaborative_scores, kind="stable"
            )

            shortlist = collaborative_order[
                np.isfinite(
                    collaborative_scores[collaborative_order]
                )
            ][:20]

            hybrid_scores = np.full(
                len(candidate_ids), -np.inf
            )

            hybrid_scores[shortlist] = (
                    0.4 * ((svd_scores[shortlist] - 0.5) / 4.5)
                    + 0.6 * (
                            (20 - np.arange(len(shortlist))) / 20
                    )
            )

            # Fill shortages with unseen popular movies.
            if len(shortlist) < k:
                selected = set(shortlist)

                fallback = [
                    i
                    for i in np.argsort(
                        -popular_scores, kind="stable"
                    )
                    if i not in selected
                ][:k - len(shortlist)]

                hybrid_scores[fallback] = (
                        -1 - np.arange(len(fallback))
                )

            scores_by_method = dict(zip(names, [
                popular_scores,
                content_scores,
                collaborative_scores,
                hybrid_scores,
                svd_scores,
                ncf_scores,
            ]))

            for name, scores in scores_by_method.items():
                if (
                        scores.shape != (len(candidate_ids),)
                        or np.isnan(scores).any()
                        or np.isposinf(scores).any()
                ):
                    raise ValueError(
                        f"Invalid {name} scores for user {uid}."
                    )

                # Unretrieved movies have -inf and are not recommended.
                order = np.argsort(
                    -scores, kind="stable"
                )
                order = order[
                    np.isfinite(scores[order])
                ][:k]

                recommended = candidate_ids[order].tolist()
                hits = len(set(recommended) & relevant)

                totals[name] += [
                    ndcg_at_k(recommended, relevant, k),
                    hits / k,
                    hits / len(relevant),
                ]

        results = {}

        for name in names:
            ndcg, precision, recall = (
                    totals[name] / len(eval_users)
            )

            results[name] = {
                f"NDCG@{k}": float(ndcg),
                f"Precision@{k}": float(precision),
                f"Recall@{k}": float(recall),
            }

        print(
            "\nShared 75/25 random-split evaluation"
            " — before constraints"
        )

        print(
            pd.DataFrame.from_dict(
                results, orient="index"
            ).to_string(
                float_format=lambda value: f"{value:.6f}"
            )
        )

        return results
    def get_user_recommendations(self, user_id, algorithm='hybrid', n=10, request=None):
        # --- Step 0: Global Seen Filter Setup ---
        user_movies = self.interactions_df[self.interactions_df['user_id'] == user_id]
        seen_movie_ids = set(user_movies['movie_id'].tolist())

        # Rank the whole catalog (1,957 films), not just n + a few: the hard constraints run BEFORE
        # the cut to n, so a strict request (e.g. children + Action) still gets n films if they exist
        fetch_limit = len(self.movies_df)

        # --- Step 1: Ranked candidate ids from whichever algorithm ---
        if algorithm == 'hybrid':
            # top 20 from collaborative + SVD, then filled up with popular films
            movie_ids = [m['movie_id'] for m in self.hybrid_rec.get_hybrid_recommendations(user_id, fetch_limit)]

        elif algorithm == 'collaborative':
            movie_ids = self.hybrid_rec.collab_rec.get_user_recommendations(user_id, fetch_limit)
            # collaborative only suggests films the 5 most similar users liked (often ~30 films),
            # so fill up with popular films, like hybrid does; duplicates are dropped below
            movie_ids = list(movie_ids) + [m['movie_id'] for m in self.hybrid_rec.rec_system.get_popular_movies(fetch_limit)]

        elif algorithm == 'content':
            if user_movies.empty:
                raw_recs = self.hybrid_rec.rec_system.get_popular_movies(fetch_limit)
            else:
                fav_movie = user_movies.loc[user_movies['rating'].idxmax(), 'movie_id']
                raw_recs = self.hybrid_rec.content_rec.get_similar_movies(fav_movie, fetch_limit)
            movie_ids = [m['movie_id'] for m in raw_recs]

        elif algorithm == 'popular':
            movie_ids = [m['movie_id'] for m in self.hybrid_rec.rec_system.get_popular_movies(fetch_limit)]

        elif algorithm == 'surprise' and self.surprise_rec:
            movie_ids = self.surprise_rec.get_user_recommendations(user_id, fetch_limit)

        elif algorithm == 'ncf' and self.ncf_rec:
            movie_ids = self.ncf_rec.get_user_recommendations(user_id, fetch_limit)
        else:
            movie_ids = []

        # Universal safety filter: drop seen films and duplicates, keep the ranking order
        movie_ids = [mid for mid in dict.fromkeys(movie_ids) if mid not in seen_movie_ids]

        # --- Step 2: Enrich with full movie data (one lookup table instead of one search per film) ---
        movie_rows = self.movies_df.drop_duplicates('movie_id').set_index('movie_id', drop=False)
        enriched = [movie_rows.loc[mid].to_dict() for mid in movie_ids if mid in movie_rows.index]

        # --- Step 3: Build user profile ---
        user_profile = self.build_constraint_profile(user_id)

        # --- Step 4: Apply hard constraints (age + request) BEFORE cutting to n ---
        filtered, removed = self.constraint_layer.apply_hard_constraints(enriched, user_profile, request)

        # soft layer switched off for now (it re-sorts by soft_score and overwrites the model order)
        # reranked = self.constraint_layer.apply_soft_constraints(filtered, user_profile)

        if len(filtered) < n:
            word = "film matches" if len(filtered) == 1 else "films match"
            print(f"\nOnly {len(filtered)} unseen {word} the constraints (asked for {n}).")

        return filtered[:n]


    def find_similar_movies(self, movie_id, n=5):
        return self.hybrid_rec.content_rec.get_similar_movies(movie_id, n)

    def get_user_profile(self, user_id):
        user_demo = self.users_df[self.users_df['user_id'] == user_id].iloc[0]
        user_interacts = self.interactions_df[self.interactions_df['user_id'] == user_id]

        if user_interacts.empty:
            return {'demographics': user_demo.to_dict(),
                    'stats': {'total_movies': 0, 'avg_rating': 0, 'favorite_genres': 'None'}}

        watched = user_interacts.merge(self.movies_df[['movie_id', 'genres']], on='movie_id')
        fav_genre = watched['genres'].mode().iloc[0] if not watched.empty else 'None'

        constraint_profile = self.build_constraint_profile(user_id)

        stats = {
            'total_movies': len(user_interacts),
            'avg_rating': round(user_interacts['rating'].mean(), 2),
            'avg_completion': round(user_interacts['completion_rate'].mean(), 2),
            'total_watch_time': user_interacts['watch_duration'].sum(),
            'favorite_genre': fav_genre
        }

        taste = {
            'liked_genres': sorted(constraint_profile['liked_genres']),
            'all_watched_genres': sorted(constraint_profile['all_watched_genres']),
            'liked_directors': sorted(constraint_profile['liked_directors']),
            'preferred_decade': constraint_profile['preferred_decade'],
            'preferred_language': constraint_profile['preferred_language'],
            'avg_runtime_tolerance': f"{constraint_profile['avg_runtime']:.0f} min",
            'liked_actors': sorted(constraint_profile['liked_actors'])  # new

        }

        return {
            'demographics': user_demo.to_dict(),
            'stats': stats,
            'taste_profile': taste
        }

    def pick_from_list(self, question, values):
        """Show a numbered list; the user types numbers or names like 1,4,horror (Enter = skip).
        Returns the chosen values in the catalog's spelling."""
        values = sorted(values)
        by_lower_name = {v.lower(): v for v in values}   # "horror" -> "Horror"
        print(f"\n{question}")
        for i in range(0, len(values), 4):
            print("".join(f"{i + j + 1:>4}. {v:<22}" for j, v in enumerate(values[i:i + 4])))
        answer = input("Numbers or names, comma separated (Enter = skip): ").strip()
        if not answer:
            return []
        chosen = []
        for part in answer.split(','):
            part = part.strip()
            if part.isdigit() and 1 <= int(part) <= len(values):
                chosen.append(values[int(part) - 1])
            elif part.lower() in by_lower_name:
                chosen.append(by_lower_name[part.lower()])
            elif part:
                print(f"  ignored {part!r} (not a number or name from the list)")
        return chosen

    def search_and_pick(self, question, values):
        """For long lists (themes, actors, directors): type part of a name, then pick from the matches."""
        chosen = []
        while True:
            text = input(f"\n{question} — type part of a name (Enter = done): ").strip().lower()
            if not text:
                return chosen
            matches = sorted(v for v in values if text in v.lower())[:30]
            if not matches:
                print("  no match, try again")
                continue
            chosen += self.pick_from_list(f"Matches for {text!r} (first 30):", matches)
            print(f"  chosen so far: {chosen}")

    def ask_preferences(self):
        """Ask for the hard-constraint request, one question at a time (Enter = skip).
        Returns the request dict that ConstraintLayer.apply_hard_constraints uses."""
        cl = self.constraint_layer
        request = {}

        audience = input("\nWho is watching? children / teens (Enter = no age limit): ").strip().lower()
        if audience:
            request["audience"] = audience

        include_genres = self.pick_from_list("Genres you WANT (a film needs at least one):", cl.all_genres)
        if include_genres:
            request["include_genres"] = include_genres

        exclude_genres = self.pick_from_list("Genres you DON'T want:", cl.all_genres)
        if exclude_genres:
            request["exclude_genres"] = exclude_genres

        include_moods = self.pick_from_list("Moods you WANT (a film needs at least one):", cl.all_moods)
        if include_moods:
            request["include_moods"] = include_moods

        exclude_moods = self.pick_from_list("Moods you DON'T want:", cl.all_moods)
        if exclude_moods:
            request["exclude_moods"] = exclude_moods

        include_themes = self.search_and_pick("Themes you want", cl.all_themes)
        if include_themes:
            request["include_themes"] = include_themes

        include_actors = self.search_and_pick("Actors you want", cl.all_actors)
        if include_actors:
            request["include_actors"] = include_actors

        include_directors = self.search_and_pick("Directors you want", cl.all_directors)
        if include_directors:
            request["include_directors"] = include_directors

        print("\nMaximum content level per dimension: 0 = none, 1 = mild, 2 = moderate, 3 = strong (Enter = any)")
        max_intensity = {}
        for dim in ("violence", "sexual_content", "language", "substance_use", "frightening_scenes"):
            answer = input(f"  max {dim}: ").strip()
            if answer:
                max_intensity[dim] = int(answer) if answer.isdigit() else answer   # validate_request reports bad values
        if max_intensity:
            request["max_intensity"] = max_intensity

        print(f"\nYour request: {request}")
        return request

    def display_movies(self, movies, title="Movies"):
        print(f"\n=== {title} ===")
        if not movies:
            print("No movies found.")
            return
        for i, movie in enumerate(movies, 1):
            print(f"{i}. {movie['title']} ({movie['year']}) - Genres: {movie['genres']} | Director: {movie['director']}")

# In production, a recommendation system trains on past behaviour and recommends for the future.
# Temporal split mirrors this exactly. Random split does not mirror anything real —
# no system trains and tests on randomly shuffled past events simultaneously.
    def evaluate_surprise_metrics(self, k=10, threshold=3.0):
        print("\nEvaluating Surprise SVD algorithm metrics...")
    
        # reader = Reader(rating_scale=(1, 5))
        # data = Dataset.load_from_df(self.interactions_df[['user_id', 'movie_id', 'rating']], reader)

        # use frozen copy
        df = self.interactions_df.copy()
        df['date'] = pd.to_datetime(df['timestamp'] , unit='s' , errors='coerce')
        df = df.dropna(subset=['date']) # drop null dates
    
        # trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

        #2. Split by cutoff 2017-01-01:
        # train_df = all ratings before 2017
        # test_df  = all ratings from 2017 onwards

        cutoff = pd.Timestamp('2017-01-01')
        train_df = df[df['date'] < cutoff]
        test_df = df[df['date'] >= cutoff]

        train_users = set(train_df['user_id'].unique())
        test_df = test_df[test_df['user_id'].isin(train_users)]

        print(f"Train ratings:                {len(train_df):,}")
        print(f"Test ratings (known users):   {len(test_df):,}")


        algo = SVD(n_epochs=15, random_state=42)

        reader = Reader(rating_scale=(0.5, 5))
        train_data = Dataset.load_from_df(
            train_df[['user_id', 'movie_id', 'rating']], reader
        )
        trainset = train_data.build_full_trainset()
        algo.fit(trainset)

        # build predictions manually from test_df
        predictions = [
            algo.predict(uid, mid, r_ui=r)
            for uid, mid, r in zip(
                test_df['user_id'],
                test_df['movie_id'],
                test_df['rating']
            )
        ]
    
        rmse = accuracy.rmse(predictions, verbose=False)
        mae = accuracy.mae(predictions, verbose=False)
    
        # Prepare for ranking metrics.
        # NOTE: these ranking metrics only rank each user's OWN test-period movies (the ones
        # they actually rated), not the whole catalog. That is a much easier task than top-N
        # recommendation, so the NDCG/precision/recall printed here are NOT comparable with
        # evaluate_movie_models() or the experiments in build_movie_texts.py / cold_start_experiment.py.
        # Use this function for RMSE/MAE (rating prediction) only.
        user_recs = defaultdict(list)
        user_true = defaultdict(set)

        # build personal mean per user from training data
        user_mean_ratings = train_df.groupby('user_id')['rating'].mean().to_dict()

        # then use personal threshold instead of global
        for pred in predictions:
            user_recs[pred.uid].append((pred.iid, pred.est))
            personal_threshold = user_mean_ratings.get(pred.uid, threshold)
            if pred.r_ui >= personal_threshold:
                user_true[pred.uid].add(pred.iid)
    
        precision_list, recall_list, ndcg_list = [], [], []
    
        for user, rec_list in user_recs.items():
            rec_list.sort(key=lambda x: x[1], reverse=True)  # Sort by predicted rating desc
            recommended_items = [iid for iid, _ in rec_list[:k]]
            relevant_set = user_true.get(user, set())
            if not relevant_set:
                continue
    
            hits = len(set(recommended_items) & relevant_set)
            precision = hits / k
            recall = hits / len(relevant_set)
            ndcg = ndcg_at_k(recommended_items, relevant_set, k)
    
            precision_list.append(precision)
            recall_list.append(recall)
            ndcg_list.append(ndcg)
    
        avg_precision = np.mean(precision_list) if precision_list else 0
        avg_recall = np.mean(recall_list) if recall_list else 0
        avg_ndcg = np.mean(ndcg_list) if ndcg_list else 0
    
        print("\nSurprise SVD performance metrics:")
        print(f"RMSE          : {rmse:.4f}")
        print(f"MAE           : {mae:.4f}")
        print(f"Precision@{k:<2}   : {avg_precision:.4f}")
        print(f"Recall@{k:<3}    : {avg_recall:.4f}")
        print(f"NDCG@{k:<3}      : {avg_ndcg:.4f}")

    def evaluate_ncf_metrics(self, k=10):
        if not self.ncf_rec:
            print("NCF recommender not available.")
            return

        from recommenders.evaluation.python_evaluation import (
            map as ncf_map, ndcg_at_k as ncf_ndcg,
            precision_at_k as ncf_prec, recall_at_k as ncf_recall
        )

        print("\nEvaluating NCF (NeuMF) metrics...")

        test = pd.read_csv(os.path.join(self.ncf_rec.model_dir, 'test.csv'))
        model = self.ncf_rec.model

        # reconstruct the training set: full interactions minus the test rows
        full = self.ncf_rec.interactions_df.rename(
            columns={'user_id': 'userID', 'movie_id': 'itemID'})
        test_keys = set(zip(test['userID'], test['itemID']))
        train_mask = [(u, i) not in test_keys
                      for u, i in zip(full['userID'], full['itemID'])]
        train = full[train_mask]

        train_items = list(self.ncf_rec.item2id.keys())
        eval_users = np.random.choice(list(self.ncf_rec.user2id.keys()),
                                      size=min(500, len(self.ncf_rec.user2id)),
                                      replace=False)

        users, items, preds = [], [], []
        for user in eval_users:
            user_list = [user] * len(train_items)
            users.extend(user_list)
            items.extend(train_items)
            preds.extend(list(model.predict(user_list, train_items, is_list=True)))

        all_predictions = pd.DataFrame({'userID': users, 'itemID': items, 'prediction': preds})

        # KEY STEP: remove movies the user already watched in training
        merged = pd.merge(train[['userID', 'itemID', 'rating']], all_predictions,
                          on=['userID', 'itemID'], how='outer')
        all_predictions = merged[merged.rating.isnull()].drop('rating', axis=1)

        test_eval = test[test['userID'].isin(eval_users)]

        m = ncf_map(test_eval, all_predictions, col_prediction='prediction', k=k)
        nd = ncf_ndcg(test_eval, all_predictions, col_prediction='prediction', k=k)
        pr = ncf_prec(test_eval, all_predictions, col_prediction='prediction', k=k)
        rc = ncf_recall(test_eval, all_predictions, col_prediction='prediction', k=k)

        print("\nNCF (NeuMF) performance metrics:")
        print(f"MAP          : {m:.4f}")
        print(f"NDCG@{k:<3}      : {nd:.4f}")
        print(f"Precision@{k:<2}   : {pr:.4f}")
        print(f"Recall@{k:<3}    : {rc:.4f}")

    def run_cli(self):
        print("\nWelcome to the Movie Recommendation CLI!")
        while True:
            print("\nMenu:")
            print("1. Get user recommendations")
            print("2. Simulate watching a movie")
            print("3. Find similar movies")
            print("4. Show user profile")
            print("5. Compare recommendation algorithms")
            print("6. Evaluate recommendation quality")
            print("7. Show popular movies")
            print("8. Exit")
            print("9. Evaluate Surprise algorithm metrics")
            print("10. Evaluate all recommendation methods")
            print("11. Compare all thesis methods (cold-start setup, validation users)")
            print("12. Leaderboard (best results per board: pilot / validation / test)")
            print("13. Recommend with preferences")

            choice = input("Select an option (1-13): ").strip()

            if choice == '1':
                user_id = int(input("User ID: "))
                algorithm = input("Algorithm (hybrid/collaborative/content/popular/surprise) [hybrid]: ").strip() or 'hybrid'
                n = int(input("Number of recommendations [5]: ") or 5)

                recs = self.get_user_recommendations(user_id, algorithm, n)
                self.display_movies(recs, f"{algorithm.title()} Recommendations")

            elif choice == '2':
                user_id = int(input("User ID: "))
                movie_id = int(input("Movie ID: "))
                rating_input = input("Rating (1-5, optional): ").strip()
                rating = float(rating_input) if rating_input else None
                duration_input = input("Watch duration in minutes (optional): ").strip()
                duration = int(duration_input) if duration_input else None
                self.simulate_watching(user_id, movie_id, rating, duration)

            elif choice == '3':
                movie_id = int(input("Movie ID: "))
                n = int(input("Number of similar movies [5]: ") or 5)
                similar = self.find_similar_movies(movie_id, n)
                self.display_movies(similar, f"Movies Similar to ID {movie_id}")


            elif choice == '4':
                user_id = int(input("User ID: "))
                profile = self.get_user_profile(user_id)
                d = profile['demographics']
                s = profile['stats']
                t = profile['taste_profile']

                print(f"\n{'=' * 50}")
                print(f"  USER {user_id} PROFILE")
                print(f"{'=' * 50}")
                print(f"\n  Demographics:")
                print(f"    Age: {d['age']} | Gender: {d['gender']} | Occupation: {d['occupation']}")
                print(f"    Age Group: {d['age_group']} | Country: {d['country']}")
                print(f"\n  Activity Stats:")
                print(f"    Movies rated:     {s['total_movies']}")
                print(f"    Average rating:   {s['avg_rating']}")
                print(f"    Avg completion:   {s['avg_completion']}")
                print(f"    Total watch time: {s['total_watch_time']} min")
                print(f"\n  Taste Profile:")
                print(f"    Preferred decade:   {t['preferred_decade']}s")
                print(f"    Preferred language: {t['preferred_language']}")
                print(f"    Avg runtime:        {t['avg_runtime_tolerance']}")
                print(f"    Liked genres:       {', '.join(t['liked_genres'])}")
                print(f"    Liked directors:    {', '.join(t['liked_directors'])}")
                print(f"    Liked actors:        {', '.join(t['liked_actors'])}")

                print(f"{'=' * 50}")

            elif choice == '5':  # Compare recommendation algorithms
                user_id = int(input("User ID: "))
                algorithms = ['hybrid', 'collaborative', 'content', 'popular', 'surprise', 'ncf']

                print(f"\n=== Comparing Recommendations for User {user_id} ===")
                for algo in algorithms:
                    try:
                        recs = self.get_user_recommendations(user_id, algo, 3)
                        print(f"\n{algo.upper()} RECOMMENDATIONS:")
                        for i, movie in enumerate(recs, 1):
                            print(f"  {i}. {movie['title']} ({movie['year']}) - {movie['genres']}")
                    except Exception as e:
                        print(f"  Could not get {algo} recommendations: {e}")


            elif choice == '6':

                user_id_input = input("Enter User ID (or press Enter to skip): ").strip()

                if user_id_input:

                    user_id = int(user_id_input)

                    print(f"\n--- Constraint Analysis for User {user_id} ---")

                    # get a large candidate pool to analyse
                    all_recs_raw = self.hybrid_rec.get_hybrid_recommendations(user_id, 30)

                    # enrich with full movie data
                    enriched = []
                    for movie in all_recs_raw:
                        full = self.movies_df[self.movies_df['movie_id'] == movie.get('movie_id')]
                        enriched.append(full.iloc[0].to_dict() if not full.empty else movie)


                    # apply constraints and measure
                    user_profile = self.build_constraint_profile(user_id)
                    filtered, removed_movies = self.constraint_layer.apply_hard_constraints(enriched, user_profile)

                    # soft layer switched off for now (it re-sorts by soft_score and overwrites the model order)
                    # reranked = self.constraint_layer.apply_soft_constraints(filtered, user_profile)

                    # constraint satisfaction rate
                    survived = len(filtered)
                    removed = len(removed_movies)
                    csr = survived / len(enriched) * 100 if enriched else 0

                    # soft constraint lift (soft layer switched off for now)
                    # if reranked and 'soft_score' in reranked[0]:
                    #     scores = [m['soft_score'] for m in reranked]
                    #     avg_score = sum(scores) / len(scores)
                    #     top_score = max(scores)
                    #     boosted = [m for m in reranked if m.get('soft_score', 1.0) > 1.0]
                    # else:
                    #     avg_score = top_score = 0
                    #     boosted = []

                    print(f"  Candidates evaluated:      {len(enriched)}")
                    print(f"  Passed hard constraints:   {survived}  ({csr:.1f}% satisfaction rate)")
                    print(f"  Removed by hard constraints: {removed}")
                    # print(f"  Boosted by soft constraints: {len(boosted)}")
                    # print(f"  Avg soft score:            {avg_score:.3f}")
                    # print(f"  Top soft score:            {top_score:.3f}")
                    print(f"  User age:                  {user_profile['age']}")
                    # runtime rule was removed from the hard constraints
                    # print(f"  Runtime ceiling:           {user_profile['avg_runtime'] * 1.5:.0f} min")
                    print(f"\n  Note: Precision/Recall metrics are reported via option 9")
                    print(f"        (temporal evaluation: train pre-2017, test 2017-2019)")
            

            elif choice == '7':  # Show popular movies
                n = int(input("Number of popular movies to show [10]: ") or 10)
                popular = self.hybrid_rec.rec_system.get_popular_movies(n)
                self.display_movies(popular, "Most Popular Movies")
                
            elif choice == '9':
                self.evaluate_surprise_metrics()

            elif choice == '8' or choice.lower() == 'exit':
                print("Exiting CLI. Goodbye!")
                break
            elif choice == '10':
                self.evaluate_movie_models(k=10)

            elif choice == '11':
                # Every thesis method in the cold-start setup of docs 06-17 (final_comparison.py):
                # warm users train, 203 validation users, 3 and 10 seeds; cached LLM answers only.
                import final_comparison
                final_comparison.run()

            elif choice == '12':
                # Leaderboard of every saved result: one board per user group (20-user pilot, 203 validation,
                # 814 test), seed count and idea; top 10/20/30 with explanations (leaderboard.py). No model calls.
                import leaderboard
                leaderboard.menu()

            elif choice == '13':
                # Hard constraints from a request: age/audience, genres, moods, themes, actors, directors,
                # content intensity. The model ranks, the constraint layer removes, then the top n are shown.
                user_id = int(input("User ID: "))
                algorithm = input("Algorithm (hybrid/collaborative/content/popular/surprise/ncf) [hybrid]: ").strip() or 'hybrid'
                n = int(input("Number of recommendations [10]: ") or 10)

                # two ways to build the same request dict: menus (code) or free text (LLM)
                mode = input("How do you want to give your preferences? 1 = menus, 2 = type a sentence (LLM) [1]: ").strip() or '1'
                if mode == '2' and self.request_parser is None:
                    print("No LLM request parser loaded, using the menus.")
                    mode = '1'

                if mode == '2':
                    text = input("\nDescribe what you want to watch: ").strip()
                    try:
                        request = self.request_parser.parse(text)
                    except (ValueError, RuntimeError) as e:
                        print(f"\nCould not use the request: {e}")
                        continue
                    info = self.request_parser.last_info
                    print(f"\nThe LLM understood: {request}")
                    print(f"  ({info.get('seconds', 0):.1f} s, {info.get('prompt_tokens', 0)} prompt tokens, "
                          f"{info.get('output_tokens', 0)} output tokens)")
                else:
                    request = self.ask_preferences()

                try:
                    recs = self.get_user_recommendations(user_id, algorithm, n, request)
                except ValueError as e:
                    print(f"\n{e}")
                    continue
                self.display_movies(recs, "Recommendations with your preferences")

            else:
                print("Invalid choice, please try again.")




class ConstraintLayer:


  def __init__(self, movies_df):
      self.movies_df = movies_df
      self.all_genres = set(self.movies_df['genres'].str.split('|').explode())
      self.all_themes = set(self.movies_df['llm_themes'].str.split('|').explode())
      self.all_moods = set(self.movies_df['llm_mood'].str.split('|').explode())
      # dropna: 9 films have no cast on TMDB
      self.all_actors = set(self.movies_df['cast_full'].dropna().str.split('|').explode())
      self.all_directors = set(self.movies_df['directors_full'].dropna().str.split('|').explode())

  def validate_request(self, request) :

      INTENSITY_DIMS = ("violence", "sexual_content", "language",
                        "substance_use", "frightening_scenes")
      AUDIENCES = ("children", "family", "teens", "adults")

      all_movies_df_genres = self.all_genres
      all_movies_df_themes = self.all_themes
      all_movies_df_moods = self.all_moods

      request_genres = request.get("include_genres" , [])

      for r in request_genres:
          if r not in all_movies_df_genres:
              raise ValueError( f"Unknown genre: {r!r}")

      request_themes = request.get("include_themes" , [])
      for r in request_themes:
          if r not in all_movies_df_themes:
              raise ValueError( f"Unknown theme: {r!r}")

      request_moods = request.get("include_moods" , [])
      for r in request_moods:
          if r not in all_movies_df_moods:
              raise ValueError( f"Unknown mood: {r!r}")

      request_actors = request.get("include_actors" , [])
      for r in request_actors:
          if r not in self.all_actors:
              raise ValueError( f"Unknown actor: {r!r}")

      request_directors = request.get("include_directors" , [])
      for r in request_directors:
          if r not in self.all_directors:
              raise ValueError( f"Unknown director: {r!r}")

      audience = request.get("audience" )
      if  audience and audience not in AUDIENCES:
          raise ValueError( f"Unknown audience: {request.get('audience')!r}")

      request_exclude_genres = request.get("exclude_genres" , [])
      for r in request_exclude_genres:
          if r not in all_movies_df_genres:
              raise ValueError( f"Unknown genre: {r!r}")

      request_exclude_moods = request.get("exclude_moods" , [])
      for r in request_exclude_moods:
          if r not in all_movies_df_moods:
              raise ValueError( f"Unknown mood: {r!r}")

      # intensity: known dimension and a whole number from 0 to 3
      # (bool is checked first because True counts as the int 1 in Python)
      request_max_intensity = request.get("max_intensity" , {})
      for dim, limit in request_max_intensity.items():
          if dim not in INTENSITY_DIMS:
              raise ValueError( f"Unknown intensity dimension: {dim!r}")
          if isinstance(limit, bool) or not isinstance(limit, int) or not 0 <= limit <= 3:
              raise ValueError( f"Intensity limit for {dim!r} must be 0-3, got {limit!r}")

      # unknown keys, e.g. "exclude_genre" without the s
      REQUEST_KEYS = ("audience", "include_genres", "exclude_genres", "include_moods", "exclude_moods",
                      "include_themes", "include_actors", "include_directors", "max_intensity")
      for key in request:
          if key not in REQUEST_KEYS:
              raise ValueError( f"Unknown request field: {key!r}")


      pass


  def apply_soft_constraints (self, recommendations, user_profile ):



      scored_movies = []
      for movie in recommendations:
          score = 1
          if movie['original_language'] != user_profile['preferred_language']:
              score *= 0.8

          if movie['director'] in user_profile['liked_directors']:
              score *= 1.3

          # actor boost
          movie_cast = set(str(movie.get('cast', '')).split('|'))
          liked_cast_overlap = movie_cast.intersection(user_profile['liked_actors'])
          if liked_cast_overlap:
              score *= (1 + 0.15 * len(liked_cast_overlap))
              # 15% boost per liked actor appearing in the film
              # max boost: 1.45 if all 3 cast members are liked actors
              print(f"  [Actor boost] {movie['title']} — "
                    f"matched: {liked_cast_overlap} — score: {score:.3f}")

          movie_decade = (movie['year'] // 10) * 10
          if movie_decade == user_profile['preferred_decade']:
              score *= 1.2

          elif abs(movie_decade - user_profile['preferred_decade']) > 20:
              score *= 0.9  # slight penalty for very distant era

          # Movies containing genres from the users liked genres (not just watched) get boosted:
          movie_genres = set(str(movie.get('genres', '')).split('|'))
          liked_genre = movie_genres.intersection(user_profile['liked_genres'])
          if liked_genre:
              score *= (1 + 0.1 * len(liked_genre))  # 10% per matching liked genre

          movie['soft_score'] = round(score, 4)
          scored_movies.append(movie)

      scored_movies.sort(key=lambda x: x['soft_score'], reverse=True)

      # print(f"\n[Soft Constraints] Re-ranked {len(scored_movies)} movies:")
      # for m in scored_movies:
      #     print(f"  {m['title']} — score: {m['soft_score']}")

      return scored_movies

  def apply_hard_constraints(self, recommendations, user_profile, request=None):

      # request
      request = request or {}
      self.validate_request(request)
      audience = request.get("audience")

      age_limit = float('inf')

      if audience == "children":
          age_limit = 12
      elif audience == "teens":
          age_limit = 16
      elif audience == "adults":
          age_limit = 18

      filtered_movies = []
      removed_movies = []

      # loop through all recommendations and filter

      # age under 13: keep children and family;
      # age 13–16: remove adults;
      # age 17 and over: no age filter.

      user_age = user_profile.get('age')
      if user_age is None or pd.isna(user_age):
          user_age = 17  # unknown age = adult (documented choice)

      # say where the age rule comes from: the request ("children requested") or the user's real age
      if age_limit < user_age:
          age_reason = f"{audience} requested"
      else:
          age_reason = f"user is {user_age}"

      user_age = min(user_age, age_limit)

      exclude_genres = set(request.get("exclude_genres", []))
      include_genres = set(request.get("include_genres", []))
      include_moods = set(request.get("include_moods", []))
      exclude_moods = set(request.get("exclude_moods", []))
      include_themes = set(request.get("include_themes", []))
      include_actors = set(request.get("include_actors", []))
      include_directors = set(request.get("include_directors", []))

      # LLMS probably needed in these scenarios
      # only winning oscar movies
      # recent movies only movies released this year

      for movie in recommendations:
          audience = str(movie.get('llm_audience', '')).strip().lower()

          # child
          if user_age < 13:
              if audience == "adults" or audience == "teens":
                  removed_movies.append(
                      (movie['title'], f'age restricted — {audience} audience ({age_reason})'))
                  continue

          # teen
          if 13 <= user_age < 17:
              if audience == "adults":
                  removed_movies.append((movie['title'], f'age restricted — adult audience ({age_reason})'))
                  continue

          # genres
          movie_genres = set(str(movie.get('genres', '')).split('|'))
          if movie_genres.intersection(exclude_genres):
              removed_movies.append((movie['title'], 'genre is in exclued list'))
              continue

          if include_genres and movie_genres.isdisjoint(include_genres):
              removed_movies.append((movie['title'], 'genre is missing'))
              continue

          # moods
          movie_moods = set(str(movie.get('llm_mood', '')).split('|'))
          if movie_moods.intersection(exclude_moods):
              removed_movies.append((movie['title'], 'mood is in exclued list'))
              continue

          if include_moods and movie_moods.isdisjoint(include_moods):
              removed_movies.append((movie['title'], 'mood is missing'))
              continue

          # themes
          movie_themes = set(str(movie.get('llm_themes', '')).split('|'))
          if include_themes and movie_themes.isdisjoint(include_themes):
              removed_movies.append((movie['title'], 'theme is missing'))
              continue

          # directors (all directors, some films have 2 or more)
          movie_directors = set(str(movie.get('directors_full', '')).split('|'))
          if include_directors and movie_directors.isdisjoint(include_directors):
              removed_movies.append((movie['title'], 'director is missing'))
              continue

          # actors (top 10 billed actors)
          movie_actors = set(str(movie.get('cast_full', '')).split('|'))
          if include_actors and movie_actors.isdisjoint(include_actors):
              removed_movies.append((movie['title'], 'actor is missing'))
              continue

          # intensity
          max_intensity_items = request.get("max_intensity", {}).items()

          over = [f"{dim} {movie[f'intensity_{dim}']} > {limit}"
                  for dim, limit in max_intensity_items
                  if movie[f"intensity_{dim}"] > limit]
          if over:
              removed_movies.append((movie['title'], "intensity: " + ", ".join(over)))
              continue

          filtered_movies.append(movie)

      # the candidate list can be the whole catalog, so only the first 10 removals are printed
      if removed_movies:
          print(f"\n[Hard Constraints] Removed {len(removed_movies)} movies, first 10:")
          for title, reason in removed_movies[:10]:
              print(f"  ✗ {title} — {reason}")

      return filtered_movies, removed_movies




class LLMRequestParser:
    """Turns a free-text request ("something funny for my 8 year old, nothing scary")
    into the same request dict that ask_preferences() builds from the menus.
    The LLM only fills the dict; ConstraintLayer does the filtering."""

    INTENSITY_DIMS = ("violence", "sexual_content", "language",
                      "substance_use", "frightening_scenes")

    def __init__(self, constraint_layer, model="deepseek-v4.1-flash:cloud",
                 ollama_url="http://127.0.0.1:11434/api/chat", timeout=120, max_retries=2):
        self.constraint_layer = constraint_layer
        self.model = model
        # 127.0.0.1, not localhost: on Windows "localhost" tries IPv6 first and adds about 2 s
        self.ollama_url = ollama_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.schema = self.build_schema()
        self.last_info = {}        # raw answer, time and tokens of the last call (for E4)

    # ------------------------------------------------------------------ schema

    def build_schema(self):
        cl = self.constraint_layer

        def list_of(values):
            # a list whose items can only be values from the catalog
            return {"type": "array", "items": {"type": "string", "enum": sorted(values)}}

        names = {"type": "array", "items": {"type": "string"}}   # actors/directors: free text

        return {
            "type": "object",
            "properties": {
                "audience": {"type": ["string", "null"], "enum": ["children", "teens", None]},
                "include_genres": list_of(cl.all_genres),
                "exclude_genres": list_of(cl.all_genres),
                "include_moods": list_of(cl.all_moods),
                "exclude_moods": list_of(cl.all_moods),
                "include_themes": list_of(cl.all_themes),
                "include_actors": names,
                "include_directors": names,
                "max_intensity": {
                    "type": "object",
                    "properties": {dim: {"type": "integer", "minimum": 0, "maximum": 3}
                                   for dim in self.INTENSITY_DIMS},
                    "additionalProperties": False,
                },
            },
            "required": ["audience", "include_genres", "exclude_genres", "include_moods",
                         "exclude_moods", "include_themes", "include_actors",
                         "include_directors", "max_intensity"],
            "additionalProperties": False,
        }

    # ------------------------------------------------------------------ prompt

    SYSTEM_PROMPT = (
        "You turn a movie request into filters for a movie recommender. "
        "Answer with JSON only, following the schema. "
        "Fill a field ONLY if the user clearly asks for it; otherwise leave it empty ([], {} or null). "
        "Never guess the user's taste from the text; only extract what they want or don't want."
    )

    def build_prompt(self, text):
        cl = self.constraint_layer
        return f"""RULES
- audience: "children" if a child under 13 will watch (e.g. "for my 8 year old", "with my kids");
  "teens" if the youngest viewer is 13-16; null otherwise.
- include_genres: genres the user wants; a film needs at least one of them.
- exclude_genres: genres the user does not want ("no horror").
- include_moods / exclude_moods: how the film should or should not feel ("something funny", "nothing depressing").
  Use the closest values from MOOD LIST.
- include_themes: what the film should be about ("about friendship"). Use the closest values from THEME LIST.
- include_actors / include_directors: full names as usually written (e.g. "Steven Spielberg", not "Spielberg").
- max_intensity: the highest allowed level, 0 = none, 1 = mild or brief, 2 = moderate, 3 = strong.
  "no sex scenes" / "no sexual content" -> sexual_content 1; "absolutely no ..." / "none at all" -> 0;
  "not too violent" -> violence 1; "nothing scary" -> frightening_scenes 1;
  "no swearing" -> language 1; "no drugs" -> substance_use 1.
  Only set the dimensions the user mentions.

GENRE LIST
{", ".join(sorted(cl.all_genres))}

MOOD LIST
{", ".join(sorted(cl.all_moods))}

THEME LIST
{", ".join(sorted(cl.all_themes))}

REQUEST
{text}"""

    # ------------------------------------------------------------------ LLM call

    def call_llm(self, prompt):
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "think": False,
            "format": self.schema,
            "options": {"temperature": 0, "seed": 42},
            "keep_alive": "10m",
        }
        last_error = None
        for attempt in range(1, self.max_retries + 2):
            start = time.perf_counter()
            try:
                res = requests.post(self.ollama_url, json=payload, timeout=self.timeout)
                res.raise_for_status()
                body = res.json()
            except (requests.RequestException, ValueError) as error:
                last_error = str(error)
                if attempt <= self.max_retries:
                    time.sleep(2 * attempt)
                continue
            self.last_info = {
                "raw": body["message"]["content"],
                "seconds": time.perf_counter() - start,
                "prompt_tokens": body.get("prompt_eval_count", 0),
                "output_tokens": body.get("eval_count", 0),
                "attempts": attempt,
            }
            return body["message"]["content"]
        raise RuntimeError(f"LLM call failed after {self.max_retries + 1} attempts: {last_error}")

    # ------------------------------------------------------------------ parse

    def fix_name_case(self, names, known):
        """'morgan freeman' -> 'Morgan Freeman' when the catalog has that name; other names stay as given
        (validate_request reports them)."""
        by_lower_name = {k.lower(): k for k in known}
        return [by_lower_name.get(n.strip().lower(), n.strip()) for n in names]

    def parse(self, text):
        """Free text -> request dict (only the fields the user asked for), checked by validate_request."""
        answer = json.loads(self.call_llm(self.build_prompt(text)))

        cl = self.constraint_layer
        answer["include_actors"] = self.fix_name_case(answer.get("include_actors", []), cl.all_actors)
        answer["include_directors"] = self.fix_name_case(answer.get("include_directors", []), cl.all_directors)

        # drop empty fields so the dict looks like one from the menu
        request = {key: value for key, value in answer.items() if value not in (None, [], {})}

        cl.validate_request(request)     # raises ValueError on an unknown actor/director
        return request




class EASE_R_Recommende ():
    def __init__(self, interactions_df, movies_df, lambda_):
        self.interactions_df = interactions_df.copy()
        self.movies_df = movies_df.copy()

        self.lambda_ = lambda_

    def build_X(self, interactions):
        user_mean = interactions.groupby("user_id")["rating"].transform("mean")
        liked = interactions[interactions["rating"] >= user_mean]

        users = liked["user_id"].unique()

        films = interactions["movie_id"].unique()

        # 1. lookups: id -> position
        film_to_col = { film_id: position for position ,film_id in enumerate(films) }

        user_to_row = {user_id: position for position, user_id in enumerate(users) }

        # self.film_to_col = film_to_col: seed film ids → column numbers;
        # self.col_to_film = films: column numbers → film ids, to turn B’s scores back into movie ids.

        self.film_to_col = film_to_col
        self.col_to_film = films

        rows = liked["user_id"].map(user_to_row)
        cols = liked["movie_id"].map(film_to_col)

        # vals = a 1 for each row of liked

        vals = np.ones(len(liked))

        # # 3. the matrix
        # X = csr_matrix of (vals, (rows, cols)), shape = (number of users, number of films)

        X = csr_matrix((vals, (rows, cols)), dtype=np.float64 , shape=(len(users), len(films)))

        return X


    def fit(self , interactions_df ):
        X = self.build_X(interactions_df)
        G = (X.T @ X).toarray()
        G[np.diag_indices_from(G)] += self.lambda_
        P = np.linalg.inv(G) # INVERSE OF G
        B = P / (-np.diag(P))
        np.fill_diagonal(B, 0.0)
        self.B = B

    def recommend(self, seed_ids, n=10, exclude=None):
        # seed ids -> column numbers (skip films without a column)
        cols = [self.film_to_col[film_id] for film_id in seed_ids if film_id in self.film_to_col]
        if not cols:
            return []

        # extra films to hide (e.g. all 10 seeds in the 3-seed run)
        ex_cols = []
        if exclude:
            ex_cols = [self.film_to_col[film_id] for film_id in exclude if film_id in self.film_to_col]

        x = np.zeros(len(self.col_to_film))
        x[cols] = 1

        scores = x @ self.B  # one score per film
        scores[cols] = -np.inf  # never recommend a seed back
        scores[ex_cols] = -np.inf  # never recommend an excluded film

        top = np.argsort(scores)[::-1][:n]  # n highest, highest first
        return self.col_to_film[top]




'''Semantic_IDs  

movies_df
The film catalog (1,957 rows). The class needs it for one thing: the order of the films (and their ids),
so that row i of the embedding matrix and row i of the codes both belong to the same film.


n_levels=3
How many tokens each film’s code has, not counting the collision token. 
Each level is one round of k-means on what’s left over from the previous round.

Level 1: broad groups.
Level 2: subgroups inside them.
Level 3: finer still.


codebook_size=16
How many clusters k-means makes at each level, i.e. how many different values one token can take (0–15). It sets how fine the split is:

level 1 with 16 codes → about 120 films per code on average;
3 levels → 16 × 16 × 16 = 4,096 possible codes for 1,957 films.
Too small (e.g. 4) and many films end up with the same code (collisions). Too large (e.g. 256) and most codes have 0–1 films, 
so shared prefixes become rare and tell you little.

seed=42
The random seed passed to k-means (random_state). 
k-means starts from random cluster centres, so without a fixed seed every run gives slightly different IDs. 
With the seed, the same code always produces the same IDs, 
which you need for reproducible thesis results and the TDK package. 42 is the seed used everywhere else in the project.



'''

class Semantic_IDs ():
    def __init__(self , movies_df, n_levels=3, codebook_size=16, seed=42) :
        self.movies_df = movies_df
        self.n_levels = n_levels
        self.codebook_size = codebook_size
        self.seed = seed


    def load_embeddings(self, name="movies_enriched_embeddinggemma_v2", folder="output/embeddings"):
        index = faiss.read_index(str(Path(folder) / f"{name}.faiss"))
        index_ids = np.load(Path(folder) / f"{name}_ids.npy")  # movie id of each row in the index

        # 2. get the vectors out of the index
        vectors = index.reconstruct_n(0, index.ntotal)  # (1957, 768), in the index's order

        # 3. reorder to the order of self.movies_df
        #     row_of  = {movie id: row number}   for each (row number, movie id) in index_ids

        row_of = {movie_id : row_number for row_number, movie_id in enumerate(index_ids)}

        # missing = movie ids in self.movies_df that are not in row_of

        missing = [movie_id for movie_id in self.movies_df["movie_id"] if movie_id not in row_of]

        assert not missing, f"{len(missing)} films have no embedding"

        vectors = vectors[[row_of[movie_id] for movie_id  in self.movies_df["movie_id"] ] ]

        # 4. normalise: every row gets length 1
        #     vectors = vectors / (length of each row)                  # np.linalg.norm(..., axis=1, keepdims=True)

        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

        # 5. keep what the rest of the class needs

        #
        #             self.movie_ids = self.movies_df["movie_id"] as a numpy array     # row i -> movie id
        #             self.row_of_movie = {movie id: row i}                            # movie id -> row i
        #

        self.vectors = vectors
        self.movie_ids = self.movies_df["movie_id"].to_numpy()

        self.row_of_movie = {movie_id: row for row  , movie_id in enumerate(self.movie_ids)}


    def fit (self):
        residual = self.vectors.copy()

        codes = np.zeros((len(self.movie_ids), self.n_levels), dtype=int)

        self.centres = []

        self.residual_norms = []

        for level in range(self.n_levels):
            kmeans = faiss.Kmeans(residual.shape[1], self.codebook_size, niter=25, nredo=10, seed=self.seed)
            kmeans.train(residual)
            _, labels = kmeans.index.search(residual, 1)  # nearest centre for each film
            labels = labels.ravel()  # (1957, 1) -> (1957,)
            codes[:, level] = labels
            residual = residual - kmeans.centroids[labels]  # remove what this level explained
            self.centres.append(kmeans.centroids)
            self.residual_norms.append ((np.linalg.norm(residual, axis=1)).mean())

        # # 4th token: 0, 1, 2 … among films with the same first n_levels tokens
        #     collision = for each film, how many earlier films have exactly the same codes
        #     self.codes = codes plus the collision column          # (1957, n_levels + 1)

        codes_df = pd.DataFrame(codes)
        collision = codes_df.groupby(list(codes_df.columns)).cumcount().to_numpy()
        self.codes = np.column_stack([codes, collision])

    def set_vectors(self, vectors, movie_ids):
        # vectors: (n_films, d) array from any source; movie_ids: the film of each row
        assert len(vectors) == len(movie_ids), f"{len(vectors)} vectors but {len(movie_ids)} movie ids"
        assert len(set(movie_ids)) == len(movie_ids), "duplicate movie ids"
        self.movies_df = self.movies_df.set_index("movie_id").loc[movie_ids].reset_index()

        vectors = np.asarray(vectors, dtype=np.float32)
        lengths = np.linalg.norm(vectors, axis=1, keepdims=True)
        assert (lengths > 0).all(), f"{(lengths == 0).sum()} films have an all-zero vector"
        vectors = vectors / lengths

        self.vectors = np.ascontiguousarray(vectors)
        self.movie_ids = self.movies_df["movie_id"].to_numpy()
        self.row_of_movie = {movie_id: row for row, movie_id in enumerate(self.movie_ids)}






    def get_id(self, movie_id):
        """The film's semantic ID, e.g. (3, 7, 1, 0)."""
        row = self.row_of_movie[movie_id]
        return tuple(int(token) for token in self.codes[row])

    def report(self):
        """Checks of the IDs: codebook use per level, leftover after each level, collisions."""
        n_films = len(self.codes)
        print(f"Semantic IDs: {n_films} films, {self.n_levels} levels x {self.codebook_size} codes")

        for level in range(self.n_levels):
            counts = np.bincount(self.codes[:, level], minlength=self.codebook_size)
            print(f"  level {level + 1}: {(counts > 0).sum()}/{self.codebook_size} codes used, "
                  f"films per code min {counts.min()} / max {counts.max()}, "
                  f"mean residual length after this level {self.residual_norms[level]:.3f}")

        prefixes = pd.DataFrame(self.codes[:, :self.n_levels])
        n_unique = len(prefixes.drop_duplicates())
        shared = prefixes.duplicated(keep=False).sum()
        print(f"  unique {self.n_levels}-token codes: {n_unique} of {n_films} films; "
              f"{shared} films share their code with another film; "
              f"largest group = {self.codes[:, -1].max() + 1} films")

    @staticmethod
    def pairs(group_sizes):
        """Number of film pairs inside groups of these sizes: c*(c-1)/2 per group."""
        sizes = np.asarray(group_sizes, dtype=float)
        return (sizes * (sizes - 1) / 2).sum()

    def meaningfulness(self):
        """Do films that share an ID prefix also share genre, director or decade, more than random pairs?"""
        n = len(self.codes)
        all_pairs = n * (n - 1) / 2
        levels = [f"l{level + 1}" for level in range(self.n_levels)]

        table = pd.DataFrame(self.codes[:, :self.n_levels], columns=levels)
        table["genre"] = self.movies_df["genres"].str.split("|").str[0].to_numpy()
        table["director"] = self.movies_df["director"].to_numpy()
        table["decade"] = (self.movies_df["year"] // 10 * 10).to_numpy()

        rows = []
        for attribute in ["genre", "director", "decade"]:
            random_rate = self.pairs(table.groupby(attribute).size()) / all_pairs
            for L in range(1, self.n_levels + 1):
                prefix = levels[:L]
                prefix_pairs = self.pairs(table.groupby(prefix).size())
                both_pairs = self.pairs(table.groupby(prefix + [attribute]).size())
                match_rate = both_pairs / prefix_pairs if prefix_pairs else float("nan")
                rows.append({"attribute": attribute, "prefix": L, "pairs": int(prefix_pairs),
                             "match_rate": match_rate, "random_rate": random_rate,
                             "lift": match_rate / random_rate})
        result = pd.DataFrame(rows)
        print(f"Meaningfulness ({self.codebook_size} codes per level): films sharing the first L tokens")
        print(result.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        return result


    def prefix_scores(self, seed_ids, weights=None):
        """One score per film: its shared ID prefix length with each seed, averaged over the seeds.
        weights: one weight per level (default 1, 1, 1 -> score = prefix length 0..n_levels)."""
        if weights is None:
            weights = np.ones(self.n_levels)
        weights = np.asarray(weights, dtype=float)

        seed_rows = [self.row_of_movie[m] for m in seed_ids if m in self.row_of_movie]
        tokens = self.codes[:, :self.n_levels]              # without the collision token
        total = np.zeros(len(tokens))
        if not seed_rows:
            return total

        for s in seed_rows:
            same = tokens == tokens[s]                      # (films, levels): same token at each level?
            matched = np.cumprod(same, axis=1)              # 1 until the first different level, then 0
            total += matched @ weights                      # weighted shared prefix length
        return total / len(seed_rows)

    def recommend(self, seed_ids, n=10, exclude=None, allowed=None, weights=None, tie_break=None):
        """Top n films by prefix score. Same call as EASE: recommend(seeds, n, exclude=...).
        allowed:   candidate movie ids (e.g. the 1,549-film pool); others are never recommended
        tie_break: one value per film (self.movie_ids order), higher first among equal scores"""
        scores = self.prefix_scores(seed_ids, weights)

        hidden = list(seed_ids) + list(exclude or [])
        scores[[self.row_of_movie[m] for m in hidden if m in self.row_of_movie]] = -np.inf
        if allowed is not None:
            scores[~np.isin(self.movie_ids, list(allowed))] = -np.inf

        if tie_break is None:
            order = np.argsort(-scores, kind="stable")
        else:
            order = np.lexsort((-np.asarray(tie_break, dtype=float), -scores))   # score first, then tie_break
        return self.movie_ids[order[:n]]






def main():
    print("Loading datasets...")



    movies_tmdb = pd.read_csv("data/data/processed_30k/movies.csv")

    movie_labels = pd.read_csv("output/experiments/deepseek_v4.1_flash_v5/movies_enriched.csv")

    movies_df = movie_labels.merge(movies_tmdb, on='movie_id', how='inner')

    movies_df = movies_df.drop(columns=['Unnamed: 0', 'enrich_model', 'prompt_version', 'repaired'])

    # top 10 billed actors and all directors (fetch_credits_full.py); only the constraint layer uses them,
    # the models keep using `cast` (top 3) and `director` (first director)
    movie_credits = pd.read_csv("data/data/credits_full/credits_full.csv",
                                usecols=['movie_id', 'cast_full', 'directors_full'])

    movies_df = movies_df.merge(movie_credits, on='movie_id', how='inner')

    users_df = pd.read_csv("data/data/processed_30k/users.csv")


    interactions_df = pd.read_csv("data/data/processed_30k/ratings.csv")

    interactions_df = interactions_df[interactions_df["movie_id"].isin(movies_df["movie_id"])]

    interactions_df = (interactions_df
                       .sort_values('timestamp')
                       .drop_duplicates(subset=['user_id', 'movie_id'], keep='last')
                       .reset_index(drop=True))

    assert len(movies_df) == 1957 and movies_df['movie_id'].is_unique
    assert movies_df['llm_audience'].notna().all()
    print(f"{len(movies_df)} films, {len(interactions_df):,} ratings, {interactions_df['user_id'].nunique():,} users")
    # expected: 1957 films, 2,547,690 ratings, 10,656 users

    print(f"Ratings after dedup: {len(interactions_df)}")

    print(f"Users: {len(users_df)}; Movies: {len(movies_df)}")
    
    # print("1. Generating dense interactions...")
    # interactions_df = generate_interactions_dense(users_df, movies_df)
    
    print("2. Initializing Content Recommender...")
    content_rec = ContentBasedRecommender(movies_df)
    
    print("3. Initializing Collaborative Recommender...")
    collab_rec = CollaborativeRecommender(interactions_df)
    
    print("4. Initializing Base Recommendation System...")
    rec_system = MovieRecommendationSystem(movies_df, users_df, interactions_df)

    surprise_rec = SurpriseRecommender(interactions_df)

    
    print("5. Initializing Enhanced Hybrid Recommender...")
    hybrid_rec = HybridRecommenderEnhanced(content_rec, collab_rec, movies_df, users_df, interactions_df, rec_system, surprise_rec)
    
    print("6. Diagnosing User-Movie Matrix...")
    diagnose_user_movie_matrix(collab_rec)
    
    print("7. Initializing Evaluator...")
    evaluator = RecommendationEvaluator(interactions_df, movies_df)
    
    print("8. Initializing Surprise Recommender (SVD)...")

    print("9. Launching CLI...")
    constraint_layer = ConstraintLayer(movies_df)

    cli = MovieRecommenderCLI(hybrid_rec, movies_df, users_df, interactions_df, evaluator, surprise_rec , constraint_layer)

    print("10. Initializing Constraint Layer...")

    print("11. Initializing NCF Recommender (NeuMF)...")
    ncf_rec = NCFRecommender(
        interactions_df,
        movies_df,
        epochs=30,
        # Trained on the half-star ratings (processed_30k regenerated 2026-10-20).
        # The old ncf_saved_10k_random75_25 model used rounded ratings and no longer matches the data.
        model_dir="output/models/ncf_saved_10k_halfstar_random75_25"
    )

    print("12. Initializing LLM Request Parser (option 13, free text)...")
    request_parser = LLMRequestParser(constraint_layer)

    cli = MovieRecommenderCLI(hybrid_rec, movies_df, users_df, interactions_df,
                              evaluator, surprise_rec, constraint_layer, ncf_rec, request_parser)



    cli.run_cli()


def main_semantic_ids():
    """Semantic IDs: build, check and compare (base vs LLM-enriched embedding, 16 vs 32 codes).
        python src/recomm.py semantic
    """
    movies_tmdb = pd.read_csv("data/data/processed_30k/movies.csv")
    movie_labels = pd.read_csv("output/experiments/deepseek_v4.1_flash_v5/movies_enriched.csv")
    movies_df = movie_labels.merge(movies_tmdb, on='movie_id', how='inner')
    titles = movies_df.set_index("movie_id")["title"]

    embeddings = {
        "base (TMDB text only)": "movies_base_embeddinggemma_v2",
        "enriched (+ LLM labels)": "movies_enriched_embeddinggemma_v2",
    }
    examples = [862, 863, 10193, 585, 12, 238, 240, 680, 120, 121]   # Toy Story 1-3, Pixar, Godfather I-II, Pulp Fiction, LOTR 1-2

    results = []
    for label, name in embeddings.items():
        for size in (16, 32):
            print(f"\n================ {label}, {size} codes per level ================")
            ids = Semantic_IDs(movies_df, n_levels=3, codebook_size=size, seed=42)
            ids.load_embeddings(name=name)
            ids.fit()
            ids.report()
            table = ids.meaningfulness()
            results.append(table.assign(embedding=label, codes=size))

            if size == 16:
                for movie_id in examples:
                    print(f"   {titles[movie_id]:<50} {ids.get_id(movie_id)}")

    # side-by-side lift: base vs enriched
    summary = pd.concat(results).pivot_table(index=["codes", "attribute", "prefix"],
                                             columns="embedding", values="lift")
    print("\nLift (x random), base vs enriched embedding:")
    print(summary.round(2).to_string())



if __name__ == "__main__":

        main_semantic_ids()

        # main()








