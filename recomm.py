
import pandas as pd
import numpy as np
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


CHILD_UNSAFE_KEYWORDS = {
    # Violence & Murder
    'murder', 'mass murder', 'serial killer', 'massacre', 'slaughter',
    'assassination', 'hitman', 'sniper', 'execution', 'death penalty',
    'electric chair', 'decapitation', 'beheading', 'severed head',
    'gore', 'brutality', 'torture', 'sadism', 'psychopath', 'cannibal',
    'dead body', 'corpse', 'severed foot', 'mutilation', 'atrocity',
    'war crimes', 'genocide', 'ethnic cleansing',

    # Sexual Content
    'rape', 'sexual abuse', 'sexual assault', 'sexual sadism',
    'sexual violence', 'sexual humiliation', 'sex crime',
    'nudity', 'pornography', 'porn industry', 'porn actor',
    'prostitute', 'prostitution', 'brothel', 'pimp', 'call girl',
    'pedophilia', 'pedophile', 'underage sex', 'eroticism',
    'adultery', 'infidelity', 'transvestism', 'transsexuality',
    'masturbation', 'erotic', 'femme fatale', 'lust',


    # Drugs & Substance Abuse
    'drug dealer', 'drug crime', 'drug trafficking', 'drug abuse',
    'drug addiction', 'drug addict', 'junkie', 'heroin', 'cocaine',
    'overdose', 'hallucination', 'marijuana', 'pot smoking',
    'drug cartel', 'narcos', 'drug smuggling', 'lsd',
    'methamphetamine', 'ecstasy', 'alcoholism', 'alcohol abuse',

    # Nazism & Extremism
    'nazi', 'nazism', 'neo-nazism', 'fascism', 'fascist',
    'nazi occupation', 'nsdap (nazi party)', 'ss (nazi schutzstaffel)',
    'swastika', 'hitler youth', 'national socialism',
    'national socialist party', 'neo-nazi', 'white supremacy',
    'terrorism', 'extremism', 'ku klux klan', 'hitler',

    # Suicide & Self Harm
    'suicide', 'suicide attempt', 'suicidal thoughts', 'seppuku',
    'self-destruction', 'self sacrifice', 'dying wish',

    # War Atrocities
    'holocaust (shoah)', 'concentration camp', 'gas chamber',
    'concentration camp prisoner', 'jew persecution',
    'auschwitz-birkenau concentration camp', 'genocide',
    'war crimes', 'death squad', 'prison camp',
    'children in wartime', 'child soldier',

    # Child Abuse & Exploitation
    'child abuse', 'child murder', 'child kidnapping',
    'children in danger', 'child labour', 'child bride',
    'sexual abuse', 'incest', 'child protection',

    # Organized Crime & Violence
    'gangster', 'gang war', 'crime boss', 'mafia',
    'organized crime', 'cartel', 'gang violence', 'yakuza',
    'street gang', 'mob', 'mobster', 'criminal mastermind',
    'bounty hunter', 'gun battle', 'shootout', 'gunfight',
    'heist', 'bank robbery', 'kidnapping', 'hostage',
    'human trafficking', 'slavery', 'slave trade',


    'mercenary', 'anti hero', 'vigilante', 'assassin',
    'shotgun', 'sniper', 'gunfight', 'gun battle', 'shootout',
    'decapitation', 'gore', 'brutality', 'sadism',

    # Dark Psychological Themes
    'nihilism', 'insanity', 'schizophrenia', 'psychosis',
    'mental breakdown', 'paranoia', 'hallucination',
    'dementia', 'electroconvulsive therapy',

    # Explicit Horror Themes
    'exorcism', 'demonic possession', 'satanism', 'occult',
    'necrophilia', 'cannibalism', 'gore',



'lunatic asylum', 'insane asylum', 'asylum',
'polygamy', 'concubine', 'oppression',
'electroconvulsive therapy', 'lobotomy',
'death row', 'execution', 'electric chair',
'forced labor', 'slave labor',
}



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

def generate_interactions_dense(users_df, movies_df, min_per_user=10, max_per_user=50, seed=42):
    np.random.seed(seed)
    interactions = []
    for user_id in users_df['user_id']:
        n = np.random.randint(min_per_user, max_per_user + 1)
        movie_choices = np.random.choice(movies_df['movie_id'], size=n, replace=False)
        for movie_id in movie_choices:
            rating = np.random.choice([1,2,3,4,5], p=[0.05,0.1,0.15,0.4,0.3])
            runtime = movies_df.loc[movies_df['movie_id'] == movie_id, 'runtime'].values[0]
            if pd.isna(runtime) or runtime <= 0:
                runtime = 120
            watch_duration = np.random.randint(max(10, int(runtime*0.1)), max(runtime,60))
            view_count = np.random.randint(1,5)
            completion_rate = min(1.0, watch_duration/runtime)
            interactions.append({
                'user_id': user_id,
                'movie_id': movie_id,
                'rating': rating,
                'watch_duration': watch_duration,
                'view_count': view_count,
                'completion_rate': completion_rate
            })
    return pd.DataFrame(interactions)

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
        return self.movies_df[self.movies_df['movie_id'].isin(top_ids)][['movie_id','title','genres','director','year']].to_dict('records')

class ContentBasedRecommender:
    def __init__(self, movies_df):
        self.movies_df = movies_df.copy()
        # --- PIPELINE 1: HARD METADATA ---
        # Initialize CountVectorizer()

        self.count_vec = CountVectorizer()
        # count_matrix = fit and transform the 'metadata_features' column
        self.meta_matrix = self.count_vec.fit_transform(self.movies_df['metadata_features'])
        # meta_sim = calculate cosine_similarity(count_matrix, count_matrix)

        self.meta_sim = cosine_similarity(self.meta_matrix, self.meta_matrix)


        # --- PIPELINE 2: SOFT TEXT ---
        self.tfidf = TfidfVectorizer(stop_words='english')

        # tfidf_matrix = fit and transform the 'text_features' column

        self.text_matrix = self.tfidf.fit_transform(self.movies_df['text_features'].fillna(''))
        # text_sim = calculate cosine_similarity(tfidf_matrix, tfidf_matrix)

        self.text_sim = cosine_similarity(self.text_matrix, self.text_matrix)

        # --- THE BLEND ---
        # Give Metadata 75% of the voting power, and Plot text 25%
        self.cosine_sim = (self.meta_sim * 0.75) + (self.text_sim * 0.25)


        ### explain this function that is responsible for finding the top n movies most similar to the movie we choose
    def get_similar_movies(self, movie_id, n=10):

        ## find the movie : search the pandas df movies_df for the row containing the movie_id passed in.
        idxs = self.movies_df[self.movies_df['movie_id'] == movie_id].index
        if len(idxs) == 0: return []
        idx = idxs[0]


        sim_scores = list(enumerate(self.cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x:x[1], reverse=True)[1:n+1]
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
        reader = Reader(rating_scale=(1,5))
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
    def __init__(self, interactions_df, movies_df, model_dir='ncf_saved',
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
        train, test = python_random_split(data, ratio=0.75, seed=DEFAULT_SEED)

        train_users = set(train['userID'].unique())
        train_items = set(train['itemID'].unique())
        test = test[test['userID'].isin(train_users) & test['itemID'].isin(train_items)]

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
            svd_normalized = (pred.est - 1) / 4

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
                 surprise_rec=None, constraint_layer=None, ncf_rec=None):
        self.hybrid_rec = hybrid_rec
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

    def get_user_recommendations(self, user_id, algorithm='hybrid', n=10):
        # --- Step 0: Global Seen Filter Setup ---
        user_movies = self.interactions_df[self.interactions_df['user_id'] == user_id]
        seen_movie_ids = set(user_movies['movie_id'].tolist())

        # Fetch extra candidates to account for movies we will filter out
        fetch_limit = n + len(seen_movie_ids) + 20

        # --- Step 1: Get raw recommendations from whichever algorithm ---
        if algorithm == 'hybrid':
            # Hybrid already handles internal filtering safely
            recs = self.hybrid_rec.get_hybrid_recommendations(user_id, n)

        elif algorithm == 'collaborative':
            movie_ids = self.hybrid_rec.collab_rec.get_user_recommendations(user_id, fetch_limit)
            # Universal safety filter
            movie_ids = [mid for mid in movie_ids if mid not in seen_movie_ids][:n]
            recs = [self.movies_df[self.movies_df['movie_id'] == mid].iloc[0].to_dict()
                    for mid in movie_ids
                    if not self.movies_df[self.movies_df['movie_id'] == mid].empty]

        elif algorithm == 'content':
            if user_movies.empty:
                raw_recs = self.hybrid_rec.rec_system.get_popular_movies(fetch_limit)
            else:
                fav_movie = user_movies.loc[user_movies['rating'].idxmax(), 'movie_id']
                raw_recs = self.hybrid_rec.content_rec.get_similar_movies(fav_movie, fetch_limit)
            # Universal safety filter
            recs = [m for m in raw_recs if m['movie_id'] not in seen_movie_ids][:n]

        elif algorithm == 'popular':
            raw_recs = self.hybrid_rec.rec_system.get_popular_movies(fetch_limit)
            # Universal safety filter
            recs = [m for m in raw_recs if m['movie_id'] not in seen_movie_ids][:n]

        elif algorithm == 'surprise' and self.surprise_rec:
            movie_ids = self.surprise_rec.get_user_recommendations(user_id, fetch_limit)
            # Universal safety filter
            movie_ids = [mid for mid in movie_ids if mid not in seen_movie_ids][:n]
            recs = [self.movies_df[self.movies_df['movie_id'] == mid].iloc[0].to_dict()
                    for mid in movie_ids
                    if not self.movies_df[self.movies_df['movie_id'] == mid].empty]

        elif algorithm == 'ncf' and self.ncf_rec:
            movie_ids = self.ncf_rec.get_user_recommendations(user_id, fetch_limit)
            # Universal safety filter
            movie_ids = [mid for mid in movie_ids if mid not in seen_movie_ids][:n]
            recs = [self.movies_df[self.movies_df['movie_id'] == mid].iloc[0].to_dict()
                    for mid in movie_ids
                    if not self.movies_df[self.movies_df['movie_id'] == mid].empty]
        else:
            recs = []

        # --- Step 2: Enrich with full movie data ---
        enriched = []
        for movie in recs:
            movie_id = movie.get('movie_id')
            full_data = self.movies_df[self.movies_df['movie_id'] == movie_id]
            if not full_data.empty:
                enriched.append(full_data.iloc[0].to_dict())
            else:
                enriched.append(movie)

        # --- Step 3: Build user profile ---
        user_profile = self.build_constraint_profile(user_id)

        # --- Step 4: Apply constraints on enriched data ---
        filtered = self.constraint_layer.apply_hard_constraints(enriched, user_profile)

        # apply soft filter
        reranked = self.constraint_layer.apply_soft_constraints(filtered, user_profile)

        return reranked


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

        reader = Reader(rating_scale=(1, 5))
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
    
        # Prepare for ranking metrics
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
            print("10. Evaluate NCF (NeuMF) metrics")
            choice = input("Select an option (1-9): ").strip()

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
                    filtered = self.constraint_layer.apply_hard_constraints(enriched, user_profile)
                    reranked = self.constraint_layer.apply_soft_constraints(filtered, user_profile)

                    # constraint satisfaction rate
                    survived = len(filtered)
                    removed = len(enriched) - survived
                    csr = survived / len(enriched) * 100 if enriched else 0

                    # soft constraint lift
                    if reranked and 'soft_score' in reranked[0]:
                        scores = [m['soft_score'] for m in reranked]
                        avg_score = sum(scores) / len(scores)
                        top_score = max(scores)
                        boosted = [m for m in reranked if m.get('soft_score', 1.0) > 1.0]
                    else:
                        avg_score = top_score = 0
                        boosted = []

                    print(f"  Candidates evaluated:      {len(enriched)}")
                    print(f"  Passed hard constraints:   {survived}  ({csr:.1f}% satisfaction rate)")
                    print(f"  Removed by hard constraints: {removed}")
                    print(f"  Boosted by soft constraints: {len(boosted)}")
                    print(f"  Avg soft score:            {avg_score:.3f}")
                    print(f"  Top soft score:            {top_score:.3f}")
                    print(f"  User age:                  {user_profile['age']}")
                    print(f"  Runtime ceiling:           {user_profile['avg_runtime'] * 1.5:.0f} min")
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
                self.evaluate_ncf_metrics()

            else:
                print("Invalid choice, please try again.")




class ConstraintLayer:
  def __init__(self, movies_df):
      self.movies_df = movies_df



  def apply_soft_constraints (self, recommendations, user_profile):

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



  def apply_hard_constraints(self, recommendations, user_profile):
    filtered_movies=[]
    removed_movies = []

    # loop through all recommendations and filter
    user_genres = user_profile['all_watched_genres']

    user_age = user_profile.get('age', 18)
    # We chose 18 as default because it is the safest assumption when age is unknown


    for movie in recommendations:
        if user_age < 13:
        # Check genres
            movie_genres = set(str(movie.get('genres', '')).split('|'))
            adult_genres = {'Horror', 'Thriller', 'Crime', 'War'}
            if movie_genres.intersection(adult_genres):
                removed_movies.append((movie['title'], f'age restricted — adult genre (user is {user_age})'))
                continue

        # Check keywords
            movie_keywords = str(movie.get('keywords', '')).lower()
            triggered = [kw for kw in CHILD_UNSAFE_KEYWORDS if kw in movie_keywords]
            if triggered:
                 removed_movies.append(
                 (movie['title'], f'age restricted — unsafe keywords: {triggered} (user is {user_age})'))
                 continue


        # movie must match user's preferred language, if not remove it
        # if movie.get('original_language') != user_profile['preferred_language']:
        #      removed_movies.append((movie['title'], 'language mismatch'))
        #
        #      continue



    # movie must share at least one genre with user's history, if not remove it and put it in removed
        movie_genres = set(str(movie.get('genres', '')).split('|'))
        if not movie_genres.intersection(user_profile['all_watched_genres']):

             removed_movies.append((movie['title'], 'no genre overlap'))
             continue


    # movie runtime must not exceed 1.5x user's average tolerance
        runtime = movie.get('runtime', 0)
        max_runtime = user_profile['avg_runtime'] * 1.5
        if runtime and runtime > max_runtime:
              removed_movies.append((movie['title'], f'runtime {runtime}min exceeds limit {max_runtime:.0f}min'))
              continue

        filtered_movies.append(movie)

    if removed_movies:
        print(f"\n[Hard Constraints] Removed {len(removed_movies)} movies:")
        for title, reason in removed_movies:
            print(f"  ✗ {title} — {reason}")

    return filtered_movies




def main():
    print("Loading datasets...")


    movies_df = pd.read_csv('data/data/processed/movies.csv')
    users_df = pd.read_csv('data/data/processed/users.csv')
    interactions_df = pd.read_csv('data/data/processed/ratings.csv')

    interactions_df = interactions_df.drop_duplicates(
        subset=['user_id', 'movie_id'],
        keep='last'  # keep most recent rating if duplicated
    )

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
    ncf_rec = NCFRecommender(interactions_df, movies_df, epochs=30)

    cli = MovieRecommenderCLI(hybrid_rec, movies_df, users_df, interactions_df,
                              evaluator, surprise_rec, constraint_layer, ncf_rec)


    movies_df = pd.read_csv('data/data/processed/movies.csv')
    ratings_df = pd.read_csv('data/data/processed/ratings.csv')

    # find which movies actually have ratings
    rated_ids = set(ratings_df['movie_id'].unique())
    rated_movies = movies_df[movies_df['movie_id'].isin(rated_ids)]
    unrated_movies = movies_df[~movies_df['movie_id'].isin(rated_ids)]

    print(f"Movies WITH ratings:    {len(rated_movies)}")
    print(f"Movies WITHOUT ratings: {len(unrated_movies)}")
    print(f"\nYear range of rated movies:")
    print(f"  Oldest: {rated_movies['year'].min()}")
    print(f"  Newest: {rated_movies['year'].max()}")
    print(f"\nYear range of unrated movies:")
    print(f"  Oldest: {unrated_movies['year'].min()}")
    print(f"  Newest: {unrated_movies['year'].max()}")


    rated_movies['year'] = pd.to_numeric(rated_movies['year'], errors='coerce')
    decade_counts = (rated_movies['year'] // 10 * 10).value_counts().sort_index()
    print(f"\nRated movies by decade:")
    print(decade_counts.to_string())

    cli.run_cli()





if __name__ == "__main__":
    main()


