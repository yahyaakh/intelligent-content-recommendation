# -*- coding: utf-8 -*-

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

class CollaborativeRecommender:
    def __init__(self, interactions_df):
        self.interactions_df = interactions_df.copy()
        if len(interactions_df) == 0:
            self.user_movie_matrix = pd.DataFrame()
            self.user_similarity_df = pd.DataFrame()
            return
        self.user_movie_matrix = self.interactions_df.pivot(index='user_id',columns='movie_id',values='rating').fillna(0)
        self.user_similarity = cosine_similarity(self.user_movie_matrix)
        self.user_similarity_df = pd.DataFrame(self.user_similarity,index=self.user_movie_matrix.index,columns=self.user_movie_matrix.index)
    def get_user_recommendations(self, user_id, n=10):
        if user_id not in self.user_movie_matrix.index: return []
        user_similarities = self.user_similarity_df[user_id].sort_values(ascending=False)
        similar_users = user_similarities.iloc[1:6].index
        user_movies = set(self.user_movie_matrix.loc[user_id][self.user_movie_matrix.loc[user_id]>0].index)
        recommendations = {}
        for user in similar_users:
            for movie_id, rating in self.user_movie_matrix.loc[user].items():
                if rating > 3.5 and movie_id not in user_movies:
                    recommendations.setdefault(movie_id,[]).append(rating*user_similarities[user])
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

class HybridRecommenderEnhanced:
    def __init__(self, content_rec, collab_rec, movies_df, users_df, interactions_df, rec_system):
        self.content_rec = content_rec
        self.collab_rec = collab_rec
        self.movies_df = movies_df
        self.users_df = users_df
        self.interactions_df = interactions_df
        self.rec_system = rec_system
    def get_hybrid_recommendations(self, user_id, n=10):
        recommendations = {}
        user_movies = self.interactions_df[self.interactions_df['user_id']==user_id]
        if len(user_movies)==0:
            return self.rec_system.get_popular_movies(n)
        user_fav = user_movies[user_movies['rating']>=4]['movie_id'].tolist()
        for mid in user_fav[:3]:
            similar_movies = self.content_rec.get_similar_movies(mid,5)
            for movie in similar_movies:
                if movie['movie_id'] not in recommendations:
                    recommendations[movie['movie_id']] = {'content_score':0,'collab_score':0,'movie_info':movie}
                recommendations[movie['movie_id']]['content_score'] +=1
        collab_recs = self.collab_rec.get_user_recommendations(user_id,10)
        for i,movie_id in enumerate(collab_recs):
            if movie_id not in recommendations:
                movie_info = self.movies_df[self.movies_df['movie_id']==movie_id].iloc[0]
                recommendations[movie_id] = {'content_score':0,'collab_score':0,'movie_info':movie_info.to_dict()}
            recommendations[movie_id]['collab_score'] = 10 - i
        for movie_id, scores in recommendations.items():
            scores['hybrid_score'] = 0.4 * scores['content_score'] + 0.6 * scores['collab_score']
        sorted_recs = sorted(recommendations.items(), key=lambda x:x[1]['hybrid_score'], reverse=True)
        seen = set(user_movies['movie_id'].tolist())
        final = []
        for mid,data in sorted_recs:
            if mid not in seen and len(final)<n:
                final.append(data['movie_info'])
        if len(final)<n:
            needed = n-len(final)
            popular = self.rec_system.get_popular_movies(needed)
            for pm in popular:
                if pm['movie_id'] not in {m['movie_id'] for m in final}:
                    final.append(pm)
                    if len(final)==n:
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
        user_interactions = self.interactions_df[self.interactions_df['user_id']==user_id]
        relevant_items = user_interactions[user_interactions['rating']>=threshold]['movie_id'].tolist()
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
    def __init__(self, hybrid_rec, movies_df, users_df, interactions_df, evaluator, surprise_rec=None):
        self.hybrid_rec = hybrid_rec
        self.movies_df = movies_df
        self.users_df = users_df
        self.interactions_df = interactions_df
        self.evaluator = evaluator
        self.surprise_rec = surprise_rec

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

    def get_user_recommendations(self, user_id, algorithm='hybrid', n=10):
        if algorithm == 'hybrid':
            return self.hybrid_rec.get_hybrid_recommendations(user_id, n)
        elif algorithm == 'collaborative':
            movie_ids = self.hybrid_rec.collab_rec.get_user_recommendations(user_id, n)
            return [self.movies_df[self.movies_df['movie_id'] == mid].iloc[0].to_dict() for mid in movie_ids if not self.movies_df[self.movies_df['movie_id'] == mid].empty]
        elif algorithm == 'content':
            user_movies = self.interactions_df[self.interactions_df['user_id'] == user_id]
            if user_movies.empty:
                return self.hybrid_rec.rec_system.get_popular_movies(n)
            fav_movie = user_movies.loc[user_movies['rating'].idxmax(), 'movie_id']
            return self.hybrid_rec.content_rec.get_similar_movies(fav_movie, n)
        elif algorithm == 'popular':
            return self.hybrid_rec.rec_system.get_popular_movies(n)
        elif algorithm == 'surprise' and self.surprise_rec:
            movie_ids = self.surprise_rec.get_user_recommendations(user_id, n)
            return [self.movies_df[self.movies_df['movie_id'] == mid].iloc[0].to_dict() for mid in movie_ids if not self.movies_df[self.movies_df['movie_id'] == mid].empty]

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
        stats = {'total_movies': len(user_interacts),
                 'avg_rating': user_interacts['rating'].mean(),
                 'avg_completion': user_interacts['completion_rate'].mean(),
                 'total_watch_time': user_interacts['watch_duration'].sum(),
                 'favorite_genre': fav_genre}
        return {'demographics': user_demo.to_dict(), 'stats': stats}

    def display_movies(self, movies, title="Movies"):
        print(f"\n=== {title} ===")
        if not movies:
            print("No movies found.")
            return
        for i, movie in enumerate(movies, 1):
            print(f"{i}. {movie['title']} ({movie['year']}) - Genres: {movie['genres']} | Director: {movie['director']}")



    def evaluate_surprise_metrics(self, k=10, threshold=3.0):
        print("\nEvaluating Surprise SVD algorithm metrics...")
    
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(self.interactions_df[['user_id', 'movie_id', 'rating']], reader)
    
        trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
        algo = SVD(n_epochs=15, random_state=42)
        algo.fit(trainset)
        predictions = algo.test(testset)
    
        rmse = accuracy.rmse(predictions, verbose=False)
        mae = accuracy.mae(predictions, verbose=False)
    
        # Prepare for ranking metrics
        user_recs = defaultdict(list)
        user_true = defaultdict(set)
    
        for pred in predictions:
            user_recs[pred.uid].append((pred.iid, pred.est))
            if pred.r_ui >= threshold:
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
                print(f"\nUser {user_id} Profile:")
                print(profile)

            elif choice == '5':  # Compare recommendation algorithms
                user_id = int(input("User ID: "))
                algorithms = ['hybrid', 'collaborative', 'content', 'popular', 'surprise']
                print(f"\n=== Comparing Recommendations for User {user_id} ===")
                for algo in algorithms:
                    try:
                        recs = self.get_user_recommendations(user_id, algo, 3)
                        print(f"\n{algo.upper()} RECOMMENDATIONS:")
                        for i, movie in enumerate(recs, 1):
                            print(f"  {i}. {movie['title']} ({movie['year']}) - {movie['genres']}")
                    except Exception as e:
                        print(f"  Could not get {algo} recommendations: {e}")

            elif choice == '6':  # Evaluate recommendation quality
                user_id_input = input("Enter User ID to see recommendation comparison (or press Enter to skip): ").strip()
                if user_id_input:
                    user_id = int(user_id_input)
                    algorithms = ['hybrid', 'collaborative', 'content', 'popular', 'surprise']
                    print(f"\n=== Side-by-side Recommendations Comparison for User {user_id} ===")
                    for algo in algorithms:
                        try:
                            recs = self.get_user_recommendations(user_id, algo, 5)
                            print(f"\n{algo.upper()} RECOMMENDATIONS:")
                            for i, movie in enumerate(recs, 1):
                                print(f"  {i}. {movie['title']} ({movie['year']}) - {movie['genres']}")
                        except Exception as e:
                            print(f"  Could not get {algo} recommendations: {e}")
                print("TBD)")
            

            elif choice == '7':  # Show popular movies
                n = int(input("Number of popular movies to show [10]: ") or 10)
                popular = self.hybrid_rec.rec_system.get_popular_movies(n)
                self.display_movies(popular, "Most Popular Movies")
                
            elif choice == '9':
                self.evaluate_surprise_metrics()

            elif choice == '8' or choice.lower() == 'exit':
                print("Exiting CLI. Goodbye!")
                break
            else:
                print("Invalid choice, please try again.")

def main():
    print("Loading datasets...")
    movies_df = pd.read_csv('data/data/processed/movies.csv')
    users_df = pd.read_csv('data/data/processed/users.csv')
    interactions_df = pd.read_csv('data/data/processed/ratings.csv')
    print(f"Users: {len(users_df)}; Movies: {len(movies_df)}")
    
    # print("1. Generating dense interactions...")
    # interactions_df = generate_interactions_dense(users_df, movies_df)
    
    print("2. Initializing Content Recommender...")
    content_rec = ContentBasedRecommender(movies_df)
    
    print("3. Initializing Collaborative Recommender...")
    collab_rec = CollaborativeRecommender(interactions_df)
    
    print("4. Initializing Base Recommendation System...")
    rec_system = MovieRecommendationSystem(movies_df, users_df, interactions_df)
    
    print("5. Initializing Enhanced Hybrid Recommender...")
    hybrid_rec = HybridRecommenderEnhanced(content_rec, collab_rec, movies_df, users_df, interactions_df, rec_system)
    
    print("6. Diagnosing User-Movie Matrix...")
    diagnose_user_movie_matrix(collab_rec)
    
    print("7. Initializing Evaluator...")
    evaluator = RecommendationEvaluator(interactions_df, movies_df)
    
    print("8. Initializing Surprise Recommender (SVD)...")
    surprise_rec = SurpriseRecommender(interactions_df)
    
    print("9. Launching CLI...")
    cli = MovieRecommenderCLI(hybrid_rec, movies_df, users_df, interactions_df, evaluator, surprise_rec)
    cli.run_cli()

if __name__ == "__main__":
    main()
