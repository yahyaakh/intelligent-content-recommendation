
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class cold_start_recommendation:
    def __init__(self, articles_df , users_df , interactions_df):
        self.articles_df = articles_df
        self.users_df = users_df
        self.interactions_df = interactions_df

        self.vectorizer = TfidfVectorizer()
        # We save the matrix as a class variable so we can reuse it instantly
        self.tfidf_matrix = self.vectorizer.fit_transform(self.articles_df['content_features'])

    def get_user_mode(self , user_id):
        user_mode = self.users_df.loc[self.users_df['user_id'] == user_id, 'mode'].values[0]
        return user_mode

    def cold_mode(self , user_id, n):
        # users with 0-2 clicks
        # strategy : recommend the most popular articles they haven't seen yet
        # most popular articles : articles that have the biggest number of clicks
        # from interaction.csv we have to count clicks per article and rank popularity
        # exclude any article this user already clicked
        # from articles.csv get the title and category of the top articles to return

        # Step 1: Get articles the user already clicked (using a set for faster lookups)
        clicked_by_user = set(self.interactions_df[self.interactions_df['user_id'] == user_id]['article_id'])

        # Get popular articles
        popular_articles = self.interactions_df['article_id'].value_counts().index

        # Step 2: Exclude clicked articles and grab exactly 'n' amount
        recommended_articles = [art for art in popular_articles if art not in clicked_by_user][:n]

        # Step 3: Get the details for these top 'n' articles all at once
        # .isin() checks if the article_id is inside the recommended_articles list
        final_df = self.articles_df[self.articles_df['article_id'].isin(recommended_articles)]

        # Return just the specific columns we want as a clean list of dictionaries
        return final_df[['article_id', 'title', 'category']].to_dict('records')


    def warm_user(self, user_id, n):
        # user with 3-9 clicks , not enough for collaborative filtering
        # use the article they clicked as seeds , find articles most similar to what they liked.
        # using TF-IDF cosine similarity
        # step 1: look at the user click history in interactions.csv
        # step 2: find their most recent or most frequent category
        # step 3: find articles with similar content feature using TF-IDF matrix
        # step 4: Return the top N most similar articles they havent clicked

        # interactions.csv → which articles did this user click (seeds for similarity)
        # articles.csv → the TF-IDF matrix built from content_features

        # Step 1: Get articles this user already clicked
        liked_articles = set(
                self.interactions_df[
                    self.interactions_df['user_id'] == user_id
                    ]['article_id']
        )

        # Step 2: Get the row indices of liked articles in the TF-IDF matrix
        liked_indices = self.articles_df[
                self.articles_df['article_id'].isin(liked_articles)
        ].index


        # Step 3: Compute similarity of liked articles against ALL articles
        # Result shape: (n_liked_articles x n_total_articles)
        sim_scores = cosine_similarity(
                self.tfidf_matrix[liked_indices],
                self.tfidf_matrix
        )

        # Step 4: Average similarity scores across all liked articles
        # Collapses the matrix to one score per article
        avg_scores = sim_scores.mean(axis=0)

        # Step 5: Sort all articles by their average similarity score
        scored_articles = sorted(
                enumerate(avg_scores),
                key=lambda x: x[1],
                reverse=True
        )

        # Step 6: Filter out already clicked articles and take top n
        recommendations = []
        for idx, score in scored_articles:
                article_id = self.articles_df.iloc[idx]['article_id']
                if article_id not in liked_articles:
                    recommendations.append(
                        self.articles_df.iloc[idx][
                            ['article_id', 'title', 'category']
                        ].to_dict()
                    )
                if len(recommendations) == n:
                    break

        return recommendations

    def hot_mode(self, user_id, n):
        # Step 1: Build user-article matrix
        user_article_matrix = self.interactions_df.pivot(
            index='user_id',
            columns='article_id',
            values='clicked'
        ).fillna(0)

        # Step 2: No mean-centering needed for binary clicks
        # Cosine similarity on raw 0/1 matrix works correctly
        user_similarity = cosine_similarity(user_article_matrix)
        user_similarity_df = pd.DataFrame(
            user_similarity,
            index=user_article_matrix.index,
            columns=user_article_matrix.index
        )

        # Step 3: Check user exists in matrix
        if user_id not in user_article_matrix.index:
            return self.warm_user(user_id, n)

        # Step 4: Find top 5 similar users
        similar_users = user_similarity_df[user_id].sort_values(
            ascending=False
        ).iloc[1:6].index

        # Step 5: Get articles this user already clicked
        seen_articles = set(
            user_article_matrix.loc[user_id][
                user_article_matrix.loc[user_id] > 0
                ].index
        )

        # Step 6: Collect recommendations from similar users
        # Filter changed from centered_score > 0 to raw_score > 0
        recommendations = {}
        for similar_user in similar_users:
            sim_weight = user_similarity_df[user_id][similar_user]
            for article_id, raw_score in user_article_matrix.loc[similar_user].items():
                if raw_score > 0 and article_id not in seen_articles:
                    recommendations.setdefault(article_id, []).append(
                        raw_score * sim_weight
                    )

        # Step 7: Sort by average weighted score
        sorted_recs = sorted(
            recommendations.items(),
            key=lambda x: np.mean(x[1]),
            reverse=True
        )[:n]

        # Step 8: Get article details and return
        result_ids = [aid for aid, _ in sorted_recs]
        return self.articles_df[
            self.articles_df['article_id'].isin(result_ids)
        ][['article_id', 'title', 'category']].to_dict('records')

    def recommend(self, user_id, n):
        # Check if user exists in users_df
        user_row = self.users_df[self.users_df['user_id'] == user_id]

        if user_row.empty:
            # User not found — treat as cold start
            print(f"[ColdStart] User {user_id} not found — defaulting to COLD mode")
            return self.cold_mode(user_id, n)

        user_mode = user_row['mode'].values[0]
        n_interactions = user_row['n_interactions'].values[0]

        print(f"[ColdStart] User {user_id} | interactions: {n_interactions} | mode: {user_mode}")

        if user_mode == "HOT":
            return self.hot_mode(user_id, n)
        elif user_mode == "WARM":
            return self.warm_user(user_id, n)
        elif user_mode == "COLD":
            return self.cold_mode(user_id, n)





articles_df = pd.read_csv('data/Mind_data/articles.csv')
users_df = pd.read_csv('data/Mind_data/users.csv')
interactions_df = pd.read_csv('data/Mind_data/interactions.csv')

handler = cold_start_recommendation(articles_df, users_df, interactions_df)
print(handler.hot_mode("U50612", n=5) )