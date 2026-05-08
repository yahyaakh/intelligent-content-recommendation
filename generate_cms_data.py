import pandas as pd
import numpy as np

import random
from datetime import datetime, timedelta

np.random.seed(42)
random.seed(42)

N_ARTICLES = 100
N_USERS = 100

N_INTERACTIONS = 1000

CATEGORIES = [
    'Technology' , 'Science' , 'Sports', 'Entertainment' , 'Health'
      , 'Business', 'Culture' , 'Politics'

]

TAGS_BY_CATEGORY = {
    'Technology': ['AI', 'machine learning', 'software', 'cybersecurity',
                   'cloud computing', 'blockchain', 'mobile', 'data science'],
    'Science':    ['climate', 'biology', 'physics', 'space', 'medicine',
                   'genetics', 'neuroscience', 'chemistry'],
    'Politics':   ['election', 'policy', 'government', 'international',
                   'economy', 'law', 'democracy', 'diplomacy'],
    'Sports':     ['football', 'basketball', 'tennis', 'olympics',
                   'transfer', 'championship', 'athletics', 'fitness'],
    'Entertainment': ['film', 'music', 'celebrity', 'streaming',
                      'gaming', 'books', 'theatre', 'awards'],
    'Health':     ['nutrition', 'mental health', 'fitness', 'medicine',
                   'wellness', 'diet', 'sleep', 'research'],
    'Business':   ['startup', 'investment', 'market', 'economy',
                   'leadership', 'innovation', 'finance', 'strategy'],
    'Culture':    ['art', 'history', 'travel', 'food',
                   'lifestyle', 'education', 'religion', 'society']
}

AUTHORS = [
    'Sarah Mitchell', 'James Chen', 'Emma Rodriguez', 'David Park',
    'Lisa Thompson', 'Michael Brown', 'Anna Kowalski', 'Robert Lee',
    'Maria Santos', 'Tom Williams', 'Julia Fischer', 'Ahmed Hassan',
    'Sophie Martin', 'Carlos Rivera', 'Nina Patel'
]

OCCUPATIONS = [
    'student', 'engineer', 'teacher', 'doctor', 'journalist',
    'manager', 'researcher', 'designer', 'lawyer', 'other'
]

TITLE_TEMPLATES = {
    'Technology': [
        'How {} Is Changing the Tech Industry',
        'The Future of {} in Modern Software',
        'Why {} Matters More Than Ever',
        '{} Explained: A Complete Guide',
        'Top Trends in {} for 2024'
    ],
    'Science': [
        'New Research Reveals Surprising Facts About {}',
        'Scientists Discover {} Could Change Everything',
        'Understanding {}: What the Data Says',
        'The Science Behind {}',
        'How {} Is Reshaping Our Understanding'
    ],
    'Politics': [
        'What the New {} Policy Means for Citizens',
        'Analysis: The Impact of {} on Global Affairs',
        '{} and Its Effects on Modern Democracy',
        'Understanding the {} Debate',
        'How {} Is Shaping the Political Landscape'
    ],
    'Sports': [
        'Breaking: {} Makes Headline Move',
        'Why {} Could Dominate This Season',
        'The Rise of {} in Professional Sports',
        'Analysis: {} Performance This Year',
        'How {} Changed the Game Forever'
    ],
    'Entertainment': [
        'Review: The Latest {} Release',
        'Why {} Is Taking Over Streaming',
        'The Story Behind {}: An Inside Look',
        '{} Breaks Records With New Release',
        'Everything You Need to Know About {}'
    ],
    'Health': [
        'How {} Can Improve Your Daily Life',
        'New Study Links {} to Better Health Outcomes',
        'The Truth About {} and Your Body',
        'Experts Weigh In on {} Benefits',
        'Why {} Should Be Part of Your Routine'
    ],
    'Business': [
        'How {} Is Disrupting the Market',
        'Why Investors Are Betting on {}',
        'The {} Strategy That Changed Everything',
        'Inside the {} Revolution',
        'How to Apply {} Principles to Your Business'
    ],
    'Culture': [
        'Exploring the Rich History of {}',
        'How {} Shapes Modern Society',
        'The Cultural Impact of {}',
        'Why {} Is Having a Moment',
        'Understanding {} Across Different Cultures'
    ]
}


def generate_articles(n):
    articles = []
    for i in range(1, n + 1):
        category = random.choice(CATEGORIES)
        tags = random.sample(TAGS_BY_CATEGORY[category], k=random.randint(2, 4))
        author = random.choice(AUTHORS)
        read_time = random.randint(3, 20)

        # Build title from template
        template = random.choice(TITLE_TEMPLATES[category])
        tag_word = random.choice(tags).title()
        title = template.format(tag_word)

        # Build content_features — mirrors your movies content_features column
        content_features = f"{category} {category} {' '.join(tags)} {author.replace(' ', '_').lower()}"

        pub_date = datetime(2020, 1, 1) + timedelta(days=random.randint(0, 1460))

        articles.append({
            'article_id': i,
            'title': title,
            'category': category,
            'tags': '|'.join(tags),
            'author': author,
            'publication_date': pub_date.strftime('%Y-%m-%d'),
            'read_time': read_time,
            'content_features': content_features
        })
    return pd.DataFrame(articles)


# ─── GENERATE USERS ───────────────────────────────────────────
def generate_users(n):
    users = []
    for i in range(1, n + 1):
        age = random.randint(16, 65)
        occupation = random.choice(OCCUPATIONS)

        # Each user has 1-3 preferred categories
        preferred = random.sample(CATEGORIES, k=random.randint(1, 3))

        users.append({
            'user_id': i,
            'age': age,
            'occupation': occupation,
            'preferred_categories': '|'.join(preferred)
        })
    return pd.DataFrame(users)


# ─── GENERATE INTERACTIONS ────────────────────────────────────
def generate_interactions(users_df, articles_df, n):
    interactions = []
    article_ids = articles_df['article_id'].tolist()

    for _ in range(n):
        user = users_df.sample(1).iloc[0]
        user_id = user['user_id']
        user_prefs = user['preferred_categories'].split('|')

        # Users are more likely to read articles in their preferred categories
        preferred_articles = articles_df[
            articles_df['category'].isin(user_prefs)
        ]['article_id'].tolist()

        if preferred_articles and random.random() < 0.7:
            article_id = random.choice(preferred_articles)
        else:
            article_id = random.choice(article_ids)

        # Rating biased toward higher if preferred category
        article_category = articles_df[
            articles_df['article_id'] == article_id
        ]['category'].values[0]

        if article_category in user_prefs:
            rating = random.choices([3, 4, 5], weights=[0.2, 0.4, 0.4])[0]
            completion = round(random.uniform(0.6, 1.0), 2)
        else:
            rating = random.choices([1, 2, 3, 4], weights=[0.2, 0.3, 0.3, 0.2])[0]
            completion = round(random.uniform(0.1, 0.7), 2)

        timestamp = datetime(2020, 1, 1) + timedelta(days=random.randint(0, 1460))

        interactions.append({
            'user_id': user_id,
            'article_id': article_id,
            'rating': rating,
            'read_completion': completion,
            'timestamp': timestamp.strftime('%Y-%m-%d')
        })

    return pd.DataFrame(interactions)


# ─── MAIN ─────────────────────────────────────────────────────
if __name__ == '__main__':
    print("Generating articles...")
    articles_df = generate_articles(N_ARTICLES)

    print("Generating users...")
    users_df = generate_users(N_USERS)

    print("Generating interactions...")
    interactions_df = generate_interactions(users_df, articles_df, N_INTERACTIONS)

    # Remove duplicate user-article pairs
    interactions_df = interactions_df.drop_duplicates(
        subset=['user_id', 'article_id']
    )

    articles_df.to_csv('data/cms/articles.csv', index=False)
    users_df.to_csv('data/cms/users.csv', index=False)
    interactions_df.to_csv('data/cms/interactions.csv', index=False)

    print(f"\nDataset generated:")
    print(f"  Articles:     {len(articles_df)}")
    print(f"  Users:        {len(users_df)}")
    print(f"  Interactions: {len(interactions_df)}")
    print(f"  Avg interactions per user: {len(interactions_df)/len(users_df):.1f}")
    print(f"\nSaved to data/cms/")