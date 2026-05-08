# Universal Recommendation System

A domain-agnostic recommendation engine supporting multiple content domains (movies, news articles), featuring a **Constraint Layer**, cold start handling, and five recommendation algorithms. Built as a university thesis project.

---

## What This Project Does

Most recommendation systems are built for one specific domain — Netflix recommends movies, Spotify recommends songs. This system is built differently: the core algorithms work on **any type of content** without modification. The same pipeline that recommends movies also recommends news articles, and can be extended to stocks, books, or any rated items.

The system was built in two phases:

1. **Movie Recommendation System** — prototype domain using real MovieLens 100K ratings and TMDB movie metadata
2. **CMS Cold Start Demo** — parallel implementation on news articles using the Microsoft MIND dataset, demonstrating the system's domain-agnostic design

The academic contribution is a **Constraint Layer** that sits above all recommendation algorithms and enforces explicit rules — something pure ML models cannot do on their own.

---

## Key Numbers

| Metric | Value |
|--------|-------|
| Movies | 973 (fetched from TMDB API) |
| Real ratings | 19,201 (MovieLens 100K) |
| Users | 943 (with demographics) |
| Recommendation algorithms | 5 |
| Constraint rules | 8 (4 hard + 4 soft) |
| News articles (CMS demo) | 5,000 (Microsoft MIND) |
| Automated test scenarios | 10 |
| SVD NDCG@10 | 0.9815 |

---

## Quick Start

### Requirements
- Python 3.9+
- Conda environment recommended

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run the movie recommendation system
```bash
python recomm.py
```

### Run the CMS cold start demo
```bash
python cold_start_demo.py
```

### Run the automated test suite
```bash
python test_recommender.py
```

### Generate all charts
```bash
python generate_charts.py
```

---

## Project Structure

```
recomm 1/
│
├── recomm.py                   ← Main movie recommendation system (CLI)
├── test_recommender.py         ← Automated test suite (10 scenarios)
├── generate_cms_data.py        ← Synthesize CMS dataset (articles, users, ratings)
│
├── cold_start_handler.py       ← CMS cold start recommender (3-stage model)
├── cold_start_demo.py          ← Cold start demo with formatted terminal output
├── prepare_mind_data.py        ← Parses Microsoft MIND dataset into usable CSVs
│
├── requirements.txt            ← Python dependencies
├── .env                        ← API keys (TMDB_API_KEY — not committed)
│
├── data/
│   ├── data/
│   │   ├── processed/          ← Ready-to-use processed data
│   │   │   ├── movies.csv      ← 973 TMDB movies with full metadata
│   │   │   ├── ratings.csv     ← 19,201 real MovieLens ratings
│   │   │   └── users.csv       ← 943 users with demographics
│   │   └── raw/
│   │       └── ml-100k/        ← Original MovieLens 100K raw files
│   │
│   ├── cms/                    ← Synthesized CMS dataset (for research)
│   │   ├── articles.csv        ← 300 synthesized news articles
│   │   ├── interactions.csv    ← Synthesized user interactions
│   │   └── users.csv           ← Synthesized users
│   │
│   ├── Mind_data/              ← Microsoft MIND dataset (real news)
│   │   ├── raw/
│   │   │   ├── news.tsv        ← Raw MIND news articles
│   │   │   └── behaviors.tsv  ← Raw MIND user click behaviour
│   │   ├── articles.csv        ← Processed: 5,000 real news articles
│   │   ├── interactions.csv    ← Processed: 8,185 real click events
│   │   └── users.csv           ← Processed: 1,626 users with cold start modes
│   │
│   └── old_data/               ← Original professor baseline data
│       ├── movies.csv          ← Original movie list (minimal metadata)
│       └── users.csv           ← Original user list
│
├── data/                       ← Data pipeline scripts (root-level data folder)
│   ├── fetch_tmdb.py           ← Fetches movie data from TMDB API
│   ├── process_tmdb.py         ← Converts raw TMDB JSON to movies.csv
│   └── prepare_movielens.py    ← Downloads MovieLens and matches to TMDB
│
├── notebooks/                  ← Jupyter notebooks (exploration)
└── src/                        ← Source modules (evaluation scripts)
```

---

## File-by-File Explanation

### `recomm.py` — Main System
The core of the project. Contains all recommendation classes and launches an interactive CLI. Run this to use the movie recommendation system.

**What's inside:**
- `ContentBasedRecommender` — dual-matrix TF-IDF similarity (genres, cast, director, keywords, plot)
- `CollaborativeRecommender` — mean-centered cosine similarity between users
- `SurpriseRecommender` — SVD matrix factorization via the Surprise library
- `HybridRecommenderEnhanced` — combines content + collaborative scores
- `MovieRecommendationSystem` — popularity ranking with multi-signal scoring
- `ConstraintLayer` — hard and soft constraint enforcement (the thesis contribution)
- `MovieRecommenderCLI` — interactive menu with 9 options

**CLI Menu:**
```
1. Get user recommendations      (choose algorithm: hybrid/collaborative/content/popular/surprise)
2. Simulate watching a movie     (add interaction to user history)
3. Find similar movies           (content-based similarity)
4. Show user profile             (taste profile: directors, decades, genres, runtime)
5. Compare recommendation algorithms   (all 5 side-by-side)
6. Evaluate recommendation quality
7. Show popular movies
8. Exit
9. Evaluate Surprise SVD metrics (RMSE, MAE, Precision, Recall, NDCG)
```

---

### `test_recommender.py` — Automated Test Suite
Runs 10 test scenarios automatically and prints a formatted report. No manual input required. Completes in under 1 second.

**Tests covered:**
1. Data integrity — checks for nulls, correct columns, rating distribution
2. User profiles — builds profiles for 3 different user types
3. Content similarity — tests 4 seed movies (Star Wars, Godfather, Interstellar, GoodFellas)
4. Collaborative filtering — verifies mean-centering works correctly
5. Algorithm comparison — runs all 5 algorithms and measures overlap
6. Child safety — verifies 0 adult content violations for User 30 (age 7)
7. Hard constraint detail — runtime ceiling and genre overlap
8. Soft constraints — director/decade/genre boost verification
9. Cold start simulation — 0→20 interaction journey
10. SVD evaluation metrics — RMSE, MAE, Precision@10, Recall@10, NDCG@10

**Usage:**
```bash
python test_recommender.py
```

---

### `cold_start_handler.py` — CMS Recommender
Handles the cold start problem for the CMS (news articles) domain. Contains one class `cold_start_recommendation` with four methods.

**Three recommendation modes:**

| Mode | Trigger | Algorithm | Reason |
|------|---------|-----------|--------|
| COLD | 0–2 clicks | Popularity-based | No personal signal available |
| WARM | 3–9 clicks | Content-based TF-IDF | Topic interest detected |
| HOT | 10+ clicks | Collaborative filtering | Behavioural patterns usable |

**Key research finding documented here:** Mean-centering (used in the movie system) does not apply to binary click data. When all positive signals are exactly 1, subtracting the user mean makes every centered score = 0. Raw cosine similarity is used instead for implicit feedback.

**Methods:**
- `get_user_mode(user_id)` — returns COLD/WARM/HOT based on interaction count
- `cold_mode(user_id, n)` — popularity-based recommendations
- `warm_user(user_id, n)` — TF-IDF content similarity
- `hot_mode(user_id, n)` — collaborative filtering
- `recommend(user_id, n)` — main entry point, routes automatically

---

### `cold_start_demo.py` — Cold Start Demo
Demonstrates the three-stage transition model in a formatted terminal output. Picks one real user from each mode (COLD, WARM, HOT) and shows what each strategy recommends and why.

**Usage:**
```bash
python cold_start_demo.py
```

**Output includes:**
- Dataset overview (article counts, category distribution, user mode breakdown)
- Stage 1: COLD user — popularity-based recommendations
- Stage 2: WARM user — content-based with dominant interest detected
- Stage 3: HOT user — collaborative filtering
- Transition threshold summary table

---

### `prepare_mind_data.py` — MIND Dataset Parser
One-time script that converts the raw Microsoft MIND dataset files into clean CSVs used by `cold_start_handler.py`.

**Input:** `data/Mind_data/raw/news.tsv` + `data/Mind_data/raw/behaviors.tsv`

**Output:**
- `data/Mind_data/articles.csv` — 5,000 articles with title, category, content_features
- `data/Mind_data/interactions.csv` — click events (user_id, article_id, clicked, timestamp)
- `data/Mind_data/users.csv` — users with interaction counts and cold start mode

**How it works:**
1. Reads `news.tsv` — keeps only article_id, category, title, abstract
2. Builds `content_features` column (category x2 + title + abstract) for TF-IDF
3. Reads `behaviors.tsv` — extracts clicks from both history column (past sessions) and impressions column (current session, where -1 = clicked)
4. Assigns COLD/WARM/HOT mode to each user based on click count
5. Saves all three output files

**Run only needed if raw MIND files are updated:**
```bash
python prepare_mind_data.py
```

---

### `generate_cms_data.py` — Synthetic CMS Dataset Generator
Generates a synthetic CMS dataset for research and testing purposes. Useful when you need a clean, controlled dataset without downloading external data.

**Output:** `data/cms/articles.csv`, `data/cms/users.csv`, `data/cms/interactions.csv`

**What it generates:**
- Articles with realistic titles, categories, tags, authors, publication dates
- 8 categories: Technology, Science, Politics, Sports, Entertainment, Health, Business, Culture
- Users with age and occupation preferences
- Interactions biased toward user's preferred categories (70% preferred, 30% random)

**Usage:**
```bash
python generate_cms_data.py
```

---

### Data Pipeline Scripts

#### `fetch_tmdb.py`
Connects to the TMDB API and fetches movie metadata. Uses the `append_to_response=credits,keywords` trick — one API call per movie instead of three. Requires a TMDB API key in `.env`.

**Output:** Raw JSON files saved locally

**Run to re-fetch movie data:**
```bash
python fetch_tmdb.py
```

#### `process_tmdb.py`
Converts raw TMDB JSON into the structured `movies.csv` used by the recommendation system.

**Key processing:**
- Extracts genres, director, top 3 cast, top 10 keywords, overview, runtime, language
- Genres repeated twice in `content_features` for TF-IDF weight boost
- Builds two separate feature columns: `metadata_features` (for CountVectorizer) and `text_features` (for TF-IDF)

**Output:** `data/data/processed/movies.csv`

#### `prepare_movielens.py`
Downloads MovieLens 100K and matches it to TMDB movies using fuzzy title matching.

**Matching strategy:**
- SequenceMatcher similarity threshold: 0.85
- Year validation: ±1 year (prevents Toy Story → Tokyo Story false matches)
- Result: 113 matched movies, 19,201 real ratings

**Output:** `data/data/processed/ratings.csv`, updated `movies.csv`

---

## The Constraint Layer — Thesis Contribution

The Constraint Layer is the core academic contribution. It intercepts recommendation outputs and applies two types of rules:

### Hard Constraints (absolute filters — movie is removed)
| Constraint | Rule |
|-----------|------|
| Age protection | Users under 13: block Horror, Thriller, Crime, War genres |
| Child safety keywords | 100+ unsafe keywords (nazi, rape, serial killer, drugs, etc.) |
| Runtime ceiling | Movie runtime > 1.5× user's average tolerance |
| Genre overlap | Movie shares zero genres with user's entire watch history |

### Soft Constraints (score adjustments — movie is re-ranked)
| Constraint | Effect |
|-----------|--------|
| Director affinity | ×1.3 boost if director is in user's liked directors |
| Decade preference | ×1.2 boost for preferred decade, ×0.9 penalty for distant era |
| Genre affinity | +10% per genre matching user's liked genres |
| Language preference | ×0.8 penalty for non-preferred language |

**Test result:** Zero adult content violations for User 30 (age 7). Director boost verified at 1.404 vs 1.131 average — a 24% measurable improvement in ranking quality for preferred directors.

---

## Algorithms

### 1. Content-Based (Dual-Matrix)
The original single TF-IDF approach was replaced with a dual-matrix architecture:
- **Pipeline 1:** CountVectorizer on hard metadata (genres ×2, cast, director, keywords) — no IDF penalty on common genres
- **Pipeline 2:** TF-IDF on plot overview — semantic topic matching
- **Blend:** 75% metadata + 25% plot text

### 2. Collaborative Filtering (Mean-Centered)
- Builds a user-movie rating matrix
- Subtracts each user's personal mean before computing cosine similarity
- Fixes the "generous vs harsh rater" problem — users with different rating scales but identical taste are now correctly identified as similar

### 3. SVD (Surprise Library)
- Matrix factorization via Singular Value Decomposition
- 15 training epochs, fixed random seed for reproducibility
- Results: RMSE 0.89, NDCG@10 0.98

### 4. Hybrid
- Combines content-based and collaborative scores
- 40% content signal + 60% collaborative signal
- Falls back to popularity for new users

### 5. Popularity-Based
- Multi-signal ranking: avg rating (30%) + rating count (30%) + view count (20%) + completion rate (20%)
- Used as fallback and cold start baseline

---

## Data Sources

| Dataset | Source | Size | License |
|---------|--------|------|---------|
| MovieLens 100K | GroupLens Research, University of Minnesota | 100,000 ratings, 943 users | Free for research |
| TMDB API | The Movie Database (themoviedb.org) | 980 movies | Free for non-commercial |
| Microsoft MIND | Microsoft Research (msnews.github.io) | 65,000+ articles, 50,000 users | Microsoft Research License |

---

## Original vs New System

| Component | Original Baseline | This Work |
|-----------|-------------------|-----------|
| Ratings data | Randomly generated with numpy | 19,201 real MovieLens ratings |
| Movie metadata | Basic CSV (genre, director, year) | 13 columns: cast, keywords, overview, language, runtime, vote_average |
| Data pipeline | None | 3 reproducible scripts |
| Content similarity | Single TF-IDF on genre+director+decade | Dual-matrix: CountVectorizer + TF-IDF, 75/25 blend |
| Collaborative filtering | Raw cosine, threshold = 3.5 (absolute) | Mean-centered cosine, threshold = 0 (relative to user) |
| User profile | Basic stats (total movies, avg rating) | Rich profile: liked directors, preferred decade, genre history, age, runtime tolerance |
| Constraint layer | None | 4 hard + 4 soft constraints |
| Child safety | None | Age detection + 100+ keyword blocklist |
| CMS support | None | Full MIND pipeline + 3-stage cold start |

---

## Test Results Summary

All 10 automated tests pass:

```
Test 1  — Data integrity          PASS   36 null values, 19,201 real ratings, 2.09% density
Test 2  — User profiles           PASS   3 user types profiled correctly
Test 3  — Content similarity      PASS   100% genre relevance (Star Wars, Godfather, Interstellar, GoodFellas)
Test 4  — Collaborative filtering PASS   Mean-centering verified (User 1 similar users: avg 3.41-3.78)
Test 5  — Algorithm comparison    NOTE   Hybrid = Collaborative (known issue — content signal not contributing)
Test 6  — Child safety            PASS   0 adult violations for age-7 user
Test 7  — Hard constraints        PASS   Runtime ceiling and genre overlap working
Test 8  — Soft constraints        PASS   Director boost: 1.404 vs 1.131 avg (24% boost verified)
Test 9  — Cold start simulation   PASS   COLD → WARM → HOT transition demonstrated
Test 10 — SVD metrics             PASS   RMSE 0.89 | NDCG@10 0.98 | Precision@10 0.45 | Recall@10 0.97
```

---

## Known Limitations

1. **Hybrid = Collaborative** — The hybrid recommender currently produces identical results to collaborative filtering alone. The content branch score is being overridden by collaborative signal. Fix planned: blend SVD scores directly into hybrid weights.

2. **Content seed selection** — For some users, warm mode seeds from a movie that produces off-topic recommendations. Fix planned: use most frequent category rather than highest-rated single movie as seed.

3. **Temporal evaluation** — SVD metrics use a random 80/20 split. A temporal split (train on older ratings, test on newer) would be more realistic.

---

## Academic Context

This project was developed as a university thesis exploring **constraint-based recommendation systems**. The core contribution is a domain-agnostic Constraint Layer that applies hard and soft rules above any recommendation algorithm — related to:

- Burke, R. (2000). *Knowledge-based recommender systems*
- Constraint Satisfaction Problems (CSP) applied to item ranking
- The distinction between explicit feedback (ratings) and implicit feedback (clicks) in collaborative filtering

The system has been implemented and tested on two domains (movies and news articles) to demonstrate domain-agnostic design. The cold start research identifies that **86% of real users** in the MIND dataset are in cold or warm state at any given moment, making cold start handling a critical practical concern.

---

## Environment Setup

Create a `.env` file in the project root with your TMDB API key:
```
TMDB_API_KEY=your_api_key_here
```

Get a free API key at: https://www.themoviedb.org/settings/api

The processed data is already included in the repository, so the API key is only needed if you want to re-fetch movie data from scratch.

---

## Dependencies

Key libraries used:

```
pandas          — data manipulation
numpy           — numerical operations
scikit-learn    — TF-IDF, cosine similarity, CountVectorizer
surprise        — SVD matrix factorization
matplotlib      — chart generation
seaborn         — statistical visualisation
```

Full list in `requirements.txt`.
