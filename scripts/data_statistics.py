"""Table 3.1 of the report: statistics of the rating data, plus the MovieLens 100K / 25M split.

Uses the same loaders as every experiment (build_movie_texts.load_movies / prepare_ratings and
cold_start_experiment.split_users), so the numbers describe exactly the data the methods see.

Usage:
    python scripts/data_statistics.py
Output:
    output/experiments/data_statistics/table_3_1.csv
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # shared modules live in src/
import sys
from pathlib import Path

import pandas as pd

sys.argv = [sys.argv[0]]                 # the imported experiment scripts read sys.argv when imported
import build_movie_texts as bmt
import cold_start_experiment as cs

FOLDER = Path("output") / "experiments" / "data_statistics"
ML100K_LAST_USER_ID = 943                # prepare_movielens25m.py shifts MovieLens 25M user ids by 943


def main():
    movies = bmt.load_movies()
    ratings = bmt.prepare_ratings(movies)
    train, cold = cs.split_users(ratings)

    per_user = ratings.groupby("user_id").size()
    rated = ratings["movie_id"].nunique()
    liked = ratings["rating"] >= ratings.groupby("user_id")["rating"].transform("mean")
    years = pd.to_datetime(ratings["timestamp"], unit="s").dt.year
    from_100k = ratings["user_id"] <= ML100K_LAST_USER_ID

    rows = [
        ("Films in the catalog", len(movies)),
        ("Films with at least one rating", rated),
        ("Films rated by warm users (candidate pool)", train["movie_id"].nunique()),
        ("Users", ratings["user_id"].nunique()),
        ("Ratings", len(ratings)),
        ("Ratings per user (median)", per_user.median()),
        ("Density of the rating matrix (rated films only)", len(ratings) / (per_user.size * rated)),
        ("Mean rating", ratings["rating"].mean()),
        ("Share of ratings that count as liked", liked.mean()),
        ("Rating period", f"{years.min()}-{years.max()}"),
        ("MovieLens 100K users / ratings", f"{ratings.loc[from_100k, 'user_id'].nunique()} / {from_100k.sum()}"),
        ("MovieLens 25M users / ratings", f"{ratings.loc[~from_100k, 'user_id'].nunique()} / {(~from_100k).sum()}"),
        ("MovieLens 100K ratings per user (median)", per_user[per_user.index <= ML100K_LAST_USER_ID].median()),
        ("Warm users / ratings", f"{train['user_id'].nunique()} / {len(train)}"),
    ]
    table = pd.DataFrame(rows, columns=["statistic", "value"])
    print(table.to_string(index=False))
    FOLDER.mkdir(parents=True, exist_ok=True)
    table.to_csv(FOLDER / "table_3_1.csv", index=False)


if __name__ == "__main__":
    main()
