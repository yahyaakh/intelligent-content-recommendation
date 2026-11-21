'''
Example films per semantic-ID code, for the ID-tree figure in Chapter 8.
Builds the main IDs (enriched embedding, 3 levels x 32 codes, k-means seed 42, as in semantic_id_eval.py)
and prints, for every level-1 code, its size and best-known films; then the level-2 and level-3
branches under the chosen codes. "Best-known" = most ratings among the training users. No model calls.

Usage:
    python experiments/semantic_ids/sid_tree_examples.py            overview of all level-1 codes
    python experiments/semantic_ids/sid_tree_examples.py 7 12       branches under level-1 codes 7 and 12
'''

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))  # shared modules live in src/
import sys

CODES = [int(c) for c in sys.argv[1:]]
sys.argv = [sys.argv[0]]
import build_movie_texts as bmt
import cold_start_experiment as cs
from recomm import Semantic_IDs

movies = bmt.load_movies()
ratings = bmt.prepare_ratings(movies)
train, _ = cs.split_users(ratings)
n_ratings = train.groupby("movie_id").size()

model = Semantic_IDs(movies, n_levels=3, codebook_size=32, seed=42)
model.load_embeddings(name="movies_enriched_embeddinggemma_v2")
model.fit()

table = movies[["movie_id", "title", "year", "genres"]].copy()
for level in range(3):
    table[f"c{level + 1}"] = model.codes[:, level]
table["collision"] = model.codes[:, 3]
table["n"] = table["movie_id"].map(n_ratings).fillna(0).astype(int)
table = table.sort_values("n", ascending=False)


def names(group, k):
    return "; ".join(f"{t} ({y})" for t, y in zip(group["title"].head(k), group["year"].head(k)))


if not CODES:
    for code, group in sorted(table.groupby("c1"), key=lambda g: -len(g[1])):
        print(f"[{code:2d}] {len(group):3d} films | {names(group, 6)}")
else:
    for code in CODES:
        group = table[table["c1"] == code]
        print(f"\n=== level 1 code {code}: {len(group)} films | {names(group, 6)}")
        for c2, sub in sorted(group.groupby("c2"), key=lambda g: -len(g[1])):
            print(f"  [{code}-{c2}] {len(sub):3d} films | {names(sub, 5)}")
            for c3, leaf in sorted(sub.groupby("c3"), key=lambda g: -len(g[1]))[:4]:
                print(f"      [{code}-{c2}-{c3}] {len(leaf):2d} | {names(leaf, 3)}")
        # films sharing a full 3-level ID (the collision token tells them apart)
    full = table.groupby(["c1", "c2", "c3"]).size()
    print(f"\nfull IDs shared by >1 film: {(full > 1).sum()} of {len(full)} IDs; max films on one ID: {full.max()}")
