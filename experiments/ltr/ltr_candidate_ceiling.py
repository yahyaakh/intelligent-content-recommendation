'''
Quick, descriptive check (no model fits): would semantic-ID picks raise the ceiling of LTR v2's candidate list?

LTR v2 reranks Hybrid E's top 30 (C30). Wider lists were tried twice and not adopted:
  doc 16 step 3: C30 + 20 by content similarity;  doc 24 option C: C30 + 20 by user-kNN.
Here, on the 9,084 cross-fitted simulated users (10 and 3 seeds), C30 is extended by the 20 best films
outside it according to:
  content     emb_max to the seeds (the doc 16 list, via ltr.wide_candidates)
  EASE IDs    shared-prefix score of the S4 EASE semantic IDs (k-means seed 42, popularity ties), doc 30
  EASE        the EASE model itself (lambda 20,000), for reference
All are built from the OTHER half's ratings, like every other feature of these users.

Measured per list: oracle NDCG@10 (every relevant candidate ranked first = the best a reranker could do),
relevant films in the list, and how many relevant films the 20 picks add.
Rule written before running: the full LTR test is only worth running if the EASE-ID picks add clearly more
relevant films than the content picks (which gave no validation gain in doc 16).

Usage: python experiments/ltr/ltr_candidate_ceiling.py      outputs in output/experiments/ltr_candidate_ceiling/
'''

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))  # shared modules live in src/
import sys

sys.argv = [sys.argv[0]]
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import build_movie_texts as bmt
import cold_start_experiment as cs
import llm_rerank_experiment as lr
import LightGBM_XBoost_rerank_experiment as ltr
import ease_tuning as et
import sid_vector_sources as svs

OUT = Path("output") / "experiments" / "ltr_candidate_ceiling"
EXTRA = 20
K = bmt.K
OTHER = {"A": "B", "B": "A"}


def oracle_ndcg(candidates, relevant):
    hits = min(len(relevant & set(candidates)), K)
    ideal = sum(1 / np.log2(i + 2) for i in range(min(len(relevant), K)))
    return sum(1 / np.log2(i + 2) for i in range(hits)) / ideal if ideal else 0.0


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = lr.load_data()
    data["emb_base"] = bmt.embedding_similarity("movies_base_embeddinggemma_v2", data["movies"])   # as ltr.load_input
    movies = data["movies"]
    pool_ids = set(int(data["movie_ids"][p]) for p in data["pool"])
    rows = []
    for n in (10, 3):
        train, in_A, users_by_half = ltr.warm_halves(data, n)
        parts = {"A": train[in_A], "B": train[~in_A]}
        for half in ("A", "B"):
            other = parts[OTHER[half]]
            view = ltr.half_view(data, other)
            ease = et.fit_on(other, svs.EASE_LAMBDA)
            vec, ids = np.load(svs.OUT / "vectors" / f"S4_{OTHER[half]}.npy"), np.load(svs.OUT / "vectors" / f"S4_{OTHER[half]}_ids.npy")
            sid = svs.fit_ids(movies, vec, ids, "S4 EASE", OTHER[half], 42, save=False)
            pop_tie = cs.popularity_scores(other).reindex(sid.movie_ids).fillna(0).to_numpy()
            for user in tqdm(users_by_half[half], desc=f"{n} seeds, half {half}", leave=False):
                relevant = set(user["relevant"])
                c30, _, _ = ltr.wide_candidates(user, view, "C30")
                c50, _, _ = ltr.wide_candidates(user, view, "C30+20")
                in30 = set(c30)
                picks = {"content (doc 16)": [m for m in c50 if m not in in30]}
                ranked = sid.recommend(user["liked"], len(sid.movie_ids), exclude=user["seen"],
                                       allowed=pool_ids - in30, weights=[1, 1, 1], tie_break=pop_tie)
                picks["EASE IDs"] = [int(m) for m in ranked[:EXTRA]]
                ranked = ease.recommend(user["liked"], len(pool_ids), exclude=user["seen"])
                picks["EASE"] = [int(m) for m in ranked if int(m) in pool_ids and int(m) not in in30][:EXTRA]
                base = {"seeds": n, "half": half, "user_id": user["user_id"]}
                rows.append({**base, "list": "C30", "oracle": oracle_ndcg(c30, relevant),
                             "relevant_in_list": len(relevant & in30), "relevant_added": 0, "candidates": len(c30)})
                for name, extra in picks.items():
                    full = c30 + extra
                    rows.append({**base, "list": f"C30 + 20 {name}", "oracle": oracle_ndcg(full, relevant),
                                 "relevant_in_list": len(relevant & set(full)),
                                 "relevant_added": len(relevant & set(extra)), "candidates": len(full)})
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "per_user.csv", index=False)
    summary = frame.groupby(["seeds", "list"], sort=False)[["candidates", "oracle", "relevant_in_list",
                                                           "relevant_added"]].mean().round(4)
    summary.to_csv(OUT / "summary.csv")
    pd.set_option("display.width", 200)
    print(summary.to_string())
    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    main()
