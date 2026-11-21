'''
The thesis system for the web app: the 1,957-film catalog, the new-user methods of the report, the
constraint layer and the LLM parts. The old engine (engine.py) is left unchanged.

Every method reuses the experiment code, so a user with the same seeds gets the same list as in the
report:
  popularity    count of liked ratings of the warm users                    (cold_start_experiment)
  hybrid_e      item-CF + popularity, alpha per seed count                  (llm_rerank_experiment)
  ease          EASE, lambda per seed count chosen on simulated users       (recomm.EASE_R_Recommende)
  ltr           the final learned ranker: tuned XGBoost + LightGBM on Hybrid E's top 30
                (LightGBM_XBoost_rerank_experiment, features R7). Trained once, saved to disk.

Constraints are applied to the method's whole ranked list (not just its top n), so the user still
gets n films after filtering. The LLM parts:
  parse_request(text)    free text -> constraint dict (recomm.LLMRequestParser, local Ollama)
  llm_rerank=True        the LLM reorders the top 30 of the (filtered) list with prompt p2
'''
import contextlib
import io
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))  # shared modules live in src/

MODEL_FOLDER = ROOT / "output" / "web_models"
K_MAX = 30
LLM_PROVIDER, LLM_MODEL = "ollama_cloud", "deepseek-v4.1-flash"

METHODS = {
    "popularity": "Popularity",
    "hybrid_e": "Hybrid E (item-CF + popularity)",
    "ease": "EASE",
    "ltr": "Learned ranker (XGBoost + LightGBM)",
}
EASE_LAMBDAS = {10: 20000, 3: 120000}       # chosen on the simulated warm users (final_comparison.py)

_engine = None


def get_engine():
    global _engine
    if _engine is None:
        _engine = ThesisEngine()
    return _engine


def _nearest(n, options):
    """The setting tuned for the seed count closest to n (ties go to the smaller count)."""
    return min(sorted(options), key=lambda key: abs(key - n))


class ThesisEngine:
    def __init__(self):
        start = time.perf_counter()
        old_argv = sys.argv
        os.chdir(ROOT)              # the experiment code uses relative paths (engine.py does the same)
        # the experiment scripts read sys.argv when imported
        sys.argv = [old_argv[0], LLM_PROVIDER, LLM_MODEL, "full"]
        try:
            from dotenv import load_dotenv
            load_dotenv(ROOT / ".env")
            import cold_start_experiment as cs
            import llm_rerank_experiment as lr
            import LightGBM_XBoost_rerank_experiment as ltr
            from recomm import EASE_R_Recommende, ConstraintLayer, LLMRequestParser
        finally:
            sys.argv = old_argv
        self.cs, self.lr, self.ltr = cs, lr, ltr

        print("Thesis engine: loading data and similarity tables...")
        data = ltr.load_input()
        data["popular"] = cs.popularity_scores(data["train"])
        data.update(ltr.latent_tables(data["train"], data))       # SVD and co-visitation, all warm users
        self.data = data
        self.pool_ids = set(int(m) for m in data["pool_ids"])

        # Hybrid E has an alpha for 1, 3, 5 and 10 seeds: other counts use the nearest one
        self.alpha_by_count = dict(data["alpha_E"])

        print("Thesis engine: fitting EASE...")
        self.ease = {}
        for n, lam in EASE_LAMBDAS.items():
            model = EASE_R_Recommende(data["train"], data["movies"], lam)
            model.fit(data["train"])
            self.ease[n] = model

        print("Thesis engine: learned ranker...")
        self.ltr_models = {n: self._ranker(n) for n in (10, 3)}
        self.ltr_columns = [c for group in ltr.MODELS[ltr.FINAL_MODEL["features"]] for c in ltr.GROUPS[group]]

        # The catalog with the v5 LLM labels and full credits, as in recomm.main(): the constraint layer
        # needs llm_audience, llm_mood, llm_themes, intensities, cast_full and directors_full
        catalog = pd.read_csv(ROOT / "data/data/processed_30k/movies.csv")
        labels = pd.read_csv(ROOT / "output/experiments/deepseek_v4.1_flash_v5/movies_enriched.csv")
        credits = pd.read_csv(ROOT / "data/data/credits_full/credits_full.csv",
                              usecols=["movie_id", "cast_full", "directors_full"])
        catalog = labels.merge(catalog, on="movie_id", how="inner").merge(credits, on="movie_id", how="inner")
        catalog = catalog.drop(columns=[c for c in ("Unnamed: 0", "enrich_model", "prompt_version", "repaired")
                                        if c in catalog.columns])
        assert len(catalog) == 1957 and catalog["movie_id"].is_unique
        self.catalog = catalog.set_index("movie_id")
        self.constraints = ConstraintLayer(catalog)
        self.parser = LLMRequestParser(self.constraints)
        print(f"Thesis engine ready ({time.perf_counter() - start:.0f} s).")

    # ------------------------------------------------------------------ learned ranker

    def _ranker(self, n):
        """The final ensemble for 10 or 3 seeds, trained on the simulated warm users (as final_test.py).
        Saved after the first fit, so later starts only load it."""
        import joblib
        import json
        MODEL_FOLDER.mkdir(parents=True, exist_ok=True)
        path = MODEL_FOLDER / f"ltr_ensemble_n{n}.joblib"
        if path.exists():
            return joblib.load(path)
        ltr = self.ltr
        suffix = "g5" if n == 10 else f"n{n}"
        train = pd.read_csv(ltr.FOLDER / f"features_simulated_{suffix}.csv")
        with open(ltr.FOLDER / "step4_chosen_params.json", encoding="utf-8") as f:
            chosen = json.load(f)
        columns = [c for group in ltr.MODELS[ltr.FINAL_MODEL["features"]] for c in ltr.GROUPS[group]]
        models = {library: ltr.fit_ranker(train, columns, library, chosen[library])
                  for library in ("xgboost", "lightgbm")}
        joblib.dump(models, path)
        return models

    # ------------------------------------------------------------------ users

    def make_user(self, movie_ratings):
        """The page's [{movie_id, rating}] -> the user dict of the experiments.
        Liked = rated at least the user's own mean rating (the definition used everywhere);
        every picked film counts as seen. Films outside the candidate pool are ignored as seeds."""
        picked = [(int(m["movie_id"]), float(m.get("rating", 5))) for m in movie_ratings]
        mean = np.mean([r for _, r in picked]) if picked else 0
        liked = [m for m, r in picked if r >= mean and m in self.pool_ids]
        return {"user_id": 0, "liked": liked, "seen": {m for m, _ in picked}, "relevant": set()}

    # ------------------------------------------------------------------ ranked lists (whole pool)

    def _with_alpha(self, user):
        n = len(user["liked"])
        self.data["alpha_E"] = {**self.alpha_by_count, n: self.alpha_by_count[_nearest(n, self.alpha_by_count)]}

    def rank_popularity(self, user):
        return self.cs.recommend_popular(self.data["popular"].index, user, len(self.data["pool"]))

    def rank_hybrid_e(self, user):
        """Hybrid E's score for every unseen pool film (lr.hybrid_e_candidates without the cut to 30)."""
        data, cs = self.data, self.cs
        position_of = data["position_of"]
        seen = {position_of[m] for m in user["seen"] if m in position_of}
        cand = np.array([p for p in data["pool"] if p not in seen])
        liked = [position_of[m] for m in user["liked"]]
        personal = cs.rescale(data["item_cf"][np.ix_(liked, cand)].mean(axis=0))
        popularity = cs.rescale(data["pop_vector"][cand])
        alpha = data["alpha_E"][len(user["liked"])]
        score = alpha * personal + (1 - alpha) * popularity
        order = np.argsort(-score, kind="stable")
        return [int(m) for m in data["movie_ids"][cand[order]]]

    def rank_ease(self, user):
        model = self.ease[_nearest(len(user["liked"]), self.ease)]
        ranked = model.recommend(user["liked"], len(model.col_to_film), exclude=user["seen"])
        return [int(m) for m in ranked if int(m) in self.pool_ids and int(m) not in user["seen"]]

    def rank_ltr(self, user):
        """The learned ranker orders Hybrid E's top 30; the rest of the list keeps Hybrid E's order."""
        ltr = self.ltr
        rows = ltr.candidate_rows(user, self.data, with_llm_ranks=False)
        table = ltr.add_g5(pd.DataFrame(rows), [user], self.data)
        models = self.ltr_models[_nearest(len(user["liked"]), self.ltr_models)]
        ranks = [ltr.within_user_rank(table, models[lib].predict(table[self.ltr_columns]))
                 for lib in ("xgboost", "lightgbm")]
        table = table.assign(score=(ranks[0] + ranks[1]) / 2)
        top = table.sort_values("score", ascending=False, kind="stable")["movie_id"].astype(int).tolist()
        rest = [m for m in self.rank_hybrid_e(user) if m not in set(top)]
        return top + rest

    # ------------------------------------------------------------------ main entry

    def recommend(self, movie_ratings, method="hybrid_e", n=10, request=None, llm_rerank=False):
        """Ranked list -> hard constraints -> (optional) LLM reorders the top 30 -> top n.
        Returns the films, the removed films with reasons, and the time and tokens used."""
        start = time.perf_counter()
        user = self.make_user(movie_ratings)
        if not user["liked"]:
            raise ValueError("None of the picked films can be used as a seed (pick films you rated highest).")
        self._with_alpha(user)

        ranked = getattr(self, f"rank_{method}")(user)
        rows = [{"movie_id": m, **self.catalog.loc[m].to_dict()} for m in ranked]
        kept, removed = self.apply_constraints(rows, request or {})
        model_seconds = time.perf_counter() - start

        info = {"method": METHODS[method], "seeds_used": len(user["liked"]), "ranked": len(rows),
                "removed": len(removed), "model_seconds": round(model_seconds, 2)}
        if llm_rerank and kept:
            top_ids = [r["movie_id"] for r in kept[:K_MAX]]
            order, llm_info = self.llm_rerank(user, top_ids)
            by_id = {r["movie_id"]: r for r in kept[:K_MAX]}
            kept = [by_id[m] for m in order] + kept[K_MAX:]
            info.update(llm_info)
        info["total_seconds"] = round(time.perf_counter() - start, 2)
        return kept[:n], removed, info

    def apply_constraints(self, rows, request):
        """ConstraintLayer.apply_hard_constraints without its console printout (the page shows the
        reasons itself, and the printout's symbols cannot be written to a Windows console pipe)."""
        with contextlib.redirect_stdout(io.StringIO()):
            return self.constraints.apply_hard_constraints(rows, {"age": None}, request)

    def llm_rerank(self, user, top_ids):
        """Prompt p2 of the report (seed films + their common genres, directors, actors and decades;
        candidates shuffled), DeepSeek V4.1 Flash. Films the LLM leaves out keep their place at the end."""
        lr = self.lr
        messages, number_to_id = lr.build_prompt(user, top_ids, self.data, "p2")
        answer = lr.call_model(messages, LLM_PROVIDER, LLM_MODEL)
        order, issues = lr.parse_ranking(answer["content"], number_to_id, top_ids)
        return order, {"llm": f"{LLM_MODEL} (prompt p2)", "llm_seconds": round(answer["seconds"], 2),
                       "llm_tokens_in": answer["tokens_in"], "llm_tokens_out": answer["tokens_out"],
                       "llm_issues": issues}

    def parse_request(self, text):
        """Free text -> constraint dict, with the time and tokens of the LLM call."""
        request = self.parser.parse(text)
        info = self.parser.last_info
        return request, {"seconds": round(info.get("seconds", 0), 2),
                         "tokens_in": info.get("prompt_tokens"), "tokens_out": info.get("output_tokens")}

    def options(self):
        """Values for the constraint panel."""
        cl = self.constraints
        clean = lambda values: sorted(v for v in values if isinstance(v, str) and v)
        return {"genres": clean(cl.all_genres), "moods": clean(cl.all_moods), "themes": clean(cl.all_themes),
                "intensity_dims": list(self.parser.INTENSITY_DIMS), "methods": METHODS}

    def search(self, query, limit=8):
        matches = self.catalog[self.catalog["title"].str.contains(query, case=False, na=False, regex=False)]
        matches = matches.assign(pool=matches.index.isin(self.pool_ids)).sort_values(
            ["pool", "vote_count"], ascending=False).head(limit)
        return [{"movie_id": int(m), "title": row["title"], "year": int(row["year"])}
                for m, row in matches.iterrows()]
