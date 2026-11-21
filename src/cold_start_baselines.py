'''
The recommenders of recomm.py (option 10) in the cold-start setup of option 11 (final_comparison.py).

Option 10 evaluates them on a random 75/25 split of ALL users, so its SVD and NCF were trained on the
ratings of the cold users too. Here every model is trained on the 9,591 warm users only, and a new
user is represented by their seed films only, exactly like every other method in option 11.

  Content        recomm.py's dual-matrix content model: 0.75 x CountVectorizer(metadata) +
                 0.25 x TF-IDF(text), cosine; a candidate's score = its similarity to the closest
                 seed (max, the same rule as the embedding content row)
  Collaborative  recomm.py's user-based CF: cosine between mean-centred rating vectors, the 5 most
                 similar warm users, score = mean of (centred rating x similarity) over the
                 neighbours who rated the film above their mean. The new user's vector is +1 on
                 each seed (the setup knows only that the seeds were liked).
                 Because the score is a MEAN, a film liked by one close neighbour can outrank a film
                 liked by all five, and more neighbours make it worse (validation, 10 seeds:
                 5 -> 0.3191, 20 -> 0.1659, 50 -> 0.0691, 200 -> 0.0207).
  User-kNN       the textbook version of the same idea: score = sum over the k most similar warm users
                 of similarity x centred rating, k chosen on the validation users (KNN_NEIGHBOURS)
  SVD            Surprise SVD as in recomm.py (n_epochs 15, random_state 42), trained on the warm
                 users' ratings. Fold-in: the item side is frozen and the new user's bias and factors
                 are the exact minimum of SVD's own loss on the seeds (ridge, lambda = reg_all x seeds).
                 SVD is a rating model, so it gets the seeds' actual ratings: MORE information than
                 the other methods, which only know which films were liked.
  NCF            NeuMF as in recomm.py (n_factors 4, layers [16, 8, 4], 30 epochs, batch 256,
                 lr 1e-3, every rating = a positive), trained on the warm users. Fold-in: every
                 network weight and item embedding is frozen; only the new user's two embeddings
                 (GMF 4 + MLP 8 numbers) are learned on the seeds as positives with 4 sampled
                 negatives each (the training ratio), Adam (steps and learning rate per seed count: FOLD).
  Hybrid         recomm.py's hybrid: the Collaborative top 20, rescored 0.4 x SVD rating
                 (rescaled 0.5-5 -> 0-1) + 0.6 x Collaborative rank; popularity fills any gap.

The trained SVD and NCF are saved in output/experiments/final_comparison/cold_start_models/
(first run: SVD a few minutes, NCF about an hour; later runs load them).
'''
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize

import build_movie_texts as bmt
import cold_start_experiment as cs

FOLDER = Path("output") / "experiments" / "final_comparison" / "cold_start_models"
SEED = 42
NEIGHBOURS = 5            # recomm.py: similar_users = iloc[1:6]
SHORTLIST = 20            # recomm.py hybrid: 20 collaborative candidates
SVD_PARAMS = dict(n_epochs=15, random_state=SEED)     # recomm.py SurpriseRecommender
NCF_PARAMS = dict(model_type="NeuMF", n_factors=4, layer_sizes=[16, 8, 4], n_epochs=30,
                  batch_size=256, learning_rate=1e-3)  # recomm.py NCFRecommender
N_NEG = 4                 # NCF training ratio of negatives to positives (recommenders default)
# Standard user-kNN: neighbours per seed count, chosen on the validation users (best of 7 values, so the
# validation number is slightly optimistic). NDCG@10 per k:
#   k         10      20      50      100     200     400     1000
#   10 seeds  0.5738  0.6098  0.6363  0.6345  0.6417  0.6377  0.6302
#   3 seeds   0.4962  0.5364  0.5792  0.6027  0.6053  0.6013  0.5966
KNN_NEIGHBOURS = {10: 200, 3: 200}

# NCF fold-in (Adam steps, learning rate) per seed count, chosen on the validation users like Hybrid E's
# alpha (10 seeds: 300 / 0.01 -> 0.5703, 1000 / 0.01 -> 0.5782, 300 / 0.05 -> 0.5860, 1000 / 0.05 -> 0.5980,
# 2000 / 0.05 -> 0.5969; 3 seeds: 300 / 0.01 -> 0.5537, 300 / 0.05 -> 0.5448, 1000 / 0.05 -> 0.5310).
# Choosing it on the same users makes NCF's number slightly optimistic, which favours NCF.
FOLD = {10: (1000, 0.05), 3: (300, 0.01)}

NAMES = {"content": "Content (option 10: CountVectorizer + TF-IDF, max)",
         "collaborative": "Collaborative (option 10: user-based CF, 5 neighbours)",
         "knn": "User-kNN (standard: similarity-weighted sum)",
         "svd": "SVD (option 10 model, warm users, fold-in)",
         "ncf": "NCF (option 10 NeuMF, warm users, fold-in)",
         "hybrid": "Hybrid (option 10: CF top 20 rescored with SVD)"}


# ───────────────────────── training (warm users only) ─────────────────────────
def train_svd(train):
    path = FOLDER / "svd_warm.pkl"
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    from surprise import SVD, Dataset, Reader
    print("Training SVD on the warm users (once)...")
    trainset = Dataset.load_from_df(train[["user_id", "movie_id", "rating"]],
                                    Reader(rating_scale=(0.5, 5))).build_full_trainset()
    algo = SVD(**SVD_PARAMS)
    algo.fit(trainset)
    raw_items = [trainset.to_raw_iid(i) for i in range(trainset.n_items)]
    model = {"mu": trainset.global_mean, "bi": algo.bi.copy(), "qi": algo.qi.copy(),
             "item_index": {int(m): i for i, m in enumerate(raw_items)}, "reg": algo.reg_pu}
    FOLDER.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    return model


def ncf_path(seed=SEED):
    """Saved NCF weights; seed 42 keeps its original file name."""
    return FOLDER / ("ncf_warm.npz" if seed == SEED else f"ncf_warm_seed{seed}.npz")


def train_ncf(train, seed=SEED, path=None):
    """NeuMF trained on `train` with this training seed (weights init + negative sampling).
    path: where to save it (default: ncf_path(seed), the model of all warm users)."""
    path = Path(path) if path else ncf_path(seed)
    if path.exists():
        saved = np.load(path)
        return {k: saved[k] for k in saved.files}
    import tensorflow as tf
    from recommenders.models.ncf.ncf_singlenode import NCF
    from recommenders.models.ncf.dataset import Dataset as NCFDataset
    print(f"Training NCF ({path.stem}, {train['user_id'].nunique():,} users, seed {seed}; once, "
          f"about {40 * len(train) / 2.29e6:.0f} minutes)...")
    path.parent.mkdir(parents=True, exist_ok=True)
    csv = path.with_suffix(".train.csv")
    (train.rename(columns={"user_id": "userID", "movie_id": "itemID"})[["userID", "itemID", "rating"]]
     .sort_values("userID").to_csv(csv, index=False))
    dataset = NCFDataset(train_file=str(csv), seed=seed)
    model = NCF(n_users=dataset.n_users, n_items=dataset.n_items, verbose=1, seed=seed, **NCF_PARAMS)
    model.fit(dataset)
    with model.sess.graph.as_default():
        variables = {v.name: v for v in tf.compat.v1.global_variables() if "Adam" not in v.name
                     and "power" not in v.name}
        values = model.sess.run(variables)
    get = lambda part: values[next(n for n in values if n.endswith(part))]
    mlp = sorted((n for n in values if n.startswith("mlp/") and n.endswith("weights:0")),
                 key=lambda n: (len(n), n))                         # fully_connected, _1, _2, ...
    weights = {"gmf_q": get("embedding_gmf_Q:0"), "mlp_q": get("embedding_mlp_Q:0"),
               "out": get("ncf/fully_connected/weights:0"),
               "item_ids": np.array([model.id2item[i] for i in range(dataset.n_items)])}
    for i, name in enumerate(mlp):
        weights[f"w{i}"] = values[name]
        weights[f"b{i}"] = values[name.replace("weights", "biases")]
    np.savez(path, **weights)
    model.sess.close()
    os.remove(csv)
    return weights


# ───────────────────────── reusable scorers (also used by ltr_ncf_knn_experiment.py) ─────────────────────────
class UserKNN:
    """User-based CF over the pool films from the ratings in `train` (mean-centred per user)."""
    def __init__(self, train, column, n_pool):
        rows, user_ids = pd.factorize(train["user_id"])
        centred = (train["rating"] - train.groupby("user_id")["rating"].transform("mean")).to_numpy()
        cols = train["movie_id"].map(column).to_numpy()
        self.centred = csr_matrix((centred, (rows, cols)), shape=(len(user_ids), n_pool))
        self.centred.eliminate_zeros()
        self.unit = normalize(self.centred)
        self.n_users = len(user_ids)

    def similarities(self, seed_cols):
        """Cosine between every user and a new user who is +1 on each seed."""
        x = np.zeros(self.centred.shape[1])
        x[seed_cols] = 1.0 / np.sqrt(len(seed_cols))
        return self.unit @ x

    def knn_scores(self, seed_cols, k, normalise=False):
        """Standard user-kNN: sum over the k most similar users of similarity x centred rating.
        normalise=True divides by the neighbours' total similarity: the same ranking within a user, but
        values comparable across users and across neighbour pools of different size."""
        sims = self.similarities(seed_cols)
        top = np.argsort(-sims, kind="stable")[:k]
        top = top[sims[top] > 0]
        scores = np.asarray(self.centred[top].T @ sims[top]).ravel()
        return scores / max(sims[top].sum(), 1e-12) if normalise else scores


class NCFFoldIn:
    """A trained NeuMF (weights from train_ncf) with every weight frozen; a new user's embeddings are
    learned from their seeds (FOLD steps / learning rate per seed count). Pool films the model never saw
    (possible for a model trained on half the users) get NaN scores."""
    def __init__(self, weights, pool_ids, fold_seed=SEED):
        import torch
        self.torch = torch
        index = {int(m): i for i, m in enumerate(weights["item_ids"])}
        self.known = np.array([int(m) in index for m in pool_ids])
        order = [index.get(int(m), 0) for m in pool_ids]
        t = lambda a: torch.tensor(a, dtype=torch.float32)
        mask = t(self.known.astype(np.float32))[:, None]                  # unseen films: zero embeddings
        self.gmf_q = t(weights["gmf_q"][order]) * mask
        self.mlp_q = t(weights["mlp_q"][order]) * mask
        self.out = t(weights["out"])
        self.layers = [(t(weights[f"w{i}"]), t(weights[f"b{i}"]))
                       for i in range(sum(k.startswith("w") for k in weights))]
        self.fold_seed = fold_seed

    def logits(self, p_gmf, p_mlp, items):
        h = self.torch.cat([p_mlp.expand(len(items), -1), self.mlp_q[items]], dim=1)
        for w, b in self.layers:
            h = self.torch.relu(h @ w + b)
        return (self.torch.cat([p_gmf * self.gmf_q[items], h], dim=1) @ self.out).squeeze(1)

    def scores(self, user_id, seed_cols, allowed):
        """Fold the user in from seed_cols (positives) with negatives drawn from the allowed, known
        films; returns one logit per pool film (NaN for films the model never saw)."""
        torch = self.torch
        g = torch.Generator().manual_seed(self.fold_seed + int(user_id))
        p_gmf = (torch.randn(self.gmf_q.shape[1], generator=g) * 0.01).requires_grad_()
        p_mlp = (torch.randn(self.mlp_q.shape[1], generator=g) * 0.01).requires_grad_()
        steps, lr = FOLD[len(seed_cols)]
        optimiser = torch.optim.Adam([p_gmf, p_mlp], lr=lr)
        negatives_from = torch.tensor(np.flatnonzero(allowed & self.known))
        positives = torch.tensor(seed_cols)
        labels = torch.cat([torch.ones(len(seed_cols)), torch.zeros(N_NEG * len(seed_cols))])
        for _ in range(steps):
            negatives = negatives_from[torch.randint(len(negatives_from), (N_NEG * len(seed_cols),), generator=g)]
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                self.logits(p_gmf, p_mlp, torch.cat([positives, negatives])), labels)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        with torch.no_grad():
            out = self.logits(p_gmf, p_mlp, torch.arange(len(self.known))).numpy().astype(np.float64)
        out[~self.known] = np.nan
        return out


# ───────────────────────── the cold-start versions ─────────────────────────
class ColdStartBaselines:
    def __init__(self, data):
        import torch
        self.torch = torch
        self.data = data
        movie_ids = data["movie_ids"]
        self.pool_ids = np.array([int(movie_ids[p]) for p in data["pool"]])
        self.column = {m: i for i, m in enumerate(self.pool_ids)}
        train = data["train"]

        print("Option 10 models in the cold-start setup: content table, user-CF, SVD, NCF...")
        self.content = bmt.content_similarity(data["movies"], with_enrichment=False)
        self.pool_positions = np.array(data["pool"])

        # user-based CF: warm users x pool films, centred on each user's mean (as recomm.py)
        self.users = UserKNN(train, self.column, len(self.pool_ids))
        self.centred, self.unit = self.users.centred, self.users.unit

        # popularity fallback of the hybrid: the shared popularity order of option 11
        self.popular = [int(m) for m in data["popular"].index]

        # SVD and NCF trained on the warm users; SVD fold-in needs the seeds' own ratings
        self.svd = train_svd(train)
        self.svd_q = self.svd["qi"][[self.svd["item_index"][m] for m in self.pool_ids]]
        self.svd_b = self.svd["bi"][[self.svd["item_index"][m] for m in self.pool_ids]]
        self.seed_rating = self._seed_ratings()

        self.load_ncf(SEED)

    def load_ncf(self, train_seed=SEED, fold_seed=None):
        """Use the NCF trained with train_seed (trained once if missing). fold_seed sets the random start
        and negative samples of the fold-in (default: the training seed, so seed 42 = option 11)."""
        self.ncf_seeds = (train_seed, train_seed if fold_seed is None else fold_seed)
        self.ncf = NCFFoldIn(train_ncf(self.data["train"], train_seed), self.pool_ids, self.ncf_seeds[1])
        assert self.ncf.known.all(), "the warm-user NCF must know every pool film"

    def _seed_ratings(self):
        """{(user_id, movie_id): rating} of every cold user's seed films (the SVD fold-in input)."""
        ratings = bmt.prepare_ratings(self.data["movies"])
        _, cold = cs.split_users(ratings)
        seeds = {(u["user_id"], m) for u in self.data["val_users"] + self.data["test_users"]
                 for m in u["seed_pool"]}
        cold = cold[[k in seeds for k in zip(cold["user_id"], cold["movie_id"])]]
        return {(u, m): r for u, m, r in zip(cold["user_id"], cold["movie_id"], cold["rating"])}

    # each scorer returns one score per pool film (higher = better)
    def collaborative_scores(self, seed_cols):
        sims = self.users.similarities(seed_cols)
        neighbours = np.argsort(-sims, kind="stable")[:NEIGHBOURS]
        block = self.centred[neighbours].toarray()
        positive = block > 0
        total = np.where(positive, block * sims[neighbours, None], 0).sum(axis=0)
        counts = positive.sum(axis=0)
        return np.divide(total, counts, out=np.full(len(self.pool_ids), -np.inf), where=counts > 0)

    def knn_scores(self, seed_cols, neighbours=None):
        """Standard user-based kNN: score = sum over the k most similar warm users of
        similarity x centred rating (0 if unrated). Unlike recomm.py's mean, a film liked by many
        close neighbours scores higher than one liked by a single neighbour."""
        return self.users.knn_scores(seed_cols, neighbours or KNN_NEIGHBOURS[len(seed_cols)])

    def svd_scores(self, user, seed_cols):
        r = np.array([self.seed_rating[(user["user_id"], m)] for m in user["liked"]])
        t = r - self.svd["mu"] - self.svd_b[seed_cols]
        X = np.hstack([np.ones((len(seed_cols), 1)), self.svd_q[seed_cols]])
        lam = self.svd["reg"] * len(seed_cols)                   # SVD's own regularisation, per rating
        w = np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ t)
        return self.svd["mu"] + w[0] + self.svd_b + self.svd_q @ w[1:]

    def ncf_scores(self, user, seed_cols, allowed):
        return self.ncf.scores(user["user_id"], seed_cols, allowed)

    def allowed_mask(self, user):
        allowed = np.ones(len(self.pool_ids), dtype=bool)
        allowed[[self.column[m] for m in user["seen"]]] = False
        return allowed

    def ncf_list(self, user, k=bmt.K):
        """NCF's top k for one cold user (the current load_ncf model and fold-in seed)."""
        allowed = self.allowed_mask(user)
        scores = np.where(allowed, self.ncf_scores(user, [self.column[m] for m in user["liked"]], allowed), -np.inf)
        return self.pool_ids[np.argsort(-scores, kind="stable")[:k]].tolist()

    def lists(self, user, k=bmt.K):
        """Top k of every option-10 method for one cold user (seen films never recommended)."""
        seed_cols = [self.column[m] for m in user["liked"]]
        allowed = np.ones(len(self.pool_ids), dtype=bool)
        allowed[[self.column[m] for m in user["seen"]]] = False

        def top(scores, n=k):
            order = np.argsort(-np.where(allowed, scores, -np.inf), kind="stable")
            order = order[np.isfinite(scores[order]) & allowed[order]]
            return order[:n]

        collaborative = self.collaborative_scores(seed_cols)
        svd = self.svd_scores(user, seed_cols)
        result = {
            NAMES["content"]: bmt.recommend(self.content, user, self.pool_positions, self.data["position_of"],
                                            self.data["movie_ids"], k, "max"),
            NAMES["collaborative"]: self.pool_ids[top(collaborative)].tolist(),
            NAMES["knn"]: self.pool_ids[top(self.knn_scores(seed_cols))].tolist(),
            NAMES["svd"]: self.pool_ids[top(svd)].tolist(),
            NAMES["ncf"]: self.pool_ids[top(self.ncf_scores(user, seed_cols, allowed))].tolist(),
        }
        # Hybrid as recomm.py: Collaborative top 20, 0.4 x SVD rating (clipped like Surprise) + 0.6 x CF rank
        shortlist = top(collaborative, SHORTLIST)
        svd_rating = (np.clip(svd[shortlist], 0.5, 5) - 0.5) / 4.5
        score = 0.4 * svd_rating + 0.6 * (SHORTLIST - np.arange(len(shortlist))) / SHORTLIST
        hybrid = self.pool_ids[shortlist[np.argsort(-score, kind="stable")]].tolist()[:k]
        hybrid += [m for m in self.popular if m not in user["seen"] and m not in hybrid][:k - len(hybrid)]
        result[NAMES["hybrid"]] = hybrid
        return result
