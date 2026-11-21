'''
Semantic IDs for rating prediction on the option 9 time split (warm users only).
Pre-registration: output/experiments/sid_rating/preregistration.md (written before any model ran).

Modes:
  check      split sizes per film slice and ID coverage (no models)
  tune       inner split (train < 2016, validate on 2016): choose lambda / epochs / SVD settings
  test       outer split (train < 2017, test >= 2017): 5 runs of every model, predictions saved
  summarise  RMSE per slice, H1 / H2 / secondary tests with user bootstrap
Usage: python experiments/semantic_ids/sid_rating_experiment.py <mode>
'''

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))  # shared modules live in src/
import sys

MODE = sys.argv[1] if len(sys.argv) > 1 else "check"
sys.argv = [sys.argv[0]]
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from surprise import SVD, BaselineOnly, Dataset, Reader

import build_movie_texts as bmt
import cold_start_experiment as cs
import sid_vector_sources as svs
from recomm import Semantic_IDs

OUT = Path("output") / "experiments" / "sid_rating"
PRED = OUT / "predictions"
OUT.mkdir(parents=True, exist_ok=True)

CUTOFF_INNER = pd.Timestamp("2016-01-01")
CUTOFF_TEST = pd.Timestamp("2017-01-01")
RUNS = 5
DIM, NCF_DIM = 64, 32
BATCH, LR = 4096, 1e-3
MAX_EPOCHS, PATIENCE = 40, 3
LAMBDAS = [0.01, 0.03, 0.1]
SVD_GRID = [dict(n_factors=f, reg_all=r, n_epochs=e, lr_all=0.005)
            for f in (50, 100, 200) for r in (0.02, 0.05) for e in (15, 30)]
CODES, LEVELS = 32, 3
SLICE_BINS = [-1, 0, 20, 100, np.inf]
SLICE_NAMES = ["0 (new)", "1-20", "21-100", ">100"]
LOW = ["0 (new)", "1-20"]
TORCH_MODELS = ["MF atomic", "MF sid", "MF both", "NCF"]
BOOT, BOOT_SEED = 1000, 42


# ───────────────────────── data ─────────────────────────
def load():
    movies = bmt.load_movies()
    ratings = bmt.prepare_ratings(movies)
    warm, cold = cs.split_users(ratings)
    warm = warm.copy()
    warm["date"] = pd.to_datetime(warm["timestamp"], unit="s")
    assert not set(warm["user_id"]) & set(cold["user_id"])
    print(f"Warm users: {warm['user_id'].nunique():,} | ratings: {len(warm):,} | cold users excluded: "
          f"{cold['user_id'].nunique():,}")
    return movies, warm


def split(warm, inner):
    if inner:
        train = warm[warm["date"] < CUTOFF_INNER]
        test = warm[(warm["date"] >= CUTOFF_INNER) & (warm["date"] < CUTOFF_TEST)]
    else:
        train = warm[warm["date"] < CUTOFF_TEST]
        test = warm[warm["date"] >= CUTOFF_TEST]
    test = test[test["user_id"].isin(set(train["user_id"]))].copy()
    test["n_train"] = test["movie_id"].map(train["movie_id"].value_counts()).fillna(0).to_numpy()
    test["slice"] = pd.cut(test["n_train"], SLICE_BINS, labels=SLICE_NAMES).astype(str)
    return train.reset_index(drop=True), test.reset_index(drop=True)


def content_keys(movies, seed):
    """Prefix tokens per film: level 1 = c1, level 2 = (c1, c2), level 3 = (c1, c2, c3)."""
    model = Semantic_IDs(movies, n_levels=LEVELS, codebook_size=CODES, seed=seed)
    model.load_embeddings(name=svs.CONTENT_EMBEDDING)
    model.fit()
    c = np.asarray(model.codes)[:, :LEVELS].astype(np.int64)
    keys = np.stack([c[:, 0], c[:, 0] * CODES + c[:, 1], (c[:, 0] * CODES + c[:, 1]) * CODES + c[:, 2]], axis=1)
    return {int(m): keys[i] for i, m in enumerate(model.movie_ids)}


class Index:
    """User and film positions; films = the whole catalogue, so new films have rows too."""
    def __init__(self, train, movies):
        self.users = {int(u): i for i, u in enumerate(np.sort(train["user_id"].unique()))}
        self.films = {int(m): i for i, m in enumerate(movies["movie_id"].to_numpy())}

    def tensors(self, df):
        u = torch.as_tensor(df["user_id"].map(self.users).to_numpy(), dtype=torch.long)
        i = torch.as_tensor(df["movie_id"].map(self.films).to_numpy(), dtype=torch.long)
        r = torch.as_tensor(df["rating"].to_numpy(), dtype=torch.float32)
        return u, i, r

    def key_matrix(self, keys):
        out = np.zeros((len(self.films), LEVELS), dtype=np.int64)
        for m, row in self.films.items():
            out[row] = keys[m]
        return out


# ───────────────────────── models ─────────────────────────
def emb(n, d, zero=False):
    e = nn.Embedding(n, d)
    (nn.init.zeros_ if zero else (lambda w: nn.init.normal_(w, std=0.01)))(e.weight)
    return e


class MF(nn.Module):
    def __init__(self, n_users, n_items, mu, variant, key_matrix=None):
        super().__init__()
        self.mu, self.atomic, self.sid = mu, variant in ("atomic", "both"), variant in ("sid", "both")
        self.pu, self.bu = emb(n_users, DIM), emb(n_users, 1, zero=True)
        if self.atomic:
            self.qi, self.bi = emb(n_items, DIM), emb(n_items, 1, zero=True)
        if self.sid:
            self.register_buffer("keys", torch.as_tensor(key_matrix))
            sizes = [CODES ** (level + 1) for level in range(LEVELS)]
            self.qt = nn.ModuleList(emb(s, DIM) for s in sizes)
            self.bt = nn.ModuleList(emb(s, 1, zero=True) for s in sizes)

    def forward(self, u, i):
        p, bu = self.pu(u), self.bu(u).squeeze(-1)
        q, b, reg = 0, 0, (p ** 2).sum(-1) + bu ** 2
        if self.atomic:
            qa, ba = self.qi(i), self.bi(i).squeeze(-1)
            q, b, reg = q + qa, b + ba, reg + (qa ** 2).sum(-1) + ba ** 2
        if self.sid:
            k = self.keys[i]
            for level in range(LEVELS):
                ql, bl = self.qt[level](k[:, level]), self.bt[level](k[:, level]).squeeze(-1)
                q, b, reg = q + ql, b + bl, reg + (ql ** 2).sum(-1) + bl ** 2
        return self.mu + bu + b + (p * q).sum(-1), reg


class NCF(nn.Module):
    """NeuMF for ratings: GMF + MLP towers, linear output, plus global mean and biases."""
    def __init__(self, n_users, n_items, mu):
        super().__init__()
        self.mu = mu
        self.gu, self.gi = emb(n_users, NCF_DIM), emb(n_items, NCF_DIM)
        self.mu_, self.mi = emb(n_users, NCF_DIM), emb(n_items, NCF_DIM)
        self.bu, self.bi = emb(n_users, 1, zero=True), emb(n_items, 1, zero=True)
        self.mlp = nn.Sequential(nn.Linear(2 * NCF_DIM, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU())
        self.out = nn.Linear(NCF_DIM + 16, 1)
        nn.init.zeros_(self.out.bias)

    def forward(self, u, i):
        gu, gi, mu_, mi = self.gu(u), self.gi(i), self.mu_(u), self.mi(i)
        bu, bi = self.bu(u).squeeze(-1), self.bi(i).squeeze(-1)
        h = self.mlp(torch.cat([mu_, mi], -1))
        pred = self.mu + bu + bi + self.out(torch.cat([gu * gi, h], -1)).squeeze(-1)
        reg = sum((x ** 2).sum(-1) for x in (gu, gi, mu_, mi)) + bu ** 2 + bi ** 2
        return pred, reg


def make_model(name, n_users, n_items, mu, key_matrix):
    if name == "NCF":
        return NCF(n_users, n_items, mu)
    return MF(n_users, n_items, mu, name.split()[1], key_matrix)


@torch.no_grad()
def predict(model, u, i):
    model.eval()
    out = torch.cat([model(u[s:s + 65536], i[s:s + 65536])[0] for s in range(0, len(u), 65536)])
    return out.clamp(0.5, 5.0).numpy()


def rmse(pred, true):
    return float(np.sqrt(np.mean((np.asarray(pred) - np.asarray(true)) ** 2)))


def train_torch(name, train_t, n_users, n_items, mu, key_matrix, lam, epochs, seed, val_t=None):
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    model = make_model(name, n_users, n_items, mu, key_matrix)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    u, i, r = train_t
    gen = torch.Generator().manual_seed(seed)
    best, history = (np.inf, 0), []
    for epoch in range(1, epochs + 1):
        model.train()
        for b in torch.randperm(len(r), generator=gen).split(BATCH):
            pred, reg = model(u[b], i[b])
            loss = ((pred - r[b]) ** 2).mean() + lam * reg.mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        if val_t is not None:
            score = rmse(predict(model, val_t[0], val_t[1]), val_t[2].numpy())
            history.append(score)
            print(f"    {name} lambda {lam} epoch {epoch}: val RMSE {score:.4f}", flush=True)
            if score < best[0]:
                best = (score, epoch)
            elif epoch - best[1] >= PATIENCE:
                break
    return model, best, history


def surprise_fit(train, algo):
    data = Dataset.load_from_df(train[["user_id", "movie_id", "rating"]], Reader(rating_scale=(0.5, 5)))
    algo.fit(data.build_full_trainset())
    return algo


def surprise_predict(algo, test):
    return np.array([algo.predict(u, m).est for u, m in zip(test["user_id"], test["movie_id"])])


# ───────────────────────── modes ─────────────────────────
def check(movies, warm):
    for inner in (True, False):
        train, test = split(warm, inner)
        label = "inner (train < 2016, validate 2016)" if inner else "outer (train < 2017, test >= 2017)"
        print(f"\n{label}: train {len(train):,} ratings, {train['user_id'].nunique():,} users | "
              f"test {len(test):,} ratings, {test['user_id'].nunique():,} users, {test['movie_id'].nunique():,} films")
        counts = test.groupby("slice").agg(ratings=("rating", "size"), films=("movie_id", "nunique"))
        counts["share"] = (counts["ratings"] / len(test)).round(3)
        print(counts.reindex(SLICE_NAMES).to_string())
    keys = content_keys(movies, 42)
    _, test = split(warm, False)
    missing = set(test["movie_id"]) - set(keys)
    print(f"\nContent IDs: {len(keys):,} films; test films without an ID: {len(missing)}")
    k = np.stack(list(keys.values()))
    print("distinct tokens per level:", [len(np.unique(k[:, level])) for level in range(LEVELS)])


def tune(movies, warm):
    train, val = split(warm, inner=True)
    index = Index(train, movies)
    train_t, val_t = index.tensors(train), index.tensors(val)
    mu = float(train["rating"].mean())
    key_matrix = index.key_matrix(content_keys(movies, 42))
    rows = []
    for name in TORCH_MODELS:
        for lam in LAMBDAS:
            start = time.time()
            _, best, history = train_torch(name, train_t, len(index.users), len(index.films), mu, key_matrix,
                                           lam, MAX_EPOCHS, seed=0, val_t=val_t)
            rows.append(dict(model=name, setting=json.dumps(dict(lam=lam, epochs=best[1])), val_rmse=best[0],
                             minutes=round((time.time() - start) / 60, 1)))
            print(f"  {name} lambda {lam}: best val RMSE {best[0]:.4f} at epoch {best[1]}")
            pd.DataFrame(rows).to_csv(OUT / "tune.csv", index=False)
    for setting in SVD_GRID:
        algo = surprise_fit(train, SVD(random_state=42, **setting))
        score = rmse(surprise_predict(algo, val), val["rating"])
        rows.append(dict(model="SVD tuned", setting=json.dumps(setting), val_rmse=score, minutes=None))
        print(f"  SVD {setting}: val RMSE {score:.4f}")
        pd.DataFrame(rows).to_csv(OUT / "tune.csv", index=False)
    table = pd.DataFrame(rows)
    chosen = {m: json.loads(g.loc[g["val_rmse"].idxmin(), "setting"]) for m, g in table.groupby("model")}
    (OUT / "chosen.json").write_text(json.dumps(chosen, indent=2))
    print("\nChosen:", json.dumps(chosen, indent=2))


def test(movies, warm):
    chosen = json.loads((OUT / "chosen.json").read_text())
    train, test_df = split(warm, inner=False)
    index = Index(train, movies)
    train_t, test_t = index.tensors(train), index.tensors(test_df)
    mu = float(train["rating"].mean())
    PRED.mkdir(exist_ok=True)
    base = test_df[["user_id", "movie_id", "rating", "n_train", "slice"]].copy()
    base["User mean"] = test_df["user_id"].map(train.groupby("user_id")["rating"].mean()).to_numpy()
    base["Baseline"] = surprise_predict(surprise_fit(train, BaselineOnly()), test_df)
    base.to_csv(PRED / "references.csv.gz", index=False)
    for run in range(RUNS):
        path = PRED / f"run{run}.csv.gz"
        if path.exists():
            print(f"run {run}: exists, skipped")
            continue
        key_matrix = index.key_matrix(content_keys(movies, 42 + run))
        out = pd.DataFrame(index=test_df.index)
        for name in TORCH_MODELS:
            start = time.time()
            setting = chosen[name]
            model, _, _ = train_torch(name, train_t, len(index.users), len(index.films), mu, key_matrix,
                                      setting["lam"], setting["epochs"], seed=run)
            out[name] = predict(model, test_t[0], test_t[1])
            print(f"run {run} {name}: RMSE {rmse(out[name], test_df['rating']):.4f} "
                  f"({(time.time() - start) / 60:.1f} min)", flush=True)
        out["SVD tuned"] = surprise_predict(surprise_fit(train, SVD(random_state=42 + run, **chosen["SVD tuned"])), test_df)
        out["SVD option 9"] = surprise_predict(surprise_fit(train, SVD(n_epochs=15, random_state=42 + run)), test_df)
        out.to_csv(path, index=False)
        print(f"run {run} saved")


def rmse_from(sse, cnt):
    return np.sqrt(sse / cnt)


def summarise():
    ref = pd.read_csv(PRED / "references.csv.gz")
    runs = [pd.read_csv(PRED / f"run{r}.csv.gz") for r in range(RUNS)]
    models = list(runs[0].columns)
    slices = {"all": np.ones(len(ref), bool), "low (0-20)": ref["slice"].isin(LOW).to_numpy(),
              **{s: (ref["slice"] == s).to_numpy() for s in SLICE_NAMES}}
    y = ref["rating"].to_numpy()

    rows = []
    for sname, mask in slices.items():
        for m in models:
            vals = [rmse(run[m][mask], y[mask]) for run in runs]
            maes = [float(np.mean(np.abs(run[m][mask] - y[mask]))) for run in runs]
            rows.append(dict(slice=sname, model=m, rmse_mean=np.mean(vals), rmse_sd=np.std(vals),
                             mae_mean=np.mean(maes), ratings=int(mask.sum())))
        for m in ("User mean", "Baseline"):
            rows.append(dict(slice=sname, model=m, rmse_mean=rmse(ref[m][mask], y[mask]), rmse_sd=0,
                             mae_mean=float(np.mean(np.abs(ref[m][mask] - y[mask]))), ratings=int(mask.sum())))
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "summary.csv", index=False)
    print(table.pivot(index="model", columns="slice", values="rmse_mean")[["all", "low (0-20)"] + SLICE_NAMES]
          .round(4).to_string())

    # user bootstrap: per user and run, the squared-error sums of both models on the slice
    users, user_row = np.unique(ref["user_id"], return_inverse=True)
    weights = np.random.default_rng(BOOT_SEED).multinomial(len(users), np.full(len(users), 1 / len(users)), size=BOOT)

    def compare(a, b, mask, level):
        diffs, wins, boot = [], 0, np.zeros(BOOT)
        cnt = np.bincount(user_row[mask], minlength=len(users)).astype(float)
        for run in runs:
            sa = np.bincount(user_row[mask], (run[a].to_numpy()[mask] - y[mask]) ** 2, minlength=len(users))
            sb = np.bincount(user_row[mask], (run[b].to_numpy()[mask] - y[mask]) ** 2, minlength=len(users))
            d = rmse_from(sa.sum(), cnt.sum()) - rmse_from(sb.sum(), cnt.sum())
            diffs.append(d)
            wins += d < 0
            boot += (rmse_from(weights @ sa, weights @ cnt) - rmse_from(weights @ sb, weights @ cnt)) / RUNS
        lo, hi = np.percentile(boot, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
        return dict(mean_diff=float(np.mean(diffs)), wins=int(wins), ci_low=float(lo), ci_high=float(hi),
                    per_run=[round(x, 5) for x in diffs])

    low, every = slices["low (0-20)"], slices["all"]
    tests = {
        "H1 MF both - MF atomic (low)": compare("MF both", "MF atomic", low, 0.95),
        "H2 MF both - MF atomic (all)": compare("MF both", "MF atomic", every, 0.95),
        "S1 MF both - SVD tuned (low)": compare("MF both", "SVD tuned", low, 1 - 0.05 / 3),
        "S2 MF both - NCF (low)": compare("MF both", "NCF", low, 1 - 0.05 / 3),
        "S3 MF sid - MF atomic (low)": compare("MF sid", "MF atomic", low, 1 - 0.05 / 3),
    }
    h1, h2 = tests["H1 MF both - MF atomic (low)"], tests["H2 MF both - MF atomic (all)"]
    tests["H1 holds"] = bool(h1["mean_diff"] <= -0.010 and h1["wins"] == RUNS and h1["ci_high"] < 0)
    tests["H2 holds"] = bool(h2["mean_diff"] <= 0.005)
    for name in ("S1 MF both - SVD tuned (low)", "S2 MF both - NCF (low)", "S3 MF sid - MF atomic (low)"):
        t = tests[name]
        t["verdict"] = "lower" if t["ci_high"] < 0 else "higher" if t["ci_low"] > 0 else "no difference"
    (OUT / "tests.json").write_text(json.dumps(tests, indent=2))
    print(json.dumps(tests, indent=2))


if __name__ == "__main__":
    if MODE == "summarise":
        summarise()
    else:
        movies, warm = load()
        {"check": check, "tune": tune, "test": test}[MODE](movies, warm)
