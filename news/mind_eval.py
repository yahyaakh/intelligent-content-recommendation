'''
One evaluation module for every MIND news model: loading, metrics, saved results and leaderboard submissions.

Metrics are those of the official leaderboard (msnews/MIND evaluate.py): per impression AUC, MRR, nDCG@5 and
nDCG@10, then the mean over impressions. As on the leaderboard, scores are first turned into ranks (1 = top,
ties broken at random with a fixed seed) and the metrics are computed from the ranks, so a dev score here is
computed exactly like a test score on Codabench. `selftest` checks this against the official code.

A model only has to produce one score per candidate, in file order:

    import mind_eval as me
    dev = me.load_impressions("dev")             # Impressions: ids, users, times, histories, candidates, labels
    scores = my_model.score(dev)                 # flat array, len == dev.n_candidates (or a list of arrays)
    me.evaluate(dev, scores, name="my_model")    # prints and saves the metrics

    test = me.load_impressions("test")
    me.write_submission(test, my_model.score(test), name="my_model")   # output/news/submissions/my_model/prediction.zip

Data: data/Mind_Large_Data/MINDlarge_{train,dev,test}/... (scripts/download_mind.py).
Results: output/news/results/<split>/<name>/{metrics.json, per_impression.csv.gz} and one summary row per run in
output/news/results/<split>_summary.csv.

Usage (from the repository root):
    python news/mind_eval.py selftest     check the metrics against the official script, plus a random baseline on dev
    python news/mind_eval.py summary      print the dev summary table
'''
import json
import re
import sys
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "Mind_Large_Data"
CACHE = ROOT / "output" / "news" / "cache"
RESULTS = ROOT / "output" / "news" / "results"
SUBMISSIONS = ROOT / "output" / "news" / "submissions"
SPLITS = ("train", "dev", "test")
METRICS = ("AUC", "MRR", "nDCG@5", "nDCG@10")
TIE_SEED = 0


# ---------------------------------------------------------------------------------------------------- data
@dataclass
class Impressions:
    """All impressions of one split as flat arrays. Candidate j of impression i is
    candidates[offsets[i] + j]; the same layout holds for labels and for a model's scores."""
    split: str
    ids: np.ndarray            # impression id (int), file order
    users: np.ndarray          # user id as int (U12345 -> 12345)
    times: np.ndarray          # datetime64[s]
    history: np.ndarray        # clicked news before the impression (int ids N12345 -> 12345), flat
    history_offsets: np.ndarray
    candidates: np.ndarray     # candidate news (int ids), flat
    offsets: np.ndarray        # len = n_impressions + 1
    labels: np.ndarray | None  # 1 = clicked, flat; None for the test split (labels are hidden)

    @property
    def n_impressions(self):
        return len(self.ids)

    @property
    def n_candidates(self):
        return len(self.candidates)

    @property
    def sizes(self):
        return np.diff(self.offsets)

    def history_of(self, i):
        return self.history[self.history_offsets[i]:self.history_offsets[i + 1]]

    def candidates_of(self, i):
        return self.candidates[self.offsets[i]:self.offsets[i + 1]]

    def labels_of(self, i):
        return None if self.labels is None else self.labels[self.offsets[i]:self.offsets[i + 1]]

    def impression_index(self):
        """Impression number of every candidate (flat, len == n_candidates)."""
        return np.repeat(np.arange(self.n_impressions), self.sizes)


def news_key(n):
    """Int id back to the MIND string id (12345 -> 'N12345')."""
    return f"N{n}"


def _parse_ids(strings, with_labels):
    """'N1-0 N2-1' (or 'N1 N2') per row -> flat int ids, flat labels, offsets."""
    joined = " ".join(strings)
    if with_labels:
        pairs = np.array(re.findall(r"N(\d+)-(\d)", joined), dtype=np.int64).reshape(-1, 2)
        ids, labels = pairs[:, 0], pairs[:, 1].astype(np.int8)
    else:
        ids, labels = np.array(re.findall(r"N(\d+)", joined), dtype=np.int64), None
    counts = np.array([s.count("N") for s in strings], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(counts)])
    assert offsets[-1] == len(ids), "could not parse every news id"
    return ids.astype(np.int32), labels, offsets


def load_impressions(split, use_cache=True):
    """Read data/Mind_Large_Data/MINDlarge_<split>/.../behaviors.tsv (cached as .npz after the first read)."""
    assert split in SPLITS, split
    cache = CACHE / f"impressions_{split}.npz"
    if use_cache and cache.exists():
        z = np.load(cache)
        return Impressions(split, z["ids"], z["users"], z["times"].astype("datetime64[s]"), z["history"],
                           z["history_offsets"], z["candidates"], z["offsets"],
                           z["labels"] if "labels" in z.files else None)

    path = DATA / f"MINDlarge_{split}" / f"MINDlarge_{split}" / "behaviors.tsv"
    if not path.exists():
        sys.exit(f"{path} not found; run scripts/download_mind.py first.")
    print(f"Reading {path.relative_to(ROOT)} ...")
    parts = {k: [] for k in ("ids", "users", "times", "history", "hist_counts", "candidates", "cand_counts", "labels")}
    reader = pd.read_csv(path, sep="\t", header=None, names=["id", "user", "time", "history", "impressions"],
                         dtype=str, keep_default_na=False, chunksize=200_000)
    with_labels = split != "test"
    for chunk in reader:
        parts["ids"].append(chunk["id"].astype(np.int64).to_numpy())
        parts["users"].append(chunk["user"].str[1:].astype(np.int64).to_numpy())
        parts["times"].append(pd.to_datetime(chunk["time"], format="%m/%d/%Y %I:%M:%S %p").to_numpy())
        h_ids, _, h_off = _parse_ids(chunk["history"].tolist(), with_labels=False)
        c_ids, c_lab, c_off = _parse_ids(chunk["impressions"].tolist(), with_labels=with_labels)
        parts["history"].append(h_ids)
        parts["hist_counts"].append(np.diff(h_off))
        parts["candidates"].append(c_ids)
        parts["cand_counts"].append(np.diff(c_off))
        if with_labels:
            parts["labels"].append(c_lab)

    cat = lambda k: np.concatenate(parts[k])
    offsets = np.concatenate([[0], np.cumsum(cat("cand_counts"))])
    history_offsets = np.concatenate([[0], np.cumsum(cat("hist_counts"))])
    data = Impressions(split, cat("ids"), cat("users"), cat("times").astype("datetime64[s]"), cat("history"),
                       history_offsets, cat("candidates"), offsets, cat("labels") if with_labels else None)

    CACHE.mkdir(parents=True, exist_ok=True)
    arrays = dict(ids=data.ids, users=data.users, times=data.times.astype(np.int64), history=data.history,
                  history_offsets=data.history_offsets, candidates=data.candidates, offsets=data.offsets)
    if data.labels is not None:
        arrays["labels"] = data.labels
    np.savez(cache, **arrays)
    print(f"{split}: {data.n_impressions:,} impressions, {data.n_candidates:,} candidates (cached)")
    return data


def load_news(split):
    """news.tsv of a split as a DataFrame indexed by int news id."""
    path = DATA / f"MINDlarge_{split}" / f"MINDlarge_{split}" / "news.tsv"
    cols = ["news_id", "category", "subcategory", "title", "abstract", "url", "title_entities", "abstract_entities"]
    news = pd.read_csv(path, sep="\t", header=None, names=cols, dtype=str, keep_default_na=False, quoting=3)
    news.index = news["news_id"].str[1:].astype(np.int32)
    return news


# ---------------------------------------------------------------------------------------------------- ranks
def _flat_scores(data, scores):
    if isinstance(scores, (list, tuple)):
        scores = np.concatenate([np.asarray(s, dtype=np.float64).ravel() for s in scores])
    scores = np.asarray(scores, dtype=np.float64).ravel()
    if len(scores) != data.n_candidates:
        raise ValueError(f"{len(scores):,} scores for {data.n_candidates:,} candidates")
    if not np.isfinite(scores).all():
        raise ValueError("scores contain NaN or inf")
    return scores


def _sort_within_impressions(data, scores):
    """Order of the flat candidates: by impression, then score (high first), ties at random (fixed seed)."""
    tiebreak = np.random.default_rng(TIE_SEED).random(data.n_candidates)
    return np.lexsort((tiebreak, -scores, data.impression_index()))


def scores_to_ranks(data, scores):
    """Rank of every candidate within its impression (1 = top), in file order: the submission format."""
    scores = _flat_scores(data, scores)
    order = _sort_within_impressions(data, scores)
    starts = np.repeat(data.offsets[:-1], data.sizes)
    ranks = np.empty(data.n_candidates, dtype=np.int64)
    ranks[order] = np.arange(data.n_candidates) - starts + 1
    return ranks


# ---------------------------------------------------------------------------------------------------- metrics
def per_impression_metrics(data, scores):
    """AUC, MRR, nDCG@5, nDCG@10 for every impression (vectorised; equal to the official script)."""
    if data.labels is None:
        raise ValueError(f"the {data.split} split has no labels; use write_submission for test")
    scores = _flat_scores(data, scores)
    order = _sort_within_impressions(data, scores)
    y = data.labels[order].astype(np.float64)              # labels in ranked order, impressions contiguous
    starts = data.offsets[:-1]
    sizes = data.sizes
    rank = np.arange(data.n_candidates) - np.repeat(starts, sizes) + 1.0

    pos = np.add.reduceat(y, starts)
    neg = sizes - pos
    with np.errstate(invalid="ignore", divide="ignore"):       # an impression without clicks gives NaN
        mrr = np.add.reduceat(y / rank, starts) / pos

    out = {"AUC": None, "MRR": mrr}
    discount = 1.0 / np.log2(rank + 1.0)
    ideal_table = np.concatenate([[0.0], np.cumsum(1.0 / np.log2(np.arange(2, sizes.max() + 2)))])
    for k in (5, 10):
        dcg = np.add.reduceat(y * discount * (rank <= k), starts)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[f"nDCG@{k}"] = dcg / ideal_table[np.minimum(pos, k).astype(int)]

    # AUC = share of (clicked, not clicked) pairs in which the clicked news is ranked higher
    neg_cum = np.cumsum(1.0 - y)
    neg_above = neg_cum - np.repeat(neg_cum[starts] - (1.0 - y[starts]), sizes)   # negatives ranked above or at
    wins = np.add.reduceat(y * (np.repeat(neg, sizes) - neg_above), starts)
    with np.errstate(invalid="ignore", divide="ignore"):
        out["AUC"] = np.where((pos > 0) & (neg > 0), wins / (pos * neg), np.nan)

    frame = pd.DataFrame({"impression_id": data.ids, "user_id": data.users, "n_candidates": sizes,
                          "n_clicked": pos.astype(int), "history_len": np.diff(data.history_offsets)})
    for m in METRICS:
        frame[m] = out[m]
    return frame


def evaluate(data, scores, name=None, notes="", save=True, verbose=True):
    """Mean metrics over impressions (as the leaderboard). With a name and save=True the result is stored in
    output/news/results/<split>/<name>/ and added to output/news/results/<split>_summary.csv."""
    per = per_impression_metrics(data, scores)
    valid = per["AUC"].notna()
    if (~valid).any():
        print(f"warning: {int((~valid).sum())} impressions have only one class and are left out")
    per = per[valid]
    means = {m: float(per[m].mean()) for m in METRICS}
    result = {"name": name, "split": data.split, **means, "n_impressions": int(len(per)),
              "date": datetime.now().isoformat(timespec="seconds"), "notes": notes}
    if verbose:
        print(f"{name or 'model'} on {data.split}: " + "  ".join(f"{m} {means[m]:.4f}" for m in METRICS))
    if save and name:
        folder = RESULTS / data.split / name
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        per.to_csv(folder / "per_impression.csv.gz", index=False)
        summary = RESULTS / f"{data.split}_summary.csv"
        row = pd.DataFrame([result])
        if summary.exists():
            old = pd.read_csv(summary)
            row = pd.concat([old[old["name"] != name], row], ignore_index=True)   # one row per name
        row.to_csv(summary, index=False)
    return result


# ---------------------------------------------------------------------------------------------------- submission
def write_submission(data, scores, name, notes=""):
    """prediction.txt ('ImpressionID [r1,r2,...]', rows in file order) zipped as prediction.zip, for Codabench:
    https://www.codabench.org/competitions/13967/ (phase 'Official Test')."""
    ranks = scores_to_ranks(data, scores)
    folder = SUBMISSIONS / name
    folder.mkdir(parents=True, exist_ok=True)
    lines = [f"{imp} [{','.join(map(str, ranks[a:b]))}]"
             for imp, a, b in zip(data.ids, data.offsets[:-1], data.offsets[1:])]
    text = "\n".join(lines) + "\n"
    zip_path = folder / "prediction.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("prediction.txt", text)            # at the top level of the zip, nothing else
    (folder / "info.json").write_text(json.dumps({"name": name, "split": data.split, "notes": notes,
                                                  "n_impressions": int(data.n_impressions),
                                                  "date": datetime.now().isoformat(timespec="seconds")}, indent=2),
                                      encoding="utf-8")
    check_submission(zip_path, data)
    print(f"Submission written: {zip_path.relative_to(ROOT)} ({zip_path.stat().st_size / 1e6:.1f} MB)")
    return zip_path


def _read_prediction(path):
    path = Path(path)
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as z:
            assert z.namelist() == ["prediction.txt"], f"zip must hold only prediction.txt, has {z.namelist()}"
            return z.read("prediction.txt").decode("utf-8").splitlines()
    return path.read_text(encoding="utf-8").splitlines()


def check_submission(path, data):
    """Format checks: one line per impression in file order, ranks a permutation of 1..n."""
    lines = _read_prediction(path)
    assert len(lines) == data.n_impressions, f"{len(lines)} lines for {data.n_impressions} impressions"
    for i, line in enumerate(lines):
        imp, ranks = line.split()
        ranks = json.loads(ranks)
        assert int(imp) == data.ids[i], f"line {i + 1}: impression {imp}, expected {data.ids[i]}"
        n = data.offsets[i + 1] - data.offsets[i]
        assert sorted(ranks) == list(range(1, n + 1)), f"line {i + 1}: ranks are not 1..{n}"
    return True


def score_prediction_file(path, data):
    """Score a prediction.txt/.zip on a labelled split the way the official script does (from the ranks)."""
    lines = _read_prediction(path)
    ranks = np.concatenate([json.loads(line.split()[1]) for line in lines]).astype(np.float64)
    return evaluate(data, 1.0 / ranks, save=False, verbose=False)


# ---------------------------------------------------------------------------------------------------- checks
def _official_reference(y_true, y_score):
    """The four functions of msnews/MIND evaluate.py, unchanged, for the self-test."""
    from sklearn.metrics import roc_auc_score

    def dcg_score(y_true, y_score, k=10):
        order = np.argsort(y_score)[::-1]
        y_true = np.take(y_true, order[:k])
        gains = 2 ** y_true - 1
        discounts = np.log2(np.arange(len(y_true)) + 2)
        return np.sum(gains / discounts)

    def ndcg_score(y_true, y_score, k=10):
        return dcg_score(y_true, y_score, k) / dcg_score(y_true, y_true, k)

    def mrr_score(y_true, y_score):
        order = np.argsort(y_score)[::-1]
        y_true = np.take(y_true, order)
        rr_score = y_true / (np.arange(len(y_true)) + 1)
        return np.sum(rr_score) / np.sum(y_true)

    return (roc_auc_score(y_true, y_score), mrr_score(y_true, y_score),
            ndcg_score(y_true, y_score, 5), ndcg_score(y_true, y_score, 10))


def selftest(n_check=3000):
    dev = load_impressions("dev")
    print(f"dev: {dev.n_impressions:,} impressions, {dev.n_candidates:,} candidates, "
          f"{int(dev.labels.sum()):,} clicks, {int((np.diff(dev.history_offsets) == 0).sum()):,} without history")

    # 1. vectorised metrics == official functions on the same ranks (random scores with many ties)
    rng = np.random.default_rng(1)
    scores = rng.integers(0, 5, dev.n_candidates).astype(float)        # deliberately full of ties
    per = per_impression_metrics(dev, scores)
    ranks = scores_to_ranks(dev, scores)
    worst = 0.0
    for i in rng.choice(dev.n_impressions, n_check, replace=False):
        a, b = dev.offsets[i], dev.offsets[i + 1]
        ref = _official_reference(dev.labels[a:b].astype("float32"), 1.0 / ranks[a:b])
        mine = per.loc[i, list(METRICS)].to_numpy()
        worst = max(worst, float(np.max(np.abs(np.array(ref) - mine))))
    print(f"check 1, metrics vs official functions on {n_check} impressions: max difference {worst:.2e}")
    assert worst < 1e-6

    # 2. round trip through the submission file, scored like the official script
    tmp = SUBMISSIONS / "_selftest"
    tmp.mkdir(parents=True, exist_ok=True)
    lines = [f"{imp} [{','.join(map(str, ranks[a:b]))}]"
             for imp, a, b in zip(dev.ids, dev.offsets[:-1], dev.offsets[1:])]
    (tmp / "prediction.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    check_submission(tmp / "prediction.txt", dev)
    direct = evaluate(dev, scores, save=False, verbose=False)
    via_file = score_prediction_file(tmp / "prediction.txt", dev)
    diff = max(abs(direct[m] - via_file[m]) for m in METRICS)
    print(f"check 2, scores vs scores read back from prediction.txt: max difference {diff:.2e}")
    assert diff < 1e-12
    (tmp / "prediction.txt").unlink()
    tmp.rmdir()

    # 3. reference points
    evaluate(dev, rng.random(dev.n_candidates), name="random", notes="uniform random scores (seed 1)")
    position = -np.arange(dev.n_candidates, dtype=float)             # keep the candidates in file order
    evaluate(dev, position, name="file_order", notes="candidates in the order of behaviors.tsv")
    print("selftest passed")


def show_summary(split="dev"):
    path = RESULTS / f"{split}_summary.csv"
    if not path.exists():
        print("no results yet")
        return
    table = pd.read_csv(path).sort_values("AUC", ascending=False)
    print(table[["name", *METRICS, "n_impressions", "date"]].to_string(index=False, float_format="%.4f"))


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "selftest"
    if mode == "selftest":
        selftest()
    elif mode == "summary":
        show_summary(sys.argv[2] if len(sys.argv) > 2 else "dev")
    else:
        sys.exit(__doc__)
