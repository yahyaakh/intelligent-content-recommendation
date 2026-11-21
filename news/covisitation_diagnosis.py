'''
Why does item co-visitation score like random on MIND (dev AUC 0.504, doc 04)? (doc 08)

Doc 07 showed that two thirds of the candidates were clicked before, so the data exists. This script checks
the model itself on a random sample of validation impressions (Nov 14), with the click data of Nov 9-13 only:

    1. Is a candidate linked to the user's history at all (any shared reader)?
    2. Is it among the top-50 neighbours of a history article (what co_visitation.py keeps)?
    3. AUC of the score with top-50 neighbours (the old model) and with no limit (every shared reader counts)
    4. Where the candidates rank in a history article's neighbour list

Similarity is the same as in co_visitation.py: shared readers / sqrt(readers of a * readers of b).
Readers = users with the article in their history or clicked in an impression of Nov 9-13.

    python news/covisitation_diagnosis.py [n_impressions]       (default 10,000)
'''
import gc
import sys
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp

import mind_eval as me
import news_protocol as npr

TOP_K = 50
SEED = 42
OUT = me.ROOT / "output" / "news" / "analysis"


def reader_matrix(train, fit):
    """Binary article x user matrix (CSR) from train histories and from clicks in impressions where fit is True."""
    hist_users = np.repeat(train.users, np.diff(train.history_offsets))
    slot_fit = np.repeat(fit, train.sizes)
    clicked = (train.labels == 1) & slot_fit
    cand_users = np.repeat(train.users, train.sizes)[clicked]
    users = np.concatenate([hist_users, cand_users])
    items = np.concatenate([train.history, train.candidates[clicked]]).astype(np.int64)
    _, user_idx = np.unique(users, return_inverse=True)
    n_news = int(max(train.history.max(), train.candidates.max())) + 1
    m = sp.csr_matrix((np.ones(len(items), dtype=np.float32), (items, user_idx)),
                      shape=(n_news, user_idx.max() + 1))
    m.data[:] = 1.0                                   # duplicates were summed: back to binary
    return m


def kth_neighbour_weight(m, readers, articles, k=TOP_K, block=256):
    """Weight of the k-th strongest neighbour of each article (0 if it has fewer than k neighbours),
    and its number of neighbours, from the full row of the similarity matrix."""
    mt = m.T.tocsr()
    kth = np.zeros(len(articles), dtype=np.float32)
    n_nb = np.zeros(len(articles), dtype=np.int64)
    start = time.time()
    for a in range(0, len(articles), block):
        rows = articles[a:a + block]
        counts = (m[rows] @ mt).tocsr()             # shared readers with every article
        for r in range(len(rows)):
            lo, hi = counts.indptr[r], counts.indptr[r + 1]
            idx, c = counts.indices[lo:hi], counts.data[lo:hi]
            keep = idx != rows[r]
            w = c[keep] / np.sqrt(readers[rows[r]] * readers[idx[keep]])
            n_nb[a + r] = len(w)
            if len(w) >= k:
                kth[a + r] = np.partition(w, len(w) - k)[len(w) - k]
        if a // block % 20 == 0:
            print(f"   neighbour lists {a + len(rows):,}/{len(articles):,}  ({time.time() - start:.0f} s)", flush=True)
    return kth, n_nb


def main(n_sample=10_000):
    OUT.mkdir(parents=True, exist_ok=True)
    train = me.load_impressions("train")
    val_m = npr.val_mask(train)
    m = reader_matrix(train, ~val_m)
    readers = np.asarray(m.sum(axis=1)).ravel()
    print(f"reader matrix: {m.shape[0]:,} article ids x {m.shape[1]:,} users, {m.nnz:,} (user, article) pairs")

    rng = np.random.default_rng(SEED)
    keep = np.zeros(train.n_impressions, dtype=bool)
    keep[rng.choice(np.flatnonzero(val_m), n_sample, replace=False)] = True
    sample = npr.subset(train, keep, "val_covis_sample")
    del train
    gc.collect()

    hist_articles = np.unique(sample.history)
    hist_articles = hist_articles[readers[hist_articles] > 0]
    print(f"sample: {sample.n_impressions:,} impressions, {sample.n_candidates:,} slots, "
          f"{len(hist_articles):,} distinct history articles with readers")
    kth, n_nb = kth_neighbour_weight(m, readers, hist_articles)
    kth_of = dict(zip(hist_articles.tolist(), kth.tolist()))

    full = np.zeros(sample.n_candidates, dtype=np.float64)       # every shared reader counts
    raw = np.zeros(sample.n_candidates, dtype=np.float64)        # shared readers, no normalisation
    cond = np.zeros(sample.n_candidates, dtype=np.float64)       # shared / readers of the history article = P(c | h)
    top = np.zeros(sample.n_candidates, dtype=np.float64)        # only top-50 neighbours (old model)
    linked = np.zeros(sample.n_candidates, dtype=bool)
    in_top = np.zeros(sample.n_candidates, dtype=bool)
    best_rank_pct = np.full(sample.n_candidates, np.nan)          # strongest link: weight / k-th weight
    for i in range(sample.n_impressions):
        h = np.unique(sample.history_of(i))
        h = h[readers[h] > 0]
        c = sample.candidates_of(i)
        a, b = sample.offsets[i], sample.offsets[i + 1]
        if len(h) == 0:
            continue
        shared = (m[h] @ m[c].T).toarray()                       # |h| x |c| shared readers
        with np.errstate(divide="ignore", invalid="ignore"):
            w = shared / np.sqrt(np.outer(readers[h], readers[c]))
        w = np.nan_to_num(w)
        w[h[:, None] == c[None, :]] = 0.0                        # an article is not its own neighbour
        thr = np.array([kth_of[x] for x in h])[:, None]
        is_top = (w > 0) & (w >= thr)
        full[a:b] = w.sum(axis=0)
        shared[h[:, None] == c[None, :]] = 0.0
        raw[a:b] = shared.sum(axis=0)
        cond[a:b] = (shared / readers[h][:, None]).sum(axis=0)
        top[a:b] = (w * is_top).sum(axis=0)
        linked[a:b] = (w > 0).any(axis=0)
        in_top[a:b] = is_top.any(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(thr > 0, w / thr, np.inf)           # >= 1 means inside the top 50
        ratio[w == 0] = np.nan
        best = np.nanmax(np.where(np.isnan(ratio), -1, ratio), axis=0)
        best_rank_pct[a:b] = np.where(best < 0, np.nan, best)

    clicked = sample.labels == 1
    groups = np.repeat(npr.history_group(np.diff(sample.history_offsets)), sample.sizes)
    rows = []
    for name, part in (("all slots", np.ones_like(clicked)), ("clicked slots", clicked)):
        rows.append({"slots": name, "linked to history (any shared reader)": linked[part].mean(),
                     "in a top-50 list": in_top[part].mean()})
    cover = pd.DataFrame(rows).set_index("slots")
    print("\n1-2. Coverage of the candidate slots")
    print(cover.to_string(float_format=lambda x: f"{x:.1%}"))

    print("\n   k-th (50th) neighbour weight of history articles: median "
          f"{np.median(kth):.4f}; history articles with fewer than 50 neighbours: {np.mean(n_nb < TOP_K):.1%}; "
          f"median number of neighbours {np.median(n_nb):,.0f}")
    linked_ratio = best_rank_pct[linked & np.isfinite(best_rank_pct)]
    print(f"   linked slots: strongest link / 50th-neighbour weight, median {np.median(linked_ratio):.3f} "
          f"(>= 1 means inside a top-50 list)")

    results = {}
    for name, scores in (("covis_top50", top), ("covis_full", full), ("shared_raw", raw), ("shared_cond", cond)):
        per = me.per_impression_metrics(sample, scores)
        per["group"] = npr.history_group(per["history_len"])
        results[name] = per.groupby("group")["AUC"].mean().reindex(npr.GROUPS)
        results[name]["all"] = per["AUC"].mean()
        results[name]["nDCG@10 all"] = per["nDCG@10"].mean()
    # the popularity reference on the same sample
    cum_ctr = np.load(me.ROOT / "output" / "news" / "features" / "cum_ctr_train.npy")
    train_sizes = me.load_impressions("train").sizes
    cum_ctr = cum_ctr[np.repeat(keep, train_sizes)]
    per = me.per_impression_metrics(sample, cum_ctr)
    per["group"] = npr.history_group(per["history_len"])
    results["cum_ctr"] = per.groupby("group")["AUC"].mean().reindex(npr.GROUPS)
    results["cum_ctr"]["all"] = per["AUC"].mean()
    results["cum_ctr"]["nDCG@10 all"] = per["nDCG@10"].mean()
    table = pd.DataFrame(results).T
    print("\n3. AUC on the sample (Nov 14), by history group")
    print(table.to_string(float_format=lambda x: f"{x:.4f}"))

    # 4. linked-only AUC: impressions where every candidate is linked vs not
    per_full = me.per_impression_metrics(sample, full)
    share_linked = pd.Series(linked).groupby(np.repeat(np.arange(sample.n_impressions), sample.sizes)).mean()
    per_full["share_linked"] = share_linked.to_numpy()
    bins = pd.cut(per_full["share_linked"], [-0.01, 0.5, 0.8, 0.95, 1.0], labels=["<=50%", "50-80%", "80-95%", ">95%"])
    by_link = per_full.groupby(bins, observed=True)["AUC"].agg(["count", "mean"])
    print("\n4. covis_full AUC by the share of an impression's candidates that are linked to the history")
    print(by_link.to_string(float_format=lambda x: f"{x:.4f}"))

    # 5. which candidates does each score put first? old (pre-log) articles are rarely clicked (doc 07)
    old = np.zeros(m.shape[0], dtype=bool)
    old[np.unique(sample.history)] = True        # proxy: in the sample's histories (all pre-log, doc 06)
    tr = me.load_impressions("train")
    old[np.unique(tr.history)] = True
    del tr
    old_slot = old[sample.candidates]
    top1 = {}
    imp_idx = np.repeat(np.arange(sample.n_impressions), sample.sizes)
    for name, scores in (("covis_full", full), ("shared_raw", raw), ("shared_cond", cond), ("cum_ctr", cum_ctr)):
        order = np.lexsort((-scores, imp_idx))
        first = order[sample.offsets[:-1]]
        top1[name] = {"top-1 is a pre-log article": old_slot[first].mean(),
                      "top-1 clicked": clicked[first].mean()}
    top1["(all slots)"] = {"top-1 is a pre-log article": old_slot.mean(), "top-1 clicked": clicked.mean()}
    top1["(clicked slots)"] = {"top-1 is a pre-log article": old_slot[clicked].mean(), "top-1 clicked": 1.0}
    top1 = pd.DataFrame(top1).T
    print("\n5. The candidate each score ranks first")
    print(top1.to_string(float_format=lambda x: f"{x:.1%}"))
    corr = pd.DataFrame({"covis_full": full, "shared_raw": raw, "shared_cond": cond, "readers": readers[sample.candidates],
                         "cum_ctr": cum_ctr}).corr(method="spearman")
    print("\n   Spearman correlation between scores over slots")
    print(corr.to_string(float_format=lambda x: f"{x:.2f}"))
    top1.to_csv(OUT / "covisitation_top1_val_sample.csv")

    cover.to_csv(OUT / "covisitation_coverage_val_sample.csv")
    table.to_csv(OUT / "covisitation_auc_val_sample.csv")
    by_link.to_csv(OUT / "covisitation_auc_by_link_val_sample.csv")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 10_000)
