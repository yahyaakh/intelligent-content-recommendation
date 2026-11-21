'''

This script is used to analyze the coverage and age of articles in the dataset. It aims to answer two key questions:
A. How many of the articles shown to users already have click history that a collaborative model could use?
B. How much does article age matter?

this evaluation is important to find out if the collaborative model is actually useful, or if it is just a proxy for article age.

'''

import gc

import numpy as np
import pandas as pd
import mind_eval as me
import news_protocol as npr
import popularity_features as pf

# Part A: Do the candidates have any click history?

LEVELS = ("in_history", "clicked_earlier", "shown_never_clicked", "never_seen")
CLICK_BUCKETS = ([0, 1, 5, 20, np.inf], ["0", "1-4", "5-19", "20+"])     # earlier clicks of the candidate

# Part B: bucket edges in hours, plus a label for "older than the log"
AGE_BUCKETS = ([0, 1, 6, 24, 48, np.inf], ["0-1h", "1-6h", "6-24h", "1-2d", ">2d"])
OLDER = "older_than_log"
MAX_ID = 1 << 21                       # bigger than any MIND news id; arrays are trimmed after the first pass
FEATURES = me.ROOT / "output" / "news" / "features"
OUT = me.ROOT / "output" / "news" / "analysis"

def history_lookup(datasets , n_news):
    """Boolean array indexed by article id: True if the article is in any history of these splits."""
    history = np.zeros(n_news, dtype=bool)
    for data in datasets:
        history[data.history] = True
    return history


def coverage_levels(data, in_history, cum_clicks, cum_shows):
    """One level label (0..3) per candidate slot, checked in the order of LEVELS."""
    levels = np.full(len(data.candidates), len(LEVELS) - 1, dtype=np.int8)  # default: never_seen
    levels[in_history[data.candidates]] = 0  # in_history
    levels[(levels == len(LEVELS) - 1) & (cum_clicks > 0)] = 1  # clicked_earlier
    levels[(levels == len(LEVELS) - 1) & (cum_shows > 0)] = 2  # shown_never_clicked
    return levels


def coverage_table(levels, labels, groups):
    """Share of all slots and of clicked slots per level, overall and per history group."""
    df = pd.DataFrame({
        "level": pd.Categorical.from_codes(levels, categories=LEVELS),
        "group": pd.Categorical(groups, categories=npr.GROUPS),
        "clicked": labels == 1,
    })

    def shares(part):
        per_group = pd.crosstab(part["level"], part["group"], normalize="columns")
        per_group["all"] = part["level"].value_counts(normalize=True)
        return per_group.reindex(index=list(LEVELS), columns=["all", *npr.GROUPS]).fillna(0)

    return pd.concat({"all slots": shares(df), "clicked slots": shares(df[df["clicked"]])}, axis=1)

def click_count_buckets(cum_clicks, labels):
    """Share of all slots and of clicked slots with 0 / 1-4 / 5-19 / 20+ earlier clicks."""
    edges, names = CLICK_BUCKETS
    bucket = pd.Series(pd.cut(cum_clicks, edges, right=False, labels=names))
    return pd.DataFrame({
        "all slots": bucket.value_counts(normalize=True),
        "clicked slots": bucket[labels == 1].value_counts(normalize=True),
    }).reindex(names)


# Part B: article age

def first_showing(splits):
    """First pass over the splits (one at a time, to save memory).

    Returns
        first     seconds since Nov 9 00:00 of each article's first showing in any impression (int64, by id)
        older     True if the article is in a train history: clicked before the log, so its real age is unknown
        n_news    largest id seen + 1
    """
    first = np.full(MAX_ID, np.iinfo(np.int64).max, dtype=np.int64)
    older = np.zeros(MAX_ID, dtype=bool)
    n_news = 0
    for split in splits:
        data = me.load_impressions(split)
        seconds = (data.times - pf.BASE_DATE).astype(np.int64)
        np.minimum.at(first, data.candidates, np.repeat(seconds, data.sizes))
        if split == "train":
            # train histories never change during Nov 9-14 (doc 06 §4), so they hold only pre-log clicks
            older[data.history] = True
        n_news = max(n_news, int(data.candidates.max()) + 1, int(data.history.max()) + 1)
        del data, seconds
        gc.collect()
    return first[:n_news], older[:n_news], n_news


def article_age(data, first, older):
    """Age in hours of every candidate slot of one split (impression time - first showing), and a flag
    for articles that are older than the log. The first showing is never after the slot itself, because
    the slot is a showing too, so the age needs no labels and leaks nothing."""
    seconds = np.repeat((data.times - pf.BASE_DATE).astype(np.int64), data.sizes)
    age = (seconds - first[data.candidates]) / 3600.0
    assert (age >= 0).all()
    return age.astype(np.float32), older[data.candidates]


def age_table(age_hours, labels, older):
    """Share of slots, share of clicked slots and click rate per age bucket."""
    edges, names = AGE_BUCKETS
    bucket = pd.cut(age_hours, edges, right=False, labels=names).astype(str)
    bucket = np.where(older, OLDER, bucket)
    df = pd.DataFrame({"bucket": bucket, "clicked": labels == 1})
    table = df.groupby("bucket")["clicked"].agg(slots="size", click_rate="mean")
    table["share of slots"] = table["slots"] / table["slots"].sum()
    table["share of clicks"] = df[df["clicked"]]["bucket"].value_counts(normalize=True)
    return table.reindex([*names, OLDER])[["slots", "share of slots", "share of clicks", "click_rate"]]


def save_feature(split, name, values):
    """One value per candidate slot in mind_eval file order, the same layout as popularity_features.py."""
    FEATURES.mkdir(parents=True, exist_ok=True)
    np.save(FEATURES / f"{name}_{split}.npy", values)


def load_features(split, names, slot_mask=None):
    """Like pf.load_feature_matrix, but takes a slot mask so the whole split is not loaded again (memory)."""
    columns = []
    for name in names:
        values = np.load(FEATURES / f"{name}_{split}.npy")
        columns.append((values if slot_mask is None else values[slot_mask]).astype(np.float32))
        del values
    return np.column_stack(columns)


def slot_groups(data):
    return np.repeat(npr.history_group(np.diff(data.history_offsets)), data.sizes)


def show(title, table, pct=True):
    print(f"\n{title}")
    print(table.to_string(float_format=(lambda x: f"{x:.1%}") if pct else (lambda x: f"{x:.4f}")))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200)

    # ---- Part B, first pass: first showing of every article (all splits), and the age feature per split
    first, older, n_news = first_showing(("train", "dev", "test"))
    print(f"articles with an id: {n_news:,}; shown at least once: {(first < np.iinfo(np.int64).max).sum():,}; "
          f"in some history (older than the log): {older.sum():,}")
    for split in ("train", "dev", "test"):
        data = me.load_impressions(split)
        age, is_older = article_age(data, first, older)
        save_feature(split, "age_hours", age)
        save_feature(split, "older_than_log", is_older)
        del data, age, is_older
        gc.collect()

    # ---- Part A and B tables for val and dev
    train = me.load_impressions("train")
    in_history_val = history_lookup([train], n_news)
    mask = npr.val_mask(train)
    val = npr.subset(train, mask, "val")
    train_sizes = train.sizes
    del train
    gc.collect()
    dev = me.load_impressions("dev")
    in_history_dev = in_history_val | history_lookup([dev], n_news)
    print(f"distinct articles in histories: train {in_history_val.sum():,}, train + dev {in_history_dev.sum():,}")

    for name, data, split, imp_mask, in_history in (("val", val, "train", mask, in_history_val),
                                                   ("dev", dev, "dev", None, in_history_dev)):
        feats = load_features(split, ["cum_clicks", "cum_shows", "cum_ctr", "age_hours", "older_than_log"],
                              None if imp_mask is None else np.repeat(imp_mask, train_sizes))
        assert len(feats) == data.n_candidates
        cum_clicks, cum_shows = feats[:, 0], feats[:, 1]

        levels = coverage_levels(data, in_history, cum_clicks, cum_shows)
        assert not ((levels == 3) & (cum_shows > 0)).any()
        assert not ((levels == 2) & (cum_clicks > 0)).any()
        groups = slot_groups(data)

        cov = coverage_table(levels, data.labels, groups)
        show(f"[{name}] A. coverage levels (share of slots; each column sums to 100%)", cov)
        cov.to_csv(OUT / f"coverage_{name}.csv")

        clicks = click_count_buckets(cum_clicks, data.labels)
        show(f"[{name}] A2. earlier clicks of the candidate", clicks)
        clicks.to_csv(OUT / f"earlier_clicks_{name}.csv")

        ages = age_table(feats[:, 3], data.labels, feats[:, 4].astype(bool))
        show(f"[{name}] B. article age", ages.drop(columns="slots"))
        print(f"   median age of shown slots {np.median(feats[:, 3]):.1f} h, of clicked slots "
              f"{np.median(feats[:, 3][data.labels == 1]):.1f} h (all slots, including older-than-log articles)")
        ages.to_csv(OUT / f"age_{name}.csv")

        # single-feature checks on validation only (dev is the final test)
        if name == "val":
            me.evaluate(data, -feats[:, 3], name="single_age",
                        notes="single feature: -age_hours (newest first), age = hours since first showing")
            me.evaluate(data, feats[:, 2], name="single_cum_ctr",
                        notes="single feature: cum_ctr (smoothed click rate on earlier days, m=10)")
        del feats, levels, groups
        gc.collect()

    print()
    npr.groups_table("val", "AUC")


if __name__ == "__main__":
    main()
