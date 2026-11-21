import numpy as np
from tqdm import tqdm

import mind_eval as me

SPLIT_ORDER = ["train", "dev", "test"]
BASE_DATE = np.datetime64("2019-11-09T00:00:00")


def build_stream(splits):
    """One row per (impression, candidate) over train, dev and test, in that order.

    splits: {"train": Impressions, "dev": Impressions, "test": Impressions}
    Returns a dict:
        candidate  int32  news id (N12345 -> 12345)
        time       int32  seconds since 2019-11-09 00:00:00
        label      int8   1 clicked, 0 not clicked, -1 unknown (test)
        bounds     dict   split -> (start, end) rows in the stream
    The stream is never reordered: later functions return their results in this row order.
    """
    candidate_parts, time_parts, label_parts = [], [], []
    bounds = {}
    counter = 0

    for split in SPLIT_ORDER:
        data = splits[split]

        candidate_parts.append(data.candidates)

        # Convert the impression times first (few rows), then give every candidate its impression's time.
        seconds = (data.times - BASE_DATE).astype(np.int32)
        time_parts.append(np.repeat(seconds, data.sizes))

        if data.labels is None:                       # test: labels are hidden
            label_parts.append(np.full(len(data.candidates), -1, dtype=np.int8))
        else:
            label_parts.append(data.labels)

        start = counter
        counter += len(data.candidates)
        bounds[split] = (start, counter)

    stream = {
        "candidate": np.concatenate(candidate_parts),
        "time": np.concatenate(time_parts),
        "label": np.concatenate(label_parts),
        "bounds": bounds,
    }
    assert stream["time"].min() >= 0
    assert len(stream["candidate"]) == len(stream["time"]) == len(stream["label"]) == counter
    return stream



'''
count_showings_before: what it’s for
For every row in the stream (one candidate in one impression), 
it answers: how many times was this same news article shown in the window before this moment? 
Typical windows are the last 1 h, 6 h and 24 h, plus “all time so far”
'''
SHIFT = 2**21
# bigger than any time (max 1,209,598)

CHUNK = 10_000_000
def count_showings_before(stream, windows):
    """Count how many times each candidate was shown in the given window before each row.

    stream: dict from build_stream()
    windows: dict name -> seconds to look back, None = all time
             (e.g. {"shown_1h": 3600, "shown_all": None})
    Returns a dict name -> int32 array of the same length as stream["candidate"], in stream order.
    """

    key = stream["candidate"].astype(np.int64)
    t = stream["time"]                        # int32, no int64 copy needed
    n = len(t)

    key  *= SHIFT

    key += t
    order = np.argsort(key, kind="stable")    # order[j] = stream row that comes j-th in sorted order
    sorted_keys = key[order]
    del key
    sorted_t = t[order]


    out = {}
    for name in windows :
        out[name] = np.empty(n, dtype=np.int32)

    # Walk through the rows in sorted order: each search lands next to the previous one,
    # which is about 50x faster than searching in stream order.
    for a in tqdm(range(0 , n , CHUNK), desc="Counting showings", unit="chunk"):
        b = min(a + CHUNK, n)
        k = sorted_keys[a:b]
        upper = np.searchsorted(sorted_keys, k , side = "left")

        for name , window in windows.items() :
            if window is None :
                back = sorted_t[a:b]
            else :
                back = np.minimum(sorted_t[a:b], window)

            lower = np.searchsorted(sorted_keys, k - back , side = "left")
            out[name][order[a:b]] = upper - lower     # write back to each row's place in the stream

    return out




'''

clicks_previous_days: what it’s for
For every row (one candidate in one impression), it answers: 

how often was this article clicked when it was shown on the most recent day you have labels for? 
From that you get the click rate: clicks ÷ showings, smoothed. In the probe, this alone reached AUC 0.6157 on dev.


The one rule: labels only from earlier days
Rows	            Labels it may use	              Source day
train, Nov 10–14	    train	                    the day before
train, Nov 9	   none (no earlier day)	         missing (NaN)
dev,              Nov 15	train only	              Nov 14
test,Nov 16–22	             train + dev         	         Nov 15, the last labelled day


Same-day labels are never used. On dev you won’t know Nov 15 clicks until the day is over.
Dev labels only ever feed test rows.
One formula covers every row: source day = min(day − 1, 6), where day 0 is Nov 9 and day 6 is Nov 15. For test rows (days 7–13) it gives 6. For day 0 it gives −1, which means missing.
Input and output
Input: the stream. You need candidate, time and label; label is −1 for test, and those rows are ignored when counting.
Output: a dict of arrays in stream order. Suggested features:

'''

N_DAYS = 16                 # table columns: days 0..13 fit
LAST_LABELLED_DAY = 6       # Nov 15 (dev); test days 7..13 have no labels
SMOOTHING = 10              # m: fake showings at the average click rate


def clicks_previous_days(stream, m=SMOOTHING):
    """Showings, clicks and smoothed click rate of each candidate on its source day: the day before the
    impression, or Nov 15 (the last labelled day) for test rows. Never uses labels of the impression's own day.

    stream: dict from build_stream()
    Returns a dict name -> float32 array of the same length as stream["candidate"], in stream order
    (NaN for Nov 9 rows, which have no earlier day):
        prev_shows, prev_clicks, prev_ctr   on the source day
        cum_shows, cum_clicks, cum_ctr      over all labelled days up to and including the source day
    """
    t = stream["time"]
    cand = stream["candidate"]
    label = stream["label"]
    n = len(t)
    day = (t // 86400).astype(np.int8)
    source_day = np.minimum(day - 1, LAST_LABELLED_DAY)   # -1 for Nov 9, 0 for Nov 10, ..., 6 for Nov 15–22

    # Small tables (article x day) from labelled rows only, indexed by the day the showing HAPPENED.
    # A row only ever reads the column of its source day, which is always before its own day.
    lab = label >= 0
    cell = cand[lab].astype(np.int64) * N_DAYS + day[lab]
    size = (int(cand.max()) + 1) * N_DAYS
    shows = np.bincount(cell, minlength=size).reshape(-1, N_DAYS).astype(np.float32)
    clicks = np.bincount(cell, weights=label[lab], minlength=size).reshape(-1, N_DAYS).astype(np.float32)
    del cell, lab
    cum_shows = np.cumsum(shows, axis=1)
    cum_clicks = np.cumsum(clicks, axis=1)

    # Overall click rate per day: the prior the smoothing pulls towards.
    p_day = (clicks.sum(axis=0) / np.maximum(shows.sum(axis=0), 1)).astype(np.float32)
    p_cum = (cum_clicks.sum(axis=0) / np.maximum(cum_shows.sum(axis=0), 1)).astype(np.float32)

    # Look up every row's (article, source day).
    ok = source_day >= 0
    c, s = cand[ok], source_day[ok]
    out = {}
    for name, table in [("prev_shows", shows), ("prev_clicks", clicks),
                        ("cum_shows", cum_shows), ("cum_clicks", cum_clicks)]:
        out[name] = np.full(n, np.nan, dtype=np.float32)
        out[name][ok] = table[c, s]
    del c, s

    # Rows with source day -1 read a wrong prior here, but their counts are NaN, so the rate stays NaN.
    out["prev_ctr"] = (out["prev_clicks"] + m * p_day[source_day]) / (out["prev_shows"] + m)
    out["cum_ctr"] = (out["cum_clicks"] + m * p_cum[source_day]) / (out["cum_shows"] + m)
    return out

def load_feature_matrix (split , names , impression_mask=None) :
    # the purpose of this function is build a table that has one row per candidate , one column per feature
    # step 1 Expand the impression mask to candidate level with np.repeat(mask, sizes),
    # the same trick as the times in build_stream.

    sizes = me.load_impressions(split).sizes
    if impression_mask is None :
        candidate_mask = np.ones(sizes.sum(), dtype=bool) # all candidates are included

    else :
        assert len(sizes) == len(impression_mask) # the mask is at impression level
        candidate_mask = np.repeat(impression_mask, sizes)

    table = np.empty((candidate_mask.sum(), len(names)), dtype=np.float32)



    for i , name in enumerate(names) :
        feature = np.load(me.ROOT / "output" / "news" / "features" / f"{name}_{split}.npy")
        table[:, i] = feature[candidate_mask]

    return table





















if __name__ == "__main__":
    splits = {s: me.load_impressions(s) for s in SPLIT_ORDER}
    stream = build_stream(splits)
    del splits                                # free the history arrays (several GB)

    windows = {"shown_1h": 3600, "shown_6h": 21600, "shown_24h": 86400, "shown_all": None}
    out = count_showings_before(stream, windows)

    feature_dir = me.ROOT / "output" / "news" / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)

    for  name , array in out.items() :
        for split in SPLIT_ORDER:
            start , end = stream["bounds"][split]
            np.save(feature_dir / f"{name}_{split}.npy", array[start:end])
    del out                                   # free the showings arrays before the next features

    out = clicks_previous_days(stream)
    for  name , array in out.items() :
        for split in SPLIT_ORDER:
            start , end = stream["bounds"][split]
            np.save(feature_dir / f"{name}_{split}.npy", array[start:end])
    print("Saved to:", feature_dir)