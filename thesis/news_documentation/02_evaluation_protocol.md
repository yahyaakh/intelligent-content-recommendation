# 02 — Evaluation: `news/mind_eval.py` and the protocol

**Status:** done. The module is used by every news model since 2026-11-06.

---

## 1. One module for every model

`news/mind_eval.py` loads the data, computes the metrics, saves the results and writes leaderboard files. A model only has to return **one score per candidate**, in file order:

```python
import mind_eval as me
dev = me.load_impressions("dev")          # ids, users, times, histories, candidates, labels as flat arrays
scores = my_model.score(dev)              # one score per candidate (flat array or list of arrays)
me.evaluate(dev, scores, name="my_model", notes="...")
me.write_submission(me.load_impressions("test"), test_scores, name="my_model")
```

- `load_impressions(split)` parses `behaviors.tsv` once and caches it in `output/news/cache/impressions_<split>.npz`. Article and user ids are stored as integers (`N12345` → 12345).
- `evaluate()` prints the four metrics and saves:
  - `output/news/results/<split>/<name>/metrics.json`
  - `output/news/results/<split>/<name>/per_impression.csv.gz`: one row of metrics per impression, used for paired tests and for results by history group;
  - one row in `output/news/results/<split>_summary.csv`, with the date and a note.
- `write_submission()` writes `output/news/submissions/<name>/prediction.zip` in the leaderboard format.
- Command line: `python news/mind_eval.py selftest` and `python news/mind_eval.py summary`.

---

## 2. Metrics

These are the official MIND leaderboard metrics, computed per impression and then averaged over impressions:

| Metric | Meaning |
|---|---|
| **AUC** | the probability that a random clicked article is ranked above a random unclicked one in the same impression; 0.5 = random |
| **MRR** | the mean of 1 / rank over the clicked articles |
| **nDCG@5, nDCG@10** | the same discounted gain as in the movie study (1 / log2(position + 1)), within the impression's own list |

**As on the leaderboard,** scores are first turned into ranks (1 = top), and ties are broken at random with a fixed seed (`TIE_SEED = 0`). A model that gives many equal scores therefore gets no hidden advantage from file order. `selftest` checks the metrics against the official `evaluate.py` code and runs a random baseline on dev.

**Two baselines check that the module is fair:**
- Random scores give AUC 0.500.
- Candidates kept in file order also give AUC 0.500. The order in `behaviors.tsv` therefore reveals nothing about the clicks.

---

## 3. Splits used so far

| Use | Data |
|---|---|
| Training | train impressions before **2019-11-14 00:00** (Nov 9–13) |
| Internal validation | train impressions of **Nov 14** (431,517 impressions) |
| Reported result | the full **dev** set (Nov 15, 376,471 impressions) |

The internal validation numbers in older files (`output/news/*_internal_val.json`) were computed with scikit-learn's nDCG, not with `mind_eval`. They are not comparable with the dev numbers and are kept only for reference.

**Planned change (doc 05):** all choices are made on Nov 14 only. Dev becomes a **pre-registered final test**, run once per method, as in movie doc 36. The runs so far were all evaluated on dev, so they count as development results, not as final-test results.

---

## 4. The rule against label leakage

A feature for an impression may only use clicks from **earlier days**:
- train rows of Nov 10–14 use clicks up to the day before;
- dev rows (Nov 15) use train clicks only;
- test rows (Nov 16–22) use train and dev clicks;
- same-day clicks are never used, because a real system would not know them yet.

Counting *showings* (how often an article was shown before this moment) is allowed up to the impression's own time, because a live system knows what it has shown. Doc 03 has the details.
