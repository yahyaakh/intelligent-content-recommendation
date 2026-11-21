# Pre-registration: news final test (MIND-large dev)

**Status: SKELETON, not frozen.** Fill in the open parts while the experiments run on validation. Before the final test, commit this file; that commit is the freeze. After the freeze, nothing in this file changes, and every method listed is run **once** on the test set.

Protocol details: `thesis/news_documentation/06_protocol.md`. Code: `news/news_protocol.py`.

---

## 1. Data

| Role | Data | Impressions |
|---|---|---|
| Training | train impressions of Nov 9–13 | 1,801,231 |
| Validation (all choices) | train impressions of Nov 14 (`val`) | 431,517 |
| **Final test** | MIND dev, Nov 15 (`dev`) | 376,471 |
| Final test, LLM methods | fixed sample `dev_llm`: 2,500 per history group | 10,000 |

- **Final models:** settings are chosen on validation. *Open: (a) refit on Nov 9–14 with those settings (recommended, since news goes out of date fast), or (b) keep the Nov 9–13 models.*
- **Features:** clicks from earlier days only (doc 02 §4). Showings up to the impression time.
- **User profile:** the MIND history field. *Open: whether the user's own clicks in earlier impressions are added (doc 06 §4).*

**Earlier use of dev.** 12 development runs were scored on dev before this protocol (doc 04, results_log.md). Those numbers are not final results. Every method below is run again under this protocol.

## 2. Metrics and groups

- **Primary metric: AUC.** It is the main metric of the MIND leaderboard. Secondary: MRR, nDCG@5, nDCG@10. nDCG@10 is the one to compare with the movie chapter.
- **Report:** every result for all impressions and for history groups 0 / 1–3 / 4–10 / >10.
- **LLM sample:** per-group means, and an estimate for the whole dev set weighted by the dev group shares (3.0% / 7.2% / 22.6% / 67.2%).

## 3. Methods (fill in from validation)

| # | Method | Settings fixed on validation | Source doc |
|---|---|---|---|
| M1 | Popularity: `cum_ctr` | none | 03 |
| M2 | … | … | … |

## 4. Hypotheses and tests (fill in)

- **Test:** paired two-sided Wilcoxon signed-rank test per impression, with a Bonferroni correction over the comparisons listed here. 95% interval of the mean difference.
- **Equivalence margin:** *open (e.g. ±0.005 AUC).*

| # | Comparison | Group(s) | Expected direction |
|---|---|---|---|
| H1 | … | … | … |

## 5. What would change the conclusions

To be written before the freeze: for each hypothesis, which result confirms it and which rejects it.
