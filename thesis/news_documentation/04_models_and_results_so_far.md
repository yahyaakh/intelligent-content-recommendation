# 04 — Models built so far and their results

**Status:** development results (runs of 2026-11-06 to 2026-11-12). All dev numbers come from `output/news/results/dev_summary.csv` (376,471 impressions, `mind_eval`). These runs were evaluated on dev directly, so they are **not** pre-registered final-test results (doc 02 §3).

---

## 1. Results on dev

| Model | AUC | MRR | nDCG@5 | nDCG@10 |
|---|---|---|---|---|
| Random scores | 0.500 | 0.219 | 0.224 | 0.287 |
| File order | 0.500 | 0.219 | 0.223 | 0.286 |
| Co-visitation, top 50 neighbours | 0.504 | 0.224 | 0.230 | 0.293 |
| Single feature `cum_ctr` (doc 03) | 0.629 | 0.287 | 0.314 | 0.373 |
| MLP on 3 category features | 0.617 | 0.297 | 0.324 | 0.384 |
| LightGBM, 5 MiniLM/category features | 0.649 | 0.318 | 0.349 | 0.408 |
| **LightGBM, 5 MiniLM + 10 popularity features** | **0.682** | **0.335** | **0.372** | **0.431** |

Run names in the results folder are `random`, `file_order`, `covisitation_top50`, `single_cum_ctr`, `news_mlp_3feat`, `lgbm_minilm_5feat` and `lgbm_minilm_pop15`.

---

## 2. The models

### 2.1 Co-visitation (item-to-item, collaborative): `news/co_visitation.py`

- **Data:** every (user, article) pair clicked in train, from histories and from clicked impression candidates.
- **Model:** the user × article click matrix. For each article, the 50 articles with the highest cosine-normalised count of shared readers (shared readers / √(readers A × readers B)).
- **Score of a candidate:** the sum of its neighbour weights to the user's history articles.
- **Result:** AUC 0.504, almost random.
- **Reason (doc 08):** only 2.6% of candidates appear in any history article's top-50 list, so almost every candidate scores 0. Without the limit, the score mostly measures how many people read the candidate (Spearman 0.86 with reader count), which does not predict clicks: AUC 0.509 on validation.
- **Lesson:** item-to-item co-occurrence fails on news, because it measures read volume rather than click rate. Dropped as a baseline (doc 08).

### 2.2 MLP on category features: `news/News_articles_Prediction.py`

- **Features per candidate (3):**
  - history size;
  - the share of the user's history in the candidate's **category**;
  - the share of the user's history in the candidate's **subcategory**.
- **Model:** Keras: normalisation → Dense 16 (ReLU) → Dense 8 (ReLU) → 1 (sigmoid); binary cross-entropy; Adam (learning rate 0.001); up to 10 epochs, early stopping with patience 2. Saved as `output/news/news_recommender.keras`.
- **Training:** train impressions before Nov 14.
- **Result:** AUC 0.617, nDCG@10 0.384. Simple "which topics does this user read" features already beat item co-visitation by far.

### 2.3 LightGBM with MiniLM content features: `news/SemanticContentModel.py`

- **Article text:** title + abstract + "Entities: " + the entity labels from `title_entities` and `abstract_entities`.
- **Embedding:** `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions), CPU. Saved in `output/minilm_title_abstract_entities_mine/`. The 2,624 dev articles missing from train's `news.tsv` are embedded the same way before scoring dev.
- **Five features per candidate:**

| Feature | Definition |
|---|---|
| `top5_similarity` | mean cosine similarity between the candidate and its 5 most similar history articles (fewer if the history is shorter) |
| `max_similarity` | the highest similarity to any one history article |
| `avg_profile_similarity` | cosine similarity to the mean embedding of the history |
| `category_preference` | the share of the history in the candidate's category |
| `history_length` | the number of distinct history articles |

- **Ranker:** `LGBMRanker`, lambdarank objective, 100 trees, learning rate 0.05, 15 leaves; one group per impression.
- **Training:** train impressions before Nov 14. The feature matrix is cached on disk.
- **Result (`lgbm_minilm_5feat`):** AUC 0.649, nDCG@10 0.408.

### 2.4 LightGBM with MiniLM and popularity features: best so far

- **The same ranker and settings** as 2.3, with the 10 popularity features of doc 03 added. That gives 15 features: the 5 MiniLM features plus `shown_1h`, `shown_6h`, `shown_24h`, `shown_all`, `prev_shows`, `prev_clicks`, `prev_ctr`, `cum_shows`, `cum_clicks` and `cum_ctr`.
- **Training:** train impressions before Nov 14. Saved as `output/news/lgbm_minilm_pop15.txt`.
- **Result (`lgbm_minilm_pop15`):** **AUC 0.682, MRR 0.335, nDCG@5 0.372, nDCG@10 0.431**.
- **Feature importance** is printed by the script but was not saved. Record it the next time the model is trained.

---

## 3. What these results say so far

1. **Content and popularity work together.**
   - Content alone (MiniLM, 5 features): AUC 0.649.
   - Popularity alone (`cum_ctr`): 0.629.
   - Both in one ranker: 0.682.

   This is the news version of the movie finding that Hybrid E (personal score plus popularity) beats each part alone.
2. **Co-visitation fails on news** (AUC 0.504), unlike item-CF on movies; the cause is in doc 08. Never-seen articles (7–9% of clicks) can only be scored from text, which is where the LLM labels and content-based semantic IDs of doc 05 should help.
3. **The learned ranker is again the strongest model** among those built so far, as it was in the movie study.

---

## 4. Not yet done (see doc 05)

- ~~No result split by history group; no paired tests.~~ Done in doc 06 §5.
- No standard neural news model (NRMS) as a reference.
- No validation-only model selection: the runs above were compared on dev.
- No LLM experiment on news yet.

---

## 5. Older work (superseded)

- **Before MIND-large,** a small pipeline used 5,000 articles and 1,626 users from MIND-small:
  - `news/prepare_mind_data.py`;
  - `news/cold_start_handler.py`: three stages (cold, warm, hot) with TF-IDF and collaborative filtering;
  - `news/cold_start_demo.py`.
- **Exploration scripts:** `news/script_analyse_MIND_Dataset.py` and `news/sementic_test_script.py`.

None of these results are comparable with the MIND-large numbers above. They are kept only for the history of the project.
