# 10 — MiniLM vectors for every dev and test article

## Question

Doc 09 found that the saved MiniLM vectors (`output/minilm_title_abstract_entities_mine/`) cover only the 101,527 articles of the train news file. 5.4% of dev candidate slots had no vector. The final test runs on dev, and the MIND test split has many more new articles. Every article therefore needs a vector, built in exactly the same way as the train vectors.

## Set-up

- **Script:** `news/embed_missing_news.py`.
- **Model:** `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions, unit length). This is the same model `news/SemanticContentModel.py` used for the train vectors.
- **Text:** title + " " + abstract + " Entities: " + the entity labels of title and abstract, deduplicated and joined with "; ". This is the same text as `SemanticContentModel.py` builds.
- **Output:** `output/minilm_title_abstract_entities_all/`. It holds the 101,527 original vectors first, in their original order, then the new ones. The original folder is not changed.
- **Run log:** `output/news/embed_missing_log.txt`. **Coverage check:** `output/news/embedding_coverage.txt`.

### Quote handling (found by the reproduction check)

The first run re-encoded 300 random train articles and compared them with the stored vectors. The mean cosine was 0.99993, but the minimum was 0.989, so the run stopped.

The cause is in the input file. The train vectors were built from `data/Mind_Large_Data/mind_large_dataset_csv/news.csv`, which is a conversion of `news.tsv` read with pandas' default quote parsing. That parsing removes the quotes around a field that starts with `"`. For example, `"Bad" cholesterol means LDL …` becomes `Bad cholesterol means LDL …`. Compared with the raw `news.tsv`, 850 abstracts, 115 titles and 11 entity fields of train articles differ.

The train vectors are kept as they are, because docs 04 and 09 depend on them. The new articles are read the same way, so that all vectors follow one text convention. Reading the train `news.tsv` with default quote parsing reproduces `news.csv` exactly: 0 differences in title, abstract and both entity fields.

In a few rows the parsing breaks: a field swallows a tab or a line break, and in the test file 2 lines are merged. For those rows the raw `news.tsv` text is used instead:

| File | Rows with raw text |
|---|---|
| train | 6 |
| dev | 5 |
| test | 11 |

The earlier dev runs of doc 04 embedded their 2,624 new dev articles on the fly (`SemanticContentModel.py`, "official dev set") with the same default quote parsing. Their results are therefore not affected by the coverage gap.

## Results

| Check | Result |
|---|---|
| Reproduction of 300 random stored train vectors | cosine min 0.9999998, mean 1.000000 |
| Articles added | 28,852 (2,624 first seen in dev, 26,228 first seen in test) |
| Total vectors | 130,379 |
| Vector length | 0.9999998 – 1.0000001 |
| Original 101,527 rows unchanged | yes (identical arrays) |
| Train: candidate slots / history entries without a vector | 0 of 83,507,374 / 0 of 73,629,868 |
| Dev: candidate slots / history entries without a vector | 0 of 14,085,557 / 0 of 12,282,163 |
| Test: candidate slots / history entries without a vector | 0 of 93,115,001 / 0 of 98,620,498 |

## What this means

- Every train, dev and test candidate and history article now has a MiniLM vector, built with the same model and text as the train vectors.
- `news/hybrid_e.py` now reads the new folder. The validation results of doc 09 are unchanged, because train and validation articles keep exactly the same vectors.
- The final test of Hybrid E on dev can now score every candidate. Without the new vectors, 5.4% of dev slots would have had a content score of 0.
