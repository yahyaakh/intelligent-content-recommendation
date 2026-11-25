'''
Embed the dev and test articles that have no MiniLM vector.

The vectors in output/minilm_title_abstract_entities_mine (news/SemanticContentModel.py) were built from the
train news file only, so 5.4% of dev candidate slots had no vector (doc 09). This script encodes every
article of the train, dev and test news files that is missing, with the same model and the same text:

    title + " " + abstract [+ " Entities: " + entity labels of title and abstract, deduplicated, "; "-joined]

The original folder is not changed. The result goes to output/minilm_title_abstract_entities_all/: the
original vectors first (same order), then the new ones.

Quote handling. The stored vectors were built from data/.../news.csv, a conversion of news.tsv that was read
with pandas' default quote parsing. That parsing removes the quotes around a field that starts with '"' (about
1% of train articles, e.g. '"Bad" cholesterol ...' -> 'Bad cholesterol ...'). To keep one text convention for
all vectors, the news files are read the same way here. Where that parsing breaks a row (an id goes missing, or
a field swallows a tab or line break; 2 rows of the test file), the raw news.tsv text is used for that article.

Before encoding, a random sample of train articles is re-encoded and compared with the stored vectors, to
check that the text and the model reproduce them.

Command line (repository root):
    python news/embed_missing_news.py
'''
import json

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

import mind_eval as me

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SOURCE = me.ROOT / "output" / "minilm_title_abstract_entities_mine"
TARGET = me.ROOT / "output" / "minilm_title_abstract_entities_all"
CHECK_N = 300
SEED = 42


def entity_names(title_entities, abstract_entities):
    """Same as SemanticContentModel.get_entity_names."""
    names = []
    for entity_json in (title_entities, abstract_entities):
        for entity in json.loads(entity_json or "[]") or []:
            name = (entity.get("Label") or "").strip()
            if name:
                names.append(name)
    return "; ".join(dict.fromkeys(names))


def article_text(row):
    names = entity_names(row["title_entities"], row["abstract_entities"])
    return row["title"] + " " + row["abstract"] + (" Entities: " + names if names else "")


COLUMNS = ["news_id", "category", "subcategory", "title", "abstract", "url", "title_entities", "abstract_entities"]
TEXT_COLUMNS = ["title", "abstract", "title_entities", "abstract_entities"]


def load_news_as_converted(split):
    """news.tsv read with pandas' default quote parsing, as news.csv was made; broken rows replaced by raw text."""
    path = me.DATA / f"MINDlarge_{split}" / f"MINDlarge_{split}" / "news.tsv"
    parsed = pd.read_csv(path, sep="\t", header=None, names=COLUMNS, dtype=str, keep_default_na=False)
    parsed = parsed[parsed["news_id"].str.fullmatch(r"N\d+")]
    parsed.index = parsed["news_id"].str[1:].astype(np.int32)
    raw = me.load_news(split)
    parsed = parsed[~parsed.index.duplicated(keep="first")].reindex(raw.index)
    broken = parsed["title"].isna()
    for column in TEXT_COLUMNS:
        broken |= parsed[column].fillna("").str.contains(r"[\t\n\r]", regex=True)
    parsed[broken] = raw[broken]
    print(f"{split}: {len(raw):,} articles, {int(broken.sum())} read from raw news.tsv (quote parsing broke the row)")
    return parsed


def all_news():
    """Union of the three news files, one row per article (first file wins: train, dev, test)."""
    frames = [load_news_as_converted(split) for split in ("train", "dev", "test")]
    news = pd.concat(frames)
    return news[~news.index.duplicated(keep="first")]


def main():
    if TARGET.exists():
        raise SystemExit(f"{TARGET.relative_to(me.ROOT)} exists; delete it by hand to rebuild")
    vectors = np.load(SOURCE / "embeddings.npy")
    ids = json.loads((SOURCE / "article_ids.json").read_text("utf-8"))
    known = {int(s[1:]) for s in ids}
    news = all_news()
    missing = news[~news.index.isin(list(known))]
    print(f"stored vectors: {len(ids):,}; articles in train+dev+test: {len(news):,}; missing: {len(missing):,}")

    model = SentenceTransformer(MODEL, device="cpu")

    # 1. reproduce stored train vectors
    rng = np.random.default_rng(SEED)
    sample = rng.choice(len(ids), CHECK_N, replace=False)
    sample_ids = [int(ids[i][1:]) for i in sample]
    texts = [article_text(news.loc[n]) for n in sample_ids]
    again = model.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    cos = np.einsum("ij,ij->i", again, vectors[sample])
    print(f"reproduction check on {CHECK_N} train articles: cosine min {cos.min():.6f}, mean {cos.mean():.6f}")
    assert cos.min() > 0.999, "stored vectors are not reproduced; text or model differs"

    # 2. encode the missing articles
    texts = [article_text(row) for _, row in missing.iterrows()]
    new = model.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=True)
    new = new.astype(np.float32)

    TARGET.mkdir(parents=True)
    np.save(TARGET / "embeddings.npy", np.vstack([vectors, new]), allow_pickle=False)
    (TARGET / "article_ids.json").write_text(
        json.dumps(ids + [me.news_key(int(n)) for n in missing.index]), encoding="utf-8")
    meta = json.loads((SOURCE / "metadata.json").read_text("utf-8"))
    meta.update({"source": "news.tsv of MINDlarge train, dev and test",
                 "rows_from_original": len(ids), "rows_added": len(missing),
                 "added_by": "news/embed_missing_news.py",
                 "reproduction_check": {"n": CHECK_N, "cos_min": float(cos.min()), "cos_mean": float(cos.mean())}})
    (TARGET / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"saved {len(ids) + len(missing):,} vectors to {TARGET.relative_to(me.ROOT)}")


if __name__ == "__main__":
    main()
