'''
Label quality: DeepSeek V4.1 Flash vs gemma4:e4b on the v4 labels, checked against the Tag Genome.

Question: does a stronger LLM assign more accurate themes and moods than the local 4B model?
Both use exactly the same prompt (v4), vocabulary (v4_genome.json), JSON schema, input text and
limits (5 themes, 3 moods). Only the model changes. The gemma4:e4b labels are the existing ones
from experiment 08 (output/experiments/gemma4_e4b_v4/movies_enriched.csv); no new gemma calls.

Ground truth: the MovieLens Tag Genome. The v4 vocabulary consists of genome tags, and the genome
gives every film a relevance (0-1) for every tag, estimated from user tags, ratings and reviews.

Sample: SAMPLE_SIZE films with genome scores, stratified by popularity (TMDB vote count thirds
among the genome films), seed 42. LLMs know famous films better, so the thirds are reported apart.

Metrics, per film and field (themes, moods), paired between the two models:
- precision        share of assigned labels with genome relevance >= 0.5 (as in experiment 08)
- mean_relevance   mean genome relevance of the assigned labels (continuous, more sensitive)
- recall           share of the genome's top-5 vocabulary themes (top-3 moods) that the model assigned
- commonness       mean share of genome films for which an assigned label has relevance >= 0.5
                   (lower = more specific labels; model-independent, so comparable between models)
- random_precision the precision of a label drawn at random from the vocabulary (baseline)
- jaccard          overlap between the two models' label sets
Statistics: paired Wilcoxon signed-rank per film, overall and per popularity third.

Usage:
    python experiments/enrichment/label_quality_experiment.py check     # one film, prints the raw answer (no evaluation)
    python experiments/enrichment/label_quality_experiment.py           # the 100-film sample + evaluation
    python experiments/enrichment/label_quality_experiment.py all       # every genome film (1,388) + evaluation
    python experiments/enrichment/label_quality_experiment.py all v5    # the same with prompt v5 (vocabulary v5 + aliases),
                                                 # compared with DeepSeek v4 and gemma v4
'''

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))  # shared modules live in src/

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import build_movie_texts as bmt
from enrich_catalog import enrich_movies, load_records

FOLDER = Path("output") / "experiments" / "label_quality"
FOLDER.mkdir(parents=True, exist_ok=True)

NEW_MODEL = "deepseek-v4.1-flash:cloud"        # through the local Ollama installation (signed in)
GEMMA_LABELS = Path("output/experiments/gemma4_e4b_v4/movies_enriched.csv")
VOCABULARY = Path("data/vocabularies/v5_genome.json")   # evaluation uses v5 (a superset of v4) for every run
SAMPLE_SIZE = 100
RELEVANT = 0.5                                 # genome relevance threshold, as in experiment 08
TOP_GENOME = {"themes": 5, "moods": 3}         # recall: the genome's top-k vocabulary tags per film


def split_labels(value):
    return [] if not isinstance(value, str) else [v.strip() for v in value.split("|") if v.strip()]


def load_genome(movie_ids, vocabulary):
    """Genome relevance of every v4 vocabulary tag for the given TMDB movie ids:
    {tmdb_id: {tag: relevance}}, plus the share of genome films where each tag is >= 0.5."""
    links = pd.read_csv("data/ml-25m/links.csv").dropna(subset=["tmdbId"])
    ml_of = dict(zip(links["tmdbId"].astype(int), links["movieId"]))
    tags = pd.read_csv("data/ml-25m/genome/genome-tags.csv")
    vocab = set(vocabulary["themes"]) | set(vocabulary["moods"])
    tag_of = {i: t for t, i in zip(tags["tag"], tags["tagId"]) if t in vocab}
    tmdb_of = {ml_of[m]: m for m in movie_ids if m in ml_of}
    relevance, commonness_counts, n_films = {}, {}, set()
    for chunk in pd.read_csv("data/ml-25m/genome/genome-scores.csv", chunksize=2_000_000):
        chunk = chunk[chunk["tagId"].isin(tag_of)]
        for r in chunk.itertuples():
            tag = tag_of[r.tagId]
            n_films.add(r.movieId)
            if r.relevance >= RELEVANT:
                commonness_counts[tag] = commonness_counts.get(tag, 0) + 1
            if r.movieId in tmdb_of:
                relevance.setdefault(tmdb_of[r.movieId], {})[tag] = r.relevance
    commonness = {t: commonness_counts.get(t, 0) / len(n_films) for t in vocab}
    return relevance, commonness


def draw_sample(movies, relevance, all_films=False):
    """SAMPLE_SIZE genome films, about a third from each TMDB vote-count third (seed 42),
    or every genome film (all_films=True)."""
    pool = movies[movies["movie_id"].isin(relevance)].copy()
    cuts = pool["vote_count"].quantile([1 / 3, 2 / 3]).to_numpy()
    pool["third"] = np.where(pool["vote_count"] <= cuts[0], "low",
                             np.where(pool["vote_count"] <= cuts[1], "mid", "high"))
    if all_films:
        return pool.reset_index(drop=True)
    per_third = {"low": SAMPLE_SIZE // 3, "mid": SAMPLE_SIZE // 3, "high": SAMPLE_SIZE - 2 * (SAMPLE_SIZE // 3)}
    parts = [pool[pool["third"] == t].sample(n, random_state=42) for t, n in per_third.items()]
    return pd.concat(parts).reset_index(drop=True)


def score_film(labels, field, rel, commonness, allowed):
    """Metrics of one model's labels for one film and field."""
    assigned = [l for l in labels if l in rel]
    top = sorted(allowed, key=lambda t: -rel.get(t, 0))[:TOP_GENOME[field]]
    return {
        "n_labels": len(labels),
        "precision": np.mean([rel[l] >= RELEVANT for l in assigned]) if assigned else np.nan,
        "mean_relevance": np.mean([rel[l] for l in assigned]) if assigned else np.nan,
        "recall": len(set(labels) & set(top)) / len(top),
        "commonness": np.mean([commonness[l] for l in labels]) if labels else np.nan,
        "random_precision": np.mean([rel.get(t, 0) >= RELEVANT for t in allowed]),
    }


def evaluate(sample, labels, relevance, commonness, vocabulary):
    """One row per film, field and model; plus the agreement between the models."""
    rows = []
    for _, film in sample.iterrows():
        m, rel = int(film["movie_id"]), relevance[int(film["movie_id"])]
        for field, allowed in (("themes", vocabulary["themes"]), ("moods", vocabulary["moods"])):
            sets = {}
            for model, table in labels.items():
                sets[model] = set(table[m][field]) if m in table else set()
                rows.append({"movie_id": m, "title": film["title"], "third": film["third"], "field": field,
                             "model": model, **score_film(list(sets[model]), field, rel, commonness, allowed)})
            # agreement with the first model (gemma4:e4b, the reference)
            reference = sets[next(iter(sets))]
            for k, model in enumerate(sets):
                both = reference | sets[model]
                rows[-len(sets) + k]["jaccard"] = len(reference & sets[model]) / len(both) if both else np.nan
    return pd.DataFrame(rows)


def compare(per_film, pairs):
    """Means per model, and paired Wilcoxon tests (b - a for every pair (a, b)) overall and per third."""
    metrics = ["precision", "mean_relevance", "recall", "commonness", "n_labels"]
    means = (per_film.groupby(["field", "model"])[metrics + ["random_precision", "jaccard"]]
             .mean().round(3))
    tests = []
    for field in ("themes", "moods"):
        for third in ("all", "low", "mid", "high"):
            part = per_film[per_film["field"] == field]
            if third != "all":
                part = part[part["third"] == third]
            wide = {m: part[part["model"] == m].set_index("movie_id") for m in part["model"].unique()}
            for a, b in pairs:
                for metric in metrics:
                    diff = (wide[b][metric] - wide[a][metric]).dropna()
                    p = wilcoxon(diff).pvalue if diff.any() else 1.0
                    tests.append({"field": field, "third": third, "comparison": f"{b} - {a}", "metric": metric,
                                  "films": len(diff), "a": wide[a][metric].mean(), "b": wide[b][metric].mean(),
                                  "difference": diff.mean(), "better": int((diff > 0).sum()),
                                  "worse": int((diff < 0).sum()), "p_value": p})
    return means, pd.DataFrame(tests)


if __name__ == "__main__":
    vocabulary = json.loads(VOCABULARY.read_text(encoding="utf-8"))
    movies = pd.read_csv("data/data/processed/movies.csv")
    relevance, commonness = load_genome(set(movies["movie_id"]), vocabulary)
    args = sys.argv[1:]
    all_films = "all" in args
    version = "v5" if "v5" in args else "v4"
    sample = draw_sample(movies, relevance, all_films)
    tag = ("all" if all_films else f"sample{SAMPLE_SIZE}") + ("" if version == "v4" else f"_{version}")
    sample.to_csv(FOLDER / f"films_{tag}.csv", index=False)
    print(f"Sample: {len(sample)} genome films ({sample['third'].value_counts().to_dict()})")

    # Same enricher, schema and settings as experiment 08; prompt v4 (only the model changes) or
    # v5 (the same prompt text with the v5 vocabulary and the alias mapping)
    enricher = bmt.FenceStrippingEnricher(movies, model_name=NEW_MODEL, provider="ollama",
                                          max_retries=4, prompt_version=version)
    cache_of = lambda v: FOLDER / f"labels_{NEW_MODEL.replace(':', '_')}_{v}.jsonl"
    cache = cache_of(version)
    if args == ["check"]:
        film = sample.iloc[[0]]
        result = enricher.call_llm(enricher.build_prompt(enricher.build_input_text(film.iloc[0])))
        print(f"\n{film.iloc[0]['title']}: {result.get('prompt_tokens')} prompt tokens, "
              f"{result.get('output_tokens')} output tokens, {result.get('wall_seconds', 0):.1f} s")
        print(result["content"])
        raise SystemExit

    enrich_movies(enricher, sample, cache)
    load_ok = lambda path: {m: r for m, r in load_records(path).items() if r.get("status") == "ok"}
    new = load_ok(cache)
    gemma = pd.read_csv(GEMMA_LABELS).set_index("movie_id")
    as_labels = lambda recs: {m: {"themes": r["themes"], "moods": r["mood"]} for m, r in recs.items()}
    labels = {"gemma4:e4b v4": {int(m): {"themes": split_labels(r["llm_themes"]), "moods": split_labels(r["llm_mood"])}
                                for m, r in gemma.iterrows()},
              "deepseek v4": as_labels(load_ok(cache_of("v4")))}
    pairs = [("gemma4:e4b v4", "deepseek v4")]
    if version == "v5":
        labels["deepseek v5"] = as_labels(new)
        pairs += [("deepseek v4", "deepseek v5"), ("gemma4:e4b v4", "deepseek v5")]
    missing = [m for m in sample["movie_id"] if m not in labels[f"deepseek {version}"]]
    print(f"{len(new)} films labelled with prompt {version}; {len(missing)} without a valid answer")
    for v in ("v4", "v5") if version == "v5" else ("v4",):
        recs = [r for m, r in load_ok(cache_of(v)).items() if m in set(sample["movie_id"])]
        dropped = sum("unknown label" in i for r in recs for i in r.get("issues", []))
        mapped = sum("mapped" in i for r in recs for i in r.get("issues", []))
        with_drop = sum(any("unknown label" in i for i in r.get("issues", [])) for r in recs)
        print(f"   DeepSeek {v}: {dropped} labels dropped (films with a drop: {with_drop}), {mapped} labels mapped by alias")

    per_film = evaluate(sample, labels, relevance, commonness, vocabulary)
    means, tests = compare(per_film, pairs)
    per_film.to_csv(FOLDER / f"per_film_{tag}.csv", index=False)
    means.to_csv(FOLDER / f"means_{tag}.csv")
    tests.to_csv(FOLDER / f"tests_{tag}.csv", index=False)

    records = list(new.values())
    tokens_in = sum(r.get("prompt_tokens", 0) for r in records)
    tokens_out = sum(r.get("output_tokens", 0) for r in records)
    pd.set_option("display.width", 250)
    print("\nMeans per field and model:")
    print(means.to_string())
    print("\nPaired tests (b - a):")
    print(tests.round(4).to_string(index=False))
    print(f"\nRepaired answers: {sum(r.get('repaired', False) for r in records)} | tokens in {tokens_in:,}, "
          f"out {tokens_out:,} | cost at peak price about ${(tokens_in * 0.30 + tokens_out * 1.20) / 1e6:.2f}")
