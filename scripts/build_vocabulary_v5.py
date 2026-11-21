"""Build the v5 theme/mood vocabulary: v4 plus the gaps found in the label-quality experiment.

Evidence (output/experiments/label_quality/, DeepSeek V4.1 Flash on 1,388 genome films with the v4
prompt): 473 labels fell outside the v4 vocabulary and were dropped. They fall into four groups:

| Group                                   | Uses | Fix in v5                                       |
|-----------------------------------------|------|-------------------------------------------------|
| genome tag, not in the v4 candidates    |   85 | add the theme tags; keep genres excluded         |
| label exists, but in the other field    |   23 | alias: move it to the right field               |
| near-duplicate of a v4 label            |   (incl. above) | alias: map to the v4 label (satire -> satirical) |
| not a genome tag (grief, faith, ...)    |  361 | not added: v5 stays checkable against the genome |
| v4 candidate dropped by the 5-film rule |    4 | unchanged (the rule stays)                        |

Rules, the same as v4: every label is an exact genome tag (so labels stay verifiable), genres stay
out (they are already in the LLM input), and a tag is kept only if the genome rates it >= 0.5 for at
least MIN_MOVIES catalog films.

The aliases are not new labels: they tell the answer validation to MAP an answer such as
"satire" (theme) to "satirical" (mood) instead of dropping it.

Usage:  python scripts/build_vocabulary_v5.py   ->  data/vocabularies/v5_genome.json
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # shared modules live in src/
import json
from pathlib import Path

import pandas as pd

import build_vocabulary_v4 as v4

OUT = Path("data/vocabularies") / "v5_genome.json"

# Genome theme tags that the models asked for and v4 did not offer (uses out of 1,388 films in
# brackets). "coming of age" was a plain omission in v4 (adolescence and high school were there).
ADDED_THEMES = ["coming of age",        # 20
                "infidelity",           # 1, adultery is in v4 but infidelity is the common word
                "morality",             # 1
                "humanity",             # 1
                "voyeurism",            # 1
                "food",                 # 1
                "poetry",               # 1
                "video game",           # 1
                "undercover cop"]       # 1, plus 6 uses of "undercover"
ADDED_MOODS = []

# Deliberately NOT added: genome tags that are genres or quality/style judgements, already in the
# input or not about the content: romance, adventure, mystery, crime, suspense, dark comedy, parody,
# neo-noir, cartoon, swashbuckler, stylish, holiday, time, storytelling, violence. Also not
# "idealism" (2 uses): the genome rates it >= 0.5 for 341 catalog films, too broad to be informative.

# Answers to map instead of drop: label as written by a model -> (field, label in the vocabulary)
ALIASES = {
    # the right label in the other field
    "tragedy": ("moods", "tragedy"), "tragic": ("moods", "tragedy"),
    "dark humor": ("moods", "dark humor"), "inspirational": ("moods", "inspirational"),
    "surreal": ("moods", "surreal"),
    # near-duplicates of a v4 label (genome tags or plain variants)
    "satire": ("moods", "satirical"), "surrealism": ("moods", "surreal"),
    "nostalgia": ("moods", "nostalgic"), "melancholy": ("moods", "melancholic"),
    "coming-of-age": ("themes", "coming of age"), "undercover": ("themes", "undercover cop"),
}


if __name__ == "__main__":
    tags = pd.read_csv("data/ml-25m/genome/genome-tags.csv")
    tag_id = dict(zip(tags["tag"], tags["tagId"]))
    themes = v4.split(v4.THEME_CANDIDATES) + ADDED_THEMES
    moods = v4.split(v4.MOOD_CANDIDATES) + ADDED_MOODS
    unknown = [t for t in themes + moods if t not in tag_id]
    assert not unknown, f"not genome tags: {unknown}"
    assert not set(themes) & set(moods), "a tag is in both lists"
    assert len(set(themes)) == len(themes) and len(set(moods)) == len(moods), "duplicate candidates"

    links = pd.read_csv("data/ml-25m/links.csv").dropna(subset=["tmdbId"])
    links["tmdbId"] = links["tmdbId"].astype(int)
    catalog = set(pd.read_csv("data/data/processed/movies.csv")["movie_id"])
    ml_ids = set(links.loc[links["tmdbId"].isin(catalog), "movieId"])
    scores = pd.read_csv("data/ml-25m/genome/genome-scores.csv")
    scores = scores[scores["movieId"].isin(ml_ids) & (scores["relevance"] >= v4.MIN_SCORE)]
    movies_per_tag = scores.groupby("tagId")["movieId"].nunique()
    count = {t: int(movies_per_tag.get(tag_id[t], 0)) for t in themes + moods}

    kept_themes = [t for t in themes if count[t] >= v4.MIN_MOVIES]
    kept_moods = [t for t in moods if count[t] >= v4.MIN_MOVIES]
    added = [t for t in ADDED_THEMES + ADDED_MOODS if t in kept_themes + kept_moods]
    not_kept = [f"{t} ({count[t]})" for t in ADDED_THEMES + ADDED_MOODS if t not in added]
    vocabulary = set(kept_themes) | set(kept_moods)
    aliases = {a: [f, t] for a, (f, t) in ALIASES.items() if t in vocabulary}

    print(f"themes: {len(kept_themes)} (v4: 246) | moods: {len(kept_moods)} (v4: 103)")
    print("added:", ", ".join(f"{t} ({count[t]} films)" for t in added))
    print("candidates below the 5-film rule:", ", ".join(not_kept) or "none")
    print("aliases:", ", ".join(f"{a} -> {t}" for a, (f, t) in aliases.items()))

    OUT.write_text(json.dumps({
        "source": "MovieLens 25M Tag Genome; v4 (build_vocabulary_v4.py) plus the gaps found in the "
                  "label-quality experiment (build_vocabulary_v5.py)",
        "filter": f"genome relevance >= {v4.MIN_SCORE} for at least {v4.MIN_MOVIES} of the 1,957 catalog movies",
        "themes": kept_themes, "moods": kept_moods,
        "added_in_v5": added,
        "aliases": aliases,
        "catalog_movies_per_tag": {t: count[t] for t in kept_themes + kept_moods},
    }, indent=2), encoding="utf-8")
    print(f"Saved {OUT}")
