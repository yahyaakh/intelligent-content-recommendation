"""E1: how well do the child-safety signals find films that are unsuitable for a child?

The recomm.py ConstraintLayer removes a film for a user under 13 when the film has a genre in
{Horror, Thriller, Crime, War} or when a keyword from CHILD_UNSAFE_KEYWORDS appears as a substring
of its TMDB keywords. The LLM enrichment gives every film an audience label (children / family /
teens / adults) and five content intensities (violence, sexual content, language, substance use,
frightening scenes; 0 = none ... 3 = strong). This script compares these signals with two ground truths.

Ground truth A (main): the US certification on TMDB (MPAA).
    unsuitable for a child (under 13) = PG-13, R or NC-17      suitable = G or PG
    unsuitable for a teen  (under 17) = R or NC-17             suitable = G, PG or PG-13
    Films without a US certification (or NR) are left out.
Ground truth B (second): the MovieLens Tag Genome. A film is unsuitable for a child when one of the
    content tags below has relevance >= 0.5 (0.4 and 0.6 shown as a sensitivity check).
    The genome covers about 1,389 of the 1,957 films. The tag "horror" is not used, so the
    ground truth does not favour the genre rule.

Rules compared (each says "unsuitable" or not):
    genre             a genre in {Horror, Thriller, Crime, War}            (recomm.py)
    kw_substring      a CHILD_UNSAFE_KEYWORDS entry inside the keyword text (recomm.py)
    kw_exact          a CHILD_UNSAFE_KEYWORDS entry equal to one keyword    (substring bug fixed)
    current           genre or kw_substring = what recomm.py does today
    audience          LLM audience is teens or adults (child) / adults (teen)
    intensity         any LLM intensity >= 2 (child) / any intensity = 3 (teen)
    audience_or_int   audience or intensity
The rules are fixed before looking at the results; nothing is tuned on the ground truth.

Metrics: precision, recall, F1, accuracy and specificity for the class "unsuitable", with 95%
bootstrap intervals over films (2,000 resamples, seed 42). Paired bootstrap of the F1 difference of
every rule against "current". Breakdown by popularity third (TMDB vote count), because LLMs know
famous films better. Per dimension: Spearman correlation of each LLM intensity with its genome tags.

Caveat: an LLM may remember a film's MPAA rating, so on famous films it can partly copy ground truth A.
That is fine for a working system, but it is why the popularity breakdown and ground truth B exist.

Usage:
    python experiments/constraints/safety_labels_experiment.py fetch    # TMDB US certifications (free, about 2,000 calls, resumable)
    python experiments/constraints/safety_labels_experiment.py run      # evaluation (no API calls)
Outputs: output/experiments/safety_labels/
"""
import ast
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PROJECT = Path(__file__).resolve().parents[2]
OUT = PROJECT / "output/experiments/safety_labels"
MOVIES = PROJECT / "data/data/processed_30k/movies.csv"
LINKS = PROJECT / "data/ml-25m/links.csv"
GENOME_SCORES = PROJECT / "data/ml-25m/genome/genome-scores.csv"
GENOME_TAGS = PROJECT / "data/ml-25m/genome/genome-tags.csv"
CERT_RAW = OUT / "tmdb_release_dates_us.jsonl"   # raw US entries, one line per film
CERTS = OUT / "tmdb_certifications.csv"

# LLM label sets (all 1,957 films). DeepSeek v5 is the main one; the others use older prompts.
LLM_LABELS = {
    "DeepSeek V4.1 Flash (v5)": "output/experiments/deepseek_v4.1_flash_v5/movies_enriched.csv",
    "gemma4:e4b (v4)":          "output/experiments/gemma4_e4b_v4/movies_enriched.csv",
    "gemma4:31b (v2)":          "output/experiments/gemma4_31b-cloud/movies_enriched.csv",
}
# gpt-oss:120b (v2) is left out: 129 of its films have no labels (failed answers), so its rules
# could not be compared with the others on the same films.
MAIN_LLM = "DeepSeek V4.1 Flash (v5)"
DIMENSIONS = ["violence", "sexual_content", "language", "substance_use", "frightening_scenes"]
ADULT_GENRES = {"Horror", "Thriller", "Crime", "War"}   # recomm.py ConstraintLayer

# Genome content tags per intensity dimension (checked against genome-tags.csv)
GENOME_GROUPS = {
    "violence": ["violence", "violent", "gratuitous violence", "gore", "goretastic", "bloody",
                 "brutal", "brutality", "torture"],
    "sexual_content": ["nudity", "notable nudity", "male nudity", "nudity (full frontal)",
                       "nudity (full frontal - brief)", "nudity (full frontal - notable)",
                       "nudity (rear)", "nudity (topless)", "nudity (topless - brief)",
                       "nudity (topless - notable)", "sex", "sexual", "sexuality", "erotic", "rape",
                       "sexual abuse", "sexualized violence"],
    "language": ["foul language"],
    "substance_use": ["drugs", "drug abuse", "drug addiction"],
    "frightening_scenes": ["scary", "disturbing", "suicide", "suicide attempt"],
}
GENOME_THRESHOLD = 0.5
CERT_ORDER = ["G", "PG", "PG-13", "R", "NC-17"]
N_BOOT = 2000
SEED = 42


# ─────────────────────────────── ground truth A: TMDB certifications ───────────────────────────────

def catalog():
    """The 1,957 enriched films (movie_id = TMDB id)."""
    ids = set(pd.read_csv(PROJECT / LLM_LABELS[MAIN_LLM])["movie_id"])
    movies = pd.read_csv(MOVIES)
    return movies[movies["movie_id"].isin(ids)].drop_duplicates("movie_id").reset_index(drop=True)


def us_certification(entries):
    """One certification from TMDB's US release dates: theatrical first, then limited, premiere,
    digital, physical, TV. Returns None when no MPAA rating is given."""
    by_type = {}
    for release in entries:
        cert = (release.get("certification") or "").strip().upper()
        if cert in CERT_ORDER:
            by_type.setdefault(release.get("type"), cert)
    for release_type in (3, 2, 1, 4, 5, 6):
        if release_type in by_type:
            return by_type[release_type]
    return None


def fetch():
    import requests
    from dotenv import load_dotenv
    load_dotenv(PROJECT / ".env")
    key = os.getenv("TMDB_API_KEY")
    if not key:
        raise RuntimeError("TMDB_API_KEY is missing from .env")
    OUT.mkdir(parents=True, exist_ok=True)
    done = set()
    if CERT_RAW.exists():
        done = {json.loads(line)["movie_id"] for line in CERT_RAW.open(encoding="utf-8")}
    films = [m for m in catalog()["movie_id"] if m not in done]
    print(f"{len(done)} films cached, {len(films)} to fetch", flush=True)
    session = requests.Session()
    with CERT_RAW.open("a", encoding="utf-8") as out:
        for i, movie_id in enumerate(films, 1):
            for attempt in range(5):
                resp = session.get(f"https://api.themoviedb.org/3/movie/{movie_id}/release_dates",
                                   params={"api_key": key}, timeout=20)
                if resp.status_code == 429 or resp.status_code >= 500:
                    time.sleep(2 ** attempt)
                    continue
                break
            if resp.status_code == 404:
                us = None
            else:
                resp.raise_for_status()
                us = next((c["release_dates"] for c in resp.json().get("results", [])
                           if c.get("iso_3166_1") == "US"), [])
            out.write(json.dumps({"movie_id": int(movie_id), "us": us}) + "\n")
            if i % 200 == 0:
                out.flush()
                print(f"  {i}/{len(films)}", flush=True)
            time.sleep(0.03)
    write_certifications()


def write_certifications():
    rows = [json.loads(line) for line in CERT_RAW.open(encoding="utf-8")]
    certs = pd.DataFrame({"movie_id": [r["movie_id"] for r in rows],
                          "certification": [us_certification(r["us"] or []) for r in rows]})
    certs = certs.drop_duplicates("movie_id", keep="last")
    certs.to_csv(CERTS, index=False)
    print(f"Saved {CERTS} ({certs['certification'].notna().sum()} of {len(certs)} with a US rating)")
    print(certs["certification"].value_counts(dropna=False).to_string())


# ─────────────────────────────── ground truth B: Tag Genome ───────────────────────────────

def genome_scores(movie_ids):
    """{dimension: Series(movie_id -> max relevance over the dimension's tags)} for genome films."""
    tags = pd.read_csv(GENOME_TAGS)
    wanted = {t for group in GENOME_GROUPS.values() for t in group}
    missing = wanted - set(tags["tag"])
    if missing:
        raise ValueError(f"Genome tags not found: {sorted(missing)}")
    tag_ids = tags[tags["tag"].isin(wanted)].set_index("tagId")["tag"]
    links = pd.read_csv(LINKS).dropna(subset=["tmdbId"])
    links["tmdbId"] = links["tmdbId"].astype(int)
    links = links[links["tmdbId"].isin(set(movie_ids))].drop_duplicates("tmdbId")
    tmdb_of = dict(zip(links["movieId"], links["tmdbId"]))
    scores = pd.read_csv(GENOME_SCORES)
    scores = scores[scores["tagId"].isin(tag_ids.index) & scores["movieId"].isin(tmdb_of)]
    scores = scores.assign(tag=scores["tagId"].map(tag_ids), movie_id=scores["movieId"].map(tmdb_of))
    wide = scores.pivot_table(index="movie_id", columns="tag", values="relevance", aggfunc="max")
    return pd.DataFrame({dim: wide[group].max(axis=1) for dim, group in GENOME_GROUPS.items()})


# ─────────────────────────────── the rules ───────────────────────────────

def child_unsafe_keywords():
    """CHILD_UNSAFE_KEYWORDS read from recomm.py without importing it (it loads models on import)."""
    tree = ast.parse((PROJECT / "recomm.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "CHILD_UNSAFE_KEYWORDS"
                                                for t in node.targets):
            return {k.lower() for k in ast.literal_eval(node.value)}
    raise ValueError("CHILD_UNSAFE_KEYWORDS not found in recomm.py")


def rule_table(films, keywords):
    """Rule outputs per film (True = unsuitable). LLM rules get one column per LLM and age."""
    genres = films["genres"].fillna("").str.split("|").apply(set)
    text = films["keywords"].fillna("").str.lower()
    tokens = text.str.split("|").apply(lambda ks: {k.strip() for k in ks})
    rules = pd.DataFrame({"movie_id": films["movie_id"]})
    rules["genre"] = genres.apply(lambda g: bool(g & ADULT_GENRES))
    rules["kw_substring"] = text.apply(lambda t: any(k in t for k in keywords))   # recomm.py:1712
    rules["kw_exact"] = tokens.apply(lambda t: bool(t & keywords))
    rules["current"] = rules["genre"] | rules["kw_substring"]
    for llm, path in LLM_LABELS.items():
        labels = pd.read_csv(PROJECT / path).drop_duplicates("movie_id").set_index("movie_id")
        labels = labels.reindex(films["movie_id"])
        if labels["llm_audience"].isna().any():
            raise ValueError(f"{llm}: films without labels")
        top = labels[[f"intensity_{d}" for d in DIMENSIONS]].max(axis=1).to_numpy()
        audience = labels["llm_audience"].to_numpy()
        rules[f"audience|{llm}|child"] = np.isin(audience, ["teens", "adults"])
        rules[f"intensity|{llm}|child"] = top >= 2
        rules[f"audience_or_int|{llm}|child"] = rules[f"audience|{llm}|child"] | rules[f"intensity|{llm}|child"]
        rules[f"audience|{llm}|teen"] = audience == "adults"
        rules[f"intensity|{llm}|teen"] = top >= 3
        rules[f"audience_or_int|{llm}|teen"] = rules[f"audience|{llm}|teen"] | rules[f"intensity|{llm}|teen"]
        for d in DIMENSIONS:
            rules[f"int_{d}|{llm}"] = labels[f"intensity_{d}"].to_numpy()
    return rules


def rule_columns(age):
    """(display name, column) for one age group; the recomm.py rules have no age setting."""
    cols = [("genre (recomm.py)", "genre"), ("keywords, substring (recomm.py)", "kw_substring"),
            ("keywords, exact", "kw_exact"), ("current = genre or substring (recomm.py)", "current")]
    for llm in LLM_LABELS:
        for rule in ("audience", "intensity", "audience_or_int"):
            cols.append((f"{rule} — {llm}", f"{rule}|{llm}|{age}"))
    return cols


# ─────────────────────────────── metrics ───────────────────────────────

def metrics(pred, truth):
    tp = int((pred & truth).sum()); fp = int((pred & ~truth).sum())
    fn = int((~pred & truth).sum()); tn = int((~pred & ~truth).sum())
    precision = tp / (tp + fp) if tp + fp else np.nan
    recall = tp / (tp + fn) if tp + fn else np.nan
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else np.nan
    return {"precision": precision, "recall": recall, "f1": f1,
            "accuracy": (tp + tn) / len(pred), "specificity": tn / (tn + fp) if tn + fp else np.nan,
            "blocked_ok_films": fp, "missed_unsafe_films": fn, "tp": tp, "tn": tn}


def boot_samples(n):
    rng = np.random.default_rng(SEED)
    return rng.integers(0, n, size=(N_BOOT, n))


def evaluate(rules, truth, age, samples, reference="current"):
    """One row per rule: metrics, bootstrap 95% CIs, paired F1 difference against the reference."""
    t = truth.to_numpy()
    ref = rules[reference].to_numpy()
    ref_f1 = np.array([metrics(ref[s], t[s])["f1"] for s in samples])
    rows = []
    for name, col in rule_columns(age):
        p = rules[col].to_numpy()
        row = {"rule": name, "column": col, "n_films": len(t), "n_unsuitable": int(t.sum()),
               "flagged": int(p.sum()), **metrics(p, t)}
        boot = [metrics(p[s], t[s]) for s in samples]
        for m in ("precision", "recall", "f1"):
            vals = np.array([b[m] for b in boot], dtype=float)
            row[f"{m}_lo"], row[f"{m}_hi"] = np.nanpercentile(vals, [2.5, 97.5])
        diff = np.array([b["f1"] for b in boot]) - ref_f1
        row["f1_vs_current"] = row["f1"] - metrics(ref, t)["f1"]
        row["f1_vs_current_lo"], row["f1_vs_current_hi"] = np.nanpercentile(diff, [2.5, 97.5])
        rows.append(row)
    return pd.DataFrame(rows)


def show(table, title):
    print(f"\n{title}  (n = {table['n_films'].iloc[0]}, unsuitable = {table['n_unsuitable'].iloc[0]})")
    print(f"{'rule':52s} {'flag':>5s} {'prec':>6s} {'recall':>6s} {'F1':>6s} {'F1 95% CI':>15s} "
          f"{'ΔF1 vs current [95% CI]':>27s} {'wrongly blocked':>15s} {'missed':>6s}")
    for _, r in table.iterrows():
        print(f"{r['rule']:52s} {r['flagged']:5d} {r['precision']:6.3f} {r['recall']:6.3f} {r['f1']:6.3f} "
              f"[{r['f1_lo']:.3f}, {r['f1_hi']:.3f}] {r['f1_vs_current']:+8.3f} "
              f"[{r['f1_vs_current_lo']:+.3f}, {r['f1_vs_current_hi']:+.3f}] "
              f"{r['blocked_ok_films']:15d} {r['missed_unsafe_films']:6d}")


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    films = catalog()
    rules = rule_table(films, child_unsafe_keywords())
    print(f"{len(films)} films; {len(child_unsafe_keywords())} unsafe keywords in recomm.py")

    # Ground truth A
    if not CERTS.exists():
        raise FileNotFoundError("Run `python experiments/constraints/safety_labels_experiment.py fetch` first.")
    certs = pd.read_csv(CERTS)
    data = films[["movie_id", "title", "year", "genres", "vote_count"]].merge(certs, on="movie_id", how="left")
    print(f"\nTMDB US certification: {data['certification'].notna().sum()} of {len(data)} films")
    print(data["certification"].value_counts(dropna=False).reindex(CERT_ORDER + [np.nan]).to_string())
    data = data.merge(rules, on="movie_id")
    rated = data[data["certification"].notna()].reset_index(drop=True)
    rated["popularity_third"] = pd.qcut(rated["vote_count"].rank(method="first"), 3,
                                        labels=["least known", "middle", "most known"])
    samples = boot_samples(len(rated))
    results = []
    for age, unsafe in (("child", ["PG-13", "R", "NC-17"]), ("teen", ["R", "NC-17"])):
        truth = rated["certification"].isin(unsafe)
        table = evaluate(rated, truth, age, samples).assign(ground_truth="TMDB US certification", age=age)
        label = "under 13 (PG-13/R/NC-17 = unsuitable)" if age == "child" else "under 17 (R/NC-17 = unsuitable)"
        show(table, f"Ground truth A, {label}")
        results.append(table)

    # Breakdown by popularity third (child, main rules only)
    main = ["current", f"audience|{MAIN_LLM}|child", f"intensity|{MAIN_LLM}|child",
            f"audience_or_int|{MAIN_LLM}|child", "kw_exact", "genre"]
    third_rows = []
    for third, group in rated.groupby("popularity_third", observed=True):
        truth = group["certification"].isin(["PG-13", "R", "NC-17"]).to_numpy()
        for col in main:
            third_rows.append({"third": third, "rule": col, "n_films": len(group),
                               "median_votes": int(group["vote_count"].median()),
                               **metrics(group[col].to_numpy(), truth)})
    thirds = pd.DataFrame(third_rows)
    print("\nBy popularity third (TMDB vote count), child, ground truth A:")
    print(thirds.pivot(index="rule", columns="third", values="f1").round(3).to_string())
    print(thirds.pivot(index="rule", columns="third", values="recall").round(3).add_suffix(" recall").to_string())

    # Ground truth B
    genome = genome_scores(films["movie_id"])
    gdata = data.merge(genome.add_prefix("genome_"), left_on="movie_id", right_index=True)
    gsamples = boot_samples(len(gdata))
    top = gdata[[f"genome_{d}" for d in DIMENSIONS]].max(axis=1)
    for threshold in (0.4, GENOME_THRESHOLD, 0.6):
        truth = top >= threshold
        table = evaluate(gdata, truth, "child", gsamples).assign(ground_truth=f"Tag Genome >= {threshold}", age="child")
        if threshold == GENOME_THRESHOLD:
            show(table, f"Ground truth B, Tag Genome content tag >= {threshold} (child)")
        else:
            print(f"\nSensitivity, Tag Genome >= {threshold}: F1 "
                  + ", ".join(f"{r['column']}={r['f1']:.3f}" for _, r in table.iterrows()
                              if r["column"] in main))
        results.append(table)

    # Agreement between the two ground truths (films that have both)
    both = gdata[gdata["certification"].notna()]
    a = both["certification"].isin(["PG-13", "R", "NC-17"])
    b = both[[f"genome_{d}" for d in DIMENSIONS]].max(axis=1) >= GENOME_THRESHOLD
    print(f"\nGround truths A and B on the {len(both)} films with both: agree on {(a == b).mean():.3f}; "
          f"A unsuitable {a.sum()}, B unsuitable {b.sum()}")

    # Per-dimension: LLM intensity vs genome relevance
    dim_rows = []
    for llm in LLM_LABELS:
        for d in DIMENSIONS:
            rho, p = spearmanr(gdata[f"int_{d}|{llm}"], gdata[f"genome_{d}"])
            dim_rows.append({"llm": llm, "dimension": d, "spearman": rho, "p": p, "n_films": len(gdata)})
    dims = pd.DataFrame(dim_rows)
    print("\nSpearman correlation, LLM intensity vs genome relevance (max over the dimension's tags):")
    print(dims.pivot(index="llm", columns="dimension", values="spearman").round(3).to_string())

    # Examples for the write-up (main LLM, child, ground truth A)
    ex = rated.assign(truth=rated["certification"].isin(["PG-13", "R", "NC-17"]))
    blocked_ok = ex[ex["current"] & ~ex["truth"]].sort_values("vote_count", ascending=False)
    missed = ex[~ex["current"] & ex["truth"]].sort_values("vote_count", ascending=False)
    llm_missed = ex[~ex[f"audience_or_int|{MAIN_LLM}|child"] & ex["truth"]].sort_values("vote_count", ascending=False)
    llm_blocked = ex[ex[f"audience_or_int|{MAIN_LLM}|child"] & ~ex["truth"]].sort_values("vote_count", ascending=False)
    examples = pd.concat([blocked_ok.head(15).assign(case="current blocks a G/PG film"),
                          missed.head(15).assign(case="current lets a PG-13/R film through"),
                          llm_blocked.head(15).assign(case="LLM rule blocks a G/PG film"),
                          llm_missed.head(15).assign(case="LLM rule lets a PG-13/R film through")])
    examples = examples[["case", "title", "year", "certification", "genres", "vote_count"]]
    for case, group in examples.groupby("case", sort=False):
        print(f"\n{case} (most known first): " + "; ".join(
            f"{r.title} ({r.certification})" for r in group.head(8).itertuples()))

    pd.concat(results).to_csv(OUT / "summary.csv", index=False)
    thirds.to_csv(OUT / "by_popularity_third.csv", index=False)
    dims.to_csv(OUT / "dimension_correlations.csv", index=False)
    examples.to_csv(OUT / "examples.csv", index=False)
    data.to_csv(OUT / "per_film.csv", index=False)
    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "run"
    if mode == "fetch":
        fetch()
    elif mode == "certs":
        write_certifications()
    elif mode == "run":
        run()
    else:
        raise SystemExit("usage: python experiments/constraints/safety_labels_experiment.py fetch | certs | run")
