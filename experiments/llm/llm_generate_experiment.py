'''
Idea 2: the LLM recommends films from its own knowledge, with no candidate list.
Question: can an LLM recommend well to a new user on its own, without collaborative
filtering? Compared with popularity, Hybrid E and idea 1 (the same LLM reranking
Hybrid E's candidates), on the same users and the same answer key.

How it works
- Users, seeds, answer key and split are the same as in experiments 06-12 and the reranking
  experiment (llm_rerank_experiment.py, idea 1).
- The LLM sees only the user's seed films and is asked for N_RECOMMEND films, as JSON with
  title and year. It never sees the catalogue or any candidate list.
- Every answer is matched to the catalogue by title and year and put into one category:
    valid        in the candidate pool, not seen, not a duplicate -> can be a hit
    seen         one of the user's 10 seed films
    not_in_pool  in the 1,957-film catalogue but not among the 1,547 candidates
    unmatched    not in the catalogue (a real film outside it, or an invented one)
    duplicate    the same film twice
- Two scores per user:
    strict  the first 10 answers in the model's order; anything not valid is a miss
    filled  the first 10 VALID answers (asking for 15 leaves room to fill the list)

Prompts
    g1  the seed films as title, year and genres                       (like p1 in idea 1)
    g2  + director, cast and the seed profile                          (like p2 in idea 1)

Usage
    python experiments/llm/llm_generate_experiment.py test     # matcher test only, no model calls
    python experiments/llm/llm_generate_experiment.py          # the runs in the main block
'''

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))  # shared modules live in src/
import difflib                   # fuzzy title matching
import json
import re
import sys
import unicodedata               # removing accents when normalising titles
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from tqdm import tqdm

import build_movie_texts as bmt
import cold_start_experiment as cs

ARGS = sys.argv[1:]
sys.argv = [sys.argv[0]]                 # llm_rerank_experiment reads sys.argv at import
import llm_rerank_experiment as lr      # load_data, movie_line, profile_from_seeds, call_model, hybrid_e_candidates
import profile_experiment as pe         # popularity_thirds (long-tail share, as in experiments 07 and 12)

FOLDER = Path("output") / "experiments" / "llm_generate"
FOLDER.mkdir(parents=True, exist_ok=True)

N_RECOMMEND = 15             # films asked for; the first 10 valid ones are scored
YEAR_TOLERANCE = 1           # an LLM's year may be off by one
FUZZY_CUTOFF = 0.9           # difflib similarity needed for a fuzzy title match
K = bmt.K                    # 10
CATEGORIES = ["valid", "seen", "not_in_pool", "unmatched", "duplicate"]


# ───────────────────────── title matching ─────────────────────────
ROMAN = {"ii": "2", "iii": "3", "iv": "4", "v": "5", "vi": "6", "vii": "7", "viii": "8", "ix": "9", "x": "10"}
LEADING_ARTICLES = ("the ", "a ", "an ")
# Well-known alternative titles an LLM may use -> the title in the catalogue
ALIAS_TITLES = {"Seven": "Se7en",
                "Harry Potter and the Sorcerer's Stone": "Harry Potter and the Philosopher's Stone"}


def normalise_title(title):
    """'The Godfather: Part II' -> 'godfather part 2'. Lowercase, no accents, no punctuation,
    '&' -> 'and', roman numerals -> digits, no leading article, single spaces; known
    alternative titles are mapped to the catalogue title."""
    text = basic_normalise(title)
    return ALIASES.get(text, text)


def basic_normalise(title):
    text = unicodedata.normalize("NFKD", str(title)).encode("ascii", "ignore").decode()
    text = text.lower().replace("&", " and ")
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    words = [ROMAN.get(w, w) for w in text.split()]
    text = " ".join(words)
    for article in LEADING_ARTICLES:
        if text.startswith(article):
            text = text[len(article):]
            break
    return text.strip()


ALIASES = {basic_normalise(a): basic_normalise(b) for a, b in ALIAS_TITLES.items()}


def main_title(title):
    """The part before a subtitle: 'Star Wars: Episode IV - A New Hope' -> 'Star Wars'."""
    return re.split(r"\s*[:–—]\s*|\s+-\s+|\s+or:\s+", str(title), maxsplit=1)[0]


def build_title_index(movie_by_id):
    """Two dictionaries, normalised title -> [(movie_id, year), ...]: one for full titles and one
    for main titles (before a subtitle). Lists, because remakes share titles."""
    full, short, compact = {}, {}, {}
    for movie_id, row in movie_by_id.iterrows():
        entry = (int(movie_id), int(row["year"]))
        key = normalise_title(row["title"])
        full.setdefault(key, []).append(entry)
        short.setdefault(normalise_title(main_title(row["title"])), []).append(entry)
        compact.setdefault(key.replace(" ", ""), []).append(entry)   # 'c r a z y' / 'spider man' -> 'crazy' / 'spiderman'
    return {"full": full, "short": short, "compact": compact, "keys": list(full)}


def pick_by_year(entries, year):
    """The entry whose year is within YEAR_TOLERANCE of `year` (closest first); with no year,
    only an unambiguous title is accepted."""
    if year is None:
        return entries[0][0] if len(entries) == 1 else None
    close = [(abs(y - year), m) for m, y in entries if abs(y - year) <= YEAR_TOLERANCE]
    return min(close)[1] if close else None


def parse_year(value):
    match = re.search(r"(18|19|20)\d{2}", str(value)) if value is not None else None
    return int(match.group()) if match else None


def match_title(title, year, index):
    """Catalogue movie_id for an LLM's (title, year), or None.
    1. exact normalised title, year within +-1;
    2. main titles (subtitles removed on either side), year within +-1;
    3. the title without spaces ('Spider-Man' = 'Spiderman', 'C.R.A.Z.Y.' = 'Crazy'), year within +-1;
    4. fuzzy title (difflib, cutoff FUZZY_CUTOFF), year within +-1."""
    key = normalise_title(title)
    if key in index["full"]:
        found = pick_by_year(index["full"][key], year)
        if found is not None:
            return found
    for short_key in {normalise_title(main_title(title)), key}:
        if short_key in index["short"]:
            found = pick_by_year(index["short"][short_key], year)
            if found is not None:
                return found
        if short_key in index["full"]:
            found = pick_by_year(index["full"][short_key], year)
            if found is not None:
                return found
    if key.replace(" ", "") in index["compact"]:
        found = pick_by_year(index["compact"][key.replace(" ", "")], year)
        if found is not None:
            return found
    for close in difflib.get_close_matches(key, index["keys"], n=3, cutoff=FUZZY_CUTOFF):
        found = pick_by_year(index["full"][close], year)
        if found is not None:
            return found
    return None


def test_matcher(data, index, n=300):
    """Automatic check without any model call: rewrite catalogue titles the way an LLM might and
    check that each variant is matched back to the right film."""
    mb = data["movie_by_id"]
    rng = np.random.default_rng(42)
    sample = mb.loc[rng.choice(mb.index.to_numpy(), size=n, replace=False)]
    variants = {
        "as in catalogue": lambda t, y: (t, y),
        "lowercase": lambda t, y: (t.lower(), y),
        "no punctuation": lambda t, y: (re.sub(r"[^\w\s]", "", t), y),
        "year + 1": lambda t, y: (t, y + 1),
        "no year": lambda t, y: (t, None),
        "'The' added or removed": lambda t, y: (t[4:] if t.startswith("The ") else "The " + t, y),
    }
    results = {}
    for name, make in variants.items():
        ok = sum(match_title(*make(row["title"], int(row["year"])), index) == movie_id
                 for movie_id, row in sample.iterrows())
        results[name] = ok / n
    # Hand-written variants of well-known titles (skipped if the film is not in the catalogue)
    hand = [("Godfather Part 2", 1974, "The Godfather Part II"), ("The Godfather: Part II", 1974, "The Godfather Part II"),
            ("Star Wars: Episode IV - A New Hope", 1977, "Star Wars"), ("Seven", 1995, "Se7en"),
            ("Se7en", 1995, "Se7en"), ("Leon: The Professional", 1994, "Léon: The Professional"),
            ("Goodfellas", 1990, "GoodFellas"), ("Dr. Strangelove", 1964, "Dr. Strangelove or: How I Learned to Stop Worrying and Love the Bomb"),
            ("Terminator 2", 1991, "Terminator 2: Judgment Day"), ("Raiders of the Lost Ark", 1981, "Raiders of the Lost Ark")]
    by_title = {row["title"]: movie_id for movie_id, row in mb.iterrows()}
    print("\nMatcher test (catalogue titles rewritten; share matched to the right film):")
    for name, share in results.items():
        print(f"   {name:24} {share:.1%}")
    print("Hand-written variants:")
    for title, year, catalogue_title in hand:
        expected = by_title.get(catalogue_title)
        got = match_title(title, year, index)
        status = "skip (not in catalogue)" if expected is None else ("ok" if got == expected else f"WRONG -> {got}")
        print(f"   {title!r:45} {status}")
    # Titles that do not exist must not be matched to anything
    fake = [("The Silent Lighthouse Keeper", 2003), ("Midnight in Copenhagen", 1998), ("Return of the Iron Orchard", 2011)]
    print("Invented titles (must be unmatched):",
          ["ok" if match_title(t, y, index) is None else "WRONG" for t, y in fake])


# ───────────────────────── prompt ─────────────────────────
def build_generate_prompt(user, data, version="g1"):
    """g1: the seeds as title, year and genres. g2: + director, cast and the seed profile."""
    detailed = version == "g2"
    seed_text = "\n".join("- " + lr.movie_line(m, data, detailed) for m in user["liked"])
    if version == "g1":
        user_prompt = f"""A new user told us they enjoyed these films:
{seed_text}

Recommend the {N_RECOMMEND} films this user is most likely to enjoy, best first.
Do not recommend any film from the list above.

Return JSON in exactly this form:
{{"recommendations": [{{"title": "film title", "year": 1999}}, ...]}}"""
    elif version == "g2":
        profile = lr.profile_from_seeds(user, data)
        user_prompt = f"""A new user told us they enjoyed these films:
{seed_text}

What these films have in common:
- Genres: {", ".join(profile["genres"])}
- Directors: {", ".join(profile["directors"])}
- Actors: {", ".join(profile["actors"])}
- Decades: {", ".join(f"{d}s" for d in profile["decades"])}

Recommend the {N_RECOMMEND} films this user is most likely to enjoy, best first. Consider the
genres, directors, actors and era they liked, and your own knowledge of films.
Do not recommend any film from the list above.

Return JSON in exactly this form:
{{"recommendations": [{{"title": "film title", "year": 1999}}, ...]}}"""
    else:
        raise ValueError(f"Unknown prompt version {version!r}")
    return [{"role": "system", "content": lr.SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}]


# ───────────────────────── parsing ─────────────────────────
def last_json_list(content, key):
    """The list under `key` in the LAST JSON object that has it (code fences, extra text and
    corrected second answers are skipped, as in lr.parse_ranking)."""
    found, decoder = None, json.JSONDecoder()
    for position, character in enumerate(content):
        if character == "{":
            try:
                candidate, _ = decoder.raw_decode(content[position:])
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, dict) and isinstance(candidate.get(key), list):
                found = candidate[key]
    return found


def parse_recommendations(content, user, data, index):
    """The model's answer -> one entry per recommended film with its category, in the model's order."""
    items = last_json_list(content, "recommendations")
    if items is None:
        return None
    seen, pool_ids, entries, taken = set(user["seen"]), data["pool_ids"], [], set()
    for item in items:
        if isinstance(item, dict):
            title, year = item.get("title"), parse_year(item.get("year"))
        else:                                                  # e.g. "Heat (1995)"
            title, year = re.sub(r"\(\s*\d{4}\s*\)", "", str(item)).strip(), parse_year(item)
        movie_id = match_title(title, year, index) if title else None
        if movie_id is None:
            category = "unmatched"
        elif movie_id in taken:
            category = "duplicate"
        elif movie_id in seen:
            category = "seen"
        elif movie_id not in pool_ids:
            category = "not_in_pool"
        else:
            category = "valid"
        if movie_id is not None:
            taken.add(movie_id)
        entries.append({"title": title, "year": year, "movie_id": movie_id, "category": category})
    return entries


# ───────────────────────── running (cached) ─────────────────────────
def cache_path(provider, model, version, n_seeds, think=None):
    safe = model.replace(":", "_").replace("/", "_")
    suffix = "" if think is None else f"_think-{think}"
    return FOLDER / f"answers_{provider}_{safe}_{version}_n{n_seeds}{suffix}.jsonl"


def run_generator(provider, model, users, data, version="g1", think=None):
    """Ask one model for recommendations for every user; answers are cached per model and setting,
    so a stopped run continues where it ended and a finished one is not paid for twice."""
    n_seeds = len(users[0]["liked"])
    path = cache_path(provider, model, version, n_seeds, think)
    cache = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            cache[record["user_id"]] = record
    answers = {}
    with open(path, "a", encoding="utf-8") as f:
        for user in tqdm(users, desc=f"{provider} / {model} [{version}] N={n_seeds}"):
            record = cache.get(user["user_id"])
            if record is None:
                result = lr.call_model(build_generate_prompt(user, data, version), provider, model, think)
                record = {"user_id": int(user["user_id"]), "provider": provider, "model": model,
                          "think": think, "version": version, "n_seeds": n_seeds,
                          "content": result["content"], "reasoning": result["reasoning"],
                          "tokens_in": result["tokens_in"], "tokens_out": result["tokens_out"],
                          "seconds": round(result["seconds"], 2)}
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
            answers[user["user_id"]] = record
    return answers


# ───────────────────────── evaluation ─────────────────────────
def popularity_list(user, data, k=K):
    """The k most popular unseen pool films, the same order as cs.recommend_popular
    (count descending, ties by movie_id)."""
    seen = set(user["seen"])
    return [m for m in data["popularity_order"] if m not in seen][:k]


def idea1_scores(model, version, n_seeds, split, user_ids):
    """Per-user NDCG of the same model reranking Hybrid E (idea 1) with the matching prompt
    (g1 <-> p1, g2 <-> p2), from any saved reranking result that covers all these users."""
    prompt = {"g1": "p1", "g2": "p2"}[version]
    label = model + ("" if prompt == "p1" else f" [{prompt}]") + ("" if n_seeds == 3 else f" N={n_seeds}")
    for path in sorted(lr.FOLDER.glob(f"compare_per_user_{split}_*users_{prompt}_n{n_seeds}.csv")):
        part = pd.read_csv(path)
        part = part[part["model"] == label].set_index("user_id")["llm_ndcg"]
        if set(user_ids) <= set(part.index):
            return part.loc[list(user_ids)]
    return None


def evaluate_model(provider, model, users, data, index, version, think=None):
    answers = run_generator(provider, model, users, data, version, think)
    rows = []
    for user in users:
        record = answers[user["user_id"]]
        entries = parse_recommendations(record["content"], user, data, index)
        relevant = user["relevant"]
        if entries is None:
            entries, unparsable = [], True
        else:
            unparsable = False
        # strict: the first 10 answers in order, anything not valid is a miss (placeholder id -1)
        strict = [e["movie_id"] if e["category"] == "valid" else -1 for e in entries[:K]]
        filled = [e["movie_id"] for e in entries if e["category"] == "valid"][:K]
        hybrid = lr.hybrid_e_candidates(user, data)[0][:K]
        popular = popularity_list(user, data)
        counts = {c: sum(e["category"] == c for e in entries) for c in CATEGORIES}
        rows.append({
            "model": model, "version": version, "n_seeds": len(user["liked"]), "user_id": user["user_id"],
            "ndcg_strict": bmt.ndcg_at_k(strict, relevant, K),
            "ndcg_filled": bmt.ndcg_at_k(filled, relevant, K),
            "precision_filled": len(set(filled) & relevant) / K,
            "recall_filled": len(set(filled) & relevant) / len(relevant),
            "ndcg_hybrid_E": bmt.ndcg_at_k(hybrid, relevant, K),
            "ndcg_popularity": bmt.ndcg_at_k(popular, relevant, K),
            "answers": len(entries), "valid_in_first_10": sum(e["category"] == "valid" for e in entries[:K]),
            "filled_length": len(filled), "unparsable": unparsable, **counts,
            "filled_ids": "|".join(map(str, filled)),
            "tokens_in": record["tokens_in"] or 0, "tokens_out": record["tokens_out"] or 0,
            "seconds": record["seconds"]})
    return pd.DataFrame(rows)


def breadth(per_user, data):
    """Catalogue coverage, mean log-popularity and long-tail share of the filled top-10 lists."""
    ids = [int(m) for s in per_user["filled_ids"] for m in str(s).split("|") if m and m != "nan"]
    if not ids:
        return {"coverage": 0.0, "mean_log_popularity": np.nan, "long_tail_share": np.nan}
    pv, po = data["pop_vector"], data["position_of"]
    return {"coverage": len(set(ids)) / len(data["pool_ids"]),
            "mean_log_popularity": float(np.mean([pv[po[m]] for m in ids])),
            "long_tail_share": float(np.mean([data["third_of"].get(m) in ("low", "mid") for m in ids]))}


def compare_generators(models, data, index, n_users=None, split="val", version="g1", n_seeds=3):
    """Run every (provider, model[, think]) on the same users; compare with Hybrid E, popularity
    and idea 1 (the same model as a reranker), Wilcoxon per user, Bonferroni over all tests."""
    users = cs.users_with_n_seeds(data[f"{split}_users"], n_seeds)
    if n_users is not None:
        users = users[:n_users]
    print(f"\nIdea 2 on {len(users)} {split} users (prompt {version}, N = {n_seeds})")
    per_user = pd.concat([evaluate_model(*entry[:2], users, data, index, version, *entry[2:])
                          for entry in models], ignore_index=True)

    summary, tests = [], []
    for model, part in per_user.groupby("model", sort=False):
        part = part.set_index("user_id")
        row = {"model": model, "version": version, "n_seeds": n_seeds, "users": len(part),
               "popularity": part["ndcg_popularity"].mean(), "hybrid_E": part["ndcg_hybrid_E"].mean(),
               "llm_strict": part["ndcg_strict"].mean(), "llm_filled": part["ndcg_filled"].mean(),
               "precision_filled": part["precision_filled"].mean(),
               "valid_share": part["valid"].sum() / max(part["answers"].sum(), 1),
               **{f"{c}_share": part[c].sum() / max(part["answers"].sum(), 1) for c in CATEGORIES[1:]},
               "short_lists": int((part["filled_length"] < K).sum()), "unparsable": int(part["unparsable"].sum()),
               "tokens_in": part["tokens_in"].mean(), "tokens_out": part["tokens_out"].mean(),
               "seconds": part["seconds"].mean(), **breadth(part.reset_index(), data)}
        idea1 = idea1_scores(model, version, n_seeds, split, part.index)
        comparisons = [("hybrid_E", part["ndcg_hybrid_E"]), ("popularity", part["ndcg_popularity"])]
        if idea1 is not None:
            row["idea1_rerank"] = idea1.mean()
            comparisons.append(("idea1_rerank", idea1))
        for name, baseline in comparisons:
            diff = part["ndcg_filled"] - baseline
            p = wilcoxon(diff).pvalue if diff.any() else 1.0
            tests.append({"model": model, "comparison": f"llm_filled vs {name}", "difference": diff.mean(),
                          "better": int((diff > 0).sum()), "worse": int((diff < 0).sum()), "p_value": p})
        summary.append(row)
    summary, tests = pd.DataFrame(summary), pd.DataFrame(tests)
    tests["p_bonferroni"] = (tests["p_value"] * len(tests)).clip(upper=1.0)

    tag = f"{split}_{len(users)}users_{version}_n{n_seeds}"
    per_user.to_csv(FOLDER / f"per_user_{tag}.csv", index=False)
    summary.to_csv(FOLDER / f"summary_{tag}.csv", index=False)
    tests.to_csv(FOLDER / f"tests_{tag}.csv", index=False)
    pd.set_option("display.width", 260)
    print(summary.round(4).to_string(index=False))
    print(tests.round(4).to_string(index=False))
    return per_user, summary, tests


def show_examples(provider, model, data, index, version="g1", n_seeds=3, n_users=5, split="val"):
    """Print the raw recommendations of a few users with their category, to check the matcher by eye."""
    users = cs.users_with_n_seeds(data[f"{split}_users"], n_seeds)[:n_users]
    answers = run_generator(provider, model, users, data, version)
    mb = data["movie_by_id"]
    for user in users:
        entries = parse_recommendations(answers[user["user_id"]]["content"], user, data, index) or []
        print(f"\nUser {user['user_id']} | seeds: " + "; ".join(mb.loc[m, "title"] for m in user["liked"]))
        for e in entries:
            hit = "HIT " if e["category"] == "valid" and e["movie_id"] in user["relevant"] else "    "
            matched = "" if e["movie_id"] is None else f" -> {mb.loc[e['movie_id'], 'title']} ({mb.loc[e['movie_id'], 'year']})"
            print(f"   {hit}{e['category']:12} {e['title']} ({e['year']}){matched}")


def prepare(data):
    """Extra lookups for idea 2, added to the data from lr.load_data()."""
    data["pool_ids"] = {int(data["movie_ids"][p]) for p in data["pool"]}
    pool_sorted = sorted(data["pool_ids"], key=lambda m: (-data["pop_vector"][data["position_of"][m]], m))
    data["popularity_order"] = pool_sorted               # log-count order = count order, ties by movie_id
    data["third_of"], _ = pe.popularity_thirds(data["movies"], data["pool_ids"])
    return data


if __name__ == "__main__":
    data = prepare(lr.load_data())
    index = build_title_index(data["movie_by_id"])

    # Check: popularity rebuilt here must equal experiment 06 (0.5598 on the test users)
    test3 = cs.users_with_n_seeds(data["test_users"], 3)
    print(f"Popularity NDCG@10 on the test users: "
          f"{np.mean([bmt.ndcg_at_k(popularity_list(u, data), u['relevant'], K) for u in test3]):.4f} (must be 0.5598)")

    test_matcher(data, index)
    if ARGS and ARGS[0] == "test":
        raise SystemExit

    # 1. Read a few raw answers by eye (local model, free; cached after the first run)
    # show_examples("ollama", "gemma4:e4b", data, index, version="g1", n_seeds=3, n_users=5)

    # 2. All 203 validation users, both prompts (g1 at N = 3, g2 at N = 10); n_users=20 for a pilot
    MODELS = [("ollama", "gemma4:e4b"), ("ollama_cloud", "deepseek-v4.1-flash")]
    for version, n_seeds in [("g1", 3), ("g2", 10)]:
        compare_generators(MODELS, data, index, n_users=None, split="val", version=version, n_seeds=n_seeds)
