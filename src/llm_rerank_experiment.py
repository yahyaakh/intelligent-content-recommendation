'''


Idea
Hybrid E (item-CF plus popularity) produces a candidate list.
DeepSeek reorders the top 30 using its knowledge of the films.
The final ranking mixes both orders, with the mixing weight tuned on validation users and reported on test users.

Setting: cold start, N = 3 seeds, the same users and split as docs 06 and 09.

'''
import json                      # cache file (jsonl) and parsing the model's JSON answer
import os                        # os.getenv("DEEPSEEK_API_KEY") etc.
import re                        # one_line_plot: split the overview into sentences
import sys                       # sys.argv: provider, model, mode from the command line
import time                      # timing each call, waiting before a retry
from pathlib import Path         # output/experiments/llm_rerank/...

import numpy as np
import pandas as pd
import requests                  # the HTTP call to Ollama / DeepSeek / OpenAI
from dotenv import load_dotenv   # reads the API keys from .env
from scipy.sparse import csr_matrix  # p5/p6: which warm users liked which films
from scipy.stats import wilcoxon # significance tests at the end
from tqdm import tqdm            # progress bar while calling the model

import build_movie_texts as bmt
import cold_start_experiment as cs

load_dotenv()

# ───────── settings ─────────
# Chosen from the command line, e.g.
#   python src/llm_rerank_experiment.py deepseek deepseek-chat pilot
#   python src/llm_rerank_experiment.py ollama gemma4:e4b full
PROVIDER   = sys.argv[1] if len(sys.argv) > 1 else "deepseek"       # "ollama" | "ollama_cloud" | "deepseek" | "openai"
MODEL_NAME = sys.argv[2] if len(sys.argv) > 2 else "deepseek-chat"  # check the provider's docs for the latest id
MODE       = sys.argv[3] if len(sys.argv) > 3 else "pilot"          # "pilot" | "full"

N_SEEDS     = 3
TOP_M       = 30                                    # candidates passed to the LLM
BETAS       = [round(b * 0.1, 1) for b in range(11)]  # 0.0, 0.1, ..., 1.0: weight of the LLM ranking
PILOT_USERS = 20                                    # pilot: the first 20 validation users only

# Only two request formats exist: Ollama's and OpenAI's. DeepSeek and ChatGPT both use the
# OpenAI one. A new provider = one new line here.
PROVIDERS = {
    "ollama":       {"url": "http://127.0.0.1:11434/api/chat",            "key_env": None,               "style": "ollama"},
    "ollama_cloud": {"url": "https://ollama.com/api/chat",                "key_env": "OLLAMA_API_KEY",   "style": "ollama"},
    "deepseek":     {"url": "https://api.deepseek.com/chat/completions",  "key_env": "DEEPSEEK_API_KEY", "style": "openai"},
    "openai":       {"url": "https://api.openai.com/v1/chat/completions", "key_env": "OPENAI_API_KEY",   "style": "openai"},
    "github": {"url": "https://models.github.ai/inference/chat/completions",
           "key_env": "GITHUB_MODELS_TOKEN", "style": "openai"},
}
if PROVIDER not in PROVIDERS:
    raise ValueError(f"Unknown provider {PROVIDER!r}; choose one of {list(PROVIDERS)}")


FOLDER = Path("output") / "experiments" / "llm_rerank"
FOLDER.mkdir(parents=True, exist_ok=True)
safe_name = MODEL_NAME.replace(":", "_").replace("/", "_")
CACHE = FOLDER / f"answers_{PROVIDER}_{safe_name}.jsonl"   # one cache per provider and model


# ───────── step 1: data (reuse, nothing new) ─────────
def load_data():
    movies = bmt.load_movies()
    ratings = bmt.prepare_ratings(movies)
    train, cold = cs.split_users(ratings)
    pool_ids = set(train["movie_id"])
    cold_users = cs.build_cold_user_data(cold, pool_ids, max(cs.SEED_COUNTS), cs.MIN_TEST_RELEVANT)
    val_users, test_users = cs.split_validation(cold_users, cs.VAL_FRACTION)

    movie_ids = movies["movie_id"].to_numpy()  # row i of every table = movie_ids[i]
    position_of = {movie_id: i for i, movie_id in enumerate(movie_ids)}
    pool = [position_of[m] for m in pool_ids]
    pool.sort()

    item_cf = cs.item_cf_similarity(train, movies, position_of)
    pop_vector = cs.popularity_vector(cs.popularity_scores(train), movies, position_of)
    alphas = pd.read_csv("output/experiments/cold_start/alphas.csv")
    rows = alphas[alphas["table"] == "E_item_cf"]
    alpha_E = dict(zip(rows["n"], rows["alpha"]))      # {1: 0.08, 3: 0.12, 5: 0.2, 10: 0.25}

    # Pool films from most to least popular, ties by movie_id: the order of cs.recommend_popular
    # (pop_vector is log(1 + count), which keeps the order of the counts)
    popularity_order = sorted((int(movie_ids[p]) for p in pool), key=lambda m: (-pop_vector[position_of[m]], m))

    # add to load_data(), and to the returned dictionary as "movie_by_id"
    movie_by_id = movies.set_index("movie_id")  # row lookup: movie_by_id.loc[movie_id]

    # v4 labels (experiment 08) for prompts p3 and p4. bmt.load_movies() holds the OLD v2 labels
    # in llm_themes / llm_mood, so the v4 ones are kept in their own columns.
    v4 = pd.read_csv("output/experiments/gemma4_e4b_v4/movies_enriched.csv").set_index("movie_id")
    movie_by_id["v4_themes"] = v4["llm_themes"].reindex(movie_by_id.index)
    movie_by_id["v4_moods"] = v4["llm_mood"].reindex(movie_by_id.index)

    # p5/p6 (doc 17, method B): which WARM users liked which films. "Liked" = rating >= the user's
    # own mean, the same definition as popularity and the answer key. Cold users are never used.
    # Rows = movie positions (same order as every other table), columns = warm users.
    user_column, warm_user_ids = pd.factorize(train["user_id"])
    is_liked = (train["rating"] >= train.groupby("user_id")["rating"].transform("mean")).to_numpy()
    liked_by = csr_matrix((np.ones(is_liked.sum(), np.float32),
                           (train["movie_id"].map(position_of).to_numpy()[is_liked], user_column[is_liked])),
                          shape=(len(movie_ids), len(warm_user_ids)))
    like_counts = np.asarray(liked_by.sum(axis=1)).ravel()   # warm users who liked each film

    return {
        "movies": movies,
        "movie_ids": movie_ids,
        "position_of": position_of,
        "pool": pool,
        "val_users": val_users,
        "test_users": test_users,
        "item_cf": item_cf,
        "pop_vector": pop_vector,
        "alpha_E": alpha_E,
        "movie_by_id": movie_by_id,  # <- add this line
        "popularity_order": popularity_order,
        "liked_by": liked_by,              # p5/p6: movies x warm users, 1 = liked
        "like_counts": like_counts,        # p5/p6: warm users who liked each film
        "n_warm_users": len(warm_user_ids),
        "warm_user_ids": warm_user_ids,    # p8: column j of liked_by = warm user warm_user_ids[j]
        "train": train,                    # p7/p8: warm ratings, to build the example users
        "pool_ids": pool_ids,
    }

PLOT_WORDS = 25            # p4: the plot line is the overview's first sentence, cut to this many words
KEYWORDS_PER_FILM = 5      # p4: the first TMDB keywords that say something about the content
UNINFORMATIVE_KEYWORDS = {"remake", "sequel", "based on novel or book", "duringcreditsstinger",
                          "aftercreditsstinger", "woman director", "based on comic", "independent film"}


def split_field(value):
    return [] if not isinstance(value, str) else [part.strip() for part in value.split("|") if part.strip()]


def one_line_plot(overview):
    if not isinstance(overview, str) or not overview.strip():
        return None
    # a sentence ends with ". " after a lowercase word, so abbreviations such as "L.A." do not cut it
    first_sentence = re.split(r"(?<=[a-z]{2})[.!?]\s", overview.strip())[0].rstrip(".")
    words = first_sentence.split()
    return " ".join(words[:PLOT_WORDS]) + ("…" if len(words) > PLOT_WORDS else "")


def movie_line(movie_id, data, detailed=False, labels=False, extras=False):
    """One line per film: what a person would know about it.
    detailed=True (p2 and up) adds the director and the cast;
    labels=True   (p3 and up) adds the v4 themes and moods from experiment 08;
    extras=True   (p4)        adds runtime, TMDB keywords and a one-line plot."""
    movie = data["movie_by_id"].loc[movie_id]
    genres = movie["genres"].replace("|", ", ")
    line = f"{movie['title']} ({movie['year']}) – {genres}"
    # example: "Heat (1995) – Action, Crime, Thriller"
    if not detailed:
        return line
    line += f" – directed by {movie['director']}"
    if isinstance(movie["cast"], str):                     # 9 films have no cast
        line += f" – starring {movie['cast'].replace('|', ', ')}"
    # example: "Heat (1995) – Action, Crime, Thriller – directed by Michael Mann – starring Al Pacino, Robert De Niro, Val Kilmer"
    if labels:
        themes, moods = split_field(movie["v4_themes"]), split_field(movie["v4_moods"])
        if themes:
            line += f" – themes: {', '.join(themes)}"
        if moods:
            line += f" – mood: {', '.join(moods)}"
    if extras:
        if pd.notna(movie["runtime"]) and movie["runtime"] > 0:
            line += f" – {int(movie['runtime'])} min"
        keywords = [k for k in split_field(movie["keywords"]) if k.lower() not in UNINFORMATIVE_KEYWORDS]
        if keywords:
            line += f" – keywords: {', '.join(keywords[:KEYWORDS_PER_FILM])}"
        plot = one_line_plot(movie["overview"])
        if plot:
            line += f" – plot: {plot}"
    return line


def get_api_key(provider_name):
    key_env = PROVIDERS[provider_name]["key_env"]
    api_key = os.getenv(key_env) if key_env else None     # local Ollama needs no key
    if key_env and not api_key:
        raise ValueError(f"Missing {key_env} in .env")
    return api_key





def percent(share):
    """0.4137 -> "41%"; tiny shares stay visible as "<1%"."""
    return "0%" if share == 0 else "<1%" if share < 0.005 else f"{share:.0%}"


def cf_evidence(user, candidate_ids, data, with_rank=False):
    """p5/p6 (doc 17, method B): the collaborative data the LLM normally lacks, one text per candidate.
    Computed from the warm users only, never from the cold user's own hidden ratings.
      co-visitation  P(candidate | seed) = warm users who liked both / warm users who liked the seed,
                     shown as the mean over the seeds and the highest seed (with its title)
      popularity     share of all warm users who liked the candidate
      with_rank      p6 only: the candidate's position in Hybrid E's list (1 = Hybrid E's best)"""
    position_of, liked_by = data["position_of"], data["liked_by"]
    seeds = [position_of[m] for m in user["liked"]]
    cands = [position_of[m] for m in candidate_ids]
    both = (liked_by[seeds] @ liked_by[cands].T).toarray()          # rows: seeds, columns: candidates
    fans = data["like_counts"][seeds][:, None]
    covis = np.divide(both, fans, out=np.zeros_like(both), where=fans > 0)
    hybrid_rank = {m: i + 1 for i, m in enumerate(hybrid_e_candidates(user, data)[0])} if with_rank else {}

    texts = {}
    for j, movie_id in enumerate(candidate_ids):
        best_seed = user["liked"][int(np.argmax(covis[:, j]))]
        text = (f"liked by {percent(covis[:, j].mean())} of the fans of the user's films "
                f"(most: {percent(covis[:, j].max())} of the fans of {data['movie_by_id'].loc[best_seed, 'title']}); "
                f"liked by {percent(data['like_counts'][cands[j]] / data['n_warm_users'])} of all users")
        if with_rank:
            rank = hybrid_rank.get(movie_id)
            text += (f"; our system's rank: {rank} of {TOP_M}" if rank else
                     f"; not in our system's top {TOP_M}")
        texts[movie_id] = text
    # example: "liked by 31% of the fans of the user's films (most: 58% of the fans of Heat); liked by 22% of all users"
    return texts


FEW_SHOT_EXAMPLES = 3       # p7/p8: worked examples (warm users) per prompt
FEW_SHOT_CANDIDATES = 10    # p7/p8: candidates shown per example, a fixed random 10 of its top 30


def warm_examples(data, n_seeds):
    """p7/p8 (doc 17, method C): warm users built exactly like the cold users (seeds = their first
    liked films in time order; "liked later" = their other liked films). Built once, on first use.
    Warm users only: a cold user's own hidden ratings never appear in any prompt."""
    key = f"warm_examples_n{n_seeds}"
    if key not in data:
        print("Building the warm example users (once):")
        built = cs.build_cold_user_data(data["train"], data["pool_ids"], max(cs.SEED_COUNTS), cs.MIN_TEST_RELEVANT)
        data[key] = {int(u["user_id"]): u for u in cs.users_with_n_seeds(built, n_seeds)}
    return data[key]


def example_user_ids(user, data, version):
    """p7: the same FEW_SHOT_EXAMPLES random warm users for everyone (fixed, seed 42).
    p8: the warm users most like this user: cosine between the seeds and the warm user's liked films,
        = seeds they liked / sqrt(films they liked), ties by user id. (A raw count of shared seeds
        picks heavy raters, who liked 83% of their candidates vs 54% for new users; cosine: 43%.)"""
    examples = warm_examples(data, len(user["liked"]))
    if version == "p7":
        rng = np.random.default_rng(42)
        return [int(i) for i in rng.choice(sorted(examples), FEW_SHOT_EXAMPLES, replace=False)]
    seeds = [data["position_of"][m] for m in user["liked"]]
    shared = np.asarray(data["liked_by"][seeds].sum(axis=0)).ravel()      # seeds liked, per warm user
    n_liked = np.asarray(data["liked_by"].sum(axis=0)).ravel()           # films liked, per warm user
    shared = shared / np.sqrt(np.maximum(n_liked, 1))                     # cosine (up to a constant)
    ids = np.asarray(data["warm_user_ids"])
    ranked = sorted(range(len(ids)), key=lambda j: (-shared[j], ids[j]))
    return [int(ids[j]) for j in ranked if int(ids[j]) in examples][:FEW_SHOT_EXAMPLES]


def example_text(example, data, number):
    """One worked example: what the warm user enjoyed first, and FEW_SHOT_CANDIDATES films from
    their own Hybrid E top 30 (a fixed random choice, not picked by outcome), each with the same
    data line as the real candidates and whether they liked it later."""
    top_ids, _ = hybrid_e_candidates(example, data)
    shown = [int(m) for m in np.random.default_rng(example["user_id"]).choice(top_ids, FEW_SHOT_CANDIDATES, replace=False)]
    evidence = cf_evidence(example, shown, data)
    movie = data["movie_by_id"]
    seeds = "; ".join(f"{movie.loc[m, 'title']} ({movie.loc[m, 'year']})" for m in example["liked"])
    lines = "\n".join(f"- {movie_line(m, data)} [data: {evidence[m]}] -> "
                      + ("LIKED" if m in example["relevant"] else "not liked or not watched")
                      for m in shown)
    return f"Example {number}. They enjoyed: {seeds}\nSome of their candidates, and what happened later:\n{lines}"


SYSTEM_PROMPT = "You are a movie recommendation expert. You always answer with JSON only."
def build_prompt(user, top_ids, data, version="p1", shuffle=0):
    """The prompt for one user, plus how to translate the model's numbers back to movie ids.
    p1: title, year and genres. p2: also director and cast, plus a profile of the seed films.
    p3: p2 + the v4 themes and moods (experiment 08), also in the profile.
    p4: p3 + runtime, TMDB keywords and a one-line plot per film.
    p5: p2 + collaborative evidence per candidate (co-visitation and popularity, doc 17 method B).
    p6: p5 + the candidate's Hybrid E rank (does the LLM just copy Hybrid E when it sees it?).
    p7: p5 + 3 worked examples: fixed random warm users, their candidates and what they liked (method C).
    p8: p5 + 3 worked examples from the warm users most similar to this user (method C)."""
    detailed = version in ("p2", "p3", "p4", "p5", "p6", "p7", "p8")
    labels = version in ("p3", "p4")
    extras = version == "p4"

    # 1. Shuffle the candidates, the same way on every run for this user,
    #    so the model cannot copy Hybrid E's order (position bias); p1 and p2 get the same order
    # shuffle 0: seed = user id (the order used in every run so far); shuffle k: another fixed order
    rng = np.random.default_rng(user["user_id"] if not shuffle else [user["user_id"], shuffle])
    shuffled = rng.permutation(top_ids)

    # 2. Number the candidates 1..TOP_M; the model answers with these numbers
    number_to_id = {i + 1: int(movie_id) for i, movie_id in enumerate(shuffled)}

    seed_text = "\n".join("- " + movie_line(m, data, detailed, labels, extras) for m in user["liked"])
    candidate_text = "\n".join(f"{number}. {movie_line(movie_id, data, detailed, labels, extras)}"
                               for number, movie_id in number_to_id.items())
    if version in ("p5", "p6", "p7", "p8"):      # one evidence line under each candidate, same shuffled order
        evidence = cf_evidence(user, list(number_to_id.values()), data, with_rank=version == "p6")
        candidate_text = "\n".join(f"{number}. {movie_line(movie_id, data, detailed)}\n   [data: {evidence[movie_id]}]"
                                   for number, movie_id in number_to_id.items())

    if version == "p1":
        # Unchanged, indentation included, so the cached p1 answers stay valid
        user_prompt = f"""A new user told us they enjoyed these films:
    {seed_text}

    Rank ALL {TOP_M} of the following films from the one this user is most likely to enjoy
    to the one they are least likely to enjoy:
    {candidate_text}

    Use every number from 1 to {TOP_M} exactly once.
    Return JSON in exactly this form: {{"ranking": [numbers, best first]}}"""

    elif version == "p2":
        profile = profile_from_seeds(user, data)
        user_prompt = f"""A new user told us they enjoyed these films:
{seed_text}

What these films have in common:
- Genres: {", ".join(profile["genres"])}
- Directors: {", ".join(profile["directors"])}
- Actors: {", ".join(profile["actors"])}
- Decades: {", ".join(f"{d}s" for d in profile["decades"])}

Rank ALL {TOP_M} of the following films from the one this user is most likely to enjoy
to the one they are least likely to enjoy. Consider the genres, directors, actors and era
they liked, and your own knowledge of the films:
{candidate_text}

Use every number from 1 to {TOP_M} exactly once.
Return JSON in exactly this form: {{"ranking": [numbers, best first]}}"""

    elif version in ("p3", "p4"):
        profile = profile_from_seeds(user, data)
        consider = ("the genres, directors, actors, themes, moods and era they liked"
                    if version == "p3" else
                    "the genres, directors, actors, themes, moods, era, plots and keywords of the films they liked")
        user_prompt = f"""A new user told us they enjoyed these films:
{seed_text}

What these films have in common:
- Genres: {", ".join(profile["genres"])}
- Directors: {", ".join(profile["directors"])}
- Actors: {", ".join(profile["actors"])}
- Themes: {", ".join(profile["themes"])}
- Moods: {", ".join(profile["moods"])}
- Decades: {", ".join(f"{d}s" for d in profile["decades"])}

Rank ALL {TOP_M} of the following films from the one this user is most likely to enjoy
to the one they are least likely to enjoy. Consider {consider},
and your own knowledge of the films:
{candidate_text}

Use every number from 1 to {TOP_M} exactly once.
Return JSON in exactly this form: {{"ranking": [numbers, best first]}}"""

    elif version in ("p5", "p6", "p7", "p8"):
        profile = profile_from_seeds(user, data)
        rank_note = (f"\n- our system's rank: where our recommendation system ranks the film for this user "
                     f"(1 = its best guess)" if version == "p6" else "")
        examples_block = ""                    # p5/p6: empty, so their text (and cached answers) stay the same
        if version in ("p7", "p8"):
            examples = warm_examples(data, len(user["liked"]))
            kind = ("other users of our site" if version == "p7" else
                    "other users of our site whose taste overlaps with this user's films")
            examples_block = (f"\n\nTo show how this data relates to what people really liked, here are "
                              f"{FEW_SHOT_EXAMPLES} {kind}:\n\n"
                              + "\n\n".join(example_text(examples[i], data, k + 1)
                                            for k, i in enumerate(example_user_ids(user, data, version))))
        user_prompt = f"""A new user told us they enjoyed these films:
{seed_text}

What these films have in common:
- Genres: {", ".join(profile["genres"])}
- Directors: {", ".join(profile["directors"])}
- Actors: {", ".join(profile["actors"])}
- Decades: {", ".join(f"{d}s" for d in profile["decades"])}

Under each candidate film is data from {data["n_warm_users"]:,} other users of our site:
- "liked by X% of the fans of the user's films": of the people who liked one of this user's films,
  the share who also liked the candidate (averaged over the user's films, and for the film where it is highest)
- "liked by X% of all users": how widely liked the candidate is in general{rank_note}{examples_block}

Rank ALL {TOP_M} of the following films from the one this user is most likely to enjoy
to the one they are least likely to enjoy. Consider this data together with the genres, directors,
actors and era they liked, and your own knowledge of the films:
{candidate_text}

Use every number from 1 to {TOP_M} exactly once.
Return JSON in exactly this form: {{"ranking": [numbers, best first]}}"""

    else:
        raise ValueError(f"Unknown prompt version {version!r}")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    return messages, number_to_id

def hybrid_e_candidates(user, data):
    """The TOP_M best candidates of Hybrid E for one user, with their Hybrid E scores.
    The LLM only reorders these, so Hybrid E is the starting point of every method."""
    position_of = data["position_of"]

    # 1. Candidates: every pool movie the user has not seen (all 10 seeds count as seen)
    seen = {position_of[m] for m in user["seen"]}
    cand = np.array([p for p in data["pool"] if p not in seen])

    # 2. Personal part: mean item-CF similarity of each candidate to the 3 seeds
    liked = [position_of[m] for m in user["liked"]]
    block = data["item_cf"][np.ix_(liked, cand)]        # rows: seeds, columns: candidates
    personal = cs.rescale(block.mean(axis=0))

    # 3. Popularity part
    popularity = cs.rescale(data["pop_vector"][cand])

    # 4. Hybrid E score (same formula as doc 06)
    alpha = data["alpha_E"][len(user["liked"])]  # the alpha chosen for this number of seeds
    score = alpha * personal + (1 - alpha) * popularity

    # 5. The best TOP_M, best first (stable sort, as in cold_start_experiment)
    top = np.argsort(-score, kind="stable")[:TOP_M]
    top_ids = [int(m) for m in data["movie_ids"][cand[top]]]   # back to movie ids, plain ints
    return top_ids, score[top]


def call_model(messages, provider_name, model_name, think=None, retries=3):
    provider = PROVIDERS[provider_name]
    api_key = get_api_key(provider_name)
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    if think is None:   # default: gpt-oss cannot switch reasoning off, "low" keeps it short
        think = "low" if model_name.startswith("gpt-oss") else False
    if provider["style"] == "ollama":
        payload = {"model": model_name, "messages": messages, "stream": False,
                   "format": "json",
                   "think": think,
                   # num_ctx: local Ollama's default context window can be 4,096 tokens, and longer
                   # prompts (p4 is about 4,500) are then silently cut; 16,384 covers every prompt
                   "options": {"temperature": 0, "seed": 42, "num_ctx": 16384}}
    else:   # "openai": DeepSeek, ChatGPT
        payload = {"model": model_name, "messages": messages, "temperature": 0,
                   "response_format": {"type": "json_object"}}


    for attempt in range(retries + 1):
        start = time.perf_counter()
        try:
            response = requests.post(provider["url"], headers=headers, json=payload, timeout=300)
        except (requests.ConnectionError, requests.Timeout):
            time.sleep(5 * (attempt + 1))  # network problem -> wait and retry
            continue
        if response.status_code == 429 or response.status_code >= 500:
            time.sleep(10 * (attempt + 1))  # rate limit / server error -> wait and retry
            continue
        if response.status_code == 400 and "think" in payload and "think" in response.text.lower():
            # some models reject the think setting: retry once with the model's own default
            print(f"  {model_name} rejected think={think!r} ({response.text[:120]}); retrying without it")
            payload.pop("think")
            continue
        response.raise_for_status()  # 400 / 401 / 404: a real error, stop here
        data = response.json()

        if provider["style"] == "ollama":
            content = data["message"]["content"]
            reasoning = data["message"].get("thinking")
            tokens_in, tokens_out = data.get("prompt_eval_count"), data.get("eval_count")
        else:
            message = data["choices"][0]["message"]
            content = message["content"]
            reasoning = message.get("reasoning_content")
            tokens_in, tokens_out = data["usage"]["prompt_tokens"], data["usage"]["completion_tokens"]

        return {"content": content, "reasoning": reasoning, "tokens_in": tokens_in,
                "tokens_out": tokens_out, "seconds": time.perf_counter() - start}

    raise RuntimeError(f"Model call failed after {retries + 1} attempts")


def parse_ranking(content, number_to_id, top_ids):
    """The model's answer -> all TOP_M movie ids in the model's order, plus what was repaired.
    Films the model left out are added at the end in Hybrid E order."""
    issues = []
    # Take the LAST JSON object with a "ranking": hosted models wrap JSON in ```json ... ```,
    # and some send a corrected second answer after the first one
    numbers, decoder = None, json.JSONDecoder()
    for position, character in enumerate(content):
        if character == "{":
            try:
                candidate, _ = decoder.raw_decode(content[position:])
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, dict) and isinstance(candidate.get("ranking"), list):
                numbers = candidate["ranking"]
    if numbers is None:
        return list(top_ids), ["unparsable"]

    order = []
    for n in numbers:
        movie_id = number_to_id.get(int(n)) if str(n).strip().isdigit() else None
        if movie_id is None:
            issues.append(f"invalid {n!r}")
        elif movie_id in order:
            issues.append(f"duplicate {n}")
        else:
            order.append(movie_id)

    missing = [m for m in top_ids if m not in order]
    if missing:
        issues.append(f"{len(missing)} missing")
    return order + missing, issues

def popularity_candidates(user, data):
    """The TOP_M most popular films the user has not seen (the same list for every user, minus
    their seeds), with their rescaled popularity. No personal information at all, so any
    personalisation in a reranking of these comes from the LLM."""
    seen = set(user["seen"])
    top_ids = [m for m in data["popularity_order"] if m not in seen][:TOP_M]
    pop = np.array([data["pop_vector"][data["position_of"][m]] for m in top_ids])
    return top_ids, pop


def uknn_candidates(user, data):
    """The TOP_M best unseen pool films of the standard user-kNN (doc 20: similarity-weighted sum of the
    centred ratings of the k most similar warm users, k chosen on validation), with their scores.
    The same scorer and tie order as the user-kNN row of option 11, so its top 10 gives the same NDCG.
    The warm-user matrix is built once and kept in data["uknn"]."""
    import cold_start_baselines as cbl
    if "uknn" not in data:
        pool_ids = np.array([int(data["movie_ids"][p]) for p in sorted(data["pool"])])
        column = {m: i for i, m in enumerate(pool_ids)}
        data["uknn"] = (cbl.UserKNN(data["train"], column, len(pool_ids)), pool_ids, column)
    knn, pool_ids, column = data["uknn"]
    seed_cols = [column[m] for m in user["liked"]]
    if len(seed_cols) not in cbl.KNN_NEIGHBOURS:
        raise ValueError(f"user-kNN's k was only chosen for {list(cbl.KNN_NEIGHBOURS)} seeds")
    scores = knn.knn_scores(seed_cols, cbl.KNN_NEIGHBOURS[len(seed_cols)])
    scores[[column[m] for m in user["seen"]]] = -np.inf          # seen films are never recommended
    top = np.argsort(-scores, kind="stable")[:TOP_M]
    return [int(m) for m in pool_ids[top]], scores[top]


CANDIDATE_SOURCES = {   # name -> (function, cache-file suffix, label); "hybrid" keeps the old names
    "hybrid":     (hybrid_e_candidates,   "",       ""),
    "popularity": (popularity_candidates, "_cpop",  " {popularity candidates}"),
    "uknn":       (uknn_candidates,       "_cuknn", " {user-kNN candidates}"),
}


def get_candidates(user, data, candidates="hybrid"):
    if candidates not in CANDIDATE_SOURCES:
        raise ValueError(f"Unknown candidate source {candidates!r}")
    return CANDIDATE_SOURCES[candidates][0](user, data)


def run_name(provider_name, model_name, n_seeds, think=None, version="p1", candidates="hybrid", shuffle=0):
    """Cache file and label of one run. Every non-default setting (think, version, number of seeds,
    candidates, shuffle) gets its own cache file and label, so answers to different prompts or
    settings never mix; the existing Hybrid E, p1, N = 3, first-shuffle caches keep their names."""
    safe_name = model_name.replace(":", "_").replace("/", "_")
    suffix, label = "", model_name
    if think is not None:
        suffix += f"_think-{think}"
        label += f" (think {think})"
    if version != "p1":
        suffix += f"_{version}"
        label += f" [{version}]"
    if n_seeds != 3:
        suffix += f"_n{n_seeds}"
        label += f" N={n_seeds}"
    suffix += CANDIDATE_SOURCES[candidates][1]
    label += CANDIDATE_SOURCES[candidates][2]
    if shuffle:
        suffix += f"_s{shuffle}"
        label += f" <shuffle {shuffle}>"
    return FOLDER / f"answers_{provider_name}_{safe_name}{suffix}.jsonl", label


def run_model(provider_name, model_name, users, data, think=None, version="p1", candidates="hybrid", shuffle=0):
    """Rerank the users' candidates with one model. Answers are cached per model,
    so a stopped run continues where it ended and a finished one is not paid for twice.
    think:      reasoning level (e.g. "high"); None = the default of call_model.
    version:    prompt version ("p1" ... "p8").
    candidates: "hybrid" = Hybrid E's top 30 (idea 1); "popularity" = the 30 most popular
                unseen films (idea 3: can the LLM personalise a non-personal list?).
    shuffle:    0 = the original candidate order shown to the model; 1, 2, ... = other random
                orders of the same candidates (to measure and average out position bias)."""
    n_seeds = len(users[0]["liked"])
    if candidates == "uknn" and version not in ("p1", "p2", "p3", "p4", "p5"):
        # p6 shows Hybrid E's rank, p7/p8 show Hybrid E candidate lists as examples
        raise ValueError(f"prompt {version} refers to Hybrid E; with user-kNN candidates use p1-p5")
    cache_path, label = run_name(provider_name, model_name, n_seeds, think, version, candidates, shuffle)

    cache = {}
    if cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            cache[record["user_id"]] = record

    rows = []
    with open(cache_path, "a", encoding="utf-8") as f:
        for user in tqdm(users, desc=f"{provider_name} / {label}"):
            top_ids, top_scores = get_candidates(user, data, candidates)
            messages, number_to_id = build_prompt(user, top_ids, data, version, shuffle)

            record = cache.get(user["user_id"])
            if record is None:
                result = call_model(messages, provider_name, model_name, think)
                record = {"user_id": int(user["user_id"]), "provider": provider_name,
                          "model": model_name, "think": think, "version": version, "shuffle": shuffle,
                          "n_seeds": n_seeds, "content": result["content"],
                          "reasoning": result["reasoning"],
                          "tokens_in": result["tokens_in"], "tokens_out": result["tokens_out"],
                          "seconds": round(result["seconds"], 2)}
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()

            llm_order, issues = parse_ranking(record["content"], number_to_id, top_ids)
            # Baselines for every user: Hybrid E's and popularity's own top 10
            hybrid_top = top_ids if candidates == "hybrid" else hybrid_e_candidates(user, data)[0]
            popular_top = top_ids if candidates == "popularity" else popularity_candidates(user, data)[0]
            hybrid = bmt.user_metrics(hybrid_top[:bmt.K], user["relevant"], bmt.K)
            popular = bmt.user_metrics(popular_top[:bmt.K], user["relevant"], bmt.K)
            own = bmt.user_metrics(top_ids[:bmt.K], user["relevant"], bmt.K)   # the candidate list's own order
            llm = bmt.user_metrics(llm_order[:bmt.K], user["relevant"], bmt.K)
            rows.append({"model": label, "provider": provider_name,
                         "version": version, "n_seeds": n_seeds, "candidates": candidates,
                         "user_id": user["user_id"],
                         "hybrid_ndcg": hybrid["ndcg"], "popularity_ndcg": popular["ndcg"],
                         "candidates_ndcg": own["ndcg"], "llm_ndcg": llm["ndcg"],
                         "llm_precision": llm["precision"], "llm_recall": llm["recall"],
                         "issues": len(issues), "unparsable": "unparsable" in issues,
                         "tokens_in": record["tokens_in"] or 0, "tokens_out": record["tokens_out"] or 0,
                         "seconds": record["seconds"]})
    return pd.DataFrame(rows)



def compare_models(models, data, n_users=None, split="val", version="p1", n_seeds=N_SEEDS, candidates="hybrid", shuffle=0):
    """Run every (provider, model) on the same users and compare them with Hybrid E and popularity.
    models:     e.g. [("ollama", "gemma4:e4b"), ("ollama_cloud", "gpt-oss:120b-cloud", "high")];
                an optional third value is the reasoning level
    n_users:    the first n users of the split (None = all)
    split:      "val" while choosing models and prompts; "test" only for the final run
    version:    prompt version, "p1" ... "p8" (p5/p6: with collaborative evidence; p7/p8: + few-shot examples)
    n_seeds:    number of seed films per user (3, 5 or 10)
    candidates: "hybrid" (Hybrid E's top 30, idea 1), "popularity" (the 30 most popular, idea 3) or
                "uknn" (user-kNN's top 30, idea 4; compared with user-kNN's own order)"""
    users = cs.users_with_n_seeds(data[f"{split}_users"], n_seeds)
    if n_users is not None:
        users = users[:n_users]
    print(f"\nComparing {len(models)} models on {len(users)} {split} users "
          f"(prompt {version}, N = {n_seeds}, candidates: {candidates})")

    per_user = pd.concat([run_model(*entry[:2], users, data, *entry[2:], version=version, candidates=candidates,
                                    shuffle=shuffle)
                          for entry in models], ignore_index=True)

    summary = []
    for model, part in per_user.groupby("model", sort=False):
        diff = part["llm_ndcg"] - part["hybrid_ndcg"]
        p = wilcoxon(diff).pvalue if diff.any() else 1.0
        diff_pop = part["llm_ndcg"] - part["popularity_ndcg"]
        p_pop = wilcoxon(diff_pop).pvalue if diff_pop.any() else 1.0
        extra = {}
        if candidates == "uknn":       # the question here: does the LLM improve user-kNN's own order?
            diff_own = part["llm_ndcg"] - part["candidates_ndcg"]
            extra = {"user_knn": part["candidates_ndcg"].mean(), "diff_vs_user_knn": diff_own.mean(),
                     "better_than_knn": int((diff_own > 0).sum()), "worse_than_knn": int((diff_own < 0).sum()),
                     "p_vs_user_knn": wilcoxon(diff_own).pvalue if diff_own.any() else 1.0}
        summary.append({"model": model, "version": version, "n_seeds": n_seeds, "candidates": candidates,
                        "users": len(part), "popularity": part["popularity_ndcg"].mean(),
                        "hybrid_E": part["hybrid_ndcg"].mean(), **extra, "llm_only": part["llm_ndcg"].mean(),
                        "difference": diff.mean(),
                        "better": int((diff > 0).sum()), "worse": int((diff < 0).sum()),
                        "p_value": p,
                        "diff_vs_popularity": diff_pop.mean(), "p_vs_popularity": p_pop,
                        "answers_with_issues": int((part["issues"] > 0).sum()),
                        "unparsable": int(part["unparsable"].sum()),
                        "tokens_in": part["tokens_in"].mean(), "tokens_out": part["tokens_out"].mean(),
                        "seconds": part["seconds"].mean()})
    summary = pd.DataFrame(summary)
    summary["p_bonferroni"] = (summary["p_value"] * len(summary)).clip(upper=1.0)
    summary["p_vs_popularity_bonferroni"] = (summary["p_vs_popularity"] * len(summary)).clip(upper=1.0)
    if candidates == "uknn":
        summary["p_vs_user_knn_bonferroni"] = (summary["p_vs_user_knn"] * len(summary)).clip(upper=1.0)

    tag = (f"{split}_{len(users)}users_{version}_n{n_seeds}" + CANDIDATE_SOURCES[candidates][1]
           + ("" if not shuffle else f"_s{shuffle}"))
    per_user.to_csv(FOLDER / f"compare_per_user_{tag}.csv", index=False)
    summary.to_csv(FOLDER / f"compare_summary_{tag}.csv", index=False)
    pd.set_option("display.width", 250)
    print(summary.round(4).to_string(index=False))
    return per_user, summary


def check_candidates(data):
    """Free checks before any model call: the first 10 must be Hybrid E (0.5970), and
    oracle@TOP_M is the best possible NDCG@10 that any reranking of the candidates can reach."""
    users = cs.users_with_n_seeds(data["test_users"], N_SEEDS)
    hybrid_ndcg, oracle_ndcg = [], []
    for user in users:
        top_ids, top_scores = hybrid_e_candidates(user, data)
        hybrid_ndcg.append(bmt.user_metrics(top_ids[:10], user["relevant"], bmt.K)["ndcg"])

        hits = [m for m in top_ids if m in user["relevant"]]
        others = [m for m in top_ids if m not in user["relevant"]]
        oracle_ndcg.append(bmt.user_metrics((hits + others)[:10], user["relevant"], bmt.K)["ndcg"])

    print(f"Hybrid E NDCG@10: {np.mean(hybrid_ndcg):.4f}  (must be 0.5970)")
    print(f"Oracle@{TOP_M}:     {np.mean(oracle_ndcg):.4f}  (the ceiling for any reranking)")

from collections import Counter




def keep_repeated(counter, top):
    """The favourites in a Counter: names that occur in 2+ seed films (a real preference),
    or, if none repeats, simply the `top` most common names."""
    repeated = [name for name, count in counter.most_common(top) if count >= 2]
    return repeated or [name for name, _ in counter.most_common(top)]


def profile_from_seeds(user, data):
    """Favourite genres, directors and actors computed ONLY from the user's seed films
    (user["liked"]). Never from their other ratings: those are the answer key."""

    seeds = [ data["movie_by_id"].loc[m] for m in user["liked"] ]

    genre_counts = Counter()
    for seed in seeds:
        for g in seed["genres"].split("|"):
            genre_counts[g.strip()] += 1


    director_counts = Counter()
    for seed in seeds:
        for d in seed["director"].split("|"):
            director_counts[d.strip()] += 1

    # of every actor in seed["cast"]     (split on "|", first 5 per film)
    actor_counts = Counter()
    for seed in seeds:
        if not isinstance(seed["cast"], str):  # 9 films have no cast (NaN)
            continue
        for a in seed["cast"].split("|"):
            actor_counts[a.strip()] += 1

    decades = sorted({(int(seed["year"]) // 10) * 10 for seed in seeds})

    # v4 themes and moods of the seeds (used from prompt p3 on)
    theme_counts = Counter(t for seed in seeds for t in split_field(seed["v4_themes"]))
    mood_counts = Counter(m for seed in seeds for m in split_field(seed["v4_moods"]))

    return {
        "genres": keep_repeated(genre_counts, 5),
        "directors": keep_repeated(director_counts, 3),
        "actors": keep_repeated(actor_counts, 5),
        "decades": decades,
        "themes": keep_repeated(theme_counts, 5),
        "moods": keep_repeated(mood_counts, 3),
    }




if __name__ == "__main__":
    data = load_data()
    check_candidates(data)
    # Done before (doc 17): method B (p5/p6) and method C (p7/p8) with DeepSeek V4.1 Flash; answers cached.

    # ── Method E (doc 17): frontier models, idea 1 (Hybrid E candidates), N = 10, validation ──
    # Two large models not tested so far, each with p2 (no CF data) and p5 (CF data).
    # DeepSeek V4.1 Flash is included from its cache (no new calls) as the reference.
    NEW = [("ollama_cloud", "kimi-k3")]         # new family (Moonshot), not tested before
           # ("ollama_cloud", "glm-5.3")]          # full GLM flagship (only 5.2 and 5.3-flash tested before)
    REFERENCE = [("ollama_cloud", "deepseek-v4.1-flash")]
    SETTING = dict(split="val", n_seeds=10, candidates="hybrid")
    PILOT = False            # True: first 20 users only; set to False for all 203

    n_users = 20 if PILOT else None
    ndcg = {}               # (model, version) -> per-user NDCG@10
    for version in ("p2", "p5"):
        per_user, _ = compare_models(REFERENCE + NEW, data, n_users=n_users, version=version, **SETTING)
        for label, part in per_user.groupby("model"):
            ndcg[(label.split(" [")[0], version)] = part.set_index("user_id")["llm_ndcg"]

    def paired(a, b, text):
        diff = ndcg[a] - ndcg[b]
        p = wilcoxon(diff).pvalue if diff.any() else 1.0
        print(f"{text}: {diff.mean():+.4f} (better {(diff > 0).sum()}, worse {(diff < 0).sum()}, p = {p:.3f})")

    print()
    for _, model in NEW:
        for version in ("p2", "p5"):
            paired((model, version), ("deepseek-v4.1-flash", version), f"{model} - DeepSeek Flash [{version}]")
        paired((model, "p5"), (model, "p2"), f"{model}: p5 - p2")
