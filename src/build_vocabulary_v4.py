"""Build the larger theme/mood vocabulary for prompt v4 from the MovieLens Tag Genome.

Source: data/ml-25m/genome/genome-tags.csv (1,128 tags curated by MovieLens; Vig et al. 2012).
The lists below are the genome tags kept after removing tags that are not themes or moods
(names, franchises, studios, awards and lists, quality judgements, production and source
details, places, years and eras, genres (already in the LLM input), nudity and age ratings,
spelling duplicates). Every entry must be an exact genome tag, so the labels can later be
checked against the genome scores.

A tag is kept only if the genome rates it as clearly relevant (score >= 0.5) for at least
MIN_MOVIES of the 1,957 catalog movies; tags that fit almost nothing would only lengthen the prompt.

Usage:  python src/build_vocabulary_v4.py   ->  data/vocabularies/v4_genome.json
"""
import json
from pathlib import Path

import pandas as pd

MIN_SCORE, MIN_MOVIES = 0.5, 5
OUT = Path("data/vocabularies") / "v4_genome.json"

# What the film is about.
THEME_CANDIDATES = """abortion | addiction | adolescence | adoption | adultery | afterlife | aging | aids | alcoholism |
alien invasion | aliens | allegory | alter ego | alternate history | alternate reality | amnesia | androids | animals |
anti-hero | anti-semitism | anti-war | apocalypse | archaeology | arms dealer | art | artificial intelligence | artist |
assassination | astronauts | atheism | autism | aviation | ballet | bank robbery | baseball | basketball | betrayal |
biblical | biography | blindness | boarding school | boxing | brainwashing | brothers | bullying | business | cancer |
cannibalism | capitalism | car chase | casino | catastrophe | catholicism | censorship | chess | child abuse |
childhood | christianity | christmas | circus | civil war | clones | cold war | college | colonialism | communism |
competition | computers | con artists | conspiracy | cooking | corruption | courage | courtroom drama | creativity |
crime gone awry | cross dressing | culture clash | cyberpunk | dance | death | death penalty | demons | depression |
destiny | detective | devil | dinosaurs | disability | disaster | divorce | dragons | dreams | drug addiction | drugs |
dysfunctional family | dystopia | end of the world | environment | espionage | ethnic conflict | evolution |
existentialism | fairy tale | family bonds | fascism | fashion | father daughter relationship | father son relationship |
fighting the system | first contact | football | freedom | friendship | gambling | gangsters | genetics | genius |
genocide | ghosts | global warming | good versus evil | greed | guilt | hackers | haunted house | heist | high school |
holocaust | homophobia | homosexuality | hostage | hunting | identity | imagination | immigrants | immortality |
innocence lost | insanity | investigation | isolation | journalism | justice | kidnapping | lawyers | loneliness |
love story | love triangles | mad scientist | mafia | magic | manipulation | marriage | martial arts | memory loss |
mental illness | mentor | metaphysics | midlife crisis | military | mistaken identity | money | monsters |
moral ambiguity | mother daughter relationship | mother-son relationship | mountain climbing | movie business | murder |
music | musicians | mutants | mythology | native americans | natural disaster | nature | nazis | nuclear war |
obsession | organized crime | orphans | paranoia | parenthood | parallel universe | philosophy | pirates | police |
police corruption | political corruption | politics | post-apocalyptic | poverty | pregnancy | prejudice | prison |
prison escape | propaganda | prostitution | psychiatry | psychology | racing | racism | rags to riches | rape |
rebellion | redemption | religion | revenge | revolution | road trip | robots | rock and roll | sacrifice | samurai |
school | science | secrets | segregation | self discovery | serial killer | sexual abuse | sexuality | single father |
sisters | slavery | small town | social commentary | space travel | spies | sports | stranded | suburbia | suicide |
super hero | supernatural | surveillance | survival | teacher | technology | terminal illness | terrorism | time loop |
time travel | torture | transformation | transgender | treasure hunt | underdog | unlikely friendships | utopia |
vampires | vietnam war | vigilantism | virtual reality | virus | war | wedding | werewolves | witches | working class |
world war i | world war ii | writers | zombies"""

# How the film feels and how it is told (tone, style, pacing, narrative).
MOOD_CANDIDATES = """absurd | affectionate | atmospheric | bittersweet | bizarre | bleak | brutal | campy | cathartic |
cerebral | character study | claustrophobic | clever | compassionate | complex | confrontational | creepy | cute |
cynical | dark | dark humor | deadpan | depressing | dialogue driven | disturbing | downbeat | dramatic | dreamlike |
earnest | eerie | emotional | enigmatic | epic | erotic | exciting | fast paced | feel-good | forceful | frightening |
fun | funny | goofy | gory | gothic | grim | gritty | hallucinatory | happy ending | harsh | heartbreaking |
heartwarming | hilarious | humorous | inspirational | intense | intimate | ironic | irreverent | light | lyrical |
macabre | meditative | melancholic | moody | moving | multiple storylines | noir | non-linear | nostalgic | ominous |
over the top | paranoid | passionate | philosophical | poignant | psychedelic | psychological | quirky | realistic |
reflective | romantic | sad | satirical | scary | sentimental | silly | slapstick | slow paced | stylized | surreal |
suspenseful | sweet | tense | thought-provoking | touching | tragedy | twist ending | understated | violent |
visceral | visually stunning | whimsical | wistful | witty"""


def split(block):
    return [t.strip() for t in block.replace("\n", " ").split("|") if t.strip()]


if __name__ == "__main__":
    tags = pd.read_csv("data/ml-25m/genome/genome-tags.csv")
    tag_id = dict(zip(tags["tag"], tags["tagId"]))
    themes, moods = split(THEME_CANDIDATES), split(MOOD_CANDIDATES)
    unknown = [t for t in themes + moods if t not in tag_id]
    assert not unknown, f"not genome tags: {unknown}"
    assert not set(themes) & set(moods), "a tag is in both lists"
    assert len(set(themes)) == len(themes) and len(set(moods)) == len(moods), "duplicate candidates"

    # Genome relevance for the catalog movies (MovieLens movieId -> TMDB id via links.csv).
    links = pd.read_csv("data/ml-25m/links.csv").dropna(subset=["tmdbId"])
    links["tmdbId"] = links["tmdbId"].astype(int)
    catalog = set(pd.read_csv("data/data/processed/movies.csv")["movie_id"])
    ml_ids = set(links.loc[links["tmdbId"].isin(catalog), "movieId"])
    scores = pd.read_csv("data/ml-25m/genome/genome-scores.csv")
    scores = scores[scores["movieId"].isin(ml_ids) & (scores["relevance"] >= MIN_SCORE)]
    movies_per_tag = scores.groupby("tagId")["movieId"].nunique()
    count = {t: int(movies_per_tag.get(tag_id[t], 0)) for t in themes + moods}
    print(f"catalog movies with genome scores: {scores['movieId'].nunique()} of {len(catalog)}")

    kept_themes = [t for t in themes if count[t] >= MIN_MOVIES]
    kept_moods = [t for t in moods if count[t] >= MIN_MOVIES]
    dropped = sorted((t for t in themes + moods if count[t] < MIN_MOVIES), key=lambda t: count[t])
    print(f"themes: {len(kept_themes)} kept of {len(themes)} candidates | moods: {len(kept_moods)} of {len(moods)}")
    print("dropped (movies with relevance >= 0.5):", ", ".join(f"{t} ({count[t]})" for t in dropped))

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps({
        "source": "MovieLens 25M Tag Genome (genome-tags.csv), manually filtered to themes and moods",
        "filter": f"genome relevance >= {MIN_SCORE} for at least {MIN_MOVIES} of the 1,957 catalog movies",
        "themes": kept_themes, "moods": kept_moods,
        "catalog_movies_per_tag": {t: count[t] for t in kept_themes + kept_moods},
    }, indent=2), encoding="utf-8")
    print(f"Saved {OUT}")
