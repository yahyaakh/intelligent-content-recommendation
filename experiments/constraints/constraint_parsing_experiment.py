'''
E4: typed requests -> constraints. Parser accuracy and constraint satisfaction (RQ5).
Rules: output/experiments/constraint_parsing/preregistration.md (written before the run).

Usage:
    python experiments/constraints/constraint_parsing_experiment.py check   no LLM: gold answers valid, menu lists satisfy 100 %
    python experiments/constraints/constraint_parsing_experiment.py run     parse the 50 requests (cached), score everything
'''
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MODE = sys.argv[1] if len(sys.argv) > 1 else "check"
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "web"))
from recommender.thesis_engine import get_engine

FOLDER = Path("output") / "experiments" / "constraint_parsing"
ANSWERS = FOLDER / "answers.jsonl"
N_USERS = 20
K = 10
INCLUDE = ("include_genres", "include_moods", "include_themes")
EXCLUDE = ("exclude_genres", "exclude_moods")
PEOPLE = ("include_actors", "include_directors")
FIELDS = ("audience",) + INCLUDE + EXCLUDE + PEOPLE + ("max_intensity",)
FILM_COLUMN = {"include_genres": "genres", "exclude_genres": "genres", "include_moods": "llm_mood",
               "exclude_moods": "llm_mood", "include_themes": "llm_themes",
               "include_actors": "cast_full", "include_directors": "directors_full"}


def values(film, field):
    return set(str(film.get(FILM_COLUMN[field], "")).split("|")) - {"", "nan"}


# ───────────────────────── gold answers ─────────────────────────
def required(gold):
    """The gold constraints a film must meet (optional ones and empty excludes left out)."""
    out = {}
    for field, spec in gold.items():
        if field in INCLUDE and not spec.get("optional"):
            out[field] = spec["any_of"]
        elif field in EXCLUDE and spec["all_of"]:
            out[field] = spec["all_of"]
        elif field in PEOPLE or field == "audience":
            out[field] = spec
        elif field == "max_intensity":
            out[field] = {dim: max(allowed) for dim, allowed in spec.items()}
    return out


def menu_request(gold):
    """The gold request in the form of recomm.py's request dict (what the menu would produce)."""
    return dict(required(gold))       # same keys and value shapes as the request dict


def satisfies(film, req):
    """Independent of ConstraintLayer: does the film meet every required constraint?"""
    for field, want in req.items():
        if field == "audience":
            audience = str(film.get("llm_audience", "")).lower()
            if want == "children" and audience not in ("children", "family"):
                return False
            if want == "teens" and audience == "adults":
                return False
        elif field in INCLUDE or field in PEOPLE:
            if values(film, field).isdisjoint(want):
                return False
        elif field in EXCLUDE:
            if not values(film, field).isdisjoint(want):
                return False
        elif field == "max_intensity":
            if any(film[f"intensity_{dim}"] > limit for dim, limit in want.items()):
                return False
    return True


# ───────────────────────── parser scoring ─────────────────────────
def field_correct(field, parsed, gold):
    spec = gold.get(field)
    got = parsed.get(field)
    if field == "audience":
        return got == spec
    if field in INCLUDE:
        got = set(got or [])
        if spec is None:
            return not got
        return (bool(got) and got <= set(spec["any_of"])) or (spec.get("optional", False) and not got)
    if field in EXCLUDE:
        got = set(got or [])
        if spec is None:
            return not got
        return set(spec["all_of"]) <= got <= set(spec["all_of"]) | set(spec.get("ok", []))
    if field in PEOPLE:
        return set(got or []) == set(spec or [])
    if field == "max_intensity":
        got, spec = got or {}, spec or {}
        return set(got) == set(spec) and all(got[d] in spec[d] for d in spec)
    raise ValueError(field)


def invented(field, parsed, gold):
    """The parser set a field the gold has no spec for (and that is not optional)."""
    return field not in gold and bool(parsed.get(field))


# ───────────────────────── modes ─────────────────────────
def load_requests():
    return json.loads((FOLDER / "requests.json").read_text(encoding="utf-8"))


def check(engine, requests):
    cl = engine.constraints
    for r in requests:
        cl.validate_request(menu_request(r["gold"]))           # every gold value is in the catalog
        for field, spec in r["gold"].items():
            for v in spec.get("any_of", []) + spec.get("all_of", []) + spec.get("ok", []) \
                    if isinstance(spec, dict) and field != "max_intensity" else []:
                pool = cl.all_genres if "genres" in field else cl.all_moods if "moods" in field else cl.all_themes
                assert v in pool, f"request {r['id']}: {v!r} not in the {field} vocabulary"
    print(f"{len(requests)} gold answers valid")


def parse_all(engine, requests):
    done = {}
    if ANSWERS.exists():
        done = {a["id"]: a for a in map(json.loads, ANSWERS.read_text(encoding="utf-8").splitlines())}
    with ANSWERS.open("a", encoding="utf-8") as f:
        for r in requests:
            if r["id"] in done:
                continue
            record = {"id": r["id"], "text": r["text"]}
            try:
                parsed, info = engine.parse_request(r["text"])
                record.update(parsed=parsed, rejected=None, **info)
            except Exception as error:                    # parser failure or validate_request rejection
                record.update(parsed=None, rejected=f"{type(error).__name__}: {error}",
                              **{k: engine.parser.last_info.get(k) for k in ("seconds",)})
            record["raw"] = engine.parser.last_info.get("raw")
            f.write(json.dumps(record) + "\n")
            f.flush()
            done[r["id"]] = record
            print(f"  {r['id']:2d} {r['text'][:50]:50s} -> {record['parsed'] or record['rejected']}", flush=True)
    return done


def satisfaction(engine, requests, answers):
    users = engine.cs.users_with_n_seeds(engine.data["val_users"], 10)[:N_USERS]
    rankings = {}
    for user in users:
        engine._with_alpha(user)
        ranked = engine.rank_hybrid_e(user)
        rankings[user["user_id"]] = [{"movie_id": m, **engine.catalog.loc[m].to_dict()} for m in ranked]
    rows = []
    for r in requests:
        req = required(r["gold"])
        if not req:
            continue
        parsed = answers[r["id"]]["parsed"]
        for user_id, rows_ranked in rankings.items():
            for kind, request in (("none", {}), ("LLM", parsed), ("menu", menu_request(r["gold"]))):
                if request is None:
                    rows.append({"id": r["id"], "category": r["category"], "user_id": user_id, "list": kind,
                                 "shown": 0, "satisfied": 0, "rejected": True})
                    continue
                kept, _ = engine.apply_constraints(rows_ranked, request)
                shown = kept[:K]
                rows.append({"id": r["id"], "category": r["category"], "user_id": user_id, "list": kind,
                             "shown": len(shown), "satisfied": sum(satisfies(film, req) for film in shown),
                             "rejected": False})
    return pd.DataFrame(rows)


def run(engine, requests):
    answers = parse_all(engine, requests)
    rows = []
    for r in requests:
        a = answers[r["id"]]
        parsed = a["parsed"]
        row = {"id": r["id"], "category": r["category"], "text": r["text"], "rejected": parsed is None,
               "seconds": a.get("seconds"), "tokens_in": a.get("tokens_in"), "tokens_out": a.get("tokens_out")}
        for field in FIELDS:
            row[f"ok_{field}"] = parsed is not None and field_correct(field, parsed, r["gold"])
            row[f"in_gold_{field}"] = field in r["gold"]
            row[f"invented_{field}"] = parsed is not None and invented(field, parsed, r["gold"])
        row["correct"] = all(row[f"ok_{f}"] for f in FIELDS)
        rows.append(row)
    per_request = pd.DataFrame(rows)
    per_request.to_csv(FOLDER / "per_request.csv", index=False)

    print(f"\nRequests fully correct: {per_request['correct'].sum()} of {len(per_request)} "
          f"({per_request['correct'].mean():.0%}); rejected: {per_request['rejected'].sum()}")
    print(per_request.groupby("category", sort=False)["correct"].agg(["sum", "count"]).to_string())
    fields = []
    for f in FIELDS:
        asked = per_request[per_request[f"in_gold_{f}"]]
        fields.append({"field": f, "requests_with_field": len(asked),
                       "accuracy": asked[f"ok_{f}"].mean() if len(asked) else np.nan,
                       "invented": int(per_request[f"invented_{f}"].sum())})
    fields = pd.DataFrame(fields)
    fields.to_csv(FOLDER / "per_field.csv", index=False)
    print("\n" + fields.round(3).to_string(index=False))
    t = per_request["seconds"].dropna()
    print(f"\nTime per call: mean {t.mean():.2f} s, median {t.median():.2f} s, max {t.max():.2f} s; "
          f"tokens in {per_request['tokens_in'].mean():.0f}, out {per_request['tokens_out'].mean():.0f}")

    sat = satisfaction(engine, requests, answers)
    sat.to_csv(FOLDER / "satisfaction_per_list.csv", index=False)
    summary = (sat[~sat["rejected"]].groupby("list")
               .apply(lambda g: pd.Series({"satisfaction": g["satisfied"].sum() / max(g["shown"].sum(), 1),
                                           "short_lists": (g["shown"] < K).mean(),
                                           "lists": len(g)}), include_groups=False))
    summary.to_csv(FOLDER / "satisfaction_summary.csv")
    by_group = (sat[~sat["rejected"]].groupby(["category", "list"])
                .apply(lambda g: g["satisfied"].sum() / max(g["shown"].sum(), 1), include_groups=False)
                .unstack("list"))
    by_group.to_csv(FOLDER / "satisfaction_by_group.csv")
    print("\nConstraint satisfaction (share of shown films that meet the request):")
    print(summary.round(3).to_string())
    print(by_group.round(3).to_string())
    menu = summary.loc["menu", "satisfaction"]
    print("check: menu lists satisfy 100 % ->", "ok" if menu == 1 else f"NOT 100 % ({menu:.3f})")


if __name__ == "__main__":
    FOLDER.mkdir(parents=True, exist_ok=True)
    engine = get_engine()
    requests = load_requests()
    check(engine, requests)
    if MODE == "run":
        run(engine, requests)
    elif MODE == "check":
        sat = satisfaction(engine, requests, {r["id"]: {"parsed": None} for r in requests})
        menu = sat[sat["list"] == "menu"]
        rate = menu["satisfied"].sum() / menu["shown"].sum()
        print(f"menu lists: satisfaction {rate:.3f} over {len(menu)} lists "
              f"({(menu['shown'] < K).mean():.0%} with fewer than {K} films)")
        none = sat[sat["list"] == "none"]
        print(f"no constraints: satisfaction {none['satisfied'].sum() / none['shown'].sum():.3f}")
