'''
NCF (NeuMF, fold-in) in the cold-start setup: three follow-ups to option 11.

  pilot   the first 20 validation users (the usual pilot group). NOT a new sample: they are part of the
          203 validation users, so this only shows the pilot board's numbers on their own. Seconds.
  seeds   how much NCF's validation result depends on chance. Trains NCF twice more on the warm users
          (training seeds 43 and 44, about 40 minutes each, once; saved), then scores the 203 validation
          users with the fixed fold-in settings:
            training seed  42, 43, 44   (fold-in seed = training seed)  -> model-to-model spread
            fold-in seed   42, 43, 44   (model 42)                      -> fold-in spread on its own
  test    the pre-registered test on the 814 test users (output/experiments/ncf_cold_start/preregistration.md).
          Refuses to run unless the pre-registration is committed to git unchanged, and runs only once.

Outputs: output/experiments/ncf_cold_start/

Usage:
    python experiments/ltr/ncf_experiments.py pilot
    python experiments/ltr/ncf_experiments.py seeds
    python experiments/ltr/ncf_experiments.py test
'''

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))  # shared modules live in src/
import hashlib
import subprocess
import sys

MODE = sys.argv[1] if len(sys.argv) > 1 else ""
sys.argv = [sys.argv[0]]
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import build_movie_texts as bmt
import cold_start_experiment as cs
import llm_rerank_experiment as lr
import final_comparison as fc
import cold_start_baselines as cbl

FOLDER = Path("output") / "experiments" / "ncf_cold_start"
PREREGISTRATION = FOLDER / "preregistration.md"
TEST_DONE = FOLDER / "test_hypotheses.csv"                 # written last: marks the test as run
K = bmt.K
SEED_COUNTS = (10, 3)
TRAINING_SEEDS = (42, 43, 44)

# Fixed in the pre-registration; the test stops if the code no longer matches
PREREGISTERED = {"fold": {10: (1000, 0.05), 3: (300, 0.01)}, "n_neg": 4, "train_seed": 42,
                 "ncf_params": dict(model_type="NeuMF", n_factors=4, layer_sizes=[16, 8, 4], n_epochs=30,
                                    batch_size=256, learning_rate=1e-3),
                 "sha256": "1f9c7c38141f6dfca3c4094a537606636f3a175c622a2090fc11f80e785d8959"}
EXPECTED = {"val NCF": {10: 0.5980, 3: 0.5537}, "test Hybrid E": {10: 0.6311, 3: 0.5970},
            "test Popularity": {10: 0.5598, 3: 0.5598}}


def score(users, recommend):
    """Per-user NDCG / precision / recall of one method."""
    return pd.DataFrame([{"user_id": u["user_id"], **bmt.user_metrics(recommend(u), u["relevant"], K)}
                         for u in users]).set_index("user_id")


def references(users, data):
    return {"Hybrid E": score(users, lambda u: lr.hybrid_e_candidates(u, data)[0][:K]),
            "Popularity": score(users, lambda u: cs.recommend_popular(data["popular"].index, u, K))}


def paired(a, b):
    diff = (a - b).dropna()
    return {"difference": diff.mean(), "better": int((diff > 0).sum()), "worse": int((diff < 0).sum()),
            "p_value": wilcoxon(diff).pvalue if diff.any() else 1.0}


def bootstrap_ci(diff, resamples=10_000):
    values = diff.to_numpy()
    idx = np.random.default_rng(42).integers(0, len(values), size=(resamples, len(values)))
    means = values[idx].mean(axis=1)
    return np.percentile(means, 2.5), np.percentile(means, 97.5)


# ───────────────────────── pilot ─────────────────────────
def run_pilot(data, models):
    rows = []
    for n in SEED_COUNTS:
        users = cs.users_with_n_seeds(data["val_users"], n)[:lr.PILOT_USERS]
        ncf = score(users, models.ncf_list)
        for name, ref in references(users, data).items():
            rows.append({"seeds": n, "users": len(users), "NCF": ncf["ndcg"].mean(), "reference": name,
                         "reference_ndcg": ref["ndcg"].mean(), **paired(ncf["ndcg"], ref["ndcg"])})
    table = pd.DataFrame(rows)
    table.to_csv(FOLDER / "pilot_val20.csv", index=False)
    print("\nNCF, 20-user pilot (the first 20 of the 203 validation users, NOT an independent sample)")
    print(table.round(4).to_string(index=False))


# ───────────────────────── training-seed study (validation only) ─────────────────────────
def run_seeds(data, models):
    runs = [(t, t) for t in TRAINING_SEEDS] + [(42, f) for f in TRAINING_SEEDS if f != 42]
    per_user, rows = [], []
    for n in SEED_COUNTS:
        users = cs.users_with_n_seeds(data["val_users"], n)
        hybrid = references(users, data)["Hybrid E"]["ndcg"]
        for train_seed, fold_seed in runs:
            models.load_ncf(train_seed, fold_seed)
            ncf = score(users, models.ncf_list)
            per_user.append(ncf.assign(seeds=n, train_seed=train_seed, fold_seed=fold_seed).reset_index())
            rows.append({"seeds": n, "train_seed": train_seed, "fold_seed": fold_seed, "ndcg": ncf["ndcg"].mean(),
                         "precision": ncf["precision"].mean(), "recall": ncf["recall"].mean(),
                         "hybrid_E": hybrid.mean(), **{f"vs_hybrid_E_{k}": v for k, v in
                                                       paired(ncf["ndcg"], hybrid).items()}})
            print(f"{n} seeds | training seed {train_seed} | fold-in seed {fold_seed}: NDCG@10 {rows[-1]['ndcg']:.4f}",
                  flush=True)
    models.load_ncf(42)                                  # back to the option-11 model
    table = pd.DataFrame(rows)
    pd.concat(per_user).to_csv(FOLDER / "seeds_per_user_val.csv", index=False)
    table.to_csv(FOLDER / "seeds_summary_val.csv", index=False)

    spread = []
    for n in SEED_COUNTS:
        part = table[table["seeds"] == n]
        for label, sel in (("training seeds 42-44 (fold-in seed = training seed)", part["train_seed"] == part["fold_seed"]),
                           ("fold-in seeds 42-44 (model 42)", part["train_seed"] == 42)):
            values = part.loc[sel, "ndcg"]
            spread.append({"seeds": n, "varied": label, "runs": len(values), "mean": values.mean(),
                           "std": values.std(ddof=1), "min": values.min(), "max": values.max(),
                           "range": values.max() - values.min(), "hybrid_E": part["hybrid_E"].iloc[0]})
    spread = pd.DataFrame(spread)
    spread.to_csv(FOLDER / "seeds_spread_val.csv", index=False)
    pd.set_option("display.width", 250)
    print("\nNCF on the 203 validation users, per run:")
    print(table[["seeds", "train_seed", "fold_seed", "ndcg", "vs_hybrid_E_difference", "vs_hybrid_E_p_value"]]
          .round(4).to_string(index=False))
    print("\nSpread:")
    print(spread.round(4).to_string(index=False))


# ───────────────────────── pre-registered test (runs once) ─────────────────────────
def check_preregistration():
    if TEST_DONE.exists():
        sys.exit(f"The NCF test has already been run ({TEST_DONE}). It is run only once.")
    if not PREREGISTRATION.exists():
        sys.exit(f"No pre-registration at {PREREGISTRATION}.")
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", PREREGISTRATION.as_posix()], capture_output=True)
    changed = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", PREREGISTRATION.as_posix()], capture_output=True)
    if tracked.returncode != 0 or changed.returncode != 0:
        sys.exit(f"Commit the pre-registration first (git add \"{PREREGISTRATION}\" and git commit), "
                 "so its date and content are on record before the test.")
    current = {"fold": cbl.FOLD, "n_neg": cbl.N_NEG, "train_seed": cbl.SEED, "ncf_params": cbl.NCF_PARAMS}
    for key, value in current.items():
        assert value == PREREGISTERED[key], f"{key} differs from the pre-registration: {value}"
    digest = hashlib.sha256(cbl.ncf_path(42).read_bytes()).hexdigest()
    assert digest == PREREGISTERED["sha256"], "check 3: the NCF file is not the pre-registered model"
    print("check 3: NCF file SHA-256 matches the pre-registration")


def run_test(data, models):
    models.load_ncf(42)                              # check_preregistration() already ran in __main__
    test_ids = {u["user_id"] for u in data["test_users"]}
    assert not (test_ids & set(data["train"]["user_id"])), "check 5: a test user is in the warm training data"
    print("check 5: no test user in the training data")
    for n in SEED_COUNTS:                                                    # check 4, validation only
        val = score(cs.users_with_n_seeds(data["val_users"], n), models.ncf_list)["ndcg"].mean()
        print(f"check 4: NCF on validation, {n} seeds: {val:.4f} (expected {EXPECTED['val NCF'][n]})")
        assert abs(val - EXPECTED["val NCF"][n]) < 5e-5, "check 4 failed"

    per_user, refs = {}, {}
    for n in SEED_COUNTS:                                                    # checks 1-2 before any NCF test score
        users = cs.users_with_n_seeds(data["test_users"], n)
        refs[n] = references(users, data)
        for name in ("Hybrid E", "Popularity"):
            got = refs[n][name]["ndcg"].mean()
            print(f"check {1 if name == 'Hybrid E' else 2}: {name} on test, {n} seeds: {got:.4f} "
                  f"(expected {EXPECTED['test ' + name][n]})")
            assert abs(got - EXPECTED["test " + name][n]) < 5e-5, "check 1/2 failed"
    for n in SEED_COUNTS:
        users = cs.users_with_n_seeds(data["test_users"], n)
        print(f"Scoring NCF on {len(users)} test users, {n} seeds...", flush=True)
        per_user[n] = pd.concat({"NCF": score(users, models.ncf_list), **refs[n]}, axis=1)

    tests = []
    for label, n, baseline in (("H1", 10, "Hybrid E"), ("H2", 3, "Hybrid E"),
                               ("H3", 10, "Popularity"), ("H4", 3, "Popularity")):
        ncf, ref = per_user[n][("NCF", "ndcg")], per_user[n][(baseline, "ndcg")]
        diff = ncf - ref
        low, high = bootstrap_ci(diff)
        r = paired(ncf, ref)
        tests.append({"hypothesis": label, "seeds": n, "comparison": f"NCF vs {baseline}", "NCF": ncf.mean(),
                      "baseline": ref.mean(), **r, "ci_low": low, "ci_high": high,
                      "p_bonferroni": min(1.0, 4 * r["p_value"])})
    tests = pd.DataFrame(tests)
    tests["confirmed"] = tests["p_bonferroni"] < 0.05
    tests["direction"] = np.where(tests["difference"] > 0, "NCF better", "NCF worse")

    summary = pd.DataFrame([{"seeds": n, "method": m, **per_user[n][m].mean().to_dict()}
                            for n in SEED_COUNTS for m in ("NCF", "Hybrid E", "Popularity")])
    for n in SEED_COUNTS:
        per_user[n].to_csv(FOLDER / f"test_per_user_n{n}.csv")
    summary.to_csv(FOLDER / "test_summary.csv", index=False)
    tests.to_csv(TEST_DONE, index=False)                                  # last: marks the test as run
    pd.set_option("display.width", 250)
    print("\nNCF, pre-registered test, 814 test users")
    print(summary.round(4).to_string(index=False))
    print(tests.round(4).to_string(index=False))


if __name__ == "__main__":
    if MODE not in ("pilot", "seeds", "test"):
        sys.exit(__doc__)
    FOLDER.mkdir(parents=True, exist_ok=True)
    if MODE == "test":
        check_preregistration()                     # stop before loading anything if not allowed
    data = fc.load()
    models = cbl.ColdStartBaselines(data)
    {"pilot": run_pilot, "seeds": run_seeds, "test": run_test}[MODE](data, models)
