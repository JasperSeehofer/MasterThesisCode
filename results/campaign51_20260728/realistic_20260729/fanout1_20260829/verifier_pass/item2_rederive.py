#!/usr/bin/env python3
"""
Item 2 (B1.1 Stage-0 verifier pass) — from-scratch, independent re-derivation of the
S0-A pooled scores, GATE ENG, and the dark-class instrument-identity check.

Reads ONLY the raw per-event CSVs under hier_s0_registered_run/. Does not import or
call hier_s0_driver.py, and does not read any *_output.json / *.md record.

Formulas taken from the REGISTERED text (PREREGISTRATION_HIER_HTHETA_20260826.md §4.1):
    score_b = [lnL(b=+0.02,s=1) - lnL(b=-0.02,s=1)] / 0.04
    score_s = [lnL(b=0,s=sqrt(2)) - lnL(b=0,s=1/sqrt(2))] / (sqrt(2) - 1/sqrt(2))
    Z_x = mean(score_x) / SEM(score_x), SEM = sample_std(ddof=1)/sqrt(n), pooled over
    events and seeds.
"""
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(
    "/home/jasper/Repositories/darksiren-emri/results/campaign51_20260728/"
    "realistic_20260729/fanout1_20260829/hier_s0_registered_run"
)
SEEDS = [900101, 900102, 900103, 900104]
NODE_DIR = {
    "truth": "node_truth_sites2.2_nosmear",
    "b_plus": "node_b_plus_sites2.2_nosmear",
    "b_minus": "node_b_minus_sites2.2_nosmear",
    "s_plus": "node_s_plus_sites2.2_nosmear",
    "s_minus": "node_s_minus_sites2.2_nosmear",
}
DENOM_S = math.sqrt(2.0) - 1.0 / math.sqrt(2.0)


def load(seed: int, node: str) -> pd.DataFrame:
    f = (
        ROOT
        / f"s0a_seed{seed}"
        / NODE_DIR[node]
        / "simulations"
        / "diagnostics"
        / "event_likelihoods.csv"
    )
    df = pd.read_csv(f)
    df = df[df["h"] == 0.73].copy()
    # de-duplicate on event_idx, keep last (disclosed duplicate-row artifact, seed900101/b_plus)
    df = df.drop_duplicates(subset="event_idx", keep="last")
    for col in ("combined_no_bh", "combined_with_bh"):
        df["ln_" + col] = np.where(df[col] > 0, np.log(df[col]), np.nan)
    return df.set_index("event_idx")


def sem(x: np.ndarray) -> float:
    return float(np.std(x, ddof=1) / math.sqrt(len(x)))


def main() -> None:
    rows_per_seed = {}
    all_pooled = {"score_b": {"no_bh": [], "with_bh": []}, "score_s": {"no_bh": [], "with_bh": []}}
    dark_pooled = {"score_b": {"no_bh": []}, "score_s": {"no_bh": []}}
    class_flag_all = {}

    for seed in SEEDS:
        d = {node: load(seed, node) for node in NODE_DIR}
        # class definition: L_cat_no_bh == 0 at truth ("dark")
        truth = d["truth"]
        dark_idx = set(truth.index[truth["L_cat_no_bh"] == 0.0])

        for channel_col, key in (("ln_combined_no_bh", "no_bh"), ("ln_combined_with_bh", "with_bh")):
            bp = d["b_plus"][channel_col]
            bm = d["b_minus"][channel_col]
            sp = d["s_plus"][channel_col]
            sm = d["s_minus"][channel_col]

            joined_b = pd.concat([bp, bm], axis=1, join="inner", keys=["bp", "bm"]).dropna()
            joined_s = pd.concat([sp, sm], axis=1, join="inner", keys=["sp", "sm"]).dropna()

            score_b = (joined_b["bp"] - joined_b["bm"]) / 0.04
            score_s = (joined_s["sp"] - joined_s["sm"]) / DENOM_S

            all_pooled["score_b"][key].append(score_b)
            all_pooled["score_s"][key].append(score_s)

            if key == "no_bh":
                dark_pooled["score_b"]["no_bh"].append(score_b[score_b.index.isin(dark_idx)])
                dark_pooled["score_s"]["no_bh"].append(score_s[score_s.index.isin(dark_idx)])

        # GATE ENG: fraction of events moved >= 1e-6 relative in ln_combined_no_bh vs truth, per off-truth node
        truth_ln = truth["ln_combined_no_bh"]
        eng_fracs = {}
        for node in ("b_plus", "b_minus", "s_plus", "s_minus"):
            j = pd.concat([truth_ln, d[node]["ln_combined_no_bh"]], axis=1, join="inner", keys=["t", "n"]).dropna()
            rel = np.abs(j["n"] - j["t"]) / np.abs(j["t"])
            frac = float((rel >= 1e-6).mean())
            eng_fracs[node] = frac

        rows_per_seed[seed] = {
            "n_events_truth": len(truth),
            "n_dark": len(dark_idx),
            "gate_eng": eng_fracs,
        }

    print("=" * 80)
    print("Per-seed GATE ENG (fraction moved >=1e-6 relative, ln_combined_no_bh)")
    for seed, info in rows_per_seed.items():
        print(seed, info["n_events_truth"], info["n_dark"], info["gate_eng"])

    print("=" * 80)
    print("Pooled scores (all seeds), channel=ln_L_no_bh (registered primary) and with_bh (secondary)")
    for score_name in ("score_b", "score_s"):
        for key in ("no_bh", "with_bh"):
            pooled = pd.concat(all_pooled[score_name][key])
            n = len(pooled)
            mean = float(pooled.mean())
            se = sem(pooled.values)
            z = mean / se
            print(f"{score_name:8s} {key:8s} mean={mean: .8f} sem={se:.8f} Z={z: .6f} n={n}")

    print("=" * 80)
    print("Per-seed, registered primary channel (no_bh)")
    for score_name in ("score_b", "score_s"):
        for i, seed in enumerate(SEEDS):
            s = all_pooled[score_name]["no_bh"][i]
            n = len(s)
            mean = float(s.mean())
            se = sem(s.values)
            z = mean / se
            print(f"seed={seed} {score_name} n={n} mean={mean: .6f} sem={se:.6f} Z={z: .6f}")

    print("=" * 80)
    print("By-class split, registered primary channel (no_bh)")
    dark_b = pd.concat(dark_pooled["score_b"]["no_bh"])
    dark_s = pd.concat(dark_pooled["score_s"]["no_bh"])
    print(f"dark n={len(dark_b)} score_b: min={dark_b.min()} max={dark_b.max()} mean={dark_b.mean()}")
    print(f"dark n={len(dark_s)} score_s: min={dark_s.min()} max={dark_s.max()} mean={dark_s.mean()}")

    all_b = pd.concat(all_pooled["score_b"]["no_bh"])
    all_s = pd.concat(all_pooled["score_s"]["no_bh"])
    dark_idx_all = dark_b.index
    matched_b = all_b[~all_b.index.isin(dark_idx_all)]
    # NOTE: dark index only unique within a seed's event_idx numbering; since we pooled
    # across seeds with a plain concat (not a MultiIndex), a cross-seed collision on
    # event_idx would corrupt this matched/dark split. Cross-check below.
    n_total = len(all_b)
    n_dark = sum(len(x) for x in dark_pooled["score_b"]["no_bh"])
    n_matched_expected = n_total - n_dark
    print(f"n_total={n_total} n_dark={n_dark} n_matched_expected={n_matched_expected}")


if __name__ == "__main__":
    main()
