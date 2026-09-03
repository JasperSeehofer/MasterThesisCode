"""INDEPENDENT byte-id verifier for b-completion-scorer (r-completion-residual).

Standing rule: verifier output is evidence, not authority -- re-derive, do not trust the
builder's script or its printed JSON. This script does NOT import completion_residual_reads.py.
It re-implements the two byte-id checks named in the launch instruction directly from the raw
files:

  (A) the 67 S3 harness checkpoints' score_at_truth.no_bh.dark.mean, read back independently and
      compared bit-for-bit against the builder's BUILD_RECORD.md-quoted list (pair 1..67), plus
      re-aggregation (mean/SEM) as an independent T_harn/SE_harn cross-check.
  (B) the T0 re-baseline mean_h = 0.666987 (iiib, 1D / combined_no_bh channel), reproduced
      independently from the raw event_likelihoods.csv using the production zero-handling
      convention (imported verbatim from darksiren_emri.validation.correspondence_1d, per the
      registration draft's explicit instruction not to re-implement physics) and the T0
      gradient-trapezoid convention documented in prod2d_closure_20260818/tier0_bootstrap_
      jackknife.py's own docstring.

Does NOT run compute_registered_statistics / the registered statistic. Anchor-reproduction only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(REPO_ROOT))

from darksiren_emri.validation.correspondence_1d import combine_log_likelihood  # noqa: E402

HARNESS_ROOT = (
    REPO_ROOT
    / "results/campaign51_20260728/realistic_20260729/tree2_20260830/b8_cal_harness_work_s4_postflip"
)
IIIB_CSV = (
    REPO_ROOT
    / "results/campaign51_20260728/realistic_20260729/graph1_20260901/retrieved/"
    "run_20260902_graph1_headrebaseline_iiib/simulations/diagnostics/event_likelihoods.csv"
)
POPULATION = 200
CELL = "S"
N_EXPECTED = 67
T0_ANCHOR_DISPLAY = 0.666987
T0_TOLERANCE = 1e-9

# Builder's BUILD_RECORD.md-quoted 67 dark_full_score_means, verbatim (transcribed by hand from
# the record for a bit-for-bit comparison against an independent read of the same files).
BUILDER_DARK_MEANS = [
    -0.03684332680707096, -0.011932528209638497, -0.02660525358654448, -0.02358195028098366,
    -0.0010693493945890227, 0.0784279920769636, -0.025728798001607652, -0.010183244139164901,
    -0.03388657085016816, 0.004105865269774426, 0.023731752097597005, 0.10119215868463913,
    0.025260387464604628, -0.018667692883903495, 0.07172738260808453, -0.0006139003054327378,
    0.07782379383070917, -0.001205113505551609, -0.08481331079745157, 0.04729880792701685,
    0.06244292902260193, 0.025356251574629764, -0.014819984047714898, 0.03487727091787526,
    0.02151834928135097, -0.03694230961310493, 0.014608079067532753, -0.040341161273799254,
    0.05250871736831085, -0.03808404972565446, -0.0623556277167366, 0.05683335890701403,
    0.001307888828097373, -0.004558712355360873, 0.09676913243112817, 0.047837721097609726,
    -0.04404359759562281, -0.05770008304042757, 0.06154580360250548, 0.08222901708541754,
    -0.004234763043548639, 0.10123823266708055, 0.00978716322728008, -0.03869145615725185,
    -0.012077805524152456, 0.04258774244463247, 0.07815949972569997, -0.05969435014684141,
    -0.056620162272346675, 0.07952884950719988, -0.0008984806036467763, -0.0012834016970765664,
    -0.01998435105463944, -0.017618410017291634, 0.003944296565434913, 0.0971613656186274,
    -0.13676645707421928, -0.10032894427449991, 0.0437279004610132, -0.03540519673679583,
    0.047814526004112616, 0.030134165176089096, -0.014216777511634094, 0.10035303029215972,
    0.01825779943064465, -0.008854623691606767, -0.008982195966789818,
]


def check_harness_byte_id() -> dict:
    files = sorted(HARNESS_ROOT.glob(f"universe_seed*_{CELL}.json"))
    checkpoints = []
    for f in files:
        d = json.loads(f.read_text())
        checkpoints.append((f.name, d))

    matched = [
        (name, d) for name, d in checkpoints if int(d["universe"]["n_draw_requested"]) == POPULATION
    ]
    matched.sort(key=lambda nd: int(nd[1]["universe"]["seed"]))

    means = [d["score_at_truth"]["no_bh"]["dark"]["mean"] for _, d in matched]
    resolved_flags_set = {json.dumps(d["resolved_flags"], sort_keys=True) for _, d in matched}
    seeds = [int(d["universe"]["seed"]) for _, d in matched]

    n_pairs = min(len(means), len(BUILDER_DARK_MEANS))
    diffs = [abs(means[i] - BUILDER_DARK_MEANS[i]) for i in range(n_pairs)]
    max_abs_dev = max(diffs) if diffs else float("nan")
    exact_equal = all(d == 0.0 for d in diffs)

    indep_mean = float(np.mean(means))
    indep_sem = float(np.std(means, ddof=1) / (len(means) ** 0.5))

    return {
        "n_checkpoint_files_globbed": len(files),
        "n_checkpoints_matched_population": len(matched),
        "n_checkpoints_expected": N_EXPECTED,
        "count_green": len(matched) == N_EXPECTED,
        "seed_min": seeds[0] if seeds else None,
        "seed_max": seeds[-1] if seeds else None,
        "resolved_flags_internally_consistent": len(resolved_flags_set) == 1,
        "n_distinct_resolved_flags_blocks": len(resolved_flags_set),
        "n_pairs_compared_vs_build_record": n_pairs,
        "max_abs_dev_vs_build_record": max_abs_dev,
        "bit_for_bit_exact_vs_build_record": exact_equal,
        "independent_mean_of_dark_means": indep_mean,
        "independent_sem_of_dark_means": indep_sem,
        "build_record_quoted_mean": 0.008215870005381617,
        "build_record_quoted_sem": 0.006314188695650197,
        "mean_matches_build_record": abs(indep_mean - 0.008215870005381617) < 1e-12,
        "sem_matches_build_record": abs(indep_sem - 0.006314188695650197) < 1e-12,
        "means_read": means,
    }


def check_t0_mean_h() -> dict:
    df = pd.read_csv(IIIB_CSV)
    h_grid = np.sort(df["h"].unique())
    piv = (
        df.pivot(index="event_idx", columns="h", values="combined_no_bh")
        .reindex(columns=h_grid)
    )
    vals = piv.to_numpy(dtype=np.float64)
    logpost = combine_log_likelihood(vals, "physics_floor")
    weights = np.gradient(h_grid)
    lp = logpost - logpost.max()
    post = np.exp(lp)
    norm = float((post * weights).sum())
    post_n = post / norm
    mean_h = float((post_n * h_grid * weights).sum())

    rounded = round(mean_h, 6)
    return {
        "n_h_grid": len(h_grid),
        "n_events": vals.shape[0],
        "computed_mean_h": mean_h,
        "target_display_precision": T0_ANCHOR_DISPLAY,
        "computed_rounded_to_6dp": rounded,
        "rounds_to_display_anchor": rounded == T0_ANCHOR_DISPLAY,
        "literal_abs_diff_from_display_anchor": abs(mean_h - T0_ANCHOR_DISPLAY),
        "literal_1e9_tolerance_satisfiable": abs(mean_h - T0_ANCHOR_DISPLAY) <= T0_TOLERANCE,
        "reproduction_basis": (
            "6-dp display anchor rounds to computed value; a literal 1e-9 abs-diff check against "
            "a 6-dp source is unsatisfiable by construction (display rounding alone can be up to "
            "5e-7) -- same disclosed basis as BUILD_RECORD.md"
        ),
    }


def main() -> None:
    harness = check_harness_byte_id()
    t0 = check_t0_mean_h()

    n_pairs = harness["n_pairs_compared_vs_build_record"] + 1  # +1 for the T0 anchor pair
    max_abs_dev = max(
        harness["max_abs_dev_vs_build_record"],
        t0["literal_abs_diff_from_display_anchor"],
    )

    green = (
        harness["count_green"]
        and harness["resolved_flags_internally_consistent"]
        and harness["bit_for_bit_exact_vs_build_record"]
        and harness["mean_matches_build_record"]
        and harness["sem_matches_build_record"]
        and t0["rounds_to_display_anchor"]
    )

    report = {
        "verdict": "GREEN" if green else "RED",
        "n_pairs": n_pairs,
        "max_abs_dev": max_abs_dev,
        "harness_byte_id": harness,
        "t0_mean_h": t0,
    }
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
