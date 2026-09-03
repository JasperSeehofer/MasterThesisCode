"""Re-derive dark/matched class counts under both criteria across the four named CSVs.

Run-once analysis script for docket ruling R8 (b-dark-class-relative). Reads the four CSVs
named in the ruling, applies both ``is_dark_exact`` and ``is_dark_relative`` from
``dark_class.py``, and prints the comparison tables consumed by BUILD_RECORD.md. Read-only;
writes nothing back into any of the four input files.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from dark_class import THRESHOLD, is_dark_exact, is_dark_relative  # noqa: E402

REPO = Path(__file__).resolve().parents[6]
G1 = REPO / "results/campaign51_20260728/realistic_20260729/graph1_20260901"

FILES: dict[str, tuple[Path, bool]] = {
    "2026-08-27 iiib head readout (then)": (
        REPO
        / "results/campaign51_20260728/realistic_20260729/headreadout_20260827/iiib/event_likelihoods.csv",
        True,
    ),
    "S0-B truth node (now)": (
        G1 / "retrieved/s0b_run_20260902/s0a_seed900101/node_truth_iiib_sites2.2_nosmear/"
        "simulations/diagnostics/event_likelihoods.csv",
        False,
    ),
    "2026-09-02 re-baseline iiib": (
        G1
        / "retrieved/run_20260902_graph1_headrebaseline_iiib/simulations/diagnostics/event_likelihoods.csv",
        True,
    ),
    "2026-09-02 re-baseline joint_r1": (
        G1
        / "retrieved/run_20260902_graph1_headrebaseline_joint_r1/simulations/diagnostics/event_likelihoods.csv",
        True,
    ),
}


def load_h073(path: Path, filter_h: bool) -> pd.DataFrame:
    df = pd.read_csv(path)
    if filter_h:
        df = df[df["h"] == 0.73]
    return df.set_index("event_idx")


def main() -> None:
    results: dict[str, pd.DataFrame] = {}
    print(f"THRESHOLD = {THRESHOLD:.1e}\n")
    print(f"{'file':45s} {'exact-zero dark':>16s} {'relative dark':>14s} {'label differs':>14s}")
    relative_dark_sets: dict[str, set[int]] = {}
    for name, (path, filter_h) in FILES.items():
        df = load_h073(path, filter_h)
        exact = is_dark_exact(df["L_cat_no_bh"].to_numpy())
        relative = is_dark_relative(df["L_cat_no_bh"].to_numpy(), df["combined_no_bh"].to_numpy())
        differs = int((exact != relative).sum())
        results[name] = df
        relative_dark_sets[name] = set(df.index[relative])
        n_matched_exact = len(df) - int(exact.sum())
        n_matched_rel = len(df) - int(relative.sum())
        print(
            f"{name:45s} {int(exact.sum()):6d}/{n_matched_exact:<5d}  "
            f"{int(relative.sum()):6d}/{n_matched_rel:<5d}   {differs:6d}"
        )

    print("\nCross-file agreement of the RELATIVE label (pairwise symmetric difference):")
    names = list(relative_dark_sets)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            sd = relative_dark_sets[a] ^ relative_dark_sets[b]
            print(f"  |{a} Δ {b}| = {len(sd)}")

    # 08-27 vs S0-B: does the relative criterion reproduce 606/982 on the 08-27 file, and
    # does applying the SAME relative criterion to S0-B reproduce the 08-27 relative set?
    then_rel = relative_dark_sets["2026-08-27 iiib head readout (then)"]
    now_rel = relative_dark_sets["S0-B truth node (now)"]
    print(
        f"\nthen(relative) dark={len(then_rel)}  now(relative) dark={len(now_rel)}  "
        f"then\\now={len(then_rel - now_rel)}  now\\then={len(now_rel - then_rel)}"
    )


if __name__ == "__main__":
    main()
