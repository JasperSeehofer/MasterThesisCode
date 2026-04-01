#!/usr/bin/env python3
"""Injection campaign yield analysis.

Loads all injection CSVs, computes per-h detection yield, waste breakdown,
z-cutoff validation, and Farr (2019) criterion check.

Conventions:
  - SI units: distances in Gpc, masses in solar masses, h dimensionless
  - flat LambdaCDM: Omega_m=0.25, Omega_DE=0.75, H=0.73
  - SNR_THRESHOLD = 15 (from constants.py)
  - h = H0 / (100 km/s/Mpc), dimensionless

Usage:
    python -m analysis.injection_yield [--csv-dir simulations/injections]
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SNR_THRESHOLD: float = 15.0
Z_CUTOFF: float = 0.5

# h-values used in the injection campaign (ordered)
H_VALUES: list[float] = [0.60, 0.65, 0.70, 0.73, 0.80, 0.85, 0.90]

# Map filename labels to numeric h-values
H_LABEL_MAP: dict[str, float] = {
    "0p6": 0.60,
    "0p65": 0.65,
    "0p7": 0.70,
    "0p73": 0.73,
    "0p8": 0.80,
    "0p85": 0.85,
    "0p9": 0.90,
}

# Filename regex: injection_h_{label}_task_{n}.csv
_FILENAME_RE = re.compile(r"injection_h_([0-9p]+)_task_(\d+)\.csv")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_injection_data(
    csv_dir: Path,
) -> dict[float, pd.DataFrame]:
    """Load all injection CSVs grouped by h-value.

    Returns
    -------
    dict mapping h-value (float) -> concatenated DataFrame of all tasks.
    """
    groups: dict[float, list[pd.DataFrame]] = {h: [] for h in H_VALUES}
    file_count = 0

    for csv_path in sorted(csv_dir.glob("injection_h_*_task_*.csv")):
        m = _FILENAME_RE.match(csv_path.name)
        if m is None:
            continue
        label = m.group(1)
        h_val = H_LABEL_MAP.get(label)
        if h_val is None:
            print(f"WARNING: Unknown h-label '{label}' in {csv_path.name}")
            continue
        df = pd.read_csv(csv_path)
        groups[h_val].append(df)
        file_count += 1

    print(f"Loaded {file_count} CSV files")

    result: dict[float, pd.DataFrame] = {}
    for h_val in H_VALUES:
        frames = groups[h_val]
        if frames:
            result[h_val] = pd.concat(frames, ignore_index=True)
        else:
            print(f"WARNING: No data for h={h_val}")

    return result


# ---------------------------------------------------------------------------
# Detection yield
# ---------------------------------------------------------------------------


def compute_yield(
    data: dict[float, pd.DataFrame],
) -> pd.DataFrame:
    """Compute detection yield per h-value.

    Returns DataFrame with columns:
        h, N_total, N_det, N_sub_threshold, f_det, max_z_det
    """
    rows = []
    for h_val in H_VALUES:
        if h_val not in data:
            continue
        df = data[h_val]
        n_total = len(df)
        detected_mask = df["SNR"] >= SNR_THRESHOLD
        n_det = int(detected_mask.sum())
        n_sub = n_total - n_det
        f_det = n_det / n_total if n_total > 0 else 0.0

        # Max detected redshift
        if n_det > 0:
            max_z_det = float(df.loc[detected_mask, "z"].max())
        else:
            max_z_det = float("nan")

        rows.append(
            {
                "h": h_val,
                "N_total": n_total,
                "N_det": n_det,
                "N_sub_threshold": n_sub,
                "f_det": f_det,
                "max_z_det": max_z_det,
            }
        )

    yield_df = pd.DataFrame(rows)

    # Verification: integer accounting
    for _, row in yield_df.iterrows():
        assert row["N_det"] + row["N_sub_threshold"] == row["N_total"], (
            f"Integer accounting failed for h={row['h']}"
        )

    return yield_df


# ---------------------------------------------------------------------------
# Waste breakdown
# ---------------------------------------------------------------------------


def compute_waste(
    yield_df: pd.DataFrame,
    failure_rates: list[float] | None = None,
) -> dict[str, pd.DataFrame]:
    """Compute waste breakdown.

    Returns dict with keys:
        "csv_only" -> 2-way (exact from CSV)
        "estimated_30pct" -> 3-way at 30% failure rate
        "estimated_50pct" -> 3-way at 50% failure rate
    """
    if failure_rates is None:
        failure_rates = [0.30, 0.50]

    result: dict[str, pd.DataFrame] = {}

    # CSV-only 2-way decomposition
    csv_rows = []
    for _, row in yield_df.iterrows():
        n_total = row["N_total"]
        frac_det = row["N_det"] / n_total if n_total > 0 else 0.0
        frac_sub = row["N_sub_threshold"] / n_total if n_total > 0 else 0.0
        csv_rows.append(
            {
                "h": row["h"],
                "frac_detected": frac_det,
                "frac_sub_threshold": frac_sub,
            }
        )

    csv_df = pd.DataFrame(csv_rows)

    # Verify 2-way sums to 1
    for _, row in csv_df.iterrows():
        total = row["frac_detected"] + row["frac_sub_threshold"]
        assert abs(total - 1.0) < 1e-12, f"CSV 2-way fractions sum to {total} for h={row['h']}"

    result["csv_only"] = csv_df

    # Estimated 3-way decompositions
    for fr in failure_rates:
        est_rows = []
        for _, row in yield_df.iterrows():
            n_csv = row["N_total"]
            # N_attempted = N_csv / (1 - failure_rate)
            n_attempted = n_csv / (1.0 - fr)
            frac_failed = fr
            frac_sub = row["N_sub_threshold"] / n_attempted
            frac_det = row["N_det"] / n_attempted
            est_rows.append(
                {
                    "h": row["h"],
                    "N_attempted_est": int(round(n_attempted)),
                    "frac_failed": frac_failed,
                    "frac_sub_threshold": frac_sub,
                    "frac_detected": frac_det,
                }
            )
        est_df = pd.DataFrame(est_rows)

        # Verify 3-way sums to 1
        for _, row in est_df.iterrows():
            total = row["frac_failed"] + row["frac_sub_threshold"] + row["frac_detected"]
            assert abs(total - 1.0) < 1e-10, (
                f"3-way fractions sum to {total} for h={row['h']} at {fr * 100}% failure"
            )

        key = f"estimated_{int(fr * 100)}pct"
        result[key] = est_df

    return result


# ---------------------------------------------------------------------------
# z-cutoff validation
# ---------------------------------------------------------------------------


def validate_zcutoff(
    data: dict[float, pd.DataFrame],
) -> pd.DataFrame:
    """Check for detections above z > Z_CUTOFF.

    Returns DataFrame with columns: h, n_above_zcutoff_detected, max_z_det
    """
    rows = []
    for h_val in H_VALUES:
        if h_val not in data:
            continue
        df = data[h_val]
        above_z = df[(df["z"] > Z_CUTOFF) & (df["SNR"] >= SNR_THRESHOLD)]
        n_above = len(above_z)

        detected = df[df["SNR"] >= SNR_THRESHOLD]
        max_z = float(detected["z"].max()) if len(detected) > 0 else float("nan")

        rows.append(
            {
                "h": h_val,
                "n_above_zcutoff_detected": n_above,
                "max_z_det": max_z,
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Farr (2019) criterion
# ---------------------------------------------------------------------------


def check_farr_criterion(
    yield_df: pd.DataFrame,
) -> pd.DataFrame:
    """Check Farr (2019) N_eff > 4*N_det criterion.

    With uniform injection weights, N_eff = N_total.
    Returns DataFrame with: h, N_total, N_det, ratio, passes
    """
    rows = []
    for _, row in yield_df.iterrows():
        n_total = row["N_total"]
        n_det = row["N_det"]
        ratio = n_total / n_det if n_det > 0 else float("inf")
        passes = ratio > 4.0
        rows.append(
            {
                "h": row["h"],
                "N_total": int(n_total),
                "N_det": int(n_det),
                "ratio_N_total_over_N_det": ratio,
                "passes_farr": passes,
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Waste visualization
# ---------------------------------------------------------------------------


def plot_waste_breakdown(
    yield_df: pd.DataFrame,
    waste: dict[str, pd.DataFrame],
    output_path: Path,
) -> None:
    """Create stacked bar chart of waste breakdown + detection fraction inset."""
    try:
        from master_thesis_code.plotting._style import apply_style

        apply_style()
    except ImportError:
        pass

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [2, 1]})

    # --- Left panel: stacked bar chart (30% failure scenario) ---
    est_df = waste["estimated_30pct"]
    h_labels = [f"{h:.2f}" for h in est_df["h"]]
    x = np.arange(len(h_labels))
    bar_width = 0.6

    # Stack order: failures (bottom), sub-threshold (middle), detected (top)
    frac_failed = est_df["frac_failed"].values
    frac_sub = est_df["frac_sub_threshold"].values
    frac_det = est_df["frac_detected"].values

    ax1.bar(x, frac_failed, bar_width, label="Waveform failures (est. 30%)", color="#d62728")
    ax1.bar(
        x,
        frac_sub,
        bar_width,
        bottom=frac_failed,
        label="Sub-threshold (SNR < 15)",
        color="#ff7f0e",
    )
    ax1.bar(
        x,
        frac_det,
        bar_width,
        bottom=frac_failed + frac_sub,
        label="Detected (SNR >= 15)",
        color="#2ca02c",
    )

    ax1.set_xlabel("Hubble parameter $h$")
    ax1.set_ylabel("Fraction of GPU compute")
    ax1.set_title("Injection campaign compute breakdown\n(estimated 30% failure rate)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(h_labels)
    ax1.set_ylim(0, 1.05)
    ax1.legend(loc="upper left", fontsize=8)

    # --- Right panel: detection fraction vs h ---
    ax2.plot(yield_df["h"], yield_df["f_det"] * 100, "o-", color="#2ca02c", markersize=6)
    ax2.set_xlabel("Hubble parameter $h$")
    ax2.set_ylabel("Detection fraction [%]")
    ax2.set_title("Detection yield vs $h$")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Figure saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(csv_dir: Path, figure_path: Path | None = None) -> None:
    """Run full injection yield analysis."""
    print("=" * 70)
    print("INJECTION CAMPAIGN YIELD ANALYSIS")
    print(f"SNR threshold: {SNR_THRESHOLD}")
    print(f"z cutoff: {Z_CUTOFF}")
    print("=" * 70)

    # 1. Load data
    print("\n--- Loading injection data ---")
    data = load_injection_data(csv_dir)
    total_events = sum(len(df) for df in data.values())
    print(f"Total events across all h: {total_events}")

    # 2. Compute yield
    print("\n--- Detection yield per h-value ---")
    yield_df = compute_yield(data)
    total_det = int(yield_df["N_det"].sum())
    print(yield_df.to_string(index=False))
    print(f"\nTotal detections (SNR >= {SNR_THRESHOLD}): {total_det}")
    print(f"Overall detection fraction: {total_det / total_events:.6f}")

    # 3. Waste breakdown
    print("\n--- CSV-only waste decomposition (2-way, exact) ---")
    waste = compute_waste(yield_df)
    print(waste["csv_only"].to_string(index=False))

    print("\n--- Estimated waste decomposition (3-way, 30% failure rate) ---")
    print(waste["estimated_30pct"].to_string(index=False))

    print("\n--- Estimated waste decomposition (3-way, 50% failure rate) ---")
    print(waste["estimated_50pct"].to_string(index=False))

    # 4. z-cutoff validation
    print("\n--- z > 0.5 cutoff validation ---")
    zcutoff_df = validate_zcutoff(data)
    print(zcutoff_df.to_string(index=False))
    n_above = int(zcutoff_df["n_above_zcutoff_detected"].sum())
    print(f"\nTotal detections above z > {Z_CUTOFF}: {n_above}")
    if n_above == 0:
        print("CONFIRMED: Zero detections above z > 0.5 for all h-values")
    else:
        print(f"WARNING: {n_above} detections above z > 0.5 -- INVESTIGATE")

    # 5. Farr criterion
    print("\n--- Farr (2019) criterion: N_total / N_det > 4 ---")
    farr_df = check_farr_criterion(yield_df)
    print(farr_df.to_string(index=False))
    all_pass = farr_df["passes_farr"].all()
    print(f"\nAll h-values pass Farr criterion: {all_pass}")

    # 6. Figure
    if figure_path is not None:
        print("\n--- Generating waste breakdown figure ---")
        plot_waste_breakdown(yield_df, waste, figure_path)


def detection_fraction(data: dict[float, pd.DataFrame]) -> pd.DataFrame:
    """Convenience wrapper returning yield DataFrame."""
    return compute_yield(data)


def waste_breakdown(yield_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Convenience wrapper returning waste dict."""
    return compute_waste(yield_df)


def zcutoff_check(data: dict[float, pd.DataFrame]) -> pd.DataFrame:
    """Convenience wrapper returning z-cutoff validation."""
    return validate_zcutoff(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Injection campaign yield analysis")
    parser.add_argument(
        "--csv-dir",
        type=Path,
        default=Path("simulations/injections"),
        help="Directory containing injection CSVs",
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=Path("figures/injection_yield_waste_breakdown.pdf"),
        help="Output path for waste breakdown figure",
    )
    args = parser.parse_args()
    main(args.csv_dir, args.figure)
