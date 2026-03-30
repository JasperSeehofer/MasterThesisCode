"""Compare two EMRI simulation runs and produce a markdown validation report.

Compares v1.1 (baseline) and v1.2 (new) Cramer-Rao bounds CSVs across detection rate,
SNR distribution, CRB analysis, d_L threshold recommendation, wall time, error analysis,
and pass/fail checks.

Usage:
    python scripts/compare_validation_runs.py \
        --baseline evaluation/run_20260328_seed100_v3 \
        --new evaluation/<v12_run_dir> \
        --output evaluation/<v12_run_dir>/comparison_report.md
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list (defaults to sys.argv[1:] when None).

    Returns:
        Parsed namespace with ``baseline``, ``new``, and ``output`` attributes.
    """
    parser = argparse.ArgumentParser(
        description="Compare two EMRI simulation runs and produce a markdown report.",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        required=True,
        help="Path to v1.1 baseline run directory",
    )
    parser.add_argument(
        "--new",
        type=str,
        required=True,
        help="Path to v1.2 new run directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to write the markdown comparison report",
    )
    parser.add_argument(
        "--snr_threshold",
        type=float,
        default=20.0,
        help="SNR threshold for detection filtering (default: 20.0)",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

CRB_CSV_SUBPATH = "simulations/cramer_rao_bounds.csv"
SNR_THRESHOLD = 20.0  # default, overridden by --snr_threshold CLI arg

# CRB diagonal column names (parameter self-covariance entries)
CRB_DIAGONAL_COLUMNS = [
    "delta_M_delta_M",
    "delta_mu_delta_mu",
    "delta_a_delta_a",
    "delta_p0_delta_p0",
    "delta_e0_delta_e0",
    "delta_x0_delta_x0",
    "delta_luminosity_distance_delta_luminosity_distance",
    "delta_qS_delta_qS",
    "delta_phiS_delta_phiS",
    "delta_qK_delta_qK",
    "delta_phiK_delta_phiK",
    "delta_Phi_phi0_delta_Phi_phi0",
    "delta_Phi_theta0_delta_Phi_theta0",
    "delta_Phi_r0_delta_Phi_r0",
]


def load_crb_csv(run_dir: Path) -> pd.DataFrame:
    """Load Cramer-Rao bounds CSV from a run directory.

    Args:
        run_dir: Path to the simulation run directory.

    Returns:
        DataFrame with CRB data.

    Raises:
        SystemExit: If CSV file is not found.
    """
    csv_path = run_dir / CRB_CSV_SUBPATH
    if not csv_path.exists():
        print(f"Error: CRB CSV not found at {csv_path}", file=sys.stderr)
        sys.exit(1)
    return pd.read_csv(csv_path)


def load_metadata(run_dir: Path) -> list[dict[str, object]]:
    """Load all run_metadata_*.json files from a run directory.

    Args:
        run_dir: Path to the simulation run directory.

    Returns:
        List of metadata dicts, sorted by filename.
    """
    metadata_files = sorted(run_dir.glob("run_metadata_*.json"))
    results: list[dict[str, object]] = []
    for f in metadata_files:
        with open(f) as fh:
            results.append(json.load(fh))
    return results


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------


def detected_events(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to events with SNR >= threshold.

    Args:
        df: Full CRB DataFrame.

    Returns:
        Filtered DataFrame containing only detected events.
    """
    return df[df["SNR"] >= SNR_THRESHOLD]


def fractional_dl_error(df: pd.DataFrame) -> pd.Series:
    """Compute fractional luminosity distance error for each event.

    Formula: sqrt(abs(delta_luminosity_distance_delta_luminosity_distance)) / luminosity_distance

    Args:
        df: DataFrame with detected events.

    Returns:
        Series of fractional d_L errors.
    """
    sigma_dl = np.sqrt(np.abs(df["delta_luminosity_distance_delta_luminosity_distance"]))
    return sigma_dl / df["luminosity_distance"]


def summary_stats(series: pd.Series) -> dict[str, float]:
    """Compute summary statistics for a numeric series.

    Args:
        series: Numeric pandas Series.

    Returns:
        Dictionary with min, median, mean, max, std.
    """
    return {
        "min": float(series.min()),
        "median": float(series.median()),
        "mean": float(series.mean()),
        "max": float(series.max()),
        "std": float(series.std()),
    }


def percentile_table(series: pd.Series) -> dict[str, float]:
    """Compute percentiles for a numeric series.

    Args:
        series: Numeric pandas Series.

    Returns:
        Dictionary mapping percentile labels to values.
    """
    pcts = [10, 25, 50, 75, 90, 95, 99]
    values = np.percentile(series.dropna().values, pcts)
    return {f"P{p}": float(v) for p, v in zip(pcts, values)}


def count_negative_diagonals(df: pd.DataFrame) -> int:
    """Count events with any negative CRB diagonal entry.

    Args:
        df: CRB DataFrame.

    Returns:
        Number of events with at least one negative diagonal.
    """
    cols = [c for c in CRB_DIAGONAL_COLUMNS if c in df.columns]
    if not cols:
        return 0
    return int((df[cols] < 0).any(axis=1).sum())


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def fmt(value: float, decimals: int = 6) -> str:
    """Format a float for markdown tables.

    Args:
        value: Number to format.
        decimals: Number of decimal places.

    Returns:
        Formatted string.
    """
    return f"{value:.{decimals}f}"


def section_metadata(
    baseline_meta: list[dict[str, object]],
    new_meta: list[dict[str, object]],
) -> str:
    """Generate Section 1: Run Metadata.

    Args:
        baseline_meta: Metadata dicts for baseline run.
        new_meta: Metadata dicts for new run.

    Returns:
        Markdown string for the metadata section.
    """
    lines = ["## 1. Run Metadata", ""]

    def extract(meta_list: list[dict[str, object]]) -> dict[str, str]:
        if not meta_list:
            return {"git_commit": "N/A", "seed": "N/A", "timestamp": "N/A", "tasks": "0"}
        first = meta_list[0]
        cli = first.get("cli_args", {})
        seed = (
            cli.get("random_seed", first.get("random_seed", "N/A"))
            if isinstance(cli, dict)
            else first.get("random_seed", "N/A")
        )  # noqa: E501
        return {
            "git_commit": str(first.get("git_commit", "N/A"))[:12],
            "seed": str(seed),
            "timestamp": str(first.get("timestamp", "N/A")),
            "tasks": str(len(meta_list)),
        }

    b = extract(baseline_meta)
    n = extract(new_meta)

    lines.append("| Attribute | Baseline (v1.1) | New (v1.2) |")
    lines.append("|-----------|----------------|------------|")
    lines.append(f"| Git commit | `{b['git_commit']}` | `{n['git_commit']}` |")
    lines.append(f"| Seed | {b['seed']} | {n['seed']} |")
    lines.append(f"| Timestamp | {b['timestamp']} | {n['timestamp']} |")
    lines.append(f"| Tasks | {b['tasks']} | {n['tasks']} |")
    lines.append("")
    return "\n".join(lines)


def section_detection_rate(baseline: pd.DataFrame, new: pd.DataFrame) -> str:
    """Generate Section 2: Detection Rate.

    Args:
        baseline: Baseline CRB DataFrame.
        new: New CRB DataFrame.

    Returns:
        Markdown string for the detection rate section.
    """
    b_total = len(baseline)
    b_det = len(detected_events(baseline))
    n_total = len(new)
    n_det = len(detected_events(new))

    lines = ["## 2. Detection Rate", ""]
    lines.append("| Metric | Baseline (v1.1) | New (v1.2) |")
    lines.append("|--------|----------------|------------|")
    lines.append(f"| Total events | {b_total} | {n_total} |")
    lines.append(f"| Detections (SNR >= {SNR_THRESHOLD:.0f}) | {b_det} | {n_det} |")
    b_rate = f"{b_det / b_total:.4f}" if b_total > 0 else "N/A"
    n_rate = f"{n_det / n_total:.4f}" if n_total > 0 else "N/A"
    lines.append(f"| Detection rate | {b_rate} | {n_rate} |")
    lines.append("")
    return "\n".join(lines)


def section_snr_distribution(baseline: pd.DataFrame, new: pd.DataFrame) -> str:
    """Generate Section 3: SNR Distribution.

    Args:
        baseline: Baseline CRB DataFrame (full, not filtered).
        new: New CRB DataFrame (full, not filtered).

    Returns:
        Markdown string for the SNR distribution section.
    """
    b_det = detected_events(baseline)
    n_det = detected_events(new)

    lines = ["## 3. SNR Distribution (detected events only)", ""]
    lines.append("| Statistic | Baseline (v1.1) | New (v1.2) |")
    lines.append("|-----------|----------------|------------|")

    if len(b_det) == 0 and len(n_det) == 0:
        lines.append("| (no detections) | N/A | N/A |")
    else:
        b_stats = summary_stats(b_det["SNR"]) if len(b_det) > 0 else {}
        n_stats = summary_stats(n_det["SNR"]) if len(n_det) > 0 else {}
        for key in ["min", "median", "mean", "max", "std"]:
            b_val = fmt(b_stats[key], 2) if key in b_stats else "N/A"
            n_val = fmt(n_stats[key], 2) if key in n_stats else "N/A"
            lines.append(f"| {key} | {b_val} | {n_val} |")

    lines.append("")
    return "\n".join(lines)


def section_crb_analysis(baseline: pd.DataFrame, new: pd.DataFrame) -> str:
    """Generate Section 4: CRB Analysis.

    Args:
        baseline: Baseline CRB DataFrame.
        new: New CRB DataFrame.

    Returns:
        Markdown string for the CRB analysis section.
    """
    b_det = detected_events(baseline)
    n_det = detected_events(new)

    lines = ["## 4. CRB Analysis", ""]
    lines.extend(["### Fractional d_L error: sqrt(|sigma^2(d_L)|) / d_L", ""])

    # Summary stats
    lines.append("| Statistic | Baseline (v1.1) | New (v1.2) |")
    lines.append("|-----------|----------------|------------|")

    b_frac = fractional_dl_error(b_det) if len(b_det) > 0 else pd.Series(dtype=float)
    n_frac = fractional_dl_error(n_det) if len(n_det) > 0 else pd.Series(dtype=float)

    b_stats = summary_stats(b_frac) if len(b_frac) > 0 else {}
    n_stats = summary_stats(n_frac) if len(n_frac) > 0 else {}

    for key in ["min", "median", "mean", "max", "std"]:
        b_val = fmt(b_stats[key]) if key in b_stats else "N/A"
        n_val = fmt(n_stats[key]) if key in n_stats else "N/A"
        lines.append(f"| {key} | {b_val} | {n_val} |")

    lines.append("")

    # Percentiles
    lines.extend(["### Percentiles", ""])
    lines.append("| Percentile | Baseline (v1.1) | New (v1.2) |")
    lines.append("|------------|----------------|------------|")

    b_pct = percentile_table(b_frac) if len(b_frac) > 0 else {}
    n_pct = percentile_table(n_frac) if len(n_frac) > 0 else {}

    for p in ["P10", "P25", "P50", "P75", "P90", "P95", "P99"]:
        b_val = fmt(b_pct[p]) if p in b_pct else "N/A"
        n_val = fmt(n_pct[p]) if p in n_pct else "N/A"
        lines.append(f"| {p} | {b_val} | {n_val} |")

    lines.append("")
    lines.append("> **Note:** Fisher matrix condition numbers are logged in SLURM output, not CSV.")
    lines.append("")
    return "\n".join(lines)


def section_threshold_recommendation(new: pd.DataFrame) -> str:
    """Generate Section 5: d_L Threshold Recommendation.

    Args:
        new: New CRB DataFrame.

    Returns:
        Markdown string for the threshold recommendation section.
    """
    n_det = detected_events(new)

    lines = ["## 5. d_L Threshold Recommendation", ""]

    if len(n_det) == 0:
        lines.append("No detections in new run -- cannot recommend threshold.")
        lines.append("")
        return "\n".join(lines)

    n_frac = fractional_dl_error(n_det)
    pct = percentile_table(n_frac)

    lines.extend(["### Full percentile table (new run only)", ""])
    lines.append("| Percentile | Fractional d_L error |")
    lines.append("|------------|---------------------|")
    for p_label, p_val in pct.items():
        lines.append(f"| {p_label} | {fmt(p_val)} |")

    p90 = pct["P90"]
    lines.append("")
    lines.append(f"**Recommended threshold (P90):** {fmt(p90)}")
    lines.append("")
    lines.append(
        "> **D-07 Note:** Recommendation only -- do NOT update `constants.py` in this phase."
    )
    lines.append("")
    return "\n".join(lines)


def section_wall_time(baseline: pd.DataFrame, new: pd.DataFrame) -> str:
    """Generate Section 6: Wall Time Analysis.

    Args:
        baseline: Baseline CRB DataFrame.
        new: New CRB DataFrame.

    Returns:
        Markdown string for the wall time section.
    """
    lines = ["## 6. Wall Time Analysis", ""]
    lines.extend(["### Per-event generation time (seconds)", ""])
    lines.append("| Statistic | Baseline (v1.1) | New (v1.2) |")
    lines.append("|-----------|----------------|------------|")

    b_stats = summary_stats(baseline["generation_time"]) if "generation_time" in baseline else {}
    n_stats = summary_stats(new["generation_time"]) if "generation_time" in new else {}

    for key in ["min", "median", "mean", "max", "std"]:
        b_val = fmt(b_stats[key], 3) if key in b_stats else "N/A"
        n_val = fmt(n_stats[key], 3) if key in n_stats else "N/A"
        lines.append(f"| {key} | {b_val} | {n_val} |")

    lines.append("")
    lines.append("> **Note:** Total wall time per SLURM task is in job logs. Check `sacct` output.")
    lines.append("")
    return "\n".join(lines)


def section_error_analysis(baseline: pd.DataFrame, new: pd.DataFrame) -> str:
    """Generate Section 7: Error Analysis.

    Args:
        baseline: Baseline CRB DataFrame.
        new: New CRB DataFrame.

    Returns:
        Markdown string for the error analysis section.
    """
    lines = ["## 7. Error Analysis", ""]
    lines.append("| Check | Baseline (v1.1) | New (v1.2) |")
    lines.append("|-------|----------------|------------|")

    b_nan = int(baseline["SNR"].isna().sum())
    n_nan = int(new["SNR"].isna().sum())
    lines.append(f"| NaN in SNR | {b_nan} | {n_nan} |")

    b_neg = count_negative_diagonals(baseline)
    n_neg = count_negative_diagonals(new)
    lines.append(f"| Events with negative CRB diagonal | {b_neg} | {n_neg} |")

    lines.append("| Timeout hit rate | See SLURM logs | See SLURM logs |")
    lines.append("| OOB error rate | See SLURM logs | See SLURM logs |")
    lines.append("")
    return "\n".join(lines)


def section_pass_fail(baseline: pd.DataFrame, new: pd.DataFrame) -> str:
    """Generate Section 8: Pass/Fail Summary.

    Args:
        baseline: Baseline CRB DataFrame.
        new: New CRB DataFrame.

    Returns:
        Markdown string for the pass/fail summary section.
    """
    b_det = detected_events(baseline)
    n_det = detected_events(new)

    lines = ["## 8. Pass/Fail Summary", ""]
    lines.append("| # | Check | Result | Detail |")
    lines.append("|---|-------|--------|--------|")

    # Check 1: Confusion noise should produce lower SNRs
    if len(n_det) == 0:
        lines.append("| 1 | Confusion noise lowers SNR | N/A | No detections in new run |")
    else:
        b_median = float(b_det["SNR"].median()) if len(b_det) > 0 else float("inf")
        n_median = float(n_det["SNR"].median())
        result_1 = "PASS" if n_median < b_median else "FAIL"
        lines.append(
            f"| 1 | Confusion noise lowers SNR | {result_1} "
            f"| Median: {fmt(b_median, 2)} -> {fmt(n_median, 2)} |"
        )

    # Check 2: 5-point stencil produces different CRBs
    if len(n_det) == 0:
        lines.append("| 2 | Stencil changes CRBs | N/A | No detections in new run |")
    else:
        b_frac_median = float(fractional_dl_error(b_det).median()) if len(b_det) > 0 else 0.0
        n_frac_median = float(fractional_dl_error(n_det).median())
        result_2 = "PASS" if abs(b_frac_median - n_frac_median) > 1e-10 else "FAIL"
        lines.append(
            f"| 2 | Stencil changes CRBs | {result_2} "
            f"| Median frac d_L: {fmt(b_frac_median)} -> {fmt(n_frac_median)} |"
        )

    # Check 3: Detection rate must remain > 0
    result_3 = "PASS" if len(n_det) > 0 else "FAIL"
    lines.append(f"| 3 | Detection rate > 0 | {result_3} | {len(n_det)} detections |")

    # Check 4: No NaN in SNR
    n_nan = int(new["SNR"].isna().sum())
    result_4 = "PASS" if n_nan == 0 else "FAIL"
    lines.append(f"| 4 | No NaN in SNR | {result_4} | {n_nan} NaN values |")

    # Check 5: No negative CRB diagonals
    n_neg = count_negative_diagonals(new)
    result_5 = "PASS" if n_neg == 0 else "FAIL"
    lines.append(f"| 5 | No negative CRB diags | {result_5} | {n_neg} events with negatives |")

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def generate_report(baseline_dir: Path, new_dir: Path) -> str:
    """Generate the full comparison report as a markdown string.

    Args:
        baseline_dir: Path to baseline run directory.
        new_dir: Path to new run directory.

    Returns:
        Full markdown report string.
    """
    baseline_df = load_crb_csv(baseline_dir)
    new_df = load_crb_csv(new_dir)
    baseline_meta = load_metadata(baseline_dir)
    new_meta = load_metadata(new_dir)

    sections = [
        "# Validation Comparison Report",
        "",
        f"Baseline: `{baseline_dir}`",
        f"New: `{new_dir}`",
        "",
        section_metadata(baseline_meta, new_meta),
        section_detection_rate(baseline_df, new_df),
        section_snr_distribution(baseline_df, new_df),
        section_crb_analysis(baseline_df, new_df),
        section_threshold_recommendation(new_df),
        section_wall_time(baseline_df, new_df),
        section_error_analysis(baseline_df, new_df),
        section_pass_fail(baseline_df, new_df),
    ]

    return "\n".join(sections)


def main(argv: list[str] | None = None) -> None:
    """Entry point for the comparison script.

    Args:
        argv: Argument list (defaults to sys.argv[1:] when None).
    """
    args = parse_args(argv)
    baseline_dir = Path(args.baseline)
    new_dir = Path(args.new)
    output_path = Path(args.output)

    global SNR_THRESHOLD  # noqa: PLW0603
    SNR_THRESHOLD = args.snr_threshold

    report = generate_report(baseline_dir, new_dir)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report)
    print(f"Report written to {output_path}")


if __name__ == "__main__":
    main()
