#!/usr/bin/env python3
"""Analyze waveform failures in the injection campaign.

Handles three scenarios:
  A. Injection CSVs available locally -> count successful events, infer failures
  B. SLURM logs available locally -> parse for exception patterns
  C. No data available -> print recommendations

Usage:
    uv run python .gpd/phases/17-injection-physics-audit/analyze_failures.py \
        --data-dir simulations/injections [--log-dir cluster/logs]
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from collections import Counter
from pathlib import Path


def find_injection_csvs(data_dir: str) -> list[str]:
    """Locate injection CSV files matching the naming pattern."""
    pattern = os.path.join(data_dir, "injection_h_*_task_*.csv")
    return sorted(glob.glob(pattern))


def find_slurm_logs(log_dir: str) -> list[str]:
    """Locate SLURM log files (*.out, *.err, *.log)."""
    logs: list[str] = []
    for ext in ("*.out", "*.err", "*.log"):
        logs.extend(glob.glob(os.path.join(log_dir, ext)))
        logs.extend(glob.glob(os.path.join(log_dir, "**", ext), recursive=True))
    return sorted(set(logs))


# Exception patterns to search for in SLURM logs
EXCEPTION_PATTERNS: dict[str, re.Pattern[str]] = {
    "Warning_MassRatio": re.compile(r"Mass ratio.*out of bounds", re.IGNORECASE),
    "Warning_Other": re.compile(r"WARNING.*Continue with new parameters"),
    "ParameterOutOfBoundsError": re.compile(r"ParameterOutOfBoundsError"),
    "RuntimeError": re.compile(r"RuntimeError during waveform"),
    "ValueError_EllipticK": re.compile(r"EllipticK error"),
    "ValueError_Brent": re.compile(r"Brent root solver"),
    "ZeroDivisionError": re.compile(r"ZeroDivisionError"),
    "TimeoutError": re.compile(r"timed out|TimeoutError"),
}

# Pattern for extracting progress info: "{counter} / {iteration} successful"
PROGRESS_PATTERN = re.compile(r"(\d+)\s*/\s*(\d+)\s*successful SNR computations")


def analyze_csvs(csv_files: list[str]) -> str:
    """Analyze injection CSVs: count events per h-value, summarize (z, M) distributions."""
    import numpy as np
    import pandas as pd

    output_lines: list[str] = []
    output_lines.append("### Scenario A: Injection CSV Analysis\n")

    # Group files by h-value
    h_groups: dict[str, list[str]] = {}
    for f in csv_files:
        basename = os.path.basename(f)
        match = re.match(r"injection_h_([^_]+)_task_(\d+)\.csv", basename)
        if match:
            h_label = match.group(1)
            h_groups.setdefault(h_label, []).append(f)

    output_lines.append(f"Found {len(csv_files)} CSV files across {len(h_groups)} h-values.\n")

    summary_rows: list[dict[str, object]] = []

    for h_label in sorted(h_groups.keys()):
        files = h_groups[h_label]
        dfs = [pd.read_csv(f) for f in files]
        combined = pd.concat(dfs, ignore_index=True)

        n_events = len(combined)
        n_tasks = len(files)
        events_per_task = n_events / n_tasks if n_tasks > 0 else 0

        # Detection statistics
        n_detected = int((combined["SNR"] >= 20).sum())
        det_rate = n_detected / n_events * 100 if n_events > 0 else 0

        # z and M ranges
        z_min, z_max = combined["z"].min(), combined["z"].max()
        z_median = combined["z"].median()
        m_min, m_max = combined["M"].min(), combined["M"].max()
        m_median = combined["M"].median()

        h_val = h_label.replace("p", ".")

        summary_rows.append(
            {
                "h": h_val,
                "tasks": n_tasks,
                "events": n_events,
                "events/task": f"{events_per_task:.0f}",
                "detections": n_detected,
                "det_rate_%": f"{det_rate:.2f}",
                "z_median": f"{z_median:.3f}",
                "z_max": f"{z_max:.3f}",
                "M_median": f"{m_median:.0f}",
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    output_lines.append("#### Summary by h-value\n")
    output_lines.append("```")
    output_lines.append(summary_df.to_string(index=False))
    output_lines.append("```\n")

    # Aggregate statistics
    all_dfs = [pd.read_csv(f) for f in csv_files]
    all_data = pd.concat(all_dfs, ignore_index=True)
    total_events = len(all_data)
    total_detected = int((all_data["SNR"] >= 20).sum())

    output_lines.append(f"**Total events across all h-values:** {total_events}")
    output_lines.append(f"**Total detections (SNR >= 20):** {total_detected}")
    output_lines.append(f"**Overall detection rate:** {total_detected / total_events * 100:.3f}%\n")

    # NOTE on failure inference:
    # The CSV records only successful SNR computations. The number of attempts
    # is NOT recorded in the CSV. We cannot compute failure counts from CSV alone.
    # The while loop in injection_campaign() runs until counter == simulation_steps
    # (i.e., the requested number of successes), so each task file should have
    # exactly simulation_steps rows if the campaign completed.
    output_lines.append("#### Failure Inference Limitation\n")
    output_lines.append(
        "The CSV records only successful SNR computations. The number of waveform "
        "generation attempts (including failures) is NOT recorded in the CSV. "
        "Failure rates can only be estimated from SLURM log progress messages "
        "(`{counter} / {iteration} successful SNR computations`).\n"
    )

    # Check for variation in events per task (incomplete tasks)
    task_sizes = [len(pd.read_csv(f)) for f in csv_files]
    unique_sizes = set(task_sizes)
    if len(unique_sizes) > 1:
        size_counter = Counter(task_sizes)
        output_lines.append("#### Task Completion Variation\n")
        output_lines.append(
            "Not all tasks have the same number of events (some may have timed out "
            "or been killed before reaching simulation_steps):\n"
        )
        output_lines.append("```")
        for size, count in sorted(size_counter.items()):
            output_lines.append(f"  {size} events: {count} tasks")
        output_lines.append("```\n")
    else:
        output_lines.append(
            f"All tasks completed with {task_sizes[0]} events each (uniform completion).\n"
        )

    # (z, M) distribution of detections vs non-detections
    detected = all_data[all_data["SNR"] >= 20]
    non_detected = all_data[all_data["SNR"] < 20]

    output_lines.append("#### Detection Distribution\n")
    output_lines.append("Detected events (SNR >= 20):")
    output_lines.append(
        f"  z: median={detected['z'].median():.3f}, "
        f"range=[{detected['z'].min():.3f}, {detected['z'].max():.3f}]"
    )
    output_lines.append(
        f"  M: median={detected['M'].median():.0f}, "
        f"range=[{detected['M'].min():.0f}, {detected['M'].max():.0f}]"
    )
    output_lines.append(
        f"  SNR: median={detected['SNR'].median():.1f}, "
        f"range=[{detected['SNR'].min():.1f}, {detected['SNR'].max():.1f}]\n"
    )

    # Bin detected events by z
    z_bins = np.array([0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0, 1.5])
    z_bin_labels = [f"({z_bins[i]:.2f}, {z_bins[i + 1]:.2f}]" for i in range(len(z_bins) - 1)]
    det_z_hist, _ = np.histogram(detected["z"].values, bins=z_bins)
    all_z_hist, _ = np.histogram(all_data["z"].values, bins=z_bins)

    output_lines.append("#### Detections by Redshift Bin\n")
    output_lines.append("```")
    output_lines.append(f"{'z bin':<18} {'total':>8} {'detected':>10} {'det_rate':>10}")
    output_lines.append("-" * 50)
    for label, total, det in zip(z_bin_labels, all_z_hist, det_z_hist):
        rate = det / total * 100 if total > 0 else 0
        output_lines.append(f"{label:<18} {total:>8} {det:>10} {rate:>9.2f}%")
    output_lines.append("```\n")

    # Bin by log10(M)
    m_bins = np.array([4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0])
    m_bin_labels = [f"({m_bins[i]:.1f}, {m_bins[i + 1]:.1f}]" for i in range(len(m_bins) - 1)]
    log_m_all = np.log10(all_data["M"].values)
    log_m_det = np.log10(detected["M"].values)
    det_m_hist, _ = np.histogram(log_m_det, bins=m_bins)
    all_m_hist, _ = np.histogram(log_m_all, bins=m_bins)

    output_lines.append("#### Detections by log10(M) Bin\n")
    output_lines.append("```")
    output_lines.append(f"{'log10(M) bin':<18} {'total':>8} {'detected':>10} {'det_rate':>10}")
    output_lines.append("-" * 50)
    for label, total, det in zip(m_bin_labels, all_m_hist, det_m_hist):
        rate = det / total * 100 if total > 0 else 0
        output_lines.append(f"{label:<18} {total:>8} {det:>10} {rate:>9.2f}%")
    output_lines.append("```\n")

    return "\n".join(output_lines)


def analyze_slurm_logs(log_files: list[str]) -> str:
    """Parse SLURM logs for exception patterns and progress messages."""
    output_lines: list[str] = []
    output_lines.append("### Scenario B: SLURM Log Analysis\n")
    output_lines.append(f"Found {len(log_files)} log files.\n")

    total_counts: Counter[str] = Counter()
    progress_entries: list[tuple[int, int]] = []

    for log_file in log_files:
        try:
            with open(log_file) as f:
                content = f.read()
        except (OSError, UnicodeDecodeError):
            continue

        for exc_type, pattern in EXCEPTION_PATTERNS.items():
            matches = pattern.findall(content)
            total_counts[exc_type] += len(matches)

        for match in PROGRESS_PATTERN.finditer(content):
            counter_val = int(match.group(1))
            iteration_val = int(match.group(2))
            if iteration_val > 0:
                progress_entries.append((counter_val, iteration_val))

    output_lines.append("#### Exception Counts\n")
    output_lines.append("```")
    for exc_type, count in sorted(total_counts.items(), key=lambda x: -x[1]):
        output_lines.append(f"  {exc_type:<35} {count:>8}")
    output_lines.append("```\n")

    if progress_entries:
        # Use the final progress entry from each log as the best estimate
        successes = sum(c for c, _ in progress_entries)
        attempts = sum(i for _, i in progress_entries)
        if attempts > 0:
            failure_rate = (1 - successes / attempts) * 100
            output_lines.append(
                f"**Estimated failure rate from progress messages:** "
                f"{successes}/{attempts} successful = {failure_rate:.1f}% failure rate\n"
            )
            output_lines.append(
                "_Note: This is approximate. Progress messages are logged periodically, "
                "not after every attempt._\n"
            )

    return "\n".join(output_lines)


def scenario_c_fallback() -> str:
    """No data available — provide recommendations."""
    lines: list[str] = []
    lines.append("### Scenario C: No Data Available\n")
    lines.append("No injection CSV data or SLURM logs were found locally.\n")
    lines.append("**Expected data locations:**")
    lines.append("- Injection CSVs: `simulations/injections/injection_h_*_task_*.csv`")
    lines.append("- SLURM logs: `cluster/logs/` or working directory on bwUniCluster\n")
    lines.append("**To obtain the data:**")
    lines.append("```bash")
    lines.append("# From bwUniCluster:")
    lines.append(
        "rsync -avP bwunicluster:/path/to/workspace/simulations/injections/ simulations/injections/"
    )
    lines.append("rsync -avP bwunicluster:/path/to/workspace/logs/ cluster/logs/")
    lines.append("```\n")
    lines.append("**Fallback analysis:** See the code-level failure characterization in ")
    lines.append("the main report section above. The expected failure-prone parameter regions ")
    lines.append("are: high eccentricity (e0 > 0.5), extreme mass ratios (M near 10^4 or 10^7), ")
    lines.append("and orbital parameters near the separatrix (low p0, high a).\n")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze waveform failures in the injection campaign."
    )
    parser.add_argument(
        "--data-dir",
        default="simulations/injections",
        help="Directory containing injection CSV files (default: simulations/injections)",
    )
    parser.add_argument(
        "--log-dir",
        default="cluster/logs",
        help="Directory containing SLURM log files (default: cluster/logs)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output file for analysis results (default: stdout only)",
    )
    args = parser.parse_args()

    sections: list[str] = []
    sections.append("## Data Analysis Results\n")

    csv_files = find_injection_csvs(args.data_dir)
    log_files = find_slurm_logs(args.log_dir)

    have_csvs = len(csv_files) > 0
    have_logs = len(log_files) > 0

    if have_csvs:
        print(f"[INFO] Found {len(csv_files)} injection CSV files in {args.data_dir}")
        sections.append(analyze_csvs(csv_files))
    else:
        print(f"[INFO] No injection CSV files found in {args.data_dir}")

    if have_logs:
        print(f"[INFO] Found {len(log_files)} SLURM log files in {args.log_dir}")
        sections.append(analyze_slurm_logs(log_files))
    else:
        print(f"[INFO] No SLURM log files found in {args.log_dir}")

    if not have_csvs and not have_logs:
        sections.append(scenario_c_fallback())

    report = "\n".join(sections)
    print("\n" + report)

    if args.output:
        Path(args.output).write_text(report)
        print(f"\n[INFO] Report written to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
