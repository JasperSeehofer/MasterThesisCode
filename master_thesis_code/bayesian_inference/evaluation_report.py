"""Evaluation report module for baseline extraction and comparison.

Provides tools to extract a baseline H0 posterior snapshot from an h-sweep,
compute credible intervals, and generate before/after comparison reports.

Used by Phases 31-34 to measure the effect of each fix on the H0 posterior.
"""

import datetime
import json
import logging
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.integrate import cumulative_trapezoid

_LOGGER = logging.getLogger(__name__)


def _get_git_commit_safe() -> str:
    """Return current git commit hash, or 'unknown' if not available."""
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


@dataclass
class BaselineSnapshot:
    """Snapshot of H0 posterior metrics extracted from a posteriors directory.

    Args:
        map_h: MAP (maximum a posteriori) Hubble constant value.
        ci_lower: Lower bound of the 68% credible interval.
        ci_upper: Upper bound of the 68% credible interval.
        ci_width: Width of the 68% credible interval (ci_upper - ci_lower).
        bias_percent: Relative bias as (MAP - true_h) / true_h * 100.
        n_events: Number of detection events contributing to the posterior.
        h_values: Sorted list of h values in the sweep.
        log_posteriors: Log-posterior values at each h.
        per_event_summaries: Per-event diagnostic data (d_L, SNR, etc.).
        created_at: ISO 8601 timestamp of when this snapshot was created.
        git_commit: Git commit hash at time of creation.
    """

    map_h: float
    ci_lower: float
    ci_upper: float
    ci_width: float
    bias_percent: float
    n_events: int
    h_values: list[float] = field(default_factory=list)
    log_posteriors: list[float] = field(default_factory=list)
    per_event_summaries: list[dict[str, float]] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.datetime.now(datetime.UTC).isoformat() + "Z")
    git_commit: str = field(default_factory=_get_git_commit_safe)

    def to_json(self) -> dict[str, object]:
        """Serialize to a JSON-compatible dictionary.

        Returns:
            Dictionary representation suitable for json.dumps.
        """
        return {
            "map_h": self.map_h,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "ci_width": self.ci_width,
            "bias_percent": self.bias_percent,
            "n_events": self.n_events,
            "h_values": self.h_values,
            "log_posteriors": self.log_posteriors,
            "per_event_summaries": self.per_event_summaries,
            "created_at": self.created_at,
            "git_commit": self.git_commit,
        }

    @classmethod
    def from_json(cls, data: dict[str, object]) -> "BaselineSnapshot":
        """Deserialize from a JSON-compatible dictionary.

        Args:
            data: Dictionary as produced by to_json().

        Returns:
            A new BaselineSnapshot instance.
        """
        return cls(
            map_h=float(data["map_h"]),  # type: ignore[arg-type]
            ci_lower=float(data["ci_lower"]),  # type: ignore[arg-type]
            ci_upper=float(data["ci_upper"]),  # type: ignore[arg-type]
            ci_width=float(data["ci_width"]),  # type: ignore[arg-type]
            bias_percent=float(data["bias_percent"]),  # type: ignore[arg-type]
            n_events=int(data["n_events"]),  # type: ignore[arg-type]
            h_values=list(data.get("h_values", [])),  # type: ignore[arg-type]
            log_posteriors=list(data.get("log_posteriors", [])),  # type: ignore[arg-type]
            per_event_summaries=list(data.get("per_event_summaries", [])),  # type: ignore[arg-type]
            created_at=str(data.get("created_at", "")),
            git_commit=str(data.get("git_commit", "unknown")),
        )


def load_posteriors(posteriors_dir: Path) -> list[dict[str, float]]:
    """Load h_*.json posterior files from a directory.

    Each file should contain a dict with key "h" (float) and integer-string
    keys mapping to [likelihood_value] arrays for each detection event.

    Args:
        posteriors_dir: Path to directory containing h_*.json files.

    Returns:
        List of dicts with keys: "h", "log_posterior", "n_detections".
        Sorted by h value in ascending order.
    """
    files = sorted(posteriors_dir.glob("h_*.json"))

    if len(files) > 100:
        _LOGGER.warning(
            "Found %d posterior files in %s. Loading all may be slow. "
            "Consider pre-filtering if this is unexpected.",
            len(files),
            posteriors_dir,
        )

    results: list[dict[str, float]] = []
    for path in files:
        data: dict[str, object] = json.loads(path.read_text())
        h = float(data["h"])  # type: ignore[arg-type]

        # Collect detection keys: all keys except "h"
        detection_keys = [k for k in data if k != "h"]
        n_detections = len(detection_keys)

        # Compute log_posterior = sum of log-likelihoods across events
        log_posterior = 0.0
        for key in detection_keys:
            likelihood_list = data[key]
            if isinstance(likelihood_list, list) and len(likelihood_list) > 0:
                lk = float(likelihood_list[0])
                if lk > 0:
                    log_posterior += np.log(lk)
                # If likelihood is 0 or negative, skip (numerical floor)

        results.append(
            {
                "h": h,
                "log_posterior": log_posterior,
                "n_detections": float(n_detections),
            }
        )

    results.sort(key=lambda r: r["h"])
    return results


def compute_credible_interval(
    h_values: list[float],
    log_posteriors: list[float],
    level: float = 0.68,
) -> tuple[float, float]:
    """Compute a credible interval from log-posterior values.

    Converts log-posteriors to a normalized probability distribution, then
    computes the CDF and finds where it crosses (1-level)/2 and (1+level)/2.

    Args:
        h_values: Sorted list of h values.
        log_posteriors: Log-posterior value at each h.
        level: Credible interval level (default 0.68 for 68% CI).

    Returns:
        Tuple (lower, upper) bounding the credible interval.
    """
    h_arr = np.array(h_values, dtype=np.float64)
    lp_arr = np.array(log_posteriors, dtype=np.float64)

    # Normalize: subtract max for numerical stability, then exponentiate
    prob = np.exp(lp_arr - lp_arr.max())

    # Normalize via trapezoid integration (np.trapezoid since NumPy 2.0)
    norm = float(np.trapezoid(prob, x=h_arr))
    if norm <= 0:
        norm = 1.0
    prob = prob / norm

    # CDF via cumulative trapezoid
    cdf_vals = cumulative_trapezoid(prob, x=h_arr, initial=0.0)

    lower_target = (1.0 - level) / 2.0
    upper_target = (1.0 + level) / 2.0

    # Linear interpolation to find crossing points
    lower = float(np.interp(lower_target, cdf_vals, h_arr))
    upper = float(np.interp(upper_target, cdf_vals, h_arr))

    return lower, upper


def extract_baseline(
    posteriors_dir: Path,
    crb_csv_path: Path | None = None,
    true_h: float = 0.73,
) -> BaselineSnapshot:
    """Extract baseline H0 posterior metrics from a posteriors directory.

    Args:
        posteriors_dir: Directory containing h_*.json files from an h-sweep.
        crb_csv_path: Optional path to CRB CSV for per-event summaries.
        true_h: True Hubble constant value for bias computation. Default 0.73.

    Returns:
        BaselineSnapshot with MAP h, 68% CI, bias %, and event count.

    Raises:
        ValueError: If fewer than 3 h-value files are found (insufficient for
            credible interval computation per D-02).
    """
    posteriors = load_posteriors(posteriors_dir)

    if len(posteriors) < 3:
        raise ValueError(
            f"Found only {len(posteriors)} h-value files in {posteriors_dir}. "
            "At least 3 h-value files are required to compute a credible interval. "
            "Run a full h-sweep before calling extract_baseline."
        )

    h_values = [r["h"] for r in posteriors]
    log_posts = [r["log_posterior"] for r in posteriors]

    # MAP h: h value with highest log-posterior
    map_idx = int(np.argmax(log_posts))
    map_h = h_values[map_idx]

    # 68% CI
    ci_lower, ci_upper = compute_credible_interval(h_values, log_posts, level=0.68)
    ci_width = ci_upper - ci_lower

    # Bias %
    bias_percent = (map_h - true_h) / true_h * 100.0

    # Event count from first file
    n_events = int(posteriors[0]["n_detections"])

    # Per-event summaries from CRB CSV
    per_event_summaries: list[dict[str, float]] = []
    if crb_csv_path is not None and crb_csv_path.exists():
        per_event_summaries = _extract_per_event_summaries(crb_csv_path)

    return BaselineSnapshot(
        map_h=map_h,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        ci_width=ci_width,
        bias_percent=bias_percent,
        n_events=n_events,
        h_values=h_values,
        log_posteriors=log_posts,
        per_event_summaries=per_event_summaries,
        created_at=datetime.datetime.now(datetime.UTC).isoformat() + "Z",
        git_commit=_get_git_commit_safe(),
    )


def _extract_per_event_summaries(crb_csv_path: Path) -> list[dict[str, float]]:
    """Extract per-event diagnostic data from a CRB CSV file.

    Args:
        crb_csv_path: Path to prepared_cramer_rao_bounds.csv.

    Returns:
        List of dicts with keys: d_L, SNR, sigma_d_L_over_d_L, condition_number,
        quality_pass.
    """
    import pandas as pd

    df = pd.read_csv(crb_csv_path)
    summaries: list[dict[str, float]] = []

    for _, row in df.iterrows():
        summary: dict[str, float] = {}
        for col in ["d_L", "SNR", "sigma_d_L_over_d_L", "condition_number", "quality_pass"]:
            if col in row.index:
                summary[col] = float(row[col])
        summaries.append(summary)

    return summaries


def generate_comparison_report(
    baseline: BaselineSnapshot,
    current: BaselineSnapshot,
    output_dir: Path,
    label: str = "current",
) -> Path:
    """Generate a comparison report between a baseline and current posterior snapshot.

    Writes both a human-readable Markdown report and a machine-readable JSON sidecar.

    Args:
        baseline: The reference (pre-change) BaselineSnapshot.
        current: The current (post-change) BaselineSnapshot.
        output_dir: Directory to write output files into (created if needed).
        label: Label suffix for output file names (default "current").

    Returns:
        Path to the generated Markdown report.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    md_path = output_dir / f"comparison_{label}.md"
    json_path = output_dir / f"comparison_{label}.json"

    # --- Compute deltas ---
    delta_map_h = current.map_h - baseline.map_h
    delta_ci_width = current.ci_width - baseline.ci_width
    delta_bias_pct = current.bias_percent - baseline.bias_percent
    delta_n_events = current.n_events - baseline.n_events

    # --- JSON sidecar ---
    comparison_data: dict[str, object] = {
        "baseline": {
            "map_h": baseline.map_h,
            "ci_lower": baseline.ci_lower,
            "ci_upper": baseline.ci_upper,
            "ci_width": baseline.ci_width,
            "bias_pct": baseline.bias_percent,
            "n_events": baseline.n_events,
        },
        "current": {
            "map_h": current.map_h,
            "ci_lower": current.ci_lower,
            "ci_upper": current.ci_upper,
            "ci_width": current.ci_width,
            "bias_pct": current.bias_percent,
            "n_events": current.n_events,
        },
        "delta": {
            "map_h": delta_map_h,
            "ci_width": delta_ci_width,
            "bias_pct": delta_bias_pct,
            "n_events": delta_n_events,
        },
    }
    json_path.write_text(json.dumps(comparison_data, indent=2))

    # --- Markdown report ---
    lines = [
        f"# H0 Posterior Comparison: baseline vs {label}",
        "",
        f"Generated: {datetime.datetime.now(datetime.UTC).isoformat()}Z",
        "",
        "## Summary Table",
        "",
        "| Metric | Baseline | Current | Delta |",
        "|--------|----------|---------|-------|",
        f"| MAP h | {baseline.map_h:.4f} | {current.map_h:.4f} | {delta_map_h:+.4f} |",
        f"| CI lower | {baseline.ci_lower:.4f} | {current.ci_lower:.4f} | {current.ci_lower - baseline.ci_lower:+.4f} |",
        f"| CI upper | {baseline.ci_upper:.4f} | {current.ci_upper:.4f} | {current.ci_upper - baseline.ci_upper:+.4f} |",
        f"| CI width | {baseline.ci_width:.4f} | {current.ci_width:.4f} | {delta_ci_width:+.4f} |",
        f"| Bias % | {baseline.bias_percent:+.1f}% | {current.bias_percent:+.1f}% | {delta_bias_pct:+.1f}% |",
        f"| N events | {baseline.n_events} | {current.n_events} | {delta_n_events:+d} |",
        "",
        "## Log-Posterior by h Value",
        "",
        "| h | log P (baseline) | log P (current) |",
        "|---|-----------------|-----------------|",
    ]

    # Build aligned table of log-posteriors
    baseline_dict = dict(zip(baseline.h_values, baseline.log_posteriors))
    current_dict = dict(zip(current.h_values, current.log_posteriors))
    all_h = sorted(set(baseline.h_values) | set(current.h_values))
    for h in all_h:
        b_lp = baseline_dict.get(h, float("nan"))
        c_lp = current_dict.get(h, float("nan"))
        lines.append(f"| {h:.3f} | {b_lp:.3f} | {c_lp:.3f} |")

    lines += [
        "",
        "## ASCII Chart",
        "",
        "```",
    ]
    lines += _ascii_chart(baseline.h_values, baseline.log_posteriors, current.h_values, current.log_posteriors)
    lines += [
        "```",
        "",
        "## Verdict",
        "",
    ]

    # Verdict section
    if abs(delta_bias_pct) < 0.5:
        verdict = "No significant change in bias."
    elif delta_bias_pct < 0:
        verdict = f"Bias IMPROVED by {abs(delta_bias_pct):.1f} percentage points."
    else:
        verdict = f"Bias WORSENED by {delta_bias_pct:.1f} percentage points."

    lines.append(verdict)
    lines.append("")

    md_path.write_text("\n".join(lines))

    return md_path


def _ascii_chart(
    h_baseline: list[float],
    lp_baseline: list[float],
    h_current: list[float],
    lp_current: list[float],
    width: int = 60,
    height: int = 10,
) -> list[str]:
    """Generate a simple ASCII chart comparing two log-posterior curves.

    Args:
        h_baseline: h values for the baseline.
        lp_baseline: Log-posteriors for the baseline.
        h_current: h values for the current run.
        lp_current: Log-posteriors for the current run.
        width: Chart width in characters.
        height: Chart height in lines.

    Returns:
        List of strings forming the ASCII chart.
    """
    if not h_baseline and not h_current:
        return ["(no data)"]

    all_h = h_baseline + h_current
    all_lp = lp_baseline + lp_current
    if not all_h:
        return ["(no data)"]

    h_min = min(all_h)
    h_max = max(all_h)
    lp_min = min(all_lp)
    lp_max = max(all_lp)

    if h_max == h_min:
        h_max = h_min + 1.0
    if lp_max == lp_min:
        lp_max = lp_min + 1.0

    grid = [[" "] * width for _ in range(height)]

    def _plot(h_vals: list[float], lp_vals: list[float], char: str) -> None:
        for h, lp in zip(h_vals, lp_vals):
            xi = int((h - h_min) / (h_max - h_min) * (width - 1))
            yi = int((lp - lp_min) / (lp_max - lp_min) * (height - 1))
            yi = height - 1 - yi  # flip y-axis
            xi = max(0, min(width - 1, xi))
            yi = max(0, min(height - 1, yi))
            grid[yi][xi] = char

    _plot(h_baseline, lp_baseline, "B")
    _plot(h_current, lp_current, "C")

    lines = []
    for row in grid:
        lines.append("".join(row))

    lines.append(f"{h_min:.2f}" + " " * (width - 9) + f"{h_max:.2f}")
    lines.append("B = baseline, C = current")
    return lines
