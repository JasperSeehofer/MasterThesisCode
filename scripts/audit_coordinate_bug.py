"""Baseline audit: count CRB events in the ±5°/±10°/±15° ecliptic-equator bands.

Produces three artifacts under ``.planning/``:

- ``audit_coordinate_bug.md`` — human-readable band-count table + histogram embed.
- ``audit_coordinate_bug.json`` — machine-readable sidecar with ``git_commit`` + ISO timestamp.
- ``audit_coordinate_bug_histogram.png`` — histogram of ``|qS − π/2|`` across the 42 events.

Scope (per .planning/phases/35-coordinate-bug-characterization/35-CONTEXT.md D-10):
single 42-event CRB CSV (``simulations/cramer_rao_bounds.csv``). The 27 h-value
posteriors reuse the same sky positions, so scanning them adds no information.

Formulas (D-11, USER-LOCKED):

- Band condition: ``|qS − π/2| < band × π/180`` for band ∈ {5, 10, 15} degrees.
- Isotropic-prior expected fraction for a ±band° ecliptic-equator slice:
  ``sin(band × π/180)`` (from ``∫ sin θ dθ / 2``).
  ±5° → 0.0872, ±10° → 0.1736, ±15° → 0.2588.

Usage::

    uv run python scripts/audit_coordinate_bug.py \\
        --csv simulations/cramer_rao_bounds.csv \\
        --output-dir .planning/

Phase 40 VERIFY-04 re-runs this script against the post-fix CRB CSV and
diffs the JSON sidecars.
"""

import argparse
import datetime
import json
from pathlib import Path

import matplotlib
import numpy as np
import numpy.typing as npt
import pandas as pd

from master_thesis_code.bayesian_inference.evaluation_report import _get_git_commit_safe
from master_thesis_code.plotting import apply_style
from master_thesis_code.plotting._helpers import get_figure, save_figure

# D-11: Audit bands (degrees). Ordered ascending for markdown table readability.
_BANDS_DEG: tuple[int, ...] = (5, 10, 15)
# D-11: Column of interest in the CRB CSV. See CONTEXT.md §canonical_refs line 97.
_QS_COLUMN: str = "qS"
# PATTERNS.md §"Bin count" — 12 bins is the sensible default for 42 events.
_HISTOGRAM_BINS: int = 12


def compute_band_counts(
    qS_values: npt.NDArray[np.float64],
    bands_deg: tuple[int, ...] = _BANDS_DEG,
) -> dict[str, int]:
    """Count events with ``|qS − π/2| < band × π/180`` for each band.

    Formula per D-11: ``|qS − π/2| < band × π/180``.

    Args:
        qS_values: recovered ecliptic polar angles in radians (1-D array).
        bands_deg: ascending tuple of band half-widths in degrees.

    Returns:
        Dict keyed by band (as str to survive JSON round-trip):
        ``{"5": count, "10": count, "15": count}``.
    """
    offset = np.abs(qS_values - np.pi / 2)
    return {str(band): int(np.sum(offset < band * np.pi / 180)) for band in bands_deg}


def compute_band_fractions(
    band_counts: dict[str, int],
    event_count: int,
) -> dict[str, float]:
    """Per-band observed fraction = count / total (float, NaN-safe).

    Args:
        band_counts: Dict mapping band (str degrees) to event count.
        event_count: Total number of events (denominator).

    Returns:
        Dict mapping band (str degrees) to observed fraction. Returns NaN
        values if ``event_count`` is zero.
    """
    if event_count == 0:
        return {band: float("nan") for band in band_counts}
    return {band: count / event_count for band, count in band_counts.items()}


def expected_fraction_isotropic(band_deg: int) -> float:
    """Isotropic-prior expected fraction in a ±band° ecliptic-equator slice.

    ``∫_{π/2 − band_rad}^{π/2 + band_rad} sin θ dθ / ∫_0^π sin θ dθ = sin(band_rad)``.

    ±5° → sin(5 × π/180) ≈ 0.0872.

    Args:
        band_deg: Half-width of the ecliptic-equator band in degrees.

    Returns:
        Expected fraction of isotropically distributed events in this band.
    """
    return float(np.sin(band_deg * np.pi / 180))


def _plot_histogram(
    offset_rad: npt.NDArray[np.float64],
    output_path_no_ext: str,
) -> None:
    """Draw and save the |qS − π/2| histogram to PNG.

    Uses the project plotting factory pattern: data in, ``(fig, ax)`` from
    ``get_figure(preset="single")``, saved via ``save_figure(..., formats=("png",))``.

    Args:
        offset_rad: 1-D array of ``|qS − π/2|`` values in radians.
        output_path_no_ext: Output path without file extension (extension
            is appended by ``save_figure``).
    """
    fig, ax = get_figure(preset="single")
    ax.hist(offset_rad, bins=_HISTOGRAM_BINS, edgecolor="black", alpha=0.7)
    ax.set_xlabel(r"$|q_S - \pi/2|$ (rad)")
    ax.set_ylabel("Event count")
    ax.set_title(f"Pre-fix baseline: ecliptic-equator offset ({len(offset_rad)} events)")
    # PATTERNS.md caveat: save_figure default is PDF; pass formats=("png",) explicitly.
    save_figure(fig, output_path_no_ext, formats=("png",))


def audit_coordinate_bug(
    csv_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    """Compute band counts + expected fractions; write MD + JSON + PNG artifacts.

    Args:
        csv_path: Path to the production CRB CSV with a ``qS`` column.
        output_dir: Directory to write artifacts into (created if missing).

    Returns:
        Summary dict (also serialized to JSON) with keys: ``event_count``,
        ``band_counts``, ``band_fractions``, ``expected_fraction_5deg``,
        ``expected_fractions_all_bands``, ``csv_source_path``, ``git_commit``,
        ``timestamp``.

    Raises:
        KeyError: If the CSV does not contain a ``qS`` column.
        FileNotFoundError: If ``csv_path`` does not exist.
    """
    df = pd.read_csv(csv_path)
    if _QS_COLUMN not in df.columns:
        raise KeyError(
            f"Expected column {_QS_COLUMN!r} in {csv_path}; got columns: {list(df.columns)[:10]}..."
        )
    qS_values = df[_QS_COLUMN].to_numpy(dtype=np.float64)
    event_count = int(len(qS_values))

    band_counts = compute_band_counts(qS_values)
    band_fractions = compute_band_fractions(band_counts, event_count)
    expected_fractions = {str(band): expected_fraction_isotropic(band) for band in _BANDS_DEG}

    # Build summary dict (flat, JSON-serializable, snake_case).
    timestamp = datetime.datetime.now(datetime.UTC).isoformat() + "Z"
    summary: dict[str, object] = {
        "event_count": event_count,
        "band_counts": band_counts,
        "band_fractions": band_fractions,
        # D-11: include single-band shortcut for ±5° (the mandatory band).
        "expected_fraction_5deg": expected_fractions["5"],
        "expected_fractions_all_bands": expected_fractions,
        "csv_source_path": str(csv_path),
        "git_commit": _get_git_commit_safe(),
        "timestamp": timestamp,
    }

    output_dir.mkdir(parents=True, exist_ok=True)

    # Write JSON sidecar (pattern from evaluation_report.py:570-609).
    json_path = output_dir / "audit_coordinate_bug.json"
    json_path.write_text(json.dumps(summary, indent=2) + "\n")

    # Draw + save histogram PNG.
    offset_rad = np.abs(qS_values - np.pi / 2)
    histogram_path_no_ext = str(output_dir / "audit_coordinate_bug_histogram")
    _plot_histogram(offset_rad, histogram_path_no_ext)

    # Write markdown report.
    md_lines: list[str] = [
        "# Coordinate Bug Baseline Audit (Phase 35)",
        "",
        f"Generated: {timestamp}",
        f"Source CSV: `{csv_path}`",
        f"Event count: **{event_count}**",
        f"Git commit: `{summary['git_commit']}`",
        "",
        "## Band counts — events within `|qS − π/2| < band × π/180`",
        "",
        "| Band | Count | Fraction (observed) | Fraction (isotropic prior) | Deviation |",
        "|------|-------|---------------------|----------------------------|-----------|",
    ]
    for band in _BANDS_DEG:
        count = band_counts[str(band)]
        obs = band_fractions[str(band)]
        exp = expected_fractions[str(band)]
        dev = obs - exp
        md_lines.append(f"| ±{band}° | {count} | {obs:.4f} | {exp:.4f} | {dev:+.4f} |")
    md_lines.extend(
        [
            "",
            "## Histogram",
            "",
            "![|qS − π/2| distribution](audit_coordinate_bug_histogram.png)",
            "",
            "## JSON sidecar",
            "",
            "`.planning/audit_coordinate_bug.json`",
            "",
            "## Summary",
            "",
            (
                "Observed fraction in ±5° band minus isotropic-prior expected "
                f"({expected_fractions['5']:.4f}) is "
                f"{band_fractions['5'] - expected_fractions['5']:+.4f}. "
                "A large positive deviation would suggest the coordinate bug "
                "is artificially piling events at the ecliptic equator via "
                "the singular BallTree embedding (handler.py:286-288). "
                "Phase 40 VERIFY-04 re-runs this audit post-fix and diffs the JSON."
            ),
            "",
        ]
    )
    md_path = output_dir / "audit_coordinate_bug.md"
    md_path.write_text("\n".join(md_lines))

    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list (defaults to sys.argv[1:] when None).

    Returns:
        Parsed argument namespace with ``csv`` and ``output_dir`` attributes.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Baseline audit: count CRB events in ±5°/±10°/±15° ecliptic-equator bands. "
            "Writes audit_coordinate_bug.{md,json,png} to --output-dir."
        )
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="simulations/cramer_rao_bounds.csv",
        help="Path to Cramer-Rao bounds CSV (column 'qS' required).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".planning",
        help="Directory to write artifacts into (default: .planning/).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point.

    Args:
        argv: Argument list (defaults to sys.argv[1:] when None).
    """
    args = parse_args(argv)
    # Apply project matplotlib style once (Agg backend + emri_thesis.mplstyle).
    # Session fixture does this in pytest; we must do it explicitly outside.
    apply_style()
    # Silence noisy matplotlib backend warnings on headless CLI.
    matplotlib.use("Agg", force=False)
    summary = audit_coordinate_bug(
        csv_path=Path(args.csv),
        output_dir=Path(args.output_dir),
    )
    print(f"Wrote audit artifacts to {args.output_dir}/ for {summary['event_count']} events.")


if __name__ == "__main__":
    main()
