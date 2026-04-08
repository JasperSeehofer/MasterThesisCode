"""Tests for generate_diagnostic_summary() in evaluation_report.py."""

import json
from pathlib import Path

import pandas as pd

from master_thesis_code.bayesian_inference.evaluation_report import (
    generate_diagnostic_summary,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CSV_COLUMNS = [
    "event_idx",
    "h",
    "f_i",
    "L_cat_no_bh",
    "L_cat_with_bh",
    "L_comp",
    "combined_no_bh",
    "combined_with_bh",
]


def _make_diagnostic_csv(
    tmp_path: Path,
    rows: list[dict[str, float]],
    filename: str = "event_likelihoods.csv",
) -> Path:
    """Write a synthetic diagnostic CSV and return its path."""
    df = pd.DataFrame(rows, columns=_CSV_COLUMNS)
    csv_path = tmp_path / filename
    df.to_csv(csv_path, index=False)
    return csv_path


# ---------------------------------------------------------------------------
# Test 1: Catalog-only scenario (all f_i=1.0, L_comp=0.0)
# ---------------------------------------------------------------------------


def test_catalog_only_csv_returns_expected_summary(tmp_path: Path) -> None:
    """For catalog-only mode (f_i=1.0, L_comp=0.0), mean_f_i should be 1.0."""
    rows = []
    for event_idx in range(5):
        for h in [0.66, 0.73, 0.80]:
            rows.append(
                {
                    "event_idx": event_idx,
                    "h": h,
                    "f_i": 1.0,
                    "L_cat_no_bh": 0.5,
                    "L_cat_with_bh": 0.6,
                    "L_comp": 0.0,
                    "combined_no_bh": 0.5,
                    "combined_with_bh": 0.6,
                }
            )
    csv_path = _make_diagnostic_csv(tmp_path, rows)
    output_dir = tmp_path / "output"

    result = generate_diagnostic_summary(csv_path, output_dir, label="catalog_only")

    assert result["mean_f_i"] == 1.0
    assert result["median_f_i"] == 1.0
    assert result["mean_L_comp"] == 0.0
    assert result["n_events"] == 5


# ---------------------------------------------------------------------------
# Test 2: Realistic varied CSV with correct frac_L_comp_pulls_low_h
# ---------------------------------------------------------------------------


def test_varied_csv_frac_pulls_low_h(tmp_path: Path) -> None:
    """For events where L_comp(h=0.66) > L_comp(h=0.73), frac should reflect that."""
    rows = []
    # Event 0: L_comp pulls low (L_comp at 0.66 > L_comp at 0.73)
    rows.append(
        {
            "event_idx": 0,
            "h": 0.66,
            "f_i": 0.8,
            "L_cat_no_bh": 0.3,
            "L_cat_with_bh": 0.4,
            "L_comp": 0.9,
            "combined_no_bh": 0.5,
            "combined_with_bh": 0.6,
        }
    )
    rows.append(
        {
            "event_idx": 0,
            "h": 0.73,
            "f_i": 0.8,
            "L_cat_no_bh": 0.5,
            "L_cat_with_bh": 0.6,
            "L_comp": 0.3,
            "combined_no_bh": 0.5,
            "combined_with_bh": 0.6,
        }
    )
    # Event 1: L_comp does NOT pull low (L_comp at 0.66 < L_comp at 0.73)
    rows.append(
        {
            "event_idx": 1,
            "h": 0.66,
            "f_i": 0.7,
            "L_cat_no_bh": 0.2,
            "L_cat_with_bh": 0.3,
            "L_comp": 0.1,
            "combined_no_bh": 0.3,
            "combined_with_bh": 0.4,
        }
    )
    rows.append(
        {
            "event_idx": 1,
            "h": 0.73,
            "f_i": 0.7,
            "L_cat_no_bh": 0.4,
            "L_cat_with_bh": 0.5,
            "L_comp": 0.5,
            "combined_no_bh": 0.5,
            "combined_with_bh": 0.6,
        }
    )

    csv_path = _make_diagnostic_csv(tmp_path, rows)
    output_dir = tmp_path / "output"

    result = generate_diagnostic_summary(csv_path, output_dir, label="varied")

    # 1 of 2 events pulls low -> frac = 0.5
    assert result["frac_L_comp_pulls_low_h"] == 0.5
    assert result["n_events_L_comp_pulls_low"] == 1
    assert result["n_events_compared"] == 2


# ---------------------------------------------------------------------------
# Test 3: frac_L_comp_pulls_low_h is bounded [0, 1]
# ---------------------------------------------------------------------------


def test_frac_pulls_low_h_bounded(tmp_path: Path) -> None:
    """frac_L_comp_pulls_low_h should be between 0.0 and 1.0."""
    rows = []
    for event_idx in range(10):
        for h in [0.66, 0.73]:
            rows.append(
                {
                    "event_idx": event_idx,
                    "h": h,
                    "f_i": 0.5,
                    "L_cat_no_bh": 0.3,
                    "L_cat_with_bh": 0.4,
                    "L_comp": 0.1 * (1 if h == 0.66 else 2),
                    "combined_no_bh": 0.3,
                    "combined_with_bh": 0.4,
                }
            )
    csv_path = _make_diagnostic_csv(tmp_path, rows)
    output_dir = tmp_path / "output"

    result = generate_diagnostic_summary(csv_path, output_dir, label="bounded")

    frac = result["frac_L_comp_pulls_low_h"]
    assert isinstance(frac, float)
    assert 0.0 <= frac <= 1.0


# ---------------------------------------------------------------------------
# Test 4: Output files (JSON and MD) are created
# ---------------------------------------------------------------------------


def test_output_files_created(tmp_path: Path) -> None:
    """generate_diagnostic_summary should create JSON and MD output files."""
    rows = []
    for event_idx in range(3):
        for h in [0.66, 0.73]:
            rows.append(
                {
                    "event_idx": event_idx,
                    "h": h,
                    "f_i": 0.9,
                    "L_cat_no_bh": 0.3,
                    "L_cat_with_bh": 0.4,
                    "L_comp": 0.05,
                    "combined_no_bh": 0.3,
                    "combined_with_bh": 0.4,
                }
            )
    csv_path = _make_diagnostic_csv(tmp_path, rows)
    output_dir = tmp_path / "output"

    generate_diagnostic_summary(csv_path, output_dir, label="test_output")

    json_path = output_dir / "diagnostic_summary_test_output.json"
    md_path = output_dir / "diagnostic_summary_test_output.md"

    assert json_path.exists()
    assert md_path.exists()

    # Verify JSON is valid and has expected keys
    data = json.loads(json_path.read_text())
    assert "mean_f_i" in data
    assert "frac_L_comp_pulls_low_h" in data
    assert "mean_L_comp" in data
    assert "n_events" in data

    # Verify MD has content
    md_content = md_path.read_text()
    assert "Diagnostic Summary" in md_content
    assert "f_i" in md_content
