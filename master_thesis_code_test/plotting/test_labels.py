"""Tests for the single LABELS provider and the Mpc/Gpc reconciliation (VR-ANNO-06)."""

from pathlib import Path

from master_thesis_code.plotting._labels import LABELS

# Source files that must NOT carry a hardcoded Gpc axis-label literal: every
# distance axis routes through LABELS["d_L_gpc"] instead.
_FACTORY_FILES = [
    "master_thesis_code/plotting/fisher_plots.py",
    "master_thesis_code/plotting/paper_figures.py",
    "master_thesis_code/plotting/evaluation_plots.py",
    "master_thesis_code/plotting/selection_plots.py",
    "master_thesis_code/plotting/catalog_plots.py",
    "master_thesis_code/plotting/simulation_plots.py",
]


def test_labels_distance_keys_are_unit_explicit() -> None:
    """d_L is Mpc, d_L_gpc is Gpc — both unit-explicit and correct."""
    assert "Mpc" in LABELS["d_L"]
    assert "Gpc" in LABELS["d_L_gpc"]
    # The two keys differ only in their unit.
    assert LABELS["d_L"] != LABELS["d_L_gpc"]


def test_no_hardcoded_gpc_axis_label_literal_in_factories() -> None:
    """The surveyed factory files contain no hardcoded [Gpc] axis-label literal.

    Comments documenting *why* the data is in Gpc may remain; only ``set_xlabel`` /
    ``set_ylabel`` lines (and bare ``x_label =`` assignments feeding them) are
    checked for a Gpc unit literal.
    """
    repo_root = Path(__file__).resolve().parents[2]
    for rel in _FACTORY_FILES:
        text = (repo_root / rel).read_text()
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue  # explanatory comment, allowed to mention Gpc
            is_axis_label_line = (
                "set_xlabel(" in stripped
                or "set_ylabel(" in stripped
                or stripped.startswith("x_label")
                or stripped.startswith("x_label =")
            )
            if not is_axis_label_line:
                continue
            assert "[Gpc]" not in stripped, f"{rel}: hardcoded [Gpc] in axis label: {stripped}"
            assert r"[\mathrm{Gpc}]" not in stripped, (
                f"{rel}: hardcoded [\\mathrm{{Gpc}}] in axis label: {stripped}"
            )
