"""Tests for the centralized color palette and label constants."""

from master_thesis_code.plotting._colors import CMAP, CYCLE, EDGE, MEAN, REFERENCE, TRUTH
from master_thesis_code.plotting._labels import LABELS


# --- _colors.py tests ---


def test_truth_is_nonempty_hex() -> None:
    assert isinstance(TRUTH, str) and TRUTH.startswith("#") and len(TRUTH) == 7


def test_mean_is_nonempty_hex() -> None:
    assert isinstance(MEAN, str) and MEAN.startswith("#") and len(MEAN) == 7


def test_edge_is_nonempty_hex() -> None:
    assert isinstance(EDGE, str) and EDGE.startswith("#") and len(EDGE) == 7


def test_reference_is_nonempty_hex() -> None:
    assert isinstance(REFERENCE, str) and REFERENCE.startswith("#") and len(REFERENCE) == 7


def test_cycle_has_at_least_six_entries() -> None:
    assert len(CYCLE) >= 6, f"CYCLE has only {len(CYCLE)} entries"


def test_cycle_entries_are_hex_strings() -> None:
    for i, color in enumerate(CYCLE):
        assert isinstance(color, str) and color.startswith("#") and len(color) == 7, (
            f"CYCLE[{i}] = {color!r} is not a valid hex color"
        )


def test_cmap_is_viridis() -> None:
    assert CMAP == "viridis"


# --- _labels.py tests ---

_EMRI_14_PARAMS = [
    "M",
    "mu",
    "a",
    "p0",
    "e0",
    "Y0",
    "d_L",
    "qS",
    "phiS",
    "qK",
    "phiK",
    "Phi_phi0",
    "Phi_theta0",
    "Phi_r0",
]


def test_labels_contains_all_14_emri_params() -> None:
    missing = [p for p in _EMRI_14_PARAMS if p not in LABELS]
    assert not missing, f"LABELS missing EMRI params: {missing}"


def test_labels_values_are_mathtext() -> None:
    for key, label in LABELS.items():
        assert label.startswith("$") and label.endswith("$"), (
            f"LABELS[{key!r}] = {label!r} is not wrapped in $...$"
        )


def test_labels_contains_observables() -> None:
    observables = ["z", "SNR", "H0", "h", "f", "t", "PSD"]
    missing = [o for o in observables if o not in LABELS]
    assert not missing, f"LABELS missing observables: {missing}"
