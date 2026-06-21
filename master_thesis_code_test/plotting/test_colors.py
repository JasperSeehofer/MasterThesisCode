"""Tests for the centralized color palette and label constants."""

from matplotlib.colors import LinearSegmentedColormap

from master_thesis_code.plotting._colors import (
    ACCENT,
    CMAP,
    CYCLE,
    EDGE,
    MEAN,
    PLANCK,
    REFERENCE,
    SEQUENTIAL_BLUES,
    SH0ES,
    TRUTH,
    VARIANT_NO_MASS,
    VARIANT_WITH_MASS,
)
from master_thesis_code.plotting._labels import LABELS

# --- _colors.py tests ---


def test_truth_is_nonempty_hex() -> None:
    assert isinstance(TRUTH, str) and TRUTH.startswith("#") and len(TRUTH) == 7


def test_truth_is_horizon_vermillion() -> None:
    """HORIZON v2: truth/injected rule is warm vermillion, reserved for that role."""
    assert TRUTH == "#C2451E"


def test_variant_no_mass_is_horizon_navy() -> None:
    """Without-M_z headline series is HORIZON observatory navy."""
    assert VARIANT_NO_MASS == "#1B2A4A"
    assert VARIANT_NO_MASS.startswith("#") and len(VARIANT_NO_MASS) == 7


def test_variant_with_mass_is_horizon_gold() -> None:
    """With-M_z series is HORIZON signal gold (strong lightness contrast vs navy)."""
    assert VARIANT_WITH_MASS == "#E8A317"
    assert VARIANT_WITH_MASS.startswith("#") and len(VARIANT_WITH_MASS) == 7


def test_variant_and_reference_colors_are_pairwise_distinct() -> None:
    """Regression guard against the two-blues / reference collision (kill #56B4E9)."""
    distinct = {VARIANT_NO_MASS, VARIANT_WITH_MASS, REFERENCE}
    assert len(distinct) == 3, (
        f"VARIANT_NO_MASS, VARIANT_WITH_MASS, REFERENCE must be pairwise distinct; "
        f"got {VARIANT_NO_MASS=}, {VARIANT_WITH_MASS=}, {REFERENCE=}"
    )


def test_planck_band_color_is_hex() -> None:
    assert PLANCK == "#3E7CB1"
    assert PLANCK.startswith("#") and len(PLANCK) == 7


def test_sh0es_band_color_is_hex() -> None:
    assert SH0ES == "#9A6FB0"
    assert SH0ES.startswith("#") and len(SH0ES) == 7


def test_band_colors_distinct_from_data_series() -> None:
    """Reserved band colors must never coincide with a data-series or truth color."""
    band = {PLANCK, SH0ES}
    series = {VARIANT_NO_MASS, VARIANT_WITH_MASS, TRUTH}
    assert band.isdisjoint(series), "PLANCK/SH0ES collide with a data-series/truth color"


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


def test_accent_is_hex() -> None:
    assert isinstance(ACCENT, str) and ACCENT.startswith("#") and len(ACCENT) == 7


def test_sequential_blues_is_cmap_object() -> None:
    assert isinstance(SEQUENTIAL_BLUES, LinearSegmentedColormap)


def test_cycle_is_okabe_ito() -> None:
    """First three CYCLE entries match Okabe-Ito orange, sky blue, bluish green."""
    assert CYCLE[0] == "#E69F00"
    assert CYCLE[1] == "#56B4E9"
    assert CYCLE[2] == "#009E73"


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
