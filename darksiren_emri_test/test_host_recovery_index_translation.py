"""Index-space translation behind the P6 host-recovery counter (INSTR-3).

``Detection.host_galaxy_index`` is a POSITION in whatever pruned + ``reset_index``
frame the INJECTION run's :class:`GalaxyCatalogueHandler` built. Under a campaign
#53 observed-catalogue realization the EVALUATION handler prunes possibly-scattered
mass/redshift values, so its own pruned frame can drop rows the baseline frame kept
(20,834,171 mass-valid parent rows but only 19,874,547 surviving the observed-column
prune -- ``HANDOFF_20260730.md`` §5). Comparing ``host_galaxy_index`` positionally
across those two frames therefore silently names the WRONG galaxy, which would make
a naive host-recovery counter garbage exactly in the regime it is meant to diagnose.

:meth:`GalaxyCatalogueHandler.resolve_host_recovery_position` translates through the
shared PRE-M/z-prune parent row-identity space. These tests pin both paths:
the unscattered identity path, the scattered divergent-frame path (including the
discriminating "shifted down by an earlier drop" case), and the inclusive-boundary
behaviour of the extracted ``_mass_redshift_prune_mask`` predicate.

Synthetic catalogues are injected onto ``object.__new__(GalaxyCatalogueHandler)``
(the repo idiom; see ``test_rate_weighted_catalog.py``) so the multi-GB GLADE file
is never read.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from darksiren_emri.galaxy_catalogue.handler import (
    GalaxyCatalogueHandler,
    InternalCatalogColumns,
    _empiric_stellar_mass_to_BH_mass_relation,
    _mass_redshift_prune_mask,
    _reduced_catalog_column_names,
)

# Wide enough that every synthetic row survives the parent-frame prune, so the
# expected translation depends ONLY on the injected evaluation-frame "index"
# column -- the quantity under test.
_M_MIN = 1.0e-30
_M_MAX = 1.0e30
_Z_MAX = 10.0


def _write_parent_csv(path: Path, n_rows: int) -> None:
    """Write a headerless ``n_rows``-row reduced-catalogue CSV in on-disk order."""
    columns = _reduced_catalog_column_names()
    rows = {
        "RIGHT_ASCENSION": np.linspace(10.0, 40.0, n_rows),
        "DECLINATION": np.linspace(-20.0, 20.0, n_rows),
        "APPARENT_B_MAG": np.full(n_rows, 15.0),
        "REDSHIFT": np.linspace(0.01, 0.04, n_rows),
        "REDSHIFT_MEASUREMENT_ERROR": np.full(n_rows, 1.0e-3),
        "STELLAR_MASS": np.full(n_rows, 12.0),
        "STELLAR_MASS_ABSOULTE_ERROR": np.full(n_rows, 1.2),
        "REDSHIFT_FLAG": np.full(n_rows, 3),
    }
    pd.DataFrame(rows)[columns].to_csv(path, header=False, index=False)


def _make_eval_frame(parent_row_labels: list[int]) -> pd.DataFrame:
    """Evaluation handler's own pruned + ``reset_index`` frame.

    The ``"index"`` column is what ``setup_galaxy_catalog_balltree``'s
    ``reset_index()`` writes: per POST-prune row position, the PRE-prune row label.
    """
    n = len(parent_row_labels)
    return pd.DataFrame(
        {
            "index": np.asarray(parent_row_labels, dtype=np.int64),
            InternalCatalogColumns.PHI_S: np.linspace(0.1, 1.0, n),
            InternalCatalogColumns.THETA_S: np.linspace(0.5, 2.0, n),
            InternalCatalogColumns.REDSHIFT: np.linspace(0.01, 0.03, n),
            InternalCatalogColumns.REDSHIFT_ERROR: np.full(n, 1.0e-3),
            InternalCatalogColumns.BH_MASS: np.full(n, 1.0e6),
            InternalCatalogColumns.BH_MASS_ERROR: np.full(n, 1.0e5),
        }
    )


def _make_handler(
    eval_frame: pd.DataFrame,
    scattered: bool,
    parent_csv: Path | None = None,
) -> GalaxyCatalogueHandler:
    handler = object.__new__(GalaxyCatalogueHandler)
    handler.reduced_galaxy_catalog = eval_frame
    handler.M_min = _M_MIN
    handler.M_max = _M_MAX
    handler.z_max = _Z_MAX
    handler._scattered = scattered
    handler.realization_metadata = (
        None if parent_csv is None else {"parent_csv": str(parent_csv), "sigma_scale": 1.0}
    )
    handler._baseline_prune_positions = None
    handler._parent_row_position_map = None
    return handler


# ---------------------------------------------------------------------------
# Test 1 -- unscattered / identity path
# ---------------------------------------------------------------------------


def test_unscattered_handler_translates_to_identity() -> None:
    """Without scatter the evaluation frame IS the injection frame."""
    handler = _make_handler(_make_eval_frame([0, 1, 2, 3]), scattered=False)

    for position in range(4):
        assert handler.resolve_host_recovery_position(position) == position


@pytest.mark.parametrize("host_galaxy_index", [-1, -7, 4, 99])
def test_unscattered_handler_rejects_out_of_range(host_galaxy_index: int) -> None:
    """Dark hosts (-1) and out-of-frame positions resolve to None, never garbage."""
    handler = _make_handler(_make_eval_frame([0, 1, 2, 3]), scattered=False)

    assert handler.resolve_host_recovery_position(host_galaxy_index) is None


# ---------------------------------------------------------------------------
# Test 2 -- scattered / divergent-frame path (the bug this method fixes)
# ---------------------------------------------------------------------------


def test_scattered_handler_translates_through_parent_row_identity(tmp_path: Path) -> None:
    """Parent row 2 fell out of the observed prune; later positions shift down.

    Toy-scale reproduction of the "958k rows vanish under the observed-column
    prune" mechanism (HANDOFF_20260730.md §5). The injection run saw a 4-galaxy
    pruned frame (positions 0..3 -> parent rows 0..3); this evaluation handler's
    own frame kept only parent rows [0, 1, 3].
    """
    parent_csv = tmp_path / "observed_parent.csv"
    _write_parent_csv(parent_csv, n_rows=4)
    handler = _make_handler(_make_eval_frame([0, 1, 3]), scattered=True, parent_csv=parent_csv)

    assert handler.resolve_host_recovery_position(0) == 0
    assert handler.resolve_host_recovery_position(1) == 1
    # Truly unrecoverable -- reported as a miss, not as some other galaxy.
    assert handler.resolve_host_recovery_position(2) is None
    # The discriminating case: a naive same-position comparison would return 3
    # (out of range) or silently name the wrong galaxy.
    assert handler.resolve_host_recovery_position(3) == 2


def test_scattered_handler_rejects_dark_and_out_of_frame(tmp_path: Path) -> None:
    parent_csv = tmp_path / "observed_parent.csv"
    _write_parent_csv(parent_csv, n_rows=4)
    handler = _make_handler(_make_eval_frame([0, 1, 3]), scattered=True, parent_csv=parent_csv)

    assert handler.resolve_host_recovery_position(-1) is None
    # Beyond the baseline (injection-time) frame length.
    assert handler.resolve_host_recovery_position(4) is None


def test_scattered_translation_caches_parent_frame(tmp_path: Path) -> None:
    """The ~1 GB parent read is memoized: paid at most once per handler."""
    parent_csv = tmp_path / "observed_parent.csv"
    _write_parent_csv(parent_csv, n_rows=4)
    handler = _make_handler(_make_eval_frame([0, 1, 3]), scattered=True, parent_csv=parent_csv)

    assert handler._baseline_prune_positions is None
    assert handler.resolve_host_recovery_position(0) == 0
    assert handler._baseline_prune_positions is not None
    assert handler._parent_row_position_map == {0: 0, 1: 1, 3: 2}

    # Deleting the parent CSV must not break a second lookup -- proof the second
    # call never re-reads it.
    parent_csv.unlink()
    assert handler.resolve_host_recovery_position(3) == 2


def test_baseline_prune_positions_match_the_production_predicate(tmp_path: Path) -> None:
    """The counter's baseline reconstruction uses the SAME prune predicate."""
    parent_csv = tmp_path / "observed_parent.csv"
    _write_parent_csv(parent_csv, n_rows=4)
    handler = _make_handler(_make_eval_frame([0, 1, 3]), scattered=True, parent_csv=parent_csv)

    positions = handler._compute_baseline_prune_positions()

    np.testing.assert_array_equal(positions, np.array([0, 1, 2, 3]))


def test_baseline_prune_drops_parent_rows_outside_the_mass_window(tmp_path: Path) -> None:
    """A parent row outside [M_min, M_max] shifts every later baseline position."""
    parent_csv = tmp_path / "observed_parent.csv"
    _write_parent_csv(parent_csv, n_rows=4)
    handler = _make_handler(_make_eval_frame([0, 1, 2]), scattered=True, parent_csv=parent_csv)
    # Squeeze M_max so that parent row 1 (identical stellar masses -> identical BH
    # masses) can be excluded only via the redshift leg: use z_max instead, which
    # is monotonically increasing across the synthetic rows.
    handler.z_max = 0.025

    positions = handler._compute_baseline_prune_positions()

    # Synthetic redshifts are linspace(0.01, 0.04, 4) = [0.01, 0.02, 0.03, 0.04]
    # with z_error 1e-3, so z - z_err <= 0.025 keeps rows 0 and 1 only.
    np.testing.assert_array_equal(positions, np.array([0, 1]))


# ---------------------------------------------------------------------------
# Test 3 -- extracted prune predicate keeps the inclusive boundaries
# ---------------------------------------------------------------------------


def test_mass_redshift_prune_mask_boundaries_are_inclusive() -> None:
    """Regression guard for the verbatim extraction out of ``_get_pruned_galaxy_catalog``.

    Rows are hand-placed exactly on each of the three boundaries; the original
    inline expression used ``>=`` / ``<=`` / ``<=``, so every on-boundary row is
    KEPT and only strictly-outside rows are dropped.
    """
    M_min, M_max, z_max = 1.0e5, 1.0e7, 0.5
    bh_mass = pd.Series([9.0e4, 8.9e4, 1.1e7, 1.11e7, 1.0e6, 1.0e6])
    bh_mass_error = pd.Series([1.0e4, 1.0e4, 1.0e6, 1.0e6, 1.0e5, 1.0e5])
    redshift = pd.Series([0.1, 0.1, 0.1, 0.1, 0.51, 0.52])
    redshift_error = pd.Series([0.01, 0.01, 0.01, 0.01, 0.01, 0.01])

    mask = _mass_redshift_prune_mask(
        bh_mass, bh_mass_error, redshift, redshift_error, M_min, M_max, z_max
    )

    # row 0: M + dM == M_min exactly -> kept;      row 1: just below -> dropped
    # row 2: M - dM == M_max exactly -> kept;      row 3: just above -> dropped
    # row 4: z - dz == z_max exactly -> kept;      row 5: just above -> dropped
    assert list(mask) == [True, False, True, False, True, False]


def test_mass_redshift_prune_mask_matches_the_inline_expression() -> None:
    """Byte-identical to the pre-extraction inline mask on a random frame."""
    rng = np.random.default_rng(20260731)
    n = 500
    bh_mass = pd.Series(10 ** rng.uniform(4.0, 8.0, size=n))
    bh_mass_error = pd.Series(bh_mass.to_numpy() * rng.uniform(0.01, 0.5, size=n))
    redshift = pd.Series(rng.uniform(0.0, 1.5, size=n))
    redshift_error = pd.Series(rng.uniform(0.0, 0.05, size=n))
    M_min, M_max, z_max = 1.0e5, 1.0e7, 0.5

    expected = (
        (bh_mass + bh_mass_error >= M_min)
        & (bh_mass - bh_mass_error <= M_max)
        & (redshift - redshift_error <= z_max)
    )
    mask = _mass_redshift_prune_mask(
        bh_mass, bh_mass_error, redshift, redshift_error, M_min, M_max, z_max
    )

    pd.testing.assert_series_equal(mask, expected)


def test_get_pruned_galaxy_catalog_uses_the_extracted_predicate() -> None:
    """The production prune and the counter's reconstruction cannot diverge."""
    stellar_mass = pd.Series([12.0, 12.0, 12.0])
    bh_mass, bh_mass_error = _empiric_stellar_mass_to_BH_mass_relation(
        stellar_mass, stellar_mass * 0.1
    )
    frame = pd.DataFrame(
        {
            InternalCatalogColumns.REDSHIFT: [0.01, 0.02, 0.9],
            InternalCatalogColumns.REDSHIFT_ERROR: [1.0e-3, 1.0e-3, 1.0e-3],
            InternalCatalogColumns.BH_MASS: bh_mass,
            InternalCatalogColumns.BH_MASS_ERROR: bh_mass_error,
        }
    )
    handler = object.__new__(GalaxyCatalogueHandler)
    handler.reduced_galaxy_catalog = frame

    pruned = handler._get_pruned_galaxy_catalog(M_min=_M_MIN, M_max=_M_MAX, z_max=0.5)

    assert list(pruned.index) == [0, 1]
