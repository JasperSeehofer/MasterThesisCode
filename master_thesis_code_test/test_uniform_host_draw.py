"""Local gate tests for the direct uniform in-catalog host draw (CHANGE 2).

These tests pin the self-consistent generative model that replaces the old
"sample (M, z) from a continuous distribution then snap to the nearest catalog
galaxy" host assignment:

    GalaxyCatalogueHandler.draw_uniform_hosts(n, rng, z_max)

draws hosts uniformly (with replacement) from the catalog rows with redshift
``z < z_max``. This implements the equal-weight in-catalog term P(g) = const
(Chen, Fishbach & Holz 2024, arXiv:2212.08694 Eq. 9), so each returned host
carries z / sky / M / errors straight from its catalog row — no nearest-neighbour
snap, no overwrite.

All tests are CPU-only and fast: they inject a SMALL SYNTHETIC
``reduced_galaxy_catalog`` DataFrame directly onto an
``object.__new__(GalaxyCatalogueHandler)`` instance (the repo idiom; see
``master_thesis_code_test/fixtures/coordinate.py`` and
``test_coordinate_roundtrip.py``) so the multi-GB GLADE file is never touched.

Assertions:
    (a) every drawn host has z < z_max;
    (b) two fixed-seed rngs give identical draws (reproducibility);
    (c) over many draws the empirical z and BH-mass distributions match the
        catalog restricted to z < z_max (chi-square uniformity over eligible
        rows + mean agreement on z and M);
    (d) a galaxy with z > z_max is never drawn;
    plus a clear ValueError when no galaxy satisfies z < z_max.
"""

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
from scipy import stats

from master_thesis_code.constants import HOST_DRAW_Z_MAX
from master_thesis_code.galaxy_catalogue.handler import (
    GalaxyCatalogueHandler,
    HostGalaxy,
    InternalCatalogColumns,
)


def _make_handler(
    redshifts: npt.NDArray[np.float64],
    masses: npt.NDArray[np.float64] | None = None,
) -> GalaxyCatalogueHandler:
    """Build a handler around a small in-memory synthetic catalog.

    Bypasses the heavy ``__init__`` (which reads GLADE off disk) via
    ``object.__new__`` and injects a DataFrame whose columns exactly match
    ``InternalCatalogColumns``.

    Args:
        redshifts: Per-galaxy redshift values (define eligibility for the draw).
        masses: Per-galaxy BH masses. Defaults to a distinct ramp so that the
            empirical mass distribution is sensitive to which rows are drawn.

    Returns:
        A ``GalaxyCatalogueHandler`` with ``reduced_galaxy_catalog`` set.
    """
    n = len(redshifts)
    rng = np.random.default_rng(0)
    if masses is None:
        # Distinct, monotone masses so each eligible row has its own M value.
        masses = np.linspace(1.0e5, 9.0e5, n).astype(np.float64)
    df = pd.DataFrame(
        {
            InternalCatalogColumns.PHI_S: rng.uniform(0.0, 2 * np.pi, size=n),
            InternalCatalogColumns.THETA_S: np.arccos(rng.uniform(-1.0, 1.0, size=n)),
            InternalCatalogColumns.REDSHIFT: redshifts.astype(np.float64),
            InternalCatalogColumns.REDSHIFT_ERROR: np.full(n, 1.0e-3, dtype=np.float64),
            InternalCatalogColumns.BH_MASS: masses,
            InternalCatalogColumns.BH_MASS_ERROR: masses * 0.1,
        }
    )
    handler = object.__new__(GalaxyCatalogueHandler)
    handler.reduced_galaxy_catalog = df
    return handler


def test_every_drawn_host_below_z_max() -> None:
    """(a) Every drawn host has redshift strictly below z_max."""
    z = np.concatenate([np.linspace(0.01, 0.45, 20), np.array([0.6, 0.8, 1.2])])
    handler = _make_handler(z)
    z_max = 0.5
    hosts = handler.draw_uniform_hosts(500, rng=np.random.default_rng(1), z_max=z_max)

    assert len(hosts) == 500
    assert all(isinstance(h, HostGalaxy) for h in hosts)
    assert all(h.z < z_max for h in hosts), "draw returned a host with z >= z_max"


def test_fixed_seed_reproducibility() -> None:
    """(b) Two generators with the same seed yield identical draws."""
    z = np.linspace(0.01, 0.49, 30)
    handler = _make_handler(z)

    hosts_a = handler.draw_uniform_hosts(200, rng=np.random.default_rng(2024), z_max=0.5)
    hosts_b = handler.draw_uniform_hosts(200, rng=np.random.default_rng(2024), z_max=0.5)

    idx_a = [h.catalog_index for h in hosts_a]
    idx_b = [h.catalog_index for h in hosts_b]
    z_a = [h.z for h in hosts_a]
    z_b = [h.z for h in hosts_b]
    m_a = [h.M for h in hosts_a]
    m_b = [h.M for h in hosts_b]

    assert idx_a == idx_b, "fixed-seed draws differ in selected catalog indices"
    assert z_a == z_b, "fixed-seed draws differ in redshift values"
    assert m_a == m_b, "fixed-seed draws differ in BH-mass values"


def test_different_seed_changes_draw() -> None:
    """Sanity: different seeds generally produce a different draw sequence."""
    z = np.linspace(0.01, 0.49, 30)
    handler = _make_handler(z)
    idx_a = [
        h.catalog_index
        for h in handler.draw_uniform_hosts(200, rng=np.random.default_rng(1), z_max=0.5)
    ]
    idx_b = [
        h.catalog_index
        for h in handler.draw_uniform_hosts(200, rng=np.random.default_rng(2), z_max=0.5)
    ]
    assert idx_a != idx_b


def test_empirical_distribution_matches_eligible_catalog() -> None:
    """(c) Draws are uniform over eligible rows; empirical z and M match the catalog.

    A chi-square goodness-of-fit on the selected-index frequencies tests
    uniformity over the eligible rows (P(g) = const). Independently, the
    empirical means of z and BH mass must agree with the eligible-catalog means
    — this directly checks that both the redshift AND BH-mass distributions of
    the draw match the catalog restricted to z < z_max.
    """
    z = np.concatenate([np.linspace(0.02, 0.48, 25), np.array([0.55, 0.7, 0.9, 1.1])])
    handler = _make_handler(z)
    z_max = 0.5

    eligible = handler.reduced_galaxy_catalog[
        handler.reduced_galaxy_catalog[InternalCatalogColumns.REDSHIFT] < z_max
    ]
    n_eligible = len(eligible)
    eligible_labels = list(eligible.index)

    n_draws = 40_000
    hosts = handler.draw_uniform_hosts(n_draws, rng=np.random.default_rng(7), z_max=z_max)

    # Chi-square uniformity over eligible catalog rows.
    counts = pd.Series([h.catalog_index for h in hosts]).value_counts()
    observed = np.array([counts.get(label, 0) for label in eligible_labels], dtype=np.float64)
    assert observed.sum() == n_draws
    expected = np.full(n_eligible, n_draws / n_eligible, dtype=np.float64)
    _stat, p_value = stats.chisquare(f_obs=observed, f_exp=expected)
    assert p_value > 0.01, (
        f"selected-index frequencies are not uniform over eligible rows "
        f"(chi-square p={p_value:.4g}); draw is not P(g)=const."
    )

    # Empirical z and BH-mass means match the eligible-catalog means.
    drawn_z = np.array([h.z for h in hosts], dtype=np.float64)
    drawn_m = np.array([h.M for h in hosts], dtype=np.float64)
    eligible_z = eligible[InternalCatalogColumns.REDSHIFT].to_numpy(dtype=np.float64)
    eligible_m = eligible[InternalCatalogColumns.BH_MASS].to_numpy(dtype=np.float64)

    # Standard error of the mean for the with-replacement draw; allow 4 sigma.
    z_sem = eligible_z.std(ddof=0) / np.sqrt(n_draws)
    m_sem = eligible_m.std(ddof=0) / np.sqrt(n_draws)
    assert abs(drawn_z.mean() - eligible_z.mean()) < 4 * z_sem, (
        "empirical redshift mean does not match eligible-catalog mean"
    )
    assert abs(drawn_m.mean() - eligible_m.mean()) < 4 * m_sem, (
        "empirical BH-mass mean does not match eligible-catalog mean"
    )


def test_high_z_galaxy_never_drawn() -> None:
    """(d) A galaxy with z > z_max is never returned by the draw."""
    # Last row is the high-z galaxy (index label n-1), well above z_max.
    z = np.concatenate([np.linspace(0.02, 0.45, 15), np.array([0.95])])
    handler = _make_handler(z)
    z_max = 0.5
    high_z_label = len(z) - 1
    assert (
        handler.reduced_galaxy_catalog[InternalCatalogColumns.REDSHIFT].iloc[high_z_label] > z_max
    )

    hosts = handler.draw_uniform_hosts(5_000, rng=np.random.default_rng(3), z_max=z_max)
    drawn_labels = {h.catalog_index for h in hosts}

    assert high_z_label not in drawn_labels, "a galaxy with z > z_max was drawn"
    assert all(h.z < z_max for h in hosts)


def test_raises_when_no_eligible_galaxy() -> None:
    """A clear ValueError is raised when no galaxy satisfies z < z_max."""
    z = np.array([0.6, 0.7, 0.9, 1.2])  # all above z_max
    handler = _make_handler(z)
    with pytest.raises(ValueError, match="redshift < z_max"):
        handler.draw_uniform_hosts(10, rng=np.random.default_rng(4), z_max=0.5)


def test_default_z_max_is_host_draw_constant() -> None:
    """The default z_max is the HOST_DRAW_Z_MAX constant (1.5, depth-1.5 campaign)."""
    z = np.concatenate([np.linspace(0.02, 0.45, 10), np.array([HOST_DRAW_Z_MAX + 0.1])])
    handler = _make_handler(z)
    hosts = handler.draw_uniform_hosts(200, rng=np.random.default_rng(5))
    assert all(h.z < HOST_DRAW_Z_MAX for h in hosts)
