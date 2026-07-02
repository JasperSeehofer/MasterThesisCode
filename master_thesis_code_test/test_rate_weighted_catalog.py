"""Local gate tests for the rate-weighted in-catalog model (CHANGE 3).

CHANGE 3 reaches research "version B": the simulation host draw AND the
inference in-catalog likelihood term weight each catalogued EMRI host by the
SAME per-MBH EMRI-rate weight

    w(g) = R_eff_per_mbh(M_g) / (1 + z_g)

so that P(g) is proportional to w(g). ``R_eff_per_mbh`` is the *effective
per-MBH* EMRI rate (Babak et al. 2017, arXiv:1703.09722). The per-MBH rate --
not the comoving volume density ``R_EMRI`` -- is correct because each catalog
galaxy is ONE realised MBH (the mass function is already sampled by the
catalog). The galaxy weighting follows Gray et al. (2020), arXiv:1908.06050.

Two pieces are pinned here, both CPU-only and fast:

A. ``GalaxyCatalogueHandler.draw_rate_weighted_hosts`` -- empirical host
   frequencies proportional to w(g) over a small synthetic catalog (chi-square
   against the expected weighted probabilities), reproducible under a fixed
   seed, all drawn hosts have z < z_max, and a clear ValueError when no galaxy
   is eligible.

B. ``weighted_ratio_of_sums`` -- (i) SCALING INVARIANCE under an arbitrary
   positive rescaling of all weights, (ii) the CONSTANT-WEIGHT LIMIT reproduces
   the plain Change-2 ratio sum(N)/sum(D), and (iii) a hand-computed example.

Synthetic catalogs are injected onto ``object.__new__(GalaxyCatalogueHandler)``
(the repo idiom; see ``test_uniform_host_draw.py``) so the multi-GB GLADE file
is never read.
"""

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
from scipy import stats

from master_thesis_code.bayesian_inference.bayesian_statistics import weighted_ratio_of_sums
from master_thesis_code.constants import HOST_DRAW_Z_MAX
from master_thesis_code.emri_rate import R_eff_per_mbh
from master_thesis_code.galaxy_catalogue.handler import (
    GalaxyCatalogueHandler,
    HostGalaxy,
    InternalCatalogColumns,
)


def _make_handler(
    redshifts: npt.NDArray[np.float64],
    masses: npt.NDArray[np.float64],
) -> GalaxyCatalogueHandler:
    """Build a handler around a small in-memory synthetic catalog.

    Bypasses the heavy ``__init__`` (which reads GLADE off disk) via
    ``object.__new__`` and injects a DataFrame whose columns exactly match
    ``InternalCatalogColumns``. The BH-mass column holds SOURCE-FRAME BH masses
    directly -- the same quantity ``HostGalaxy.M`` reads and the rate weight
    ``R_eff_per_mbh(M)/(1+z)`` consumes.
    """
    n = len(redshifts)
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            InternalCatalogColumns.PHI_S: rng.uniform(0.0, 2 * np.pi, size=n),
            InternalCatalogColumns.THETA_S: np.arccos(rng.uniform(-1.0, 1.0, size=n)),
            InternalCatalogColumns.REDSHIFT: redshifts.astype(np.float64),
            InternalCatalogColumns.REDSHIFT_ERROR: np.full(n, 1.0e-3, dtype=np.float64),
            InternalCatalogColumns.BH_MASS: masses.astype(np.float64),
            InternalCatalogColumns.BH_MASS_ERROR: masses.astype(np.float64) * 0.1,
        }
    )
    handler = object.__new__(GalaxyCatalogueHandler)
    handler.reduced_galaxy_catalog = df
    return handler


def _expected_probabilities(
    redshifts: npt.NDArray[np.float64],
    masses: npt.NDArray[np.float64],
    z_max: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
    """Expected P(g) proportional to R_eff_per_mbh(M_g)/(1+z_g) over eligible rows.

    Returns ``(probabilities, eligible_positions)`` where ``probabilities`` is
    aligned to ``eligible_positions`` (positional indices into the full catalog,
    which equal the catalog_index labels for the default RangeIndex catalogs
    built by :func:`_make_handler`).
    """
    eligible = np.flatnonzero(redshifts < z_max)
    w = np.asarray(R_eff_per_mbh(masses[eligible]), dtype=np.float64) / (1.0 + redshifts[eligible])
    return w / w.sum(), eligible


# ---------------------------------------------------------------------------
# A. draw_rate_weighted_hosts
# ---------------------------------------------------------------------------


def test_draw_all_hosts_below_z_max() -> None:
    """Every drawn host has redshift strictly below z_max."""
    z = np.concatenate([np.linspace(0.02, 0.45, 12), np.array([0.6, 0.9, 1.3])])
    masses = np.linspace(1.0e5, 9.0e5, len(z))
    handler = _make_handler(z, masses)

    hosts = handler.draw_rate_weighted_hosts(500, rng=np.random.default_rng(1), z_max=0.5)

    assert len(hosts) == 500
    assert all(isinstance(h, HostGalaxy) for h in hosts)
    assert all(h.z < 0.5 for h in hosts), "draw returned a host with z >= z_max"


def test_draw_fixed_seed_reproducibility() -> None:
    """Two generators with the same seed yield identical draws."""
    z = np.linspace(0.01, 0.49, 25)
    masses = np.linspace(1.0e5, 9.0e5, len(z))
    handler = _make_handler(z, masses)

    a = handler.draw_rate_weighted_hosts(200, rng=np.random.default_rng(2024), z_max=0.5)
    b = handler.draw_rate_weighted_hosts(200, rng=np.random.default_rng(2024), z_max=0.5)

    assert [h.catalog_index for h in a] == [h.catalog_index for h in b]
    assert [h.M for h in a] == [h.M for h in b]
    assert [h.z for h in a] == [h.z for h in b]


def test_draw_different_seed_changes_draw() -> None:
    """Sanity: different seeds generally produce a different draw sequence."""
    z = np.linspace(0.01, 0.49, 30)
    masses = np.linspace(1.0e5, 9.0e5, len(z))
    handler = _make_handler(z, masses)

    idx_a = [
        h.catalog_index
        for h in handler.draw_rate_weighted_hosts(200, rng=np.random.default_rng(1), z_max=0.5)
    ]
    idx_b = [
        h.catalog_index
        for h in handler.draw_rate_weighted_hosts(200, rng=np.random.default_rng(2), z_max=0.5)
    ]
    assert idx_a != idx_b


def test_draw_frequencies_match_rate_weight() -> None:
    """Empirical host frequencies are proportional to R_eff_per_mbh(M_g)/(1+z_g).

    A chi-square goodness-of-fit compares the per-row selected frequencies to the
    expected weighted probabilities ``p_g = w_g / Σ w``. High-z rows (z >= z_max)
    must be excluded entirely.
    """
    # Distinct masses span the R_eff(M) power law; distinct redshifts span the
    # 1/(1+z) dilation; the last two rows lie above z_max and must drop out.
    z = np.array([0.05, 0.10, 0.18, 0.25, 0.33, 0.42, 0.47, 0.65, 0.95])
    masses = np.array([1.0e5, 2.0e5, 3.5e5, 5.0e5, 6.5e5, 8.0e5, 9.5e5, 4.0e5, 7.0e5])
    z_max = 0.5
    handler = _make_handler(z, masses)

    expected_p, eligible = _expected_probabilities(z, masses, z_max)
    n_draws = 60_000
    hosts = handler.draw_rate_weighted_hosts(n_draws, rng=np.random.default_rng(7), z_max=z_max)

    counts = pd.Series([h.catalog_index for h in hosts]).value_counts()
    observed = np.array([counts.get(int(pos), 0) for pos in eligible], dtype=np.float64)
    assert observed.sum() == n_draws

    expected_counts = expected_p * n_draws
    assert np.all(expected_counts > 5), "chi-square validity requires expected counts > 5"
    _stat, p_value = stats.chisquare(f_obs=observed, f_exp=expected_counts)
    assert p_value > 0.01, (
        f"host frequencies are not proportional to R_eff_per_mbh(M)/(1+z) "
        f"(chi-square p={p_value:.4g}); the draw weighting is wrong."
    )

    # No galaxy above z_max was ever drawn.
    drawn = {h.catalog_index for h in hosts}
    for pos in np.flatnonzero(z >= z_max):
        assert int(pos) not in drawn, "a galaxy with z >= z_max was drawn"


def test_draw_uses_source_frame_mass_and_redshift() -> None:
    """Two-row catalog pins the exact w = R_eff(M)/(1+z) split (not flat/volume).

    Row 0 (low z, low M) has the larger weight via both higher R_eff (shallow
    M^-0.13 law) and larger 1/(1+z); row 1 the smaller. The empirical fraction of
    row-0 draws must match the analytic ``p_0`` within a binomial 4-sigma band.
    """
    z = np.array([0.05, 0.45])
    masses = np.array([1.0e5, 9.0e5])
    handler = _make_handler(z, masses)

    expected_p, _ = _expected_probabilities(z, masses, z_max=0.5)
    n_draws = 80_000
    hosts = handler.draw_rate_weighted_hosts(n_draws, rng=np.random.default_rng(11), z_max=0.5)

    counts = pd.Series([h.catalog_index for h in hosts]).value_counts()
    frac0 = counts.get(0, 0) / n_draws
    sem = float(np.sqrt(expected_p[0] * (1.0 - expected_p[0]) / n_draws))
    assert abs(frac0 - float(expected_p[0])) < 4 * sem, (
        f"row-0 draw fraction {frac0:.4f} != expected {float(expected_p[0]):.4f} "
        f"(weight is not R_eff(M)/(1+z))."
    )
    # The weighted split is not the equal-weight 0.5 (guards against a silent
    # fallback to draw_uniform_hosts behavior).
    assert abs(float(expected_p[0]) - 0.5) > 0.02


def test_draw_raises_when_no_eligible_galaxy() -> None:
    """A clear ValueError is raised when no galaxy satisfies z < z_max."""
    z = np.array([0.6, 0.7, 0.9, 1.2])  # all above z_max
    masses = np.linspace(1.0e5, 9.0e5, len(z))
    handler = _make_handler(z, masses)
    with pytest.raises(ValueError, match="redshift < z_max"):
        handler.draw_rate_weighted_hosts(10, rng=np.random.default_rng(4), z_max=0.5)


def test_draw_default_z_max_is_host_draw_constant() -> None:
    """The default z_max is the HOST_DRAW_Z_MAX constant (0.5)."""
    z = np.concatenate([np.linspace(0.02, 0.45, 10), np.array([HOST_DRAW_Z_MAX + 0.1])])
    masses = np.linspace(1.0e5, 9.0e5, len(z))
    handler = _make_handler(z, masses)
    hosts = handler.draw_rate_weighted_hosts(200, rng=np.random.default_rng(5))
    assert all(h.z < HOST_DRAW_Z_MAX for h in hosts)


# ---------------------------------------------------------------------------
# B. weighted_ratio_of_sums
# ---------------------------------------------------------------------------


def test_wros_scaling_invariance() -> None:
    """Rescaling all weights by an arbitrary positive constant leaves the result
    unchanged -- the overall normalization (including emri_rate.C_NORM) cancels.
    """
    rng = np.random.default_rng(0)
    num = rng.uniform(0.1, 10.0, size=7).tolist()
    den = rng.uniform(0.1, 10.0, size=7).tolist()
    w = rng.uniform(0.1, 10.0, size=7).tolist()

    base = weighted_ratio_of_sums(num, den, w)
    for c in (1e-6, 0.5, 3.0, 1e6):
        scaled = weighted_ratio_of_sums(num, den, [c * wi for wi in w])
        assert abs(scaled - base) < 1e-10 * max(1.0, abs(base)), (
            f"scaling weights by {c} changed the ratio (not scale-invariant)."
        )


def test_wros_constant_weight_limit_equals_plain_ratio() -> None:
    """Equal weights reproduce the plain Change-2 ratio sum(N)/sum(D)."""
    rng = np.random.default_rng(1)
    num = rng.uniform(0.1, 10.0, size=9).tolist()
    den = rng.uniform(0.1, 10.0, size=9).tolist()
    plain = sum(num) / sum(den)

    for w0 in (1.0, 0.25, 17.0):
        weighted = weighted_ratio_of_sums(num, den, [w0] * len(num))
        assert abs(weighted - plain) < 1e-10, (
            f"constant weight {w0} did not reproduce the equal-weight ratio of sums."
        )


def test_wros_hand_computed_example() -> None:
    """Small example with a known answer.

    N = [2, 4], D = [1, 1], w = [3, 1]:
      Σ w·N = 3·2 + 1·4 = 10; Σ w·D = 3·1 + 1·1 = 4; ratio = 2.5.
    """
    assert abs(weighted_ratio_of_sums([2.0, 4.0], [1.0, 1.0], [3.0, 1.0]) - 2.5) < 1e-12


def test_wros_nonpositive_denominator_returns_zero() -> None:
    """A non-positive weighted denominator returns 0.0 (matches the unweighted guard)."""
    assert weighted_ratio_of_sums([1.0, 2.0], [0.0, 0.0], [1.0, 1.0]) == 0.0
    assert weighted_ratio_of_sums([], [], []) == 0.0
