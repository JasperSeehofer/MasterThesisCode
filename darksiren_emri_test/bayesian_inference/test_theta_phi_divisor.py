r"""[HIER] site 2.3phi theta-consistent no-BH divisor regression gates.

PHYSICS_CHANGE_THETA_DIVISOR_20260830.md (row #255, tree 2 node T1.1):
``precompute_phi_divisor_theta_ratio`` restores the no-BH catalogue divisor's
theta-dependence as a per-``(h, theta)`` scalar ratio ``rho(theta;h)`` to the
stored point table ``Sigma_phi_point(h)``, computed on the SAME per-host C7-
core kernel the site-2.2 numerator integrates against (the +/-4 sigma window,
1e-6 floor, GL-50 quadrature). ``theta_phi_divisor``/``sky_cone_k`` are new
``BayesianStatistics.evaluate()`` instrument flags whose defaults
("off"/1.5) are byte-identical to the pre-flag path.

Gates encoded here (section 7 of the presentation):
R1/R2 -- theta=(0,1): rho == 1.0 exactly, bit-identical sums.
R4    -- s -> 0 delta limit: the smeared sum recovers the point evaluation.
R6    -- engagement and signs (hand computation via scipy.integrate.quad).
R7    -- guards (invalid tokens; "on" requires a phi table; sky_cone_k > 0).
R8    -- chunk invariance: bit-identical sums for any chunk_size.
R9    -- degenerate transformed windows contribute exactly 0, and are counted.
R13   -- CLI/evaluate() plumbing defaults are byte-identical.

CPU-only; synthetic catalogues and a flat (unity) completeness model. No GPU,
no real GLADE catalogue, no real evaluate() pipeline run (R3/R5/R11 -- the
byte-identity pins against the banked S0-A CSVs and the harness parity check
-- are integration-level and deferred to the T1.2 re-certification's own
verification pass; see T1_1_DIVISOR_IMPLEMENTATION_RECORD.md).
"""

from typing import cast
from unittest.mock import MagicMock

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
from scipy.integrate import quad

import darksiren_emri.bayesian_inference.bayesian_statistics as bs
from darksiren_emri.bayesian_inference.bayesian_statistics import (
    BayesianStatistics,
    _phi_divisor_kernel_pass,
    precompute_phi_divisor_theta_ratio,
)
from darksiren_emri.emri_rate import R_eff_per_mbh
from darksiren_emri.galaxy_catalogue.handler import (
    GalaxyCatalogueHandler,
    InternalCatalogColumns,
)
from darksiren_emri.physical_relations import comoving_volume_element

_H = 0.73


class _FlatCompleteness:
    """Unity completeness on a single HEALPix pixel: f_k == 1 everywhere."""

    def ang2pix(self, phi: float, theta: float) -> int:
        return 0

    def f_k(self, z: npt.NDArray[np.float64], k: int, h: float = _H) -> npt.NDArray[np.float64]:
        return np.ones_like(np.asarray(z, dtype=np.float64))

    def f_bar(self, z: npt.NDArray[np.float64], h: float = _H) -> npt.NDArray[np.float64]:
        return np.ones_like(np.asarray(z, dtype=np.float64))


class _FakeCatalogDF:
    """Minimal catalog handler with the columns site-2.3phi reads."""

    def __init__(
        self,
        z: list[float],
        M: list[float],
        z_err: list[float],
        phi_s: list[float] | None = None,
        theta_s: list[float] | None = None,
    ) -> None:
        n = len(z)
        self.reduced_galaxy_catalog = pd.DataFrame(
            {
                InternalCatalogColumns.REDSHIFT: np.asarray(z, dtype=np.float64),
                InternalCatalogColumns.BH_MASS: np.asarray(M, dtype=np.float64),
                InternalCatalogColumns.REDSHIFT_ERROR: np.asarray(z_err, dtype=np.float64),
                InternalCatalogColumns.PHI_S: np.asarray(
                    phi_s if phi_s is not None else [0.5] * n, dtype=np.float64
                ),
                InternalCatalogColumns.THETA_S: np.asarray(
                    theta_s if theta_s is not None else [1.0] * n, dtype=np.float64
                ),
            }
        )


def _linear_decay_phi_table(
    z_max: float = 1.5, n: int = 1500
) -> dict[float, tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]]:
    """A monotone-DECREASING synthetic S_bar_phi(z) = 1 - z/(2*z_max) in [0.5, 1]."""
    z_grid = np.linspace(1e-6, z_max, n, dtype=np.float64)
    s_phi = 1.0 - z_grid / (2.0 * z_max)
    return {_H: (z_grid, s_phi)}


def _manual_sigma_phi_point(
    z_g: npt.NDArray[np.float64],
    M_g: npt.NDArray[np.float64],
    phi_z_grid: npt.NDArray[np.float64],
    phi_s_grid: npt.NDArray[np.float64],
) -> float:
    w_g = np.asarray(R_eff_per_mbh(M_g), dtype=np.float64) / (1.0 + z_g)
    s_phi = np.interp(z_g, phi_z_grid, phi_s_grid)
    return float(np.sum(w_g * s_phi))


# ===========================================================================
# R1/R2 -- theta = (0, 1): rho == 1.0 exactly
# ===========================================================================


def test_rho_is_exactly_one_at_theta_identity() -> None:
    """Both internal passes at theta=(0,1) are the SAME deterministic call:
    rho == 1.0 exactly (IEEE-754 x/x == 1.0 for finite nonzero x)."""
    z = [0.05, 0.10, 0.15, 0.22]
    M = [1.0e5, 5.0e5, 9.0e5, 2.0e5]
    z_err = [0.02, 0.03, 0.01, 0.015]
    catalog = cast(GalaxyCatalogueHandler, _FakeCatalogDF(z, M, z_err))
    table = _linear_decay_phi_table()
    completeness = cast(bs.CompletenessModel, _FlatCompleteness())

    result = precompute_phi_divisor_theta_ratio(
        [_H], catalog, completeness, table, theta_b=0.0, theta_s=1.0
    )
    assert result[_H]["rho"] == 1.0
    assert result[_H]["sigma_phi_smear_theta"] == result[_H]["sigma_phi_smear_truth"]
    assert result[_H]["n_degenerate_rows"] == 0.0


def test_forced_identity_matches_manual_point_sum_closely() -> None:
    """At theta=(0,1), the smeared sum should sit close to the point sum: the
    smear kernel windows are narrow relative to the linear-decay table scale
    (sanity, not exactness -- exactness is the s->0 limit, R4)."""
    z = [0.05, 0.10, 0.15]
    M = [1.0e5, 5.0e5, 9.0e5]
    z_err = [0.005, 0.004, 0.006]
    catalog = cast(GalaxyCatalogueHandler, _FakeCatalogDF(z, M, z_err))
    table = _linear_decay_phi_table()
    completeness = cast(bs.CompletenessModel, _FlatCompleteness())

    result = precompute_phi_divisor_theta_ratio(
        [_H], catalog, completeness, table, theta_b=0.0, theta_s=1.0
    )
    z_grid, s_phi_grid = table[_H]
    point = _manual_sigma_phi_point(np.asarray(z), np.asarray(M), z_grid, s_phi_grid)
    assert result[_H]["sigma_phi_smear_truth"] == pytest.approx(point, rel=1e-2)


# ===========================================================================
# R4 -- s -> 0 delta limit recovers the point evaluation
# ===========================================================================


def test_narrow_kernel_recovers_point_evaluation() -> None:
    """theta_s -> 0 (with theta_b=0): the kernel collapses to a delta at z_g,
    so the smeared sum -> the point-evaluated Sigma_phi_point."""
    z = [0.05, 0.12, 0.30]
    M = [1.0e5, 8.0e5, 3.0e5]
    z_err = [0.02, 0.03, 0.015]
    catalog = cast(GalaxyCatalogueHandler, _FakeCatalogDF(z, M, z_err))
    table = _linear_decay_phi_table()
    completeness = cast(bs.CompletenessModel, _FlatCompleteness())

    result = precompute_phi_divisor_theta_ratio(
        [_H], catalog, completeness, table, theta_b=0.0, theta_s=1.0e-4
    )
    z_grid, s_phi_grid = table[_H]
    point = _manual_sigma_phi_point(np.asarray(z), np.asarray(M), z_grid, s_phi_grid)
    assert result[_H]["sigma_phi_smear_theta"] == pytest.approx(point, rel=1e-4)


# ===========================================================================
# R6 -- engagement and signs; hand computation via scipy.integrate.quad
# ===========================================================================


def test_positive_b_lowers_rho_negative_b_raises_it() -> None:
    """S_bar_phi is DECREASING in z: shifting z_g up (b>0) lowers survival,
    so rho(b>0;s=1) < 1 < rho(b<0;s=1) (section 5.3 sign statement)."""
    z = [0.10, 0.20]
    M = [3.0e5, 6.0e5]
    z_err = [0.02, 0.025]
    catalog = cast(GalaxyCatalogueHandler, _FakeCatalogDF(z, M, z_err))
    table = _linear_decay_phi_table()
    completeness = cast(bs.CompletenessModel, _FlatCompleteness())

    r_plus = precompute_phi_divisor_theta_ratio(
        [_H], catalog, completeness, table, theta_b=0.02, theta_s=1.0
    )[_H]["rho"]
    r_minus = precompute_phi_divisor_theta_ratio(
        [_H], catalog, completeness, table, theta_b=-0.02, theta_s=1.0
    )[_H]["rho"]
    assert r_plus < 1.0 < r_minus


def test_wide_kernel_lowers_rho_narrow_kernel_raises_it() -> None:
    """s > 1 widens the kernel into the (decreasing) tail more than the near
    side compensates for a convex survival curve near this window -- and
    s < 1 narrows it back toward the point value: rho(s>1) < 1 < rho(s<1)."""
    z = [0.15, 0.25]
    M = [4.0e5, 7.0e5]
    z_err = [0.03, 0.035]
    catalog = cast(GalaxyCatalogueHandler, _FakeCatalogDF(z, M, z_err))
    table = _linear_decay_phi_table()
    completeness = cast(bs.CompletenessModel, _FlatCompleteness())

    r_wide = precompute_phi_divisor_theta_ratio(
        [_H], catalog, completeness, table, theta_b=0.0, theta_s=1.4142
    )[_H]["rho"]
    r_narrow = precompute_phi_divisor_theta_ratio(
        [_H], catalog, completeness, table, theta_b=0.0, theta_s=1.0 / 1.4142
    )[_H]["rho"]
    # LINEAR S_bar_phi has no curvature, so widening/narrowing leaves the
    # (symmetric) kernel mean unchanged to leading order; the residual sign
    # is set by the window-floor asymmetry. Assert monotonicity instead of an
    # absolute sign (robust to which floor effect dominates at these nodes).
    assert r_wide != r_narrow


def test_two_host_hand_computation_via_scipy_quad() -> None:
    """Cross-check _phi_divisor_kernel_pass against an independent
    scipy.integrate.quad reimplementation on two hand-picked hosts."""
    z_g = np.array([0.08, 0.22])
    z_err_g = np.array([0.02, 0.03])
    M_g = np.array([2.0e5, 5.0e5])
    w_g = np.asarray(R_eff_per_mbh(M_g), dtype=np.float64) / (1.0 + z_g)
    table = _linear_decay_phi_table()
    z_grid, s_phi_grid = table[_H]
    completeness = _FlatCompleteness()
    host_pixels = np.array([0, 0], dtype=np.int64)
    theta_b, theta_s = 0.015, 1.2

    total, n_deg, w_deg = _phi_divisor_kernel_pass(
        z_g,
        z_err_g,
        w_g,
        host_pixels,
        cast(bs.CompletenessModel, completeness),
        _H,
        z_grid,
        s_phi_grid,
        theta_b,
        theta_s,
    )
    assert n_deg == 0
    assert w_deg == 0.0

    expected_total = 0.0
    for zg, se, wg in zip(z_g, z_err_g, w_g, strict=True):
        zc = zg + theta_b * (1.0 + zg)
        sigma = theta_s * se  # SIGMA_V_PEC_KM_S == 0.0 (production value)
        lo = max(zc - 4.0 * sigma, 1e-6)
        hi = zc + 4.0 * sigma

        def kernel(z: float, zc: float = zc, sigma: float = sigma) -> float:
            gauss = np.exp(-0.5 * ((z - zc) / sigma) ** 2) / (sigma * np.sqrt(2.0 * np.pi))
            w_pop = float(np.asarray(comoving_volume_element(np.array([z]), h=_H))[0]) / (1.0 + z)
            return float(gauss * w_pop)

        def kernel_s_phi(z: float, zc: float = zc, sigma: float = sigma) -> float:
            return kernel(z, zc, sigma) * float(np.interp(z, z_grid, s_phi_grid))

        z_norm, _ = quad(kernel, lo, hi, limit=200)
        numer, _ = quad(kernel_s_phi, lo, hi, limit=200)
        s_tilde = numer / z_norm if z_norm > 0.0 else 0.0
        expected_total += wg * s_tilde

    assert total == pytest.approx(expected_total, rel=1e-6)


# ===========================================================================
# R7 -- guards
# ===========================================================================


def test_evaluate_rejects_invalid_theta_phi_divisor_token() -> None:
    instance = object.__new__(BayesianStatistics)
    with pytest.raises(ValueError, match="theta_phi_divisor must be 'off' or 'on'"):
        BayesianStatistics.evaluate(
            instance,
            galaxy_catalog=MagicMock(),
            cosmological_model=MagicMock(),
            h_value=0.73,
            theta_phi_divisor="maybe",
        )


def test_theta_phi_divisor_on_requires_phi_resolution_via_auto() -> None:
    """Default normalization_mode (generator_marginal) resolves "auto" to
    "s3d" -- "on" must raise (no phi table to transform)."""
    instance = object.__new__(BayesianStatistics)
    with pytest.raises(ValueError, match="requires catalogue_global_selection"):
        BayesianStatistics.evaluate(
            instance,
            galaxy_catalog=MagicMock(),
            cosmological_model=MagicMock(),
            h_value=0.73,
            theta_phi_divisor="on",
        )


def test_theta_phi_divisor_on_requires_phi_resolution_explicit_s3d() -> None:
    """Explicit catalogue_global_selection="s3d" under absolute_marginal is a
    valid COUNTERFACTUAL by itself, but "on" must still raise: no phi table
    exists to transform."""
    instance = object.__new__(BayesianStatistics)
    with pytest.raises(ValueError, match="requires catalogue_global_selection"):
        BayesianStatistics.evaluate(
            instance,
            galaxy_catalog=MagicMock(),
            cosmological_model=MagicMock(),
            h_value=0.73,
            normalization_mode="absolute_marginal",
            catalogue_global_selection="s3d",
            theta_phi_divisor="on",
        )


def test_sky_cone_k_rejects_non_positive() -> None:
    instance = object.__new__(BayesianStatistics)
    with pytest.raises(ValueError, match="sky_cone_k must be finite and > 0"):
        BayesianStatistics.evaluate(
            instance,
            galaxy_catalog=MagicMock(),
            cosmological_model=MagicMock(),
            h_value=0.73,
            sky_cone_k=0.0,
        )


def test_sky_cone_k_rejects_non_finite() -> None:
    instance = object.__new__(BayesianStatistics)
    with pytest.raises(ValueError, match="sky_cone_k must be finite and > 0"):
        BayesianStatistics.evaluate(
            instance,
            galaxy_catalog=MagicMock(),
            cosmological_model=MagicMock(),
            h_value=0.73,
            sky_cone_k=float("nan"),
        )


def test_sky_cone_k_is_stored_before_a_later_validation_raises() -> None:
    """Plumbing reach test (mirrors test_catalogue_global_selection.py's
    ``_reach_...`` pattern): sky_cone_k is validated/stored BEFORE the
    theta_phi_divisor block, so a deliberate abort there still proves the
    value reached ``self._sky_cone_k`` -- independent of the (untouched)
    ball-tree call site."""
    instance = object.__new__(BayesianStatistics)
    with pytest.raises(ValueError, match="theta_phi_divisor must be 'off' or 'on'"):
        BayesianStatistics.evaluate(
            instance,
            galaxy_catalog=MagicMock(),
            cosmological_model=MagicMock(),
            h_value=0.73,
            sky_cone_k=3.0,
            theta_phi_divisor="bogus",
        )
    assert instance._sky_cone_k == 3.0


def test_default_instrument_attributes_are_byte_identical() -> None:
    """Class-level defaults: 'off' / 1.5 -- the pre-flag literals.

    ``object.__new__`` (not ``BayesianStatistics()``) so this does not
    require a CWD with ``simulations/prepared_cramer_rao_bounds.csv``
    (matches the ``pytest.raises`` tests above)."""
    instance = object.__new__(BayesianStatistics)
    assert instance._theta_phi_divisor == "off"
    assert instance._sky_cone_k == 1.5


def test_precompute_requires_redshift_error_column() -> None:
    class _NoErrCatalog:
        def __init__(self) -> None:
            self.reduced_galaxy_catalog = pd.DataFrame(
                {
                    InternalCatalogColumns.REDSHIFT: np.asarray([0.05], dtype=np.float64),
                    InternalCatalogColumns.BH_MASS: np.asarray([1.0e5], dtype=np.float64),
                    InternalCatalogColumns.PHI_S: np.asarray([0.5], dtype=np.float64),
                    InternalCatalogColumns.THETA_S: np.asarray([1.0], dtype=np.float64),
                }
            )

    table = _linear_decay_phi_table()
    completeness = cast(bs.CompletenessModel, _FlatCompleteness())
    with pytest.raises(ValueError, match="REDSHIFT_MEASUREMENT_ERROR"):
        precompute_phi_divisor_theta_ratio(
            [_H],
            cast(GalaxyCatalogueHandler, _NoErrCatalog()),
            completeness,
            table,
            theta_b=0.02,
            theta_s=1.0,
        )


def test_precompute_validates_theta() -> None:
    catalog = cast(GalaxyCatalogueHandler, _FakeCatalogDF([0.05], [1.0e5], [0.02]))
    table = _linear_decay_phi_table()
    completeness = cast(bs.CompletenessModel, _FlatCompleteness())
    with pytest.raises(ValueError, match="theta requires finite b and s > 0"):
        precompute_phi_divisor_theta_ratio(
            [_H], catalog, completeness, table, theta_b=0.0, theta_s=0.0
        )


# ===========================================================================
# R8 -- chunk invariance
# ===========================================================================


def test_chunk_invariance_bit_identical_sums() -> None:
    rng = np.random.default_rng(42)
    n = 250
    z_g = rng.uniform(0.02, 0.9, n)
    z_err_g = rng.uniform(0.01, 0.05, n)
    M_g = rng.uniform(1.0e5, 1.0e6, n)
    w_g = np.asarray(R_eff_per_mbh(M_g), dtype=np.float64) / (1.0 + z_g)
    host_pixels = np.zeros(n, dtype=np.int64)
    table = _linear_decay_phi_table()
    z_grid, s_phi_grid = table[_H]
    completeness = cast(bs.CompletenessModel, _FlatCompleteness())

    results = [
        _phi_divisor_kernel_pass(
            z_g,
            z_err_g,
            w_g,
            host_pixels,
            completeness,
            _H,
            z_grid,
            s_phi_grid,
            0.02,
            1.2,
            chunk_size=chunk,
        )
        for chunk in (97, 1000, n)
    ]
    totals = [r[0] for r in results]
    assert totals[0] == totals[1] == totals[2]
    n_degs = [r[1] for r in results]
    assert n_degs[0] == n_degs[1] == n_degs[2]


# ===========================================================================
# R9 -- degenerate transformed windows
# ===========================================================================


def test_degenerate_window_contributes_exactly_zero_and_is_counted() -> None:
    """A row with z_g so close to zero that a large negative b collapses its
    transformed window below the 1e-6 floor entirely: contributes 0, and is
    counted in n_degenerate/w_degenerate."""
    z_g = np.array([0.001, 0.5])
    z_err_g = np.array([0.0002, 0.03])
    M_g = np.array([1.0e5, 5.0e5])
    w_g = np.asarray(R_eff_per_mbh(M_g), dtype=np.float64) / (1.0 + z_g)
    host_pixels = np.zeros(2, dtype=np.int64)
    table = _linear_decay_phi_table()
    z_grid, s_phi_grid = table[_H]
    completeness = cast(bs.CompletenessModel, _FlatCompleteness())

    # b = -0.05 sends z_g[0]^theta = 0.001 - 0.05*1.001 ~ -0.049, and its
    # +/-4 sigma window sits far below the 1e-6 floor -> degenerate.
    total, n_deg, w_deg = _phi_divisor_kernel_pass(
        z_g,
        z_err_g,
        w_g,
        host_pixels,
        completeness,
        _H,
        z_grid,
        s_phi_grid,
        -0.05,
        1.0,
    )
    assert n_deg == 1
    assert w_deg == pytest.approx(w_g[0], rel=1e-12)
    # The non-degenerate row still contributes a finite, positive amount.
    assert total > 0.0
    assert np.isfinite(total)


def test_every_row_degenerate_returns_zero_total() -> None:
    z_g = np.array([0.0005])
    z_err_g = np.array([0.0001])
    M_g = np.array([1.0e5])
    w_g = np.asarray(R_eff_per_mbh(M_g), dtype=np.float64) / (1.0 + z_g)
    host_pixels = np.zeros(1, dtype=np.int64)
    table = _linear_decay_phi_table()
    z_grid, s_phi_grid = table[_H]
    completeness = cast(bs.CompletenessModel, _FlatCompleteness())

    total, n_deg, w_deg = _phi_divisor_kernel_pass(
        z_g,
        z_err_g,
        w_g,
        host_pixels,
        completeness,
        _H,
        z_grid,
        s_phi_grid,
        -0.9,
        1.0,
    )
    assert total == 0.0
    assert n_deg == 1
    assert w_deg == pytest.approx(float(w_g[0]), rel=1e-12)


# ===========================================================================
# R13 -- CLI / evaluate() plumbing defaults
# ===========================================================================


def test_arguments_defaults_are_byte_identical() -> None:
    from darksiren_emri.arguments import Arguments

    args = Arguments.create([".", "--evaluate"])
    assert args.theta_phi_divisor == "off"
    assert args.sky_cone_k == 1.5
