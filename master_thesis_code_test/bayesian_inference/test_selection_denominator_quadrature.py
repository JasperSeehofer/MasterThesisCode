"""Tests for the deterministic Gauss-Legendre selection denominator.

These cover :func:`_selection_denominator_with_bh_mass`, which replaced the former
unseeded Monte-Carlo importance sampler whose ~1% per-(host, H0) noise produced a
white-noise comb in the with-BH-mass H0 posterior. The tests assert:

* determinism (bit-identical across reruns — the whole point of the change),
* node-doubling convergence (default order is converged to << 1% MC noise),
* agreement with an adaptive ``dblquad`` ground truth on the same window,
* the constant-``p_det`` limiting case, and
* the lower-mass clip (no unphysical negative mass reaches ``p_det``).

A smooth analytic mock ``p_det`` is used so the quadrature math is validated without
the full injection grid / galaxy catalog (CPU-only, no GPU).
"""

import math

import numpy as np
import numpy.typing as npt
from scipy.integrate import dblquad
from scipy.stats import norm

import master_thesis_code.bayesian_inference.bayesian_statistics as bs
from master_thesis_code.physical_relations import dist

# +/-4 sigma window normalization of a single Gaussian axis (no clipping).
_GAUSS_4SIGMA_MASS = math.erf(4.0 / math.sqrt(2.0))


class _SmoothMockDetectionProbability:
    """Smooth, bounded p_det(d_L, M_z): logistic falloff in d_L, gentle rise in log10 M_z."""

    def detection_probability_with_bh_mass_interpolated(
        self,
        d_L: npt.NDArray[np.float64],
        M_z: npt.NDArray[np.float64],
        phi: npt.NDArray[np.float64],
        theta: npt.NDArray[np.float64],
        h: float,
    ) -> npt.NDArray[np.float64]:
        d_L = np.asarray(d_L, dtype=float)
        M_z = np.asarray(M_z, dtype=float)
        falloff = 1.0 / (1.0 + np.exp((d_L - 4000.0) / 800.0))
        rise = 1.0 / (1.0 + np.exp(-(np.log10(M_z) - 6.0) / 0.5))
        return falloff * rise


class _ConstantMockDetectionProbability:
    """p_det == const everywhere (for the analytic limiting-case test)."""

    def __init__(self, value: float) -> None:
        self.value = value

    def detection_probability_with_bh_mass_interpolated(
        self,
        d_L: npt.NDArray[np.float64],
        M_z: npt.NDArray[np.float64],
        phi: npt.NDArray[np.float64],
        theta: npt.NDArray[np.float64],
        h: float,
    ) -> npt.NDArray[np.float64]:
        return np.full(np.asarray(d_L, dtype=float).shape, self.value)


class _RecordingMockDetectionProbability:
    """Records every M_z it is queried at, to assert positivity (clip guard)."""

    def __init__(self) -> None:
        self.seen_M_z: list[float] = []

    def detection_probability_with_bh_mass_interpolated(
        self,
        d_L: npt.NDArray[np.float64],
        M_z: npt.NDArray[np.float64],
        phi: npt.NDArray[np.float64],
        theta: npt.NDArray[np.float64],
        h: float,
    ) -> npt.NDArray[np.float64]:
        M_z = np.asarray(M_z, dtype=float)
        self.seen_M_z.extend(M_z.ravel().tolist())
        return np.full(M_z.shape, 0.5)


_H = 0.7
_MULT = 4.0


def _window(host_z: float, host_z_err: float) -> tuple[float, float]:
    return host_z - _MULT * host_z_err, host_z + _MULT * host_z_err


def test_determinism_bit_identical() -> None:
    """Two identical calls must return exactly the same float (the core fix)."""
    dp = _SmoothMockDetectionProbability()
    host_z, host_z_err, host_M, host_M_err = 0.30, 0.01, 3.0e6, 2.0e5
    zlo, zhi = _window(host_z, host_z_err)
    zdist, mdist = norm(host_z, host_z_err), norm(host_M, host_M_err)

    args = (dp, zdist, mdist, host_M, host_M_err, 1.0, 0.5, _H, zlo, zhi, _MULT)
    d1 = bs._selection_denominator_with_bh_mass(*args)
    d2 = bs._selection_denominator_with_bh_mass(*args)
    assert d1 == d2  # bit-identical, no RNG


def test_node_doubling_convergence() -> None:
    """Default (32, 24) order must agree with doubled order to << 1% (the old MC noise)."""
    dp = _SmoothMockDetectionProbability()
    host_z, host_z_err, host_M, host_M_err = 0.30, 0.01, 3.0e6, 2.0e5
    zlo, zhi = _window(host_z, host_z_err)
    zdist, mdist = norm(host_z, host_z_err), norm(host_M, host_M_err)
    base = (dp, zdist, mdist, host_M, host_M_err, 1.0, 0.5, _H, zlo, zhi, _MULT)

    d_default = bs._selection_denominator_with_bh_mass(*base)
    d_doubled = bs._selection_denominator_with_bh_mass(*base, n_z=64, n_M=48)
    assert abs(d_default - d_doubled) / d_doubled < 1e-3


def test_matches_adaptive_dblquad_reference() -> None:
    """GL quadrature must match an adaptive dblquad on the same window to < 1e-3."""
    dp = _SmoothMockDetectionProbability()
    host_z, host_z_err, host_M, host_M_err = 0.30, 0.01, 3.0e6, 2.0e5
    zlo, zhi = _window(host_z, host_z_err)
    mlo = max(host_M - _MULT * host_M_err, bs.DENOMINATOR_QUAD_M_FLOOR_FRACTION * host_M)
    mhi = host_M + _MULT * host_M_err
    zdist, mdist = norm(host_z, host_z_err), norm(host_M, host_M_err)

    gl = bs._selection_denominator_with_bh_mass(
        dp, zdist, mdist, host_M, host_M_err, 1.0, 0.5, _H, zlo, zhi, _MULT, n_z=64, n_M=48
    )

    def integrand(m: float, z: float) -> float:
        d_l = dist(z, h=_H)
        p = float(
            dp.detection_probability_with_bh_mass_interpolated(
                np.array([d_l]), np.array([m * (1 + z)]), np.array([1.0]), np.array([0.5]), _H
            )[0]
        )
        return p * float(zdist.pdf(z)) * float(mdist.pdf(m))

    reference, _ = dblquad(integrand, zlo, zhi, mlo, mhi)
    assert abs(gl - reference) / reference < 1e-3


def test_constant_pdet_limiting_case() -> None:
    """For p_det == c (no clip), D_g = c * [erf(4/sqrt2)]^2 (the +/-4 sigma window mass)."""
    c = 0.42
    dp = _ConstantMockDetectionProbability(c)
    # host with host_M - 4 sigma_M > 0 so the lower-M clip does NOT trigger.
    host_z, host_z_err, host_M, host_M_err = 0.30, 0.01, 3.0e6, 2.0e5
    zlo, zhi = _window(host_z, host_z_err)
    zdist, mdist = norm(host_z, host_z_err), norm(host_M, host_M_err)

    d_g = bs._selection_denominator_with_bh_mass(
        dp, zdist, mdist, host_M, host_M_err, 1.0, 0.5, _H, zlo, zhi, _MULT
    )
    expected = c * _GAUSS_4SIGMA_MASS**2
    assert abs(d_g - expected) / expected < 1e-3


def test_lower_mass_clip_keeps_pdet_positive() -> None:
    """A large-sigma_M host (host_M - 4 sigma_M < 0) must never query p_det at M_z <= 0."""
    dp = _RecordingMockDetectionProbability()
    host_z, host_z_err = 0.30, 0.01
    host_M, host_M_err = 3.0e6, 1.0e6  # host_M - 4*sigma_M = -1e6 < 0 -> clip triggers
    zlo, zhi = _window(host_z, host_z_err)
    zdist, mdist = norm(host_z, host_z_err), norm(host_M, host_M_err)

    bs._selection_denominator_with_bh_mass(
        dp, zdist, mdist, host_M, host_M_err, 1.0, 0.5, _H, zlo, zhi, _MULT
    )
    assert dp.seen_M_z, "p_det was never queried"
    assert min(dp.seen_M_z) > 0.0
