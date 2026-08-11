"""Correctness gate for the semi-analytic with-BH-mass selection denominator.

``_bh_mass_denominator_inner_m_integral`` computes the exact inner mass integral
``g(z) = INTEGRAL p_det(d_L(z), M(1+z)) N(M; mu, sigma) dM`` in closed form
(erf-sum), exploiting that ``p_det`` is piecewise-linear in ``M_z`` on the
injection interpolator grid. This test proves that closed form against an
*independent* adaptive-quadrature reference that subdivides at the kink sites,
plus the constant-detectability limiting case. It is the /physics-change
correctness proof for the estimator that replaced the 10k-sample Monte-Carlo
(``[PHYSICS]`` 2026-07-08).

The ``_Grid2DPdet`` stub mirrors the production ``SimulationDetectionProbability``
2-D interface exactly (``method="linear"`` bilinear + clamp ``M_z``/``d_L`` to the
grid range + clip to ``[0, 1]``), so the erf-sum's piecewise-linear assumption is
the true behaviour of the object under test. It is reused by the 4D branch of
``test_kernel_parity``.
"""

import numpy as np
import numpy.typing as npt
import pytest
from scipy.integrate import quad
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import norm

from darksiren_emri.bayesian_inference.bayesian_statistics import (
    _bh_mass_denominator_inner_m_integral,
)
from darksiren_emri.physical_relations import dist_vectorized


class _Grid2DPdet:
    """Real 2-D ``RegularGridInterpolator`` p_det mirroring production clamp+bilinear+clip.

    Matches ``SimulationDetectionProbability.detection_probability_with_bh_mass_interpolated``
    (``simulation_detection_probability.py``): ``d_L`` and ``M_z`` are clipped to
    the grid range (nearest-edge / constant-clamp), bilinearly interpolated, and
    the result clipped to ``[0, 1]``. ``_get_or_build_grid`` returns the 2-D
    interpolator first (its ``.grid[1]`` are the ``M_z`` knots the erf-sum reads).
    """

    def __init__(
        self,
        dl_centers: npt.NDArray[np.float64],
        m_centers: npt.NDArray[np.float64],
        grid_vals: npt.NDArray[np.float64],
    ) -> None:
        self._dl = dl_centers
        self._m = m_centers
        self._interp = RegularGridInterpolator(
            (dl_centers, m_centers), grid_vals, method="linear", bounds_error=False, fill_value=None
        )

    def _get_or_build_grid(self, h: float) -> tuple[RegularGridInterpolator, None]:
        return self._interp, None

    def detection_probability_with_bh_mass_interpolated(
        self,
        d_L: npt.NDArray[np.float64],
        M_z: npt.NDArray[np.float64],
        phi: npt.NDArray[np.float64],
        theta: npt.NDArray[np.float64],
        h: float,
    ) -> npt.NDArray[np.float64]:
        d = np.clip(np.asarray(d_L, dtype=np.float64), self._dl[0], self._dl[-1])
        m = np.clip(np.asarray(M_z, dtype=np.float64), self._m[0], self._m[-1])
        result = np.clip(self._interp(np.column_stack([d, m])), 0.0, 1.0)
        return np.asarray(result, dtype=np.float64)


def make_grid2d_pdet(kind: str = "peaked") -> _Grid2DPdet:
    """Build a representative 2-D p_det stub (log-spaced M_z, linear d_L)."""
    dl = np.linspace(0.05, 40.0, 55)
    m = np.logspace(4.2, 6.8, 41)
    if kind == "const_half":
        vals = np.full((dl.size, m.size), 0.5)
    else:
        base = np.exp(-dl[:, None] / 8.0) * np.exp(-((np.log10(m)[None, :] - 5.5) ** 2) / 0.6)
        vals = np.clip(base, 0.0, 1.0)
    return _Grid2DPdet(dl, m, np.asarray(vals, dtype=np.float64))


def _reference_inner(
    dp: _Grid2DPdet, z: float, mu: float, sigma: float, h: float, phi: float, qs: float
) -> float:
    """Independent adaptive-quad reference for the inner M-integral, kink-subdivided."""
    d_l = float(dist_vectorized(np.array([z]), h=h)[0])

    def integrand(mass: float) -> float:
        p = dp.detection_probability_with_bh_mass_interpolated(
            np.array([d_l]), np.array([mass * (1.0 + z)]), np.array([phi]), np.array([qs]), h=h
        )[0]
        return float(p) * float(norm.pdf(mass, loc=mu, scale=sigma))

    a, b = mu - 8.0 * sigma, mu + 8.0 * sigma
    # kink sites in M: where M(1+z) crosses an M_z grid node.
    kinks = dp._m / (1.0 + z)
    pts = sorted(float(k) for k in kinks if a < k < b)
    value, _ = quad(integrand, a, b, points=pts if pts else None, limit=400)
    return float(value)


_CASES: list[tuple[float, float, float]] = [
    # (z, host_M_eff, sigma_M): near/far, spec-z-like/photo-z-like mass widths.
    (0.05, 5.0e5, 5.0e4),
    (0.10, 3.0e5, 3.0e4),
    (0.10, 3.0e5, 1.5e5),  # wide mass error -> spans many kinks
    (0.30, 1.0e6, 2.0e5),
    (0.50, 8.0e5, 1.0e5),
    (0.02, 2.0e5, 5.0e4),
]


@pytest.mark.parametrize("z,mu,sigma", _CASES)
def test_inner_m_integral_matches_adaptive_quad(z: float, mu: float, sigma: float) -> None:
    """The erf-sum reproduces an independent kink-subdivided adaptive quad to 1e-8."""
    dp = make_grid2d_pdet("peaked")
    h = 0.73
    got = float(_bh_mass_denominator_inner_m_integral(np.array([z]), dp, 1.2, 1.0, mu, sigma, h)[0])
    ref = _reference_inner(dp, z, mu, sigma, h, 1.2, 1.0)
    assert got == pytest.approx(ref, rel=1e-8, abs=1e-15)


def test_inner_m_integral_constant_detectability_limit() -> None:
    """p_det == 0.5 everywhere -> inner M-integral == 0.5 (full-Gaussian normalisation)."""
    dp = make_grid2d_pdet("const_half")
    h = 0.73
    z = np.array([0.02, 0.1, 0.3, 0.5])
    got = _bh_mass_denominator_inner_m_integral(z, dp, 1.2, 1.0, 4.0e5, 8.0e4, h)
    np.testing.assert_allclose(got, 0.5, rtol=1e-10, atol=1e-12)


def test_inner_m_integral_vectorized_matches_scalar() -> None:
    """Vectorized z-array call equals per-z scalar calls (no cross-z contamination)."""
    dp = make_grid2d_pdet("peaked")
    h = 0.73
    z = np.array([0.03, 0.12, 0.4])
    vec = _bh_mass_denominator_inner_m_integral(z, dp, 1.2, 1.0, 6.0e5, 1.2e5, h)
    scal = np.array(
        [
            float(
                _bh_mass_denominator_inner_m_integral(
                    np.array([zi]), dp, 1.2, 1.0, 6.0e5, 1.2e5, h
                )[0]
            )
            for zi in z
        ]
    )
    np.testing.assert_allclose(vec, scal, rtol=1e-12, atol=0.0)
