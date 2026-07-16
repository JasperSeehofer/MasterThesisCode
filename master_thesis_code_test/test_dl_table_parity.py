"""Category-A parity gate for the d_L spline-table drop-in (/physics-change 2026-07-08).

``dist_vectorized`` and ``dist_to_redshift`` now use precomputed clamped
cubic-spline tables of the comoving integral ``I(z)`` (and its monotone inverse)
on the fiducial-LCDM, in-range fast path, falling back to the exact ``hyp2f1`` /
``fsolve`` paths otherwise. These tests pin the *parity bound*: the table must
agree with the incumbent to ``<<`` the ~5% d_L measurement error (and, for the
hot path, below the ``rel=1e-9`` per-host-likelihood pin), so the H0 posterior
MAP cannot move. See ``experiments/eval_perf/FINDINGS.md`` sections 3.1/3.2.
"""

import numpy as np
import numpy.typing as npt

from master_thesis_code.constants import GPC_TO_MPC, KM_TO_M, OMEGA_DE, OMEGA_M, C
from master_thesis_code.physical_relations import (
    _DL_TABLE_Z_MAX,
    comoving_volume_element,
    dist_to_redshift,
    dist_vectorized,
    lambda_cdm_analytic_distance,
)


def _hyp2f1_baseline(z: npt.NDArray[np.float64], h: float) -> npt.NDArray[np.float64]:
    """Incumbent d_L via the exact hyp2f1 antiderivative (the pre-change formula)."""
    h_0 = h * 100.0 * KM_TO_M / GPC_TO_MPC ** (-1)
    integral = np.array(
        [float(lambda_cdm_analytic_distance(float(zz), OMEGA_M, OMEGA_DE)) for zz in z],
        dtype=np.float64,
    )
    return C / h_0 * (1.0 + z) * integral


def test_dist_vectorized_matches_hyp2f1_below_pin() -> None:
    """Table d_L agrees with hyp2f1 to < 1e-9 across the tabulated range and all run h."""
    z = np.linspace(1e-6, _DL_TABLE_Z_MAX, 5000)
    for h in (0.60, 0.67, 0.73, 0.80, 0.86):
        got = dist_vectorized(z, h=h)
        exp = _hyp2f1_baseline(z, h)
        max_rel = float(np.max(np.abs(got / exp - 1.0)))
        assert max_rel < 1e-9, f"h={h}: d_L table vs hyp2f1 max rel err {max_rel:.2e} >= 1e-9"


def test_dist_vectorized_falls_back_exactly_beyond_table() -> None:
    """z above the table range must reproduce the exact hyp2f1 value bit-for-bit."""
    z = np.array([_DL_TABLE_Z_MAX + 0.5, 3.0, 5.0])
    got = dist_vectorized(z, h=0.73)
    exp = _hyp2f1_baseline(z, 0.73)
    np.testing.assert_allclose(got, exp, rtol=0.0, atol=0.0)


def test_dist_vectorized_zero_and_monotone() -> None:
    """d_L(0)=0 exactly and d_L is strictly increasing in z."""
    assert float(dist_vectorized(np.array([0.0]), h=0.73)[0]) == 0.0
    z = np.linspace(0.0, _DL_TABLE_Z_MAX, 1000)
    d = dist_vectorized(z, h=0.73)
    assert np.all(np.diff(d) > 0.0)


def test_dist_to_redshift_roundtrip() -> None:
    """z -> d_L -> z round-trips to < 1e-7 across h (inverse-table vs fsolve target)."""
    z_true = np.linspace(1e-4, 1.5, 400)
    for h in (0.60, 0.73, 0.86):
        d = dist_vectorized(z_true, h=h)
        z_back = np.array([dist_to_redshift(float(dd), h=h) for dd in d])
        max_rel = float(np.max(np.abs(z_back / z_true - 1.0)))
        assert max_rel < 1e-7, f"h={h}: dist_to_redshift round-trip max rel err {max_rel:.2e}"


def test_dist_to_redshift_zero() -> None:
    """Zero distance maps to zero redshift."""
    assert dist_to_redshift(0.0, h=0.73) == 0.0


def test_comoving_volume_element_uses_table_and_stays_positive() -> None:
    """comoving_volume_element (which calls dist_vectorized) stays finite/positive.

    z^2 scaling limit at low z: dV_c/dz/dOmega ~ (c/H0)^3 z^2 (Hogg 1999 Eq. 28).
    """
    z = np.linspace(1e-4, 1.5, 500)
    dvc = np.asarray(comoving_volume_element(z, h=0.73))
    assert np.all(np.isfinite(dvc))
    assert np.all(dvc > 0.0)
    # low-z ratio dVc(2z0)/dVc(z0) -> 4 (z^2 scaling)
    lo = float(np.asarray(comoving_volume_element(np.array([1e-3]), h=0.73))[0])
    hi = float(np.asarray(comoving_volume_element(np.array([2e-3]), h=0.73))[0])
    assert abs(hi / lo - 4.0) < 1e-2
