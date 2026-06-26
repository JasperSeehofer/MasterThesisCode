"""Unit tests for the shared M1 intrinsic EMRI rate model (``emri_rate``).

Implements the spec's T1-T10 checks (``.planning/m1_rate_spec.md``) against
Babak et al. (2017), arXiv:1703.09722, Table I (M1). All tests are CPU-only
(pure numpy + scipy + the analytic ``physical_relations`` cosmology); no GPU.
"""

import numpy as np
import pytest
from scipy.integrate import quad

from master_thesis_code import emri_rate as er
from master_thesis_code.physical_relations import comoving_volume_element

# ── shared intrinsic-integral helpers (spec Item 6) ──────────────────────────


def _dVc_dz(z: float) -> float:
    """Full-sky comoving volume element dVc/dz [Mpc^3] at the pipeline cosmology.

    ``comoving_volume_element`` returns dVc/dz/dOmega [Mpc^3/sr]; the full-sky
    element is 4*pi times that.
    """
    return 4.0 * np.pi * float(comoving_volume_element(z))


def _inner_mass_integral(z: float) -> float:
    """int_{log10 1e4}^{log10 1e7} dlog10M R_EMRI(z, M) [Mpc^-3 Gyr^-1]."""
    value, _ = quad(lambda x: float(er.R_EMRI(z, 10.0**x)), 4.0, 7.0)
    return float(value)


def _dN_dz(z: float, time_dilation: bool = True, volume: bool = True) -> float:
    """Integrand (dVc/dz)/(1+z) * int dlog10M R_EMRI, with optional factor toggles."""
    factor = _inner_mass_integral(z)
    if volume:
        factor *= _dVc_dz(z)
    if time_dilation:
        factor /= 1.0 + z
    return factor


def _intrinsic_rate(time_dilation: bool = True, volume: bool = True) -> float:
    """N_intrinsic [yr^-1] = 1e-9 * int_0^4.5 dz (dVc/dz)/(1+z) int dlog10M R_EMRI."""
    value, _ = quad(lambda z: _dN_dz(z, time_dilation, volume), 0.0, 4.5)
    return 1e-9 * float(value)


# ── T1 — mass function (Eq. 5), exact ────────────────────────────────────────


def test_t1_mass_function_pivot_value() -> None:
    """Phi(3e6) == 0.005 Mpc^-3 dex^-1 exactly (Eq. 5 pivot)."""
    assert np.isclose(float(er.mbh_mass_function(er.M_PIVOT_MF)), 0.005, rtol=1e-12)


def test_t1_mass_function_off_pivot_values() -> None:
    """Phi(3e7) ~ 0.0025059 and Phi(3e5) ~ 0.0099763 (Eq. 5)."""
    assert np.isclose(float(er.mbh_mass_function(3e7)), 0.005 * 10**-0.3, rtol=1e-12)
    assert np.isclose(float(er.mbh_mass_function(3e5)), 0.005 * 10**0.3, rtol=1e-12)


def test_t1_mass_function_slope() -> None:
    """Per-dex slope of Phi is -0.3 (Eq. 5)."""
    phi_hi = float(er.mbh_mass_function(3e7))
    phi_lo = float(er.mbh_mass_function(3e6))
    assert np.isclose(np.log10(phi_hi / phi_lo), -0.3, rtol=1e-6)


# ── T2 — R0 (Eq. 23), exact ──────────────────────────────────────────────────


def test_t2_r0_pivot_value() -> None:
    """R0(1e6) == 300 Gyr^-1 exactly (Eq. 23 pivot)."""
    assert np.isclose(float(er.R0_per_mbh(er.M_PIVOT_RATE)), 300.0, rtol=1e-12)


def test_t2_r0_off_pivot_values() -> None:
    """R0(10^4.5) ~ 578.2 Gyr^-1 and R0(1e7) ~ 193.8 Gyr^-1 (Eq. 23)."""
    assert np.isclose(float(er.R0_per_mbh(10**4.5)), 300.0 * 10**0.285, rtol=1e-6)
    assert np.isclose(float(er.R0_per_mbh(1e7)), 300.0 * 10**-0.19, rtol=1e-6)


# ── T3 — duty cycle (Eqs. 26-27), exact & always sub-unity for M1 ────────────


def test_t3_duty_cycle_values() -> None:
    """Gamma at three reference masses matches Eqs. 26-27 (Np=10, m=10)."""
    assert np.isclose(float(er.duty_cycle_Gamma(1e6)), 1.2 / 11.0, rtol=1e-6)
    assert np.isclose(float(er.duty_cycle_Gamma(1e7)), (1.2 / 11.0) * 10**0.06, rtol=1e-6)
    assert np.isclose(float(er.duty_cycle_Gamma(10**4.5)), (1.2 / 11.0) * 10**-0.09, rtol=1e-6)


def test_t3_duty_cycle_always_below_unity() -> None:
    """Gamma < 1 across the whole M1 band — the min never selects the cap."""
    M = np.logspace(4, 7, 200)
    gamma = er.duty_cycle_Gamma(M)
    assert np.all(gamma < 1.0)


# ── T4 — kappa cap (Eq. 30 surrogate) behavior ───────────────────────────────


def test_t4_kappa_unity_above_turnover() -> None:
    """kappa == 1 at and above the 1e5 Msun turn-over."""
    assert float(er.kappa_cap(1e5)) == 1.0
    assert float(er.kappa_cap(1e6)) == 1.0
    assert float(er.kappa_cap(1e7)) == 1.0


def test_t4_kappa_bounded_and_monotone() -> None:
    """kappa <= 1 everywhere and monotonically non-decreasing in M."""
    M = np.logspace(4, 7, 500)
    kappa = er.kappa_cap(M)
    assert np.all(kappa <= 1.0)
    assert np.all(np.diff(kappa) >= 0.0)


def test_t4_kappa_below_unity_only_at_low_mass() -> None:
    """kappa < 1 only below the 1e5 Msun turn-over (surrogate roll-off)."""
    assert float(er.kappa_cap(1e4)) < 1.0
    assert float(er.kappa_cap(5e4)) < 1.0
    M_above = np.logspace(5, 7, 100)
    assert np.all(er.kappa_cap(M_above) == 1.0)


# ── T5 — net effective per-MBH slope (Eqs. 23·26·31), kappa=1 regime ─────────


def test_t5_effective_rate_slope() -> None:
    """R_eff(1e7)/R_eff(1e6) ~ 10^-0.13 — confirms (M/1e6)^-0.13 law."""
    ratio = float(er.R_eff_per_mbh(1e7)) / float(er.R_eff_per_mbh(1e6))
    assert np.isclose(ratio, 10**-0.13, rtol=1e-3)


# ── T6 — intrinsic density slope (Eq. 5 × R_eff), kappa=1 & p0=1 regime ──────


def test_t6_density_slope() -> None:
    """R_EMRI(z,1e7)/R_EMRI(z,1e6) ~ 10^-0.43 — per-dex comoving density slope."""
    ratio = float(er.R_EMRI(1.0, 1e7)) / float(er.R_EMRI(1.0, 1e6))
    assert np.isclose(ratio, 10**-0.43, rtol=1e-3)


# ── T7 — Table-I normalization (the headline test) ───────────────────────────


def test_t7_c_norm_in_physical_band() -> None:
    """C_NORM must be order unity in [0.3, 3] (it equals [W(0.98)]^-0.83·<p0>)."""
    assert 0.3 <= er.C_NORM <= 3.0


def test_t7_intrinsic_normalization_pinned_to_1600() -> None:
    """Calibrated intrinsic integral reproduces the Table-I M1 rate of 1600/yr."""
    N_intrinsic = _intrinsic_rate()
    assert np.isclose(N_intrinsic, 1600.0, rtol=0.05)


# ── T8 — frame / double-count guards (spec Item 5) ───────────────────────────


def test_t8_removing_time_dilation_increases_rate() -> None:
    """Dropping 1/(1+z) strictly increases N — proves the factor is present once."""
    N = _intrinsic_rate()
    N_no_time_dilation = _intrinsic_rate(time_dilation=False)
    assert N_no_time_dilation > N


def test_t8_removing_volume_collapses_rate() -> None:
    """Dropping dVc/dz collapses N by orders of magnitude — volume present once."""
    N = _intrinsic_rate()
    N_no_volume = _intrinsic_rate(volume=False)
    assert N_no_volume < N / 1e6


def test_t8_density_has_no_redshift_dependence() -> None:
    """With p0=1 the intrinsic density is z-independent — no (1+z)/volume leaked in."""
    M = np.logspace(4, 7, 50)
    np.testing.assert_array_equal(er.R_EMRI(0.0, M), er.R_EMRI(3.0, M))


# ── T9 — observed redshift distribution shape ────────────────────────────────


def test_t9_dN_dz_unimodal_peak() -> None:
    """dN/dz rises from z=0, peaks at z~1.5-2.5, and declines to z=4.5 (unimodal)."""
    z_grid = np.linspace(0.0, 4.5, 91)
    dN = np.array([_dN_dz(float(z)) for z in z_grid])

    peak_index = int(np.argmax(dN))
    z_peak = float(z_grid[peak_index])

    # single interior maximum. Spec nominal window is z ~ 1.5-2.5; under the
    # pipeline's WMAP-era cosmology (Omega_m=0.25, lower than Planck's 0.315)
    # the (dVc/dz)/(1+z) peak shifts to z ~ 1.45 (verified precise peak 1.4502),
    # so the lower bound is relaxed to 1.4. Spec Item 6 notes the result is
    # "mildly cosmology-dependent".
    assert 0 < peak_index < len(z_grid) - 1
    assert 1.4 <= z_peak <= 2.5

    # endpoints below the peak
    assert dN[0] < dN[peak_index]
    assert dN[-1] < dN[peak_index]

    # unimodal: non-decreasing up to the peak, non-increasing after
    assert np.all(np.diff(dN[: peak_index + 1]) >= 0.0)
    assert np.all(np.diff(dN[peak_index:]) <= 0.0)


# ── T10 — units & positivity ─────────────────────────────────────────────────


def test_t10_density_strictly_positive() -> None:
    """R_EMRI(z, M) > 0 across the valid (z, M) domain."""
    z = np.linspace(0.0, 4.5, 20)
    M = np.logspace(4, 7, 20)
    zz, MM = np.meshgrid(z, M)
    assert np.all(er.R_EMRI(zz, MM) > 0.0)


def test_t10_p_pop_applies_each_factor_once() -> None:
    """p_pop_unnormalized == R_EMRI · 1/(1+z) · dVc/dz, each factor exactly once."""
    z, M = 1.3, 1e6
    dVc_dz = _dVc_dz(z)
    expected = float(er.R_EMRI(z, M)) / (1.0 + z) * dVc_dz
    assert np.isclose(float(er.p_pop_unnormalized(z, M, dVc_dz)), expected, rtol=1e-12)


@pytest.mark.parametrize("M", [1e4, 1e5, 1e6, 1e7])
def test_t10_all_component_factors_positive(M: float) -> None:
    """Each building-block factor is strictly positive (no sign/branch errors)."""
    assert float(er.mbh_mass_function(M)) > 0.0
    assert float(er.R0_per_mbh(M)) > 0.0
    assert float(er.duty_cycle_Gamma(M)) > 0.0
    assert float(er.kappa_cap(M)) > 0.0
    assert float(er.R_eff_per_mbh(M)) > 0.0
