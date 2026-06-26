"""Shared analytic M1 intrinsic EMRI event-rate model (Babak et al. 2017).

This module provides the *intrinsic, comoving, source-frame* EMRI rate density
used identically by (a) the event-drawing simulation loop and (b) the H0
population prior in the Bayesian inference. It implements the "M1" astrophysical
model of Babak et al. (2017) (PRD 95, 103012, arXiv:1703.09722, Table I):

    M1 = Barausse12 mass function, near-extremal spin a = 0.98, cusp erosion on,
         Gültekin09 M-sigma relation, plunge:EMRI ratio Np = 10, CO mass m = 10 Msun,
         total intrinsic rate (z < 4.5) = 1600 EMRIs / yr.

Frame / volume bookkeeping (the crucial point, spec Item 5):
    Every quantity here is already a *comoving, source-frame* density. The
    cosmological volume element ``dVc/dz`` and the source-to-detector time
    dilation ``1/(1+z)`` are deliberately kept OUT of :func:`R_EMRI`. Callers
    must apply each of them exactly once (see :func:`p_pop_unnormalized`).

Units:
    ``mbh_mass_function``  -> Mpc^-3 dex^-1   (comoving number density per dex)
    ``R0_per_mbh``         -> Gyr^-1          (per-MBH, MBH rest frame)
    ``R_eff_per_mbh``      -> Gyr^-1          (per-MBH, MBH rest frame)
    ``R_EMRI``             -> Mpc^-3 dex^-1 Gyr^-1   (comoving, source-frame)

Modeling choices (explicitly NOT taken verbatim from the paper; see spec):
    1. ``C_NORM`` folds the spin enhancement ``[W(0.98)]^-0.83`` (Eq. 34, ~1.5-2.5)
       and the band-averaged cusp-retention ``<p0>`` (Eq. 21) into a single
       order-unity calibration constant fixed by the Table-I 1600/yr normalization.
    2. :func:`kappa_cap` low-mass roll-off is a surrogate for Eqs. (28)-(30).
    3. :func:`p0_cusp_retention` defaults to 1 (surrogate for Eq. 21), justified
       because the EMRI host population is dominated by M <~ 1e6 Msun where the
       mean number of disrupting mergers N_m is small.

References:
    Babak, Gair, Sesana, Barausse, Sopuerta, Berry, Berti, Amaro-Seoane,
        Petiteau, Klein (2017), "Science with the space-based interferometer
        LISA. V. Extreme mass-ratio inspirals", Phys. Rev. D 95, 103012,
        arXiv:1703.09722. Eqs. (5), (21), (23), (26)-(27), (30)-(31), (34);
        Table I (M1 row).
    Amaro-Seoane & Preto (2011), CQG 28, 094017, arXiv:1106.1429 (origin of R0).
    Barausse (2012), MNRAS 423, 2533, arXiv:1201.5888 (merger history behind p0).
"""

import numpy as np
import numpy.typing as npt

# ── M1 fixed parameters (Babak et al. 2017, Table I) ─────────────────────────
M_PIVOT_MF: float = 3.0e6  # Msun, Eq. (5) mass-function pivot
M_PIVOT_RATE: float = 1.0e6  # Msun, Eq. (23)/(26) rate pivot
NP_M1: int = 10  # plunges:EMRIs ratio, Table I col. 6
M_CO: float = 10.0  # Msun, compact-object mass, Table I col. 7
SPIN_A_M1: float = 0.98  # near-extremal MBH spin "a98", Table I col. 3

# Calibration constant. Folds the (mass-independent) spin enhancement
# [W(0.98)]^-0.83 (Eq. 34) and the band-averaged cusp-retention <p0> (Eq. 21)
# into ONE order-unity number, fixed by requiring the intrinsic integral
# (spec Item 6) to reproduce the Table-I M1 rate of 1600 EMRIs/yr.
#
# Calibration (pipeline cosmology h=0.73, Omega_m=0.25, Omega_de=0.75;
# spec Item 6 integral to z=4.5, log10 M in [4, 7]):
#   N_intrinsic(C_NORM=1) = 6.281029e+02 yr^-1
#   C_NORM = 1600 / 6.281029e+02 = 2.54735
# This lands inside the physically required band [0.3, 3] (it must equal the
# plausible [W(0.98)]^-0.83 * <p0> ~ O(1-2)); a value outside that band would
# signal a units/frame bug. MODELING CHOICE — see module docstring.
C_NORM: float = 2.547353


def mbh_mass_function(M: float | npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    r"""Massive black hole mass function ``dn/dlog10(M)`` (Eq. 5, Barausse12).

    Comoving number density of MBHs per dex of mass, treated as redshift
    independent over the LISA-relevant band (the paper's Fig. 1 fit holds at
    z = 0, 1, 2, 3). It is *already* a comoving density: it contains no volume
    element and no ``(1+z)`` factor.

    .. math::

        \frac{dn}{d\log_{10} M} = 0.005 \left(\frac{M}{3\times10^6\,M_\odot}\right)^{-0.3}
        \quad [\mathrm{Mpc^{-3}\,dex^{-1}}]

    Args:
        M: MBH mass in solar masses (scalar or array). Valid band [1e4, 1e7].

    Returns:
        Comoving MBH number density per dex, in Mpc^-3 dex^-1.

    References:
        Babak et al. (2017), arXiv:1703.09722, Eq. (5).
    """
    M_arr = np.asarray(M, dtype=np.float64)
    # Eq. (5) in Babak et al. (2017), arXiv:1703.09722
    result: npt.NDArray[np.float64] = 0.005 * (M_arr / M_PIVOT_MF) ** (-0.3)
    return result


def R0_per_mbh(M: float | npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    r"""Steady-state EMRI rate per MBH ``R0(M)`` (Eq. 23), source frame.

    The cusp-present, relaxation-driven capture rate of a single compact object
    by one MBH, *before* the duty-cycle, overgrowth-cap, and spin corrections.
    Expressed in the MBH rest frame (source frame); no ``(1+z)`` baked in.

    .. math::

        R_0(M) = 300 \left(\frac{M}{10^6\,M_\odot}\right)^{-0.19}
        \quad [\mathrm{Gyr^{-1}}]

    Args:
        M: MBH mass in solar masses (scalar or array).

    Returns:
        Per-MBH EMRI rate in Gyr^-1 (source frame).

    References:
        Babak et al. (2017), arXiv:1703.09722, Eq. (23).
        Amaro-Seoane & Preto (2011), arXiv:1106.1429 (original derivation).
    """
    M_arr = np.asarray(M, dtype=np.float64)
    # Eq. (23) in Babak et al. (2017), arXiv:1703.09722
    result: npt.NDArray[np.float64] = 300.0 * (M_arr / M_PIVOT_RATE) ** (-0.19)
    return result


def duty_cycle_Gamma(
    M: float | npt.NDArray[np.float64],
    Np: int = NP_M1,
    m: float = M_CO,
) -> npt.NDArray[np.float64]:
    r"""Duty cycle ``Gamma(M) = min(t_d / t_relax, 1)`` (Eqs. 26-27).

    Fraction of time the MBH can supply EMRIs, set by the competition between
    CO-supply depletion (``t_d``) and two-body relaxation (``t_relax``). For the
    M1 parameters (Np = 10, m = 10 Msun) the ratio stays well below 1 across the
    whole mass band, so the ``min`` never selects the cap — Gamma acts as a
    near-constant ~10x suppression with a very shallow ``M^{0.06}`` tilt.

    .. math::

        \frac{t_d}{t_\mathrm{relax}} = \frac{1.2}{1+N_p}
            \left(\frac{m}{10\,M_\odot}\right)^{-1}
            \left(\frac{M}{10^6\,M_\odot}\right)^{0.06},
        \qquad \Gamma = \min\!\left(\frac{t_d}{t_\mathrm{relax}}, 1\right)

    Args:
        M: MBH mass in solar masses (scalar or array).
        Np: Ratio of direct plunges to EMRIs (Table I col. 6).
        m: Compact-object mass in solar masses (Table I col. 7).

    Returns:
        Dimensionless duty cycle in (0, 1].

    References:
        Babak et al. (2017), arXiv:1703.09722, Eqs. (26)-(27).
    """
    M_arr = np.asarray(M, dtype=np.float64)
    # Eq. (26) in Babak et al. (2017), arXiv:1703.09722
    ratio = (1.2 / (1.0 + Np)) * (m / 10.0) ** (-1.0) * (M_arr / M_PIVOT_RATE) ** 0.06
    # Eq. (27) in Babak et al. (2017), arXiv:1703.09722
    result: npt.NDArray[np.float64] = np.minimum(ratio, 1.0)
    return result


def kappa_cap(
    M: float | npt.NDArray[np.float64],
    M_turn: float = 1.0e5,
) -> npt.NDArray[np.float64]:
    r"""Overgrowth-cap factor ``kappa(M)`` (Eq. 30 surrogate).

    The exact cap ``kappa = min(e^{-1} M / Delta M, 1)`` (Eq. 30) requires the
    cusp-present lifetime ``t_EMRI = int (dt/dz) p0`` (Eqs. 28-29) and is ~1
    above ~1e5 Msun, biting only at the low-mass edge. This SURROGATE reproduces
    that behavior with a smooth low-mass roll-off:

    .. math::

        \kappa(M) = \begin{cases} 1 & M \ge M_\mathrm{turn} \\
                     (M/M_\mathrm{turn})^{1/2} & M < M_\mathrm{turn} \end{cases}

    It is monotonically non-decreasing in M and bounded above by 1. MODELING
    CHOICE (surrogate for Eqs. 28-30).

    Args:
        M: MBH mass in solar masses (scalar or array).
        M_turn: Turn-over mass below which the cap suppresses the rate (Msun).

    Returns:
        Dimensionless cap factor in (0, 1].

    References:
        Babak et al. (2017), arXiv:1703.09722, Eq. (30) (surrogate form).
    """
    M_arr = np.asarray(M, dtype=np.float64)
    # Eq. (30) in Babak et al. (2017), arXiv:1703.09722 (surrogate roll-off)
    result: npt.NDArray[np.float64] = np.where(M_arr >= M_turn, 1.0, (M_arr / M_turn) ** 0.5)
    return result


def p0_cusp_retention(
    M: float | npt.NDArray[np.float64],
    z: float | npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    r"""Cusp-retention probability ``p0(M, z) = exp(-N_m(M, z))`` (Eq. 21 surrogate).

    The probability that the MBH still hosts a relaxed stellar cusp (Poisson
    probability of zero disrupting major mergers within the cusp-regrowth time).
    There is NO closed form: ``N_m(M, z)`` comes from the Barausse (2012)
    semi-analytic merger trees. This SURROGATE returns 1 over the whole domain,
    justified because the EMRI host population is dominated by M <~ 1e6 Msun
    where ``N_m`` is small. The signature keeps the ``(M, z)`` hook so a fitted
    ``exp(-N_m)`` can be swapped in for high-mass / high-z accuracy. MODELING
    CHOICE (surrogate for Eq. 21).

    Args:
        M: MBH mass in solar masses (scalar or array).
        z: Redshift (scalar or array). Broadcast against ``M``.

    Returns:
        Dimensionless retention probability in (0, 1], broadcast over (M, z).

    References:
        Babak et al. (2017), arXiv:1703.09722, Eq. (21) (surrogate p0 = 1).
        Barausse (2012), arXiv:1201.5888 (merger history behind N_m).
    """
    # Eq. (21) in Babak et al. (2017), arXiv:1703.09722 (surrogate: p0 = 1)
    result: npt.NDArray[np.float64] = np.ones_like(
        np.asarray(M, dtype=np.float64) * np.asarray(z, dtype=np.float64)
    )
    return result


def R_eff_per_mbh(M: float | npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    r"""Effective per-MBH EMRI rate for M1 ``R_eff(M)`` (Eqs. 31 x 34), source frame.

    Combines the bare rate (Eq. 23), the duty cycle (Eqs. 26-27), the overgrowth
    cap (Eq. 30 surrogate) and — via ``C_NORM`` — the constant spin enhancement
    ``[W(0.98)]^{-0.83}`` (Eq. 34):

    .. math::

        R_\mathrm{eff}(M) = C_\mathrm{NORM}\,\kappa(M)\,\Gamma(M)\,R_0(M)
        \quad [\mathrm{Gyr^{-1}}]

    In the ``kappa ~ 1`` regime this reduces to the shallow power law
    ``R_eff ∝ (M/1e6)^{-0.13}`` (slope = 0.06 - 0.19) noted in the spec.

    Args:
        M: MBH mass in solar masses (scalar or array).

    Returns:
        Effective per-MBH EMRI rate in Gyr^-1 (source frame).

    References:
        Babak et al. (2017), arXiv:1703.09722, Eqs. (31) and (34).
    """
    # Eqs. (31) x (34) in Babak et al. (2017), arXiv:1703.09722
    result: npt.NDArray[np.float64] = C_NORM * kappa_cap(M) * duty_cycle_Gamma(M) * R0_per_mbh(M)
    return result


def R_EMRI(
    z: float | npt.NDArray[np.float64],
    M: float | npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    r"""Intrinsic, comoving, source-frame EMRI rate density for M1.

    .. math::

        R_\mathrm{EMRI}(z, M) = \frac{dn}{d\log_{10} M}(M)\;R_\mathrm{eff}(M)\;p_0(M, z)
        \quad [\mathrm{Mpc^{-3}\,dex^{-1}\,Gyr^{-1}}]

    This is the single shared density used identically for (a) drawing events
    and (b) the H0 population prior. It is purely comoving and source-frame:
    the volume element ``dVc/dz`` and the time dilation ``1/(1+z)`` are NOT
    included here and must be supplied by the caller exactly once each
    (see :func:`p_pop_unnormalized` and spec Item 5). With the default
    ``p0 = 1`` surrogate the density is z-independent.

    Args:
        z: Redshift (scalar or array). Broadcast against ``M`` via ``p0``.
        M: MBH mass in solar masses (scalar or array).

    Returns:
        Comoving source-frame EMRI rate density in Mpc^-3 dex^-1 Gyr^-1.

    References:
        Babak et al. (2017), arXiv:1703.09722, Sec. III.4 (catalog construction).
    """
    # Intrinsic comoving source-frame density (spec Item 5); each cosmological
    # factor (dVc/dz, 1/(1+z)) is deliberately excluded here.
    result: npt.NDArray[np.float64] = (
        mbh_mass_function(M) * R_eff_per_mbh(M) * p0_cusp_retention(M, z)
    )
    return result


def p_pop_unnormalized(
    z: float | npt.NDArray[np.float64],
    M: float | npt.NDArray[np.float64],
    dVc_dz: float | npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    r"""Unnormalized EMRI population prior ``p_pop(z, M)`` for the H0 inference.

    Applies the two cosmological factors that :func:`R_EMRI` deliberately omits,
    each exactly once: the source-to-detector time dilation ``1/(1+z)`` and the
    full-sky comoving volume element ``dVc/dz`` (spec Item 5):

    .. math::

        p_\mathrm{pop}(z, M) \propto R_\mathrm{EMRI}(z, M)\,\frac{1}{1+z}\,\frac{dV_c}{dz}

    The caller normalizes this over the ``(z, M)`` domain.

    Args:
        z: Redshift (scalar or array).
        M: MBH mass in solar masses (scalar or array).
        dVc_dz: Full-sky comoving volume element ``dVc/dz`` in Mpc^3, evaluated
            at the trial cosmology (e.g. ``4*pi*comoving_volume_element(z)``).

    Returns:
        Unnormalized population-prior density (Mpc^3 x Mpc^-3 dex^-1 Gyr^-1
        = dex^-1 Gyr^-1 up to the overall normalization).

    References:
        Babak et al. (2017), arXiv:1703.09722, Sec. III.4 (frame/volume content).
    """
    # 1/(1+z): source-frame -> detector-frame time dilation (applied once).
    # dVc_dz: full-sky comoving volume element (applied once). See spec Item 5.
    result: npt.NDArray[np.float64] = (
        R_EMRI(z, M) / (1.0 + np.asarray(z, dtype=np.float64)) * dVc_dz
    )
    return result
