"""Cross-term instrument for the EMRI dark-siren pipeline (BUILD ONLY — prereg-gated).

FIXED 2026-08-05 against the adversarial math review
(instrument_math_review_20260805.json, verdict DEFECTIVE, DEFECT-1..5).

Implements the pairwise leading-order Eq. (31) cross-term of
CLAIM_HITCHHIKER_INDEPENDENCE_20260805.md.DRAFT ("The cross-term instrument",
lines 605-691) for each C-4 census pair (i, j) and each h on the requested grid.

TARGET MATHEMATICS (derived against production, verified this session):
Production's per-event catalogue leg is L_cat,i(h) = S_i(h) / D(h) with

    S_i(h) = Sum_{g in ball_i} w_g N_g,i(h)     (raw ball sum,
             bayesian_statistics.py:3900-3906 weights, :4024-4028 weighted_sum)
    D(h)   = the global, EVENT-INDEPENDENT absolute_marginal normalizer
             (self._global_cat_denom_*, :3950-3951 — a per-(h, channel) scalar).

The factorized pair product used by the production sum-of-logs
(posterior_combination.py:324) is therefore

    S_i S_j = Sum_g Sum_g' w_g w_g' N_g,i N_g',j            (over ball_i x ball_j).

The Eq. (31) leading pairwise correction replaces ONLY the SHARED-galaxy
diagonal (g == g', g in S_ij = ball_i INTERSECT ball_j) of that double sum —
each diagonal term carries w_g^2, one factor of w_g from EACH event's galaxy
sum (DEFECT-1 fix) — with the joint single-latent-z integral:

    L_pair^(31)(h) = [ S_i S_j
                       - Sum_{g in S_ij} w_g^2 N_g,i N_g,j
                       + Sum_{g in S_ij} w_g^2 J_g,ij ] / D(h)^2

    Delta_ij(h) = ln L_pair^(31) - ln(L_cat,i L_cat,j)
                = ln[ (S_i S_j - Sum w^2 N N + Sum w^2 J) / (S_i S_j) ]

NORMALIZER CANCELLATION (DEFECT-5 fix): both the corrected joint and the
factorized product carry the same D(h)^-2, so D(h) cancels exactly in the
ratio; likewise any global rescaling of w_g (e.g. C_NORM inside
R_eff_per_mbh) scales numerator and denominator by the same c^2 and cancels.
Delta_ij is therefore the Eq. (31) correction to the factorized joint
CATALOGUE-LEG ln L in true nats, computed entirely from raw sums. Scope note:
the full production per-event likelihood mixes the catalogue leg with the
completion leg via w_G; the mixture-level joint-ln L correction is
    ln[1 + (w_G,i w_G,j L_cat,i L_cat,j / (L_i L_j)) (e^Delta - 1)]
— rows carry w_G_i/j, L_cat_i/j and combined_i/j so the analysis layer can
compose it without re-running anything.

PERFECT-REDSHIFT EXACTNESS (DEFECT-2 fix): as sigma_z -> 0, rho_g -> delta(z -
z_g), so J_g,ij -> L_GW,i(z_g) L_GW,j(z_g) = N_g,i N_g,j per shared galaxy and
Delta_ij -> 0, for any n_shared — matching the paper's statement that
Eq. (15)/(31) is exact when galaxy redshifts are known perfectly. The former
convenience delta ln(sum_wJ) - ln(sum_wN_i sum_wN_j / K) violated this limit
for n_shared >= 2 and has been REMOVED. Known floor (production-faithful, not
an error): the production host kernel is renormalized over the 4-sigma host
window (Z_g = int_{±4 sigma} phi ≈ 0.99993666), and since N_i N_j carries
Z_g^-2 while J carries Z_g^-1, the limit is Delta -> Sum w^2 N_i N_j (Z_g - 1)
/ (S_i S_j), i.e. a floor of magnitude <= |ln Z_g| ≈ 6.334e-5 nats times the
shared-diagonal share (measured in test_crossterm_toy.py).

Ingredients (exactly as in production ``single_host_likelihood``,
bayesian_statistics.py:4658-5175, volume_deconv kernel, absolute_marginal
assembly; the only new objects are the products of the two events' likelihoods
under ONE shared z — and, for 2D, ONE shared latent host mass):

    N_g,i(h)  = fixed_quad( L_i(z;h) * rho_g(z;h), event-window_i, n=50 )
    J_g,ij(h) = fixed_quad( Ljoint_ij(z;h) * rho_g(z;h),
                            event-window_i INTERSECT event-window_j, n=50 )
    rho_g(z;h) = N(z; z_g, sigma_z) * w_pop_eff(z;h) / Z_g
    w_pop_eff(z;h) = comoving_volume_element(z, h)/(1+z) * f_k(z, pix_g, h)
    w_g = R_eff_per_mbh(M_g) / (1 + z_g)

1D channel: L_i = L_GW,i (3D sky+distance Gaussian), Ljoint_ij = L_GW,i L_GW,j.
2D channel (DEFECT-3 fix): the factorized L_i = L_GW,i * mz_i keeps each
event's own independent-M Gaussian mass marginal (production :5080-5094,
Eddington-shifted M_eff per G2d) — because S_i and S_j are the production sums
— but the JOINT integrand marginalizes the shared galaxy's latent true mass
ONCE for the pair:

    Ljoint_ij(z) = L_GW,i(z) L_GW,j(z) * mzjoint_ij(z)
    mzjoint_ij(z) = Integral dM  N(mu_cond,i(z); M a_i(z), sigma2_cond,i)
                                 N(mu_cond,j(z); M a_j(z), sigma2_cond,j)
                                 N(M; M_eff, sigma_M^2),   a_e(z) = (1+z)/M_det,e

evaluated in closed form (product-of-Gaussians; see
``scaled_shared_latent_mass_joint``). At sigma_M -> 0 the closed form reduces
exactly to mz_i(z) * mz_j(z), so the 2D Delta -> 0 iff BOTH sigma_z -> 0 and
sigma_M -> 0 (the same shared-latent argument that mandates the z coupling
mandates the M coupling).

Stratification fields per row ([A2] discipline + M-4 convention): ball-overlap
degree of both events (C-4 census partners), w_G of both events at the matched
grid h, shared-galaxy rate-weight share of each ball
(sum_w_shared / ball-w-total, the M-4 negligibility denominator), shared count
and ball sizes, plus the frozeng stored-N cross-check sums S_i_frozeng /
S_j_frozeng where the per-h frozeng file exists.

HARD RAILS honored by this file:
  * BUILD ONLY. Running on production data requires BOTH an explicit
    ``--output`` path outside any production results directory AND the
    ``--confirm-run`` flag (the pre-registration gate). Without ``--confirm-run``
    the CLI performs a dry-run plan print and exits.
  * The pure math core (everything above the "PRODUCTION LAYER" marker) imports
    ONLY numpy/scipy and is exercised by test_crossterm_toy.py without any
    production data. All ``master_thesis_code`` imports are lazy (inside
    functions) so importing this module needs no repo data.
  * Never instantiate BayesianStatistics (broken ``simulations`` symlink trap);
    the venue CRB CSV is read directly.

Run (when authorized, from the repo root):
    cd /home/jasper/Repositories/MasterThesisCode && uv run python \
        results/campaign51_20260728/realistic_20260729/crossterm_instrument/crossterm_instrument.py \
        --venue iiib --h-grid 0.60,0.73,0.81,0.86 --channel 1d \
        --output results/campaign51_20260728/realistic_20260729/crossterm_instrument/out_iiib_1d.json \
        --confirm-run
"""

import argparse
import getpass
import json
import math
import platform
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.integrate import fixed_quad
from scipy.stats import norm

# ============================================================================
# PURE MATH CORE — no production data, no master_thesis_code imports.
# Exercised by test_crossterm_toy.py.
# ============================================================================

ZFunc = Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]

#: Production quadrature order (bayesian_statistics._HOST_QUAD_N). The core
#: accepts an override so toy-limit tests can resolve narrow kernels; the
#: production layer always passes 50.
QUAD_N_PRODUCTION: int = 50

#: Production host-window half-width in sigmas
#: (single_host_likelihood integration_limit_sigma_multiplier).
HOST_WINDOW_SIGMA: float = 4.0

#: Production lower z floor of the host window (volume_deconv path).
HOST_WINDOW_Z_FLOOR: float = 1e-6

# --- R-2 remedy (prereg §8): per-galaxy adaptive quadrature escalation -------
#
# fixed_quad n=50 degrades when R = integration-window-width / sigma_z exceeds
# ~30 (rr1_boundary_check.json calibration: R<=30 -> <3e-9 nats; R=45 -> 3e-5;
# R=69 -> 3e-3; R=110 -> 6e-2, undiluted single-galaxy scale; R is the sole
# error controller — verified scale invariance). Remedy: keep the production
# n=50 for R <= R_ADAPTIVE_THRESHOLD (regression: bit-identical to the
# validated behavior there) and above it escalate n so the Gauss-Legendre
# node-per-sigma density is ADAPTIVE_NODE_MARGIN times the validated R=30
# density (50/30 nodes/sigma gave <3e-9 nats):
#
#     n(R) = ceil(base_n * (R / 30) * 4),  capped at QUAD_N_ADAPTIVE_CAP.
#
# At the single production instance above R=45 (joint_r1/1D pair (114, 1035),
# catalog_index 1349264, sigma_z=0.0019636, R=68.17; rr1_worst_R_table.json)
# this yields n=455 >= 400, the order measured sufficient at that worst
# geometry. The cap 4000 resolves R up to 600 at full margin — production
# worst is 68.2. The escalation is applied IDENTICALLY to N_g,i, N_g,j and
# J_g,ij of a shared galaxy (one override per SharedGalaxyTerm) and to the
# same galaxy's N_g inside the full-ball sums (BallMember override), so every
# ratio entering Delta is formed from consistently-resolved integrals. The
# host-kernel normalizer Z_g is NOT escalated: its own window is exactly
# 2 * HOST_WINDOW_SIGMA = 8 sigmas wide (R = 8, always inside the validated
# regime) and production faithfulness requires the n=50 normalization.

#: R = window/sigma_z above which n=50 is escalated (validated: <3e-9 nats at
#: R<=30; escalation starts at the edge of the validated region).
R_ADAPTIVE_THRESHOLD: float = 30.0

#: Safety factor on the validated node-per-sigma density (see block comment).
ADAPTIVE_NODE_MARGIN: float = 4.0

#: Hard cap on the escalated Gauss-Legendre order (documented; resolves
#: R <= 600 at full margin — production worst measured R is 68.2).
QUAD_N_ADAPTIVE_CAP: int = 4000


def adaptive_quad_n(
    window_width: float,
    sigma_z: float,
    base_n: int = QUAD_N_PRODUCTION,
) -> int:
    """Per-galaxy Gauss-Legendre order for the R-2 quadrature remedy.

    Args:
        window_width: Width of the integration window the kernel is evaluated
            on (for a pair term: the max of the two event-window widths, which
            bounds the intersection width as well).
        sigma_z: The galaxy's host-kernel width. Non-positive or non-finite
            values return ``base_n`` (sigma_z = 0 rows would NaN-poison the
            kernel itself; verified absent from every ball and shared set —
            rr1_worst_R_table session read).
        base_n: Baseline order (production: 50).

    Returns:
        ``base_n`` when R = window_width/sigma_z <= R_ADAPTIVE_THRESHOLD
        (bit-identical regression guarantee for the validated regime), else
        ``min(QUAD_N_ADAPTIVE_CAP, ceil(base_n * (R/30) * 4))``.
    """
    if not math.isfinite(sigma_z) or sigma_z <= 0.0:
        return base_n
    if not math.isfinite(window_width) or window_width <= 0.0:
        return base_n
    R = window_width / sigma_z
    if R <= R_ADAPTIVE_THRESHOLD:
        return base_n
    n = math.ceil(base_n * (R / R_ADAPTIVE_THRESHOLD) * ADAPTIVE_NODE_MARGIN)
    return min(QUAD_N_ADAPTIVE_CAP, n)


@dataclass
class GalaxyZKernel:
    """Normalized host-redshift kernel rho_g(z) = N(z; z_g, sigma_z) w(z) / Z_g.

    Mirrors production ``galaxy_redshift_prior_pdf`` (bayesian_statistics.py
    :4877-4881) on the volume_deconv path: the Gaussian times the population
    weight, renormalized by ``Z_g`` computed with ``fixed_quad`` over the host
    window ``[max(z_g - 4 sigma_z, 1e-6), z_g + 4 sigma_z]`` (guard: Z_g <= 0
    falls back to 1.0, :4874-4875). ``rho`` itself is NOT truncated to the
    window (production evaluates it on the event window).
    """

    z_g: float
    sigma_z: float
    z_lo: float
    z_hi: float
    z_norm: float
    w_pop_eff: ZFunc | None = None
    _frozen: Any = field(default=None, repr=False)

    def rho(self, z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        base = np.asarray(self._frozen.pdf(z), dtype=np.float64)
        if self.w_pop_eff is not None:
            base = base * np.asarray(self.w_pop_eff(z), dtype=np.float64)
        return base / self.z_norm


def make_galaxy_z_kernel(
    z_g: float,
    sigma_z: float,
    w_pop_eff: ZFunc | None = None,
    quad_n: int = QUAD_N_PRODUCTION,
    sigma_multiplier: float = HOST_WINDOW_SIGMA,
    z_floor: float = HOST_WINDOW_Z_FLOOR,
) -> GalaxyZKernel:
    """Build the normalized host-z kernel exactly as production stages it.

    Args:
        z_g: Catalogue host redshift.
        sigma_z: Effective host redshift error (production: photo-z error;
            SIGMA_V_PEC_KM_S = 0.0 so no PV term — trap 7).
        w_pop_eff: Optional population weight callable (production:
            ``comoving_volume_element(z, h)/(1+z) * f_k``); ``None`` means a
            constant weight (toy limit / non-volume modes).
        quad_n: Gauss-Legendre order for Z_g (production: 50).
        sigma_multiplier: Host-window half-width in sigmas (production: 4.0).
        z_floor: Lower z floor (production: 1e-6).

    Returns:
        A :class:`GalaxyZKernel` with ``z_norm`` already computed.
    """
    z_lo = max(z_g - sigma_multiplier * sigma_z, z_floor)
    z_hi = z_g + sigma_multiplier * sigma_z
    frozen = norm(loc=z_g, scale=sigma_z)

    def _unnorm(z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        base = np.asarray(frozen.pdf(z), dtype=np.float64)
        if w_pop_eff is not None:
            base = base * np.asarray(w_pop_eff(z), dtype=np.float64)
        return base

    if z_lo >= z_hi:
        z_norm = 1.0
    else:
        z_norm = float(fixed_quad(_unnorm, z_lo, z_hi, n=quad_n)[0])
        if z_norm <= 0.0:
            z_norm = 1.0  # production guard, bayesian_statistics.py:4874-4875
    return GalaxyZKernel(
        z_g=z_g,
        sigma_z=sigma_z,
        z_lo=z_lo,
        z_hi=z_hi,
        z_norm=z_norm,
        w_pop_eff=w_pop_eff,
        _frozen=frozen,
    )


def per_galaxy_numerator(
    l_gw: ZFunc,
    rho: ZFunc,
    z_lo: float,
    z_hi: float,
    quad_n: int = QUAD_N_PRODUCTION,
) -> float:
    """Per-galaxy 1-event numerator N_g,i = fixed_quad(L_GW * rho, window, n).

    Guard (trap 9): an empty or inverted window returns 0.0 — ``fixed_quad`` on
    an inverted interval would return a NEGATIVE value, never rely on it.
    """
    if z_lo >= z_hi:
        return 0.0
    return float(fixed_quad(lambda z: l_gw(z) * rho(z), z_lo, z_hi, n=quad_n)[0])


def pair_joint_integral(
    l_gw_i: ZFunc,
    l_gw_j: ZFunc,
    rho: ZFunc,
    window_i: tuple[float, float],
    window_j: tuple[float, float],
    quad_n: int = QUAD_N_PRODUCTION,
    *,
    joint_l: ZFunc | None = None,
) -> float:
    """Per-galaxy paired integral J_g,ij over the event-window intersection.

    J_g,ij = fixed_quad( Ljoint_ij(z) * rho_g(z),
                         [max(zlo_i, zlo_j), min(zhi_i, zhi_j)], n )

    ``Ljoint_ij`` defaults to the product ``l_gw_i * l_gw_j`` (1D channel: the
    two event Gaussians under ONE shared z). For the 2D channel pass
    ``joint_l`` — the shared-latent-mass joint integrand (DEFECT-3 fix), which
    is NOT the product of the two factorized per-event callables. Empty
    intersection returns 0.0 (guard like bayesian_statistics.py:4246; trap 9).
    """
    z_lo = max(window_i[0], window_j[0])
    z_hi = min(window_i[1], window_j[1])
    if z_lo >= z_hi:
        return 0.0
    if joint_l is not None:
        integrand = joint_l
    else:

        def integrand(z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            return l_gw_i(z) * l_gw_j(z)

    return float(fixed_quad(lambda z: integrand(z) * rho(z), z_lo, z_hi, n=quad_n)[0])


def shared_latent_mass_joint(
    mu_i: "float | npt.NDArray[np.float64]",
    mu_j: "float | npt.NDArray[np.float64]",
    s2_i: "float | npt.NDArray[np.float64]",
    s2_j: "float | npt.NDArray[np.float64]",
    M_g: "float | npt.NDArray[np.float64]",
    sM2: "float | npt.NDArray[np.float64]",
) -> "Any":
    """Closed form of the ONE-shared-latent-mass pair marginal (DEFECT-3 fix).

    Integral dM  N(mu_i; M, s2_i) N(mu_j; M, s2_j) N(M; M_g, sM2)
      = N(mu_i; mu_j, s2_i + s2_j) * N(m_ij; M_g, v_ij + sM2)
    with m_ij = (mu_i s2_j + mu_j s2_i)/(s2_i + s2_j),
         v_ij = s2_i s2_j / (s2_i + s2_j)
    (standard product-of-Gaussians identity applied twice; verified by the
    reviewer against adaptive quadrature to 1.3e-15 rel and re-verified by
    test_crossterm_toy.py). At sM2 = 0 this reduces EXACTLY to
    N(mu_i; M_g, s2_i) * N(mu_j; M_g, s2_j) — the product of the independent
    per-event marginals — so the 2D cross-term vanishes with sigma_M.
    """
    v_sum = s2_i + s2_j
    m_ij = (mu_i * s2_j + mu_j * s2_i) / v_sum
    v_ij = s2_i * s2_j / v_sum
    return _norm_pdf(mu_i, mu_j, v_sum) * _norm_pdf(m_ij, M_g, v_ij + sM2)


def scaled_shared_latent_mass_joint(
    mu_i: "float | npt.NDArray[np.float64]",
    mu_j: "float | npt.NDArray[np.float64]",
    a_i: "float | npt.NDArray[np.float64]",
    a_j: "float | npt.NDArray[np.float64]",
    s2c_i: "float | npt.NDArray[np.float64]",
    s2c_j: "float | npt.NDArray[np.float64]",
    M_g: "float | npt.NDArray[np.float64]",
    sM2: "float | npt.NDArray[np.float64]",
) -> "Any":
    """Shared-latent-mass joint with per-event linear scale factors.

    Integral dM  N(mu_i; M a_i, s2c_i) N(mu_j; M a_j, s2c_j) N(M; M_g, sM2)
      = (1/(a_i a_j)) * shared_latent_mass_joint(mu_i/a_i, mu_j/a_j,
                                                 s2c_i/a_i^2, s2c_j/a_j^2,
                                                 M_g, sM2)
    via the change of variable x_e = mu_e / a_e (N(mu; M a, s2) as a density in
    mu equals (1/a) N(mu/a; M, s2/a^2)). This is the production 2D-channel
    parametrization: mu_e = mu_cond,e(z) is the fractional detector-frame mass
    coordinate of event e and a_e = (1+z)/M_det,e (bayesian_statistics.py
    :5085-5094 per-event convention, coupled here through the ONE latent M).
    """
    x_i = mu_i / a_i
    x_j = mu_j / a_j
    return shared_latent_mass_joint(x_i, x_j, s2c_i / a_i**2, s2c_j / a_j**2, M_g, sM2) / (
        a_i * a_j
    )


def _norm_pdf(x: "Any", m: "Any", v: "Any") -> "Any":
    """Vectorized N(x; m, v) with variance argument (pure numpy)."""
    return np.exp(-0.5 * (np.asarray(x) - np.asarray(m)) ** 2 / np.asarray(v)) / np.sqrt(
        2.0 * np.pi * np.asarray(v)
    )


@dataclass
class SharedGalaxyTerm:
    """One shared galaxy g in S_ij with its per-event likelihood callables.

    ``l_gw_i``/``l_gw_j`` are the FACTORIZED per-event callables (1D: the 3D GW
    Gaussian along the z ray; 2D: its product with that event's OWN
    independent-M mass marginal — exactly the objects inside the production
    sums S_i, S_j). ``joint_l``, when set, is the joint integrand
    Ljoint_ij(z) used for J_g,ij instead of ``l_gw_i * l_gw_j``; the 2D channel
    MUST set it to the shared-latent-mass form (DEFECT-3), the 1D channel
    leaves it ``None``. ``quad_n_override``, when set, replaces the caller's
    ``quad_n`` for THIS galaxy's N_g,i, N_g,j AND J_g,ij identically (the R-2
    adaptive escalation; ``None`` — the default — keeps the caller's order,
    preserving pre-R-2 behavior exactly).
    """

    w_g: float
    rho: ZFunc
    l_gw_i: ZFunc
    l_gw_j: ZFunc
    joint_l: ZFunc | None = None
    quad_n_override: int | None = None


@dataclass
class BallMember:
    """One galaxy in a single event's candidate ball (for the raw ball sum).

    ``quad_n_override`` (R-2 remedy): per-galaxy Gauss-Legendre order for this
    member's N_g; ``None`` keeps the caller's ``quad_n`` (pre-R-2 behavior).
    """

    w_g: float
    rho: ZFunc
    l_ev: ZFunc
    quad_n_override: int | None = None


def compute_ball_sum(
    members: Sequence[BallMember],
    window: tuple[float, float],
    quad_n: int = QUAD_N_PRODUCTION,
) -> float:
    """Raw production ball sum S = Sum_{g in ball} w_g N_g over ONE event.

    Replicates bayesian_statistics.py:4024-4028 ``weighted_sum`` over the exact
    candidate list, with N_g the Step-2 quadrature over the event window.
    Members with ``quad_n_override`` set use that order instead of ``quad_n``
    (R-2 adaptive escalation; ``None`` preserves pre-R-2 behavior exactly).
    """
    total = 0.0
    for m in members:
        n_m = quad_n if m.quad_n_override is None else m.quad_n_override
        total += m.w_g * per_galaxy_numerator(m.l_ev, m.rho, window[0], window[1], quad_n=n_m)
    return total


@dataclass
class PairSums:
    """Raw per-pair sums (no normalizer baked in; see delta_joint_lnL_nats).

    ``S_i``/``S_j`` are the FULL-ball raw sums Sum_{g in ball} w_g N_g — the
    production catalogue-leg numerators up to the global D(h) that cancels in
    the Delta ratio. ``shared_diag_fact``/``shared_diag_joint`` are the
    w_g^2-weighted shared-galaxy diagonal sums (DEFECT-1 fix). The w^1 shared
    sums are retained as diagnostics/stratification fields only — they are NOT
    sufficient to form the cross-term for n_shared >= 2 (review check C).
    """

    S_i: float
    S_j: float
    shared_diag_fact: float  # Sum_{g in shared} w_g^2 N_g,i N_g,j
    shared_diag_joint: float  # Sum_{g in shared} w_g^2 J_g,ij
    sum_wJ: float  # Sum_{g in shared} w_g J_g,ij      (diagnostic)
    sum_wN_i: float  # Sum_{g in shared} w_g N_g,i     (diagnostic)
    sum_wN_j: float  # Sum_{g in shared} w_g N_g,j     (diagnostic)
    sum_w: float  # Sum_{g in shared} w_g  (M-4 share-convention numerator)
    n_shared: int


def compute_pair_sums(
    terms: Sequence[SharedGalaxyTerm],
    window_i: tuple[float, float],
    window_j: tuple[float, float],
    quad_n: int = QUAD_N_PRODUCTION,
    *,
    S_i: float | None = None,
    S_j: float | None = None,
) -> PairSums:
    """Accumulate the shared-galaxy diagonal sums + carry the full-ball sums.

    N_g,i integrates over event i's window, N_g,j over event j's window, J_g
    over their intersection (via ``terms[k].joint_l`` when set — 2D channel).
    ``S_i``/``S_j`` are the FULL-ball raw sums, computed separately with
    :func:`compute_ball_sum` in production. When ``None`` (toy convention,
    documented), the ball is taken to BE the shared set: S_i := Sum w_g N_g,i
    over ``terms``. Production ALWAYS passes both explicitly. A term's
    ``quad_n_override`` (R-2 adaptive escalation) replaces ``quad_n`` for that
    galaxy's N_i, N_j AND J identically, keeping the ratios consistent;
    ``None`` preserves pre-R-2 behavior exactly.
    """
    sum_wJ = 0.0
    sum_wN_i = 0.0
    sum_wN_j = 0.0
    sum_w = 0.0
    shared_diag_fact = 0.0
    shared_diag_joint = 0.0
    for t in terms:
        n_t = quad_n if t.quad_n_override is None else t.quad_n_override
        N_i = per_galaxy_numerator(t.l_gw_i, t.rho, window_i[0], window_i[1], quad_n=n_t)
        N_j = per_galaxy_numerator(t.l_gw_j, t.rho, window_j[0], window_j[1], quad_n=n_t)
        J = pair_joint_integral(
            t.l_gw_i,
            t.l_gw_j,
            t.rho,
            window_i,
            window_j,
            quad_n=n_t,
            joint_l=t.joint_l,
        )
        sum_wN_i += t.w_g * N_i
        sum_wN_j += t.w_g * N_j
        sum_wJ += t.w_g * J
        shared_diag_fact += t.w_g**2 * N_i * N_j
        shared_diag_joint += t.w_g**2 * J
        sum_w += t.w_g
    return PairSums(
        S_i=sum_wN_i if S_i is None else S_i,
        S_j=sum_wN_j if S_j is None else S_j,
        shared_diag_fact=shared_diag_fact,
        shared_diag_joint=shared_diag_joint,
        sum_wJ=sum_wJ,
        sum_wN_i=sum_wN_i,
        sum_wN_j=sum_wN_j,
        sum_w=sum_w,
        n_shared=len(terms),
    )


def delta_joint_lnL_nats(sums: PairSums) -> float:
    """Eq. (31) pairwise cross-term as a correction to the joint ln L, in nats.

    Delta_ij = ln[ (S_i S_j - shared_diag_fact + shared_diag_joint)
                   / (S_i S_j) ]
             = log1p( (shared_diag_joint - shared_diag_fact) / (S_i S_j) )

    The global production normalizer D(h) (and any global rescaling of w_g)
    multiplies both the corrected joint and the factorized product by the same
    factor and CANCELS in this ratio — see the module header — so the value is
    the correction to the joint catalogue-leg ln L in true nats. There is no
    caller-chosen K anymore (the former convenience delta was DEFECT-2).

    Returns:
        0.0 exactly when ``n_shared == 0`` (no shared galaxies, no
        correction); ``nan`` when the factorized reference is undefined
        (either full-ball sum <= 0 or non-finite); ``-inf`` when the corrected
        joint is <= 0 against a positive factorized reference (mathematically
        the corrected joint is >= the off-diagonal sum >= 0; a negative value
        can only arise from floating-point cancellation and is clamped to
        -inf).
    """
    if not (math.isfinite(sums.S_i) and math.isfinite(sums.S_j)):
        return float("nan")
    if sums.S_i <= 0.0 or sums.S_j <= 0.0:
        return float("nan")
    denom = sums.S_i * sums.S_j
    if not math.isfinite(denom) or denom <= 0.0:
        return float("nan")
    x = (sums.shared_diag_joint - sums.shared_diag_fact) / denom
    if 1.0 + x <= 0.0:
        return float("-inf")
    return math.log1p(x)


# ============================================================================
# PRODUCTION LAYER — venue wiring. Lazy master_thesis_code imports throughout.
# NEVER RUN without the pre-registration gate (--confirm-run).
# ============================================================================

REPO_ROOT = Path("/home/jasper/Repositories/MasterThesisCode")

#: CRB CSV (md5-identical between the two postfix venue copies,
#: 9a1f2a14384a9281c97ca3be312ddaab; 1590 data rows).
CRB_PATH = (
    REPO_ROOT / "results/run_20260804_postfix/joint_r1/diagnostics/prepared_cramer_rao_bounds.csv"
)

_STAGED = REPO_ROOT / "results/campaign51_20260728/realistic_20260729/realizations_staged"

VENUE_CONFIGS: dict[str, dict[str, Path]] = {
    # Baseline unscattered parent catalogue (cluster parent, sha256 7af3f4f4...).
    "iiib": {
        "catalogue": _STAGED / "cluster_parent_reduced_galaxy_catalogue.csv",
        "frozeng_dir": REPO_ROOT / "results/run_20260804_frozeng/iiib",
        "event_likelihoods": REPO_ROOT
        / "results/run_20260804_postfix/iiib/diagnostics/event_likelihoods.csv",
    },
    # Scattered realistic realization (seed 900001, sigma_scale 1.0).
    "joint_r1": {
        "catalogue": _STAGED / "observed_catalogue_seed900001.csv",
        "frozeng_dir": REPO_ROOT / "results/run_20260804_frozeng/joint_r1",
        "event_likelihoods": REPO_ROOT
        / "results/run_20260804_postfix/joint_r1/diagnostics/event_likelihoods.csv",
    },
}

DEFAULT_H_GRID: tuple[float, ...] = (0.60, 0.73, 0.81, 0.86)

#: Production event filter (bayesian_statistics.py:369 + SNR_THRESHOLD).
SNR_THRESHOLD = 20.0
FRACTIONAL_DL_ERROR_THRESHOLD = 0.10

#: Production Fisher condition-number exclusion threshold (venue pin).
FISHER_COND_THRESHOLD = 1e16


@dataclass
class StagedEvent:
    """Per-event 3D/4D Gaussian staging (bayesian_statistics.py:3157-3294)."""

    event_idx: int
    d_L: float
    d_L_unc: float
    M: float
    M_unc: float
    phi: float
    theta: float
    mean_3d: npt.NDArray[np.float64]
    cov_inv_3d: npt.NDArray[np.float64]
    log_norm_3d: float
    mean_4d: npt.NDArray[np.float64]
    sigma2_cond: float
    proj: npt.NDArray[np.float64]
    excluded: bool
    cond_3d: float
    cond_4d: float


def load_filtered_events(crb_path: Path) -> "Any":
    """Read the CRB CSV and apply the production event filter.

    SNR >= 20 (drops 0 events on this CSV) then relative d_L error < 0.10
    (bayesian_statistics.py:369) -> 1588 events. The ORIGINAL row index is
    preserved as ``event_idx`` (aligns with event_likelihoods.csv and the
    frozeng per-event JSON keys).
    """
    import pandas as pd

    df = pd.read_csv(crb_path)
    rel_err = (
        np.sqrt(df["delta_luminosity_distance_delta_luminosity_distance"])
        / df["luminosity_distance"]
    )
    keep = (df["SNR"] >= SNR_THRESHOLD) & (rel_err < FRACTIONAL_DL_ERROR_THRESHOLD)
    return df[keep]


def stage_event(event_idx: int, row: "Any") -> StagedEvent:
    """Replicate the per-event Gaussian staging verbatim (lines 3157-3294)."""
    from master_thesis_code.datamodels.detection import Detection

    det = Detection(row)
    cov_3d = np.array(
        [
            [
                det.phi_error**2,
                det.theta_phi_covariance,
                det.d_L_phi_covariance / det.d_L,
            ],
            [
                det.theta_phi_covariance,
                det.theta_error**2,
                det.d_L_theta_covariance / det.d_L,
            ],
            [
                det.d_L_phi_covariance / det.d_L,
                det.d_L_theta_covariance / det.d_L,
                det.d_L_uncertainty**2 / det.d_L**2,
            ],
        ]
    )
    cov_4d = np.array(
        [
            [
                det.phi_error**2,
                det.theta_phi_covariance,
                det.d_L_phi_covariance / det.d_L,
                det.M_phi_covariance / det.M,
            ],
            [
                det.theta_phi_covariance,
                det.theta_error**2,
                det.d_L_theta_covariance / det.d_L,
                det.M_theta_covariance / det.M,
            ],
            [
                det.d_L_phi_covariance / det.d_L,
                det.d_L_theta_covariance / det.d_L,
                det.d_L_uncertainty**2 / det.d_L**2,
                det.d_L_M_covariance / det.d_L / det.M,
            ],
            [
                det.M_phi_covariance / det.M,
                det.M_theta_covariance / det.M,
                det.d_L_M_covariance / det.d_L / det.M,
                det.M_uncertainty**2 / det.M**2,
            ],
        ]
    )
    cond_3d = float(np.linalg.cond(cov_3d))
    cond_4d = float(np.linalg.cond(cov_4d))
    excluded = cond_3d > FISHER_COND_THRESHOLD or cond_4d > FISHER_COND_THRESHOLD

    mean_3d = np.array([det.phi, det.theta, 1.0])
    mean_4d = np.array([det.phi, det.theta, 1.0, 1.0])
    cov_inv_3d = np.linalg.pinv(cov_3d)
    sign_3d, logdet_3d = np.linalg.slogdet(cov_3d)
    sign_4d, _logdet_4d = np.linalg.slogdet(cov_4d)
    if sign_3d <= 0 or sign_4d <= 0:
        excluded = True
        log_norm_3d = float("nan")
    else:
        log_norm_3d = -0.5 * (3 * np.log(2 * np.pi) + logdet_3d)

    # Conditional (Bishop PRML Eq. 2.81-2.82), lines 3272-3285.
    cov_cross = cov_4d[3, :3]
    cov_mz = cov_4d[3, 3]
    sigma2_cond = max(float(cov_mz - cov_cross @ cov_inv_3d @ cov_cross), 1e-30)
    proj = cov_cross @ cov_inv_3d

    return StagedEvent(
        event_idx=event_idx,
        d_L=float(det.d_L),
        d_L_unc=float(det.d_L_uncertainty),
        M=float(det.M),
        M_unc=float(det.M_uncertainty),
        phi=float(det.phi),
        theta=float(det.theta),
        mean_3d=mean_3d,
        cov_inv_3d=cov_inv_3d,
        log_norm_3d=log_norm_3d,
        mean_4d=mean_4d,
        sigma2_cond=sigma2_cond,
        proj=np.asarray(proj, dtype=np.float64),
        excluded=excluded,
        cond_3d=cond_3d,
        cond_4d=cond_4d,
    )


def c4_pair_census(crb_df_all: "Any") -> tuple[list[tuple[int, int]], dict[int, int]]:
    """Reproduce the C-4 census on the FULL 1590-row CRB frame.

    Recipe pinned by recon_c4_census.py (verified against the draft this
    session: 1620 sky pairs / 981 events; 279 sky+2sigma-d_L pairs / 385
    events): ball radius r = 2 sqrt(lambda_max(J Sigma J^T)) with
    J = diag(|sin qS|, 1) (chord on the unit sphere); sky overlap iff Euclidean
    chord distance <= r_i + r_j; d_L compatibility iff the 2-sigma intervals
    intersect. Returns the pair list (row-index pairs, == event_idx convention)
    and the per-event overlap degree map (partners among the census pairs).
    """
    n = len(crb_df_all)
    theta = crb_df_all["qS"].to_numpy()
    phi = crb_df_all["phiS"].to_numpy()
    s_phi2 = crb_df_all["delta_phiS_delta_phiS"].to_numpy()
    s_theta2 = crb_df_all["delta_qS_delta_qS"].to_numpy()
    cov = crb_df_all["delta_phiS_delta_qS"].to_numpy()
    dl = crb_df_all["luminosity_distance"].to_numpy()
    s_dl = np.sqrt(crb_df_all["delta_luminosity_distance_delta_luminosity_distance"].to_numpy())
    idx = crb_df_all.index.to_numpy()

    r = np.empty(n)
    for k in range(n):
        sig = np.array([[s_phi2[k], cov[k]], [cov[k], s_theta2[k]]])
        jac = np.diag([abs(np.sin(theta[k])), 1.0])
        lam = float(np.linalg.eigvalsh(jac @ sig @ jac.T).max())
        r[k] = 2.0 * np.sqrt(max(lam, 0.0))

    st = np.sin(theta)
    xyz = np.stack([st * np.cos(phi), st * np.sin(phi), np.cos(theta)], axis=1)
    d = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=2)
    rsum = r[:, None] + r[None, :]
    iu = np.triu_indices(n, k=1)
    sky = d[iu] <= rsum[iu]
    ii, jj = iu[0][sky], iu[1][sky]

    lo = dl - 2.0 * s_dl
    hi = dl + 2.0 * s_dl
    win = (lo[ii] <= hi[jj]) & (lo[jj] <= hi[ii])
    pi, pj = ii[win], jj[win]

    degree: dict[int, int] = {}
    pairs: list[tuple[int, int]] = []
    for a, b in zip(pi, pj):
        ea, eb = int(idx[a]), int(idx[b])
        pairs.append((ea, eb))
        degree[ea] = degree.get(ea, 0) + 1
        degree[eb] = degree.get(eb, 0) + 1
    return pairs, degree


def load_ball_sets(
    frozeng_dir: Path, h_file: str = "h_0_73.json"
) -> tuple[dict[int, set[int]], dict[int, set[int]]]:
    """Zero-compute ball membership from the frozeng per-galaxy JSON (M-4 recipe 1).

    ball_1d(event) = {row[0] for galaxy_likelihoods[event]} UNION
                     {row[0] for additional_galaxies_without_bh_mass[event]}
    ball_2d(event) = galaxy_likelihoods rows only.

    These are the EXACT candidate lists production consumed (p_Di stores them
    verbatim). The ball is h-independent within a run (fixed h_min/h_max search
    bounds); callers should spot-check a second h file for set equality before
    trusting the sets (see --spot-check-balls).
    """
    path = frozeng_dir / "posteriors_with_bh_mass" / h_file
    with open(path) as fh:
        data = json.load(fh)
    gl = data.get("galaxy_likelihoods", {})
    add = data.get("additional_galaxies_without_bh_mass", {})
    ball_1d: dict[int, set[int]] = {}
    ball_2d: dict[int, set[int]] = {}
    keys = set(gl.keys()) | set(add.keys())
    for key in keys:
        if not key.isdigit():  # trap 11: per-event scalar keys alongside dicts
            continue
        ev = int(key)
        with_bh = {int(row[0]) for row in gl.get(key, [])}
        additional = {int(row[0]) for row in add.get(key, [])}
        ball_2d[ev] = with_bh
        ball_1d[ev] = with_bh | additional
    return ball_1d, ball_2d


def frozeng_h_file_name(h: float) -> str:
    """Frozeng per-h JSON name: ``h_{round(h, 3) with . -> _}.json``.

    Verified against the on-disk convention this session: the grid endpoints
    are stored with trailing zeros stripped (h = 0.60 -> ``h_0_6.json``), and
    ``h_0_6.json`` / ``h_0_73.json`` / ``h_0_81.json`` / ``h_0_86.json`` all
    exist in both venues' ``posteriors_with_bh_mass`` directories.
    """
    return f"h_{str(round(h, 3)).replace('.', '_')}.json"


def load_frozeng_per_galaxy_N(
    frozeng_dir: Path, h_file: str
) -> tuple[dict[int, dict[int, float]], dict[int, dict[int, float]]]:
    """Stored per-galaxy Step-2 numerators from the frozeng emits (cross-check).

    Returns ``(N_nobh, N_wbh)`` maps ``event -> {catalog_index -> N}``.
    ``N_nobh`` covers the full 1D ball (galaxy_likelihoods UNION additional;
    single_host_likelihood return index 0, bayesian_statistics.py:5162-5175);
    ``N_wbh`` covers the with-BH list only (return index 2). Used to emit the
    production-exact cross-check sums S_i_frozeng = Sum w_g N_g(stored)
    alongside the instrument's own-quadrature S_i.
    """
    path = frozeng_dir / "posteriors_with_bh_mass" / h_file
    with open(path) as fh:
        data = json.load(fh)
    gl = data.get("galaxy_likelihoods", {})
    add = data.get("additional_galaxies_without_bh_mass", {})
    n_nobh: dict[int, dict[int, float]] = {}
    n_wbh: dict[int, dict[int, float]] = {}
    keys = set(gl.keys()) | set(add.keys())
    for key in keys:
        if not key.isdigit():  # trap 11: per-event scalar keys alongside dicts
            continue
        ev = int(key)
        nobh: dict[int, float] = {}
        wbh: dict[int, float] = {}
        for row in gl.get(key, []):
            nobh[int(row[0])] = float(row[1][0])
            wbh[int(row[0])] = float(row[1][2])
        for row in add.get(key, []):
            nobh[int(row[0])] = float(row[1][0])
        n_nobh[ev] = nobh
        n_wbh[ev] = wbh
    return n_nobh, n_wbh


def build_handler(venue: str) -> "Any":
    """Build the venue GalaxyCatalogueHandler exactly as main.py:154-163 does."""
    from master_thesis_code.constants import (
        M_SOURCE_FRAME_MAX,
        M_SOURCE_FRAME_MIN,
    )
    from master_thesis_code.galaxy_catalogue.handler import GalaxyCatalogueHandler

    cfg = VENUE_CONFIGS[venue]
    return GalaxyCatalogueHandler(
        M_min=M_SOURCE_FRAME_MIN,
        M_max=M_SOURCE_FRAME_MAX,
        z_max=1.5,  # Model1CrossCheck.max_redshift
        observed_catalogue_path=str(cfg["catalogue"]),
    )


def event_window(staged: StagedEvent, h: float) -> tuple[float, float]:
    """Event-level GW z-window [dist_to_redshift(d_L -/+ 4 sigma, h)] (:4768-4773)."""
    from master_thesis_code.physical_relations import dist_to_redshift

    z_lo = dist_to_redshift(staged.d_L - HOST_WINDOW_SIGMA * staged.d_L_unc, h=h)
    z_hi = dist_to_redshift(staged.d_L + HOST_WINDOW_SIGMA * staged.d_L_unc, h=h)
    return float(z_lo), float(z_hi)


def make_l_gw(staged: StagedEvent, phi_g: float, theta_g: float, h: float) -> ZFunc:
    """GW likelihood factor L_GW,i(z;h) at the galaxy's sky position (:4886-4904)."""
    from master_thesis_code.bayesian_inference.bayesian_statistics import _mvn_pdf
    from master_thesis_code.physical_relations import dist_vectorized

    def l_gw(z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        d_L = np.asarray(dist_vectorized(z, h=h), dtype=np.float64)
        frac = d_L / staged.d_L
        phi = np.full_like(z, phi_g)
        theta = np.full_like(z, theta_g)
        return _mvn_pdf(
            np.vstack([phi, theta, frac]).T,
            staged.mean_3d,
            staged.cov_inv_3d,
            staged.log_norm_3d,
        )

    return l_gw


def make_mz_factor(
    staged: StagedEvent,
    phi_g: float,
    theta_g: float,
    M_eff: float,
    sigma_M: float,
    h: float,
) -> ZFunc:
    """2D-channel analytic Gaussian mass-marginal factor mz_integral(z) (:5048-5094).

    host_mass_kernel='auto' + absolute_marginal resolves to 'gaussian';
    ``M_eff`` must already carry the G2d Eddington shift
    (``eddington_shifted_host_mass``, :5030-5034).
    """
    from master_thesis_code.physical_relations import dist_vectorized

    mu_obs_4d = staged.mean_4d
    proj = staged.proj
    sigma2_cond = staged.sigma2_cond

    def mz(z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        d_L = np.asarray(dist_vectorized(z, h=h), dtype=np.float64)
        frac = d_L / staged.d_L
        phi = np.full_like(z, phi_g)
        theta = np.full_like(z, theta_g)
        x_obs = np.vstack([phi, theta, frac]).T  # (N, 3)
        mu_cond = mu_obs_4d[3] + (x_obs - mu_obs_4d[:3]) @ proj  # (N,)
        mu_gal_frac = M_eff * (1.0 + z) / staged.M
        sigma_gal_frac = sigma_M * (1.0 + z) / staged.M
        sigma2_sum = sigma2_cond + sigma_gal_frac**2
        result: npt.NDArray[np.float64] = np.exp(
            -0.5 * (mu_cond - mu_gal_frac) ** 2 / sigma2_sum
        ) / np.sqrt(2.0 * np.pi * sigma2_sum)
        return result

    return mz


def make_mz_joint_factor(
    staged_i: StagedEvent,
    staged_j: StagedEvent,
    phi_g: float,
    theta_g: float,
    M_eff: float,
    sigma_M: float,
    h: float,
) -> ZFunc:
    """2D-channel PAIR mass factor with the shared latent mass marginalized ONCE.

    DEFECT-3 fix: the joint integrand for a shared host must carry

        mzjoint_ij(z) = Integral dM  N(mu_cond,i(z); M a_i(z), sigma2_cond,i)
                                     N(mu_cond,j(z); M a_j(z), sigma2_cond,j)
                                     N(M; M_eff, sigma_M^2)

    with a_e(z) = (1+z)/M_det,e — NOT the product mz_i(z) * mz_j(z) of the two
    events' independent-M marginals (that treats the one galaxy's true mass as
    two independent latents). Evaluated in closed form via
    :func:`scaled_shared_latent_mass_joint`. Each mu_cond,e(z) is the
    production Bishop-conditional mean given (sky, d_L_frac) at the SHARED z
    (bayesian_statistics.py:5080-5094); ``M_eff`` must already carry the G2d
    Eddington shift (:5030-5034). At sigma_M -> 0 this reduces exactly to
    mz_i * mz_j (see the closed-form docstring), so the factorized/joint
    distinction disappears with the mass scatter, as it must.
    """
    from master_thesis_code.physical_relations import dist_vectorized

    def mu_cond_of(staged: StagedEvent, z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        d_L = np.asarray(dist_vectorized(z, h=h), dtype=np.float64)
        frac = d_L / staged.d_L
        phi = np.full_like(z, phi_g)
        theta = np.full_like(z, theta_g)
        x_obs = np.vstack([phi, theta, frac]).T  # (N, 3)
        result: npt.NDArray[np.float64] = (
            staged.mean_4d[3] + (x_obs - staged.mean_4d[:3]) @ staged.proj
        )
        return result

    def mzj(z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        mu_i = mu_cond_of(staged_i, z)
        mu_j = mu_cond_of(staged_j, z)
        a_i = (1.0 + z) / staged_i.M
        a_j = (1.0 + z) / staged_j.M
        result: npt.NDArray[np.float64] = np.asarray(
            scaled_shared_latent_mass_joint(
                mu_i,
                mu_j,
                a_i,
                a_j,
                staged_i.sigma2_cond,
                staged_j.sigma2_cond,
                M_eff,
                sigma_M**2,
            ),
            dtype=np.float64,
        )
        return result

    return mzj


def make_w_pop_eff(
    completeness: Any,
    phi_g: float,
    theta_g: float,
    z_g: float,
    sigma_z: float,
    h: float,
) -> ZFunc:
    """Population weight w_pop_eff(z) = dVc/dz/(1+z) * f_k with the ZoA fallback.

    Replicates :4830-4856: the host pixel's f_k enters (clipped to [0, 1])
    UNLESS f_k == 0 at ALL 50 GL nodes of the host window, in which case the
    f_k factor is dropped entirely (never clamped elementwise).
    """
    from master_thesis_code.physical_relations import comoving_volume_element

    host_pixel: int | None = int(completeness.ang2pix(phi_g, theta_g))
    z_lo = max(z_g - HOST_WINDOW_SIGMA * sigma_z, HOST_WINDOW_Z_FLOOR)
    z_hi = z_g + HOST_WINDOW_SIGMA * sigma_z
    nodes, _ = np.polynomial.legendre.leggauss(QUAD_N_PRODUCTION)
    z_nodes = 0.5 * (z_hi - z_lo) * nodes + 0.5 * (z_hi + z_lo)
    f_probe = np.clip(
        np.asarray(completeness.f_k(z_nodes, host_pixel, h), dtype=np.float64), 0.0, 1.0
    )
    if not bool(np.any(f_probe > 0.0)):
        host_pixel = None  # ZoA fallback

    def w_pop_eff(z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        w_pop = np.asarray(comoving_volume_element(z, h=h), dtype=np.float64) / (1.0 + z)
        if host_pixel is not None:
            f_k = np.clip(
                np.asarray(completeness.f_k(z, host_pixel, h), dtype=np.float64),
                0.0,
                1.0,
            )
            return w_pop * f_k
        return w_pop

    return w_pop_eff


@dataclass
class GalaxyRecord:
    """Dereferenced catalogue row for one candidate galaxy."""

    catalog_index: int
    phi_g: float
    theta_g: float
    z_g: float
    sigma_z: float
    M_g: float
    sigma_M: float
    w_g: float


def dereference_galaxy(handler: Any, catalog_index: int) -> GalaxyRecord:
    """Read one candidate's scalars from the venue handler frame + rate weight."""
    from master_thesis_code.emri_rate import R_eff_per_mbh
    from master_thesis_code.galaxy_catalogue.handler import InternalCatalogColumns

    row = handler.reduced_galaxy_catalog.iloc[catalog_index]
    z_g = float(row[InternalCatalogColumns.REDSHIFT])
    M_g = float(row[InternalCatalogColumns.BH_MASS])
    return GalaxyRecord(
        catalog_index=catalog_index,
        phi_g=float(row[InternalCatalogColumns.PHI_S]),
        theta_g=float(row[InternalCatalogColumns.THETA_S]),
        z_g=z_g,
        sigma_z=float(row[InternalCatalogColumns.REDSHIFT_ERROR]),
        M_g=M_g,
        sigma_M=float(row[InternalCatalogColumns.BH_MASS_ERROR]),
        # w_g = R_eff_per_mbh(M_g) / (1 + z_g)  == bs._rate_weight(host)
        w_g=float(R_eff_per_mbh(M_g)) / (1.0 + z_g),
    )


def load_event_scalars(event_likelihoods_path: Path) -> "Any":
    """Load per-(event, h) Path-A scalars (w_G, L_cat_*, combined_*) for stratification."""
    import pandas as pd

    return pd.read_csv(event_likelihoods_path)


def match_grid_h(h_requested: float, h_grid_values: npt.NDArray[np.float64]) -> float | None:
    """Match a requested h to the diagnostics h grid with np.isclose (float trap)."""
    diffs = np.abs(h_grid_values - h_requested)
    k = int(np.argmin(diffs))
    if diffs[k] < 1e-3:
        return float(h_grid_values[k])
    return None


def load_pair_list(path: Path, venue: str) -> dict[str, Any]:
    """Load an explicit target pair set (R-3 ingestion; make_target_pairs.py schema).

    Schema: ``{"meta": {"venue": ..., "counts": ...}, "pairs": {"1d": [{"i",
    "j", "n_shared", "in_c4"}, ...], "2d": [...]}}``. The file's venue must
    match ``--venue`` (cross-venue application refused). Only the pair
    ENUMERATION comes from the file — every shared set, ball sum and Delta is
    still computed from the frozeng ball emits and the venue catalogue (the
    math core is untouched by R-3).
    """
    with open(path) as fh:
        data = json.load(fh)
    file_venue = data.get("meta", {}).get("venue")
    if file_venue != venue:
        raise SystemExit(
            f"REFUSED: pair-list venue {file_venue!r} does not match --venue {venue!r}"
        )
    if not isinstance(data.get("pairs"), dict):
        raise SystemExit(f"REFUSED: pair-list {path} has no 'pairs' mapping")
    for ch, plist in data["pairs"].items():
        for p in plist:
            if not (isinstance(p, dict) and "i" in p and "j" in p):
                raise SystemExit(
                    f"REFUSED: pair-list {path} channel {ch!r} has a row without 'i'/'j': {p!r}"
                )
    return data


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return out.stdout.strip()
    except Exception:  # noqa: BLE001 — provenance best-effort only
        return "unknown"


def _guard_output_path(output: Path) -> None:
    """Refuse output paths that could collide with production/paper JSONs."""
    resolved = str(output.resolve())
    if output.suffix != ".json":
        raise SystemExit(f"REFUSED: output must be a .json path, got {output}")
    if output.exists():
        raise SystemExit(f"REFUSED: output already exists: {output}")
    forbidden_markers = ["/results/run_", "/diagnostics/", "/posteriors"]
    for marker in forbidden_markers:
        if marker in resolved:
            raise SystemExit(
                f"REFUSED: output path {resolved} matches production marker {marker!r}; "
                "write into crossterm_instrument/ instead"
            )


def run_instrument(args: argparse.Namespace) -> None:
    """Full instrument run (PRODUCTION DATA — gated by --confirm-run)."""
    from master_thesis_code.bayesian_inference.bayesian_statistics import (
        eddington_shifted_host_mass,
    )
    from master_thesis_code.galaxy_catalogue.pixel_completeness import from_cache_or_build

    venue = args.venue
    cfg = VENUE_CONFIGS[venue]
    h_grid = [float(x) for x in args.h_grid.split(",")]
    output = Path(args.output)
    _guard_output_path(output)

    import pandas as pd

    t0 = time.time()
    crb_all = pd.read_csv(args.crb_path)
    crb_filtered = load_filtered_events(Path(args.crb_path))
    pairs_all, degree = c4_pair_census(crb_all)
    c4_set = set(pairs_all)
    filtered_idx = set(int(i) for i in crb_filtered.index)
    channels = ["1d", "2d"] if args.channel == "both" else [args.channel]

    pair_list_meta: dict[str, Any] | None = None
    if args.pair_list:
        # R-3 ingestion: explicit target pair set (the M-4 truly-sharing
        # census), one list per channel. The C-4 census above is still
        # computed — it feeds the overlap-degree stratification fields and
        # the in_c4 row flag, never the enumeration.
        pair_list_data = load_pair_list(Path(args.pair_list), venue)
        pair_list_meta = dict(pair_list_data.get("meta", {}))
        raw_by_channel = {
            ch: [(int(p["i"]), int(p["j"])) for p in pair_list_data["pairs"][ch]] for ch in channels
        }
    else:
        raw_by_channel = {ch: list(pairs_all) for ch in channels}

    pairs_by_channel: dict[str, list[tuple[int, int]]] = {}
    for ch in channels:
        chp = [(i, j) for (i, j) in raw_by_channel[ch] if i in filtered_idx and j in filtered_idx]
        if args.pairs_limit is not None:
            chp = chp[: args.pairs_limit]
        pairs_by_channel[ch] = chp
    # Union enumeration, (i, j)-major / channel-minor row order (matches the
    # pre-R-3 census behavior, where both channels shared one pair list).
    pairs = sorted({p for chp in pairs_by_channel.values() for p in chp})

    ball_1d, ball_2d = load_ball_sets(cfg["frozeng_dir"], h_file=args.ball_h_file)
    if args.spot_check_balls:
        b1_alt, b2_alt = load_ball_sets(cfg["frozeng_dir"], h_file=args.spot_check_balls)
        mism = [
            ev
            for ev in ball_1d
            if ball_1d.get(ev) != b1_alt.get(ev) or ball_2d.get(ev) != b2_alt.get(ev)
        ]
        if mism:
            raise SystemExit(
                f"BALL SPOT-CHECK FAILED: {len(mism)} events differ between "
                f"{args.ball_h_file} and {args.spot_check_balls}"
            )

    balls_by_channel = {"1d": ball_1d, "2d": ball_2d}

    # Events actually needed.
    needed_events = sorted({e for p in pairs for e in p})
    staged: dict[int, StagedEvent] = {}
    for ev in needed_events:
        se = stage_event(ev, crb_all.loc[ev])
        if not se.excluded:
            staged[ev] = se
    pairs = [(i, j) for (i, j) in pairs if i in staged and j in staged]
    pair_sets_by_channel: dict[str, set[tuple[int, int]]] = {
        ch: {p for p in pairs_by_channel[ch] if p[0] in staged and p[1] in staged}
        for ch in channels
    }

    handler = build_handler(venue)
    completeness = from_cache_or_build()
    scalars = load_event_scalars(cfg["event_likelihoods"])
    h_grid_values = np.sort(scalars["h"].unique())
    scalars_by_key: dict[tuple[int, float], Any] = {
        (int(r.event_idx), float(r.h)): r for r in scalars.itertuples(index=False)
    }

    galaxy_cache: dict[int, GalaxyRecord] = {}

    def galaxy(ci: int) -> GalaxyRecord:
        if ci not in galaxy_cache:
            galaxy_cache[ci] = dereference_galaxy(handler, ci)
        return galaxy_cache[ci]

    # Per-event ball rate-weight totals per channel (for the shared-share strat).
    ball_w_totals: dict[tuple[str, int], float] = {}
    for ch in channels:
        for ev in needed_events:
            if ev not in staged:
                continue
            ball = balls_by_channel[ch].get(ev, set())
            ball_w_totals[(ch, ev)] = float(sum(galaxy(ci).w_g for ci in ball))

    rows: list[dict[str, Any]] = []
    kernel_cache: dict[tuple[int, float], GalaxyZKernel] = {}

    def kernel(ci: int, h: float) -> GalaxyZKernel:
        key = (ci, h)
        if key not in kernel_cache:
            g = galaxy(ci)
            w_pop = make_w_pop_eff(completeness, g.phi_g, g.theta_g, g.z_g, g.sigma_z, h)
            kernel_cache[key] = make_galaxy_z_kernel(g.z_g, g.sigma_z, w_pop_eff=w_pop)
        return kernel_cache[key]

    def event_galaxy_l(ch: str, ev: int, ci: int, h: float) -> ZFunc:
        """FACTORIZED per-event callable for one ball galaxy (production Step 2)."""
        g = galaxy(ci)
        l_ev: ZFunc = make_l_gw(staged[ev], g.phi_g, g.theta_g, h)
        if ch == "2d":
            M_eff = eddington_shifted_host_mass(g.M_g, g.sigma_M)
            l_ev = _product(
                l_ev, make_mz_factor(staged[ev], g.phi_g, g.theta_g, M_eff, g.sigma_M, h)
            )
        return l_ev

    # Full-ball raw sums S(ch, ev, h) = Sum_{g in ball} w_g N_g — the production
    # catalogue-leg numerators up to the global D(h) (which cancels in Delta).
    ball_sum_cache: dict[tuple[str, int, float], float] = {}

    def ball_sum(ch: str, ev: int, h: float, window: tuple[float, float]) -> float:
        key = (ch, ev, h)
        if key not in ball_sum_cache:
            width = window[1] - window[0]
            members = [
                BallMember(
                    w_g=galaxy(ci).w_g,
                    rho=kernel(ci, h).rho,
                    l_ev=event_galaxy_l(ch, ev, ci, h),
                    # R-2 remedy: escalate narrow kernels (R > 30) inside the
                    # ball sum with the SAME rule as the shared terms; R <= 30
                    # members keep the production n=50 bit-identically.
                    quad_n_override=adaptive_quad_n(width, galaxy(ci).sigma_z),
                )
                for ci in sorted(balls_by_channel[ch].get(ev, set()))
            ]
            ball_sum_cache[key] = compute_ball_sum(members, window)
        return ball_sum_cache[key]

    frozeng_missing: list[str] = []

    for h in h_grid:
        h_matched = match_grid_h(h, np.asarray(h_grid_values, dtype=np.float64))
        windows: dict[int, tuple[float, float]] = {ev: event_window(staged[ev], h) for ev in staged}
        # Production-exact stored-N cross-check sums (frozeng emits), when the
        # per-h file exists; None otherwise.
        s_frozeng: dict[tuple[str, int], float | None] = {}
        fro_file = frozeng_h_file_name(h)
        fro_path = cfg["frozeng_dir"] / "posteriors_with_bh_mass" / fro_file
        if fro_path.exists():
            n_nobh_map, n_wbh_map = load_frozeng_per_galaxy_N(cfg["frozeng_dir"], fro_file)
            for ch in channels:
                stored = n_nobh_map if ch == "1d" else n_wbh_map
                for ev in staged:
                    ev_map = stored.get(ev, {})
                    ball = balls_by_channel[ch].get(ev, set())
                    if ball and all(ci in ev_map for ci in ball):
                        s_frozeng[(ch, ev)] = float(sum(galaxy(ci).w_g * ev_map[ci] for ci in ball))
                    else:
                        s_frozeng[(ch, ev)] = None
            del n_nobh_map, n_wbh_map
        else:
            frozeng_missing.append(fro_file)
            for ch in channels:
                for ev in staged:
                    s_frozeng[(ch, ev)] = None
        for i, j in pairs:
            for ch in channels:
                if (i, j) not in pair_sets_by_channel[ch]:
                    continue
                shared = sorted(
                    balls_by_channel[ch].get(i, set()) & balls_by_channel[ch].get(j, set())
                )
                # R-2 remedy: one adaptive order per shared galaxy, from the
                # max event-window width (bounds the intersection width too),
                # applied identically to N_g,i, N_g,j and J_g,ij.
                width_max = max(windows[i][1] - windows[i][0], windows[j][1] - windows[j][0])
                quad_n_max_shared = QUAD_N_PRODUCTION
                n_escalated_shared = 0
                terms: list[SharedGalaxyTerm] = []
                sum_wZ_shared = 0.0
                for ci in shared:
                    g = galaxy(ci)
                    kern = kernel(ci, h)
                    sum_wZ_shared += g.w_g * kern.z_norm
                    n_g = adaptive_quad_n(width_max, g.sigma_z)
                    if n_g > QUAD_N_PRODUCTION:
                        n_escalated_shared += 1
                    quad_n_max_shared = max(quad_n_max_shared, n_g)
                    l_i: ZFunc = make_l_gw(staged[i], g.phi_g, g.theta_g, h)
                    l_j: ZFunc = make_l_gw(staged[j], g.phi_g, g.theta_g, h)
                    joint_l: ZFunc | None = None
                    if ch == "2d":
                        M_eff = eddington_shifted_host_mass(g.M_g, g.sigma_M)
                        mz_i = make_mz_factor(staged[i], g.phi_g, g.theta_g, M_eff, g.sigma_M, h)
                        mz_j = make_mz_factor(staged[j], g.phi_g, g.theta_g, M_eff, g.sigma_M, h)
                        # DEFECT-3 fix: the joint integrand marginalizes the
                        # shared galaxy's latent mass ONCE for the pair; the
                        # factorized callables keep the per-event
                        # independent-M marginals (they live inside S_i, S_j).
                        mz_joint = make_mz_joint_factor(
                            staged[i], staged[j], g.phi_g, g.theta_g, M_eff, g.sigma_M, h
                        )
                        joint_l = _product(_product(l_i, l_j), mz_joint)
                        l_i = _product(l_i, mz_i)
                        l_j = _product(l_j, mz_j)
                    terms.append(
                        SharedGalaxyTerm(
                            w_g=g.w_g,
                            rho=kern.rho,
                            l_gw_i=l_i,
                            l_gw_j=l_j,
                            joint_l=joint_l,
                            quad_n_override=n_g,
                        )
                    )
                S_i = ball_sum(ch, i, h, windows[i])
                S_j = ball_sum(ch, j, h, windows[j])
                sums = compute_pair_sums(terms, windows[i], windows[j], S_i=S_i, S_j=S_j)
                si = scalars_by_key.get((i, h_matched)) if h_matched is not None else None
                sj = scalars_by_key.get((j, h_matched)) if h_matched is not None else None
                w_tot_i = ball_w_totals.get((ch, i), 0.0)
                w_tot_j = ball_w_totals.get((ch, j), 0.0)
                rows.append(
                    {
                        "event_i": i,
                        "event_j": j,
                        "h_requested": h,
                        "h_grid_matched": h_matched,
                        "channel": ch,
                        "in_c4": (i, j) in c4_set,
                        # --- R-2 adaptive-quadrature audit fields ---
                        "quad_n_max_shared": quad_n_max_shared,
                        "n_escalated_shared": n_escalated_shared,
                        # --- the Eq.(31) cross-term and its raw ingredients ---
                        "S_i_raw": sums.S_i,
                        "S_j_raw": sums.S_j,
                        "S_i_frozeng": s_frozeng.get((ch, i)),
                        "S_j_frozeng": s_frozeng.get((ch, j)),
                        "shared_diag_fact": sums.shared_diag_fact,
                        "shared_diag_joint": sums.shared_diag_joint,
                        "delta_joint_lnL_nats": delta_joint_lnL_nats(sums),
                        # --- shared-set diagnostics (w^1 sums; NOT sufficient
                        # to form Delta for n_shared >= 2 — review check C) ---
                        "sum_wJ_shared": sums.sum_wJ,
                        "sum_wN_i_shared": sums.sum_wN_i,
                        "sum_wN_j_shared": sums.sum_wN_j,
                        "sum_w_shared": sums.sum_w,
                        "sum_wZ_shared": sum_wZ_shared,
                        "n_shared": sums.n_shared,
                        "n_ball_i": len(balls_by_channel[ch].get(i, set())),
                        "n_ball_j": len(balls_by_channel[ch].get(j, set())),
                        # --- stratification ([A2] + M-4 convention) ---
                        "w_share_ball_i": (sums.sum_w / w_tot_i) if w_tot_i > 0 else None,
                        "w_share_ball_j": (sums.sum_w / w_tot_j) if w_tot_j > 0 else None,
                        "overlap_degree_i": degree.get(i, 0),
                        "overlap_degree_j": degree.get(j, 0),
                        "w_G_i": float(si.w_G) if si is not None else None,
                        "w_G_j": float(sj.w_G) if sj is not None else None,
                        "L_cat_i": (
                            float(si.L_cat_no_bh if ch == "1d" else si.L_cat_with_bh)
                            if si is not None
                            else None
                        ),
                        "L_cat_j": (
                            float(sj.L_cat_no_bh if ch == "1d" else sj.L_cat_with_bh)
                            if sj is not None
                            else None
                        ),
                        "combined_i": (
                            float(si.combined_no_bh if ch == "1d" else si.combined_with_bh)
                            if si is not None
                            else None
                        ),
                        "combined_j": (
                            float(sj.combined_no_bh if ch == "1d" else sj.combined_with_bh)
                            if sj is not None
                            else None
                        ),
                    }
                )

    meta = {
        "instrument": "crossterm_instrument",
        "spec": "CLAIM_HITCHHIKER_INDEPENDENCE_20260805.md.DRAFT lines 605-691",
        "venue": venue,
        "channel": args.channel,
        "h_grid": h_grid,
        "quad_n": QUAD_N_PRODUCTION,
        "crb_path": str(args.crb_path),
        "catalogue_path": str(cfg["catalogue"]),
        "frozeng_ball_source": str(
            cfg["frozeng_dir"] / "posteriors_with_bh_mass" / args.ball_h_file
        ),
        "event_likelihoods_path": str(cfg["event_likelihoods"]),
        "n_pairs_census": len(pairs_all),
        "n_pairs_evaluated": len(pairs),
        "pair_source": (
            str(Path(args.pair_list).resolve()) if args.pair_list else "c4_pair_census"
        ),
        "pair_list_meta": pair_list_meta,
        "n_pairs_by_channel": {ch: len(pair_sets_by_channel[ch]) for ch in channels},
        "n_events_staged": len(staged),
        "quad_n_adaptive": {
            "R_threshold": R_ADAPTIVE_THRESHOLD,
            "node_margin": ADAPTIVE_NODE_MARGIN,
            "cap": QUAD_N_ADAPTIVE_CAP,
            "rule": (
                "R-2 remedy: per-galaxy n = base 50 for R = window/sigma_z <= "
                "30 (bit-identical to production quadrature there); above, "
                "n = ceil(50 * (R/30) * 4) capped at 4000, applied "
                "identically to N_g,i, N_g,j, J_g,ij (one override per "
                "shared term, from the max event-window width) and to the "
                "same rule per member inside the full-ball sums; the host "
                "kernel normalizer Z_g stays at production n=50 (its window "
                "is a fixed 8 sigma, R = 8)."
            ),
        },
        "git_commit": _git_commit(),
        "user": getpass.getuser(),
        "hostname": platform.node(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "runtime_s": round(time.time() - t0, 1),
        "frozeng_files_missing": frozeng_missing,
        "formula": (
            "delta_joint_lnL_nats = ln[(S_i S_j - Sum_{g in shared} w_g^2 "
            "N_g,i N_g,j + Sum_{g in shared} w_g^2 J_g,ij) / (S_i S_j)] with "
            "S_e = Sum_{g in ball_e} w_g N_g,e the raw production ball sums "
            "(bayesian_statistics.py:3900-3906, :4024-4028). Eq.(31) leading "
            "pairwise correction: the shared-galaxy diagonal of the factorized "
            "pair product (each term carries w_g^2, one factor per event) is "
            "replaced by the joint one-shared-latent-z integral J_g,ij; 2D "
            "additionally marginalizes ONE shared latent host mass inside "
            "J_g,ij (closed form) while the factorized diagonal keeps the "
            "per-event independent-M marginals."
        ),
        "normalizer_cancellation": (
            "Production L_cat,e = S_e / D(h) with D(h) the global "
            "event-independent absolute_marginal denominator "
            "(bayesian_statistics.py:3950-3951). Corrected joint and "
            "factorized product both carry D(h)^-2, so D(h) cancels exactly "
            "in the Delta ratio (as does any global rescaling of w_g, e.g. "
            "C_NORM); delta_joint_lnL_nats is therefore the Eq.(31) "
            "correction to the joint CATALOGUE-LEG ln L in true nats. "
            "Mixture-level composition (completion leg, w_G): "
            "ln[1 + (w_G_i w_G_j L_cat_i L_cat_j / (combined_i combined_j)) "
            "* (exp(Delta) - 1)] from the emitted row fields."
        ),
        "s_source_note": (
            "S_i_raw/S_j_raw are own-quadrature full-ball sums "
            "(self-consistent with shared_diag_*); S_i_frozeng/S_j_frozeng "
            "are the production-exact stored-N sums from the frozeng "
            "per-galaxy emits (cross-check; None where the per-h file or a "
            "ball member's stored row is missing)."
        ),
        "defect_fixes": [
            "DEFECT-1: shared diagonal carries w_g^2 (one factor per event)",
            "DEFECT-2: sigma_z->0 gives J->N_i N_j so Delta->0 exactly; "
            "convenience delta (caller K) removed",
            "DEFECT-3: 2D joint term marginalizes ONE shared latent mass "
            "(closed form), factorized diagonal keeps per-event marginals",
            "DEFECT-5: Delta formed from full-ball raw sums; global "
            "normalizer cancels (see normalizer_cancellation)",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as fh:
        json.dump({"meta": meta, "rows": rows}, fh, indent=1)
    print(f"wrote {len(rows)} rows -> {output}")


def _product(f: ZFunc, g: ZFunc) -> ZFunc:
    """Pointwise product of two vectorized z-callables."""

    def prod(z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return f(z) * g(z)

    return prod


def run_validate(args: argparse.Namespace) -> None:
    """Cross-check Step-2 N_g,i against frozeng per-galaxy ground truth.

    Recomputes N_nobh for a handful of (event, galaxy) rows from
    posteriors_with_bh_mass/<h file> and prints relative deviations. Gated by
    --confirm-run like the instrument (reads production data).
    """
    from master_thesis_code.galaxy_catalogue.pixel_completeness import from_cache_or_build

    cfg = VENUE_CONFIGS[args.venue]
    h = float(args.h_grid.split(",")[0])
    h_tag = f"h_{str(round(h, 3)).replace('.', '_')}.json"
    path = cfg["frozeng_dir"] / "posteriors_with_bh_mass" / h_tag
    with open(path) as fh:
        data = json.load(fh)
    gl = data["galaxy_likelihoods"]

    crb_all = __import__("pandas").read_csv(args.crb_path)
    handler = build_handler(args.venue)
    completeness = from_cache_or_build()

    n_checked = 0
    for key in sorted(gl.keys(), key=lambda k: int(k) if k.isdigit() else 10**9):
        if not key.isdigit() or n_checked >= args.pairs_limit:
            break
        ev = int(key)
        se = stage_event(ev, crb_all.loc[ev])
        if se.excluded:
            continue
        win = event_window(se, h)
        for row in gl[key][:3]:
            ci = int(row[0])
            ref_N_nobh = float(row[1][0])
            g = dereference_galaxy(handler, ci)
            w_pop = make_w_pop_eff(completeness, g.phi_g, g.theta_g, g.z_g, g.sigma_z, h)
            kern = make_galaxy_z_kernel(g.z_g, g.sigma_z, w_pop_eff=w_pop)
            l_gw = make_l_gw(se, g.phi_g, g.theta_g, h)
            ours = per_galaxy_numerator(l_gw, kern.rho, win[0], win[1])
            rel = abs(ours - ref_N_nobh) / ref_N_nobh if ref_N_nobh != 0 else float("nan")
            print(f"event {ev} galaxy {ci}: ours={ours:.8e} ref={ref_N_nobh:.8e} rel={rel:.2e}")
        n_checked += 1


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Pairwise Eq.(31) cross-term instrument (BUILD ONLY — running on "
            "production data requires --confirm-run per the pre-registration gate)."
        )
    )
    p.add_argument("--venue", choices=sorted(VENUE_CONFIGS.keys()), required=True)
    p.add_argument(
        "--h-grid",
        default=",".join(str(x) for x in DEFAULT_H_GRID),
        help="Comma-separated h values (floor: 0.60,0.73,0.81,0.86)",
    )
    p.add_argument(
        "--output",
        required=True,
        help="Output JSON path. REQUIRED and refused inside production dirs — "
        "there is deliberately no default.",
    )
    p.add_argument("--channel", choices=["1d", "2d", "both"], default="1d")
    p.add_argument("--crb-path", default=str(CRB_PATH))
    p.add_argument(
        "--ball-h-file",
        default="h_0_73.json",
        help="frozeng posteriors_with_bh_mass file used for ball membership",
    )
    p.add_argument(
        "--spot-check-balls",
        default=None,
        metavar="H_FILE",
        help="Second frozeng h file; abort if ball sets differ (h-independence check)",
    )
    p.add_argument("--pairs-limit", type=int, default=None, help="Smoke-test cap")
    p.add_argument(
        "--pair-list",
        default=None,
        metavar="JSON",
        help=(
            "Explicit target pair set (R-3): a make_target_pairs.py JSON with "
            "per-channel pair lists (the M-4 truly-sharing census). Replaces "
            "the C-4 census as the pair ENUMERATION only; the census still "
            "feeds the overlap-degree fields and the in_c4 flag."
        ),
    )
    p.add_argument(
        "--mode",
        choices=["instrument", "validate"],
        default="instrument",
        help="'validate' cross-checks Step-2 N_g against frozeng per-galaxy rows",
    )
    p.add_argument(
        "--confirm-run",
        action="store_true",
        help="Pre-registration gate: without this flag the CLI prints the plan and exits.",
    )
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if not args.confirm_run:
        print("DRY RUN (pre-registration gate: --confirm-run not given). Plan:")
        print(f"  mode        = {args.mode}")
        print(f"  venue       = {args.venue}")
        print(f"  channel     = {args.channel}")
        print(f"  h grid      = {args.h_grid}")
        print(f"  CRB         = {args.crb_path}")
        print(f"  catalogue   = {VENUE_CONFIGS[args.venue]['catalogue']}")
        print(
            f"  ball source = {VENUE_CONFIGS[args.venue]['frozeng_dir']}/posteriors_with_bh_mass/{args.ball_h_file}"
        )
        print(f"  pair source = {args.pair_list or 'C-4 pair census (c4_pair_census)'}")
        print(f"  output      = {args.output}")
        h_values = [float(x) for x in args.h_grid.split(",")]
        channels = ["1d", "2d"] if args.channel == "both" else [args.channel]
        if args.pair_list:
            # Plan detail from the target-set file (the ONLY file a dry run
            # reads; it is this directory's own zero-compute census emit, not
            # production data — no CRB/catalogue/frozeng read happens here).
            data = load_pair_list(Path(args.pair_list), args.venue)
            total_rows = 0
            for ch in channels:
                plist = data["pairs"].get(ch, [])
                n_pairs = (
                    len(plist) if args.pairs_limit is None else min(len(plist), args.pairs_limit)
                )
                n_in = sum(1 for p in plist[:n_pairs] if p.get("in_c4"))
                print(
                    f"  plan[{ch}]    = {n_pairs} pairs "
                    f"({n_in} in-C-4, {n_pairs - n_in} outside) x {len(h_values)} h "
                    f"= {n_pairs * len(h_values)} rows"
                )
                total_rows += n_pairs * len(h_values)
            print(
                f"  plan total  = {total_rows} rows "
                f"({len(h_values)} h x {len(channels)} channel(s))"
            )
            counts = data.get("meta", {}).get("counts")
            if counts:
                print(f"  target-set meta counts = {counts}")
            print("Only the pair-list JSON was read; nothing was written.")
        else:
            print("Nothing was read or written.")
        print("Re-run with --confirm-run once the")
        print("pre-registration (negligibility band X/Y) is locked.")
        return 0
    if args.mode == "validate":
        if args.pairs_limit is None:
            args.pairs_limit = 5
        run_validate(args)
    else:
        run_instrument(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
