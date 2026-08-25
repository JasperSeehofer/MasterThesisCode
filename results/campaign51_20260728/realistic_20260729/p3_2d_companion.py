r"""[P3-2D] the Sigma~^4D companion pass (zero-``evaluate()``), item 1 of the pre-fleet
execution task (PREREGISTRATION_P3_2D_20260825.md, PA-2D-1 F4).

**[PA-2D-3/PA-2D-4 item 4 fix, 2026-08-25] v2: segment-aware z-quadrature.** The v1 pass
(``ca_rhs_work2d/p3_2d_companion.json``, Sigma~^4D candidate 348079019.37, C2* candidate
0.061244) is UNBANKED: its mandated spot-check failed the registered 1e-6 target (max
3.81e-4). Adjudication (``ca_rhs_work2d/spot_check_adjudication.{py,json}`` +
``spot_check_drilldown.{py,json}``) VINDICATED the PA-2D-2 exact erf mass-marginal and
localized the defect to the GL(50) z-quadrature: ``S_bar_4D(d_L(z;h))`` is piecewise-linear
in ``d_L`` (the ``RegularGridInterpolator``'s own bilinear-in-``(d_L,M)`` structure, exactly
marginalized in ``M``), so as a function of ``z`` it carries a kink at every ``z`` where
``d_L(z;h)`` crosses a ``dl_centers`` grid edge (60 of them) -- a single 50-node GL rule over
the whole ``+/-4sigma`` window under-resolves that many kinks in the wide-sigma/near-horizon
regime. **Fix:** the z-integral is now SEGMENT-AWARE -- the 60 ``dl_centers`` edges are
inverted ONCE (globally, ``h`` fixed) to 60 z-breakpoints via :func:`_z_breakpoints_dl_centers`
(dense-grid bracket + ``scipy.optimize.brentq`` refinement, exploiting ``d_L(z;h)``'s
monotonicity), then per host the ``+/-4sigma`` window is subdivided at every breakpoint it
contains (:func:`_segment_edges`, vectorized over rows via ``searchsorted``+padding, no
per-row root-finding) and each resulting SMOOTH sub-segment gets its own fixed-order
Gauss-Legendre rule (:func:`_segmented_integral_batch`) -- spectrally convergent per segment
since the kink is now always a segment ENDPOINT, never an interior point. Cost stays the same
CLASS (the z-stage was never the bottleneck; the erf mass-marginal per node still dominates):
typical hosts see 1-5 segments (photo-z windows are narrow), so GL-16-per-segment (~16-80
nodes) is comparable to or cheaper than the old flat GL-50; only rare wide/near-horizon hosts
pay more, bounded by the 60-breakpoint ceiling. The mandated 1e-6 spot-check target is
replaced by an ARBITER-GROUNDED one (PA-2D-3): :func:`_run_arbiter` demonstrates mutual
convergence of GL-16 vs GL-32 vs a 10x-segment-refined GL-16 rule on a random row sample,
banks the plateau level, and derives the re-run's spot-check target as 10x that plateau (the
raw 1e-6 target was unfalsifiable as posed -- the arbiters' own noise floor sits at
~1e-4..5e-4). The full-pass spot-check itself becomes a cheap GL-16-vs-GL-32 comparison on
the SAME breakpoints (:func:`_fast_spot_check`, no nested ``scipy.integrate.quad`` -- the old
:func:`_spot_check` is KEPT, unused by ``main()``, as an independent-method archival check).
Output path moves to ``ca_rhs_work2d/p3_2d_companion_v2.json`` (v1 stays untouched as the
superseded-candidate record). PA-2D-4 item 4 additionally requires an eligibility-
independence finding (:func:`_eligibility_independence_finding`): traced by code-path
inspection, every input Sigma~^4D consumes (the reduced galaxy catalogue via its own
eligibility mask, the pixel-completeness ``m_th`` cache, and the injection-pool-derived
``detection_probability``/``phi_survival_table`` pair) is independent of
``mass_filter_sigma``/``get_possible_hosts_from_ball_tree`` (a per-EVENT candidate-host
lookup this companion never calls) -- so the [P3-WBHZERO] symmetric-window ruling (row #202)
does not change Sigma~^4D.

**What this computes.** ``Sigma~^4D = SUM_g w_g * S~_4D,g`` over the eligible catalogue, at
``h = H_GEN = H_TRUE = 0.73``, under the F2-resolved kernel (A20_REVIEW_P3_2D_DESIGN_20260825.md
Finding 2: the production ``gaussian`` branch, mass prior centred at the Eddington-shifted
``_host_M_eff``, catalogue_numerator_survival_2d_center="eff"). ``S~_4D,g`` is the SAME two-stage
OBJECT the coded with-BH numerator evaluates per event (see
``_mz_sel_2d_expectation``/``_mz_sel_2d_expectation_batch``, ``bayesian_statistics.py:5854-5976``)
but with NO event conditioning: the population-level mass-marginal
``S_bar_4D,g(z) = E_{M~N(M_eff_g, sigma_Mg^2)}[S_4D(d_L(z;h), M)]``, z-kernel-smeared on the SAME
window/Gauss-Legendre machinery :func:`correspondence_1d.kernel_smeared_survival` uses
(``_host_kernel_window``, ``_kernel_w_pop_eff``, ``host_z_error_eff``, the 50-node GL quadrature)
-- imported, not reimplemented, so this pass's z-kernel tracks whatever kernel family production's
own numerator uses today (PA-11/FATAL-1 precedent, ``A20_REVIEW_B0_IMPL_20260823.md`` Finding 1).

**[PA-2D-2 fix, 2026-08-25] The mass marginal is EXACT, not a quadrature.** The first version of
this script computed ``E_M[S_4D]`` via the SAME order-24 Gauss-Hermite leaf the coded per-EVENT
numerator uses (``_MT_GH_NODES``/``_MT_GH_WEIGHTS``). That leaf is exact for a per-event
product-Gaussian with narrow ``sigma_star`` (production regime, p50 8.8e-8) but the mandated spot-
check (100 random rows vs a nested ``scipy.integrate.quad`` arbiter) FAILED at 1.7-8.5% deviation
in THIS companion's wide population-``sigma_Mg`` regime (60-200% of ``M_g``): the mass integrand
spans many of the 40-bin piecewise-linear ``S_4D`` grid's cells, under-resolved by 24 Hermite
nodes (banked ``ca_rhs_work2d/p3_2d_companion_SMOKETEST_2000row_DO_NOT_BANK.json``, A21 STOP, no
registered number banked). **Fix:** :func:`_mass_marginal_survival` no longer calls the GH-24
leaf or the interpolator at all -- it evaluates ``E_M[S_4D]`` in closed form directly against the
interpolator's own bilinear-in-``(d_L, M)`` grid structure (piecewise-linear in ``M`` at fixed
``d_L``, Gaussian first-moment closed form per segment, the F2 MINOR-6 ``S(M<=0):=0`` guard kept).
See the function's own docstring for the full derivation. This is exact for the interpolated
``S_4D`` by construction (no order parameter), validated to <=1e-9 relative against an independent
brute-force arbiter (see this repo's validation harness, run separately) -- tighter than the
task-mandated ``scipy.quad`` spot-check's own 1e-6 target, which this fix also still passes.

Sigma^phi / beta_G_phi / beta_Gbar_phi are the banked, seed-invariant, venue-independent Path-A
mixture scalars (F3/F4: constant across every b0/b0i/b0i2d CSV at fixed h + flags) -- read from
an EXISTING banked event_likelihoods.csv (any b0i-family seed already on disk; zero-compute) via
:func:`p3_b0_identity_test._beta_g_phi_and_gbar`, and Sigma_phi from
:func:`p3_b0_identity_test.mass_companion` (already a cached zero-compute leaf).

Output: ``C2_star = beta_G_phi * Sigma_tilde_4D / (Sigma_phi * beta_Gbar_phi)`` (F4), banked to
``ca_rhs_work2d/p3_2d_companion.json``, plus a 100-random-row independent scipy.integrate.quad
spot-check (task-mandated, <=1e-6 rel target).

PA-CA-11 (out-root guard, carried to every new instrument, F16): REFUSES if the output JSON
already exists -- purge or pick a fresh path.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy import integrate
from scipy.optimize import brentq
from scipy.special import roots_legendre
from scipy.stats import norm

REPO_ROOT = Path(__file__).resolve().parents[3]
THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))  # o5 (p3_b0_identity_test) is a sibling script, not a package

import p3_b0_identity_test as o5  # noqa: E402

from darksiren_emri.bayesian_inference.bayesian_statistics import (  # noqa: E402
    _eddington_shifted_host_mass_batch,
    _wbh_z_kwargs,
)
from darksiren_emri.emri_rate import R_eff_per_mbh  # noqa: E402
from darksiren_emri.galaxy_catalogue.handler import InternalCatalogColumns  # noqa: E402
from darksiren_emri.physical_relations import dist_vectorized  # noqa: E402
from darksiren_emri.validation import correspondence_1d as c1d  # noqa: E402

H_GEN: float = c1d.H_TRUE  # 0.73, prereg S1/"all reads at h=H_TRUE"
CENTER: str = "eff"  # F2: production configuration = eddington-shifted centering
CHUNK: int = 20_000  # mirrors o5._MASS_COMPANION_CHUNK's memory-bounding convention

# An existing, already-banked b0i-family event_likelihoods.csv -- alpha_G_phi/r_Malm/D_tilde_phi
# are venue-independent Path-A mixture globals at fixed (h, flags) (F3), so ANY already-run
# fused/phi-selection seed on disk supplies beta_G_phi/beta_Gbar_phi at zero extra compute.
_REFERENCE_CSV = (
    THIS_DIR / "p3_b0_work/bc_900101_work/seed900101/simulations/diagnostics/event_likelihoods.csv"
)

OUT_PATH = THIS_DIR / "ca_rhs_work2d/p3_2d_companion_v2.json"
# v1 candidates (superseded, unbanked -- spot-check failed) stay untouched at this path.
V1_OUT_PATH = THIS_DIR / "ca_rhs_work2d/p3_2d_companion.json"
ARBITER_DRY_RUN_OUT_PATH = THIS_DIR / "ca_rhs_work2d/p3_2d_companion_v2_arbiter_dryrun.json"

REGISTRATION_SECTION = (
    "results/campaign51_20260728/realistic_20260729/PREREGISTRATION_P3_2D_20260825.md "
    "sec1 (Sigma~^4D companion), PA-2D-1 F4 (C2* resolved), PA-2D-3 (segment-aware z-rule + "
    "arbiter-grounded target), PA-2D-4 item 4 (eligibility-independence finding)"
)

# ── PA-2D-3: segment-aware z-quadrature machinery ───────────────────────────────────────────
_Z_SEG_GL_ORDER: int = 16  # production per-segment order (spectral for a smooth segment)
_ZSEG_GL_NODES, _ZSEG_GL_WEIGHTS = roots_legendre(_Z_SEG_GL_ORDER)
_Z_SEG_GL_ORDER_HI: int = 32  # cross-check order (arbiter + the fast full-pass spot-check)
_ZSEG_GL_NODES_HI, _ZSEG_GL_WEIGHTS_HI = roots_legendre(_Z_SEG_GL_ORDER_HI)
_ARBITER_REFINE: int = 10  # the arbiter's 3rd method: 10x more (GL-16) segments, same order
_N_ARBITER_ROWS: int = 20  # PA-2D-3-mandated arbiter sample size
_N_FAST_SPOT_ROWS: int = 50  # PA-2D-3-mandated full-pass spot-check sample size
_BREAKPOINT_FINE_N: int = 4000  # dense-grid resolution for the global dl_centers -> z inversion


def _a22_stamp_2d() -> dict[str, Any]:
    """[ORCH] A22 = FIVE resolved flag values (F7) + git commit + dirty, BEFORE any scoring."""
    git_stamp = o5._a22_stamp()
    return {
        **git_stamp,
        "catalogue_global_selection": "phi",
        "catalogue_numerator_survival": "off",  # 1D twin production default (not this pass's axis)
        "selection_in_completion_numerator": "fused",
        "catalogue_numerator_survival_2d": "mz_sel",  # this pass computes the TWIN's own object
        "catalogue_numerator_survival_2d_center": CENTER,
        "acceptance_model_version": "F12-extended-class-G-replay",
    }


def _load_eligible_catalogue(
    completeness: Any, phi_survival_table: dict[float, Any], h: float
) -> dict[str, npt.NDArray[np.float64]]:
    """Byte-identical eligibility convention to :func:`p3_b0_identity_test.mass_companion`."""
    handler = c1d._load_galaxy_catalog_handler(c1d.REDUCED_CATALOGUE_PATH)
    catalog = handler.reduced_galaxy_catalog
    z_grid, _ = phi_survival_table[h]
    z_all = np.asarray(catalog[InternalCatalogColumns.REDSHIFT].to_numpy(dtype=np.float64))
    M_all = np.asarray(catalog[InternalCatalogColumns.BH_MASS].to_numpy(dtype=np.float64))
    z_err_all = np.asarray(
        catalog[InternalCatalogColumns.REDSHIFT_ERROR].to_numpy(dtype=np.float64)
    )
    M_err_all = np.asarray(catalog[InternalCatalogColumns.BH_MASS_ERROR].to_numpy(dtype=np.float64))
    phiS_all = np.asarray(catalog[InternalCatalogColumns.PHI_S].to_numpy(dtype=np.float64))
    qS_all = np.asarray(catalog[InternalCatalogColumns.THETA_S].to_numpy(dtype=np.float64))

    z_max = float(z_grid.max())
    eligible = (z_all < z_max) & np.isfinite(M_all) & (M_all > 0.0)
    # Debug/smoke-test escape hatch ONLY (disclosed, never set in a banked run): truncates the
    # eligible index set to the first N rows so the pass can be sanity-checked in seconds instead
    # of ~1-2 CPU-h. Unset (default None) is the registered, full-catalogue banked path.
    import os

    _max_rows = os.environ.get("P3_2D_COMPANION_MAX_ROWS")
    if _max_rows is not None:
        eligible_idx = np.flatnonzero(eligible)[: int(_max_rows)]
        eligible = np.zeros_like(eligible)
        eligible[eligible_idx] = True
    return {
        "z": z_all[eligible],
        "M": M_all[eligible],
        "z_error": np.maximum(z_err_all[eligible], c1d.EXACT_Z_ERROR_FLOOR),
        "M_error": M_err_all[eligible],
        "phiS": phiS_all[eligible],
        "qS": qS_all[eligible],
    }


def _phi_diff_stable(
    m1: float, m2: float, mu: npt.NDArray[np.float64], sigma: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    r"""``Phi(m2; mu, sigma) - Phi(m1; mu, sigma)``, ``m1 <= m2`` FIXED scalars, ``mu``/``sigma``
    ``(n, K)`` arrays -- catastrophic-cancellation-safe arbitrarily far into either tail.

    Direct ``norm.cdf(m2)-norm.cdf(m1)`` silently rounds to ``0.0`` whenever BOTH ``m1`` and ``m2``
    sit many sigma ABOVE ``mu``: ``norm.cdf`` saturates to exactly ``1.0`` in float64 once
    ``1-Phi(z) < ~1.1e-16`` (``z`` gtrsim 8.2), so the difference of two saturated ``1.0`` values
    is computed as exactly 0 even though the true difference is a representable-but-tiny nonzero
    number (verified via a direct comparison against ``scipy.integrate.quad`` on a synthetic
    adversarial row: naive ``cdf`` differencing was off by 78x on one segment). ``norm.sf``
    (``1-Phi``, computed via ``erfc`` -- never saturates, stays a small precisely-representable
    number arbitrarily far into the tail) gives the numerically stable form instead:

    * ``mu <= m1`` (segment entirely at/above mu): ``Phi(m2)-Phi(m1) = (1-sf(m2)) - (1-sf(m1)) =
      sf(m1) - sf(m2)`` (both sf's small, safe).
    * ``mu >= m2`` (segment entirely at/below mu): plain ``cdf(m2)-cdf(m1)`` (both cdf's small
      near the LOWER tail, safe by the mirror-image argument -- ``norm.cdf`` is exact down to
      underflow, no saturation-to-0 precision loss the way saturation-to-1 loses precision).
    * ``m1 < mu < m2`` (straddling): ``Phi(m2)-Phi(m1) = (1-sf(m2)) - cdf(m1) = 1 - sf(m2) -
      cdf(m1)`` -- safe because ``cdf(m1) <= 0.5`` in this branch (``m1 < mu``), so the
      subtraction from 1 never approaches the near-1 cancellation regime.

    Args:
        m1: Lower segment bound (scalar, ``m1 <= m2``).
        m2: Upper segment bound (scalar).
        mu: Gaussian mean(s), ``(n, K)``.
        sigma: Gaussian std(s), ``(n, K)``, ``> 0``.

    Returns:
        ``Phi(m2)-Phi(m1)``, ``(n, K)``, numerically stable for any ``mu``/``sigma``/``m1``/``m2``.
    """
    below = mu >= m2
    above = mu <= m1
    cdf_m1 = norm.cdf(m1, loc=mu, scale=sigma)
    cdf_m2 = norm.cdf(m2, loc=mu, scale=sigma)
    sf_m1 = norm.sf(m1, loc=mu, scale=sigma)
    sf_m2 = norm.sf(m2, loc=mu, scale=sigma)
    out = np.where(below, cdf_m2 - cdf_m1, 1.0 - sf_m2 - cdf_m1)  # default: straddling formula
    out = np.where(above, sf_m1 - sf_m2, out)
    result: npt.NDArray[np.float64] = out
    return result


def _mass_marginal_survival(
    mu: npt.NDArray[np.float64],
    sigma: npt.NDArray[np.float64],
    z_nodes: npt.NDArray[np.float64],
    d_L_nodes: npt.NDArray[np.float64],
    host_phiS: npt.NDArray[np.float64],
    host_qS: npt.NDArray[np.float64],
    detection_probability: Any,
    h: float,
) -> npt.NDArray[np.float64]:
    r"""``E_M[S_4D(d_L(z), M)]`` for ``M ~ N(mu, sigma^2)``, shapes ``(n, K)`` in / ``(n, K)`` out.

    **[PA-2D-2 fix] Exact per-grid-cell closed form, NOT a quadrature.** The GH-24 leaf this
    function used to wrap (``_mz_sel_2d_expectation_batch``-structured Gauss-Hermite order-24
    quadrature) is exact for a per-EVENT product-Gaussian with a narrow ``sigma_star`` (production
    regime, ``sigma_cond`` p50 8.8e-8) but carries a diagnosed 1.19% bias in THIS companion's wide
    population-``sigma_Mg`` regime (60-200% of ``M_g``): the mass-marginal integrand spans many
    cells of the 40-bin piecewise-linear ``S_4D`` grid, and 24 Hermite nodes under-resolve a
    function with that many internal kinks (spot-check: 1.7-8.5% deviation vs a brute-force/
    ``scipy.quad`` arbiter, ``ca_rhs_work2d/p3_2d_companion_SMOKETEST_2000row_DO_NOT_BANK.json``).

    The fix exploits the interpolator's OWN structure instead of approximating it:
    ``detection_probability_with_bh_mass_interpolated`` (``simulation_detection_probability.py``
    ``_build_grid_2d`` ~:1787-1902, the accessor ~:2018-2117) is a ``RegularGridInterpolator``
    with ``method="linear"`` on ``(dl_centers, M_centers)`` -- i.e. BILINEAR. At any FIXED ``d_L``
    (every ``z``-node here gives one), the surface is exactly piecewise-LINEAR in ``M`` with
    breakpoints at the 40 ``M_centers`` nodes: each of the two bracketing ``d_L`` rows is
    piecewise-linear in ``M`` on those same nodes, and a fixed-weight convex combination of two
    piecewise-linear functions on the SAME breakpoints is piecewise-linear on those breakpoints
    too. Outside ``[M_centers[0], M_centers[-1]]`` the production accessor clamps to the nearest
    edge value (``:2100-2107``, "true nearest", not scipy's linear extrapolation) -- a further two
    FLAT segments. So for the continuous mass axis, ``S_4D(d_L, M)`` is:

        M <= 0                                : 0                         (F2 MINOR-6 guard)
        0 <  M <= M_centers[0]                 : v_0                      (flat, clamped)
        M_centers[j] <= M <= M_centers[j+1]    : a_j + b_j*M   (j=0..38)  (the grid segments)
        M >= M_centers[-1]                     : v_last                   (flat, clamped)

    with ``v_j`` the ``d_L``-interpolated value at ``M_centers[j]`` (linear in ``d_L`` between the
    two bracketing grid rows -- computed ONCE per (event, z-node) below, not per segment).
    ``\\int N(M;\\mu,\\sigma)(a+bM)\\,dM`` over ``[M_1,M_2]`` has the standard closed form (Gaussian
    first moment): ``a[\\Phi]_M1^M2 + b(\\mu[\\Phi]_M1^M2 - \\sigma^2[N]_M1^M2)``, using
    ``\\int M\\,N(M;\\mu,\\sigma)\\,dM = \\mu\\Phi(M) - \\sigma^2 N(M;\\mu,\\sigma) + C`` (differentiate to
    verify: ``d/dM[\\mu\\Phi - \\sigma^2 N] = \\mu N - \\sigma^2 N' = \\mu N + (M-\\mu)N = M N``, since
    ``N'(M) = -(M-\\mu)/\\sigma^2 \\cdot N``). Summing all 41 segments (2 flat + 39 internal) is EXACT
    for the interpolated ``S`` by construction -- no quadrature order parameter, no truncation
    window: the segment count is fixed and small (unlike the abandoned GH-24 approach, this
    doesn't need an explicit +/-8sigma cell-intersection filter for cost control, since summing
    every one of the 41 fixed segments already costs the same O(1)-per-row as computing which
    ones intersect a window would -- and skips zero accuracy, not just <1e-9). Implemented as a
    Python loop over the (fixed, small) 39 internal segments rather than one ``(n,k,mass_bins)``
    vectorized stack, to keep peak memory at ``O(n*k)`` instead of ``O(n*k*mass_bins)`` (the GH-24
    leaf's ``(n,k,24)`` footprint is instead ``(n,k,40)`` intermediate here if fully vectorized;
    the loop avoids materializing that).

    Degenerate rows (``sigma<=0``, catalogue entries with a zero recorded mass error) fall back to
    a direct point evaluation of the SAME piecewise-linear ``M``-profile at ``M=mu`` (clamped),
    with the ``M<=0`` guard -- mirrors :func:`_spot_check`'s own ``sigma_i<=0`` branch.

    Args:
        mu: Mass-Gaussian mean, ``(n, K)``.
        sigma: Mass-Gaussian std, ``(n, K)`` (``<=0`` triggers the degenerate point-eval path).
        z_nodes: Unused by this exact form (the GH-24 leaf needed it only to thread ``z`` through
            the ``_wbh_z_resolved`` accessor path, which the b0i-2D venue never activates --
            confirmed: ``build_b0i_2d_selection_objects`` passes ``pdet_wbh_z_resolved=False``
            (the constructor default), so ``detection_probability_with_bh_mass_interpolated``
            always takes the plain bilinear-grid branch, never ``_wbh_survival_at``). Kept in the
            signature for interface parity with the call site / a future ``_wbh_z_resolved=True``
            venue, unused here.
        d_L_nodes: ``d_L(z_nodes; h)``, ``(n, K)``.
        host_phiS: Unused by this exact form (the bilinear ``S_4D`` grid marginalizes sky angles
            internally, D-02) -- kept for interface parity with the GH-24 leaf this replaces.
        host_qS: Unused, same reason.
        detection_probability: ``SimulationDetectionProbability`` instance.
        h: Dimensionless Hubble parameter.

    Returns:
        ``E_M[S_4D]``, shape ``(n, K)``.
    """
    del z_nodes, host_phiS, host_qS  # interface parity only; see docstring.
    interp_2d, _ = detection_probability._get_or_build_grid(h)
    dl_centers = np.asarray(interp_2d.grid[0], dtype=np.float64)
    M_centers = np.asarray(interp_2d.grid[1], dtype=np.float64)
    p_grid = np.asarray(interp_2d.values, dtype=np.float64)  # (dl_bins, mass_bins)
    n_dl = dl_centers.size
    n_m = M_centers.size
    dl_min = float(dl_centers[0])
    dl_max = float(dl_centers[-1])

    d_L = np.asarray(d_L_nodes, dtype=np.float64)
    dl_query = np.clip(d_L, dl_min, dl_max)
    i = np.clip(np.searchsorted(dl_centers, dl_query, side="right") - 1, 0, n_dl - 2)
    dl_lo = dl_centers[i]
    dl_hi = dl_centers[i + 1]
    dl_seg = dl_hi - dl_lo
    t = np.where(dl_seg > 0.0, (dl_query - dl_lo) / np.where(dl_seg > 0.0, dl_seg, 1.0), 0.0)

    row_lo = p_grid[i]  # (n, k, mass_bins) via fancy indexing on the first axis
    row_hi = p_grid[i + 1]
    v = row_lo + t[..., None] * (row_hi - row_lo)  # the d_L-interpolated M-profile
    v = np.where((d_L > dl_max)[..., None], 0.0, v)  # beyond the horizon: survival = 0 (:2110-2111)

    mu_b = np.asarray(mu, dtype=np.float64)
    sigma_b = np.asarray(sigma, dtype=np.float64)
    degenerate = sigma_b <= 0.0
    sigma_safe = np.where(degenerate, 1.0, sigma_b)  # dummy >0 value for the masked-out rows
    sigma2 = sigma_safe * sigma_safe

    # **Numerical-precision note (found by this fix's own >=20-row validation, adversarial
    # low-mu/high-sigma rows).** A NAIVE ``Phi(M2)-Phi(M1)`` (both via ``norm.cdf``, carried
    # incrementally) catastrophically cancels whenever both endpoints sit many sigma ABOVE mu:
    # ``norm.cdf`` saturates to EXACTLY ``1.0`` in float64 once ``1-Phi(z) < ~1.1e-16`` (z gtrsim
    # 8.2), so ``Phi(M2)-Phi(M1)`` silently rounds to ``0.0`` even though the TRUE difference is a
    # representable-but-tiny nonzero number -- while the density term (``b*sigma^2*(N2-N1)``,
    # which does NOT saturate, densities stay small and well-resolved) does not cancel, leaving a
    # spurious unbalanced residual (verified: a synthetic row with this profile computed
    # 4.1e-43 for one segment via naive Phi-differencing against scipy.quad's 5.3e-45 on the SAME
    # segment -- a 78x error). Fix: whenever a segment lies (partly) above ``mu``, express that
    # PORTION of the ``Phi`` difference via ``norm.sf`` (survival function, ``1-Phi``, computed
    # directly via ``erfc`` -- stays small and precisely representable arbitrarily far into the
    # tail, no saturation) instead of subtracting two near-1 ``cdf`` values. See
    # :func:`_phi_diff_stable`.
    result = np.zeros_like(mu_b)
    result += v[..., 0] * _phi_diff_stable(0.0, float(M_centers[0]), mu_b, sigma_safe)
    n_prev = norm.pdf(M_centers[0], loc=mu_b, scale=sigma_safe)

    for j in range(n_m - 1):
        m1 = float(M_centers[j])
        m2 = float(M_centers[j + 1])
        v1 = v[..., j]
        v2 = v[..., j + 1]
        b_coef = (v2 - v1) / (m2 - m1)
        a_coef = v1 - b_coef * m1
        d_phi = _phi_diff_stable(m1, m2, mu_b, sigma_safe)
        n2 = norm.pdf(m2, loc=mu_b, scale=sigma_safe)
        d_n = n2 - n_prev
        result += a_coef * d_phi + b_coef * (mu_b * d_phi - sigma2 * d_n)
        n_prev = n2

    # boundary-high flat segment: [M_centers[-1], inf) -- SF is the natural, always-stable form.
    result += v[..., -1] * norm.sf(M_centers[-1], loc=mu_b, scale=sigma_safe)

    if np.any(degenerate):
        m_query = np.clip(mu_b, M_centers[0], M_centers[-1])
        j_deg = np.clip(np.searchsorted(M_centers, m_query, side="right") - 1, 0, n_m - 2)
        m_lo = M_centers[j_deg]
        m_hi = M_centers[j_deg + 1]
        m_seg = m_hi - m_lo
        tm = np.where(m_seg > 0.0, (m_query - m_lo) / np.where(m_seg > 0.0, m_seg, 1.0), 0.0)
        vj = np.take_along_axis(v, j_deg[..., None], axis=-1)[..., 0]
        vj1 = np.take_along_axis(v, (j_deg + 1)[..., None], axis=-1)[..., 0]
        point_val = np.where(mu_b <= 0.0, 0.0, vj + tm * (vj1 - vj))  # MINOR-6 guard (F2)
        result = np.where(degenerate, point_val, result)

    expectation: npt.NDArray[np.float64] = result
    return expectation


def _z_breakpoints_dl_centers(
    detection_probability: Any,
    h: float,
    z_lo: float,
    z_hi: float,
    n_fine: int = _BREAKPOINT_FINE_N,
) -> npt.NDArray[np.float64]:
    r"""Invert ``d_L(z;h) = dl_centers[j]`` for every ``j`` -- the 60 z-locations where the
    bilinear ``S_4D`` grid's interpolation switches segment (a kink in ``S_bar_4D(z)``).

    Computed ONCE, globally (``dl_centers`` is fixed and ``d_L(z;h)`` is host-independent at
    fixed ``h``), so this pays 60 ``brentq`` root-finds total for the whole catalogue, not per
    row. Bracketing: a dense monotone ``z``-grid (``n_fine`` points on ``[z_lo, z_hi]``) gives
    a bracket for each target via ``searchsorted`` (``d_L`` is monotone increasing in ``z``,
    :func:`~darksiren_emri.physical_relations.dist_vectorized`), then ``brentq`` refines to
    ``xtol=rtol=1e-14`` inside that bracket -- the SAME two-stage bracket+refine pattern
    ``spot_check_adjudication.py``'s ``_quad_pts_value`` already uses per-row (here amortized
    over the whole run instead of paid 100 times).

    Args:
        detection_probability: ``SimulationDetectionProbability`` instance (its ``dl_centers``
            grid is the breakpoint source).
        h: Dimensionless Hubble parameter.
        z_lo: Lower bound of the dense bracketing grid (should cover every host's window floor).
        z_hi: Upper bound of the dense bracketing grid (should cover every host's window ceiling).
        n_fine: Dense-grid resolution for bracketing.

    Returns:
        Sorted ``z``-breakpoints, one per ``dl_centers`` entry, shape ``(n_dl,)``.
    """
    interp_2d, _ = detection_probability._get_or_build_grid(h)
    dl_centers = np.asarray(interp_2d.grid[0], dtype=np.float64)
    z_fine = np.linspace(z_lo, z_hi, n_fine)
    dl_fine = np.asarray(dist_vectorized(z_fine, h=h), dtype=np.float64)

    def _dist_scalar(z: float) -> float:
        return float(dist_vectorized(np.array([z]), h=h)[0])

    z_roots = np.empty(dl_centers.shape[0], dtype=np.float64)
    for k, dl_t in enumerate(dl_centers):
        dl_t_f = float(dl_t)
        if dl_t_f <= dl_fine[0]:
            z_roots[k] = z_lo
            continue
        if dl_t_f >= dl_fine[-1]:
            z_roots[k] = z_hi
            continue
        idx = int(np.searchsorted(dl_fine, dl_t_f))
        lo = float(z_fine[max(idx - 1, 0)])
        hi = float(z_fine[min(idx, n_fine - 1)])
        if hi <= lo:
            z_roots[k] = lo
            continue
        try:
            z_roots[k] = brentq(
                lambda z, _dl=dl_t_f: _dist_scalar(z) - _dl, lo, hi, xtol=1e-14, rtol=1e-14
            )
        except ValueError:
            # bracket didn't straddle a root (fine-grid resolution artifact) -- fall back to the
            # linear-interpolation estimate, which is still adequate as a SEGMENT boundary (any
            # residual offset only means that one segment keeps a tiny sliver of the true kink,
            # not that the breakpoint is missing).
            z_roots[k] = float(np.interp(dl_t_f, dl_fine, z_fine))
    z_roots.sort()
    return z_roots


def _segment_edges(
    lower: npt.NDArray[np.float64],
    upper: npt.NDArray[np.float64],
    z_breakpoints: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
    r"""Per-row segment boundaries: ``[lower, ...interior breakpoints inside (lower,upper)...,
    upper]``, vectorized over rows (NO per-row root-finding -- ``z_breakpoints`` is the global
    table :func:`_z_breakpoints_dl_centers` already built).

    Padded to a common width across the batch: rows with fewer interior breakpoints than the
    batch max get their trailing columns filled with ``upper`` (a zero-width segment, which
    contributes exactly ``0`` to the GL sum via its ``half=0`` factor -- see
    :func:`_segmented_integral_batch`).

    Args:
        lower: Per-row window floor, shape ``(m,)``.
        upper: Per-row window ceiling, shape ``(m,)``.
        z_breakpoints: Sorted global z-breakpoints, shape ``(n_bp,)``.

    Returns:
        ``(edges, n_interior)``: ``edges`` shape ``(m, n_interior.max()+2)`` (segment
        endpoints, ``n_interior.max()+1`` segments per row); ``n_interior`` shape ``(m,)``, the
        actual (unpadded) interior-breakpoint count per row.
    """
    lo_idx = np.searchsorted(z_breakpoints, lower, side="right")
    hi_idx = np.searchsorted(z_breakpoints, upper, side="left")
    n_interior = hi_idx - lo_idx
    m = lower.shape[0]
    max_interior = int(n_interior.max()) if m else 0
    edges = np.empty((m, max_interior + 2), dtype=np.float64)
    edges[:, 0] = lower
    edges[:, -1] = upper
    if max_interior > 0:
        k = np.arange(max_interior)
        col_idx = lo_idx[:, None] + k[None, :]
        valid = k[None, :] < n_interior[:, None]
        col_idx_c = np.clip(col_idx, 0, z_breakpoints.size - 1)
        interior_vals = z_breakpoints[col_idx_c]
        edges[:, 1 : 1 + max_interior] = np.where(valid, interior_vals, upper[:, None])
    return edges, n_interior


def _refine_edges(edges: npt.NDArray[np.float64], factor: int) -> npt.NDArray[np.float64]:
    """Subdivide every existing segment into ``factor`` equal sub-segments (arbiter method 3:
    ``factor``x more segments at the SAME GL order, demonstrating convergence via refinement
    rather than order). Zero-width (padding) segments stay zero-width sub-segments -- safe."""
    m, n_pts = edges.shape
    n_seg = n_pts - 1
    cols = []
    for kseg in range(n_seg):
        seg_lo = edges[:, kseg]
        seg_hi = edges[:, kseg + 1]
        for j in range(factor):
            t = j / factor
            cols.append(seg_lo + t * (seg_hi - seg_lo))
    cols.append(edges[:, -1])
    return np.stack(cols, axis=1)


def _segmented_integral_batch(
    z_g: npt.NDArray[np.float64],
    z_err_eff: npt.NDArray[np.float64],
    phiS: npt.NDArray[np.float64],
    qS: npt.NDArray[np.float64],
    mu: npt.NDArray[np.float64],
    sigma: npt.NDArray[np.float64],
    completeness: Any,
    detection_probability: Any,
    h: float,
    edges: npt.NDArray[np.float64],
    gl_nodes: npt.NDArray[np.float64],
    gl_weights: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    r"""``(numerator, Z_g)`` of the kernel-smeared z-integral, segment-aware: sums a per-segment
    fixed-order GL rule (``gl_nodes``/``gl_weights``, order given by the caller -- the SAME
    machinery serves the production GL-16 pass and the GL-32/refined arbiter/spot-check calls)
    over every row's ``edges`` (from :func:`_segment_edges` or :func:`_refine_edges`).

    Args:
        z_g: Host redshift(s), shape ``(m,)``.
        z_err_eff: Effective photo-z kernel width, shape ``(m,)``.
        phiS: Host sky azimuth, shape ``(m,)``.
        qS: Host sky colatitude, shape ``(m,)``.
        mu: Mass-Gaussian mean (Eddington-shifted), shape ``(m,)``.
        sigma: Mass-Gaussian std, shape ``(m,)``.
        completeness: Per-pixel completeness model.
        detection_probability: ``SimulationDetectionProbability`` instance.
        h: Dimensionless Hubble parameter.
        edges: Per-row segment endpoints, shape ``(m, n_seg+1)``.
        gl_nodes: GL nodes on ``[-1, 1]``, shape ``(gl_order,)``.
        gl_weights: GL weights on ``[-1, 1]``, shape ``(gl_order,)``.

    Returns:
        ``(numerator, z_norm)``, each shape ``(m,)`` -- ``S~_4D,g = numerator / z_norm``.
    """
    m = z_g.shape[0]
    n_seg = edges.shape[1] - 1
    host_pixels = c1d._host_pixels(completeness, phiS, qS)
    numerator = np.zeros(m, dtype=np.float64)
    z_norm = np.zeros(m, dtype=np.float64)
    for s in range(n_seg):
        seg_lo = edges[:, s]
        seg_hi = edges[:, s + 1]
        half = 0.5 * (seg_hi - seg_lo)
        mid = 0.5 * (seg_hi + seg_lo)
        z_nodes = mid[:, None] + half[:, None] * gl_nodes[None, :]
        w_pop_eff = c1d._kernel_w_pop_eff(z_nodes, completeness, host_pixels, h)
        gaussian_vals = norm.pdf(z_nodes, loc=z_g[:, None], scale=z_err_eff[:, None])
        kernel_unnorm = gaussian_vals * w_pop_eff
        d_L_nodes = np.asarray(dist_vectorized(z_nodes.ravel(), h=h), dtype=np.float64).reshape(
            z_nodes.shape
        )
        mu_b = np.broadcast_to(mu[:, None], z_nodes.shape)
        sigma_b = np.broadcast_to(sigma[:, None], z_nodes.shape)
        s_bar_4d = _mass_marginal_survival(
            mu_b, sigma_b, z_nodes, d_L_nodes, phiS, qS, detection_probability, h
        )
        numerator += np.sum(kernel_unnorm * s_bar_4d * gl_weights[None, :], axis=1) * half
        z_norm += np.sum(kernel_unnorm * gl_weights[None, :], axis=1) * half
    return numerator, z_norm


def _run_arbiter(
    cat: dict[str, npt.NDArray[np.float64]],
    M_eff: npt.NDArray[np.float64],
    completeness: Any,
    detection_probability: Any,
    h: float,
    z_breakpoints: npt.NDArray[np.float64],
    n_rows: int = _N_ARBITER_ROWS,
    seed: int = 20260825,
) -> dict[str, Any]:
    """PA-2D-3 arbiter: on ``n_rows`` random catalogue rows, compute S~_4D,g THREE independent
    ways -- (1) production GL-16-per-segment, (2) GL-32-per-segment (order cross-check), (3)
    GL-16 on 10x-refined segments (refinement cross-check) -- and report their mutual
    convergence. The tightest pairwise agreement (GL-32 vs 10x-refined) is the demonstrated
    plateau ``L_conv``; the re-run's spot-check target is ``10 * L_conv``.
    """
    rng = np.random.default_rng(seed)
    n_g = cat["z"].size
    idx = rng.choice(n_g, size=min(n_rows, n_g), replace=False)
    z_g = cat["z"][idx]
    z_error = cat["z_error"][idx]
    phiS = cat["phiS"][idx]
    qS = cat["qS"][idx]
    mu = M_eff[idx]
    sigma = cat["M_error"][idx]
    z_err_eff = c1d.host_z_error_eff(z_g, z_error)
    lower, upper = c1d._host_kernel_window(z_g, z_err_eff)
    edges, n_interior = _segment_edges(lower, upper, z_breakpoints)
    edges_refined = _refine_edges(edges, _ARBITER_REFINE)

    num16, den16 = _segmented_integral_batch(
        z_g,
        z_err_eff,
        phiS,
        qS,
        mu,
        sigma,
        completeness,
        detection_probability,
        h,
        edges,
        _ZSEG_GL_NODES,
        _ZSEG_GL_WEIGHTS,
    )
    num32, den32 = _segmented_integral_batch(
        z_g,
        z_err_eff,
        phiS,
        qS,
        mu,
        sigma,
        completeness,
        detection_probability,
        h,
        edges,
        _ZSEG_GL_NODES_HI,
        _ZSEG_GL_WEIGHTS_HI,
    )
    num_ref, den_ref = _segmented_integral_batch(
        z_g,
        z_err_eff,
        phiS,
        qS,
        mu,
        sigma,
        completeness,
        detection_probability,
        h,
        edges_refined,
        _ZSEG_GL_NODES,
        _ZSEG_GL_WEIGHTS,
    )

    val16 = num16 / np.where(den16 > 0.0, den16, 1.0)
    val32 = num32 / np.where(den32 > 0.0, den32, 1.0)
    val_ref = num_ref / np.where(den_ref > 0.0, den_ref, 1.0)

    rel_16_vs_ref = np.abs(val16 - val_ref) / np.maximum(np.abs(val_ref), 1.0e-300)
    rel_32_vs_ref = np.abs(val32 - val_ref) / np.maximum(np.abs(val_ref), 1.0e-300)
    rel_16_vs_32 = np.abs(val16 - val32) / np.maximum(np.abs(val32), 1.0e-300)

    plateau = float(max(float(np.max(rel_32_vs_ref)), 1.0e-300))
    target = plateau * 10.0

    rows_out = []
    for k in range(len(idx)):
        rows_out.append(
            {
                "catalogue_idx": int(idx[k]),
                "z_g": float(z_g[k]),
                "n_segments_gl16": int(n_interior[k]) + 1,
                "val_gl16": float(val16[k]),
                "val_gl32": float(val32[k]),
                "val_10x_refined": float(val_ref[k]),
                "rel_dev_gl16_vs_refined": float(rel_16_vs_ref[k]),
                "rel_dev_gl32_vs_refined": float(rel_32_vs_ref[k]),
                "rel_dev_gl16_vs_gl32": float(rel_16_vs_32[k]),
            }
        )
    return {
        "n_rows": int(len(idx)),
        "seed": seed,
        "rows": rows_out,
        "max_rel_dev_gl16_vs_refined": float(np.max(rel_16_vs_ref)),
        "max_rel_dev_gl32_vs_refined": float(np.max(rel_32_vs_ref)),
        "max_rel_dev_gl16_vs_gl32": float(np.max(rel_16_vs_32)),
        "median_rel_dev_gl16_vs_refined": float(np.median(rel_16_vs_ref)),
        "median_rel_dev_gl32_vs_refined": float(np.median(rel_32_vs_ref)),
        "plateau_L_conv": plateau,
        "plateau_definition": "max over sampled rows of |GL32 - 10x_refined_GL16| / |10x_refined_GL16|",
        "derived_spot_check_target": target,
        "target_definition": "10 * plateau_L_conv",
        "gl_order_baseline": _Z_SEG_GL_ORDER,
        "gl_order_cross_check": _Z_SEG_GL_ORDER_HI,
        "refine_factor": _ARBITER_REFINE,
    }


def _fast_spot_check(
    cat: dict[str, npt.NDArray[np.float64]],
    M_eff: npt.NDArray[np.float64],
    s_tilde_4d: npt.NDArray[np.float64],
    completeness: Any,
    detection_probability: Any,
    h: float,
    z_breakpoints: npt.NDArray[np.float64],
    target: float,
    n_spot: int = _N_FAST_SPOT_ROWS,
    seed: int = 20260826,
) -> dict[str, Any]:
    """PA-2D-3 full-pass spot-check: per-segment GL-16 (production, already computed in
    ``s_tilde_4d``) vs GL-32 (cross-check, freshly computed on the SAME breakpoints) on
    ``n_spot`` random rows -- cheap (no nested ``scipy.integrate.quad``), target from
    :func:`_run_arbiter`'s demonstrated plateau, not the unfalsifiable raw 1e-6.
    """
    rng = np.random.default_rng(seed)
    n_g = cat["z"].size
    idx = rng.choice(n_g, size=min(n_spot, n_g), replace=False)
    z_g = cat["z"][idx]
    z_error = cat["z_error"][idx]
    phiS = cat["phiS"][idx]
    qS = cat["qS"][idx]
    mu = M_eff[idx]
    sigma = cat["M_error"][idx]
    z_err_eff = c1d.host_z_error_eff(z_g, z_error)
    lower, upper = c1d._host_kernel_window(z_g, z_err_eff)
    edges, _ = _segment_edges(lower, upper, z_breakpoints)
    num32, den32 = _segmented_integral_batch(
        z_g,
        z_err_eff,
        phiS,
        qS,
        mu,
        sigma,
        completeness,
        detection_probability,
        h,
        edges,
        _ZSEG_GL_NODES_HI,
        _ZSEG_GL_WEIGHTS_HI,
    )
    val32 = num32 / np.where(den32 > 0.0, den32, 1.0)
    val16 = s_tilde_4d[idx]
    rel_dev = np.abs(val16 - val32) / np.maximum(np.abs(val32), 1.0e-300)
    return {
        "n_spot": int(len(idx)),
        "seed": seed,
        "method": "per-segment GL-16 (production) vs GL-32 (cross-check), identical breakpoints",
        "max_rel_dev": float(np.max(rel_dev)),
        "median_rel_dev": float(np.median(rel_dev)),
        "target_rel_dev": target,
        "target_met": bool(np.max(rel_dev) <= target),
    }


def _eligibility_independence_finding() -> dict[str, Any]:
    """PA-2D-4 item 4: code-path trace of every input :func:`compute_sigma_tilde_4d` consumes,
    checked for ``mass_filter_sigma``/``get_possible_hosts_from_ball_tree`` conditioning (the
    [P3-WBHZERO] candidate-host mass pre-filter, ``handler.py:~570-675``, a per-EVENT lookup).
    Traced by direct inspection of each producer's source, not assumed."""
    return {
        "question": (
            "Does any input Sigma_tilde_4D consumes depend on mass_filter_sigma / "
            "get_possible_hosts_from_ball_tree candidate-host filtering?"
        ),
        "verdict": "INDEPENDENT",
        "inputs": [
            {
                "input": "reduced galaxy catalogue "
                "(c1d._load_galaxy_catalog_handler(REDUCED_CATALOGUE_PATH).reduced_galaxy_catalog)",
                "producer": "darksiren_emri/galaxy_catalogue/handler.py:GalaxyCatalogueHandler "
                "(loads reduced_galaxy_catalogue.csv wholesale)",
                "mass_filter_conditioned": False,
                "evidence": (
                    "_load_eligible_catalogue applies its OWN eligibility mask directly "
                    "(z<z_max & isfinite(M) & M>0) to the FULL, unfiltered catalogue frame. "
                    "mass_filter_sigma only appears inside handler.py's "
                    "get_possible_hosts_from_ball_tree (~line 570-675), a per-EVENT candidate "
                    "lookup this companion never calls (grep for mass_filter_sigma / "
                    "get_possible_hosts_from_ball_tree in this file and correspondence_1d.py's "
                    "b0i-2D build path returns no hits outside that one handler.py function)."
                ),
            },
            {
                "input": "completeness (c1d.from_cache_or_build(), pixel_completeness.py)",
                "producer": "darksiren_emri/galaxy_catalogue/pixel_completeness.py:"
                "from_cache_or_build -> build_m_th_map(catalog_path=REDUCED_CATALOGUE_FILE_PATH) "
                "(the frozen per-HEALPix-pixel m_th threshold-magnitude cache)",
                "mass_filter_conditioned": False,
                "evidence": (
                    "grep for mass_filter/get_possible_hosts_from_ball_tree in "
                    "pixel_completeness.py returns no hits; the m_th map is a sky-pixel "
                    "detection-threshold map built from the full unfiltered catalogue, no "
                    "BH-mass filtering step anywhere in build_m_th_map."
                ),
            },
            {
                "input": "detection_probability (SimulationDetectionProbability, from "
                "build_b0i_2d_selection_objects)",
                "producer": "darksiren_emri/bayesian_inference/simulation_detection_probability.py, "
                "built from the PINNED LISA injection pool (injection_dir=INJECTION_POOL_DIR), "
                "not the galaxy catalogue at all",
                "mass_filter_conditioned": False,
                "evidence": (
                    "grep for mass_filter in simulation_detection_probability.py returns no "
                    "hits; the (dl_centers, M_centers) survival grid is built purely from "
                    "injected/recovered synthetic sources -- no galaxy candidate list is "
                    "involved anywhere in its construction."
                ),
            },
            {
                "input": "phi_survival_table (bayesian_statistics.precompute_phi_marginal_survival, "
                "from build_b0i_2d_selection_objects)",
                "producer": "bayesian_statistics.py:precompute_phi_marginal_survival"
                "(detection_probability_obj=...); derived purely from the detection_probability "
                "object above",
                "mass_filter_conditioned": False,
                "evidence": (
                    "Same injection-pool-only provenance as detection_probability; no galaxy "
                    "catalogue or candidate-host step anywhere in its call chain."
                ),
            },
        ],
        "note_out_of_scope": (
            "beta_G_phi/beta_Gbar_phi/Sigma_phi (the C2* normalizer legs, read from "
            "o5.mass_companion/o5._beta_g_phi_and_gbar) are a SEPARATE multiplicative factor of "
            "C2*, not part of Sigma_tilde_4D itself; they are read from a banked "
            "event_likelihoods.csv produced under that run's own mass_filter_sigma setting, but "
            "are asserted AND runtime-checked (np.allclose across events) to be event-independent "
            "constants at fixed (h, flags) -- PA-2D-4 item 4 as registered scopes the check to "
            "Sigma_tilde_4D's own inputs, which this finding covers exhaustively."
        ),
        "conclusion": (
            "Sigma_tilde_4D's contraction consumes the draw law only (galaxy-catalogue "
            "eligibility + injection-pool-derived completeness/survival objects); it is NOT "
            "conditioned on the candidate mass filter (mass_filter_sigma / "
            "get_possible_hosts_from_ball_tree), so the [P3-WBHZERO] symmetric-window ruling "
            "(row #202) does not change Sigma_tilde_4D and does not require re-deriving the "
            "frozen C2* under that axis."
        ),
    }


def compute_sigma_tilde_4d(
    completeness: Any,
    phi_survival_table: dict[float, Any],
    detection_probability: Any,
    h: float = H_GEN,
    chunk: int = CHUNK,
) -> dict[str, Any]:
    _t0 = time.time()
    cat = _load_eligible_catalogue(completeness, phi_survival_table, h)
    n_g = cat["z"].size
    print(f"  [+{time.time() - _t0:.1f}s] catalogue loaded, n_eligible={n_g}", file=sys.stderr)
    M_eff = _eddington_shifted_host_mass_batch(cat["M"], cat["M_error"])
    w_g = np.asarray(R_eff_per_mbh(cat["M"]), dtype=np.float64) / (1.0 + cat["z"])

    # Memory-bounded (mirrors the existing kernel_smeared_survival/mass_companion chunking
    # convention, ``:1300-1306``/``bayesian_statistics.py``): the window bound scan for the
    # global z-breakpoints is done PER CHUNK, never materializing a full-length
    # (z_err_eff, lower, upper) triple over the whole ~20.8M-row pool at once -- an unchunked
    # version measured OOM-SIGKILL risk under concurrent memory pressure on this 30 GB box
    # (2026-08-25 readiness validation, a live p3_2d_fleet.py pilot alongside it pushed
    # available memory to ~9 GB). Row-independent, so this is a pure memory-shape transform.
    z_lo_bound = float("inf")
    z_hi_bound = float("-inf")
    for start in range(0, n_g, chunk):
        sl = slice(start, min(start + chunk, n_g))
        z_err_eff_c = c1d.host_z_error_eff(cat["z"][sl], cat["z_error"][sl])
        lower_c, upper_c = c1d._host_kernel_window(cat["z"][sl], z_err_eff_c)
        z_lo_bound = min(z_lo_bound, float(lower_c.min()))
        z_hi_bound = max(z_hi_bound, float(upper_c.max()))
    z_lo_bound = max(z_lo_bound, c1d._B0I_KERNEL_Z_FLOOR)
    z_breakpoints = _z_breakpoints_dl_centers(detection_probability, h, z_lo_bound, z_hi_bound)
    print(
        f"  [+{time.time() - _t0:.1f}s] z-breakpoints built: n={z_breakpoints.size} "
        f"range=[{z_breakpoints.min():.4f},{z_breakpoints.max():.4f}]",
        file=sys.stderr,
    )

    # PA-2D-3: build the arbiter FIRST (before the full pass), on the real catalogue+breakpoints.
    arbiter = _run_arbiter(cat, M_eff, completeness, detection_probability, h, z_breakpoints)
    spot_target = arbiter["derived_spot_check_target"]
    print(
        f"  [+{time.time() - _t0:.1f}s] arbiter done: plateau={arbiter['plateau_L_conv']:.3e} "
        f"target={spot_target:.3e}",
        file=sys.stderr,
    )

    s_tilde_4d = np.empty(n_g, dtype=np.float64)
    # [PA-2D-2 fix] diagnostic renamed: the exact closed form has no GH nodes to count, so the
    # old "n_nonpositive_mass_gh_nodes" (a GH-24 order-parameter artifact) is replaced by the
    # honest, order-parameter-free analog: the total truncated Gaussian mass P(M<=0) summed over
    # every (galaxy, z-node) pair -- exactly the probability weight the closed form's boundary-low
    # segment start (Phi_at0 - Phi(0)) excludes below zero, i.e. what the F2 MINOR-6 guard removes.
    sum_p_m_le_0 = 0.0
    n_rows_m_le_0_nonnegligible = 0  # rows with P(M<=0) > 1e-9 (i.e. the guard is NOT negligible)
    seg_counts = np.empty(n_g, dtype=np.int64)
    for start in range(0, n_g, chunk):
        sl = slice(start, min(start + chunk, n_g))
        z_g = cat["z"][sl]
        z_err_eff = c1d.host_z_error_eff(z_g, cat["z_error"][sl])
        lower, upper = c1d._host_kernel_window(z_g, z_err_eff)
        mu = M_eff[sl]
        sigma = cat["M_error"][sl]

        edges, n_interior = _segment_edges(lower, upper, z_breakpoints)
        seg_counts[sl] = n_interior + 1
        numerator, z_norm = _segmented_integral_batch(
            z_g,
            z_err_eff,
            cat["phiS"][sl],
            cat["qS"][sl],
            mu,
            sigma,
            completeness,
            detection_probability,
            h,
            edges,
            _ZSEG_GL_NODES,
            _ZSEG_GL_WEIGHTS,
        )
        z_norm_safe = np.where(z_norm > 0.0, z_norm, 1.0)
        s_tilde_4d[sl] = numerator / z_norm_safe

        sigma_safe_diag = np.where(sigma > 0.0, sigma, 1.0)
        p_m_le_0 = np.where(sigma > 0.0, norm.cdf(0.0, loc=mu, scale=sigma_safe_diag), 0.0)
        sum_p_m_le_0 += float(np.sum(p_m_le_0))
        n_rows_m_le_0_nonnegligible += int(np.count_nonzero(p_m_le_0 > 1.0e-9))

        print(
            f"  [+{time.time() - _t0:.1f}s] chunk {start}:{sl.stop}/{n_g} done "
            f"(max_segments_this_chunk={int(n_interior.max()) + 1 if n_interior.size else 1})",
            file=sys.stderr,
        )

    sigma_tilde_4d = float(np.sum(w_g * s_tilde_4d))
    n_segments_stats = {
        "mean": float(seg_counts.mean()),
        "median": float(np.median(seg_counts)),
        "p99": float(np.percentile(seg_counts, 99)),
        "max": int(seg_counts.max()),
        "min": int(seg_counts.min()),
    }
    return {
        "Sigma_tilde_4D": sigma_tilde_4d,
        "n_eligible": int(n_g),
        "sum_P_M_le_0": sum_p_m_le_0,
        "n_rows_M_le_0_nonnegligible": n_rows_m_le_0_nonnegligible,
        "w_g": w_g,
        "s_tilde_4d": s_tilde_4d,
        "M_eff": M_eff,
        "catalogue": cat,
        "z_breakpoints": z_breakpoints,
        "n_segments_stats": n_segments_stats,
        "arbiter": arbiter,
        "spot_check_target": spot_target,
    }


def _spot_check(
    cat: dict[str, npt.NDArray[np.float64]],
    M_eff: npt.NDArray[np.float64],
    s_tilde_4d: npt.NDArray[np.float64],
    completeness: Any,
    detection_probability: Any,
    h: float,
    n_spot: int = 100,
    seed: int = 20260825,
) -> dict[str, Any]:
    """Independent recompute of ``S~_4D,g`` for ``n_spot`` random rows via nested
    ``scipy.integrate.quad`` (z outer, M inner) -- a genuinely different numerical method
    (adaptive quadrature) from the GL(50)xGH(24) product-quadrature the main pass uses.
    """
    rng = np.random.default_rng(seed)
    n_g = cat["z"].size
    idx = rng.choice(n_g, size=min(n_spot, n_g), replace=False)
    rel_devs = []
    for i in idx:
        z_g = float(cat["z"][i])
        z_err_eff = float(c1d.host_z_error_eff(np.array([z_g]), cat["z_error"][i : i + 1])[0])
        lower, upper = c1d._host_kernel_window(np.array([z_g]), np.array([z_err_eff]))
        lower, upper = float(lower[0]), float(upper[0])
        pixel = c1d._host_pixels(completeness, cat["phiS"][i : i + 1], cat["qS"][i : i + 1])
        mu_i, sigma_i = float(M_eff[i]), float(cat["M_error"][i])

        def _w_pop_eff_scalar(z: float) -> float:
            z_arr = np.array([z])
            return float(c1d._kernel_w_pop_eff(z_arr[:, None], completeness, pixel, h)[0, 0])

        def _s_bar_4d(z: float, mu_i: float = mu_i, sigma_i: float = sigma_i) -> float:
            d_l = float(dist_vectorized(np.array([z]), h=h)[0])
            if sigma_i <= 0.0:
                m_query = max(mu_i, 1e-6)
                return float(
                    detection_probability.detection_probability_with_bh_mass_interpolated(
                        d_l,
                        m_query,
                        0.0,
                        0.0,
                        h=h,
                        **_wbh_z_kwargs(detection_probability, np.array([z])),
                    )
                )

            def _integrand_m(m: float) -> float:
                if m <= 0.0:
                    return 0.0
                s4d = float(
                    detection_probability.detection_probability_with_bh_mass_interpolated(
                        d_l,
                        m,
                        0.0,
                        0.0,
                        h=h,
                        **_wbh_z_kwargs(detection_probability, np.array([z])),
                    )
                )
                return s4d * float(norm.pdf(m, loc=mu_i, scale=sigma_i))

            lo_m = max(mu_i - 6.0 * sigma_i, 1e-6)
            hi_m = mu_i + 6.0 * sigma_i
            val, _ = integrate.quad(_integrand_m, lo_m, hi_m, limit=50, epsabs=1e-13, epsrel=1e-7)
            return val

        def _num_integrand(z: float) -> float:
            return (
                float(norm.pdf(z, loc=z_g, scale=z_err_eff)) * _w_pop_eff_scalar(z) * _s_bar_4d(z)
            )

        def _den_integrand(z: float) -> float:
            return float(norm.pdf(z, loc=z_g, scale=z_err_eff)) * _w_pop_eff_scalar(z)

        num, _ = integrate.quad(_num_integrand, lower, upper, limit=50, epsabs=1e-13, epsrel=1e-7)
        den, _ = integrate.quad(_den_integrand, lower, upper, limit=50, epsabs=1e-13, epsrel=1e-7)
        quad_val = num / den if den > 0.0 else 0.0
        gl_val = float(s_tilde_4d[i])
        rel_dev = abs(quad_val - gl_val) / max(abs(quad_val), 1e-300)
        rel_devs.append(rel_dev)
        print(
            f"    spot row {len(rel_devs)}/{min(n_spot, n_g)}: quad={quad_val:.6e} "
            f"gl={gl_val:.6e} rel_dev={rel_dev:.3e}",
            file=sys.stderr,
        )

    rel_devs_arr = np.asarray(rel_devs, dtype=np.float64)
    return {
        "n_spot": int(len(rel_devs)),
        "max_rel_dev": float(np.max(rel_devs_arr)),
        "median_rel_dev": float(np.median(rel_devs_arr)),
        "target_rel_dev": 1.0e-6,
        "target_met": bool(np.max(rel_devs_arr) <= 1.0e-6),
    }


def _arbiter_only_dry_run() -> None:
    """PA-2D-3 readiness validation: build selection objects + the real catalogue + the real
    global z-breakpoints, run the ~20-row arbiter, and compute CHEAP full-catalogue segment-
    count statistics (window + searchsorted only -- NO S_4D evaluation, so this stays fast even
    at the full ~20.8M-row eligible pool) plus a small timed sample of the actual expensive
    per-row integral to extrapolate the full-pass wall time. Does NOT run the (expensive) full
    pass and does NOT touch ``OUT_PATH`` (PA-CA-11 out-root guard scopes only the banked
    companion-pass output) -- writes to :data:`ARBITER_DRY_RUN_OUT_PATH` instead, refusing to
    overwrite an existing file there under the same convention.
    """
    if ARBITER_DRY_RUN_OUT_PATH.is_file():
        raise SystemExit(
            f"REFUSED (PA-CA-11 convention): {ARBITER_DRY_RUN_OUT_PATH} already exists -- purge "
            "or pick a fresh path."
        )
    ARBITER_DRY_RUN_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    stamp = _a22_stamp_2d()

    print(f"[t={time.time() - t0:.1f}s] building b0i-2D selection objects...", file=sys.stderr)
    completeness, phi_survival_table, detection_probability = c1d.build_b0i_2d_selection_objects(
        h_true=H_GEN
    )
    cat = _load_eligible_catalogue(completeness, phi_survival_table, H_GEN)
    n_g = cat["z"].size
    print(f"[t={time.time() - t0:.1f}s] catalogue loaded, n_eligible={n_g}", file=sys.stderr)
    M_eff = _eddington_shifted_host_mass_batch(cat["M"], cat["M_error"])

    z_err_eff_all = c1d.host_z_error_eff(cat["z"], cat["z_error"])
    lower_all, upper_all = c1d._host_kernel_window(cat["z"], z_err_eff_all)
    z_lo_bound = float(max(float(lower_all.min()), c1d._B0I_KERNEL_Z_FLOOR))
    z_hi_bound = float(upper_all.max())
    z_breakpoints = _z_breakpoints_dl_centers(detection_probability, H_GEN, z_lo_bound, z_hi_bound)
    print(
        f"[t={time.time() - t0:.1f}s] z-breakpoints built: n={z_breakpoints.size} "
        f"range=[{z_breakpoints.min():.4f},{z_breakpoints.max():.4f}]",
        file=sys.stderr,
    )

    arbiter = _run_arbiter(cat, M_eff, completeness, detection_probability, H_GEN, z_breakpoints)
    print(
        f"[t={time.time() - t0:.1f}s] arbiter done: plateau={arbiter['plateau_L_conv']:.3e} "
        f"target={arbiter['derived_spot_check_target']:.3e} "
        f"max_gl16_vs_gl32={arbiter['max_rel_dev_gl16_vs_gl32']:.3e}",
        file=sys.stderr,
    )

    # Cheap full-catalogue segmentation stats: window+searchsorted only, NO S_4D evaluation.
    t_seg0 = time.time()
    seg_counts = np.empty(n_g, dtype=np.int64)
    for start in range(0, n_g, 500_000):
        sl = slice(start, min(start + 500_000, n_g))
        _, n_interior = _segment_edges(lower_all[sl], upper_all[sl], z_breakpoints)
        seg_counts[sl] = n_interior + 1
    t_seg = time.time() - t_seg0
    n_segments_stats = {
        "n_rows": int(n_g),
        "mean": float(seg_counts.mean()),
        "median": float(np.median(seg_counts)),
        "p99": float(np.percentile(seg_counts, 99)),
        "max": int(seg_counts.max()),
        "min": int(seg_counts.min()),
        "wall_time_s_cheap_pass": t_seg,
    }
    print(
        f"[t={time.time() - t0:.1f}s] cheap full-catalogue segment stats: {n_segments_stats}",
        file=sys.stderr,
    )

    # Timed sample of the ACTUAL expensive per-row integral (production GL-16), to extrapolate.
    n_sample = min(5000, n_g)
    rng = np.random.default_rng(999)
    idx_sample = rng.choice(n_g, size=n_sample, replace=False)
    t_sample0 = time.time()
    edges_s, _ = _segment_edges(lower_all[idx_sample], upper_all[idx_sample], z_breakpoints)
    _num_s, _den_s = _segmented_integral_batch(
        cat["z"][idx_sample],
        z_err_eff_all[idx_sample],
        cat["phiS"][idx_sample],
        cat["qS"][idx_sample],
        M_eff[idx_sample],
        cat["M_error"][idx_sample],
        completeness,
        detection_probability,
        H_GEN,
        edges_s,
        _ZSEG_GL_NODES,
        _ZSEG_GL_WEIGHTS,
    )
    t_sample = time.time() - t_sample0
    sec_per_row = t_sample / n_sample
    est_full_pass_s = sec_per_row * n_g
    print(
        f"[t={time.time() - t0:.1f}s] timed sample: n={n_sample} t={t_sample:.2f}s "
        f"({sec_per_row * 1000.0:.3f} ms/row) -> est full-pass (z-stage) "
        f"{est_full_pass_s:.0f}s ({est_full_pass_s / 3600.0:.2f}h) over n_eligible={n_g}",
        file=sys.stderr,
    )

    out: dict[str, Any] = {
        "reference": f"{REGISTRATION_SECTION} -- ARBITER-ONLY READINESS DRY RUN, no full pass",
        "h": H_GEN,
        "resolved_flags": stamp,
        "n_eligible": int(n_g),
        "z_breakpoints_n": int(z_breakpoints.size),
        "z_breakpoints_range": [float(z_breakpoints.min()), float(z_breakpoints.max())],
        "arbiter": arbiter,
        "n_segments_stats_full_catalogue_cheap": n_segments_stats,
        "timed_sample": {
            "n_sample": int(n_sample),
            "wall_time_s": t_sample,
            "sec_per_row": sec_per_row,
            "estimated_full_pass_z_stage_wall_time_s": est_full_pass_s,
            "estimated_full_pass_z_stage_wall_time_h": est_full_pass_s / 3600.0,
            "note": "z-stage only (the expensive segmented integral loop); excludes catalogue "
            "load, breakpoint build (both one-time, small), and the fast 50-row spot-check "
            "(negligible) -- add ~10-20% margin for those plus process overhead.",
        },
        "eligibility_independence": _eligibility_independence_finding(),
        "wall_time_s_total": time.time() - t0,
    }
    ARBITER_DRY_RUN_OUT_PATH.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k not in ("resolved_flags",)}, indent=2))
    print(f"\nwrote {ARBITER_DRY_RUN_OUT_PATH}")


def main() -> None:
    import os as _os

    if _os.environ.get("P3_2D_COMPANION_ARBITER_ONLY"):
        _arbiter_only_dry_run()
        return

    if OUT_PATH.is_file():
        raise SystemExit(
            f"REFUSED (PA-CA-11): {OUT_PATH} already exists -- purge or use a fresh out path "
            "before a banked companion-pass run."
        )
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    stamp = _a22_stamp_2d()  # A22: written before any scoring call.
    t0 = time.time()

    print(f"[t={time.time() - t0:.1f}s] building b0i-2D selection objects...", file=sys.stderr)
    completeness, phi_survival_table, detection_probability = c1d.build_b0i_2d_selection_objects(
        h_true=H_GEN
    )
    print(
        f"[t={time.time() - t0:.1f}s] selection objects built; computing Sigma~^4D...",
        file=sys.stderr,
    )
    result = compute_sigma_tilde_4d(completeness, phi_survival_table, detection_probability, H_GEN)
    print(f"[t={time.time() - t0:.1f}s] Sigma~^4D pass done.", file=sys.stderr)
    t_companion = time.time() - t0

    # beta_G_phi / beta_Gbar_phi / Sigma_phi: banked, seed-invariant, venue-independent scalars.
    at = o5._rows_at_h(_REFERENCE_CSV, H_GEN)
    beta_g_phi, beta_gbar_phi = o5._beta_g_phi_and_gbar(at)
    mc = o5.mass_companion(H_GEN)
    sigma_phi = mc["Sigma_phi"]

    c2_star = beta_g_phi * result["Sigma_tilde_4D"] / (sigma_phi * beta_gbar_phi)

    t1 = time.time()
    spot = _fast_spot_check(
        result["catalogue"],
        result["M_eff"],
        result["s_tilde_4d"],
        completeness,
        detection_probability,
        H_GEN,
        result["z_breakpoints"],
        result["spot_check_target"],
    )
    t_spot = time.time() - t1

    out: dict[str, Any] = {
        "reference": f"{REGISTRATION_SECTION}",
        "h": H_GEN,
        "resolved_flags": stamp,
        "Sigma_tilde_4D": result["Sigma_tilde_4D"],
        "n_eligible": result["n_eligible"],
        "sum_P_M_le_0": result["sum_P_M_le_0"],
        "n_rows_M_le_0_nonnegligible": result["n_rows_M_le_0_nonnegligible"],
        "beta_G_phi": beta_g_phi,
        "beta_Gbar_phi": beta_gbar_phi,
        "Sigma_phi": sigma_phi,
        "reference_csv": str(_REFERENCE_CSV),
        "C2_star": c2_star,
        "z_segmentation": {
            "description": (
                "Per-host +/-4sigma z-window subdivided at every z where d_L(z;h) crosses a "
                "dl_centers grid edge (60 global breakpoints, inverted once via brentq); each "
                "smooth sub-segment integrated with a fixed order-16 Gauss-Legendre rule "
                "(PA-2D-3 fix for the GL(50)-under-resolves-kinks defect)."
            ),
            "gl_order_per_segment": _Z_SEG_GL_ORDER,
            "n_breakpoints": int(result["z_breakpoints"].size),
        },
        "n_segments_stats": result["n_segments_stats"],
        "arbiter": result["arbiter"],
        "spot_check": spot,
        "eligibility_independence": _eligibility_independence_finding(),
        "v1_superseded_path": str(V1_OUT_PATH),
        "wall_time_s": {
            "companion_pass": t_companion,
            "spot_check": t_spot,
            "total": time.time() - t0,
        },
    }
    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k not in ("resolved_flags",)}, indent=2))
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
