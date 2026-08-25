r"""[P3-2D] the Sigma~^4D companion pass (zero-``evaluate()``), item 1 of the pre-fleet
execution task (PREREGISTRATION_P3_2D_20260825.md, PA-2D-1 F4).

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
import pandas as pd
from scipy import integrate
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

OUT_PATH = THIS_DIR / "ca_rhs_work2d/p3_2d_companion.json"

REGISTRATION_SECTION = (
    "results/campaign51_20260728/realistic_20260729/PREREGISTRATION_P3_2D_20260825.md "
    "sec1 (Sigma~^4D companion), PA-2D-1 F4 (C2* resolved)"
)


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

    s_tilde_4d = np.empty(n_g, dtype=np.float64)
    # [PA-2D-2 fix] diagnostic renamed: the exact closed form has no GH nodes to count, so the
    # old "n_nonpositive_mass_gh_nodes" (a GH-24 order-parameter artifact) is replaced by the
    # honest, order-parameter-free analog: the total truncated Gaussian mass P(M<=0) summed over
    # every (galaxy, z-node) pair -- exactly the probability weight the closed form's boundary-low
    # segment start (Phi_at0 - Phi(0)) excludes below zero, i.e. what the F2 MINOR-6 guard removes.
    sum_p_m_le_0 = 0.0
    n_rows_m_le_0_nonnegligible = 0  # rows with P(M<=0) > 1e-9 (i.e. the guard is NOT negligible)
    for start in range(0, n_g, chunk):
        sl = slice(start, min(start + chunk, n_g))
        z_g = cat["z"][sl]
        z_err_eff = c1d.host_z_error_eff(z_g, cat["z_error"][sl])
        lower, upper = c1d._host_kernel_window(z_g, z_err_eff)
        half = 0.5 * (upper - lower)
        mid = 0.5 * (upper + lower)
        z_nodes = mid[:, None] + half[:, None] * c1d._GL_NODES_B0I[None, :]  # (m, 50)
        host_pixels = c1d._host_pixels(completeness, cat["phiS"][sl], cat["qS"][sl])
        w_pop_eff = c1d._kernel_w_pop_eff(z_nodes, completeness, host_pixels, h)
        gaussian_vals = norm.pdf(z_nodes, loc=z_g[:, None], scale=z_err_eff[:, None])
        kernel_unnorm = gaussian_vals * w_pop_eff  # (m, 50)

        d_L_nodes = np.asarray(dist_vectorized(z_nodes.ravel(), h=h), dtype=np.float64).reshape(
            z_nodes.shape
        )
        mu = np.broadcast_to(M_eff[sl][:, None], z_nodes.shape)
        sigma = np.broadcast_to(cat["M_error"][sl][:, None], z_nodes.shape)
        s_bar_4d_nodes = _mass_marginal_survival(
            mu, sigma, z_nodes, d_L_nodes, cat["phiS"][sl], cat["qS"][sl], detection_probability, h
        )
        sigma_safe_diag = np.where(sigma > 0.0, sigma, 1.0)
        p_m_le_0 = np.where(sigma > 0.0, norm.cdf(0.0, loc=mu, scale=sigma_safe_diag), 0.0)
        sum_p_m_le_0 += float(np.sum(p_m_le_0))
        n_rows_m_le_0_nonnegligible += int(np.count_nonzero(p_m_le_0 > 1.0e-9))

        numerator = (
            np.sum(kernel_unnorm * s_bar_4d_nodes * c1d._GL_WEIGHTS_B0I[None, :], axis=1) * half
        )
        z_norm = np.sum(kernel_unnorm * c1d._GL_WEIGHTS_B0I[None, :], axis=1) * half
        z_norm = np.where(z_norm > 0.0, z_norm, 1.0)
        s_tilde_4d[sl] = numerator / z_norm
        print(f"  [+{time.time() - _t0:.1f}s] chunk {start}:{sl.stop}/{n_g} done", file=sys.stderr)

    sigma_tilde_4d = float(np.sum(w_g * s_tilde_4d))
    return {
        "Sigma_tilde_4D": sigma_tilde_4d,
        "n_eligible": int(n_g),
        "sum_P_M_le_0": sum_p_m_le_0,
        "n_rows_M_le_0_nonnegligible": n_rows_m_le_0_nonnegligible,
        "w_g": w_g,
        "s_tilde_4d": s_tilde_4d,
        "M_eff": M_eff,
        "catalogue": cat,
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


def main() -> None:
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
    import os as _os

    n_spot = int(_os.environ.get("P3_2D_COMPANION_N_SPOT", "100"))
    spot = _spot_check(
        result["catalogue"],
        result["M_eff"],
        result["s_tilde_4d"],
        completeness,
        detection_probability,
        H_GEN,
        n_spot=n_spot,
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
        "spot_check": spot,
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
