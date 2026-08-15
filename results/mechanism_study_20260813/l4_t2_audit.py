"""L4-T2 — numeric audit of the D1-D6 diff ledger (author-approved, ledger row #108).

Follow-on to ``L4_DER_PART1_20260815.md`` (the generative model + correct likelihood +
enumerated coded-vs-correct diff, ``D1``-``D6``, commit ``5e77e196``). That document is
**PRESENTED, NOT ADJUDICATED**: every sign/magnitude in its diff table (§3) is a
prediction the author explicitly deferred to this numeric audit (§4, the registered
protocol) before any conclusion is drawn. This script computes -- it does not
adjudicate.

Computes, on the pinned 982-event population (``build_venue_context`` on the MN0X
cell's committed config -- same CRB CSV / frozeng emit / pruned catalogue / injection
pool pins already checked by V-T3):

1. Per-term tilt at truth (nats/h), each restated or numerically verified:
   - D1 (alpha): the registered analytic 1.036*N/h, PLUS a numeric verification of the
     1.036 exponent from the actual ``log_alpha(h)`` table the estimator builds
     (``build_context``'s alpha(h) = INTEGRAL w_pop*S_phi dz).
   - D2 (missing density prefactor): the naive +N/h, PLUS the z*-tracking-corrected
     value -- the population's per-event integrand peak z*(h) is re-found at each
     h-grid neighbour (not assumed fixed at z_true) and the prefactor's own h-tilt is
     measured along that moving peak. Both exact-z (kernel -> delta) and GLADE-mix
     smeared (kernel width = the event's representative decile sigma_z) readings are
     computed; each is further split into with/without p_pop in the peak-defining
     integrand (ambiguous per Part 1's wording -- both reported, never picked
     silently).
   - D3 (exponent scale mu vs d_obs): T_D3 computed at exact-z (host z_true, kernel ->
     delta) and at the GLADE-mix sigma_z-kernel-smeared population, via vectorized
     Gauss-Legendre quadrature (matching the estimator's own +-5 sigma_z window,
     50-node rule) over the SAME two terms with everything else (kernel, prefactor)
     held fixed between the "correct" and "coded" exponent forms -- isolating exactly
     the exponent-scale diff. Dose trend at f in {0.25, 0.5, 1.0} (dose scales the
     kernel width the same way ``_apply_dose_mask`` scales sigma_z on the instrument).
   - D4 (p_pop / M1 quadratic account): restated verbatim from the M6R adversarial
     verifier's addendum (`M6R_L0_NOTE_20260815.md`, "Four corrections of record" item
     3): predicted -111/-384/-990 nats/h at f_i = 0.25/0.5/1.0, ALREADY flagged there as
     refuted against measured T_res. Restated here, not recomputed.
   - D5: restated (REN -0.0019 in MAP-bias units b; M3 <= 1e-6), converted to a nats/h
     tilt via AJREN's own local-quadratic curvature Abar (T ~= bias * Abar), the same
     displacement-law geometry L4-T1 already used out-of-sample.
2. Ledger closure: predicted T(MN0X)/T(AM2P)/T(AJREN) assembled from the D-terms plus
   the two installed repairs (AM2P: J = -N/h + the verifier's D'-tracking piece +291
   nats/h; AJREN: J plus the MEASURED instrument REN tilt, since the REN repair's own
   W_k-division mechanics are not among the D1-D6 base-estimator diff terms -- Part 1
   is explicit that D5 describes the BASELINE window truncation, not the AJREN repair).
   Compared against measured T(MN0X)=2624.9+-18.8, T(AM2P)=1492.0+-30.7 (M6R table,
   verbatim), T(AJREN) read from L4_T1_output.json (L4-T1's own measurement, not
   recomputed here) -- each a pull in sigma.
3. Dose-structure test: does D3+D4's predicted T_res(f) reproduce the measured
   T_res(f) = +699/+149/-62 nats/h (M6R verifier addendum, pooled-f_h residual beyond
   alpha+full-dose-J) within 3x the level SEs (150 nats/h, as specified)?
4. 2D-channel estimate: a rough proxy for whether D3 interacts with the g_i
   (mass-channel) factor to produce a 2D-only excess of the observed sub-additivity
   order (+0.0027) -- event-weighted T_D3 (weight ~ 1/sigma_Mz, a crude sharpness
   proxy for where the mass channel is informative) vs the unweighted T_D3, the
   weighted-minus-unweighted difference converted to bias units via /Abar.

Output: ``L4_T2_output.json`` next to this file. Report: ``L4_T2_AUDIT_20260815.md``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.special import roots_legendre

from darksiren_emri.physical_relations import dist_vectorized
from darksiren_emri.validation import closed_loop_gfrac as cl
from darksiren_emri.validation import venue_transfer as vt

RESULTS_DIR = Path(__file__).parent

# ---- registered/reused constants (verbatim from prior committed notes) ----
ALPHA_N = 982
ALPHA_H = 0.730
ALPHA_SLOPE_COEFF = 1.036
ALPHA_TILT_NATS_PER_H = ALPHA_SLOPE_COEFF * ALPHA_N / ALPHA_H  # +1393.63 nats/h

D2_NAIVE_NATS_PER_H = ALPHA_N / ALPHA_H  # +1345.21 nats/h

# Measured references (M6R_L0_output.json / M6R_L0_NOTE_20260815.md §1, verbatim).
T_MN0X_MEAN, T_MN0X_SE = 2624.9, 18.8
T_AM2P_MEAN, T_AM2P_SE = 1492.0, 30.7

# J-restoration repair pieces (PREREGISTRATION_M2PRIME_ABLATION.md §2 + the M6R
# addendum item 2 "on-prediction" correction), restated, not recomputed.
J_FULL_DOSE_NATS_PER_H = -D2_NAIVE_NATS_PER_H  # -1345.21, the -N/h piece
J_DPRIME_FULL_TRACKING = 291.0  # verifier's D'-tracking piece, full tracking
J_DPRIME_ATTENUATED = 254.0  # same, with kernel/GW-width attenuation

# D4 -- M6R verifier addendum item 3, verbatim (predicted vs measured T_res).
D4_PREDICTED_NATS = {"0.25": -111.0, "0.5": -384.0, "1.0": -990.0}
D4_STATUS = (
    "REFUTED quantitatively by the M6R adversarial verifier (addendum item 3): wrong "
    "sign at the two lower doses, ~16x too negative at full dose, inverted shape "
    "(measured T_res decelerates -550 -> -212 across sub-levels the note tracked; the "
    "quadratic account accelerates). Restated here as the registered D4 value, not "
    "re-derived, and not treated as validated."
)

# Measured T_res(f) -- M6R verifier addendum item 3, verbatim (pooled over f_h,
# residual beyond alpha + full-dose-predicted-J at f_i = 0.25/0.5/1.0).
T_RES_MEASURED = {"0.25": 699.0, "0.5": 149.0, "1.0": -62.0}
T_RES_TOLERANCE_NATS = 150.0  # 3x the level SEs, as specified in the audit protocol

# D5 -- restated measured values (bias units b; NOT nats/h -- converted below).
D5_REN_BIAS = -0.0019
D5_M3_BIAS_CEILING = 1e-6

STAGE3_2D_SUBADDITIVE_NATS = 0.0027  # STAGE3_READOUT.md §2, restated for context

_KERNEL_WINDOW_SIGMA = 5.0  # matches calibration_gate._IMPOSTOR_KERNEL_WINDOW
_N_QUAD = 50  # matches the production per-candidate GL order
_H_STEP_LABEL = "grid-neighbour central difference (h=0.725, 0.735; Delta h=0.01)"


def _load_json(path: Path) -> dict[str, Any]:
    with open(path) as fh:
        result: dict[str, Any] = json.load(fh)
    return result


# ---------------------------------------------------------------------------
# Context: the pinned 982-event population, via the production loader
# ---------------------------------------------------------------------------


def build_population_context() -> tuple[vt.VenueContext, int, int]:
    """Build the venue context from MN0X's committed config (V-T3 pin-checked).

    Returns:
        ``(vctx, i_lo, i_hi)`` -- the venue context and the h-grid neighbour
        indices bracketing ``h_true`` (0.725, 0.735).
    """
    cfg = _load_json(RESULTS_DIR / "MN0X_h0p730_results_seeds0_100.json")["config"]
    vcfg = vt.VenueConfig(
        cell=cfg["cell"],
        h_true=cfg["h_true"],
        balls=cfg["balls"],
        sigma_mode=cfg["sigma_mode"],
        flat_sigma_z=cfg["flat_sigma_z"],
        lambda_poisson=cfg["lambda_poisson"],
        dose_target=cfg["dose_target"],
        dose_scales=cfg["dose_scales"],
        crb_reference_csv=cfg["crb_reference_csv"],
        frozeng_emit_json=cfg["frozeng_emit_json"],
        pruned_catalogue_csv=cfg["pruned_catalogue_csv"],
        injection_data_dir=cfg["injection_data_dir"],
        n_events_cap=cfg["n_events_cap"],
        chunk_pairs=cfg["chunk_pairs"],
        h_grid=cfg["h_grid"],
    )
    vctx = vt.build_venue_context(vcfg)
    h_grid = np.asarray(vctx.gctx.cl_ctx.config.h_grid, dtype=np.float64)
    h_true = float(vctx.gctx.cl_ctx.config.h_true)
    i_true = int(np.argmin(np.abs(h_grid - h_true)))
    i_lo, i_hi = i_true - 1, i_true + 1
    assert abs(h_grid[i_lo] - 0.725) < 1e-9 and abs(h_grid[i_hi] - 0.735) < 1e-9, (
        f"unexpected h-grid neighbours {h_grid[i_lo]}, {h_grid[i_hi]}"
    )
    return vctx, i_lo, i_hi


def representative_sigma_z(vctx: vt.VenueContext) -> npt.NDArray[np.float64]:
    """Per-event representative (decile-median) GLADE sigma_z, deterministic.

    Uses the SAME z-decile sampler tables the venue's stochastic per-seed draw
    uses (``build_sigma_sampler``), but takes each decile's MEDIAN sigma_z
    instead of a random draw -- a population-level (not per-seed) "GLADE mix"
    representative value, matching the audit's "at truth" (no noise-draw)
    convention used throughout this ledger.
    """
    edges = vctx.z_decile_edges
    pools = vctx.sigma_pool_deciles
    dec = np.searchsorted(edges, vctx.z_true, side="right")
    dec_median = np.array([float(np.median(p)) for p in pools], dtype=np.float64)
    return dec_median[dec]


# ---------------------------------------------------------------------------
# D1 -- alpha term: restate + numeric exponent verification
# ---------------------------------------------------------------------------


def d1_alpha(vctx: vt.VenueContext, i_lo: int, i_hi: int, n: int) -> dict[str, Any]:
    h_grid = np.asarray(vctx.gctx.cl_ctx.config.h_grid, dtype=np.float64)
    log_alpha = np.asarray(vctx.gctx.cl_ctx.log_alpha, dtype=np.float64)
    dh = h_grid[i_hi] - h_grid[i_lo]
    dlnalpha_dh = (log_alpha[i_hi] - log_alpha[i_lo]) / dh
    h_true = float(vctx.gctx.cl_ctx.config.h_true)
    exponent_numeric = -h_true * dlnalpha_dh  # alpha(h) ~ h^{-exponent}
    tilt_numeric = -float(n) * dlnalpha_dh  # d/dh[-N ln alpha(h)]
    return {
        "analytic_tilt_nats_per_h": ALPHA_TILT_NATS_PER_H,
        "analytic_exponent": ALPHA_SLOPE_COEFF,
        "numeric_tilt_nats_per_h": tilt_numeric,
        "numeric_exponent_at_h_true": exponent_numeric,
        "exponent_relative_diff": (exponent_numeric - ALPHA_SLOPE_COEFF) / ALPHA_SLOPE_COEFF,
        "method": (
            "central difference of the estimator's OWN log_alpha(h) table "
            "(closed_loop_gfrac.build_context, alpha(h) = INTEGRAL w_pop*S_phi dz) "
            "at the h=0.725/0.735 grid neighbours; exponent read off "
            "alpha(h) ~ h^{-p} => p = -h*d(ln alpha)/dh at h_true."
        ),
    }


# ---------------------------------------------------------------------------
# D2 -- missing density prefactor: naive, exact-z, and z*-tracking-corrected
# ---------------------------------------------------------------------------


def _d2_exponent_terms(
    z: npt.NDArray[np.float64],
    h: float,
    d_true: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """mu(z,h) = D(z)/h, vectorized over a (n_events, n_z) grid."""
    mu = np.asarray(dist_vectorized(np.maximum(z, 1e-8), h=h), dtype=np.float64)
    return mu


def _gw_peak_width(
    z_true: npt.NDArray[np.float64], sigma_d: npt.NDArray[np.float64], h_true: float
) -> npt.NDArray[np.float64]:
    """Local characteristic width in z of the GW-density term alone.

    ``sigma_d / |d ln D/dz|`` at ``z_true`` -- the scale over which
    ``exp(exponent)`` falls by O(1) as z moves off z_true at h = h_true.
    Some events (tiny CRB distance error) have this width MANY orders of
    magnitude narrower than the sigma_z kernel window; without this floor a
    fixed kernel-only quadrature grid catastrophically under-resolves those
    events' GW peak (verified empirically: worst event has GW width ~1.9e-5
    in z vs a kernel window of O(0.05-0.25)) and injects pure node-placement
    noise into any correct-vs-coded comparison.
    """
    eps = 1e-6
    d_hi = np.asarray(dist_vectorized(z_true + eps, h=h_true), dtype=np.float64)
    d_lo = np.asarray(dist_vectorized(np.maximum(z_true - eps, 1e-8), h=h_true), dtype=np.float64)
    dlnD_dz = (np.log(d_hi) - np.log(d_lo)) / (2.0 * eps)
    return sigma_d / np.maximum(np.abs(dlnD_dz), 1e-8)


def _effective_half_width(
    z_true: npt.NDArray[np.float64],
    sigma_d: npt.NDArray[np.float64],
    sigma_rep: npt.NDArray[np.float64],
    h_true: float,
    n_sigma: float = 5.0,
) -> npt.NDArray[np.float64]:
    """Quadrature half-width resolving whichever of (kernel, GW peak) is narrower.

    Both the kernel (width ``sigma_rep``) and the GW-density term (local width
    ``_gw_peak_width``) are centered near ``z_true`` at ``h`` close to
    ``h_true`` (since ``d_true = D(z_true)/h_true`` by construction) -- taking
    the narrower of the two as the resolution scale, floored at a numerical
    minimum, keeps a fixed-node quadrature well-resolved for both regimes
    without changing the physical window when the kernel is the binding
    constraint (the ordinary case).
    """
    gw_width = _gw_peak_width(z_true, sigma_d, h_true)
    eff = np.minimum(sigma_rep, gw_width)
    eff = np.maximum(eff, 1e-6)
    return n_sigma * eff


def d2_prefactor(
    vctx: vt.VenueContext,
    i_lo: int,
    i_hi: int,
    sigma_rep: npt.NDArray[np.float64],
) -> dict[str, Any]:
    z_true = vctx.z_true
    d_true = vctx.d_L
    sigma_d = vctx.sigma_dL
    h_grid = np.asarray(vctx.gctx.cl_ctx.config.h_grid, dtype=np.float64)
    h_lo, h_hi = float(h_grid[i_lo]), float(h_grid[i_hi])
    dh = h_hi - h_lo
    n = z_true.size

    # exact-z limit: z* = z_true fixed at every h (kernel -> delta); g(h) =
    # -ln(sigma_d) - ln(mu(z_true,h)).
    mu_lo_exact = _d2_exponent_terms(z_true, h_lo, d_true)
    mu_hi_exact = _d2_exponent_terms(z_true, h_hi, d_true)
    g_lo_exact = -np.log(sigma_d) - np.log(mu_lo_exact)
    g_hi_exact = -np.log(sigma_d) - np.log(mu_hi_exact)
    # g(h) is the CORRECT term's own value; the diff convention (coded-correct)
    # is 0 - g(h), so its tilt is minus g's own central-difference slope.
    t_exact = -float(np.sum((g_hi_exact - g_lo_exact) / dh))

    # GLADE-mix smeared: find z*(h) = argmax of the correct-form integrand
    # (kernel x GW-density [x p_pop]) at each h neighbour, then evaluate the
    # prefactor piece along the MOVING peak. Vectorized dense-grid search +
    # local parabolic refinement (no per-event scipy calls).
    half_width_eff = _effective_half_width(z_true, sigma_d, sigma_rep, ALPHA_H)

    def zstar(h: float, include_ppop: bool) -> npt.NDArray[np.float64]:
        half_width = half_width_eff
        n_grid = 161
        u = np.linspace(-1.0, 1.0, n_grid)
        z_grid = z_true[:, None] + half_width[:, None] * u[None, :]
        z_grid = np.maximum(z_grid, 1e-8)
        mu = np.asarray(dist_vectorized(z_grid.reshape(-1), h=h), dtype=np.float64).reshape(
            z_grid.shape
        )
        ln_gw = (
            -np.log(sigma_d)[:, None]
            - np.log(mu)
            - 0.5 * ((d_true[:, None] - mu) / (sigma_d[:, None] * mu)) ** 2
        )
        ln_kernel = -0.5 * ((z_grid - z_true[:, None]) / np.maximum(sigma_rep, 1e-6)[:, None]) ** 2
        total = ln_gw + ln_kernel
        if include_ppop:
            w = np.asarray(cl._w_pop(z_grid.reshape(-1), ALPHA_H), dtype=np.float64).reshape(
                z_grid.shape
            )
            total = total + np.log(np.maximum(w, 1e-300))
        idx = np.argmax(total, axis=1)
        idx = np.clip(idx, 1, n_grid - 2)
        rows = np.arange(z_grid.shape[0])
        y0 = total[rows, idx - 1]
        y1 = total[rows, idx]
        y2 = total[rows, idx + 1]
        denom = y0 - 2.0 * y1 + y2
        delta = np.where(np.abs(denom) > 1e-12, 0.5 * (y0 - y2) / denom, 0.0)
        delta = np.clip(delta, -1.0, 1.0)
        z_peak = z_grid[rows, idx] + delta * (z_grid[rows, idx + 1] - z_grid[rows, idx])
        return np.asarray(np.maximum(z_peak, 1e-8), dtype=np.float64)

    out: dict[str, Any] = {
        "naive_tilt_nats_per_h": -D2_NAIVE_NATS_PER_H,
        "exact_z_limit_tilt_nats_per_h": t_exact,
        "exact_z_matches_naive": bool(np.isclose(t_exact, -D2_NAIVE_NATS_PER_H, rtol=1e-3)),
    }
    for label, include_ppop in (("with_ppop", True), ("without_ppop", False)):
        z_lo = zstar(h_lo, include_ppop)
        z_hi = zstar(h_hi, include_ppop)
        mu_lo = _d2_exponent_terms(z_lo, h_lo, d_true)
        mu_hi = _d2_exponent_terms(z_hi, h_hi, d_true)
        g_lo = -np.log(sigma_d) - np.log(mu_lo)
        g_hi = -np.log(sigma_d) - np.log(mu_hi)
        t_smeared = -float(np.sum((g_hi - g_lo) / dh))  # diff convention, see d3_exact_z note
        out[f"glade_mix_zstar_tracking_tilt_nats_per_h_{label}"] = t_smeared
        out[f"glade_mix_zstar_mean_shift_{label}"] = float(
            np.mean(np.abs(z_hi - z_lo) - np.abs(0.0))
        )
    out["method"] = (
        "z*(h) found per event by a 161-point grid search + local parabolic refine "
        "over [z_true - 5*sigma_rep, z_true + 5*sigma_rep] (matches the estimator's "
        "own +-5 sigma_z kernel window), maximizing the correct-form integrand "
        "(kernel x GW-density, with and without p_pop); the prefactor piece "
        "-ln(sigma_d * mu(z*(h))/h) is then evaluated at the MOVING z*(h) and "
        "central-differenced at the h=0.725/0.735 grid neighbours, summed over events. "
        "sigma_rep = per-event decile-median GLADE sigma_z (deterministic, no RNG)."
    )
    return out


# ---------------------------------------------------------------------------
# D3 -- exponent scale mu vs d_obs
# ---------------------------------------------------------------------------


def d3_exact_z(vctx: vt.VenueContext, i_lo: int, i_hi: int) -> dict[str, Any]:
    z_true = vctx.z_true
    d_true = vctx.d_L
    sigma_d = vctx.sigma_dL
    h_grid = np.asarray(vctx.gctx.cl_ctx.config.h_grid, dtype=np.float64)
    h_lo, h_hi = float(h_grid[i_lo]), float(h_grid[i_hi])
    dh = h_hi - h_lo

    def exponent_diff(h: float) -> npt.NDArray[np.float64]:
        mu = _d2_exponent_terms(z_true, h, d_true)
        correct = -0.5 * ((d_true - mu) / (sigma_d * mu)) ** 2
        coded = -0.5 * ((mu - d_true) / (sigma_d * d_true)) ** 2
        return correct - coded

    diff_lo = exponent_diff(h_lo)
    diff_hi = exponent_diff(h_hi)
    t_exact = float(np.sum((diff_hi - diff_lo) / dh))
    return {
        "tilt_nats_per_h": t_exact,
        "mean_abs_per_event_diff_at_h_hi": float(np.mean(np.abs(diff_hi))),
        "note": (
            "Analytically, the correct-minus-coded exponent diff is O((1-u)^3) in "
            "u = h_true/h near u=1, so its h-derivative at h_true vanishes to leading "
            "order; expect this tilt to be small (verified numerically, not assumed)."
        ),
    }


def _gl_nodes_weights() -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    x, w = roots_legendre(_N_QUAD)
    return np.asarray(x, dtype=np.float64), np.asarray(w, dtype=np.float64)


def d3_smeared(
    vctx: vt.VenueContext,
    i_lo: int,
    i_hi: int,
    sigma_rep: npt.NDArray[np.float64],
    doses: tuple[float, ...] = (0.25, 0.5, 1.0),
) -> dict[str, Any]:
    z_true = vctx.z_true
    d_true = vctx.d_L
    sigma_d = vctx.sigma_dL
    h_grid = np.asarray(vctx.gctx.cl_ctx.config.h_grid, dtype=np.float64)
    h_lo, h_hi = float(h_grid[i_lo]), float(h_grid[i_hi])
    dh = h_hi - h_lo
    x, w = _gl_nodes_weights()

    def log_integral(
        h: float,
        sig: npt.NDArray[np.float64],
        node_half_width: npt.NDArray[np.float64],
        exponent_kind: str,
    ) -> npt.NDArray[np.float64]:
        a = np.maximum(z_true - node_half_width, 1e-8)
        b = z_true + node_half_width
        half = 0.5 * (b - a)
        mid = 0.5 * (b + a)
        z_nodes = mid[:, None] + half[:, None] * x[None, :]
        mu = np.asarray(
            dist_vectorized(np.maximum(z_nodes.reshape(-1), 1e-8), h=h), dtype=np.float64
        ).reshape(z_nodes.shape)
        kernel = np.exp(-0.5 * ((z_nodes - z_true[:, None]) / sig[:, None]) ** 2)
        if exponent_kind == "correct":
            expo = -0.5 * ((d_true[:, None] - mu) / (sigma_d[:, None] * mu)) ** 2
        elif exponent_kind == "coded":
            expo = -0.5 * ((mu - d_true[:, None]) / (sigma_d[:, None] * d_true[:, None])) ** 2
        else:
            raise ValueError(exponent_kind)
        integ = kernel * np.exp(expo)
        val = half * (integ @ w)
        val = np.maximum(val, 1e-300)
        return np.log(val)

    out: dict[str, Any] = {"doses": {}}
    for f in doses:
        sig = f * sigma_rep
        sig = np.maximum(sig, 1e-6)
        # Node window resolves whichever of (dosed kernel, GW peak) is
        # narrower (see _effective_half_width) -- the KERNEL SHAPE itself
        # still uses the full dosed sig, only the quadrature node placement
        # is adapted.
        node_half_width = _effective_half_width(z_true, sigma_d, sig, ALPHA_H)
        ln_correct_lo = log_integral(h_lo, sig, node_half_width, "correct")
        ln_correct_hi = log_integral(h_hi, sig, node_half_width, "correct")
        ln_coded_lo = log_integral(h_lo, sig, node_half_width, "coded")
        ln_coded_hi = log_integral(h_hi, sig, node_half_width, "coded")
        diff_lo = ln_correct_lo - ln_coded_lo
        diff_hi = ln_correct_hi - ln_coded_hi
        t_f = float(np.sum((diff_hi - diff_lo) / dh))
        out["doses"][str(f)] = {"tilt_nats_per_h": t_f}
    out["method"] = (
        "Per event, 50-node Gauss-Legendre quadrature over z in "
        "[z_true - w, z_true + w], w = 5*min(f*sigma_rep, GW-peak-width) (node "
        "placement resolves whichever of the dosed kernel or the GW-density term "
        "is narrower -- see _effective_half_width; the kernel SHAPE itself always "
        "uses the full dosed f*sigma_rep) of kernel(z) x exp(exponent), for the correct "
        "exponent -(d_obs-mu)^2/(2 sigma_d^2 mu^2) and the coded exponent "
        "-(mu-d_obs)^2/(2 sigma_d^2 d_obs^2), holding everything else (kernel, no "
        "prefactor) identical between the two so the diff isolates purely the "
        "exponent-scale term; central-differenced at the h=0.725/0.735 grid "
        "neighbours, summed over events. Dose f scales sigma_rep, matching how "
        "_apply_dose_mask scales sigma_z on the real instrument."
    )
    return out


# ---------------------------------------------------------------------------
# D5 -- restated, converted to nats/h via the displacement law
# ---------------------------------------------------------------------------


def d5_ren_and_m3(abar_1d: float, abar_2d: float) -> dict[str, Any]:
    return {
        "ren_measured_bias_b": D5_REN_BIAS,
        "m3_bias_ceiling_b": D5_M3_BIAS_CEILING,
        "ren_tilt_nats_per_h_1d": D5_REN_BIAS * abar_1d,
        "ren_tilt_nats_per_h_2d": D5_REN_BIAS * abar_2d,
        "m3_tilt_ceiling_nats_per_h_1d": D5_M3_BIAS_CEILING * abar_1d,
        "note": (
            "D5 is measured in MAP-bias units (b), not nats/h; converted via the "
            "displacement-law geometry T ~= bias * Abar using AJREN's OWN local "
            "curvature (same object L4-T1 section 3 used), NOT a refit. This describes "
            "the BASELINE window truncation + its stage-3 renormalization diagnostic, "
            "not the AJREN repair's own W_k-division mechanics (which is measured "
            "directly as the 'instrument REN tilt' in L4-T1 and used as-is in the "
            "T(AJREN) ledger line below)."
        ),
    }


# ---------------------------------------------------------------------------
# Item 4 -- rough 2D estimate
# ---------------------------------------------------------------------------


def estimate_2d_channel(
    vctx: vt.VenueContext,
    i_lo: int,
    i_hi: int,
    sigma_rep: npt.NDArray[np.float64],
    abar_1d: float,
    abar_2d: float,
) -> dict[str, Any]:
    """Rough proxy: event-weight T_D3(full dose) by 1/sigma_Mz vs unweighted.

    ``sigma_Mz`` (fractional host-mass CRB error) is a crude sharpness proxy for
    where the 2D (mass) channel's g_i factor adds information; a channel-specific
    excess in D3 would show up as a systematic difference between the
    mass-precision-weighted and unweighted tilt. This is explicitly a ROUGH
    estimate (task item 4 permits it), not a g_i-exact recomputation.
    """
    z_true = vctx.z_true
    d_true = vctx.d_L
    sigma_d = vctx.sigma_dL
    sigma_Mz = vctx.sigma_Mz
    h_grid = np.asarray(vctx.gctx.cl_ctx.config.h_grid, dtype=np.float64)
    h_lo, h_hi = float(h_grid[i_lo]), float(h_grid[i_hi])
    dh = h_hi - h_lo
    x, w = _gl_nodes_weights()
    sig = np.maximum(sigma_rep, 1e-6)
    node_half_width = _effective_half_width(z_true, sigma_d, sig, ALPHA_H)

    def per_event_tilt(h_a: float, h_b: float) -> npt.NDArray[np.float64]:
        def log_integral(h: float, exponent_kind: str) -> npt.NDArray[np.float64]:
            a = np.maximum(z_true - node_half_width, 1e-8)
            b = z_true + node_half_width
            half = 0.5 * (b - a)
            mid = 0.5 * (b + a)
            z_nodes = mid[:, None] + half[:, None] * x[None, :]
            mu = np.asarray(
                dist_vectorized(np.maximum(z_nodes.reshape(-1), 1e-8), h=h), dtype=np.float64
            ).reshape(z_nodes.shape)
            kernel = np.exp(-0.5 * ((z_nodes - z_true[:, None]) / sig[:, None]) ** 2)
            if exponent_kind == "correct":
                expo = -0.5 * ((d_true[:, None] - mu) / (sigma_d[:, None] * mu)) ** 2
            else:
                expo = -0.5 * ((mu - d_true[:, None]) / (sigma_d[:, None] * d_true[:, None])) ** 2
            integ = kernel * np.exp(expo)
            val = np.maximum(half * (integ @ w), 1e-300)
            return np.log(val)

        diff_a = log_integral(h_a, "correct") - log_integral(h_a, "coded")
        diff_b = log_integral(h_b, "correct") - log_integral(h_b, "coded")
        return (diff_b - diff_a) / (h_b - h_a)

    per_event = per_event_tilt(h_lo, h_hi)
    t_unweighted = float(np.sum(per_event))
    weight = 1.0 / np.maximum(sigma_Mz, 1e-6)
    weight_norm = weight * (per_event.size / np.sum(weight))
    t_weighted = float(np.sum(per_event * weight_norm))
    excess_nats = t_weighted - t_unweighted
    excess_bias_1d = excess_nats / abar_1d
    excess_bias_2d = excess_nats / abar_2d
    return {
        "t_d3_unweighted_full_dose": t_unweighted,
        "t_d3_mass_precision_weighted_full_dose": t_weighted,
        "excess_nats_per_h": excess_nats,
        "excess_in_bias_units_via_abar_1d": excess_bias_1d,
        "excess_in_bias_units_via_abar_2d": excess_bias_2d,
        "stage3_2d_only_subadditivity_reference": STAGE3_2D_SUBADDITIVE_NATS,
        "order_of_magnitude_match": bool(
            abs(excess_bias_2d) > 0.1 * STAGE3_2D_SUBADDITIVE_NATS
            and abs(excess_bias_2d) < 10.0 * STAGE3_2D_SUBADDITIVE_NATS
        ),
        "method": (
            "ROUGH proxy only (task item 4 explicitly permits this): per-event T_D3 "
            "at full dose (GLADE-mix sigma_rep), reweighted by 1/sigma_Mz (normalized "
            "to preserve the event count) as a crude stand-in for the 2D channel's "
            "g_i mass-precision weighting, vs the plain unweighted sum. NOT a "
            "recomputation of the actual g_i factor (completion_mass_factor_g), which "
            "would require the full per-event host-mass CRB machinery -- flagged as an "
            "order-of-magnitude estimate, not a point prediction."
        ),
    }


# ---------------------------------------------------------------------------
# Ledger closure
# ---------------------------------------------------------------------------


def ledger_closure(
    d1: dict[str, Any],
    d2: dict[str, Any],
    d3_full_dose: float,
    d5: dict[str, Any],
    ajren: dict[str, Any] | None,
) -> dict[str, Any]:
    d1_tilt = d1["analytic_tilt_nats_per_h"]
    d2_naive = d2["naive_tilt_nats_per_h"]
    d2_smeared_with = d2["glade_mix_zstar_tracking_tilt_nats_per_h_with_ppop"]
    d2_smeared_without = d2["glade_mix_zstar_tracking_tilt_nats_per_h_without_ppop"]
    d4_full = D4_PREDICTED_NATS["1.0"]
    d5_1d = d5["ren_tilt_nats_per_h_1d"]

    rows = {}
    for d2_label, d2_val in (
        ("naive", d2_naive),
        ("zstar_with_ppop", d2_smeared_with),
        ("zstar_without_ppop", d2_smeared_without),
    ):
        base = d1_tilt + d2_val + d3_full_dose + d4_full + d5_1d
        pred_mn0x = base
        pull_mn0x = (pred_mn0x - T_MN0X_MEAN) / T_MN0X_SE

        for j_label, j_dprime in (
            ("full_tracking", J_DPRIME_FULL_TRACKING),
            ("attenuated", J_DPRIME_ATTENUATED),
        ):
            j_total = J_FULL_DOSE_NATS_PER_H + j_dprime
            pred_am2p = base + j_total
            pull_am2p = (pred_am2p - T_AM2P_MEAN) / T_AM2P_SE

            row: dict[str, Any] = {
                "predicted_T_MN0X": pred_mn0x,
                "measured_T_MN0X": T_MN0X_MEAN,
                "measured_T_MN0X_se": T_MN0X_SE,
                "pull_MN0X_sigma": pull_mn0x,
                "predicted_T_AM2P": pred_am2p,
                "measured_T_AM2P": T_AM2P_MEAN,
                "measured_T_AM2P_se": T_AM2P_SE,
                "pull_AM2P_sigma": pull_am2p,
            }
            if ajren is not None:
                ren_instrument = ajren["section2_ren_tilt_vs_toy"][
                    "instrument_ren_tilt_1d_nats_per_h"
                ]
                ren_instrument_se = ajren["section2_ren_tilt_vs_toy"]["instrument_ren_tilt_1d_se"]
                pred_ajren = pred_am2p + ren_instrument
                t_ajren_mean = ajren["section1_tilt_vs_alpha"]["T1_mean"]
                t_ajren_se = ajren["section1_tilt_vs_alpha"]["T1_se"]
                pull_ajren = (pred_ajren - t_ajren_mean) / np.hypot(t_ajren_se, ren_instrument_se)
                row.update(
                    {
                        "predicted_T_AJREN": pred_ajren,
                        "measured_T_AJREN": t_ajren_mean,
                        "measured_T_AJREN_se": t_ajren_se,
                        "instrument_ren_tilt_used": ren_instrument,
                        "instrument_ren_tilt_se": ren_instrument_se,
                        "pull_AJREN_sigma": pull_ajren,
                    }
                )
            rows[f"d2_{d2_label}__j_{j_label}"] = row
    return rows


# ---------------------------------------------------------------------------
# Dose-structure test
# ---------------------------------------------------------------------------


def dose_structure_test(d3_smeared_out: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for f_str, measured in T_RES_MEASURED.items():
        t_d3_f = d3_smeared_out["doses"][f_str]["tilt_nats_per_h"]
        d4_f = D4_PREDICTED_NATS[f_str]
        predicted = t_d3_f + d4_f
        diff = predicted - measured
        out[f_str] = {
            "T_D3": t_d3_f,
            "D4_predicted": d4_f,
            "T_res_predicted": predicted,
            "T_res_measured": measured,
            "diff": diff,
            "within_3x_level_se_tolerance": bool(abs(diff) <= T_RES_TOLERANCE_NATS),
        }
    return out


def main() -> None:
    vctx, i_lo, i_hi = build_population_context()
    n = vctx.z_true.size
    sigma_rep = representative_sigma_z(vctx)

    l4_t1 = None
    l4_t1_path = RESULTS_DIR / "L4_T1_output.json"
    if l4_t1_path.exists():
        l4_t1 = _load_json(l4_t1_path)

    d1 = d1_alpha(vctx, i_lo, i_hi, n)
    d2 = d2_prefactor(vctx, i_lo, i_hi, sigma_rep)
    d3_exact = d3_exact_z(vctx, i_lo, i_hi)
    d3_smeared_out = d3_smeared(vctx, i_lo, i_hi, sigma_rep)

    abar_1d = (
        l4_t1["section3_displacement_law"]["channel_1d"]["Abar_mean_at_truth"]
        if l4_t1
        else float("nan")
    )
    abar_2d = (
        l4_t1["section3_displacement_law"]["channel_2d"]["Abar_mean_at_truth"]
        if l4_t1
        else float("nan")
    )
    d5 = d5_ren_and_m3(abar_1d, abar_2d)

    d3_full_dose = d3_smeared_out["doses"]["1.0"]["tilt_nats_per_h"]
    ledger = ledger_closure(d1, d2, d3_full_dose, d5, l4_t1)
    dose_test = dose_structure_test(d3_smeared_out)
    two_d = estimate_2d_channel(vctx, i_lo, i_hi, sigma_rep, abar_1d, abar_2d)

    output = {
        "note": "results/mechanism_study_20260813/L4_T2_AUDIT_20260815.md",
        "parent": "results/mechanism_study_20260813/L4_DER_PART1_20260815.md (commit 5e77e196)",
        "l4_t1_input": "results/mechanism_study_20260813/L4_T1_output.json"
        if l4_t1 is not None
        else "NOT FOUND -- T(AJREN) ledger rows omitted",
        "population": {"n_events": int(n), "h_true": ALPHA_H},
        "constants": {
            "ALPHA_N": ALPHA_N,
            "ALPHA_H": ALPHA_H,
            "ALPHA_SLOPE_COEFF": ALPHA_SLOPE_COEFF,
            "ALPHA_TILT_NATS_PER_H": ALPHA_TILT_NATS_PER_H,
            "D2_NAIVE_NATS_PER_H": D2_NAIVE_NATS_PER_H,
            "J_FULL_DOSE_NATS_PER_H": J_FULL_DOSE_NATS_PER_H,
            "J_DPRIME_FULL_TRACKING": J_DPRIME_FULL_TRACKING,
            "J_DPRIME_ATTENUATED": J_DPRIME_ATTENUATED,
            "D4_PREDICTED_NATS": D4_PREDICTED_NATS,
            "D4_STATUS": D4_STATUS,
            "T_RES_MEASURED": T_RES_MEASURED,
            "T_RES_TOLERANCE_NATS": T_RES_TOLERANCE_NATS,
        },
        "D1_alpha": d1,
        "D2_prefactor": d2,
        "D3_exact_z": d3_exact,
        "D3_smeared": d3_smeared_out,
        "D5": d5,
        "ledger_closure": ledger,
        "dose_structure_test": dose_test,
        "estimate_2d_channel": two_d,
    }

    out_path = RESULTS_DIR / "L4_T2_output.json"
    with open(out_path, "w") as fh:
        json.dump(output, fh, indent=2, sort_keys=False)
    print(f"wrote {out_path}")
    print(
        json.dumps(
            {
                "D1_analytic": d1["analytic_tilt_nats_per_h"],
                "D1_numeric": d1["numeric_tilt_nats_per_h"],
                "D1_exponent_numeric": d1["numeric_exponent_at_h_true"],
                "D2_naive": d2["naive_tilt_nats_per_h"],
                "D2_exact_z": d2["exact_z_limit_tilt_nats_per_h"],
                "D2_zstar_with_ppop": d2["glade_mix_zstar_tracking_tilt_nats_per_h_with_ppop"],
                "D2_zstar_without_ppop": d2[
                    "glade_mix_zstar_tracking_tilt_nats_per_h_without_ppop"
                ],
                "D3_exact_z": d3_exact["tilt_nats_per_h"],
                "D3_smeared_full_dose": d3_full_dose,
                "dose_structure_test": dose_test,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
