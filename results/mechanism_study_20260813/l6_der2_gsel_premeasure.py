"""L6-DER2 §4 step 1 — mirror pre-measurement of the fused A-FULL-2D `g_sel`
candidate (ledger rows #112/#114; derivation
`results/mechanism_study_20260813/L6_DER2_CORRECT_FORM_2D_20260816.md`).

**Status: PRESENTED, NOT ADJUDICATED.** This script measures the registered
prediction of L6-DER2 §2/§4 as-is; the result is reported without tuning
toward it. Adjudication (xhigh verifier) is the NEXT registered step, not
this one.

Three configs, same 15 MN0X seed replays, same c1/c2 mirror method
(``l6_c2_switch_decomposition.py`` base) and k=20 (h=0.725) / k=22 (h=0.735):

- ``base`` — the coded original variant (``ESTIMATOR_VARIANT_BASE``): c1 =
  integral of kern*p_gw dz, c2 = integral of kern*p_gw*g dz, with g the
  coded ``completion_mass_factor_g`` via ``venue_transfer._g_ball_capped``.
  Validated bit-exactly against the stored ``ln_post_1d``/``ln_post_2d``
  per-seed vectors in ``MN0X_h0p730_results_seeds0_100.json``.

- ``a_full`` — the A-FULL variant (``ESTIMATOR_VARIANT_A_FULL``) mirrored
  verbatim from ``venue_transfer._channel_terms_at_h``: d_obs-density GW
  factor ``p_gw_full = N(d_obs/d_L; 1, sigma)/d_L``, numerator node weight
  ``w_sel = w_pop(z;h) * S_bar_phi(z;h)`` (``S_bar_phi`` the phi-marginal
  survival table, ``gctx.cl_ctx.s_phi_tables[k]``), and the per-candidate
  leave-one-out impostor weight ``loo_w = 1/imp_k``
  (``venue_transfer._loo_impostor_weights``). c2 = integral of
  ``kern*p_gw_full*w_sel*loo_w*g`` dz — SAME ``g`` as ``base`` (the coded
  mass factor does not depend on estimator_variant; the real
  ``_channel_terms_at_h`` computes it once per chunk regardless of variant,
  so sharing one ``_g_ball_capped`` call between the ``base`` and ``a_full``
  columns of this mirror is bit-exact, not an approximation). Cross-validated
  against a direct call of
  ``venue_transfer.log_channel_posteriors_ball_sigma_vector(...,
  estimator_variant="a_full")`` for >=2 seeds x both k.

- ``a_full_gsel`` (THE CANDIDATE, L6-DER2 §3) — identical to ``a_full`` in
  the 1D channel (c1 IS ``a_full``'s c1 array, not merely equal to it — see
  ``channel_terms_all_configs`` below; this documents the by-construction
  sharing rather than "discovering" an independent match, the same
  convention ``l6_c2_switch_decomposition.py`` uses for its switch-invariant
  ln1). In the 2D channel the pair (S_bar_phi(z) node-weight x coded g) is
  replaced by the single fused object

      g_sel(z, f; h) = INTEGRAL dx_M  N(x_M; mu_cond(f), sigma_cond)
                       * phi_x(x_M; z) * S(x_M * M_z_obs_i; z, h)

  with S the UNMARGINALIZED with-BH detection survival, queried the SAME way
  ``precompute_phi_marginal_survival`` queries it:
  ``detection_probability_with_bh_mass_interpolated(d_L, M_z, 0, 0, h=h,
  **_wbh_z_kwargs(detection, z))`` — isotropic (phi=theta=0), FIX-3 z
  pass-through if the detection object has it active (verified here: it does
  not, in the venue's ``ClosedLoopContext``; the kwarg call is included
  anyway for correctness under a future flag flip). c2 = integral of
  ``kern*p_gw_full*w_pop(z)*loo_w*g_sel`` dz (NO S_bar_phi factor — it is
  absorbed into ``g_sel``).

  Convention choices (the derivation's ``g_sel`` formula is schematic; these
  pin it to the coded measure conventions, per the task brief):

  * ``mu_cond``, ``sigma_cond``, the (d_L_frac, M_z_frac) 2x2-block
    projection, and the ``x_M`` dimensionless-mass measure are copied
    VERBATIM from ``completion_mass_factor_g`` / ``_g_ball_capped`` (same
    ``phi_x(x_M;z) = phi(x_M * M_z_obs_i/(1+z)) * M_z_obs_i/(1+z)``).
  * The Gauss-Hermite contraction reuses the SAME order
    (``n_hermite = gctx.cl_ctx.config.n_hermite``, 64) and the SAME
    ``roots_hermite`` nodes/weights as ``completion_mass_factor_g``'s
    ``_contract_group`` — but NON-ADAPTIVE (always full order 64, no
    fast-order n=8 branch). This is a deliberate convention choice, not
    something the derivation specifies: the coded object's Route-1 adaptive
    order is a truncation-error optimization for a *smooth* phi_x integrand;
    folding in S(x_M) (a detection-horizon survival with a much sharper
    d_L-cutoff structure) makes the n=8 fast order's error bound
    (derived for phi_x alone, `completion_mass_factor_g` docstring) inapplicable,
    so this script pins the exact order everywhere and records the runtime
    cost as a convention, not a physics choice.
  * The detector-frame mass queried against S is ``M_z = x_M * M_z_obs_i``
    (``x_M`` is defined as ``M_z / M_z_obs_i`` in the coded convention — see
    ``completion_mass_factor_g``'s docstring, "``x_M = M_z/M_z,det,i``" —
    so this is the SAME x_M the mass-factor already integrates over, not a
    new coordinate).
  * The d_L queried against S is the SAME absolute ``d_L(z;h)`` node value
    already computed for the outer kernel/GW-density integral (before the
    ``/d_obs`` ratio is taken) — no re-derivation.

Registered prediction (L6-DER2 §2/§4, committed 09c02c06 BEFORE this run;
QUOTED, not recomputed): "the 2D-1D excess collapses under the fused form
(channel B, ~+139 nats/h scale in the coded form, cancelled to the ~few-nat
level), and the 1D channel is bit-untouched." This script reports the
measured dT2(gsel) = T2(a_full_gsel) - T2(a_full) and
d_excess = excess(a_full_gsel) - excess(a_full) as-is against that qualitative
target; it does NOT compute a numeric registered value to compare against
(none was pre-registered — L6-DER2 gives a scale reference, +139 nats/h, from
the DIFFERENT `base`-vs-switch measurement in ``l6_c2_switch_decomposition.py``,
not a numeric prediction for this fused-form measurement).

Output: ``L6_DER2_GSEL_PREMEASURE_output.json``.
"""

import math
import multiprocessing as mp
import sys
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.special import roots_hermite
from scipy.stats import norm

RESULTS_DIR = Path(__file__).parent
sys.path.insert(0, str(RESULTS_DIR))

import json  # noqa: E402

from l4_t2_audit import build_population_context  # noqa: E402

from darksiren_emri.bayesian_inference import bayesian_statistics as bs  # noqa: E402
from darksiren_emri.physical_relations import dist_vectorized  # noqa: E402
from darksiren_emri.validation import calibration_gate as cg  # noqa: E402
from darksiren_emri.validation import closed_loop_gfrac as cl  # noqa: E402
from darksiren_emri.validation import venue_transfer as vt  # noqa: E402

N_SEEDS = 15
N_VALIDATION_SEEDS = 2  # a_full direct-call cross-validation subset
K_LO = 20  # h = 0.725
K_HI = 22  # h = 0.735
CONFIGS_1D = ("base", "afull")  # afull's c1 IS gsel's c1 (shared array)
CONFIGS_2D = ("base", "afull", "gsel")

REGISTERED_PREDICTION_NOTE = (
    "L6-DER2 SS2/SS4 (committed 09c02c06 BEFORE this run): the 2D-1D excess "
    "collapses under the fused g_sel form (channel B, coded-form scale "
    "~+139 nats/h per l6_c2_switch_decomposition.py's dT2_sb, cancelled to "
    "the ~few-nat level in the correct form), and the 1D channel is "
    "bit-untouched. Qualitative target -- no numeric value was "
    "pre-registered for THIS measurement; reported as measured, not tuned."
)


def _load_json(path: Path) -> dict[str, Any]:
    with open(path) as fh:
        result: dict[str, Any] = json.load(fh)
    return result


# ── the fused g_sel mass factor ("_S_PHI_Z_CHUNK"-style chunking is not
#    needed here: the flattened-node cap already bounds each interpolator
#    call to node_chunk * n_hermite <= 200_000 * 64 = 1.28e7 points) ────────


def g_sel_mass_factor(
    z_nodes: npt.NDArray[np.float64],
    d_L_fraction: npt.NDArray[np.float64],
    d_L_abs: npt.NDArray[np.float64],
    det_M_z: float,
    proj_d_L_to_M: float,
    sigma_cond_M: float,
    h: float,
    detection_obj: Any,
    *,
    n_hermite: int,
    force_S_one: bool = False,
) -> npt.NDArray[np.float64]:
    """The fused L6-DER2 SS3 object, one event, flattened 1D node arrays.

    Verbatim-conventions mirror of ``completion_mass_factor_g``'s
    ``_contract_group`` (non-adaptive path) with the UNmarginalized with-BH
    survival ``S`` folded into the same Gauss-Hermite contraction, queried
    the same way ``precompute_phi_marginal_survival`` queries it. With
    ``force_S_one=True`` this reduces to EXACTLY
    ``completion_mass_factor_g(..., n_hermite=n_hermite, adaptive=False)``
    (same operation order) -- the S≡1 refactor validation gate.

    Args:
        z_nodes: Quadrature redshifts, shape ``(k,)``.
        d_L_fraction: ``d_L(z;h)/d_L,obs,i`` at the same nodes, shape ``(k,)``.
        d_L_abs: ``d_L(z;h)`` (Gpc, absolute) at the same nodes, shape ``(k,)``.
        det_M_z: The event's measured detector-frame BH mass ``M_z,obs,i``.
        proj_d_L_to_M: ``cov_4d[3,2]/cov_4d[2,2]``.
        sigma_cond_M: ``sqrt(cov_4d[3,3] - cov_4d[3,2]^2/cov_4d[2,2])``.
        h: Hubble parameter (survival query + record-keeping only; z_nodes
            and d_L_abs already encode it).
        detection_obj: The gate's ``SimulationDetectionProbability``.
        n_hermite: Gauss-Hermite order (pinned, non-adaptive).
        force_S_one: S≡1 refactor-check mode (see module docstring).

    Returns:
        ``g_sel`` at the nodes, shape ``(k,)``, same units as
        ``completion_mass_factor_g`` (density in ``x_M``).
    """
    x_nodes, x_weights = roots_hermite(n_hermite)
    z_arr = np.asarray(z_nodes, dtype=np.float64)
    scale = det_M_z / (1.0 + z_arr)  # (k,)
    mu_cond = 1.0 + proj_d_L_to_M * (np.asarray(d_L_fraction, dtype=np.float64) - 1.0)  # (k,)
    x_M = mu_cond[:, None] + math.sqrt(2.0) * sigma_cond_M * x_nodes[None, :]  # (k, n_h)
    M_source = x_M * scale[:, None]
    phi_x = bs.dark_mass_density_per_mass(M_source) * scale[:, None]

    if force_S_one:
        integrand = phi_x
    else:
        M_z_query = x_M * det_M_z  # (k, n_h) -- x_M = M_z / M_z_obs_i (coded convention)
        d_L_query = np.broadcast_to(
            np.asarray(d_L_abs, dtype=np.float64)[:, None], M_z_query.shape
        )
        z_query = np.broadcast_to(z_arr[:, None], M_z_query.shape)
        zeros = np.zeros(M_z_query.size, dtype=np.float64)
        S = np.asarray(
            detection_obj.detection_probability_with_bh_mass_interpolated(
                d_L_query.reshape(-1),
                M_z_query.reshape(-1),
                zeros,
                zeros,
                h=h,
                **bs._wbh_z_kwargs(detection_obj, z_query.reshape(-1)),
            ),
            dtype=np.float64,
        ).reshape(M_z_query.shape)
        integrand = phi_x * S

    return np.asarray((integrand @ x_weights) / math.sqrt(math.pi), dtype=np.float64)


def _g_sel_ball_capped(
    gctx: cg.GateContext,
    universe: Any,
    event_idx: npt.NDArray[np.int64],
    z_nodes: npt.NDArray[np.float64],
    d_L_frac: npt.NDArray[np.float64],
    d_L_abs: npt.NDArray[np.float64],
    valid: npt.NDArray[np.bool_],
    h: float,
    detection_obj: Any,
    *,
    node_chunk: int = vt._G_NODE_CHUNK,
    force_S_one: bool = False,
) -> npt.NDArray[np.float64]:
    """Per-event-loop, memory-capped mirror producing ``g_sel`` at nodes.

    Structurally identical to ``venue_transfer._g_ball_capped`` (same
    per-event loop, same ``node_chunk`` splitting convention) with the extra
    ``d_L_abs`` array threaded through and ``g_sel_mass_factor`` in place of
    ``completion_mass_factor_g``.
    """
    s_dd = universe.sigma_dL**2
    s_dm = universe.rho * universe.sigma_dL * universe.sigma_Mz
    s_mm = universe.sigma_Mz**2
    proj = np.where(s_dd > 0.0, s_dm / np.maximum(s_dd, 1e-300), 0.0)
    sigma_cond = np.sqrt(np.maximum(s_mm - proj * s_dm, 1e-30))

    out = np.zeros_like(z_nodes)
    n_hermite = gctx.cl_ctx.config.n_hermite
    n_quad = z_nodes.shape[1]
    if node_chunk <= 0:
        max_rows_per_call = int(z_nodes.shape[0]) + 1
    else:
        max_rows_per_call = max(node_chunk // max(n_quad, 1), 1)
    present = np.unique(event_idx)
    starts = np.searchsorted(event_idx, present, side="left")
    stops = np.searchsorted(event_idx, present, side="right")
    for i, s, e in zip(present.tolist(), starts.tolist(), stops.tolist(), strict=True):
        rows = np.arange(s, e)
        rows = rows[valid[rows]]
        if rows.size == 0:
            continue
        for r0 in range(0, rows.size, max_rows_per_call):
            rr = rows[r0 : r0 + max_rows_per_call]
            zz = z_nodes[rr].reshape(-1)
            ff = d_L_frac[rr].reshape(-1)
            dd = d_L_abs[rr].reshape(-1)
            out[rr] = g_sel_mass_factor(
                zz,
                ff,
                dd,
                float(universe.M_z_obs[i]),
                float(proj[i]),
                float(sigma_cond[i]),
                h,
                detection_obj,
                n_hermite=n_hermite,
                force_S_one=force_S_one,
            ).reshape(rr.size, n_quad)
    return out


# ── the c1/c2 mirror, all three configs in one pass ──────────────────────────


def channel_terms_all_configs(
    vctx: vt.VenueContext,
    universe: Any,
    ball: Any,
    sig_z: npt.NDArray[np.float64],
    k: int,
) -> tuple[dict[str, float], dict[str, float]]:
    """One (seed, k): base/a_full/a_full_gsel c1, c2 in a single chunk pass.

    Args:
        vctx: The venue context.
        universe: The seed's synthetic universe.
        ball: The seed's candidate balls.
        sig_z: Per-candidate sigma_z, aligned with ``ball.z_obs``.
        k: The h-grid index to evaluate at.

    Returns:
        ``(ln1, ln2)`` dicts. ``ln1["gsel"]`` is literally ``ln1["afull"]``
        (by-construction sharing, see module docstring) rather than an
        independently accumulated value -- this documents c1 bit-identity,
        it does not merely assert it after the fact.
    """
    gctx = vctx.gctx
    h = gctx.cl_ctx.config.h_grid[k]
    n = universe.z_true.size
    x = gctx.cl_ctx.gl_nodes
    w_gl = gctx.cl_ctx.gl_weights
    ev = ball.event_idx
    z_obs = ball.z_obs
    d_obs_e = universe.d_L_obs
    sig_e = universe.sigma_dL
    d_obs_p = d_obs_e[ev]
    sig_p = sig_e[ev]
    K = np.maximum(ball.K, 1)
    n_pairs = int(z_obs.size)
    chunk_pairs = vctx.vcfg.chunk_pairs
    chunks = vt._pair_chunks(n_pairs, chunk_pairs)
    g_node_chunk = vt._G_NODE_CHUNK if chunk_pairs > 0 else 0

    d_L_nodes, z_tab = gctx.cl_ctx.z_of_dl_tables[k]
    z_hi_e = np.interp(d_obs_e * (1.0 + cl._SIGMA_WINDOW * sig_e), d_L_nodes, z_tab)
    z_lo_e = np.interp(d_obs_e * (1.0 - cl._SIGMA_WINDOW * sig_e), d_L_nodes, z_tab)
    z_lo_e = np.maximum(z_lo_e, 1e-6)
    z_hi_e = np.minimum(z_hi_e, z_tab[-1])
    z_lo_p = z_lo_e[ev]
    z_hi_p = z_hi_e[ev]

    loo_w = vt._loo_impostor_weights(gctx, universe, ball, sig_z)
    z_s_tab, s_phi_tab = gctx.cl_ctx.s_phi_tables[k]
    detection = gctx.cl_ctx.detection

    c1 = {"base": np.zeros(n_pairs), "afull": np.zeros(n_pairs)}
    c2 = {name: np.zeros(n_pairs) for name in CONFIGS_2D}

    for a0, a1 in chunks:
        sl = np.arange(a0, a1, dtype=np.int64)
        sig_c = sig_z[sl]
        q = sig_c > 0.0
        if np.any(q):
            rows_q = sl[q]
            zo = z_obs[rows_q]
            so = sig_c[q]
            a = np.maximum(z_lo_p[rows_q], zo - cg._IMPOSTOR_KERNEL_WINDOW * so)
            b = np.minimum(z_hi_p[rows_q], zo + cg._IMPOSTOR_KERNEL_WINDOW * so)
            valid = b > a
            half = 0.5 * (b - a)
            mid = 0.5 * (b + a)
            z_nodes = mid[:, None] + half[:, None] * x[None, :]
            z_nodes_floor = np.maximum(z_nodes.reshape(-1), 1e-8)
            d_L_n = np.asarray(
                dist_vectorized(z_nodes_floor, h=h), dtype=np.float64
            ).reshape(z_nodes.shape)
            d_L_frac = d_L_n / d_obs_p[rows_q][:, None]
            kern = norm.pdf(z_nodes, loc=zo[:, None], scale=so[:, None])

            # base (coded original variant)
            p_gw_base = norm.pdf(d_L_frac, loc=1.0, scale=sig_p[rows_q][:, None])
            integ_base = kern * p_gw_base
            c1q_base = half * (integ_base @ w_gl)
            g_shared = vt._g_ball_capped(
                gctx, universe, ev[rows_q], z_nodes, d_L_frac, valid, node_chunk=g_node_chunk
            )
            c2q_base = half * ((integ_base * g_shared) @ w_gl)
            c1["base"][rows_q] = np.where(valid, c1q_base, 0.0)
            c2["base"][rows_q] = np.where(valid, c2q_base, 0.0)

            # a_full shared ingredients (verbatim ESTIMATOR_VARIANT_A_FULL)
            ratio = d_obs_p[rows_q][:, None] / d_L_n
            p_gw_full = norm.pdf(ratio, loc=1.0, scale=sig_p[rows_q][:, None]) / d_L_n
            w_pop_z = np.asarray(cl._w_pop(z_nodes_floor, h), dtype=np.float64).reshape(
                z_nodes.shape
            )
            s_phi_z = np.interp(z_nodes, z_s_tab, s_phi_tab)
            w_sel = w_pop_z * s_phi_z
            integ_afull = kern * p_gw_full * w_sel * loo_w[rows_q][:, None]
            c1q_afull = half * (integ_afull @ w_gl)
            # g_shared: same call as `base`'s, reused (bit-exact — see module docstring)
            c2q_afull = half * ((integ_afull * g_shared) @ w_gl)
            c1["afull"][rows_q] = np.where(valid, c1q_afull, 0.0)
            c2["afull"][rows_q] = np.where(valid, c2q_afull, 0.0)

            # a_full_gsel: c1 == afull's c1 (SS3: 1D channel untouched); c2
            # drops the S_bar_phi factor (absorbed into g_sel).
            integ_gsel_2d_weight = kern * p_gw_full * w_pop_z * loo_w[rows_q][:, None]
            g_sel_vals = _g_sel_ball_capped(
                gctx,
                universe,
                ev[rows_q],
                z_nodes,
                d_L_frac,
                d_L_n,
                valid,
                h,
                detection,
                node_chunk=g_node_chunk,
            )
            c2q_gsel = half * ((integ_gsel_2d_weight * g_sel_vals) @ w_gl)
            c2["gsel"][rows_q] = np.where(valid, c2q_gsel, 0.0)

        if not np.all(q):
            rows_p = sl[~q]
            zo = z_obs[rows_p]
            valid_p = (zo >= z_lo_p[rows_p]) & (zo <= z_hi_p[rows_p])
            zo_floor = np.maximum(zo, 1e-8)
            d_pt = np.asarray(dist_vectorized(zo_floor, h=h), dtype=np.float64)
            frac = d_pt / d_obs_p[rows_p]

            p_gw_p_base = norm.pdf(frac, loc=1.0, scale=sig_p[rows_p])
            g_pt_shared = vt._g_ball_capped(
                gctx,
                universe,
                ev[rows_p],
                zo[:, None],
                frac[:, None],
                valid_p,
                node_chunk=g_node_chunk,
            )[:, 0]
            c1["base"][rows_p] = np.where(valid_p, p_gw_p_base, 0.0)
            c2["base"][rows_p] = np.where(valid_p, p_gw_p_base * g_pt_shared, 0.0)

            p_gw_p_full = norm.pdf(1.0 / frac, loc=1.0, scale=sig_p[rows_p]) / d_pt
            w_pop_p = np.asarray(cl._w_pop(zo_floor, h), dtype=np.float64)
            s_phi_p = np.interp(zo_floor, z_s_tab, s_phi_tab)
            weight_afull_p = p_gw_p_full * w_pop_p * s_phi_p * loo_w[rows_p]
            c1["afull"][rows_p] = np.where(valid_p, weight_afull_p, 0.0)
            c2["afull"][rows_p] = np.where(valid_p, weight_afull_p * g_pt_shared, 0.0)

            weight_gsel_p_2d = p_gw_p_full * w_pop_p * loo_w[rows_p]
            g_sel_pt = _g_sel_ball_capped(
                gctx,
                universe,
                ev[rows_p],
                zo[:, None],
                frac[:, None],
                d_pt[:, None],
                valid_p,
                h,
                detection,
                node_chunk=g_node_chunk,
            )[:, 0]
            c2["gsel"][rows_p] = np.where(valid_p, weight_gsel_p_2d * g_sel_pt, 0.0)

    ln1: dict[str, float] = {}
    for name in CONFIGS_1D:
        L1v = np.bincount(ev, weights=c1[name], minlength=n) / K
        ok = (L1v > 0.0) & np.isfinite(L1v)
        lnL = np.where(ok, np.log(np.where(ok, L1v, 1.0)), cg._LN_ZERO_EVENT)
        ln1[name] = float(np.sum(lnL)) - float(n) * gctx.cl_ctx.log_alpha[k]
    ln1["gsel"] = ln1["afull"]  # by construction, see docstring

    ln2: dict[str, float] = {}
    for name in CONFIGS_2D:
        L2v = np.bincount(ev, weights=c2[name], minlength=n) / K
        ok = (L2v > 0.0) & np.isfinite(L2v)
        lnL = np.where(ok, np.log(np.where(ok, L2v, 1.0)), cg._LN_ZERO_EVENT)
        ln2[name] = float(np.sum(lnL)) - float(n) * gctx.cl_ctx.log_alpha[k]

    return ln1, ln2


# ── worker plumbing (fork start method; context built in the parent) ────────

_CTX: vt.VenueContext | None = None


def _seed_task(seed: int) -> dict[str, Any]:
    """One seed: draw once, mirror c1/c2 at k=20,22 for every config."""
    assert _CTX is not None
    vctx = _CTX
    universe, ball, sigma_pairs = vt._draw_seed_realization(seed, vctx)
    out: dict[str, Any] = {"seed": seed}
    for k, tag in ((K_LO, "lo"), (K_HI, "hi")):
        ln1, ln2 = channel_terms_all_configs(vctx, universe, ball, sigma_pairs, k)
        for name in CONFIGS_1D:
            out[f"ln1_{tag}_{name}"] = ln1[name]
        out[f"ln1_{tag}_gsel"] = ln1["gsel"]
        for name in CONFIGS_2D:
            out[f"ln2_{tag}_{name}"] = ln2[name]
    return out


def _mean_se(vals: list[float]) -> tuple[float, float]:
    arr = np.asarray(vals, dtype=np.float64)
    return float(np.mean(arr)), float(np.std(arr, ddof=1) / np.sqrt(arr.size))


# ── validation gates ──────────────────────────────────────────────────────


def validate_afull_direct(
    vctx: vt.VenueContext, pool_rows_by_seed: dict[int, dict[str, Any]], seeds: list[int]
) -> dict[str, float]:
    """Cross-validate the a_full mirror against a direct venue_transfer call.

    Reuses the pool's already-computed ``ln1_*_afull``/``ln2_*_afull`` values
    (the pool runs the SAME mirror as ``channel_terms_all_configs``) instead
    of recomputing them here -- recomputing would re-pay the full
    ``a_full_gsel`` Hermite-survival cost for these seeds a second time
    (``channel_terms_all_configs`` always computes all three configs
    together), which is wasted work since the pool already has it. Only the
    DIRECT venue_transfer call (cheap -- no Hermite-survival folding) is done
    here, fresh, against a re-drawn realization (``_draw_seed_realization``
    is a deterministic, cheap draw given the seed).
    """
    max_diff_ln1 = 0.0
    max_diff_ln2 = 0.0
    for seed in seeds:
        universe, ball, sigma_pairs = vt._draw_seed_realization(seed, vctx)
        ln1_direct, ln2_direct, _ = vt.log_channel_posteriors_ball_sigma_vector(
            vctx.gctx,
            universe,
            ball,
            sigma_pairs,
            chunk_pairs=vctx.vcfg.chunk_pairs,
            estimator_variant=vt.ESTIMATOR_VARIANT_A_FULL,
        )
        r = pool_rows_by_seed[seed]
        max_diff_ln1 = max(
            max_diff_ln1,
            abs(r["ln1_lo_afull"] - ln1_direct[K_LO]),
            abs(r["ln1_hi_afull"] - ln1_direct[K_HI]),
        )
        max_diff_ln2 = max(
            max_diff_ln2,
            abs(r["ln2_lo_afull"] - ln2_direct[K_LO]),
            abs(r["ln2_hi_afull"] - ln2_direct[K_HI]),
        )
    return {"max_abs_diff_ln1": max_diff_ln1, "max_abs_diff_ln2": max_diff_ln2}


def validate_S_equals_one(vctx: vt.VenueContext, seed: int) -> dict[str, float]:
    """S=1 refactor check: g_sel(force_S_one=True) vs completion_mass_factor_g.

    Uses a small sample of nodes/events (one seed's first chunk, kernel-branch
    rows only) at k=K_LO -- full-order (n_hermite=64), non-adaptive on BOTH
    sides, per the task brief's "bump n_hermite to force the pinned path"
    recipe (``completion_mass_factor_g``'s adaptive switch only fires when
    ``n_hermite == _G_I_HERMITE_NODES`` AND ``adaptive`` is left at its
    default; passing ``adaptive=False`` explicitly is the cleaner, exact way
    to pin it, used here instead of bumping the order).
    """
    gctx = vctx.gctx
    h = gctx.cl_ctx.config.h_grid[K_LO]
    universe, ball, sigma_pairs = vt._draw_seed_realization(seed, vctx)
    ev = ball.event_idx
    z_obs = ball.z_obs
    d_obs_e = universe.d_L_obs
    sig_e = universe.sigma_dL
    d_obs_p = d_obs_e[ev]
    sig_p = sig_e[ev]
    x = gctx.cl_ctx.gl_nodes

    d_L_nodes, z_tab = gctx.cl_ctx.z_of_dl_tables[K_LO]
    z_hi_e = np.interp(d_obs_e * (1.0 + cl._SIGMA_WINDOW * sig_e), d_L_nodes, z_tab)
    z_lo_e = np.interp(d_obs_e * (1.0 - cl._SIGMA_WINDOW * sig_e), d_L_nodes, z_tab)
    z_lo_e = np.maximum(z_lo_e, 1e-6)
    z_hi_e = np.minimum(z_hi_e, z_tab[-1])
    z_lo_p = z_lo_e[ev]
    z_hi_p = z_hi_e[ev]

    sig_c = sigma_pairs[:2000]
    q = sig_c > 0.0
    rows_q = np.arange(2000)[q][:200]  # a modest sample: <=200 kernel-branch rows
    zo = z_obs[rows_q]
    so = sigma_pairs[rows_q]
    a = np.maximum(z_lo_p[rows_q], zo - cg._IMPOSTOR_KERNEL_WINDOW * so)
    b = np.minimum(z_hi_p[rows_q], zo + cg._IMPOSTOR_KERNEL_WINDOW * so)
    half = 0.5 * (b - a)
    mid = 0.5 * (b + a)
    z_nodes = mid[:, None] + half[:, None] * x[None, :]
    z_nodes_floor = np.maximum(z_nodes.reshape(-1), 1e-8)
    d_L_n = np.asarray(dist_vectorized(z_nodes_floor, h=h), dtype=np.float64).reshape(
        z_nodes.shape
    )
    d_L_frac = d_L_n / d_obs_p[rows_q][:, None]

    s_dd = universe.sigma_dL**2
    s_dm = universe.rho * universe.sigma_dL * universe.sigma_Mz
    s_mm = universe.sigma_Mz**2
    proj = np.where(s_dd > 0.0, s_dm / np.maximum(s_dd, 1e-300), 0.0)
    sigma_cond = np.sqrt(np.maximum(s_mm - proj * s_dm, 1e-30))
    n_hermite = gctx.cl_ctx.config.n_hermite

    max_rel_diff = 0.0
    for row, i in enumerate(ev[rows_q]):
        zz = z_nodes[row]
        ff = d_L_frac[row]
        ref = bs.completion_mass_factor_g(
            zz,
            ff,
            float(universe.M_z_obs[i]),
            float(proj[i]),
            float(sigma_cond[i]),
            n_hermite=n_hermite,
            adaptive=False,
        )
        mine = g_sel_mass_factor(
            zz,
            ff,
            d_L_n[row],
            float(universe.M_z_obs[i]),
            float(proj[i]),
            float(sigma_cond[i]),
            h,
            gctx.cl_ctx.detection,
            n_hermite=n_hermite,
            force_S_one=True,
        )
        denom = np.maximum(np.abs(ref), 1e-300)
        rel = np.max(np.abs(mine - ref) / denom)
        max_rel_diff = max(max_rel_diff, float(rel))

    return {"n_events_sampled": int(len(set(ev[rows_q].tolist()))), "max_rel_diff": max_rel_diff}


def main() -> None:
    mn0x = _load_json(RESULTS_DIR / "MN0X_h0p730_results_seeds0_100.json")
    per_seed = mn0x["per_seed"][:N_SEEDS]
    seeds = [int(r["seed"]) for r in per_seed]
    h_grid = np.asarray(mn0x["config"]["h_grid"], dtype=np.float64)
    dh = float(h_grid[K_HI] - h_grid[K_LO])
    stored = {
        int(r["seed"]): (
            float(r["ln_post_1d"][K_LO]),
            float(r["ln_post_1d"][K_HI]),
            float(r["ln_post_2d"][K_LO]),
            float(r["ln_post_2d"][K_HI]),
        )
        for r in per_seed
    }

    print("building context (full dose) ...", flush=True)
    vctx_full, a_lo, a_hi = build_population_context()
    assert (a_lo, a_hi) == (K_LO, K_HI)

    print("validation: S=1 refactor check ...", flush=True)
    s1_check = validate_S_equals_one(vctx_full, seeds[0])
    print(f"  n_events_sampled={s1_check['n_events_sampled']} "
          f"max_rel_diff={s1_check['max_rel_diff']:.3e}", flush=True)

    global _CTX
    _CTX = vctx_full

    print(f"running {N_SEEDS} seed tasks (c1+c2 mirror x3 configs x2 k) ...", flush=True)
    with mp.get_context("fork").Pool(processes=min(8, mp.cpu_count())) as pool:
        rows = pool.map(_seed_task, seeds)

    pool_rows_by_seed = {int(r["seed"]): r for r in rows}
    print(f"validation: a_full direct-call cross-check ({N_VALIDATION_SEEDS} seeds, "
          "reusing pool's a_full values, direct call only recomputed) ...", flush=True)
    afull_check = validate_afull_direct(
        vctx_full, pool_rows_by_seed, seeds[:N_VALIDATION_SEEDS]
    )
    print(f"  max_abs_diff_ln1={afull_check['max_abs_diff_ln1']:.3e} "
          f"max_abs_diff_ln2={afull_check['max_abs_diff_ln2']:.3e}", flush=True)

    # base-vs-stored validation (as l6_c2_switch_decomposition.py)
    max_diff_ln1_base = 0.0
    max_diff_ln2_base = 0.0
    max_diff_ln1_gsel_afull = 0.0  # c1 bit-identity, gsel vs afull
    for r in rows:
        slo1, shi1, slo2, shi2 = stored[r["seed"]]
        max_diff_ln1_base = max(
            max_diff_ln1_base, abs(r["ln1_lo_base"] - slo1), abs(r["ln1_hi_base"] - shi1)
        )
        max_diff_ln2_base = max(
            max_diff_ln2_base, abs(r["ln2_lo_base"] - slo2), abs(r["ln2_hi_base"] - shi2)
        )
        for tag in ("lo", "hi"):
            max_diff_ln1_gsel_afull = max(
                max_diff_ln1_gsel_afull,
                abs(r[f"ln1_{tag}_gsel"] - r[f"ln1_{tag}_afull"]),
            )
    print(f"validation: max |base mirror - stored ln_post_1d| = {max_diff_ln1_base:.3e}")
    print(f"validation: max |base mirror - stored ln_post_2d| = {max_diff_ln2_base:.3e}")
    print(f"validation: c1 bit-identity (gsel vs afull) max diff = "
          f"{max_diff_ln1_gsel_afull:.3e}")
    assert max_diff_ln1_gsel_afull == 0.0, "a_full_gsel c1 not bit-identical to a_full's c1"

    per_seed_out = []
    for r in rows:
        row_out: dict[str, Any] = {"seed": r["seed"]}
        for name in CONFIGS_1D:
            row_out[f"T1_{name}"] = (r[f"ln1_hi_{name}"] - r[f"ln1_lo_{name}"]) / dh
        row_out["T1_gsel"] = row_out["T1_afull"]
        for name in CONFIGS_2D:
            row_out[f"T2_{name}"] = (r[f"ln2_hi_{name}"] - r[f"ln2_lo_{name}"]) / dh
        for name in CONFIGS_2D:
            row_out[f"excess_{name}"] = row_out[f"T2_{name}"] - row_out[f"T1_{name}"]
        row_out["dT2_gsel_vs_afull"] = row_out["T2_gsel"] - row_out["T2_afull"]
        row_out["d_excess_gsel_vs_afull"] = row_out["excess_gsel"] - row_out["excess_afull"]
        per_seed_out.append(row_out)

    aggregates: dict[str, Any] = {}
    agg_keys = (
        ["T1_base", "T1_afull", "T1_gsel"]
        + [f"T2_{n}" for n in CONFIGS_2D]
        + [f"excess_{n}" for n in CONFIGS_2D]
        + ["dT2_gsel_vs_afull", "d_excess_gsel_vs_afull"]
    )
    for key in agg_keys:
        m, se = _mean_se([row[key] for row in per_seed_out])
        aggregates[key] = {"mean": m, "se": se}

    results: dict[str, Any] = {
        "note": (
            "L6-DER2 SS4 step 1: mirror pre-measurement of the fused g_sel "
            "A-FULL-2D candidate (ledger rows #112/#114). Tilts nats/h, "
            "grid-neighbour central difference at h=0.730 (k=20 h=0.725, "
            "k=22 h=0.735), 15 MN0X seed replays, full dose. Non-adaptive "
            "n_hermite=64 for g_sel (convention choice, see module "
            "docstring). PRESENTED, NOT ADJUDICATED -- reported as measured, "
            "not tuned toward the registered prediction."
        ),
        "seeds": seeds,
        "n_seeds": N_SEEDS,
        "n_validation_seeds": N_VALIDATION_SEEDS,
        "k_lo": K_LO,
        "k_hi": K_HI,
        "validation": {
            "base_vs_stored": {
                "max_abs_diff_ln1": max_diff_ln1_base,
                "max_abs_diff_ln2": max_diff_ln2_base,
            },
            "afull_vs_direct_call": afull_check,
            "c1_bit_identity_gsel_vs_afull": max_diff_ln1_gsel_afull,
            "S_equals_one_refactor_check": s1_check,
        },
        "registered_prediction": REGISTERED_PREDICTION_NOTE,
        "aggregates": aggregates,
        "per_seed_rows": per_seed_out,
    }

    out_path = RESULTS_DIR / "L6_DER2_GSEL_PREMEASURE_output.json"
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"wrote {out_path}")

    print("\n=== summary (nats/h), PRESENTED NOT ADJUDICATED ===")
    for name in CONFIGS_2D:
        t1_key = "T1_gsel" if name == "gsel" else f"T1_{name}"
        t1 = aggregates[t1_key]
        t2 = aggregates[f"T2_{name}"]
        ex = aggregates[f"excess_{name}"]
        print(f"[{name}] T1: {t1['mean']:+.1f} +- {t1['se']:.1f}  "
              f"T2: {t2['mean']:+.1f} +- {t2['se']:.1f}  "
              f"excess(T2-T1): {ex['mean']:+.1f} +- {ex['se']:.1f}")
    dT2 = aggregates["dT2_gsel_vs_afull"]
    dex = aggregates["d_excess_gsel_vs_afull"]
    print(f"dT2(gsel vs afull): {dT2['mean']:+.1f} +- {dT2['se']:.1f}")
    print(f"d_excess(gsel vs afull): {dex['mean']:+.1f} +- {dex['se']:.1f}")
    print(f"\nregistered prediction (qualitative): {REGISTERED_PREDICTION_NOTE}")


if __name__ == "__main__":
    main()
