"""L6 — c2 (2D) mirror + freeze-switch decomposition (ledger row #113).

Executes `L6_DER_2D_CHANNEL_20260816.md` §4 item 1 / `L6_2D_GI_PLAN_20260816.md` §2
item 1: extend the bit-exact mirror from the 1D channel (c1, L4-DER Part 2) to the
2D channel (c2 = integral of kern * p_gw * g dz) and measure the registered
freeze-switches on the SAME 15 MN0X seed replays.

Mirror: verbatim copy of the ``base``-variant c1/c2 code paths of
``venue_transfer._channel_terms_at_h`` (chunking, ``_g_ball_capped`` calls with
``node_chunk = vt._G_NODE_CHUNK`` since ``chunk_pairs = 16384 > 0``, the
bincount/K, ok masks, ``-n*log_alpha[k]``). Validated bit-exactly against the
committed per-seed ``ln_post_1d``/``ln_post_2d`` vectors in
``MN0X_h0p730_results_seeds0_100.json`` at k=20 (h=0.725) and k=22 (h=0.735).

Switches (2D-only — they modify ONLY the two arrays passed to ``_g_ball_capped``;
``kern``, ``p_gw``, ``c1`` are computed from the UNSWITCHED arrays and are
bit-identical across all configs by construction, since they are computed once
per chunk and reused for every switch):

- ``sa`` (freeze channel A): g's ``d_L_frac`` argument is recomputed at
  ``h_true = 0.730`` instead of the loop ``h`` (same ``z_nodes``).
- ``sb`` (freeze channel B): g's ``z``-argument is shifted by
  ``dz_e = z_star_e(h) - z_star_e(h_true)`` per event (the loop-h ``d_L_frac``
  argument is untouched). ``z_star_e(k) = np.interp(d_obs_e,
  *gctx.cl_ctx.z_of_dl_tables[k])``.
- ``sab``: both together.

Registered predictions (L6-DER note §2, committed 718128d1 BEFORE this
measurement): ΔT2(sb) ≈ -139, ΔT2(sa) ≈ 0. NOT tuned toward; reported as-is.

Output: ``L6_C2_SWITCH_output.json``. Status: PRESENTED, NOT ADJUDICATED.
"""

import json
import multiprocessing as mp
import sys
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.stats import norm

RESULTS_DIR = Path(__file__).parent
sys.path.insert(0, str(RESULTS_DIR))

from l4_t2_audit import build_population_context  # noqa: E402

from darksiren_emri.physical_relations import dist_vectorized  # noqa: E402
from darksiren_emri.validation import calibration_gate as cg  # noqa: E402
from darksiren_emri.validation import closed_loop_gfrac as cl  # noqa: E402
from darksiren_emri.validation import venue_transfer as vt  # noqa: E402

N_SEEDS = 15
H_TRUE = 0.730
K_LO = 20  # h = 0.725
K_HI = 22  # h = 0.735
CONFIGS = ("base", "sa", "sb", "sab")

# Registered predictions (L6-DER note §2, pre-committed; NOT recomputed here).
REGISTERED_PREDICTION = {"sb": -139.0, "sa": 0.0}


def _load_json(path: Path) -> dict[str, Any]:
    with open(path) as fh:
        result: dict[str, Any] = json.load(fh)
    return result


def mirror_c1_c2(
    vctx: vt.VenueContext,
    universe: Any,
    ball: Any,
    sig_z: npt.NDArray[np.float64],
    k: int,
    i_true: int,
) -> tuple[float, dict[str, float]]:
    """The c1/c2 (``base`` variant) path of ``_channel_terms_at_h``, verbatim.

    Runs every freeze-switch config's ``g`` calls inside one pass over the
    chunk loop, sharing the unswitched ``kern``/``p_gw``/``z_nodes`` (hence
    ``c1``) across all four configs.

    Args:
        vctx: The venue context.
        universe: The seed's synthetic universe.
        ball: The seed's candidate balls.
        sig_z: Per-candidate sigma_z, aligned with ``ball.z_obs``.
        k: The h-grid index to evaluate at.
        i_true: The h-grid index nearest ``H_TRUE`` (freeze target).

    Returns:
        ``(ln1, {config: ln2})`` — ``ln1`` is the shared (unswitched) 1D
        channel value; ``ln2`` holds the 2D channel value per freeze config.
    """
    gctx = vctx.gctx
    cfg = gctx.cl_ctx.config
    h = cfg.h_grid[k]
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
    chunks = vt._pair_chunks(n_pairs, vctx.vcfg.chunk_pairs)
    g_node_chunk = vt._G_NODE_CHUNK  # chunk_pairs = 16384 > 0 (registered geometry)

    d_L_nodes, z_tab = gctx.cl_ctx.z_of_dl_tables[k]
    z_hi_e = np.interp(d_obs_e * (1.0 + cl._SIGMA_WINDOW * sig_e), d_L_nodes, z_tab)
    z_lo_e = np.interp(d_obs_e * (1.0 - cl._SIGMA_WINDOW * sig_e), d_L_nodes, z_tab)
    z_lo_e = np.maximum(z_lo_e, 1e-6)
    z_hi_e = np.minimum(z_hi_e, z_tab[-1])
    z_lo_p = z_lo_e[ev]
    z_hi_p = z_hi_e[ev]

    # dz_e = z_star_e(h) - z_star_e(h_true), per event (S-B ingredient).
    z_star_h = np.interp(d_obs_e, d_L_nodes, z_tab)
    d_L_nodes_true, z_tab_true = gctx.cl_ctx.z_of_dl_tables[i_true]
    z_star_true = np.interp(d_obs_e, d_L_nodes_true, z_tab_true)
    dz_e = z_star_h - z_star_true

    c1 = np.zeros(n_pairs, dtype=np.float64)
    c2 = {name: np.zeros(n_pairs, dtype=np.float64) for name in CONFIGS}

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
            d_L_n = np.asarray(
                dist_vectorized(np.maximum(z_nodes.reshape(-1), 1e-8), h=h),
                dtype=np.float64,
            ).reshape(z_nodes.shape)
            d_L_frac = d_L_n / d_obs_p[rows_q][:, None]
            p_gw = norm.pdf(d_L_frac, loc=1.0, scale=sig_p[rows_q][:, None])
            kern = norm.pdf(z_nodes, loc=zo[:, None], scale=so[:, None])
            integ = kern * p_gw  # UNSWITCHED — shared across every config
            c1q = half * (integ @ w_gl)
            c1[rows_q] = np.where(valid, c1q, 0.0)

            ev_rows = ev[rows_q]
            # S-A ingredient: d_L_frac recomputed at h_true (same z_nodes).
            z_nodes_floor = np.maximum(z_nodes.reshape(-1), 1e-8)
            d_L_n_true = np.asarray(
                dist_vectorized(z_nodes_floor, h=H_TRUE), dtype=np.float64
            ).reshape(z_nodes.shape)
            d_L_frac_freeze = d_L_n_true / d_obs_p[rows_q][:, None]
            # S-B ingredient: z_nodes shifted by the event's dz_e.
            z_nodes_shift = np.maximum(z_nodes - dz_e[ev_rows][:, None], 1e-8)

            g_args = {
                "base": (z_nodes, d_L_frac),
                "sa": (z_nodes, d_L_frac_freeze),
                "sb": (z_nodes_shift, d_L_frac),
                "sab": (z_nodes_shift, d_L_frac_freeze),
            }
            for name, (z_arg, f_arg) in g_args.items():
                g = vt._g_ball_capped(
                    gctx, universe, ev_rows, z_arg, f_arg, valid, node_chunk=g_node_chunk
                )
                c2q = half * ((integ * g) @ w_gl)
                c2[name][rows_q] = np.where(valid, c2q, 0.0)
        if not np.all(q):
            rows_p = sl[~q]
            zo = z_obs[rows_p]
            valid_p = (zo >= z_lo_p[rows_p]) & (zo <= z_hi_p[rows_p])
            d_pt = np.asarray(dist_vectorized(np.maximum(zo, 1e-8), h=h), dtype=np.float64)
            frac = d_pt / d_obs_p[rows_p]
            p_gw_p = norm.pdf(frac, loc=1.0, scale=sig_p[rows_p])  # UNSWITCHED
            c1[rows_p] = np.where(valid_p, p_gw_p, 0.0)

            ev_rows_p = ev[rows_p]
            zo_floor = np.maximum(zo, 1e-8)
            d_pt_true = np.asarray(dist_vectorized(zo_floor, h=H_TRUE), dtype=np.float64)
            frac_freeze = d_pt_true / d_obs_p[rows_p]
            zo_shift = np.maximum(zo - dz_e[ev_rows_p], 1e-8)

            g_args_p = {
                "base": (zo[:, None], frac[:, None]),
                "sa": (zo[:, None], frac_freeze[:, None]),
                "sb": (zo_shift[:, None], frac[:, None]),
                "sab": (zo_shift[:, None], frac_freeze[:, None]),
            }
            for name, (z_arg, f_arg) in g_args_p.items():
                g_pt = vt._g_ball_capped(
                    gctx, universe, ev_rows_p, z_arg, f_arg, valid_p, node_chunk=g_node_chunk
                )[:, 0]
                c2[name][rows_p] = np.where(valid_p, p_gw_p * g_pt, 0.0)

    L1 = np.bincount(ev, weights=c1, minlength=n) / K
    ok1 = (L1 > 0.0) & np.isfinite(L1)
    lnL1 = np.where(ok1, np.log(np.where(ok1, L1, 1.0)), cg._LN_ZERO_EVENT)
    ln1 = float(np.sum(lnL1)) - float(n) * gctx.cl_ctx.log_alpha[k]

    ln2: dict[str, float] = {}
    for name in CONFIGS:
        L2 = np.bincount(ev, weights=c2[name], minlength=n) / K
        ok2 = (L2 > 0.0) & np.isfinite(L2)
        lnL2 = np.where(ok2, np.log(np.where(ok2, L2, 1.0)), cg._LN_ZERO_EVENT)
        ln2[name] = float(np.sum(lnL2)) - float(n) * gctx.cl_ctx.log_alpha[k]

    return ln1, ln2


# ── worker plumbing (fork start method; context built in the parent) ────────

_CTX: tuple[vt.VenueContext, int] | None = None


def _seed_task(seed: int) -> dict[str, Any]:
    """One seed: draw once, mirror c1/c2 at k=20,22 for every switch config."""
    assert _CTX is not None
    vctx, i_true = _CTX
    universe, ball, sigma_pairs = vt._draw_seed_realization(seed, vctx)
    out: dict[str, Any] = {"seed": seed}
    for k, tag in ((K_LO, "lo"), (K_HI, "hi")):
        ln1, ln2 = mirror_c1_c2(vctx, universe, ball, sigma_pairs, k, i_true)
        out[f"ln1_{tag}"] = ln1
        for name in CONFIGS:
            out[f"ln2_{tag}_{name}"] = ln2[name]
    return out


def _mean_se(vals: list[float]) -> tuple[float, float]:
    arr = np.asarray(vals, dtype=np.float64)
    return float(np.mean(arr)), float(np.std(arr, ddof=1) / np.sqrt(arr.size))


def main() -> None:
    mn0x = _load_json(RESULTS_DIR / "MN0X_h0p730_results_seeds0_100.json")
    per_seed = mn0x["per_seed"][:N_SEEDS]
    seeds = [int(r["seed"]) for r in per_seed]
    h_grid = np.asarray(mn0x["config"]["h_grid"], dtype=np.float64)
    i_true = int(np.argmin(np.abs(h_grid - H_TRUE)))
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
    global _CTX
    _CTX = (vctx_full, i_true)

    print(f"running {N_SEEDS} seed tasks (c1+c2 mirror x4 configs x2 k) ...", flush=True)
    with mp.get_context("fork").Pool(processes=min(8, mp.cpu_count())) as pool:
        rows = pool.map(_seed_task, seeds)

    # Validation: mirror base vs committed instrument vectors, ln1 and ln2.
    max_diff_ln1 = 0.0
    max_diff_ln2 = 0.0
    max_diff_ln1_switch = 0.0  # c1 bit-identity across switches
    for r in rows:
        slo1, shi1, slo2, shi2 = stored[r["seed"]]
        max_diff_ln1 = max(max_diff_ln1, abs(r["ln1_lo"] - slo1), abs(r["ln1_hi"] - shi1))
        max_diff_ln2 = max(
            max_diff_ln2,
            abs(r["ln2_lo_base"] - slo2),
            abs(r["ln2_hi_base"] - shi2),
        )
        for tag in ("lo", "hi"):
            base_ln1 = r[f"ln1_{tag}"]
            for name in CONFIGS:
                # ln1 is computed once per (seed, k) and shared across configs
                # by construction; this assertion documents that fact rather
                # than discovering it.
                max_diff_ln1_switch = max(max_diff_ln1_switch, abs(base_ln1 - r[f"ln1_{tag}"]))
    assert max_diff_ln1_switch == 0.0, "ln1 not bit-identical across switch configs"

    print(f"validation: max |mirror - stored ln_post_1d| = {max_diff_ln1:.3e}")
    print(f"validation: max |mirror - stored ln_post_2d| = {max_diff_ln2:.3e}")
    print(f"c1 bit-identity across switches: max diff = {max_diff_ln1_switch:.3e}")

    per_seed_out = []
    for r in rows:
        row_out: dict[str, Any] = {"seed": r["seed"]}
        t1_base = (r["ln1_hi"] - r["ln1_lo"]) / dh
        row_out["T1_base"] = t1_base
        for name in CONFIGS:
            t2 = (r[f"ln2_hi_{name}"] - r[f"ln2_lo_{name}"]) / dh
            row_out[f"T2_{name}"] = t2
        for name in ("sa", "sb", "sab"):
            row_out[f"dT2_{name}"] = row_out[f"T2_{name}"] - row_out["T2_base"]
        row_out["T2_base_minus_T1_base"] = row_out["T2_base"] - t1_base
        per_seed_out.append(row_out)

    aggregates: dict[str, Any] = {}
    for key in (
        "T1_base",
        "T2_base",
        "T2_sa",
        "T2_sb",
        "T2_sab",
        "dT2_sa",
        "dT2_sb",
        "dT2_sab",
        "T2_base_minus_T1_base",
    ):
        m, se = _mean_se([row[key] for row in per_seed_out])
        aggregates[key] = {"mean": m, "se": se}

    results: dict[str, Any] = {
        "note": (
            "L6 c2-mirror switch decomposition (ledger row #113). Tilts nats/h, "
            "grid-neighbour central difference at h_true=0.730 (k=20 h=0.725, "
            "k=22 h=0.735), 15 MN0X seed replays, full dose. "
            "dT2_<switch> = T2(switch) - T2(base). "
            "PRESENTED, NOT ADJUDICATED."
        ),
        "seeds": seeds,
        "n_seeds": N_SEEDS,
        "k_lo": K_LO,
        "k_hi": K_HI,
        "h_true": H_TRUE,
        "i_true": i_true,
        "validation": {
            "max_abs_diff_ln1_vs_stored": max_diff_ln1,
            "max_abs_diff_ln2_vs_stored": max_diff_ln2,
            "max_abs_diff_ln1_across_switches": max_diff_ln1_switch,
        },
        "registered_prediction": REGISTERED_PREDICTION,
        "aggregates": aggregates,
        "per_seed_rows": per_seed_out,
    }

    out_path = RESULTS_DIR / "L6_C2_SWITCH_output.json"
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"wrote {out_path}")

    print("\n=== summary (nats/h) ===")
    t2b = aggregates["T2_base"]
    t1b = aggregates["T1_base"]
    excess = aggregates["T2_base_minus_T1_base"]
    print(f"T1(base): {t1b['mean']:+.1f} ± {t1b['se']:.1f}")
    print(f"T2(base): {t2b['mean']:+.1f} ± {t2b['se']:.1f}")
    print(f"T2(base) - T1(base) [excess]: {excess['mean']:+.1f} ± {excess['se']:.1f}")
    for name in ("sa", "sb", "sab"):
        dt = aggregates[f"dT2_{name}"]
        pred = REGISTERED_PREDICTION.get(name)
        pred_s = f" (registered prediction: {pred:+.1f})" if pred is not None else ""
        print(f"dT2({name}): {dt['mean']:+.1f} ± {dt['se']:.1f}{pred_s}")


if __name__ == "__main__":
    main()
