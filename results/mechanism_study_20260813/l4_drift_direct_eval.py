"""L4 drift-term direct evaluation (ledger row #109 item 1, the registered recompute).

Evaluates the Part-2 SS2 closed-form score decomposition INDEPENDENTLY of the switch
machinery (no quadrature, no mirror): per candidate, the two-Gaussian model

    c1_k(h) ~= (sigma_gw/sigma_d) * N(z_obs,k ; z*(h), s_k),  s_k^2 = sigma_k^2 + sigma_gw^2

gives per-event score components at h_true (responsibilities r_k from the closed form):

    mass  : G_e = (1/h)(1 - D D''/D'^2)                       (candidate-independent)
    drift : sum_k r_k (z_obs,k - z*) / s_k^2 * dz*/dh          (the T_res candidate)
    width : sum_k r_k [ -sg*sg' / s_k^2 + (z_obs,k - z*)^2 * sg*sg' / s_k^4 ],  sg' = sg*G_e

on the SAME 15 seed realizations x 3 dose levels the switch decomposition used.  The model
is deliberately clip-free and linearized, so its total is expected to differ from the
mirror's T_cand by the window/exponent-scale/higher-order shares; the decisive comparison
is **drift(f) vs the switch leftover (+866.7 / +344.1 / +39.1)** — hardening (or breaking)
the ratified-with-hedge identification T_res == leftover(drift + interactions).

Output: ``L4_DRIFT_EVAL_output.json``.  Report: ``L4_DRIFT_EVAL_20260815.md``.
Status: PRESENTED, NOT ADJUDICATED.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

RESULTS_DIR = Path(__file__).parent
sys.path.insert(0, str(RESULTS_DIR))

from l4_der_part2_switch_decomposition import (  # noqa: E402
    DOSE_LEVELS,
    H_TRUE,
    N_SEEDS,
    _load_json,
    build_dose_context,
)
from l4_t2_audit import build_population_context  # noqa: E402

from darksiren_emri.physical_relations import dist_vectorized  # noqa: E402
from darksiren_emri.validation import venue_transfer as vt  # noqa: E402

LEFTOVER_REFERENCE = {"0.25": 866.68, "0.5": 344.15, "1.0": 39.15}  # switch decomposition


def _d_derivs(z: npt.NDArray[np.float64]) -> tuple[npt.NDArray[np.float64], ...]:
    """D(z) = dist(z, h=1) and its first two z-derivatives (central differences)."""
    eps = 1e-5
    d0 = np.asarray(dist_vectorized(z, h=1.0), dtype=np.float64)
    dp = (
        np.asarray(dist_vectorized(z + eps, h=1.0), dtype=np.float64)
        - np.asarray(dist_vectorized(np.maximum(z - eps, 1e-8), h=1.0), dtype=np.float64)
    ) / (2.0 * eps)
    dpp = (
        np.asarray(dist_vectorized(z + eps, h=1.0), dtype=np.float64)
        - 2.0 * d0
        + np.asarray(dist_vectorized(np.maximum(z - eps, 1e-8), h=1.0), dtype=np.float64)
    ) / eps**2
    return d0, dp, dpp


def seed_components(vctx: vt.VenueContext, seed: int, i_true: int) -> dict[str, float]:
    """Closed-form per-seed score components, summed over the 982 events."""
    universe, ball, sig_z = vt._draw_seed_realization(seed, vctx)
    gctx = vctx.gctx
    d_L_nodes, z_tab = gctx.cl_ctx.z_of_dl_tables[i_true]
    d_obs = np.asarray(universe.d_L_obs, dtype=np.float64)
    sig_d = np.asarray(universe.sigma_dL, dtype=np.float64)
    z_star = np.interp(d_obs, d_L_nodes, z_tab)
    d0, dp, dpp = _d_derivs(z_star)
    g_e = (1.0 - d0 * dpp / dp**2) / H_TRUE
    dzstar_dh = d_obs / dp
    sigma_gw = sig_d * d0 / dp

    ev = ball.event_idx
    zo = np.asarray(ball.z_obs, dtype=np.float64)
    sk = np.asarray(sig_z, dtype=np.float64)
    zs_p = z_star[ev]
    sg_p = sigma_gw[ev]
    s2 = sk**2 + sg_p**2
    # Closed-form candidate weight (event-common factors sigma_gw/sigma_d drop out of r_k).
    w = np.exp(-0.5 * (zo - zs_p) ** 2 / s2) / np.sqrt(s2)
    n = int(universe.z_true.size)
    w_sum = np.bincount(ev, weights=w, minlength=n)
    r = w / np.maximum(w_sum[ev], 1e-300)

    drift_k = r * (zo - zs_p) / s2 * dzstar_dh[ev]
    sg_prime = sg_p * g_e[ev]
    width_k = r * (-sg_p * sg_prime / s2 + (zo - zs_p) ** 2 * sg_p * sg_prime / s2**2)

    return {
        "mass": float(np.sum(g_e)),
        "drift": float(np.sum(drift_k)),
        "width": float(np.sum(width_k)),
    }


def main() -> None:
    mn0x = _load_json(RESULTS_DIR / "MN0X_h0p730_results_seeds0_100.json")
    seeds = [int(r["seed"]) for r in mn0x["per_seed"][:N_SEEDS]]
    h_grid = np.asarray(mn0x["config"]["h_grid"], dtype=np.float64)
    i_true = int(np.argmin(np.abs(h_grid - H_TRUE)))

    part2 = _load_json(RESULTS_DIR / "L4_DER_PART2_output.json")
    alpha_numeric = float(part2["alpha_tilt_numeric"])
    t_base_by_dose_seed = {
        (r["dose"], r["seed"]): float(r["T_base"]) for r in part2["per_seed_rows"]
    }

    print("building contexts ...", flush=True)
    ctxs = {"1.0": build_population_context()[0]}
    for lab, val in (("0.25", 0.25), ("0.5", 0.5)):
        ctxs[lab] = build_dose_context(val)

    results: dict[str, Any] = {
        "note": (
            "Closed-form direct evaluation of the Part-2 SS2 score decomposition at h_true "
            "(no quadrature; clip-free linearized model). Decisive line: drift(f) vs the "
            "switch-decomposition leftover. PRESENTED, NOT ADJUDICATED."
        ),
        "seeds": seeds,
        "leftover_reference": LEFTOVER_REFERENCE,
        "by_dose": {},
        "per_seed": [],
    }
    for dose in DOSE_LEVELS:
        comp_rows = []
        for s in seeds:
            c = seed_components(ctxs[dose], s, i_true)
            c["seed"] = s
            c["dose"] = dose
            c["model_total"] = c["mass"] + c["drift"] + c["width"]
            c["T_cand_mirror"] = t_base_by_dose_seed[(dose, s)] - alpha_numeric
            comp_rows.append(c)
            results["per_seed"].append(c)
        arr = {
            k: np.array([r[k] for r in comp_rows])
            for k in ("mass", "drift", "width", "model_total", "T_cand_mirror")
        }
        blk = {
            k: {"mean": float(v.mean()), "se": float(v.std(ddof=1) / np.sqrt(v.size))}
            for k, v in arr.items()
        }
        blk["leftover_reference"] = LEFTOVER_REFERENCE[dose]
        blk["drift_minus_leftover"] = blk["drift"]["mean"] - LEFTOVER_REFERENCE[dose]
        results["by_dose"][dose] = blk
        print(
            f"f_i={dose}: mass {blk['mass']['mean']:+.1f} | drift {blk['drift']['mean']:+.1f}"
            f"±{blk['drift']['se']:.1f} (leftover ref {LEFTOVER_REFERENCE[dose]:+.1f}) | "
            f"width {blk['width']['mean']:+.1f} | model total {blk['model_total']['mean']:+.1f} "
            f"vs mirror T_cand {blk['T_cand_mirror']['mean']:+.1f}",
            flush=True,
        )

    out = RESULTS_DIR / "L4_DRIFT_EVAL_output.json"
    with open(out, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
