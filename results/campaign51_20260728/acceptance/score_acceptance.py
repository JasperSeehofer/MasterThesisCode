"""#51 campaign pool acceptance scoring — pre-registered criteria.

Rebuilds the FIX-3 §7.1 joint (u, m) with-BH survival grid from the DELIVERED
pool via the production estimator (SimulationDetectionProbability,
pdet_z_resolved=True, pdet_wbh_z_resolved=True) and scores the pre-registered
acceptance criteria (docs/campaign_redesign_51_design.md §4;
SIZING_ANALYSIS.md §6):

  1. catalogue-weighted median ESS >= 1000
  2. catalogue weight-fraction on ESS < 500 nodes <= 1 %
  3. reachable-weight w_bar = E_W[ESS/(ESS+10)] >= 0.99
     (unreachable ridge weight m > 7 + log10(1+z) exempt, reported separately)

Because the delivered estimator builds its m-axis from the pool's own range
with 31 nodes (_WBH_ZRES_M_NODES) while the sizing analysis assumed the
61 x 69 / 0.05-dex design grid on m in [4, 7.398], BOTH grids are scored:
  - "as_built":       the estimator's own _wbh_ess (61 x 31, pool m-range)
  - "design_projected": ESS recomputed on the design grid (61 x 69) with the
    identical kernel conventions (Scott d=2 N^(-1/6) std both axes, Abramson
    sqrt-law on u via the estimator's own _abramson_lambda_u pilot machinery,
    sigma_m = the estimator's _compute_bandwidths value) from the SAME
    delivered (u, m) sample (all 200,100 rows — measure-free joint leg).

Catalogue projection conventions are REUSED verbatim from
results/lcat_h_dependence_20260725/campaign_sizing_20260728/s1_sizing.py
(W_z_lm cell centres as query nodes, cell weight = W, grid-box clamp on m,
bilinear ESS interpolation, Kish shrinkage w = ESS/(ESS+10), n0 = 10).

Read-only w.r.t. master_thesis_code/.  CPU-only.
"""

import json
import logging
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/jasper/Repositories/MasterThesisCode")

BASE = "/home/jasper/Repositories/MasterThesisCode"
POOL = f"{BASE}/results/campaign51_20260728/pool_mix200k"
OUT = f"{BASE}/results/campaign51_20260728/acceptance"
PROFILE = f"{BASE}/results/lcat_h_dependence_20260725/zres_survival/catalog_zw_profile.json"

# capture the estimator's INFO build log (measure-match + ESS diagnostic lines)
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(f"{OUT}/build_log.txt", mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
)

from master_thesis_code.bayesian_inference.simulation_detection_probability import (  # noqa: E402
    SimulationDetectionProbability,
)
from master_thesis_code.constants import SNR_THRESHOLD  # noqa: E402

N0 = 10.0
# design grid (sizing s1_sizing.py conventions)
Z_MAX = 1.5
U_MAX_DESIGN = float(np.log1p(Z_MAX))  # = ln 2.5 (matches _WBH_ZRES_U_MAX)
M_LO, M_HI = 4.0, 7.0 + float(np.log10(2.5))
U_NODES_DESIGN = np.linspace(0.0, U_MAX_DESIGN, 61)
M_NODES_DESIGN = np.linspace(M_LO, M_HI, 69)


def ridge_m(z: np.ndarray) -> np.ndarray:
    return 7.0 + np.log10(1.0 + z)


def bilinear(ess, u_nodes, m_nodes, uq, mq):
    a = np.interp(uq, u_nodes, np.arange(len(u_nodes)))
    b = np.interp(mq, m_nodes, np.arange(len(m_nodes)))
    a0 = np.clip(np.floor(a).astype(int), 0, len(u_nodes) - 2)
    b0 = np.clip(np.floor(b).astype(int), 0, len(m_nodes) - 2)
    fa, fb = a - a0, b - b0
    return ((1 - fa) * (1 - fb) * ess[a0, b0] + fa * (1 - fb) * ess[a0 + 1, b0]
            + (1 - fa) * fb * ess[a0, b0 + 1] + fa * fb * ess[a0 + 1, b0 + 1])


def wmedian(x, w):
    o = np.argsort(x)
    cw = np.cumsum(w[o])
    return float(x[o][np.searchsorted(cw, 0.5 * cw[-1])])


def cat_metrics(ess, u_nodes, m_nodes, q_u, q_m_clamped, q_w, q_reach):
    e_q = bilinear(ess, u_nodes, m_nodes, q_u, q_m_clamped)
    w_shrink = e_q / (e_q + N0)
    out = {}
    for tag, mask in (("all_clamped", np.ones_like(q_reach)), ("reachable", q_reach)):
        w = q_w[mask]
        e = e_q[mask]
        s = w_shrink[mask]
        out[tag] = {
            "wbar": float(np.sum(w * s) / np.sum(w)),
            "median_ESS": wmedian(e, w),
            "min_ESS_at_queries": float(e.min()),
            "wfrac_ESS_lt_10": float(np.sum(w[e < 10]) / np.sum(w)),
            "wfrac_ESS_lt_100": float(np.sum(w[e < 100]) / np.sum(w)),
            "wfrac_ESS_lt_500": float(np.sum(w[e < 500]) / np.sum(w)),
            "wfrac_ESS_lt_1000": float(np.sum(w[e < 1000]) / np.sum(w)),
        }
    return out


def grid_metrics(ess, u_nodes, m_nodes):
    node_reach = m_nodes[None, :] <= (7.0 + u_nodes[:, None] / np.log(10.0))
    er = ess[node_reach]
    return {
        "n_nodes": int(ess.size),
        "n_nodes_reachable": int(node_reach.sum()),
        "min_ESS_all_nodes": float(ess.min()),
        "min_ESS_reachable": float(er.min()),
        "median_ESS_reachable": float(np.median(er)),
        "frac_reachable_ESS_lt_10": float((er < 10).mean()),
        "frac_reachable_ESS_lt_100": float((er < 100).mean()),
        "frac_reachable_ESS_lt_500": float((er < 500).mean()),
    }


def main() -> None:
    # ---------------- estimator build (production code path) ----------------
    sdp = SimulationDetectionProbability(
        injection_data_dir=POOL,
        snr_threshold=float(SNR_THRESHOLD),
        expected_z_max=1.5,
        pdet_z_resolved=True,
        pdet_wbh_z_resolved=True,
    )
    ess_built = np.asarray(sdp._wbh_ess)
    w_built = np.asarray(sdp._wbh_w)
    u_nodes_b = np.asarray(sdp._wbh_u_nodes)
    m_nodes_b = np.asarray(sdp._wbh_m_nodes)
    n_all = int(sdp._d_hor_all.size)
    n_a = int(sdp._d_hor.size)
    print(f"built grid: {len(u_nodes_b)} u x {len(m_nodes_b)} m; "
          f"u [{u_nodes_b.min():.4f},{u_nodes_b.max():.4f}] "
          f"m [{m_nodes_b.min():.4f},{m_nodes_b.max():.4f}]; "
          f"joint rows={n_all}, a rows={n_a}")

    # ---------------- pool-delivery facts ----------------
    df = sdp._pooled_df
    m_all = np.log10(df["M"].to_numpy(dtype=np.float64))
    z_all = df["z"].to_numpy(dtype=np.float64)
    snr_all = df["SNR"].to_numpy(dtype=np.float64)
    strat = df["stratum"].fillna("a").astype(str).to_numpy()
    facts = {
        "n_rows": int(len(df)),
        "n_files": 707,
        "per_stratum_rows": {s: int((strat == s).sum()) for s in sorted(set(strat))},
        "snr_detected_frac_overall": float((snr_all >= SNR_THRESHOLD).mean()),
        "snr_detected_frac_per_stratum": {
            s: float((snr_all[strat == s] >= SNR_THRESHOLD).mean())
            for s in sorted(set(strat))
        },
        "m_range_detector_frame_log10": [float(m_all.min()), float(m_all.max())],
        "z_range": [float(z_all.min()), float(z_all.max())],
        "h_inj_unique": sorted(float(h) for h in df["h_inj"].unique()),
        "z_cut_unique": sorted(float(z) for z in df["z_cut"].dropna().unique()),
        "code_rev_unique": sorted(str(c) for c in df["code_rev"].unique()),
        "n_rows_with_t_plunge_yr": int(df["t_plunge_yr"].notna().sum())
        if "t_plunge_yr" in df.columns else 0,
        "snr_threshold_used": float(SNR_THRESHOLD),
    }

    # ---------------- catalogue query projection (s1 conventions) ----------------
    prof = json.load(open(PROFILE))
    z_edges = np.array(prof["z_edges"])
    lm_edges = np.array(prof["lm_edges"])
    W_zlm = np.array(prof["W_z_lm"])
    zc = 0.5 * (z_edges[:-1] + z_edges[1:])
    lmc = 0.5 * (lm_edges[:-1] + lm_edges[1:])
    CZ, CM = np.meshgrid(zc, lmc, indexing="ij")
    sel = W_zlm > 0
    q_z = CZ[sel]
    q_m_raw = CM[sel]
    q_w = W_zlm[sel]
    q_u = np.log1p(q_z)
    q_reach = q_m_raw <= ridge_m(q_z)
    W_TOT = float(q_w.sum())
    unreachable_wfrac = float(q_w[~q_reach].sum() / W_TOT)

    # ---------------- design-projected ESS rebuild (sizing conventions) --------
    u = np.log1p(sdp._z_arr_all)
    m = np.asarray(sdp._log_M_z_all)
    n = len(u)
    sigma_u = float(n ** (-1.0 / 6.0) * u.std())
    _, sigma_m = sdp._compute_bandwidths(sdp._dl_raw_all, sdp._log_M_z_all)
    lam = sdp._abramson_lambda_u(u, sigma_u)  # estimator's own pilot machinery
    sig_i = sigma_u * lam

    def ess_grid(u_nodes, m_nodes):
        S1 = np.zeros((len(u_nodes), len(m_nodes)))
        S2 = np.zeros_like(S1)
        for lo in range(0, n, 200_000):
            hi = min(lo + 200_000, n)
            wu = np.exp(-0.5 * ((u[lo:hi, None] - u_nodes[None, :]) / sig_i[lo:hi, None]) ** 2)
            wm = np.exp(-0.5 * ((m[lo:hi, None] - m_nodes[None, :]) / sigma_m) ** 2)
            S1 += wu.T @ wm
            S2 += (wu * wu).T @ (wm * wm)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(S2 > 0, S1 * S1 / S2, 0.0)

    ess_design = ess_grid(U_NODES_DESIGN, M_NODES_DESIGN)

    # cross-check: replicate the as-built grid ESS with the standalone kernel
    ess_built_replica = ess_grid(u_nodes_b, m_nodes_b)
    replica_max_rel_dev = float(
        np.max(np.abs(ess_built_replica - ess_built) / np.maximum(ess_built, 1e-12))
    )

    # ---------------- scoring ----------------
    results = {}
    for tag, ess, un, mn in (
        ("as_built", ess_built, u_nodes_b, m_nodes_b),
        ("design_projected", ess_design, U_NODES_DESIGN, M_NODES_DESIGN),
    ):
        q_m_cl = np.clip(q_m_raw, mn.min(), mn.max())
        cm = cat_metrics(ess, un, mn, q_u, q_m_cl, q_w, q_reach)
        gm = grid_metrics(ess, un, mn)
        crit = {
            "c1_median_ESS_ge_1000": {
                "measured_reachable": cm["reachable"]["median_ESS"],
                "measured_all_clamped": cm["all_clamped"]["median_ESS"],
                "pass": bool(cm["reachable"]["median_ESS"] >= 1000.0),
            },
            "c2_wfrac_ESS_lt_500_le_1pct": {
                "measured_reachable": cm["reachable"]["wfrac_ESS_lt_500"],
                "measured_all_clamped": cm["all_clamped"]["wfrac_ESS_lt_500"],
                "pass": bool(cm["reachable"]["wfrac_ESS_lt_500"] <= 0.01),
            },
            "c3_wbar_reachable_ge_0p99": {
                "measured_reachable": cm["reachable"]["wbar"],
                "measured_all_clamped": cm["all_clamped"]["wbar"],
                "pass": bool(cm["reachable"]["wbar"] >= 0.99),
            },
        }
        results[tag] = {
            "grid": {
                "n_u_nodes": len(un), "n_m_nodes": len(mn),
                "u_range": [float(un.min()), float(un.max())],
                "m_range": [float(mn.min()), float(mn.max())],
                "m_spacing_dex": float(mn[1] - mn[0]),
            },
            "criteria": crit,
            "catalogue_metrics": cm,
            "grid_node_metrics": gm,
        }

    # as-built extra: the estimator's ACTUAL shipped K5 weights at query nodes
    w_q = bilinear(w_built, u_nodes_b, m_nodes_b, q_u,
                   np.clip(q_m_raw, m_nodes_b.min(), m_nodes_b.max()))
    results["as_built"]["shipped_K5_wbar_reachable"] = float(
        np.sum(q_w[q_reach] * w_q[q_reach]) / np.sum(q_w[q_reach])
    )
    results["as_built"]["shipped_K5_wbar_all_clamped"] = float(
        np.sum(q_w * w_q) / np.sum(q_w)
    )

    out = {
        "meta": {
            "pool": POOL,
            "date": "2026-07-28",
            "estimator": "SimulationDetectionProbability(pdet_z_resolved=True, "
                         "pdet_wbh_z_resolved=True, expected_z_max=1.5, "
                         f"snr_threshold={float(SNR_THRESHOLD)})",
            "n_rows_joint_leg_all_strata": n_all,
            "n_rows_marginal_leg_a_stratum": n_a,
            "sigma_u": sigma_u,
            "sigma_m": float(sigma_m),
            "n0": N0,
            "ess_replica_max_rel_dev_vs_estimator": replica_max_rel_dev,
            "catalogue_profile": PROFILE,
            "catalogue_W_total": W_TOT,
            "unreachable_ridge_wfrac": unreachable_wfrac,
            "sizing_predictions_N200k_mix3": {
                "wbar_reachable": 0.9985, "median_ESS": 8160.0,
                "wfrac_ESS_lt_500": 0.0001, "grid_min_ESS_reachable": 172.0,
            },
        },
        "pool_delivery_facts": facts,
        "results": results,
    }
    with open(f"{OUT}/acceptance_numbers.json", "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out["results"]["as_built"]["criteria"], indent=1))
    print(json.dumps(out["results"]["design_projected"]["criteria"], indent=1))
    print("done -> acceptance_numbers.json")


if __name__ == "__main__":
    main()
