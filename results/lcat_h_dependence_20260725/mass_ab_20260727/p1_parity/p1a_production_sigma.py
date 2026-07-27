"""P1 parity audit, part A — PRODUCTION-mode Sigma_glob_wbh, row-by-row.

Reproduces the exact production code path for the with-BH catalogue selection
sum Sigma_glob_wbh (bayesian_statistics.precompute_global_catalog_selection,
with_bh=True branch) by IMPORTING the production modules (no source edits):

  - SimulationDetectionProbability built from the canonical seed-1000 pool
    (results/lcat_h_dependence_20260725/data/injections, identical to the
    A/B cells' pool — md5-verified), flag OFF (pooled-in-z 2D grid) and
    flag ON (--pdet_wbh_z_resolved joint z x M_z shrunk table), same
    constructor kwargs as bs.py:2064-2081.
  - Catalogue rows from GalaxyCatalogueHandler(M_min=10^4.5, M_max=10^6,
    z_max=1.5) — the same object production sums over; anchors checked
    against catalog_zw_profile.json (z0).
  - Per h in the venue grid {0.60,...,0.86}: eligibility z < z_max(h) =
    dist_to_redshift(get_dl_max(h)), w_g = R_eff(M_g)/(1+z_g),
    p_det queried through detection_probability_with_bh_mass_interpolated
    exactly as bs.py:1621-1631 (isotropic sky, z kwarg when flag on).

Measures (all written to p1a_results.json):
  (validation) Sigma_off(h) vs the cluster cellApp log values, and
      Sigma_on(h) vs the values extracted from the zmzApp/cellApp per-event
      L_cat_with_bh ratio (constant across events at 1e-13).
  (iv) clamp fraction: w_g-weighted fraction of the query set with
      log10 M_z >= pool m_max (6.000), the p_det-weighted contribution
      share per arm, and the clamped/unclamped decomposition of the
      conditioning movement (hybrid sums: only-unclamped-switched /
      only-clamped-switched).
  (iv-diag) u-sensitivity of the shipped table AT the top m-node
      (S-tilde spread across u at fixed d_L; shrinkage weight w and ESS
      along the top m-node row) — the direct test of the readout's
      "clamped queries cannot feel the u-conditioning" mechanism.
  (i) value/slope parity vs the probe's binned M_z-only and shrunk-joint
      tables (z2/z3_results.json).
  (translation) exact per-event A-cell re-prediction: with the flag's
      A-cell effect measured to be exactly L_cat_wbh -> L_cat_wbh *
      Sigma_off/Sigma_on (invariance of w_G/L_cat_no_bh/B_num/L_comp
      verified at 0 delta), the predicted profile shift for ANY Sigma
      table pair is Sum_i ln(1 + s_i (r-1)) computed from the cellApp
      diagnostics — the correct axis translation the P2/P3 gate lacked.

Grid-only arm: run with MTC_WBH_GRID_ONLY=1 (module-level env read at
import) — `uv run python p1a_production_sigma.py --grid-only` writes
p1a_gridonly.json with the same Sigma table for the u-kernel-disabled build.

Offline, CPU-only, read-only w.r.t. production source and prior results.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

REPO = "/home/jasper/Repositories/MasterThesisCode"
BASE = f"{REPO}/results/lcat_h_dependence_20260725"
HERE = f"{BASE}/mass_ab_20260727/p1_parity"
INJ = f"{BASE}/data/injections"
SCRATCH = os.environ.get(
    "P1_CACHE_DIR",
    "/tmp/claude-1000/-home-jasper-Repositories-MasterThesisCode/"
    "7cb124bf-2889-4fc6-987f-d69ae709b032/scratchpad",
)
sys.path.insert(0, REPO)

parser = argparse.ArgumentParser()
parser.add_argument("--grid-only", action="store_true")
args = parser.parse_args()

if args.grid_only:
    os.environ["MTC_WBH_GRID_ONLY"] = "1"

from master_thesis_code.bayesian_inference.simulation_detection_probability import (  # noqa: E402
    SimulationDetectionProbability,
)
from master_thesis_code.constants import HOST_DRAW_Z_MAX, SNR_THRESHOLD  # noqa: E402
from master_thesis_code.emri_rate import R_eff_per_mbh  # noqa: E402
from master_thesis_code.physical_relations import (  # noqa: E402
    dist_to_redshift,
    dist_vectorized,
)

H_VENUE = [0.60, 0.65, 0.70, 0.73, 0.76, 0.80, 0.86]

# Cluster-measured Sigma tables (provenance: cellApp/logs/evaluate_*.err
# "Global catalog selection (with_bh=True)" lines, jobs 6061150 array; and
# Sigma_on/gridonly back-solved from the per-event L_cat_with_bh ratios
# zmzApp / zmzGridOnly vs cellApp (constant across 876 catalogue events to
# std < 1e-13; extraction: this audit, 2026-07-28).
SIG_OFF_CLUSTER = {
    0.60: 2.700808e8, 0.65: 2.759635e8, 0.70: 2.815376e8, 0.73: 2.847445e8,
    0.76: 2.878574e8, 0.80: 2.918739e8, 0.86: 2.976373e8,
}
SIG_ON_CLUSTER = {
    0.60: 2.767679e8, 0.65: 2.826148e8, 0.70: 2.885021e8, 0.73: 2.920316e8,
    0.76: 2.955726e8, 0.80: 3.003153e8, 0.86: 3.074504e8,
}
SIG_GRIDONLY_CLUSTER = {
    0.60: 2.838463e8, 0.65: 2.899200e8, 0.70: 2.956500e8, 0.73: 2.989355e8,
    0.76: 3.021176e8, 0.80: 3.062085e8, 0.86: 3.120567e8,
}

# ---------------- catalogue rows (cached) ----------------
cache = f"{SCRATCH}/p1_catalogue_zM.npz"
if os.path.exists(cache):
    d = np.load(cache)
    z_all, M_all = d["z"], d["M"]
    print(f"catalogue from cache: {len(z_all)} rows", flush=True)
else:
    t0 = time.time()
    os.chdir(REPO)  # handler reads the reduced catalogue via a repo-relative path
    from master_thesis_code.galaxy_catalogue.handler import (
        GalaxyCatalogueHandler,
        InternalCatalogColumns,
    )

    handler = GalaxyCatalogueHandler(M_min=10**4.5, M_max=10**6.0, z_max=1.5)
    cat = handler.reduced_galaxy_catalog
    z_all = cat[InternalCatalogColumns.REDSHIFT].to_numpy(dtype=np.float64)
    M_all = cat[InternalCatalogColumns.BH_MASS].to_numpy(dtype=np.float64)
    np.savez_compressed(cache, z=z_all, M=M_all)
    print(f"handler built in {time.time() - t0:.0f}s: {len(z_all)} rows", flush=True)

# Anchors vs the z0 profile (the probe's catalogue snapshot).
prof = json.load(open(f"{BASE}/zres_survival/catalog_zw_profile.json"))
ok = np.isfinite(M_all) & (M_all > 0.0) & np.isfinite(z_all) & (z_all >= 0.0) & (z_all < 1.5)
z_c, M_c = z_all[ok], M_all[ok]
w_c = np.asarray(R_eff_per_mbh(M_c), dtype=np.float64) / (1.0 + z_c)
anchors = {
    "n_rows_pruned": int(len(z_all)),
    "n_used": int(ok.sum()),
    "n_z_lt_0992": int(np.sum(z_c < 0.992)),
    "W_total_z15": float(w_c.sum()),
    "profile_n_rows_pruned": prof["n_rows_pruned"],
    "profile_n_z_lt_0992": prof["n_z_lt_0992"],
    "profile_W_total_z15": prof["W_total_z15"],
}
anchors["match"] = (
    anchors["n_rows_pruned"] == prof["n_rows_pruned"]
    and anchors["n_z_lt_0992"] == prof["n_z_lt_0992"]
    and abs(anchors["W_total_z15"] / prof["W_total_z15"] - 1.0) < 1e-9
)
print("anchors:", json.dumps(anchors, indent=1), flush=True)

# ---------------- production detection-probability objects ----------------
common = dict(
    injection_data_dir=INJ,
    snr_threshold=SNR_THRESHOLD,
    dl_bins=60,
    mass_bins=40,
    estimator="local_linear",
    expected_z_max=HOST_DRAW_Z_MAX,
    pdet_z_resolved=True,
)
t0 = time.time()
sdp_off = SimulationDetectionProbability(**common, pdet_wbh_z_resolved=False)
print(f"flag-OFF SDP built in {time.time() - t0:.0f}s", flush=True)
t0 = time.time()
sdp_on = SimulationDetectionProbability(**common, pdet_wbh_z_resolved=True)
print(f"flag-ON SDP built in {time.time() - t0:.0f}s", flush=True)

m_nodes = sdp_on._wbh_m_nodes
u_nodes = sdp_on._wbh_u_nodes
m_max_pool = float(m_nodes[-1])
grid2d, _ = sdp_off._get_or_build_grid(0.73)
m_centers_off = np.asarray(grid2d.grid[1])
res: dict = {
    "grid_only_env": bool(args.grid_only),
    "anchors": anchors,
    "pool_m_max_joint_nodes": m_max_pool,
    "flagoff_grid_top_center_log10Mz": float(np.log10(m_centers_off[-1])),
}

# ---------------- per-h row-by-row sums ----------------
tab = {}
for h in H_VENUE:
    z_max = dist_to_redshift(sdp_off.get_dl_max(h), h=h)
    elig = (z_all < z_max) & np.isfinite(M_all) & (M_all > 0.0)
    z_g = z_all[elig]
    M_g = M_all[elig]
    w_g = np.asarray(R_eff_per_mbh(M_g), dtype=np.float64) / (1.0 + z_g)
    d_L_g = np.asarray(dist_vectorized(z_g, h=h), dtype=np.float64)
    M_z_g = M_g * (1.0 + z_g)
    lm_g = np.log10(M_z_g)
    zeros = np.zeros_like(z_g)
    p_off = np.asarray(
        sdp_off.detection_probability_with_bh_mass_interpolated(
            d_L_g, M_z_g, zeros, zeros, h=h
        ),
        dtype=np.float64,
    )
    p_on = np.asarray(
        sdp_on.detection_probability_with_bh_mass_interpolated(
            d_L_g, M_z_g, zeros, zeros, h=h, z=z_g
        ),
        dtype=np.float64,
    )
    clamped = lm_g >= m_max_pool
    s_off = float(np.sum(w_g * p_off))
    s_on = float(np.sum(w_g * p_on))
    s_off_cl = float(np.sum(w_g[clamped] * p_off[clamped]))
    s_on_cl = float(np.sum(w_g[clamped] * p_on[clamped]))
    tab[h] = {
        "n_eligible": int(elig.sum()),
        "z_max": float(z_max),
        "W_eligible": float(w_g.sum()),
        "W_frac_clamped": float(w_g[clamped].sum() / w_g.sum()),
        "Sigma_off": s_off,
        "Sigma_on": s_on,
        "Sigma_off_clamped_part": s_off_cl,
        "Sigma_on_clamped_part": s_on_cl,
        # hybrids: switch the conditioning ONLY on one subset
        "Sigma_hyb_unclamped_switched": s_off_cl + (s_on - s_on_cl),
        "Sigma_hyb_clamped_switched": s_on_cl + (s_off - s_off_cl),
        "cluster_Sigma_off": SIG_OFF_CLUSTER[h],
        "cluster_Sigma_on": SIG_ON_CLUSTER[h],
        "cluster_Sigma_gridonly": SIG_GRIDONLY_CLUSTER[h],
        "rel_diff_off_vs_cluster": s_off / SIG_OFF_CLUSTER[h] - 1.0,
        "rel_diff_on_vs_cluster": s_on / SIG_ON_CLUSTER[h] - 1.0,
    }
    if h == 0.73:
        # pooled no-BH survival row-by-row (0.556/0.589 parity denominator)
        dh_sorted = sdp_off._d_hor_sorted
        n_inj = len(dh_sorted)
        s_pooled = (n_inj - np.searchsorted(dh_sorted, d_L_g, side="left")) / float(n_inj)
        res["Sigma_nobh_pooled_rowbyrow_073"] = float(np.sum(w_g * s_pooled))
        # ESS / shrinkage weight interpolated at the actual query coords
        uq = np.log1p(z_g)
        a0 = np.clip(
            np.floor(np.interp(uq, u_nodes, np.arange(u_nodes.size))).astype(int),
            0,
            u_nodes.size - 2,
        )
        fa = np.clip(np.interp(uq, u_nodes, np.arange(u_nodes.size)) - a0, 0, 1)
        b0 = np.clip(
            np.floor(np.interp(lm_g, m_nodes, np.arange(m_nodes.size))).astype(int),
            0,
            m_nodes.size - 2,
        )
        fb = np.clip(np.interp(lm_g, m_nodes, np.arange(m_nodes.size)) - b0, 0, 1)
        wtab = sdp_on._wbh_w
        w_at = (
            (1 - fa) * (1 - fb) * wtab[a0, b0]
            + fa * (1 - fb) * wtab[a0 + 1, b0]
            + (1 - fa) * fb * wtab[a0, b0 + 1]
            + fa * fb * wtab[a0 + 1, b0 + 1]
        )
        res["catalogue_weighted_wbar_rowbyrow_073"] = float(np.sum(w_g * w_at) / np.sum(w_g))
        res["clamp_073"] = {
            "W_frac_clamped_queries": float(w_g[clamped].sum() / w_g.sum()),
            "contrib_frac_clamped_off": s_off_cl / s_off,
            "contrib_frac_clamped_on": s_on_cl / s_on,
        }
    print(f"h={h}: off={s_off:.6e} on={s_on:.6e}  (cluster off={SIG_OFF_CLUSTER[h]:.6e})", flush=True)

res["per_h"] = {str(h): tab[h] for h in H_VENUE}


def dln(table_key: str, h1: float = 0.73, h2: float = 0.86) -> float:
    return float(np.log(tab[h2][table_key] / tab[h1][table_key]))


res["dln_073_086"] = {
    k: dln(k)
    for k in (
        "Sigma_off",
        "Sigma_on",
        "Sigma_hyb_unclamped_switched",
        "Sigma_hyb_clamped_switched",
        "cluster_Sigma_off",
        "cluster_Sigma_on",
        "cluster_Sigma_gridonly",
    )
}
res["dln_073_080"] = {
    k: dln(k, 0.73, 0.80)
    for k in (
        "Sigma_off",
        "Sigma_on",
        "Sigma_hyb_unclamped_switched",
        "Sigma_hyb_clamped_switched",
        "cluster_Sigma_off",
        "cluster_Sigma_on",
        "cluster_Sigma_gridonly",
    )
}

# ---------------- u-sensitivity at the top m-node (hypothesis iv mechanism) ----------------
stilde = sdp_on._wbh_stilde  # (n_q, n_u, n_m)
dlq = sdp_on._wbh_dlq
u_cat = np.log1p(z_c)
u_q = np.quantile(u_cat, [0.1, 0.25, 0.5, 0.75, 0.9])
ai = np.clip(np.searchsorted(u_nodes, u_q), 0, u_nodes.size - 1)
sens = {}
for dl_probe in (2.0, 4.0, 6.0, 8.0):
    j = int(np.clip(np.searchsorted(dlq, dl_probe), 0, dlq.size - 1))
    row = stilde[j, :, -1]  # top m-node, all u
    sens[f"dL_{dl_probe:g}"] = {
        "S_at_cat_u_quantiles": [float(row[a]) for a in ai],
        "S_full_u_range_min": float(row.min()),
        "S_full_u_range_max": float(row.max()),
    }
res["top_mnode_u_sensitivity"] = sens
res["top_mnode_shrinkage"] = {
    "w_min": float(sdp_on._wbh_w[:, -1].min()),
    "w_median": float(np.median(sdp_on._wbh_w[:, -1])),
    "w_max": float(sdp_on._wbh_w[:, -1].max()),
    "ess_min": float(sdp_on._wbh_ess[:, -1].min()),
    "ess_median": float(np.median(sdp_on._wbh_ess[:, -1])),
}

out_name = "p1a_gridonly.json" if args.grid_only else "p1a_results.json"
with open(f"{HERE}/{out_name}", "w") as f:
    json.dump(res, f, indent=1)
print(f"wrote {HERE}/{out_name}")
