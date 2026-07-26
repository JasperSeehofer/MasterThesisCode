"""Step 2 — candidate hosts per selected event + global-mode selection sums.

Replicates the production p_D candidate lookup EXACTLY with the pipeline's own
functions (GalaxyCatalogueHandler.get_possible_hosts_from_ball_tree with the same
window arguments p_D uses) and caches, per selected event:
  - the candidate host arrays (reduced = without-BH-mass-filter minus with-BH set,
    ordered as p_Di orders them: reduced first, then with_bh),
  - the rate weights w_g = R_eff(M_g)/(1+z_g),
  - the (h-independent) candidate window [z_min, z_max].

Also precomputes Sigma_global(h) = precompute_global_catalog_selection over the
full 41-h grid (needed for the normalization_mode="global" sanity swap).

Heavy in RAM/time (1.6 GB catalogue + two BallTrees) — run once, everything
downstream reads the JSON caches.
"""

import json
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/jasper/Repositories/MasterThesisCode")

from master_thesis_code.bayesian_inference.bayesian_statistics import (  # noqa: E402
    _rate_weight,
    precompute_global_catalog_selection,
)
from master_thesis_code.bayesian_inference.simulation_detection_probability import (  # noqa: E402
    SimulationDetectionProbability,
)
from master_thesis_code.constants import SNR_THRESHOLD  # noqa: E402
from master_thesis_code.cosmological_model import LamCDMScenario  # noqa: E402
from master_thesis_code.datamodels.detection import Detection  # noqa: E402
from master_thesis_code.galaxy_catalogue.handler import GalaxyCatalogueHandler  # noqa: E402
from master_thesis_code.physical_relations import get_redshift_outer_bounds  # noqa: E402

OUT = "/home/jasper/Repositories/MasterThesisCode/results/lcat_h_dependence_20260725"
VENUE = "/home/jasper/Repositories/MasterThesisCode/results/campaign_phase2_runs/run_20260703_seed1000"
INJ = f"{OUT}/data/injections"

sel = json.load(open(f"{OUT}/selected_events.json"))
h_grid = sel["h_grid"]
event_ids = sorted(int(k) for k in sel["events"])
print("events:", event_ids)

# --- detections (same filtering as evaluate(); index == diagnostics event_idx) ---
crb = pd.read_csv(f"{VENUE}/simulations/prepared_cramer_rao_bounds.csv")
crb = crb[crb["SNR"] >= SNR_THRESHOLD]
scen = LamCDMScenario()

# --- detection probability from the canonical depth15 pool (grid coverage check) ---
t0 = time.time()
detprob = SimulationDetectionProbability(
    injection_data_dir=INJ,
    snr_threshold=SNR_THRESHOLD,
    dl_bins=60,
    mass_bins=40,
    estimator="local_linear",
    expected_z_max=1.5,
)
detprob._get_or_build_grid(0.73)
print(f"detprob built in {time.time() - t0:.1f}s, dl_max={detprob.get_dl_max(0.73):.4f} Gpc")

# --- galaxy catalogue handler (heavy) ---
t0 = time.time()
handler = GalaxyCatalogueHandler(M_min=10**4.5, M_max=10**6.0, z_max=1.5)
print(f"handler built in {time.time() - t0:.1f}s: {len(handler.reduced_galaxy_catalog)} galaxies")

# --- global-mode selection sums over the full h grid (no-BH channel; also with-BH) ---
t0 = time.time()
glob_no_bh = precompute_global_catalog_selection(
    h_values=[float(h) for h in h_grid],
    galaxy_catalog=handler,
    detection_probability_obj=detprob,
    with_bh_mass=False,
    z_max_cap=1.5,
)
print(f"global sums (no BH) in {time.time() - t0:.1f}s")
with open(f"{OUT}/global_sums.json", "w") as f:
    json.dump({"no_bh": {f"{h:.4f}": v for h, v in glob_no_bh.items()}}, f, indent=1)

# --- per-event candidate hosts (exact p_D replication; h-independent window) ---
cand_out = {}
for idx in event_ids:
    det = Detection(crb.loc[idx])
    z_min, z_max = get_redshift_outer_bounds(
        distance=det.d_L,
        distance_error=det.d_L_uncertainty,
        h_min=scen.h.lower_limit,
        h_max=scen.h.upper_limit,
        Omega_m_min=scen.Omega_m.lower_limit,
        Omega_m_max=scen.Omega_m.upper_limit,
        sigma_multiplier=2.0,
    )
    z_max = min(z_max, 1.5)  # redshift_upper_limit = cosmological_model.max_redshift
    res = handler.get_possible_hosts_from_ball_tree(
        phi=det.phi,
        theta=det.theta,
        phi_sigma=det.phi_error,
        theta_sigma=det.theta_error,
        cov_theta_phi=det.theta_phi_covariance,
        z_min=z_min,
        z_max=z_max,
        M_z=det.M,
        M_z_sigma=det.M_uncertainty,
        sigma_multiplier=1.5,
    )
    assert res is not None, f"event {idx}: no hosts (expected host-found)"
    hosts_no_bh, hosts_with_bh = res
    with_set = set(hosts_with_bh)
    reduced = [g for g in hosts_no_bh if g not in with_set]
    ordered = reduced + hosts_with_bh  # p_Di ordering for the no-BH channel
    cand_out[str(idx)] = {
        "z_min": z_min,
        "z_max": z_max,
        "n_no_bh_ball": len(hosts_no_bh),
        "n_with_bh": len(hosts_with_bh),
        "n_reduced": len(reduced),
        "det": {
            "d_L": det.d_L,
            "d_L_unc": det.d_L_uncertainty,
            "phi": det.phi,
            "theta": det.theta,
            "M": det.M,
        },
        "hosts": {
            "phiS": [g.phiS for g in ordered],
            "qS": [g.qS for g in ordered],
            "z": [g.z for g in ordered],
            "z_error": [g.z_error for g in ordered],
            "M": [g.M for g in ordered],
            "M_error": [g.M_error for g in ordered],
            "catalog_index": [int(g.catalog_index) for g in ordered],
            "w": [_rate_weight(g) for g in ordered],
        },
    }
    print(
        f"event {idx:5d}: window z=[{z_min:.4f},{z_max:.4f}] "
        f"hosts={len(ordered)} (with_bh={len(hosts_with_bh)})"
    )

with open(f"{OUT}/candidates.json", "w") as f:
    json.dump(cand_out, f)
print("wrote candidates.json + global_sums.json")
