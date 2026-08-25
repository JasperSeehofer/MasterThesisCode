"""Gate-B counterfactual: symmetric +/-1.5 sigma mass window vs production asymmetric filter.

For every zero-class event (L_cat_with_bh==0 & L_cat_no_bh>0 at h=0.73) in the b0i2d
pilot (p3_2d_work, seed 900101) and the b0i fleet (p3_b0_work, bc/bt x 900101-900112):
reproduce the production candidate search (ball + z filter + asymmetric mass filter,
sigma_multiplier=1.5, h-window [0.50, 0.86] per the PA-CA-10 pin) and then apply the
symmetric variant (galaxy side widened to +/-1.5*BH_MASS_ERROR). Zero-compute: catalogue
+ CRB CSVs only, no likelihood evaluation.
"""

import csv
import glob
import json
import re
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

REPO = "/home/jasper/Repositories/darksiren-emri"
sys.path.insert(0, REPO)

from darksiren_emri.constants import HOST_DRAW_Z_MAX, M_SOURCE_FRAME_MAX, M_SOURCE_FRAME_MIN
from darksiren_emri.datamodels.detection import Detection
from darksiren_emri.galaxy_catalogue.handler import (
    GalaxyCatalogueHandler,
    InternalCatalogColumns,
    _polar_to_cartesian,
)
from darksiren_emri.physical_relations import get_redshift_outer_bounds

BASE = f"{REPO}/results/campaign51_20260728/realistic_20260729"
H_LO, H_HI = 0.50, 0.86  # widened prior window (PA-CA-10 pin, run_mirror_seed_inprocess)
Z_CAP = 1.5  # Model1CrossCheck.max_redshift
SIG = 1.5  # production sigma_multiplier (bayesian_statistics.py:4662)


def zero_events(diag_csv: str) -> list[int]:
    out = []
    for row in csv.DictReader(open(diag_csv)):
        if abs(float(row["h"]) - 0.73) > 1e-9:
            continue
        if float(row["L_cat_with_bh"]) == 0.0 and float(row["L_cat_no_bh"]) > 0.0:
            out.append(int(row["event_idx"]))
    return out


def analyze(handler: GalaxyCatalogueHandler, det: Detection) -> dict:
    z_min, z_max = get_redshift_outer_bounds(
        distance=det.d_L,
        distance_error=det.d_L_uncertainty,
        h_min=H_LO,
        h_max=H_HI,
        Omega_m_min=0.04,
        Omega_m_max=0.5,
        sigma_multiplier=2.0,  # NB: ignored inside (PHY-03) -- hardcoded 3 sigma
    )
    z_max = min(z_max, Z_CAP)

    # --- replicate handler.get_possible_hosts_from_ball_tree geometry exactly ---
    query_point = _polar_to_cartesian(np.array([det.theta]), np.array([det.phi]))
    sigma_matrix = np.array(
        [
            [det.phi_error**2, det.theta_phi_covariance],
            [det.theta_phi_covariance, det.theta_error**2],
        ]
    )
    jacobian = np.diag([abs(np.sin(det.theta)), 1.0])
    sigma_scaled = jacobian @ sigma_matrix @ jacobian.T
    lambda_max = float(np.linalg.eigvalsh(sigma_scaled).max())
    radius = float(SIG * np.sqrt(max(lambda_max, 0.0)))
    indices = handler.catalog_ball_tree.query_radius(query_point, r=radius)[0]
    cand = handler.reduced_galaxy_catalog.iloc[indices]

    z = cand[InternalCatalogColumns.REDSHIFT]
    ze = cand[InternalCatalogColumns.REDSHIFT_ERROR]
    zmask = (z_min <= z + ze) & (z_max >= z - ze)
    nb = cand[zmask]  # = candidate_hosts_without_bh_mass

    m = nb[InternalCatalogColumns.BH_MASS]
    me = nb[InternalCatalogColumns.BH_MASS_ERROR]
    lo = (det.M - det.M_uncertainty * SIG) / (1 + z_max)
    hi = (det.M + det.M_uncertainty * SIG) / (1 + z_min)

    asym = (lo <= m + me) & (m - me <= hi)  # production: galaxy +/- 1 sigma
    sym = (lo <= m + SIG * me) & (m - SIG * me <= hi)  # counterfactual: +/- 1.5 sigma

    # per-candidate distance in units of the galaxy's own mass error
    pulls = []
    for mm, ee in zip(m.values, me.values):
        if ee > 0:
            if mm < lo:
                pulls.append((lo - mm) / ee)
            elif mm > hi:
                pulls.append((mm - hi) / ee)
            else:
                pulls.append(0.0)
    return {
        "n_ball": int(len(cand)),
        "n_no_bh": int(len(nb)),
        "n_asym": int(asym.sum()),
        "n_sym": int(sym.sum()),
        "pulls": [round(float(p), 3) for p in pulls],
        "z_win": [round(z_min, 4), round(z_max, 4)],
    }


def main() -> None:
    handler = GalaxyCatalogueHandler(
        M_min=M_SOURCE_FRAME_MIN, M_max=M_SOURCE_FRAME_MAX, z_max=HOST_DRAW_Z_MAX
    )

    results = {"pilot": {}, "fleet": {}}

    # ---- pilot (b0i2d, seed 900101; bc and bt share the CRB) ----
    for arm in ["bc", "bt"]:
        root = f"{BASE}/p3_2d_work/{arm}_900101_work/seed900101/simulations"
        evs = zero_events(f"{root}/diagnostics/event_likelihoods.csv")
        crb = pd.read_csv(f"{root}/prepared_cramer_rao_bounds.csv")
        for e in evs:
            det = Detection(crb.loc[e])
            results["pilot"][f"{arm}:{e}"] = analyze(handler, det)

    # ---- fleet (b0i, bc/bt x 12 seeds) ----
    tally = defaultdict(int)
    for diag in sorted(
        glob.glob(f"{BASE}/p3_b0_work/b[ct]_9001??_work/seed*/simulations/diagnostics/event_likelihoods.csv")
    ) + sorted(
        glob.glob(f"{BASE}/p3_b0_work/eb0a_*_work/seed*/simulations/diagnostics/event_likelihoods.csv")
    ) + sorted(
        glob.glob(f"{BASE}/p3_b0_work/replica_*_work/seed*/simulations/diagnostics/event_likelihoods.csv")
    ):
        m = re.search(r"(b[ct]|eb0a|replica)_(\d+)_work", diag)
        arm, seed = m.group(1), m.group(2)
        root = diag.rsplit("/diagnostics/", 1)[0]
        import os

        crb_path = f"{root}/prepared_cramer_rao_bounds.csv"
        if not os.path.isfile(crb_path):
            nz = len(zero_events(diag))
            tally["skipped_runs"] += 1
            tally["skipped_zeros"] += nz
            print(f"SKIP (no CRB): {arm}_{seed} with {nz} zero rows")
            continue
        crb = pd.read_csv(crb_path)
        for e in zero_events(diag):
            det = Detection(crb.loc[e])
            r = analyze(handler, det)
            results["fleet"][f"{arm}_{seed}:{e}"] = r
            tally["n_zero"] += 1
            tally["reproduced_asym0"] += int(r["n_asym"] == 0)
            tally["nonempty_ball"] += int(r["n_no_bh"] > 0)
            tally["sym_retains"] += int(r["n_sym"] > 0)

    print(json.dumps(results["pilot"], indent=1))
    print("FLEET TALLY:", dict(tally))
    p_sym = sum(1 for r in results["pilot"].values() if r["n_sym"] > 0)
    p_rep = sum(1 for r in results["pilot"].values() if r["n_asym"] == 0 and r["n_no_bh"] > 0)
    print(f"PILOT: {len(results['pilot'])} zero rows; reproduced (asym empties a non-empty list): {p_rep}; symmetric retains: {p_sym}")
    with open(
        "/tmp/claude-1000/-home-jasper-Repositories-darksiren-emri/f76e9d1f-e875-48cc-888f-70b6e70d2905/scratchpad/counterfactual_out.json",
        "w",
    ) as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
