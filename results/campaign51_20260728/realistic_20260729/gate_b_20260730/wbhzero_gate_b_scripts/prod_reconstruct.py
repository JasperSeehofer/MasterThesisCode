"""Gate-B item 2: reconstruct the production candidate structure for run_20260804_postfix.

Same CRB in iiib and joint_r1 (md5 9a1f2a14...), baseline pinned catalogue
(c52c13b5...). Assumed flags (postfix_baseline/iiib run_metadata_0.json family):
default LamCDMScenario h limits [0.60, 0.86], max_redshift=1.5, sigma_multiplier=1.5.
Classifies every event: empty ball / z-filter-emptied / mass-filter-emptied (defect
class) / with-BH candidates present. Then intersects with the observed zero sets.
"""

import csv
import json
import sys

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

H_LO, H_HI = 0.60, 0.86  # production LamCDMScenario defaults
Z_CAP = 1.5
SIG = 1.5

SCRATCH = "/tmp/claude-1000/-home-jasper-Repositories-darksiren-emri/f76e9d1f-e875-48cc-888f-70b6e70d2905/scratchpad"


def zeros(p: str) -> set:
    s = set()
    for row in csv.DictReader(open(p)):
        if (
            abs(float(row["h"]) - 0.73) < 1e-9
            and float(row["L_cat_with_bh"]) == 0.0
            and float(row["L_cat_no_bh"]) > 0.0
        ):
            s.add(int(row["event_idx"]))
    return s


def main() -> None:
    handler = GalaxyCatalogueHandler(
        M_min=M_SOURCE_FRAME_MIN, M_max=M_SOURCE_FRAME_MAX, z_max=HOST_DRAW_Z_MAX
    )
    cat = handler.reduced_galaxy_catalog
    zc = cat[InternalCatalogColumns.REDSHIFT].to_numpy(float)
    zec = cat[InternalCatalogColumns.REDSHIFT_ERROR].to_numpy(float)
    mc = cat[InternalCatalogColumns.BH_MASS].to_numpy(float)
    mec = cat[InternalCatalogColumns.BH_MASS_ERROR].to_numpy(float)

    crb = pd.read_csv(f"{REPO}/results/run_20260804_postfix/iiib/diagnostics/prepared_cramer_rao_bounds.csv")
    out = {}
    for e in crb.index:
        det = Detection(crb.loc[e])
        z_min, z_max = get_redshift_outer_bounds(det.d_L, det.d_L_uncertainty, h_min=H_LO, h_max=H_HI)
        z_max = min(z_max, Z_CAP)
        q = _polar_to_cartesian(np.array([det.theta]), np.array([det.phi]))
        S = np.array(
            [[det.phi_error**2, det.theta_phi_covariance], [det.theta_phi_covariance, det.theta_error**2]]
        )
        J = np.diag([abs(np.sin(det.theta)), 1.0])
        lam = float(np.linalg.eigvalsh(J @ S @ J.T).max())
        r = float(SIG * np.sqrt(max(lam, 0.0)))
        idx = handler.catalog_ball_tree.query_radius(q, r=r)[0]
        zi, zei, mi, mei = zc[idx], zec[idx], mc[idx], mec[idx]
        zmask = (z_min <= zi + zei) & (z_max >= zi - zei)
        n_nb = int(zmask.sum())
        lo = (det.M - det.M_uncertainty * SIG) / (1 + z_max)
        hi = (det.M + det.M_uncertainty * SIG) / (1 + z_min)
        m_nb, me_nb = mi[zmask], mei[zmask]
        asym = (lo <= m_nb + me_nb) & (m_nb - me_nb <= hi)
        sym = (lo <= m_nb + SIG * me_nb) & (m_nb - SIG * me_nb <= hi)
        out[int(e)] = {
            "n_ball": int(len(idx)),
            "n_no_bh": n_nb,
            "n_asym": int(asym.sum()),
            "n_sym": int(sym.sum()),
        }

    with open(f"{SCRATCH}/prod_structure.json", "w") as f:
        json.dump(out, f)

    S_def = {e for e, v in out.items() if v["n_no_bh"] > 0 and v["n_asym"] == 0}
    S_sym = {e for e, v in out.items() if e in S_def and v["n_sym"] > 0}
    empty_ball = {e for e, v in out.items() if v["n_no_bh"] == 0}
    print(f"events: {len(out)}; structural mass-filter-emptied (defect class): {len(S_def)}"
          f" ({100 * len(S_def) / len(out):.1f}%); of those symmetric window retains: {len(S_sym)}")
    print(f"empty no-BH list (ball/z): {len(empty_ball)}")
    for tag in ["iiib", "joint_r1"]:
        Z = zeros(f"{REPO}/results/run_20260804_postfix/{tag}/diagnostics/event_likelihoods.csv")
        print(f"{tag}: zeros@0.73 {len(Z)}; in defect class: {len(Z & S_def)}; "
              f"not structural (numeric/config zeros): {len(Z - S_def)}; "
              f"defect-class events NOT zero in this run: {len(S_def - Z)}")


if __name__ == "__main__":
    main()
