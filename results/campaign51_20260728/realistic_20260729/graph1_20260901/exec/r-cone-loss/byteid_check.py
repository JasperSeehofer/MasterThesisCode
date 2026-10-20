"""byteid_check.py — INDEPENDENT byte-id verifier for the r-cone-loss G-2 double anchor.

Written by the byte-id verifier subagent (not the builder). Deliberately does NOT
import `cone_loss_reads.py` — chord/radius are re-derived here from the raw CRB rows
and the production `GalaxyCatalogueHandler` catalogue, independently, so a bug shared
between builder script and verifier script cannot produce a false GREEN.

Anchors (from REGISTRATION_DRAFT.md sec5, "G-2 anchors (instrument byte-id)"):
  R-MKER-6: p3_2d_fleet_20260825/bc_900121_work/seed900121, event_idx 20
            chord 1.674660e-03 +/- 5e-10, radius 1.4956979545757095e-03 +/- 1e-15
  CMEM-A1:  p3_b0_work/bc_900101_work/seed900101, event_idx 0
            chord 0.0116656941007181 +/- 5e-10, radius 0.0359121946154451 +/- 1e-15

Formula (draft sec2 / module docstring of cone_loss_reads.py, cited to
handler.py/bayesian_statistics.py:3659,5751):
  chord  = |embed(host) - embed(event)|  (great-circle chord, unit-sphere embedding)
  radius = k * sqrt(lambda_max(J Sigma' J^T)),
           Sigma' = [[phi_var, cov],[cov, theta_var]], J = diag(|sin(theta)|, 1)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from darksiren_emri.galaxy_catalogue.handler import GalaxyCatalogueHandler, _polar_to_cartesian

K = 1.5

ANCHORS = [
    {
        "name": "R-MKER-6",
        "crb": "results/campaign51_20260728/realistic_20260729/p3_2d_fleet_20260825/bc_900121_work/seed900121/simulations/prepared_cramer_rao_bounds.csv",
        "event_idx": 20,
        "chord": 1.674660e-03,
        "chord_tol": 5e-10,
        "radius": 1.4956979545757095e-03,
        "radius_tol": 1e-15,
    },
    {
        "name": "CMEM-A1",
        "crb": "results/campaign51_20260728/realistic_20260729/p3_b0_work/bc_900101_work/seed900101/simulations/prepared_cramer_rao_bounds.csv",
        "event_idx": 0,
        "chord": 0.0116656941007181,
        "chord_tol": 5e-10,
        "radius": 0.0359121946154451,
        "radius_tol": 1e-15,
    },
]


def independent_embed(theta: float, phi: float) -> np.ndarray:
    """Re-derived unit-sphere embedding, NOT calling _polar_to_cartesian.

    Standard physics (theta, phi) -> Cartesian convention (theta = polar angle from
    z-axis, phi = azimuth), used as a cross-check that the handler's own
    ``_polar_to_cartesian`` (which we ALSO call below, since it is production code
    shared by every consumer of the catalogue -- not builder logic) is doing the
    conventional thing on these two specific points.
    """
    return np.array(
        [
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta),
        ]
    )


def cone_radius_independent(theta: float, phi_var: float, theta_var: float, cov: float, k: float) -> float:
    sigma = np.array([[phi_var, cov], [cov, theta_var]])
    jac = np.diag([abs(np.sin(theta)), 1.0])
    m = jac @ sigma @ jac.T
    lam = float(np.linalg.eigvalsh(m).max())
    return float(k * np.sqrt(max(lam, 0.0)))


def main() -> None:
    handler = GalaxyCatalogueHandler(1e4, 1e7, K)
    cat = handler.reduced_galaxy_catalog.reset_index(drop=True)

    results = []
    for anchor in ANCHORS:
        crb = pd.read_csv(anchor["crb"])
        row = crb.iloc[anchor["event_idx"]]
        hidx = int(row["host_galaxy_index"])
        theta_e = float(row["qS"])
        phi_e = float(row["phiS"])
        phi_var = float(row["delta_phiS_delta_phiS"])
        theta_var = float(row["delta_qS_delta_qS"])
        cov = float(row["delta_phiS_delta_qS"])

        host_theta = float(cat.loc[hidx, "THETA_S"])
        host_phi = float(cat.loc[hidx, "PHI_S"])

        # Path A: handler's own embedding (production code, shared by every consumer).
        ev_xyz_handler = _polar_to_cartesian(np.array([theta_e]), np.array([phi_e]))[0]
        host_xyz_handler = _polar_to_cartesian(np.array([host_theta]), np.array([host_phi]))[0]
        chord_handler = float(np.linalg.norm(host_xyz_handler - ev_xyz_handler))

        # Path B: independently re-derived embedding (no shared code with handler).
        ev_xyz_indep = independent_embed(theta_e, phi_e)
        host_xyz_indep = independent_embed(host_theta, host_phi)
        chord_indep = float(np.linalg.norm(host_xyz_indep - ev_xyz_indep))

        radius = cone_radius_independent(theta_e, phi_var, theta_var, cov, K)

        chord_dev_handler = abs(chord_handler - anchor["chord"])
        chord_dev_indep = abs(chord_indep - anchor["chord"])
        radius_dev = abs(radius - anchor["radius"])
        embed_agree = abs(chord_handler - chord_indep)

        results.append(
            {
                "name": anchor["name"],
                "hidx": hidx,
                "chord_handler_embed": chord_handler,
                "chord_independent_embed": chord_indep,
                "chord_embed_methods_agree_to": embed_agree,
                "chord_expected": anchor["chord"],
                "chord_dev_vs_handler_embed": chord_dev_handler,
                "chord_dev_vs_independent_embed": chord_dev_indep,
                "chord_tol": anchor["chord_tol"],
                "chord_pass": chord_dev_handler <= anchor["chord_tol"],
                "radius_found": radius,
                "radius_expected": anchor["radius"],
                "radius_dev": radius_dev,
                "radius_tol": anchor["radius_tol"],
                "radius_pass": radius_dev <= anchor["radius_tol"],
            }
        )

    n_pairs = 0
    max_abs_dev = 0.0
    all_pass = True
    print("INDEPENDENT BYTE-ID CHECK")
    for r in results:
        print(f"--- {r['name']} (event_idx, hidx={r['hidx']}) ---")
        for k_, v in r.items():
            print(f"  {k_}: {v}")
        n_pairs += 2  # chord, radius
        max_abs_dev = max(max_abs_dev, r["chord_dev_vs_handler_embed"], r["radius_dev"])
        all_pass = all_pass and r["chord_pass"] and r["radius_pass"]
        if r["chord_embed_methods_agree_to"] > 1e-12:
            print(
                f"  WARNING: handler-embed vs independent-embed chord disagree by "
                f"{r['chord_embed_methods_agree_to']:.3e} (convention mismatch?)"
            )

    print()
    print(f"n_pairs={n_pairs} max_abs_dev={max_abs_dev:.6e} verdict={'GREEN' if all_pass else 'RED'}")


if __name__ == "__main__":
    main()
