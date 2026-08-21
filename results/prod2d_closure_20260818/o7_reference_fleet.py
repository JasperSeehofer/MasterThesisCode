r"""O7 arm R7 -- the 15-seed reference fleet, from committed code (zero-evaluate()).

Registered in ``PREREGISTRATION_SELFGEN_CONTROL.md`` "O7 -- FLEET TRANSFER CLOSE"
(2026-08-22, ledger row #159 item 1). Loops the UNMODIFIED committed O6 reference
computation (:func:`o6_reference_derivation.compute_reference`) over all 15 F
seeds, banking the per-seed ``r_prod``/``r_A`` vectors and the fleet mean/SEM --
repairing the provenance gap where the restored-arm fleet statistics existed
only as prose from a reviewer-scratchpad computation (A20_REVIEW_O4_20260821.md).

Zero ``BayesianStatistics.evaluate()`` calls by construction: the shared
selection objects and the S_bar_phi/beta_Gbar_phi tables are built ONCE from the
pinned production leaf functions, and each seed's event set is the deterministic
``draw_csg_realization`` redraw (bit-exactness across venues proven by O4 GATE R4).
"""

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import o4_pairing_test as o4  # noqa: E402
import o6_reference_derivation as o6r  # noqa: E402

from darksiren_emri.validation.selfgen_control import (  # noqa: E402
    CRB_CSV_PATH,
    build_csg_selection_objects,
    draw_csg_realization,
)

OUT_PATH = Path(__file__).parent / "o7_reference_fleet_output.json"
REGISTRATION_SECTION = (
    "results/prod2d_closure_20260818/PREREGISTRATION_SELFGEN_CONTROL.md, "
    "O7 -- FLEET TRANSFER CLOSE: REGISTRATION (2026-08-22, row #159 item 1), arm R7"
)
N_EVENTS = 200


def main() -> int:
    completeness, detection_probability = build_csg_selection_objects(h_gen=o4.H_GEN)
    donor_rows = pd.read_csv(CRB_CSV_PATH)
    phi_table, beta_gbar_phi = o4.build_aligned_tables(completeness, detection_probability)

    per_seed: list[dict[str, Any]] = []
    for seed in o4.F_SEEDS:
        rows, _diag = draw_csg_realization(
            seed, "csgf", N_EVENTS, completeness, detection_probability, donor_rows
        )
        geos = o4.event_geometries(rows, completeness)
        result = o6r.compute_reference(geos, completeness, phi_table, beta_gbar_phi)
        per_seed.append(
            {
                "seed": seed,
                "r_prod": result["r_prod"],
                "r_A_REPORTED_ONLY": result["r_A"],
            }
        )
        print(
            f"seed {seed}: r_prod = {result['r_prod']['mean_score']:+.6f}  "
            f"r_A = {result['r_A']['mean_score']:+.6f}",
            flush=True,
        )

    r_prod_vec = np.array([s["r_prod"]["mean_score"] for s in per_seed], dtype=np.float64)
    r_a_vec = np.array([s["r_A_REPORTED_ONLY"]["mean_score"] for s in per_seed], dtype=np.float64)
    n = len(r_prod_vec)
    fleet = {
        "n_seeds": n,
        "r_prod_fleet_mean": float(r_prod_vec.mean()),
        "r_prod_fleet_sd": float(r_prod_vec.std(ddof=1)),
        "r_prod_fleet_sem": float(r_prod_vec.std(ddof=1) / np.sqrt(n)),
        "r_A_fleet_mean_REPORTED_ONLY": float(r_a_vec.mean()),
        "r_A_fleet_sem_REPORTED_ONLY": float(r_a_vec.std(ddof=1) / np.sqrt(n)),
        "reference": (
            "A18: these fleet statistics subtract nothing -- they ARE the banked "
            "reference the O7 fleet claim names; realized SEM recomputed from this "
            "vector per the O7 registration's A17 line (not carried from "
            "A20_REVIEW_O4_20260821.md prose)."
        ),
    }
    output: dict[str, Any] = {
        "registered_in": REGISTRATION_SECTION,
        "instrument": "o7_reference_fleet.py (loops o6_reference_derivation.compute_reference, unmodified)",
        "arm": "csgf",
        "h_gen": o4.H_GEN,
        "h_lo": o4.H_LO,
        "h_hi": o4.H_HI,
        "n_events": N_EVENTS,
        "redshift_upper_limit_used": o4.REDSHIFT_UPPER_LIMIT,
        "production_quad_n": o4.PRODUCTION_QUAD_N,
        "zero_evaluate_note": (
            "No BayesianStatistics.evaluate() call; tables from the pinned "
            "production leaf functions, event sets from the deterministic "
            "draw_csg_realization redraw (O4 GATE R4 bit-exactness)."
        ),
        "per_seed": per_seed,
        "fleet": fleet,
    }
    OUT_PATH.write_text(json.dumps(output, indent=2))
    print(
        f"\nfleet r_prod = {fleet['r_prod_fleet_mean']:+.6f} "
        f"+/- {fleet['r_prod_fleet_sem']:.6f} (SEM, n={n})\nwrote {OUT_PATH}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
