"""RR1 worst-R table: R = max(event-window width)/sigma_z for every shared
galaxy with sigma_z < 0.02, over all (venue, channel, pair, h in floor grid).

R is the sole controller of the fixed_quad n=50 Delta error (verified scale
invariance in rr1_boundary_check.json: identical error at equal R across
different absolute widths). Calibration measured there:
R<=30: <3e-9 nats | R=36: 4e-7 | R=45: 3e-5 | R=55: 3e-4 | R=69: 3e-3 |
R=72: 5e-3 | R=110: 6e-2 (all undiluted single-galaxy scale).
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from crossterm_instrument import (  # noqa: E402
    CRB_PATH,
    VENUE_CONFIGS,
    c4_pair_census,
    load_ball_sets,
    load_filtered_events,
)

from master_thesis_code.galaxy_catalogue.handler import (  # noqa: E402
    _empiric_stellar_mass_to_BH_mass_relation,
    _mass_redshift_prune_mask,
)
from master_thesis_code.physical_relations import dist_to_redshift  # noqa: E402

OUT = Path(__file__).resolve().parent / "rr1_worst_R_table.json"
COLS = ["REDSHIFT", "REDSHIFT_ERROR", "STELLAR_MASS", "STELLAR_MASS_ERROR", "REDSHIFT_FLAG"]
H_GRID = [0.60, 0.73, 0.81, 0.86]


def pruned_sigma_z(csv_path: Path) -> np.ndarray:
    parts = []
    for chunk in pd.read_csv(
        csv_path,
        names=["RA", "DEC", "B", *COLS],
        usecols=[3, 4, 5, 6, 7],
        chunksize=2_000_000,
    ):
        bh_mass, bh_err = _empiric_stellar_mass_to_BH_mass_relation(
            chunk["STELLAR_MASS"], chunk["STELLAR_MASS_ERROR"]
        )
        keep = (~pd.isna(bh_mass)) & _mass_redshift_prune_mask(
            bh_mass, bh_err, chunk["REDSHIFT"], chunk["REDSHIFT_ERROR"], 1e4, 1e7, 1.5
        )
        parts.append(chunk.loc[keep, "REDSHIFT_ERROR"].to_numpy(dtype=np.float64))
    return np.concatenate(parts)


crb_all = pd.read_csv(CRB_PATH)
crb_filtered = load_filtered_events(CRB_PATH)
pairs_all, _ = c4_pair_census(crb_all)
fidx = set(int(i) for i in crb_filtered.index)
pairs = [(i, j) for (i, j) in pairs_all if i in fidx and j in fidx]
needed = sorted({e for p in pairs for e in p})

width_max: dict[int, float] = {}
for ev in needed:
    row = crb_all.loc[ev]
    d = float(row["luminosity_distance"])
    sd = float(math.sqrt(row["delta_luminosity_distance_delta_luminosity_distance"]))
    width_max[ev] = max(
        float(dist_to_redshift(d + 4 * sd, h=h)) - float(dist_to_redshift(d - 4 * sd, h=h))
        for h in H_GRID
    )

results: dict = {}
for venue, cfg in VENUE_CONFIGS.items():
    sig = pruned_sigma_z(cfg["catalogue"])
    ball_1d, ball_2d = load_ball_sets(cfg["frozeng_dir"])
    for ch, balls in (("1d", ball_1d), ("2d", ball_2d)):
        rows = []
        worst = 0.0
        for i, j in pairs:
            shared = balls.get(i, set()) & balls.get(j, set())
            if not shared:
                continue
            W = max(width_max[i], width_max[j])
            idx = np.array(sorted(shared), dtype=np.int64)
            ssig = sig[idx]
            R_all = W / ssig
            worst = max(worst, float(R_all.max()))
            narrow = ssig < 0.02
            for k in np.nonzero(narrow)[0]:
                rows.append(
                    {
                        "pair": [i, j],
                        "catalog_index": int(idx[k]),
                        "sigma_z": float(ssig[k]),
                        "W_max_over_h": W,
                        "R": float(R_all[k]),
                        "n_shared": int(idx.size),
                    }
                )
        rows.sort(key=lambda r: -r["R"])
        results[f"{venue}/{ch}"] = {
            "worst_R_any_shared_galaxy": worst,
            "n_shared_galaxy_instances_sigma_lt_0.02": len(rows),
            "instances_R_gt_45": [r for r in rows if r["R"] > 45],
            "instances_R_gt_30": [r for r in rows if r["R"] > 30][:15],
        }
    del sig

with open(OUT, "w") as fh:
    json.dump(results, fh, indent=1)
print(json.dumps(results, indent=1))
