"""RR1 census: sigma_z of ACTUAL ball members / shared galaxies (zero-compute reads).

Determines whether the fixed_quad n=50 breakdown regime (sigma_z <~ 5e-3 on the
event window, measured in rr1_toy_attacks.py section B) and the sigma_z = 0 NaN
poison (6284 exact-zero rows in the parent catalogue; scipy norm(scale=0).pdf
= nan) actually intersect the production run's candidate balls and the 279
census pairs' shared sets.

Reconstructs the handler's positional catalog_index frame WITHOUT building the
handler (no BallTree): read CSV -> _empiric_stellar_mass_to_BH_mass_relation ->
drop NaN BH mass -> _mass_redshift_prune_mask(M_min=1e4, M_max=1e7, z_max=1.5)
-> reset index. This is the verbatim init chain of GalaxyCatalogueHandler
(handler.py:348-352) minus the order-preserving coordinate rotations.

Zero instrument compute; reads only (same class as review_integrity_reads.py).
"""

import json
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

M_MIN, M_MAX, Z_MAX = 1e4, 1e7, 1.5
COLS = ["REDSHIFT", "REDSHIFT_ERROR", "STELLAR_MASS", "STELLAR_MASS_ERROR", "REDSHIFT_FLAG"]

OUT = Path(__file__).resolve().parent / "rr1_ball_sigma_census.json"


def pruned_sigma_z(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """(sigma_z, flag) arrays indexed by the handler's positional catalog_index."""
    sig_parts: list[np.ndarray] = []
    flag_parts: list[np.ndarray] = []
    for chunk in pd.read_csv(
        csv_path,
        names=["RA", "DEC", "B", *COLS],
        usecols=[3, 4, 5, 6, 7],
        chunksize=2_000_000,
    ):
        bh_mass, bh_err = _empiric_stellar_mass_to_BH_mass_relation(
            chunk["STELLAR_MASS"], chunk["STELLAR_MASS_ERROR"]
        )
        has_mass = ~pd.isna(bh_mass)
        keep = has_mass & _mass_redshift_prune_mask(
            bh_mass, bh_err, chunk["REDSHIFT"], chunk["REDSHIFT_ERROR"], M_MIN, M_MAX, Z_MAX
        )
        sig_parts.append(chunk.loc[keep, "REDSHIFT_ERROR"].to_numpy(dtype=np.float64))
        flag_parts.append(chunk.loc[keep, "REDSHIFT_FLAG"].to_numpy(dtype=np.float64))
    return np.concatenate(sig_parts), np.concatenate(flag_parts)


def stats(sig: np.ndarray) -> dict:
    if sig.size == 0:
        return {"n": 0}
    nz = sig[sig > 0]
    return {
        "n": int(sig.size),
        "n_zero": int(np.sum(sig == 0.0)),
        "n_lt_1e-3": int(np.sum(sig < 1e-3)),
        "n_lt_2e-3": int(np.sum(sig < 2e-3)),
        "n_lt_5e-3": int(np.sum(sig < 5e-3)),
        "n_lt_1e-2": int(np.sum(sig < 1e-2)),
        "min": float(sig.min()),
        "min_nonzero": float(nz.min()) if nz.size else None,
        "median": float(np.median(sig)),
    }


results: dict = {}

crb_all = pd.read_csv(CRB_PATH)
crb_filtered = load_filtered_events(CRB_PATH)
pairs_all, _deg = c4_pair_census(crb_all)
fidx = set(int(i) for i in crb_filtered.index)
pairs = [(i, j) for (i, j) in pairs_all if i in fidx and j in fidx]
results["n_pairs"] = len(pairs)

for venue, cfg in VENUE_CONFIGS.items():
    sig, flag = pruned_sigma_z(cfg["catalogue"])
    vres: dict = {
        "pruned_rows": int(sig.size),
        "pruned_sigma_stats": stats(sig),
        "flag3_sigma_max": float(sig[flag == 3].max()) if np.any(flag == 3) else None,
        "flag3_count": int(np.sum(flag == 3)),
        "flag1_sigma_min": float(sig[flag == 1].min()) if np.any(flag == 1) else None,
    }
    ball_1d, ball_2d = load_ball_sets(cfg["frozeng_dir"])
    needed = sorted({e for p in pairs for e in p})
    for ch, balls in (("1d", ball_1d), ("2d", ball_2d)):
        member_union: set[int] = set()
        for ev in needed:
            member_union |= balls.get(ev, set())
        oob = [ci for ci in member_union if ci >= sig.size]
        mem_idx = np.array(sorted(member_union), dtype=np.int64)
        mem_sig = sig[mem_idx] if mem_idx.size else np.array([])
        events_with_zero = [
            ev
            for ev in needed
            if any(sig[ci] == 0.0 for ci in balls.get(ev, set()) if ci < sig.size)
        ]
        pair_rows = []
        n_pairs_narrow = 0
        n_pairs_zero = 0
        for i, j in pairs:
            shared = balls.get(i, set()) & balls.get(j, set())
            if not shared:
                continue
            ssig = sig[np.array(sorted(shared), dtype=np.int64)]
            row = {
                "pair": [i, j],
                "n_shared": int(ssig.size),
                "min_sigma_z_shared": float(ssig.min()),
                "n_shared_lt_5e-3": int(np.sum(ssig < 5e-3)),
                "n_shared_zero": int(np.sum(ssig == 0.0)),
            }
            if row["n_shared_lt_5e-3"] > 0:
                n_pairs_narrow += 1
                pair_rows.append(row)
            if row["n_shared_zero"] > 0:
                n_pairs_zero += 1
        vres[ch] = {
            "n_events": len(needed),
            "ball_member_union": int(mem_idx.size),
            "index_out_of_bounds": len(oob),
            "ball_sigma_stats": stats(mem_sig),
            "n_events_with_sigma0_ball_member": len(events_with_zero),
            "events_with_sigma0_ball_member": events_with_zero[:20],
            "n_pairs_with_shared_sigma_lt_5e-3": n_pairs_narrow,
            "n_pairs_with_shared_sigma0": n_pairs_zero,
            "narrow_pairs_detail": pair_rows[:30],
        }
    results[venue] = vres

with open(OUT, "w") as fh:
    json.dump(results, fh, indent=1)
print(json.dumps(results, indent=1))
