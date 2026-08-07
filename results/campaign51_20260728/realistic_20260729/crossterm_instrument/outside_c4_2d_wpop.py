"""Prereg §7.4 mandated pre-run read: outside-C-4 2D shared-w_pop shares.

Fills the declared gap in ``m4_results.json`` — its
``global_pairs_shares.outside_c4_records`` carry 1D shares only, while the
prereg per-unit 2D denominators W need the min-side shared ``w_pop`` shares of
the outside-C-4 2D-sharing pairs (77 joint_r1 / 16 iiib). Computes them with
the EXACT M-4 convention (m4_shared_galaxy_census.py / m4_global_shares.py):

    share(pair (i,j), side e) = Sum_{g in ball2d_i INT ball2d_j} w_g
                                / Sum_{g in ball2d_e} w_g,
    w_g = R_eff_per_mbh(M_g) / (1 + z_g),

with (z_g, M_g) dereferenced at catalog_index in the bit-faithful pruned+reset
catalogue frame (production mass-mapping + prune functions; sky rotation
membership-neutral, skipped), ball2d = the frozeng h_0_73.json
``galaxy_likelihoods`` list (M-4 recipe; h-independence verified in M-4/V2).
The wN variant (weights x per-galaxy stored N_wbh at h=0.73) is emitted as a
diagnostic alongside, as in M-4.

BLINDNESS (prereg §7.4: "zero-compute, deterministic, blindness-preserving —
w_pop shares are independent of any Delta"): this script computes NO Delta, NO
quadrature, NO likelihood — only ball-set intersections and catalogue-column
weight sums. It reads frozeng posterior JSONs + catalogues + m4_results.json
and writes ONLY outside_c4_2d_wpop.json next to this script.

Cross-checks (hard asserts):
  * m4_results.json sha256 == prereg §7.1 pin.
  * The outside-C-4 2D pair sets equal BOTH m4 outside_c4_records
    (n_shared_2d > 0) and global_sharing_check.outside_pairs_2d (77 / 16).
  * The in-C-4 2D min-side share sums recomputed here equal
    band_derivation.json's W parts (1.9678973990927773 joint_r1 /
    0.3092916997125221 iiib) — same machinery, same numbers.

Run: cd /home/jasper/Repositories/MasterThesisCode && uv run python \
    results/campaign51_20260728/realistic_20260729/crossterm_instrument/outside_c4_2d_wpop.py
"""

import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = "/home/jasper/Repositories/MasterThesisCode"
sys.path.insert(0, REPO)
os.chdir(REPO)

from master_thesis_code.emri_rate import R_eff_per_mbh  # noqa: E402
from master_thesis_code.galaxy_catalogue.handler import (  # noqa: E402
    _empiric_stellar_mass_to_BH_mass_relation,
    _mass_redshift_prune_mask,
    _reduced_catalog_column_names,
)

HERE = Path(__file__).resolve().parent
M4_PATH = HERE / "m4_results.json"
M4_SHA256_PIN = "46907913b35369bfdb96a705d1782a5c8d4eb9bf7955d1cb851434ba9d3f1c6b"
OUT_PATH = HERE / "outside_c4_2d_wpop.json"

STAGED = Path(REPO) / "results/campaign51_20260728/realistic_20260729/realizations_staged"
FROZENG = Path(REPO) / "results/run_20260804_frozeng"
CATS = {
    "joint_r1": STAGED / "observed_catalogue_seed900001.csv",
    "iiib": STAGED / "cluster_parent_reduced_galaxy_catalogue.csv",
}

#: band_derivation.json per_unit_shared_weight 2D in-C-4 W parts (cross-check).
W_IN_C4_2D = {"joint_r1": 1.9678973990927773, "iiib": 0.3092916997125221}

#: Locked band values (prereg §7.2/§7.3); per-unit thresholds are the locked
#: FORMULA x = X/W, y = Y/W — only the W denominators were pending this read.
X_LOCKED = 2.78
Y_LOCKED = 7.96

M_SOURCE_FRAME_MIN, M_SOURCE_FRAME_MAX, Z_MAX = 1e4, 1e7, 1.5


def qtiles(a: list[float]) -> dict[str, float] | None:
    arr = np.asarray(a, dtype=float)
    if arr.size == 0:
        return None
    return {
        "n": int(arr.size),
        "min": float(arr.min()),
        "median": float(np.median(arr)),
        "mean": float(arr.mean()),
        "p90": float(np.percentile(arr, 90)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(arr.max()),
    }


def load_wbh(path: Path) -> tuple[dict[int, frozenset[int]], dict[int, dict[int, float]]]:
    """Frozeng 2D ball sets + stored per-galaxy N_wbh maps (M-4 recipe)."""
    d = json.loads(path.read_text())
    gl = d["galaxy_likelihoods"]
    wbh: dict[int, frozenset[int]] = {}
    n2: dict[int, dict[int, float]] = {}
    for k in gl:
        if not k.isdigit():
            continue
        ev = int(k)
        wbh[ev] = frozenset(r[0] for r in gl[k])
        n2[ev] = {r[0]: r[1][2] for r in gl[k]}
    return wbh, n2


def load_pruned_zm(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Bit-faithful pruned+reset (z, M_bh) columns (M-4 machinery verbatim)."""
    names = _reduced_catalog_column_names()
    cat = pd.read_csv(path, names=names, usecols=[3, 4, 5, 6])
    z = cat["REDSHIFT"].to_numpy(np.float64)
    sz = cat["REDSHIFT_MEASUREMENT_ERROR"].to_numpy(np.float64)
    ms = cat["STELLAR_MASS"].to_numpy(np.float64)
    mse = cat["STELLAR_MASS_ABSOULTE_ERROR"].to_numpy(np.float64)
    del cat
    mbh, mbh_err = _empiric_stellar_mass_to_BH_mass_relation(ms, mse)
    del ms, mse
    keep = ~np.isnan(mbh)
    z, sz, mbh, mbh_err = z[keep], sz[keep], mbh[keep], mbh_err[keep]
    mask = _mass_redshift_prune_mask(
        pd.Series(mbh),
        pd.Series(mbh_err),
        pd.Series(z),
        pd.Series(sz),
        M_SOURCE_FRAME_MIN,
        M_SOURCE_FRAME_MAX,
        Z_MAX,
    ).to_numpy()
    return z[mask], mbh[mask]


def process_venue(venue: str, m4: dict) -> dict[str, object]:
    """Compute the venue's outside-C-4 2D shares (M-4 convention verbatim)."""
    tv = time.time()
    ball_json = FROZENG / venue / "posteriors_with_bh_mass" / "h_0_73.json"
    wbh, n2 = load_wbh(ball_json)
    zz, mm = load_pruned_zm(CATS[venue])

    def w_of(idx: np.ndarray) -> np.ndarray:
        idx = np.asarray(idx, dtype=np.int64)
        return R_eff_per_mbh(mm[idx]) / (1.0 + zz[idx])

    # --- pair sets from m4 (verified against the global check) --------------
    outside_recs = [
        r for r in m4[venue]["global_pairs_shares"]["outside_c4_records"] if r["n_shared_2d"] > 0
    ]
    outside_set = {(r["i"], r["j"]) for r in outside_recs}
    gp2 = {tuple(p) for p in m4[venue]["global_sharing_check"]["outside_pairs_2d"]}
    assert outside_set == gp2, (
        f"{venue}: outside 2D sets disagree ({len(outside_set)} vs {len(gp2)})"
    )
    in_c4_recs = [r for r in m4[venue]["pair_records"] if r["n_shared_2d"] > 0]

    # --- per-event 2D ball w_pop / wN denominators --------------------------
    involved = sorted({e for r in outside_recs + in_c4_recs for e in (r["i"], r["j"])})
    den: dict[int, tuple[float, float]] = {}
    for ev in involved:
        idx = np.array(sorted(wbh[ev]), dtype=np.int64)
        w = w_of(idx)
        nv = np.array([n2[ev][g] for g in idx.tolist()])
        den[ev] = (float(w.sum()), float((w * nv).sum()))

    def pair_shares(recs: list[dict]) -> list[dict[str, object]]:
        rows = []
        for r in sorted(recs, key=lambda r: (r["i"], r["j"])):
            i, j = r["i"], r["j"]
            shared = wbh[i] & wbh[j]
            assert len(shared) == r["n_shared_2d"], (
                f"{venue} pair ({i},{j}): recomputed n_shared_2d "
                f"{len(shared)} != m4 {r['n_shared_2d']}"
            )
            idx = np.array(sorted(shared), dtype=np.int64)
            w_s = w_of(idx)
            sum_w = float(w_s.sum())
            row: dict[str, object] = {
                "i": i,
                "j": j,
                "n_shared_2d": len(shared),
                "in_c4": r.get("in_c4", True),
            }
            mins = []
            for side, ev in (("i", i), ("j", j)):
                dw, dwn = den[ev]
                nv = np.array([n2[ev][g] for g in idx.tolist()])
                share = sum_w / dw if dw > 0 else None
                row[f"w_share_2d_{side}"] = share
                row[f"wN_share_2d_{side}"] = float((w_s * nv).sum()) / dwn if dwn > 0 else None
                if share is not None:
                    mins.append(share)
            row["min_side_w_share_2d"] = min(mins) if len(mins) == 2 else None
            rows.append(row)
        return rows

    outside_rows = pair_shares(outside_recs)
    in_rows = pair_shares(in_c4_recs)

    # --- cross-check: in-C-4 W part reproduces band_derivation.json ---------
    w_in = sum(r["min_side_w_share_2d"] for r in in_rows)
    assert math.isclose(w_in, W_IN_C4_2D[venue], rel_tol=1e-12), (
        f"{venue}: in-C-4 2D min-side sum {w_in!r} != band_derivation {W_IN_C4_2D[venue]!r}"
    )

    w_out = sum(r["min_side_w_share_2d"] for r in outside_rows)
    w_total = w_in + w_out
    mins_out = [r["min_side_w_share_2d"] for r in outside_rows]
    sides_out = [
        r[k] for r in outside_rows for k in ("w_share_2d_i", "w_share_2d_j") if r[k] is not None
    ]

    result: dict[str, object] = {
        "inputs": {
            "ball_json_h073": str(ball_json),
            "catalogue": str(CATS[venue]),
            "n_catalogue_pruned_rows": int(len(zz)),
        },
        "n_outside_c4_2d_pairs": len(outside_rows),
        "n_in_c4_2d_pairs": len(in_rows),
        "outside_records": outside_rows,
        "outside_min_side_w_share_stats": qtiles(mins_out),
        "outside_w_share_per_pair_side_stats": qtiles(sides_out),
        "W_outside_minside_sum": w_out,
        "W_in_c4_minside_sum_recomputed": w_in,
        "W_in_c4_minside_sum_band_derivation": W_IN_C4_2D[venue],
        "W_total_2d": w_total,
        "per_unit_thresholds_locked_formula": {
            "X_locked": X_LOCKED,
            "Y_locked": Y_LOCKED,
            "x_neglect_per_unit": X_LOCKED / w_total,
            "y_regard_per_unit": Y_LOCKED / w_total,
            "note": (
                "x = X/W, y = Y/W per prereg §7.4 (locked formula; this "
                "read only supplies the W denominator)"
            ),
        },
        "runtime_s": round(time.time() - tv, 1),
    }
    print(
        f"{venue}: {len(outside_rows)} outside-C-4 2D pairs; "
        f"W_out = {w_out:.6f}, W_in = {w_in:.6f}, W_total = {w_total:.6f}; "
        f"min-side stats {result['outside_min_side_w_share_stats']}"
    )
    return result


def main() -> None:
    t0 = time.time()
    m4_sha = hashlib.sha256(M4_PATH.read_bytes()).hexdigest()
    assert m4_sha == M4_SHA256_PIN, f"m4_results.json sha256 {m4_sha} != prereg pin {M4_SHA256_PIN}"
    m4 = json.loads(M4_PATH.read_text())

    out: dict[str, object] = {
        "script": str(Path(__file__).resolve()),
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "purpose": (
            "Prereg §7.4 mandated pre-run read: min-side shared w_pop shares "
            "(M-4 convention) of the outside-C-4 2D-sharing pairs, completing "
            "the per-unit 2D denominators W = W_in_c4 + W_outside"
        ),
        "blindness": (
            "BLIND READ: no Delta, no quadrature, no likelihood was computed "
            "anywhere in this script; w_pop shares depend only on ball "
            "memberships and catalogue (z, M) columns and are independent of "
            "any cross-term value. crossterm_instrument.py has still never "
            "been executed with --confirm-run."
        ),
        "inputs": {"m4_results_sha256": m4_sha},
    }

    for venue in ("joint_r1", "iiib"):
        out[venue] = process_venue(venue, m4)

    out["total_runtime_s"] = round(time.time() - t0, 1)
    with open(OUT_PATH, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
