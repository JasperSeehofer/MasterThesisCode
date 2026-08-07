# ruff: noqa: B023, F821
"""M-4 follow-up 2: shares + C-4-predicate diagnosis for the GLOBAL sharing pair set.

For every event pair that truly shares >=1 catalogue galaxy (1D ball, from
m4_global_sharing_check inverted index), compute the same per-pair-side w_pop /
w*N shares as m4_shared_galaxy_census.py, and for pairs OUTSIDE the C-4 set
diagnose which C-4 predicate they fail (sky 2sigma-chord-sum vs d_L 2sigma
window). Updates m4_results.json in place with 'global_pairs_shares' per venue.

Run: cd /home/jasper/Repositories/MasterThesisCode && uv run python results/campaign51_20260728/realistic_20260729/crossterm_instrument/m4_global_shares.py
"""

import json
import os
import sys
import time
from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd

T0 = time.time()
REPO = "/home/jasper/Repositories/MasterThesisCode"
sys.path.insert(0, REPO)
os.chdir(REPO)

from master_thesis_code.emri_rate import R_eff_per_mbh  # noqa: E402
from master_thesis_code.galaxy_catalogue.handler import (  # noqa: E402
    _empiric_stellar_mass_to_BH_mass_relation,
    _mass_redshift_prune_mask,
    _reduced_catalog_column_names,
)

OUT = os.path.join(
    REPO, "results/campaign51_20260728/realistic_20260729/crossterm_instrument/m4_results.json"
)
CRB_CSV = os.path.join(
    REPO, "results/run_20260804_postfix/joint_r1/diagnostics/prepared_cramer_rao_bounds.csv"
)
STAGED = os.path.join(REPO, "results/campaign51_20260728/realistic_20260729/realizations_staged")
FROZENG = os.path.join(REPO, "results/run_20260804_frozeng")
CATS = {
    "joint_r1": os.path.join(STAGED, "observed_catalogue_seed900001.csv"),
    "iiib": os.path.join(STAGED, "cluster_parent_reduced_galaxy_catalogue.csv"),
}


def qtiles(a):
    a = np.asarray(a, dtype=float)
    if a.size == 0:
        return None
    return {
        "n": int(a.size),
        "min": float(a.min()),
        "median": float(np.median(a)),
        "mean": float(a.mean()),
        "p90": float(np.percentile(a, 90)),
        "p99": float(np.percentile(a, 99)),
        "max": float(a.max()),
    }


def main():
    crb = pd.read_csv(CRB_CSV)
    n = len(crb)
    theta = crb["qS"].to_numpy()
    phi = crb["phiS"].to_numpy()
    s_phi2 = crb["delta_phiS_delta_phiS"].to_numpy()
    s_theta2 = crb["delta_qS_delta_qS"].to_numpy()
    covtp = crb["delta_phiS_delta_qS"].to_numpy()
    dl = crb["luminosity_distance"].to_numpy()
    s_dl = np.sqrt(crb["delta_luminosity_distance_delta_luminosity_distance"].to_numpy())
    r2 = np.empty(n)
    for i in range(n):
        sig = np.array([[s_phi2[i], covtp[i]], [covtp[i], s_theta2[i]]])
        jac = np.diag([abs(np.sin(theta[i])), 1.0])
        r2[i] = 2.0 * np.sqrt(max(float(np.linalg.eigvalsh(jac @ sig @ jac.T).max()), 0.0))
    st = np.sin(theta)
    xyz = np.stack([st * np.cos(phi), st * np.sin(phi), np.cos(theta)], axis=1)

    results = json.load(open(OUT))
    c4_set = set()
    # rebuild C-4 set from stored pair_records is insufficient (only evaluated);
    # rebuild via predicates:
    d = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=2)
    iu = np.triu_indices(n, k=1)
    sky_mask = d[iu] <= (r2[:, None] + r2[None, :])[iu]
    ii, jj = iu[0][sky_mask], iu[1][sky_mask]
    lo, hi = dl - 2 * s_dl, dl + 2 * s_dl
    win = (lo[ii] <= hi[jj]) & (lo[jj] <= hi[ii])
    c4_set = set(zip(ii[win].tolist(), jj[win].tolist()))
    assert len(c4_set) == 279

    for venue in ("joint_r1", "iiib"):
        jp = os.path.join(FROZENG, venue, "posteriors_with_bh_mass/h_0_73.json")
        dj = json.load(open(jp))
        gl, add = dj["galaxy_likelihoods"], dj["additional_galaxies_without_bh_mass"]
        ball, wbh, n1 = {}, {}, {}
        inv = defaultdict(list)
        for k in gl:
            ev = int(k)
            wbh[ev] = frozenset(r[0] for r in gl[k])
            ball[ev] = wbh[ev] | frozenset(r[0] for r in add[k])
            m = {r[0]: r[1][0] for r in add[k]}
            m.update({r[0]: r[1][0] for r in gl[k]})
            n1[ev] = m
            for g in ball[ev]:
                inv[g].append(ev)
        pairs = set()
        for g, evs in inv.items():
            if len(evs) > 1:
                for a, b in combinations(sorted(set(evs)), 2):
                    pairs.add((a, b))
        del inv, dj, gl, add

        names = _reduced_catalog_column_names()
        cat = pd.read_csv(CATS[venue], names=names, usecols=[3, 4, 5, 6])
        z = cat["REDSHIFT"].to_numpy(np.float64)
        sz = cat["REDSHIFT_MEASUREMENT_ERROR"].to_numpy(np.float64)
        ms = cat["STELLAR_MASS"].to_numpy(np.float64)
        mse = cat["STELLAR_MASS_ABSOULTE_ERROR"].to_numpy(np.float64)
        del cat
        mbh, mbh_err = _empiric_stellar_mass_to_BH_mass_relation(ms, mse)
        keep = ~np.isnan(mbh)
        z, sz, mbh, mbh_err = z[keep], sz[keep], mbh[keep], mbh_err[keep]
        mask = _mass_redshift_prune_mask(
            pd.Series(mbh), pd.Series(mbh_err), pd.Series(z), pd.Series(sz), 1e4, 1e7, 1.5
        ).to_numpy()
        zz, mm = z[mask], mbh[mask]
        del z, sz, mbh, mbh_err, ms, mse

        def w_of(idx):
            idx = np.asarray(idx, dtype=np.int64)
            return R_eff_per_mbh(mm[idx]) / (1.0 + zz[idx])

        den = {}
        involved = sorted({e for p in pairs for e in p})
        for ev in involved:
            idx = np.array(sorted(ball[ev]), dtype=np.int64)
            w = w_of(idx)
            nv = np.array([n1[ev][g] for g in idx.tolist()])
            den[ev] = (float(w.sum()), float((w * nv).sum()))

        recs = []
        for i, j in sorted(pairs):
            s = ball[i] & ball[j]
            idx = np.array(sorted(s), dtype=np.int64)
            w = w_of(idx)
            sw = float(w.sum())
            rec = {
                "i": i,
                "j": j,
                "n_shared_1d": len(s),
                "in_c4": (i, j) in c4_set,
                "n_shared_2d": len(wbh[i] & wbh[j]),
            }
            for side, ev in (("i", i), ("j", j)):
                dw, dwn = den[ev]
                nv = np.array([n1[ev][g] for g in idx.tolist()])
                rec[f"w_share_1d_{side}"] = sw / dw if dw > 0 else None
                rec[f"wN_share_1d_{side}"] = float((w * nv).sum()) / dwn if dwn > 0 else None
            if not rec["in_c4"]:
                chord = float(np.linalg.norm(xyz[i] - xyz[j]))
                rec["c4_sky_pass"] = bool(chord <= r2[i] + r2[j])
                rec["c4_dl_pass"] = bool(abs(dl[i] - dl[j]) <= 2 * (s_dl[i] + s_dl[j]))
            recs.append(rec)

        outside = [r for r in recs if not r["in_c4"]]
        inside = [r for r in recs if r["in_c4"]]
        sky_pass = sum(1 for r in outside if r["c4_sky_pass"])
        dl_pass = sum(1 for r in outside if r["c4_dl_pass"])

        def share_stats(rs):
            ws = [
                r[k] for r in rs for k in ("w_share_1d_i", "w_share_1d_j") if r.get(k) is not None
            ]
            wns = [
                r[k] for r in rs for k in ("wN_share_1d_i", "wN_share_1d_j") if r.get(k) is not None
            ]
            mins_w = [
                min(r["w_share_1d_i"], r["w_share_1d_j"])
                for r in rs
                if r.get("w_share_1d_i") is not None and r.get("w_share_1d_j") is not None
            ]
            return {
                "n_pairs": len(rs),
                "shared_count_stats": qtiles([r["n_shared_1d"] for r in rs]),
                "w_share_per_pair_side": qtiles(ws),
                "wN_share_per_pair_side": qtiles(wns),
                "min_side_w_share": qtiles(mins_w),
            }

        results[venue]["global_pairs_shares"] = {
            "n_global_sharing_pairs_1d": len(recs),
            "n_in_c4": len(inside),
            "n_outside_c4": len(outside),
            "outside_c4_sky_predicate_pass": sky_pass,
            "outside_c4_dl_predicate_pass": dl_pass,
            "outside_c4_records": outside,
            "stats_in_c4": share_stats(inside),
            "stats_outside_c4": share_stats(outside),
            "stats_all": share_stats(recs),
        }
        print(
            f"[{time.time() - T0:6.1f}s] {venue}: {len(recs)} global sharing pairs "
            f"({len(inside)} in C-4, {len(outside)} outside; of outside: sky-pass "
            f"{sky_pass}, dL-pass {dl_pass})"
        )
        del ball, wbh, n1, zz, mm, den

    results["global_shares_runtime_s"] = round(time.time() - T0, 1)
    with open(OUT, "w") as f:
        json.dump(results, f, indent=1)
    print(f"[{time.time() - T0:6.1f}s] updated {OUT}")


if __name__ == "__main__":
    main()
