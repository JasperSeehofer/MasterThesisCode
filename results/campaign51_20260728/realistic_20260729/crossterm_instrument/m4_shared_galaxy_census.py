# ruff: noqa: B023, F821
"""M-4: exact shared-galaxy census over the C-4 candidate pair set.

C-4 (CLAIM_HITCHHIKER_INDEPENDENCE_20260805.md.DRAFT:277-287, spec :569-574, :677-680)
gave an UPPER BOUND of 279 pairs (sky-ball overlap + 2-sigma d_L-window) touching 385
of 1590 CRB events, computed WITHOUT galaxy lists. This script computes the TRUE
pairwise candidate-ball intersections using the EXACT per-event catalog_index sets
production consumed, read from the frozeng per-galaxy JSON emits
(results/run_20260804_frozeng/<venue>/posteriors_with_bh_mass/h_0_73.json,
keys 'galaxy_likelihoods' [with-BH ball] and 'additional_galaxies_without_bh_mass'
[no-BH-only remainder]; 1D ball = union, 2D ball = with-BH list; p_Di stores these
verbatim, bayesian_statistics.py:3866-3884).

Weights: w_g = R_eff_per_mbh(M_g)/(1+z_g) (bayesian_statistics._rate_weight), with
(z_g, M_g) dereferenced at catalog_index in a reconstruction of the venue handler's
pruned+reset frame (handler.py __init__ pipeline; sky rotation skipped — it never
changes row membership; mass mapping + NaN drop + prune mask reproduced by calling
the production functions themselves).

Venues: joint_r1 (observed_catalogue_seed900001.csv) and iiib
(cluster_parent_reduced_galaxy_catalogue.csv). NEVER intersects indices across venues.

Read-only w.r.t. production data; output: m4_results.json next to this script.
Run: cd /home/jasper/Repositories/MasterThesisCode && uv run python results/campaign51_20260728/realistic_20260729/crossterm_instrument/m4_shared_galaxy_census.py
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd

T0 = time.time()
REPO = "/home/jasper/Repositories/MasterThesisCode"
sys.path.insert(0, REPO)
os.chdir(REPO)

assert os.environ.get("MTC_HOST_QUAD_N") is None, "MTC_HOST_QUAD_N must be unset"

from master_thesis_code.emri_rate import R_eff_per_mbh  # noqa: E402
from master_thesis_code.galaxy_catalogue.handler import (  # noqa: E402
    _empiric_stellar_mass_to_BH_mass_relation,
    _mass_redshift_prune_mask,
    _reduced_catalog_column_names,
)
from master_thesis_code.physical_relations import get_redshift_outer_bounds  # noqa: E402

OUT_DIR = os.path.join(REPO, "results/campaign51_20260728/realistic_20260729/crossterm_instrument")
CRB_CSV = os.path.join(
    REPO, "results/run_20260804_postfix/joint_r1/diagnostics/prepared_cramer_rao_bounds.csv"
)
STAGED = os.path.join(REPO, "results/campaign51_20260728/realistic_20260729/realizations_staged")
FROZENG = os.path.join(REPO, "results/run_20260804_frozeng")
POSTFIX = os.path.join(REPO, "results/run_20260804_postfix")

VENUES = {
    "joint_r1": {
        "catalogue": os.path.join(STAGED, "observed_catalogue_seed900001.csv"),
        "json73": os.path.join(FROZENG, "joint_r1/posteriors_with_bh_mass/h_0_73.json"),
        "json60": os.path.join(FROZENG, "joint_r1/posteriors_with_bh_mass/h_0_6.json"),
        "diag": os.path.join(POSTFIX, "joint_r1/diagnostics/event_likelihoods.csv"),
    },
    "iiib": {
        "catalogue": os.path.join(STAGED, "cluster_parent_reduced_galaxy_catalogue.csv"),
        "json73": os.path.join(FROZENG, "iiib/posteriors_with_bh_mass/h_0_73.json"),
        "json60": os.path.join(FROZENG, "iiib/posteriors_with_bh_mass/h_0_6.json"),
        "diag": os.path.join(POSTFIX, "iiib/diagnostics/event_likelihoods.csv"),
    },
}

M_SOURCE_FRAME_MIN, M_SOURCE_FRAME_MAX, Z_MAX = 1e4, 1e7, 1.5


# ---------------------------------------------------------------- C-4 pair set
def c4_pairs():
    """Reproduce the C-4 census pair set exactly (recon_c4_census.py recipe)."""
    df = pd.read_csv(CRB_CSV)
    n = len(df)
    theta = df["qS"].to_numpy()
    phi = df["phiS"].to_numpy()
    s_phi2 = df["delta_phiS_delta_phiS"].to_numpy()
    s_theta2 = df["delta_qS_delta_qS"].to_numpy()
    cov = df["delta_phiS_delta_qS"].to_numpy()
    dl = df["luminosity_distance"].to_numpy()
    s_dl = np.sqrt(df["delta_luminosity_distance_delta_luminosity_distance"].to_numpy())

    r = np.empty(n)
    for i in range(n):
        sig = np.array([[s_phi2[i], cov[i]], [cov[i], s_theta2[i]]])
        jac = np.diag([abs(np.sin(theta[i])), 1.0])
        lam = float(np.linalg.eigvalsh(jac @ sig @ jac.T).max())
        r[i] = 2.0 * np.sqrt(max(lam, 0.0))

    st = np.sin(theta)
    xyz = np.stack([st * np.cos(phi), st * np.sin(phi), np.cos(theta)], axis=1)
    d = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=2)
    iu = np.triu_indices(n, k=1)
    sky = d[iu] <= (r[:, None] + r[None, :])[iu]
    ii, jj = iu[0][sky], iu[1][sky]
    lo, hi = dl - 2 * s_dl, dl + 2 * s_dl
    win = (lo[ii] <= hi[jj]) & (lo[jj] <= hi[ii])
    pairs = list(zip(ii[win].tolist(), jj[win].tolist()))
    touched = sorted(set(ii[win].tolist()) | set(jj[win].tolist()))
    return df, pairs, touched, int(sky.sum())


# ------------------------------------------------------- frozeng ball sets
def load_balls(path):
    """Return (wbh_sets, add_sets, N1d_maps, N2d_maps) keyed by int event_idx.

    wbh rows: [catalog_index, [N_nobh, D_nobh, N_wbh, D_wbh, q_num, q_den]]
    add rows: [catalog_index, [N_nobh, D_nobh, q_num, q_den]]
    """
    d = json.load(open(path))
    gl, add = d["galaxy_likelihoods"], d["additional_galaxies_without_bh_mass"]
    wbh_sets, add_sets, n1, n2 = {}, {}, {}, {}
    for k in gl:
        ev = int(k)
        rows_g, rows_a = gl[k], add[k]
        wbh_sets[ev] = frozenset(r[0] for r in rows_g)
        add_sets[ev] = frozenset(r[0] for r in rows_a)
        m1 = {r[0]: r[1][0] for r in rows_a}
        m1.update({r[0]: r[1][0] for r in rows_g})
        n1[ev] = m1
        n2[ev] = {r[0]: r[1][2] for r in rows_g}
    return wbh_sets, add_sets, n1, n2, float(d["h"])


# ------------------------------------------------ pruned catalogue z/M frame
def load_pruned_zm(path):
    """Reconstruct the venue handler's pruned+reset frame (z, sigma_z, M_bh, sigma_M).

    Replicates handler __init__ order: read -> _map_stellar_masses_to_BH_masses ->
    (rotation: membership-neutral, skipped) -> _remove_galaxies_without_mass_information
    -> _mass_redshift_prune_mask -> positional reset. Uses the production functions.
    """
    names = _reduced_catalog_column_names()
    cat = pd.read_csv(path, names=names, usecols=[3, 4, 5, 6])
    n_raw = len(cat)
    z = cat["REDSHIFT"].to_numpy(dtype=np.float64)
    sz = cat["REDSHIFT_MEASUREMENT_ERROR"].to_numpy(dtype=np.float64)
    ms = cat["STELLAR_MASS"].to_numpy(dtype=np.float64)
    mse = cat["STELLAR_MASS_ABSOULTE_ERROR"].to_numpy(dtype=np.float64)
    del cat
    mbh, mbh_err = _empiric_stellar_mass_to_BH_mass_relation(ms, mse)
    del ms, mse
    keep_nan = ~np.isnan(mbh)
    z, sz, mbh, mbh_err = z[keep_nan], sz[keep_nan], mbh[keep_nan], mbh_err[keep_nan]
    mask = _mass_redshift_prune_mask(
        pd.Series(mbh),
        pd.Series(mbh_err),
        pd.Series(z),
        pd.Series(sz),
        M_SOURCE_FRAME_MIN,
        M_SOURCE_FRAME_MAX,
        Z_MAX,
    ).to_numpy()
    return {
        "n_raw": n_raw,
        "z": z[mask],
        "sz": sz[mask],
        "M": mbh[mask],
        "sM": mbh_err[mask],
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


def hist_counts(a, edges_labels):
    out = {}
    for lab, pred in edges_labels:
        out[lab] = int(sum(1 for x in a if pred(x)))
    return out


COUNT_BINS = [
    ("0", lambda x: x == 0),
    ("1", lambda x: x == 1),
    ("2-10", lambda x: 2 <= x <= 10),
    ("11-100", lambda x: 11 <= x <= 100),
    ("101-1000", lambda x: 101 <= x <= 1000),
    (">1000", lambda x: x > 1000),
]


def main():
    results = {
        "script": __file__,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    crb, pairs, touched, n_sky = c4_pairs()
    assert len(pairs) == 279, f"C-4 pair reproduction FAILED: {len(pairs)} != 279"
    assert len(touched) == 385, f"C-4 touched-event reproduction FAILED: {len(touched)} != 385"
    print(
        f"[{time.time() - T0:7.1f}s] C-4 pair set reproduced: {n_sky} sky pairs -> "
        f"{len(pairs)} sky+dL pairs touching {len(touched)} events"
    )
    results["c4_reproduction"] = {
        "crb_csv": CRB_CSV,
        "n_crb_rows": len(crb),
        "n_sky_pairs": n_sky,
        "n_pairs": len(pairs),
        "n_touched_events": len(touched),
    }

    dl = crb["luminosity_distance"].to_numpy()
    sdl = np.sqrt(crb["delta_luminosity_distance_delta_luminosity_distance"].to_numpy())
    mz = crb["M"].to_numpy()
    smz = np.sqrt(crb["delta_M_delta_M"].to_numpy())

    rng = np.random.default_rng(20260805)

    for venue, cfg in VENUES.items():
        tv = time.time()
        wbh_sets, add_sets, n1_maps, n2_maps, h73 = load_balls(cfg["json73"])
        assert abs(h73 - 0.73) < 1e-12
        evaluated = set(wbh_sets)
        dropped = sorted(set(range(len(crb))) - evaluated)
        ball_sets = {ev: wbh_sets[ev] | add_sets[ev] for ev in evaluated}
        # V-dedup: with-BH and additional lists must be disjoint (trap 10)
        n_overlap = sum(len(wbh_sets[ev] & add_sets[ev]) for ev in evaluated)
        assert n_overlap == 0, f"{venue}: wbh/additional lists overlap ({n_overlap} rows)"

        # V2: ball h-invariance vs h=0.60 file
        wbh60, add60, _, _, h60 = load_balls(cfg["json60"])
        assert abs(h60 - 0.60) < 1e-12
        h_invariant = set(wbh60) == evaluated and all(
            wbh60[ev] == wbh_sets[ev] and add60[ev] == add_sets[ev] for ev in evaluated
        )
        del wbh60, add60

        # V3: zero-ball events == L_cat_no_bh==0 count; zero-wbh == L_cat_with_bh==0
        diag = pd.read_csv(cfg["diag"])
        d73 = diag[np.isclose(diag["h"], 0.73)]
        n_zero_1d_diag = int((d73["L_cat_no_bh"] == 0).sum())
        n_zero_2d_diag = int((d73["L_cat_with_bh"] == 0).sum())
        n_zero_1d_ball = sum(1 for ev in evaluated if len(ball_sets[ev]) == 0)
        n_zero_2d_ball = sum(1 for ev in evaluated if len(wbh_sets[ev]) == 0)
        del diag, d73

        # pair census
        pair_rows = []
        involved = set()
        for i, j in pairs:
            if i not in evaluated or j not in evaluated:
                continue
            s1 = ball_sets[i] & ball_sets[j]
            s2 = wbh_sets[i] & wbh_sets[j]
            pair_rows.append({"i": i, "j": j, "shared_1d": s1, "shared_2d": s2})
            if s1:
                involved.add(i)
                involved.add(j)
        n_pairs_eval = len(pair_rows)
        n_pairs_shared_1d = sum(1 for p in pair_rows if p["shared_1d"])
        n_pairs_shared_2d = sum(1 for p in pair_rows if p["shared_2d"])
        print(
            f"[{time.time() - T0:7.1f}s] {venue}: {n_pairs_eval} evaluated pairs; "
            f"{n_pairs_shared_1d} share >=1 galaxy (1D), {n_pairs_shared_2d} (2D)"
        )

        # catalogue frame for w_g
        zm = load_pruned_zm(cfg["catalogue"])
        n_pruned = len(zm["z"])
        max_idx_seen = max((max(b) for b in ball_sets.values() if b), default=-1)
        assert max_idx_seen < n_pruned, (
            f"{venue}: catalog_index {max_idx_seen} >= pruned frame size {n_pruned}"
        )
        print(
            f"[{time.time() - T0:7.1f}s] {venue}: catalogue raw {zm['n_raw']} -> pruned {n_pruned} rows"
        )

        def w_of(idx_arr):
            idx_arr = np.asarray(idx_arr, dtype=np.int64)
            return R_eff_per_mbh(zm["M"][idx_arr]) / (1.0 + zm["z"][idx_arr])

        # V4: frame-alignment predicate check on sampled events with nonempty wbh
        cand = [ev for ev in evaluated if len(wbh_sets[ev]) > 0]
        sample = list(rng.choice(cand, size=min(25, len(cand)), replace=False))
        # ensure biggest-ball event included
        sample.append(max(cand, key=lambda ev: len(ball_sets[ev])))
        v4_checked = v4_z_ok = v4_m_ok = v4_add_mass_excluded = v4_add_total = 0
        for ev in sample:
            z_min, z_max = get_redshift_outer_bounds(
                distance=dl[ev],
                distance_error=sdl[ev],
                h_min=0.6,
                h_max=0.86,
                Omega_m_min=0.04,
                Omega_m_max=0.5,
                sigma_multiplier=2.0,
            )
            z_max = min(z_max, Z_MAX)
            idx1 = np.array(sorted(ball_sets[ev]), dtype=np.int64)
            zg, szg = zm["z"][idx1], zm["sz"][idx1]
            z_ok = (z_min <= zg + szg) & (z_max >= zg - szg)
            v4_checked += len(idx1)
            v4_z_ok += int(z_ok.sum())
            idxw = np.array(sorted(wbh_sets[ev]), dtype=np.int64)
            mg, smg = zm["M"][idxw], zm["sM"][idxw]
            lo_band = (mz[ev] - 1.5 * smz[ev]) / (1 + z_max)
            hi_band = (mz[ev] + 1.5 * smz[ev]) / (1 + z_min)
            m_ok = (lo_band <= mg + smg) & (mg - smg <= hi_band)
            v4_m_ok += int(m_ok.sum())
            idxa = np.array(sorted(add_sets[ev]), dtype=np.int64)
            if len(idxa):
                ma, sma = zm["M"][idxa], zm["sM"][idxa]
                m_fail = ~((lo_band <= ma + sma) & (ma - sma <= hi_band))
                v4_add_total += len(idxa)
                v4_add_mass_excluded += int(m_fail.sum())
        v4 = {
            "n_events_sampled": len(sample),
            "ball_members_checked": v4_checked,
            "z_predicate_pass": v4_z_ok,
            "wbh_mass_predicate_pass": v4_m_ok,
            "wbh_members_checked": int(sum(len(wbh_sets[ev]) for ev in sample)),
            "additional_members_checked": v4_add_total,
            "additional_mass_predicate_excluded": v4_add_mass_excluded,
        }
        print(f"[{time.time() - T0:7.1f}s] {venue}: V4 frame check {v4}")

        # per-event ball denominators (only involved events)
        ev_denoms = {}
        for ev in involved:
            idx1 = np.array(sorted(ball_sets[ev]), dtype=np.int64)
            w1 = w_of(idx1)
            n1 = np.array([n1_maps[ev][g] for g in idx1.tolist()])
            idx2 = np.array(sorted(wbh_sets[ev]), dtype=np.int64)
            if len(idx2):
                w2 = w_of(idx2)
                n2 = np.array([n2_maps[ev][g] for g in idx2.tolist()])
                den_w2, den_wn2 = float(w2.sum()), float((w2 * n2).sum())
            else:
                den_w2, den_wn2 = 0.0, 0.0
            ev_denoms[ev] = {
                "n_ball_1d": len(idx1),
                "n_ball_2d": len(idx2),
                "den_w1": float(w1.sum()),
                "den_wn1": float((w1 * n1).sum()),
                "den_w2": den_w2,
                "den_wn2": den_wn2,
            }

        # per-pair shares
        pair_records = []
        ev_union_shared = {}
        for p in pair_rows:
            i, j = p["i"], p["j"]
            s1, s2 = p["shared_1d"], p["shared_2d"]
            rec = {
                "i": i,
                "j": j,
                "n_shared_1d": len(s1),
                "n_shared_2d": len(s2),
                "n_ball_1d_i": len(ball_sets[i]),
                "n_ball_1d_j": len(ball_sets[j]),
                "n_ball_2d_i": len(wbh_sets[i]),
                "n_ball_2d_j": len(wbh_sets[j]),
            }
            if s1:
                idx = np.array(sorted(s1), dtype=np.int64)
                w_s = w_of(idx)
                sum_w = float(w_s.sum())
                n_eff_1d_i = n_eff_1d_j = 0
                for side, ev in (("i", i), ("j", j)):
                    d = ev_denoms[ev]
                    n_vals = np.array([n1_maps[ev][g] for g in idx.tolist()])
                    if side == "i":
                        n_eff_1d_i = int((n_vals > 0).sum())
                    else:
                        n_eff_1d_j = int((n_vals > 0).sum())
                    rec[f"w_share_1d_{side}"] = sum_w / d["den_w1"] if d["den_w1"] > 0 else None
                    rec[f"wN_share_1d_{side}"] = (
                        float((w_s * n_vals).sum()) / d["den_wn1"] if d["den_wn1"] > 0 else None
                    )
                rec["n_shared_1d_Npos_i"] = n_eff_1d_i
                rec["n_shared_1d_Npos_j"] = n_eff_1d_j
                ev_union_shared.setdefault(i, set()).update(s1)
                ev_union_shared.setdefault(j, set()).update(s1)
            if s2:
                idx = np.array(sorted(s2), dtype=np.int64)
                w_s = w_of(idx)
                sum_w = float(w_s.sum())
                both_pos = 0
                nv_i = np.array([n2_maps[i][g] for g in idx.tolist()])
                nv_j = np.array([n2_maps[j][g] for g in idx.tolist()])
                both_pos = int(((nv_i > 0) & (nv_j > 0)).sum())
                rec["n_shared_2d_Npos_both"] = both_pos
                for side, ev, nv in (("i", i, nv_i), ("j", j, nv_j)):
                    d = ev_denoms[ev]
                    rec[f"w_share_2d_{side}"] = sum_w / d["den_w2"] if d["den_w2"] > 0 else None
                    rec[f"wN_share_2d_{side}"] = (
                        float((w_s * nv).sum()) / d["den_wn2"] if d["den_wn2"] > 0 else None
                    )
            pair_records.append(rec)

        # per-event union-of-shared w_pop share (negligibility denominator)
        ev_records = []
        for ev, shared_union in sorted(ev_union_shared.items()):
            d = ev_denoms[ev]
            idx = np.array(sorted(shared_union), dtype=np.int64)
            w_s = w_of(idx)
            n_vals = np.array([n1_maps[ev][g] for g in idx.tolist()])
            ev_records.append(
                {
                    "event_idx": ev,
                    "n_shared_union_1d": len(idx),
                    "n_ball_1d": d["n_ball_1d"],
                    "w_share_union_1d": float(w_s.sum()) / d["den_w1"] if d["den_w1"] > 0 else None,
                    "wN_share_union_1d": (
                        float((w_s * n_vals).sum()) / d["den_wn1"] if d["den_wn1"] > 0 else None
                    ),
                }
            )

        shared1 = [r["n_shared_1d"] for r in pair_records]
        shared2 = [r["n_shared_2d"] for r in pair_records]
        w_shares_1d = [
            r[k]
            for r in pair_records
            for k in ("w_share_1d_i", "w_share_1d_j")
            if r.get(k) is not None
        ]
        wn_shares_1d = [
            r[k]
            for r in pair_records
            for k in ("wN_share_1d_i", "wN_share_1d_j")
            if r.get(k) is not None
        ]
        w_shares_2d = [
            r[k]
            for r in pair_records
            for k in ("w_share_2d_i", "w_share_2d_j")
            if r.get(k) is not None
        ]
        wn_shares_2d = [
            r[k]
            for r in pair_records
            for k in ("wN_share_2d_i", "wN_share_2d_j")
            if r.get(k) is not None
        ]
        # pairwise suppression proxies (1D): min-side and geometric-mean of w-shares
        supp_min_w, supp_geo_w, supp_min_wn, supp_geo_wn = [], [], [], []
        for r in pair_records:
            a, b = r.get("w_share_1d_i"), r.get("w_share_1d_j")
            if a is not None and b is not None:
                supp_min_w.append(min(a, b))
                supp_geo_w.append(float(np.sqrt(a * b)))
            a, b = r.get("wN_share_1d_i"), r.get("wN_share_1d_j")
            if a is not None and b is not None:
                supp_min_wn.append(min(a, b))
                supp_geo_wn.append(float(np.sqrt(a * b)))
        count_frac = [
            r["n_shared_1d"] / min(r["n_ball_1d_i"], r["n_ball_1d_j"])
            for r in pair_records
            if r["n_shared_1d"] > 0 and min(r["n_ball_1d_i"], r["n_ball_1d_j"]) > 0
        ]

        results[venue] = {
            "inputs": {
                "ball_json_h073": cfg["json73"],
                "ball_json_h060_check": cfg["json60"],
                "catalogue": cfg["catalogue"],
                "diagnostics_csv": cfg["diag"],
            },
            "n_events_evaluated": len(evaluated),
            "dropped_crb_row_indices": dropped,
            "n_catalogue_raw_rows": zm["n_raw"],
            "n_catalogue_pruned_rows": n_pruned,
            "validation": {
                "V2_ball_sets_h_invariant_073_vs_060": bool(h_invariant),
                "V3_zero_1d_ball_events": n_zero_1d_ball,
                "V3_L_cat_no_bh_zero_at_h073": n_zero_1d_diag,
                "V3_zero_2d_ball_events": n_zero_2d_ball,
                "V3_L_cat_with_bh_zero_at_h073": n_zero_2d_diag,
                "V4_frame_alignment": v4,
                "V_wbh_additional_disjoint": True,
                "V_max_catalog_index_seen": int(max_idx_seen),
            },
            "pair_census": {
                "n_c4_pairs": len(pairs),
                "n_pairs_both_evaluated": n_pairs_eval,
                "n_pairs_dropped_endpoint": len(pairs) - n_pairs_eval,
                "n_pairs_shared_ge1_1d": n_pairs_shared_1d,
                "n_pairs_shared_ge1_2d": n_pairs_shared_2d,
                "n_events_involved_1d": len(involved),
                "shared_count_1d_stats": qtiles(shared1),
                "shared_count_1d_hist": hist_counts(shared1, COUNT_BINS),
                "shared_count_2d_stats": qtiles(shared2),
                "shared_count_2d_hist": hist_counts(shared2, COUNT_BINS),
            },
            "shares": {
                "w_share_1d_per_pair_side": qtiles(w_shares_1d),
                "wN_share_1d_per_pair_side": qtiles(wn_shares_1d),
                "w_share_2d_per_pair_side": qtiles(w_shares_2d),
                "wN_share_2d_per_pair_side": qtiles(wn_shares_2d),
                "count_share_1d_min_side": qtiles(count_frac),
                "suppression_1d_min_w_share": qtiles(supp_min_w),
                "suppression_1d_geomean_w_share": qtiles(supp_geo_w),
                "suppression_1d_min_wN_share": qtiles(supp_min_wn),
                "suppression_1d_geomean_wN_share": qtiles(supp_geo_wn),
            },
            "per_event_union": {
                "records": ev_records,
                "w_share_union_1d_stats": qtiles(
                    [r["w_share_union_1d"] for r in ev_records if r["w_share_union_1d"] is not None]
                ),
                "wN_share_union_1d_stats": qtiles(
                    [
                        r["wN_share_union_1d"]
                        for r in ev_records
                        if r["wN_share_union_1d"] is not None
                    ]
                ),
            },
            "pair_records": pair_records,
            "runtime_s": round(time.time() - tv, 1),
        }
        del zm, wbh_sets, add_sets, ball_sets, n1_maps, n2_maps, ev_denoms
        print(f"[{time.time() - T0:7.1f}s] {venue} done.")

    results["total_runtime_s"] = round(time.time() - T0, 1)
    out_path = os.path.join(OUT_DIR, "m4_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=1)
    print(f"[{time.time() - T0:7.1f}s] wrote {out_path}")


if __name__ == "__main__":
    main()
