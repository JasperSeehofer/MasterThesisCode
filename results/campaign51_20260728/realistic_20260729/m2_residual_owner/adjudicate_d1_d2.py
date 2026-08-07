"""ADVERSARIAL VERIFICATION of the two stage-1 reads D-1 (component decomposition) and
D-2 (extended-covariate confounding check) on the M-2 matched 2D overlap residual.

Independent implementation: reuses NOTHING from d1_component_decomposition.py /
d2_confounding_check.py (their JSONs are read only for comparison at the end).
Production modules (emri_rate.R_eff_per_mbh, galaxy_catalogue.handler helpers) and the
committed M-2/M-4 SPEC (census recipe, matching definition, w_g convention) are the
established record and are re-implemented here from spec, not copied.

Checks:
  (1) D-1 headline: LMDI additive decomposition of the combined chord, top component
      (T_Lcomp) matched diff + fraction of total, closure to stated tolerance.
  (2) D-2 baseline reproduction (must equal +0.02225/+0.02070) and final-rung (m4)
      residual + p; all rungs recomputed (m1, m2, m3, m3e, m4) with fresh RNG.
  (3) Traps: zero-coverage components reported as null; balance degradation at richer
      rungs (10-covariate SMD tables at every rung); p-hacking via rung choice (full
      single-covariate panel + an adversarial geometry-only combined rung m_geo);
      over-matching absorption diagnostics.

FREE READS ONLY: existing CSVs/JSONs; no cluster, no likelihood evaluations.
Output: adjudication_results.json in this directory. RNG seed 424242 (fresh,
independent of M-2's 20260805 / D-1's 20260805/99 / D-2's 20260807).

Run: cd /home/jasper/Repositories/MasterThesisCode && uv run python \
    results/campaign51_20260728/realistic_20260729/m2_residual_owner/adjudicate_d1_d2.py
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = "/home/jasper/Repositories/MasterThesisCode"
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from master_thesis_code.emri_rate import R_eff_per_mbh  # noqa: E402
from master_thesis_code.galaxy_catalogue.handler import (  # noqa: E402
    _empiric_stellar_mass_to_BH_mass_relation,
    _mass_redshift_prune_mask,
    _reduced_catalog_column_names,
)

HERE = f"{ROOT}/results/campaign51_20260728/realistic_20260729/m2_residual_owner"
OUT = f"{HERE}/adjudication_results.json"
SCRATCH = (
    "/tmp/claude-1000/-home-jasper-Repositories-MasterThesisCode/"
    "512252d8-926d-429a-ac6b-9d0701dbb800/scratchpad"
)
os.makedirs(SCRATCH, exist_ok=True)

CRB_PATH = (
    f"{ROOT}/results/run_20260804_postfix/joint_r1/diagnostics/prepared_cramer_rao_bounds.csv"
)
EL_PATHS = {
    "iiib": f"{ROOT}/results/run_20260804_postfix/iiib/diagnostics/event_likelihoods.csv",
    "joint_r1": f"{ROOT}/results/run_20260804_postfix/joint_r1/diagnostics/event_likelihoods.csv",
}
BALL_PATHS = {
    "iiib": f"{ROOT}/results/run_20260804_frozeng/iiib/posteriors_with_bh_mass/h_0_73.json",
    "joint_r1": f"{ROOT}/results/run_20260804_frozeng/joint_r1/posteriors_with_bh_mass/h_0_73.json",
}
STAGED = f"{ROOT}/results/campaign51_20260728/realistic_20260729/realizations_staged"
CATS = {
    "joint_r1": f"{STAGED}/observed_catalogue_seed900001.csv",
    "iiib": f"{STAGED}/cluster_parent_reduced_galaxy_catalogue.csv",
}
M_MIN, M_MAX, Z_MAX = 1e4, 1e7, 1.5

SEED = 424242
NPERM = 20000
RNG = np.random.default_rng(SEED)

t0 = time.time()
res: dict = {
    "read": "adversarial verification of D-1 and D-2 (independent implementation)",
    "seed": SEED,
    "n_perm": NPERM,
}


# ------------------------------------------------------------------ stats helpers
def signflip_p(diffs: np.ndarray) -> float:
    obs = abs(diffs.mean())
    signs = RNG.integers(0, 2, size=(NPERM, len(diffs))) * 2 - 1
    perm = np.abs((signs * diffs[None, :]).mean(axis=1))
    return float((np.count_nonzero(perm >= obs) + 1) / (NPERM + 1))


def cluster_signflip_p(diffs: np.ndarray, clusters: np.ndarray) -> float:
    obs = abs(diffs.mean())
    _, inv = np.unique(clusters, return_inverse=True)
    g = inv.max() + 1
    sums = np.zeros(g)
    np.add.at(sums, inv, diffs)
    signs = RNG.integers(0, 2, size=(NPERM, g)) * 2 - 1
    perm = np.abs(signs @ sums) / len(diffs)
    return float((np.count_nonzero(perm >= obs) + 1) / (NPERM + 1))


def smd(a: np.ndarray, b: np.ndarray) -> float:
    sp = np.sqrt(0.5 * (a.var(ddof=1) + b.var(ddof=1)))
    return float((a.mean() - b.mean()) / sp) if sp > 0 else 0.0


def paired_stats(diffs: np.ndarray, clusters: np.ndarray) -> dict:
    return {
        "mean_paired_diff": float(diffs.mean()),
        "median_paired_diff": float(np.median(diffs)),
        "se": float(diffs.std(ddof=1) / np.sqrt(len(diffs))),
        "n_pairs": int(len(diffs)),
        "signflip_p": signflip_p(diffs),
        "cluster_signflip_p": cluster_signflip_p(diffs, clusters),
    }


# ------------------------------------------------------------------ C-4 census (own code)
crb = pd.read_csv(CRB_PATH)
assert len(crb) == 1590
theta = crb["qS"].to_numpy()
phi = crb["phiS"].to_numpy()
a2 = crb["delta_phiS_delta_phiS"].to_numpy()  # sigma_phi^2
b2 = crb["delta_qS_delta_qS"].to_numpy()  # sigma_theta^2
c12 = crb["delta_phiS_delta_qS"].to_numpy()
dl = crb["luminosity_distance"].to_numpy()
s_dl = np.sqrt(crb["delta_luminosity_distance_delta_luminosity_distance"].to_numpy())
snr = crb["SNR"].to_numpy()

# closed-form max eigenvalue of J Sigma J^T, J = diag(|sin theta|, 1)
s = np.abs(np.sin(theta))
aa = s * s * a2
bb = b2
cc = s * c12
lam_max = 0.5 * ((aa + bb) + np.sqrt((aa - bb) ** 2 + 4 * cc * cc))
radius = 2.0 * np.sqrt(np.maximum(lam_max, 0.0))

assert (np.sin(theta) >= 0).all(), "qS outside [0, pi] would break the chord embedding"
xyz = np.stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)], axis=1)
dmat = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=2)
iu, ju = np.triu_indices(1590, k=1)
sky_hit = dmat[iu, ju] <= radius[iu] + radius[ju]
lo, hi = dl - 2 * s_dl, dl + 2 * s_dl
ii, jj = iu[sky_hit], ju[sky_hit]
dl_hit = (lo[ii] <= hi[jj]) & (lo[jj] <= hi[ii])
overlap = np.zeros(1590, dtype=bool)
overlap[ii[dl_hit]] = True
overlap[jj[dl_hit]] = True
res["census"] = {
    "sky_pairs": int(sky_hit.sum()),
    "sky_dl_pairs": int(dl_hit.sum()),
    "overlap_events": int(overlap.sum()),
    "expected": [1620, 279, 385],
}
assert res["census"]["sky_pairs"] == 1620, res["census"]
assert res["census"]["sky_dl_pairs"] == 279, res["census"]
assert res["census"]["overlap_events"] == 385, res["census"]
print(f"[{time.time() - t0:6.1f}s] census OK: 1620/279/385", flush=True)


# ------------------------------------------------------------------ matching (own code)
def match_nn(
    cov_cols: np.ndarray, ev: np.ndarray, ov_mask: np.ndarray, band: np.ndarray | None = None
) -> np.ndarray:
    """1-NN with replacement, control -> each overlap event, standardized euclidean.
    Returns positions (into ev) of matched controls, aligned with ev[ov_mask]."""
    x = cov_cols[ev]
    z = (x - x.mean(axis=0)) / x.std(axis=0, ddof=1)
    pos = np.arange(len(ev))
    ovp, ctp = pos[ov_mask], pos[~ov_mask]
    out = np.empty(len(ovp), dtype=int)
    if band is None:
        d2 = ((z[ovp][:, None, :] - z[ctp][None, :, :]) ** 2).sum(axis=2)
        out[:] = ctp[d2.argmin(axis=1)]
    else:
        bv = band[ev]
        for lev in np.unique(bv[ovp]):
            om = bv[ovp] == lev
            cm = bv[ctp] == lev
            assert cm.sum() > 0, f"no controls in band {lev}"
            sub_ct = ctp[cm]
            d2 = ((z[ovp[om]][:, None, :] - z[sub_ct][None, :, :]) ** 2).sum(axis=2)
            out[om] = sub_ct[d2.argmin(axis=1)]
    return out


# ------------------------------------------------------------------ load venue chords
venue_data: dict = {}
for venue, path in EL_PATHS.items():
    el = pd.read_csv(path)
    assert len(el) == 65108, (venue, len(el))
    a = el[np.isclose(el.h, 0.60)].set_index("event_idx").sort_index()
    b = el[np.isclose(el.h, 0.73)].set_index("event_idx").sort_index()
    ev = np.array(sorted(set(a.index) & set(b.index)))
    assert len(ev) == 1588, (venue, len(ev))
    venue_data[venue] = {"a": a.loc[ev], "b": b.loc[ev], "ev": ev}
    # w_G etc. event-independence
    for col in ["w_G", "alpha_G_phi", "r_Malm", "D_tilde_phi"]:
        assert a[col].nunique() == 1 and b[col].nunique() == 1, (venue, col)
print(f"[{time.time() - t0:6.1f}s] event_likelihoods loaded", flush=True)

ev = venue_data["iiib"]["ev"]
assert np.array_equal(ev, venue_data["joint_r1"]["ev"]), "venue event sets differ"
ov_mask = overlap[ev]
assert ov_mask.sum() == 385

# baseline m1 matching (radius, SNR) — identical for both venues
cov_m1 = np.stack([np.log10(radius), snr], axis=1)
nn_m1 = match_nn(cov_m1, ev, ov_mask)  # positions into ev
ovp = np.arange(len(ev))[ov_mask]
res["m1_balance"] = {
    "log10_radius_smd_after": smd(np.log10(radius)[ev[ovp]], np.log10(radius)[ev[nn_m1]]),
    "SNR_smd_after": smd(snr[ev[ovp]], snr[ev[nn_m1]]),
    "n_unique_controls": int(len(np.unique(nn_m1))),
}

# ------------------------------------------------------------------ D-1 verification
print(f"[{time.time() - t0:6.1f}s] === D-1 verification ===", flush=True)
d1_out: dict = {}
for venue in ["iiib", "joint_r1"]:
    va, vb = venue_data[venue]["a"], venue_data[venue]["b"]
    wt60 = float(va["w_G"].iloc[0])
    wt73 = float(vb["w_G"].iloc[0])
    rM60 = float(va["r_Malm"].iloc[0])
    rM73 = float(vb["r_Malm"].iloc[0])
    dln_wt = np.log(wt60) - np.log(wt73)
    dln_1mwt = np.log(1 - wt60) - np.log(1 - wt73)
    dln_rM = np.log(rM60) - np.log(rM73)

    vout: dict = {
        "wt_060": wt60,
        "wt_073": wt73,
        "dln_wt": float(dln_wt),
        "dln_one_minus_wt": float(dln_1mwt),
        "dln_r_Malm": float(dln_rM),
    }

    for ch in ["2d", "1d"]:
        if ch == "2d":
            lcat60 = va["L_cat_with_bh"].to_numpy()
            lcat73 = vb["L_cat_with_bh"].to_numpy()
            g60 = va["g_frac"].to_numpy()
            g73 = vb["g_frac"].to_numpy()
            lcomp60 = va["L_comp"].to_numpy()
            lcomp73 = vb["L_comp"].to_numpy()
            A60, A73 = wt60 * lcat60, wt73 * lcat73
            B60, B73 = (1 - wt60) * g60 * lcomp60, (1 - wt73) * g73 * lcomp73
            comb60 = va["combined_with_bh"].to_numpy()
            comb73 = vb["combined_with_bh"].to_numpy()
        else:
            lcat60 = va["L_cat_no_bh"].to_numpy()
            lcat73 = vb["L_cat_no_bh"].to_numpy()
            g60 = g73 = None
            lcomp60 = va["L_comp"].to_numpy()
            lcomp73 = vb["L_comp"].to_numpy()
            A60, A73 = (wt60 / rM60) * lcat60, (wt73 / rM73) * lcat73
            B60, B73 = (1 - wt60) * lcomp60, (1 - wt73) * lcomp73
            comb60 = va["combined_no_bh"].to_numpy()
            comb73 = vb["combined_no_bh"].to_numpy()

        # (a) composition identity vs CSV
        recon60, recon73 = A60 + B60, A73 + B73
        rel60 = np.abs(recon60 - comb60) / comb60
        rel73 = np.abs(recon73 - comb73) / comb73
        max_rel = float(max(rel60.max(), rel73.max()))

        # (b) h-stability of the zero sets of both legs
        z60, z73 = A60 == 0.0, A73 == 0.0
        assert np.array_equal(z60, z73), (venue, ch, "leg-A zero set not h-stable")
        nz = ~z60
        zb60, zb73 = B60 == 0.0, B73 == 0.0
        assert np.array_equal(zb60, zb73), (venue, ch, "leg-B zero set not h-stable")
        nzb = ~zb60

        # (c) LMDI decomposition (own implementation)
        def logmean(x, y):
            out = np.where(x == y, x, 0.0)
            m = (x != y) & (x > 0) & (y > 0)
            out = out.astype(float)
            out[m] = (x[m] - y[m]) / (np.log(x[m]) - np.log(y[m]))
            # one-sided zero (x>0,y==0 or vice versa) -> limit 0; must not occur for legs
            return out

        LF = logmean(recon60, recon73)
        SA = np.zeros(len(ev))
        SB = logmean(B60, B73) / LF
        SA[nz] = logmean(A60, A73)[nz] / LF[nz]

        dlnA = np.zeros(len(ev))
        dlnA[nz] = np.log(A60[nz]) - np.log(A73[nz])
        dlnB = np.zeros(len(ev))
        dlnB[nzb] = np.log(B60[nzb]) - np.log(B73[nzb])
        chord_recon = np.log(recon60) - np.log(recon73)
        chord_actual = np.log(comb60) - np.log(comb73)
        closure = np.abs(SA * dlnA + SB * dlnB - chord_recon)
        resid_recon = chord_actual - chord_recon

        T_legA = SA * dlnA
        T_legB = SB * dlnB
        dln_lcat = np.zeros(len(ev))
        dln_lcat[nz] = np.log(lcat60[nz]) - np.log(lcat73[nz])
        T_cat = SA * dln_lcat
        if ch == "2d":
            T_wG = SA * dln_wt
            dln_g = np.zeros(len(ev))
            dln_g[nzb] = np.log(g60[nzb]) - np.log(g73[nzb])
            T_gfrac = SB * dln_g
            T_w1m = SB * dln_1mwt
        else:
            T_wG = SA * (dln_wt - dln_rM)
            T_gfrac = None
            T_w1m = SB * dln_1mwt
        dln_lc = np.zeros(len(ev))
        dln_lc[nzb] = np.log(lcomp60[nzb]) - np.log(lcomp73[nzb])
        T_Lcomp = SB * dln_lc

        terms = {
            "total_chord": chord_actual,
            "T_legA": T_legA,
            "T_legB": T_legB,
            "T_cat": T_cat,
            "T_wG": T_wG,
            "T_Lcomp": T_Lcomp,
            "T_w1m": T_w1m,
            "resid_recon": resid_recon,
        }
        if T_gfrac is not None:
            terms["T_gfrac"] = T_gfrac

        # closure of the sub-splits
        sub_sum = T_wG + T_cat + T_Lcomp + T_w1m + (T_gfrac if T_gfrac is not None else 0.0)
        split_closure = np.abs(sub_sum - chord_recon)

        chout: dict = {
            "composition_max_rel_err": max_rel,
            "lmdi_closure_max_abs": float(closure.max()),
            "subsplit_closure_max_abs": float(split_closure.max()),
            "resid_recon_max_abs": float(np.abs(resid_recon).max()),
            "n_legA_zero_events": int(z60.sum()),
            "n_legA_zero_overlap": int(z60[ov_mask].sum()),
            "n_legA_zero_matched_controls": int(z60[nn_m1].sum()),
        }
        total_md = float((chord_actual[ovp] - chord_actual[nn_m1]).mean())
        chout["terms"] = {}
        for tname, tarr in terms.items():
            d = tarr[ovp] - tarr[nn_m1]
            st = paired_stats(d, ev[nn_m1])
            st["unmatched_mean_diff"] = float(tarr[ov_mask].mean() - tarr[~ov_mask].mean())
            if tname != "total_chord":
                st["fraction_of_total"] = float(d.mean() / total_md)
            st["n_pairs_both_exact_zero"] = int(((tarr[ovp] == 0) & (tarr[nn_m1] == 0)).sum())
            chout["terms"][tname] = st

        # (d) raw-column chord coverage probe (the L_cat trap)
        both_pos = nz  # L_cat>0 at both h
        pair_ok = both_pos[ovp] & both_pos[nn_m1]
        chout["raw_Lcat_chord"] = {
            "n_pairs_usable": int(pair_ok.sum()),
            "n_overlap_usable": int(both_pos[ovp].sum()),
            "mean_paired_diff_usable": float((dln_lcat[ovp] - dln_lcat[nn_m1])[pair_ok].mean())
            if pair_ok.sum() > 0
            else None,
        }
        # raw L_comp chord (full coverage)
        chout["raw_Lcomp_chord_matched"] = float((dln_lc[ovp] - dln_lc[nn_m1]).mean())
        vout[ch] = chout
    d1_out[venue] = vout
    print(f"[{time.time() - t0:6.1f}s] D-1 {venue} done", flush=True)

# cross-venue completion-leg identity
xv = {}
for col in ["L_comp", "B_num_wbh", "g_frac", "B_num"]:
    for hh, key in [("a", "h0.60"), ("b", "h0.73")]:
        x = venue_data["iiib"][hh][col].to_numpy()
        y = venue_data["joint_r1"][hh][col].to_numpy()
        denom = np.where(np.abs(y) > 0, np.abs(y), 1.0)
        xv[f"{col}_{key}"] = float(np.max(np.abs(x - y) / denom))
d1_out["cross_venue_completion_max_rel_diff"] = xv

# d_L imbalance probe over the m1 matched pairs
d1_out["dL_probe"] = {
    "d_L_smd_after_m1_matching": smd(dl[ev[ovp]], dl[ev[nn_m1]]),
    "d_L_matched_mean_paired_diff_Gpc": float((dl[ev[ovp]] - dl[ev[nn_m1]]).mean()),
    "d_L_overlap_mean": float(dl[ev[ovp]].mean()),
    "d_L_matched_control_mean": float(dl[ev[nn_m1]].mean()),
    "d_L_signflip_p": signflip_p(dl[ev[ovp]] - dl[ev[nn_m1]]),
    "log10_dL_smd_after_m1_matching": smd(np.log10(dl)[ev[ovp]], np.log10(dl)[ev[nn_m1]]),
}

# D-1 sensitivity re-match: (log10 radius, SNR, d_L raw)
cov_dl = np.stack([np.log10(radius), snr, dl], axis=1)
nn_dl = match_nn(cov_dl, ev, ov_mask)
sens: dict = {
    "covariates": ["log10_radius_chord", "SNR", "d_L_raw"],
    "balance_smd_after": {
        "log10_radius_chord": smd(np.log10(radius)[ev[ovp]], np.log10(radius)[ev[nn_dl]]),
        "SNR": smd(snr[ev[ovp]], snr[ev[nn_dl]]),
        "d_L": smd(dl[ev[ovp]], dl[ev[nn_dl]]),
    },
    "n_unique_controls": int(len(np.unique(nn_dl))),
}
for venue in ["iiib", "joint_r1"]:
    va, vb = venue_data[venue]["a"], venue_data[venue]["b"]
    ch2 = np.log(va["combined_with_bh"].to_numpy()) - np.log(vb["combined_with_bh"].to_numpy())
    ch1 = np.log(va["combined_no_bh"].to_numpy()) - np.log(vb["combined_no_bh"].to_numpy())
    d2v = ch2[ovp] - ch2[nn_dl]
    d1v = ch1[ovp] - ch1[nn_dl]
    sens[venue] = {
        "2d_total": float(d2v.mean()),
        "2d_signflip_p": signflip_p(d2v),
        "2d_cluster_p": cluster_signflip_p(d2v, ev[nn_dl]),
        "1d_total": float(d1v.mean()),
        "1d_signflip_p": signflip_p(d1v),
    }
d1_out["sensitivity_rematch_dL"] = sens
res["d1"] = d1_out
print(f"[{time.time() - t0:6.1f}s] D-1 block complete", flush=True)

# ------------------------------------------------------------------ D-2 covariates
print(f"[{time.time() - t0:6.1f}s] === D-2 verification ===", flush=True)

# galactic / ecliptic latitude (COORD-03: BarycentricTrueEcliptic J2000 -> Galactic)
import astropy.units as u  # noqa: E402
from astropy.coordinates import BarycentricTrueEcliptic, Galactic, SkyCoord  # noqa: E402

ecl_lat_deg = 90.0 - np.degrees(theta)
sc = SkyCoord(
    lon=np.degrees(phi) * u.deg,
    lat=ecl_lat_deg * u.deg,
    frame=BarycentricTrueEcliptic(equinox="J2000"),
)
gal_b_deg = sc.transform_to(Galactic()).b.deg
abs_sin_gal = np.abs(np.sin(np.radians(gal_b_deg)))
abs_sin_ecl = np.abs(np.cos(theta))
BAND_EDGES = [15.0, 45.0]
gal_band = np.digitize(np.abs(gal_b_deg), BAND_EDGES)
ecl_band = np.digitize(np.abs(ecl_lat_deg), BAND_EDGES)


def venue_ball_covs(venue: str) -> dict[str, np.ndarray]:
    cache = f"{SCRATCH}/adj_ballcovs_{venue}.npz"
    if os.path.exists(cache):
        z = np.load(cache)
        return {k: z[k] for k in z.files}
    d = json.load(open(BALL_PATHS[venue]))
    gl, ag = d["galaxy_likelihoods"], d["additional_galaxies_without_bh_mass"]
    assert set(int(k) for k in gl) == set(ev.tolist()), venue
    # pruned+reset catalogue frame (M-4 convention, production helpers)
    names = _reduced_catalog_column_names()
    cat = pd.read_csv(CATS[venue], names=names, usecols=[3, 4, 5, 6])
    z_c = cat["REDSHIFT"].to_numpy(np.float64)
    sz_c = cat["REDSHIFT_MEASUREMENT_ERROR"].to_numpy(np.float64)
    ms = cat["STELLAR_MASS"].to_numpy(np.float64)
    mse = cat["STELLAR_MASS_ABSOULTE_ERROR"].to_numpy(np.float64)
    del cat
    mbh, mbh_err = _empiric_stellar_mass_to_BH_mass_relation(ms, mse)
    keep = ~np.isnan(mbh)
    z_c, sz_c, mbh, mbh_err = z_c[keep], sz_c[keep], mbh[keep], mbh_err[keep]
    mask = _mass_redshift_prune_mask(
        pd.Series(mbh),
        pd.Series(mbh_err),
        pd.Series(z_c),
        pd.Series(sz_c),
        M_MIN,
        M_MAX,
        Z_MAX,
    ).to_numpy()
    z_c, mbh = z_c[mask], mbh[mask]
    n_pruned = len(z_c)
    w_all = R_eff_per_mbh(mbh) / (1.0 + z_c)
    n2 = np.zeros(1590)
    n1 = np.zeros(1590)
    w2 = np.zeros(1590)
    w1 = np.zeros(1590)
    for k in gl:
        e = int(k)
        idx2 = np.array([r[0] for r in gl[k]], dtype=np.int64)
        idx_ag = np.array([r[0] for r in ag[k]], dtype=np.int64)
        idx1 = np.union1d(idx2, idx_ag)
        n2[e] = len(idx2)
        n1[e] = len(idx2) + len(idx_ag)
        if len(idx2):
            assert idx2.max() < n_pruned, (venue, e)
            w2[e] = w_all[idx2].sum()
        if len(idx1):
            assert idx1.max() < n_pruned, (venue, e)
            w1[e] = w_all[idx1].sum()
    out = {"n2": n2, "n1": n1, "w2": w2, "w1": w1, "n_pruned": np.array([n_pruned], dtype=np.int64)}
    np.savez(cache, **out)
    return out


COV_NAMES = [
    "log10_radius_chord",
    "SNR",
    "log10_n_ball_2d",
    "log10_n_ball_1d",
    "log10_W_pop_2d",
    "log10_W_pop_1d",
    "log10_dL",
    "log10_rel_dL_err",
    "abs_sin_gal_lat",
    "abs_sin_ecl_lat",
]

d2_out: dict = {"band_edges_deg": BAND_EDGES}
for venue in ["iiib", "joint_r1"]:
    bc = venue_ball_covs(venue)
    print(
        f"[{time.time() - t0:6.1f}s] {venue}: ball covs ready "
        f"(pruned rows {int(bc['n_pruned'][0])})",
        flush=True,
    )
    cov_all = np.stack(
        [
            np.log10(radius),
            snr,
            np.log10(1 + bc["n2"]),
            np.log10(1 + bc["n1"]),
            np.log10(1 + bc["w2"]),
            np.log10(1 + bc["w1"]),
            np.log10(dl),
            np.log10(s_dl / dl),
            abs_sin_gal,
            abs_sin_ecl,
        ],
        axis=1,
    )

    va, vb = venue_data[venue]["a"], venue_data[venue]["b"]
    ch2 = np.log(va["combined_with_bh"].to_numpy()) - np.log(vb["combined_with_bh"].to_numpy())
    ch1 = np.log(va["combined_no_bh"].to_numpy()) - np.log(vb["combined_no_bh"].to_numpy())

    def run_rung(cols: list[int], band: np.ndarray | None) -> dict:
        nn = match_nn(cov_all[:, cols], ev, ov_mask, band=band)  # noqa: B023
        bal = {}
        for k, nm in enumerate(COV_NAMES):
            bal[nm] = {
                "smd_before": round(
                    smd(cov_all[ev[ovp], k], cov_all[ev[np.arange(len(ev))[~ov_mask]], k]),  # noqa: B023
                    4,
                ),
                "smd_after": round(smd(cov_all[ev[ovp], k], cov_all[ev[nn], k]), 4),  # noqa: B023
                "matched_on": k in cols,
            }
        d2v = ch2[ovp] - ch2[nn]  # noqa: B023
        d1v = ch1[ovp] - ch1[nn]  # noqa: B023
        return {
            "covariates": [COV_NAMES[k] for k in cols],
            "n_unique_controls": int(len(np.unique(nn))),
            "balance": bal,
            "max_abs_smd_after_matched_on": float(
                max(abs(bal[COV_NAMES[k]]["smd_after"]) for k in cols)
            ),
            "2d": paired_stats(d2v, ev[nn]),
            "1d": paired_stats(d1v, ev[nn]),
        }

    rungs = {
        "m1": run_rung([0, 1], None),
        "m2": run_rung([0, 1, 2], None),
        "m3": run_rung([0, 1, 2], gal_band),
        "m3e": run_rung([0, 1, 2], ecl_band),
        "m4": run_rung(list(range(10)), gal_band),
        # adversarial extra rung: geometry-only, no density covariates
        "m_geo": run_rung([0, 1, 6, 7, 8, 9], None),
        # adversarial extra rung: density-only addition via W_pop instead of counts
        "m2w": run_rung([0, 1, 5], None),
    }
    m1_eff = rungs["m1"]["2d"]["mean_paired_diff"]
    trajectory = {
        r: {
            "eff_2d": rungs[r]["2d"]["mean_paired_diff"],
            "p_2d": rungs[r]["2d"]["signflip_p"],
            "clp_2d": rungs[r]["2d"]["cluster_signflip_p"],
            "ratio_to_m1": rungs[r]["2d"]["mean_paired_diff"] / m1_eff,
        }
        for r in rungs
    }

    # single-covariate panel (2D): baseline (0,1) + one covariate
    panel = {}
    for k in range(2, 10):
        nn = match_nn(cov_all[:, [0, 1, k]], ev, ov_mask)
        d2v = ch2[ovp] - ch2[nn]
        panel[COV_NAMES[k]] = {
            "eff_2d": float(d2v.mean()),
            "signflip_p": signflip_p(d2v),
            "cluster_signflip_p": cluster_signflip_p(d2v, ev[nn]),
            "rel_change_vs_m1": float(d2v.mean() / m1_eff - 1.0),
            "smd_after_added_cov": smd(cov_all[ev[ovp], k], cov_all[ev[nn], k]),
        }

    # over-matching diagnostic: covariate/outcome correlation among controls
    from scipy.stats import spearmanr

    ctp_all = np.arange(len(ev))[~ov_mask]
    overmatch = {}
    for k, nm in enumerate(COV_NAMES):
        rho, _ = spearmanr(cov_all[ev[ctp_all], k], ch2[ctp_all])
        overmatch[nm] = round(float(rho), 4)

    d2_out[venue] = {
        "pruned_catalogue_rows": int(bc["n_pruned"][0]),
        "rungs": rungs,
        "trajectory_2d": trajectory,
        "single_covariate_panel_2d": panel,
        "spearman_cov_vs_2d_chord_controls": overmatch,
    }
    print(f"[{time.time() - t0:6.1f}s] D-2 {venue} rungs+panel done", flush=True)

res["d2"] = d2_out

# ------------------------------------------------------------------ comparison vs reported
d1_rep = json.load(open(f"{HERE}/d1_results.json"))
d2_rep = json.load(open(f"{HERE}/d2_results.json"))
m2_rep = json.load(
    open(
        f"{ROOT}/results/campaign51_20260728/realistic_20260729/"
        "crossterm_instrument/m2_results.json"
    )
)

cmp: dict = {"m2_baseline": {}, "d1": {}, "d2": {}}
for venue in ["iiib", "joint_r1"]:
    mine_2d = res["d2"][venue]["rungs"]["m1"]["2d"]["mean_paired_diff"]
    m2_val = m2_rep["venues"][venue]["channels"]["2d"]["matched"]["mean_paired_diff"]
    cmp["m2_baseline"][venue] = {
        "mine": mine_2d,
        "m2_committed": m2_val,
        "abs_diff": abs(mine_2d - m2_val),
    }
    # D-1 top component
    mine_t = res["d1"][venue]["2d"]["terms"]["T_Lcomp"]
    rep_t = d1_rep["venues"][venue]["channels"]["2d"]["decomposition_terms"]["T_Lcomp"]
    cmp["d1"][venue] = {
        "T_Lcomp_mine": mine_t["mean_paired_diff"],
        "T_Lcomp_reported": rep_t["matched"]["mean_paired_diff"],
        "abs_diff": abs(mine_t["mean_paired_diff"] - rep_t["matched"]["mean_paired_diff"]),
        "fraction_mine": mine_t["fraction_of_total"],
        "fraction_reported": rep_t["fraction_of_total_matched_diff"],
    }
    cmp["d2"][venue] = {}
    for r in ["m1", "m2", "m3", "m3e", "m4"]:
        mine = res["d2"][venue]["rungs"][r]["2d"]["mean_paired_diff"]
        rep = d2_rep["rungs"][venue][r]["channels"]["2d"]["mean_paired_diff"]
        cmp["d2"][venue][r] = {"mine": mine, "reported": rep, "abs_diff": abs(mine - rep)}
res["comparison_vs_reported"] = cmp
res["runtime_s"] = round(time.time() - t0, 1)

with open(OUT, "w") as f:
    json.dump(res, f, indent=1)
print(f"[{time.time() - t0:6.1f}s] wrote {OUT}", flush=True)
print(json.dumps(cmp, indent=1))
