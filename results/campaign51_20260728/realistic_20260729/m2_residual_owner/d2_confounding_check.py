"""D-2: extended-covariate confounding check on the M-2 matched 2D overlap residual.

Question (ch13 Part A, read D-2): does the matched +0.02225 (iiib) / +0.02070
(joint_r1) nat/event 2D overlap residual survive matching on covariates BEYOND
the M-2 pair (log10 ball-radius chord, SNR)?  Sky-overlap correlates with sky
position and local catalogue density by construction — if the residual is a
density/selection artifact, richer matching kills it; if it is physical
shared-structure, it survives.

Design (stage 0-1 of docs/RESEARCH_CYCLE.md — FREE READS ONLY):

  Reproduction anchor (hard asserts): a VERBATIM port of
  crossterm_instrument/m2_overlap_stratified.py (same census recipe, same
  covariate standardization, same brute-force 1-NN argmin, same RNG seed
  20260805 consumed in the same order) must reproduce m2_results.json exactly:
  mean paired diffs, SMDs, n_unique_controls to 1e-12 rel, permutation /
  sign-flip p-values bitwise.  Headline asserts: iiib 2d matched residual
  0.022252643015992925, joint_r1 2d 0.020697491999731973.

  Extended covariates (all free reads):
    from prepared_cramer_rao_bounds.csv (md5-identical across venues):
      log10_radius_chord, SNR                      [M-2 originals]
      log10_dL          = log10(luminosity_distance)
      log10_rel_dL_err  = log10(sigma_dL / dL)
      abs_sin_ecl_lat   = |cos(qS)|                (qS = ecliptic colatitude)
      abs_sin_gal_lat   = |sin(b_gal)|             (astropy BarycentricTrueEcliptic
                                                    (equinox J2000) -> Galactic,
                                                    the COORD-03 frame convention)
      gal_band / ecl_band = |lat| bands {<15, 15-45, >=45} degrees
    from run_20260804_frozeng/<venue>/posteriors_with_bh_mass/h_0_73.json
    (venue-specific ball emits; M-4 recipe, h-independence verified in M-4/V2):
      log10_n_ball_2d = log10(1 + len(galaxy_likelihoods[ev]))
      log10_n_ball_1d = log10(1 + n_2d + len(additional_galaxies_without_bh_mass[ev]))
      log10_W_pop_2d  = log10(1 + sum_{g in ball2d} w_g)
      log10_W_pop_1d  = log10(1 + sum_{g in ball1d} w_g)
      with w_g = R_eff_per_mbh(M_g)/(1+z_g), (z_g, M_g) dereferenced at
      catalog_index in the bit-faithful pruned+reset catalogue frame — the
      EXACT M-4 / outside_c4_2d_wpop.py convention (load_pruned_zm verbatim).
      Both ball lists live in the same reduced_galaxy_catalog frame
      (handler.get_possible_hosts_from_ball_tree: the with/without split is a
      mass-window filter on one frame), so the deref is valid for both.

  Matching rungs (same machinery: standardize over the 1588 evaluated events,
  brute-force 1-NN with replacement control->overlap; band rungs restrict the
  candidate pool to same-band controls before the argmin):
    m1  : (log10_r, SNR)                                  [baseline, asserted]
    m2  : + log10_n_ball_2d
    m3  : m2 + EXACT match on gal_band  (ZoA / catalogue-density axis)
    m3e : m2 + EXACT match on ecl_band  (variant, reported not primary)
    m4  : all 10 continuous covariates + EXACT gal_band

  Tests per rung x venue x channel (1d, 2d; primary = 2d): vectorized
  sign-flip p (20000) and cluster-robust sign-flip p (20000, pairs sharing a
  control flip together — the survival test of the established record), from
  ONE fresh RNG seeded 20260807 consumed in documented fixed order.  The m1
  rung re-enters this loop so the trajectory is internally consistent (same
  test implementation at every rung); its fresh-RNG cluster p must land near
  the recorded 0.0050/0.0042 (consistency check, not an assert).

  Single-covariate augmentation panel (2D only): baseline + each extended
  covariate alone -> which covariates move the residual ("which covariates
  change the residual", reported symmetrically).

  Trajectory call (pre-stated, primary channel 2d, per venue):
    ratio r = E(m4)/E(m1); sig(m4) = signflip p < 0.0455 AND cluster p < 0.0455
    SURVIVES     iff sign stable across m1..m4 AND sig(m4) AND r >= 0.5
                     AND matched-on |SMD| <= 0.10 at m4
    KILLED       iff NOT sig(m4) AND (r < 0.5 or sign flip at m4)
    UNDETERMINED otherwise (incl. balance failure deciding the call)
    Overall: SURVIVES/KILLED iff both venues agree; else UNDETERMINED.
  Over-matching trap (stated up front, applied symmetrically): the ball
  density covariates (n_2d, n_1d, W_2d, W_1d) are themselves mechanism
  candidates — the LIVE CLUE says the owner must act THROUGH the composition
  weights or outside the annihilated catalogue path, and local catalogue
  density is exactly the kind of variable that could BE the physics.  A
  collapse under density matching is therefore reported as
  'density-absorbed', NOT auto-interpreted as artifact; only collapse under
  pure sky-position/geometry covariates (bands, latitudes, d_L) is a clean
  artifact signature.

Read-only on production artifacts.  Writes d2_results.json next to this file.
Run: cd /home/jasper/Repositories/MasterThesisCode && uv run python \
    results/campaign51_20260728/realistic_20260729/m2_residual_owner/d2_confounding_check.py
"""

import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd

REPO = "/home/jasper/Repositories/MasterThesisCode"
sys.path.insert(0, REPO)
os.chdir(REPO)

import astropy.units as u  # noqa: E402
from astropy.coordinates import BarycentricTrueEcliptic, Galactic, SkyCoord  # noqa: E402

from master_thesis_code.emri_rate import R_eff_per_mbh  # noqa: E402
from master_thesis_code.galaxy_catalogue.handler import (  # noqa: E402
    _empiric_stellar_mass_to_BH_mass_relation,
    _mass_redshift_prune_mask,
    _reduced_catalog_column_names,
)

HERE = f"{REPO}/results/campaign51_20260728/realistic_20260729/m2_residual_owner"
CI = f"{REPO}/results/campaign51_20260728/realistic_20260729/crossterm_instrument"
OUT = f"{HERE}/d2_results.json"
M2_RESULTS = f"{CI}/m2_results.json"
CRB = f"{REPO}/results/run_20260804_postfix/joint_r1/diagnostics/prepared_cramer_rao_bounds.csv"
CRB_IIIB = f"{REPO}/results/run_20260804_postfix/iiib/diagnostics/prepared_cramer_rao_bounds.csv"
VENUES = {
    "iiib": f"{REPO}/results/run_20260804_postfix/iiib/diagnostics/event_likelihoods.csv",
    "joint_r1": f"{REPO}/results/run_20260804_postfix/joint_r1/diagnostics/event_likelihoods.csv",
}
BALLS = {
    v: f"{REPO}/results/run_20260804_frozeng/{v}/posteriors_with_bh_mass/h_0_73.json"
    for v in VENUES
}
STAGED = f"{REPO}/results/campaign51_20260728/realistic_20260729/realizations_staged"
CATS = {
    "joint_r1": f"{STAGED}/observed_catalogue_seed900001.csv",
    "iiib": f"{STAGED}/cluster_parent_reduced_galaxy_catalogue.csv",
}
CHANNELS = {"1d": "combined_no_bh", "2d": "combined_with_bh"}
N_PERM = 20000
ALPHA = 0.0455
SMD_FAIL = 0.10
M_SOURCE_FRAME_MIN, M_SOURCE_FRAME_MAX, Z_MAX = 1e4, 1e7, 1.5
BAND_EDGES_DEG = [15.0, 45.0]

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
RUNGS: dict[str, dict] = {
    "m1": {"cont": [0, 1], "band": None},
    "m2": {"cont": [0, 1, 2], "band": None},
    "m3": {"cont": [0, 1, 2], "band": "gal"},
    "m3e": {"cont": [0, 1, 2], "band": "ecl"},
    "m4": {"cont": list(range(10)), "band": "gal"},
}
PANEL_COVS = [2, 3, 4, 5, 6, 7, 8, 9]  # single-covariate augmentation on top of [0, 1]

t0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


def md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


# ===================================================================== census
# VERBATIM port of m2_overlap_stratified.py (asserted numbers included).
log("loading CRB + census (M-2 verbatim)")
df = pd.read_csv(CRB)
n = len(df)
assert n == 1590, n
theta = df["qS"].to_numpy()
phi = df["phiS"].to_numpy()
s_phi2 = df["delta_phiS_delta_phiS"].to_numpy()
s_theta2 = df["delta_qS_delta_qS"].to_numpy()
cov = df["delta_phiS_delta_qS"].to_numpy()
dl = df["luminosity_distance"].to_numpy()
s_dl = np.sqrt(df["delta_luminosity_distance_delta_luminosity_distance"].to_numpy())
snr = df["SNR"].to_numpy()

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
n_pairs = int(win.sum())
overlap = np.zeros(n, dtype=bool)
overlap[ii[win]] = True
overlap[jj[win]] = True
n_overlap_1590 = int(overlap.sum())
assert n_pairs == 279, n_pairs
assert n_overlap_1590 == 385, n_overlap_1590
assert int(sky.sum()) == 1620

# ================================================================== helpers
# smd / perm_p_mean_diff / signflip_p are VERBATIM M-2 (loop-based, RNG order
# preserved) — used ONLY inside the reproduction anchor with RNG_M2.
RNG_M2 = np.random.default_rng(20260805)


def smd(a: np.ndarray, b: np.ndarray) -> float:
    sp = np.sqrt(0.5 * (a.var(ddof=1) + b.var(ddof=1)))
    return float((a.mean() - b.mean()) / sp) if sp > 0 else 0.0


def perm_p_mean_diff_m2(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    obs = x.mean() - y.mean()
    pooled = np.concatenate([x, y])
    nx = len(x)
    cnt = 0
    for _ in range(N_PERM):
        p = RNG_M2.permutation(pooled)
        if abs(p[:nx].mean() - p[nx:].mean()) >= abs(obs):
            cnt += 1
    return float(obs), (cnt + 1) / (N_PERM + 1)


def signflip_p_m2(diffs: np.ndarray) -> tuple[float, float]:
    obs = diffs.mean()
    m = len(diffs)
    cnt = 0
    for _ in range(N_PERM):
        s = RNG_M2.choice([-1.0, 1.0], size=m)
        if abs((s * diffs).mean()) >= abs(obs):
            cnt += 1
    return float(obs), (cnt + 1) / (N_PERM + 1)


# Fresh-RNG vectorized tests for the rung table (one stream, fixed order).
RNG_D2 = np.random.default_rng(20260807)


def signflip_p_vec(diffs: np.ndarray) -> float:
    obs = abs(diffs.mean())
    m = len(diffs)
    signs = RNG_D2.integers(0, 2, size=(N_PERM, m)) * 2 - 1
    stats = np.abs(signs @ diffs) / m
    return float((np.sum(stats >= obs) + 1) / (N_PERM + 1))


def cluster_signflip_p_vec(diffs: np.ndarray, cluster_ids: np.ndarray) -> float:
    obs = abs(diffs.mean())
    uniq, inv = np.unique(cluster_ids, return_inverse=True)
    k = len(uniq)
    csum = np.zeros(k)
    np.add.at(csum, inv, diffs)
    m = len(diffs)
    signs = RNG_D2.integers(0, 2, size=(N_PERM, k)) * 2 - 1
    stats = np.abs(signs @ csum) / m
    return float((np.sum(stats >= obs) + 1) / (N_PERM + 1))


# ====================================================== chords per venue
log("loading event_likelihoods + chords")
chords: dict[str, dict[str, pd.Series]] = {}
ev_sets: dict[str, np.ndarray] = {}
for venue, path in VENUES.items():
    el = pd.read_csv(path)
    assert len(el) == 65108, (venue, len(el))
    for h0 in (0.60, 0.73):
        assert el.loc[np.isclose(el.h, h0), "w_G"].nunique() == 1, (venue, h0)
    a = el[np.isclose(el.h, 0.60)].set_index("event_idx")
    b = el[np.isclose(el.h, 0.73)].set_index("event_idx")
    ev = np.array(sorted(set(a.index) & set(b.index)))
    assert len(ev) == 1588, (venue, len(ev))
    ev_sets[venue] = ev
    chords[venue] = {}
    for ch, col in CHANNELS.items():
        la, lb = a[col], b[col]
        assert (la.loc[ev] > 0).all() and (lb.loc[ev] > 0).all(), (venue, ch)
        chords[venue][ch] = pd.Series(
            np.log(la.loc[ev].to_numpy()) - np.log(lb.loc[ev].to_numpy()), index=ev
        )

# =============================================== reproduction anchor (m1 exact)
log("reproduction anchor: M-2 verbatim loop, seed 20260805")
with open(M2_RESULTS) as f:
    m2_ref = json.load(f)

log10_r = np.log10(r)
cov_mat_m2 = np.stack([log10_r, snr], axis=1)
repro: dict = {}
for venue in VENUES:
    ev = ev_sets[venue]
    ov_mask = overlap[ev]
    ct_mask = ~ov_mask
    ov_ev, ct_ev = ev[ov_mask], ev[ct_mask]
    z = (cov_mat_m2[ev] - cov_mat_m2[ev].mean(axis=0)) / cov_mat_m2[ev].std(axis=0, ddof=1)
    z_ov, z_ct = z[ov_mask], z[ct_mask]
    dist2 = ((z_ov[:, None, :] - z_ct[None, :, :]) ** 2).sum(axis=2)
    nn = dist2.argmin(axis=1)
    matched_ct_ev = ct_ev[nn]

    ref_bal = m2_ref["venues"][venue]["balance"]
    for k, name in enumerate(["log10_radius_chord", "SNR"]):
        sb = smd(cov_mat_m2[ov_ev, k], cov_mat_m2[ct_ev, k])
        sa = smd(cov_mat_m2[ov_ev, k], cov_mat_m2[matched_ct_ev, k])
        assert abs(sb - ref_bal[name]["smd_before"]) < 1e-12, (venue, name, sb)
        assert abs(sa - ref_bal[name]["smd_after"]) < 1e-12, (venue, name, sa)
    assert int(len(np.unique(nn))) == ref_bal["n_unique_controls_used"], venue

    repro[venue] = {}
    for ch in CHANNELS:
        chord = chords[venue][ch]
        x_ov = chord.loc[ov_ev].to_numpy()
        x_ct = chord.loc[ct_ev].to_numpy()
        x_mct = chord.loc[matched_ct_ev].to_numpy()
        um_diff, um_p = perm_p_mean_diff_m2(x_ov, x_ct)
        pd_diffs = x_ov - x_mct
        m_diff, m_p = signflip_p_m2(pd_diffs)
        ref_ch = m2_ref["venues"][venue]["channels"][ch]
        assert abs(um_diff - ref_ch["unmatched"]["mean_diff"]) < 1e-12, (venue, ch)
        assert um_p == ref_ch["unmatched"]["perm_p"], (venue, ch, um_p)
        assert abs(m_diff - ref_ch["matched"]["mean_paired_diff"]) < 1e-12, (venue, ch, m_diff)
        assert m_p == ref_ch["matched"]["signflip_p"], (venue, ch, m_p)
        repro[venue][ch] = {
            "matched_mean_paired_diff": m_diff,
            "matched_signflip_p": m_p,
            "unmatched_mean_diff": um_diff,
            "unmatched_perm_p": um_p,
        }

# task-mandated headline hard asserts
assert abs(repro["iiib"]["2d"]["matched_mean_paired_diff"] - 0.022252643015992925) < 1e-15
assert abs(repro["joint_r1"]["2d"]["matched_mean_paired_diff"] - 0.020697491999731973) < 1e-15
log("reproduction anchor PASS (all hard asserts)")

# ====================================================== extended covariates
log("building extended covariates: sky bands (astropy)")
ecl_lat_rad = np.pi / 2.0 - theta  # qS = ecliptic colatitude
sc = SkyCoord(
    lon=phi * u.rad, lat=ecl_lat_rad * u.rad, frame=BarycentricTrueEcliptic(equinox="J2000")
)
gal_b_deg = sc.transform_to(Galactic()).b.deg
abs_sin_gal_lat = np.abs(np.sin(np.deg2rad(gal_b_deg)))
abs_sin_ecl_lat = np.abs(np.cos(theta))
gal_band = np.digitize(np.abs(gal_b_deg), BAND_EDGES_DEG)
ecl_band = np.digitize(np.abs(np.rad2deg(ecl_lat_rad)), BAND_EDGES_DEG)
log10_dL = np.log10(dl)
log10_rel_dL_err = np.log10(s_dl / dl)


def load_pruned_zm(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Bit-faithful pruned+reset (z, M_bh) columns (M-4 / outside_c4_2d_wpop verbatim)."""
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


ball_cov: dict[str, np.ndarray] = {}  # venue -> (1590, 4): n2d, n1d, W2d, W1d (raw)
ball_meta: dict[str, dict] = {}
for venue in VENUES:
    log(f"building ball covariates for {venue} (JSON + pruned catalogue)")
    with open(BALLS[venue]) as f:
        ball = json.load(f)
    gl = ball["galaxy_likelihoods"]
    ag = ball["additional_galaxies_without_bh_mass"]
    zz, mm = load_pruned_zm(CATS[venue])
    n_cat = len(zz)
    w_all = R_eff_per_mbh(mm) / (1.0 + zz)
    arr = np.full((n, 4), np.nan)
    n_missing = 0
    for e in range(n):
        k = str(e)
        if k not in gl:
            n_missing += 1
            continue
        ids2 = np.array([row[0] for row in gl[k]], dtype=np.int64)
        ids_extra = np.array([row[0] for row in ag.get(k, [])], dtype=np.int64)
        ids1 = np.concatenate([ids2, ids_extra])
        if len(ids1):
            assert ids1.max() < n_cat and ids1.min() >= 0, (venue, e)
        w2 = float(w_all[ids2].sum()) if len(ids2) else 0.0
        w1 = float(w_all[ids1].sum()) if len(ids1) else 0.0
        arr[e] = [len(ids2), len(ids1), w2, w1]
    ev = ev_sets[venue]
    assert not np.isnan(arr[ev]).any(), (venue, "ball covariates missing for evaluated events")
    ball_cov[venue] = arr
    ball_meta[venue] = {
        "ball_json_md5": md5(BALLS[venue]),
        "n_events_in_gl": len([k for k in gl if k.isdigit()]),
        "n_events_missing_of_1590": n_missing,
        "pruned_catalogue_rows": n_cat,
        "n_ball_2d": {
            "min": float(np.nanmin(arr[:, 0])),
            "median": float(np.nanmedian(arr[:, 0])),
            "max": float(np.nanmax(arr[:, 0])),
        },
        "n_ball_1d_median": float(np.nanmedian(arr[:, 1])),
        "W_pop_2d_median": float(np.nanmedian(arr[:, 2])),
    }
    del ball, gl, ag, zz, mm, w_all


def cov_matrix(venue: str) -> np.ndarray:
    """(1590, 10) covariate matrix in COV_NAMES order (NaN outside evaluated events)."""
    b = ball_cov[venue]
    return np.stack(
        [
            log10_r,
            snr,
            np.log10(1.0 + b[:, 0]),
            np.log10(1.0 + b[:, 1]),
            np.log10(1.0 + b[:, 2]),
            np.log10(1.0 + b[:, 3]),
            log10_dL,
            log10_rel_dL_err,
            abs_sin_gal_lat,
            abs_sin_ecl_lat,
        ],
        axis=1,
    )


BANDS = {"gal": gal_band, "ecl": ecl_band}


def match(
    venue: str, cont_idx: list[int], band_key: str | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """1-NN with replacement on standardized covariates, optional exact band.

    Returns (ov_ev, matched_ct_ev, nn_cluster_ids, balance_dict).
    """
    ev = ev_sets[venue]
    C_full = cov_matrix(venue)
    ov_mask = overlap[ev]
    ov_ev, ct_ev = ev[ov_mask], ev[~ov_mask]
    C = C_full[ev][:, cont_idx]
    z = (C - C.mean(axis=0)) / C.std(axis=0, ddof=1)
    z_ov, z_ct = z[ov_mask], z[~ov_mask]
    n_fallback = 0
    if band_key is None:
        dist2 = ((z_ov[:, None, :] - z_ct[None, :, :]) ** 2).sum(axis=2)
        nn = dist2.argmin(axis=1)
    else:
        band = BANDS[band_key]
        b_ov, b_ct = band[ov_ev], band[ct_ev]
        nn = np.empty(len(ov_ev), dtype=np.int64)
        for i in range(len(ov_ev)):
            pool = np.where(b_ct == b_ov[i])[0]
            if len(pool) == 0:  # fallback: full pool (counted, not expected)
                pool = np.arange(len(ct_ev))
                n_fallback += 1
            d2 = ((z_ct[pool] - z_ov[i]) ** 2).sum(axis=1)
            nn[i] = pool[d2.argmin()]
    matched_ct_ev = ct_ev[nn]

    bal: dict = {"per_covariate": {}}
    max_abs_matched_on = 0.0
    max_abs_full = 0.0
    for k, name in enumerate(COV_NAMES):
        sb = smd(C_full[ov_ev, k], C_full[ct_ev, k])
        sa = smd(C_full[ov_ev, k], C_full[matched_ct_ev, k])
        bal["per_covariate"][name] = {
            "smd_before": round(sb, 4),
            "smd_after": round(sa, 4),
            "matched_on": k in cont_idx,
        }
        max_abs_full = max(max_abs_full, abs(sa))
        if k in cont_idx:
            max_abs_matched_on = max(max_abs_matched_on, abs(sa))
    if band_key is not None:
        bal["band_exact_fraction"] = float(
            np.mean(BANDS[band_key][ov_ev] == BANDS[band_key][matched_ct_ev])
        )
        bal["band_key"] = band_key
    bal["n_unique_controls_used"] = int(len(np.unique(nn)))
    bal["n_band_fallback"] = n_fallback
    bal["max_abs_smd_matched_on"] = round(max_abs_matched_on, 4)
    bal["max_abs_smd_full_set"] = round(max_abs_full, 4)
    bal["balance_ok_matched_on"] = bool(max_abs_matched_on <= SMD_FAIL)
    return ov_ev, matched_ct_ev, nn, bal


# ============================================================ rung table
# RNG_D2 consumption order (documented, deterministic):
#   for venue in (iiib, joint_r1): for rung in (m1, m2, m3, m3e, m4):
#     for ch in (1d, 2d): signflip_p_vec, cluster_signflip_p_vec
# then the single-covariate panel:
#   for venue: for cov in PANEL_COVS: (2d only) signflip_p_vec, cluster_signflip_p_vec
log("rung table (fresh RNG 20260807)")
rung_results: dict = {}
for venue in VENUES:
    rung_results[venue] = {}
    for rung, spec in RUNGS.items():
        ov_ev, mct_ev, nn, bal = match(venue, spec["cont"], spec["band"])
        entry: dict = {
            "covariates": [COV_NAMES[i] for i in spec["cont"]],
            "band_exact": spec["band"],
            "balance": bal,
            "channels": {},
        }
        for ch in CHANNELS:
            chord = chords[venue][ch]
            pd_diffs = chord.loc[ov_ev].to_numpy() - chord.loc[mct_ev].to_numpy()
            eff = float(pd_diffs.mean())
            p_sf = signflip_p_vec(pd_diffs)
            p_cl = cluster_signflip_p_vec(pd_diffs, mct_ev)
            entry["channels"][ch] = {
                "mean_paired_diff": eff,
                "median_paired_diff": float(np.median(pd_diffs)),
                "paired_diff_std": float(pd_diffs.std(ddof=1)),
                "se": float(pd_diffs.std(ddof=1) / np.sqrt(len(pd_diffs))),
                "n_pairs": int(len(pd_diffs)),
                "signflip_p": p_sf,
                "cluster_signflip_p": p_cl,
            }
        rung_results[venue][rung] = entry
        log(
            f"  {venue}/{rung}: 2d eff={entry['channels']['2d']['mean_paired_diff']:+.5f} "
            f"p={entry['channels']['2d']['signflip_p']:.2e} "
            f"clp={entry['channels']['2d']['cluster_signflip_p']:.2e} "
            f"maxSMD(on)={bal['max_abs_smd_matched_on']}"
        )

# m1 fresh-RNG consistency check vs the recorded cluster-robust values (no assert)
m1_consistency = {
    "recorded_cluster_p": {"iiib/2d": 0.004999750012499375, "joint_r1/2d": 0.004249787510624469},
    "fresh_rng_cluster_p": {
        f"{v}/2d": rung_results[v]["m1"]["channels"]["2d"]["cluster_signflip_p"] for v in VENUES
    },
}

# ================================================= single-covariate panel (2d)
log("single-covariate augmentation panel")
panel: dict = {}
for venue in VENUES:
    base_eff = rung_results[venue]["m1"]["channels"]["2d"]["mean_paired_diff"]
    panel[venue] = {"baseline_m1_effect_2d": base_eff, "augmented": {}}
    for k in PANEL_COVS:
        ov_ev, mct_ev, nn, bal = match(venue, [0, 1, k], None)
        chord = chords[venue]["2d"]
        pd_diffs = chord.loc[ov_ev].to_numpy() - chord.loc[mct_ev].to_numpy()
        eff = float(pd_diffs.mean())
        p_sf = signflip_p_vec(pd_diffs)
        p_cl = cluster_signflip_p_vec(pd_diffs, mct_ev)
        panel[venue]["augmented"][COV_NAMES[k]] = {
            "mean_paired_diff": eff,
            "delta_vs_baseline": eff - base_eff,
            "rel_change": (eff - base_eff) / base_eff,
            "signflip_p": p_sf,
            "cluster_signflip_p": p_cl,
            "smd_after_added_cov": bal["per_covariate"][COV_NAMES[k]]["smd_after"],
            "smd_before_added_cov": bal["per_covariate"][COV_NAMES[k]]["smd_before"],
        }

# ================================================================ trajectory
DENSITY_COVS = {"log10_n_ball_2d", "log10_n_ball_1d", "log10_W_pop_2d", "log10_W_pop_1d"}
MAIN_RUNGS = ["m1", "m2", "m3", "m4"]
trajectory: dict = {}
venue_calls: dict[str, str] = {}
for venue in VENUES:
    effs = {rg: rung_results[venue][rg]["channels"]["2d"]["mean_paired_diff"] for rg in MAIN_RUNGS}
    ses = {rg: rung_results[venue][rg]["channels"]["2d"]["se"] for rg in MAIN_RUNGS}
    p_sf = {rg: rung_results[venue][rg]["channels"]["2d"]["signflip_p"] for rg in MAIN_RUNGS}
    p_cl = {
        rg: rung_results[venue][rg]["channels"]["2d"]["cluster_signflip_p"] for rg in MAIN_RUNGS
    }
    e1, e4 = effs["m1"], effs["m4"]
    ratio = e4 / e1
    sign_stable = all(np.sign(effs[rg]) == np.sign(e1) for rg in MAIN_RUNGS)
    sig_m4 = (p_sf["m4"] < ALPHA) and (p_cl["m4"] < ALPHA)
    bal_ok_m4 = rung_results[venue]["m4"]["balance"]["balance_ok_matched_on"]
    tol = 0.25 * ses["m1"]
    seq = [effs[rg] for rg in MAIN_RUNGS]
    monotone_decline = all(seq[i + 1] <= seq[i] + tol for i in range(3))
    stable_within_errors = abs(e4 - e1) <= 2.0 * max(ses["m1"], ses["m4"])
    if sign_stable and sig_m4 and ratio >= 0.5 and bal_ok_m4:
        call = "SURVIVES"
    elif (not sig_m4) and (ratio < 0.5 or np.sign(e4) != np.sign(e1)):
        call = "KILLED"
    else:
        call = "UNDETERMINED"
    venue_calls[venue] = call
    trajectory[venue] = {
        "effects_2d": effs,
        "se_2d": ses,
        "signflip_p_2d": p_sf,
        "cluster_signflip_p_2d": p_cl,
        "ratio_m4_over_m1": ratio,
        "sign_stable": sign_stable,
        "sig_m4": sig_m4,
        "balance_ok_m4_matched_on": bal_ok_m4,
        "monotone_decline": monotone_decline,
        "stable_within_errors_2se": stable_within_errors,
        "call": call,
    }
overall = (
    "SURVIVES"
    if all(c == "SURVIVES" for c in venue_calls.values())
    else ("KILLED" if all(c == "KILLED" for c in venue_calls.values()) else "UNDETERMINED")
)

# over-matching symmetry note: identify the biggest single-covariate movers
movers: dict = {}
for venue in VENUES:
    aug = panel[venue]["augmented"]
    ranked = sorted(aug.items(), key=lambda kv: -abs(kv[1]["delta_vs_baseline"]))
    movers[venue] = [
        {
            "covariate": name,
            "delta_vs_baseline": v["delta_vs_baseline"],
            "rel_change": v["rel_change"],
            "is_density_mechanism_candidate": name in DENSITY_COVS,
        }
        for name, v in ranked[:4]
    ]

results = {
    "read": "D-2 extended-covariate confounding check on the M-2 matched 2D overlap residual",
    "date": "2026-08-07",
    "spec": "ch13 Part A (author-approved 2026-08-07); stage 0-1 free reads only",
    "question": (
        "does the matched +0.02225 (iiib) / +0.02070 (joint_r1) nat/event 2D overlap "
        "residual survive matching on covariates beyond (log10 ball-radius chord, SNR)?"
    ),
    "provenance": {
        "crb": {"path": CRB, "md5": md5(CRB), "md5_iiib_identical": md5(CRB_IIIB) == md5(CRB)},
        "event_likelihoods_md5": {v: md5(p) for v, p in VENUES.items()},
        "ball_meta": ball_meta,
        "catalogues": CATS,
        "m2_reference": M2_RESULTS,
        "census": {"sky_pairs": 1620, "sky_dl_pairs": 279, "overlap_events": 385},
        "rng": {
            "m2_reproduction_seed": 20260805,
            "d2_fresh_seed": 20260807,
            "n_perm": N_PERM,
            "order": (
                "reproduction: verbatim M-2 loop (venue x channel: unmatched perm then "
                "signflip); fresh: venue x rung(m1,m2,m3,m3e,m4) x channel(1d,2d) x "
                "(signflip, cluster); then panel venue x cov x (signflip, cluster)"
            ),
        },
    },
    "covariate_definitions": {
        "log10_radius_chord": "log10(2 sqrt(lam_max(J Sigma J^T))), M-2 original",
        "SNR": "prepared CRB SNR, M-2 original",
        "log10_n_ball_2d": "log10(1 + #galaxy_likelihoods[ev]) from frozeng h_0_73.json",
        "log10_n_ball_1d": "log10(1 + #ball2d + #additional_galaxies_without_bh_mass[ev])",
        "log10_W_pop_2d": "log10(1 + sum w_g over ball2d), w_g = R_eff_per_mbh(M_g)/(1+z_g), M-4 deref",
        "log10_W_pop_1d": "log10(1 + sum w_g over ball1d union)",
        "log10_dL": "log10(luminosity_distance)",
        "log10_rel_dL_err": "log10(sigma_dL/dL)",
        "abs_sin_gal_lat": "|sin b_gal|, astropy BarycentricTrueEcliptic(J2000)->Galactic",
        "abs_sin_ecl_lat": "|cos qS|",
        "bands": f"|lat| deg bands with edges {BAND_EDGES_DEG} (3 levels), exact-match rungs",
    },
    "reproduction_anchor": {
        "status": "PASS (hard asserts: SMDs, n_unique_controls, mean diffs to 1e-12; p bitwise)",
        "values": repro,
        "headline_asserts": {
            "iiib_2d_matched": 0.022252643015992925,
            "joint_r1_2d_matched": 0.020697491999731973,
        },
    },
    "rungs": rung_results,
    "m1_fresh_rng_consistency": m1_consistency,
    "single_covariate_panel": panel,
    "trajectory": trajectory,
    "venue_calls": venue_calls,
    "overall_call": overall,
    "top_movers": movers,
    "overmatching_note": (
        "Symmetric interpretation mandated: the ball density covariates "
        "(n_ball_2d/1d, W_pop_2d/1d) are mechanism candidates (the LIVE CLUE places the "
        "owner outside the annihilated catalogue path or THROUGH the composition weights; "
        "local catalogue density can BE the physics). Collapse under those covariates is "
        "'density-absorbed', not automatically 'artifact'. Only collapse under pure "
        "geometry covariates (latitude bands, d_L, rel d_L error) cleanly indicates a "
        "selection artifact. See top_movers for which covariates actually move the residual."
    ),
}

with open(OUT, "w") as f:
    json.dump(results, f, indent=2)
log(f"wrote {OUT}")
print(
    json.dumps(
        {
            "venue_calls": venue_calls,
            "overall_call": overall,
            "trajectory": trajectory,
            "top_movers": movers,
        },
        indent=2,
    )
)
