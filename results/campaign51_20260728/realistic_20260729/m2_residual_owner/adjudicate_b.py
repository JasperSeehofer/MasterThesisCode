"""Adversarial verification of Instrument B (adjudicate_b.py).

Independent re-implementation of every decision statistic in
PREREGISTRATION_B_COMPLETION_WEIGHT_FUNCTIONAL.md as executed by
b1_completion_weight_functional.py -> b1_results.json, PLUS attack batteries.

Independence choices (deliberately different numerical routes):
  * OLS fits via QR decomposition (numpy.linalg.qr + scipy solve_triangular)
    instead of np.linalg.lstsq (SVD/gelsd).
  * 1-NN matching via scipy.spatial.cKDTree instead of brute-force argmin
    on squared distances.
  * Sky-radius r from the ANALYTIC 2x2 symmetric eigenvalue formula
    lam_max = (a+b)/2 + sqrt(((a-b)/2)^2 + c^2) instead of np.linalg.eigvalsh
    (cross-checked against eigvalsh; census counts asserted).
  * Fresh permutation seeds (signflip 424243, cluster 971117), N_PERM 20000.
  * Spearman/Pearson via scipy (same library, spot-check only).

Ball-covariate builder and pruning functions are reused from the repo /
D-2 verbatim BY PREREG MANDATE (they define the covariates; fidelity is
independently pinned by recomputing the V-4 D-2 m2-rung anchor bitwise).

Attack batteries:
  ATT-1 overfitting: out-of-fold (cross-fitted) predictions for D and J
        (deterministic pos%5 folds AND fresh random 5-fold, seed 777);
        E_D/E_J/A_rho recomputed from cross-fitted predictions.
  ATT-2 half-split stability: fit on random halves (seed 778, 3 draws,
        both halves), predict all, A_rho per fit.
  ATT-3 attribution-order instability: reverse order (P first) -> E_P,
        A_dL = matched(chat_J - chat_P); common/unique share table.
  ATT-4 orthogonalization invariance: J with P columns residualized on D
        (same span) -> max |chat_J - chat_J_orth| must be numerical noise.
  ATT-5 smooth-model (D) spec sensitivity: D total-degree 2/3/4/5 and
        full-tensor degree 3 (16 cols); E_D, E_J, A_rho per spec.

Output: adjudicate_b_results.json (this directory).  Free-compute read
only; no likelihood evaluations; no edits to existing files.

Run: cd /home/jasper/Repositories/MasterThesisCode && uv run python \
  results/campaign51_20260728/realistic_20260729/m2_residual_owner/adjudicate_b.py
"""

import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.linalg import solve_triangular
from scipy.spatial import cKDTree
from scipy.stats import pearsonr, spearmanr

REPO = "/home/jasper/Repositories/MasterThesisCode"
sys.path.insert(0, REPO)
os.chdir(REPO)

from master_thesis_code.emri_rate import R_eff_per_mbh  # noqa: E402
from master_thesis_code.galaxy_catalogue.handler import (  # noqa: E402
    _empiric_stellar_mass_to_BH_mass_relation,
    _mass_redshift_prune_mask,
    _reduced_catalog_column_names,
)

HERE = f"{REPO}/results/campaign51_20260728/realistic_20260729/m2_residual_owner"
OUT = f"{HERE}/adjudicate_b_results.json"
B1 = f"{HERE}/b1_results.json"
CRB = f"{REPO}/results/run_20260804_postfix/joint_r1/diagnostics/prepared_cramer_rao_bounds.csv"
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

N_PERM = 20000
SEED_SF = 424243  # fresh, != run's 20260808
SEED_CL = 971117  # fresh, != run's 20260808
SEED_FOLDS = 777
SEED_HALVES = 778
PRIMARY_DEGREE = 3

# prereg-locked anchors (quoted from the committed prereg, section 4)
M2_TOTALS = {"iiib": 0.022252643015992925, "joint_r1": 0.020697491999731973}
A2_PRIMARY_R2 = {"c_pure": 0.8832406614871592, "c_gfrac": 0.8747947939465979}
A2_E_D = {"c_pure": 0.008340732036016641, "c_gfrac": 0.008352697414993901}
A2_E_D_CSE = {"c_pure": 0.0029160903955559583, "c_gfrac": 0.0029146831062962105}
A2_CHAIN_RATIO_D = {"iiib": 0.6662301458609439, "joint_r1": 0.652514656137263}
A2_WT_073 = {"iiib": 0.0619668411108587, "joint_r1": 0.0708022510819941}
A2_DLN_1MWT = {"iiib": -0.022898659328417684, "joint_r1": -0.027145606456715793}
A2_OBS_TLEGA = {"iiib": -0.004711922212657903, "joint_r1": -0.005129927211131639}
D2_M2_RUNG = {"iiib": 0.003949491314625633, "joint_r1": 0.003845526436421696}
EXPECTED_MD5 = {
    VENUES["iiib"]: "ee9c997b7f41b18a34049e7e0ff1a20f",
    VENUES["joint_r1"]: "c895f2e4a5b4fd127e347a941d6b6263",
    CRB: "9a1f2a14384a9281c97ca3be312ddaab",
    BALLS["iiib"]: "34c50e91028b6a6458a2b145db545705",
    BALLS["joint_r1"]: "6c5aff4896459105a8ac047f1a48ca8c",
}
M_MIN, M_MAX, Z_MAX = 1e4, 1e7, 1.5

t0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


def md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------ QR-route OLS
def qr_fit_predict(X: np.ndarray, y: np.ndarray, fit_mask: np.ndarray | None = None) -> np.ndarray:
    """OLS via reduced QR on the fit subsample; prediction over all rows."""
    if fit_mask is None:
        fit_mask = np.ones(len(y), dtype=bool)
    Q, R = np.linalg.qr(X[fit_mask])
    beta = solve_triangular(R, Q.T @ y[fit_mask])
    return X @ beta


def r2(y: np.ndarray, yhat: np.ndarray) -> float:
    return 1.0 - float(((y - yhat) ** 2).sum()) / float(((y - y.mean()) ** 2).sum())


def poly_design(x1: np.ndarray, x2: np.ndarray, degree: int) -> np.ndarray:
    cols = [x1**i * x2**j for i in range(degree + 1) for j in range(degree + 1 - i)]
    return np.stack(cols, axis=1)


def tensor_design(x1: np.ndarray, x2: np.ndarray, degree: int) -> np.ndarray:
    """FULL tensor-product design (degree+1)^2 columns (ATT-5 alternative)."""
    cols = [x1**i * x2**j for i in range(degree + 1) for j in range(degree + 1)]
    return np.stack(cols, axis=1)


def log_mean(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    out = np.empty_like(x)
    close = np.isclose(x, y, rtol=1e-12, atol=0.0)
    out[close] = 0.5 * (x[close] + y[close])
    nc = ~close
    out[nc] = (x[nc] - y[nc]) / (np.log(x[nc]) - np.log(y[nc]))
    return out


def perm_p(diffs: np.ndarray, groups: np.ndarray | None, seed: int) -> float:
    """Two-sided sign-flip permutation p; groups=None -> per-pair flips."""
    rng = np.random.default_rng(seed)
    obs = abs(diffs.mean())
    if groups is None:
        inv = np.arange(len(diffs))
        n_units = len(diffs)
    else:
        _, inv = np.unique(groups, return_inverse=True)
        n_units = int(inv.max()) + 1
    count = 0
    B = 2000
    done = 0
    while done < N_PERM:
        b = min(B, N_PERM - done)
        signs = rng.integers(0, 2, size=(b, n_units)).astype(np.float64) * 2.0 - 1.0
        stats = np.abs((signs[:, inv] * diffs[None, :]).mean(axis=1))
        count += int((stats >= obs).sum())
        done += b
    return float((count + 1) / (N_PERM + 1))


# ------------------------------------------------------------------ inputs, md5
log("md5 asserts (independent)")
md5_got = {}
for path, exp in EXPECTED_MD5.items():
    got = md5(path)
    md5_got[path] = got
    assert got == exp, ("md5 mismatch", path, got, exp)
cat_md5 = {v: md5(CATS[v]) for v in CATS}  # provenance record only (prereg has no anchor)
log(f"catalogue md5 (record only): {cat_md5}")

# ------------------------------------------------------------------ census (independent route)
log("census: independent implementation (analytic 2x2 eigenvalues)")
df = pd.read_csv(CRB)
n = len(df)
assert n == 1590, n
theta = df["qS"].to_numpy()
phi = df["phiS"].to_numpy()
a11 = df["delta_phiS_delta_phiS"].to_numpy()
a22 = df["delta_qS_delta_qS"].to_numpy()
a12 = df["delta_phiS_delta_qS"].to_numpy()
dl = df["luminosity_distance"].to_numpy()
s_dl = np.sqrt(df["delta_luminosity_distance_delta_luminosity_distance"].to_numpy())
snr = df["SNR"].to_numpy()

# J Sigma J^T with J = diag(|sin theta|, 1): [[s^2*a11, s*a12], [s*a12, a22]]
s = np.abs(np.sin(theta))
A = s**2 * a11
Bq = a22
Cc = s * a12
lam_max = 0.5 * (A + Bq) + np.sqrt((0.5 * (A - Bq)) ** 2 + Cc**2)
r_analytic = 2.0 * np.sqrt(np.maximum(lam_max, 0.0))
# cross-check vs eigvalsh route
r_eig = np.empty(n)
for i in range(n):
    sig = np.array([[a11[i], a12[i]], [a12[i], a22[i]]])
    jac = np.diag([abs(np.sin(theta[i])), 1.0])
    r_eig[i] = 2.0 * np.sqrt(max(float(np.linalg.eigvalsh(jac @ sig @ jac.T).max()), 0.0))
r_route_dev = float(np.max(np.abs(r_analytic - r_eig) / np.maximum(r_analytic, 1e-300)))
r = r_eig  # use eigvalsh values for census/matching identity; analytic is the cross-check
log(f"radius route max rel dev (analytic vs eigvalsh): {r_route_dev:.3e}")

st = np.sin(theta)
xyz = np.stack([st * np.cos(phi), st * np.sin(phi), np.cos(theta)], axis=1)
d = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=2)
iu = np.triu_indices(n, k=1)
sky = d[iu] <= (r[:, None] + r[None, :])[iu]
ii, jj = iu[0][sky], iu[1][sky]
lo, hi = dl - 2 * s_dl, dl + 2 * s_dl
win = (lo[ii] <= hi[jj]) & (lo[jj] <= hi[ii])
overlap = np.zeros(n, dtype=bool)
overlap[ii[win]] = True
overlap[jj[win]] = True
census = {
    "sky_pairs": int(sky.sum()),
    "window_pairs": int(win.sum()),
    "overlap_events": int(overlap.sum()),
}
assert census == {"sky_pairs": 1620, "window_pairs": 279, "overlap_events": 385}, census
# census under analytic radii (route-stability check)
sky_a = d[iu] <= (r_analytic[:, None] + r_analytic[None, :])[iu]
win_a = (lo[iu[0][sky_a]] <= hi[iu[1][sky_a]]) & (lo[iu[1][sky_a]] <= hi[iu[0][sky_a]])
census_analytic = {"sky_pairs": int(sky_a.sum()), "window_pairs": int(win_a.sum())}

log10_r = np.log10(r)
log10_dL = np.log10(dl)
log10_rel = np.log10(s_dl / dl)

# ------------------------------------------------------------------ venues, chords
log("loading venue CSVs")
frames = {}
ev_ref = None
for venue, path in VENUES.items():
    el = pd.read_csv(path)
    assert len(el) == 65108, (venue, len(el))
    a = el[np.abs(el.h - 0.60) < 1e-9].set_index("event_idx").sort_index()
    b = el[np.abs(el.h - 0.73) < 1e-9].set_index("event_idx").sort_index()
    evs = np.array(sorted(set(a.index) & set(b.index)))
    assert len(evs) == 1588, (venue, len(evs))
    if ev_ref is None:
        ev_ref = evs
    else:
        assert (evs == ev_ref).all()
    frames[venue] = (a.loc[evs], b.loc[evs])
ev = ev_ref
dropped = sorted(set(range(1590)) - set(ev.tolist()))
log(f"dropped events: {dropped}")

# V-3 venue identity (independent assert)
a_i, b_i = frames["iiib"]
a_j, b_j = frames["joint_r1"]
v3 = {}
for col in ("L_comp", "B_num", "B_num_wbh", "g_frac"):
    same60 = bool((a_i[col].to_numpy() == a_j[col].to_numpy()).all())
    same73 = bool((b_i[col].to_numpy() == b_j[col].to_numpy()).all())
    v3[col] = {"h0.60": same60, "h0.73": same73}
    assert same60 and same73, ("V-3 failure", col)

Lc60 = a_i.L_comp.to_numpy()
Lc73 = b_i.L_comp.to_numpy()
g60 = a_i.g_frac.to_numpy()
g73 = b_i.g_frac.to_numpy()
c_pure = np.log(Lc60) - np.log(Lc73)
c_gfrac = c_pure + (np.log(g60) - np.log(g73))
CHORDS = {"c_pure": c_pure, "c_gfrac": c_gfrac}

x1 = log10_dL[ev]
x2 = log10_rel[ev]
ov_mask = overlap[ev]
ct_mask = ~ov_mask

# ------------------------------------------------------------------ matching (cKDTree route)
log("matching via cKDTree (independent route)")
cov_mat = np.stack([log10_r, snr], axis=1)[ev]
z_std = (cov_mat - cov_mat.mean(axis=0)) / cov_mat.std(axis=0, ddof=1)
ov_idx = np.where(ov_mask)[0]
ct_idx = np.where(ct_mask)[0]
tree = cKDTree(z_std[ct_mask])
_, nn = tree.query(z_std[ov_mask], k=1)
mct_idx = ct_idx[nn]
assert len(ov_idx) == 385 and len(ct_idx) == 1203
n_clusters = int(len(np.unique(nn)))
assert n_clusters == 234, n_clusters


def matched(vals: np.ndarray) -> dict:
    diffs = vals[ov_idx] - vals[mct_idx]
    m = diffs.mean()
    cse = 0.0
    for c in np.unique(nn):
        cse += float((diffs[nn == c] - m).sum()) ** 2
    return {
        "mean_paired_diff": float(m),
        "paired_se": float(diffs.std(ddof=1) / np.sqrt(len(diffs))),
        "cluster_robust_se": float(np.sqrt(cse) / len(diffs)),
    }


def matched_p(vals: np.ndarray) -> dict:
    diffs = vals[ov_idx] - vals[mct_idx]
    out = matched(vals)
    out["signflip_p"] = perm_p(diffs, None, SEED_SF)
    out["cluster_signflip_p"] = perm_p(diffs, nn, SEED_CL)
    return out


# V-1: bitwise M-2 totals through MY census+matching route (no fitting involved)
v1 = {}
for venue, (a, b) in frames.items():
    chord_csv = np.log(a.combined_with_bh.to_numpy()) - np.log(b.combined_with_bh.to_numpy())
    got = matched(chord_csv)["mean_paired_diff"]
    v1[venue] = {"matched_2d_total": got, "anchor": M2_TOTALS[venue], "bitwise": got == M2_TOTALS[venue]}
    assert got == M2_TOTALS[venue], (venue, got)
log("V-1 bitwise PASS through independent census+matching route")

# ------------------------------------------------------------------ D design + V-2 (QR route)
D = poly_design(x1, x2, PRIMARY_DEGREE)
fits_D = {}
fits_D_ct = {}
v2 = {}
for name, y in CHORDS.items():
    yhat = qr_fit_predict(D, y)
    fits_D[name] = yhat
    fits_D_ct[name] = qr_fit_predict(D, y, fit_mask=ct_mask)
    got_r2 = r2(y, yhat)
    ed = matched(y - yhat)
    v2[name] = {
        "r2_dev_vs_anchor": got_r2 - A2_PRIMARY_R2[name],
        "E_D_dev_vs_anchor": ed["mean_paired_diff"] - A2_E_D[name],
        "E_D_cse_dev_vs_anchor": ed["cluster_robust_se"] - A2_E_D_CSE[name],
        "r2": got_r2,
        "E_D": ed["mean_paired_diff"],
        "E_D_cluster_se": ed["cluster_robust_se"],
    }
    assert abs(v2[name]["r2_dev_vs_anchor"]) < 1e-10, (name, v2[name])
    assert abs(v2[name]["E_D_dev_vs_anchor"]) < 1e-11, (name, v2[name])
    assert abs(v2[name]["E_D_cse_dev_vs_anchor"]) < 1e-11, (name, v2[name])
E_D_ref = {name: matched_p(CHORDS[name] - fits_D[name]) for name in CHORDS}

# chain constants + S_A per venue (independent LMDI build)
venue_chain = {}
for venue, (a, b) in frames.items():
    wt60, wt73 = float(a.w_G.iloc[0]), float(b.w_G.iloc[0])
    assert a.w_G.nunique() == 1 and b.w_G.nunique() == 1
    dln_w1m = float(np.log1p(-wt60) - np.log1p(-wt73))
    assert wt73 == A2_WT_073[venue] and dln_w1m == A2_DLN_1MWT[venue]
    Lcat60 = a.L_cat_with_bh.to_numpy()
    Lcat73 = b.L_cat_with_bh.to_numpy()
    A60, A73 = wt60 * Lcat60, wt73 * Lcat73
    B60, B73 = (1 - wt60) * g60 * Lc60, (1 - wt73) * g73 * Lc73
    F60 = a.combined_with_bh.to_numpy()
    F73 = b.combined_with_bh.to_numpy()
    recomb_dev = float(
        np.max(np.abs((A60 + B60) - F60) / F60) + np.max(np.abs((A73 + B73) - F73) / F73)
    )
    LF = log_mean(A60 + B60, A73 + B73)
    SB = log_mean(B60, B73) / LF
    T_legB = SB * (dln_w1m + c_gfrac)
    T_legA = (np.log(A60 + B60) - np.log(A73 + B73)) - T_legB
    obs_total = matched(np.log(F60) - np.log(F73))
    obs_TlegA = matched(T_legA)
    assert obs_TlegA["mean_paired_diff"] == A2_OBS_TLEGA[venue], (venue, obs_TlegA)
    TlegB_hat_D = SB * (dln_w1m + fits_D["c_gfrac"])
    ratio_D = (
        obs_TlegA["mean_paired_diff"] + matched(TlegB_hat_D)["mean_paired_diff"]
    ) / obs_total["mean_paired_diff"]
    ratio_D_dev = ratio_D - A2_CHAIN_RATIO_D[venue]
    assert abs(ratio_D_dev) < 1e-9, (venue, ratio_D, ratio_D_dev)
    z60, z73 = Lcat60 == 0.0, Lcat73 == 0.0
    assert (z60 == z73).all()
    nz = ~z60
    SA = np.zeros_like(LF)
    SA[nz] = log_mean(A60[nz], A73[nz]) / LF[nz]
    venue_chain[venue] = {
        "wt60": wt60,
        "wt73": wt73,
        "dln_w1m": dln_w1m,
        "SB": SB,
        "SA": SA,
        "obs_total": obs_total,
        "obs_TlegA": obs_TlegA,
        "ratio_D": ratio_D,
        "ratio_D_dev": ratio_D_dev,
        "recombination_rel_dev": recomb_dev,
    }
log("V-2 anchors reproduced (QR route, dev < 1e-10) + chain constants bitwise")

# ------------------------------------------------------------------ ball covariates (mandated builder)
log("ball covariates (D-2/prereg-mandated builder; V-4 pins fidelity)")


def load_pruned_zm(path: str) -> tuple:
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
        pd.Series(mbh), pd.Series(mbh_err), pd.Series(z), pd.Series(sz), M_MIN, M_MAX, Z_MAX
    ).to_numpy()
    return z[mask], mbh[mask]


ball_cov = {}
ball_meta = {}
for venue in VENUES:
    log(f"  {venue}: ball json + pruned catalogue")
    with open(BALLS[venue]) as f:
        ball = json.load(f)
    gl = ball["galaxy_likelihoods"]
    ag = ball["additional_galaxies_without_bh_mass"]
    zz, mm = load_pruned_zm(CATS[venue])
    w_all = R_eff_per_mbh(mm) / (1.0 + zz)
    arr = np.full((n, 4), np.nan)
    for e in range(n):
        k = str(e)
        if k not in gl:
            continue
        ids2 = np.array([row[0] for row in gl[k]], dtype=np.int64)
        ids_extra = np.array([row[0] for row in ag.get(k, [])], dtype=np.int64)
        ids1 = np.concatenate([ids2, ids_extra])
        w2 = float(w_all[ids2].sum()) if len(ids2) else 0.0
        w1 = float(w_all[ids1].sum()) if len(ids1) else 0.0
        arr[e] = [len(ids2), len(ids1), w2, w1]
    assert not np.isnan(arr[ev]).any()
    ball_cov[venue] = arr
    ball_meta[venue] = {"pruned_rows": int(len(zz))}
    del ball, gl, ag, zz, mm, w_all

ZCOV = {}
for venue in VENUES:
    barr = ball_cov[venue]
    ZCOV[venue] = {
        "z1": np.log10(1.0 + barr[ev, 0]),
        "z2": np.log10(1.0 + barr[ev, 1]),
        "z3": np.log10(1.0 + barr[ev, 2]),
        "z4": np.log10(1.0 + barr[ev, 3]),
    }

# V-4: D-2 m2-rung bitwise through MY matching route
v4 = {}
for venue, (a, b) in frames.items():
    C = np.stack([log10_r, snr, np.log10(1.0 + ball_cov[venue][:, 0])], axis=1)[ev]
    zst = (C - C.mean(axis=0)) / C.std(axis=0, ddof=1)
    tr3 = cKDTree(zst[ct_mask])
    _, nn3 = tr3.query(zst[ov_mask], k=1)
    chord = np.log(a.combined_with_bh.to_numpy()) - np.log(b.combined_with_bh.to_numpy())
    eff = float((chord[ov_idx] - chord[ct_idx[nn3]]).mean())
    v4[venue] = {"m2_rung": eff, "anchor": D2_M2_RUNG[venue], "bitwise": eff == D2_M2_RUNG[venue]}
    assert eff == D2_M2_RUNG[venue], (venue, eff)
log("V-4 bitwise PASS through independent matching route")

# ------------------------------------------------------------------ designs P, J
P_of = {}
J_of = {}
for venue in VENUES:
    zc = ZCOV[venue]
    P = poly_design(zc["z1"], zc["z3"], PRIMARY_DEGREE)
    P_of[venue] = P
    J_of[venue] = np.concatenate([D, P[:, 1:]], axis=1)
    assert J_of[venue].shape == (1588, 19)

# ------------------------------------------------------------------ DS-1 (independent)
log("DS-1: decorrelation gate (independent)")
ds1 = {}
for venue in VENUES:
    zc = ZCOV[venue]
    cross = {}
    for zn in ("z1", "z2", "z3", "z4"):
        cross[zn] = r2(zc[zn], qr_fit_predict(D, zc[zn]))
    corr = {}
    for zn in ("z1", "z3"):
        corr[zn] = {
            "x1_pearson": float(pearsonr(zc[zn], x1).statistic),
            "x1_spearman": float(spearmanr(zc[zn], x1).statistic),
            "x2_pearson": float(pearsonr(zc[zn], x2).statistic),
            "x2_spearman": float(spearmanr(zc[zn], x2).statistic),
        }
    J = J_of[venue]
    Jn = J[:, 1:]
    Jn_std = (Jn - Jn.mean(axis=0)) / Jn.std(axis=0, ddof=1)
    cond_std = float(np.linalg.cond(Jn_std))
    cond_raw = float(np.linalg.cond(J))
    # VIF spot checks (3 representative columns) via QR route
    vif_spot = {}
    for kcol, lab in ((1, "x1^0*x2^1"), (10, "z1^0*z3^1"), (14, "z1^1*z3^1")):
        others = np.delete(J, kcol, axis=1)
        r2k = r2(J[:, kcol], qr_fit_predict(others, J[:, kcol]))
        vif_spot[lab] = float(1.0 / (1.0 - r2k))
    gate = bool((cross["z1"] <= 0.80 or cross["z3"] <= 0.80) and cond_std < 1e8)
    ds1[venue] = {
        "cross_family_r2_on_D": cross,
        "pairwise_corr_z1_z3": corr,
        "cond_standardized_J_non_intercept": cond_std,
        "cond_raw_J": cond_raw,
        "vif_spot_checks": vif_spot,
        "gate_pass": gate,
    }

# ------------------------------------------------------------------ DS-2/DS-3 (independent)
log("DS-2/DS-3 (QR route, fresh-seed p)")


def excess(chat_D_map: dict, chat_J_map: dict) -> dict:
    out = {}
    for name, y in CHORDS.items():
        cd, cj = chat_D_map[name], chat_J_map[name]
        e_d = matched(y - cd)["mean_paired_diff"]
        ej = matched_p(y - cj)
        ar = matched_p(cj - cd)
        out[name] = {
            "E_D": e_d,
            "E_J": ej,
            "A_rho": ar,
            "share": ar["mean_paired_diff"] / e_d,
            "identity_gap": e_d - ar["mean_paired_diff"] - ej["mean_paired_diff"],
            "abs_EJ_over_cse": abs(ej["mean_paired_diff"]) / ej["cluster_robust_se"],
        }
    return out


fits_J = {}
fits_J_ct = {}
fits_P = {}
ds2_ds3 = {}
r2_table = {}
for venue in VENUES:
    fits_J[venue] = {name: qr_fit_predict(J_of[venue], y) for name, y in CHORDS.items()}
    fits_J_ct[venue] = {
        name: qr_fit_predict(J_of[venue], y, fit_mask=ct_mask) for name, y in CHORDS.items()
    }
    fits_P[venue] = {name: qr_fit_predict(P_of[venue], y) for name, y in CHORDS.items()}
    ds2_ds3[venue] = {
        "all_events_fit": excess(fits_D, fits_J[venue]),
        "controls_only_fit": excess(fits_D_ct, fits_J_ct[venue]),
    }
    r2_table[venue] = {
        name: {
            "r2_D": r2(y, fits_D[name]),
            "r2_P": r2(y, fits_P[venue][name]),
            "r2_J": r2(y, fits_J[venue][name]),
        }
        for name, y in CHORDS.items()
    }
    for name in CHORDS:
        gap = ds2_ds3[venue]["all_events_fit"][name]["identity_gap"]
        assert abs(gap) < 1e-13, (venue, name, gap)

# ------------------------------------------------------------------ DS-4 (independent)
ds4 = {}
for venue in VENUES:
    vc = venue_chain[venue]
    entry = {"rho_D_anchor_dev": vc["ratio_D_dev"], "rho_D": vc["ratio_D"]}
    for label, fj in (("all_events_fit", fits_J[venue]), ("controls_only_fit", fits_J_ct[venue])):
        TlegB_hat = vc["SB"] * (vc["dln_w1m"] + fj["c_gfrac"])
        pred_total = vc["obs_TlegA"]["mean_paired_diff"] + matched(TlegB_hat)["mean_paired_diff"]
        entry[f"rho_J_{label}"] = pred_total / vc["obs_total"]["mean_paired_diff"]
    ds4[venue] = entry

# ------------------------------------------------------------------ DS-5 (independent)
log("DS-5: radius overlay + coherence (QR route)")
r_ev = log10_r[ev]
R_COLS = np.stack([r_ev, r_ev**2, r_ev**3], axis=1)
D_r = np.concatenate([D, R_COLS], axis=1)
ds5 = {}
for venue in VENUES:
    J_r = np.concatenate([J_of[venue], R_COLS], axis=1)
    overlay = {}
    for name, y in CHORDS.items():
        cd = qr_fit_predict(D_r, y)
        cj = qr_fit_predict(J_r, y)
        overlay[name] = {
            "A_rho_plus_r": matched_p(cj - cd),
            "E_J_plus_r": matched_p(y - cj),
            "E_D_plus_r": matched(y - cd)["mean_paired_diff"],
        }
    zc = ZCOV[venue]
    coher = {}
    for name, y in CHORDS.items():
        cob = {}
        for zn in ("z1", "z3"):
            zarr = zc[zn]
            ry = y - qr_fit_predict(D, y)
            rz = zarr - qr_fit_predict(D, zarr)
            pr_all = float(pearsonr(ry, rz).statistic)
            ry_c = y - qr_fit_predict(D, y, fit_mask=ct_mask)
            rz_c = zarr - qr_fit_predict(D, zarr, fit_mask=ct_mask)
            cob[zn] = {
                "partial_all": pr_all,
                "partial_ctfit_eval_all": float(pearsonr(ry_c, rz_c).statistic),
                "partial_ctfit_eval_controls": float(
                    pearsonr(ry_c[ct_mask], rz_c[ct_mask]).statistic
                ),
            }
            # CV folds: deterministic pos%5 (compare) AND fresh random folds (attack)
            for tag, folds in (
                ("det", np.arange(1588) % 5),
                ("rand", np.random.default_rng(SEED_FOLDS).permutation(1588) % 5),
            ):
                vals = []
                for kf in range(5):
                    te = folds == kf
                    tr = ~te
                    ry_f = y - qr_fit_predict(D, y, fit_mask=tr)
                    rz_f = zarr - qr_fit_predict(D, zarr, fit_mask=tr)
                    vals.append(float(pearsonr(ry_f[te], rz_f[te]).statistic))
                cob[zn][f"cv_{tag}_fold_partials"] = vals
                cob[zn][f"cv_{tag}_all_same_sign"] = bool(
                    all(np.sign(v) == np.sign(vals[0]) for v in vals)
                )
        cob["z1_z3_same_sign"] = bool(
            np.sign(cob["z1"]["partial_all"]) == np.sign(cob["z3"]["partial_all"])
        )
        coher[name] = cob
    ds5[venue] = {"radius_overlay": overlay, "coherence": coher}

# ------------------------------------------------------------------ DS-6 (independent)
log("DS-6: weight channel")
ds6 = {}
for venue in VENUES:
    SA = venue_chain[venue]["SA"]
    pos = SA > 0
    y3 = np.log10(SA[pos])
    Ds, Ps, Js = D[pos], P_of[venue][pos], J_of[venue][pos]
    yD = qr_fit_predict(Ds, y3)
    yP = qr_fit_predict(Ps, y3)
    yJ = qr_fit_predict(Js, y3)
    zc = ZCOV[venue]
    partials = {}
    for zn in ("z1", "z3"):
        rz = zc[zn][pos] - qr_fit_predict(Ds, zc[zn][pos])
        partials[f"y3_vs_{zn}_given_D"] = float(pearsonr(y3 - yD, rz).statistic)
    for xn, xa in (("x1", x1), ("x2", x2)):
        rx = xa[pos] - qr_fit_predict(Ps, xa[pos])
        partials[f"y3_vs_{xn}_given_P"] = float(pearsonr(y3 - yP, rx).statistic)
    ds6[venue] = {
        "n_SA_zero": int((~pos).sum()),
        "n_fitted": int(pos.sum()),
        "n_overlap_pos": int(pos[ov_idx].sum()),
        "r2_D": r2(y3, yD),
        "r2_P": r2(y3, yP),
        "r2_J": r2(y3, yJ),
        "dR2_P_given_D": r2(y3, yJ) - r2(y3, yD),
        "dR2_D_given_P": r2(y3, yJ) - r2(y3, yP),
        "partials": partials,
    }

# ------------------------------------------------------------------ R-1/R-2/R-3 spot recompute
log("R-1/R-2/R-3 recompute")


def multipoly_deg2(vs: list) -> np.ndarray:
    exps = sorted(e for e in np.ndindex(*([3] * len(vs))) if sum(e) <= 2)
    cols = []
    for e in exps:
        col = np.ones_like(vs[0])
        for vv, p in zip(vs, e):
            col = col * vv**p
        cols.append(col)
    return np.stack(cols, axis=1)


robust = {"R1": {}, "R2": {}, "R3": {}}
for venue in VENUES:
    zc = ZCOV[venue]
    P1 = multipoly_deg2([zc["z1"], zc["z2"], zc["z3"], zc["z4"]])
    J1 = np.concatenate([D, P1[:, 1:]], axis=1)
    robust["R1"][venue] = excess(fits_D, {nm: qr_fit_predict(J1, y) for nm, y in CHORDS.items()})
    ext = np.stack([zc["z2"], zc["z4"], zc["z2"] ** 2, zc["z4"] ** 2], axis=1)
    J2 = np.concatenate([D, P_of[venue][:, 1:], ext], axis=1)
    robust["R2"][venue] = excess(fits_D, {nm: qr_fit_predict(J2, y) for nm, y in CHORDS.items()})
    robust["R3"][venue] = {
        f"degree_{dg}": {
            nm: r2(y, qr_fit_predict(poly_design(zc["z1"], zc["z3"], dg), y))
            for nm, y in CHORDS.items()
        }
        for dg in (1, 2, 3, 4)
    }

# ================================================================== ATTACKS
log("ATT-1: out-of-fold cross-fitted E_D/E_J/A_rho")
att1 = {}
fold_sets = {
    "det_pos_mod_5": np.arange(1588) % 5,
    "rand_seed777": np.random.default_rng(SEED_FOLDS).permutation(1588) % 5,
}
for venue in VENUES:
    att1[venue] = {}
    for tag, folds in fold_sets.items():
        cd_cf = {nm: np.empty(1588) for nm in CHORDS}
        cj_cf = {nm: np.empty(1588) for nm in CHORDS}
        for kf in range(5):
            te = folds == kf
            tr = ~te
            for nm, y in CHORDS.items():
                cd_cf[nm][te] = qr_fit_predict(D, y, fit_mask=tr)[te]
                cj_cf[nm][te] = qr_fit_predict(J_of[venue], y, fit_mask=tr)[te]
        block = {}
        for nm, y in CHORDS.items():
            e_d = matched(y - cd_cf[nm])["mean_paired_diff"]
            ej = matched_p(y - cj_cf[nm])
            ar = matched_p(cj_cf[nm] - cd_cf[nm])
            block[nm] = {
                "E_D_oof": e_d,
                "E_J_oof": ej,
                "A_rho_oof": ar,
                "share_oof": ar["mean_paired_diff"] / e_d,
            }
        att1[venue][tag] = block

log("ATT-2: half-split stability of A_rho")
att2 = {}
rng_h = np.random.default_rng(SEED_HALVES)
half_masks = []
for rep in range(3):
    perm = rng_h.permutation(1588)
    mA = np.zeros(1588, dtype=bool)
    mA[perm[:794]] = True
    half_masks.append((f"rep{rep}_A", mA))
    half_masks.append((f"rep{rep}_B", ~mA))
for venue in VENUES:
    vals = {}
    for tag, mask in half_masks:
        cd = qr_fit_predict(D, c_pure, fit_mask=mask)
        cj = qr_fit_predict(J_of[venue], c_pure, fit_mask=mask)
        vals[tag] = matched(cj - cd)["mean_paired_diff"]
    arr = np.array(list(vals.values()))
    att2[venue] = {
        "A_rho_half_fits_c_pure": vals,
        "min": float(arr.min()),
        "max": float(arr.max()),
        "all_positive": bool((arr > 0).all()),
    }

log("ATT-3: attribution order reversal")
att3 = {}
for venue in VENUES:
    block = {}
    for nm, y in CHORDS.items():
        e_0 = matched(y)["mean_paired_diff"]
        e_p = matched(y - fits_P[venue][nm])["mean_paired_diff"]
        e_d = matched(y - fits_D[nm])["mean_paired_diff"]
        e_j = matched(y - fits_J[venue][nm])["mean_paired_diff"]
        a_rho = e_d - e_j  # density-unique (P after D)
        a_dl = e_p - e_j  # dL-unique (D after P)
        common = e_0 - e_j - a_rho - a_dl
        block[nm] = {
            "E_0_raw_matched": e_0,
            "E_P": e_p,
            "E_D": e_d,
            "E_J": e_j,
            "density_unique_A_rho": a_rho,
            "dL_unique_A_dL": a_dl,
            "common_shared": common,
            "density_total_E0_minus_EP": e_0 - e_p,
            "dL_total_E0_minus_ED": e_0 - e_d,
            "A_dL_matched_p": matched_p(fits_J[venue][nm] - fits_P[venue][nm]),
        }
    att3[venue] = block

log("ATT-4: orthogonalization invariance")
att4 = {}
for venue in VENUES:
    P = P_of[venue]
    P_orth = np.column_stack(
        [P[:, k] - qr_fit_predict(D, P[:, k]) for k in range(1, P.shape[1])]
    )
    J_orth = np.concatenate([D, P_orth], axis=1)
    devs = {}
    for nm, y in CHORDS.items():
        cj_o = qr_fit_predict(J_orth, y)
        devs[nm] = float(np.max(np.abs(cj_o - fits_J[venue][nm])))
    att4[venue] = {"max_abs_pred_dev_orth_vs_raw_J": devs}

log("ATT-5: D-spec sensitivity of E_D / E_J / A_rho (c_pure)")
att5 = {}
D_variants = {
    "deg2": poly_design(x1, x2, 2),
    "deg3_primary": D,
    "deg4": poly_design(x1, x2, 4),
    "deg5": poly_design(x1, x2, 5),
    "tensor3": tensor_design(x1, x2, 3),
}
for venue in VENUES:
    att5[venue] = {}
    for tag, Dv in D_variants.items():
        Jv = np.concatenate([Dv, P_of[venue][:, 1:]], axis=1)
        cd = qr_fit_predict(Dv, c_pure)
        cj = qr_fit_predict(Jv, c_pure)
        ed = matched(c_pure - cd)
        ej = matched(c_pure - cj)
        ar = matched_p(cj - cd)
        att5[venue][tag] = {
            "D_cols": int(Dv.shape[1]),
            "E_D": ed["mean_paired_diff"],
            "E_D_cluster_se": ed["cluster_robust_se"],
            "E_J": ej["mean_paired_diff"],
            "A_rho": ar["mean_paired_diff"],
            "A_rho_cluster_se": ar["cluster_robust_se"],
            "A_rho_cluster_signflip_p": ar["cluster_signflip_p"],
            "share": ar["mean_paired_diff"] / ed["mean_paired_diff"],
        }

# ================================================================== compare vs b1_results.json
log("quote-compare vs b1_results.json")
with open(B1) as f:
    b1 = json.load(f)

TOL = 5e-9
comp = {"mismatches": [], "max_devs": {}}


def cmp(path: str, mine: float, theirs: float, tol: float = TOL) -> None:
    dev = abs(mine - theirs)
    comp["max_devs"][path] = dev
    if dev > tol:
        comp["mismatches"].append({"path": path, "mine": mine, "b1": theirs, "dev": dev})


for venue in VENUES:
    b = b1["ds1_family_decorrelation_gate"][venue]
    for zn in ("z1", "z2", "z3", "z4"):
        cmp(f"ds1.{venue}.cross_r2.{zn}", ds1[venue]["cross_family_r2_on_D"][zn],
            b["cross_family_r2_on_D"][zn])
    cmp(f"ds1.{venue}.cond_std", ds1[venue]["cond_standardized_J_non_intercept"],
        b["condition_number_standardized_J_non_intercept"], tol=1.0)
    if ds1[venue]["gate_pass"] != b["gate_pass_measured"]:
        comp["mismatches"].append({"path": f"ds1.{venue}.gate", "mine": ds1[venue]["gate_pass"]})
    for fit_tag, b_tag in (("all_events_fit", "all_events_fit_primary"),
                           ("controls_only_fit", "controls_only_fit")):
        for nm in CHORDS:
            mine = ds2_ds3[venue][fit_tag][nm]
            theirs = b1["ds2_ds3_excess_and_attribution"][venue][b_tag][nm]
            cmp(f"ds2.{venue}.{fit_tag}.{nm}.E_D", mine["E_D"], theirs["E_D_mean_paired_diff"])
            cmp(f"ds2.{venue}.{fit_tag}.{nm}.E_J", mine["E_J"]["mean_paired_diff"],
                theirs["E_J"]["mean_paired_diff"])
            cmp(f"ds2.{venue}.{fit_tag}.{nm}.E_J_cse", mine["E_J"]["cluster_robust_se"],
                theirs["E_J"]["cluster_robust_se"])
            cmp(f"ds3.{venue}.{fit_tag}.{nm}.A_rho", mine["A_rho"]["mean_paired_diff"],
                theirs["A_rho"]["mean_paired_diff"])
            cmp(f"ds3.{venue}.{fit_tag}.{nm}.A_rho_cse", mine["A_rho"]["cluster_robust_se"],
                theirs["A_rho"]["cluster_robust_se"])
            cmp(f"ds3.{venue}.{fit_tag}.{nm}.share", mine["share"], theirs["A_rho_share_of_E_D"])
    cmp(f"ds4.{venue}.rho_J_all", ds4[venue]["rho_J_all_events_fit"],
        b1["ds4_chain_ratio"][venue]["rho_J_all_events_fit"]["ratio_pred_over_obs"])
    cmp(f"ds4.{venue}.rho_J_controls", ds4[venue]["rho_J_controls_only_fit"],
        b1["ds4_chain_ratio"][venue]["rho_J_controls_only_fit"]["ratio_pred_over_obs"])
    for nm in CHORDS:
        mine5 = ds5[venue]["radius_overlay"][nm]
        theirs5 = b1["ds5_radius_overlay_and_coherence"][venue]["radius_overlay"][nm]
        cmp(f"ds5.{venue}.{nm}.A_rho_plus_r", mine5["A_rho_plus_r"]["mean_paired_diff"],
            theirs5["A_rho_plus_r"]["mean_paired_diff"])
        cmp(f"ds5.{venue}.{nm}.E_J_plus_r", mine5["E_J_plus_r"]["mean_paired_diff"],
            theirs5["E_J_plus_r"]["mean_paired_diff"])
        for zn in ("z1", "z3"):
            cmp(f"ds5.{venue}.{nm}.{zn}.partial_all",
                ds5[venue]["coherence"][nm][zn]["partial_all"],
                b1["ds5_radius_overlay_and_coherence"][venue][
                    "partial_association_sign_coherence"][nm][zn]["partial_corr_all_events"])
    b6 = b1["ds6_weight_channel"][venue]
    cmp(f"ds6.{venue}.r2_D", ds6[venue]["r2_D"], b6["r2_D"])
    cmp(f"ds6.{venue}.r2_P", ds6[venue]["r2_P"], b6["r2_P"])
    cmp(f"ds6.{venue}.r2_J", ds6[venue]["r2_J"], b6["r2_J"])
    if ds6[venue]["n_SA_zero"] != b6["n_SA_zero_structural"]:
        comp["mismatches"].append({"path": f"ds6.{venue}.n_SA_zero", "mine": ds6[venue]["n_SA_zero"]})
    for rk, b_rk in (("R1", "R1"), ("R2", "R2")):
        for nm in CHORDS:
            mine_r = robust[rk][venue][nm]
            theirs_r = b1["robustness"][b_rk][venue][nm]
            cmp(f"{rk}.{venue}.{nm}.E_J", mine_r["E_J"]["mean_paired_diff"],
                theirs_r["E_J"]["mean_paired_diff"])
            cmp(f"{rk}.{venue}.{nm}.A_rho", mine_r["A_rho"]["mean_paired_diff"],
                theirs_r["A_rho"]["mean_paired_diff"])
    for dg in (1, 2, 3, 4):
        for nm in CHORDS:
            cmp(f"R3.{venue}.deg{dg}.{nm}", robust["R3"][venue][f"degree_{dg}"][nm],
                b1["robustness"]["R3"][venue][f"degree_{dg}"][nm])

max_dev_overall = max(comp["max_devs"].values())
log(f"comparison: {len(comp['mismatches'])} mismatches; max dev {max_dev_overall:.3e}")

# ================================================================== write
results = {
    "adjudication": "Instrument B adversarial verification (independent QR/cKDTree/analytic-eig route)",
    "date": time.strftime("%Y-%m-%d"),
    "seeds": {"signflip": SEED_SF, "cluster": SEED_CL, "folds": SEED_FOLDS, "halves": SEED_HALVES},
    "inputs_md5_verified": md5_got,
    "catalogue_md5_record_only": cat_md5,
    "census": census,
    "census_analytic_radius_route": census_analytic,
    "radius_route_max_rel_dev": r_route_dev,
    "dropped_events": dropped,
    "n_control_clusters": n_clusters,
    "V1_bitwise": v1,
    "V2_anchor_devs": v2,
    "V3_venue_identity": v3,
    "V4_bitwise": v4,
    "venue_chain_checks": {
        v: {
            "recombination_rel_dev": venue_chain[v]["recombination_rel_dev"],
            "ratio_D_dev": venue_chain[v]["ratio_D_dev"],
        }
        for v in VENUES
    },
    "E_D_reference_fresh_p": E_D_ref,
    "ds1": ds1,
    "ds2_ds3": ds2_ds3,
    "ds4": ds4,
    "ds5": ds5,
    "ds6": ds6,
    "r2_table": r2_table,
    "robustness": robust,
    "att1_out_of_fold": att1,
    "att2_half_split": att2,
    "att3_order_reversal": att3,
    "att4_orthogonalization": att4,
    "att5_D_spec_sensitivity": att5,
    "comparison_vs_b1": comp,
}

with open(OUT, "w") as f:
    json.dump(results, f, indent=2, default=float)
log(f"wrote {OUT}")
