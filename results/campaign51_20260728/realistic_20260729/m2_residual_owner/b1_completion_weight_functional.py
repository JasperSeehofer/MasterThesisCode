"""Instrument B: the completion-weight functional read (pre-registered).

Executes PREREGISTRATION_B_COMPLETION_WEIGHT_FUNCTIONAL.md (registered commit
36fe7800130d538be146e5e1696629b7a416977a, main; parent c188f460) with ZERO
improvisation: DS-1..DS-6, R-1..R-3, validity criteria V-1..V-5.

Decision statistics (prereg section 3, verbatim):
  DS-1  family-decorrelation gate: cross-family R^2 of z1..z4 on the d_L
        design D; pairwise Pearson/Spearman across families; VIFs of J;
        condition number of the column-standardized J.  Gate: attribution
        proceeds iff at least one of (z1, z3) has cross-family R^2 <= 0.80
        AND standardized-J condition number < 1e8.
  DS-2  fixed-d_L excess re-estimated: E_D (bitwise A2 anchors) and E_J =
        matched mean paired diff of (c - c_hat_J) over the exact M-2 385
        pairs / 234 control clusters, both chord objects.
  DS-3  exact additive attribution: A_rho = matched diff of (c_hat_J -
        c_hat_D); identity E_D = A_rho + E_J asserted at machine precision;
        share A_rho/E_D.
  DS-4  stratum-composition chain ratio rho_J per venue (observed T_legA +
        predicted T_legB from c_hat_gfrac via A2's exact S_B/dln(1-wt)
        chain) / observed total.
  DS-5  radius-overlay stability: A_rho^{+r} with [r, r^2, r^3] appended to
        both designs; z1/z3 partial-association sign coherence, 5-fold CV +
        controls-only.
  DS-6  weight channel (secondary): y3 = log10(S_A) on the S_A > 0 subset
        per venue, fitted on D/P/J with semi-partials.

Designs: D = OLS total-degree-3 bivariate poly in (x1=log10_dL,
x2=log10_rel_dL_err), a2 poly_design byte-compatible, 10 cols.  P =
total-degree-3 bivariate poly in (z1=log10_n_ball_2d, z3=log10_W_pop_2d),
D-2 builder verbatim, 10 cols.  J = column union of D and P (single
intercept), 19 cols.  All fits numpy.linalg.lstsq, all 1588 events unless
stated (controls-only variants reported).

Machinery reused verbatim: a2_completion_functional.py (census, matching,
cluster_se, poly_design, matched_read, chain composition);
d1_component_decomposition.py (log_mean, LMDI S_A, signflip_p,
cluster_signflip_p); d2_confounding_check.py:383-412 (ball-covariate
builder, load_pruned_zm, m2-rung match).

RNG policy: all decision-bearing point estimates and SEs RNG-free;
p-values N_PERM = 20000, fresh seeds signflip 20260808, cluster 20260808.

Conservative conventions where the prereg leaves micro-freedom (disclosed):
  * "condition number of the column-standardized J": non-intercept columns
    centered/scaled to unit sd; cond of that 1588x18 matrix (standardizing
    the intercept column is degenerate).  Raw-J cond also reported.
  * VIFs: reported for the 18 non-intercept columns of J (VIF of an
    intercept is undefined); each column regressed on all others incl.
    the intercept.
  * DS-5 5-fold CV: RNG-free deterministic folds, fold k = event positions
    with index % 5 == k (V-5 forbids RNG in decision-bearing reads and no
    CV fold recipe exists in A2's committed script).
  * DS-5 controls-only partial associations: projections fitted on
    controls; correlation evaluated on all 1588 events (primary) and on
    the 1203 controls (also reported).
  * DS-3 identity: asserted to < 1e-13 absolute ("machine precision" for
    385-pair means of O(1e-2) objects); the exact gap is reported.
  * File names: the prereg (section 2) names b1_completion_weight_functional.py
    -> b1_results.json; those names govern.

FREE READ: existing CSVs + frozeng ball emits + staged pruned catalogues;
no likelihood evaluations, no cluster jobs.  New files only.
Output: b1_results.json (deterministic apart from the "timestamp" field).

Run: cd /home/jasper/Repositories/MasterThesisCode && uv run python \
  results/campaign51_20260728/realistic_20260729/m2_residual_owner/b1_completion_weight_functional.py
"""

import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd
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
OUT = f"{HERE}/b1_results.json"
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

# ---- locked registration constants (prereg sections 2-5; NOT adjustable) ----
N_PERM = 20000
SEED_SIGNFLIP = 20260808
SEED_CLUSTER = 20260808
PRIMARY_DEGREE = 3
REGISTERED = {
    "prereg": "PREREGISTRATION_B_COMPLETION_WEIGHT_FUNCTIONAL.md",
    "prereg_commit": "36fe7800130d538be146e5e1696629b7a416977a",
    "ds1_cross_r2_gate": 0.80,
    "ds1_condition_gate": 1e8,
    "collapse_se_multiple": 1.0,
    "alpha": 0.0455,
    "repro_band": [0.70, 1.30],
    "n_perm": N_PERM,
    "seed_signflip": SEED_SIGNFLIP,
    "seed_cluster": SEED_CLUSTER,
}

# ---- bitwise anchors (prereg section 4, V-1..V-4) ----
M2_TOTALS = {"iiib": 0.022252643015992925, "joint_r1": 0.020697491999731973}
A2_PRIMARY_R2 = {"c_pure": 0.8832406614871592, "c_gfrac": 0.8747947939465979}
A2_E_D = {"c_pure": 0.008340732036016641, "c_gfrac": 0.008352697414993901}
A2_E_D_CLUSTER_SE = {"c_pure": 0.0029160903955559583, "c_gfrac": 0.0029146831062962105}
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
M_SOURCE_FRAME_MIN, M_SOURCE_FRAME_MAX, Z_MAX = 1e4, 1e7, 1.5

t0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


def md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


git_commit = subprocess.run(
    ["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True, text=True, check=True
).stdout.strip()

# ---------------------------------------------------------------- V-5: input md5s
log("V-5: input md5 asserts")
for path, exp in EXPECTED_MD5.items():
    got = md5(path)
    assert got == exp, ("md5 mismatch", path, got, exp)

# ---------------------------------------------------------------- C-4 census (verbatim M-2/A2)
log("census (verbatim M-2/A2)")
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
overlap = np.zeros(n, dtype=bool)
overlap[ii[win]] = True
overlap[jj[win]] = True
assert int(win.sum()) == 279
assert int(overlap.sum()) == 385
assert int(sky.sum()) == 1620

log10_r = np.log10(r)
log10_dL = np.log10(dl)
log10_rel = np.log10(s_dl / dl)

# ---------------------------------------------------------------- helpers (verbatim A2/D-1)


def cluster_se(diffs: np.ndarray, clusters: np.ndarray) -> float:
    """Cluster-robust SE of the mean paired diff (clusters = shared-control ids)."""
    m = diffs.mean()
    nn_ = len(diffs)
    tot = 0.0
    for c in np.unique(clusters):
        tot += float((diffs[clusters == c] - m).sum()) ** 2
    return float(np.sqrt(tot) / nn_)


def r2(y: np.ndarray, yhat: np.ndarray) -> float:
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot


def poly_design(x1: np.ndarray, x2: np.ndarray, degree: int) -> np.ndarray:
    cols = [x1**i * x2**j for i in range(degree + 1) for j in range(degree + 1 - i)]
    return np.stack(cols, axis=1)


def poly_labels(a: str, b: str, degree: int) -> list:
    return [f"{a}^{i}*{b}^{j}" for i in range(degree + 1) for j in range(degree + 1 - i)]


def poly_fit_predict(
    x1: np.ndarray,
    x2: np.ndarray,
    y: np.ndarray,
    degree: int,
    fit_mask: np.ndarray | None = None,
) -> np.ndarray:
    """OLS polynomial fit (optionally on a subsample), predicted over all rows."""
    X = poly_design(x1, x2, degree)
    if fit_mask is None:
        fit_mask = np.ones(len(y), dtype=bool)
    beta, *_ = np.linalg.lstsq(X[fit_mask], y[fit_mask], rcond=None)
    return X @ beta


def design_fit_predict(X: np.ndarray, y: np.ndarray, fit_mask: np.ndarray | None = None):
    if fit_mask is None:
        fit_mask = np.ones(len(y), dtype=bool)
    beta, *_ = np.linalg.lstsq(X[fit_mask], y[fit_mask], rcond=None)
    return X @ beta


def log_mean(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Logarithmic mean L(x,y) = (x-y)/ln(x/y); L(x,x) = x. Requires x,y > 0."""
    out = np.empty_like(x)
    close = np.isclose(x, y, rtol=1e-12, atol=0.0)
    out[close] = 0.5 * (x[close] + y[close])
    nc = ~close
    out[nc] = (x[nc] - y[nc]) / (np.log(x[nc]) - np.log(y[nc]))
    return out


def signflip_p(diffs: np.ndarray, seed: int = SEED_SIGNFLIP) -> float:
    """Two-sided sign-flip permutation p, vectorized, fresh rng per test (D-1 verbatim)."""
    rng = np.random.default_rng(seed)
    obs = abs(diffs.mean())
    m = len(diffs)
    signs = rng.choice([-1.0, 1.0], size=(N_PERM, m))
    stats = np.abs((signs * diffs[None, :]).mean(axis=1))
    return float((int((stats >= obs).sum()) + 1) / (N_PERM + 1))


def cluster_signflip_p(diffs: np.ndarray, clusters: np.ndarray, seed: int = SEED_CLUSTER) -> float:
    """Two-sided sign-flip p flipping all pairs sharing a control together (D-1 verbatim)."""
    rng = np.random.default_rng(seed)
    obs = abs(diffs.mean())
    uniq, inv = np.unique(clusters, return_inverse=True)
    signs = rng.choice([-1.0, 1.0], size=(N_PERM, len(uniq)))
    stats = np.abs((signs[:, inv] * diffs[None, :]).mean(axis=1))
    return float((int((stats >= obs).sum()) + 1) / (N_PERM + 1))


# ---------------------------------------------------------------- load venues, V-3 identity
log("loading venues + V-3 identity asserts")
frames: dict = {}
ev_ref = None
for venue, path in VENUES.items():
    el = pd.read_csv(path)
    assert len(el) == 65108, (venue, len(el))
    a = el[np.isclose(el.h, 0.60)].set_index("event_idx").sort_index()
    b = el[np.isclose(el.h, 0.73)].set_index("event_idx").sort_index()
    ev = np.array(sorted(set(a.index) & set(b.index)))
    assert len(ev) == 1588, (venue, len(ev))
    if ev_ref is None:
        ev_ref = ev
    else:
        assert (ev == ev_ref).all(), "venues disagree on the 1588-event set"
    frames[venue] = (a.loc[ev], b.loc[ev])
ev = ev_ref
assert ev is not None

identity: dict = {}
COMPLETION_COLS = ("L_comp", "B_num", "B_num_wbh", "g_frac")
for k, h0 in ((0, "h0.60"), (1, "h0.73")):
    for col in COMPLETION_COLS:
        xi = frames["iiib"][k][col].to_numpy()
        xj = frames["joint_r1"][k][col].to_numpy()
        bitwise = bool((xi == xj).all())
        identity[f"{col}_{h0}"] = {"bitwise_identical": bitwise}
        assert bitwise, (col, h0, "completion-leg column NOT bit-identical across venues")

# ---------------------------------------------------------------- chords (venue-identical)
a_i, b_i = frames["iiib"]
Lc60 = a_i.L_comp.to_numpy()
Lc73 = b_i.L_comp.to_numpy()
g60 = a_i.g_frac.to_numpy()
g73 = b_i.g_frac.to_numpy()
assert (Lc60 > 0).all() and (Lc73 > 0).all() and (g60 > 0).all() and (g73 > 0).all()
c_pure = np.log(Lc60) - np.log(Lc73)
c_gfrac = c_pure + (np.log(g60) - np.log(g73))
CHORDS = {"c_pure": c_pure, "c_gfrac": c_gfrac}

x1 = log10_dL[ev]
x2 = log10_rel[ev]
ov_mask = overlap[ev]
ct_mask = ~ov_mask

# ---------------------------------------------------------------- matching (verbatim M-2/A2)
log("matching (verbatim M-2/A2 385 pairs)")
cov_mat = np.stack([log10_r, snr], axis=1)
z_std = (cov_mat[ev] - cov_mat[ev].mean(axis=0)) / cov_mat[ev].std(axis=0, ddof=1)
z_ov, z_ct = z_std[ov_mask], z_std[ct_mask]
dist2 = ((z_ov[:, None, :] - z_ct[None, :, :]) ** 2).sum(axis=2)
nn = dist2.argmin(axis=1)
ov_idx = np.where(ov_mask)[0]
ct_idx = np.where(ct_mask)[0]
mct_idx = ct_idx[nn]
assert len(ov_idx) == 385 and len(ct_idx) == 1203
n_clusters = int(len(np.unique(nn)))
assert n_clusters == 234, n_clusters  # 234 control clusters (prereg statistic spec)


def matched_read(vals: np.ndarray) -> dict:
    diffs = vals[ov_idx] - vals[mct_idx]
    return {
        "mean_paired_diff": float(diffs.mean()),
        "paired_se": float(diffs.std(ddof=1) / np.sqrt(len(diffs))),
        "cluster_robust_se": cluster_se(diffs, nn),
    }


def matched_read_p(vals: np.ndarray) -> dict:
    """matched_read + locked-seed p-values (fresh rng per test, D-1 machinery)."""
    diffs = vals[ov_idx] - vals[mct_idx]
    out = matched_read(vals)
    out["signflip_p"] = signflip_p(diffs)
    out["cluster_signflip_p"] = cluster_signflip_p(diffs, nn)
    return out


# ---------------------------------------------------------------- V-1: M-2 totals
log("V-1: M-2 anchor reproduction")
v1_out: dict = {}
for venue, (a, b) in frames.items():
    F60 = a.combined_with_bh.to_numpy()
    F73 = b.combined_with_bh.to_numpy()
    chord_csv = np.log(F60) - np.log(F73)
    obs_total = matched_read(chord_csv)
    assert obs_total["mean_paired_diff"] == M2_TOTALS[venue], (venue, obs_total)
    v1_out[venue] = {"matched_2d_total": obs_total["mean_paired_diff"], "anchor": M2_TOTALS[venue]}

# ---------------------------------------------------------------- V-2: A2 anchors (D fits)
log("V-2: A2 anchor reproduction (D design fits)")
D = poly_design(x1, x2, PRIMARY_DEGREE)
D_LABELS = poly_labels("x1", "x2", PRIMARY_DEGREE)
fits_D: dict = {}
fits_D_ct: dict = {}
E_D: dict = {}
for name, y in CHORDS.items():
    yhat = poly_fit_predict(x1, x2, y, PRIMARY_DEGREE)
    fits_D[name] = yhat
    got_r2 = r2(y, yhat)
    assert got_r2 == A2_PRIMARY_R2[name], (name, got_r2)
    fits_D_ct[name] = poly_fit_predict(x1, x2, y, PRIMARY_DEGREE, fit_mask=ct_mask)
    ed = matched_read(y - yhat)
    assert ed["mean_paired_diff"] == A2_E_D[name], (name, ed)
    assert ed["cluster_robust_se"] == A2_E_D_CLUSTER_SE[name], (name, ed)
    E_D[name] = matched_read_p(y - yhat)

# V-2 chain-ratio anchors + DS-4 prep (A2 venue loop verbatim)
venue_chain: dict = {}
for venue, (a, b) in frames.items():
    for frame in (a, b):
        for c in ("w_G", "alpha_G_phi", "r_Malm", "D_tilde_phi"):
            assert frame[c].nunique() == 1, (venue, c)
    wt60, wt73 = float(a.w_G.iloc[0]), float(b.w_G.iloc[0])
    dln_w1m = float(np.log1p(-wt60) - np.log1p(-wt73))
    assert wt73 == A2_WT_073[venue], (venue, wt73)
    assert dln_w1m == A2_DLN_1MWT[venue], (venue, dln_w1m)

    Lcat60 = a.L_cat_with_bh.to_numpy()
    Lcat73 = b.L_cat_with_bh.to_numpy()
    B60 = (1 - wt60) * g60 * Lc60
    B73 = (1 - wt73) * g73 * Lc73
    A60 = wt60 * Lcat60
    A73 = wt73 * Lcat73
    F60 = a.combined_with_bh.to_numpy()
    F73 = b.combined_with_bh.to_numpy()
    chord_csv = np.log(F60) - np.log(F73)
    LF = log_mean(A60 + B60, A73 + B73)
    SB = log_mean(B60, B73) / LF
    T_legB = SB * (dln_w1m + c_gfrac)
    T_legA = (np.log(A60 + B60) - np.log(A73 + B73)) - T_legB

    obs_total = matched_read(chord_csv)
    obs_TlegA = matched_read(T_legA)
    assert obs_TlegA["mean_paired_diff"] == A2_OBS_TLEGA[venue], (venue, obs_TlegA)

    # D-fit chain ratio (V-2 anchor)
    TlegB_hat_D = SB * (dln_w1m + fits_D["c_gfrac"])
    pred_total_D = obs_TlegA["mean_paired_diff"] + matched_read(TlegB_hat_D)["mean_paired_diff"]
    ratio_D = pred_total_D / obs_total["mean_paired_diff"]
    assert ratio_D == A2_CHAIN_RATIO_D[venue], (venue, ratio_D)

    # zero-set + S_A for DS-6 (D-1 LMDI verbatim, 2D channel)
    z60, z73 = Lcat60 == 0.0, Lcat73 == 0.0
    assert (z60 == z73).all(), (venue, "L_cat zero set not h-stable")
    nz = ~z60
    SA = np.zeros_like(LF)
    SA[nz] = log_mean(A60[nz], A73[nz]) / LF[nz]

    venue_chain[venue] = {
        "wt60": wt60,
        "wt73": wt73,
        "dln_w1m": dln_w1m,
        "SB": SB,
        "SA": SA,
        "nz": nz,
        "obs_total": obs_total,
        "obs_TlegA": obs_TlegA,
        "ratio_D_all_events": ratio_D,
    }
log("V-1/V-2/V-3 PASS (bitwise)")

# ---------------------------------------------------------------- ball covariates (D-2 verbatim)
log("building ball covariates (D-2 builder verbatim)")


def load_pruned_zm(path: str) -> tuple:
    """Bit-faithful pruned+reset (z, M_bh) columns (M-4 / D-2 verbatim)."""
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


ball_cov: dict = {}  # venue -> (1590, 4): n2d, n1d, W2d, W1d (raw)
ball_meta: dict = {}
for venue in VENUES:
    log(f"  ball covariates for {venue}")
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
    assert not np.isnan(arr[ev]).any(), (venue, "ball covariates missing for evaluated events")
    ball_cov[venue] = arr
    ball_meta[venue] = {
        "ball_json_md5": EXPECTED_MD5[BALLS[venue]],
        "n_events_missing_of_1590": n_missing,
        "pruned_catalogue_rows": n_cat,
    }
    del ball, gl, ag, zz, mm, w_all

# density covariates over the 1588 events (prereg z1..z4, D-2 order)
ZCOV: dict = {}
for venue in VENUES:
    barr = ball_cov[venue]
    ZCOV[venue] = {
        "z1": np.log10(1.0 + barr[ev, 0]),  # log10_n_ball_2d
        "z2": np.log10(1.0 + barr[ev, 1]),  # log10_n_ball_1d
        "z3": np.log10(1.0 + barr[ev, 2]),  # log10_W_pop_2d
        "z4": np.log10(1.0 + barr[ev, 3]),  # log10_W_pop_1d
    }

# ---------------------------------------------------------------- V-4: D-2 m2-rung fidelity
log("V-4: D-2 m2-rung reproduction (density-covariate fidelity)")
v4_out: dict = {}
for venue, (a, b) in frames.items():
    C3 = np.stack([log10_r, snr, np.log10(1.0 + ball_cov[venue][:, 0])], axis=1)
    C = C3[ev]
    zst = (C - C.mean(axis=0)) / C.std(axis=0, ddof=1)
    z_ov3, z_ct3 = zst[ov_mask], zst[ct_mask]
    d2m = ((z_ov3[:, None, :] - z_ct3[None, :, :]) ** 2).sum(axis=2)
    nn3 = d2m.argmin(axis=1)
    ov_ev3 = ev[ov_mask]
    ct_ev3 = ev[ct_mask]
    mct_ev3 = ct_ev3[nn3]
    chord = pd.Series(
        np.log(a.combined_with_bh.to_numpy()) - np.log(b.combined_with_bh.to_numpy()), index=ev
    )
    eff = float((chord.loc[ov_ev3].to_numpy() - chord.loc[mct_ev3].to_numpy()).mean())
    assert eff == D2_M2_RUNG[venue], (venue, eff)
    v4_out[venue] = {"m2_rung_effect_2d": eff, "anchor": D2_M2_RUNG[venue]}
log("V-4 PASS (bitwise)")

# ================================================================ designs P, J per venue
P_LABELS = poly_labels("z1", "z3", PRIMARY_DEGREE)
J_LABELS = D_LABELS + P_LABELS[1:]
r_ev = log10_r[ev]
R_COLS = np.stack([r_ev, r_ev**2, r_ev**3], axis=1)

P_of: dict = {}
J_of: dict = {}
for venue in VENUES:
    zc = ZCOV[venue]
    P = poly_design(zc["z1"], zc["z3"], PRIMARY_DEGREE)
    P_of[venue] = P
    J_of[venue] = np.concatenate([D, P[:, 1:]], axis=1)
    assert J_of[venue].shape == (1588, 19)

# ================================================================ DS-1: decorrelation gate
log("DS-1: family-decorrelation gate")
ds1: dict = {}
for venue in VENUES:
    zc = ZCOV[venue]
    cross_r2: dict = {}
    for zname in ("z1", "z2", "z3", "z4"):
        zhat = design_fit_predict(D, zc[zname])
        cross_r2[zname] = r2(zc[zname], zhat)
    corr: dict = {}
    for zname in ("z1", "z2", "z3", "z4"):
        corr[zname] = {}
        for xname, xarr in (("x1_log10_dL", x1), ("x2_log10_rel_dL_err", x2)):
            pr = pearsonr(zc[zname], xarr)
            sr = spearmanr(zc[zname], xarr)
            corr[zname][xname] = {
                "pearson": float(pr.statistic),
                "spearman": float(sr.statistic),
            }
    J = J_of[venue]
    # VIFs of the 18 non-intercept columns (each on all others incl. intercept)
    vif: dict = {}
    for kcol in range(1, J.shape[1]):
        others = np.delete(J, kcol, axis=1)
        yk = J[:, kcol]
        yhat = design_fit_predict(others, yk)
        r2k = r2(yk, yhat)
        vif[J_LABELS[kcol]] = float(1.0 / (1.0 - r2k)) if r2k < 1.0 else float("inf")
    Jn = J[:, 1:]
    Jn_std = (Jn - Jn.mean(axis=0)) / Jn.std(axis=0, ddof=1)
    cond_std = float(np.linalg.cond(Jn_std))
    cond_raw = float(np.linalg.cond(J))
    gate_pass = bool(
        (
            (cross_r2["z1"] <= REGISTERED["ds1_cross_r2_gate"])
            or (cross_r2["z3"] <= REGISTERED["ds1_cross_r2_gate"])
        )
        and (cond_std < REGISTERED["ds1_condition_gate"])
    )
    ds1[venue] = {
        "cross_family_r2_on_D": cross_r2,
        "pairwise_correlations_vs_dL_family": corr,
        "vif_J_non_intercept": vif,
        "condition_number_standardized_J_non_intercept": cond_std,
        "condition_number_raw_J": cond_raw,
        "gate_pass_measured": gate_pass,
        "gate_definition": "min(cross_R2[z1], cross_R2[z3]) <= 0.80 AND cond(std J) < 1e8",
    }

# ================================================================ fits on P and J
log("fits on P/J (all-events + controls-only)")
fits_J: dict = {}
fits_J_ct: dict = {}
fits_P: dict = {}
r2_table: dict = {}
for venue in VENUES:
    fits_J[venue] = {}
    fits_J_ct[venue] = {}
    fits_P[venue] = {}
    r2_table[venue] = {}
    for name, y in CHORDS.items():
        yhat_J = design_fit_predict(J_of[venue], y)
        yhat_J_ct = design_fit_predict(J_of[venue], y, fit_mask=ct_mask)
        yhat_P = design_fit_predict(P_of[venue], y)
        fits_J[venue][name] = yhat_J
        fits_J_ct[venue][name] = yhat_J_ct
        fits_P[venue][name] = yhat_P
        r2_D_ = r2(y, fits_D[name])
        r2_P_ = r2(y, yhat_P)
        r2_J_ = r2(y, yhat_J)
        r2_table[venue][name] = {
            "r2_D": r2_D_,
            "r2_P": r2_P_,
            "r2_J": r2_J_,
            "semi_partial_dR2_P_given_D": r2_J_ - r2_D_,
            "semi_partial_dR2_D_given_P": r2_J_ - r2_P_,
        }


def excess_reads(chat_D_map: dict, chat_J_map: dict) -> dict:
    """DS-2/DS-3 statistics for one (fit-variant) pair of prediction maps."""
    out: dict = {}
    for name, y in CHORDS.items():
        cd = chat_D_map[name]
        cj = chat_J_map[name]
        e_d = matched_read(y - cd)["mean_paired_diff"]
        ej = matched_read_p(y - cj)
        ar = matched_read_p(cj - cd)
        identity_gap = e_d - (ar["mean_paired_diff"] + ej["mean_paired_diff"])
        out[name] = {
            "E_D_mean_paired_diff": e_d,
            "E_J": ej,
            "A_rho": ar,
            "A_rho_share_of_E_D": ar["mean_paired_diff"] / e_d,
            "identity_gap_E_D_minus_A_rho_minus_E_J": identity_gap,
        }
    return out


# ================================================================ DS-2 + DS-3
log("DS-2/DS-3: fixed-d_L excess re-estimation + additive attribution")
ds2_ds3: dict = {}
for venue in VENUES:
    prim = excess_reads(fits_D, fits_J[venue])
    ctrl = excess_reads(fits_D_ct, fits_J_ct[venue])
    for name in CHORDS:
        gap = prim[name]["identity_gap_E_D_minus_A_rho_minus_E_J"]
        assert abs(gap) < 1e-13, (venue, name, "DS-3 identity failure", gap)
    ds2_ds3[venue] = {"all_events_fit_primary": prim, "controls_only_fit": ctrl}

# ================================================================ DS-4: chain ratio with J
log("DS-4: stratum-composition chain ratio rho_J")
ds4: dict = {}
for venue in VENUES:
    vc = venue_chain[venue]
    entry: dict = {
        "weight_constants": {
            "wt_060": vc["wt60"],
            "wt_073": vc["wt73"],
            "dln_one_minus_wt": vc["dln_w1m"],
        },
        "observed_total_2d": vc["obs_total"]["mean_paired_diff"],
        "observed_T_legA": vc["obs_TlegA"]["mean_paired_diff"],
        "rho_D_all_events_anchor": vc["ratio_D_all_events"],
    }
    for label, fj in (("all_events_fit", fits_J[venue]), ("controls_only_fit", fits_J_ct[venue])):
        TlegB_hat = vc["SB"] * (vc["dln_w1m"] + fj["c_gfrac"])
        pred_TlegB = matched_read(TlegB_hat)
        pred_total = vc["obs_TlegA"]["mean_paired_diff"] + pred_TlegB["mean_paired_diff"]
        entry[f"rho_J_{label}"] = {
            "predicted_T_legB": pred_TlegB,
            "predicted_total_2d": pred_total,
            "ratio_pred_over_obs": pred_total / vc["obs_total"]["mean_paired_diff"],
        }
    ds4[venue] = entry

# ================================================================ DS-5: radius overlay + coherence
log("DS-5: radius overlay + sign coherence")
D_r = np.concatenate([D, R_COLS], axis=1)
ds5: dict = {}
FOLDS = np.arange(1588) % 5  # deterministic, RNG-free (disclosed convention)
for venue in VENUES:
    J_r = np.concatenate([J_of[venue], R_COLS], axis=1)
    entry: dict = {"designs": {"D_plus_r_cols": D_r.shape[1], "J_plus_r_cols": J_r.shape[1]}}
    overlay: dict = {}
    for name, y in CHORDS.items():
        cd = design_fit_predict(D_r, y)
        cj = design_fit_predict(J_r, y)
        ar = matched_read_p(cj - cd)
        ej = matched_read_p(y - cj)
        ed = matched_read(y - cd)
        overlay[name] = {
            "A_rho_plus_r": ar,
            "E_J_plus_r": ej,
            "E_D_plus_r_mean_paired_diff": ed["mean_paired_diff"],
            "sign_A_rho_plus_r": float(np.sign(ar["mean_paired_diff"])),
        }
    entry["radius_overlay"] = overlay

    # partial-association sign coherence: chord vs z1/z3, residualized on D
    zc = ZCOV[venue]
    coher: dict = {}
    for name, y in CHORDS.items():
        cob: dict = {}
        for zname in ("z1", "z3"):
            zarr = zc[zname]
            # all-events residualization
            ry = y - design_fit_predict(D, y)
            rz = zarr - design_fit_predict(D, zarr)
            pr_all = float(pearsonr(ry, rz).statistic)
            # controls-only-fit residualization (projections fit on controls)
            ry_c = y - design_fit_predict(D, y, fit_mask=ct_mask)
            rz_c = zarr - design_fit_predict(D, zarr, fit_mask=ct_mask)
            pr_ctfit_all = float(pearsonr(ry_c, rz_c).statistic)
            pr_ctfit_controls = float(pearsonr(ry_c[ct_mask], rz_c[ct_mask]).statistic)
            # 5-fold CV: fit residualizations on complement, evaluate out-of-fold
            fold_signs = []
            fold_vals = []
            for kf in range(5):
                te = FOLDS == kf
                tr = ~te
                ry_f = y - design_fit_predict(D, y, fit_mask=tr)
                rz_f = zarr - design_fit_predict(D, zarr, fit_mask=tr)
                pv = float(pearsonr(ry_f[te], rz_f[te]).statistic)
                fold_vals.append(pv)
                fold_signs.append(float(np.sign(pv)))
            cob[zname] = {
                "partial_corr_all_events": pr_all,
                "partial_corr_controls_only_fit_eval_all": pr_ctfit_all,
                "partial_corr_controls_only_fit_eval_controls": pr_ctfit_controls,
                "cv_fold_partial_corrs": fold_vals,
                "cv_fold_signs": fold_signs,
                "sign_all_events": float(np.sign(pr_all)),
            }
        cob["z1_z3_same_sign_all_events"] = bool(
            np.sign(cob["z1"]["partial_corr_all_events"])
            == np.sign(cob["z3"]["partial_corr_all_events"])
        )
        coher[name] = cob
    entry["partial_association_sign_coherence"] = coher
    ds5[venue] = entry

# ================================================================ DS-6: weight channel (secondary)
log("DS-6: weight channel y3 = log10(S_A)")
ds6: dict = {}
for venue in VENUES:
    vc = venue_chain[venue]
    SA = vc["SA"]
    pos = SA > 0
    n_zero = int((~pos).sum())
    y3 = np.log10(SA[pos])
    Ds = D[pos]
    Ps = P_of[venue][pos]
    Js = J_of[venue][pos]
    yD = design_fit_predict(Ds, y3)
    yP = design_fit_predict(Ps, y3)
    yJ = design_fit_predict(Js, y3)
    r2D, r2P, r2J = r2(y3, yD), r2(y3, yP), r2(y3, yJ)
    zc = ZCOV[venue]
    partials: dict = {}
    for zname in ("z1", "z3"):
        zarr = zc[zname][pos]
        rz = zarr - design_fit_predict(Ds, zarr)
        partials[f"y3_vs_{zname}_given_D"] = float(pearsonr(y3 - yD, rz).statistic)
    for xname, xarr in (("x1", x1), ("x2", x2)):
        xa = xarr[pos]
        rx = xa - design_fit_predict(Ps, xa)
        partials[f"y3_vs_{xname}_given_P"] = float(pearsonr(y3 - yP, rx).statistic)
    ds6[venue] = {
        "n_SA_zero_structural": n_zero,
        "n_SA_positive_fitted": int(pos.sum()),
        "n_SA_positive_overlap": int(pos[ov_idx].sum()),
        "r2_D": r2D,
        "r2_P": r2P,
        "r2_J": r2J,
        "semi_partial_dR2_P_given_D": r2J - r2D,
        "semi_partial_dR2_D_given_P": r2J - r2P,
        "partial_association": partials,
        "partial_association_signs": {k: float(np.sign(v)) for k, v in partials.items()},
    }

# ================================================================ R-1 / R-2 / R-3 (robustness)
log("R-1/R-2/R-3 robustness reads")


def multipoly_design_deg2(vs: list) -> np.ndarray:
    """Total-degree-2 polynomial in len(vs) variables; deterministic exponent order."""
    nvars = len(vs)
    cols = []
    exps = []
    for e in np.ndindex(*([3] * nvars)):
        if sum(e) <= 2:
            exps.append(e)
    exps.sort()  # deterministic lexicographic order
    for e in exps:
        col = np.ones_like(vs[0])
        for vv, p in zip(vs, e):
            col = col * vv**p
        cols.append(col)
    return np.stack(cols, axis=1)


robustness: dict = {"R1": {}, "R2": {}, "R3": {}}
for venue in VENUES:
    zc = ZCOV[venue]
    # R-1: 4-variable total-degree-2 density design replacing P (15 cols)
    P_r1 = multipoly_design_deg2([zc["z1"], zc["z2"], zc["z3"], zc["z4"]])
    assert P_r1.shape[1] == 15, P_r1.shape
    J_r1 = np.concatenate([D, P_r1[:, 1:]], axis=1)
    fj = {name: design_fit_predict(J_r1, y) for name, y in CHORDS.items()}
    robustness["R1"][venue] = excess_reads(fits_D, fj)
    robustness["R1"][venue]["design_cols"] = {"P": 15, "J": int(J_r1.shape[1])}
    # R-2: primary P extended with [z2, z4, z2^2, z4^2]
    ext = np.stack([zc["z2"], zc["z4"], zc["z2"] ** 2, zc["z4"] ** 2], axis=1)
    P_r2 = np.concatenate([P_of[venue], ext], axis=1)
    J_r2 = np.concatenate([D, P_r2[:, 1:]], axis=1)
    fj2 = {name: design_fit_predict(J_r2, y) for name, y in CHORDS.items()}
    robustness["R2"][venue] = excess_reads(fits_D, fj2)
    robustness["R2"][venue]["design_cols"] = {"P": int(P_r2.shape[1]), "J": int(J_r2.shape[1])}
    # R-3: density-design degree ladder (R^2 of the chord objects on P_deg)
    ladder: dict = {}
    for deg in (1, 2, 3, 4):
        Pd = poly_design(zc["z1"], zc["z3"], deg)
        ladder[f"degree_{deg}"] = {
            name: r2(y, design_fit_predict(Pd, y)) for name, y in CHORDS.items()
        }
    robustness["R3"][venue] = ladder

# ================================================================ assemble + write
results = {
    "read": "Instrument B: completion-weight functional read (pre-registered)",
    "prereg": {
        "file": "PREREGISTRATION_B_COMPLETION_WEIGHT_FUNCTIONAL.md",
        "registered_commit": REGISTERED["prereg_commit"],
        "run_git_commit": git_commit,
    },
    "date": "2026-08-08",
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "registered_constants": REGISTERED,
    "chord_definition": "ln X(h=0.60) - ln X(h=0.73) per event; positive = prefers low h",
    "objects": {
        "c_pure": "dln L_comp (T_Lcomp carrier chord; venue-identical, asserted)",
        "c_gfrac": "dln(g_frac * L_comp) (2D completion-leg product chord)",
        "y3": "log10(S_A), S_A = LMDI catalogue-leg share (per venue, S_A>0 subset)",
    },
    "designs": {
        "D": f"total-degree-3 poly in (x1=log10_dL, x2=log10_rel_dL_err), {D.shape[1]} cols "
        "(a2 poly_design byte-compatible)",
        "P": "total-degree-3 poly in (z1=log10_n_ball_2d, z3=log10_W_pop_2d), 10 cols, per venue",
        "J": "column union of D and P (single intercept), 19 cols",
        "column_labels": {"D": D_LABELS, "J": J_LABELS},
    },
    "inputs": {
        "md5": {p: m for p, m in EXPECTED_MD5.items()},
        "catalogues": CATS,
        "ball_meta": ball_meta,
        "census": {"sky_pairs": 1620, "window_pairs": 279, "overlap_events": 385},
        "n_control_clusters": n_clusters,
        "inherited_assumption": "event_idx == CRB row index (flagged M-2 assumption, inherited)",
    },
    "validity": {
        "V1_m2_totals": v1_out,
        "V2_a2_anchors": {
            "primary_r2": {k: r2(CHORDS[k], fits_D[k]) for k in CHORDS},
            "E_D": {
                k: {
                    "mean_paired_diff": E_D[k]["mean_paired_diff"],
                    "cluster_robust_se": E_D[k]["cluster_robust_se"],
                }
                for k in CHORDS
            },
            "chain_ratio_D_all_events": {v: venue_chain[v]["ratio_D_all_events"] for v in VENUES},
        },
        "V3_venue_identity": identity,
        "V4_d2_m2_rung": v4_out,
        "V5": {
            "input_md5_asserts": "PASS",
            "census_asserts": "PASS (1620/279/385, 65108 rows/CSV, 1588 events)",
            "determinism": "double-run byte-identity apart from 'timestamp' checked externally "
            "by the run agent (recorded in the run report)",
        },
        "all_hard_asserts": "PASS (script reaches output only if every assert holds)",
    },
    "ds1_family_decorrelation_gate": ds1,
    "ds2_ds3_excess_and_attribution": ds2_ds3,
    "E_D_reference_with_p": E_D,
    "chord_r2_semi_partials": r2_table,
    "ds4_chain_ratio": ds4,
    "ds5_radius_overlay_and_coherence": ds5,
    "ds6_weight_channel": ds6,
    "robustness": robustness,
    "notes": [
        "No branch scoring performed here: the Close agent scores against the locked bands.",
        "DS-1 gate_pass_measured is the pre-stated abort diagnostic, reported factually.",
        "Conservative conventions (disclosed in script docstring): standardized-J condition "
        "number computed on centered/scaled non-intercept columns; VIFs on non-intercept "
        "columns; DS-5 CV folds deterministic (position % 5); DS-5 controls-only partial "
        "associations evaluated on all events (primary) and controls; DS-3 identity asserted "
        "to < 1e-13 with the exact gap reported.",
    ],
}

with open(OUT, "w") as f:
    json.dump(results, f, indent=2)
log(f"wrote {OUT}")
print(
    json.dumps(
        {
            "ds1_gate": {v: ds1[v]["gate_pass_measured"] for v in VENUES},
            "ds1_cross_r2": {v: ds1[v]["cross_family_r2_on_D"] for v in VENUES},
            "ds1_cond": {
                v: ds1[v]["condition_number_standardized_J_non_intercept"] for v in VENUES
            },
            "E_J_primary": {
                v: {
                    k: ds2_ds3[v]["all_events_fit_primary"][k]["E_J"]["mean_paired_diff"]
                    for k in CHORDS
                }
                for v in VENUES
            },
            "A_rho_primary": {
                v: {
                    k: ds2_ds3[v]["all_events_fit_primary"][k]["A_rho"]["mean_paired_diff"]
                    for k in CHORDS
                }
                for v in VENUES
            },
            "rho_J": {
                v: ds4[v]["rho_J_all_events_fit"]["ratio_pred_over_obs"] for v in VENUES
            },
        },
        indent=2,
    )
)
