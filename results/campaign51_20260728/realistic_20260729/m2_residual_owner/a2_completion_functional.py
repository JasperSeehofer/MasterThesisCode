"""A2: completion-leg bundle-functional sanity read (stage-1 loophole close, FREE READ).

Executes STAGE1_READOUT_20260807.md section 5 (A)(2): regression/decomposition of the
per-event 2D completion-leg chord against (d_L, sigma_dL) over ALL 1588 events per
venue — not just the C-4 overlap stratum — to test whether the chord is the smooth
deterministic function of the event's d_L posterior that reading (b) of readout
section 4.2 predicts ("deterministic d_L-dependence of the completion-leg h-response
sampled unevenly by the C-4 census").

Objects characterized (chord convention identical to D-1: ln X(h=0.60) - ln X(h=0.73),
positive = prefers low h):

    c_pure  = ln L_comp(0.60)  - ln L_comp(0.73)              (T_Lcomp carrier chord)
    c_gfrac = c_pure + [ln g_frac(0.60) - ln g_frac(0.73)]    (g_frac-composed 2D
              variant: the exact ln-chord of the 2D completion-leg product
              g_frac * L_comp; T_legB = S_B * (dln(1-wt) + c_gfrac) exactly)

Predictors (same construction as d2_confounding_check.py):
    x1 = log10_dL         = log10(luminosity_distance)          [CRB]
    x2 = log10_rel_dL_err = log10(sigma_dL / dL)                [CRB]

PRE-STATED analysis constants (fixed before any fit number was seen; this is a
sanity read, not a prereg — bands are stated here for symmetric reporting):
  - Primary smooth fit: OLS polynomial in (x1, x2), total degree 3 (10 coefficients,
    1588 events). Degrees 1,2,4 and nonparametric 12x12 / 20x20 quantile-binned means
    reported as robustness/upper bound.
  - SMOOTH criterion: primary-fit R^2 >= 0.95 (and binned means confirm the poly is
    not underfitting structure, binned R^2 - poly R^2 < 0.03).
  - REPRODUCTION criterion (the readout's upgrade half): full-chain predicted matched
    2D total residual within ratio band [0.70, 1.30] of the observed total
    (+0.022253 iiib / +0.020697 joint_r1) at BOTH venues. Full chain = observed
    T_legA matched diff + predicted T_legB matched diff, where predicted
    T_legB = S_B * (dln(1-wt) + fhat_gfrac(x1,x2)) replaces ONLY the completion-leg
    chord by its smooth (d_L, rel-err) prediction, keeping the venue's observed
    composition algebra (S_B shares, weight constants, catalogue leg) untouched.
    Carrier-level reproduction (predicted vs observed matched diff of c_pure,
    +0.035424) reported with the same band as supporting evidence.
  - UPGRADE (per readout section 5(A)(2)): met iff SMOOTH holds AND full-chain
    REPRODUCTION holds at both venues. Then "confounding-absorbable" upgrades to
    "EXPLAINED: d_L-dependent completion-leg h-response x stratum d_L composition".

Machinery: verbatim M-2/D-1 census (1620/279/385 asserted) and deterministic 1-NN
matching on standardized (log10 radius chord, SNR) — the SAME 385 pairs as M-2/D-1;
M-2 headline totals asserted bitwise. Venue identity of the completion-leg columns
(L_comp, B_num, B_num_wbh, g_frac) asserted EXACTLY (bitwise) at both anchor h.
No RNG anywhere: every number here is deterministic. Uncertainty on matched diffs is
reported as naive paired SE and cluster-robust SE (clusters = pairs sharing a
control, same clustering as M-2/D-1's cluster sign-flip).

FREE READ: existing CSVs only; no likelihood evaluations, no cluster jobs.
Output: a2_results.json. Rails: new files only, in m2_residual_owner/.
"""

import hashlib
import json
import re
import subprocess

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = "/home/jasper/Repositories/MasterThesisCode"
OUTDIR = f"{ROOT}/results/campaign51_20260728/realistic_20260729/m2_residual_owner"
OUT = f"{OUTDIR}/a2_results.json"
CRB = f"{ROOT}/results/run_20260804_postfix/joint_r1/diagnostics/prepared_cramer_rao_bounds.csv"
VENUES = {
    "iiib": f"{ROOT}/results/run_20260804_postfix/iiib/diagnostics/event_likelihoods.csv",
    "joint_r1": f"{ROOT}/results/run_20260804_postfix/joint_r1/diagnostics/event_likelihoods.csv",
}
M2_TOTALS = {
    "iiib": 0.022252643015992925,
    "joint_r1": 0.020697491999731973,
}
# D-1 committed raw-carrier matched diffs (venue-identical columns -> venue-identical)
D1_RAW_LCOMP_DIFF = 0.035424306275216304
D1_RAW_BNUMWBH_DIFF = 0.03454539606780529

# ---- pre-stated constants (see docstring) ----
PRIMARY_DEGREE = 3
SMOOTH_R2_BAR = 0.95
BINNED_MINUS_POLY_BAR = 0.03
REPRO_BAND = (0.70, 1.30)

# ---------------------------------------------------------------- provenance


def md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


git_commit = subprocess.run(
    ["git", "-C", ROOT, "rev-parse", "HEAD"], capture_output=True, text=True, check=True
).stdout.strip()

# ---------------------------------------------------------------- C-4 census (verbatim M-2/D-1)
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

# ---------------------------------------------------------------- helpers


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


def binned_means_predict(
    x1: np.ndarray, x2: np.ndarray, y: np.ndarray, nb: int
) -> tuple[np.ndarray, int]:
    """Nonparametric nb x nb quantile-binned means; returns (yhat, n_nonempty_cells)."""
    e1 = np.quantile(x1, np.linspace(0, 1, nb + 1))
    e2 = np.quantile(x2, np.linspace(0, 1, nb + 1))
    b1 = np.clip(np.digitize(x1, e1[1:-1]), 0, nb - 1)
    b2 = np.clip(np.digitize(x2, e2[1:-1]), 0, nb - 1)
    cell = b1 * nb + b2
    yhat = np.empty_like(y)
    ncells = 0
    for c in np.unique(cell):
        m = cell == c
        yhat[m] = y[m].mean()
        ncells += 1
    return yhat, ncells


def dec_step(s: str) -> float:
    """Decimal quantization step of a CSV-printed float (10^(exp - n_decimals))."""
    s = s.strip()
    mm = re.match(r"^-?(\d+)(?:\.(\d+))?(?:[eE]([+-]?\d+))?$", s)
    assert mm is not None, s
    dec = len(mm.group(2) or "")
    ex = int(mm.group(3) or 0)
    return 10.0 ** (ex - dec)


# ---------------------------------------------------------------- load venues, venue-identity assert
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

# venue identity: completion-leg carrier columns bitwise-identical across venues at both h
identity: dict = {}
COMPLETION_COLS = ("L_comp", "B_num", "B_num_wbh", "g_frac")
for k, h0 in ((0, "h0.60"), (1, "h0.73")):
    for col in COMPLETION_COLS:
        xi = frames["iiib"][k][col].to_numpy()
        xj = frames["joint_r1"][k][col].to_numpy()
        bitwise = bool((xi == xj).all())
        identity[f"{col}_{h0}"] = {
            "bitwise_identical": bitwise,
            "max_abs_rel_diff": float(np.max(np.abs(xi - xj) / np.maximum(np.abs(xj), 1e-300))),
        }
        assert bitwise, (col, h0, "completion-leg column NOT bit-identical across venues")

# ---------------------------------------------------------------- chords (venue-identical; built once)
a_i, b_i = frames["iiib"]
Lc60 = a_i.L_comp.to_numpy()
Lc73 = b_i.L_comp.to_numpy()
g60 = a_i.g_frac.to_numpy()
g73 = b_i.g_frac.to_numpy()
assert (Lc60 > 0).all() and (Lc73 > 0).all() and (g60 > 0).all() and (g73 > 0).all()
c_pure = np.log(Lc60) - np.log(Lc73)
c_gfrac = c_pure + (np.log(g60) - np.log(g73))

x1 = log10_dL[ev]
x2 = log10_rel[ev]
ov_mask = overlap[ev]
ct_mask = ~ov_mask

# ---------------------------------------------------------------- CSV precision floor
floors: dict = {}
for venue, path in list(VENUES.items())[:1]:  # columns venue-identical; one venue suffices
    els = pd.read_csv(path, dtype=str)
    elf = pd.read_csv(path)
    for col in ("L_comp", "g_frac"):
        rels = {}
        for h0 in (0.60, 0.73):
            pos = np.where(np.isclose(elf.h, h0))[0]
            pos = pos[np.argsort(elf.event_idx.to_numpy()[pos])]
            vals = elf[col].to_numpy()[pos]
            steps = np.array([dec_step(s) for s in els[col].to_numpy()[pos]])
            rels[h0] = steps / np.abs(vals)
        # chord noise std from independent uniform quantization of both endpoints
        floor_per_event = np.sqrt((rels[0.60] ** 2 + rels[0.73] ** 2) / 12.0)
        floors[col] = {
            "median_rel_step_h060": float(np.median(rels[0.60])),
            "median_rel_step_h073": float(np.median(rels[0.73])),
            "chord_floor_rms": float(np.sqrt((floor_per_event**2).mean())),
            "chord_floor_max": float(floor_per_event.max()),
        }
floor_pure = floors["L_comp"]["chord_floor_rms"]
floor_gfrac = float(
    np.sqrt(floors["L_comp"]["chord_floor_rms"] ** 2 + floors["g_frac"]["chord_floor_rms"] ** 2)
)

# ---------------------------------------------------------------- functional characterization
functional: dict = {}
fits_primary: dict = {}
fits_controls_only: dict = {}
for name, y in (("c_pure", c_pure), ("c_gfrac", c_gfrac)):
    entry: dict = {
        "chord_mean": float(y.mean()),
        "chord_std": float(y.std(ddof=1)),
        "chord_min": float(y.min()),
        "chord_max": float(y.max()),
    }
    # polynomial ladder
    poly: dict = {}
    for deg in (1, 2, 3, 4):
        yhat = poly_fit_predict(x1, x2, y, deg)
        poly[f"degree_{deg}"] = {
            "r2": r2(y, yhat),
            "resid_rms": float(np.sqrt(((y - yhat) ** 2).mean())),
        }
        if deg == PRIMARY_DEGREE:
            fits_primary[name] = yhat
    entry["poly_fits_all_events"] = poly
    # controls-only fit (robustness: stratum cannot shape its own prediction)
    yhat_ct = poly_fit_predict(x1, x2, y, PRIMARY_DEGREE, fit_mask=ct_mask)
    fits_controls_only[name] = yhat_ct
    entry["poly_degree3_controls_only_fit"] = {
        "r2_on_controls": r2(y[ct_mask], yhat_ct[ct_mask]),
        "r2_on_overlap_out_of_sample": r2(y[ov_mask], yhat_ct[ov_mask]),
    }
    # nonparametric binned means
    binned: dict = {}
    for nb in (12, 20):
        yhat_b, ncells = binned_means_predict(x1, x2, y, nb)
        binned[f"{nb}x{nb}_quantile_bins"] = {
            "r2": r2(y, yhat_b),
            "resid_rms": float(np.sqrt(((y - yhat_b) ** 2).mean())),
            "n_nonempty_cells": ncells,
        }
    entry["binned_means_all_events"] = binned
    # 1-covariate fits: which variable carries the function?
    for label, xa in (("log10_dL_only", x1), ("log10_rel_err_only", x2)):
        X = np.stack([xa**k for k in range(PRIMARY_DEGREE + 1)], axis=1)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        entry[f"poly_degree3_{label}_r2"] = r2(y, X @ beta)
    # monotonicity / shape
    rho1 = spearmanr(y, x1)
    rho2 = spearmanr(y, x2)
    entry["spearman_vs_log10_dL"] = {"rho": float(rho1.statistic), "p": float(rho1.pvalue)}
    entry["spearman_vs_log10_rel_err"] = {"rho": float(rho2.statistic), "p": float(rho2.pvalue)}
    dec_edges = np.quantile(x1, np.linspace(0, 1, 11))
    db = np.clip(np.digitize(x1, dec_edges[1:-1]), 0, 9)
    prof = [float(y[db == k].mean()) for k in range(10)]
    entry["decile_profile_along_log10_dL"] = {
        "decile_means": prof,
        "decile_edges_log10_dL": [float(v) for v in dec_edges],
        "n_sign_changes_of_increments": int((np.diff(np.sign(np.diff(prof))) != 0).sum()),
        "monotone_increasing": bool(np.all(np.diff(prof) > 0)),
    }
    functional[name] = entry

# residual scatter vs precision floor
for name, floor in (("c_pure", floor_pure), ("c_gfrac", floor_gfrac)):
    resid_rms = functional[name]["poly_fits_all_events"][f"degree_{PRIMARY_DEGREE}"]["resid_rms"]
    functional[name]["residual_vs_csv_floor"] = {
        "primary_fit_resid_rms": resid_rms,
        "csv_chord_floor_rms": floor,
        "ratio_resid_over_floor": resid_rms / floor if floor > 0 else None,
    }

# smoothness verdict (pre-stated bars)
smooth_checks: dict = {}
for name in ("c_pure", "c_gfrac"):
    r2p = functional[name]["poly_fits_all_events"][f"degree_{PRIMARY_DEGREE}"]["r2"]
    r2b = functional[name]["binned_means_all_events"]["12x12_quantile_bins"]["r2"]
    smooth_checks[name] = {
        "primary_r2": r2p,
        "binned_12x12_r2": r2b,
        "r2_bar": SMOOTH_R2_BAR,
        "passes_r2_bar": bool(r2p >= SMOOTH_R2_BAR),
        "binned_minus_poly": r2b - r2p,
        "poly_not_underfitting": bool((r2b - r2p) < BINNED_MINUS_POLY_BAR),
    }
smooth_ok = bool(
    all(c["passes_r2_bar"] and c["poly_not_underfitting"] for c in smooth_checks.values())
)

# ---------------------------------------------------------------- matching (verbatim M-2/D-1)
cov_mat = np.stack([log10_r, snr], axis=1)
z = (cov_mat[ev] - cov_mat[ev].mean(axis=0)) / cov_mat[ev].std(axis=0, ddof=1)
z_ov, z_ct = z[ov_mask], z[ct_mask]
dist2 = ((z_ov[:, None, :] - z_ct[None, :, :]) ** 2).sum(axis=2)
nn = dist2.argmin(axis=1)
ov_idx = np.where(ov_mask)[0]
ct_idx = np.where(ct_mask)[0]
mct_idx = ct_idx[nn]
assert len(ov_idx) == 385 and len(ct_idx) == 1203


def matched_read(vals: np.ndarray) -> dict:
    diffs = vals[ov_idx] - vals[mct_idx]
    return {
        "mean_paired_diff": float(diffs.mean()),
        "paired_se": float(diffs.std(ddof=1) / np.sqrt(len(diffs))),
        "cluster_robust_se": cluster_se(diffs, nn),
    }


def log_mean(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    out = np.empty_like(x)
    close = np.isclose(x, y, rtol=1e-12, atol=0.0)
    out[close] = 0.5 * (x[close] + y[close])
    nc = ~close
    out[nc] = (x[nc] - y[nc]) / (np.log(x[nc]) - np.log(y[nc]))
    return out


# stratum composition in the predictors (context for the composition step)
composition_context = {
    "log10_dL": {
        "overlap_mean": float(x1[ov_idx].mean()),
        "matched_control_mean": float(x1[mct_idx].mean()),
        "matched_mean_paired_diff": float((x1[ov_idx] - x1[mct_idx]).mean()),
    },
    "log10_rel_err": {
        "overlap_mean": float(x2[ov_idx].mean()),
        "matched_control_mean": float(x2[mct_idx].mean()),
        "matched_mean_paired_diff": float((x2[ov_idx] - x2[mct_idx]).mean()),
    },
}

# carrier-level reproduction (venue-independent)
obs_pure = matched_read(c_pure)
assert abs(obs_pure["mean_paired_diff"] - D1_RAW_LCOMP_DIFF) < 1e-12
obs_gfrac = matched_read(c_gfrac)
carrier: dict = {"observed": {"c_pure": obs_pure, "c_gfrac": obs_gfrac}, "predicted": {}}
carrier["predicted"] = {
    fit_label: {
        name: {
            **matched_read(fits[name]),
            "ratio_pred_over_obs": matched_read(fits[name])["mean_paired_diff"]
            / carrier["observed"][name]["mean_paired_diff"],
        }
        for name in ("c_pure", "c_gfrac")
    }
    for fit_label, fits in (
        ("all_events_fit", fits_primary),
        ("controls_only_fit", fits_controls_only),
    )
}
# residual-level: does what the fit misses still carry stratum information?
carrier["fit_residual_matched_diff"] = {
    name: matched_read(y - fits_primary[name])
    for name, y in (("c_pure", c_pure), ("c_gfrac", c_gfrac))
}

# ---------------------------------------------------------------- full-chain composition per venue
venues_out: dict = {}
repro_flags: list = []
for venue, (a, b) in frames.items():
    for frame in (a, b):
        for c in ("w_G", "alpha_G_phi", "r_Malm", "D_tilde_phi"):
            assert frame[c].nunique() == 1, (venue, c)
    wt60, wt73 = float(a.w_G.iloc[0]), float(b.w_G.iloc[0])
    dln_w1m = float(np.log1p(-wt60) - np.log1p(-wt73))

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
    T_legA = (np.log(A60 + B60) - np.log(A73 + B73)) - T_legB  # exact complement

    obs_total = matched_read(chord_csv)
    assert abs(obs_total["mean_paired_diff"] - M2_TOTALS[venue]) < 1e-12, venue
    obs_TlegA = matched_read(T_legA)
    obs_TlegB = matched_read(T_legB)

    ventry: dict = {
        "weight_constants": {"wt_060": wt60, "wt_073": wt73, "dln_one_minus_wt": dln_w1m},
        "observed": {
            "total_2d": obs_total,
            "T_legA": obs_TlegA,
            "T_legB": obs_TlegB,
        },
        "predicted_full_chain": {},
    }
    for fit_label, fits in (
        ("all_events_fit", fits_primary),
        ("controls_only_fit", fits_controls_only),
    ):
        TlegB_hat = SB * (dln_w1m + fits["c_gfrac"])
        pred_TlegB = matched_read(TlegB_hat)
        pred_total = obs_TlegA["mean_paired_diff"] + pred_TlegB["mean_paired_diff"]
        ratio = pred_total / obs_total["mean_paired_diff"]
        in_band = bool(REPRO_BAND[0] <= ratio <= REPRO_BAND[1])
        ventry["predicted_full_chain"][fit_label] = {
            "predicted_T_legB": pred_TlegB,
            "predicted_total_2d": pred_total,
            "observed_total_2d": obs_total["mean_paired_diff"],
            "ratio_pred_over_obs": ratio,
            "abs_gap": pred_total - obs_total["mean_paired_diff"],
            "gap_over_cluster_se": (pred_total - obs_total["mean_paired_diff"])
            / obs_total["cluster_robust_se"],
            "in_repro_band": in_band,
        }
        if fit_label == "all_events_fit":
            repro_flags.append(in_band)
    venues_out[venue] = ventry

repro_ok = bool(all(repro_flags))
upgrade_met = bool(smooth_ok and repro_ok)

# ---------------------------------------------------------------- assemble + write
results = {
    "read": "A2 completion-leg bundle-functional sanity read (stage-1 loophole close)",
    "definition_source": "STAGE1_READOUT_20260807.md section 5 (A)(2)",
    "date": "2026-08-08",
    "git_commit": git_commit,
    "chord_definition": "ln X(h=0.60) - ln X(h=0.73) per event; positive = prefers low h",
    "objects": {
        "c_pure": "dln L_comp (T_Lcomp carrier chord)",
        "c_gfrac": "dln(g_frac * L_comp) (2D completion-leg product chord; "
        "T_legB = S_B*(dln(1-wt) + c_gfrac) exactly)",
    },
    "predictors": {
        "x1": "log10_dL = log10(luminosity_distance) [CRB]",
        "x2": "log10_rel_dL_err = log10(sigma_dL/dL) [CRB]",
    },
    "pre_stated_criteria": {
        "primary_fit": f"OLS total-degree-{PRIMARY_DEGREE} polynomial in (x1,x2), all 1588 events",
        "smooth": f"primary R^2 >= {SMOOTH_R2_BAR} and binned(12x12) R^2 - poly R^2 < "
        f"{BINNED_MINUS_POLY_BAR}",
        "reproduction": f"full-chain predicted matched 2D total within ratio band {REPRO_BAND} "
        "of observed at BOTH venues (all-events fit)",
        "upgrade": "smooth AND reproduction -> 'EXPLAINED: d_L-dependent completion-leg "
        "h-response x stratum d_L composition'",
    },
    "inputs": {
        "crb_csv": {"path": CRB, "md5": md5(CRB), "n_rows": 1590},
        "event_likelihoods": {v: {"path": p, "md5": md5(p)} for v, p in VENUES.items()},
        "census_asserts": {"sky_pairs": 1620, "window_pairs": 279, "overlap_events": 385},
        "inherited_assumption": "event_idx == CRB row index (flagged M-2 assumption, inherited)",
    },
    "venue_identity_check": {
        "note": "completion-leg carrier columns asserted BITWISE identical across venues "
        "at both anchor h; functional analysis therefore done once on shared columns",
        "columns": identity,
    },
    "csv_precision_floor": floors,
    "functional_characterization": functional,
    "smoothness_verdict": {"checks": smooth_checks, "smooth_ok": smooth_ok},
    "stratum_composition_in_predictors": composition_context,
    "carrier_level_reproduction": carrier,
    "full_chain_reproduction_per_venue": venues_out,
    "verdict": {
        "smooth_ok": smooth_ok,
        "reproduction_ok_both_venues": repro_ok,
        "upgrade_met": upgrade_met,
        "upgrade_statement_if_met": "EXPLAINED: d_L-dependent completion-leg h-response "
        "x stratum d_L composition",
    },
}

with open(OUT, "w") as f:
    json.dump(results, f, indent=2)
print(json.dumps(results["smoothness_verdict"], indent=2))
print(json.dumps(results["carrier_level_reproduction"]["predicted"], indent=2))
print(json.dumps({v: venues_out[v]["predicted_full_chain"] for v in venues_out}, indent=2))
print(json.dumps(results["verdict"], indent=2))
