"""ADVERSARIAL ADJUDICATION of loophole reads A1 and A2 (independent implementations, fresh RNG).

Verifier for the two stage-1 loophole-closing free reads:
  A1 (a1_graph_signflip.py / a1_results.json): H-e decisive test — component-level sign-flip
      over C-4 overlap-graph connected components, both venues.
  A2 (a2_completion_functional.py / a2_results.json): completion-leg bundle-functional
      sanity read against (log10 d_L, log10 sigma_dL/d_L).

Independent choices deliberately made here (vs the reads under audit):
  - connected components via scipy.sparse.csgraph, not hand-rolled union-find;
  - permutation statistics via per-component signed sums (signs @ comp_sums), fresh seeds
    {20260808, 31337, 5551212}, N_PERM = 100000 (5x the reads' 20000);
  - BOTH graph readings of "the C-4 overlap graph" are run: (a) the 279 sky+2sigma-d_L
    'win' pairs that define overlap (A1's choice), (b) the full 1620 sky-pair graph named
    in the intake's H-e prose ("linked by the 1620-sky-pair graph") — the harsher clustering;
  - an even harsher merged clustering (overlap-graph components UNIONED with shared-control
    links) as an adversarial extra;
  - the alternative literal reading of prereg item (ii) "re-matching under different RNG
    seeds": stochastic re-matching (random pick among k=3 nearest controls) under 3 seeds;
  - A2 smooth-fit attacked with 5-fold CV, random half-splits, degree ladder 1..6, and
    alternative binnings; reproduction attacked with a control-vs-control placebo and the
    D-1 3-covariate d_L-matched postdiction.

FREE READ: existing CSVs only. Output: adjudicate_a1_a2_results.json (new file). No edits
to any existing claim/prereg/ledger/readout file. No commits.
"""

import hashlib
import itertools
import json

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.stats import spearmanr

ROOT = "/home/jasper/Repositories/MasterThesisCode"
OUTDIR = f"{ROOT}/results/campaign51_20260728/realistic_20260729/m2_residual_owner"
OUT = f"{OUTDIR}/adjudicate_a1_a2_results.json"
CRB = f"{ROOT}/results/run_20260804_postfix/joint_r1/diagnostics/prepared_cramer_rao_bounds.csv"
VENUES = {
    "iiib": f"{ROOT}/results/run_20260804_postfix/iiib/diagnostics/event_likelihoods.csv",
    "joint_r1": f"{ROOT}/results/run_20260804_postfix/joint_r1/diagnostics/event_likelihoods.csv",
}
ALPHA = 0.0455
N_PERM = 100_000
FRESH_SEEDS = [20260808, 31337, 5551212]
REMATCH_SEEDS = [11, 22, 33]
M2_TOTALS_2D = {"iiib": 0.022252643015992925, "joint_r1": 0.020697491999731973}
D1_RAW_LCOMP_DIFF = 0.035424306275216304
D1_DL_MATCHED_TOTALS = {"iiib": 0.0070643091523159875, "joint_r1": 0.007156114950633314}
A1_CLAIMED = {
    "iiib": {"p_primary": 0.0030498475076246186, "loco_max_frac": 0.15364246216529995},
    "joint_r1": {"p_primary": 0.0022998850057497125, "loco_max_frac": 0.14218856317801196},
}
A2_CLAIMED = {
    "r2_deg3_c_pure": 0.8832406614871592,
    "r2_deg3_c_gfrac": 0.8747947939465979,
    "carrier_pred_diff": 0.027083574239199663,
    "carrier_ratio": 0.7645477664060278,
    "fit_resid_diff": 0.008340732036016641,
    "fit_resid_cluster_se": 0.0029160903955559583,
    "fullchain_ratio": {"iiib": 0.6662301458609439, "joint_r1": 0.652514656137263},
    "comp_dl_diff": 0.04039580232842289,
    "comp_relerr_diff": -0.0029038660032495037,
}


def md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- C-4 census (recon recipe)
df = pd.read_csv(CRB)
n = len(df)
assert n == 1590
theta = df["qS"].to_numpy()
phi = df["phiS"].to_numpy()
s_phi2 = df["delta_phiS_delta_phiS"].to_numpy()
s_theta2 = df["delta_qS_delta_qS"].to_numpy()
cv = df["delta_phiS_delta_qS"].to_numpy()
dl = df["luminosity_distance"].to_numpy()
s_dl = np.sqrt(df["delta_luminosity_distance_delta_luminosity_distance"].to_numpy())
snr = df["SNR"].to_numpy()

r = np.empty(n)
for i in range(n):
    sig = np.array([[s_phi2[i], cv[i]], [cv[i], s_theta2[i]]])
    jac = np.diag([abs(np.sin(theta[i])), 1.0])
    r[i] = 2.0 * np.sqrt(max(float(np.linalg.eigvalsh(jac @ sig @ jac.T).max()), 0.0))

st = np.sin(theta)
xyz = np.stack([st * np.cos(phi), st * np.sin(phi), np.cos(theta)], axis=1)
dm = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=2)
iu = np.triu_indices(n, k=1)
sky = dm[iu] <= (r[:, None] + r[None, :])[iu]
ii, jj = iu[0][sky], iu[1][sky]
lo, hi = dl - 2 * s_dl, dl + 2 * s_dl
win = (lo[ii] <= hi[jj]) & (lo[jj] <= hi[ii])
overlap = np.zeros(n, dtype=bool)
overlap[ii[win]] = True
overlap[jj[win]] = True
assert int(sky.sum()) == 1620 and int(win.sum()) == 279 and int(overlap.sum()) == 385

log10_r = np.log10(r)
log10_dL = np.log10(dl)
log10_rel = np.log10(s_dl / dl)


def graph_components(ei: np.ndarray, ej: np.ndarray) -> np.ndarray:
    g = coo_matrix((np.ones(len(ei)), (ei, ej)), shape=(n, n))
    _, labels = connected_components(g, directed=False)
    return labels


comp_win = graph_components(ii[win], jj[win])  # A1's reading: 279 overlap-defining pairs
comp_sky = graph_components(ii, jj)  # harsher reading: full 1620 sky-pair graph
# intermediate reading: sky edges restricted to overlap-event endpoints (no pass-through
# linkage via non-overlap events)
both_ov = overlap[ii] & overlap[jj]
comp_sky_r = graph_components(ii[both_ov], jj[both_ov])

# ---------------------------------------------------------------- venues + chords
frames = {}
ev_ref = None
for venue, path in VENUES.items():
    el = pd.read_csv(path)
    assert len(el) == 65108
    a = el[np.isclose(el.h, 0.60)].set_index("event_idx").sort_index()
    b = el[np.isclose(el.h, 0.73)].set_index("event_idx").sort_index()
    ev = np.array(sorted(set(a.index) & set(b.index)))
    assert len(ev) == 1588
    if ev_ref is None:
        ev_ref = ev
    else:
        assert (ev == ev_ref).all()
    frames[venue] = (a.loc[ev], b.loc[ev])
ev = ev_ref
dropped = sorted(set(range(n)) - set(ev.tolist()))
dropped_are_controls = all(not overlap[i] for i in dropped)

# venue-identity of completion columns (bitwise)
ident_ok = True
for k in (0, 1):
    for col in ("L_comp", "B_num", "B_num_wbh", "g_frac"):
        xi = frames["iiib"][k][col].to_numpy()
        xj = frames["joint_r1"][k][col].to_numpy()
        ident_ok = ident_ok and bool((xi == xj).all())

a_i, b_i = frames["iiib"]
Lc60, Lc73 = a_i.L_comp.to_numpy(), b_i.L_comp.to_numpy()
g60, g73 = a_i.g_frac.to_numpy(), b_i.g_frac.to_numpy()
c_pure = np.log(Lc60) - np.log(Lc73)
c_gfrac = c_pure + (np.log(g60) - np.log(g73))

ov_mask = overlap[ev]
ct_mask = ~ov_mask
ov_idx = np.where(ov_mask)[0]
ct_idx = np.where(ct_mask)[0]
assert len(ov_idx) == 385 and len(ct_idx) == 1203

x1 = log10_dL[ev]
x2 = log10_rel[ev]

# ---------------------------------------------------------------- M-2 matching (2 covariates)
covm = np.stack([log10_r[ev], snr[ev]], axis=1)
z = (covm - covm.mean(axis=0)) / covm.std(axis=0, ddof=1)
d2 = ((z[ov_idx][:, None, :] - z[ct_idx][None, :, :]) ** 2).sum(axis=2)
nn = d2.argmin(axis=1)
mct_idx = ct_idx[nn]
# tie audit: how many overlap events have >1 control at exactly the min distance?
tie_counts = (d2 == d2.min(axis=1, keepdims=True)).sum(axis=1)
n_tied = int((tie_counts > 1).sum())

chords = {}
for venue, (a, b) in frames.items():
    F60 = a.combined_with_bh.to_numpy()
    F73 = b.combined_with_bh.to_numpy()
    chords[venue] = np.log(F60) - np.log(F73)
    got = float((chords[venue][ov_idx] - chords[venue][mct_idx]).mean())
    assert abs(got - M2_TOTALS_2D[venue]) < 1e-12, (venue, got)


# ---------------------------------------------------------------- permutation machinery
def comp_signflip_p(diffs: np.ndarray, cid: np.ndarray, seed: int, n_perm: int = N_PERM) -> float:
    """Two-sided sign-flip p flipping clusters (given by cid) as units, via component sums."""
    uniq, inv = np.unique(cid, return_inverse=True)
    sums = np.zeros(len(uniq))
    np.add.at(sums, inv, diffs)
    rng = np.random.default_rng(seed)
    signs = rng.integers(0, 2, size=(n_perm, len(uniq))) * 2.0 - 1.0
    stats = np.abs(signs @ sums) / len(diffs)
    obs = abs(diffs.mean())
    return float((int((stats >= obs - 1e-15).sum()) + 1) / (n_perm + 1))


def cluster_se(diffs: np.ndarray, clusters: np.ndarray) -> float:
    m = diffs.mean()
    tot = 0.0
    for c in np.unique(clusters):
        tot += float((diffs[clusters == c] - m).sum()) ** 2
    return float(np.sqrt(tot) / len(diffs))


def merged_cluster_ids(cid_overlap: np.ndarray, control_ids: np.ndarray) -> np.ndarray:
    """Merge overlap-graph clusters with shared-control links (harshest exchangeability)."""
    npairs = len(cid_overlap)
    parent = list(range(npairs))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for key in (cid_overlap, control_ids):
        first: dict = {}
        for p in range(npairs):
            k = key[p]
            if k in first:
                union(first[k], p)
            else:
                first[k] = p
    return np.array([find(p) for p in range(npairs)])


# ---------------------------------------------------------------- A1 adjudication
a1_out: dict = {
    "dropped_event_indices_1590_minus_1588": dropped,
    "dropped_are_non_overlap": dropped_are_controls,
    "matching_tie_audit_n_overlap_events_with_tied_argmin": n_tied,
    "graphs": {},
    "venues": {},
}
GRAPHS = (
    ("win279", comp_win),
    ("sky1620", comp_sky),
    ("sky_restricted_overlap_endpoints", comp_sky_r),
)
for gname, comp in GRAPHS:
    cid = comp[ev[ov_idx]]
    uniq, counts = np.unique(cid, return_counts=True)
    a1_out["graphs"][gname] = {
        "n_components_over_385": int(len(uniq)),
        "largest_component": int(counts.max()),
        "size_distribution_top10": sorted(counts.tolist(), reverse=True)[:10],
    }

for venue in VENUES:
    diffs = chords[venue][ov_idx] - chords[venue][mct_idx]
    vout: dict = {"matched_mean_2d": float(diffs.mean())}
    for gname, comp in GRAPHS:
        cid = comp[ev[ov_idx]]
        ps = {str(s): comp_signflip_p(diffs, cid, s) for s in FRESH_SEEDS}
        # jackknife over components
        m_full = diffs.mean()
        deltas = []
        any_flip = False
        comp_sums = {}
        for c in np.unique(cid):
            keep = cid != c
            lm = diffs[keep].mean()
            deltas.append(lm - m_full)
            any_flip = any_flip or (np.sign(lm) != np.sign(m_full))
            comp_sums[int(c)] = float(diffs[cid == c].sum())
        deltas = np.array(deltas)
        top_share = max(comp_sums.values(), key=abs) / float(diffs.sum())
        vout[gname] = {
            "component_signflip_p_by_fresh_seed": ps,
            "all_p_below_alpha": bool(all(p < ALPHA for p in ps.values())),
            "jackknife_max_abs_delta_frac_of_mean": float(np.max(np.abs(deltas)) / abs(m_full)),
            "jackknife_any_sign_flip": bool(any_flip),
            "top_component_share_of_signed_sum": float(top_share),
        }
    # harshest merged clustering (win-graph components + shared-control links)
    cid_w = comp_win[ev[ov_idx]]
    merged = merged_cluster_ids(cid_w, mct_idx)
    n_merged = int(len(np.unique(merged)))
    ps_m = {str(s): comp_signflip_p(diffs, merged, s) for s in FRESH_SEEDS}
    vout["merged_component_plus_sharedcontrol"] = {
        "n_flip_units": n_merged,
        "p_by_fresh_seed": ps_m,
        "all_p_below_alpha": bool(all(p < ALPHA for p in ps_m.values())),
    }
    # alternative literal reading of (ii): stochastic re-matching among k=3 NN controls
    order = np.argsort(d2, axis=1)[:, :3]
    rem = {}
    for s in REMATCH_SEEDS:
        rng = np.random.default_rng(s)
        pick = order[np.arange(385), rng.integers(0, 3, size=385)]
        alt_ct = ct_idx[pick]
        alt_diffs = chords[venue][ov_idx] - chords[venue][alt_ct]
        rem[str(s)] = {
            "matched_mean": float(alt_diffs.mean()),
            "component_signflip_p_win279": comp_signflip_p(
                alt_diffs, comp_win[ev[ov_idx]], s
            ),
        }
    vout["stochastic_rematch_k3"] = {
        "by_seed": rem,
        "all_p_below_alpha": bool(
            all(v["component_signflip_p_win279"] < ALPHA for v in rem.values())
        ),
    }
    a1_out["venues"][venue] = vout

a1_out["claim_check"] = {
    venue: {
        "a1_reported_p_primary": A1_CLAIMED[venue]["p_primary"],
        "my_win279_p_range": [
            min(a1_out["venues"][venue]["win279"]["component_signflip_p_by_fresh_seed"].values()),
            max(a1_out["venues"][venue]["win279"]["component_signflip_p_by_fresh_seed"].values()),
        ],
    }
    for venue in VENUES
}

# ---------------------------------------------------------------- A2 adjudication


def poly_design(u: np.ndarray, v: np.ndarray, degree: int) -> np.ndarray:
    cols = [u**i * v**j for i, j in itertools.product(range(degree + 1), repeat=2) if i + j <= degree]
    return np.stack(cols, axis=1)


def r2(y: np.ndarray, yhat: np.ndarray) -> float:
    return 1.0 - float(((y - yhat) ** 2).sum()) / float(((y - y.mean()) ** 2).sum())


def fit_predict(y: np.ndarray, degree: int, mask: np.ndarray | None = None) -> np.ndarray:
    X = poly_design(x1, x2, degree)
    m = np.ones(len(y), dtype=bool) if mask is None else mask
    beta, *_ = np.linalg.lstsq(X[m], y[m], rcond=None)
    return X @ beta


a2_out: dict = {}
# degree ladder + CV
ladder = {}
for deg in range(1, 7):
    yhat = fit_predict(c_pure, deg)
    ladder[f"deg{deg}"] = {"r2_insample": r2(c_pure, yhat)}
a2_out["degree_ladder_c_pure"] = ladder
a2_out["r2_deg3_recomputed"] = {
    "c_pure": r2(c_pure, fit_predict(c_pure, 3)),
    "c_gfrac": r2(c_gfrac, fit_predict(c_gfrac, 3)),
}

# 5-fold CV (fresh shuffles) — overfitting attack on the smooth fit
cvres = {}
for name, y in (("c_pure", c_pure), ("c_gfrac", c_gfrac)):
    accs = []
    for s in FRESH_SEEDS:
        rng = np.random.default_rng(s)
        perm = rng.permutation(len(y))
        folds = np.array_split(perm, 5)
        yhat = np.empty_like(y)
        for f in folds:
            m = np.ones(len(y), dtype=bool)
            m[f] = False
            X = poly_design(x1, x2, 3)
            beta, *_ = np.linalg.lstsq(X[m], y[m], rcond=None)
            yhat[f] = X[f] @ beta
        accs.append(r2(y, yhat))
    cvres[name] = {"cv5_r2_by_seed": dict(zip(map(str, FRESH_SEEDS), accs, strict=True))}
a2_out["cross_validation_deg3"] = cvres

# random half-split: fit half, evaluate other half
half = {}
for s in FRESH_SEEDS:
    rng = np.random.default_rng(s)
    perm = rng.permutation(len(c_pure))
    h1, h2 = perm[: len(perm) // 2], perm[len(perm) // 2 :]
    X = poly_design(x1, x2, 3)
    beta, *_ = np.linalg.lstsq(X[h1], c_pure[h1], rcond=None)
    half[str(s)] = r2(c_pure[h2], X[h2] @ beta)
a2_out["half_split_r2_c_pure"] = half

# alternative binnings (8x8, 16x16) + shape stats
bins = {}
for nb in (8, 16):
    e1 = np.quantile(x1, np.linspace(0, 1, nb + 1))
    e2 = np.quantile(x2, np.linspace(0, 1, nb + 1))
    b1 = np.clip(np.digitize(x1, e1[1:-1]), 0, nb - 1)
    b2 = np.clip(np.digitize(x2, e2[1:-1]), 0, nb - 1)
    cell = b1 * nb + b2
    yhat = np.empty_like(c_pure)
    for c in np.unique(cell):
        m = cell == c
        yhat[m] = c_pure[m].mean()
    bins[f"{nb}x{nb}"] = r2(c_pure, yhat)
a2_out["alt_binned_r2_c_pure"] = bins
rho1 = spearmanr(c_pure, x1)
a2_out["spearman_c_pure_vs_log10dL"] = float(rho1.statistic)
dec_e = np.quantile(x1, np.linspace(0, 1, 11))
db = np.clip(np.digitize(x1, dec_e[1:-1]), 0, 9)
prof = np.array([c_pure[db == k].mean() for k in range(10)])
a2_out["decile_profile_monotone_increasing"] = bool(np.all(np.diff(prof) > 0))
a2_out["decile_profile_first_last"] = [float(prof[0]), float(prof[-1])]

# composition + carrier + fit-residual (matched pairs)
comp_dl = float((x1[ov_idx] - x1[mct_idx]).mean())
comp_re = float((x2[ov_idx] - x2[mct_idx]).mean())
obs_carrier = float((c_pure[ov_idx] - c_pure[mct_idx]).mean())
fhat_pure = fit_predict(c_pure, 3)
fhat_gfrac = fit_predict(c_gfrac, 3)
pred_carrier = float((fhat_pure[ov_idx] - fhat_pure[mct_idx]).mean())
resid = c_pure - fhat_pure
resid_diffs = resid[ov_idx] - resid[mct_idx]
a2_out["composition"] = {"log10_dL_matched_diff": comp_dl, "log10_relerr_matched_diff": comp_re}
a2_out["carrier"] = {
    "observed_matched_diff": obs_carrier,
    "matches_d1_raw_lcomp": bool(abs(obs_carrier - D1_RAW_LCOMP_DIFF) < 1e-12),
    "predicted_matched_diff": pred_carrier,
    "ratio": pred_carrier / obs_carrier,
    "fit_residual_matched_diff": float(resid_diffs.mean()),
    "fit_residual_cluster_se": cluster_se(resid_diffs, nn),
}

# full chain per venue
def log_mean(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    out = np.empty_like(x)
    close = np.isclose(x, y, rtol=1e-12, atol=0.0)
    out[close] = 0.5 * (x[close] + y[close])
    nc = ~close
    out[nc] = (x[nc] - y[nc]) / (np.log(x[nc]) - np.log(y[nc]))
    return out


fullchain = {}
venue_SB = {}
venue_TlegA = {}
venue_dlnw = {}
for venue, (a, b) in frames.items():
    wt60, wt73 = float(a.w_G.iloc[0]), float(b.w_G.iloc[0])
    dln_w1m = float(np.log1p(-wt60) - np.log1p(-wt73))
    A60 = wt60 * a.L_cat_with_bh.to_numpy()
    A73 = wt73 * b.L_cat_with_bh.to_numpy()
    B60 = (1 - wt60) * g60 * Lc60
    B73 = (1 - wt73) * g73 * Lc73
    LF = log_mean(A60 + B60, A73 + B73)
    SB = log_mean(B60, B73) / LF
    T_legB = SB * (dln_w1m + c_gfrac)
    T_legA = (np.log(A60 + B60) - np.log(A73 + B73)) - T_legB
    venue_SB[venue] = SB
    venue_TlegA[venue] = T_legA
    venue_dlnw[venue] = dln_w1m
    obs_total = float((chords[venue][ov_idx] - chords[venue][mct_idx]).mean())
    TlegB_hat = SB * (dln_w1m + fhat_gfrac)
    pred_total = float((venue_TlegA[venue][ov_idx] - venue_TlegA[venue][mct_idx]).mean()) + float(
        (TlegB_hat[ov_idx] - TlegB_hat[mct_idx]).mean()
    )
    fullchain[venue] = {
        "observed_total": obs_total,
        "predicted_total": pred_total,
        "ratio": pred_total / obs_total,
    }
a2_out["full_chain"] = fullchain

# ---- ATTACK 1: control-vs-control placebo (pseudo-treated = the 385 matched-control slots)
d2_cc = ((z[mct_idx][:, None, :] - z[ct_idx][None, :, :]) ** 2).sum(axis=2)
for p in range(385):
    d2_cc[p, np.where(ct_idx == mct_idx[p])[0]] = np.inf  # exclude self
nn_cc = d2_cc.argmin(axis=1)
pl_ct = ct_idx[nn_cc]
placebo = {}
pl_resid = resid[mct_idx] - resid[pl_ct]
placebo["fit_residual_placebo_diff"] = float(pl_resid.mean())
placebo["fit_residual_placebo_cluster_se"] = cluster_se(pl_resid, nn_cc)
placebo["observed_carrier_placebo_diff"] = float((c_pure[mct_idx] - c_pure[pl_ct]).mean())
placebo["predicted_carrier_placebo_diff"] = float((fhat_pure[mct_idx] - fhat_pure[pl_ct]).mean())
for venue in VENUES:
    placebo[f"observed_total_placebo_{venue}"] = float(
        (chords[venue][mct_idx] - chords[venue][pl_ct]).mean()
    )
a2_out["placebo_control_vs_control"] = placebo

# ---- ATTACK 2: D-1 3-covariate d_L-matched postdiction
covm3 = np.stack([log10_r[ev], snr[ev], dl[ev]], axis=1)
z3 = (covm3 - covm3.mean(axis=0)) / covm3.std(axis=0, ddof=1)
d2_3 = ((z3[ov_idx][:, None, :] - z3[ct_idx][None, :, :]) ** 2).sum(axis=2)
nn3 = d2_3.argmin(axis=1)
mct3 = ct_idx[nn3]
postdict = {}
for venue in VENUES:
    obs3 = float((chords[venue][ov_idx] - chords[venue][mct3]).mean())
    TlegB_hat = venue_SB[venue] * (venue_dlnw[venue] + fhat_gfrac)
    pred3 = float((venue_TlegA[venue][ov_idx] - venue_TlegA[venue][mct3]).mean()) + float(
        (TlegB_hat[ov_idx] - TlegB_hat[mct3]).mean()
    )
    postdict[venue] = {
        "observed_total_dL_matched": obs3,
        "d1_committed_value": D1_DL_MATCHED_TOTALS[venue],
        "reproduces_d1": bool(abs(obs3 - D1_DL_MATCHED_TOTALS[venue]) < 1e-9),
        "function_predicted_total_dL_matched": pred3,
        "gap_obs_minus_pred": obs3 - pred3,
    }
resid3 = resid[ov_idx] - resid[mct3]
postdict["carrier_fit_residual_dL_matched_diff"] = float(resid3.mean())
postdict["carrier_fit_residual_dL_matched_cluster_se"] = cluster_se(resid3, nn3)
postdict["composition_dL_matched_log10dL_diff"] = float((x1[ov_idx] - x1[mct3]).mean())
a2_out["dL_matched_postdiction"] = postdict

# ---- consistency of c_gfrac matched diff with D-1's raw B_num_wbh chord
c_gfrac_diff = float((c_gfrac[ov_idx] - c_gfrac[mct_idx]).mean())
a2_out["c_gfrac_matched_diff_vs_d1_bnumwbh"] = {
    "c_gfrac_matched_diff": c_gfrac_diff,
    "d1_raw_bnumwbh": 0.03454539606780529,
    "abs_gap": abs(c_gfrac_diff - 0.03454539606780529),
}

# ---------------------------------------------------------------- provenance + write
results = {
    "adjudication": "adversarial verification of A1 (a1_results.json) and A2 (a2_results.json)",
    "inputs_md5": {
        "crb": md5(CRB),
        **{v: md5(p) for v, p in VENUES.items()},
        "a1_script": md5(f"{OUTDIR}/a1_graph_signflip.py"),
        "a2_script": md5(f"{OUTDIR}/a2_completion_functional.py"),
    },
    "n_perm": N_PERM,
    "fresh_seeds": FRESH_SEEDS,
    "venue_identity_bitwise_ok": ident_ok,
    "A1": a1_out,
    "A2": a2_out,
}
with open(OUT, "w") as f:
    json.dump(results, f, indent=2)
print(json.dumps(results, indent=2))
