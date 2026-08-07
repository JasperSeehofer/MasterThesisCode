"""D-1: component decomposition of the M-2 matched 2D overlap residual.

Reuses the EXACT matched-pair machinery of
crossterm_instrument/m2_overlap_stratified.py (same C-4 census asserts
1620/279/385, same 1-NN-with-replacement matching on standardized
(log10 ball-radius chord, SNR), same 385 pairs, signflip seed 20260805 +
cluster-robust seed 99, 20000 perms) but scores the chord
ln X(0.60) - ln X(0.73) COMPONENT-WISE.

Composition (verified in-script against event_likelihoods.csv, Path (A)
mixture, bayesian_statistics.py:4376-4421 + path_a_mixture_objects:1998):

    2D: combined_with_bh = wt * L_cat_with_bh + (1 - wt) * g_frac * L_comp
    1D: combined_no_bh   = (wt / r_Malm) * L_cat_no_bh + (1 - wt) * L_comp

with wt = w_G = alpha_G_phi / D_tilde_phi EVENT-INDEPENDENT at fixed h
(asserted), g_frac = B_num_wbh / B_num, L_comp = B_num / beta_Gbar
(beta_Gbar event-independent, asserted via B_num/L_comp constancy).

Additive decomposition (exact, LMDI / log-mean Divisia):  with
A = catalogue leg, B = completion leg, F = A + B, L(x,y) = (x-y)/ln(x/y)
the logarithmic mean (L(x,x) = x),

    chord(F) = (A60 - A73)/L(F60,F73) + (B60 - B73)/L(F60,F73)
             = T_legA + T_legB                      (exact, all events)
    T_legA   = S_A * (dln wt + dln L_cat),  S_A = L(A60,A73)/L(F60,F73)
               (T_legA = 0 exactly where the ball is empty: A == 0 at both h)
    T_legB   = S_B * (dln(1-wt) + dln g_frac + dln L_comp)   [2D]
             = S_B * (dln(1-wt) + dln L_comp)                [1D]

so the per-event chord splits EXACTLY (up to the CSV 7-sig-fig recon
residual, reported) into:
    2D: T_cat + T_wG + T_gfrac + T_Lcomp + T_w1m + resid
    1D: T_cat + T_wG (incl. -dln r_Malm) + T_Lcomp + T_w1m + resid
where T_wG = S_A * dln(wt) (2D) / S_A * dln(wt/r_Malm) (1D) and
T_w1m = S_B * dln(1-wt) are the composition-weight channels (the weight
chords are event-independent CONSTANTS; they enter per event only through
the leg shares S_A, S_B).

Raw per-column chords (L_cat_*, B_num*, g_frac, L_comp, weight columns)
are read alongside with zero/undefined guards and coverage reporting.

Tests per component: matched paired diff (overlap - matched control) with
sign-flip permutation p (fresh default_rng(20260805) per test, vectorized,
20000 perms — matched-pair ASSIGNMENT is deterministic and identical to
M-2; p-values reproduce to MC error) and cluster-robust sign-flip p
(clusters = pairs sharing a control, fresh default_rng(99) per test,
20000 perms). Unmatched mean diff reported WITHOUT p (M-2 established the
unmatched read is selection-dominated; D-1's question is the matched
residual). M-2 headline totals are asserted to reproduce exactly.

FREE READ: existing CSVs only; no production runs. Output: d1_results.json.
"""

import json

import numpy as np
import pandas as pd

N_PERM = 20000
SEED_SIGNFLIP = 20260805
SEED_CLUSTER = 99
ROOT = "/home/jasper/Repositories/MasterThesisCode"
OUTDIR = f"{ROOT}/results/campaign51_20260728/realistic_20260729/m2_residual_owner"
OUT = f"{OUTDIR}/d1_results.json"
CRB = f"{ROOT}/results/run_20260804_postfix/joint_r1/diagnostics/prepared_cramer_rao_bounds.csv"
VENUES = {
    "iiib": f"{ROOT}/results/run_20260804_postfix/iiib/diagnostics/event_likelihoods.csv",
    "joint_r1": f"{ROOT}/results/run_20260804_postfix/joint_r1/diagnostics/event_likelihoods.csv",
}
# M-2 committed matched totals (m2_results.json) — asserted below.
M2_TOTALS = {
    ("iiib", "2d"): 0.022252643015992925,
    ("iiib", "1d"): 0.006042136841274665,
    ("joint_r1", "2d"): 0.020697491999731973,
    ("joint_r1", "1d"): 0.007429539319923175,
}

# ---------------------------------------------------------------- C-4 census (verbatim M-2)
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

# ---------------------------------------------------------------- helpers


def smd(a: np.ndarray, b: np.ndarray) -> float:
    sp = np.sqrt(0.5 * (a.var(ddof=1) + b.var(ddof=1)))
    return float((a.mean() - b.mean()) / sp) if sp > 0 else 0.0


def log_mean(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Logarithmic mean L(x,y) = (x-y)/ln(x/y); L(x,x) = x. Requires x,y > 0."""
    out = np.empty_like(x)
    close = np.isclose(x, y, rtol=1e-12, atol=0.0)
    out[close] = 0.5 * (x[close] + y[close])
    nc = ~close
    out[nc] = (x[nc] - y[nc]) / (np.log(x[nc]) - np.log(y[nc]))
    return out


def signflip_p(diffs: np.ndarray, seed: int = SEED_SIGNFLIP) -> float:
    """Two-sided sign-flip permutation p, vectorized, fresh rng per test."""
    rng = np.random.default_rng(seed)
    obs = abs(diffs.mean())
    m = len(diffs)
    signs = rng.choice([-1.0, 1.0], size=(N_PERM, m))
    stats = np.abs((signs * diffs[None, :]).mean(axis=1))
    return float((int((stats >= obs).sum()) + 1) / (N_PERM + 1))


def cluster_signflip_p(diffs: np.ndarray, clusters: np.ndarray, seed: int = SEED_CLUSTER) -> float:
    """Two-sided sign-flip p flipping all pairs sharing a control together."""
    rng = np.random.default_rng(seed)
    obs = abs(diffs.mean())
    uniq, inv = np.unique(clusters, return_inverse=True)
    signs = rng.choice([-1.0, 1.0], size=(N_PERM, len(uniq)))
    stats = np.abs((signs[:, inv] * diffs[None, :]).mean(axis=1))
    return float((int((stats >= obs).sum()) + 1) / (N_PERM + 1))


# ---------------------------------------------------------------- per-venue read
results: dict = {
    "read": "D-1 component decomposition of the M-2 matched 2D overlap residual",
    "chord_definition": "ln X(h=0.60) - ln X(h=0.73) per event; positive = prefers low h",
    "machinery": "verbatim M-2 census+matching (crossterm_instrument/m2_overlap_stratified.py); "
    "matched pairs identical (deterministic 1-NN argmin); signflip seed 20260805, "
    "cluster-robust seed 99, 20000 perms, fresh rng per component test (p to MC error)",
    "composition": {
        "2d": "combined_with_bh = wt*L_cat_with_bh + (1-wt)*g_frac*L_comp",
        "1d": "combined_no_bh = (wt/r_Malm)*L_cat_no_bh + (1-wt)*L_comp",
        "wt": "w_G = alpha_G_phi/D_tilde_phi, event-independent at fixed h (asserted)",
        "decomposition": "exact LMDI: chord = sum of terms, T_x = S_leg * dln(x), "
        "S_leg = logmean(leg60,leg73)/logmean(F60,F73); T_legA = 0 exactly for empty-ball events",
    },
    "venues": {},
}

log10_r = np.log10(r)
cov_mat = np.stack([log10_r, snr], axis=1)

for venue, path in VENUES.items():
    el = pd.read_csv(path)
    assert len(el) == 65108, (venue, len(el))
    a = el[np.isclose(el.h, 0.60)].set_index("event_idx").sort_index()
    b = el[np.isclose(el.h, 0.73)].set_index("event_idx").sort_index()
    ev = np.array(sorted(set(a.index) & set(b.index)))
    assert len(ev) == 1588, (venue, len(ev))
    a = a.loc[ev]
    b = b.loc[ev]

    # -- weight columns event-independent at fixed h (asserted, as in M-2)
    for frame, h0 in ((a, 0.60), (b, 0.73)):
        for c in ("w_G", "alpha_G_phi", "r_Malm", "D_tilde_phi"):
            assert frame[c].nunique() == 1, (venue, h0, c)
    wt60, wt73 = float(a.w_G.iloc[0]), float(b.w_G.iloc[0])
    rM60, rM73 = float(a.r_Malm.iloc[0]), float(b.r_Malm.iloc[0])
    dln_wt = np.log(wt60) - np.log(wt73)
    dln_w1m = np.log1p(-wt60) - np.log1p(-wt73)
    dln_rM = np.log(rM60) - np.log(rM73)
    # beta_Gbar event-independence via B_num/L_comp constancy
    for frame in (a, b):
        ratio = frame.B_num.to_numpy() / frame.L_comp.to_numpy()
        assert np.nanstd(ratio) / np.nanmean(ratio) < 1e-5, venue

    # -- strata + matching (verbatim M-2; deterministic)
    ov_mask = overlap[ev]
    ct_mask = ~ov_mask
    ov_ev, ct_ev = ev[ov_mask], ev[ct_mask]
    assert len(ov_ev) == 385 and len(ct_ev) == 1203, venue
    z = (cov_mat[ev] - cov_mat[ev].mean(axis=0)) / cov_mat[ev].std(axis=0, ddof=1)
    z_ov, z_ct = z[ov_mask], z[ct_mask]
    dist2 = ((z_ov[:, None, :] - z_ct[None, :, :]) ** 2).sum(axis=2)
    nn = dist2.argmin(axis=1)
    matched_ct_ev = ct_ev[nn]
    bal = {
        name: {
            "smd_before": smd(cov_mat[ov_ev, k], cov_mat[ct_ev, k]),
            "smd_after": smd(cov_mat[ov_ev, k], cov_mat[matched_ct_ev, k]),
        }
        for k, name in enumerate(["log10_radius_chord", "SNR"])
    }
    bal["n_unique_controls_used"] = int(len(np.unique(nn)))

    # positional index maps: ev -> position
    pos = {e: i for i, e in enumerate(ev)}
    ov_pos = np.array([pos[e] for e in ov_ev])
    mct_pos = np.array([pos[e] for e in matched_ct_ev])
    ct_pos = np.array([pos[e] for e in ct_ev])

    venue_out: dict = {
        "n_events": int(len(ev)),
        "n_overlap": 385,
        "n_control": 1203,
        "balance": bal,
        "weight_constants": {
            "wt_060": wt60,
            "wt_073": wt73,
            "dln_wt": float(dln_wt),
            "dln_one_minus_wt": float(dln_w1m),
            "r_Malm_060": rM60,
            "r_Malm_073": rM73,
            "dln_r_Malm": float(dln_rM),
            "dln_alpha_G_phi": float(np.log(a.alpha_G_phi.iloc[0]) - np.log(b.alpha_G_phi.iloc[0])),
            "dln_D_tilde_phi": float(np.log(a.D_tilde_phi.iloc[0]) - np.log(b.D_tilde_phi.iloc[0])),
        },
        "channels": {},
    }

    for ch in ("2d", "1d"):
        if ch == "2d":
            Lcat60 = a.L_cat_with_bh.to_numpy()
            Lcat73 = b.L_cat_with_bh.to_numpy()
            wA60, wA73 = wt60, wt73
            dln_wA = dln_wt
            B60 = (1 - wt60) * a.g_frac.to_numpy() * a.L_comp.to_numpy()
            B73 = (1 - wt73) * b.g_frac.to_numpy() * b.L_comp.to_numpy()
            F60 = a.combined_with_bh.to_numpy()
            F73 = b.combined_with_bh.to_numpy()
            raw_cols = ["L_cat_with_bh", "B_num_wbh", "g_frac", "L_comp"]
        else:
            Lcat60 = a.L_cat_no_bh.to_numpy()
            Lcat73 = b.L_cat_no_bh.to_numpy()
            wA60, wA73 = wt60 / rM60, wt73 / rM73
            dln_wA = dln_wt - dln_rM
            B60 = (1 - wt60) * a.L_comp.to_numpy()
            B73 = (1 - wt73) * b.L_comp.to_numpy()
            F60 = a.combined_no_bh.to_numpy()
            F73 = b.combined_no_bh.to_numpy()
            raw_cols = ["L_cat_no_bh", "B_num", "L_comp"]

        # zero-set h-stability (empty balls are h-independent)
        z60, z73 = Lcat60 == 0.0, Lcat73 == 0.0
        assert (z60 == z73).all(), (venue, ch, "L_cat zero set not h-stable")
        nz = ~z60  # events with a non-empty (mass-conditioned) ball
        assert (F60 > 0).all() and (F73 > 0).all(), (venue, ch)
        assert (B60 > 0).all() and (B73 > 0).all(), (venue, ch)

        chord_csv = np.log(F60) - np.log(F73)

        # -- exact LMDI decomposition on reconstructed legs
        A60 = wA60 * Lcat60
        A73 = wA73 * Lcat73
        Frec60, Frec73 = A60 + B60, A73 + B73
        LF = log_mean(Frec60, Frec73)
        SB = log_mean(B60, B73) / LF
        SA = np.zeros_like(LF)
        SA[nz] = log_mean(A60[nz], A73[nz]) / LF[nz]
        dln_Lcat = np.full_like(LF, np.nan)
        dln_Lcat[nz] = np.log(Lcat60[nz]) - np.log(Lcat73[nz])
        dln_Lcomp = np.log(a.L_comp.to_numpy()) - np.log(b.L_comp.to_numpy())

        T = {}
        T["T_cat"] = np.where(nz, SA * np.nan_to_num(dln_Lcat), 0.0)
        T["T_wG"] = SA * dln_wA
        T["T_Lcomp"] = SB * dln_Lcomp
        T["T_w1m"] = SB * dln_w1m
        if ch == "2d":
            dln_g = np.log(a.g_frac.to_numpy()) - np.log(b.g_frac.to_numpy())
            T["T_gfrac"] = SB * dln_g
        chord_rec = np.log(Frec60) - np.log(Frec73)
        term_sum = sum(T.values())
        # exactness of the LMDI split on the reconstructed composition
        assert np.abs(term_sum - chord_rec).max() < 1e-10, (venue, ch)
        T["resid_recon"] = chord_csv - chord_rec  # CSV 7-sig-fig rounding only
        T["T_legA"] = T["T_cat"] + T["T_wG"]
        T["T_legB"] = term_sum - T["T_legA"]
        T["total_chord"] = chord_csv

        # -- component read: matched + unmatched, with coverage guards
        comp_out: dict = {}

        def read_component(vals: np.ndarray, name: str, event_independent: bool = False) -> dict:
            """Matched + unmatched stratum read of a per-event quantity (NaN = undefined)."""
            def_mask = np.isfinite(vals)
            cov_all = float(def_mask.mean())
            ov_def = def_mask[ov_pos]  # noqa: B023
            mct_def = def_mask[mct_pos]  # noqa: B023
            ct_def = def_mask[ct_pos]  # noqa: B023
            pair_ok = ov_def & mct_def
            n_pairs = int(pair_ok.sum())
            out: dict = {
                "coverage": {
                    "all_events": cov_all,
                    "overlap": float(ov_def.mean()),
                    "control": float(ct_def.mean()),
                    "n_pairs_used": n_pairs,
                },
                "event_independent_constant": event_independent,
            }
            if n_pairs == 0:
                out["matched"] = None
                return out
            diffs = vals[ov_pos[pair_ok]] - vals[mct_pos[pair_ok]]  # noqa: B023
            um = (
                float(np.nanmean(vals[ov_pos[ov_def]]) - np.nanmean(vals[ct_pos[ct_def]]))  # noqa: B023
                if ct_def.any() and ov_def.any()
                else None
            )
            out["unmatched_mean_diff"] = um
            if event_independent:
                out["matched"] = {
                    "mean_paired_diff": float(diffs.mean()),
                    "note": "constant per event at fixed h -> matched diff identically 0",
                }
                return out
            out["matched"] = {
                "mean_paired_diff": float(diffs.mean()),
                "median_paired_diff": float(np.median(diffs)),
                "paired_diff_std": float(diffs.std(ddof=1)),
                "signflip_p": signflip_p(diffs),
                "cluster_signflip_p": cluster_signflip_p(diffs, nn[pair_ok]),  # noqa: B023
            }
            return out

        # exact-decomposition terms (full coverage by construction)
        for name in (
            "total_chord",
            "T_legA",
            "T_legB",
            "T_cat",
            "T_wG",
            "T_gfrac",
            "T_Lcomp",
            "T_w1m",
            "resid_recon",
        ):
            if name not in T:
                continue
            comp_out[name] = read_component(T[name], name)

        # assert M-2 headline reproduction (exact machinery)
        tot = comp_out["total_chord"]["matched"]["mean_paired_diff"]
        assert abs(tot - M2_TOTALS[(venue, ch)]) < 1e-12, (venue, ch, tot)

        # fractions of total (level-1 exact additivity: legA + legB + resid = total)
        for name in (
            "T_legA",
            "T_legB",
            "T_cat",
            "T_wG",
            "T_gfrac",
            "T_Lcomp",
            "T_w1m",
            "resid_recon",
        ):
            if name in comp_out and comp_out[name]["matched"] is not None:
                comp_out[name]["fraction_of_total_matched_diff"] = (
                    comp_out[name]["matched"]["mean_paired_diff"] / tot
                )

        # raw per-column chords (guarded; partial coverage where zeros)
        raw_out: dict = {}
        for col in raw_cols:
            x60 = a[col].to_numpy().astype(float)
            x73 = b[col].to_numpy().astype(float)
            ok = (x60 > 0) & (x73 > 0) & np.isfinite(x60) & np.isfinite(x73)
            vals = np.full(len(ev), np.nan)
            vals[ok] = np.log(x60[ok]) - np.log(x73[ok])
            raw_out[col] = read_component(vals, col)
            raw_out[col]["n_zero_or_undefined"] = int((~ok).sum())
        for col in ("w_G", "alpha_G_phi", "r_Malm", "D_tilde_phi"):
            x60 = a[col].to_numpy().astype(float)
            x73 = b[col].to_numpy().astype(float)
            vals = np.log(x60) - np.log(x73)
            raw_out[col] = read_component(vals, col, event_independent=True)
            raw_out[col]["constant_chord"] = float(vals[0])

        # decomposition fidelity
        rel60 = np.abs(Frec60 - F60) / F60
        rel73 = np.abs(Frec73 - F73) / F73
        venue_out["channels"][ch] = {
            "reconstruction_max_rel_err": float(max(rel60.max(), rel73.max())),
            "resid_recon_max_abs_nats": float(np.abs(T["resid_recon"]).max()),
            "n_empty_ball_events": int((~nz).sum()),
            "n_empty_ball_overlap": int((~nz[ov_pos]).sum()),
            "n_empty_ball_matched_controls": int((~nz[mct_pos]).sum()),
            "decomposition_terms": comp_out,
            "raw_column_chords": raw_out,
            # leg-share diagnostics: does the overlap stratum sit deeper in the catalogue leg?
            "S_A_matched_diff": {
                "mean_paired_diff": float((SA[ov_pos] - SA[mct_pos]).mean()),
                "signflip_p": signflip_p(SA[ov_pos] - SA[mct_pos]),
                "cluster_signflip_p": cluster_signflip_p(SA[ov_pos] - SA[mct_pos], nn),
                "overlap_mean_S_A": float(SA[ov_pos].mean()),
                "matched_control_mean_S_A": float(SA[mct_pos].mean()),
                "control_mean_S_A": float(SA[ct_pos].mean()),
            },
        }

    results["venues"][venue] = venue_out

# ---------------------------------------------------------------- supplementary probes
# (1) Cross-venue completion-leg identity: L_comp/B_num/g_frac depend only on the
# event's GW posterior + homogeneous completion model, not on the catalogue, so the
# columns should be identical across venues (explains the venue-stability of the
# 2D residual). Verified numerically here.
el_i = pd.read_csv(VENUES["iiib"])
el_j = pd.read_csv(VENUES["joint_r1"])
xvenue: dict = {}
for h0 in (0.60, 0.73):
    ai = el_i[np.isclose(el_i.h, h0)].set_index("event_idx").sort_index()
    aj = el_j[np.isclose(el_j.h, h0)].set_index("event_idx").sort_index()
    for col in ("L_comp", "B_num", "B_num_wbh", "g_frac"):
        xi, xj = ai[col].to_numpy(), aj[col].to_numpy()
        rel = np.abs(xi - xj) / np.maximum(np.abs(xj), 1e-300)
        xvenue[f"{col}_h{h0}"] = {"max_rel_diff_iiib_vs_joint_r1": float(np.nanmax(rel))}
results["cross_venue_completion_leg_identity"] = {
    "note": "completion-leg columns compared element-wise across venues at both anchor h",
    "columns": xvenue,
}

# (2) d_L confounder probe: the C-4 overlap predicate selects on d_L (2-sigma window
# intersection), which the M-2 matching covariates (log10 radius, SNR) do NOT control.
# The localized carrier (L_comp chord) is a pure function of the event's d_L posterior,
# so a matched d_L imbalance is the natural non-interaction explanation. Matched diffs
# of d_L, sigma_dL, and relative d_L error over the SAME 385 pairs:
ev_any = np.array(sorted(set(el_j[np.isclose(el_j.h, 0.73)].event_idx)))
probe_out: dict = {}
_z = (cov_mat[ev_any] - cov_mat[ev_any].mean(axis=0)) / cov_mat[ev_any].std(axis=0, ddof=1)
_ovm = overlap[ev_any]
_ov_ev, _ct_ev = ev_any[_ovm], ev_any[~_ovm]
_d2 = ((_z[_ovm][:, None, :] - _z[~_ovm][None, :, :]) ** 2).sum(axis=2)
_nn = _d2.argmin(axis=1)
_mct_ev = _ct_ev[_nn]
for name, arr in (
    ("d_L_Gpc", dl),
    ("sigma_dL_Gpc", s_dl),
    ("rel_dL_err", s_dl / dl),
    ("log10_radius_chord", log10_r),
    ("SNR", snr),
):
    diffs = arr[_ov_ev] - arr[_mct_ev]
    probe_out[name] = {
        "matched_mean_paired_diff": float(diffs.mean()),
        "matched_median_paired_diff": float(np.median(diffs)),
        "signflip_p": signflip_p(diffs),
        "cluster_signflip_p": cluster_signflip_p(diffs, _nn),
        "overlap_mean": float(arr[_ov_ev].mean()),
        "matched_control_mean": float(arr[_mct_ev].mean()),
        "control_mean": float(arr[_ct_ev].mean()),
        "smd_after_matching": smd(arr[_ov_ev], arr[_mct_ev]),
    }
results["dL_confounder_probe"] = {
    "note": "covariates from prepared_cramer_rao_bounds.csv over the SAME 385 matched pairs; "
    "d_L and sigma_dL were NOT matching covariates in M-2 — an imbalance here is a "
    "candidate non-interaction owner of the completion-leg residual",
    "covariates": probe_out,
}

# (3) SENSITIVITY (not the primary read): 1-NN re-match adding d_L as a THIRD
# standardized covariate (log10 radius, SNR, d_L). If the completion-leg carrier is a
# d_L-composition effect of the C-4 d_L-window predicate (not an interaction), the
# matched 2D total and its T_Lcomp component should shrink/lose significance here.
sens: dict = {"covariates": ["log10_radius_chord", "SNR", "d_L"], "venues": {}}
cov3 = np.stack([log10_r, snr, dl], axis=1)
for venue, path in VENUES.items():
    el = pd.read_csv(path)
    a = el[np.isclose(el.h, 0.60)].set_index("event_idx").sort_index()
    b = el[np.isclose(el.h, 0.73)].set_index("event_idx").sort_index()
    ev = np.array(sorted(set(a.index) & set(b.index)))
    a, b = a.loc[ev], b.loc[ev]
    ovm = overlap[ev]
    ov_ev3, ct_ev3 = ev[ovm], ev[~ovm]
    z3 = (cov3[ev] - cov3[ev].mean(axis=0)) / cov3[ev].std(axis=0, ddof=1)
    d23 = ((z3[ovm][:, None, :] - z3[~ovm][None, :, :]) ** 2).sum(axis=2)
    nn3 = d23.argmin(axis=1)
    mct_ev3 = ct_ev3[nn3]
    bal3 = {
        name: smd(cov3[ov_ev3, k], cov3[mct_ev3, k])
        for k, name in enumerate(["log10_radius_chord", "SNR", "d_L"])
    }
    wt60_, wt73_ = float(a.w_G.iloc[0]), float(b.w_G.iloc[0])
    vout: dict = {"balance_smd_after": bal3, "n_unique_controls_used": int(len(np.unique(nn3)))}
    for ch, col in (("2d", "combined_with_bh"), ("1d", "combined_no_bh")):
        chord = np.log(a[col].to_numpy()) - np.log(b[col].to_numpy())
        cs = pd.Series(chord, index=ev)
        diffs = cs.loc[ov_ev3].to_numpy() - cs.loc[mct_ev3].to_numpy()
        entry = {
            "total_matched_mean_paired_diff": float(diffs.mean()),
            "signflip_p": signflip_p(diffs),
            "cluster_signflip_p": cluster_signflip_p(diffs, nn3),
        }
        if ch == "2d":
            # T_Lcomp under the re-match
            B60s = (1 - wt60_) * a.g_frac.to_numpy() * a.L_comp.to_numpy()
            B73s = (1 - wt73_) * b.g_frac.to_numpy() * b.L_comp.to_numpy()
            A60s = wt60_ * a.L_cat_with_bh.to_numpy()
            A73s = wt73_ * b.L_cat_with_bh.to_numpy()
            LFs = log_mean(A60s + B60s, A73s + B73s)
            SBs = log_mean(B60s, B73s) / LFs
            tl = pd.Series(
                SBs * (np.log(a.L_comp.to_numpy()) - np.log(b.L_comp.to_numpy())), index=ev
            )
            td = tl.loc[ov_ev3].to_numpy() - tl.loc[mct_ev3].to_numpy()
            entry["T_Lcomp_matched_mean_paired_diff"] = float(td.mean())
            entry["T_Lcomp_signflip_p"] = signflip_p(td)
            entry["T_Lcomp_cluster_signflip_p"] = cluster_signflip_p(td, nn3)
        vout[ch] = entry
    sens["venues"][venue] = vout
results["sensitivity_dL_matched"] = sens

with open(OUT, "w") as f:
    json.dump(results, f, indent=2)
print(json.dumps(results, indent=2))
