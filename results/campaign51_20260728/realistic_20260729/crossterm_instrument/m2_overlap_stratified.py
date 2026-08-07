"""M-2: overlap-stratified per-event chord read (spec: CLAIM_HITCHHIKER_INDEPENDENCE_20260805.md.DRAFT
lines 550-560).

Partition events into the C-4 overlap stratum (sky-overlapping + 2-sigma d_L-compatible partner,
reproduced EXACTLY via the recon_c4_census.py recipe from prepared_cramer_rao_bounds.csv) vs the
rest.  Compare per-event chords  ln L(h=0.60) - ln L(h=0.73)  in 1D (combined_no_bh) and 2D
(combined_with_bh) at both venues (run_20260804_postfix/{iiib,joint_r1}), as an aggregate
(unmatched) AND with the [A2]-mandated matched read.

Matching covariates: log10(ball radius chord) and SNR (per-event drivers of localisation area /
w_G).  NOTE (verified this session): the operative w_G column in event_likelihoods.csv is
EVENT-INDEPENDENT at fixed h (nunique == 1 per h per venue), so "match on w_G" is discharged by
matching on its per-event drivers; the script asserts this constancy.

Tests: unmatched = label-permutation p on the difference of stratum means (two-sided);
matched = 1-NN matching with replacement (control pool -> each overlap event) in standardized
(log10 r, SNR) space, then a sign-flip permutation p on the paired differences (two-sided).
Pre-registered expectation: NULL.  Call criterion (stated): NON-NULL iff matched-read p < 0.0455
(2 sigma) AND the unmatched read agrees in sign; NULL iff matched p >= 0.0455; UNDETERMINED iff
matched p < 0.0455 but post-match balance fails (|SMD| > 0.10 on any matching covariate) or the
matched and unmatched effects disagree in sign.

Read-only on production artifacts.  Output: m2_results.json in this directory.
"""

import json

import numpy as np
import pandas as pd

RNG = np.random.default_rng(20260805)
N_PERM = 20000
ROOT = "/home/jasper/Repositories/MasterThesisCode"
OUT = f"{ROOT}/results/campaign51_20260728/realistic_20260729/crossterm_instrument/m2_results.json"
CRB = f"{ROOT}/results/run_20260804_postfix/joint_r1/diagnostics/prepared_cramer_rao_bounds.csv"
VENUES = {
    "iiib": f"{ROOT}/results/run_20260804_postfix/iiib/diagnostics/event_likelihoods.csv",
    "joint_r1": f"{ROOT}/results/run_20260804_postfix/joint_r1/diagnostics/event_likelihoods.csv",
}
CHANNELS = {"1d": "combined_no_bh", "2d": "combined_with_bh"}
ALPHA = 0.0455  # 2 sigma, two-sided
SMD_FAIL = 0.10

# ---------------------------------------------------------------- C-4 census
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

# ---------------------------------------------------------------- helpers


def smd(a: np.ndarray, b: np.ndarray) -> float:
    """Standardized mean difference (pooled SD)."""
    sp = np.sqrt(0.5 * (a.var(ddof=1) + b.var(ddof=1)))
    return float((a.mean() - b.mean()) / sp) if sp > 0 else 0.0


def perm_p_mean_diff(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Two-sided label-permutation p for mean(x) - mean(y)."""
    obs = x.mean() - y.mean()
    pooled = np.concatenate([x, y])
    nx = len(x)
    cnt = 0
    for _ in range(N_PERM):
        p = RNG.permutation(pooled)
        if abs(p[:nx].mean() - p[nx:].mean()) >= abs(obs):
            cnt += 1
    return float(obs), (cnt + 1) / (N_PERM + 1)


def signflip_p(diffs: np.ndarray) -> tuple[float, float]:
    """Two-sided sign-flip permutation p for mean of paired diffs."""
    obs = diffs.mean()
    m = len(diffs)
    cnt = 0
    for _ in range(N_PERM):
        s = RNG.choice([-1.0, 1.0], size=m)
        if abs((s * diffs).mean()) >= abs(obs):
            cnt += 1
    return float(obs), (cnt + 1) / (N_PERM + 1)


def summarize(x: np.ndarray) -> dict:
    return {
        "n": int(len(x)),
        "mean": float(x.mean()),
        "median": float(np.median(x)),
        "std": float(x.std(ddof=1)),
        "iqr": float(np.subtract(*np.percentile(x, [75, 25]))),
        "sem": float(x.std(ddof=1) / np.sqrt(len(x))),
    }


# ---------------------------------------------------------------- per-venue read
results: dict = {
    "spec": "CLAIM_HITCHHIKER_INDEPENDENCE_20260805.md.DRAFT lines 550-560 (M-2)",
    "chord_definition": "ln L(h=0.60) - ln L(h=0.73) per event, from combined_no_bh (1D) / combined_with_bh (2D)",
    "census": {
        "source": CRB,
        "n_crb_rows": n,
        "sky_pairs": int(sky.sum()),
        "sky_dl_pairs": n_pairs,
        "overlap_events_of_1590": n_overlap_1590,
        "recipe": "recon_c4_census.py: r=2*sqrt(lam_max(J Sigma J^T)), chord metric, 2-sigma d_L window intersection",
    },
    "matching": {
        "covariates": ["log10_radius_chord", "SNR"],
        "method": "1-NN with replacement, control->overlap, standardized euclidean",
        "w_G_note": "w_G in event_likelihoods.csv is event-independent at fixed h (asserted nunique==1); matched on its per-event drivers (radius, SNR) per [A2]",
    },
    "tests": {
        "unmatched": f"label-permutation on mean difference, two-sided, {N_PERM} perms",
        "matched": f"sign-flip permutation on paired differences, two-sided, {N_PERM} flips",
        "criterion": f"NON-NULL iff matched p < {ALPHA} and unmatched same sign; UNDETERMINED iff matched p < {ALPHA} but post-match |SMD|>{SMD_FAIL} or sign disagreement; else NULL",
    },
    "venues": {},
}

log10_r = np.log10(r)
cov_mat = np.stack([log10_r, snr], axis=1)

for venue, path in VENUES.items():
    el = pd.read_csv(path)
    assert len(el) == 65108, (venue, len(el))
    # w_G event-independence assertion at both anchor h values
    for h0 in (0.60, 0.73):
        assert el.loc[np.isclose(el.h, h0), "w_G"].nunique() == 1, (venue, h0)
    a = el[np.isclose(el.h, 0.60)].set_index("event_idx")
    b = el[np.isclose(el.h, 0.73)].set_index("event_idx")
    ev = np.array(sorted(set(a.index) & set(b.index)))
    assert len(ev) == 1588, (venue, len(ev))

    venue_out: dict = {"n_events": int(len(ev))}
    ov_mask = overlap[ev]  # event_idx == CRB row index
    ct_mask = ~ov_mask
    ov_ev, ct_ev = ev[ov_mask], ev[ct_mask]
    venue_out["n_overlap"] = int(ov_mask.sum())
    venue_out["n_control"] = int(ct_mask.sum())

    # matching (venue-independent covariates, but evaluated-event set is per venue)
    z = (cov_mat[ev] - cov_mat[ev].mean(axis=0)) / cov_mat[ev].std(axis=0, ddof=1)
    z_ov, z_ct = z[ov_mask], z[ct_mask]
    dist2 = ((z_ov[:, None, :] - z_ct[None, :, :]) ** 2).sum(axis=2)
    nn = dist2.argmin(axis=1)  # index into control pool, with replacement
    matched_ct_ev = ct_ev[nn]

    bal = {}
    for k, name in enumerate(["log10_radius_chord", "SNR"]):
        bal[name] = {
            "smd_before": smd(cov_mat[ov_ev, k], cov_mat[ct_ev, k]),
            "smd_after": smd(cov_mat[ov_ev, k], cov_mat[matched_ct_ev, k]),
        }
    bal["n_unique_controls_used"] = int(len(np.unique(nn)))
    venue_out["balance"] = bal
    bal_ok = all(abs(v["smd_after"]) <= SMD_FAIL for v in (bal["log10_radius_chord"], bal["SNR"]))

    venue_out["channels"] = {}
    for ch, col in CHANNELS.items():
        la, lb = a[col], b[col]
        assert (la.loc[ev] > 0).all() and (lb.loc[ev] > 0).all(), (venue, ch)
        chord = pd.Series(np.log(la.loc[ev].to_numpy()) - np.log(lb.loc[ev].to_numpy()), index=ev)
        x_ov = chord.loc[ov_ev].to_numpy()
        x_ct = chord.loc[ct_ev].to_numpy()
        x_mct = chord.loc[matched_ct_ev].to_numpy()

        um_diff, um_p = perm_p_mean_diff(x_ov, x_ct)
        pd_diffs = x_ov - x_mct
        m_diff, m_p = signflip_p(pd_diffs)

        if m_p < ALPHA:
            same_sign = np.sign(m_diff) == np.sign(um_diff)
            call = "NON-NULL" if (same_sign and bal_ok) else "UNDETERMINED"
            direction = "overlap_low_revives_H4" if m_diff < 0 else "overlap_high_revives_H2"
        else:
            call = "NULL"
            direction = None

        venue_out["channels"][ch] = {
            "overlap_stratum": summarize(x_ov),
            "control_stratum": summarize(x_ct),
            "matched_control": summarize(x_mct),
            "unmatched": {
                "mean_diff": um_diff,
                "median_diff": float(np.median(x_ov) - np.median(x_ct)),
                "perm_p": um_p,
            },
            "matched": {
                "mean_paired_diff": m_diff,
                "median_paired_diff": float(np.median(pd_diffs)),
                "paired_diff_std": float(pd_diffs.std(ddof=1)),
                "signflip_p": m_p,
            },
            "call": call,
            "direction_if_nonnull": direction,
        }
    results["venues"][venue] = venue_out

with open(OUT, "w") as f:
    json.dump(results, f, indent=2)
print(json.dumps(results, indent=2))
