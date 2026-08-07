# ruff: noqa: E741
"""ADVERSARIAL ADJUDICATION of M-2 (independent recompute; does NOT reuse
m2_overlap_stratified.py internals).

Recomputes from raw CSVs:
  1. C-4 census pair set (sky chord-overlap + 2-sigma d_L window) -> expect
     1620 sky pairs, 279 sky+dL pairs, 385 overlap events of 1590.
  2. Per-event chords ln L(0.60) - ln L(0.73) for combined_no_bh (1d) and
     combined_with_bh (2d), both venues.
  3. Unmatched stratum difference (overlap vs control) + permutation p.
  4. [A2] matched read: 1-NN-with-replacement matching (scipy cKDTree) on
     standardized (log10 radius chord, SNR); SMD before/after; sign-flip p;
     cluster-robust sign-flip (pairs sharing a control flip together).

Independent implementation choices: analytic 2x2 eigenvalue (not eigvalsh),
cKDTree NN (not brute-force argmin), own permutation loops, RNG seed 777777.
Read-only on production artifacts; writes m2_adjudication.json here.
"""

import hashlib
import json

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

RNG = np.random.default_rng(777777)
N_PERM = 40000
ROOT = "/home/jasper/Repositories/MasterThesisCode"
DIAG = f"{ROOT}/results/run_20260804_postfix"
OUT = (
    f"{ROOT}/results/campaign51_20260728/realistic_20260729/"
    "crossterm_instrument/m2_adjudication.json"
)

# ---------------------------------------------------------------- CRB / census
crb_paths = {
    v: f"{DIAG}/{v}/diagnostics/prepared_cramer_rao_bounds.csv" for v in ("iiib", "joint_r1")
}
md5 = {v: hashlib.md5(open(p, "rb").read()).hexdigest() for v, p in crb_paths.items()}

crb = pd.read_csv(crb_paths["joint_r1"])
n = len(crb)
theta = crb["qS"].to_numpy()
phi = crb["phiS"].to_numpy()
a11 = crb["delta_phiS_delta_phiS"].to_numpy()  # s_phi^2
a22 = crb["delta_qS_delta_qS"].to_numpy()  # s_theta^2
a12 = crb["delta_phiS_delta_qS"].to_numpy()
dl = crb["luminosity_distance"].to_numpy()
sdl = np.sqrt(crb["delta_luminosity_distance_delta_luminosity_distance"].to_numpy())
snr = crb["SNR"].to_numpy()

# M = J Sigma J^T with J = diag(|sin theta|, 1):
s = np.abs(np.sin(theta))
m11 = s * s * a11
m22 = a22
m12 = s * a12
# analytic lambda_max of symmetric 2x2
tr = m11 + m22
disc = np.sqrt((m11 - m22) ** 2 + 4.0 * m12 * m12)
lam_max = 0.5 * (tr + disc)
r = 2.0 * np.sqrt(np.clip(lam_max, 0.0, None))

# unit vectors, pairwise chord
sin_t = np.sin(theta)
xyz = np.column_stack([sin_t * np.cos(phi), sin_t * np.sin(phi), np.cos(theta)])
# vectorized pairwise via dot products: chord^2 = 2 - 2 cos
g = np.clip(xyz @ xyz.T, -1.0, 1.0)
chord_d = np.sqrt(np.maximum(2.0 - 2.0 * g, 0.0))
I, J = np.triu_indices(n, k=1)
sky_mask = chord_d[I, J] <= r[I] + r[J]
n_sky = int(sky_mask.sum())
si, sj = I[sky_mask], J[sky_mask]
lo, hi = dl - 2.0 * sdl, dl + 2.0 * sdl
dlw = np.maximum(lo[si], lo[sj]) <= np.minimum(hi[si], hi[sj])
n_skydl = int(dlw.sum())
overlap1590 = np.zeros(n, dtype=bool)
overlap1590[si[dlw]] = True
overlap1590[sj[dlw]] = True
n_overlap = int(overlap1590.sum())

census = {
    "crb_rows": n,
    "crb_md5_identical_across_venues": md5["iiib"] == md5["joint_r1"],
    "crb_md5": md5,
    "sky_pairs": n_sky,
    "sky_dl_pairs": n_skydl,
    "overlap_events": n_overlap,
    "expected": {"sky_pairs": 1620, "sky_dl_pairs": 279, "overlap_events": 385},
}

# ---------------------------------------------------------------- helpers


def smd(a: np.ndarray, b: np.ndarray) -> float:
    sp = np.sqrt(0.5 * (np.var(a, ddof=1) + np.var(b, ddof=1)))
    return float((np.mean(a) - np.mean(b)) / sp) if sp > 0 else 0.0


def perm_p_meandiff(x: np.ndarray, y: np.ndarray) -> float:
    obs = abs(x.mean() - y.mean())
    pool = np.concatenate([x, y])
    nx = len(x)
    exc = 0
    for _ in range(N_PERM):
        RNG.shuffle(pool)
        if abs(pool[:nx].mean() - pool[nx:].mean()) >= obs:
            exc += 1
    return (exc + 1) / (N_PERM + 1)


def signflip_p(d: np.ndarray) -> float:
    obs = abs(d.mean())
    m = len(d)
    signs = RNG.integers(0, 2, size=(N_PERM, m)) * 2 - 1
    stats = np.abs(signs @ d) / m
    return float((np.sum(stats >= obs) + 1) / (N_PERM + 1))


def cluster_signflip_p(d: np.ndarray, cluster_ids: np.ndarray) -> float:
    """Flip all pairs sharing a control (cluster) together."""
    obs = abs(d.mean())
    uniq, inv = np.unique(cluster_ids, return_inverse=True)
    k = len(uniq)
    # per-cluster sums of paired diffs
    csum = np.zeros(k)
    np.add.at(csum, inv, d)
    m = len(d)
    signs = RNG.integers(0, 2, size=(N_PERM, k)) * 2 - 1
    stats = np.abs(signs @ csum) / m
    return float((np.sum(stats >= obs) + 1) / (N_PERM + 1))


# ---------------------------------------------------------------- venues
result: dict = {"census": census, "venues": {}}
covariates = np.column_stack([np.log10(r), snr])

for venue in ("iiib", "joint_r1"):
    el = pd.read_csv(f"{DIAG}/{venue}/diagnostics/event_likelihoods.csv")
    n_rows = len(el)
    # w_G event-independence check (independent of original's assertion)
    wg_check = {}
    for h0 in (0.60, 0.73):
        sub = el[np.isclose(el["h"], h0)]
        wg_check[f"h={h0}"] = {
            "nunique_w_G": int(sub["w_G"].nunique()),
            "w_G_value": float(sub["w_G"].iloc[0]),
            "n_events": int(len(sub)),
        }
    piv60 = el[np.isclose(el["h"], 0.60)].set_index("event_idx")
    piv73 = el[np.isclose(el["h"], 0.73)].set_index("event_idx")
    ev = np.array(sorted(set(piv60.index).intersection(piv73.index)))
    dropped = sorted(set(range(n)) - set(ev))

    ov_ev = ev[overlap1590[ev]]
    ct_ev = ev[~overlap1590[ev]]

    # matching: standardize covariates over evaluated events
    C = covariates[ev]
    z = (C - C.mean(axis=0)) / C.std(axis=0, ddof=1)
    z_ov = z[overlap1590[ev]]
    z_ct = z[~overlap1590[ev]]
    tree = cKDTree(z_ct)
    _, nn = tree.query(z_ov, k=1)
    matched_ct = ct_ev[nn]

    balance = {}
    for k_i, name in enumerate(("log10_radius_chord", "SNR")):
        balance[name] = {
            "smd_before": round(smd(covariates[ov_ev, k_i], covariates[ct_ev, k_i]), 4),
            "smd_after": round(smd(covariates[ov_ev, k_i], covariates[matched_ct, k_i]), 4),
        }
    n_unique_controls = int(len(np.unique(nn)))

    vout: dict = {
        "likelihood_rows": n_rows,
        "w_G_constancy": wg_check,
        "n_evaluated_events": int(len(ev)),
        "dropped_event_idx": dropped,
        "n_overlap": int(len(ov_ev)),
        "n_control": int(len(ct_ev)),
        "balance": balance,
        "n_unique_controls_used": n_unique_controls,
        "channels": {},
    }

    for ch, col in (("1d", "combined_no_bh"), ("2d", "combined_with_bh")):
        L60 = piv60.loc[ev, col].to_numpy()
        L73 = piv73.loc[ev, col].to_numpy()
        assert (L60 > 0).all() and (L73 > 0).all()
        chord = np.log(L60) - np.log(L73)
        cs = pd.Series(chord, index=ev)
        x_ov = cs.loc[ov_ev].to_numpy()
        x_ct = cs.loc[ct_ev].to_numpy()
        x_m = cs.loc[matched_ct].to_numpy()
        pdiff = x_ov - x_m

        vout["channels"][ch] = {
            "overlap_mean": round(float(x_ov.mean()), 5),
            "control_mean": round(float(x_ct.mean()), 5),
            "unmatched_diff": round(float(x_ov.mean() - x_ct.mean()), 5),
            "unmatched_perm_p": perm_p_meandiff(x_ov.copy(), x_ct.copy()),
            "matched_mean_paired_diff": round(float(pdiff.mean()), 5),
            "matched_median_paired_diff": round(float(np.median(pdiff)), 5),
            "matched_paired_diff_std": round(float(pdiff.std(ddof=1)), 4),
            "n_pairs": int(len(pdiff)),
            "matched_signflip_p": signflip_p(pdiff),
            "cluster_robust_signflip_p": cluster_signflip_p(pdiff, nn),
        }
    result["venues"][venue] = vout

with open(OUT, "w") as f:
    json.dump(result, f, indent=2)
print(json.dumps(result, indent=2))
