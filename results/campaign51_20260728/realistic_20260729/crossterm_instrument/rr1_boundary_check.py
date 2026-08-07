# ruff: noqa: B023
"""RR1 boundary check: where exactly does fixed_quad n=50 become safe?

(a) Census refinement: min shared-galaxy sigma_z per venue/channel and counts at
    thresholds {2e-3, 5e-3, 1e-2, 1.5e-2, 2e-2}; identify the narrow galaxy in
    the affected pair (114, 1035) joint_r1/1d.
(b) Real event-window widths: z-window [dist_to_redshift(d -/+ 4 sigma_d, h)]
    for all events in the 279 pairs at the four floor h values (pure function
    evaluation on CRB scalars — no instrument run).
(c) Toy calibration of the n=50 Delta error vs (window width W, sigma_z),
    including the affected pair's actual geometry.

Zero instrument compute on production data; reads + pure math only.
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from crossterm_instrument import (  # noqa: E402
    CRB_PATH,
    VENUE_CONFIGS,
    BallMember,
    SharedGalaxyTerm,
    c4_pair_census,
    compute_ball_sum,
    compute_pair_sums,
    delta_joint_lnL_nats,
    load_ball_sets,
    load_filtered_events,
    make_galaxy_z_kernel,
)
from scipy.integrate import quad  # noqa: E402

from master_thesis_code.galaxy_catalogue.handler import (  # noqa: E402
    _empiric_stellar_mass_to_BH_mass_relation,
    _mass_redshift_prune_mask,
)

# ---- helpers duplicated from rr1_toy_attacks.py / rr1_ball_sigma_census.py
# (those are scripts; importing them would re-execute their whole payload) ----

M_MIN, M_MAX, Z_MAX = 1e4, 1e7, 1.5
COLS = ["REDSHIFT", "REDSHIFT_ERROR", "STELLAR_MASS", "STELLAR_MASS_ERROR", "REDSHIFT_FLAG"]


def pruned_sigma_z(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    sig_parts: list[np.ndarray] = []
    flag_parts: list[np.ndarray] = []
    for chunk in pd.read_csv(
        csv_path,
        names=["RA", "DEC", "B", *COLS],
        usecols=[3, 4, 5, 6, 7],
        chunksize=2_000_000,
    ):
        bh_mass, bh_err = _empiric_stellar_mass_to_BH_mass_relation(
            chunk["STELLAR_MASS"], chunk["STELLAR_MASS_ERROR"]
        )
        has_mass = ~pd.isna(bh_mass)
        keep = has_mass & _mass_redshift_prune_mask(
            bh_mass, bh_err, chunk["REDSHIFT"], chunk["REDSHIFT_ERROR"], M_MIN, M_MAX, Z_MAX
        )
        sig_parts.append(chunk.loc[keep, "REDSHIFT_ERROR"].to_numpy(dtype=np.float64))
        flag_parts.append(chunk.loc[keep, "REDSHIFT_FLAG"].to_numpy(dtype=np.float64))
    return np.concatenate(sig_parts), np.concatenate(flag_parts)


def gpdf(z, mu, sig):
    z = np.asarray(z, dtype=np.float64)
    return np.exp(-0.5 * ((z - mu) / sig) ** 2) / (sig * math.sqrt(2 * math.pi))


def make_l(mu, sig):
    def f(z):
        return gpdf(z, mu, sig)

    return f


def ref_kernel_norm(z_g, s_z):
    lo, hi = max(z_g - 4 * s_z, 1e-6), z_g + 4 * s_z
    val, _ = quad(
        lambda z: float(gpdf(z, z_g, s_z)), lo, hi, epsabs=1e-300, epsrel=1e-13, limit=500
    )
    return val


def ref_integral(f, lo, hi, z_g, s_z):
    if lo >= hi:
        return 0.0
    pts = [p for p in (z_g - 4 * s_z, z_g, z_g + 4 * s_z) if lo < p < hi]
    val, _ = quad(f, lo, hi, points=pts or None, epsabs=1e-300, epsrel=1e-12, limit=800)
    return val


def ref_delta(gals_shared, ball_i_extra, ball_j_extra, l_i, l_j, win_i, win_j):
    def N(l_fn, z_g, s_z, win):
        Z = ref_kernel_norm(z_g, s_z)
        return (
            ref_integral(
                lambda z: float(l_fn(z)) * float(gpdf(z, z_g, s_z)), win[0], win[1], z_g, s_z
            )
            / Z
        )

    lo, hi = max(win_i[0], win_j[0]), min(win_i[1], win_j[1])
    S_i = sum(w * N(l_i, zg, sz, win_i) for zg, sz, w in gals_shared + ball_i_extra)
    S_j = sum(w * N(l_j, zg, sz, win_j) for zg, sz, w in gals_shared + ball_j_extra)
    corrected = S_i * S_j
    for zg, sz, w in gals_shared:
        Ni = N(l_i, zg, sz, win_i)
        Nj = N(l_j, zg, sz, win_j)
        Z = ref_kernel_norm(zg, sz)
        J = (
            ref_integral(
                lambda z: float(l_i(z)) * float(l_j(z)) * float(gpdf(z, zg, sz)), lo, hi, zg, sz
            )
            / Z
        )
        corrected += w * w * (J - Ni * Nj)
    return math.log(corrected) - math.log(S_i * S_j), S_i, S_j


def inst_delta(gals_shared, ball_i_extra, ball_j_extra, l_i, l_j, win_i, win_j, n):
    terms = []
    for zg, sz, w in gals_shared:
        kern = make_galaxy_z_kernel(zg, sz, quad_n=max(n, 50))
        terms.append(SharedGalaxyTerm(w_g=w, rho=kern.rho, l_gw_i=l_i, l_gw_j=l_j))
    mem_i = [
        BallMember(w_g=w, rho=make_galaxy_z_kernel(zg, sz, quad_n=max(n, 50)).rho, l_ev=l_i)
        for zg, sz, w in gals_shared + ball_i_extra
    ]
    mem_j = [
        BallMember(w_g=w, rho=make_galaxy_z_kernel(zg, sz, quad_n=max(n, 50)).rho, l_ev=l_j)
        for zg, sz, w in gals_shared + ball_j_extra
    ]
    S_i = compute_ball_sum(mem_i, win_i, quad_n=n)
    S_j = compute_ball_sum(mem_j, win_j, quad_n=n)
    sums = compute_pair_sums(terms, win_i, win_j, quad_n=n, S_i=S_i, S_j=S_j)
    return delta_joint_lnL_nats(sums), sums


OUT = Path(__file__).resolve().parent / "rr1_boundary_check.json"
results: dict = {}

crb_all = pd.read_csv(CRB_PATH)
crb_filtered = load_filtered_events(CRB_PATH)
pairs_all, _deg = c4_pair_census(crb_all)
fidx = set(int(i) for i in crb_filtered.index)
pairs = [(i, j) for (i, j) in pairs_all if i in fidx and j in fidx]
needed = sorted({e for p in pairs for e in p})

# ---------------------------------------------------------------- (a) census
TH = [2e-3, 5e-3, 1e-2, 1.5e-2, 2e-2]
for venue, cfg in VENUE_CONFIGS.items():
    sig, _flag = pruned_sigma_z(cfg["catalogue"])
    ball_1d, ball_2d = load_ball_sets(cfg["frozeng_dir"])
    vres = {}
    for ch, balls in (("1d", ball_1d), ("2d", ball_2d)):
        shared_min = None
        counts = {f"pairs_with_shared_lt_{t:g}": 0 for t in TH}
        for i, j in pairs:
            shared = balls.get(i, set()) & balls.get(j, set())
            if not shared:
                continue
            ssig = sig[np.array(sorted(shared), dtype=np.int64)]
            m = float(ssig.min())
            shared_min = m if shared_min is None else min(shared_min, m)
            for t in TH:
                if np.any(ssig < t):
                    counts[f"pairs_with_shared_lt_{t:g}"] += 1
        vres[ch] = {"min_shared_sigma_z": shared_min, **counts}
    # identify the narrow shared galaxy of (114, 1035) in this venue's 1d
    shared_np = ball_1d.get(114, set()) & ball_1d.get(1035, set())
    if shared_np:
        idx = np.array(sorted(shared_np), dtype=np.int64)
        ss = sig[idx]
        k = int(np.argmin(ss))
        vres["pair_114_1035_1d"] = {
            "n_shared": int(idx.size),
            "min_sigma_z": float(ss[k]),
            "argmin_catalog_index": int(idx[k]),
            "n_lt_1e-2": int(np.sum(ss < 1e-2)),
            "n_lt_1.5e-2": int(np.sum(ss < 1.5e-2)),
        }
    results[venue] = vres
    del sig

# ------------------------------------------------------- (b) window widths
from master_thesis_code.physical_relations import dist_to_redshift  # noqa: E402

H_GRID = [0.60, 0.73, 0.81, 0.86]
widths = []
for ev in needed:
    row = crb_all.loc[ev]
    d = float(row["luminosity_distance"])
    sd = float(np.sqrt(row["delta_luminosity_distance_delta_luminosity_distance"]))
    for h in H_GRID:
        lo = float(dist_to_redshift(d - 4.0 * sd, h=h))
        hi = float(dist_to_redshift(d + 4.0 * sd, h=h))
        widths.append({"event": ev, "h": h, "z_lo": lo, "z_hi": hi, "width": hi - lo})
wa = np.array([w["width"] for w in widths])
results["event_window_widths"] = {
    "n": int(wa.size),
    "min": float(wa.min()),
    "median": float(np.median(wa)),
    "p90": float(np.percentile(wa, 90)),
    "max": float(wa.max()),
}
w_114 = [w for w in widths if w["event"] == 114]
w_1035 = [w for w in widths if w["event"] == 1035]
results["windows_114"] = w_114
results["windows_1035"] = w_1035


# ------------------------------------------------- (c) toy error vs (W, sigma)
def toy_error(W: float, s_z: float) -> dict:
    mu_i, mu_j = 0.02 * W, -0.02 * W  # slight offsets around center c
    c = 0.45
    l_i = make_l(c + mu_i, W / 8.0)
    l_j = make_l(c + mu_j, W / 7.2)
    win_i = (c - W / 2, c + W / 2)
    win_j = (c - 0.55 * W, c + 0.45 * W)
    gals = [(c - 0.01 * W, s_z, 1.0)]
    d50, _ = inst_delta(gals, [], [], l_i, l_j, win_i, win_j, 50)
    d_r, _, _ = ref_delta(gals, [], [], l_i, l_j, win_i, win_j)
    return {
        "W": W,
        "sigma_z": s_z,
        "ratio_W_over_sigma": W / s_z,
        "delta_n50": d50,
        "delta_ref": d_r,
        "abs_err_nats": d50 - d_r,
    }


grid_rows = []
for W in [0.15, 0.36, 0.55]:
    for s_z in [5e-3, 8e-3, 1e-2, 1.2e-2, 1.5e-2, 2e-2]:
        grid_rows.append(toy_error(W, s_z))
results["toy_error_grid"] = grid_rows

# affected pair actual geometry: width from (b) at each h, sigma = 1.9636e-3
aff = []
for h, wi, wj in [(w1["h"], w1["width"], w2["width"]) for w1, w2 in zip(w_114, w_1035)]:
    W = max(wi, wj)
    aff.append({"h": h, **toy_error(W, 1.9636170795231e-3)})
results["affected_pair_toy_scale"] = aff

with open(OUT, "w") as fh:
    json.dump(results, fh, indent=1)
print(json.dumps(results, indent=1))
