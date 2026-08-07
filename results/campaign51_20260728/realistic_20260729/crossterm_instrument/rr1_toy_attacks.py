# ruff: noqa: B023, E731
"""Adversarial numerics/limits re-review, round 1 (2026-08-05) — TOY ATTACKS ONLY.

Independent verification of the FIXED crossterm_instrument.py against:
  A. sigma_z -> 0 pair-level limit at n_shared >= 2 (own config + reviewer
     check-B config), measured convergence rate, sign, and truncation floor.
  B. fixed_quad n=50 adequacy on narrow kernels (production sigma_z values),
     quantified Delta error and minimal adequate n.
  C. catastrophic-cancellation stability of log1p((joint - fact)/(S_i S_j))
     at small shares / small residuals, float64 vs mpmath.
  D. invariance of Delta to global w_g rescaling and per-event likelihood
     rescaling (normalizer cancellation).
  E. guard contract edge cases incl. sigma_z = 0 (catalogue has 6284 zeros).

All references here are written independently (own Gaussian, scipy.integrate
.quad with spike points, mpmath) — nothing reuses the instrument's fixed_quad
path except the object under test itself.
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.special import roots_legendre

sys.path.insert(0, str(Path(__file__).resolve().parent))

from crossterm_instrument import (  # noqa: E402
    BallMember,
    PairSums,
    SharedGalaxyTerm,
    compute_ball_sum,
    compute_pair_sums,
    delta_joint_lnL_nats,
    make_galaxy_z_kernel,
    scaled_shared_latent_mass_joint,
)

OUT = Path(__file__).resolve().parent / "rr1_toy_attacks.json"
results: dict = {}


def gpdf(z, mu, sig):
    z = np.asarray(z, dtype=np.float64)
    return np.exp(-0.5 * ((z - mu) / sig) ** 2) / (sig * math.sqrt(2 * math.pi))


def make_l(mu, sig):
    def f(z):
        return gpdf(z, mu, sig)

    return f


ERF_Z4 = math.erf(4.0 / math.sqrt(2.0))  # truncated-kernel mass in ±4 sigma


# --------------------------------------------------------------------------
# Independent Eq.(31) reference (adaptive quad, spike-aware via points=)
# --------------------------------------------------------------------------


def ref_kernel_norm(z_g, s_z):
    lo, hi = max(z_g - 4 * s_z, 1e-6), z_g + 4 * s_z
    val, _ = quad(
        lambda z: float(gpdf(z, z_g, s_z)), lo, hi, epsabs=1e-300, epsrel=1e-13, limit=500
    )
    return val


def ref_integral(f, lo, hi, z_g, s_z):
    """Adaptive integral of f over [lo, hi] with spike hints at the kernel."""
    if lo >= hi:
        return 0.0
    pts = [p for p in (z_g - 4 * s_z, z_g, z_g + 4 * s_z) if lo < p < hi]
    val, _ = quad(f, lo, hi, points=pts or None, epsabs=1e-300, epsrel=1e-12, limit=800)
    return val


def ref_delta(gals_shared, ball_i_extra, ball_j_extra, l_i, l_j, win_i, win_j):
    """Own Eq.(31) Delta. gals: (z_g, s_z, w_g). Extras are private ball members."""

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


# ==========================================================================
# A. sigma_z -> 0 pair-level limit, n_shared = 2 — OWN configuration
# ==========================================================================

l_i_A = make_l(0.45, 0.045)
l_j_A = make_l(0.43, 0.055)
win_A = (0.12, 0.85)
shared_A = lambda s: [(0.37, s, 0.7), (0.50, s, 2.2)]  # noqa: E731

rows_A = []
sigmas_A = [0.02, 0.01, 0.005, 0.0025, 0.00125, 6.25e-4, 3.125e-4]
for s_z in sigmas_A:
    width = win_A[1] - win_A[0]
    n = int(min(20000, max(800, 4.0 * math.pi * (width / 2) / s_z)))
    d_i, _ = inst_delta(shared_A(s_z), [], [], l_i_A, l_j_A, win_A, win_A, n)
    d_r, _, _ = ref_delta(shared_A(s_z), [], [], l_i_A, l_j_A, win_A, win_A)
    rows_A.append(
        {
            "sigma_z": s_z,
            "quad_n": n,
            "delta_inst": d_i,
            "delta_ref": d_r,
            "rel_diff": abs(d_i - d_r) / max(abs(d_r), 1e-300),
        }
    )

# analytic truncation floor for this config (ball == shared):
Ni0 = {zg: float(l_i_A(zg)) / ERF_Z4 for zg, _, _ in shared_A(1e-9)}
Nj0 = {zg: float(l_j_A(zg)) / ERF_Z4 for zg, _, _ in shared_A(1e-9)}
S_i0 = sum(w * Ni0[zg] for zg, _, w in shared_A(1e-9))
S_j0 = sum(w * Nj0[zg] for zg, _, w in shared_A(1e-9))
floor_A = math.log1p(
    sum(w * w * Ni0[zg] * Nj0[zg] * (ERF_Z4 - 1.0) for zg, _, w in shared_A(1e-9)) / (S_i0 * S_j0)
)
ratios_A = [
    abs(rows_A[k + 1]["delta_inst"] - floor_A) / abs(rows_A[k]["delta_inst"] - floor_A)
    for k in range(len(rows_A) - 1)
]
results["A_own_config"] = {
    "rows": rows_A,
    "floor_prediction_nats": floor_A,
    "floor_over_lnZ": floor_A / math.log(ERF_Z4),
    "halving_ratios_after_floor_subtraction": ratios_A,
    "raw_halving_ratios": [
        abs(rows_A[k + 1]["delta_inst"]) / abs(rows_A[k]["delta_inst"])
        for k in range(len(rows_A) - 1)
    ],
}

# reviewer check-B config verbatim (cross-check against recorded truths)
l_i_B = make_l(0.42, 0.05)
l_j_B = make_l(0.40, 0.06)
win_B = (0.1, 0.9)
truth_B = {
    0.02: 0.07266511574437917,
    0.01: 0.022282229346776816,
    0.005: 0.005871222918222507,
    0.0025: 0.0014603682234266557,
}
rows_B = []
for s_z in [0.02, 0.01, 0.005, 0.0025, 0.00125, 6.25e-4]:
    gals = [(0.35, s_z, 1.0), (0.47, s_z, 3.0)]
    n = int(min(20000, max(800, 4.0 * math.pi * 0.4 / s_z)))
    d_i, _ = inst_delta(gals, [], [], l_i_B, l_j_B, win_B, win_B, n)
    d_r, _, _ = ref_delta(gals, [], [], l_i_B, l_j_B, win_B, win_B)
    row = {
        "sigma_z": s_z,
        "delta_inst": d_i,
        "delta_ref": d_r,
        "rel_diff": abs(d_i - d_r) / max(abs(d_r), 1e-300),
    }
    if s_z in truth_B:
        row["reviewer_truth"] = truth_B[s_z]
        row["rel_vs_reviewer"] = abs(d_i - truth_B[s_z]) / abs(truth_B[s_z])
    rows_B.append(row)
results["A_checkB_config"] = {"rows": rows_B}

# ==========================================================================
# B. fixed_quad n=50 adequacy — production-like windows, narrow kernels
# ==========================================================================

l_i_Q = make_l(0.42, 0.045)
l_j_Q = make_l(0.40, 0.050)
win_i_Q = (0.24, 0.60)  # mu ± 4 sigma_GW (production event window shape)
win_j_Q = (0.20, 0.60)

# worst-case galaxy placement: midpoint of the two central n=50 GL nodes of win_i
nodes50, _ = roots_legendre(50)
zn = 0.5 * (win_i_Q[1] - win_i_Q[0]) * nodes50 + 0.5 * (win_i_Q[1] + win_i_Q[0])
mid = np.searchsorted(zn, 0.41)
z_between = 0.5 * (zn[mid - 1] + zn[mid])  # exactly between two nodes near 0.41

rows_Q = []
for s_z in [0.05, 0.035, 0.02, 0.01, 0.005, 0.002, 0.001, 5.24e-4]:
    for z_g, tag in [(0.41, "generic"), (float(z_between), "between_nodes")]:
        gals = [(z_g, s_z, 1.0)]
        d50, sums50 = inst_delta(gals, [], [], l_i_Q, l_j_Q, win_i_Q, win_j_Q, 50)
        d_r, S_i_r, S_j_r = ref_delta(gals, [], [], l_i_Q, l_j_Q, win_i_Q, win_j_Q)
        rows_Q.append(
            {
                "sigma_z": s_z,
                "z_g": z_g,
                "placement": tag,
                "delta_n50": d50,
                "delta_ref": d_r,
                "delta_abs_err_nats": (d50 - d_r) if math.isfinite(d50) else None,
                "N_i_rel_err": abs(sums50.sum_wN_i - S_i_r) / S_i_r,
                "N_j_rel_err": abs(sums50.sum_wN_j - S_j_r) / S_j_r,
            }
        )
results["B_quadrature_n50"] = rows_Q

# realistic mixed ball: one narrow shared + one wide shared + private wides
mix_rows = []
for s_narrow in [0.002, 0.001, 5.24e-4]:
    shared = [(0.41, s_narrow, 1.0), (0.43, 0.05, 1.0)]
    priv_i = [(0.30, 0.04, 3.0), (0.52, 0.05, 3.0)]
    priv_j = [(0.28, 0.05, 3.0), (0.50, 0.04, 3.0)]
    d50, _ = inst_delta(shared, priv_i, priv_j, l_i_Q, l_j_Q, win_i_Q, win_j_Q, 50)
    d_r, _, _ = ref_delta(shared, priv_i, priv_j, l_i_Q, l_j_Q, win_i_Q, win_j_Q)
    mix_rows.append(
        {"sigma_narrow": s_narrow, "delta_n50": d50, "delta_ref": d_r, "abs_err_nats": d50 - d_r}
    )
results["B_mixed_ball"] = mix_rows

# minimal adequate n at the two narrowest production sigmas
adeq = {}
for s_z in [0.002, 0.001, 5.24e-4]:
    gals = [(0.41, s_z, 1.0)]
    d_r, _, _ = ref_delta(gals, [], [], l_i_Q, l_j_Q, win_i_Q, win_j_Q)
    scan = []
    for n in [50, 100, 200, 400, 800, 1600, 3200, 6400]:
        d_n, _ = inst_delta(gals, [], [], l_i_Q, l_j_Q, win_i_Q, win_j_Q, n)
        scan.append({"n": n, "abs_err_nats": d_n - d_r})
    n_ok = next((r["n"] for r in scan if abs(r["abs_err_nats"]) < 1e-4), None)
    adeq[str(s_z)] = {"delta_ref": d_r, "scan": scan, "min_n_for_1e-4_nats": n_ok}
results["B_min_adequate_n"] = adeq

# wide-kernel J adequacy (joint integrand narrower than either factor)
kern_w = make_galaxy_z_kernel(0.41, 0.05, quad_n=50)
from crossterm_instrument import pair_joint_integral  # noqa: E402

J50 = pair_joint_integral(l_i_Q, l_j_Q, kern_w.rho, win_i_Q, win_j_Q, quad_n=50)
Zw = ref_kernel_norm(0.41, 0.05)
Jref = ref_integral(
    lambda z: float(l_i_Q(z)) * float(l_j_Q(z)) * float(gpdf(z, 0.41, 0.05)) / Zw,
    max(win_i_Q[0], win_j_Q[0]),
    min(win_i_Q[1], win_j_Q[1]),
    0.41,
    0.05,
)
results["B_wide_kernel_J_n50_rel_err"] = abs(J50 - Jref) / Jref

# ==========================================================================
# C. cancellation stability — float64 vs 50-digit Decimal
# ==========================================================================
from decimal import Decimal, getcontext  # noqa: E402

getcontext().prec = 50


def dec_log1p(x: Decimal) -> Decimal:
    return (Decimal(1) + x).ln()


canc = []
for share in [1.0, 1e-3, 1e-6]:
    for r in [1e-2, 1e-4, 6.3e-5, 1e-6, 1e-8]:
        fact = share
        joint = float(share * (1.0 + r))  # float64 rounding as in accumulation
        sums = PairSums(
            S_i=1.0,
            S_j=1.0,
            shared_diag_fact=fact,
            shared_diag_joint=joint,
            sum_wJ=0.0,
            sum_wN_i=0.0,
            sum_wN_j=0.0,
            sum_w=0.0,
            n_shared=1,
        )
        d64 = delta_joint_lnL_nats(sums)
        d_mp = float(dec_log1p(Decimal(joint) - Decimal(fact)))
        d_true = float(dec_log1p(Decimal(share) * Decimal(r)))  # exact intent
        canc.append(
            {
                "share": share,
                "r": r,
                "delta_f64": d64,
                "rel_err_vs_stored_inputs": abs(d64 - d_mp) / max(abs(d_mp), 1e-300),
                "rel_err_vs_intent": abs(d64 - d_true) / max(abs(d_true), 1e-300),
            }
        )
results["C_stored_input_cancellation"] = canc

# accumulation across many shared galaxies (max production n_shared ~ 9898)
rng = np.random.default_rng(42)
n_g = 10000
w = rng.lognormal(0.0, 1.0, n_g)
Ni = rng.lognormal(-2.0, 1.5, n_g)
Nj = rng.lognormal(-2.0, 1.5, n_g)
r_g = 1e-4 * rng.uniform(0.5, 1.5, n_g) * rng.choice([1.0, 1.0, 1.0, -1.0], n_g)
Jg = Ni * Nj * (1.0 + r_g)
fact64 = 0.0
joint64 = 0.0
for k in range(n_g):  # scalar accumulation exactly as compute_pair_sums does
    fact64 += w[k] ** 2 * Ni[k] * Nj[k]
    joint64 += w[k] ** 2 * Jg[k]
target_share = 1e-3
S2 = fact64 / target_share  # denom chosen so fact/denom = 1e-3
d64_acc = math.log1p((joint64 - fact64) / S2)
fact_mp = Decimal(0)
joint_mp = Decimal(0)
for k in range(n_g):
    fact_mp += Decimal(w[k]) ** 2 * Decimal(Ni[k]) * Decimal(Nj[k])
    joint_mp += Decimal(w[k]) ** 2 * Decimal(Jg[k])
d_mp_acc = float(dec_log1p((joint_mp - fact_mp) / Decimal(S2)))
results["C_accumulation_10000_terms"] = {
    "delta_f64": d64_acc,
    "delta_decimal50": d_mp_acc,
    "rel_err": abs(d64_acc - d_mp_acc) / abs(d_mp_acc),
    "note": "share=1e-3, per-galaxy residual ~1e-4 mixed sign, n_g=10000",
}

# ==========================================================================
# D. invariance — global w rescale, per-event likelihood rescale
# ==========================================================================
shared_D = [(0.37, 0.02, 0.7), (0.50, 0.015, 2.2)]
priv_i_D = [(0.30, 0.03, 1.1)]
priv_j_D = [(0.55, 0.025, 0.6)]
d_base, _ = inst_delta(shared_D, priv_i_D, priv_j_D, l_i_A, l_j_A, win_A, win_A, 800)
inv = {}
for c in [1e6, 1e-6, 3.7]:
    sc = lambda g, c=c: [(zg, sz, w * c) for zg, sz, w in g]  # noqa: E731
    d_c, _ = inst_delta(sc(shared_D), sc(priv_i_D), sc(priv_j_D), l_i_A, l_j_A, win_A, win_A, 800)
    inv[f"w_scale_{c:g}"] = {"delta": d_c, "abs_diff": abs(d_c - d_base)}
for a_i, a_j in [(3.7e5, 2.1e-6), (1e-30, 1e30)]:
    li_s = lambda z, a=a_i: a * l_i_A(z)  # noqa: E731
    lj_s = lambda z, a=a_j: a * l_j_A(z)  # noqa: E731
    d_s, _ = inst_delta(shared_D, priv_i_D, priv_j_D, li_s, lj_s, win_A, win_A, 800)
    inv[f"l_scale_{a_i:g}_{a_j:g}"] = {"delta": d_s, "abs_diff": abs(d_s - d_base)}
results["D_invariance"] = {"delta_base": d_base, **inv}

# ==========================================================================
# E. guards — incl. sigma_z = 0 (6284 exact zeros in the parent catalogue)
# ==========================================================================
guards = {}
l_g1 = make_l(0.4, 0.05)
l_g2 = make_l(0.6, 0.05)
kern_g = make_galaxy_z_kernel(0.5, 0.02)
terms_g = [SharedGalaxyTerm(w_g=1.0, rho=kern_g.rho, l_gw_i=l_g1, l_gw_j=l_g2)]
sums_disjoint = compute_pair_sums(terms_g, (0.2, 0.45), (0.55, 0.8))
guards["disjoint_ball_eq_shared"] = delta_joint_lnL_nats(sums_disjoint)  # expect -inf
sums_disj_sup = compute_pair_sums(terms_g, (0.2, 0.45), (0.55, 0.8), S_i=5.0, S_j=5.0)
guards["disjoint_superset_ball"] = delta_joint_lnL_nats(sums_disj_sup)  # finite < 0
guards["n_shared_0"] = delta_joint_lnL_nats(
    compute_pair_sums([], (0.2, 0.45), (0.55, 0.8), S_i=1.0, S_j=1.0)
)  # exactly 0.0
mk = lambda si, sj, f, j: PairSums(
    S_i=si,
    S_j=sj,
    shared_diag_fact=f,
    shared_diag_joint=j,  # noqa: E731
    sum_wJ=0,
    sum_wN_i=0,
    sum_wN_j=0,
    sum_w=0,
    n_shared=1,
)
guards["S_zero"] = str(delta_joint_lnL_nats(mk(0.0, 1.0, 0, 0)))
guards["S_negative"] = str(delta_joint_lnL_nats(mk(-1.0, 1.0, 0, 0)))
guards["S_nan"] = str(delta_joint_lnL_nats(mk(float("nan"), 1.0, 0, 0)))
guards["S_inf"] = str(delta_joint_lnL_nats(mk(float("inf"), 1.0, 0, 0)))
guards["corrected_nonpos"] = str(delta_joint_lnL_nats(mk(1.0, 1.0, 2.0, 0.5)))
guards["denom_underflow"] = str(delta_joint_lnL_nats(mk(1e-170, 1e-170, 0.0, 1e-100)))
guards["x_overflow"] = str(delta_joint_lnL_nats(mk(1e-160, 1e-160, 0.0, 1.0)))

# sigma_z = 0 chain (production has 6284 such galaxies in the parent catalogue)
import warnings  # noqa: E402

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    kern_z0 = make_galaxy_z_kernel(0.4, 0.0)
    guards["sigma0_z_norm"] = kern_z0.z_norm
    guards["sigma0_rho_at_zg"] = str(float(kern_z0.rho(np.array([0.4]))[0]))
    t0 = [SharedGalaxyTerm(w_g=1.0, rho=kern_z0.rho, l_gw_i=l_g1, l_gw_j=l_g1)]
    s0 = compute_pair_sums(t0, (0.2, 0.6), (0.2, 0.6))
    guards["sigma0_delta"] = str(delta_joint_lnL_nats(s0))
    guards["sigma0_S_i"] = str(s0.S_i)
results["E_guards"] = guards

# ==========================================================================
# extra: independent adaptive-quad check of the scaled 2D closed form (10 draws)
# ==========================================================================
rng2 = np.random.default_rng(7)
worst = 0.0
for _ in range(10):
    mu_i, mu_j = rng2.normal(0.5, 0.1, 2)
    a_i, a_j = rng2.uniform(0.3, 2.0, 2)
    s2i, s2j = rng2.uniform(1e-4, 1e-2, 2)
    Mg = rng2.uniform(0.2, 1.5)
    sM = rng2.uniform(0.05, 2.0)
    closed = float(scaled_shared_latent_mass_joint(mu_i, mu_j, a_i, a_j, s2i, s2j, Mg, sM**2))

    def integrand(M, mu_i=mu_i, mu_j=mu_j, a_i=a_i, a_j=a_j, s2i=s2i, s2j=s2j, Mg=Mg, sM=sM):
        return (
            math.exp(-0.5 * (mu_i - M * a_i) ** 2 / s2i)
            / math.sqrt(2 * math.pi * s2i)
            * math.exp(-0.5 * (mu_j - M * a_j) ** 2 / s2j)
            / math.sqrt(2 * math.pi * s2j)
            * math.exp(-0.5 * (M - Mg) ** 2 / sM**2)
            / math.sqrt(2 * math.pi * sM**2)
        )

    # spike hints: the product-of-two-event factor peaks near mu_e/a_e
    pts = sorted({Mg, mu_i / a_i, mu_j / a_j})
    refv, _ = quad(
        integrand,
        Mg - 15 * sM,
        Mg + 15 * sM,
        points=[p for p in pts if Mg - 15 * sM < p < Mg + 15 * sM] or None,
        epsabs=1e-300,
        epsrel=1e-12,
        limit=800,
    )
    if refv > 0:
        worst = max(worst, abs(closed - refv) / refv)
results["extra_2d_closed_form_quad_worst_rel"] = worst

with open(OUT, "w") as fh:
    json.dump(results, fh, indent=1, default=str)
print(json.dumps(results, indent=1, default=str))
