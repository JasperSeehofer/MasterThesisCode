# ruff: noqa: B023
"""ADVERSARIAL MATH RE-REVIEW round 1 (2026-08-05) — independent brute force.

Fully independent reference implementation (own Gaussian pdf, own Z_g via
scipy adaptive quad over the production 4-sigma host window, own nested
z x M quadrature for the 2D channel). The instrument's functions are used
ONLY as the device under test. NEW constructions beyond the prior review:

  N1  superset balls + UNEQUAL event windows + NONCONSTANT w_pop kernel weight
      + heterogeneous sigma_z, n_shared = 3 (the prior review never scored
      Delta with a nonconstant w_pop or unequal windows).
  N2  the prior reviewer's check-C discriminating universes, re-scored against
      MY OWN reference (not their brute_force_pair).
  N3  full 2D pair with z-DEPENDENT mu_cond,e(z) (linear in z, different
      slopes), UNEQUAL detector masses (a_i(z) != a_j(z)), sigma_M = 0.6,
      w_pop-weighted kernel — reference is nested adaptive quad over (z, M).
  N4  sigma_M = 0 2D reduction: J(joint_l at sM=0) must equal the factorized
      1D-type J of l_e = gw_e * mz_e exactly.
  N5  invariance attacks: global rescale of all w_g (c=7.3) and per-event
      likelihood rescale (c_i=0.2, c_j=41.0) must leave Delta invariant
      (the normalizer-cancellation claim, raw-sum side).
  N6  edge-overlap pair: shared galaxy centred at the window-intersection
      EDGE -> J suppressed, finite NEGATIVE Delta; brute-force scored.
  N7  severe-cancellation stress: shared set = 99.99% of both balls' weight,
      sigma_z small -> Delta near the truncation floor; brute-force scored.
  N8  mixture-composition algebra: the emitted-row formula
      ln[1 + w_Gi w_Gj Lcat_i Lcat_j (e^D - 1)/(comb_i comb_j)] must equal
      ln(joint_corrected_mixture) - ln(comb_i comb_j) for the mixture
      identity comb = w_G Lcat + (1-w_G) Lcomp (numeric, random draws).
  N9  floor constant: ln(erf(4/sqrt(2))) vs the documented 6.334e-5 nats.

Run:
    cd /home/jasper/Repositories/MasterThesisCode && uv run python \
        results/campaign51_20260728/realistic_20260729/crossterm_instrument/rereview_round1_independent_checks.py
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import quad

sys.path.insert(0, str(Path(__file__).resolve().parent))

from crossterm_instrument import (  # noqa: E402
    BallMember,
    SharedGalaxyTerm,
    compute_ball_sum,
    compute_pair_sums,
    delta_joint_lnL_nats,
    make_galaxy_z_kernel,
    scaled_shared_latent_mass_joint,
)

out = {}
failures = []


# ----------------------------------------------------------------------------
# my own reference pieces (no instrument code)
# ----------------------------------------------------------------------------
def npdf(x, m, s):
    return math.exp(-0.5 * ((x - m) / s) ** 2) / (s * math.sqrt(2.0 * math.pi))


def ref_rho(z_g, s_z, w_pop=None):
    """rho(z) = N(z;z_g,s_z) w_pop(z) / Z, Z over [max(z_g-4s,1e-6), z_g+4s]."""
    lo = max(z_g - 4.0 * s_z, 1e-6)
    hi = z_g + 4.0 * s_z
    if w_pop is None:
        Z, _ = quad(lambda z: npdf(z, z_g, s_z), lo, hi, epsabs=1e-15, limit=400)
    else:
        Z, _ = quad(lambda z: npdf(z, z_g, s_z) * w_pop(z), lo, hi, epsabs=1e-15, limit=400)

    def rho(z):
        base = npdf(z, z_g, s_z)
        if w_pop is not None:
            base *= w_pop(z)
        return base / Z

    return rho


def ref_N(l_fn, rho, win):
    v, _ = quad(lambda z: l_fn(z) * rho(z), win[0], win[1], epsabs=1e-15, limit=400)
    return v


def ref_J(joint_fn, rho, win_i, win_j):
    lo, hi = max(win_i[0], win_j[0]), min(win_i[1], win_j[1])
    if lo >= hi:
        return 0.0
    v, _ = quad(lambda z: joint_fn(z) * rho(z), lo, hi, epsabs=1e-16, limit=400)
    return v


def ref_delta(ball_i, ball_j, shared_keys, l_of_i, l_of_j, joint_of, win_i, win_j):
    """ball_*: dict key -> (w_g, rho). l_of_e(key) -> callable. joint_of(key) -> callable.

    Delta = ln[(S_i S_j - Sum_sh w^2 N_i N_j + Sum_sh w^2 J) / (S_i S_j)].
    """
    S_i = sum(w * ref_N(l_of_i(k), rho, win_i) for k, (w, rho) in ball_i.items())
    S_j = sum(w * ref_N(l_of_j(k), rho, win_j) for k, (w, rho) in ball_j.items())
    corr = 0.0
    for k in shared_keys:
        w, rho = ball_i[k]
        N_i = ref_N(l_of_i(k), rho, win_i)
        N_j = ref_N(l_of_j(k), rho, win_j)
        J = ref_J(joint_of(k), rho, win_i, win_j)
        corr += w * w * (J - N_i * N_j)
    return math.log1p(corr / (S_i * S_j)), S_i, S_j


def gauss_l(mu, s):
    def f(z):
        z = np.asarray(z, dtype=np.float64)
        return np.exp(-0.5 * ((z - mu) / s) ** 2) / (s * math.sqrt(2.0 * math.pi))

    return f


def toy_w_pop_np(z):
    z = np.asarray(z, dtype=np.float64)
    return z**2 / (1.0 + z)


def toy_w_pop_sc(z):
    return z * z / (1.0 + z)


# ----------------------------------------------------------------------------
# N1: superset balls + unequal windows + w_pop kernel + heterogeneous sigma_z
# ----------------------------------------------------------------------------
win_i = (0.18, 0.72)
win_j = (0.25, 0.88)
l_i = gauss_l(0.44, 0.05)
l_j = gauss_l(0.41, 0.07)
shared_gals = [(0.38, 0.020, 1.4), (0.43, 0.035, 0.6), (0.49, 0.012, 2.2)]
priv_i = [(0.30, 0.030, 1.1), (0.36, 0.015, 0.5)]
priv_j = [(0.55, 0.022, 1.9)]

# device
quad_n = 800
terms = []
for z_g, s_z, w_g in shared_gals:
    kern = make_galaxy_z_kernel(z_g, s_z, w_pop_eff=toy_w_pop_np, quad_n=quad_n)
    terms.append(SharedGalaxyTerm(w_g=w_g, rho=kern.rho, l_gw_i=l_i, l_gw_j=l_j))
members_i = [
    BallMember(
        w_g=w, rho=make_galaxy_z_kernel(z, s, w_pop_eff=toy_w_pop_np, quad_n=quad_n).rho, l_ev=l_i
    )
    for z, s, w in shared_gals + priv_i
]
members_j = [
    BallMember(
        w_g=w, rho=make_galaxy_z_kernel(z, s, w_pop_eff=toy_w_pop_np, quad_n=quad_n).rho, l_ev=l_j
    )
    for z, s, w in shared_gals + priv_j
]
S_i_dev = compute_ball_sum(members_i, win_i, quad_n=quad_n)
S_j_dev = compute_ball_sum(members_j, win_j, quad_n=quad_n)
sums = compute_pair_sums(terms, win_i, win_j, quad_n=quad_n, S_i=S_i_dev, S_j=S_j_dev)
d_dev = delta_joint_lnL_nats(sums)

# reference (fully my own)
ball_i_ref = {f"s{k}": (w, ref_rho(z, s, toy_w_pop_sc)) for k, (z, s, w) in enumerate(shared_gals)}
ball_i_ref.update(
    {f"pi{k}": (w, ref_rho(z, s, toy_w_pop_sc)) for k, (z, s, w) in enumerate(priv_i)}
)
ball_j_ref = {f"s{k}": (w, ref_rho(z, s, toy_w_pop_sc)) for k, (z, s, w) in enumerate(shared_gals)}
ball_j_ref.update(
    {f"pj{k}": (w, ref_rho(z, s, toy_w_pop_sc)) for k, (z, s, w) in enumerate(priv_j)}
)


def sc(l_np):
    return lambda z: float(l_np(np.array([z]))[0])


d_ref, S_i_ref, S_j_ref = ref_delta(
    ball_i_ref,
    ball_j_ref,
    [f"s{k}" for k in range(3)],
    lambda k: sc(l_i),
    lambda k: sc(l_j),
    lambda k: lambda z: sc(l_i)(z) * sc(l_j)(z),
    win_i,
    win_j,
)
ok1 = (
    abs(d_dev - d_ref) < max(1e-9, 1e-7 * abs(d_ref))
    and abs(S_i_dev - S_i_ref) / S_i_ref < 1e-9
    and abs(S_j_dev - S_j_ref) / S_j_ref < 1e-9
)
out["N1_superset_wpop_unequal_windows"] = {
    "delta_device": d_dev,
    "delta_reference": d_ref,
    "abs_diff": abs(d_dev - d_ref),
    "S_i_rel": abs(S_i_dev - S_i_ref) / S_i_ref,
    "S_j_rel": abs(S_j_dev - S_j_ref) / S_j_ref,
    "ok": ok1,
}
if not ok1:
    failures.append("N1")

# ----------------------------------------------------------------------------
# N2: prior reviewer's check-C universes, scored against MY reference
# ----------------------------------------------------------------------------
l_i2 = gauss_l(0.42, 0.05)
l_j2 = gauss_l(0.40, 0.06)
win2 = (0.1, 0.9)
gals_z = [(0.34, 0.02), (0.38, 0.025), (0.42, 0.03), (0.46, 0.025), (0.50, 0.02)]
# rebuild the nullspace weights exactly as the reviewer did (device kernels for
# the CONSTRUCTION only; the SCORING reference below is mine)
from crossterm_instrument import pair_joint_integral, per_galaxy_numerator  # noqa: E402

per = []
for z_g, s_z in gals_z:
    kern = make_galaxy_z_kernel(z_g, s_z, quad_n=800)
    per.append(
        (
            per_galaxy_numerator(l_i2, kern.rho, *win2, quad_n=800),
            per_galaxy_numerator(l_j2, kern.rho, *win2, quad_n=800),
            pair_joint_integral(l_i2, l_j2, kern.rho, win2, win2, quad_n=800),
        )
    )
A = np.array([[1.0] * 5, [p[0] for p in per], [p[1] for p in per], [p[2] for p in per]])
w0 = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
_, _, Vt = np.linalg.svd(A)
w1 = w0 + 0.8 * Vt[-1] / np.max(np.abs(Vt[-1]))
assert np.all(w1 > 0)
res2 = {}
ok2 = True
for tag, wv in (("u1", w0), ("u2", w1)):
    gals = [(z, s, float(w)) for (z, s), w in zip(gals_z, wv)]
    terms2 = []
    for z_g, s_z, w_g in gals:
        kern = make_galaxy_z_kernel(z_g, s_z, quad_n=800)
        terms2.append(SharedGalaxyTerm(w_g=w_g, rho=kern.rho, l_gw_i=l_i2, l_gw_j=l_j2))
    d_dev2 = delta_joint_lnL_nats(compute_pair_sums(terms2, win2, win2, quad_n=800))
    ball = {k: (w, ref_rho(z, s)) for k, (z, s, w) in enumerate(gals)}
    d_ref2, _, _ = ref_delta(
        ball,
        ball,
        list(range(5)),
        lambda k: sc(l_i2),
        lambda k: sc(l_j2),
        lambda k: lambda z: sc(l_i2)(z) * sc(l_j2)(z),
        win2,
        win2,
    )
    row_ok = abs(d_dev2 - d_ref2) < max(1e-9, 1e-6 * abs(d_ref2))
    ok2 = ok2 and row_ok
    res2[tag] = {"delta_device": d_dev2, "delta_my_reference": d_ref2, "ok": row_ok}
ok2 = ok2 and abs(res2["u1"]["delta_device"] - res2["u2"]["delta_device"]) > 0.001
out["N2_checkC_universes_my_reference"] = res2 | {"ok": ok2}
if not ok2:
    failures.append("N2")

# ----------------------------------------------------------------------------
# N3: 2D pair, z-dependent mu_cond & unequal a_e(z), nested-quad reference
# ----------------------------------------------------------------------------
M_i_det, M_j_det = 2.0, 3.5
s2c_i, s2c_j = 0.03**2, 0.05**2
M_eff, sM = 1.0, 0.6
z_g3, s_z3 = 0.41, 0.025
w_g3 = 1.7
win3 = (0.15, 0.85)
gw_i = gauss_l(0.43, 0.05)
gw_j = gauss_l(0.40, 0.06)


def mu_i_of(z):
    z = np.asarray(z, dtype=np.float64)
    return 0.30 + 0.55 * z  # z-dependent conditional mean, event i


def mu_j_of(z):
    z = np.asarray(z, dtype=np.float64)
    return 0.52 - 0.28 * z  # different slope, event j


def a_i_of(z):
    z = np.asarray(z, dtype=np.float64)
    return (1.0 + z) / M_i_det


def a_j_of(z):
    z = np.asarray(z, dtype=np.float64)
    return (1.0 + z) / M_j_det


def mz_i_np(z):
    a = a_i_of(z)
    v = s2c_i + (sM * a) ** 2
    return np.exp(-0.5 * (mu_i_of(z) - M_eff * a) ** 2 / v) / np.sqrt(2 * np.pi * v)


def mz_j_np(z):
    a = a_j_of(z)
    v = s2c_j + (sM * a) ** 2
    return np.exp(-0.5 * (mu_j_of(z) - M_eff * a) ** 2 / v) / np.sqrt(2 * np.pi * v)


def l_i3(z):
    return gw_i(z) * mz_i_np(z)


def l_j3(z):
    return gw_j(z) * mz_j_np(z)


def joint_l3(z):
    return (
        gw_i(z)
        * gw_j(z)
        * scaled_shared_latent_mass_joint(
            mu_i_of(z), mu_j_of(z), a_i_of(z), a_j_of(z), s2c_i, s2c_j, M_eff, sM**2
        )
    )


kern3 = make_galaxy_z_kernel(z_g3, s_z3, w_pop_eff=toy_w_pop_np, quad_n=800)
terms3 = [SharedGalaxyTerm(w_g=w_g3, rho=kern3.rho, l_gw_i=l_i3, l_gw_j=l_j3, joint_l=joint_l3)]
sums3 = compute_pair_sums(terms3, win3, win3, quad_n=800)
d_dev3 = delta_joint_lnL_nats(sums3)


# my reference: nested adaptive quad over (z outer, M inner) — no closed form
def mzjoint_ref_sc(z):
    a_i = (1.0 + z) / M_i_det
    a_j = (1.0 + z) / M_j_det
    mu_i = 0.30 + 0.55 * z
    mu_j = 0.52 - 0.28 * z
    v, _ = quad(
        lambda M: (
            npdf(mu_i, M * a_i, math.sqrt(s2c_i))
            * npdf(mu_j, M * a_j, math.sqrt(s2c_j))
            * npdf(M, M_eff, sM)
        ),
        M_eff - 10 * sM,
        M_eff + 10 * sM,
        epsabs=1e-18,
        limit=300,
    )
    return v


rho3_ref = ref_rho(z_g3, s_z3, toy_w_pop_sc)
ball3 = {"g": (w_g3, rho3_ref)}
d_ref3, _, _ = ref_delta(
    ball3,
    ball3,
    ["g"],
    lambda k: sc(l_i3),
    lambda k: sc(l_j3),
    lambda k: lambda z: sc(gw_i)(z) * sc(gw_j)(z) * mzjoint_ref_sc(z),
    win3,
    win3,
)
ok3 = abs(d_dev3 - d_ref3) < max(1e-8, 1e-6 * abs(d_ref3))
out["N3_2d_zdependent_mu_nested_quad"] = {
    "delta_device": d_dev3,
    "delta_reference_nested_quad": d_ref3,
    "abs_diff": abs(d_dev3 - d_ref3),
    "ok": ok3,
}
if not ok3:
    failures.append("N3")


# ----------------------------------------------------------------------------
# N4: sigma_M = 0 -> J(joint) == J(factorized product) exactly
# ----------------------------------------------------------------------------
def mz_i_0(z):
    a = a_i_of(z)
    v = s2c_i
    return np.exp(-0.5 * (mu_i_of(z) - M_eff * a) ** 2 / v) / np.sqrt(2 * np.pi * v)


def mz_j_0(z):
    a = a_j_of(z)
    v = s2c_j
    return np.exp(-0.5 * (mu_j_of(z) - M_eff * a) ** 2 / v) / np.sqrt(2 * np.pi * v)


def l_i4(z):
    return gw_i(z) * mz_i_0(z)


def l_j4(z):
    return gw_j(z) * mz_j_0(z)


def joint_l4(z):
    return (
        gw_i(z)
        * gw_j(z)
        * scaled_shared_latent_mass_joint(
            mu_i_of(z), mu_j_of(z), a_i_of(z), a_j_of(z), s2c_i, s2c_j, M_eff, 0.0
        )
    )


t_joint = SharedGalaxyTerm(w_g=w_g3, rho=kern3.rho, l_gw_i=l_i4, l_gw_j=l_j4, joint_l=joint_l4)
t_fact = SharedGalaxyTerm(w_g=w_g3, rho=kern3.rho, l_gw_i=l_i4, l_gw_j=l_j4, joint_l=None)
s_joint = compute_pair_sums([t_joint], win3, win3, quad_n=800)
s_fact = compute_pair_sums([t_fact], win3, win3, quad_n=800)
rel4 = abs(s_joint.shared_diag_joint - s_fact.shared_diag_joint) / s_fact.shared_diag_joint
ok4 = rel4 < 1e-12
out["N4_sigmaM_zero_reduction"] = {
    "J_joint_sM0": s_joint.shared_diag_joint,
    "J_factorized": s_fact.shared_diag_joint,
    "rel_diff": rel4,
    "ok": ok4,
}
if not ok4:
    failures.append("N4")

# ----------------------------------------------------------------------------
# N5: invariance — global w rescale and per-event likelihood rescale
# ----------------------------------------------------------------------------
c_w, c_i, c_j = 7.3, 0.2, 41.0
terms5 = []
for z_g, s_z, w_g in shared_gals:
    kern = make_galaxy_z_kernel(z_g, s_z, w_pop_eff=toy_w_pop_np, quad_n=quad_n)
    terms5.append(
        SharedGalaxyTerm(
            w_g=w_g * c_w,
            rho=kern.rho,
            l_gw_i=lambda z, f=l_i: c_i * f(z),
            l_gw_j=lambda z, f=l_j: c_j * f(z),
        )
    )
members_i5 = [
    BallMember(
        w_g=w * c_w,
        rho=make_galaxy_z_kernel(z, s, w_pop_eff=toy_w_pop_np, quad_n=quad_n).rho,
        l_ev=lambda z, f=l_i: c_i * f(z),
    )
    for z, s, w in shared_gals + priv_i
]
members_j5 = [
    BallMember(
        w_g=w * c_w,
        rho=make_galaxy_z_kernel(z, s, w_pop_eff=toy_w_pop_np, quad_n=quad_n).rho,
        l_ev=lambda z, f=l_j: c_j * f(z),
    )
    for z, s, w in shared_gals + priv_j
]
S_i5 = compute_ball_sum(members_i5, win_i, quad_n=quad_n)
S_j5 = compute_ball_sum(members_j5, win_j, quad_n=quad_n)
sums5 = compute_pair_sums(terms5, win_i, win_j, quad_n=quad_n, S_i=S_i5, S_j=S_j5)
d_dev5 = delta_joint_lnL_nats(sums5)
ok5 = abs(d_dev5 - d_dev) < 1e-12 * max(1.0, abs(d_dev))
out["N5_rescale_invariance"] = {
    "delta_base": d_dev,
    "delta_rescaled": d_dev5,
    "abs_diff": abs(d_dev5 - d_dev),
    "ok": ok5,
}
if not ok5:
    failures.append("N5")

# ----------------------------------------------------------------------------
# N6: edge-overlap pair — shared galaxy at the window-intersection edge
# ----------------------------------------------------------------------------
win_i6 = (0.10, 0.52)
win_j6 = (0.48, 0.90)  # intersection [0.48, 0.52]
z_g6, s_z6, w_g6 = 0.50, 0.03, 1.0
l_i6 = gauss_l(0.45, 0.06)
l_j6 = gauss_l(0.55, 0.06)
kern6 = make_galaxy_z_kernel(z_g6, s_z6, quad_n=800)
terms6 = [SharedGalaxyTerm(w_g=w_g6, rho=kern6.rho, l_gw_i=l_i6, l_gw_j=l_j6)]
# balls: shared + one private each so the corrected joint stays positive
p_i6 = (0.40, 0.02, 3.0)
p_j6 = (0.60, 0.02, 3.0)
mem_i6 = [
    BallMember(w_g=w_g6, rho=kern6.rho, l_ev=l_i6),
    BallMember(w_g=p_i6[2], rho=make_galaxy_z_kernel(p_i6[0], p_i6[1], quad_n=800).rho, l_ev=l_i6),
]
mem_j6 = [
    BallMember(w_g=w_g6, rho=kern6.rho, l_ev=l_j6),
    BallMember(w_g=p_j6[2], rho=make_galaxy_z_kernel(p_j6[0], p_j6[1], quad_n=800).rho, l_ev=l_j6),
]
S_i6 = compute_ball_sum(mem_i6, win_i6, quad_n=800)
S_j6 = compute_ball_sum(mem_j6, win_j6, quad_n=800)
sums6 = compute_pair_sums(terms6, win_i6, win_j6, quad_n=800, S_i=S_i6, S_j=S_j6)
d_dev6 = delta_joint_lnL_nats(sums6)
ball_i6 = {"s": (w_g6, ref_rho(z_g6, s_z6)), "p": (p_i6[2], ref_rho(p_i6[0], p_i6[1]))}
ball_j6 = {"s": (w_g6, ref_rho(z_g6, s_z6)), "p": (p_j6[2], ref_rho(p_j6[0], p_j6[1]))}
d_ref6, _, _ = ref_delta(
    ball_i6,
    ball_j6,
    ["s"],
    lambda k: sc(l_i6),
    lambda k: sc(l_j6),
    lambda k: lambda z: sc(l_i6)(z) * sc(l_j6)(z),
    win_i6,
    win_j6,
)
ok6 = abs(d_dev6 - d_ref6) < max(1e-9, 1e-6 * abs(d_ref6)) and d_dev6 < 0.0
out["N6_edge_overlap_negative_delta"] = {
    "delta_device": d_dev6,
    "delta_reference": d_ref6,
    "abs_diff": abs(d_dev6 - d_ref6),
    "ok": ok6,
}
if not ok6:
    failures.append("N6")

# ----------------------------------------------------------------------------
# N7: severe cancellation — shared set dominates both balls, small sigma_z
# ----------------------------------------------------------------------------
s_z7 = 0.004
shared7 = [(0.40, s_z7, 5.0), (0.44, s_z7, 5.0)]
priv7 = (0.60, 0.02, 1e-3)  # negligible private weight
l_i7 = gauss_l(0.42, 0.05)
l_j7 = gauss_l(0.41, 0.06)
win7 = (0.1, 0.9)
qn7 = 4000
terms7 = []
for z_g, s_z, w_g in shared7:
    kern = make_galaxy_z_kernel(z_g, s_z, quad_n=qn7)
    terms7.append(SharedGalaxyTerm(w_g=w_g, rho=kern.rho, l_gw_i=l_i7, l_gw_j=l_j7))
mem_i7 = [
    BallMember(w_g=w, rho=make_galaxy_z_kernel(z, s, quad_n=qn7).rho, l_ev=l_i7)
    for z, s, w in shared7 + [priv7]
]
mem_j7 = [
    BallMember(w_g=w, rho=make_galaxy_z_kernel(z, s, quad_n=qn7).rho, l_ev=l_j7)
    for z, s, w in shared7 + [priv7]
]
S_i7 = compute_ball_sum(mem_i7, win7, quad_n=qn7)
S_j7 = compute_ball_sum(mem_j7, win7, quad_n=qn7)
sums7 = compute_pair_sums(terms7, win7, win7, quad_n=qn7, S_i=S_i7, S_j=S_j7)
d_dev7 = delta_joint_lnL_nats(sums7)
ball_i7 = {k: (w, ref_rho(z, s)) for k, (z, s, w) in enumerate(shared7 + [priv7])}
d_ref7, _, _ = ref_delta(
    ball_i7,
    ball_i7,
    [0, 1],
    lambda k: sc(l_i7),
    lambda k: sc(l_j7),
    lambda k: lambda z: sc(l_i7)(z) * sc(l_j7)(z),
    win7,
    win7,
)
ok7 = abs(d_dev7 - d_ref7) < max(5e-7, 5e-4 * abs(d_ref7))
out["N7_cancellation_stress_small_sigma"] = {
    "delta_device": d_dev7,
    "delta_reference": d_ref7,
    "abs_diff": abs(d_dev7 - d_ref7),
    "floor_scale_lnZ4sigma": math.log(math.erf(4.0 / math.sqrt(2.0))),
    "ok": ok7,
}
if not ok7:
    failures.append("N7")

# ----------------------------------------------------------------------------
# N8: mixture-composition algebra (numeric, random draws)
# ----------------------------------------------------------------------------
rng = np.random.default_rng(31415)
max_err = 0.0
for _ in range(200):
    w_Gi, w_Gj = rng.uniform(0.05, 0.99, size=2)
    Lcat_i, Lcat_j = rng.uniform(1e-8, 5.0, size=2)
    Lcomp_i, Lcomp_j = rng.uniform(1e-8, 5.0, size=2)
    D = rng.uniform(-0.5, 0.5)
    comb_i = w_Gi * Lcat_i + (1 - w_Gi) * Lcomp_i
    comb_j = w_Gj * Lcat_j + (1 - w_Gj) * Lcomp_j
    # corrected mixture joint: replace the cat x cat product by e^D-scaled
    joint = comb_i * comb_j + w_Gi * w_Gj * Lcat_i * Lcat_j * (math.exp(D) - 1.0)
    lhs = math.log(joint) - math.log(comb_i * comb_j)
    rhs = math.log1p(w_Gi * w_Gj * Lcat_i * Lcat_j * (math.exp(D) - 1.0) / (comb_i * comb_j))
    max_err = max(max_err, abs(lhs - rhs))
ok8 = max_err < 1e-12
out["N8_mixture_composition_identity"] = {"max_abs_err": max_err, "ok": ok8}
if not ok8:
    failures.append("N8")

# ----------------------------------------------------------------------------
# N9: floor constant
# ----------------------------------------------------------------------------
floor = math.log(math.erf(4.0 / math.sqrt(2.0)))
ok9 = abs(abs(floor) - 6.334e-5) < 2e-8
out["N9_floor_constant"] = {"ln_erf_4_over_sqrt2": floor, "documented": -6.334e-5, "ok": ok9}
if not ok9:
    failures.append("N9")

out["OVERALL"] = {"all_pass": len(failures) == 0, "failures": failures}
print(json.dumps(out, indent=1))
with open(Path(__file__).resolve().parent / "rereview_round1_independent_checks.json", "w") as fh:
    json.dump(out, fh, indent=1)
sys.exit(0 if not failures else 1)
