# ruff: noqa: E741
"""ADVERSARIAL REVIEW CHECKS (2026-08-05) — pure toy math, no production data.

Tests whether crossterm_instrument.py's emitted raw sums
(sum_wJ = Σ w_g J_g, sum_wN_i = Σ w_g N_g,i, sum_wN_j, sum_w) can represent the
leading pairwise Eq.(31) cross-term of Gair et al. 2023 (arXiv:2212.08694).

Ground truth (independent brute force, this file only):
The production per-event catalogue numerator is L_cat_raw,i = Σ_{g∈ball_i} w_g N_g,i
(bayesian_statistics.py:4011-4028 weighted_sum; host prior P(g) ∝ w_g,
_rate_weight :964). The pair product therefore contains the DIAGONAL terms
w_g^2 N_g,i N_g,j for g in S_ij = ball_i ∩ ball_j (one factor w_g from EACH
event's galaxy sum), and the exact Eq.(31) paired marginalization replaces
exactly those with w_g^2 J_g:

    L_pair_exact = Σ_{g≠g'} w_g w_g' N_g,i N_g',j + Σ_{g∈S_ij} w_g^2 J_g
    Delta_true   = ln L_pair_exact − ln( L_cat_raw,i · L_cat_raw,j )

(all per-event normalizers cancel in the ratio).

Checks:
  A. n_shared = 1: instrument delta with K = sum_w equals the shared-restricted
     truth (sanity — the instrument is fine here).
  B. n_shared >= 2, perfect-redshift limit (sigma_z -> 0): Delta_true -> 0
     (paper's exactness statement), but delta_nats(sums, K=sum_w) -> a NONZERO
     constant. No scalar K fixes it (the required K is data-dependent).
  C. Two toy universes with IDENTICAL emitted quintuples
     (sum_wJ, sum_wN_i, sum_wN_j, sum_w, n_shared) but DIFFERENT Delta_true:
     the emitted schema cannot represent the Eq.(31) cross-term for
     n_shared >= 2, for ANY analysis-layer convention.
  D. 2D channel: the built joint integrand multiplies the two events'
     INDEPENDENTLY mass-marginalized factors mz_i * mz_j, i.e. it treats the
     shared galaxy's true mass as two independent latents. The Eq.(31)-exact
     with-mass pair term marginalizes ONE shared latent mass:
       exact:      ∫ N(mu_i; M, s_i^2) N(mu_j; M, s_j^2) N(M; M_g, sM^2) dM
                 = N(mu_i; mu_j, s_i^2+s_j^2) · N(m_ij; M_g, v_ij + sM^2)
       instrument: N(mu_i; M_g, s_i^2+sM^2) · N(mu_j; M_g, s_j^2+sM^2)
     At production-plausible sM >> s_i,s_j these differ by ~ln(sM/s) nats.

Run:
    cd /home/jasper/Repositories/MasterThesisCode && uv run python \
        results/campaign51_20260728/realistic_20260729/crossterm_instrument/adversarial_refutation_checks.py
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import quad

sys.path.insert(0, str(Path(__file__).resolve().parent))

from crossterm_instrument import (  # noqa: E402
    SharedGalaxyTerm,
    compute_pair_sums,
    delta_nats,
    make_galaxy_z_kernel,
    pair_joint_integral,
    per_galaxy_numerator,
)


def gaussian(mu, sigma):
    def f(z):
        return np.exp(-0.5 * ((z - mu) / sigma) ** 2) / (sigma * math.sqrt(2 * math.pi))

    return f


def brute_force_pair(galaxies_i, galaxies_j, shared, l_i, l_j, win_i, win_j, quad_n=800):
    """Independent Eq.(31)-pairwise ground truth via adaptive quad.

    galaxies_i/j: list of (z_g, sigma_z, w_g) in each ball (shared listed in both).
    shared: indices (into both lists — shared galaxies must be at the same
    list positions with identical parameters).
    Returns (Delta_true, L_fact_i_raw, L_fact_j_raw, L_pair_exact).
    """

    def N_of(l, gal, win):
        z_g, s_z, _ = gal
        kern = make_galaxy_z_kernel(z_g, s_z, quad_n=quad_n)
        val, _ = quad(lambda z: l(z) * float(kern.rho(np.array([z]))[0]), *win, limit=300)
        return val

    def J_of(gal):
        z_g, s_z, _ = gal
        kern = make_galaxy_z_kernel(z_g, s_z, quad_n=quad_n)
        lo = max(win_i[0], win_j[0])
        hi = min(win_i[1], win_j[1])
        if lo >= hi:
            return 0.0
        val, _ = quad(
            lambda z: l_i(z) * l_j(z) * float(kern.rho(np.array([z]))[0]), lo, hi, limit=300
        )
        return val

    N_i = [N_of(l_i, g, win_i) for g in galaxies_i]
    N_j = [N_of(l_j, g, win_j) for g in galaxies_j]
    w_i = [g[2] for g in galaxies_i]
    w_j = [g[2] for g in galaxies_j]
    L_fact_i = sum(w * n for w, n in zip(w_i, N_i))
    L_fact_j = sum(w * n for w, n in zip(w_j, N_j))
    # exact pair: product minus shared diagonals + shared joint diagonals
    L_pair = L_fact_i * L_fact_j
    for k in shared:
        w = w_i[k]
        assert galaxies_i[k] == galaxies_j[k], "shared galaxy must be identical"
        L_pair -= w * N_i[k] * w * N_j[k]
        L_pair += w * w * J_of(galaxies_i[k])
    delta_true = math.log(L_pair) - math.log(L_fact_i * L_fact_j)
    return delta_true, L_fact_i, L_fact_j, L_pair


def instrument_sums(shared_gals, l_i, l_j, win_i, win_j, quad_n=800):
    terms = []
    for z_g, s_z, w_g in shared_gals:
        kern = make_galaxy_z_kernel(z_g, s_z, quad_n=quad_n)
        terms.append(SharedGalaxyTerm(w_g=w_g, rho=kern.rho, l_gw_i=l_i, l_gw_j=l_j))
    return compute_pair_sums(terms, win_i, win_j, quad_n=quad_n)


out = {}

# ---------------------------------------------------------------------------
# A. n_shared = 1, shared galaxy is the whole ball: instrument OK
# ---------------------------------------------------------------------------
l_i = gaussian(0.42, 0.05)
l_j = gaussian(0.40, 0.06)
win = (0.1, 0.9)
g1 = (0.41, 0.03, 2.0)
d_true, *_ = brute_force_pair([g1], [g1], [0], l_i, l_j, win, win)
sums = instrument_sums([g1], l_i, l_j, win, win)
d_inst = delta_nats(sums, sums.sum_w)
out["A_single_shared"] = {
    "delta_true": d_true,
    "delta_instrument_K_sum_w": d_inst,
    "abs_diff": abs(d_true - d_inst),
    "verdict_ok": abs(d_true - d_inst) < 1e-6,
}

# ---------------------------------------------------------------------------
# B. n_shared = 2, sigma_z -> 0: Delta_true -> 0, instrument delta -> const != 0
# ---------------------------------------------------------------------------
rows = []
z_a, z_b, w_a, w_b = 0.35, 0.47, 1.0, 3.0  # asymmetric: l_i rises a->b, l_j falls
for s_z in (0.02, 0.01, 0.005, 0.0025):
    gals = [(z_a, s_z, w_a), (z_b, s_z, w_b)]
    d_true, *_ = brute_force_pair(gals, gals, [0, 1], l_i, l_j, win, win)
    sums = instrument_sums(gals, l_i, l_j, win, win)
    d_inst = delta_nats(sums, sums.sum_w)
    # the K that WOULD be needed to reproduce d_true from the emitted sums:
    # ln(sum_wJ) - ln(sum_wN_i sum_wN_j / K) = d_true
    K_needed = math.exp(d_true) * sums.sum_wN_i * sums.sum_wN_j / sums.sum_wJ
    rows.append(
        {
            "sigma_z": s_z,
            "delta_true": d_true,
            "delta_instrument_K_sum_w": d_inst,
            "K_needed": K_needed,
            "sum_w": sums.sum_w,
        }
    )
# analytic sigma_z -> 0 limit of the instrument's convenience delta:
# ln[ (Σ w l_i l_j)(Σ w) / (Σ w l_i)(Σ w l_j) ]  (Chebyshev covariance term)
li_a, li_b = l_i(np.array([z_a]))[0], l_i(np.array([z_b]))[0]
lj_a, lj_b = l_j(np.array([z_a]))[0], l_j(np.array([z_b]))[0]
sw = w_a + w_b
inst_limit = math.log(
    (w_a * li_a * lj_a + w_b * li_b * lj_b)
    * sw
    / ((w_a * li_a + w_b * li_b) * (w_a * lj_a + w_b * lj_b))
)
out["B_two_shared_deltaz_limit"] = {
    "rows": rows,
    "instrument_delta_analytic_limit_sigma0": inst_limit,
    "note": (
        "delta_true -> 0 (paper exactness at perfect z); instrument delta with "
        "K=sum_w converges to the NONZERO covariance term; the K needed to "
        "match truth is data-dependent, so no fixed convention works."
    ),
}

# ---------------------------------------------------------------------------
# C. identical emitted quintuple, different Delta_true
# ---------------------------------------------------------------------------
# Universe 1: two shared galaxies with weights (w1, w2) = (1, 3).
# Universe 2: SAME two galaxies with weights swapped (3, 1).
# Both balls = shared set. Emitted sums differ then; instead build the swap so
# the emitted quintuple is preserved: swap the weights AND the galaxy z's so
# that (w, z) pairings change but the unordered multisets of per-galaxy
# products are permuted. Because N_g,i depends on z_g while w is a free label,
# swapping w between two galaxies with different N's changes Σ w^2 (J - N N)
# but leaves Σ w J, Σ w N unchanged ONLY if the per-galaxy (J, N_i, N_j) are
# also swapped — i.e., relabeling. To break it, use THREE galaxies and a
# weight permutation that is not a symmetry of the (N, J) triples but
# preserves all four w^1 sums. Solve linearly: find two weight vectors
# w, w' >= 0 with  Σ w = Σ w', Σ w N_i = Σ w' N_i, Σ w N_j = Σ w' N_j,
# Σ w J = Σ w' J  (4 constraints, 5 galaxies -> 1-dim family).
gals_z = [(0.34, 0.02), (0.38, 0.025), (0.42, 0.03), (0.46, 0.025), (0.50, 0.02)]
per = []
for z_g, s_z in gals_z:
    kern = make_galaxy_z_kernel(z_g, s_z, quad_n=800)
    N_i = per_galaxy_numerator(l_i, kern.rho, *win, quad_n=800)
    N_j = per_galaxy_numerator(l_j, kern.rho, *win, quad_n=800)
    J = pair_joint_integral(l_i, l_j, kern.rho, win, win, quad_n=800)
    per.append((N_i, N_j, J))
A = np.array(
    [
        [1.0] * 5,
        [p[0] for p in per],
        [p[1] for p in per],
        [p[2] for p in per],
    ]
)
w0 = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
# nullspace direction of A
_, _, Vt = np.linalg.svd(A)
null = Vt[-1]
w1 = w0 + 0.8 * null / np.max(np.abs(null))  # keep positive
assert np.all(w1 > 0), w1
check = A @ (w1 - w0)
gals_u1 = [(z, s, float(w)) for (z, s), w in zip(gals_z, w0)]
gals_u2 = [(z, s, float(w)) for (z, s), w in zip(gals_z, w1)]
d1, *_ = brute_force_pair(gals_u1, gals_u1, list(range(5)), l_i, l_j, win, win)
d2, *_ = brute_force_pair(gals_u2, gals_u2, list(range(5)), l_i, l_j, win, win)
s1 = instrument_sums(gals_u1, l_i, l_j, win, win)
s2 = instrument_sums(gals_u2, l_i, l_j, win, win)
out["C_identical_sums_different_truth"] = {
    "constraint_residual_max": float(np.max(np.abs(check))),
    "emitted_u1": [s1.sum_wJ, s1.sum_wN_i, s1.sum_wN_j, s1.sum_w],
    "emitted_u2": [s2.sum_wJ, s2.sum_wN_i, s2.sum_wN_j, s2.sum_w],
    "emitted_max_rel_diff": float(
        max(
            abs(a - b) / abs(a)
            for a, b in zip(
                [s1.sum_wJ, s1.sum_wN_i, s1.sum_wN_j, s1.sum_w],
                [s2.sum_wJ, s2.sum_wN_i, s2.sum_wN_j, s2.sum_w],
            )
        )
    ),
    "delta_true_u1": d1,
    "delta_true_u2": d2,
    "delta_true_rel_diff": abs(d1 - d2) / max(abs(d1), abs(d2)),
    "note": (
        "Same emitted quintuple (to numerical precision), different Eq.(31) "
        "cross-term -> the schema cannot encode the cross-term for n_shared>=2."
    ),
}


# ---------------------------------------------------------------------------
# D. 2D channel: shared latent mass vs product of independent mass marginals
# ---------------------------------------------------------------------------
def n_pdf(x, m, v):
    return math.exp(-0.5 * (x - m) ** 2 / v) / math.sqrt(2 * math.pi * v)


def d_check(mu_i, mu_j, s_i, s_j, M_g, sM):
    # instrument: product of independent marginals
    inst = n_pdf(mu_i, M_g, s_i**2 + sM**2) * n_pdf(mu_j, M_g, s_j**2 + sM**2)
    # exact shared-latent-M marginal
    v_sum = s_i**2 + s_j**2
    m_ij = (mu_i * s_j**2 + mu_j * s_i**2) / v_sum
    v_ij = s_i**2 * s_j**2 / v_sum
    exact = n_pdf(mu_i, mu_j, v_sum) * n_pdf(m_ij, M_g, v_ij + sM**2)
    # numeric confirmation of 'exact'
    num, _ = quad(
        lambda M: n_pdf(mu_i, M, s_i**2) * n_pdf(mu_j, M, s_j**2) * n_pdf(M, M_g, sM**2),
        M_g - 12 * sM,
        M_g + 12 * sM,
        limit=300,
    )
    return {
        "params": {"mu_i": mu_i, "mu_j": mu_j, "s_i": s_i, "s_j": s_j, "M_g": M_g, "sM": sM},
        "instrument_product": inst,
        "exact_shared_latent": exact,
        "exact_numeric_check_rel": abs(num - exact) / exact,
        "log_ratio_nats": math.log(exact / inst),
    }


# production-plausible: GW conditional mass precision few %, host prior sM ~ 60%
out["D_2d_shared_mass"] = [
    d_check(1.00, 1.02, 0.03, 0.04, 1.0, 0.6),
    d_check(1.00, 1.30, 0.03, 0.04, 1.0, 0.6),  # discrepant events
    d_check(1.00, 1.02, 0.10, 0.12, 1.0, 2.0),  # sM = 200%
]

print(json.dumps(out, indent=1))
with open(Path(__file__).resolve().parent / "adversarial_refutation_checks.json", "w") as fh:
    json.dump(out, fh, indent=1)
