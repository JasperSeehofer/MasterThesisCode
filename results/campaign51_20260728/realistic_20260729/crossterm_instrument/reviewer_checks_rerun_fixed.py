"""RERUN of the adversarial review checks against the FIXED instrument (2026-08-05).

The original adversarial_refutation_checks.py (kept untouched as the review
record) proved the OLD emitted schema could not represent the Eq.(31) pairwise
cross-term (DEFECT-1/-2) and that the 2D channel factorized the shared latent
mass (DEFECT-3). This script re-runs the SAME constructions — same galaxies,
same weights, same windows, same independent adaptive-quad brute force — but
scores the FIXED instrument (w^2 shared diagonal, full-ball sums,
delta_joint_lnL_nats, shared-latent-mass joint). Checks are NOT weakened: the
brute-force ground truth is built exactly as the reviewer built it (their
brute_force_pair replicated verbatim below), and the acceptance criteria are
strict agreement where the review measured disagreement.

  A. n_shared = 1: fixed delta == brute force (parity with the old check).
  B. n_shared = 2, sigma_z -> 0: the review measured the old delta converging
     to -0.02886 nats (WRONG SIGN); the fixed delta must track delta_true at
     every sigma and -> 0.
  C. the review's nullspace universes had IDENTICAL old-schema emissions but
     10.3%-different truths; the fixed instrument must match truth on BOTH
     (and thereby distinguish them).
  D. 2D: the instrument's shared-latent-mass joint must equal the reviewer's
     closed form (which they verified against quadrature) at their exact
     parameter triples.

Run:
    cd /home/jasper/Repositories/MasterThesisCode && uv run python \
        results/campaign51_20260728/realistic_20260729/crossterm_instrument/reviewer_checks_rerun_fixed.py
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
    delta_joint_lnL_nats,
    make_galaxy_z_kernel,
    pair_joint_integral,
    per_galaxy_numerator,
    scaled_shared_latent_mass_joint,
    shared_latent_mass_joint,
)


def gaussian(mu, sigma):
    def f(z):
        return np.exp(-0.5 * ((z - mu) / sigma) ** 2) / (sigma * math.sqrt(2 * math.pi))

    return f


def brute_force_pair(galaxies_i, galaxies_j, shared, l_i, l_j, win_i, win_j, quad_n=800):
    """Reviewer's independent Eq.(31)-pairwise ground truth, replicated verbatim.

    (adversarial_refutation_checks.py:71-112 — adaptive quad for N and J,
    make_galaxy_z_kernel only for the Z_g normalization, explicit
    diagonal-replacement assembly.)
    """

    def N_of(l_fn, gal, win):
        z_g, s_z, _ = gal
        kern = make_galaxy_z_kernel(z_g, s_z, quad_n=quad_n)
        val, _ = quad(lambda z: l_fn(z) * float(kern.rho(np.array([z]))[0]), *win, limit=300)
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
    L_pair = L_fact_i * L_fact_j
    for k in shared:
        w = w_i[k]
        assert galaxies_i[k] == galaxies_j[k], "shared galaxy must be identical"
        L_pair -= w * N_i[k] * w * N_j[k]
        L_pair += w * w * J_of(galaxies_i[k])
    delta_true = math.log(L_pair) - math.log(L_fact_i * L_fact_j)
    return delta_true, L_fact_i, L_fact_j, L_pair


def fixed_instrument_delta(shared_gals, l_i, l_j, win_i, win_j, quad_n=800):
    """The FIXED instrument's pair delta (balls == shared set, as in the review)."""
    terms = []
    for z_g, s_z, w_g in shared_gals:
        kern = make_galaxy_z_kernel(z_g, s_z, quad_n=quad_n)
        terms.append(SharedGalaxyTerm(w_g=w_g, rho=kern.rho, l_gw_i=l_i, l_gw_j=l_j))
    sums = compute_pair_sums(terms, win_i, win_j, quad_n=quad_n)
    return delta_joint_lnL_nats(sums), sums


out = {}
failures = []

# ---------------------------------------------------------------------------
# A. n_shared = 1, shared galaxy is the whole ball
# ---------------------------------------------------------------------------
l_i = gaussian(0.42, 0.05)
l_j = gaussian(0.40, 0.06)
win = (0.1, 0.9)
g1 = (0.41, 0.03, 2.0)
d_true, *_ = brute_force_pair([g1], [g1], [0], l_i, l_j, win, win)
d_fixed, _ = fixed_instrument_delta([g1], l_i, l_j, win, win)
ok_a = abs(d_true - d_fixed) < 1e-9
out["A_single_shared"] = {
    "delta_true": d_true,
    "delta_fixed_instrument": d_fixed,
    "abs_diff": abs(d_true - d_fixed),
    "verdict_ok": ok_a,
}
if not ok_a:
    failures.append("A")

# ---------------------------------------------------------------------------
# B. n_shared = 2, sigma_z -> 0 (the DEFECT-2 killer): fixed delta must track
#    delta_true at every sigma (old delta converged to -0.02886, wrong sign).
# ---------------------------------------------------------------------------
rows = []
z_a, z_b, w_a, w_b = 0.35, 0.47, 1.0, 3.0
ok_b = True
for s_z in (0.02, 0.01, 0.005, 0.0025):
    quad_n = 800 if s_z >= 0.01 else 4000  # resolve narrow kernels (math-limit)
    gals = [(z_a, s_z, w_a), (z_b, s_z, w_b)]
    d_true, *_ = brute_force_pair(gals, gals, [0, 1], l_i, l_j, win, win, quad_n=quad_n)
    d_fixed, _ = fixed_instrument_delta(gals, l_i, l_j, win, win, quad_n=quad_n)
    row_ok = abs(d_true - d_fixed) < max(1e-8, 1e-6 * abs(d_true))
    ok_b = ok_b and row_ok and (d_fixed > 0.0)  # sign must be right too
    rows.append(
        {
            "sigma_z": s_z,
            "delta_true": d_true,
            "delta_fixed_instrument": d_fixed,
            "abs_diff": abs(d_true - d_fixed),
            "row_ok": row_ok,
        }
    )
# convergence to zero at O(sigma^2)
mags = [abs(r["delta_fixed_instrument"]) for r in rows]
conv_ok = mags[1] < 0.5 * mags[0] and mags[2] < 0.5 * mags[1] and mags[3] < 0.5 * mags[2]
ok_b = ok_b and conv_ok and mags[3] < 2e-3
out["B_two_shared_deltaz_limit"] = {
    "rows": rows,
    "old_instrument_asymptote_for_reference": -0.028855717813541442,
    "convergence_ratios": [mags[1] / mags[0], mags[2] / mags[1], mags[3] / mags[2]],
    "verdict_ok": ok_b,
    "note": (
        "FIXED: delta tracks the brute-force Eq.(31) truth at every sigma and "
        "-> 0 at O(sigma^2) (perfect-redshift exactness), with the correct "
        "positive sign throughout — the old convenience delta flipped sign "
        "below sigma_z ~ 0.005 and converged to -0.0289."
    ),
}
if not ok_b:
    failures.append("B")

# ---------------------------------------------------------------------------
# C. nullspace universes: identical OLD-schema emissions, different truths.
#    The fixed instrument must match truth on BOTH universes.
# ---------------------------------------------------------------------------
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
_, _, Vt = np.linalg.svd(A)
null = Vt[-1]
w1 = w0 + 0.8 * null / np.max(np.abs(null))
assert np.all(w1 > 0), w1
gals_u1 = [(z, s, float(w)) for (z, s), w in zip(gals_z, w0)]
gals_u2 = [(z, s, float(w)) for (z, s), w in zip(gals_z, w1)]
d1_true, *_ = brute_force_pair(gals_u1, gals_u1, list(range(5)), l_i, l_j, win, win)
d2_true, *_ = brute_force_pair(gals_u2, gals_u2, list(range(5)), l_i, l_j, win, win)
d1_fixed, s1 = fixed_instrument_delta(gals_u1, l_i, l_j, win, win)
d2_fixed, s2 = fixed_instrument_delta(gals_u2, l_i, l_j, win, win)
old_quintuple_rel_diff = float(
    max(
        abs(a - b) / abs(a)
        for a, b in zip(
            [s1.sum_wJ, s1.sum_wN_i, s1.sum_wN_j, s1.sum_w],
            [s2.sum_wJ, s2.sum_wN_i, s2.sum_wN_j, s2.sum_w],
        )
    )
)
new_fields_rel_diff = float(
    max(
        abs(a - b) / abs(a)
        for a, b in zip(
            [s1.shared_diag_fact, s1.shared_diag_joint],
            [s2.shared_diag_fact, s2.shared_diag_joint],
        )
    )
)
ok_c = (
    abs(d1_fixed - d1_true) < max(1e-9, 1e-6 * abs(d1_true))
    and abs(d2_fixed - d2_true) < max(1e-9, 1e-6 * abs(d2_true))
    and abs(d1_fixed - d2_fixed) / max(abs(d1_fixed), abs(d2_fixed)) > 0.05
)
out["C_identical_old_sums_different_truth"] = {
    "old_quintuple_max_rel_diff": old_quintuple_rel_diff,
    "new_w2_fields_max_rel_diff": new_fields_rel_diff,
    "delta_true_u1": d1_true,
    "delta_true_u2": d2_true,
    "delta_fixed_u1": d1_fixed,
    "delta_fixed_u2": d2_fixed,
    "abs_diff_u1": abs(d1_fixed - d1_true),
    "abs_diff_u2": abs(d2_fixed - d2_true),
    "fixed_rel_separation": abs(d1_fixed - d2_fixed) / max(abs(d1_fixed), abs(d2_fixed)),
    "verdict_ok": ok_c,
    "note": (
        "The w^1 quintuple is still degenerate between the universes (by "
        "construction), but the FIXED w^2 diagonal fields differ and the fixed "
        "delta matches the brute-force truth on BOTH universes — the schema "
        "now encodes the Eq.(31) cross-term for n_shared >= 2."
    ),
}
if not ok_c:
    failures.append("C")


# ---------------------------------------------------------------------------
# D. 2D shared latent mass: instrument closed form vs reviewer's exact values
# ---------------------------------------------------------------------------
def n_pdf(x, m, v):
    return math.exp(-0.5 * (x - m) ** 2 / v) / math.sqrt(2 * math.pi * v)


d_rows = []
ok_d = True
for mu_i, mu_j, s_i, s_j, M_g, sM in [
    (1.00, 1.02, 0.03, 0.04, 1.0, 0.6),
    (1.00, 1.30, 0.03, 0.04, 1.0, 0.6),
    (1.00, 1.02, 0.10, 0.12, 1.0, 2.0),
]:
    inst_joint = float(shared_latent_mass_joint(mu_i, mu_j, s_i**2, s_j**2, M_g, sM**2))
    v_sum = s_i**2 + s_j**2
    m_ij = (mu_i * s_j**2 + mu_j * s_i**2) / v_sum
    v_ij = s_i**2 * s_j**2 / v_sum
    reviewer_exact = n_pdf(mu_i, mu_j, v_sum) * n_pdf(m_ij, M_g, v_ij + sM**2)
    num, _ = quad(
        lambda M, mu_i=mu_i, mu_j=mu_j, s_i=s_i, s_j=s_j, M_g=M_g, sM=sM: (
            n_pdf(mu_i, M, s_i**2) * n_pdf(mu_j, M, s_j**2) * n_pdf(M, M_g, sM**2)
        ),
        M_g - 12 * sM,
        M_g + 12 * sM,
        limit=300,
    )
    row_ok = (
        abs(inst_joint - reviewer_exact) / reviewer_exact < 1e-12
        and abs(inst_joint - num) / num < 1e-5
    )
    # scaled variant must agree at a_i = a_j = 1
    scaled = float(
        scaled_shared_latent_mass_joint(mu_i, mu_j, 1.0, 1.0, s_i**2, s_j**2, M_g, sM**2)
    )
    row_ok = row_ok and abs(scaled - inst_joint) / inst_joint < 1e-14
    ok_d = ok_d and row_ok
    d_rows.append(
        {
            "params": {"mu_i": mu_i, "mu_j": mu_j, "s_i": s_i, "s_j": s_j, "M_g": M_g, "sM": sM},
            "instrument_shared_latent_joint": inst_joint,
            "reviewer_exact_closed_form": reviewer_exact,
            "adaptive_quad": num,
            "rel_vs_reviewer": abs(inst_joint - reviewer_exact) / reviewer_exact,
            "rel_vs_quad": abs(inst_joint - num) / num,
            "row_ok": row_ok,
        }
    )
out["D_2d_shared_mass"] = {"rows": d_rows, "verdict_ok": ok_d}
if not ok_d:
    failures.append("D")

out["OVERALL"] = {
    "all_checks_pass": len(failures) == 0,
    "failed_checks": failures,
}

print(json.dumps(out, indent=1))
with open(Path(__file__).resolve().parent / "reviewer_checks_rerun_fixed.json", "w") as fh:
    json.dump(out, fh, indent=1)
sys.exit(0 if not failures else 1)
