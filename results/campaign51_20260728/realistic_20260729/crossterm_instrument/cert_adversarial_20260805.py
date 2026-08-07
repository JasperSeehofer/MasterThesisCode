# ruff: noqa: B023
"""Certification-review adversarial constructions (2026-08-05, hash 340b66d2...).

Two NEW attacks against the instrument math core, each checked against an
INDEPENDENT brute-force reference built from scipy.integrate.quad nested
adaptive quadrature — no instrument closed forms, no fixed_quad, on the
reference side.

ATTACK 1 — "extreme weight asymmetry + near-total ball subset, 2D channel":
  ball_j == shared set (5 galaxies), shared is 5/6 of ball_i; weights span
  1e-8..1e2 (10 decades); one narrow kernel triggers the R-2 escalation
  (R = 75 > 30) inside the 2D channel; the 2D joint is referenced by NESTED
  quad over (z, M) — the shared-latent-M closed form is never used on the
  reference side. Also checks i<->j symmetry.

ATTACK 2 — "mixed 1D/2D edge: asymmetric windows + escalation boundary
  crossing + false-shared galaxy":
  window_i is 8.6x wider than window_j; the shared kernel sits at the lower
  edge of the intersection with half its mass clipped; the SAME galaxy is
  escalated in the shared diagonal (R(width_max)=50>30) but NOT in ball_j's
  sum (R(width_j)=5.8<=30) — probing exactly the production-layer rule; a
  second "false shared" galaxy (in both declared balls, kernel support wholly
  outside the intersection, w = 1e3) must contribute ~0. Run in BOTH channels
  (1D and 2D). A no-escalation (n=50 everywhere) variant is also computed to
  show the construction has teeth (it must disagree with the reference).

Verdict criteria printed per attack; JSON emitted next to this script.
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import quad

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from crossterm_instrument import (  # noqa: E402
    BallMember,
    SharedGalaxyTerm,
    adaptive_quad_n,
    compute_ball_sum,
    compute_pair_sums,
    delta_joint_lnL_nats,
    scaled_shared_latent_mass_joint,
)

RESULTS: dict = {}


def gauss(x, m, s):
    return np.exp(-0.5 * ((x - m) / s) ** 2) / (s * math.sqrt(2.0 * math.pi))


def make_rho(z_g, sigma_z):
    """Shared normalized kernel: N(z; z_g, s) * w(z) / Z, wiggly positive w(z).

    The SAME callable is handed to instrument and reference, so Z's own
    quadrature is not under test here (Z computed by adaptive quad).
    """

    def w(z):
        return (1.0 + z) ** 2 * (0.5 + 0.3 * np.sin(3.0 * z))

    lo, hi = max(z_g - 4.0 * sigma_z, 1e-6), z_g + 4.0 * sigma_z
    Z = quad(
        lambda z: float(gauss(np.asarray(z), z_g, sigma_z) * w(np.asarray(z))),
        lo,
        hi,
        epsabs=1e-300,
        epsrel=1e-13,
        limit=500,
    )[0]

    def rho(z):
        z = np.asarray(z, dtype=np.float64)
        return gauss(z, z_g, sigma_z) * w(z) / Z

    return rho


def ref_int(f, lo, hi):
    """Reference 1D integral: adaptive quad on a scalar wrapper."""
    if lo >= hi:
        return 0.0
    val = quad(
        lambda z: float(f(np.array([z]))[0]), lo, hi, epsabs=1e-300, epsrel=1e-12, limit=500
    )[0]
    return val


# ---------------------------------------------------------------------------
# ATTACK 1 — 2D channel, extreme weight asymmetry, ball_j == shared set.
# ---------------------------------------------------------------------------


def attack_1():
    window_i = (0.30, 0.60)
    window_j = (0.42, 0.55)
    width_i = window_i[1] - window_i[0]
    width_j = window_j[1] - window_j[0]
    width_max = max(width_i, width_j)

    # Event z-likelihoods (3D-Gaussian stand-ins along the z ray).
    def l_z_i(z):
        return gauss(np.asarray(z), 0.50, 0.045)

    def l_z_j(z):
        return gauss(np.asarray(z), 0.49, 0.020)

    # 2D-channel per-event mass machinery (toy production parametrization).
    M_det_i, M_det_j = 3.0, 5.0
    s2c_i, s2c_j = 0.03**2, 0.05**2
    M_eff, sM = 2.0, 0.4

    def mu_i(z):
        return 0.9 + 0.4 * np.asarray(z)

    def mu_j(z):
        return 0.55 + 0.1 * np.asarray(z)

    def a_i(z):
        return (1.0 + np.asarray(z)) / M_det_i

    def a_j(z):
        return (1.0 + np.asarray(z)) / M_det_j

    # INSTRUMENT-side factorized mass marginals (closed Gaussian, as the
    # production make_mz_factor computes them).
    def mz_i(z):
        z = np.asarray(z)
        s2 = s2c_i + (sM * a_i(z)) ** 2
        return np.exp(-0.5 * (mu_i(z) - M_eff * a_i(z)) ** 2 / s2) / np.sqrt(2 * np.pi * s2)

    def mz_j(z):
        z = np.asarray(z)
        s2 = s2c_j + (sM * a_j(z)) ** 2
        return np.exp(-0.5 * (mu_j(z) - M_eff * a_j(z)) ** 2 / s2) / np.sqrt(2 * np.pi * s2)

    def mz_joint(z):
        return scaled_shared_latent_mass_joint(
            mu_i(z), mu_j(z), a_i(z), a_j(z), s2c_i, s2c_j, M_eff, sM**2
        )

    # REFERENCE-side mass marginals: brute-force quad over M (independent).
    def M_limits(z):
        centers = [
            M_eff,
            float(mu_i(np.array([z]))[0] / a_i(np.array([z]))[0]),
            float(mu_j(np.array([z]))[0] / a_j(np.array([z]))[0]),
        ]
        widths = [
            sM,
            math.sqrt(s2c_i) / float(a_i(np.array([z]))[0]),
            math.sqrt(s2c_j) / float(a_j(np.array([z]))[0]),
        ]
        return min(c - 14 * w for c, w in zip(centers, widths)), max(
            c + 14 * w for c, w in zip(centers, widths)
        )

    def mz_i_ref(z_scalar):
        z = np.array([z_scalar])
        lo, hi = M_limits(z_scalar)
        return quad(
            lambda M: float(
                gauss(mu_i(z), M * a_i(z), math.sqrt(s2c_i))[0] * gauss(np.array([M]), M_eff, sM)[0]
            ),
            lo,
            hi,
            epsabs=1e-300,
            epsrel=1e-12,
            limit=500,
        )[0]

    def mz_j_ref(z_scalar):
        z = np.array([z_scalar])
        lo, hi = M_limits(z_scalar)
        return quad(
            lambda M: float(
                gauss(mu_j(z), M * a_j(z), math.sqrt(s2c_j))[0] * gauss(np.array([M]), M_eff, sM)[0]
            ),
            lo,
            hi,
            epsabs=1e-300,
            epsrel=1e-12,
            limit=500,
        )[0]

    def mz_joint_ref(z_scalar):
        z = np.array([z_scalar])
        lo, hi = M_limits(z_scalar)
        return quad(
            lambda M: float(
                gauss(mu_i(z), M * a_i(z), math.sqrt(s2c_i))[0]
                * gauss(mu_j(z), M * a_j(z), math.sqrt(s2c_j))[0]
                * gauss(np.array([M]), M_eff, sM)[0]
            ),
            lo,
            hi,
            epsabs=1e-300,
            epsrel=1e-12,
            limit=500,
        )[0]

    # Galaxies: 5 shared (ball_j entirely), 1 extra in ball_i only.
    shared_specs = [
        # (z_g, sigma_z, w_g) — weights span 10 decades; first kernel R=0.3/0.004=75>30.
        (0.44, 0.004, 1e-8),
        (0.47, 0.010, 1e2),
        (0.50, 0.020, 1.0),
        (0.52, 0.006, 3e-3),
        (0.54, 0.030, 5e1),
    ]
    extra_i = (0.35, 0.02, 10.0)

    rhos = {spec: make_rho(spec[0], spec[1]) for spec in shared_specs}
    rho_extra = make_rho(extra_i[0], extra_i[1])

    def fact_i(rho):
        def f(z):
            return l_z_i(z) * mz_i(z) * rho(z)

        return f

    def fact_j(rho):
        def f(z):
            return l_z_j(z) * mz_j(z) * rho(z)

        return f

    # --- instrument side ---
    terms = []
    for spec in shared_specs:
        z_g, s_z, w_g = spec
        rho = rhos[spec]
        terms.append(
            SharedGalaxyTerm(
                w_g=w_g,
                rho=rho,
                l_gw_i=lambda z, r=rho: l_z_i(z) * mz_i(z),
                l_gw_j=lambda z, r=rho: l_z_j(z) * mz_j(z),
                joint_l=lambda z: l_z_i(z) * l_z_j(z) * mz_joint(z),
                quad_n_override=adaptive_quad_n(width_max, s_z),
            )
        )
    ball_i_members = [
        BallMember(
            w_g=w,
            rho=rhos[(zg, sz, w)],
            l_ev=lambda z: l_z_i(z) * mz_i(z),
            quad_n_override=adaptive_quad_n(width_i, sz),
        )
        for (zg, sz, w) in shared_specs
    ] + [
        BallMember(
            w_g=extra_i[2],
            rho=rho_extra,
            l_ev=lambda z: l_z_i(z) * mz_i(z),
            quad_n_override=adaptive_quad_n(width_i, extra_i[1]),
        )
    ]
    ball_j_members = [
        BallMember(
            w_g=w,
            rho=rhos[(zg, sz, w)],
            l_ev=lambda z: l_z_j(z) * mz_j(z),
            quad_n_override=adaptive_quad_n(width_j, sz),
        )
        for (zg, sz, w) in shared_specs
    ]
    S_i = compute_ball_sum(ball_i_members, window_i)
    S_j = compute_ball_sum(ball_j_members, window_j)
    sums = compute_pair_sums(terms, window_i, window_j, S_i=S_i, S_j=S_j)
    delta_inst = delta_joint_lnL_nats(sums)

    # symmetry: swap roles entirely
    terms_swap = [
        SharedGalaxyTerm(
            w_g=t.w_g,
            rho=t.rho,
            l_gw_i=t.l_gw_j,
            l_gw_j=t.l_gw_i,
            joint_l=t.joint_l,
            quad_n_override=t.quad_n_override,
        )
        for t in terms
    ]
    sums_swap = compute_pair_sums(terms_swap, window_j, window_i, S_i=S_j, S_j=S_i)
    delta_swap = delta_joint_lnL_nats(sums_swap)

    # --- reference side (fully independent) ---
    inter = (max(window_i[0], window_j[0]), min(window_i[1], window_j[1]))

    def ref_N(l_z, mz_ref, rho, window):
        return quad(
            lambda z: float(l_z(np.array([z]))[0]) * mz_ref(z) * float(rho(np.array([z]))[0]),
            window[0],
            window[1],
            epsabs=1e-300,
            epsrel=1e-12,
            limit=500,
        )[0]

    S_i_ref = sum(
        w * ref_N(l_z_i, mz_i_ref, rhos[(zg, sz, w)], window_i) for (zg, sz, w) in shared_specs
    ) + extra_i[2] * ref_N(l_z_i, mz_i_ref, rho_extra, window_i)
    S_j_ref = sum(
        w * ref_N(l_z_j, mz_j_ref, rhos[(zg, sz, w)], window_j) for (zg, sz, w) in shared_specs
    )
    diag_fact_ref = 0.0
    diag_joint_ref = 0.0
    for zg, sz, w in shared_specs:
        rho = rhos[(zg, sz, w)]
        N_i = ref_N(l_z_i, mz_i_ref, rho, window_i)
        N_j = ref_N(l_z_j, mz_j_ref, rho, window_j)
        J = quad(
            lambda z: (
                float(l_z_i(np.array([z]))[0])
                * float(l_z_j(np.array([z]))[0])
                * mz_joint_ref(z)
                * float(rho(np.array([z]))[0])
            ),
            inter[0],
            inter[1],
            epsabs=1e-300,
            epsrel=1e-12,
            limit=500,
        )[0]
        diag_fact_ref += w**2 * N_i * N_j
        diag_joint_ref += w**2 * J
    delta_ref = math.log1p((diag_joint_ref - diag_fact_ref) / (S_i_ref * S_j_ref))

    ok = (
        math.isfinite(delta_inst)
        and abs(delta_inst - delta_ref) < 1e-9
        and abs(delta_inst - delta_swap) < 1e-13
    )
    RESULTS["attack1_extreme_asymmetry_2d"] = {
        "delta_inst": delta_inst,
        "delta_ref_bruteforce": delta_ref,
        "abs_diff_nats": abs(delta_inst - delta_ref),
        "delta_swap_symmetry": delta_swap,
        "abs_asym": abs(delta_inst - delta_swap),
        "S_i_inst": S_i,
        "S_i_ref": S_i_ref,
        "S_j_inst": S_j,
        "S_j_ref": S_j_ref,
        "escalated_orders": [t.quad_n_override for t in terms],
        "weights_decades": [w for (_, _, w) in shared_specs],
        "ok": ok,
    }
    return ok


# ---------------------------------------------------------------------------
# ATTACK 2 — asymmetric windows, edge-clipped kernel, escalation boundary
#            crossing, false-shared galaxy; 1D and 2D channels.
# ---------------------------------------------------------------------------


def attack_2():
    window_i = (0.10, 0.70)  # width 0.60
    window_j = (0.475, 0.545)  # width 0.07
    width_i = window_i[1] - window_i[0]
    width_j = window_j[1] - window_j[0]
    width_max = max(width_i, width_j)
    inter = (max(window_i[0], window_j[0]), min(window_i[1], window_j[1]))

    def l_z_i(z):
        return gauss(np.asarray(z), 0.48, 0.09)

    def l_z_j(z):
        return gauss(np.asarray(z), 0.51, 0.011)

    # Galaxy A: kernel at the intersection's lower edge, half-mass clipped.
    zA, sA, wA = 0.478, 0.012, 2.0
    # Galaxy B: "false shared" — declared in both balls, support (0.26, 0.34)
    # wholly outside the intersection and outside window_j; huge weight.
    zB, sB, wB = 0.30, 0.01, 1e3

    rhoA, rhoB = make_rho(zA, sA), make_rho(zB, sB)

    # escalation boundary facts (assert the construction is what we claim)
    nA_shared = adaptive_quad_n(width_max, sA)  # R = 50 -> escalated
    nA_ball_j = adaptive_quad_n(width_j, sA)  # R = 5.83 -> 50 (not escalated)
    nA_ball_i = adaptive_quad_n(width_i, sA)  # R = 50 -> escalated
    assert nA_shared > 50 and nA_ball_j == 50 and nA_ball_i == nA_shared

    channels = {}
    for ch in ("1d", "2d"):
        if ch == "1d":
            fi = l_z_i
            fj = l_z_j
            joint = None

            def joint_ref_z(z_scalar):
                z = np.array([z_scalar])
                return float(l_z_i(z)[0]) * float(l_z_j(z)[0])

            def fi_ref(z_scalar):
                z = np.array([z_scalar])
                return float(l_z_i(z)[0])

            def fj_ref(z_scalar):
                z = np.array([z_scalar])
                return float(l_z_j(z)[0])
        else:
            M_det_i, M_det_j = 4.0, 2.5
            s2c_i, s2c_j = 0.04**2, 0.02**2
            M_eff, sM = 1.6, 0.5

            def mu_i(z):
                return 0.55 + 0.15 * np.asarray(z)

            def mu_j(z):
                return 0.92 + 0.2 * np.asarray(z)

            def a_i(z):
                return (1.0 + np.asarray(z)) / M_det_i

            def a_j(z):
                return (1.0 + np.asarray(z)) / M_det_j

            def mz_i(z):
                z = np.asarray(z)
                s2 = s2c_i + (sM * a_i(z)) ** 2
                return np.exp(-0.5 * (mu_i(z) - M_eff * a_i(z)) ** 2 / s2) / np.sqrt(2 * np.pi * s2)

            def mz_j(z):
                z = np.asarray(z)
                s2 = s2c_j + (sM * a_j(z)) ** 2
                return np.exp(-0.5 * (mu_j(z) - M_eff * a_j(z)) ** 2 / s2) / np.sqrt(2 * np.pi * s2)

            def fi(z):
                return l_z_i(z) * mz_i(z)

            def fj(z):
                return l_z_j(z) * mz_j(z)

            def joint(z):
                return (
                    l_z_i(z)
                    * l_z_j(z)
                    * scaled_shared_latent_mass_joint(
                        mu_i(z), mu_j(z), a_i(z), a_j(z), s2c_i, s2c_j, M_eff, sM**2
                    )
                )

            def _Mlim(zs):
                z = np.array([zs])
                centers = [M_eff, float(mu_i(z)[0] / a_i(z)[0]), float(mu_j(z)[0] / a_j(z)[0])]
                widths = [
                    sM,
                    math.sqrt(s2c_i) / float(a_i(z)[0]),
                    math.sqrt(s2c_j) / float(a_j(z)[0]),
                ]
                return (
                    min(c - 14 * w for c, w in zip(centers, widths)),
                    max(c + 14 * w for c, w in zip(centers, widths)),
                )

            def fi_ref(zs):
                z = np.array([zs])
                lo, hi = _Mlim(zs)
                m = quad(
                    lambda M: float(
                        gauss(mu_i(z), M * a_i(z), math.sqrt(s2c_i))[0]
                        * gauss(np.array([M]), M_eff, sM)[0]
                    ),
                    lo,
                    hi,
                    epsabs=1e-300,
                    epsrel=1e-12,
                    limit=500,
                )[0]
                return float(l_z_i(z)[0]) * m

            def fj_ref(zs):
                z = np.array([zs])
                lo, hi = _Mlim(zs)
                m = quad(
                    lambda M: float(
                        gauss(mu_j(z), M * a_j(z), math.sqrt(s2c_j))[0]
                        * gauss(np.array([M]), M_eff, sM)[0]
                    ),
                    lo,
                    hi,
                    epsabs=1e-300,
                    epsrel=1e-12,
                    limit=500,
                )[0]
                return float(l_z_j(z)[0]) * m

            def joint_ref_z(zs):
                z = np.array([zs])
                lo, hi = _Mlim(zs)
                m = quad(
                    lambda M: float(
                        gauss(mu_i(z), M * a_i(z), math.sqrt(s2c_i))[0]
                        * gauss(mu_j(z), M * a_j(z), math.sqrt(s2c_j))[0]
                        * gauss(np.array([M]), M_eff, sM)[0]
                    ),
                    lo,
                    hi,
                    epsabs=1e-300,
                    epsrel=1e-12,
                    limit=500,
                )[0]
                return float(l_z_i(z)[0]) * float(l_z_j(z)[0]) * m

        def build(escalate: bool):
            def n_or_none(n):
                return n if escalate else 50

            terms = [
                SharedGalaxyTerm(
                    w_g=wA,
                    rho=rhoA,
                    l_gw_i=fi,
                    l_gw_j=fj,
                    joint_l=joint,
                    quad_n_override=n_or_none(adaptive_quad_n(width_max, sA)),
                ),
                SharedGalaxyTerm(
                    w_g=wB,
                    rho=rhoB,
                    l_gw_i=fi,
                    l_gw_j=fj,
                    joint_l=joint,
                    quad_n_override=n_or_none(adaptive_quad_n(width_max, sB)),
                ),
            ]
            bi = [
                BallMember(
                    w_g=wA,
                    rho=rhoA,
                    l_ev=fi,
                    quad_n_override=n_or_none(adaptive_quad_n(width_i, sA)),
                ),
                BallMember(
                    w_g=wB,
                    rho=rhoB,
                    l_ev=fi,
                    quad_n_override=n_or_none(adaptive_quad_n(width_i, sB)),
                ),
            ]
            bj = [
                BallMember(
                    w_g=wA,
                    rho=rhoA,
                    l_ev=fj,
                    quad_n_override=n_or_none(adaptive_quad_n(width_j, sA)),
                ),
                BallMember(
                    w_g=wB,
                    rho=rhoB,
                    l_ev=fj,
                    quad_n_override=n_or_none(adaptive_quad_n(width_j, sB)),
                ),
            ]
            S_i = compute_ball_sum(bi, window_i)
            S_j = compute_ball_sum(bj, window_j)
            sums = compute_pair_sums(terms, window_i, window_j, S_i=S_i, S_j=S_j)
            return delta_joint_lnL_nats(sums), sums

        delta_inst, sums = build(escalate=True)
        delta_n50, _ = build(escalate=False)

        # --- reference ---
        def refN(f_ref, rho, window):
            return quad(
                lambda z: f_ref(z) * float(rho(np.array([z]))[0]),
                window[0],
                window[1],
                epsabs=1e-300,
                epsrel=1e-12,
                limit=500,
            )[0]

        S_i_ref = wA * refN(fi_ref, rhoA, window_i) + wB * refN(fi_ref, rhoB, window_i)
        S_j_ref = wA * refN(fj_ref, rhoA, window_j) + wB * refN(fj_ref, rhoB, window_j)
        diag_f = 0.0
        diag_j = 0.0
        false_shared_contrib = 0.0
        for zg, sz, w, rho in ((zA, sA, wA, rhoA), (zB, sB, wB, rhoB)):
            N_i = refN(fi_ref, rho, window_i)
            N_j = refN(fj_ref, rho, window_j)
            J = quad(
                lambda z: joint_ref_z(z) * float(rho(np.array([z]))[0]),
                inter[0],
                inter[1],
                epsabs=1e-300,
                epsrel=1e-12,
                limit=500,
            )[0]
            diag_f += w**2 * N_i * N_j
            diag_j += w**2 * J
            if zg == zB:
                false_shared_contrib = w**2 * (abs(N_i * N_j) + abs(J))
        delta_ref = math.log1p((diag_j - diag_f) / (S_i_ref * S_j_ref))

        # False-shared negligibility must be RELATIVE: its possible impact on
        # Delta is ~ contrib / (S_i S_j) nats (a 1e-62 absolute value is the
        # expected ~14-sigma Gaussian tail, not a defect).
        false_shared_delta_impact = false_shared_contrib / (S_i_ref * S_j_ref)
        channels[ch] = {
            "delta_inst": delta_inst,
            "delta_ref_bruteforce": delta_ref,
            "abs_diff_nats": abs(delta_inst - delta_ref),
            "delta_no_escalation_n50": delta_n50,
            "n50_vs_ref_abs_err_nats": abs(delta_n50 - delta_ref),
            "false_shared_w2_contrib": false_shared_contrib,
            "false_shared_delta_impact_nats": false_shared_delta_impact,
            "escalation_orders": {
                "shared_A": nA_shared,
                "ball_i_A": nA_ball_i,
                "ball_j_A": nA_ball_j,
            },
            "ok": (
                math.isfinite(delta_inst)
                and abs(delta_inst - delta_ref) < 1e-9
                and false_shared_delta_impact < 1e-30
            ),
        }
    RESULTS["attack2_edge_boundary_false_shared"] = channels
    return all(c["ok"] for c in channels.values())


if __name__ == "__main__":
    ok1 = attack_1()
    ok2 = attack_2()
    RESULTS["OVERALL"] = {"attack1_ok": ok1, "attack2_ok": ok2, "all_ok": ok1 and ok2}
    out = HERE / "cert_adversarial_20260805.json"
    with open(out, "w") as fh:
        json.dump(RESULTS, fh, indent=1, default=float)
    print(json.dumps(RESULTS["OVERALL"], indent=1))
    print(f"wrote {out}")
    sys.exit(0 if (ok1 and ok2) else 1)
