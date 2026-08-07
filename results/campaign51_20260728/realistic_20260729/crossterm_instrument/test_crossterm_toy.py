"""Self-contained toy-limit tests for the FIXED cross-term instrument core.

Runs WITHOUT any production data: exercises only the pure math core of
``crossterm_instrument.py`` (fixed_quad kernels, per-galaxy numerators, the
pairwise joint integral, the shared-latent-mass closed forms, and the
delta_joint_lnL_nats assembly). The GW likelihoods are stand-in 1D Gaussians in
z — the sky factors of the production 3D Gaussian are constant along the z ray
for a fixed galaxy and cancel in every ratio tested here.

Coverage (rebuilt 2026-08-05 against instrument_math_review_20260805.json;
DEFECT-4 fix — the pair-level n_shared >= 2 regime is now the CENTER of the
suite, not the gap):
  0.  kernel normalization.
  1a. single shared galaxy, sigma_z -> 0: pair-level Delta -> 0 at O(sigma^2).
  1b. distinct hosts: per-galaxy joint integral factorizes as sigma_z -> 0.
  1c. n_shared = 2 (reviewer check-B configuration): pair-level Delta -> 0 at
      O(sigma^2) AND matches an independent adaptive-quad Eq.(31) brute force
      at every sigma — the exact regime the DEFECTIVE build failed.
  1d. n_shared = 2 with balls STRICT SUPERSETS of the shared set (S_i, S_j
      plumbing) vs brute force.
  1e. the production 4-sigma truncation floor: Delta -> ln Z_4sigma
      (~ -6.334e-5 nats) as sigma_z -> 0, single shared galaxy.
  2.  single shared galaxy: closed-form + high-precision quad reference.
  2b. reviewer check-C two-universe discrimination: identical OLD-schema
      quintuples, different Eq.(31) truths — the fixed instrument must now
      distinguish them and match brute force on BOTH.
  2c. 2D shared-latent-mass: closed form vs adaptive quad (incl. the
      reviewer's check-D triples verbatim), sigma_M -> 0 factorization
      identity, joint_l wiring, and the sigma_z -> 0 pair-level 2D limit
      against a fully independent nested-quadrature reference.
  3.  symmetry: Delta_ij == Delta_ji (with superset balls).
  4.  guards: empty/inverted windows, n_shared == 0, undefined references.
  5.  randomized finiteness/no-NaN with superset balls.

Run:
    cd /home/jasper/Repositories/MasterThesisCode && uv run pytest \
        results/campaign51_20260728/realistic_20260729/crossterm_instrument/test_crossterm_toy.py -v
"""

import math
import sys
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
from scipy.integrate import quad

sys.path.insert(0, str(Path(__file__).resolve().parent))

from crossterm_instrument import (  # noqa: E402
    QUAD_N_ADAPTIVE_CAP,
    QUAD_N_PRODUCTION,
    R_ADAPTIVE_THRESHOLD,
    BallMember,
    GalaxyZKernel,
    PairSums,
    SharedGalaxyTerm,
    ZFunc,
    adaptive_quad_n,
    compute_ball_sum,
    compute_pair_sums,
    delta_joint_lnL_nats,
    make_galaxy_z_kernel,
    pair_joint_integral,
    per_galaxy_numerator,
    scaled_shared_latent_mass_joint,
    shared_latent_mass_joint,
)


def gaussian(mu: float, sigma: float) -> ZFunc:
    """Independent Gaussian pdf callable (deliberately not scipy.stats.norm)."""

    def f(z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        result: npt.NDArray[np.float64] = np.exp(-0.5 * ((z - mu) / sigma) ** 2) / (
            sigma * math.sqrt(2.0 * math.pi)
        )
        return result

    return f


def gaussian_value(a: float, b: float, var: float) -> float:
    """N(a; b, var) — scalar Gaussian density, independent implementation."""
    return math.exp(-0.5 * (a - b) ** 2 / var) / math.sqrt(2.0 * math.pi * var)


def n_pdf(z: float, mu: float, sigma: float) -> float:
    return math.exp(-0.5 * ((z - mu) / sigma) ** 2) / (sigma * math.sqrt(2 * math.pi))


# ---------------------------------------------------------------------------
# Independent Eq.(31) pairwise brute force (adaptive quadrature only; own
# kernel normalization — shares NOTHING with the instrument's compute path).
# ---------------------------------------------------------------------------


def rho_ref_factory(z_g: float, sigma_z: float):
    """Production-shaped reference kernel: 4-sigma window, own adaptive-quad Z."""
    z_lo = max(z_g - 4.0 * sigma_z, 1e-6)
    z_hi = z_g + 4.0 * sigma_z
    Z, _ = quad(lambda z: n_pdf(z, z_g, sigma_z), z_lo, z_hi, epsabs=1e-14, limit=200)

    def rho(z: float) -> float:
        return n_pdf(z, z_g, sigma_z) / Z

    return rho, Z


def brute_force_delta(
    gal_i: list[tuple[float, float, float]],
    gal_j: list[tuple[float, float, float]],
    shared_pos: list[int],
    l_i,
    l_j,
    win_i: tuple[float, float],
    win_j: tuple[float, float],
    joint_integrands: dict[int, object] | None = None,
) -> tuple[float, float, float]:
    """Independent Eq.(31) reference: Delta, S_i, S_j.

    gal_i / gal_j: (z_g, sigma_z, w_g) per ball member; shared members must sit
    at the SAME positions in both lists with identical (z_g, sigma_z).
    ``joint_integrands`` optionally overrides the joint (non-rho) integrand per
    shared position (2D channel); default is l_i(z) * l_j(z).

        Delta = ln[(S_i S_j - Sum_shared w^2 N_i N_j + Sum_shared w^2 J)
                   / (S_i S_j)]
    """

    def N_of(l_fn, gal, win):
        z_g, s_z, _ = gal
        rho, _ = rho_ref_factory(z_g, s_z)
        val, _ = quad(lambda z: float(l_fn(z)) * rho(z), *win, epsabs=1e-15, limit=300)
        return val

    z_lo = max(win_i[0], win_j[0])
    z_hi = min(win_i[1], win_j[1])

    def J_of(pos):
        gal = gal_i[pos]
        z_g, s_z, _ = gal
        rho, _ = rho_ref_factory(z_g, s_z)
        if z_lo >= z_hi:
            return 0.0
        if joint_integrands is not None and pos in joint_integrands:
            jl = joint_integrands[pos]
            val, _ = quad(lambda z: float(jl(z)) * rho(z), z_lo, z_hi, epsabs=1e-15, limit=300)
        else:
            val, _ = quad(
                lambda z: float(l_i(z)) * float(l_j(z)) * rho(z),
                z_lo,
                z_hi,
                epsabs=1e-16,
                limit=300,
            )
        return val

    N_i = [N_of(l_i, g, win_i) for g in gal_i]
    N_j = [N_of(l_j, g, win_j) for g in gal_j]
    S_i = sum(g[2] * n for g, n in zip(gal_i, N_i))
    S_j = sum(g[2] * n for g, n in zip(gal_j, N_j))
    corrected = S_i * S_j
    for pos in shared_pos:
        assert gal_i[pos][:2] == gal_j[pos][:2], "shared galaxy must be identical"
        assert gal_i[pos][2] == gal_j[pos][2], "shared galaxy weight must be identical"
        w = gal_i[pos][2]
        corrected -= w * N_i[pos] * w * N_j[pos]
        corrected += w * w * J_of(pos)
    return math.log(corrected) - math.log(S_i * S_j), S_i, S_j


def build_terms(shared_gals, l_i, l_j, quad_n):
    terms = []
    for z_g, s_z, w_g in shared_gals:
        kern = make_galaxy_z_kernel(z_g, s_z, quad_n=quad_n)
        terms.append(SharedGalaxyTerm(w_g=w_g, rho=kern.rho, l_gw_i=l_i, l_gw_j=l_j))
    return terms


# ---------------------------------------------------------------------------
# 0. kernel sanity: rho integrates to ~1 over its window (w_pop = const)
# ---------------------------------------------------------------------------


def test_kernel_normalizes_to_unity_on_window() -> None:
    kern = make_galaxy_z_kernel(z_g=0.4, sigma_z=0.03, quad_n=200)
    integral, _ = quad(lambda z: float(kern.rho(np.array([z]))[0]), kern.z_lo, kern.z_hi)
    assert abs(integral - 1.0) < 1e-8


# ---------------------------------------------------------------------------
# 1a. delta-kernel limit, single shared galaxy: pair-level Delta -> 0
# ---------------------------------------------------------------------------


def test_delta_kernel_limit_single_shared_galaxy() -> None:
    """Pair-level Delta -> 0 as sigma_z -> 0 (ball == shared set).

    With one shared galaxy comprising both balls,
    Delta = log1p((w^2 J - w^2 N_i N_j)/(w^2 N_i N_j)) = ln J - ln(N_i N_j),
    and as rho_g -> delta(z - z_g): J -> l_i(z_g) l_j(z_g) -> N_i N_j.
    High quadrature order is used so the narrow kernels stay resolved — this is
    a math-limit test, not a production-quadrature test.
    """
    l_i = gaussian(0.42, 0.05)
    l_j = gaussian(0.38, 0.06)
    window = (0.1, 0.9)
    w_g = 2.5
    deltas: list[float] = []
    for sigma_z in (0.02, 0.01, 0.005):
        terms = build_terms([(0.40, sigma_z, w_g)], l_i, l_j, quad_n=800)
        sums = compute_pair_sums(terms, window, window, quad_n=800)
        deltas.append(delta_joint_lnL_nats(sums))
    mags = [abs(d) for d in deltas]
    assert all(math.isfinite(d) for d in deltas)
    # O(sigma_z^2) convergence: halving sigma shrinks |Delta| ~4x asymptotically;
    # the first halving still carries higher-order terms (measured ratios with an
    # independent scipy.integrate.quad reference: 0.438 then 0.294), so require
    # strictly < 0.5 per halving plus an absolute end-point bound.
    assert mags[1] < 0.5 * mags[0]
    assert mags[2] < 0.5 * mags[1]
    assert mags[2] < 2e-3


# ---------------------------------------------------------------------------
# 1b. distinct hosts: per-galaxy factorization as sigma_z -> 0
# ---------------------------------------------------------------------------


def test_delta_kernel_limit_distinct_hosts_per_galaxy_factorization() -> None:
    """Per-galaxy joint integral -> product of the one-event numerators.

    Two DISTINCT hosts (different z_g); for EACH galaxy g,
    ln J_g - ln(N_g,i N_g,j) -> 0 as sigma_z -> 0.
    """
    l_i = gaussian(0.40, 0.05)
    l_j = gaussian(0.44, 0.05)
    window = (0.1, 0.9)
    for z_g in (0.38, 0.46):
        residuals: list[float] = []
        for sigma_z in (0.02, 0.01, 0.005):
            kern = make_galaxy_z_kernel(z_g=z_g, sigma_z=sigma_z, quad_n=800)
            n_i = per_galaxy_numerator(l_i, kern.rho, *window, quad_n=800)
            n_j = per_galaxy_numerator(l_j, kern.rho, *window, quad_n=800)
            j_g = pair_joint_integral(l_i, l_j, kern.rho, window, window, quad_n=800)
            residuals.append(abs(math.log(j_g) - math.log(n_i * n_j)))
        assert residuals[1] < 0.4 * residuals[0]
        assert residuals[2] < 0.4 * residuals[1]
        assert residuals[2] < 5e-3


# ---------------------------------------------------------------------------
# 1c. n_shared = 2 (reviewer check-B config): Delta -> 0 AND matches brute force
# ---------------------------------------------------------------------------


def test_pair_level_two_shared_sigma_to_zero_matches_brute_force() -> None:
    """THE DEFECT-1/-2 regression (review check B, same construction verbatim).

    Two shared galaxies (z 0.35 / 0.47, w 1 / 3, balls == shared set); as
    sigma_z -> 0 the true Eq.(31) pair Delta -> 0 at O(sigma^2) — the paper's
    perfect-redshift exactness. The DEFECTIVE build converged to -0.02886 nats
    with the wrong sign; the fixed instrument must track the independent
    adaptive-quad brute force at every sigma (reviewer-measured truth:
    0.072665 / 0.022282 / 0.005871 / 0.001460).
    """
    l_i = gaussian(0.42, 0.05)
    l_j = gaussian(0.40, 0.06)
    win = (0.1, 0.9)
    z_a, z_b, w_a, w_b = 0.35, 0.47, 1.0, 3.0
    reviewer_truth = {
        0.02: 0.07266511574437917,
        0.01: 0.022282229346776816,
        0.005: 0.005871222918222507,
        0.0025: 0.0014603682234266557,
    }
    deltas: list[float] = []
    for s_z in (0.02, 0.01, 0.005, 0.0025):
        gals = [(z_a, s_z, w_a), (z_b, s_z, w_b)]
        quad_n = 800 if s_z >= 0.01 else 4000  # keep narrow kernels resolved
        terms = build_terms(gals, l_i, l_j, quad_n=quad_n)
        sums = compute_pair_sums(terms, win, win, quad_n=quad_n)
        d_inst = delta_joint_lnL_nats(sums)
        d_ref, _, _ = brute_force_delta(gals, gals, [0, 1], l_i, l_j, win, win)
        assert d_inst == pytest.approx(d_ref, rel=1e-6, abs=1e-9), (s_z, d_inst, d_ref)
        assert d_inst == pytest.approx(reviewer_truth[s_z], rel=1e-4), (s_z, d_inst)
        deltas.append(d_inst)
    mags = [abs(d) for d in deltas]
    # O(sigma^2): each halving of sigma shrinks |Delta| by ~4x asymptotically.
    assert mags[1] < 0.5 * mags[0]
    assert mags[2] < 0.5 * mags[1]
    assert mags[3] < 0.5 * mags[2]
    assert mags[3] < 2e-3
    # sign sanity: the true cross-term is POSITIVE here at every sigma
    # (the defective build flipped sign below sigma_z ~ 0.005).
    assert all(d > 0.0 for d in deltas)


def test_pair_level_superset_balls_match_brute_force() -> None:
    """S_i/S_j plumbing: balls are strict supersets of the shared set.

    ball_i has a private galaxy at z=0.30, ball_j one at z=0.55; two shared
    galaxies. Delta must match the independent brute force with the full-ball
    sums in the denominator (the old schema had no S at all — DEFECT-5).
    """
    l_i = gaussian(0.42, 0.05)
    l_j = gaussian(0.40, 0.06)
    win = (0.1, 0.9)
    s_z = 0.02
    shared = [(0.37, s_z, 1.5), (0.45, s_z, 0.8)]
    private_i = (0.30, 0.03, 2.0)
    private_j = (0.55, 0.025, 1.2)
    gal_i = shared + [private_i]
    gal_j = shared + [private_j]

    quad_n = 800
    terms = build_terms(shared, l_i, l_j, quad_n=quad_n)
    members_i = [
        BallMember(w_g=w, rho=make_galaxy_z_kernel(z, s, quad_n=quad_n).rho, l_ev=l_i)
        for z, s, w in gal_i
    ]
    members_j = [
        BallMember(w_g=w, rho=make_galaxy_z_kernel(z, s, quad_n=quad_n).rho, l_ev=l_j)
        for z, s, w in gal_j
    ]
    S_i = compute_ball_sum(members_i, win, quad_n=quad_n)
    S_j = compute_ball_sum(members_j, win, quad_n=quad_n)
    sums = compute_pair_sums(terms, win, win, quad_n=quad_n, S_i=S_i, S_j=S_j)
    d_inst = delta_joint_lnL_nats(sums)

    d_ref, S_i_ref, S_j_ref = brute_force_delta(gal_i, gal_j, [0, 1], l_i, l_j, win, win)
    assert S_i == pytest.approx(S_i_ref, rel=1e-8)
    assert S_j == pytest.approx(S_j_ref, rel=1e-8)
    assert d_inst == pytest.approx(d_ref, rel=1e-6, abs=1e-10)
    # the correction is diluted by the private members: |Delta| strictly smaller
    # than the shared-only construction's.
    sums_bare = compute_pair_sums(terms, win, win, quad_n=quad_n)
    assert abs(d_inst) < abs(delta_joint_lnL_nats(sums_bare))


def test_truncation_floor_sigma_to_zero() -> None:
    """The production 4-sigma host-window floor: Delta -> ln Z_4sigma.

    Since N_i N_j carries Z_g^-2 and J carries Z_g^-1, the single-shared-galaxy
    sigma_z -> 0 limit is ln Z_4sigma = ln(erf(4/sqrt(2))) ~ -6.334e-5 nats —
    the documented, production-faithful floor (module header), NOT zero and NOT
    the defective build's -0.0289.
    """
    l_i = gaussian(0.42, 0.05)
    l_j = gaussian(0.40, 0.06)
    win = (0.1, 0.9)
    terms = build_terms([(0.41, 0.001, 1.0)], l_i, l_j, quad_n=4000)
    sums = compute_pair_sums(terms, win, win, quad_n=4000)
    delta = delta_joint_lnL_nats(sums)
    ln_z_floor = math.log(math.erf(4.0 / math.sqrt(2.0)))  # ~ -6.3342e-5
    assert delta == pytest.approx(ln_z_floor, abs=2e-5)


# ---------------------------------------------------------------------------
# 2. single shared galaxy: closed-form / high-precision reference
# ---------------------------------------------------------------------------


def test_single_shared_galaxy_against_closed_form_and_quad() -> None:
    a_i, s_i = 0.45, 0.04
    a_j, s_j = 0.50, 0.05
    z_g, sigma_z = 0.48, 0.03
    w_g = 1.0
    window_i = (0.2, 0.8)
    window_j = (0.25, 0.85)
    quad_n = 200

    l_i = gaussian(a_i, s_i)
    l_j = gaussian(a_j, s_j)
    kern = make_galaxy_z_kernel(z_g=z_g, sigma_z=sigma_z, quad_n=quad_n)

    terms = [SharedGalaxyTerm(w_g=w_g, rho=kern.rho, l_gw_i=l_i, l_gw_j=l_j)]
    sums = compute_pair_sums(terms, window_i, window_j, quad_n=quad_n)

    # --- independent high-precision reference (scipy.integrate.quad) --------
    z_lo_host = z_g - 4.0 * sigma_z
    z_hi_host = z_g + 4.0 * sigma_z
    Z_ref, _ = quad(lambda z: n_pdf(z, z_g, sigma_z), z_lo_host, z_hi_host, epsabs=1e-14)

    def rho_ref(z: float) -> float:
        return n_pdf(z, z_g, sigma_z) / Z_ref

    N_i_ref, _ = quad(lambda z: n_pdf(z, a_i, s_i) * rho_ref(z), *window_i, epsabs=1e-15, limit=200)
    N_j_ref, _ = quad(lambda z: n_pdf(z, a_j, s_j) * rho_ref(z), *window_j, epsabs=1e-15, limit=200)
    zjl = max(window_i[0], window_j[0])
    zjh = min(window_i[1], window_j[1])
    J_ref, _ = quad(
        lambda z: n_pdf(z, a_i, s_i) * n_pdf(z, a_j, s_j) * rho_ref(z),
        zjl,
        zjh,
        epsabs=1e-16,
        limit=200,
    )

    assert sums.sum_wN_i == pytest.approx(w_g * N_i_ref, rel=1e-9)
    assert sums.sum_wN_j == pytest.approx(w_g * N_j_ref, rel=1e-9)
    assert sums.sum_wJ == pytest.approx(w_g * J_ref, rel=1e-9)
    assert sums.shared_diag_fact == pytest.approx(w_g**2 * N_i_ref * N_j_ref, rel=1e-9)
    assert sums.shared_diag_joint == pytest.approx(w_g**2 * J_ref, rel=1e-9)
    # toy convention: ball == shared set -> S defaults to the shared w^1 sums
    assert sums.S_i == sums.sum_wN_i
    assert sums.S_j == sums.sum_wN_j

    # --- closed-form product-of-Gaussians reference (untruncated) -----------
    # N(z;a_i,s_i^2) N(z;a_j,s_j^2) = N(a_i; a_j, s_i^2+s_j^2) N(z; m_ij, v_ij)
    v_sum = s_i**2 + s_j**2
    m_ij = (a_i * s_j**2 + a_j * s_i**2) / v_sum
    v_ij = (s_i**2 * s_j**2) / v_sum
    J_closed = (
        gaussian_value(a_i, a_j, v_sum) * gaussian_value(m_ij, z_g, v_ij + sigma_z**2) / Z_ref
    )
    N_i_closed = gaussian_value(a_i, z_g, s_i**2 + sigma_z**2) / Z_ref
    N_j_closed = gaussian_value(a_j, z_g, s_j**2 + sigma_z**2) / Z_ref

    # Window truncation is negligible here (integrand support ~0.48 +/- a few
    # 0.02-widths, deep inside every window), so 1e-8 relative agreement holds.
    assert sums.sum_wJ == pytest.approx(w_g * J_closed, rel=1e-8)
    assert sums.sum_wN_i == pytest.approx(w_g * N_i_closed, rel=1e-8)
    assert sums.sum_wN_j == pytest.approx(w_g * N_j_closed, rel=1e-8)

    # single shared galaxy, ball == shared:
    # Delta = ln[(w^2 N N - w^2 N N + w^2 J)/(w^2 N N)] = ln J - ln(N_i N_j)
    delta_closed = math.log(J_closed) - math.log(N_i_closed * N_j_closed)
    assert delta_joint_lnL_nats(sums) == pytest.approx(delta_closed, rel=1e-8, abs=1e-10)


# ---------------------------------------------------------------------------
# 2b. reviewer check C: two universes with identical OLD-schema quintuples
# ---------------------------------------------------------------------------


def test_check_c_two_universe_discrimination() -> None:
    """The fixed instrument distinguishes the reviewer's check-C universes.

    Reviewer construction verbatim: five shared galaxies; a nullspace weight
    perturbation preserves ALL w^1 sums (the old emitted quintuple was
    identical to 1.7e-16 rel) while the true Eq.(31) cross-terms differ by
    10.3% (0.013724 vs 0.015303 nats). The fixed w^2-diagonal formula must
    (a) match the independent brute force on BOTH universes and (b) separate
    them.
    """
    l_i = gaussian(0.42, 0.05)
    l_j = gaussian(0.40, 0.06)
    win = (0.1, 0.9)
    quad_n = 800
    gals_z = [(0.34, 0.02), (0.38, 0.025), (0.42, 0.03), (0.46, 0.025), (0.50, 0.02)]
    per = []
    for z_g, s_z in gals_z:
        kern = make_galaxy_z_kernel(z_g, s_z, quad_n=quad_n)
        N_i = per_galaxy_numerator(l_i, kern.rho, *win, quad_n=quad_n)
        N_j = per_galaxy_numerator(l_j, kern.rho, *win, quad_n=quad_n)
        J = pair_joint_integral(l_i, l_j, kern.rho, win, win, quad_n=quad_n)
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
    assert np.all(w1 > 0)
    assert float(np.max(np.abs(A @ (w1 - w0)))) < 1e-12  # w^1 sums preserved

    deltas = []
    for w_vec in (w0, w1):
        gals = [(z, s, float(w)) for (z, s), w in zip(gals_z, w_vec)]
        terms = build_terms(gals, l_i, l_j, quad_n=quad_n)
        sums = compute_pair_sums(terms, win, win, quad_n=quad_n)
        d_inst = delta_joint_lnL_nats(sums)
        d_ref, _, _ = brute_force_delta(gals, gals, list(range(5)), l_i, l_j, win, win)
        assert d_inst == pytest.approx(d_ref, rel=1e-6, abs=1e-10)
        deltas.append(d_inst)
    # reviewer-measured truths for the same construction
    assert deltas[0] == pytest.approx(0.013723819496744838, rel=1e-4)
    assert deltas[1] == pytest.approx(0.015303175488242005, rel=1e-4)
    rel_sep = abs(deltas[0] - deltas[1]) / max(abs(deltas[0]), abs(deltas[1]))
    assert rel_sep > 0.05  # the universes are now DISTINGUISHED (was 0 by schema)


# ---------------------------------------------------------------------------
# 2c. 2D channel: shared latent mass marginalized ONCE
# ---------------------------------------------------------------------------


def test_shared_latent_mass_closed_form_vs_quad_reviewer_triples() -> None:
    """Closed form == adaptive quad, incl. the reviewer's check-D triples."""
    triples = [
        # (mu_i, mu_j, s_i, s_j, M_g, sM, reviewer_log_ratio)
        (1.00, 1.02, 0.03, 0.04, 1.0, 0.6, 2.408054240100485),
        (1.00, 1.30, 0.03, 0.04, 1.0, 0.6, -15.404154191011509),
        (1.00, 1.02, 0.10, 0.12, 1.0, 2.0, 2.5438860139384305),
    ]
    for mu_i, mu_j, s_i, s_j, M_g, sM, ref_log_ratio in triples:
        exact = float(shared_latent_mass_joint(mu_i, mu_j, s_i**2, s_j**2, M_g, sM**2))
        num, _ = quad(
            lambda M, mu_i=mu_i, mu_j=mu_j, s_i=s_i, s_j=s_j, M_g=M_g, sM=sM: (
                n_pdf(mu_i, M, s_i) * n_pdf(mu_j, M, s_j) * n_pdf(M, M_g, sM)
            ),
            M_g - 12 * sM,
            M_g + 12 * sM,
            limit=300,
        )
        assert exact == pytest.approx(num, rel=1e-5)
        product = gaussian_value(mu_i, M_g, s_i**2 + sM**2) * gaussian_value(
            mu_j, M_g, s_j**2 + sM**2
        )
        assert math.log(exact / product) == pytest.approx(ref_log_ratio, abs=1e-9)


def test_scaled_shared_latent_mass_joint_vs_quad_unequal_scales() -> None:
    """The a_e = (1+z)/M_det,e scale handling against adaptive quadrature."""
    mu_i, mu_j = 0.52, 0.49
    a_i, a_j = 1.3, 0.8
    s2c_i, s2c_j = 0.03**2, 0.05**2
    M_g, sM = 0.40, 0.25
    closed = float(scaled_shared_latent_mass_joint(mu_i, mu_j, a_i, a_j, s2c_i, s2c_j, M_g, sM**2))
    num, _ = quad(
        lambda M: (
            n_pdf(mu_i, M * a_i, math.sqrt(s2c_i))
            * n_pdf(mu_j, M * a_j, math.sqrt(s2c_j))
            * n_pdf(M, M_g, sM)
        ),
        M_g - 12 * sM,
        M_g + 12 * sM,
        limit=300,
    )
    assert closed == pytest.approx(num, rel=1e-10)


def test_shared_latent_mass_sigma_M_zero_factorizes() -> None:
    """sM -> 0: the joint reduces EXACTLY to the product of per-event marginals.

    (Product-of-Gaussians identity; this is why the 2D Delta vanishes only when
    BOTH sigma_z -> 0 and sigma_M -> 0.)
    """
    rng = np.random.default_rng(7)
    for _ in range(20):
        mu_i, mu_j = rng.normal(1.0, 0.3, size=2)
        s2_i, s2_j = rng.uniform(0.001, 0.05, size=2)
        M_g = rng.normal(1.0, 0.2)
        joint0 = float(shared_latent_mass_joint(mu_i, mu_j, s2_i, s2_j, M_g, 0.0))
        product = gaussian_value(mu_i, M_g, s2_i) * gaussian_value(mu_j, M_g, s2_j)
        assert joint0 == pytest.approx(product, rel=1e-12)


def test_2d_pair_level_shared_latent_mass_toy() -> None:
    """Pair-level 2D toy: joint_l wiring + independent nested-quad reference +
    the sigma_z -> 0 limit against the closed-form mass-coupling ratio.

    Toy staging (pure numpy, no production imports): detector-frame masses
    M_i = 2.0, M_j = 3.0; per-event conditional fractional-mass observations
    mu_i = 0.52 +/- sqrt(s2c_i), mu_j = 0.50 +/- sqrt(s2c_j) (z-constant);
    a_e(z) = (1+z)/M_e; host M_eff = 1.0, sigma_M = 0.6 (production-plausible
    ~60%). Factorized callables carry each event's OWN independent-M marginal;
    the joint carries the ONE-shared-M closed form.
    """
    M_i_det, M_j_det = 2.0, 3.0
    mu_i, mu_j = 0.52, 0.50
    s2c_i, s2c_j = 0.03**2, 0.04**2
    M_eff, sM = 1.0, 0.6
    z_g = 0.41
    win = (0.1, 0.9)
    l3d_i = gaussian(0.42, 0.05)
    l3d_j = gaussian(0.40, 0.06)

    def a_of(z, M_det):
        return (1.0 + z) / M_det

    def mz_i(z):
        a = a_of(z, M_i_det)
        v = s2c_i + (sM * a) ** 2
        return np.exp(-0.5 * (mu_i - M_eff * a) ** 2 / v) / np.sqrt(2 * np.pi * v)

    def mz_j(z):
        a = a_of(z, M_j_det)
        v = s2c_j + (sM * a) ** 2
        return np.exp(-0.5 * (mu_j - M_eff * a) ** 2 / v) / np.sqrt(2 * np.pi * v)

    def mz_joint(z):
        return scaled_shared_latent_mass_joint(
            mu_i, mu_j, a_of(z, M_i_det), a_of(z, M_j_det), s2c_i, s2c_j, M_eff, sM**2
        )

    def l_i(z):
        return l3d_i(z) * mz_i(z)

    def l_j(z):
        return l3d_j(z) * mz_j(z)

    def joint_l(z):
        return l3d_i(z) * l3d_j(z) * mz_joint(z)

    # --- (a) wiring: pair_joint_integral honors joint_l ----------------------
    sigma_z = 0.02
    kern = make_galaxy_z_kernel(z_g, sigma_z, quad_n=800)
    J_via_term = pair_joint_integral(l_i, l_j, kern.rho, win, win, quad_n=800, joint_l=joint_l)
    from scipy.integrate import fixed_quad

    J_manual = float(fixed_quad(lambda z: joint_l(z) * kern.rho(z), *win, n=800)[0])
    assert J_via_term == pytest.approx(J_manual, rel=1e-14)
    # and it is NOT the factorized product integral (the DEFECT-3 signature)
    J_factorized = pair_joint_integral(l_i, l_j, kern.rho, win, win, quad_n=800)
    assert abs(math.log(J_via_term / J_factorized)) > 0.5

    # --- (b) independent nested-quadrature Eq.(31) reference ------------------
    def mz_joint_ref(z: float) -> float:
        a_i, a_j = a_of(z, M_i_det), a_of(z, M_j_det)
        val, _ = quad(
            lambda M: (
                n_pdf(mu_i, M * a_i, math.sqrt(s2c_i))
                * n_pdf(mu_j, M * a_j, math.sqrt(s2c_j))
                * n_pdf(M, M_eff, sM)
            ),
            M_eff - 10 * sM,
            M_eff + 10 * sM,
            limit=200,
        )
        return val

    def joint_l_ref(z: float) -> float:
        return float(l3d_i(z)) * float(l3d_j(z)) * mz_joint_ref(z)

    w_g = 1.7
    gals = [(z_g, sigma_z, w_g)]
    terms = [SharedGalaxyTerm(w_g=w_g, rho=kern.rho, l_gw_i=l_i, l_gw_j=l_j, joint_l=joint_l)]
    sums = compute_pair_sums(terms, win, win, quad_n=800)
    d_inst = delta_joint_lnL_nats(sums)
    d_ref, _, _ = brute_force_delta(
        gals, gals, [0], l_i, l_j, win, win, joint_integrands={0: joint_l_ref}
    )
    assert d_inst == pytest.approx(d_ref, rel=1e-6, abs=1e-9)

    # --- (c) sigma_z -> 0: Delta -> ln Z + ln(mzjoint/(mz_i mz_j))|_{z_g} -----
    sigma_small = 0.004
    kern_s = make_galaxy_z_kernel(z_g, sigma_small, quad_n=2000)
    terms_s = [SharedGalaxyTerm(w_g=w_g, rho=kern_s.rho, l_gw_i=l_i, l_gw_j=l_j, joint_l=joint_l)]
    sums_s = compute_pair_sums(terms_s, win, win, quad_n=2000)
    d_small = delta_joint_lnL_nats(sums_s)
    zg_arr = np.array([z_g])
    ratio_closed = math.log(
        float(mz_joint(zg_arr)[0]) / (float(mz_i(zg_arr)[0]) * float(mz_j(zg_arr)[0]))
    ) + math.log(math.erf(4.0 / math.sqrt(2.0)))
    # the mass-coupling residual survives perfect redshift (O(sigma_z^2) drift
    # tolerance); the 1D part of the cross-term has vanished by here.
    assert d_small == pytest.approx(ratio_closed, abs=0.02)
    assert abs(ratio_closed) > 0.1  # the 2D residual is genuinely nonzero


# ---------------------------------------------------------------------------
# 2d. R-2 remedy: adaptive quadrature escalation for high-R geometries
# ---------------------------------------------------------------------------


def test_adaptive_quad_n_policy() -> None:
    """The R-2 escalation policy: baseline preserved, margin honored, capped.

    Regression clause: for R = window/sigma_z <= 30 the returned order IS the
    production 50 (bit-identical behavior in the validated regime). Above:
    n = ceil(50 * (R/30) * 4), so the node-per-sigma density is 4x the
    validated R=30 point; at the single production instance above R=45
    (joint_r1/1D pair (114, 1035): sigma_z=0.0019636, W_max=0.1338662,
    R=68.17 — rr1_worst_R_table.json) the result must be >= 400, the order
    measured sufficient at that worst geometry. Cap: 4000.
    """
    # validated regime: base order returned exactly
    assert adaptive_quad_n(0.8, 0.8 / 30.0) == QUAD_N_PRODUCTION
    assert adaptive_quad_n(0.3, 0.3 / 29.9) == QUAD_N_PRODUCTION
    assert adaptive_quad_n(1e-3, 1.0) == QUAD_N_PRODUCTION  # R << 1
    # guards: non-positive / non-finite sigma or width -> base order
    assert adaptive_quad_n(0.8, 0.0) == QUAD_N_PRODUCTION
    assert adaptive_quad_n(0.8, -1.0) == QUAD_N_PRODUCTION
    assert adaptive_quad_n(0.0, 0.01) == QUAD_N_PRODUCTION
    assert adaptive_quad_n(float("nan"), 0.01) == QUAD_N_PRODUCTION
    assert adaptive_quad_n(0.8, float("nan")) == QUAD_N_PRODUCTION
    # escalation formula: n = ceil(50 * (R/30) * 4)
    assert adaptive_quad_n(0.9, 0.02) == math.ceil(50 * (45.0 / 30.0) * 4)  # 300
    # the production worst instance: R = 68.17 -> n = 455 >= 400
    n_worst = adaptive_quad_n(0.1338662494270748, 0.0019636170795231)
    assert n_worst == 455
    assert n_worst >= 400
    # monotone non-decreasing in R
    ns = [adaptive_quad_n(w, 0.002) for w in (0.05, 0.1, 0.2, 0.4, 0.8, 1.2)]
    assert ns == sorted(ns)
    # cap honored (R = 600 hits the cap exactly; beyond stays capped)
    assert adaptive_quad_n(1.2, 0.002) == QUAD_N_ADAPTIVE_CAP
    assert adaptive_quad_n(10.0, 0.002) == QUAD_N_ADAPTIVE_CAP
    # threshold constant is the documented one
    assert R_ADAPTIVE_THRESHOLD == 30.0


def test_high_R_adaptive_escalation_matches_adaptive_quad() -> None:
    """R-2 regression: a production-like high-R pair, fixed vs adaptive quad.

    Geometry mirrors the single production instance above R=45
    (rr1_worst_R_table.json): window width 0.1338662, sigma_z = 0.0019636
    -> R = 68.17. The instrument path is EXACTLY the production one — base
    quad_n = 50 with per-term/per-member ``quad_n_override =
    adaptive_quad_n(width, sigma_z)`` (455 here) — and must agree with an
    independent points-hinted scipy.integrate.quad reference to rel <= 1e-8
    on Delta (measured: ~3e-10) and rel <= 1e-10 per integral. The
    UNescalated n=50 path is also measured and must show the documented
    degradation (>= 1e-3 nats absolute here, undiluted; calibration scale
    rr1_boundary_check.json) — the defect this remedy closes.
    """
    win = (0.30, 0.30 + 0.1338662494270748)
    sigma_z = 0.0019636170795231
    z_g = 0.365
    w_g = 2.0
    l_i = gaussian(0.37, 0.03)
    l_j = gaussian(0.355, 0.04)
    width = win[1] - win[0]
    assert width / sigma_z == pytest.approx(68.17, abs=0.01)
    n_ad = adaptive_quad_n(width, sigma_z)
    assert n_ad == 455

    # --- independent adaptive-quadrature reference (points-hinted) ----------
    z_lo_h = max(z_g - 4.0 * sigma_z, 1e-6)
    z_hi_h = z_g + 4.0 * sigma_z
    Z_ref, _ = quad(lambda z: n_pdf(z, z_g, sigma_z), z_lo_h, z_hi_h, epsabs=1e-15, epsrel=1e-13)

    def rho_ref(z: float) -> float:
        return n_pdf(z, z_g, sigma_z) / Z_ref

    pts = [z_g - 4.0 * sigma_z, z_g, z_g + 4.0 * sigma_z]
    N_i_ref, _ = quad(
        lambda z: float(l_i(z)) * rho_ref(z),
        *win,
        epsabs=1e-15,
        epsrel=1e-13,
        limit=400,
        points=pts,
    )
    N_j_ref, _ = quad(
        lambda z: float(l_j(z)) * rho_ref(z),
        *win,
        epsabs=1e-15,
        epsrel=1e-13,
        limit=400,
        points=pts,
    )
    J_ref, _ = quad(
        lambda z: float(l_i(z)) * float(l_j(z)) * rho_ref(z),
        *win,
        epsabs=1e-15,
        epsrel=1e-13,
        limit=400,
        points=pts,
    )
    delta_ref = math.log(J_ref) - math.log(N_i_ref * N_j_ref)

    kern = make_galaxy_z_kernel(z_g, sigma_z)  # Z_g at production n=50 (R = 8)
    assert kern.z_norm == pytest.approx(Z_ref, rel=1e-12)

    # --- escalated instrument path (ball == shared single galaxy) -----------
    terms_ad = [
        SharedGalaxyTerm(w_g=w_g, rho=kern.rho, l_gw_i=l_i, l_gw_j=l_j, quad_n_override=n_ad)
    ]
    sums_ad = compute_pair_sums(terms_ad, win, win, quad_n=QUAD_N_PRODUCTION)
    d_ad = delta_joint_lnL_nats(sums_ad)
    assert sums_ad.sum_wN_i / w_g == pytest.approx(N_i_ref, rel=1e-10)
    assert sums_ad.sum_wN_j / w_g == pytest.approx(N_j_ref, rel=1e-10)
    assert sums_ad.sum_wJ / w_g == pytest.approx(J_ref, rel=1e-10)
    assert d_ad == pytest.approx(delta_ref, rel=1e-8)

    # --- the unescalated n=50 path shows the documented degradation ---------
    terms_50 = [SharedGalaxyTerm(w_g=w_g, rho=kern.rho, l_gw_i=l_i, l_gw_j=l_j)]
    sums_50 = compute_pair_sums(terms_50, win, win, quad_n=QUAD_N_PRODUCTION)
    d_50 = delta_joint_lnL_nats(sums_50)
    assert abs(d_50 - delta_ref) > 1e-3  # measured ~2.4e-2 at this geometry
    assert abs(d_50 - delta_ref) > 1e4 * abs(d_ad - delta_ref)

    # --- superset balls: BallMember escalation matches the reference --------
    z_p, s_p, w_p = 0.40, 0.03, 1.5  # private wide-kernel member (R ~ 4.5)
    kern_p = make_galaxy_z_kernel(z_p, s_p)
    Z_p_ref, _ = quad(
        lambda z: n_pdf(z, z_p, s_p),
        z_p - 4.0 * s_p,
        z_p + 4.0 * s_p,
        epsabs=1e-15,
        epsrel=1e-13,
    )
    N_p_ref, _ = quad(
        lambda z: float(l_i(z)) * n_pdf(z, z_p, s_p) / Z_p_ref,
        *win,
        epsabs=1e-15,
        epsrel=1e-13,
        limit=400,
    )
    members_i = [
        BallMember(
            w_g=w_g,
            rho=kern.rho,
            l_ev=l_i,
            quad_n_override=adaptive_quad_n(width, sigma_z),
        ),
        BallMember(
            w_g=w_p,
            rho=kern_p.rho,
            l_ev=l_i,
            quad_n_override=adaptive_quad_n(width, s_p),  # R <= 30 -> stays 50
        ),
    ]
    assert members_i[1].quad_n_override == QUAD_N_PRODUCTION
    S_i = compute_ball_sum(members_i, win, quad_n=QUAD_N_PRODUCTION)
    S_i_ref = w_g * N_i_ref + w_p * N_p_ref
    assert S_i == pytest.approx(S_i_ref, rel=1e-9)
    S_j = sums_ad.sum_wN_j  # ball_j == the shared single galaxy (S_j = w_g N_j)
    sums_sup = compute_pair_sums(terms_ad, win, win, quad_n=QUAD_N_PRODUCTION, S_i=S_i, S_j=S_j)
    d_sup = delta_joint_lnL_nats(sums_sup)
    corrected_ref = S_i_ref * (w_g * N_j_ref) - w_g**2 * N_i_ref * N_j_ref + w_g**2 * J_ref
    d_sup_ref = math.log(corrected_ref) - math.log(S_i_ref * (w_g * N_j_ref))
    assert d_sup == pytest.approx(d_sup_ref, rel=1e-8)


# ---------------------------------------------------------------------------
# 3. symmetry: Delta_ij == Delta_ji (with superset balls)
# ---------------------------------------------------------------------------


def test_symmetry_delta_ij_equals_delta_ji() -> None:
    l_i = gaussian(0.35, 0.05)
    l_j = gaussian(0.42, 0.07)
    window_i = (0.15, 0.75)
    window_j = (0.20, 0.90)
    galaxies = [(0.34, 0.02, 0.7), (0.40, 0.035, 1.3), (0.45, 0.05, 0.4)]
    private_i = (0.30, 0.03, 0.9)
    private_j = (0.50, 0.02, 1.1)

    terms_ij: list[SharedGalaxyTerm] = []
    terms_ji: list[SharedGalaxyTerm] = []
    kerns = {}
    for z_g, sigma_z, w_g in galaxies + [private_i, private_j]:
        kerns[(z_g, sigma_z)] = make_galaxy_z_kernel(z_g=z_g, sigma_z=sigma_z)
    for z_g, sigma_z, w_g in galaxies:
        kern = kerns[(z_g, sigma_z)]
        terms_ij.append(SharedGalaxyTerm(w_g=w_g, rho=kern.rho, l_gw_i=l_i, l_gw_j=l_j))
        terms_ji.append(SharedGalaxyTerm(w_g=w_g, rho=kern.rho, l_gw_i=l_j, l_gw_j=l_i))

    members_i = [
        BallMember(w_g=w, rho=kerns[(z, s)].rho, l_ev=l_i) for z, s, w in galaxies + [private_i]
    ]
    members_j = [
        BallMember(w_g=w, rho=kerns[(z, s)].rho, l_ev=l_j) for z, s, w in galaxies + [private_j]
    ]
    S_i = compute_ball_sum(members_i, window_i)
    S_j = compute_ball_sum(members_j, window_j)

    sums_ij = compute_pair_sums(terms_ij, window_i, window_j, S_i=S_i, S_j=S_j)
    sums_ji = compute_pair_sums(terms_ji, window_j, window_i, S_i=S_j, S_j=S_i)

    assert sums_ij.sum_wJ == pytest.approx(sums_ji.sum_wJ, rel=1e-14)
    assert sums_ij.sum_wN_i == pytest.approx(sums_ji.sum_wN_j, rel=1e-14)
    assert sums_ij.sum_wN_j == pytest.approx(sums_ji.sum_wN_i, rel=1e-14)
    assert sums_ij.shared_diag_fact == pytest.approx(sums_ji.shared_diag_fact, rel=1e-14)
    assert sums_ij.shared_diag_joint == pytest.approx(sums_ji.shared_diag_joint, rel=1e-14)
    assert sums_ij.S_i == sums_ji.S_j and sums_ij.S_j == sums_ji.S_i
    assert delta_joint_lnL_nats(sums_ij) == pytest.approx(delta_joint_lnL_nats(sums_ji), rel=1e-12)


# ---------------------------------------------------------------------------
# 4. guards: windows, empty shared set, undefined references
# ---------------------------------------------------------------------------


def test_empty_or_inverted_window_guards() -> None:
    l_i = gaussian(0.4, 0.05)
    l_j = gaussian(0.6, 0.05)
    kern = make_galaxy_z_kernel(z_g=0.5, sigma_z=0.02)
    # Disjoint event windows -> empty intersection -> exact 0.0 (never negative).
    assert pair_joint_integral(l_i, l_j, kern.rho, (0.2, 0.45), (0.55, 0.8)) == 0.0
    # Inverted window -> 0.0 (fixed_quad would return a negative value; trap 9).
    assert per_galaxy_numerator(l_i, kern.rho, 0.7, 0.3) == 0.0

    # Ball == shared single galaxy, disjoint windows: corrected joint collapses
    # to exactly zero (S_i S_j - w^2 N N + 0 == 0) -> -inf, not NaN.
    terms = [SharedGalaxyTerm(w_g=1.0, rho=kern.rho, l_gw_i=l_i, l_gw_j=l_j)]
    sums = compute_pair_sums(terms, (0.2, 0.45), (0.55, 0.8))
    assert sums.shared_diag_joint == 0.0
    assert sums.sum_wN_i > 0.0 and sums.sum_wN_j > 0.0
    assert delta_joint_lnL_nats(sums) == -math.inf

    # Superset ball (private member in ball_i): corrected joint stays positive
    # -> finite NEGATIVE Delta (the pair jointly excludes the shared host).
    kern_p = make_galaxy_z_kernel(z_g=0.35, sigma_z=0.03)
    members_i = [
        BallMember(w_g=1.0, rho=kern.rho, l_ev=l_i),
        BallMember(w_g=2.0, rho=kern_p.rho, l_ev=l_i),
    ]
    S_i = compute_ball_sum(members_i, (0.2, 0.45))
    sums_sup = compute_pair_sums(terms, (0.2, 0.45), (0.55, 0.8), S_i=S_i, S_j=sums.sum_wN_j)
    d_sup = delta_joint_lnL_nats(sums_sup)
    assert math.isfinite(d_sup) and d_sup < 0.0

    # n_shared == 0 with positive full-ball sums -> Delta == 0.0 exactly.
    none_shared = compute_pair_sums([], (0.2, 0.45), (0.55, 0.8), S_i=1.3, S_j=0.7)
    assert delta_joint_lnL_nats(none_shared) == 0.0

    # Undefined factorized reference (S <= 0) -> NaN.
    empty = PairSums(
        S_i=0.0,
        S_j=0.0,
        shared_diag_fact=0.0,
        shared_diag_joint=0.0,
        sum_wJ=0.0,
        sum_wN_i=0.0,
        sum_wN_j=0.0,
        sum_w=0.0,
        n_shared=0,
    )
    assert math.isnan(delta_joint_lnL_nats(empty))

    # Floating-point pathological corrected <= 0 -> -inf (clamped, documented).
    patho = PairSums(
        S_i=1.0,
        S_j=1.0,
        shared_diag_fact=2.0,
        shared_diag_joint=0.5,
        sum_wJ=0.5,
        sum_wN_i=1.0,
        sum_wN_j=1.0,
        sum_w=1.0,
        n_shared=1,
    )
    assert delta_joint_lnL_nats(patho) == -math.inf


# ---------------------------------------------------------------------------
# 5. randomized finiteness / no-NaN with superset balls
# ---------------------------------------------------------------------------


def test_randomized_toy_finite_no_nan() -> None:
    rng = np.random.default_rng(20260805)

    def toy_w_pop(z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        result: npt.NDArray[np.float64] = z**2 / (1.0 + z)
        return result

    for trial in range(30):
        mu_i, mu_j = rng.uniform(0.1, 1.0, size=2)
        s_i, s_j = rng.uniform(0.01, 0.2, size=2)
        l_i = gaussian(float(mu_i), float(s_i))
        l_j = gaussian(float(mu_j), float(s_j))
        lo_i = float(mu_i) - rng.uniform(0.5, 4.0) * float(s_i)
        hi_i = float(mu_i) + rng.uniform(0.5, 4.0) * float(s_i)
        lo_j = float(mu_j) - rng.uniform(0.5, 4.0) * float(s_j)
        hi_j = float(mu_j) + rng.uniform(0.5, 4.0) * float(s_j)
        window_i = (max(lo_i, 1e-6), hi_i)
        window_j = (max(lo_j, 1e-6), hi_j)

        terms: list[SharedGalaxyTerm] = []
        members_i: list[BallMember] = []
        members_j: list[BallMember] = []
        for _ in range(int(rng.integers(1, 5))):
            z_g = float(rng.uniform(0.05, 1.2))
            sigma_z = float(rng.uniform(0.005, 0.08))
            w_g = float(rng.uniform(0.1, 10.0))
            w_pop = toy_w_pop if rng.random() < 0.5 else None
            kern = make_galaxy_z_kernel(z_g=z_g, sigma_z=sigma_z, w_pop_eff=w_pop)
            assert isinstance(kern, GalaxyZKernel)
            assert math.isfinite(kern.z_norm) and kern.z_norm > 0.0
            terms.append(SharedGalaxyTerm(w_g=w_g, rho=kern.rho, l_gw_i=l_i, l_gw_j=l_j))
            members_i.append(BallMember(w_g=w_g, rho=kern.rho, l_ev=l_i))
            members_j.append(BallMember(w_g=w_g, rho=kern.rho, l_ev=l_j))
        # private (non-shared) ball members, 0-2 per event
        for members, l_ev in ((members_i, l_i), (members_j, l_j)):
            for _ in range(int(rng.integers(0, 3))):
                z_g = float(rng.uniform(0.05, 1.2))
                sigma_z = float(rng.uniform(0.005, 0.08))
                w_g = float(rng.uniform(0.1, 10.0))
                kern = make_galaxy_z_kernel(z_g=z_g, sigma_z=sigma_z)
                members.append(BallMember(w_g=w_g, rho=kern.rho, l_ev=l_ev))

        S_i = compute_ball_sum(members_i, window_i)
        S_j = compute_ball_sum(members_j, window_j)
        sums = compute_pair_sums(terms, window_i, window_j, S_i=S_i, S_j=S_j)
        for value in (
            sums.S_i,
            sums.S_j,
            sums.shared_diag_fact,
            sums.shared_diag_joint,
            sums.sum_wJ,
            sums.sum_wN_i,
            sums.sum_wN_j,
            sums.sum_w,
        ):
            assert math.isfinite(value), f"trial {trial}: non-finite sum {value}"
            assert value >= 0.0, f"trial {trial}: negative sum {value}"
        # mathematical positivity of the corrected joint (up to roundoff):
        corrected = sums.S_i * sums.S_j - sums.shared_diag_fact + sums.shared_diag_joint
        assert corrected >= -1e-12 * max(sums.S_i * sums.S_j, 1e-300), f"trial {trial}"

        delta = delta_joint_lnL_nats(sums)
        if sums.S_i > 0.0 and sums.S_j > 0.0:
            assert delta == -math.inf or math.isfinite(delta), f"trial {trial}: {delta}"
            if corrected > 0.0:
                assert math.isfinite(delta), f"trial {trial}: delta not finite: {delta}"
        else:
            assert math.isnan(delta)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
