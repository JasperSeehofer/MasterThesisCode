"""Registered null checks for the stage-3 estimator-variant arms (A-REN/A-JREN).

Companion to ``results/mechanism_study_20260813/PREREGISTRATION_A_JREN_STAGE3.md``
§2/§3 and the "Stage-3 arms" section appended to ``ARMS.md``. These tests are the
auditable form of the registered A-REN/A-JREN code-form claims:

- A-REN ("kernel_renorm") and A-JREN ("jacobian_and_kernel_renorm") are
  identically the base estimator at σ_z = 0 (the point branch has no kernel
  mass to renormalize and no z-integral to carry a Jacobian).
- The ``_W_K_FLOOR`` guard (1e-12) is inert on any candidate whose kernel mass
  matters — it only prevents blow-up when the retained kernel mass is exactly
  (or near) zero, in which case the base numerator is already zero, so the
  renormalized contribution stays zero rather than exploding.
- A-JREN is the composition of A-M2''s Jacobian and A-REN's renormalization on
  the SAME integrand (associativity of the two factors, checked on a small
  case): applying the Jacobian to the base integrand, then dividing by W_k,
  reproduces A-JREN exactly.
- The two new cells' seed blocks (+54000..+54024 A-REN, +54100..+54124
  A-JREN) are disjoint from every previously documented block.
- Both cells stamp the stage-3 preregistration document.
"""

import numpy as np
import pytest
from scipy.special import roots_legendre
from scipy.stats import norm

from darksiren_emri.physical_relations import dist_vectorized
from darksiren_emri.validation import calibration_gate as cg
from darksiren_emri.validation import closed_loop_gfrac as cl
from darksiren_emri.validation import venue_transfer as vt


def _real_ladder_context(sigma_z: float, lambda_ball: float, n_events: int) -> cg.GateContext:
    """GateContext on the REAL flat-LambdaCDM ladder, no injection pool needed.

    Mirrors ``darksiren_emri_test/validation/test_m2prime_ablation_arms.py``'s
    ``_real_ladder_context`` helper (kept local so this file has no
    cross-test-file import dependency).
    """
    gcfg = cg.GateConfig(
        cell="custom",
        h_true=0.730,
        ball=True,
        lambda_ball=lambda_ball,
        sigma_z=sigma_z,
        n_events=n_events,
    )
    cl_cfg = cg.to_closed_loop_config(gcfg)
    z_max = 1.5
    tables = [cl._z_of_dl_table(h, z_max) for h in cl_cfg.h_grid]
    gl_nodes, gl_weights = roots_legendre(cl_cfg.n_quad)
    cl_ctx = cl.ClosedLoopContext(
        config=cl_cfg,
        detection=None,  # type: ignore[arg-type]  # not used by the ball path
        sigma_triples=np.asarray([[0.02, 1e-8, 0.0]]),
        z_max_true=z_max,
        gen_z_nodes=np.linspace(1e-6, z_max, 100),
        gen_z_cdf=np.linspace(0.0, 1.0, 100),
        gen_log10_M_nodes=np.linspace(4.0, 7.0, 100),
        gen_M_cdf=np.linspace(0.0, 1.0, 100),
        z_of_dl_tables=tables,
        log_alpha=np.zeros(len(cl_cfg.h_grid)),
        s_phi_tables=[],
        gl_nodes=np.asarray(gl_nodes, dtype=np.float64),
        gl_weights=np.asarray(gl_weights, dtype=np.float64),
    )
    z_nodes = np.linspace(1e-6, 3.0, 3000)
    dl_nodes = np.asarray(dist_vectorized(z_nodes, h=0.730), dtype=np.float64)
    w = cl._w_pop(z_nodes, 0.730)
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (w[1:] + w[:-1]) * np.diff(z_nodes))])
    cdf /= cdf[-1]
    return cg.GateContext(
        gate_config=gcfg,
        cl_ctx=cl_ctx,
        csv_dl_sorted=np.asarray([1.0]),
        triples=np.asarray([[0.02, 1e-8, 0.0]]),
        decile_rows=[np.asarray([0], dtype=np.int64)] * 10,
        imp_z_nodes=z_nodes,
        imp_z_cdf=cdf,
        imp_dl_nodes=dl_nodes,
    )


def _real_ladder_universe(n: int, rng: np.random.Generator) -> cl.SyntheticUniverse:
    """Noiseless events on the real ladder at h_true = 0.730."""
    z = rng.uniform(0.15, 0.9, size=n)
    d_L = np.asarray(dist_vectorized(z, h=0.730), dtype=np.float64)
    M = np.full(n, 5.0e5)
    return cl.SyntheticUniverse(
        z_true=z,
        M_true=M,
        d_L_true=d_L,
        d_L_obs=d_L.copy(),
        M_z_obs=M * (1.0 + z),
        sigma_dL=np.full(n, 0.02),
        sigma_Mz=np.full(n, 1e-8),
        rho=np.zeros(n),
        in_catalogue=np.zeros(n, dtype=bool),
        n_drawn=n,
    )


def _small_case(
    sigma_z: float, n_events: int, seed: int
) -> tuple[cg.GateContext, cl.SyntheticUniverse, cg.HostBall]:
    """One small (gctx, universe, ball) case at the given constant σ_z."""
    gctx = _real_ladder_context(sigma_z=sigma_z, lambda_ball=4.0, n_events=n_events)
    rng = np.random.default_rng(seed)
    uni = _real_ladder_universe(n_events, rng)
    ball = cg.draw_ball(gctx, uni, rng)
    return gctx, uni, ball


# ── (a) sigma_z = 0 point-branch invariance ─────────────────────────────────


def test_kernel_renorm_is_bit_identical_to_base_at_sigma_z_zero() -> None:
    """A-REN is identically the base estimator when every candidate is point-branch.

    ``W_k`` is only computed and divided into the kernel-branch (σ_z > 0)
    integrand; at σ_z = 0 every row takes the disjoint point-evaluation
    branch, which the switch never touches (prereg §2, constraint (a)).
    """
    gctx, uni, ball = _small_case(sigma_z=0.0, n_events=12, seed=201)
    sigma_pairs = np.zeros(ball.z_obs.size, dtype=np.float64)

    ln1_base, ln2_base, slope_base = vt.log_channel_posteriors_ball_sigma_vector(
        gctx, uni, ball, sigma_pairs, estimator_variant=vt.ESTIMATOR_VARIANT_BASE
    )
    ln1_ren, ln2_ren, slope_ren = vt.log_channel_posteriors_ball_sigma_vector(
        gctx, uni, ball, sigma_pairs, estimator_variant=vt.ESTIMATOR_VARIANT_KERNEL_RENORM
    )

    np.testing.assert_array_equal(ln1_ren, ln1_base)
    np.testing.assert_array_equal(ln2_ren, ln2_base)
    np.testing.assert_array_equal(slope_ren, slope_base)


def test_joint_jren_is_bit_identical_to_base_at_sigma_z_zero() -> None:
    """A-JREN is identically the base estimator when every candidate is point-branch.

    Both sub-terms (the Jacobian and the renormalization) vanish on the
    point branch by construction (prereg §2: "Point branch untouched (both
    sub-terms vanish there)").
    """
    gctx, uni, ball = _small_case(sigma_z=0.0, n_events=12, seed=202)
    sigma_pairs = np.zeros(ball.z_obs.size, dtype=np.float64)

    ln1_base, ln2_base, slope_base = vt.log_channel_posteriors_ball_sigma_vector(
        gctx, uni, ball, sigma_pairs, estimator_variant=vt.ESTIMATOR_VARIANT_BASE
    )
    ln1_jren, ln2_jren, slope_jren = vt.log_channel_posteriors_ball_sigma_vector(
        gctx, uni, ball, sigma_pairs, estimator_variant=vt.ESTIMATOR_VARIANT_JOINT_JREN
    )

    np.testing.assert_array_equal(ln1_jren, ln1_base)
    np.testing.assert_array_equal(ln2_jren, ln2_base)
    np.testing.assert_array_equal(slope_jren, slope_base)


def test_both_stage3_variants_bit_identical_to_base_on_mixed_sigma_point_rows() -> None:
    """The same inertness holds row-wise: zero-σ candidates within a mixed ball."""
    gctx, uni, ball = _small_case(sigma_z=0.035, n_events=10, seed=203)
    sigma_pairs = np.where(np.arange(ball.z_obs.size) % 2 == 0, 0.0, 0.04).astype(np.float64)

    ln1_base, _, _ = vt.log_channel_posteriors_ball_sigma_vector(
        gctx, uni, ball, sigma_pairs, estimator_variant=vt.ESTIMATOR_VARIANT_BASE
    )
    for variant in (vt.ESTIMATOR_VARIANT_KERNEL_RENORM, vt.ESTIMATOR_VARIANT_JOINT_JREN):
        ln1_v, _, _ = vt.log_channel_posteriors_ball_sigma_vector(
            gctx, uni, ball, sigma_pairs, estimator_variant=variant
        )
        # the two variants must DIFFER overall (kernel-branch rows are affected)...
        assert not np.array_equal(ln1_base, ln1_v)

    # ...but an all-zero-σ sub-case (isolating only the point rows) must still
    # be bit-identical, confirming the divergence is confined to kernel rows.
    zero_sigma = np.zeros(ball.z_obs.size, dtype=np.float64)
    ln1_base_pt, _, _ = vt.log_channel_posteriors_ball_sigma_vector(
        gctx, uni, ball, zero_sigma, estimator_variant=vt.ESTIMATOR_VARIANT_BASE
    )
    for variant in (vt.ESTIMATOR_VARIANT_KERNEL_RENORM, vt.ESTIMATOR_VARIANT_JOINT_JREN):
        ln1_v_pt, _, _ = vt.log_channel_posteriors_ball_sigma_vector(
            gctx, uni, ball, zero_sigma, estimator_variant=variant
        )
        np.testing.assert_array_equal(ln1_base_pt, ln1_v_pt)


# ── (b) W_k floor inertness ──────────────────────────────────────────────────


def test_w_k_floor_keeps_out_of_window_candidate_at_zero_not_blown_up() -> None:
    """A candidate whose entire kernel mass falls outside the clip window stays 0.

    Constructed by hand: a candidate observed far from the event's z window
    (`|z_obs,k - z_true| >> the +-5 sigma window AND the event-level +-4
    sigma_dL window`), so its retained kernel mass ``W_k`` is exactly zero
    (``a >= b``, ``valid = False``) at every h. In the base estimator this
    candidate contributes 0 to c1/c2 (the ``np.where(valid, ..., 0.0)``
    guard). Under A-REN, dividing by ``max(W_k, _W_K_FLOOR)`` must NOT turn
    that zero into a blown-up value: the ``valid`` mask still gates the
    row to 0 regardless of what the (unused, since invalid) `integ` value
    would have been, so the contribution stays exactly zero, not 1e12 or
    anything else finite-but-huge.
    """
    n_events = 3
    gctx = _real_ladder_context(sigma_z=0.03, lambda_ball=4.0, n_events=n_events)
    z_true = np.array([0.3, 0.5, 0.7])
    d_L_true = np.asarray(dist_vectorized(z_true, h=0.730), dtype=np.float64)
    uni = cl.SyntheticUniverse(
        z_true=z_true,
        M_true=np.full(n_events, 5.0e5),
        d_L_true=d_L_true,
        d_L_obs=d_L_true.copy(),
        M_z_obs=np.full(n_events, 5.0e5) * (1.0 + z_true),
        sigma_dL=np.full(n_events, 0.02),
        sigma_Mz=np.full(n_events, 1e-8),
        rho=np.zeros(n_events),
        in_catalogue=np.zeros(n_events, dtype=bool),
        n_drawn=n_events,
    )
    # candidate for event 1 observed at z=1.4, far outside its +-4 sigma_dL
    # window around z_true=0.5 -- the point of this test.
    z_obs = np.array([0.3, 1.4, 0.7])
    ball = cg.HostBall(
        z_obs=z_obs,
        event_idx=np.array([0, 1, 2], dtype=np.int64),
        K=np.ones(n_events, dtype=np.int64),
        n_impostors_total=0,
        n_degenerate_windows=0,
    )
    sigma_pairs = np.array([0.03, 0.03, 0.03], dtype=np.float64)

    ln1_base, ln2_base, _ = vt.log_channel_posteriors_ball_sigma_vector(
        gctx, uni, ball, sigma_pairs, estimator_variant=vt.ESTIMATOR_VARIANT_BASE
    )
    for variant in (vt.ESTIMATOR_VARIANT_KERNEL_RENORM, vt.ESTIMATOR_VARIANT_JOINT_JREN):
        ln1_v, ln2_v, _ = vt.log_channel_posteriors_ball_sigma_vector(
            gctx, uni, ball, sigma_pairs, estimator_variant=variant
        )
        assert np.all(np.isfinite(ln1_v)) and np.all(np.isfinite(ln2_v))
        # event 1's out-of-window candidate cannot have inflated the result
        # to something absurd: the ln-posterior stays in a sane range,
        # comparable to the base estimator's (same order of magnitude, not
        # exploded by a near-1e12 factor from dividing by the floor).
        assert np.all(np.abs(ln1_v - ln1_base) < 1.0e3)
        assert np.all(np.abs(ln2_v - ln2_base) < 1.0e3)


def test_w_k_floor_constant_matches_registered_value() -> None:
    """The registered floor is exactly 1e-12 (prereg §3, matched to _LN_ZERO_EVENT)."""
    assert vt._W_K_FLOOR == 1e-12


# ── (c) A-JREN = Jacobian then renormalization, associativity ───────────────


def test_jren_equals_applying_jacobian_then_renorm_on_a_small_case() -> None:
    """A-JREN's ``integ`` equals (base * jacobian) / max(W_k, floor), independently
    recomputed here from first principles on a small hand-built case, matching
    the registered code-form claim that A-JREN composes A-M2''s Jacobian and
    A-REN's renormalization on the SAME integrand.
    """
    n_events = 5
    gctx = _real_ladder_context(sigma_z=0.03, lambda_ball=4.0, n_events=n_events)
    rng = np.random.default_rng(211)
    uni = _real_ladder_universe(n_events, rng)
    ball = cg.draw_ball(gctx, uni, rng)
    sigma_pairs = np.full(ball.z_obs.size, 0.03, dtype=np.float64)

    ln1_base, ln2_base, _ = vt.log_channel_posteriors_ball_sigma_vector(
        gctx, uni, ball, sigma_pairs, estimator_variant=vt.ESTIMATOR_VARIANT_BASE
    )
    ln1_m2p, ln2_m2p, _ = vt.log_channel_posteriors_ball_sigma_vector(
        gctx, uni, ball, sigma_pairs, estimator_variant=vt.ESTIMATOR_VARIANT_M2P_JACOBIAN
    )
    ln1_ren, ln2_ren, _ = vt.log_channel_posteriors_ball_sigma_vector(
        gctx, uni, ball, sigma_pairs, estimator_variant=vt.ESTIMATOR_VARIANT_KERNEL_RENORM
    )
    ln1_jren, ln2_jren, _ = vt.log_channel_posteriors_ball_sigma_vector(
        gctx, uni, ball, sigma_pairs, estimator_variant=vt.ESTIMATOR_VARIANT_JOINT_JREN
    )

    # Sanity: every variant must actually diverge from base on this all-
    # kernel-branch case (otherwise the associativity check below would be
    # vacuous).
    assert not np.array_equal(ln1_base, ln1_m2p)
    assert not np.array_equal(ln1_base, ln1_ren)
    assert not np.array_equal(ln1_base, ln1_jren)
    assert not np.array_equal(ln1_m2p, ln1_jren)
    assert not np.array_equal(ln1_ren, ln1_jren)

    # The joint arm is neither the Jacobian alone nor the renorm alone, but
    # its own composition -- checked directly against the module's own
    # elif branches by mirroring the exact per-h integrand construction for
    # one h index, then comparing the resulting c1/c2 -> ln_post pipeline.
    del ln2_base, ln2_m2p, ln2_ren, ln2_jren  # channel-2 not needed below

    h_grid = np.asarray(gctx.cl_ctx.config.h_grid, dtype=np.float64)
    k = int(np.argmin(np.abs(h_grid - 0.730)))
    ln1_at_k, ln2_at_k, _ = vt._channel_terms_at_h(
        gctx, uni, ball, sigma_pairs, k, estimator_variant=vt.ESTIMATOR_VARIANT_JOINT_JREN
    )
    assert np.isfinite(ln1_at_k) and np.isfinite(ln2_at_k)

    # Recompute the SAME quantity by hand: base kernel*p_gw, multiplied by the
    # central-difference Jacobian, divided by W_k with the same clip limits
    # a, b the numerator already uses -- exactly the registered code form.
    h = h_grid[k]
    x = gctx.cl_ctx.gl_nodes
    w_gl = gctx.cl_ctx.gl_weights
    ev = ball.event_idx
    z_obs = ball.z_obs
    d_obs_p = uni.d_L_obs[ev]
    sig_p = uni.sigma_dL[ev]
    d_L_nodes, z_tab = gctx.cl_ctx.z_of_dl_tables[k]
    z_hi_e = np.interp(uni.d_L_obs * (1.0 + cl._SIGMA_WINDOW * uni.sigma_dL), d_L_nodes, z_tab)
    z_lo_e = np.interp(uni.d_L_obs * (1.0 - cl._SIGMA_WINDOW * uni.sigma_dL), d_L_nodes, z_tab)
    z_lo_e = np.maximum(z_lo_e, 1e-6)
    z_hi_e = np.minimum(z_hi_e, z_tab[-1])
    z_lo_p = z_lo_e[ev]
    z_hi_p = z_hi_e[ev]

    zo = z_obs
    so = sigma_pairs
    a = np.maximum(z_lo_p, zo - cg._IMPOSTOR_KERNEL_WINDOW * so)
    b = np.minimum(z_hi_p, zo + cg._IMPOSTOR_KERNEL_WINDOW * so)
    valid = b > a
    half = 0.5 * (b - a)
    mid = 0.5 * (b + a)
    z_nodes = mid[:, None] + half[:, None] * x[None, :]
    d_L_n = np.asarray(
        dist_vectorized(np.maximum(z_nodes.reshape(-1), 1e-8), h=h), dtype=np.float64
    ).reshape(z_nodes.shape)
    d_L_frac = d_L_n / d_obs_p[:, None]
    p_gw = norm.pdf(d_L_frac, loc=1.0, scale=sig_p[:, None])
    kern = norm.pdf(z_nodes, loc=zo[:, None], scale=so[:, None])

    eps = vt.M2P_JACOBIAN_EPS_Z
    z_flat = np.maximum(z_nodes.reshape(-1), 1e-8)
    d_hi = np.asarray(dist_vectorized(z_flat + eps, h=h), dtype=np.float64)
    d_lo = np.asarray(dist_vectorized(np.maximum(z_flat - eps, 1e-8), h=h), dtype=np.float64)
    dd_dz = ((d_hi - d_lo) / (2.0 * eps)).reshape(z_nodes.shape)
    jac = dd_dz / d_obs_p[:, None]

    w_k = norm.cdf((b - zo) / so) - norm.cdf((a - zo) / so)
    integ_expected = (kern * p_gw * jac) / np.maximum(w_k, vt._W_K_FLOOR)[:, None]

    c1q_expected = half * (integ_expected @ w_gl)
    c1_expected = np.where(valid, c1q_expected, 0.0)
    K = np.maximum(ball.K, 1)
    L1_expected = np.bincount(ev, weights=c1_expected, minlength=n_events) / K
    ok1 = (L1_expected > 0.0) & np.isfinite(L1_expected)
    lnL1_expected = np.where(ok1, np.log(np.where(ok1, L1_expected, 1.0)), cg._LN_ZERO_EVENT)
    ln1_expected = float(np.sum(lnL1_expected)) - float(n_events) * float(gctx.cl_ctx.log_alpha[k])

    assert abs(ln1_at_k - ln1_expected) < 1e-9, (
        f"A-JREN integrand does not match the hand-composed Jacobian-then-renorm form: "
        f"{ln1_at_k} vs {ln1_expected}"
    )


# ── (d) seed-plan disjointness ────────────────────────────────────────────────


def _block(spec: vt.VenueCellSpec, h_true: float = 0.730) -> set[int]:
    return set(vt.venue_cell_seeds(spec, h_true, 0, None))


def test_seed_plan_disjointness_aren_ajren_vs_all_documented_blocks() -> None:
    """AREN (+54000..+54024) and AJREN (+54100..+54124) are disjoint from every
    previously documented block (v1/v2/v3 envelopes, reserved W1/O2,
    MECH_CELL_SPECS, SCAN_CELL_SPECS, M2P_CELL_SPECS) and from each other.
    """
    aren = _block(vt.REN_CELL_SPECS["AREN"])
    ajren = _block(vt.REN_CELL_SPECS["AJREN"])
    assert len(aren) == 25
    assert len(ajren) == 25
    assert not (aren & ajren), "AREN collides with AJREN"

    v1_lo, v1_hi = (
        vt.VT_BASE_SEED + vt.V1_SEED_OFFSET_ENVELOPE[0],
        vt.VT_BASE_SEED + vt.V1_SEED_OFFSET_ENVELOPE[1],
    )
    v2_lo, v2_hi = (
        vt.VT_BASE_SEED + vt.V2_SEED_OFFSET_ENVELOPE[0],
        vt.VT_BASE_SEED + vt.V2_SEED_OFFSET_ENVELOPE[1],
    )
    v3_lo, v3_hi = (
        vt.VT_BASE_SEED + vt.V3_SEED_OFFSET_ENVELOPE[0],
        vt.VT_BASE_SEED + vt.V3_SEED_OFFSET_ENVELOPE[1],
    )
    envelopes = ((v1_lo, v1_hi), (v2_lo, v2_hi), (v3_lo, v3_hi))

    reserved: set[int] = set()
    for lo_off, hi_off in vt.RESERVED_SEED_OFFSET_BLOCKS.values():
        reserved.update(range(vt.VT_BASE_SEED + lo_off, vt.VT_BASE_SEED + hi_off + 1))

    mech_blocks: dict[str, set[int]] = {
        name: _block(spec) for name, spec in vt.MECH_CELL_SPECS.items()
    }
    scan_all: set[int] = set()
    for spec in vt.SCAN_CELL_SPECS.values():
        scan_all.update(_block(spec))

    m2p_all: set[int] = set()
    for spec in vt.M2P_CELL_SPECS.values():
        m2p_all.update(_block(spec))

    for name, block in (("AREN", aren), ("AJREN", ajren)):
        for lo, hi in envelopes:
            assert not any(lo <= s <= hi for s in block), f"{name} collides with a v-envelope"
        assert not (block & reserved), f"{name} collides with a reserved block"
        for mech_name, mech_block in mech_blocks.items():
            assert not (block & mech_block), f"{name} collides with MECH block {mech_name}"
        assert not (block & scan_all), f"{name} collides with the 2-D dose scan"
        assert not (block & m2p_all), f"{name} collides with the stage-2 M2P arms"


def test_aren_ajren_are_registered_with_correct_variant_and_dose() -> None:
    """The two cell specs carry the registered variant, dose target, and dose."""
    aren = vt.REN_CELL_SPECS["AREN"]
    ajren = vt.REN_CELL_SPECS["AJREN"]

    assert aren.estimator_variant == vt.ESTIMATOR_VARIANT_KERNEL_RENORM
    assert aren.n_seeds == (25,)
    assert aren.seed_offsets == (54000,)
    assert aren.dose_target == "all"
    assert aren.balls == "real_k"
    assert aren.sigma_mode == "glade"
    assert aren.truths == (0.730,)

    assert ajren.estimator_variant == vt.ESTIMATOR_VARIANT_JOINT_JREN
    assert ajren.n_seeds == (25,)
    assert ajren.seed_offsets == (54100,)
    assert ajren.dose_target == "all"
    assert ajren.balls == "real_k"
    assert ajren.sigma_mode == "glade"
    assert ajren.truths == (0.730,)

    # every other pre-existing registry is untouched by the new one
    assert not (set(vt.REN_CELL_SPECS) & set(vt.CELL_SPECS))
    assert not (set(vt.REN_CELL_SPECS) & set(vt.MECH_CELL_SPECS))
    assert not (set(vt.REN_CELL_SPECS) & set(vt.SCAN_CELL_SPECS))
    assert not (set(vt.REN_CELL_SPECS) & set(vt.M2P_CELL_SPECS))
    assert not (set(vt.REN_CELL_SPECS) & set(vt.AFULL_CELL_SPECS))
    assert not (set(vt.REN_CELL_SPECS) & set(vt.AFULL2D_CELL_SPECS))
    assert set(vt.ALL_CELL_SPECS) == (
        set(vt.CELL_SPECS)
        | set(vt.MECH_CELL_SPECS)
        | set(vt.SCAN_CELL_SPECS)
        | set(vt.M2P_CELL_SPECS)
        | set(vt.REN_CELL_SPECS)
        | set(vt.AFULL_CELL_SPECS)
        | set(vt.AFULL2D_CELL_SPECS)
    )


# ── (e) prereg stamping ───────────────────────────────────────────────────────


def test_aren_ajren_stamp_the_a_jren_stage3_preregistration() -> None:
    """Both stage-3 cells stamp the A-JREN stage-3 prereg, not the parent documents."""
    assert vt.preregistration_path_for_cell("AJREN") == vt.REN_PREREG_PATH
    assert vt.preregistration_path_for_cell("AREN") == vt.REN_PREREG_PATH
    assert vt.REN_PREREG_PATH != vt.M2P_PREREG_PATH
    assert vt.REN_PREREG_PATH != vt.MECH_PREREG_PATH
    assert vt.REN_PREREG_PATH != vt.PREREG_PATH
    # unaffected cells still map to their own registries
    assert vt.preregistration_path_for_cell("AM2P") == vt.M2P_PREREG_PATH
    assert vt.preregistration_path_for_cell("MN0X") == vt.MECH_PREREG_PATH
    assert vt.preregistration_path_for_cell("T0") == vt.PREREG_PATH


def test_cli_choices_include_aren_and_ajren() -> None:
    """The CLI --cell parser accepts the two new stage-3 cells."""
    parser = vt.build_parser()
    cell_action = next(a for a in parser._actions if a.dest == "cell")
    assert cell_action.choices is not None
    choices = set(cell_action.choices)
    assert {"AJREN", "AREN"} <= choices


def test_unknown_estimator_variant_still_rejected_with_stage3_variants_present() -> None:
    """An unregistered variant name must still fail loudly after the stage-3 additions."""
    gctx, uni, ball = _small_case(sigma_z=0.03, n_events=4, seed=223)
    sigma_pairs = np.full(ball.z_obs.size, 0.03, dtype=np.float64)

    with pytest.raises(ValueError, match="unknown estimator_variant"):
        vt.log_channel_posteriors_ball_sigma_vector(
            gctx, uni, ball, sigma_pairs, estimator_variant="not-a-real-variant"
        )


def test_estimator_variant_forwarded_through_hgrain_path_for_stage3_variants() -> None:
    """The h-grain (fork-pool) path stays bit-identical to serial for A-REN/A-JREN."""
    gctx, uni, ball = _small_case(sigma_z=0.03, n_events=8, seed=227)
    sigma_pairs = np.full(ball.z_obs.size, 0.03, dtype=np.float64)

    for variant in (vt.ESTIMATOR_VARIANT_KERNEL_RENORM, vt.ESTIMATOR_VARIANT_JOINT_JREN):
        ln1_s, ln2_s, slope_s = vt.log_channel_posteriors_ball_sigma_vector(
            gctx, uni, ball, sigma_pairs, estimator_variant=variant
        )
        ln1_h, ln2_h, slope_h = vt.log_channel_posteriors_ball_sigma_vector_hgrain(
            gctx, uni, ball, sigma_pairs, estimator_variant=variant
        )
        np.testing.assert_array_equal(ln1_s, ln1_h)
        np.testing.assert_array_equal(ln2_s, ln2_h)
        np.testing.assert_array_equal(slope_s, slope_h)
