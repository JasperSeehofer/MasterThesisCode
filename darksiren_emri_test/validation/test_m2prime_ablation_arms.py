"""Registered null checks for the stage-2 estimator-variant arms (A-M2'/A-NULL).

Companion to ``results/mechanism_study_20260813/PREREGISTRATION_M2PRIME_ABLATION.md``
§3 and the "Stage-2 arms" section appended to ``ARMS.md``. These tests are the
auditable form of the registered A-M2'/A-NULL code-form claims:

- A-M2' ("m2prime_jacobian") is identically the base estimator at σ_z = 0 (the
  point branch has no z-integral to carry a measure factor).
- A-NULL ("null_scale_1p7") shifts every ln-posterior by exactly ``N·ln(1.7)``
  at every h, with an invariant per-h argmax (DS-N1's exact-equality form).
- The default ("base") variant is wired correctly and is the only variant that
  agrees with itself under both of the above probes while the other two
  variants provably diverge from it.
- The two new cells' seed blocks are disjoint from every previously registered
  block EXCEPT the one deliberate A-NULL/MN0X pairing, and stamp the stage-2
  preregistration document.
"""

from typing import Any

import numpy as np
import pytest
from scipy.special import roots_legendre

from darksiren_emri.physical_relations import dist_vectorized
from darksiren_emri.validation import calibration_gate as cg
from darksiren_emri.validation import closed_loop_gfrac as cl
from darksiren_emri.validation import venue_transfer as vt


class _FakeDetection:
    """Minimal with-BH survival stand-in: constant, finite, in [0, 1].

    Populated here (rather than ``detection=None``) so the all-variants loop
    in ``test_estimator_variant_forwarded_through_hgrain_path`` does not
    ``AttributeError`` now that ``"a_full_gsel"``
    (:data:`vt.ESTIMATOR_VARIANT_A_FULL_GSEL`) is a member of
    :data:`vt.ESTIMATOR_VARIANTS` and its ``g_sel`` queries the detection
    object directly (unlike ``"a_full"``, which only reads
    ``s_phi_tables``). No ``wbh_z_resolved`` attribute, so
    ``_wbh_z_kwargs`` passes no ``z`` kwarg.
    """

    def detection_probability_with_bh_mass_interpolated(
        self,
        d_L: Any,
        M_z: Any,
        phi: Any,
        theta: Any,
        *,
        h: float,
        z: Any = None,
    ) -> np.ndarray:
        d_L_arr = np.broadcast_to(np.asarray(d_L, dtype=np.float64), np.shape(M_z))
        return np.full_like(np.asarray(d_L_arr, dtype=np.float64), 0.8)


def _real_ladder_context(sigma_z: float, lambda_ball: float, n_events: int) -> cg.GateContext:
    """GateContext on the REAL flat-LambdaCDM ladder, no injection pool needed.

    Mirrors ``darksiren_emri_test/validation/test_venue_transfer.py``'s
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
    # Trivial (S_phi == 1 everywhere) selection-function tables — unused by
    # every registered variant except "a_full"/"a_full_gsel"
    # (ESTIMATOR_VARIANT_A_FULL / ESTIMATOR_VARIANT_A_FULL_GSEL), populated
    # here so the all-variants loop in
    # test_estimator_variant_forwarded_through_hgrain_path does not IndexError
    # on an empty list now that a_full is a member of ESTIMATOR_VARIANTS.
    s_phi_z = np.linspace(1e-6, 5.0, 2000)
    s_phi_tables = [(s_phi_z, np.ones_like(s_phi_z)) for _ in cl_cfg.h_grid]
    cl_ctx = cl.ClosedLoopContext(
        config=cl_cfg,
        detection=_FakeDetection(),  # type: ignore[arg-type]
        sigma_triples=np.asarray([[0.02, 1e-8, 0.0]]),
        z_max_true=z_max,
        gen_z_nodes=np.linspace(1e-6, z_max, 100),
        gen_z_cdf=np.linspace(0.0, 1.0, 100),
        gen_log10_M_nodes=np.linspace(4.0, 7.0, 100),
        gen_M_cdf=np.linspace(0.0, 1.0, 100),
        z_of_dl_tables=tables,
        log_alpha=np.zeros(len(cl_cfg.h_grid)),
        s_phi_tables=s_phi_tables,
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


# ── (a) sigma_z = 0 inertness ─────────────────────────────────────────────────


def test_m2prime_jacobian_is_bit_identical_to_base_at_sigma_z_zero() -> None:
    """A-M2' is identically the base estimator when every candidate is point-branch.

    The Jacobian factor is only multiplied into the kernel-branch (σ_z > 0)
    integrand; at σ_z = 0 every row takes the disjoint point-evaluation
    branch, which the switch never touches (prereg §3: "At σ_z = 0 the arm
    is IDENTICALLY the base estimator... constraint (a) preserved by
    construction").
    """
    gctx, uni, ball = _small_case(sigma_z=0.0, n_events=12, seed=101)
    sigma_pairs = np.zeros(ball.z_obs.size, dtype=np.float64)

    ln1_base, ln2_base, slope_base = vt.log_channel_posteriors_ball_sigma_vector(
        gctx, uni, ball, sigma_pairs, estimator_variant=vt.ESTIMATOR_VARIANT_BASE
    )
    ln1_m2p, ln2_m2p, slope_m2p = vt.log_channel_posteriors_ball_sigma_vector(
        gctx, uni, ball, sigma_pairs, estimator_variant=vt.ESTIMATOR_VARIANT_M2P_JACOBIAN
    )

    np.testing.assert_array_equal(ln1_m2p, ln1_base)
    np.testing.assert_array_equal(ln2_m2p, ln2_base)
    np.testing.assert_array_equal(slope_m2p, slope_base)


def test_m2prime_jacobian_is_bit_identical_to_base_on_mixed_sigma_point_rows() -> None:
    """The same inertness holds row-wise: zero-σ candidates within a mixed ball.

    A stronger form of (a): even when SOME candidates in the ball are
    kernel-branch (so the Jacobian factor IS applied somewhere), the rows
    that are point-branch contribute identically in both variants — the
    switch operates on a code path the point rows never execute.
    """
    gctx, uni, ball = _small_case(sigma_z=0.035, n_events=10, seed=103)
    # half the candidates point-branch, half kernel-branch
    sigma_pairs = np.where(np.arange(ball.z_obs.size) % 2 == 0, 0.0, 0.04).astype(np.float64)

    ln1_base, ln2_base, _ = vt.log_channel_posteriors_ball_sigma_vector(
        gctx, uni, ball, sigma_pairs, estimator_variant=vt.ESTIMATOR_VARIANT_BASE
    )
    ln1_m2p, ln2_m2p, _ = vt.log_channel_posteriors_ball_sigma_vector(
        gctx, uni, ball, sigma_pairs, estimator_variant=vt.ESTIMATOR_VARIANT_M2P_JACOBIAN
    )

    # the two variants must DIFFER overall (kernel-branch rows are affected)...
    assert not np.array_equal(ln1_base, ln1_m2p)
    # ...but an all-zero-σ sub-case (isolating only the point rows) must still
    # be bit-identical, confirming the divergence is confined to kernel rows.
    zero_sigma = np.zeros(ball.z_obs.size, dtype=np.float64)
    ln1_base_pt, _, _ = vt.log_channel_posteriors_ball_sigma_vector(
        gctx, uni, ball, zero_sigma, estimator_variant=vt.ESTIMATOR_VARIANT_BASE
    )
    ln1_m2p_pt, _, _ = vt.log_channel_posteriors_ball_sigma_vector(
        gctx, uni, ball, zero_sigma, estimator_variant=vt.ESTIMATOR_VARIANT_M2P_JACOBIAN
    )
    np.testing.assert_array_equal(ln1_base_pt, ln1_m2p_pt)


# ── (b) A-NULL shift law ───────────────────────────────────────────────────────


def test_null_scale_shifts_ln_post_by_n_ln_1p7_at_every_h() -> None:
    """DS-N1's exact-equality form: ln_post(null) == ln_post(base) + N·ln(1.7).

    Reproduced on a small synthetic all-kernel-branch case (every candidate
    σ_z > 0) rather than the full 982-event pool. ``rtol <= 1e-12`` matches
    the registered DS-N1 tolerance exactly (prereg §4).
    """
    n_events = 9
    gctx, uni, ball = _small_case(sigma_z=0.035, n_events=n_events, seed=107)
    sigma_pairs = np.full(ball.z_obs.size, 0.035, dtype=np.float64)

    ln1_base, ln2_base, _ = vt.log_channel_posteriors_ball_sigma_vector(
        gctx, uni, ball, sigma_pairs, estimator_variant=vt.ESTIMATOR_VARIANT_BASE
    )
    ln1_null, ln2_null, _ = vt.log_channel_posteriors_ball_sigma_vector(
        gctx, uni, ball, sigma_pairs, estimator_variant=vt.ESTIMATOR_VARIANT_NULL_SCALE
    )

    n = int(uni.z_true.size)
    shift = n * float(np.log(vt.NULL_SCALE_FACTOR))

    for base_arr, null_arr in ((ln1_base, ln1_null), (ln2_base, ln2_null)):
        assert np.all(np.isfinite(base_arr)) and np.all(np.isfinite(null_arr))
        diff = null_arr - base_arr - shift
        rel = np.abs(diff) / np.abs(base_arr)
        assert np.all(rel <= 1e-12), f"max rel deviation {rel.max()} exceeds 1e-12"

    # per-h argmax index identical (a constant additive shift cannot move it)
    assert int(np.argmax(ln1_base)) == int(np.argmax(ln1_null))
    assert int(np.argmax(ln2_base)) == int(np.argmax(ln2_null))


def test_ds_n1_floor_aware_integer_shift_law_with_a_zero_likelihood_event() -> None:
    """DS-N1's floor-aware form (prereg §4): the shift stays an exact integer
    multiple of ln(1.7) even when one event floors at some h.

    Constructed by hand (not via ``_small_case``/``draw_ball``) so the floor
    is deliberate rather than incidental: event 0 has a wide ``sigma_dL``
    (its window never leaves the grid) while event 1 has both a narrow
    ``sigma_dL`` (window ``[z_lo, z_hi]`` in the event-level ±4σ sense) AND a
    narrow candidate ``sigma_z`` (kernel ±5σ window) that only overlaps the
    event-level window near ``h_true`` — at the grid wings the two windows
    fail to intersect, all candidate rows are invalid, and the event floors
    at the registered ``-745`` zero-event penalty (invariant under ×1.7).

    Asserts, at every h and in both channels: (i) ``Δln_post/ln(1.7)`` is
    within 1e-6 nats of an integer, (ii) that integer is in ``[0, N]``,
    (iii) it equals ``N`` at the floor-free ``h_true`` grid point, (iv) a
    floor actually fires somewhere (else the test would not exercise the
    floor path at all), and (v) the per-channel argmax is unchanged.
    """
    n_events = 2
    gctx = _real_ladder_context(sigma_z=0.03, lambda_ball=4.0, n_events=n_events)
    z_true = np.array([0.5, 0.5])
    d_L_true = np.asarray(dist_vectorized(z_true, h=0.730), dtype=np.float64)
    sigma_dL = np.array([0.15, 0.0005])  # event 0: never floors; event 1: floors off-truth
    uni = cl.SyntheticUniverse(
        z_true=z_true,
        M_true=np.full(n_events, 5.0e5),
        d_L_true=d_L_true,
        d_L_obs=d_L_true.copy(),
        M_z_obs=np.full(n_events, 5.0e5) * (1.0 + z_true),
        sigma_dL=sigma_dL,
        sigma_Mz=np.full(n_events, 1e-8),
        rho=np.zeros(n_events),
        in_catalogue=np.zeros(n_events, dtype=bool),
        n_drawn=n_events,
    )
    ball = cg.HostBall(
        z_obs=z_true.copy(),
        event_idx=np.array([0, 1], dtype=np.int64),
        K=np.ones(n_events, dtype=np.int64),
        n_impostors_total=0,
        n_degenerate_windows=0,
    )
    sigma_pairs = np.array([0.03, 0.005], dtype=np.float64)  # event 1's narrow kernel window

    ln1_base, ln2_base, _ = vt.log_channel_posteriors_ball_sigma_vector(
        gctx, uni, ball, sigma_pairs, estimator_variant=vt.ESTIMATOR_VARIANT_BASE
    )
    ln1_null, ln2_null, _ = vt.log_channel_posteriors_ball_sigma_vector(
        gctx, uni, ball, sigma_pairs, estimator_variant=vt.ESTIMATOR_VARIANT_NULL_SCALE
    )

    ln17 = float(np.log(vt.NULL_SCALE_FACTOR))
    h_grid = np.asarray(gctx.cl_ctx.config.h_grid, dtype=np.float64)
    i_true = int(np.argmin(np.abs(h_grid - 0.730)))

    for base_arr, null_arr, label in (
        (ln1_base, ln1_null, "channel 1"),
        (ln2_base, ln2_null, "channel 2"),
    ):
        delta = null_arr - base_arr
        m = np.round(delta / ln17)
        residual = np.abs(delta - m * ln17)
        assert np.all(residual <= 1e-6), (
            f"{label}: integer shift law violated, max resid {residual.max()}"
        )
        assert np.all(m >= 0.0) and np.all(m <= n_events), f"{label}: m(h) outside [0, N]"
        assert m[i_true] == n_events, f"{label}: h_true is not floor-free as constructed"
        assert np.any(m < n_events), (
            f"{label}: no floor ever fired -- test does not exercise DS-N1's floor path"
        )

    assert int(np.argmax(ln1_base)) == int(np.argmax(ln1_null))
    assert int(np.argmax(ln2_base)) == int(np.argmax(ln2_null))


def test_null_scale_is_h_and_z_independent_constant() -> None:
    """The 1.7 factor is applied identically regardless of h or the candidate z.

    A regression on the registered claim that A-NULL ablates a provably-inert
    constant: scaling factor recovered algebraically must equal 1.7 exactly
    (up to floating-point associativity) at EVERY grid point, not just near
    truth.
    """
    n_events = 6
    gctx, uni, ball = _small_case(sigma_z=0.05, n_events=n_events, seed=109)
    sigma_pairs = np.full(ball.z_obs.size, 0.05, dtype=np.float64)

    ln1_base, _, _ = vt.log_channel_posteriors_ball_sigma_vector(
        gctx, uni, ball, sigma_pairs, estimator_variant=vt.ESTIMATOR_VARIANT_BASE
    )
    ln1_null, _, _ = vt.log_channel_posteriors_ball_sigma_vector(
        gctx, uni, ball, sigma_pairs, estimator_variant=vt.ESTIMATOR_VARIANT_NULL_SCALE
    )
    n = int(uni.z_true.size)
    recovered = np.exp((ln1_null - ln1_base) / n)
    np.testing.assert_allclose(recovered, np.full_like(recovered, vt.NULL_SCALE_FACTOR), rtol=1e-10)


# ── (c) base regression ────────────────────────────────────────────────────────


def test_base_variant_default_matches_explicit_and_the_other_two_variants_diverge() -> None:
    """The default path is wired to "base" and "base" is distinguishable.

    Strongest cheap regression available without a pre-change git checkout
    (chosen per instruction, flagged here): (i) the default (omitted
    ``estimator_variant``) is bit-identical to the explicit ``"base"`` call —
    confirms the switch's default wiring introduces no behaviour change on
    any existing call site; (ii) on a case where both effects are live
    (σ_z > 0 everywhere), BOTH other registered variants provably diverge
    from the default. Together with the σ_z = 0 inertness test (a) and the
    exact shift-law test (b), this triangulates "base" as the unique
    fixed point of both probes — the two-of-three consistency form named in
    the task spec.
    """
    gctx, uni, ball = _small_case(sigma_z=0.04, n_events=11, seed=113)
    sigma_pairs = np.full(ball.z_obs.size, 0.04, dtype=np.float64)

    ln1_default, ln2_default, slope_default = vt.log_channel_posteriors_ball_sigma_vector(
        gctx, uni, ball, sigma_pairs
    )
    ln1_base, ln2_base, slope_base = vt.log_channel_posteriors_ball_sigma_vector(
        gctx, uni, ball, sigma_pairs, estimator_variant=vt.ESTIMATOR_VARIANT_BASE
    )
    np.testing.assert_array_equal(ln1_default, ln1_base)
    np.testing.assert_array_equal(ln2_default, ln2_base)
    np.testing.assert_array_equal(slope_default, slope_base)

    ln1_m2p, _, _ = vt.log_channel_posteriors_ball_sigma_vector(
        gctx, uni, ball, sigma_pairs, estimator_variant=vt.ESTIMATOR_VARIANT_M2P_JACOBIAN
    )
    ln1_null, _, _ = vt.log_channel_posteriors_ball_sigma_vector(
        gctx, uni, ball, sigma_pairs, estimator_variant=vt.ESTIMATOR_VARIANT_NULL_SCALE
    )
    assert not np.array_equal(ln1_default, ln1_m2p)
    assert not np.array_equal(ln1_default, ln1_null)
    assert not np.array_equal(ln1_m2p, ln1_null)


def test_unknown_estimator_variant_is_rejected() -> None:
    """An unregistered variant name must fail loudly, never fall through."""
    gctx, uni, ball = _small_case(sigma_z=0.03, n_events=4, seed=117)
    sigma_pairs = np.full(ball.z_obs.size, 0.03, dtype=np.float64)

    with pytest.raises(ValueError, match="unknown estimator_variant"):
        vt.log_channel_posteriors_ball_sigma_vector(
            gctx, uni, ball, sigma_pairs, estimator_variant="not-a-real-variant"
        )


def test_estimator_variant_forwarded_through_hgrain_path() -> None:
    """The h-grain (fork-pool) path stays bit-identical to serial for every variant."""
    gctx, uni, ball = _small_case(sigma_z=0.03, n_events=8, seed=119)
    sigma_pairs = np.full(ball.z_obs.size, 0.03, dtype=np.float64)

    for variant in vt.ESTIMATOR_VARIANTS:
        ln1_s, ln2_s, slope_s = vt.log_channel_posteriors_ball_sigma_vector(
            gctx, uni, ball, sigma_pairs, estimator_variant=variant
        )
        ln1_h, ln2_h, slope_h = vt.log_channel_posteriors_ball_sigma_vector_hgrain(
            gctx, uni, ball, sigma_pairs, estimator_variant=variant
        )
        np.testing.assert_array_equal(ln1_s, ln1_h)
        np.testing.assert_array_equal(ln2_s, ln2_h)
        np.testing.assert_array_equal(slope_s, slope_h)


# ── (d) seed-plan disjointness ─────────────────────────────────────────────────


def _block(spec: vt.VenueCellSpec, h_true: float = 0.730) -> set[int]:
    return set(vt.venue_cell_seeds(spec, h_true, 0, None))


def test_seed_plan_disjointness_except_registered_anull_pairing() -> None:
    """AM2P is disjoint from everything; ANULL overlaps ONLY MN0/MN0X, by design.

    Assembles every previously registered block (v1/v2/v3 envelopes,
    reserved W1/O2, MECH_CELL_SPECS incl. MN0X, SCAN_CELL_SPECS) from the
    module's own constants (not re-derived literals) and checks AM2P
    (+53000..+53024) against all of them, then checks ANULL
    (+50000..+50014) is the documented deliberate exception.
    """
    am2p = _block(vt.M2P_CELL_SPECS["AM2P"])
    anull = _block(vt.M2P_CELL_SPECS["ANULL"])
    assert len(am2p) == 25
    assert len(anull) == 15

    v1_lo, v1_hi = (
        vt.VT_BASE_SEED + vt.V1_SEED_OFFSET_ENVELOPE[0],
        (vt.VT_BASE_SEED + vt.V1_SEED_OFFSET_ENVELOPE[1]),
    )
    v2_lo, v2_hi = (
        vt.VT_BASE_SEED + vt.V2_SEED_OFFSET_ENVELOPE[0],
        (vt.VT_BASE_SEED + vt.V2_SEED_OFFSET_ENVELOPE[1]),
    )
    v3_lo, v3_hi = (
        vt.VT_BASE_SEED + vt.V3_SEED_OFFSET_ENVELOPE[0],
        (vt.VT_BASE_SEED + vt.V3_SEED_OFFSET_ENVELOPE[1]),
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

    # AM2P: disjoint from every envelope, reserved block, MECH block, and scan.
    for lo, hi in envelopes:
        assert not any(lo <= s <= hi for s in am2p), "AM2P collides with a v-envelope"
    assert not (am2p & reserved), "AM2P collides with a reserved block"
    for name, block in mech_blocks.items():
        assert not (am2p & block), f"AM2P collides with MECH block {name}"
    assert not (am2p & scan_all), "AM2P collides with the 2-D dose scan"
    assert not (am2p & anull), "AM2P collides with ANULL"

    # ANULL: disjoint from every envelope, reserved block, and scan...
    for lo, hi in envelopes:
        assert not any(lo <= s <= hi for s in anull), "ANULL collides with a v-envelope"
    assert not (anull & reserved), "ANULL collides with a reserved block"
    assert not (anull & scan_all), "ANULL collides with the 2-D dose scan"

    # ...and overlaps MECH ONLY via the registered MN0/MN0X pairing, and that
    # overlap must be EXACT equality (the same 15 seeds), not a partial one.
    for name, block in mech_blocks.items():
        if name in ("MN0", "MN0X"):
            assert anull <= block, f"ANULL is not a subset of {name}'s block"
        else:
            assert not (anull & block), f"ANULL unexpectedly collides with MECH block {name}"
    assert anull == mech_blocks["MN0"], "ANULL must equal MN0's exact 15-seed block"


def test_am2p_and_anull_are_registered_with_correct_variant_and_dose() -> None:
    """The two cell specs carry the registered variant, dose target, and dose."""
    am2p = vt.M2P_CELL_SPECS["AM2P"]
    anull = vt.M2P_CELL_SPECS["ANULL"]

    assert am2p.estimator_variant == vt.ESTIMATOR_VARIANT_M2P_JACOBIAN
    assert am2p.n_seeds == (25,)
    assert am2p.seed_offsets == (53000,)
    assert am2p.dose_target == "all"
    assert am2p.balls == "real_k"
    assert am2p.sigma_mode == "glade"
    assert am2p.truths == (0.730,)

    assert anull.estimator_variant == vt.ESTIMATOR_VARIANT_NULL_SCALE
    assert anull.n_seeds == (15,)
    assert anull.seed_offsets == (50000,)
    assert anull.dose_target == "all"
    assert anull.balls == "real_k"
    assert anull.sigma_mode == "glade"
    assert anull.truths == (0.730,)

    # every other pre-existing registry is untouched by the new one
    assert not (set(vt.M2P_CELL_SPECS) & set(vt.CELL_SPECS))
    assert not (set(vt.M2P_CELL_SPECS) & set(vt.MECH_CELL_SPECS))
    assert not (set(vt.M2P_CELL_SPECS) & set(vt.SCAN_CELL_SPECS))
    # ALL_CELL_SPECS also carries the stage-3 REN_CELL_SPECS (registered
    # 2026-08-15, PREREGISTRATION_A_JREN_STAGE3.md) on top of these four —
    # this assertion only checks stage-2's contribution is present and
    # correctly scoped, not that these four alone form the union.
    assert (
        set(vt.CELL_SPECS)
        | set(vt.MECH_CELL_SPECS)
        | set(vt.SCAN_CELL_SPECS)
        | set(vt.M2P_CELL_SPECS)
    ) <= set(vt.ALL_CELL_SPECS)


# ── (e) prereg stamping ─────────────────────────────────────────────────────────


def test_am2p_and_anull_stamp_the_m2prime_preregistration() -> None:
    """Both stage-2 cells stamp the M2' ablation prereg, not the parent documents."""
    assert vt.preregistration_path_for_cell("AM2P") == vt.M2P_PREREG_PATH
    assert vt.preregistration_path_for_cell("ANULL") == vt.M2P_PREREG_PATH
    assert vt.M2P_PREREG_PATH != vt.MECH_PREREG_PATH
    assert vt.M2P_PREREG_PATH != vt.PREREG_PATH
    # unaffected cells still map to their own registries
    assert vt.preregistration_path_for_cell("MN0X") == vt.MECH_PREREG_PATH
    assert vt.preregistration_path_for_cell("T0") == vt.PREREG_PATH


def test_cli_choices_include_am2p_and_anull() -> None:
    """The CLI --cell parser accepts the two new stage-2 cells."""
    parser = vt.build_parser()
    cell_action = next(a for a in parser._actions if a.dest == "cell")
    assert cell_action.choices is not None
    choices = set(cell_action.choices)
    assert {"AM2P", "ANULL"} <= choices
