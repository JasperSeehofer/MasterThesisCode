"""Registered checks for the A-FULL (FULL-F) estimator variant.

Companion to ``results/mechanism_study_20260813/DRAFT_A_FULL_ESTIMATOR_20260815.md``
§1 + §3 + addendum A1 (ledger row #110). These tests are the auditable form of the
installed A-FULL code-form claims:

- ``"a_full"`` (:data:`venue_transfer.ESTIMATOR_VARIANT_A_FULL`) is registered in
  :data:`venue_transfer.ESTIMATOR_VARIANTS` and is rejected/accepted by the same
  validation as every other variant (the ``ValueError`` fallthrough branch).
- Unlike A-M2'/A-NULL/A-REN/A-JREN, A-FULL is NOT inert on the point branch
  (σ_z = 0): the d_obs-density GW factor, selected-population prior, and LOO
  weight all apply there too (draft §3: "The point branch ... takes the same
  factors evaluated at z_obs").
- :func:`venue_transfer._loo_impostor_weights` returns finite, strictly
  positive values bounded above by the registered floor's reciprocal (1000,
  since ``imp_k`` is floored at ``1e-3``).
- The h-grain (fork-pool) path stays bit-identical to serial for A-FULL.
- Every other registered variant's byte-identity to its pre-switch behaviour
  is untouched by this addition (probed indirectly: base's own regression
  tests live in test_m2prime_ablation_arms.py / test_a_jren_stage3_arms.py
  and are not touched by this file).
"""

import numpy as np
import pytest
from scipy.special import roots_legendre

from darksiren_emri.physical_relations import dist_vectorized
from darksiren_emri.validation import calibration_gate as cg
from darksiren_emri.validation import closed_loop_gfrac as cl
from darksiren_emri.validation import venue_transfer as vt


def _real_ladder_context(sigma_z: float, lambda_ball: float, n_events: int) -> cg.GateContext:
    """GateContext on the REAL flat-LambdaCDM ladder, no injection pool needed.

    Mirrors ``test_m2prime_ablation_arms.py``'s ``_real_ladder_context`` helper
    (kept local so this file has no cross-test-file import dependency), with a
    non-trivial (but physically arbitrary — S_phi == 0.7 everywhere) selection
    table populated, since A-FULL is the first variant that reads
    ``gctx.cl_ctx.s_phi_tables``.
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
    s_phi_z = np.linspace(1e-6, 5.0, 2000)
    s_phi_tables = [(s_phi_z, np.full_like(s_phi_z, 0.7)) for _ in cl_cfg.h_grid]
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


# ── (a) registration ─────────────────────────────────────────────────────────


def test_a_full_is_registered() -> None:
    """The variant constant and its string value are registered."""
    assert vt.ESTIMATOR_VARIANT_A_FULL == "a_full"
    assert vt.ESTIMATOR_VARIANT_A_FULL in vt.ESTIMATOR_VARIANTS


def test_unknown_estimator_variant_still_rejected_with_a_full_present() -> None:
    """An unregistered variant name must still fail loudly after this addition."""
    gctx, uni, ball = _small_case(sigma_z=0.03, n_events=4, seed=301)
    sigma_pairs = np.full(ball.z_obs.size, 0.03, dtype=np.float64)

    with pytest.raises(ValueError, match="unknown estimator_variant"):
        vt.log_channel_posteriors_ball_sigma_vector(
            gctx, uni, ball, sigma_pairs, estimator_variant="not-a-real-variant"
        )


# ── (b) numerical behaviour ───────────────────────────────────────────────────


def test_a_full_runs_finite_and_differs_from_base_on_kernel_branch() -> None:
    """A-FULL produces finite output that differs from the base estimator."""
    gctx, uni, ball = _small_case(sigma_z=0.035, n_events=10, seed=303)
    sigma_pairs = np.full(ball.z_obs.size, 0.035, dtype=np.float64)

    ln1_base, ln2_base, slope_base = vt.log_channel_posteriors_ball_sigma_vector(
        gctx, uni, ball, sigma_pairs, estimator_variant=vt.ESTIMATOR_VARIANT_BASE
    )
    ln1_full, ln2_full, slope_full = vt.log_channel_posteriors_ball_sigma_vector(
        gctx, uni, ball, sigma_pairs, estimator_variant=vt.ESTIMATOR_VARIANT_A_FULL
    )

    assert np.all(np.isfinite(ln1_full))
    assert np.all(np.isfinite(ln2_full))
    assert np.all(np.isfinite(slope_full))
    assert not np.array_equal(ln1_base, ln1_full)
    assert not np.array_equal(ln2_base, ln2_full)


def test_a_full_differs_from_base_on_point_branch_too() -> None:
    """Unlike every other variant, A-FULL is NOT inert at σ_z = 0 (draft §3:
    the point branch takes the same three factors evaluated at z_obs).
    """
    gctx, uni, ball = _small_case(sigma_z=0.0, n_events=12, seed=305)
    sigma_pairs = np.zeros(ball.z_obs.size, dtype=np.float64)

    ln1_base, ln2_base, _ = vt.log_channel_posteriors_ball_sigma_vector(
        gctx, uni, ball, sigma_pairs, estimator_variant=vt.ESTIMATOR_VARIANT_BASE
    )
    ln1_full, ln2_full, _ = vt.log_channel_posteriors_ball_sigma_vector(
        gctx, uni, ball, sigma_pairs, estimator_variant=vt.ESTIMATOR_VARIANT_A_FULL
    )

    assert np.all(np.isfinite(ln1_full))
    assert np.all(np.isfinite(ln2_full))
    assert not np.array_equal(ln1_base, ln1_full)
    assert not np.array_equal(ln2_base, ln2_full)


def test_a_full_runs_finite_on_mixed_sigma_rows() -> None:
    """A ball with both kernel- and point-branch candidates stays finite."""
    gctx, uni, ball = _small_case(sigma_z=0.035, n_events=10, seed=307)
    sigma_pairs = np.where(np.arange(ball.z_obs.size) % 2 == 0, 0.0, 0.04).astype(np.float64)

    ln1_full, ln2_full, slope_full = vt.log_channel_posteriors_ball_sigma_vector(
        gctx, uni, ball, sigma_pairs, estimator_variant=vt.ESTIMATOR_VARIANT_A_FULL
    )
    assert np.all(np.isfinite(ln1_full))
    assert np.all(np.isfinite(ln2_full))
    assert np.all(np.isfinite(slope_full))


# ── (c) LOO impostor weights ─────────────────────────────────────────────────


def test_loo_impostor_weights_positive_finite_and_floor_bounded() -> None:
    """1/imp_k is strictly positive, finite, and bounded above by 1/floor (1000)
    per the registered floor (imp_k >= 1e-3, matching the reference
    implementation ``loo_weights`` in l4_afull_premeasure.py).
    """
    gctx, uni, ball = _small_case(sigma_z=0.035, n_events=15, seed=309)
    sigma_pairs = np.full(ball.z_obs.size, 0.035, dtype=np.float64)

    loo_w = vt._loo_impostor_weights(gctx, uni, ball, sigma_pairs)

    assert loo_w.shape == ball.z_obs.shape
    assert np.all(np.isfinite(loo_w))
    assert np.all(loo_w > 0.0)
    assert np.all(loo_w <= 1000.0 + 1e-9)


def test_loo_impostor_weights_h_independent_signature() -> None:
    """The helper takes no h argument — its return depends only on the seed
    realization (gctx/universe/ball/sig_z), never on k or h (draft addendum
    A1: "h-independent, per-seed").
    """
    import inspect

    sig = inspect.signature(vt._loo_impostor_weights)
    assert list(sig.parameters) == ["gctx", "universe", "ball", "sig_z"]


# ── (d) h-grain forwarding ────────────────────────────────────────────────────


def test_a_full_forwarded_through_hgrain_path() -> None:
    """The h-grain (fork-pool) path stays bit-identical to serial for A-FULL."""
    gctx, uni, ball = _small_case(sigma_z=0.03, n_events=8, seed=311)
    sigma_pairs = np.full(ball.z_obs.size, 0.03, dtype=np.float64)

    ln1_s, ln2_s, slope_s = vt.log_channel_posteriors_ball_sigma_vector(
        gctx, uni, ball, sigma_pairs, estimator_variant=vt.ESTIMATOR_VARIANT_A_FULL
    )
    ln1_h, ln2_h, slope_h = vt.log_channel_posteriors_ball_sigma_vector_hgrain(
        gctx, uni, ball, sigma_pairs, estimator_variant=vt.ESTIMATOR_VARIANT_A_FULL
    )
    np.testing.assert_array_equal(ln1_s, ln1_h)
    np.testing.assert_array_equal(ln2_s, ln2_h)
    np.testing.assert_array_equal(slope_s, slope_h)


# ── (e) AFULL cell registration (ledger row #110, stage-5) ───────────────────
# Companion to results/mechanism_study_20260813/PREREGISTRATION_A_FULL_STAGE5.md
# §2. Mirrors test_a_jren_stage3_arms.py's registration/disjointness tests for
# AREN/AJREN, applied to the single AFULL cell (+54200..+54224).


def _block(spec: vt.VenueCellSpec, h_true: float = 0.730) -> set[int]:
    return set(vt.venue_cell_seeds(spec, h_true, 0, None))


def test_afull_is_registered_with_correct_variant_and_dose() -> None:
    """The AFULL cell spec carries the registered variant, dose target, and dose."""
    afull = vt.AFULL_CELL_SPECS["AFULL"]

    assert afull.estimator_variant == vt.ESTIMATOR_VARIANT_A_FULL
    assert afull.n_seeds == (25,)
    assert afull.seed_offsets == (54200,)
    assert afull.dose_target == "all"
    assert afull.balls == "real_k"
    assert afull.sigma_mode == "glade"
    assert afull.truths == (0.730,)
    assert afull.prereg_cell == "A-FULL"

    # every other pre-existing registry is untouched by the new one
    assert not (set(vt.AFULL_CELL_SPECS) & set(vt.CELL_SPECS))
    assert not (set(vt.AFULL_CELL_SPECS) & set(vt.MECH_CELL_SPECS))
    assert not (set(vt.AFULL_CELL_SPECS) & set(vt.SCAN_CELL_SPECS))
    assert not (set(vt.AFULL_CELL_SPECS) & set(vt.M2P_CELL_SPECS))
    assert not (set(vt.AFULL_CELL_SPECS) & set(vt.REN_CELL_SPECS))
    assert set(vt.ALL_CELL_SPECS) == (
        set(vt.CELL_SPECS)
        | set(vt.MECH_CELL_SPECS)
        | set(vt.SCAN_CELL_SPECS)
        | set(vt.M2P_CELL_SPECS)
        | set(vt.REN_CELL_SPECS)
        | set(vt.AFULL_CELL_SPECS)
    )


def test_seed_plan_disjointness_afull_vs_all_documented_blocks() -> None:
    """AFULL (+54200..+54224) is disjoint from every previously documented
    block (v1/v2/v3 envelopes, reserved W1/O2, MECH_CELL_SPECS,
    SCAN_CELL_SPECS, M2P_CELL_SPECS, REN_CELL_SPECS).
    """
    afull = _block(vt.AFULL_CELL_SPECS["AFULL"])
    assert len(afull) == 25
    assert afull == set(range(20260808 + 54200, 20260808 + 54225))

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
    for lo, hi in envelopes:
        assert not any(lo <= s <= hi for s in afull), "AFULL collides with a v-envelope"

    reserved: set[int] = set()
    for lo_off, hi_off in vt.RESERVED_SEED_OFFSET_BLOCKS.values():
        reserved.update(range(vt.VT_BASE_SEED + lo_off, vt.VT_BASE_SEED + hi_off + 1))
    assert not (afull & reserved), "AFULL collides with a reserved block"

    mech_blocks: dict[str, set[int]] = {
        name: _block(spec) for name, spec in vt.MECH_CELL_SPECS.items()
    }
    for mech_name, mech_block in mech_blocks.items():
        assert not (afull & mech_block), f"AFULL collides with MECH block {mech_name}"

    scan_all: set[int] = set()
    for spec in vt.SCAN_CELL_SPECS.values():
        scan_all.update(_block(spec))
    assert not (afull & scan_all), "AFULL collides with the 2-D dose scan"

    m2p_all: set[int] = set()
    for spec in vt.M2P_CELL_SPECS.values():
        m2p_all.update(_block(spec))
    assert not (afull & m2p_all), "AFULL collides with the stage-2 M2P arms"

    ren_all: set[int] = set()
    for spec in vt.REN_CELL_SPECS.values():
        ren_all.update(_block(spec))
    assert not (afull & ren_all), "AFULL collides with the stage-3 REN/JREN arms"


def test_afull_stamps_the_a_full_stage5_preregistration() -> None:
    """AFULL maps to its own stage-5 prereg, not any parent document."""
    assert vt.preregistration_path_for_cell("AFULL") == vt.AFULL_PREREG_PATH
    assert vt.AFULL_PREREG_PATH != vt.REN_PREREG_PATH
    assert vt.AFULL_PREREG_PATH != vt.M2P_PREREG_PATH
    assert vt.AFULL_PREREG_PATH != vt.MECH_PREREG_PATH
    assert vt.AFULL_PREREG_PATH != vt.PREREG_PATH
    # unaffected cells still map to their own registries
    assert vt.preregistration_path_for_cell("AJREN") == vt.REN_PREREG_PATH
    assert vt.preregistration_path_for_cell("AM2P") == vt.M2P_PREREG_PATH
    assert vt.preregistration_path_for_cell("MN0X") == vt.MECH_PREREG_PATH
    assert vt.preregistration_path_for_cell("T0") == vt.PREREG_PATH


def test_cli_choices_include_afull() -> None:
    """The CLI --cell parser accepts the new AFULL cell."""
    parser = vt.build_parser()
    cell_action = next(a for a in parser._actions if a.dest == "cell")
    assert cell_action.choices is not None
    assert "AFULL" in set(cell_action.choices)
