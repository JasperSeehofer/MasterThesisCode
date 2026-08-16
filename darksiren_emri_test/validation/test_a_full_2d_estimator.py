"""Registered checks for the A-FULL-2D (fused ``g_sel``) estimator variant.

Companion to ``results/mechanism_study_20260813/PREREGISTRATION_A_FULL_2D.md``
§2 + §3 (ledger row #115 item 2). These tests are the auditable form of the
installed A-FULL-2D code-form claims, mirroring
``test_a_full_estimator.py``'s structure for the A-FULL variant it extends:

- ``"a_full_gsel"`` (:data:`venue_transfer.ESTIMATOR_VARIANT_A_FULL_GSEL`) is
  registered in :data:`venue_transfer.ESTIMATOR_VARIANTS`.
- Its 1D channel (``ln_post_1d``) is bit-identical to ``"a_full"``'s on the
  SAME realization (prereg §3, DS-G4) — on both the kernel branch and the
  point branch (σ_z = 0).
- Its 2D channel (``ln_post_2d``) is finite and differs from both ``"base"``
  and ``"a_full"`` (the fused ``g_sel`` replaces the coded ``g`` and drops the
  ``S_bar_phi`` node-weight factor).
- The h-grain (fork-pool) path stays bit-identical to serial.
- The AFULL2D cell (+54300..+54324) is registered, disjoint from every
  previously documented block including AFULL_CELL_SPECS (+54200..+54224).
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

    No ``wbh_z_resolved`` attribute (``getattr(..., False)`` default), so
    :func:`~darksiren_emri.bayesian_inference.bayesian_statistics._wbh_z_kwargs`
    passes no ``z`` kwarg — matching the venue's real
    ``ClosedLoopContext.detection`` object, per the module docstring of
    ``l6_der2_gsel_premeasure.py`` ("verified here: it does not [have FIX-3
    active]").
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

    Local copy of ``test_a_full_estimator.py``'s helper (kept local so this
    file has no cross-test-file import dependency), with a non-mock
    ``detection`` object since A-FULL-2D's ``g_sel`` queries it (unlike
    A-FULL, which only reads ``s_phi_tables``).
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


# ── (a) registration ─────────────────────────────────────────────────────────


def test_a_full_gsel_is_registered() -> None:
    """The variant constant and its string value are registered."""
    assert vt.ESTIMATOR_VARIANT_A_FULL_GSEL == "a_full_gsel"
    assert vt.ESTIMATOR_VARIANT_A_FULL_GSEL in vt.ESTIMATOR_VARIANTS


# ── (b) numerical behaviour: 1D channel bit-identity to a_full (DS-G4) ──────


def test_a_full_gsel_1d_channel_bit_identical_to_a_full_kernel_branch() -> None:
    """A-FULL-2D's ln1 must be bit-identical to A-FULL's on the same
    realization (prereg §3: '1D channel byte-identical to a_full')."""
    gctx, uni, ball = _small_case(sigma_z=0.035, n_events=10, seed=403)
    sigma_pairs = np.full(ball.z_obs.size, 0.035, dtype=np.float64)

    ln1_full, ln2_full, _ = vt.log_channel_posteriors_ball_sigma_vector(
        gctx, uni, ball, sigma_pairs, estimator_variant=vt.ESTIMATOR_VARIANT_A_FULL
    )
    ln1_gsel, ln2_gsel, _ = vt.log_channel_posteriors_ball_sigma_vector(
        gctx, uni, ball, sigma_pairs, estimator_variant=vt.ESTIMATOR_VARIANT_A_FULL_GSEL
    )

    np.testing.assert_array_equal(ln1_full, ln1_gsel)
    assert np.all(np.isfinite(ln2_gsel))
    assert not np.array_equal(ln2_full, ln2_gsel)


def test_a_full_gsel_1d_channel_bit_identical_to_a_full_point_branch() -> None:
    """Same bit-identity check on the point branch (σ_z = 0 rows)."""
    gctx, uni, ball = _small_case(sigma_z=0.0, n_events=12, seed=405)
    sigma_pairs = np.zeros(ball.z_obs.size, dtype=np.float64)

    ln1_full, ln2_full, _ = vt.log_channel_posteriors_ball_sigma_vector(
        gctx, uni, ball, sigma_pairs, estimator_variant=vt.ESTIMATOR_VARIANT_A_FULL
    )
    ln1_gsel, ln2_gsel, _ = vt.log_channel_posteriors_ball_sigma_vector(
        gctx, uni, ball, sigma_pairs, estimator_variant=vt.ESTIMATOR_VARIANT_A_FULL_GSEL
    )

    np.testing.assert_array_equal(ln1_full, ln1_gsel)
    assert np.all(np.isfinite(ln2_gsel))
    assert not np.array_equal(ln2_full, ln2_gsel)


def test_a_full_gsel_runs_finite_on_mixed_sigma_rows() -> None:
    """A ball with both kernel- and point-branch candidates stays finite."""
    gctx, uni, ball = _small_case(sigma_z=0.035, n_events=10, seed=407)
    sigma_pairs = np.where(np.arange(ball.z_obs.size) % 2 == 0, 0.0, 0.04).astype(np.float64)

    ln1_gsel, ln2_gsel, slope_gsel = vt.log_channel_posteriors_ball_sigma_vector(
        gctx, uni, ball, sigma_pairs, estimator_variant=vt.ESTIMATOR_VARIANT_A_FULL_GSEL
    )
    assert np.all(np.isfinite(ln1_gsel))
    assert np.all(np.isfinite(ln2_gsel))
    assert np.all(np.isfinite(slope_gsel))


def test_a_full_gsel_2d_channel_differs_from_base() -> None:
    """The fused g_sel 2D channel differs from the coded base estimator too."""
    gctx, uni, ball = _small_case(sigma_z=0.035, n_events=10, seed=409)
    sigma_pairs = np.full(ball.z_obs.size, 0.035, dtype=np.float64)

    _, ln2_base, _ = vt.log_channel_posteriors_ball_sigma_vector(
        gctx, uni, ball, sigma_pairs, estimator_variant=vt.ESTIMATOR_VARIANT_BASE
    )
    _, ln2_gsel, _ = vt.log_channel_posteriors_ball_sigma_vector(
        gctx, uni, ball, sigma_pairs, estimator_variant=vt.ESTIMATOR_VARIANT_A_FULL_GSEL
    )
    assert not np.array_equal(ln2_base, ln2_gsel)


# ── (c) h-grain forwarding ────────────────────────────────────────────────────


def test_a_full_gsel_forwarded_through_hgrain_path() -> None:
    """The h-grain (fork-pool) path stays bit-identical to serial for A-FULL-2D."""
    gctx, uni, ball = _small_case(sigma_z=0.03, n_events=8, seed=411)
    sigma_pairs = np.full(ball.z_obs.size, 0.03, dtype=np.float64)

    ln1_s, ln2_s, slope_s = vt.log_channel_posteriors_ball_sigma_vector(
        gctx, uni, ball, sigma_pairs, estimator_variant=vt.ESTIMATOR_VARIANT_A_FULL_GSEL
    )
    ln1_h, ln2_h, slope_h = vt.log_channel_posteriors_ball_sigma_vector_hgrain(
        gctx, uni, ball, sigma_pairs, estimator_variant=vt.ESTIMATOR_VARIANT_A_FULL_GSEL
    )
    np.testing.assert_array_equal(ln1_s, ln1_h)
    np.testing.assert_array_equal(ln2_s, ln2_h)
    np.testing.assert_array_equal(slope_s, slope_h)


# ── (d) AFULL2D cell registration (ledger row #115 item 2) ──────────────────


def _block(spec: vt.VenueCellSpec, h_true: float = 0.730) -> set[int]:
    return set(vt.venue_cell_seeds(spec, h_true, 0, None))


def test_afull2d_is_registered_with_correct_variant_and_dose() -> None:
    """The AFULL2D cell spec carries the registered variant, dose target, and dose."""
    afull2d = vt.AFULL2D_CELL_SPECS["AFULL2D"]

    assert afull2d.estimator_variant == vt.ESTIMATOR_VARIANT_A_FULL_GSEL
    assert afull2d.n_seeds == (25,)
    assert afull2d.seed_offsets == (54300,)
    assert afull2d.dose_target == "all"
    assert afull2d.balls == "real_k"
    assert afull2d.sigma_mode == "glade"
    assert afull2d.truths == (0.730,)
    assert afull2d.prereg_cell == "A-FULL-2D"

    # every other pre-existing registry is untouched by the new one
    assert not (set(vt.AFULL2D_CELL_SPECS) & set(vt.CELL_SPECS))
    assert not (set(vt.AFULL2D_CELL_SPECS) & set(vt.MECH_CELL_SPECS))
    assert not (set(vt.AFULL2D_CELL_SPECS) & set(vt.SCAN_CELL_SPECS))
    assert not (set(vt.AFULL2D_CELL_SPECS) & set(vt.M2P_CELL_SPECS))
    assert not (set(vt.AFULL2D_CELL_SPECS) & set(vt.REN_CELL_SPECS))
    assert not (set(vt.AFULL2D_CELL_SPECS) & set(vt.AFULL_CELL_SPECS))
    assert set(vt.ALL_CELL_SPECS) == (
        set(vt.CELL_SPECS)
        | set(vt.MECH_CELL_SPECS)
        | set(vt.SCAN_CELL_SPECS)
        | set(vt.M2P_CELL_SPECS)
        | set(vt.REN_CELL_SPECS)
        | set(vt.AFULL_CELL_SPECS)
        | set(vt.AFULL2D_CELL_SPECS)
    )


def test_seed_plan_disjointness_afull2d_vs_all_documented_blocks() -> None:
    """AFULL2D (+54300..+54324) is disjoint from every previously documented
    block (v1/v2/v3 envelopes, reserved W1/O2, MECH_CELL_SPECS,
    SCAN_CELL_SPECS, M2P_CELL_SPECS, REN_CELL_SPECS, AFULL_CELL_SPECS).
    """
    afull2d = _block(vt.AFULL2D_CELL_SPECS["AFULL2D"])
    assert len(afull2d) == 25
    assert afull2d == set(range(20260808 + 54300, 20260808 + 54325))

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
        assert not any(lo <= s <= hi for s in afull2d), "AFULL2D collides with a v-envelope"

    reserved: set[int] = set()
    for lo_off, hi_off in vt.RESERVED_SEED_OFFSET_BLOCKS.values():
        reserved.update(range(vt.VT_BASE_SEED + lo_off, vt.VT_BASE_SEED + hi_off + 1))
    assert not (afull2d & reserved), "AFULL2D collides with a reserved block"

    mech_blocks: dict[str, set[int]] = {
        name: _block(spec) for name, spec in vt.MECH_CELL_SPECS.items()
    }
    for mech_name, mech_block in mech_blocks.items():
        assert not (afull2d & mech_block), f"AFULL2D collides with MECH block {mech_name}"

    scan_all: set[int] = set()
    for spec in vt.SCAN_CELL_SPECS.values():
        scan_all.update(_block(spec))
    assert not (afull2d & scan_all), "AFULL2D collides with the 2-D dose scan"

    m2p_all: set[int] = set()
    for spec in vt.M2P_CELL_SPECS.values():
        m2p_all.update(_block(spec))
    assert not (afull2d & m2p_all), "AFULL2D collides with the stage-2 M2P arms"

    ren_all: set[int] = set()
    for spec in vt.REN_CELL_SPECS.values():
        ren_all.update(_block(spec))
    assert not (afull2d & ren_all), "AFULL2D collides with the stage-3 REN/JREN arms"

    afull_all: set[int] = set()
    for spec in vt.AFULL_CELL_SPECS.values():
        afull_all.update(_block(spec))
    assert not (afull2d & afull_all), "AFULL2D collides with the stage-5 AFULL arm"


def test_afull2d_stamps_the_a_full_2d_preregistration() -> None:
    """AFULL2D maps to its own prereg, not any parent document."""
    assert vt.preregistration_path_for_cell("AFULL2D") == vt.AFULL2D_PREREG_PATH
    assert vt.AFULL2D_PREREG_PATH != vt.AFULL_PREREG_PATH
    assert vt.AFULL2D_PREREG_PATH != vt.REN_PREREG_PATH
    assert vt.AFULL2D_PREREG_PATH != vt.M2P_PREREG_PATH
    assert vt.AFULL2D_PREREG_PATH != vt.MECH_PREREG_PATH
    assert vt.AFULL2D_PREREG_PATH != vt.PREREG_PATH
    # unaffected cells still map to their own registries
    assert vt.preregistration_path_for_cell("AFULL") == vt.AFULL_PREREG_PATH
    assert vt.preregistration_path_for_cell("AJREN") == vt.REN_PREREG_PATH
    assert vt.preregistration_path_for_cell("AM2P") == vt.M2P_PREREG_PATH
    assert vt.preregistration_path_for_cell("MN0X") == vt.MECH_PREREG_PATH
    assert vt.preregistration_path_for_cell("T0") == vt.PREREG_PATH


def test_cli_choices_include_afull2d() -> None:
    """The CLI --cell parser accepts the new AFULL2D cell."""
    parser = vt.build_parser()
    cell_action = next(a for a in parser._actions if a.dest == "cell")
    assert cell_action.choices is not None
    assert "AFULL2D" in set(cell_action.choices)


def test_unknown_estimator_variant_still_rejected_with_a_full_gsel_present() -> None:
    """An unregistered variant name must still fail loudly after this addition."""
    gctx, uni, ball = _small_case(sigma_z=0.03, n_events=4, seed=401)
    sigma_pairs = np.full(ball.z_obs.size, 0.03, dtype=np.float64)

    with pytest.raises(ValueError, match="unknown estimator_variant"):
        vt.log_channel_posteriors_ball_sigma_vector(
            gctx, uni, ball, sigma_pairs, estimator_variant="not-a-real-variant"
        )
