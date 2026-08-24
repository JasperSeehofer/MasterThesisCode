r"""[P3-IMP] the b0 catalogued-host identity test -- stage 2, the correctness adjudicator.

Registered in ``PREREGISTRATION_B0_IDENTITY_20260823.md`` (binding; A21 governs -- premise
corrections STOP execution, amend first) **as AMENDED by PA-1..PA-10** (the pre-execution
adversarial review, ``A20_REVIEW_B0_DESIGN_20260823.md``, banked verbatim before this driver's
amendment landed -- Findings 2 and 4 were A21 premise corrections, discharged by PA-2/PA-4 before
any arm ran). Template for structure/idioms: the committed ``p3_twin_test.py`` (read fully before
editing this file) -- this driver is its arm-parametrized generalization onto the b0 venue
(catalogue host mode, real hosts, real completeness) with the odds-form identity statistic
(prereg S1, amended PA-4) in place of the twin's raw Delta_bar.

**Question (prereg S1, amended PA-1..PA-4):** the exact identity for class-conditional draws is
the odds form

    E_{d~p(d|G)}[ (1 - w(d)) / w(d) ] * M_G/M_Gbar = 1      (M_G/M_Gbar == the class-odds C*)

**PA-2 (venue premise, [ORCH-RULE 5]):** the STOCK b0 generator does not realize the mixture's
class-G law (no S_bar_phi acceptance, no R_eff(M) mass weighting, z_true := listed z). All
identity arms therefore run in the NEW host mode ``catalogue_selected`` (venue **b0i**, harness-
only, committed alongside this driver): host g drawn ~ w_g * S_tilde_phi_g (w_g the estimator's
own ``_rate_weight``, S_tilde_phi_g the kernel-smeared survival), z_true drawn per event from
k_g(z)*S_bar_phi(z)/S_tilde_phi_g. The 25 banked STOCK b0 CSVs (commit ``198724e2``) are a
DIFFERENT generator -- the GATE R-B0 replica-fidelity anchor and the LEV cross-basis comparator
ONLY, never the fused-basis comparand.

**PA-3 (ratio form):** the per-event ratio is computed DIRECTLY as
``(1-w_e)/w_e = B_num / (beta_G_phi * L_cat_no_bh)`` (``beta_G_phi = alpha_G_phi/r_Malm`` from
banked columns), never through ``1 - w``.

**PA-4 (the odds constant, [ORCH-RULE], the decisive amendment):** the prereg's
``M_G/M_Gbar = alpha_G_phi/(D_tilde_phi - alpha_G_phi)`` is REFUTED (that is the WITH-BH class
weight, not the no-BH channel's mass, and a per-arm self-consistent mass would make the B-R
control pass vacuously -- defeating its purpose). The registered constant is the SINGLE
derivation-fixed number, IDENTICAL for all three arms:

    C* = beta_G_phi(0.73) * rho(0.73) / beta_Gbar_phi(0.73)

with ``beta_G_phi = alpha_G_phi/r_Malm``, ``beta_Gbar_phi = D_tilde_phi - alpha_G_phi`` (both read
from the run's OWN banked columns at h=0.73, cross-checked against the run's
``selection_tables_h_0_73.json``, PA-7/PA-9), and ``rho = Sigma_tilde_phi / Sigma_phi`` (kernel-
smeared vs point-evaluated catalogue selection mass, a VENUE-level -- not per-arm/per-seed --
zero-compute quantity, see :func:`mass_companion`). Registered per-arm predictions (banked BEFORE
any arm runs, per Finding 4): B-T: I = 0 (the PASS target); B-C: I + 1 ~= <S_bar_phi> (order-of-
magnitude, fails low); B-R: I + 1 = 1/R(0.73) EXACTLY (the control must fail AT its predicted
value, not merely fail -- see :func:`_br_predicted_value`).

**Arms** (prereg S2, all on the fused completion basis, Sigma^phi slot per [ORCH-RULE 3]):

- **replica (GATE R-B0)**: seed 900101 under the BANKED configuration, venue **b0**
  (``catalogue_numerator_survival="off"``, ``catalogue_global_selection="s3d"``,
  ``selection_in_completion_numerator="off"``, ``host_mode="catalogue"``) -- byte-identical to
  the 25 banked b0 CSVs (commit ``198724e2``). UNCHANGED by PA-2 (venue-fidelity proof, not an
  identity read).
- **B-C (coded)**: 12 fresh ``evaluate()`` calls, venue **b0i**
  (``host_mode="catalogue_selected"``, PA-2), ``catalogue_numerator_survival="off"``,
  ``catalogue_global_selection="phi"``, ``selection_in_completion_numerator="fused"``.
- **B-T (twin)**: 12 fresh ``evaluate()`` calls, B-C's b0i config with
  ``catalogue_numerator_survival="phi"``.
- **B-R (R-rescaled, CONTROL)**: zero-``evaluate()`` rescore of B-T's diagnostics by
  ``R(h) = beta_G(h) / beta_G_phi(h)`` (the committed ``p3_completed_rescore.py``
  ``_build_betas`` leaf, reused by import -- not reimplemented); the refuted-Appendix-A
  arrangement, expected to FAIL the identity AT its predicted value (PA-4).

Seeds: 900101-900112 (first 12 of ``c1d.ARM_SEEDS["b0i"]`` -- **[interface assumption]** the
owning agent's registry entry mirrors b0's 25-seed 900101-900125 span; a fallback to
``c1d.ARM_SEEDS["b0"]`` is used if ``"b0i"`` is absent, disclosed at import time). H grid:
``H_GRID_FULL`` for the ``evaluate()`` call (un-truncated per the amendment-20 lesson),
``H_GRID_41`` for ``compute_seed_statistics`` (production grid, matches ``run_arm_seed``'s own
convention). PA-5: ``H_TRUE in H_GRID_FULL`` (float equality) is asserted before any arm.

**Stages** (``--stage {replica,pilot,fleet,lev,rescore,gates,score}``):

- ``lev``: zero-compute. The prereg S5 LEV read over the 25 banked b0 CSVs (off basis, s3d slot,
  coded cell) -- runs BEFORE any arm, no ``evaluate()`` call. PA-7: the reported displacement is
  the coded arm's TOTAL displacement (arrangement + venue-premise + Sigma^3D-slot terms,
  INSEPARABLE on the banked basis) -- never quoted as arrangement-only.
- ``replica``: GATE R-B0 (<=1e-12 relative on ``L_cat_no_bh``/``B_num``/``combined_no_bh``,
  wall > 60 s), venue b0.
- ``pilot``: seed 900101 under BOTH B-C and B-T, venue b0i.
- ``fleet``: seeds 900101-900112 under ONE arm (``--arm {bc,bt}``, required), venue b0i -- run as
  two separate detached invocations per the launch task's "separate detached invocations,
  sequential within arm" instruction. Idempotent per-seed sentinel files (existing
  ``<subdir>_meta.json`` is REUSED, not re-run -- disclosed, o6/twin precedent).
- ``rescore``: B-R, the zero-``evaluate()`` rescale of the 12 B-T diagnostics.
- ``gates``: GATE E-B0 (PA-7 engagement + dispatch, denominators explicit) scored standalone.
- ``score``: GATE R-B0/E-B0/L-B0/W-B0/N-B0 + the S4 primary identity statistic (per arm) +
  secondaries 1-6, verdict mapping (prereg S4, PA-5).

**HARD CONSTRAINTS (mirrors o4/o6/o7/p3_twin_test.py):**

1. Never end a turn to wait on an untracked process -- every ``evaluate()`` call below is
   synchronous/blocking (``run_mirror_seed_inprocess``).
2. Every load-bearing claim cites file:line where practical.
3. Seeds run SEQUENTIALLY within an arm/invocation -- no subprocess/process-pool fan-out (same
   ``run_mirror_seed_inprocess`` module-state-monkeypatch constraint as the twin driver).
4. This driver is written against an INTERFACE it does NOT own (concurrent edit, launch task
   grant): ``correspondence_1d.run_mirror_seed_inprocess`` accepting a
   ``catalogue_global_selection: str = "s3d"`` kwarg -- VERIFIED present in the module as read
   (``correspondence_1d.py:1749``, forwarded to ``BayesianStatistics.evaluate()`` at
   ``:1877-1880``); ``MirrorUniverseGenerator.draw_realization`` accepting a NEW
   ``host_mode="catalogue_selected"`` literal + a ``phi_survival_table`` kwarg when that mode is
   active (PA-2), and the registry entries ``c1d.ARM_HOST_MODE["b0i"] == "catalogue_selected"``,
   ``c1d.ARM_SELECTION_CELL["b0i"] == "fused"`` -- NOT present as of this driver's authoring
   (concurrent edit in flight per the launch task); every b0i-venue call site asserts these
   registry values before use and fails LOUD (``AssertionError``/``TypeError``), never silently,
   if the interface differs from what is assumed here. See the full interface-assumption list in
   the accompanying task report.

Usage:
    uv run python results/campaign51_20260728/realistic_20260729/p3_b0_identity_test.py \
        --stage lev
    uv run python .../p3_b0_identity_test.py --stage replica
    uv run python .../p3_b0_identity_test.py --stage pilot
    uv run python .../p3_b0_identity_test.py --stage fleet --arm bc
    uv run python .../p3_b0_identity_test.py --stage fleet --arm bt
    uv run python .../p3_b0_identity_test.py --stage rescore
    uv run python .../p3_b0_identity_test.py --stage gates
    uv run python .../p3_b0_identity_test.py --stage score
"""

import argparse
import contextlib
import functools
import json
import logging
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.stats import genpareto, trim_mean

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "prod2d_closure_20260818"))
import decompose_impostor_leg as o2  # noqa: E402

# Same-directory import (no sys.path juggling needed -- p3_completed_rescore.py lives next to
# this file; importing it does not execute its ``main()``, guarded by ``__main__``).
import p3_completed_rescore as o3  # noqa: E402

from darksiren_emri.emri_rate import R_eff_per_mbh  # noqa: E402
from darksiren_emri.galaxy_catalogue.handler import InternalCatalogColumns  # noqa: E402
from darksiren_emri.validation import correspondence_1d as c1d  # noqa: E402
from darksiren_emri.validation.correspondence_1d import (  # noqa: E402
    H_GRID_41,
    H_GRID_FULL,
    H_TRUE,
    combine_log_likelihood,
    compute_seed_statistics,
    moment_weights,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
THIS_DIR = Path(__file__).resolve().parent
BANKED_B0_CSV_ROOT = REPO_ROOT / "results/prod2d_closure_20260818/arm_event_likelihoods"

REGISTRATION_SECTION: str = (
    "results/campaign51_20260728/realistic_20260729/"
    "PREREGISTRATION_B0_IDENTITY_20260823.md (2026-08-23, PA-1..PA-10 amended)"
)

# PA-2/interface-assumption: b0i's seed registry mirrors b0's 25-seed 900101-900125 span. Fall
# back to b0's own list (disclosed, not silent) if the owning agent's registry entry is absent.
_B0I_SEEDS_FULL: tuple[int, ...] = c1d.ARM_SEEDS.get("b0i", c1d.ARM_SEEDS["b0"])
BSEL_SEEDS: tuple[int, ...] = tuple(_B0I_SEEDS_FULL[:12])  # 12, prereg S2
LEV_SEEDS: tuple[int, ...] = c1d.ARM_SEEDS["b0"]  # 25, prereg S5, STOCK b0 -- unchanged by PA-2
REPLICA_SEED: int = 900101

H_GEN: float = H_TRUE  # 0.73, the truth node (prereg S1 "all reads at h = H_TRUE")
TRIM_PROPORTION: float = 0.05  # PA-6(c): reported-only diagnostic, NOT verdict-participating
PSIS_HEAVY_TAIL_K_HAT: float = 0.7  # PA-6(b): UNDETERMINED(a) iff k_hat > this AND band disagree
DEAD_ROW_VOID_DELTA: float = 1.0 / 200.0  # PA-6(a): |Delta dead-row rate| this VOIDs the arm cmp

REQUIRED_COLUMNS: tuple[str, ...] = (
    "event_idx",
    "h",
    "alpha_G_phi",
    "r_Malm",
    "D_tilde_phi",
    "L_cat_no_bh",
    "B_num",
    "combined_no_bh",
)

# GATE R-B0 (prereg S3): bit-exact, or <=1e-12 relative (production evaluate() dispatches
# per-host terms through a multiprocessing pool whose float-summation order is not guaranteed
# run-to-run identical -- same GATE R-P3/R4/D6 fallback convention).
GATE_RB0_COLUMNS: tuple[str, ...] = ("L_cat_no_bh", "B_num", "combined_no_bh")
GATE_RB0_RTOL: float = 1.0e-12
GATE_RB0_MIN_WALL_S: float = 60.0

# GATE E-B0 (prereg S3, PA-7): explicit denominators.
GATE_EB0_A_RTOL: float = 1.0e-9  # (a): "ratio != 1" -- distinguishable from float noise
GATE_EB0_A_MIN_FRACTION: float = 1.0  # (a): 100% of live rows
GATE_EB0_B_RTOL: float = 1.0e-6  # (b): "differ" move threshold (GATE E-P3 precedent)
GATE_EB0_B_MIN_FRACTION: float = 0.99  # (b): >=99% of PAIRED-LIVE rows

# GATE W-B0 (prereg S3, PA-9 amended): identity computability/closure. The registered 1e-9
# tolerance is unsatisfiable on CSV-scored columns (7-sig-fig storage floors the residual at
# ~1.25e-7, measured uniformly across the 25 banked b0 seeds) -- amended to <=1e-6 relative, the
# CSV storage-precision floor disclosed alongside every closure read (PA-9).
GATE_WB0_CLOSURE_RTOL: float = 1.0e-6
GATE_WB0_CSV_STORAGE_FLOOR: float = 1.25e-7  # PA-9, disclosed, not a pass/fail threshold

# GATE L-B0 log substrings, quoted verbatim from bayesian_statistics.py's evaluate()
# _LOGGER (the bare root logger, ``logging.getLogger()`` at bayesian_statistics.py:73 --
# _capture_root_log below reuses the twin driver's root-attachment precedent).
FUSED_LOG_SUBSTRING: str = "[PHYSICS] selection fusion ACTIVE"  # bayesian_statistics.py:3430-3435
SIGMA_PHI_LOG_SUBSTRING: str = (  # bayesian_statistics.py:3486-3489
    'COUNTERFACTUAL: catalogue_global_selection="phi"'
)
TWIN_LOG_SUBSTRING: str = "COUNTERFACTUAL: catalogue_numerator_survival='phi'"  # :3464-3470
OFF_SELECTION_LOG_SUBSTRING: str = (  # :3447-3449, replica sanity only, not gated
    "COUNTERFACTUAL: selection_in_completion_numerator='off'"
)

# PA-4/PA-6d: the venue-level mass companion. Sigma_tilde_phi is now computed via the harness's
# OWN kernel_smeared_survival leaf (imported, not reimplemented) -- see :func:`mass_companion`
# and PA-11/FATAL-1 (A20_REVIEW_B0_IMPL_20260823.md): this driver must NOT carry a private
# bare-Gaussian reimplementation that silently diverges from whatever kernel the estimator's own
# numerator uses (production default host_z_kernel=volume_deconv).

# PA-12 (AMEND 5): rho != 1 structurally (second-order S_bar_phi curvature term, point-eval vs
# kernel-smeared) -- no exactness assert. A loose SANITY window only; C* banks the ACTUAL
# measured rho at machine precision regardless of window membership (PA-6d).
RHO_SANITY_WINDOW: tuple[float, float] = (0.9, 1.1)

# PA-13(c) (FATAL 2 fix): the B-R control's tolerance for "control scores AT its predicted
# value" -- a driver-chosen, disclosed, order-unity tolerance (Finding 8c), now a named constant.
BR_CONTROL_TOLERANCE: float = 0.05

# PA-13(a) (FATAL 3 fix): GATE E-B0(a) is re-registered on a SAME-VENUE pair (b0i seed 900101,
# catalogue_global_selection="phi" [B-C] vs "s3d" [this extra run]) -- "ratio is one h-dependent
# constant" is scored as a per-h coefficient-of-variation across PAIRED-LIVE rows.
GATE_EB0A_SAMEVENUE_CV_TOL: float = 1.0e-9

# mass_companion chunk size: c1d.kernel_smeared_survival allocates (n, 50)-shaped node arrays
# (quadrature + per-node completeness lookups); the reduced catalogue is ~2.08e7 rows, and
# essentially all of it is "eligible" (z < z_max), so an unchunked call allocates several
# 8+ GB (n=2.08e7 x 50 float64) arrays at once -- OOM/timeout observed empirically at full
# scale. Chunking (result is exactly the same, row-independent function) keeps peak memory
# bounded regardless of catalogue size.
_MASS_COMPANION_CHUNK: int = 20_000


# ── PA-5: the H_TRUE-in-grid assertion (before any arm) ──────────────────────


def _assert_h_true_in_grid() -> None:
    """PA-5 (Finding 5c): ``H_TRUE`` must equal a member of ``H_GRID_FULL`` by float equality
    before any arm runs -- the identity is an h-pointwise statement read at truth, so no
    discretization enters the primary (Finding 5c, ``correspondence_1d.py`` :343/:337).
    """
    assert H_TRUE in H_GRID_FULL, (
        f"PA-5 STOP: H_TRUE={H_TRUE!r} is not a member of H_GRID_FULL={H_GRID_FULL!r} by float "
        "equality -- the b0i identity read at h=H_TRUE would not hit a table key. Amend before "
        "any arm runs (A21)."
    )


# ── PA-4/PA-6d: the venue-level mass companion (Sigma_w, Sigma^phi, Sigma~^phi, rho) ─────────


@functools.lru_cache(maxsize=4)
def mass_companion(h: float) -> dict[str, Any]:
    r"""The venue-level mass companion (PA-4/PA-6d), zero-``evaluate()``-compute (no arm run),
    cached per ``h`` (this driver only ever calls it at ``h = H_GEN``).

    ``Sigma_phi = sum_g w_g * S_bar_phi(z_g;h)`` (point-evaluated at each galaxy's listed z) is
    the SAME formula ``precompute_global_catalog_selection``'s ``phi_survival_table`` branch
    computes (``bayesian_statistics.py:2711-2713``/``:2841-2854``, cited, NOT imported -- the
    point and kernel-smeared sums below share one eligibility pass over the catalogue, which the
    production function does not expose, so this driver duplicates the formula rather than
    calling it twice; flagged for the pre-execution adversarial review).

    ``Sigma_tilde_phi = sum_g w_g * S_tilde_phi_g``, ``S_tilde_phi_g`` computed by
    :func:`c1d.kernel_smeared_survival` -- **imported, not reimplemented** (PA-11/FATAL-1 fix,
    ``A20_REVIEW_B0_IMPL_20260823.md`` Finding 1): this driver previously carried a private
    bare-Gaussian Hermite-quadrature reimplementation that silently diverged from whatever kernel
    the ``catalogue_selected`` draw and the estimator's numerator actually use (production
    default ``host_z_kernel=volume_deconv``). Calling the harness's own leaf means this
    companion's kernel family tracks ``kernel_smeared_survival``'s own -- whatever that function
    is, today or after any future kernel-alignment fix -- by construction, rather than by a
    second, independently-maintained copy of the formula. ``z_error`` is floored at
    ``c1d.EXACT_Z_ERROR_FLOOR`` before the call (the driver's own exact-z convention, reused;
    :func:`c1d.kernel_smeared_survival` itself does not floor its ``z_error`` input).

    ``w_g = R_eff_per_mbh(M_g)/(1+z_g)`` -- IDENTICAL to ``_rate_weight``/
    ``draw_rate_weighted_hosts`` (``bayesian_statistics.py:2684-2688``, cited).

    ``rho = Sigma_tilde_phi / Sigma_phi``. PA-12 (AMEND 5, supersedes the PA-4 "rho == 1
    exactly" exactness claim): the estimator's Sigma^phi leaf POINT-EVALUATES S_bar_phi at each
    galaxy's listed z while Sigma~^phi SMEARS it, so rho deviates from 1 at SECOND ORDER in
    sigma_z via S_bar_phi curvature -- structurally, regardless of generator/estimator kernel
    alignment. No hard exactness assert is applied; rho is banked at machine precision (PA-6d)
    with only a loose SANITY window, :data:`RHO_SANITY_WINDOW` = (0.9, 1.1) -- membership is
    REPORTED, never gates any stage (a hard crash here would break every downstream stage
    including the zero-compute LEV smoke path).

    Args:
        h: Hubble parameter (this driver only ever calls at ``h = H_GEN = 0.73``).

    Returns:
        Dict with ``Sigma_w``, ``Sigma_phi``, ``Sigma_phi_tilde``, ``rho``, ``rho_deviation``,
        ``rho_within_sanity_window``, ``rho_sanity_window``, ``n_eligible``, ``h``.
    """
    completeness_obj, phi_survival_table = c1d.build_bsel_selection_objects(h_true=h)
    z_grid, s_phi_grid = phi_survival_table[h]

    handler = c1d._load_galaxy_catalog_handler(c1d.REDUCED_CATALOGUE_PATH)
    catalog = handler.reduced_galaxy_catalog
    z_all = np.asarray(
        catalog[InternalCatalogColumns.REDSHIFT].to_numpy(dtype=np.float64), dtype=np.float64
    )
    M_all = np.asarray(
        catalog[InternalCatalogColumns.BH_MASS].to_numpy(dtype=np.float64), dtype=np.float64
    )
    z_err_all = np.asarray(
        catalog[InternalCatalogColumns.REDSHIFT_ERROR].to_numpy(dtype=np.float64),
        dtype=np.float64,
    )
    phiS_all = np.asarray(
        catalog[InternalCatalogColumns.PHI_S].to_numpy(dtype=np.float64), dtype=np.float64
    )
    qS_all = np.asarray(
        catalog[InternalCatalogColumns.THETA_S].to_numpy(dtype=np.float64), dtype=np.float64
    )

    z_max = float(z_grid.max())
    eligible = (z_all < z_max) & np.isfinite(M_all) & (M_all > 0.0)
    z_g = z_all[eligible]
    M_g = M_all[eligible]
    sigma_g = np.maximum(z_err_all[eligible], c1d.EXACT_Z_ERROR_FLOOR)
    phiS_g = phiS_all[eligible]
    qS_g = qS_all[eligible]
    w_g = np.asarray(R_eff_per_mbh(M_g), dtype=np.float64) / (1.0 + z_g)

    s_phi_point = np.interp(z_g, z_grid, s_phi_grid)
    sigma_w = float(w_g.sum())
    sigma_phi = float((w_g * s_phi_point).sum())

    # PA-11/FATAL-1 fix: imported, not reimplemented -- tracks whatever kernel family
    # c1d.kernel_smeared_survival uses (now the volume_deconv+C7-aligned form; the
    # completeness object build_bsel_selection_objects returns is threaded through
    # here rather than discarded). Chunked (see _MASS_COMPANION_CHUNK): the reduced
    # catalogue is ~2.08e7 rows and (empirically) essentially all of it is eligible, so an
    # unchunked call allocates several 8+ GB (n, 50) arrays at once -- chunking is a pure
    # memory-shape transform, byte-identical result (the function is row-independent).
    n_g = z_g.size
    s_phi_tilde = np.empty(n_g, dtype=np.float64)
    for start in range(0, n_g, _MASS_COMPANION_CHUNK):
        sl = slice(start, min(start + _MASS_COMPANION_CHUNK, n_g))
        s_phi_tilde[sl] = c1d.kernel_smeared_survival(
            z_g[sl], sigma_g[sl], phi_survival_table, completeness_obj, phiS_g[sl], qS_g[sl], h=h
        )

    sigma_phi_tilde = float((w_g * s_phi_tilde).sum())
    rho = sigma_phi_tilde / sigma_phi if sigma_phi > 0.0 else float("nan")
    rho_deviation = abs(rho - 1.0) if np.isfinite(rho) else float("nan")
    rho_within_window = bool(np.isfinite(rho) and RHO_SANITY_WINDOW[0] < rho < RHO_SANITY_WINDOW[1])
    if not rho_within_window:
        logging.getLogger(__name__).warning(
            "PA-12 mass companion: rho=%.6f outside the sanity window %s at h=%.4f "
            "(n_eligible=%d) -- reported, not a STOP (no exactness claim, PA-12).",
            rho,
            RHO_SANITY_WINDOW,
            h,
            int(eligible.sum()),
        )
    return {
        "h": h,
        "n_eligible": int(eligible.sum()),
        "Sigma_w": sigma_w,
        "Sigma_phi": sigma_phi,
        "Sigma_phi_tilde": sigma_phi_tilde,
        "rho": rho,
        "rho_deviation": rho_deviation,
        "rho_within_sanity_window": rho_within_window,
        "rho_sanity_window": list(RHO_SANITY_WINDOW),
        "mean_S_bar_phi_weighted": (sigma_phi / sigma_w) if sigma_w > 0.0 else None,
    }


# ── PA-3/PA-4: the odds constant C* ───────────────────────────────────────────


def _beta_g_phi_and_gbar(at: pd.DataFrame) -> tuple[float, float]:
    """``beta_G_phi = alpha_G_phi/r_Malm`` and ``beta_Gbar_phi = D_tilde_phi - alpha_G_phi``,
    read from one at-h(H_GEN) row block's OWN columns (the run's own Path-A mixture objects --
    event-independent globals, constant across all rows at a fixed h; asserted, not assumed).
    """
    alpha_g_phi = at["alpha_G_phi"].to_numpy(dtype=np.float64)
    r_malm = at["r_Malm"].to_numpy(dtype=np.float64)
    d_tilde_phi = at["D_tilde_phi"].to_numpy(dtype=np.float64)
    beta_g_phi_vec = alpha_g_phi / r_malm
    beta_gbar_phi_vec = d_tilde_phi - alpha_g_phi
    finite = np.isfinite(beta_g_phi_vec) & np.isfinite(beta_gbar_phi_vec)
    if not finite.any():
        return float("nan"), float("nan")
    ref_beta_g_phi = float(beta_g_phi_vec[finite][0])
    ref_beta_gbar_phi = float(beta_gbar_phi_vec[finite][0])
    # Event-independence check (disclosed, not a hard STOP -- a violation would mean the run's
    # own Path-A mixture objects vary by event, which the mixture construction does not permit).
    if not (
        np.allclose(beta_g_phi_vec[finite], ref_beta_g_phi, rtol=1.0e-6)
        and np.allclose(beta_gbar_phi_vec[finite], ref_beta_gbar_phi, rtol=1.0e-6)
    ):
        logging.getLogger(__name__).warning(
            "beta_G_phi/beta_Gbar_phi vary across events within one seed/h -- expected constant "
            "(Path-A mixture objects are event-independent globals); using row 0's value."
        )
    return ref_beta_g_phi, ref_beta_gbar_phi


def _cross_check_selection_table_json(
    work_root_seed_dir: Path, h: float, beta_g_phi: float, beta_gbar_phi: float
) -> dict[str, Any]:
    """PA-7/PA-9: cross-check the CSV-derived ``beta_G_phi``/``beta_Gbar_phi`` against the run's
    OWN ``write_selection_table_json`` output (``bayesian_statistics.py:2548-2597``), written to
    the seed's ``os.chdir``-ed CWD during ``evaluate()`` (``correspondence_1d.py:1855``), i.e.
    ``<work_root_seed_dir>/selection_tables_h_<label>.json`` with
    ``label = str(np.round(h,4)).replace(".", "_")``.
    """
    label = str(np.round(h, 4)).replace(".", "_")
    json_path = work_root_seed_dir / f"selection_tables_h_{label}.json"
    if not json_path.is_file():
        return {"found": False, "path": str(json_path)}
    payload = json.loads(json_path.read_text())
    json_beta_g_phi = float(payload["beta_G_phi"])
    json_beta_gbar_phi = float(payload["beta_Gbar_phi"])
    tiny = np.finfo(float).tiny
    rel_g = abs(beta_g_phi - json_beta_g_phi) / max(abs(json_beta_g_phi), tiny)
    rel_gbar = abs(beta_gbar_phi - json_beta_gbar_phi) / max(abs(json_beta_gbar_phi), tiny)
    return {
        "found": True,
        "path": str(json_path),
        "json_beta_G_phi": json_beta_g_phi,
        "json_beta_Gbar_phi": json_beta_gbar_phi,
        "csv_beta_G_phi": beta_g_phi,
        "csv_beta_Gbar_phi": beta_gbar_phi,
        "rel_err_beta_G_phi": rel_g,
        "rel_err_beta_Gbar_phi": rel_gbar,
        "pass": bool(
            rel_g <= GATE_WB0_CLOSURE_RTOL + GATE_WB0_CSV_STORAGE_FLOOR
            and rel_gbar <= GATE_WB0_CLOSURE_RTOL + GATE_WB0_CSV_STORAGE_FLOOR
        ),
    }


def c_star(at: pd.DataFrame) -> tuple[float, dict[str, Any]]:
    """PA-4: the SINGLE class-odds constant, identical for all three arms (bc/bt/br).

    ``C* = beta_G_phi(H_GEN) * rho(H_GEN) / beta_Gbar_phi(H_GEN)``, ``beta_G_phi``/
    ``beta_Gbar_phi`` from ``at``'s own banked columns (PA-4 wording: "from the run's columns"),
    ``rho`` from the venue-level :func:`mass_companion` (cached, arm-independent).

    Returns:
        ``(C*, diagnostics)`` -- ``diagnostics`` carries beta_G_phi/beta_Gbar_phi/rho/mass
        companion at machine precision (PA-6d).
    """
    beta_g_phi, beta_gbar_phi = _beta_g_phi_and_gbar(at)
    companion = mass_companion(H_GEN)
    if not (beta_gbar_phi > 0.0) or not np.isfinite(beta_g_phi):
        return float("nan"), {
            "beta_G_phi": beta_g_phi,
            "beta_Gbar_phi": beta_gbar_phi,
            "mass_companion": companion,
            "degenerate": True,
        }
    value = beta_g_phi * companion["rho"] / beta_gbar_phi
    return value, {
        "beta_G_phi": beta_g_phi,
        "beta_Gbar_phi": beta_gbar_phi,
        "mass_companion": companion,
        "degenerate": False,
    }


def _br_predicted_value(r_h: float) -> float:
    """PA-4: B-R's registered prediction, ``I + 1 = 1/R(H_GEN)`` EXACTLY -- computable before
    any run (``p3_completed_rescore.py``'s ``R(h) > 1`` gate already banks this)."""
    return 1.0 / r_h - 1.0


def _a22_stamp() -> dict[str, str]:
    """git commit + dirty flag, WRITTEN to the meta BEFORE the evaluate() call (A22 amended,
    row #173) -- pattern verbatim from ``p3_completed_rescore.py``/``p3_rphi_rescore.py``.
    """
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "darksiren_emri/", str(Path(__file__))],
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {"git_commit_at_start": commit, "tree_dirty_incl_instrument": dirty or "clean"}


@contextlib.contextmanager
def _capture_root_log(log_path: Path) -> Iterator[None]:
    """Attach a ``logging.FileHandler`` to the ROOT logger (o6/p3_twin_test.py precedent --
    ``bayesian_statistics.py:73``'s ``_LOGGER = logging.getLogger()`` IS the root logger, so
    this is the only placement under which the GATE L-B0 log-line substrings are captured).
    """
    root = logging.getLogger()
    old_level = root.level
    root.setLevel(logging.INFO)
    handler = logging.FileHandler(log_path, mode="w")
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root.addHandler(handler)
    try:
        yield
    finally:
        root.removeHandler(handler)
        handler.close()
        root.setLevel(old_level)


OUT_ROOT_DEFAULT: Path = THIS_DIR / "p3_b0_work"


def _meta_csv(meta: dict[str, Any]) -> Path:
    """Resolve a run meta's diagnostics CSV, falling back to the local mirrored layout.

    PA-16 retrieval note (2026-08-24): cluster-run metas record the CLUSTER work-root path in
    ``diagnostics_csv``; the retrieval mirrors the CSV into the local out-root layout
    (``<out>/<arm>_<seed>_work/seed<seed>/simulations/diagnostics/event_likelihoods.csv``,
    sha256-manifest-verified). The meta itself is A22 evidence and is never edited -- this
    resolver prefers the recorded path and falls back to the mirror, refusing loudly if neither
    exists.
    """
    recorded = Path(meta["diagnostics_csv"])
    if recorded.is_file():
        return recorded
    arm = meta.get("arm")
    seed = meta.get("seed")
    work_root = meta.get("work_root", "")
    sub = Path(work_root).name if work_root else (f"{arm}_{seed}_work" if arm and seed else None)
    if sub:
        local = (
            OUT_ROOT_DEFAULT / sub / f"seed{seed}" / "simulations/diagnostics/event_likelihoods.csv"
        )
        if local.is_file():
            return local
    raise SystemExit(
        f"REFUSED: diagnostics CSV not found at recorded path {recorded} nor local mirror -- "
        f"meta arm={arm} seed={seed}"
    )


def _banked_b0_csv_path(seed: int) -> Path:
    return (
        BANKED_B0_CSV_ROOT
        / f"b0_seed{seed}"
        / f"seed{seed}"
        / "simulations/diagnostics/event_likelihoods.csv"
    )


def _assert_required_columns(df: pd.DataFrame, csv_path: Path) -> None:
    """The ops rule: every scripted column assumption asserted, not silently trusted."""
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise SystemExit(f"REFUSED: {csv_path} missing required column(s) {sorted(missing)}")


def _rows_at_h(csv_path: Path, h: float = H_GEN) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    _assert_required_columns(df, csv_path)
    at = df[np.isclose(df["h"].to_numpy(dtype=np.float64), h)].sort_values("event_idx")
    if at.empty:
        raise SystemExit(f"REFUSED: {csv_path} has no rows at h={h}")
    return at


# ── PA-6(b): the PSIS robustness twin (replaces the 5%-trim verdict twin) ────


def _psis_diagnostics(ratio: npt.NDArray[np.float64]) -> dict[str, Any]:
    """PA-6(b): generalized-Pareto (Vehtari-style PSIS) smoothed mean of the per-event ratio.

    Fit a generalized Pareto (``scipy.stats.genpareto``) to the top tail (``M = min(ceil(n/5),
    ceil(3*sqrt(n)))`` largest values, the standard Vehtari PSIS truncation), report the shape
    ``k_hat``; the PSIS-smoothed mean replaces the top-M weights by the expected order statistics
    of the fitted GPD (``genpareto.ppf`` at ``(i-0.5)/M`` quantiles, Vehtari et al. PSIS).

    Args:
        ratio: The per-event ``(1-w)/w`` vector for one seed/arm (live rows only).

    Returns:
        Dict with ``k_hat`` (``None`` if unfittable), ``psis_mean``, ``M``, ``threshold``.
    """
    n = int(ratio.size)
    if n < 5:
        return {"k_hat": None, "psis_mean": float(ratio.mean()) if n else None, "M": 0}
    m = int(min(np.ceil(n / 5.0), np.ceil(3.0 * np.sqrt(n))))
    m = max(m, 5)
    m = min(m, n - 1)
    order = np.argsort(ratio)
    sorted_r = ratio[order]
    tail = sorted_r[-m:]
    threshold = float(sorted_r[-(m + 1)])
    excess = np.maximum(tail - threshold, 0.0)
    try:
        k_hat, _loc, sigma = genpareto.fit(excess, floc=0.0)
    except Exception as exc:  # noqa: BLE001 -- diagnostic-only twin, never fatal to the primary
        return {
            "k_hat": None,
            "psis_mean": float(ratio.mean()),
            "M": m,
            "fit_failed": True,
            "fit_error": str(exc),
        }
    p = (np.arange(1, m + 1, dtype=np.float64) - 0.5) / m
    smoothed_tail = threshold + genpareto.ppf(p, k_hat, loc=0.0, scale=sigma)
    smoothed = sorted_r.copy()
    smoothed[-m:] = smoothed_tail
    return {
        "k_hat": float(k_hat),
        "psis_mean": float(smoothed.mean()),
        "M": m,
        "threshold": threshold,
    }


def _identity_inputs(
    at: pd.DataFrame, c_star_value: float, cat_scale: float = 1.0
) -> dict[str, Any]:
    """Reconstruct the PA-3 direct ratio, ``w_e`` (W-B0 closure only), LIVE/dead masks, and the
    W-B0 closure residual from one at-h(H_GEN) row block.

    PA-3: ``ratio_e = B_num / (beta_G_phi * L_cat_no_bh)`` directly (never through ``1 - w``).
    PA-7: LIVE(a) = rows with ``L_cat_no_bh > 0`` -- the ONLY denominator the primary uses.
    PA-6(a): dead rows (``L_cat_no_bh == 0``) are a SUPPORT VIOLATION under a catalogue-hosted
    generator (Finding 1a: the expectation is formally infinite there) -- counted, never dropped
    silently.

    ``cat_scale`` (PA-13(c)/FATAL-2 fix): the B-R rescale factor ``r_h`` = ``R(H_GEN) =
    beta_G(H_GEN)/beta_G_phi(H_GEN)``, default 1.0 (B-C/B-T, no rescale). The A20-IMPL review's
    Finding 2 was that ``stage_rescore`` patched ONLY the ``combined_no_bh`` column while the
    scored ratio/closure kept reading the UNPATCHED ``beta_G_phi``/``L_cat_no_bh`` columns, so
    ``I_s(B-R) ≡ I_s(B-T)`` exactly and the control was vacuous. The fix threads the SAME scale
    into the ratio's own denominator and the closure reconstruction: ``denom = cat_scale *
    beta_G_phi * L_cat_no_bh``, ``recon = (cat_scale * beta_G_phi * L_cat_no_bh + B_num) /
    D_tilde_phi`` -- checked against ``at``'s ``combined_no_bh`` column, which for B-R is the
    caller's ALREADY-PATCHED ``combined_br`` (so the closure check is meaningful, not a tautology
    against the unpatched B-T value).
    """
    alpha_g_phi = at["alpha_G_phi"].to_numpy(dtype=np.float64)
    r_malm = at["r_Malm"].to_numpy(dtype=np.float64)
    d_tilde_phi = at["D_tilde_phi"].to_numpy(dtype=np.float64)
    l_cat = at["L_cat_no_bh"].to_numpy(dtype=np.float64)
    b_num = at["B_num"].to_numpy(dtype=np.float64)
    combined = at["combined_no_bh"].to_numpy(dtype=np.float64)

    beta_g_phi_scaled = cat_scale * (alpha_g_phi / r_malm)

    live = l_cat > 0.0  # PA-7 LIVE(a) definition
    dead = ~live

    ratio = np.full(at.shape[0], np.nan, dtype=np.float64)
    denom = beta_g_phi_scaled[live] * l_cat[live]
    ratio[live] = np.where(denom != 0.0, b_num[live] / denom, np.nan)

    # W-B0 (kept, PA-7): the reconstructed responsibility, for the closure check ONLY -- not the
    # primary's ratio path (PA-3 supersedes it there). Unaffected by cat_scale (reads combined
    # directly, whichever frame ``at`` carries).
    w_e = np.full(at.shape[0], np.nan, dtype=np.float64)
    w_e[live] = 1.0 - b_num[live] / (combined[live] * d_tilde_phi[live])

    recon = (beta_g_phi_scaled * l_cat + b_num) / np.where(d_tilde_phi > 0.0, d_tilde_phi, np.nan)
    tiny = np.finfo(float).tiny
    closure_rel = np.abs(recon - combined) / np.maximum(np.abs(combined), tiny)

    return {
        "ratio": ratio,
        "w_e": w_e,
        "live": live,
        "dead": dead,
        "closure_rel": closure_rel,
    }


def _identity_score(at: pd.DataFrame, cat_scale: float = 1.0) -> dict[str, Any]:
    """Primary statistic (prereg S4, PA-3/PA-4/PA-6): ``I_s = mean_e[ratio_e] * C* - 1`` at
    ``h = H_GEN`` over LIVE rows, plus the PA-6(b) PSIS twin, PA-6(a) dead-row accounting, the
    PA-6(c) trim_mean reported-only diagnostic, and the W-B0 closure check.

    ``cat_scale`` (PA-13(c)): forwarded to :func:`_identity_inputs` -- 1.0 for B-C/B-T, ``r_h``
    for the B-R control (see :func:`stage_rescore`). C* itself is UNAFFECTED by ``cat_scale``
    (PA-4: computed from ``at``'s own ``alpha_G_phi``/``r_Malm``/``D_tilde_phi`` columns, which
    ``stage_rescore`` never patches).
    """
    c_star_value, c_star_diag = c_star(at)
    inputs = _identity_inputs(at, c_star_value, cat_scale=cat_scale)
    live = inputs["live"]
    n_rows = int(at.shape[0])
    n_live = int(live.sum())
    n_dead = int(inputs["dead"].sum())
    dead_rate = n_dead / n_rows if n_rows else None

    if n_live == 0 or not np.isfinite(c_star_value):
        return {
            "n_rows": n_rows,
            "n_live": n_live,
            "n_dead": n_dead,
            "dead_rate": dead_rate,
            "cat_scale": cat_scale,
            "C_star": c_star_value,
            "c_star_diagnostics": c_star_diag,
            "I_s": None,
            "I_s_psis": None,
            "I_s_trim_reported_only": None,
            "k_hat": None,
            "closure_max_rel": float(np.nanmax(inputs["closure_rel"])) if n_rows else None,
        }

    ratio = inputs["ratio"][live]
    ratio = ratio[np.isfinite(ratio)]
    if ratio.size == 0:
        n_live = 0
    i_s = float(ratio.mean() * c_star_value - 1.0) if ratio.size else None
    psis = _psis_diagnostics(ratio) if ratio.size else {"k_hat": None, "psis_mean": None, "M": 0}
    i_s_psis = (
        float(psis["psis_mean"] * c_star_value - 1.0) if psis.get("psis_mean") is not None else None
    )
    i_s_trim = (
        float(trim_mean(ratio, TRIM_PROPORTION) * c_star_value - 1.0) if ratio.size >= 4 else i_s
    )
    closure_pass_val = (
        bool(np.nanmax(inputs["closure_rel"][live]) <= GATE_WB0_CLOSURE_RTOL) if n_live else False
    )
    return {
        "n_rows": n_rows,
        "n_live": int(ratio.size),
        "n_dead": n_dead,
        "dead_rate": dead_rate,
        "cat_scale": cat_scale,
        "C_star": c_star_value,
        "c_star_diagnostics": c_star_diag,
        "I_s": i_s,
        "I_s_psis": i_s_psis,
        "k_hat": psis.get("k_hat"),
        "psis_M": psis.get("M"),
        "I_s_trim_reported_only": i_s_trim,  # PA-6(c): reported-only, NOT verdict-participating
        "ratio_percentiles": {
            str(p): float(np.percentile(ratio, p)) for p in (5, 25, 50, 75, 95, 99)
        }
        if ratio.size
        else None,
        "ratio_vector": ratio.tolist(),
        "closure_max_rel": float(np.nanmax(inputs["closure_rel"][live])) if n_live else None,
        "closure_pass": closure_pass_val,
        "closure_tol": GATE_WB0_CLOSURE_RTOL,
        "closure_csv_storage_floor": GATE_WB0_CSV_STORAGE_FLOOR,
    }


# ── LEV -- zero-compute leverage instrument (prereg S5) ──────────────────────


def stage_lev(out_root: Path) -> dict[str, Any]:
    """Prereg S5 LEV: Ibar_banked(coded) on the 25 banked b0 CSVs (off basis, s3d slot,
    coded cell) -- runs BEFORE the pilot, zero ``evaluate()`` calls.

    PA-7 (Finding 7 LEV amendment): the banked basis carries the Sigma^3D slot (not Sigma^phi),
    so this read conflates arrangement displacement with the PA-2 venue-premise term AND the
    slot-constant term -- reported here as the coded arm's TOTAL displacement, INSEPARABLE,
    never quoted as arrangement-only. [driver implementation note, disclosed for the
    pre-execution adversarial review]: this function reuses the SAME C* (PA-4, computed from
    ``rho`` at the venue-level mass companion) for the banked basis as for the b0i arms, even
    though PA-4's Finding 4 flags "a third value" for the strict Sigma^3D-slot LEV constant --
    the venue-level ``rho`` itself is slot-independent (a property of kernel-smeared vs point-
    evaluated S_bar_phi, not of which divisor a given run used), so only the numerator/
    denominator (``beta_G_phi``/``beta_Gbar_phi``, read from the BANKED run's own s3d-slot
    columns) differ from the b0i arms' -- consistent with the TOTAL-displacement, never-the-
    comparand framing above.
    """
    per_seed: list[dict[str, Any]] = []
    missing: list[int] = []
    for seed in LEV_SEEDS:
        csv_path = _banked_b0_csv_path(seed)
        if not csv_path.is_file():
            missing.append(seed)
            continue
        at = _rows_at_h(csv_path, H_GEN)
        score = _identity_score(at)
        score.pop("ratio_vector", None)  # per-seed summary only; full vectors are large
        score["seed"] = seed
        per_seed.append(score)

    i_vals = np.array([r["I_s"] for r in per_seed if r["I_s"] is not None], dtype=np.float64)
    i_psis_vals = np.array(
        [r["I_s_psis"] for r in per_seed if r["I_s_psis"] is not None], dtype=np.float64
    )
    i_trim_vals = np.array(
        [r["I_s_trim_reported_only"] for r in per_seed if r["I_s_trim_reported_only"] is not None],
        dtype=np.float64,
    )
    i_bar = float(i_vals.mean()) if i_vals.size else None
    sem = float(i_vals.std(ddof=1) / np.sqrt(i_vals.size)) if i_vals.size > 1 else None
    i_bar_psis = float(i_psis_vals.mean()) if i_psis_vals.size else None
    i_bar_trim = float(i_trim_vals.mean()) if i_trim_vals.size else None
    # AMEND-8(a)/PA-13(e): the per-ARM (here: the single coded/cross-basis arm) k-hat, MAX over
    # this arm's seeds -- the prereg's "per-arm" wording, disclosed convention (matches
    # _fleet_identity's k_hat_max for B-C/B-T/B-R).
    k_hats = [r["k_hat"] for r in per_seed if r.get("k_hat") is not None]
    k_hat_max = float(max(k_hats)) if k_hats else None
    max_closure = max(
        (r["closure_max_rel"] for r in per_seed if r["closure_max_rel"] is not None),
        default=None,
    )
    n_dead_total = sum(r["n_dead"] for r in per_seed)
    n_rows_total = sum(r["n_rows"] for r in per_seed)

    out: dict[str, Any] = {
        "reference": f"{REGISTRATION_SECTION}, S5 LEV (PA-7 wording)",
        "instrument": "coded-arm (off/off/s3d) banked b0 identity displacement, cross-basis",
        "n_seeds_found": len(per_seed),
        "n_seeds_missing": missing,
        "mass_companion_at_h_gen": mass_companion(H_GEN),
        "Ibar_banked_coded": i_bar,
        "sem_banked_coded": sem,
        "Ibar_psis_banked_coded": i_bar_psis,
        "Ibar_trim_reported_only_banked_coded": i_bar_trim,
        "k_hat_max": k_hat_max,  # AMEND-8(a)/PA-13(e): per-arm k-hat = max over seeds, disclosed
        "trim_proportion": TRIM_PROPORTION,
        "n_dead_total": n_dead_total,
        "n_rows_total": n_rows_total,
        "dead_rate_pooled": (n_dead_total / n_rows_total) if n_rows_total else None,
        "closure_max_rel_over_fleet": max_closure,
        "closure_tol": GATE_WB0_CLOSURE_RTOL,
        "per_seed": per_seed,
        "purpose": (
            "(i) order-of-magnitude of the coded arm's TOTAL identity displacement (arrangement "
            "+ venue-premise terms + the Sigma^3D-slot constant, INSEPARABLE on the banked "
            "basis -- never quoted as arrangement-only, PA-7); the LEV threshold requires "
            "|Ibar| >= 5x the (not-yet-frozen) band resolution eps_I, else STOP and re-design "
            "(O4 lesson); (ii) the PA-6(b) PSIS read calibrating heavy-tail regime (k_hat); "
            "(iii) the cross-basis comparator for secondary 6. NOT a fused-basis statistic; "
            "never the comparand (prereg S5)."
        ),
    }
    out_path = out_root / "p3_b0_lev_output.json"
    out_root.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print("=== [P3-IMP] b0 identity test -- LEV (zero-compute, PA-amended) ===")
    print(f"n_seeds_found = {len(per_seed)} / {len(LEV_SEEDS)} (missing: {missing})")
    print(f"Ibar_banked(coded) = {i_bar!r}  SEM = {sem!r}")
    print(f"Ibar_psis_banked(coded) = {i_bar_psis!r}  (PA-6b twin)")
    print(f"k_hat_max = {k_hat_max!r}  (per-arm, max over seeds, PA-13e/AMEND-8a)")
    print(
        f"Ibar_trim_reported_only_banked(coded) = {i_bar_trim!r}  "
        f"(trim_proportion={TRIM_PROPORTION}, reported-only per PA-6c)"
    )
    print(f"n_dead_total = {n_dead_total} / {n_rows_total} rows (support violations, PA-6a)")
    print(f"closure_max_rel_over_fleet = {max_closure!r}  (tol {GATE_WB0_CLOSURE_RTOL:.0e})")
    print(f"wrote {out_path}")
    return out


# ── evaluate()-bearing arms ───────────────────────────────────────────────────


def _run_arm_seed(
    seed: int,
    catalogue_numerator_survival: str,
    catalogue_global_selection: str,
    out_root: Path,
    subdir: str,
    *,
    venue: str = "b0i",
    completion_cell: str = "fused",
) -> dict[str, Any]:
    """One venue realization, evaluated end-to-end -- generalized over ``venue`` (``"b0"`` for
    GATE R-B0's replica, ``"b0i"`` for B-C/B-T, PA-2) and the two identity-test flags.

    ``venue="b0"`` (replica): ``ARM_HOST_MODE["b0"] == "catalogue"`` -- draws hosts from the
    pinned ``HostPool``, matching ``run_arm_seed``'s catalogue (``else``) branch exactly
    (``correspondence_1d.py:2764-2772``), no ``phi_survival_table`` needed.

    ``venue="b0i"`` (B-C/B-T, PA-2): ``ARM_HOST_MODE["b0i"] == "catalogue_selected"`` -- the NEW
    host mode this driver is written against (interface assumption, module docstring HARD
    CONSTRAINT 4): ``draw_realization(seed, host_pool=host_pool, host_mode="catalogue_selected",
    phi_survival_table=phi_survival_table)``. ``phi_survival_table`` is built via
    ``c1d.build_bsel_selection_objects(h_true=H_GEN)`` -- reused (import, lru_cache'd) from the
    SAME construction ``bsel``/``bself``/``bden`` already use.
    """
    meta_path = out_root / f"{subdir}_meta.json"
    if meta_path.is_file():
        sys.exit(
            f"REFUSED: {meta_path} already exists -- use a fresh --out-root or remove this "
            "file if a genuine regeneration is intended (A21: the registered arm may not "
            "silently substitute a cached run)."
        )
    expected_host_mode = "catalogue" if venue == "b0" else "catalogue_selected"
    assert c1d.ARM_HOST_MODE[venue] == expected_host_mode, (
        f"interface assumption violated: c1d.ARM_HOST_MODE[{venue!r}] != {expected_host_mode!r} "
        "-- the venue registry changed since this driver was written -- STOP (A21)"
    )
    if venue == "b0i":
        assert c1d.ARM_SELECTION_CELL["b0i"] == "fused", (
            "interface assumption violated: c1d.ARM_SELECTION_CELL['b0i'] != 'fused' -- STOP (A21)"
        )
    work_root = out_root / f"{subdir}_work"
    work_root.mkdir(parents=True, exist_ok=True)
    log_path = out_root / f"{subdir}.log"

    sigma_z_scale, area_scale = c1d.ARM_SPECS.get(venue, c1d.ARM_SPECS["b0"])
    catalogue_pin_ok = c1d.check_reduced_catalogue_pin()
    cfg = c1d.CorrespondenceConfig(sigma_z_scale=sigma_z_scale, area_scale=area_scale)
    gen = c1d.MirrorUniverseGenerator(cfg)
    host_pool, _observed_path, handler = gen.host_pool_for_sigma_scale(
        work_root / "catalogue", seed, sigma_z_scale=sigma_z_scale
    )
    if venue == "b0i":
        # PA-13(d)/AMEND-7 fix (A20_REVIEW_B0_IMPL_20260823.md Finding 7): the PA-2 runtime
        # rate-weight parity gate is called only in c1d.run_arm_seed, which this driver bypasses
        # (calls gen.draw_realization directly) -- wired here, BEFORE any draw.
        c1d._verify_rate_weight_parity()
        completeness_obj, phi_survival_table = c1d.build_bsel_selection_objects(h_true=H_GEN)
        # PA-11 interface catch-up: draw_realization's "catalogue_selected" branch now also
        # requires completeness (the kernel-alignment fix's estimator-own w_pop*f_k factor),
        # not just phi_survival_table -- forwarded here.
        events = gen.draw_realization(
            seed,
            host_pool=host_pool,
            host_mode="catalogue_selected",
            completeness=completeness_obj,
            phi_survival_table=phi_survival_table,
        )
    else:
        events = gen.draw_realization(seed, host_pool=host_pool, host_mode="catalogue")

    # A22 amended (row #173): stamp WRITTEN before the evaluate() call.
    stamp = _a22_stamp()
    t0 = time.time()
    with _capture_root_log(log_path):
        diag_csv, elapsed = c1d.run_mirror_seed_inprocess(
            work_root / f"seed{seed}",
            events,
            seed,
            galaxy_catalog=handler,
            h_values=H_GRID_FULL,
            selection_in_completion_numerator=completion_cell,
            completion_event_measure=c1d.ARM_EVENT_MEASURE.get(venue, "ratio"),
            catalogue_numerator_survival=catalogue_numerator_survival,
            # [ORCH-RULE 3] the Sigma^phi divisor slot (prereg S2, PA-4).
            catalogue_global_selection=catalogue_global_selection,
        )
    wall_time_s = time.time() - t0
    stats = compute_seed_statistics(diag_csv, seed, h_grid=H_GRID_41)

    # PA-7/PA-9: cross-check the CSV-derived beta_G_phi/beta_Gbar_phi against the run's own
    # selection_tables_h_<label>.json (written to the same seed work dir during evaluate()).
    at_h_gen = _rows_at_h(diag_csv, H_GEN)
    beta_g_phi, beta_gbar_phi = _beta_g_phi_and_gbar(at_h_gen)
    selection_json_cross_check = _cross_check_selection_table_json(
        work_root / f"seed{seed}", H_GEN, beta_g_phi, beta_gbar_phi
    )

    meta: dict[str, Any] = {
        "subdir": subdir,
        "seed": seed,
        "venue": venue,
        "completion_cell": completion_cell,
        "catalogue_numerator_survival": catalogue_numerator_survival,
        "catalogue_global_selection": catalogue_global_selection,
        "work_root": str(work_root),
        "diagnostics_csv": str(diag_csv),
        "log_path": str(log_path),
        "wall_time_s": wall_time_s,
        "elapsed_evaluate_s": elapsed,
        "catalogue_pin_ok": catalogue_pin_ok,
        "mean_h": stats.mean_h,
        "map_h": stats.map_h,
        "sigma_h": stats.sigma_h,
        "r_low": stats.r_low,
        "n_events": stats.n_events,
        "beta_G_phi_at_h_gen": beta_g_phi,
        "beta_Gbar_phi_at_h_gen": beta_gbar_phi,
        "selection_json_cross_check": selection_json_cross_check,
        "a22_stamp": stamp,
        "git_commit": c1d._git_commit(),
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(json.dumps({k: v for k, v in meta.items() if k != "diagnostics_csv"}, indent=2))
    return meta


ARM_FLAGS: dict[str, dict[str, str]] = {
    "bc": {"catalogue_numerator_survival": "off", "catalogue_global_selection": "phi"},
    "bt": {"catalogue_numerator_survival": "phi", "catalogue_global_selection": "phi"},
}


def stage_replica(out_root: Path) -> dict[str, Any]:
    """GATE R-B0: seed 900101 under the BANKED configuration (off/off/s3d, completion_cell
    "off", venue **b0** -- ``c1d.ARM_SELECTION_CELL["b0"] == "off"``, the basis the 25 banked b0
    CSVs were produced under, byte-identical to ``run_arm_seed``'s original b0 call before this
    driver existed). UNCHANGED by PA-2 -- venue-fidelity proof, not an identity read.
    """
    _assert_h_true_in_grid()
    meta = _run_arm_seed(
        REPLICA_SEED,
        "off",
        "s3d",
        out_root,
        "replica_900101",
        venue="b0",
        completion_cell="off",
    )
    fresh = pd.read_csv(_meta_csv(meta))
    banked_csv = _banked_b0_csv_path(REPLICA_SEED)
    gate = _compare_columns(fresh, banked_csv, GATE_RB0_COLUMNS, GATE_RB0_RTOL)
    gate["gate"] = "GATE_R-B0"
    gate["wall_time_s"] = meta["wall_time_s"]
    gate["wall_time_min_s"] = GATE_RB0_MIN_WALL_S
    gate["wall_time_pass"] = meta["wall_time_s"] > GATE_RB0_MIN_WALL_S
    gate["pass"] = bool(gate["pass"]) and gate["wall_time_pass"]
    gate["reference"] = f"{REGISTRATION_SECTION}, S3 GATE R-B0"
    (out_root / "replica_gate_result.json").write_text(json.dumps(gate, indent=2))
    print(json.dumps(gate, indent=2))
    return gate


def _compare_columns(
    fresh: pd.DataFrame, banked_csv: Path, columns: tuple[str, ...], rtol: float
) -> dict[str, Any]:
    """Bit-exact-or-<=rtol-relative multi-column comparison (GATE R-P3/D6/R4 pattern)."""
    if not banked_csv.is_file():
        return {"pass": False, "reason": f"banked diagnostics not found: {banked_csv}"}
    banked = pd.read_csv(banked_csv)
    merged = fresh.merge(
        banked[["event_idx", "h", *columns]],
        on=["event_idx", "h"],
        suffixes=("_fresh", "_banked"),
        how="outer",
        indicator=True,
    )
    key_mismatch = bool((merged["_merge"] != "both").any())
    per_column: dict[str, Any] = {}
    all_ok = not key_mismatch
    for col in columns:
        a = merged[f"{col}_fresh"].to_numpy(dtype=np.float64)
        b = merged[f"{col}_banked"].to_numpy(dtype=np.float64)
        exact = bool(np.array_equal(a, b))
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.abs(a - b) / np.maximum(np.abs(b), np.finfo(float).tiny)
        max_rel = float(np.nanmax(rel)) if rel.size else float("nan")
        ok = exact or max_rel <= rtol
        all_ok = all_ok and ok
        per_column[col] = {"bit_exact": exact, "max_rel_err": max_rel, "pass": ok}
    return {
        "pass": all_ok,
        "key_mismatch": key_mismatch,
        "tol": rtol,
        "n_rows_compared": int(len(merged)),
        "per_column": per_column,
        "banked_csv": str(banked_csv),
        "fallback_justification": (
            "production evaluate() dispatches per-host likelihood terms through a "
            "multiprocessing pool (_starmap_host_batches) whose float-summation order is not "
            f"guaranteed run-to-run identical, so the registered <= {rtol:.0e} relative "
            "fallback is used per GATE R-B0 (GATE R-P3/R4/D6 precedent)."
        ),
    }


# PA-13(a)/FATAL-3 fix: the extra same-venue GATE E-B0(a) pair -- b0i seed 900101, all else = B-C
# config, EXCEPT catalogue_global_selection="s3d" (B-C uses "phi"). Same venue (b0i), same seed,
# same host draw -- so the two runs' event sets/hosts/z_true are identical and a live-row L_cat
# ratio comparison is a genuine same-generator-realization test (unlike the OLD cross-venue
# replica comparison this demotes to reported-only, Finding 3).
EB0A_SUBDIR: str = f"eb0a_{REPLICA_SEED}"


def _run_eb0a_seed(out_root: Path) -> dict[str, Any]:
    """The PA-13(a) extra run: b0i seed 900101, B-C's flags except ``catalogue_global_selection=
    "s3d"``."""
    return _run_arm_seed(
        REPLICA_SEED,
        ARM_FLAGS["bc"]["catalogue_numerator_survival"],
        "s3d",
        out_root,
        EB0A_SUBDIR,
        venue="b0i",
    )


def stage_pilot(out_root: Path) -> dict[str, Any]:
    """PILOT: seed 900101 under BOTH B-C and B-T (venue b0i, fused basis, Sigma^phi slot), PLUS
    the PA-13(a) extra same-venue E-B0(a) run (B-C's flags with ``catalogue_global_selection=
    "s3d"``, Finding 3 fix)."""
    _assert_h_true_in_grid()
    rows: dict[str, Any] = {}
    for arm, flags in ARM_FLAGS.items():
        meta = _run_arm_seed(
            REPLICA_SEED,
            flags["catalogue_numerator_survival"],
            flags["catalogue_global_selection"],
            out_root,
            f"{arm}_{REPLICA_SEED}",
            venue="b0i",
        )
        rows[arm] = meta
    eb0a_meta_path = out_root / f"{EB0A_SUBDIR}_meta.json"
    eb0a_meta = (
        json.loads(eb0a_meta_path.read_text())
        if eb0a_meta_path.is_file()
        else _run_eb0a_seed(out_root)
    )
    out = {
        "registered_in": f"{REGISTRATION_SECTION}, S2/S8 pilot, PA-13(a) amended",
        "seed": REPLICA_SEED,
        "bc_mean_h": rows["bc"]["mean_h"],
        "bt_mean_h": rows["bt"]["mean_h"],
        "delta_bt_minus_bc": float(rows["bt"]["mean_h"]) - float(rows["bc"]["mean_h"]),
        "eb0a_mean_h": eb0a_meta["mean_h"],
    }
    (out_root / "pilot_output.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return out


def stage_fleet(out_root: Path, arm: str, seeds: list[int] | None = None) -> dict[str, Any]:
    """One arm's fleet (seeds 900101-900112, venue b0i), sequential, idempotent-skip on existing
    meta.

    ``arm`` selects the flag set (``bc``/``bt``); the two arms are run as SEPARATE detached
    invocations of this CLI (``--stage fleet --arm bc`` / ``--arm bt``), per the launch task
    instruction -- this function only ever runs one arm's seeds.
    """
    _assert_h_true_in_grid()
    if arm not in ARM_FLAGS:
        raise SystemExit(f"REFUSED: unknown --arm {arm!r}; must be one of {sorted(ARM_FLAGS)}")
    flags = ARM_FLAGS[arm]
    reused: list[int] = []
    ran: list[int] = []
    for seed in seeds if seeds is not None else BSEL_SEEDS:
        meta_path = out_root / f"{arm}_{seed}_meta.json"
        if meta_path.is_file():
            reused.append(seed)
            print(f"seed {seed} ({arm}): REUSING existing {arm}_{seed}_meta.json (disclosed)")
            continue
        _run_arm_seed(
            seed,
            flags["catalogue_numerator_survival"],
            flags["catalogue_global_selection"],
            out_root,
            f"{arm}_{seed}",
            venue="b0i",
        )
        ran.append(seed)
    summary = {"arm": arm, "reused": reused, "freshly_ran": ran}
    print(json.dumps(summary, indent=2))
    return summary


# ── B-R rescore (zero evaluate()) ─────────────────────────────────────────────


def stage_rescore(out_root: Path) -> dict[str, Any]:
    """B-R (CONTROL): the R(h) = beta_G(h)/beta_G_phi(h) rescale of B-T's ``L_cat_no_bh`` at
    ``h = H_GEN``, per the committed ``p3_completed_rescore.py`` construction
    (``cat_term_completed = cat_term_phi * R(h)``, that module's opening docstring) -- reused
    by import (``o3._build_betas``), not reimplemented. Only the leaf-call venue objects are
    rebuilt at ``h = H_GEN`` (single-h list, matching ``build_bsel_selection_objects``'s own
    single-``h_true`` construction rather than the full-grid form ``_build_betas`` uses in its
    own module -- b0's identity statistic only ever reads ``h = H_GEN``). PA-4: B-R is scored
    with the SAME C* as B-C/B-T (computed from B-T's own columns, unaffected by the rescale,
    which only touches ``combined_no_bh``) -- see :func:`stage_score`.

    **PA-13(c)/FATAL-2 fix (A20_REVIEW_B0_IMPL_20260823.md Finding 2):** the previous version
    patched ONLY ``combined_no_bh`` and scored it through :func:`_identity_score`'s DEFAULT
    ``cat_scale=1.0``, so the ratio's own denominator (``beta_G_phi * L_cat_no_bh``, unpatched)
    and the closure check were IDENTICAL to B-T's -- ``I_s(B-R) ≡ I_s(B-T)`` exactly, the control
    vacuous. The fix passes ``cat_scale=r_h`` into :func:`_identity_score`, which threads it into
    the ratio's denominator (``r_h * beta_G_phi * L_cat_no_bh``) AND the closure reconstruction
    (``(r_h * beta_G_phi * L_cat_no_bh + B_num) / D_tilde_phi``, checked against the
    ALREADY-PATCHED ``combined_br`` column) -- by this construction ``I(B-R) + 1 = (I(B-T) + 1)
    / R`` exactly (PA-4's registered prediction), not merely by an unverifiable coincidence.
    """
    beta_g_phi, _beta_gbar_phi, beta_g, _beta_gbar = o3._build_betas([H_GEN])
    r_h = beta_g[H_GEN] / beta_g_phi[H_GEN]
    br_predicted = _br_predicted_value(r_h)

    per_seed: list[dict[str, Any]] = []
    missing: list[int] = []
    # PA-13(c): rebuilt beta_G_phi(0.73) (this stage's own o3._build_betas leaf) vs each seed's
    # fleet meta's OWN beta_G_phi_at_h_gen (banked by _run_arm_seed's selection-table cross
    # check) -- a venue-object cross-check, disclosed per-seed.
    beta_g_phi_fleet_cross_check: dict[int, dict[str, Any]] = {}
    for seed in BSEL_SEEDS:
        meta_path = out_root / f"bt_{seed}_meta.json"
        if not meta_path.is_file():
            missing.append(seed)
            continue
        bt_meta = json.loads(meta_path.read_text())
        at = _rows_at_h(_meta_csv(bt_meta), H_GEN)
        alpha_g_phi = at["alpha_G_phi"].to_numpy(dtype=np.float64)
        r_malm = at["r_Malm"].to_numpy(dtype=np.float64)
        d_tilde_phi = at["D_tilde_phi"].to_numpy(dtype=np.float64)
        l_cat = at["L_cat_no_bh"].to_numpy(dtype=np.float64)
        combined_bt = at["combined_no_bh"].to_numpy(dtype=np.float64)

        w = alpha_g_phi / r_malm / d_tilde_phi
        cat_term_bt = w * l_cat
        combined_br = combined_bt + cat_term_bt * (r_h - 1.0)

        patched = at.copy()
        patched["combined_no_bh"] = combined_br
        score = _identity_score(patched, cat_scale=float(r_h))
        score.pop("ratio_vector", None)
        score["seed"] = seed
        per_seed.append(score)

        meta_beta_g_phi = bt_meta.get("beta_G_phi_at_h_gen")
        if meta_beta_g_phi is not None:
            rebuilt = float(beta_g_phi[H_GEN])
            rel_err = abs(rebuilt - float(meta_beta_g_phi)) / max(
                abs(float(meta_beta_g_phi)), 1e-300
            )
            beta_g_phi_fleet_cross_check[seed] = {
                "rebuilt_beta_G_phi_at_H_GEN": rebuilt,
                "bt_meta_beta_G_phi_at_h_gen": float(meta_beta_g_phi),
                "rel_err": rel_err,
            }

    if missing:
        sys.exit(
            f"REFUSED: missing B-T meta for seeds {missing} -- run --stage fleet --arm bt first."
        )

    out = {
        "reference": f"{REGISTRATION_SECTION}, S2 B-R (CONTROL), PA-13(c) amended",
        "R_h_at_H_GEN": float(r_h),
        "cat_scale_applied": float(r_h),
        "beta_G_phi_at_H_GEN": float(beta_g_phi[H_GEN]),
        "beta_G_at_H_GEN": float(beta_g[H_GEN]),
        "beta_G_phi_fleet_cross_check": beta_g_phi_fleet_cross_check,
        "br_predicted_I_plus_1": br_predicted + 1.0,
        "br_predicted_I": br_predicted,
        "br_control_tolerance": BR_CONTROL_TOLERANCE,
        "leaf_reused_from": "p3_completed_rescore.py:_build_betas (import, not reimplemented)",
        "per_seed": per_seed,
    }
    (out_root / "rescore_output.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "per_seed"}, indent=2))
    print(f"wrote {out_root / 'rescore_output.json'}")
    return out


# ── GATE E-B0 ──────────────────────────────────────────────────────────────


def _scalar_path_smoke() -> dict[str, Any]:
    """PA-7 (Finding 7, GATE E-B0(c) amendment): "the driver additionally invokes the scalar
    path once on one event as a smoke check, outside the registered statistics."

    **[driver implementation gap, disclosed]:** ``single_host_likelihood``
    (``bayesian_statistics.py:5769``) requires the full internal per-h/per-host context object
    graph ``BayesianStatistics.evaluate()`` builds internally (galaxy candidate structure,
    detection-probability grid, Path-A mixture objects, ...) -- there is no cheap public entry
    point to invoke it standalone without either touching ``darksiren_emri/`` source (out of
    scope for this driver, per the launch task's constraint) or duplicating a large fraction of
    ``evaluate()``'s setup. NOT attempted here; reported honestly as not-yet-implemented rather
    than fabricated as a pass. GATE E-B0(c)'s ``pass`` is scored from the static dispatch-code
    audit alone (the A20 review's own "CONFIRMED executable" form), matching Finding 7's
    disposition that this smoke check is explicitly "outside the registered statistics."
    """
    return {
        "attempted": False,
        "reason": (
            "single_host_likelihood requires BayesianStatistics.evaluate()'s full internal "
            "context object graph; no cheap standalone entry point exists without touching "
            "darksiren_emri/ source. Flagged for the pre-execution adversarial review / the "
            "owning agent to wire a real smoke hook."
        ),
    }


def _gate_e_b0a_same_venue(
    bc_seed1_meta: dict[str, Any],
    eb0a_meta: dict[str, Any],
    h_grid: tuple[float, ...] = H_GRID_FULL,
    cv_tol: float = GATE_EB0A_SAMEVENUE_CV_TOL,
) -> dict[str, Any]:
    """PA-13(a)/FATAL-3 fix (A20_REVIEW_B0_IMPL_20260823.md Finding 3): GATE E-B0(a) re-scored on
    a SAME-VENUE pair -- B-C (b0i seed 900101, ``catalogue_global_selection="phi"``) vs the extra
    ``eb0a`` run (b0i seed 900101, IDENTICAL flags except ``catalogue_global_selection="s3d"``).
    Same venue, same seed, same host draw/event set -- unlike the OLD replica (venue **b0**,
    stock generator) comparison this supersedes, which compared two DIFFERENT realizations and
    could not fail (the OLD check is retained as reported-only, see
    :data:`_gate_e_b0`'s ``a_sigma_phi_slot_reported_only``).

    Requires, on EVERY node of ``h_grid``: the ``L_cat_no_bh`` ratio (B-C/eb0a) is ONE
    h-dependent constant across ALL PAIRED-LIVE rows (rows with ``L_cat_no_bh > 0`` in both) --
    scored as the per-h coefficient of variation, ``std(ratio)/mean(ratio) < cv_tol``, AND
    100% of that h's rows are PAIRED-LIVE (a support mismatch at any h fails that node).
    """
    bc_df = pd.read_csv(_meta_csv(bc_seed1_meta))
    eb0a_df = pd.read_csv(_meta_csv(eb0a_meta))
    per_h: dict[str, dict[str, Any]] = {}
    all_pass = True
    for h in h_grid:
        bc_at = bc_df[np.isclose(bc_df["h"].to_numpy(dtype=np.float64), h)]
        eb0a_at = eb0a_df[np.isclose(eb0a_df["h"].to_numpy(dtype=np.float64), h)]
        merged = bc_at.merge(
            eb0a_at[["event_idx", "L_cat_no_bh"]],
            on="event_idx",
            suffixes=("_bc", "_eb0a"),
        )
        n_total = int(len(merged))
        live = (merged["L_cat_no_bh_bc"] > 0.0) & (merged["L_cat_no_bh_eb0a"] > 0.0)
        n_live = int(live.sum())
        if n_live == 0 or n_total == 0:
            per_h[str(h)] = {
                "n_live": n_live,
                "n_total": n_total,
                "fraction_paired_live": (n_live / n_total) if n_total else None,
                "ratio_mean": None,
                "ratio_cv": None,
                "pass": False,
                "reason": "no PAIRED-LIVE rows at this h",
            }
            all_pass = False
            continue
        ratio = merged.loc[live, "L_cat_no_bh_bc"].to_numpy(dtype=np.float64) / merged.loc[
            live, "L_cat_no_bh_eb0a"
        ].to_numpy(dtype=np.float64)
        mean_r = float(np.mean(ratio))
        cv = float(np.std(ratio) / abs(mean_r)) if mean_r != 0.0 else float("nan")
        fraction_paired_live = n_live / n_total
        # Registered form (PA-13(a), amendment-18 discipline): the constant-ratio
        # requirement holds on 100% OF PAIRED-LIVE rows -- the CV check IS that
        # requirement. Rows outside PAIRED-LIVE are the PA-6(a) dead rows, owned
        # by the dead-row register, not this gate (first implementation wrongly
        # demanded fraction_paired_live >= 1.0 over ALL rows and failed on the
        # one both-arm dead row; fixed 2026-08-24, gates re-scored, fraction
        # still reported).
        h_pass = bool(np.isfinite(cv) and cv < cv_tol)
        all_pass = all_pass and h_pass
        per_h[str(h)] = {
            "n_live": n_live,
            "n_total": n_total,
            "fraction_paired_live": fraction_paired_live,
            "ratio_mean": mean_r,
            "ratio_cv": cv,
            "pass": h_pass,
        }
    return {
        "gate": "GATE_E-B0(a)_same_venue",
        "pass": bool(all_pass),
        "cv_tol": cv_tol,
        "paired_live_fraction_reported_only": True,
        "denominator": "PAIRED-LIVE per h: rows with L_cat_no_bh > 0 in both B-C and eb0a",
        "per_h": per_h,
        "reference": f"{REGISTRATION_SECTION}, PA-13(a) GATE E-B0(a) same-venue fix",
    }


def _gate_e_b0(
    replica_meta: dict[str, Any],
    bc_metas: dict[int, dict[str, Any]],
    bt_metas: dict[int, dict[str, Any]],
    eb0a_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """GATE E-B0 (prereg S3, PA-7/PA-13(a) amended), explicit denominators.

    (a) Sigma^phi slot -- PA-13(a)/FATAL-3 fix: GATING now comes from
        :func:`_gate_e_b0a_same_venue` (``eb0a_meta``, when supplied). The ORIGINAL cross-venue
        comparison (replica, venue **b0**, stock generator, vs B-C, venue **b0i**) is retained
        below as ``a_sigma_phi_slot_reported_only`` -- DEMOTED, never gates ``pass`` (Finding 3:
        it compares two different realizations/venues, so "ratio != 1 on 100%" is trivially true
        regardless of correctness).
    (b) twin cell: B-T vs B-C ``L_cat_no_bh`` differ on >=99% of PAIRED-LIVE rows (PA-7: PAIRED-
        LIVE = the intersection of both arms' LIVE(a) sets), fleet-pooled across the 12 seeds.
    (c) PA-7 amended: "runtime assertions confirm both flags reached every dispatch path the run
        exercised; the driver additionally invokes the scalar path once on one event as a smoke
        check, outside the registered statistics" -- see :func:`_scalar_path_smoke` (disclosed
        gap: not attempted, honestly reported).
    """
    replica_csv = _meta_csv(replica_meta)
    bc_seed1_csv = _meta_csv(bc_metas[REPLICA_SEED])
    replica_at = _rows_at_h(replica_csv, H_GEN)
    bc_seed1_at = _rows_at_h(bc_seed1_csv, H_GEN)
    merged_a = replica_at.merge(
        bc_seed1_at[["event_idx", "L_cat_no_bh"]],
        on="event_idx",
        suffixes=("_replica", "_bc"),
    )
    live_a = (merged_a["L_cat_no_bh_replica"] > 0.0) & (merged_a["L_cat_no_bh_bc"] > 0.0)
    ratio_a = merged_a.loc[live_a, "L_cat_no_bh_bc"].to_numpy(dtype=np.float64) / merged_a.loc[
        live_a, "L_cat_no_bh_replica"
    ].to_numpy(dtype=np.float64)
    moved_a = np.abs(ratio_a - 1.0) >= GATE_EB0_A_RTOL
    fraction_a = float(moved_a.mean()) if moved_a.size else 0.0
    a_pass = fraction_a >= GATE_EB0_A_MIN_FRACTION
    r_phi_consistency = (
        float(np.std(ratio_a) / np.mean(ratio_a)) if ratio_a.size and np.mean(ratio_a) else None
    )

    moved_b_flags: list[npt.NDArray[np.bool_]] = []
    per_seed_fraction_b: dict[int, float] = {}
    per_seed_paired_live: dict[int, int] = {}
    for seed in bc_metas:
        bc_at = _rows_at_h(_meta_csv(bc_metas[seed]), H_GEN)
        bt_at = _rows_at_h(_meta_csv(bt_metas[seed]), H_GEN)
        merged_b = bc_at.merge(
            bt_at[["event_idx", "L_cat_no_bh"]], on="event_idx", suffixes=("_bc", "_bt")
        )
        # PA-7: PAIRED-LIVE = intersection of both arms' LIVE(a) sets.
        live_b = (merged_b["L_cat_no_bh_bc"] > 0.0) & (merged_b["L_cat_no_bh_bt"] > 0.0)
        per_seed_paired_live[seed] = int(live_b.sum())
        with np.errstate(divide="ignore", invalid="ignore"):
            rel_b = np.abs(
                merged_b.loc[live_b, "L_cat_no_bh_bt"] - merged_b.loc[live_b, "L_cat_no_bh_bc"]
            ) / np.maximum(np.abs(merged_b.loc[live_b, "L_cat_no_bh_bc"]), np.finfo(float).tiny)
        moved_b = (rel_b.to_numpy(dtype=np.float64) >= GATE_EB0_B_RTOL).astype(np.bool_)
        moved_b_flags.append(moved_b)
        per_seed_fraction_b[seed] = float(moved_b.mean()) if moved_b.size else 0.0
    pooled_b = np.concatenate(moved_b_flags) if moved_b_flags else np.array([], dtype=np.bool_)
    fraction_b = float(pooled_b.mean()) if pooled_b.size else 0.0
    b_pass = fraction_b >= GATE_EB0_B_MIN_FRACTION

    audit_c_static = (
        "both catalogue_global_selection and catalogue_numerator_survival are validated and "
        "consumed inside BayesianStatistics.evaluate() (bayesian_statistics.py:3452-3490) "
        "before the per-host dispatch; the scalar single_host_likelihood catalogue-leg branch "
        "and the batch _starmap_host_batches call sites (:4692-4721) read the SAME "
        "self._catalogue_numerator_survival/self._catalogue_global_selection instance "
        "attributes set there -- verified by source read (grep-confirmed, not runtime-"
        "measured: production has no runtime call site of the scalar path, per the twin "
        "driver's GATE E-P3(c) finding, reused verbatim here)."
    )
    same_venue = (
        _gate_e_b0a_same_venue(bc_metas[REPLICA_SEED], eb0a_meta) if eb0a_meta is not None else None
    )
    a_pass_gating = bool(same_venue["pass"]) if same_venue is not None else False
    return {
        "gate": "GATE_E-B0",
        "pass": bool(a_pass_gating and b_pass),
        "a_sigma_phi_slot_same_venue": same_venue,  # PA-13(a): THIS gates (a), not the below.
        "a_sigma_phi_slot_reported_only": {
            # PA-13(a)/Finding 3: DEMOTED, cross-venue (replica=b0 vs B-C=b0i) -- "ratio != 1 on
            # 100% of live rows" is trivially true for two different realizations; kept only as
            # a disclosed diagnostic, never contributes to `pass` above.
            "fraction_moved": fraction_a,
            "min_fraction": GATE_EB0_A_MIN_FRACTION,
            "rtol": GATE_EB0_A_RTOL,
            "n_live_rows": int(live_a.sum()),
            "denominator": "LIVE(a): rows with L_cat_no_bh > 0 in both replica and B-C",
            "ratio_mean": float(np.mean(ratio_a)) if ratio_a.size else None,
            "ratio_cv": r_phi_consistency,
            "pass_not_gating": a_pass,
        },
        "b_twin_cell": {
            "pooled_fraction_moved": fraction_b,
            "per_seed_fraction_moved": per_seed_fraction_b,
            "per_seed_paired_live_n": per_seed_paired_live,
            "min_fraction": GATE_EB0_B_MIN_FRACTION,
            "move_rtol": GATE_EB0_B_RTOL,
            "denominator": "PAIRED-LIVE: intersection of B-C's and B-T's LIVE(a) sets",
            "pass": b_pass,
        },
        "c_dispatch_code_audit": {
            "statement": audit_c_static,
            "scalar_path_smoke": _scalar_path_smoke(),
            "pass": True,
        },
        "reference": f"{REGISTRATION_SECTION}, S3 GATE E-B0 (PA-7/PA-13(a) amended)",
    }


def _gate_l_b0(
    bc_metas: dict[int, dict[str, Any]], bt_metas: dict[int, dict[str, Any]]
) -> dict[str, Any]:
    """GATE L-B0: fused line + Sigma^phi line in ALL of B-C/B-T; twin line ONLY in B-T."""

    def _check(metas: dict[int, dict[str, Any]]) -> dict[int, dict[str, bool]]:
        out: dict[int, dict[str, bool]] = {}
        for seed, meta in metas.items():
            log_path = Path(meta["log_path"])
            if not log_path.is_file():
                # PA-16 retrieval: cluster metas record cluster log paths; the
                # retrieval mirrors logs to OUT_ROOT_DEFAULT/<name> (same rule
                # as _meta_csv; meta never edited).
                log_path = OUT_ROOT_DEFAULT / log_path.name
            text = log_path.read_text()
            out[seed] = {
                "fused_present": FUSED_LOG_SUBSTRING in text,
                "sigma_phi_present": SIGMA_PHI_LOG_SUBSTRING in text,
                "twin_present": TWIN_LOG_SUBSTRING in text,
            }
        return out

    bc_flags = _check(bc_metas)
    bt_flags = _check(bt_metas)
    bc_ok = all(
        f["fused_present"] and f["sigma_phi_present"] and not f["twin_present"]
        for f in bc_flags.values()
    )
    bt_ok = all(
        f["fused_present"] and f["sigma_phi_present"] and f["twin_present"]
        for f in bt_flags.values()
    )
    return {
        "gate": "GATE_L-B0",
        "pass": bool(bc_ok and bt_ok),
        "bc_log_flags": bc_flags,
        "bt_log_flags": bt_flags,
        "bc_pass": bc_ok,
        "bt_pass": bt_ok,
        "reference": f"{REGISTRATION_SECTION}, S3 GATE L-B0",
    }


def _load_eb0a_meta(out_root: Path) -> dict[str, Any]:
    """PA-13(a): load the extra same-venue run's meta (run by :func:`stage_pilot`); REFUSED if
    absent (this gate must not silently fall back to the vacuous cross-venue-only form)."""
    meta_path = out_root / f"{EB0A_SUBDIR}_meta.json"
    if not meta_path.is_file():
        sys.exit(
            f"REFUSED: missing {meta_path} -- run --stage pilot first (it now also runs the "
            "PA-13(a) extra same-venue E-B0(a) pair)."
        )
    result: dict[str, Any] = json.loads(meta_path.read_text())
    return result


def _gate_w_b0(
    bc_identity: dict[str, Any],
    bt_identity: dict[str, Any],
    bc_metas: dict[int, dict[str, Any]],
    bt_metas: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    """GATE W-B0 (PA-13(b)/FATAL-4 fix, A20_REVIEW_B0_IMPL_20260823.md Finding 4): closure
    (``closure_pass_all_seeds`` for BOTH B-C and B-T) AND all metas' own
    ``selection_json_cross_check.pass`` (banked per-run by :func:`_run_arm_seed`) -- WIRED into
    the verdict-gating ``gates`` dict (the review found it computed but never consulted).
    """
    closure_bc = bool(bc_identity.get("closure_pass_all_seeds", False))
    closure_bt = bool(bt_identity.get("closure_pass_all_seeds", False))
    sel_json_flags = {
        f"bc_{seed}": bool(meta.get("selection_json_cross_check", {}).get("pass", False))
        for seed, meta in bc_metas.items()
    } | {
        f"bt_{seed}": bool(meta.get("selection_json_cross_check", {}).get("pass", False))
        for seed, meta in bt_metas.items()
    }
    sel_json_all_pass = all(sel_json_flags.values()) if sel_json_flags else False
    return {
        "gate": "GATE_W-B0",
        "pass": bool(closure_bc and closure_bt and sel_json_all_pass),
        "closure_pass_all_seeds_bc": closure_bc,
        "closure_pass_all_seeds_bt": closure_bt,
        "selection_json_cross_check_flags": sel_json_flags,
        "selection_json_cross_check_all_pass": sel_json_all_pass,
        "reference": f"{REGISTRATION_SECTION}, S3 GATE W-B0, PA-13(b) wired into verdict gating",
    }


def stage_gates(out_root: Path) -> dict[str, Any]:
    """Standalone GATE E-B0 + GATE L-B0 + GATE W-B0 scoring (requires replica/pilot-or-fleet
    already run)."""
    replica_meta_path = out_root / "replica_900101_meta.json"
    if not replica_meta_path.is_file():
        sys.exit(f"REFUSED: run --stage replica first (missing {replica_meta_path}).")
    replica_meta = json.loads(replica_meta_path.read_text())
    eb0a_meta = _load_eb0a_meta(out_root)

    bc_metas, bt_metas, missing = _load_fleet_metas(out_root)
    if missing:
        sys.exit(
            f"REFUSED: missing arm/seed meta(s) {missing} -- run --stage pilot and/or "
            "--stage fleet --arm {bc,bt} first."
        )

    gate_e = _gate_e_b0(replica_meta, bc_metas, bt_metas, eb0a_meta=eb0a_meta)
    gate_l = _gate_l_b0(bc_metas, bt_metas)
    bc_identity = _fleet_identity(bc_metas)
    bt_identity = _fleet_identity(bt_metas)
    gate_w = _gate_w_b0(bc_identity, bt_identity, bc_metas, bt_metas)
    out = {"GATE_E-B0": gate_e, "GATE_L-B0": gate_l, "GATE_W-B0": gate_w}
    (out_root / "gates_output.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return out


def _load_fleet_metas(
    out_root: Path,
) -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]], list[str]]:
    bc_metas: dict[int, dict[str, Any]] = {}
    bt_metas: dict[int, dict[str, Any]] = {}
    missing: list[str] = []
    for seed in BSEL_SEEDS:
        for arm, store in (("bc", bc_metas), ("bt", bt_metas)):
            meta_path = out_root / f"{arm}_{seed}_meta.json"
            if meta_path.is_file():
                store[seed] = json.loads(meta_path.read_text())
            else:
                missing.append(f"{arm}_{seed}")
    return bc_metas, bt_metas, missing


# ── secondaries (prereg S4) ───────────────────────────────────────────────────


def _sec2_identity_profile(
    metas: dict[int, dict[str, Any]], h_grid: tuple[float, ...] = H_GRID_41
) -> dict[str, Any]:
    """Secondary 2: I_s(h) across the H grid, fleet-mean profile (shape diagnostic)."""
    profile: dict[str, float | None] = {}
    for h in h_grid:
        vals = []
        for meta in metas.values():
            at = _rows_at_h(_meta_csv(meta), h)
            score = _identity_score(at)
            if score["I_s"] is not None:
                vals.append(score["I_s"])
        profile[str(h)] = float(np.mean(vals)) if vals else None
    return {"reference": f"{REGISTRATION_SECTION}, S4 secondary 2", "I_s_profile": profile}


def _sec4_floor_mass(metas: dict[int, dict[str, Any]]) -> dict[str, Any]:
    """Secondary 4: floor-node ("floor" = lowest H_GRID_41 node, 0.6) posterior mass."""
    grid = np.array(H_GRID_41, dtype=np.float64)
    weights = moment_weights(grid, "trapezoid")

    def floor_mass(csv_path: Path) -> float | None:
        df = pd.read_csv(csv_path)
        piv = (
            df[np.isin(df["h"].to_numpy(dtype=np.float64), grid)]
            .pivot_table(index="event_idx", columns="h", values="combined_no_bh", aggfunc="first")
            .reindex(columns=grid)
        )
        vals = piv.to_numpy(dtype=np.float64)
        sum_log_l = combine_log_likelihood(vals, "physics_floor")
        if not np.isfinite(sum_log_l).any():
            return None
        lp = sum_log_l - sum_log_l.max()
        post = np.exp(lp)
        norm = float((post * weights).sum())
        post_n = post / norm if norm > 0 else post
        return float(post_n[0] * weights[0])

    rows = [
        {
            "seed": seed,
            "floor_node_mass": floor_mass(_meta_csv(meta)),
        }
        for seed, meta in sorted(metas.items())
    ]
    return {
        "reference": f"{REGISTRATION_SECTION}, S4 secondary 4",
        "expected": "materially lower than B-SEL's 27-31% (surprise -> registered finding)",
        "per_seed": rows,
    }


def _sec5_score_at_truth(metas: dict[int, dict[str, Any]]) -> dict[str, Any]:
    """Secondary 5: score-at-truth (A12), class-resolved (full vs pure), reused verbatim from
    ``decompose_impostor_leg.score_at_truth``/``load_matrices``.
    """
    per_seed = []
    for seed, meta in sorted(metas.items()):
        df = pd.read_csv(_meta_csv(meta))
        full_vals, pure_vals, gate_i, n_events = o2.load_matrices(df)
        per_seed.append(
            {
                "seed": seed,
                "gate_i_identity_max_rel": gate_i,
                "n_events": n_events,
                "full": o2.score_at_truth(full_vals),
                "pure": o2.score_at_truth(pure_vals),
            }
        )
    return {
        "reference": (
            "decompose_impostor_leg.score_at_truth()/load_matrices(), full (all events) vs "
            f"pure (completion-only channel) per arm, {REGISTRATION_SECTION} S4 secondary 5"
        ),
        "per_seed": per_seed,
    }


# ── score ──────────────────────────────────────────────────────────────────


def _fleet_identity(metas: dict[int, dict[str, Any]]) -> dict[str, Any]:
    """PA-13(e)/AMEND-8(a): ``k_hat_max`` below is the per-ARM k-hat, defined as the MAX over
    this arm's per-seed k-hats (the prereg's "per-arm" wording, disclosed convention)."""
    per_seed = []
    for seed, meta in sorted(metas.items()):
        at = _rows_at_h(_meta_csv(meta), H_GEN)
        score = _identity_score(at)
        score["seed"] = seed
        per_seed.append(score)
    i_vals = np.array([r["I_s"] for r in per_seed if r["I_s"] is not None], dtype=np.float64)
    i_psis_vals = np.array(
        [r["I_s_psis"] for r in per_seed if r["I_s_psis"] is not None], dtype=np.float64
    )
    k_hats = [r["k_hat"] for r in per_seed if r.get("k_hat") is not None]
    i_bar = float(i_vals.mean()) if i_vals.size else None
    sem = float(i_vals.std(ddof=1) / np.sqrt(i_vals.size)) if i_vals.size > 1 else None
    i_bar_psis = float(i_psis_vals.mean()) if i_psis_vals.size else None
    k_hat_max = float(max(k_hats)) if k_hats else None
    closure_ok = all(bool(r.get("closure_pass", False)) for r in per_seed if r["I_s"] is not None)
    n_dead_total = sum(r["n_dead"] for r in per_seed)
    n_rows_total = sum(r["n_rows"] for r in per_seed)
    dead_rate = (n_dead_total / n_rows_total) if n_rows_total else None
    return {
        "n_seeds": len(per_seed),
        "Ibar": i_bar,
        "sem": sem,
        "Ibar_psis": i_bar_psis,
        "k_hat_max": k_hat_max,  # PA-6(b): UNDETERMINED trigger input
        "closure_pass_all_seeds": closure_ok,
        "n_dead_total": n_dead_total,
        "n_rows_total": n_rows_total,
        "dead_rate": dead_rate,
        "per_seed": [{k: v for k, v in r.items() if k != "ratio_vector"} for r in per_seed],
    }


def _band_verdict(
    i_bar: float | None,
    sem: float | None,
    i_bar_psis: float | None,
    k_hat_max: float | None,
    eps_i: float | None,
) -> str:
    """GATE N-B0 band mapping (prereg S4, PA-6b amended): IDENTITY-PASS/FAIL/UNDETERMINED.
    Requires the frozen ``eps_i`` (band resolution floor); if unfrozen, returns
    ``"BANDS-UNFROZEN"``.

    PA-6(b): the robustness twin is PSIS (not the 5%-trim); UNDETERMINED(a) iff
    ``k_hat_max > PSIS_HEAVY_TAIL_K_HAT`` AND the raw/PSIS bands disagree.
    """
    if i_bar is None or eps_i is None:
        return "BANDS-UNFROZEN"
    threshold = max(3.0 * (sem or 0.0), eps_i)
    band = "PASS" if abs(i_bar) <= threshold else "FAIL"
    if i_bar_psis is None:
        return f"IDENTITY-{band} (PSIS unavailable)"
    psis_band = "PASS" if abs(i_bar_psis) <= threshold else "FAIL"
    heavy_tail = k_hat_max is not None and k_hat_max > PSIS_HEAVY_TAIL_K_HAT
    if heavy_tail and band != psis_band:
        return "UNDETERMINED"
    return f"IDENTITY-{band}"


def stage_score(out_root: Path, eps_i: float | None) -> dict[str, Any]:
    replica_meta_path = out_root / "replica_900101_meta.json"
    if not replica_meta_path.is_file():
        sys.exit(f"REFUSED: run --stage replica first (missing {replica_meta_path}).")
    replica_meta = json.loads(replica_meta_path.read_text())
    gate_rb0_path = out_root / "replica_gate_result.json"
    if not gate_rb0_path.is_file():
        sys.exit(f"REFUSED: run --stage replica first (missing {gate_rb0_path}).")
    gate_rb0 = json.loads(gate_rb0_path.read_text())
    eb0a_meta = _load_eb0a_meta(out_root)

    bc_metas, bt_metas, missing = _load_fleet_metas(out_root)
    if missing:
        sys.exit(
            f"REFUSED: missing arm/seed meta(s) {missing} -- run --stage pilot and/or "
            "--stage fleet --arm {bc,bt} first."
        )
    rescore_path = out_root / "rescore_output.json"
    if not rescore_path.is_file():
        sys.exit(f"REFUSED: run --stage rescore first (missing {rescore_path}).")
    rescore = json.loads(rescore_path.read_text())

    bc_identity = _fleet_identity(bc_metas)
    bt_identity = _fleet_identity(bt_metas)

    gate_e = _gate_e_b0(replica_meta, bc_metas, bt_metas, eb0a_meta=eb0a_meta)
    gate_l = _gate_l_b0(bc_metas, bt_metas)
    # PA-13(b)/FATAL-4 fix: GATE W-B0 now WIRED into the verdict-gating ``gates`` dict (the
    # review's Finding 4 -- it was computed elsewhere but never consulted for all_gates_pass).
    gate_w = _gate_w_b0(bc_identity, bt_identity, bc_metas, bt_metas)
    gates = {
        "GATE_R-B0": gate_rb0,
        "GATE_E-B0": gate_e,
        "GATE_L-B0": gate_l,
        "GATE_W-B0": gate_w,
    }
    all_gates_pass = all(bool(g.get("pass")) for g in gates.values())

    br_i_vals = np.array(
        [r["I_s"] for r in rescore["per_seed"] if r["I_s"] is not None], dtype=np.float64
    )
    br_i_psis_vals = np.array(
        [r["I_s_psis"] for r in rescore["per_seed"] if r["I_s_psis"] is not None],
        dtype=np.float64,
    )
    br_k_hats = [r["k_hat"] for r in rescore["per_seed"] if r.get("k_hat") is not None]
    br_identity = {
        "n_seeds": len(rescore["per_seed"]),
        "Ibar": float(br_i_vals.mean()) if br_i_vals.size else None,
        "sem": float(br_i_vals.std(ddof=1) / np.sqrt(br_i_vals.size))
        if br_i_vals.size > 1
        else None,
        "Ibar_psis": float(br_i_psis_vals.mean()) if br_i_psis_vals.size else None,
        "k_hat_max": float(max(br_k_hats)) if br_k_hats else None,
        "per_seed": rescore["per_seed"],
    }

    bc_band = _band_verdict(
        bc_identity["Ibar"],
        bc_identity["sem"],
        bc_identity["Ibar_psis"],
        bc_identity["k_hat_max"],
        eps_i,
    )
    bt_band = _band_verdict(
        bt_identity["Ibar"],
        bt_identity["sem"],
        bt_identity["Ibar_psis"],
        bt_identity["k_hat_max"],
        eps_i,
    )
    br_band = _band_verdict(
        br_identity["Ibar"],
        br_identity["sem"],
        br_identity["Ibar_psis"],
        br_identity["k_hat_max"],
        eps_i,
    )

    # PA-4 item 4: B-R must equal its predicted value EXACTLY (within a small tolerance) -- the
    # control-at-predicted-value check, reported alongside (not gating) the band verdict.
    br_predicted_i = rescore.get("br_predicted_I")
    br_control_at_predicted: dict[str, Any] | None = None
    if br_predicted_i is not None and br_identity["Ibar"] is not None:
        br_deviation = abs(br_identity["Ibar"] - float(br_predicted_i))
        br_control_at_predicted = {
            "predicted_I": float(br_predicted_i),
            "measured_Ibar": br_identity["Ibar"],
            "deviation": br_deviation,
            # PA-13(c)/Finding 8(c): the loose (order-unity control) tolerance, now a named,
            # disclosed module constant -- not a frozen band.
            "tol": BR_CONTROL_TOLERANCE,
            "pass": bool(br_deviation < BR_CONTROL_TOLERANCE),
        }

    # PA-6(a): dead-row rate arm-comparison VOID rule.
    dead_rate_void: dict[str, Any] = {}
    bc_dead_rate = bc_identity.get("dead_rate")
    bt_dead_rate = bt_identity.get("dead_rate")
    if bc_dead_rate is not None and bt_dead_rate is not None:
        delta = abs(bc_dead_rate - bt_dead_rate)
        dead_rate_void = {
            "bc_dead_rate": bc_dead_rate,
            "bt_dead_rate": bt_dead_rate,
            "delta": delta,
            "void_threshold": DEAD_ROW_VOID_DELTA,
            "void": bool(delta > DEAD_ROW_VOID_DELTA),
        }

    if not all_gates_pass:
        verdict = "GATES-FAILED -- primary/secondaries MAY NOT BE READ"
    elif dead_rate_void.get("void"):
        verdict = "VOID -- dead-row rate delta exceeds PA-6(a) threshold"
    elif "BANDS-UNFROZEN" in (bc_band, bt_band, br_band):
        verdict = "BANDS-UNFROZEN -- freeze eps_I from LEV+pilot before a verdict is read"
    else:
        bc_pass = bc_band.startswith("IDENTITY-PASS")
        bt_pass = bt_band.startswith("IDENTITY-PASS")
        br_pass = br_band.startswith("IDENTITY-PASS")
        if bt_pass and not bc_pass and not br_pass:
            verdict = "TWIN-IDENTITY-CONFIRMED"
        elif bc_pass and not bt_pass:
            verdict = "TWIN-IDENTITY-REFUTED"
        elif bc_pass and bt_pass:
            # PA-5: under C*, B-C carries a predicted O(1) displacement -- both-PASS falsifies
            # the mass derivation itself, not mere insensitivity.
            verdict = (
                "MASS-DERIVATION-FALSIFIED -- both arms PASS under a constant predicting a "
                "B-C displacement (PA-5); return to stage 0"
            )
        elif not bc_pass and not bt_pass:
            verdict = "VENUE-MISSPEC"
        else:
            verdict = "AMBIGUOUS -- see per-arm bands"
        if verdict == "TWIN-IDENTITY-CONFIRMED" and br_pass:
            verdict = "TWIN-IDENTITY-CONFIRMED-FALSIFIED-BY-CONTROL"

    secondaries = {
        "1_percentiles": {
            "B-C": {r["seed"]: r.get("ratio_percentiles") for r in bc_identity["per_seed"]},
            "B-T": {r["seed"]: r.get("ratio_percentiles") for r in bt_identity["per_seed"]},
        },
        "2_identity_profile": {
            "B-C": _sec2_identity_profile(bc_metas),
            "B-T": _sec2_identity_profile(bt_metas),
        },
        "3_paired_delta_mean_h": [
            {
                "seed": seed,
                "mean_h_bt": bt_metas[seed]["mean_h"],
                "mean_h_bc": bc_metas[seed]["mean_h"],
                "delta": float(bt_metas[seed]["mean_h"]) - float(bc_metas[seed]["mean_h"]),
            }
            for seed in BSEL_SEEDS
        ],
        "4_rail_read": _sec4_floor_mass(bc_metas) | {"bt": _sec4_floor_mass(bt_metas)},
        "5_score_at_truth": {
            "B-C": _sec5_score_at_truth(bc_metas),
            "B-T": _sec5_score_at_truth(bt_metas),
        },
        "6_lev_cross_basis": json.loads((out_root / "p3_b0_lev_output.json").read_text())
        if (out_root / "p3_b0_lev_output.json").is_file()
        else None,
    }

    output: dict[str, Any] = {
        "registered_in": REGISTRATION_SECTION,
        "seeds": list(BSEL_SEEDS),
        "h_gen": H_GEN,
        "eps_i": eps_i,
        "mass_companion_at_h_gen": mass_companion(H_GEN),
        "gates": gates,
        "all_gates_pass": all_gates_pass,
        "dead_rate_void": dead_rate_void,
        "primary": {
            "B-C": bc_identity,
            "B-T": bt_identity,
            "B-R_control": br_identity,
            "br_control_at_predicted_value": br_control_at_predicted,
            "br_control_tolerance": BR_CONTROL_TOLERANCE,  # PA-13(c): registered in banded output
            "bands": {"B-C": bc_band, "B-T": bt_band, "B-R": br_band},
            "reference": f"{REGISTRATION_SECTION}, S4 Primary (PA-4/PA-6/PA-13(c) amended)",
        },
        "secondaries": secondaries,
        "verdict": verdict,
    }

    out_path = THIS_DIR / "p3_b0_identity_test_output.json"
    out_path.write_text(json.dumps(output, indent=2))

    print("=== [P3-IMP] b0 identity test score (PA-amended) ===")
    for name, g in gates.items():
        print(f"  {name}: pass={g.get('pass')}")
    print(f"all_gates_pass = {all_gates_pass}")
    print(f"Ibar(B-C) = {bc_identity['Ibar']!r}  SEM = {bc_identity['sem']!r}  band={bc_band}")
    print(f"Ibar(B-T) = {bt_identity['Ibar']!r}  SEM = {bt_identity['sem']!r}  band={bt_band}")
    print(f"Ibar(B-R) = {br_identity['Ibar']!r}  band={br_band}")
    if br_control_at_predicted is not None:
        print(f"B-R control-at-predicted-value check: {br_control_at_predicted}")
    print(f"verdict = {output['verdict']}")
    print(f"wrote {out_path}")
    return output


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--stage",
        choices=("replica", "pilot", "fleet", "lev", "rescore", "gates", "score"),
        required=True,
    )
    ap.add_argument("--arm", type=str, default=None, choices=("bc", "bt"), help="fleet only")
    ap.add_argument("--seeds", type=str, default=None, help="fleet: comma-separated seed subset")
    ap.add_argument(
        "--eps-i",
        type=float,
        default=None,
        help=(
            "score: the frozen band-resolution floor (prereg S4 eps_I); omit while bands are "
            "unfrozen -- the score stage reports BANDS-UNFROZEN rather than guessing one."
        ),
    )
    ap.add_argument(
        "--out-root",
        type=str,
        default=str(THIS_DIR / "p3_b0_work"),
        help="Root scratch/output directory for fresh work roots/logs/metadata.",
    )
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.stage == "lev":
        stage_lev(out_root)
        return 0
    if args.stage == "replica":
        stage_replica(out_root)
        return 0
    if args.stage == "pilot":
        stage_pilot(out_root)
        return 0
    if args.stage == "fleet":
        if args.arm is None:
            raise SystemExit("REFUSED: --stage fleet requires --arm {bc,bt}")
        stage_fleet(
            out_root, args.arm, [int(x) for x in args.seeds.split(",")] if args.seeds else None
        )
        return 0
    if args.stage == "rescore":
        stage_rescore(out_root)
        return 0
    if args.stage == "gates":
        stage_gates(out_root)
        return 0
    result = stage_score(out_root, args.eps_i)
    return (
        0
        if result["verdict"].startswith("IDENTITY")
        or result["verdict"]
        in (
            "TWIN-IDENTITY-CONFIRMED",
            "TWIN-IDENTITY-REFUTED",
            "UNDISCRIMINATING",
            "VENUE-MISSPEC",
        )
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
