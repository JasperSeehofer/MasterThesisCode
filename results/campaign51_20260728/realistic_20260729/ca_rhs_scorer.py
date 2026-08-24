r"""[P3-IMP] the C-A bounded-transform identity RHS scorer -- stage 2, correctness centerpiece.

Registered in ``PREREGISTRATION_CA_BOUNDED_IDENTITY_20260824.md`` (binding; A21 governs). Derived
in ``CLAIM_B0_FINITE_MOMENT_20260824.md`` §3.1 (C-A candidate) and adjudicated in
``GATE_B_ADJUDICATION_FINITE_MOMENT_20260824.md`` §3a/§4. Template for structure/idioms:
``p3_b0_identity_test.py`` (read fully before editing this file) -- this instrument is the RHS
half of the identity the LHS half of which is ALREADY BANKED (the 24 b0i CSVs, zero-compute).

**The identity (prereg §1, adjudication-fixed conventions).** For the class mixture at
``h = H_TRUE``, with ``phi(w) = w`` and ``R = (1-w)/w`` the class-odds ratio:

    C* . E_{d~q_G}[ phi(w).R ]  =  E_{d~q_Gbar}[ phi(w) ]   <=>   C* . E_G[1-w] = E_Gbar[w]

LHS (banked, zero-compute): ``C* . mean(1-w)`` over ALL accepted G-class (b0i, "catalogue_selected")
rows, dead rows INCLUDED (adjudication 3a AMEND: dead rows are bounded/well-defined -- w=0 there,
contribute ``1-w=1``; excluding them is an unregistered conditioning). RHS (THIS instrument, new
compute): ``E_Gbar[w_a]`` for arm ``a`` in {twin, coded} -- a Monte-Carlo mean of ``w`` over
synthetic events drawn from the mixture's OWN completion-class predictive ``q_Gbar = B~/beta_Gbar_phi``,
scored through production's OWN per-event ``L_cat_no_bh``/``B_num`` machinery (imported via
``run_mirror_seed_inprocess``, never reimplemented), never through any venue Gbar-class draw
(the decisive difference from the refuted reciprocal-form candidate, adjudication §3a).

**The completion-class predictive q_Gbar is ALREADY a registered generator (no new sampling code
needed for the draw).** ``correspondence_1d.py``'s AMENDMENT A-3 (module docstring, "bsel"/"bself"
arms) established ``host_mode="population_selected"`` as EXACTLY the estimator's own model of
*detected* dark events: host z ~ ``w_pop(z) . (1 - f_bar(z;h)) . S_bar_phi(z;h)``
(:func:`~darksiren_emri.validation.correspondence_1d.draw_selected_population_redshifts`), sky
isotropic, donor Fisher row SNR-weighted without replacement from the pinned production CRB pool
(:data:`~darksiren_emri.validation.correspondence_1d.CRB_CSV_PATH`, 1590 rows, all SNR>=20) --
BYTE-IDENTICAL to the claim file's Part-B/§2.5 q_Gbar draw recipe ("z ~ mu, donor ~ SNR-weighted
CRB pool, d_hat = d_L(z;h)+sigma_dL.eps, isotropic sky"). "bself" (:data:`ARM_SELECTION_CELL`
``== "fused"``) additionally confirms the numerator/denominator convention this scorer needs
(``selection_in_completion_numerator="fused"``) is already a registered, production-basis cell.
This instrument therefore ADDS: (i) the P3-IMP twin-cell axis
(``catalogue_numerator_survival in {"phi","off"}``) on top of "bself"'s config, scored at ONE
h-node instead of the full grid (10-100x cheaper per event: the b0i venue's own 46-h-node,
200-event runs cost ~2000s wall each -- see the module-level ``_WALLTIME_NOTE`` below); (ii) the
F-0/GATE-ACC acceptance model (:func:`stage_acceptance`); (iii) the GATE RHS-F fidelity replay
(:func:`stage_fidelity`); (iv) chunked accumulation to a target accepted-event count with running
SE (:func:`stage_score`).

**Slot pin ([ORCH-RULE 3], task-mandated).** ``catalogue_global_selection`` is pinned to
``"phi"`` (:data:`CATALOGUE_GLOBAL_SELECTION_SLOT`) for EVERY scored call in this module -- the
banked b0i LHS CSVs (``p3_b0_work/b{c,t}_<seed>_meta.json``) are ALL phi-slot
(``catalogue_global_selection=phi`` in every meta, verified 2026-08-24; matches row #178's
Sigma^phi-slot adoption, which also makes ``run_mirror_seed_inprocess``'s own default ("auto")
resolve to "phi" under ``absolute_marginal`` -- pinning explicitly here is belt-and-braces, not a
behavior change, and is recorded in every output JSON's ``meta`` so a slot drift on either side
(banked LHS vs this RHS) cannot silently misalign the identity's two sides).

**F-0 (task-mandated, mirrors ``bayesian_statistics.py:386``/``:5540-5554`` semantics).**
``sigma_dL/d_hat < FRACTIONAL_LUMINOSITY_DISTANCE_ERROR_THRESHOLD (0.10)`` AND ``SNR >=
SNR_THRESHOLD (20)``. Production enforces this UPSTREAM of the diagnostics CSV (``use_detection``,
:5540-5554): a drawn event either appears in ``event_likelihoods.csv`` (F-0 passed) or never does
-- so every row this scorer reads from a scored diagnostics CSV is, by construction, ALREADY an
F-0-accepted event; no separate acceptance mask is applied downstream of scoring. The ACCEPTANCE
MODEL (GATE ACC, :func:`stage_acceptance`) is a SEPARATE, cheap, zero-``evaluate()`` computation:
F-0 depends only on an event's own ``(d_hat, sigma_dL, SNR)``, none of which require a catalogue
candidate search or Bayesian scoring, so :meth:`MirrorUniverseGenerator.draw_realization`'s OWN
output columns (``luminosity_distance``, ``delta_luminosity_distance_delta_luminosity_distance``,
``SNR``) are read DIRECTLY -- the SAME F-0 formula (imported
``FRACTIONAL_LUMINOSITY_DISTANCE_ERROR_THRESHOLD``/``SNR_THRESHOLD`` constants), applied to the
SAME generative draw the venue itself used, at much higher n than any single 200-event realization
-- a strictly MORE faithful acceptance model than the claim file's own point-evaluated MC (which
disclosed a 7% class-G error against the realized fleet P_G=0.5821; beating that is GATE ACC's own
stated bar, adjudication §4).

Stages (``--stage {score,acceptance,fidelity,lhs,manifest,determinism}``, PA-CA-1..9 amended
2026-08-24 per ``A20_REVIEW_CA_DESIGN_20260824.md``, folded into
``PREREGISTRATION_CA_BOUNDED_IDENTITY_20260824.md``'s PA-CA-1..9 block):

- ``lhs`` (PA-CA-1/PA-CA-4/PA-CA-7b): zero-compute pass over the 24 banked b0i CSVs -- the
  drawn-count-normalized LHS per arm, the paired Delta, GATE B-R's banked side, and the C-TCI
  LHS profile, cross-checked against the review's banked values. No ``evaluate()`` call, no
  synthetic draw.
- ``acceptance`` (GATE ACC, PA-CA-3 amended): predicts each banked b0i seed's ``n_kept`` (out of
  200 drawn) via a large chunked MC replay of BOTH host modes (``catalogue_selected``=G-class,
  ``population_selected``=Gbar-class); PASS iff every checked seed's realized n_kept falls in
  the model's 99.6%-coverage per-seed binomial band AND the fleet P_G is within 2*sigma_P of the
  realized 0.5821 (sigma_P per PA-CA-3's registered formula). Venue-fidelity only -- no
  acceptance-model number enters the verdict statistic (PA-CA-1). Zero-compute beyond the MC
  itself (no ``evaluate()`` calls -- see the F-0 note above).
- ``fidelity`` (GATE RHS-F): replays ONE banked b0i seed's own G-class ("catalogue_selected")
  draw (same seed => byte-identical realization) through THIS scorer's ``_score_events`` plumbing
  under both arrangements, and diffs ``L_cat_no_bh``/``B_num``/``combined_no_bh`` at ``h=H_GEN``
  against the banked ``bt_<seed>``/``bc_<seed>`` diagnostics CSVs at <=1e-6 relative (the CSV
  storage floor, GATE W-B0's own tolerance, reused).
- ``score`` (the RHS itself, PA-CA-1/2/4/7b amended): chunked draws from ``q_Gbar``
  ("population_selected"), scored under BOTH arrangements at ``h=H_GEN`` only, accumulating the
  drawn-count-normalized ``RHS_model(a)`` (SE over chunk means), the PA-CA-2 coherence-slope
  accumulators (``D_C``, ``kappa_hat``), the PA-CA-4 ``RHS_BR`` companion, and the PA-CA-7(b)
  C-TCI indicator profile -- all on the SAME synthetic set, until a target DRAWN (not accepted)
  event count is reached (``--n-events``).
- ``manifest`` (PA-CA-7e): the complete 24-CSV + 24-selection-JSON sha256 manifest.
- ``determinism`` (PA-CA-9): verifies the class-G draw-weight cache is byte-identical to the
  unpatched leaf, for a fixed seed.

A22 (stamped BEFORE any scoring call, into every stage's output JSON, per the module import of
:func:`p3_b0_identity_test._a22_stamp`; PA-CA-6 -- all three RESOLVED flag values recorded, never
"auto").

PA-CA-8: chunk size is HARD-PINNED to :data:`REGISTERED_CHUNK_SIZE` (200) for every registered
run of ``--stage score``/``acceptance``; ``--unsafe-chunk-size`` is the only escape hatch,
loudly disclosed, never valid for a banked verdict.

Usage:
    uv run python results/campaign51_20260728/realistic_20260729/ca_rhs_scorer.py --stage lhs
    uv run python .../ca_rhs_scorer.py --stage acceptance --n-mc 20000 \
        --check-seeds 900101,900102
    uv run python .../ca_rhs_scorer.py --stage fidelity --fidelity-seed 900101
    uv run python .../ca_rhs_scorer.py --stage score --n-events 5000 --seed 960001
    uv run python .../ca_rhs_scorer.py --stage manifest
    uv run python .../ca_rhs_scorer.py --stage determinism
"""

import argparse
import functools
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

# Same-directory import (p3_b0_identity_test.py lives next to this file; importing it does not
# execute its main(), guarded by __main__). Reuses C*/mass_companion/beta_G_phi/A22/log-capture
# leaves -- "C* and every model-side moment from the SAME leaf builds as the RHS scorer" (prereg
# §1 convention (ii)). ``o5.o3`` is ``p3_completed_rescore`` (o5's own same-directory import),
# reused here for :func:`_r_h_gen` (PA-CA-4) -- never a second, independently-maintained copy of
# ``_build_betas``.
import p3_b0_identity_test as o5  # noqa: E402
import pandas as pd
from scipy.stats import binom, chi2

from darksiren_emri.bayesian_inference.bayesian_statistics import (  # noqa: E402
    FRACTIONAL_LUMINOSITY_DISTANCE_ERROR_THRESHOLD,
)
from darksiren_emri.constants import SNR_THRESHOLD  # noqa: E402
from darksiren_emri.validation import correspondence_1d as c1d  # noqa: E402

# ── PA-CA-9 ([ORCH-DO], cost structure): cache the class-G draw weights across chunks ────────
#
# ``c1d.draw_realization``'s "catalogue_selected" branch calls ``catalogue_selected_host_draw_
# weights`` as a bare module-global lookup at call time (correspondence_1d.py:1688), so
# reassigning the module attribute here (THIS PROCESS ONLY) is sufficient to memoize it -- no
# production source file is edited, and the ORIGINAL leaf is called verbatim on every cache miss
# (never reimplemented). This scorer's process builds exactly one ``(pool, phi_survival_table,
# completeness)`` triple per stage (:func:`_load_handler_and_pool`/`_completion_class_objects`,
# both cached/reused) and calls ``draw_realization`` many times against it at a fixed
# ``h=c1d.H_TRUE`` -- the per-chunk full-catalogue recompute the review costed is paid ONCE.
# Pure memoization, no math change (verified: :func:`_determinism_check`, ``--stage determinism``).
_ORIGINAL_CATALOGUE_SELECTED_HOST_DRAW_WEIGHTS = c1d.catalogue_selected_host_draw_weights
_draw_weights_cache: dict[
    tuple[int, int, int, float],
    tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]],
] = {}


def _cached_catalogue_selected_host_draw_weights(
    pool: Any, phi_survival_table: Any, completeness: Any, h: float = c1d.H_TRUE
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """PA-CA-9 memoized wrapper -- keyed on object identity (this process only ever constructs
    one ``pool``/``phi_survival_table``/``completeness`` per stage, reused across every chunk),
    not a deep hash of the catalogue arrays. Calls the UNPATCHED original on a cache miss.
    """
    key = (id(pool), id(phi_survival_table), id(completeness), h)
    cached = _draw_weights_cache.get(key)
    if cached is None:
        cached = _ORIGINAL_CATALOGUE_SELECTED_HOST_DRAW_WEIGHTS(
            pool, phi_survival_table, completeness, h=h
        )
        _draw_weights_cache[key] = cached
    return cached


c1d.catalogue_selected_host_draw_weights = _cached_catalogue_selected_host_draw_weights

REPO_ROOT = Path(__file__).resolve().parents[3]
THIS_DIR = Path(__file__).resolve().parent
OUT_ROOT_DEFAULT: Path = THIS_DIR / "ca_rhs_work"
BANKED_B0I_META_ROOT: Path = THIS_DIR / "p3_b0_work"

REGISTRATION_SECTION: str = (
    "results/campaign51_20260728/realistic_20260729/"
    "PREREGISTRATION_CA_BOUNDED_IDENTITY_20260824.md §2-§3 "
    "(instrument: ca_rhs_scorer.py, per CLAIM_B0_FINITE_MOMENT_20260824.md §3.1 + "
    "GATE_B_ADJUDICATION_FINITE_MOMENT_20260824.md §3a/§4)"
)

H_GEN: float = o5.H_GEN  # H_TRUE = 0.73

# [ORCH-RULE 3]/task mandate: the banked b0i LHS CSVs are ALL phi-slot (verified in every
# p3_b0_work/b{c,t}_<seed>_meta.json, catalogue_global_selection="phi"); pin the RHS scorer to
# match. See the module docstring's "Slot pin" section.
CATALOGUE_GLOBAL_SELECTION_SLOT: str = "phi"

# The registered production-basis completion-numerator cell (PRODUCTION_FLAGS, row #159's D2
# ruling) -- matches every b0i/bself call site; never varied by this instrument.
SELECTION_IN_COMPLETION_NUMERATOR: str = "fused"

# P3-IMP twin-cell axis: the two arrangements this scorer evaluates w under (prereg §2 arm table).
ARRANGEMENT_FLAGS: dict[str, str] = {"twin": "phi", "coded": "off"}

# Donor pool size (prepared_cramer_rao_bounds.csv, verified 2026-08-24: 1590 rows, all SNR>=20)
# bounds any single draw_realization(..., host_mode=...) call's n_events (SNR-weighted WITHOUT
# replacement draw, MirrorUniverseGenerator.draw_realization item (b)).
DONOR_POOL_SIZE: int = 1590

# [FINDING, smoke-test-derived, 2026-08-24] chunk_size is NOT merely a memory/performance knob
# (unlike mass_companion's/kernel_smeared_survival's row-chunking, the "20.8M-row lesson" the
# module docstring's usage note originally analogized to) -- draw_realization's donor-row draw is
# WITHOUT REPLACEMENT WITHIN one call (item (b)), so drawing chunk_size >> 200 (the registered
# per-realization n_events, ARM_SPECS/b0i) forces inclusion of MOST of the 1590-row donor pool,
# including many low-SNR/high-sigma_dL rows a genuine 200-event realization's SNR-weighted draw
# would rarely reach -- SYSTEMATICALLY biasing any F-0-acceptance-rate estimate low. Measured:
# chunk_size=1200 replicated on 900101's own seed still reproduces the banked n_accept=106/200
# exactly when drawn AT n=200 (single-seed replay, confirms the per-event formula is right), but
# a GATE-ACC MC run at chunk_size=1200/n_mc=5000 gave P_G_model=0.4555 -- 19.8 model-SE below the
# realized fleet P_G=0.5821, i.e. WORSE than the claim's own disclosed 7% error, the exact bar
# GATE ACC is meant to beat. The default below is therefore pinned to the REGISTERED
# per-realization size (200) -- many independent 200-draws (chunked realizations, each internally
# without-replacement, mutually independent) is the estimator that is provably faithful to
# draw_realization's OWN implemented sampling law, at the cost of more run_mirror_seed_inprocess/
# draw_realization calls for a given target n. (Whether the venue's TRUE q_G/q_Gbar donor marginal
# should instead be a WITH-replacement i.i.d. SNR-weighted resample -- matching the claim file's
# own large-n [2e5-draw] acceptance MC, which could not have been drawing without replacement at
# that scale from a 1590-row pool -- is a design question this instrument surfaces but does not
# resolve; see the accompanying task report.)
# PA-CA-8: the venue's own realization law (200-event realizations) is registered; this scorer
# refuses any other chunk size for registered stages (score/acceptance) unless the caller opts
# into the explicitly-non-registered ``--unsafe-chunk-size`` escape hatch (see ``_cli``).
DEFAULT_CHUNK_SIZE: int = 200
REGISTERED_CHUNK_SIZE: int = 200

# GATE ACC (PA-CA-3, reviewer's replacement text verbatim): per-seed coverage 99.6% (joint
# false-STOP ~= 5% over 12 seeds) -- supersedes the pre-review 95% level (~54% all-12-pass rate
# for a correct model at 95%, disclosed in the review as the defect this replaces).
GATE_ACC_BAND_LEVEL: float = 0.996
GATE_ACC_REALIZED_FLEET_P_G: float = 0.5821  # GATE_B_ADJUDICATION_FINITE_MOMENT_20260824.md §1b
# PA-CA-3 registered overdispersion note (disclosed constants from the review's own recompute;
# this driver recomputes both from the checked-seed n_kept fleet at runtime -- these are the
# review's banked reference values, not asserted, printed alongside the fresh recompute):
GATE_ACC_REVIEW_REALIZED_SEED_SD: float = 8.84
GATE_ACC_REVIEW_COMMON_P_SD: float = 6.98
GATE_ACC_REVIEW_CHI2_PVALUE: float = 0.09

# PA-CA-7(b): the C-TCI robustness-twin tau grid (indicator member only, winsorized member
# dropped per the review).
C_TCI_TAUS: tuple[float, ...] = (30.0, 100.0, 300.0, 1000.0)

# PA-CA-1/PA-CA-4: the review's banked (drawn-count-normalized) LHS cross-check values, frozen
# BEFORE this instrument runs (A20_REVIEW_CA_DESIGN_20260824.md, PA-CA-1/PA-CA-4 blocks). The
# zero-compute ``--stage lhs`` pass recomputes these independently from the 24 banked CSVs and
# asserts agreement.
BANKED_LHS_BT_DC: float = 0.04233
BANKED_LHS_BT_DC_SE: float = 0.00108
BANKED_LHS_BC_DC: float = 0.03741
BANKED_LHS_BC_DC_SE: float = 0.00095
BANKED_LHS_DELTA_DC: float = 0.004919
BANKED_LHS_DELTA_DC_SE: float = 0.000146
BANKED_LHS_BR_DC: float = 0.03571
BANKED_LHS_BR_DC_SE: float = 0.00093
LHS_BANKED_AGREEMENT_ATOL: float = 5.0e-5  # task-mandated "assert agreement to ~1e-5"

# PA-CA-4: R(H_GEN), registered literal cross-check (A20_REVIEW_CA_DESIGN_20260824.md; the r_h
# this driver computes via :func:`_r_h_gen` -- the SAME leaf B-R's control uses -- must reproduce
# this to high precision; printed, not silently asserted, since a tiny catalogue/table drift
# would otherwise hard-crash every stage that reads r_h).
R_H_GEN_REGISTERED_LITERAL: float = 1.515548762178686

# GATE RHS-F: the CSV storage-precision floor, reused from GATE W-B0 (p3_b0_identity_test.py).
GATE_RHSF_RTOL: float = o5.GATE_WB0_CLOSURE_RTOL  # 1e-6

# The 12 banked b0i seeds (prereg §2's B-T/B-C/B-R fleet -- p3_b0_work/b{c,t}_<seed>_meta.json).
BANKED_SEEDS: tuple[int, ...] = tuple(range(900101, 900113))

# Draw-seed scheme for THIS instrument's OWN synthetic events: disjoint from every registered
# ARM_SEEDS range (which top out at 900125) and from p3_leverage_estimate's disjoint choices --
# arbitrary, disclosed, never a registered campaign seed.
DEFAULT_SCORE_BASE_SEED: int = 960001
DEFAULT_MC_BASE_SEED: int = 970001

# _WALLTIME_NOTE: p3_b0_work/bt_900101_meta.json records wall_time_s=2018.8 for 106-out-of-200
# events scored over c1d.H_GRID_FULL (46 h-nodes). This scorer restricts h_values to (H_GEN,)
# ONLY (the identity is a single-h-node statement, prereg §1) -- an ~46x reduction in per-event-h
# work relative to that reference run, before accounting for the fixed per-call BayesianStatistics
# context-build cost (SimulationDetectionProbability grid, paid once per run_mirror_seed_inprocess
# call regardless of h_values length, AMENDMENT A-3's disclosed "paid twice" note).


def _completion_class_objects(h: float = H_GEN) -> tuple[Any, Any]:
    """(completeness, phi_survival_table) at ``h`` -- imported, cached (functools.lru_cache
    inside :func:`c1d.build_bsel_selection_objects`; shared with :func:`o5.mass_companion`'s own
    internal call, so a process that calls both pays this construction cost once)."""
    return c1d.build_bsel_selection_objects(h_true=h)


def _load_handler_and_pool() -> tuple[Any, Any]:
    """(galaxy_catalog handler, HostPool-with-M) -- reused across every chunk/stage in one
    process (the G-2 reuse finding; no repeated candidate-structure build)."""
    handler = c1d._load_galaxy_catalog_handler(c1d.REDUCED_CATALOGUE_PATH)
    pool = c1d._host_pool_from_handler(handler)
    return handler, pool


def _score_events(
    events: pd.DataFrame,
    work_root: Path,
    seed: int,
    galaxy_catalog: Any,
    catalogue_numerator_survival: str,
) -> pd.DataFrame:
    """Score one synthetic (or replayed) event batch through production's OWN per-event
    ``L_cat_no_bh``/``B_num`` machinery -- ``run_mirror_seed_inprocess`` (imported, never
    reimplemented; the SAME call the b0i venue's own ``_run_arm_seed`` uses), restricted to
    ``h_values=(H_GEN,)`` (single-node, prereg §1) and pinned to
    :data:`CATALOGUE_GLOBAL_SELECTION_SLOT`/:data:`SELECTION_IN_COMPLETION_NUMERATOR`.

    Returns:
        The scored diagnostics rows at ``h=H_GEN`` (every row present already F-0-accepted --
        production writes only accepted events, see the module docstring's F-0 section).
    """
    with o5._capture_root_log(work_root.parent / f"{work_root.name}.log"):
        diag_csv, elapsed = c1d.run_mirror_seed_inprocess(
            work_root,
            events,
            seed,
            galaxy_catalog=galaxy_catalog,
            h_values=(H_GEN,),
            selection_in_completion_numerator=SELECTION_IN_COMPLETION_NUMERATOR,
            catalogue_numerator_survival=catalogue_numerator_survival,
            catalogue_global_selection=CATALOGUE_GLOBAL_SELECTION_SLOT,
            # PA-CA-10 ([P3-HGRID], rows #182-#184): pin the candidate-ball
            # h-bounds to the banked fleet's own H_GRID_FULL extremes so the
            # single-h replay is bit-compatible with the banked CSVs (proven:
            # bounds alone reproduce bc_900101 exactly, all three columns).
            h_bounds=(min(c1d.H_GRID_FULL), max(c1d.H_GRID_FULL)),
        )
    at = o5._rows_at_h(diag_csv, H_GEN)
    at.attrs["elapsed_s"] = elapsed
    at.attrs["diag_csv"] = str(diag_csv)
    return at


def _w_from_csv_columns(at: pd.DataFrame) -> npt.NDArray[np.float64]:
    """``w_e = A_e / (A_e + B_num_e)``, ``A_e = beta_G_phi . L_cat_no_bh_e`` -- the prereg §1
    DEFINITION, read from production's OWN written columns (``alpha_G_phi``, ``r_Malm``,
    ``L_cat_no_bh``, ``B_num``; ``beta_G_phi`` via :func:`o5._beta_g_phi_and_gbar`, imported).
    Dead rows (``L_cat_no_bh == 0`` => ``A_e == 0``) fall out at ``w_e = 0`` automatically (no
    special-casing needed, unlike the LHS's ``(1-w)/w`` ratio form) -- the C-A registration's
    "dead rows INCLUDED" convention (adjudication §3a AMEND) is satisfied by construction.
    """
    beta_g_phi, _beta_gbar_phi = o5._beta_g_phi_and_gbar(at)
    a_e = beta_g_phi * at["L_cat_no_bh"].to_numpy(dtype=np.float64)
    b_num = at["B_num"].to_numpy(dtype=np.float64)
    denom = a_e + b_num
    w_e = np.divide(a_e, denom, out=np.zeros_like(a_e), where=denom > 0.0)
    return np.asarray(w_e, dtype=np.float64)


@functools.lru_cache(maxsize=1)
def _r_h_gen() -> float:
    """PA-CA-4: ``R(H_GEN) = beta_G(H_GEN)/beta_G_phi(H_GEN)`` -- the SAME leaf
    (``p3_completed_rescore._build_betas``, reused via ``o5.o3``) the B-R control in
    ``p3_b0_identity_test.py`` uses for its own ``r_h`` (:func:`o5.stage_rescore`), never a
    second, independently-maintained copy. Cached (this driver only ever calls at ``h=H_GEN``).
    """
    beta_g_phi, _beta_gbar_phi, beta_g, _beta_gbar = o5.o3._build_betas([H_GEN])
    return float(beta_g[H_GEN] / beta_g_phi[H_GEN])


# ── GATE ACC: the F-0 acceptance model (zero-``evaluate()``) ─────────────────


def _acceptance_mc(
    host_mode: str,
    n_target: int,
    base_seed: int,
    pool: Any,
    completeness_obj: Any,
    phi_survival_table: Any,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> dict[str, Any]:
    """Chunked MC replay of one host mode's OWN draw, F-0-filtered directly from the drawn
    events' ``(luminosity_distance, delta_luminosity_distance_delta_luminosity_distance, SNR)``
    columns -- NO ``evaluate()`` call (see the module docstring's F-0 section). Reuses
    :meth:`MirrorUniverseGenerator.draw_realization` unchanged.
    """
    n_drawn = 0
    n_accept = 0
    n_chunks = 0
    chunk = min(chunk_size, DONOR_POOL_SIZE - 2)
    cfg = c1d.CorrespondenceConfig(sigma_z_scale=1.0, area_scale=1.0, n_events=chunk)
    gen = c1d.MirrorUniverseGenerator(cfg)
    while n_drawn < n_target:
        seed = base_seed + n_chunks
        events = gen.draw_realization(
            seed,
            host_pool=pool,
            host_mode=host_mode,  # type: ignore[arg-type]
            completeness=completeness_obj,
            phi_survival_table=phi_survival_table,
        )
        d_hat = events["luminosity_distance"].to_numpy(dtype=np.float64)
        sigma_dl = np.sqrt(
            events["delta_luminosity_distance_delta_luminosity_distance"].to_numpy(dtype=np.float64)
        )
        snr = events["SNR"].to_numpy(dtype=np.float64)
        frac_err = sigma_dl / np.clip(d_hat, 1.0e-12, None)
        acc = (frac_err < FRACTIONAL_LUMINOSITY_DISTANCE_ERROR_THRESHOLD) & (snr >= SNR_THRESHOLD)
        n_drawn += int(events.shape[0])
        n_accept += int(acc.sum())
        n_chunks += 1
    p_hat = n_accept / n_drawn if n_drawn else float("nan")
    se = float(np.sqrt(p_hat * (1.0 - p_hat) / n_drawn)) if n_drawn else float("nan")
    return {
        "host_mode": host_mode,
        "n_drawn": n_drawn,
        "n_accept": n_accept,
        "p_hat": p_hat,
        "se": se,
        "n_chunks": n_chunks,
        "base_seed": base_seed,
        "chunk_size": chunk,
    }


def stage_acceptance(
    n_mc: int,
    mc_seed: int,
    check_seeds: list[int],
    out_path: Path,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> dict[str, Any]:
    """GATE ACC: predict each banked b0i seed's ``n_kept`` (out of 200) and the fleet-level P_G.

    PASS iff every checked seed's realized ``n_kept`` (read from the banked
    ``p3_b0_work/bt_<seed>_meta.json``, zero-compute) falls inside the model's central
    :data:`GATE_ACC_BAND_LEVEL` binomial band, ``Binomial(200, P_G_model)``, AND the fleet-level
    ``P_G_model`` is within its own MC standard error of the realized 0.5821
    (:data:`GATE_ACC_REALIZED_FLEET_P_G`).
    """
    t0 = time.time()
    completeness_obj, phi_survival_table = _completion_class_objects(H_GEN)
    _handler, pool = _load_handler_and_pool()
    c1d._verify_rate_weight_parity()

    g_mc = _acceptance_mc(
        "catalogue_selected", n_mc, mc_seed, pool, completeness_obj, phi_survival_table, chunk_size
    )
    gbar_mc = _acceptance_mc(
        "population_selected",
        n_mc,
        mc_seed + 1_000_000,  # disjoint chunk-seed stream from the G-class draw
        pool,
        completeness_obj,
        phi_survival_table,
        chunk_size,
    )

    n_drawn_per_realization = 200  # ARM_SPECS/b0i's own registered n_events (prereg S2)
    lo_q = (1.0 - GATE_ACC_BAND_LEVEL) / 2.0
    hi_q = 1.0 - lo_q
    band_lo = int(binom.ppf(lo_q, n_drawn_per_realization, g_mc["p_hat"]))
    band_hi = int(binom.ppf(hi_q, n_drawn_per_realization, g_mc["p_hat"]))

    per_seed: list[dict[str, Any]] = []
    for seed in check_seeds:
        meta_path = BANKED_B0I_META_ROOT / f"bt_{seed}_meta.json"
        if not meta_path.is_file():
            per_seed.append({"seed": seed, "found": False})
            continue
        meta = json.loads(meta_path.read_text())
        n_kept = int(meta["n_events"])
        passed = band_lo <= n_kept <= band_hi
        per_seed.append(
            {
                "seed": seed,
                "found": True,
                "n_kept_realized": n_kept,
                "band_lo": band_lo,
                "band_hi": band_hi,
                "pass": passed,
            }
        )
    all_found_pass = all(r.get("pass", False) for r in per_seed if r.get("found"))
    any_missing = any(not r.get("found", False) for r in per_seed)

    # PA-CA-3(ii): |P_model - 0.5821| <= 2*sigma_P, sigma_P^2 = 0.5821*0.4179/2400 + SE_model^2.
    sigma_p_sq = (
        GATE_ACC_REALIZED_FLEET_P_G * (1.0 - GATE_ACC_REALIZED_FLEET_P_G) / 2400.0 + g_mc["se"] ** 2
    )
    sigma_p = float(np.sqrt(sigma_p_sq))
    fleet_delta = g_mc["p_hat"] - GATE_ACC_REALIZED_FLEET_P_G
    fleet_within_2sigma_p = abs(fleet_delta) <= 2.0 * sigma_p

    # PA-CA-3 registered overdispersion note: realized seed sd of n_kept vs the common-p
    # binomial(200, GATE_ACC_REALIZED_FLEET_P_G) sd, chi2_{n-1} test -- recomputed fresh here
    # (not asserted against the review's own disclosed 8.84/6.98/p~=0.09; those are printed
    # alongside as the banked reference).
    n_kept_arr = np.array(
        [r["n_kept_realized"] for r in per_seed if r.get("found")], dtype=np.float64
    )
    common_p_sd = float(
        np.sqrt(200.0 * GATE_ACC_REALIZED_FLEET_P_G * (1.0 - GATE_ACC_REALIZED_FLEET_P_G))
    )
    if n_kept_arr.size > 1:
        realized_seed_sd = float(n_kept_arr.std(ddof=1))
        chi2_stat = float(
            np.sum((n_kept_arr - 200.0 * GATE_ACC_REALIZED_FLEET_P_G) ** 2)
            / (200.0 * GATE_ACC_REALIZED_FLEET_P_G * (1.0 - GATE_ACC_REALIZED_FLEET_P_G))
        )
        chi2_dof = int(n_kept_arr.size - 1)
        chi2_p_value = float(chi2.sf(chi2_stat, chi2_dof))
    else:
        realized_seed_sd, chi2_stat, chi2_dof, chi2_p_value = (
            float("nan"),
            float("nan"),
            0,
            float("nan"),
        )

    # PA-CA-3: clause (i) fail + clause (ii) pass is itself a FINDING on the draw law's seed
    # conditioning (STOP, A21-amend) -- distinct from an ordinary FAIL (both clauses off, or
    # clause (ii) itself off).
    if any_missing:
        gate_acc_verdict = "INCOMPLETE"
    elif all_found_pass and fleet_within_2sigma_p:
        gate_acc_verdict = "PASS"
    elif (not all_found_pass) and fleet_within_2sigma_p:
        gate_acc_verdict = "FINDING_SEED_CONDITIONING"
    else:
        gate_acc_verdict = "FAIL"

    out: dict[str, Any] = {
        "reference": f"{REGISTRATION_SECTION}, GATE ACC (PA-CA-3 amended)",
        "h_gen": H_GEN,
        "catalogue_global_selection_slot": CATALOGUE_GLOBAL_SELECTION_SLOT,
        "a22_stamp": o5._a22_stamp(),
        "venue_fidelity_only": True,  # PA-CA-3: with PA-CA-1, no acceptance-model number enters T_w
        "P_G_model": g_mc["p_hat"],
        "P_G_model_se": g_mc["se"],
        "P_Gbar_model": gbar_mc["p_hat"],
        "P_Gbar_model_se": gbar_mc["se"],
        "g_class_mc": g_mc,
        "gbar_class_mc": gbar_mc,
        "band_level": GATE_ACC_BAND_LEVEL,
        "p_bar_seed_invariant": True,  # this implementation's one MC p_hat is used for every
        # seed's band (disclosed per PA-CA-3's "state whether p_bar is seed-invariant")
        "n_drawn_per_realization": n_drawn_per_realization,
        "band_lo": band_lo,
        "band_hi": band_hi,
        "per_seed": per_seed,
        "realized_fleet_P_G": GATE_ACC_REALIZED_FLEET_P_G,
        "fleet_P_G_model_minus_realized": fleet_delta,
        "sigma_P": sigma_p,
        "sigma_P_sq": sigma_p_sq,
        "fleet_within_2sigma_P": fleet_within_2sigma_p,
        "overdispersion_note": {
            "realized_seed_sd": realized_seed_sd,
            "common_p_sd": common_p_sd,
            "chi2_stat": chi2_stat,
            "chi2_dof": chi2_dof,
            "chi2_p_value": chi2_p_value,
            "review_banked_realized_seed_sd": GATE_ACC_REVIEW_REALIZED_SEED_SD,
            "review_banked_common_p_sd": GATE_ACC_REVIEW_COMMON_P_SD,
            "review_banked_chi2_p_value": GATE_ACC_REVIEW_CHI2_PVALUE,
        },
        "gate_acc_verdict": gate_acc_verdict,
        "elapsed_s": time.time() - t0,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print("=== [P3-IMP] ca_rhs_scorer -- GATE ACC (F-0 acceptance model, PA-CA-3 amended) ===")
    print(f"P_G_model    = {g_mc['p_hat']:.4f} +/- {g_mc['se']:.4f}  (n={g_mc['n_drawn']})")
    print(
        f"P_Gbar_model = {gbar_mc['p_hat']:.4f} +/- {gbar_mc['se']:.4f}  (n={gbar_mc['n_drawn']})"
    )
    print(
        f"realized fleet P_G = {GATE_ACC_REALIZED_FLEET_P_G}  "
        f"model-realized = {fleet_delta:+.4f}  sigma_P={sigma_p:.4f}  "
        f"within_2sigma_P={fleet_within_2sigma_p}"
    )
    print(
        f"binomial band ({GATE_ACC_BAND_LEVEL:.1%}, n=200, common p_bar) = [{band_lo}, {band_hi}]"
    )
    for r in per_seed:
        if r.get("found"):
            print(f"  seed {r['seed']}: n_kept={r['n_kept_realized']}  pass={r['pass']}")
        else:
            print(f"  seed {r['seed']}: banked meta NOT FOUND")
    print(
        f"overdispersion note: realized seed sd={realized_seed_sd:.2f} "
        f"(review-banked {GATE_ACC_REVIEW_REALIZED_SEED_SD})  common-p sd={common_p_sd:.2f} "
        f"(review-banked {GATE_ACC_REVIEW_COMMON_P_SD})  chi2({chi2_dof})={chi2_stat:.2f} "
        f"p={chi2_p_value:.3f} (review-banked p~={GATE_ACC_REVIEW_CHI2_PVALUE})"
    )
    print(f"GATE ACC verdict = {gate_acc_verdict}")
    print(f"wrote {out_path}")
    return out


# ── LHS bank: zero-compute pass over the 24 banked b0i CSVs (PA-CA-1/4/7b) ───


def stage_lhs(out_path: Path) -> dict[str, Any]:
    """PA-CA-1/PA-CA-4/PA-CA-7(b): the zero-compute LHS side of every registered amended
    quantity, read straight from the 24 banked b0i CSVs -- no synthetic draws, no
    ``evaluate()`` calls.

    Computes, per seed and fleet-averaged over the 12 B-T/B-C pairs:

    - PA-CA-1(a) drawn-count-normalized LHS per arm: ``LHS_s(a) = (C*/200) * sum_{accepted
      rows}(1-w_e)`` (denominator FIXED at 200 -- the per-realization draw count -- never the
      accepted row count; dead rows included, ``w_e=0`` there per PA-CA-5).
    - The paired Delta = LHS_s(B-T) - LHS_s(B-C), per seed, fleet-averaged.
    - PA-CA-4 GATE B-R's banked side: ``LHS_BR_s = (C*/200) * sum_acc (1-w_e)/(1+(r-1)w_e)``
      over the 12 B-T frames, ``r = R(H_GEN)`` (:func:`_r_h_gen`).
    - PA-CA-7(b) C-TCI LHS profile: ``(C*/200) * sum_acc R_e * 1{R_e<=tau}``,
      ``R_e = (1-w_e)/w_e`` (``inf`` at ``w_e=0`` -- excluded from every tau by construction,
      the "dead rows auto-consistent" property), for both arrangements.

    Cross-checked against the review's banked values (asserted, printed -- Finding 5/PA-CA-1/
    PA-CA-4, ``A20_REVIEW_CA_DESIGN_20260824.md``).
    """
    o5._assert_h_true_in_grid()
    t0 = time.time()
    r_h = _r_h_gen()

    per_seed: dict[str, list[dict[str, Any]]] = {"twin": [], "coded": []}
    lhs_br_per_seed: list[dict[str, Any]] = []
    tci_per_seed: dict[str, list[dict[str, Any]]] = {"twin": [], "coded": []}
    _arm_dirprefix = {"twin": "bt", "coded": "bc"}

    for seed in BANKED_SEEDS:
        w_by_arm: dict[str, npt.NDArray[np.float64]] = {}
        c_star_by_arm: dict[str, float] = {}
        for arm_name, prefix in _arm_dirprefix.items():
            meta = json.loads((BANKED_B0I_META_ROOT / f"{prefix}_{seed}_meta.json").read_text())
            at = o5._rows_at_h(o5._meta_csv(meta), H_GEN)
            w_e = _w_from_csv_columns(at)
            c_star_value, _diag = o5.c_star(at)
            n_rows = int(at.shape[0])
            lhs_s = float(c_star_value / 200.0 * np.sum(1.0 - w_e))
            per_seed[arm_name].append(
                {"seed": seed, "n_rows": n_rows, "C_star": c_star_value, "LHS_s": lhs_s}
            )
            w_by_arm[arm_name] = w_e
            c_star_by_arm[arm_name] = c_star_value

            r_e = np.divide(1.0 - w_e, w_e, out=np.full_like(w_e, np.inf), where=w_e > 0.0)
            tci_row: dict[str, Any] = {"seed": seed, "n_rows": n_rows}
            for tau in C_TCI_TAUS:
                tci_row[f"tau_{int(tau)}"] = float(c_star_value / 200.0 * np.sum(r_e[r_e <= tau]))
            tci_per_seed[arm_name].append(tci_row)

        w_bt = w_by_arm["twin"]
        c_star_bt = c_star_by_arm["twin"]
        lhs_br_s = float(c_star_bt / 200.0 * np.sum((1.0 - w_bt) / (1.0 + (r_h - 1.0) * w_bt)))
        lhs_br_per_seed.append({"seed": seed, "LHS_BR_s": lhs_br_s})

    def _fleet(vals: list[float]) -> tuple[float, float | None]:
        arr = np.array(vals, dtype=np.float64)
        mean = float(arr.mean())
        se = float(arr.std(ddof=1) / np.sqrt(arr.size)) if arr.size > 1 else None
        return mean, se

    lhs_bt_mean, lhs_bt_se = _fleet([r["LHS_s"] for r in per_seed["twin"]])
    lhs_bc_mean, lhs_bc_se = _fleet([r["LHS_s"] for r in per_seed["coded"]])
    paired_delta_vals = [
        rt["LHS_s"] - rc["LHS_s"]
        for rt, rc in zip(per_seed["twin"], per_seed["coded"], strict=True)
    ]
    delta_mean, delta_se = _fleet(paired_delta_vals)
    lhs_br_mean, lhs_br_se = _fleet([r["LHS_BR_s"] for r in lhs_br_per_seed])

    tci_fleet: dict[str, dict[str, Any]] = {}
    for arm_name in ("twin", "coded"):
        tci_fleet[arm_name] = {}
        for tau in C_TCI_TAUS:
            key = f"tau_{int(tau)}"
            mean, se = _fleet([r[key] for r in tci_per_seed[arm_name]])
            tci_fleet[arm_name][key] = {"mean": mean, "se": se}

    def _agree(label: str, computed: float, banked: float) -> dict[str, Any]:
        diff = computed - banked
        return {
            "label": label,
            "computed": computed,
            "banked": banked,
            "abs_diff": abs(diff),
            "within_tol": abs(diff) <= LHS_BANKED_AGREEMENT_ATOL,
        }

    crosschecks = [
        _agree("LHS(B-T)", lhs_bt_mean, BANKED_LHS_BT_DC),
        _agree("LHS(B-C)", lhs_bc_mean, BANKED_LHS_BC_DC),
        _agree("paired_Delta", delta_mean, BANKED_LHS_DELTA_DC),
        _agree("LHS_BR", lhs_br_mean, BANKED_LHS_BR_DC),
    ]
    all_agree = all(c["within_tol"] for c in crosschecks)

    out: dict[str, Any] = {
        "reference": f"{REGISTRATION_SECTION}, --stage lhs (PA-CA-1/4/7b, zero-compute)",
        "h_gen": H_GEN,
        "r_h_gen": r_h,
        "r_h_gen_registered_literal": R_H_GEN_REGISTERED_LITERAL,
        "r_h_gen_rel_diff_vs_registered_literal": abs(r_h - R_H_GEN_REGISTERED_LITERAL)
        / abs(R_H_GEN_REGISTERED_LITERAL),
        "a22_stamp": o5._a22_stamp(),
        "n_syn_denominator_per_seed": 200,
        "LHS": {
            "B-T": {"mean": lhs_bt_mean, "se": lhs_bt_se, "per_seed": per_seed["twin"]},
            "B-C": {"mean": lhs_bc_mean, "se": lhs_bc_se, "per_seed": per_seed["coded"]},
        },
        "paired_delta": {"mean": delta_mean, "se": delta_se},
        "LHS_BR": {"mean": lhs_br_mean, "se": lhs_br_se, "per_seed": lhs_br_per_seed},
        "C_TCI_LHS_profile": tci_fleet,
        "c_tci_taus": list(C_TCI_TAUS),
        "banked_crosschecks": crosschecks,
        "banked_crosschecks_atol": LHS_BANKED_AGREEMENT_ATOL,
        "all_banked_crosschecks_within_tol": all_agree,
        "elapsed_s": time.time() - t0,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))

    print("=== [P3-IMP] ca_rhs_scorer -- LHS bank (--stage lhs, zero-compute, PA-CA-1/4/7b) ===")
    print(
        f"LHS(B-T) = {lhs_bt_mean:.5f} +/- {lhs_bt_se:.5f}  "
        f"(banked {BANKED_LHS_BT_DC} +/- {BANKED_LHS_BT_DC_SE})"
    )
    print(
        f"LHS(B-C) = {lhs_bc_mean:.5f} +/- {lhs_bc_se:.5f}  "
        f"(banked {BANKED_LHS_BC_DC} +/- {BANKED_LHS_BC_DC_SE})"
    )
    print(
        f"paired Delta = {delta_mean:+.6f} +/- {delta_se:.6f}  "
        f"(banked {BANKED_LHS_DELTA_DC:+.6f} +/- {BANKED_LHS_DELTA_DC_SE})"
    )
    print(
        f"LHS_BR = {lhs_br_mean:.5f} +/- {lhs_br_se:.5f}  "
        f"(banked {BANKED_LHS_BR_DC} +/- {BANKED_LHS_BR_DC_SE}, r_h={r_h!r})"
    )
    for c in crosschecks:
        print(
            f"  crosscheck {c['label']}: computed={c['computed']!r} banked={c['banked']!r} "
            f"abs_diff={c['abs_diff']:.2e} within_tol={c['within_tol']}"
        )
    for arm_name in ("twin", "coded"):
        print(f"C-TCI LHS profile ({arm_name}):")
        for tau in C_TCI_TAUS:
            key = f"tau_{int(tau)}"
            e = tci_fleet[arm_name][key]
            print(f"  tau={tau:.0f}: mean={e['mean']!r} se={e['se']!r}")
    print(f"all_banked_crosschecks_within_tol = {all_agree}")
    assert all_agree, (
        f"PA-CA-7(h) STOP: LHS crosschecks disagree with the banked review values beyond "
        f"{LHS_BANKED_AGREEMENT_ATOL:.1e} absolute -- {crosschecks}"
    )
    print(f"wrote {out_path}")
    return out


# ── manifest: the complete 24-CSV + 24-selection-JSON sha256 manifest (PA-CA-7e) ─────────────


def stage_manifest(out_path: Path) -> dict[str, Any]:
    """PA-CA-7(e): the review's Finding-7e gap -- ``retrieval_manifest_20260824.json`` covers
    only the cluster-retrieved remainder (seeds 900102/900103 uncovered). This generates the
    COMPLETE manifest: every ``event_likelihoods.csv`` + ``selection_tables_h_0_73.json`` for
    the 12 banked B-T/B-C seed pairs (24 + 24 = 48 files).
    """
    import hashlib

    def _hash_entry(p: Path) -> dict[str, Any]:
        h = hashlib.sha256()
        with p.open("rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
        return {
            "path": str(p.relative_to(REPO_ROOT)),
            "sha256": h.hexdigest(),
            "size_bytes": p.stat().st_size,
        }

    files: list[dict[str, Any]] = []
    missing: list[str] = []
    for arm, prefix in (("twin", "bt"), ("coded", "bc")):
        for seed in BANKED_SEEDS:
            seed_dir = BANKED_B0I_META_ROOT / f"{prefix}_{seed}_work" / f"seed{seed}"
            csv_path = seed_dir / "simulations/diagnostics/event_likelihoods.csv"
            sel_path = seed_dir / "selection_tables_h_0_73.json"
            for kind, p in (
                ("event_likelihoods_csv", csv_path),
                ("selection_table_json", sel_path),
            ):
                if p.is_file():
                    entry = _hash_entry(p)
                    entry.update({"arm": arm, "arm_prefix": prefix, "seed": seed, "kind": kind})
                    files.append(entry)
                else:
                    missing.append(f"{prefix}_{seed}:{kind}:{p}")

    out: dict[str, Any] = {
        "reference": f"{REGISTRATION_SECTION}, --stage manifest (PA-CA-7e)",
        "description": (
            "Complete sha256 manifest of the 24 banked b0i event_likelihoods.csv + 24 "
            "selection_tables_h_0_73.json (bt/bc x 12 seeds each) -- closes the review's "
            "Finding-7e gap (retrieval_manifest_20260824.json covered only the cluster-"
            "retrieved remainder, seeds 900102/900103 absent)."
        ),
        "n_expected": 48,
        "n_found": len(files),
        "missing": missing,
        "files": files,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print("=== [P3-IMP] ca_rhs_scorer -- manifest (--stage manifest, PA-CA-7e) ===")
    print(f"n_found = {len(files)} / 48   missing = {missing}")
    print(f"wrote {out_path}")
    return out


# ── determinism check: the PA-CA-9 draw-weight cache ──────────────────────────


def _determinism_check(seed: int = 900101) -> dict[str, Any]:
    """PA-CA-9: verify :func:`_cached_catalogue_selected_host_draw_weights` produces
    BYTE-IDENTICAL ``draw_realization`` output to a cold cache, for a fixed seed, AND that the
    cached weights themselves equal the UNPATCHED original leaf's own output -- pure
    memoization, no math change.
    """
    completeness_obj, phi_survival_table = _completion_class_objects(H_GEN)
    _handler, pool = _load_handler_and_pool()

    _draw_weights_cache.clear()
    cfg = c1d.CorrespondenceConfig(sigma_z_scale=1.0, area_scale=1.0, n_events=200)
    gen = c1d.MirrorUniverseGenerator(cfg)
    events_cold = gen.draw_realization(
        seed,
        host_pool=pool,
        host_mode="catalogue_selected",
        completeness=completeness_obj,
        phi_survival_table=phi_survival_table,
    )
    cache_size_after_first = len(_draw_weights_cache)
    events_warm = gen.draw_realization(
        seed,
        host_pool=pool,
        host_mode="catalogue_selected",
        completeness=completeness_obj,
        phi_survival_table=phi_survival_table,
    )
    cache_size_after_second = len(_draw_weights_cache)
    identical = bool(events_cold.equals(events_warm))

    cached_w, cached_wg, cached_stilde = _cached_catalogue_selected_host_draw_weights(
        pool, phi_survival_table, completeness_obj, h=c1d.H_TRUE
    )
    orig_w, orig_wg, orig_stilde = _ORIGINAL_CATALOGUE_SELECTED_HOST_DRAW_WEIGHTS(
        pool, phi_survival_table, completeness_obj, h=c1d.H_TRUE
    )
    weights_match = bool(
        np.array_equal(cached_w, orig_w)
        and np.array_equal(cached_wg, orig_wg)
        and np.array_equal(cached_stilde, orig_stilde)
    )

    out: dict[str, Any] = {
        "reference": f"{REGISTRATION_SECTION}, PA-CA-9 determinism check",
        "seed": seed,
        "cache_size_after_first_call": cache_size_after_first,
        "cache_size_after_second_call": cache_size_after_second,
        "cache_hit_on_second_call": cache_size_after_second == cache_size_after_first,
        "draws_byte_identical_cold_vs_warm": identical,
        "cached_weights_match_unpatched_original": weights_match,
        "pass": bool(
            identical and weights_match and cache_size_after_second == cache_size_after_first
        ),
    }
    print("=== [P3-IMP] ca_rhs_scorer -- PA-CA-9 determinism check (draw-weight cache) ===")
    print(f"cache size after 1st draw_realization call = {cache_size_after_first}")
    print(
        f"cache size after 2nd (same seed) call       = {cache_size_after_second}  "
        f"(hit={out['cache_hit_on_second_call']})"
    )
    print(f"draws byte-identical, cold vs warm cache     = {identical}")
    print(f"cached weights == unpatched-original weights = {weights_match}")
    print(f"PA-CA-9 determinism check verdict = {'PASS' if out['pass'] else 'FAIL'}")
    return out


# ── GATE RHS-F: generator/scorer fidelity replay ─────────────────────────────


def stage_fidelity(seed: int, out_root: Path, out_path: Path) -> dict[str, Any]:
    """GATE RHS-F: replay banked b0i seed ``seed``'s own G-class draw through THIS scorer's
    :func:`_score_events` plumbing (both arrangements), diff against the banked
    ``bt_<seed>``/``bc_<seed>`` diagnostics CSVs at ``h=H_GEN``.

    PASS iff the max relative difference on ``L_cat_no_bh``/``B_num``/``combined_no_bh``, over
    the intersection of scored and banked ``event_idx`` (paired by index), is <=
    :data:`GATE_RHSF_RTOL` for BOTH arrangements.
    """
    t0 = time.time()
    completeness_obj, phi_survival_table = _completion_class_objects(H_GEN)
    handler, pool = _load_handler_and_pool()
    c1d._verify_rate_weight_parity()

    cfg = c1d.CorrespondenceConfig(sigma_z_scale=1.0, area_scale=1.0, n_events=200)
    gen = c1d.MirrorUniverseGenerator(cfg)
    events = gen.draw_realization(
        seed,
        host_pool=pool,
        host_mode="catalogue_selected",
        completeness=completeness_obj,
        phi_survival_table=phi_survival_table,
    )

    stamp = o5._a22_stamp()
    per_arrangement: dict[str, Any] = {}
    for arm_name, flag in ARRANGEMENT_FLAGS.items():
        scored = _score_events(
            events, out_root / f"fidelity_{seed}_{arm_name}_work", seed, handler, flag
        )
        banked_meta = json.loads(
            (BANKED_B0I_META_ROOT / f"b{arm_name[0]}_{seed}_meta.json").read_text()
        )
        banked_csv = o5._meta_csv(banked_meta)
        banked = o5._rows_at_h(banked_csv, H_GEN)
        common_idx = sorted(set(scored["event_idx"]) & set(banked["event_idx"]))
        s = scored.set_index("event_idx").loc[common_idx]
        b = banked.set_index("event_idx").loc[common_idx]
        col_report: dict[str, float] = {}
        for col in ("L_cat_no_bh", "B_num", "combined_no_bh"):
            sv = s[col].to_numpy(dtype=np.float64)
            bv = b[col].to_numpy(dtype=np.float64)
            rel = np.abs(sv - bv) / np.maximum(np.abs(bv), np.finfo(np.float64).tiny)
            col_report[col] = float(np.max(rel)) if rel.size else float("nan")
        per_arrangement[arm_name] = {
            "n_scored": int(scored.shape[0]),
            "n_banked": int(banked.shape[0]),
            "n_common": len(common_idx),
            "max_rel": col_report,
            "pass": all(v <= GATE_RHSF_RTOL for v in col_report.values()) if common_idx else False,
        }

    verdict = "PASS" if all(v["pass"] for v in per_arrangement.values()) else "FAIL"
    out: dict[str, Any] = {
        "reference": f"{REGISTRATION_SECTION}, GATE RHS-F",
        "seed": seed,
        "h_gen": H_GEN,
        "catalogue_global_selection_slot": CATALOGUE_GLOBAL_SELECTION_SLOT,
        "rtol": GATE_RHSF_RTOL,
        "a22_stamp": stamp,
        "a22_flags": {  # PA-CA-6: all three RESOLVED flag values, never "auto"
            "catalogue_global_selection": CATALOGUE_GLOBAL_SELECTION_SLOT,
            "selection_in_completion_numerator": SELECTION_IN_COMPLETION_NUMERATOR,
            "catalogue_numerator_survival": dict(ARRANGEMENT_FLAGS),
        },
        "per_arrangement": per_arrangement,
        "gate_rhsf_verdict": verdict,
        "elapsed_s": time.time() - t0,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print("=== [P3-IMP] ca_rhs_scorer -- GATE RHS-F (generator/scorer fidelity) ===")
    for arm_name, rep in per_arrangement.items():
        print(
            f"  {arm_name}: n_common={rep['n_common']} max_rel={rep['max_rel']} pass={rep['pass']}"
        )
    print(f"GATE RHS-F verdict = {verdict}")
    print(f"wrote {out_path}")
    return out


# ── score: the RHS accumulator ────────────────────────────────────────────────


def stage_score(
    n_events_target: int,
    base_seed: int,
    out_root: Path,
    out_path: Path,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> dict[str, Any]:
    """The RHS itself: chunked draws from ``q_Gbar`` ("population_selected"), scored under BOTH
    arrangements at ``h=H_GEN`` only, accumulating the PA-CA-1 DRAWN-COUNT-normalized RHS_model
    with running SE **computed over chunk means** (PA-CA-1(a)), plus the PA-CA-2 coherence-slope
    accumulators and the PA-CA-4/PA-CA-7(b) companions -- all on the SAME synthetic set.

    ``RHS_model(a) = (1/N_syn) * sum over ALL synthetic draws of w_a . 1_acc`` -- normalized by
    the TOTAL drawn count (``N_syn``), never the accepted-conditional mean: since production
    writes only F-0-accepted rows, an unaccepted draw already contributes exactly 0 to every
    per-chunk sum, and each chunk's OWN drawn count (``n_drawn_this_chunk`` -- fixed at
    :data:`REGISTERED_CHUNK_SIZE` per PA-CA-8) is the correct per-chunk denominator.

    ``n_events_target`` counts TOTAL DRAWN synthetic events (not accepted; F-0 acceptance is
    ~60-90% empirically for the b0i venue, per the claim file's conditioned-target forensics) --
    chunked at ``chunk_size`` (PA-CA-8: hard-pinned to :data:`REGISTERED_CHUNK_SIZE` for any
    registered run; a different value here means the caller passed ``--unsafe-chunk-size``),
    each chunk drawn with a FRESH internal seed (``base_seed + chunk_index``, disjoint from every
    registered campaign seed range).
    """
    o5._assert_h_true_in_grid()
    t0 = time.time()
    stamp = o5._a22_stamp()  # A22: written BEFORE any scoring call.
    r_h = _r_h_gen()

    completeness_obj, phi_survival_table = _completion_class_objects(H_GEN)
    handler, pool = _load_handler_and_pool()

    chunk = min(chunk_size, DONOR_POOL_SIZE - 2)
    cfg = c1d.CorrespondenceConfig(sigma_z_scale=1.0, area_scale=1.0, n_events=chunk)
    gen = c1d.MirrorUniverseGenerator(cfg)

    # PA-CA-1(a): per-chunk means (drawn-count normalized, i.e. divided by THIS chunk's own
    # n_drawn, not its n_accepted) -- SE is computed over these, never over the per-event
    # variance/n (which would silently reintroduce the accepted-conditional mean the amendment
    # removes). Bounded peak memory regardless of the total target event count (the "20.8M-row
    # lesson": chunk, never hold the full accumulation in one dense in-memory table).
    chunk_means: dict[str, list[float]] = {arm: [] for arm in ARRANGEMENT_FLAGS}
    dc_chunk_means: list[float] = []  # PA-CA-2: E_Gbar[W~.w_BC.1_acc] per-chunk mean
    br_chunk_means: list[float] = []  # PA-CA-4: E_Gbar[w_BT/(1+(r-1)w_BT).1_acc] per-chunk mean
    tci_chunk_means: dict[str, dict[float, list[float]]] = {
        arm: {tau: [] for tau in C_TCI_TAUS} for arm in ARRANGEMENT_FLAGS
    }
    running: dict[str, dict[str, float]] = {
        arm: {"sum": 0.0, "n_accepted": 0.0} for arm in ARRANGEMENT_FLAGS
    }
    per_chunk: list[dict[str, Any]] = []
    n_drawn_total = 0
    chunk_idx = 0
    while n_drawn_total < n_events_target:
        seed = base_seed + chunk_idx
        events = gen.draw_realization(
            seed,
            host_pool=pool,
            host_mode="population_selected",
            completeness=completeness_obj,
            phi_survival_table=phi_survival_table,
        )
        n_this_chunk = int(events.shape[0])
        n_drawn_total += n_this_chunk

        idx_by_arm: dict[str, npt.NDArray[np.int64]] = {}
        w_by_arm: dict[str, npt.NDArray[np.float64]] = {}
        l_cat_by_arm: dict[str, npt.NDArray[np.float64]] = {}
        for arm_name, flag in ARRANGEMENT_FLAGS.items():
            scored = _score_events(
                events, out_root / f"score_chunk{chunk_idx}_{arm_name}_work", seed, handler, flag
            )
            w_e = _w_from_csv_columns(scored)
            idx_by_arm[arm_name] = scored["event_idx"].to_numpy(dtype=np.int64)
            w_by_arm[arm_name] = w_e
            l_cat_by_arm[arm_name] = scored["L_cat_no_bh"].to_numpy(dtype=np.float64)

            running[arm_name]["sum"] += float(w_e.sum())
            running[arm_name]["n_accepted"] += float(w_e.size)
            # PA-CA-1(a): divide by THIS chunk's drawn count, not its accepted count.
            chunk_means[arm_name].append(float(w_e.sum()) / n_this_chunk)

            r_e = np.divide(1.0 - w_e, w_e, out=np.full_like(w_e, np.inf), where=w_e > 0.0)
            for tau in C_TCI_TAUS:
                tci_chunk_means[arm_name][tau].append(float(np.sum(r_e <= tau)) / n_this_chunk)

        # F-0 is arrangement-independent (depends only on d_hat/sigma_dL/SNR, unaffected by
        # catalogue_numerator_survival) -- the accepted event_idx set must match between arms.
        idx_sets = [set(v.tolist()) for v in idx_by_arm.values()]
        arrangement_consistent = all(s == idx_sets[0] for s in idx_sets[1:])
        if not arrangement_consistent:
            print(
                f"WARNING chunk {chunk_idx} (seed {seed}): accepted event_idx differs between "
                "arrangements -- F-0 is expected to be arrangement-independent; investigate "
                "before trusting this chunk's contribution.",
                file=sys.stderr,
            )

        # PA-CA-2: E_Gbar[W~.w_BC.1_acc], W~ = L_cat^BT/L_cat^BC per synthetic row, SAME rows
        # scored under both flags this chunk -- aligned by event_idx (rows with L_cat^BC=0
        # contribute 0, per the registered convention).
        if arrangement_consistent and idx_by_arm["twin"].size:
            twin_order = np.argsort(idx_by_arm["twin"])
            coded_order = np.argsort(idx_by_arm["coded"])
            l_bt_sorted = l_cat_by_arm["twin"][twin_order]
            l_bc_sorted = l_cat_by_arm["coded"][coded_order]
            w_bc_sorted = w_by_arm["coded"][coded_order]
            w_tilde = np.divide(
                l_bt_sorted, l_bc_sorted, out=np.zeros_like(l_bt_sorted), where=l_bc_sorted > 0.0
            )
            dc_term = w_tilde * w_bc_sorted
        else:
            dc_term = np.array([], dtype=np.float64)
        dc_chunk_means.append(float(dc_term.sum()) / n_this_chunk)

        # PA-CA-4: E_Gbar[w_BT/(1+(r-1)w_BT).1_acc], twin arrangement only.
        w_bt = w_by_arm["twin"]
        br_term = w_bt / (1.0 + (r_h - 1.0) * w_bt)
        br_chunk_means.append(float(br_term.sum()) / n_this_chunk)

        per_chunk.append(
            {
                "chunk": chunk_idx,
                "seed": seed,
                "n_drawn": n_this_chunk,
                "n_accepted_twin": int(w_by_arm["twin"].size),
                "n_accepted_coded": int(w_by_arm["coded"].size),
                "arrangement_consistent": arrangement_consistent,
                "mean_w_twin_chunk_dc": chunk_means["twin"][-1],
                "mean_w_coded_chunk_dc": chunk_means["coded"][-1],
                "dc_term_chunk": dc_chunk_means[-1],
                "br_term_chunk": br_chunk_means[-1],
            }
        )
        chunk_idx += 1

    def _chunk_agg(vals: list[float]) -> dict[str, Any]:
        arr = np.array(vals, dtype=np.float64)
        mean = float(arr.mean()) if arr.size else float("nan")
        se = float(arr.std(ddof=1) / np.sqrt(arr.size)) if arr.size > 1 else float("nan")
        return {"mean": mean, "se": se, "n_chunks": int(arr.size)}

    arms_out: dict[str, Any] = {}
    for arm_name in ARRANGEMENT_FLAGS:
        agg = _chunk_agg(chunk_means[arm_name])
        arms_out[arm_name] = {
            "RHS_w": agg["mean"],
            "SE": agg["se"],  # PA-CA-1(a): SE over chunk means, not per-event variance/n
            "n_chunks": agg["n_chunks"],
            "n_accepted_total": int(running[arm_name]["n_accepted"]),
        }

    dc_agg = _chunk_agg(dc_chunk_means)
    br_agg = _chunk_agg(br_chunk_means)
    kappa_hat = (
        dc_agg["mean"] / arms_out["twin"]["RHS_w"] if arms_out["twin"]["RHS_w"] else float("nan")
    )

    tci_out: dict[str, dict[str, Any]] = {}
    for arm_name in ARRANGEMENT_FLAGS:
        tci_out[arm_name] = {
            f"tau_{int(tau)}": _chunk_agg(tci_chunk_means[arm_name][tau]) for tau in C_TCI_TAUS
        }

    result: dict[str, Any] = {
        "reference": f"{REGISTRATION_SECTION}, RHS (--stage score, PA-CA-1/2/4/7b amended)",
        "h_gen": H_GEN,
        "r_h_gen": r_h,
        "catalogue_global_selection_slot": CATALOGUE_GLOBAL_SELECTION_SLOT,
        "selection_in_completion_numerator": SELECTION_IN_COMPLETION_NUMERATOR,
        "a22_stamp": stamp,
        "a22_flags": {  # PA-CA-6: all three RESOLVED flag values, never "auto"
            "catalogue_global_selection": CATALOGUE_GLOBAL_SELECTION_SLOT,
            "selection_in_completion_numerator": SELECTION_IN_COMPLETION_NUMERATOR,
            "catalogue_numerator_survival": dict(ARRANGEMENT_FLAGS),
        },
        "n_events_target": n_events_target,
        "n_drawn_total": n_drawn_total,
        "n_syn_total_drawn": n_drawn_total,  # PA-CA-1(a) wording alias
        "base_seed": base_seed,
        "chunk_size": chunk,
        "registered_chunk_size": REGISTERED_CHUNK_SIZE,
        "chunk_size_is_registered": chunk == REGISTERED_CHUNK_SIZE,
        "n_chunks": chunk_idx,
        "per_chunk": per_chunk,
        "arms": arms_out,
        "D_C_accumulator": dc_agg,  # PA-CA-2: E_Gbar[W~.w_BC.1_acc]
        "kappa_hat": kappa_hat,  # PA-CA-2: E_Gbar[W~.w_BC.1_acc] / E_Gbar[w_BT.1_acc]
        "RHS_BR": br_agg,  # PA-CA-4: E_Gbar[w_BT/(1+(r-1)w_BT).1_acc]
        "C_TCI_indicator_profile": tci_out,  # PA-CA-7(b)
        "c_tci_taus": list(C_TCI_TAUS),
    }
    result["elapsed_s"] = time.time() - t0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print("=== [P3-IMP] ca_rhs_scorer -- RHS (--stage score, PA-CA-1/2/4/7b amended) ===")
    print(f"n_drawn_total={n_drawn_total} (target={n_events_target}), n_chunks={chunk_idx}")
    if chunk != REGISTERED_CHUNK_SIZE:
        print(
            f"WARNING: chunk_size={chunk} != REGISTERED_CHUNK_SIZE={REGISTERED_CHUNK_SIZE} -- "
            "NON-REGISTERED run (--unsafe-chunk-size), not valid for any banked verdict.",
            file=sys.stderr,
        )
    for arm_name, r in arms_out.items():
        print(
            f"  RHS_w({arm_name}) = {r['RHS_w']!r}  SE={r['SE']!r}  "
            f"n_accepted_total={r['n_accepted_total']}  n_chunks={r['n_chunks']}"
        )
    print(f"D_C accumulator (E_Gbar[W~.w_BC.1_acc]) = {dc_agg['mean']!r}  SE={dc_agg['se']!r}")
    print(f"kappa_hat = {kappa_hat!r}")
    print(f"RHS_BR (E_Gbar[w_BT/(1+(r-1)w_BT).1_acc]) = {br_agg['mean']!r}  SE={br_agg['se']!r}")
    for arm_name in ARRANGEMENT_FLAGS:
        print(f"C-TCI indicator profile ({arm_name}):")
        for tau in C_TCI_TAUS:
            e = tci_out[arm_name][f"tau_{int(tau)}"]
            print(f"  tau={tau:.0f}: mean={e['mean']!r} se={e['se']!r}")
    print(f"elapsed = {result['elapsed_s']:.1f} s")
    print(f"wrote {out_path}")
    return result


def _cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("score", "acceptance", "fidelity", "lhs", "manifest", "determinism"),
        default="score",
    )
    parser.add_argument(
        "--n-events",
        type=int,
        default=5000,
        help="--stage score: TOTAL drawn synthetic events target (chunked; accepted subset is "
        "smaller, F-0-filtered).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SCORE_BASE_SEED,
        help="--stage score: base seed for the chunk-seed sequence (chunk i uses seed+i).",
    )
    parser.add_argument(
        "--unsafe-chunk-size",
        type=int,
        default=None,
        help="NON-REGISTERED escape hatch (PA-CA-8): override the hard-pinned chunk size "
        f"({REGISTERED_CHUNK_SIZE}, the venue's own per-realization draw count) for --stage "
        "score/acceptance. Any value here voids the registered venue-realization-law fidelity; "
        "loudly disclosed at runtime, never a valid input to a banked verdict.",
    )
    parser.add_argument(
        "--n-mc",
        type=int,
        default=200_000,
        help="--stage acceptance: total MC draws per class (G/Gbar), chunked.",
    )
    parser.add_argument("--mc-seed", type=int, default=DEFAULT_MC_BASE_SEED)
    parser.add_argument(
        "--check-seeds",
        type=str,
        default=",".join(str(s) for s in BANKED_SEEDS),
        help="--stage acceptance: comma-separated banked seeds to check n_kept against the "
        "model's binomial band (default: all 12).",
    )
    parser.add_argument("--fidelity-seed", type=int, default=900101)
    parser.add_argument(
        "--determinism-seed",
        type=int,
        default=900101,
        help="--stage determinism: seed for the PA-CA-9 cold/warm-cache draw comparison.",
    )
    parser.add_argument("--out-root", type=str, default=str(OUT_ROOT_DEFAULT))
    parser.add_argument("--out", type=str, default=None, help="Override the output JSON path.")
    args = parser.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # PA-CA-8: hard pin, disclosed escape hatch only.
    if args.unsafe_chunk_size is not None:
        chunk_size = args.unsafe_chunk_size
        print(
            f"WARNING: --unsafe-chunk-size={chunk_size} overrides the PA-CA-8 registered pin "
            f"({REGISTERED_CHUNK_SIZE}) -- NON-REGISTERED, not a valid input to any banked "
            "verdict.",
            file=sys.stderr,
        )
    else:
        chunk_size = REGISTERED_CHUNK_SIZE

    if args.stage == "lhs":
        out_path = Path(args.out) if args.out else out_root / "ca_rhs_lhs_output.json"
        stage_lhs(out_path)
        return 0

    if args.stage == "manifest":
        out_path = (
            Path(args.out) if args.out else BANKED_B0I_META_ROOT / "ca_lhs_manifest_20260824.json"
        )
        result = stage_manifest(out_path)
        return 0 if not result["missing"] else 1

    if args.stage == "determinism":
        result = _determinism_check(args.determinism_seed)
        return 0 if result["pass"] else 1

    if args.stage == "acceptance":
        out_path = Path(args.out) if args.out else out_root / "ca_rhs_acceptance_output.json"
        check_seeds = [int(x) for x in args.check_seeds.split(",") if x]
        result = stage_acceptance(args.n_mc, args.mc_seed, check_seeds, out_path, chunk_size)
        return 0 if result["gate_acc_verdict"] == "PASS" else 1

    if args.stage == "fidelity":
        out_path = Path(args.out) if args.out else out_root / "ca_rhs_fidelity_output.json"
        result = stage_fidelity(args.fidelity_seed, out_root, out_path)
        return 0 if result["gate_rhsf_verdict"] == "PASS" else 1

    out_path = Path(args.out) if args.out else out_root / "ca_rhs_score_output.json"
    stage_score(args.n_events, args.seed, out_root, out_path, chunk_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
