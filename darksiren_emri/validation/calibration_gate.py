r"""Calibration-gate instrument — P–P/coverage leg + multi-candidate host balls + σ–d_L texture.

**What this instrument is.** The stage-4 calibration gate, v1-registered in
``results/calibration_gate_20260808/PREREGISTRATION_CALIBRATION_GATE.md``
(commit ``b50ccc65``, bands locked blind BEFORE this build) and re-registered
as v2 in
``results/calibration_gate_v2_20260810/PREREGISTRATION_CALIBRATION_GATE_V2.md``
after v1 fired GATE-NOT-TRUSTWORTHY on its own validity checks (see the v2
divergence log below): the missing input
of ``docs/RESEARCH_CYCLE.md`` §Stage 4 leg 1 — "SBC / P–P coverage of the FULL
two-channel estimator on truth-known synthetic universes at the production
venue" — built as a **thin extension module over two existing instruments,
modifying neither**:

* :mod:`darksiren_emri.validation.closed_loop_gfrac` (registered
  instrument, code ``77b524af``) is imported **as a library**: universe
  generator (production ``phi`` with the ``kappa_cap`` kink, ``w_pop``,
  ``S_4D`` Bernoulli selection, CRB-bootstrap noise), estimator quadrature
  (per-h ``g_i`` verbatim, shared ``alpha(h)``), the canonical 41-point h
  grid, and the worker-pool sweep pattern. A3(i) (2-channel, ``g`` recomputed
  per h) and A3(ii) (production ``N_det = 1500``) are **inherited** from it.
* :mod:`darksiren_emri.validation.pp_coverage` supplies the HPD
  credible-region test as a **certified port** (:func:`hpd_contains`, ~16
  dependency-free lines) and the impostor-ball *design pattern* (its
  ``SyntheticCatalogue`` is 1D-only and production-independent, so its code is
  not imported — the runtime module stays independent of ``pp_coverage``; the
  port is certified boolean-exactly against the original in
  ``darksiren_emri_test/validation/test_calibration_gate.py`` (V2)).

New capabilities (prereg §4): the P–P/coverage readout (PIT + 50/68/90 % HPD
containment + KS distance, §4.1), multi-candidate host balls (A3-iii, a
redshift-window Poisson caricature of the localisation cone, §4.2), and the
σ–d_L joint texture (decile rank-matching of the production CRB triples,
§4.3). Cells, seed blocks, decision statistics DS-1…DS-7, the edge guard, and
every band are fixed in the prereg and mirrored here as constants — nothing is
tuned on data this instrument produces.

**Deliberate divergences from the parents (documented, per the build mandate):**

1. *Placement*: the module lives here (prereg §0 names exactly this path);
   ``results/calibration_gate_20260808/calibration_gate.py`` is a thin CLI
   shim delegating to :func:`main`, so the results directory is self-driving.
2. *``dl_binned`` texture is a post-draw override*: :func:`draw_universe_gate`
   first calls the parent's ``draw_universe`` verbatim (bit-compatible when
   ``sigma_texture="independent"``), then — for ``dl_binned`` — replaces the
   independently drawn triples with decile rank-matched ones and redraws the
   observation noise from the same seeded stream. The parent's accept/reject
   selection machinery is reused untouched.
3. *Impostor z table extends beyond ``z_max_true``*: the ball window
   ``W_i`` is built from ``d_L_obs (1 ± 4 σ_dL)`` and observation noise can
   push its upper edge past the detection horizon; ``w_pop`` is a population
   density defined there, so the impostor inverse-CDF table covers the
   noise-widened envelope (the ball is a cut of the population field, not of
   the detected set).
4. *Ball numerators carry no ``w_pop`` and no selection factor* — that is the
   registered production kernel form (prereg §4.2, "that mismatch **is** the
   production kernel form, and measuring its in-loop calibration is the
   point"), unlike the parent's single-host completion branch which carries
   ``w_pop``. Both subtract ``N_ok · ln alpha(h)``.
5. *Degenerate windows*: events whose clipped ``W_i`` has zero population mass
   get ``n_impostors = 0`` for that event (rare; counted in the ball stats).
6. *Dirty-tree STOP with an explicit escape*: the prereg mandates STOP on a
   dirty tree; ``--allow-dirty`` exists for smoke/dev runs only and is
   recorded in the output JSON. Registered cells must run clean.
7. *V4 risk found at build time, not silently fixed*: decile rank-matching
   applied to the CRB CSV itself attenuates ``corr(ln sigma_dL/d_L, ln d_L)``
   from 0.816 to ≈ 0.69 ± 0.02 (20-replica measurement), i.e. *below* the
   registered V4 band ``0.82 ± 0.10``. The decile count is locked by the
   prereg and is NOT changed here; V4 is measured on the synthetic detected
   set and reported — if it fails, the texture cells are void per §10.
8. *Smoke* mirrors the parent's pattern (3 seeds, ``n_events = 300``,
   1 worker) and additionally re-runs the first seed to spot-check V3
   determinism; the registered 10-seeds/cell smoke is reachable via
   ``--n-seeds``.
9. *DS-7 structural undercount found at build time*: the parent's
   ``draw_universe`` counts proposals in whole 4096-batches and truncates
   accepted overshoot, biasing the registered raw DS-7 ratio by ≈ −5 % at
   (p_bar ≈ 0.095, N = 1500) — the size of the blind-locked band. The raw
   registered statistic is reported unchanged; a granularity-corrected
   companion is reported alongside (see :func:`ds7_accounting`); which one
   carries V-class weight is an author call for the prereg appendix.
10. *Ball normalisation uses the prereg's own formula, NOT the parent's
    dropout convention*: the prereg §4.2 posterior is
    ``ln P(h) = sum_i ln L_i(h) - N_det ln alpha(h)`` with ``N_det`` fixed;
    an event whose candidates all fall outside the window at some h has
    ``L_i = 0`` and must EXCLUDE that h (finite ``-745``/event penalty as
    the JSON-safe ``-inf`` stand-in). The parent's ``N_ok`` convention
    (drop the event, subtract only ``N_ok ln alpha``) is correct in the
    inherited single-host path, where the numerator carries ``w_pop`` and
    windows are never empty, but in the bare-kernel ball path it rewards
    dropout and rails the posterior to the edge that invalidates every
    event — caught by the V1 smoke, fixed to the registered formula.
    Registered tension left for the author: a true host scattered ``>4
    sigma`` in ``d_L`` (per-event probability ~6e-5, so ~9 % of N = 1500
    seeds contain one) is outside its own window at the truth, so the V1
    "MAP = 0.730 exactly, every seed" expectation can fail for understood
    tail-``epsilon`` reasons, not plumbing; reported per seed, never
    patched around.

**v2 divergence log (2026-08-10, prereg
``results/calibration_gate_v2_20260810/PREREGISTRATION_CALIBRATION_GATE_V2.md``).**
The v1 campaign fired GATE-NOT-TRUSTWORTHY on its own validity checks (readout
``CALIBRATION_GATE_READOUT_20260808.md``, adjudication confirmed) — the correct
honest outcome. v2 repairs the five enumerated defects; every instrument-side
edit is logged here (items 11-17). v1 artifacts under
``results/calibration_gate_20260808/`` are untouched and remain the committed
v1 record. All v2 design choices are AUTHOR-RATIFY (author autonomy mandate;
the final gate verdict is the author's).

11. *V4 band re-derived from the pre-declared analysis (defect 1)*: the v1
    band ``0.82 ± 0.10`` was mis-set against this module's own item-7
    build-time decile-attenuation analysis (predicted ``0.69 ± 0.02``,
    20-replica SD); v1 measured 0.664-0.666 and V4 fired. The v2 band is
    derived FROM the pre-declared prediction and its stated uncertainty:
    ``0.69 ± 3 x 0.02 = [0.63, 0.75]`` (3-sigma of the replica scatter; the
    detected-set restriction adds attenuation of the same order, covered by
    the 3-sigma width). The v1 measured value is cited only as post-hoc
    consistency (0.664-0.666 lies inside the derived band); it is NOT the
    source of the band.
12. *DS-7 demoted to REPORT-ONLY in both forms (defect 2)*: the registered v1
    raw form violated 6/9 and is MC-seed-fragile (adjudication: 8/9 under
    another MC seed — the band edge sits inside the p_bar MC noise); the
    granularity-corrected form passed 9/9. The form choice was reserved to the
    author (item 9) and remains OPEN. In v2 neither form carries V-class or
    branch weight; both are emitted, now with ``p_bar_mc_se`` so the fragility
    is quantified in the record. DS-7 is removed from the v2
    GATE-NOT-TRUSTWORTHY trigger set (prereg v2 §10).
13. *Degenerate-PIT exemption marker (defect 3)*: at ``sigma_z = 0`` the ball
    posterior is near-delta at truth, so PIT = 0.5, C_beta = 1, KS D = 0.5 by
    construction and DS-1/DS-2 labels are structurally meaningless. B0/V1 are
    plumbing/validity controls scored only on their V-checks and DS-3/DS-4
    (prereg v2 registered exemption); :func:`aggregate_gate` emits
    ``ds1_ds2_degenerate_pit_exempt`` for ball cells at ``sigma_z = 0``.
14. *A-cell extended h grid (defect 4)*: v1 A-2D was 91-93 % edge-loaded at
    all three truths — on the 0.600-0.860 grid no truth placement can clear
    the edges (needed span ≈ bias + 3 x map_sd + 2.33 x post_sd per side
    ≈ 0.31 > 0.26 available), so the fix is :data:`EXTENDED_H_GRID_A`
    (75 points: 0.460-0.590 and 0.870-1.060 at the 0.01 wing spacing around
    the canonical 41). The canonical grid is a strict subgrid, so the
    v1-comparable restricted read (argmax/PIT over 0.600-0.860) stays
    mechanical at readout. B0/B1/B2/V1 keep the canonical grid unchanged
    (v1-comparability for the reproduction targets).
15. *Clean rule enforced (defect 5)*: v1 ran all registered cells
    ``--allow-dirty`` contra its own STOP clause. v2 rule (precise,
    enforceable): ZERO uncommitted changes — modified or untracked — under
    ``darksiren_emri/`` and ``darksiren_emri_test/`` (the import
    path); the runner REFUSES otherwise (:func:`_enforce_clean_import_path`);
    ``--allow-dirty`` is accepted ONLY with ``--smoke`` or ``--validate``
    (never on a registered cell run); the full dirt inventory of everything
    else (:func:`_classify_porcelain`) is recorded in every output JSON and
    never blocks.
16. *v2 seed plan (fresh registered sample)*: all v1 offsets shifted by
    +20000 (A: 20000/21000/22000, B0: 23000, B1: 24000, B2:
    25000/26000/27000, O1 reserved: 28000, V1: 29000) — the v2 absolute seed
    set is disjoint from v1's ``[20260808, 20269857]`` by construction
    (tested). v2 is a re-registration with disjoint seeds, not a re-score of
    v1 seeds.
17. *Registration identity*: ``PREREG_PATH``/``DEFAULT_OUT_DIR`` now name the
    v2 registration; the module and its tests are committed IN the
    registering commit, so the v1 gap (instrument untracked at run time,
    prereg §11 empty) cannot recur.

**What this module never does**: import ``BayesianStatistics``; modify either
parent; produce a production posterior (every posterior is a synthetic-universe
diagnostic, quotable only against its own truth); adjudicate the branch call
(band comparisons are mechanical; the branch is presented to the author).

Cell O1 (``volume``-kernel arm) is **not built**: per prereg §9 item 6 the
kernel-form sensitivity is NOT-EVALUABLE and the CLI says so explicitly.

CPU-only. No cupy import, direct or transitive.

References:
    Mandel, Farr & Gair (2019), arXiv:1809.02063, Eqs. (5)-(7).
    Gray et al. (2020), arXiv:1908.06050, Eqs. (29), (32).
    Cook, Gelman & Rubin (2006), J. Comp. Graph. Stat. 15(3) — PIT/SBC logic.
    Talts et al. (2018), arXiv:1804.06788 — simulation-based calibration.
"""

import argparse
import json
import logging
import math
import multiprocessing as mp
import os
import subprocess
import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.stats import norm

from darksiren_emri.bayesian_inference.bayesian_statistics import (
    completion_mass_factor_g,
)
from darksiren_emri.physical_relations import dist_vectorized
from darksiren_emri.validation import closed_loop_gfrac as cl

_LOGGER = logging.getLogger(__name__)

PREREG_PATH = "results/calibration_gate_v2_20260810/PREREGISTRATION_CALIBRATION_GATE_V2.md"
PREREG_V1_PATH = "results/calibration_gate_20260808/PREREGISTRATION_CALIBRATION_GATE.md"
R0_RESULTS_JSON = "results/closed_loop_gfrac_20260805/closed_loop_results.json"
DEFAULT_OUT_DIR = "results/calibration_gate_v2_20260810"
GATE_BASE_SEED = 20260808  # seed-plan base (v2 blocks are base + v1 offsets + 20000)
PARENT_CLOSED_LOOP_COMMIT = "77b524af"

# v1 absolute-seed envelope [base+0, base+9049] — the v2 offsets (+20000) are
# disjoint from it by construction (divergence 16; tested).
V1_SEED_OFFSET_ENVELOPE: tuple[int, int] = (0, 9049)

HPD_LEVELS: tuple[float, ...] = (0.50, 0.68, 0.90)
_IMPOSTOR_KERNEL_WINDOW = 5.0  # per-candidate ±5 sigma_z kernel window (prereg §4.2)
_N_DECILES = 10  # prereg §4.3 — locked; see module docstring divergence 7
_DS7_N_MC = 1_000_000  # prereg §7 DS-7 fresh-proposal MC size
_DS7_MC_SEED = 20260808
_DS7_BAND = 0.05  # v2: REPORT-ONLY in both forms (divergence 12); no branch weight
EDGE_MASS_THRESHOLD = 0.01  # prereg §8
EDGE_CONTAMINATION_FRACTION = 0.10  # prereg §8
_LN_ZERO_EVENT = -745.0  # ln of the smallest positive double: JSON-safe -inf stand-in
_KS_C95 = 1.358  # prereg §7 DS-2
_KS_C99 = 1.628
_DS3_IN_BAND = 0.010  # prereg §7 DS-3 (closed-loop §6 frozen edges)
_DS3_DEFECT = 0.030
# V4 band (v2, divergence 11): derived from the PRE-DECLARED build-time
# decile-attenuation analysis (divergence 7: rank-matching attenuates the
# CSV's 0.816 to 0.69 ± 0.02, 20-replica SD) — band = prediction ± 3 x SD.
# The v1 measured 0.664-0.666 is post-hoc consistency only, not the source.
_V4_CORR_CENTER = 0.69
_V4_CORR_TOL = 0.06  # 3 x 0.02 replica SD
_V5_RTOL = 1e-12  # prereg §10 V5
_NONFINITE_ABORT_FRACTION = 0.01  # prereg §10 abort (b)

# Import path whose cleanliness the v2 clean rule enforces (divergence 15).
_IMPORT_PATH_PREFIXES: tuple[str, ...] = ("darksiren_emri/", "darksiren_emri_test/")

# Extended A-cell h grid (v2, divergence 14): canonical 41 points plus 0.01-
# spaced wings 0.460-0.590 (14 points) and 0.870-1.060 (20 points) = 75 points.
# Derivation from v1 committed A-2D numbers (prereg v2 §D4): needed clearance
# per side ≈ bias + 3 x map_sd + 2.33 x post_sd ≈ 0.04 + 0.21 + 0.12 ≈ 0.31
# exceeds the 0.26 canonical span, so no truth placement can fix defect (4);
# the grid must extend. The canonical grid is a strict subgrid (restricted
# v1-comparable reads stay mechanical).
EXTENDED_H_GRID_A: tuple[float, ...] = (
    tuple(round(0.460 + 0.010 * i, 3) for i in range(14))
    + cl.CANONICAL_H_GRID
    + tuple(round(0.870 + 0.010 * i, 3) for i in range(20))
)


# ── Cell registry (prereg §5, verbatim) ──────────────────────────────────────


@dataclass(frozen=True)
class CellSpec:
    """One prereg §5 cell: configuration + truths + seed blocks.

    Attributes:
        name: Cell id (prereg §5 table).
        ball: Whether the multi-candidate ball code path is used.
        lambda_ball: Poisson mean of the impostor count per event.
        sigma_z: Flat per-candidate photo-z scatter.
        sigma_texture: ``"dl_binned"`` or ``"independent"`` (prereg §4.3).
        truths: Injected ``h_true`` values.
        n_seeds: Registered seeds per truth (400; V1: 50).
        seed_offsets: Per-truth offsets from ``GATE_BASE_SEED`` (disjoint
            blocks; a seed appears in exactly one cell; v2 offsets are the v1
            offsets + 20000, disjoint from every v1 seed — divergence 16).
        h_grid: The cell's h grid (A: :data:`EXTENDED_H_GRID_A`, divergence
            14; all other cells: the canonical 41-point grid, unchanged for
            v1-comparability).
    """

    name: str
    ball: bool
    lambda_ball: float
    sigma_z: float
    sigma_texture: str
    truths: tuple[float, ...]
    n_seeds: int
    seed_offsets: tuple[int, ...]
    h_grid: tuple[float, ...] = cl.CANONICAL_H_GRID


CELL_SPECS: dict[str, CellSpec] = {
    "A": CellSpec(
        "A",
        False,
        0.0,
        0.0,
        "dl_binned",
        (0.690, 0.730, 0.770),
        400,
        (20000, 21000, 22000),
        EXTENDED_H_GRID_A,
    ),
    "B0": CellSpec("B0", True, 4.0, 0.0, "dl_binned", (0.730,), 400, (23000,)),
    "B1": CellSpec("B1", True, 4.0, 0.010, "dl_binned", (0.730,), 400, (24000,)),
    "B2": CellSpec(
        "B2", True, 4.0, 0.035, "dl_binned", (0.690, 0.730, 0.770), 400, (25000, 26000, 27000)
    ),
    "V1": CellSpec("V1", True, 0.0, 0.0, "dl_binned", (0.730,), 50, (29000,)),
}
# O1 (v2 offset 28000) is registered but NOT built — NOT-EVALUABLE per prereg §9 item 6.


@dataclass(frozen=True)
class GateConfig:
    """Frozen configuration of one calibration-gate cell run.

    Attributes:
        cell: Cell id ("A", "B0", "B1", "B2", "V1", or "custom").
        h_true: Injected Hubble parameter of this cell×truth.
        ball: Multi-candidate ball path on/off (off = the parent's single-host
            completion branch, called verbatim).
        lambda_ball: Poisson mean impostor count (prereg §4.2, registered 4).
        sigma_z: Flat per-candidate photo-z scatter (prereg §4.2).
        sigma_texture: ``"independent"`` (bit-compatible with the registered
            closed-loop draw) or ``"dl_binned"`` (prereg §4.3 rank-matching).
        f_incl: Host-inclusion probability; 1.0 registered (host always in
            the ball — incompleteness is out of scope v1, prereg §3).
        n_events: Detections per universe (production venue N).
        injection_data_dir: Injection pool defining ``S_4D``.
        crb_reference_csv: Production prepared-CRB CSV (noise + texture).
        h_grid: The cell's h grid — canonical 41 points, except A cells use
            :data:`EXTENDED_H_GRID_A` (75 points, divergence 14).
    """

    cell: str
    h_true: float
    ball: bool
    lambda_ball: float
    sigma_z: float
    sigma_texture: str = "dl_binned"
    f_incl: float = 1.0
    n_events: int = cl.DEFAULT_N_EVENTS
    injection_data_dir: str = cl.DEFAULT_INJECTION_DIR
    crb_reference_csv: str = cl.DEFAULT_CRB_CSV
    h_grid: tuple[float, ...] = cl.CANONICAL_H_GRID


def to_closed_loop_config(gcfg: GateConfig) -> cl.ClosedLoopConfig:
    """Project a :class:`GateConfig` onto the parent's configuration.

    ``f_cat = 0.0`` and ``numerator_pdet = "off"`` are pinned: the gate always
    runs the shipped-estimator convention (prereg §5).

    Args:
        gcfg: The gate configuration.

    Returns:
        The equivalent :class:`~darksiren_emri.validation.closed_loop_gfrac.ClosedLoopConfig`.
    """
    return cl.ClosedLoopConfig(
        injection_data_dir=gcfg.injection_data_dir,
        crb_reference_csv=gcfg.crb_reference_csv,
        n_events=gcfg.n_events,
        h_true=gcfg.h_true,
        h_grid=gcfg.h_grid,
        f_cat=0.0,
        numerator_pdet="off",
    )


# ── P–P / HPD readout layer (prereg §4.1) ────────────────────────────────────


def hpd_contains(
    h_grid: npt.NDArray[np.float64],
    post: npt.NDArray[np.float64],
    h_true: float,
    level: float,
) -> bool:
    """True if ``h_true`` lies inside the HPD credible region of mass ``level``.

    Verbatim port of ``pp_coverage._hpd_contains`` (prereg §3 table row 2);
    certified boolean-exactly against the original by the V2 unit test. The
    runtime module deliberately does not import ``pp_coverage``.

    Args:
        h_grid: The h grid.
        post: Normalised posterior density on that grid.
        h_true: The injected truth.
        level: Credible mass (e.g. 0.90).

    Returns:
        Containment boolean.
    """
    dh = np.gradient(h_grid)
    mass = post * dh
    order = np.argsort(post)[::-1]
    csum = np.cumsum(mass[order])
    k = int(np.searchsorted(csum, level))
    k = min(k, order.size - 1)
    thresh = float(post[order[k]])
    p_true = float(np.interp(h_true, h_grid, post))
    return p_true >= thresh


def pp_readout(
    h_grid: npt.NDArray[np.float64],
    ln_post: npt.NDArray[np.float64],
    h_true: float,
) -> dict[str, float]:
    """PIT, HPD containment, posterior sd, and edge mass of one posterior.

    Prereg §4.1: from the 41-point unnormalised ``ln P(h)``
    (trapezoid-normalised on the grid) compute the PIT
    ``q = INTEGRAL_{0.600}^{h_true} P(h) dh``, the 50/68/90 % HPD containment
    booleans, the grid-moment posterior sd (DS-5), and the edge mass
    ``E`` = mass in the first plus last grid interval (§8 guard).

    Args:
        h_grid: The h grid.
        ln_post: Unnormalised log posterior on the grid.
        h_true: The injected truth.

    Returns:
        ``{"pit", "hpd50", "hpd68", "hpd90", "post_sd", "edge_mass"}``
        (HPD booleans as 0.0/1.0 for JSON friendliness).
    """
    finite = np.isfinite(ln_post)
    if not np.all(finite):
        return {
            "pit": float("nan"),
            "hpd50": float("nan"),
            "hpd68": float("nan"),
            "hpd90": float("nan"),
            "post_sd": float("nan"),
            "edge_mass": float("nan"),
        }
    p = np.exp(ln_post - float(np.max(ln_post)))
    norm_c = float(np.trapezoid(p, h_grid))
    post = p / norm_c
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (post[1:] + post[:-1]) * np.diff(h_grid))])
    pit = float(np.interp(h_true, h_grid, cum))
    mean = float(np.trapezoid(post * h_grid, h_grid))
    var = float(np.trapezoid(post * h_grid**2, h_grid)) - mean**2
    edge = float(cum[1] + (cum[-1] - cum[-2]))
    out: dict[str, float] = {
        "pit": pit,
        "post_sd": math.sqrt(max(var, 0.0)),
        "edge_mass": edge,
    }
    for lv in HPD_LEVELS:
        out[f"hpd{int(round(lv * 100))}"] = float(hpd_contains(h_grid, post, h_true, lv))
    return out


def ks_distance(pits: npt.NDArray[np.float64]) -> float:
    """One-sample KS distance of a PIT sample against Uniform(0, 1) (DS-2).

    Args:
        pits: PIT values in [0, 1].

    Returns:
        ``D_N = sup |ECDF(q) - q|``.
    """
    q = np.sort(np.asarray(pits, dtype=np.float64))
    n = q.size
    if n == 0:
        return float("nan")
    i = np.arange(1, n + 1, dtype=np.float64)
    return float(np.max(np.maximum(i / n - q, q - (i - 1.0) / n)))


# ── σ–d_L joint texture (prereg §4.3) ────────────────────────────────────────


def load_sigma_triples_with_dl(
    csv_path: str,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """CRB error triples plus each row's ``d_L`` (for decile rank-matching).

    Applies exactly the parent's row filter (``cl.load_sigma_triples``); a unit
    test asserts the triples returned here equal the parent's array.

    Args:
        csv_path: Path to a ``prepared_cramer_rao_bounds.csv``.

    Returns:
        ``(triples, d_L)``: triples shape ``(n_rows, 3)`` as the parent's;
        ``d_L`` shape ``(n_rows,)`` aligned with the triples.
    """
    import pandas as pd

    df = pd.read_csv(csv_path)
    d_L = np.asarray(df["luminosity_distance"], dtype=np.float64)
    M = np.asarray(df["M"], dtype=np.float64)
    s_d = (
        np.sqrt(
            np.asarray(df["delta_luminosity_distance_delta_luminosity_distance"], dtype=np.float64)
        )
        / d_L
    )
    s_m = np.sqrt(np.asarray(df["delta_M_delta_M"], dtype=np.float64)) / M
    cov = np.asarray(df["delta_luminosity_distance_delta_M"], dtype=np.float64) / d_L / M
    with np.errstate(divide="ignore", invalid="ignore"):
        rho = cov / (s_d * s_m)
    ok = np.isfinite(s_d) & np.isfinite(s_m) & np.isfinite(rho)
    ok &= (s_d > 0.0) & (s_m > 0.0) & (np.abs(rho) < 1.0)
    triples = np.column_stack([s_d[ok], s_m[ok], rho[ok]]).astype(np.float64)
    if triples.shape[0] == 0:
        raise ValueError(f"No usable CRB error triples in '{csv_path}'")
    return triples, d_L[ok].astype(np.float64)


# ── Context ──────────────────────────────────────────────────────────────────


@dataclass
class GateContext:
    """Per-process shared, seed-independent tables for one cell×truth.

    Wraps the parent's :class:`~darksiren_emri.validation.closed_loop_gfrac.ClosedLoopContext`
    (all production objects live there) and adds the gate-only tables.
    """

    gate_config: GateConfig
    cl_ctx: cl.ClosedLoopContext
    csv_dl_sorted: npt.NDArray[np.float64]
    triples: npt.NDArray[np.float64]
    decile_rows: list[npt.NDArray[np.int64]]
    imp_z_nodes: npt.NDArray[np.float64]
    imp_z_cdf: npt.NDArray[np.float64]
    imp_dl_nodes: npt.NDArray[np.float64]


def build_gate_context(gcfg: GateConfig) -> GateContext:
    """Build the parent context plus the texture and impostor tables.

    Args:
        gcfg: The gate configuration.

    Returns:
        A ready :class:`GateContext`.
    """
    cl_ctx = cl.build_context(to_closed_loop_config(gcfg))
    triples, dl = load_sigma_triples_with_dl(gcfg.crb_reference_csv)

    # Decile membership by rank within the CSV's own d_L distribution.
    n_rows = dl.size
    rank = np.argsort(np.argsort(dl))
    decile_of_row = np.clip((_N_DECILES * rank) // n_rows, 0, _N_DECILES - 1)
    decile_rows = [np.where(decile_of_row == b)[0].astype(np.int64) for b in range(_N_DECILES)]
    csv_dl_sorted = np.sort(dl)

    # Impostor z table: cover the observation-noise-widened +4 sigma window
    # (module docstring divergence 3). Envelope: d_L_obs <= d_L_true (1 + 5 s)
    # and the window edge multiplies by (1 + 4 s).
    s_max = float(np.max(triples[:, 0]))
    cover = (1.0 + _IMPOSTOR_KERNEL_WINDOW * s_max) * (1.0 + cl._SIGMA_WINDOW * s_max)
    cover = min(cover, 4.0)
    dl_max_true = float(
        np.asarray(dist_vectorized(np.asarray([cl_ctx.z_max_true]), h=gcfg.h_true))[0]
    )
    d_nodes_gen, z_nodes_gen = cl._z_of_dl_table(gcfg.h_true, 6.0)
    z_ext = float(np.interp(dl_max_true * cover, d_nodes_gen, z_nodes_gen))
    z_ext = min(max(z_ext, cl_ctx.z_max_true), 6.0)
    imp_z_nodes = np.linspace(1e-6, z_ext, cl._Z_TABLE_POINTS, dtype=np.float64)
    w = cl._w_pop(imp_z_nodes, gcfg.h_true)
    imp_z_cdf = np.concatenate([[0.0], np.cumsum(0.5 * (w[1:] + w[:-1]) * np.diff(imp_z_nodes))])
    imp_z_cdf /= imp_z_cdf[-1]
    imp_dl_nodes = np.asarray(dist_vectorized(imp_z_nodes, h=gcfg.h_true), dtype=np.float64)

    return GateContext(
        gate_config=gcfg,
        cl_ctx=cl_ctx,
        csv_dl_sorted=csv_dl_sorted,
        triples=triples,
        decile_rows=decile_rows,
        imp_z_nodes=imp_z_nodes,
        imp_z_cdf=imp_z_cdf,
        imp_dl_nodes=imp_dl_nodes,
    )


# ── Generator: texture + ball ────────────────────────────────────────────────


def draw_universe_gate(gctx: GateContext, rng: np.random.Generator) -> cl.SyntheticUniverse:
    """Draw one universe; apply the σ–d_L texture if configured.

    ``sigma_texture="independent"`` returns the parent's draw untouched
    (bit-compatible with the registered closed-loop behaviour, prereg §4.3).
    ``"dl_binned"`` replaces the triples with decile rank-matched draws
    (matching each event's true ``d_L`` empirical quantile within the CSV,
    nearest bin outside its range) and redraws the observation noise.

    Args:
        gctx: The gate context.
        rng: Seeded generator.

    Returns:
        The (possibly re-textured) synthetic universe.
    """
    uni = cl.draw_universe(gctx.cl_ctx, rng)
    if gctx.gate_config.sigma_texture == "independent":
        return uni
    if gctx.gate_config.sigma_texture != "dl_binned":
        raise ValueError(f"unknown sigma_texture '{gctx.gate_config.sigma_texture}'")

    n = uni.z_true.size
    q = np.searchsorted(gctx.csv_dl_sorted, uni.d_L_true, side="right") / gctx.csv_dl_sorted.size
    dec = np.clip((q * _N_DECILES).astype(np.int64), 0, _N_DECILES - 1)
    rows = np.empty(n, dtype=np.int64)
    for b in range(_N_DECILES):
        m = dec == b
        if np.any(m):
            pool = gctx.decile_rows[b]
            rows[m] = pool[rng.integers(0, pool.size, size=int(m.sum()))]
    sigma_dL = gctx.triples[rows, 0]
    sigma_Mz = gctx.triples[rows, 1]
    rho = gctx.triples[rows, 2]

    e1 = rng.standard_normal(n)
    e2 = rng.standard_normal(n)
    frac_d = sigma_dL * e1
    frac_m = sigma_Mz * (rho * e1 + np.sqrt(np.maximum(1.0 - rho**2, 0.0)) * e2)
    M_z_true = uni.M_true * (1.0 + uni.z_true)
    return cl.SyntheticUniverse(
        z_true=uni.z_true,
        M_true=uni.M_true,
        d_L_true=uni.d_L_true,
        d_L_obs=uni.d_L_true * (1.0 + frac_d),
        M_z_obs=M_z_true * (1.0 + frac_m),
        sigma_dL=sigma_dL,
        sigma_Mz=sigma_Mz,
        rho=rho,
        in_catalogue=uni.in_catalogue,
        n_drawn=uni.n_drawn,
    )


@dataclass
class HostBall:
    """The multi-candidate host balls of one universe (prereg §4.2).

    Flattened (event, candidate) representation; ``event_idx`` is
    nondecreasing and within each event the candidate order is shuffled (the
    estimator never learns which member is the host).
    """

    z_obs: npt.NDArray[np.float64]  # (n_pairs,)
    event_idx: npt.NDArray[np.int64]  # (n_pairs,), nondecreasing
    K: npt.NDArray[np.int64]  # (n_events,) candidates per event
    n_impostors_total: int
    n_degenerate_windows: int


def draw_ball(
    gctx: GateContext,
    universe: cl.SyntheticUniverse,
    rng: np.random.Generator,
) -> HostBall:
    """Draw the impostor balls for every event (prereg §4.2 steps 1–3).

    Window ``W_i = [z(d_L_obs (1 - 4 s); h_true), z(d_L_obs (1 + 4 s); h_true)]``
    on the truth ladder; ``n_i ~ Poisson(lambda_ball)`` impostors i.i.d.
    ``w_pop | W_i`` (Slivnyak–Mecke: the ball is a cut of the same field the
    host lives in); the ball is host + impostors, order shuffled; every member
    gets ``z_obs = z + sigma_z * eps`` with flat ``sigma_z``.

    Args:
        gctx: The gate context.
        universe: The detected event set.
        rng: Seeded generator.

    Returns:
        The :class:`HostBall`.
    """
    gcfg = gctx.gate_config
    n = universe.z_true.size

    d_lo = universe.d_L_obs * (1.0 - cl._SIGMA_WINDOW * universe.sigma_dL)
    d_hi = universe.d_L_obs * (1.0 + cl._SIGMA_WINDOW * universe.sigma_dL)
    z_lo = np.interp(np.maximum(d_lo, 0.0), gctx.imp_dl_nodes, gctx.imp_z_nodes)
    z_hi = np.interp(d_hi, gctx.imp_dl_nodes, gctx.imp_z_nodes)
    F_lo = np.interp(z_lo, gctx.imp_z_nodes, gctx.imp_z_cdf)
    F_hi = np.interp(z_hi, gctx.imp_z_nodes, gctx.imp_z_cdf)

    if gcfg.lambda_ball > 0.0:
        n_imp = rng.poisson(gcfg.lambda_ball, size=n).astype(np.int64)
    else:
        n_imp = np.zeros(n, dtype=np.int64)
    degenerate = F_hi <= F_lo
    n_imp[degenerate] = 0
    total_imp = int(n_imp.sum())

    imp_event = np.repeat(np.arange(n, dtype=np.int64), n_imp)
    u = rng.random(total_imp)
    u_scaled = F_lo[imp_event] + (F_hi[imp_event] - F_lo[imp_event]) * u
    z_imp = np.interp(u_scaled, gctx.imp_z_cdf, gctx.imp_z_nodes)

    z_all = np.concatenate([universe.z_true, z_imp])
    ev_all = np.concatenate([np.arange(n, dtype=np.int64), imp_event])
    # Shuffle within each event: sort by (event, random key).
    key = rng.random(z_all.size)
    order = np.lexsort((key, ev_all))
    z_all = z_all[order]
    ev_all = ev_all[order]

    if gcfg.sigma_z > 0.0:
        z_obs = z_all + gcfg.sigma_z * rng.standard_normal(z_all.size)
    else:
        z_obs = z_all.copy()

    K = np.bincount(ev_all, minlength=n).astype(np.int64)
    return HostBall(
        z_obs=z_obs,
        event_idx=ev_all,
        K=K,
        n_impostors_total=total_imp,
        n_degenerate_windows=int(degenerate.sum()),
    )


# ── Ball estimator (prereg §4.2, estimator side) ─────────────────────────────


def _g_ball(
    gctx: GateContext,
    universe: cl.SyntheticUniverse,
    event_idx: npt.NDArray[np.int64],
    z_nodes: npt.NDArray[np.float64],
    d_L_frac: npt.NDArray[np.float64],
    valid: npt.NDArray[np.bool_],
) -> npt.NDArray[np.float64]:
    """Production ``g_i(z;h)`` at every candidate node, grouped per event.

    Same conditional-Gaussian parameters as the parent's ``_g_at_nodes``
    (Bishop 2006 Eqs. 2.81-2.82 from the event's own 2x2 block); one verbatim
    :func:`completion_mass_factor_g` call per event over its valid candidates'
    flattened nodes.

    Args:
        gctx: The gate context.
        universe: The event set (per-event ``M_z_obs`` and 2x2 block).
        event_idx: Nondecreasing event index per candidate row.
        z_nodes: ``(n_pairs, n_quad)`` quadrature redshifts.
        d_L_frac: ``(n_pairs, n_quad)`` values of ``d_L(z;h)/d_L_obs``.
        valid: ``(n_pairs,)`` rows with a nonempty integration window.

    Returns:
        ``g`` at the nodes, shape ``(n_pairs, n_quad)``; invalid rows are 0.
    """
    n = universe.z_true.size
    s_dd = universe.sigma_dL**2
    s_dm = universe.rho * universe.sigma_dL * universe.sigma_Mz
    s_mm = universe.sigma_Mz**2
    proj = np.where(s_dd > 0.0, s_dm / np.maximum(s_dd, 1e-300), 0.0)
    sigma_cond = np.sqrt(np.maximum(s_mm - proj * s_dm, 1e-30))

    out = np.zeros_like(z_nodes)
    starts = np.searchsorted(event_idx, np.arange(n, dtype=np.int64), side="left")
    stops = np.searchsorted(event_idx, np.arange(n, dtype=np.int64), side="right")
    n_hermite = gctx.cl_ctx.config.n_hermite
    for i in range(n):
        rows = np.arange(starts[i], stops[i])
        rows = rows[valid[rows]]
        if rows.size == 0:
            continue
        zz = z_nodes[rows].reshape(-1)
        ff = d_L_frac[rows].reshape(-1)
        out[rows] = completion_mass_factor_g(
            zz,
            ff,
            float(universe.M_z_obs[i]),
            float(proj[i]),
            float(sigma_cond[i]),
            n_hermite=n_hermite,
        ).reshape(rows.size, z_nodes.shape[1])
    return out


def log_channel_posteriors_ball(
    gctx: GateContext,
    universe: cl.SyntheticUniverse,
    ball: HostBall,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    r"""Unnormalised log posteriors of both channels under multi-candidate balls.

    Prereg §4.2 estimator side (bare kernel × distance likelihood, equal
    candidate prior, no selection factor in the numerator, no ``w_pop`` —
    the registered production kernel form):

    .. math::

        L_i(h) = \frac{1}{K_i} \sum_k \int \mathrm{d}z\,
            \mathcal{N}(z; z_{obs,k}, \sigma_z)\,
            \mathcal{N}\!\bigl(d_L(z;h)/d_L^{obs}_i; 1, \sigma_{dL,i}\bigr)
            \,[\,g_i(z;h)\,]

    with per-candidate 50-node Gauss-Legendre on
    ``[max(z_lo(h), z_obs - 5 sigma_z), min(z_hi(h), z_obs + 5 sigma_z)]``,
    ``[z_lo, z_hi]`` the production ±4 sigma window capped at ``z_max(h)``;
    at ``sigma_z = 0`` a point evaluation at ``z_obs``. Both channels
    subtract ``N_det ln alpha(h)`` (N fixed, the prereg formula) with the
    parent's shared ``alpha``; a zero-likelihood event excludes that h via
    the finite ``_LN_ZERO_EVENT`` penalty (module docstring divergence 10).

    Args:
        gctx: The gate context.
        universe: The event set.
        ball: The candidate balls.

    Returns:
        ``(ln_post_1d, ln_post_2d, sum_dlog_gfrac_dh)`` exactly as the
        parent's ``log_channel_posteriors`` (the slope from joint-ok events).
    """
    cfg = gctx.cl_ctx.config
    gcfg = gctx.gate_config
    n_h = len(cfg.h_grid)
    n = universe.z_true.size
    ln1 = np.zeros(n_h, dtype=np.float64)
    ln2 = np.zeros(n_h, dtype=np.float64)
    ln_gfrac = np.zeros(n_h, dtype=np.float64)

    x = gctx.cl_ctx.gl_nodes
    w_gl = gctx.cl_ctx.gl_weights
    ev = ball.event_idx
    z_obs = ball.z_obs
    d_obs_e = universe.d_L_obs
    sig_e = universe.sigma_dL
    d_obs_p = d_obs_e[ev]
    sig_p = sig_e[ev]
    K = np.maximum(ball.K, 1)

    for k, h in enumerate(cfg.h_grid):
        d_L_nodes, z_tab = gctx.cl_ctx.z_of_dl_tables[k]
        z_hi_e = np.interp(d_obs_e * (1.0 + cl._SIGMA_WINDOW * sig_e), d_L_nodes, z_tab)
        z_lo_e = np.interp(d_obs_e * (1.0 - cl._SIGMA_WINDOW * sig_e), d_L_nodes, z_tab)
        z_lo_e = np.maximum(z_lo_e, 1e-6)
        z_hi_e = np.minimum(z_hi_e, z_tab[-1])
        z_lo_p = z_lo_e[ev]
        z_hi_p = z_hi_e[ev]

        if gcfg.sigma_z > 0.0:
            a = np.maximum(z_lo_p, z_obs - _IMPOSTOR_KERNEL_WINDOW * gcfg.sigma_z)
            b = np.minimum(z_hi_p, z_obs + _IMPOSTOR_KERNEL_WINDOW * gcfg.sigma_z)
            valid = b > a
            half = 0.5 * (b - a)
            mid = 0.5 * (b + a)
            z_nodes = mid[:, None] + half[:, None] * x[None, :]
            d_L_n = np.asarray(
                dist_vectorized(np.maximum(z_nodes.reshape(-1), 1e-8), h=h),
                dtype=np.float64,
            ).reshape(z_nodes.shape)
            d_L_frac = d_L_n / d_obs_p[:, None]
            p_gw = norm.pdf(d_L_frac, loc=1.0, scale=sig_p[:, None])
            kern = norm.pdf(z_nodes, loc=z_obs[:, None], scale=gcfg.sigma_z)
            integ = kern * p_gw
            c1 = half * (integ @ w_gl)
            g = _g_ball(gctx, universe, ev, z_nodes, d_L_frac, valid)
            c2 = half * ((integ * g) @ w_gl)
            c1 = np.where(valid, c1, 0.0)
            c2 = np.where(valid, c2, 0.0)
        else:
            valid = (z_obs >= z_lo_p) & (z_obs <= z_hi_p)
            d_pt = np.asarray(dist_vectorized(np.maximum(z_obs, 1e-8), h=h), dtype=np.float64)
            frac = d_pt / d_obs_p
            p_gw = norm.pdf(frac, loc=1.0, scale=sig_p)
            g_pt = _g_ball(gctx, universe, ev, z_obs[:, None], frac[:, None], valid)[:, 0]
            c1 = np.where(valid, p_gw, 0.0)
            c2 = np.where(valid, p_gw * g_pt, 0.0)

        L1 = np.bincount(ev, weights=c1, minlength=n) / K
        L2 = np.bincount(ev, weights=c2, minlength=n) / K
        # Prereg §4.2 normalisation, N_det FIXED: ln P = sum_i ln L_i - N ln a.
        # An event with L_i = 0 (all candidates outside the window) EXCLUDES
        # this h — represented by the finite -745/event penalty (ln of the
        # smallest double; JSON-safe stand-in for -inf). The parent's N_ok
        # dropout convention must NOT be used here: without w_pop the per-event
        # contributions are negative, so dropping events would RAISE ln P and
        # rail the posterior to whatever edge invalidates everything (found in
        # the V1 smoke; module docstring divergence 10).
        ok1 = (L1 > 0.0) & np.isfinite(L1)
        ok2 = (L2 > 0.0) & np.isfinite(L2)
        lnL1 = np.where(ok1, np.log(np.where(ok1, L1, 1.0)), _LN_ZERO_EVENT)
        lnL2 = np.where(ok2, np.log(np.where(ok2, L2, 1.0)), _LN_ZERO_EVENT)
        ln1[k] = float(np.sum(lnL1)) - float(n) * gctx.cl_ctx.log_alpha[k]
        ln2[k] = float(np.sum(lnL2)) - float(n) * gctx.cl_ctx.log_alpha[k]
        both = ok1 & ok2
        ln_gfrac[k] = float(np.sum(np.log(L2[both] / L1[both])))

    h_arr = np.asarray(cfg.h_grid, dtype=np.float64)
    i_true = int(np.argmin(np.abs(h_arr - cfg.h_true)))
    lo = max(i_true - 1, 0)
    hi = min(i_true + 1, n_h - 1)
    slope = (ln_gfrac[hi] - ln_gfrac[lo]) / (h_arr[hi] - h_arr[lo])
    return ln1, ln2, np.asarray([slope], dtype=np.float64)


# ── Per-seed driver ──────────────────────────────────────────────────────────

_GCTX: GateContext | None = None


def _gate_worker_init(gcfg: GateConfig) -> None:
    """Build (or inherit) the shared gate context in a worker process."""
    global _GCTX
    if _GCTX is None:
        _GCTX = build_gate_context(gcfg)


def run_seed_gate(seed: int, gctx: GateContext | None = None) -> dict[str, Any]:
    """Run one synthetic universe end to end and emit the §6 record.

    Args:
        seed: The universe's random seed.
        gctx: Shared context; falls back to the process-global one.

    Returns:
        A JSON-serialisable per-seed record: the parent's fields plus
        ``pit_*``, ``hpd*``, ``post_sd_*``, ``edge_mass_*``, ball statistics,
        texture correlation, ``sigma_texture`` and the cell id (prereg §6).
    """
    context = gctx if gctx is not None else _GCTX
    if context is None:
        raise RuntimeError("calibration-gate context not initialised")
    gcfg = context.gate_config
    rng = np.random.default_rng(seed)
    universe = draw_universe_gate(context, rng)

    if gcfg.ball:
        ball = draw_ball(context, universe, rng)
        ln1, ln2, slope = log_channel_posteriors_ball(context, universe, ball)
        k_mean = float(np.mean(ball.K))
        n_imp = ball.n_impostors_total
        n_degen = ball.n_degenerate_windows
    else:
        ln1, ln2, slope = cl.log_channel_posteriors(context.cl_ctx, universe)
        k_mean = 1.0
        n_imp = 0
        n_degen = 0

    h_arr = np.asarray(gcfg.h_grid, dtype=np.float64)
    r1 = cl.posterior_readout(h_arr, ln1)
    r2 = cl.posterior_readout(h_arr, ln2)
    pp1 = pp_readout(h_arr, ln1, gcfg.h_true)
    pp2 = pp_readout(h_arr, ln2, gcfg.h_true)
    with np.errstate(divide="ignore", invalid="ignore"):
        texture_corr = float(
            np.corrcoef(np.log(universe.sigma_dL), np.log(universe.d_L_true))[0, 1]
        )

    return {
        "seed": int(seed),
        "cell": gcfg.cell,
        "h_true": float(gcfg.h_true),
        "sigma_texture": gcfg.sigma_texture,
        "sigma_z": float(gcfg.sigma_z),
        "f_incl": float(gcfg.f_incl),
        "lambda_ball": float(gcfg.lambda_ball),
        "n_events": int(gcfg.n_events),
        "n_proposed": int(universe.n_drawn),
        "z_median": float(np.median(universe.z_true)),
        "M_source_median": float(np.median(universe.M_true)),
        "frac_below_kink": float(np.mean(universe.M_true < 1.0e5)),
        "K_mean": k_mean,
        "n_impostors_total": int(n_imp),
        "n_degenerate_windows": int(n_degen),
        "texture_corr": texture_corr,
        "map_1d": r1["map"],
        "map_2d": r2["map"],
        "map_1d_refined": r1["map_refined"],
        "map_2d_refined": r2["map_refined"],
        "mean_1d": r1["mean"],
        "mean_2d": r2["mean"],
        "railed_low_1d": r1["railed_low"],
        "railed_high_1d": r1["railed_high"],
        "railed_low_2d": r2["railed_low"],
        "railed_high_2d": r2["railed_high"],
        "sum_dlog_gfrac_dh": float(slope[0]),
        "pit_1d": pp1["pit"],
        "pit_2d": pp2["pit"],
        "hpd50_1d": pp1["hpd50"],
        "hpd68_1d": pp1["hpd68"],
        "hpd90_1d": pp1["hpd90"],
        "hpd50_2d": pp2["hpd50"],
        "hpd68_2d": pp2["hpd68"],
        "hpd90_2d": pp2["hpd90"],
        "post_sd_1d": pp1["post_sd"],
        "post_sd_2d": pp2["post_sd"],
        "edge_mass_1d": pp1["edge_mass"],
        "edge_mass_2d": pp2["edge_mass"],
        "ln_post_1d": [float(v) for v in ln1],
        "ln_post_2d": [float(v) for v in ln2],
    }


# ── Aggregation: DS-1 … DS-7 ─────────────────────────────────────────────────


def _channel_aggregate(
    records: list[dict[str, Any]], channel: str, h_true: float
) -> dict[str, Any]:
    """DS-1/2/3/4/5 + §8 edge guard for one channel of one cell×truth.

    Band comparisons are mechanical extractions of the prereg §7 locked bands;
    the branch call is never made here.

    Args:
        records: Per-seed records.
        channel: ``"1d"`` or ``"2d"``.
        h_true: The cell's truth.

    Returns:
        The channel aggregate block.
    """
    n = len(records)
    pits = np.asarray([r[f"pit_{channel}"] for r in records], dtype=np.float64)
    maps = np.asarray([r[f"map_{channel}"] for r in records], dtype=np.float64)
    maps_ref = np.asarray([r[f"map_{channel}_refined"] for r in records], dtype=np.float64)
    means = np.asarray([r[f"mean_{channel}"] for r in records], dtype=np.float64)
    sds = np.asarray([r[f"post_sd_{channel}"] for r in records], dtype=np.float64)
    edges = np.asarray([r[f"edge_mass_{channel}"] for r in records], dtype=np.float64)

    # DS-1 — HPD coverage with binomial nulls.
    coverage: dict[str, Any] = {}
    ds1_status = "PASS"
    for lv in HPD_LEVELS:
        key = f"hpd{int(round(lv * 100))}"
        c = float(np.mean([r[f"{key}_{channel}"] for r in records]))
        sig = math.sqrt(lv * (1.0 - lv) / n)
        band2 = (lv - 2.0 * sig, lv + 2.0 * sig)
        band3 = (lv - 3.0 * sig, lv + 3.0 * sig)
        inside2 = band2[0] <= c <= band2[1]
        inside3 = band3[0] <= c <= band3[1]
        coverage[key] = {
            "value": c,
            "binomial_sigma": sig,
            "band_2sigma": list(band2),
            "band_3sigma": list(band3),
            "inside_2sigma": inside2,
            "inside_3sigma": inside3,
        }
        if not inside3:
            ds1_status = "FAIL"
        elif not inside2 and ds1_status != "FAIL":
            ds1_status = "MARGINAL"

    # DS-2 — P–P/KS against Uniform(0,1).
    finite_pits = pits[np.isfinite(pits)]
    d_ks = ks_distance(finite_pits)
    n_ks = finite_pits.size
    d95 = _KS_C95 / math.sqrt(n_ks) if n_ks else float("nan")
    d99 = _KS_C99 / math.sqrt(n_ks) if n_ks else float("nan")
    ds2_status = "PASS" if d_ks <= d95 else ("FAIL" if d_ks > d99 else "MARGINAL")

    # DS-3 — MAP bias (grid-argmax primary; refined + mean reported alongside).
    bias = float(np.mean(maps)) - h_true
    mc = float(np.std(maps, ddof=1) / math.sqrt(n)) if n > 1 else 0.0
    if abs(bias) <= _DS3_IN_BAND:
        ds3_status = "IN-BAND"
    elif abs(bias) >= _DS3_DEFECT:
        ds3_status = "DEFECT-SCALE"
    else:
        ds3_status = "MIXED-SCALE"

    # DS-4 — rail statistic.
    r_low = float(np.mean([r[f"railed_low_{channel}"] for r in records]))
    r_high = float(np.mean([r[f"railed_high_{channel}"] for r in records]))

    # §8 edge-contamination guard (DS-1/DS-2 reads only; DS-4/DS-6 exempt).
    edge_loaded = float(np.mean(edges > EDGE_MASS_THRESHOLD))
    contaminated = edge_loaded > EDGE_CONTAMINATION_FRACTION

    return {
        "n_seeds": n,
        "ds1_coverage": coverage,
        "ds1_status": ds1_status,
        "ds2_ks": {
            "D": d_ks,
            "n": int(n_ks),
            "D_95": d95,
            "D_99": d99,
            "status": ds2_status,
        },
        "ds3_map_bias": {
            "bias": bias,
            "mc_error": mc,
            "mean_map": float(np.mean(maps)),
            "mean_map_refined": float(np.mean(maps_ref)),
            "mean_posterior_mean": float(np.mean(means)),
            "map_sd": float(np.std(maps, ddof=1)) if n > 1 else 0.0,
            "status": ds3_status,
        },
        "ds4_rails": {"railed_low_frac": r_low, "railed_high_frac": r_high},
        "ds5_width": {
            "post_sd_median": float(np.median(sds[np.isfinite(sds)]))
            if np.any(np.isfinite(sds))
            else float("nan"),
            "note": "F5 comparison is readout-side (prereg DS-5: factor-2 screen)",
        },
        "edge_guard": {
            "edge_loaded_frac": edge_loaded,
            "edge_contaminated": contaminated,
            "note": "if contaminated, DS-1/DS-2 carry no gate weight (prereg §8)",
        },
    }


def ds7_accounting(gctx: GateContext, records: list[dict[str, Any]]) -> dict[str, Any]:
    """DS-7 — in-loop generator-closure accounting identity (prereg §7).

    ``|N_det / (<n_drawn> p_bar) - 1| <= 0.05`` with ``p_bar`` the mean
    ``S_4D`` acceptance over a fresh 1e6-proposal MC at the cell's truth.
    A violation is a V-class instrument defect, not a physics finding.

    **Build-time deviation, reported not hidden (module docstring pattern):**
    the parent's ``draw_universe`` counts ``n_drawn`` in whole 4096-proposal
    batches and discards accepted events beyond ``N_det`` — a structural
    UNDER-estimate of the ratio of ≈ 5 % at (p_bar ≈ 0.095, N = 1500), i.e.
    comparable to the blind-locked 0.05 band. The registered raw ratio is
    reported unchanged (``ratio``, ``pass_raw``); a granularity-corrected
    companion (``ratio_corrected``, ``pass_corrected``) divides out the
    expected batch overcount, obtained by simulating the batch stopping rule
    at ``p_bar`` — the corrected statistic is ~1 exactly when generator and
    estimator share ``S_4D``.

    **v2 status (divergence 12 / defect 2): REPORT-ONLY in both forms.** The
    v1 adjudication showed the raw form is MC-seed-fragile (6/9 vs 8/9
    violations under a different p_bar MC seed — the 0.05 band edge sits
    inside the p_bar MC noise at these ratios); the corrected form passed
    9/9. The author call on which form (if either) carries V-class weight
    remains OPEN; in v2 neither form is in the trigger set. ``p_bar_mc_se``
    (the MC standard error of ``p_bar``) is emitted so the fragility is
    quantified in the record.

    Args:
        gctx: The gate context.
        records: Per-seed records (supply ``n_proposed``).

    Returns:
        The DS-7 block.
    """
    cfg = gctx.cl_ctx.config
    rng = np.random.default_rng(_DS7_MC_SEED)
    u_z = rng.random(_DS7_N_MC)
    z = np.interp(u_z, gctx.cl_ctx.gen_z_cdf, gctx.cl_ctx.gen_z_nodes)
    u_m = rng.random(_DS7_N_MC)
    M = 10.0 ** np.interp(u_m, gctx.cl_ctx.gen_M_cdf, gctx.cl_ctx.gen_log10_M_nodes)
    d_L = np.asarray(dist_vectorized(z, h=cfg.h_true), dtype=np.float64)
    p = np.asarray(
        gctx.cl_ctx.detection.detection_probability_with_bh_mass_interpolated(
            d_L, M * (1.0 + z), 0.0, 0.0, h=cfg.h_true
        ),
        dtype=np.float64,
    )
    p_bar = float(np.mean(p))
    p_bar_mc_se = float(np.std(p, ddof=1) / math.sqrt(p.size))
    mean_drawn = float(np.mean([r["n_proposed"] for r in records]))
    ratio = cfg.n_events / (mean_drawn * p_bar) if mean_drawn * p_bar > 0 else float("nan")

    # Expected batch overcount under the parent's stopping rule (simulated at
    # p_bar with the parent's batch size): E[n_drawn_batched] / (N_det / p_bar).
    batch = 4096  # the parent's draw_universe default
    n_sim = 2000
    drawn_sim = np.empty(n_sim, dtype=np.float64)
    for j in range(n_sim):
        have = 0
        batches = 0
        while have < cfg.n_events:
            have += int(rng.binomial(batch, p_bar))
            batches += 1
        drawn_sim[j] = batches * batch
    overcount = float(np.mean(drawn_sim)) * p_bar / cfg.n_events
    ratio_corrected = ratio * overcount

    return {
        "status": "REPORT-ONLY",
        "p_bar": p_bar,
        "p_bar_mc_se": p_bar_mc_se,
        "mean_n_proposed": mean_drawn,
        "ratio": ratio,
        "band": _DS7_BAND,
        "pass_raw": bool(abs(ratio - 1.0) <= _DS7_BAND),
        "expected_batch_overcount": overcount,
        "ratio_corrected": ratio_corrected,
        "pass_corrected": bool(abs(ratio_corrected - 1.0) <= _DS7_BAND),
        "note": (
            "v2: REPORT-ONLY in both forms (divergence 12 / defect 2) — no "
            "V-class or branch weight; raw form is MC-seed-fragile at the band "
            "edge (v1 adjudication 6/9 vs 8/9 by MC seed), corrected form "
            "divides out the parent's batch-granularity undercount; the "
            "raw-vs-corrected author call remains open"
        ),
        "n_mc": _DS7_N_MC,
        "mc_seed": _DS7_MC_SEED,
    }


def aggregate_gate(records: list[dict[str, Any]], gcfg: GateConfig) -> dict[str, Any]:
    """Aggregate one cell×truth's per-seed records into the DS readout.

    Args:
        records: Per-seed records.
        gcfg: The cell configuration.

    Returns:
        The aggregate block (both channels always reported together).
    """
    slopes = np.asarray([r["sum_dlog_gfrac_dh"] for r in records], dtype=np.float64)
    nonfinite = float(
        np.mean(
            [
                (not np.all(np.isfinite(r["ln_post_1d"])))
                or (not np.all(np.isfinite(r["ln_post_2d"])))
                for r in records
            ]
        )
    )
    tex = np.asarray([r["texture_corr"] for r in records], dtype=np.float64)
    tex = tex[np.isfinite(tex)]
    # Degenerate-PIT exemption (v2, divergence 13 / defect 3): at sigma_z = 0
    # the ball posterior is near-delta at truth => PIT = 0.5, C_beta = 1,
    # KS D = 0.5 by construction; DS-1/DS-2 labels are structurally
    # meaningless and carry no weight in any form (prereg v2 registered
    # exemption — B0/V1 are scored on their V-checks and DS-3/DS-4 only).
    degenerate_exempt = bool(gcfg.ball and gcfg.sigma_z == 0.0)
    return {
        "cell": gcfg.cell,
        "h_true": gcfg.h_true,
        "n_seeds": len(records),
        "ds1_ds2_degenerate_pit_exempt": degenerate_exempt,
        "channel_1d": _channel_aggregate(records, "1d", gcfg.h_true),
        "channel_2d": _channel_aggregate(records, "2d", gcfg.h_true),
        "ball": {
            "K_mean": float(np.mean([r["K_mean"] for r in records])),
            "n_impostors_total": int(np.sum([r["n_impostors_total"] for r in records])),
            "n_degenerate_windows": int(np.sum([r["n_degenerate_windows"] for r in records])),
            "sigma_z": gcfg.sigma_z,
            "f_incl": gcfg.f_incl,
            "lambda_ball": gcfg.lambda_ball,
        },
        "texture": {
            "sigma_texture": gcfg.sigma_texture,
            "corr_ln_sigma_dl_ln_dl_median": float(np.median(tex)) if tex.size else float("nan"),
            "v4_band": [_V4_CORR_CENTER - _V4_CORR_TOL, _V4_CORR_CENTER + _V4_CORR_TOL],
            "v4_pass": bool(
                tex.size and abs(float(np.median(tex)) - _V4_CORR_CENTER) <= _V4_CORR_TOL
            ),
        },
        "sum_dlog_gfrac_dh": {
            "mean": float(np.mean(slopes)),
            "production_reference_nats_per_h": 243.5,
        },
        "nonfinite_ln_post_frac": nonfinite,
        "abort_b_triggered": bool(nonfinite > _NONFINITE_ABORT_FRACTION),
    }


# ── Sweep driver ─────────────────────────────────────────────────────────────


def _classify_porcelain(status: str) -> dict[str, list[str]]:
    """Split ``git status --porcelain`` lines by the v2 clean rule.

    Divergence 15 / defect 5: a line counts as **import-path dirt** if any of
    its path components (both sides of a rename) is under
    ``darksiren_emri/`` or ``darksiren_emri_test/``; everything else
    (doc edits, untracked results dirs, rule files, ...) is **other dirt** —
    recorded in full in every output JSON, never blocking.

    Args:
        status: Raw ``git status --porcelain`` output.

    Returns:
        ``{"import_path": [...], "other": [...]}`` with the verbatim
        porcelain lines.
    """
    import_dirt: list[str] = []
    other_dirt: list[str] = []
    for line in status.splitlines():
        if not line.strip():
            continue
        path_part = line[3:] if len(line) > 3 else line
        targets = [t.strip().strip('"') for t in path_part.split(" -> ")]
        if any(t.startswith(_IMPORT_PATH_PREFIXES) for t in targets):
            import_dirt.append(line)
        else:
            other_dirt.append(line)
    return {"import_path": import_dirt, "other": other_dirt}


def _git_state() -> tuple[str, dict[str, list[str]]]:
    """Return (commit, dirt inventory) of the working tree (v2 clean rule).

    Returns:
        The HEAD commit and the :func:`_classify_porcelain` inventory. If git
        is unavailable the import path is conservatively reported dirty.
    """
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL
        )
        return commit, _classify_porcelain(status)
    except (subprocess.CalledProcessError, OSError):
        return "unknown", {"import_path": ["<git unavailable>"], "other": []}


def _enforce_clean_import_path(allow_dirty: bool) -> tuple[str, dict[str, list[str]]]:
    """Enforce the v2 clean rule; return the provenance to embed in the JSON.

    The rule (divergence 15 / defect 5): ZERO uncommitted changes — modified
    or untracked — under the import path. ``allow_dirty`` (smoke/validate
    only; :func:`main` rejects it for registered cell runs) is the only
    escape, and it is recorded in the output.

    Args:
        allow_dirty: The CLI escape flag.

    Returns:
        ``(commit, dirt_inventory)``.

    Raises:
        SystemExit: If the import path is dirty and ``allow_dirty`` is False.
    """
    commit, dirt = _git_state()
    if dirt["import_path"] and not allow_dirty:
        raise SystemExit(
            "STOP: uncommitted changes under the import path "
            f"({', '.join(_IMPORT_PATH_PREFIXES)}):\n  "
            + "\n  ".join(dirt["import_path"])
            + "\nRegistered cells refuse to run (prereg v2 clean rule, defect 5). "
            "Commit first; --allow-dirty is accepted only with --smoke/--validate."
        )
    return commit, dirt


def run_cell(
    gcfg: GateConfig,
    seeds: list[int],
    workers: int,
    *,
    allow_dirty: bool = False,
) -> dict[str, Any]:
    """Run one cell×truth sweep and assemble the results document.

    Args:
        gcfg: The cell configuration.
        seeds: Seeds to run.
        workers: Worker processes (``<= 1`` runs in-process).
        allow_dirty: Permit a dirty IMPORT PATH (smoke/validate only —
            :func:`main` rejects it for registered cell runs; recorded).
            Non-import-path dirt never blocks and is always inventoried.

    Returns:
        The full results dict (written to JSON by :func:`main`).
    """
    commit, dirt = _enforce_clean_import_path(allow_dirty)
    global _GCTX
    if workers > 1:
        t0 = time.monotonic()
        ctx_mp = mp.get_context("fork")
        with ctx_mp.Pool(
            processes=workers, initializer=_gate_worker_init, initargs=(gcfg,)
        ) as pool:
            records = pool.map(run_seed_gate, seeds, chunksize=1)
        wall = time.monotonic() - t0
        if _GCTX is None or _GCTX.gate_config != gcfg:
            _GCTX = build_gate_context(gcfg)  # local context for DS-7
    else:
        if _GCTX is None or _GCTX.gate_config != gcfg:
            _GCTX = build_gate_context(gcfg)  # stale-config guard
        t0 = time.monotonic()  # context build excluded: per-seed time is honest
        records = [run_seed_gate(s) for s in seeds]
        wall = time.monotonic() - t0
    assert _GCTX is not None
    agg = aggregate_gate(records, gcfg)
    agg["ds7"] = ds7_accounting(_GCTX, records)
    return {
        "instrument": "calibration_gate",
        "preregistration": PREREG_PATH,
        "parent_instruments": {
            "closed_loop_gfrac": PARENT_CLOSED_LOOP_COMMIT,
            "pp_coverage": "hpd_contains port only (V2-certified); not imported at runtime",
        },
        "git_commit": commit,
        "git_dirty": bool(dirt["import_path"] or dirt["other"]),
        "import_path_clean": not dirt["import_path"],
        "dirt_inventory": dirt,
        "allow_dirty": allow_dirty,
        "config": asdict(gcfg),
        "seeds": [int(s) for s in seeds],
        "workers": workers,
        "wall_time_s": wall,
        "wall_time_per_seed_s": wall / max(len(seeds), 1),
        "aggregate": agg,
        "per_seed": records,
    }


def cell_seeds(spec: CellSpec, h_true: float, start: int, count: int | None) -> list[int]:
    """Seeds of one cell×truth block, optionally chunked.

    Args:
        spec: The cell spec.
        h_true: The truth (must be one of the spec's).
        start: Offset within the block (chunking; 0 = block start).
        count: Number of seeds (``None`` = the rest of the block).

    Returns:
        Absolute seed list.
    """
    idx = spec.truths.index(h_true)
    base = GATE_BASE_SEED + spec.seed_offsets[idx]
    n = spec.n_seeds - start if count is None else count
    if start < 0 or start + n > spec.n_seeds:
        raise ValueError(
            f"seed chunk [{start}, {start + n}) exceeds cell {spec.name} block of {spec.n_seeds}"
        )
    return [base + start + i for i in range(n)]


# ── R0 retro-read + V5 ───────────────────────────────────────────────────────


def retro_read_r0(path: str = R0_RESULTS_JSON) -> dict[str, Any]:
    """Cell R0: P–P/HPD retro-read of the committed registered closed-loop run.

    Zero compute — reads ``per_seed[].ln_post_{1d,2d}`` from the committed
    JSON. Before the HPD/PIT read is quoted, V5 requires the readout layer to
    reproduce the committed aggregate MAP statistics to ``<= 1e-12`` relative
    (prereg §10). R0 is anchor-only and carries no gate weight (§5).

    Args:
        path: The committed ``closed_loop_results.json``.

    Returns:
        An R0 results document: V5 block + gate-style aggregate.
    """
    with open(path) as fh:
        doc = json.load(fh)
    cfg_dict = dict(doc["config"])
    cfg_dict["h_grid"] = tuple(cfg_dict["h_grid"])
    cl_cfg = cl.ClosedLoopConfig(**cfg_dict)
    h_arr = np.asarray(cl_cfg.h_grid, dtype=np.float64)
    h_true = float(cl_cfg.h_true)

    # V5: recompute the committed aggregate from per_seed with the parent's
    # own aggregation and compare numerically.
    recomputed = cl.aggregate(list(doc["per_seed"]), cl_cfg)
    committed = doc["aggregate"]

    def _cmp(a: Any, b: Any, path_: str, errs: list[str]) -> None:
        if isinstance(a, dict) and isinstance(b, dict):
            for key in a:
                if key in b:
                    _cmp(a[key], b[key], f"{path_}.{key}", errs)
            return
        if isinstance(a, int | float) and isinstance(b, int | float):
            denom = max(abs(float(a)), abs(float(b)), 1e-300)
            if abs(float(a) - float(b)) / denom > _V5_RTOL and abs(float(a) - float(b)) > 1e-15:
                errs.append(f"{path_}: {a} != {b}")

    errs: list[str] = []
    _cmp(recomputed, committed, "aggregate", errs)
    v5_pass = len(errs) == 0

    gcfg = GateConfig(
        cell="R0",
        h_true=h_true,
        ball=False,
        lambda_ball=0.0,
        sigma_z=0.0,
        sigma_texture="independent",
        n_events=cl_cfg.n_events,
        injection_data_dir=cl_cfg.injection_data_dir,
        crb_reference_csv=cl_cfg.crb_reference_csv,
        h_grid=cl_cfg.h_grid,
    )
    records: list[dict[str, Any]] = []
    for r in doc["per_seed"]:
        ln1 = np.asarray(r["ln_post_1d"], dtype=np.float64)
        ln2 = np.asarray(r["ln_post_2d"], dtype=np.float64)
        pp1 = pp_readout(h_arr, ln1, h_true)
        pp2 = pp_readout(h_arr, ln2, h_true)
        rec = dict(r)
        rec.update(
            {
                "cell": "R0",
                "h_true": h_true,
                "sigma_texture": "independent",
                "sigma_z": 0.0,
                "f_incl": 1.0,
                "lambda_ball": 0.0,
                "K_mean": 1.0,
                "n_impostors_total": 0,
                "n_degenerate_windows": 0,
                "texture_corr": float("nan"),
                "pit_1d": pp1["pit"],
                "pit_2d": pp2["pit"],
                "hpd50_1d": pp1["hpd50"],
                "hpd68_1d": pp1["hpd68"],
                "hpd90_1d": pp1["hpd90"],
                "hpd50_2d": pp2["hpd50"],
                "hpd68_2d": pp2["hpd68"],
                "hpd90_2d": pp2["hpd90"],
                "post_sd_1d": pp1["post_sd"],
                "post_sd_2d": pp2["post_sd"],
                "edge_mass_1d": pp1["edge_mass"],
                "edge_mass_2d": pp2["edge_mass"],
            }
        )
        records.append(rec)
    commit, dirt = _git_state()
    return {
        "instrument": "calibration_gate",
        "cell": "R0",
        "preregistration": PREREG_PATH,
        "source_json": path,
        "git_commit": commit,
        "git_dirty": bool(dirt["import_path"] or dirt["other"]),
        "import_path_clean": not dirt["import_path"],
        "dirt_inventory": dirt,
        "config": asdict(gcfg),
        "v5": {"pass": v5_pass, "mismatches": errs, "rtol": _V5_RTOL},
        "note": "R0 is anchor-only, carries no gate weight (prereg §5)",
        "aggregate": aggregate_gate(records, gcfg),
        "per_seed": records,
    }


# ── Validation mode (V3, V4, V5; V1 is a cell, V2 is a unit test) ────────────


def run_validate(workers: int) -> dict[str, Any]:
    """Run the §10 validity checks executable without a full sweep.

    V3 (determinism, reduced-N spot check on the maximal code path),
    V4 (texture certification on full-N drawn universes),
    V5 (R0 aggregate reproduction). V1 is the registered 50-seed cell;
    V2 is the pytest unit test
    (``darksiren_emri_test/validation/test_calibration_gate.py``).

    Args:
        workers: Unused (kept for CLI symmetry); checks run in-process.

    Returns:
        The validation document.
    """
    out: dict[str, Any] = {}

    # V3 — determinism: same seed, same config => bit-identical record.
    gcfg_v3 = GateConfig(
        cell="custom",
        h_true=0.730,
        ball=True,
        lambda_ball=4.0,
        sigma_z=0.035,
        sigma_texture="dl_binned",
        n_events=300,
    )
    ctx_v3 = build_gate_context(gcfg_v3)
    rec_a = run_seed_gate(GATE_BASE_SEED, ctx_v3)
    rec_b = run_seed_gate(GATE_BASE_SEED, ctx_v3)
    v3_pass = json.dumps(rec_a, sort_keys=True) == json.dumps(rec_b, sort_keys=True)
    out["v3"] = {"pass": bool(v3_pass), "seed": GATE_BASE_SEED, "n_events": 300}

    # V4 — texture certification at full N (3 seeds).
    gcfg_v4 = GateConfig(
        cell="custom",
        h_true=0.730,
        ball=False,
        lambda_ball=0.0,
        sigma_z=0.0,
        sigma_texture="dl_binned",
    )
    ctx_v4 = build_gate_context(gcfg_v4)
    corrs: list[float] = []
    for i in range(3):
        rng = np.random.default_rng(GATE_BASE_SEED + i)
        uni = draw_universe_gate(ctx_v4, rng)
        corrs.append(float(np.corrcoef(np.log(uni.sigma_dL), np.log(uni.d_L_true))[0, 1]))
    med = float(np.median(corrs))
    out["v4"] = {
        "corrs": corrs,
        "median": med,
        "band": [_V4_CORR_CENTER - _V4_CORR_TOL, _V4_CORR_CENTER + _V4_CORR_TOL],
        "pass": bool(abs(med - _V4_CORR_CENTER) <= _V4_CORR_TOL),
        "note": (
            "v2 band [0.63, 0.75] derived from the PRE-DECLARED build-time "
            "decile-attenuation analysis (0.69 ± 3 x 0.02 replica SD, "
            "divergences 7 + 11 / defect 1); v1 measured 0.664-0.666 is "
            "post-hoc consistency only; failure voids texture cells per §10"
        ),
    }

    # V5 — R0 reproduction.
    if os.path.isfile(R0_RESULTS_JSON):
        r0 = retro_read_r0()
        out["v5"] = r0["v5"]
    else:
        out["v5"] = {"pass": None, "note": f"{R0_RESULTS_JSON} not present"}

    out["v2"] = {
        "note": (
            "run: uv run pytest darksiren_emri_test/validation/"
            "test_calibration_gate.py -m 'not gpu' (HPD port certification)"
        )
    }
    return out


# ── CLI ──────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    p = argparse.ArgumentParser(
        description=(
            "Calibration-gate instrument v2 (prereg "
            "results/calibration_gate_v2_20260810/PREREGISTRATION_CALIBRATION_GATE_V2.md; "
            "v1 prereg b50ccc65, v1 verdict GATE-NOT-TRUSTWORTHY)"
        )
    )
    p.add_argument(
        "--cell",
        choices=("A", "B0", "B1", "B2", "V1", "R0", "O1"),
        help="prereg §5 cell to run (R0 = zero-compute retro-read)",
    )
    p.add_argument("--truth", type=float, default=None, help="h_true (must be in the cell's set)")
    p.add_argument(
        "--seed-range",
        type=str,
        default=None,
        help="START:COUNT chunk within the cell's registered seed block",
    )
    p.add_argument(
        "--seeds", type=str, default=None, help="explicit comma-separated absolute seeds"
    )
    p.add_argument("--n-seeds", type=int, default=None, help="cap the number of seeds")
    p.add_argument("--n-events", type=int, default=None, help="override N_det (smoke/dev only)")
    p.add_argument("--out", type=str, default=None, help="output JSON path")
    p.add_argument("--workers", type=int, default=max(mp.cpu_count() - 2, 1))
    p.add_argument("--smoke", action="store_true", help="3 seeds, N=300, 1 worker + V3 spot-check")
    p.add_argument("--validate", action="store_true", help="run V3/V4/V5 validity checks")
    p.add_argument(
        "--allow-dirty",
        action="store_true",
        help=(
            "permit a dirty IMPORT PATH — accepted only with --smoke or "
            "--validate, never on a registered cell run (v2 clean rule, "
            "divergence 15; recorded in the JSON)"
        ),
    )
    p.add_argument("--log-level", type=str, default="INFO")
    return p


def _guard_out_path(out: str) -> None:
    """Refuse output paths inside production run/campaign directories."""
    norm_path = os.path.normpath(out)
    parts = norm_path.split(os.sep)
    for part in parts[:-1]:
        if part.startswith("run_20") or part.startswith("campaign"):
            raise SystemExit(
                f"STOP: refusing to write '{out}' — production run/campaign directory. "
                f"Use {DEFAULT_OUT_DIR}/ (prereg §0)."
            )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Argument vector (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code.
    """
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    # v2 clean rule (divergence 15 / defect 5): --allow-dirty is smoke/validate
    # only; a registered cell run can never take it.
    if args.allow_dirty and not (args.smoke or args.validate):
        raise SystemExit(
            "STOP: --allow-dirty is accepted only with --smoke or --validate "
            "(v2 clean rule — registered cells must run on a clean import path)."
        )

    if args.validate:
        _enforce_clean_import_path(args.allow_dirty)
        doc = run_validate(args.workers)
        out = args.out or os.path.join(DEFAULT_OUT_DIR, "validate_results.json")
        _guard_out_path(out)
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        with open(out, "w") as fh:
            json.dump(doc, fh, indent=2)
        _LOGGER.info("validate: %s", {k: v.get("pass") for k, v in doc.items()})
        return 0

    if args.cell is None:
        raise SystemExit("one of --cell or --validate is required")
    if args.cell == "O1":
        raise SystemExit(
            "Cell O1 (volume-kernel arm) is NOT built: kernel-form sensitivity is "
            "NOT-EVALUABLE per prereg §9 item 6."
        )
    if args.cell == "R0":
        _enforce_clean_import_path(args.allow_dirty)
        doc = retro_read_r0()
        out = args.out or os.path.join(DEFAULT_OUT_DIR, "R0_results.json")
        _guard_out_path(out)
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        with open(out, "w") as fh:
            json.dump(doc, fh, indent=2)
        _LOGGER.info("R0 retro-read: V5 pass=%s  (anchor-only, no gate weight)", doc["v5"]["pass"])
        return 0

    spec = CELL_SPECS[args.cell]
    truth = args.truth if args.truth is not None else spec.truths[0]
    if not any(abs(truth - t) < 1e-12 for t in spec.truths):
        raise SystemExit(f"truth {truth} not in cell {spec.name} registered set {spec.truths}")
    truth = next(t for t in spec.truths if abs(truth - t) < 1e-12)

    smoke = args.smoke
    n_events = args.n_events if args.n_events is not None else (300 if smoke else 1500)
    gcfg = GateConfig(
        cell=spec.name,
        h_true=truth,
        ball=spec.ball,
        lambda_ball=spec.lambda_ball,
        sigma_z=spec.sigma_z,
        sigma_texture=spec.sigma_texture,
        n_events=n_events,
        h_grid=spec.h_grid,
    )

    if args.seeds is not None:
        seeds = [int(s) for s in args.seeds.split(",")]
    elif args.seed_range is not None:
        start_s, count_s = args.seed_range.split(":")
        seeds = cell_seeds(spec, truth, int(start_s), int(count_s))
    else:
        seeds = cell_seeds(spec, truth, 0, None)
    if smoke and args.n_seeds is None:
        seeds = seeds[:3]
    elif args.n_seeds is not None:
        seeds = seeds[: args.n_seeds]

    workers = 1 if smoke else args.workers
    doc = run_cell(gcfg, seeds, workers, allow_dirty=args.allow_dirty)
    doc["smoke"] = smoke

    if smoke:
        # V3 spot-check: re-run the first seed, must be bit-identical.
        rec_again = run_seed_gate(seeds[0])
        first = next(r for r in doc["per_seed"] if r["seed"] == seeds[0])
        doc["v3_smoke"] = {
            "pass": json.dumps(rec_again, sort_keys=True) == json.dumps(first, sort_keys=True),
            "seed": seeds[0],
        }

    out = args.out or os.path.join(
        DEFAULT_OUT_DIR, f"{spec.name}_h{truth:.3f}_results.json".replace("0.", "0p")
    )
    _guard_out_path(out)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as fh:
        json.dump(doc, fh, indent=2)

    agg = doc["aggregate"]
    _LOGGER.info(
        "cell %s h_true=%.3f n=%d | 1D: C90=%.3f KS=%.3f bias=%+.4f R_low=%.3f | "
        "2D: C90=%.3f KS=%.3f bias=%+.4f R_low=%.3f | DS7 ratio=%.3f | %.1f s/seed",
        gcfg.cell,
        gcfg.h_true,
        agg["n_seeds"],
        agg["channel_1d"]["ds1_coverage"]["hpd90"]["value"],
        agg["channel_1d"]["ds2_ks"]["D"],
        agg["channel_1d"]["ds3_map_bias"]["bias"],
        agg["channel_1d"]["ds4_rails"]["railed_low_frac"],
        agg["channel_2d"]["ds1_coverage"]["hpd90"]["value"],
        agg["channel_2d"]["ds2_ks"]["D"],
        agg["channel_2d"]["ds3_map_bias"]["bias"],
        agg["channel_2d"]["ds4_rails"]["railed_low_frac"],
        agg["ds7"]["ratio"],
        doc["wall_time_per_seed_s"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
