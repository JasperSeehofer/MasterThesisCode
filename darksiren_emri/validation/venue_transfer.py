r"""Venue-transfer instrument — production-matched ball venue for the σ_z coverage-collapse transfer.

**What this instrument is.** The author-named decisive measurement (ruling R4,
2026-08-11) of the calibration-gate v2 clause-(b) DEFECT verdict, registered in
``results/venue_transfer_20260811/PREREGISTRATION_VENUE_TRANSFER.md``
(prereg-first, VT-D0: the registration was committed BEFORE this build; this
module + its test file must land in a descendant commit before any registered
cell runs, and a registered run commit must have an empty import-path diff to
the instrument commit — the R1-ratified D-4/D-5 pattern).

It rebuilds the v2 gate's multi-candidate ball venue on production-matched
inputs and asks: **does the σ_z-dosed coverage collapse (uniform +σ_z MAP bias,
delta-narrow posteriors, 0 % HPD coverage — v2 DS-8 T2, quotable per R2)
survive production realism?** Adopted axes (prereg §2): (a) the real detected
event population — pinned to the production CRB CSV rows (VT-D1); (b) the real
per-event candidate-ball multiplicities ``K_i`` — pinned to the frozeng
per-galaxy emit (VT-D2, multiplicity only; weights stay equal 1/K by the
registered bracketing argument); (c) the real heterogeneous GLADE per-galaxy
σ_z — z-decile-matched empirical draws from the iiib pruned catalogue frame,
spec-z tail included (VT-D3). Excluded with registered justifications: the
production estimator code path (VT-D4 — replaced by the V-T5 bit-reproduction
certification + the T-0/T-a anchors) and per-galaxy rate weights (W1 arm
reserved, NOT built).

**Thin extension over registered instruments, modifying neither.**
:mod:`darksiren_emri.validation.calibration_gate` (code identity
``065e7f58``, run ``dbde71dc``) and
:mod:`darksiren_emri.validation.closed_loop_gfrac` (``77b524af``) are
imported as libraries: parent context build, ``alpha(h)`` tables, ``z_of_dl``
ladders, ``pp_readout``/``hpd_contains``, GL/GH quadrature orders, the
canonical 41-point h grid, the clean rule (:func:`calibration_gate.
_enforce_clean_import_path`, quoted VERBATIM per V-T4 by importing it), the
output-path guard, and the production ``g_i`` evaluation
(:func:`calibration_gate._g_ball`) are all inherited. New capabilities
(prereg §3): (i) the pinned-event universe (VT-D1); (ii) pinned-K
real-multiplicity balls (VT-D2); (iii) the z-decile σ_z sampler + the
per-candidate-σ estimator core — a vectorized generalization of the gate's
scalar-σ_z ball path, certified by V-T5 bit-reproduction in v2-compat mode
(committed ``B2_h0p730_results.json`` per-seed records on v2 seeds
20286808–20286810).

**Cells (prereg §5; canonical 41-point grid; all share the pinned event set):**

=====  ==================  ===============  ==========================  =================
cell   balls               σ_z              truths × seeds              seed blocks
=====  ==================  ===============  ==========================  =================
T-0    real ``K_i``        0 (anchor)       0.730 × 200                 +40000…+40199
T-a    Poisson λ = 4       0.035 flat       0.730 × 200                 +41000…+41199
T-b    real ``K_i``        0.035 flat       0.730 × 200                 +42000…+42199
T-c    real ``K_i``        GLADE sampler    0.690×200/0.730×400/0.770×200  +43000/+44000/+45000
W1     *(NOT built)*       —                reserved                    +46000…+46399
O2     *(NOT built)*       —                reserved                    +47000…+47399
=====  ==================  ===============  ==========================  =================

Seed base 20260808 (the gate's), v3 offsets in the +40000 decade — disjoint by
construction from v1 (+[0, 9049]) and v2 (+[20000, 29049]); unit-tested.

**Deliberate divergences from the parents (documented, per the build mandate):**

1. *Context is built in the parent process BEFORE forking* (the gate builds it
   per worker in the pool initializer). The venue context holds the ~20.8M-row
   pruned-frame σ_z pools (~330 MB); fork shares them copy-on-write, so
   build-before-fork avoids 64 redundant builds and 64 memory copies. The
   initializer keeps the gate's ``if _VCTX is None`` guard as a safety net.
   No statistic changes (V-T2/V-T5 govern).
2. *Chunked pair evaluation* (registered implementation freedom, prereg §3):
   candidate-pair rows are evaluated in fixed-size chunks (``chunk_pairs``
   default 16384) and the ``(nodes, n_hermite)`` intermediate inside the
   ``g_i`` evaluation is capped at :data:`_G_NODE_CHUNK` flattened nodes per
   :func:`~darksiren_emri.bayesian_inference.bayesian_statistics.completion_mass_factor_g`
   call (:func:`_g_ball_capped`, the memory-capped mirror of the gate's
   ``_g_ball`` — needed because the peak single event has K = 245,364, i.e.
   ~12.3M quadrature nodes: gate-shaped single calls would allocate ~19 GB).
   Chunking is fully deterministic (V-T2) and mathematically identity-
   preserving: per-row operations are row-independent (all estimator nodes
   satisfy ``0 <= z <= z_max(h) <= 1.52 < 1.6``, the fiducial ``dist``
   fast-path table range, so the array-global branch never flips) and the
   per-h event sums run once over the full pair arrays in the gate's order.
   The ONLY chunk-shape sensitivity is O(1 ULP) from shape-dependent BLAS
   accumulation inside the ``@`` reductions (measured at build time:
   ≲ 5e-16 relative); the registered campaign therefore runs ONE fixed chunk
   geometry, recorded in every output JSON (``config.chunk_pairs``).
   ``chunk_pairs = 0`` is the EXACT gate-shape mode (single chunk, unsplit
   per-event ``g`` calls) — bit-identical to the gate's full-array path by
   construction, used by the V-T5 certification and the parity unit tests;
   it is memory-infeasible only for the real-K venue cells.
3. *Census pins are stored at full recomputed precision*: the prereg prints
   truncated decimals (mean 751.702…, p99 11325.26, nonempty mean 1215.58…);
   the module pins the design-time full-precision recomputation
   (751.7021410579345 / 11325.26 / 1215.5835030549897); integer pins are
   exact as printed. σ_z pins are exact float64 round-trips of the committed
   ``rr1_ball_sigma_census.json`` values.
4. *Cell ids* ``T0/Ta/Tb/Tc`` in CLI/JSON stand for the prereg's
   ``T-0/T-a/T-b/T-c`` (recorded per cell as ``prereg_cell``).
5. *Chunked runs write per-chunk JSONs* (``--seed-range`` appends
   ``_seedsSTART_COUNT`` to the default output name) — combined at readout;
   the gate had one task per cell so its default naming sufficed.
6. *Horizon guard measured at design time*: 0 of the pinned 982 events exceed
   ``0.999 × dl_max(h_true)`` at any registered truth (``get_dl_max`` is
   h-invariant for this pool); the > 5 % VENUE-CONFOUNDED guard (abort (d))
   is enforced anyway at every context build.
7. *T-a reuses* :func:`calibration_gate.draw_ball` *verbatim* (Poisson
   impostors, flat σ_z applied inside, the parent's RNG order). Real-K cells
   split ball placement (:func:`draw_ball_pinned`) from the σ_z texture step —
   the prereg §4 step order (noise → ball → σ_z draws → ε) is the registered
   order for them.
8. *σ_z sampler decile edges are rank-based* on the pruned frame's own z
   distribution (the house ``dl_binned`` pattern applied to σ_z, VT-D3); ball
   members whose z exceeds the last edge (impostors above the pruned frame's
   z ≤ 1.5 range are possible in wide windows) map to the top decile —
   disclosed, not modeled.
9. *``--n-events-cap`` is a smoke/validate-only dev override* (truncates the
   pinned event list for timing/determinism spot checks); like
   ``--allow-dirty`` it is refused outright on a registered cell run and
   recorded in the output JSON when used.
10. *Prereg §5 smoke is full-N* (3 seeds of T-0/T-a/T-c(0.730), measuring
    per-seed CPU against the derived 13.06/2.84 CPU-ms/pair anchors — the
    abort-(a) input); ``--smoke`` runs full N by default and only reduces
    fidelity when ``--n-events-cap`` is passed explicitly.
11. *Opt-in intra-seed h-grain parallel mode* (``--grain h``; the default
    ``--grain seed`` is the registered campaign's pool-over-seeds mode,
    untouched). Motivation: one seed is a 982-event × ΣK = 1,193,703-pair ×
    41-h unit (~3.79 CPU-h measured), so the seed-grain unit is hours long;
    the 41 h-points of the estimator loop are independent given the seed's
    draws. Construction (bit-identity by design): the registered RNG phase —
    prereg §4 steps 2–4, one sequential PCG64 stream per seed in the
    registered order (divergence 7) — runs SERIALLY in the parent via
    :func:`_draw_seed_realization` (the exact code the seed-grain path runs,
    sub-second), so the draws are the registered ones by construction; the
    estimator's per-h body (:func:`_channel_terms_at_h`, which consumes no
    RNG) is then farmed to a fork pool over the 41 h-points, each task
    executing the identical code with the IDENTICAL fixed chunk geometry
    (``chunk_pairs``/:data:`_G_NODE_CHUNK` — identical array shapes ⇒
    identical BLAS kernels ⇒ identical floats, divergence 2) and returning
    three scalars; reassembly is array indexing, and the slope + readout +
    record assembly run in the parent unchanged. Per-seed records are
    BYTE-IDENTICAL to seed-grain output for ANY worker count (unit-tested
    cross-mode over every venue cell type, including a real-K capped-``g``
    multi-chunk case). h-grain (41 units/seed) was chosen over raw per-event
    grain per the measured load imbalance: the peak event (K = 245,364)
    holds ~20.6 % of all pairs, capping pure event-parallel speedup at ~5×.
    A (seed, event_idx)-rekeyed RNG mode was rejected: the sequential-shared
    stream is positionally coupled across events (vectorized draws; the
    glade sampler consumes decile-by-decile across ALL events), so rekeying
    would change every registered draw and require a new seed plan +
    re-certification. Both modes fork from the same parent process, so the
    BLAS threading environment is inherited identically; campaigns must not
    change ``OMP_NUM_THREADS``/``OPENBLAS_NUM_THREADS`` between modes they
    intend to compare byte-for-byte. NOT mirrored into
    ``calibration_gate.py``: the gate's per-h loop
    (``log_channel_posteriors_ball``) is separate certified code (identity
    ``065e7f58``, pinned by the running campaign) and this module modifies
    neither parent — TODO(seed-grain-parents): if the gate is ever
    re-certified, port the ``_channel_terms_at_h`` split there the same way.

**Estimator (prereg §4 step 5, gate math with vector σ):** the gate's
certified mirror generalized to per-candidate σ_z vectors — bare kernel ×
distance likelihood, equal 1/K candidate prior, no ``w_pop``, no selection
factor in the numerator, per-candidate ±5σ_k GL-50 clip on
``[max(z_lo(h), z_obs,k − 5σ_k), min(z_hi(h), z_obs,k + 5σ_k)]`` (σ_k = 0 ⇒
point evaluation), ``ln P(h) = Σ_i ln L_i(h) − N_det ln α(h)`` with N fixed
and the finite −745 zero-event penalty — all carried verbatim from the gate,
divergence-10 convention included. The matched-model principle carries:
generator and estimator share each candidate's σ_z, so any coverage failure
is the estimator's photo-z HANDLING, not misspecification.

**What this module never does**: import ``BayesianStatistics``; modify a
parent; produce a production posterior (every posterior is a
synthetic-universe diagnostic, quotable only against its own truth);
adjudicate the branch (band comparisons are mechanical; the branch call is
presented to the author, never self-adjudicated).

Cells W1 (rate-weights arm) and O2 (``volume_deconv`` arm) are **not built**:
NOT-EVALUABLE per prereg §9 items 3 and 2; the CLI says so explicitly.

CPU-only. No cupy import, direct or transitive.

References:
    Mandel, Farr & Gair (2019), arXiv:1809.02063, Eqs. (5)-(7).
    Gray et al. (2020), arXiv:1908.06050, Eqs. (29), (32).
    Talts et al. (2018), arXiv:1804.06788 — simulation-based calibration.
"""

import argparse
import hashlib
import json
import logging
import multiprocessing as mp
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.stats import norm

from darksiren_emri.bayesian_inference.bayesian_statistics import (
    completion_mass_factor_g,
)
from darksiren_emri.physical_relations import dist_vectorized
from darksiren_emri.validation import calibration_gate as cg
from darksiren_emri.validation import closed_loop_gfrac as cl

_LOGGER = logging.getLogger(__name__)

PREREG_PATH = "results/venue_transfer_20260811/PREREGISTRATION_VENUE_TRANSFER.md"
DEFAULT_OUT_DIR = "results/venue_transfer_20260811"
VT_BASE_SEED = cg.GATE_BASE_SEED  # 20260808 — the gate's base, v3 offsets +40000 decade (VT-D7)
PARENT_CALGATE_CODE_COMMIT = "065e7f58"
PARENT_CALGATE_RUN_COMMIT = "dbde71dc"
PARENT_CLOSED_LOOP_COMMIT = "77b524af"

# Registered input pins (prereg V-T3; any mismatch => STOP).
CRB_CSV_PATH = cl.DEFAULT_CRB_CSV
CRB_CSV_MD5 = "9a1f2a14384a9281c97ca3be312ddaab"
FROZENG_EMIT_JSON = "results/run_20260804_frozeng/iiib/posteriors_with_bh_mass/h_0_73.json"
FROZENG_EMIT_MD5 = "34c50e91028b6a6458a2b145db545705"
PRUNED_CATALOGUE_CSV = (
    "results/campaign51_20260728/realistic_20260729/realizations_staged/"
    "cluster_parent_reduced_galaxy_catalogue.csv"
)
B2_H0P730_RESULTS_JSON = "results/calibration_gate_v2_20260810/B2_h0p730_results.json"
V_T5_SEEDS: tuple[int, ...] = (20286808, 20286809, 20286810)

# VT-D2 registered K-census pins (recomputed at design time from the emit;
# exact-match required — divergence 3: full precision behind the prereg's
# printed truncations).
K_CENSUS_PINS: dict[str, float | int] = {
    "n_events_evaluated": 1588,
    "zeros": 606,
    "ones": 74,
    "median": 6.0,
    "mean": 751.7021410579345,
    "p99": 11325.26,
    "max": 245364,
    "sum_K": 1_193_703,
    "nonempty_n": 982,
    "nonempty_median": 84.0,
    "nonempty_mean": 1215.5835030549897,
}
# VT-D3 registered pruned-frame σ_z pins (committed rr1_ball_sigma_census.json
# ``iiib.pruned_sigma_stats``; exact float64 round-trips).
SIGMA_PINS: dict[str, float | int] = {
    "n": 20_834_171,
    "median": 0.0393412950539589,
    "min": 0.0005317263419419,
    "n_lt_5e-3": 231_098,
    "n_lt_1e-2": 235_731,
}
_PRUNE_M_MIN = 1.0e4  # committed m4 recipe (load_pruned_zm)
_PRUNE_M_MAX = 1.0e7
_PRUNE_Z_MAX = 1.5

_N_SIGMA_DECILES = 10  # VT-D3 z-decile matching (house dl_binned pattern)
_HORIZON_FACTOR = 0.999  # VT-D1 horizon-drop guard edge
_HORIZON_GUARD_FRACTION = 0.05  # abort (d)
DEFAULT_CHUNK_PAIRS = 16384  # divergence 2 — pair-row chunk target (memory only)
_G_NODE_CHUNK = 200_000  # divergence 2 — flattened-node cap per g_i call (memory only)

# Seed-plan envelopes (VT-D7; unit-tested disjointness).
V1_SEED_OFFSET_ENVELOPE: tuple[int, int] = (0, 9049)
V2_SEED_OFFSET_ENVELOPE: tuple[int, int] = (20000, 29049)
V3_SEED_OFFSET_ENVELOPE: tuple[int, int] = (40000, 45399)
RESERVED_SEED_OFFSET_BLOCKS: dict[str, tuple[int, int]] = {
    "W1": (46000, 46399),  # rate-weights arm — NOT built (prereg §9 item 3)
    "O2": (47000, 47399),  # volume_deconv arm — NOT built (prereg §9 item 2)
}

# DS-VT3 registered dose-ratio band (prereg §7; committed v2 anchors
# [0.997, 1.095] with margin for mixture/Jensen + population shift).
R_DOSE_BAND: tuple[float, float] = (0.75, 1.25)
_RAIL_EMERGENT_THRESHOLD = 0.90  # DS-VT4 pre-named distinct pattern
# V-T1 T-0 anchor edges (prereg §10).
_VT1_BIAS_PASS = 0.010
_VT1_BIAS_HARD = 0.030
_VT1_RAIL_MAX = 0.05


# ── Cell registry (prereg §5, verbatim) ──────────────────────────────────────


@dataclass(frozen=True)
class VenueCellSpec:
    """One prereg §5 cell: ball mode, σ_z mode, truths and seed blocks.

    Attributes:
        name: Module cell id (``T0``/``Ta``/``Tb``/``Tc`` — divergence 4).
        prereg_cell: The prereg §5 spelling (``T-0``…).
        balls: ``"real_k"`` (pinned ``K_i``, VT-D2) or ``"poisson4"`` (the
            gate's λ = 4 caricature, T-a axis arm).
        sigma_mode: ``"zero"`` (T-0 anchor), ``"flat035"`` (v2 B2 dose) or
            ``"glade"`` (VT-D3 z-decile empirical sampler).
        truths: Injected ``h_true`` values.
        n_seeds: Registered seeds per truth (aligned with ``truths``).
        seed_offsets: Per-truth offsets from :data:`VT_BASE_SEED`.
    """

    name: str
    prereg_cell: str
    balls: str
    sigma_mode: str
    truths: tuple[float, ...]
    n_seeds: tuple[int, ...]
    seed_offsets: tuple[int, ...]


CELL_SPECS: dict[str, VenueCellSpec] = {
    "T0": VenueCellSpec("T0", "T-0", "real_k", "zero", (0.730,), (200,), (40000,)),
    "Ta": VenueCellSpec("Ta", "T-a", "poisson4", "flat035", (0.730,), (200,), (41000,)),
    "Tb": VenueCellSpec("Tb", "T-b", "real_k", "flat035", (0.730,), (200,), (42000,)),
    "Tc": VenueCellSpec(
        "Tc",
        "T-c",
        "real_k",
        "glade",
        (0.690, 0.730, 0.770),
        (200, 400, 200),
        (43000, 44000, 45000),
    ),
}
# W1 (+46000) and O2 (+47000) are registered but NOT built — NOT-EVALUABLE
# per prereg §9 items 3 and 2; reserved blocks are never run post-hoc.


@dataclass(frozen=True)
class VenueConfig:
    """Frozen configuration of one venue-transfer cell run.

    Attributes:
        cell: Cell id (``T0``/``Ta``/``Tb``/``Tc`` or ``custom``).
        h_true: Injected Hubble parameter of this cell×truth.
        balls: ``"real_k"`` or ``"poisson4"`` (see :class:`VenueCellSpec`).
        sigma_mode: ``"zero"``, ``"flat035"`` or ``"glade"``.
        flat_sigma_z: The flat dose (0.035, the v2 B2 dose; prereg §4 step 4).
        lambda_poisson: Poisson mean for the T-a arm (4.0, the v2 caricature).
        crb_reference_csv: The pinned production CRB CSV (VT-D1).
        frozeng_emit_json: The pinned frozeng per-galaxy emit (VT-D2).
        pruned_catalogue_csv: The iiib production catalogue (VT-D3 recipe
            input).
        injection_data_dir: Injection pool defining ``S_4D`` (inherited
            estimator config, v2 §5 verbatim).
        n_events_cap: Smoke/validate-only truncation of the pinned event list
            (divergence 9); ``None`` on registered runs.
        chunk_pairs: Event-aligned chunk target (divergence 2; never changes
            any statistic).
        h_grid: Canonical 41-point grid (prereg §4 step 6).
    """

    cell: str
    h_true: float
    balls: str
    sigma_mode: str
    flat_sigma_z: float = 0.035
    lambda_poisson: float = 4.0
    crb_reference_csv: str = CRB_CSV_PATH
    frozeng_emit_json: str = FROZENG_EMIT_JSON
    pruned_catalogue_csv: str = PRUNED_CATALOGUE_CSV
    injection_data_dir: str = cl.DEFAULT_INJECTION_DIR
    n_events_cap: int | None = None
    chunk_pairs: int = DEFAULT_CHUNK_PAIRS
    h_grid: tuple[float, ...] = cl.CANONICAL_H_GRID


def venue_cell_seeds(
    spec: VenueCellSpec, h_true: float, start: int, count: int | None
) -> list[int]:
    """Seeds of one cell×truth block, optionally chunked (mirrors the gate's).

    Args:
        spec: The cell spec.
        h_true: The truth (must be one of the spec's).
        start: Offset within the block (chunking; 0 = block start).
        count: Number of seeds (``None`` = the rest of the block).

    Returns:
        Absolute seed list.
    """
    idx = spec.truths.index(h_true)
    base = VT_BASE_SEED + spec.seed_offsets[idx]
    n_block = spec.n_seeds[idx]
    n = n_block - start if count is None else count
    if start < 0 or start + n > n_block:
        raise ValueError(
            f"seed chunk [{start}, {start + n}) exceeds cell {spec.name} block of {n_block}"
        )
    return [base + start + i for i in range(n)]


# ── V-T3 pin integrity ───────────────────────────────────────────────────────


def _md5_of_file(path: str, chunk: int = 1 << 22) -> str:
    """MD5 hex digest of a file (streamed).

    Args:
        path: File path.
        chunk: Read block size.

    Returns:
        The hex digest.
    """
    h = hashlib.md5()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def load_pinned_k(emit_path: str) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Per-event 1D ball counts from the frozeng per-galaxy emit (VT-D2).

    ``K_i = len(galaxy_likelihoods[i]) + len(additional_galaxies_without_bh_mass[i])``
    (disjoint lists, h-invariant — committed ``m4_results.json``
    ``iiib.validation``). Events are the production-evaluated CRB row indices.

    Args:
        emit_path: The pinned ``h_0_73.json`` emit.

    Returns:
        ``(event_rows, K)``: ascending CRB row indices of every evaluated
        event and the aligned 1D ball counts.
    """
    with open(emit_path) as fh:
        doc = json.load(fh)
    gl = doc["galaxy_likelihoods"]
    add = doc["additional_galaxies_without_bh_mass"]
    rows = np.asarray(sorted(int(k) for k in gl), dtype=np.int64)
    counts = np.asarray([len(gl[str(r)]) + len(add[str(r)]) for r in rows], dtype=np.int64)
    return rows, counts


def k_census(K: npt.NDArray[np.int64]) -> dict[str, float | int]:
    """The VT-D2 census statistics of a per-event K array.

    Args:
        K: Per-event 1D ball counts over the evaluated event set.

    Returns:
        The census dict, keys as :data:`K_CENSUS_PINS`.
    """
    nonempty = K[K > 0]
    return {
        "n_events_evaluated": int(K.size),
        "zeros": int(np.sum(K == 0)),
        "ones": int(np.sum(K == 1)),
        "median": float(np.median(K)),
        "mean": float(np.mean(K)),
        "p99": float(np.percentile(K, 99)),
        "max": int(np.max(K)),
        "sum_K": int(np.sum(K)),
        "nonempty_n": int(nonempty.size),
        "nonempty_median": float(np.median(nonempty)),
        "nonempty_mean": float(np.mean(nonempty)),
    }


def load_pruned_z_sigma(
    catalogue_csv: str,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """The iiib pruned-frame ``(z, sigma_z)`` arrays via the committed m4 recipe.

    Replicates ``crossterm_instrument/m4_shared_galaxy_census.py::load_pruned_zm``
    (production column parse → ``_empiric_stellar_mass_to_BH_mass_relation`` →
    NaN drop → ``_mass_redshift_prune_mask`` at M ∈ [1e4, 1e7], z ≤ 1.5) by
    calling the production functions themselves (VT-D3).

    Args:
        catalogue_csv: The production reduced-catalogue CSV.

    Returns:
        ``(z, sigma_z)`` of the pruned frame, positional order.
    """
    import pandas as pd

    from darksiren_emri.galaxy_catalogue.handler import (
        _empiric_stellar_mass_to_BH_mass_relation,
        _mass_redshift_prune_mask,
        _reduced_catalog_column_names,
    )

    names = _reduced_catalog_column_names()
    cat = pd.read_csv(catalogue_csv, names=names, usecols=[3, 4, 5, 6])
    z = cat["REDSHIFT"].to_numpy(dtype=np.float64)
    sz = cat["REDSHIFT_MEASUREMENT_ERROR"].to_numpy(dtype=np.float64)
    ms = cat["STELLAR_MASS"].to_numpy(dtype=np.float64)
    mse = cat["STELLAR_MASS_ABSOULTE_ERROR"].to_numpy(dtype=np.float64)
    del cat
    # The production relation is float-annotated but numpy-vectorized (the
    # committed m4 recipe calls it on arrays; same here).
    mbh_raw, mbh_err_raw = _empiric_stellar_mass_to_BH_mass_relation(ms, mse)
    mbh = np.asarray(mbh_raw, dtype=np.float64)
    mbh_err = np.asarray(mbh_err_raw, dtype=np.float64)
    del ms, mse
    keep = ~np.isnan(mbh)
    z, sz, mbh, mbh_err = z[keep], sz[keep], mbh[keep], mbh_err[keep]
    mask = _mass_redshift_prune_mask(
        pd.Series(mbh),
        pd.Series(mbh_err),
        pd.Series(z),
        pd.Series(sz),
        _PRUNE_M_MIN,
        _PRUNE_M_MAX,
        _PRUNE_Z_MAX,
    ).to_numpy()
    return z[mask].astype(np.float64), sz[mask].astype(np.float64)


def sigma_stats(sz: npt.NDArray[np.float64]) -> dict[str, float | int]:
    """The VT-D3 pruned-frame σ_z statistics.

    Args:
        sz: Per-galaxy σ_z of the pruned frame.

    Returns:
        Stats dict, keys as :data:`SIGMA_PINS`.
    """
    return {
        "n": int(sz.size),
        "median": float(np.median(sz)),
        "min": float(np.min(sz)),
        "n_lt_5e-3": int(np.sum(sz < 5.0e-3)),
        "n_lt_1e-2": int(np.sum(sz < 1.0e-2)),
    }


def check_pin_integrity(vcfg: VenueConfig) -> dict[str, Any]:
    """Run the V-T3 pin-integrity block: md5s + K census + σ_z sampler pins.

    Every registered cell run executes this before any seed and embeds the
    recomputed block in its output JSON; any mismatch is a STOP (abort (c)).

    Args:
        vcfg: The venue configuration (supplies the pinned paths).

    Returns:
        The pin-integrity block with per-check ``match`` booleans and an
        overall ``pass`` flag.
    """
    crb_md5 = _md5_of_file(vcfg.crb_reference_csv)
    emit_md5 = _md5_of_file(vcfg.frozeng_emit_json)
    _rows, K = load_pinned_k(vcfg.frozeng_emit_json)
    census = k_census(K)
    census_match = {k: census[k] == K_CENSUS_PINS[k] for k in K_CENSUS_PINS}
    _z, sz = load_pruned_z_sigma(vcfg.pruned_catalogue_csv)
    stats = sigma_stats(sz)
    sigma_match = {k: stats[k] == SIGMA_PINS[k] for k in SIGMA_PINS}
    ok = (
        crb_md5 == CRB_CSV_MD5
        and emit_md5 == FROZENG_EMIT_MD5
        and all(census_match.values())
        and all(sigma_match.values())
    )
    return {
        "pass": bool(ok),
        "crb_csv_md5": {"value": crb_md5, "pin": CRB_CSV_MD5, "match": crb_md5 == CRB_CSV_MD5},
        "frozeng_emit_md5": {
            "value": emit_md5,
            "pin": FROZENG_EMIT_MD5,
            "match": emit_md5 == FROZENG_EMIT_MD5,
        },
        "k_census": {"value": census, "pins": K_CENSUS_PINS, "match": census_match},
        "sigma_stats": {"value": stats, "pins": SIGMA_PINS, "match": sigma_match},
    }


# ── Context ──────────────────────────────────────────────────────────────────


@dataclass
class VenueContext:
    """Per-process shared, seed-independent tables for one cell×truth.

    Wraps the gate's :class:`~darksiren_emri.validation.calibration_gate.GateContext`
    (parent tables: ``alpha(h)``, ``z_of_dl`` ladders, impostor ``w_pop``
    CDF, GL/GH quadrature) and adds the venue-only pinned arrays.
    """

    vcfg: VenueConfig
    gctx: cg.GateContext
    event_rows: npt.NDArray[np.int64]  # CSV row indices of the pinned run set (ascending)
    d_L: npt.NDArray[np.float64]
    M_row: npt.NDArray[np.float64]
    sigma_dL: npt.NDArray[np.float64]
    sigma_Mz: npt.NDArray[np.float64]
    rho: npt.NDArray[np.float64]
    z_true: npt.NDArray[np.float64]
    K: npt.NDArray[np.int64]  # pinned per-event 1D ball counts (>= 1)
    n_horizon_dropped: int
    pin_integrity: dict[str, Any] = field(default_factory=dict)
    # VT-D3 sampler tables (glade mode; empty otherwise).
    z_decile_edges: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(0, dtype=np.float64)
    )
    sigma_pool_deciles: list[npt.NDArray[np.float64]] = field(default_factory=list)


def _load_pinned_rows(
    csv_path: str,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    """Row-aligned ``(d_L, M, sigma_dL, sigma_Mz, rho)`` from the pinned CRB CSV.

    Applies the parent's triples filter and REQUIRES that every row passes it
    (verified at design time: all 1590 rows pass), so CSV row index ==
    frozeng-emit event index stays exact (VT-D1).

    Args:
        csv_path: The pinned ``prepared_cramer_rao_bounds.csv``.

    Returns:
        Arrays over all CSV rows, positional order.

    Raises:
        ValueError: If any row fails the parent filter (alignment would break).
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
    if not bool(np.all(ok)):
        bad = np.where(~ok)[0][:10].tolist()
        raise ValueError(
            f"CRB CSV rows {bad} fail the parent triples filter — row alignment "
            "with the frozeng emit would break (VT-D1 requires all rows pass)"
        )
    return d_L, M, s_d.astype(np.float64), s_m.astype(np.float64), rho.astype(np.float64)


def build_sigma_sampler(
    z: npt.NDArray[np.float64], sz: npt.NDArray[np.float64]
) -> tuple[npt.NDArray[np.float64], list[npt.NDArray[np.float64]]]:
    """Build the VT-D3 z-decile σ_z sampler tables.

    Rank-based deciles of the pruned frame's z distribution (house
    ``dl_binned`` pattern); pool ``b`` holds the σ_z of the galaxies whose z
    rank falls in decile ``b``.

    Args:
        z: Pruned-frame redshifts.
        sz: Aligned pruned-frame σ_z.

    Returns:
        ``(edges, pools)``: the 9 internal decile edges (z values) and the 10
        σ_z pools.
    """
    order = np.argsort(z, kind="stable")
    z_sorted = z[order]
    sz_sorted = sz[order]
    n = z.size
    bounds = [(n * b) // _N_SIGMA_DECILES for b in range(_N_SIGMA_DECILES + 1)]
    edges = z_sorted[np.asarray(bounds[1:-1], dtype=np.int64)]
    pools = [sz_sorted[bounds[b] : bounds[b + 1]].copy() for b in range(_N_SIGMA_DECILES)]
    return edges.astype(np.float64), pools


def draw_member_sigma_z(
    vctx: VenueContext, z_members: npt.NDArray[np.float64], rng: np.random.Generator
) -> npt.NDArray[np.float64]:
    """Draw one empirical σ_z per ball member, z-decile-matched (VT-D3).

    The member's true z selects the decile (``searchsorted`` on the edges;
    members beyond the last edge map to the top decile — divergence 8); the
    σ_z is drawn uniformly from that decile's pool. Decile-by-decile RNG
    consumption mirrors the gate's texture pattern (deterministic given the
    member order).

    Args:
        vctx: The venue context (sampler tables).
        z_members: True member redshifts, flattened pair order.
        rng: Seeded generator.

    Returns:
        Per-member σ_z, same shape.
    """
    if not vctx.sigma_pool_deciles:
        raise RuntimeError("sigma sampler tables not built (sigma_mode != 'glade'?)")
    dec = np.searchsorted(vctx.z_decile_edges, z_members, side="right")
    out = np.empty(z_members.size, dtype=np.float64)
    for b in range(_N_SIGMA_DECILES):
        m = dec == b
        if np.any(m):
            pool = vctx.sigma_pool_deciles[b]
            out[m] = pool[rng.integers(0, pool.size, size=int(m.sum()))]
    return out


def build_venue_context(vcfg: VenueConfig, *, check_pins: bool = True) -> VenueContext:
    """Build the parent gate context plus the pinned venue tables.

    Order: V-T3 pin integrity (STOP on mismatch when ``check_pins``), pinned
    rows + K, event restriction (nonempty 1D ball, VT-D5), horizon-drop guard
    (VT-D1, abort (d)), truth-ladder ``z_true`` inversion, σ_z sampler tables
    (glade mode), then the gate context (``alpha`` tables etc.).

    Args:
        vcfg: The venue configuration.
        check_pins: Run and enforce V-T3 (always True on registered runs).

    Returns:
        A ready :class:`VenueContext`.

    Raises:
        SystemExit: On a V-T3 pin mismatch or a fired horizon guard.
    """
    pin_block: dict[str, Any] = {}
    if check_pins:
        pin_block = check_pin_integrity(vcfg)
        if not pin_block["pass"]:
            raise SystemExit(
                "STOP: V-T3 pin-integrity FAILURE (prereg §10 abort (c)) — "
                + json.dumps(
                    {
                        k: v
                        for k, v in pin_block.items()
                        if isinstance(v, dict) and not v.get("match", v.get("pass", True))
                    },
                    default=str,
                )[:2000]
            )

    d_L_all, M_all, s_d_all, s_m_all, rho_all = _load_pinned_rows(vcfg.crb_reference_csv)
    rows_eval, K_eval = load_pinned_k(vcfg.frozeng_emit_json)
    if int(np.max(rows_eval)) >= d_L_all.size:
        raise SystemExit("STOP: frozeng emit event index exceeds CRB CSV rows (VT-D1 pin broken)")

    # VT-D5: restrict to nonempty production 1D balls (982 events).
    nonempty = K_eval > 0
    rows = rows_eval[nonempty]
    K = K_eval[nonempty]

    # Gate context (alpha tables, ladders, impostor w_pop CDF, quadrature).
    gcfg = cg.GateConfig(
        cell=vcfg.cell,
        h_true=vcfg.h_true,
        ball=True,
        lambda_ball=vcfg.lambda_poisson if vcfg.balls == "poisson4" else 0.0,
        sigma_z=vcfg.flat_sigma_z if vcfg.balls == "poisson4" else 0.0,
        sigma_texture="dl_binned",
        n_events=int(rows.size),
        injection_data_dir=vcfg.injection_data_dir,
        crb_reference_csv=vcfg.crb_reference_csv,
        h_grid=vcfg.h_grid,
    )
    gctx = cg.build_gate_context(gcfg)

    # VT-D1 horizon guard at the cell truth.
    dl_max = float(gctx.cl_ctx.detection.get_dl_max(vcfg.h_true))
    drop = d_L_all[rows] > _HORIZON_FACTOR * dl_max
    n_dropped = int(np.sum(drop))
    if rows.size > 0 and n_dropped / rows.size > _HORIZON_GUARD_FRACTION:
        raise SystemExit(
            f"STOP: horizon-drop guard fired — {n_dropped}/{rows.size} pinned events "
            f"exceed {_HORIZON_FACTOR} x dl_max({vcfg.h_true}) = "
            f"{_HORIZON_FACTOR * dl_max:.4f} (> {_HORIZON_GUARD_FRACTION:.0%}; "
            "prereg §10 abort (d) => VENUE-CONFOUNDED)"
        )
    rows = rows[~drop]
    K = K[~drop]

    if vcfg.n_events_cap is not None:
        rows = rows[: vcfg.n_events_cap]
        K = K[: vcfg.n_events_cap]

    # z_true by truth-ladder inversion (the parent's _z_of_dl_table device).
    d_nodes, z_nodes = cl._z_of_dl_table(vcfg.h_true, 6.0)
    z_true = np.interp(d_L_all[rows], d_nodes, z_nodes)

    edges = np.zeros(0, dtype=np.float64)
    pools: list[npt.NDArray[np.float64]] = []
    if vcfg.sigma_mode == "glade":
        z_cat, sz_cat = load_pruned_z_sigma(vcfg.pruned_catalogue_csv)
        edges, pools = build_sigma_sampler(z_cat, sz_cat)
        del z_cat, sz_cat

    return VenueContext(
        vcfg=vcfg,
        gctx=gctx,
        event_rows=rows,
        d_L=d_L_all[rows],
        M_row=M_all[rows],
        sigma_dL=s_d_all[rows],
        sigma_Mz=s_m_all[rows],
        rho=rho_all[rows],
        z_true=z_true.astype(np.float64),
        K=K,
        n_horizon_dropped=n_dropped,
        pin_integrity=pin_block,
        z_decile_edges=edges,
        sigma_pool_deciles=pools,
    )


# ── Generator: pinned-K balls (VT-D2) ────────────────────────────────────────


def draw_ball_pinned(
    vctx: VenueContext,
    universe: cl.SyntheticUniverse,
    rng: np.random.Generator,
) -> cg.HostBall:
    """Draw the candidate balls with per-event PINNED multiplicity (VT-D2).

    The gate's :func:`~darksiren_emri.validation.calibration_gate.draw_ball`
    window and impostor machinery verbatim, with two registered changes:
    ``n_impostors,i = K_i - 1`` (pinned, no Poisson draw) and NO σ_z applied
    here — the σ_z texture is the separate prereg §4 step 4 (divergence 7).
    A degenerate window (zero population mass) gets ``n_impostors = 0``
    (host-only ball; counted — the gate's divergence-5 convention carried).

    Args:
        vctx: The venue context (pinned ``K``; impostor tables via ``gctx``).
        universe: The pinned event set with this seed's observation noise.
        rng: Seeded generator.

    Returns:
        The :class:`~darksiren_emri.validation.calibration_gate.HostBall`
        with ``z_obs`` = TRUE member redshifts (σ_z texture applied by the
        caller).
    """
    gctx = vctx.gctx
    n = universe.z_true.size

    d_lo = universe.d_L_obs * (1.0 - cl._SIGMA_WINDOW * universe.sigma_dL)
    d_hi = universe.d_L_obs * (1.0 + cl._SIGMA_WINDOW * universe.sigma_dL)
    z_lo = np.interp(np.maximum(d_lo, 0.0), gctx.imp_dl_nodes, gctx.imp_z_nodes)
    z_hi = np.interp(d_hi, gctx.imp_dl_nodes, gctx.imp_z_nodes)
    F_lo = np.interp(z_lo, gctx.imp_z_nodes, gctx.imp_z_cdf)
    F_hi = np.interp(z_hi, gctx.imp_z_nodes, gctx.imp_z_cdf)

    n_imp = np.maximum(vctx.K.astype(np.int64) - 1, 0)
    degenerate = F_hi <= F_lo
    n_imp = np.where(degenerate, 0, n_imp)
    total_imp = int(n_imp.sum())

    imp_event = np.repeat(np.arange(n, dtype=np.int64), n_imp)
    u = rng.random(total_imp)
    u_scaled = F_lo[imp_event] + (F_hi[imp_event] - F_lo[imp_event]) * u
    z_imp = np.interp(u_scaled, gctx.imp_z_cdf, gctx.imp_z_nodes)

    z_all = np.concatenate([universe.z_true, z_imp])
    ev_all = np.concatenate([np.arange(n, dtype=np.int64), imp_event])
    key = rng.random(z_all.size)
    order = np.lexsort((key, ev_all))
    z_all = z_all[order]
    ev_all = ev_all[order]

    K_real = np.bincount(ev_all, minlength=n).astype(np.int64)
    return cg.HostBall(
        z_obs=z_all.copy(),
        event_idx=ev_all,
        K=K_real,
        n_impostors_total=total_imp,
        n_degenerate_windows=int(degenerate.sum()),
    )


# ── Estimator: vector-σ generalization of the gate's ball path ───────────────


def _pair_chunks(n_pairs: int, chunk_pairs: int) -> list[tuple[int, int]]:
    """Split the flattened pair range into fixed-size row chunks.

    Chunking is memory-only and deterministic (divergence 2): every per-row
    estimator operation is row-independent and the per-h event sums run once
    over the full-length arrays; chunk shape can move ``@``-reduction
    accumulation order by O(1 ULP) (BLAS), which is why registered runs use
    one fixed geometry and V-T5 runs the exact mode (``chunk_pairs = 0``).

    Args:
        n_pairs: Total pair rows.
        chunk_pairs: Rows per chunk (``<= 0`` = one chunk).

    Returns:
        Half-open ``(start, stop)`` pair-index ranges covering all pairs.
    """
    if n_pairs == 0:
        return []
    if chunk_pairs <= 0:
        return [(0, n_pairs)]
    return [(a, min(a + chunk_pairs, n_pairs)) for a in range(0, n_pairs, chunk_pairs)]


def _g_ball_capped(
    gctx: cg.GateContext,
    universe: cl.SyntheticUniverse,
    event_idx: npt.NDArray[np.int64],
    z_nodes: npt.NDArray[np.float64],
    d_L_frac: npt.NDArray[np.float64],
    valid: npt.NDArray[np.bool_],
    *,
    node_chunk: int = _G_NODE_CHUNK,
) -> npt.NDArray[np.float64]:
    """Memory-capped mirror of the gate's ``_g_ball`` (bit-identical output).

    Same conditional-Gaussian parameters and per-event verbatim
    :func:`completion_mass_factor_g` calls as
    :func:`calibration_gate._g_ball`, with two memory-only changes: the loop
    visits only the events present in ``event_idx`` (a chunk may hold few),
    and each event's flattened node set is split so the internal
    ``(nodes, n_hermite)`` intermediate stays ``<= node_chunk`` rows — the
    peak venue event (K = 245,364 → ~12.3M nodes × 64 Hermite nodes) would
    otherwise allocate ~19 GB across the intermediates. With
    ``node_chunk <= 0`` (exact mode) each event goes in ONE gate-shaped call
    and the output is bit-identical to ``calibration_gate._g_ball``
    (unit-tested); with a finite cap the split moves the internal
    matrix-vector accumulation by O(1 ULP) (shape-dependent BLAS kernels,
    divergence 2) — deterministic for a fixed cap.

    Args:
        gctx: The gate context (supplies the Gauss-Hermite order).
        universe: The event set (per-event ``M_z_obs`` and 2x2 block).
        event_idx: Nondecreasing event index per candidate row (chunk slice).
        z_nodes: ``(n_rows, n_quad)`` quadrature redshifts.
        d_L_frac: ``(n_rows, n_quad)`` values of ``d_L(z;h)/d_L_obs``.
        valid: ``(n_rows,)`` rows with a nonempty integration window.
        node_chunk: Flattened-node cap per ``completion_mass_factor_g`` call
            (``<= 0`` = exact gate-shape mode, no splitting).

    Returns:
        ``g`` at the nodes, shape ``(n_rows, n_quad)``; invalid rows are 0.
    """
    s_dd = universe.sigma_dL**2
    s_dm = universe.rho * universe.sigma_dL * universe.sigma_Mz
    s_mm = universe.sigma_Mz**2
    proj = np.where(s_dd > 0.0, s_dm / np.maximum(s_dd, 1e-300), 0.0)
    sigma_cond = np.sqrt(np.maximum(s_mm - proj * s_dm, 1e-30))

    out = np.zeros_like(z_nodes)
    n_hermite = gctx.cl_ctx.config.n_hermite
    n_quad = z_nodes.shape[1]
    if node_chunk <= 0:
        max_rows_per_call = int(z_nodes.shape[0]) + 1  # exact mode: one call per event
    else:
        max_rows_per_call = max(node_chunk // max(n_quad, 1), 1)
    present = np.unique(event_idx)
    starts = np.searchsorted(event_idx, present, side="left")
    stops = np.searchsorted(event_idx, present, side="right")
    for i, s, e in zip(present.tolist(), starts.tolist(), stops.tolist(), strict=True):
        rows = np.arange(s, e)
        rows = rows[valid[rows]]
        if rows.size == 0:
            continue
        for r0 in range(0, rows.size, max_rows_per_call):
            rr = rows[r0 : r0 + max_rows_per_call]
            zz = z_nodes[rr].reshape(-1)
            ff = d_L_frac[rr].reshape(-1)
            out[rr] = completion_mass_factor_g(
                zz,
                ff,
                float(universe.M_z_obs[i]),
                float(proj[i]),
                float(sigma_cond[i]),
                n_hermite=n_hermite,
            ).reshape(rr.size, n_quad)
    return out


def log_channel_posteriors_ball_sigma_vector(
    gctx: cg.GateContext,
    universe: cl.SyntheticUniverse,
    ball: cg.HostBall,
    sigma_z_pairs: npt.NDArray[np.float64],
    *,
    chunk_pairs: int = DEFAULT_CHUNK_PAIRS,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    r"""Both channels' unnormalised log posteriors with PER-CANDIDATE σ_z.

    The gate's
    :func:`~darksiren_emri.validation.calibration_gate.log_channel_posteriors_ball`
    generalized to a σ_z vector (prereg §4 step 5, VT-D3 matched-model):

    .. math::

        L_i(h) = \frac{1}{K_i} \sum_k \int \mathrm{d}z\,
            \mathcal{N}(z; z_{obs,k}, \sigma_{z,k})\,
            \mathcal{N}\!\bigl(d_L(z;h)/d_L^{obs}_i; 1, \sigma_{dL,i}\bigr)
            \,[\,g_i(z;h)\,]

    per-candidate 50-node Gauss-Legendre on
    ``[max(z_lo(h), z_obs,k - 5 sigma_z,k), min(z_hi(h), z_obs,k + 5 sigma_z,k)]``
    (``sigma_z,k = 0`` ⇒ point evaluation at ``z_obs,k``), ``[z_lo, z_hi]``
    the production ±4σ window capped at ``z_max(h)``; both channels subtract
    ``N_det ln alpha(h)`` with N FIXED and the finite −745 zero-event penalty
    (the gate's divergence-10 convention, verbatim). In exact mode
    (``chunk_pairs = 0``) a constant σ vector bit-reproduces the gate's
    scalar path (V-T5; unit-tested); chunked mode is deterministic and agrees
    to O(1 ULP) (divergence 2).

    Args:
        gctx: The gate context (ladders, ``alpha``, quadrature, ``g_i``).
        universe: The event set.
        ball: The candidate balls (``z_obs`` already σ_z-scattered).
        sigma_z_pairs: Per-candidate σ_z, aligned with ``ball.z_obs``.
        chunk_pairs: Pair-row chunk target (divergence 2; ``0`` = exact
            gate-shape mode, single chunk + unsplit ``g`` calls).

    Returns:
        ``(ln_post_1d, ln_post_2d, sum_dlog_gfrac_dh)`` exactly as the gate's.
    """
    cfg = gctx.cl_ctx.config
    n_h = len(cfg.h_grid)
    ln1 = np.zeros(n_h, dtype=np.float64)
    ln2 = np.zeros(n_h, dtype=np.float64)
    ln_gfrac = np.zeros(n_h, dtype=np.float64)

    sig_z = np.asarray(sigma_z_pairs, dtype=np.float64)
    if sig_z.shape != ball.z_obs.shape:
        raise ValueError(f"sigma_z_pairs shape {sig_z.shape} != pairs shape {ball.z_obs.shape}")

    for k in range(n_h):
        ln1[k], ln2[k], ln_gfrac[k] = _channel_terms_at_h(
            gctx, universe, ball, sig_z, k, chunk_pairs=chunk_pairs
        )

    return ln1, ln2, _slope_at_truth(cfg, ln_gfrac)


def _slope_at_truth(
    cfg: cl.ClosedLoopConfig, ln_gfrac: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """``sum_dlog_gfrac_dh`` from neighbouring grid values (verbatim math).

    Args:
        cfg: The closed-loop config (grid + truth).
        ln_gfrac: Per-h ``Σ ln(L2/L1)`` over both-finite events.

    Returns:
        The 1-element slope array, exactly as the gate's tail computes it.
    """
    n_h = len(cfg.h_grid)
    h_arr = np.asarray(cfg.h_grid, dtype=np.float64)
    i_true = int(np.argmin(np.abs(h_arr - cfg.h_true)))
    lo = max(i_true - 1, 0)
    hi = min(i_true + 1, n_h - 1)
    slope = (ln_gfrac[hi] - ln_gfrac[lo]) / (h_arr[hi] - h_arr[lo])
    return np.asarray([slope], dtype=np.float64)


def _channel_terms_at_h(
    gctx: cg.GateContext,
    universe: cl.SyntheticUniverse,
    ball: cg.HostBall,
    sig_z: npt.NDArray[np.float64],
    k: int,
    *,
    chunk_pairs: int = DEFAULT_CHUNK_PAIRS,
) -> tuple[float, float, float]:
    """One h-point of the vector-σ estimator loop (the divergence-11 unit).

    The per-h body of :func:`log_channel_posteriors_ball_sigma_vector`,
    verbatim: it consumes no RNG, depends on no cross-h state (the
    ``z_of_dl_tables[k]`` ladder is precomputed context), and runs the SAME
    fixed chunk geometry as the serial loop (``_pair_chunks`` partition +
    :data:`_G_NODE_CHUNK` cap) — identical array shapes, hence identical
    BLAS kernels, hence bit-identical floats whichever process executes it.
    The tiny per-call recomputation of exact indexing products
    (``d_obs_p = d_L_obs[ev]`` etc.) is FP-free.

    Args:
        gctx: The gate context.
        universe: The event set.
        ball: The candidate balls.
        sig_z: Per-candidate σ_z as a validated float64 array.
        k: Index into the h grid.
        chunk_pairs: Pair-row chunk target (divergence 2; ``0`` = exact
            gate-shape mode).

    Returns:
        ``(ln1[k], ln2[k], ln_gfrac[k])`` for this h.
    """
    cfg = gctx.cl_ctx.config
    h = cfg.h_grid[k]
    n = universe.z_true.size
    x = gctx.cl_ctx.gl_nodes
    w_gl = gctx.cl_ctx.gl_weights
    ev = ball.event_idx
    z_obs = ball.z_obs
    d_obs_e = universe.d_L_obs
    sig_e = universe.sigma_dL
    d_obs_p = d_obs_e[ev]
    sig_p = sig_e[ev]
    K = np.maximum(ball.K, 1)
    n_pairs = int(z_obs.size)
    chunks = _pair_chunks(n_pairs, chunk_pairs)
    g_node_chunk = _G_NODE_CHUNK if chunk_pairs > 0 else 0  # exact mode couples both

    d_L_nodes, z_tab = gctx.cl_ctx.z_of_dl_tables[k]
    z_hi_e = np.interp(d_obs_e * (1.0 + cl._SIGMA_WINDOW * sig_e), d_L_nodes, z_tab)
    z_lo_e = np.interp(d_obs_e * (1.0 - cl._SIGMA_WINDOW * sig_e), d_L_nodes, z_tab)
    z_lo_e = np.maximum(z_lo_e, 1e-6)
    z_hi_e = np.minimum(z_hi_e, z_tab[-1])
    z_lo_p = z_lo_e[ev]
    z_hi_p = z_hi_e[ev]

    c1 = np.zeros(n_pairs, dtype=np.float64)
    c2 = np.zeros(n_pairs, dtype=np.float64)
    for a0, a1 in chunks:
        sl = np.arange(a0, a1, dtype=np.int64)
        sig_c = sig_z[sl]
        q = sig_c > 0.0
        if np.any(q):
            rows_q = sl[q]
            zo = z_obs[rows_q]
            so = sig_c[q]
            a = np.maximum(z_lo_p[rows_q], zo - cg._IMPOSTOR_KERNEL_WINDOW * so)
            b = np.minimum(z_hi_p[rows_q], zo + cg._IMPOSTOR_KERNEL_WINDOW * so)
            valid = b > a
            half = 0.5 * (b - a)
            mid = 0.5 * (b + a)
            z_nodes = mid[:, None] + half[:, None] * x[None, :]
            d_L_n = np.asarray(
                dist_vectorized(np.maximum(z_nodes.reshape(-1), 1e-8), h=h),
                dtype=np.float64,
            ).reshape(z_nodes.shape)
            d_L_frac = d_L_n / d_obs_p[rows_q][:, None]
            p_gw = norm.pdf(d_L_frac, loc=1.0, scale=sig_p[rows_q][:, None])
            kern = norm.pdf(z_nodes, loc=zo[:, None], scale=so[:, None])
            integ = kern * p_gw
            c1q = half * (integ @ w_gl)
            g = _g_ball_capped(
                gctx, universe, ev[rows_q], z_nodes, d_L_frac, valid, node_chunk=g_node_chunk
            )
            c2q = half * ((integ * g) @ w_gl)
            c1[rows_q] = np.where(valid, c1q, 0.0)
            c2[rows_q] = np.where(valid, c2q, 0.0)
        if not np.all(q):
            rows_p = sl[~q]
            zo = z_obs[rows_p]
            valid_p = (zo >= z_lo_p[rows_p]) & (zo <= z_hi_p[rows_p])
            d_pt = np.asarray(dist_vectorized(np.maximum(zo, 1e-8), h=h), dtype=np.float64)
            frac = d_pt / d_obs_p[rows_p]
            p_gw_p = norm.pdf(frac, loc=1.0, scale=sig_p[rows_p])
            g_pt = _g_ball_capped(
                gctx,
                universe,
                ev[rows_p],
                zo[:, None],
                frac[:, None],
                valid_p,
                node_chunk=g_node_chunk,
            )[:, 0]
            c1[rows_p] = np.where(valid_p, p_gw_p, 0.0)
            c2[rows_p] = np.where(valid_p, p_gw_p * g_pt, 0.0)

    L1 = np.bincount(ev, weights=c1, minlength=n) / K
    L2 = np.bincount(ev, weights=c2, minlength=n) / K
    # Prereg §4 step 5 normalisation, N_det FIXED (the gate's divergence-10
    # convention verbatim): an event with L_i = 0 excludes this h via the
    # finite -745/event penalty.
    ok1 = (L1 > 0.0) & np.isfinite(L1)
    ok2 = (L2 > 0.0) & np.isfinite(L2)
    lnL1 = np.where(ok1, np.log(np.where(ok1, L1, 1.0)), cg._LN_ZERO_EVENT)
    lnL2 = np.where(ok2, np.log(np.where(ok2, L2, 1.0)), cg._LN_ZERO_EVENT)
    ln1_k = float(np.sum(lnL1)) - float(n) * gctx.cl_ctx.log_alpha[k]
    ln2_k = float(np.sum(lnL2)) - float(n) * gctx.cl_ctx.log_alpha[k]
    both = ok1 & ok2
    ln_gfrac_k = float(np.sum(np.log(L2[both] / L1[both])))
    return float(ln1_k), float(ln2_k), float(ln_gfrac_k)


# ── Divergence 11: intra-seed h-grain parallel estimator ─────────────────────

_H_STATE: (
    tuple[cg.GateContext, cl.SyntheticUniverse, cg.HostBall, npt.NDArray[np.float64], int] | None
) = None


def _h_task(k: int) -> tuple[float, float, float]:
    """Fork-pool task: one h-point read from the module-global state."""
    if _H_STATE is None:
        raise RuntimeError("h-grain task state not initialised (fork-only worker)")
    gctx, universe, ball, sig_z, chunk_pairs = _H_STATE
    return _channel_terms_at_h(gctx, universe, ball, sig_z, k, chunk_pairs=chunk_pairs)


def log_channel_posteriors_ball_sigma_vector_hgrain(
    gctx: cg.GateContext,
    universe: cl.SyntheticUniverse,
    ball: cg.HostBall,
    sigma_z_pairs: npt.NDArray[np.float64],
    *,
    chunk_pairs: int = DEFAULT_CHUNK_PAIRS,
    workers: int = 1,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """h-grain twin of :func:`log_channel_posteriors_ball_sigma_vector`.

    Farms the 41 independent h-points to a fork pool (divergence 11): every
    task runs :func:`_channel_terms_at_h` — the identical per-h code with the
    identical chunk geometry — and returns three scalars; reassembly is array
    indexing and the slope is :func:`_slope_at_truth` verbatim, so the result
    is BIT-IDENTICAL to the serial function for any ``workers``. The large
    inputs reach the workers via fork copy-on-write of a module global (no
    pickling of arrays); ``workers <= 1`` degenerates to the serial loop.

    Args:
        gctx: The gate context.
        universe: The event set.
        ball: The candidate balls (``z_obs`` already σ_z-scattered).
        sigma_z_pairs: Per-candidate σ_z, aligned with ``ball.z_obs``.
        chunk_pairs: Pair-row chunk target (divergence 2; must match the
            geometry being compared against, as in the serial function).
        workers: Fork-pool size (capped at the grid length).

    Returns:
        ``(ln_post_1d, ln_post_2d, sum_dlog_gfrac_dh)`` exactly as the
        serial function's.
    """
    global _H_STATE
    cfg = gctx.cl_ctx.config
    n_h = len(cfg.h_grid)
    sig_z = np.asarray(sigma_z_pairs, dtype=np.float64)
    if sig_z.shape != ball.z_obs.shape:
        raise ValueError(f"sigma_z_pairs shape {sig_z.shape} != pairs shape {ball.z_obs.shape}")

    _H_STATE = (gctx, universe, ball, sig_z, chunk_pairs)
    try:
        if workers > 1:
            ctx_mp = mp.get_context("fork")
            with ctx_mp.Pool(processes=min(workers, n_h)) as pool:
                terms = pool.map(_h_task, range(n_h), chunksize=1)
        else:
            terms = [_h_task(k) for k in range(n_h)]
    finally:
        _H_STATE = None

    ln1 = np.zeros(n_h, dtype=np.float64)
    ln2 = np.zeros(n_h, dtype=np.float64)
    ln_gfrac = np.zeros(n_h, dtype=np.float64)
    for k, (a, b, c) in enumerate(terms):
        ln1[k] = a
        ln2[k] = b
        ln_gfrac[k] = c
    return ln1, ln2, _slope_at_truth(cfg, ln_gfrac)


# ── Per-seed driver ──────────────────────────────────────────────────────────

_VCTX: VenueContext | None = None


def _venue_worker_init(vcfg: VenueConfig) -> None:
    """Inherit (fork) or build the shared venue context in a worker process."""
    global _VCTX
    if _VCTX is None or _VCTX.vcfg != vcfg:
        _VCTX = build_venue_context(vcfg)


def run_seed_venue(seed: int, vctx: VenueContext | None = None) -> dict[str, Any]:
    """Run one conditional (fixed-design) universe end to end (prereg §4).

    Steps, in the registered RNG order: (2) correlated observation noise from
    each pinned row's own 2×2 block; (3) ball — pinned-K
    (:func:`draw_ball_pinned`) or the gate's Poisson λ = 4
    (:func:`calibration_gate.draw_ball`, T-a); (4) σ_z texture per mode;
    (5) vector-σ estimator; (6) readout on the canonical grid.

    Args:
        seed: The seed of this noise + ball + σ_z realization.
        vctx: Shared context; falls back to the process-global one.

    Returns:
        A JSON-serialisable per-seed record: the gate's §6 fields plus the
        prereg §6 venue fields (``sigma_z_mean_pairs``,
        ``sigma_z_median_pairs``, ``frac_pairs_sigma_lt_5e-3``, ``K_sum``,
        ``n_events_run``, ``n_horizon_dropped``).
    """
    context = vctx if vctx is not None else _VCTX
    if context is None:
        raise RuntimeError("venue-transfer context not initialised")
    universe, ball, sigma_pairs = _draw_seed_realization(seed, context)
    ln1, ln2, slope = log_channel_posteriors_ball_sigma_vector(
        context.gctx, universe, ball, sigma_pairs, chunk_pairs=context.vcfg.chunk_pairs
    )
    return _assemble_seed_record(seed, context, universe, ball, sigma_pairs, ln1, ln2, slope)


def run_seed_venue_hgrain(
    seed: int, vctx: VenueContext | None = None, workers: int = 1
) -> dict[str, Any]:
    """h-grain twin of :func:`run_seed_venue` (divergence 11).

    Phase 1 — the registered RNG phase (prereg §4 steps 2–4) runs serially on
    the single seeded stream via :func:`_draw_seed_realization`, so the draws
    are the registered ones by construction. Phase 2 — the RNG-free estimator
    is parallelised over the h grid
    (:func:`log_channel_posteriors_ball_sigma_vector_hgrain`, bit-identical
    to the serial loop). Phase 3 — slope/readout/record assembly run
    unchanged. The record is BYTE-IDENTICAL to :func:`run_seed_venue`'s for
    any ``workers`` (unit-tested).

    Args:
        seed: The seed of this noise + ball + σ_z realization.
        vctx: Shared context; falls back to the process-global one.
        workers: Fork-pool size for the h grid (``<= 1`` = in-process).

    Returns:
        The per-seed record, exactly as :func:`run_seed_venue`'s.
    """
    context = vctx if vctx is not None else _VCTX
    if context is None:
        raise RuntimeError("venue-transfer context not initialised")
    universe, ball, sigma_pairs = _draw_seed_realization(seed, context)
    ln1, ln2, slope = log_channel_posteriors_ball_sigma_vector_hgrain(
        context.gctx,
        universe,
        ball,
        sigma_pairs,
        chunk_pairs=context.vcfg.chunk_pairs,
        workers=workers,
    )
    return _assemble_seed_record(seed, context, universe, ball, sigma_pairs, ln1, ln2, slope)


def _draw_seed_realization(
    seed: int, context: VenueContext
) -> tuple[cl.SyntheticUniverse, cg.HostBall, npt.NDArray[np.float64]]:
    """Prereg §4 steps 2–4 on the single seeded stream (registered RNG order).

    The generation phase of :func:`run_seed_venue`, factored out verbatim so
    the h-grain mode (divergence 11) consumes the IDENTICAL sequential draws:
    noise → ball → σ_z texture, one ``np.random.default_rng(seed)`` consumed
    in the registered order (divergence 7). Consumes ALL of the seed's
    randomness; everything downstream is deterministic.

    Args:
        seed: The realization seed.
        context: The venue context.

    Returns:
        ``(universe, ball, sigma_pairs)`` ready for the vector-σ estimator.
    """
    vcfg = context.vcfg
    gctx = context.gctx
    rng = np.random.default_rng(seed)
    n = context.z_true.size

    # Step 2 — the parent's correlated fractional noise, verbatim, on the
    # pinned rows (M_z_true = M_row (1 + z_true): the committed F5
    # d_L-reinterpretation device, VT-D1).
    e1 = rng.standard_normal(n)
    e2 = rng.standard_normal(n)
    frac_d = context.sigma_dL * e1
    frac_m = context.sigma_Mz * (
        context.rho * e1 + np.sqrt(np.maximum(1.0 - context.rho**2, 0.0)) * e2
    )
    M_z_true = context.M_row * (1.0 + context.z_true)
    universe = cl.SyntheticUniverse(
        z_true=context.z_true,
        M_true=context.M_row,
        d_L_true=context.d_L,
        d_L_obs=context.d_L * (1.0 + frac_d),
        M_z_obs=M_z_true * (1.0 + frac_m),
        sigma_dL=context.sigma_dL,
        sigma_Mz=context.sigma_Mz,
        rho=context.rho,
        in_catalogue=np.zeros(n, dtype=bool),
        n_drawn=n,
    )

    # Steps 3 + 4 — ball, then σ_z texture.
    if vcfg.balls == "poisson4":
        # T-a: the gate's draw verbatim (Poisson λ=4, flat σ_z applied inside).
        ball = cg.draw_ball(gctx, universe, rng)
        sigma_pairs = np.full(ball.z_obs.size, vcfg.flat_sigma_z, dtype=np.float64)
    else:
        ball = draw_ball_pinned(context, universe, rng)
        if vcfg.sigma_mode == "zero":
            sigma_pairs = np.zeros(ball.z_obs.size, dtype=np.float64)
        elif vcfg.sigma_mode == "flat035":
            sigma_pairs = np.full(ball.z_obs.size, vcfg.flat_sigma_z, dtype=np.float64)
            ball.z_obs = ball.z_obs + sigma_pairs * rng.standard_normal(ball.z_obs.size)
        elif vcfg.sigma_mode == "glade":
            sigma_pairs = draw_member_sigma_z(context, ball.z_obs, rng)
            ball.z_obs = ball.z_obs + sigma_pairs * rng.standard_normal(ball.z_obs.size)
        else:
            raise ValueError(f"unknown sigma_mode '{vcfg.sigma_mode}'")

    return universe, ball, sigma_pairs


def _assemble_seed_record(
    seed: int,
    context: VenueContext,
    universe: cl.SyntheticUniverse,
    ball: cg.HostBall,
    sigma_pairs: npt.NDArray[np.float64],
    ln1: npt.NDArray[np.float64],
    ln2: npt.NDArray[np.float64],
    slope: npt.NDArray[np.float64],
) -> dict[str, Any]:
    """Readout + prereg §6 record from precomputed posteriors (RNG-free).

    The tail of :func:`run_seed_venue`, factored out verbatim and shared by
    both grain modes (divergence 11) — deterministic given its inputs.

    Args:
        seed: The realization seed.
        context: The venue context.
        universe: The drawn event set.
        ball: The drawn candidate balls.
        sigma_pairs: Per-candidate σ_z.
        ln1: 1D unnormalised log posterior.
        ln2: 2D unnormalised log posterior.
        slope: ``sum_dlog_gfrac_dh`` scalar array.

    Returns:
        The JSON-serialisable per-seed record.
    """
    vcfg = context.vcfg
    n = context.z_true.size
    h_arr = np.asarray(vcfg.h_grid, dtype=np.float64)
    r1 = cl.posterior_readout(h_arr, ln1)
    r2 = cl.posterior_readout(h_arr, ln2)
    pp1 = cg.pp_readout(h_arr, ln1, vcfg.h_true)
    pp2 = cg.pp_readout(h_arr, ln2, vcfg.h_true)
    with np.errstate(divide="ignore", invalid="ignore"):
        texture_corr = float(
            np.corrcoef(np.log(universe.sigma_dL), np.log(universe.d_L_true))[0, 1]
        )

    return {
        "seed": int(seed),
        "cell": vcfg.cell,
        "h_true": float(vcfg.h_true),
        "balls": vcfg.balls,
        "sigma_mode": vcfg.sigma_mode,
        "f_incl": 1.0,
        "n_events": n,
        "n_events_run": n,
        "n_horizon_dropped": int(context.n_horizon_dropped),
        "z_median": float(np.median(universe.z_true)),
        "M_source_median": float(np.median(universe.M_true)),
        "frac_below_kink": float(np.mean(universe.M_true < 1.0e5)),
        "K_mean": float(np.mean(ball.K)),
        "K_sum": int(np.sum(ball.K)),
        "n_impostors_total": int(ball.n_impostors_total),
        "n_degenerate_windows": int(ball.n_degenerate_windows),
        "texture_corr": texture_corr,
        "sigma_z_mean_pairs": float(np.mean(sigma_pairs)) if sigma_pairs.size else 0.0,
        "sigma_z_median_pairs": float(np.median(sigma_pairs)) if sigma_pairs.size else 0.0,
        "frac_pairs_sigma_lt_5e-3": float(np.mean(sigma_pairs < 5.0e-3))
        if sigma_pairs.size
        else 0.0,
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


# ── Aggregation: DS-VT1 … DS-VT4 + per-channel classification ────────────────


def classify_channel(
    ch: dict[str, Any],
    n_seeds: int,
    sigma_bar_pairs: float,
    *,
    degenerate_exempt: bool,
) -> dict[str, Any]:
    """Mechanical per-channel cell classification (prereg §7, bands locked).

    ``COLLAPSE-REPRODUCED`` = C90 ≤ collapse band AND both rails within the
    DS-VT4 collapse band AND bias ≥ +0.030 AND ``R_dose`` in [0.75, 1.25].
    ``CALIBRATED`` = DS-VT1 all three levels inside 3σ AND DS-VT2 PASS AND
    |bias| ≤ 0.010 AND rails within the collapse band. ``OTHER`` = anything
    else. Any rail ≥ 0.90 additionally flags ``RAIL-EMERGENT`` (forces MIXED
    at readout). Collapse bands by registered N: 0.02 (N ≥ 400) / 0.04
    (N ≥ 200) / 0.08 (N = 100 fallback). Band comparisons only — the branch
    call is never made here.

    Args:
        ch: A :func:`calibration_gate._channel_aggregate` block.
        n_seeds: Seeds in the cell.
        sigma_bar_pairs: The cell's realized mean per-candidate σ_z (DS-VT3;
            0.035 exactly for flat-dose arms).
        degenerate_exempt: T-0 exemption — DS-1/DS-2 carry no meaning at
            σ_z = 0 (v2 D3 carried, VT-D8).

    Returns:
        The classification block.
    """
    thr = 0.02 if n_seeds >= 400 else (0.04 if n_seeds >= 200 else 0.08)
    c90 = float(ch["ds1_coverage"]["hpd90"]["value"])
    r_low = float(ch["ds4_rails"]["railed_low_frac"])
    r_high = float(ch["ds4_rails"]["railed_high_frac"])
    bias = float(ch["ds3_map_bias"]["bias"])
    r_dose = bias / sigma_bar_pairs if sigma_bar_pairs > 0.0 else float("nan")
    rails_in_band = r_low <= thr and r_high <= thr
    rail_emergent = max(r_low, r_high) >= _RAIL_EMERGENT_THRESHOLD
    ds1_all3 = all(ch["ds1_coverage"][k]["inside_3sigma"] for k in ("hpd50", "hpd68", "hpd90"))
    ds2_pass = ch["ds2_ks"]["status"] == "PASS"
    r_dose_in_band = R_DOSE_BAND[0] <= r_dose <= R_DOSE_BAND[1] if np.isfinite(r_dose) else False

    if degenerate_exempt:
        label = "DS1-DS2-EXEMPT"  # T-0: scored on V-T1 + DS-3/DS-4 only (VT-D8)
    elif c90 <= thr and rails_in_band and bias >= cg._DS3_DEFECT and r_dose_in_band:
        label = "COLLAPSE-REPRODUCED"
    elif ds1_all3 and ds2_pass and abs(bias) <= cg._DS3_IN_BAND and rails_in_band:
        label = "CALIBRATED"
    else:
        label = "OTHER"

    return {
        "label": label,
        "n_seeds": n_seeds,
        "registered_n": n_seeds in (400, 200, 100),
        "collapse_band": thr,
        "c90": c90,
        "r_low": r_low,
        "r_high": r_high,
        "rails_in_collapse_band": rails_in_band,
        "rail_emergent": bool(rail_emergent),
        "bias": bias,
        "sigma_bar_pairs": sigma_bar_pairs,
        "r_dose": r_dose,
        "r_dose_band": list(R_DOSE_BAND),
        "r_dose_in_band": bool(r_dose_in_band),
        "ds1_all_inside_3sigma": bool(ds1_all3),
        "ds2_pass": bool(ds2_pass),
        "degenerate_pit_exempt": bool(degenerate_exempt),
    }


def _vt1_anchor(ch: dict[str, Any]) -> dict[str, Any]:
    """V-T1 T-0 anchor read of one channel (prereg §10, edges locked).

    Args:
        ch: A channel-aggregate block.

    Returns:
        ``{"bias", "r_low", "r_high", "status"}`` with status PASS /
        ANCHOR-MARGINAL / HARD-TRIGGER.
    """
    bias = float(ch["ds3_map_bias"]["bias"])
    r_low = float(ch["ds4_rails"]["railed_low_frac"])
    r_high = float(ch["ds4_rails"]["railed_high_frac"])
    rail_bad = r_low > _VT1_RAIL_MAX or r_high > _VT1_RAIL_MAX
    if abs(bias) >= _VT1_BIAS_HARD or rail_bad:
        status = "HARD-TRIGGER"  # => VENUE-CONFOUNDED + first-class new raw finding
    elif abs(bias) > _VT1_BIAS_PASS:
        status = "ANCHOR-MARGINAL"
    else:
        status = "PASS"
    return {"bias": bias, "r_low": r_low, "r_high": r_high, "status": status}


def aggregate_venue(records: list[dict[str, Any]], vcfg: VenueConfig) -> dict[str, Any]:
    """Aggregate one cell×truth's per-seed records into the DS-VT readout.

    Args:
        records: Per-seed records.
        vcfg: The cell configuration.

    Returns:
        The aggregate block (both channels always reported together; the
        registered headline verdict channel is 1D, VT-D6).
    """
    n = len(records)
    ch1 = cg._channel_aggregate(records, "1d", vcfg.h_true)
    ch2 = cg._channel_aggregate(records, "2d", vcfg.h_true)
    sigma_bar = float(np.mean([r["sigma_z_mean_pairs"] for r in records]))
    degenerate_exempt = vcfg.sigma_mode == "zero"
    nonfinite = float(
        np.mean(
            [
                (not np.all(np.isfinite(r["ln_post_1d"])))
                or (not np.all(np.isfinite(r["ln_post_2d"])))
                for r in records
            ]
        )
    )
    doc: dict[str, Any] = {
        "cell": vcfg.cell,
        "prereg_cell": CELL_SPECS[vcfg.cell].prereg_cell if vcfg.cell in CELL_SPECS else vcfg.cell,
        "h_true": vcfg.h_true,
        "n_seeds": n,
        "headline_channel": "1d",  # VT-D6
        "ds1_ds2_degenerate_pit_exempt": degenerate_exempt,
        "channel_1d": ch1,
        "channel_2d": ch2,
        "classification_1d": classify_channel(
            ch1, n, sigma_bar, degenerate_exempt=degenerate_exempt
        ),
        "classification_2d": classify_channel(
            ch2, n, sigma_bar, degenerate_exempt=degenerate_exempt
        ),
        "dose": {
            "sigma_bar_pairs": sigma_bar,
            "sigma_z_median_pairs_mean": float(
                np.mean([r["sigma_z_median_pairs"] for r in records])
            ),
            "frac_pairs_sigma_lt_5e-3_mean": float(
                np.mean([r["frac_pairs_sigma_lt_5e-3"] for r in records])
            ),
        },
        "ball": {
            "K_mean": float(np.mean([r["K_mean"] for r in records])),
            "K_sum_mean": float(np.mean([r["K_sum"] for r in records])),
            "n_impostors_total": int(np.sum([r["n_impostors_total"] for r in records])),
            "n_degenerate_windows": int(np.sum([r["n_degenerate_windows"] for r in records])),
            "balls": vcfg.balls,
            "sigma_mode": vcfg.sigma_mode,
        },
        "events": {
            "n_events_run": int(records[0]["n_events_run"]) if records else 0,
            "n_horizon_dropped": int(records[0]["n_horizon_dropped"]) if records else 0,
        },
        "nonfinite_ln_post_frac": nonfinite,
        "abort_b_triggered": bool(nonfinite > cg._NONFINITE_ABORT_FRACTION),
    }
    if vcfg.cell == "T0" or (vcfg.sigma_mode == "zero" and vcfg.balls == "real_k"):
        doc["vt1_anchor"] = {
            "channel_1d": _vt1_anchor(ch1),
            "channel_2d": _vt1_anchor(ch2),
            "note": (
                "V-T1: HARD-TRIGGER => VENUE-CONFOUNDED and simultaneously a "
                "first-class NEW raw finding (prereg §10); branch presented to "
                "the author, never self-adjudicated"
            ),
        }
    return doc


# ── Sweep driver ─────────────────────────────────────────────────────────────


def run_cell_venue(
    vcfg: VenueConfig,
    seeds: list[int],
    workers: int,
    *,
    allow_dirty: bool = False,
    grain: str = "seed",
) -> dict[str, Any]:
    """Run one cell×truth sweep and assemble the results document.

    The V-T4 clean rule is enforced by the imported gate function (verbatim);
    the context is built parent-side before forking (divergence 1).

    Args:
        vcfg: The cell configuration.
        seeds: Seeds to run.
        workers: Worker processes (``<= 1`` runs in-process). At
            ``grain="seed"`` this is the pool-over-seeds size; at
            ``grain="h"`` it is the per-seed fork-pool size over the h grid.
        allow_dirty: Permit a dirty IMPORT PATH (smoke/validate only —
            :func:`main` rejects it for registered cell runs; recorded).
        grain: ``"seed"`` (registered default — the campaign's
            pool-over-seeds mode, untouched) or ``"h"`` (divergence 11:
            seeds run serially in the parent, each seed's estimator loop
            forked over the 41 h-points; per-seed records byte-identical).

    Returns:
        The full results dict (written to JSON by :func:`main`).

    Raises:
        ValueError: On an unknown ``grain``.
    """
    if grain not in ("seed", "h"):
        raise ValueError(f"unknown grain '{grain}' (expected 'seed' or 'h')")
    commit, dirt = cg._enforce_clean_import_path(allow_dirty)
    global _VCTX
    if _VCTX is None or _VCTX.vcfg != vcfg:
        _VCTX = build_venue_context(vcfg)
    context = _VCTX
    t0 = time.monotonic()
    if grain == "h":
        records = [run_seed_venue_hgrain(s, context, workers) for s in seeds]
    elif workers > 1:
        ctx_mp = mp.get_context("fork")
        with ctx_mp.Pool(
            processes=workers, initializer=_venue_worker_init, initargs=(vcfg,)
        ) as pool:
            records = pool.map(run_seed_venue, seeds, chunksize=1)
    else:
        records = [run_seed_venue(s) for s in seeds]
    wall = time.monotonic() - t0
    agg = aggregate_venue(records, vcfg)
    return {
        "instrument": "venue_transfer",
        "preregistration": PREREG_PATH,
        "parent_instruments": {
            "calibration_gate": {
                "code": PARENT_CALGATE_CODE_COMMIT,
                "run": PARENT_CALGATE_RUN_COMMIT,
            },
            "closed_loop_gfrac": PARENT_CLOSED_LOOP_COMMIT,
        },
        "git_commit": commit,
        "git_dirty": bool(dirt["import_path"] or dirt["other"]),
        "import_path_clean": not dirt["import_path"],
        "dirt_inventory": dirt,
        "allow_dirty": allow_dirty,
        "config": asdict(vcfg),
        "pin_integrity": context.pin_integrity,
        "seeds": [int(s) for s in seeds],
        "workers": workers,
        "grain": grain,
        "wall_time_s": wall,
        "wall_time_per_seed_s": wall / max(len(seeds), 1),
        "aggregate": agg,
        "per_seed": records,
    }


# ── V-T5 no-drift anchor (v2-compat bit-reproduction) ────────────────────────


def _gate_style_record(
    gctx: cg.GateContext,
    seed: int,
    universe: cl.SyntheticUniverse,
    ball: cg.HostBall,
    ln1: npt.NDArray[np.float64],
    ln2: npt.NDArray[np.float64],
    slope: npt.NDArray[np.float64],
) -> dict[str, Any]:
    """Assemble the gate's §6 ball-cell record from precomputed posteriors.

    Field-for-field the ball branch of
    :func:`calibration_gate.run_seed_gate` (deterministic given the inputs) —
    used by the V-T5 shared-field comparison.

    Args:
        gctx: The gate context.
        seed: The seed.
        universe: The drawn universe.
        ball: The drawn ball.
        ln1: 1D unnormalised log posterior.
        ln2: 2D unnormalised log posterior.
        slope: ``sum_dlog_gfrac_dh`` scalar array.

    Returns:
        The gate-format per-seed record.
    """
    gcfg = gctx.gate_config
    h_arr = np.asarray(gcfg.h_grid, dtype=np.float64)
    r1 = cl.posterior_readout(h_arr, ln1)
    r2 = cl.posterior_readout(h_arr, ln2)
    pp1 = cg.pp_readout(h_arr, ln1, gcfg.h_true)
    pp2 = cg.pp_readout(h_arr, ln2, gcfg.h_true)
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
        "K_mean": float(np.mean(ball.K)),
        "n_impostors_total": int(ball.n_impostors_total),
        "n_degenerate_windows": int(ball.n_degenerate_windows),
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


def run_v_t5_compat_check(
    committed_json: str = B2_H0P730_RESULTS_JSON,
    seeds: tuple[int, ...] = V_T5_SEEDS,
) -> dict[str, Any]:
    """V-T5 no-drift anchor: bit-reproduce committed v2 B2(0.730) records.

    v2-compat mode (prereg §10): universe via imported
    :func:`calibration_gate.draw_universe_gate`, ball via imported
    :func:`calibration_gate.draw_ball` (N = 1500, λ = 4, σ_z = 0.035,
    canonical grid), estimator = THIS module's vector-σ core with a constant
    σ_z vector. Every shared per-seed field must equal the committed
    ``B2_h0p730_results.json`` record bit-identically — certifying the
    vector-σ estimator core equals the committed gate math. Failure ⇒ STOP.

    Args:
        committed_json: The committed v2 B2(0.730) results document.
        seeds: The registered v2 seeds to reproduce (20286808–20286810).

    Returns:
        The V-T5 block: per-seed mismatch lists and an overall ``pass``.
    """
    with open(committed_json) as fh:
        committed = json.load(fh)
    cfg_dict = dict(committed["config"])
    cfg_dict["h_grid"] = tuple(cfg_dict["h_grid"])
    gcfg = cg.GateConfig(**cfg_dict)
    gctx = cg.build_gate_context(gcfg)
    committed_by_seed = {r["seed"]: r for r in committed["per_seed"]}

    per_seed: list[dict[str, Any]] = []
    all_ok = True
    for seed in seeds:
        rng = np.random.default_rng(seed)
        universe = cg.draw_universe_gate(gctx, rng)
        ball = cg.draw_ball(gctx, universe, rng)
        sigma_pairs = np.full(ball.z_obs.size, gcfg.sigma_z, dtype=np.float64)
        # Exact gate-shape mode (chunk_pairs=0): bit-identity is the claim.
        ln1, ln2, slope = log_channel_posteriors_ball_sigma_vector(
            gctx, universe, ball, sigma_pairs, chunk_pairs=0
        )
        mine = _gate_style_record(gctx, seed, universe, ball, ln1, ln2, slope)
        ref = committed_by_seed.get(seed)
        if ref is None:
            per_seed.append({"seed": seed, "error": "seed not in committed record"})
            all_ok = False
            continue
        shared = sorted(set(mine) & set(ref))
        mismatches = [k for k in shared if mine[k] != ref[k]]
        ok = not mismatches
        all_ok = all_ok and ok
        per_seed.append(
            {
                "seed": seed,
                "n_shared_fields": len(shared),
                "mismatched_fields": mismatches,
                "pass": ok,
            }
        )
    return {
        "pass": bool(all_ok),
        "committed_json": committed_json,
        "seeds": list(seeds),
        "per_seed": per_seed,
        "note": (
            "V-T5 no-drift anchor: vector-sigma core in v2-compat mode must "
            "bit-reproduce the committed gate records (shared fields); "
            "failure => STOP (prereg §10)"
        ),
    }


# ── Validation mode (V-T2, V-T3, V-T5; seed-plan assertions) ─────────────────


def run_validate(*, n_events_cap: int = 40, skip_v_t5: bool = False) -> dict[str, Any]:
    """Run the §10 validity checks executable without a full sweep.

    V-T2 (determinism spot check on the maximal code path, dev-capped events),
    V-T3 (pin integrity against the registered md5/census/sampler pins),
    V-T5 (v2-compat bit-reproduction of committed gate records), plus the
    VT-D7 seed-plan disjointness assertions. Checks needing absent files are
    reported as skipped, not failed.

    Args:
        n_events_cap: Event cap for the V-T2 spot check (dev-only fidelity).
        skip_v_t5: Skip the (minutes-long) V-T5 compat computation.

    Returns:
        The validation document.
    """
    out: dict[str, Any] = {}

    # Seed-plan disjointness (VT-D7) — pure arithmetic, always runs.
    used: set[int] = set()
    per_cell_ok = True
    for spec in CELL_SPECS.values():
        for t in spec.truths:
            block = set(venue_cell_seeds(spec, t, 0, None))
            per_cell_ok = per_cell_ok and not (used & block)
            used.update(block)
    envelopes_ok = all(
        not any(
            VT_BASE_SEED + lo <= s <= VT_BASE_SEED + hi
            for lo, hi in (V1_SEED_OFFSET_ENVELOPE, V2_SEED_OFFSET_ENVELOPE)
        )
        for s in used
    )
    in_v3 = all(
        VT_BASE_SEED + V3_SEED_OFFSET_ENVELOPE[0] <= s <= VT_BASE_SEED + V3_SEED_OFFSET_ENVELOPE[1]
        for s in used
    )
    out["seed_plan"] = {
        "pass": bool(per_cell_ok and envelopes_ok and in_v3),
        "n_registered_seeds": len(used),
        "disjoint_within_v3": per_cell_ok,
        "disjoint_from_v1_v2": envelopes_ok,
        "inside_v3_envelope": in_v3,
    }

    # V-T3 — pin integrity.
    files_present = all(
        os.path.isfile(p) for p in (CRB_CSV_PATH, FROZENG_EMIT_JSON, PRUNED_CATALOGUE_CSV)
    )
    if files_present:
        vcfg_pins = VenueConfig(cell="Tc", h_true=0.730, balls="real_k", sigma_mode="glade")
        out["v_t3"] = check_pin_integrity(vcfg_pins)
    else:
        out["v_t3"] = {"pass": None, "note": "pinned input files not present in this checkout"}

    # V-T2 — determinism: same seed => bit-identical record (maximal path:
    # real-K balls + glade sampler), dev-capped events.
    pool_present = os.path.isdir(cl.DEFAULT_INJECTION_DIR)
    if files_present and pool_present:
        vcfg_v2 = VenueConfig(
            cell="Tc",
            h_true=0.730,
            balls="real_k",
            sigma_mode="glade",
            n_events_cap=n_events_cap,
        )
        vctx = build_venue_context(vcfg_v2, check_pins=False)
        rec_a = run_seed_venue(VT_BASE_SEED + 43000, vctx)
        rec_b = run_seed_venue(VT_BASE_SEED + 43000, vctx)
        out["v_t2"] = {
            "pass": json.dumps(rec_a, sort_keys=True) == json.dumps(rec_b, sort_keys=True),
            "seed": VT_BASE_SEED + 43000,
            "n_events_cap": n_events_cap,
        }
    else:
        out["v_t2"] = {"pass": None, "note": "injection pool / pinned files not present"}

    # V-T5 — v2-compat bit-reproduction.
    if skip_v_t5:
        out["v_t5"] = {"pass": None, "note": "skipped (--skip-v-t5)"}
    elif os.path.isfile(B2_H0P730_RESULTS_JSON) and pool_present and os.path.isfile(CRB_CSV_PATH):
        out["v_t5"] = run_v_t5_compat_check()
    else:
        out["v_t5"] = {"pass": None, "note": "committed B2 JSON / injection pool not present"}

    out["v_t4"] = {
        "note": (
            "clean rule enforced by the imported gate functions "
            "(_enforce_clean_import_path / _classify_porcelain, V-T4 verbatim); "
            "unit-tested in darksiren_emri_test/validation/test_venue_transfer.py"
        )
    }
    return out


# ── CLI ──────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    p = argparse.ArgumentParser(
        description=(
            "Venue-transfer instrument (prereg "
            "results/venue_transfer_20260811/PREREGISTRATION_VENUE_TRANSFER.md; "
            "the author-named decisive measurement of the v2 clause-(b) DEFECT)"
        )
    )
    p.add_argument(
        "--cell",
        choices=("T0", "Ta", "Tb", "Tc", "W1", "O2"),
        help="prereg §5 cell to run (W1/O2 are reserved, NOT built)",
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
    p.add_argument(
        "--n-events-cap",
        type=int,
        default=None,
        help="truncate the pinned event list (smoke/validate only; divergence 9)",
    )
    p.add_argument(
        "--chunk-pairs",
        type=int,
        default=DEFAULT_CHUNK_PAIRS,
        help="event-aligned pair-chunk target (memory only; never changes a statistic)",
    )
    p.add_argument("--out", type=str, default=None, help="output JSON path")
    p.add_argument("--workers", type=int, default=max(mp.cpu_count() - 2, 1))
    p.add_argument(
        "--grain",
        choices=("seed", "h"),
        default="seed",
        help=(
            "parallelism grain (divergence 11): 'seed' = the registered "
            "campaign's pool-over-seeds mode (default, untouched); 'h' = "
            "opt-in intra-seed fork pool over the 41 h-points — per-seed "
            "records are byte-identical to seed grain for any worker count"
        ),
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="3 seeds, 1 worker, + V-T2 spot-check (full pinned N unless --n-events-cap)",
    )
    p.add_argument("--validate", action="store_true", help="run V-T2/V-T3/V-T5 + seed-plan checks")
    p.add_argument(
        "--skip-v-t5",
        action="store_true",
        help="with --validate: skip the minutes-long V-T5 compat computation",
    )
    p.add_argument(
        "--allow-dirty",
        action="store_true",
        help=(
            "permit a dirty IMPORT PATH — accepted only with --smoke or "
            "--validate, never on a registered cell run (V-T4 clean rule, "
            "carried verbatim from v2 D5; recorded in the JSON)"
        ),
    )
    p.add_argument("--log-level", type=str, default="INFO")
    return p


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Argument vector (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code.
    """
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    # V-T4 clause 2 (verbatim from v2 D5): --allow-dirty only with
    # --smoke/--validate; there is no other escape.
    if args.allow_dirty and not (args.smoke or args.validate):
        raise SystemExit(
            "STOP: --allow-dirty is accepted only with --smoke or --validate "
            "(V-T4 clean rule — registered cells must run on a clean import path)."
        )
    # Divergence 9: the pinned-event cap is smoke/validate-only.
    if args.n_events_cap is not None and not (args.smoke or args.validate):
        raise SystemExit(
            "STOP: --n-events-cap is a smoke/validate-only dev override — "
            "registered cells run the full pinned event set (VT-D1/VT-D5)."
        )

    if args.validate:
        cg._enforce_clean_import_path(args.allow_dirty)
        doc = run_validate(
            n_events_cap=args.n_events_cap if args.n_events_cap is not None else 40,
            skip_v_t5=args.skip_v_t5,
        )
        out = args.out or os.path.join(DEFAULT_OUT_DIR, "validate_results.json")
        cg._guard_out_path(out)
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        with open(out, "w") as fh:
            json.dump(doc, fh, indent=2)
        _LOGGER.info(
            "validate: %s",
            {k: (v.get("pass") if isinstance(v, dict) else v) for k, v in doc.items()},
        )
        return 0

    if args.cell is None:
        raise SystemExit("one of --cell or --validate is required")
    if args.cell in ("W1", "O2"):
        reason = {
            "W1": "per-galaxy rate-weights arm — NOT-EVALUABLE per prereg §9 item 3",
            "O2": "volume_deconv kernel arm — NOT-EVALUABLE per prereg §9 item 2",
        }[args.cell]
        raise SystemExit(
            f"Cell {args.cell} is reserved but NOT built: {reason}; its seed block "
            "is never run post-hoc (VT-D7) — buildable only by author order."
        )

    spec = CELL_SPECS[args.cell]
    truth = args.truth if args.truth is not None else spec.truths[0]
    if not any(abs(truth - t) < 1e-12 for t in spec.truths):
        raise SystemExit(f"truth {truth} not in cell {spec.name} registered set {spec.truths}")
    truth = next(t for t in spec.truths if abs(truth - t) < 1e-12)

    vcfg = VenueConfig(
        cell=spec.name,
        h_true=truth,
        balls=spec.balls,
        sigma_mode=spec.sigma_mode,
        n_events_cap=args.n_events_cap,
        chunk_pairs=args.chunk_pairs,
    )

    if args.seeds is not None:
        seeds = [int(s) for s in args.seeds.split(",")]
    elif args.seed_range is not None:
        start_s, count_s = args.seed_range.split(":")
        seeds = venue_cell_seeds(spec, truth, int(start_s), int(count_s))
    else:
        seeds = venue_cell_seeds(spec, truth, 0, None)
    if args.smoke and args.n_seeds is None:
        seeds = seeds[:3]
    elif args.n_seeds is not None:
        seeds = seeds[: args.n_seeds]

    workers = 1 if args.smoke else args.workers
    doc = run_cell_venue(vcfg, seeds, workers, allow_dirty=args.allow_dirty, grain=args.grain)
    doc["smoke"] = args.smoke

    if args.smoke:
        # V-T2 spot check: re-run the first seed, must be bit-identical.
        rec_again = run_seed_venue(seeds[0])
        first = next(r for r in doc["per_seed"] if r["seed"] == seeds[0])
        doc["v_t2_smoke"] = {
            "pass": json.dumps(rec_again, sort_keys=True) == json.dumps(first, sort_keys=True),
            "seed": seeds[0],
        }

    if args.out is not None:
        out = args.out
    else:
        stem = f"{spec.name}_h{truth:.3f}_results".replace("0.", "0p")
        if args.seed_range is not None:
            start_s, count_s = args.seed_range.split(":")
            stem += f"_seeds{int(start_s)}_{int(count_s)}"  # divergence 5
        if args.smoke:
            stem += "_smoke"  # never collide with a registered output name
        out = os.path.join(DEFAULT_OUT_DIR, stem + ".json")
    cg._guard_out_path(out)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as fh:
        json.dump(doc, fh, indent=2)

    agg = doc["aggregate"]
    _LOGGER.info(
        "cell %s h_true=%.3f n=%d | 1D: %s C90=%.3f bias=%+.4f R_dose=%.3f rails=%.3f/%.3f | "
        "2D: %s bias=%+.4f | sigma_bar=%.5f K_sum=%.0f | %.1f s/seed",
        vcfg.cell,
        vcfg.h_true,
        agg["n_seeds"],
        agg["classification_1d"]["label"],
        agg["classification_1d"]["c90"],
        agg["classification_1d"]["bias"],
        agg["classification_1d"]["r_dose"],
        agg["classification_1d"]["r_low"],
        agg["classification_1d"]["r_high"],
        agg["classification_2d"]["label"],
        agg["classification_2d"]["bias"],
        agg["dose"]["sigma_bar_pairs"],
        agg["ball"]["K_sum_mean"],
        doc["wall_time_per_seed_s"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
