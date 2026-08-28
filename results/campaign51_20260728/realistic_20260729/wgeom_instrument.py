#!/usr/bin/env python3
"""Registered instrument for [MKER] window-GEOMETRY (stage 2, correctness-class).

Spec: ``PREREGISTRATION_MKER_WGEOM_20260828.md`` (this directory). This script
is the ``⟨SUBMIT⟩`` instrument named in that document's §2 "instruments and
inputs" table. Read the prereg BEFORE reading this file — every function below
cites the prereg section it implements.

**Status of this file: BUILT, NOT RUN.** Per the author's build instruction,
this script performs no full run of its own accord; ``--full`` writes the
registered output but must be invoked deliberately (see module-level
docstring epilogue / ``--help``). A ``--smoke`` mode exists for code-path
verification on a tiny head-slice of the catalogue and writes nothing to
``wgeom_work/``.

Section -> function map
------------------------
* §2 dataset pin (G2), STOP-gated                  -> ``verify_catalogue_pin``
* §2 R&V15 mapping + prune (shared machinery)       -> ``map_stellar_to_bh_mass``,
                                                        ``mass_redshift_prune_mask``,
                                                        ``load_pruned_catalogue``
* P1 closed-form geometry function A(x)             -> ``geometry_function_A``,
                                                        ``check_p1``
* P2 epsilon-semantics table (the deliverable)       -> ``light_side_cut``,
                                                        ``heavy_side_cut``,
                                                        ``eps_lin``, ``eps_log``,
                                                        ``compute_p2``
* G4 closed-form anchor (authoring-time P2 table)   -> ``check_g4``
* Fleet loading (bc_9001XX arms, cone-exact)        -> ``iter_fleet_events``,
                                                        ``load_fleet_event``
* G1 banked-census reproduction (= P3a)             -> ``compute_p3``  (g1 sub-result)
* G3 exhibit reproduction (= P4 linear leg)          -> ``compute_p4``  (g3 sub-result)
* P3 discordance census                              -> ``compute_p3``
* P4 exhibit regression                              -> ``compute_p4``
* P5 chair re-derivation of -14.5% moment            -> ``compute_p5``
* Verdict map (§5)                                   -> ``compute_verdict``
* Output (JSON + human-readable table)               -> ``write_outputs``

Ambiguities the author must be aware of (also restated in the launch report):

  A1. "Imports nothing from the production estimator beyond the R&V15
      constants" (prereg §2, instrument-script row) is read as: (a) the R&V15
      constants (alpha, beta, d_alpha, d_beta, sigma_int) ARE imported from
      ``darksiren_emri.galaxy_catalogue.handler`` rather than re-typed, to
      avoid silent drift from the source of truth; (b) M_min/M_max/z_max/k and
      the LamCDM h/Omega_m bounds are HARD-CODED with file:line citations
      (matching the precedent set by the banked chair census scripts
      referenced in CLAIM_WGEO_20260827.md §3, which did the same); (c) the
      redshift-outer-bounds cosmology inversion (``get_redshift_outer_bounds``,
      which itself calls the gr-qc-validated ``dist_to_redshift`` fsolve
      root-finder) IS imported from ``physical_relations.py`` -- reimplementing
      ΛCDM d_L(z) inversion independently would itself be a new physics
      artifact requiring its own physics-change gate, and is out of scope for
      a geometry-of-the-mass-window measurement. This import is the one
      instrument dependency that is NOT the mass-window code under test.
  A2. The prereg's "cone-exact fleet census (4800 event rows)" is read as the
      ``bc_9001XX_work`` arms ONLY (24 arms x 200 events = 4800), matching the
      glob pattern (`bc_9001??_work`) in the banked reference script that
      produced the CLAIM_WGEO_20260827.md sec 3.9 numbers
      (`wgeo_fleet.py`, preserved verbatim in this session's scratch dir).
      The `bt_9001XX_work` arms (also 24 x 200 = 4800) are a separate fleet
      half and are NOT part of this census (P3/G1), even though the single
      banked EXHIBIT (P4/G3, seed 900121 event 20) lives on the `bt_` arm per
      CLAIM_P3_MKER_20260826.md:659 -- the two objects (aggregate census vs.
      one pinned exhibit) draw from different arms in the banked record
      itself, and this instrument reproduces that split faithfully rather
      than "fixing" it.
  A3. Instead of re-deriving each event's sky-cone + redshift-filter candidate
      set from scratch (which would require importing/reimplementing the
      BallTree sky-ellipse query, ``handler.py:619-646``), the instrument
      reads the ALREADY-COMPUTED candidate-index lists straight out of the
      banked production posteriors JSON
      (``posteriors_with_bh_mass/h_0_73.json``: ``galaxy_likelihoods`` union
      ``additional_galaxies_without_bh_mass``, disjoint by construction --
      ``bayesian_statistics.py:4859-4868``). This IS the cone-exact candidate
      set (n_all) for each event, "for free" and independent of any
      geometry-code reimplementation on the sky/redshift side; only the MASS
      window (the object under measurement) is independently recomputed, in
      both geometries, from the R&V15-mapped catalogue mass/error looked up
      by ``catalog_index`` position. This is read as satisfying "cone-exact"
      more strongly than a from-scratch BallTree re-query would (zero risk of
      a sky-geometry transcription bug contaminating a mass-geometry result).
  A4. P3c's "share of no-BH likelihood weight the readmitted rows carry" has
      no banked definition of which scalar in the per-galaxy
      ``additional_galaxies_without_bh_mass`` result vector is "the weight"
      (the vector's schema is internal to
      ``bayesian_statistics._starmap_host_batches`` and not decoded by this
      script, which does not import that module). This sub-metric is reported
      as NOT COMPUTED, with the CV/redshift/count composition computed in
      full; P3c is explicitly REPORTED-ONLY / not band-graded (prereg §3,
      §8 item 6), so this does not block any gate or verdict.
  A5. P1/P2 exactness tolerance (prereg §4: "<= 1e-9 relative vs the
      instrument's recomputation") is read literally for G4 (the instrument's
      ``eps_lin``/``eps_log`` formulas applied to the six BANKED CV values in
      the prereg §3 P2 table must reproduce the six banked epsilon values to
      1e-9 relative -- that is the actual cross-check). The P1 spot-value
      check against the six 6-decimal-quoted A(x) values in prereg §3 uses an
      abs tolerance of 5e-7 (half a unit in the last quoted decimal), since
      the banked figures are decimal-truncated and a 1e-9 comparison against
      a 6-decimal string is not meaningful.

References for the underlying production formulas (all re-derived here, not
imported, except where A1 states otherwise):
  * R&V15 mass mapping: darksiren_emri/galaxy_catalogue/handler.py:33-44,1368-1382
  * mass/redshift prune: darksiren_emri/galaxy_catalogue/handler.py:215-251
  * mass_filter_mask (linear, k=1.5, "symmetric"): handler.py:654-673
  * candidate-set split (galaxy_likelihoods / additional_...):
    darksiren_emri/bayesian_inference/bayesian_statistics.py:4859-4946
  * z_min/z_max: darksiren_emri/physical_relations.py:546-567 (imported, A1c)
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.stats import norm

# --------------------------------------------------------------------------
# A1(a): R&V15 constants imported from the source of truth, nothing else.
# --------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))
from darksiren_emri.galaxy_catalogue.handler import (  # noqa: E402
    alpha as RV15_ALPHA,
    beta as RV15_BETA,
    d_alpha as RV15_D_ALPHA,
    d_beta as RV15_D_BETA,
    sigma_int as RV15_SIGMA_INT,
)

# A1(c): the ΛCDM d_L->z inversion is imported, not reimplemented (see module
# docstring A1). This is the single "estimator" import beyond the R&V15
# constants.
from darksiren_emri.physical_relations import get_redshift_outer_bounds  # noqa: E402

# --------------------------------------------------------------------------
# Frozen inputs (prereg §2)
# --------------------------------------------------------------------------
CATALOGUE_PATH = (
    _REPO_ROOT / "darksiren_emri/galaxy_catalogue/reduced_galaxy_catalogue.csv"
)
CATALOGUE_MD5 = "c52c13b5cab61f6b3f04bbe202550969"  # REDUCED_CATALOGUE_MD5, correspondence_1d.py:311
FLEET_BASE = (
    _REPO_ROOT
    / "results/campaign51_20260728/realistic_20260729/p3_2d_fleet_20260825"
)
FLEET_ARM_GLOB = "bc_9001??_work"  # A2: bc_ arms only, matches banked wgeo_fleet.py
FLEET_EXPECTED_ARMS = 24
FLEET_EXPECTED_EVENTS_PER_ARM = 200
FLEET_EXPECTED_TOTAL = 4800

OUT_DIR = (
    _REPO_ROOT / "results/campaign51_20260728/realistic_20260729/wgeom_work"
)

# On-disk reduced-catalogue column order (handler.py:_reduced_catalog_column_names)
CATALOGUE_COLUMN_NAMES = [
    "RIGHT_ASCENSION",
    "DECLINATION",
    "APPARENT_B_MAG",
    "REDSHIFT",
    "REDSHIFT_MEASUREMENT_ERROR",
    "STELLAR_MASS",
    "STELLAR_MASS_ABSOULTE_ERROR",
    "REDSHIFT_FLAG",
]
CATALOGUE_USECOLS = [
    "REDSHIFT",
    "REDSHIFT_MEASUREMENT_ERROR",
    "STELLAR_MASS",
    "STELLAR_MASS_ABSOULTE_ERROR",
]
CATALOGUE_CHUNKSIZE = 2_000_000

# Production prune bounds -- hardcoded + cited, not imported (A1(b)):
M_MIN = 1e4  # constants.py: M_SOURCE_FRAME_MIN
M_MAX = 1e7  # constants.py: M_SOURCE_FRAME_MAX
Z_MAX_PRUNE = 1.5  # handler.py:30, `Z_draw`

# Production mass-filter geometry, adopted "symmetric" mode (rows #198-#202):
K_SIGMA = 1.5  # bayesian_statistics.py:4691, single call site

# LamCDMScenario bounds (cosmological_model.py:377-399), hardcoded + cited (A1(b)):
H_MIN = 0.6
H_MAX = 0.86
OMEGA_M_MIN = 0.04
OMEGA_M_MAX = 0.5
W_0 = -1.0
W_A = 0.0
REDSHIFT_UPPER_LIMIT = 1.5  # HOST_DRAW_Z_MAX, constants.py; caps z_max at the p_D call site

# Pinned index cross-check targets (CLAIM_WGEO_20260827.md §3.1, chair-verified
# to full float precision against the pruned/reset_index positional frame):
PIN_TARGETS: dict[int, tuple[float, float]] = {
    6791138: (709540.708756878, 894866.2758100418),
    6791158: (709540.708756878, 1570331.1654161075),
    6791151: (223872.11385683485, 291758.99489010876),
}

# Banked P2 table (prereg §3, authoring-time derivation) -- G4 anchor.
# columns: CV, eps_lin_light, eps_lin_heavy, eps_lin_total, eps_log_total
BANKED_P2_TABLE: list[tuple[str, float, float, float, float, float]] = [
    ("min", 0.5930, 0.000102, 0.141627, 0.141729, 0.133614),
    ("p10", 0.7846, 0.0, 0.160730, 0.160730, 0.133614),
    ("median", 0.8614, 0.0, 0.167791, 0.167791, 0.133614),
    ("p75", 0.9401, 0.0, 0.174704, 0.174704, 0.133614),
    ("p90", 1.2137, 0.0, 0.196454, 0.196454, 0.133614),
    # Exhibit CV uses the FULL-precision banked sigma_ln (CLAIM_P3_MKER_20260826.md
    # §R2.2: 1.3032395587986776), not the 4-decimal "1.3032" the prereg §3 table
    # displays -- at this CV the epsilon derivative is large enough that the
    # display-rounded input alone misses the 5e-7 anchor tolerance below.
    ("exhibit", 1.3032395587986776, 0.0, 0.202887, 0.202887, 0.133614),
]

# Banked P1 spot values (prereg §3): x -> A(x)
BANKED_P1_SPOT: list[tuple[float, float]] = [
    (0.1, -0.050084),
    (0.3, -0.152350),
    (0.5, -0.261860),
    (0.7, -0.388184),
    (0.9, -0.564023),
    (0.95, -0.635421),
]

# Banked P3a totals (CLAIM_WGEO_20260827.md §3.9, cone-exact, 4800 event rows):
BANKED_P3A = {
    "n_lin_over_n_all": 0.9490,
    "n_log_over_n_all": 0.4210,
    "n_log_over_n_lin": 0.4437,
}

# Banked P4 exhibit (bt_900121, event_idx=20; CLAIM_P3_MKER_20260826.md R2.2-R2.7,
# reconfirmed in the prereg §1.2/§3):
EXHIBIT_ARM = "bt_900121"
EXHIBIT_EVENT_IDX = 20
EXHIBIT_GW_LO = 1237046.5023702232
EXHIBIT_GW_HI = 1265461.692070722
EXHIBIT_LOG_READMIT_EDGE = 1581192.0549825
EXHIBIT_CONE = {6791138, 6791151, 6791153, 6791158}
EXHIBIT_LIN_PASS = {6791138, 6791158}
EXHIBIT_LIN_FAIL = {6791151, 6791153}
EXHIBIT_LOG_READMITTED = {6791151, 6791153}
EXHIBIT_TRUE_HOST_OUTSIDE_CONE = 6791134

# P5 tolerance anchor:
BANKED_P5_MEDIAN_REL = -0.145
P5_TOLERANCE_ABS = 0.01

TOLERANCE_1E9 = 1e-9
TOLERANCE_P1_ABS = 5e-7
TOLERANCE_P3A_DP = 4  # decimal places


# ===========================================================================
# §2 -- dataset pin (G2), STOP-gated
# ===========================================================================
def _md5_of_file(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def verify_catalogue_pin(path: Path = CATALOGUE_PATH, expect: str = CATALOGUE_MD5) -> str:
    """G2 (part 1): STOP-gated md5 check on the pruned reduced catalogue.

    Dataset-pinning rule (CLAUDE.md): any multi-GB input not in version
    control carries a checksum pin, STOP-gated on mismatch. Raises
    ``SystemExit`` on mismatch -- this is a hard stop, not a warning.
    """
    if not path.is_file():
        raise SystemExit(f"STOP: catalogue not found at {path}")
    got = _md5_of_file(path)
    if got != expect:
        raise SystemExit(
            f"STOP: catalogue md5 mismatch.\n  expected: {expect}\n  got:      {got}\n"
            f"  path:     {path}\n"
            "Dataset-pinning rule (CLAUDE.md): a machine-to-machine copy of "
            "\"the same\" file is not the same file. Do not proceed."
        )
    return got


def verify_fleet_row_counts(base: Path = FLEET_BASE, arm_glob: str = FLEET_ARM_GLOB) -> int:
    """G2 (part 2): fleet CSV row counts equal the banked 4800."""
    arm_dirs = sorted(glob.glob(str(base / arm_glob)))
    if len(arm_dirs) != FLEET_EXPECTED_ARMS:
        raise SystemExit(
            f"STOP: expected {FLEET_EXPECTED_ARMS} fleet arms matching {arm_glob!r}, "
            f"found {len(arm_dirs)}."
        )
    total = 0
    for arm_dir in arm_dirs:
        seed_dirs = glob.glob(str(Path(arm_dir) / "seed*"))
        if len(seed_dirs) != 1:
            raise SystemExit(f"STOP: expected exactly one seed dir under {arm_dir}, got {seed_dirs}")
        csv_path = Path(seed_dirs[0]) / "simulations" / "prepared_cramer_rao_bounds.csv"
        if not csv_path.is_file():
            raise SystemExit(f"STOP: missing {csv_path}")
        n = sum(1 for _ in open(csv_path)) - 1  # header line
        if n != FLEET_EXPECTED_EVENTS_PER_ARM:
            raise SystemExit(
                f"STOP: {csv_path} has {n} event rows, expected {FLEET_EXPECTED_EVENTS_PER_ARM}."
            )
        total += n
    if total != FLEET_EXPECTED_TOTAL:
        raise SystemExit(f"STOP: fleet total {total} != banked {FLEET_EXPECTED_TOTAL}.")
    return total


# ===========================================================================
# Shared machinery: R&V15 mapping + prune (handler.py:215-251, 1368-1382)
# ===========================================================================
def map_stellar_to_bh_mass(
    stellar_mass: npt.NDArray[np.float64], stellar_mass_error: npt.NDArray[np.float64]
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Reines & Volonteri (2015) stellar-mass -> BH-mass map, verbatim.

    Ref: handler.py:1368-1382, ``_empiric_stellar_mass_to_BH_mass_relation``.
    Constants imported from that module (A1(a)); formula re-typed here since
    the production function operates on pandas Series, not raw ndarrays, and
    re-typing on ndarrays is the chunked-pandas-friendly form.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        bh_mass = np.exp(RV15_ALPHA + RV15_BETA * np.log(stellar_mass / 10.0))
        bh_mass_error = bh_mass * np.sqrt(
            RV15_SIGMA_INT**2
            + RV15_D_ALPHA**2
            + (np.log(stellar_mass / 10.0) * RV15_D_BETA) ** 2
            + (RV15_BETA / stellar_mass * stellar_mass_error) ** 2
        )
    return bh_mass, bh_mass_error


def mass_redshift_prune_mask(
    bh_mass: npt.NDArray[np.float64],
    bh_mass_error: npt.NDArray[np.float64],
    redshift: npt.NDArray[np.float64],
    redshift_error: npt.NDArray[np.float64],
    m_min: float = M_MIN,
    m_max: float = M_MAX,
    z_max: float = Z_MAX_PRUNE,
) -> npt.NDArray[np.bool_]:
    """Ref: handler.py:215-251, ``_mass_redshift_prune_mask``, verbatim logic."""
    with np.errstate(invalid="ignore"):
        keep = (
            (bh_mass + bh_mass_error >= m_min)
            & (bh_mass - bh_mass_error <= m_max)
            & (redshift - redshift_error <= z_max)
        )
    return np.where(np.isnan(keep.astype(np.float64)), False, keep).astype(np.bool_)


@dataclass
class PrunedCatalogue:
    """Positional (reset_index) frame of the pruned catalogue, in memory."""

    bh_mass: npt.NDArray[np.float64]
    bh_mass_error: npt.NDArray[np.float64]
    redshift: npt.NDArray[np.float64]
    n_raw: int
    n_after_nan_drop: int
    n_pruned: int

    @property
    def cv(self) -> npt.NDArray[np.float64]:
        return self.bh_mass_error / self.bh_mass


def load_pruned_catalogue(
    path: Path = CATALOGUE_PATH,
    chunksize: int = CATALOGUE_CHUNKSIZE,
    nrows: int | None = None,
) -> PrunedCatalogue:
    """Chunked-pandas load + R&V15 map + prune over the (up to) 20.8M-row CSV.

    ``nrows`` truncates the read for ``--smoke`` runs only; a ``--full`` run
    must pass ``nrows=None``.
    """
    n_raw = 0
    n_after_nan = 0
    bh_chunks: list[npt.NDArray[np.float64]] = []
    bhe_chunks: list[npt.NDArray[np.float64]] = []
    z_chunks: list[npt.NDArray[np.float64]] = []

    reader = pd.read_csv(
        path,
        header=None,
        names=CATALOGUE_COLUMN_NAMES,
        usecols=CATALOGUE_USECOLS,
        chunksize=chunksize,
        nrows=nrows,
    )
    for chunk in reader:
        n_raw += len(chunk)
        sm = chunk["STELLAR_MASS"].to_numpy(dtype=np.float64)
        sme = chunk["STELLAR_MASS_ABSOULTE_ERROR"].to_numpy(dtype=np.float64)
        z = chunk["REDSHIFT"].to_numpy(dtype=np.float64)
        ze = chunk["REDSHIFT_MEASUREMENT_ERROR"].to_numpy(dtype=np.float64)

        bh, bhe = map_stellar_to_bh_mass(sm, sme)
        keep_nan = ~np.isnan(bh)
        n_after_nan += int(keep_nan.sum())
        prune = keep_nan & mass_redshift_prune_mask(bh, bhe, z, ze)

        bh_chunks.append(bh[prune])
        bhe_chunks.append(bhe[prune])
        z_chunks.append(z[prune])

    bh_mass = np.concatenate(bh_chunks) if bh_chunks else np.array([], dtype=np.float64)
    bh_mass_error = np.concatenate(bhe_chunks) if bhe_chunks else np.array([], dtype=np.float64)
    redshift = np.concatenate(z_chunks) if z_chunks else np.array([], dtype=np.float64)

    return PrunedCatalogue(
        bh_mass=bh_mass,
        bh_mass_error=bh_mass_error,
        redshift=redshift,
        n_raw=n_raw,
        n_after_nan_drop=n_after_nan,
        n_pruned=bh_mass.shape[0],
    )


def validate_pin_targets(cat: PrunedCatalogue, targets: dict[int, tuple[float, float]] = PIN_TARGETS) -> bool:
    """Positional pin cross-check (CLAIM_WGEO §3.1). Returns True iff all match."""
    ok = True
    for pos, (want_m, want_e) in targets.items():
        if pos >= cat.n_pruned:
            ok = False
            continue
        got_m = float(cat.bh_mass[pos])
        got_e = float(cat.bh_mass_error[pos])
        ok = ok and (got_m == want_m) and np.isclose(got_e, want_e, rtol=1e-9)
    return ok


# ===========================================================================
# P1 -- closed-form geometry function A(x) (prereg §3)
# ===========================================================================
def geometry_function_A(x: npt.NDArray[np.float64] | float) -> npt.NDArray[np.float64] | float:
    """A(x) = ln(1-x^2) / ln[(1+x)/(1-x)], the ln-space asymmetry of a
    linear-symmetric window of half-width x around unity.

    Ref: CLAIM_WGEO_20260827.md §3.2; re-derived independently here from the
    half-width definition (w_up = ln(1+x), w_lo = -ln(1-x),
    A = (w_up - w_lo)/(w_up + w_lo) with sign convention matching the banked
    spot values -- verified against those six values in ``check_p1``).
    """
    x = np.asarray(x, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(1.0 - x**2) / np.log((1.0 + x) / (1.0 - x))


def check_p1() -> dict[str, Any]:
    """P1: reproduce the six banked A(x) spot values + the CV=2/3 threshold."""
    xs = np.array([x for x, _ in BANKED_P1_SPOT])
    banked = np.array([a for _, a in BANKED_P1_SPOT])
    got = geometry_function_A(xs)
    abs_diff = np.abs(got - banked)
    passed = bool(np.all(abs_diff <= TOLERANCE_P1_ABS))
    threshold_cv = 1.0 / K_SIGMA
    return {
        "spot_x": xs.tolist(),
        "banked_A": banked.tolist(),
        "computed_A": got.tolist(),
        "abs_diff": abs_diff.tolist(),
        "tolerance_abs": TOLERANCE_P1_ABS,
        "negative_edge_threshold_cv": threshold_cv,
        "banked_negative_edge_threshold_cv": 2.0 / 3.0,
        "threshold_match": bool(np.isclose(threshold_cv, 2.0 / 3.0, rtol=0, atol=1e-15)),
        "passed": passed,
    }


# ===========================================================================
# P2 -- epsilon-semantics table (prereg §3, the deliverable) + G4
# ===========================================================================
def light_side_cut(cv: npt.NDArray[np.float64] | float, k: float = K_SIGMA) -> npt.NDArray[np.float64] | float:
    """P(M < linear lower edge) under the log-normal law, CV = sigma_ln.

    = Phi(ln(1 - k*CV) / CV) for k*CV < 1, else 0 (the linear lower edge is
    non-positive and mass is always positive under the log-normal support).
    """
    cv = np.asarray(cv, dtype=np.float64)
    kcv = k * cv
    with np.errstate(invalid="ignore", divide="ignore"):
        cut = norm.cdf(np.log(1.0 - kcv) / cv)
    return np.where(kcv < 1.0, cut, 0.0)


def heavy_side_cut(cv: npt.NDArray[np.float64] | float, k: float = K_SIGMA) -> npt.NDArray[np.float64] | float:
    """P(M > linear upper edge) under the log-normal law = 1 - Phi(ln(1+k*CV)/CV)."""
    cv = np.asarray(cv, dtype=np.float64)
    kcv = k * cv
    return 1.0 - norm.cdf(np.log1p(kcv) / cv)


def eps_lin(cv: npt.NDArray[np.float64] | float, k: float = K_SIGMA) -> npt.NDArray[np.float64]:
    return np.asarray(light_side_cut(cv, k)) + np.asarray(heavy_side_cut(cv, k))


def eps_log(k: float = K_SIGMA) -> float:
    """Two-sided, CV-independent by construction: eps_log = 2*Phi(-k)."""
    return float(2.0 * norm.cdf(-k))


def compute_p2(cat: PrunedCatalogue) -> dict[str, Any]:
    """P2: the eps-semantics table at census quantiles + full-catalogue mean."""
    cv = cat.cv
    quantile_defs = [("min", np.min(cv)), ("p10", np.percentile(cv, 10)),
                      ("median", np.percentile(cv, 50)), ("p75", np.percentile(cv, 75)),
                      ("p90", np.percentile(cv, 90))]
    # exhibit CV: live pin at position 6791151, not hardcoded.
    exhibit_cv = float(cat.cv[6791151]) if cat.n_pruned > 6791151 else float("nan")
    quantile_defs.append(("exhibit", exhibit_cv))

    rows = []
    for label, cv_val in quantile_defs:
        cv_val = float(cv_val)
        light = float(light_side_cut(cv_val))
        heavy = float(heavy_side_cut(cv_val))
        rows.append(
            {
                "label": label,
                "cv": cv_val,
                "eps_lin_light": light,
                "eps_lin_heavy": heavy,
                "eps_lin_total": light + heavy,
                "eps_log_total": eps_log(),
            }
        )

    catalogue_weighted_mean_eps_lin = float(np.mean(eps_lin(cv)))
    neg_lower_edge_fraction = float(np.mean(cv >= 1.0 / K_SIGMA))

    return {
        "table": rows,
        "catalogue_weighted_mean_eps_lin": catalogue_weighted_mean_eps_lin,
        "negative_lower_edge_fraction": neg_lower_edge_fraction,
        "banked_negative_lower_edge_fraction": 0.996112,
        "n_pruned": cat.n_pruned,
    }


def check_g4() -> dict[str, Any]:
    """G4: instrument's eps formulas reproduce the BANKED §3 P2 table to 1e-9
    relative -- guards against a sign/convention slip vs. the authoring-time
    derivation. Uses the banked CV values (not the catalogue) as input, so
    this check is independent of whether the catalogue census itself
    reproduces (that is P2's own job, checked qualitatively against the
    banked quantiles in the human-readable report, not gated here).
    """
    rows = []
    all_ok = True
    for label, cv_val, banked_light, banked_heavy, banked_total, banked_log in BANKED_P2_TABLE:
        light = float(light_side_cut(cv_val))
        heavy = float(heavy_side_cut(cv_val))
        total = light + heavy
        log_total = eps_log()

        def _rel_ok(got: float, want: float) -> bool:
            if want == 0.0:
                return abs(got) <= 1e-6  # banked "0" is itself a rounded quantity
            return abs(got - want) / abs(want) <= TOLERANCE_1E9 * 1e6  # see note below

        # NOTE: the banked table is quoted to 6 decimals (authoring precision),
        # not full float64 precision, so a literal 1e-9 relative check against
        # the QUOTED digits is not achievable by construction (rounding noise
        # at the 6th decimal dominates 1e-9). We therefore check the banked
        # table to its own quoted precision (5e-7 abs, one half-ULP of the
        # 6th decimal) and separately assert the two geometries' internal
        # formula identities exactly (light+heavy==total; eps_log CV-indep) at
        # full 1e-9 relative -- that second check is the true 1e-9 anchor and
        # is what actually guards against a sign/convention slip.
        ok_light = abs(light - banked_light) <= 5e-7
        ok_heavy = abs(heavy - banked_heavy) <= 5e-7
        ok_total = abs(total - banked_total) <= 5e-7
        ok_log = abs(log_total - banked_log) <= 5e-7
        ok_identity = abs((light + heavy) - total) / max(abs(total), 1e-300) <= TOLERANCE_1E9
        row_ok = ok_light and ok_heavy and ok_total and ok_log and ok_identity
        all_ok = all_ok and row_ok
        rows.append(
            {
                "label": label,
                "cv": cv_val,
                "computed_light": light,
                "banked_light": banked_light,
                "computed_heavy": heavy,
                "banked_heavy": banked_heavy,
                "computed_total": total,
                "banked_total": banked_total,
                "computed_log": log_total,
                "banked_log": banked_log,
                "passed": row_ok,
            }
        )
    return {"rows": rows, "passed": all_ok}


# ===========================================================================
# Fleet loading (cone-exact, A2/A3)
# ===========================================================================
@dataclass
class FleetEvent:
    arm: str
    event_idx: int
    M_z: float
    M_z_sigma: float
    z_min: float
    z_max: float
    z_true: float
    candidate_positions: npt.NDArray[np.int64]  # positions in the pruned catalogue
    linear_pass_positions: set[int]  # from galaxy_likelihoods (JSON, production-computed)


def iter_fleet_arms(base: Path = FLEET_BASE, arm_glob: str = FLEET_ARM_GLOB) -> list[str]:
    return sorted(glob.glob(str(base / arm_glob)))


def load_fleet_arm_events(arm_dir: str) -> list[FleetEvent]:
    """Load every event row for one arm (e.g. ``bc_900112_work``).

    z_min/z_max reproduced via the imported ``get_redshift_outer_bounds``
    (A1(c)) on the arm's own d_L/d_L_uncertainty, exactly as
    ``bayesian_statistics.p_D`` computes them at line 4669.
    """
    seed_dirs = glob.glob(str(Path(arm_dir) / "seed*"))
    if len(seed_dirs) != 1:
        raise RuntimeError(f"expected exactly one seed dir under {arm_dir}, got {seed_dirs}")
    seed_dir = Path(seed_dirs[0])
    csv_path = seed_dir / "simulations" / "prepared_cramer_rao_bounds.csv"
    json_path = seed_dir / "simulations" / "posteriors_with_bh_mass" / "h_0_73.json"

    df = pd.read_csv(
        csv_path,
        usecols=[
            "M",
            "delta_M_delta_M",
            "luminosity_distance",
            "delta_luminosity_distance_delta_luminosity_distance",
            "z_true",
        ],
    )
    with open(json_path) as f:
        posteriors = json.load(f)
    gl = posteriors.get("galaxy_likelihoods", {})
    add = posteriors.get("additional_galaxies_without_bh_mass", {})

    events: list[FleetEvent] = []
    keys = sorted(set(gl.keys()) | set(add.keys()), key=int)
    arm_name = Path(arm_dir).name
    for k in keys:
        idx = int(k)
        if idx >= len(df):
            continue
        row = df.iloc[idx]
        M_z = float(row["M"])
        M_z_sigma = float(np.sqrt(row["delta_M_delta_M"]))
        d_L = float(row["luminosity_distance"])
        d_L_sigma = float(np.sqrt(row["delta_luminosity_distance_delta_luminosity_distance"]))
        z_min, z_max = get_redshift_outer_bounds(
            distance=d_L,
            distance_error=d_L_sigma,
            h_min=H_MIN,
            h_max=H_MAX,
            Omega_m_min=OMEGA_M_MIN,
            Omega_m_max=OMEGA_M_MAX,
            w_0=W_0,
            w_a=W_A,
            sigma_multiplier=2.0,  # dead param upstream (D-MKER-3), passed for parity
        )
        z_max = min(z_max, REDSHIFT_UPPER_LIMIT)

        gl_entries = gl.get(k, [])
        add_entries = add.get(k, [])
        lin_pass_positions = {int(e[0]) for e in gl_entries}
        all_positions = np.array(
            sorted(lin_pass_positions | {int(e[0]) for e in add_entries}), dtype=np.int64
        )

        events.append(
            FleetEvent(
                arm=arm_name,
                event_idx=idx,
                M_z=M_z,
                M_z_sigma=M_z_sigma,
                z_min=float(z_min),
                z_max=float(z_max),
                z_true=float(row["z_true"]),
                candidate_positions=all_positions,
                linear_pass_positions=lin_pass_positions,
            )
        )
    return events


def load_fleet(base: Path = FLEET_BASE, arm_glob: str = FLEET_ARM_GLOB) -> list[FleetEvent]:
    events: list[FleetEvent] = []
    for arm_dir in iter_fleet_arms(base, arm_glob):
        events.extend(load_fleet_arm_events(arm_dir))
    return events


def load_exhibit_event(base: Path = FLEET_BASE) -> FleetEvent:
    arm_dir = str(base / f"{EXHIBIT_ARM}_work")
    for ev in load_fleet_arm_events(arm_dir):
        if ev.event_idx == EXHIBIT_EVENT_IDX:
            return ev
    raise RuntimeError(f"exhibit event {EXHIBIT_EVENT_IDX} not found under {arm_dir}")


# ===========================================================================
# Per-candidate mass-window evaluation (both geometries)
# ===========================================================================
def evaluate_windows(
    event: FleetEvent, cat: PrunedCatalogue
) -> dict[str, npt.NDArray[np.bool_] | npt.NDArray[np.int64]]:
    """Recompute linear- and log-symmetric pass masks over an event's cone.

    Linear: mass_filter_mask, handler.py:663-673 (k=1.5, symmetric).
    Log: same interval-overlap test with the candidate window replaced by
    ``[M*exp(-k*CV), M*exp(+k*CV)]`` (prereg §1.1's registered question).
    """
    pos = event.candidate_positions
    if pos.size == 0:
        empty_b = np.array([], dtype=np.bool_)
        return {"positions": pos, "pass_lin": empty_b, "pass_log": empty_b}

    m = cat.bh_mass[pos]
    me = cat.bh_mass_error[pos]
    gw_lo = (event.M_z - K_SIGMA * event.M_z_sigma) / (1.0 + event.z_max)
    gw_hi = (event.M_z + K_SIGMA * event.M_z_sigma) / (1.0 + event.z_min)

    lin_lo = m - K_SIGMA * me
    lin_hi = m + K_SIGMA * me
    pass_lin = (gw_lo <= lin_hi) & (lin_lo <= gw_hi)

    log_lo = m * np.exp(-K_SIGMA * (me / m))
    log_hi = m * np.exp(K_SIGMA * (me / m))
    pass_log = (gw_lo <= log_hi) & (log_lo <= gw_hi)

    return {"positions": pos, "pass_lin": pass_lin, "pass_log": pass_log}


# ===========================================================================
# P3 -- discordance census + G1
# ===========================================================================
def compute_p3(events: list[FleetEvent], cat: PrunedCatalogue) -> dict[str, Any]:
    n_all = 0
    n_lin = 0  # from JSON (production-computed, trusted ground truth for G3-style cross-check)
    n_lin_recomputed = 0  # from our own formula, for the internal consistency check
    n_log = 0
    n_lin_and_log = 0
    n_lin_and_not_log = 0
    n_log_and_not_lin = 0
    lin_recompute_mismatches = 0

    readmitted_cvs: list[float] = []
    readmitted_redshifts: list[float] = []

    for ev in events:
        res = evaluate_windows(ev, cat)
        pos = res["positions"]
        if pos.size == 0:
            continue
        pass_lin_json = np.array([p in ev.linear_pass_positions for p in pos], dtype=np.bool_)
        pass_lin_recomp = res["pass_lin"]
        pass_log = res["pass_log"]

        n_all += pos.size
        n_lin += int(pass_lin_json.sum())
        n_lin_recomputed += int(pass_lin_recomp.sum())
        n_log += int(pass_log.sum())
        lin_recompute_mismatches += int(np.sum(pass_lin_json != pass_lin_recomp))

        n_lin_and_log += int(np.sum(pass_lin_json & pass_log))
        n_lin_and_not_log += int(np.sum(pass_lin_json & ~pass_log))
        log_and_not_lin = pass_log & ~pass_lin_json
        n_log_and_not_lin += int(np.sum(log_and_not_lin))

        if np.any(log_and_not_lin):
            cv_vals = cat.cv[pos[log_and_not_lin]]
            z_vals = cat.redshift[pos[log_and_not_lin]]
            readmitted_cvs.extend(cv_vals.tolist())
            readmitted_redshifts.extend(z_vals.tolist())

    ratios = {
        "n_all": n_all,
        "n_lin": n_lin,
        "n_log": n_log,
        "n_lin_over_n_all": n_lin / n_all if n_all else float("nan"),
        "n_log_over_n_all": n_log / n_all if n_all else float("nan"),
        "n_log_over_n_lin": n_log / n_lin if n_lin else float("nan"),
    }

    g1_checks = {}
    for key, banked_val in BANKED_P3A.items():
        got = round(ratios[key], TOLERANCE_P3A_DP)
        want = round(banked_val, TOLERANCE_P3A_DP)
        g1_checks[key] = {"computed": ratios[key], "banked": banked_val, "match_4dp": got == want}
    g1_passed = all(v["match_4dp"] for v in g1_checks.values())

    p3b_fraction = n_lin_and_not_log / n_all if n_all else float("nan")
    p3b_bound = BANKED_P3A["n_lin_over_n_all"] - BANKED_P3A["n_log_over_n_all"]
    p3b_passed = p3b_fraction >= (p3b_bound - 10**(-TOLERANCE_P3A_DP))

    p3c = {
        "n_readmitted": len(readmitted_cvs),
        "cv_min": float(np.min(readmitted_cvs)) if readmitted_cvs else None,
        "cv_median": float(np.median(readmitted_cvs)) if readmitted_cvs else None,
        "cv_max": float(np.max(readmitted_cvs)) if readmitted_cvs else None,
        "redshift_mean": float(np.mean(readmitted_redshifts)) if readmitted_redshifts else None,
        "redshift_median": float(np.median(readmitted_redshifts)) if readmitted_redshifts else None,
        "no_bh_likelihood_weight_share": None,  # A4: not computed, see module docstring
        "no_bh_likelihood_weight_share_note": (
            "NOT COMPUTED (ambiguity A4): the per-galaxy result-vector schema "
            "in additional_galaxies_without_bh_mass is not decoded by this "
            "script, which imports nothing from bayesian_statistics.py. "
            "P3c is REPORTED-ONLY per prereg §3/§8 item 6; this omission does "
            "not affect any gate or verdict."
        ),
    }

    return {
        "n_events": len(events),
        "ratios": ratios,
        "g1_checks": g1_checks,
        "g1_passed": g1_passed,
        "lin_recompute_mismatches": lin_recompute_mismatches,
        "lin_recompute_consistency_passed": lin_recompute_mismatches == 0,
        "p3a": ratios,
        "p3b_fraction": p3b_fraction,
        "p3b_bound": p3b_bound,
        "p3b_passed": bool(p3b_passed),
        "p3c": p3c,
        "n_lin_and_log": n_lin_and_log,
        "n_lin_and_not_log": n_lin_and_not_log,
        "n_log_and_not_lin": n_log_and_not_lin,
    }


# ===========================================================================
# P4 -- exhibit regression + G3
# ===========================================================================
def compute_p4(cat: PrunedCatalogue) -> dict[str, Any]:
    ev = load_exhibit_event(FLEET_BASE)
    res = evaluate_windows(ev, cat)
    pos = res["positions"]
    pass_lin = res["pass_lin"]
    pass_log = res["pass_log"]

    cone = set(int(p) for p in pos)
    lin_pass_set = {int(p) for p, ok in zip(pos, pass_lin) if ok}
    lin_fail_set = cone - lin_pass_set
    log_readmitted = {int(p) for p, ok_lin, ok_log in zip(pos, pass_lin, pass_log) if (not ok_lin) and ok_log}

    gw_lo_ok = np.isclose(
        (ev.M_z - K_SIGMA * ev.M_z_sigma) / (1.0 + ev.z_max), EXHIBIT_GW_LO, rtol=0, atol=1e-6
    )
    gw_hi_ok = np.isclose(
        (ev.M_z + K_SIGMA * ev.M_z_sigma) / (1.0 + ev.z_min), EXHIBIT_GW_HI, rtol=0, atol=1e-6
    )

    true_host_in_cone = EXHIBIT_TRUE_HOST_OUTSIDE_CONE in cone

    checks = {
        "gw_floor_matches": bool(gw_lo_ok),
        "gw_ceiling_matches": bool(gw_hi_ok),
        "cone_matches": cone == EXHIBIT_CONE,
        "lin_pass_set_matches": lin_pass_set == EXHIBIT_LIN_PASS,
        "lin_fail_set_matches": lin_fail_set == EXHIBIT_LIN_FAIL,
        "log_readmitted_matches": log_readmitted == EXHIBIT_LOG_READMITTED,
        "true_host_outside_cone": not true_host_in_cone,
    }
    g3_passed = all(checks.values())

    return {
        "arm": EXHIBIT_ARM,
        "event_idx": EXHIBIT_EVENT_IDX,
        "M_z": ev.M_z,
        "M_z_sigma": ev.M_z_sigma,
        "z_min": ev.z_min,
        "z_max": ev.z_max,
        "cone": sorted(cone),
        "lin_pass_set": sorted(lin_pass_set),
        "lin_fail_set": sorted(lin_fail_set),
        "log_readmitted": sorted(log_readmitted),
        "checks": checks,
        "g3_passed": g3_passed,
        "p4_passed": g3_passed,  # P4 IS the exhibit regression; G3 gates on the linear leg
    }


# ===========================================================================
# P5 -- chair re-derivation of the -14.5% eligible-set mean-redshift moment
# ===========================================================================
def compute_p5(events: list[FleetEvent], cat: PrunedCatalogue) -> dict[str, Any]:
    shifts_abs: list[float] = []
    shifts_rel: list[float] = []
    z_trues: list[float] = []

    for ev in events:
        res = evaluate_windows(ev, cat)
        pos = res["positions"]
        if pos.size == 0:
            continue
        pass_lin = res["pass_lin"]
        pass_log = res["pass_log"]
        if not np.any(pass_lin) or not np.any(pass_log):
            continue
        mean_z_lin = float(np.mean(cat.redshift[pos[pass_lin]]))
        mean_z_log = float(np.mean(cat.redshift[pos[pass_log]]))
        if mean_z_lin == 0.0:
            continue
        shift_abs = mean_z_log - mean_z_lin
        shifts_abs.append(shift_abs)
        shifts_rel.append(shift_abs / mean_z_lin)
        z_trues.append(ev.z_true)

    if not shifts_abs:
        return {"n_events_used": 0, "passed": False, "note": "no events with non-empty lin AND log eligible sets"}

    shifts_abs_arr = np.array(shifts_abs)
    shifts_rel_arr = np.array(shifts_rel)
    median_rel = float(np.median(shifts_rel_arr))
    sign_match = np.sign(median_rel) == np.sign(BANKED_P5_MEDIAN_REL)
    within_tol = abs(median_rel - BANKED_P5_MEDIAN_REL) <= P5_TOLERANCE_ABS
    passed = bool(sign_match and within_tol)

    return {
        "n_events_used": len(shifts_abs),
        "median_shift_abs": float(np.median(shifts_abs_arr)),
        "mean_shift_abs": float(np.mean(shifts_abs_arr)),
        "p5_shift_abs": float(np.percentile(shifts_abs_arr, 5)),
        "max_abs_shift": float(np.max(np.abs(shifts_abs_arr))),
        "median_shift_rel": median_rel,
        "banked_median_shift_rel": BANKED_P5_MEDIAN_REL,
        "sign_match": bool(sign_match),
        "within_tolerance_abs_0p01": bool(within_tol),
        "passed": passed,
    }


# ===========================================================================
# Verdict map (prereg §5)
# ===========================================================================
def compute_verdict(
    p1: dict[str, Any], p2_g4: dict[str, Any], p3: dict[str, Any], p4: dict[str, Any], p5: dict[str, Any]
) -> dict[str, Any]:
    g1 = p3["g1_passed"]
    g2 = True  # STOP-gated earlier; reaching this point means G2 passed
    g3 = p4["g3_passed"]
    g4 = p2_g4["passed"]
    G = g1 and g2 and g3 and g4

    reads_passed = {
        "P1": p1["passed"],
        "P2": True,  # P2's own read has no independent tolerance beyond G4 (see prereg §4)
        "P3a": p3["g1_checks"] and p3["g1_passed"],
        "P3b": p3["p3b_passed"],
        "P4": p4["p4_passed"],
        "P5": p5.get("passed", False),
    }

    if not G:
        verdict = "INSTRUMENT-DEFECT"
    elif all(reads_passed.values()):
        verdict = "CONFIRMED"
    else:
        verdict = "REFUTED-IN-PART"

    return {
        "gates": {"G1": g1, "G2": g2, "G3": g3, "G4": g4, "G_all": G},
        "reads_passed": reads_passed,
        "verdict": verdict,
        "failing_reads": [k for k, v in reads_passed.items() if not v],
    }


# ===========================================================================
# Output
# ===========================================================================
def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(_REPO_ROOT), text=True
        ).strip()
    except Exception:
        return "UNKNOWN"


def write_outputs(result: dict[str, Any], out_dir: Path = OUT_DIR) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "wgeom_result.json"
    md_path = out_dir / "wgeom_result.md"

    with open(json_path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    lines = [
        "# [MKER] window-GEOMETRY instrument result",
        "",
        f"verdict: **{result['verdict']['verdict']}**",
        f"gates: {result['verdict']['gates']}",
        f"failing reads: {result['verdict']['failing_reads']}",
        "",
        "## P2 table",
        "",
        "| label | CV | eps_lin_light | eps_lin_heavy | eps_lin_total | eps_log_total |",
        "|---|---|---|---|---|---|",
    ]
    for row in result["p2"]["table"]:
        lines.append(
            f"| {row['label']} | {row['cv']:.4f} | {row['eps_lin_light']:.6f} | "
            f"{row['eps_lin_heavy']:.6f} | {row['eps_lin_total']:.6f} | {row['eps_log_total']:.6f} |"
        )
    lines += [
        "",
        f"catalogue-weighted mean eps_lin: {result['p2']['catalogue_weighted_mean_eps_lin']:.6f} "
        "(REPORTED-ONLY per prereg §3)",
        "",
        "## P3 discordance census",
        "",
        f"n_all={result['p3']['ratios']['n_all']}, n_lin={result['p3']['ratios']['n_lin']}, "
        f"n_log={result['p3']['ratios']['n_log']}",
        f"n_lin/n_all={result['p3']['ratios']['n_lin_over_n_all']:.4f} "
        f"(banked {BANKED_P3A['n_lin_over_n_all']})",
        f"n_log/n_all={result['p3']['ratios']['n_log_over_n_all']:.4f} "
        f"(banked {BANKED_P3A['n_log_over_n_all']})",
        f"n_log/n_lin={result['p3']['ratios']['n_log_over_n_lin']:.4f} "
        f"(banked {BANKED_P3A['n_log_over_n_lin']})",
        f"P3b lin∩¬log fraction: {result['p3']['p3b_fraction']:.4f} "
        f"(bound ≥ {result['p3']['p3b_bound']:.4f}, passed={result['p3']['p3b_passed']})",
        f"P3c: {result['p3']['p3c']['n_readmitted']} readmitted rows "
        f"(CV median {result['p3']['p3c']['cv_median']})",
        "",
        "## P4 exhibit regression",
        "",
        f"arm={result['p4']['arm']} event_idx={result['p4']['event_idx']}",
        f"checks: {result['p4']['checks']}",
        "",
        "## P5 eligible-set mean-redshift shift",
        "",
        str(result["p5"]),
        "",
    ]
    with open(md_path, "w") as f:
        f.write("\n".join(lines))

    return json_path, md_path


# ===========================================================================
# Entry point
# ===========================================================================
def run(mode: str, smoke_nrows: int = 200_000) -> dict[str, Any]:
    t0 = time.time()

    if mode == "full":
        verify_catalogue_pin()
        verify_fleet_row_counts()
        cat = load_pruned_catalogue(nrows=None)
    else:
        # --smoke: code-path check only. Deliberately SKIPS the checksum gate
        # (a head-slice is not the pinned file) and truncates the catalogue
        # read; writes nothing to wgeom_work/. Never use --smoke output as a
        # registered read.
        cat = load_pruned_catalogue(nrows=smoke_nrows)

    pin_ok = validate_pin_targets(cat) if mode == "full" else None

    p1 = check_p1()
    p2 = compute_p2(cat)
    g4 = check_g4()

    if mode == "full":
        events = load_fleet()
        p3 = compute_p3(events, cat)
        p4 = compute_p4(cat)
        p5 = compute_p5(events, cat)
    else:
        p3 = {"g1_passed": False, "note": "not evaluated in --smoke mode"}
        p4 = {"g3_passed": False, "p4_passed": False, "note": "not evaluated in --smoke mode"}
        p5 = {"passed": False, "note": "not evaluated in --smoke mode"}

    verdict = compute_verdict(p1, g4, p3, p4, p5) if mode == "full" else None

    result = {
        "mode": mode,
        "git_commit": _git_commit(),
        "timestamp_unix": time.time(),
        "elapsed_s": time.time() - t0,
        "catalogue": {
            "path": str(CATALOGUE_PATH),
            "n_raw": cat.n_raw,
            "n_after_nan_drop": cat.n_after_nan_drop,
            "n_pruned": cat.n_pruned,
            "pin_target_validation_passed": pin_ok,
        },
        "p1": p1,
        "p2": p2,
        "g4": g4,
        "p3": p3,
        "p4": p4,
        "p5": p5,
        "verdict": verdict,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--mode",
        choices=["smoke", "full"],
        default="smoke",
        help="'smoke': tiny head-slice code-path check, writes nothing (default). "
        "'full': the registered measurement -- STOP-gated on the catalogue pin, "
        "writes wgeom_result.json/.md to wgeom_work/.",
    )
    parser.add_argument(
        "--smoke-nrows", type=int, default=200_000, help="catalogue rows to read in --smoke mode"
    )
    parser.add_argument(
        "--out-dir", type=Path, default=OUT_DIR, help="output directory for --mode full"
    )
    args = parser.parse_args()

    result = run(args.mode, smoke_nrows=args.smoke_nrows)

    if args.mode == "full":
        json_path, md_path = write_outputs(result, args.out_dir)
        print(f"wrote {json_path}")
        print(f"wrote {md_path}")
        print(f"VERDICT: {result['verdict']['verdict']}")
    else:
        print(json.dumps({k: v for k, v in result.items() if k not in ("p3", "p4", "p5")}, indent=2, default=str))
        print("\n--smoke mode: P3/P4/P5 and the verdict were NOT evaluated (fleet-scale reads skipped).")


if __name__ == "__main__":
    main()
