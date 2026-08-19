r"""Option-B 1D production-correspondence harness (fidelity pilot G-0 only).

**What this instrument is.** The Option-B measurement registered in
``results/prod2d_closure_20260818/PREREGISTRATION_1D_CORRESPONDENCE.md`` (v2):
decompose the production 1D base tilt into information-starvation vs
form-defect components. This module currently implements **only gate G-0**
(prereg §4, the STOP gate that must pass before ANY arm runs) — the
production-wholesale fidelity pilot. It does **not** implement the
mirror-universe generator (B-0/B-σ/B-D2/E-DEN arms) yet: see
:class:`MirrorUniverseGenerator`, a registered-shape stub that raises
``NotImplementedError`` until a later build.

**D-A (fidelity: production-wholesale).** Per the prereg, the harness must
call production's own per-event assembly wholesale rather than re-deriving
the estimator from the written formulas (contrast
:mod:`darksiren_emri.validation.pp_coverage`, which is deliberately
independent). Two complementary layers are exercised here:

1. **End-to-end wholesale layer** (:func:`run_production_wholesale`): drives
   the real ``python -m darksiren_emri --evaluate`` entry point in a
   sandboxed CWD with the production iiib venue flags (``run_metadata_0.json``
   of the post-fix baseline), symlinked to the pinned production inputs
   (:data:`CRB_CSV_PATH`, the local reduced GLADE catalogue, the injection
   pool). This is the ``probe_n0_local.py`` pattern
   (``results/prod2d_closure_20260818/probe_n0_local.py``) generalized to two
   probe h-values in one context build.
2. **Harness-side combine re-orchestration** (:func:`reassemble_combine_no_bh`):
   calls the REAL module-level production functions
   :func:`~darksiren_emri.bayesian_inference.bayesian_statistics.path_a_mixture_objects`
   and
   :func:`~darksiren_emri.bayesian_inference.bayesian_statistics.path_a_completion_numerators`
   directly (not reimplemented) to re-derive ``alpha_G_phi``/``D_tilde_phi``/
   ``B_num_phi`` from the catalogue-leg and completion-leg per-event outputs,
   then assembles ``combined_no_bh`` with the harness's own formula
   ``(beta_G_phi * L_cat_no_bh + B_num_phi) / D_tilde_phi`` (recovered from
   ``bayesian_statistics.py:5248-5255`; ``beta_G_phi = alpha_G_phi / r_Malm``,
   ``beta_Gbar_phi = D_tilde_phi - alpha_G_phi``, both invertible from the
   three banked/produced Path-A columns) — this is the "own completion/
   assembly layer" the prereg's G-0 text requires, proven against production's
   already-written columns.

**Known scope limitation (disclosed, not fudged).** The completion numerator
``B_num(h)`` integrand (``completion_numerator_integrand``, Gray et al. 2020
Eq. 32) is a NESTED CLOSURE inside
:meth:`~darksiren_emri.bayesian_inference.bayesian_statistics.BayesianStatistics.p_Di`
(``bayesian_statistics.py:4852-4893``, not a module-level export) — the exact
"class-method closure" flagged in prereg §4's G-0 text. This harness does
**not** independently re-derive ``B_num``/``L_cat`` from ``single_host_likelihood``
per candidate; it takes them from the wholesale production run (layer 1) and
re-derives only the combine layer (layer 2) from leaf functions. Building an
independent leaf-level ``L_cat``/``B_num`` re-derivation (needed for the
mirror-universe generator, since synthetic events will not always have a
``BayesianStatistics``-shaped ``Detection``/candidate-host context) is
flagged as follow-up work, NOT claimed as done by this G-0 pass.

CPU-only. No cupy import, direct or transitive.

References:
    Gray et al. (2020), arXiv:1908.06050, Eqs. (29), (32), (A.9), (A.10).
    Mandel, Farr & Gair (2019), arXiv:1809.02063, Eqs. (5)-(7).
"""

import argparse
import hashlib
import json
import logging
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

from darksiren_emri.bayesian_inference.bayesian_statistics import (
    path_a_completion_numerators,
    path_a_mixture_objects,
)

_LOGGER = logging.getLogger(__name__)

PREREG_PATH = "results/prod2d_closure_20260818/PREREGISTRATION_1D_CORRESPONDENCE.md"

# ── D-B/G-3 registered pinned inputs (production iiib venue) ────────────────
# CRB CSV: the seed61000 event realization, VT-D1-pinned (identical to
# venue_transfer.py's CRB_CSV_MD5 -- verified 2026-08-19 recon:
# 9a1f2a14384a9281c97ca3be312ddaab).
_REPO_ROOT = Path(__file__).resolve().parents[2]
CRB_CSV_PATH = str(
    _REPO_ROOT / "results/run_20260804_postfix/iiib/diagnostics/prepared_cramer_rao_bounds.csv"
)
CRB_CSV_MD5 = "9a1f2a14384a9281c97ca3be312ddaab"
# Injection pool for SimulationDetectionProbability's p_det grid. 2026-08-19
# recon (this build) resolved an input-availability ambiguity flagged by the
# task: the local mix200k pool
# (results/campaign51_20260728/realistic_20260729/gate_b_20260730/
# injection_pool_mix200k_20260728/, closed_loop_gfrac.DEFAULT_INJECTION_DIR)
# and the cluster's run_20260729_seed61000/simulations/injections ARE THE
# SAME POOL -- the cluster run directory's injections/ subfolder is itself a
# set of symlinks back to injection_pool_mix200k_20260728/ (confirmed by
# `ssh bwunicluster` inspection: rsync -a of the "seed61000" path pulled
# broken symlinks pointing at .../injection_pool_mix200k_20260728/*, i.e. the
# same local directory already present). No STOP was needed; recorded here so
# the ambiguity is not silently re-discovered.
INJECTION_POOL_DIR = str(
    _REPO_ROOT
    / "results/campaign51_20260728/realistic_20260729/gate_b_20260730"
    / "injection_pool_mix200k_20260728"
)
# The banked post-fix-baseline production reference (2026-08-19), venue iiib.
BANKED_CSV_PATH = str(
    _REPO_ROOT / "results/prod2d_closure_20260818/postfix_baseline/iiib/event_likelihoods.csv"
)
PACKAGE_SRC = str(_REPO_ROOT / "darksiren_emri")
# The GLADE reduced catalogue GalaxyCatalogueHandler reads for production
# --evaluate. 2026-08-19 G-0 finding: the dev-machine copy was a stale Jul-1
# snapshot (no in-code md5 pin existed anywhere for this file, unlike
# CRB_CSV_MD5 above), which was the confirmed root cause of the first G-0
# FAIL (L_cat_no_bh/L_cat_with_bh and the h-only global-selection-sum
# precompute tables alpha_G_phi/D_tilde_phi all diverged from the banked
# reference). Replaced with the cluster copy of record (redshift columns
# regenerated 2026-07-27; mass columns identical) and pinned here, mirroring
# CRB_CSV_MD5, so this drift cannot silently recur.
REDUCED_CATALOGUE_PATH = str(
    _REPO_ROOT / "darksiren_emri/galaxy_catalogue/reduced_galaxy_catalogue.csv"
)
REDUCED_CATALOGUE_MD5 = "c52c13b5cab61f6b3f04bbe202550969"

# G-0 registered flags (postfix_baseline/iiib run_metadata_0.json, verbatim).
PRODUCTION_FLAGS: dict[str, str] = {
    "--normalization_mode": "absolute_marginal",
    "--host_z_kernel": "volume_deconv",
    "--selection_in_completion_numerator": "off",
    "--catalogue_mass_overlap": "production",
    "--completion_b_scale": "derived",
    "--pdet_dl_bins": "60",
    "--pdet_mass_bins": "40",
    "--pdet_estimator": "local_linear",
}
# --pdet_z_resolved is a store_true flag (True in run_metadata_0.json).

# G-0 registered probe grid + gate (prereg §4).
G0_PROBE_H: tuple[float, ...] = (0.675, 0.700)
G0_MIN_EVENTS = 3
G0_RTOL = 1.0e-6


# ── Config scaffold (future arms; G-0 exercises only the fidelity layer) ────


@dataclass(frozen=True)
class CorrespondenceConfig:
    """Registered-shape config for the full Option-B harness (prereg §1/§2).

    Only the fields G-0 actually reads are used today
    (``crb_reference_csv``/``injection_data_dir``/``pruned_catalogue_csv``/
    ``h_probe``); the rest are the D-C/D-D scale-and-budget knobs for the
    B-0/B-sigma/B-D2/E-DEN arms, carried here so the later build extends this
    dataclass rather than replacing it (prereg §5: "structured for the later
    arms").

    Attributes:
        n_events: Mirror-universe events per realization (D-C: 200).
        sigma_z_scale: Multiplicative dose on host photo-z sigma relative to
            the GLADE empirical baseline (B-sigma arm; 1.0 = B-0).
        seeds: Per-realization seed list (paired across arms per D-C).
        area_scale: Localization-area multiplier (E-DEN arm; 1.0 = baseline).
        d2_form: Placeholder for the B-D2 density-form toggle (P12,
            REPORTED-ONLY behind its own parity gate) -- ``"ratio_pdf"`` is
            production's form; ``"density"`` is the D-ii defect-arm form, not
            yet implemented (raises in :class:`MirrorUniverseGenerator`).
        crb_reference_csv: Pinned production CRB CSV (VT-D1 analog).
        injection_data_dir: Pinned injection pool (p_det grid input).
        pruned_catalogue_csv: The real GLADE candidate-structure catalogue
            (D-B).
        h_probe: The G-0 fidelity-pilot probe grid (prereg §4: 2 probe h).
    """

    n_events: int = 200
    sigma_z_scale: float = 1.0
    seeds: tuple[int, ...] = field(default_factory=tuple)
    area_scale: float = 1.0
    d2_form: Literal["ratio_pdf", "density"] = "ratio_pdf"
    crb_reference_csv: str = CRB_CSV_PATH
    injection_data_dir: str = INJECTION_POOL_DIR
    pruned_catalogue_csv: str = ""
    h_probe: tuple[float, ...] = G0_PROBE_H


class MirrorUniverseGenerator:
    """Registered-shape stub for the D-B real-catalogue mirror-universe draw.

    Not implemented in this build (G-0 exercises only the production-context
    fidelity layer, per the task scope). Per D-B, this will resample entire
    per-event Fisher rows (full covariance + detected parameters, incl. sky
    localization) from :data:`CRB_CSV_PATH`, SNR-weighted, and dose host
    z_obs by ``sigma_z_scale`` / localization area by ``area_scale``.
    """

    def __init__(self, config: CorrespondenceConfig) -> None:
        self.config = config

    def draw_realization(self, seed: int) -> None:
        """Draw one mirror-universe realization (NOT IMPLEMENTED).

        Args:
            seed: Realization seed.

        Raises:
            NotImplementedError: Always, in this build.
        """
        raise NotImplementedError(
            "MirrorUniverseGenerator is a G-0-scope stub; the D-B real-catalogue "
            "resampling draw is not built (harness prereg §1 D-B; see module docstring)."
        )


# ── Layer 1: production-wholesale evaluate() driver ──────────────────────────


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


def check_crb_pin(crb_path: str = CRB_CSV_PATH) -> bool:
    """Verify the pinned CRB CSV matches the venue_transfer.py V-T3 pin.

    Args:
        crb_path: Path to the CRB CSV under test.

    Returns:
        ``True`` iff the md5 matches :data:`CRB_CSV_MD5`.
    """
    return _md5_of_file(crb_path) == CRB_CSV_MD5


def check_reduced_catalogue_pin(catalogue_path: str = REDUCED_CATALOGUE_PATH) -> bool:
    """Verify the reduced GLADE catalogue matches the pinned cluster copy.

    Mirrors :func:`check_crb_pin`. Registered 2026-08-19 after a G-0 FAIL was
    traced to a stale local copy of this file (no prior in-code pin existed).

    Args:
        catalogue_path: Path to the reduced catalogue CSV under test.

    Returns:
        ``True`` iff the md5 matches :data:`REDUCED_CATALOGUE_MD5`.
    """
    return _md5_of_file(catalogue_path) == REDUCED_CATALOGUE_MD5


def _setup_wholesale_cwd(work_root: Path, crb_path: str, injection_dir: str) -> Path:
    """Build a sandboxed CWD with the production symlink layout.

    Mirrors ``results/prod2d_closure_20260818/probe_n0_local.py::_setup_cwd``.

    Args:
        work_root: Parent directory for the sandboxed CWD.
        crb_path: The pinned CRB CSV.
        injection_dir: The injection pool directory.

    Returns:
        The sandboxed CWD path.
    """
    cwd = work_root / "cwd"
    sims = cwd / "simulations"
    sims.mkdir(parents=True, exist_ok=True)
    _symlink(sims / "prepared_cramer_rao_bounds.csv", Path(crb_path))
    # true_cramer_rao_bounds.csv is read in BayesianStatistics.__init__ but
    # never consumed downstream of evaluate() -- harmless stand-in.
    _symlink(sims / "cramer_rao_bounds.csv", Path(crb_path))
    _symlink(cwd / "darksiren_emri", Path(PACKAGE_SRC))
    _symlink(sims / "injections", Path(injection_dir))
    return cwd


def _symlink(link: Path, target: Path) -> None:
    if link.is_symlink() or link.exists():
        if link.resolve() == target.resolve():
            return
        link.unlink()
    link.symlink_to(target)


def run_production_wholesale(
    work_root: Path,
    h_values: tuple[float, ...] = G0_PROBE_H,
    seed: int = 777010,
    crb_path: str = CRB_CSV_PATH,
    injection_dir: str = INJECTION_POOL_DIR,
) -> tuple[Path, float]:
    """Drive the real ``python -m darksiren_emri --evaluate`` entry point.

    Layer 1 of D-A: production-wholesale, no re-derivation. One context
    build serves every ``h_values`` probe (the ``--h_values`` fused-grid CLI
    path), so the elapsed wall time is the G-0 context-construction cost
    anchor (prereg D-D cost pilot).

    Args:
        work_root: Scratch directory (created if absent).
        h_values: Probe h-grid (prereg §4: 2 probe h).
        seed: CLI ``--seed`` (arbitrary for a deterministic evaluate() pass;
            recorded, not physics-relevant here).
        crb_path: The pinned CRB CSV.
        injection_dir: The injection pool directory.

    Returns:
        ``(diagnostics_csv_path, elapsed_seconds)``.

    Raises:
        RuntimeError: If the subprocess fails.
    """
    work_root.mkdir(parents=True, exist_ok=True)
    cwd = _setup_wholesale_cwd(work_root, crb_path, injection_dir)
    out_dir = work_root / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "darksiren_emri",
        str(out_dir),
        "--evaluate",
        "--h_values",
        ",".join(str(h) for h in h_values),
        "--seed",
        str(seed),
        "--pdet_z_resolved",
        "--log_level",
        "INFO",
    ]
    for flag, value in PRODUCTION_FLAGS.items():
        cmd.extend([flag, value])
    _LOGGER.info("running production wholesale: %s", " ".join(cmd))
    start = time.time()
    result = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    elapsed = time.time() - start
    if result.returncode != 0:
        raise RuntimeError(
            "production evaluate() subprocess failed:\n"
            f"STDOUT (tail):\n{result.stdout[-4000:]}\n"
            f"STDERR (tail):\n{result.stderr[-4000:]}"
        )
    csv_path = cwd / "simulations" / "diagnostics" / "event_likelihoods.csv"
    if not csv_path.is_file():
        raise RuntimeError(f"expected diagnostics CSV not found: {csv_path}")
    return csv_path, elapsed


# ── Layer 2: harness-side combine re-orchestration ───────────────────────────


def reassemble_combine_no_bh(
    df: pd.DataFrame,
) -> npt.NDArray[np.float64]:
    """Re-derive ``combined_no_bh`` from leaf production functions.

    Calls the REAL module-level
    :func:`~darksiren_emri.bayesian_inference.bayesian_statistics.path_a_mixture_objects`
    and
    :func:`~darksiren_emri.bayesian_inference.bayesian_statistics.path_a_completion_numerators`
    (not reimplemented) to reconstruct ``beta_G_phi``/``D_tilde_phi``/
    ``B_num_phi`` from the banked/produced Path-A columns, then assembles
    ``combined_no_bh = (beta_G_phi * L_cat_no_bh + B_num_phi) / D_tilde_phi``
    (``bayesian_statistics.py:5248-5251``). ``alpha_G_phi``/``r_Malm``/
    ``D_tilde_phi`` are h-only (identical across all rows at fixed h), so
    ``beta_G_phi = alpha_G_phi / r_Malm`` and
    ``beta_Gbar_phi = D_tilde_phi - alpha_G_phi`` invert exactly; the
    absolute normalization of ``sigma_phi``/``sigma_4d`` individually is not
    needed (only their ratio ``r_Malm`` enters), so ``sigma_phi = 1.0`` is an
    arbitrary but harmless choice inside :func:`path_a_mixture_objects`.

    Args:
        df: Rows at a SINGLE h (must have ``alpha_G_phi``, ``r_Malm``,
            ``D_tilde_phi`` constant across rows), with columns
            ``L_cat_no_bh``, ``B_num``, ``B_num_wbh``.

    Returns:
        Per-row ``combined_no_bh`` reconstructed via the harness's own
        combine assembly.

    Raises:
        ValueError: If the h-only columns are not constant across ``df``.
    """
    for col in ("alpha_G_phi", "r_Malm", "D_tilde_phi"):
        if df[col].nunique() != 1:
            raise ValueError(f"{col} is not constant across df -- pass rows at a single h")
    alpha_G_phi = float(df["alpha_G_phi"].iloc[0])
    r_Malm = float(df["r_Malm"].iloc[0])
    D_tilde_phi_bank = float(df["D_tilde_phi"].iloc[0])
    beta_G_phi = alpha_G_phi / r_Malm
    beta_Gbar_phi = D_tilde_phi_bank - alpha_G_phi
    mix = path_a_mixture_objects(
        beta_G_phi=beta_G_phi,
        beta_Gbar_phi=beta_Gbar_phi,
        sigma_phi=1.0,
        sigma_4d=r_Malm,
    )
    B_num_phi, _B_num_wbh_phi, _b_scale = path_a_completion_numerators(
        df["B_num"].to_numpy(dtype=np.float64),
        df["B_num_wbh"].to_numpy(dtype=np.float64),
        beta_Gbar_phi,
        beta_Gbar=float("nan"),
        mode="derived",
    )
    combined: npt.NDArray[np.float64] = (
        beta_G_phi * df["L_cat_no_bh"].to_numpy(dtype=np.float64) + B_num_phi
    ) / mix["D_tilde_phi"]
    return combined


def _max_rel_diff(a: npt.NDArray[np.float64], b: npt.NDArray[np.float64]) -> float:
    """Max relative difference, both-zero rows scored as 0.0 (banked-CSV convention).

    Args:
        a: Reference array.
        b: Candidate array.

    Returns:
        Max relative difference.
    """
    denom = a.copy()
    denom[denom == 0.0] = 1.0
    rel = np.abs(a - b) / np.abs(denom)
    both_zero = (a == 0.0) & (b == 0.0)
    rel[both_zero] = 0.0
    return float(rel.max()) if rel.size else 0.0


@dataclass
class G0StageResult:
    """Per-h, per-stage G-0 fidelity numbers.

    Attributes:
        h: Probe h.
        n_events: Number of events compared at this h.
        max_rel_L_cat_no_bh: Layer-1 (wholesale) vs banked, catalogue leg.
        max_rel_B_num: Layer-1 (wholesale) vs banked, completion numerator.
        max_rel_combined_no_bh_wholesale: Layer-1 (wholesale) vs banked,
            combined_no_bh.
        max_rel_combined_no_bh_reassembled: Layer-2 (harness combine
            re-orchestration, applied to the WHOLESALE run's own L_cat/B_num)
            vs the wholesale run's own combined_no_bh.
    """

    h: float
    n_events: int
    max_rel_L_cat_no_bh: float
    max_rel_B_num: float
    max_rel_combined_no_bh_wholesale: float
    max_rel_combined_no_bh_reassembled: float


@dataclass
class G0Result:
    """Full G-0 fidelity-pilot result.

    Attributes:
        stages: Per-h stage results.
        n_events_evaluated: Distinct events compared (>= :data:`G0_MIN_EVENTS`
            required to pass).
        context_build_seconds: Wall time of the single production context
            build serving all probe h (cost-pilot anchor, prereg D-D).
        crb_pin_ok: Whether the CRB CSV matched the V-T3 pin.
        catalogue_pin_ok: Whether the reduced GLADE catalogue matched
            :data:`REDUCED_CATALOGUE_MD5`.
        verdict: ``"PASS"``, ``"FAIL"``, or ``"STOP"`` (an input pin
            mismatch — the run was not attempted, mirroring
            ``venue_transfer.py``'s V-T3 pin-integrity STOP).
    """

    stages: list[G0StageResult]
    n_events_evaluated: int
    context_build_seconds: float
    crb_pin_ok: bool
    catalogue_pin_ok: bool
    verdict: str


def run_g0_fidelity_pilot(
    work_root: Path,
    banked_csv: str = BANKED_CSV_PATH,
    h_values: tuple[float, ...] = G0_PROBE_H,
    crb_path: str = CRB_CSV_PATH,
    injection_dir: str = INJECTION_POOL_DIR,
    catalogue_path: str = REDUCED_CATALOGUE_PATH,
    seed: int = 777010,
) -> G0Result:
    """Run gate G-0 (prereg §4): the fidelity pilot.

    Checks the V-T3-style input pins FIRST (:func:`check_crb_pin`,
    :func:`check_reduced_catalogue_pin`) and STOPs (returns a ``"STOP"``
    verdict without running the expensive wholesale evaluate()) on any
    mismatch — the 2026-08-19 G-0 FAIL was traced to exactly this class of
    silent input drift (a stale local reduced-catalogue copy with no pin to
    catch it), so this gate is now enforced up front rather than discovered
    downstream in a 4-5 minute wholesale run.

    Drives the production-wholesale layer once (serving every probe h),
    compares its per-event ``L_cat_no_bh``/``B_num``/``combined_no_bh``
    against the banked post-fix-baseline reference, and separately verifies
    the harness's own combine re-orchestration (layer 2) against the SAME
    wholesale run's ``combined_no_bh`` (an identity check that must hold to
    float precision, since layer 2 consumes layer 1's own L_cat/B_num — this
    isolates the combine formula from any wholesale-vs-banked environment
    drift).

    Args:
        work_root: Scratch directory for the sandboxed evaluate() run.
        banked_csv: The banked production reference CSV.
        h_values: Probe h-grid.
        crb_path: The pinned CRB CSV.
        injection_dir: The injection pool directory.
        catalogue_path: The pinned reduced GLADE catalogue.
        seed: CLI ``--seed`` for the wholesale run.

    Returns:
        The :class:`G0Result`.
    """
    crb_pin_ok = check_crb_pin(crb_path)
    catalogue_pin_ok = check_reduced_catalogue_pin(catalogue_path)
    if not (crb_pin_ok and catalogue_pin_ok):
        _LOGGER.error(
            "G-0 STOP: input pin mismatch (crb_pin_ok=%s, catalogue_pin_ok=%s) — "
            "wholesale evaluate() NOT run.",
            crb_pin_ok,
            catalogue_pin_ok,
        )
        return G0Result(
            stages=[],
            n_events_evaluated=0,
            context_build_seconds=0.0,
            crb_pin_ok=crb_pin_ok,
            catalogue_pin_ok=catalogue_pin_ok,
            verdict="STOP",
        )
    csv_path, elapsed = run_production_wholesale(
        work_root, h_values=h_values, seed=seed, crb_path=crb_path, injection_dir=injection_dir
    )
    produced = pd.read_csv(csv_path)
    banked = pd.read_csv(banked_csv)

    stages: list[G0StageResult] = []
    min_events = 10**9
    for h in h_values:
        prod_h = produced[np.isclose(produced["h"], h)].sort_values("event_idx")
        bank_h = banked[np.isclose(banked["h"], h)].sort_values("event_idx")
        merged = bank_h.merge(prod_h, on="event_idx", suffixes=("_bank", "_prod"), how="inner")
        n = len(merged)
        min_events = min(min_events, n)

        rel_L_cat = _max_rel_diff(
            merged["L_cat_no_bh_bank"].to_numpy(dtype=np.float64),
            merged["L_cat_no_bh_prod"].to_numpy(dtype=np.float64),
        )
        rel_B_num = _max_rel_diff(
            merged["B_num_bank"].to_numpy(dtype=np.float64),
            merged["B_num_prod"].to_numpy(dtype=np.float64),
        )
        rel_combined = _max_rel_diff(
            merged["combined_no_bh_bank"].to_numpy(dtype=np.float64),
            merged["combined_no_bh_prod"].to_numpy(dtype=np.float64),
        )

        # Layer 2: reassemble from the WHOLESALE run's own L_cat/B_num/columns.
        prod_cols = prod_h[
            [
                "event_idx",
                "alpha_G_phi",
                "r_Malm",
                "D_tilde_phi",
                "L_cat_no_bh",
                "B_num",
                "B_num_wbh",
                "combined_no_bh",
            ]
        ].reset_index(drop=True)
        reassembled = reassemble_combine_no_bh(prod_cols)
        rel_reassembled = _max_rel_diff(
            prod_cols["combined_no_bh"].to_numpy(dtype=np.float64), reassembled
        )

        stages.append(
            G0StageResult(
                h=h,
                n_events=n,
                max_rel_L_cat_no_bh=rel_L_cat,
                max_rel_B_num=rel_B_num,
                max_rel_combined_no_bh_wholesale=rel_combined,
                max_rel_combined_no_bh_reassembled=rel_reassembled,
            )
        )

    overall_max = max(
        max(
            s.max_rel_L_cat_no_bh,
            s.max_rel_B_num,
            s.max_rel_combined_no_bh_wholesale,
            s.max_rel_combined_no_bh_reassembled,
        )
        for s in stages
    )
    verdict = (
        "PASS"
        if (
            crb_pin_ok
            and catalogue_pin_ok
            and min_events >= G0_MIN_EVENTS
            and overall_max <= G0_RTOL
        )
        else "FAIL"
    )
    return G0Result(
        stages=stages,
        n_events_evaluated=min_events,
        context_build_seconds=elapsed,
        crb_pin_ok=crb_pin_ok,
        catalogue_pin_ok=catalogue_pin_ok,
        verdict=verdict,
    )


def _cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--work-root",
        default="/tmp/correspondence_1d_g0",
        help="Scratch directory for the sandboxed evaluate() run.",
    )
    parser.add_argument("--seed", type=int, default=777010)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    result = run_g0_fidelity_pilot(Path(args.work_root), seed=args.seed)
    print(json.dumps({"verdict": result.verdict}, indent=2))
    for s in result.stages:
        print(
            f"h={s.h}: n={s.n_events} "
            f"L_cat_no_bh={s.max_rel_L_cat_no_bh:.3e} "
            f"B_num={s.max_rel_B_num:.3e} "
            f"combined_no_bh(wholesale-vs-bank)={s.max_rel_combined_no_bh_wholesale:.3e} "
            f"combined_no_bh(harness-reassembly)={s.max_rel_combined_no_bh_reassembled:.3e}"
        )
    print(f"context_build_seconds={result.context_build_seconds:.1f}")
    print(f"crb_pin_ok={result.crb_pin_ok}")
    print(f"catalogue_pin_ok={result.catalogue_pin_ok}")
    return 0 if result.verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(_cli())
