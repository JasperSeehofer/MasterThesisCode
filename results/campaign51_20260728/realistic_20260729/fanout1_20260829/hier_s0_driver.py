#!/usr/bin/env python3
"""[HIER] Stage-0 driver -- S0-A / S0-R / S0-C.

Registration: ``PREREGISTRATION_HIER_HTHETA_20260826.md`` (binding) --
section 1.2 (theta identity, line 40), section 2.1 (Stage-0 arms, lines
139-176), section 4.1 (score definition, lines 384-410), section 5.1
(invariants), GATE ENG / GATE PARITY / GATE D3 (section 3), and the
amendment log PA-HIER-1..30.

**BUILD ONLY -- charter node B1.1, launched under rows #222/#223.** Rule 2
(verifier independence): this file's author may only SMOKE-TEST it; the
registered measurement (the real 4-seed / 5-node grid) must be RUN by a
different agent. The ``--smoke`` flag exists for exactly that boundary --
it truncates events and node count so the *code path* is exercised without
banking a registered number.

Disclosures carried forward from the amendment log (do not re-litigate,
re-check by reading the cited amendment before changing anything below):

* **S0-R is a disclosed NULL INSTRUMENT (PA-HIER-3, PA-HIER-22).**
  ``realize_observed_catalogue(sigma_scale=1.5)`` round-trips the quoted
  ``z_error`` column verbatim and feeds the SAME realized catalogue to both
  the generator and the estimator (``host_pool_for_sigma_scale`` returns one
  shared ``GalaxyCatalogueHandler``) -- ``sigma_kernel == sigma_realized``
  identically at every dose. Truth-theta after the call is still (0, 1), not
  (0, 1.5); no z-kernel misspecification is injected. A real s-axis positive
  control needs new code (C3, NEEDS-CODE, unbuilt) that this driver does not
  attempt to build.
* **PA-HIER-28 item 5 = FALLBACK (author ruling, 2026-08-28): D7's early
  exit is DISARMED and Stage 0 is re-scoped to S0-A + S0-C ONLY.** This
  driver still implements S0-R (per this build task's explicit instruction,
  and because the code is cheap and may be useful if C3 ever lands), but its
  verdict function never emits B0-R/B0-R'/LEVER-DEAD-AT-N -- it reports
  NULL-INSTRUMENT and cites PA-HIER-3/22/28. Treat any S0-R number this
  driver produces as a disclosed diagnostic, not a registered verdict.
* **All [HIER] verdicts are capped REPORTED-ONLY (PA-HIER-28 item 9 =
  AFFORDABLE).** No band in this driver's output may be read as CALIBRATED.
* GATE SEQ (section 3.7): no ``sbatch`` for any [HIER] stage until
  [P3-MKER] stage-1 is banked. This driver runs LOCAL processes only and
  never touches ``sbatch`` -- it is orthogonal to GATE SEQ, but a future
  cluster port of this driver is NOT authorized by this build.

Venue (bc / b0i, PA-HIER-27/28 item 1 RATIFIED): ``host_mode=
"catalogue_selected"``, kwargs copied EXACTLY from
``results/campaign51_20260728/realistic_20260729/p3_b0_identity_test.py``'s
``_run_arm_seed(venue="b0i")`` (module docstring lines ~883-895, body
~915-935) and ``ARM_FLAGS["bc"]`` (line ~997): ``catalogue_numerator_
survival="off"``, ``catalogue_global_selection="phi"``,
``selection_in_completion_numerator="fused"``. Banked bc seeds 900101..
900112 live under ``results/campaign51_20260728/realistic_20260729/
p3_b0_work/bc_<seed>_work/``.
"""

import argparse
import json
import math
import multiprocessing as mp
import os
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]
REALISTIC_DIR = REPO_ROOT / "results/campaign51_20260728/realistic_20260729"
BC_WORK_ROOT = REALISTIC_DIR / "p3_b0_work"
REGISTRATION = REALISTIC_DIR / "PREREGISTRATION_HIER_HTHETA_20260826.md"

import darksiren_emri.validation.correspondence_1d as c1d  # noqa: E402
from darksiren_emri.bayesian_inference.bayesian_statistics import (  # noqa: E402
    _completeness_at_host_nodes,
    _host_pixels,
)
from darksiren_emri.galaxy_catalogue.handler import InternalCatalogColumns  # noqa: E402
from darksiren_emri.physical_relations import comoving_volume_element  # noqa: E402
from darksiren_emri.validation.correspondence_1d import H_GRID_41, H_TRUE  # noqa: E402

# ── Registered constants ──────────────────────────────────────────────────

H_GEN: float = H_TRUE  # 0.73 -- the mirror-universe truth h (constants.H)
# Invariant #2 (registration §5.1): "h grid = H_GRID_41 verbatim, h_bounds =
# (0.50, 0.86) pin". This is ALSO the pin p3_b0_identity_test.py's banked bc
# CSVs were produced under (H_GRID_FULL = H_WING_LOW | H_GRID_41, whose
# min/max are exactly 0.50/0.86) -- P3-HGRID (rows #182-#184) proved a
# single-h caller reproducing a full-grid run's L_cat "must pass
# h_bounds=(min(grid), max(grid)) explicitly (proven bit-exact vs the banked
# b0i CSVs)". Passing it here is what makes GATE PARITY a real, checkable
# claim rather than an apples-to-oranges comparison.
H_BOUNDS: tuple[float, float] = (0.50, 0.86)
assert min(H_GRID_41) >= H_BOUNDS[0] and max(H_GRID_41) <= H_BOUNDS[1]

# bc/b0i venue flags, copied verbatim from p3_b0_identity_test.py:
#   ARM_FLAGS["bc"] = {"catalogue_numerator_survival": "off",
#                       "catalogue_global_selection": "phi"}
#   _run_arm_seed(..., completion_cell="fused")  (its own default)
BC_CATALOGUE_NUMERATOR_SURVIVAL = "off"
BC_CATALOGUE_GLOBAL_SELECTION = "phi"
BC_COMPLETION_CELL = "fused"
BC_EVENT_MEASURE = "ratio"  # c1d.ARM_EVENT_MEASURE.get("b0i", "ratio")

# First 4 of the banked bc span 900101-900112 (12 = first 12 of
# c1d.ARM_SEEDS["b0i"], p3_b0_identity_test.py BSEL_SEEDS).
DEFAULT_BC_SEEDS: tuple[int, ...] = (900101, 900102, 900103, 900104)

# FT venue flags (B4.2 KW-Q1, SYNTHESIS_DOCKET_1_20260829.md sec 2 B4;
# CLAIM_IMPOSTOR_DRAG_20260829.md sec 1.3), copied EXACTLY from
# p3_twin_test.py's ``_run_bsel_seed(seed, "phi", ..., completion_cell=
# "fused")`` (fusedarm stage, ``--survival phi --completion-cell fused --tag
# ft``, p3_twin_test.py:186-224): ``sigma_z_scale, area_scale =
# c1d.ARM_SPECS["bsel"]`` (both 1.0), ``host_mode="population_selected"``
# (c1d.ARM_HOST_MODE["bsel"]), ``completeness_obj, phi_survival_table =
# c1d.build_bsel_selection_objects()`` (no ``h_true`` kwarg -- default is
# c1d.H_TRUE, identical to H_GEN), ``catalogue_numerator_survival="phi"``,
# ``selection_in_completion_numerator="fused"``,
# ``completion_event_measure="ratio"`` (c1d.ARM_EVENT_MEASURE["bsel"]).
# Unlike build_bc_venue, ``_run_bsel_seed`` never calls
# ``c1d._verify_rate_weight_parity()`` -- disclosed, not added (this driver
# copies the arm EXACTLY, including that omission). ``catalogue_global_
# selection`` is likewise never passed by ``_run_bsel_seed`` -- left at
# run_mirror_seed_inprocess's own "auto" default, which resolves to "phi"
# under normalization_mode="absolute_marginal" (production default),
# functionally identical to BC_CATALOGUE_GLOBAL_SELECTION="phi" but kept
# implicit here to match the copied call site literally.
FT_CATALOGUE_NUMERATOR_SURVIVAL = "phi"
FT_COMPLETION_CELL = "fused"
FT_EVENT_MEASURE = "ratio"  # c1d.ARM_EVENT_MEASURE["bsel"]
FT_SIGMA_Z_SCALE: float = 1.0  # c1d.ARM_SPECS["bsel"][0]
FT_AREA_SCALE: float = 1.0  # c1d.ARM_SPECS["bsel"][1]

# --config choices (b0i = pre-existing hardcoded bc venue, unchanged default;
# ft = KW-Q1's B-SEL/phi/fused venue, new).
CONFIG_CHOICES: tuple[str, ...] = ("b0i", "ft")
# --theta-sites choices: exactly evaluate()'s own validated set
# (bayesian_statistics.py's theta_sites guard, "all"/"2.1"/"2.2"/"2.3" only --
# no other string is accepted by the estimator).
THETA_SITES_CHOICES: tuple[str, ...] = ("all", "2.1", "2.2", "2.3")
# --smear choices: "auto" reproduces this driver's ORIGINAL, pre-P1 dispatch
# (smear_global_selection = theta_engaged, unconditionally on theta_sites) --
# see run_theta_node's docstring for the byte-identical-by-default argument.
SMEAR_CHOICES: tuple[str, ...] = ("auto", "on", "off")

# The registered 5-node theta-cross (prereg §2.1 S0-A row / §4.1):
# {(0,1), (+-0.02, 1), (0, 1/sqrt(2)), (0, sqrt(2))} at h = 0.73 only.
THETA_NODES: dict[str, tuple[float, float]] = {
    "truth": (0.0, 1.0),
    "b_plus": (0.02, 1.0),
    "b_minus": (-0.02, 1.0),
    "s_plus": (0.0, math.sqrt(2.0)),
    "s_minus": (0.0, 1.0 / math.sqrt(2.0)),
}
NODE_ORDER: tuple[str, ...] = ("truth", "b_plus", "b_minus", "s_plus", "s_minus")

# S0-R's registered dose (prereg §2.1; DISCLOSED NULL per PA-HIER-3/22 above).
S0_R_SIGMA_SCALE = 1.5

# Band thresholds (prereg §4.1). The 3.0 anchor is NOT chosen by this driver
# -- it is `.claude/skills/research-cycle/SKILL.md`'s own registered
# coherent-class-displacement threshold, cited verbatim in the prereg.
Z_THRESHOLD = 3.0
# GATE ENG (prereg §3.4): >=10% of scored events move by >=1e-6 relative in
# per-event ln L versus truth-theta.
ENG_REL_THRESHOLD = 1e-6
ENG_EVENT_FRACTION = 0.10
# GATE PARITY tolerance for this driver's own truth-node reproduction check
# (distinct from the registration's own GATE PARITY, prereg §3.3, which is
# about correspondence_1d.py:1173's separate stale-comment claim and is NOT
# re-litigated here). We assert exact byte equality is the TARGET (per the
# P3-HGRID note's "proven bit-exact" language); anything not exactly 0 is
# reported at whatever rtol it actually lands at, never silently widened.
PARITY_TARGET_EXACT = True
PARITY_FALLBACK_RTOL = 1e-9

DIAG_VALUE_COLUMNS = ("combined_no_bh", "combined_with_bh")


# ── bc/b0i venue construction (copied exactly, see module docstring) ──────


def build_bc_venue(
    work_root: Path, seed: int, sigma_z_scale: float = 1.0
) -> tuple[pd.DataFrame, Any]:
    """Build one b0i-venue mirror realization.

    Kwargs copied EXACTLY from ``p3_b0_identity_test.py``'s
    ``_run_arm_seed(venue="b0i")`` (as of this driver's authoring, 2026-08-29):
    ``c1d._verify_rate_weight_parity()`` before any draw, ``completeness_obj,
    phi_survival_table = c1d.build_bsel_selection_objects(h_true=H_GEN)``,
    then ``gen.draw_realization(seed, host_pool=host_pool,
    host_mode="catalogue_selected", completeness=completeness_obj,
    phi_survival_table=phi_survival_table)``.

    ``sigma_z_scale`` is the S0-A/S0-R fork: S0-A passes 1.0 (the banked bc
    dose, truth-theta=(0,1)); S0-R passes 1.5 (prereg §2.1's registered
    dose -- disclosed NULL per PA-HIER-3/22, module docstring). Note
    ``CorrespondenceConfig.sigma_z_scale`` is a SEPARATE, unrelated field
    (dead for the ``catalogue_selected`` draw path -- only ``n_events``/
    ``area_scale`` are read off it, per the class docstring); the dose that
    matters is ``host_pool_for_sigma_scale``'s own ``sigma_z_scale`` kwarg,
    which is what this function forwards.

    Returns:
        ``(events, handler)`` -- the realization's synthetic CRB rows and the
        ``GalaxyCatalogueHandler`` built from the (possibly re-realized)
        catalogue, ready for ``run_mirror_seed_inprocess``.
    """
    cfg = c1d.CorrespondenceConfig()  # n_events=200, area_scale=1.0 (b0i ARM_SPECS)
    gen = c1d.MirrorUniverseGenerator(cfg)
    host_pool, _observed_path, handler = gen.host_pool_for_sigma_scale(
        work_root / "catalogue", seed, sigma_z_scale=sigma_z_scale
    )
    c1d._verify_rate_weight_parity()
    completeness_obj, phi_survival_table = c1d.build_bsel_selection_objects(h_true=H_GEN)
    events = gen.draw_realization(
        seed,
        host_pool=host_pool,
        host_mode="catalogue_selected",
        completeness=completeness_obj,
        phi_survival_table=phi_survival_table,
    )
    return events, handler


def build_ft_venue(
    work_root: Path, seed: int, sigma_z_scale: float = 1.0
) -> tuple[pd.DataFrame, Any]:
    """Build one FT-venue (bsel/phi/fused) mirror realization -- KW-Q1 (B4.2).

    Kwargs copied EXACTLY from ``p3_twin_test.py``'s ``_run_bsel_seed(seed,
    "phi", ..., completion_cell="fused")`` (fusedarm stage, as of this
    driver's authoring, 2026-08-29; ``p3_twin_test.py:186-201``):
    ``sigma_z_scale, area_scale = c1d.ARM_SPECS["bsel"]`` (both 1.0),
    ``cfg = c1d.CorrespondenceConfig(sigma_z_scale=sigma_z_scale,
    area_scale=area_scale)``, ``completeness_obj, phi_survival_table =
    c1d.build_bsel_selection_objects()`` (no ``h_true`` kwarg -- default is
    ``c1d.H_TRUE``, identical to :data:`H_GEN`), ``host_mode=
    "population_selected"``. Unlike :func:`build_bc_venue`, this arm does
    NOT call ``c1d._verify_rate_weight_parity()`` -- disclosed, not added
    (this function copies the arm EXACTLY, including that omission).

    ``sigma_z_scale`` is accepted for signature parity with
    :func:`build_bc_venue` (both are threaded through ``_build_venue``
    below) but the FT/bsel arm's own ``ARM_SPECS`` pins it to 1.0 -- KW-Q1
    never doses this axis (S0-R's ``sigma_z_scale=1.5`` fork has no
    FT-config analogue). A caller passing anything else is a build error,
    raised here rather than silently ignored.

    Returns:
        ``(events, handler)`` -- as :func:`build_bc_venue`.
    """
    if sigma_z_scale != FT_SIGMA_Z_SCALE:
        raise ValueError(
            f"build_ft_venue: sigma_z_scale must be {FT_SIGMA_Z_SCALE!r} "
            f"(c1d.ARM_SPECS['bsel'] pin), got {sigma_z_scale!r} -- the FT "
            "config has no S0-R-style dosed analogue."
        )
    cfg = c1d.CorrespondenceConfig(sigma_z_scale=FT_SIGMA_Z_SCALE, area_scale=FT_AREA_SCALE)
    gen = c1d.MirrorUniverseGenerator(cfg)
    host_pool, _observed_path, handler = gen.host_pool_for_sigma_scale(
        work_root / "catalogue", seed, sigma_z_scale=FT_SIGMA_Z_SCALE
    )
    completeness_obj, phi_survival_table = c1d.build_bsel_selection_objects()
    events = gen.draw_realization(
        seed,
        host_pool=host_pool,
        host_mode="population_selected",
        completeness=completeness_obj,
        phi_survival_table=phi_survival_table,
    )
    return events, handler


def _build_venue(
    config: str, work_root: Path, seed: int, sigma_z_scale: float
) -> tuple[pd.DataFrame, Any]:
    """Dispatch to :func:`build_bc_venue` (``config="b0i"``, unchanged default)
    or :func:`build_ft_venue` (``config="ft"``, KW-Q1/B4.2, new)."""
    if config == "b0i":
        return build_bc_venue(work_root, seed, sigma_z_scale=sigma_z_scale)
    if config == "ft":
        return build_ft_venue(work_root, seed, sigma_z_scale=sigma_z_scale)
    raise ValueError(f"config must be one of {CONFIG_CHOICES}, got {config!r}")


def _resolve_smear(theta_engaged: bool, theta_sites: str, smear: str) -> bool:
    """Resolve ``--smear {auto,on,off}`` to the actual ``smear_global_selection``
    bool passed to ``evaluate()``.

    ``"auto"`` reproduces this driver's ORIGINAL (pre-P1) dispatch exactly
    when ``theta_sites == "all"`` (the only value the driver ever passed
    before P1): ``smear = theta_engaged`` -- BYTE-IDENTICAL default
    behaviour. For ``theta_sites in ("2.1", "2.2")`` (P1's whole point),
    ``"auto"`` does NOT force smearing -- those sites never consume the
    smeared table (P1's own source-read finding, SYNTHESIS_DOCKET_1_
    20260829.md sec 2 B1 P1), so the unsmeared path is both sufficient and
    ~18x cheaper. ``"on"``/``"off"`` force the flag regardless of engagement
    or sites (the caller's responsibility to keep this consistent with
    evaluate()'s own guard -- see the raise below).
    """
    if smear == "auto":
        return theta_engaged and theta_sites in ("all", "2.3")
    if smear == "on":
        return True
    if smear == "off":
        return False
    raise ValueError(f"smear must be one of {SMEAR_CHOICES}, got {smear!r}")


def _node_dir_suffix(
    theta_sites: str,
    smear: str,
    config: str,
    theta_phi_divisor: str = "off",
    sky_cone_k: float = 1.5,
    catalogue_leg_1d_mass_aware: str = "off",
    theta_zwindow: str = "off",
    z_window_k: float = 1.0,
) -> str:
    """Node output-directory suffix encoding the P1/KW-Q1/T1.1/T1.3-zwin/T2.3
    variant, so a non-default run never overwrites another variant's (or the
    default's) banked node outputs. Byte-identical default (``theta_sites=
    "all"``, ``smear="auto"``, ``config="b0i"``, ``theta_phi_divisor="off"``,
    ``sky_cone_k=1.5``, ``catalogue_leg_1d_mass_aware="off"``,
    ``theta_zwindow="off"``, ``z_window_k=1.0``) -> empty suffix -> the
    ORIGINAL ``node_<name>`` paths, unchanged.

    T1.2 (row #255 tree 2 node T1.2, driver-gap fix for T1.1's site 2.3phi
    instrument, PHYSICS_CHANGE_THETA_DIVISOR_20260830.md §2.2/§2.5):
    ``theta_phi_divisor="on"`` appends ``_divisor``; a non-default
    ``sky_cone_k`` (anything != 1.5) appends ``_conek<value>`` (``:g``
    formatted, so ``2.0`` -> ``conek2``, ``2.25`` -> ``conek2.25``).

    T1.3-zwin (row #255 tree 2 node T1.3-zwin, PHYSICS_CHANGE_THETA_ZWINDOW_
    20260830.md §2.2): ``theta_zwindow="on"`` appends ``_zwin``; a non-default
    ``z_window_k`` (anything != 1.0) appends ``_zk<value>`` (``:g`` formatted,
    so ``4.0`` -> ``zk4``).

    T2.3 (row #255 tree 2 node T2.3, PHYSICS_CHANGE_MASS_AWARE_1D_LEG_
    20260830.md §2): ``catalogue_leg_1d_mass_aware="on"`` appends ``_ma1d``.
    """
    parts: list[str] = []
    if config != "b0i":
        parts.append(config)
    if theta_sites != "all":
        parts.append(f"sites{theta_sites}")
    if smear != "auto":
        parts.append("smearon" if smear == "on" else "nosmear")
    if theta_phi_divisor != "off":
        parts.append("divisor")
    if sky_cone_k != 1.5:
        parts.append(f"conek{sky_cone_k:g}")
    if theta_zwindow != "off":
        parts.append("zwin")
    if z_window_k != 1.0:
        parts.append(f"zk{z_window_k:g}")
    if catalogue_leg_1d_mass_aware != "off":
        parts.append("ma1d")
    return ("_" + "_".join(parts)) if parts else ""


def run_theta_node(
    work_root: Path,
    events: pd.DataFrame,
    seed: int,
    handler: Any,
    theta_b: float,
    theta_s: float,
    h_values: tuple[float, ...] = (H_GEN,),
    theta_sites: str = "all",
    smear: str = "auto",
    config: str = "b0i",
    candidate_dump_dir: str | None = None,
    theta_phi_divisor: str = "off",
    sky_cone_k: float = 1.5,
    catalogue_leg_1d_mass_aware: str = "off",
    theta_zwindow: str = "off",
    z_window_k: float = 1.0,
) -> tuple[Path, float]:
    """Evaluate one theta node (prereg §2.1 S0-A/S0-R row row; KW-Q1 reuses this
    for the FT config at h_values=(0.725, 0.735)).

    ``smear`` resolves via :func:`_resolve_smear` (see its docstring for the
    byte-identical-by-default argument -- the ORIGINAL unconditional
    ``smear_global_selection = theta_engaged`` dispatch is exactly
    ``theta_sites="all", smear="auto"``, this function's defaults). This
    keeps the truth node (identity theta) on the byte-identical unsmeared
    path (GATE PARITY) while off-truth nodes at the default P1-untested
    config engage the registered site-2.3 kernel (GATE ENG) -- an explicit
    driver-level decision, disclosed in ``B1_1_HIER_BUILD_NOTE.md`` section
    "ambiguities resolved", extending GATE D3(a)'s stated principle (force
    the branch on engagement, never leave it to a separately-set flag) to
    this driver's own dispatch.

    ``config`` selects the venue's fixed flags (:data:`BC_*` for "b0i",
    :data:`FT_*` for "ft", B4.2 KW-Q1) -- see :data:`CONFIG_CHOICES`.

    ``candidate_dump_dir`` (T2.2, row #255 tree 2 node T2.2; A10 =
    instrumentation guard, not a physics gate) is forwarded verbatim to
    ``run_mirror_seed_inprocess``/``BayesianStatistics.evaluate()``. ``None``
    (default) is byte-identical (GATE BI); the CLI's ``--candidate-dump``
    arms it with a per-node subdirectory (see :func:`main`).

    ``theta_phi_divisor``, ``sky_cone_k`` (T1.2, row #255 tree 2 node T1.2,
    the T1.1 driver-gap fix; PHYSICS_CHANGE_THETA_DIVISOR_20260830.md
    §2.2/§2.5) are forwarded verbatim to ``run_mirror_seed_inprocess``/
    ``BayesianStatistics.evaluate()``. Defaults (``"off"``, ``1.5``) are
    byte-identical (GATE BI). Forwarded unconditionally to every node,
    including the truth node (theta=(0,1)) -- the truth node is a no-op
    under the divisor per GATE T-ID (the transform is theta-consistent, so
    it is the identity at theta=(0,1)), so unconditional forwarding does not
    disturb GATE PARITY.

    ``catalogue_leg_1d_mass_aware`` (T2.3, row #255 tree 2 node T2.3;
    PHYSICS_CHANGE_MASS_AWARE_1D_LEG_20260830.md §2) is forwarded verbatim
    to ``run_mirror_seed_inprocess``/``BayesianStatistics.evaluate()``.
    Default ``"off"`` is byte-identical (GATE BI). Forwarded unconditionally
    to every node -- ``evaluate()``'s own setup guard raises if the
    resolved ``catalogue_numerator_survival``/``catalogue_global_selection``
    are not both ``"phi"`` or if ``theta_phi_divisor`` is engaged.

    ``theta_zwindow``, ``z_window_k`` (row #255 tree 2 node T1.3-zwin;
    PHYSICS_CHANGE_THETA_ZWINDOW_20260830.md §2.2) are forwarded verbatim to
    ``run_mirror_seed_inprocess``/``BayesianStatistics.evaluate()``. Defaults
    (``"off"``, ``1.0``) are byte-identical (GATE BI). Forwarded
    unconditionally to every node, including the truth node -- the truth
    node is a no-op under the flag per GATE T-ID (the theta-transformed
    window is the identity at theta=(0,1)).
    """
    theta_engaged = theta_b != 0.0 or theta_s != 1.0
    smear_flag = _resolve_smear(theta_engaged, theta_sites, smear)
    if theta_engaged and theta_sites in ("all", "2.3") and not smear_flag:
        # Mirrors evaluate()'s own guard (bayesian_statistics.py's theta_sites
        # validation) -- raised HERE, before the (expensive) evaluate() call,
        # with a driver-level message; --smear off + --theta-sites all/2.3 is
        # also refused earlier, at CLI parse time in main() (clearer still,
        # since it does not require a theta-engaged node to be selected to
        # surface) -- this is the library-level backstop for any other
        # caller (e.g. kwq1_score.py's GATE PARITY re-evaluation) that
        # invokes run_theta_node directly.
        raise ValueError(
            f"theta engaged (b={theta_b}, s={theta_s}) with theta_sites={theta_sites!r} "
            "requires smear_global_selection=True (evaluate()'s own guard) -- "
            "pass theta_sites='2.1' or '2.2' together with --smear off"
        )
    common_kwargs: dict[str, Any] = dict(
        candidate_dump_dir=candidate_dump_dir,
        h_values=h_values,
        h_bounds=H_BOUNDS,
        theta_b=theta_b,
        theta_s=theta_s,
        theta_sites=theta_sites,
        smear_global_selection=smear_flag,
        theta_phi_divisor=theta_phi_divisor,
        sky_cone_k=sky_cone_k,
        catalogue_leg_1d_mass_aware=catalogue_leg_1d_mass_aware,
        theta_zwindow=theta_zwindow,
        z_window_k=z_window_k,
    )
    # [P3-2D] the with-BH catalogue-leg twin flipped to production default
    # "mz_sel"/"eff" (row #223 standing grant, charter node B7.3;
    # PHYSICS_CHANGE_2D_TWIN_ADOPTION_20260829.md §6.1 Class-B site B3). This
    # driver's registered CoR-P form is the PRE-adoption estimator (cf.
    # cluster/wave2_c1_s0b_TEMPLATE.sbatch:162), so every call site below
    # pins the counterfactual explicitly to keep the banked Stage-0/KW-Q1
    # comparands byte-identical.
    cat_num_surv_2d_kwargs: dict[str, str] = dict(
        catalogue_numerator_survival_2d="off",
        catalogue_numerator_survival_2d_center="unset",
    )
    if config == "b0i":
        diag_csv, elapsed = c1d.run_mirror_seed_inprocess(
            work_root,
            events,
            seed,
            galaxy_catalog=handler,
            selection_in_completion_numerator=BC_COMPLETION_CELL,
            completion_event_measure=BC_EVENT_MEASURE,
            catalogue_numerator_survival=BC_CATALOGUE_NUMERATOR_SURVIVAL,
            catalogue_global_selection=BC_CATALOGUE_GLOBAL_SELECTION,
            **cat_num_surv_2d_kwargs,
            **common_kwargs,
        )
    elif config == "ft":
        diag_csv, elapsed = c1d.run_mirror_seed_inprocess(
            work_root,
            events,
            seed,
            galaxy_catalog=handler,
            selection_in_completion_numerator=FT_COMPLETION_CELL,
            completion_event_measure=FT_EVENT_MEASURE,
            catalogue_numerator_survival=FT_CATALOGUE_NUMERATOR_SURVIVAL,
            # catalogue_global_selection deliberately NOT passed -- "auto"
            # resolves to "phi" under absolute_marginal, matching
            # p3_twin_test.py's _run_bsel_seed call site exactly (see
            # FT_CATALOGUE_NUMERATOR_SURVIVAL's docstring comment above).
            **cat_num_surv_2d_kwargs,
            **common_kwargs,
        )
    else:
        raise ValueError(f"config must be one of {CONFIG_CHOICES}, got {config!r}")
    return diag_csv, elapsed


# ── PA-HIER-32(d): the closed-form Es_null_det_i (site-2.2 score_lns's own
# deterministic expectation at truth, under each host's OWN generator kernel)
# ───────────────────────────────────────────────────────────────────────────

_ES_NULL_DET_FILENAME = "es_null_det.csv"
_SQRT2 = math.sqrt(2.0)
_ES_NULL_DET_Z_FLOOR = 1e-6  # site 2.2's own floor (bayesian_statistics.py's _z_lower_floor)
_ES_NULL_DET_WINDOW_SIGMA = 4.0  # integration_limit_sigma_multiplier (bayesian_statistics.py)


def _es_null_det_closed_form(
    z_g: npt.NDArray[np.float64],
    sigma_g: npt.NDArray[np.float64],
    host_pixels: npt.NDArray[np.int64],
    completeness: Any,
    h: float,
    n_grid: int = 4001,
) -> npt.NDArray[np.float64]:
    r"""Per-host closed-form ``Es_null_det_i`` (PA-HIER-32(d)):

    .. math::

        E_i = \frac{\int k_i(z)\, \mathrm{secs}_i(z)\, dz}{\int k_i(z)\, dz}
              \Big|_{z \in W^-_i}

    where ``k_i(z) = kern(i, b=0, s=1, z)`` is the site-2.2 host-``i`` kernel
    at theta=(0,1) (a Gaussian in ``z`` of width ``sigma_g[i]``, weighted by
    the comoving-volume element and the host's own pixel completeness
    ``f_k(z)``, normalized over its own +/-4 sigma window and floored at
    ``1e-6`` -- IDENTICAL in form to
    ``bayesian_statistics.py``'s ``single_host_likelihood_batch`` site-2.2
    kernel), ``secs_i(z) = [ln kern(i,0,sqrt2,z) - ln kern(i,0,1/sqrt2,z)] /
    ln(2)`` is the (ln-``s``) secant of that SAME kernel's log evaluated
    pointwise at the fixed observation ``z`` while varying only the kernel's
    own width parameter ``s`` -- i.e. exactly what ``score_lns`` computes for
    a single host's likelihood (PA-HIER-32(d): "the closed-form expectation
    of ``score_lns_i``", NOT of ``score_s_raw``; the denominator MUST match
    ``score_lns``'s own ``ln(2)``, not ``score_s_raw``'s ``sqrt2 - 1/sqrt2``)
    -- and ``W^-_i`` is the (narrower) ``s=1/sqrt2`` window, the intersection
    of the ``s=sqrt2`` and ``s=1/sqrt2`` windows (outside it one of the two
    secant terms is ``-inf``, so the pointwise finite-difference is
    undefined).

    This is deterministic and DATA-INDEPENDENT: a function only of host
    ``i``'s window (via ``z_g[i]``, ``sigma_g[i]``), the shared floor/width-
    multiplier constants above, and its sky-pixel completeness -- never of a
    realized ``z_true`` or observed ``d_L`` (PA-HIER-32(d)'s own definition).
    Vectorized port of the forensic instrument
    ``b1_1_forensic_work/f4_mechanism.py``'s ``kern()``/``E()`` (Es_null_det
    column only -- the survival-weighted "gen" variant is a different,
    non-registered statistic); the per-host loop is required because each
    host's window/grid differs, but every array op inside it is vectorized
    over the ``n_grid`` redshift nodes.

    Args:
        z_g, sigma_g: Per-host listed redshift and its (bare, quoted)
            REDSHIFT_MEASUREMENT_ERROR, shape ``(n_hosts,)``.
        host_pixels: HEALPix pixel index per host (:func:`_host_pixels`),
            shape ``(n_hosts,)``.
        completeness: Per-pixel completeness model (``f_k`` accessor),
            e.g. from ``c1d.build_bsel_selection_objects``.
        h: Dimensionless Hubble parameter (the SAME truth ``h`` the venue's
            realization was drawn under -- ``H_TRUE``/``H_GEN`` for every
            registered arm; the closed form is about the generator-kernel
            identity at truth-theta, not about the estimator's evaluated h).
        n_grid: Trapezoidal-quadrature node count per per-host integral
            (default 4001, matching ``f4_mechanism.py``'s ``NG``; tests use a
            smaller value for speed -- the form is exact only in the
            ``n_grid -> infinity`` limit, same as any trapezoidal quadrature).

    Returns:
        ``Es_null_det_i`` per host, shape ``(n_hosts,)``.

    References:
        PREREGISTRATION_HIER_HTHETA_20260826.md PA-HIER-32(d) (definition,
            the +0.0455 +/- 0.0005 per-unit-s unweighted expectation).
        B1_1_S0A_DEFECT_FORENSIC_20260829.md E13 (independent measurement),
            b1_1_forensic_work/f4_mechanism.py (the archived closed-form
            instrument this function re-derives generically).
    """
    n_hosts = z_g.shape[0]
    out = np.full(n_hosts, np.nan, dtype=np.float64)
    # PA-HIER-32(d): Es_null_det_i is the closed-form expectation of
    # score_lns_i (NOT score_s_raw) -- the secant denominator below MUST be
    # score_lns's own ln(2), matching compute_scores' denom_lns exactly (a
    # verifier MUST_FIX: the raw secant's sqrt2 - 1/sqrt2 denominator is
    # 1.02014x too large and was previously used here in error).
    denom_lns = math.log(2.0)
    for i in range(n_hosts):
        zg = float(z_g[i])
        sg = float(sigma_g[i])
        pix = host_pixels[i : i + 1]
        if not (sg > 0.0) or not np.isfinite(zg) or not np.isfinite(sg):
            continue

        lo0 = max(zg - _ES_NULL_DET_WINDOW_SIGMA * sg, _ES_NULL_DET_Z_FLOOR)
        hi0 = zg + _ES_NULL_DET_WINDOW_SIGMA * sg
        zz = np.linspace(lo0, hi0, n_grid)
        k0 = _es_null_det_kernel(0.0, 1.0, zz, zg, sg, pix, completeness, h, n_grid)

        def _ln_kernel(
            b: float,
            s: float,
            zg: float = zg,
            sg: float = sg,
            pix: npt.NDArray[np.int64] = pix,
            zz: npt.NDArray[np.float64] = zz,
        ) -> npt.NDArray[np.float64]:
            return np.log(
                np.clip(
                    _es_null_det_kernel(b, s, zz, zg, sg, pix, completeness, h, n_grid),
                    1e-300,
                    None,
                )
            )

        secs = (_ln_kernel(0.0, _SQRT2) - _ln_kernel(0.0, 1.0 / _SQRT2)) / denom_lns
        window_minus = (
            zz >= max(zg - _ES_NULL_DET_WINDOW_SIGMA * sg / _SQRT2, _ES_NULL_DET_Z_FLOOR)
        ) & (zz <= zg + _ES_NULL_DET_WINDOW_SIGMA * sg / _SQRT2)
        weight = np.where(window_minus, k0, 0.0)
        weight_sum = np.trapezoid(weight, zz)
        if weight_sum > 0.0:
            out[i] = float(np.trapezoid(weight * secs, zz) / weight_sum)
    return out


def _es_null_det_kernel(
    b: float,
    s: float,
    z_eval: npt.NDArray[np.float64],
    zg: float,
    sg: float,
    pix: npt.NDArray[np.int64],
    completeness: Any,
    h: float,
    n_grid: int,
) -> npt.NDArray[np.float64]:
    """The site-2.2 host kernel ``kern(b, s, z_eval)`` of
    :func:`_es_null_det_closed_form` -- extracted to module level (no closure
    over a per-host loop variable, avoiding a B023-class capture bug) so it
    is independently callable/testable."""
    zc = zg + b * (1.0 + zg)
    sc = s * sg
    lo = max(zc - _ES_NULL_DET_WINDOW_SIGMA * sc, _ES_NULL_DET_Z_FLOOR)
    hi = zc + _ES_NULL_DET_WINDOW_SIGMA * sc
    w_eval = _completeness_at_host_nodes(completeness, z_eval[None, :], pix, h)[0]
    if not np.any(w_eval > 0.0):
        w_eval = np.ones_like(z_eval)
    z_norm = np.linspace(lo, hi, n_grid)
    w_norm = _completeness_at_host_nodes(completeness, z_norm[None, :], pix, h)[0]
    if not np.any(w_norm > 0.0):
        w_norm = np.ones_like(z_norm)
    gauss_norm = np.exp(-0.5 * ((z_norm - zc) / sc) ** 2) / (sc * math.sqrt(2.0 * math.pi))
    vol_norm = np.asarray(comoving_volume_element(z_norm, h=h), dtype=np.float64)
    normalization = np.trapezoid(gauss_norm * vol_norm / (1.0 + z_norm) * w_norm, z_norm)
    gauss_eval = np.exp(-0.5 * ((z_eval - zc) / sc) ** 2) / (sc * math.sqrt(2.0 * math.pi))
    vol_eval = np.asarray(comoving_volume_element(z_eval, h=h), dtype=np.float64)
    kern_vals = gauss_eval * vol_eval / (1.0 + z_eval) * w_eval / normalization
    return np.where((z_eval >= lo) & (z_eval <= hi), kern_vals, 0.0)


def compute_es_null_det_table(
    events: pd.DataFrame,
    handler: Any,
    h: float = H_TRUE,
    n_grid: int = 4001,
) -> pd.DataFrame:
    """Per-event ``Es_null_det_i`` table (PA-HIER-32(d)), columns
    ``["event_idx", "es_null_det"]``.

    ``events`` is the realization DataFrame returned by :func:`_build_venue`
    (or :func:`build_bc_venue`/:func:`build_ft_venue`) -- ``events.index``
    (after any ``--event-cap`` truncation, which uses
    ``reset_index(drop=True)``) IS ``event_idx`` as written to
    ``event_likelihoods.csv`` (:func:`read_event_ln_l`'s join key), and
    ``events["host_galaxy_index"]`` indexes DIRECTLY into ``handler.
    reduced_galaxy_catalog`` -- ``handler`` is the SAME object both the
    injection draw and the evaluation call use (``run_mirror_seed_inprocess``
    receives it as ``galaxy_catalog=handler``), so no catalogue-frame
    repositioning (cf. ``GalaxyCatalogueHandler.resolve_host_recovery_
    position``'s docstring) is needed here.

    Dark-class events (``host_galaxy_index == -1``, no in-catalogue host) are
    OMITTED -- they carry no ``Es_null_det`` by construction (F4 in the gate
    doc: the dark class scores exactly 0.0 on every axis).
    """
    host_idx = events["host_galaxy_index"].to_numpy()
    in_catalogue = host_idx >= 0
    if not np.any(in_catalogue):
        return pd.DataFrame(
            {"event_idx": pd.Series(dtype=np.int64), "es_null_det": pd.Series(dtype=np.float64)}
        )
    catalog = handler.reduced_galaxy_catalog
    idx = host_idx[in_catalogue]
    z_g = catalog[InternalCatalogColumns.REDSHIFT].to_numpy(dtype=np.float64)[idx]
    sigma_g = catalog[InternalCatalogColumns.REDSHIFT_ERROR].to_numpy(dtype=np.float64)[idx]
    phi_s = catalog[InternalCatalogColumns.PHI_S].to_numpy(dtype=np.float64)[idx]
    theta_s = catalog[InternalCatalogColumns.THETA_S].to_numpy(dtype=np.float64)[idx]
    completeness_obj, _phi_table = c1d.build_bsel_selection_objects(h_true=h)
    host_pixels = _host_pixels(completeness_obj, phi_s, theta_s)
    es_null_det = _es_null_det_closed_form(
        z_g, sigma_g, host_pixels, completeness_obj, h, n_grid=n_grid
    )
    return pd.DataFrame(
        {
            "event_idx": events.index.to_numpy()[in_catalogue],
            "es_null_det": es_null_det,
        }
    )


def _write_es_null_det_cache(work_root: Path, table: pd.DataFrame) -> None:
    """Write :func:`compute_es_null_det_table`'s output to
    ``work_root/es_null_det.csv`` (node-independent -- one file per seed,
    NOT per node) so a later ``--score-only`` invocation can read it back
    with NO venue reconstruction (:func:`_read_es_null_det_cache`)."""
    table.to_csv(work_root / _ES_NULL_DET_FILENAME, index=False)


def _read_es_null_det_cache(work_root: Path) -> pd.DataFrame | None:
    """Read back :func:`_write_es_null_det_cache`'s file, or ``None`` if
    absent (e.g. a pre-T1.3-zwin banked run) -- callers degrade gracefully
    (:func:`compute_scores`'s ``score_s_available``)."""
    path = work_root / _ES_NULL_DET_FILENAME
    if not path.is_file():
        return None
    return pd.read_csv(path)


# ── Diagnostics readback ───────────────────────────────────────────────────


def read_event_ln_l(
    diag_csv: Path,
    h: float,
    rtol: float = 1e-9,
    es_null_det: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Read ``event_likelihoods.csv`` and return per-event ln L at ``h``.

    Columns returned: ``event_idx``, ``ln_L_no_bh``, ``ln_L_with_bh`` (NaN
    where the corresponding ``combined_*`` column is non-positive -- the
    same non-positivity the estimator's own ``num_log_term_*`` diagnostic
    columns guard against, bayesian_statistics.py:5788-5794), plus
    ``es_null_det`` when *es_null_det* is given (PA-HIER-32(d), row #255 tree
    2 node T1.3-zwin): a left-merge on ``event_idx`` of *es_null_det*'s own
    ``["event_idx", "es_null_det"]`` columns (:func:`compute_es_null_det_table`
    / :func:`_read_es_null_det_cache`), NaN for events absent from it (e.g.
    dark-class events, which have no in-catalogue host). ``None`` (default)
    omits the column entirely -- :func:`compute_scores` treats its absence
    from EVERY node's frame the same way as an all-NaN column (the corrected
    ``score_s`` is reported unavailable, ``score_s_raw`` is unaffected).
    """
    df = pd.read_csv(diag_csv)
    mask = np.isclose(df["h"].to_numpy(dtype=float), h, rtol=rtol, atol=1e-12)
    sub = df.loc[mask, ["event_idx", *DIAG_VALUE_COLUMNS]].copy()
    if sub.empty:
        raise RuntimeError(
            f"no rows at h={h!r} in {diag_csv} (h values present: {sorted(set(df['h']))})"
        )
    sub = sub.drop_duplicates(subset="event_idx", keep="last")
    for col, out in (("combined_no_bh", "ln_L_no_bh"), ("combined_with_bh", "ln_L_with_bh")):
        vals = sub[col].to_numpy(dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            sub[out] = np.where(vals > 0.0, np.log(vals), np.nan)
    out_cols = ["event_idx", "ln_L_no_bh", "ln_L_with_bh"]
    result = sub[out_cols].sort_values("event_idx").reset_index(drop=True)
    if es_null_det is not None:
        result = result.merge(es_null_det[["event_idx", "es_null_det"]], on="event_idx", how="left")
    return result


@dataclass
class NodeResult:
    node: str
    theta_b: float
    theta_s: float
    seed: int
    diag_csv: str
    elapsed_s: float
    n_events: int
    ln_l: pd.DataFrame = field(repr=False)


def _resolve_score_h(h_values: tuple[float, ...], score_h: float | None) -> float:
    """Resolve the single h used for this driver's own internal per-event
    readback/scoring (``read_event_ln_l``'s ``h`` argument -- distinct from
    ``h_values``, the possibly-multi-h grid actually evaluated).

    Byte-identical default: ``score_h=None`` with the default ``h_values=
    (H_GEN,)`` resolves to ``H_GEN``, exactly the old hardcoded call. For a
    KW-Q1-style multi-h run (``h_values=(0.725, 0.735)``, H_GEN absent),
    ``score_h`` must be passed explicitly (this driver's own compute_scores/
    gate_eng/gate_parity are not KW-Q1's statistic -- KW-Q1 is scored by
    ``kwq1_score.py`` directly from the diagnostics CSVs, which read both h
    rows itself); if omitted in that case, the first h_values entry is used
    as an inert placeholder so run_arm_seed_s0a/s0r still produce a valid
    (if not directly meaningful) NodeResult.ln_l rather than crashing.
    """
    if score_h is not None:
        return score_h
    if H_GEN in h_values:
        return H_GEN
    return h_values[0]


def run_arm_seed_s0a(
    seed: int,
    out_root: Path,
    nodes: tuple[str, ...],
    event_cap: int | None,
    theta_sites: str = "all",
    smear: str = "auto",
    config: str = "b0i",
    h_values: tuple[float, ...] = (H_GEN,),
    score_h: float | None = None,
    candidate_dump_dir: str | None = None,
    theta_phi_divisor: str = "off",
    sky_cone_k: float = 1.5,
    catalogue_leg_1d_mass_aware: str = "off",
    theta_zwindow: str = "off",
    z_window_k: float = 1.0,
) -> list[NodeResult]:
    """S0-A: one seed, the theta-cross at h=H_GEN, sigma_z_scale=1.0 (truth-theta=(0,1)).

    ``theta_sites``/``smear``/``config``/``h_values``/``score_h`` default to
    exactly the pre-P1/pre-KW-Q1 behaviour (BYTE-IDENTICAL: theta_sites=
    "all", smear="auto" -> smear_global_selection=theta_engaged, config=
    "b0i" -> the original bc venue/flags, h_values=(H_GEN,) -> the single
    h=0.73 node, score_h=None -> H_GEN). ``theta_phi_divisor``/``sky_cone_k``
    (T1.2) and ``theta_zwindow``/``z_window_k`` (T1.3-zwin) default to
    exactly the pre-T1.1/pre-T1.3-zwin behaviour ("off"/1.5, "off"/1.0). Node
    output directories gain a suffix (:func:`_node_dir_suffix`) that is
    EMPTY at these defaults, so default-run paths are unchanged.

    PA-HIER-32(d) score_s support: this function computes and caches the
    per-event closed-form ``Es_null_det_i`` (:func:`compute_es_null_det_table`)
    ONCE per seed (node-independent -- a property of site 2.2's own kernel
    at theta=(0,1), never of the realized data), writes it to
    ``work_root/es_null_det.csv`` (so a later ``--score-only`` invocation on
    this seed can read it back with NO venue reconstruction), and merges it
    into every node's ``ln_l`` (:func:`read_event_ln_l`'s ``es_null_det``
    kwarg) so :func:`compute_scores` can compute the corrected ``score_s``.
    """
    work_root = out_root / f"s0a_seed{seed}"
    work_root.mkdir(parents=True, exist_ok=True)
    events, handler = _build_venue(config, work_root, seed, sigma_z_scale=1.0)
    if event_cap is not None:
        events = events.head(event_cap).reset_index(drop=True)
    es_null_det = compute_es_null_det_table(events, handler)
    _write_es_null_det_cache(work_root, es_null_det)
    suffix = _node_dir_suffix(
        theta_sites,
        smear,
        config,
        theta_phi_divisor,
        sky_cone_k,
        catalogue_leg_1d_mass_aware,
        theta_zwindow,
        z_window_k,
    )
    read_h = _resolve_score_h(h_values, score_h)
    results: list[NodeResult] = []
    for node in nodes:
        theta_b, theta_s = THETA_NODES[node]
        node_root = work_root / f"node_{node}{suffix}"
        node_root.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        diag_csv, elapsed = run_theta_node(
            node_root,
            events,
            seed,
            handler,
            theta_b,
            theta_s,
            h_values=h_values,
            theta_sites=theta_sites,
            smear=smear,
            config=config,
            # T2.2 (row #255 A10): per-(seed, node) subdirectory so parallel
            # cells never overwrite each other's per_candidate_h_*.csv.
            candidate_dump_dir=(
                str(Path(candidate_dump_dir) / f"seed{seed}_node{node}")
                if candidate_dump_dir
                else None
            ),
            theta_phi_divisor=theta_phi_divisor,
            sky_cone_k=sky_cone_k,
            catalogue_leg_1d_mass_aware=catalogue_leg_1d_mass_aware,
            theta_zwindow=theta_zwindow,
            z_window_k=z_window_k,
        )
        wall = time.time() - t0
        ln_l = read_event_ln_l(diag_csv, read_h, es_null_det=es_null_det)
        results.append(
            NodeResult(
                node=node,
                theta_b=theta_b,
                theta_s=theta_s,
                seed=seed,
                diag_csv=str(diag_csv),
                elapsed_s=elapsed,
                n_events=len(ln_l),
                ln_l=ln_l,
            )
        )
        print(
            f"[S0-A seed={seed} node={node} theta=({theta_b},{theta_s}) "
            f"theta_sites={theta_sites} smear={smear} config={config} "
            f"theta_phi_divisor={theta_phi_divisor} sky_cone_k={sky_cone_k} "
            f"theta_zwindow={theta_zwindow} z_window_k={z_window_k}] "
            f"n_events={len(ln_l)} evaluate_s={elapsed:.2f} wall_s={wall:.2f} -> {diag_csv}",
            flush=True,
        )
    return results


def run_arm_seed_s0r(
    seed: int,
    out_root: Path,
    nodes: tuple[str, ...],
    event_cap: int | None,
    theta_sites: str = "all",
    smear: str = "auto",
    config: str = "b0i",
    h_values: tuple[float, ...] = (H_GEN,),
    score_h: float | None = None,
    candidate_dump_dir: str | None = None,
    theta_phi_divisor: str = "off",
    sky_cone_k: float = 1.5,
    catalogue_leg_1d_mass_aware: str = "off",
    theta_zwindow: str = "off",
    z_window_k: float = 1.0,
) -> list[NodeResult]:
    """S0-R: one seed, the theta-cross at h=H_GEN, sigma_z_scale=1.5 (DISCLOSED NULL, see module docstring).

    Same byte-identical-default argument as :func:`run_arm_seed_s0a` (this
    arm's own S0_R_SIGMA_SCALE dose is orthogonal to the P1/KW-Q1/T1.2/
    T1.3-zwin axes); same ``Es_null_det`` caching (:func:`compute_es_null_det_table`,
    ``work_root/es_null_det.csv``) as :func:`run_arm_seed_s0a`.
    """
    work_root = out_root / f"s0r_seed{seed}"
    work_root.mkdir(parents=True, exist_ok=True)
    events, handler = _build_venue(config, work_root, seed, sigma_z_scale=S0_R_SIGMA_SCALE)
    if event_cap is not None:
        events = events.head(event_cap).reset_index(drop=True)
    es_null_det = compute_es_null_det_table(events, handler)
    _write_es_null_det_cache(work_root, es_null_det)
    suffix = _node_dir_suffix(
        theta_sites,
        smear,
        config,
        theta_phi_divisor,
        sky_cone_k,
        catalogue_leg_1d_mass_aware,
        theta_zwindow,
        z_window_k,
    )
    read_h = _resolve_score_h(h_values, score_h)
    results: list[NodeResult] = []
    for node in nodes:
        theta_b, theta_s = THETA_NODES[node]
        node_root = work_root / f"node_{node}{suffix}"
        node_root.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        diag_csv, elapsed = run_theta_node(
            node_root,
            events,
            seed,
            handler,
            theta_b,
            theta_s,
            h_values=h_values,
            theta_sites=theta_sites,
            smear=smear,
            config=config,
            # T2.2 (row #255 A10): per-(seed, node) subdirectory so parallel
            # cells never overwrite each other's per_candidate_h_*.csv.
            candidate_dump_dir=(
                str(Path(candidate_dump_dir) / f"seed{seed}_node{node}")
                if candidate_dump_dir
                else None
            ),
            theta_phi_divisor=theta_phi_divisor,
            sky_cone_k=sky_cone_k,
            catalogue_leg_1d_mass_aware=catalogue_leg_1d_mass_aware,
            theta_zwindow=theta_zwindow,
            z_window_k=z_window_k,
        )
        wall = time.time() - t0
        ln_l = read_event_ln_l(diag_csv, read_h, es_null_det=es_null_det)
        results.append(
            NodeResult(
                node=node,
                theta_b=theta_b,
                theta_s=theta_s,
                seed=seed,
                diag_csv=str(diag_csv),
                elapsed_s=elapsed,
                n_events=len(ln_l),
                ln_l=ln_l,
            )
        )
        print(
            f"[S0-R seed={seed} node={node} theta=({theta_b},{theta_s}) "
            f"theta_sites={theta_sites} smear={smear} config={config} "
            f"theta_phi_divisor={theta_phi_divisor} sky_cone_k={sky_cone_k} "
            f"theta_zwindow={theta_zwindow} z_window_k={z_window_k}] "
            f"n_events={len(ln_l)} evaluate_s={elapsed:.2f} wall_s={wall:.2f} -> {diag_csv}",
            flush=True,
        )
    return results


def run_seed_s0c(
    seed: int,
    out_root: Path,
    event_cap: int | None,
    theta_phi_divisor: str = "off",
    sky_cone_k: float = 1.5,
    catalogue_leg_1d_mass_aware: str = "off",
    theta_zwindow: str = "off",
    z_window_k: float = 1.0,
) -> dict[str, Any]:
    """S0-C: one seed, theta=(0,1), the full 41-node H_GRID_41 (costing probe, prereg §2.1).

    ``theta_phi_divisor``/``sky_cone_k`` (T1.2, row #255 tree 2 node T1.2)
    and ``theta_zwindow``/``z_window_k`` (T1.3-zwin, row #255 tree 2 node
    T1.3-zwin) are forwarded verbatim to ``run_mirror_seed_inprocess`` for
    parity with S0-A/S0-R -- unconditional forwarding (GATE T-ID: both are
    no-ops at theta=(0,1), S0-C's only theta). Defaults ("off", 1.5, "off",
    1.0) are byte-identical; S0-C's output directory name
    (``node_truth_fullgrid``, unparameterized by any other axis either) is
    unchanged regardless.
    """
    work_root = out_root / f"s0c_seed{seed}"
    work_root.mkdir(parents=True, exist_ok=True)
    events, handler = build_bc_venue(work_root, seed, sigma_z_scale=1.0)
    if event_cap is not None:
        events = events.head(event_cap).reset_index(drop=True)
    node_root = work_root / "node_truth_fullgrid"
    node_root.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    diag_csv, elapsed = c1d.run_mirror_seed_inprocess(
        node_root,
        events,
        seed,
        galaxy_catalog=handler,
        h_values=H_GRID_41,
        h_bounds=H_BOUNDS,
        selection_in_completion_numerator=BC_COMPLETION_CELL,
        completion_event_measure=BC_EVENT_MEASURE,
        catalogue_numerator_survival=BC_CATALOGUE_NUMERATOR_SURVIVAL,
        catalogue_global_selection=BC_CATALOGUE_GLOBAL_SELECTION,
        # [P3-2D] pinned explicitly to the pre-adoption COUNTERFACTUAL after
        # the production default flip (row #223, charter node B7.3;
        # PHYSICS_CHANGE_2D_TWIN_ADOPTION_20260829.md §6.1 Class-B site B3)
        # so this banked Stage-0/KW-Q1 comparand stays byte-identical.
        catalogue_numerator_survival_2d="off",
        catalogue_numerator_survival_2d_center="unset",
        theta_b=0.0,
        theta_s=1.0,
        theta_sites="all",
        smear_global_selection=False,
        theta_phi_divisor=theta_phi_divisor,
        sky_cone_k=sky_cone_k,
        catalogue_leg_1d_mass_aware=catalogue_leg_1d_mass_aware,
        theta_zwindow=theta_zwindow,
        z_window_k=z_window_k,
    )
    wall = time.time() - t0
    # Per-h marginal cost: posterior JSONs are written progressively as each
    # h completes (bayesian_statistics.py ~4600-4633), so their mtimes give
    # real per-h wall-clock deltas -- the actual point of S0-C (prereg §7:
    # "the MEASURED marginal per-h cost").
    posteriors_dir = node_root / "simulations" / "posteriors"
    per_h_files = sorted(posteriors_dir.glob("h_*.json"), key=lambda p: p.stat().st_mtime)
    mtimes = [p.stat().st_mtime for p in per_h_files]
    deltas = [mtimes[0] - t0] + [b - a for a, b in zip(mtimes, mtimes[1:])]
    return {
        "seed": seed,
        "n_h": len(H_GRID_41),
        "diag_csv": str(diag_csv),
        "evaluate_s": elapsed,
        "wall_s": wall,
        "per_h_files": [p.name for p in per_h_files],
        "per_h_delta_s": deltas,
        "mean_marginal_h_cost_s": (float(np.mean(deltas[1:])) if len(deltas) > 1 else None),
        "first_h_cost_s": deltas[0] if deltas else None,
    }


# ── Registered statistics (prereg §4.1) ─────────────────────────────────────


def compute_scores(
    all_nodes: dict[str, list[NodeResult]],
    seeds: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    """Pool per-event score_b/score_s over every event and seed (prereg §4.1).

    ``all_nodes`` maps node name -> list of NodeResult (one per seed), each
    already restricted to the SAME event_cap-truncated event set per seed
    (event sets differ ACROSS seeds -- that is fine, pooling is over the
    union of (seed, event_idx) pairs, not a per-event paired comparison).

    score_b       = [lnL(b=+0.02,s=1) - lnL(b=-0.02,s=1)] / 0.04
    score_s_raw   = [lnL(b=0,s=sqrt2) - lnL(b=0,s=1/sqrt2)] / (sqrt2 - 1/sqrt2)
                    (the OLD/superseded raw linear secant, kept for
                    continuity -- PA-HIER-4 registered this centred on the
                    interval's ARITHMETIC midpoint, s=1.0606..., not s=1;
                    PREREGISTRATION_HIER_HTHETA_20260826.md PA-HIER-4)
    score_lns     = [lnL(b=0,ln s=+ln sqrt2) - lnL(b=0,ln s=-ln sqrt2)] / ln(2)
                    (PA-HIER-4's correction: the SAME two nodes, re-centred
                    in ln s -- exactly centred on truth ln s = 0; same
                    numerator as score_s_raw, denominator ln(2) = 2 ln(sqrt2)
                    in place of (sqrt2 - 1/sqrt2))
    score_s       = score_lns - Es_null_det   (PA-HIER-32(d), the CORRECTED
                    and now-primary s-statistic: score_lns's own deterministic
                    expectation at truth under each host's OWN generator
                    kernel, Es_null_det_i, is non-zero -- E[score_lns |
                    truth]_unweighted = +0.0455 +/- 0.0005 per unit s
                    (PREREGISTRATION_HIER_HTHETA_20260826.md PA-HIER-32(d),
                    B1_1_S0A_DEFECT_FORENSIC_20260829.md E13) -- so
                    subtracting the per-host Es_null_det_i, which is
                    data-independent (a function only of that host's window,
                    floor and sigma_g, never of the realized z_true or
                    observed d_L; :func:`compute_es_null_det_table`'s closed
                    form), removes exactly the deterministic part of the
                    secant's response at truth. Available only for events
                    whose ``es_null_det`` column both s_plus and s_minus
                    ``NodeResult.ln_l`` frames carry (:func:`read_event_ln_l`'s
                    ``es_null_det`` kwarg / :func:`compute_es_null_det_table`);
                    when absent for every event, ``score_s``'s stats report
                    ``n_pooled=0``/NaN and ``score_s_available`` is ``False``
                    -- ``score_s_raw`` and ``score_lns`` are unaffected.
    Z_x = mean(score_x) / SEM(score_x), pooled over events and seeds.

    Args:
        seeds: The full seed set this caller expected results for, used ONLY
            to name exactly which (seed, node) pairs are missing in the error
            message below. Optional so this function stays callable with just
            ``all_nodes`` (its pre-fix signature); when omitted, the missing-
            node list is reported without a per-seed breakdown.

    Raises:
        ValueError: if any of the 4 off-truth nodes has ZERO ``NodeResult``
            entries (runner-disclosed P0 crash fix, ``B1_2_DRIVER_EXTENSION_
            NOTE.md`` "Crash fix"): every ``pd.concat`` below used to receive
            an empty list in that case and raise pandas' own opaque "No
            objects to concatenate" with no indication of WHICH (seed, node)
            evaluate() call never produced a result. Both call sites
            (:func:`run_arm`, :func:`score_only_payload`) already gate on a
            "produced" check before calling this, so this guard is a second,
            defensive line -- it fires only if a future caller skips that
            check.
    """

    # Axis-independent gating (T1.3-zwin, row #255): the registered P1 arm's
    # own node list is {truth, s_plus, s_minus} -- NO b-nodes (PHYSICS_CHANGE_
    # THETA_ZWINDOW_20260830.md §5.6: "b-axis: NOT re-run under P1 ... T1.2's
    # own b-axis certification stands unchanged"). The pre-T1.3-zwin gate
    # unconditionally required all 4 of b_plus/b_minus/s_plus/s_minus, so it
    # could never score ANY b-node-free arm, including this one. Relaxed
    # here to be PER-AXIS: an axis (b, s) is "ready" only if BOTH its nodes
    # are present (a lone b_plus with no b_minus is still an error -- a
    # broken pair, not a deliberate omission); at least one axis must be
    # ready or there is nothing to pool at all. A ready axis's own
    # (seed, node) completeness is still checked below (the original
    # crash-fix diagnostic, per axis).
    def _axis_missing(pair: tuple[str, str]) -> list[str]:
        return [n for n in pair if not all_nodes.get(n)]

    b_missing = _axis_missing(("b_plus", "b_minus"))
    s_missing = _axis_missing(("s_plus", "s_minus"))
    has_b = not b_missing
    has_s = not s_missing
    # Check for a genuine BROKEN pair first (exactly one of an axis's two
    # nodes present -- some seed's worker for the other one raised, or the
    # caller only requested one of the pair by mistake): this is checked
    # BEFORE the "nothing to score" catch-all below so its more specific,
    # per-axis diagnostic always wins over the generic one, regardless of
    # the OTHER axis's state.
    for axis_name, missing, ready in (("b", b_missing, has_b), ("s", s_missing, has_s)):
        # ready (missing == []): fully present, fine. missing == both nodes:
        # the axis was never requested at all, ALSO fine (the relaxation
        # this node adds -- e.g. the registered P1 arm's node list has no
        # b_plus/b_minus at all). Only missing == exactly one of the two is
        # a genuine broken pair -- that is still the original crash-fix
        # error.
        if ready or len(missing) != 1:
            continue
        n_present_by_node = {
            n: len(all_nodes.get(n, [])) for n in ("b_plus", "b_minus", "s_plus", "s_minus")
        }
        if seeds is not None:
            missing_pairs = [
                (seed, n)
                for n in missing
                for seed in seeds
                if seed not in {r.seed for r in all_nodes.get(n, [])}
            ]
            detail = f"missing (seed, node) pairs: {missing_pairs}"
        else:
            detail = f"missing nodes (no seed list given): {missing}"
        raise ValueError(
            f"compute_scores: the {axis_name}-axis has an incomplete node pair -- {detail}. "
            f"n_present_by_node={n_present_by_node}. Every one of these evaluate() calls "
            "either was never attempted or raised inside its worker -- check the caller's "
            "printed WORKER ERROR lines / payload['errors'] (run_arm) or missing_csv_paths "
            "(--score-only) for the real underlying cause."
        )
    if not has_b and not has_s:
        n_present_by_node = {
            n: len(all_nodes.get(n, [])) for n in ("b_plus", "b_minus", "s_plus", "s_minus")
        }
        raise ValueError(
            "compute_scores: cannot pool score_b or score_s -- both axes are incomplete "
            f"(b_missing={b_missing}, s_missing={s_missing}). n_present_by_node="
            f"{n_present_by_node}. Every one of these evaluate() calls either was never "
            "attempted or raised inside its worker -- check the caller's printed WORKER "
            "ERROR lines / payload['errors'] (run_arm) or missing_csv_paths (--score-only) "
            "for the real underlying cause."
        )

    channels = ("ln_L_no_bh", "ln_L_with_bh")
    out: dict[str, Any] = {}
    for channel in channels:
        if has_b:
            # Join b_plus/b_minus per (seed, event_idx).
            bp = pd.concat(
                [
                    r.ln_l.assign(seed=r.seed)[["seed", "event_idx", channel]]
                    for r in all_nodes["b_plus"]
                ],
                ignore_index=True,
            ).rename(columns={channel: "b_plus"})
            bm = pd.concat(
                [
                    r.ln_l.assign(seed=r.seed)[["seed", "event_idx", channel]]
                    for r in all_nodes["b_minus"]
                ],
                ignore_index=True,
            ).rename(columns={channel: "b_minus"})
            b_join = bp.merge(bm, on=["seed", "event_idx"], how="inner")
            score_b = (b_join["b_plus"] - b_join["b_minus"]) / 0.04

        if has_s:
            # es_null_det (PA-HIER-32(d)) is node-independent (one value per
            # (seed, event_idx), the SAME whether read off s_plus's or
            # s_minus's frame -- see compute_es_null_det_table); include it
            # from s_plus ONLY, so the s_plus/s_minus merge below never
            # produces a duplicate (es_null_det_x/es_null_det_y) column.
            _sp_has_es = all("es_null_det" in r.ln_l.columns for r in all_nodes["s_plus"]) and bool(
                all_nodes["s_plus"]
            )
            sp_cols = ["seed", "event_idx", channel, *(["es_null_det"] if _sp_has_es else [])]
            sp = pd.concat(
                [r.ln_l.assign(seed=r.seed)[sp_cols] for r in all_nodes["s_plus"]],
                ignore_index=True,
            ).rename(columns={channel: "s_plus"})
            sm = pd.concat(
                [
                    r.ln_l.assign(seed=r.seed)[["seed", "event_idx", channel]]
                    for r in all_nodes["s_minus"]
                ],
                ignore_index=True,
            ).rename(columns={channel: "s_minus"})
            s_join = sp.merge(sm, on=["seed", "event_idx"], how="inner")
            denom_s_raw = math.sqrt(2.0) - 1.0 / math.sqrt(2.0)
            score_s_raw = (s_join["s_plus"] - s_join["s_minus"]) / denom_s_raw
            # PA-HIER-4's ln-s-centred secant (same numerator, re-centred
            # denominator -- see this function's docstring).
            denom_lns = math.log(2.0)
            score_lns = (s_join["s_plus"] - s_join["s_minus"]) / denom_lns

        def _mean_sem(series: pd.Series) -> tuple[float, float, float, int]:
            vals = series.to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            n = vals.size
            if n == 0:
                return float("nan"), float("nan"), float("nan"), 0
            mean = float(np.mean(vals))
            sem = float(np.std(vals, ddof=1) / math.sqrt(n)) if n > 1 else float("nan")
            z = mean / sem if sem and np.isfinite(sem) and sem > 0 else float("nan")
            return mean, sem, z, n

        if has_b:
            mean_b, sem_b, z_b, n_b = _mean_sem(score_b)
        else:
            mean_b, sem_b, z_b, n_b = float("nan"), float("nan"), float("nan"), 0

        if has_s:
            mean_s_raw, sem_s_raw, z_s_raw, n_s_raw = _mean_sem(score_s_raw)
            mean_lns, sem_lns, z_lns, n_lns = _mean_sem(score_lns)
            # PA-HIER-32(d): score_s (corrected, now PRIMARY) = score_lns -
            # Es_null_det, only for events carrying a (non-NaN) Es_null_det_i.
            score_s_available = bool(_sp_has_es and s_join["es_null_det"].notna().any())
            if score_s_available:
                score_s = score_lns - s_join["es_null_det"]
                mean_s, sem_s, z_s, n_s = _mean_sem(score_s)
            else:
                mean_s, sem_s, z_s, n_s = float("nan"), float("nan"), float("nan"), 0
        else:
            mean_s_raw, sem_s_raw, z_s_raw, n_s_raw = float("nan"), float("nan"), float("nan"), 0
            mean_lns, sem_lns, z_lns, n_lns = float("nan"), float("nan"), float("nan"), 0
            mean_s, sem_s, z_s, n_s = float("nan"), float("nan"), float("nan"), 0
            score_s_available = False

        out[channel] = {
            "score_b": {"mean": mean_b, "sem": sem_b, "Z": z_b, "n_pooled": n_b},
            "score_b_available": has_b,
            # "score_s" is the CORRECTED (PA-HIER-32(d) primary) statistic --
            # gate_parity/verdict_s0a/verdict_s0r read this key. Falls back
            # to n_pooled=0/NaN (never a silent wrong number) when
            # Es_null_det is unavailable, OR when the s-axis nodes were not
            # requested at all; score_s_raw/score_lns are always computed
            # and reported alongside for continuity/diagnosis whenever the
            # s-axis IS present.
            "score_s": {"mean": mean_s, "sem": sem_s, "Z": z_s, "n_pooled": n_s},
            "score_s_available": score_s_available,
            "score_lns": {"mean": mean_lns, "sem": sem_lns, "Z": z_lns, "n_pooled": n_lns},
            "score_s_raw": {
                "mean": mean_s_raw,
                "sem": sem_s_raw,
                "Z": z_s_raw,
                "n_pooled": n_s_raw,
            },
        }
    return out


def gate_eng(all_nodes: dict[str, list[NodeResult]], channel: str = "ln_L_no_bh") -> dict[str, Any]:
    """GATE ENG (prereg §3.4): >=10% of scored events move by >=1e-6 relative vs truth, per node."""
    truth_by_seed = {r.seed: r.ln_l.set_index("event_idx")[channel] for r in all_nodes["truth"]}
    out: dict[str, Any] = {}
    for node in NODE_ORDER:
        if node == "truth":
            continue
        fracs = []
        for r in all_nodes.get(node, []):
            truth = truth_by_seed.get(r.seed)
            if truth is None:
                continue
            node_vals = r.ln_l.set_index("event_idx")[channel]
            common = node_vals.index.intersection(truth.index)
            if len(common) == 0:
                continue
            a = node_vals.loc[common].to_numpy(dtype=float)
            b = truth.loc[common].to_numpy(dtype=float)
            finite = np.isfinite(a) & np.isfinite(b) & (b != 0.0)
            if finite.sum() == 0:
                continue
            rel = np.abs(a[finite] - b[finite]) / np.abs(b[finite])
            frac_moved = float(np.mean(rel >= ENG_REL_THRESHOLD))
            fracs.append(frac_moved)
        mean_frac = float(np.mean(fracs)) if fracs else float("nan")
        out[node] = {
            "per_seed_fraction_moved": fracs,
            "mean_fraction_moved": mean_frac,
            "pass": bool(np.isfinite(mean_frac) and mean_frac >= ENG_EVENT_FRACTION),
        }
    return out


def gate_parity(
    all_nodes: dict[str, list[NodeResult]], bc_work_root: Path = BC_WORK_ROOT
) -> dict[int, dict[str, Any]]:
    """GATE PARITY (this driver's own check, distinct from prereg §3.3's correspondence_1d.py:1173
    claim, NOT re-litigated here): theta=(0,1) reproduces the banked bc seed's h=0.73 row.

    Compares this driver's truth-node ``combined_no_bh``/``combined_with_bh``
    values against the banked ``bc_<seed>_work/seed<seed>/simulations/
    diagnostics/event_likelihoods.csv`` at h=H_GEN, for the SAME event_idx
    values (valid because event_idx is assigned by row order and this
    driver's ``events`` truncation, if any, only ever DROPS trailing rows --
    see ``run_arm_seed_s0a``'s docstring / the build note).
    """
    out: dict[int, dict[str, Any]] = {}
    for r in all_nodes.get("truth", []):
        banked_csv = (
            bc_work_root
            / f"bc_{r.seed}_work"
            / f"seed{r.seed}"
            / "simulations"
            / "diagnostics"
            / "event_likelihoods.csv"
        )
        if not banked_csv.is_file():
            out[r.seed] = {"status": "NO_BANKED_CSV", "path": str(banked_csv)}
            continue
        banked = read_event_ln_l(banked_csv, H_GEN)
        merged = r.ln_l.merge(banked, on="event_idx", suffixes=("_driver", "_banked"))
        if merged.empty:
            out[r.seed] = {"status": "NO_OVERLAPPING_EVENTS"}
            continue
        diffs: dict[str, dict[str, Any]] = {}
        for chan in ("ln_L_no_bh", "ln_L_with_bh"):
            a = merged[f"{chan}_driver"].to_numpy(dtype=float)
            b = merged[f"{chan}_banked"].to_numpy(dtype=float)
            finite = np.isfinite(a) & np.isfinite(b)
            if finite.sum() == 0:
                diffs[chan] = {"n": 0}
                continue
            abs_diff = np.abs(a[finite] - b[finite])
            rel_diff = abs_diff / np.maximum(np.abs(b[finite]), 1e-300)
            diffs[chan] = {
                "n": int(finite.sum()),
                "max_abs_diff": float(np.max(abs_diff)),
                "max_rel_diff": float(np.max(rel_diff)),
                "exact": bool(np.max(abs_diff) == 0.0),
            }
        out[r.seed] = {
            "status": "COMPARED",
            "n_events_compared": len(merged),
            "diffs": diffs,
            "pass_exact": all(d.get("exact", False) for d in diffs.values() if d.get("n", 0) > 0),
            "pass_fallback_rtol": all(
                d.get("max_rel_diff", 1.0) <= PARITY_FALLBACK_RTOL
                for d in diffs.values()
                if d.get("n", 0) > 0
            ),
        }
    return out


def verdict_s0a(
    scores: dict[str, Any], eng: dict[str, Any], parity: dict[int, dict[str, Any]]
) -> dict[str, Any]:
    """Prereg §4.1 B0-A / B0-A' verdict line -- the RUNNER (not the builder) reads this off real data."""
    z_b = scores["ln_L_no_bh"]["score_b"]["Z"]
    z_s = scores["ln_L_no_bh"]["score_s"]["Z"]
    control_null = bool(
        np.isfinite(z_b)
        and np.isfinite(z_s)
        and abs(z_b) <= Z_THRESHOLD
        and abs(z_s) <= Z_THRESHOLD
    )
    return {
        "band": "B0-A" if control_null else "B0-A'",
        "verdict": "EXPECTED-NULL, instrument certified, proceed"
        if control_null
        else "INSTRUMENT-DEFECT -- STOP (prereg §4.5)",
        "Z_b": z_b,
        "Z_s": z_s,
        "threshold": Z_THRESHOLD,
        "channel": "ln_L_no_bh (registered primary; ln_L_with_bh reported alongside, see scores)",
        "gate_eng": eng,
        "gate_parity": parity,
        "reported_only": True,  # PA-HIER-28 item 9 = AFFORDABLE
    }


def verdict_s0r(scores: dict[str, Any], eng: dict[str, Any]) -> dict[str, Any]:
    """S0-R is a DISCLOSED NULL INSTRUMENT (PA-HIER-3/22) and PA-HIER-28 item 5 = FALLBACK
    disarms D7's early exit. This function NEVER emits B0-R/B0-R'/LEVER-DEAD-AT-N -- doing so
    would bank a verdict about an axis the amendment log proved this instrument cannot move.
    """
    return {
        "band": "NULL-INSTRUMENT (not B0-R/B0-R')",
        "verdict": (
            "S0-R AS CONSTRUCTED CANNOT BANK A LEVER VERDICT -- PA-HIER-3/22 proved "
            "sigma_kernel == sigma_realized identically at every sigma_scale (one shared "
            "GalaxyCatalogueHandler serves generator and estimator); truth-theta after the "
            "realize_observed_catalogue(sigma_scale=1.5) call is still (0,1), not (0,1.5). "
            "PA-HIER-28 item 5 (author ruling) DISARMS D7's early exit and re-scopes Stage 0 "
            "to S0-A + S0-C only. The numbers below are a disclosed diagnostic (do the "
            "z-mismatch and mass-mismatch scores at least land somewhere sane), never a "
            "registered read."
        ),
        "diagnostic_scores": scores,
        "gate_eng": eng,
        "reported_only": True,
    }


# ── Multi-seed orchestration with a bounded process budget ─────────────────


def _pin_worker_affinity(cpu_budget: int) -> None:
    """Pin this process to a ``cpu_budget``-sized CPU slice, DISJOINT from every other
    concurrent pool worker's slice.

    Root cause of the P0 crash (runner-disclosed 2026-08-29, see
    ``B1_2_DRIVER_EXTENSION_NOTE.md`` "Crash fix"): the previous version of this
    function ran INSIDE ``_run_one_seed_worker`` and computed
    ``all_cpus[:cpu_budget]`` independently in every worker -- with ``--jobs
    N>1`` every one of the ``N`` concurrent workers pinned itself to the SAME
    leading ``cpu_budget`` cores instead of disjoint slices, so ``N`` workers
    (each ALSO launching its own internal multiprocessing pool inside
    ``BayesianStatistics.evaluate``, sized off that artificially narrowed
    affinity mask) oversubscribed the same handful of cores N-fold. Under the
    registered P0 command (``--jobs 2 --total-cpu-budget 14``, cpu_budget=7)
    this reproducibly killed ``evaluate()`` partway through EVERY seed's FIRST
    node -- confirmed post-mortem: every ``node_truth_sites2.2_nosmear`` dir
    held only the early-written ``prepared_cramer_rao_bounds.csv``/
    ``selection_tables_h_*.json`` artifacts and no
    ``simulations/diagnostics/event_likelihoods.csv``, i.e. ``evaluate()``
    never returned for ANY seed's first node, each per-seed worker's
    ``except Exception`` therefore fired before the loop ever reached a second
    node, and ``run_arm`` was left with zero ``NodeResult`` for every node
    (not just the off-truth ones) -- which is what made
    ``compute_scores``'s ``pd.concat([])`` raise "No objects to concatenate".

    Used two ways, both keyed off ``multiprocessing.current_process().
    _identity`` (the 1-based ``(n,)`` slot the Pool machinery assigns each
    worker ONCE at spawn, stable for the worker's whole lifetime -- used here
    ONLY to pick a disjoint CPU slice, never anything statistic-facing):

    * ``--jobs 1`` (no ``Pool``, direct call from ``run_arm``'s list
      comprehension): ``_identity`` is empty, slot=0 -- BYTE-IDENTICAL to the
      pre-fix single leading-slice pin (``all_cpus[:cpu_budget]``).
    * ``--jobs N>1``: passed as ``ctx.Pool(..., initializer=
      _pin_worker_affinity, initargs=(cpu_per_job,))`` -- called ONCE per
      worker process, at Pool startup, before any task runs, while the OS
      affinity mask this process inherited from its parent is still the FULL
      original set (not yet narrowed by any prior pin), so each worker's
      disjoint slice is computed correctly regardless of task-dispatch order.
    """
    try:
        all_cpus = sorted(os.sched_getaffinity(0))
        budget = max(1, min(cpu_budget, len(all_cpus)))
        identity = mp.current_process()._identity  # noqa: SLF001 -- documented public contract
        slot = (identity[0] - 1) if identity else 0
        start = (slot * budget) % len(all_cpus)
        chosen = {all_cpus[(start + i) % len(all_cpus)] for i in range(budget)}
        os.sched_setaffinity(0, chosen)
    except (AttributeError, OSError):
        pass  # affinity control unavailable (e.g. non-Linux); proceed unpinned, disclosed


def _run_one_seed_worker(
    args: tuple[
        str,
        int,
        Path,
        tuple[str, ...],
        int | None,
        int,
        str,
        str,
        str,
        tuple[float, ...],
        float | None,
        str | None,
        str,
        float,
        str,
        str,
        float,
    ],
) -> Any:
    """Top-level (picklable) worker: run one seed's cells for the given arm.

    CPU affinity is pinned via :func:`_pin_worker_affinity` -- for ``--jobs
    1`` (no ``Pool``), directly here, since there is no ``Pool(initializer=
    ...)`` to have done it already; for ``--jobs N>1`` it was already pinned,
    disjointly, once per worker, by the ``Pool``'s own initializer (see
    :func:`_pin_worker_affinity`'s docstring) -- pinning again here would
    re-read the ALREADY-narrowed affinity mask and corrupt the disjoint slice,
    so it must not be repeated for the Pool-worker case.

    Args tuple extended (P1/KW-Q1, byte-identical when the 5 new trailing
    fields are theta_sites="all", smear="auto", config="b0i",
    h_values=(H_GEN,), score_h=None) with ``theta_sites``, ``smear``,
    ``config``, ``h_values``, ``score_h`` -- forwarded verbatim to
    run_arm_seed_s0a/s0r (S0-C ignores them, per its own "stays as is" scope).
    Extended again (T2.2, row #255 A10, byte-identical at the trailing
    ``None`` default) with ``candidate_dump_dir``, forwarded verbatim to
    run_arm_seed_s0a/s0r (S0-C ignores it, same scope note as above).
    Extended again (T1.2, row #255 tree 2 node T1.2, byte-identical at the
    trailing ``"off"``/``1.5`` defaults) with ``theta_phi_divisor``,
    ``sky_cone_k``, forwarded verbatim to run_arm_seed_s0a/s0r/run_seed_s0c
    (all three, unlike the two extensions above -- GATE T-ID makes the
    divisor a no-op at S0-C's truth-only theta, so unconditional forwarding
    is correct there too).
    Extended again (T2.3, row #255 tree 2 node T2.3, byte-identical at the
    trailing ``"off"`` default) with ``catalogue_leg_1d_mass_aware``,
    forwarded verbatim to run_arm_seed_s0a/s0r/run_seed_s0c (all three, same
    unconditional-forwarding rationale as ``theta_phi_divisor`` above --
    evaluate()'s own setup guard is the backstop).
    Extended again (T1.3-zwin, row #255 tree 2 node T1.3-zwin, byte-identical
    at the trailing ``"off"``/``1.0`` defaults) with ``theta_zwindow``,
    ``z_window_k``, forwarded verbatim to run_arm_seed_s0a/s0r/run_seed_s0c
    (all three -- GATE T-ID makes the theta-transformed window a no-op at
    S0-C's truth-only theta, so unconditional forwarding is correct there
    too).
    """
    (
        arm,
        seed,
        out_root,
        nodes,
        event_cap,
        cpu_budget,
        theta_sites,
        smear,
        config,
        h_values,
        score_h,
        candidate_dump_dir,
        theta_phi_divisor,
        sky_cone_k,
        catalogue_leg_1d_mass_aware,
        theta_zwindow,
        z_window_k,
    ) = args
    if not mp.current_process()._identity:  # noqa: SLF001 -- see docstring above
        _pin_worker_affinity(cpu_budget)
    try:
        if arm == "S0-A":
            results = run_arm_seed_s0a(
                seed,
                out_root,
                nodes,
                event_cap,
                theta_sites=theta_sites,
                smear=smear,
                config=config,
                h_values=h_values,
                score_h=score_h,
                candidate_dump_dir=candidate_dump_dir,
                theta_phi_divisor=theta_phi_divisor,
                sky_cone_k=sky_cone_k,
                catalogue_leg_1d_mass_aware=catalogue_leg_1d_mass_aware,
                theta_zwindow=theta_zwindow,
                z_window_k=z_window_k,
            )
        elif arm == "S0-R":
            results = run_arm_seed_s0r(
                seed,
                out_root,
                nodes,
                event_cap,
                theta_sites=theta_sites,
                smear=smear,
                config=config,
                h_values=h_values,
                score_h=score_h,
                candidate_dump_dir=candidate_dump_dir,
                theta_phi_divisor=theta_phi_divisor,
                sky_cone_k=sky_cone_k,
                catalogue_leg_1d_mass_aware=catalogue_leg_1d_mass_aware,
                theta_zwindow=theta_zwindow,
                z_window_k=z_window_k,
            )
        elif arm == "S0-C":
            return {
                "seed": seed,
                "s0c": run_seed_s0c(
                    seed,
                    out_root,
                    event_cap,
                    theta_phi_divisor=theta_phi_divisor,
                    sky_cone_k=sky_cone_k,
                    catalogue_leg_1d_mass_aware=catalogue_leg_1d_mass_aware,
                    theta_zwindow=theta_zwindow,
                    z_window_k=z_window_k,
                ),
            }
        else:
            raise ValueError(f"unknown arm {arm!r}")
        return {
            "seed": seed,
            "nodes": {
                r.node: {
                    "theta_b": r.theta_b,
                    "theta_s": r.theta_s,
                    "diag_csv": r.diag_csv,
                    "elapsed_s": r.elapsed_s,
                    "n_events": r.n_events,
                    "ln_l_records": r.ln_l.to_dict(orient="records"),
                }
                for r in results
            },
        }
    except Exception as exc:  # noqa: BLE001 -- surfaced to the parent as a structured failure
        return {
            "seed": seed,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }


def _records_to_node_results(seed: int, nodes_payload: dict[str, Any]) -> list[NodeResult]:
    out = []
    for node, payload in nodes_payload.items():
        ln_l = pd.DataFrame.from_records(payload["ln_l_records"])
        out.append(
            NodeResult(
                node=node,
                theta_b=payload["theta_b"],
                theta_s=payload["theta_s"],
                seed=seed,
                diag_csv=payload["diag_csv"],
                elapsed_s=payload["elapsed_s"],
                n_events=payload["n_events"],
                ln_l=ln_l,
            )
        )
    return out


def run_arm(
    arm: str,
    seeds: tuple[int, ...],
    out_root: Path,
    jobs: int,
    total_cpu_budget: int,
    nodes: tuple[str, ...],
    event_cap: int | None,
    theta_sites: str = "all",
    smear: str = "auto",
    config: str = "b0i",
    h_values: tuple[float, ...] = (H_GEN,),
    score_h: float | None = None,
    candidate_dump_dir: str | None = None,
    theta_phi_divisor: str = "off",
    sky_cone_k: float = 1.5,
    catalogue_leg_1d_mass_aware: str = "off",
    theta_zwindow: str = "off",
    z_window_k: float = 1.0,
) -> dict[str, Any]:
    out_root.mkdir(parents=True, exist_ok=True)
    jobs = max(1, min(jobs, len(seeds)))
    cpu_per_job = max(1, total_cpu_budget // jobs)
    task_args = [
        (
            arm,
            seed,
            out_root,
            nodes,
            event_cap,
            cpu_per_job,
            theta_sites,
            smear,
            config,
            h_values,
            score_h,
            candidate_dump_dir,
            theta_phi_divisor,
            sky_cone_k,
            catalogue_leg_1d_mass_aware,
            theta_zwindow,
            z_window_k,
        )
        for seed in seeds
    ]

    t0 = time.time()
    if jobs == 1:
        raw_results = [_run_one_seed_worker(a) for a in task_args]
    else:
        ctx = mp.get_context("spawn")
        # initializer=_pin_worker_affinity: pin each worker's CPU slice ONCE,
        # disjointly, at Pool startup (see _pin_worker_affinity's docstring --
        # this is the runner-disclosed P0 crash fix, B1_2_DRIVER_EXTENSION_
        # NOTE.md "Crash fix"). Passing cpu_per_job here (not via task_args)
        # is what makes it "once per worker" rather than "once per task".
        with ctx.Pool(
            processes=jobs, initializer=_pin_worker_affinity, initargs=(cpu_per_job,)
        ) as pool:
            raw_results = pool.map(_run_one_seed_worker, task_args)
    wall_s = time.time() - t0

    errors = [r for r in raw_results if "error" in r]
    ok = [r for r in raw_results if "error" not in r]
    # Surface swallowed per-seed exceptions IMMEDIATELY -- previously these
    # tracebacks only ever reached a Python dict (payload["errors"]) that, if
    # compute_scores below raised first (e.g. from a genuinely empty node
    # pool), was NEVER written to disk or printed: the top-level exception
    # killed the process before out_json.write_text() ran, so the real
    # underlying cause was invisible in the runner's log (runner-disclosed
    # P0 crash, B1_2_DRIVER_EXTENSION_NOTE.md "Crash fix").
    for _err in errors:
        print(f"[{arm} seed={_err['seed']}] WORKER ERROR: {_err['error']}", flush=True)
        if _err.get("traceback"):
            print(_err["traceback"], flush=True)

    payload: dict[str, Any] = {
        "arm": arm,
        "seeds_requested": list(seeds),
        "jobs": jobs,
        "cpu_per_job": cpu_per_job,
        "wall_s": wall_s,
        "n_seeds_ok": len(ok),
        "n_seeds_error": len(errors),
        "errors": errors,
        "theta_sites": theta_sites,
        "smear": smear,
        "config": config,
        "h_values": list(h_values),
        "score_h": score_h,
        "theta_phi_divisor": theta_phi_divisor,
        "sky_cone_k": sky_cone_k,
        "catalogue_leg_1d_mass_aware": catalogue_leg_1d_mass_aware,
        "theta_zwindow": theta_zwindow,
        "z_window_k": z_window_k,
        "node_dir_suffix": _node_dir_suffix(
            theta_sites,
            smear,
            config,
            theta_phi_divisor,
            sky_cone_k,
            catalogue_leg_1d_mass_aware,
            theta_zwindow,
            z_window_k,
        ),
    }

    if arm == "S0-C":
        payload["per_seed"] = [r["s0c"] for r in ok]
        return payload

    all_nodes: dict[str, list[NodeResult]] = {n: [] for n in nodes}
    for r in ok:
        for node, node_results in _grouped_by_node(r["seed"], r["nodes"]).items():
            all_nodes[node].extend(node_results)

    payload["per_seed_summary"] = [
        {
            "seed": r["seed"],
            "nodes": {
                n: {"elapsed_s": p["elapsed_s"], "n_events": p["n_events"]}
                for n, p in r["nodes"].items()
            },
        }
        for r in ok
    ]

    # Per-axis (b, s) requested/produced check (T1.3-zwin, row #255): the
    # registered P1 arm's own node list is {truth, s_plus, s_minus} -- NO
    # b-nodes (PHYSICS_CHANGE_THETA_ZWINDOW_20260830.md §5.6, "b-axis: NOT
    # re-run under P1"). The pre-T1.3-zwin "all 4 or nothing" gate could
    # never score that arm at all. Relaxed to per-axis, mirroring
    # compute_scores' own has_b/has_s split: an axis is ready only if BOTH
    # its nodes were requested AND both produced >=1 NodeResult (the
    # runner-disclosed P0 crash-fix check, B1_2_DRIVER_EXTENSION_NOTE.md
    # "Crash fix", now per axis instead of all-four).
    def _axis_ready(pair: tuple[str, str]) -> bool:
        return all(n in nodes for n in pair) and all(len(all_nodes.get(n, [])) > 0 for n in pair)

    b_ready = _axis_ready(("b_plus", "b_minus"))
    s_ready = _axis_ready(("s_plus", "s_minus"))
    any_axis_requested = any(n in nodes for n in ("b_plus", "b_minus", "s_plus", "s_minus"))

    if b_ready or s_ready:
        scores = compute_scores(all_nodes, seeds=seeds)
        eng = gate_eng(all_nodes)
        payload["scores"] = scores
        payload["gate_eng"] = eng
        if not (b_ready and s_ready):
            payload["note"] = (
                f"nodes={nodes}: only the {'b' if b_ready else 's'}-axis is ready "
                f"(b_ready={b_ready}, s_ready={s_ready}) -- the OTHER axis's score in "
                "payload['scores'] is unavailable (n_pooled=0/NaN), by design, NOT a crash. "
                "This is the registered P1 arm's own node list (T1.3-zwin gate doc §5.6, "
                "b-axis NOT re-run) when only s_ready is True."
            )
        if arm == "S0-A":
            parity = gate_parity(all_nodes)
            payload["gate_parity"] = parity
            payload["verdict"] = verdict_s0a(scores, eng, parity)
        elif arm == "S0-R":
            payload["verdict"] = verdict_s0r(scores, eng)
    elif not any_axis_requested:
        payload["note"] = (
            f"nodes={nodes} does not include a complete b-axis (b_plus,b_minus) or s-axis "
            "(s_plus,s_minus) pair -- scores/gates/verdict need at least one; this is expected "
            "for a --smoke run with 1-2 nodes and is NOT a registered read."
        )
        if "truth" in nodes and arm == "S0-A":
            # Still run GATE PARITY if the truth node is present -- it needs no other node.
            payload["gate_parity"] = gate_parity(all_nodes)
    else:
        # At least one axis WAS fully requested but produced ZERO NodeResult
        # for at least one of its two nodes across every seed -- every
        # seed's worker for that node either raised (see payload["errors"] /
        # the WORKER ERROR lines printed above) or the arm otherwise never
        # reached it. Report exactly what's missing instead of letting
        # compute_scores crash.
        have = {n: len(all_nodes.get(n, [])) for n in ("b_plus", "b_minus", "s_plus", "s_minus")}
        missing_pairs = [
            (seed, n)
            for n in ("b_plus", "b_minus", "s_plus", "s_minus")
            if n in nodes
            for seed in seeds
            if seed not in {r.seed for r in all_nodes.get(n, [])}
        ]
        payload["note"] = (
            f"a requested axis produced ZERO results for at least one of its two nodes "
            f"(n_present_by_node={have}); missing (seed, node) pairs: {missing_pairs}. This is "
            f"NOT the --smoke-subset case: n_seeds_error={len(errors)} of {len(seeds)} seeds "
            "errored (see the WORKER ERROR lines printed above and payload['errors'] for the "
            "real per-seed tracebacks). scores/gate_eng/verdict are skipped rather than raised "
            "from an empty pool."
        )
        if "truth" in nodes and arm == "S0-A" and all_nodes.get("truth"):
            payload["gate_parity"] = gate_parity(all_nodes)

    return payload


def _grouped_by_node(seed: int, nodes_payload: dict[str, Any]) -> dict[str, list[NodeResult]]:
    out: dict[str, list[NodeResult]] = {}
    for r in _records_to_node_results(seed, nodes_payload):
        out.setdefault(r.node, []).append(r)
    return out


# ── --score-only: pooled score_b/score_s/Z_b/Z_s from on-disk node outputs,
# NO evaluation (P0 completion, SYNTHESIS_DOCKET_1_20260829.md sec 2 B1 P0) ──


def gather_node_results_from_disk(
    arm: str,
    seeds: tuple[int, ...],
    out_root: Path,
    nodes: tuple[str, ...],
    theta_sites: str,
    smear: str,
    config: str,
    score_h: float,
    theta_phi_divisor: str = "off",
    sky_cone_k: float = 1.5,
    catalogue_leg_1d_mass_aware: str = "off",
    theta_zwindow: str = "off",
    z_window_k: float = 1.0,
) -> tuple[dict[str, list[NodeResult]], list[str]]:
    """Read ``event_likelihoods.csv`` for every requested (seed, node) pair
    directly off disk -- NO ``evaluate()`` call, NO venue construction. Used
    by ``--score-only`` to compute the pooled prereg §4.1 statistic from
    whatever nodes/seeds a prior (possibly partial, possibly multi-invocation)
    run already banked, e.g. "seed 900101 nodes b_minus,s_plus,s_minus" run
    separately from "truth,b_plus" (P0's "remaining nodes" case) -- each
    invocation writes fresh node dirs (``run_theta_node`` always calls
    ``evaluate()``; nothing here is a stale reuse), and this function simply
    unions whatever is present at scoring time.

    PA-HIER-32(d): also reads each seed's ``es_null_det.csv`` cache (written
    by :func:`run_arm_seed_s0a`/:func:`run_arm_seed_s0r` at
    ``<seed work_root>/es_null_det.csv``, node-independent -- NO venue
    reconstruction needed here either) and merges it into every node's
    ``ln_l`` so :func:`compute_scores` can compute the corrected
    ``score_s = score_lns - Es_null_det``. Missing for a given seed (e.g. an
    older banked run predating this cache) degrades gracefully: that seed's
    events simply have no ``es_null_det`` value and are excluded from the
    corrected statistic (:func:`compute_scores` reports
    ``score_s_available``); ``score_s_raw`` is unaffected.

    Returns ``(all_nodes, missing_paths)`` -- ``missing_paths`` lists every
    (seed, node) CSV that was requested but not found on disk (reported, not
    fatal: :func:`score_only_payload` computes whatever the union of present
    nodes allows and states plainly what could not be pooled).
    """
    prefix = {"S0-A": "s0a_seed", "S0-R": "s0r_seed"}.get(arm)
    if prefix is None:
        raise ValueError(
            f"--score-only supports S0-A/S0-R only (no node cross to pool for {arm!r})"
        )
    suffix = _node_dir_suffix(
        theta_sites,
        smear,
        config,
        theta_phi_divisor,
        sky_cone_k,
        catalogue_leg_1d_mass_aware,
        theta_zwindow,
        z_window_k,
    )
    all_nodes: dict[str, list[NodeResult]] = {n: [] for n in nodes}
    missing: list[str] = []
    for seed in seeds:
        es_null_det = _read_es_null_det_cache(out_root / f"{prefix}{seed}")
        for node in nodes:
            theta_b, theta_s = THETA_NODES[node]
            diag_csv = (
                out_root
                / f"{prefix}{seed}"
                / f"node_{node}{suffix}"
                / "simulations"
                / "diagnostics"
                / "event_likelihoods.csv"
            )
            if not diag_csv.is_file():
                missing.append(str(diag_csv))
                continue
            ln_l = read_event_ln_l(diag_csv, score_h, es_null_det=es_null_det)
            all_nodes[node].append(
                NodeResult(
                    node=node,
                    theta_b=theta_b,
                    theta_s=theta_s,
                    seed=seed,
                    diag_csv=str(diag_csv),
                    elapsed_s=float(
                        "nan"
                    ),  # not measured -- no evaluation happened this invocation
                    n_events=len(ln_l),
                    ln_l=ln_l,
                )
            )
    return all_nodes, missing


def score_only_payload(
    arm: str,
    seeds: tuple[int, ...],
    nodes: tuple[str, ...],
    all_nodes: dict[str, list[NodeResult]],
    missing: list[str],
) -> dict[str, Any]:
    """Build the same scores/gate_eng/gate_parity/verdict payload
    :func:`run_arm` computes, from disk-gathered ``all_nodes`` -- reuses
    :func:`compute_scores`/:func:`gate_eng`/:func:`gate_parity`/
    :func:`verdict_s0a`/:func:`verdict_s0r` verbatim (the statistic is
    IDENTICAL whether the ``ln_l`` frames came from a fresh ``evaluate()``
    call or an on-disk CSV -- both are the same columns read the same way).
    """
    payload: dict[str, Any] = {
        "arm": arm,
        "seeds_requested": list(seeds),
        "nodes_requested": list(nodes),
        "score_only": True,
        "n_present_by_node": {n: len(v) for n, v in all_nodes.items()},
        "seeds_present_by_node": {n: sorted({r.seed for r in v}) for n, v in all_nodes.items()},
        "n_missing_csv": len(missing),
        "missing_csv_paths": missing,
    }
    # Per-axis ready check (T1.3-zwin, row #255) -- see run_arm's identical
    # relaxation for the rationale (the registered P1 arm's node list has no
    # b-nodes at all).
    b_ready = all(len(all_nodes.get(n, [])) > 0 for n in ("b_plus", "b_minus"))
    s_ready = all(len(all_nodes.get(n, [])) > 0 for n in ("s_plus", "s_minus"))
    if b_ready or s_ready:
        scores = compute_scores(all_nodes, seeds=seeds)
        eng = gate_eng(all_nodes)
        payload["scores"] = scores
        payload["gate_eng"] = eng
        if not (b_ready and s_ready):
            payload["note"] = (
                f"only the {'b' if b_ready else 's'}-axis is ready on disk (b_ready={b_ready}, "
                f"s_ready={s_ready}) -- the OTHER axis's score in payload['scores'] is "
                "unavailable (n_pooled=0/NaN), by design, NOT an error."
            )
        if len(all_nodes.get("truth", [])) > 0:
            parity = gate_parity(all_nodes)
            payload["gate_parity"] = parity
            if arm == "S0-A":
                payload["verdict"] = verdict_s0a(scores, eng, parity)
        if arm == "S0-R":
            payload["verdict"] = verdict_s0r(scores, eng)
    else:
        have = {n: len(all_nodes.get(n, [])) for n in ("b_plus", "b_minus", "s_plus", "s_minus")}
        payload["note"] = (
            "on-disk node set is INCOMPLETE for pooling -- scores/gate_eng/verdict require "
            f">=1 seed present for EACH node of AT LEAST ONE axis (b_plus AND b_minus, OR "
            f"s_plus AND s_minus); have {have}. "
            "This is not an error: run the remaining (seed, node) combinations, then re-invoke "
            "--score-only."
        )
    return payload


def write_score_markdown(payload: dict[str, Any], md_path: Path) -> None:
    """Render :func:`score_only_payload`'s (or :func:`run_arm`'s) output as a
    short human-readable markdown summary -- the "per-arm score ... md" P0
    deliverable (SYNTHESIS_DOCKET_1_20260829.md sec 2 B1 P0)."""
    lines = [
        f"# {payload['arm']} pooled score (prereg §4.1) -- {'score-only, zero-compute read' if payload.get('score_only') else 'from a live run'}",
        "",
        f"Seeds requested: {payload.get('seeds_requested')}",
        f"Nodes requested: {payload.get('nodes_requested', payload.get('nodes'))}",
    ]
    if "n_present_by_node" in payload:
        lines.append(f"Nodes present on disk (n seeds each): {payload['n_present_by_node']}")
    if payload.get("missing_csv_paths"):
        lines.append(
            f"Missing CSVs ({payload['n_missing_csv']}): first 5 -> {payload['missing_csv_paths'][:5]}"
        )
    lines.append("")
    if "scores" in payload:
        for channel, d in payload["scores"].items():
            lines.append(f"## {channel}")
            # "score_s" (PA-HIER-32(d), CORRECTED -- the band-evaluated
            # primary) first; "score_s_raw" (the OLD/superseded raw linear
            # secant) and "score_lns" (the intermediate ln-s-centred secant
            # before the Es_null_det correction) alongside for continuity.
            for stat_name in ("score_b", "score_s", "score_s_raw", "score_lns"):
                if stat_name not in d:
                    continue
                s = d[stat_name]
                lines.append(
                    f"- {stat_name}: mean={s['mean']!r} sem={s['sem']!r} Z={s['Z']!r} n_pooled={s['n_pooled']}"
                )
            if "score_b_available" in d:
                lines.append(
                    f"- score_b_available (b-axis nodes present): {d['score_b_available']}"
                )
            if "score_s_available" in d:
                lines.append(
                    f"- score_s_available (Es_null_det cache found): {d['score_s_available']}"
                )
            lines.append("")
    if "verdict" in payload:
        v = payload["verdict"]
        lines.append(f"## Verdict: band={v.get('band')!r}")
        lines.append(f"{v.get('verdict')}")
        lines.append("")
    if "gate_eng" in payload:
        lines.append("## GATE ENG (mean fraction of events moved >=1e-6 rel, per node)")
        for node, d in payload["gate_eng"].items():
            lines.append(
                f"- {node}: mean_fraction_moved={d['mean_fraction_moved']!r} pass={d['pass']}"
            )
        lines.append("")
    if "gate_parity" in payload:
        lines.append("## GATE PARITY (truth node vs banked bc CSV, per seed)")
        for seed, d in payload["gate_parity"].items():
            lines.append(f"- seed {seed}: {d.get('status')}, pass_exact={d.get('pass_exact')}")
        lines.append("")
    if payload.get("note"):
        lines.append(f"**Note:** {payload['note']}")
        lines.append("")
    md_path.write_text("\n".join(lines))


# ── CLI ──────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--arm", required=True, choices=("S0-A", "S0-R", "S0-C"))
    ap.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated seeds; default is the first 4 banked bc seeds "
        f"{DEFAULT_BC_SEEDS} (S0-C uses only the first of these unless overridden).",
    )
    ap.add_argument(
        "--out-root",
        type=str,
        default=str(REALISTIC_DIR / "fanout1_20260829" / "hier_s0_work"),
    )
    ap.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Mirror cells (seeds) run concurrently; choose so jobs * cpu-per-job <= --total-cpu-budget.",
    )
    ap.add_argument(
        "--total-cpu-budget",
        type=int,
        default=14,
        help="Total CPUs this driver may use across all concurrent cells (repo convention: leave 2 free of nproc).",
    )
    ap.add_argument(
        "--nodes",
        type=str,
        default=None,
        help="Comma-separated theta-node names (subset of truth,b_plus,b_minus,s_plus,s_minus); "
        "default is all 5 for S0-A/S0-R (ignored for S0-C, which is always the truth node).",
    )
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="1 seed, a small node subset (default: truth,b_plus), and a tiny event cap. "
        "Builder-only per rule 2 -- never bank a --smoke output as the registered measurement.",
    )
    ap.add_argument(
        "--event-cap",
        type=int,
        default=None,
        help="Truncate each seed's realized events to the first N rows (order-preserving, so "
        "event_idx values remain comparable to the banked full-N CSVs for GATE PARITY). "
        "--smoke implies a small cap unless this is set explicitly.",
    )
    ap.add_argument(
        "--theta-sites",
        type=str,
        default="all",
        choices=THETA_SITES_CHOICES,
        help="Forwarded to run_mirror_seed_inprocess's theta_sites kwarg (P1 equivalence gate, "
        "SYNTHESIS_DOCKET_1_20260829.md sec 2 B1 P1). Default 'all' is BYTE-IDENTICAL to every "
        "pre-P1 invocation of this driver. '2.1'/'2.2' isolate the per-host numerator sites "
        "(never require smearing); '2.3' isolates the global-selection denominator site (always "
        "requires --smear on/auto, since it IS the smeared table).",
    )
    ap.add_argument(
        "--smear",
        type=str,
        default="auto",
        choices=SMEAR_CHOICES,
        help="'auto' (default) reproduces this driver's ORIGINAL dispatch exactly at "
        "--theta-sites all (smear_global_selection = theta_engaged) -- BYTE-IDENTICAL default. "
        "'on'/'off' force the flag. 'off' is REFUSED at parse time (see below) if --theta-sites "
        "is 'all' or '2.3' and any requested node is theta-engaged (evaluate()'s own guard would "
        "otherwise raise mid-run, after paying the venue setup cost).",
    )
    ap.add_argument(
        "--config",
        type=str,
        default="b0i",
        choices=CONFIG_CHOICES,
        help="'b0i' (default, BYTE-IDENTICAL) is this driver's original hardcoded bc/b0i venue "
        "(hier_s0_driver.py:94-97-era flags). 'ft' is the KW-Q1/B4.2 venue (catalogue_numerator_"
        "survival='phi', fused, HEAD Sigma^phi, host_mode='population_selected' -- copied "
        "EXACTLY from p3_twin_test.py's fusedarm/--survival phi stage). Applies to S0-A/S0-R "
        "only (S0-C stays 'b0i', per its own registered costing-probe scope).",
    )
    ap.add_argument(
        "--h-nodes",
        type=str,
        default=None,
        help="Comma-separated h values fused into ONE evaluate() call per theta node (S0-A/S0-R "
        "only; S0-C's H_GRID_41 sweep is unaffected). Default (unset) is the single H_GEN=0.73 "
        "node -- BYTE-IDENTICAL. KW-Q1 (B4.2) uses '0.725,0.735'.",
    )
    ap.add_argument(
        "--score-h",
        type=float,
        default=None,
        help="Which evaluated h this driver's OWN internal ln-L readback (compute_scores/"
        "gate_eng/gate_parity) uses, when --h-nodes does not include H_GEN=0.73 (e.g. KW-Q1's "
        "0.725/0.735 grid). Default (unset) resolves to H_GEN if present in --h-nodes, else the "
        "first --h-nodes value -- see _resolve_score_h. Irrelevant/unused for the default "
        "--h-nodes (single H_GEN node).",
    )
    ap.add_argument(
        "--candidate-dump",
        type=str,
        default=None,
        dest="candidate_dump",
        help="T2.2 (row #255 tree 2 node T2.2; A10 = instrumentation guard, not a physics "
        "gate; B4_3_MIXTURE_WEIGHT_DERIVATION_20260830.md sec 6). Directory root for the "
        "per-(event, candidate) diagnostic dump, forwarded to BayesianStatistics.evaluate()'s "
        "candidate_dump_dir kwarg (per-seed-per-node subdirectories are created underneath). "
        "Default (unset) is None -- BYTE-IDENTICAL, no dump files written (GATE BI). S0-C "
        "ignores this flag (out of T2.2's registered scope).",
    )
    ap.add_argument(
        "--theta-phi-divisor",
        type=str,
        default="off",
        choices=("off", "on"),
        dest="theta_phi_divisor",
        help="T1.2 (row #255 tree 2 node T1.2, driver-gap fix for T1.1; "
        "PHYSICS_CHANGE_THETA_DIVISOR_20260830.md sec 2.2). Forwarded to run_mirror_seed_"
        "inprocess's theta_phi_divisor kwarg (site 2.3phi theta-consistent no-BH divisor). "
        "Default 'off' is BYTE-IDENTICAL to every pre-T1.1 invocation of this driver. 'on' "
        "requires catalogue_global_selection to resolve to 'phi' under normalization_mode="
        "'absolute_marginal' (evaluate()'s own guard) -- true for both --config b0i and ft at "
        "their defaults. Forwarded unconditionally to every node/arm, including the truth node "
        "and S0-C (GATE T-ID: the divisor is theta-consistent, so it is a no-op at theta=(0,1)).",
    )
    ap.add_argument(
        "--sky-cone-k",
        type=float,
        default=1.5,
        dest="sky_cone_k",
        help="T1.2 (same reference, sec 2.5). Forwarded to run_mirror_seed_inprocess's "
        "sky_cone_k kwarg (sky-cone-radius instrument, must be finite and > 0). Default 1.5 is "
        "BYTE-IDENTICAL to the pre-flag sigma_multiplier literal.",
    )
    ap.add_argument(
        "--theta-zwindow",
        type=str,
        default="off",
        choices=("off", "on"),
        dest="theta_zwindow",
        help="T1.3-zwin (row #255 tree 2 node T1.3-zwin; "
        "PHYSICS_CHANGE_THETA_ZWINDOW_20260830.md sec 2.2). Forwarded to run_mirror_seed_"
        "inprocess's theta_zwindow kwarg (theta-consistent candidate z-window instrument). "
        "Default 'off' is BYTE-IDENTICAL to every pre-T1.3-zwin invocation of this driver. 'on' "
        "replaces the galaxy-side centre/width of the candidate z-filter by the theta-"
        "transformed site-2.2 kernel; INDEPENDENT of --theta-sites and --theta-phi-divisor. At "
        "theta=(0,1) the literal skip applies (GATE T-ID). Forwarded unconditionally to every "
        "node/arm, including the truth node and S0-C.",
    )
    ap.add_argument(
        "--z-window-k",
        type=float,
        default=1.0,
        dest="z_window_k",
        help="T1.3-zwin (same reference, sec 2.2). Forwarded to run_mirror_seed_inprocess's "
        "z_window_k kwarg (candidate z-window half-width, must be finite and > 0). Default 1.0 "
        "is BYTE-IDENTICAL to today's implicit +/- 1 sigma_g literal. The registered decisive "
        "arm (P1) uses 4.0 (= site 2.2's own integration_limit_sigma_multiplier).",
    )
    ap.add_argument(
        "--catalogue-leg-1d-mass-aware",
        type=str,
        default="off",
        choices=("off", "on"),
        dest="catalogue_leg_1d_mass_aware",
        help="T2.3 (row #255 tree 2 node T2.3; "
        "PHYSICS_CHANGE_MASS_AWARE_1D_LEG_20260830.md sec 2). Forwarded to run_mirror_seed_"
        "inprocess's catalogue_leg_1d_mass_aware kwarg (mass-aware 1D catalogue leg instrument: "
        "S_4D(d_L(z;h), M_g(1+z)) replaces S_bar_phi(z;h) in the WITHOUT-BH catalogue numerator, "
        "Sigma_4D replaces Sigma^phi as the global divisor, alpha_G_phi replaces beta_G_phi as "
        "the mixture weight). Default 'off' is BYTE-IDENTICAL to every pre-T2.3 invocation of "
        "this driver. 'on' requires evaluate()'s own resolved catalogue_numerator_survival AND "
        "catalogue_global_selection to both be 'phi' and theta_phi_divisor='off' -- true for "
        "--config ft (FT_CATALOGUE_NUMERATOR_SURVIVAL='phi'), NOT true for the default --config "
        "b0i (BC_CATALOGUE_NUMERATOR_SURVIVAL='off'; evaluate() raises there). Forwarded "
        "unconditionally to every node/arm, including the truth node and S0-C (not a production "
        "posterior).",
    )
    ap.add_argument(
        "--score-only",
        action="store_true",
        help="P0 completion (SYNTHESIS_DOCKET_1_20260829.md sec 2 B1 P0): compute the pooled "
        "prereg §4.1 score_b/score_s/Z_b/Z_s/GATE ENG/GATE PARITY/verdict from event_likelihoods."
        "csv files ALREADY ON DISK under --out-root (matching --seeds/--nodes/--theta-sites/"
        "--smear/--config/--theta-phi-divisor/--sky-cone-k/--theta-zwindow/--z-window-k, for "
        "locating the right node directories) -- NO evaluate() call, NO venue construction. "
        "Also reads each seed's es_null_det.csv cache (PA-HIER-32(d)) if present, so the "
        "corrected score_s is emitted whenever the run that produced the CSVs also cached it; "
        "score_s_raw is always emitted regardless. S0-A/S0-R only. Writes "
        "<arm>_score_output.json and <arm>_score.md.",
    )
    args = ap.parse_args()

    out_root = Path(args.out_root)

    seeds: tuple[int, ...]
    nodes: tuple[str, ...]
    if args.smoke:
        seeds = (int(args.seeds.split(",")[0]) if args.seeds else DEFAULT_BC_SEEDS[0],)
        nodes = tuple(args.nodes.split(",")) if args.nodes else ("truth", "b_plus")
        event_cap = args.event_cap if args.event_cap is not None else 12
        jobs = 1
    else:
        seeds = tuple(int(x) for x in args.seeds.split(",")) if args.seeds else DEFAULT_BC_SEEDS
        nodes = tuple(args.nodes.split(",")) if args.nodes else NODE_ORDER
        event_cap = args.event_cap
        jobs = args.jobs

    for n in nodes:
        if n not in THETA_NODES:
            raise SystemExit(f"unknown node {n!r}; must be one of {sorted(THETA_NODES)}")

    h_values: tuple[float, ...] = (
        tuple(float(x) for x in args.h_nodes.split(",")) if args.h_nodes else (H_GEN,)
    )
    if not h_values:
        raise SystemExit("--h-nodes must not resolve to an empty grid")

    # CLI-level validation (clearer than evaluate()'s own mid-run ValueError,
    # and fires BEFORE the (expensive) venue setup): --smear off is
    # incompatible with --theta-sites all/2.3 whenever any requested node is
    # theta-engaged (every THETA_NODES entry except "truth").
    any_theta_engaged_node = any(n != "truth" for n in nodes)
    if args.smear == "off" and args.theta_sites in ("all", "2.3") and any_theta_engaged_node:
        raise SystemExit(
            f"--smear off is incompatible with --theta-sites {args.theta_sites!r} while "
            f"--nodes={nodes} includes a theta-engaged node (all THETA_NODES except 'truth') -- "
            "evaluate() REQUIRES smear_global_selection=True whenever theta is engaged and "
            "theta_sites includes '2.3'/'all' (bayesian_statistics.py's theta_sites guard). "
            "Pass --theta-sites 2.1 or 2.2 together with --smear off, or drop --smear off."
        )

    if args.score_only:
        if args.arm == "S0-C":
            raise SystemExit("--score-only is not supported for --arm S0-C (no node cross to pool)")
        score_h = _resolve_score_h(h_values, args.score_h)
        all_nodes, missing = gather_node_results_from_disk(
            args.arm,
            seeds,
            out_root,
            nodes,
            theta_sites=args.theta_sites,
            smear=args.smear,
            config=args.config,
            score_h=score_h,
            theta_phi_divisor=args.theta_phi_divisor,
            sky_cone_k=args.sky_cone_k,
            catalogue_leg_1d_mass_aware=args.catalogue_leg_1d_mass_aware,
            theta_zwindow=args.theta_zwindow,
            z_window_k=args.z_window_k,
        )
        result = score_only_payload(args.arm, seeds, nodes, all_nodes, missing)
        result["theta_sites"] = args.theta_sites
        result["smear"] = args.smear
        result["config"] = args.config
        result["h_values"] = list(h_values)
        result["score_h"] = score_h
        result["theta_phi_divisor"] = args.theta_phi_divisor
        result["sky_cone_k"] = args.sky_cone_k
        result["catalogue_leg_1d_mass_aware"] = args.catalogue_leg_1d_mass_aware
        result["theta_zwindow"] = args.theta_zwindow
        result["z_window_k"] = args.z_window_k
        result["registration"] = str(REGISTRATION)
        out_root.mkdir(parents=True, exist_ok=True)
        out_json = out_root / f"{args.arm.lower().replace('-', '')}_score_output.json"
        out_json.write_text(json.dumps(result, indent=2, default=str))
        out_md = out_root / f"{args.arm.lower().replace('-', '')}_score.md"
        write_score_markdown(result, out_md)
        print(f"wrote {out_json}")
        print(f"wrote {out_md}")
        print(
            json.dumps(
                {k: v for k, v in result.items() if k != "missing_csv_paths"}, indent=2, default=str
            )[:4000]
        )
        return 0 if not missing or result.get("scores") else 1

    result = run_arm(
        args.arm,
        seeds,
        out_root,
        jobs,
        args.total_cpu_budget,
        nodes,
        event_cap,
        theta_sites=args.theta_sites,
        smear=args.smear,
        config=args.config,
        h_values=h_values,
        score_h=args.score_h,
        candidate_dump_dir=args.candidate_dump,
        theta_phi_divisor=args.theta_phi_divisor,
        sky_cone_k=args.sky_cone_k,
        catalogue_leg_1d_mass_aware=args.catalogue_leg_1d_mass_aware,
        theta_zwindow=args.theta_zwindow,
        z_window_k=args.z_window_k,
    )
    result["smoke"] = bool(args.smoke)
    result["event_cap"] = event_cap
    result["registration"] = str(REGISTRATION)

    out_root.mkdir(parents=True, exist_ok=True)
    tag = "smoke" if args.smoke else "full"
    out_json = out_root / f"{args.arm.lower().replace('-', '')}_{tag}_output.json"
    out_json.write_text(json.dumps(result, indent=2, default=str))
    print(f"wrote {out_json}")
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ("per_seed_summary",)},
            indent=2,
            default=str,
        )[:4000]
    )
    return 0 if not result.get("errors") else 1


if __name__ == "__main__":
    raise SystemExit(main())
