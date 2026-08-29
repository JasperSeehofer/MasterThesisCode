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
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]
REALISTIC_DIR = REPO_ROOT / "results/campaign51_20260728/realistic_20260729"
BC_WORK_ROOT = REALISTIC_DIR / "p3_b0_work"
REGISTRATION = REALISTIC_DIR / "PREREGISTRATION_HIER_HTHETA_20260826.md"

import darksiren_emri.validation.correspondence_1d as c1d  # noqa: E402
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


def build_ft_venue(work_root: Path, seed: int, sigma_z_scale: float = 1.0) -> tuple[pd.DataFrame, Any]:
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


def _build_venue(config: str, work_root: Path, seed: int, sigma_z_scale: float) -> tuple[pd.DataFrame, Any]:
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


def _node_dir_suffix(theta_sites: str, smear: str, config: str) -> str:
    """Node output-directory suffix encoding the P1/KW-Q1 variant, so a
    non-default run never overwrites another variant's (or the default's)
    banked node outputs. Byte-identical default (``theta_sites="all"``,
    ``smear="auto"``, ``config="b0i"``) -> empty suffix -> the ORIGINAL
    ``node_<name>`` paths, unchanged.
    """
    parts: list[str] = []
    if config != "b0i":
        parts.append(config)
    if theta_sites != "all":
        parts.append(f"sites{theta_sites}")
    if smear != "auto":
        parts.append("smearon" if smear == "on" else "nosmear")
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
        h_values=h_values,
        h_bounds=H_BOUNDS,
        theta_b=theta_b,
        theta_s=theta_s,
        theta_sites=theta_sites,
        smear_global_selection=smear_flag,
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
            **common_kwargs,
        )
    else:
        raise ValueError(f"config must be one of {CONFIG_CHOICES}, got {config!r}")
    return diag_csv, elapsed


# ── Diagnostics readback ───────────────────────────────────────────────────


def read_event_ln_l(diag_csv: Path, h: float, rtol: float = 1e-9) -> pd.DataFrame:
    """Read ``event_likelihoods.csv`` and return per-event ln L at ``h``.

    Columns returned: ``event_idx``, ``ln_L_no_bh``, ``ln_L_with_bh`` (NaN
    where the corresponding ``combined_*`` column is non-positive -- the
    same non-positivity the estimator's own ``num_log_term_*`` diagnostic
    columns guard against, bayesian_statistics.py:5788-5794).
    """
    df = pd.read_csv(diag_csv)
    mask = np.isclose(df["h"].to_numpy(dtype=float), h, rtol=rtol, atol=1e-12)
    sub = df.loc[mask, ["event_idx", *DIAG_VALUE_COLUMNS]].copy()
    if sub.empty:
        raise RuntimeError(f"no rows at h={h!r} in {diag_csv} (h values present: {sorted(set(df['h']))})")
    sub = sub.drop_duplicates(subset="event_idx", keep="last")
    for col, out in (("combined_no_bh", "ln_L_no_bh"), ("combined_with_bh", "ln_L_with_bh")):
        vals = sub[col].to_numpy(dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            sub[out] = np.where(vals > 0.0, np.log(vals), np.nan)
    return sub[["event_idx", "ln_L_no_bh", "ln_L_with_bh"]].sort_values("event_idx").reset_index(drop=True)


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
) -> list[NodeResult]:
    """S0-A: one seed, the theta-cross at h=H_GEN, sigma_z_scale=1.0 (truth-theta=(0,1)).

    ``theta_sites``/``smear``/``config``/``h_values``/``score_h`` default to
    exactly the pre-P1/pre-KW-Q1 behaviour (BYTE-IDENTICAL: theta_sites=
    "all", smear="auto" -> smear_global_selection=theta_engaged, config=
    "b0i" -> the original bc venue/flags, h_values=(H_GEN,) -> the single
    h=0.73 node, score_h=None -> H_GEN). Node output directories gain a
    suffix (:func:`_node_dir_suffix`) that is EMPTY at these defaults, so
    default-run paths are unchanged.
    """
    work_root = out_root / f"s0a_seed{seed}"
    work_root.mkdir(parents=True, exist_ok=True)
    events, handler = _build_venue(config, work_root, seed, sigma_z_scale=1.0)
    if event_cap is not None:
        events = events.head(event_cap).reset_index(drop=True)
    suffix = _node_dir_suffix(theta_sites, smear, config)
    read_h = _resolve_score_h(h_values, score_h)
    results: list[NodeResult] = []
    for node in nodes:
        theta_b, theta_s = THETA_NODES[node]
        node_root = work_root / f"node_{node}{suffix}"
        node_root.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        diag_csv, elapsed = run_theta_node(
            node_root, events, seed, handler, theta_b, theta_s,
            h_values=h_values, theta_sites=theta_sites, smear=smear, config=config,
        )
        wall = time.time() - t0
        ln_l = read_event_ln_l(diag_csv, read_h)
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
            f"theta_sites={theta_sites} smear={smear} config={config}] "
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
) -> list[NodeResult]:
    """S0-R: one seed, the theta-cross at h=H_GEN, sigma_z_scale=1.5 (DISCLOSED NULL, see module docstring).

    Same byte-identical-default argument as :func:`run_arm_seed_s0a` (this
    arm's own S0_R_SIGMA_SCALE dose is orthogonal to the P1/KW-Q1 axes).
    """
    work_root = out_root / f"s0r_seed{seed}"
    work_root.mkdir(parents=True, exist_ok=True)
    events, handler = _build_venue(config, work_root, seed, sigma_z_scale=S0_R_SIGMA_SCALE)
    if event_cap is not None:
        events = events.head(event_cap).reset_index(drop=True)
    suffix = _node_dir_suffix(theta_sites, smear, config)
    read_h = _resolve_score_h(h_values, score_h)
    results: list[NodeResult] = []
    for node in nodes:
        theta_b, theta_s = THETA_NODES[node]
        node_root = work_root / f"node_{node}{suffix}"
        node_root.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        diag_csv, elapsed = run_theta_node(
            node_root, events, seed, handler, theta_b, theta_s,
            h_values=h_values, theta_sites=theta_sites, smear=smear, config=config,
        )
        wall = time.time() - t0
        ln_l = read_event_ln_l(diag_csv, read_h)
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
            f"theta_sites={theta_sites} smear={smear} config={config}] "
            f"n_events={len(ln_l)} evaluate_s={elapsed:.2f} wall_s={wall:.2f} -> {diag_csv}",
            flush=True,
        )
    return results


def run_seed_s0c(seed: int, out_root: Path, event_cap: int | None) -> dict[str, Any]:
    """S0-C: one seed, theta=(0,1), the full 41-node H_GRID_41 (costing probe, prereg §2.1)."""
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
        theta_b=0.0,
        theta_s=1.0,
        theta_sites="all",
        smear_global_selection=False,
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

    score_b = [lnL(b=+0.02,s=1) - lnL(b=-0.02,s=1)] / 0.04
    score_s = [lnL(b=0,s=sqrt2) - lnL(b=0,s=1/sqrt2)] / (sqrt2 - 1/sqrt2)
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
    required_nodes = ("b_plus", "b_minus", "s_plus", "s_minus")
    missing_nodes = [n for n in required_nodes if not all_nodes.get(n)]
    if missing_nodes:
        n_present_by_node = {n: len(all_nodes.get(n, [])) for n in required_nodes}
        if seeds is not None:
            missing_pairs = [
                (seed, n) for n in missing_nodes for seed in seeds if seed not in {r.seed for r in all_nodes.get(n, [])}
            ]
            detail = f"missing (seed, node) pairs: {missing_pairs}"
        else:
            detail = f"missing nodes (no seed list given): {missing_nodes}"
        raise ValueError(
            "compute_scores: cannot pool score_b/score_s -- "
            f"{detail}. n_present_by_node={n_present_by_node}. Every one of these evaluate() "
            "calls either was never attempted or raised inside its worker -- check the caller's "
            "printed WORKER ERROR lines / payload['errors'] (run_arm) or missing_csv_paths "
            "(--score-only) for the real underlying cause."
        )
    channels = ("ln_L_no_bh", "ln_L_with_bh")
    out: dict[str, Any] = {}
    for channel in channels:
        # Join b_plus/b_minus per (seed, event_idx).
        bp = pd.concat(
            [r.ln_l.assign(seed=r.seed)[["seed", "event_idx", channel]] for r in all_nodes["b_plus"]],
            ignore_index=True,
        ).rename(columns={channel: "b_plus"})
        bm = pd.concat(
            [r.ln_l.assign(seed=r.seed)[["seed", "event_idx", channel]] for r in all_nodes["b_minus"]],
            ignore_index=True,
        ).rename(columns={channel: "b_minus"})
        sp = pd.concat(
            [r.ln_l.assign(seed=r.seed)[["seed", "event_idx", channel]] for r in all_nodes["s_plus"]],
            ignore_index=True,
        ).rename(columns={channel: "s_plus"})
        sm = pd.concat(
            [r.ln_l.assign(seed=r.seed)[["seed", "event_idx", channel]] for r in all_nodes["s_minus"]],
            ignore_index=True,
        ).rename(columns={channel: "s_minus"})

        b_join = bp.merge(bm, on=["seed", "event_idx"], how="inner")
        s_join = sp.merge(sm, on=["seed", "event_idx"], how="inner")

        score_b = (b_join["b_plus"] - b_join["b_minus"]) / 0.04
        denom_s = math.sqrt(2.0) - 1.0 / math.sqrt(2.0)
        score_s = (s_join["s_plus"] - s_join["s_minus"]) / denom_s

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

        mean_b, sem_b, z_b, n_b = _mean_sem(score_b)
        mean_s, sem_s, z_s, n_s = _mean_sem(score_s)
        out[channel] = {
            "score_b": {"mean": mean_b, "sem": sem_b, "Z": z_b, "n_pooled": n_b},
            "score_s": {"mean": mean_s, "sem": sem_s, "Z": z_s, "n_pooled": n_s},
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
        banked_csv = bc_work_root / f"bc_{r.seed}_work" / f"seed{r.seed}" / "simulations" / "diagnostics" / "event_likelihoods.csv"
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
                d.get("max_rel_diff", 1.0) <= PARITY_FALLBACK_RTOL for d in diffs.values() if d.get("n", 0) > 0
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
        np.isfinite(z_b) and np.isfinite(z_s) and abs(z_b) <= Z_THRESHOLD and abs(z_s) <= Z_THRESHOLD
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
    args: tuple[str, int, Path, tuple[str, ...], int | None, int, str, str, str, tuple[float, ...], float | None],
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
    """
    arm, seed, out_root, nodes, event_cap, cpu_budget, theta_sites, smear, config, h_values, score_h = args
    if not mp.current_process()._identity:  # noqa: SLF001 -- see docstring above
        _pin_worker_affinity(cpu_budget)
    try:
        if arm == "S0-A":
            results = run_arm_seed_s0a(
                seed, out_root, nodes, event_cap,
                theta_sites=theta_sites, smear=smear, config=config, h_values=h_values, score_h=score_h,
            )
        elif arm == "S0-R":
            results = run_arm_seed_s0r(
                seed, out_root, nodes, event_cap,
                theta_sites=theta_sites, smear=smear, config=config, h_values=h_values, score_h=score_h,
            )
        elif arm == "S0-C":
            return {"seed": seed, "s0c": run_seed_s0c(seed, out_root, event_cap)}
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
        return {"seed": seed, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}


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
) -> dict[str, Any]:
    out_root.mkdir(parents=True, exist_ok=True)
    jobs = max(1, min(jobs, len(seeds)))
    cpu_per_job = max(1, total_cpu_budget // jobs)
    task_args = [
        (arm, seed, out_root, nodes, event_cap, cpu_per_job, theta_sites, smear, config, h_values, score_h)
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
        with ctx.Pool(processes=jobs, initializer=_pin_worker_affinity, initargs=(cpu_per_job,)) as pool:
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
        "node_dir_suffix": _node_dir_suffix(theta_sites, smear, config),
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
            "nodes": {n: {"elapsed_s": p["elapsed_s"], "n_events": p["n_events"]} for n, p in r["nodes"].items()},
        }
        for r in ok
    ]

    requested_all_four = all(n in nodes for n in ("b_plus", "b_minus", "s_plus", "s_minus"))
    # Requesting a node (CLI-level) is NOT the same as having produced a
    # NodeResult for it (runner-disclosed P0 crash, B1_2_DRIVER_EXTENSION_
    # NOTE.md "Crash fix"): the old `need_all_four = all(n in nodes ...)`
    # check only looked at what was ASKED FOR, so when every seed's worker
    # errored out (all_nodes left completely empty for every node)
    # compute_scores was still called and pd.concat([]) raised an opaque
    # "No objects to concatenate" instead of this driver ever reporting WHY.
    produced_all_four = all(len(all_nodes.get(n, [])) > 0 for n in ("b_plus", "b_minus", "s_plus", "s_minus"))
    if requested_all_four and produced_all_four:
        scores = compute_scores(all_nodes, seeds=seeds)
        eng = gate_eng(all_nodes)
        payload["scores"] = scores
        payload["gate_eng"] = eng
        if arm == "S0-A":
            parity = gate_parity(all_nodes)
            payload["gate_parity"] = parity
            payload["verdict"] = verdict_s0a(scores, eng, parity)
        elif arm == "S0-R":
            payload["verdict"] = verdict_s0r(scores, eng)
    elif not requested_all_four:
        payload["note"] = (
            f"nodes={nodes} does not include all 4 off-truth nodes -- scores/gates/verdict "
            "require {b_plus,b_minus,s_plus,s_minus}; this is expected for a --smoke run with "
            "1-2 nodes and is NOT a registered read."
        )
        if "truth" in nodes and arm == "S0-A":
            # Still run GATE PARITY if the truth node is present -- it needs no other node.
            payload["gate_parity"] = gate_parity(all_nodes)
    else:
        # All 4 off-truth nodes WERE requested but at least one produced ZERO
        # NodeResult across every seed -- every seed's worker for that node
        # either raised (see payload["errors"] / the WORKER ERROR lines
        # printed above) or the arm otherwise never reached it. Report
        # exactly what's missing instead of letting compute_scores crash.
        have = {n: len(all_nodes.get(n, [])) for n in ("b_plus", "b_minus", "s_plus", "s_minus")}
        missing_pairs = [
            (seed, n)
            for n in ("b_plus", "b_minus", "s_plus", "s_minus")
            for seed in seeds
            if seed not in {r.seed for r in all_nodes.get(n, [])}
        ]
        payload["note"] = (
            f"all 4 off-truth nodes were REQUESTED but produced ZERO results for at least one "
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

    Returns ``(all_nodes, missing_paths)`` -- ``missing_paths`` lists every
    (seed, node) CSV that was requested but not found on disk (reported, not
    fatal: :func:`score_only_payload` computes whatever the union of present
    nodes allows and states plainly what could not be pooled).
    """
    prefix = {"S0-A": "s0a_seed", "S0-R": "s0r_seed"}.get(arm)
    if prefix is None:
        raise ValueError(f"--score-only supports S0-A/S0-R only (no node cross to pool for {arm!r})")
    suffix = _node_dir_suffix(theta_sites, smear, config)
    all_nodes: dict[str, list[NodeResult]] = {n: [] for n in nodes}
    missing: list[str] = []
    for seed in seeds:
        for node in nodes:
            theta_b, theta_s = THETA_NODES[node]
            diag_csv = out_root / f"{prefix}{seed}" / f"node_{node}{suffix}" / "simulations" / "diagnostics" / "event_likelihoods.csv"
            if not diag_csv.is_file():
                missing.append(str(diag_csv))
                continue
            ln_l = read_event_ln_l(diag_csv, score_h)
            all_nodes[node].append(
                NodeResult(
                    node=node,
                    theta_b=theta_b,
                    theta_s=theta_s,
                    seed=seed,
                    diag_csv=str(diag_csv),
                    elapsed_s=float("nan"),  # not measured -- no evaluation happened this invocation
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
    need_all_four = all(len(all_nodes.get(n, [])) > 0 for n in ("b_plus", "b_minus", "s_plus", "s_minus"))
    if need_all_four:
        scores = compute_scores(all_nodes, seeds=seeds)
        eng = gate_eng(all_nodes)
        payload["scores"] = scores
        payload["gate_eng"] = eng
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
            f">=1 seed present for EACH of b_plus/b_minus/s_plus/s_minus; have {have}. "
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
        lines.append(f"Missing CSVs ({payload['n_missing_csv']}): first 5 -> {payload['missing_csv_paths'][:5]}")
    lines.append("")
    if "scores" in payload:
        for channel, d in payload["scores"].items():
            lines.append(f"## {channel}")
            for stat_name in ("score_b", "score_s"):
                s = d[stat_name]
                lines.append(
                    f"- {stat_name}: mean={s['mean']!r} sem={s['sem']!r} Z={s['Z']!r} n_pooled={s['n_pooled']}"
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
            lines.append(f"- {node}: mean_fraction_moved={d['mean_fraction_moved']!r} pass={d['pass']}")
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
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
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
        "--score-only",
        action="store_true",
        help="P0 completion (SYNTHESIS_DOCKET_1_20260829.md sec 2 B1 P0): compute the pooled "
        "prereg §4.1 score_b/score_s/Z_b/Z_s/GATE ENG/GATE PARITY/verdict from event_likelihoods."
        "csv files ALREADY ON DISK under --out-root (matching --seeds/--nodes/--theta-sites/"
        "--smear/--config, for locating the right node directories) -- NO evaluate() call, NO "
        "venue construction. S0-A/S0-R only. Writes <arm>_score_output.json and <arm>_score.md.",
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
            args.arm, seeds, out_root, nodes,
            theta_sites=args.theta_sites, smear=args.smear, config=args.config, score_h=score_h,
        )
        result = score_only_payload(args.arm, seeds, nodes, all_nodes, missing)
        result["theta_sites"] = args.theta_sites
        result["smear"] = args.smear
        result["config"] = args.config
        result["h_values"] = list(h_values)
        result["score_h"] = score_h
        result["registration"] = str(REGISTRATION)
        out_root.mkdir(parents=True, exist_ok=True)
        out_json = out_root / f"{args.arm.lower().replace('-', '')}_score_output.json"
        out_json.write_text(json.dumps(result, indent=2, default=str))
        out_md = out_root / f"{args.arm.lower().replace('-', '')}_score.md"
        write_score_markdown(result, out_md)
        print(f"wrote {out_json}")
        print(f"wrote {out_md}")
        print(json.dumps({k: v for k, v in result.items() if k != "missing_csv_paths"}, indent=2, default=str)[:4000])
        return 0 if not missing or result.get("scores") else 1

    result = run_arm(
        args.arm, seeds, out_root, jobs, args.total_cpu_budget, nodes, event_cap,
        theta_sites=args.theta_sites, smear=args.smear, config=args.config,
        h_values=h_values, score_h=args.score_h,
    )
    result["smoke"] = bool(args.smoke)
    result["event_cap"] = event_cap
    result["registration"] = str(REGISTRATION)

    out_root.mkdir(parents=True, exist_ok=True)
    tag = "smoke" if args.smoke else "full"
    out_json = out_root / f"{args.arm.lower().replace('-', '')}_{tag}_output.json"
    out_json.write_text(json.dumps(result, indent=2, default=str))
    print(f"wrote {out_json}")
    print(json.dumps({k: v for k, v in result.items() if k not in ("per_seed_summary",)}, indent=2, default=str)[:4000])
    return 0 if not result.get("errors") else 1


if __name__ == "__main__":
    raise SystemExit(main())
