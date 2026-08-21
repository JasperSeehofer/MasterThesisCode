r"""Option-B 1D production-correspondence harness (G-0/G-1/G-2 + arms fleet + AMENDMENT A-2).

**AMENDMENT A-2 (2026-08-20, registered in the prereg's append-only VERDICT
section after the b0/bsig005/bsig025/eden05/eden2 fleet ran).** The arms
fleet found the mirror universe was drawing 100% in-catalogue hosts, the
OPPOSITE regime from production (~4.79% in-catalogue, ~95%
completion-leg-dominated) -- S-CORR mechanically had to fail. A-2 re-poses
the correspondence question in production's actual regime with two new
arms, both added here:

- **B-OUT** (:data:`ARM_SPECS` key ``"bout"``, 15 seeds): hosts drawn from
  the POPULATION model the estimator's own completion leg assumes --
  :func:`draw_population_redshifts`, the bare form of the nested closure
  ``_w_pop_eff`` (``bayesian_statistics.py:5775-5783``,
  ``w_pop(z) = dV_c/dz(z, h_true) / (1+z)``) -- rather than from the pinned
  catalogue, and NEVER inserted into the candidate set (``host_galaxy_index
  = -1``, ``in_catalog = False`` -- the exact production "dark"/completion-
  leg bookkeeping convention, ``bayesian_statistics.py:4485``). Real GLADE
  galaxies falling in the localization ball are the only candidates
  (impostors only); the catalogue file itself is never modified. See
  :meth:`MirrorUniverseGenerator.draw_realization`'s ``host_mode="population"``
  branch.
- **B-F1** (:data:`ARM_SPECS` key ``"bf1"``, 2 seeds): the B-0 configuration
  (catalogue-resident hosts, ``sigma_z_scale=1.0``) run under the P14 ``f=1``
  completeness shim (:class:`_UnityCompleteness`, already used by G-1) --
  isolates completeness from the exact-z difference that separates G-1 from
  B-sigma-0.05x.

**Population-model choice, registered here (scientifically load-bearing,
flagged for review).** The completion leg's population weight is
``w_pop(z) = dV_c/dz(z, h) / (1+z)`` (``comoving_volume_element(z, h) /
(1+z)``, the SAME functional form used throughout ``bayesian_statistics.py``
for D(h)/beta_Gbar(h)/the in-catalogue volume-deconvolved kernel -- Gray et
al. 2020, arXiv:1908.06050, Eq. A.10/33; the module's ``_w_pop_eff`` is its
per-call, completeness-multiplied nested-closure instance). B-OUT's host
draw uses this BARE form (no completeness factor -- population draws are
model draws of the true universe, not observed-catalogue density) evaluated
at the mirror truth :data:`H_TRUE` (D-B item c/d convention: the mirror's
true cosmology is h_true, matching how B-0's true d_L is
``dist(host_z, h_true)``). **Domain choice:** ``z in [1e-6,
HOST_DRAW_Z_MAX]`` (:data:`POPULATION_Z_MAX`), NOT the h-dependent detection
horizon ``z_max(h) = dist_to_redshift(get_dl_max(h), h)`` production's D(h)/
beta_Gbar(h) integrals actually use. Reason: ``get_dl_max`` requires a
constructed ``SimulationDetectionProbability`` (the ~20-25 min/h injection-
pool selection-grid cost the G-2 cost decomposition already flags as the
per-h dominant cost) -- building one JUST to draw hosts before evaluate()
duplicates that exact cost. Production's own comment
(``bayesian_statistics.py:1238-1240``, the issue #30 selection-domain-cap
note) records that ``z_max(h) <= ~1.33`` for every ``h`` in the registered
prior range ``[0.60, 0.86]``, strictly inside ``HOST_DRAW_Z_MAX = 1.5`` --
so this domain is a documented, cheap, conservative SUPERSET of every
h-dependent completion-leg integration window actually used at the h values
this harness probes, not an independent/looser choice. It is also the SAME
domain the in-catalogue host draw already respects by construction (every
pinned-catalogue host satisfies ``z < HOST_DRAW_Z_MAX`` by the pruning in
:func:`~darksiren_emri.galaxy_catalogue.handler.GalaxyCatalogueHandler`),
so B-OUT and B-0/B-sigma/E-DEN draw from directly comparable z-support.

---

**AMENDMENT A-3 (2026-08-20, registered in the prereg's append-only VERDICT
section BEFORE this arm ran).** A-2's B-OUT reproduced production's dark
rail, but the orchestrator self-catch showed B-OUT matches the estimator's
POPULATION (bare ``w_pop(z)``) but NOT its SELECTION (the estimator models
*detected* dark events as ``w_pop(z) * (1 - f_bar(z)) * S_bar_phi(z;h)``, not
bare ``w_pop(z)``) -- B-OUT therefore has a data-vs-model mismatch of its
own. A-3 closes that gap:

- **B-SEL** (:data:`ARM_SPECS` key ``"bsel"``, 15 seeds): hosts drawn from
  :func:`draw_selected_population_redshifts`, weighted by
  :func:`selected_population_z_weights` = ``w_pop(z) * (1 - f_bar(z;h_true))
  * S_bar_phi(z;h_true)`` -- the estimator's OWN assumed distribution of
  *detected* dark events, matching the model in BOTH population and
  selection. Never a candidate-set member (same ``host_galaxy_index=-1``/
  ``in_catalog=False`` convention as B-OUT). See
  :meth:`MirrorUniverseGenerator.draw_realization`'s
  ``host_mode="population_selected"`` branch.

  **The weighting objects, precisely (scientifically load-bearing).**
  ``f_bar(z;h)`` is :meth:`~darksiren_emri.galaxy_catalogue.pixel_completeness.PixelCompleteness.f_bar`
  on the object :func:`~darksiren_emri.galaxy_catalogue.pixel_completeness.from_cache_or_build`
  returns -- the IDENTICAL call production's ``evaluate()`` makes
  (``bayesian_statistics.py:3704``). ``S_bar_phi(z;h_true)`` is read from the
  table :func:`~darksiren_emri.bayesian_inference.bayesian_statistics.precompute_phi_marginal_survival`
  returns for ``h_values=[H_TRUE]`` -- the IDENTICAL function production's
  ``evaluate()`` calls (``bayesian_statistics.py:3773``) over a freshly
  constructed :class:`~darksiren_emri.bayesian_inference.simulation_detection_probability.SimulationDetectionProbability`
  built with the SAME constructor arguments production's ``evaluate()`` uses
  (``bayesian_statistics.py:3654-3673``: ``injection_data_dir``,
  ``snr_threshold=SNR_THRESHOLD``, ``dl_bins``/``mass_bins``/``estimator`` at
  the registered :data:`PRODUCTION_FLAGS` values, ``pdet_z_resolved=True``).
  This is the SAME MECHANISM the estimator's own context build uses, not an
  independent re-derivation -- but it is a SEPARATE object (a second
  construction), not the literal same Python instance ``evaluate()`` builds
  internally later: :func:`build_bsel_selection_objects` (below) is called
  BEFORE any mirror event is drawn (:func:`run_arm_seed`'s bsel branch),
  since neither construction depends on the mirror's synthetic events -- only
  on the pinned injection pool/completeness cache and ``h_true`` (the G-2
  cost-decomposition note already flagged these per-(h, catalogue) tables as
  event-set-independent). ``evaluate()`` then rebuilds its own instance
  internally when the realization is analysed (no injection point exists in
  its public API to hand it a pre-built ``SimulationDetectionProbability``) --
  the construction cost is paid twice, registered/disclosed rather than
  optimized away (the amendment's own text: "the draw is cheap; the ordering
  is what matters", not the total cost). Both constructions are deterministic
  given the same pinned injection pool/cache and ``h``, so the two objects
  are functionally identical even though not the same instance.

**AMENDMENT A-4 (2026-08-20, first bisection step, a flag flip).** Under the
runs-of-record basis (``selection_in_completion_numerator="off"``) the
completion NUMERATOR carries no detection weight while its normalization
carries detection weight -- an asymmetry whose fused form already exists as
a shipped production flag ([P1]/[P2], commit ``2b10b8b8``). In a
MODEL-MATCHED universe (which B-SEL already is) the correctly normalized
likelihood should be the one whose numerator and denominator use the same
detection model.

- **B-SELF** (:data:`ARM_SPECS` key ``"bself"``, 15 seeds): B-SEL's
  configuration verbatim -- same :data:`ARM_HOST_MODE`
  (``"population_selected"``), same :data:`ARM_UNITY_COMPLETENESS`
  (``False``), same :data:`ARM_SEEDS` -- with ONLY
  ``selection_in_completion_numerator="fused"`` (:data:`ARM_SELECTION_CELL`)
  in place of B-SEL's ``"off"``. Isolates the numerator/denominator
  detection-model convention as the single varied axis between the two arms.

**AMENDMENT A-5 (2026-08-20, next bisection step: the event-term measure).**
``docs/derivations/completion_numerator_data_measure.md`` §2/§6 (author
approval 2026-08-20): the completion numerator's GW event term is a density
in the dimensionless distance ratio ``d_L(z;h)/d_L,det`` -- which does NOT
integrate to 1 over the observable, unlike the completion denominator's
measure. This is the falsifier instrument B-DEN registered by the
prereg's AMENDMENT A-5.

- **B-DEN** (:data:`ARM_SPECS` key ``"bden"``, 15 seeds): B-SEL's
  configuration verbatim -- same :data:`ARM_HOST_MODE`
  (``"population_selected"``), same :data:`ARM_UNITY_COMPLETENESS`
  (``False``), same :data:`ARM_SELECTION_CELL` (``"off"``), same
  :data:`ARM_SEEDS` -- with ONLY ``completion_event_measure="data"``
  (:data:`ARM_EVENT_MEASURE`) in place of B-SEL's ``"ratio"``. Isolates the
  completion numerator's event-term measure as the single varied axis
  between the two arms.

---

**What this instrument is (G-0/G-1/G-2 base build).** The Option-B measurement registered in
``results/prod2d_closure_20260818/PREREGISTRATION_1D_CORRESPONDENCE.md`` (v2):
decompose the production 1D base tilt into information-starvation vs
form-defect components. Gate G-0 (prereg §4, the STOP gate that must pass
before ANY arm runs — the production-wholesale fidelity pilot) is PASSED
(see the append-only VERDICT section of the prereg). This build adds the
**mirror-universe generator** (:class:`MirrorUniverseGenerator`, D-B),
gate **G-1** (:func:`run_g1_null`, the mirror sanity null) and gate **G-2**
(:func:`run_g2_cost_pilot`, the D-D cost pilot). The adjudicating arms
(B-0/B-σ/B-D2/E-DEN, prereg §2) are NOT run by this build — only the
machinery + the two STOP gates, per task scope.

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
import dataclasses
import functools
import hashlib
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

from darksiren_emri.bayesian_inference.bayesian_statistics import (
    FRACTIONAL_LUMINOSITY_DISTANCE_ERROR_THRESHOLD,
    BayesianStatistics,
    path_a_completion_numerators,
    path_a_mixture_objects,
    precompute_phi_marginal_survival,
)
from darksiren_emri.bayesian_inference.simulation_detection_probability import (
    SimulationDetectionProbability,
)
from darksiren_emri.constants import (
    HOST_DRAW_Z_MAX,
    M_SOURCE_FRAME_MAX,
    M_SOURCE_FRAME_MIN,
    SNR_THRESHOLD,
    H,
)
from darksiren_emri.cosmological_model import Model1CrossCheck
from darksiren_emri.galaxy_catalogue.handler import (
    GalaxyCatalogueHandler,
    InternalCatalogColumns,
)
from darksiren_emri.galaxy_catalogue.pixel_completeness import (
    CompletenessModel,
    from_cache_or_build,
)
from darksiren_emri.physical_relations import comoving_volume_element, dist_vectorized

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

# G-0 registered flags (postfix_baseline/iiib run_metadata_0.json, verbatim) —
# EXCEPT the completion cell: "fused" is the production basis for all FUTURE
# runs-of-record per the author's D2 ruling (ledger row #159, 2026-08-22;
# [PHYSICS] gate package: docs/derivations/
# PROPOSAL_FUSED_BASIS_AND_IMPOSTOR_DIRECTION_20260821.md §1-D2). Historical
# runs-of-record (through 2026-08-21) ran "off" — the ratified
# IMPLEMENTATION-CONVENTION DEFECT cell (rows #157/#158) — and STAND with the
# row #119 bridge; regenerating a banked off-cell artifact requires passing
# selection_in_completion_numerator="off" EXPLICITLY (the O6 GATE D6 /
# P3 GATE R-P3 pattern).
PRODUCTION_FLAGS: dict[str, str] = {
    "--normalization_mode": "absolute_marginal",
    "--host_z_kernel": "volume_deconv",
    "--selection_in_completion_numerator": "fused",
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

# ── D-B/S-RAIL registered h grid (prereg §3 S-RAIL) ──────────────────────────
# The production H_VALUES 41-node hybrid grid [0.600, 0.860], VERBATIM,
# extracted from the banked postfix_baseline/iiib event_likelihoods.csv (the
# grid production actually ran, not re-derived from a formula) plus the
# REPORTED-ONLY diagnostic low wing {0.50, 0.52, ..., 0.58} (never
# band-bearing; prereg S-RAIL).
H_GRID_41: tuple[float, ...] = (
    0.6, 0.61, 0.62, 0.63, 0.64, 0.65, 0.655, 0.66, 0.665, 0.67, 0.675, 0.68,
    0.685, 0.69, 0.695, 0.7, 0.705, 0.71, 0.715, 0.72, 0.725, 0.73, 0.735,
    0.74, 0.745, 0.75, 0.755, 0.76, 0.765, 0.77, 0.775, 0.78, 0.785, 0.79,
    0.8, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86,
)  # fmt: skip
H_WING_LOW: tuple[float, ...] = (0.50, 0.52, 0.54, 0.56, 0.58)
H_GRID_FULL: tuple[float, ...] = tuple(sorted(set(H_WING_LOW) | set(H_GRID_41)))
H_TRUE: float = H  # mirror-universe truth h_true = 0.73 (constants.H, D-B)
R_LOW_THRESHOLD = 0.605  # DS-6 rail statistic (prereg S-RAIL)

# G-1 "exact z" operational mechanism (harness-registered, NOT in the prereg's
# P14 text -- that text fixes only the completeness mechanism). "sigma_z_scale
# -> 0" cannot be realized via realize_observed_catalogue(sigma_scale=0)
# (that call is a documented BYTE-IDENTICAL copy of the parent -- it changes
# NEITHER z_obs NOR z_error, so it reproduces B-0, not a null). The harness
# instead builds its own exact-z catalogue variant: z_obs left AT the stored
# (already-truth-by-convention, D-B item d) value, and REDSHIFT_MEASUREMENT_ERROR
# floored to a tiny width so the host-z kernel integrates against an
# effectively delta-function redshift. Flagged for review.
EXACT_Z_ERROR_FLOOR = 1.0e-6

# Host-draw weighting floor (harness bug fix, 2026-08-20 diagnosis of the
# bsig025 fleet failure, job 6383719). realize_observed_catalogue's z-floor
# CLIP (GALAXY_CATALOG_REDSHIFT_LOWER_LIMIT = 1e-5, point mass, no redraw --
# observed_realization.py:23-25/332, an author-accepted approximation of the
# REALIZATION, not a claim about the host population) produces a handful of
# rows pinned to the identical z=1e-5 value; how many depends on
# sigma_z_scale (354 rows at 0.05, 4188 at 0.25, per the job-6383719 sidecar
# logs -- monotone increasing but not linearly). _host_draw_weights (below)
# weights purely by 1/d_L(z)^2 with only a numerical divide-by-zero guard
# (1e-6 Gpc, ~1 kpc) -- since d_L(z=1e-5, h=0.73) ~ 4.1e-5 Gpc (~41 kpc) sits
# ABOVE that guard, the clip never engages and these realization-artifact
# rows get weight ~4e7-1e8x a typical host. Empirically the REAL (baseline,
# unscattered) GLADE pool's closest galaxy sits at z ~ 0.00195 -- two orders
# of magnitude above the technical clip -- so flooring the WEIGHTING z at
# 1e-3 (an order of magnitude below the real population's minimum, two
# orders above the realization clip) leaves every genuine host's weight
# untouched while capping the clipped-artifact rows down to roughly par with
# the closest real galaxies instead of ~40000x above them. This resolves the
# observed pathology: at sigma_z_scale=0.25 the artifact rows' combined
# weight (pre-fix) swamped the entire 200-event weighted-without-replacement
# draw, placing every mirror event at a near-zero true d_L against an
# unrelated (donor-row, ABSOLUTE) sigma_dL -- distance_relative_error >> 0.1
# for all 200 events -> 0 detections after use_detection() quality
# filtering -> the harness's own "expected diagnostics CSV not found" guard.
# Only the WEIGHT is floored here; the actual event PLACEMENT (draw_realization
# item c, true_d_L from the drawn host's own z) is untouched -- a host that
# does get drawn near the realization floor should still legitimately fail
# the quality filter on its own merits, just not for ALL 200 events at once.
HOST_DRAW_WEIGHT_Z_FLOOR = 1.0e-3

# ── Fleet arm registry (cluster-fleet CLI stage, prereg §2 arm doses) ────────
# arm -> (sigma_z_scale, area_scale), verbatim from the task spec's registered
# mapping. b0 = B-0 (production-mapped); bsig005/bsig025 = B-sigma starvation
# ladder (0.05x/0.25x); eden05/eden2 = E-DEN (area x0.5/x2, exploratory);
# bout = B-OUT (AMENDMENT A-2, population-model host draw -- sigma_z_scale/
# area_scale carried here only for schema uniformity with the run_arm_seed
# record; B-OUT's host draw ignores host_pool/sigma_z_scale entirely, see
# ARM_HOST_MODE below); bf1 = B-F1 (AMENDMENT A-2, B-0 config + f=1
# completeness control, see ARM_UNITY_COMPLETENESS below); bsel = B-SEL
# (AMENDMENT A-3, population x (1-completeness) x survival host draw --
# sigma_z_scale/area_scale carried here only for schema uniformity, same as
# bout; see ARM_HOST_MODE below).
ARM_SPECS: dict[str, tuple[float, float]] = {
    "b0": (1.0, 1.0),
    "bsig005": (0.05, 1.0),
    "bsig025": (0.25, 1.0),
    "eden05": (1.0, 0.5),
    "eden2": (1.0, 2.0),
    "bout": (1.0, 1.0),
    "bf1": (1.0, 1.0),
    "bsel": (1.0, 1.0),
    "bself": (1.0, 1.0),
    "bden": (1.0, 1.0),
}
# AMENDMENT A-2/A-3: per-arm host-draw mode. "catalogue" (default, all
# pre-A-2 arms) draws hosts FROM the pinned catalogue's HostPool (D-B item
# a); "population" (bout only) draws hosts from the estimator's own
# completion-leg population model (draw_population_redshifts + isotropic
# sky) and NEVER inserts them into the candidate set (module docstring,
# AMENDMENT A-2); "population_selected" (bsel only, AMENDMENT A-3) draws
# hosts from the estimator's own assumed distribution of DETECTED dark
# events (draw_selected_population_redshifts + isotropic sky), likewise
# never inserted into the candidate set.
ARM_HOST_MODE: dict[str, Literal["catalogue", "population", "population_selected"]] = {
    "b0": "catalogue",
    "bsig005": "catalogue",
    "bsig025": "catalogue",
    "eden05": "catalogue",
    "eden2": "catalogue",
    "bout": "population",
    "bf1": "catalogue",
    "bsel": "population_selected",
    "bself": "population_selected",
    "bden": "population_selected",
}
# AMENDMENT A-2: per-arm completeness override. True (bf1 only) monkeypatches
# the real GLADE completeness object with the P14 f=1 shim
# (:class:`_UnityCompleteness`, the SAME mechanism G-1 uses) for the duration
# of that arm's evaluate() call -- the B-F1 completeness control. bsel uses
# the REAL completeness object both for its host-draw weighting (via
# build_bsel_selection_objects) AND for its evaluate() call -- False here,
# same as every other non-bf1 arm.
ARM_UNITY_COMPLETENESS: dict[str, bool] = {
    "b0": False,
    "bsig005": False,
    "bsig025": False,
    "eden05": False,
    "eden2": False,
    "bout": False,
    "bf1": True,
    "bsel": False,
    "bself": False,
    "bden": False,
}
# AMENDMENT A-4: per-arm ``selection_in_completion_numerator`` convention
# (mirrors production's own flag of the same name,
# ``bayesian_statistics.py:3010``). Every pre-A-4 arm defaults to "off" --
# the runs-of-record basis (:data:`PRODUCTION_FLAGS`) -- so this registry
# introduces NO behaviour change for any existing arm; only "bself" (the
# first bisection step, PREREGISTRATION_1D_CORRESPONDENCE.md AMENDMENT A-4)
# is "fused". "bself" is otherwise IDENTICAL to "bsel" (same ARM_SPECS,
# ARM_HOST_MODE="population_selected", ARM_UNITY_COMPLETENESS=False, same
# seed list) -- the completion numerator/denominator detection-model
# convention is the ONLY axis this arm varies relative to bsel.
ARM_SELECTION_CELL: dict[str, str] = {
    "b0": "off",
    "bsig005": "off",
    "bsig025": "off",
    "eden05": "off",
    "eden2": "off",
    "bout": "off",
    "bf1": "off",
    "bsel": "off",
    "bself": "fused",
    "bden": "off",
}
# B-DEN falsifier instrument (docs/derivations/completion_numerator_data_measure.md
# §6; AMENDMENT A-5, results/prod2d_closure_20260818/
# PREREGISTRATION_1D_CORRESPONDENCE.md). Mirrors production's own
# ``--completion_event_measure`` flag. Every pre-A-5 arm defaults to "ratio"
# -- the runs-of-record basis (:data:`PRODUCTION_FLAGS` predates this flag
# entirely) -- so this registry introduces NO behaviour change for any
# existing arm; only "bden" (A-5's falsifier instrument) is "data". "bden"
# is otherwise IDENTICAL to "bsel" (same ARM_SPECS, ARM_HOST_MODE=
# "population_selected", ARM_UNITY_COMPLETENESS=False, ARM_SELECTION_CELL=
# "off", same seed list) -- the completion numerator's event-term measure is
# the ONLY axis this arm varies relative to bsel.
ARM_EVENT_MEASURE: dict[str, str] = {
    "b0": "ratio",
    "bsig005": "ratio",
    "bsig025": "ratio",
    "eden05": "ratio",
    "eden2": "ratio",
    "bout": "ratio",
    "bf1": "ratio",
    "bsel": "ratio",
    "bself": "ratio",
    "bden": "data",
}
# Registered paired-seed discipline (prereg §1 D-C, extended by AMENDMENT
# A-2/A-3): b0/bsig005 get the adjudicating N=25; bsig025/eden05/eden2 are
# the N=10 reported-only doses; bout is the AMENDMENT A-2 adjudicating arm
# (N=15); bf1 is the AMENDMENT A-2 completeness control (N=2); bsel is the
# AMENDMENT A-3 adjudicating arm (N=15, the true isolation test). All arm
# seed lists start at the SAME 900101 anchor (paired across arms by
# construction, so a B-sigma/E-DEN/B-OUT/B-F1/B-SEL seed at index i is the
# same universe construction seed as B-0's seed at index i).
ARM_SEEDS: dict[str, tuple[int, ...]] = {
    "b0": tuple(range(900101, 900126)),
    "bsig005": tuple(range(900101, 900126)),
    "bsig025": tuple(range(900101, 900111)),
    "eden05": tuple(range(900101, 900111)),
    "eden2": tuple(range(900101, 900111)),
    "bout": tuple(range(900101, 900116)),
    "bf1": tuple(range(900101, 900103)),
    "bsel": tuple(range(900101, 900116)),
    "bself": tuple(range(900101, 900116)),
    "bden": tuple(range(900101, 900116)),
}


class _UnityCompleteness:
    """P14: f≡1 completeness shim satisfying the ``CompletenessModel`` Protocol.

    Every method returns 1.0 (broadcast to the input shape), i.e. the
    catalogue is treated as fully complete everywhere -- G-1's "full
    completeness" mechanism (prereg §4 P14). The real GLADE completeness
    object cannot be dialed to f=1, so this harness-owned stand-in is
    monkeypatched over
    :func:`darksiren_emri.galaxy_catalogue.pixel_completeness.from_cache_or_build`
    at its ``bayesian_statistics`` module-level import site for the duration
    of the G-1 run only (never a production code edit).
    """

    def f_bar(
        self, z: float | npt.NDArray[np.floating[Any]], h: float = H_TRUE
    ) -> float | npt.NDArray[np.floating[Any]]:
        arr = np.asarray(z, dtype=np.float64)
        return np.ones_like(arr) if arr.ndim > 0 else 1.0

    def f_k(
        self, z: float | npt.NDArray[np.floating[Any]], k: int, h: float = H_TRUE
    ) -> float | npt.NDArray[np.floating[Any]]:
        return self.f_bar(z, h)

    def ang2pix(self, phi: float, theta: float) -> int:
        return 0

    def get_completeness_at_redshift(
        self, z: float | npt.NDArray[np.floating[Any]], h: float = H_TRUE
    ) -> float | npt.NDArray[np.floating[Any]]:
        return self.f_bar(z, h)


# ── Config scaffold (future arms; G-0 exercises only the fidelity layer) ────


@dataclass(frozen=True)
class CorrespondenceConfig:
    """Registered-shape config for the full Option-B harness (prereg §1/§2).

    G-0 reads only ``crb_reference_csv``/``injection_data_dir``/``h_probe``.
    :class:`MirrorUniverseGenerator` (D-B) reads ``n_events``/``crb_reference_csv``.
    ``sigma_z_scale``/``area_scale``/``d2_form`` are the B-sigma/E-DEN/B-D2
    arm knobs -- NOT run by this build (the adjudicating arms, prereg §2);
    carried here so a later build extends this dataclass rather than
    replacing it (prereg §5: "structured for the later arms").
    ``pruned_catalogue_csv`` is unused (the real candidate structure is built
    by :func:`_load_galaxy_catalog_handler` from :data:`REDUCED_CATALOGUE_PATH`
    directly, not from a config field) -- kept for prereg-shape compatibility.

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


@dataclass(frozen=True)
class HostPool:
    """The candidate-structure host pool a realization draws from (D-B).

    Built from the SAME pruned/rotated/mass-mapped catalogue production's own
    ``GalaxyCatalogueHandler`` builds (``M_min=M_SOURCE_FRAME_MIN,
    M_max=M_SOURCE_FRAME_MAX, z_max=HOST_DRAW_Z_MAX``), so the row order
    (hence any ``host_galaxy_index`` this harness writes) is IDENTICAL to what
    a wholesale ``--evaluate`` run over the same catalogue file will itself
    build -- deterministic given fixed inputs, verified structurally (not
    re-verified per-seed, since the pruning is a pure function of the file).

    Attributes:
        phiS: Ecliptic sky azimuth (rad), one per host.
        qS: Ecliptic sky polar angle (rad), one per host.
        z: Host redshift -- BY THE MIRROR CONVENTION (D-B item d, registered)
            this catalogue value is treated as EXACT TRUTH for the mirror
            universe (real galaxies have no independently known true z; the
            catalogue's own z_obs is declared truth here).
        z_error: Host redshift measurement error (as stored/realized).
        n: Number of hosts in the pool.
    """

    phiS: npt.NDArray[np.float64]
    qS: npt.NDArray[np.float64]
    z: npt.NDArray[np.float64]
    z_error: npt.NDArray[np.float64]
    n: int


@functools.lru_cache(maxsize=8)
def _load_galaxy_catalog_handler(catalogue_path: str) -> GalaxyCatalogueHandler:
    """Build (or return the cached) ``GalaxyCatalogueHandler`` for a catalogue file.

    Calls production's own class wholesale (D-A fidelity extended to the
    candidate-structure builder), so the mirror's host geometry/pruning is
    byte-identical to what a wholesale evaluate() run over the same
    catalogue file builds. Cached (harness-side reuse, not a production
    change; a G-2 cost finding): repeated seeds/h-values at the same
    ``sigma_z_scale`` (hence the same catalogue file) pay this cost --
    catalogue read + prune + BallTree build -- exactly once per process, and
    the SAME handler instance is handed to
    :func:`run_mirror_seed_inprocess`'s ``BayesianStatistics.evaluate`` call,
    so the host-draw pool and the evaluate() candidate structure are not just
    consistent but the IDENTICAL object.

    Args:
        catalogue_path: Absolute path to a reduced-catalogue-schema CSV (the
            baseline :data:`REDUCED_CATALOGUE_PATH` or an
            ``observed_catalogue_*``/exact-z variant).

    Returns:
        The (possibly cached) handler.
    """
    is_baseline = Path(catalogue_path).resolve() == Path(REDUCED_CATALOGUE_PATH).resolve()
    return GalaxyCatalogueHandler(
        M_min=M_SOURCE_FRAME_MIN,
        M_max=M_SOURCE_FRAME_MAX,
        z_max=HOST_DRAW_Z_MAX,
        observed_catalogue_path=None if is_baseline else catalogue_path,
    )


def _host_pool_from_handler(handler: GalaxyCatalogueHandler) -> HostPool:
    """Extract a :class:`HostPool` from a built handler's pruned catalogue."""
    df = handler.reduced_galaxy_catalog.reset_index(drop=True)
    return HostPool(
        phiS=df[InternalCatalogColumns.PHI_S].to_numpy(dtype=np.float64),
        qS=df[InternalCatalogColumns.THETA_S].to_numpy(dtype=np.float64),
        z=df[InternalCatalogColumns.REDSHIFT].to_numpy(dtype=np.float64),
        z_error=df[InternalCatalogColumns.REDSHIFT_ERROR].to_numpy(dtype=np.float64),
        n=len(df),
    )


def _load_host_pool(catalogue_path: str) -> HostPool:
    """Convenience wrapper: cached handler -> :class:`HostPool`."""
    return _host_pool_from_handler(_load_galaxy_catalog_handler(catalogue_path))


def build_exact_z_catalogue(
    output_csv_path: str, catalogue_path: str = REDUCED_CATALOGUE_PATH
) -> str:
    """Build the G-1 "exact z" catalogue variant (harness-registered mechanism).

    Copies the reduced catalogue schema byte-for-byte EXCEPT
    ``REDSHIFT_MEASUREMENT_ERROR``, which is floored to
    :data:`EXACT_Z_ERROR_FLOOR` so the host-z kernel integrates against an
    effectively delta-function redshift (see the module-level "G-1 exact z"
    note -- this is NOT ``realize_observed_catalogue(sigma_scale=0)``, which
    is a documented byte-identical copy and would reproduce B-0, not a null).

    Args:
        output_csv_path: Destination CSV path (headerless, reduced-catalogue
            schema, same column order as the parent).
        catalogue_path: Parent reduced catalogue (pinned baseline default).

    Returns:
        ``output_csv_path``.
    """
    from darksiren_emri.galaxy_catalogue.handler import _reduced_catalog_column_names

    names = _reduced_catalog_column_names()
    df = pd.read_csv(catalogue_path, names=names)
    df[InternalCatalogColumns.REDSHIFT_ERROR] = EXACT_Z_ERROR_FLOOR
    df.to_csv(output_csv_path, header=False, index=False)
    return output_csv_path


# ── AMENDMENT A-2: population-model host draw (B-OUT) ────────────────────────
# Domain: see the module docstring's "Population-model choice" section for
# the full registered justification (h_true evaluation, [1e-6, HOST_DRAW_Z_MAX]
# domain vs the h-dependent completion horizon).
POPULATION_Z_MIN: float = 1.0e-6
POPULATION_Z_MAX: float = HOST_DRAW_Z_MAX
_POPULATION_Z_GRID_N = 4001


def population_z_weights(z: npt.NDArray[np.float64], h: float = H_TRUE) -> npt.NDArray[np.float64]:
    """The estimator's own completion-leg population weight, bare form.

    ``w_pop(z) = dV_c/dz(z, h) / (1+z)`` -- byte-identical functional form to
    the production nested closure ``_w_pop_eff``
    (``bayesian_statistics.py:5775-5783``) with its ``f_k`` completeness
    factor OMITTED (a population draw of the true universe, not the observed
    catalogue's completeness-weighted density; the module docstring's
    "Population-model choice" section registers this).

    Args:
        z: Redshift grid/values.
        h: Dimensionless Hubble parameter (default: the mirror truth
            :data:`H_TRUE`).

    Returns:
        ``w_pop(z)``, same shape as ``z``.

    References:
        Gray et al. (2020), arXiv:1908.06050, Eq. (A.10)/(33).
    """
    z_arr = np.asarray(z, dtype=np.float64)
    return np.asarray(comoving_volume_element(z_arr, h=h), dtype=np.float64) / (1.0 + z_arr)


def _inverse_cdf_draw(
    rng: np.random.Generator,
    n: int,
    z_grid: npt.NDArray[np.float64],
    w: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Shared inverse-CDF sampler: ``n`` draws from weights ``w`` on ``z_grid``.

    Trapezoid-quadrature CDF on ``z_grid``, then linear interpolation of
    ``n`` uniform draws through it. Deterministic given ``rng``'s state
    (consumes exactly ``n`` uniform draws). Shared by
    :func:`draw_population_redshifts` (B-OUT, AMENDMENT A-2) and
    :func:`draw_selected_population_redshifts` (B-SEL, AMENDMENT A-3) so both
    arms' host-z draws use byte-identical inverse-CDF machinery -- only the
    weight function differs.

    Args:
        rng: Seeded generator (consumes exactly ``n`` uniform draws).
        n: Number of redshifts to draw.
        z_grid: Quadrature/interpolation grid, shape ``(n_grid,)``.
        w: Non-negative weight at each ``z_grid`` node, shape ``(n_grid,)``.

    Returns:
        Drawn redshifts, shape ``(n,)``.

    Raises:
        ValueError: If ``w`` integrates to <= 0 over ``z_grid``.
    """
    segment_mass = 0.5 * (w[1:] + w[:-1]) * np.diff(z_grid)
    cdf = np.concatenate(([0.0], np.cumsum(segment_mass)))
    total = cdf[-1]
    if total <= 0.0:
        raise ValueError("weights integrate to <= 0 over the domain")
    cdf = cdf / total
    u = rng.uniform(0.0, 1.0, size=n)
    return np.interp(u, cdf, z_grid)


def draw_population_redshifts(
    rng: np.random.Generator,
    n: int,
    h: float = H_TRUE,
    z_min: float = POPULATION_Z_MIN,
    z_max: float = POPULATION_Z_MAX,
    n_grid: int = _POPULATION_Z_GRID_N,
) -> npt.NDArray[np.float64]:
    """Inverse-CDF draw of ``n`` redshifts from :func:`population_z_weights`.

    Deterministic given ``rng``'s state (single ``rng.uniform`` call via
    :func:`_inverse_cdf_draw`): builds a trapezoid-quadrature CDF on a dense
    ``z`` grid, then linearly interpolates ``n`` uniform draws through it.

    Args:
        rng: Seeded generator (consumes exactly ``n`` uniform draws).
        n: Number of redshifts to draw.
        h: Dimensionless Hubble parameter for :func:`population_z_weights`
            (default: :data:`H_TRUE`).
        z_min: Lower domain bound (default :data:`POPULATION_Z_MIN`).
        z_max: Upper domain bound (default :data:`POPULATION_Z_MAX`).
        n_grid: Quadrature/interpolation grid resolution.

    Returns:
        Drawn redshifts, shape ``(n,)``.
    """
    z_grid = np.linspace(z_min, z_max, n_grid, dtype=np.float64)
    w = population_z_weights(z_grid, h=h)
    return _inverse_cdf_draw(rng, n, z_grid, w)


# ── AMENDMENT A-3: estimator-self-consistent host draw (B-SEL) ──────────────
# The estimator's OWN assumed density of DETECTED dark (out-of-catalogue)
# events -- see the module docstring's "AMENDMENT A-3" section for the full
# registered justification of which objects are reused and how.


def selected_population_z_weights(
    z: npt.NDArray[np.float64],
    completeness: CompletenessModel,
    phi_survival_table: dict[float, tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]],
    h: float = H_TRUE,
) -> npt.NDArray[np.float64]:
    r"""``w_pop(z) * (1 - f_bar(z;h)) * S_bar_phi(z;h)`` -- B-SEL's host weight.

    The estimator's own assumed distribution of *detected* dark events: the
    bare population weight (:func:`population_z_weights`) times the
    catalogue-incompleteness deficit (``1 - f_bar``, ``f_bar`` from
    ``completeness``) times the phi-marginal detection survival
    (``S_bar_phi``, read off ``phi_survival_table`` by linear interpolation,
    ``0`` outside the tabulated ``z`` domain -- production's own
    "undetectable beyond the table's z_max" convention). Both ``completeness``
    and ``phi_survival_table`` are expected to come from
    :func:`build_bsel_selection_objects` (module docstring "AMENDMENT A-3").

    Args:
        z: Redshift grid/values.
        completeness: A ``CompletenessModel``-satisfying object (production's
            ``f_bar(z, h)`` accessor).
        phi_survival_table: ``h -> (z_grid, S_bar_phi(z_grid))``, the output
            of :func:`~darksiren_emri.bayesian_inference.bayesian_statistics.precompute_phi_marginal_survival`.
        h: Dimensionless Hubble parameter (default: :data:`H_TRUE`); must be
            a key of ``phi_survival_table``.

    Returns:
        Non-negative weights, same shape as ``z`` (clipped at 0 to absorb
        float noise; the product of a probability, a [0,1] deficit, and a
        [0,1] survival is non-negative by construction).

    Raises:
        KeyError: If ``h`` is not a key of ``phi_survival_table``.

    References:
        Gray et al. (2020), arXiv:1908.06050, Eqs. (29), (33).
        FIXB_PATHA_PACKAGE.md §3.2 (``S_bar_phi`` definition).
    """
    if h not in phi_survival_table:
        raise KeyError(
            f"phi_survival_table has no entry for h={h!r}; keys={sorted(phi_survival_table)}"
        )
    z_arr = np.asarray(z, dtype=np.float64)
    w_pop = population_z_weights(z_arr, h=h)
    f_bar = np.asarray(completeness.f_bar(z_arr, h), dtype=np.float64)
    z_grid, s_phi = phi_survival_table[h]
    s_interp = np.interp(z_arr, z_grid, s_phi, left=0.0, right=0.0)
    w = w_pop * (1.0 - f_bar) * s_interp
    return np.clip(w, 0.0, None)


def draw_selected_population_redshifts(
    rng: np.random.Generator,
    n: int,
    completeness: CompletenessModel,
    phi_survival_table: dict[float, tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]],
    h: float = H_TRUE,
    z_min: float = POPULATION_Z_MIN,
    z_max: float = POPULATION_Z_MAX,
    n_grid: int = _POPULATION_Z_GRID_N,
) -> npt.NDArray[np.float64]:
    """Inverse-CDF draw of ``n`` redshifts from :func:`selected_population_z_weights`.

    The B-SEL host-z draw (AMENDMENT A-3): population x (1-completeness) x
    survival, matching the estimator's OWN model of detected dark events in
    both population and selection (contrast :func:`draw_population_redshifts`,
    B-OUT's bare-population draw).

    Args:
        rng: Seeded generator (consumes exactly ``n`` uniform draws).
        n: Number of redshifts to draw.
        completeness: See :func:`selected_population_z_weights`.
        phi_survival_table: See :func:`selected_population_z_weights`.
        h: Dimensionless Hubble parameter (default: :data:`H_TRUE`).
        z_min: Lower domain bound (default :data:`POPULATION_Z_MIN`).
        z_max: Upper domain bound (default :data:`POPULATION_Z_MAX`).
        n_grid: Quadrature/interpolation grid resolution.

    Returns:
        Drawn redshifts, shape ``(n,)``.
    """
    z_grid = np.linspace(z_min, z_max, n_grid, dtype=np.float64)
    w = selected_population_z_weights(z_grid, completeness, phi_survival_table, h=h)
    return _inverse_cdf_draw(rng, n, z_grid, w)


@functools.lru_cache(maxsize=4)
def build_bsel_selection_objects(
    h_true: float = H_TRUE,
    injection_dir: str = INJECTION_POOL_DIR,
    pdet_dl_bins: int = 60,
    pdet_mass_bins: int = 40,
    pdet_estimator: str = "local_linear",
    allow_low_pdet_coverage: bool = True,
    z_max_cap: float = HOST_DRAW_Z_MAX,
) -> tuple[CompletenessModel, dict[float, tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]]]:
    r"""Build (completeness, S_bar_phi table) at ``h_true`` -- B-SEL's weighting objects.

    Calls the SAME production construction functions, with the SAME
    arguments, that ``BayesianStatistics.evaluate()`` calls internally
    (``bayesian_statistics.py:3654-3673`` for the
    :class:`~darksiren_emri.bayesian_inference.simulation_detection_probability.SimulationDetectionProbability`
    constructor, ``:3704`` for :func:`~darksiren_emri.galaxy_catalogue.pixel_completeness.from_cache_or_build`,
    ``:3773`` for :func:`~darksiren_emri.bayesian_inference.bayesian_statistics.precompute_phi_marginal_survival`)
    -- restructured to run BEFORE any mirror event is drawn, per AMENDMENT
    A-3's registered ordering ("the draw is cheap; the ordering is what
    matters"): neither construction depends on the mirror's synthetic
    events, only on the pinned injection pool/completeness cache and
    ``h_true``, so this pays the (event-set-independent, per the G-2 cost
    finding) detection-probability-grid construction cost a SECOND time
    relative to the ``evaluate()`` call that follows the draw (no injection
    point exists in ``evaluate()``'s public API to hand it a pre-built
    ``SimulationDetectionProbability`` instance) -- registered/disclosed, not
    optimized away. ``functools.lru_cache``-d (harness-side reuse only, not a
    production change): a process that runs multiple bsel seeds pays this
    cost once.

    Args:
        h_true: The mirror-universe truth (default: :data:`H_TRUE`); the
            ONLY h-value the survival table is tabulated at.
        injection_dir: The pinned injection pool (default:
            :data:`INJECTION_POOL_DIR`).
        pdet_dl_bins: :data:`PRODUCTION_FLAGS`\ ``["--pdet_dl_bins"]`` value.
        pdet_mass_bins: :data:`PRODUCTION_FLAGS`\ ``["--pdet_mass_bins"]`` value.
        pdet_estimator: :data:`PRODUCTION_FLAGS`\ ``["--pdet_estimator"]`` value.
        allow_low_pdet_coverage: Forwarded to ``SimulationDetectionProbability``
            (default ``True``, harness-registered -- same convention as
            :func:`run_mirror_seed_inprocess`'s B-OUT note: the population
            draw's full-domain support sits near/beyond the injection pool's
            calibrated depth by construction, not by defect).
        z_max_cap: Analysis-depth cap forwarded to
            :func:`~darksiren_emri.bayesian_inference.bayesian_statistics.precompute_phi_marginal_survival`
            (default :data:`HOST_DRAW_Z_MAX`, matching
            ``cosmological_model.max_redshift``'s default of 1.5 -- the SAME
            ``REDSHIFT_UPPER_LIMIT`` production's own ``evaluate()`` call
            uses when it rebuilds this table internally).

    Returns:
        ``(completeness, phi_survival_table)`` -- ``phi_survival_table`` has
        exactly one key, ``h_true``.
    """
    completeness: CompletenessModel = from_cache_or_build()
    detection_probability = SimulationDetectionProbability(
        injection_data_dir=injection_dir,
        snr_threshold=SNR_THRESHOLD,
        dl_bins=pdet_dl_bins,
        mass_bins=pdet_mass_bins,
        estimator=pdet_estimator,  # type: ignore[arg-type]
        expected_z_max=HOST_DRAW_Z_MAX,
        allow_shallow_pool=allow_low_pdet_coverage,
        pdet_z_resolved=True,
    )
    detection_probability._get_or_build_grid(h_true)
    phi_survival_table = precompute_phi_marginal_survival(
        h_values=[h_true],
        detection_probability_obj=detection_probability,
        z_max_cap=z_max_cap,
    )
    return completeness, phi_survival_table


def draw_isotropic_sky(
    rng: np.random.Generator, n: int
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Isotropic sky direction draw: ``phiS ~ U(0, 2pi)``, ``cos(qS) ~ U(-1, 1)``.

    ``qS`` is the ecliptic polar angle (colatitude) in ``[0, pi]``; uniform
    ``cos(qS)`` is the standard isotropic-on-sphere construction (equal
    solid-angle density per unit ``(phiS, cos qS)``).

    Args:
        rng: Seeded generator (consumes exactly ``2*n`` uniform draws).
        n: Number of directions to draw.

    Returns:
        ``(phiS, qS)``, each shape ``(n,)``.
    """
    phi_s = rng.uniform(0.0, 2.0 * np.pi, size=n)
    q_s = np.arccos(1.0 - 2.0 * rng.uniform(0.0, 1.0, size=n))
    return phi_s, q_s


class MirrorUniverseGenerator:
    """D-B real-catalogue mirror-universe draw.

    Per realization (seed): (a) draws ``n_events`` hosts from the pinned
    catalogue's host pool, weighted by a detection-realistic proxy (see
    :meth:`_host_draw_weights` -- **registered design choice, flagged for
    review**); (b) resamples ``n_events`` ENTIRE per-event Fisher rows (full
    covariance + detected parameters) from the pinned production CRB CSV,
    SNR-weighted, WITHOUT replacement (the donor pool, ~1588 rows, comfortably
    exceeds 200; sampling without replacement avoids duplicate covariance
    structures in one realization); (c) places each event at its host: true
    d_L from the host's z at ``h_true`` (:data:`H_TRUE`), the sky
    localization Gaussian RECENTERED at the host's own (phiS, qS) with the
    resampled row's own (phiS, qS) 2x2 covariance sub-block (this is the
    harness's operational reading of "rotate the Fisher row's localization to
    the host's sky location" -- a RECENTER, not a literal spherical tensor
    rotation of the covariance; **registered design choice, flagged for
    review**), and a correlated draw of the OBSERVED (phiS, qS, d_L) about
    that true position using the row's own covariance (item (c)'s "draw the
    observed d_L from the row's sigma_dL about the true d_L", applied
    identically to the sky sector); (d) host photo-z: B-0 (``sigma_z_scale
    == 1.0``) uses the catalogue's OWN z_obs/z_error columns AS-IS (D-B item
    d: "z_true := the catalogue z_obs treated as exact ... for the mirror
    universe's truth" -- the catalogue's stored photo-z error is already the
    width the host-z kernel is meant to integrate against, so B-0 needs no
    extra re-scattering pass); ``sigma_z_scale`` doses (the B-sigma arm, NOT
    run by this build) are realized via production's own
    :func:`~darksiren_emri.galaxy_catalogue.observed_realization.realize_observed_catalogue`
    (the exact registered mechanism, D-B item d) -- see
    :meth:`host_pool_for_sigma_scale`.

    Mass columns (M, M_error, and their Fisher covariance entries) are left
    at the resampled row's own values -- unlinked to the newly assigned
    host's mass. This is harmless for the registered B-0/B-sigma/S-RAIL
    statistics (all defined on ``combined_no_bh``, which does not consume the
    with-BH-mass branch); flagged for review as a scope limitation.
    """

    def __init__(self, config: CorrespondenceConfig) -> None:
        self.config = config
        self._donor_rows: pd.DataFrame = pd.read_csv(config.crb_reference_csv)
        # AMENDMENT A-3 (B-SEL) diagnostic sidecar: populated only by
        # draw_realization's "population_selected" branch; empty otherwise.
        # See draw_realization's docstring / run_arm_seed's "host_z_quantiles"
        # record.
        self.last_diagnostics: dict[str, Any] = {}

    @staticmethod
    def _host_draw_weights(pool: HostPool) -> npt.NDArray[np.float64]:
        """Detection-realistic host-draw weighting (D-B item a, registered choice).

        w_i proportional to 1 / d_L(z_i, h_true)^2 -- the standard
        inverse-square SNR/flux falloff (nearer hosts are more likely to host
        a DETECTABLE EMRI), evaluated at the mirror truth :data:`H_TRUE`.
        Deliberately decoupled from the independently SNR-weighted Fisher-row
        draw (item b): both draws bias toward "more easily detectable"
        systems in the same physical sense (nearby/high-SNR), without
        requiring a per-row true-distance match, which the "place each event
        at its host" step (item c) makes unnecessary -- the row's own d_L is
        discarded and replaced by ``dist(host_z, h_true)``. Flagged for
        review: this is a SIMPLE proxy (Gray et al. 2020's own catalogue
        weighting is luminosity/rate-based, not distance-only); a
        rate-weighted alternative (``galaxy_catalog.draw_rate_weighted_hosts``,
        already used by the injection side, ``dark_siren_injection.py``) was
        considered and rejected here only because it requires the injection
        machinery's rate-table construction, out of scope for this harness's
        n=200 pilot cost budget.

        The redshift is floored at :data:`HOST_DRAW_WEIGHT_Z_FLOOR` (1e-3)
        before computing the weighting distance -- a bug fix (2026-08-20,
        job 6383719 diagnosis, see the constant's module-level comment): the
        B-sigma arm's ``realize_observed_catalogue`` z-floor point-mass clip
        (1e-5, a documented realization artifact, not a real host) otherwise
        acquires a pathologically dominant 1/d_L^2 weight (~4e7-1e8x a
        typical host) that swamps the WHOLE weighted-without-replacement
        draw at doses (sigma_z_scale=0.25) where enough rows hit the clip.
        This floor sits an order of magnitude below the real GLADE pool's
        empirical minimum z (~0.00195), so genuine hosts are unaffected.

        Args:
            pool: The host pool.

        Returns:
            Normalized weights, shape ``(pool.n,)``.
        """
        z_for_weight = np.clip(pool.z, HOST_DRAW_WEIGHT_Z_FLOOR, None)
        d_l = dist_vectorized(z_for_weight, h=H_TRUE)
        w = 1.0 / np.clip(d_l, 1.0e-6, None) ** 2
        total = w.sum()
        return w / total if total > 0 else np.full(pool.n, 1.0 / pool.n)

    def host_pool_for_sigma_scale(
        self, work_root: Path, seed: int, sigma_z_scale: float
    ) -> tuple[HostPool, str | None, GalaxyCatalogueHandler]:
        """Resolve the host pool + (optional) observed-catalogue path for a dose.

        ``sigma_z_scale == 1.0`` (B-0): the pinned baseline catalogue, as-is
        (D-B item d). ``sigma_z_scale == 0.0`` (G-1's "exact z"): the
        harness's own exact-z variant (:func:`build_exact_z_catalogue`).
        Any other value (the B-sigma arm, NOT run by this build): production's
        ``realize_observed_catalogue`` at that ``sigma_scale``.

        Args:
            work_root: Scratch directory for a written catalogue variant.
            seed: Realization seed (only consumed by the ``realize_observed_catalogue``
                branch).
            sigma_z_scale: The dose.

        Returns:
            ``(host_pool, observed_catalogue_path_or_None, handler)`` -- the
            handler is the SAME object the host pool was extracted from, for
            direct reuse as ``BayesianStatistics.evaluate``'s ``galaxy_catalog``
            argument (G-2 reuse finding: no second candidate-structure build).
        """
        if sigma_z_scale == 1.0:
            handler = _load_galaxy_catalog_handler(REDUCED_CATALOGUE_PATH)
            return _host_pool_from_handler(handler), None, handler
        if sigma_z_scale == 0.0:
            work_root.mkdir(parents=True, exist_ok=True)
            out = str(work_root / "exact_z_catalogue.csv")
            build_exact_z_catalogue(out)
            handler = _load_galaxy_catalog_handler(out)
            return _host_pool_from_handler(handler), out, handler
        from darksiren_emri.galaxy_catalogue.observed_realization import (
            observed_catalogue_filename,
            realize_observed_catalogue,
        )

        work_root.mkdir(parents=True, exist_ok=True)
        out = str(work_root / observed_catalogue_filename(seed))
        realize_observed_catalogue(REDUCED_CATALOGUE_PATH, out, seed, sigma_scale=sigma_z_scale)
        handler = _load_galaxy_catalog_handler(out)
        return _host_pool_from_handler(handler), out, handler

    def draw_realization(
        self,
        seed: int,
        host_pool: HostPool | None = None,
        host_mode: Literal["catalogue", "population", "population_selected"] = "catalogue",
        completeness: CompletenessModel | None = None,
        phi_survival_table: dict[float, tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]]
        | None = None,
    ) -> pd.DataFrame:
        """Draw one mirror-universe realization: ``n_events`` synthetic CRB rows.

        Args:
            seed: Realization seed (drives BOTH the host draw and the row
                draw + noise draws via independent RNG sub-streams, for
                reproducibility -- see the ``test_correspondence_1d.py``
                determinism test).
            host_pool: Pre-resolved host pool (reuse across seeds at the same
                dose via :meth:`host_pool_for_sigma_scale`); defaults to the
                pinned baseline (``sigma_z_scale == 1.0``, B-0). Ignored when
                ``host_mode`` is ``"population"`` or ``"population_selected"``.
            host_mode: ``"catalogue"`` (default, D-B item a): draws hosts
                FROM ``host_pool``, detectability-weighted, and stamps them
                as in-catalogue (``host_galaxy_index >= 0``,
                ``in_catalog=True``). ``"population"`` (AMENDMENT A-2, B-OUT):
                draws host redshift from :func:`draw_population_redshifts`
                and an isotropic sky direction from :func:`draw_isotropic_sky`
                -- NEVER a pinned-catalogue host -- and stamps
                ``host_galaxy_index=-1``, ``in_catalog=False`` (the exact
                production "dark"/completion-leg convention,
                ``bayesian_statistics.py:4485``), so the host is never a
                candidate-set member by construction.
                ``"population_selected"`` (AMENDMENT A-3, B-SEL): same
                bookkeeping convention as ``"population"``, but the host
                redshift is drawn from :func:`draw_selected_population_redshifts`
                (population x (1-completeness) x survival, requires
                ``completeness``/``phi_survival_table``) instead of the bare
                population weight -- see the module docstring's "AMENDMENT
                A-3" section. Also populates :attr:`last_diagnostics` with
                the drawn host-z quantiles (weighted) AND a same-seed
                B-OUT-style unweighted comparison draw's quantiles, so the
                readout can quantify how much the selection weighting moved
                the distribution.
            completeness: Required (only) for ``host_mode="population_selected"``
                -- see :func:`build_bsel_selection_objects`.
            phi_survival_table: Required (only) for
                ``host_mode="population_selected"`` -- see
                :func:`build_bsel_selection_objects`.

        Returns:
            A DataFrame with the SAME columns/order as
            :data:`CRB_CSV_PATH` (:attr:`config.crb_reference_csv`), the
            mirror-universe's ``n_events`` synthetic events.
        """
        n = self.config.n_events
        rng = np.random.default_rng(seed)
        self.last_diagnostics = {}

        # (b) SNR-weighted row draw, without replacement.
        snr = self._donor_rows["SNR"].to_numpy(dtype=np.float64)
        row_p = snr / snr.sum()
        row_idx = rng.choice(len(self._donor_rows), size=n, replace=False, p=row_p)
        rows = self._donor_rows.iloc[row_idx].reset_index(drop=True).copy()

        if host_mode == "catalogue":
            pool = host_pool if host_pool is not None else _load_host_pool(REDUCED_CATALOGUE_PATH)
            # (a) detectability-weighted host draw, without replacement.
            host_w = self._host_draw_weights(pool)
            host_idx = rng.choice(pool.n, size=n, replace=False, p=host_w)
            host_z = pool.z[host_idx]
            host_phiS = pool.phiS[host_idx]
            host_qS = pool.qS[host_idx]
            host_index_col = host_idx.astype(np.int64)
            in_catalog_col = True
        elif host_mode == "population":
            # (a) AMENDMENT A-2 (B-OUT): population-model host draw, never a
            # pinned-catalogue member -- see the module docstring's
            # "Population-model choice" section.
            host_z = draw_population_redshifts(rng, n, h=H_TRUE)
            host_phiS, host_qS = draw_isotropic_sky(rng, n)
            host_index_col = np.full(n, -1, dtype=np.int64)
            in_catalog_col = False
        elif host_mode == "population_selected":
            # (a) AMENDMENT A-3 (B-SEL): the estimator's OWN assumed
            # distribution of DETECTED dark events -- see the module
            # docstring's "AMENDMENT A-3" section.
            if completeness is None or phi_survival_table is None:
                raise ValueError(
                    "host_mode='population_selected' requires both completeness "
                    "and phi_survival_table (build via build_bsel_selection_objects)"
                )
            host_z = draw_selected_population_redshifts(
                rng, n, completeness, phi_survival_table, h=H_TRUE
            )
            host_phiS, host_qS = draw_isotropic_sky(rng, n)
            host_index_col = np.full(n, -1, dtype=np.int64)
            in_catalog_col = False
            # Diagnostic-only comparison draw: a FRESH generator seeded
            # IDENTICALLY (same seed, task spec item 2's "same rng") draws
            # the B-OUT-style UNWEIGHTED w_pop(z) sample -- never consumed by
            # the realization itself (a separate stream, so it cannot
            # perturb the weighted draw above), purely so the readout can
            # show how much the selection weighting moved the host-z
            # distribution relative to the bare population draw.
            diag_rng = np.random.default_rng(seed)
            unweighted_diag_z = draw_population_redshifts(diag_rng, n, h=H_TRUE)
            _levels = (0.05, 0.25, 0.5, 0.75, 0.95)
            self.last_diagnostics = {
                "quantile_levels": list(_levels),
                "host_z_quantiles_weighted": [float(np.quantile(host_z, q)) for q in _levels],
                "host_z_quantiles_unweighted_population": [
                    float(np.quantile(unweighted_diag_z, q)) for q in _levels
                ],
            }
        else:
            raise ValueError(
                f"unknown host_mode {host_mode!r}; expected "
                "'catalogue'/'population'/'population_selected'"
            )

        # (c) true d_L from host z at h_true; observed d_L about it.
        true_d_L = dist_vectorized(host_z, h=H_TRUE)
        sigma_dL = np.sqrt(
            rows["delta_luminosity_distance_delta_luminosity_distance"].to_numpy(dtype=np.float64)
        )
        obs_d_L = true_d_L + rng.normal(size=n) * sigma_dL
        obs_d_L = np.clip(obs_d_L, 1.0e-6, None)

        # (c) sky: correlated draw about the host's true position using the
        # resampled row's own (phiS, qS) 2x2 covariance sub-block, scaled by
        # config.area_scale (E-DEN arm, registered mechanism -- see the
        # module docstring's E-DEN implementation note). For a 2x2 covariance
        # block, localization AREA ~ sqrt(det(cov)); scaling the WHOLE
        # covariance block by a scalar s scales det(cov) by s^2, hence area
        # by s -- so multiplying phi_var/theta_var/cov_theta_phi by
        # area_scale scales the drawn localization area by exactly
        # area_scale, matching the registered "(phi,theta) covariance
        # sub-block scaled by area_scale" mechanism. area_scale == 1.0 is a
        # byte-identical no-op (B-0/B-sigma arms).
        area_scale = float(self.config.area_scale)
        phi_var = rows["delta_phiS_delta_phiS"].to_numpy(dtype=np.float64) * area_scale
        theta_var = rows["delta_qS_delta_qS"].to_numpy(dtype=np.float64) * area_scale
        cov_theta_phi = rows["delta_phiS_delta_qS"].to_numpy(dtype=np.float64) * area_scale
        obs_phiS = np.empty(n, dtype=np.float64)
        obs_qS = np.empty(n, dtype=np.float64)
        for i in range(n):
            cov = np.array([[phi_var[i], cov_theta_phi[i]], [cov_theta_phi[i], theta_var[i]]])
            try:
                chol = np.linalg.cholesky(cov)
            except np.linalg.LinAlgError:
                chol = np.diag([np.sqrt(max(phi_var[i], 0.0)), np.sqrt(max(theta_var[i], 0.0))])
            offset = chol @ rng.normal(size=2)
            obs_phiS[i] = host_phiS[i] + offset[0]
            obs_qS[i] = host_qS[i] + offset[1]
        obs_phiS = np.mod(obs_phiS, 2.0 * np.pi)
        obs_qS = np.clip(obs_qS, 0.0, np.pi)

        rows["luminosity_distance"] = obs_d_L
        rows["phiS"] = obs_phiS
        rows["qS"] = obs_qS
        rows["host_galaxy_index"] = host_index_col
        rows["in_catalog"] = in_catalog_col
        return rows


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


# ── Mirror-universe in-process evaluation driver ─────────────────────────────


def write_mirror_crb_csv(events: pd.DataFrame, out_path: str) -> str:
    """Write a mirror realization's synthetic CRB rows to disk.

    Args:
        events: A :meth:`MirrorUniverseGenerator.draw_realization` result.
        out_path: Destination CSV path.

    Returns:
        ``out_path``.
    """
    events.to_csv(out_path, index=False)
    return out_path


def run_mirror_seed_inprocess(
    work_root: Path,
    events: pd.DataFrame,
    seed: int,
    galaxy_catalog: GalaxyCatalogueHandler,
    h_values: tuple[float, ...] = H_GRID_41,
    completeness_override: bool = False,
    injection_dir: str = INJECTION_POOL_DIR,
    allow_low_pdet_coverage: bool = True,
    selection_in_completion_numerator: str = PRODUCTION_FLAGS[
        "--selection_in_completion_numerator"
    ],
    completion_event_measure: str = "ratio",
) -> tuple[Path, float]:
    """Evaluate one mirror realization in-process (D-A wholesale, no subprocess).

    Calls the REAL ``BayesianStatistics().evaluate(...)`` method directly
    (imported, not reimplemented -- same D-A fidelity as
    :func:`run_production_wholesale`'s subprocess layer, since
    ``darksiren_emri.main.evaluate`` is itself nothing but this call plus CLI
    parsing). Running in-process (rather than via subprocess) buys two
    things the task explicitly asks the harness to exploit: (1) a
    monkeypatchable completeness object for G-1's f=1 shim (impossible over a
    subprocess boundary without a CLI flag, which production does not have --
    see :class:`_UnityCompleteness`); (2) a caller-supplied ``galaxy_catalog``
    that can be REUSED across seeds/h without rebuilding the candidate
    structure (the G-2 reuse finding). No production file is edited by
    either use.

    Args:
        work_root: Scratch directory; only ``work_root/simulations`` needs to
            exist (for the CRB-CSV writer and the diagnostics-CSV output --
            evaluate() writes ``simulations/diagnostics/event_likelihoods.csv``
            relative to the process CWD).
        events: The mirror realization's synthetic CRB rows.
        seed: Realization seed (threaded into ``BayesianStatistics.evaluate``'s
            ``base_seed``, and into ``Model1CrossCheck``'s rng -- the latter
            is structural only here, since ``evaluate()`` does not draw new
            events; it exists to keep the call signature identical to
            production's ``main.py`` construction).
        galaxy_catalog: A pre-built handler (:func:`_load_galaxy_catalog_handler`
            output) -- REUSED, not rebuilt, per realization.
        h_values: The h-grid to evaluate (single ``evaluate()`` call fuses the
            whole grid -- production's own h-list-fusion feature, prereg D-D's
            "context build serves every h" cost anchor).
        completeness_override: If ``True``, monkeypatch
            ``bayesian_statistics.from_cache_or_build`` to return
            :class:`_UnityCompleteness` for the duration of this call (G-1
            only; restored in a ``finally``).
        injection_dir: The pinned injection pool (p_det grid input).
        allow_low_pdet_coverage: Forwarded to ``BayesianStatistics.evaluate``
            (default ``True``, harness-registered). Production's own
            ``evaluate()`` STOPs (``RuntimeError``) if the P_det grid covers
            < 95% of events' 4-sigma d_L windows -- a real production guard
            against a stale/shallow injection pool. It fires harmlessly on
            B-OUT (AMENDMENT A-2): the population-model host draw samples
            the FULL :data:`POPULATION_Z_MAX` domain (unlike the
            catalogue-resident draw, which stays within the pool's
            calibrated depth by construction), so a materially larger
            fraction of events sit near/beyond the injection pool's
            detection horizon by design, not by defect. This is a
            deliberate diagnostic run of the estimator on an
            out-of-calibration universe (a fact of the B-OUT construction
            worth reporting, not silencing) -- ``True`` is a no-op for
            every G-1/G-2/B-0/B-sigma/E-DEN/B-F1 call (none of them ever
            triggered the guard) and only changes behavior for B-OUT.
        selection_in_completion_numerator: Forwarded verbatim to
            ``BayesianStatistics.evaluate`` (AMENDMENT A-4). Default is
            :data:`PRODUCTION_FLAGS`'s registered value ("fused" since the D2
            ruling, row #159; was "off" through 2026-08-21) -- byte-identical
            to every call site that does not pass this kwarg explicitly
            (G-1, G-2, and every pre-A-4 fleet arm). Only ``run_arm_seed``'s
            ``bself`` arm passes ``"fused"`` (via :data:`ARM_SELECTION_CELL`),
            isolating the numerator/denominator detection-model convention
            from B-SEL's otherwise-identical configuration.
        completion_event_measure: Forwarded verbatim to
            ``BayesianStatistics.evaluate`` (AMENDMENT A-5, docs/derivations/
            completion_numerator_data_measure.md §6). Default ``"ratio"`` is
            byte-identical to every call site that does not pass this kwarg
            explicitly (G-1, G-2, and every pre-A-5 fleet arm). Only
            ``run_arm_seed``'s ``bden`` arm passes ``"data"`` (via
            :data:`ARM_EVENT_MEASURE`), isolating the completion numerator's
            event-term measure from B-SEL's otherwise-identical
            configuration.

    Returns:
        ``(diagnostics_csv_path, elapsed_seconds)``.

    Note:
        **Low-wing h-bounds widening (harness-registered mechanism, flagged
        for review).** ``BayesianStatistics.evaluate`` STOPs
        (``ValueError("Hubble constant out of bounds.")``) if any requested
        h falls outside its OWN freshly-constructed ``LamCDMScenario``'s
        registered ``[0.6, 0.86]`` window (``bayesian_statistics.py:3199``,
        ``:3620-3624``) -- this is production's real parameter-space bound,
        unrelated to :data:`H_GRID_41`. The prereg's S-RAIL low wing
        (:data:`H_WING_LOW`, ``[0.50, 0.58]``, REPORTED-ONLY diagnostic) is
        outside it by construction. Since ``bs`` here is a FRESH, throwaway
        ``BayesianStatistics()`` instance local to this call (discarded on
        return -- no shared/module state), this function widens
        ``bs.cosmological_model.h.lower_limit``/``upper_limit`` to cover
        ``min(h_values)``/``max(h_values)`` (a no-op, byte-identical
        widening when ``h_values`` already sits inside ``[0.6, 0.86]``, e.g.
        every G-1/G-2 call) rather than editing production's
        ``LamCDMScenario`` class default -- no production file is touched.
    """
    import darksiren_emri.bayesian_inference.bayesian_statistics as _bs_mod

    sims = work_root / "simulations"
    sims.mkdir(parents=True, exist_ok=True)
    crb_path = sims / "prepared_cramer_rao_bounds.csv"
    write_mirror_crb_csv(events, str(crb_path))
    # true_cramer_rao_bounds.csv is read at __init__ but unused downstream of
    # evaluate() (mirrors _setup_wholesale_cwd's subprocess-route symlink).
    (sims / "cramer_rao_bounds.csv").write_bytes(crb_path.read_bytes())
    _symlink(sims / "injections", Path(injection_dir))

    original_cwd = Path.cwd()
    original_from_cache = _bs_mod.from_cache_or_build
    try:
        if completeness_override:
            _bs_mod.from_cache_or_build = lambda *a, **k: _UnityCompleteness()  # type: ignore[assignment]
        os.chdir(work_root)
        cosmological_model = Model1CrossCheck(rng=np.random.default_rng(seed))
        bs = BayesianStatistics()
        # Low-wing widening (see the docstring Note above): no-op when
        # h_values is already inside bs's own registered [0.6, 0.86] bound.
        bs.cosmological_model.h.lower_limit = min(
            bs.cosmological_model.h.lower_limit, min(h_values)
        )
        bs.cosmological_model.h.upper_limit = max(
            bs.cosmological_model.h.upper_limit, max(h_values)
        )
        start = time.time()
        bs.evaluate(
            galaxy_catalog,
            cosmological_model,
            h_value=h_values[0],
            h_values=list(h_values),
            base_seed=seed,
            pdet_z_resolved=True,
            normalization_mode=PRODUCTION_FLAGS["--normalization_mode"],
            host_z_kernel=PRODUCTION_FLAGS["--host_z_kernel"],
            selection_in_completion_numerator=selection_in_completion_numerator,
            completion_event_measure=completion_event_measure,
            catalogue_mass_overlap=PRODUCTION_FLAGS["--catalogue_mass_overlap"],
            completion_b_scale=PRODUCTION_FLAGS["--completion_b_scale"],
            pdet_dl_bins=int(PRODUCTION_FLAGS["--pdet_dl_bins"]),
            pdet_mass_bins=int(PRODUCTION_FLAGS["--pdet_mass_bins"]),
            pdet_estimator=PRODUCTION_FLAGS["--pdet_estimator"],
            allow_low_pdet_coverage=allow_low_pdet_coverage,
        )
        elapsed = time.time() - start
    finally:
        _bs_mod.from_cache_or_build = original_from_cache
        os.chdir(original_cwd)
    diag_csv = work_root / "simulations" / "diagnostics" / "event_likelihoods.csv"
    if not diag_csv.is_file():
        raise RuntimeError(f"expected diagnostics CSV not found: {diag_csv}")
    return diag_csv, elapsed


# ── Per-seed registered statistics (prereg §2/§3) ────────────────────────────


@dataclass
class SeedStats:
    """Per-seed registered statistics (prereg §2: mean_h, MAP, sigma_h, coverage, R_low).

    Attributes:
        seed: Realization seed.
        n_events: Distinct events contributing to the combine.
        mean_h: Posterior mean over :data:`H_GRID_41`.
        map_h: Posterior mode (argmax) over :data:`H_GRID_41`.
        sigma_h: Posterior std over :data:`H_GRID_41`.
        c50: Whether ``h_true`` falls inside the 50% HPD set.
        c68: Whether ``h_true`` falls inside the 68% HPD set.
        c90: Whether ``h_true`` falls inside the 90% HPD set.
        r_low: DS-6 rail indicator, ``map_h <= R_LOW_THRESHOLD``.
    """

    seed: int
    n_events: int
    mean_h: float
    map_h: float
    sigma_h: float
    c50: bool
    c68: bool
    c90: bool
    r_low: bool


def _hpd_contains(
    post_n: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64],
    target_idx: int,
    level: float,
) -> bool:
    """Whether grid node ``target_idx`` is in the smallest-density-first HPD set at ``level``."""
    order = np.argsort(-post_n)
    cum = 0.0
    for idx in order:
        cum += float(post_n[idx] * weights[idx])
        if idx == target_idx:
            return True
        if cum >= level:
            return False
    return False


ZeroHandling = Literal["physics_floor", "legacy_sentinel"]
MomentWeights = Literal["trapezoid", "legacy_gradient"]


def combine_log_likelihood(
    vals: npt.NDArray[np.float64],
    zero_handling: ZeroHandling = "physics_floor",
) -> npt.NDArray[np.float64]:
    """Sum per-event log-likelihoods over events, handling zero likelihoods.

    ``"physics_floor"`` (default, and the corrected behaviour) reproduces
    production's registered ``CombinationStrategy.PHYSICS_FLOOR``
    (``bayesian_inference/posterior_combination.py`` ``_physics_floor``): a zero
    is replaced by that event's own smallest non-zero likelihood, and an event
    that is zero at *every* node is excluded outright.

    ``"legacy_sentinel"`` reproduces the pre-2026-08-20 behaviour bit-for-bit --
    a ``-1.0e300`` floor applied in LOG space. It is retained ONLY so the banked
    fleet (``results/prod2d_closure_20260818/correspondence_arms/*.json``) stays
    reproducible, which GATE R-0a of AMENDMENT A-7 depends on. Never use it for
    new measurements.

    The two agree bit-for-bit whenever at least one grid node survives -- verified
    ``max|delta mean_h| = 0.000e+00`` across all 98 such banked seeds. They differ
    only when EVERY node carries a zero, where the sentinel produces a finite,
    normalizable vector that a correct ``-inf`` would have refused. See ledger row
    #145 and ``docs/derivations/GATE_PRESENTATION_SENTINEL_COMBINE_20260820.md``.

    Args:
        vals: ``(n_events, n_nodes)`` per-event likelihoods (linear, not log).
        zero_handling: Strategy, as above.

    Returns:
        ``(n_nodes,)`` summed log-likelihood.

    Raises:
        ValueError: If ``zero_handling`` is not a registered strategy.
    """
    if zero_handling == "legacy_sentinel":
        positive = vals > 0.0
        # `out=` pre-fills the non-positive entries, so nothing uninitialized is
        # ever read; bit-identical to the original `np.where(...)` form, which
        # discarded those entries anyway.
        log_l = np.full_like(vals, -np.inf)
        with np.errstate(divide="ignore", invalid="ignore"):
            np.log(vals, where=positive, out=log_l)
        summed: npt.NDArray[np.float64] = np.nansum(
            np.where(np.isfinite(log_l), log_l, -1.0e300), axis=0
        )
        return summed
    if zero_handling != "physics_floor":
        msg = f"unknown zero_handling {zero_handling!r}"
        raise ValueError(msg)

    # posterior_combination._physics_floor (:219-273): per-event min-nonzero
    # floor; events that are zero at every node have no floor and are excluded.
    floored = vals.copy()
    keep = np.ones(floored.shape[0], dtype=bool)
    for i, row in enumerate(floored):
        zero = row == 0.0
        if not zero.any():
            continue
        nonzero = row[~zero & ~np.isnan(row)]
        if nonzero.size == 0:
            keep[i] = False
        else:
            floored[i, zero] = float(nonzero.min())
    if not keep.any():
        return np.full(floored.shape[1], -np.inf, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        total: npt.NDArray[np.float64] = np.log(floored[keep]).sum(axis=0)
    return total


def moment_weights(
    grid: npt.NDArray[np.float64],
    convention: MomentWeights = "trapezoid",
) -> npt.NDArray[np.float64]:
    """Quadrature weights for posterior moments on a (possibly non-uniform) grid.

    ``"trapezoid"`` (default, and the corrected behaviour) is the composite
    trapezoid rule: ``w_i = (h_{i+1} - h_{i-1})/2`` in the interior and
    ``w = delta/2`` at each endpoint, so ``sum(w)`` equals the interval length.

    ``"legacy_gradient"`` reproduces the pre-2026-08-20 ``np.gradient(grid)``,
    which is the central-difference derivative stencil: it matches trapezoid in
    the interior but returns the FULL one-sided spacing at both boundaries,
    doubling each endpoint weight (0.010 vs 0.005 on ``H_GRID_41``) and
    over-counting the interval by one grid step (0.27 vs 0.26). Retained only for
    reproducing the banked fleet.

    Args:
        grid: Monotonically increasing h-grid.
        convention: Weight convention, as above.

    Returns:
        Weights, same shape as ``grid``.

    Raises:
        ValueError: If ``convention`` is not registered.
    """
    if convention == "legacy_gradient":
        return np.gradient(grid)
    if convention != "trapezoid":
        msg = f"unknown moment-weight convention {convention!r}"
        raise ValueError(msg)
    w = np.empty_like(grid)
    w[1:-1] = (grid[2:] - grid[:-2]) / 2.0
    w[0] = (grid[1] - grid[0]) / 2.0
    w[-1] = (grid[-1] - grid[-2]) / 2.0
    return w


def compute_seed_statistics(
    diagnostics_csv: str | Path,
    seed: int,
    h_grid: tuple[float, ...] = H_GRID_41,
    h_true: float = H_TRUE,
    zero_handling: ZeroHandling = "physics_floor",
    weights_convention: MomentWeights = "trapezoid",
) -> SeedStats:
    """Per-seed 1D posterior (Sigma log combined_no_bh, trapezoid) + registered statistics.

    Zero per-event likelihoods are handled by production's registered
    ``PHYSICS_FLOOR`` strategy and moments use true composite-trapezoid weights.
    Both were corrected on 2026-08-20 (ledger row #145,
    ``docs/derivations/GATE_PRESENTATION_SENTINEL_COMBINE_20260820.md``); the
    superseded behaviours remain reachable via ``zero_handling`` /
    ``weights_convention`` solely to reproduce the banked fleet.

    Args:
        diagnostics_csv: A wholesale/in-process run's
            ``event_likelihoods.csv``.
        seed: The realization seed (recorded, not consumed).
        h_grid: The registered production grid (S-RAIL; the low wing, if
            present in the CSV, is excluded here -- REPORTED-ONLY).
        h_true: The mirror-universe truth.
        zero_handling: See :func:`combine_log_likelihood`.
        weights_convention: See :func:`moment_weights`.

    Returns:
        The :class:`SeedStats`.
    """
    df = pd.read_csv(diagnostics_csv)
    grid = np.array(sorted(h_grid), dtype=np.float64)
    df = df[np.isin(df["h"].to_numpy(dtype=np.float64), grid)]
    piv = df.pivot_table(index="event_idx", columns="h", values="combined_no_bh", aggfunc="first")
    piv = piv.reindex(columns=grid)
    vals = piv.to_numpy(dtype=np.float64)
    return seed_statistics_from_matrix(vals, seed, grid, h_true, zero_handling, weights_convention)


def seed_statistics_from_matrix(
    vals: npt.NDArray[np.float64],
    seed: int,
    grid: npt.NDArray[np.float64],
    h_true: float = H_TRUE,
    zero_handling: ZeroHandling = "physics_floor",
    weights_convention: MomentWeights = "trapezoid",
) -> SeedStats:
    """Core ``(n_events, n_nodes)`` likelihood matrix -> :class:`SeedStats` reduction.

    Extracted from :func:`compute_seed_statistics` (pure refactor, 2026-08-21,
    ADDITIVE per the C-SG v3 build task -- no existing arm's behaviour
    changes: :func:`compute_seed_statistics` is now a thin CSV-to-matrix
    wrapper that delegates here) so a caller with a likelihood matrix that is
    NOT the ``combined_no_bh`` column pivot -- e.g.
    :mod:`darksiren_emri.validation.selfgen_control`'s C-SG matched/pure/full
    channel matrices (``B_num/(D_tilde_phi - alpha_G_phi)``,
    ``B_num/D_tilde_phi``) -- gets the IDENTICAL scoring statistics
    (physics-floor zero handling, trapezoid moments, HPD coverage, DS-6 rail)
    without re-deriving them.

    Args:
        vals: ``(n_events, n_nodes)`` per-event likelihoods (linear, not log),
            columns aligned with ``grid``.
        seed: The realization seed (recorded, not consumed).
        grid: The (sorted) h-grid ``vals``' columns are aligned to.
        h_true: The mirror-universe truth.
        zero_handling: See :func:`combine_log_likelihood`.
        weights_convention: See :func:`moment_weights`.

    Returns:
        The :class:`SeedStats`.
    """
    sum_log_l = combine_log_likelihood(vals, zero_handling)

    # Ledger row #145: a seed whose every node is masked carries no information.
    # The superseded -1.0e300 floor turned that into a plausible FINITE vector
    # (mean_h = the grid midpoint, which coincides with H_TRUE) that was banked
    # silently; correct -inf turns it into NaN statistics. Neither is acceptable
    # as a result, so refuse it explicitly rather than emit a number.
    if not np.isfinite(sum_log_l).any():
        msg = (
            f"seed {seed}: no h-node carries a finite summed log-likelihood "
            f"(every node masked by zero per-event likelihoods) -- this seed is "
            f"uninformative and must not be scored"
        )
        raise ValueError(msg)

    weights = moment_weights(grid, weights_convention)
    lp = sum_log_l - sum_log_l.max()
    post = np.exp(lp)
    norm = float((post * weights).sum())
    post_n = post / norm if norm > 0 else post
    mean_h = float((post_n * grid * weights).sum())
    var = float((post_n * (grid - mean_h) ** 2 * weights).sum())
    sigma_h = float(np.sqrt(max(var, 0.0)))
    map_h = float(grid[int(np.argmax(sum_log_l))])

    target_idx_arr = np.nonzero(np.isclose(grid, h_true))[0]
    target_idx = (
        int(target_idx_arr[0]) if target_idx_arr.size else int(np.argmin(np.abs(grid - h_true)))
    )
    c50 = _hpd_contains(post_n, weights, target_idx, 0.50)
    c68 = _hpd_contains(post_n, weights, target_idx, 0.68)
    c90 = _hpd_contains(post_n, weights, target_idx, 0.90)
    r_low = map_h <= R_LOW_THRESHOLD

    return SeedStats(
        seed=seed,
        n_events=int(vals.shape[0]),
        mean_h=mean_h,
        map_h=map_h,
        sigma_h=sigma_h,
        c50=c50,
        c68=c68,
        c90=c90,
        r_low=r_low,
    )


# ── G-1 (mirror sanity, STOP) ─────────────────────────────────────────────────


@dataclass
class G1Result:
    """G-1 result (prereg §4): mirror sanity null.

    Attributes:
        stats: The single-seed :class:`SeedStats`.
        se_proxy: The SE proxy against which ``|mean_h - h_true|`` is
            gated -- ``sigma_h`` itself (a single-realization run has no
            ensemble SE; ``sigma_h`` is the harness's registered-compatible
            stand-in, flagged for review).
        bias: ``mean_h - h_true``.
        verdict: ``"PASS"`` or ``"STOP"``.
    """

    stats: SeedStats
    se_proxy: float
    bias: float
    verdict: str


def run_g1_null(
    work_root: Path,
    seed: int = 900001,
    config: CorrespondenceConfig | None = None,
) -> G1Result:
    """Run gate G-1 (prereg §4): sigma_z_scale -> 0 (exact z) AND f=1 completeness.

    Args:
        work_root: Scratch directory.
        seed: Realization seed.
        config: Optional override (default: the registered n_events=200).

    Returns:
        The :class:`G1Result`.
    """
    cfg = config or CorrespondenceConfig()
    gen = MirrorUniverseGenerator(cfg)
    host_pool, _observed_path, handler = gen.host_pool_for_sigma_scale(
        work_root / "catalogue", seed, sigma_z_scale=0.0
    )
    events = gen.draw_realization(seed, host_pool=host_pool)
    diag_csv, elapsed = run_mirror_seed_inprocess(
        work_root / f"seed{seed}",
        events,
        seed,
        galaxy_catalog=handler,
        h_values=H_GRID_41,
        completeness_override=True,
    )
    _LOGGER.info("G-1 evaluate() elapsed: %.1fs", elapsed)
    stats = compute_seed_statistics(diag_csv, seed)
    bias = stats.mean_h - H_TRUE
    se_proxy = max(stats.sigma_h, 1.0e-6)
    verdict = "PASS" if abs(bias) <= 2.0 * se_proxy else "STOP"
    return G1Result(stats=stats, se_proxy=se_proxy, bias=bias, verdict=verdict)


# ── G-2 (cost pilot, STOP) ─────────────────────────────────────────────────────


@dataclass
class G2Result:
    """G-2 result (prereg §1 D-D): B-0 cost pilot.

    Attributes:
        per_seed_elapsed_seconds: One entry per pilot seed-run.
        anchor_cpu_h: The registered 0.969 CPU-h/seed-run anchor.
        realized_cpu_h_per_seed: Mean of ``per_seed_elapsed_seconds`` in CPU-h
            (single-worker wall time used as the CPU-h proxy -- the harness
            runs single-process; see report notes on multi-worker scaling).
        ratio_to_anchor: ``realized_cpu_h_per_seed / anchor_cpu_h``.
        verdict: ``"PROCEED"`` if ``ratio_to_anchor <= 2.0`` else ``"STOP"``.
    """

    per_seed_elapsed_seconds: list[float]
    anchor_cpu_h: float
    realized_cpu_h_per_seed: float
    ratio_to_anchor: float
    verdict: str


G2_ANCHOR_CPU_H = 0.969
G2_STOP_MULTIPLE = 2.0


def run_g2_cost_pilot(
    work_root: Path,
    seeds: tuple[int, ...] = (900101, 900102),
    config: CorrespondenceConfig | None = None,
) -> G2Result:
    """Run gate G-2 (prereg §1 D-D): 2 full B-0 seed-runs, timed.

    B-0 = production-mapped form, sigma_z_scale = 1.0 (GLADE empirical, the
    catalogue as-is), over the FULL registered grid (:data:`H_GRID_41`).
    The galaxy-catalogue handler is built ONCE and reused across both pilot
    seeds (the G-2 "context reuse across seeds" finding, see module
    docstring / :func:`_load_galaxy_catalog_handler`).

    Args:
        work_root: Scratch directory.
        seeds: Exactly the 2 pilot seeds (D-D).
        config: Optional override (default: the registered n_events=200).

    Returns:
        The :class:`G2Result`.
    """
    cfg = config or CorrespondenceConfig()
    gen = MirrorUniverseGenerator(cfg)
    host_pool, _observed_path, handler = gen.host_pool_for_sigma_scale(
        work_root / "catalogue", seeds[0], sigma_z_scale=1.0
    )
    elapsed_list: list[float] = []
    for seed in seeds:
        events = gen.draw_realization(seed, host_pool=host_pool)
        _diag_csv, elapsed = run_mirror_seed_inprocess(
            work_root / f"seed{seed}",
            events,
            seed,
            galaxy_catalog=handler,
            h_values=H_GRID_41,
            completeness_override=False,
        )
        _LOGGER.info("G-2 seed=%d evaluate() elapsed: %.1fs", seed, elapsed)
        elapsed_list.append(elapsed)
    realized_cpu_h = float(np.mean(elapsed_list)) / 3600.0
    ratio = realized_cpu_h / G2_ANCHOR_CPU_H
    verdict = "PROCEED" if ratio <= G2_STOP_MULTIPLE else "STOP"
    return G2Result(
        per_seed_elapsed_seconds=elapsed_list,
        anchor_cpu_h=G2_ANCHOR_CPU_H,
        realized_cpu_h_per_seed=realized_cpu_h,
        ratio_to_anchor=ratio,
        verdict=verdict,
    )


# ── D-1 diagnostic (AMENDMENT A-6, prereg VERDICT section): does the B-SEL ──
# mirror match the estimator's model AT SURVIVAL TIME, not just at draw
# time? (results/prod2d_closure_20260818/PREREGISTRATION_1D_CORRESPONDENCE.md,
# "AMENDMENT A-6 (registered pre-run) -- test the premise before bisecting
# further".) Zero-compute diagnostic: generator + the SAME quality filter
# BayesianStatistics.__init__ applies, no evaluate() call.

D1_CDF_GAP_BAND: float = 0.05  # AMENDMENT A-6's registered tolerance (row #137).
D1_QUANTILE_LEVELS: tuple[float, ...] = (0.05, 0.25, 0.5, 0.75, 0.95)
D1_OUTPUT_PATH: str = "results/prod2d_closure_20260818/d1_premise_check.json"


def _cumulative_trapezoid(
    y: npt.NDArray[np.float64], x: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Cumulative trapezoidal integral of ``y`` over ``x``, ``cum[0] == 0``.

    A tiny local helper so this module does not add a ``scipy`` dependency
    (:func:`scipy.integrate.cumulative_trapezoid`) for a single call site.

    Args:
        y: Integrand values, shape ``(n,)``.
        x: Strictly increasing abscissas, shape ``(n,)``.

    Returns:
        Cumulative integral, shape ``(n,)``, ``cum[0] == 0``.
    """
    dx = np.diff(x)
    avg = (y[1:] + y[:-1]) / 2.0
    return np.concatenate(([0.0], np.cumsum(avg * dx)))


def _normalized_model_cdf(
    z_grid: npt.NDArray[np.float64], density_grid: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Normalize a (non-negative) density on ``z_grid`` to a CDF in ``[0, 1]``.

    Args:
        z_grid: Strictly increasing redshift grid.
        density_grid: Non-negative density values on ``z_grid``.

    Returns:
        CDF values on ``z_grid``, ``cdf[0] == 0``, ``cdf[-1] == 1``.

    Raises:
        ValueError: If the density integrates to (numerically) zero over
            ``z_grid`` -- the model support is degenerate and no CDF can be
            formed.
    """
    cum = _cumulative_trapezoid(density_grid, z_grid)
    total = cum[-1]
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError(
            f"model density integrates to {total!r} over the grid -- degenerate support"
        )
    return np.asarray(cum / total, dtype=np.float64)


def _max_cdf_gap(
    sample_z: npt.NDArray[np.float64],
    z_grid: npt.NDArray[np.float64],
    density_grid: npt.NDArray[np.float64],
) -> float:
    """Max |CDF gap| (Kolmogorov-Smirnov statistic) between a sample and a model density.

    Standard two-sided KS construction: the model CDF (from
    :func:`_normalized_model_cdf`) is evaluated at each sorted sample point,
    and compared against BOTH the left- and right-continuous empirical CDF
    at that point (the empirical step function jumps exactly at each sample
    value, so both sides of the jump must be checked to find the true max
    gap).

    Args:
        sample_z: Sample redshifts (unsorted; sorted internally). May be
            empty (returns ``nan``).
        z_grid: Strictly increasing redshift grid the model density is
            tabulated on; must cover ``sample_z``'s range (points outside
            are clamped to the grid's own CDF endpoints, 0 or 1).
        density_grid: Non-negative model density values on ``z_grid``.

    Returns:
        The max |CDF gap|, or ``nan`` if ``sample_z`` is empty.
    """
    z_sorted = np.sort(np.asarray(sample_z, dtype=np.float64))
    n = z_sorted.size
    if n == 0:
        return float("nan")
    cdf_grid = _normalized_model_cdf(z_grid, density_grid)
    model_at_points = np.interp(z_sorted, z_grid, cdf_grid, left=0.0, right=1.0)
    emp_upper = np.arange(1, n + 1, dtype=np.float64) / n
    emp_lower = np.arange(0, n, dtype=np.float64) / n
    gap = np.maximum(np.abs(emp_upper - model_at_points), np.abs(emp_lower - model_at_points))
    return float(np.max(gap))


@dataclass
class D1PremiseCheckResult:
    """AMENDMENT A-6 D-1: B-SEL mirror-vs-model correspondence AT SURVIVAL TIME.

    Answers: does the mirror match the estimator's model at SURVIVAL time,
    not just at draw time? B-SEL/B-SELF/B-DEN draw hosts from
    ``w_pop(z)*(1-f_bar(z))*S_bar_phi(z;h_true)`` (the estimator's own
    assumed distribution of detected dark events), but (a) each event then
    takes a donor Fisher row resampled SNR-weighted from real production
    events and (b) production's quality filter
    (``BayesianStatistics.__init__``, ``bayesian_statistics.py:3668-3676``,
    ``use_detection``/``bayesian_statistics.py:5388-5401``: SNR >=
    :data:`~darksiren_emri.constants.SNR_THRESHOLD` AND
    ``d_L_uncertainty/d_L < FRACTIONAL_LUMINOSITY_DISTANCE_ERROR_THRESHOLD``)
    removes ~10% of drawn events. Neither step is part of the estimator's
    selection model, so the REALIZED distribution of SURVIVING mirror events
    need not equal the model density even though the DRAWN distribution does
    by construction.

    Attributes:
        arm: Always ``"bsel"`` (the arm this diagnostic tests; B-SELF/B-DEN
            share the identical host-draw mechanism per their own docstrings
            so the same premise question applies to them, but only bsel is
            measured here per the registered scope).
        seed: The single probed seed (default 900101, B-SEL's first
            registered seed, :data:`ARM_SEEDS`\\ ``["bsel"][0]``).
        n_drawn: Number of hosts drawn (200, the registered D-C dose).
        n_surviving: Number of DRAWN events passing the quality filter --
            THIS is production's own ``n_eff`` decision rule, reproduced
            exactly (not re-derived): SNR threshold then
            ``distance_relative_error`` threshold, both evaluated on the
            SAME per-event quantities (``rows["SNR"]``, the donor row's own
            ``d_L_uncertainty``, and the realization's noisy ``obs_d_L``)
            that ``Detection``/``use_detection`` would compute from the
            identical CRB row.
        survival_fraction: ``n_surviving / n_drawn``.
        max_cdf_gap_surviving_vs_model: **Band-bearing.** Max |CDF gap|
            between the SURVIVING events' empirical z-distribution and the
            model's own normalized detected-dark density
            (:func:`selected_population_z_weights`) on the same support.
        verdict: ``"MIRROR-MATCHED"`` if
            ``max_cdf_gap_surviving_vs_model <= `` :data:`D1_CDF_GAP_BAND`
            (0.05, the pool-vs-events provenance-check tolerance, row #137)
            else ``"MIRROR-MISMATCHED"`` -- the registered AMENDMENT A-6
            bands.
        max_cdf_gap_drawn_vs_model: **Context only, NOT band-bearing.** The
            same statistic for the full DRAWN sample (before filtering) vs
            the model density -- expected small by construction (the draw
            uses inverse-CDF sampling directly from this density). A large
            value here indicates a SEPARATE bug in the draw/quadrature
            itself, not a survival-time effect; see
            :attr:`drawn_vs_model_anomaly`.
        drawn_vs_model_anomaly: ``True`` iff ``max_cdf_gap_drawn_vs_model >``
            :data:`D1_CDF_GAP_BAND` -- flags the draw-time anomaly described
            above.
        z_quantiles: ``{"drawn": [...], "surviving": [...], "model": [...]}``,
            each a list aligned with :attr:`quantile_levels`.
        quantile_levels: The probability levels :attr:`z_quantiles` is
            reported at (:data:`D1_QUANTILE_LEVELS`).
        elapsed_s: Wall-clock seconds for the diagnostic (generator + filter
            + model evaluation only -- no ``evaluate()`` call, expected
            minutes not the ~29-45 min/seed full arm cost).
        git_commit: Repository commit the diagnostic ran at.
    """

    arm: str
    seed: int
    n_drawn: int
    n_surviving: int
    survival_fraction: float
    max_cdf_gap_surviving_vs_model: float
    verdict: Literal["MIRROR-MATCHED", "MIRROR-MISMATCHED"]
    max_cdf_gap_drawn_vs_model: float
    drawn_vs_model_anomaly: bool
    z_quantiles: dict[str, list[float]]
    quantile_levels: list[float]
    elapsed_s: float
    git_commit: str


def run_d1_premise_check(
    seed: int = 900101,
    n_model_grid: int = 2001,
    config: CorrespondenceConfig | None = None,
) -> D1PremiseCheckResult:
    r"""Run the AMENDMENT A-6 D-1 diagnostic for one B-SEL seed.

    Reuses the SAME generative building blocks the ``bsel`` arm's
    :func:`run_arm_seed` uses -- :func:`build_bsel_selection_objects`
    (production's own completeness/``S_bar_phi`` construction) and
    :func:`draw_selected_population_redshifts` (the host-z draw) -- but
    replicates only the ``population_selected`` prefix of
    :meth:`MirrorUniverseGenerator.draw_realization` (the SNR-weighted donor
    row draw, then the host-z/sky draw, then the noisy ``d_L`` placement)
    so the drawn host redshift ``z`` can be kept as a named local array
    (:meth:`~MirrorUniverseGenerator.draw_realization` does not return it --
    only the resulting CRB-row DataFrame). The two draws consume the SAME
    seeded ``rng`` in the SAME order :meth:`draw_realization` does (row draw
    before host-z draw, per its own "(b) ... (a) ..." comments), so the
    events/host-z pairing here is exactly what a real ``bsel`` seed run at
    this ``seed`` would produce -- this function does NOT call
    :meth:`draw_realization` itself only because that function does not
    expose ``host_z``.

    Then applies production's OWN quality filter
    (``BayesianStatistics.__init__``, ``bayesian_statistics.py:3668-3676``)
    verbatim: SNR >= :data:`~darksiren_emri.constants.SNR_THRESHOLD`, then
    ``distance_relative_error = d_L_uncertainty / d_L <``
    ``FRACTIONAL_LUMINOSITY_DISTANCE_ERROR_THRESHOLD`` (the exact
    ``use_detection`` predicate, ``bayesian_statistics.py:5388-5401``) --
    does NOT run ``evaluate()``.

    Args:
        seed: The probed seed (default 900101, B-SEL's first registered
            seed).
        n_model_grid: Resolution of the ``z`` grid the model density
            (:func:`selected_population_z_weights`) is evaluated on for the
            CDF-gap/quantile computation.
        config: Optional override (default: the registered ``n_events=200``,
            B-SEL's ``ARM_SPECS`` dose ``(1.0, 1.0)`` -- ``sigma_z_scale``/
            ``area_scale`` are irrelevant to this diagnostic's host-z draw,
            carried only for schema parity).

    Returns:
        The :class:`D1PremiseCheckResult`.
    """
    t0 = time.time()
    cfg = config or CorrespondenceConfig(sigma_z_scale=1.0, area_scale=1.0)
    n = cfg.n_events
    gen = MirrorUniverseGenerator(cfg)
    completeness_obj, phi_survival_table = build_bsel_selection_objects()

    rng = np.random.default_rng(seed)
    # (b) SNR-weighted donor-row draw, without replacement -- IDENTICAL to
    # draw_realization's item (b), consumed FIRST from the same rng stream.
    snr = gen._donor_rows["SNR"].to_numpy(dtype=np.float64)
    row_p = snr / snr.sum()
    row_idx = rng.choice(len(gen._donor_rows), size=n, replace=False, p=row_p)
    rows = gen._donor_rows.iloc[row_idx].reset_index(drop=True).copy()

    # (a) AMENDMENT A-3 (B-SEL) host-z draw, consumed SECOND -- IDENTICAL to
    # draw_realization's "population_selected" branch.
    host_z = draw_selected_population_redshifts(
        rng, n, completeness_obj, phi_survival_table, h=H_TRUE
    )
    draw_isotropic_sky(rng, n)  # sky draw: consumed for rng-stream parity, unused here.

    # (c) true d_L from host z at h_true; observed d_L about it -- IDENTICAL
    # to draw_realization's item (c) distance placement.
    true_d_L = dist_vectorized(host_z, h=H_TRUE)
    sigma_dL = np.sqrt(
        rows["delta_luminosity_distance_delta_luminosity_distance"].to_numpy(dtype=np.float64)
    )
    obs_d_L = true_d_L + rng.normal(size=n) * sigma_dL
    obs_d_L = np.clip(obs_d_L, 1.0e-6, None)
    snr_col = rows["SNR"].to_numpy(dtype=np.float64)

    # Production's OWN quality filter, verbatim (bayesian_statistics.py
    # :3668-3676 SNR filter + :5388-5401 use_detection's distance_relative_error
    # threshold) -- this IS how the real arm decides n_eff.
    passes_snr = snr_col >= SNR_THRESHOLD
    distance_relative_error = sigma_dL / obs_d_L
    passes_quality = distance_relative_error < FRACTIONAL_LUMINOSITY_DISTANCE_ERROR_THRESHOLD
    surviving_mask = passes_snr & passes_quality

    drawn_z = host_z
    surviving_z = host_z[surviving_mask]
    n_surviving = int(surviving_mask.sum())

    z_grid = np.linspace(POPULATION_Z_MIN, POPULATION_Z_MAX, n_model_grid, dtype=np.float64)
    model_density = selected_population_z_weights(
        z_grid, completeness_obj, phi_survival_table, h=H_TRUE
    )
    model_cdf = _normalized_model_cdf(z_grid, model_density)

    gap_surviving = _max_cdf_gap(surviving_z, z_grid, model_density)
    gap_drawn = _max_cdf_gap(drawn_z, z_grid, model_density)
    verdict: Literal["MIRROR-MATCHED", "MIRROR-MISMATCHED"] = (
        "MIRROR-MATCHED" if gap_surviving <= D1_CDF_GAP_BAND else "MIRROR-MISMATCHED"
    )
    anomaly = bool(gap_drawn > D1_CDF_GAP_BAND)

    model_quantiles = [float(np.interp(q, model_cdf, z_grid)) for q in D1_QUANTILE_LEVELS]
    surviving_quantiles = (
        [float(np.quantile(surviving_z, q)) for q in D1_QUANTILE_LEVELS]
        if n_surviving > 0
        else [float("nan")] * len(D1_QUANTILE_LEVELS)
    )
    z_quantiles = {
        "drawn": [float(np.quantile(drawn_z, q)) for q in D1_QUANTILE_LEVELS],
        "surviving": surviving_quantiles,
        "model": model_quantiles,
    }

    elapsed = time.time() - t0
    return D1PremiseCheckResult(
        arm="bsel",
        seed=seed,
        n_drawn=n,
        n_surviving=n_surviving,
        survival_fraction=n_surviving / n if n > 0 else float("nan"),
        max_cdf_gap_surviving_vs_model=gap_surviving,
        verdict=verdict,
        max_cdf_gap_drawn_vs_model=gap_drawn,
        drawn_vs_model_anomaly=anomaly,
        z_quantiles=z_quantiles,
        quantile_levels=list(D1_QUANTILE_LEVELS),
        elapsed_s=elapsed,
        git_commit=_git_commit(),
    )


# ── Fleet arm-runner stage (cluster fleet execution, task spec item 1) ──────


def compute_full_log_posterior_vector(
    diagnostics_csv: str | Path,
    h_grid: tuple[float, ...] = H_GRID_FULL,
    zero_handling: ZeroHandling = "physics_floor",
) -> tuple[list[float], list[float]]:
    """Full (41-node production grid + REPORTED-ONLY low wing) log-posterior vector.

    Same aggregation as :func:`compute_seed_statistics` (Sigma log
    ``combined_no_bh`` over events, per h-node, no normalization/shift
    applied) but over :data:`H_GRID_FULL` rather than the production-only
    :data:`H_GRID_41` subset, so the fleet JSON carries the full vector the
    prereg's S-RAIL diagnostic wing needs (task spec item 1: "the full 41+low
    -wing log-posterior vector").

    Args:
        diagnostics_csv: A mirror-seed run's ``event_likelihoods.csv``
            (must have been evaluated over a superset of ``h_grid``, e.g. via
            :func:`run_mirror_seed_inprocess` with ``h_values=H_GRID_FULL``).
        h_grid: The full grid (default: production 41-node grid + low wing).
        zero_handling: See :func:`combine_log_likelihood`. Defaults to the
            corrected ``"physics_floor"``; ``"legacy_sentinel"`` reproduces the
            banked fleet's vectors bit-for-bit (ledger row #145).

    Returns:
        ``(h_grid_sorted, sum_log_l)`` -- parallel lists, same length as
        ``h_grid``.
    """
    df = pd.read_csv(diagnostics_csv)
    grid = np.array(sorted(h_grid), dtype=np.float64)
    df = df[np.isin(df["h"].to_numpy(dtype=np.float64), grid)]
    piv = df.pivot_table(index="event_idx", columns="h", values="combined_no_bh", aggfunc="first")
    piv = piv.reindex(columns=grid)
    vals = piv.to_numpy(dtype=np.float64)
    sum_log_l = combine_log_likelihood(vals, zero_handling)
    return grid.tolist(), sum_log_l.tolist()


def _git_commit() -> str:
    """Current HEAD short-circuit lookup (recorded per fleet-task JSON, best-effort)."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(_REPO_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, OSError):
        return "unknown"


def run_arm_seed(
    work_root: Path,
    arm: str,
    seed: int,
    out_dir: Path,
    config: CorrespondenceConfig | None = None,
) -> Path:
    """Run one (arm, seed) fleet task (cluster-fleet CLI stage, task spec item 1).

    Maps ``arm`` to ``(sigma_z_scale, area_scale)`` via :data:`ARM_SPECS`,
    draws one mirror-universe realization at that dose, evaluates it
    in-process over :data:`H_GRID_FULL` (production 41-node grid + the
    REPORTED-ONLY diagnostic low wing), and writes
    ``<out_dir>/<arm>_seed<seed>.json``.

    **Idempotent** (walltime-kill safety, task spec item 1): if the output
    JSON already exists, the function returns immediately WITHOUT rebuilding
    the catalogue or re-running ``evaluate()`` -- a resubmitted array task
    resumes rather than re-paying the ~29 min/seed-run cost.

    Args:
        work_root: Per-task scratch directory for the sandboxed evaluate()
            run (catalogue variant + CRB-CSV write + diagnostics output).
        arm: One of :data:`ARM_SPECS`' keys
            (``b0``/``bsig005``/``bsig025``/``eden05``/``eden2``/``bout``/
            ``bf1``/``bsel``/``bself``/``bden``).
        seed: Realization seed. Expected to be a member of
            ``ARM_SEEDS[arm]`` per the registered paired-seed discipline
            (prereg §1 D-C) -- not enforced here (kept testable with
            arbitrary seeds); the sbatch task-list construction is the
            actual enforcement point.
        out_dir: Fleet output directory (one JSON per task).
        config: Optional override (default: the registered n_events=200,
            with ``sigma_z_scale``/``area_scale`` taken from
            ``ARM_SPECS[arm]``).

    Returns:
        The (written-or-pre-existing) JSON path.

    Raises:
        KeyError: If ``arm`` is not a registered arm.
    """
    if arm not in ARM_SPECS:
        raise KeyError(f"unknown arm {arm!r}; registered arms: {sorted(ARM_SPECS)}")
    sigma_z_scale, area_scale = ARM_SPECS[arm]
    host_mode = ARM_HOST_MODE.get(arm, "catalogue")
    unity_completeness = ARM_UNITY_COMPLETENESS.get(arm, False)
    selection_cell = ARM_SELECTION_CELL.get(arm, "off")
    event_measure = ARM_EVENT_MEASURE.get(arm, "ratio")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{arm}_seed{seed}.json"
    if out_path.is_file():
        _LOGGER.info(
            "arm=%s seed=%d: output already exists, skipping (idempotent) -- %s",
            arm,
            seed,
            out_path,
        )
        return out_path

    cfg = config or CorrespondenceConfig(sigma_z_scale=sigma_z_scale, area_scale=area_scale)
    catalogue_pin_ok = check_reduced_catalogue_pin()
    gen = MirrorUniverseGenerator(cfg)
    # bout/bsel (host_mode="population"/"population_selected") still resolve
    # the pinned-catalogue handler (for the REAL GLADE candidate structure
    # evaluate() searches -- impostors only, AMENDMENT A-2/A-3) but
    # host_pool itself is ignored by draw_realization's population(_selected)
    # branches.
    host_pool, _observed_path, handler = gen.host_pool_for_sigma_scale(
        work_root / "catalogue", seed, sigma_z_scale=sigma_z_scale
    )
    if host_mode == "population_selected":
        # AMENDMENT A-3 (B-SEL): build the completeness/S_bar_phi weighting
        # objects BEFORE drawing the realization -- see the module
        # docstring's "AMENDMENT A-3" section and
        # :func:`build_bsel_selection_objects`'s docstring for exactly which
        # production construction calls this reuses and why the ordering
        # (not the total cost) is what the amendment requires.
        completeness_obj, phi_survival_table = build_bsel_selection_objects()
        events = gen.draw_realization(
            seed,
            host_pool=host_pool,
            host_mode=host_mode,
            completeness=completeness_obj,
            phi_survival_table=phi_survival_table,
        )
    else:
        events = gen.draw_realization(seed, host_pool=host_pool, host_mode=host_mode)
    diag_csv, elapsed = run_mirror_seed_inprocess(
        work_root / f"seed{seed}",
        events,
        seed,
        galaxy_catalog=handler,
        h_values=H_GRID_FULL,
        completeness_override=unity_completeness,
        selection_in_completion_numerator=selection_cell,
        completion_event_measure=event_measure,
    )
    stats = compute_seed_statistics(diag_csv, seed, h_grid=H_GRID_41)
    h_grid, log_posterior = compute_full_log_posterior_vector(diag_csv, h_grid=H_GRID_FULL)

    # AMENDMENT A-2 sanity assertion (B-OUT): the host must NEVER be a
    # candidate-set member by construction -- host_galaxy_index == -1 is
    # production's own "dark"/completion-leg convention
    # (bayesian_statistics.py:4485), so this fraction is the exact,
    # zero-extra-cost check that the population draw never smuggled a
    # catalogue-resident host in. Recorded for every arm (not just bout) so
    # the JSON is directly comparable across the fleet: b0/bsig*/eden*/bf1
    # (host_mode="catalogue") must read 1.0; bout/bsel must read ~0.0.
    host_in_catalogue_fraction = float((events["host_galaxy_index"].to_numpy() >= 0).mean())

    record: dict[str, Any] = {
        "arm": arm,
        "seed": seed,
        "sigma_z_scale": sigma_z_scale,
        "area_scale": area_scale,
        "host_mode": host_mode,
        # AMENDMENT A-3 (task spec item 2): an explicitly named alias of
        # host_mode, so a bsel JSON is self-describing without cross-
        # referencing ARM_HOST_MODE.
        "host_draw_mode": host_mode,
        "unity_completeness": unity_completeness,
        # AMENDMENT A-4: which selection_in_completion_numerator convention
        # produced this row -- "off" for every pre-A-4 arm (byte-identical
        # to the runs-of-record basis), "fused" only for bself, so a reader
        # can verify which convention a given JSON was scored under without
        # cross-referencing ARM_SELECTION_CELL.
        "selection_cell": selection_cell,
        # AMENDMENT A-5: which completion_event_measure convention produced
        # this row -- "ratio" for every pre-A-5 arm (byte-identical to the
        # runs-of-record basis), "data" only for bden, so a reader can verify
        # which convention a given JSON was scored under without
        # cross-referencing ARM_EVENT_MEASURE.
        "event_measure": event_measure,
        "host_in_catalogue_fraction": host_in_catalogue_fraction,
        "n_events_drawn": cfg.n_events,
        "n_eff": stats.n_events,
        "mean_h": stats.mean_h,
        "map_h": stats.map_h,
        "sigma_h": stats.sigma_h,
        "c50": stats.c50,
        "c68": stats.c68,
        "c90": stats.c90,
        "r_low": stats.r_low,
        "h_grid": h_grid,
        "log_posterior": log_posterior,
        "elapsed_s": elapsed,
        "git_commit": _git_commit(),
        "catalogue_pin_ok": catalogue_pin_ok,
    }
    if host_mode == "population_selected":
        # AMENDMENT A-3 diagnostic (task spec item 2): weighted (B-SEL) vs
        # unweighted-w_pop (B-OUT-style, same seed) host-z quantiles --
        # quantifies how much the selection weighting moved the drawn
        # distribution. See MirrorUniverseGenerator.draw_realization's
        # "population_selected" branch for how this is built.
        record["host_z_quantiles"] = gen.last_diagnostics
    out_path.write_text(json.dumps(record, indent=2))
    _LOGGER.info(
        "arm=%s seed=%d: wrote %s (elapsed=%.1fs, n_eff=%d, mean_h=%.4f)",
        arm,
        seed,
        out_path,
        elapsed,
        stats.n_events,
        stats.mean_h,
    )
    return out_path


def _cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("g0", "g1", "g2", "arm", "d1"), default="g0")
    parser.add_argument(
        "--work-root",
        default="/tmp/correspondence_1d_g0",
        help="Scratch directory for the sandboxed evaluate() run.",
    )
    parser.add_argument("--seed", type=int, default=777010)
    parser.add_argument(
        "--arm",
        choices=tuple(ARM_SPECS),
        default=None,
        help=(
            "Fleet arm (--stage arm only): "
            "b0/bsig005/bsig025/eden05/eden2/bout/bf1/bsel/bself/bden."
        ),
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Fleet output directory (--stage arm only): "
        "<out-dir>/<arm>_seed<seed>.json is written (idempotent).",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    if args.stage == "arm":
        if args.arm is None or args.out_dir is None:
            parser.error("--stage arm requires --arm and --out-dir")
        out_path = run_arm_seed(Path(args.work_root), args.arm, args.seed, Path(args.out_dir))
        print(json.dumps({"out_path": str(out_path)}, indent=2))
        return 0

    if args.stage == "d1":
        # AMENDMENT A-6 D-1: default seed 900101 (bsel's first registered
        # seed) unless the caller explicitly passed --seed.
        d1_seed = args.seed if "--seed" in sys.argv else 900101
        d1 = run_d1_premise_check(seed=d1_seed)
        out_path = Path(D1_OUTPUT_PATH)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(dataclasses.asdict(d1), indent=2))
        print(
            json.dumps(
                {
                    "verdict": d1.verdict,
                    "max_cdf_gap_surviving_vs_model": d1.max_cdf_gap_surviving_vs_model,
                    "max_cdf_gap_drawn_vs_model": d1.max_cdf_gap_drawn_vs_model,
                    "drawn_vs_model_anomaly": d1.drawn_vs_model_anomaly,
                    "n_drawn": d1.n_drawn,
                    "n_surviving": d1.n_surviving,
                    "survival_fraction": d1.survival_fraction,
                    "z_quantiles": d1.z_quantiles,
                    "out_path": str(out_path),
                },
                indent=2,
            )
        )
        return 0 if d1.verdict == "MIRROR-MATCHED" else 1

    if args.stage == "g0":
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
    if args.stage == "g1":
        g1 = run_g1_null(Path(args.work_root), seed=args.seed)
        print(
            json.dumps(
                {
                    "verdict": g1.verdict,
                    "bias": g1.bias,
                    "se_proxy": g1.se_proxy,
                    "mean_h": g1.stats.mean_h,
                    "map_h": g1.stats.map_h,
                    "sigma_h": g1.stats.sigma_h,
                    "n_events": g1.stats.n_events,
                },
                indent=2,
            )
        )
        return 0 if g1.verdict == "PASS" else 1
    g2 = run_g2_cost_pilot(Path(args.work_root))
    print(
        json.dumps(
            {
                "verdict": g2.verdict,
                "per_seed_elapsed_seconds": g2.per_seed_elapsed_seconds,
                "realized_cpu_h_per_seed": g2.realized_cpu_h_per_seed,
                "anchor_cpu_h": g2.anchor_cpu_h,
                "ratio_to_anchor": g2.ratio_to_anchor,
            },
            indent=2,
        )
    )
    return 0 if g2.verdict == "PROCEED" else 1


if __name__ == "__main__":
    raise SystemExit(_cli())
