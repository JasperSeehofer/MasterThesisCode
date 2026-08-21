r"""C-SG v3 -- the self-generated positive control (matched channel primary).

Implements the generator + scoring/gates registered in
``results/prod2d_closure_20260818/PREREGISTRATION_SELFGEN_CONTROL.md`` (v2
sections 0-10, "design B" of section 2, UNCHANGED by v3) with the v3
scoring-channel design (the appended "C-SG v3 -- DESIGN CHANGE" block: the
MATCHED channel ``B_num/(D_tilde_phi - alpha_G_phi)`` is PRIMARY; full-mixture
and pure ``B_num/D_tilde_phi`` are reported-only).

**HARD CONSTRAINT (author mandate, carried from the launch task).** This
module is FORBIDDEN from running the registered C-SG measurement: no scoring
of C-SG seeds against a band, no ``mean_h`` fleet statement. Every gate/scorer
here is a pure function over whatever seed(s) the CALLER chooses to run; the
pilot mandate (prereg section 6, "MANDATORY 4-seed C-SG-F PILOT") and the
post-pilot band-setting discipline are the orchestrator's decision, not this
module's -- this module only supplies the machinery the pilot needs.

Generator (prereg section 2, design B -- reused verbatim across csgf/csge/
csgdm/csgdp, only ``h_gen``/sigma-mode differ):

1. **(z, Omega) jointly** proportional to ``w_pop(z) * (1 - f_k(Omega;z;h_gen))``.
   **Recon-1 flagged this as needing new code** (no existing convenience
   sampler drew ``(z, Omega)`` jointly per-pixel); a wider search of this
   build found one: :func:`darksiren_emri.dark_siren_injection._draw_dark_hosts_pixelated`
   is *exactly* this draw (FIX-A Change 5.5, arXiv:2111.04629 Sec. V) --
   production's own out-of-catalogue EMRI-host sampler for the injection
   pipeline, reused here WHOLESALE (D-A fidelity: the SAME per-pixel
   inverse-CDF machinery real dark-siren injections use, not a
   harness-reimplemented approximation of it). No deviation from recon-1's
   "correct closest factorization" fallback was needed because the exact
   object already existed; this is disclosed as a correction of recon-1's
   coverage, not a design deviation.
2. **mass** ``log10 M ~ phi``, the SAME
   :func:`~darksiren_emri.bayesian_inference.bayesian_statistics._phi_dark_mass_log10_grid`
   the estimator contracts, drawn via the shared
   :func:`~darksiren_emri.validation.correspondence_1d._inverse_cdf_draw`
   (grid/weight-agnostic, reused unmodified).
3. **selection**, accept ONCE with
   ``S_4D(d_L(z;h_gen), M(1+z))`` --
   :meth:`~darksiren_emri.bayesian_inference.simulation_detection_probability.SimulationDetectionProbability.detection_probability_with_bh_mass_interpolated`,
   the SAME with-BH-mass survival object production's own
   ``S_bar_phi``/completion-numerator machinery is built from -- never also
   weighted by ``S_bar_phi`` (prereg section 0 item 2's fix). Implemented as
   batched rejection sampling (**new code**, prereg section 2's "accept ONCE"
   has no existing convenience wrapper): each batch draws candidates via (1)
   and (2), computes ``S_4D`` per candidate, accepts with probability
   ``S_4D`` via a uniform draw from the SAME seeded stream, and keeps
   batching until ``n_events`` are accepted (capped, raises with a diagnostic
   message on runaway rejection -- see :data:`CSGConfig.max_batches`).
4. **measurement sigma**, per arm (prereg section 3): C-SG-F/C-SG-delta-\*
   (``sigma_mode="fixed"``) use ``sigma_dL := 0.0373 * d_L(z;h_gen)``
   (:data:`SIGMA_FRAC_FIXED`, the pinned CRB pool's median sigma_frac,
   STATED explicitly -- never ``0.0373 * d_hat``); C-SG-E
   (``sigma_mode="empirical"``) draws sigma_frac i.i.d. from the pinned CRB
   pool (:func:`_empirical_sigma_frac_pool`), independent of z.
5. **observation is LINEAR** (prereg section 0 item 1, the v1-overturning
   finding): ``d_hat = d_L(z;h_gen) + sigma_dL * eps``, ``eps ~ N(0,1)`` --
   never the ratio-kernel draw v1 proposed.

Donor-row frame (prereg section 5). Each accepted event still borrows an
ENTIRE Fisher row (SNR-weighted, WITH replacement across batches -- a
disclosed deviation from B-SEL's without-replacement draw; see the
"Registered deviations" section below) from the pinned CRB pool, so the
full 128-column Fisher structure (sky covariance block, all cross-terms) is
inherited exactly as B-SEL's is, EXCEPT: ``luminosity_distance`` (the drawn
``d_hat``), ``phiS``/``qS`` (the drawn sky position), ``M`` (the drawn,
REDSHIFTED ``M(1+z)`` -- so the ball-tree candidate search,
``bayesian_statistics.py:4443-4453``, uses the self-consistent mass), and
``host_galaxy_index``/``in_catalog`` (the production dark/completion-leg
bookkeeping convention, always ``-1``/``False``) are overwritten, and (GATE Q
mandate) the 13 ``d_L``-linked Fisher cross-covariance columns
(:data:`DL_CROSS_COV_COLUMNS`) are rescaled by ``sigma_dL / sigma_donor`` --
linear in one marginal sigma with the correlation held fixed -- to keep the
3x3/4x4 Fisher blocks positive-definite (applied UNCONDITIONALLY; attrition
before/after is reported by :func:`gate_q`, never silently accepted above
1%). **FIX ROUND (2026-08-21, adversarial finding #1, MAJOR):** the SAME
rescale is now applied to the M-linked Fisher block
(:data:`M_CROSS_COV_COLUMNS` plus ``delta_M_delta_M`` and the M-component of
``delta_luminosity_distance_delta_M``) by ``m_z / m_donor`` -- the drawn,
REDSHIFTED mass can differ from the donor's own mass by up to ~3 orders of
magnitude (``M_SOURCE_FRAME_MIN``/``_MAX`` span 1e4-1e7 source-frame,
:mod:`darksiren_emri.constants`), and production's own ball-tree candidate
search (``bayesian_statistics.py:4443-4453``) uses
``M_uncertainty = sqrt(delta_M_delta_M)`` as the mass-window half-width, so
an unrescaled mass-Fisher block would leak the donor's mass-scale-mismatched
precision into the impostor-candidate search exactly as an unrescaled
``d_L`` block would leak into the distance-window search. The catalogue-sector
"impostor leg" (real GLADE galaxies falling in the localization cone) is NOT
removed -- same disclosed scope limitation as B-SEL (prereg section 5).

**Registered deviations from the launch task's recon (all disclosed loudly,
per the task's own instruction):**

1. **GATE H's literal function-anchor citations
   (``draw_selected_population_redshifts(:1213)``,
   ``build_bsel_selection_objects(h_true=h_gen)(:894)``) do not apply to C-SG
   v3's actual call graph.** Prereg section 0 item 2's fix REMOVES
   ``S_bar_phi`` from the ``(z, Omega)`` proposal density entirely (recon-1
   section 2 independently reached the same conclusion: "``draw_selected_
   population_redshifts``/``selected_population_z_weights`` ... cannot be
   reused unmodified for C-SG"). C-SG v3's h_gen instead threads through
   :func:`~darksiren_emri.dark_siren_injection._draw_dark_hosts_pixelated`
   (the joint proposal), :func:`~darksiren_emri.physical_relations.dist_vectorized`
   (true d_L), :func:`build_csg_selection_objects` (this module's h_gen-keyed
   analog of ``build_bsel_selection_objects``, same ``lru_cache(maxsize=4)``
   pattern), and :func:`~darksiren_emri.validation.correspondence_1d.compute_seed_statistics`/
   :func:`~darksiren_emri.validation.correspondence_1d.seed_statistics_from_matrix`
   at scoring time. :func:`gate_h` STILL calls ``build_bsel_selection_objects
   (h_true=h_gen)`` -- but only to report ``S_bar_phi``'s ``z_max(h_gen)`` as
   a REPORTED-ONLY diagnostic (prereg section 6 GATE H's second clause,
   which IS generator-agnostic); the C-SG generator itself never consumes the
   returned ``phi_survival_table``.
2. **Donor-row draw is WITH replacement, per batch** (unlike B-SEL's single
   without-replacement draw of exactly ``n_events`` rows). Batched rejection
   sampling draws an a-priori-unknown number of candidates (accept rate
   depends on ``h_gen``/``S_4D`` coverage), so a single without-replacement
   budget over the ~1590-row pool cannot be fixed in advance without either
   under- or over-provisioning it; with-replacement SNR-weighting is the
   simplest correct fix and the pool (~1590 rows) is drawn from far below
   exhaustion in every registered arm's expected regime (typical accept
   rates keep total candidates drawn to a low multiple of ``n_events``, per
   :data:`CSGConfig` batching parameters).
3. **GATE V's ``sigma_prior`` is a registered NEW convention**, not specified
   numerically anywhere in the prereg: the std of ``Uniform(0.6, 0.86)`` (the
   registered ``H_GRID_41``/production h-prior support), ``(0.86-0.6)/
   sqrt(12)``. Flagged for author review in :func:`gate_v`'s output
   (``sigma_prior_convention`` field).
4. **GATE D's model target reuses**
   :func:`~darksiren_emri.validation.correspondence_1d.selected_population_z_weights`
   (B-SEL's exact model-density function) rather than re-deriving a new
   density function, because the two are mathematically IDENTICAL for the
   ACCEPTED z-marginal here: integrating design B's accepted joint density
   ``w_pop(z) (1 - f_k(Omega;z)) phi(log10 M) S_4D(d_L(z),M(1+z))`` over
   Omega (isotropic ``S_4D``, so the pixel sum of ``1 - f_k`` recovers
   ``npix (1 - f_bar(z))``) and over mass (``INTEGRAL phi S_4D dlog10 M =
   S_bar_phi(z;h_gen)`` by :func:`~darksiren_emri.bayesian_inference.bayesian_statistics.precompute_phi_marginal_survival`'s
   own defining identity) gives EXACTLY ``w_pop(z) (1-f_bar(z;h_gen))
   S_bar_phi(z;h_gen)`` -- ``selected_population_z_weights``'s formula,
   verbatim. This is a derived consistency fact, not a coincidence of
   convenience, and is worth a fresh author [RULE] read if it is ever relied
   on beyond a diagnostic gate.

CPU-only. No cupy import, direct or transitive.

References:
    Gray et al. (2020), arXiv:1908.06050, Eqs. (29), (32), (A.9), (A.10).
    Gray, Messenger & Veitch (2022), arXiv:2111.04629, Sec. V (pixelated dark
    host draw).
    Mandel, Farr & Gair (2019), arXiv:1809.02063, Eqs. (5)-(7).
    Massey (1951), JASA 46(253):68-78 (asymptotic two-sided KS critical
    value, :func:`ks_d_crit`).
"""

import argparse
import dataclasses
import functools
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
import numpy.typing as npt
import pandas as pd

from darksiren_emri.bayesian_inference.bayesian_statistics import (
    FRACTIONAL_LUMINOSITY_DISTANCE_ERROR_THRESHOLD,
    _check_covariance_quality,
    _phi_dark_mass_log10_grid,
)
from darksiren_emri.bayesian_inference.simulation_detection_probability import (
    SimulationDetectionProbability,
)
from darksiren_emri.constants import HOST_DRAW_Z_MAX, SNR_THRESHOLD
from darksiren_emri.dark_siren_injection import _draw_dark_hosts_pixelated
from darksiren_emri.galaxy_catalogue.pixel_completeness import from_cache_or_build
from darksiren_emri.physical_relations import dist_vectorized
from darksiren_emri.validation import correspondence_1d as c1d
from darksiren_emri.validation.correspondence_1d import (
    CRB_CSV_PATH,
    H_GRID_41,
    H_GRID_FULL,
    H_TRUE,
    INJECTION_POOL_DIR,
    POPULATION_Z_MAX,
    POPULATION_Z_MIN,
    REDUCED_CATALOGUE_PATH,
    SeedStats,
    _inverse_cdf_draw,
    _max_cdf_gap,
)

_LOGGER = logging.getLogger(__name__)

_POPULATION_Z_GRID_N = 4001  # matches correspondence_1d's private module constant (B-SEL parity)


# ── Structural typing (D-B/A-3 parity: the completeness object satisfies both
# the narrower pixel_completeness.CompletenessModel Protocol B-SEL uses AND
# the extra per-pixel methods dark_siren_injection._draw_dark_hosts_pixelated
# needs). Declared locally (rather than importing dark_siren_injection's
# private _PixelDarkSampler Protocol) so this module's public signatures do
# not depend on another module's private name; structurally identical, so
# any concrete object satisfying this satisfies both call sites. The one
# production return type, PixelCompleteness, satisfies it directly (no
# cast needed).
class CsgCompletenessModel(Protocol):
    """Structural type: everything C-SG's generator + B-SEL diagnostics need."""

    @property
    def npix(self) -> int: ...

    def f_k(
        self, z: float | npt.NDArray[np.floating[Any]], k: int, h: float = ...
    ) -> float | npt.NDArray[np.float64]: ...

    def f_bar(
        self, z: float | npt.NDArray[np.floating[Any]], h: float = ...
    ) -> float | npt.NDArray[np.floating[Any]]: ...

    def ang2pix(self, phi: float, theta: float) -> int: ...

    def get_completeness_at_redshift(
        self, z: float | npt.NDArray[np.floating[Any]], h: float = ...
    ) -> float | npt.NDArray[np.floating[Any]]: ...

    def pixel_dark_weights(
        self,
        z_grid: npt.NDArray[np.float64],
        p_pop: npt.NDArray[np.float64],
        h: float,
    ) -> npt.NDArray[np.float64]: ...

    def sample_sky_in_pixels(
        self, pix: npt.NDArray[np.int_], rng: np.random.Generator
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: ...


class CsgDetectionProbabilityModel(Protocol):
    """Structural type: the two ``SimulationDetectionProbability`` methods C-SG needs.

    Declared locally (narrower than the concrete class) so the generator/gate
    functions accept ANY object exposing these two methods -- including a
    lightweight test fake -- rather than requiring the real, pool-backed
    :class:`~darksiren_emri.bayesian_inference.simulation_detection_probability.SimulationDetectionProbability`.
    The one production instance satisfies it directly (no cast needed).
    """

    def detection_probability_with_bh_mass_interpolated(
        self,
        d_L: float | npt.NDArray[np.float64],
        M_z: float | npt.NDArray[np.float64],
        phi: float | npt.NDArray[np.float64],
        theta: float | npt.NDArray[np.float64],
        *,
        h: float,
        z: float | npt.NDArray[np.float64] | None = ...,
    ) -> float | npt.NDArray[np.float64]: ...

    def get_dl_max(self, h: float) -> float: ...


# ── Arm registry (prereg section 4) ──────────────────────────────────────────
CSG_H_GEN: dict[str, float] = {
    "csgf": 0.73,
    "csge": 0.73,
    "csgdm": 0.68,
    "csgdp": 0.78,
}
CSG_SIGMA_MODE: dict[str, Literal["fixed", "empirical"]] = {
    "csgf": "fixed",
    "csge": "empirical",
    "csgdm": "fixed",
    "csgdp": "fixed",
}
# Seeds start at 910101 (prereg section 4). Counts per the registered table:
# C-SG-F 15, C-SG-E 15, C-SG-delta- 8, C-SG-delta+ 8 (46 total).
CSG_SEEDS: dict[str, tuple[int, ...]] = {
    "csgf": tuple(range(910101, 910101 + 15)),
    "csge": tuple(range(910101, 910101 + 15)),
    "csgdm": tuple(range(910101, 910101 + 8)),
    "csgdp": tuple(range(910101, 910101 + 8)),
}
CSG_ARMS: tuple[str, ...] = tuple(CSG_H_GEN)

# C-SG-F "stated explicitly" sigma_frac (prereg section 3): the pinned CRB
# reference (n=1590) empirical median, NOT re-derived here (the pin/loader is
# correspondence_1d.CRB_CSV_PATH/CRB_CSV_MD5, reused unmodified).
SIGMA_FRAC_FIXED: float = 0.0373

# bayesian_statistics.py:3251 evaluate()'s own default -- reused for parity
# so GATE Q's offline PD-exclusion replica applies the SAME threshold
# production's real evaluate() call would.
FISHER_COND_THRESHOLD: float = 1e16

# GATE Q mandate (prereg section 6): the 13 luminosity_distance cross-
# covariance columns of the pinned CRB CSV's 128, EXCLUDING the variance
# itself (delta_luminosity_distance_delta_luminosity_distance, forced
# directly to sigma_dL**2, not rescaled) and the value column
# (luminosity_distance, overwritten with d_hat, not a covariance). Verified
# directly against the pinned CSV header (2026-08-21 recon).
DL_CROSS_COV_COLUMNS: tuple[str, ...] = (
    "delta_luminosity_distance_delta_M",
    "delta_luminosity_distance_delta_mu",
    "delta_luminosity_distance_delta_a",
    "delta_luminosity_distance_delta_p0",
    "delta_luminosity_distance_delta_e0",
    "delta_luminosity_distance_delta_x0",
    "delta_qS_delta_luminosity_distance",
    "delta_phiS_delta_luminosity_distance",
    "delta_qK_delta_luminosity_distance",
    "delta_phiK_delta_luminosity_distance",
    "delta_Phi_phi0_delta_luminosity_distance",
    "delta_Phi_theta0_delta_luminosity_distance",
    "delta_Phi_r0_delta_luminosity_distance",
)

# FIX ROUND (adversarial finding #1, MAJOR): the M-linked Fisher block, the
# direct mass-analog of DL_CROSS_COV_COLUMNS. M itself is overwritten with a
# self-consistent draw from phi (:data:`~darksiren_emri.constants.M_SOURCE_FRAME_MIN`/
# ``_MAX`` span 1e4-1e7 source-frame -- up to ~3 orders of magnitude from the
# SNR-weighted donor's own mass) but the donor's mass-scale Fisher curvature
# (``delta_M_delta_M``, and its cross-terms with sky position) was, before
# this fix, inherited UNRESCALED -- the same class of defect GATE Q's dL
# mandate exists to catch, just on the mass axis instead of the distance
# axis. ``delta_luminosity_distance_delta_M`` is deliberately EXCLUDED here
# (kept in :data:`DL_CROSS_COV_COLUMNS` above): it is rescaled by BOTH
# ratios (applied in sequence in :func:`draw_csg_realization`), since both
# its marginal sigmas change.
M_CROSS_COV_COLUMNS: tuple[str, ...] = (
    "delta_phiS_delta_M",
    "delta_qS_delta_M",
)

# GATE Q mandate: "any arm above 1% attrition from that cut is redesigned,
# not run."
CSG_GATE_Q_NONPD_BAND: float = 0.01

# GATE V (prereg section 6, applied to the matched-channel posterior per v3
# item 5). GATE V AMENDMENT 1 (2026-08-21): v2's (5.0, 0.5) were full-channel
# thresholds; ported unchanged they false-failed 5/12 banked B-SEL matched
# posteriors (known-informative reference) and STOPped 3/4 pilot seeds. The
# amended values target the flat-null vacuity signature (span=0, ratio=1.0);
# reference false-fail 0/16, and the B-F1 flat mode still fails both prongs.
# See gate_v's docstring + the prereg's "PILOT GATE V AMENDMENT" block.
CSG_GATE_V_MIN_SPAN_NATS: float = 1.0
CSG_GATE_V_SIGMA_PRIOR_FRACTION: float = 0.9

# GATE D band: D_crit(alpha=5%) at the ACTUAL n, never the retired fixed
# 0.05 (prereg section 6). See ks_d_crit.
CSG_D1_ALPHA: float = 0.05

# Channel columns pivoted from the per-event diagnostics CSV (v3 item 4:
# every C-SG seed banks these so matched/pure/full are recomputable at zero
# compute -- production's own _write_diagnostic_csv column set, no per-arm
# configuration needed; see bayesian_statistics.py:4343-4363).
_CSG_CHANNEL_COLUMNS: tuple[str, ...] = ("combined_no_bh", "B_num", "alpha_G_phi", "D_tilde_phi")


@dataclass(frozen=True)
class CSGConfig:
    """Registered-shape config for the C-SG generator/runner.

    Attributes:
        n_events: Events per realization (D-C parity: 200).
        oversample_factor: Rejection-sampling batch-size multiplier on the
            remaining accept deficit (new machinery; prereg section 2 stage 3
            specifies WHAT to accept with, not a batching strategy).
        batch_floor: Minimum candidates drawn per batch (keeps the first,
            typically highest-deficit, batch from being too small to be
            vectorization-efficient).
        max_batches: Safety cap; :func:`draw_csg_realization` raises
            ``RuntimeError`` with a diagnostic message if exceeded (a
            registered accept/reject failure, not silently truncated data).
    """

    n_events: int = 200
    oversample_factor: int = 8
    batch_floor: int = 256
    max_batches: int = 500


def _h_gen_for_arm(arm: str) -> float:
    """Resolve ``arm -> h_gen``, raising on an unregistered arm."""
    if arm not in CSG_H_GEN:
        msg = f"unknown C-SG arm {arm!r}; registered arms: {sorted(CSG_H_GEN)}"
        raise KeyError(msg)
    return CSG_H_GEN[arm]


@functools.lru_cache(maxsize=4)
def build_csg_selection_objects(
    h_gen: float = H_TRUE,
    injection_dir: str = INJECTION_POOL_DIR,
    pdet_dl_bins: int = 60,
    pdet_mass_bins: int = 40,
    pdet_estimator: str = "local_linear",
    allow_low_pdet_coverage: bool = True,
) -> tuple[CsgCompletenessModel, CsgDetectionProbabilityModel]:
    r"""Build ``(completeness, detection_probability)`` at ``h_gen`` -- C-SG's weighting objects.

    The h_gen-keyed analog of
    :func:`~darksiren_emri.validation.correspondence_1d.build_bsel_selection_objects`
    -- SAME production construction calls
    (``bayesian_statistics.py:3654-3673`` for the
    :class:`~darksiren_emri.bayesian_inference.simulation_detection_probability.SimulationDetectionProbability`
    constructor, ``:3704`` for
    :func:`~darksiren_emri.galaxy_catalogue.pixel_completeness.from_cache_or_build`),
    but ALSO returns the constructed ``detection_probability`` object itself
    (bsel's version discards it after building the phi-marginal survival
    table; C-SG's accept step needs direct access to
    ``detection_probability_with_bh_mass_interpolated`` -- see the module
    docstring's "Registered deviations" item 1 for why
    ``precompute_phi_marginal_survival`` is NOT called here).
    ``functools.lru_cache``-d exactly like ``build_bsel_selection_objects``
    (``maxsize=4``): the four registered arms use 3 distinct ``h_gen`` values
    (csgf/csge share 0.73), comfortably inside the cache.

    Args:
        h_gen: The mirror-universe generating truth for this arm.
        injection_dir: The pinned injection pool.
        pdet_dl_bins: :data:`~darksiren_emri.validation.correspondence_1d.PRODUCTION_FLAGS`
            ``["--pdet_dl_bins"]`` value.
        pdet_mass_bins: :data:`~darksiren_emri.validation.correspondence_1d.PRODUCTION_FLAGS`
            ``["--pdet_mass_bins"]`` value.
        pdet_estimator: :data:`~darksiren_emri.validation.correspondence_1d.PRODUCTION_FLAGS`
            ``["--pdet_estimator"]`` value.
        allow_low_pdet_coverage: Forwarded to ``SimulationDetectionProbability``
            (default ``True``, harness-registered -- see GATE H's silenced-STOP
            report).

    Returns:
        ``(completeness, detection_probability)``.
    """
    completeness = from_cache_or_build()
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
    detection_probability._get_or_build_grid(h_gen)
    return completeness, detection_probability


@functools.lru_cache(maxsize=1)
def _empirical_sigma_frac_pool(crb_path: str = CRB_CSV_PATH) -> npt.NDArray[np.float64]:
    """C-SG-E's i.i.d. sigma_frac pool (prereg section 3): the pinned CRB reference, n=1590.

    Args:
        crb_path: The pinned CRB CSV (default: the registered
            :data:`~darksiren_emri.validation.correspondence_1d.CRB_CSV_PATH`).

    Returns:
        Read-only ``sigma_frac = sqrt(delta_luminosity_distance_delta_luminosity_distance)
        / luminosity_distance`` array, one value per donor row.
    """
    df = pd.read_csv(crb_path)
    sigma = np.sqrt(
        df["delta_luminosity_distance_delta_luminosity_distance"].to_numpy(dtype=np.float64)
    )
    d_l = df["luminosity_distance"].to_numpy(dtype=np.float64)
    pool: npt.NDArray[np.float64] = np.asarray(sigma / d_l, dtype=np.float64)
    pool.setflags(write=False)
    return pool


def draw_csg_realization(
    seed: int,
    arm: str,
    n_events: int,
    completeness: CsgCompletenessModel,
    detection_probability: CsgDetectionProbabilityModel,
    donor_rows: pd.DataFrame,
    oversample_factor: int = 8,
    batch_floor: int = 256,
    max_batches: int = 500,
    rescale_cross_covariance: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    r"""Draw one C-SG realization: design B (prereg section 2), v3 unchanged.

    See the module docstring for the full per-stage derivation. Determinism:
    a single seeded ``rng = default_rng(seed)`` drives every draw (candidate
    batches, donor-row indices, sigma_frac draws for C-SG-E, the observation
    noise ``eps``) in a fixed order, so the same ``seed``/``arm`` reproduces
    byte-identically.

    Args:
        seed: Realization seed.
        arm: One of :data:`CSG_H_GEN`'s keys (``csgf``/``csge``/``csgdm``/
            ``csgdp``); resolves ``h_gen``/``sigma_mode``.
        n_events: Number of ACCEPTED events to return.
        completeness: See :func:`build_csg_selection_objects`.
        detection_probability: See :func:`build_csg_selection_objects`.
        donor_rows: The pinned CRB reference (128-column Fisher-row pool);
            typically ``pd.read_csv(correspondence_1d.CRB_CSV_PATH)``.
        oversample_factor: See :class:`CSGConfig`.
        batch_floor: See :class:`CSGConfig`.
        max_batches: See :class:`CSGConfig`.
        rescale_cross_covariance: If ``True`` (default, the GATE Q mandate),
            rescale :data:`DL_CROSS_COV_COLUMNS` by ``sigma_dL/sigma_donor``.
            ``False`` reproduces the UN-rescaled draw with the SAME rng
            consumption (the rescale is a pure post-hoc column transform, no
            extra randomness) -- used by :func:`gate_q` to report attrition
            before vs. after, deterministically paired.

    Returns:
        ``(rows, diagnostics)`` -- ``rows`` has the SAME 128 columns as
        ``donor_rows`` (no columns added/dropped, matching B-SEL's
        convention); ``diagnostics`` carries the drawn ``z``/``d_L``/
        ``sigma_dL``/mass arrays and batching/acceptance bookkeeping for the
        gates.

    Raises:
        KeyError: If ``arm`` is not registered.
        RuntimeError: If ``max_batches`` is exceeded before ``n_events``
            candidates are accepted.
    """
    h_gen = _h_gen_for_arm(arm)
    sigma_mode = CSG_SIGMA_MODE[arm]
    rng = np.random.default_rng(seed)

    log10_m_grid, _m_grid, phi, _z_phi = _phi_dark_mass_log10_grid()
    snr = donor_rows["SNR"].to_numpy(dtype=np.float64)
    row_p = snr / snr.sum()

    z_parts: list[npt.NDArray[np.float64]] = []
    phi_s_parts: list[npt.NDArray[np.float64]] = []
    q_s_parts: list[npt.NDArray[np.float64]] = []
    log10_m_parts: list[npt.NDArray[np.float64]] = []
    donor_idx_parts: list[npt.NDArray[np.int64]] = []
    n_candidates_drawn = 0
    n_true_accepted = 0  # every candidate with u < S_4D, even if surplus-discarded
    n_accepted = 0  # candidates actually kept into the output (capped at n_events)
    n_batches = 0

    while n_accepted < n_events:
        n_batches += 1
        if n_batches > max_batches:
            msg = (
                f"C-SG arm={arm!r} seed={seed}: accept/reject failed to reach "
                f"n_events={n_events} after {max_batches} batches "
                f"({n_accepted}/{n_events} accepted, {n_candidates_drawn} candidates "
                f"drawn); h_gen={h_gen} -- S_4D coverage may be too shallow, check the "
                "injection pool / oversample_factor."
            )
            raise RuntimeError(msg)
        remaining = n_events - n_accepted
        batch = max(remaining * oversample_factor, batch_floor)

        # Stage 1: (z, Omega) jointly ∝ w_pop(z)*(1-f_k(Omega;z;h_gen)) --
        # production's own FIX-A joint dark-host sampler, reused wholesale.
        z_cand, phi_s_cand, q_s_cand = _draw_dark_hosts_pixelated(
            batch,
            rng,
            completeness,
            h_gen,
            POPULATION_Z_MIN,
            POPULATION_Z_MAX,
            _POPULATION_Z_GRID_N,
        )
        # Stage 2: log10 M ~ phi, the estimator's own grid + the shared
        # inverse-CDF sampler.
        log10_m_cand = _inverse_cdf_draw(rng, batch, log10_m_grid, phi)
        m_cand = 10.0**log10_m_cand
        d_l_cand = np.asarray(dist_vectorized(z_cand, h=h_gen), dtype=np.float64)
        m_z_cand = m_cand * (1.0 + z_cand)

        # Stage 3: accept ONCE with S_4D(d_L(z;h_gen), M(1+z)) -- never also
        # weighted by S_bar_phi (prereg section 0 item 2).
        zeros = np.zeros(batch, dtype=np.float64)
        s4d = np.asarray(
            detection_probability.detection_probability_with_bh_mass_interpolated(
                d_l_cand, m_z_cand, zeros, zeros, h=h_gen
            ),
            dtype=np.float64,
        )
        s4d = np.clip(s4d, 0.0, 1.0)
        u = rng.uniform(0.0, 1.0, size=batch)
        accept_mask = u < s4d

        # Donor-row draw (SNR-weighted, WITH replacement per batch -- see the
        # module docstring's "Registered deviations" item 2).
        donor_idx_cand = rng.choice(len(donor_rows), size=batch, replace=True, p=row_p)
        n_candidates_drawn += batch
        n_true_accepted += int(accept_mask.sum())

        take_idx = np.flatnonzero(accept_mask)[: max(remaining, 0)]
        if take_idx.size:
            z_parts.append(z_cand[take_idx])
            phi_s_parts.append(phi_s_cand[take_idx])
            q_s_parts.append(q_s_cand[take_idx])
            log10_m_parts.append(log10_m_cand[take_idx])
            donor_idx_parts.append(donor_idx_cand[take_idx])
            n_accepted += int(take_idx.size)

    z = np.concatenate(z_parts)[:n_events]
    phi_s = np.concatenate(phi_s_parts)[:n_events]
    q_s = np.concatenate(q_s_parts)[:n_events]
    log10_m = np.concatenate(log10_m_parts)[:n_events]
    donor_idx = np.concatenate(donor_idx_parts)[:n_events]
    m_source = 10.0**log10_m
    m_z = m_source * (1.0 + z)

    rows = donor_rows.iloc[donor_idx].reset_index(drop=True).copy()

    # Stage 4: measurement sigma.
    d_l_true = np.asarray(dist_vectorized(z, h=h_gen), dtype=np.float64)
    if sigma_mode == "fixed":
        sigma_dl = SIGMA_FRAC_FIXED * d_l_true
    elif sigma_mode == "empirical":
        pool = _empirical_sigma_frac_pool()
        sigma_frac_draw = rng.choice(pool, size=n_events, replace=True)
        sigma_dl = sigma_frac_draw * d_l_true
    else:  # pragma: no cover -- unreachable given CSG_SIGMA_MODE's registered values
        msg = f"unknown sigma_mode {sigma_mode!r}"
        raise ValueError(msg)

    # Stage 5: LINEAR observation (prereg section 0 item 1) -- d_hat = d_L(z;h_gen) + sigma_dL*eps.
    eps = rng.normal(size=n_events)
    d_hat = d_l_true + sigma_dl * eps
    d_hat = np.clip(d_hat, 1.0e-6, None)

    # GATE Q mandate: rescale the d_L cross-covariance columns proportionally
    # to sigma_new/sigma_donor before overwriting the variance itself.
    sigma_donor = np.sqrt(
        rows["delta_luminosity_distance_delta_luminosity_distance"].to_numpy(dtype=np.float64)
    )
    ratio = sigma_dl / np.clip(sigma_donor, 1.0e-300, None)
    if rescale_cross_covariance:
        for col in DL_CROSS_COV_COLUMNS:
            rows[col] = rows[col].to_numpy(dtype=np.float64) * ratio

    rows["delta_luminosity_distance_delta_luminosity_distance"] = sigma_dl**2

    # FIX ROUND (adversarial finding #1, MAJOR): rescale the M-linked Fisher
    # block by the SAME sigma-ratio convention as the d_L block above, before
    # M itself is overwritten below. Convention: hold the donor's FRACTIONAL
    # mass precision fixed (sigma_M/M unchanged) across the mass-scale jump
    # -- the direct M-analog of holding the d_L correlation coefficient
    # fixed while sigma_dL changes. ``m_donor`` is read BEFORE the "M"
    # column is overwritten with ``m_z`` two lines below.
    m_donor = rows["M"].to_numpy(dtype=np.float64)
    ratio_m = m_z / np.clip(m_donor, 1.0e-300, None)
    if rescale_cross_covariance:
        # delta_luminosity_distance_delta_M already picked up the d_L-ratio
        # factor in the DL_CROSS_COV_COLUMNS loop above; apply the M-ratio
        # factor on top of it (both marginal sigmas change for this term).
        rows["delta_luminosity_distance_delta_M"] = (
            rows["delta_luminosity_distance_delta_M"].to_numpy(dtype=np.float64) * ratio_m
        )
        for col in M_CROSS_COV_COLUMNS:
            rows[col] = rows[col].to_numpy(dtype=np.float64) * ratio_m
    rows["delta_M_delta_M"] = rows["delta_M_delta_M"].to_numpy(dtype=np.float64) * ratio_m**2

    rows["luminosity_distance"] = d_hat
    rows["phiS"] = phi_s
    rows["qS"] = q_s
    rows["M"] = m_z
    rows["host_galaxy_index"] = np.full(n_events, -1, dtype=np.int64)
    rows["in_catalog"] = False

    diagnostics: dict[str, Any] = {
        "arm": arm,
        "h_gen": h_gen,
        "h_gen_threaded_to_joint_draw": h_gen,
        "h_gen_threaded_to_dist_vectorized": h_gen,
        "sigma_mode": sigma_mode,
        "n_events": n_events,
        "n_candidates_drawn": n_candidates_drawn,
        "n_true_accepted": n_true_accepted,
        "n_batches": n_batches,
        # The TRUE empirical S_4D acceptance rate (every candidate with
        # u < S_4D, even the surplus discarded once n_events is reached) --
        # NOT n_events/n_candidates_drawn, which would be deflated by
        # oversampling waste whenever S_4D is high (a candidate batch can
        # accept far more than the remaining deficit needs).
        "accept_rate": (n_true_accepted / n_candidates_drawn)
        if n_candidates_drawn
        else float("nan"),
        "z_true": z,
        "d_L_true": d_l_true,
        "sigma_dL": sigma_dl,
        "M_source": m_source,
        "M_z": m_z,
        "rescale_cross_covariance": rescale_cross_covariance,
        "cross_covariance_ratio": ratio,
        "cross_covariance_ratio_m": ratio_m,
    }
    return rows, diagnostics


# ── Scoring: matched/pure/full channels (prereg section 7 / v3 item 4) ──────


def _pivot_columns(
    df: pd.DataFrame, cols: tuple[str, ...], h_grid: tuple[float, ...]
) -> dict[str, npt.NDArray[np.float64]]:
    """Pivot ``event_idx x h`` for each of ``cols`` (mirrors ``decompose_matched_channel.py``)."""
    grid = np.array(sorted(h_grid), dtype=np.float64)
    df = df[np.isin(df["h"].to_numpy(dtype=np.float64), grid)]
    out: dict[str, npt.NDArray[np.float64]] = {}
    for c in cols:
        out[c] = (
            df.pivot_table(index="event_idx", columns="h", values=c, aggfunc="first")
            .reindex(columns=grid)
            .to_numpy(dtype=np.float64)
        )
    return out


def csg_channel_matrices(
    diagnostics_csv: str | Path, h_grid: tuple[float, ...] = H_GRID_41
) -> dict[str, npt.NDArray[np.float64]]:
    r"""Build the full/matched/pure per-event likelihood matrices from a diagnostics CSV.

    ``full = combined_no_bh`` (production's registered no-BH-mass combine);
    ``matched = B_num / beta_Gbar_phi`` where ``beta_Gbar_phi = D_tilde_phi -
    alpha_G_phi`` (``bayesian_statistics.py:2427``'s mixture-normalization
    split, PRIMARY per v3 item 1); ``pure = B_num / D_tilde_phi`` (pre-check
    O2's channel, reported-only per v3 item 2).

    Args:
        diagnostics_csv: A C-SG (or any arm's) ``event_likelihoods.csv``.
        h_grid: The h-grid to pivot onto (default: production ``H_GRID_41``).

    Returns:
        ``{"full", "matched", "pure", "alpha_G_phi", "D_tilde_phi", "B_num",
        "beta_Gbar_phi"}``, each ``(n_events, n_nodes)``.
    """
    df = pd.read_csv(diagnostics_csv)
    cols = _pivot_columns(df, _CSG_CHANNEL_COLUMNS, h_grid)
    alpha, dtil, bnum = cols["alpha_G_phi"], cols["D_tilde_phi"], cols["B_num"]
    beta_gbar = dtil - alpha
    with np.errstate(divide="ignore", invalid="ignore"):
        matched = bnum / beta_gbar
        pure = bnum / dtil
    return {
        "full": cols["combined_no_bh"],
        "matched": matched,
        "pure": pure,
        "alpha_G_phi": alpha,
        "D_tilde_phi": dtil,
        "B_num": bnum,
        "beta_Gbar_phi": beta_gbar,
    }


def gate_t_h_only(
    diagnostics_csv: str | Path,
    h_grid: tuple[float, ...] = H_GRID_41,
    tol: float = 2.0e-6,
) -> dict[str, Any]:
    r"""GATE T (pre-check O3): ``alpha_G_phi``/``D_tilde_phi`` are h-only across events.

    Reproduces ``decompose_matched_channel.py``'s GATE T exactly: max relative
    spread across events, per h-node, of ``alpha_G_phi`` and ``D_tilde_phi``
    must be ``<= tol`` (O2 GATE AMENDMENT 1's re-set tolerance, 3x the 7-sig-fig
    storage-precision bound), AND ``beta_Gbar_phi = D_tilde_phi - alpha_G_phi
    > 0`` everywhere.

    Args:
        diagnostics_csv: A C-SG seed's ``event_likelihoods.csv``.
        h_grid: The h-grid to check (default: production ``H_GRID_41``).
        tol: Max relative spread tolerance (default: the registered 2e-6).

    Returns:
        ``{"max_rel_spread", "min_beta_gbar_phi", "tol", "pass"}``.
    """
    df = pd.read_csv(diagnostics_csv)
    cols = _pivot_columns(df, ("alpha_G_phi", "D_tilde_phi"), h_grid)
    alpha, dtil = cols["alpha_G_phi"], cols["D_tilde_phi"]
    beta_gbar = dtil - alpha

    def _max_rel_spread(mat: npt.NDArray[np.float64]) -> float:
        lo = np.nanmin(mat, axis=0)
        hi = np.nanmax(mat, axis=0)
        scale = np.maximum(np.abs(lo), np.finfo(float).tiny)
        return float(np.max((hi - lo) / scale))

    spread = max(_max_rel_spread(alpha), _max_rel_spread(dtil))
    min_beta = float(np.nanmin(beta_gbar))
    return {
        "max_rel_spread": spread,
        "min_beta_gbar_phi": min_beta,
        "tol": tol,
        "pass": spread <= tol and min_beta > 0.0,
    }


def csg_channel_scores(
    diagnostics_csv: str | Path,
    seed: int,
    h_grid: tuple[float, ...] = H_GRID_41,
    h_true: float = H_TRUE,
) -> dict[str, Any]:
    r"""Matched/pure/full posterior moments (item 7) via production's own reduction.

    Uses :func:`~darksiren_emri.validation.correspondence_1d.seed_statistics_from_matrix`
    (the ``compute_seed_statistics`` core, extracted so it applies to ANY
    ``(n_events, n_nodes)`` likelihood matrix, not just the ``combined_no_bh``
    pivot) -- so ``physics_floor`` zero-handling and trapezoid moment weights
    are IDENTICAL to every other arm's scoring, on all three channels.

    Args:
        diagnostics_csv: A C-SG seed's ``event_likelihoods.csv``.
        seed: The realization seed (recorded, not consumed).
        h_grid: The h-grid to score on (default: production ``H_GRID_41``;
            GATE V forbids scoring on the banked 46-node grid).
        h_true: The value ``h_true`` in each channel's ``SeedStats`` is read
            against -- pass ``h_gen`` for the arm under test (C-SG's own
            "truth" for coverage purposes).

    Returns:
        ``{"full": {...}, "matched": {...}, "pure": {...}, "gate_t": {...}}``,
        each channel a ``dataclasses.asdict(SeedStats)``.
    """
    mats = csg_channel_matrices(diagnostics_csv, h_grid)
    grid = np.array(sorted(h_grid), dtype=np.float64)
    out: dict[str, Any] = {}
    for channel in ("full", "matched", "pure"):
        stats = c1d.seed_statistics_from_matrix(mats[channel], seed, grid, h_true)
        out[channel] = dataclasses.asdict(stats)
    out["gate_t"] = gate_t_h_only(diagnostics_csv, h_grid)
    return out


def score_at_h_gen(
    vals: npt.NDArray[np.float64],
    h_gen: float,
    h_grid: tuple[float, ...] = H_GRID_41,
) -> dict[str, Any]:
    r"""The primary statistic (prereg section 6): per-event score at ``h_gen``.

    Central difference of ``ln(likelihood)`` over the two h-grid nodes
    immediately bracketing ``h_gen`` (``h_gen`` must be an EXACT interior
    node of ``h_grid`` -- true for all four registered arms' ``h_gen`` in
    ``H_GRID_41``: 0.68/0.73/0.78 are all present with neighbours on both
    sides). Generalizes ``decompose_impostor_leg.py``'s ``score_at_truth``
    (which hardcodes the 0.725/0.735 pair around ``H_TRUE=0.73``) to an
    arbitrary registered ``h_gen``, so C-SG's delta arms score at their OWN
    truth rather than at 0.73.

    Args:
        vals: ``(n_events, n_nodes)`` per-event likelihoods, columns aligned
            with ``h_grid``.
        h_gen: The generating truth to center the difference on.
        h_grid: The h-grid ``vals``' columns are aligned to.

    Returns:
        ``{"mean_score", "sem_score", "n_used", "n_skipped", "h_lo", "h_hi"}``.

    Raises:
        ValueError: If ``h_gen`` is not an interior node of ``h_grid``.
    """
    grid = np.array(sorted(h_grid), dtype=np.float64)
    idx_arr = np.nonzero(np.isclose(grid, h_gen))[0]
    if idx_arr.size == 0 or idx_arr[0] == 0 or idx_arr[0] == grid.size - 1:
        msg = f"h_gen={h_gen} must be an interior node of h_grid for a central difference"
        raise ValueError(msg)
    idx = int(idx_arr[0])
    i_lo, i_hi = idx - 1, idx + 1
    lo, hi = vals[:, i_lo], vals[:, i_hi]
    ok = (lo > 0.0) & (hi > 0.0) & np.isfinite(lo) & np.isfinite(hi)
    h_lo, h_hi = float(grid[i_lo]), float(grid[i_hi])
    if not ok.any():
        return {
            "mean_score": None,
            "sem_score": None,
            "n_used": 0,
            "n_skipped": int(vals.shape[0]),
            "h_lo": h_lo,
            "h_hi": h_hi,
        }
    s = (np.log(hi[ok]) - np.log(lo[ok])) / (h_hi - h_lo)
    return {
        "mean_score": float(s.mean()),
        "sem_score": float(s.std(ddof=1) / np.sqrt(s.size)) if s.size > 1 else None,
        "n_used": int(ok.sum()),
        "n_skipped": int((~ok).sum()),
        "h_lo": h_lo,
        "h_hi": h_hi,
    }


# ── GATE D: KS critical value at the actual n ────────────────────────────────


def ks_d_crit(alpha: float, n: int) -> float:
    r"""Two-sided one-sample Kolmogorov-Smirnov asymptotic critical value.

    ``D_crit(alpha, n) = sqrt(-ln(alpha/2) / (2n))`` (Massey 1951, the
    standard large-``n`` asymptotic form). Verified: ``ks_d_crit(0.05, 200) =
    0.09604...`` matches the prereg's quoted ``D_crit(5%) = 0.0960`` at
    ``n=200`` (GATE D, prereg section 6) to the quoted precision.

    Args:
        alpha: Two-sided significance level, in ``(0, 1)``.
        n: Sample size.

    Returns:
        The critical max-CDF-gap value.

    Raises:
        ValueError: If ``alpha`` is not in ``(0, 1)`` or ``n <= 0``.
    """
    if not (0.0 < alpha < 1.0):
        msg = f"alpha must be in (0, 1), got {alpha}"
        raise ValueError(msg)
    if n <= 0:
        msg = f"n must be positive, got {n}"
        raise ValueError(msg)
    return float(math.sqrt(-math.log(alpha / 2.0) / (2.0 * n)))


# ── Gates (prereg section 6, all zero-compute-ish pre-flight checks) ────────


def gate_h(
    arm: str,
    seed: int,
    n_events: int = 200,
    completeness: CsgCompletenessModel | None = None,
    detection_probability: CsgDetectionProbabilityModel | None = None,
    skip_s_bar_phi_diagnostic: bool = False,
) -> dict[str, Any]:
    r"""GATE H (h_gen threading), per the module docstring's "Registered deviations" item 1.

    Draws one realization to record where ``h_gen`` actually threads, then
    reports ``S_bar_phi``'s ``z_max(h_gen)`` (via
    :func:`~darksiren_emri.validation.correspondence_1d.build_bsel_selection_objects`,
    called PURELY for this diagnostic -- the C-SG generator itself never
    consumes its return value), the fraction of drawn events beyond the
    injection pool's calibrated depth, and the silenced-STOP disclosure
    (``allow_low_pdet_coverage=True``).

    Args:
        arm: One of :data:`CSG_H_GEN`'s keys.
        seed: Realization seed.
        n_events: Events to draw for the diagnostic (need not be the full
            registered dose; kept small for a fast gate).
        completeness: Pre-built selection object (default: build via
            :func:`build_csg_selection_objects`); injectable for pool-free
            plumbing tests.
        detection_probability: Pre-built selection object (default: build via
            :func:`build_csg_selection_objects`); injectable for pool-free
            plumbing tests.
        skip_s_bar_phi_diagnostic: If ``True``, skip the
            ``build_bsel_selection_objects`` diagnostic call (which always
            touches the REAL pinned injection pool, not the injected
            ``detection_probability`` -- pool-free tests must skip it).

    Returns:
        The GATE H report dict.
    """
    h_gen = _h_gen_for_arm(arm)
    if completeness is None or detection_probability is None:
        completeness, detection_probability = build_csg_selection_objects(h_gen=h_gen)
    donor_rows = pd.read_csv(CRB_CSV_PATH)
    _rows, diag = draw_csg_realization(
        seed, arm, n_events, completeness, detection_probability, donor_rows
    )

    dl_max = detection_probability.get_dl_max(h_gen)
    z_true = diag["z_true"]
    frac_beyond = float(np.mean(diag["d_L_true"] > dl_max)) if z_true.size else float("nan")

    s_bar_phi_z_max: float | None = None
    if not skip_s_bar_phi_diagnostic:
        # Diagnostic-only: report S_bar_phi's z_max(h_gen) (asymmetric domain
        # shrink/grow at 0.68/0.78, prereg section 6) -- NOT consumed by the
        # generator (see the module docstring). Always touches the REAL
        # pinned injection pool via build_bsel_selection_objects.
        _bsel_completeness, phi_survival_table = c1d.build_bsel_selection_objects(h_true=h_gen)
        z_grid_h, _s_phi = phi_survival_table[h_gen]
        s_bar_phi_z_max = float(z_grid_h[-1])

    return {
        "arm": arm,
        "seed": seed,
        "h_gen": h_gen,
        "threading": {
            "joint_z_omega_draw_h": diag["h_gen_threaded_to_joint_draw"],
            "dist_vectorized_h": diag["h_gen_threaded_to_dist_vectorized"],
            "build_csg_selection_objects_h_gen": h_gen,
        },
        "s_bar_phi_z_max": s_bar_phi_z_max,
        "injection_pool_dl_max_gpc": dl_max,
        "frac_drawn_beyond_calibrated_depth": frac_beyond,
        "allow_low_pdet_coverage_silences_stop": True,
        "design_deviation_gate_h_anchors": (
            "prereg GATE H cites draw_selected_population_redshifts(:1213) and "
            "build_bsel_selection_objects(h_true=h_gen)(:894) as h_gen threading sites. "
            "C-SG v3's generator does not call draw_selected_population_redshifts -- "
            "prereg section 0 item 2's fix removes S_bar_phi from the (z,Omega) proposal "
            "entirely (recon-1 independently confirmed draw_selected_population_redshifts/"
            "selected_population_z_weights cannot be reused unmodified for C-SG). h_gen "
            "instead threads through dark_siren_injection._draw_dark_hosts_pixelated (the "
            "joint proposal), physical_relations.dist_vectorized (true d_L), this module's "
            "build_csg_selection_objects (the h_gen-keyed analog of "
            "build_bsel_selection_objects), and compute_seed_statistics/"
            "seed_statistics_from_matrix at scoring time. build_bsel_selection_objects"
            "(h_true=h_gen) IS still called above, but only to report S_bar_phi's "
            "z_max(h_gen) as a diagnostic."
        ),
    }


def _attrition_report(
    rows: pd.DataFrame, fisher_cond_threshold: float = FISHER_COND_THRESHOLD
) -> dict[str, Any]:
    """Per-cut attrition on a drawn (pre-``evaluate()``) realization -- GATE Q's core."""
    n = len(rows)
    snr = rows["SNR"].to_numpy(dtype=np.float64)
    d_hat = rows["luminosity_distance"].to_numpy(dtype=np.float64)
    sigma_dl = np.sqrt(
        rows["delta_luminosity_distance_delta_luminosity_distance"].to_numpy(dtype=np.float64)
    )
    n_snr = int((snr < SNR_THRESHOLD).sum())
    ratio = sigma_dl / d_hat
    n_sigma = int((ratio >= FRACTIONAL_LUMINOSITY_DISTANCE_ERROR_THRESHOLD).sum())
    n_nonpd = _fisher_pd_exclusion_count(rows, fisher_cond_threshold)
    return {
        "n_total": n,
        "snr_below_threshold": {"n": n_snr, "fraction": (n_snr / n) if n else float("nan")},
        "sigma_over_dhat_ge_010": {"n": n_sigma, "fraction": (n_sigma / n) if n else float("nan")},
        "fisher_nonpd": {"n": n_nonpd, "fraction": (n_nonpd / n) if n else float("nan")},
    }


def _fisher_pd_exclusion_count(
    rows: pd.DataFrame, fisher_cond_threshold: float = FISHER_COND_THRESHOLD
) -> int:
    r"""Replicate ``bayesian_statistics.py:3960-4060``'s per-event 3D/4D Fisher PD exclusion.

    Reuses production's own
    :func:`~darksiren_emri.bayesian_inference.bayesian_statistics._check_covariance_quality`
    (the condition-number gate) plus its ``slogdet``-sign gate, on the SAME
    ``d_L``/``M``-normalized ``cov_3d``/``cov_4d`` construction ``evaluate()``
    builds internally -- D-A wholesale reuse of the gate function, not a
    re-derivation of the exclusion RULE (only the per-row matrix assembly,
    which is a fixed column-to-matrix mapping, is repeated here since
    ``evaluate()`` does not expose this as a standalone callable).

    Args:
        rows: A drawn (pre-``evaluate()``) realization, 128-column CRB
            schema.
        fisher_cond_threshold: See ``bayesian_statistics.py:3251``'s default.

    Returns:
        Count of events excluded by either the 3D or 4D check.
    """
    d_l = rows["luminosity_distance"].to_numpy(dtype=np.float64)
    m = rows["M"].to_numpy(dtype=np.float64)
    phi_var = rows["delta_phiS_delta_phiS"].to_numpy(dtype=np.float64)
    theta_var = rows["delta_qS_delta_qS"].to_numpy(dtype=np.float64)
    theta_phi_cov = rows["delta_phiS_delta_qS"].to_numpy(dtype=np.float64)
    dl_phi_cov = rows["delta_phiS_delta_luminosity_distance"].to_numpy(dtype=np.float64)
    dl_theta_cov = rows["delta_qS_delta_luminosity_distance"].to_numpy(dtype=np.float64)
    dl_var = rows["delta_luminosity_distance_delta_luminosity_distance"].to_numpy(dtype=np.float64)
    m_phi_cov = rows["delta_phiS_delta_M"].to_numpy(dtype=np.float64)
    m_theta_cov = rows["delta_qS_delta_M"].to_numpy(dtype=np.float64)
    dl_m_cov = rows["delta_luminosity_distance_delta_M"].to_numpy(dtype=np.float64)
    m_var = rows["delta_M_delta_M"].to_numpy(dtype=np.float64)

    n_excluded = 0
    for i in range(len(rows)):
        cov_3d = np.array(
            [
                [phi_var[i], theta_phi_cov[i], dl_phi_cov[i] / d_l[i]],
                [theta_phi_cov[i], theta_var[i], dl_theta_cov[i] / d_l[i]],
                [dl_phi_cov[i] / d_l[i], dl_theta_cov[i] / d_l[i], dl_var[i] / d_l[i] ** 2],
            ]
        )
        cov_4d = np.array(
            [
                [phi_var[i], theta_phi_cov[i], dl_phi_cov[i] / d_l[i], m_phi_cov[i] / m[i]],
                [theta_phi_cov[i], theta_var[i], dl_theta_cov[i] / d_l[i], m_theta_cov[i] / m[i]],
                [
                    dl_phi_cov[i] / d_l[i],
                    dl_theta_cov[i] / d_l[i],
                    dl_var[i] / d_l[i] ** 2,
                    dl_m_cov[i] / d_l[i] / m[i],
                ],
                [
                    m_phi_cov[i] / m[i],
                    m_theta_cov[i] / m[i],
                    dl_m_cov[i] / d_l[i] / m[i],
                    m_var[i] / m[i] ** 2,
                ],
            ]
        )
        _cond_3d, exclude_3d = _check_covariance_quality(cov_3d, fisher_cond_threshold)
        _cond_4d, exclude_4d = _check_covariance_quality(cov_4d, fisher_cond_threshold)
        if exclude_3d or exclude_4d:
            n_excluded += 1
            continue
        sign_3d, _logdet_3d = np.linalg.slogdet(cov_3d)
        sign_4d, _logdet_4d = np.linalg.slogdet(cov_4d)
        if sign_3d <= 0 or sign_4d <= 0:
            n_excluded += 1
    return n_excluded


def gate_q(
    arm: str,
    seed: int,
    n_events: int = 200,
    fisher_cond_threshold: float = FISHER_COND_THRESHOLD,
    completeness: CsgCompletenessModel | None = None,
    detection_probability: CsgDetectionProbabilityModel | None = None,
) -> dict[str, Any]:
    r"""GATE Q (attrition), before AND after the cross-covariance rescale.

    Draws the SAME realization twice (identical rng consumption -- the
    rescale is a deterministic post-hoc column transform, see
    :func:`draw_csg_realization`'s ``rescale_cross_covariance`` docs), once
    with the rescale applied and once without, and reports per-cut attrition
    (``SNR < 20``, ``sigma_dL/d_hat >= 0.10``, Fisher non-PD) for both. The
    single ``rescale_cross_covariance`` flag now toggles BOTH the ``d_L``-
    linked (:data:`DL_CROSS_COV_COLUMNS`) and ``M``-linked
    (:data:`M_CROSS_COV_COLUMNS`) blocks together (FIX ROUND 2026-08-21,
    finding #1), so this before/after comparison already reflects the M-block
    fix -- no separate gate needed.

    Args:
        arm: One of :data:`CSG_H_GEN`'s keys.
        seed: Realization seed.
        n_events: Events to draw.
        fisher_cond_threshold: See ``bayesian_statistics.py:3251``'s default.
        completeness: Pre-built selection object (default: build via
            :func:`build_csg_selection_objects`); injectable for pool-free
            plumbing tests.
        detection_probability: Pre-built selection object (default: build via
            :func:`build_csg_selection_objects`); injectable for pool-free
            plumbing tests.

    Returns:
        ``{"before_rescale", "after_rescale", "arm_redesign_required"}`` --
        the mandate: any arm above :data:`CSG_GATE_Q_NONPD_BAND` (1%) Fisher
        non-PD attrition AFTER the rescale is redesigned, not run.
    """
    h_gen = _h_gen_for_arm(arm)
    if completeness is None or detection_probability is None:
        completeness, detection_probability = build_csg_selection_objects(h_gen=h_gen)
    donor_rows = pd.read_csv(CRB_CSV_PATH)
    rows_after, _diag_after = draw_csg_realization(
        seed,
        arm,
        n_events,
        completeness,
        detection_probability,
        donor_rows,
        rescale_cross_covariance=True,
    )
    rows_before, _diag_before = draw_csg_realization(
        seed,
        arm,
        n_events,
        completeness,
        detection_probability,
        donor_rows,
        rescale_cross_covariance=False,
    )
    report_before = _attrition_report(rows_before, fisher_cond_threshold)
    report_after = _attrition_report(rows_after, fisher_cond_threshold)
    redesign = report_after["fisher_nonpd"]["fraction"] > CSG_GATE_Q_NONPD_BAND
    return {
        "arm": arm,
        "seed": seed,
        "h_gen": h_gen,
        "before_rescale": report_before,
        "after_rescale": report_after,
        "nonpd_band": CSG_GATE_Q_NONPD_BAND,
        "arm_redesign_required": bool(redesign),
    }


def gate_d(
    arm: str,
    seed: int,
    n_events: int = 200,
    n_model_grid: int = 2001,
    completeness: CsgCompletenessModel | None = None,
    detection_probability: CsgDetectionProbabilityModel | None = None,
    skip_model_density: bool = False,
) -> dict[str, Any]:
    r"""GATE D (premise): surviving-event z-distribution vs. the model density.

    Applies production's OWN quality filter (SNR >= :data:`SNR_THRESHOLD`,
    then ``distance_relative_error < FRACTIONAL_LUMINOSITY_DISTANCE_ERROR_THRESHOLD``,
    the exact ``use_detection`` predicate) to the drawn realization, then
    compares the SURVIVING events' z-distribution against
    :func:`~darksiren_emri.validation.correspondence_1d.selected_population_z_weights`
    -- see the module docstring's "Registered deviations" item 4 for why that
    B-SEL function is the mathematically correct model target for C-SG's
    ACCEPTED z-marginal (not merely a convenient reuse). Band is
    :func:`ks_d_crit` at the ACTUAL surviving ``n`` (prereg section 6: "never
    the retired 0.05").

    Args:
        arm: One of :data:`CSG_H_GEN`'s keys.
        seed: Realization seed.
        n_events: Events to draw.
        n_model_grid: Resolution of the model-density z-grid.
        completeness: Pre-built selection object (default: build via
            :func:`build_csg_selection_objects`); injectable for pool-free
            plumbing tests.
        detection_probability: Pre-built selection object (default: build via
            :func:`build_csg_selection_objects`); injectable for pool-free
            plumbing tests.
        skip_model_density: If ``True``, skip the
            ``build_bsel_selection_objects`` model-density comparison (which
            always touches the REAL pinned injection pool) -- pool-free tests
            get ``verdict=None`` and ``nan`` gap fields but still exercise the
            quality-filter/survival-count plumbing.

    Returns:
        The GATE D report dict, including ``verdict`` (``"MIRROR-MATCHED"``/
        ``"MIRROR-MISMATCHED"``/ ``None`` if skipped).
    """
    h_gen = _h_gen_for_arm(arm)
    if completeness is None or detection_probability is None:
        completeness, detection_probability = build_csg_selection_objects(h_gen=h_gen)
    donor_rows = pd.read_csv(CRB_CSV_PATH)
    rows, diag = draw_csg_realization(
        seed, arm, n_events, completeness, detection_probability, donor_rows
    )

    snr = rows["SNR"].to_numpy(dtype=np.float64)
    d_hat = rows["luminosity_distance"].to_numpy(dtype=np.float64)
    sigma_dl = np.sqrt(
        rows["delta_luminosity_distance_delta_luminosity_distance"].to_numpy(dtype=np.float64)
    )
    passes_snr = snr >= SNR_THRESHOLD
    distance_relative_error = sigma_dl / d_hat
    passes_quality = distance_relative_error < FRACTIONAL_LUMINOSITY_DISTANCE_ERROR_THRESHOLD
    surviving_mask = passes_snr & passes_quality

    z = diag["z_true"]
    surviving_z = z[surviving_mask]
    n_surviving = int(surviving_mask.sum())

    verdict: Literal["MIRROR-MATCHED", "MIRROR-MISMATCHED"] | None = None
    gap_surviving: float = float("nan")
    gap_drawn: float = float("nan")
    band: float = float("nan")
    if not skip_model_density:
        _bsel_completeness, phi_survival_table = c1d.build_bsel_selection_objects(h_true=h_gen)
        z_grid = np.linspace(POPULATION_Z_MIN, POPULATION_Z_MAX, n_model_grid, dtype=np.float64)
        model_density = c1d.selected_population_z_weights(
            z_grid, completeness, phi_survival_table, h=h_gen
        )
        band = ks_d_crit(CSG_D1_ALPHA, n_surviving) if n_surviving > 0 else float("nan")
        gap_surviving = (
            _max_cdf_gap(surviving_z, z_grid, model_density) if n_surviving > 0 else float("nan")
        )
        gap_drawn = _max_cdf_gap(z, z_grid, model_density)
        verdict = (
            "MIRROR-MATCHED" if (n_surviving > 0 and gap_surviving <= band) else "MIRROR-MISMATCHED"
        )
    return {
        "arm": arm,
        "seed": seed,
        "h_gen": h_gen,
        "n_drawn": n_events,
        "n_surviving": n_surviving,
        "survival_fraction": (n_surviving / n_events) if n_events else float("nan"),
        "max_cdf_gap_surviving_vs_model": gap_surviving,
        "max_cdf_gap_drawn_vs_model": gap_drawn,
        "d_crit_alpha": CSG_D1_ALPHA,
        "d_crit_at_n_surviving": band,
        "verdict": verdict,
    }


def gate_v(
    vals_matched: npt.NDArray[np.float64],
    seed_stats_matched: SeedStats,
    h_grid: tuple[float, ...] = H_GRID_41,
    min_span_nats: float = CSG_GATE_V_MIN_SPAN_NATS,
    sigma_prior_fraction: float = CSG_GATE_V_SIGMA_PRIOR_FRACTION,
) -> dict[str, Any]:
    r"""GATE V (anti-vacuity), applied to the MATCHED channel per v3 item 5.

    GATE V AMENDMENT 1 (2026-08-21, prereg appendix "PILOT GATE V AMENDMENT"):
    the v2 thresholds (span >= 5 nats, sigma_h <= 0.5*sigma_prior) were written
    for the FULL-channel posterior and, ported unchanged to the matched
    channel, false-fail 5/12 banked B-SEL matched posteriors (42%) -- the
    known-informative reference data that carries the O3 measurement -- and
    fired the registered STOP on 3/4 pilot seeds. Amended thresholds target
    the ACTUAL vacuity signature (prereg section 0 item 4: a FLAT log-posterior,
    span identically 0, sigma_h == sigma_prior, mean at the weight-convention
    flat mean):

    - ``span >= 1`` nat: a flat posterior has span 0 exactly; every genuine
      matched posterior in the reference set has span >= 2.01. Reference
      false-fail 0/12 (B-SEL) + 0/4 (pilot); the B-F1 flat mode (span 0.0)
      fails it decisively -- the gate CAN fail (A15).
    - ``sigma_h <= 0.9 * sigma_prior``: the flat null has ratio 1.0 exactly;
      reference max ratio 0.812. Reference false-fail 0/16.
    - the flat-mean coincidence (|mean_h - 0.73| < 1e-6 AND span < 1) is
      REPORTED as ``flat_mean_coincidence`` -- a genuinely centered posterior
      near 0.73 is the self-consistent EXPECTATION, so mean alone never fails.

    The superseded v2 numbers are still computed and returned
    (``v2_span_pass``/``v2_sigma_pass``) so the pilot's fired STOP remains
    reproducible from any banked JSON.

    Args:
        vals_matched: The matched-channel ``(n_events, n_nodes)`` matrix
            (:func:`csg_channel_matrices`'s ``"matched"`` key).
        seed_stats_matched: The matched channel's :class:`SeedStats` (from
            :func:`csg_channel_scores`, reconstructed via
            ``SeedStats(**channel_scores["matched"])`` if needed).
        h_grid: The score grid.
        min_span_nats: The amended vacuity-span floor (1 nat).
        sigma_prior_fraction: The amended sigma_h ceiling fraction (0.9).

    Returns:
        The GATE V report dict, including ``"pass"``.
    """
    grid = np.array(sorted(h_grid), dtype=np.float64)
    sum_log_l = c1d.combine_log_likelihood(vals_matched, "physics_floor")
    finite = sum_log_l[np.isfinite(sum_log_l)]
    span = float(finite.max() - finite.min()) if finite.size else float("nan")
    sigma_prior = float((grid.max() - grid.min()) / math.sqrt(12.0))
    span_pass = span >= min_span_nats
    sigma_pass = seed_stats_matched.sigma_h <= sigma_prior_fraction * sigma_prior
    return {
        "span_nats": span,
        "min_span_nats": min_span_nats,
        "span_pass": bool(span_pass),
        "sigma_h": seed_stats_matched.sigma_h,
        "sigma_prior": sigma_prior,
        "sigma_prior_fraction": sigma_prior_fraction,
        "sigma_pass": bool(sigma_pass),
        "pass": bool(span_pass and sigma_pass),
        "flat_mean_coincidence": bool(
            span < 1.0 and abs(seed_stats_matched.mean_h - 0.73) < 1.0e-6
        ),
        "v2_span_pass": bool(span >= 5.0),
        "v2_sigma_pass": bool(seed_stats_matched.sigma_h <= 0.5 * sigma_prior),
        "amendment": (
            "GATE V AMENDMENT 1 (2026-08-21): thresholds re-derived for the matched "
            "channel against the 12 banked B-SEL matched posteriors (v2 numbers "
            "false-failed 5/12 known-informative reference seeds); flat-null "
            "signature is span=0 / ratio=1.0; reference false-fail 0/16."
        ),
        "sigma_prior_convention": (
            "std of Uniform(min(h_grid), max(h_grid)) = (b-a)/sqrt(12) -- REGISTERED "
            "NEW convention, not specified numerically by the prereg; flagged for "
            "author review."
        ),
    }


# ── Execution: one (arm, seed) -> banked JSON + diagnostics CSV ─────────────


def run_csg_arm_seed(
    work_root: Path,
    arm: str,
    seed: int,
    out_dir: Path,
    n_events: int = 200,
    config: CSGConfig | None = None,
) -> Path:
    r"""Run one C-SG ``(arm, seed)`` task: draw, evaluate, score, bank.

    Wired exactly like ``correspondence_1d.run_arm_seed``'s ``bsel`` branch:
    builds the selection objects, draws the realization, calls
    :func:`~darksiren_emri.validation.correspondence_1d.run_mirror_seed_inprocess`
    UNMODIFIED (production's real ``evaluate()``, over ``H_GRID_FULL``), then
    scores the matched/pure/full channels on ``H_GRID_41`` ONLY (via
    :func:`csg_channel_scores` -> ``seed_statistics_from_matrix``, GATE V's
    "scoring on the banked 46-node grid is FORBIDDEN" mandate). **Idempotent**
    (walltime-kill safety): if ``<out_dir>/<arm>_seed<seed>.json`` already
    exists, returns immediately.

    Args:
        work_root: Per-task scratch directory.
        arm: One of :data:`CSG_H_GEN`'s keys.
        seed: Realization seed (expected to be a member of
            ``CSG_SEEDS[arm]``, not enforced here).
        out_dir: Fleet output directory.
        n_events: Events per realization (D-C parity default: 200).
        config: Optional :class:`CSGConfig` override (default: derived from
            ``n_events``).

    Returns:
        The (written-or-pre-existing) JSON path.

    Raises:
        KeyError: If ``arm`` is not registered.
    """
    h_gen = _h_gen_for_arm(arm)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{arm}_seed{seed}.json"
    if out_path.is_file():
        _LOGGER.info(
            "csg arm=%s seed=%d: output already exists, skipping (idempotent) -- %s",
            arm,
            seed,
            out_path,
        )
        return out_path

    cfg = config or CSGConfig(n_events=n_events)
    catalogue_pin_ok = c1d.check_reduced_catalogue_pin()
    crb_pin_ok = c1d.check_crb_pin()

    completeness, detection_probability = build_csg_selection_objects(h_gen=h_gen)
    donor_rows = pd.read_csv(CRB_CSV_PATH)
    rows, draw_diag = draw_csg_realization(
        seed,
        arm,
        cfg.n_events,
        completeness,
        detection_probability,
        donor_rows,
        oversample_factor=cfg.oversample_factor,
        batch_floor=cfg.batch_floor,
        max_batches=cfg.max_batches,
    )

    # Real GLADE candidate structure for evaluate()'s ball-tree (impostor leg,
    # prereg section 5 -- NOT removed for C-SG, same as B-SEL).
    handler = c1d._load_galaxy_catalog_handler(REDUCED_CATALOGUE_PATH)

    diag_csv, elapsed = c1d.run_mirror_seed_inprocess(
        work_root / f"seed{seed}",
        rows,
        seed,
        galaxy_catalog=handler,
        h_values=H_GRID_FULL,
    )

    channel_scores = csg_channel_scores(diag_csv, seed, h_grid=H_GRID_41, h_true=h_gen)
    mats = csg_channel_matrices(diag_csv, H_GRID_41)
    matched_stats = SeedStats(**channel_scores["matched"])
    v = gate_v(mats["matched"], matched_stats, h_grid=H_GRID_41)
    scores_at_h_gen = {
        channel: score_at_h_gen(mats[channel], h_gen, H_GRID_41)
        for channel in ("full", "matched", "pure")
    }
    h_grid_full, log_posterior_full = c1d.compute_full_log_posterior_vector(
        diag_csv, h_grid=H_GRID_FULL
    )

    record: dict[str, Any] = {
        "arm": arm,
        "seed": seed,
        "h_gen": h_gen,
        "sigma_mode": CSG_SIGMA_MODE[arm],
        "n_events_requested": cfg.n_events,
        "n_events_drawn": draw_diag["n_events"],
        "n_candidates_drawn": draw_diag["n_candidates_drawn"],
        "n_batches": draw_diag["n_batches"],
        "accept_rate": draw_diag["accept_rate"],
        "channel_scores": channel_scores,
        "score_at_h_gen": scores_at_h_gen,
        "gate_v": v,
        "h_grid": h_grid_full,
        "log_posterior_full_channel": log_posterior_full,
        "elapsed_s": elapsed,
        "git_commit": c1d._git_commit(),
        "catalogue_pin_ok": catalogue_pin_ok,
        "crb_pin_ok": crb_pin_ok,
        "diagnostics_csv": str(diag_csv),
    }
    out_path.write_text(json.dumps(record, indent=2))
    _LOGGER.info(
        "csg arm=%s seed=%d: wrote %s (elapsed=%.1fs, matched_mean_h=%s)",
        arm,
        seed,
        out_path,
        elapsed,
        channel_scores["matched"]["mean_h"],
    )
    return out_path


def _cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=CSG_ARMS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--work-root", required=True, help="Scratch directory for this task.")
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Fleet output directory; defaults to <work-root>/csg_arms.",
    )
    parser.add_argument("--n-events", type=int, default=200)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    work_root = Path(args.work_root)
    out_dir = Path(args.out_dir) if args.out_dir else work_root / "csg_arms"
    out_path = run_csg_arm_seed(work_root, args.arm, args.seed, out_dir, n_events=args.n_events)
    print(json.dumps({"out_path": str(out_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
