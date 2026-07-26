"""Simulation-based detection probability from injection campaign data.

Replaces :class:`DetectionProbability` (KDE-based) with a detection-horizon
survival-function approach that loads raw injection CSVs and applies an SNR
threshold at evaluation time.

Detection is *deterministic* optimal SNR (SNR = sqrt(<h|h>), no detector
noise; the threshold is passed as ``snr_threshold``).  Because the GW strain
amplitude scales as 1/d_L, each injection has an **h-invariant detection
horizon**

    d_hor_k = SNR_k * d_L_k / snr_threshold                            [Gpc]

(the d_L at which event k would sit exactly at threshold), and the detection
probability is *exactly* the survival function of the horizon distribution:

    p_det(d_L) = P(d_hor >= d_L) = fraction of injections detectable at d_L.

The horizon set is independent of the trial Hubble parameter h (the 1/d_L
amplitude scaling and the d_L rescaling cancel), so the survival grid is built
once and reused for every h.

All injection data is pooled regardless of the Hubble parameter value used
during the injection campaign.  The legacy SNR-rescaling helper
:meth:`_rescale_snr` is retained (it is exact physics and is unit-tested
directly) but the survival estimator does not require it.

References:
    * Finn & Chernoff (1993), arXiv:gr-qc/9301003 — detection horizon / SNR
      threshold for inspirals.
    * Finn (1996), arXiv:gr-qc/9601048 — p_det = P(Theta > Theta_thr) as a
      survival function of the orientation/distance factor.
    * Gray et al. (2020), arXiv:1908.06050, Section III.B-C — selection
      function structure for the numerator/denominator.
    * Mandel, Farr & Gair (2019), arXiv:1809.02063 — fixed-injection selection
      function evaluated at hypothesis observer-frame parameters.
    * SNR ~ 1/d_L: Hogg (1999), arXiv:astro-ph/9905116, Eq. (16).
"""

# ASSERT_CONVENTION: natural_units=SI, distance=Gpc, mass=solar_masses,
#   h=dimensionless_H0_over_100, SNR=dimensionless

import glob
import logging
import re
import warnings
from collections import OrderedDict
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

from master_thesis_code.physical_relations import dist_vectorized

logger = logging.getLogger(__name__)

# Default number of bins for the P_det grids
_DEFAULT_DL_BINS: int = 60
_DEFAULT_M_BINS: int = 40

# ── Change 1: ecliptic-latitude sky bands for the response-anisotropy axis ──
# The LISA orbit-averaged response depends on the source ecliptic latitude
# beta = pi/2 - qS (best near the ecliptic plane, weakest at the poles), and is
# azimuthally symmetric after multi-year averaging (Cutler 1998,
# arXiv:gr-qc/9703068; arXiv:1201.3684).  Route A re-bins the EXISTING isotropic
# injection horizons by |sin beta| into equal-solid-angle (equal-|sin beta|)
# bands and builds one detection-horizon survival per band -- no separation
# ansatz, no new simulation (PHYSICS-CHANGE-PROTOCOL Change 1).
_DEFAULT_N_SKY_BANDS: int = 6
# Under-populated polar-band guard (analogous to the 2D grid n_total>=10 check):
# a band with fewer injections than this falls back to the pooled (isotropic)
# horizon so a noise-starved band never distorts the survival.
_MIN_BAND_INJECTIONS: int = 10

# Maximum number of cached grids (legacy LRU eviction parameter).  The
# detection horizon is h-invariant so a single grid now serves every h; the
# constant is retained for backward-compatible imports.
_MAX_CACHE_SIZE: int = 20

# Default prior support of the trial Hubble constant h (LamCDMScenario, see
# cosmological_model.py:304-305).  Retained for the deprecated h_prior_range
# parameter; no longer used to size the d_L support (the compact horizon-based
# support is computed directly from the injections, see
# ``_compute_dl_global_max``).
_DEFAULT_H_PRIOR_MIN: float = 0.60
_DEFAULT_H_PRIOR_MAX: float = 0.86

# 10% headroom above the maximum detection horizon, so the survival grid
# fully covers the support of the horizon distribution with empty-bin margin.
_DL_PADDING_FACTOR: float = 1.1

# Scott's rule exponent for the (single) observer-frame log-mass kernel
# dimension used by the 2D survival estimator.  The d_L axis carries NO
# kernel: the survival function is exact in d_L.
_SCOTT_EXPONENT_2D: float = -1.0 / 6.0  # N^(-1/(d+4)) with d=1 → N^(-1/6)

# ── FIX-2: z-resolved detection survival S(d_L | z) ──
# Scott's rule exponent for the 1-dimensional u = ln(1+z) kernel of the
# z-resolved survival estimator: sigma_u = N^(-1/(d+4)) std(u) with d = 1.
# NOTE: this is the textbook d=1 value; the pre-existing _SCOTT_EXPONENT_2D
# above is arithmetically the d=2 exponent (its comment says d=1) — both give
# the same D-slopes to 0.015 (DERIVATION_ZRESOLVED_SURVIVAL.md §3.3), the
# textbook value is adopted here and the discrepancy documented.
# Scott (1992), Multivariate Density Estimation, Ch. 6.
_SCOTT_EXPONENT_1D: float = -1.0 / 5.0
# Number of u = ln(1+z) kernel nodes for the (u x d_L) survival table
# (convergence verified by the packet's bandwidth/binning sweep).
_ZRES_U_NODES: int = 121
# Pilot-KDE histogram resolution for the Abramson (1982) sqrt-law adaptive
# bandwidth (variance-stabilising, no tunable threshold).
_ZRES_PILOT_BINS: int = 400
_ZRES_PILOT_DENSITY_FLOOR: float = 1e-12

# Estimator selector retained for API compatibility.  The detection-horizon
# survival function is exact in d_L, so this parameter no longer affects the
# d_L treatment; it is accepted only so existing call sites and the NW
# regression escape hatch continue to construct.
_Estimator = Literal["local_linear", "nadaraya_watson"]
_DEFAULT_ESTIMATOR: _Estimator = "local_linear"


class SimulationDetectionProbability:
    """Simulation-based detection probability from injection campaign data.

    Loads raw injection CSVs (z, M, phiS, qS, SNR, h_inj, luminosity_distance),
    pools ALL events regardless of h_inj, and builds the detection-horizon
    survival grids once (they are h-invariant).

    For a source with measured optimal SNR_raw at luminosity distance d_L_k,
    the **detection horizon** is the distance at which it would sit exactly at
    threshold:

        d_hor_k = SNR_raw_k * d_L_k / snr_threshold                    [Gpc]

    Detection is deterministic (SNR >= threshold ⇔ d_L <= d_hor), so the
    detection probability is the survival function of the horizon
    distribution:

        p_det(d_L) = P(d_hor >= d_L).

    The horizon set is independent of h, so the survival grids are built once
    and the same cached interpolators are returned for every queried h.

    Args:
        injection_data_dir: Directory containing injection CSV files matching
            ``injection_h_*_task_*.csv`` or ``injection_h_*.csv``.
        snr_threshold: SNR threshold for detection. Events with
            SNR >= snr_threshold are considered detected.
        h_grid: **Deprecated.** Previously used to specify h grid points for
            pre-computed grids.  Now ignored (a single h-invariant survival
            grid serves all h).  Passing this parameter emits a deprecation
            warning.
        dl_bins: Number of d_L grid centers for the survival grids.
        mass_bins: Number of observer-frame M_z grid centers (2D grid).
        h_prior_range: **Deprecated for the d_L support.** Accepted for API
            compatibility; the d_L support is now derived from the (compact,
            h-invariant) detection-horizon distribution.
        bandwidth_scale: Multiplier on Scott's-rule bandwidth for the
            observer-frame log-mass kernel of the 2D survival estimator.
        estimator: **Irrelevant to the d_L treatment** (the survival function
            is exact in d_L).  Accepted for API compatibility / the NW
            regression escape hatch.
        _force_unit_weights: Internal flag for testing. When True, passes
            explicit ``weights=np.ones(N)`` to ``_build_grid_2d`` to verify
            IS estimator backward compatibility.
        pdet_z_resolved: FIX-2 (default False = pooled, byte-identical to
            pre-FIX-2 behaviour).  When True, every 3D (without-BH-mass)
            survival query returns the z-CONDITIONAL survival
            ``S(d_L | z) = P(d_hor >= d_L | z)`` (Gaussian kernel in
            ``u = ln(1+z)``, Scott d=1 bandwidth, Abramson-adaptive; exact
            suffix-survival in d_L per node) and the 3D accessors REQUIRE the
            ``z`` keyword.  The 2D (M_z-conditioned) grid keeps its current
            form.  DERIVATION_ZRESOLVED_SURVIVAL.md.

    References:
        Finn & Chernoff (1993), arXiv:gr-qc/9301003.
        Finn (1996), arXiv:gr-qc/9601048.
        Gray et al. (2020), arXiv:1908.06050, Section III.B-C.
        Mandel, Farr & Gair (2019), arXiv:1809.02063.
    """

    def __init__(
        self,
        injection_data_dir: str,
        snr_threshold: float,
        h_grid: list[float] | None = None,
        *,
        dl_bins: int = _DEFAULT_DL_BINS,
        mass_bins: int = _DEFAULT_M_BINS,
        h_prior_range: tuple[float, float] = (_DEFAULT_H_PRIOR_MIN, _DEFAULT_H_PRIOR_MAX),
        bandwidth_scale: float = 1.0,
        estimator: _Estimator = _DEFAULT_ESTIMATOR,
        n_sky_bands: int = _DEFAULT_N_SKY_BANDS,
        _force_unit_weights: bool = False,
        expected_z_max: float | None = None,
        allow_shallow_pool: bool = False,
        pdet_z_resolved: bool = False,
    ) -> None:
        self._dl_bins = dl_bins
        self._mass_bins = mass_bins
        self._snr_threshold = snr_threshold
        self._force_unit_weights = _force_unit_weights
        if int(n_sky_bands) < 1:
            msg = f"n_sky_bands must be >= 1, got {n_sky_bands}"
            raise ValueError(msg)
        self._n_sky_bands: int = int(n_sky_bands)
        # Estimator no longer affects the d_L treatment (survival is exact);
        # retained for API compat and the NW regression escape hatch.
        if estimator not in ("local_linear", "nadaraya_watson"):
            msg = f"estimator must be 'local_linear' or 'nadaraya_watson', got {estimator!r}"
            raise ValueError(msg)
        self._estimator: _Estimator = estimator
        # bandwidth_scale still scales the observer-frame M_z kernel of the
        # 2D survival estimator (see ``_compute_bandwidths``).
        if bandwidth_scale <= 0.0:
            msg = f"bandwidth_scale must be positive, got {bandwidth_scale}"
            raise ValueError(msg)
        self._bandwidth_scale: float = float(bandwidth_scale)
        if h_prior_range[0] >= h_prior_range[1]:
            msg = f"h_prior_range must satisfy lower < upper, got {h_prior_range}"
            raise ValueError(msg)
        self._h_prior_min: float = float(h_prior_range[0])
        self._h_prior_max: float = float(h_prior_range[1])

        if h_grid is not None:
            warnings.warn(
                "The 'h_grid' parameter is deprecated and ignored. "
                "SimulationDetectionProbability now builds a single h-invariant "
                "detection-horizon survival grid from pooled injection data.",
                DeprecationWarning,
                stacklevel=2,
            )

        # Glob CSV files matching expected patterns
        patterns = [
            f"{injection_data_dir}/injection_h_*_task_*.csv",
            f"{injection_data_dir}/injection_h_*.csv",
        ]
        csv_files: list[str] = []
        for pattern in patterns:
            csv_files.extend(glob.glob(pattern))

        # Remove duplicates (a file may match both patterns)
        csv_files = sorted(set(csv_files))

        if not csv_files:
            msg = (
                f"No injection CSV files found in '{injection_data_dir}'. "
                "Expected files matching 'injection_h_*_task_*.csv' or 'injection_h_*.csv'."
            )
            raise FileNotFoundError(msg)

        # Extract h values from filenames for reference
        h_pattern = re.compile(r"injection_h_(\d+p\d+)")
        h_values_found: set[float] = set()

        # Load ALL CSVs and pool into a single DataFrame
        dfs: list[pd.DataFrame] = []
        for f in csv_files:
            match = h_pattern.search(f)
            if match:
                h_label = match.group(1)
                h_val = float(h_label.replace("p", "."))
                h_values_found.add(h_val)
            dfs.append(pd.read_csv(f))

        if not dfs:
            msg = (
                f"Could not parse any injection CSV files in '{injection_data_dir}'. "
                "Expected format: 'injection_h_0p70_task_001.csv'."
            )
            raise FileNotFoundError(msg)

        self._pooled_df: pd.DataFrame = pd.concat(dfs, ignore_index=True)
        self._h_values_found: list[float] = sorted(h_values_found)

        logger.info(
            "Pooled %d injection events from %d files (h values: %s).",
            len(self._pooled_df),
            len(csv_files),
            ", ".join(f"{h:.2f}" for h in self._h_values_found),
        )

        # Validate required columns.  "qS" (ecliptic colatitude) is now required:
        # it carries the response-anisotropy (ecliptic-latitude) axis of p_det
        # (Change 1).  It is already written by every injection campaign
        # (main.py:injection_campaign), so this is a no-op for the canonical CSVs.
        required_cols = {"z", "M", "SNR", "h_inj", "luminosity_distance", "qS"}
        missing = required_cols - set(self._pooled_df.columns)
        if missing:
            msg = f"Injection CSV missing required columns: {missing}"
            raise ValueError(msg)

        # Pre-extract arrays for efficient rescaling / horizon construction
        self._z_arr: npt.NDArray[np.float64] = self._pooled_df["z"].values.astype(np.float64)
        self._M_arr: npt.NDArray[np.float64] = self._pooled_df["M"].values.astype(np.float64)
        self._snr_raw: npt.NDArray[np.float64] = self._pooled_df["SNR"].values.astype(np.float64)
        self._h_inj_arr: npt.NDArray[np.float64] = self._pooled_df["h_inj"].values.astype(
            np.float64
        )
        self._dl_raw: npt.NDArray[np.float64] = self._pooled_df[
            "luminosity_distance"
        ].values.astype(np.float64)

        # ── Depth / provenance gates (issue #20 stale-pool hazard, 2026-07-03) ──
        # Deepening the host draw (HOST_DRAW_Z_MAX -> 1.5) outdates every
        # z_cut = 0.5-era pool: deep hosts reach d_L ~ 13 Gpc while a shallow
        # pool's survival grid tops out below ~1 Gpc, so p_det = 0 for
        # essentially all events — silently valid-looking garbage posteriors.
        # The regenerated campaign writes the SAME filenames, so a partial
        # rsync / leftover task file mixes eras undetectably by name alone.
        # Production constructors pass expected_z_max=HOST_DRAW_Z_MAX;
        # tests and synthetic pools leave it None (no depth gate).
        if "z_cut" in self._pooled_df.columns:
            n_missing = int(self._pooled_df["z_cut"].isna().sum())
            z_cuts = sorted(float(z) for z in self._pooled_df["z_cut"].dropna().unique())
            if n_missing > 0 or len(z_cuts) > 1:
                msg = (
                    f"Injection pool mixes provenance: z_cut values {z_cuts} plus "
                    f"{n_missing} rows lacking the column (legacy files). Leftover "
                    "task files from a retired pool or a partial rsync poison the "
                    f"survival grid — purge/archive '{injection_data_dir}' and use "
                    "one consistently-generated pool."
                )
                raise ValueError(msg)
        else:
            logger.warning(
                "Injection pool has no provenance columns (z_cut/code_rev) — "
                "pre-2026-07-03 writer. Depth is still gated below if "
                "expected_z_max is set."
            )
        if "code_rev" in self._pooled_df.columns and self._pooled_df["code_rev"].nunique() > 1:
            logger.warning(
                "Injection pool spans %d code revisions (%s) — legitimate for "
                "straggler resubmits after a non-physics fix, but verify none of "
                "them changed SNR semantics.",
                self._pooled_df["code_rev"].nunique(),
                ", ".join(str(c)[:8] for c in self._pooled_df["code_rev"].unique()),
            )
        if expected_z_max is not None:
            pool_z_max = float(np.max(self._z_arr)) if len(self._z_arr) else 0.0
            if pool_z_max < 0.9 * float(expected_z_max) and not allow_shallow_pool:
                msg = (
                    f"Injection pool is SHALLOW: max injected z = {pool_z_max:.3f} "
                    f"< 0.9 x expected_z_max = {0.9 * float(expected_z_max):.3f}. "
                    "The survival grid cannot cover the host-draw volume "
                    f"(HOST_DRAW_Z_MAX-era depth mismatch). Regenerate the pool at "
                    f"the campaign depth, or pass allow_shallow_pool=True (e.g. for "
                    "a deliberate re-evaluation of an archived shallow baseline)."
                )
                raise ValueError(msg)

        # h-invariant detection horizon for each injection.
        # p_det = survival function of the detection horizon, P(d_hor >= d_L),
        # with d_hor = SNR·d_L/threshold.
        # Finn & Chernoff (1993), arXiv:gr-qc/9301003; Finn (1996),
        # arXiv:gr-qc/9601048 (p_det = P(Theta > Theta_thr)).
        self._d_hor: npt.NDArray[np.float64] = self._snr_raw * self._dl_raw / self._snr_threshold
        # Observer-frame log10 mass (M_z) for the 2D survival kernel axis.  The
        # injection CSV "M" column already stores the DETECTOR-FRAME mass
        # M_z = M_source·(1+z) (lifted at injection time in
        # main.py:injection_campaign), consistent with the event CRBs and the
        # inference's det.M = M_z convention, so NO (1+z) re-lift is applied here.
        # Maggiore (2008) GW Vol. 1 §4.1.4.
        self._log_M_z: npt.NDArray[np.float64] = np.log10(self._M_arr)

        # Sort the horizon set ONCE for exact searchsorted-based survival.
        sort_idx = np.argsort(self._d_hor, kind="mergesort")
        self._d_hor_sorted: npt.NDArray[np.float64] = self._d_hor[sort_idx]
        # Uniform importance weights (1 per injection); suffix-cumsum from the
        # right gives the count of injections with d_hor >= threshold.
        self._n_inj: int = len(self._d_hor_sorted)

        # ── Change 1: ecliptic-latitude sky bands ──
        # Ecliptic colatitude qS = theta; latitude beta = pi/2 - qS, so the
        # equal-solid-angle variable is |sin beta| = |cos qS| (Cutler 1998,
        # arXiv:gr-qc/9703068 -- azimuthally symmetric orbit-averaged response).
        self._qS_arr: npt.NDArray[np.float64] = self._pooled_df["qS"].values.astype(np.float64)
        self._build_sky_bands()

        # ── FIX-2 (opt-in, --pdet_z_resolved): z-resolved detection survival ──
        # S(d_L | z) = P(d_hor >= d_L | z), Gaussian kernel in u = ln(1+z)
        # (Scott d=1 bandwidth, Abramson sqrt-law adaptive), exact
        # suffix-survival in d_L per kernel node.  Default OFF: the pooled
        # survival above is byte-identical to pre-FIX-2 behaviour.
        # results/lcat_h_dependence_20260725/DERIVATION_ZRESOLVED_SURVIVAL.md
        # Eq. (4)-(5); Finn & Chernoff (1993), arXiv:gr-qc/9301003; Finn
        # (1996), arXiv:gr-qc/9601048; Mandel, Farr & Gair (2019),
        # arXiv:1809.02063 (selection at hypothesis specifies z).
        self._z_resolved: bool = bool(pdet_z_resolved)
        self._zres_degenerate: bool = False
        self._zres_u_nodes: npt.NDArray[np.float64] | None = None
        self._zres_suffix_w: npt.NDArray[np.float64] | None = None
        self._zres_total_w: npt.NDArray[np.float64] | None = None
        self._zres_ess: npt.NDArray[np.float64] | None = None
        self._zres_band_suffix_w: list[npt.NDArray[np.float64]] | None = None
        self._zres_band_total_w: list[npt.NDArray[np.float64]] | None = None
        self._zres_band_ess: list[npt.NDArray[np.float64]] | None = None
        self._zres_band_fallback: npt.NDArray[np.bool_] | None = None
        if self._z_resolved:
            self._build_zres_survival()

        # Cache holder for the (single, h-invariant) survival interpolators.
        self._grid_cache: OrderedDict[
            float,
            tuple[RegularGridInterpolator, RegularGridInterpolator],
        ] = OrderedDict()
        # Single built-once grid shared across all h (horizon is h-invariant).
        self._shared_grid: tuple[RegularGridInterpolator, RegularGridInterpolator] | None = None

        # Quality flags cache
        self._quality_flags: dict[
            float, dict[str, npt.NDArray[np.float64] | npt.NDArray[np.bool_]]
        ] = {}

        # Compact, h-invariant d_L support derived from the horizon set.
        self._dl_global_max: float = self._compute_dl_global_max()

    def __getstate__(self) -> dict[str, Any]:
        """Exclude heavy data from pickle that workers don't need.

        Workers only call detection_probability_*_interpolated() which uses the
        pre-built survival interpolators (and, for the exact 1D survival, the
        stored sorted horizon array).  The raw injection arrays not needed for
        those lookups are dropped to keep the pickle small.
        """
        state = self.__dict__.copy()
        state["_pooled_df"] = None
        # Raw injection arrays not needed once the grid + sorted horizon are
        # built.  ``_d_hor_sorted`` IS retained because the exact 1D survival
        # accessor uses it directly.
        state["_z_arr"] = None
        state["_M_arr"] = None
        state["_snr_raw"] = None
        state["_h_inj_arr"] = None
        state["_dl_raw"] = None
        state["_d_hor"] = None
        state["_log_M_z"] = None
        # Raw per-injection latitude not needed post-build; the per-band SORTED
        # horizons (``_d_hor_sorted_by_band``) ARE retained -- the sky accessor
        # and ``survival_per_band`` use them directly in workers.
        state["_qS_arr"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)

    def _rescale_snr(
        self, h_target: float
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Rescale SNR values from injection h to target h.

        For each event at redshift z injected at h_inj with SNR_raw:
            d_L_inj = dist(z, h_inj)   [from injection campaign]
            d_L_target = dist(z, h_target)
            SNR_target = SNR_raw * d_L_inj / d_L_target

        The d_L_inj values are recomputed from (z, h_inj) rather than using
        the stored luminosity_distance column, ensuring consistency with the
        cosmological model in physical_relations.py.

        This helper is exact physics and is retained / unit-tested directly.
        The detection-horizon survival estimator does not require it (the
        horizon ``d_hor = SNR·d_L/threshold`` is h-invariant), but downstream
        code and tests still call it.

        Args:
            h_target: Target Hubble parameter value.

        Returns:
            Tuple of (d_L_target, SNR_rescaled) arrays, each shape (N,).

        References:
            SNR ~ 1/d_L: gravitational wave amplitude h(t) ~ 1/d_L.
            Gray et al. (2020), arXiv:1908.06050, Section III.B-C.
        """
        # Compute d_L at injection h for each event
        # Group by unique h_inj values for efficiency
        unique_h_inj = np.unique(self._h_inj_arr)
        d_L_inj = np.empty_like(self._z_arr)
        for h_inj in unique_h_inj:
            mask = self._h_inj_arr == h_inj
            d_L_inj[mask] = dist_vectorized(self._z_arr[mask], h=float(h_inj))

        # Compute d_L at target h for all events
        d_L_target = dist_vectorized(self._z_arr, h=h_target)

        # Rescale SNR: SNR(h_target) = SNR_raw * d_L(z, h_inj) / d_L(z, h_target)
        # Guard against d_L_target = 0 (z = 0 edge case)
        with np.errstate(divide="ignore", invalid="ignore"):
            snr_rescaled = np.where(
                d_L_target > 0,
                self._snr_raw * d_L_inj / d_L_target,
                0.0,
            )

        return (
            np.asarray(d_L_target, dtype=np.float64),
            np.asarray(snr_rescaled, dtype=np.float64),
        )

    def _compute_dl_global_max(self) -> float:
        """Compact d_L support for the survival grids.

        The detection-horizon distribution has finite support: no injection is
        detectable beyond ``max_k d_hor_k``, so ``p_det(d_L) = 0`` for
        ``d_L > max d_hor``.  The grid d_L axis therefore extends only to

            max_k (SNR_k * d_L_k / snr_threshold) * _DL_PADDING_FACTOR

        (≈0.86 Gpc on the canonical injections), NOT the old
        ``dist(z, h_min)`` (~13 Gpc).  This support is h-INVARIANT because the
        horizon ``d_hor = SNR·d_L/threshold`` is h-invariant.

        Returns:
            float: ``max_k d_hor_k * _DL_PADDING_FACTOR`` in Gpc.

        References:
            Finn & Chernoff (1993), arXiv:gr-qc/9301003 (detection horizon).
            Mandel, Farr & Gair (2019), arXiv:1809.02063 (selection function
            support).
        """
        return float(np.max(self._d_hor)) * _DL_PADDING_FACTOR

    def _compute_bandwidths(
        self,
        dl_vals: npt.NDArray[np.float64],
        log_M_vals: npt.NDArray[np.float64],
    ) -> tuple[float, float]:
        """Kernel bandwidths for the survival estimator's mass axis.

        Scott's rule (1D log-mass kernel): σ = bandwidth_scale · n^(-1/6) ·
        std(log10 M_z).  The d_L axis carries NO kernel (the survival function
        is exact there); the returned ``sigma_dl`` is kept for backward
        compatibility with callers/tests that unpack two values, computed with
        the same Scott's-rule scaling.

        Args:
            dl_vals: per-injection d_L (Gpc).  Used only for the back-compat
                ``sigma_dl`` return value.
            log_M_vals: per-injection log10(M_z) (observer-frame), dimensionless.

        Returns:
            (σ_dl in Gpc, σ_logM in dex).  Both ≥ a small numerical floor.

        References:
            Scott (1992), Multivariate Density Estimation, Ch. 6.
        """
        n = float(len(dl_vals))
        scale = self._bandwidth_scale * n**_SCOTT_EXPONENT_2D
        sigma_dl = max(scale * float(np.std(dl_vals, ddof=0)), 1e-12)
        sigma_log_M = max(scale * float(np.std(log_M_vals, ddof=0)), 1e-12)
        return sigma_dl, sigma_log_M

    def _survival_at(
        self,
        query: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Exact weighted survival p_det(d_L) = P(d_hor >= d_L) (uniform weights).

        Uses the stored sorted detection horizon: for a query ``d_L``,
        ``np.searchsorted(d_hor_sorted, d_L, side='left')`` gives the number of
        injections with ``d_hor < d_L``, so the count with ``d_hor >= d_L`` is
        ``N - that``, and the survival is that count divided by ``N``.

        Guarantees by construction: p(0)=1, p(d_L > max d_hor)=0, monotone
        non-increasing in d_L.

        Args:
            query: d_L query points [Gpc], any shape.

        Returns:
            Survival values in [0, 1], same shape as ``query``.
        """
        idx_below = np.searchsorted(self._d_hor_sorted, query, side="left")
        count_ge = self._n_inj - idx_below
        surv = count_ge.astype(np.float64) / float(self._n_inj)
        return np.asarray(np.clip(surv, 0.0, 1.0), dtype=np.float64)

    # ------------------------------------------------------------------
    # FIX-2: z-resolved detection survival S(d_L | z)
    # ------------------------------------------------------------------

    @property
    def z_resolved(self) -> bool:
        """True iff the z-resolved (FIX-2) survival estimator is active."""
        return self._z_resolved

    def _build_zres_survival(self) -> None:
        r"""Build the z-conditional survival tables ``S(d_L | z)`` (FIX-2).

        Estimator (DERIVATION_ZRESOLVED_SURVIVAL.md, Eq. (4)-(5)):

        .. math::

            S(d_L | z) = \frac{\sum_k K((u_k - u)/\sigma_k)\,
                                 \mathbb{1}[d_{hor,k} \ge d_L]}
                              {\sum_k K((u_k - u)/\sigma_k)},
            \qquad u = \ln(1+z),

        with Gaussian ``K``, global width ``sigma_u = bandwidth_scale ·
        N^{-1/5} · std(u)`` (Scott 1992, Ch. 6, d=1) and per-injection
        Abramson (1982) square-root-law adaptive factors ``sigma_k = sigma_u ·
        sqrt(g_hat / f_hat(u_k))`` from the pool's own pilot KDE.  Exact
        suffix-count survival in ``d_L`` per kernel node — the identical
        computational pattern as the M_z-kernel ``_build_grid_2d``.

        The kernel coordinate ``u = ln(1+z)`` is derived, not chosen: the
        detector-frame lifts ``M_z = M(1+z)``, ``f_obs = f_src/(1+z)`` are
        multiplicative in (1+z), so a z-shift is a TRANSLATION in u, and a
        single-bandwidth kernel is correct precisely in that coordinate
        (packet §3.2).  Everything here is h-free: ``d_hor_k`` and ``u_k``
        come from the injections; only the query ``d_L(z;h)`` moves with h,
        so the tables are built once per run.

        Starved-regime policy: the Abramson law widens the kernel where the
        pool is sparse (low z), degrading CONTINUOUSLY toward the pooled
        survival — no threshold constants.  Sky-band × z cells whose kernel
        ESS falls below the repo's existing reliability floor
        (``_MIN_BAND_INJECTIONS``) fall back to the z-only (band-marginal)
        conditional, NOT the fully pooled survival (packet §7).

        References:
            Finn & Chernoff (1993), arXiv:gr-qc/9301003; Finn (1996),
            arXiv:gr-qc/9601048 — horizon-survival framework.
            Mandel, Farr & Gair (2019), arXiv:1809.02063 — selection at
            hypothesis: the population at hypothesis specifies z, so the
            detection kernel is P(det | z, h), not the z-marginal.
            Scott (1992), Multivariate Density Estimation, Ch. 6.
            Abramson (1982), Ann. Statist. 10:1217.
            results/lcat_h_dependence_20260725/DERIVATION_ZRESOLVED_SURVIVAL.md.
        """
        n = self._n_inj
        u = np.log1p(self._z_arr)
        std_u = float(np.std(u, ddof=0))
        if n < 2 or std_u <= 0.0:
            # Degenerate u-distribution: stratification is unidentifiable;
            # the z-conditional survival IS the pooled survival (limiting
            # case (i) of the packet, exact).
            self._zres_degenerate = True
            logger.info(
                "z-resolved survival: degenerate u = ln(1+z) distribution "
                "(N=%d, std=%.3e) — falling back to the pooled survival.",
                n,
                std_u,
            )
            return

        # Scott's rule, d=1 (sigma_u = N^(-1/5) std(u)); bandwidth_scale is
        # the pre-existing sensitivity knob (default 1).
        # Scott (1992), Multivariate Density Estimation, Ch. 6.
        sigma_u = self._bandwidth_scale * float(n) ** _SCOTT_EXPONENT_1D * std_u
        u_max = float(np.max(u))

        # Pilot KDE for the Abramson sqrt-law: histogram density convolved
        # with a Gaussian of width sigma_u (packet zres_survival construction).
        edges = np.linspace(0.0, max(u_max, sigma_u), _ZRES_PILOT_BINS + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        du = float(edges[1] - edges[0])
        hist, _ = np.histogram(u, bins=edges, density=True)
        # Cap the tap count (np.convolve 'same' needs len(taps) <= len(hist)):
        # for very large bandwidths (sigma_u >> support) the pilot is ~flat
        # and lambda -> 1 regardless.
        kh = int(min(np.ceil(4.0 * sigma_u / du), (_ZRES_PILOT_BINS - 1) // 2))
        taps = np.exp(-0.5 * (np.arange(-kh, kh + 1) * du / sigma_u) ** 2)
        pilot = np.convolve(hist, taps / taps.sum(), mode="same")
        pilot = np.clip(pilot, _ZRES_PILOT_DENSITY_FLOOR, None)
        f_at = np.interp(u, centers, pilot)
        g_mean = float(np.exp(np.mean(np.log(f_at))))
        # Abramson (1982), Ann. Statist. 10:1217 — square-root law:
        # sigma_k = sigma_u * (g_hat / f_hat(u_k))^(1/2).
        lam = np.sqrt(g_mean / f_at)
        sigma_i = sigma_u * lam  # per-injection adaptive bandwidth

        u_nodes = np.linspace(0.0, u_max, _ZRES_U_NODES)
        self._zres_u_nodes = u_nodes

        # Order injections by d_hor ascending (same values as _d_hor_sorted)
        # so a suffix-cumsum of kernel weights gives the weighted survival at
        # any d_L via np.searchsorted — the _build_grid_2d pattern with
        # (log10 M_z, sigma_lm) -> (ln(1+z), sigma_u).
        sort_idx = np.argsort(self._d_hor, kind="mergesort")
        u_sorted = u[sort_idx]
        sig_sorted = sigma_i[sort_idx]

        def _suffix_tables(
            u_sub: npt.NDArray[np.float64],
            sig_sub: npt.NDArray[np.float64],
        ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
            """(suffix_w, total_w, ess) for one d_hor-sorted injection subset."""
            diff = (u_sub[:, None] - u_nodes[None, :]) / sig_sub[:, None]
            w = np.exp(-0.5 * diff * diff)  # (n_sub, n_nodes)
            total = w.sum(axis=0)
            sq = np.einsum("ij,ij->j", w, w)
            with np.errstate(divide="ignore", invalid="ignore"):
                ess = np.where(sq > 0.0, total * total / sq, 0.0)
            # Numerically-empty node guard (cannot happen with the Abramson
            # widening on a non-degenerate pool, but keep S well-defined):
            # fall back to uniform weights = pooled survival at that node.
            empty = total <= 0.0
            if np.any(empty):
                w[:, empty] = 1.0
                total = w.sum(axis=0)
                ess[empty] = float(len(u_sub))
            suffix = np.cumsum(w[::-1, :], axis=0)[::-1, :]
            return suffix, total, ess

        self._zres_suffix_w, self._zres_total_w, self._zres_ess = _suffix_tables(
            u_sorted, sig_sorted
        )

        # Per sky-band z-conditionals S(d_L | z, band b): Eq. (4) restricted
        # to band-b injections, with the per-(band, node) ESS floor fallback
        # to the z-only conditional (packet §7).
        nband = self._n_sky_bands
        sin_beta_abs = np.abs(np.cos(self._qS_arr))
        band_of_inj = np.clip(
            np.searchsorted(self._band_edges, sin_beta_abs, side="right") - 1,
            0,
            nband - 1,
        )
        band_suffix: list[npt.NDArray[np.float64]] = []
        band_total: list[npt.NDArray[np.float64]] = []
        band_ess: list[npt.NDArray[np.float64]] = []
        fallback = np.zeros((nband, u_nodes.size), dtype=np.bool_)
        for b in range(nband):
            if self._band_underpopulated[b]:
                # The pooled sky-band build already fell back to the full
                # pool for this band; its z-conditional IS the z-only one.
                band_suffix.append(self._zres_suffix_w)
                band_total.append(self._zres_total_w)
                band_ess.append(self._zres_ess)
                continue
            mask = band_of_inj == b
            d_hor_b = self._d_hor[mask]
            order_b = np.argsort(d_hor_b, kind="mergesort")
            sfx, tot, ess_b = _suffix_tables(u[mask][order_b], sigma_i[mask][order_b])
            band_suffix.append(sfx)
            band_total.append(tot)
            band_ess.append(ess_b)
            # ESS floor: repo's existing reliability convention (n >= 10,
            # _MIN_BAND_INJECTIONS), reused unchanged (packet §7).
            fallback[b, :] = ess_b < float(_MIN_BAND_INJECTIONS)
        self._zres_band_suffix_w = band_suffix
        self._zres_band_total_w = band_total
        self._zres_band_ess = band_ess
        self._zres_band_fallback = fallback

        logger.info(
            "z-resolved survival built: %d u-nodes on [0, %.4f], sigma_u=%.5f "
            "(Scott d=1, scale %.3g), Abramson lambda in [%.3g, %.3g], node ESS "
            "min/median = %.0f/%.0f, sky-band cells below ESS floor: %d/%d.",
            u_nodes.size,
            u_max,
            sigma_u,
            self._bandwidth_scale,
            float(np.min(lam)),
            float(np.max(lam)),
            float(np.min(self._zres_ess)),
            float(np.median(self._zres_ess)),
            int(np.sum(fallback)),
            int(fallback.size),
        )

    def _zres_node_pos(
        self, z: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.int_], npt.NDArray[np.float64]]:
        """u-node bracket (k0, frac) for query redshifts (clamped to the node span)."""
        assert self._zres_u_nodes is not None
        u_nodes = self._zres_u_nodes
        u = np.log1p(np.maximum(np.asarray(z, dtype=np.float64), 0.0))
        pos = np.interp(u, u_nodes, np.arange(u_nodes.size, dtype=np.float64))
        k0 = np.clip(np.floor(pos).astype(np.int_), 0, u_nodes.size - 2)
        frac = np.clip(pos - k0, 0.0, 1.0)
        return k0, frac

    def _zres_survival_at(
        self,
        query: npt.NDArray[np.float64],
        z: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        r"""Exact-in-d_L z-conditional survival ``S(d_L | z)`` (FIX-2, Eq. (4)).

        Linear interpolation in S across the two bracketing u-nodes (like the
        |sin beta| band interpolation); EXACT suffix-count survival in d_L at
        each node.  Guarantees by construction: S(0|z)=1, S(d_L > max
        d_hor|z)=0, monotone non-increasing in d_L, bounded [0, 1].

        Args:
            query: d_L query points [Gpc], 1-D.
            z: redshifts conditioning each query point (same shape).

        Returns:
            Survival values in [0, 1], same shape as ``query``.
        """
        if self._zres_degenerate:
            return self._survival_at(query)
        assert self._zres_suffix_w is not None and self._zres_total_w is not None
        k0, frac = self._zres_node_pos(z)
        idx = np.searchsorted(self._d_hor_sorted, query, side="left")
        inside = idx < self._n_inj
        idx_c = np.minimum(idx, self._n_inj - 1)
        s0 = np.where(inside, self._zres_suffix_w[idx_c, k0] / self._zres_total_w[k0], 0.0)
        s1 = np.where(inside, self._zres_suffix_w[idx_c, k0 + 1] / self._zres_total_w[k0 + 1], 0.0)
        surv = (1.0 - frac) * s0 + frac * s1
        return np.asarray(np.clip(surv, 0.0, 1.0), dtype=np.float64)

    def _zres_survival_at_band(
        self,
        band_idx: int,
        query: npt.NDArray[np.float64],
        z: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Per-band z-conditional survival ``S(d_L | z, band b)`` with ESS fallback.

        Eq. (4) restricted to band-b injections; a (band, u-node) cell below
        the ESS floor uses the z-only (band-marginal) conditional instead —
        NOT the fully pooled survival (packet §7 policy).
        """
        if self._zres_degenerate:
            return self._survival_at_band(band_idx, query)
        assert (
            self._zres_band_suffix_w is not None
            and self._zres_band_total_w is not None
            and self._zres_band_fallback is not None
            and self._zres_suffix_w is not None
            and self._zres_total_w is not None
        )
        k0, frac = self._zres_node_pos(z)
        # Global (z-only) node values — the fallback target.
        idx_g = np.searchsorted(self._d_hor_sorted, query, side="left")
        inside_g = idx_g < self._n_inj
        idx_gc = np.minimum(idx_g, self._n_inj - 1)
        # Band node values.
        d_hor_sorted_b = self._d_hor_sorted_by_band[band_idx]
        n_b = len(d_hor_sorted_b)
        sfx_b = self._zres_band_suffix_w[band_idx]
        tot_b = self._zres_band_total_w[band_idx]
        idx_b = np.searchsorted(d_hor_sorted_b, query, side="left")
        inside_b = idx_b < n_b
        idx_bc = np.minimum(idx_b, n_b - 1)
        fb = self._zres_band_fallback[band_idx]
        out = np.zeros_like(np.asarray(query, dtype=np.float64))
        for k, w_k in ((k0, 1.0 - frac), (k0 + 1, frac)):
            s_band = np.where(inside_b, sfx_b[idx_bc, k] / tot_b[k], 0.0)
            s_glob = np.where(inside_g, self._zres_suffix_w[idx_gc, k] / self._zres_total_w[k], 0.0)
            out = out + w_k * np.where(fb[k], s_glob, s_band)
        return np.asarray(np.clip(out, 0.0, 1.0), dtype=np.float64)

    def _require_zres_z(
        self,
        z: float | npt.NDArray[np.float64] | None,
        shape: tuple[int, ...],
    ) -> npt.NDArray[np.float64]:
        """Validate + broadcast the conditioning redshift for a flag-on query."""
        if z is None:
            msg = (
                "pdet_z_resolved is active: every 3D p_det query must pass the "
                "conditioning redshift z (coherent-consumer guard, FIX-2)."
            )
            raise ValueError(msg)
        z_arr = np.atleast_1d(np.asarray(z, dtype=np.float64))
        return np.ascontiguousarray(np.broadcast_to(z_arr, shape), dtype=np.float64)

    # ------------------------------------------------------------------
    # Sky-band survival (Change 1: ecliptic-latitude response anisotropy)
    # ------------------------------------------------------------------

    def _build_sky_bands(self) -> None:
        r"""Bin injections into equal-``|sin beta|`` bands and sort each band's horizon.

        Route A (PHYSICS-CHANGE-PROTOCOL Change 1): the sky dependence is
        MEASURED, not modelled.  ``beta = pi/2 - qS`` is the ecliptic latitude;
        ``u = |sin beta| = |cos qS|`` is uniform on ``[0, 1]`` for an isotropic
        sky, so equal-width ``u`` bands are equal-solid-angle bands.  Each
        injection is assigned to one band and each band gets its OWN sorted
        detection horizon, giving a per-band survival
        ``S_b(d_L) = P(d_hor >= d_L | band b)``.  Under-populated polar bands
        (< ``_MIN_BAND_INJECTIONS``) fall back to the pooled horizon.

        # Empirical per-band detection-horizon survival; azimuthal symmetry of
        # the LISA orbit-averaged response R = R(beta) (Cutler 1998,
        # arXiv:gr-qc/9703068; arXiv:1201.3684). 1/d_L amplitude scaling exact
        # (Hogg 1999, arXiv:astro-ph/9905116, Eq. 16). No separation ansatz.
        """
        nband = self._n_sky_bands
        sin_beta_abs = np.abs(np.cos(self._qS_arr))  # |sin beta| in [0, 1]
        # Equal-|sin beta| (equal-solid-angle) band edges and centres.
        self._band_edges: npt.NDArray[np.float64] = np.linspace(0.0, 1.0, nband + 1)
        self._band_centers: npt.NDArray[np.float64] = 0.5 * (
            self._band_edges[:-1] + self._band_edges[1:]
        )
        band_of_inj = np.clip(
            np.searchsorted(self._band_edges, sin_beta_abs, side="right") - 1,
            0,
            nband - 1,
        )
        self._d_hor_sorted_by_band: list[npt.NDArray[np.float64]] = []
        self._n_inj_by_band: list[int] = []
        self._band_underpopulated: list[bool] = []
        for b in range(nband):
            mask = band_of_inj == b
            n_b = int(np.count_nonzero(mask))
            if n_b < _MIN_BAND_INJECTIONS:
                # Fall back to the pooled (isotropic) horizon for this band.
                self._d_hor_sorted_by_band.append(self._d_hor_sorted)
                self._n_inj_by_band.append(self._n_inj)
                self._band_underpopulated.append(True)
                logger.warning(
                    "Sky band %d/%d (|sin beta| in [%.3f, %.3f]) under-populated "
                    "(%d < %d injections); falling back to the pooled isotropic horizon.",
                    b,
                    nband,
                    self._band_edges[b],
                    self._band_edges[b + 1],
                    n_b,
                    _MIN_BAND_INJECTIONS,
                )
            else:
                self._d_hor_sorted_by_band.append(np.sort(self._d_hor[mask], kind="mergesort"))
                self._n_inj_by_band.append(n_b)
                self._band_underpopulated.append(False)
        logger.info(
            "Sky bands built: %d equal-|sin beta| bands, per-band injection counts %s.",
            nband,
            self._n_inj_by_band,
        )

    def band_edges_sin_beta(self) -> npt.NDArray[np.float64]:
        """Equal-``|sin beta|`` band edges (length ``n_sky_bands + 1``) in ``[0, 1]``.

        The SAME edges the inference must use to bin pixels into bands so the
        sky marginal is invariant (PHYSICS-CHANGE-PROTOCOL test T3).
        """
        return self._band_edges.copy()

    def band_centers_sin_beta(self) -> npt.NDArray[np.float64]:
        """Band centres in ``|sin beta|`` (length ``n_sky_bands``)."""
        return self._band_centers.copy()

    def _survival_at_band(
        self, band_idx: int, query: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Exact survival ``P(d_hor >= d_L | band)`` for a single band."""
        d_hor_sorted = self._d_hor_sorted_by_band[band_idx]
        n_b = self._n_inj_by_band[band_idx]
        idx_below = np.searchsorted(d_hor_sorted, query, side="left")
        surv = (n_b - idx_below).astype(np.float64) / float(n_b)
        return np.asarray(np.clip(surv, 0.0, 1.0), dtype=np.float64)

    def survival_per_band(
        self,
        d_L: float | npt.NDArray[np.float64],
        z: float | npt.NDArray[np.float64] | None = None,
    ) -> npt.NDArray[np.float64]:
        r"""Per-band detection-horizon survival ``S_b(d_L)``; shape ``(n_sky_bands, Nq)``.

        The building block of the sky-resolved selection integrals: the
        inference forms the sky sum ``(1/Npix) sum_k p_det(d_L, Omega_k)`` as
        ``sum_b (n_pix_b/Npix) S_b(d_L)`` (each pixel takes its band's flat
        survival), and the missing-completion integral weights ``S_b`` by the
        per-band incompleteness ``(1/Npix) sum_{k in b}(1 - f_k(z))``.

        When the FIX-2 z-resolved estimator is active (``pdet_z_resolved``),
        ``z`` is REQUIRED and each band returns the z-conditional
        ``S(d_L | z, band b)`` (with the per-cell ESS-floor fallback to the
        z-only conditional, packet §7).

        Parameters
        ----------
        d_L : float or ndarray
            Luminosity distance query points [Gpc].
        z : float or ndarray, optional
            Conditioning redshift per query point (FIX-2 only; required when
            ``z_resolved`` is True, ignored otherwise).

        Returns
        -------
        ndarray, shape ``(n_sky_bands, Nq)``
            Band-resolved survival in ``[0, 1]``.
        """
        q = np.atleast_1d(np.asarray(d_L, dtype=np.float64))
        out = np.empty((self._n_sky_bands, q.size), dtype=np.float64)
        if self._z_resolved:
            z_arr = self._require_zres_z(z, q.shape)
            for b in range(self._n_sky_bands):
                out[b, :] = self._zres_survival_at_band(b, q, z_arr)
            return out
        for b in range(self._n_sky_bands):
            out[b, :] = self._survival_at_band(b, q)
        return out

    def _interp_survival_in_sin_beta(
        self,
        dl_arr: npt.NDArray[np.float64],
        sin_beta_abs: npt.NDArray[np.float64],
        z: npt.NDArray[np.float64] | None = None,
    ) -> npt.NDArray[np.float64]:
        """Survival interpolated linearly in ``|sin beta|`` across band centres.

        Avoids step artefacts at band boundaries; nearest-band clamp outside the
        centre range.  ``dl_arr`` and ``sin_beta_abs`` are the same shape.
        """
        s_all = self.survival_per_band(dl_arr, z)  # (nband, N)
        centers = self._band_centers
        n_query = dl_arr.size
        if self._n_sky_bands == 1:
            return np.asarray(np.clip(s_all[0, :], 0.0, 1.0), dtype=np.float64)
        idx_hi = np.clip(
            np.searchsorted(centers, sin_beta_abs, side="left"),
            1,
            self._n_sky_bands - 1,
        )
        idx_lo = idx_hi - 1
        c_lo = centers[idx_lo]
        c_hi = centers[idx_hi]
        # Clamp weight to [0, 1] => nearest-band value for u outside centre span.
        w = np.clip((sin_beta_abs - c_lo) / (c_hi - c_lo), 0.0, 1.0)
        cols = np.arange(n_query)
        s_lo = s_all[idx_lo, cols]
        s_hi = s_all[idx_hi, cols]
        result = (1.0 - w) * s_lo + w * s_hi
        return np.asarray(np.clip(result, 0.0, 1.0), dtype=np.float64)

    def detection_probability_without_bh_mass_sky(
        self,
        d_L: float | npt.NDArray[np.float64],
        phi: float | npt.NDArray[np.float64],
        theta: float | npt.NDArray[np.float64],
        *,
        h: float,
        z: float | npt.NDArray[np.float64] | None = None,
    ) -> float | npt.NDArray[np.float64]:
        r"""Sky-resolved detection probability ``p_det(d_L | Omega)`` (Change 1).

        Maps the ecliptic sky direction ``(phi, theta)`` to the ecliptic
        latitude band via ``|sin beta| = |cos theta|`` (``beta = pi/2 - theta``)
        and returns that band's detection-horizon survival, interpolated
        linearly in ``|sin beta|`` across band centres.  ``phi`` is accepted but
        unused (azimuthal symmetry of the orbit-averaged response, Cutler 1998,
        arXiv:gr-qc/9703068).

        # p_det = P(d_hor >= d_L | ecliptic-latitude band); empirical per-band
        # survival, azimuthally symmetric (Cutler 1998, arXiv:gr-qc/9703068).

        Reduces to the pooled isotropic survival when ``n_sky_bands == 1`` (the
        regression fallback; PHYSICS-CHANGE-PROTOCOL test T1).

        Parameters
        ----------
        d_L : float or ndarray
            Luminosity distance [Gpc].
        phi : float or ndarray
            Ecliptic azimuth [rad] (unused; azimuthal symmetry).
        theta : float or ndarray
            Ecliptic colatitude [rad]; ``beta = pi/2 - theta``.
        h : float
            Dimensionless Hubble parameter (horizon is h-invariant).

        Returns
        -------
        float or ndarray
            Detection probability in ``[0, 1]``.
        """
        self._get_or_build_grid(h)  # parity: register the (h-invariant) grid/flags
        dl_arr = np.atleast_1d(np.asarray(d_L, dtype=np.float64))
        theta_arr = np.atleast_1d(np.asarray(theta, dtype=np.float64))
        dl_b, theta_b = np.broadcast_arrays(dl_arr, theta_arr)
        dl_b = np.ascontiguousarray(dl_b, dtype=np.float64)
        sin_beta_abs = np.abs(np.cos(theta_b))  # |sin beta| = |cos theta|
        z_flat: npt.NDArray[np.float64] | None = None
        if self._z_resolved:
            z_flat = self._require_zres_z(z, dl_b.shape).ravel()
        result = self._interp_survival_in_sin_beta(dl_b.ravel(), sin_beta_abs.ravel(), z_flat)
        result = result.reshape(dl_b.shape)
        if np.ndim(d_L) == 0 and np.ndim(theta) == 0:
            return float(result.reshape(-1)[0])
        return np.asarray(result, dtype=np.float64)

    def _get_or_build_grid(
        self, h: float
    ) -> tuple[RegularGridInterpolator, RegularGridInterpolator]:
        """Return the (single, h-invariant) survival interpolators.

        The detection horizon is h-invariant, so the survival grids are built
        ONCE and the same cached interpolators are returned for ANY h — this
        is the intended speed win (no per-h rebuild).  The ``h`` argument is
        accepted for API compatibility and used only for the (per-h)
        quality-flags registration.

        Args:
            h: Hubble parameter value (only used to register quality flags).

        Returns:
            Tuple of (2D interpolator, 1D interpolator).
        """
        if self._shared_grid is None:
            interp_2d = self._build_grid_2d(self._snr_threshold)
            interp_1d = self._build_grid_1d(self._snr_threshold)
            self._shared_grid = (interp_2d, interp_1d)

        # Register the same single grid object for this h (back-compat with
        # call sites that look it up in ``_grid_cache``).
        self._grid_cache[h] = self._shared_grid

        # Register per-h quality flags (the values are h-invariant, but the
        # keying by h is preserved for the existing API).
        if h not in self._quality_flags:
            self._register_quality_flags(h)

        return self._shared_grid

    def _grid_support(
        self,
    ) -> tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
    ]:
        """Fixed d_L / M_z grid edges and centers (h-invariant)."""
        dl_max = self._dl_global_max
        dl_edges = np.linspace(0.0, dl_max, self._dl_bins + 1)
        dl_centers = 0.5 * (dl_edges[:-1] + dl_edges[1:])

        M_min = float(np.min(self._log_M_z))  # noqa: N806
        M_max = float(np.max(self._log_M_z))  # noqa: N806
        # M grid is geomspace over the observer-frame M_z range (M_k·(1+z_k)).
        m_lo = 10.0**M_min * 0.9
        m_hi = 10.0**M_max * 1.1
        M_edges = np.geomspace(m_lo, m_hi, self._mass_bins + 1)  # noqa: N806
        M_centers = np.sqrt(M_edges[:-1] * M_edges[1:])  # noqa: N806
        return dl_edges, dl_centers, M_edges, M_centers

    def _build_grid_1d(
        self,
        snr_threshold: float,
        *,
        h_val: float | None = None,
    ) -> RegularGridInterpolator:
        """Build the 1D detection-horizon survival grid p_det(d_L).

        p_det_1d[i] = weighted survival at dl_centers[i]
                    = (Σ_k 1[d_hor_k >= dl_centers[i]]) / N      (uniform w).

        # p_det = survival function of the detection horizon, P(d_hor >= d_L),
        # with d_hor = SNR·d_L/threshold.
        # Finn & Chernoff (1993), arXiv:gr-qc/9301003; Finn (1996),
        # arXiv:gr-qc/9601048 (p_det = P(Theta > Theta_thr)).

        Computed efficiently from the sorted horizon via ``np.searchsorted``
        on the suffix-count of (uniform) weights.

        Args:
            snr_threshold: SNR detection threshold (recorded for symmetry with
                the 2D builder; the horizon already encodes it).
            h_val: Unused; accepted for diagnostic symmetry.

        Returns:
            RegularGridInterpolator for P_det(d_L) on the fixed grid centers
            (length ``dl_bins``).
        """
        _, dl_centers, _, _ = self._grid_support()
        # Exact survival at each center (monotone by construction).
        p_det_1d = self._survival_at(dl_centers)

        # fill_value=None → LINEAR extrapolation outside the grid (scipy
        # semantics); harmless here because the public accessors below
        # override out-of-grid behavior with the EXACT searchsorted survival.
        return RegularGridInterpolator(
            (dl_centers,),
            p_det_1d,
            method="linear",
            bounds_error=False,
            fill_value=None,
        )

    def _build_grid_2d(
        self,
        snr_threshold: float,
        *,
        h_val: float | None = None,
        weights: npt.NDArray[np.float64] | None = None,
    ) -> RegularGridInterpolator:
        """Build the 2D detection-horizon survival grid p_det(d_L, M_z).

        p_det_grid[i, j] = K_M(log10 M_z,k − log10 M_z,j)-weighted survival at
        (dl_centers[i], M_centers[j]):

            p_det(d_L, M_z) = Σ_k K_M(log10 M_z,k − log10 M_z) · 1[d_hor_k ≥ d_L]
                              / Σ_k K_M(log10 M_z,k − log10 M_z)

        # p_det = survival function of the detection horizon, P(d_hor >= d_L),
        # with d_hor = SNR·d_L/threshold.
        # Finn & Chernoff (1993), arXiv:gr-qc/9301003; Finn (1996),
        # arXiv:gr-qc/9601048 (p_det = P(Theta > Theta_thr)).

        The kernel acts ONLY in log10 M_z (Gaussian, bandwidth σ_logM from
        Scott's rule via ``_compute_bandwidths``).  The survival is exact in
        d_L (no kernel / bandwidth in d_L).  Importance-sampling ``weights``
        compose multiplicatively with the M_z kernel weights.

        Computed efficiently: sort d_hor ascending once (done in ``__init__``);
        for each M-center the per-injection weight is ``K_M · w_IS``; a
        suffix-cumsum of those weights over the d_hor-sorted order gives the
        weighted survival at each ``dl_center`` via ``np.searchsorted``.

        The M axis is observer-frame M_z = M_source · (1 + z_inj).
        Maggiore (2008) Vol 1 §4.1.4; Mandel, Farr & Gair (2019)
        arXiv:1809.02063 §2.

        Args:
            snr_threshold: SNR detection threshold (already encoded in the
                horizon).
            h_val: Unused; accepted for diagnostic symmetry.
            weights: Per-injection importance weights, shape (N,).  If None,
                all weights are 1.

        Returns:
            RegularGridInterpolator for P_det(d_L, M_z) on the fixed centers.

        References:
            Gray et al. (2020), arXiv:1908.06050 — selection-function form.
            Scott (1992), Multivariate Density Estimation, Ch. 6 — bandwidth.
            Mandel, Farr & Gair (2019), arXiv:1809.02063 Eq. 18 — IS weights.
        """
        n_events = self._n_inj
        if weights is not None:
            w_is = np.asarray(weights, dtype=np.float64)
            if len(w_is) != n_events:
                msg = f"weights length {len(w_is)} != injection count {n_events}"
                raise ValueError(msg)
        else:
            w_is = np.ones(n_events, dtype=np.float64)

        _, dl_centers, _, M_centers = self._grid_support()
        log_M_centers = np.log10(M_centers)  # noqa: N806

        # Mass-axis bandwidth (Scott's rule); the d_L axis carries no kernel.
        _, sigma_log_M = self._compute_bandwidths(self._dl_raw, self._log_M_z)
        inv_two_sigma_log_M_sq = 0.5 / (sigma_log_M * sigma_log_M)

        # Sort by d_hor ascending so a suffix-sum gives survival counts.
        sort_idx = np.argsort(self._d_hor, kind="mergesort")
        d_hor_sorted = self._d_hor[sort_idx]
        log_M_sorted = self._log_M_z[sort_idx]  # noqa: N806
        w_is_sorted = w_is[sort_idx]

        # For each d_L center, count injections with d_hor >= d_L → index into
        # the sorted-ascending suffix.  lo[i] = first sorted index with
        # d_hor >= dl_centers[i].
        lo = np.searchsorted(d_hor_sorted, dl_centers, side="left")  # shape (dl_bins,)

        p_det_grid = np.zeros((self._dl_bins, self._mass_bins), dtype=np.float64)

        # Per M-center weights K_M(log10 M_z,k − log10 M_z,j) · w_IS_k, sorted
        # in the d_hor-ascending order.  Suffix-cumsum (reverse cumsum) gives
        # Σ_{k: d_hor_k >= d_L} K_M w_IS at every cut point.
        # Total kernel mass per M-center (denominator, d_L-independent):
        diff_log_M = log_M_sorted[:, None] - log_M_centers[None, :]  # noqa: N806
        K_M = np.exp(-inv_two_sigma_log_M_sq * diff_log_M**2)  # noqa: N806
        kernel_w = K_M * w_is_sorted[:, None]  # (N, M_bins)
        total_w = kernel_w.sum(axis=0)  # (M_bins,)

        # Reverse cumulative sum over the sorted axis → suffix sums.
        suffix_w = np.cumsum(kernel_w[::-1, :], axis=0)[::-1, :]  # (N, M_bins)

        for i in range(self._dl_bins):
            cut = int(lo[i])
            if cut >= n_events:
                # No injection has d_hor >= dl_centers[i] → survival 0.
                continue
            num = suffix_w[cut, :]  # Σ_{d_hor >= dl_centers[i]} K_M w_IS
            p_det_grid[i, :] = np.divide(
                num,
                total_w,
                out=np.zeros(self._mass_bins, dtype=np.float64),
                where=total_w > 0.0,
            )

        p_det_grid = np.clip(p_det_grid, 0.0, 1.0)

        # fill_value=None → LINEAR extrapolation outside the grid (scipy
        # semantics, NOT nearest).  Tolerated only because the public 2D
        # accessor clamps BOTH axes before querying: d_L below first center /
        # above last center, and M_z to the grid edges (true nearest).
        return RegularGridInterpolator(
            (dl_centers, M_centers),
            p_det_grid,
            method="linear",
            bounds_error=False,
            fill_value=None,
        )

    def _register_quality_flags(self, h: float) -> None:
        """Compute and store per (dl-bin × M-bin) quality flags for ``h``.

        Per cell (uniform weights):
        - ``n_total``: injection count per (dl-bin, M-bin).
        - ``n_detected``: detected (d_hor >= dl-bin lower edge) count per cell.
        - ``reliable``: n_total >= 10.
        - ``n_eff``: n_total (uniform weights).
        - ``dl_edges`` / ``M_edges``: bin edges.

        The flags are h-invariant (the horizon is h-invariant); the per-h key
        is preserved for the existing API.
        """
        dl_edges, _, M_edges, _ = self._grid_support()

        # Histogram injections into (dl, M_z) cells.
        # dl-axis: by the injection's own d_L (luminosity_distance); M-axis: by
        # observer-frame log10 M_z.
        log_M_edges = np.log10(M_edges)  # noqa: N806
        # Detection at the lower edge of each dl-bin: d_hor >= dl_edges[i].
        # n_total per cell counts ALL injections falling in the cell; we use
        # the injection d_L as the dl-coordinate for "count per cell".
        dl_coord = self._dl_raw
        m_coord = self._log_M_z

        # Bin indices (np.digitize-style via searchsorted on edges).
        dl_idx = np.clip(
            np.searchsorted(dl_edges, dl_coord, side="right") - 1, 0, self._dl_bins - 1
        )
        m_idx = np.clip(
            np.searchsorted(log_M_edges, m_coord, side="right") - 1, 0, self._mass_bins - 1
        )

        n_total = np.zeros((self._dl_bins, self._mass_bins), dtype=np.float64)
        n_detected = np.zeros((self._dl_bins, self._mass_bins), dtype=np.float64)

        np.add.at(n_total, (dl_idx, m_idx), 1.0)
        # Detected within the cell: injection's horizon reaches the cell's
        # lower dl edge (d_hor >= dl_edges[dl_idx]).
        detected = self._d_hor >= dl_edges[dl_idx]
        np.add.at(n_detected, (dl_idx, m_idx), detected.astype(np.float64))

        self._quality_flags[h] = {
            "n_total": n_total,
            "n_detected": n_detected,
            "reliable": (n_total >= 10.0),
            "dl_edges": dl_edges.copy(),
            "M_edges": M_edges.copy(),
            "n_eff": n_total.copy(),
        }
        # FIX-2 diagnostics: per-u-node kernel ESS of the z-resolved survival
        # (packet §7 "report per-node ESS alongside").
        if (
            self._z_resolved
            and not self._zres_degenerate
            and self._zres_ess is not None
            and self._zres_u_nodes is not None
        ):
            self._quality_flags[h]["zres_u_nodes"] = self._zres_u_nodes.copy()
            self._quality_flags[h]["zres_ess"] = self._zres_ess.copy()

    def quality_flags(self, h: float) -> dict[str, npt.NDArray[np.float64] | npt.NDArray[np.bool_]]:
        """Return per-bin quality metadata for the given h value.

        Quality flags are diagnostic metadata.  They do **not** affect the
        P_det survival result.  If the grid for this h has not been built yet,
        it will be built (the single h-invariant survival grid).

        The returned dict contains:

        - ``n_total``: float array (dl_bins, M_bins) -- injection count per cell.
        - ``n_detected``: float array (dl_bins, M_bins) -- detected count per
          cell (d_hor >= cell lower dl edge).
        - ``reliable``: bool array (dl_bins, M_bins) -- True where n_total >= 10.
        - ``dl_edges``: float array (dl_bins+1,) -- d_L bin edges in Gpc.
        - ``M_edges``: float array (M_bins+1,) -- M_z bin edges.
        - ``n_eff``: float array (dl_bins, M_bins) -- effective sample size
          (= n_total under uniform weights).

        Args:
            h: Hubble parameter value.

        Returns:
            Dict of quality flag arrays.

        Raises:
            ValueError: If no quality flags are available after construction.
        """
        if h not in self._quality_flags:
            self._get_or_build_grid(h)

        if h not in self._quality_flags:
            msg = f"No quality flags for h={h:.4f} after grid construction."
            raise ValueError(msg)
        return self._quality_flags[h]

    def detection_probability_with_bh_mass_interpolated(
        self,
        d_L: float | npt.NDArray[np.float64],
        M_z: float | npt.NDArray[np.float64],
        phi: float | npt.NDArray[np.float64],
        theta: float | npt.NDArray[np.float64],
        *,
        h: float,
    ) -> float | npt.NDArray[np.float64]:
        """Detection probability including BH mass dependence (survival form).

        Interpolates the 2D detection-horizon survival grid
        ``p_det(d_L, M_z) = K_M-weighted P(d_hor >= d_L)`` with a linear
        ``RegularGridInterpolator`` (``bounds_error=False``, ``fill_value=None``).

        Boundary handling (no extrapolation machinery — the survival grid is
        naturally boundary-correct):

        * d_L below the first center → clamp to the first center (survival ≈ 1
          there, since the grid starts near d_L = 0).
        * d_L above the last center → 0 (no injection's horizon reaches there).
        * M_z outside the grid range → clamped to the nearest grid edge.
          (``fill_value=None`` alone would LINEARLY extrapolate — made-up but
          plausible-looking values; the explicit clip enforces true nearest.)

        The result is monotone non-increasing in d_L and bounded in [0, 1].

        Sky angles (phi, theta) are accepted for API compatibility but are
        marginalized over internally (D-02).

        Args:
            d_L: Luminosity distance in Gpc.
            M_z: Observer-frame (redshifted) BH mass in solar masses.
            phi: Sky angle phi (unused, marginalized over).
            theta: Sky angle theta (unused, marginalized over).
            h: Dimensionless Hubble parameter (accepted; horizon is
                h-invariant).

        Returns:
            Detection probability in [0, 1].

        References:
            Finn & Chernoff (1993), arXiv:gr-qc/9301003; Finn (1996),
            arXiv:gr-qc/9601048.
            Gray et al. (2020), arXiv:1908.06050, Eq. (8).
            Mandel, Farr & Gair (2019), arXiv:1809.02063.
        """
        interp_2d, _ = self._get_or_build_grid(h)
        dl_centers = np.asarray(interp_2d.grid[0])
        M_centers = np.asarray(interp_2d.grid[1])  # noqa: N806

        dl_arr = np.atleast_1d(np.asarray(d_L, dtype=np.float64))
        M_arr = np.atleast_1d(np.asarray(M_z, dtype=np.float64))  # noqa: N806

        dl_min = float(dl_centers[0])
        dl_max = float(dl_centers[-1])

        # d_L below first center → clamp to first center (survival ≈ 1 there);
        # d_L above last center → 0.  M_z outside range → clamp to the grid
        # edge (true nearest): RegularGridInterpolator with fill_value=None
        # would silently LINEAR-extrapolate outside the M_z axis, inventing
        # p_det values beyond the injected mass support (readiness sweep
        # A2-EXTRAP, 2026-07-03).
        dl_query = np.clip(dl_arr, dl_min, dl_max)
        M_query = np.clip(M_arr, float(M_centers[0]), float(M_centers[-1]))  # noqa: N806
        result = np.clip(interp_2d(np.column_stack([dl_query, M_query])), 0.0, 1.0)

        # Above the last center the survival is exactly 0.
        result = np.where(dl_arr > dl_max, 0.0, result)

        result = np.asarray(np.clip(result, 0.0, 1.0), dtype=np.float64)

        if np.ndim(d_L) == 0 and np.ndim(M_z) == 0:
            return float(result[0])
        return result

    def get_dl_max(self, h: float) -> float:
        """Return the maximum d_L of the 1D P_det grid for the given h value.

        This is the upper edge of the d_L support, i.e. the maximum detection
        horizon padded by ``_DL_PADDING_FACTOR``.  Needed to compute
        ``z_max(h)`` for the full-volume denominator integral.

        Args:
            h: Dimensionless Hubble parameter (horizon is h-invariant).

        Returns:
            Maximum d_L in Gpc.
        """
        # Ensure grid is built (populates cache)
        self._get_or_build_grid(h)
        # Reconstruct dl_max from the 1D interpolator's grid points
        _, interp_1d = self._grid_cache[h]
        dl_centers = interp_1d.grid[0]
        spacing = float(dl_centers[1] - dl_centers[0])
        return float(dl_centers[-1] + spacing / 2)

    def validate_coverage(
        self,
        h: float,
        crb_df: pd.DataFrame,
    ) -> float:
        """Compute fraction of events whose 4-sigma d_L bounds fall within the P_det grid.

        For each event, compute d_L +/- 4*sigma_dL from the Cramer-Rao bounds.
        Check if both bounds fall within the grid's d_L range.

        Args:
            h: Hubble parameter value (to build/retrieve grid).
            crb_df: DataFrame with columns ``luminosity_distance`` and
                ``delta_luminosity_distance_delta_luminosity_distance`` (variance).

        Returns:
            Coverage fraction in [0, 1].
        """
        # Build/retrieve the grid to get d_L edge range
        self._get_or_build_grid(h)
        _, interp_1d = self._grid_cache[h]
        dl_centers = interp_1d.grid[0]
        spacing = float(dl_centers[1] - dl_centers[0])
        dl_grid_min = float(dl_centers[0] - spacing / 2)
        dl_grid_max = float(dl_centers[-1] + spacing / 2)

        # Extract per-event d_L and sigma_dL from CRB DataFrame
        d_L_vals = crb_df["luminosity_distance"].values.astype(np.float64)
        sigma_dL = np.sqrt(
            crb_df["delta_luminosity_distance_delta_luminosity_distance"].values.astype(np.float64)
        )

        # Compute 4-sigma bounds
        lower_bounds = d_L_vals - 4.0 * sigma_dL
        upper_bounds = d_L_vals + 4.0 * sigma_dL

        # Event is covered if both bounds fall within the grid range
        covered = (lower_bounds >= dl_grid_min) & (upper_bounds <= dl_grid_max)
        n_covered = int(np.sum(covered))
        n_total = len(d_L_vals)

        coverage_fraction = n_covered / n_total if n_total > 0 else 1.0

        logger.info(
            "P_det grid coverage: %.1f%% of events have 4-sigma d_L bounds within grid (%d/%d)",
            coverage_fraction * 100,
            n_covered,
            n_total,
        )
        if coverage_fraction < 0.95:
            logger.warning(
                "P_det grid coverage %.1f%% is below 95%% threshold. "
                "Consider increasing --pdet_dl_bins.",
                coverage_fraction * 100,
            )

        return coverage_fraction

    def detection_probability_without_bh_mass_interpolated_zero_fill(
        self,
        d_L: float | npt.NDArray[np.float64],
        phi: float | npt.NDArray[np.float64],
        theta: float | npt.NDArray[np.float64],
        *,
        h: float,
        z: float | npt.NDArray[np.float64] | None = None,
    ) -> float | npt.NDArray[np.float64]:
        """Detection probability marginalized over BH mass (exact survival).

        Returns the EXACT detection-horizon survival
        ``p_det(d_L) = P(d_hor >= d_L)`` via ``np.searchsorted`` on the stored
        sorted horizon with (uniform) suffix weights.  This guarantees by
        construction:

        * ``p(0) = 1`` (every injection's horizon is >= 0),
        * ``p(d_L > max d_hor) = 0``,
        * monotone non-increasing in d_L.

        The function name retains the legacy ``_zero_fill`` suffix for
        backward compatibility with the >=6 call sites in
        :mod:`bayesian_statistics`.  No bridge / slope-matched-clamp
        extrapolation is needed: the survival is naturally boundary-correct.

        # p_det = survival function of the detection horizon, P(d_hor >= d_L),
        # with d_hor = SNR·d_L/threshold.
        # Finn & Chernoff (1993), arXiv:gr-qc/9301003; Finn (1996),
        # arXiv:gr-qc/9601048 (p_det = P(Theta > Theta_thr)).

        Sky angles (phi, theta) are accepted for API compatibility but are
        marginalized over internally (D-02).

        Args:
            d_L: Luminosity distance in Gpc.
            phi: Sky angle phi (unused, marginalized over).
            theta: Sky angle theta (unused, marginalized over).
            h: Dimensionless Hubble parameter (accepted; horizon is
                h-invariant).

        Returns:
            Detection probability in [0, 1].

        References:
            Finn & Chernoff (1993), arXiv:gr-qc/9301003; Finn (1996),
            arXiv:gr-qc/9601048.
            Gray et al. (2020), arXiv:1908.06050, Eq. (A.19).
        """
        # Ensure the (h-invariant) grid / quality flags are registered for h.
        self._get_or_build_grid(h)

        dl_arr = np.atleast_1d(np.asarray(d_L, dtype=np.float64))
        if self._z_resolved:
            # FIX-2: z-conditional survival S(d_L | z), Eq. (2)/(4) in
            # DERIVATION_ZRESOLVED_SURVIVAL.md.  Finn & Chernoff (1993),
            # arXiv:gr-qc/9301003; Mandel-Farr-Gair (2019), arXiv:1809.02063.
            z_arr = self._require_zres_z(z, dl_arr.shape)
            result = self._zres_survival_at(dl_arr, z_arr)
        else:
            result = self._survival_at(dl_arr)

        if np.ndim(d_L) == 0:
            return float(result[0])
        return result

    def detection_probability_without_bh_mass_interpolated(
        self,
        d_L: float | npt.NDArray[np.float64],
        phi: float | npt.NDArray[np.float64],
        theta: float | npt.NDArray[np.float64],
        *,
        h: float,
        z: float | npt.NDArray[np.float64] | None = None,
    ) -> float | npt.NDArray[np.float64]:
        """Detection probability marginalized over BH mass (exact survival).

        Drop-in replacement for
        ``DetectionProbability.detection_probability_without_bh_mass_interpolated``
        with an additional ``h`` keyword.  Returns the EXACT detection-horizon
        survival ``p_det(d_L) = P(d_hor >= d_L)`` (identical to the
        ``_zero_fill`` accessor — the survival is naturally boundary-correct,
        so the two accessors coincide).

        # p_det = survival function of the detection horizon, P(d_hor >= d_L),
        # with d_hor = SNR·d_L/threshold.
        # Finn & Chernoff (1993), arXiv:gr-qc/9301003; Finn (1996),
        # arXiv:gr-qc/9601048 (p_det = P(Theta > Theta_thr)).

        Sky angles (phi, theta) are accepted for API compatibility but are
        marginalized over internally (D-02).

        Args:
            d_L: Luminosity distance in Gpc.
            phi: Sky angle phi (unused, marginalized over).
            theta: Sky angle theta (unused, marginalized over).
            h: Dimensionless Hubble parameter.

        Returns:
            Detection probability in [0, 1].

        References:
            Finn & Chernoff (1993), arXiv:gr-qc/9301003; Finn (1996),
            arXiv:gr-qc/9601048.
            Gray et al. (2020), arXiv:1908.06050, Eq. (8).
        """
        self._get_or_build_grid(h)

        dl_arr = np.atleast_1d(np.asarray(d_L, dtype=np.float64))
        if self._z_resolved:
            # FIX-2: z-conditional survival S(d_L | z) — identical to the
            # _zero_fill accessor (the two accessors coincide).
            z_arr = self._require_zres_z(z, dl_arr.shape)
            result = self._zres_survival_at(dl_arr, z_arr)
        else:
            result = self._survival_at(dl_arr)

        if np.ndim(d_L) == 0:
            return float(result[0])
        return result
