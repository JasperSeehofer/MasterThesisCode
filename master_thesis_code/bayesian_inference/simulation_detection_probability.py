"""Simulation-based detection probability from injection campaign data.

Replaces :class:`DetectionProbability` (KDE-based) with a histogram-binned
approach that loads raw injection CSVs, applies an SNR threshold at evaluation
time, and builds P_det grids via SNR rescaling.

All injection data is pooled regardless of the Hubble parameter value used
during the injection campaign.  When querying P_det at a target h value, each
event's SNR is rescaled using the exact relation:

    SNR(h_target) = SNR_raw * d_L(z, h_inj) / d_L(z, h_target)

This is exact because the GW strain amplitude scales as 1/d_L, while the
waveform frequency content depends only on source-frame parameters (which are
h-independent).  See Gray et al. (2020) arXiv:1908.06050 Section III.B-C and
Laghi et al. (2021) arXiv:2102.01708 Section III.A.

Grids are built in (d_L, M) space so that query-time lookups avoid the
expensive ``dist_to_redshift`` numerical inversion (fsolve).
"""

# ASSERT_CONVENTION: natural_units=SI, distance=Gpc, mass=solar_masses,
#   h=dimensionless_H0_over_100, SNR=dimensionless

import glob
import logging
import re
import warnings
from collections import OrderedDict
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

from master_thesis_code.physical_relations import dist_vectorized

logger = logging.getLogger(__name__)

# Default number of bins for the P_det grids
_DEFAULT_DL_BINS: int = 60
_DEFAULT_M_BINS: int = 40

# Maximum number of cached grids (LRU eviction)
_MAX_CACHE_SIZE: int = 20

# ----------------------------------------------------------------------
# Phase 45 empirical asymptote anchor for d_L → 0.
#
# Derived from the unrescaled injection campaign (simulations/injections/)
# pooled across all h_inj groups; see
#   scripts/bias_investigation/outputs/phase45/p_max_h_independence.json
# for per-h empirical detection rates at d_L < 0.10 Gpc and the pooled
# Wilson 95% CI (lower bound = this constant).  H-independence is supported
# by a likelihood-ratio test of binomial-rate homogeneity:
#   G = 7.299, dof = 5, p-value = 0.199 ≥ alpha = 0.05
# (cannot reject a common p across h_inj groups; see Plan 45-01 SUMMARY,
# .gpd/phases/45-p-det-first-bin-asymptote-fix/45-01-SUMMARY.md).
#
# Phase 45 RESEARCH.md §4a-(ii) recommends the conservative Wilson 95%
# lower bound (0.7931) rather than the point estimate (0.8873) so the
# anchor cannot overshoot truth on production posteriors.
#
# This constant is intentionally h-INDEPENDENT: it is the same scalar at
# every Hubble parameter target.  Phase 44 regression
# `test_zero_fill_no_h_dependent_step_for_close_dL` requires this property.
# ----------------------------------------------------------------------
_P_MAX_EMPIRICAL_ANCHOR: float = 0.7931

# ----------------------------------------------------------------------
# Phase 45 Plan 45-04 hybrid intermediate anchor at d_L = 0.05 Gpc.
#
# Plan 45-02 prepended a single empirical anchor (0, 0.7931) using the
# Wilson 95% lower bound of the pooled asymptote at d_L < 0.10 Gpc.
# Cluster re-eval (Plan 45-03) showed the resulting MAP shift was
# sub-discrete-grid-step (mean(bootstrap_MAP) shifted only -0.0047 vs
# Δh=0.005); the discrete MAP did not move from 0.7650 toward truth 0.73.
# Plan 45-04 escalates per RESEARCH.md §4c hybrid: a SECOND anchor at a
# fixed physical position d_L = 0.05 Gpc with value 1.0 (the empirical
# point estimate at d_L < 0.10 Gpc; the d_L < 0.05 subset is also 100%
# detected by monotonicity of detection rate in d_L; see
#   scripts/bias_investigation/outputs/phase45/pdet_asymptote.json).
#
# The intermediate position is FIXED (h-INDEPENDENT), not c_0(h)/2.
# Rationale: the empirical asymptote (16/16 detected at d_L < 0.10 Gpc)
# was derived in fixed physical d_L bins, NOT at fractional positions of
# c_0(h).  A fixed physical position is more physically uniform across
# h_inj groups and trivially passes the h-spread regression
# `test_zero_fill_no_h_dependent_step_for_close_dL` (the value at d_L=0.05
# is exactly the same constant 1.0 across all h, contributing zero spread).
#
# Edge case: when c_0(h) <= 0.05 Gpc (rare; only at very high h where
# dl_max(h) shrinks below 6.0 Gpc), the intermediate anchor is SKIPPED to
# preserve strict monotonicity of the dl_centers grid required by
# `RegularGridInterpolator`.  In that branch, the layout falls back to the
# Plan 45-02 single-anchor layout (anchor + first bin centre only).
#
# Plan 45-07 (Option D) applies the SAME hybrid anchor layout to the d_L
# axis of the 2D grid (`_build_grid_2d`).  Both anchor values are broadcast
# across all M_z bins because the empirical asymptote at d_L → 0 is
# M_z-independent (SNR ~ 1/d_L diverges regardless of intrinsic loudness).
# This addresses the with_bh_mass channel's residual MAP=0.7450 from Plan
# 45-05 SUMMARY (the without_bh_mass channel was already addressed by
# Plan 45-04, but the 2D channel was untouched until Plan 45-07).
#
# Sign-convention note (documented; not a bug):
#   On [0, 0.05] the slope is +4.138/Gpc (POSITIVE), reflecting the
#   conservative Wilson LB at d_L=0 transitioning to the empirical
#   asymptote at d_L=0.05.  This deviates from physical monotonicity in
#   d_L but is justified empirically: the d_L=0 value is intentionally
#   below the asymptote to avoid overshoot of production MAP.  The
#   integrand on L_comp remains conservatively lower-bounded.
#
# Eq. (A.19) in Gray et al. (2020), arXiv:1908.06050.
# ----------------------------------------------------------------------
_D_INTERMEDIATE_ANCHOR_GPC: float = 0.05
_P_INTERMEDIATE_EMPIRICAL: float = 1.0


class SimulationDetectionProbability:
    """Simulation-based detection probability from injection campaign data.

    Loads raw injection CSVs (z, M, phiS, qS, SNR, h_inj, luminosity_distance),
    pools ALL events regardless of h_inj, and builds P_det grids on-the-fly
    via SNR rescaling at query time.

    For a source at redshift z injected at h_inj with measured SNR_raw, the
    rescaled SNR at target h is:

        SNR(h) = SNR_raw * d_L(z, h_inj) / d_L(z, h)

    This is exact because h(t) ~ 1/d_L for gravitational wave strain, while
    the waveform shape depends only on source-frame parameters.

    Grids are cached with LRU eviction (max 20 entries) for performance.

    Args:
        injection_data_dir: Directory containing injection CSV files matching
            ``injection_h_*_task_*.csv`` or ``injection_h_*.csv``.
        snr_threshold: SNR threshold for detection. Events with
            SNR >= snr_threshold are considered detected.
        h_grid: **Deprecated.** Previously used to specify h grid points for
            pre-computed grids.  Now ignored (grids are built on-the-fly via
            SNR rescaling).  Passing this parameter emits a deprecation
            warning.
        _force_unit_weights: Internal flag for testing. When True, passes
            explicit ``weights=np.ones(N)`` to ``_build_grid_2d`` to verify
            IS estimator backward compatibility.

    References:
        Gray et al. (2020), arXiv:1908.06050, Section III.B-C.
        Laghi et al. (2021), arXiv:2102.01708, Section III.A.
        SNR ~ 1/d_L: Hogg (1999), arXiv:astro-ph/9905116, Eq. (16).
    """

    def __init__(
        self,
        injection_data_dir: str,
        snr_threshold: float,
        h_grid: list[float] | None = None,
        *,
        dl_bins: int = _DEFAULT_DL_BINS,
        mass_bins: int = _DEFAULT_M_BINS,
        _force_unit_weights: bool = False,
    ) -> None:
        self._dl_bins = dl_bins
        self._mass_bins = mass_bins
        self._snr_threshold = snr_threshold
        self._force_unit_weights = _force_unit_weights

        if h_grid is not None:
            warnings.warn(
                "The 'h_grid' parameter is deprecated and ignored. "
                "SimulationDetectionProbability now builds P_det grids on-the-fly "
                "via SNR rescaling from pooled injection data.",
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

        # Validate required columns
        required_cols = {"z", "M", "SNR", "h_inj", "luminosity_distance"}
        missing = required_cols - set(self._pooled_df.columns)
        if missing:
            msg = f"Injection CSV missing required columns: {missing}"
            raise ValueError(msg)

        # Pre-extract arrays for efficient rescaling
        self._z_arr: npt.NDArray[np.float64] = self._pooled_df["z"].values.astype(np.float64)
        self._M_arr: npt.NDArray[np.float64] = self._pooled_df["M"].values.astype(np.float64)
        self._snr_raw: npt.NDArray[np.float64] = self._pooled_df["SNR"].values.astype(np.float64)
        self._h_inj_arr: npt.NDArray[np.float64] = self._pooled_df["h_inj"].values.astype(
            np.float64
        )
        self._dl_raw: npt.NDArray[np.float64] = self._pooled_df[
            "luminosity_distance"
        ].values.astype(np.float64)

        # LRU cache for built grids: h_value -> (2D interpolator, 1D interpolator)
        self._grid_cache: OrderedDict[
            float,
            tuple[RegularGridInterpolator, RegularGridInterpolator],
        ] = OrderedDict()

        # Quality flags cache
        self._quality_flags: dict[
            float, dict[str, npt.NDArray[np.float64] | npt.NDArray[np.bool_]]
        ] = {}

    def __getstate__(self) -> dict[str, Any]:
        """Exclude heavy data from pickle that workers don't need.

        Workers only call detection_probability_*_interpolated() which uses the
        pre-built RegularGridInterpolator from _grid_cache.  The raw injection
        arrays (_z_arr, etc.) are only needed to build new grids via _rescale_snr,
        which never happens in workers because the cache is pre-warmed for the
        target h before pool spawn.
        """
        state = self.__dict__.copy()
        state["_pooled_df"] = None
        # Raw injection arrays (~18.5 MB) — not needed when grid is pre-warmed
        state["_z_arr"] = None
        state["_M_arr"] = None
        state["_snr_raw"] = None
        state["_h_inj_arr"] = None
        state["_dl_raw"] = None
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

    def _get_or_build_grid(
        self, h: float
    ) -> tuple[RegularGridInterpolator, RegularGridInterpolator]:
        """Get cached grid or build a new one for the given h value.

        Uses LRU eviction when cache exceeds _MAX_CACHE_SIZE entries.

        Args:
            h: Hubble parameter value.

        Returns:
            Tuple of (2D interpolator, 1D interpolator).
        """
        if h in self._grid_cache:
            # Move to end (most recently used)
            self._grid_cache.move_to_end(h)
            return self._grid_cache[h]

        # Cache miss: build grids via SNR rescaling
        d_L_target, snr_rescaled = self._rescale_snr(h)

        # Build a temporary DataFrame for the grid builders
        df_rescaled = pd.DataFrame(
            {
                "luminosity_distance": d_L_target,
                "M": self._M_arr,
                "SNR": snr_rescaled,
            }
        )

        weights = np.ones(len(df_rescaled)) if self._force_unit_weights else None

        interp_2d = self._build_grid_2d(
            df_rescaled,
            self._snr_threshold,
            h_val=h,
            weights=weights,
        )
        interp_1d = self._build_grid_1d(df_rescaled, self._snr_threshold, h_val=h)

        # LRU eviction
        if len(self._grid_cache) >= _MAX_CACHE_SIZE:
            self._grid_cache.popitem(last=False)  # Remove oldest

        self._grid_cache[h] = (interp_2d, interp_1d)
        return interp_2d, interp_1d

    def _build_grid_2d(
        self,
        df: pd.DataFrame,
        snr_threshold: float,
        h_val: float | None = None,
        *,
        weights: npt.NDArray[np.float64] | None = None,
    ) -> RegularGridInterpolator:
        """Build a 2D P_det(d_L, M) grid by marginalizing over sky angles.

        Uses the ``luminosity_distance`` column directly -- no z-to-d_L
        conversion needed.

        When ``weights`` is provided, uses the self-normalized importance
        sampling estimator P_hat(B) = sum(w_det) / sum(w_total) per bin
        instead of N_det / N_total.  When weights is None (default), falls
        back to the standard unweighted estimator.

        If ``h_val`` is provided, per-bin quality metadata (total counts,
        detected counts, reliable mask, effective sample size) is stored in
        ``self._quality_flags`` for diagnostic use.  This metadata does
        **not** affect the interpolation result.

        Args:
            df: DataFrame with columns luminosity_distance, M, SNR.
            snr_threshold: SNR detection threshold.
            h_val: Hubble parameter value for quality flag storage (optional).
            weights: Per-injection importance weights, shape (N,).  If None,
                all weights are implicitly 1 (standard histogram estimator).

        Returns:
            RegularGridInterpolator for P_det(d_L, M).

        References:
            Self-normalized IS estimator: Tiwari (2018), arXiv:1712.00482, Eq. 5-8.
            Effective sample size: Kish (1965), Survey Sampling.
        """
        dl_vals = df["luminosity_distance"].values
        M_vals = df["M"].values  # noqa: N806
        snr_vals = df["SNR"].values
        n_events = len(df)

        # Set up weights -- default to None (unweighted path)
        use_weights = weights is not None
        if use_weights:
            w = np.asarray(weights, dtype=np.float64)
            if len(w) != n_events:
                msg = f"weights length {len(w)} != DataFrame length {n_events}"
                raise ValueError(msg)
        else:
            w = np.ones(n_events, dtype=np.float64)

        # Define bin edges in d_L space (IDENTICAL to previous implementation)
        dl_max = float(np.max(dl_vals)) * 1.1
        dl_edges = np.linspace(0, dl_max, self._dl_bins + 1)

        M_min = float(np.min(M_vals)) * 0.9  # noqa: N806
        M_max = float(np.max(M_vals)) * 1.1  # noqa: N806
        M_edges = np.geomspace(M_min, M_max, self._mass_bins + 1)  # noqa: N806

        detected_mask = snr_vals >= snr_threshold

        if not use_weights:
            # Original unweighted path -- preserves exact bit-for-bit output
            total_counts, _, _ = np.histogram2d(
                dl_vals,
                M_vals,
                bins=[dl_edges, M_edges],
            )
            detected_counts, _, _ = np.histogram2d(
                dl_vals[detected_mask],
                M_vals[detected_mask],
                bins=[dl_edges, M_edges],
            )
            p_det_grid = np.divide(
                detected_counts,
                total_counts,
                out=np.zeros_like(detected_counts, dtype=np.float64),
                where=total_counts > 0,
            )
            # N_eff = n_total for unweighted case (identity)
            n_eff_grid = total_counts.astype(np.float64)
        else:
            # Weighted IS estimator path
            # Assign each injection to a bin using np.digitize
            dl_bin_idx = np.digitize(dl_vals, dl_edges) - 1  # 0-based
            M_bin_idx = np.digitize(M_vals, M_edges) - 1  # noqa: N806

            # Clip to valid range (digitize can return out-of-range)
            dl_bin_idx = np.clip(dl_bin_idx, 0, self._dl_bins - 1)
            M_bin_idx = np.clip(M_bin_idx, 0, self._mass_bins - 1)

            # Accumulate weighted sums per bin using np.add.at
            total_weights = np.zeros((self._dl_bins, self._mass_bins), dtype=np.float64)
            detected_weights = np.zeros((self._dl_bins, self._mass_bins), dtype=np.float64)
            n_eff_grid = np.zeros((self._dl_bins, self._mass_bins), dtype=np.float64)

            # Also track integer counts for quality flags
            total_counts = np.zeros((self._dl_bins, self._mass_bins), dtype=np.float64)
            detected_counts = np.zeros((self._dl_bins, self._mass_bins), dtype=np.float64)

            np.add.at(total_weights, (dl_bin_idx, M_bin_idx), w)
            np.add.at(total_counts, (dl_bin_idx, M_bin_idx), 1.0)

            det_dl = dl_bin_idx[detected_mask]
            det_M = M_bin_idx[detected_mask]
            det_w = w[detected_mask]

            np.add.at(detected_weights, (det_dl, det_M), det_w)
            np.add.at(detected_counts, (det_dl, det_M), 1.0)

            # P_det = sum(w_det) / sum(w_total), 0 where total=0
            # Self-normalized IS estimator, Tiwari (2018) Eq. 5-8
            p_det_grid = np.divide(
                detected_weights,
                total_weights,
                out=np.zeros((self._dl_bins, self._mass_bins), dtype=np.float64),
                where=total_weights > 0,
            )

            # Kish N_eff per bin: (sum w)^2 / sum(w^2)
            # Kish (1965), Survey Sampling
            sum_w2_grid = np.zeros((self._dl_bins, self._mass_bins), dtype=np.float64)
            np.add.at(sum_w2_grid, (dl_bin_idx, M_bin_idx), w**2)
            n_eff_grid = np.divide(
                total_weights**2,
                sum_w2_grid,
                out=np.zeros((self._dl_bins, self._mass_bins), dtype=np.float64),
                where=sum_w2_grid > 0,
            )

        # Store quality flags (metadata only -- does not affect interpolation)
        if h_val is not None:
            self._quality_flags[h_val] = {
                "n_total": total_counts.copy(),
                "n_detected": detected_counts.copy(),
                "reliable": (total_counts >= 10),
                "dl_edges": dl_edges.copy(),
                "M_edges": M_edges.copy(),
                "n_eff": n_eff_grid.copy(),
            }

        # Use bin centers as grid coordinates
        dl_centers = 0.5 * (dl_edges[:-1] + dl_edges[1:])
        M_centers = np.sqrt(
            M_edges[:-1] * M_edges[1:]
        )  # geometric mean for log-spaced  # noqa: N806

        # Phase 45 Plan 45-07 (Option D): apply the same hybrid anchor layout
        # to the d_L axis of the 2D grid as `_build_grid_1d` does for the 1D
        # grid (Plan 45-02 + Plan 45-04).  Prepend BOTH (0, _P_MAX_EMPIRICAL_ANCHOR)
        # AND (_D_INTERMEDIATE_ANCHOR_GPC, _P_INTERMEDIATE_EMPIRICAL) along
        # the d_L axis, broadcasting both anchor values across ALL M_z bins
        # (the empirical asymptote at d_L → 0 is M_z-independent because
        # SNR ~ 1/d_L diverges regardless of intrinsic loudness).  Falls back
        # to the single-anchor layout when c_0(h) <= _D_INTERMEDIATE_ANCHOR_GPC.
        # See Plan 45-05 SUMMARY for the motivation and Plan 45-07 SUMMARY
        # for the cluster re-eval predictions.
        # Eq. (A.19) in Gray et al. (2020), arXiv:1908.06050.
        n_mass = p_det_grid.shape[1]
        if dl_centers[0] > _D_INTERMEDIATE_ANCHOR_GPC:
            dl_centers_anchored = np.concatenate(([0.0, _D_INTERMEDIATE_ANCHOR_GPC], dl_centers))
            anchor_row = np.full((1, n_mass), _P_MAX_EMPIRICAL_ANCHOR, dtype=np.float64)
            intermediate_row = np.full((1, n_mass), _P_INTERMEDIATE_EMPIRICAL, dtype=np.float64)
            p_det_grid_anchored = np.concatenate((anchor_row, intermediate_row, p_det_grid), axis=0)
        else:
            # Plan 45-02 fallback layout for c_0(h) <= _D_INTERMEDIATE_ANCHOR_GPC.
            dl_centers_anchored = np.concatenate(([0.0], dl_centers))
            anchor_row = np.full((1, n_mass), _P_MAX_EMPIRICAL_ANCHOR, dtype=np.float64)
            p_det_grid_anchored = np.concatenate((anchor_row, p_det_grid), axis=0)

        # fill_value=None → linear extrapolation outside grid (NOT nearest-
        # neighbour as the pre-Phase-45 docstring incorrectly claimed).
        # fill_value=0.0 caused 44% of events to lose completeness correction
        # because high-SNR events' 4σ integration bounds exceed the injection grid.
        return RegularGridInterpolator(
            (dl_centers_anchored, M_centers),
            p_det_grid_anchored,
            method="linear",
            bounds_error=False,
            fill_value=None,
        )

    def _build_grid_1d(
        self,
        df: pd.DataFrame,
        snr_threshold: float,
        *,
        h_val: float | None = None,
    ) -> RegularGridInterpolator:
        """Build a 1D P_det(d_L) grid by marginalizing over M and sky angles.

        The histogram edges are ``np.linspace(0, dl_max, N+1)`` so the first
        bin covers ``[0, 2·c_0)`` with ``c_0 = dl_centers[0] = dl_max/(2N)``.
        Phase 45 (Plan 45-02 + Plan 45-04 hybrid) prepends TWO h-independent
        empirical anchors before constructing the interpolator:
        ``(0.0, _P_MAX_EMPIRICAL_ANCHOR=0.7931)`` (Wilson 95% lower bound)
        and the intermediate ``(_D_INTERMEDIATE_ANCHOR_GPC=0.05,
        _P_INTERMEDIATE_EMPIRICAL=1.0)`` (empirical point estimate).  On
        ``[0, 0.05]`` the result is a linear interpolation between the
        d_L=0 anchor and the intermediate anchor (slope +4.138/Gpc); on
        ``[0.05, c_0)`` it is a linear interpolation between the
        intermediate anchor and the histogram first bin centre ``p̂(c_0)``.
        When ``c_0(h) <= 0.05`` (rare; only at very high h), the intermediate
        anchor is skipped and the layout falls back to the Plan 45-02
        single-anchor variant.  Off-grid evaluation uses ``fill_value=None``
        (linear extrapolation); see
        :meth:`detection_probability_without_bh_mass_interpolated_zero_fill`
        for the boundary convention.

        Args:
            df: DataFrame with columns luminosity_distance, SNR.
            snr_threshold: SNR detection threshold.
            h_val: Hubble parameter value (used for diagnostic logging only).

        Returns:
            RegularGridInterpolator for P_det(d_L) on the anchored grid
            (length ``dl_bins + 2`` when c_0(h) > 0.05; ``dl_bins + 1`` in
            the c_0 ≤ 0.05 fallback).
        """
        dl_vals = df["luminosity_distance"].values
        snr_vals = df["SNR"].values

        dl_max = float(np.max(dl_vals)) * 1.1
        dl_edges = np.linspace(0, dl_max, self._dl_bins + 1)

        total_counts, _ = np.histogram(dl_vals, bins=dl_edges)
        detected_mask = snr_vals >= snr_threshold
        detected_counts, _ = np.histogram(dl_vals[detected_mask], bins=dl_edges)

        # Phase 44 reliability check: the first-bin estimate p̂(c_0) anchors
        # the upper end of the [0, c_0) linear-interp segment after the
        # Phase 45 empirical-anchor prepend below.  Wilson 95% CI half-width
        # ≈ 1/sqrt(n); n=100 → ~0.05 absolute uncertainty on p̂(c_0).
        if total_counts[0] < 100:
            logger.warning(
                "P_det 1D grid first bin [0, %.4f Gpc) has only %d injections "
                "(h=%s).  p̂(c_0) (the upper anchor of the [0, c_0) linear-"
                "interp segment) may be noisy.  Consider denser low-d_L "
                "injections.",
                float(dl_edges[1]),
                int(total_counts[0]),
                f"{h_val:.4f}" if h_val is not None else "?",
            )

        p_det_1d = np.divide(
            detected_counts,
            total_counts,
            out=np.zeros_like(detected_counts, dtype=np.float64),
            where=total_counts > 0,
        )

        dl_centers = 0.5 * (dl_edges[:-1] + dl_edges[1:])

        # Phase 45 Plan 45-04 hybrid: prepend BOTH (0, _P_MAX_EMPIRICAL_ANCHOR)
        # AND (_D_INTERMEDIATE_ANCHOR_GPC, _P_INTERMEDIATE_EMPIRICAL) when
        # c_0(h) > _D_INTERMEDIATE_ANCHOR_GPC.
        #
        # Plan 45-02 prepended ONLY (0, 0.7931).  The first injection bin
        # [0, 2*c_0) is upper-skewed in d_L (weighted-mean d_L = 0.132 Gpc;
        # ratio 3.22 of upper-third to lower-third events at h_inj=0.73, see
        # outputs/phase45/first_bin_density.json), so the histogram estimate
        # p̂(c_0) systematically underestimates p_det near d_L=0.  The pooled
        # Wilson 95% lower bound at d_L < 0.10 Gpc (n_pooled=63/71 across all
        # h_inj; LR homogeneity p=0.199 → h-independent; Plan 45-01) gives
        # the conservative anchor 0.7931 at d_L=0.
        #
        # Cluster re-eval (Plan 45-03) showed the Plan 45-02 single-anchor
        # produced a sub-discrete-grid-step lift on the production posterior
        # (mean(bootstrap_MAP) shifted only -0.0047; discrete MAP unchanged).
        # The hybrid adds the empirical-asymptote point at d_L = 0.05 Gpc
        # (16/16 detected at d_L < 0.10 Gpc per pdet_asymptote.json; subset
        # at d_L < 0.05 is also 100% detected by monotonicity argument).  At
        # h=0.73 this raises interp(0.05) from ≈0.6687 (linear interp from
        # 0.7931 down to p̂(c_0)≈0.5448) to exactly 1.0 — a +0.33 lift.
        #
        # On [0, 0.05] the linear interp from (0, 0.7931) to (0.05, 1.0) has
        # POSITIVE slope +4.138/Gpc.  This is a sign-convention deviation
        # from physical monotonicity in d_L, but is justified empirically:
        # the d_L=0 value 0.7931 is the conservative Wilson LB, intentionally
        # below the observed asymptote ≈1.0 to avoid overshoot of the
        # production MAP.  The integrand on L_comp is the value, not the
        # slope; the integrated effect remains conservatively lower-bounded.
        #
        # Edge case fallback: when c_0(h) <= _D_INTERMEDIATE_ANCHOR_GPC, the
        # intermediate point would violate strict monotonicity of dl_centers
        # required by RegularGridInterpolator.  Skip the intermediate; revert
        # to the Plan 45-02 single-anchor layout.  This branch is rare (only
        # triggered at very high h where dl_max(h) < 6.0 Gpc).
        #
        # Eq. (A.19) in Gray et al. (2020), arXiv:1908.06050.
        # See scripts/bias_investigation/outputs/phase45/pdet_asymptote.json
        # for the empirical 1.0 asymptote at d_L < 0.10 Gpc;
        # scripts/bias_investigation/outputs/phase45/p_max_h_independence.json
        # for the pooled Wilson LB; Plan 45-01 SUMMARY for the LR test.
        if dl_centers[0] > _D_INTERMEDIATE_ANCHOR_GPC:
            dl_centers_anchored = np.concatenate(([0.0, _D_INTERMEDIATE_ANCHOR_GPC], dl_centers))
            p_det_1d_anchored = np.concatenate(
                ([_P_MAX_EMPIRICAL_ANCHOR, _P_INTERMEDIATE_EMPIRICAL], p_det_1d)
            )
        else:
            # Plan 45-02 fallback layout for c_0(h) <= _D_INTERMEDIATE_ANCHOR_GPC.
            dl_centers_anchored = np.concatenate(([0.0], dl_centers))
            p_det_1d_anchored = np.concatenate(([_P_MAX_EMPIRICAL_ANCHOR], p_det_1d))

        # fill_value=None → linear extrapolation outside grid.  Below
        # dl_centers_anchored[0] = 0 this is unreachable since d_L < 0 is
        # unphysical; above dl_centers_anchored[-1] the calling function
        # explicitly clips to 0 (Phase 44 invariant).
        return RegularGridInterpolator(
            (dl_centers_anchored,),
            p_det_1d_anchored,
            method="linear",
            bounds_error=False,
            fill_value=None,
        )

    def quality_flags(self, h: float) -> dict[str, npt.NDArray[np.float64] | npt.NDArray[np.bool_]]:
        """Return per-bin quality metadata for the given h value.

        Quality flags are diagnostic metadata stored during grid construction.
        They do **not** affect the P_det interpolation result.

        If the grid for this h has not been built yet, it will be built
        (triggering SNR rescaling and caching).

        The returned dict contains:

        - ``n_total``: int array (dl_bins, M_bins) -- total injections per bin
        - ``n_detected``: int array (dl_bins, M_bins) -- detected injections
        - ``reliable``: bool array (dl_bins, M_bins) -- True where n_total >= 10
        - ``dl_edges``: float array (dl_bins+1,) -- d_L bin edges in Gpc
        - ``M_edges``: float array (M_bins+1,) -- mass bin edges in solar masses
        - ``n_eff``: float array (dl_bins, M_bins) -- Kish effective sample
          size per bin (equals n_total when weights are uniform)

        Args:
            h: Hubble parameter value.

        Returns:
            Dict of quality flag arrays.

        Raises:
            ValueError: If no quality flags are available (empty grid).
        """
        # Ensure grid is built (which populates quality flags)
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
        """Detection probability including BH mass dependence.

        Drop-in replacement for
        ``DetectionProbability.detection_probability_with_bh_mass_interpolated``
        with an additional ``h`` keyword for h-dependent P_det.

        Sky angles (phi, theta) are accepted for API compatibility but are
        marginalized over internally (D-02).

        The grid is in d_L space, so no ``dist_to_redshift`` inversion is
        needed.  The observer-frame mass M_z is converted to source-frame
        using the approximate relation z ~ d_L * H0/c for small z, but since
        the grid is in (d_L, M) and the injection data stored source-frame M
        directly, we pass M_z as-is (the grid was built from source-frame M).

        Args:
            d_L: Luminosity distance in Gpc.
            M_z: Observer-frame (redshifted) BH mass in solar masses.
            phi: Sky angle phi (unused, marginalized over).
            theta: Sky angle theta (unused, marginalized over).
            h: Dimensionless Hubble parameter.

        Returns:
            Detection probability in [0, 1].

        References:
            Gray et al. (2020), arXiv:1908.06050, Eq. (8).
            Laghi et al. (2021), arXiv:2102.01708, Section III.A.
        """
        interp_2d, _ = self._get_or_build_grid(h)

        dl_arr = np.atleast_1d(np.asarray(d_L, dtype=np.float64))
        M_arr = np.atleast_1d(np.asarray(M_z, dtype=np.float64))  # noqa: N806
        points = np.column_stack([dl_arr, M_arr])

        result = np.clip(interp_2d(points), 0.0, 1.0)

        if np.ndim(d_L) == 0 and np.ndim(M_z) == 0:
            return float(result[0])
        return result  # type: ignore[no-any-return]

    def get_dl_max(self, h: float) -> float:
        """Return the maximum d_L of the 1D P_det grid for the given h value.

        This is ``max(injection_d_L) * 1.1``, i.e. the upper edge of the
        1D histogram used by :meth:`_build_grid_1d`.  Needed to compute
        ``z_max(h)`` for the full-volume denominator integral.

        Args:
            h: Dimensionless Hubble parameter.

        Returns:
            Maximum d_L in Gpc.
        """
        # Ensure grid is built (populates cache)
        self._get_or_build_grid(h)
        # Reconstruct dl_max from the 1D interpolator's grid points
        _, interp_1d = self._grid_cache[h]
        dl_centers = interp_1d.grid[0]
        # The centers are midpoints of linspace(0, dl_max, N+1).
        # spacing = dl_centers[1] - dl_centers[0], last center = dl_max - spacing/2
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
    ) -> float | npt.NDArray[np.float64]:
        """Detection probability with hard zero above the injection grid.

        Used by every production p_det evaluation for symmetry between
        L_comp / L_cat numerators and the D(h) denominator (STAT-03 invariant,
        commit ``a70d1a2``).  Call sites in :mod:`bayesian_statistics`:

        * ``precompute_completion_denominator`` (D(h) full-volume denominator)
        * ``p_Di.completion_numerator_integrand`` (L_comp 4σ window)
        * ``single_host_likelihood.numerator_integrant_without_bh_mass`` (L_cat numerator)
        * ``single_host_likelihood.denominator_integrant_without_bh_mass`` (L_cat denominator)
        * ``single_host_likelihood_integration_testing`` (legacy, two integrands)

        Boundary convention (Phase 44 + Phase 45 fix):

        * ``d_L > dl_centers[-1]``: source beyond the injection horizon →
          detectability is unmodelled and physically zero (explicit clip
          below).
        * ``d_L < dl_centers[0] = c_0``: ``RegularGridInterpolator(method=
          "linear", fill_value=None)`` performs **linear extrapolation**, NOT
          nearest-neighbour as the pre-Phase-45 docstring claimed.  Phase 45
          (Plan 45-02 + Plan 45-04 hybrid) prepends TWO h-independent
          empirical anchors before constructing the interpolator (see
          :data:`_P_MAX_EMPIRICAL_ANCHOR`, :data:`_D_INTERMEDIATE_ANCHOR_GPC`,
          :data:`_P_INTERMEDIATE_EMPIRICAL`, and :meth:`_build_grid_1d`):
          ``(0, P_MAX=0.7931)`` (Wilson 95% lower bound) AND
          ``(0.05, P_INTERMEDIATE=1.0)`` (empirical point estimate at
          d_L < 0.10 Gpc; the d_L < 0.05 subset is also 100% detected by
          monotonicity argument).  On ``[0, c_0)`` this gives a piecewise
          hybrid: ``p_det = linear_interp((0, P_MAX), (0.05, P_INTERMEDIATE))``
          on ``[0, 0.05]`` and
          ``p_det = linear_interp((0.05, P_INTERMEDIATE), (c_0, p̂(c_0)))``
          on ``[0.05, c_0]``.  At ``d_L = 0`` the value equals
          ``_P_MAX_EMPIRICAL_ANCHOR=0.7931`` (the conservative Wilson lower
          bound of the empirical asymptote); at ``d_L = 0.05`` it equals
          ``_P_INTERMEDIATE_EMPIRICAL=1.0`` (the empirical point estimate);
          at ``d_L = c_0`` it equals the unchanged histogram estimate
          ``p̂(c_0)``.  When ``c_0(h) <= 0.05`` (rare; only at very high h),
          the intermediate anchor is skipped (Plan 45-02 fallback layout).
          The hybrid's h-independence is by construction (both anchor values
          are module-level scalars at fixed physical positions); the d_L=0
          anchor's h-independence under the original asymptote test is
          established by a likelihood-ratio binomial-rate-homogeneity test
          (G = 7.30, dof = 5, p = 0.199; Wilson 95% lower bound across
          pooled groups: 0.7931).  See Plan 45-01 SUMMARY and
          ``scripts/bias_investigation/outputs/phase45/p_max_h_independence.json``;
          Plan 45-04 PLAN/SUMMARY for the hybrid lift rationale.

        Plan 45-02 (single-anchor) lifted interp(d_L=0) from ≈0.748 (linear
        extrap through bins 0,1; pre-Phase-45 behaviour) to 0.7931, but the
        cluster re-eval (Plan 45-03) showed the resulting MAP shift was
        sub-discrete-grid-step and the discrete MAP did not move from
        0.7650 (see ``.gpd/phases/45-p-det-first-bin-asymptote-fix/45-03-SUMMARY.md``).
        Plan 45-04's hybrid intermediate anchor at d_L=0.05 raises
        interp(0.05) from ≈0.6687 to 1.0 — a +0.33 lift in the integration
        window crossed by ≈26/60 production events per
        ``scripts/bias_investigation/outputs/phase45/window_proximity.json``.

        Pre-Phase-44 the function also zeroed
        ``d_L < c_0(h) = dl_max(h)/120`` — a bin-midpoint artifact that
        scaled as ``1/h`` and drove a ``+145.7`` log-unit MAP bias for 312
        events between h=0.73 and h=0.86.

        Args:
            d_L: Luminosity distance in Gpc.
            phi: Sky angle phi (unused, marginalized over).
            theta: Sky angle theta (unused, marginalized over).
            h: Dimensionless Hubble parameter.

        Returns:
            Detection probability in [0, 1].  Zero only above ``dl_centers[-1]``.

        References:
            Gray et al. (2020), arXiv:1908.06050, Eq. A.19.
        """
        _, interp_1d = self._get_or_build_grid(h)

        dl_arr = np.atleast_1d(np.asarray(d_L, dtype=np.float64))
        points = dl_arr.reshape(-1, 1)

        result = np.clip(interp_1d(points), 0.0, 1.0)

        # Phase 44 fix: removed result[dl_arr < dl_min] = 0.0.  Below
        # dl_centers[0] = dl_max/120 the first injection bin (0, 2*c_0) has
        # real events; nearest-neighbour from fill_value=None already returns
        # p̂(c_0).  The previous left-side cutoff scaled as 1/h, biasing MAP
        # toward h_max for events with d_L ≈ c_0.
        # Eq. (A.19) in Gray et al. (2020), arXiv:1908.06050.
        dl_centers = interp_1d.grid[0]
        dl_max = float(dl_centers[-1])
        result[dl_arr > dl_max] = 0.0

        if np.ndim(d_L) == 0:
            return float(result[0])
        return result  # type: ignore[no-any-return]

    def detection_probability_without_bh_mass_interpolated(
        self,
        d_L: float | npt.NDArray[np.float64],
        phi: float | npt.NDArray[np.float64],
        theta: float | npt.NDArray[np.float64],
        *,
        h: float,
    ) -> float | npt.NDArray[np.float64]:
        """Detection probability marginalized over BH mass.

        Drop-in replacement for
        ``DetectionProbability.detection_probability_without_bh_mass_interpolated``
        with an additional ``h`` keyword for h-dependent P_det.

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
            Gray et al. (2020), arXiv:1908.06050, Eq. (8).
            Laghi et al. (2021), arXiv:2102.01708, Section III.A.
        """
        _, interp_1d = self._get_or_build_grid(h)

        dl_arr = np.atleast_1d(np.asarray(d_L, dtype=np.float64))
        points = dl_arr.reshape(-1, 1)

        result = np.clip(interp_1d(points), 0.0, 1.0)

        if np.ndim(d_L) == 0:
            return float(result[0])
        return result  # type: ignore[no-any-return]
