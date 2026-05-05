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
# Out-of-grid detection-probability behavior (1D and 2D channels).
#
# As of 2026-05-05, both channels use a principled monotonic-asymptotic
# extrapolation scheme implemented inside the corresponding interpolated
# probability functions:
#
# * Saturating face (d_L < d_L_min): linear bridge from (d_L_min, p_edge)
#   to (0, 1).  The d_L=0 limit is the unique natural physical scale
#   where the asymptote p_det=1 is exact (no source closer than the
#   observer).
#
# * Suppressing faces (d_L > d_L_max; M_z out of bounds in the 2D case):
#   slope-matched linear extrapolation from the boundary, clamped to
#   [0, p_edge] (Option A directional clamp).
#
# * Corner cells (2D, both axes outside): min of the two face
#   extrapolations.
#
# The scheme replaces the Phase 45 Plan 45-02/04 anchor approach (Wilson
# 95% LB at d_L=0 plus an empirical point estimate at d_L=0.05), which
# was fitted to production-posterior behavior and incompatible with the
# project's principled-physics preference.  See
# `.planning/2D-CHANNEL-AUDIT-20260505.md` for the full audit and
# rationale.
# ----------------------------------------------------------------------


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

        # fill_value=None → nearest-neighbor extrapolation outside grid.
        # fill_value=0.0 caused 44% of events to lose completeness correction
        # because high-SNR events' 4σ integration bounds exceed the injection grid.
        return RegularGridInterpolator(
            (dl_centers, M_centers),
            p_det_grid,
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
        The grid is the raw histogram with bin centers in d_L; off-grid
        extrapolation behaviour is implemented in
        :meth:`detection_probability_without_bh_mass_interpolated_zero_fill`
        as a principled scheme (linear bridge to the saturated asymptote
        at d_L=0, slope-matched linear extrapolation toward 0 above
        d_L_max), aligned with the 2D channel.

        Args:
            df: DataFrame with columns luminosity_distance, SNR.
            snr_threshold: SNR detection threshold.
            h_val: Hubble parameter value (used for diagnostic logging only).

        Returns:
            RegularGridInterpolator for P_det(d_L) on the histogram bin
            centers (length ``dl_bins``).
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

        # Eq. (A.19) in Gray et al. (2020), arXiv:1908.06050.
        # No anchors are prepended: out-of-grid extrapolation is the
        # responsibility of
        # :meth:`detection_probability_without_bh_mass_interpolated_zero_fill`,
        # which applies a principled scheme (linear bridge to the saturated
        # asymptote at d_L=0; slope-matched linear extrapolation toward 0
        # above d_L_max).  Replaced the Plan 45-02/04 anchor scheme on
        # 2026-05-05 — see ``.planning/2D-CHANNEL-AUDIT-20260505.md``.
        return RegularGridInterpolator(
            (dl_centers,),
            p_det_1d,
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

        Sky angles (phi, theta) are accepted for API compatibility but are
        marginalized over internally (D-02).

        The grid is in (d_L, M) space, so no ``dist_to_redshift`` inversion is
        needed.  The observer-frame mass M_z is queried as-is (the grid was
        built from source-frame M; this is a known approximation, see
        :class:`SimulationDetectionProbability` docstring).

        **Out-of-grid policy: principled monotonic-asymptotic extrapolation.**

        For queries inside the (d_L, M) grid, returns scipy's bilinear
        interpolation (clipped to [0, 1]).  For queries outside the grid,
        applies a slope-matched linear extrapolation from the nearest face,
        clamped per the physical asymptote table:

        +------------------+--------------+--------------------------------+
        | Direction        | Asymptote    | Reason                         |
        +==================+==============+================================+
        | d_L > d_L_max    | 0            | SNR-suppressed (sources too    |
        |                  |              | distant for SNR ≥ threshold)   |
        +------------------+--------------+--------------------------------+
        | d_L < d_L_min    | 1            | SNR-saturated (very nearby     |
        |                  |              | sources detected with prob 1)  |
        +------------------+--------------+--------------------------------+
        | M_z > M_z_max    | 0            | EMRI rate / waveform model     |
        |                  |              | breakdown above the EMRI mass  |
        +------------------+--------------+--------------------------------+
        | M_z < M_z_min    | 0            | SNR-suppressed and/or rate     |
        |                  |              | cutoff at low M                |
        +------------------+--------------+--------------------------------+
        | corner (both     | min of the   | "Either axis suppresses" — the |
        | axes outside)    | two faces    | only saturating face is        |
        |                  |              | (d_L<min, M in grid), which is |
        |                  |              | NOT a corner                   |
        +------------------+--------------+--------------------------------+

        **Construction.**  Two cases.

        *Saturating face (d_L<d_L_min):* there is a unique natural scale
        d_L=0 where the asymptote p_det=1 is exact (no source closer than
        the observer).  Linearly interpolate from (dl_min, p_edge) to
        (0, 1) — C0 continuous at dl_min, reaches the asymptote at the
        natural physical boundary, and uses no fitted parameters or noisy
        boundary slope estimates.  Explicitly:
        ``p(dl) = 1 - (1 - p_edge) * (dl / dl_min)`` for ``dl ∈ [0, dl_min]``.

        *Suppressing faces (d_L>d_L_max, M_z>M_max, M_z<M_min):* no analogous
        finite asymptote scale exists (suppression is at infinity).  Use
        slope-matched linear extrapolation from the boundary, computed
        from the last two grid centers along the relevant axis with the
        slope evaluated at the projected position on the other axis.
        ``p_extrap = p_edge + slope · (query - edge)``, clamped to
        ``[0, p_edge]`` (Option A: never overshoots boundary value, floor
        at the asymptote 0).

        Corner cells (both axes outside) take the ``min`` of the two face
        extrapolations.  The only saturating face is (d_L<d_L_min,
        M in grid), which is not a corner; all true corners involve at
        least one suppressing axis, so ``min`` is monotone-correct.

        **Properties.**

        * **C0 continuous at the boundary** by construction (extrapolation
          formula evaluates to p_edge at the boundary, matching the in-grid
          interpolation).  Removes the discontinuity that previously
          drove spurious h_trial-dependence in the joint posterior as
          hosts crossed the grid boundary at small Δh.
        * **Bounded in [0, 1]** by construction (clamp + asymptote table).
          The previous policy (raw scipy linear extrapolation + clip) could
          drift to negative values at the d_L→0 boundary due to KDE noise
          in the low-d_L bins (only ~7 injections per bin in production),
          systematically returning ≈0 instead of ≈1 for 6–12% of events.
          See ``.planning/2D-CHANNEL-AUDIT-20260505.md`` Step 1b for the
          quantitative diagnostic.
        * **Slope from the simulated KDE** (no fitted anchor): the
          boundary slope is the same one scipy would use for its own
          linear extrapolation; the difference from the prior policy is
          the directional clamp (Option A) that prevents wrong-direction
          KDE noise from running into the wrong asymptote.

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
            Maggiore (2008), Gravitational Waves Vol 1 §7.7 (SNR scaling
            for inspirals → monotonicity argument).
            Babak et al. (2017), arXiv:1703.09722 §III (EMRI rate density
            with high/low M cutoffs → asymptote table).
        """
        interp_2d, _ = self._get_or_build_grid(h)
        dl_centers = np.asarray(interp_2d.grid[0])
        M_centers = np.asarray(interp_2d.grid[1])  # noqa: N806
        p_grid = np.asarray(interp_2d.values, dtype=np.float64)

        dl_arr = np.atleast_1d(np.asarray(d_L, dtype=np.float64))
        M_arr = np.atleast_1d(np.asarray(M_z, dtype=np.float64))  # noqa: N806

        dl_min = float(dl_centers[0])
        dl_max = float(dl_centers[-1])
        M_min = float(M_centers[0])  # noqa: N806
        M_max = float(M_centers[-1])  # noqa: N806

        # Project queries onto the in-grid box, then evaluate the in-grid
        # interpolator at the projected point — this is p_edge for face/corner
        # cells, and the in-grid value for in-grid cells.
        dl_clamp = np.clip(dl_arr, dl_min, dl_max)
        M_clamp = np.clip(M_arr, M_min, M_max)  # noqa: N806
        p_edge = np.clip(interp_2d(np.column_stack([dl_clamp, M_clamp])), 0.0, 1.0)

        # Start from p_edge; overlay face extrapolations where applicable.
        # In-grid cells keep p_edge as their final value (no face triggers).
        result = p_edge.copy()

        # ---- d_L < d_L_min face (saturating direction; asymptote 1) ----
        # The d_L=0 limit is the unique natural physical scale where the
        # asymptote p_det=1 is exact (closest possible source).  Use a
        # linear bridge from (dl_min, p_edge) to (0, 1) — guaranteed to
        # be C0 continuous at dl_min (matches p_edge by construction) and
        # to reach the asymptote at dl=0.  This deliberately ignores the
        # boundary KDE slope, which is unreliable in the first d_L bin
        # (~7 injections in production; warned by ``_build_grid_1d``).
        # Using the KDE slope here would let counting noise drive the
        # extrapolation; the bridge is the same scheme as the boundary,
        # extended monotonically to the natural asymptote location.
        out_dl_low = dl_arr < dl_min
        if out_dl_low.any():
            idx = np.where(out_dl_low)[0]
            # Bridge: p(dl) = p_edge + (1 - p_edge) * (dl_min - dl) / dl_min
            #              = 1 - (1 - p_edge) * dl / dl_min
            # At dl=dl_min: p = p_edge (C0).  At dl=0: p = 1 (asymptote).
            p_bridge = 1.0 - (1.0 - p_edge[idx]) * (dl_arr[idx] / dl_min)
            # Clamp into [p_edge, 1] for queries with dl < 0 (defensive).
            result[idx] = np.clip(p_bridge, p_edge[idx], 1.0)

        # ---- d_L > d_L_max face (suppressing direction; asymptote 0) ----
        out_dl_high = dl_arr > dl_max
        if out_dl_high.any():
            idx = np.where(out_dl_high)[0]
            slope_row = (p_grid[-1, :] - p_grid[-2, :]) / (dl_centers[-1] - dl_centers[-2])
            slope_at_query = np.interp(M_clamp[idx], M_centers, slope_row)
            delta = dl_arr[idx] - dl_max  # positive
            p_extrap = p_edge[idx] + slope_at_query * delta
            # Option A: floor at 0, ceiling at p_edge (asymptote 0).
            result[idx] = np.clip(p_extrap, 0.0, p_edge[idx])

        # ---- M < M_min face (suppressing; asymptote 0) ----
        out_M_low = M_arr < M_min  # noqa: N806
        if out_M_low.any():
            idx = np.where(out_M_low)[0]
            slope_col = (p_grid[:, 0] - p_grid[:, 1]) / (M_centers[0] - M_centers[1])
            slope_at_query = np.interp(dl_clamp[idx], dl_centers, slope_col)
            delta = M_arr[idx] - M_min  # negative
            p_extrap = p_edge[idx] + slope_at_query * delta
            face_M_low = np.clip(p_extrap, 0.0, p_edge[idx])
            # Corner rule: min of the per-face values for cells outside on
            # both axes.
            result[idx] = np.minimum(result[idx], face_M_low)

        # ---- M > M_max face (suppressing; asymptote 0) ----
        out_M_high = M_arr > M_max  # noqa: N806
        if out_M_high.any():
            idx = np.where(out_M_high)[0]
            slope_col = (p_grid[:, -1] - p_grid[:, -2]) / (M_centers[-1] - M_centers[-2])
            slope_at_query = np.interp(dl_clamp[idx], dl_centers, slope_col)
            delta = M_arr[idx] - M_max  # positive
            p_extrap = p_edge[idx] + slope_at_query * delta
            face_M_high = np.clip(p_extrap, 0.0, p_edge[idx])
            result[idx] = np.minimum(result[idx], face_M_high)

        # Final safety clip (already enforced face-by-face but be defensive).
        result = np.clip(result, 0.0, 1.0)

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
        """Detection probability with principled out-of-grid extrapolation.

        Mirror of :meth:`detection_probability_with_bh_mass_interpolated`
        (2D channel) specialized to the 1D (d_L only) grid.  Call sites in
        :mod:`bayesian_statistics`:

        * ``precompute_completion_denominator`` (D(h) full-volume denominator)
        * ``p_Di.completion_numerator_integrand`` (L_comp 4σ window)
        * ``single_host_likelihood.numerator_integrant_without_bh_mass`` (L_cat numerator)
        * ``single_host_likelihood.denominator_integrant_without_bh_mass`` (L_cat denominator)
        * ``single_host_likelihood_integration_testing`` (legacy, two integrands)

        **Out-of-grid policy: principled monotonic-asymptotic extrapolation.**

        +------------------+--------------+--------------------------------+
        | Direction        | Asymptote    | Reason                         |
        +==================+==============+================================+
        | d_L > d_L_max    | 0            | SNR-suppressed (sources too    |
        |                  |              | distant for SNR ≥ threshold)   |
        +------------------+--------------+--------------------------------+
        | d_L < d_L_min    | 1            | SNR-saturated (very nearby     |
        |                  |              | sources detected with prob 1)  |
        +------------------+--------------+--------------------------------+

        **Construction.**

        * **Saturating face (d_L < d_L_min):** linear bridge from
          ``(d_L_min, p_edge)`` to ``(0, 1)``.  The d_L=0 limit is the
          natural physical scale where the asymptote p_det=1 is exact (no
          source closer than the observer).  Explicitly:
          ``p(dl) = 1 - (1 - p_edge) * (dl / d_L_min)`` for
          ``dl ∈ [0, d_L_min]``.  C0 continuous at d_L_min by construction;
          reaches the asymptote 1 at d_L=0 by construction.  Uses no
          fitted constants and no boundary KDE slope (the first bin has
          ~7 injections in production; the bridge construction is
          insensitive to that noise).

        * **Suppressing face (d_L > d_L_max):** slope-matched linear
          extrapolation from the boundary, computed from the last two grid
          centers along d_L.  ``p_extrap = p_edge + slope · (query - edge)``,
          clamped to ``[0, p_edge]`` (Option A: never exceeds boundary,
          asymptotic floor at 0).

        **Why this replaces the Phase 45 anchor scheme (2026-05-05):**
        the previous scheme prepended two empirical anchors at d_L=0
        (Wilson 95% LB = 0.7931) and d_L=0.05 (empirical point estimate =
        1.0).  The Wilson LB was deliberately chosen to "not overshoot
        truth on production posteriors" — fitted to truth, against the
        project's principled-modeling preference.  The augmented Phase 46
        injection campaign now gives p̂(c_0) ≈ 1.0 at the first bin, so
        the Wilson anchor is actively *suppressing* the empirical 1.0 down
        to 0.7931 — the opposite of its original lift purpose.  The bridge
        construction here is the same scheme as the boundary
        (linear, C0) extended to the natural asymptote location; no fit,
        no anchor.  See ``.planning/2D-CHANNEL-AUDIT-20260505.md`` for the
        full rationale and the 2D-channel sibling that motivated this
        alignment.

        Args:
            d_L: Luminosity distance in Gpc.
            phi: Sky angle phi (unused, marginalized over).
            theta: Sky angle theta (unused, marginalized over).
            h: Dimensionless Hubble parameter.

        Returns:
            Detection probability in [0, 1].

        References:
            Gray et al. (2020), arXiv:1908.06050, Eq. (A.19).
            Maggiore (2008), Gravitational Waves Vol 1 §7.7 (SNR scaling
            for inspirals → monotonicity argument).
        """
        _, interp_1d = self._get_or_build_grid(h)
        dl_centers = np.asarray(interp_1d.grid[0])
        p_grid = np.asarray(interp_1d.values, dtype=np.float64)

        dl_arr = np.atleast_1d(np.asarray(d_L, dtype=np.float64))
        dl_min = float(dl_centers[0])
        dl_max = float(dl_centers[-1])

        # Project to in-grid; evaluate at projected point → p_edge.
        dl_clamp = np.clip(dl_arr, dl_min, dl_max)
        p_edge = np.clip(interp_1d(dl_clamp.reshape(-1, 1)), 0.0, 1.0)
        result = p_edge.copy()

        # ---- d_L < d_L_min face (saturating; asymptote 1) ----
        # Linear bridge from (dl_min, p_edge) to (0, 1).  At dl=dl_min:
        # p = p_edge (C0); at dl=0: p = 1 (asymptote).
        out_low = dl_arr < dl_min
        if out_low.any():
            idx = np.where(out_low)[0]
            p_bridge = 1.0 - (1.0 - p_edge[idx]) * (dl_arr[idx] / dl_min)
            result[idx] = np.clip(p_bridge, p_edge[idx], 1.0)

        # ---- d_L > d_L_max face (suppressing; asymptote 0) ----
        # Slope-matched linear extrapolation, clamped to [0, p_edge].
        out_high = dl_arr > dl_max
        if out_high.any():
            idx = np.where(out_high)[0]
            slope = (p_grid[-1] - p_grid[-2]) / (dl_centers[-1] - dl_centers[-2])
            delta = dl_arr[idx] - dl_max  # positive
            p_extrap = p_edge[idx] + slope * delta
            result[idx] = np.clip(p_extrap, 0.0, p_edge[idx])

        # Final safety clip.
        result = np.clip(result, 0.0, 1.0)

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
