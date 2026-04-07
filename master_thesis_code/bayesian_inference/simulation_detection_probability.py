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

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

from master_thesis_code.physical_relations import dist_vectorized

logger = logging.getLogger(__name__)

# Number of bins for the P_det grids
_DL_BINS: int = 30
_M_BINS: int = 20

# Maximum number of cached grids (LRU eviction)
_MAX_CACHE_SIZE: int = 20


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
        _force_unit_weights: bool = False,
    ) -> None:
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
        interp_1d = self._build_grid_1d(df_rescaled, self._snr_threshold)

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
        dl_edges = np.linspace(0, dl_max, _DL_BINS + 1)

        M_min = float(np.min(M_vals)) * 0.9  # noqa: N806
        M_max = float(np.max(M_vals)) * 1.1  # noqa: N806
        M_edges = np.geomspace(M_min, M_max, _M_BINS + 1)  # noqa: N806

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
            dl_bin_idx = np.clip(dl_bin_idx, 0, _DL_BINS - 1)
            M_bin_idx = np.clip(M_bin_idx, 0, _M_BINS - 1)

            # Accumulate weighted sums per bin using np.add.at
            total_weights = np.zeros((_DL_BINS, _M_BINS), dtype=np.float64)
            detected_weights = np.zeros((_DL_BINS, _M_BINS), dtype=np.float64)
            n_eff_grid = np.zeros((_DL_BINS, _M_BINS), dtype=np.float64)

            # Also track integer counts for quality flags
            total_counts = np.zeros((_DL_BINS, _M_BINS), dtype=np.float64)
            detected_counts = np.zeros((_DL_BINS, _M_BINS), dtype=np.float64)

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
                out=np.zeros((_DL_BINS, _M_BINS), dtype=np.float64),
                where=total_weights > 0,
            )

            # Kish N_eff per bin: (sum w)^2 / sum(w^2)
            # Kish (1965), Survey Sampling
            sum_w2_grid = np.zeros((_DL_BINS, _M_BINS), dtype=np.float64)
            np.add.at(sum_w2_grid, (dl_bin_idx, M_bin_idx), w**2)
            n_eff_grid = np.divide(
                total_weights**2,
                sum_w2_grid,
                out=np.zeros((_DL_BINS, _M_BINS), dtype=np.float64),
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

        return RegularGridInterpolator(
            (dl_centers, M_centers),
            p_det_grid,
            method="linear",
            bounds_error=False,
            fill_value=0.0,
        )

    def _build_grid_1d(self, df: pd.DataFrame, snr_threshold: float) -> RegularGridInterpolator:
        """Build a 1D P_det(d_L) grid by marginalizing over M and sky angles.

        Args:
            df: DataFrame with columns luminosity_distance, SNR.
            snr_threshold: SNR detection threshold.

        Returns:
            RegularGridInterpolator for P_det(d_L).
        """
        dl_vals = df["luminosity_distance"].values
        snr_vals = df["SNR"].values

        dl_max = float(np.max(dl_vals)) * 1.1
        dl_edges = np.linspace(0, dl_max, _DL_BINS + 1)

        total_counts, _ = np.histogram(dl_vals, bins=dl_edges)
        detected_mask = snr_vals >= snr_threshold
        detected_counts, _ = np.histogram(dl_vals[detected_mask], bins=dl_edges)

        p_det_1d = np.divide(
            detected_counts,
            total_counts,
            out=np.zeros_like(detected_counts, dtype=np.float64),
            where=total_counts > 0,
        )

        dl_centers = 0.5 * (dl_edges[:-1] + dl_edges[1:])

        return RegularGridInterpolator(
            (dl_centers,),
            p_det_1d,
            method="linear",
            bounds_error=False,
            fill_value=0.0,
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
