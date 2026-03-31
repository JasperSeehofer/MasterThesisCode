"""Simulation-based detection probability from injection campaign data.

Replaces :class:`DetectionProbability` (KDE-based) with a histogram-binned
approach that loads raw injection CSVs, applies an SNR threshold at evaluation
time, and builds P_det interpolation grids per Hubble parameter value.

The class provides drop-in replacement methods for the old
``DetectionProbability`` interface, with an additional ``h`` keyword argument
to support h-dependent detection probability lookups.
"""

import glob
import logging
import re
from bisect import bisect_right

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

from master_thesis_code.physical_relations import dist_to_redshift

logger = logging.getLogger(__name__)

# Number of bins for the P_det grids
_Z_BINS: int = 30
_M_BINS: int = 20


class SimulationDetectionProbability:
    """Simulation-based detection probability from injection campaign data.

    Loads raw injection CSVs (z, M, phiS, qS, SNR, h_inj), applies an SNR
    threshold at evaluation time, bins into P_det grids, and provides
    interpolated lookups.

    Per D-02: starts with P_det(z, M | h) by marginalizing over sky angles.
    Per D-03: raw SNR stored; threshold applied here at evaluation time.
    Per D-06: interpolates P_det between h grid points.

    Args:
        injection_data_dir: Directory containing injection CSV files matching
            ``injection_h_*_task_*.csv`` or ``injection_h_*.csv``.
        snr_threshold: SNR threshold for detection. Events with
            SNR >= snr_threshold are considered detected.
        h_grid: Explicit list of h values to use. If None, auto-detected
            from filenames.
    """

    def __init__(
        self,
        injection_data_dir: str,
        snr_threshold: float,
        h_grid: list[float] | None = None,
    ) -> None:
        self._snr_threshold = snr_threshold

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

        # Group files by h value extracted from filename
        h_file_map: dict[float, list[str]] = {}
        h_pattern = re.compile(r"injection_h_(\d+p\d+)")
        for f in csv_files:
            match = h_pattern.search(f)
            if match:
                h_label = match.group(1)
                h_val = float(h_label.replace("p", "."))
                h_file_map.setdefault(h_val, []).append(f)

        if not h_file_map:
            msg = (
                f"Could not extract h values from filenames in '{injection_data_dir}'. "
                "Expected format: 'injection_h_0p70_task_001.csv'."
            )
            raise FileNotFoundError(msg)

        # Set h grid
        if h_grid is not None:
            self._h_grid: list[float] = sorted(h_grid)
        else:
            self._h_grid = sorted(h_file_map.keys())

        # Load data and build interpolators per h value
        self._interpolators: dict[float, RegularGridInterpolator] = {}
        self._interpolators_1d: dict[float, RegularGridInterpolator] = {}

        for h_val in self._h_grid:
            files = h_file_map.get(h_val, [])
            if not files:
                logger.warning(
                    "No injection files found for h=%.4f, skipping.", h_val
                )
                continue

            dfs = [pd.read_csv(f) for f in files]
            df = pd.concat(dfs, ignore_index=True)

            logger.info(
                "Loaded %d injection events for h=%.4f from %d files.",
                len(df),
                h_val,
                len(files),
            )

            self._interpolators[h_val] = self._build_grid_2d(df, snr_threshold)
            self._interpolators_1d[h_val] = self._build_grid_1d(df, snr_threshold)

    def _build_grid_2d(
        self, df: pd.DataFrame, snr_threshold: float
    ) -> RegularGridInterpolator:
        """Build a 2D P_det(z, M) grid by marginalizing over sky angles.

        Args:
            df: DataFrame with columns z, M, SNR (at minimum).
            snr_threshold: SNR detection threshold.

        Returns:
            RegularGridInterpolator for P_det(z, M).
        """
        z_vals = df["z"].values
        M_vals = df["M"].values  # noqa: N806
        snr_vals = df["SNR"].values

        # Define bin edges
        z_max = float(np.max(z_vals)) * 1.1
        z_edges = np.linspace(0, z_max, _Z_BINS + 1)

        M_min = float(np.min(M_vals)) * 0.9  # noqa: N806
        M_max = float(np.max(M_vals)) * 1.1  # noqa: N806
        M_edges = np.geomspace(M_min, M_max, _M_BINS + 1)  # noqa: N806

        # Total events histogram
        total_counts, _, _ = np.histogram2d(
            z_vals, M_vals, bins=[z_edges, M_edges]
        )

        # Detected events histogram
        detected_mask = snr_vals >= snr_threshold
        detected_counts, _, _ = np.histogram2d(
            z_vals[detected_mask],
            M_vals[detected_mask],
            bins=[z_edges, M_edges],
        )

        # P_det = detected / total, with 0/0 -> 0.0 (conservative)
        p_det_grid = np.divide(
            detected_counts,
            total_counts,
            out=np.zeros_like(detected_counts, dtype=np.float64),
            where=total_counts > 0,
        )

        # Use bin centers as grid coordinates
        z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])
        M_centers = np.sqrt(M_edges[:-1] * M_edges[1:])  # geometric mean for log-spaced  # noqa: N806

        return RegularGridInterpolator(
            (z_centers, M_centers),
            p_det_grid,
            method="linear",
            bounds_error=False,
            fill_value=0.0,
        )

    def _build_grid_1d(
        self, df: pd.DataFrame, snr_threshold: float
    ) -> RegularGridInterpolator:
        """Build a 1D P_det(z) grid by marginalizing over M and sky angles.

        Args:
            df: DataFrame with columns z, SNR (at minimum).
            snr_threshold: SNR detection threshold.

        Returns:
            RegularGridInterpolator for P_det(z) (1D, but wrapped as 2D-API
            compatible with a single axis).
        """
        z_vals = df["z"].values
        snr_vals = df["SNR"].values

        z_max = float(np.max(z_vals)) * 1.1
        z_edges = np.linspace(0, z_max, _Z_BINS + 1)

        total_counts, _ = np.histogram(z_vals, bins=z_edges)
        detected_mask = snr_vals >= snr_threshold
        detected_counts, _ = np.histogram(
            z_vals[detected_mask], bins=z_edges
        )

        p_det_1d = np.divide(
            detected_counts,
            total_counts,
            out=np.zeros_like(detected_counts, dtype=np.float64),
            where=total_counts > 0,
        )

        z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])

        return RegularGridInterpolator(
            (z_centers,),
            p_det_1d,
            method="linear",
            bounds_error=False,
            fill_value=0.0,
        )

    def _interpolate_at_h(
        self,
        z: float | npt.NDArray[np.float64],
        M: float | npt.NDArray[np.float64],  # noqa: N803
        h: float,
    ) -> float | npt.NDArray[np.float64]:
        """Interpolate P_det(z, M) between h grid points.

        Args:
            z: Redshift value(s).
            M: Source-frame BH mass value(s).
            h: Hubble parameter value.

        Returns:
            Detection probability, linearly interpolated between bracketing
            h grid values.
        """
        z_arr = np.atleast_1d(np.asarray(z, dtype=np.float64))
        M_arr = np.atleast_1d(np.asarray(M, dtype=np.float64))  # noqa: N806
        points = np.column_stack([z_arr, M_arr])

        # Find bracketing h values
        idx = bisect_right(self._h_grid, h)

        if idx == 0:
            # h below grid range -- clamp to lowest
            result = self._interpolators[self._h_grid[0]](points)
        elif idx >= len(self._h_grid):
            # h above grid range -- clamp to highest
            result = self._interpolators[self._h_grid[-1]](points)
        elif h == self._h_grid[idx - 1]:
            # Exact match
            result = self._interpolators[h](points)
        else:
            # Linear interpolation between bracketing values
            h_low = self._h_grid[idx - 1]
            h_high = self._h_grid[idx]
            t = (h - h_low) / (h_high - h_low)
            p_low = self._interpolators[h_low](points)
            p_high = self._interpolators[h_high](points)
            result = (1 - t) * p_low + t * p_high

        # Clip to [0, 1] for safety
        result = np.clip(result, 0.0, 1.0)

        if np.ndim(z) == 0 and np.ndim(M) == 0:
            return float(result[0])
        return result  # type: ignore[no-any-return]

    def _interpolate_at_h_1d(
        self,
        z: float | npt.NDArray[np.float64],
        h: float,
    ) -> float | npt.NDArray[np.float64]:
        """Interpolate P_det(z) between h grid points (1D, marginalized over M).

        Args:
            z: Redshift value(s).
            h: Hubble parameter value.

        Returns:
            Detection probability, linearly interpolated between bracketing
            h grid values.
        """
        z_arr = np.atleast_1d(np.asarray(z, dtype=np.float64))
        # RegularGridInterpolator expects shape (N, ndim) even for 1D
        points = z_arr.reshape(-1, 1)

        idx = bisect_right(self._h_grid, h)

        if idx == 0:
            result = self._interpolators_1d[self._h_grid[0]](points)
        elif idx >= len(self._h_grid):
            result = self._interpolators_1d[self._h_grid[-1]](points)
        elif h == self._h_grid[idx - 1]:
            result = self._interpolators_1d[h](points)
        else:
            h_low = self._h_grid[idx - 1]
            h_high = self._h_grid[idx]
            t = (h - h_low) / (h_high - h_low)
            p_low = self._interpolators_1d[h_low](points)
            p_high = self._interpolators_1d[h_high](points)
            result = (1 - t) * p_low + t * p_high

        result = np.clip(result, 0.0, 1.0)

        if np.ndim(z) == 0:
            return float(result[0])
        return result  # type: ignore[no-any-return]

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

        Args:
            d_L: Luminosity distance in Gpc.
            M_z: Observer-frame (redshifted) BH mass in solar masses.
            phi: Sky angle phi (unused, marginalized over).
            theta: Sky angle theta (unused, marginalized over).
            h: Dimensionless Hubble parameter.

        Returns:
            Detection probability in [0, 1].
        """
        # Convert d_L to redshift
        z: float | npt.NDArray[np.float64]
        if np.ndim(d_L) == 0:
            z = dist_to_redshift(float(d_L), h=h)
        else:
            z = np.array([dist_to_redshift(float(dl), h=h) for dl in np.atleast_1d(d_L)])

        # Convert observer-frame mass to source-frame mass: M_true = M_z / (1 + z)
        M_true = np.asarray(M_z, dtype=np.float64) / (1 + np.asarray(z, dtype=np.float64))  # noqa: N806

        return self._interpolate_at_h(z, M_true, h)

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
        """
        z: float | npt.NDArray[np.float64]
        if np.ndim(d_L) == 0:
            z = dist_to_redshift(float(d_L), h=h)
        else:
            z = np.array([dist_to_redshift(float(dl), h=h) for dl in np.atleast_1d(d_L)])

        return self._interpolate_at_h_1d(z, h)
