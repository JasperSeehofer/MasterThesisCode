"""KDE-based detection probability for Pipeline B.

:class:`DetectionProbability` builds 4D (with BH mass) and 3D (without BH mass)
kernel density estimates from detected/undetected event DataFrames, then provides
both raw KDE evaluation and pre-computed :class:`RegularGridInterpolator` look-ups
for fast repeated queries inside the multiprocessing likelihood loop.
"""

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import gaussian_kde


class DetectionProbability:
    """Detection probability of a given event."""

    def __init__(
        self,
        luminosity_distance_lower_limit: float,
        luminosity_distance_upper_limit: float,
        mass_lower_limit: float,
        mass_upper_limit: float,
        detected_events: pd.DataFrame,
        undetected_events: pd.DataFrame,
        bandwidth: float | None = None,
    ) -> None:
        self.luminosity_distance_lower_limit = luminosity_distance_lower_limit
        self.luminosity_distance_upper_limit = luminosity_distance_upper_limit
        self.mass_lower_limit = mass_lower_limit
        self.mass_upper_limit = mass_upper_limit

        undetected_events_points = np.array(
            [
                (undetected_events["luminosity_distance"] - luminosity_distance_lower_limit)
                / (luminosity_distance_upper_limit - luminosity_distance_lower_limit),
                (np.log10(undetected_events["M"]) - np.log10(mass_lower_limit))
                / (np.log10(mass_upper_limit) - np.log10(mass_lower_limit)),
                undetected_events["phiS"] / (2 * np.pi),
                undetected_events["qS"] / np.pi,
            ]
        )

        detected_events_points = np.array(
            [
                (detected_events["luminosity_distance"] - luminosity_distance_lower_limit)
                / (luminosity_distance_upper_limit - luminosity_distance_lower_limit),
                (np.log10(detected_events["M"]) - np.log10(mass_lower_limit))
                / (np.log10(mass_upper_limit) - np.log10(mass_lower_limit)),
                detected_events["phiS"] / (2 * np.pi),
                detected_events["qS"] / np.pi,
            ]
        )

        self.kde_undetected_with_bh_mass = gaussian_kde(
            undetected_events_points, bw_method=bandwidth
        )
        self.kde_detected_with_bh_mass = gaussian_kde(detected_events_points, bw_method=bandwidth)

        # create kde and detection probability function for the case without BH mass
        undetected_events_points_without_bh_mass = np.delete(undetected_events_points, 1, axis=0)
        detected_events_points_without_bh_mass = np.delete(detected_events_points, 1, axis=0)

        self.kde_detected_without_bh_mass = gaussian_kde(
            detected_events_points_without_bh_mass, bw_method=bandwidth
        )
        self.kde_undetected_without_bh_mass = gaussian_kde(
            undetected_events_points_without_bh_mass, bw_method=bandwidth
        )

        self._setup_interpolator(d_L_steps=40, M_z_steps=50, phi_steps=20, theta_steps=20)

    def evaluate_with_bh_mass(
        self,
        d_L: float,
        M_z: float,
        phi: float,
        theta: float,
    ) -> float:
        # check if the input values are within the limits
        if any(
            [
                d_L < self.luminosity_distance_lower_limit,
                d_L > self.luminosity_distance_upper_limit,
                M_z < self.mass_lower_limit,
                M_z > self.mass_upper_limit,
                phi < 0,
                phi >= 2 * np.pi,
                theta < 0,
                theta > np.pi,
            ]
        ):
            return 0.0
        # normalize the input values to the range [0, 1]
        d_L, M_z, phi, theta = self._normalize_parameters(  # type: ignore[assignment,misc]
            d_L, phi, theta, M_z
        )

        detected_evaluated = self.kde_detected_with_bh_mass.evaluate([d_L, M_z, phi, theta])[0]
        undetected_evaluated = self.kde_undetected_with_bh_mass.evaluate([d_L, M_z, phi, theta])[0]
        if undetected_evaluated + detected_evaluated == 0.0:
            return 0.0
        return float(detected_evaluated / (undetected_evaluated + detected_evaluated))

    def evaluate_with_bh_mass_vectorized(
        self,
        d_L: npt.NDArray[np.float64],
        M_z: npt.NDArray[np.float64],
        phi: npt.NDArray[np.float64],
        theta: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        # check if the input values are within the limits
        valid_mask = (
            (d_L >= self.luminosity_distance_lower_limit)
            & (d_L <= self.luminosity_distance_upper_limit)
            & (M_z >= self.mass_lower_limit)
            & (M_z <= self.mass_upper_limit)
            & (phi >= 0)
            & (phi < 2 * np.pi)
            & (theta >= 0)
            & (theta <= np.pi)
        )

        # Initialize result array with zeros (or another default value)
        probabilities = np.zeros_like(d_L, dtype=float)

        # Apply the mask to filter valid values
        d_L_valid = d_L[valid_mask]
        M_z_valid = M_z[valid_mask]
        phi_valid = phi[valid_mask]
        theta_valid = theta[valid_mask]

        # Normalize the valid parameters
        d_L_norm, M_z_norm, phi_norm, theta_norm = self._normalize_parameters(  # type: ignore[misc]
            d_L_valid, phi_valid, theta_valid, M_z_valid
        )

        # Evaluate KDEs for valid values
        detected_evaluated = self.kde_detected_with_bh_mass.evaluate(
            np.vstack((d_L_norm, M_z_norm, phi_norm, theta_norm))
        )
        undetected_evaluated = self.kde_undetected_with_bh_mass.evaluate(
            np.vstack((d_L_norm, M_z_norm, phi_norm, theta_norm))
        )

        # Compute probabilities for valid values
        probabilities_valid = np.divide(
            detected_evaluated,
            detected_evaluated + undetected_evaluated,
            out=np.zeros_like(detected_evaluated),
            where=(detected_evaluated + undetected_evaluated) != 0,
        )

        # Assign probabilities back to the result array
        probabilities[valid_mask] = probabilities_valid

        return probabilities

    def evaluate_without_bh_mass(
        self,
        d_L: float,
        phi: float,
        theta: float,
    ) -> float:
        # check if the input values are within the limits
        if any(
            [
                d_L < self.luminosity_distance_lower_limit,
                d_L > self.luminosity_distance_upper_limit,
                phi < 0,
                phi > 2 * np.pi,
                theta < 0,
                theta > np.pi,
            ]
        ):
            return 0.0
        # normalize the input values to the range [0, 1]
        d_L, phi, theta = self._normalize_parameters(  # type: ignore[assignment,misc]
            d_L, phi, theta
        )
        detected_evaluated = self.kde_detected_without_bh_mass.evaluate([d_L, phi, theta])[0]
        undetected_evaluated = self.kde_undetected_without_bh_mass.evaluate([d_L, phi, theta])[0]
        if undetected_evaluated + detected_evaluated == 0:
            return 0.0
        return float(detected_evaluated / (undetected_evaluated + detected_evaluated))

    def _normalize_parameters(
        self,
        d_L: float | npt.NDArray[np.float64],
        phi: float | npt.NDArray[np.float64],
        theta: float | npt.NDArray[np.float64],
        M_z: float | npt.NDArray[np.float64] | None = None,
    ) -> (
        tuple[
            float | npt.NDArray[np.float64],
            float | npt.NDArray[np.float64],
            float | npt.NDArray[np.float64],
            float | npt.NDArray[np.float64],
        ]
        | tuple[
            float | npt.NDArray[np.float64],
            float | npt.NDArray[np.float64],
            float | npt.NDArray[np.float64],
        ]
    ):
        # normalize the input values to the range [0, 1]
        d_L = (d_L - self.luminosity_distance_lower_limit) / (
            self.luminosity_distance_upper_limit - self.luminosity_distance_lower_limit
        )
        phi = phi / (2 * np.pi)
        theta = theta / np.pi
        if M_z is not None:
            M_z = (np.log10(M_z) - np.log10(self.mass_lower_limit)) / (
                np.log10(self.mass_upper_limit) - np.log10(self.mass_lower_limit)
            )
            return d_L, M_z, phi, theta
        return d_L, phi, theta

    def _setup_interpolator(
        self, d_L_steps: int, M_z_steps: int, phi_steps: int, theta_steps: int
    ) -> None:
        # setup grid
        d_L_range = np.linspace(
            self.luminosity_distance_lower_limit,
            self.luminosity_distance_upper_limit,
            d_L_steps,
        )
        M_z_range = np.geomspace(self.mass_lower_limit, self.mass_upper_limit, M_z_steps)
        phi_range = np.linspace(0, 2 * np.pi, phi_steps)
        theta_range = np.linspace(0, np.pi, theta_steps)

        # normalize the ranges to [0, 1]
        d_L_range_norm, M_z_range_norm, phi_range_norm, theta_range_norm = (
            self._normalize_parameters(  # type: ignore[misc]
                d_L_range, phi_range, theta_range, M_z_range
            )
        )

        # create meshgrid
        d_L_grid, phi_grid, theta_grid = np.meshgrid(
            d_L_range_norm, phi_range_norm, theta_range_norm, indexing="ij"
        )

        # flatten the grid
        d_L_flat = d_L_grid.flatten()
        phi_flat = phi_grid.flatten()
        theta_flat = theta_grid.flatten()

        # evaluate the kde for the grid points without bh mass
        kde_values_without_bh_mass = self.kde_detected_without_bh_mass.evaluate(
            np.vstack((d_L_flat, phi_flat, theta_flat))
        )
        kde_values_undetected_without_bh_mass = self.kde_undetected_without_bh_mass.evaluate(
            np.vstack((d_L_flat, phi_flat, theta_flat))
        )

        detection_probabilities_without_bh_mass = np.divide(
            kde_values_without_bh_mass,
            kde_values_without_bh_mass + kde_values_undetected_without_bh_mass,
            out=np.zeros_like(kde_values_without_bh_mass),
            where=(kde_values_without_bh_mass + kde_values_undetected_without_bh_mass) != 0,
        )

        # reshape to grid
        detection_probabilities_without_bh_mass = detection_probabilities_without_bh_mass.reshape(
            d_L_steps, phi_steps, theta_steps
        )

        self.detection_probability_without_bh_mass_interpolator = RegularGridInterpolator(
            (d_L_range, phi_range, theta_range),
            detection_probabilities_without_bh_mass,
            method="linear",
            bounds_error=False,
            fill_value=0.0,
        )

        # evaluate the kde for the grid points with bh mass
        d_L_grid, M_z_grid, phi_grid, theta_grid = np.meshgrid(
            d_L_range_norm, M_z_range_norm, phi_range_norm, theta_range_norm, indexing="ij"
        )
        d_L_flat = d_L_grid.flatten()
        M_z_flat = M_z_grid.flatten()
        phi_flat = phi_grid.flatten()
        theta_flat = theta_grid.flatten()

        kde_values_with_bh_mass = self.kde_detected_with_bh_mass.evaluate(
            np.vstack((d_L_flat, M_z_flat, phi_flat, theta_flat))
        )
        kde_values_undetected_with_bh_mass = self.kde_undetected_with_bh_mass.evaluate(
            np.vstack((d_L_flat, M_z_flat, phi_flat, theta_flat))
        )
        detection_probabilities_with_bh_mass = np.divide(
            kde_values_with_bh_mass,
            kde_values_with_bh_mass + kde_values_undetected_with_bh_mass,
            out=np.zeros_like(kde_values_with_bh_mass),
            where=(kde_values_with_bh_mass + kde_values_undetected_with_bh_mass) != 0,
        )
        # reshape to grid
        detection_probabilities_with_bh_mass = detection_probabilities_with_bh_mass.reshape(
            d_L_steps, M_z_steps, phi_steps, theta_steps
        )

        self.detection_probability_with_bh_mass_interpolator = RegularGridInterpolator(
            (d_L_range, M_z_range, phi_range, theta_range),
            detection_probabilities_with_bh_mass,
            method="linear",
            bounds_error=False,
            fill_value=0.0,
        )

    def detection_probability_with_bh_mass_interpolated(
        self,
        d_L: float | npt.NDArray[np.float64],
        M_z: float | npt.NDArray[np.float64],
        phi: float | npt.NDArray[np.float64],
        theta: float | npt.NDArray[np.float64],
    ) -> float | npt.NDArray[np.float64]:
        # check if the input values are floats
        if all([isinstance(attribute, float) for attribute in [d_L, M_z, phi, theta]]):
            # if all attributes are float, convert them to 1D arrays
            d_L = np.array([d_L])
            M_z = np.array([M_z])
            phi = np.array([phi])
            theta = np.array([theta])
        return self.detection_probability_with_bh_mass_interpolator(  # type: ignore[no-any-return]
            np.array([d_L, M_z, phi, theta]).T
        )

    def detection_probability_without_bh_mass_interpolated(
        self,
        d_L: float | npt.NDArray[np.float64],
        phi: float | npt.NDArray[np.float64],
        theta: float | npt.NDArray[np.float64],
    ) -> float | npt.NDArray[np.float64]:
        # check if the input values are floats
        if all([isinstance(attribute, float) for attribute in [d_L, phi, theta]]):
            # if all attributes are float, convert them to 1D arrays
            d_L = np.array([d_L])
            phi = np.array([phi])
            theta = np.array([theta])
        return self.detection_probability_without_bh_mass_interpolator(  # type: ignore[no-any-return]
            np.array([d_L, phi, theta]).T
        )
