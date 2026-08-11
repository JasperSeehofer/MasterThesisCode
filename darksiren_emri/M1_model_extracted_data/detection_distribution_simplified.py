from dataclasses import dataclass

import numpy as np


@dataclass
class SimpleDetectionDistribution:
    M_bins = np.array([4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0])
    z_bins = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
    z_values = np.array([20, 60, 70, 60, 50, 40, 30, 15, 9])
    M_values = np.array([0, 9, 120, 110, 40, 4, 0])
    detection_distribution: np.ndarray

    def __init__(self) -> None:
        self.detection_distribution = np.array(
            [self.z_values * M_value for M_value in self.M_values]
        )
        # normalize
        self.detection_distribution = self.detection_distribution / np.sum(
            self.detection_distribution
        )

    def get_emri_probability(self, z: float, M: float) -> float:
        M = np.log10(M)
        # find bins for z and mass
        z_index = np.digitize(z, self.z_bins) - 1
        M_index = np.digitize(M, self.M_bins) - 1

        return float(self.detection_distribution[M_index, z_index])
