from dataclasses import dataclass
from typing import List
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


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
        """self.detection_distribution = gaussian_filter(
            self.detection_distribution, sigma=1
        )"""
        self.plot_emri_distribution()

    def plot_emri_distribution(self) -> None:
        redshift_range = np.linspace(0, 4, 100)
        mass_range = np.geomspace(1e4, 1e7, 100)

        # evaluate the detection distribution
        detection_distribution = np.array(
            [
                [self.get_emri_probability(z, m) for z in redshift_range]
                for m in mass_range
            ]
        )

        reduced_redshift_range = np.linspace(0, 0.499, 100)
        reduced_mass_range = np.geomspace(1e4, 1e7 - 1e-3, 100)
        reduced_detection_distribution = np.array(
            [
                [self.get_emri_probability(z, m) for z in reduced_redshift_range]
                for m in reduced_mass_range
            ]
        )

        fig, ax = plt.subplots(1, 2, sharey=True)
        cax = ax[0].pcolormesh(
            redshift_range,
            mass_range,
            detection_distribution,
            cmap="viridis",
        )
        cax1 = ax[1].pcolormesh(
            reduced_redshift_range,
            reduced_mass_range,
            reduced_detection_distribution,
            cmap="viridis",
        )
        fig.colorbar(cax1, orientation="horizontal", ax=ax[1])
        fig.colorbar(cax, orientation="horizontal", ax=ax[0])
        ax[0].set_ylabel("M [M_sol]")
        ax[0].set_yscale("log")
        ax[0].set_xlabel("z")
        ax[1].set_xlabel("z")
        plt.savefig(
            "saved_figures/cosmological_model/detection_distribution_m1.png", dpi=300
        )
        plt.close()

    def get_emri_probability(self, z: float, M: float) -> float:
        M = np.log10(M)
        # find bins for z and mass
        z_index = np.digitize(z, self.z_bins) - 1
        M_index = np.digitize(M, self.M_bins) - 1

        return self.detection_distribution[M_index, z_index]
