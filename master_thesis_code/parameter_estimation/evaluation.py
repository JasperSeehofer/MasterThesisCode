import pandas as pd
import os
from itertools import combinations_with_replacement
import numpy as np
from master_thesis_code.datamodels.parameter_space import (
    parameters_configuration,
    get_parameter_configuration,
)
import matplotlib.pyplot as plt
from matplotlib import cm

from master_thesis_code.constants import RADIAN_TO_DEGREE


class DataEvaluation:

    def __init__(
        self,
        path_to_cramer_rao_bounds_file: str = "./simulations/cramer_rao_bounds.csv",
    ):
        self._cramer_rao_bounds = pd.read_csv(path_to_cramer_rao_bounds_file)

    def visualize(self) -> None:
        # ensure directory is given
        figures_directory = f"evaluation/"
        if not os.path.isdir(figures_directory):
            os.makedirs(figures_directory)
        if not os.path.isdir(figures_directory + "plots/"):
            os.makedirs(figures_directory + "plots/")

        parameter_symbol_list = [
            parameter.symbol for parameter in parameters_configuration
        ]
        parameter_combinations = combinations_with_replacement(parameter_symbol_list, 2)

        mean_cramer_rao_bounds = pd.DataFrame(
            index=parameter_symbol_list, columns=parameter_symbol_list
        )

        for a, b in parameter_combinations:
            column_name = f"delta_{b}_delta_{a}"
            mean_cramer_rao_bounds.at[a, b] = self._cramer_rao_bounds[
                column_name
            ].mean()

        mean_cramer_rao_bounds.to_excel(f"{figures_directory}mean_bounds.xlsx")

        # interesting plots
        bounds_parameters = ["M", "qS", "phiS", "qK", "phiK"]
        dependency_parameters = ["qS", "phiS", "qK", "phiK", "Phi_phi0"]

        weird_ones = self._cramer_rao_bounds[
            self._cramer_rao_bounds["delta_phiS_delta_phiS"] > 10**16
        ]
        weird_ones.to_excel(f"{figures_directory}weird_ones.xlsx")

        for parameter_symbol in parameter_symbol_list:
            uncertainty_column_name = (
                f"delta_{parameter_symbol}_delta_{parameter_symbol}"
            )
            parameter_config = get_parameter_configuration(parameter_symbol)
            plt.figure(figsize=(16, 9))
            plt.scatter(
                self._cramer_rao_bounds[parameter_symbol],
                self._cramer_rao_bounds[uncertainty_column_name] ** (1 / 2),
                label=f"uncertainty of {parameter_symbol}",
            )
            plt.xlabel(f"{parameter_symbol} [{parameter_config.unit}]")
            plt.ylabel(
                f"uncertainty bounds {parameter_symbol} [{parameter_config.unit}]"
            )
            plt.legend()
            plt.yscale("log")
            plt.savefig(f"{figures_directory}plots/error_{parameter_symbol}.png")
            plt.close()

        # plot skylocalization uncertainty
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

        d_Omega = _compute_skylocalization_uncertainty(
            self._cramer_rao_bounds["qS"],
            self._cramer_rao_bounds["delta_qS_delta_qS"] ** (1 / 2),
            self._cramer_rao_bounds["delta_phiS_delta_phiS"] ** (1 / 2),
            self._cramer_rao_bounds["delta_phiS_delta_qS"],
        )
        ax.scatter(
            self._cramer_rao_bounds["qS"] * RADIAN_TO_DEGREE,
            self._cramer_rao_bounds["phiS"] * RADIAN_TO_DEGREE,
            d_Omega,
        )
        ax.set_ylabel("Phi in deg")
        ax.set_xlabel("Theta in deg")
        ax.set_zlabel("d_Omega in deg^2")
        plt.show()

        """for bounds_parameter in bounds_parameters:
            bounds_column_name = f"delta_{bounds_parameter}_delta_{bounds_parameter}"
            for dependency_parameter in parameter_symbol_list:
                bound_parameter_config = get_parameter_configuration(bounds_parameter)
                dependency_parameter_config = get_parameter_configuration(
                    dependency_parameter
                )
                plt.figure(figsize=(16, 9))
                plt.scatter(
                    self._cramer_rao_bounds[dependency_parameter],
                    self._cramer_rao_bounds[bounds_column_name],
                    label=f"error of {bounds_parameter}",
                )
                plt.scatter(
                    weird_ones[dependency_parameter],
                    weird_ones[bounds_column_name],
                    c="red",
                    marker="x",
                    label=f"weird ones",
                )
                plt.xlabel(
                    f"{dependency_parameter} [{dependency_parameter_config.unit}]"
                )
                plt.ylabel(
                    f"error bounds {bounds_parameter} [{bound_parameter_config.unit}]"
                )
                plt.legend()
                plt.yscale("log")
                plt.savefig(
                    f"{figures_directory}plots/error_{bounds_parameter}_{dependency_parameter}.png"
                )
                plt.close()"""


def _compute_skylocalization_uncertainty(
    theta, var_theta, var_phi, cov_theta_phi
) -> float:
    return (
        2 * np.pi * np.sin(theta) * np.sqrt(var_phi * var_theta - (cov_theta_phi) ** 2)
    )
