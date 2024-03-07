import pandas as pd
import os
from itertools import combinations_with_replacement
import numpy as np
import matplotlib.pyplot as plt
from master_thesis_code.datamodels.parameter_space import ParameterSpace, Parameter
from scipy.interpolate import griddata

from master_thesis_code.constants import RADIAN_TO_DEGREE, C, H0, GPC_TO_MPC


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

        parameter_space = ParameterSpace()

        parameter_symbol_list = list(parameter_space._parameters_to_dict().keys())
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

        for parameter in vars(parameter_space).values():
            assert isinstance(parameter, Parameter)
            uncertainty_column_name = (
                f"delta_{parameter.symbol}_delta_{parameter.symbol}"
            )
            plt.figure(figsize=(16, 9))
            plt.scatter(
                self._cramer_rao_bounds[parameter.symbol],
                self._cramer_rao_bounds[uncertainty_column_name] ** (1 / 2),
                label=f"relativ uncertainty of {parameter.symbol}",
            )
            plt.xlabel(f"{parameter.symbol} [{parameter.unit}]")
            plt.ylabel(
                f"relative uncertainty bounds {parameter.symbol} [{parameter.unit}]"
            )
            plt.legend()
            plt.yscale("log")
            plt.savefig(f"{figures_directory}plots/error_{parameter.symbol}.png")
            plt.close()

        # plot SNR vs uncertainty
        for parameter in vars(parameter_space).values():
            assert isinstance(parameter, Parameter)
            uncertainty_column_name = (
                f"delta_{parameter.symbol}_delta_{parameter.symbol}"
            )
            plt.figure(figsize=(16, 9))
            plt.scatter(
                self._cramer_rao_bounds["SNR"],
                self._cramer_rao_bounds[uncertainty_column_name] ** (1 / 2),
                label=f"relativ uncertainty of {parameter.symbol}",
            )
            plt.xlabel(f"SNR")
            plt.ylabel(
                f"relative uncertainty bounds {parameter.symbol} [{parameter.unit}]"
            )
            plt.legend()
            plt.yscale("log")
            plt.savefig(f"{figures_directory}plots/error_{parameter.symbol}_SNR.png")
            plt.close()

        # plt SNR and waveform generation time vs parameters
        for column in ["SNR", "generation_time"]:
            for parameter in vars(parameter_space).values():
                assert isinstance(parameter, Parameter)
                plt.figure(figsize=(16, 9))
                plt.scatter(
                    self._cramer_rao_bounds[parameter.symbol],
                    self._cramer_rao_bounds[column],
                    label=column,
                )
                plt.xlabel(f"{parameter.symbol}")
                plt.ylabel(column)
                plt.legend()
                plt.yscale("log")
                plt.savefig(f"{figures_directory}plots/{parameter.symbol}_{column}.png")
                plt.close()

        # boxplot of cramer rao bounds
        use_relative_uncertainty = ["M", "dist", "mu"]
        uncertainies_list = []
        for parameter in vars(parameter_space).values():
            assert isinstance(parameter, Parameter)
            uncertainty_column_name = (
                f"delta_{parameter.symbol}_delta_{parameter.symbol}"
            )
            uncertainties = self._cramer_rao_bounds[uncertainty_column_name] ** (1 / 2)

            if parameter.symbol in use_relative_uncertainty:
                uncertainties = (
                    uncertainties / self._cramer_rao_bounds[parameter.symbol]
                )
            uncertainies_list.append(uncertainties)

        sky_uncertainty = [
            _compute_skylocalization_uncertainty(qs, var_qs, var_phis, cov_qs_phis)
            for qs, var_qs, var_phis, cov_qs_phis in zip(
                self._cramer_rao_bounds["qS"],
                self._cramer_rao_bounds["delta_qS_delta_qS"],
                self._cramer_rao_bounds["delta_phiS_delta_phiS"],
                self._cramer_rao_bounds["delta_phiS_delta_qS"],
            )
        ]

        uncertainies_list.append(sky_uncertainty)

        fig, ax = plt.subplots()
        ax.violinplot(uncertainies_list, showmeans=True)
        ax.set_xticks(np.arange(1, len(parameter_symbol_list) + 2))
        ax.set_xticklabels(parameter_symbol_list + ["skylocalization"])
        plt.yscale("log")
        plt.savefig(f"{figures_directory}plots/boxplot_uncertainties.png", dpi=300)
        plt.close()

        # plot redshift detections distribution
        # convert distance to redshift
        redshifts = (
            self._cramer_rao_bounds["dist"] * GPC_TO_MPC / C * H0
        )  # converting to Mpc and to m

        bin_edges = np.arange(0, 7, 0.5)

        plt.figure(figsize=(16, 9))
        plt.hist(redshifts, bins=bin_edges, histtype="step")
        plt.xlabel("redshift")
        plt.ylabel("detections")
        plt.yscale("log")
        plt.ylim(1e-1, 1e3)
        plt.xlim(0, 7)
        plt.savefig(f"{figures_directory}plots/redshift_detections.png")
        plt.close()

        # plot mass detections distribution
        # convert distance to redshift
        redshifts = (
            self._cramer_rao_bounds["dist"] * 10**3 / C * H0
        )  # converting to Mpc

        source_masses = self._cramer_rao_bounds["M"] / (1 + redshifts)

        bin_edges = np.arange(3.5, 7.5, 0.5)

        plt.figure(figsize=(16, 9))
        plt.hist(np.log10(source_masses), bins=bin_edges, histtype="step")
        plt.xlabel("log_10 source mass [solar masses]")
        plt.ylabel("detections")
        plt.yscale("log")
        plt.ylim(1, 1e4)
        plt.xlim(3.5, 7.5)
        plt.savefig(f"{figures_directory}plots/source_masses_detections.png")
        plt.close()

        # plot SNR detections distribution
        # convert distance to redshift

        bin_edges = np.arange(1, 4, 0.25)

        plt.figure(figsize=(16, 9))
        plt.hist(
            np.log10(self._cramer_rao_bounds["SNR"]), bins=bin_edges, histtype="step"
        )
        plt.xlabel("log_10 SNR")
        plt.ylabel("detections")
        plt.yscale("log")
        plt.ylim(1e-2, 1e4)
        plt.xlim(1, 4)
        plt.savefig(f"{figures_directory}plots/SNR_detections.png")
        plt.close()

        # plot mass redshift detection fraction
        grid_x, grid_y = np.mgrid[0:5:20j, 4:6.5:20j]

        hist_detections, _, _ = np.histogram2d(
            redshifts, np.log10(source_masses), bins=[grid_x[:, 0], grid_y[0, :]]
        )

        non_detections = np.array(
            [
                np.random.random_sample(10000) * 5,
                np.random.random_sample(10000) * (6.5 - 4) + 4,
            ]
        ).transpose()
        hist_non_detections, _, _ = np.histogram2d(
            non_detections[:, 0],
            non_detections[:, 1],
            bins=[grid_x[:, 0], grid_y[0, :]],
        )

        detection_fraction = hist_detections / (hist_detections + hist_non_detections)

        grid_x, grid_y = grid_x[:-1, :-1], grid_y[:-1, :-1]
        fig, ax = plt.subplots()
        contour = ax.contourf(grid_x, grid_y, detection_fraction, cmap="viridis")
        fig.colorbar(contour, label="detection fraction")
        plt.xlabel("redshift")
        plt.ylabel("log_10 source mass [solar masses]")
        plt.savefig(
            f"{figures_directory}plots/mass_redshift_detection_fraction.png", dpi=300
        )

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
        plt.savefig(
            f"{figures_directory}plots/sky_localization_uncertainty.png", dpi=300
        )
        plt.close()

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
        2
        * np.pi
        * np.abs(np.sin(theta))
        * np.sqrt(var_phi * var_theta - (cov_theta_phi) ** 2)
    )
