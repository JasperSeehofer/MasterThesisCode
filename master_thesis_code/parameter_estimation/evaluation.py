import pandas as pd
import os
from itertools import combinations_with_replacement
import numpy as np
import matplotlib.pyplot as plt
from master_thesis_code.datamodels.parameter_space import ParameterSpace, Parameter
from master_thesis_code.cosmological_model import Detection
from scipy.interpolate import griddata
from scipy.stats import multivariate_normal
from statistics import NormalDist
from master_thesis_code.physical_relations import (
    get_redshift_outer_bounds,
    dist_to_redshift,
    dist,
)
from master_thesis_code.galaxy_catalogue.handler import GalaxyCatalogueHandler

from master_thesis_code.constants import RADIAN_TO_DEGREE, C, H0, GPC_TO_MPC


class DataEvaluation:

    def __init__(
        self,
        path_to_cramer_rao_bounds_file: str = "./simulations/cramer_rao_bounds.csv",
        path_to_snr_analysis_file: str = "./simulations/snr_analysis.csv",
        path_to_prepared_cramer_rao_bounds="./simulations/prepared_cramer_rao_bounds.csv",
        path_to_undetected_events_file: str = "./simulations/undetected_events.csv",
    ):
        self._cramer_rao_bounds = pd.read_csv(path_to_cramer_rao_bounds_file)
        self._snr_analysis_file = pd.read_csv(path_to_snr_analysis_file)
        self._prepared_cramer_rao_bounds = pd.read_csv(
            path_to_prepared_cramer_rao_bounds
        )
        self._undetected_events = pd.read_csv(path_to_undetected_events_file)

    def visualize(self, galaxy_catalog: GalaxyCatalogueHandler) -> None:
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
            mean = self._cramer_rao_bounds[column_name].mean()
            mean_cramer_rao_bounds.at[a, b] = mean
            mean_cramer_rao_bounds.at[b, a] = mean

        # mean_cramer_rao_bounds.to_excel(f"{figures_directory}mean_bounds.xlsx")
        print(mean_cramer_rao_bounds.values.astype(float))

        # create 2d plot of matrix
        plt.figure(figsize=(16, 9))
        plt.matshow(
            mean_cramer_rao_bounds.values.astype(float),
            cmap="viridis",
            vmin=-1e-4,
            vmax=1e-4,
        )
        plt.colorbar()
        plt.xticks(
            range(len(parameter_symbol_list)),
            parameter_symbol_list,
            rotation=-45,
            ha="right",
        )
        plt.yticks(
            range(len(parameter_symbol_list)),
            parameter_symbol_list,
            rotation=45,
            ha="right",
        )
        plt.savefig(f"{figures_directory}plots/mean_bounds.png", dpi=300)
        plt.close()

        # create 3d spherical coordinates plot of detections

        fig = plt.figure(figsize=(16, 9))
        ax = fig.add_subplot(111, projection="3d")

        r = self._cramer_rao_bounds["dist"] * GPC_TO_MPC * H0 / C
        theta = self._cramer_rao_bounds["phiS"]
        phi = self._cramer_rao_bounds["qS"]
        snr = self._cramer_rao_bounds["SNR"]

        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)

        ax.scatter(x, y, z, c=snr, cmap="viridis")
        # show colorbar
        cbar = plt.colorbar(ax.scatter(x, y, z, c=snr, cmap="viridis"))
        cbar.set_label("SNR")
        plt.savefig(f"{figures_directory}plots/detections_3d.png", dpi=300)
        plt.close()

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

        # plot gaussian for bh mass uncertainty
        for detection_index, detection in self._prepared_cramer_rao_bounds.sample(
            20
        ).iterrows():
            detection = Detection(detection)
            possible_host = galaxy_catalog.get_host_galaxy_by_index(
                detection.host_galaxy_index
            )
            print(possible_host)
            # create full covariance matrix for all parameters
            covariance_matrix = np.array(
                [
                    [
                        detection.M_uncertainty**2,
                        detection.d_L_M_covariance,
                        detection.M_phi_covariance,
                        detection.M_theta_covariance,
                    ],
                    [
                        detection.d_L_M_covariance,
                        detection.d_L_uncertainty**2,
                        detection.d_L_phi_covariance,
                        detection.d_L_theta_covariance,
                    ],
                    [
                        detection.M_phi_covariance,
                        detection.d_L_phi_covariance,
                        detection.phi_error**2,
                        detection.theta_phi_covariance,
                    ],
                    [
                        detection.M_theta_covariance,
                        detection.d_L_theta_covariance,
                        detection.theta_phi_covariance,
                        detection.theta_error**2,
                    ],
                ]
            )
            gaussian_mass = multivariate_normal(
                mean=[detection.M, detection.d_L, detection.phi, detection.theta],
                cov=covariance_matrix,
            )
            normal_bh_mass = NormalDist(possible_host.M, possible_host.M_error)
            normal_z = NormalDist(possible_host.z, possible_host.z_error)
            phi = detection.phi
            theta = detection.theta
            z_min, z_max = get_redshift_outer_bounds(
                detection.d_L, detection.d_L_uncertainty
            )
            redshifts = np.linspace(
                possible_host.z - 3 * possible_host.z_error,
                possible_host.z + 3 * possible_host.z_error,
                10000,
            )
            distances = np.array([dist(redshift) for redshift in redshifts])
            masses = detection.M / (1 + redshifts)
            redshifted_masses = np.ones_like(masses) * detection.M

            positions = np.array(
                [
                    redshifted_masses,
                    distances,
                    np.ones_like(masses) * phi,
                    np.ones_like(masses) * theta,
                ]
            ).T

            probabilities = gaussian_mass.pdf(positions)
            mass_weight = np.array([normal_bh_mass.pdf(m) for m in masses])
            approximated_mass_weight = normal_bh_mass.pdf(
                detection.M / (1 + possible_host.z)
            )
            z_weight = np.array([normal_z.pdf(z) for z in redshifts])

            approximated_probabilities = (
                probabilities * approximated_mass_weight * z_weight
            )
            probabilities = probabilities * mass_weight * z_weight

            # integrate over redshift and mass
            integrated_probabilities = np.trapz(probabilities, redshifts)
            approximated_integral = np.trapz(approximated_probabilities, redshifts)

            """
            # try better approximation
            five_point_mass_approximation = (
                np.ones(shape=(2001, len(redshifts))) * possible_host.M
            )
            for index_1, step in enumerate(range(-1000, 1001, 1)):
                five_point_mass_approximation[index_1] = five_point_mass_approximation[
                    index_1
                ] + step / 1000 * 20 * detection.M_uncertainty

            # make distances same dimension as masses
            new_distances = np.array([distances for _ in range(2001)])

            five_point_mass_positions = np.array(
                [
                    five_point_mass_approximation.flatten(),
                    new_distances.flatten(),
                    np.ones_like(new_distances.flatten()) * phi,
                    np.ones_like(new_distances.flatten()) * theta,
                ]
            ).T
            better_approximation_probabilities = gaussian_mass.pdf(
                five_point_mass_positions
            )
            # reshape
            better_approximation_probabilities = (
                better_approximation_probabilities.reshape(
                    five_point_mass_approximation.shape
                )
            )

            better_approximation_z_probabilities = []
            for index_2, fixed_z in enumerate(better_approximation_probabilities.T):
                better_approximation_z_probabilities.append(
                    np.trapz(fixed_z, x=five_point_mass_approximation.T[index_2])
                )
            better_approximated_integral = np.trapz(
                better_approximation_z_probabilities, x=redshifts, axis=0
            )

            # try better approximation
            ten_point_mass_approximation = (
                np.ones(shape=(41, len(redshifts))) * detection.M
            )
            for index_3, step in enumerate(range(-20, 21, 1)):
                ten_point_mass_approximation[index_3] = ten_point_mass_approximation[
                    index_3
                ] + (step / 20 * 10) * detection.M_uncertainty / (1 + redshifts)

            # make distances same dimension as masses
            new_distances = np.array([distances for _ in range(41)])

            ten_point_mass_positions = np.array(
                [
                    ten_point_mass_approximation.flatten(),
                    new_distances.flatten(),
                    np.ones_like(new_distances.flatten()) * phi,
                    np.ones_like(new_distances.flatten()) * theta,
                ]
            ).T
            ten_point_approximation_probabilities = gaussian_mass.pdf(
                ten_point_mass_positions
            )
            # reshape
            ten_point_approximation_probabilities = (
                ten_point_approximation_probabilities.reshape(
                    ten_point_mass_approximation.shape
                )
            )

            ten_point_approximation_z_probabilities = []
            for index_4, fixed_z in enumerate(ten_point_approximation_probabilities.T):
                ten_point_approximation_z_probabilities.append(
                    np.trapz(fixed_z, x=ten_point_mass_approximation.T[index_4])
                )
            better_approximated_ten_integral = np.trapz(
                ten_point_approximation_z_probabilities, x=redshifts, axis=0
            )

            print(
                f"better approximated integral (101 steps): {better_approximated_integral}, 10 steps (deviation): {better_approximated_integral - better_approximated_ten_integral}"
            )

            redshifts = np.array([redshifts for _ in range(41)])
            print(redshifts)
            print(ten_point_mass_approximation)
            """
            # plot likelihood over redshift
            plt.figure(figsize=(16, 9))
            plt.plot(
                redshifts, probabilities, label=f"integral: {integrated_probabilities}"
            )
            plt.plot(
                redshifts,
                approximated_probabilities,
                label=f"approximated integral: {approximated_integral}",
                linestyle="--",
            )
            plt.xlabel("redshift")
            plt.ylabel("likelihood")
            plt.yscale("log")
            plt.legend()
            plt.savefig(
                f"{figures_directory}plots/likelihood_over_redshift_{detection_index}.png"
            )
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
        use_relative_uncertainty = []  # "M", "dist", "mu"
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

        violin_parts = ax.violinplot(
            uncertainies_list, showmeans=True, showextrema=False
        )
        for partname in ["cmeans"]:
            vp = violin_parts[partname]
            vp.set_edgecolor("seagreen")
            vp.set_linewidth(1)
        for pc in violin_parts["bodies"]:
            pc.set_color("seagreen")
            pc.set_alpha(0.6)
        ax.set_xticks(np.arange(1, len(parameter_symbol_list) + 2))
        ax.set_xticklabels(
            parameter_symbol_list + ["skylocalization"],
            rotation=45,
            ha="right",
            fontsize=8,
        )
        plt.yscale("log")
        # make sure ticks are not cut off
        plt.tight_layout()
        plt.savefig(f"{figures_directory}plots/boxplot_uncertainties.png", dpi=300)
        plt.close()

        # violin plot of cramer rao bounds covariance
        parameters_of_interest = ["M", "dist", "phiS", "qS"]
        for parameter in parameters_of_interest:
            covariance_list = []
            for parameter_2 in parameter_symbol_list:
                covariance_column_name = f"delta_{parameter}_delta_{parameter_2}"
                try:
                    covariances = self._cramer_rao_bounds[covariance_column_name]
                except KeyError:
                    covariance_column_name = f"delta_{parameter_2}_delta_{parameter}"
                    covariances = self._cramer_rao_bounds[covariance_column_name]

                covariance_list.append(covariances)

            fig, ax = plt.subplots()
            # set title
            ax.set_title(f"covariances of {parameter}")
            violin_parts = ax.violinplot(
                covariance_list, showmeans=True, showextrema=False
            )
            for partname in ["cmeans"]:
                vp = violin_parts[partname]
                vp.set_edgecolor("seagreen")
                vp.set_linewidth(1)

            for pc in violin_parts["bodies"]:
                pc.set_color("seagreen")
                pc.set_alpha(0.6)

            ax.set_xticks(np.arange(1, len(parameter_symbol_list) + 1))
            ax.set_xticklabels(
                parameter_symbol_list,
                rotation=45,
                ha="right",
                fontsize=8,
            )
            plt.tight_layout()
            plt.savefig(
                f"{figures_directory}plots/boxplot_covariances_{parameter}.png", dpi=300
            )
            plt.close()

        # plot redshift detections distribution
        # convert distance to redshift
        redshifts = (
            self._cramer_rao_bounds["dist"] * GPC_TO_MPC / C * H0
        )  # converting to Mpc and to m

        bin_edges = np.arange(0, max(redshifts), int(max(redshifts) * 100))

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

        hist_non_detections, _, _ = np.histogram2d(
            self._undetected_events["dist"] * 10**3 / C * H0,
            np.log10(
                self._undetected_events["M"]
                / (1 + self._undetected_events["dist"] * 10**3 / C * H0)
            ),
            bins=[grid_x[:, 0], grid_y[0, :]],
        )

        total_number_of_events = _remove_zeros_from_grid(
            hist_non_detections + hist_detections
        )

        detection_fraction = hist_detections / total_number_of_events

        grid_x, grid_y = grid_x[:-1, :-1], grid_y[:-1, :-1]
        fig, ax = plt.subplots()
        contour = ax.contourf(grid_x, grid_y, hist_detections, cmap="viridis")
        fig.colorbar(contour, label="detections")
        plt.xlabel("redshift")
        plt.ylabel("log_10 source mass [solar masses]")
        plt.savefig(f"{figures_directory}plots/mass_redshift_detections.png", dpi=300)
        plt.close()

        # plot non detected events
        fig, ax = plt.subplots()
        contour = ax.contourf(grid_x, grid_y, hist_non_detections, cmap="viridis")
        fig.colorbar(contour, label="not detected")
        plt.xlabel("redshift")
        plt.ylabel("log_10 source mass [solar masses]")
        plt.savefig(f"{figures_directory}plots/mass_redshift_not_detected.png", dpi=300)
        plt.close()

        # plt detection fraction
        fig, ax = plt.subplots()
        contour = ax.contourf(grid_x, grid_y, detection_fraction, cmap="viridis")
        fig.colorbar(contour, label="detection fraction")
        plt.xlabel("redshift")
        plt.ylabel("log_10 source mass [solar masses]")
        plt.savefig(
            f"{figures_directory}plots/mass_redshift_detection_fraction.png", dpi=300
        )
        plt.close()

        # plot skylocalization uncertainty
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

        d_Omega = _compute_skylocalization_uncertainty(
            self._cramer_rao_bounds["qS"],
            self._cramer_rao_bounds["delta_qS_delta_qS"],
            self._cramer_rao_bounds["delta_phiS_delta_phiS"],
            self._cramer_rao_bounds["delta_phiS_delta_qS"],
        )
        ax.scatter(
            self._cramer_rao_bounds["qS"] * RADIAN_TO_DEGREE,
            self._cramer_rao_bounds["phiS"] * RADIAN_TO_DEGREE,
            d_Omega,
        )
        ax.set_ylabel("Phi in deg")
        ax.set_xlabel("Theta in deg")
        ax.set_zlabel("d_Omega in rad^2")
        plt.savefig(
            f"{figures_directory}plots/sky_localization_uncertainty.png", dpi=300
        )
        plt.close()

        # plot skylocalization uncertainty vs distance
        plt.figure(figsize=(16, 9))
        plt.scatter(
            self._cramer_rao_bounds["dist"],
            d_Omega * (RADIAN_TO_DEGREE) ** 2,
        )

        plt.xlabel("distance in Gpc")
        plt.ylabel("d_Omega in deg^2")
        plt.yscale("log")
        plt.savefig(
            f"{figures_directory}plots/sky_localization_uncertainty_distance.png",
            dpi=300,
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

    def evaluate_snr_analysis(self) -> None:
        # ensure directory is given
        figures_directory = f"evaluation/"
        if not os.path.isdir(figures_directory):
            os.makedirs(figures_directory)
        if not os.path.isdir(figures_directory + "snr_analysis/"):
            os.makedirs(figures_directory + "snr_analysis/")

        figures_directory = f"evaluation/snr_analysis/"

        # easy check SNR vs observation time averaged
        for name, parameter_set in self._snr_analysis_file.groupby(["M"]):
            if len(parameter_set) != 5:
                continue
            final_SNR = parameter_set["SNR"].iloc[-1]
            plt.plot(parameter_set["T"], parameter_set["SNR"] / final_SNR)

        plt.xlabel("observation time T in years")
        plt.ylabel("normalized SNR")
        plt.savefig(f"evaluation/plots/SNR_observation_time.png")

        for name, parameter_set in self._snr_analysis_file.groupby(["M"]):
            if len(parameter_set) != 5:
                continue
            final_generation_time = parameter_set["generation_time"].iloc[-1]
            plt.plot(
                parameter_set["T"],
                parameter_set["generation_time"] / final_generation_time,
            )

        plt.xlabel("observation time T in years")
        plt.ylabel("normalized generation time")
        plt.savefig(f"evaluation/plots/generation_time_observation_time.png")


def _compute_skylocalization_uncertainty(
    theta, var_theta, var_phi, cov_theta_phi
) -> float:
    return (
        2
        * np.pi
        * np.abs(np.sin(theta))
        * np.sqrt(var_phi * var_theta - (cov_theta_phi) ** 2)
    )


@np.vectorize
def _remove_zeros_from_grid(x) -> float:
    if x == 0:
        return 1.0
    return x
