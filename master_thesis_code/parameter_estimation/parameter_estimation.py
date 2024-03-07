import warnings
import numpy as np
import pandas as pd
from typing import List
import os
import time
import matplotlib as mpl

mpl.rcParams["agg.path.chunksize"] = 1000
import matplotlib.pyplot as plt

import logging
import sys
import cupy as cp
import cupyx.scipy.fft as cufft

from fastlisaresponse import ResponseWrapper

from master_thesis_code.exceptions import (
    ParameterEstimationError,
    ParameterOutOfBoundsError,
)
from master_thesis_code.waveform_generator import (
    create_lisa_response_generator,
    WaveGeneratorType,
)
from master_thesis_code.decorators import timer_decorator, if_plotting_activated
from master_thesis_code.constants import (
    SNR_ANALYSIS_PATH,
    CRAMER_RAO_BOUNDS_PATH,
    MINIMAL_FREQUENCY,
    MAXIMAL_FREQUENCY,
    ESA_TDI_CHANNELS,
)
from master_thesis_code.datamodels.parameter_space import ParameterSpace, Parameter
from master_thesis_code.LISA_configuration import LisaTdiConfiguration

_LOGGER = logging.getLogger()


class ParameterEstimation:
    parameter_space: ParameterSpace
    lisa_response_generator: ResponseWrapper
    dt = 10  # time sampling in sec
    T = 5  # observation time in years

    def __init__(
        self,
        waveform_generation_type: WaveGeneratorType,
        parameter_space: ParameterSpace,
    ):
        self.parameter_space = parameter_space
        self.lisa_response_generator = create_lisa_response_generator(
            waveform_generation_type,
            self.dt,
            self.T,
        )
        self.lisa_configuration = LisaTdiConfiguration()
        _LOGGER.info("parameter estimation initialized.")

    @timer_decorator
    def generate_lisa_response(self) -> List:
        return self.lisa_response_generator(
            *self.parameter_space._parameters_to_dict().values(), T=self.T
        )

    @timer_decorator
    def five_point_stencil_derivative(self, parameter: Parameter) -> cp.array:
        """Compute (numerically) partial derivative of the currently set parameters w.r.t. the provided parameter.

        Args:
            parameter_symbol (str): parameter w.r.t. which the derivative is taken (Note: symbol string has to coincide with that in the ParameterSpace list!)

        Returns:
            cp.array[float]: data series of derivative
        """

        parameter_evaluated_at = parameter
        derivative_epsilon = parameter.derivative_epsilon

        # check that neighboring points are in parameter range as well
        if (
            (parameter_evaluated_at.value - 2 * derivative_epsilon)
            < parameter.lower_limit
        ) or (
            (parameter_evaluated_at.value + 2 * derivative_epsilon)
            > parameter.upper_limit
        ):
            raise ParameterOutOfBoundsError(
                "Tried to set parameter to value out of bounds in derivative."
            )

        five_point_stencil_steps = [-2.0, -1.0, 1.0, 2.0]
        lisa_responses = []
        for step in five_point_stencil_steps:
            parameter.value = parameter_evaluated_at.value + step * derivative_epsilon
            setattr(
                self.parameter_space,
                parameter.symbol,
                parameter,
            )
            lisa_responses.append(self.generate_lisa_response())
        lisa_responses = self._crop_to_same_length(lisa_responses)

        # set parameter value back to value that the derivative was evaluated at.
        setattr(
            self.parameter_space,
            parameter.symbol,
            parameter_evaluated_at,
        )

        lisa_response_derivative = (
            (
                -lisa_responses[3]
                + 8 * lisa_responses[2]
                - 8 * lisa_responses[1]
                + lisa_responses[0]
            )
            / 12
            / derivative_epsilon
        )

        _LOGGER.info(
            f"Finished computing partial derivative of the waveform w.r.t. {parameter.symbol}."
        )
        del lisa_responses
        return lisa_response_derivative

    @staticmethod
    def _crop_to_same_length(
        signal_collection: List[List[cp.array]],
    ) -> List[List[cp.array]]:
        max_possible_length = min(
            min(
                [
                    min(
                        [len(tdi_channel) for tdi_channel in tdi_channels]
                        for tdi_channels in signal_collection
                    )
                ]
            )
        )
        return cp.array(
            [
                [tdi_channel[:max_possible_length] for tdi_channel in tdi_channels]
                for tdi_channels in signal_collection
            ]
        )

    def scalar_product_of_functions(
        self, tdi_channels_a: cp.ndarray, tdi_channels_b: cp.ndarray
    ) -> float:
        result = 0
        for channel, tdi_channel_a, tdi_channel_b in zip(
            ESA_TDI_CHANNELS, tdi_channels_a, tdi_channels_b
        ):
            fs = cufft.rfftfreq(len(tdi_channel_a), self.dt)[1:]
            a_fft = cufft.rfft(tdi_channel_a)[1:]
            b_fft_cc = cp.conjugate(cufft.rfft(tdi_channel_b))[1:]
            power_spectral_density = self.lisa_configuration.power_spectral_density(
                frequencies=fs, channel=channel
            )

            # crop all arrays to shortest length
            reduced_length = min(a_fft.shape[0], b_fft_cc.shape[0], fs.shape[0])
            a_fft = a_fft[:reduced_length]
            b_fft_cc = b_fft_cc[:reduced_length]
            fs = fs[:reduced_length]
            power_spectral_density = power_spectral_density[:reduced_length]

            integrant = cp.divide(cp.multiply(a_fft, b_fft_cc), power_spectral_density)
            fs, integrant = self._crop_frequency_domain(fs, integrant)

            # plt.plot(cp.asnumpy(fs).real, cp.asnumpy(integrant).real)

            result += 4 * cp.trapz(y=integrant, x=fs).real
            # _LOGGER.debug(f"current scalar product result: {result}")
        # plt.xscale("log")
        # plt.yscale("log")
        # plt.savefig("scalar_product_integrants.png", dpi=300)
        # plt.close()
        del fs
        del a_fft
        del b_fft_cc
        del power_spectral_density
        del integrant
        del tdi_channels_a
        del tdi_channels_b
        return result

    @staticmethod
    def _crop_frequency_domain(fs: cp.array, integrant: cp.array) -> tuple:
        if len(fs) != len(integrant):
            _LOGGER.warning("length of frequency domain and integrant are not equal.")

        # find lowest frequency
        lower_limit_index = cp.argmax(fs >= MINIMAL_FREQUENCY)
        upper_limit_index = cp.argmax(fs >= MAXIMAL_FREQUENCY)
        if upper_limit_index == 0:
            upper_limit_index = len(fs)
        return (
            fs[lower_limit_index:upper_limit_index],
            integrant[lower_limit_index:upper_limit_index],
        )

    def compute_fisher_information_matrix(self) -> cp.ndarray:
        # compute derivatives for fisher information matrix
        lisa_response_derivatives = {}

        parameter_symbol_list = list(self.parameter_space._parameters_to_dict().keys())

        for parameter_symbol in parameter_symbol_list:
            lisa_response_derivative = self.five_point_stencil_derivative(
                parameter=getattr(self.parameter_space, parameter_symbol)
            )
            lisa_response_derivatives[parameter_symbol] = lisa_response_derivative
            del lisa_response_derivative

        fisher_information_matrix = cp.zeros(
            shape=(len(parameter_symbol_list), len(parameter_symbol_list)), dtype=float
        )

        for col, column_parameter_symbol in enumerate(parameter_symbol_list):
            for row, row_parameter_symbol in enumerate(parameter_symbol_list):
                fisher_information_matrix_element = self.scalar_product_of_functions(
                    lisa_response_derivatives[column_parameter_symbol],
                    lisa_response_derivatives[row_parameter_symbol],
                )
                fisher_information_matrix[col][row] = fisher_information_matrix_element

        _LOGGER.info("Fisher information matrix has been computed.")
        del lisa_response_derivatives
        return fisher_information_matrix

    @timer_decorator
    def compute_Cramer_Rao_bounds(self) -> dict:
        fisher_information_matrix = self.compute_fisher_information_matrix()

        cramer_rao_bounds = np.matrix(cp.asnumpy(fisher_information_matrix)).I
        _LOGGER.debug("matrix inversion completed.")
        parameter_symbol_list = list(self.parameter_space._parameters_to_dict().keys())

        independent_cramer_rao_bounds = {
            f"delta_{parameter_symbol_list[row]}_delta_{parameter_symbol_list[column]}": cramer_rao_bounds[
                row, column
            ]
            for row in range(len(parameter_symbol_list))
            for column in range(row + 1)
        }

        _LOGGER.info("Finished computing Cramer Rao bounds.")
        del fisher_information_matrix
        del cramer_rao_bounds
        return independent_cramer_rao_bounds

    def compute_signal_to_noise_ratio(self) -> float:
        start = time.time()
        waveform = self.generate_lisa_response()
        end = time.time()
        self.waveform_generation_time = round(end - start, 3)

        """
        for i, channel in enumerate(waveform):
            plt.plot(cp.asnumpy(channel), label=str(i))
        plt.savefig("channels.png", dpi=300)
        plt.close()
        """

        self.current_waveform = waveform
        snr = cp.sqrt(self.scalar_product_of_functions(waveform, waveform))
        del waveform
        return snr

    def save_cramer_rao_bound(
        self, cramer_rao_bound_dictionary: dict, snr: float, host_galaxy_index: int
    ) -> None:
        try:
            cramer_rao_bounds = pd.read_csv(CRAMER_RAO_BOUNDS_PATH)

        except FileNotFoundError:
            parameters_list = list(self.parameter_space._parameters_to_dict().keys())
            parameters_list.extend(list(cramer_rao_bound_dictionary.keys()))
            parameters_list.extend(
                ["T", "dt", "SNR", "generation_time", "host_galaxy_index"]
            )
            cramer_rao_bounds = pd.DataFrame(columns=parameters_list)

        new_cramer_rao_bounds_dict = (
            self.parameter_space._parameters_to_dict() | cramer_rao_bound_dictionary
        )
        new_cramer_rao_bounds_dict = new_cramer_rao_bounds_dict | {
            "T": self.T,
            "dt": self.dt,
            "SNR": snr,
            "generation_time": self.waveform_generation_time,
            "host_galaxy_index": host_galaxy_index,
        }

        new_cramer_rao_bounds = pd.DataFrame([new_cramer_rao_bounds_dict])

        cramer_rao_bounds = pd.concat(
            [cramer_rao_bounds, new_cramer_rao_bounds], ignore_index=True
        )
        cramer_rao_bounds.to_csv(CRAMER_RAO_BOUNDS_PATH, index=False)
        _LOGGER.info(f"Saved current Cramer-Rao bound to {CRAMER_RAO_BOUNDS_PATH}")
        del cramer_rao_bound_dictionary
        del cramer_rao_bounds
        del new_cramer_rao_bounds
        del new_cramer_rao_bounds_dict

    def _visualize_cramer_rao_bounds(self) -> None:
        mean_errors_data = pd.read_csv(CRAMER_RAO_BOUNDS_PATH)
        parameter_columns = [
            column_name
            for column_name in mean_errors_data.columns
            if "delta" not in column_name
        ]

        # ensure directory is given
        figures_directory = f"saved_figures/parameter_estimation/"
        if not os.path.isdir(figures_directory):
            os.makedirs(figures_directory)

        # 3d plot of coverage in the configuration space of M, theta_S and phi_S
        M_configuration = self.parameter_space.M
        qS_configuration = self.parameter_space.qS
        phiS_configuration = self.parameter_space.phiS

        x1 = mean_errors_data["M"]
        y1 = mean_errors_data["qS"]
        z1 = mean_errors_data["phiS"]

        plt.figure(figsize=(16, 9))
        axes = plt.axes(projection="3d")
        axes.scatter3D(x1, y1, z1)

        axes.set_xlabel("M")
        axes.set_ylabel("qS")
        axes.set_zlabel("phiS")

        axes.set_xlim(M_configuration.lower_limit, M_configuration.upper_limit)
        axes.set_ylim(qS_configuration.lower_limit, qS_configuration.upper_limit)
        axes.set_zlim(phiS_configuration.lower_limit, phiS_configuration.upper_limit)
        plt.savefig(figures_directory + "coverage_parameter_space.png", dpi=300)
        # plt.show()

        # create plots for error correlation
        for column_name in parameter_columns:
            fig, (ax1, ax2) = plt.subplots(1, 2)
            ax1.plot(
                mean_errors_data[column_name],
                np.sqrt(mean_errors_data["delta_M_delta_M"]),
                ".",
                label=f"bounds: delta M",
            )

            ax2.plot(
                mean_errors_data[column_name],
                np.sqrt(mean_errors_data["delta_qS_delta_qS"]),
                ".",
                label=f"bounds: qS",
            )

            ax2.plot(
                mean_errors_data[column_name],
                np.sqrt(mean_errors_data["delta_phiS_delta_phiS"]),
                ".",
                label=f"bounds: phiS",
            )

            ax1.set_yscale("log")
            ax1.set_xlabel(f"{column_name}")
            ax2.set_yscale("log")
            ax2.set_xlabel(f"{column_name}")
            ax1.legend()
            ax2.legend()
            plt.savefig(
                figures_directory + f"mean_error_{column_name}_correlation.png", dpi=300
            )
            plt.close()

        # create plots for computation time correlation
        for column_name in parameter_columns:
            fig = plt.figure(figsize=(16, 9))
            plt.plot(
                mean_errors_data[column_name],
                mean_errors_data["generation_time"],
                ".",
                label=f"simulation data",
            )
            plt.xlabel(f"{column_name}")
            plt.ylabel("t [s]")
            plt.legend()
            plt.savefig(
                figures_directory
                + f"waveform_generation_time_{column_name}_correlation.png",
                dpi=300,
            )
            plt.close()

    def SNR_analysis(self) -> None:
        # setup waveformgenerators for different observation times
        waveform_generators = {
            0: create_lisa_response_generator(WaveGeneratorType.PN5_AAK, self.dt, 0.5),
            1: create_lisa_response_generator(WaveGeneratorType.PN5_AAK, self.dt, 1),
            2: create_lisa_response_generator(WaveGeneratorType.PN5_AAK, self.dt, 2),
            3: create_lisa_response_generator(WaveGeneratorType.PN5_AAK, self.dt, 3),
            4: create_lisa_response_generator(WaveGeneratorType.PN5_AAK, self.dt, 5),
        }
        parameter_set_index = 0
        for _ in range(200):
            self.parameter_space.randomize_parameters()
            for T, waveform_generator in zip(
                [0.5, 1, 2, 3, 5], waveform_generators.values()
            ):
                self.lisa_response_generator = waveform_generator
                self.T = T
                try:
                    warnings.filterwarnings("error")
                    snr = self.compute_signal_to_noise_ratio()
                    warnings.resetwarnings()
                except Warning as e:
                    if "Mass ratio" in str(e):
                        _LOGGER.warning(
                            "Caught warning that mass ratio is out of bounds. Continue with new parameters..."
                        )
                        continue
                    else:
                        _LOGGER.warning(f"{str(e)}. Continue with new parameters...")
                        continue
                except ParameterOutOfBoundsError as e:
                    _LOGGER.warning(
                        f"Caught ParameterOutOfBoundsError during parameter estimation: {str(e)}. Continue with new parameters..."
                    )
                    continue
                except AssertionError as e:
                    _LOGGER.warning(
                        f"caught AssertionError: {str(e)}. Continue with new parameters..."
                    )
                    continue
                except RuntimeError as e:
                    _LOGGER.warning(
                        f"Caught RuntimeError during waveform generation : {str(e)} .\n Continue with new parameters..."
                    )
                    continue
                except ValueError as e:
                    if "EllipticK" in str(e):
                        _LOGGER.warning(
                            "Caught EllipticK error from waveform generator. Continue with new parameters..."
                        )
                        continue
                    elif "Brent root solver does not converge" in str(e):
                        _LOGGER.warning(
                            "Caught brent root solver error because it did not converge. Continue with new parameters..."
                        )
                        continue
                    else:
                        raise ValueError(e)

                self.save_snr_analysis(snr, parameter_set_index)
            parameter_set_index += 1

    def save_snr_analysis(self, snr: float, parameter_set_index: int) -> None:
        try:
            snr_analysis = pd.read_csv(SNR_ANALYSIS_PATH)

        except FileNotFoundError:
            parameters_list = list(self.parameter_space._parameters_to_dict().keys())
            parameters_list.extend(["T", "SNR", "generation_time"])
            snr_analysis = pd.DataFrame(columns=parameters_list)

        new_snr_analysis_dict = self.parameter_space._parameters_to_dict() | {
            "T": self.T,
            "dt": self.dt,
            "SNR": snr,
            "generation_time": self.waveform_generation_time,
            "parameter_set_index": parameter_set_index,
        }

        new_snr_analysis = pd.DataFrame([new_snr_analysis_dict])

        snr_analysis = pd.concat([snr_analysis, new_snr_analysis], ignore_index=True)
        snr_analysis.to_csv(SNR_ANALYSIS_PATH, index=False)
