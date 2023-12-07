import numpy as np
import pandas as pd
import typing
import os
import json
import time
import matplotlib.pyplot as plt
import logging
import sys
import cupy as cp
import cupyx.scipy.fft as cufft
from master_thesis_code.exceptions import ParameterEstimationError

from enum import Enum
from few.waveform import GenerateEMRIWaveform

from master_thesis_code.decorators import timer_decorator, if_plotting_activated
from master_thesis_code.constants import (
    REAL_PART, IMAGINARY_PART, SIMULATION_PATH, SIMULATION_CONFIGURATION_FILE, DEFAULT_SIMULATION_PATH, CRAMER_RAO_BOUNDS_PATH, MINIMAL_FREQUENCY, MAXIMAL_FREQUENCY)
from master_thesis_code.datamodels.parameter_space import ParameterSpace
from master_thesis_code.LISA_configuration import LISAConfiguration

_LOGGER = logging.getLogger()

class WaveGeneratorType(Enum):
    schwarzschild_fully_relativistic = 1
    pn5 = 2

# CONFIGURATION FOR SCHWARZSCHILD ECCENTRIC FLUX WWAVEFORM GENERATOR

# keyword arguments for inspiral generator (RunSchwarzEccFluxInspiral)
inspiral_kwargs={
    "DENSE_STEPPING": 0,  # we want a sparsely sampled trajectory
    "max_init_len": int(1e5),  # all of the trajectories will be well under len = 1000
}

# keyword arguments for inspiral generator (RomanAmplitude)
amplitude_kwargs = {
    "max_init_len": int(1e5),  # all of the trajectories will be well under len = 1000
    "use_gpu": True  # GPU is available in this class
}

# keyword arguments for Ylm generator (GetYlms)
Ylm_kwargs = {
    "assume_positive_m": False  # if we assume positive m, it will generate negative m for all m>0
}

# keyword arguments for summation generator (InterpolatedModeSum)
sum_kwargs = {
    "use_gpu": True,  # GPU is available for this type of summation
    "pad_output": False,
}

# CONFIGURATION FOR PN5 WAVEFORM GENERATOR

class ParameterEstimation():
    parameter_space: ParameterSpace
    waveform_generator: GenerateEMRIWaveform # generate waveform in SSB frame
    lisa_configuration: LISAConfiguration  # can be used to transform the waveform into the rotating detector frame
    dt: float = 5
    T: float = 3
    waveform_generation_time: int = 0

    def __init__(self, wave_generation_type: WaveGeneratorType, use_gpu: bool):
        self.parameter_space = ParameterSpace()
        self._use_gpu = use_gpu
        if wave_generation_type == "FastSchwarzschildEccentricFlux":
            self.waveform_generator = GenerateEMRIWaveform(
                waveform_class="FastSchwarzschildEccentricFlux",
                use_gpu=use_gpu
            )
            _LOGGER.info("Parameter estimation is setup up with the 'FastSchwarzschildEccentricFlux' wave generator.")
        elif wave_generation_type == WaveGeneratorType.pn5:
            self.waveform_generator = GenerateEMRIWaveform(
                waveform_class="Pn5AAKWaveform",
                inspiral_kwargs=inspiral_kwargs,
                sum_kwargs=sum_kwargs,
                frame="detector",
                use_gpu=use_gpu
            )
            _LOGGER.info("Parameter estimation is setup up with the 'PN5AAKwaveform' wave generator.")
        else:
            raise ParameterEstimationError(
                "Wave generator class could not be matched to FastSchwarzschildEccentricFlux or PN5AAKwaveform." 
                "please check configuration in main."
                )
        self.lisa_configuration = LISAConfiguration(parameter_space=self.parameter_space, dt=self.dt)
    
    @timer_decorator
    def generate_waveform(self, update_parameters: dict = {}, use_antenna_pattern_functions: bool = True) -> cp.ndarray:
        waveform = self.waveform_generator(
            **(self.parameter_space._parameters_to_dict() | update_parameters),
            dt=self.dt,
            T=self.T)

        if use_antenna_pattern_functions:
            return_waveform = self.lisa_configuration.transform_from_ssb_to_lisa_frame(waveform=waveform)
        else:
            return_waveform = waveform.real**2 + waveform.imag**2
        del waveform
        return return_waveform

    @timer_decorator
    def finite_difference(self,  waveform: cp.ndarray, parameter_symbol: str) -> cp.ndarray:
        """Compute (numerically) partial derivative of the currently set parameters w.r.t. the provided parameter.

        Args:
            parameter_symbol (str): parameter w.r.t. which the derivative is taken (Note: symbol string has to coincide with that in the ParameterSpace list!)

        Returns:
            cp.array[float]: data series of derivative
        """
        derivative_parameter_configuration = next(
            (parameter for parameter in self.parameter_space.parameters_configuration if parameter.symbol == parameter_symbol), None)
        
        if derivative_parameter_configuration is None:
            raise ParameterEstimationError(
                f"The provided derivative parameter symbol {parameter_symbol} does not match any defined parameter in the parameter space."
                )

        # save current parameter value
        parameter_evaluated_at = getattr(self.parameter_space, parameter_symbol)

        derivative_epsilon = derivative_parameter_configuration.derivative_epsilon

        setattr(self.parameter_space, parameter_symbol, parameter_evaluated_at + derivative_epsilon)

        neighbouring_waveform = self.generate_waveform()

        self._plot_waveform(waveforms=[waveform, neighbouring_waveform], plot_name="waveform_for_derivative")

        # set parameter back to evaluated value
        setattr(self.parameter_space, parameter_symbol, parameter_evaluated_at)

        waveform, neighbouring_waveform = self._crop_to_same_length([waveform, neighbouring_waveform])

        waveform_derivative = (neighbouring_waveform - waveform)/derivative_epsilon
        self._plot_waveform(waveforms=[waveform_derivative], plot_name=f"{parameter_symbol}_derivative")
        _LOGGER.info(f"Finished computing partial derivative of the waveform w.r.t. {parameter_symbol}.")
        del neighbouring_waveform
        del waveform
        return waveform_derivative

    @staticmethod
    def _crop_to_same_length(signals: typing.List[cp.array]) -> typing.List[cp.array]:
        minimal_length = min([len(signal) for signal in signals])
        signals = [signal[:minimal_length] for signal in signals]
        return signals

    def five_point_stencil_derivative(self, parameter_symbol: str) -> cp.array:
        """Compute (numerically) partial derivative of the currently set parameters w.r.t. the provided parameter.

        Args:
            parameter_symbol (str): parameter w.r.t. which the derivative is taken (Note: symbol string has to coincide with that in the ParameterSpace list!)

        Returns:
            cp.array[float]: data series of derivative
        """
        derivative_parameter_configuration = next(
            (parameter for parameter in self.parameter_space.parameters_configuration if parameter.symbol == parameter_symbol), None)
        
        if derivative_parameter_configuration is None:
            _LOGGER.error(f"The provided derivative parameter symbol {parameter_symbol} does not match any defined parameter in the parameter space.")
            sys.exit()
        parameter_evaluated_at = getattr(self.parameter_space, parameter_symbol)
        derivative_epsilon = derivative_parameter_configuration.derivative_epsilon
        five_stencil_points = [{parameter_symbol: parameter_evaluated_at + step*derivative_epsilon} for step in [-2., -1., 1., 2.]]

        waveforms = [self.generate_waveform(update_parameters=params) for params in five_stencil_points]
        waveforms = self._crop_to_same_length(waveforms)

        #self._plot_waveform(waveforms=waveforms, plot_name="waveform_for_derivative")

        waveform_derivative = (-waveforms[3] + 8*waveforms[2] - 8*waveforms[1] + waveforms[0])/12/derivative_epsilon
        #self._plot_waveform(waveforms=[waveform_derivative], plot_name=f"{parameter_symbol}_derivative")
        _LOGGER.info(f"Finished computing partial derivative of the waveform w.r.t. {parameter_symbol}.")
        del waveforms
        return waveform_derivative

    @if_plotting_activated
    def _plot_waveform(
            self, 
            waveforms: typing.List, 
            parameter_symbol: str = None, 
            plot_name: str = "", 
            x_label: str = "t [s]", 
            xs: cp.array = None,
            use_log_scale: bool = False) -> None:
        waveforms = [cp.asnumpy(waveform) for waveform in waveforms]

        figures_directory = f"saved_figures/waveforms/"

        if not os.path.isdir(figures_directory):
            os.makedirs(figures_directory)

        parameter_value = ""

        if parameter_symbol is not None:
            parameter_value = str(np.round(getattr(self.parameter_space, parameter_symbol),3))

        # create plots
        # plot power spectral density
        if xs is None:
            xs = np.array([index*self.dt for index in range(len(waveforms[0]))])


        if use_log_scale:
            indices = np.round(np.geomspace(1, xs.shape[0], 1000)).astype(int)[:-1]
        else:
            indices = np.round(np.linspace(0, xs.shape[0], 1000)).astype(int)[:-1]

        fig = plt.figure(figsize = (12, 8))
        for index, waveform in enumerate(waveforms):
            plt.plot(
                xs[indices], 
                waveform[indices],
                '-',
                linewidth = 0.5,
                label = f"{plot_name} {parameter_value}")

        plt.xlabel(x_label)
        if use_log_scale:
            plt.xscale("log")
            plt.yscale("log")
        plt.legend()
        plt.savefig(figures_directory + f"{plot_name}.png", dpi=300)
        plt.close()

    @timer_decorator
    def scalar_product_of_functions(self, a: cp.ndarray, b: cp.ndarray) -> float:
        fs = cufft.rfftfreq(a.__len__(), self.dt)[1:]
        a_fft = cufft.rfft(a)[1:]
        b_fft_cc = cp.conjugate(cufft.rfft(b))[1:]
        power_spectral_density = self.lisa_configuration.power_spectral_density(frequencies=fs)
        
        # crop all arrays to shortest length
        reduced_length = min(a_fft.shape[0], b_fft_cc.shape[0], fs.shape[0])
        a_fft = a_fft[:reduced_length]
        b_fft_cc = b_fft_cc[:reduced_length]
        fs = fs[:reduced_length]
        power_spectral_density = power_spectral_density[:reduced_length]

        integrant = cp.divide(cp.multiply(a_fft, b_fft_cc), power_spectral_density)

        #self._plot_waveform(waveforms=[integrant.real], xs=fs, plot_name="scalar_product_integrant_real", x_label="f [Hz]", use_log_scale=True)

        fs, integrant = self._crop_frequency_domain(fs, integrant)

        #self._plot_waveform(waveforms=[integrant.real], xs=fs, plot_name="scalar_product_integrant_real_cropped", x_label="f [Hz]", use_log_scale=True)
        
        result = 4*cp.trapz(y=integrant, x=fs).real
        del fs
        del a_fft
        del b_fft_cc
        del power_spectral_density
        del integrant
        del a
        del b 
        return result

    @staticmethod
    @timer_decorator
    def _crop_frequency_domain(fs: cp.array, integrant: cp.array) -> tuple:
        if len(fs) != len(integrant):
            _LOGGER.warning("length of frequency domain and integrant are not equal.")

        # find lowest frequency
        lower_limit_index = cp.argmax(fs >= MINIMAL_FREQUENCY)
        upper_limit_index = cp.argmax(fs >= MAXIMAL_FREQUENCY)
        if upper_limit_index == 0:
            upper_limit_index = len(fs)
        return fs[lower_limit_index:upper_limit_index], integrant[lower_limit_index:upper_limit_index]

    @timer_decorator
    def compute_fisher_information_matrix(self) -> cp.ndarray:
        # compute derivatives for fisher information matrix
        waveform_derivatives = {}

        parameter_symbol_list = [parameter.symbol for parameter in self.parameter_space.parameters_configuration]

        for parameter_symbol in parameter_symbol_list:
            waveform_derivative = self.five_point_stencil_derivative(parameter_symbol=parameter_symbol)
            waveform_derivatives[parameter_symbol] = waveform_derivative
            del waveform_derivative

        fisher_information_matrix = cp.zeros(
            shape=(len(parameter_symbol_list), len(parameter_symbol_list)),
            dtype=float)

        for col, column_parameter_symbol in enumerate(parameter_symbol_list):
            for row, row_parameter_symbol in enumerate(parameter_symbol_list):
                fisher_information_matrix_element = self.scalar_product_of_functions(
                    waveform_derivatives[column_parameter_symbol], 
                    waveform_derivatives[row_parameter_symbol])
                fisher_information_matrix[col][row] = fisher_information_matrix_element    
        
        _LOGGER.info("Fisher information matrix has been computed.")
        del waveform_derivatives
        return fisher_information_matrix
    
    @timer_decorator
    def compute_Cramer_Rao_bounds(self) -> dict:

        fisher_information_matrix = self.compute_fisher_information_matrix()

        cramer_rao_bounds = np.matrix(cp.asnumpy(fisher_information_matrix)).I
        _LOGGER.debug("matrix inversion completed.")
        parameter_symbol_list = [parameter.symbol for parameter in self.parameter_space.parameters_configuration]

        independent_cramer_rao_bounds = {
            f"delta_{parameter_symbol_list[row]}_delta_{parameter_symbol_list[column]}": cramer_rao_bounds[row, column]
            for row in range(len(parameter_symbol_list)) for column in range(row+1)}
        
        _LOGGER.info("Finished computing Cramer Rao bounds.")
        del fisher_information_matrix
        del cramer_rao_bounds
        return independent_cramer_rao_bounds

    @timer_decorator
    def save_cramer_rao_bound(self, cramer_rao_bound_dictionary: dict, snr: float) -> None:
        try:
            cramer_rao_bounds = pd.read_csv(CRAMER_RAO_BOUNDS_PATH)

        except FileNotFoundError:
            parameters_list = list(self.parameter_space._parameters_to_dict().keys())
            parameters_list.extend(list(cramer_rao_bound_dictionary.keys()))
            parameters_list.extend(["T", "dt", "SNR", "generation_time"])
            cramer_rao_bounds = pd.DataFrame(columns=parameters_list)

        new_cramer_rao_bounds_dict = self.parameter_space._parameters_to_dict() | cramer_rao_bound_dictionary
        new_cramer_rao_bounds_dict = new_cramer_rao_bounds_dict | {"T": self.T, "dt": self.dt, "SNR": snr, "generation_time": self.waveform_generation_time}

        new_cramer_rao_bounds = pd.DataFrame([new_cramer_rao_bounds_dict])

        cramer_rao_bounds = pd.concat([cramer_rao_bounds, new_cramer_rao_bounds], ignore_index=True)
        cramer_rao_bounds.to_csv(CRAMER_RAO_BOUNDS_PATH, index=False)
        _LOGGER.info(f"Saved current Cramer-Rao bound to {CRAMER_RAO_BOUNDS_PATH}")
        del cramer_rao_bound_dictionary
        del cramer_rao_bounds
        del new_cramer_rao_bounds
        del new_cramer_rao_bounds_dict
    

    def _visualize_cramer_rao_bounds(self) -> None:
        mean_errors_data = pd.read_csv(CRAMER_RAO_BOUNDS_PATH)
        parameter_columns = [column_name for column_name in mean_errors_data.columns if "delta" not in column_name]

        # ensure directory is given
        figures_directory = f"saved_figures/parameter_estimation/"
        if not os.path.isdir(figures_directory):
            os.makedirs(figures_directory)

        # 3d plot of coverage in the configuration space of M, theta_S and phi_S
        M_configuration = next((config for config in self.parameter_space.parameters_configuration if config.symbol == "M"), None)
        qS_configuration = next((config for config in self.parameter_space.parameters_configuration if config.symbol == "qS"), None)
        phiS_configuration = next((config for config in self.parameter_space.parameters_configuration if config.symbol == "phiS"), None)

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
        #plt.show()

        # create plots for error correlation
        for column_name in parameter_columns:
            fig, (ax1, ax2) = plt.subplots(1,2)
            ax1.plot(mean_errors_data[column_name], 
                    np.sqrt(mean_errors_data["delta_M_delta_M"]),
                    '.',
                    label = f"bounds: delta M")

            ax2.plot(
                mean_errors_data[column_name], 
                np.sqrt(mean_errors_data["delta_qS_delta_qS"]),
                '.',
                label = f"bounds: qS")
            
            ax2.plot(
                mean_errors_data[column_name], 
                np.sqrt(mean_errors_data["delta_phiS_delta_phiS"]),
                '.',
                label = f"bounds: phiS")

            ax1.set_yscale("log")
            ax1.set_xlabel(f"{column_name}")
            ax2.set_yscale("log")
            ax2.set_xlabel(f"{column_name}")
            ax1.legend()
            ax2.legend()
            plt.savefig(figures_directory + f"mean_error_{column_name}_correlation.png", dpi=300)
            plt.close()

        # create plots for computation time correlation
        for column_name in parameter_columns:
            fig = plt.figure(figsize=(16,9))
            plt.plot(
                mean_errors_data[column_name],
                mean_errors_data["generation_time"],
                ".",
                label=f"simulation data")
            plt.xlabel(f"{column_name}")
            plt.ylabel("t [s]")
            plt.legend()
            plt.savefig(figures_directory +  f"waveform_generation_time_{column_name}_correlation.png", dpi=300)
            plt.close()

    @timer_decorator
    def compute_signal_to_noise_ratio(self) -> float:
        
        start = time.time()
        waveform = self.generate_waveform()
        end = time.time()
        self.waveform_generation_time = int(end-start)

        self.current_waveform = waveform
        return cp.sqrt(self.scalar_product_of_functions(a=waveform, b=waveform))

    @timer_decorator
    def check_parameter_dependency(self, parameter_symbol: str, steps: int = 5):
        _LOGGER.info(f"Start parameter dependency check for {parameter_symbol}.")
        parameter_configuration = next((config for config in self.parameter_space.parameters_configuration if config.symbol == parameter_symbol), None)

        self.parameter_space.a = 0. 
        self.parameter_space.x0 = 1.
        self.parameter_space.Phi_theta0 = 0.


        if parameter_configuration is None:
            raise ParameterEstimationError("check_parameter_dependency couldn't match parameter symbol.")

        parameter_steps = cp.linspace(parameter_configuration.lower_limit, parameter_configuration.upper_limit, steps)

        print(parameter_steps)

        delta_parameter = parameter_steps[1] - parameter_steps[0]

        waveforms = []

        # save current parameter value
        current_parameter_value = getattr(self.parameter_space, parameter_symbol)

        for i, parameter_step in enumerate(parameter_steps):
            setattr(self.parameter_space, parameter_symbol, parameter_step)

            if parameter_symbol == "phiS" and i > 0:
                new_phiK = self.parameter_space.phiK + delta_parameter
                self.parameter_space.phiK = new_phiK
            elif parameter_symbol =="qS":
                self.parameter_space.phiK = self.parameter_space.phiS
                self.parameter_space.qK = self.parameter_space.qS

            waveforms.append(self.generate_waveform(use_antenna_pattern_functions=False))
            _LOGGER.info(f"Parameter dependency for {parameter_symbol}: {i+1}/{len(parameter_steps)} waveforms generated.")

        self._plot_waveform(waveforms=waveforms, plot_name=f"dependency_{parameter_symbol}", use_log_scale=True)

        # set parameter value back to original one.
        setattr(self.parameter_space, parameter_symbol, current_parameter_value)
        _LOGGER.info(f"Finished parameter dependency check for {parameter_symbol}.")
