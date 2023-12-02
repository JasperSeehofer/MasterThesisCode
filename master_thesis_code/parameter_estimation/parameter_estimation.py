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
from scipy.fft import rfft, rfftfreq, set_global_backend
set_global_backend(cufft)

from enum import Enum
from few.waveform import GenerateEMRIWaveform

from master_thesis_code.decorators import timer_decorator
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
    "max_init_len": int(1e3),  # all of the trajectories will be well under len = 1000
}

# keyword arguments for inspiral generator (RomanAmplitude)
amplitude_kwargs = {
    "max_init_len": int(1e3),  # all of the trajectories will be well under len = 1000
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
    dt: float = 10
    T: float = 1
    M_derivative_steps: int = 1000
    M_steps: list[float] = []
    waveform_generation_time: int = 0
    current_waveform: cp.array = None

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
                frame="detector",
                sum_kwargs=sum_kwargs,
                use_gpu=use_gpu
            )
            _LOGGER.info("Parameter estimation is setup up with the 'PN5AAKwaveform' wave generator.")
        else:
            _LOGGER.error("Wave generator class could not be matched to FastSchwarzschildEccentricFlux or PN5AAKwaveform, please check configuration in main.")
            sys.exit()
        self.lisa_configuration = LISAConfiguration(parameter_space=self.parameter_space, dt=self.dt)
    
    @timer_decorator
    def generate_waveform(self, use_antenna_pattern_functions: bool = True) -> cp.ndarray:
        waveform = self.waveform_generator(
            **self.parameter_space._parameters_to_dict(),
            dt=self.dt,
            T=self.T)

        if use_antenna_pattern_functions:
            return_waveform = self.lisa_configuration.transform_from_ssb_to_lisa_frame(waveform=waveform)
        else:
            return_waveform = waveform.real**2 + waveform.imag**2
        return return_waveform

    def multiple_numeric_M_derivative(self) -> typing.Tuple[any, pd.DataFrame, pd.DataFrame]:
        M_configuration = next(
            (parameter_configuration 
            for parameter_configuration in self.parameter_space.parameters_configuration
            if parameter_configuration.symbol == "M"), 
            None)
        
        if M_configuration is None:
            _LOGGER.warning("Configuration of Black hole mass not given.")
            sys.exit()

        dM = (M_configuration.upper_limit - M_configuration.lower_limit)/self.M_derivative_steps

        self.M_steps = list(cp.arange(
            start=M_configuration.lower_limit,
            stop=M_configuration.upper_limit,
            step=dM))
        
        waveforms_M = pd.DataFrame()

        for count, M in enumerate(self.M_steps, 1):
            self.parameter_space.M = M
            waveform = self.generate_waveform()
            _LOGGER.info(f"{count}/{self.M_derivative_steps} waveforms generated.")

            column_indices = pd.MultiIndex.from_tuples(
                    [(REAL_PART, f"M_{count}"),
                    (IMAGINARY_PART, f"M_{count}")], 
                    names=["first", "second"])

            additional_columns = pd.DataFrame(data=cp.array([waveform.real, waveform.imag]).T, columns=column_indices)

            if count == 1:
                waveforms_M = additional_columns
            else:    
                waveforms_M = pd.concat([waveforms_M, additional_columns], axis=1)

        self.save_waveform(generated_waveform=waveforms_M)

        waveforms_derivatives_M = waveforms_M.diff(periods=-2, axis=1).div(dM)

        return waveforms_derivatives_M.dropna(axis=1 , how="all")
    
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
            _LOGGER.error(f"The provided derivative parameter symbol {parameter_symbol} does not match any defined parameter in the parameter space.")
            sys.exit()

        # save current parameter value
        parameter_evaluated_at = getattr(self.parameter_space, parameter_symbol)

        derivative_epsilon = derivative_parameter_configuration.derivative_epsilon

        setattr(self.parameter_space, parameter_symbol, parameter_evaluated_at + derivative_epsilon)

        neighbouring_waveform = self.generate_waveform()

        self._plot_waveform(waveforms=[waveform, neighbouring_waveform], plot_name="waveform_for_derivative")

        # set parameter back to evaluated value
        setattr(self.parameter_space, parameter_symbol, parameter_evaluated_at)

        waveform, neighbouring_waveform = self._crop_to_same_length(waveform, neighbouring_waveform)

        waveform_derivative = (neighbouring_waveform - waveform)/derivative_epsilon
        self._plot_waveform(waveforms=[waveform_derivative], plot_name=f"{parameter_symbol}_derivative")
        _LOGGER.info(f"Finished computing partial derivative of the waveform w.r.t. {parameter_symbol}.")
        return waveform_derivative

    @staticmethod
    def _crop_to_same_length(signal_1: cp.array, signal_2: cp.array) -> tuple:
        minimal_length = min(len(signal_1), len(signal_2))
        return signal_1[:minimal_length], signal_2[:minimal_length]

    def five_point_stencil_derivative(self, parameter_symbol: str) -> cp.ndarray:
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

        five_stencil_points = [parameter_evaluated_at + step*derivative_epsilon for step in [-2., -1., 1., 2.]]

        waveforms = []
        for parameter_value in five_stencil_points:
            setattr(self.parameter_space, parameter_symbol, parameter_value)
            waveform = self.generate_waveform()
            waveforms.append(waveform)

        self._plot_waveform(waveforms=waveforms, plot_name="waveform_for_derivative")


        # set parameter back to evaluated value
        setattr(self.parameter_space, parameter_symbol, parameter_evaluated_at)

        waveform_derivative = (-waveforms[3] + 8*waveforms[2] - 8*waveforms[1] + waveforms[0])/12/derivative_epsilon
        self._plot_waveform(waveforms=[waveform_derivative], plot_name=f"{parameter_symbol}_derivative")
        _LOGGER.info(f"Finished computing partial derivative of the waveform w.r.t. {parameter_symbol}.")
        return waveform_derivative

    def save_waveform(self, generated_waveform: pd.DataFrame) -> None:
        try:
            with open(SIMULATION_CONFIGURATION_FILE, "r") as file:
                simulation_configuration = json.load(file)
                simulation_path = simulation_configuration[SIMULATION_PATH]

        except FileNotFoundError:
            _LOGGER.warning(f"No simulation_configuration.json file in root directory. Will use default directory name: {DEFAULT_SIMULATION_PATH}.")
            simulation_path = DEFAULT_SIMULATION_PATH

        new_simulation_path = simulation_path
        counter = 1

        while os.path.isdir(new_simulation_path):
            new_simulation_path = f"{simulation_path}_{str(counter)}"
            counter += 1

        os.makedirs(new_simulation_path)

        with open(f"{new_simulation_path}/parameters.json", "w") as file:
            simulation_parameters = {
                "dt": self.dt,
                "T": self.T,
                "M_derivative_steps": self.M_derivative_steps,
                "M_steps": self.M_steps
            }
            json.dump(self.parameter_space._parameters_to_dict() | simulation_parameters, file)
        print(generated_waveform)
        generated_waveform.to_csv(f"{new_simulation_path}/waveform.csv")
    
    @timer_decorator
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

    def _plot_M_derivative(self, waveform_derivative_M: pd.DataFrame) -> None:
        
        # check directory and if it doesn't exit create it
        figures_directory = f"saved_figures/M_derivative_{self.M_derivative_steps}_steps/"

        if not os.path.isdir(figures_directory):
            os.makedirs(figures_directory)

        with open(f"{figures_directory}parameters.json", "w") as file:
            simulation_parameters = {
                "dt": self.dt,
                "T": self.T,
                "M_derivative_steps": self.M_derivative_steps,
                "M_steps": self.M_steps
            }
            json.dump(self.parameter_space._parameters_to_dict() | simulation_parameters, file)

        # create plots
        plt.figure(figsize = (12, 8))
        t_indices =  cp.round(cp.linspace(0, len(waveform_derivative_M.index) - 1, 10)).astype(int)

        for t_index in t_indices:

            plt.plot(self.M_steps[:-1], 
                    waveform_derivative_M.loc[t_index , REAL_PART], 
                    '-',
                    label = f"Re[dh(t)/dM](M) for {self.M_derivative_steps} steps")
            plt.plot(self.M_steps[:-1], 
                    waveform_derivative_M.loc[t_index , IMAGINARY_PART], 
                    '-',
                    label = f"Im[dh(t)/dM] for {self.M_derivative_steps} steps")
            
            plt.xlabel("M in solar masses")
            plt.legend()
            plt.savefig(figures_directory + f"t_approx_{int(t_index*self.dt/3600/24)}days.png", dpi=300)
            plt.clf()

    @timer_decorator
    def scalar_product_of_functions(self, a: cp.ndarray, b: cp.ndarray) -> float:

        fs = rfftfreq(len(a), self.dt)

        power_spectral_density = cp.array([self.lisa_configuration.power_spectral_density(f=f) for f in fs])

        a_fft = rfft(a)
        b_fft_cc = cp.conjugate(rfft(b))

        # crop all arrays to shortest length
        reduced_length = min(a_fft.shape[0], b_fft_cc.shape[0], fs.shape[0])
        a_fft = a_fft[:reduced_length]
        b_fft_cc = b_fft_cc[:reduced_length]
        fs = fs[:reduced_length]
        power_spectral_density = power_spectral_density[:reduced_length]

        integrant = cp.divide(cp.multiply(a_fft, b_fft_cc), power_spectral_density)

        self._plot_waveform(waveforms=[integrant.real], xs=fs, plot_name="scalar_product_integrant_real", x_label="f [Hz]", use_log_scale=True)

        fs, integrant = self._crop_frequency_domain(fs, integrant)

        self._plot_waveform(waveforms=[integrant.real], xs=fs, plot_name="scalar_product_integrant_real_cropped", x_label="f [Hz]", use_log_scale=True)

        return 4*cp.trapz(y=integrant, x=fs).real

    @staticmethod
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

        current_waveform = self.current_waveform

        parameter_symbol_list = [parameter.symbol for parameter in self.parameter_space.parameters_configuration]

        for parameter_symbol in parameter_symbol_list:
            waveform_derivative = self.finite_difference(waveform=current_waveform, parameter_symbol=parameter_symbol)
            waveform_derivatives[parameter_symbol] = waveform_derivative

        fisher_information_matrix: cp.empty(
            shape=(len(parameter_symbol_list)-1, len(parameter_symbol_list)-1),
            dtype=float)

        for col, column_parameter_symbol in enumerate(parameter_symbol_list):
            for row, row_parameter_symbol in enumerate(parameter_symbol_list):
                fisher_information_matrix_element = self.scalar_product_of_functions(
                    waveform_derivatives[column_parameter_symbol], 
                    waveform_derivatives[row_parameter_symbol])
                fisher_information_matrix[col][row] = fisher_information_matrix_element    
        
        _LOGGER.info("Fisher information matrix has been computed.")
        return fisher_information_matrix
    
    @timer_decorator
    def compute_Cramer_Rao_bounds(self) -> dict:

        fisher_information_matrix = self.compute_fisher_information_matrix()

        cramer_rao_bounds = cp.linalg.inv(fisher_information_matrix)

        parameter_symbol_list = [parameter.symbol for parameter in self.parameter_space.parameters_configuration]

        independent_cramer_rao_bounds = {}
        row_index = 0
        for row_parameter, row_cramer_rao_bounds in zip(parameter_symbol_list, cramer_rao_bounds):
            print(row_cramer_rao_bounds)
            reduced_parameter_list = parameter_symbol_list[row_index:]
            reduced_error_row = row_cramer_rao_bounds[row_index:]
            for col_parameter, cramer_rao_bound in zip(reduced_parameter_list, reduced_error_row):
                independent_cramer_rao_bounds[f"delta_{row_parameter}_delta_{col_parameter}"] = cramer_rao_bound
            row_index += 1
        _LOGGER.info("Finished computing Cramer Rao bounds.")
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

    def _visualize_cramer_rao_bounds(self) -> None:
        mean_errors_data = pd.read_csv(CRAMER_RAO_BOUNDS_PATH)
        error_column_list = [column_name for column_name in mean_errors_data.columns if "delta" in column_name]
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
                    cp.sqrt(mean_errors_data["delta_M_delta_M"]),
                    '.',
                    label = f"bounds: delta M")

            ax2.plot(
                mean_errors_data[column_name], 
                cp.sqrt(mean_errors_data["delta_qS_delta_qS"]),
                '.',
                label = f"bounds: qS")
            
            ax2.plot(
                mean_errors_data[column_name], 
                cp.sqrt(mean_errors_data["delta_phiS_delta_phiS"]),
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
            _LOGGER.warning("check_parameter_dependency couldn't match parameter symbol.")
            sys.exit()

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
