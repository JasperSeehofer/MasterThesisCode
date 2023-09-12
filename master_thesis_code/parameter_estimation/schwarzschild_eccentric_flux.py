import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
plt.style.use("seaborn-poster")

import logging
import sys

from datamodels.parameter_space import SchwarzschildParameterSpace
from few.waveform import FastSchwarzschildEccentricFlux

use_gpu = False

# keyword arguments for inspiral generator (RunSchwarzEccFluxInspiral)
inspiral_kwargs={
        "DENSE_STEPPING": 0,  # we want a sparsely sampled trajectory
        "max_init_len": int(1e3),  # all of the trajectories will be well under len = 1000
    }

# keyword arguments for inspiral generator (RomanAmplitude)
amplitude_kwargs = {
    "max_init_len": int(1e3),  # all of the trajectories will be well under len = 1000
    "use_gpu": use_gpu  # GPU is available in this class
}

# keyword arguments for Ylm generator (GetYlms)
Ylm_kwargs = {
    "assume_positive_m": False  # if we assume positive m, it will generate negative m for all m>0
}

# keyword arguments for summation generator (InterpolatedModeSum)
sum_kwargs = {
    "use_gpu": use_gpu,  # GPU is availabel for this type of summation
    "pad_output": False,
}

class SchwarzschildParameterEstimation():
    parameter_space: SchwarzschildParameterSpace
    waveform_generator: FastSchwarzschildEccentricFlux
    dt: float = 10.0
    T: float = 1.0
    M_derivative_steps: int = 2


    def __init__(self):
        self.parameter_space = SchwarzschildParameterSpace()
        self.waveform_generator = FastSchwarzschildEccentricFlux(
            inspiral_kwargs=inspiral_kwargs,
            amplitude_kwargs=amplitude_kwargs,
            Ylm_kwargs=Ylm_kwargs,
            sum_kwargs=sum_kwargs,
            use_gpu=use_gpu,
        )

    def generate_waveform(self) -> np.ndarray[complex]:
        return self.waveform_generator(
            **self.parameter_space._parameters_to_dict(),
            dt=self.dt,
            T=self.T)
    
    def numeric_M_derivative(self) -> tuple:
        M_configuration = next(
            (parameter_configuration 
            for parameter_configuration in self.parameter_space.parameters_configuration
            if parameter_configuration.symbol == "M"), 
            None)
        
        if M_configuration is None:
            logging.warning("Configuration of Black hole mass not given.")
            sys.exit()

        dM = (M_configuration.upper_limit - M_configuration.lower_limit)/self.M_derivative_steps

        M_steps = np.arange(
            start=M_configuration.lower_limit,
            stop=M_configuration.upper_limit,
            step=dM)
        
        waveforms_M_real = pd.DataFrame()
        waveforms_M_imag = pd.DataFrame()
        for count, M in enumerate(M_steps, 1):
            self.parameter_space.M = M
            waveform = self.generate_waveform()
            logging.info(f"{count}/{self.M_derivative_steps} waveforms generated.")
            waveforms_M_real[f"M_{count}"] = waveform.real
            waveforms_M_imag[f"M_{count}"] = waveform.imag

        return ( M_steps, np.diff(waveforms_M_real)/dM, np.diff(waveforms_M_imag)/dM)
    
    
    def _plot_M_derivative(self, M_steps: np.array, M_differences_real: np.array, M_differences_imag) -> None:
        
        plt.figure(figsize = (12, 8))
        plt.plot(M_steps, 
                 M_differences_real, 
                 '--',
                 label = f"Re[dh(t)/dM] for {self.M_derivative_steps} steps")
        plt.plot(M_steps, 
                 M_differences_imag, 
                 '--',
                 label = f"Re[dh(t)/dM] for {self.M_derivative_steps} steps")

        plt.legend()
        plt.savefig(f"M_derivative_{self.M_derivative_steps}_steps.png", dpi=300)




