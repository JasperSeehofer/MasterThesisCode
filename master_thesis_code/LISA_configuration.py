import numpy as np
import cupy as cp
from dataclasses import dataclass
import matplotlib.pyplot as plt
import os
import logging



from master_thesis_code.decorators import timer_decorator
from master_thesis_code.datamodels.parameter_space import ParameterSpace
from master_thesis_code.constants import MAXIMAL_FREQUENCY, MINIMAL_FREQUENCY

_LOGGER = logging.getLogger()

@dataclass
class LISAConfiguration:

    # constants
    qS: float = 0 # binary orientation w.r.t. fixed reference frame
    phiS: float = 0 # binary orientation w.r.t. fixed reference frame
    qK: float = 0 # direction of the binary's angular momentum in the ecliptic reference system
    phiK: float = 0 # direction of the binary's angular momentum in the ecliptic reference system
    dt: float = 0
    is_LISA_second_measurement: bool = False
    is_schwarzschild: bool = False # for Schwarzschild waveforms the inclination is not regarded ()
    
    def __init__(self, parameter_space: ParameterSpace, dt: float) -> None:
        self.qS = parameter_space.qS
        self.phiS = parameter_space.phiS
        self.qK = parameter_space.qK
        self.phiK = parameter_space.phiK
        self.dt = dt


    # antenna pattern functions 41550...PDF
    def F_plus(self, t: float) -> float:
        """Antenna pattern function for + polarization of the gravitational wave. from PDF (41550...)

        Returns:
            float: projection of + polarization into the detector frame.
        """
        return (
            (1 + self.cos_theta_solar_barycenter(t)**2)/2*cp.cos(2*self.phi_solar_barycenter(t))*cp.cos(2*self.psi_solar_barycenter(t))
            - self.cos_theta_solar_barycenter(t)*cp.sin(2*self.phi_solar_barycenter(t))*cp.sin(2*self.psi_solar_barycenter(t)))

    def F_cross(self, t: float) -> float:
        """Antenna pattern function for + polarization of the gravitational wave. from PDF (41550...)

        Returns:
            float: projection of + polarization into the detector frame evaluated at angles provided by the class.
        """
        return (
            (1 + self.cos_theta_solar_barycenter(t)**2)/2*cp.cos(2*self.phi_solar_barycenter(t))*cp.cos(2*self.psi_solar_barycenter(t)) 
            + self.cos_theta_solar_barycenter(t)*cp.sin(2*self.phi_solar_barycenter(t))*cp.sin(2*self.psi_solar_barycenter(t)))


    def cos_theta_solar_barycenter(self, t):
        """Cosine of theta as a function of time and of the sky-localization in the solar system barycenter frame

        Args:
            t (_type_): time in s

        Returns:
            _type_: value of cosine at given time
        """
        return cp.cos(self.qS)/2 - cp.sqrt(3)/2*cp.sin(self.qS)*cp.cos(self.phi_t(t) - self.phiS)
    
    def phi_solar_barycenter(self, t):
        phi =  self.phi_t(t) + cp.arctan(
            (cp.sqrt(3)*cp.cos(self.qS) + cp.sin(self.qS)*cp.cos(self.phi_t(t) - self.phiS))/(2*cp.sin(self.qS)*cp.sin(self.phi_t(t) - self.phiS)))
        if self.is_LISA_second_measurement:
            phi += -cp.pi/4
        return phi

    def psi_solar_barycenter(self, t):
        Lz = cp.cos(self.qK)/2 - cp.sqrt(3)/2*cp.sin(self.qK)*cp.cos(self.phi_t(t) - self.phiK) # scalar product of L and z
        LN = cp.cos(self.qK)*cp.cos(self.qS) + cp.sin(self.qK)*cp.sin(self.qS)*cp.cos(self.phiK - self.phiS) # scalar product of L and N
        zN = self.cos_theta_solar_barycenter(t) # scalar product of z and N
        NLxz = (
            cp.sin(self.qK)*cp.sin(self.qS)*cp.sin(self.phiK - self.phiS)
            - cp.sqrt(3)/2*cp.cos(self.phi_t(t))*(
                cp.cos(self.qK)*cp.sin(self.qS)*cp.sin(self.phiS) 
                - cp.cos(self.qS*cp.sin(self.qK)*cp.sin(self.phiK))
                )
            - cp.sqrt(3)/2*cp.sin(self.phi_t(t))*(
                cp.cos(self.qS)*cp.sin(self.qK)*cp.cos(self.phiK)
                - cp.cos(self.qK)*cp.sin(self.qS)*cp.cos(self.phiS)
            )) # scalar product of N and (cross product of L and z)
        return cp.arctan(
            (Lz - LN*zN)/NLxz
        )

    def phi_t(self, t: float) -> float:
        """Phase of LISA's rotation around the sun.

        Args:
            t (_type_): given time in seconds

        Returns:
            float: current orbital phase
        """
        T = 31557600 # in s (1 year)
        return cp.multiply(t, 2*cp.pi/T)

    def power_spectral_density(self, f: float) -> float:
        return self.power_spectral_density_instrumental(f) + self.power_spectral_density_confusion_noise(f)

    def power_spectral_density_instrumental(self, f: float) -> float:
        """Noise spectral density from PDF (41550...)

        Args:
            f (float): frequency

        Returns:
            float: noise spectral density
        """
        L = 2.5e9 #m
        A_1 = 10./3./L**2 #1/m^2
        f_ast = 19.09e-3 #Hz
        if f == 0:
            f = 1e-9

        return A_1*(self.P_OMS(f) + 2.*(1. + cp.cos(f/f_ast)**2) * self.P_acc(f) / (2.*cp.pi*f)**4) * (1. + 6./10. * (f/f_ast)**2)

    def power_spectral_density_confusion_noise(self, f: float) -> float:
        """DEPENDS ON OBSERVATION TIME !! TODO

        Args:
            f (float): _description_

        Returns:
            float: _description_
        """
        A_2 = 9e-45
        alpha = 0.171
        beta = 292.
        kappa = 1020.
        gamma = 1680.
        f_k = 0.00215

        if f == 0:
            return 10**20

        return A_2*f**(-7/3)*cp.exp(-f**alpha + beta*f*cp.sin(kappa*f))*(1 + cp.tanh(gamma*(f_k-f)))

    @staticmethod
    def P_OMS(f: float) -> float:
        return (1.5e-11)**2 * (1+(2e-3/f)**4)

    @staticmethod
    def P_acc(f: float) -> float:
        return (3e-15)**2*(1.+(0.4e-3/f)**2)*(1.+(f/8e-3)**4)
    
    @timer_decorator
    def transform_from_ssb_to_lisa_frame(self, waveform: cp.ndarray) -> cp.array:
        waveform = cp.asnumpy(waveform)
        time_series = cp.array([index*self.dt for index, _ in enumerate(waveform.real)])
    
        measurement_1 = (waveform.real*self.F_plus(time_series) - waveform.imag*self.F_cross(time_series))*cp.sqrt(3)/2
        #self.is_LISA_second_measurement = True
        #measurement_2 = (waveform.real*self.F_plus(time_series) - waveform.imag*self.F_cross(time_series))*cp.sqrt(3)/2
        return cp.array(measurement_1)

    @timer_decorator
    def _visualize_lisa_configuration(self) -> None:
        figures_directory = f"saved_figures/LISA_configuration/"

        if not os.path.isdir(figures_directory):
            os.makedirs(figures_directory)

        # create plots
        # plot power spectral density
        fs = cp.linspace(MINIMAL_FREQUENCY, MAXIMAL_FREQUENCY, 10000)
        fig = plt.figure(figsize = (12, 8))
        plt.plot(
            fs, 
            [cp.sqrt(self.power_spectral_density_confusion_noise(f)) for f in fs], 
            '--',
            linewidth=1,
            label = "sqrt(S_WD(f))")
        plt.plot(
            fs, 
            [cp.sqrt(self.power_spectral_density_instrumental(f)) for f in fs], 
            '--',
            linewidth=1,
            label = "sqrt(S_INS(f))")
        plt.plot(
            fs, 
            [cp.sqrt(self.power_spectral_density(f)) for f in fs], 
            '-',
            label = "sqrt(S_n(f))")
        plt.xlabel("f [Hz]")
        plt.legend()
        plt.xscale("log")
        plt.yscale("log")
        plt.ylim(1e-21, 1e-14)
        plt.savefig(figures_directory + "LISA_PSD.png", dpi=300)
        plt.clf()

        # plot antenna pattern functions
        year_in_sec = int(365.5*24*60*60)
        steps = 100_000
        dt = year_in_sec/steps

        time_series = [index*dt for index in range(steps)]

        F_plus = self.F_plus(time_series)
        F_cross = self.F_cross(time_series)
        F = F_plus**2 + F_cross**2

        fig = plt.figure(figsize = (12, 8))
        plt.plot(
            time_series, 
            F_plus, 
            '-',
            label = "F_plus(t)")
        plt.plot(
            time_series, 
            F_cross, 
            '-',
            label = "F_cross(t)")
        plt.plot(
            time_series,
            F,
            '-',
            label = "F = F_plus^2 + F_cross^2"
        )

        plt.xlabel("t [s]")
        plt.legend()
        plt.savefig(figures_directory + "antenna_pattern_functions.png", dpi=300)

        plt.close()
