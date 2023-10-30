import numpy as np
import pandas as pd
from dataclasses import dataclass
import typing
import matplotlib.pyplot as plt
from matplotlib import cm
import os
import typing

from datamodels.parameter_space import ParameterSpace
from constants import REAL_PART, IMAGINARY_PART, INFINITY, MAXIMAL_FREQUENCY, MINIMAL_FREQUENCY


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
            (1 + self.cos_theta_solar_barycenter(t)**2)/2*np.cos(2*self.phi_solar_barycenter(t))*np.cos(2*self.psi_solar_barycenter(t))
            - self.cos_theta_solar_barycenter(t)*np.sin(2*self.phi_solar_barycenter(t))*np.sin(2*self.psi_solar_barycenter(t)))

    def F_cross(self, t: float) -> float:
        """Antenna pattern function for + polarization of the gravitational wave. from PDF (41550...)

        Returns:
            float: projection of + polarization into the detector frame evaluated at angles provided by the class.
        """
        return (
            (1 + self.cos_theta_solar_barycenter(t)**2)/2*np.cos(2*self.phi_solar_barycenter(t))*np.cos(2*self.psi_solar_barycenter(t)) 
            + self.cos_theta_solar_barycenter(t)*np.sin(2*self.phi_solar_barycenter(t))*np.sin(2*self.psi_solar_barycenter(t)))

    def F_plus_del_theta(self, t: float) -> float:
        """Partial derivative of the + polarization antenna pattern function with respect to the sky localization angle theta.

        Returns:
            float: derivative evaluated at angles provided by the class.
        """
        return (
            np.sin(2*self.phi_solar_barycenter(t))*np.sin(2*self.psi_solar_barycenter(t))*np.sin(self.theta_solar_barycenter(t)) 
            - np.sin(self.theta_solar_barycenter(t))*np.cos(2*self.phi_solar_barycenter(t))*np.cos(2*self.psi_solar_barycenter(t))*self.cos_theta_solar_barycenter(t))

    def F_plus_del_phi(self, t: float) -> float:
        """Partial derivative of the + polarization antenna pattern function with respect to the sky localization angle phi.

        Returns:
            float: derivative evaluated at angles provided by the class.
        """
        return (
            -2*(self.cos_theta_solar_barycenter(t)**2/2 + 1/2)*np.sin(2*self.phi_solar_barycenter(t))*np.cos(2*self.psi_solar_barycenter(t)) 
            - 2*np.sin(2*self.psi_solar_barycenter(t))*np.cos(2*self.phi_solar_barycenter(t))*self.cos_theta_solar_barycenter(t))

    def F_cross_del_theta(self, t: float) -> float:
        """Partial derivative of the x polarization antenna pattern function with respect to the sky localization angle theta.

        Returns:
            float: derivative evaluated at angles provided by the class.
        """
        return (
            np.sin(2*self.phi_solar_barycenter(t))*np.sin(2*self.psi_solar_barycenter(t))*np.sin(self.theta_solar_barycenter(t)) 
            - np.sin(self.theta_solar_barycenter(t))*np.cos(2*self.phi_solar_barycenter(t))*np.cos(2*self.psi_solar_barycenter(t))*self.cos_theta_solar_barycenter(t))

    def F_cross_del_phi(self, t: float) -> float:
        """Partial derivative of the x polarization antenna pattern function with respect to the sky localization angle phi.

        Returns:
            float: derivative evaluated at angles provided by the class.
        """
        return (
            -2*(self.cos_theta_solar_barycenter(t)**2/2 + 1/2)*np.sin(2*self.phi_solar_barycenter(t))*np.cos(2*self.psi_solar_barycenter(t)) 
            - 2*np.sin(2*self.psi_solar_barycenter(t))*np.cos(2*self.phi_solar_barycenter(t))*self.cos_theta_solar_barycenter(t))

    def cos_theta_solar_barycenter(self, t):
        """Cosine of theta as a function of time and of the sky-localization in the solar system barycenter frame

        Args:
            t (_type_): time in s

        Returns:
            _type_: value of cosine at given time
        """
        return np.cos(self.qS)/2 - np.sqrt(3)/2*np.sin(self.qS)*np.cos(self.phi_t(t) - self.phiS)
    
    def phi_solar_barycenter(self, t):
        phi =  self.phi_t(t) + np.arctan(
            (np.sqrt(3)*np.cos(self.qS) + np.sin(self.qS)*np.cos(self.phi_t(t) - self.phiS))/(2*np.sin(self.qS)*np.sin(self.phi_t(t) - self.phiS)))
        if self.is_LISA_second_measurement:
            phi += -np.pi/4
        return phi

    def psi_solar_barycenter(self, t):
        Lz = np.cos(self.qK)/2 - np.sqrt(3)/2*np.sin(self.qK)*np.cos(self.phi_t(t) - self.phiK) # scalar product of L and z
        LN = np.cos(self.qK)*np.cos(self.qS) + np.sin(self.qK)*np.sin(self.qS)*np.cos(self.phiK - self.phiS) # scalar product of L and N
        zN = self.cos_theta_solar_barycenter(t) # scalar product of z and N
        NLxz = (
            np.sin(self.qK)*np.sin(self.qS)*np.sin(self.phiK - self.phiS)
            - np.sqrt(3)/2*np.cos(self.phi_t(t))*(
                np.cos(self.qK)*np.sin(self.qS)*np.sin(self.phiS) 
                - np.cos(self.qS*np.sin(self.qK)*np.sin(self.phiK))
                )
            - np.sqrt(3)/2*np.sin(self.phi_t(t))*(
                np.cos(self.qS)*np.sin(self.qK)*np.cos(self.phiK)
                - np.cos(self.qK)*np.sin(self.qS)*np.cos(self.phiS)
            )) # scalar product of N and (cross product of L and z)
        return np.arctan(
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
        return np.multiply(t, 2*np.pi/T)

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

        return A_1*(self.P_OMS(f) + 2.*(1. + np.cos(f/f_ast)**2) * self.P_acc(f) / (2.*np.pi*f)**4) * (1. + 6./10. * (f/f_ast)**2)

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

        return A_2*f**(-7/3)*np.exp(-f**alpha + beta*f*np.sin(kappa*f))*(1 + np.tanh(gamma*(f_k-f)))

    @staticmethod
    def P_OMS(f: float) -> float:
        return (1.5e-11)**2 * (1+(2e-3/f)**4)

    @staticmethod
    def P_acc(f: float) -> float:
        return (3e-15)**2*(1.+(0.4e-3/f)**2)*(1.+(f/8e-3)**4)
    
    def transform_to_solar_barycenter_frame(self, waveform: typing.Union[pd.DataFrame, np.ndarray[complex]]) -> typing.Union[pd.DataFrame, np.ndarray[float]]:
        time_series = np.array([index*self.dt for index, _ in enumerate(waveform.real)])
        if isinstance(waveform, pd.DataFrame):
            return waveform.sort_index(axis=1, level=1, ascending=False).mul([self.F_plus(time_series), self.F_cross(time_series)], axis="columns", level=1)
        elif np.iscomplexobj(waveform):
            
            measurement_1 = (waveform.real*self.F_plus(time_series) - waveform.imag*self.F_cross(time_series))*np.sqrt(3)/2
            #self.is_LISA_second_measurement = True
            #measurement_2 = (waveform.real*self.F_plus(time_series) - waveform.imag*self.F_cross(time_series))*np.sqrt(3)/2
            return measurement_1

    def transform_to_solar_barycenter_frame_derivative_theta(self, waveform: np.ndarray[complex]) -> np.ndarray[float]:
        time_series = np.array([index*self.dt for index, _ in enumerate(waveform.real)])
        return (waveform.real*self.F_plus_del_theta(time_series) - waveform.imag*self.F_cross_del_theta(time_series))*np.sqrt(3)/2
    
    def transform_to_solar_barycenter_frame_derivative_phi(self, waveform: np.ndarray[complex]) -> np.ndarray[float]:
        time_series = np.array([index*self.dt for index, _ in enumerate(waveform.real)])
        return (waveform.real*self.F_plus_del_phi(time_series) - waveform.imag*self.F_cross_del_phi(time_series))*np.sqrt(3)/2

    def _visualize_lisa_configuration(self) -> None:
        figures_directory = f"saved_figures/LISA_configuration/"

        if not os.path.isdir(figures_directory):
            os.makedirs(figures_directory)

        # create plots
        # plot power spectral density
        fs = np.linspace(MINIMAL_FREQUENCY, MAXIMAL_FREQUENCY, 10000)
        fig = plt.figure(figsize = (12, 8))
        plt.plot(
            fs, 
            [np.sqrt(self.power_spectral_density_confusion_noise(f)) for f in fs], 
            '--',
            linewidth=1,
            label = "sqrt(S_WD(f))")
        plt.plot(
            fs, 
            [np.sqrt(self.power_spectral_density_instrumental(f)) for f in fs], 
            '--',
            linewidth=1,
            label = "sqrt(S_INS(f))")
        plt.plot(
            fs, 
            [np.sqrt(self.power_spectral_density(f)) for f in fs], 
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
