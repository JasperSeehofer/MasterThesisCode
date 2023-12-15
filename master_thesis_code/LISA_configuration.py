import numpy as np
import cupy as cp
from dataclasses import dataclass
import matplotlib.pyplot as plt
import os
import logging


from master_thesis_code.decorators import timer_decorator
from master_thesis_code.datamodels.parameter_space import ParameterSpace
from master_thesis_code.constants import MAXIMAL_FREQUENCY, MINIMAL_FREQUENCY, G, C

_LOGGER = logging.getLogger()


# constants
L = 2.5e9  # m
A_1 = 10.0 / 3.0 / L**2  # 1/m^2
f_ast = 19.09e-3  # Hz
A_2 = 9e-45
alpha = 0.171
beta = 292.0
kappa = 1020.0
gamma = 1680.0
f_k = 0.00215
YEAR_IN_SEC = int(365.5 * 24 * 60 * 60)
STEPS = 100_00
DT = YEAR_IN_SEC / STEPS

@dataclass
class LISAConfiguration:
    # constants
    qS: float = 0  # binary orientation w.r.t. fixed reference frame
    phiS: float = 0  # binary orientation w.r.t. fixed reference frame
    qK: float = (
        0  # direction of the binary's angular momentum in the ecliptic reference system
    )
    phiK: float = (
        0  # direction of the binary's angular momentum in the ecliptic reference system
    )
    dt: float = 0
    is_LISA_second_measurement: bool = False
    is_schwarzschild: bool = (
        False  # for Schwarzschild waveforms the inclination is not regarded ()
    )

    def __init__(self, parameter_space: ParameterSpace, dt: float) -> None:
        self.qS = parameter_space.qS
        self.phiS = parameter_space.phiS
        self.qK = parameter_space.qK
        self.phiK = parameter_space.phiK
        self.M = parameter_space.M
        self.mu = parameter_space.mu
        self.dist = parameter_space.dist
        self.dt = dt

        self._snr_estimation_factor = np.sqrt(5/6)/np.pi**(2/3)*C*(G/C**3)**(5/6)*self._compute_LISA_snr_frequency_factor()

    def update_parameters(self, parameter_space: ParameterSpace) -> None:
        self.qS = parameter_space.qS
        self.phiS = parameter_space.phiS
        self.qK = parameter_space.qK
        self.phiK = parameter_space.phiK
        self.M = parameter_space.M
        self.mu = parameter_space.mu
        self.dist = parameter_space.dist

    def _compute_LISA_snr_frequency_factor(self) -> float:
        fs = cp.linspace(MINIMAL_FREQUENCY, MAXIMAL_FREQUENCY, 10000)
        integrant = cp.divide( fs**(-7/3), self.power_spectral_density(fs))
        return cp.sqrt(cp.trapz(integrant,fs))
    
    def Q_snr_check(self) -> float:
        time_series = cp.arange(0,STEPS)*DT
        F = cp.mean(cp.sqrt(self.F_plus(time_series)**2 + self.F_cross(time_series)**2))
        _LOGGER.debug(f"In easy SNR check F={cp.round(F, 3)}.")
        return F
    
    def SNR_sanity_check(self) -> float:
        return self.Q_snr_check()*self._snr_estimation_factor*((self.M*self.mu)**(3/5)/(self.M + self.mu)**(1/5))**(5/6)/self.dist

    # antenna pattern functions 41550...PDF
    def F_plus(self, time_series: cp.array) -> cp.array:
        """Antenna pattern function for + polarization of the gravitational wave. from PDF (41550...)

        Returns:
            float: projection of + polarization into the detector frame.
        """
        return (1 + self.cos_theta_solar_barycenter(time_series)**2) / 2 * cp.cos(
            2 * self.phi_solar_barycenter(time_series)
        ) * cp.cos(
            2 * self.psi_solar_barycenter(time_series)
        ) - self.cos_theta_solar_barycenter(
            time_series
        ) * cp.sin(
            2 * self.phi_solar_barycenter(time_series)
        ) * cp.sin(
            2 * self.psi_solar_barycenter(time_series)
        )

    def F_cross(self, time_series: cp.array) -> cp.array:
        """Antenna pattern function for + polarization of the gravitational wave. from PDF (41550...)

        Returns:
            float: projection of + polarization into the detector frame evaluated at angles provided by the class.
        """
        return (1 + self.cos_theta_solar_barycenter(time_series) ** 2) / 2 * cp.cos(
            2 * self.phi_solar_barycenter(time_series)
        ) * cp.sin(
            2 * self.psi_solar_barycenter(time_series)
        ) + self.cos_theta_solar_barycenter(
            time_series
        ) * cp.sin(
            2 * self.phi_solar_barycenter(time_series)
        ) * cp.cos(
            2 * self.psi_solar_barycenter(time_series)
        )

    def cos_theta_solar_barycenter(self, time_series: cp.array) -> cp.array:
        """Cosine of theta as a function of time and of the sky-localization in the solar system barycenter frame

        Args:
            t (_type_): time in s

        Returns:
            _type_: value of cosine at given time
        """
        return cp.cos(self.qS) / 2 - cp.sqrt(3) / 2 * cp.sin(self.qS) * cp.cos(
            self.phi_t(time_series) - self.phiS
        )

    def phi_solar_barycenter(self, time_series: cp.array) -> cp.array:
        phi = self.phi_t(time_series) + cp.arctan(
            (
                cp.sqrt(3) * cp.cos(self.qS)
                + cp.sin(self.qS) * cp.cos(self.phi_t(time_series) - self.phiS)
            )
            / (2 * cp.sin(self.qS) * cp.sin(self.phi_t(time_series) - self.phiS))
        )
        if self.is_LISA_second_measurement:
            phi += -cp.pi / 4
        return phi

    def psi_solar_barycenter(self, time_series: cp.array) -> cp.array:
        Lz = cp.cos(self.qK) / 2 - cp.sqrt(3) / 2 * cp.sin(self.qK) * cp.cos(
            self.phi_t(time_series) - self.phiK
        )  # scalar product of L and z
        LN = cp.cos(self.qK) * cp.cos(self.qS) + cp.sin(self.qK) * cp.sin(
            self.qS
        ) * cp.cos(
            self.phiK - self.phiS
        )  # scalar product of L and N
        zN = self.cos_theta_solar_barycenter(time_series)  # scalar product of z and N
        NLxz = (
            cp.sin(self.qK) * cp.sin(self.qS) * cp.sin(self.phiK - self.phiS)/2
            - cp.sqrt(3)
            / 2
            * cp.cos(self.phi_t(time_series))
            * (
                cp.cos(self.qK) * cp.sin(self.qS) * cp.sin(self.phiS)
                - cp.cos(self.qS) * cp.sin(self.qK) * cp.sin(self.phiK)
            )
            - cp.sqrt(3)
            / 2
            * cp.sin(self.phi_t(time_series))
            * (
                cp.cos(self.qS) * cp.sin(self.qK) * cp.cos(self.phiK)
                - cp.cos(self.qK) * cp.sin(self.qS) * cp.cos(self.phiS)
            )
        )  # scalar product of N and (cross product of L and z)
        result = cp.arctan((Lz - LN * zN) / NLxz)
        del Lz
        del zN
        del NLxz
        del LN
        return result

    def phi_t(self, time_series: cp.array) -> cp.array:
        """Phase of LISA's rotation around the sun.

        Args:
            time_series: time series in seconds

        Returns:
            float: current orbital phase
        """
        T = 31557600  # in s (1 year)
        return cp.multiply(time_series, 2 * cp.pi / T)

    def power_spectral_density(self, frequencies: cp.ndarray) -> cp.array:
        return self.power_spectral_density_instrumental(
            frequencies
        ) + self.power_spectral_density_confusion_noise(frequencies)

    def power_spectral_density_instrumental(self, frequencies: cp.array) -> cp.array:
        """Noise spectral density from PDF (41550...)

        Args:
            f (float): frequency

        Returns:
            float: noise spectral density
        """
        return (
            A_1
            * (
                self.P_OMS(frequencies)
                + 2.0
                * (1.0 + cp.cos(frequencies / f_ast) ** 2)
                * self.P_acc(frequencies)
                / (2.0 * cp.pi * frequencies) ** 4
            )
            * (1.0 + 6.0 / 10.0 * (frequencies / f_ast) ** 2)
        )

    def power_spectral_density_confusion_noise(self, frequencies: cp.array) -> cp.array:
        """DEPENDS ON OBSERVATION TIME !! TODO

        Args:
            frequencies: frequencies

        Returns:
            PSD: _description_
        """
        return (
            A_2
            * frequencies ** (-7 / 3)
            * cp.exp(-(frequencies**alpha) + beta * frequencies * cp.sin(kappa * frequencies))
            * (1 + cp.tanh(gamma * (f_k - frequencies)))
        )

    @staticmethod
    def P_OMS(frequencies: cp.array) -> cp.array:
        return (1.5e-11) ** 2 * (1 + (2e-3 / frequencies) ** 4)

    @staticmethod
    def P_acc(frequencies: cp.array) -> cp.array:
        return (3e-15) ** 2 * (1.0 + (0.4e-3 / frequencies) ** 2) * (1.0 + (frequencies / 8e-3) ** 4)

    @timer_decorator
    def transform_from_ssb_to_lisa_frame(self, waveform: cp.ndarray) -> cp.array:
        time_series = cp.multiply(cp.arange(0, waveform.shape[0]), self.dt)

        measurement_1 = (
            (
                cp.subtract(
                    cp.multiply(waveform.real, self.F_plus(time_series)),
                    cp.multiply(waveform.imag, self.F_cross(time_series))
                )
            )
            * cp.sqrt(3)
            / 2
        )
        # self.is_LISA_second_measurement = True
        # measurement_2 = (waveform.real*self.F_plus(time_series) - waveform.imag*self.F_cross(time_series))*cp.sqrt(3)/2
        del waveform
        del time_series
        return cp.array(measurement_1)

    @timer_decorator
    def _visualize_lisa_configuration(self) -> None:
        figures_directory = f"saved_figures/LISA_configuration/"

        if not os.path.isdir(figures_directory):
            os.makedirs(figures_directory)

        # create plots
        # plot power spectral density
        fs = cp.linspace(MINIMAL_FREQUENCY, MAXIMAL_FREQUENCY, 10000)
        fig = plt.figure(figsize=(12, 8))
        plt.plot(
            cp.asnumpy(fs),
            cp.asnumpy(cp.sqrt(self.power_spectral_density_confusion_noise(fs))),
            "--",
            linewidth=1,
            label="sqrt(S_WD(f))",
        )
        plt.plot(
            cp.asnumpy(fs),
            cp.asnumpy(cp.sqrt(self.power_spectral_density_instrumental(fs))),
            "--",
            linewidth=1,
            label="sqrt(S_INS(f))",
        )
        plt.plot(
            cp.asnumpy(fs),
            cp.asnumpy(cp.sqrt(self.power_spectral_density(fs))),
            "-",
            label="sqrt(S_n(f))",
        )
        plt.xlabel("f [Hz]")
        plt.legend()
        plt.xscale("log")
        plt.yscale("log")
        plt.ylim(1e-21, 1e-14)
        plt.savefig(figures_directory + "LISA_PSD.png", dpi=300)
        plt.clf()

        # plot antenna pattern functions
        year_in_sec = int(365.5 * 24 * 60 * 60)
        steps = 100_000
        dt = year_in_sec / steps

        time_series = cp.arange(0, steps)*dt

        F_plus = self.F_plus(time_series)
        F_cross = self.F_cross(time_series)
        F = F_plus**2 + F_cross**2

        time_series = cp.asnumpy(time_series)

        fig = plt.figure(figsize=(12, 8))
        plt.plot(time_series, cp.asnumpy(F_plus), "-", label="F_plus(t)")
        plt.plot(time_series, cp.asnumpy(F_cross), "-", label="F_cross(t)")
        plt.plot(time_series, cp.asnumpy(F), "-", label="F = F_plus^2 + F_cross^2")

        plt.xlabel("t [s]")
        plt.legend()
        plt.savefig(figures_directory + "antenna_pattern_functions.png", dpi=300)

        plt.close()
