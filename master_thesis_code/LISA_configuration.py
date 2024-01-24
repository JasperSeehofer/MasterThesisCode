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
STEPS = 10_000
DT = YEAR_IN_SEC / STEPS
M_IN_GPC = 3.24077e-26


class LisaTdiConfiguration:
    def power_spectral_density(
        self, frequencies: cp.array, channel: str = "A"
    ) -> cp.array:
        """PSD noise for AET channels from https://arxiv.org/pdf/2303.15929.pdf assuming equal arm length."""
        if channel.upper() in ["A", "E"]:
            self.power_spectral_density_a_channel(frequencies)
        elif channel.upper() == "T":
            self.power_spectral_density_t_channel(frequencies)

    def power_spectral_density_a_channel(self, frequencies: cp.array) -> cp.array:
        """from https://arxiv.org/pdf/2303.15929.pdf"""

        return (
            8
            * cp.sin(2 * cp.pi * frequencies * L) ** 2
            * (
                self.S_OMS(frequencies) * (cp.cos(2 * cp.pi * frequencies * L) + 2)
                + 2
                * (
                    3
                    + 2 * cp.cos(2 * cp.pi * frequencies * L)
                    + cp.cos(4 * cp.pi * frequencies * L)
                )
                * self.S_TM(frequencies)
            )
        )

    def power_spectral_density_t_channel(self, frequencies: cp.array) -> cp.array:
        """from https://arxiv.org/pdf/2303.15929.pdf"""
        return (
            16
            / 3
            * cp.sin(cp.pi * frequencies * L) ** 2
            * cp.sin(2 * cp.pi * frequencies * L) ** 2
            * self.S_zz(frequencies)
        )

    def S_zz(self, frequencies: cp.array) -> cp.array:
        return 6 * (
            self.S_OMS(frequencies)
            + 2 * (1 - cp.cos(2 * cp.pi * frequencies * L) * self.S_TM(frequencies))
        )

    def S_OMS(frequencies: cp.array) -> cp.array:
        return (
            15**2e-24
            * (1 + (2e-3 / frequencies) ** 4)
            * (2 * cp.pi * frequencies / C) ** 2
            * M_IN_GPC**2
        )

    def S_TM(frequencies: cp.array) -> cp.array:
        return (
            9e-30
            * (1 + (0.4e-3 / frequencies) ** 2)
            * (1 + (frequencies / 8e-3) ** 4)
            * (1 / 2 / cp.pi / frequencies / C) ** 2
            * M_IN_GPC**2
        )


@dataclass
class LISAConfiguration:
    def power_spectral_density(self, frequencies: cp.ndarray) -> cp.array:
        return self._power_spectral_density_instrumental(
            frequencies
        ) + self._power_spectral_density_confusion_noise(frequencies)

    def _power_spectral_density_instrumental(self, frequencies: cp.array) -> cp.array:
        """Noise spectral density from PDF (41550...)

        Args:
            f (float): frequency

        Returns:
            float: noise spectral density
        """
        return (
            A_1
            * (
                self._P_OMS(frequencies)
                + 2.0
                * (1.0 + cp.cos(frequencies / f_ast) ** 2)
                * self._P_acc(frequencies)
                / (2.0 * cp.pi * frequencies) ** 4
            )
            * (1.0 + 6.0 / 10.0 * (frequencies / f_ast) ** 2)
        )

    def _power_spectral_density_confusion_noise(
        self, frequencies: cp.array
    ) -> cp.array:
        """DEPENDS ON OBSERVATION TIME !! TODO

        Args:
            frequencies: frequencies

        Returns:
            PSD: _description_
        """
        return (
            A_2
            * frequencies ** (-7 / 3)
            * cp.exp(
                -(frequencies**alpha)
                + beta * frequencies * cp.sin(kappa * frequencies)
            )
            * (1 + cp.tanh(gamma * (f_k - frequencies)))
        )

    @staticmethod
    def _P_OMS(frequencies: cp.array) -> cp.array:
        return (1.5e-11) ** 2 * (1 + (2e-3 / frequencies) ** 4)

    @staticmethod
    def _P_acc(frequencies: cp.array) -> cp.array:
        return (
            (3e-15) ** 2
            * (1.0 + (0.4e-3 / frequencies) ** 2)
            * (1.0 + (frequencies / 8e-3) ** 4)
        )

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
            cp.asnumpy(cp.sqrt(self._power_spectral_density_confusion_noise(fs))),
            "--",
            linewidth=1,
            label="sqrt(S_WD(f))",
        )
        plt.plot(
            cp.asnumpy(fs),
            cp.asnumpy(cp.sqrt(self._power_spectral_density_instrumental(fs))),
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

        time_series = cp.arange(0, steps) * dt

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
