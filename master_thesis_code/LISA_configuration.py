from typing import Tuple
import cupy as cp
from dataclasses import dataclass
import matplotlib as mpl

mpl.rcParams["agg.path.chunksize"] = 1000
import matplotlib.pyplot as plt
import os
import logging


from master_thesis_code.decorators import timer_decorator
from master_thesis_code.datamodels.parameter_space import ParameterSpace
from master_thesis_code.constants import (
    MAXIMAL_FREQUENCY,
    MINIMAL_FREQUENCY,
    G,
    C,
    M_IN_GPC,
)

_LOGGER = logging.getLogger()


# constants
L = 2.5e9  # m
YEAR_IN_SEC = int(365.5 * 24 * 60 * 60)
STEPS = 10_000
DT = YEAR_IN_SEC / STEPS

A = 1.14e-44
alpha = 1.8
f_2 = 0.31e-3
a_1 = -0.25
b_1 = -2.7
a_k = -0.27
b_k = -2.47


@dataclass
class LisaTdiConfiguration:
    observation_time: float
    f_1: float
    f_k: float

    def __init__(self, observation_time) -> None:
        self.observation_time = observation_time
        self.f_1, self.f_k = _compute_galactic_confusion_noise_parameters(
            observation_time
        )

    def power_spectral_density(
        self, frequencies: cp.array, channel: str = "A"
    ) -> cp.array:
        """PSD noise for AET channels from https://arxiv.org/pdf/2303.15929.pdf assuming equal arm length."""
        if channel.upper() in ["A", "E"]:
            return self.power_spectral_density_a_channel(frequencies)
        elif channel.upper() == "T":
            return self.power_spectral_density_t_channel(frequencies)

    def power_spectral_density_a_channel(self, frequencies: cp.array) -> cp.array:
        """from https://arxiv.org/pdf/2303.15929.pdf"""

        return (
            8
            * cp.sin(2 * cp.pi * frequencies * L / C) ** 2
            * (
                self.S_OMS(frequencies) * (cp.cos(2 * cp.pi * frequencies * L / C) + 2)
                + 2
                * (
                    3
                    + 2 * cp.cos(2 * cp.pi * frequencies * L / C)
                    + cp.cos(4 * cp.pi * frequencies * L / C)
                )
                * self.S_TM(frequencies)
            )
        )

    def power_spectral_density_t_channel(self, frequencies: cp.array) -> cp.array:
        """from https://arxiv.org/pdf/2303.15929.pdf NOT UPDATED"""
        return (
            16
            / 3
            * cp.sin(cp.pi * frequencies * L / C) ** 2
            * cp.sin(2 * cp.pi * frequencies * L / C) ** 2
            * self.S_zz(frequencies)
        )

    def S_zz(self, frequencies: cp.array) -> cp.array:
        return 6 * (
            self.S_OMS(frequencies)
            + 2 * (1 - cp.cos(2 * cp.pi * frequencies * L / C) * self.S_TM(frequencies))
        )

    @staticmethod
    def S_OMS(frequencies: cp.array) -> cp.array:
        return (
            15**2
            * 1e-24
            * (1 + (2e-3 / frequencies) ** 4)
            * (2 * cp.pi * frequencies / C) ** 2
        )

    @staticmethod
    def S_TM(frequencies: cp.array) -> cp.array:
        return (
            9e-30
            * (1 + (0.4e-3 / frequencies) ** 2)
            * (1 + (frequencies / 8e-3) ** 4)
            * (1 / 2 / cp.pi / frequencies / C) ** 2
        )

    @timer_decorator
    def _visualize_lisa_configuration(self) -> None:
        figures_directory = f"saved_figures/LISA_configuration/"

        if not os.path.isdir(figures_directory):
            os.makedirs(figures_directory)

        # create plots
        # plot power spectral density
        fs = cp.logspace(-4, 0, 10000)
        fig = plt.figure(figsize=(12, 8))

        plt.plot(
            cp.asnumpy(fs),
            cp.asnumpy(self.S_OMS(fs)),
            "-",
            linewidth=1,
            label="S_OMS(f)",
        )
        plt.plot(
            cp.asnumpy(fs),
            cp.asnumpy(self.S_TM(fs)),
            "-",
            linewidth=1,
            label="S_TM(f)",
        )
        plt.plot(
            cp.asnumpy(fs),
            cp.asnumpy(self._power_spectral_density_confusion_noise(fs)),
            linewidth=1,
            label="S_gal(f)",
        )

        plt.xlabel("f [Hz]")
        plt.legend()
        plt.xscale("log")
        plt.yscale("log")
        plt.savefig(figures_directory + "LISA_noise_functions.png", dpi=300)
        plt.clf()

        # check characteristic functions S_OMS S_ACC
        fig = plt.figure(figsize=(12, 8))
        for channel in ["A"]:
            plt.plot(
                cp.asnumpy(fs),
                cp.asnumpy(self.power_spectral_density(fs, channel=channel)),
                "-",
                linewidth=1,
                label=f"S_{channel}(f)",
            )

        plt.xlabel("f [Hz]")
        plt.legend()
        plt.xscale("log")
        plt.yscale("log")
        plt.savefig(figures_directory + "LISA_PSD.png", dpi=300)
        plt.clf()


def _compute_galactic_confusion_noise_parameters(
    observation_time: float,
) -> Tuple[float, float]:
    f_1 = 10 ** (a_1 * cp.log10(observation_time) + b_1)
    f_k = 10 ** (a_k * cp.log10(observation_time) + b_k)
    return (f_1, f_k)
