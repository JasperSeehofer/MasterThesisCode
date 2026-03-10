import logging
import os
import types
from dataclasses import dataclass
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

mpl.rcParams["agg.path.chunksize"] = 1000

try:
    import cupy as cp

    _CUPY_AVAILABLE = True
except ImportError:
    cp = None
    _CUPY_AVAILABLE = False

from master_thesis_code.constants import (
    C,
)
from master_thesis_code.decorators import timer_decorator

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


def _get_xp(arr: Any) -> types.ModuleType:
    """Return cupy if arr is a cupy array and cupy is available, else numpy."""
    if _CUPY_AVAILABLE and cp is not None and isinstance(arr, cp.ndarray):
        return cp  # type: ignore[no-any-return]
    return np


@dataclass
class LisaTdiConfiguration:
    def power_spectral_density(
        self,
        # xp may be numpy or cupy at runtime; annotation reflects the numpy-compatible interface
        frequencies: npt.NDArray[np.float64],
        channel: str = "A",
    ) -> npt.NDArray[np.float64]:
        """PSD noise for AET channels from https://arxiv.org/pdf/2303.15929.pdf assuming equal arm length."""
        if channel.upper() in ["A", "E"]:
            return self.power_spectral_density_a_channel(frequencies)
        elif channel.upper() == "T":
            return self.power_spectral_density_t_channel(frequencies)
        return np.zeros_like(frequencies)

    def power_spectral_density_a_channel(
        self, frequencies: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """from https://arxiv.org/pdf/2303.15929.pdf"""
        xp = _get_xp(frequencies)
        return (  # type: ignore[no-any-return]
            8
            * xp.sin(2 * xp.pi * frequencies * L / C) ** 2
            * (
                self.S_OMS(frequencies) * (xp.cos(2 * xp.pi * frequencies * L / C) + 2)
                + 2
                * (
                    3
                    + 2 * xp.cos(2 * xp.pi * frequencies * L / C)
                    + xp.cos(4 * xp.pi * frequencies * L / C)
                )
                * self.S_TM(frequencies)
            )
        )

    def power_spectral_density_t_channel(
        self, frequencies: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """from https://arxiv.org/pdf/2303.15929.pdf NOT UPDATED"""
        xp = _get_xp(frequencies)
        return (  # type: ignore[no-any-return]
            16
            / 3
            * xp.sin(xp.pi * frequencies * L / C) ** 2
            * xp.sin(2 * xp.pi * frequencies * L / C) ** 2
            * self.S_zz(frequencies)
        )

    def S_zz(self, frequencies: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        xp = _get_xp(frequencies)
        return 6 * (  # type: ignore[no-any-return]
            self.S_OMS(frequencies)
            + 2 * (1 - xp.cos(2 * xp.pi * frequencies * L / C) * self.S_TM(frequencies))
        )

    @staticmethod
    def S_OMS(frequencies: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        xp = _get_xp(frequencies)
        return 15**2 * 1e-24 * (1 + (2e-3 / frequencies) ** 4) * (2 * xp.pi * frequencies / C) ** 2  # type: ignore[no-any-return]

    @staticmethod
    def S_TM(frequencies: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        xp = _get_xp(frequencies)
        return (  # type: ignore[no-any-return]
            9e-30
            * (1 + (0.4e-3 / frequencies) ** 2)
            * (1 + (frequencies / 8e-3) ** 4)
            * (1 / 2 / xp.pi / frequencies / C) ** 2
        )

    @timer_decorator
    def _visualize_lisa_configuration(self) -> None:
        figures_directory = "saved_figures/LISA_configuration/"

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
