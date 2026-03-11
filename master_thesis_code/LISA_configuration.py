"""LISA TDI noise configuration: PSD, antenna patterns, and frame transformations.

Provides :class:`LisaTdiConfiguration` with the noise power spectral density for
the A/E/T TDI channels, the one-sided optical metrology and test-mass noise
components, and the sky-averaged F+/F× antenna pattern functions.
"""

import logging
import types
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

try:
    import cupy as cp

    _CUPY_AVAILABLE = True
except ImportError:
    cp = None
    _CUPY_AVAILABLE = False

from master_thesis_code.constants import (
    LISA_ARM_LENGTH as L,
)
from master_thesis_code.constants import (
    C,
)

_LOGGER = logging.getLogger()


def _get_xp(arr: Any) -> types.ModuleType:
    """Return cupy if arr is a cupy array and cupy is available, else numpy."""
    if _CUPY_AVAILABLE and cp is not None and isinstance(arr, cp.ndarray):
        return cp  # type: ignore[no-any-return]
    return np


@dataclass
class LisaTdiConfiguration:
    """LISA TDI noise model and antenna pattern functions.

    Implements the noise power spectral density for the A, E, and T TDI channels
    and the sky-averaged F+/F× antenna pattern functions for the LISA constellation,
    following the equal-arm-length approximation.

    References:
        Babak et al. (2023), arXiv:2303.15929
    """

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
