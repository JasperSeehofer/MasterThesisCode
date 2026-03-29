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
    LISA_PSD_A,
    LISA_PSD_A1,
    LISA_PSD_AK,
    LISA_PSD_ALPHA,
    LISA_PSD_B1,
    LISA_PSD_BK,
    LISA_PSD_F2,
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

    The A/E-channel PSD includes an optional galactic confusion noise foreground
    S_c(f) from unresolved white dwarf binaries, controlled by
    ``include_confusion_noise`` (default True).  The observation time
    ``t_obs_years`` sets the level of foreground subtraction.

    References:
        Babak et al. (2023), arXiv:2303.15929
        Cornish & Robson (2017), arXiv:1703.09858
        Robson, Cornish & Liu (2019), arXiv:1803.01944
    """

    t_obs_years: float = 4.0
    include_confusion_noise: bool = True

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

    # Eq. (3) in Cornish & Robson (2017), arXiv:1703.09858
    # LDC parameterization with continuous T_obs dependence
    # See also Robson, Cornish & Liu (2019), arXiv:1803.01944, Eq. (14)
    def _confusion_noise(self, frequencies: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Galactic foreground confusion noise S_c(f).

        Computes the residual foreground from unresolved white dwarf binaries
        using the LDC parameterization with continuous observation-time
        dependence.  Dominates the LISA sensitivity in the 0.1--3 mHz band.

        Args:
            frequencies: Positive frequency array in Hz.

        Returns:
            S_c(f) in Hz^{-1} (one-sided strain PSD contribution).
        """
        xp = _get_xp(frequencies)
        # Power-law coefficients (a1, b1, ak, bk) were fitted with T_obs in years.
        f1 = 10.0 ** (LISA_PSD_A1 * xp.log10(self.t_obs_years) + LISA_PSD_B1)
        fk = 10.0 ** (LISA_PSD_AK * xp.log10(self.t_obs_years) + LISA_PSD_BK)
        return (  # type: ignore[no-any-return]
            LISA_PSD_A
            * frequencies ** (-7.0 / 3.0)
            * xp.exp(-((frequencies / f1) ** LISA_PSD_ALPHA))
            * 0.5
            * (1.0 + xp.tanh(-(frequencies - fk) / LISA_PSD_F2))
        )

    def power_spectral_density_a_channel(
        self, frequencies: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """A/E-channel PSD including optional galactic confusion noise.

        References:
            Instrumental noise: Babak et al. (2023), arXiv:2303.15929
            Confusion noise: Cornish & Robson (2017), arXiv:1703.09858
        """
        xp = _get_xp(frequencies)
        instrumental = (
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
        if self.include_confusion_noise:
            instrumental = instrumental + self._confusion_noise(frequencies)
        return instrumental  # type: ignore[no-any-return]

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
