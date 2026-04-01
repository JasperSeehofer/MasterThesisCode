"""Factory functions for simulation-phase plots and the PlottingCallback.

Each factory takes data in and returns ``(fig, ax)`` out — no side effects.
The :class:`PlottingCallback` collects data during the simulation loop and
calls the factory functions in :meth:`on_simulation_end`.
"""

import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from master_thesis_code.plotting._helpers import _fig_from_ax, save_figure


def plot_gpu_usage(
    time_series: list[float],
    memory_pool_usage: list[float],
    gpu_usage: list[list[float]],
    *,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot GPU memory usage over time.

    Parameters
    ----------
    time_series:
        Wall-clock seconds since simulation start.
    memory_pool_usage:
        CuPy memory pool total bytes (GB) at each stamp.
    gpu_usage:
        Per-GPU memory usage (GB) at each stamp.  Shape: ``(n_stamps, n_gpus)``.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 9))
    else:
        fig = _fig_from_ax(ax)

    ax.plot(time_series, memory_pool_usage, "-", label="Memory Pool")
    for gpu_index, per_gpu in enumerate(np.array(gpu_usage).T):
        ax.plot(time_series, per_gpu, "-", label=f"GPU {gpu_index + 1}")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("Memory [GB]")
    ax.legend()
    return fig, ax


def plot_lisa_psd(
    frequencies: npt.NDArray[np.float64],
    psd_values: dict[str, npt.NDArray[np.float64]],
    *,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot LISA power spectral density curve(s).

    Parameters
    ----------
    frequencies:
        Frequency array in Hz.
    psd_values:
        Mapping from channel name (e.g. ``"A"``) to PSD array.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
    else:
        fig = _fig_from_ax(ax)

    for label, psd in psd_values.items():
        ax.plot(frequencies, psd, "-", linewidth=1, label=f"S_{label}(f)")
    ax.set_xlabel("f [Hz]")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()
    return fig, ax


def plot_lisa_noise_components(
    frequencies: npt.NDArray[np.float64],
    s_oms: npt.NDArray[np.float64],
    s_tm: npt.NDArray[np.float64],
    *,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot individual LISA noise components (S_OMS and S_TM)."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
    else:
        fig = _fig_from_ax(ax)

    ax.plot(frequencies, s_oms, "-", linewidth=1, label="S_OMS(f)")
    ax.plot(frequencies, s_tm, "-", linewidth=1, label="S_TM(f)")
    ax.set_xlabel("f [Hz]")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()
    return fig, ax


def plot_cramer_rao_coverage(
    M: npt.NDArray[np.float64],
    qS: npt.NDArray[np.float64],
    phiS: npt.NDArray[np.float64],
    M_limits: tuple[float, float],
    qS_limits: tuple[float, float],
    phiS_limits: tuple[float, float],
    *,
    ax: Any | None = None,
) -> tuple[Figure, Any]:
    """3D scatter plot of parameter-space coverage from detected events."""
    fig = plt.figure(figsize=(16, 9))
    ax3d = fig.add_subplot(111, projection="3d")
    ax3d.scatter3D(M, qS, phiS)
    ax3d.set_xlabel("M")
    ax3d.set_ylabel("qS")
    ax3d.set_zlabel("phiS")
    ax3d.set_xlim(*M_limits)
    ax3d.set_ylim(*qS_limits)
    ax3d.set_zlim(*phiS_limits)
    return fig, ax3d


class PlottingCallback:
    """Collects simulation data and produces plots at the end.

    Implements :class:`~master_thesis_code.callbacks.SimulationCallback`.
    """

    def __init__(self, output_dir: str) -> None:
        self._output_dir = output_dir
        self._time_series: list[float] = []
        self._memory_pool_usage: list[float] = []
        self._gpu_usage: list[list[float]] = []

    def on_simulation_start(self, total_steps: int) -> None:
        pass

    def on_snr_computed(self, step: int, snr: float, passed: bool) -> None:
        pass

    def on_detection(
        self, step: int, snr: float, cramer_rao: dict[str, float], host_idx: int
    ) -> None:
        pass

    def on_step_end(self, step: int, iteration: int) -> None:
        pass

    def record_gpu_stamp(
        self,
        elapsed: float,
        memory_pool_gb: float,
        per_gpu_gb: list[float],
    ) -> None:
        """Record a GPU memory snapshot (called from the simulation loop)."""
        self._time_series.append(elapsed)
        self._memory_pool_usage.append(memory_pool_gb)
        self._gpu_usage.append(per_gpu_gb)

    def on_simulation_end(self, total_detections: int, total_iterations: int) -> None:
        if self._time_series:
            fig, _ = plot_gpu_usage(self._time_series, self._memory_pool_usage, self._gpu_usage)
            save_figure(
                fig,
                os.path.join(self._output_dir, "monitoring", "GPU_usage"),
                formats=("png",),
            )
