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

from master_thesis_code.plotting._colors import CYCLE, EDGE, MEAN, REFERENCE
from master_thesis_code.plotting._helpers import _fig_from_ax, get_figure, save_figure
from master_thesis_code.plotting._labels import LABELS


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
        fig, ax = get_figure(preset="double")
    else:
        fig = _fig_from_ax(ax)

    ax.plot(time_series, memory_pool_usage, "-", label="Memory Pool")
    for gpu_index, per_gpu in enumerate(np.array(gpu_usage).T):
        ax.plot(time_series, per_gpu, "-", label=f"GPU {gpu_index + 1}")
    ax.set_xlabel(LABELS["t"])
    ax.set_ylabel("Memory [GB]")
    ax.legend()
    return fig, ax


def plot_lisa_psd(
    frequencies: npt.NDArray[np.float64],
    psd_values: dict[str, npt.NDArray[np.float64]] | None = None,
    *,
    decompose: bool = False,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot LISA power spectral density curve(s).

    Parameters
    ----------
    frequencies:
        Frequency array in Hz.
    psd_values:
        Mapping from channel name (e.g. ``"A"``) to PSD array.
        Used for backward-compatible mode.
    decompose:
        If ``True``, compute and plot three PSD curves: total, instrument-only,
        and galactic confusion noise.  Requires ``LisaTdiConfiguration``.
    """
    if ax is None:
        fig, ax = get_figure(preset="single")
    else:
        fig = _fig_from_ax(ax)

    if decompose:
        # Deferred import to avoid CPU import issues with LISA_configuration
        from master_thesis_code.LISA_configuration import LisaTdiConfiguration

        lisa_total = LisaTdiConfiguration(include_confusion_noise=True)
        lisa_inst = LisaTdiConfiguration(include_confusion_noise=False)

        psd_total = lisa_total.power_spectral_density_a_channel(frequencies)
        psd_inst = lisa_inst.power_spectral_density_a_channel(frequencies)
        psd_confusion = np.maximum(psd_total - psd_inst, 0.0)

        ax.plot(
            frequencies, psd_total,
            color=EDGE, linestyle="-", linewidth=2,
            label=r"$S_n(f)$ total",
        )
        ax.plot(
            frequencies, psd_inst,
            color=REFERENCE, linestyle="--", linewidth=1.5,
            label=r"$S_\mathrm{inst}(f)$",
        )
        ax.plot(
            frequencies, psd_confusion,
            color=CYCLE[1], linestyle="-.", linewidth=1.5,
            label=r"$S_\mathrm{gal}(f)$",
        )
    elif psd_values is not None:
        for label, psd in psd_values.items():
            ax.plot(frequencies, psd, "-", linewidth=1, label=f"S_{label}(f)")

    ax.set_xlabel(LABELS["f"])
    ax.set_ylabel(LABELS["PSD"])
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
        fig, ax = get_figure(preset="single")
    else:
        fig = _fig_from_ax(ax)

    ax.plot(frequencies, s_oms, "-", linewidth=1, label="S_OMS(f)")
    ax.plot(frequencies, s_tm, "-", linewidth=1, label="S_TM(f)")
    ax.set_xlabel(LABELS["f"])
    ax.set_ylabel(LABELS["PSD"])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()
    return fig, ax


def plot_detection_yield(
    injected_redshifts: npt.NDArray[np.float64],
    detected_redshifts: npt.NDArray[np.float64],
    *,
    bins: int = 30,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Injected vs detected redshift histograms with detection fraction curve.

    Parameters
    ----------
    injected_redshifts:
        Redshifts of all injected events.
    detected_redshifts:
        Redshifts of events that passed the SNR threshold.
    bins:
        Number of histogram bins.
    ax:
        Optional pre-existing Axes to draw on.

    Returns
    -------
    tuple[Figure, Axes]
        Figure and the primary (left) Axes.
    """
    if ax is None:
        fig, ax = get_figure(preset="single")
    else:
        fig = _fig_from_ax(ax)

    # Shared bin edges
    lo = min(float(injected_redshifts.min()), float(detected_redshifts.min()))
    hi = max(float(injected_redshifts.max()), float(detected_redshifts.max()))
    bin_edges_arr = np.linspace(lo, hi, bins + 1)
    bin_edges: list[float] = bin_edges_arr.tolist()
    bin_centers = 0.5 * (bin_edges_arr[:-1] + bin_edges_arr[1:])

    # Left y-axis: injected (outline) + detected (filled)
    ax.hist(
        injected_redshifts, bins=bin_edges,
        histtype="step", color=CYCLE[0], linewidth=1.5, label="Injected",
    )
    ax.hist(
        detected_redshifts, bins=bin_edges,
        alpha=0.6, color=CYCLE[0], label="Detected",
    )

    # Right y-axis: detection fraction
    counts_inj, _ = np.histogram(injected_redshifts, bins=bin_edges_arr)
    counts_det, _ = np.histogram(detected_redshifts, bins=bin_edges_arr)
    fraction = np.where(counts_inj > 0, counts_det / counts_inj, 0.0)

    ax2 = ax.twinx()
    ax2.plot(bin_centers, fraction, color=MEAN, linewidth=1.5, label="Detection fraction")
    ax2.set_ylabel("Detection fraction")
    ax2.set_ylim(0, 1)

    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2)

    ax.set_xlabel(LABELS["z"])
    ax.set_ylabel("Count")
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
