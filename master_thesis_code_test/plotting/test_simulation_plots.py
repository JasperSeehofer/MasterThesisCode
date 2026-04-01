"""Smoke tests for simulation_plots factory functions."""

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from master_thesis_code.plotting.simulation_plots import (
    plot_cramer_rao_coverage,
    plot_gpu_usage,
    plot_lisa_noise_components,
    plot_lisa_psd,
)


def test_plot_gpu_usage() -> None:
    fig, ax = plot_gpu_usage(
        time_series=[0.0, 1.0, 2.0],
        memory_pool_usage=[0.5, 1.0, 1.5],
        gpu_usage=[[2.0, 3.0], [2.5, 3.5], [3.0, 4.0]],
    )
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)


def test_plot_lisa_psd() -> None:
    rng = np.random.default_rng(42)
    frequencies = np.geomspace(1e-4, 1e-1, 100)
    psd_values = {"A": rng.random(100), "E": rng.random(100)}
    fig, ax = plot_lisa_psd(frequencies, psd_values)
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)


def test_plot_lisa_noise_components() -> None:
    rng = np.random.default_rng(42)
    frequencies = np.geomspace(1e-4, 1e-1, 100)
    fig, ax = plot_lisa_noise_components(frequencies, rng.random(100), rng.random(100))
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)


def test_plot_cramer_rao_coverage() -> None:
    fig, ax = plot_cramer_rao_coverage(
        M=np.array([1e5, 2e5, 3e5]),
        qS=np.array([0.5, 1.0, 1.5]),
        phiS=np.array([0.1, 0.5, 1.0]),
        M_limits=(1e4, 1e6),
        qS_limits=(0.0, np.pi),
        phiS_limits=(0.0, 2 * np.pi),
    )
    assert isinstance(fig, Figure)
