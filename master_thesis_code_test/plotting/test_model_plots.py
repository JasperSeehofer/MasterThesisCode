"""Smoke tests for model_plots factory functions."""

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from master_thesis_code.plotting.model_plots import (
    plot_detection_probability_grid,
    plot_emri_distribution,
    plot_emri_rate,
    plot_emri_sampling,
)


def test_plot_emri_distribution() -> None:
    z = np.linspace(0.1, 2.0, 15)
    m = np.geomspace(1e4, 1e7, 10)
    Z, M_grid = np.meshgrid(z, m)
    dist = np.random.default_rng(42).random((10, 15))
    fig, ax = plot_emri_distribution(Z, M_grid, dist)
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)


def test_plot_emri_rate(sample_masses: np.ndarray) -> None:
    rates = np.random.default_rng(42).random(len(sample_masses))
    fig, ax = plot_emri_rate(sample_masses, rates)
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)


def test_plot_emri_sampling() -> None:
    rng = np.random.default_rng(42)
    z_events = rng.uniform(0.1, 2.0, 50)
    m_events = 10 ** rng.uniform(4, 7, 50)
    z_bins = np.linspace(0.1, 2.0, 10)
    m_bins = np.geomspace(1e4, 1e7, 10)
    fig, ax = plot_emri_sampling(z_events, m_events, z_bins, m_bins)
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)


def test_plot_detection_probability_grid() -> None:
    d_L = np.linspace(0.1, 10.0, 12)
    M = np.geomspace(1e4, 1e7, 8)
    D, MG = np.meshgrid(d_L, M)
    prob = np.random.default_rng(42).random((8, 12))
    fig, ax = plot_detection_probability_grid(D, MG, prob)
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
