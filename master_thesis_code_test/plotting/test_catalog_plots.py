"""Smoke tests for catalog_plots factory functions."""

import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from master_thesis_code.plotting.catalog_plots import (
    plot_bh_mass_distribution,
    plot_comoving_volume_sampling,
    plot_glade_completeness,
    plot_redshift_distribution,
)


def test_plot_bh_mass_distribution(
    sample_masses: npt.NDArray[np.float64],
) -> None:
    """Smoke test: plot_bh_mass_distribution returns (Figure, Axes)."""
    fig, ax = plot_bh_mass_distribution(sample_masses)
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)


def test_plot_redshift_distribution(
    sample_redshifts: npt.NDArray[np.float64],
) -> None:
    """Smoke test: plot_redshift_distribution returns (Figure, Axes)."""
    fig, ax = plot_redshift_distribution(sample_redshifts)
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)


def test_plot_glade_completeness(
    sample_distances: npt.NDArray[np.float64],
) -> None:
    """Smoke test: plot_glade_completeness returns (Figure, Axes)."""
    completeness = np.linspace(1.0, 0.3, len(sample_distances))
    fig, ax = plot_glade_completeness(sample_distances, completeness)
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)


def test_plot_comoving_volume_sampling(
    sample_redshifts: npt.NDArray[np.float64],
) -> None:
    """Smoke test: plot_comoving_volume_sampling returns (Figure, Axes)."""
    fig, ax = plot_comoving_volume_sampling(sample_redshifts)
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
