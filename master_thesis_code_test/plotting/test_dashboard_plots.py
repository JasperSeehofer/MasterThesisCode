"""Smoke tests for the campaign dashboard composite factory."""

import numpy as np
import pytest
from matplotlib.figure import Figure

from master_thesis_code.plotting.dashboard_plots import plot_campaign_dashboard


@pytest.fixture()
def dashboard_data() -> dict:
    """Minimal synthetic data for all four dashboard panels."""
    rng = np.random.default_rng(42)
    h_values = np.linspace(0.6, 0.9, 50)
    posterior = np.exp(-((h_values - 0.73) ** 2) / 0.01)
    snr_values = rng.uniform(10, 100, 50)
    injected_z = rng.uniform(0.1, 2.0, 100)
    detected_z = injected_z[:60]
    theta_s = rng.uniform(0, np.pi, 50)
    phi_s = rng.uniform(0, 2 * np.pi, 50)
    return {
        "h_values": h_values,
        "posterior": posterior,
        "true_h": 0.73,
        "snr_values": snr_values,
        "injected_redshifts": injected_z,
        "detected_redshifts": detected_z,
        "theta_s": theta_s,
        "phi_s": phi_s,
        "sky_snr": snr_values,
    }


def test_returns_figure_and_dict_with_four_panels(
    dashboard_data: dict,
) -> None:
    """plot_campaign_dashboard returns (Figure, dict) with keys posterior, snr, yield, sky."""
    fig, axd = plot_campaign_dashboard(**dashboard_data)
    assert isinstance(fig, Figure)
    assert isinstance(axd, dict)
    assert set(axd.keys()) == {"posterior", "snr", "yield", "sky"}


def test_sky_panel_is_mollweide(dashboard_data: dict) -> None:
    """The sky panel uses Mollweide projection."""
    _, axd = plot_campaign_dashboard(**dashboard_data)
    assert axd["sky"].name == "mollweide"


def test_figure_width_is_double_preset(dashboard_data: dict) -> None:
    """Figure width is approximately 7.0 inches (preset='double')."""
    fig, _ = plot_campaign_dashboard(**dashboard_data)
    width = fig.get_size_inches()[0]
    assert width == pytest.approx(7.0, abs=0.1)


def test_all_panels_have_artists(dashboard_data: dict) -> None:
    """All 4 Axes have at least one artist (not empty)."""
    _, axd = plot_campaign_dashboard(**dashboard_data)
    for key in ("posterior", "snr", "yield", "sky"):
        ax = axd[key]
        has_content = (
            len(ax.lines) > 0
            or len(ax.patches) > 0
            or len(ax.collections) > 0
            or len(ax.images) > 0
        )
        assert has_content, f"Panel '{key}' has no artists"
