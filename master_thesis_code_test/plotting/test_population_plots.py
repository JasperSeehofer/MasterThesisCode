"""Tests for the population "where do the constraints come from" composite.

``plot_population_constraint_view`` orchestrates four already-tested
single-panel encodings into one 2x2 figure:

  - TOP-LEFT  : SNR x z driver scatter
  - TOP-RIGHT : Mollweide sky map (delegated to sky_plots)
  - BOTTOM-LEFT  : de-emphasized per-event spaghetti, colored by a per-event
    scalar (delegated to bayesian_plots.plot_event_posteriors color_by)
  - BOTTOM-RIGHT : canonical stacked combined posterior (delegated to
    bayesian_plots.plot_combined_posterior)

Only the 2x2 composition is novel; every sub-encoding is a reuse.
"""

import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from master_thesis_code.plotting._helpers import _PRESETS
from master_thesis_code.plotting.population_plots import plot_population_constraint_view


def _axes_list(axes: object) -> list[Axes]:
    """Flatten the composite's axes return (dict or array) into a list of Axes."""
    if isinstance(axes, dict):
        candidates: list[object] = list(axes.values())
    else:
        candidates = list(np.atleast_1d(np.asarray(axes, dtype=object)).ravel())
    return [a for a in candidates if isinstance(a, Axes)]


def _synth_inputs(
    n_events: int = 8,
    n_h: int = 60,
    peak_h: float = 0.73,
) -> tuple[
    npt.NDArray[np.float64],
    list[npt.NDArray[np.float64]],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    """Aligned synthetic inputs with a combined posterior peaked at *peak_h*."""
    rng = np.random.default_rng(0)
    h_values = np.linspace(0.6, 0.9, n_h)
    event_posteriors = [
        np.exp(-0.5 * ((h_values - (peak_h + rng.normal(0, 0.02))) / 0.05) ** 2)
        for _ in range(n_events)
    ]
    # Combined posterior with an unambiguous single peak at peak_h.
    combined = np.exp(-0.5 * ((h_values - peak_h) / 0.03) ** 2)
    theta_s = rng.uniform(0.2, np.pi - 0.2, n_events)
    phi_s = rng.uniform(0.0, 2 * np.pi, n_events)
    snr = rng.uniform(20.0, 80.0, n_events)
    redshift = rng.uniform(0.2, 1.4, n_events)
    return h_values, event_posteriors, combined, theta_s, phi_s, snr, redshift


def test_population_view_returns_figure_and_four_axes() -> None:
    """Smoke: composite returns (Figure, four Axes) without raising."""
    h, ev, comb, th, ph, snr, z = _synth_inputs()
    fig, axes = plot_population_constraint_view(h, ev, comb, 0.73, th, ph, snr, z, color_by="snr")
    assert isinstance(fig, Figure)
    flat = _axes_list(axes)
    assert len(flat) >= 4, f"expected 4 sub-panels, found {len(flat)}"


def test_population_view_uses_double_preset_size() -> None:
    """Figure size equals the get_figure 'double' preset (no hardcoded figsize)."""
    h, ev, comb, th, ph, snr, z = _synth_inputs()
    fig, _ = plot_population_constraint_view(h, ev, comb, 0.73, th, ph, snr, z)
    w, hgt = fig.get_size_inches()
    ew, eh = _PRESETS["double"]
    assert np.isclose(w, ew) and np.isclose(hgt, eh), (
        f"figure size ({w}, {hgt}) != double preset {_PRESETS['double']}"
    )


def test_population_view_spaghetti_has_color_by_colorbar() -> None:
    """The spaghetti panel surfaces color_by: a colorbar spanning the SNR range."""
    h, ev, comb, th, ph, snr, z = _synth_inputs()
    fig, _ = plot_population_constraint_view(h, ev, comb, 0.73, th, ph, snr, z, color_by="snr")
    cbars = [a for a in fig.axes if a.get_label() == "<colorbar>"]
    # At least one colorbar (the spaghetti color_by; the sky map also carries one).
    assert cbars, "expected a color_by colorbar on the spaghetti panel"
    norms = [c._colorbar.norm for c in cbars]  # type: ignore[attr-defined]
    spans_snr = any(
        np.isclose(n.vmin, float(snr.min())) and np.isclose(n.vmax, float(snr.max())) for n in norms
    )
    assert spans_snr, "no colorbar norm spans the supplied SNR range"


def test_population_view_stacked_is_dominant_over_spaghetti() -> None:
    """The stacked combined line is visually dominant (thicker) over spaghetti.

    The spaghetti per-event curves are de-emphasized (thin, alpha < 1) while the
    canonical combined posterior is the hero line — its linewidth strictly
    exceeds every per-event linewidth.
    """
    h, ev, comb, th, ph, snr, z = _synth_inputs()
    _, axes = plot_population_constraint_view(h, ev, comb, 0.73, th, ph, snr, z, color_by="snr")
    flat = _axes_list(axes)

    # Spaghetti panel: many thin de-emphasized lines (alpha < 1).
    spaghetti_axes = [
        a for a in flat if sum(1 for ln in a.get_lines() if (ln.get_alpha() or 1.0) < 1.0) >= 3
    ]
    assert spaghetti_axes, "expected a de-emphasized spaghetti panel"
    spaghetti_lw = [
        ln.get_linewidth() for ln in spaghetti_axes[0].get_lines() if (ln.get_alpha() or 1.0) < 1.0
    ]
    max_spaghetti_lw = max(spaghetti_lw)

    # Stacked panel: the canonical combined posterior — its main curve linewidth
    # exceeds the per-event spaghetti linewidth.
    stacked_axes = [a for a in flat if r"$h$" in a.get_xlabel() and a not in spaghetti_axes]
    assert stacked_axes, "expected a stacked-posterior panel"
    hero_lw = max(ln.get_linewidth() for ln in stacked_axes[0].get_lines())
    assert hero_lw > max_spaghetti_lw, (
        f"hero linewidth {hero_lw} should exceed spaghetti {max_spaghetti_lw}"
    )


def test_population_view_stacked_map_matches_canonical() -> None:
    """The stacked-posterior panel's MAP equals the fed combined posterior's peak."""
    peak_h = 0.71
    h, ev, comb, th, ph, snr, z = _synth_inputs(peak_h=peak_h)
    _, axes = plot_population_constraint_view(h, ev, comb, 0.73, th, ph, snr, z, color_by="snr")
    flat = _axes_list(axes)
    spaghetti_axes = [
        a for a in flat if sum(1 for ln in a.get_lines() if (ln.get_alpha() or 1.0) < 1.0) >= 3
    ]
    stacked_axes = [a for a in flat if r"$h$" in a.get_xlabel() and a not in spaghetti_axes]
    assert stacked_axes, "expected a stacked-posterior panel"
    ax_stacked = stacked_axes[0]
    # Recover the hero (thickest) line and assert its argmax-h equals peak_h.
    hero = max(ax_stacked.get_lines(), key=lambda ln: ln.get_linewidth())
    xd = hero.get_xdata()
    yd = hero.get_ydata()
    map_h = float(np.asarray(xd)[int(np.argmax(np.asarray(yd)))])
    assert np.isclose(map_h, peak_h, atol=0.01), f"stacked MAP {map_h} != canonical peak {peak_h}"
