"""Interactive Plotly figure factory functions for EMRI thesis results.

Provides browser-explorable versions of 4 key thesis figures:
- Combined H0 posterior
- Sky localization map
- Fisher matrix error ellipses
- H0 convergence

All factory functions return ``plotly.graph_objects.Figure`` instances.
Call ``fig.write_html(path, include_plotlyjs="cdn")`` to export.

The top-level :func:`generate_all_interactive` convenience function loads
data from a working directory and writes all 4 HTML files.
"""

import logging
import os
import re

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from master_thesis_code.plotting._colors import CYCLE, EDGE, REFERENCE, TRUTH
from master_thesis_code.plotting._data import PARAMETER_NAMES
from master_thesis_code.plotting._labels import LABELS

_LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

# Default subsets for convergence analysis (mirrors convergence_plots._DEFAULT_SUBSETS)
_DEFAULT_SUBSETS: list[int] = [1, 5, 10, 25, 50, 100]

# Default parameter pairs for Fisher ellipses (mirrors fisher_plots._DEFAULT_PAIRS)
_DEFAULT_PAIRS: list[tuple[str, str]] = [
    ("M", "mu"),
    ("luminosity_distance", "qS"),
    ("qS", "phiS"),
]

# Planck 2018 and SH0ES reference h ranges (dimensionless h = H0/100)
_PLANCK_H_RANGE: tuple[float, float] = (0.6736 - 0.0054, 0.6736 + 0.0054)
_SHOES_H_RANGE: tuple[float, float] = (0.7304 - 0.0104, 0.7304 + 0.0104)


def _strip_latex(label: str) -> str:
    """Strip LaTeX markup from a label string for use as a plain-text axis label.

    Handles the specific patterns present in :data:`LABELS`. This is NOT a
    general LaTeX parser -- only the patterns used in this project are handled.

    Parameters
    ----------
    label:
        LaTeX-formatted label string, e.g. ``r"$M_\\bullet \\, [M_\\odot]$"``.

    Returns
    -------
    str
        Plain-text version, e.g. ``"M_bullet [M_sun]"``.
    """
    s = label
    # Remove surrounding $
    s = s.replace("$", "")
    # \mathrm{...} -> contents
    s = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", s)
    # Physics symbols
    s = s.replace(r"\bullet", "bullet")
    s = s.replace(r"\odot", "sun")
    s = s.replace(r"\rho", "rho")
    s = s.replace(r"\theta", "theta")
    s = s.replace(r"\phi", "phi")
    s = s.replace(r"\Phi", "Phi")
    s = s.replace(r"\mu", "mu")
    # Thin space
    s = s.replace(r"\,", " ")
    # Subscript/superscript (just keep inner text)
    s = re.sub(r"_\{([^}]*)\}", r"_\1", s)
    s = re.sub(r"\^\{([^}]*)\}", r"^\1", s)
    # Collapse multiple spaces
    s = re.sub(r"  +", " ", s)
    return s.strip()


def _credible_interval_bounds(
    h_values: npt.NDArray[np.float64],
    posterior: npt.NDArray[np.float64],
    level: float = 0.68,
) -> tuple[float, float]:
    """Return (lo, hi) bounds of the central credible interval at *level*.

    Parameters
    ----------
    h_values:
        Grid of h values.
    posterior:
        Posterior density on *h_values*.
    level:
        Probability mass enclosed (default 68%).

    Returns
    -------
    tuple[float, float]
        ``(lo, hi)`` bounds.
    """
    norm = np.trapezoid(posterior, h_values)
    if norm <= 0:
        return float(h_values[0]), float(h_values[-1])
    p = posterior / norm
    dh = np.gradient(h_values)
    cdf = np.cumsum(p * dh)
    cdf /= cdf[-1]
    lo = float(np.interp((1.0 - level) / 2.0, cdf, h_values))
    hi = float(np.interp((1.0 + level) / 2.0, cdf, h_values))
    return lo, hi


def _credible_interval_width(
    h_values: npt.NDArray[np.float64],
    posterior: npt.NDArray[np.float64],
    level: float = 0.68,
) -> float:
    """Return the width of the central credible interval at *level*."""
    lo, hi = _credible_interval_bounds(h_values, posterior, level)
    return hi - lo


# ---------------------------------------------------------------------------
# Public factory functions
# ---------------------------------------------------------------------------


def interactive_combined_posterior(
    h_values: npt.NDArray[np.float64],
    posterior: npt.NDArray[np.float64],
    true_h: float,
    *,
    label: str | None = None,
    show_credible: bool = True,
    show_references: bool = True,
) -> go.Figure:
    """Interactive combined H0 posterior figure.

    Parameters
    ----------
    h_values:
        Grid of h = H0/100 values.
    posterior:
        Posterior density evaluated on *h_values*.
    true_h:
        True (injected) h value; shown as a vertical line.
    label:
        Optional trace name override.
    show_credible:
        If True, add 68% and 95% credible interval shading.
    show_references:
        If True, add Planck and SH0ES reference bands.

    Returns
    -------
    go.Figure
        Plotly figure.
    """
    trace_name = label or "Posterior"
    fig = go.Figure()

    # Convert hex color to rgba string for fill
    _hex = CYCLE[0].lstrip("#")
    _r, _g, _b = int(_hex[0:2], 16), int(_hex[2:4], 16), int(_hex[4:6], 16)
    _fill_color = f"rgba({_r},{_g},{_b},0.15)"

    # Main posterior trace
    fig.add_trace(
        go.Scatter(
            x=h_values.tolist(),
            y=posterior.tolist(),
            mode="lines",
            name=trace_name,
            line={"color": CYCLE[0], "width": 2},
            fill="tozeroy",
            fillcolor=_fill_color,
            hovertemplate="h = %{x:.4f}<br>Density = %{y:.4f}<extra></extra>",
        )
    )

    # Credible intervals
    if show_credible:
        for level, alpha_hex in ((0.95, "33"), (0.68, "55")):
            lo, hi = _credible_interval_bounds(h_values, posterior, level)
            label_ci = f"{int(level * 100)}% CI"
            fig.add_vrect(
                x0=lo,
                x1=hi,
                fillcolor=CYCLE[0],
                opacity=0.15 if level == 0.95 else 0.25,
                line_width=0,
                annotation_text=label_ci,
                annotation_position="top left",
                annotation_font_size=11,
                name=label_ci,
                legendgroup=label_ci,
                showlegend=True,
            )

    # Reference bands
    if show_references:
        for (rlo, rhi), rname, rcolor in (
            (_PLANCK_H_RANGE, "Planck 2018", REFERENCE),
            (_SHOES_H_RANGE, "SH0ES", CYCLE[4]),
        ):
            fig.add_vrect(
                x0=rlo,
                x1=rhi,
                fillcolor=rcolor,
                opacity=0.2,
                line_color=rcolor,
                line_width=1,
                name=rname,
                legendgroup=rname,
                showlegend=True,
            )

    # Truth line
    fig.add_vline(
        x=true_h,
        line_color=TRUTH,
        line_dash="dash",
        line_width=2,
        annotation_text="Truth",
        annotation_position="top right",
    )

    fig.update_layout(
        title="Combined H\u2080 Posterior",
        xaxis_title=_strip_latex(LABELS["h"]),
        yaxis_title="Posterior density",
        legend={"orientation": "v"},
        hovermode="x unified",
    )
    return fig


def interactive_sky_map(
    theta_s: npt.NDArray[np.float64],
    phi_s: npt.NDArray[np.float64],
    snr: npt.NDArray[np.float64],
    *,
    redshifts: npt.NDArray[np.float64] | None = None,
    distances: npt.NDArray[np.float64] | None = None,
) -> go.Figure:
    """Interactive sky localization map.

    Parameters
    ----------
    theta_s:
        Source colatitude in radians, shape (N,), range [0, pi].
    phi_s:
        Source longitude in radians, shape (N,), range [0, 2*pi].
    snr:
        Signal-to-noise ratio per event, shape (N,).
    redshifts:
        Optional redshifts per event, shape (N,).
    distances:
        Optional luminosity distances in Mpc, shape (N,).

    Returns
    -------
    go.Figure
        Plotly Scattergeo figure with Mollweide projection.
    """
    # Convert: colatitude [0, pi] -> latitude [-90, 90]
    lat = np.degrees(np.pi / 2.0 - theta_s)
    # Convert: phi [0, 2pi] -> longitude [-180, 180]
    lon = np.degrees(phi_s)
    lon = np.where(lon > 180.0, lon - 360.0, lon)

    n_events = len(snr)
    event_indices = list(range(n_events))

    # Build hover text
    hover_parts = ["Event %{customdata[0]}", "SNR = %{customdata[1]:.2f}"]
    custom_data: list[list[float | int]] = [
        [event_indices[i], float(snr[i])] for i in range(n_events)
    ]
    if redshifts is not None:
        hover_parts.append("z = %{customdata[2]:.4f}")
        custom_data = [cd + [float(redshifts[i])] for i, cd in enumerate(custom_data)]
    if distances is not None:
        col_idx = 3 if redshifts is not None else 2
        hover_parts.append(f"d_L = %{{customdata[{col_idx}]:.1f}} Mpc")
        custom_data = [cd + [float(distances[i])] for i, cd in enumerate(custom_data)]
    hover_template = "<br>".join(hover_parts) + "<extra></extra>"

    fig = go.Figure(
        go.Scattergeo(
            lat=lat.tolist(),
            lon=lon.tolist(),
            mode="markers",
            marker={
                "color": snr.tolist(),
                "colorscale": "Viridis",
                "showscale": True,
                "colorbar": {"title": "SNR"},
                "size": 8,
                "opacity": 0.8,
                "line": {"width": 0.5, "color": EDGE},
            },
            customdata=custom_data,
            hovertemplate=hover_template,
            name="EMRI events",
        )
    )

    fig.update_geos(
        projection_type="mollweide",
        showland=False,
        showcoastlines=False,
        showframe=True,
        bgcolor="#f8f8f8",
        lataxis_showgrid=True,
        lonaxis_showgrid=True,
        lataxis_gridwidth=0.5,
        lonaxis_gridwidth=0.5,
    )
    fig.update_layout(
        title="EMRI Sky Localization Map",
        margin={"l": 0, "r": 0, "t": 40, "b": 0},
    )
    return fig


def interactive_fisher_ellipses(
    events: list[tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]],
    pairs: list[tuple[str, str]] | None = None,
    *,
    sigma_levels: tuple[float, ...] = (1.0, 2.0),
) -> go.Figure:
    """Interactive Fisher matrix error ellipses.

    Parameters
    ----------
    events:
        List of ``(covariance_14x14, param_values_14)`` tuples.
    pairs:
        Parameter pairs to show. Defaults to
        ``[("M", "mu"), ("luminosity_distance", "qS"), ("qS", "phiS")]``.
    sigma_levels:
        Sigma levels for ellipse boundaries (number of sigma).

    Returns
    -------
    go.Figure
        Plotly figure with one subplot column per parameter pair.
    """

    if pairs is None:
        pairs = _DEFAULT_PAIRS

    n_pairs = len(pairs)
    subplot_titles = [f"{p[0]} vs {p[1]}" for p in pairs]

    fig = make_subplots(
        rows=1,
        cols=n_pairs,
        subplot_titles=subplot_titles,
        shared_xaxes=False,
    )

    n_theta = 100
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta)

    for pair_idx, (name_x, name_y) in enumerate(pairs):
        idx_x = PARAMETER_NAMES.index(name_x)
        idx_y = PARAMETER_NAMES.index(name_y)
        col = pair_idx + 1

        for ev_idx, (cov, vals) in enumerate(events):
            color = CYCLE[ev_idx % len(CYCLE)]
            cx = float(vals[idx_x])
            cy = float(vals[idx_y])
            indices = [idx_x, idx_y]
            cov_2x2: npt.NDArray[np.float64] = cov[np.ix_(indices, indices)]
            eigenvalues, eigenvectors = np.linalg.eigh(cov_2x2)
            eigenvalues = np.maximum(eigenvalues, 0.0)

            for level_idx, level in enumerate(sorted(sigma_levels, reverse=True)):
                # Parametric ellipse in principal-axis frame, then rotate
                a_ax = level * float(np.sqrt(eigenvalues[1]))
                b_ax = level * float(np.sqrt(eigenvalues[0]))
                x_local = a_ax * np.cos(theta)
                y_local = b_ax * np.sin(theta)
                # Rotate by eigenvectors
                x_rot = eigenvectors[0, 1] * x_local + eigenvectors[0, 0] * y_local
                y_rot = eigenvectors[1, 1] * x_local + eigenvectors[1, 0] * y_local
                x_ellipse = (cx + x_rot).tolist()
                y_ellipse = (cy + y_rot).tolist()

                show_legend = level_idx == 0 and pair_idx == 0
                alpha = 0.35 / level

                fig.add_trace(
                    go.Scatter(
                        x=x_ellipse,
                        y=y_ellipse,
                        mode="lines",
                        fill="toself",
                        fillcolor=color,
                        opacity=alpha,
                        line={"color": color, "width": 1},
                        name=f"Event {ev_idx} ({level:.0f}\u03c3)",
                        legendgroup=f"event_{ev_idx}",
                        showlegend=show_legend,
                        hovertemplate=(
                            f"Event {ev_idx}, {level:.0f}\u03c3<br>"
                            f"{name_x} = %{{x:.4g}}<br>{name_y} = %{{y:.4g}}"
                            "<extra></extra>"
                        ),
                    ),
                    row=1,
                    col=col,
                )

        # Axis labels
        x_label_key = name_x if name_x in LABELS else name_x
        y_label_key = name_y if name_y in LABELS else name_y
        x_label = _strip_latex(LABELS.get(x_label_key, x_label_key))
        y_label = _strip_latex(LABELS.get(y_label_key, y_label_key))
        fig.update_xaxes(title_text=x_label, row=1, col=col)
        fig.update_yaxes(title_text=y_label, row=1, col=col)

    fig.update_layout(title="Fisher Matrix Error Ellipses")
    return fig


def interactive_h0_convergence(
    h_values: npt.NDArray[np.float64],
    event_posteriors: list[npt.NDArray[np.float64]],
    *,
    true_h: float | None = None,
    subset_sizes: list[int] | None = None,
    seed: int = 42,
    level: float = 0.68,
) -> go.Figure:
    """Interactive H0 convergence two-panel figure.

    Parameters
    ----------
    h_values:
        Grid of h values shared across posteriors.
    event_posteriors:
        Per-event posterior arrays on *h_values*.
    true_h:
        Optional truth value; shown as vertical line on left panel.
    subset_sizes:
        List of event subset sizes to show.
    seed:
        RNG seed for reproducible subset selection.
    level:
        Credible interval probability mass (default 68%).

    Returns
    -------
    go.Figure
        Two-panel Plotly figure (posteriors left, CI width right).
    """
    n_events = len(event_posteriors)

    if subset_sizes is None:
        sizes = [s for s in _DEFAULT_SUBSETS if s <= n_events]
        if not sizes:
            sizes = [n_events]
    else:
        sizes = [min(s, n_events) for s in subset_sizes]

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["Combined Posterior vs N", "CI Width vs N"],
    )

    rng = np.random.default_rng(seed)
    ci_widths: list[float] = []
    sizes_used: list[int] = []

    for idx, n in enumerate(sizes):
        indices = rng.choice(n_events, size=n, replace=False)
        log_posteriors = [np.log(np.maximum(event_posteriors[int(i)], 1e-300)) for i in indices]
        log_combined = np.sum(log_posteriors, axis=0)
        log_combined -= log_combined.max()
        combined = np.exp(log_combined)
        norm = np.trapezoid(combined, h_values)
        if norm > 0:
            combined /= norm

        color = CYCLE[idx % len(CYCLE)]
        fig.add_trace(
            go.Scatter(
                x=h_values.tolist(),
                y=combined.tolist(),
                mode="lines",
                name=f"N={n}",
                line={"color": color, "width": 2},
                legendgroup=f"N{n}",
                hovertemplate=f"N={n}<br>h = %{{x:.4f}}<br>Density = %{{y:.4f}}<extra></extra>",
            ),
            row=1,
            col=1,
        )

        width = _credible_interval_width(h_values, combined, level=level)
        ci_widths.append(width)
        sizes_used.append(n)

    # Truth line on left panel
    if true_h is not None:
        fig.add_vline(
            x=true_h,
            line_color=TRUTH,
            line_dash="dash",
            line_width=2,
            row=1,
            col=1,
            annotation_text="Truth",
        )

    # Right panel: CI width vs N
    sizes_arr = np.asarray(sizes_used, dtype=np.float64)
    fig.add_trace(
        go.Scatter(
            x=sizes_arr.tolist(),
            y=ci_widths,
            mode="lines+markers",
            name=f"{int(level * 100)}% CI width",
            line={"color": CYCLE[0]},
            marker={"size": 8},
            hovertemplate="N = %{x}<br>CI width = %{y:.4f}<extra></extra>",
        ),
        row=1,
        col=2,
    )

    # 1/sqrt(N) reference curve
    if len(sizes_used) > 1 and ci_widths[0] > 0:
        ref = ci_widths[0] * float(np.sqrt(sizes_arr[0])) / np.sqrt(sizes_arr)
        fig.add_trace(
            go.Scatter(
                x=sizes_arr.tolist(),
                y=ref.tolist(),
                mode="lines",
                name="1/sqrt(N) ref",
                line={"color": CYCLE[1], "dash": "dash"},
                opacity=0.6,
                hovertemplate="N = %{x}<br>1/sqrt(N) ref = %{y:.4f}<extra></extra>",
            ),
            row=1,
            col=2,
        )

    h_label = _strip_latex(LABELS["h"])
    fig.update_xaxes(title_text=h_label, row=1, col=1)
    fig.update_yaxes(title_text="Posterior density", row=1, col=1)
    fig.update_xaxes(title_text="Number of events", row=1, col=2)
    fig.update_yaxes(title_text=f"{int(level * 100)}% CI width", row=1, col=2)
    fig.update_layout(title="H\u2080 Posterior Convergence")
    return fig


# ---------------------------------------------------------------------------
# Convenience entry point
# ---------------------------------------------------------------------------


def generate_all_interactive(output_dir: str, data_dir: str) -> list[str]:
    """Load data from *data_dir* and write all 4 interactive HTML figures to *output_dir*.

    Gracefully skips figures when required data is missing -- logs a warning and
    continues to the next figure.

    Parameters
    ----------
    output_dir:
        Directory where HTML files are written.
    data_dir:
        Working directory containing CRB CSVs and posterior JSON subdirectories.

    Returns
    -------
    list[str]
        Paths to all HTML files that were written successfully.
    """
    import glob
    from pathlib import Path

    import pandas as pd

    os.makedirs(output_dir, exist_ok=True)

    written: list[str] = []

    # ------------------------------------------------------------------
    # Data loading helpers
    # ------------------------------------------------------------------

    # Search both data_dir/ and data_dir/simulations/ for data files
    _search_dirs = [Path(data_dir)]
    _sim_dir = Path(data_dir) / "simulations"
    if _sim_dir.is_dir():
        _search_dirs.insert(0, _sim_dir)

    def _load_crb_data() -> pd.DataFrame | None:
        for search in _search_dirs:
            csv_files = sorted(glob.glob(str(search / "*cramer_rao_bounds*.csv")))
            if csv_files:
                frames = [pd.read_csv(f) for f in csv_files]
                return pd.concat(frames, ignore_index=True)
        return None

    def _load_posteriors(
        subdir: str,
    ) -> tuple[npt.NDArray[np.float64], list[npt.NDArray[np.float64]]] | None:
        from master_thesis_code.bayesian_inference.posterior_combination import (
            load_posterior_jsons,
        )

        posteriors_dir: Path | None = None
        for search in _search_dirs:
            candidate = search / subdir
            if candidate.is_dir():
                posteriors_dir = candidate
                break
        if posteriors_dir is None:
            return None
        if not posteriors_dir.is_dir():
            return None
        try:
            h_values_list, event_likelihoods = load_posterior_jsons(posteriors_dir)
            h_values: npt.NDArray[np.float64] = np.array(h_values_list, dtype=np.float64)
            event_posteriors: list[npt.NDArray[np.float64]] = []
            for event_idx in sorted(event_likelihoods.keys()):
                lh = event_likelihoods[event_idx]
                event_posteriors.append(
                    np.array([lh.get(h, 0.0) for h in h_values_list], dtype=np.float64)
                )
            return h_values, event_posteriors
        except (FileNotFoundError, ValueError):
            return None

    # Try both posterior subdirectory variants
    post_data = _load_posteriors("posteriors") or _load_posteriors("posteriors_with_bh_mass")
    crb_df = _load_crb_data()

    # ------------------------------------------------------------------
    # Figure 1: Combined H0 posterior
    # ------------------------------------------------------------------
    if post_data is not None:
        h_values, event_posteriors = post_data
        try:
            from master_thesis_code.constants import H as TRUE_H

            # Combine all event posteriors (log-sum-exp)
            log_posts = [np.log(np.maximum(p, 1e-300)) for p in event_posteriors]
            log_combined = np.sum(log_posts, axis=0)
            log_combined -= log_combined.max()
            combined: npt.NDArray[np.float64] = np.exp(log_combined)
            norm = np.trapezoid(combined, h_values)
            if norm > 0:
                combined /= norm

            fig1 = interactive_combined_posterior(h_values, combined, TRUE_H)
            path1 = os.path.join(output_dir, "combined_posterior.html")
            fig1.write_html(path1, include_plotlyjs="cdn")
            written.append(path1)
            _LOGGER.info("Written: %s", path1)
        except Exception as exc:
            _LOGGER.warning("Skipping combined_posterior.html: %s", exc)
    else:
        _LOGGER.warning(
            "No posterior data found in %s -- skipping combined_posterior.html", data_dir
        )

    # ------------------------------------------------------------------
    # Figure 2: Sky map
    # ------------------------------------------------------------------
    if crb_df is not None and "qS" in crb_df.columns and "phiS" in crb_df.columns:
        try:
            theta_s: npt.NDArray[np.float64] = crb_df["qS"].to_numpy(dtype=np.float64)
            phi_s: npt.NDArray[np.float64] = crb_df["phiS"].to_numpy(dtype=np.float64)
            snr_col = "SNR" if "SNR" in crb_df.columns else None
            snr_vals: npt.NDArray[np.float64] = (
                crb_df[snr_col].to_numpy(dtype=np.float64)
                if snr_col is not None
                else np.ones(len(theta_s), dtype=np.float64)
            )
            redshift_vals: npt.NDArray[np.float64] | None = (
                crb_df["z"].to_numpy(dtype=np.float64) if "z" in crb_df.columns else None
            )
            dist_vals: npt.NDArray[np.float64] | None = (
                crb_df["luminosity_distance"].to_numpy(dtype=np.float64)
                if "luminosity_distance" in crb_df.columns
                else None
            )
            fig2 = interactive_sky_map(
                theta_s, phi_s, snr_vals, redshifts=redshift_vals, distances=dist_vals
            )
            path2 = os.path.join(output_dir, "sky_map.html")
            fig2.write_html(path2, include_plotlyjs="cdn")
            written.append(path2)
            _LOGGER.info("Written: %s", path2)
        except Exception as exc:
            _LOGGER.warning("Skipping sky_map.html: %s", exc)
    else:
        _LOGGER.warning("No CRB data with sky position columns -- skipping sky_map.html")

    # ------------------------------------------------------------------
    # Figure 3: Fisher ellipses
    # ------------------------------------------------------------------
    if crb_df is not None:
        try:
            from master_thesis_code.plotting._data import reconstruct_covariance

            events_list: list[tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]] = []
            param_cols = [c for c in PARAMETER_NAMES if c in crb_df.columns]
            # Limit to first 10 events to keep the figure tractable
            for _, row in crb_df.head(10).iterrows():
                try:
                    cov = reconstruct_covariance(row)
                    param_vals: npt.NDArray[np.float64] = np.array(
                        [
                            float(row[name]) if name in row.index else 0.0
                            for name in PARAMETER_NAMES
                        ],
                        dtype=np.float64,
                    )
                    events_list.append((cov, param_vals))
                except (KeyError, ValueError):
                    continue

            if events_list:
                fig3 = interactive_fisher_ellipses(events_list)
                path3 = os.path.join(output_dir, "fisher_ellipses.html")
                fig3.write_html(path3, include_plotlyjs="cdn")
                written.append(path3)
                _LOGGER.info("Written: %s", path3)
            else:
                _LOGGER.warning("No valid CRB rows for Fisher ellipses -- skipping")
        except Exception as exc:
            _LOGGER.warning("Skipping fisher_ellipses.html: %s", exc)
    else:
        _LOGGER.warning("No CRB data -- skipping fisher_ellipses.html")

    # ------------------------------------------------------------------
    # Figure 4: H0 convergence
    # ------------------------------------------------------------------
    if post_data is not None:
        h_values, event_posteriors = post_data
        if len(event_posteriors) >= 1:
            try:
                from master_thesis_code.constants import H as TRUE_H

                fig4 = interactive_h0_convergence(h_values, event_posteriors, true_h=TRUE_H)
                path4 = os.path.join(output_dir, "h0_convergence.html")
                fig4.write_html(path4, include_plotlyjs="cdn")
                written.append(path4)
                _LOGGER.info("Written: %s", path4)
            except Exception as exc:
                _LOGGER.warning("Skipping h0_convergence.html: %s", exc)
        else:
            _LOGGER.warning("Not enough posterior data for convergence -- skipping")
    else:
        _LOGGER.warning("No posterior data -- skipping h0_convergence.html")

    return written
