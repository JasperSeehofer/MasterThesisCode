"""Tests for interactive Plotly figure factory functions."""

import os
import re
import tempfile

import numpy as np
import numpy.typing as npt
import pytest

plotly = pytest.importorskip("plotly")
import plotly.graph_objects as go  # noqa: E402

from master_thesis_code.plotting._colors import (  # noqa: E402
    VARIANT_NO_MASS,
    VARIANT_WITH_MASS,
)
from master_thesis_code.plotting._plotly_theme import (  # noqa: E402
    HORIZON_TEMPLATE,
    WEB_FONT_SIZE,  # noqa: E402
    horizon_plotly_template,
)
from master_thesis_code.plotting.interactive import (  # noqa: E402
    _STATIC_TWINS,
    _strip_latex,
    generate_all_interactive,
    interactive_combined_posterior,
    interactive_fisher_ellipses,
    interactive_h0_convergence,
    interactive_sky_map,
)

# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------


def _make_h_values() -> npt.NDArray[np.float64]:
    return np.linspace(0.6, 0.8, 200, dtype=np.float64)


def _make_posterior(
    h_values: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    center = 0.73
    width = 0.02
    posterior = np.exp(-0.5 * ((h_values - center) / width) ** 2)
    result: npt.NDArray[np.float64] = (posterior / np.trapezoid(posterior, h_values)).astype(
        np.float64
    )
    return result


def _make_covariance(seed: int = 0) -> npt.NDArray[np.float64]:
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((14, 14)) * 1e-3
    result: npt.NDArray[np.float64] = (A @ A.T + np.eye(14) * 1e-4).astype(np.float64)
    return result


def _make_param_values(seed: int = 0) -> npt.NDArray[np.float64]:
    rng = np.random.default_rng(seed)
    vals = rng.standard_normal(14)
    # Set some sensible ranges
    vals[0] = abs(vals[0]) * 1e6 + 1e6  # M
    vals[1] = abs(vals[1]) * 10 + 10  # mu
    vals[7] = abs(vals[7]) % np.pi  # qS
    vals[8] = abs(vals[8]) % (2 * np.pi)  # phiS
    result: npt.NDArray[np.float64] = vals.astype(np.float64)
    return result


# ---------------------------------------------------------------------------
# HORIZON Plotly template tests (VR-INT-01)
# ---------------------------------------------------------------------------


class TestHorizonPlotlyTemplate:
    def test_colorway_navy_gold_first(self) -> None:
        """colorway[0:2] are the HORIZON navy/gold tokens imported from _colors."""
        tmpl = horizon_plotly_template()
        assert tuple(tmpl.layout.colorway[:2]) == (VARIANT_NO_MASS, VARIANT_WITH_MASS)
        assert (VARIANT_NO_MASS, VARIANT_WITH_MASS) == ("#1B2A4A", "#E8A317")

    def test_sequential_colorscale_is_cividis_not_viridis(self) -> None:
        """Sequential ramp derives from cividis (not Plotly default Viridis)."""
        tmpl = horizon_plotly_template()
        seq = tmpl.layout.colorscale.sequential
        assert len(seq) > 0
        # cividis endpoints: dark navy-blue at t=0, bright yellow at t=1.
        first_t, first_rgb = seq[0]
        last_t, last_rgb = seq[-1]
        assert first_t == 0.0
        assert last_t == 1.0
        # cividis dark end ~ rgb(0,34,78); Viridis dark end ~ rgb(68,1,84).
        assert first_rgb == "rgb(0,34,78)"
        # cividis bright end ~ rgb(254,232,56); never a green-yellow Viridis end.
        assert last_rgb == "rgb(254,232,56)"

    def test_combined_posterior_carries_template_colorway(self) -> None:
        """The factory applies the template (colorway present, not None)."""
        h_values = _make_h_values()
        posterior = _make_posterior(h_values)
        fig = interactive_combined_posterior(h_values, posterior, true_h=0.73)
        cw = fig.layout.template.layout.colorway
        assert cw is not None
        assert tuple(cw[:2]) == (VARIANT_NO_MASS, VARIANT_WITH_MASS)

    def test_sky_map_carries_template_colorway(self) -> None:
        """A second factory also carries the shared template colorway."""
        rng = np.random.default_rng(7)
        n = 30
        theta_s = rng.uniform(0.0, np.pi, n).astype(np.float64)
        phi_s = rng.uniform(0.0, 2 * np.pi, n).astype(np.float64)
        snr = rng.uniform(20.0, 90.0, n).astype(np.float64)
        fig = interactive_sky_map(theta_s, phi_s, snr)
        cw = fig.layout.template.layout.colorway
        assert cw is not None
        assert tuple(cw[:2]) == (VARIANT_NO_MASS, VARIANT_WITH_MASS)

    def test_module_singleton_matches_factory(self) -> None:
        """HORIZON_TEMPLATE singleton carries the same colorway as a fresh build."""
        assert tuple(HORIZON_TEMPLATE.layout.colorway[:2]) == tuple(
            horizon_plotly_template().layout.colorway[:2]
        )


# ---------------------------------------------------------------------------
# _strip_latex tests
# ---------------------------------------------------------------------------


class TestStripLatex:
    def test_m_bullet(self) -> None:
        from master_thesis_code.plotting._labels import LABELS

        result = _strip_latex(LABELS["M"])
        assert "$" not in result
        assert "bullet" in result
        assert "sun" in result

    def test_plain_h(self) -> None:
        from master_thesis_code.plotting._labels import LABELS

        result = _strip_latex(LABELS["h"])
        assert "$" not in result
        assert "h" in result

    def test_h0_label(self) -> None:
        from master_thesis_code.plotting._labels import LABELS

        result = _strip_latex(LABELS["H0"])
        assert "$" not in result
        assert "H0" in result or "H_0" in result or "km" in result

    def test_mathrm_stripped(self) -> None:
        result = _strip_latex(r"$\mathrm{Mpc}$")
        assert "mathrm" not in result
        assert "Mpc" in result


# ---------------------------------------------------------------------------
# Factory function tests
# ---------------------------------------------------------------------------


class TestInteractiveCombinedPosterior:
    def test_returns_figure(self) -> None:
        h_values = _make_h_values()
        posterior = _make_posterior(h_values)
        fig = interactive_combined_posterior(h_values, posterior, true_h=0.73)
        assert isinstance(fig, go.Figure)

    def test_has_traces(self) -> None:
        h_values = _make_h_values()
        posterior = _make_posterior(h_values)
        fig = interactive_combined_posterior(h_values, posterior, true_h=0.73)
        assert len(fig.data) > 0

    def test_no_credible_no_references(self) -> None:
        h_values = _make_h_values()
        posterior = _make_posterior(h_values)
        fig = interactive_combined_posterior(
            h_values,
            posterior,
            true_h=0.73,
            show_credible=False,
            show_references=False,
        )
        assert isinstance(fig, go.Figure)


class TestInteractiveSkyMap:
    def test_returns_figure(self) -> None:
        n = 50
        rng = np.random.default_rng(42)
        theta_s = rng.uniform(0.0, np.pi, n).astype(np.float64)
        phi_s = rng.uniform(0.0, 2 * np.pi, n).astype(np.float64)
        snr = rng.uniform(20.0, 100.0, n).astype(np.float64)
        fig = interactive_sky_map(theta_s, phi_s, snr)
        assert isinstance(fig, go.Figure)

    def test_with_optional_columns(self) -> None:
        n = 20
        rng = np.random.default_rng(0)
        theta_s = rng.uniform(0.0, np.pi, n).astype(np.float64)
        phi_s = rng.uniform(0.0, 2 * np.pi, n).astype(np.float64)
        snr = rng.uniform(20.0, 80.0, n).astype(np.float64)
        redshifts = rng.uniform(0.01, 1.0, n).astype(np.float64)
        distances = rng.uniform(100.0, 3000.0, n).astype(np.float64)
        fig = interactive_sky_map(theta_s, phi_s, snr, redshifts=redshifts, distances=distances)
        assert isinstance(fig, go.Figure)

    def test_single_event(self) -> None:
        theta_s = np.array([np.pi / 2], dtype=np.float64)
        phi_s = np.array([np.pi], dtype=np.float64)
        snr = np.array([30.0], dtype=np.float64)
        fig = interactive_sky_map(theta_s, phi_s, snr)
        assert isinstance(fig, go.Figure)


class TestInteractiveFisherEllipses:
    def test_returns_figure(self) -> None:
        events = [
            (_make_covariance(0), _make_param_values(0)),
            (_make_covariance(1), _make_param_values(1)),
        ]
        fig = interactive_fisher_ellipses(events)
        assert isinstance(fig, go.Figure)

    def test_single_event(self) -> None:
        events = [(_make_covariance(0), _make_param_values(0))]
        fig = interactive_fisher_ellipses(events)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_custom_pairs(self) -> None:
        events = [(_make_covariance(0), _make_param_values(0))]
        fig = interactive_fisher_ellipses(events, pairs=[("M", "mu")])
        assert isinstance(fig, go.Figure)

    def test_custom_sigma_levels(self) -> None:
        events = [(_make_covariance(0), _make_param_values(0))]
        fig = interactive_fisher_ellipses(events, sigma_levels=(1.0, 3.0))
        assert isinstance(fig, go.Figure)


class TestInteractiveH0Convergence:
    def test_returns_figure(self) -> None:
        h_values = _make_h_values()
        event_posteriors = [_make_posterior(h_values) for _ in range(20)]
        fig = interactive_h0_convergence(h_values, event_posteriors)
        assert isinstance(fig, go.Figure)

    def test_with_true_h(self) -> None:
        h_values = _make_h_values()
        event_posteriors = [_make_posterior(h_values) for _ in range(5)]
        fig = interactive_h0_convergence(h_values, event_posteriors, true_h=0.73)
        assert isinstance(fig, go.Figure)

    def test_custom_subset_sizes(self) -> None:
        h_values = _make_h_values()
        event_posteriors = [_make_posterior(h_values) for _ in range(10)]
        fig = interactive_h0_convergence(h_values, event_posteriors, subset_sizes=[2, 5, 10])
        assert isinstance(fig, go.Figure)

    def test_single_event(self) -> None:
        h_values = _make_h_values()
        event_posteriors = [_make_posterior(h_values)]
        fig = interactive_h0_convergence(h_values, event_posteriors)
        assert isinstance(fig, go.Figure)


# ---------------------------------------------------------------------------
# generate_all_interactive tests
# ---------------------------------------------------------------------------


def _write_synthetic_crb_csv(data_dir: str) -> str:
    """Write a minimal cramer_rao_bounds.csv so the sky map HTML is emitted.

    Returns the CSV path. Only the sky-map factory's required columns
    (``qS``, ``phiS``, ``SNR``) are provided so the synthetic data dir yields
    at least one HTML without needing a full posteriors tree.
    """
    rng = np.random.default_rng(123)
    n = 12
    rows = [
        "qS,phiS,SNR",
        *(
            f"{float(rng.uniform(0.1, np.pi - 0.1))},"
            f"{float(rng.uniform(0.0, 2 * np.pi))},"
            f"{float(rng.uniform(20.0, 90.0))}"
            for _ in range(n)
        ),
    ]
    csv_path = os.path.join(data_dir, "cramer_rao_bounds.csv")
    with open(csv_path, "w") as fh:
        fh.write("\n".join(rows) + "\n")
    return csv_path


class TestGenerateAllInteractive:
    def test_empty_data_returns_empty_list(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "interactive_out")
            result = generate_all_interactive(output_dir=output_dir, data_dir=tmpdir)
        assert result == []

    def test_plotly_js_written_to_output_dir(self) -> None:
        """include_plotlyjs='directory' drops a local plotly*.js into output_dir."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_synthetic_crb_csv(tmpdir)
            output_dir = os.path.join(tmpdir, "interactive_out")
            result = generate_all_interactive(output_dir=output_dir, data_dir=tmpdir)
            assert result, "expected at least one HTML from the synthetic CRB csv"
            js_files = [f for f in os.listdir(output_dir) if "plotly" in f and f.endswith(".js")]
            assert js_files, f"no local plotly*.js bundle in {output_dir}: {os.listdir(output_dir)}"

    def test_generated_html_is_offline_no_cdn(self) -> None:
        """Each HTML references the local plotly script and contains no CDN URL."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_synthetic_crb_csv(tmpdir)
            output_dir = os.path.join(tmpdir, "interactive_out")
            result = generate_all_interactive(output_dir=output_dir, data_dir=tmpdir)
            assert result
            for html_path in result:
                with open(html_path) as fh:
                    html_text = fh.read()
                # Local reference present (relative plotly script), no CDN URL.
                assert "plotly" in html_text
                assert "cdn.plot.ly" not in html_text
                assert "https://cdn" not in html_text
                # No absolute http(s) URL pointing at a remote plotly bundle.
                assert not re.search(r"https?://[^\"']*plotly[^\"']*\.js", html_text)

    def test_output_dir_created(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "new_subdir")
            assert not os.path.isdir(output_dir)
            generate_all_interactive(output_dir=output_dir, data_dir=tmpdir)
            assert os.path.isdir(output_dir)

    def test_returns_list(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            result = generate_all_interactive(output_dir=tmpdir, data_dir=tmpdir)
        assert isinstance(result, list)
        for path in result:
            assert isinstance(path, str)


# ---------------------------------------------------------------------------
# Per-group trace toggling tests (VR-INT-03)
# ---------------------------------------------------------------------------


def _write_single_event_fixture(data_dir: str, event_ids: list[int]) -> None:
    """Write synthetic posteriors/ + posteriors_with_bh_mass/ JSON for events.

    Each event gets a couple of candidate hosts in galaxy_likelihoods (2D
    channel) and a scalar per-event likelihood in the 1D channel, on a small
    h-grid, so interactive_single_event_detail can build real traces.
    """
    import json

    h_grid = [0.70, 0.73, 0.76]
    wm_dir = os.path.join(data_dir, "posteriors_with_bh_mass")
    no_dir = os.path.join(data_dir, "posteriors")
    os.makedirs(wm_dir, exist_ok=True)
    os.makedirs(no_dir, exist_ok=True)
    for hi, h in enumerate(h_grid):
        # galaxy_likelihoods: event_key -> [[gid, [num_no, den_no, num_w, den_w]], ...]
        gal = {
            str(eid): [
                [eid * 10 + 1, [1.0 + hi, 2.0, 0.5 + hi, 1.0]],
                [eid * 10 + 2, [0.5, 2.0, 0.2, 1.0]],
            ]
            for eid in event_ids
        }
        wm_payload: dict[str, object] = {"h": h, "galaxy_likelihoods": gal}
        for eid in event_ids:
            wm_payload[str(eid)] = [float(1.0 + 0.1 * hi + 0.01 * eid)]
        with open(os.path.join(wm_dir, f"h_{str(h).replace('.', '_')}.json"), "w") as fh:
            json.dump(wm_payload, fh)
        no_payload: dict[str, object] = {"h": h}
        for eid in event_ids:
            no_payload[str(eid)] = [float(1.0 + 0.05 * hi + 0.01 * eid)]
        with open(os.path.join(no_dir, f"h_{str(h).replace('.', '_')}.json"), "w") as fh:
            json.dump(no_payload, fh)


def _make_synthetic_bank(n_sizes: int = 4) -> object:
    """Build a minimal ImprovementBank for interactive_m_z_improvement."""
    from master_thesis_code.plotting.convergence_analysis import ImprovementBank

    sizes = [1, 5, 10, 25][:n_sizes]
    h_grid = np.linspace(0.6, 0.85, 40, dtype=np.float64)

    def _triplet() -> dict[str, list[float]]:
        return {
            "median": [float(1.0 / (i + 1)) for i in range(len(sizes))],
            "p16": [float(0.8 / (i + 1)) for i in range(len(sizes))],
            "p84": [float(1.2 / (i + 1)) for i in range(len(sizes))],
        }

    metric_keys = ["hdi68_width", "rel_precision", "kl_from_uniform", "bias_pct"]
    metrics = {k: _triplet() for k in metric_keys}
    rep = [np.exp(-0.5 * ((h_grid - 0.73) / 0.02) ** 2) for _ in sizes]
    return ImprovementBank(
        h_grid=h_grid,
        h_true=0.73,
        sizes=sizes,
        n_bootstrap=4,
        seed=0,
        metrics_no_mass={k: dict(v) for k, v in metrics.items()},
        metrics_with_mass={k: dict(v) for k, v in metrics.items()},
        fractional_improvement=_triplet(),
        effective_event_gain=_triplet(),
        jsd_bits=_triplet(),
        representative_posteriors_no_mass=list(rep),
        representative_posteriors_with_mass=list(rep),
        n_events_no_mass=max(sizes),
        n_events_with_mass=max(sizes),
    )


class TestPerGroupTraces:
    def test_single_event_traces_carry_event_legendgroup(self) -> None:
        from pathlib import Path

        from master_thesis_code.plotting.interactive import interactive_single_event_detail

        event_ids = [3, 7, 11]
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_single_event_fixture(tmpdir, event_ids)
            fig = interactive_single_event_detail(Path(tmpdir), event_ids)
        # Every trace carries an event legendgroup.
        groups = {tr.legendgroup for tr in fig.data}
        for eid in event_ids:
            assert f"event_{eid}" in groups
        assert all(
            isinstance(tr.legendgroup, str) and tr.legendgroup.startswith("event_")
            for tr in fig.data
        )

    def test_single_event_dropdown_toggles_one_group(self) -> None:
        from pathlib import Path

        from master_thesis_code.plotting.interactive import interactive_single_event_detail

        event_ids = [3, 7, 11]
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_single_event_fixture(tmpdir, event_ids)
            fig = interactive_single_event_detail(Path(tmpdir), event_ids)
        n_traces = len(fig.data)
        buttons = fig.layout.updatemenus[0].buttons
        assert len(buttons) == len(event_ids)
        for button, eid in zip(buttons, event_ids, strict=True):
            vis = button.args[0]["visible"]
            # Length-equals-total-traces invariant.
            assert len(vis) == n_traces
            # Exactly this event's group is visible; computed from membership.
            expected = [tr.legendgroup == f"event_{eid}" for tr in fig.data]
            assert list(vis) == expected
            assert any(vis)

    def test_single_event_initial_one_group_visible(self) -> None:
        from pathlib import Path

        from master_thesis_code.plotting.interactive import interactive_single_event_detail

        event_ids = [3, 7, 11]
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_single_event_fixture(tmpdir, event_ids)
            fig = interactive_single_event_detail(Path(tmpdir), event_ids)
        visible_groups = {tr.legendgroup for tr in fig.data if tr.visible is not False}
        assert visible_groups == {"event_3"}

    def test_m_z_metric_traces_carry_meta_group(self) -> None:
        from master_thesis_code.plotting.interactive import interactive_m_z_improvement

        bank = _make_synthetic_bank()
        fig = interactive_m_z_improvement(bank)
        # Each metric block's traces share a per-metric meta group.
        meta_groups = {
            tr.meta["group"] for tr in fig.data if isinstance(tr.meta, dict) and "group" in tr.meta
        }
        for key in ("hdi68_width", "rel_precision", "kl_from_uniform", "bias_pct"):
            assert f"metric_{key}" in meta_groups
        assert "ref" in meta_groups
        assert "panel_b" in meta_groups
        assert "panel_c" in meta_groups

    def test_m_z_dropdown_computed_from_group_membership(self) -> None:
        from master_thesis_code.plotting.interactive import interactive_m_z_improvement

        bank = _make_synthetic_bank()
        fig = interactive_m_z_improvement(bank)
        n_traces = len(fig.data)
        buttons = fig.layout.updatemenus[0].buttons
        metric_keys = ["hdi68_width", "rel_precision", "kl_from_uniform", "bias_pct"]
        assert len(buttons) == len(metric_keys)
        for button, key in zip(buttons, metric_keys, strict=True):
            vis = list(button.args[0]["visible"])
            # Length-equals-total-traces invariant (self-correcting on add).
            assert len(vis) == n_traces
            always_on = {"panel_b", "panel_c"}
            show_ref = key == "hdi68_width"
            expected = []
            for tr in fig.data:
                g = tr.meta.get("group") if isinstance(tr.meta, dict) else None
                expected.append(g == f"metric_{key}" or g in always_on or (g == "ref" and show_ref))
            assert vis == expected

    def test_m_z_returns_figure(self) -> None:
        from master_thesis_code.plotting.interactive import interactive_m_z_improvement

        bank = _make_synthetic_bank()
        fig = interactive_m_z_improvement(bank)
        assert isinstance(fig, go.Figure)


# ---------------------------------------------------------------------------
# Theme wiring + static-twin mapping tests (VR-INT-04)
# ---------------------------------------------------------------------------


class TestThemeAndStaticTwins:
    def test_generate_applies_web_theme_font(self) -> None:
        """The web theme (HORIZON web-scaled font) is in force on interactive output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_synthetic_crb_csv(tmpdir)
            output_dir = os.path.join(tmpdir, "interactive_out")
            # theme="web" must run without exception and emit at least one HTML.
            result = generate_all_interactive(output_dir=output_dir, data_dir=tmpdir, theme="web")
            assert result

        # The HORIZON template carries the web-scaled font size (not paper default),
        # which is what drives every interactive figure's typography.
        from master_thesis_code.plotting.interactive import interactive_sky_map

        rng = np.random.default_rng(3)
        sky = interactive_sky_map(
            rng.uniform(0.1, 3.0, 5).astype(np.float64),
            rng.uniform(0.0, 6.0, 5).astype(np.float64),
            rng.uniform(20.0, 80.0, 5).astype(np.float64),
        )
        assert sky.layout.template.layout.font.size == WEB_FONT_SIZE

    def test_static_twins_cover_all_eight(self) -> None:
        """_STATIC_TWINS maps exactly the 8 interactive factories."""
        expected = {
            "interactive_combined_posterior",
            "interactive_sky_map",
            "interactive_fisher_ellipses",
            "interactive_h0_convergence",
            "interactive_m_z_improvement",
            "interactive_single_event_detail",
            "interactive_closure_test_overlay",
            "interactive_catalog_completeness",
        }
        assert set(_STATIC_TWINS) == expected

    @pytest.mark.parametrize("interactive_name", sorted(_STATIC_TWINS))
    def test_static_twin_importable_and_callable(self, interactive_name: str) -> None:
        """Each interactive factory's static twin imports and is callable."""
        import importlib

        twin_spec = _STATIC_TWINS[interactive_name]
        module_path, func_name = twin_spec.split(":")
        module = importlib.import_module(module_path)
        func = getattr(module, func_name)
        assert callable(func)

    @pytest.mark.parametrize("interactive_name", sorted(_STATIC_TWINS))
    def test_interactive_factory_exists(self, interactive_name: str) -> None:
        """Each interactive_* key in the mapping exists as a module function."""
        import master_thesis_code.plotting.interactive as interactive_mod

        func = getattr(interactive_mod, interactive_name, None)
        assert callable(func), f"{interactive_name} missing from interactive.py"

    def test_smoke_generate_no_exception(self) -> None:
        """generate_all_interactive runs over synthetic data with no exception."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_synthetic_crb_csv(tmpdir)
            output_dir = os.path.join(tmpdir, "interactive_out")
            result = generate_all_interactive(output_dir=output_dir, data_dir=tmpdir)
            assert isinstance(result, list)
            assert result  # at least the sky map
