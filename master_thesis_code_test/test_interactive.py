"""Tests for interactive Plotly figure factory functions."""

import os
import tempfile

import numpy as np
import numpy.typing as npt
import pytest

plotly = pytest.importorskip("plotly")
import plotly.graph_objects as go  # noqa: E402

from master_thesis_code.plotting.interactive import (  # noqa: E402
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


class TestGenerateAllInteractive:
    def test_empty_data_returns_empty_list(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "interactive_out")
            result = generate_all_interactive(output_dir=output_dir, data_dir=tmpdir)
        assert result == []

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
