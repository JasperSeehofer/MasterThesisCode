"""Once-per-event dedup of the out-of-grid quadrature warning (log hygiene)."""

import logging

import pytest

from darksiren_emri.bayesian_inference import bayesian_statistics as bs


@pytest.fixture(autouse=True)
def _reset_dedup_state() -> None:
    bs._quadrature_outside_grid_warned_events.clear()
    bs._quadrature_outside_grid_suppressed_repeats = 0


def test_warns_once_per_event_and_counts_repeats(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        bs._warn_quadrature_weight_outside_grid(7, 0.10, 0.02)
        bs._warn_quadrature_weight_outside_grid(7, 0.11, 0.03)
        bs._warn_quadrature_weight_outside_grid(7, 0.12, 0.04)
    warnings = [r for r in caplog.records if "quadrature weight outside" in r.message]
    assert len(warnings) == 1
    assert bs._quadrature_outside_grid_suppressed_repeats == 2


def test_distinct_events_each_warn(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        bs._warn_quadrature_weight_outside_grid(1, 0.10, 0.02)
        bs._warn_quadrature_weight_outside_grid(2, 0.10, 0.02)
    warnings = [r for r in caplog.records if "quadrature weight outside" in r.message]
    assert len(warnings) == 2
    assert bs._quadrature_outside_grid_suppressed_repeats == 0
