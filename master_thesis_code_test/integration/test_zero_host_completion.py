"""Zero-host event handling in ``BayesianStatistics.evaluate`` (issue #29).

When ``get_possible_hosts_from_ball_tree`` returns ``None`` (no catalogue
galaxy inside the event's sky-ellipse x redshift window), the evaluate loop
historically dropped the event silently (``continue`` after the
``posterior_data[index] = []`` init) — the event contributed NO factor to the
joint likelihood. On the depth-1.5 Phase-2 campaign this dropped 58% of all
events and railed the combined posterior (seed1000 diagnosis, 2026-07-10;
``results/campaign_phase2_runs/run_20260703_seed1000/FINDINGS_COMBINE_20260710.md``).

This module pins that behavior. It reuses the deterministic synthetic pipeline
of ``test_pipeline_parity`` and forces the zero-host path for one chosen event
by wrapping the catalogue lookup.

Regression discipline (physics-change protocol): the first commit of this file
asserts the OLD behavior (silent skip -> empty entry); the [PHYSICS] fallback
commit flips the assertions to the NEW behavior (pure-completion likelihood
``p_i = B_num/D``, Gray et al. 2020 arXiv:1908.06050 Eqs. 29+32) so the diff
makes the behavioral change explicit.
"""

import csv
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from master_thesis_code_test.integration.conftest import build_galaxy_catalog_for_n_detections
from master_thesis_code_test.integration.test_pipeline_parity import _setup_sim_env

if TYPE_CHECKING:
    from master_thesis_code.cosmological_model import Model1CrossCheck

# Fixture has 5 events (CRB indices 0-4), all passing the quality filters
# (see golden/pipeline_parity_pins.json). Lookups happen in index order.
_ZERO_HOST_EVENT = 2
_N_EVENTS = 5
_H_VALUE = 0.73


def _run_evaluate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cosmological_model: "Model1CrossCheck",
    zero_host_event: int | None,
) -> dict[str, Any]:
    """Run one single-h evaluate(); optionally force one event's lookup to None.

    Returns per-event 1D/2D likelihood lists plus the per-event diagnostics rows
    (w_G, L_cat, B_num, L_comp, combined) read back from the CSV the pipeline
    writes.
    """
    tmp_path.mkdir(parents=True, exist_ok=True)
    _setup_sim_env(tmp_path, monkeypatch)

    from master_thesis_code.bayesian_inference.bayesian_statistics import BayesianStatistics

    galaxy_catalog = build_galaxy_catalog_for_n_detections(_N_EVENTS)

    if zero_host_event is not None:
        real_lookup = galaxy_catalog.get_possible_hosts_from_ball_tree
        call_state = {"n": 0}

        def _lookup_with_zero_host(*args: Any, **kwargs: Any) -> Any:
            event_index = call_state["n"]
            call_state["n"] += 1
            if event_index == zero_host_event:
                return None
            return real_lookup(*args, **kwargs)

        monkeypatch.setattr(
            galaxy_catalog, "get_possible_hosts_from_ball_tree", _lookup_with_zero_host
        )

    bayesian_stats = BayesianStatistics()
    bayesian_stats.evaluate(
        galaxy_catalog=galaxy_catalog,
        cosmological_model=cosmological_model,
        h_value=_H_VALUE,
    )

    diagnostics: dict[int, dict[str, float]] = {}
    diag_path = tmp_path / "simulations" / "diagnostics" / "event_likelihoods.csv"
    if diag_path.exists():
        with open(diag_path) as f:
            for row in csv.DictReader(f):
                diagnostics[int(row["event_idx"])] = {
                    k: float(v) for k, v in row.items() if k != "event_idx"
                }

    json_path = tmp_path / "simulations" / "posteriors" / "h_0_73.json"
    with open(json_path) as f:
        written_1d = json.load(f)

    return {
        "1d": {k: list(v) for k, v in bayesian_stats.posterior_data.items() if isinstance(k, int)},
        "2d": {
            k: list(v)
            for k, v in bayesian_stats.posterior_data_with_bh_mass.items()
            if isinstance(k, int) and isinstance(v, list)
        },
        "diagnostics": diagnostics,
        "written_1d": written_1d,
    }


@pytest.mark.slow
def test_zero_host_event_is_silently_skipped_legacy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cosmological_model: "Model1CrossCheck",
) -> None:
    """OLD behavior pin: a zero-host event leaves an empty entry at every h.

    This is the silent-drop defect of issue #29 (bayesian_statistics.py
    ``if possible_hosts is None: continue``): the event key exists (initialised
    to ``[]``) but receives no likelihood, so the combine's full-mask
    intersection excludes it — a hidden, catalogue-support-conditioned
    selection of the event sample.
    """
    np.random.seed(42)
    result = _run_evaluate(tmp_path, monkeypatch, cosmological_model, _ZERO_HOST_EVENT)

    # The dropped event: entry exists, stays empty, lands empty in the JSON too.
    assert result["1d"][_ZERO_HOST_EVENT] == []
    assert result["2d"][_ZERO_HOST_EVENT] == []
    assert result["written_1d"][str(_ZERO_HOST_EVENT)] == []
    # No diagnostics row is written for the dropped event.
    assert _ZERO_HOST_EVENT not in result["diagnostics"]

    # All other events evaluate normally: exactly one positive likelihood.
    for idx in range(_N_EVENTS):
        if idx == _ZERO_HOST_EVENT:
            continue
        assert len(result["1d"][idx]) == 1 and result["1d"][idx][0] > 0.0
        assert len(result["2d"][idx]) == 1


@pytest.mark.slow
def test_zero_host_fallback_does_not_disturb_other_events(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cosmological_model: "Model1CrossCheck",
) -> None:
    """Events WITH hosts are bit-identical whether or not another event is zero-host."""
    np.random.seed(42)
    reference = _run_evaluate(tmp_path / "ref", monkeypatch, cosmological_model, None)
    np.random.seed(42)
    with_drop = _run_evaluate(tmp_path / "drop", monkeypatch, cosmological_model, _ZERO_HOST_EVENT)

    for idx in range(_N_EVENTS):
        if idx == _ZERO_HOST_EVENT:
            continue
        assert with_drop["1d"][idx] == reference["1d"][idx]
        assert with_drop["2d"][idx] == reference["2d"][idx]
