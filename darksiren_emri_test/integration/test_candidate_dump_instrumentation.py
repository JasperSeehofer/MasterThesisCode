"""Tests for the T2.2 (row #255 tree 2 node T2.2, A10) per-candidate diagnostic
dump instrumentation on ``BayesianStatistics.evaluate``
(``candidate_dump_dir``).

Spec: results/campaign51_20260728/realistic_20260729/tree2_20260830/
B4_3_MIXTURE_WEIGHT_DERIVATION_20260830.md section 6 ("the per-candidate
instrumented run — design"). ``candidate_dump_dir`` is OPT-IN: ``None``
(default) is byte-identical to the pre-flag path — no computed value is read
or written differently (GATE BI). When set to a directory, a READ-ONLY
serialiser writes ``per_candidate_h_<label>.csv`` (one row per (event,
candidate)) and ``per_event_h_<label>.csv`` (one row per event) there, built
entirely from state ``p_Di`` already computes for a normal run.

Two gates, mirroring this repo's other opt-in-instrumentation test modules
(e.g. ``test_theta_phi_divisor.py``, the frozen-g_frac counterfactual tests):

- GATE BI: with the flag on, every existing output (posterior JSONs, the
  event-likelihoods diagnostics CSV) is byte-identical to the unhooked run.
- GATE SCHEMA: with the flag on, the two dump CSVs exist, carry the columns
  section 6.2 specifies, and have one row per event (per_event) / one row
  per (event, candidate) with a plausible count (per_candidate).

Reuses the deterministic synthetic fixture of ``test_pipeline_parity.py``
(seeded galaxy catalogue, seeded ``Model1CrossCheck``, single-h evaluate()).
CPU-only; marked ``slow`` like the other full-``evaluate()`` pipeline tests.
"""

import csv
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

from darksiren_emri_test.integration.conftest import build_galaxy_catalog_for_n_detections
from darksiren_emri_test.integration.test_pipeline_parity import _setup_sim_env

if TYPE_CHECKING:
    from darksiren_emri.cosmological_model import Model1CrossCheck

_N_EVENTS = 5
_H_VALUE = 0.73

_CANDIDATE_COLUMNS = [
    "event_idx",
    "h",
    "catalog_index",
    "batch",
    "z_g",
    "z_err_g",
    "M_g",
    "M_err_g",
    "phiS_g",
    "qS_g",
    "w_g",
    "N_g_used",
    "D_g",
    "s_bar_phi_zg",
    "s_4d_zg_mg",
    "u_g",
    "sky_mahalanobis",
    "is_true_host",
]

_EVENT_COLUMNS = [
    "event_idx",
    "h",
    "d_hat",
    "sigma_dL",
    "z_true",
    "host_galaxy_index",
    "n_cand_no_bh",
    "n_cand_with_bh",
    "f_bar_z_true",
    "f_k_z_true",
    "L_cat_no_bh",
    "B_num",
    "D_tilde_phi",
]


def _run_evaluate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cosmological_model: "Model1CrossCheck",
    candidate_dump_dir: str | None,
) -> dict[str, bytes]:
    """Run one single-h evaluate(); return raw bytes of every written output.

    Args:
        candidate_dump_dir: forwarded verbatim to ``evaluate()``.

    Returns:
        Mapping of a stable logical name to the raw file bytes, for the
        byte-identity comparison (GATE BI).
    """
    tmp_path.mkdir(parents=True, exist_ok=True)
    _setup_sim_env(tmp_path, monkeypatch)

    from darksiren_emri.bayesian_inference.bayesian_statistics import BayesianStatistics

    np.random.seed(42)
    galaxy_catalog = build_galaxy_catalog_for_n_detections(_N_EVENTS)
    bayesian_stats = BayesianStatistics()
    bayesian_stats.evaluate(
        galaxy_catalog=galaxy_catalog,
        cosmological_model=cosmological_model,
        h_value=_H_VALUE,
        candidate_dump_dir=candidate_dump_dir,
    )

    out: dict[str, bytes] = {}
    posteriors_1d = tmp_path / "simulations" / "posteriors" / "h_0_73.json"
    posteriors_2d = tmp_path / "simulations" / "posteriors_with_bh_mass" / "h_0_73.json"
    diagnostics = tmp_path / "simulations" / "diagnostics" / "event_likelihoods.csv"
    out["posteriors_1d"] = posteriors_1d.read_bytes()
    out["posteriors_2d"] = posteriors_2d.read_bytes()
    out["diagnostics"] = diagnostics.read_bytes()
    return out


@pytest.mark.slow
def test_candidate_dump_off_is_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cosmological_model: "Model1CrossCheck",
) -> None:
    """``candidate_dump_dir`` omitted writes no dump files (default None)."""
    _run_evaluate(tmp_path / "default", monkeypatch, cosmological_model, None)
    assert not (tmp_path / "default" / "candidate_dump").exists()
    # No stray per_candidate_h_*/per_event_h_* files anywhere under the run.
    assert not list((tmp_path / "default").rglob("per_candidate_h_*.csv"))
    assert not list((tmp_path / "default").rglob("per_event_h_*.csv"))


@pytest.mark.slow
def test_candidate_dump_on_is_byte_identical_to_off(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cosmological_model: "Model1CrossCheck",
) -> None:
    """GATE BI: arming the dump changes NO computed value.

    Every existing output (both posterior JSONs, the diagnostics CSV) is
    byte-for-byte identical whether or not ``candidate_dump_dir`` is set —
    the hook is placed strictly after ``p_Di`` returns and only reads
    already-computed state.
    """
    off = _run_evaluate(tmp_path / "off", monkeypatch, cosmological_model, None)
    dump_dir = tmp_path / "on" / "candidate_dump"
    on = _run_evaluate(tmp_path / "on", monkeypatch, cosmological_model, str(dump_dir))

    assert off["posteriors_1d"] == on["posteriors_1d"]
    assert off["posteriors_2d"] == on["posteriors_2d"]
    assert off["diagnostics"] == on["diagnostics"]

    # And the dump itself was actually produced (the comparison above isn't
    # vacuously true because the flag silently did nothing).
    assert list(dump_dir.glob("per_candidate_h_0_73.csv"))
    assert list(dump_dir.glob("per_event_h_0_73.csv"))


@pytest.mark.slow
def test_candidate_dump_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cosmological_model: "Model1CrossCheck",
) -> None:
    """GATE SCHEMA: the two dump CSVs carry section 6.2's exact columns.

    Also checks basic row-count sanity (one event row per active detection;
    at least one candidate row overall) and that ``is_true_host`` parses as a
    boolean and ``event_idx``/``h`` are consistent between the two files.
    """
    dump_dir = tmp_path / "dump"
    _run_evaluate(tmp_path, monkeypatch, cosmological_model, str(dump_dir))

    cand_path = dump_dir / "per_candidate_h_0_73.csv"
    event_path = dump_dir / "per_event_h_0_73.csv"
    assert cand_path.is_file()
    assert event_path.is_file()

    with open(cand_path, newline="") as f:
        cand_rows = list(csv.DictReader(f))
    with open(event_path, newline="") as f:
        event_rows = list(csv.DictReader(f))

    assert cand_rows, "expected at least one per-candidate row"
    assert event_rows, "expected at least one per-event row"
    assert set(cand_rows[0].keys()) == set(_CANDIDATE_COLUMNS)
    assert set(event_rows[0].keys()) == set(_EVENT_COLUMNS)

    # One event row per event that reached p_Di (<= _N_EVENTS; some may be
    # excluded by the SNR/Fisher-quality filters).
    event_indices = {int(row["event_idx"]) for row in event_rows}
    assert 0 < len(event_indices) <= _N_EVENTS
    assert len(event_rows) == len(event_indices), "one row per event, not per candidate"

    cand_event_indices = {int(row["event_idx"]) for row in cand_rows}
    assert cand_event_indices <= event_indices

    for row in cand_rows:
        assert row["batch"] in ("with_bh", "no_bh_only")
        assert row["is_true_host"] in ("True", "False")
        assert float(row["h"]) == pytest.approx(_H_VALUE)
        # z_g / N_g_used / D_g must be finite (real candidates, not placeholders).
        assert np.isfinite(float(row["z_g"]))
        assert np.isfinite(float(row["N_g_used"]))
        assert np.isfinite(float(row["D_g"]))

    for row in event_rows:
        assert float(row["h"]) == pytest.approx(_H_VALUE)
        assert int(row["n_cand_no_bh"]) >= 0
        assert int(row["n_cand_with_bh"]) >= 0
