"""Regression tests for the row #288 S4 harness repair (node b-s4-harness-repair, ledger row
#290 decisions-table row 3, Branch A).

Covers the three S4 defects repaired in ``b8_cal_harness.py``:

  (a) seed-population separation -- ``score_only`` must refuse to pool checkpoints whose
      ``n_draw_requested`` (population tag) differs, and must cleanly aggregate a single
      declared population when one is passed explicitly.
  (b) missing cell-T aggregation -- ``score_only(cell="T")`` must produce a clean aggregate
      (it is not special-cased away), and ``score_ratio_t_over_s`` must compute the T/S SD
      ratio the design note (line 233) registers as the S4 input.
  (c) wall-limited stop rule -- ``score_only`` must report, per cell, whether the driver's last
      invocation was wall-limited or completion-limited, via the ``run_status`` sidecar; and
      must say so explicitly (not silently) when no sidecar exists yet.

CPU-only, no GPU, no cluster, no real generative context (fake minimal checkpoint JSONs) -- fast.

Runnable directly with ``uv run pytest`` (this file is picked up as a plain ``test_*.py`` module
by pytest's rootdir-relative discovery; it lives next to ``b8_cal_harness.py`` on purpose so the
``importlib`` load below resolves ``THIS_DIR``-relative paths, e.g. ``b8_information_floor.json``,
the same way the harness resolves them when invoked directly).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

_THIS_DIR = Path(__file__).resolve().parent
_HARNESS_PATH = _THIS_DIR / "b8_cal_harness.py"


def _load_harness() -> Any:
    """Load b8_cal_harness.py by path (it is not an importable package module)."""
    spec = importlib.util.spec_from_file_location("b8_cal_harness_under_test", _HARNESS_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["b8_cal_harness_under_test"] = mod
    spec.loader.exec_module(mod)
    return mod


h = _load_harness()


def _fake_checkpoint(
    seed: int, cell: str, n_draw_requested: int, sd: float = 0.05, pit: float = 0.5
) -> dict[str, Any]:
    """A minimal checkpoint dict carrying every field score_only() reads."""
    n_bins = len(h.B3_1_BIN_EDGES) - 1
    channel_post = {
        "sd": sd,
        "pit": pit,
        "map_h": 0.73,
        "n_events_scored": 50,
        "hpd50": True,
        "hpd68": True,
        "hpd90": True,
        "hpd95": True,
    }
    channel_score = {"available": False}
    return {
        "universe": {
            "seed": seed,
            "cell": cell,
            "n_draw_requested": n_draw_requested,
            "n_realized_draw": n_draw_requested,
        },
        "posterior": {"no_bh": dict(channel_post), "with_bh": dict(channel_post)},
        "score_at_truth": {"no_bh": dict(channel_score), "with_bh": dict(channel_score)},
        "z_true_hist": {
            "bin_edges": list(h.B3_1_BIN_EDGES),
            "counts": [0] * n_bins,
            "counts_catalogue_hosted": [0] * n_bins,
            "counts_dark": [0] * n_bins,
        },
        "n_pred_by_bin": {
            "n_pred_shape": [1.0 / n_bins] * n_bins,
            "self_check": {"ok": True},
        },
    }


def _write_checkpoints(
    work_root: Path, cell: str, seeds_and_populations: list[tuple[int, int]]
) -> None:
    for seed, pop in seeds_and_populations:
        ckpt = _fake_checkpoint(seed, cell, pop)
        h.checkpoint_path(work_root, cell, seed).write_text(json.dumps(ckpt))


# ── (a) seed-population separation ────────────────────────────────────────────


def test_score_only_refuses_mixed_population(tmp_path: Path) -> None:
    """0 mixed rows: a cell spanning >1 population must be REFUSED, not silently pooled."""
    _write_checkpoints(tmp_path, "S", [(1, 200), (2, 200), (3, 100)])
    with pytest.raises(h.PopulationMixError):
        h.score_only(tmp_path, "S")


def test_score_only_explicit_population_excludes_other_rows(tmp_path: Path) -> None:
    _write_checkpoints(tmp_path, "S", [(1, 200), (2, 200), (3, 100)])
    out = h.score_only(tmp_path, "S", population=200)
    assert out["n_universes"] == 2
    assert out["population"] == 200
    assert len(out["excluded_other_population"]) == 1
    assert out["excluded_other_population"][0]["n_draw_requested"] == 100
    assert out["populations_present_before_filter"] == [100, 200]


def test_score_only_single_population_needs_no_explicit_arg(tmp_path: Path) -> None:
    """The common (unmixed) case is unaffected -- g-byte-id on the untouched path."""
    _write_checkpoints(tmp_path, "S", [(1, 200), (2, 200)])
    out = h.score_only(tmp_path, "S")
    assert out["n_universes"] == 2
    assert out["population"] == 200
    assert out["excluded_other_population"] == []


# ── (b) cell-T aggregation + T/S ratio ────────────────────────────────────────


def test_score_only_aggregates_cell_t(tmp_path: Path) -> None:
    """Cell T is not special-cased away -- score_only(cell='T') must produce a clean aggregate."""
    _write_checkpoints(tmp_path, "T", [(1, 200), (2, 200)])
    out = h.score_only(tmp_path, "T")
    assert out["n_universes"] == 2
    assert out["cell"] == "T"
    assert out["no_bh"]["sigma_h_harness_median_sd"] == pytest.approx(0.05)


def test_score_ratio_t_over_s(tmp_path: Path) -> None:
    _write_checkpoints(tmp_path, "S", [(1, 200), (2, 200)])
    _write_checkpoints(tmp_path, "T", [(11, 200)])
    # give T a different SD so the ratio is not trivially 1
    t_file = h.checkpoint_path(tmp_path, "T", 11)
    ckpt = json.loads(t_file.read_text())
    ckpt["posterior"]["no_bh"]["sd"] = 0.10
    ckpt["posterior"]["with_bh"]["sd"] = 0.10
    t_file.write_text(json.dumps(ckpt))

    out = h.score_ratio_t_over_s(tmp_path, population=200)
    assert out["cell_s"]["n_universes"] == 2
    assert out["cell_t"]["n_universes"] == 1
    assert out["ratio"]["no_bh"]["T_over_S"] == pytest.approx(0.10 / 0.05)


def test_score_ratio_t_over_s_missing_cell_reports_reason(tmp_path: Path) -> None:
    _write_checkpoints(tmp_path, "S", [(1, 200)])
    out = h.score_ratio_t_over_s(tmp_path, population=200)
    assert out["cell_t"]["n_universes"] == 0
    assert "reason" in out["ratio"]


# ── (c) wall-limited stop rule ────────────────────────────────────────────────


def test_score_only_reports_run_status_when_present(tmp_path: Path) -> None:
    _write_checkpoints(tmp_path, "S", [(1, 200)])
    h.run_status_path(tmp_path, "S").write_text(
        json.dumps(
            {
                "cell": "S",
                "stopped_reason": "wall_limited",
                "n_universes_requested_this_invocation": 100,
                "n_done_this_invocation": 1,
                "n_checkpoints_total_under_work_root": 1,
                "max_wall_s": 43200.0,
                "wall_elapsed_s_this_invocation": 43712.5,
            }
        )
    )
    out = h.score_only(tmp_path, "S", population=200)
    assert out["run_status"]["available"] is True
    assert out["run_status"]["wall_limited"] is True
    assert out["run_status"]["stopped_reason"] == "wall_limited"
    assert out["run_status"]["n_done_this_invocation"] == 1


def test_score_only_reports_run_status_absent_explicitly(tmp_path: Path) -> None:
    """No silent gap: absence of the sidecar must be reported, not omitted."""
    _write_checkpoints(tmp_path, "S", [(1, 200)])
    out = h.score_only(tmp_path, "S", population=200)
    assert out["run_status"]["available"] is False
    assert "reason" in out["run_status"]


def test_run_status_path_completion_limited(tmp_path: Path) -> None:
    _write_checkpoints(tmp_path, "S", [(1, 200)])
    h.run_status_path(tmp_path, "S").write_text(
        json.dumps(
            {
                "cell": "S",
                "stopped_reason": "exhausted_n_universes",
                "n_universes_requested_this_invocation": 1,
                "n_done_this_invocation": 1,
                "n_checkpoints_total_under_work_root": 1,
                "max_wall_s": 43200.0,
                "wall_elapsed_s_this_invocation": 12.3,
            }
        )
    )
    out = h.score_only(tmp_path, "S", population=200)
    assert out["run_status"]["wall_limited"] is False
    assert out["run_status"]["stopped_reason"] == "exhausted_n_universes"
