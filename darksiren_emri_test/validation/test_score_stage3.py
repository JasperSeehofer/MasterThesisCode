"""Tests for the stage-3 readout scorer (results/mechanism_study_20260813/score_stage3.py).

Written and committed BEFORE the A-JREN/A-REN arm data exists (PREREGISTRATION_A_JREN_STAGE3.md,
thread 17 analysis-code-freeze discipline, mirroring test_score_m2prime_stage2.py's own discipline).

The scorer module lives outside any Python package (``results/`` is data/analysis output, not
source), so it is loaded here via ``importlib`` from its file path.
"""

import importlib.util
import json
import math
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCORER_PATH = REPO_ROOT / "results" / "mechanism_study_20260813" / "score_stage3.py"
MN0X_PATH = (
    REPO_ROOT / "results" / "mechanism_study_20260813" / "MN0X_h0p730_results_seeds0_100.json"
)


def _load_scorer_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("score_stage3", SCORER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def scorer() -> ModuleType:
    if not SCORER_PATH.exists():
        pytest.skip(f"scorer not found at {SCORER_PATH}")
    return _load_scorer_module()


@pytest.fixture(scope="module")
def mn0x_data(scorer: ModuleType) -> dict[str, Any]:
    if not MN0X_PATH.exists():
        pytest.skip(f"committed reference not found at {MN0X_PATH}")
    data: dict[str, Any] = scorer.load(MN0X_PATH)
    return data


def test_mn0x_committed_file_present() -> None:
    """The paired reference must be a committed artifact, not a fresh run."""
    assert MN0X_PATH.exists(), "MN0X_h0p730_results_seeds0_100.json must be committed"


def test_scorer_recomputes_b_ref_from_mn0x(scorer: ModuleType, mn0x_data: dict[str, Any]) -> None:
    """DS-M1's b_ref pin (+0.037250) must reproduce exactly from the raw ln_post vectors,
    identically to the stage-2 scorer (same B_REF, same recomputation kernel)."""
    ds_m1 = scorer.score_ds_m1(mn0x_data)
    assert ds_m1["1d"]["n"] == 100
    assert math.isclose(ds_m1["1d"]["mean_bias"], scorer.B_REF, abs_tol=1e-9)


def test_mn0x_classifies_term_innocent_trivially(
    scorer: ModuleType, mn0x_data: dict[str, Any]
) -> None:
    """Sanity gate (task-required): treating MN0X itself as "the arm" must classify
    TERM-INNOCENT trivially — its bias IS b_ref by construction, so
    |bias| >= DEFECT and |bias - b_ref| == 0 <= NULL_TOL hold by definition, in
    BOTH channels (no split), giving branch 4 for the "aren"-style routing (the
    class-to-branch map applies identically regardless of which arm produced
    the classification)."""
    ds_m1 = scorer.score_ds_m1(mn0x_data)
    assert ds_m1["1d"]["class"] == "TERM-INNOCENT"
    assert ds_m1["2d"]["class"] == "TERM-INNOCENT"

    branch = scorer.determine_branch("aren", True, ds_m1, None, True)
    assert branch["status"] == "PRESENTED, NOT ADJUDICATED"
    assert branch["branch"] == "4. REN-INNOCENT"


def test_ds_j1_coverage_restoration_pass(scorer: ModuleType) -> None:
    """DS-J1: both channels' HPD90 >= 0.60 -> coverage_restored True/True/True."""
    ds_m1 = {
        "1d": {"hpd90_cov": 0.64},
        "2d": {"hpd90_cov": 0.72},
    }
    result = scorer.score_ds_j1(ds_m1)
    assert result["1d"]["coverage_restored"] is True
    assert result["2d"]["coverage_restored"] is True
    assert result["coverage_restored_both_channels"] is True


def test_ds_j1_coverage_restoration_one_channel_short(scorer: ModuleType) -> None:
    """DS-J1: one channel below 0.60 fails the COMBINED restoration flag."""
    ds_m1 = {
        "1d": {"hpd90_cov": 0.61},
        "2d": {"hpd90_cov": 0.55},
    }
    result = scorer.score_ds_j1(ds_m1)
    assert result["1d"]["coverage_restored"] is True
    assert result["2d"]["coverage_restored"] is False
    assert result["coverage_restored_both_channels"] is False


def test_ds_j1_missing_channel_is_not_restored(scorer: ModuleType) -> None:
    """DS-J1: a NO-FINITE-SEEDS channel (no hpd90_cov key) must not silently pass."""
    ds_m1 = {"1d": {"hpd90_cov": 0.90}, "2d": {}}
    result = scorer.score_ds_j1(ds_m1)
    assert result["2d"]["coverage_restored"] is False
    assert result["coverage_restored_both_channels"] is False


@pytest.mark.parametrize(
    ("which", "bias", "expected_read"),
    [
        ("aren", 0.0354, "INSIDE"),  # exactly the registered center
        ("aren", 0.0354 - 0.006, "INSIDE"),  # exactly at the lower edge
        ("aren", 0.0354 - 0.006 - 1e-6, "BELOW"),  # just under the lower edge
        ("aren", 0.0354 + 0.006 + 1e-6, "ABOVE"),  # just over the upper edge
        ("ajren", 0.0173, "INSIDE"),
        ("ajren", 0.0173 - 0.012 - 1e-6, "BELOW"),
        ("ajren", 0.0173 + 0.012 + 1e-6, "ABOVE"),
    ],
)
def test_f2_window_reads(scorer: ModuleType, which: str, bias: float, expected_read: str) -> None:
    ds_m1 = {"1d": {"mean_bias": bias}}
    result = scorer.score_f2_window(which, ds_m1)
    assert result["read"] == expected_read
    assert result["status"] == "WEAK, non-branch-carrying"


def test_f2_window_reports_no_finite_seeds_when_bias_missing(scorer: ModuleType) -> None:
    ds_m1: dict[str, Any] = {"1d": {}}
    result = scorer.score_f2_window("ajren", ds_m1)
    assert result["read"] == "NO FINITE SEEDS"
    assert result["measured_1d_bias"] is None


def test_f2_windows_match_registration_finalization_block(scorer: ModuleType) -> None:
    """The registered F2 numbers (2026-08-15 finalization block): A-REN
    b ~ +0.0354 +/- 0.006; A-JREN b ~ +0.0173 +/- 0.012."""
    assert scorer.F2_WINDOWS["aren"] == {"center": 0.0354, "half_width": 0.006}
    assert scorer.F2_WINDOWS["ajren"] == {"center": 0.0173, "half_width": 0.012}


def test_classify_edges(scorer: ModuleType) -> None:
    b_ref = scorer.B_REF
    assert scorer.classify(0.005, 0.70, b_ref) == "TERM-OWNS"
    assert scorer.classify(0.005, 0.50, b_ref) != "TERM-OWNS"  # coverage conjunct required
    assert scorer.classify(0.020, 0.70, b_ref) == "TERM-PARTIAL"
    assert scorer.classify(b_ref, 0.0, b_ref) == "TERM-INNOCENT"
    assert scorer.classify(0.5, 0.0, b_ref) == "OTHER"


def test_branch_withheld_when_arm_missing(scorer: ModuleType) -> None:
    """Execution-completeness clause (§5, carried): no branch is adjudicated with the arm missing."""
    branch = scorer.determine_branch("aren", False, None, None, True)
    assert branch["status"] == "NOT PRESENTED"
    assert "execution-completeness" in branch["reason"]


def test_branch_study_confounded_on_invalid_mn0x_cross_check(scorer: ModuleType) -> None:
    ds_m1 = {"1d": {"class": "TERM-OWNS"}, "2d": {"class": "TERM-OWNS"}}
    branch = scorer.determine_branch("aren", True, ds_m1, None, False)
    assert branch["status"] == "PRESENTED, NOT ADJUDICATED"
    assert branch["branch"] == "1. STUDY-CONFOUNDED"


def test_branch_split_precedence(scorer: ModuleType) -> None:
    """Any 1D/2D class split routes to branch 5, taking precedence over branches 2-4."""
    ds_m1 = {"1d": {"class": "TERM-OWNS"}, "2d": {"class": "TERM-PARTIAL"}}
    branch = scorer.determine_branch("aren", True, ds_m1, None, True)
    assert branch["status"] == "PRESENTED, NOT ADJUDICATED"
    assert branch["branch"] == "5. OTHER / SPLIT"


@pytest.mark.parametrize(
    ("cls_both", "expected_branch"),
    [
        ("TERM-OWNS", "2. REN-OWNS"),
        ("TERM-PARTIAL", "3. REN-PARTIAL"),
        ("TERM-INNOCENT", "4. REN-INNOCENT"),
        ("OTHER", "5. OTHER / SPLIT"),
    ],
)
def test_branch_class_routing_aren(scorer: ModuleType, cls_both: str, expected_branch: str) -> None:
    ds_m1 = {"1d": {"class": cls_both}, "2d": {"class": cls_both}}
    branch = scorer.determine_branch("aren", True, ds_m1, None, True)
    assert branch["branch"] == expected_branch


def test_branch_ajren_is_diagnostic_not_a_point_prediction_branch(scorer: ModuleType) -> None:
    """A-JREN has no registered point-prediction branch (prereg §4/§7): its
    read is diagnostic, reporting the DS-M1 class alongside the DS-J1
    coverage-restoration flag rather than routing through the REN-* map."""
    ds_m1 = {"1d": {"class": "TERM-PARTIAL"}, "2d": {"class": "TERM-PARTIAL"}}
    ds_j1 = {"coverage_restored_both_channels": True}
    branch = scorer.determine_branch("ajren", True, ds_m1, ds_j1, True)
    assert branch["status"] == "PRESENTED, NOT ADJUDICATED"
    assert "diagnostic" in branch["branch"]
    assert "TERM-PARTIAL" in branch["fired_by"]
    assert "True" in branch["fired_by"]


def test_main_runs_cleanly_with_arm_missing(scorer: ModuleType, tmp_path: Path) -> None:
    """The script must fail cleanly (exit 0, branch withheld) when the arm file doesn't exist yet."""
    out_path = tmp_path / "score_stage3_output.json"
    rc = scorer.main(
        [
            "--arm",
            str(tmp_path / "does_not_exist_AJREN.json"),
            "--which",
            "ajren",
            "--mn0x",
            str(MN0X_PATH),
            "--out",
            str(out_path),
        ]
    )
    assert rc == 0
    assert out_path.exists()
    payload = json.loads(out_path.read_text())
    assert payload["branch"]["status"] == "NOT PRESENTED"
    assert payload["inputs"]["arm_present"] is False


def test_main_runs_cleanly_treating_mn0x_as_the_aren_arm(
    scorer: ModuleType, tmp_path: Path
) -> None:
    """End-to-end smoke test: point --arm at the committed MN0X file itself (as if
    it were an AREN result) and confirm the full pipeline (DS-M1, F2 window,
    branch) runs and reproduces the trivial TERM-INNOCENT / branch 4 result."""
    out_path = tmp_path / "score_stage3_output.json"
    rc = scorer.main(
        [
            "--arm",
            str(MN0X_PATH),
            "--which",
            "aren",
            "--mn0x",
            str(MN0X_PATH),
            "--out",
            str(out_path),
        ]
    )
    assert rc == 0
    payload = json.loads(out_path.read_text())
    assert payload["ds_m1"]["1d"]["class"] == "TERM-INNOCENT"
    assert payload["branch"]["branch"] == "4. REN-INNOCENT"
    assert payload["ds_j1"] is None  # only computed for --which ajren
