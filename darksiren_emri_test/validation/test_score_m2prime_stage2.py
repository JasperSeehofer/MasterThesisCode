"""Tests for the stage-2 readout scorer (results/mechanism_study_20260813/score_m2prime_stage2.py).

Written and committed BEFORE the A-M2'/A-NULL arm data exists (PREREGISTRATION_M2PRIME_ABLATION.md,
thread 17 analysis-code-freeze discipline). Exercises the DS-N1 floor-aware integer shift law
machinery against the real committed ``MN0X_h0p730_results_seeds0_100.json`` reference in the two
registered unit-test scenarios:

1. ANULL := MN0X itself (scale factor 1.0): expect ``m(h) = 0`` everywhere and every per-seed
   1D/2D MAP grid index equal.
2. ANULL := MN0X with ``N * ln(1.7)`` added to every stored ``ln_post`` vector (the exact effect
   of A-NULL's registered ``x1.7`` estimator switch, reconstructed from the ``ln_post`` vectors
   alone since the per-event contributions are not otherwise recoverable): expect ``m(h) = 982``
   everywhere and MAP indices still equal (a uniform multiplicative shift never moves the argmax).

The scorer module lives outside any Python package (``results/`` is data/analysis output, not
source), so it is loaded here via ``importlib`` from its file path.
"""

import copy
import importlib.util
import json
import math
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCORER_PATH = REPO_ROOT / "results" / "mechanism_study_20260813" / "score_m2prime_stage2.py"
MN0X_PATH = (
    REPO_ROOT / "results" / "mechanism_study_20260813" / "MN0X_h0p730_results_seeds0_100.json"
)


def _load_scorer_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("score_m2prime_stage2", SCORER_PATH)
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
    """DS-M1's b_ref pin (+0.037250) must reproduce exactly from the raw ln_post vectors."""
    ds_m1 = scorer.score_ds_m1(mn0x_data)
    assert ds_m1["1d"]["n"] == 100
    assert math.isclose(ds_m1["1d"]["mean_bias"], scorer.B_REF, abs_tol=1e-9)


def test_ds_n1_anull_equals_mn0x_scale_1p0(scorer: ModuleType, mn0x_data: dict[str, Any]) -> None:
    """Scenario 1 (registered): ANULL := MN0X itself, scale factor 1.0.

    Expect m(h) = 0 at every grid point (in both channels, at every seed's MAP index) and every
    per-seed 1D/2D MAP grid index exactly equal — the DS-N1 machinery must recognize a null shift
    as a null shift.
    """
    first_15 = mn0x_data["per_seed"][:15]
    anull_like = {"config": mn0x_data["config"], "per_seed": first_15}

    result = scorer.score_ds_n1(anull_like, mn0x_data)

    assert result["status"] == "PASS"
    assert result["grid_match"] is True
    assert result["n_paired"] == 15
    assert result["all_index_eq"] is True
    assert result["all_law_ok"] is True

    for row in result["rows"]:
        assert row["status"] == "OK"
        for ch in ("1d", "2d"):
            assert row[f"{ch}_index_eq"] is True
            assert row[f"{ch}_m_at_anull_map_index"] == 0
            assert row[f"{ch}_m_at_mn0x_map_index"] == 0
            assert row[f"{ch}_m_min"] == 0
            assert row[f"{ch}_m_max"] == 0
            assert row[f"{ch}_law_ok"] is True
            assert row[f"{ch}_max_law_resid"] <= 1e-6


def test_ds_n1_anull_synthetic_scale_1p7(scorer: ModuleType, mn0x_data: dict[str, Any]) -> None:
    """Scenario 2 (registered): ANULL := MN0X with N*ln(1.7) added to every stored ln_post vector.

    Per-event contributions are not reconstructable from the stored ln_post vectors alone, so this
    reconstructs the exact effect the A-NULL ``x1.7`` estimator switch has on the *aggregate*
    ln-posterior: at every h, ln_post -> ln_post + N*ln(1.7). Expect m(h) = 982 (= N_EVENTS_PIN)
    at every grid point in both channels, and MAP indices still exactly equal (a uniform additive
    shift in ln-posterior space never moves the argmax).
    """
    shift = scorer.N_EVENTS_PIN * scorer.LN_1P7
    shifted_records = copy.deepcopy(mn0x_data["per_seed"][:15])
    for rec in shifted_records:
        rec["ln_post_1d"] = [v + shift for v in rec["ln_post_1d"]]
        rec["ln_post_2d"] = [v + shift for v in rec["ln_post_2d"]]
    anull_shifted = {"config": mn0x_data["config"], "per_seed": shifted_records}

    result = scorer.score_ds_n1(anull_shifted, mn0x_data)

    assert result["status"] == "PASS"
    assert result["all_index_eq"] is True
    assert result["all_law_ok"] is True

    for row in result["rows"]:
        assert row["status"] == "OK"
        for ch in ("1d", "2d"):
            assert row[f"{ch}_index_eq"] is True
            assert row[f"{ch}_m_at_anull_map_index"] == scorer.N_EVENTS_PIN
            assert row[f"{ch}_m_at_mn0x_map_index"] == scorer.N_EVENTS_PIN
            assert row[f"{ch}_m_min"] == scorer.N_EVENTS_PIN
            assert row[f"{ch}_m_max"] == scorer.N_EVENTS_PIN
            assert row[f"{ch}_law_ok"] is True
            assert row[f"{ch}_max_law_resid"] <= 1e-6


def test_ds_n1_flags_broken_shift_law(scorer: ModuleType, mn0x_data: dict[str, Any]) -> None:
    """A non-integer-multiple-of-ln(1.7) shift must FAIL the law, not silently pass."""
    shifted_records = copy.deepcopy(mn0x_data["per_seed"][:15])
    for rec in shifted_records:
        rec["ln_post_1d"] = [v + 1.2345 for v in rec["ln_post_1d"]]
        rec["ln_post_2d"] = [v + 1.2345 for v in rec["ln_post_2d"]]
    anull_broken = {"config": mn0x_data["config"], "per_seed": shifted_records}

    result = scorer.score_ds_n1(anull_broken, mn0x_data)

    assert result["status"] == "FAIL"
    assert result["all_law_ok"] is False


def test_ds_n1_flags_missing_seeds(scorer: ModuleType, mn0x_data: dict[str, Any]) -> None:
    """Fewer than the registered 15 paired seeds must not silently PASS."""
    anull_short = {"config": mn0x_data["config"], "per_seed": mn0x_data["per_seed"][:10]}

    result = scorer.score_ds_n1(anull_short, mn0x_data)

    assert result["status"] == "FAIL"
    assert result["n_paired"] == 10
    assert result["all_present"] is False


def test_branch_withheld_when_arms_missing(scorer: ModuleType) -> None:
    """Execution-completeness clause (§5): no branch is adjudicated with arms missing."""
    branch = scorer.determine_branch(False, False, None, None, True)
    assert branch["status"] == "NOT PRESENTED"
    assert "execution-completeness" in branch["reason"]

    branch_one_missing = scorer.determine_branch(True, False, {"1d": {}, "2d": {}}, None, True)
    assert branch_one_missing["status"] == "NOT PRESENTED"


def test_branch_study_confounded_on_ds_n1_fail(scorer: ModuleType) -> None:
    ds_m1_am2p = {
        "1d": {"class": "TERM-OWNS"},
        "2d": {"class": "TERM-OWNS"},
    }
    ds_n1_fail = {"status": "FAIL"}
    branch = scorer.determine_branch(True, True, ds_m1_am2p, ds_n1_fail, True)
    assert branch["status"] == "PRESENTED, NOT ADJUDICATED"
    assert branch["branch"] == "1. STUDY-CONFOUNDED"


def test_branch_split_precedence(scorer: ModuleType) -> None:
    """Any 1D/2D class split routes to branch 5, taking precedence over branches 2-4."""
    ds_m1_am2p = {
        "1d": {"class": "TERM-OWNS"},
        "2d": {"class": "TERM-PARTIAL"},
    }
    ds_n1_pass = {"status": "PASS"}
    branch = scorer.determine_branch(True, True, ds_m1_am2p, ds_n1_pass, True)
    assert branch["status"] == "PRESENTED, NOT ADJUDICATED"
    assert branch["branch"] == "5. OTHER / SPLIT"


@pytest.mark.parametrize(
    ("cls_both", "expected_branch"),
    [
        ("TERM-OWNS", "2. M2'-OWNS"),
        ("TERM-PARTIAL", "3. M2'-PARTIAL"),
        ("TERM-INNOCENT", "4. M2'-INNOCENT"),
        ("OTHER", "5. OTHER / SPLIT"),
    ],
)
def test_branch_class_routing(scorer: ModuleType, cls_both: str, expected_branch: str) -> None:
    ds_m1_am2p = {"1d": {"class": cls_both}, "2d": {"class": cls_both}}
    ds_n1_pass = {"status": "PASS"}
    branch = scorer.determine_branch(True, True, ds_m1_am2p, ds_n1_pass, True)
    assert branch["branch"] == expected_branch


def test_classify_edges(scorer: ModuleType) -> None:
    b_ref = scorer.B_REF
    assert scorer.classify(0.005, 0.70, b_ref) == "TERM-OWNS"
    assert scorer.classify(0.005, 0.50, b_ref) != "TERM-OWNS"  # coverage conjunct required
    assert scorer.classify(0.020, 0.70, b_ref) == "TERM-PARTIAL"
    assert scorer.classify(b_ref, 0.0, b_ref) == "TERM-INNOCENT"
    assert scorer.classify(0.5, 0.0, b_ref) == "OTHER"


def test_main_runs_cleanly_with_arms_missing(scorer: ModuleType, tmp_path: Path) -> None:
    """The script must fail cleanly (exit 0, branch withheld) when the arm files don't exist yet."""
    out_path = tmp_path / "score_m2prime_stage2_output.json"
    rc = scorer.main(
        [
            "--am2p",
            str(tmp_path / "does_not_exist_AM2P.json"),
            "--anull",
            str(tmp_path / "does_not_exist_ANULL.json"),
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
    assert payload["inputs"]["am2p_present"] is False
    assert payload["inputs"]["anull_present"] is False
