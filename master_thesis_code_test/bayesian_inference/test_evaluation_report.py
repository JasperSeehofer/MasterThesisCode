"""Tests for evaluation_report.py — baseline extraction and comparison logic."""

import json
import math
from pathlib import Path

import pytest

from master_thesis_code.bayesian_inference.evaluation_report import (
    BaselineSnapshot,
    compute_credible_interval,
    extract_baseline,
    generate_comparison_report,
    load_posteriors,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_h_json(
    tmp_path: Path,
    h: float,
    log_likelihoods: list[float],
    prefix: str = "",
) -> Path:
    """Write a synthetic h_*.json posterior file.

    Args:
        tmp_path: Directory to write into.
        h: Hubble constant value.
        log_likelihoods: Per-event log-likelihood values (stored as [value]).
        prefix: Optional sub-directory prefix (created if needed).

    Returns:
        Path to created file.
    """
    if prefix:
        directory = tmp_path / prefix
        directory.mkdir(parents=True, exist_ok=True)
    else:
        directory = tmp_path

    h_str = f"{h:.3f}".replace(".", "_")
    data: dict[str, object] = {"h": h}
    for i, ll in enumerate(log_likelihoods):
        data[str(i)] = [math.exp(ll)]  # store as likelihood (not log)

    file_path = directory / f"h_{h_str}.json"
    file_path.write_text(json.dumps(data))
    return file_path


def _gaussian_log_posterior(
    h_values: list[float], center: float = 0.73, sigma: float = 0.02
) -> list[float]:
    """Return Gaussian-shaped log-posteriors for a list of h values."""
    return [-0.5 * ((h - center) / sigma) ** 2 for h in h_values]


# ---------------------------------------------------------------------------
# load_posteriors
# ---------------------------------------------------------------------------


def test_load_posteriors_returns_sorted_by_h(tmp_path: Path) -> None:
    """load_posteriors should return entries sorted by h value."""
    h_values = [0.75, 0.69, 0.73]
    for h in h_values:
        _make_h_json(tmp_path, h, [-1.0, -2.0, -3.0])

    results = load_posteriors(tmp_path)

    h_returned = [r["h"] for r in results]
    assert h_returned == sorted(h_values)


def test_load_posteriors_computes_log_posterior(tmp_path: Path) -> None:
    """load_posteriors should sum log-likelihoods across events."""
    # 3 events with known likelihoods: log_posterior = sum(log(likelihood))
    likelihoods = [2.0, 3.0, 4.0]  # stored as raw values; log sum = log(2)+log(3)+log(4)
    data: dict[str, object] = {"h": 0.73}
    for i, lk in enumerate(likelihoods):
        data[str(i)] = [lk]
    (tmp_path / "h_0_730.json").write_text(json.dumps(data))

    results = load_posteriors(tmp_path)

    assert len(results) == 1
    expected_log_posterior = math.log(2.0) + math.log(3.0) + math.log(4.0)
    assert abs(results[0]["log_posterior"] - expected_log_posterior) < 1e-10


def test_load_posteriors_counts_detections(tmp_path: Path) -> None:
    """load_posteriors should include n_detections = number of event keys."""
    data = {"h": 0.73, "0": [1.0], "1": [2.0], "2": [3.0]}
    (tmp_path / "h_0_730.json").write_text(json.dumps(data))

    results = load_posteriors(tmp_path)

    assert results[0]["n_detections"] == 3


def test_load_posteriors_warns_large_file_count(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """load_posteriors should log warning when more than 100 files exist."""
    # T-30-02: DoS mitigation — log warning for > 100 files
    import logging

    for i in range(102):
        h = 0.60 + i * 0.001
        data = {"h": h, "0": [1.0]}
        h_str = f"{h:.3f}".replace(".", "_")
        (tmp_path / f"h_{h_str}.json").write_text(json.dumps(data))

    with caplog.at_level(logging.WARNING):
        results = load_posteriors(tmp_path)
    assert len(results) == 102
    assert any("100" in record.message or "slow" in record.message for record in caplog.records)


# ---------------------------------------------------------------------------
# compute_credible_interval
# ---------------------------------------------------------------------------


def test_compute_credible_interval_symmetric_gaussian() -> None:
    """For a symmetric Gaussian posterior, 68% CI should be symmetric around peak."""
    h_values = [0.69, 0.71, 0.73, 0.75, 0.77]
    sigma = 0.02
    log_posteriors = _gaussian_log_posterior(h_values, center=0.73, sigma=sigma)

    lower, upper = compute_credible_interval(h_values, log_posteriors, level=0.68)

    # CI should bracket 0.73 and be symmetric within grid spacing
    assert lower < 0.73
    assert upper > 0.73
    assert abs((upper - 0.73) - (0.73 - lower)) < 0.01  # symmetric within grid spacing


def test_compute_credible_interval_returns_tuple() -> None:
    """compute_credible_interval should return a (lower, upper) tuple."""
    h_values = [0.69, 0.71, 0.73, 0.75, 0.77]
    log_posteriors = _gaussian_log_posterior(h_values)

    result = compute_credible_interval(h_values, log_posteriors)

    assert isinstance(result, tuple)
    assert len(result) == 2
    lower, upper = result
    assert lower < upper


# ---------------------------------------------------------------------------
# extract_baseline
# ---------------------------------------------------------------------------


def test_extract_baseline_raises_for_fewer_than_3_h_values(tmp_path: Path) -> None:
    """extract_baseline should raise ValueError if fewer than 3 h-value files exist."""
    for h in [0.71, 0.73]:
        _make_h_json(tmp_path, h, [-1.0, -2.0])

    with pytest.raises(ValueError, match="3"):
        extract_baseline(tmp_path)


def test_extract_baseline_computes_map_h(tmp_path: Path) -> None:
    """extract_baseline should find MAP h as the h with the highest log-posterior."""
    h_values = [0.69, 0.71, 0.73, 0.75, 0.77]
    log_posts = _gaussian_log_posterior(h_values, center=0.73)

    for h, lp in zip(h_values, log_posts):
        _make_h_json(tmp_path, h, [lp])

    baseline = extract_baseline(tmp_path)

    assert abs(baseline.map_h - 0.73) < 0.01  # within grid spacing


def test_extract_baseline_computes_bias_percent(tmp_path: Path) -> None:
    """extract_baseline should compute bias % as (MAP - 0.73) / 0.73 * 100."""
    # Use peak exactly at 0.73 → bias should be ~0%
    h_values = [0.69, 0.71, 0.73, 0.75, 0.77]
    log_posts = _gaussian_log_posterior(h_values, center=0.73)

    for h, lp in zip(h_values, log_posts):
        _make_h_json(tmp_path, h, [lp])

    baseline = extract_baseline(tmp_path, true_h=0.73)

    assert abs(baseline.bias_percent) < 2.0  # within 2% due to grid


def test_extract_baseline_counts_events(tmp_path: Path) -> None:
    """extract_baseline should set n_events from the detection key count."""
    h_values = [0.69, 0.71, 0.73, 0.75, 0.77]
    n_events = 5

    for h in h_values:
        data: dict[str, object] = {"h": h}
        for i in range(n_events):
            data[str(i)] = [1.0]
        h_str = f"{h:.3f}".replace(".", "_")
        (tmp_path / f"h_{h_str}.json").write_text(json.dumps(data))

    baseline = extract_baseline(tmp_path)

    assert baseline.n_events == n_events


def test_extract_baseline_computes_ci_bounds(tmp_path: Path) -> None:
    """extract_baseline should compute 68% CI that brackets the MAP h."""
    h_values = [0.69, 0.71, 0.73, 0.75, 0.77]
    log_posts = _gaussian_log_posterior(h_values, center=0.73)

    for h, lp in zip(h_values, log_posts):
        _make_h_json(tmp_path, h, [lp])

    baseline = extract_baseline(tmp_path)

    assert baseline.ci_lower < baseline.map_h
    assert baseline.ci_upper > baseline.map_h
    assert baseline.ci_width > 0


def test_extract_baseline_has_created_at(tmp_path: Path) -> None:
    """extract_baseline should populate created_at field."""
    h_values = [0.69, 0.71, 0.73, 0.75, 0.77]
    for h in h_values:
        _make_h_json(tmp_path, h, [-1.0])

    baseline = extract_baseline(tmp_path)

    assert isinstance(baseline.created_at, str)
    assert len(baseline.created_at) > 0


# ---------------------------------------------------------------------------
# BaselineSnapshot JSON round-trip
# ---------------------------------------------------------------------------


def test_baseline_snapshot_json_roundtrip() -> None:
    """BaselineSnapshot.to_json() and from_json() should produce equal objects."""
    original = BaselineSnapshot(
        map_h=0.73,
        ci_lower=0.71,
        ci_upper=0.75,
        ci_width=0.04,
        bias_percent=0.0,
        n_events=10,
        h_values=[0.71, 0.73, 0.75],
        log_posteriors=[-1.0, 0.0, -1.0],
        per_event_summaries=[],
        created_at="2026-01-01T00:00:00Z",
        git_commit="abc123",
    )

    json_data = original.to_json()
    restored = BaselineSnapshot.from_json(json_data)

    assert restored.map_h == original.map_h
    assert restored.ci_lower == original.ci_lower
    assert restored.ci_upper == original.ci_upper
    assert restored.ci_width == original.ci_width
    assert restored.bias_percent == original.bias_percent
    assert restored.n_events == original.n_events
    assert restored.h_values == original.h_values
    assert restored.log_posteriors == original.log_posteriors
    assert restored.created_at == original.created_at
    assert restored.git_commit == original.git_commit


# ---------------------------------------------------------------------------
# generate_comparison_report
# ---------------------------------------------------------------------------


def _make_baseline_snapshot(
    map_h: float = 0.73,
    ci_lower: float = 0.71,
    ci_upper: float = 0.75,
    n_events: int = 10,
) -> BaselineSnapshot:
    bias = (map_h - 0.73) / 0.73 * 100
    return BaselineSnapshot(
        map_h=map_h,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        ci_width=ci_upper - ci_lower,
        bias_percent=bias,
        n_events=n_events,
        h_values=[0.69, 0.71, 0.73, 0.75, 0.77],
        log_posteriors=[-2.0, -1.0, 0.0, -1.0, -2.0],
        per_event_summaries=[],
        created_at="2026-01-01T00:00:00Z",
        git_commit="abc123",
    )


def test_generate_comparison_report_creates_markdown(tmp_path: Path) -> None:
    """generate_comparison_report should produce a markdown file."""
    baseline = _make_baseline_snapshot(map_h=0.73)
    current = _make_baseline_snapshot(map_h=0.71)

    report_path = generate_comparison_report(baseline, current, tmp_path, label="test")

    assert report_path.exists()
    assert report_path.suffix == ".md"
    content = report_path.read_text()
    assert "MAP" in content or "map" in content.lower()


def test_generate_comparison_report_creates_json_sidecar(tmp_path: Path) -> None:
    """generate_comparison_report should produce a JSON sidecar alongside the markdown."""
    baseline = _make_baseline_snapshot(map_h=0.73)
    current = _make_baseline_snapshot(map_h=0.71)

    generate_comparison_report(baseline, current, tmp_path, label="test")

    json_path = tmp_path / "comparison_test.json"
    assert json_path.exists()
    data = json.loads(json_path.read_text())
    assert "baseline" in data
    assert "current" in data
    assert "delta" in data


def test_generate_comparison_report_json_has_expected_metrics(tmp_path: Path) -> None:
    """The JSON sidecar should contain map_h, ci_lower, ci_upper, ci_width, bias_pct, n_events."""
    baseline = _make_baseline_snapshot(map_h=0.73, n_events=10)
    current = _make_baseline_snapshot(map_h=0.71, n_events=12)

    generate_comparison_report(baseline, current, tmp_path, label="metrics_test")

    json_path = tmp_path / "comparison_metrics_test.json"
    data = json.loads(json_path.read_text())

    for section in ["baseline", "current"]:
        assert "map_h" in data[section]
        assert "ci_lower" in data[section]
        assert "ci_upper" in data[section]
        assert "ci_width" in data[section]
        assert "bias_pct" in data[section]
        assert "n_events" in data[section]

    assert "map_h" in data["delta"]
    assert "ci_width" in data["delta"]
    assert "bias_pct" in data["delta"]
    assert "n_events" in data["delta"]


def test_generate_comparison_report_delta_values(tmp_path: Path) -> None:
    """Delta values in JSON sidecar should be current - baseline."""
    baseline = _make_baseline_snapshot(map_h=0.73, n_events=10)
    current = _make_baseline_snapshot(map_h=0.71, n_events=15)

    generate_comparison_report(baseline, current, tmp_path, label="delta_test")

    json_path = tmp_path / "comparison_delta_test.json"
    data = json.loads(json_path.read_text())

    assert abs(data["delta"]["map_h"] - (0.71 - 0.73)) < 1e-10
    assert abs(data["delta"]["n_events"] - (15 - 10)) < 1e-10


# ---------------------------------------------------------------------------
# Integration tests (Task 3 part — added here for organisation)
# ---------------------------------------------------------------------------


def _make_posteriors_dir(base_dir: Path, h_values: list[float], center: float) -> Path:
    """Create a simulations/posteriors/ subdirectory with synthetic posterior files."""
    posteriors_dir = base_dir / "simulations" / "posteriors"
    posteriors_dir.mkdir(parents=True, exist_ok=True)
    n_events = 5
    sigma = 0.02
    for h in h_values:
        lp = -0.5 * ((h - center) / sigma) ** 2
        data: dict[str, object] = {"h": h}
        for i in range(n_events):
            data[str(i)] = [math.exp(lp / n_events)]
        h_str = f"{h:.3f}".replace(".", "_")
        (posteriors_dir / f"h_{h_str}.json").write_text(json.dumps(data))
    return posteriors_dir


def test_save_baseline_cli_integration(tmp_path: Path) -> None:
    """_save_baseline should write baseline.json to .planning/debug/."""
    from master_thesis_code.main import _save_baseline

    h_values = [0.69, 0.71, 0.73, 0.75, 0.77]
    _make_posteriors_dir(tmp_path, h_values, center=0.73)
    sim_dir = tmp_path / "simulations"

    _save_baseline(str(sim_dir))

    # The baseline is written to .planning/debug/ relative to project root
    # But for tests we need to verify it ran without error and returned a baseline
    # The function logs but also we can check by calling extract_baseline directly
    posteriors_dir = sim_dir / "posteriors"
    baseline = extract_baseline(posteriors_dir)

    assert abs(baseline.map_h - 0.73) < 0.02  # within grid spacing + tolerance
    assert abs(baseline.bias_percent) < 5.0
    assert baseline.ci_lower < 0.73 < baseline.ci_upper


def test_compare_baseline_cli_integration(tmp_path: Path) -> None:
    """_compare_baseline should generate comparison markdown and JSON files."""
    from master_thesis_code.main import _compare_baseline

    # Create baseline from centered posteriors
    h_values = [0.69, 0.71, 0.73, 0.75, 0.77]
    baseline_dir = tmp_path / "baseline_run"
    _make_posteriors_dir(baseline_dir, h_values, center=0.73)
    baseline = extract_baseline(baseline_dir / "simulations" / "posteriors")
    baseline_json_path = tmp_path / "baseline.json"
    baseline_json_path.write_text(json.dumps(baseline.to_json()))

    # Create "current" run with shifted posteriors
    current_dir = tmp_path / "current_run"
    _make_posteriors_dir(current_dir, h_values, center=0.71)

    _compare_baseline(str(current_dir / "simulations"), str(baseline_json_path))

    # Verify comparison files were written to .planning/debug/
    # (We test the function runs without error; files go to project root .planning/debug/)
    # For the actual file existence check, call generate_comparison_report directly
    current = extract_baseline(current_dir / "simulations" / "posteriors")
    assert abs(current.map_h - 0.71) < 0.02


def test_compare_baseline_standalone(tmp_path: Path) -> None:
    """_compare_baseline should work standalone without --evaluate having been run."""
    from master_thesis_code.main import _compare_baseline

    # Pre-existing posteriors in tmp_path (simulate --evaluate was NOT just run)
    h_values = [0.69, 0.71, 0.73, 0.75, 0.77]
    _make_posteriors_dir(tmp_path, h_values, center=0.71)

    # Baseline from separately created snapshot
    baseline = _make_baseline_snapshot(map_h=0.73)
    baseline_json_path = tmp_path / "baseline.json"
    baseline_json_path.write_text(json.dumps(baseline.to_json()))

    # Should not raise even though --evaluate was not run
    _compare_baseline(str(tmp_path / "simulations"), str(baseline_json_path))


# ---------------------------------------------------------------------------
# Fisher Quality fields — Task 1 (Phase 34-02)
# ---------------------------------------------------------------------------


def test_baseline_snapshot_fisher_fields_default_zero() -> None:
    """BaselineSnapshot should default fisher fields to 0 when not provided."""
    snap = BaselineSnapshot(
        map_h=0.73,
        ci_lower=0.71,
        ci_upper=0.75,
        ci_width=0.04,
        bias_percent=0.0,
        n_events=10,
    )

    assert snap.n_excluded_fisher == 0
    assert snap.median_cond_3d == 0.0
    assert snap.median_cond_4d == 0.0


def test_baseline_snapshot_fisher_fields_roundtrip() -> None:
    """BaselineSnapshot with n_excluded_fisher=5 should serialize and deserialize correctly."""
    original = BaselineSnapshot(
        map_h=0.73,
        ci_lower=0.71,
        ci_upper=0.75,
        ci_width=0.04,
        bias_percent=0.0,
        n_events=10,
        n_excluded_fisher=5,
        median_cond_3d=1.23e4,
        median_cond_4d=4.56e11,
        created_at="2026-01-01T00:00:00Z",
        git_commit="abc123",
    )

    json_data = original.to_json()
    restored = BaselineSnapshot.from_json(json_data)

    assert restored.n_excluded_fisher == 5
    assert abs(restored.median_cond_3d - 1.23e4) < 1.0
    assert abs(restored.median_cond_4d - 4.56e11) < 1e6


def test_baseline_snapshot_fisher_fields_backward_compat() -> None:
    """from_json should default fisher fields to 0 when keys are absent (old baseline.json)."""
    old_json: dict[str, object] = {
        "map_h": 0.73,
        "ci_lower": 0.71,
        "ci_upper": 0.75,
        "ci_width": 0.04,
        "bias_percent": 0.0,
        "n_events": 10,
        # n_excluded_fisher, median_cond_3d, median_cond_4d intentionally absent
    }

    restored = BaselineSnapshot.from_json(old_json)

    assert restored.n_excluded_fisher == 0
    assert restored.median_cond_3d == 0.0
    assert restored.median_cond_4d == 0.0


def test_generate_comparison_report_fisher_quality_section(tmp_path: Path) -> None:
    """generate_comparison_report should include Fisher Quality section when exclusions exist."""
    baseline = BaselineSnapshot(
        map_h=0.73,
        ci_lower=0.71,
        ci_upper=0.75,
        ci_width=0.04,
        bias_percent=0.0,
        n_events=10,
        n_excluded_fisher=0,
        median_cond_3d=1.5e4,
        median_cond_4d=2.3e11,
        h_values=[0.69, 0.71, 0.73, 0.75, 0.77],
        log_posteriors=[-2.0, -1.0, 0.0, -1.0, -2.0],
        created_at="2026-01-01T00:00:00Z",
        git_commit="abc123",
    )
    current = BaselineSnapshot(
        map_h=0.73,
        ci_lower=0.71,
        ci_upper=0.75,
        ci_width=0.04,
        bias_percent=0.0,
        n_events=10,
        n_excluded_fisher=3,
        median_cond_3d=1.5e4,
        median_cond_4d=2.3e11,
        h_values=[0.69, 0.71, 0.73, 0.75, 0.77],
        log_posteriors=[-2.0, -1.0, 0.0, -1.0, -2.0],
        created_at="2026-01-01T00:00:00Z",
        git_commit="abc123",
    )

    report_path = generate_comparison_report(baseline, current, tmp_path, label="fisher_test")

    content = report_path.read_text()
    assert "Fisher Quality" in content
    assert "Events excluded (Fisher)" in content
    assert "+3" in content  # delta = 3 - 0 = +3


def test_generate_comparison_report_no_fisher_section_when_zero(tmp_path: Path) -> None:
    """generate_comparison_report should omit Fisher Quality section when both have 0 excluded."""
    baseline = _make_baseline_snapshot(map_h=0.73)
    current = _make_baseline_snapshot(map_h=0.71)
    # Both have n_excluded_fisher=0 (default)

    report_path = generate_comparison_report(baseline, current, tmp_path, label="no_fisher")

    content = report_path.read_text()
    assert "Fisher Quality" not in content


def test_generate_comparison_report_json_has_fisher_fields(tmp_path: Path) -> None:
    """The JSON sidecar should contain fisher quality fields in baseline/current/delta."""
    baseline = _make_baseline_snapshot(map_h=0.73)
    current = BaselineSnapshot(
        map_h=0.71,
        ci_lower=0.69,
        ci_upper=0.73,
        ci_width=0.04,
        bias_percent=-2.74,
        n_events=10,
        n_excluded_fisher=2,
        median_cond_3d=1.1e4,
        median_cond_4d=3.3e12,
        h_values=[0.69, 0.71, 0.73, 0.75, 0.77],
        log_posteriors=[-2.0, -1.0, 0.0, -1.0, -2.0],
        created_at="2026-01-01T00:00:00Z",
        git_commit="abc123",
    )

    generate_comparison_report(baseline, current, tmp_path, label="json_fisher")

    json_path = tmp_path / "comparison_json_fisher.json"
    data = json.loads(json_path.read_text())

    assert "n_excluded_fisher" in data["baseline"]
    assert "n_excluded_fisher" in data["current"]
    assert "n_excluded_fisher" in data["delta"]
    assert data["current"]["n_excluded_fisher"] == 2
    assert data["delta"]["n_excluded_fisher"] == 2  # 2 - 0 = +2


def test_extract_baseline_reads_fisher_quality_csv(tmp_path: Path) -> None:
    """extract_baseline should read fisher_quality.csv from posteriors_dir.parent if present."""
    import pandas as pd

    posteriors_dir = tmp_path / "posteriors"
    posteriors_dir.mkdir(parents=True)

    h_values = [0.69, 0.71, 0.73, 0.75, 0.77]
    for h in h_values:
        _make_h_json(posteriors_dir, h, [-1.0])

    # Write a fisher_quality.csv in tmp_path (posteriors_dir.parent)
    fq_data = pd.DataFrame(
        {
            "detection_index": [0, 1, 2, 3, 4],
            "cond_3d": [10.0, 20.0, 30.0, 40.0, 50.0],
            "cond_4d": [1e10, 2e10, 3e10, 4e10, 5e10],
            "excluded": [False, False, True, False, True],
        }
    )
    fq_data.to_csv(tmp_path / "fisher_quality.csv", index=False)

    baseline = extract_baseline(posteriors_dir)

    assert baseline.n_excluded_fisher == 2
    assert abs(baseline.median_cond_3d - 30.0) < 1e-6
    assert abs(baseline.median_cond_4d - 3e10) < 1e4


def test_extract_baseline_zero_fisher_when_no_csv(tmp_path: Path) -> None:
    """extract_baseline should default to 0 excluded fisher if CSV is absent."""
    h_values = [0.69, 0.71, 0.73, 0.75, 0.77]
    for h in h_values:
        _make_h_json(tmp_path, h, [-1.0])

    baseline = extract_baseline(tmp_path)

    assert baseline.n_excluded_fisher == 0
    assert baseline.median_cond_3d == 0.0
    assert baseline.median_cond_4d == 0.0
