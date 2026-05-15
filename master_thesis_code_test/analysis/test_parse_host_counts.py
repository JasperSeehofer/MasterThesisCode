"""Tests for master_thesis_code.analysis.parse_host_counts."""

from pathlib import Path

import pandas as pd
import pytest

from master_thesis_code.analysis.parse_host_counts import (
    _HOST_COUNT_RE,
    build_host_count_csv,
    parse_host_count_lines,
    parse_host_counts,
)


def _make_log(tmp_path: Path, name: str, pairs: list[tuple[int, int]]) -> Path:
    """Write a synthetic inference log file with *pairs* host-count lines."""
    path = tmp_path / name
    lines: list[str] = ["2026-05-06 14:00:00,000 [main.py:41 - main()] STARTING\n"]
    for no, wm in pairs:
        lines.append(
            "2026-05-06 14:19:47,998 [handler.py:399 - get_possible_hosts_from_ball_tree()] "
            f"Found {no} possible hosts without BH mass and {wm} possible hosts with BH mass.\n"
        )
        # Add an unrelated line to ensure the regex is anchored on the right thing.
        lines.append("2026-05-06 14:19:48,000 [bayesian_statistics.py:784 - p_D()] foo bar\n")
    path.write_text("".join(lines))
    return path


class TestRegex:
    def test_matches_known_handler_line(self) -> None:
        line = (
            "2026-05-06 14:19:47,998 [handler.py:399 - get_possible_hosts_from_ball_tree()] "
            "Found 58 possible hosts without BH mass and 34 possible hosts with BH mass."
        )
        m = _HOST_COUNT_RE.search(line)
        assert m is not None
        assert int(m.group("no")) == 58
        assert int(m.group("wm")) == 34

    def test_does_not_match_other_lines(self) -> None:
        assert _HOST_COUNT_RE.search("Found 58 cats and 34 dogs.") is None
        assert _HOST_COUNT_RE.search("possible hosts without BH mass = 58") is None


class TestParseHostCountLines:
    def test_returns_pairs_in_order(self, tmp_path: Path) -> None:
        pairs = [(58, 34), (376, 138), (239, 76)]
        log = _make_log(tmp_path, "test.log", pairs)
        assert parse_host_count_lines(log) == pairs

    def test_empty_log_returns_empty_list(self, tmp_path: Path) -> None:
        log = tmp_path / "empty.log"
        log.write_text("[main.py] no host count lines here\n")
        assert parse_host_count_lines(log) == []


class TestParseHostCounts:
    def test_single_log_dataframe_shape_and_columns(self, tmp_path: Path) -> None:
        pairs = [(58, 34), (376, 138), (239, 76), (19, 5)]
        log = _make_log(tmp_path, "h_0_73.log", pairs)
        df = parse_host_counts([log])
        assert list(df.columns) == [
            "event_idx",
            "n_without_mass",
            "n_with_mass",
            "reduction_frac",
        ]
        assert len(df) == 4
        assert list(df["event_idx"]) == [0, 1, 2, 3]
        assert list(df["n_without_mass"]) == [58, 376, 239, 19]
        assert list(df["n_with_mass"]) == [34, 138, 76, 5]

    def test_reduction_frac_is_correct(self, tmp_path: Path) -> None:
        pairs = [(100, 50), (10, 10), (4, 1), (1000, 0)]
        log = _make_log(tmp_path, "h.log", pairs)
        df = parse_host_counts([log])
        assert df["reduction_frac"].iloc[0] == pytest.approx(0.5)
        assert df["reduction_frac"].iloc[1] == pytest.approx(0.0)
        assert df["reduction_frac"].iloc[2] == pytest.approx(0.75)
        assert df["reduction_frac"].iloc[3] == pytest.approx(1.0)

    def test_n_with_le_n_without(self, tmp_path: Path) -> None:
        pairs = [(58, 34), (376, 138)]
        log = _make_log(tmp_path, "h.log", pairs)
        df = parse_host_counts([log])
        assert (df["n_with_mass"] <= df["n_without_mass"]).all()

    def test_multiple_logs_cross_check_agreement(self, tmp_path: Path) -> None:
        pairs = [(58, 34), (376, 138)]
        log1 = _make_log(tmp_path, "h_a.log", pairs)
        log2 = _make_log(tmp_path, "h_b.log", pairs)
        df = parse_host_counts([log1, log2])
        # Cross-check should match; both logs produce identical DataFrames.
        assert len(df) == 2
        assert list(df["n_without_mass"]) == [58, 376]

    def test_missing_log_files_warns_but_returns_empty(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level("WARNING"):
            df = parse_host_counts([tmp_path / "does_not_exist.log"])
        assert df.empty
        assert "not found" in caplog.text.lower()

    def test_returns_empty_with_correct_dtypes_on_empty(self, tmp_path: Path) -> None:
        df = parse_host_counts([])
        assert isinstance(df, pd.DataFrame)
        assert df.empty
        # Dtypes should still be correct so downstream consumers don't crash.
        assert df["n_without_mass"].dtype.kind == "i"
        assert df["reduction_frac"].dtype.kind == "f"


class TestBuildHostCountCSV:
    def test_writes_csv_to_default_path(self, tmp_path: Path) -> None:
        pairs = [(58, 34), (376, 138)]
        # Layout: data_dir/logs/master_thesis_code_*_h_0_73.log (mirror of cluster sync)
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        _make_log(log_dir, "master_thesis_code_20260506_141633_h_0_73.log", pairs)
        df = build_host_count_csv(tmp_path)
        out_csv = tmp_path / "diagnostics" / "host_counts.csv"
        assert out_csv.is_file()
        loaded = pd.read_csv(out_csv)
        assert len(loaded) == len(df) == 2

    def test_raises_when_no_logs_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="No inference logs"):
            build_host_count_csv(tmp_path)

    def test_raises_when_logs_have_no_host_counts(self, tmp_path: Path) -> None:
        # A log file that exists but has no host-count lines (e.g. fig-gen log).
        (tmp_path / "master_thesis_code_20260515_h_0_73.log").write_text(
            "[main.py] no host-count line here\n"
        )
        with pytest.raises(FileNotFoundError, match="rsync from the cluster"):
            build_host_count_csv(tmp_path)

    def test_h_value_filter(self, tmp_path: Path) -> None:
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        _make_log(log_dir, "master_thesis_code_h_0_73.log", [(58, 34)])
        _make_log(log_dir, "master_thesis_code_h_0_65.log", [(99, 99)])
        df = build_host_count_csv(tmp_path, h_value_filter="h_0_73")
        assert len(df) == 1
        assert df["n_without_mass"].iloc[0] == 58
