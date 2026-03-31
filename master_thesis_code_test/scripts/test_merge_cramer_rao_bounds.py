"""Tests for batch-compatible merge_cramer_rao_bounds script."""

from pathlib import Path

import pandas as pd

from scripts.merge_cramer_rao_bounds import main


def _write_dummy_csv(path: Path, rows: list[dict[str, float]]) -> None:
    """Write a minimal CSV with header and data rows."""
    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _make_source_csvs(
    tmp_path: Path,
    prefix: str = "cramer_rao_bounds_simulation_",
    count: int = 3,
) -> list[Path]:
    """Create per-index CSV files in tmp_path/simulations/."""
    sim_dir = tmp_path / "simulations"
    sim_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(count):
        p = sim_dir / f"{prefix}{i}.csv"
        _write_dummy_csv(p, [{"col_a": float(i), "col_b": float(i * 10)}])
        paths.append(p)
    return paths



class TestMergeNoDelete:
    """main(["--workdir", tmpdir]) merges CSVs and keeps source files."""

    def test_source_files_still_exist(self, tmp_path: Path) -> None:
        sources = _make_source_csvs(tmp_path)
        main(["--workdir", str(tmp_path)])
        for src in sources:
            assert src.exists(), f"Source file should still exist: {src}"

    def test_output_file_created(self, tmp_path: Path) -> None:
        _make_source_csvs(tmp_path)
        main(["--workdir", str(tmp_path)])
        output = tmp_path / "simulations" / "cramer_rao_bounds.csv"
        assert output.exists()

    def test_output_contains_all_rows(self, tmp_path: Path) -> None:
        _make_source_csvs(tmp_path, count=3)
        main(["--workdir", str(tmp_path)])
        output = tmp_path / "simulations" / "cramer_rao_bounds.csv"
        df = pd.read_csv(output)
        assert len(df) == 3  # one row per source file


class TestMergeWithDelete:
    """main(["--workdir", tmpdir, "--delete-sources"]) deletes source files."""

    def test_source_files_deleted(self, tmp_path: Path) -> None:
        sources = _make_source_csvs(tmp_path)
        main(["--workdir", str(tmp_path), "--delete-sources"])
        for src in sources:
            assert not src.exists(), f"Source file should be deleted: {src}"

    def test_output_file_exists(self, tmp_path: Path) -> None:
        _make_source_csvs(tmp_path)
        main(["--workdir", str(tmp_path), "--delete-sources"])
        output = tmp_path / "simulations" / "cramer_rao_bounds.csv"
        assert output.exists()


class TestMergeEmpty:
    """main(["--workdir", tmpdir]) with no source CSVs exits cleanly."""

    def test_no_source_files_exits_cleanly(self, tmp_path: Path) -> None:
        sim_dir = tmp_path / "simulations"
        sim_dir.mkdir(parents=True, exist_ok=True)
        # Should not raise
        main(["--workdir", str(tmp_path)])


class TestNoInputCalls:
    """The module must not contain any input() calls."""

    def test_no_input_in_source(self) -> None:
        source_file = Path(__file__).resolve().parents[2] / "scripts" / "merge_cramer_rao_bounds.py"
        content = source_file.read_text()
        assert "input(" not in content, "merge_cramer_rao_bounds.py must not contain input() calls"


