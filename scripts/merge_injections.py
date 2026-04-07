"""Safely merge new injection CSVs into an existing injection directory.

Validates columns, checks h_inj consistency with filenames, renumbers task
indices to avoid collisions, and maintains a merge log for bookkeeping.

Designed for non-interactive use; ``--dry-run`` previews without copying.
"""

import argparse
import json
import re
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

# Required columns in injection CSVs (SimulationDetectionProbability contract)
REQUIRED_COLUMNS: frozenset[str] = frozenset(
    {"z", "M", "phiS", "qS", "SNR", "h_inj", "luminosity_distance"}
)

# Regex for injection filenames: injection_h_{label}_task_{index}.csv
_FILENAME_RE = re.compile(r"injection_h_(\d+p\d+)(?:_task_(\d+))?\.csv")


def _label_to_h(label: str) -> float:
    """Convert filename label like '0p65' to float 0.65."""
    return float(label.replace("p", "."))


def _h_to_label(h: float) -> str:
    """Convert float h-value to filename label like '0p65'."""
    return f"{h:.2f}".replace(".", "p").rstrip("0").rstrip("p") or "0"


def _next_task_index(target_dir: Path, h_label: str) -> int:
    """Find the next available task index for a given h-value in target."""
    existing = list(target_dir.glob(f"injection_h_{h_label}_task_*.csv"))
    if not existing:
        return 0
    indices: list[int] = []
    for f in existing:
        m = _FILENAME_RE.match(f.name)
        if m and m.group(2) is not None:
            indices.append(int(m.group(2)))
    return max(indices) + 1 if indices else 0


def _validate_csv(path: Path) -> tuple[bool, str]:
    """Validate an injection CSV file.

    Returns:
        Tuple of (is_valid, error_message). error_message is empty on success.
    """
    try:
        df = pd.read_csv(path, nrows=5)
    except Exception as e:
        return False, f"Cannot read CSV: {e}"

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        return False, f"Missing columns: {sorted(missing)}"

    # Check h_inj consistency with filename
    m = _FILENAME_RE.match(path.name)
    if not m:
        return False, f"Filename does not match expected pattern: {path.name}"

    h_from_name = _label_to_h(m.group(1))
    h_from_data = df["h_inj"].iloc[0]
    if abs(h_from_name - h_from_data) > 0.005:
        return (
            False,
            f"h_inj mismatch: filename says {h_from_name}, data says {h_from_data}",
        )

    return True, ""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list (defaults to sys.argv[1:] when None).

    Returns:
        Parsed namespace.
    """
    parser = argparse.ArgumentParser(
        description="Merge new injection CSVs into an existing injection directory.",
    )
    parser.add_argument(
        "--source-dir",
        type=str,
        required=True,
        help="Directory containing new injection CSVs to merge",
    )
    parser.add_argument(
        "--target-dir",
        type=str,
        required=True,
        help="Existing consolidated injection directory",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and report without copying any files",
    )
    parser.add_argument(
        "--move",
        action="store_true",
        help="Move files instead of copying (deletes source after transfer)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow re-merging from a source directory that was already merged",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Entry point for the injection merge script.

    Args:
        argv: Argument list (defaults to sys.argv[1:] when None).
    """
    args = parse_args(argv)
    source_dir = Path(args.source_dir)
    target_dir = Path(args.target_dir)

    if not source_dir.is_dir():
        print(f"Error: source directory not found: {source_dir}")
        sys.exit(1)

    if not target_dir.is_dir():
        print(f"Error: target directory not found: {target_dir}")
        sys.exit(1)

    # Check for duplicate source merge
    source_runs_path = target_dir / "source_runs.json"
    source_runs: list[str] = []
    if source_runs_path.exists():
        source_runs = json.loads(source_runs_path.read_text())

    source_key = str(source_dir.resolve())
    if source_key in source_runs and not args.force:
        print(f"Error: {source_dir} was already merged into {target_dir}. Use --force to re-merge.")
        sys.exit(1)

    # Discover source CSVs
    source_csvs = sorted(source_dir.glob("injection_h_*.csv"))
    if not source_csvs:
        print(f"No injection CSVs found in {source_dir}")
        sys.exit(1)

    print(f"Found {len(source_csvs)} injection CSV(s) in {source_dir}")

    # Validate all files first
    valid_files: list[tuple[Path, str, int]] = []  # (path, h_label, event_count)
    errors: list[str] = []

    for csv_path in source_csvs:
        is_valid, error_msg = _validate_csv(csv_path)
        if not is_valid:
            errors.append(f"  {csv_path.name}: {error_msg}")
            continue

        m = _FILENAME_RE.match(csv_path.name)
        assert m is not None  # validated above
        h_label = m.group(1)
        event_count = len(pd.read_csv(csv_path))
        valid_files.append((csv_path, h_label, event_count))

    if errors:
        print(f"\nValidation errors ({len(errors)} files):")
        for err in errors:
            print(err)

    if not valid_files:
        print("\nNo valid files to merge.")
        sys.exit(1)

    # Group by h-value for summary
    events_by_h: dict[str, int] = {}
    for _, h_label, count in valid_files:
        h_val = str(_label_to_h(h_label))
        events_by_h[h_val] = events_by_h.get(h_val, 0) + count

    # Plan the merge (renumber task indices)
    merge_plan: list[tuple[Path, Path, str]] = []  # (source, target, h_label)

    for csv_path, h_label, _ in valid_files:
        new_index = _next_task_index(target_dir, h_label)
        # Check if other files in this batch already claimed indices
        for _, planned_target, planned_h in merge_plan:
            if planned_h == h_label:
                planned_m = _FILENAME_RE.match(planned_target.name)
                if planned_m and planned_m.group(2) is not None:
                    new_index = max(new_index, int(planned_m.group(2)) + 1)
        target_name = f"injection_h_{h_label}_task_{new_index}.csv"
        target_path = target_dir / target_name
        merge_plan.append((csv_path, target_path, h_label))

    # Print summary
    print(f"\nMerge plan: {len(valid_files)} files")
    print(f"  Source: {source_dir}")
    print(f"  Target: {target_dir}")
    print(f"  Mode: {'move' if args.move else 'copy'}")
    print("\n  Events by h-value:")
    for h_val in sorted(events_by_h):
        print(f"    h={h_val}: {events_by_h[h_val]} events")
    print("\n  File mapping:")
    for src, tgt, _ in merge_plan:
        print(f"    {src.name} -> {tgt.name}")

    if args.dry_run:
        print("\n[DRY RUN] No files were copied.")
        return

    # Execute the merge
    files_added: list[str] = []
    source_mapping: dict[str, str] = {}

    for src, tgt, _ in merge_plan:
        if args.move:
            shutil.move(str(src), str(tgt))
        else:
            shutil.copy2(str(src), str(tgt))
        files_added.append(tgt.name)
        source_mapping[tgt.name] = src.name

    print(f"\nMerged {len(files_added)} files.")

    # Update source_runs.json
    if source_key not in source_runs:
        source_runs.append(source_key)
    source_runs_path.write_text(json.dumps(source_runs, indent=2) + "\n")

    # Append to merge log
    log_path = target_dir / "injection_merge_log.json"
    log_entries: list[dict[str, object]] = []
    if log_path.exists():
        log_entries = json.loads(log_path.read_text())

    log_entries.append(
        {
            "merge_timestamp": datetime.now(UTC).isoformat(),
            "source_dir": str(source_dir),
            "files_added": files_added,
            "files_source_mapping": source_mapping,
            "events_added_by_h": events_by_h,
        }
    )
    log_path.write_text(json.dumps(log_entries, indent=2) + "\n")
    print(f"Updated merge log: {log_path}")


if __name__ == "__main__":
    main()
