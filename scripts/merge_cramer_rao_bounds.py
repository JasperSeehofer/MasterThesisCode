"""Merge per-index Cramer-Rao bounds and undetected events CSVs.

Designed for non-interactive use in SLURM batch jobs. All interactive
prompts have been replaced by the ``--delete-sources`` flag.
"""

import argparse
import os
from pathlib import Path

import pandas as pd

from master_thesis_code.constants import (
    CRAMER_RAO_BOUNDS_OUTPUT_PATH,
    CRAMER_RAO_BOUNDS_PATH,
    UNDETECTED_EVENTS_OUTPUT_PATH,
    UNDETECTED_EVENTS_PATH,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the merge script.

    Args:
        argv: Argument list (defaults to sys.argv[1:] when None).

    Returns:
        Parsed namespace with ``workdir`` and ``delete_sources`` attributes.
    """
    parser = argparse.ArgumentParser(
        description="Merge per-index Cramer-Rao bounds and undetected events CSVs.",
    )
    parser.add_argument(
        "--workdir",
        type=str,
        default=".",
        help="Working directory; paths resolved relative to this",
    )
    parser.add_argument(
        "--delete-sources",
        action="store_true",
        help="Delete per-index source CSVs after successful merge",
    )
    return parser.parse_args(argv)


def delete_files(used_files: list[str]) -> None:
    """Delete a list of files from the filesystem.

    Args:
        used_files: Absolute or relative paths to delete.
    """
    for file in used_files:
        os.remove(file)
        print(f"Deleted {file}.")


def merge_cramer_rao_bounds(workdir: Path, delete_sources: bool) -> None:
    """Merge per-index Cramer-Rao bounds CSVs into one output file.

    Args:
        workdir: Working directory; all paths resolved relative to this.
        delete_sources: If True, delete source CSVs after successful merge.
    """
    output_path = workdir / CRAMER_RAO_BOUNDS_OUTPUT_PATH
    directory = output_path.parent
    if not directory.exists():
        print("No simulations directory found.")
        return

    common_file_name = CRAMER_RAO_BOUNDS_PATH.split(".")[0].replace("$index", "").split("/")[1]
    files = os.listdir(directory)
    cramer_rao_bounds_files = [str(directory / file) for file in files if common_file_name in file]

    used_files: list[str] = []

    if len(cramer_rao_bounds_files) == 0:
        print("No cramer rao bounds files found.")
        return

    # check for existing output file
    if output_path.exists():
        cramer_rao_bounds = pd.read_csv(output_path)
    else:
        first_file = cramer_rao_bounds_files.pop(0)
        cramer_rao_bounds = pd.read_csv(first_file)
        used_files.append(first_file)

    if len(cramer_rao_bounds_files) == 0:
        cramer_rao_bounds.to_csv(output_path, index=False)
        print(f"Only one cramer rao bounds file found. Saved as {output_path}.")
        if delete_sources:
            delete_files(used_files)
        return

    print(f"Merging {len(cramer_rao_bounds_files)} cramer rao bounds files.")

    for file in cramer_rao_bounds_files:
        cramer_rao_bounds = pd.concat([cramer_rao_bounds, pd.read_csv(file)])
        used_files.append(file)

    cramer_rao_bounds.to_csv(output_path, index=False)
    print(f"Saved merged cramer rao bounds as {output_path}.")
    if delete_sources:
        delete_files(used_files)


def merge_undetected_events(workdir: Path, delete_sources: bool) -> None:
    """Merge per-index undetected events CSVs into one output file.

    Args:
        workdir: Working directory; all paths resolved relative to this.
        delete_sources: If True, delete source CSVs after successful merge.
    """
    output_path = workdir / UNDETECTED_EVENTS_OUTPUT_PATH
    directory = output_path.parent
    if not directory.exists():
        print("No simulations directory found.")
        return

    common_file_name = UNDETECTED_EVENTS_PATH.split(".")[0].replace("$index", "").split("/")[1]
    files = os.listdir(directory)
    undetected_events_files = [str(directory / file) for file in files if common_file_name in file]

    used_files: list[str] = []

    if len(undetected_events_files) == 0:
        print("No undetected events files found.")
        return

    if output_path.exists():
        undetected_events = pd.read_csv(output_path)
    else:
        first_file = undetected_events_files.pop(0)
        undetected_events = pd.read_csv(first_file)
        used_files.append(first_file)

    if len(undetected_events_files) == 0:
        undetected_events.to_csv(output_path, index=False)
        print(f"Only one undetected events file found. Saved as {output_path}.")
        if delete_sources:
            delete_files(used_files)
        return

    print(f"Merging {len(undetected_events_files)} undetected events files.")

    for file in undetected_events_files:
        undetected_events = pd.concat([undetected_events, pd.read_csv(file)])
        used_files.append(file)

    undetected_events.to_csv(output_path, index=False)
    print(f"Saved merged undetected events as {output_path}.")
    if delete_sources:
        delete_files(used_files)


def main(argv: list[str] | None = None) -> None:
    """Entry point for the merge script.

    Args:
        argv: Argument list (defaults to sys.argv[1:] when None).
    """
    args = parse_args(argv)
    workdir = Path(args.workdir)
    merge_cramer_rao_bounds(workdir, args.delete_sources)
    merge_undetected_events(workdir, args.delete_sources)


if __name__ == "__main__":
    main()
