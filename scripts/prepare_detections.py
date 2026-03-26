"""Convert detection parameters to best-guess parameters for Bayesian inference.

Reads merged Cramer-Rao bounds CSV, applies truncated-normal sampling to convert
maximum-likelihood estimates to best-guess values, and writes the prepared CSV.

Designed for non-interactive use in SLURM batch jobs.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

from master_thesis_code.constants import (
    CRAMER_RAO_BOUNDS_OUTPUT_PATH,
    PREPARED_CRAMER_RAO_BOUNDS_PATH,
)
from master_thesis_code.cosmological_model import Detection


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the prepare script.

    Args:
        argv: Argument list (defaults to sys.argv[1:] when None).

    Returns:
        Parsed namespace with ``workdir`` attribute.
    """
    parser = argparse.ArgumentParser(
        description="Convert detection parameters to best-guess parameters.",
    )
    parser.add_argument(
        "--workdir",
        type=str,
        default=".",
        help="Working directory; paths resolved relative to this",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Entry point for the prepare detections script.

    Args:
        argv: Argument list (defaults to sys.argv[1:] when None).
    """
    args = parse_args(argv)
    workdir = Path(args.workdir)

    input_path = workdir / CRAMER_RAO_BOUNDS_OUTPUT_PATH
    output_path = workdir / PREPARED_CRAMER_RAO_BOUNDS_PATH

    if not input_path.exists():
        print(f"Error: input file not found: {input_path}")
        sys.exit(1)

    print(f"Reading cramer rao bounds and detection parameters from {input_path}")
    cramer_rao_bounds = pd.read_csv(input_path)

    prepared_cramer_rao_bounds = cramer_rao_bounds.copy()
    unusable_detections: list[int] = []

    print("Converting detection parameters to best guess parameters")
    for detection_index, detection_row in cramer_rao_bounds.iterrows():
        print(f"progress: {detection_index}/{len(cramer_rao_bounds)}")
        if detection_row["delta_phiS_delta_phiS"] < 0:
            unusable_detections.append(int(detection_index))  # type: ignore[arg-type]
            continue
        detection = Detection(detection_row)
        print(f"Detection {detection_index}: {detection}")
        detection.convert_to_best_guess_parameters()
        prepared_cramer_rao_bounds.at[detection_index, "M"] = detection.M
        prepared_cramer_rao_bounds.at[detection_index, "luminosity_distance"] = detection.d_L
        prepared_cramer_rao_bounds.at[detection_index, "phiS"] = detection.phi
        prepared_cramer_rao_bounds.at[detection_index, "qS"] = detection.theta

    if len(unusable_detections) > 0:
        print(f"{len(unusable_detections)} detections had to be deleted")
        cramer_rao_bounds.drop(unusable_detections, inplace=True)
        prepared_cramer_rao_bounds.drop(unusable_detections, inplace=True)
        cramer_rao_bounds.to_csv(input_path, index=False)

    prepared_cramer_rao_bounds.to_csv(output_path, index=False)
    print(f"Saved prepared cramer rao bounds to {output_path}")


if __name__ == "__main__":
    main()
