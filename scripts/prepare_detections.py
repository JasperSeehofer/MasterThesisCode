"""Convert detection parameters to best-guess parameters for Bayesian inference.

Reads merged Cramer-Rao bounds CSV, applies correlated multivariate-normal
sampling (or independent truncated-normal as legacy fallback) to convert
maximum-likelihood estimates to best-guess values, and writes the prepared CSV.

Safety features:
- ``--seed`` for reproducible random draws
- ``--force`` required to overwrite an existing prepared file
- Metadata sidecar (JSON) recording provenance and integrity checksums
- Input file (``cramer_rao_bounds.csv``) is never mutated

Designed for non-interactive use in SLURM batch jobs.
"""

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
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
        Parsed namespace with ``workdir``, ``seed``, and ``force`` attributes.
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
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible draws (enables correlated sampling)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing prepared file without error",
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
    meta_path = output_path.with_suffix(".meta.json")

    if not input_path.exists():
        print(f"Error: input file not found: {input_path}")
        sys.exit(1)

    # Idempotency guard: refuse to overwrite without --force
    if output_path.exists() and not args.force:
        print(
            f"Error: {output_path} already exists. "
            "Use --force to overwrite, or delete the file first."
        )
        sys.exit(1)

    # Create RNG if seed provided (enables correlated multivariate sampling)
    rng: np.random.Generator | None = None
    if args.seed is not None:
        rng = np.random.default_rng(args.seed)
        print(f"Using seed {args.seed} (correlated multivariate sampling)")
    else:
        print("No seed provided (legacy independent truncated-normal sampling)")

    # Compute input checksum before any processing
    input_sha256 = hashlib.sha256(input_path.read_bytes()).hexdigest()

    print(f"Reading cramer rao bounds and detection parameters from {input_path}")
    cramer_rao_bounds = pd.read_csv(input_path)

    prepared_cramer_rao_bounds = cramer_rao_bounds.copy()
    unusable_detections: list[int] = []

    print("Converting detection parameters to best guess parameters")
    for detection_index, detection_row in cramer_rao_bounds.iterrows():
        print(f"progress: {detection_index}/{len(cramer_rao_bounds)}")
        if detection_row["delta_phiS_delta_phiS"] < 0:
            unusable_detections.append(int(detection_index))
            continue
        detection = Detection(detection_row)
        print(f"Detection {detection_index}: {detection}")
        detection.convert_to_best_guess_parameters(rng=rng)
        prepared_cramer_rao_bounds.at[detection_index, "M"] = detection.M
        prepared_cramer_rao_bounds.at[detection_index, "luminosity_distance"] = detection.d_L
        prepared_cramer_rao_bounds.at[detection_index, "phiS"] = detection.phi
        prepared_cramer_rao_bounds.at[detection_index, "qS"] = detection.theta

    if len(unusable_detections) > 0:
        print(f"{len(unusable_detections)} detections had to be deleted")
        # Only drop from the prepared output — input file stays immutable.
        prepared_cramer_rao_bounds.drop(unusable_detections, inplace=True)

    prepared_cramer_rao_bounds.to_csv(output_path, index=False)
    print(f"Saved prepared cramer rao bounds to {output_path}")

    # Write metadata sidecar for forensic traceability
    meta = {
        "prepared_at": datetime.now(UTC).isoformat(),
        "seed": args.seed,
        "sampling_method": "multivariate_normal" if rng is not None else "independent_truncnorm",
        "input_file": str(input_path),
        "input_sha256": input_sha256,
        "input_rows": len(cramer_rao_bounds),
        "output_rows": len(prepared_cramer_rao_bounds),
        "unusable_detections_dropped": len(unusable_detections),
        "unusable_detection_indices": unusable_detections,
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"Saved metadata to {meta_path}")


if __name__ == "__main__":
    main()
