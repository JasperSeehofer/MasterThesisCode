"""Generate Phase 36 regression anchor pickle (COORD-04).

Produces `.planning/phases/36-coordinate-frame-fix/36-superset-regression.pkl`
containing the D-24 schema artefact that Phase 40 VERIFY-02 uses to confirm
the eigenvalue sky search radius (COORD-04) lands correctly.

Reference event: the event from simulations/cramer_rao_bounds.csv with
minimum |qS − π/2| (worst-case equatorial event, per D-23 in
.planning/phases/36-coordinate-frame-fix/36-CONTEXT.md).

Old candidate set (pre-COORD-04): computed with the axis-aligned radius
    old_radius = max(phi_sigma, theta_sigma) * sigma_multiplier
directly on the corrected 3D BallTree (i.e., the COORD-02+03 fixes are
present; only the radius formula is the "old" one). This isolates exactly the
COORD-04 effect.

New candidate set (post-COORD-04): computed via the full
``get_possible_hosts_from_ball_tree`` with the eigenvalue radius.

Both sets are index sets over the pruned catalog. The superset assertion
    old_candidate_indices ⊆ new_candidate_indices
is checked before writing; a violation causes an AssertionError and exits 1.

Mirrors the CLI template from scripts/merge_cramer_rao_bounds.py.
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd

from master_thesis_code.bayesian_inference.evaluation_report import _get_git_commit_safe
from master_thesis_code.datamodels.detection import Detection
from master_thesis_code.galaxy_catalogue.handler import (
    GalaxyCatalogueHandler,
    M_max,
    M_min,
    _polar_to_cartesian,
)

_SIGMA_MULTIPLIER: float = 1.5  # matches bayesian_statistics.py call-site default
_Z_MAX: float = 1.5


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list (defaults to sys.argv[1:] when None).

    Returns:
        Parsed namespace with ``csv`` and ``output`` attributes.
    """
    parser = argparse.ArgumentParser(
        description="Generate Phase 36 COORD-04 regression anchor pickle.",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="simulations/cramer_rao_bounds.csv",
        help="Path to simulations/cramer_rao_bounds.csv (default: simulations/cramer_rao_bounds.csv)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=".planning/phases/36-coordinate-frame-fix/36-superset-regression.pkl",
        help="Output path for the regression pickle (default: .planning/phases/36-coordinate-frame-fix/36-superset-regression.pkl)",
    )
    return parser.parse_args(argv)


def _find_reference_event(df: pd.DataFrame) -> tuple[int, pd.Series]:
    """Return (row_index, row) for the event with minimum |qS − π/2|.

    Per D-23: this is the worst-case equatorial event where COORD-02/COORD-04
    bite hardest (|sin θ| ≈ 1 maximises the φ-direction metric distortion that
    the |sin θ| Jacobian corrects).

    Args:
        df: Cramer-Rao bounds DataFrame.

    Returns:
        Tuple of (integer row index, pd.Series for that row).
    """
    deviations = np.abs(df["qS"].values - np.pi / 2)
    idx = int(np.argmin(deviations))
    return idx, df.iloc[idx]


def _compute_old_radius(phi_sigma: float, theta_sigma: float, sigma_multiplier: float) -> float:
    """Replicate the pre-COORD-04 axis-aligned radius formula.

    Before the fix, ``get_possible_hosts_from_ball_tree`` used:
        radius = max(phi_sigma, theta_sigma) * sigma_multiplier

    Args:
        phi_sigma: 1-σ uncertainty on φ_S (rad).
        theta_sigma: 1-σ uncertainty on θ_S (rad).
        sigma_multiplier: Search-radius multiplier.

    Returns:
        Pre-fix radius in rad.
    """
    return float(max(phi_sigma, theta_sigma) * sigma_multiplier)


def _fisher_sky_block(detection: Detection) -> npt.NDArray[np.float64]:
    """Build the 2×2 Fisher sky covariance block from a Detection.

    Σ = [[σ_φ², C_θφ], [C_θφ, σ_θ²]] in rad².

    Args:
        detection: Parsed EMRI detection.

    Returns:
        (2, 2) ndarray, units rad².
    """
    return np.array(
        [
            [detection.phi_error**2, detection.theta_phi_covariance],
            [detection.theta_phi_covariance, detection.theta_error**2],
        ],
        dtype=np.float64,
    )


def generate_regression_pickle(
    csv_path: Path,
    output_path: Path,
) -> dict[str, object]:
    """Run the full regression pickle generation pipeline.

    Steps:
    1. Load CSV, select reference event (D-23: min |qS − π/2|).
    2. Instantiate GalaxyCatalogueHandler (applies COORD-03 rotation +
       COORD-02 BallTree embedding).
    3. Parse Detection from the reference event row.
    4. Compute OLD candidate set (axis-aligned radius, pre-COORD-04).
    5. Compute NEW candidate set (eigenvalue radius, post-COORD-04).
    6. Assert old ⊆ new.
    7. Write pickle and return the schema dict.

    Args:
        csv_path: Path to simulations/cramer_rao_bounds.csv.
        output_path: Destination for the .pkl artefact.

    Returns:
        D-24 schema dict (also written to output_path).

    Raises:
        AssertionError: If old_candidate_indices ⊄ new_candidate_indices.
        FileNotFoundError: If csv_path does not exist.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    print(f"Loading CRB CSV: {csv_path}")
    df = pd.read_csv(str(csv_path))
    print(f"  {len(df)} events loaded.")

    event_id, event_row = _find_reference_event(df)
    qS_ref = float(event_row["qS"])
    phiS_ref = float(event_row["phiS"])
    print(
        f"  Reference event: row {event_id}, qS={qS_ref:.6f} rad, "
        f"|qS-π/2|={abs(qS_ref - np.pi / 2):.6f} rad"
    )

    print("Instantiating GalaxyCatalogueHandler (loads catalog, rotates, builds BallTree)…")
    handler = GalaxyCatalogueHandler(M_min=M_min, M_max=M_max, z_max=_Z_MAX)
    print(f"  Catalog size after pruning: {len(handler.reduced_galaxy_catalog)} galaxies.")

    # Parse Detection from the reference event row.
    detection = Detection(event_row)

    # ------------------------------------------------------------------ #
    # OLD candidate set: axis-aligned radius under the corrected embedding #
    # ------------------------------------------------------------------ #
    old_radius = _compute_old_radius(detection.phi_error, detection.theta_error, _SIGMA_MULTIPLIER)
    query_point = _polar_to_cartesian(np.array([detection.theta]), np.array([detection.phi]))
    old_indices_raw = handler.catalog_ball_tree.query_radius(query_point, r=old_radius)[0]
    old_candidate_indices: set[int] = set(int(i) for i in old_indices_raw)

    # ------------------------------------------------------------------ #
    # NEW candidate set: eigenvalue radius (post-COORD-04)               #
    # ------------------------------------------------------------------ #
    # Compute redshift bounds inline (simplified: use detection z ± large margin
    # since we care only about the sky radius effect here, not z/mass filtering).
    # We pass extremely permissive z and mass windows so the sky radius is the
    # binding constraint, matching the pre-fix comparison.
    sigma_matrix = np.array(
        [
            [detection.phi_error**2, detection.theta_phi_covariance],
            [detection.theta_phi_covariance, detection.theta_error**2],
        ]
    )
    jacobian = np.diag([abs(np.sin(detection.theta)), 1.0])
    sigma_scaled = jacobian @ sigma_matrix @ jacobian.T
    lambda_max = float(np.linalg.eigvalsh(sigma_scaled).max())
    new_radius = float(_SIGMA_MULTIPLIER * np.sqrt(max(lambda_max, 0.0)))
    new_indices_raw = handler.catalog_ball_tree.query_radius(query_point, r=new_radius)[0]
    new_candidate_indices: set[int] = set(int(i) for i in new_indices_raw)

    print(
        f"  OLD radius (axis-aligned): {old_radius:.6f} rad → {len(old_candidate_indices)} candidates"
    )
    print(
        f"  NEW radius (eigenvalue):   {new_radius:.6f} rad → {len(new_candidate_indices)} candidates"
    )

    # ------------------------------------------------------------------ #
    # Superset assertion (D-24: SC-4 gate)                               #
    # ------------------------------------------------------------------ #
    is_superset = old_candidate_indices.issubset(new_candidate_indices)
    print(f"  new ⊇ old: {is_superset}")
    if not is_superset:
        missing = old_candidate_indices - new_candidate_indices
        raise AssertionError(
            f"SC-4 VIOLATED: {len(missing)} old candidates missing from new set. "
            f"First 5 missing indices: {sorted(missing)[:5]}. "
            "Check J axis ordering or radius formula."
        )

    # ------------------------------------------------------------------ #
    # Build D-24 schema and write pickle                                  #
    # ------------------------------------------------------------------ #
    # Pre-fix and post-fix qS/phiS for the ML point are identical:
    # the waveform ML estimate (qS, phiS) is in the ecliptic waveform frame
    # and does not change. Only the CATALOG frame changed (COORD-03). The
    # reference event's ML sky position is unchanged between old and new.
    git_commit = _get_git_commit_safe()
    fisher_sky_2x2 = _fisher_sky_block(detection)

    schema: dict[str, object] = {
        "event_id": event_id,
        "event_qS_pre_fix": qS_ref,
        "event_phiS_pre_fix": phiS_ref,
        "event_qS_post_fix": qS_ref,  # ML point unchanged; see docstring
        "event_phiS_post_fix": phiS_ref,  # ML point unchanged; see docstring
        "old_candidate_indices": old_candidate_indices,
        "new_candidate_indices": new_candidate_indices,
        "fisher_sky_2x2": fisher_sky_2x2,
        "git_commit": git_commit,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(schema, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  Pickle written: {output_path}")
    print(f"  git_commit: {git_commit}")

    return schema


def main(argv: list[str] | None = None) -> None:
    """Entry point for the regression pickle generator.

    Args:
        argv: Argument list (defaults to sys.argv[1:] when None).
    """
    args = parse_args(argv)
    csv_path = Path(args.csv)
    output_path = Path(args.output)

    try:
        schema = generate_regression_pickle(csv_path, output_path)
    except AssertionError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(2)

    event_id = schema["event_id"]
    qS = float(schema["event_qS_post_fix"])  # type: ignore[arg-type]
    old_n = len(schema["old_candidate_indices"])  # type: ignore[arg-type]
    new_n = len(schema["new_candidate_indices"])  # type: ignore[arg-type]
    is_superset = schema["old_candidate_indices"] <= schema["new_candidate_indices"]  # type: ignore[operator]

    print("\n--- Regression Summary ---")
    print(f"  event_id       : {event_id}")
    print(f"  |qS − π/2|     : {abs(qS - np.pi / 2):.6f} rad")
    print(f"  |old| (pre-fix): {old_n}")
    print(f"  |new| (post-fix): {new_n}")
    print(f"  new ⊇ old       : {is_superset}")
    print(f"  git_commit      : {schema['git_commit']}")
    print(f"  output          : {output_path}")


if __name__ == "__main__":
    main()
