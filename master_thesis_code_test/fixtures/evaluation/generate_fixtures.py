"""Generate synthetic CSV fixtures for evaluation pipeline integration tests.

Produces three CSV files that mimic the output of the EMRI simulation pipeline:
- synthetic_cramer_rao_bounds.csv (5 detected events)
- synthetic_prepared_cramer_rao_bounds.csv (identical — skips randomness)
- synthetic_undetected_events.csv (20 sub-threshold events)

Run as a script to regenerate:
    uv run python master_thesis_code_test/fixtures/evaluation/generate_fixtures.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

from master_thesis_code.physical_relations import dist

# 14 EMRI parameter names in CSV order
PARAM_NAMES = [
    "M",
    "mu",
    "a",
    "p0",
    "e0",
    "x0",
    "luminosity_distance",
    "qS",
    "phiS",
    "qK",
    "phiK",
    "Phi_phi0",
    "Phi_theta0",
    "Phi_r0",
]

# Fisher matrix column names: lower-triangular entries
FISHER_COLUMNS: list[str] = []
for i, p_i in enumerate(PARAM_NAMES):
    for j in range(i + 1):
        p_j = PARAM_NAMES[j]
        FISHER_COLUMNS.append(f"delta_{p_i}_delta_{p_j}")

METADATA_COLUMNS = [
    "T",
    "dt",
    "SNR",
    "generation_time",
    "host_galaxy_index",
    "_coord_frame",
    "_cov_frame",
]

DETECTED_COLUMNS = PARAM_NAMES + FISHER_COLUMNS + METADATA_COLUMNS
UNDETECTED_COLUMNS = PARAM_NAMES + ["T", "dt", "SNR", "generation_time"]

# ── Detected events specification ──────────────────────────────────────────

DETECTED_SPECS = [
    {"z": 0.05, "M_z": 5e4, "phiS": 1.0, "qS": 0.8, "host_idx": 0},
    {"z": 0.10, "M_z": 1e5, "phiS": 2.0, "qS": 1.2, "host_idx": 5},
    {"z": 0.15, "M_z": 2e5, "phiS": 3.0, "qS": 0.5, "host_idx": 10},
    {"z": 0.20, "M_z": 3e5, "phiS": 4.0, "qS": 1.5, "host_idx": 15},
    {"z": 0.25, "M_z": 5e5, "phiS": 5.0, "qS": 2.0, "host_idx": 20},
]


def _make_detected_row(spec: dict) -> dict:
    """Build a single detected-event CSV row from a specification dict."""
    d_L = dist(spec["z"])
    M_z = spec["M_z"]
    phiS = spec["phiS"]
    qS = spec["qS"]

    row: dict = {
        "M": M_z,
        "mu": 10.0,
        "a": 0.98,
        "p0": 10.0,
        "e0": 0.1,
        "x0": 1.0,
        "luminosity_distance": d_L,
        "qS": qS,
        "phiS": phiS,
        "qK": 1.0,
        "phiK": 2.0,
        "Phi_phi0": 0.0,
        "Phi_theta0": 0.0,
        "Phi_r0": 0.0,
    }

    # Fisher matrix entries: all zero except diagonal variances
    for col in FISHER_COLUMNS:
        row[col] = 0.0

    # Set diagonal variances to give 1.5% relative error on d_L,
    # small errors on angles and mass
    row["delta_M_delta_M"] = (M_z * 0.005) ** 2  # 0.5% mass error
    row["delta_mu_delta_mu"] = (10.0 * 0.001) ** 2
    row["delta_a_delta_a"] = (0.98 * 0.001) ** 2
    row["delta_p0_delta_p0"] = (10.0 * 0.001) ** 2
    row["delta_e0_delta_e0"] = (0.1 * 0.01) ** 2
    row["delta_x0_delta_x0"] = (1.0 * 0.001) ** 2
    row["delta_luminosity_distance_delta_luminosity_distance"] = (d_L * 0.015) ** 2
    row["delta_qS_delta_qS"] = (0.005) ** 2  # ~0.3 deg
    row["delta_phiS_delta_phiS"] = (0.005) ** 2
    row["delta_qK_delta_qK"] = (0.005) ** 2
    row["delta_phiK_delta_phiK"] = (0.005) ** 2
    row["delta_Phi_phi0_delta_Phi_phi0"] = (0.01) ** 2
    row["delta_Phi_theta0_delta_Phi_theta0"] = (0.01) ** 2
    row["delta_Phi_r0_delta_Phi_r0"] = (0.01) ** 2

    # Metadata
    row["T"] = 5.0
    row["dt"] = 10.0
    row["SNR"] = 30.0
    row["generation_time"] = 1.0
    row["host_galaxy_index"] = spec["host_idx"]
    row["_coord_frame"] = "ecliptic_BarycentricTrue_J2000"
    row["_cov_frame"] = "ecliptic_BarycentricTrue_J2000"

    return row


def _make_undetected_row(rng: np.random.Generator, index: int) -> dict:
    """Build a single undetected-event CSV row with SNR < 20."""
    z = rng.uniform(0.01, 1.0)
    d_L = dist(z)
    M = 10 ** rng.uniform(4.5, 6.0)
    return {
        "M": M,
        "mu": 10.0,
        "a": rng.uniform(0.5, 0.998),
        "p0": rng.uniform(10.0, 16.0),
        "e0": rng.uniform(0.05, 0.2),
        "x0": rng.uniform(-1.0, 1.0),
        "luminosity_distance": d_L,
        "qS": np.arccos(rng.uniform(-1.0, 1.0)),
        "phiS": rng.uniform(0.0, 2 * np.pi),
        "qK": np.arccos(rng.uniform(-1.0, 1.0)),
        "phiK": rng.uniform(0.0, 2 * np.pi),
        "Phi_phi0": rng.uniform(0.0, 2 * np.pi),
        "Phi_theta0": rng.uniform(0.0, 2 * np.pi),
        "Phi_r0": rng.uniform(0.0, 2 * np.pi),
        "T": 5.0,
        "dt": 10.0,
        "SNR": rng.uniform(5.0, 19.9),
        "generation_time": 1.0,
    }


def generate_fixtures(output_dir: Path | None = None) -> None:
    """Generate and write the synthetic fixture CSVs.

    Produces the full 5-detection set plus 1-detection and 3-detection
    subsets used by the posterior-narrowing integration test.
    """
    if output_dir is None:
        output_dir = Path(__file__).parent

    # Detected events
    detected_rows = [_make_detected_row(spec) for spec in DETECTED_SPECS]
    df_detected = pd.DataFrame(detected_rows, columns=DETECTED_COLUMNS)

    df_detected.to_csv(output_dir / "synthetic_cramer_rao_bounds.csv", index=False)
    df_detected.to_csv(output_dir / "synthetic_prepared_cramer_rao_bounds.csv", index=False)

    # Write 1-detection and 3-detection subsets
    for n in (1, 3):
        subset = df_detected.iloc[:n]
        subset.to_csv(output_dir / f"synthetic_cramer_rao_bounds_{n}det.csv", index=False)
        subset.to_csv(output_dir / f"synthetic_prepared_cramer_rao_bounds_{n}det.csv", index=False)

    # Undetected events
    rng = np.random.default_rng(seed=42)
    undetected_rows = [_make_undetected_row(rng, i) for i in range(20)]
    df_undetected = pd.DataFrame(undetected_rows, columns=UNDETECTED_COLUMNS)
    df_undetected.to_csv(output_dir / "synthetic_undetected_events.csv", index=False)

    print(f"Fixtures written to {output_dir}")
    print(f"  detected:   {len(df_detected)} rows x {len(df_detected.columns)} cols")
    print("  subsets:    1det, 3det")
    print(f"  undetected: {len(df_undetected)} rows x {len(df_undetected.columns)} cols")


if __name__ == "__main__":
    generate_fixtures()
