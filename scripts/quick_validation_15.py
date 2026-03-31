#!/usr/bin/env python3
"""Quick validation for Phase 15: run evaluation at 4 h-values with P_det=1.

Compares post-fix "with BH mass" posterior against pre-fix baseline.
The /(1+z) fix should shift the peak from h<=0.600 toward h=0.678.

Usage:
    uv run python scripts/quick_validation_15.py
"""

import json
import logging
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

# Monkey-patch SimulationDetectionProbability BEFORE importing bayesian_statistics
# This allows running locally without injection campaign data.
mock_det_prob = MagicMock()
mock_det_prob.detection_probability_without_bh_mass_interpolated.return_value = 1.0
mock_det_prob.detection_probability_with_bh_mass_interpolated.return_value = 1.0

# Patch the module so BayesianStatistics can import it
import master_thesis_code.bayesian_inference.simulation_detection_probability as sdp_mod

_OrigClass = sdp_mod.SimulationDetectionProbability


class MockSimulationDetectionProbability:
    """P_det = 1 everywhere (matches pre-fix baseline conditions)."""

    def __init__(self, *args, **kwargs):  # noqa: ANN002, ANN003, ANN204
        pass

    def detection_probability_without_bh_mass_interpolated(
        self,
        *args,
        **kwargs,  # noqa: ANN002, ANN003
    ) -> float:
        return 1.0

    def detection_probability_with_bh_mass_interpolated(
        self,
        *args,
        **kwargs,  # noqa: ANN002, ANN003
    ) -> float:
        return 1.0


sdp_mod.SimulationDetectionProbability = MockSimulationDetectionProbability  # type: ignore[misc]

# Now import the rest
from master_thesis_code.bayesian_inference.bayesian_statistics import BayesianStatistics
from master_thesis_code.cosmological_model import Model1CrossCheck
from master_thesis_code.galaxy_catalogue.handler import GalaxyCatalogueHandler

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
_LOGGER = logging.getLogger(__name__)


def main() -> None:
    h_values = [0.652, 0.678, 0.704, 0.730]

    cosmological_model = Model1CrossCheck()
    galaxy_catalog = GalaxyCatalogueHandler(
        M_min=cosmological_model.parameter_space.M.lower_limit,
        M_max=cosmological_model.parameter_space.M.upper_limit,
        z_max=cosmological_model.max_redshift,
    )

    results_with_bh: dict[float, float] = {}
    results_without_bh: dict[float, float] = {}

    for h in h_values:
        _LOGGER.info(f"=== Evaluating h = {h} ===")
        stats = BayesianStatistics()
        stats.evaluate(galaxy_catalog, cosmological_model, h, num_workers=4)

        # Extract sum of per-detection likelihoods (product in log-space)
        detection_keys = sorted(
            [k for k in stats.posterior_data.keys() if isinstance(k, int)], key=int
        )
        sum_without = sum(
            stats.posterior_data[k][-1] for k in detection_keys if stats.posterior_data[k]
        )
        sum_with = sum(
            stats.posterior_data_with_bh_mass[k][-1]
            for k in detection_keys
            if stats.posterior_data_with_bh_mass[k]
        )
        log_prod_without = sum(
            np.log(stats.posterior_data[k][-1])
            for k in detection_keys
            if stats.posterior_data[k] and stats.posterior_data[k][-1] > 0
        )
        log_prod_with = sum(
            np.log(stats.posterior_data_with_bh_mass[k][-1])
            for k in detection_keys
            if stats.posterior_data_with_bh_mass[k] and stats.posterior_data_with_bh_mass[k][-1] > 0
        )

        results_with_bh[h] = log_prod_with
        results_without_bh[h] = log_prod_without

        _LOGGER.info(
            f"h={h}: with_BH_log={log_prod_with:.4f}, without_BH_log={log_prod_without:.4f}"
        )
        _LOGGER.info(f"h={h}: with_BH_sum={sum_with:.6e}, without_BH_sum={sum_without:.6e}")

    # Print comparison table
    print("\n" + "=" * 70)
    print("PHASE 15 QUICK VALIDATION RESULTS (P_det = 1)")
    print("=" * 70)
    print(f"\n{'h':>6} | {'with BH (log)':>14} | {'w/o BH (log)':>14}")
    print("-" * 45)
    for h in h_values:
        print(f"{h:>6.3f} | {results_with_bh[h]:>14.4f} | {results_without_bh[h]:>14.4f}")

    # Directional test
    print("\n--- DIRECTIONAL TEST ---")
    if results_with_bh[0.678] > results_with_bh[0.652]:
        print("PASS: 'with BH mass' posterior at h=0.678 > h=0.652")
        print("      Peak has shifted from h<=0.600 toward h=0.678")
    else:
        print("FAIL: 'with BH mass' posterior at h=0.678 <= h=0.652")
        print("      Peak may not have shifted as expected")

    # Find approximate peak
    peak_h = max(h_values, key=lambda h: results_with_bh[h])
    print(f"\nApproximate 'with BH mass' peak: h ~ {peak_h}")
    print("Pre-fix baseline peak: h <= 0.600 (monotonically decreasing)")

    # Save results
    output = {
        "h_values": h_values,
        "with_bh_mass_log_posterior": {str(h): results_with_bh[h] for h in h_values},
        "without_bh_mass_log_posterior": {str(h): results_without_bh[h] for h in h_values},
        "p_det": "P_det = 1 (mock)",
        "fix_applied": "/(1+z) removed from line 655",
    }
    output_path = Path(".gpd/phases/15-code-audit-fix/15-quick-validation-results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
