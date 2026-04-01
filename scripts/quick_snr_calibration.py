"""Compute quick (1yr) and full (5yr) SNR for the same events to calibrate the threshold.

Run on GPU cluster:
    python scripts/quick_snr_calibration.py --steps 200 --seed 42
"""

import argparse
import signal
import warnings

import numpy as np

from master_thesis_code.cosmological_model import Model1CrossCheck
from master_thesis_code.galaxy_catalogue.handler import GalaxyCatalogueHandler, HostGalaxy
from master_thesis_code.parameter_estimation.parameter_estimation import ParameterEstimation
from master_thesis_code.waveform_generator import WaveGeneratorType


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick SNR calibration test")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    cosmological_model = Model1CrossCheck()
    galaxy_catalog = GalaxyCatalogueHandler()
    parameter_estimation = ParameterEstimation(
        WaveGeneratorType.PN5_AAK,
        cosmological_model.parameter_space,
        use_gpu=True,
        use_five_point_stencil=True,
    )

    parameter_samples = cosmological_model.sample_emri_events(200)
    host_galaxies = iter(galaxy_catalog.get_hosts_from_parameter_samples(parameter_samples))

    print("quick_snr,full_snr,ratio")

    pairs = 0
    iteration = 0
    while pairs < args.steps:
        iteration += 1
        try:
            host_galaxy = next(host_galaxies)
        except StopIteration:
            parameter_samples = cosmological_model.sample_emri_events(200)
            host_galaxies = iter(
                galaxy_catalog.get_hosts_from_parameter_samples(parameter_samples)
            )
            host_galaxy = next(host_galaxies)
        assert isinstance(host_galaxy, HostGalaxy)

        parameter_estimation.parameter_space.randomize_parameters(rng=rng)
        parameter_estimation.parameter_space.set_host_galaxy_parameters(host_galaxy)

        try:
            warnings.filterwarnings("error")
            signal.alarm(30)

            quick_snr = parameter_estimation.compute_signal_to_noise_ratio(
                use_snr_check_generator=True
            )

            # Skip events with negligible quick SNR to save time
            if quick_snr < 1.0:
                signal.alarm(0)
                warnings.resetwarnings()
                continue

            full_snr = parameter_estimation.compute_signal_to_noise_ratio()
            signal.alarm(0)
            warnings.resetwarnings()

            ratio = quick_snr / full_snr if full_snr > 0 else 0.0
            print(f"{quick_snr:.4f},{full_snr:.4f},{ratio:.4f}")
            pairs += 1

        except (Warning, Exception):
            signal.alarm(0)
            warnings.resetwarnings()
            continue

    print(f"\n# Collected {pairs} pairs from {iteration} attempts", flush=True)


if __name__ == "__main__":
    main()
