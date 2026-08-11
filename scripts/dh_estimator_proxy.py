"""D(h) estimator proxy: old histogram p_det vs F4 Nadaraya-Watson kernel.

Isolates the *estimator* effect on the selection-function denominator D(h)
on a fixed injection set, holding cosmology / integrator / constants constant
(all byte-identical between d1087f1^ and HEAD).

D(h) enters the joint posterior as effectively -N*log D(h) (post-Tier-3, via
each per-event L_comp = num/D), so the SLOPE of log D(h) is what pulls the MAP.
If F4 makes log D(h) more steeply decreasing in h, F4 pushes the MAP higher.

Usage: uv run python scripts/dh_estimator_proxy.py
"""

import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from darksiren_emri.bayesian_inference.bayesian_statistics import (
    precompute_completion_denominator,
)
from darksiren_emri.bayesian_inference.simulation_detection_probability import (
    SimulationDetectionProbability as F4Estimator,
)
from darksiren_emri.constants import OMEGA_DE, OMEGA_M, SNR_THRESHOLD

INJ_DIR = "simulations/injections"
H_VALUES = [0.72, 0.73, 0.74, 0.76, 0.78]
N_EVENTS = 937  # phase50 events-used, for the -N log D magnitude scale
OLD_PARENT = "d1087f1^"


def _load_old_estimator() -> type:
    """Load the pre-F4 histogram estimator class from git, standalone."""
    src = subprocess.run(
        [
            "git",
            "show",
            f"{OLD_PARENT}:darksiren_emri/bayesian_inference/simulation_detection_probability.py",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    tmp = Path(tempfile.mkdtemp()) / "old_sim_det_prob.py"
    tmp.write_text(src)
    spec = importlib.util.spec_from_file_location("old_sim_det_prob", tmp)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod.SimulationDetectionProbability


def _dh(estimator: object) -> np.ndarray:
    table = precompute_completion_denominator(
        H_VALUES,
        estimator,
        Omega_m=OMEGA_M,
        Omega_DE=OMEGA_DE,
    )
    return np.array([table[h] for h in H_VALUES], dtype=np.float64)


def main() -> None:
    OldEstimator = _load_old_estimator()
    print(f"Building estimators on {INJ_DIR} (snr_thr={SNR_THRESHOLD}, 60x40 grid)...")
    old = OldEstimator(INJ_DIR, SNR_THRESHOLD, dl_bins=60, mass_bins=40)
    f4 = F4Estimator(INJ_DIR, SNR_THRESHOLD, dl_bins=60, mass_bins=40)

    d_old = _dh(old)
    d_f4 = _dh(f4)

    i73 = H_VALUES.index(0.73)
    # The MAP-pull contribution of the D(h) channel, relative to truth:
    #   pull(h) = -N [log D(h) - log D(0.73)]
    pull_old = -N_EVENTS * (np.log(d_old) - np.log(d_old[i73]))
    pull_f4 = -N_EVENTS * (np.log(d_f4) - np.log(d_f4[i73]))

    print("\n h      D_old        D_F4        | -N dlogD vs 0.73 (the MAP pull)")
    print(" " + "-" * 70)
    for k, h in enumerate(H_VALUES):
        print(
            f" {h:.3f}  {d_old[k]:.4e}  {d_f4[k]:.4e}  |  old {pull_old[k]:+8.2f}   F4 {pull_f4[k]:+8.2f}"
        )

    print("\nInterpretation:")
    print("  'pull' = D(h)-channel contribution to [joint(h) - joint(0.73)].")
    print("  argmax over h = where the D(h) channel alone wants the MAP.")
    print(f"  OLD histogram : D-channel favours h = {H_VALUES[int(np.argmax(pull_old))]:.3f}")
    print(f"  F4 kernel     : D-channel favours h = {H_VALUES[int(np.argmax(pull_f4))]:.3f}")
    dmax = float(np.max(np.abs(pull_f4 - pull_old)))
    print(f"  max |F4 - old| pull difference across grid = {dmax:.2f} log-units")
    print("  (>~1-2 log-units => estimator materially reshapes D(h) => F4 implicated;")
    print("   ~0 => D(h) channel unchanged by F4 => points to seed/N, run full eval to confirm.)")


if __name__ == "__main__":
    sys.exit(main())
