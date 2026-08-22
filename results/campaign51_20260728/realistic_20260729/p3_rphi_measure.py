"""[P3-RPHI] stage-0 re-measurement: r_phi(h) = Sigma^phi / Sigma^3D on the B-SEL venue objects."""

import numpy as np

from darksiren_emri.bayesian_inference.bayesian_statistics import (
    precompute_global_catalog_selection,
    precompute_phi_marginal_survival,
)
from darksiren_emri.bayesian_inference.simulation_detection_probability import (
    SimulationDetectionProbability,
)
from darksiren_emri.constants import SNR_THRESHOLD
from darksiren_emri.validation.correspondence_1d import (
    HOST_DRAW_Z_MAX,
    INJECTION_POOL_DIR,
    REDUCED_CATALOGUE_PATH,
    _load_galaxy_catalog_handler,
)

H_PROBE = [0.6, 0.665, 0.73, 0.795, 0.86]


def main() -> int:
    handler = _load_galaxy_catalog_handler(REDUCED_CATALOGUE_PATH)
    det = SimulationDetectionProbability(
        injection_data_dir=INJECTION_POOL_DIR,
        snr_threshold=SNR_THRESHOLD,
        dl_bins=60,
        mass_bins=40,
        estimator="local_linear",
        expected_z_max=HOST_DRAW_Z_MAX,
        allow_shallow_pool=True,
        pdet_z_resolved=True,
    )
    phi_table = precompute_phi_marginal_survival(
        h_values=H_PROBE, detection_probability_obj=det, z_max_cap=HOST_DRAW_Z_MAX
    )
    sigma_3d = precompute_global_catalog_selection(
        h_values=H_PROBE,
        galaxy_catalog=handler,
        detection_probability_obj=det,
        with_bh_mass=False,
        z_max_cap=HOST_DRAW_Z_MAX,
        smear_sigma_z=False,
    )
    sigma_phi = precompute_global_catalog_selection(
        h_values=H_PROBE,
        galaxy_catalog=handler,
        detection_probability_obj=det,
        with_bh_mass=False,
        z_max_cap=HOST_DRAW_Z_MAX,
        smear_sigma_z=False,
        phi_survival_table=phi_table,
    )
    print("h      Sigma^phi        Sigma^3D         r_phi")
    r = {}
    for h in H_PROBE:
        r[h] = sigma_phi[h] / sigma_3d[h]
        print(f"{h:.3f}  {sigma_phi[h]:.6e}  {sigma_3d[h]:.6e}  {r[h]:.6f}")
    lo, hi = min(H_PROBE), max(H_PROBE)
    print(
        f"r_phi(0.73) = {r[0.73]:.6f}; d ln r_phi/dh (chord) = "
        f"{(np.log(r[hi]) - np.log(r[lo])) / (hi - lo):+.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
