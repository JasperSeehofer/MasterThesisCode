"""G1 soundness-gate check: discrete GLADE Sigma_global(h) vs continuous beta_G(h).

The Option-A partition-norm likelihood relies on the discrete catalogue sum

    Sigma_global(h) = sum_{g: z_g < z_max(h)} w_g P_det(d_L(z_g, h)),
    w_g = R_eff(M_g) / (1 + z_g)

being a faithful Monte-Carlo realisation of the continuous in-catalogue
selection integral (Gray et al. 2020, arXiv:1908.06050, Eq. 29)

    beta_G(h) = INTEGRAL f(z) P_det(d_L(z,h)) dV_c/(1+z) dz = D(h) - beta_Gbar(h)

up to an h-INDEPENDENT constant (n_gal x mass-integrated rate) that cancels in
the likelihood. The commission flagged this cancellation as "delicate ...
should be checked against the real GLADE sum" (verification report §7 /
scratch/d2 RESULT 3 caveat) — this script performs that check.

Only the h-SHAPE matters: an h-dependent ratio Sigma_global(h)/beta_G(h) is a
direct multiplicative distortion of the in-catalogue channel's h-dependence
(and of the w_G = beta_G/D mixing weight consistency). We report the ratio
normalised at the fiducial h and its maximum fractional deviation.

Usage:
    uv run python scripts/verify_beta_g_discrete_sum.py \
        --injections_dir <dir with injection_h_*.csv> \
        [--h_min 0.60 --h_max 0.86 --h_steps 14] \
        [--output_json .planning/gate/G1_beta_g_check.json]

Writes a JSON result table + prints a verdict. Read-only w.r.t. all inputs.
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from darksiren_emri.bayesian_inference.bayesian_statistics import (
    precompute_completion_denominator,
    precompute_global_catalog_selection,
    precompute_missing_completion_denominator,
)
from darksiren_emri.bayesian_inference.simulation_detection_probability import (
    SimulationDetectionProbability,
)
from darksiren_emri.constants import SNR_THRESHOLD
from darksiren_emri.cosmological_model import Model1CrossCheck
from darksiren_emri.galaxy_catalogue.handler import GalaxyCatalogueHandler
from darksiren_emri.galaxy_catalogue.pixel_completeness import from_cache_or_build

_LOGGER = logging.getLogger(__name__)

# Fiducial cosmology of the inference (constants.py) — the check is at fixed
# Omega_m; the ratio shape is insensitive to the exact value.
from darksiren_emri.constants import OMEGA_DE as _OMEGA_DE  # noqa: E402
from darksiren_emri.constants import OMEGA_M as _OMEGA_M  # noqa: E402

_H_FIDUCIAL = 0.73


def run_check(
    injections_dir: str,
    h_values: list[float],
    n_sky_bands: int,
    output_json: str,
) -> dict[str, object]:
    """Compute Sigma_global(h), beta_G(h) = D(h)-beta_Gbar(h), and their ratio shape."""
    _LOGGER.info("Loading detection probability from %s ...", injections_dir)
    pdet = SimulationDetectionProbability(
        injections_dir, snr_threshold=float(SNR_THRESHOLD), n_sky_bands=n_sky_bands
    )
    _LOGGER.info("Loading galaxy catalogue (this reads the full reduced catalogue) ...")
    # Same bounds as production main.py:94 (rate-model parameter space).
    model = Model1CrossCheck()
    catalog = GalaxyCatalogueHandler(
        M_min=model.parameter_space.M.lower_limit,
        M_max=model.parameter_space.M.upper_limit,
        z_max=model.max_redshift,
    )
    _LOGGER.info("Building per-pixel completeness ...")
    completeness = from_cache_or_build()

    _LOGGER.info("Continuous D(h) ...")
    D_h = precompute_completion_denominator(
        h_values, pdet, _OMEGA_M, _OMEGA_DE, completeness=completeness
    )
    _LOGGER.info("Continuous beta_Gbar(h) ...")
    beta_gbar = precompute_missing_completion_denominator(h_values, pdet, completeness)
    _LOGGER.info("Discrete Sigma_global(h) over the full catalogue ...")
    sigma_global = precompute_global_catalog_selection(h_values, catalog, pdet, with_bh_mass=False)

    beta_g = {h: D_h[h] - beta_gbar[h] for h in h_values}
    ratio = {h: sigma_global[h] / beta_g[h] if beta_g[h] > 0 else np.nan for h in h_values}

    h_ref = min(h_values, key=lambda h: abs(h - _H_FIDUCIAL))
    r_ref = ratio[h_ref]
    shape = {h: ratio[h] / r_ref for h in h_values}
    max_dev = float(max(abs(s - 1.0) for s in shape.values()))

    # EXPECTED h-dependence of the raw ratio: the catalogue is a FIXED set of
    # galaxies, so its implied comoving number density scales as
    # n_gal(h) ∝ 1/V_c ∝ h^3 (distances ∝ 1/h). This n_gal(h) factor is COMMON
    # to the discrete numerator and denominator sums of L_cat and cancels in the
    # likelihood; the h^3-corrected shape below isolates the residual Option-A
    # cancellation error that does NOT cancel (it multiplies the 'global'-mode
    # in-catalogue channel directly; local_ratio/volume_deconv never use
    # Sigma_global, so they are immune to this leg).
    shape_h3 = {h: shape[h] * (h_ref / h) ** 3 for h in h_values}
    max_dev_h3 = float(max(abs(s - 1.0) for s in shape_h3.values()))

    # The shape error enters the joint posterior once per event through the
    # in-catalogue channel; N_events amplifies a coherent tilt. Report both the
    # raw max deviation and the end-to-end tilt across the grid.
    h_lo, h_hi = min(h_values), max(h_values)
    tilt = float(shape[h_hi] - shape[h_lo])
    tilt_h3 = float(shape_h3[h_hi] - shape_h3[h_lo])

    result: dict[str, object] = {
        "injections_dir": injections_dir,
        "n_sky_bands": n_sky_bands,
        "snr_threshold": float(SNR_THRESHOLD),
        "h_ref": h_ref,
        "table": {
            f"{h:.4f}": {
                "D": D_h[h],
                "beta_Gbar": beta_gbar[h],
                "beta_G": beta_g[h],
                "Sigma_global": sigma_global[h],
                "ratio": ratio[h],
                "shape_vs_href": shape[h],
                "shape_h3_corrected": shape_h3[h],
            }
            for h in h_values
        },
        "max_shape_deviation_raw": max_dev,
        "end_to_end_tilt_raw": tilt,
        "max_shape_deviation_h3_corrected": max_dev_h3,
        "end_to_end_tilt_h3_corrected": tilt_h3,
        "verdict": (
            "PASS: h3-corrected discrete/continuous ratio is h-flat (Option-A cancellation holds)"
            if max_dev_h3 < 0.01
            else "MARGINAL: 1-5% residual shape after h3 correction — quantify posterior impact"
            if max_dev_h3 < 0.05
            else "FAIL: large residual h-dependence after h3 correction — 'global' mode "
            "in-catalogue channel is distorted (local modes immune: no Sigma_global)"
        ),
    }

    out = Path(output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    _LOGGER.info("Wrote %s", out)

    print(
        f"\n{'h':>8} {'beta_G (cont)':>16} {'Sigma_global':>16} {'ratio':>12} "
        f"{'shape':>10} {'shape/h^3':>10}"
    )
    for h in h_values:
        print(
            f"{h:>8.4f} {beta_g[h]:>16.6e} {sigma_global[h]:>16.6e} "
            f"{ratio[h]:>12.6e} {shape[h]:>10.6f} {shape_h3[h]:>10.6f}"
        )
    print(f"\nraw:          max |shape - 1| = {max_dev:.4%}   end-to-end tilt = {tilt:+.4%}")
    print(f"h3-corrected: max |shape - 1| = {max_dev_h3:.4%}   end-to-end tilt = {tilt_h3:+.4%}")
    print(f"VERDICT: {result['verdict']}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--injections_dir", required=True)
    parser.add_argument("--h_min", type=float, default=0.60)
    parser.add_argument("--h_max", type=float, default=0.86)
    parser.add_argument("--h_steps", type=int, default=14)
    parser.add_argument("--n_sky_bands", type=int, default=6)
    parser.add_argument("--output_json", default=".planning/gate/G1_beta_g_check.json")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    h_values = [round(float(h), 4) for h in np.linspace(args.h_min, args.h_max, args.h_steps)]
    run_check(args.injections_dir, h_values, args.n_sky_bands, args.output_json)


if __name__ == "__main__":
    main()
