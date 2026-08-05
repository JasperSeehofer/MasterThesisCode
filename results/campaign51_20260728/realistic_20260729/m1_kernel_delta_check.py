"""M-1 check: is the iiib venue's host-z kernel a delta function, or does it carry finite width?

Spec: results/campaign51_20260728/realistic_20260729/
      CLAIM_HITCHHIKER_INDEPENDENCE_20260805.md.DRAFT, section 6, M-1.

Read-only. No source edits. Writes m1_kernel_delta_check.json alongside this script.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
STAGED = HERE / "realizations_staged"
PARENT_CSV = STAGED / "cluster_parent_reduced_galaxy_catalogue.csv"
OBSERVED_CSV = STAGED / "observed_catalogue_seed900001.csv"

# Column order on disk (headerless), per galaxy_catalogue/handler.py
# _reduced_catalog_column_names(): RA, DEC, APPARENT_B_MAG, REDSHIFT,
# REDSHIFT_MEASUREMENT_ERROR, STELLAR_MASS, STELLAR_MASS_ABSOULTE_ERROR, REDSHIFT_FLAG.
COL_NAMES = ["ra", "dec", "b_mag", "z", "z_error", "stellar_mass", "stellar_mass_error", "flag"]
USECOLS = [3, 4]  # z, z_error


def column_stats(path: Path) -> dict:
    z_vals = []
    zerr_vals = []
    chunksize = 2_000_000
    for chunk in pd.read_csv(
        path,
        header=None,
        names=COL_NAMES,
        usecols=USECOLS,
        chunksize=chunksize,
        dtype={"z": np.float64, "z_error": np.float64},
    ):
        z_vals.append(chunk["z"].to_numpy())
        zerr_vals.append(chunk["z_error"].to_numpy())
    z = np.concatenate(z_vals)
    zerr = np.concatenate(zerr_vals)
    n = zerr.size
    n_nan = int(np.sum(np.isnan(zerr)))
    finite = ~np.isnan(zerr)
    zerr_f = zerr[finite]
    z_f = z[finite]
    n_f = zerr_f.size
    n_exact_zero = int(np.sum(zerr_f == 0.0))
    n_below_1e4 = int(np.sum(zerr_f < 1e-4))
    sigma_over_1pz = zerr_f / (1.0 + z_f)
    return {
        "n_rows": n,
        "n_nan_z_error": n_nan,
        "frac_nan_z_error": n_nan / n,
        "n_finite": n_f,
        "z_error_min": float(np.min(zerr_f)),
        "z_error_median": float(np.median(zerr_f)),
        "z_error_max": float(np.max(zerr_f)),
        "z_error_mean": float(np.mean(zerr_f)),
        "frac_exactly_zero": n_exact_zero / n_f,
        "frac_below_1e-4": n_below_1e4 / n_f,
        "sigma_over_1plusz_median": float(np.median(sigma_over_1pz)),
        "sigma_over_1plusz_min": float(np.min(sigma_over_1pz)),
        "sigma_over_1plusz_max": float(np.max(sigma_over_1pz)),
    }


def main() -> None:
    result: dict = {"spec": "M-1, CLAIM_HITCHHIKER_INDEPENDENCE_20260805.md.DRAFT sec 6"}

    # --- Step 1: code-path evidence (recorded manually from source read, not computed) ---
    result["code_path_evidence"] = {
        "resolve_host_z_kernel": (
            "master_thesis_code/bayesian_inference/bayesian_statistics.py:149-210. "
            "With host_z_kernel='volume_deconv' (explicit, not 'auto'), the function "
            "returns 'volume_deconv' unconditionally at line 195 "
            "(`resolved = host_z_kernel` branch of the ternary at :192-196) "
            "regardless of normalization_mode or catalogue_scattered -- 'point' "
            "(the delta kernel) is never selected."
        ),
        "iiib_cli_args": (
            "results/run_20260804_frozeng/iiib/run_metadata_0.json and "
            "results/run_20260805_d1/a1_iiib/run_metadata_0.json (same-day iiib arms; "
            "run_20260804_postfix/iiib itself has no run_metadata_0.json staged locally) "
            "both record host_z_kernel='volume_deconv', normalization_mode='absolute_marginal', "
            "observed_catalogue=null, realize_observed_catalogue=false -- i.e. the parent "
            "(unscattered) catalogue loaded directly, kernel explicitly volume_deconv."
        ),
        "sigma_source": (
            "bayesian_statistics.py:4662-4794 (single-host path) and :5182-5304 "
            "(vectorized path): host_z_error is read per-galaxy from the loaded "
            "catalogue (galaxy_catalogue/handler.py InternalCatalogColumns.REDSHIFT_ERROR "
            "= on-disk column REDSHIFT_MEASUREMENT_ERROR, handler.py:78/163/182), then "
            "combined in quadrature with a peculiar-velocity floor: "
            "sigma_z_pv = (1+z_g) * SIGMA_V_PEC_KM_S / c (bayesian_statistics.py:4765-4766, "
            "5259-5260); host_z_error_eff = sqrt(z_error**2 + sigma_z_pv**2). "
            "SIGMA_V_PEC_KM_S = 0.0 (constants.py:95, an inference-time residual knob, "
            "default off) so this particular floor is inert at iiib; the effective width "
            "is therefore just the catalogue's own REDSHIFT_MEASUREMENT_ERROR."
        ),
        "catalogue_parse_time_floor": (
            "The catalogue-level floor is baked in earlier, at parse time, NOT at "
            "inference time: galaxy_catalogue/handler.py:434-479 "
            "(parse_to_reduced_catalog) folds a peculiar-velocity term into "
            "REDSHIFT_MEASUREMENT_ERROR in quadrature for EVERY row -- "
            "sqrt(raw_z_error**2 + sigma_pv_catalogue**2 + sigma_z_pv_class**2), where "
            "sigma_z_pv_class = (1+z) * SIGMA_V_PV_UNCORRECTED_KM_S / c for PV-uncorrected "
            "rows (SIGMA_V_PV_UNCORRECTED_KM_S = 500.0 km/s, constants.py) or the smaller "
            "SIGMA_V_PV_RESIDUAL_CORRECTED_KM_S for BORG-corrected rows. This applies "
            "identically to the parent (unscattered) and any observed-realization "
            "catalogue, since both derive from the same parse. The parent catalogue's "
            "z_error column is therefore NOT zero by construction -- the PV floor alone "
            "gives (1+z)*500/c ~ 0.00167 at z~0."
        ),
    }

    # --- Step 2/3: data check ---
    result["parent_catalogue_path"] = str(PARENT_CSV)
    result["observed_catalogue_path"] = str(OBSERVED_CSV)
    result["parent_stats"] = column_stats(PARENT_CSV)
    result["observed_stats"] = column_stats(OBSERVED_CSV)

    parent_med_ratio = result["parent_stats"]["sigma_over_1plusz_median"]
    glade_photoz_scale = 0.035
    result["comparison"] = {
        "parent_sigma_over_1plusz_median": parent_med_ratio,
        "glade_photoz_scale_joint_venue": glade_photoz_scale,
        "ratio_parent_to_glade_photoz": parent_med_ratio / glade_photoz_scale,
        "paper_sec3_3_demo_deltaz_over_z_low": 0.003,
        "paper_sec3_3_demo_deltaz_over_z_high": 0.03,
        "ratio_parent_to_paper_demo_low": parent_med_ratio / 0.003,
        "ratio_parent_to_paper_demo_high": parent_med_ratio / 0.03,
    }

    # --- Step 4: verdict ---
    zerr = result["parent_stats"]
    is_delta = zerr["z_error_max"] == 0.0
    result["verdict"] = "KERNEL-δ" if is_delta else "KERNEL-FINITE"
    result["verdict_rationale"] = (
        "iiib's run_metadata pins host_z_kernel='volume_deconv' explicitly (never resolves "
        "to 'point'/delta), and the parent catalogue's REDSHIFT_MEASUREMENT_ERROR column "
        "is nonzero for essentially all rows (a parse-time peculiar-velocity floor is "
        "folded in regardless of catalogue realization) -- both the KERNEL CHOICE and the "
        "per-galaxy WIDTH INPUT are non-delta at iiib."
    )

    out_path = HERE / "m1_kernel_delta_check.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
