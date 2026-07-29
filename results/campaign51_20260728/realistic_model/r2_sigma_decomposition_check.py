"""R2 — realized-sigma vs kernel-sigma consistency check (counted-once invariant).

Supporting numerical check for docs/derivations/realistic_host_observation_model.md
(RATIFY-R2: the realized observation noise must equal the inference kernel width
EXACTLY, per row — no double counting, no missing component).

Establishes three facts from the production reduced catalogue
(post-#40b schema; z_error = TOTAL sigma: GLADE+ measurement (+) catalogue BORG
PV error (+) per-class PV term folded at parse time, handler.py:344-389):

  F1  The stored total obeys the parse-time floor z_error >= (1+z)*150 km/s / c
      for (almost) every row — a schema-integrity check that the #40b fold is
      present in the CSV actually on disk (violations => stale catalogue).
  F2  The PV class split (150 vs 500 km/s) is NOT recoverable from the reduced
      CSV (the PV-correction flag and the BORG sigma_tot column are dropped at
      parse): the fraction of rows where z_error < (1+z)*500/c (which are
      PROVABLY corrected-class) is reported; all other rows are ambiguous.
      Consequence: the realization must draw ONE total N(0, z_error) rather than
      re-deriving components (RATIFY-R2 recommendation).
  F3  MC identity: component-wise realization (any split sigma_a^2 + sigma_b^2 =
      z_error^2) and single-total realization produce the same law; sample std of
      1e6 draws matches z_error to MC precision for representative rows — i.e.
      sigma_realized = sigma_kernel holds by construction when both read the
      same stored column.

Run:  uv run python results/campaign51_20260728/realistic_model/r2_sigma_decomposition_check.py
Output: r2_sigma_decomposition.json. Runtime ~2 min.
"""

import json
import pathlib

import numpy as np
import pandas as pd

from master_thesis_code.constants import (
    SIGMA_V_PV_RESIDUAL_CORRECTED_KM_S,
    SIGMA_V_PV_UNCORRECTED_KM_S,
    SPEED_OF_LIGHT_KM_S,
)

C_KM_S = SPEED_OF_LIGHT_KM_S

CATALOGUE = pathlib.Path("master_thesis_code/galaxy_catalogue/reduced_galaxy_catalogue.csv")
OUT = pathlib.Path(__file__).parent / "r2_sigma_decomposition.json"


def main() -> None:
    df = pd.read_csv(
        CATALOGUE,
        header=None,
        usecols=[3, 4, 7],
        names=["z", "z_error", "flag"],
        dtype=np.float64,
    )

    # Drop rows with missing z/z_error (present in the raw reduced CSV; the
    # production loader prunes them via the mass/z masks downstream).
    n_raw = len(df)
    df = df.dropna(subset=["z", "z_error"]).reset_index(drop=True)
    print(f"dropped {n_raw - len(df)} rows with NaN z/z_error out of {n_raw}")

    one_plus_z = 1.0 + df["z"].to_numpy()
    sigma_pv_150 = one_plus_z * SIGMA_V_PV_RESIDUAL_CORRECTED_KM_S / C_KM_S
    sigma_pv_500 = one_plus_z * SIGMA_V_PV_UNCORRECTED_KM_S / C_KM_S
    z_err = df["z_error"].to_numpy()
    flag = df["flag"].to_numpy()

    # F1: floor check (total must contain at least the smaller class term).
    viol_150 = int(np.count_nonzero(z_err < sigma_pv_150 * (1.0 - 1e-12)))
    # F2: rows provably corrected-class (total below the 500 km/s term alone).
    provably_corrected = z_err < sigma_pv_500
    n_provably_corrected = int(np.count_nonzero(provably_corrected))
    n_provably_corrected_spec = int(np.count_nonzero(provably_corrected & (flag == 3)))
    n_spec = int(np.count_nonzero(flag == 3))
    n = len(df)

    # F3: MC identity on 5 representative rows (min/median/max spec + median photo).
    rng = np.random.default_rng(53_2026)
    rows = []
    spec_idx = np.flatnonzero(flag == 3)
    photo_idx = np.flatnonzero(flag == 1)
    picks = {
        "spec_min_sigma": spec_idx[np.argmin(z_err[spec_idx])],
        "spec_median_sigma": spec_idx[np.argsort(z_err[spec_idx])[len(spec_idx) // 2]],
        "photo_median_sigma": photo_idx[np.argsort(z_err[photo_idx])[len(photo_idx) // 2]],
    }
    n_mc = 1_000_000
    for label, i in picks.items():
        sigma_tot = z_err[i]
        # single-total draw
        std_total = float(np.std(rng.normal(0.0, sigma_tot, n_mc)))
        # component-wise draw with the class-PV split assumed at its floor value
        sigma_pv = min(sigma_pv_150[i], sigma_tot)
        sigma_rest = float(np.sqrt(max(sigma_tot**2 - sigma_pv**2, 0.0)))
        std_comp = float(
            np.std(rng.normal(0.0, sigma_pv, n_mc) + rng.normal(0.0, sigma_rest, n_mc))
        )
        rows.append(
            {
                "row": label,
                "z": float(df["z"].iloc[i]),
                "flag": int(flag[i]),
                "sigma_kernel(z_error)": float(sigma_tot),
                "mc_std_single_total": std_total,
                "mc_std_componentwise": std_comp,
                "rel_err_single": abs(std_total / sigma_tot - 1.0),
                "rel_err_componentwise": abs(std_comp / sigma_tot - 1.0),
            }
        )
        print(rows[-1])

    results = {
        "n_rows": n,
        "F1_floor_violations_150kms": viol_150,
        "F1_pass": viol_150 == 0,
        "F2_n_provably_corrected_class": n_provably_corrected,
        "F2_frac_provably_corrected": n_provably_corrected / n,
        "F2_n_spec_rows": n_spec,
        "F2_n_spec_provably_corrected": n_provably_corrected_spec,
        "F2_note": (
            "rows with z_error >= (1+z)*500km/s/c are ambiguous between "
            "{uncorrected} and {corrected with large measurement error}; the PV "
            "class split is irrecoverable from the reduced CSV"
        ),
        "F3_mc_rows": rows,
        "F3_pass": all(r["rel_err_single"] < 5e-3 and r["rel_err_componentwise"] < 5e-3 for r in rows),
    }
    OUT.write_text(json.dumps(results, indent=2))
    print(
        f"F1 floor violations: {viol_150}/{n};  F2 provably-corrected fraction: "
        f"{n_provably_corrected / n:.4f} (spec rows: {n_provably_corrected_spec}/{n_spec});  "
        f"F3 pass: {results['F3_pass']}"
    )
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
