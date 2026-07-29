"""R1 — spec-z vs photo-z host fractions in the pruned catalogue at campaign depth.

Supporting numerical check for docs/derivations/realistic_host_observation_model.md
(campaign #53 realistic host-observation model, RATIFY-R5/R6 evidence).

Measures, from the production reduced catalogue
(master_thesis_code/galaxy_catalogue/reduced_galaxy_catalogue.csv, post-#40b schema:
[RA, Dec, B_mag, z, z_error, M_star(1e10 Msun), M_star_err, REDSHIFT_FLAG]):

  - the count fraction and the EMRI-rate-weighted fraction of SPECTROSCOPIC
    (flag 3) vs PHOTOMETRIC (flag 1) rows, after replicating the production
    pruning (_map_stellar_masses_to_BH_masses -> _remove_galaxies_without_mass
    -> _get_pruned_galaxy_catalog with [M_SOURCE_FRAME_MIN, M_SOURCE_FRAME_MAX],
    z_max = 1.5), at several z cuts including the information-carrying venue
    z < 0.15 (IDEALIZATION_LEDGER.md section 1);
  - per-flag z_error statistics (the kernel widths a per-flag analysis realizes).

The rate weight is the production host-draw weight
w(g) = R_eff_per_mbh(M_BH) / (1 + z) (handler.draw_rate_weighted_hosts), so the
weighted fraction IS the probability that a drawn in-catalogue host is spec-z.

Run:  uv run python results/campaign51_20260728/realistic_model/r1_catalog_flag_fractions.py
Output: r1_flag_fractions.json (this directory). Runtime ~2 min, ~2.5 GB RAM.
"""

import json
import pathlib

import numpy as np
import pandas as pd

from master_thesis_code.constants import M_SOURCE_FRAME_MAX, M_SOURCE_FRAME_MIN
from master_thesis_code.emri_rate import R_eff_per_mbh
from master_thesis_code.galaxy_catalogue.handler import (
    _empiric_stellar_mass_to_BH_mass_relation,
)

CATALOGUE = pathlib.Path("master_thesis_code/galaxy_catalogue/reduced_galaxy_catalogue.csv")
OUT = pathlib.Path(__file__).parent / "r1_flag_fractions.json"

Z_CUTS = [0.05, 0.10, 0.15, 0.30, 0.50, 1.50]


def main() -> None:
    # Headerless 8-column reduced catalogue; positional usecols per
    # _reduced_catalog_column_names(): 3=z, 4=z_error, 5=M_star, 6=M_star_err, 7=flag.
    df = pd.read_csv(
        CATALOGUE,
        header=None,
        usecols=[3, 4, 5, 6, 7],
        names=["z", "z_error", "mstar", "mstar_err", "flag"],
        dtype={
            "z": np.float64,
            "z_error": np.float64,
            "mstar": np.float64,
            "mstar_err": np.float64,
            "flag": np.float64,  # tolerate legacy "1.0"/"3.0" round-trips
        },
    )
    n_total = len(df)

    # Replicate the production load path (handler.__init__ order):
    # 1. stellar -> BH mass via the exact repo relation (RV2015; M_star in 1e10 Msun).
    df = df[~df["mstar"].isna()].copy()
    bh_mass, bh_mass_err = _empiric_stellar_mass_to_BH_mass_relation(
        df["mstar"].to_numpy(), df["mstar_err"].fillna(0.0).to_numpy()
    )
    df["M_BH"] = bh_mass
    df["M_BH_err"] = bh_mass_err
    # 2. _get_pruned_galaxy_catalog: error-inflated mass window + z window.
    pruned = df[
        (df["M_BH"] + df["M_BH_err"] >= M_SOURCE_FRAME_MIN)
        & (df["M_BH"] - df["M_BH_err"] <= M_SOURCE_FRAME_MAX)
        & (df["z"] - df["z_error"] <= 1.5)
    ].copy()

    # Production host-draw rate weight (draw_rate_weighted_hosts).
    pruned["w"] = np.asarray(R_eff_per_mbh(pruned["M_BH"].to_numpy())) / (1.0 + pruned["z"])

    results: dict = {
        "catalogue": str(CATALOGUE),
        "n_rows_reduced_csv": int(n_total),
        "n_rows_pruned": int(len(pruned)),
        "mass_window_source_frame": [M_SOURCE_FRAME_MIN, M_SOURCE_FRAME_MAX],
        "z_cuts": {},
    }

    for z_cut in Z_CUTS:
        sel = pruned[pruned["z"] < z_cut]
        spec = sel[sel["flag"] == 3]
        photo = sel[sel["flag"] == 1]
        w_tot = float(sel["w"].sum())
        entry = {
            "n": int(len(sel)),
            "n_spec": int(len(spec)),
            "n_photo": int(len(photo)),
            "count_frac_spec": float(len(spec) / len(sel)) if len(sel) else None,
            "rate_weighted_frac_spec": float(spec["w"].sum() / w_tot) if w_tot > 0 else None,
            "spec_z_error": {
                "median": float(spec["z_error"].median()) if len(spec) else None,
                "p16": float(spec["z_error"].quantile(0.16)) if len(spec) else None,
                "p84": float(spec["z_error"].quantile(0.84)) if len(spec) else None,
            },
            "photo_z_error": {
                "median": float(photo["z_error"].median()) if len(photo) else None,
                "p16": float(photo["z_error"].quantile(0.16)) if len(photo) else None,
                "p84": float(photo["z_error"].quantile(0.84)) if len(photo) else None,
            },
            "spec_median_z": float(spec["z"].median()) if len(spec) else None,
            "photo_median_z": float(photo["z"].median()) if len(photo) else None,
        }
        results["z_cuts"][f"z<{z_cut}"] = entry
        print(
            f"z<{z_cut}: n={entry['n']:>9d}  spec count-frac={entry['count_frac_spec']:.4f}"
            f"  spec rate-weighted-frac={entry['rate_weighted_frac_spec']:.4f}"
            f"  median sigma_z spec={entry['spec_z_error']['median']}"
            f" photo={entry['photo_z_error']['median']}"
        )

    # Differential z shells (for a z-resolved f_spec(z) in the R3 forecast):
    # rate-weighted P(flag=3 | host drawn in shell).
    shell_edges = [0.0, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30]
    results["z_shells"] = []
    for lo, hi in zip(shell_edges[:-1], shell_edges[1:]):
        sel = pruned[(pruned["z"] >= lo) & (pruned["z"] < hi)]
        w_tot = float(sel["w"].sum())
        w_spec = float(sel.loc[sel["flag"] == 3, "w"].sum())
        results["z_shells"].append(
            {
                "z_lo": lo,
                "z_hi": hi,
                "n": int(len(sel)),
                "rate_weighted_frac_spec": (w_spec / w_tot) if w_tot > 0 else None,
            }
        )
        print(f"shell [{lo}, {hi}): rate-weighted f_spec = {w_spec / w_tot if w_tot else float('nan'):.4f}")

    OUT.write_text(json.dumps(results, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
