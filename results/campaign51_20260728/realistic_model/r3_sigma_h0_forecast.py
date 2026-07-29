"""R3 — sigma_H0 forecast per realistic-model option (campaign #53 pre-registration).

Supporting numerical check for docs/derivations/realistic_host_observation_model.md
(RATIFY-R6/R8: option comparison + pre-registered predictions).

Per-event Fisher information on h from a single in-catalogue event with a
GW-measured d_L and a host redshift known to width sigma_z:

    d_L(z, h) = (c/H0) * g(z)  with g h-independent (flat LCDM, Omega_m fixed)
    => delta h / h = -delta d_L/d_L + (dln d_L/dz) * delta z
    => I_e = (h * sigma_eff)^-2,
       sigma_eff^2 = (sigma_dL/d_L)^2 + (dln d_L/dz)^2 * sigma_z^2 .

Inputs: the 76 aligned information-carrying hosts of seed 61000
(results/campaign51_20260728/idealization_audit/incat_hosts_seed61000.csv:
d_L, CRB var(d_L), host_z, host_zerr, host_flag) and the z-resolved
rate-weighted spectroscopic host fraction f_spec(z) measured by R1
(r1_flag_fractions.json, z_shells). dln d_L/dz is the exact numerical
derivative of the repo's dist().

Scenarios (labels match IDEALIZATION_LEDGER.md section 3 + the derivation's
option letters):

  A_pipeline_as_run   sigma_z = 0                             (cross-check: 0.027-0.032)
  opt_a_catalogue     sigma_z = host_zerr (stored total; all 76 photo-z)
  B_all_spec_150      sigma_z = sqrt(0.0017^2 + ((1+z)*150/c)^2)  (ledger B)
  C_all_spec_500      sigma_z = sqrt(0.0017^2 + ((1+z)*500/c)^2)  (ledger C)
  opt_b_spec_only     expected info = sum_e f_spec(z_e) * I_e^spec (photo hosts
                      carry no in-catalogue term); 150 and 500 km/s variants
  opt_c_hybrid        sum_e [f_spec*I_e^spec + (1-f_spec)*I_e^photo(own width)]

ASSUMED for opt_b/opt_c: the golden-event geometry (z, d_L, sigma_dL
distribution) is host-flag independent — the rate-weighted draw is mass/z
weighted, not flag weighted, so this holds to first order; flag correlates
with brightness, so treat as an expectation, not a per-seed prediction.

Run:  uv run python results/campaign51_20260728/realistic_model/r3_sigma_h0_forecast.py
Output: r3_forecast.json.
"""

import json
import pathlib

import numpy as np
import pandas as pd

from master_thesis_code.constants import SPEED_OF_LIGHT_KM_S
from master_thesis_code.physical_relations import dist

HERE = pathlib.Path(__file__).parent
HOSTS = HERE.parent / "idealization_audit" / "incat_hosts_seed61000.csv"
R1 = HERE / "r1_flag_fractions.json"
OUT = HERE / "r3_forecast.json"

H_TRUE = 0.73
SIGMA_Z_SPEC_MEAS = 0.0017  # GLADE+ spectroscopic measurement width (Dalya et al. 2022)


def dln_dl_dz(z: np.ndarray, h: float) -> np.ndarray:
    dz = 1e-5
    d0 = np.array([dist(float(zi), h=h) for zi in z])
    d1 = np.array([dist(float(zi) + dz, h=h) for zi in z])
    return (d1 - d0) / dz / d0


def f_spec_of_z(z: np.ndarray, shells: list[dict]) -> np.ndarray:
    """Piecewise-constant rate-weighted spec fraction from the R1 z shells."""
    out = np.zeros_like(z)
    for s in shells:
        m = (z >= s["z_lo"]) & (z < s["z_hi"])
        out[m] = s["rate_weighted_frac_spec"] or 0.0
    return out


def combined_sigma_h0(info: np.ndarray) -> float:
    return 100.0 * H_TRUE / np.sqrt(float(info.sum())) / H_TRUE  # = 100/sqrt(I_tot)


def main() -> None:
    df = pd.read_csv(HOSTS)
    shells = json.loads(R1.read_text())["z_shells"]

    z = df["host_z"].to_numpy(dtype=np.float64)
    zerr = df["host_zerr"].to_numpy(dtype=np.float64)
    d_L = df["luminosity_distance"].to_numpy(dtype=np.float64)
    sigma_dl = np.sqrt(
        df["delta_luminosity_distance_delta_luminosity_distance"].to_numpy(dtype=np.float64)
    )
    frac_dl = sigma_dl / d_L
    dlnD = dln_dl_dz(z, H_TRUE)
    fspec = f_spec_of_z(z, shells)

    def info(sigma_z: np.ndarray | float) -> np.ndarray:
        sigma_eff2 = frac_dl**2 + (dlnD * np.asarray(sigma_z)) ** 2
        return 1.0 / (H_TRUE**2 * sigma_eff2)

    sig_spec_150 = np.sqrt(
        SIGMA_Z_SPEC_MEAS**2 + ((1.0 + z) * 150.0 / SPEED_OF_LIGHT_KM_S) ** 2
    )
    sig_spec_500 = np.sqrt(
        SIGMA_Z_SPEC_MEAS**2 + ((1.0 + z) * 500.0 / SPEED_OF_LIGHT_KM_S) ** 2
    )

    scenarios: dict[str, np.ndarray] = {
        "A_pipeline_as_run": info(0.0),
        "opt_a_catalogue": info(zerr),
        "B_all_spec_150": info(sig_spec_150),
        "C_all_spec_500": info(sig_spec_500),
        "opt_b_spec_only_150": fspec * info(sig_spec_150),
        "opt_b_spec_only_500": fspec * info(sig_spec_500),
        "opt_c_hybrid_150": fspec * info(sig_spec_150) + (1.0 - fspec) * info(zerr),
        "opt_c_hybrid_500": fspec * info(sig_spec_500) + (1.0 - fspec) * info(zerr),
    }

    results = {
        "n_hosts": int(len(df)),
        "expected_n_spec_hosts": float(fspec.sum()),
        "f_spec_top3_loudest": [float(x) for x in fspec[np.argsort(-df["SNR"].to_numpy())[:3]]],
        "sigma_H0_km_s_Mpc": {},
    }
    for name, i_e in scenarios.items():
        sigma = 100.0 / np.sqrt(float(i_e.sum()))
        results["sigma_H0_km_s_Mpc"][name] = sigma
        print(f"{name:>24s}: sigma_H0 = {sigma:.3f} km/s/Mpc   (I_tot = {i_e.sum():.1f})")
    print(f"expected number of spec-z golden hosts: {results['expected_n_spec_hosts']:.1f} of 76")

    OUT.write_text(json.dumps(results, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
