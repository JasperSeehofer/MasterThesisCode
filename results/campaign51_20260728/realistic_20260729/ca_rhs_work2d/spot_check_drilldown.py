r"""Follow-up drill-down: isolate WHICH stage of the erf-rule pipeline disagrees with the
brute-force arbiter -- the exact mass-marginal formula (_mass_marginal_survival) itself,
or the GL(50)-node z-quadrature that wraps it (the [PA-2D-2] validation JSON only tested
the mass-marginal at a SINGLE fixed z per row, never the z-integrated quantity).

Test (e): use the SAME exact mass-marginal formula (imported, not reimplemented) but
evaluate it on a FINE z-grid (same 1000-point grid as the brute-force arbiter) and
integrate over z with Simpson instead of GL(50). If (e) ~= (d) to <=1e-8 while (a) [full
GL(50)+exact-M pipeline] disagrees with (d) at the ~1e-4 level seen in
spot_check_adjudication.json, the defect is specifically GL(50) under-resolving the
z-kernel window (the mass-marginal formula itself is vindicated). If (e) still disagrees
with (d), the exact mass-marginal formula has a problem in this z-integration context.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import integrate
from scipy.stats import norm

THIS_DIR = Path(__file__).resolve().parent
CAMPAIGN_DIR = THIS_DIR.parent
sys.path.insert(0, str(CAMPAIGN_DIR))

import p3_2d_companion as comp  # noqa: E402
import spot_check_adjudication as sca  # noqa: E402

from darksiren_emri.bayesian_inference.bayesian_statistics import (  # noqa: E402
    _eddington_shifted_host_mass_batch,
)
from darksiren_emri.physical_relations import dist_vectorized  # noqa: E402
from darksiren_emri.validation import correspondence_1d as c1d  # noqa: E402

OUT_PATH = THIS_DIR / "spot_check_drilldown.json"
TARGET_SPOT_ROWS = [64, 65, 70]


def _exact_M_fine_z_value(
    z_g: float,
    z_err_raw: float,
    phiS: float,
    qS: float,
    mu: float,
    sigma: float,
    completeness,
    detection_probability,
    h: float,
    n_z: int = 1000,
) -> float:
    z_err_eff = float(c1d.host_z_error_eff(np.array([z_g]), np.array([z_err_raw]))[0])
    lower, upper = c1d._host_kernel_window(np.array([z_g]), np.array([z_err_eff]))
    lower, upper = float(lower[0]), float(upper[0])
    pixel = c1d._host_pixels(completeness, np.array([phiS]), np.array([qS]))

    z_grid = np.linspace(lower, upper, n_z)
    d_L_grid = np.asarray(dist_vectorized(z_grid, h=h), dtype=np.float64)

    mu_b = np.full((1, n_z), mu, dtype=np.float64)
    sigma_b = np.full((1, n_z), sigma, dtype=np.float64)
    s_bar_4d = comp._mass_marginal_survival(
        mu_b,
        sigma_b,
        z_grid[None, :],
        d_L_grid[None, :],
        np.array([[phiS]]),
        np.array([[qS]]),
        detection_probability,
        h,
    )[0]

    w_pop = c1d._kernel_w_pop_eff(z_grid[None, :], completeness, pixel, h)[0, :]
    gauss_z = norm.pdf(z_grid, loc=z_g, scale=z_err_eff)

    num = float(integrate.simpson(gauss_z * w_pop * s_bar_4d, x=z_grid))
    den = float(integrate.simpson(gauss_z * w_pop, x=z_grid))
    return num / den if den > 0.0 else 0.0


def main() -> None:
    if OUT_PATH.is_file():
        raise SystemExit(f"REFUSED: {OUT_PATH} already exists.")

    t0 = time.time()
    completeness, phi_survival_table, detection_probability = c1d.build_b0i_2d_selection_objects(
        h_true=sca.H
    )
    sca.DETECTION_PROBABILITY = detection_probability
    print(f"[t={time.time() - t0:.1f}s] selection objects built", file=sys.stderr)

    cat = comp._load_eligible_catalogue(completeness, phi_survival_table, sca.H)
    n_g = cat["z"].size
    print(f"[t={time.time() - t0:.1f}s] catalogue loaded n_g={n_g}", file=sys.stderr)

    spot_idx = sca._reconstruct_spot_indices(n_g)
    target_gi = np.array(
        [int(spot_idx[spot_row - 1]) for spot_row in TARGET_SPOT_ROWS], dtype=np.int64
    )
    M_eff_targets = _eddington_shifted_host_mass_batch(
        cat["M"][target_gi], cat["M_error"][target_gi]
    )

    # Load the adjudication results for (a) GL50-full-pipeline and (d) brute arbiter.
    adj_path = THIS_DIR / "spot_check_adjudication.json"
    adj = json.loads(adj_path.read_text())
    adj_by_row = {r["spot_row"]: r for r in adj["rows"]}

    rows_out = []
    for k, spot_row in enumerate(TARGET_SPOT_ROWS):
        gi = int(target_gi[k])
        z_g = float(cat["z"][gi])
        z_err_raw = float(cat["z_error"][gi])
        phiS = float(cat["phiS"][gi])
        qS = float(cat["qS"][gi])
        mu = float(M_eff_targets[k])
        sigma = float(cat["M_error"][gi])

        val_e = _exact_M_fine_z_value(
            z_g, z_err_raw, phiS, qS, mu, sigma, completeness, detection_probability, sca.H
        )
        a_val = adj_by_row[spot_row]["a_erf_rule"]
        d_val = adj_by_row[spot_row]["d_brute_simpson"]
        rel_e_vs_d = abs(val_e - d_val) / abs(d_val)
        rel_e_vs_a = abs(val_e - a_val) / abs(a_val)
        print(
            f"row {spot_row}: a(GL50+exactM)={a_val:.10e} e(fineZ+exactM)={val_e:.10e} "
            f"d(brute)={d_val:.10e} | rel(e,d)={rel_e_vs_d:.3e} rel(e,a)={rel_e_vs_a:.3e}",
            file=sys.stderr,
        )
        rows_out.append(
            {
                "spot_row": spot_row,
                "a_gl50_exact_m": a_val,
                "e_finez_exact_m": val_e,
                "d_brute_simpson": d_val,
                "rel_dev_e_vs_d": rel_e_vs_d,
                "rel_dev_e_vs_a": rel_e_vs_a,
            }
        )

    max_e_vs_d = max(r["rel_dev_e_vs_d"] for r in rows_out)
    if max_e_vs_d <= 1e-6:
        diagnosis = "GL50-Z-QUADRATURE-UNDER-RESOLVED (mass-marginal erf formula vindicated)"
    else:
        diagnosis = "MASS-MARGINAL-ERF-FORMULA-ITSELF-DEFECTIVE"

    out = {"rows": rows_out, "max_rel_dev_e_vs_d": max_e_vs_d, "diagnosis": diagnosis}
    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"\ndiagnosis: {diagnosis}")


if __name__ == "__main__":
    main()
