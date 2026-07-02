"""Generate the PP / calibration plot and a coverage summary table.

Left panel  : PP-plot (empirical credible-level CDF vs nominal) for the FLAT
              (production-style) and VOLUME (corrected) in-catalogue numerator,
              from the clean single-host controlled experiment.
Right panel : coverage @ {50,68,90}% for the four full estimators (A, B_local,
              B_exact, B_naive) from the completion-term run, vs the nominal
              diagonal.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import clean_singlehost_test as cst
import pp_coverage_test as m

HERE = Path(__file__).resolve().parent


def pp_value(h_grid, post, h_true):
    """HPD credible level at which h_true sits on the region boundary.

    = fraction of posterior mass with density >= density(h_true).
    Calibrated -> these values are Uniform(0,1) over realizations.
    """
    dh = np.gradient(h_grid)
    p_true = float(np.interp(h_true, h_grid, post))
    return float(np.sum((post >= p_true) * post * dh))


def clean_pp_values(h_true, h_grid, n_real, seed):
    """Re-run the clean single-host experiment, returning pp-values per realization."""
    zint = np.linspace(m.Z_MIN, m.Z_MAX_POP, 3000)
    wpop_i = m.wpop_of_z(zint)
    Dh = np.trapezoid(m.p_det(m.A_of_z(zint)[:, None] / h_grid[None, :]) * wpop_i[:, None],
                      zint, axis=0)
    logDh = np.log(Dh)
    master = np.random.default_rng(seed)
    pp = {"A": [], "V": []}
    sigma_z, sigma_dl_frac, n_events = 0.035, 0.05, 250
    for _ in range(n_real):
        rng = np.random.default_rng(master.integers(1 << 62))
        z_host = m.sample_from_density(m.detected_pop_density(h_true), m.Z_MIN, m.Z_MAX_POP, n_events, rng)
        dL_host = m.A_of_z(z_host) / h_true
        dL_obs = np.clip(dL_host + rng.normal(0.0, sigma_dl_frac * dL_host), 1e-3, None)
        sig_dl = sigma_dl_frac * dL_obs
        z_g = np.clip(z_host + rng.normal(0.0, sigma_z, n_events), m.Z_MIN, None)
        logL = {"A": np.zeros(h_grid.size), "V": np.zeros(h_grid.size)}
        for i in range(n_events):
            z_lo = max(m.Z_MIN, float(m.z_from_dL(dL_obs[i] - 5 * sig_dl[i], h_grid.min())) - 4 * sigma_z)
            z_hi = min(m._ZG[-1], float(m.z_from_dL(dL_obs[i] + 5 * sig_dl[i], h_grid.max())) + 4 * sigma_z)
            zq = np.linspace(z_lo, z_hi, 160)
            wq = np.gradient(zq)
            dLg = m.A_of_z(zq)[:, None] / h_grid[None, :]
            pGW = m.norm_pdf(dLg, dL_obs[i], sig_dl[i])
            K = m.norm_pdf(zq, z_g[i], sigma_z)
            Kw = K * m.wpop_of_z(zq)
            Kw = Kw / max(np.trapezoid(Kw, zq), 1e-300)
            logL["A"] += np.log(np.clip(np.einsum("z,zh,z->h", wq, pGW, K), 1e-300, None)) - logDh
            logL["V"] += np.log(np.clip(np.einsum("z,zh,z->h", wq, pGW, Kw), 1e-300, None)) - logDh
        for key in ("A", "V"):
            post = m.posterior_from_logL(logL[key], h_grid)
            pp[key].append(pp_value(h_grid, post, h_true))
    return {k: np.array(v) for k, v in pp.items()}


def main():
    h_grid = np.arange(0.600, 0.8601, 0.004)
    H_TRUE = 0.72
    N = 250
    print("computing PP-values (clean single-host)...")
    pp = clean_pp_values(H_TRUE, h_grid, n_real=N, seed=99)

    grid = np.linspace(0, 1, 200)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.4))

    # ---- Panel 1: PP-plot ----
    ax1.plot([0, 1], [0, 1], "k--", lw=1, label="calibrated (diagonal)")
    for key, lab, col in (("A", "FLAT numerator (production)", "C3"),
                          ("V", "VOLUME-weighted numerator (fix)", "C0")):
        ecdf = np.array([np.mean(pp[key] <= x) for x in grid])
        ax1.plot(grid, ecdf, color=col, lw=2, label=lab)
    # 1-sigma binomial band around the diagonal
    band = np.sqrt(grid * (1 - grid) / N)
    ax1.fill_between(grid, grid - band, grid + band, color="k", alpha=0.10,
                     label=r"$\pm1\sigma$ (N=%d)" % N)
    ax1.set_xlabel("nominal credible level  X")
    ax1.set_ylabel("empirical coverage  (fraction with $H_0^{true}$ in CI)")
    ax1.set_title("PP-plot: in-catalogue numerator z-prior\n(clean single-host, f=1, %d realizations)" % N)
    ax1.legend(loc="upper left", fontsize=8)
    ax1.set_xlim(0, 1); ax1.set_ylim(0, 1); ax1.set_aspect("equal")

    # ---- Panel 2: full-estimator coverage points ----
    res_path = HERE / "coverage_results.json"
    ax2.plot([0, 1], [0, 1], "k--", lw=1, label="calibrated")
    if res_path.exists():
        res = json.loads(res_path.read_text())["primary"]
        nom = np.array([0.50, 0.68, 0.90])
        styles = {"A_prod": ("C3", "o", "A_prod: production bare-Gaussian numerator"),
                  "B_corr": ("C0", "^", "B_corr: volume-prior deconvolution (fix)"),
                  "B_naive": ("C2", "v", "B_naive: dVc mult, no renorm"),
                  "A_global": ("0.5", "x", "A_global: literal global denom (fragile)")}
        for key, (col, mk, lab) in styles.items():
            if key not in res:
                continue
            cov = np.array([res[key]["coverage"][s] for s in ("50", "68", "90")])
            ax2.plot(nom, cov, color=col, marker=mk, lw=1.5, ms=8, label=lab)
    ax2.set_xlabel("nominal credible level  X")
    ax2.set_ylabel("empirical coverage")
    ax2.set_title("Full estimators with completion term\n($f(z)$ incomplete, interlopers, $H_0^{true}=0.72$)")
    ax2.legend(loc="upper left", fontsize=7.5)
    ax2.set_xlim(0.4, 1.0); ax2.set_ylim(0, 1.0)

    fig.tight_layout()
    out = HERE / "pp_coverage_plot.png"
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")

    # also dump the clean pp summary
    summ = {}
    for key in ("A", "V"):
        summ[key] = {X: float(np.mean(pp[key] <= X / 100)) for X in (50, 68, 90)}
        summ[key]["frac_pp_below_0.5"] = float(np.mean(pp[key] <= 0.5))
    (HERE / "clean_pp_summary.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
