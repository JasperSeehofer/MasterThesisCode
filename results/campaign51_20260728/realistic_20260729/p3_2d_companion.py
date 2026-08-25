r"""[P3-2D] the Sigma~^4D companion pass (zero-``evaluate()``), item 1 of the pre-fleet
execution task (PREREGISTRATION_P3_2D_20260825.md, PA-2D-1 F4).

**What this computes.** ``Sigma~^4D = SUM_g w_g * S~_4D,g`` over the eligible catalogue, at
``h = H_GEN = H_TRUE = 0.73``, under the F2-resolved kernel (A20_REVIEW_P3_2D_DESIGN_20260825.md
Finding 2: the production ``gaussian`` branch, mass prior centred at the Eddington-shifted
``_host_M_eff``, catalogue_numerator_survival_2d_center="eff"). ``S~_4D,g`` is the SAME
two-stage object the coded with-BH numerator evaluates per event (see
``_mz_sel_2d_expectation``/``_mz_sel_2d_expectation_batch``,
``bayesian_statistics.py:5854-5976``) but with NO event conditioning: the population-level
mass-marginal ``S_bar_4D,g(z) = E_{M~N(M_eff_g, sigma_Mg^2)}[S_4D(d_L(z;h), M)]`` (Gauss-Hermite,
the SAME ``_MT_GH_NODES``/``_MT_GH_WEIGHTS`` order-24 nodes the mass_trunc/2D-twin kernel already
uses), z-kernel-smeared on the SAME window/Gauss-Legendre machinery
:func:`correspondence_1d.kernel_smeared_survival` uses (``_host_kernel_window``,
``_kernel_w_pop_eff``, ``host_z_error_eff``, the 50-node GL quadrature) -- imported, not
reimplemented, so this pass's z-kernel tracks whatever kernel family production's own numerator
uses today (PA-11/FATAL-1 precedent, ``A20_REVIEW_B0_IMPL_20260823.md`` Finding 1).

**Disclosed diff (the ONE new formula this script adds):** the coded ``_mz_sel_2d_expectation*``
leaves do not floor negative Gauss-Hermite mass nodes -- production never needs to, because a
per-EVENT product-Gaussian mean (``mu_star``) sits close to the GW-measured mass with a narrow
``sigma_star``, so nodes 24-sigma out are numerically irrelevant. This companion evaluates the
mass GH quadrature at the CATALOGUE's raw ``sigma_Mg``, which for some low-mass/high-relative-
error hosts extends into non-physical (M<=0) territory. F2 (the review) explicitly carries "the
MINOR-6 non-positive-mass guard (S:=0 at M_z<=0, ``:2322-2335`` pattern)" into the venue's latent
draw; this script applies the SAME S:=0 override to any GH node with ``a_node<=0`` before the
GH-weighted sum (nodes are clamped to a tiny positive epsilon for the interpolator CALL only, so
no ``log10``/interpolation warning fires; their contribution is then zeroed). This is the one
instrument-side addition beyond leaf reuse, disclosed per the task's "instrument-side bugfixes
with disclosed diffs are OK" allowance.

Sigma^phi / beta_G_phi / beta_Gbar_phi are the banked, seed-invariant, venue-independent Path-A
mixture scalars (F3/F4: constant across every b0/b0i/b0i2d CSV at fixed h + flags) -- read from
an EXISTING banked event_likelihoods.csv (any b0i-family seed already on disk; zero-compute) via
:func:`p3_b0_identity_test._beta_g_phi_and_gbar`, and Sigma_phi from
:func:`p3_b0_identity_test.mass_companion` (already a cached zero-compute leaf).

Output: ``C2_star = beta_G_phi * Sigma_tilde_4D / (Sigma_phi * beta_Gbar_phi)`` (F4), banked to
``ca_rhs_work2d/p3_2d_companion.json``, plus a 100-random-row independent scipy.integrate.quad
spot-check (task-mandated, <=1e-6 rel target).

PA-CA-11 (out-root guard, carried to every new instrument, F16): REFUSES if the output JSON
already exists -- purge or pick a fresh path.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import integrate
from scipy.stats import norm

REPO_ROOT = Path(__file__).resolve().parents[3]
THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))  # o5 (p3_b0_identity_test) is a sibling script, not a package

import p3_b0_identity_test as o5  # noqa: E402

from darksiren_emri.bayesian_inference.bayesian_statistics import (  # noqa: E402
    _MT_GH_NODES,
    _MT_GH_WEIGHTS,
    _eddington_shifted_host_mass_batch,
    _wbh_z_kwargs,
)
from darksiren_emri.emri_rate import R_eff_per_mbh  # noqa: E402
from darksiren_emri.galaxy_catalogue.handler import InternalCatalogColumns  # noqa: E402
from darksiren_emri.physical_relations import dist_vectorized  # noqa: E402
from darksiren_emri.validation import correspondence_1d as c1d  # noqa: E402

H_GEN: float = c1d.H_TRUE  # 0.73, prereg S1/"all reads at h=H_TRUE"
CENTER: str = "eff"  # F2: production configuration = eddington-shifted centering
CHUNK: int = 20_000  # mirrors o5._MASS_COMPANION_CHUNK's memory-bounding convention

# An existing, already-banked b0i-family event_likelihoods.csv -- alpha_G_phi/r_Malm/D_tilde_phi
# are venue-independent Path-A mixture globals at fixed (h, flags) (F3), so ANY already-run
# fused/phi-selection seed on disk supplies beta_G_phi/beta_Gbar_phi at zero extra compute.
_REFERENCE_CSV = (
    THIS_DIR
    / "p3_b0_work/bc_900101_work/seed900101/simulations/diagnostics/event_likelihoods.csv"
)

OUT_PATH = THIS_DIR / "ca_rhs_work2d/p3_2d_companion.json"

REGISTRATION_SECTION = (
    "results/campaign51_20260728/realistic_20260729/PREREGISTRATION_P3_2D_20260825.md "
    "sec1 (Sigma~^4D companion), PA-2D-1 F4 (C2* resolved)"
)


def _a22_stamp_2d() -> dict[str, Any]:
    """[ORCH] A22 = FIVE resolved flag values (F7) + git commit + dirty, BEFORE any scoring."""
    git_stamp = o5._a22_stamp()
    return {
        **git_stamp,
        "catalogue_global_selection": "phi",
        "catalogue_numerator_survival": "off",  # 1D twin production default (not this pass's axis)
        "selection_in_completion_numerator": "fused",
        "catalogue_numerator_survival_2d": "mz_sel",  # this pass computes the TWIN's own object
        "catalogue_numerator_survival_2d_center": CENTER,
        "acceptance_model_version": "F12-extended-class-G-replay",
    }


def _load_eligible_catalogue(
    completeness: Any, phi_survival_table: dict[float, Any], h: float
) -> dict[str, npt.NDArray[np.float64]]:
    """Byte-identical eligibility convention to :func:`p3_b0_identity_test.mass_companion`."""
    handler = c1d._load_galaxy_catalog_handler(c1d.REDUCED_CATALOGUE_PATH)
    catalog = handler.reduced_galaxy_catalog
    z_grid, _ = phi_survival_table[h]
    z_all = np.asarray(catalog[InternalCatalogColumns.REDSHIFT].to_numpy(dtype=np.float64))
    M_all = np.asarray(catalog[InternalCatalogColumns.BH_MASS].to_numpy(dtype=np.float64))
    z_err_all = np.asarray(
        catalog[InternalCatalogColumns.REDSHIFT_ERROR].to_numpy(dtype=np.float64)
    )
    M_err_all = np.asarray(
        catalog[InternalCatalogColumns.BH_MASS_ERROR].to_numpy(dtype=np.float64)
    )
    phiS_all = np.asarray(catalog[InternalCatalogColumns.PHI_S].to_numpy(dtype=np.float64))
    qS_all = np.asarray(catalog[InternalCatalogColumns.THETA_S].to_numpy(dtype=np.float64))

    z_max = float(z_grid.max())
    eligible = (z_all < z_max) & np.isfinite(M_all) & (M_all > 0.0)
    # Debug/smoke-test escape hatch ONLY (disclosed, never set in a banked run): truncates the
    # eligible index set to the first N rows so the pass can be sanity-checked in seconds instead
    # of ~1-2 CPU-h. Unset (default None) is the registered, full-catalogue banked path.
    import os

    _max_rows = os.environ.get("P3_2D_COMPANION_MAX_ROWS")
    if _max_rows is not None:
        eligible_idx = np.flatnonzero(eligible)[: int(_max_rows)]
        eligible = np.zeros_like(eligible)
        eligible[eligible_idx] = True
    return {
        "z": z_all[eligible],
        "M": M_all[eligible],
        "z_error": np.maximum(z_err_all[eligible], c1d.EXACT_Z_ERROR_FLOOR),
        "M_error": M_err_all[eligible],
        "phiS": phiS_all[eligible],
        "qS": qS_all[eligible],
    }


def _mass_marginal_survival(
    mu: npt.NDArray[np.float64],
    sigma: npt.NDArray[np.float64],
    z_nodes: npt.NDArray[np.float64],
    d_L_nodes: npt.NDArray[np.float64],
    host_phiS: npt.NDArray[np.float64],
    host_qS: npt.NDArray[np.float64],
    detection_probability: Any,
    h: float,
) -> npt.NDArray[np.float64]:
    """``E_M[S_4D(d_L(z), M)]`` for ``M ~ N(mu, sigma^2)``, shapes ``(n, K)`` in / ``(n, K)`` out.

    Structurally :func:`_mz_sel_2d_expectation_batch` called at ``det_M=1`` (so its
    fraction-coordinate ``x`` IS the actual mass), rewritten inline to add the disclosed
    non-positive-mass guard (module docstring) the coded leaf does not carry.
    """
    n, k = mu.shape
    n_g = _MT_GH_NODES.size
    a_nodes = mu[..., None] + np.sqrt(2.0) * sigma[..., None] * _MT_GH_NODES  # (n, k, G)
    pos = a_nodes > 0.0
    a_safe = np.where(pos, a_nodes, 1.0)  # dummy positive value for masked-out entries
    m_flat = a_safe.reshape(-1)
    d_L_flat = np.repeat(d_L_nodes.reshape(-1), n_g)
    z_flat = np.repeat(z_nodes.reshape(-1), n_g)
    phi_flat = np.repeat(host_phiS, k * n_g)
    theta_flat = np.repeat(host_qS, k * n_g)
    s4d = np.asarray(
        detection_probability.detection_probability_with_bh_mass_interpolated(
            d_L_flat,
            m_flat,
            phi_flat,
            theta_flat,
            h=h,
            **_wbh_z_kwargs(detection_probability, z_flat),
        ),
        dtype=np.float64,
    ).reshape(n, k, n_g)
    s4d = np.where(pos, s4d, 0.0)  # MINOR-6 guard: S(M<=0) := 0 (F2)
    expectation: npt.NDArray[np.float64] = (s4d @ _MT_GH_WEIGHTS) / np.sqrt(np.pi)
    return expectation


def compute_sigma_tilde_4d(
    completeness: Any,
    phi_survival_table: dict[float, Any],
    detection_probability: Any,
    h: float = H_GEN,
    chunk: int = CHUNK,
) -> dict[str, Any]:
    _t0 = time.time()
    cat = _load_eligible_catalogue(completeness, phi_survival_table, h)
    n_g = cat["z"].size
    print(f"  [+{time.time()-_t0:.1f}s] catalogue loaded, n_eligible={n_g}", file=sys.stderr)
    M_eff = _eddington_shifted_host_mass_batch(cat["M"], cat["M_error"])
    w_g = np.asarray(R_eff_per_mbh(cat["M"]), dtype=np.float64) / (1.0 + cat["z"])

    s_tilde_4d = np.empty(n_g, dtype=np.float64)
    n_nonpositive_nodes = 0
    for start in range(0, n_g, chunk):
        sl = slice(start, min(start + chunk, n_g))
        z_g = cat["z"][sl]
        z_err_eff = c1d.host_z_error_eff(z_g, cat["z_error"][sl])
        lower, upper = c1d._host_kernel_window(z_g, z_err_eff)
        half = 0.5 * (upper - lower)
        mid = 0.5 * (upper + lower)
        z_nodes = mid[:, None] + half[:, None] * c1d._GL_NODES_B0I[None, :]  # (m, 50)
        host_pixels = c1d._host_pixels(completeness, cat["phiS"][sl], cat["qS"][sl])
        w_pop_eff = c1d._kernel_w_pop_eff(z_nodes, completeness, host_pixels, h)
        gaussian_vals = norm.pdf(z_nodes, loc=z_g[:, None], scale=z_err_eff[:, None])
        kernel_unnorm = gaussian_vals * w_pop_eff  # (m, 50)

        d_L_nodes = np.asarray(dist_vectorized(z_nodes.ravel(), h=h), dtype=np.float64).reshape(
            z_nodes.shape
        )
        mu = np.broadcast_to(M_eff[sl][:, None], z_nodes.shape)
        sigma = np.broadcast_to(cat["M_error"][sl][:, None], z_nodes.shape)
        s_bar_4d_nodes = _mass_marginal_survival(
            mu, sigma, z_nodes, d_L_nodes, cat["phiS"][sl], cat["qS"][sl], detection_probability, h
        )
        n_nonpositive_nodes += int(
            np.count_nonzero(
                mu[..., None] + np.sqrt(2.0) * sigma[..., None] * _MT_GH_NODES <= 0.0
            )
        )

        numerator = np.sum(kernel_unnorm * s_bar_4d_nodes * c1d._GL_WEIGHTS_B0I[None, :], axis=1) * half
        z_norm = np.sum(kernel_unnorm * c1d._GL_WEIGHTS_B0I[None, :], axis=1) * half
        z_norm = np.where(z_norm > 0.0, z_norm, 1.0)
        s_tilde_4d[sl] = numerator / z_norm
        print(
            f"  [+{time.time()-_t0:.1f}s] chunk {start}:{sl.stop}/{n_g} done", file=sys.stderr
        )

    sigma_tilde_4d = float(np.sum(w_g * s_tilde_4d))
    return {
        "Sigma_tilde_4D": sigma_tilde_4d,
        "n_eligible": int(n_g),
        "n_nonpositive_mass_gh_nodes": int(n_nonpositive_nodes),
        "w_g": w_g,
        "s_tilde_4d": s_tilde_4d,
        "M_eff": M_eff,
        "catalogue": cat,
    }


def _spot_check(
    cat: dict[str, npt.NDArray[np.float64]],
    M_eff: npt.NDArray[np.float64],
    s_tilde_4d: npt.NDArray[np.float64],
    completeness: Any,
    detection_probability: Any,
    h: float,
    n_spot: int = 100,
    seed: int = 20260825,
) -> dict[str, Any]:
    """Independent recompute of ``S~_4D,g`` for ``n_spot`` random rows via nested
    ``scipy.integrate.quad`` (z outer, M inner) -- a genuinely different numerical method
    (adaptive quadrature) from the GL(50)xGH(24) product-quadrature the main pass uses.
    """
    rng = np.random.default_rng(seed)
    n_g = cat["z"].size
    idx = rng.choice(n_g, size=min(n_spot, n_g), replace=False)
    rel_devs = []
    for i in idx:
        z_g = float(cat["z"][i])
        z_err_eff = float(c1d.host_z_error_eff(np.array([z_g]), cat["z_error"][i : i + 1])[0])
        lower, upper = c1d._host_kernel_window(np.array([z_g]), np.array([z_err_eff]))
        lower, upper = float(lower[0]), float(upper[0])
        pixel = c1d._host_pixels(completeness, cat["phiS"][i : i + 1], cat["qS"][i : i + 1])
        mu_i, sigma_i = float(M_eff[i]), float(cat["M_error"][i])

        def _w_pop_eff_scalar(z: float) -> float:
            z_arr = np.array([z])
            return float(c1d._kernel_w_pop_eff(z_arr[:, None], completeness, pixel, h)[0, 0])

        def _s_bar_4d(z: float, mu_i: float = mu_i, sigma_i: float = sigma_i) -> float:
            d_l = float(dist_vectorized(np.array([z]), h=h)[0])
            if sigma_i <= 0.0:
                m_query = max(mu_i, 1e-6)
                return float(
                    detection_probability.detection_probability_with_bh_mass_interpolated(
                        d_l, m_query, 0.0, 0.0, h=h,
                        **_wbh_z_kwargs(detection_probability, np.array([z])),
                    )
                )

            def _integrand_m(m: float) -> float:
                if m <= 0.0:
                    return 0.0
                s4d = float(
                    detection_probability.detection_probability_with_bh_mass_interpolated(
                        d_l, m, 0.0, 0.0, h=h,
                        **_wbh_z_kwargs(detection_probability, np.array([z])),
                    )
                )
                return s4d * float(norm.pdf(m, loc=mu_i, scale=sigma_i))

            lo_m = max(mu_i - 6.0 * sigma_i, 1e-6)
            hi_m = mu_i + 6.0 * sigma_i
            val, _ = integrate.quad(
                _integrand_m, lo_m, hi_m, limit=50, epsabs=1e-13, epsrel=1e-7
            )
            return val

        def _num_integrand(z: float) -> float:
            return float(norm.pdf(z, loc=z_g, scale=z_err_eff)) * _w_pop_eff_scalar(z) * _s_bar_4d(z)

        def _den_integrand(z: float) -> float:
            return float(norm.pdf(z, loc=z_g, scale=z_err_eff)) * _w_pop_eff_scalar(z)

        num, _ = integrate.quad(_num_integrand, lower, upper, limit=50, epsabs=1e-13, epsrel=1e-7)
        den, _ = integrate.quad(_den_integrand, lower, upper, limit=50, epsabs=1e-13, epsrel=1e-7)
        quad_val = num / den if den > 0.0 else 0.0
        gl_val = float(s_tilde_4d[i])
        rel_dev = abs(quad_val - gl_val) / max(abs(quad_val), 1e-300)
        rel_devs.append(rel_dev)
        print(
            f"    spot row {len(rel_devs)}/{min(n_spot, n_g)}: quad={quad_val:.6e} "
            f"gl={gl_val:.6e} rel_dev={rel_dev:.3e}",
            file=sys.stderr,
        )

    rel_devs_arr = np.asarray(rel_devs, dtype=np.float64)
    return {
        "n_spot": int(len(rel_devs)),
        "max_rel_dev": float(np.max(rel_devs_arr)),
        "median_rel_dev": float(np.median(rel_devs_arr)),
        "target_rel_dev": 1.0e-6,
        "target_met": bool(np.max(rel_devs_arr) <= 1.0e-6),
    }


def main() -> None:
    if OUT_PATH.is_file():
        raise SystemExit(
            f"REFUSED (PA-CA-11): {OUT_PATH} already exists -- purge or use a fresh out path "
            "before a banked companion-pass run."
        )
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    stamp = _a22_stamp_2d()  # A22: written before any scoring call.
    t0 = time.time()

    print(f"[t={time.time()-t0:.1f}s] building b0i-2D selection objects...", file=sys.stderr)
    completeness, phi_survival_table, detection_probability = c1d.build_b0i_2d_selection_objects(
        h_true=H_GEN
    )
    print(f"[t={time.time()-t0:.1f}s] selection objects built; computing Sigma~^4D...", file=sys.stderr)
    result = compute_sigma_tilde_4d(completeness, phi_survival_table, detection_probability, H_GEN)
    print(f"[t={time.time()-t0:.1f}s] Sigma~^4D pass done.", file=sys.stderr)
    t_companion = time.time() - t0

    # beta_G_phi / beta_Gbar_phi / Sigma_phi: banked, seed-invariant, venue-independent scalars.
    at = o5._rows_at_h(_REFERENCE_CSV, H_GEN)
    beta_g_phi, beta_gbar_phi = o5._beta_g_phi_and_gbar(at)
    mc = o5.mass_companion(H_GEN)
    sigma_phi = mc["Sigma_phi"]

    c2_star = beta_g_phi * result["Sigma_tilde_4D"] / (sigma_phi * beta_gbar_phi)

    t1 = time.time()
    import os as _os

    n_spot = int(_os.environ.get("P3_2D_COMPANION_N_SPOT", "100"))
    spot = _spot_check(
        result["catalogue"],
        result["M_eff"],
        result["s_tilde_4d"],
        completeness,
        detection_probability,
        H_GEN,
        n_spot=n_spot,
    )
    t_spot = time.time() - t1

    out: dict[str, Any] = {
        "reference": f"{REGISTRATION_SECTION}",
        "h": H_GEN,
        "resolved_flags": stamp,
        "Sigma_tilde_4D": result["Sigma_tilde_4D"],
        "n_eligible": result["n_eligible"],
        "n_nonpositive_mass_gh_nodes": result["n_nonpositive_mass_gh_nodes"],
        "beta_G_phi": beta_g_phi,
        "beta_Gbar_phi": beta_gbar_phi,
        "Sigma_phi": sigma_phi,
        "reference_csv": str(_REFERENCE_CSV),
        "C2_star": c2_star,
        "spot_check": spot,
        "wall_time_s": {"companion_pass": t_companion, "spot_check": t_spot, "total": time.time() - t0},
    }
    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k not in ("resolved_flags",)}, indent=2))
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
