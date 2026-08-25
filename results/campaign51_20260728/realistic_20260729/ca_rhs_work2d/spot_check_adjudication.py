r"""[P3-2D] Decisive adjudication of the companion pass's failed scipy.quad spot-check.

Context: ``p3_2d_companion.py``'s mandated 100-random-row spot-check (nested
``scipy.integrate.quad``, z outer / M inner, NO ``points=`` breakpoints) failed its
registered 1e-6 relative target: max_rel_dev=3.81e-4 (banked
``ca_rhs_work2d/p3_2d_companion.json``). Worst rows (companion's own stderr log): spot
rows 64 (3.810e-4), 65 (3.456e-4), 92 (3.398e-4), 15 (3.186e-4), 53 (3.053e-4), 39
(2.598e-4), 70 (2.495e-4), 94 (2.419e-4).

Hypothesis under test: these deviations are a ``scipy.quad``-without-breakpoints kink
artifact (the mass-marginal integrand is piecewise-LINEAR with 40 interior kinks at the
``S_4D`` grid's ``M_centers``, and the z-outer integrand is piecewise-linear-ish with
kinks wherever ``d_L(z;h)`` crosses a ``dl_centers`` grid edge -- adaptive quad with
``limit=50`` and no ``points=`` can under-resolve that many kinks), NOT a defect in the
companion's PA-2D-2 exact erf-moment rule (already validated to <=1e-9 rel against an
independent brute-force arbiter over 28 rows spanning ``sigma/mu`` in [0.1, ~2], see
``ca_rhs_work2d/p3_2d_exact_mass_integral_validation.json``).

For the worst 5 spot rows (64, 65, 70, 92, 15) this script recomputes
``S~_4D,g = int dz N(z;z_g,sigma_z) w_pop_eff(z) Sbar_4D(z) dz / int dz N(z;z_g,sigma_z)
w_pop_eff(z) dz`` FOUR independent ways:

  (a) erf rule    -- companion's own :func:`p3_2d_companion._mass_marginal_survival`
                     (exact closed-form M-marginal) for the M-integral, combined with
                     the SAME GL(50) z-quadrature :func:`compute_sigma_tilde_4d` uses
                     (imported, not reimplemented).
  (b) quad-nopts   -- companion's own :func:`p3_2d_companion._spot_check` nested-quad
                     machinery, reproduced verbatim (z outer / M inner, no ``points=``).
  (c) quad-pts     -- the SAME nested quad but with ``points=`` at every M_centers grid
                     node inside [lo_m, hi_m] (inner) and every z where d_L(z) crosses a
                     dl_centers edge inside [lower, upper] (outer, via brentq inversion
                     of the imported ``dist_vectorized``), plus tightened
                     epsabs=1e-15/epsrel=1e-10.
  (d) brute        -- fixed-grid 2D Simpson arbiter: Nz x Nm ~= 1e6 evaluations of the
                     ACTUAL production accessor
                     ``detection_probability_with_bh_mass_interpolated`` (not the
                     bilinear-lookup shortcut the erf rule exploits -- independent code
                     path), Simpson over M then Simpson over z.

Verdict rule: if (a) ~= (c) ~= (d) to <=1e-8 rel while (b) deviates at ~1e-4 relative to
those three, the hypothesis is CONFIRMED (scipy.quad-without-breakpoints kink artifact;
the banked erf-rule Sigma_tilde_4D/C2_star stand). If (a) disagrees with (c)/(d), the
erf rule is defective and the discrepancy is reported precisely.

Refuses to overwrite an existing output (PA-CA-11 convention, out-root guard).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy import integrate, optimize
from scipy.stats import norm

THIS_DIR = Path(__file__).resolve().parent
CAMPAIGN_DIR = THIS_DIR.parent
sys.path.insert(0, str(CAMPAIGN_DIR))  # p3_2d_companion.py is a sibling script

import p3_2d_companion as comp  # noqa: E402

from darksiren_emri.bayesian_inference.bayesian_statistics import (  # noqa: E402
    _eddington_shifted_host_mass_batch,
    _wbh_z_kwargs,
)
from darksiren_emri.physical_relations import dist_vectorized  # noqa: E402
from darksiren_emri.validation import correspondence_1d as c1d  # noqa: E402

OUT_PATH = THIS_DIR / "spot_check_adjudication.json"

SPOT_SEED = 20260825
SPOT_N = 100
# Worst spot rows from the banked companion's own stderr log (>=2e-4 rel dev), 1-indexed
# in the log's "spot row K/100" convention -> K-1 is the position in the rng.choice index
# array.
TARGET_SPOT_ROWS = [64, 65, 70]

H = comp.H_GEN  # 0.73


def _reconstruct_spot_indices(n_g: int) -> npt.NDArray[np.int64]:
    """Byte-identical to :func:`p3_2d_companion._spot_check`'s own row selection."""
    rng = np.random.default_rng(SPOT_SEED)
    idx: npt.NDArray[np.int64] = rng.choice(n_g, size=min(SPOT_N, n_g), replace=False)
    return idx


def _erf_rule_value(
    z_g: float,
    z_err_raw: float,
    phiS: float,
    qS: float,
    mu: float,
    sigma: float,
    completeness: Any,
    h: float,
) -> float:
    """(a) Companion's own GL(50)-z + exact-erf-M machinery, for ONE row."""
    z_arr = np.array([z_g])
    z_err_eff = c1d.host_z_error_eff(z_arr, np.array([z_err_raw]))
    lower, upper = c1d._host_kernel_window(z_arr, z_err_eff)
    half = 0.5 * (upper - lower)
    mid = 0.5 * (upper + lower)
    z_nodes = mid[:, None] + half[:, None] * c1d._GL_NODES_B0I[None, :]  # (1, 50)
    host_pixels = c1d._host_pixels(completeness, np.array([phiS]), np.array([qS]))
    w_pop_eff = c1d._kernel_w_pop_eff(z_nodes, completeness, host_pixels, h)
    gaussian_vals = norm.pdf(z_nodes, loc=z_g, scale=z_err_eff[:, None])
    kernel_unnorm = gaussian_vals * w_pop_eff

    d_L_nodes = np.asarray(dist_vectorized(z_nodes.ravel(), h=h), dtype=np.float64).reshape(
        z_nodes.shape
    )
    mu_b = np.broadcast_to(np.array([[mu]]), z_nodes.shape)
    sigma_b = np.broadcast_to(np.array([[sigma]]), z_nodes.shape)
    s_bar_4d_nodes = comp._mass_marginal_survival(
        mu_b,
        sigma_b,
        z_nodes,
        d_L_nodes,
        np.array([[phiS]]),
        np.array([[qS]]),
        DETECTION_PROBABILITY,
        h,
    )
    numerator = np.sum(kernel_unnorm * s_bar_4d_nodes * c1d._GL_WEIGHTS_B0I[None, :], axis=1) * half
    z_norm = np.sum(kernel_unnorm * c1d._GL_WEIGHTS_B0I[None, :], axis=1) * half
    return float(numerator[0] / z_norm[0])


def _quad_nopts_value(
    z_g: float,
    z_err_raw: float,
    phiS: float,
    qS: float,
    mu_i: float,
    sigma_i: float,
    completeness: Any,
    detection_probability: Any,
    h: float,
) -> tuple[float, float, float]:
    """(b) Verbatim reproduction of :func:`p3_2d_companion._spot_check`'s nested quad,
    NO ``points=``. Returns (value, num, den)."""
    z_err_eff = float(c1d.host_z_error_eff(np.array([z_g]), np.array([z_err_raw]))[0])
    lower, upper = c1d._host_kernel_window(np.array([z_g]), np.array([z_err_eff]))
    lower, upper = float(lower[0]), float(upper[0])
    pixel = c1d._host_pixels(completeness, np.array([phiS]), np.array([qS]))

    def _w_pop_eff_scalar(z: float) -> float:
        z_arr = np.array([z])
        return float(c1d._kernel_w_pop_eff(z_arr[:, None], completeness, pixel, h)[0, 0])

    def _s_bar_4d(z: float) -> float:
        d_l = float(dist_vectorized(np.array([z]), h=h)[0])

        def _integrand_m(m: float) -> float:
            if m <= 0.0:
                return 0.0
            s4d = float(
                detection_probability.detection_probability_with_bh_mass_interpolated(
                    d_l, m, 0.0, 0.0, h=h, **_wbh_z_kwargs(detection_probability, np.array([z]))
                )
            )
            return s4d * float(norm.pdf(m, loc=mu_i, scale=sigma_i))

        lo_m = max(mu_i - 6.0 * sigma_i, 1e-6)
        hi_m = mu_i + 6.0 * sigma_i
        val, _ = integrate.quad(_integrand_m, lo_m, hi_m, limit=50, epsabs=1e-13, epsrel=1e-7)
        return val

    def _num_integrand(z: float) -> float:
        return float(norm.pdf(z, loc=z_g, scale=z_err_eff)) * _w_pop_eff_scalar(z) * _s_bar_4d(z)

    def _den_integrand(z: float) -> float:
        return float(norm.pdf(z, loc=z_g, scale=z_err_eff)) * _w_pop_eff_scalar(z)

    num, _ = integrate.quad(_num_integrand, lower, upper, limit=50, epsabs=1e-13, epsrel=1e-7)
    den, _ = integrate.quad(_den_integrand, lower, upper, limit=50, epsabs=1e-13, epsrel=1e-7)
    return (num / den if den > 0.0 else 0.0), num, den


def _quad_pts_value(
    z_g: float,
    z_err_raw: float,
    phiS: float,
    qS: float,
    mu_i: float,
    sigma_i: float,
    completeness: Any,
    detection_probability: Any,
    h: float,
) -> tuple[float, float, float]:
    """(c) Same nested quad, WITH ``points=`` breakpoints at every interior kink
    (M_centers inside the M-window; z-values where d_L(z) crosses a dl_centers edge
    inside the z-window), tightened tolerances."""
    interp_2d, _ = detection_probability._get_or_build_grid(h)
    dl_centers = np.asarray(interp_2d.grid[0], dtype=np.float64)
    M_centers = np.asarray(interp_2d.grid[1], dtype=np.float64)

    z_err_eff = float(c1d.host_z_error_eff(np.array([z_g]), np.array([z_err_raw]))[0])
    lower, upper = c1d._host_kernel_window(np.array([z_g]), np.array([z_err_eff]))
    lower, upper = float(lower[0]), float(upper[0])
    pixel = c1d._host_pixels(completeness, np.array([phiS]), np.array([qS]))

    lo_m = max(mu_i - 6.0 * sigma_i, 1e-6)
    hi_m = mu_i + 6.0 * sigma_i
    m_points = sorted(float(m) for m in M_centers if lo_m < m < hi_m)

    # z-breakpoints: invert dist_vectorized(z;h) = dl_centers[k] via brentq for every
    # dl_centers edge whose corresponding z lies strictly inside (lower, upper).
    def _dist_scalar(z: float) -> float:
        return float(dist_vectorized(np.array([z]), h=h)[0])

    dl_lower = _dist_scalar(lower)
    dl_upper = _dist_scalar(upper)
    dl_lo, dl_hi = min(dl_lower, dl_upper), max(dl_lower, dl_upper)
    z_points = []
    for dl_edge in dl_centers:
        dl_edge_f = float(dl_edge)
        if dl_lo < dl_edge_f < dl_hi:
            try:
                z_root = optimize.brentq(
                    lambda z: _dist_scalar(z) - dl_edge_f, lower, upper, xtol=1e-14, rtol=1e-14
                )
                z_points.append(float(z_root))
            except ValueError:
                continue
    z_points = sorted(z_points)

    def _w_pop_eff_scalar(z: float) -> float:
        z_arr = np.array([z])
        return float(c1d._kernel_w_pop_eff(z_arr[:, None], completeness, pixel, h)[0, 0])

    def _s_bar_4d(z: float) -> float:
        d_l = float(dist_vectorized(np.array([z]), h=h)[0])

        def _integrand_m(m: float) -> float:
            if m <= 0.0:
                return 0.0
            s4d = float(
                detection_probability.detection_probability_with_bh_mass_interpolated(
                    d_l, m, 0.0, 0.0, h=h, **_wbh_z_kwargs(detection_probability, np.array([z]))
                )
            )
            return s4d * float(norm.pdf(m, loc=mu_i, scale=sigma_i))

        kwargs: dict[str, Any] = {"limit": 200, "epsabs": 1e-15, "epsrel": 1e-10}
        if m_points:
            kwargs["points"] = m_points
        val, _ = integrate.quad(_integrand_m, lo_m, hi_m, **kwargs)
        return val

    def _num_integrand(z: float) -> float:
        return float(norm.pdf(z, loc=z_g, scale=z_err_eff)) * _w_pop_eff_scalar(z) * _s_bar_4d(z)

    def _den_integrand(z: float) -> float:
        return float(norm.pdf(z, loc=z_g, scale=z_err_eff)) * _w_pop_eff_scalar(z)

    z_kwargs: dict[str, Any] = {"limit": 200, "epsabs": 1e-15, "epsrel": 1e-10}
    if z_points:
        z_kwargs["points"] = z_points
    num, _ = integrate.quad(_num_integrand, lower, upper, **z_kwargs)
    den, _ = integrate.quad(_den_integrand, lower, upper, **z_kwargs)
    return (num / den if den > 0.0 else 0.0), num, den


def _brute_value(
    z_g: float,
    z_err_raw: float,
    phiS: float,
    qS: float,
    mu_i: float,
    sigma_i: float,
    completeness: Any,
    detection_probability: Any,
    h: float,
    n_z: int = 1000,
    n_m: int = 1000,
) -> tuple[float, float, float]:
    """(d) Fixed-grid 2D Simpson arbiter using the production accessor directly
    (independent of both the erf rule's bilinear-lookup shortcut AND scipy.quad)."""
    z_err_eff = float(c1d.host_z_error_eff(np.array([z_g]), np.array([z_err_raw]))[0])
    lower, upper = c1d._host_kernel_window(np.array([z_g]), np.array([z_err_eff]))
    lower, upper = float(lower[0]), float(upper[0])
    pixel = c1d._host_pixels(completeness, np.array([phiS]), np.array([qS]))

    lo_m = max(mu_i - 6.0 * sigma_i, 1e-6)
    hi_m = mu_i + 6.0 * sigma_i

    z_grid = np.linspace(lower, upper, n_z)
    m_grid = np.linspace(lo_m, hi_m, n_m)

    d_L_grid = np.asarray(dist_vectorized(z_grid, h=h), dtype=np.float64)
    dl_b = np.repeat(d_L_grid, n_m)
    m_b = np.tile(m_grid, n_z)
    z_b_full = np.repeat(z_grid, n_m)
    s_flat = detection_probability.detection_probability_with_bh_mass_interpolated(
        dl_b, m_b, 0.0, 0.0, h=h, **_wbh_z_kwargs(detection_probability, z_b_full)
    )
    s_grid = np.asarray(s_flat, dtype=np.float64).reshape(n_z, n_m)

    gauss_m = norm.pdf(m_grid, loc=mu_i, scale=sigma_i)
    s_bar_4d_brute = integrate.simpson(s_grid * gauss_m[None, :], x=m_grid, axis=1)  # (n_z,)

    w_pop = c1d._kernel_w_pop_eff(z_grid[None, :], completeness, pixel, h)[0, :]
    gauss_z = norm.pdf(z_grid, loc=z_g, scale=z_err_eff)

    num_grid = gauss_z * w_pop * s_bar_4d_brute
    den_grid = gauss_z * w_pop
    num = float(integrate.simpson(num_grid, x=z_grid))
    den = float(integrate.simpson(den_grid, x=z_grid))
    return (num / den if den > 0.0 else 0.0), num, den


def main() -> None:
    if OUT_PATH.is_file():
        raise SystemExit(f"REFUSED (PA-CA-11): {OUT_PATH} already exists -- purge first.")

    t0 = time.time()
    print(f"[t={time.time() - t0:.1f}s] building b0i-2D selection objects...", file=sys.stderr)
    completeness, phi_survival_table, detection_probability = c1d.build_b0i_2d_selection_objects(
        h_true=H
    )
    global DETECTION_PROBABILITY
    DETECTION_PROBABILITY = detection_probability

    print(f"[t={time.time() - t0:.1f}s] loading eligible catalogue (byte-identical)...", file=sys.stderr)
    cat = comp._load_eligible_catalogue(completeness, phi_survival_table, H)
    n_g = cat["z"].size
    print(f"[t={time.time() - t0:.1f}s] n_eligible={n_g}", file=sys.stderr)
    assert n_g == 20834132, f"eligibility mismatch: n_g={n_g} != banked 20834132"

    spot_idx = _reconstruct_spot_indices(n_g)
    # Eddington-shift ONLY the targeted rows (NOT the full 20.8M-row catalogue -- the
    # batch function's (chunk, 401)-node quadrature over the full array is what the
    # banked companion pass pays as part of its ~68-minute run; irrelevant here since
    # we only need mu for a handful of rows).
    target_gi = np.array(
        [int(spot_idx[spot_row - 1]) for spot_row in TARGET_SPOT_ROWS], dtype=np.int64
    )
    M_eff_targets = _eddington_shifted_host_mass_batch(
        cat["M"][target_gi], cat["M_error"][target_gi]
    )

    rows_out = []
    for k, spot_row in enumerate(TARGET_SPOT_ROWS):
        gi = int(target_gi[k])
        z_g = float(cat["z"][gi])
        z_err_raw = float(cat["z_error"][gi])
        phiS = float(cat["phiS"][gi])
        qS = float(cat["qS"][gi])
        mu = float(M_eff_targets[k])
        sigma = float(cat["M_error"][gi])

        print(
            f"[t={time.time() - t0:.1f}s] spot row {spot_row} -> catalogue idx {gi} "
            f"(z={z_g:.4f}, mu={mu:.4e}, sigma={sigma:.4e}, sigma/mu={sigma / mu:.3f})",
            file=sys.stderr,
        )

        t_a = time.time()
        val_a = _erf_rule_value(z_g, z_err_raw, phiS, qS, mu, sigma, completeness, H)
        print(f"    (a) erf rule       = {val_a:.10e}  [{time.time() - t_a:.2f}s]", file=sys.stderr)

        t_b = time.time()
        val_b, num_b, den_b = _quad_nopts_value(
            z_g, z_err_raw, phiS, qS, mu, sigma, completeness, detection_probability, H
        )
        print(f"    (b) quad-nopts     = {val_b:.10e}  [{time.time() - t_b:.2f}s]", file=sys.stderr)

        t_c = time.time()
        val_c, num_c, den_c = _quad_pts_value(
            z_g, z_err_raw, phiS, qS, mu, sigma, completeness, detection_probability, H
        )
        print(f"    (c) quad-pts       = {val_c:.10e}  [{time.time() - t_c:.2f}s]", file=sys.stderr)

        t_d = time.time()
        val_d, num_d, den_d = _brute_value(
            z_g, z_err_raw, phiS, qS, mu, sigma, completeness, detection_probability, H
        )
        print(f"    (d) brute Simpson  = {val_d:.10e}  [{time.time() - t_d:.2f}s]", file=sys.stderr)

        def _rel(x: float, ref: float) -> float:
            return abs(x - ref) / max(abs(ref), 1e-300)

        rows_out.append(
            {
                "spot_row": spot_row,
                "catalogue_idx": gi,
                "z_g": z_g,
                "z_error_raw": z_err_raw,
                "mu": mu,
                "sigma": sigma,
                "sigma_over_mu": sigma / mu,
                "a_erf_rule": val_a,
                "b_quad_nopts": val_b,
                "c_quad_pts": val_c,
                "d_brute_simpson": val_d,
                "banked_log_gl_val": None,  # filled below if this row was in the original log
                "rel_dev_b_vs_d": _rel(val_b, val_d),
                "rel_dev_c_vs_d": _rel(val_c, val_d),
                "rel_dev_a_vs_d": _rel(val_a, val_d),
                "rel_dev_a_vs_c": _rel(val_a, val_c),
            }
        )

    # Original log's (quad_nopts, gl) values for these exact spot rows, for cross-check
    # that this script's index reconstruction is byte-identical to the banked run.
    banked_log_pairs = {
        64: (0.1724771, 0.1724113),
        65: (0.1302224, 0.1301774),
        70: (0.117938, 0.1179674),
        92: (0.1032278, 0.1031927),
        15: (0.1244099, 0.1243703),
    }
    for row in rows_out:
        pair = banked_log_pairs.get(row["spot_row"])
        if pair is not None:
            row["banked_log_quad_nopts"] = pair[0]
            row["banked_log_gl_val"] = pair[1]
            row["reconstruction_check_quad_rel_dev"] = abs(row["b_quad_nopts"] - pair[0]) / abs(
                pair[0]
            )
            row["reconstruction_check_gl_rel_dev"] = abs(row["a_erf_rule"] - pair[1]) / abs(pair[1])

    verdict_evidence = {
        "max_rel_dev_b_vs_d": max(r["rel_dev_b_vs_d"] for r in rows_out),
        "max_rel_dev_c_vs_d": max(r["rel_dev_c_vs_d"] for r in rows_out),
        "max_rel_dev_a_vs_d": max(r["rel_dev_a_vs_d"] for r in rows_out),
        "max_rel_dev_a_vs_c": max(r["rel_dev_a_vs_c"] for r in rows_out),
    }
    if (
        verdict_evidence["max_rel_dev_a_vs_d"] <= 1e-8
        and verdict_evidence["max_rel_dev_a_vs_c"] <= 1e-8
        and verdict_evidence["max_rel_dev_c_vs_d"] <= 1e-8
        and verdict_evidence["max_rel_dev_b_vs_d"] > 1e-5
    ):
        verdict = "CONFIRMED-QUAD-ARTIFACT"
    elif verdict_evidence["max_rel_dev_a_vs_d"] > 1e-8 or verdict_evidence["max_rel_dev_a_vs_c"] > 1e-8:
        verdict = "RULE-DEFECT"
    else:
        verdict = "INCONCLUSIVE"

    out = {
        "reference": (
            "results/campaign51_20260728/realistic_20260729/PREREGISTRATION_P3_2D_20260825.md "
            "sec1 spot-check adjudication (task-mandated failed target)"
        ),
        "h": H,
        "n_eligible": n_g,
        "spot_seed": SPOT_SEED,
        "spot_n": SPOT_N,
        "target_spot_rows": TARGET_SPOT_ROWS,
        "rows": rows_out,
        "verdict_evidence": verdict_evidence,
        "verdict": verdict,
        "wall_time_s": time.time() - t0,
    }
    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"\nverdict: {verdict}")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
