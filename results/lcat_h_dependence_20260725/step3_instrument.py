"""Step 3 — instrumented L_cat(h) recomputation, validation, and factor decomposition.

For each selected event and each of the 41 h values:

  BASELINE (pipeline's own code): single_host_likelihood_batch(...,
  normalization_mode="volume_deconv") -> per-host N_g, D_g; L_cat =
  weighted_ratio_of_sums. Validated against the shipped diagnostics CSV values.

  FACTORS (instrumented mirror of the same quadrature, validated against the
  kernel outputs to numerical noise):
    - Z_g(h): the volume_deconv per-host kernel normalisation
    - p_g(z) h-invariance check (the volume kernel's h^-3 cancels in Z_g)
    - numerator counterfactuals: freeze the GW window at h_ref vs freeze the
      integrand at h_ref (windows vs physics attribution)
    - denominator counterfactual: freeze the d_L(z;h) map inside p_det at h_ref
      (the only h-dependence in D_g -- p_det itself is an h-invariant survival)
    - per-host preferred h*_g = h_ref * d_L(z_g; h_ref) / d_L_det (exact, since
      d_L ~ 1/h at fixed z in LCDM)

  MODE SWAPS: normalization_mode="local_ratio" (bare Gaussian kernel) full curve;
  "global" mode curve = weighted_sum(N_bare)/Sigma_global(h) from step 2.

Outputs decomposition_results.json (compact: per-event per-h factor curves).
"""

import json
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/jasper/Repositories/MasterThesisCode")

import master_thesis_code.bayesian_inference.bayesian_statistics as bs  # noqa: E402
from master_thesis_code.bayesian_inference.bayesian_statistics import (  # noqa: E402
    _batched_gl_nodes,
    _batched_gl_reduce,
    _gaussian_pdf,
    _mvn_pdf,
    child_process_init,
    single_host_likelihood,
    single_host_likelihood_batch,
    use_detection,
    weighted_ratio_of_sums,
    weighted_sum,
)
from master_thesis_code.bayesian_inference.simulation_detection_probability import (  # noqa: E402
    SimulationDetectionProbability,
)
from master_thesis_code.constants import (  # noqa: E402
    SIGMA_V_PEC_KM_S,
    SNR_THRESHOLD,
    SPEED_OF_LIGHT_KM_S,
)
from master_thesis_code.datamodels.detection import Detection  # noqa: E402
from master_thesis_code.physical_relations import (  # noqa: E402
    comoving_volume_element,
    dist_to_redshift,
    dist_vectorized,
)

OUT = "/home/jasper/Repositories/MasterThesisCode/results/lcat_h_dependence_20260725"
VENUE = "/home/jasper/Repositories/MasterThesisCode/results/campaign_phase2_runs/run_20260703_seed1000"
H_REF = 0.73
QN = 50  # fixed_quad order used by the kernel

sel = json.load(open(f"{OUT}/selected_events.json"))
cand = json.load(open(f"{OUT}/candidates.json"))
glob_sums = {float(k): v for k, v in json.load(open(f"{OUT}/global_sums.json"))["no_bh"].items()}
h_grid = [float(h) for h in sel["h_grid"]]
event_ids = sorted(int(k) for k in sel["events"])

# ---------- rebuild detection arrays exactly as evaluate() does ----------
crb = pd.read_csv(f"{VENUE}/simulations/prepared_cramer_rao_bounds.csv")
crb = crb[crb["SNR"] >= SNR_THRESHOLD]
for index, row in crb.iterrows():
    if not use_detection(Detection(row)):
        crb.drop(index, inplace=True)
print(f"detections after filters: {len(crb)}")

det_indices = list(crb.index)
n_det = len(det_indices)
slot_of = {int(i): s for s, i in enumerate(det_indices)}
means_3d = np.zeros((n_det, 3))
cov_inv_3d = np.zeros((n_det, 3, 3))
log_norm_3d = np.zeros(n_det)
det_d_L = np.zeros(n_det)
det_d_L_unc = np.zeros(n_det)
det_M = np.zeros(n_det)
det_phi = np.zeros(n_det)
det_theta = np.zeros(n_det)
for index, row in crb.iterrows():
    det = Detection(row)
    s = slot_of[int(index)]
    det_d_L[s], det_d_L_unc[s], det_M[s] = det.d_L, det.d_L_uncertainty, det.M
    det_phi[s], det_theta[s] = det.phi, det.theta
    cov_3d = np.array(
        [
            [det.phi_error**2, det.theta_phi_covariance, det.d_L_phi_covariance / det.d_L],
            [det.theta_phi_covariance, det.theta_error**2, det.d_L_theta_covariance / det.d_L],
            [
                det.d_L_phi_covariance / det.d_L,
                det.d_L_theta_covariance / det.d_L,
                det.d_L_uncertainty**2 / det.d_L**2,
            ],
        ]
    )
    means_3d[s] = [det.phi, det.theta, 1]
    cov_inv_3d[s] = np.linalg.pinv(cov_3d)
    _, logdet = np.linalg.slogdet(cov_3d)
    log_norm_3d[s] = -0.5 * (3 * np.log(2 * np.pi) + logdet)

detprob = SimulationDetectionProbability(
    injection_data_dir=f"{OUT}/data/injections",
    snr_threshold=SNR_THRESHOLD,
    dl_bins=60,
    mass_bins=40,
    estimator="local_linear",
    expected_z_max=1.5,
)

dummy4 = np.zeros((n_det, 4))
child_process_init(
    0.0,
    1.5,
    10**4.5,
    10**6.0,
    detprob,
    means_3d,
    cov_inv_3d,
    log_norm_3d,
    dummy4,
    np.zeros((n_det, 4, 4)),
    np.zeros(n_det),
    slot_of,
    np.zeros(n_det),
    np.zeros((n_det, 3)),
    det_d_L,
    det_d_L_unc,
    det_M,
    det_phi,
    det_theta,
    None,
)

GLN = bs._GL_NODES_50
GLW = bs._GL_WEIGHTS_50

results: dict[str, dict] = {"h_grid": h_grid, "h_ref": H_REF, "events": {}}

for idx in event_ids:
    t0 = time.time()
    c = cand[str(idx)]
    hosts = c["hosts"]
    hz = np.asarray(hosts["z"])
    hz_err = np.asarray(hosts["z_error"])
    hphi = np.asarray(hosts["phiS"])
    hq = np.asarray(hosts["qS"])
    hM = np.asarray(hosts["M"])
    hM_err = np.asarray(hosts["M_error"])
    w = np.asarray(hosts["w"])
    n = hz.size
    s = slot_of[idx]
    dL_det, dL_unc = det_d_L[s], det_d_L_unc[s]

    # host-window pieces (h-INDEPENDENT by construction)
    sigma_z_pv = (1.0 + hz) * SIGMA_V_PEC_KM_S / SPEED_OF_LIGHT_KM_S
    sig_eff = np.sqrt(hz_err**2 + sigma_z_pv**2)
    den_lo = np.maximum(hz - 4.0 * sig_eff, 1e-6)
    den_hi = hz + 4.0 * sig_eff
    y_den = _batched_gl_nodes(den_lo, den_hi, GLN)  # (n, 50)
    gauss_den = _gaussian_pdf(y_den, hz[:, None], sig_eff[:, None])
    dL_den_ref = dist_vectorized(y_den.reshape(-1), h=H_REF).reshape(n, QN)

    # exact per-host preferred h (d_L ∝ 1/h at fixed z)
    h_star = H_REF * dist_vectorized(hz, h=H_REF) / dL_det

    mean3, covinv3, lognorm3 = means_3d[s], cov_inv_3d[s], log_norm_3d[s]

    def z_prior(y_nodes: np.ndarray, lo: np.ndarray, hi: np.ndarray, h: float, volume: bool):
        """Mirror of the kernel's per-host z prior at arbitrary nodes; returns (pdf, Z_g)."""
        base = _gaussian_pdf(y_nodes, hz[:, None], sig_eff[:, None])
        if not volume:
            return base, np.ones(n)
        w_pop = (
            np.asarray(comoving_volume_element(y_nodes.reshape(-1), h=h), dtype=np.float64)
            / (1.0 + y_nodes.reshape(-1))
        ).reshape(y_nodes.shape)
        gd = _gaussian_pdf(y_den, hz[:, None], sig_eff[:, None])
        wpd = (
            np.asarray(comoving_volume_element(y_den.reshape(-1), h=h), dtype=np.float64)
            / (1.0 + y_den.reshape(-1))
        ).reshape(n, QN)
        Zg = _batched_gl_reduce(den_lo, den_hi, GLW, gd * wpd)
        Zg = np.where(Zg <= 0.0, 1.0, Zg)
        return base * w_pop / Zg[:, None], Zg

    def num_windows(h: float):
        zu = dist_to_redshift(dL_det + 4.0 * dL_unc, h=h)
        zl = dist_to_redshift(dL_det - 4.0 * dL_unc, h=h)
        return zl, zu

    def numerator(h_window: float, h_integrand: float, volume: bool):
        """N_g with the GW window taken at h_window and the integrand at h_integrand."""
        zl, zu = num_windows(h_window)
        y1d = (zu - zl) * (GLN + 1) / 2.0 + zl  # (50,)
        y = np.broadcast_to(y1d[None, :], (n, QN))
        dL = dist_vectorized(y1d, h=h_integrand)
        frac = np.broadcast_to((dL / dL_det)[None, :], (n, QN))
        x = np.empty((n, QN, 3))
        x[:, :, 0] = hphi[:, None]
        x[:, :, 1] = hq[:, None]
        x[:, :, 2] = frac
        gw = _mvn_pdf(x.reshape(-1, 3), mean3, covinv3, lognorm3).reshape(n, QN)
        prior, _ = z_prior(y, np.full(n, zl), np.full(n, zu), h_integrand, volume)
        return _batched_gl_reduce(np.full(n, zl), np.full(n, zu), GLW, gw * prior)

    def denominator(h_map: float, h_prior: float, volume: bool):
        """D_g with d_L(z;h_map) inside p_det and the z-prior at h_prior."""
        dL = dist_vectorized(y_den.reshape(-1), h=h_map)
        p_det = np.asarray(
            detprob.detection_probability_without_bh_mass_interpolated_zero_fill(
                dL, np.repeat(hphi, QN), np.repeat(hq, QN), h=h_map
            )
        ).reshape(n, QN)
        prior, _ = z_prior(y_den, den_lo, den_hi, h_prior, volume)
        return _batched_gl_reduce(den_lo, den_hi, GLW, p_det * prior)

    ev = {
        "n_hosts": n,
        "role": sel["events"][str(idx)]["role"],
        "z_inj": sel["events"][str(idx)]["z_inj"],
        "d_L": dL_det,
        "host_z_summary": {
            "min": float(hz.min()),
            "wq16": float(np.quantile(hz, 0.16)),
            "wmed": float(np.quantile(hz, 0.5)),
            "wq84": float(np.quantile(hz, 0.84)),
            "max": float(hz.max()),
        },
        "h_star_summary": {
            "min": float(h_star.min()),
            "q16": float(np.quantile(h_star, 0.16)),
            "med": float(np.quantile(h_star, 0.5)),
            "q84": float(np.quantile(h_star, 0.84)),
            "max": float(h_star.max()),
            "frac_below_060": float(np.mean(h_star < 0.60)),
            "frac_in_grid": float(np.mean((h_star >= 0.60) & (h_star <= 0.86))),
        },
        "curves": {
            k: []
            for k in [
                "L_cat_kernel_vd",
                "L_cat_shipped",
                "sumN_vd",
                "sumD_vd",
                "sumN_bare",
                "sumD_bare",
                "L_cat_bare",
                "L_cat_global",
                "sumN_win_frozen",
                "sumN_integrand_frozen",
                "sumD_pdet_map_frozen",
                "Zg_scale_check",
                "prior_hinv_maxdev",
                "num_window",
                "kernel_batch_vs_mirror_maxrel",
            ]
        },
    }

    for h in h_grid:
        # --- pipeline baseline (production batch kernel) ---
        res_vd = single_host_likelihood_batch(
            hphi, hq, hz, hz_err, hM, hM_err, idx, h, False, "volume_deconv"
        )
        N_vd, D_vd = res_vd[:, 0], res_vd[:, 1]
        L_vd = weighted_ratio_of_sums(N_vd, D_vd, w)

        res_br = single_host_likelihood_batch(
            hphi, hq, hz, hz_err, hM, hM_err, idx, h, False, "local_ratio"
        )
        N_br, D_br = res_br[:, 0], res_br[:, 1]
        L_br = weighted_ratio_of_sums(N_br, D_br, w)
        L_gl = weighted_sum(N_br, w) / glob_sums[h]

        # --- instrumented mirror + validation vs kernel ---
        N_mirror = numerator(h, h, volume=True)
        D_mirror = denominator(h, h, volume=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.nanmax(
                np.abs(
                    np.concatenate(
                        [
                            np.where(N_vd != 0, (N_mirror - N_vd) / N_vd, 0.0),
                            np.where(D_vd != 0, (D_mirror - D_vd) / D_vd, 0.0),
                        ]
                    )
                )
            )

        # --- counterfactuals ---
        N_winfrozen = numerator(H_REF, h, volume=True)  # window pinned at h_ref
        N_intfrozen = numerator(h, H_REF, volume=True)  # integrand pinned at h_ref
        # D with the d_L(z;h)->p_det map frozen at h_ref (prior at h; prior is
        # h-invariant anyway -- checked below)
        p_det_ref = np.asarray(
            detprob.detection_probability_without_bh_mass_interpolated_zero_fill(
                dL_den_ref.reshape(-1), np.repeat(hphi, QN), np.repeat(hq, QN), h=H_REF
            )
        ).reshape(n, QN)
        prior_h, Zg_h = z_prior(y_den, den_lo, den_hi, h, volume=True)
        D_pdetfrozen = _batched_gl_reduce(den_lo, den_hi, GLW, p_det_ref * prior_h)

        # kernel h-invariance checks
        prior_ref, Zg_ref = z_prior(y_den, den_lo, den_hi, H_REF, volume=True)
        zg_scale = float(np.max(np.abs(Zg_h * (h / H_REF) ** 3 / Zg_ref - 1.0)))
        prior_dev = float(
            np.max(np.abs(prior_h - prior_ref) / np.maximum(np.max(prior_ref, axis=1), 1e-300)[:, None])
        )

        cur = ev["curves"]
        cur["L_cat_kernel_vd"].append(float(L_vd))
        cur["L_cat_shipped"].append(
            sel["events"][str(idx)]["shipped_L_cat_no_bh"][h_grid.index(h)]
        )
        cur["sumN_vd"].append(float(weighted_sum(N_vd, w)))
        cur["sumD_vd"].append(float(weighted_sum(D_vd, w)))
        cur["sumN_bare"].append(float(weighted_sum(N_br, w)))
        cur["sumD_bare"].append(float(weighted_sum(D_br, w)))
        cur["L_cat_bare"].append(float(L_br))
        cur["L_cat_global"].append(float(L_gl))
        cur["sumN_win_frozen"].append(float(weighted_sum(N_winfrozen, w)))
        cur["sumN_integrand_frozen"].append(float(weighted_sum(N_intfrozen, w)))
        cur["sumD_pdet_map_frozen"].append(float(weighted_sum(D_pdetfrozen, w)))
        cur["Zg_scale_check"].append(zg_scale)
        cur["prior_hinv_maxdev"].append(prior_dev)
        cur["num_window"].append(list(num_windows(h)))
        cur["kernel_batch_vs_mirror_maxrel"].append(float(rel))

    # per-host quantities at the grid edges + ref (P1/P3 host-level tests)
    per_host = {}
    for hh in (0.60, H_REF, 0.86):
        r = single_host_likelihood_batch(
            hphi, hq, hz, hz_err, hM, hM_err, idx, hh, False, "volume_deconv"
        )
        per_host[f"N_{hh:.2f}"] = r[:, 0].tolist()
        per_host[f"D_{hh:.2f}"] = r[:, 1].tolist()
    per_host["z"] = hz.tolist()
    per_host["sig_z_eff"] = sig_eff.tolist()
    per_host["w"] = w.tolist()
    per_host["h_star"] = h_star.tolist()
    ev["per_host"] = per_host

    # scalar-kernel spot check at h_ref (production equivalence chain)
    j = int(np.argmin(np.abs(np.asarray(h_grid) - H_REF)))
    spot = np.array(
        [
            single_host_likelihood(
                hphi[i], hq[i], hz[i], hz_err[i], hM[i], hM_err[i], idx, h_grid[j], False
            )[:2]
            for i in range(min(n, 5))
        ]
    )
    res_vd = single_host_likelihood_batch(
        hphi, hq, hz, hz_err, hM, hM_err, idx, h_grid[j], False, "volume_deconv"
    )
    ev["scalar_vs_batch_maxrel_href"] = float(
        np.max(np.abs(spot - res_vd[: spot.shape[0], :2]) / np.maximum(np.abs(spot), 1e-300))
    )

    ship = np.asarray(ev["curves"]["L_cat_shipped"])
    mine = np.asarray(ev["curves"]["L_cat_kernel_vd"])
    ev["validation_maxrel_vs_shipped"] = float(np.max(np.abs(mine - ship) / np.abs(ship)))
    results["events"][str(idx)] = ev
    print(
        f"event {idx:5d}: n={n:5d}  maxrel(shipped)={ev['validation_maxrel_vs_shipped']:.3e}  "
        f"mirror={max(ev['curves']['kernel_batch_vs_mirror_maxrel']):.3e}  "
        f"scalarXbatch={ev['scalar_vs_batch_maxrel_href']:.3e}  ({time.time() - t0:.1f}s)"
    )

with open(f"{OUT}/decomposition_results.json", "w") as f:
    json.dump(results, f)
print("wrote decomposition_results.json")
