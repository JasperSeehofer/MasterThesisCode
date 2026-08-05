"""N-2 measurement M2: offline 2D-inertness check.

Spec (N2_SELECTION_NUMERATOR_DERIVATION_20260805.md.DRAFT section 3.1 / 6.4 M2
/ 6.3 P-4): verify offline that adding the derived selection factor S_4D to
the 2D completion numerator changes the 2D per-event ln-likelihood by an
h-INDEPENDENT per-event constant (i.e. zero tilt), to the pre-registered
tolerance |Sigma_i Delta ln p_i^2D| <= 20 nats/h (P-4). If confirmed, the
draft's own rule (section 6.4 M2) says the "both" counterfactual cell can be
dropped and only "1d" is needed.

Derivation being tested (T1''), draft section 3.1:

    g_i^corr(z;h) ~= g_i(z;h) * S_4D( d_L(z;h), mu_cond(z;h) * M_z,det )

    mu_cond(z;h) = 1 + proj * (x_dL(z;h) - 1),   x_dL = d_L(z;h)/d_L,det

At the GW peak z = z*(d_L,det;h): x_dL = 1, mu_cond = 1, so both S_4D
arguments equal the DATA (d_L,det, M_z,det) -- h-independent by construction,
proven algebraically, not measured here. The claimed residual comes from the
quadrature TAILS, where p_gw has support away from the peak and x_dL != 1.

What this script measures (offline, no BayesianStatistics instance built --
S_4D's grid is h-invariant per
`detection_probability_with_bh_mass_interpolated`'s docstring, so it needs
only the injection pool + per-event (d_L,det, d_L_uncertainty, M_z,det) from
the CRB CSV):

For each event, build an 11-node Gauss-Hermite quadrature IN Z-SPACE (matching
the production z-quadrature that _completion_numerators integrates over),
centered on z_center(h) = dist_to_redshift(d_L,det, h) with width
sigma_z(h) obtained by propagating the event's OWN fractional distance error
sigma_dLfrac = d_L_uncertainty/d_L,det through the local dz/d(ln d_L) slope at
z_center(h) (the GW likelihood p_gw is a Gaussian in d_L-fraction; this is the
leading-order proxy for the quadrature's *_comp_sigma_dLfrac* width used in
production -- see the honest-gap note below). This is NOT the same as building
the quadrature directly in d_L-space around the (h-independent) data d_L,det:
that construction was tried first and is WRONG -- it makes S_4D's first
argument h-independent by fiat and trivially returns zero tilt regardless of
the physics; z-space construction is required so d_L(z_node;h) genuinely
moves with h away from the peak. At each h, map each z node to
d_L(z_node;h) = dist_vectorized(z_node, h), then evaluate S_4D there under TWO
bracketing assumptions for the mass slot (proj is not exposed outside a live
BayesianStatistics instance -- see AMBIGUITY note):

    proj=0 (mass held at data M_z,det -- the leading-order case, since
            sigma_cond ~ 8.8e-8 already pins the mass slot near M_z,det and
            proj*(x_dL-1) is a second-order correction on top of an already
            small x_dL-1)
    proj=1 (mass tracks distance 1:1: M_z query = x_dL * M_z,det -- an
            upper-sensitivity bound)

giving the GH-weighted average survival S_bar_quad_i(h) under each
assumption, and

    Delta_ln_i(h) = ln( S_bar_quad_i(h) ) - ln( S_4D(d_L,det_i, M_z,det_i) )

which isolates the RESIDUAL (h-independent peak value subtracted off) --
exactly what "an h-independent per-event constant" claims should vanish.
Sigma_i Delta_ln_i(h) is then compared to the P-4 tolerance (20 nats/h) via
its central-difference slope at h=0.73 and its chord slope over the full grid.

AMBIGUITY flagged, not guessed: this uses sigma_dLfrac = d_L_uncertainty/d_L
(direct CRB fractional 1-sigma error) as a proxy for the production
`_comp_sigma_dLfrac` (a 3D-covariance-projected quantity,
bayesian_statistics.py:4045) and omits the (1-f_k)*dVc/(1+z) quadrature
weight (a smooth, slowly-varying-in-z factor relative to the sharply peaked
p_gw, and common to both the corrected and uncorrected numerator so it
approximately cancels in the ratio that defines Delta_ln_i). It also does not
have access to the per-event Fisher `proj` (d_L-mass correlation
projection), hence the proj=0/proj=1 bracket rather than the single
production value.

Read-only. No source modified. No run launched.
"""

import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/jasper/Repositories/MasterThesisCode")

from master_thesis_code.bayesian_inference.simulation_detection_probability import (  # noqa: E402
    SimulationDetectionProbability,
)
from master_thesis_code.constants import C, GPC_TO_MPC, KM_TO_M, OMEGA_DE, OMEGA_M  # noqa: E402
from master_thesis_code.physical_relations import (  # noqa: E402
    _get_z_from_dl_ratio_spline,
    dist_to_redshift,
    dist_vectorized,
)


def dist_to_redshift_fast(distance: np.ndarray, h: float) -> np.ndarray:
    """Vectorized fiducial-LCDM inverse of dist_vectorized (bypasses the
    per-call fsolve/spline-scalar path in dist_to_redshift for speed; this
    script needs O(1590 * 41 * 11) inversions). Falls back to the exact
    scalar dist_to_redshift for any out-of-spline-range points (rare, only
    at the widest Gauss-Hermite tails for high-sigma_dLfrac events)."""
    H_0 = h * 100.0 * KM_TO_M / GPC_TO_MPC ** (-1)
    spline = _get_z_from_dl_ratio_spline(OMEGA_M, OMEGA_DE)
    u = np.asarray(distance, dtype=np.float64) * H_0 / C
    in_range = (u >= 0.0) & (u <= float(spline.x[-1]))
    z = np.empty_like(u)
    z[in_range] = spline(u[in_range])
    if (~in_range).any():
        z[~in_range] = [
            float(dist_to_redshift(d, h=float(h))) for d in distance[~in_range]
        ]
    return z

POOL = (
    "/home/jasper/Repositories/MasterThesisCode/results/campaign51_20260728/"
    "realistic_20260729/gate_b_20260730/injection_pool_mix200k_20260728"
)
BASE = "/home/jasper/Repositories/MasterThesisCode/results/run_20260804_postfix"
VENUES = ["iiib", "joint_r1"]
H_TARGET = 0.73
TOLERANCE_NATS_PER_H = 20.0
GH_N = 11

sdp = SimulationDetectionProbability(
    injection_data_dir=POOL,
    snr_threshold=20.0,
    expected_z_max=1.5,
    allow_shallow_pool=True,
)
print("pool loaded", flush=True)

# Gauss-Hermite nodes/weights for integral (1/sqrt(pi)) INTEGRAL f(x) e^{-x^2} dx
# ~= sum w_k f(x_k)  ==  E[f(X)] for X ~ N(0, 1/2). Rescale to X ~ N(0,1):
# t = sqrt(2) * x, so a standard-normal-weighted average of f is
# sum w_k f(sqrt(2)*x_k) / sqrt(pi).
gh_x, gh_w = np.polynomial.hermite.hermgauss(GH_N)
gh_t = np.sqrt(2.0) * gh_x
gh_w_norm = gh_w / np.sqrt(np.pi)  # sums to 1


def s4d(dl: np.ndarray, mz: np.ndarray, h: float) -> np.ndarray:
    return np.asarray(
        sdp.detection_probability_with_bh_mass_interpolated(
            dl, mz, phi=0.0, theta=0.0, h=h
        ),
        dtype=np.float64,
    )


results: dict[str, dict] = {}

for venue in VENUES:
    print(f"=== venue {venue} ===", flush=True)
    csv_path = f"{BASE}/{venue}/diagnostics/event_likelihoods.csv"
    crb_path = f"{BASE}/{venue}/diagnostics/prepared_cramer_rao_bounds.csv"
    df = pd.read_csv(csv_path)
    crb = pd.read_csv(crb_path)

    h_grid = np.array(sorted(df["h"].unique()), dtype=float)
    n_events_csv = df["event_idx"].nunique()
    used_idx = np.array(sorted(df["event_idx"].unique()), dtype=int)
    print(f"{venue}: {n_events_csv} events (csv), {h_grid.size} h points, "
          f"CRB {len(crb)} rows", flush=True)

    dl_det = crb["luminosity_distance"].to_numpy(dtype=float)[used_idx]
    dl_unc = np.sqrt(
        crb["delta_luminosity_distance_delta_luminosity_distance"].to_numpy(dtype=float)
    )[used_idx]
    mz_det = crb["M"].to_numpy(dtype=float)[used_idx]
    sigma_dlfrac = dl_unc / dl_det

    n_events = used_idx.size

    # Peak value (h-independent by construction): S_4D(d_L,det, M_z,det)
    s_peak = s4d(dl_det, mz_det, H_TARGET)  # h-invariant grid; h value is a no-op
    ln_s_peak = np.log(np.clip(s_peak, 1e-300, None))

    sum_delta_ln_proj0 = np.zeros(h_grid.size)
    sum_delta_ln_proj1 = np.zeros(h_grid.size)
    n_negative_dl_clipped = 0

    # The quadrature MUST be built in z-space (matching the production
    # z-quadrature that _completion_numerators actually integrates over) and
    # THEN mapped to d_L(z;h) -- NOT built directly in d_L-space around the
    # (h-independent) data d_L,det, which would make every node's first S_4D
    # argument h-independent by construction and trivially yield zero tilt
    # regardless of the true physics. z_center(h) = dist_to_redshift(d_L,det, h)
    # is h-dependent; sigma_z(h) is obtained by propagating the event's own
    # sigma_dLfrac through the LOCAL d(z)/d(ln d_L) slope at z_center(h) (finite
    # difference on the fast vectorized inversion -- avoids the slow scalar
    # dist_derivative, which does a 1000-point trapz integral per call).
    eps = 1e-4
    for j, h in enumerate(h_grid):
        z_center = dist_to_redshift_fast(dl_det, float(h))  # (n_events,)
        z_plus = dist_to_redshift_fast(dl_det * (1.0 + eps), float(h))
        z_minus = dist_to_redshift_fast(dl_det * (1.0 - eps), float(h))
        dz_dlnDL = (z_plus - z_minus) / (2.0 * eps)  # local slope dz/d(ln d_L)
        sigma_z = np.abs(dz_dlnDL) * sigma_dlfrac  # (n_events,)

        z_nodes = z_center[:, None] + sigma_z[:, None] * gh_t[None, :]
        neg_mask = z_nodes <= 1e-6
        n_negative_dl_clipped += int(neg_mask.sum())
        z_nodes = np.clip(z_nodes, 1e-6, None)

        dl_nodes = np.asarray(dist_vectorized(z_nodes, h=float(h)), dtype=np.float64)
        x_dl_nodes = dl_nodes / dl_det[:, None]

        # round-trip sanity: center node (t=0 exactly, unclipped) must
        # reproduce d_L,det to numerical precision
        dl_center_check = np.asarray(
            dist_vectorized(z_center, h=float(h)), dtype=np.float64
        )
        roundtrip_err = np.max(
            np.abs(dl_center_check - dl_det) / np.maximum(dl_det, 1e-30)
        )

        mz_proj0 = np.broadcast_to(mz_det[:, None], dl_nodes.shape)
        mz_proj1 = x_dl_nodes * mz_det[:, None]

        s_proj0 = s4d(dl_nodes.ravel(), mz_proj0.ravel(), float(h)).reshape(dl_nodes.shape)
        s_proj1 = s4d(dl_nodes.ravel(), mz_proj1.ravel(), float(h)).reshape(dl_nodes.shape)

        s_bar_proj0 = (s_proj0 * gh_w_norm[None, :]).sum(axis=1)
        s_bar_proj1 = (s_proj1 * gh_w_norm[None, :]).sum(axis=1)

        ln_s_bar_proj0 = np.log(np.clip(s_bar_proj0, 1e-300, None))
        ln_s_bar_proj1 = np.log(np.clip(s_bar_proj1, 1e-300, None))

        delta_proj0 = ln_s_bar_proj0 - ln_s_peak
        delta_proj1 = ln_s_bar_proj1 - ln_s_peak

        sum_delta_ln_proj0[j] = float(delta_proj0.sum())
        sum_delta_ln_proj1[j] = float(delta_proj1.sum())

        if j == 0 or j == h_grid.size - 1:
            print(f"{venue}: h={h:.4f} center-node d_L round-trip max rel err "
                  f"{roundtrip_err:.3e}", flush=True)

    if n_negative_dl_clipped:
        print(
            f"{venue}: WARNING {n_negative_dl_clipped} (event,h,node) triples had "
            "d_L <= 0 at the wide GH tail, clipped to 1e-6 Gpc",
            flush=True,
        )

    h_sorted = h_grid
    i73 = int(np.argmin(np.abs(h_sorted - H_TARGET)))
    h_lo, h_hi = float(h_sorted.min()), float(h_sorted.max())

    def central_diff(arr: np.ndarray, i: int) -> float:
        if 0 < i < arr.size - 1:
            return float(
                (arr[i + 1] - arr[i - 1]) / (h_sorted[i + 1] - h_sorted[i - 1])
            )
        return float("nan")

    chord0 = float((sum_delta_ln_proj0[-1] - sum_delta_ln_proj0[0]) / (h_hi - h_lo))
    chord1 = float((sum_delta_ln_proj1[-1] - sum_delta_ln_proj1[0]) / (h_hi - h_lo))
    slope0_73 = central_diff(sum_delta_ln_proj0, i73)
    slope1_73 = central_diff(sum_delta_ln_proj1, i73)

    max_abs_sum0 = float(np.max(np.abs(sum_delta_ln_proj0)))
    max_abs_sum1 = float(np.max(np.abs(sum_delta_ln_proj1)))

    verdict_proj0 = "CONFIRMED (inert)" if max(abs(slope0_73), abs(chord0)) <= TOLERANCE_NATS_PER_H else "TILT EXCEEDS TOLERANCE"
    verdict_proj1 = "CONFIRMED (inert)" if max(abs(slope1_73), abs(chord1)) <= TOLERANCE_NATS_PER_H else "TILT EXCEEDS TOLERANCE"

    results[venue] = {
        "n_events": int(n_events),
        "h_grid_bounds": [h_lo, h_hi],
        "tolerance_nats_per_h": TOLERANCE_NATS_PER_H,
        "n_negative_dl_clipped_triples": int(n_negative_dl_clipped),
        "proj0_mass_fixed_at_data": {
            "sum_delta_ln_by_h": {str(h): float(v) for h, v in zip(h_sorted, sum_delta_ln_proj0)},
            "central_diff_slope_nats_per_h_at_073": slope0_73,
            "chord_slope_nats_per_h_full_grid": chord0,
            "max_abs_sum_over_grid_nats": max_abs_sum0,
            "verdict": verdict_proj0,
        },
        "proj1_mass_tracks_distance": {
            "sum_delta_ln_by_h": {str(h): float(v) for h, v in zip(h_sorted, sum_delta_ln_proj1)},
            "central_diff_slope_nats_per_h_at_073": slope1_73,
            "chord_slope_nats_per_h_full_grid": chord1,
            "max_abs_sum_over_grid_nats": max_abs_sum1,
            "verdict": verdict_proj1,
        },
    }
    print(
        f"{venue}: proj0 central-diff@0.73={slope0_73:.4f} nats/h, chord={chord0:.4f} "
        f"nats/h -> {verdict_proj0}",
        flush=True,
    )
    print(
        f"{venue}: proj1 central-diff@0.73={slope1_73:.4f} nats/h, chord={chord1:.4f} "
        f"nats/h -> {verdict_proj1}",
        flush=True,
    )

print(json.dumps(results, indent=2))
with open(
    "/home/jasper/Repositories/MasterThesisCode/.planning/derivation-gfrac-20260805/"
    "n2_m2_2d_inertness_results.json",
    "w",
) as f:
    json.dump(results, f, indent=2)
