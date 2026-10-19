"""Item 13 (B8.1 [CAL] F5 information floor) — independent re-derivation.

End-of-fan-out verifier pass, ledger row #232 / registration item 13.

This script does NOT import or call anything from
fanout1_20260829/b8_information_floor.py. It re-implements the Fisher-floor
computation from first principles, straight off:
  - the raw CRB CSV (seed61000/prepared_cramer_rao_bounds.csv),
  - darksiren_emri.physical_relations.dist / dist_vectorized (production code),
  - darksiren_emri.constants (H, OMEGA_M, OMEGA_DE, SNR_THRESHOLD),
  - bayesian_statistics.py's own FRACTIONAL_LUMINOSITY_DISTANCE_ERROR_THRESHOLD.

Two independent checks:
  (1) EXPLICIT 2x2 FISHER MATRIX in (h, z), built term-by-term (not the
      pre-collapsed single-line formula in the target script), then Schur-
      complement it by hand. This directly tests the claim under dispute:
      "only the sigma_dL^2 term carries the h^2 Jacobian, not the sigma_z
      term" -- by deriving d(d_L)/dh and d(d_L)/dz via NUMERICAL
      differentiation of the actual dist_vectorized function (no analytic
      formula copied from the target script or the record).
  (2) A brute-force grid confirmation of the Route-A finite-difference
      instability, using a DIFFERENT method (direct 2nd-derivative of a
      finely sampled log-likelihood curve via np.gradient / polynomial fit,
      not a 3-point stencil at one dh) to check whether the reported
      Route-A "instability, not a bug" characterization holds up under a
      different numerical estimator of the same curvature.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO_ROOT))

from darksiren_emri.constants import H, OMEGA_DE, OMEGA_M, SNR_THRESHOLD  # noqa: E402
from darksiren_emri.physical_relations import dist_vectorized  # noqa: E402

H_TRUE = H
FRAC_DL_ERR_THRESHOLD = 0.10  # bayesian_statistics.py:386

CRB_PATH = (
    REPO_ROOT
    / "results/campaign51_20260728/realistic_20260729/seed61000/prepared_cramer_rao_bounds.csv"
)

print(f"[check] H_TRUE={H_TRUE}  OMEGA_M={OMEGA_M}  OMEGA_DE={OMEGA_DE}  SNR_THRESHOLD={SNR_THRESHOLD}")
assert H_TRUE == 0.73
assert OMEGA_M == 0.2726
assert OMEGA_DE == 0.7274
assert SNR_THRESHOLD == 20

# ---------------------------------------------------------------------------
# Step 0: independently load + filter the CRB CSV (own mask logic, not copy).
# ---------------------------------------------------------------------------
df = pd.read_csv(CRB_PATH)
n_raw = len(df)
d_meas = df["luminosity_distance"].to_numpy(dtype=np.float64)
sigma_dL = np.sqrt(df["delta_luminosity_distance_delta_luminosity_distance"].to_numpy(dtype=np.float64))
Mz_meas = df["M"].to_numpy(dtype=np.float64)
sigma_Mz = np.sqrt(df["delta_M_delta_M"].to_numpy(dtype=np.float64))
snr = df["SNR"].to_numpy(dtype=np.float64)
frac_err = sigma_dL / d_meas

keep = (snr >= SNR_THRESHOLD) & (frac_err < FRAC_DL_ERR_THRESHOLD)
n_events = int(keep.sum())
print(f"[check] raw rows={n_raw}  kept(SNR>=20 & frac_dL_err<10%)={n_events}")
assert n_events == 1588, f"EXPECTED 1588 events, got {n_events}"

d_meas = d_meas[keep]
sigma_dL = sigma_dL[keep]
Mz_meas = Mz_meas[keep]
sigma_Mz = sigma_Mz[keep]

# ---------------------------------------------------------------------------
# Step 1: recover z_i from d_meas at h_true via inversion of the PRODUCTION
# distance law (own interpolation table, independent resolution/range from
# the target script's d_to_z()).
# ---------------------------------------------------------------------------
z_table = np.linspace(1e-6, 2.5, 300_000)
d_table = dist_vectorized(z_table, h=H_TRUE)
z_i = np.interp(d_meas, d_table, z_table)
print(f"[check] z_i: median={np.median(z_i):.4f} p10={np.percentile(z_i,10):.4f} "
      f"p90={np.percentile(z_i,90):.4f} max={np.max(z_i):.4f}")
assert abs(np.median(z_i) - 0.4902) < 5e-4

# ---------------------------------------------------------------------------
# Step 2: EXPLICIT 2x2 Fisher matrix in (h, z), term by term, via numerical
# derivatives of dist_vectorized -- NOT the pre-collapsed formula.
#
# Model: d_L,model(z,h) = dist(z, h)   [call it directly, do NOT assume the
#        D(z)/h factorization -- let the derivative discover it]
#   Observable 1: d_meas ~ N(d_L,model(z,h), sigma_dL^2)
#   Observable 2: z_cat  ~ N(z, sigma_z^2)         [1D]
#   Observable 3 (2D only): M_z_meas ~ N(Mg*(1+z), sigma_Mz^2+(sigma_M*Mg*(1+z))^2)
#
# Fisher matrix elements at the true point:
#   F_hh = (d dL/dh)^2 / sigma_dL^2
#   F_hz = (d dL/dh)(d dL/dz) / sigma_dL^2
#   F_zz = (d dL/dz)^2 / sigma_dL^2 + 1/sigma_z_eff^2
# Profiled (Schur complement): I_h = F_hh - F_hz^2/F_zz
# ---------------------------------------------------------------------------
eps_h = 1e-6
eps_z = 1e-6
dL_plus_h = dist_vectorized(z_i, h=H_TRUE + eps_h)
dL_minus_h = dist_vectorized(z_i, h=H_TRUE - eps_h)
ddL_dh = (dL_plus_h - dL_minus_h) / (2 * eps_h)

dL_plus_z = dist_vectorized(z_i + eps_z, h=H_TRUE)
dL_minus_z = dist_vectorized(z_i - eps_z, h=H_TRUE)
ddL_dz = (dL_plus_z - dL_minus_z) / (2 * eps_z)

# Sanity check on the claimed factorization d_L=D(z)/h => ddL/dh = -d_L/h
d_L_at_truth = dist_vectorized(z_i, h=H_TRUE)
implied_ddL_dh_from_factorization = -d_L_at_truth / H_TRUE
rel_diff = np.abs(ddL_dh - implied_ddL_dh_from_factorization) / np.abs(implied_ddL_dh_from_factorization)
print(f"[check] d(dL)/dh numeric vs -dL/h (factorization) : max rel diff = {rel_diff.max():.3e}")
assert rel_diff.max() < 1e-6, "factorization d_L=D(z)/h not confirmed by direct numerical derivative"

# This ALSO directly tests the disputed h^2 Jacobian claim:
#   F_hh = (ddL/dh)^2 / sigma_dL^2 = (dL/h)^2/sigma_dL^2   -- matches record's
#   F_hz = (ddL/dh)(ddL/dz)/sigma_dL^2 = -(dL/h)(D'/h)/sigma_dL^2
#   F_zz = (ddL/dz)^2/sigma_dL^2 + 1/sigma_z_eff^2 = (D'/h)^2/sigma_dL^2 + ...
# Note F_zz's GW term ALSO carries a 1/h^2 factor (since ddL/dz = D'(z)/h),
# not a bare h^2 -- confirming the record's claim that "only the sigma_dL^2
# TERM carries the Jacobian" is really about which term has 1/sigma_dL^2 vs
# 1/sigma_z^2, and that H_TRUE^2 multiplies sigma_dL^2 in the DENOMINATOR of
# I_i, not sigma_z^2. This is what we check numerically below.


def sigma_z_eff2(sigma_z: float, sigma_M: float | None, z: np.ndarray, one_d: bool) -> np.ndarray:
    if one_d or sigma_M is None:
        return np.full_like(z, sigma_z**2)
    inv = 1.0 / sigma_z**2 + 1.0 / (sigma_M * (1.0 + z)) ** 2
    return 1.0 / inv


def explicit_schur_fisher(sigma_z: float, sigma_M: float | None, one_d: bool) -> np.ndarray:
    F_hh = ddL_dh**2 / sigma_dL**2
    F_hz = (ddL_dh * ddL_dz) / sigma_dL**2
    sz2 = sigma_z_eff2(sigma_z, sigma_M, z_i, one_d)
    F_zz = ddL_dz**2 / sigma_dL**2 + 1.0 / sz2
    I_h = F_hh - F_hz**2 / F_zz
    return I_h


def floor_from_I(I: np.ndarray) -> float:
    return float(1.0 / np.sqrt(np.sum(I)))


SIGMA_Z_GLADE_PHOTO = 0.035
SIGMA_Z_GLADE_SPEC = 0.0017
SIGMA_M_REALISTIC = 1.99  # 0.55 dex total predictive (R&V15)
SIGMA_M_F5_THRESHOLD = 0.02

I_1d_photo = explicit_schur_fisher(SIGMA_Z_GLADE_PHOTO, None, one_d=True)
floor_1d_photo = floor_from_I(I_1d_photo)
print(f"\n[RESULT] 1D closed-form floor, sigma_z=0.035 (independent Schur derivation): "
      f"{floor_1d_photo:.6f}  ({100*floor_1d_photo/H_TRUE:.4f}% of h)")

I_1d_spec = explicit_schur_fisher(SIGMA_Z_GLADE_SPEC, None, one_d=True)
floor_1d_spec = floor_from_I(I_1d_spec)
print(f"[RESULT] 1D closed-form floor, sigma_z=0.0017 (spec): {floor_1d_spec:.6f}")

I_2d_realistic = explicit_schur_fisher(SIGMA_Z_GLADE_PHOTO, SIGMA_M_REALISTIC, one_d=False)
floor_2d_realistic = floor_from_I(I_2d_realistic)
print(f"[RESULT] 2D closed-form floor, sigma_z=0.035, sigma_M=1.99 (realistic): "
      f"{floor_2d_realistic:.6f}")

I_2d_thresh = explicit_schur_fisher(SIGMA_Z_GLADE_PHOTO, SIGMA_M_F5_THRESHOLD, one_d=False)
floor_2d_thresh = floor_from_I(I_2d_thresh)
print(f"[RESULT] 2D closed-form floor, sigma_z=0.035, sigma_M=0.02 (F5 threshold, informational): "
      f"{floor_2d_thresh:.6f}")

# ---------------------------------------------------------------------------
# Also test the SPECIFIC bug the record discloses catching: multiplying
# BOTH terms of the denominator by H_TRUE^2 (the "earlier draft" slip).
# If we deliberately reintroduce that slip, how far off does it land, and
# does that match the record's own quoted ~1.9x overstatement claim?
# ---------------------------------------------------------------------------
def buggy_fisher_both_h2(sigma_z: float, sigma_M: float | None, one_d: bool) -> np.ndarray:
    """Reproduce the DISCLOSED bug: denom = h^2*sigma_dL^2 + h^2*D'^2*sigma_z_eff^2."""
    Dprime = ddL_dz * H_TRUE  # D'(z) = dDdz where D(z)=dist(z,1); ddL_dz = D'/h => D' = ddL_dz*h
    sz2 = sigma_z_eff2(sigma_z, sigma_M, z_i, one_d)
    denom_buggy = (H_TRUE**2) * sigma_dL**2 + (H_TRUE**2) * Dprime**2 * sz2
    d_data = d_L_at_truth
    return d_data**2 / denom_buggy


I_buggy_1d = buggy_fisher_both_h2(SIGMA_Z_GLADE_PHOTO, None, one_d=True)
floor_buggy_1d = floor_from_I(I_buggy_1d)
print(f"\n[BUG-CHECK] 1D floor if BOTH terms wrongly carry h^2 (the disclosed slip): "
      f"{floor_buggy_1d:.6f}  vs corrected {floor_1d_photo:.6f}  "
      f"ratio(buggy/corrected)={floor_buggy_1d/floor_1d_photo:.4f}  "
      f"(record claims ~1/H_TRUE ~= {1/H_TRUE:.4f} in floor, i.e. ~1.9x in INFO overstatement)")

# ---------------------------------------------------------------------------
# Step 3: independent confirmation of the Route-A finite-difference
# instability, via a DIFFERENT numerical method: for the specific flagged
# event (idx 889 in the record, z~0.0213), scan log-likelihood over a dense
# h-grid directly and look at its local shape near h_true, rather than a
# 3-point stencil at one dh.
# ---------------------------------------------------------------------------
print("\n[Route-A cross-check] scanning marginal log-likelihood shape for shallow, tight events")

# Recompute z_i-sorted index to find very shallow (z<0.03) events among the kept set,
# independent of the record's claimed idx 889 (which is an index into the ORIGINAL,
# unfiltered/unsorted CRB rows -- here we just confirm the MECHANISM: does the
# marginal log-likelihood near h_true look flat/monotonic, not peaked, for the
# shallowest events under GLADE photo-z?)
shallow_mask = z_i < 0.03
n_shallow = int(shallow_mask.sum())
print(f"[Route-A cross-check] events with z<0.03: {n_shallow} / {n_events}")

sigma_z = SIGMA_Z_GLADE_PHOTO
h_grid = np.linspace(H_TRUE - 0.05, H_TRUE + 0.05, 401)


def marginal_logL_1d(idx: int, h_grid: np.ndarray, n_z: int = 2000) -> np.ndarray:
    z0 = z_i[idx]
    d_data = d_meas[idx]  # use the ACTUAL measured d_meas (not the noiseless self-consistent
    # point the target script's Route A uses) -- an intentionally different, more
    # conservative construction to see if the qualitative flatness finding survives.
    sdl = sigma_dL[idx]
    zmin = max(z0 - 7 * sigma_z, 1e-6)
    zmax = z0 + 7 * sigma_z
    zg = np.linspace(zmin, zmax, n_z)
    Dshape = dist_vectorized(zg, h=1.0)
    photo = -0.5 * ((zg - z0) / sigma_z) ** 2
    out = np.empty_like(h_grid)
    dz = zg[1] - zg[0]
    for k, h in enumerate(h_grid):
        dL_model = Dshape / h
        gw = -0.5 * ((d_data - dL_model) / sdl) ** 2
        integrand = photo + gw
        m = integrand.max()
        L = np.sum(np.exp(integrand - m)) * dz
        out[k] = np.log(max(L, 1e-300)) + m
    return out


shallow_idx = np.where(shallow_mask)[0]
if len(shallow_idx) > 0:
    test_idx = shallow_idx[np.argmin(sigma_dL[shallow_idx] / d_meas[shallow_idx])]  # tightest dL error
    logL = marginal_logL_1d(test_idx, h_grid)
    # curvature via a robust 2nd-derivative estimate: fit a local quadratic in a
    # window around h_true, independent of the single-dh 3-point stencil.
    i_true = np.argmin(np.abs(h_grid - H_TRUE))
    window = slice(max(0, i_true - 40), min(len(h_grid), i_true + 41))
    coeffs = np.polyfit(h_grid[window] - H_TRUE, logL[window], deg=2)
    curvature_quadfit = -2 * coeffs[0]  # d2logL/dh2 = 2*coeffs[0]
    # compare against a coarse window (full +-0.05) quadratic fit
    coeffs_wide = np.polyfit(h_grid - H_TRUE, logL, deg=2)
    curvature_wide = -2 * coeffs_wide[0]
    print(f"[Route-A cross-check] test event idx={test_idx}, z={z_i[test_idx]:.4f}, "
          f"frac_dL_err={sigma_dL[test_idx]/d_meas[test_idx]:.4f}")
    print(f"  logL range over h in [0.68,0.78]: min={logL.min():.3f} max={logL.max():.3f} "
          f"argmax_h={h_grid[np.argmax(logL)]:.4f}  (flat/monotonic would show argmax far from "
          f"H_TRUE={H_TRUE} or a very shallow range)")
    print(f"  local quadfit curvature (+-0.02 window): {curvature_quadfit:.3f}  "
          f"-> sigma_h from this event alone ~ {1/np.sqrt(max(curvature_quadfit,1e-12)):.4f}")
    print(f"  wide quadfit curvature (+-0.05 window): {curvature_wide:.3f}  "
          f"-> sigma_h from this event alone ~ {1/np.sqrt(max(curvature_wide,1e-12)):.4f}")
    print(f"  curvature ratio (wide/local) = {curvature_wide/curvature_quadfit if curvature_quadfit>0 else float('nan'):.3f} "
          f"-- large deviation from 1.0 confirms non-quadratic/degenerate local shape "
          f"(independent evidence for the record's Route-A instability claim)")

# ---------------------------------------------------------------------------
# Step 4: measured HEAD posterior comparison numbers (verbatim citation
# check only -- confirm the record's arithmetic, not re-measure the head
# readout itself, which is out of scope / a different source file).
# ---------------------------------------------------------------------------
measured_2d_sigma_h = 0.01847  # venue-average per record: (0.01833+0.01861)/2
measured_2d_bias = -0.0668  # venue-average per record: (-0.0666-0.0670)/2
computed_avg_sigma = (0.01833 + 0.01861) / 2
computed_avg_bias = (-0.0666 + -0.0670) / 2
print(f"\n[check] measured 2D sigma_h venue avg recompute: {computed_avg_sigma:.5f} "
      f"(record states {measured_2d_sigma_h})")
print(f"[check] measured 2D bias venue avg recompute: {computed_avg_bias:.5f} "
      f"(record states {measured_2d_bias})")
width_ratio = computed_avg_sigma / floor_2d_realistic
bias_ratio = abs(computed_avg_bias) / floor_2d_realistic
print(f"[check] width/floor = {width_ratio:.2f}x  (record states ~10.6x)")
print(f"[check] |bias|/floor = {bias_ratio:.2f}x  (record states ~38.2x)")

print("\n[SUMMARY]")
print(f"  1D floor (sz=0.035):            {floor_1d_photo:.6f}  vs record 0.001747")
print(f"  2D floor (sz=0.035, sM=1.99):    {floor_2d_realistic:.6f}  vs record 0.001747")
print(f"  2D floor (sz=0.035, sM=0.02):    {floor_2d_thresh:.6f}  vs record 0.001295")
print(f"  1D floor (sz=0.0017, spec):      {floor_1d_spec:.6f}  vs record 0.000560")
