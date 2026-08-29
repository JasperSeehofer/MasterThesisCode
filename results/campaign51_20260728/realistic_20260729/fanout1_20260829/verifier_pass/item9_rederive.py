"""Independent re-derivation for B6.1 [ALIGN] — verifier item 9/20.

Does NOT import or trust the assertions in test_theta_hook.py. Only reuses
its fixture helpers (_install_worker_globals, _case_grid) to get realistic
host/worker data, matching how the actual site-2.1/2.2/2.3 functions are
called in production. All closed-form arithmetic and pass/fail judgment
below is written independently from the registered formulas:
  - PREREGISTRATION_HIER_HTHETA_20260826.md sec 1.2
  - PHYSICS_CHANGE_THETA_HOOK_20260828.md "Appended note 2026-08-29" sec 2

Checks, at sigma_pv=200 km/s (SIGMA_V_PEC_KM_S monkeypatched), theta_s=1.4142,
theta_b=0.0, whether the production single_host_likelihood / _batch /
_smeared_global_pdet_expectation match the REGISTERED "s scales RAW error
before the PV fold" closed form (NEW), and whether they DIFFER from the
"s scales the folded width" (OLD, 2026-08-28) closed form -- confirming both
that the discriminator is non-vacuous and that current code implements NEW.

Also independently checks the b-order primary-finding resolution: with
theta_b != 0 and SIGMA_V_PEC_KM_S != 0, does the code compute sigma_z_pv from
the RAW host_z (prose / original 2026-08-28 pin) or from z-tilde (the note's
own Sec 2 formula literal)?
"""

import sys

import numpy as np

sys.path.insert(0, ".")

import darksiren_emri.bayesian_inference.bayesian_statistics as bs  # noqa: E402
from darksiren_emri_test.bayesian_inference.test_kernel_parity import (  # noqa: E402
    _case_grid,
    _install_worker_globals,
)

SIGMA_PV_TEST = 200.0  # km/s
RTOL = 1e-9

results: dict[str, object] = {}


def old_form(host_z_error: float, host_z: float, s: float, sigma_v: float, c: float) -> float:
    """2026-08-28 OLD pin: s scales the FOLDED width."""
    sigma_z_pv = (1.0 + host_z) * sigma_v / c
    folded = np.sqrt(host_z_error**2 + sigma_z_pv**2)
    return float(s * folded)


def new_form(host_z_error: float, host_z: float, s: float, sigma_v: float, c: float) -> float:
    """Row #221 item 4 NEW form (registered HIER sec1.2): s scales RAW error
    before the fold; sigma_z_pv from RAW (unshifted) host_z."""
    sigma_z_pv = (1.0 + host_z) * sigma_v / c
    return float(np.sqrt((s * host_z_error) ** 2 + sigma_z_pv**2))


def new_form_ztilde(
    host_z_error: float, host_z: float, b: float, s: float, sigma_v: float, c: float
) -> float:
    """The NOTE's own Sec 2 FORMULA LITERAL (uses ztilde, post-b-shift, inside
    sigma_z_pv) -- the reading the builder explicitly did NOT implement."""
    z_tilde = host_z + b * (1.0 + host_z)
    sigma_z_pv = (1.0 + z_tilde) * sigma_v / c
    return float(np.sqrt((s * host_z_error) ** 2 + sigma_z_pv**2))


_install_worker_globals()
kw = _case_grid()["near_photoz_match_vd_3d"]
host_z = float(kw["host_z"])
host_z_error = float(kw["host_z_error"])
c = bs.SPEED_OF_LIGHT_KM_S

# ---- Check 0: constant byte-identity claim ----
results["SIGMA_V_PEC_KM_S_is_0.0_today"] = bool(bs.SIGMA_V_PEC_KM_S == 0.0)

# ---- Check 1: discriminator non-vacuous ----
old_v = old_form(host_z_error, host_z, 1.4142, SIGMA_PV_TEST, c)
new_v = new_form(host_z_error, host_z, 1.4142, SIGMA_PV_TEST, c)
results["old_vs_new_differ"] = bool(not np.isclose(old_v, new_v, rtol=1e-6))
results["old_value"] = old_v
results["new_value"] = new_v

# ---- Check 2: site 2.1 scalar production call vs NEW closed form, at
#      sigma_pv=200, theta_s=1.4142, theta_b=0.0. Cross-check via the
#      "equivalent no-hook call" method (independent of the test file): call
#      the production function with SIGMA_V_PEC_KM_S patched, and separately
#      call it with SIGMA_V_PEC_KM_S=0 feeding host_z_error_eff computed from
#      NEW form directly as host_z_error (valid because SIGMA_V_PEC_KM_S=0
#      makes the PV fold a no-op) -- this is a genuine closed-form check, not
#      trust in the test file.
b, s = 0.0, 1.4142
orig_sigma_v = float(bs.SIGMA_V_PEC_KM_S)
try:
    bs.SIGMA_V_PEC_KM_S = SIGMA_PV_TEST
    hooked_scalar = np.array(bs.single_host_likelihood(**kw, theta_b=b, theta_s=s))

    bs.SIGMA_V_PEC_KM_S = 0.0
    eff_new = new_form(host_z_error, host_z, s, SIGMA_PV_TEST, c)
    equiv_kw = dict(kw)
    equiv_kw["host_z_error"] = eff_new
    equiv_scalar_new = np.array(bs.single_host_likelihood(**equiv_kw, theta_b=0.0, theta_s=1.0))

    eff_old = old_form(host_z_error, host_z, s, SIGMA_PV_TEST, c)
    equiv_kw_old = dict(kw)
    equiv_kw_old["host_z_error"] = eff_old
    equiv_scalar_old = np.array(bs.single_host_likelihood(**equiv_kw_old, theta_b=0.0, theta_s=1.0))
finally:
    bs.SIGMA_V_PEC_KM_S = orig_sigma_v

match_new = bool(np.allclose(hooked_scalar, equiv_scalar_new, rtol=RTOL, atol=0.0))
match_old = bool(np.allclose(hooked_scalar, equiv_scalar_old, rtol=RTOL, atol=0.0))
results["site2_1_matches_NEW_form_rtol1e-9"] = match_new
results["site2_1_matches_OLD_form_rtol1e-9"] = match_old
results["site2_1_hooked"] = hooked_scalar.tolist()
results["site2_1_equiv_new"] = equiv_scalar_new.tolist()
results["site2_1_equiv_old"] = equiv_scalar_old.tolist()

# ---- Check 3: site 2.3 smeared, array form, independent of site 2.1 code path ----
z_g = np.array([0.10, 0.25, 0.60])
M_g = np.array([3.0e5, 1.0e6, 5.0e5])
z_err = np.array([0.0015, 0.03, 0.05])
common = dict(
    h=0.73,
    detection_probability_obj=bs.detection_probability,
    with_bh_mass=False,
    sky_aware=False,
)
try:
    bs.SIGMA_V_PEC_KM_S = SIGMA_PV_TEST
    hooked_23 = bs._smeared_global_pdet_expectation(
        z_g, M_g, z_err, None, theta_b=0.0, theta_s=s, **common
    )
    bs.SIGMA_V_PEC_KM_S = 0.0
    sigma_z_pv_arr = (1.0 + z_g) * SIGMA_PV_TEST / c
    new_eff_arr = np.sqrt((s * z_err) ** 2 + sigma_z_pv_arr**2)
    equiv_23 = bs._smeared_global_pdet_expectation(
        z_g, M_g, new_eff_arr, None, theta_b=0.0, theta_s=1.0, **common
    )
finally:
    bs.SIGMA_V_PEC_KM_S = orig_sigma_v

results["site2_3_matches_NEW_form_rtol1e-9"] = bool(
    np.allclose(hooked_23, equiv_23, rtol=RTOL, atol=0.0)
)

# ---- Check 4: b-order primary-finding resolution (RAW-z vs z-tilde in sigma_z_pv) ----
b2 = 0.02
try:
    bs.SIGMA_V_PEC_KM_S = SIGMA_PV_TEST
    hooked_b = np.array(bs.single_host_likelihood(**kw, theta_b=b2, theta_s=1.0))

    bs.SIGMA_V_PEC_KM_S = 0.0
    z_tilde = host_z + b2 * (1.0 + host_z)
    eff_raw_pv = float(
        np.sqrt(host_z_error**2 + ((1.0 + host_z) * SIGMA_PV_TEST / c) ** 2)
    )
    eff_ztilde_pv = float(
        np.sqrt(host_z_error**2 + ((1.0 + z_tilde) * SIGMA_PV_TEST / c) ** 2)
    )
    equiv_kw_raw = dict(kw)
    equiv_kw_raw["host_z"] = z_tilde
    equiv_kw_raw["host_z_error"] = eff_raw_pv
    equiv_b_raw = np.array(bs.single_host_likelihood(**equiv_kw_raw, theta_b=0.0, theta_s=1.0))

    equiv_kw_zt = dict(kw)
    equiv_kw_zt["host_z"] = z_tilde
    equiv_kw_zt["host_z_error"] = eff_ztilde_pv
    equiv_b_zt = np.array(bs.single_host_likelihood(**equiv_kw_zt, theta_b=0.0, theta_s=1.0))
finally:
    bs.SIGMA_V_PEC_KM_S = orig_sigma_v

results["b_order_matches_RAWz_reading_(prose/original-2026-08-28-pin)"] = bool(
    np.allclose(hooked_b, equiv_b_raw, rtol=RTOL, atol=0.0)
)
results["b_order_matches_ZTILDE_reading_(notes-own-sec2-formula-literal)"] = bool(
    np.allclose(hooked_b, equiv_b_zt, rtol=RTOL, atol=0.0)
)
results["eff_raw_pv"] = eff_raw_pv
results["eff_ztilde_pv"] = eff_ztilde_pv
results["raw_vs_ztilde_pv_forms_differ"] = bool(
    not np.isclose(eff_raw_pv, eff_ztilde_pv, rtol=1e-9)
)

for k, v in results.items():
    print(f"{k}: {v}")
