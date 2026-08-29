"""[HIER] θ-hook (C1+C2) regression gates.

PHYSICS_CHANGE_THETA_HOOK_20260828.md (author-approved, ledger row #216), as
realigned by the appended note 2026-08-29 (row #221 item 4, charter node
B6.1): θ = (b, s) reparametrizes the host-z kernel at sites 2.1/2.2/2.3 as
``z̃ = z + b(1+z)``, ``host_z_error_eff = sqrt((s·host_z_error_raw)**2 +
sigma_z_pv**2)`` — s scales the RAW catalogue error BEFORE the
peculiar-velocity quadrature fold; b is unchanged, still shifting the kernel
centre AFTER the fold, with sigma_z_pv computed from the UNSHIFTED host
redshift. A literal skip at θ = (0, 1) (GATE T-ID) is unaffected. The OLD
(2026-08-28, "s scales the folded width") and NEW forms are bit-identical
whenever ``SIGMA_V_PEC_KM_S == 0.0`` (today's value) — see the s-placement
discriminator tests below, which patch that constant nonzero to actually
exercise the distinction.

Gates encoded here:
1. Pre-change value pins — the production-default path reproduces values
   captured at the pre-hook HEAD, bit-for-bit (the byte-identity regression).
2. θ = (0, 1) explicit == default, bit-for-bit (scalar, batch, smeared 2.3).
3. θ engaged == the pre-hook kernel called on substituted inputs
   (z̃, s·σ_z) — exact today because ``SIGMA_V_PEC_KM_S == 0.0`` makes the
   PV-fold ordering moot (precondition asserted, not assumed).
4. θ engaged actually moves the result (guards compute-then-discard).
5. PA-HIER-11 twin-parity tripwire: the shared kernel expressions are
   verbatim-identical between the production kernel and the
   integration-testing twin (site 2.7, deliberately NOT θ-parameterized).
6. Production-default guard: evaluate()'s θ defaults are (0.0, 1.0, "all").
7. HIER §1.2 s-placement discriminators (row #221 item 4): nonzero-σ_pv
   tests that only pass against the NEW (pre-fold) closed form, plus a
   b-order pin guarding against the z̃-in-sigma_z_pv defect flagged in the
   s-placement review of the appended note.
"""

import inspect
from typing import Any

import numpy as np
import pytest

import darksiren_emri.bayesian_inference.bayesian_statistics as bs
from darksiren_emri_test.bayesian_inference.test_kernel_parity import (
    _case_grid,
    _install_worker_globals,
)

_HOST_KEYS = ["host_phiS", "host_qS", "host_z", "host_z_error", "host_M", "host_M_error"]

# Captured at pre-hook HEAD e6278c16 (2026-08-28), repr-exact.
_PRECHANGE_PINS: dict[str, list[float]] = {
    "near_photoz_match_vd_3d": [
        528.3360821816322,
        0.9021532232745918,
        0.0,
        0.009515991992022478,
    ],
    "near_photoz_match_vd_4d": [
        528.3360821816322,
        0.9021532232745918,
        1484.968959304299,
        0.9334751611278179,
        0.0,
        0.009515991992022478,
    ],
    "near_offset_lr_3d": [246.92638172121093, 0.9282202496520747, 0.0, 0.0],
}

_THETA_ENGAGED = (0.02, 1.4142)


def _batch_call(kw: dict[str, Any], hosts: list[dict[str, float]], **extra: Any) -> np.ndarray:
    arrays = {k: np.array([hv[k] for hv in hosts], dtype=np.float64) for k in _HOST_KEYS}
    return bs.single_host_likelihood_batch(
        arrays["host_phiS"],
        arrays["host_qS"],
        arrays["host_z"],
        arrays["host_z_error"],
        arrays["host_M"],
        arrays["host_M_error"],
        detection_index=kw["detection_index"],
        h=kw["h"],
        evaluate_with_bh_mass=kw["evaluate_with_bh_mass"],
        normalization_mode=kw["normalization_mode"],
        **extra,
    )


@pytest.mark.parametrize("case_id", sorted(_PRECHANGE_PINS.keys()))
def test_default_path_pins_prechange_values(case_id: str) -> None:
    """The default (no-θ) scalar kernel reproduces pre-hook values bit-for-bit."""
    _install_worker_globals()
    kw = _case_grid()[case_id]
    got = bs.single_host_likelihood(**kw)
    assert got == _PRECHANGE_PINS[case_id]


@pytest.mark.parametrize("case_id", sorted(_PRECHANGE_PINS.keys()))
def test_theta_identity_bit_equality_scalar(case_id: str) -> None:
    """Explicit θ = (0, 1) takes the literal-skip path: bit-equal to default."""
    _install_worker_globals()
    kw = _case_grid()[case_id]
    assert bs.single_host_likelihood(**kw, theta_b=0.0, theta_s=1.0) == bs.single_host_likelihood(
        **kw
    )


def test_theta_identity_bit_equality_batch() -> None:
    _install_worker_globals()
    kw = _case_grid()["near_photoz_match_vd_3d"]
    hosts = [{k: float(kw[k]) for k in _HOST_KEYS}]
    hosts.append({**hosts[0], "host_z": hosts[0]["host_z"] * 1.05})
    default = _batch_call(kw, hosts)
    ident = _batch_call(kw, hosts, theta_b=0.0, theta_s=1.0)
    assert (default == ident).all()


def _substituted(kw: dict[str, Any], b: float, s: float) -> dict[str, Any]:
    """The pre-hook kernel's inputs after the registered substitution.

    Exact only while ``SIGMA_V_PEC_KM_S == 0`` (σ_eff == σ_z): then
    s·σ_eff == sqrt((s·σ_z)² + 0) and the b-after-PV-fold pin is moot.
    """
    z = float(kw["host_z"])
    out = dict(kw)
    out["host_z"] = z + b * (1.0 + z)
    out["host_z_error"] = s * float(kw["host_z_error"])
    return out


@pytest.mark.parametrize("case_id", ["near_photoz_match_vd_3d", "near_photoz_match_vd_4d"])
def test_theta_engaged_equals_substituted_inputs_scalar(case_id: str) -> None:
    """θ = (b, s) == the un-hooked kernel on (z̃, s·σ_z) — the closed form."""
    assert bs.SIGMA_V_PEC_KM_S == 0.0, "closed form needs updating for nonzero PV"
    _install_worker_globals()
    kw = _case_grid()[case_id]
    b, s = _THETA_ENGAGED
    hooked = np.array(bs.single_host_likelihood(**kw, theta_b=b, theta_s=s))
    subst = np.array(bs.single_host_likelihood(**_substituted(kw, b, s)))
    np.testing.assert_allclose(hooked, subst, rtol=1e-12, atol=0.0)


def test_theta_engaged_changes_result_scalar_and_batch() -> None:
    """A non-identity θ must move the numbers (no compute-then-discard)."""
    _install_worker_globals()
    kw = _case_grid()["near_photoz_match_vd_3d"]
    b, s = _THETA_ENGAGED
    default = np.array(bs.single_host_likelihood(**kw))
    hooked = np.array(bs.single_host_likelihood(**kw, theta_b=b, theta_s=s))
    assert not np.array_equal(default, hooked)
    hosts = [{k: float(kw[k]) for k in _HOST_KEYS}]
    default_b = _batch_call(kw, hosts)
    hooked_b = _batch_call(kw, hosts, theta_b=b, theta_s=s)
    assert not np.array_equal(default_b, hooked_b)


def test_smeared_site23_identity_and_substitution() -> None:
    """Site 2.3: θ identity is bit-equal; θ engaged matches substituted inputs."""
    assert bs.SIGMA_V_PEC_KM_S == 0.0
    _install_worker_globals()
    z_g = np.array([0.10, 0.25, 0.60])
    M_g = np.array([3.0e5, 1.0e6, 5.0e5])
    z_err = np.array([0.0015, 0.03, 0.05])
    common = dict(
        h=0.73,
        detection_probability_obj=bs.detection_probability,
        with_bh_mass=False,
        sky_aware=False,
    )
    default = bs._smeared_global_pdet_expectation(z_g, M_g, z_err, None, **common)
    ident = bs._smeared_global_pdet_expectation(
        z_g, M_g, z_err, None, theta_b=0.0, theta_s=1.0, **common
    )
    assert (default == ident).all()
    b, s = _THETA_ENGAGED
    hooked = bs._smeared_global_pdet_expectation(
        z_g, M_g, z_err, None, theta_b=b, theta_s=s, **common
    )
    subst = bs._smeared_global_pdet_expectation(
        z_g + b * (1.0 + z_g), M_g, s * z_err, None, **common
    )
    np.testing.assert_allclose(hooked, subst, rtol=1e-12, atol=0.0)
    assert not np.array_equal(default, hooked)


def test_theta_hook_counters_increment() -> None:
    """PA-HIER-16 hook-inventory corroborant: engaged sites count invocations."""
    _install_worker_globals()
    kw = _case_grid()["near_photoz_match_vd_3d"]
    before = dict(bs._THETA_HOOK_COUNTERS)
    bs.single_host_likelihood(**kw, theta_b=0.01, theta_s=1.0)
    assert bs._THETA_HOOK_COUNTERS["site_2_1"] == before["site_2_1"] + 1
    bs.single_host_likelihood(**kw)
    assert bs._THETA_HOOK_COUNTERS["site_2_1"] == before["site_2_1"] + 1


def test_twin_parity_expressions_verbatim() -> None:
    """PA-HIER-11 tripwire: the shared kernel expression is shape-identical in
    the production scalar kernel and the site-2.7 twin (which stays un-hooked).
    If the hook ever refactors the shared expression, the twin must be updated
    in the same commit — this test forces that conversation."""
    twin_src = inspect.getsource(bs.single_host_likelihood_integration_testing)
    prod_src = inspect.getsource(bs.single_host_likelihood)
    assert "* SIGMA_V_PEC_KM_S / SPEED_OF_LIGHT_KM_S" in twin_src
    assert "* SIGMA_V_PEC_KM_S / SPEED_OF_LIGHT_KM_S" in prod_src
    assert "theta_b" not in twin_src  # site 2.7 is deliberately NOT θ-hooked


def test_evaluate_theta_defaults() -> None:
    """Production-default guard: θ params default to the literal identity."""
    sig = inspect.signature(bs.BayesianStatistics.evaluate)
    assert sig.parameters["theta_b"].default == 0.0
    assert sig.parameters["theta_s"].default == 1.0
    assert sig.parameters["theta_sites"].default == "all"


def test_theta_validation_errors() -> None:
    """Guard pattern, not silent no-ops: bad θ inputs raise."""
    _install_worker_globals()
    kw = _case_grid()["near_photoz_match_vd_3d"]
    with pytest.raises(ValueError):
        bs.single_host_likelihood(**kw, theta_b=0.0, theta_s=0.0)
    with pytest.raises(ValueError):
        bs.single_host_likelihood(**kw, theta_b=0.0, theta_s=-1.0)


# ---------------------------------------------------------------------------
# HIER §1.2 s-placement alignment (row #221 item 4; appended note 2026-08-29
# in PHYSICS_CHANGE_THETA_HOOK_20260828.md, superseding the 2026-08-28
# "s scales the folded width" pin). These are the "nonzero-σ_pv discriminator"
# tests specified in the note's §6(b): every pin above holds bit-identically
# under BOTH the OLD and NEW s-placement whenever SIGMA_V_PEC_KM_S == 0.0 (the
# note's limiting case 1), so they cannot exercise this change. These tests
# monkeypatch SIGMA_V_PEC_KM_S nonzero and only pass against the NEW
# (pre-fold) closed form: ``host_z_error_eff = sqrt((s*host_z_error_raw)**2 +
# sigma_z_pv**2)``, with sigma_z_pv computed from the UNSHIFTED host redshift
# (b's placement is unchanged by this note).
# ---------------------------------------------------------------------------

_NONZERO_SIGMA_V = 200.0  # km/s; matches the removed runtime-addition magnitude
# cited at constants.py:90-94.


def test_theta_s_placement_old_new_forms_diverge() -> None:
    """Sanity check on the discriminator itself: OLD (post-fold) and NEW
    (pre-fold) closed forms for host_z_error_eff must actually differ once
    SIGMA_V_PEC_KM_S != 0 and theta_s != 1 — otherwise the tests below would
    pass vacuously regardless of which form the code implements."""
    host_z = 0.10
    host_z_error = 0.03
    s = 1.4142
    sigma_z_pv = (1.0 + host_z) * _NONZERO_SIGMA_V / bs.SPEED_OF_LIGHT_KM_S
    old_form = s * float(np.sqrt(host_z_error**2 + sigma_z_pv**2))
    new_form = float(np.sqrt((s * host_z_error) ** 2 + sigma_z_pv**2))
    assert not np.isclose(old_form, new_form, rtol=1e-6)


def test_theta_s_placement_prefold_scalar(monkeypatch: pytest.MonkeyPatch) -> None:
    """Site 2.1: s scales RAW host_z_error before the PV fold, not the folded
    width. Only passes against the NEW closed form (note §2)."""
    _install_worker_globals()
    kw = _case_grid()["near_photoz_match_vd_3d"]
    host_z = float(kw["host_z"])
    host_z_error = float(kw["host_z_error"])
    b, s = 0.0, 1.4142

    monkeypatch.setattr(bs, "SIGMA_V_PEC_KM_S", _NONZERO_SIGMA_V)
    sigma_z_pv = (1.0 + host_z) * _NONZERO_SIGMA_V / bs.SPEED_OF_LIGHT_KM_S
    new_eff = float(np.sqrt((s * host_z_error) ** 2 + sigma_z_pv**2))
    hooked = np.array(bs.single_host_likelihood(**kw, theta_b=b, theta_s=s))

    # Equivalent no-hook call: with SIGMA_V_PEC_KM_S == 0.0 the PV fold is a
    # no-op, so feeding host_z_error_eff in directly as host_z_error
    # reproduces the NEW closed form exactly, with theta left at identity.
    monkeypatch.setattr(bs, "SIGMA_V_PEC_KM_S", 0.0)
    equiv_kw = dict(kw)
    equiv_kw["host_z_error"] = new_eff
    equivalent = np.array(bs.single_host_likelihood(**equiv_kw, theta_b=0.0, theta_s=1.0))
    np.testing.assert_allclose(hooked, equivalent, rtol=1e-9, atol=0.0)


def test_theta_s_placement_prefold_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Site 2.2 (batch): same discriminator as 2.1, vectorized."""
    _install_worker_globals()
    kw = _case_grid()["near_photoz_match_vd_3d"]
    b, s = 0.0, 1.4142
    hosts = [{k: float(kw[k]) for k in _HOST_KEYS}]
    hosts.append({**hosts[0], "host_z": hosts[0]["host_z"] * 1.05})

    monkeypatch.setattr(bs, "SIGMA_V_PEC_KM_S", _NONZERO_SIGMA_V)
    hooked = _batch_call(kw, hosts, theta_b=b, theta_s=s)

    monkeypatch.setattr(bs, "SIGMA_V_PEC_KM_S", 0.0)
    equiv_hosts = []
    for hv in hosts:
        z = float(hv["host_z"])
        sigma_z_pv = (1.0 + z) * _NONZERO_SIGMA_V / bs.SPEED_OF_LIGHT_KM_S
        new_eff = float(np.sqrt((s * hv["host_z_error"]) ** 2 + sigma_z_pv**2))
        equiv_hosts.append({**hv, "host_z_error": new_eff})
    equivalent = _batch_call(kw, equiv_hosts, theta_b=0.0, theta_s=1.0)
    np.testing.assert_allclose(hooked, equivalent, rtol=1e-9, atol=0.0)


def test_theta_s_placement_prefold_smeared_site23(monkeypatch: pytest.MonkeyPatch) -> None:
    """Site 2.3 (smeared global selection): same discriminator, array form."""
    _install_worker_globals()
    z_g = np.array([0.10, 0.25, 0.60])
    M_g = np.array([3.0e5, 1.0e6, 5.0e5])
    z_err = np.array([0.0015, 0.03, 0.05])
    s = 1.4142
    common = dict(
        h=0.73,
        detection_probability_obj=bs.detection_probability,
        with_bh_mass=False,
        sky_aware=False,
    )

    monkeypatch.setattr(bs, "SIGMA_V_PEC_KM_S", _NONZERO_SIGMA_V)
    hooked = bs._smeared_global_pdet_expectation(
        z_g, M_g, z_err, None, theta_b=0.0, theta_s=s, **common
    )

    monkeypatch.setattr(bs, "SIGMA_V_PEC_KM_S", 0.0)
    sigma_z_pv = (1.0 + z_g) * _NONZERO_SIGMA_V / bs.SPEED_OF_LIGHT_KM_S
    new_eff = np.sqrt((s * z_err) ** 2 + sigma_z_pv**2)
    equivalent = bs._smeared_global_pdet_expectation(
        z_g, M_g, new_eff, None, theta_b=0.0, theta_s=1.0, **common
    )
    np.testing.assert_allclose(hooked, equivalent, rtol=1e-9, atol=0.0)


def test_theta_b_order_unchanged_uses_raw_host_z_for_pv(monkeypatch: pytest.MonkeyPatch) -> None:
    """Guard against the specific defect flagged in the s-placement review:
    the appended note's own NEW-formula text uses z̃ (post-b-shift) inside
    sigma_z_pv, while its prose claims 'b still shifts the centre; only the
    fold ORDER for s moves'. This implementation follows the prose (b's order
    is genuinely unchanged: sigma_z_pv is computed from the RAW host_z, and
    the b-shift is applied to the centre only, after sigma_z_pv is folded in).
    This test pins that choice: with theta_s == 1.0 (isolating b) and
    SIGMA_V_PEC_KM_S != 0, the raw-host_z and shifted-z̃ formulas for
    host_z_error_eff must differ, and the hooked call must match the
    RAW-host_z (unchanged-b-order) form, not the z̃-based one."""
    _install_worker_globals()
    kw = _case_grid()["near_photoz_match_vd_3d"]
    host_z = float(kw["host_z"])
    host_z_error = float(kw["host_z_error"])
    b = 0.02
    z_tilde = host_z + b * (1.0 + host_z)

    raw_pv = (1.0 + host_z) * _NONZERO_SIGMA_V / bs.SPEED_OF_LIGHT_KM_S
    shifted_pv = (1.0 + z_tilde) * _NONZERO_SIGMA_V / bs.SPEED_OF_LIGHT_KM_S
    eff_raw = float(np.sqrt(host_z_error**2 + raw_pv**2))
    eff_shifted = float(np.sqrt(host_z_error**2 + shifted_pv**2))
    assert not np.isclose(eff_raw, eff_shifted, rtol=1e-6), (
        "discriminator is vacuous: raw- and shifted-z pv forms coincide"
    )

    monkeypatch.setattr(bs, "SIGMA_V_PEC_KM_S", _NONZERO_SIGMA_V)
    hooked = np.array(bs.single_host_likelihood(**kw, theta_b=b, theta_s=1.0))

    monkeypatch.setattr(bs, "SIGMA_V_PEC_KM_S", 0.0)
    equiv_kw = dict(kw)
    equiv_kw["host_z"] = z_tilde
    equiv_kw["host_z_error"] = eff_raw
    equivalent = np.array(bs.single_host_likelihood(**equiv_kw, theta_b=0.0, theta_s=1.0))
    np.testing.assert_allclose(hooked, equivalent, rtol=1e-9, atol=0.0)
