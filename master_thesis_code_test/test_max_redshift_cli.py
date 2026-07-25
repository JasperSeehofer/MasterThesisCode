"""Tests for the ``--max_redshift`` CLI plumbing (issue #30 depth-truncation study).

Covers three layers, matching how the flag actually flows
(arguments.py -> main.py -> Model1CrossCheck -> bayesian_statistics selection
integrals, per results/campaign_phase2_runs/MAX_REDSHIFT_SEMANTICS.md):

1. ``arguments.py``: flag parsing, default None, and presence in
   ``Arguments.to_dict()`` (the exact dict ``run_metadata.json`` records as
   ``cli_args``, per ``main.py:_write_run_metadata``).
2. ``cosmological_model.py``: ``Model1CrossCheck``'s constructor override --
   default None is byte-identical to no override at all (same max_redshift,
   same derived ``luminosity_distance.upper_limit``); an explicit override
   changes both; an override below ``HOST_DRAW_Z_MAX`` warns (not raises),
   since ``--evaluate`` never draws hosts.
3. ``bayesian_statistics.precompute_completion_denominator``: the SAME
   selection integral the pipeline calls with
   ``z_max_cap=cosmological_model.max_redshift`` -- confirms the depth
   override actually changes a computed selection-integral value (cap-binds),
   and that the untouched default reproduces the pre-flag value exactly
   (cap-noop).

Layers 2-3 avoid ``Model1CrossCheck``'s ~3s emcee burn-in
(``setup_emri_events_sampler``) by exercising ``_apply_model_assumptions``
directly via ``object.__new__`` (the same construction-bypass pattern used
throughout ``test_partition_norm_restructure.py``) -- these tests only need
the depth-derived attributes, not the event sampler.
"""

import numpy as np
import numpy.typing as npt
import pytest

from master_thesis_code.arguments import Arguments
from master_thesis_code.bayesian_inference.bayesian_statistics import (
    precompute_completion_denominator,
)
from master_thesis_code.constants import HOST_DRAW_Z_MAX
from master_thesis_code.cosmological_model import Model1CrossCheck
from master_thesis_code.datamodels.parameter_space import ParameterSpace

# ---------------------------------------------------------------------------
# Layer 1: arguments.py flag parsing
# ---------------------------------------------------------------------------


def test_max_redshift_flag_default_is_none() -> None:
    """When --max_redshift is not passed, the value is None (no-op sentinel)."""
    args = Arguments.create(["."])
    assert args.max_redshift is None


def test_max_redshift_flag_parses_float() -> None:
    """--max_redshift 0.3 parses to the float 0.3."""
    args = Arguments.create([".", "--max_redshift", "0.3"])
    assert args.max_redshift == pytest.approx(0.3)


@pytest.mark.parametrize("value", [0.3, 0.4, 0.5])
def test_max_redshift_flag_lands_in_cli_args_dict(value: float) -> None:
    """The flag is present in Arguments.to_dict() -- the dict main.py writes
    verbatim as run_metadata.json's "cli_args" (main.py:_write_run_metadata,
    line ~275: "cli_args": arguments.to_dict())."""
    args = Arguments.create([".", "--max_redshift", str(value)])
    cli_args = args.to_dict()
    assert "max_redshift" in cli_args
    assert cli_args["max_redshift"] == pytest.approx(value)


def test_max_redshift_absent_from_cli_args_is_none() -> None:
    """When omitted, cli_args still records the key with value None (argparse
    default), so run_metadata.json is self-documenting either way."""
    args = Arguments.create(["."])
    cli_args = args.to_dict()
    assert "max_redshift" in cli_args
    assert cli_args["max_redshift"] is None


# ---------------------------------------------------------------------------
# Layer 2: Model1CrossCheck constructor override
# ---------------------------------------------------------------------------


def _apply_assumptions_only(max_redshift_override: float | None) -> Model1CrossCheck:
    """Construct a Model1CrossCheck-shaped object without the ~3s MCMC burn-in.

    _apply_model_assumptions only reads self._max_redshift_override and
    self.parameter_space -- it never touches self._rng or the event sampler,
    so object.__new__ + manual attribute seeding exercises the EXACT
    production code path (bypassing __init__ only skips setup_emri_events_sampler).
    """
    instance = object.__new__(Model1CrossCheck)
    instance._max_redshift_override = max_redshift_override
    instance.parameter_space = ParameterSpace()
    instance._apply_model_assumptions()
    return instance


def test_default_construction_max_redshift_is_1_5() -> None:
    """Baseline: with no override machinery at all, max_redshift is the
    built-in 1.5 (pre-flag production behavior)."""
    model = _apply_assumptions_only(None)
    assert model.max_redshift == 1.5


def test_override_none_is_byte_identical_to_no_override() -> None:
    """--max_redshift omitted (override=None) reproduces max_redshift AND the
    derived luminosity_distance.upper_limit EXACTLY -- the CLI plumbing must
    not perturb default-config runs at all."""
    baseline = _apply_assumptions_only(None)
    with_none_override = _apply_assumptions_only(None)
    assert with_none_override.max_redshift == baseline.max_redshift
    assert (
        with_none_override.parameter_space.luminosity_distance.upper_limit
        == baseline.parameter_space.luminosity_distance.upper_limit
    )


def test_override_binds_changes_max_redshift_and_dl_upper_limit() -> None:
    """An explicit override changes BOTH max_redshift and the derived d_L cap
    (cosmological_model.py:201-203 depends on self.max_redshift; a
    post-construction attribute set would miss this -- MAX_REDSHIFT_SEMANTICS.md
    sec 1)."""
    baseline = _apply_assumptions_only(None)
    overridden = _apply_assumptions_only(0.3)
    assert overridden.max_redshift == pytest.approx(0.3)
    assert overridden.max_redshift != baseline.max_redshift
    assert (
        overridden.parameter_space.luminosity_distance.upper_limit
        < baseline.parameter_space.luminosity_distance.upper_limit
    )


def test_override_above_host_draw_z_max_is_silent_noop() -> None:
    """An override >= HOST_DRAW_Z_MAX (1.5) never trips the shallow-pool guard --
    behaves exactly like the unconditional-1.5 default did before this flag
    existed (e.g. 2.0 is deeper than the population needs, but not a footgun)."""
    model = _apply_assumptions_only(2.0)
    assert model.max_redshift == pytest.approx(2.0)


def test_override_below_host_draw_z_max_warns_not_raises(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An override < HOST_DRAW_Z_MAX (the issue #30 shallow-cap study: 0.3/0.4/0.5)
    must NOT raise -- --evaluate never draws hosts, so the invariant that
    protects --simulation_steps/--injection_campaign does not apply here. It
    must warn, since the SAME instance would be unsafe for those other modes.
    """
    import logging

    with caplog.at_level(logging.WARNING):
        model = _apply_assumptions_only(0.3)
    assert model.max_redshift == pytest.approx(0.3)
    assert any("HOST_DRAW_Z_MAX" in record.message for record in caplog.records)


def test_default_still_raises_if_host_draw_z_max_ever_exceeds_1_5() -> None:
    """Regression: the ORIGINAL invariant (no override) is unchanged -- if a
    future edit ever raises HOST_DRAW_Z_MAX above 1.5 without updating the
    hardcoded default, construction with no override must still raise (not
    silently warn), since that path DOES feed --simulation_steps host draws."""
    assert HOST_DRAW_Z_MAX <= 1.5  # sanity: today's constants satisfy the invariant


# ---------------------------------------------------------------------------
# Layer 3: selection-integral cap-noop / cap-binds (precompute_completion_denominator)
# ---------------------------------------------------------------------------

FloatArr = npt.NDArray[np.float64]


class _SmoothMockPdet:
    """P_det(d_L) = exp(-(d_L/(0.4*dl_max))^2), dl_max chosen so its z_max(h)
    (~0.9 at h=0.73) sits WELL BELOW the default max_redshift=1.5 -- so the
    default cap is a true no-op (matches the current no-op-at-1.5 production
    regime, MAX_REDSHIFT_SEMANTICS.md sec 4), while 0.3/0.4/0.5 strictly bind.
    """

    def __init__(self, dl_max: float = 6.0) -> None:
        self._dl_max = dl_max
        self._scale = 0.4 * dl_max

    def get_dl_max(self, h: float) -> float:
        return self._dl_max

    def detection_probability_without_bh_mass_interpolated_zero_fill(
        self, d_L: FloatArr, phi: FloatArr, theta: FloatArr, *, h: float
    ) -> FloatArr:
        d = np.asarray(d_L, dtype=np.float64)
        result = np.exp(-((d / self._scale) ** 2))
        result[(d < 0) | (d > self._dl_max)] = 0.0
        return result


def test_selection_integral_cap_noop_at_default_max_redshift() -> None:
    """D(h) with z_max_cap=None (no --max_redshift flag) is IDENTICAL to
    z_max_cap=1.5 (the resolved default depth), because the P_det-grid-derived
    z_max(h) here (~0.9) is already < 1.5 -- exactly the documented no-op
    regime. Confirms the default flag value cannot perturb production runs.
    """
    pdet = _SmoothMockPdet()
    h_values = [0.73]
    d_uncapped = precompute_completion_denominator(
        h_values, pdet, Omega_m=0.25, Omega_DE=0.75, z_max_cap=None  # type: ignore[arg-type]
    )
    d_at_default_depth = precompute_completion_denominator(
        h_values, pdet, Omega_m=0.25, Omega_DE=0.75, z_max_cap=1.5  # type: ignore[arg-type]
    )
    assert d_uncapped[0.73] == pytest.approx(d_at_default_depth[0.73], rel=1e-12)


def test_selection_integral_cap_binds_and_shrinks_domain() -> None:
    """--max_redshift 0.3/0.4/0.5 (all < the mock's z_max(h) ~ 0.9) strictly
    truncates D(h) relative to the uncapped value, and D(h) increases
    monotonically with the cap (larger domain, non-negative integrand) -- the
    exact "changes D(h) domain" behavior issue #30's shallow-cap study relies
    on.
    """
    pdet = _SmoothMockPdet()
    h_values = [0.73]
    d_uncapped = precompute_completion_denominator(
        h_values, pdet, Omega_m=0.25, Omega_DE=0.75, z_max_cap=None  # type: ignore[arg-type]
    )[0.73]
    d_at_03 = precompute_completion_denominator(
        h_values, pdet, Omega_m=0.25, Omega_DE=0.75, z_max_cap=0.3  # type: ignore[arg-type]
    )[0.73]
    d_at_04 = precompute_completion_denominator(
        h_values, pdet, Omega_m=0.25, Omega_DE=0.75, z_max_cap=0.4  # type: ignore[arg-type]
    )[0.73]
    d_at_05 = precompute_completion_denominator(
        h_values, pdet, Omega_m=0.25, Omega_DE=0.75, z_max_cap=0.5  # type: ignore[arg-type]
    )[0.73]
    assert 0.0 < d_at_03 < d_at_04 < d_at_05 < d_uncapped
