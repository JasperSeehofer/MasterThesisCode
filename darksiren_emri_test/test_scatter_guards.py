"""Scattered-catalogue prior-consistency guards (campaign #53).

[PHYSICS] Realistic host-observation model, RATIFIED 2026-07-29
(docs/derivations/realistic_host_observation_model.md §3.4 / §9). The guard
set pinned here:

1. sigma_scale > 0 and a point-resolving host-z kernel => raise.
2. sigma_scale > 0 and normalization_mode == 'generator_marginal' => raise.
3. sigma_scale == 0 (or no realization at all) => every baseline mode stays
   permitted — the guard is ONE-DIRECTIONAL.

Plus the with-BH mass-channel counterpart, and the CLI-level refusal of
pairing an observed catalogue with a generative stage (convention (A)).
"""

import pytest

from darksiren_emri.arguments import Arguments
from darksiren_emri.bayesian_inference import bayesian_statistics as bs
from darksiren_emri.exceptions import ArgumentsError

# --------------------------------------------------------------------------
# Guard 1: point host-z kernel under scatter
# --------------------------------------------------------------------------


def test_point_kernel_is_refused_on_a_scattered_catalogue() -> None:
    with pytest.raises(ValueError, match="SCATTERED observed realization"):
        bs.resolve_host_z_kernel("point", "absolute_marginal", catalogue_scattered=True)


def test_auto_kernel_resolving_to_point_is_refused_under_scatter() -> None:
    """'auto' + generator_marginal resolves to 'point' — must also raise."""
    with pytest.raises(ValueError, match="SCATTERED observed realization"):
        bs.resolve_host_z_kernel("auto", "generator_marginal", catalogue_scattered=True)


def test_volume_deconv_kernel_is_permitted_under_scatter() -> None:
    """[RATIFY-R3]: the width kernel is the licensed pairing under scatter."""
    assert (
        bs.resolve_host_z_kernel("volume_deconv", "absolute_marginal", catalogue_scattered=True)
        == "volume_deconv"
    )
    assert (
        bs.resolve_host_z_kernel("auto", "absolute_marginal", catalogue_scattered=True)
        == "volume_deconv"
    )


# --------------------------------------------------------------------------
# Guard 3: one-directional — unscattered catalogues keep every baseline mode
# --------------------------------------------------------------------------


def test_point_kernel_stays_legal_on_an_unscattered_catalogue() -> None:
    assert bs.resolve_host_z_kernel("point", "absolute_marginal") == "point"
    assert bs.resolve_host_z_kernel("auto", "generator_marginal") == "point"
    assert (
        bs.resolve_host_z_kernel("point", "absolute_marginal", catalogue_scattered=False) == "point"
    )


def test_generator_marginal_stays_legal_on_an_unscattered_catalogue() -> None:
    # No raise for any baseline combination.
    bs.validate_scatter_guards(
        normalization_mode="generator_marginal",
        host_z_kernel="auto",
        host_mass_kernel="auto",
        catalogue_scattered=False,
    )
    bs.validate_scatter_guards(
        normalization_mode="absolute_marginal",
        host_z_kernel="point",
        host_mass_kernel="gaussian",
        catalogue_scattered=False,
    )


def test_default_kwarg_preserves_the_historical_resolver_behaviour() -> None:
    """No silent default-path change: resolution without the flag is unchanged."""
    assert bs.resolve_host_z_kernel("auto", "generator_marginal") == "point"
    assert bs.resolve_host_z_kernel("auto", "absolute_marginal") == "volume_deconv"
    assert bs.resolve_host_z_kernel("volume_deconv", "generator_marginal") == "volume_deconv"
    assert bs.resolve_host_mass_kernel("auto", "mass_trunc", "volume_deconv") == "trunc_lognormal"
    assert bs.resolve_host_mass_kernel("auto", "absolute_marginal", "auto") == "gaussian"


# --------------------------------------------------------------------------
# Guard 2: generator_marginal under scatter
# --------------------------------------------------------------------------


def test_generator_marginal_is_refused_on_a_scattered_catalogue() -> None:
    with pytest.raises(ValueError, match="generator_marginal.*refused"):
        bs.validate_scatter_guards(
            normalization_mode="generator_marginal",
            host_z_kernel="volume_deconv",
            host_mass_kernel="auto",
            catalogue_scattered=True,
        )


def test_validate_scatter_guards_refuses_point_kernel() -> None:
    with pytest.raises(ValueError, match="SCATTERED observed realization"):
        bs.validate_scatter_guards(
            normalization_mode="absolute_marginal",
            host_z_kernel="point",
            host_mass_kernel="auto",
            catalogue_scattered=True,
        )


def test_ratified_pairing_passes_the_guard_set() -> None:
    """[RATIFY-R3] production pairing: absolute_marginal x volume_deconv."""
    bs.validate_scatter_guards(
        normalization_mode="absolute_marginal",
        host_z_kernel="volume_deconv",
        host_mass_kernel="auto",
        catalogue_scattered=True,
    )
    bs.validate_scatter_guards(
        normalization_mode="absolute_marginal",
        host_z_kernel="volume_deconv",
        host_mass_kernel="trunc_lognormal",
        catalogue_scattered=True,
    )


# --------------------------------------------------------------------------
# With-BH mass channel
# --------------------------------------------------------------------------


def test_mass_kernel_guard_refuses_point_anchored_pairing_under_scatter() -> None:
    with pytest.raises(ValueError, match="SCATTERED observed realization"):
        bs.resolve_host_mass_kernel(
            "gaussian", "absolute_marginal", "point", catalogue_scattered=True
        )
    with pytest.raises(ValueError, match="SCATTERED observed realization"):
        bs.resolve_host_mass_kernel("auto", "generator_marginal", "auto", catalogue_scattered=True)


def test_mass_kernel_width_pairing_is_permitted_under_scatter() -> None:
    assert (
        bs.resolve_host_mass_kernel(
            "trunc_lognormal", "absolute_marginal", "volume_deconv", catalogue_scattered=True
        )
        == "trunc_lognormal"
    )
    assert (
        bs.resolve_host_mass_kernel(
            "gaussian", "absolute_marginal", "volume_deconv", catalogue_scattered=True
        )
        == "gaussian"
    )


def test_existing_point_z_x_trunc_mass_guard_still_fires_unscattered() -> None:
    """The pre-existing #40 guard is untouched."""
    with pytest.raises(ValueError, match="prior-inconsistent"):
        bs.resolve_host_mass_kernel("trunc_lognormal", "absolute_marginal", "point")


# --------------------------------------------------------------------------
# CLI-level guards (convention (A))
# --------------------------------------------------------------------------


def test_observed_catalogue_cannot_be_paired_with_a_generative_stage(
    tmp_path: object,
) -> None:
    arguments = Arguments.create(
        [".", "--observed_catalogue", "obs.csv", "--simulation_steps", "5"]
    )
    with pytest.raises(ArgumentsError, match="evaluation-side override only"):
        arguments.validate()


def test_realize_observed_catalogue_requires_an_explicit_seed() -> None:
    arguments = Arguments.create([".", "--realize_observed_catalogue"])
    with pytest.raises(ArgumentsError, match="requires an explicit"):
        arguments.validate()
    ok = Arguments.create([".", "--realize_observed_catalogue", "--realization_seed", "7"])
    ok.validate()
    assert ok.realization_seed == 7
    assert ok.realization_sigma_scale == 1.0


def test_negative_realization_sigma_scale_is_refused() -> None:
    arguments = Arguments.create(
        [
            ".",
            "--realize_observed_catalogue",
            "--realization_seed",
            "7",
            "--realization_sigma_scale",
            "-0.5",
        ]
    )
    with pytest.raises(ArgumentsError, match="must be >= 0"):
        arguments.validate()


def test_realization_flags_default_to_the_unchanged_path() -> None:
    arguments = Arguments.create(["."])
    arguments.validate()
    assert arguments.realize_observed_catalogue is False
    assert arguments.observed_catalogue is None
    assert arguments.realization_parent is None
    # Provenance: the flags land in run_metadata via the full namespace dump.
    as_dict = arguments.to_dict()
    for key in (
        "realize_observed_catalogue",
        "realization_seed",
        "realization_sigma_scale",
        "realization_parent",
        "observed_catalogue",
    ):
        assert key in as_dict
