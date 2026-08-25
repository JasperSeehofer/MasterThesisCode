"""Tests for the ``mass_filter_sigma`` instrumentation flag (ledger row #198).

``GalaxyCatalogueHandler.get_possible_hosts_from_ball_tree`` builds a
candidate-host mass window by widening the GW mass estimate by
``M_z_sigma * sigma_multiplier`` on both sides, but historically compared it
against the galaxy's own ``BH_MASS_ERROR`` at its bare (×1) value -- an
asymmetric ±1.5σ-vs-±1σ window (the verified defect candidate). This module
pins:

1. The default ``"asymmetric"`` cell reproduces the exact pre-flag mask
   (regression -- byte-identical candidate set).
2. The ``"symmetric"`` counterfactual retains a boundary candidate (the
   event-113 class: between 1σ and 1.5σ of its own ``BH_MASS_ERROR`` outside
   the GW window edge) that ``"asymmetric"`` rejects.
3. An invalid flag value raises ``ValueError`` at the single read/validate
   site.

Scope: the MASS filter only -- the redshift filter shares the asymmetric
convention but is outside this instrumentation grant.
"""

import numpy as np
import pandas as pd
import pytest

from darksiren_emri.galaxy_catalogue.handler import (
    GalaxyCatalogueHandler,
    HostGalaxy,
    InternalCatalogColumns,
)

# All galaxies sit at the same sky position as the query, on the ecliptic
# equator (theta=pi/2) and at redshift 0 with a small error -- isolates the
# mass-filter branch from the sky-radius and redshift-filter logic.
_PHI = 0.0
_THETA = np.pi / 2
_Z = 0.0
_Z_ERROR = 0.01

# GW mass estimate: window = [M_z - M_z_sigma*sigma_multiplier,
# M_z + M_z_sigma*sigma_multiplier] = [985_000, 1_015_000] (z_min=z_max=0
# collapses the (1+z) divisors to 1).
_M_Z = 1_000_000.0
_M_Z_SIGMA = 10_000.0
_SIGMA_MULTIPLIER = 1.5
_WINDOW_LOW = _M_Z - _M_Z_SIGMA * _SIGMA_MULTIPLIER
_WINDOW_HIGH = _M_Z + _M_Z_SIGMA * _SIGMA_MULTIPLIER

# "inside": comfortably inside the GW window under either convention.
_INSIDE_BH_MASS = 1_000_000.0
_INSIDE_BH_MASS_ERROR = 5_000.0

# "outside": far enough outside that even the widened (symmetric) window
# misses it -- a control showing "symmetric" does not accept everything.
_OUTSIDE_BH_MASS = 500_000.0
_OUTSIDE_BH_MASS_ERROR = 5_000.0

# "boundary" (the event-113 class): window_low - BH_MASS = 10_999, which is
# between 1x and 1.5x its own BH_MASS_ERROR (8_000) -- so the ×1 asymmetric
# window rejects it (974_001 + 8_000 = 982_001 < 985_000) while the ×1.5
# symmetric window accepts it (974_001 + 12_000 = 986_001 >= 985_000).
_BOUNDARY_BH_MASS = 974_001.0
_BOUNDARY_BH_MASS_ERROR = 8_000.0


def _build_catalog() -> pd.DataFrame:
    """Three-galaxy synthetic catalog: inside (0), boundary (1), outside (2)."""
    rows = [
        {
            InternalCatalogColumns.PHI_S: _PHI,
            InternalCatalogColumns.THETA_S: _THETA,
            InternalCatalogColumns.REDSHIFT: _Z,
            InternalCatalogColumns.REDSHIFT_ERROR: _Z_ERROR,
            InternalCatalogColumns.BH_MASS: _INSIDE_BH_MASS,
            InternalCatalogColumns.BH_MASS_ERROR: _INSIDE_BH_MASS_ERROR,
        },
        {
            InternalCatalogColumns.PHI_S: _PHI,
            InternalCatalogColumns.THETA_S: _THETA,
            InternalCatalogColumns.REDSHIFT: _Z,
            InternalCatalogColumns.REDSHIFT_ERROR: _Z_ERROR,
            InternalCatalogColumns.BH_MASS: _BOUNDARY_BH_MASS,
            InternalCatalogColumns.BH_MASS_ERROR: _BOUNDARY_BH_MASS_ERROR,
        },
        {
            InternalCatalogColumns.PHI_S: _PHI,
            InternalCatalogColumns.THETA_S: _THETA,
            InternalCatalogColumns.REDSHIFT: _Z,
            InternalCatalogColumns.REDSHIFT_ERROR: _Z_ERROR,
            InternalCatalogColumns.BH_MASS: _OUTSIDE_BH_MASS,
            InternalCatalogColumns.BH_MASS_ERROR: _OUTSIDE_BH_MASS_ERROR,
        },
    ]
    return pd.DataFrame(rows)


def _make_handler() -> GalaxyCatalogueHandler:
    """Build a GalaxyCatalogueHandler shim with an in-memory BallTree.

    Follows the repo idiom (test_coordinate_roundtrip.py /
    darksiren_emri_test/fixtures/coordinate.py): bypass ``__init__`` via
    ``object.__new__`` and wire up only what
    ``get_possible_hosts_from_ball_tree`` needs.
    """
    instance = object.__new__(GalaxyCatalogueHandler)
    instance.reduced_galaxy_catalog = _build_catalog()
    instance.setup_galaxy_catalog_balltree()
    return instance


def _query_kwargs() -> dict[str, float]:
    return {
        "phi": _PHI,
        "theta": _THETA,
        # Sky sigma large enough that the BallTree radius comfortably
        # captures all three co-located synthetic galaxies.
        "phi_sigma": 1.0,
        "theta_sigma": 1.0,
        "cov_theta_phi": 0.0,
        "z_min": _Z,
        "z_max": _Z,
        "M_z": _M_Z,
        "M_z_sigma": _M_Z_SIGMA,
        "sigma_multiplier": _SIGMA_MULTIPLIER,
    }


def _with_bh_indices(hosts: list[HostGalaxy]) -> set[int]:
    return {int(host.catalog_index) for host in hosts}


class TestMassFilterSigmaRegression:
    """(a) The DEFAULT is "symmetric" (production adoption, PROPOSAL_MASS_FILTER_
    SYMMETRIC_20260825.md §7(a) ruling); explicit "asymmetric" pins the exact
    pre-flag mask as the counterfactual."""

    def test_default_is_symmetric_and_retains_boundary(self) -> None:
        handler = _make_handler()
        result = handler.get_possible_hosts_from_ball_tree(**_query_kwargs())  # type: ignore[arg-type]
        assert result is not None
        _, with_bh_mass = result
        assert _with_bh_indices(with_bh_mass) == {0, 1}

    def test_explicit_asymmetric_pins_pre_flag_mask(self) -> None:
        handler_explicit = _make_handler()
        result_explicit = handler_explicit.get_possible_hosts_from_ball_tree(
            mass_filter_sigma="asymmetric",
            **_query_kwargs(),  # type: ignore[arg-type]
        )
        assert result_explicit is not None
        _, with_bh_explicit = result_explicit
        assert _with_bh_indices(with_bh_explicit) == {0}

    def test_explicit_symmetric_is_bit_identical_to_default(self) -> None:
        handler_default = _make_handler()
        handler_explicit = _make_handler()
        result_default = handler_default.get_possible_hosts_from_ball_tree(
            **_query_kwargs()  # type: ignore[arg-type]
        )
        result_explicit = handler_explicit.get_possible_hosts_from_ball_tree(
            mass_filter_sigma="symmetric",
            **_query_kwargs(),  # type: ignore[arg-type]
        )
        assert result_default is not None
        assert result_explicit is not None
        without_bh_default, with_bh_default = result_default
        without_bh_explicit, with_bh_explicit = result_explicit

        assert _with_bh_indices(with_bh_default) == _with_bh_indices(with_bh_explicit)
        assert {int(h.catalog_index) for h in without_bh_default} == {
            int(h.catalog_index) for h in without_bh_explicit
        }
        # Every candidate's own (M, M_error) is untouched by the flag.
        for host_default, host_explicit in zip(
            sorted(with_bh_default, key=lambda h: h.catalog_index),
            sorted(with_bh_explicit, key=lambda h: h.catalog_index),
            strict=True,
        ):
            assert host_default.M == host_explicit.M
            assert host_default.M_error == host_explicit.M_error


class TestMassFilterSigmaSymmetricCounterfactual:
    """(b) "symmetric" retains the event-113-class boundary candidate."""

    def test_symmetric_retains_boundary_candidate_asymmetric_rejects(self) -> None:
        handler_asymmetric = _make_handler()
        handler_symmetric = _make_handler()

        result_asymmetric = handler_asymmetric.get_possible_hosts_from_ball_tree(
            mass_filter_sigma="asymmetric",
            **_query_kwargs(),  # type: ignore[arg-type]
        )
        result_symmetric = handler_symmetric.get_possible_hosts_from_ball_tree(
            mass_filter_sigma="symmetric",
            **_query_kwargs(),  # type: ignore[arg-type]
        )
        assert result_asymmetric is not None
        assert result_symmetric is not None
        _, with_bh_asymmetric = result_asymmetric
        _, with_bh_symmetric = result_symmetric

        # Asymmetric: only the comfortably-inside candidate (index 0).
        assert _with_bh_indices(with_bh_asymmetric) == {0}
        # Symmetric: the boundary candidate (index 1) is additionally
        # retained; the far-outside candidate (index 2) is still rejected.
        assert _with_bh_indices(with_bh_symmetric) == {0, 1}


class TestMassFilterSigmaInvalidValue:
    """(c) An invalid flag value raises at the single read/validate site."""

    def test_invalid_value_raises_value_error(self) -> None:
        handler = _make_handler()
        with pytest.raises(ValueError, match="mass_filter_sigma"):
            handler.get_possible_hosts_from_ball_tree(
                mass_filter_sigma="bogus",
                **_query_kwargs(),  # type: ignore[arg-type]
            )
