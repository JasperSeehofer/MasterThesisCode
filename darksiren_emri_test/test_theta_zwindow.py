"""Tests for the theta-consistent candidate z-window instrument flag (row
#255 tree 2 node T1.3-zwin, results/campaign51_20260728/realistic_20260729/
tree2_20260830/PHYSICS_CHANGE_THETA_ZWINDOW_20260830.md section 2).

``GalaxyCatalogueHandler.get_possible_hosts_from_ball_tree`` gains two new
independent parameters: ``theta_zwindow`` ("off"/"on") and ``z_window_k``
(the candidate z-window half-width, decoupled from the implicit +/- 1 sigma_g
literal). This module pins the gate presentation's section 7 regression plan
(R1-R6), restricted to the handler-level mask (the changes to
``bayesian_statistics.py`` -- CLI/evaluate() plumbing defaults, the call-site
theta passthrough -- are pinned separately in
``darksiren_emri_test/bayesian_inference/test_theta_zwindow.py``).

CPU-only; a synthetic 2000-row-scale-representative but small catalogue. No
GPU, no real GLADE catalogue.
"""

import numpy as np
import pandas as pd
import pytest

from darksiren_emri.galaxy_catalogue.handler import (
    GalaxyCatalogueHandler,
    HostGalaxy,
    InternalCatalogColumns,
)

_PHI = 0.0
_THETA = np.pi / 2
_SIGMA_MULTIPLIER = 1.5  # sky_cone_k -- generous, never excludes these co-located rows

# GW-side envelope (theta-free, unchanged by this flag) -- a fixed window
# the various host redshifts below sit inside, at the edge of, or outside.
_Z_MIN = 0.20
_Z_MAX = 0.40


def _row(z_g: float, sigma_g: float, bh_mass: float = 1.0e6, bh_mass_error: float = 1.0e5) -> dict:
    return {
        InternalCatalogColumns.PHI_S: _PHI,
        InternalCatalogColumns.THETA_S: _THETA,
        InternalCatalogColumns.REDSHIFT: z_g,
        InternalCatalogColumns.REDSHIFT_ERROR: sigma_g,
        InternalCatalogColumns.BH_MASS: bh_mass,
        InternalCatalogColumns.BH_MASS_ERROR: bh_mass_error,
    }


def _make_handler(catalog: pd.DataFrame) -> GalaxyCatalogueHandler:
    """Same object.__new__ shim idiom as test_mass_filter_sigma.py /
    test_mass_filter_geometry.py / fixtures/coordinate.py."""
    instance = object.__new__(GalaxyCatalogueHandler)
    instance.reduced_galaxy_catalog = catalog
    instance.setup_galaxy_catalog_balltree()
    return instance


def _query_kwargs(**overrides: object) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "phi": _PHI,
        "theta": _THETA,
        "phi_sigma": 1.0,
        "theta_sigma": 1.0,
        "cov_theta_phi": 0.0,
        "z_min": _Z_MIN,
        "z_max": _Z_MAX,
        # Wide enough that the mass filter never excludes any row here --
        # these tests are about the z-filter (the "without BH mass" list),
        # read off result[0].
        "M_z": 1.0e6,
        "M_z_sigma": 1.0e12,
        "sigma_multiplier": _SIGMA_MULTIPLIER,
    }
    kwargs.update(overrides)
    return kwargs


def _no_bh_indices(hosts: list[HostGalaxy]) -> set[int]:
    return {int(host.catalog_index) for host in hosts}


def _z_filtered_indices(handler: GalaxyCatalogueHandler, **kwargs: object) -> set[int]:
    result = handler.get_possible_hosts_from_ball_tree(**kwargs)  # type: ignore[arg-type]
    assert result is not None
    without_bh, _with_bh = result
    return _no_bh_indices(without_bh)


# A representative host mix: inside, at each edge, and outside the [0.20,
# 0.40] envelope at k=1 (bare +/- 1 sigma_g), spanning a range of sigma_g.
_ROWS = {
    "deep_inside": _row(z_g=0.30, sigma_g=0.02),  # [0.28, 0.32] subset of envelope
    "low_edge_in": _row(z_g=0.205, sigma_g=0.02),  # [0.185, 0.225] overlaps low edge
    "low_edge_out": _row(z_g=0.15, sigma_g=0.02),  # [0.13, 0.17] entirely below z_min
    "high_edge_in": _row(z_g=0.395, sigma_g=0.02),  # [0.375, 0.415] overlaps high edge
    "high_edge_out": _row(z_g=0.50, sigma_g=0.02),  # [0.48, 0.52] entirely above z_max
    "wide_sigma": _row(z_g=0.10, sigma_g=0.30),  # [-0.20, 0.40] wide, touches z_max at k=1
}


def _catalog() -> pd.DataFrame:
    return pd.DataFrame(list(_ROWS.values()), index=list(range(len(_ROWS))))


def _label_to_index() -> dict[str, int]:
    return {label: i for i, label in enumerate(_ROWS)}


# ===========================================================================
# R1 -- byte-identity at "off": reproduces the OLD bare +/- 1 sigma_g mask
# ===========================================================================


class TestThetaZwindowOffByteIdentity:
    def test_off_default_matches_hand_computed_bare_mask(self) -> None:
        handler = _make_handler(_catalog())
        idx = _label_to_index()
        got = _z_filtered_indices(handler, **_query_kwargs())

        expected = set()
        for label, row in _ROWS.items():
            z_g = row[InternalCatalogColumns.REDSHIFT]
            sigma_g = row[InternalCatalogColumns.REDSHIFT_ERROR]
            if _Z_MIN <= z_g + sigma_g and _Z_MAX >= z_g - sigma_g:
                expected.add(idx[label])
        assert got == expected
        # Sanity: the fixture actually discriminates (not all-in/all-out).
        assert 0 < len(expected) < len(_ROWS)

    def test_off_explicit_matches_implicit_default(self) -> None:
        handler_default = _make_handler(_catalog())
        handler_explicit = _make_handler(_catalog())
        got_default = _z_filtered_indices(handler_default, **_query_kwargs())
        got_explicit = _z_filtered_indices(
            handler_explicit, theta_zwindow="off", z_window_k=1.0, **_query_kwargs()
        )
        assert got_default == got_explicit

    def test_off_ignores_theta_b_theta_s_entirely(self) -> None:
        """ "off" never reads theta_b/theta_s -- passing engaged values must
        not move the mask at all (R1 byte-identity independent of theta
        engagement elsewhere, per the gate doc's call-site design)."""
        handler_identity = _make_handler(_catalog())
        handler_engaged = _make_handler(_catalog())
        got_identity = _z_filtered_indices(handler_identity, **_query_kwargs())
        got_engaged = _z_filtered_indices(
            handler_engaged,
            theta_zwindow="off",
            theta_b=0.5,
            theta_s=3.0,
            **_query_kwargs(),
        )
        assert got_identity == got_engaged


# ===========================================================================
# R2/R3 (GATE T-ID) -- "on" at theta=(0,1) is a LITERAL SKIP: identical to
# "off" at the same z_window_k, for k in {1, 2, 4} (R4's k-consistency)
# ===========================================================================


class TestThetaZwindowIdentityAtThetaZeroOne:
    @pytest.mark.parametrize("k", [1.0, 2.0, 4.0])
    def test_on_at_identity_theta_matches_off_at_same_k(self, k: float) -> None:
        handler_off = _make_handler(_catalog())
        handler_on = _make_handler(_catalog())
        got_off = _z_filtered_indices(
            handler_off, theta_zwindow="off", z_window_k=k, **_query_kwargs()
        )
        got_on = _z_filtered_indices(
            handler_on,
            theta_zwindow="on",
            z_window_k=k,
            theta_b=0.0,
            theta_s=1.0,
            **_query_kwargs(),
        )
        assert got_off == got_on

    def test_k_equal_1_and_k_equal_4_are_nested_at_identity(self) -> None:
        """R4: the k=4 set contains the k=1 set (a wider window admits a
        superset), independent of theta_zwindow state at the identity."""
        handler_k1 = _make_handler(_catalog())
        handler_k4 = _make_handler(_catalog())
        got_k1 = _z_filtered_indices(
            handler_k1, theta_zwindow="on", z_window_k=1.0, **_query_kwargs()
        )
        got_k4 = _z_filtered_indices(
            handler_k4, theta_zwindow="on", z_window_k=4.0, **_query_kwargs()
        )
        assert got_k1 <= got_k4
        assert got_k1 != got_k4  # the fixture must actually discriminate


# ===========================================================================
# z_window_k=2.0 byte-identity: "on"/"off" agree at theta=(0,1), k=2.0, and
# reproduce the hand-computed +/- 2 sigma_g bare mask at "off"
# ===========================================================================


class TestZWindowK2ByteIdentity:
    def test_off_k2_matches_hand_computed_pm2_sigma_mask(self) -> None:
        handler = _make_handler(_catalog())
        idx = _label_to_index()
        got = _z_filtered_indices(handler, theta_zwindow="off", z_window_k=2.0, **_query_kwargs())
        expected = set()
        for label, row in _ROWS.items():
            z_g = row[InternalCatalogColumns.REDSHIFT]
            sigma_g = row[InternalCatalogColumns.REDSHIFT_ERROR]
            if _Z_MIN <= z_g + 2.0 * sigma_g and _Z_MAX >= z_g - 2.0 * sigma_g:
                expected.add(idx[label])
        assert got == expected


# ===========================================================================
# Engagement -- the window changes under s != 1 (and b != 0) with the flag on
# ===========================================================================


class TestThetaZwindowEngagement:
    def test_wider_s_recovers_more_candidates(self) -> None:
        """s > 1 widens sigma_g^theta = s*sigma_g -> superset of the s=1 set.
        s=3.0 is chosen so the *_out rows' gap (0.05 / 0.10) is bridged by
        the widened +/- 1 sigma_g^theta window (0.06), newly admitting
        low_edge_out (unlike a smaller s, see the k-consistency test above)."""
        handler_s1 = _make_handler(_catalog())
        handler_s3 = _make_handler(_catalog())
        got_s1 = _z_filtered_indices(
            handler_s1, theta_zwindow="on", theta_b=0.0, theta_s=1.0, **_query_kwargs()
        )
        got_s3 = _z_filtered_indices(
            handler_s3, theta_zwindow="on", theta_b=0.0, theta_s=3.0, **_query_kwargs()
        )
        assert got_s1 <= got_s3
        assert got_s1 != got_s3

    def test_b_shift_moves_the_candidate_set(self) -> None:
        """A centre shift b(1+z_g) moves which edge rows are admitted."""
        handler_b0 = _make_handler(_catalog())
        handler_bpos = _make_handler(_catalog())
        got_b0 = _z_filtered_indices(
            handler_b0, theta_zwindow="on", theta_b=0.0, theta_s=1.0, **_query_kwargs()
        )
        got_bpos = _z_filtered_indices(
            handler_bpos, theta_zwindow="on", theta_b=0.15, theta_s=1.0, **_query_kwargs()
        )
        assert got_b0 != got_bpos

    def test_on_engaged_differs_from_off_at_same_k(self) -> None:
        handler_off = _make_handler(_catalog())
        handler_on = _make_handler(_catalog())
        got_off = _z_filtered_indices(handler_off, theta_zwindow="off", **_query_kwargs())
        got_on = _z_filtered_indices(
            handler_on, theta_zwindow="on", theta_b=0.1, theta_s=1.8, **_query_kwargs()
        )
        assert got_off != got_on


# ===========================================================================
# theta-consistency: "on" reproduces the 2.1 hand-computed mask exactly
# (SIGMA_V_PEC_KM_S = 0.0 today, so sigma_pv,g = 0 -- sigma_g^theta = s*sigma_g)
# ===========================================================================


class TestThetaZwindowHandComputedMask:
    @pytest.mark.parametrize(
        "theta_b,theta_s,k", [(0.1, 1.8, 1.0), (-0.05, 0.6, 2.0), (0.2, 3.0, 4.0)]
    )
    def test_on_matches_2_1_closed_form(self, theta_b: float, theta_s: float, k: float) -> None:
        handler = _make_handler(_catalog())
        got = _z_filtered_indices(
            handler,
            theta_zwindow="on",
            theta_b=theta_b,
            theta_s=theta_s,
            z_window_k=k,
            **_query_kwargs(),
        )
        idx = _label_to_index()
        expected = set()
        for label, row in _ROWS.items():
            z_g = row[InternalCatalogColumns.REDSHIFT]
            sigma_g = row[InternalCatalogColumns.REDSHIFT_ERROR]
            z_g_theta = z_g + theta_b * (1.0 + z_g)
            sigma_g_theta = theta_s * sigma_g  # sigma_pv,g == 0.0 (SIGMA_V_PEC_KM_S == 0.0)
            if _Z_MIN <= z_g_theta + k * sigma_g_theta and _Z_MAX >= z_g_theta - k * sigma_g_theta:
                expected.add(idx[label])
        assert got == expected


# ===========================================================================
# Guards
# ===========================================================================


class TestThetaZwindowGuards:
    def test_invalid_theta_zwindow_token_raises(self) -> None:
        handler = _make_handler(_catalog())
        with pytest.raises(ValueError, match="theta_zwindow"):
            handler.get_possible_hosts_from_ball_tree(
                theta_zwindow="bogus",
                **_query_kwargs(),  # type: ignore[arg-type]
            )

    @pytest.mark.parametrize("bad_k", [0.0, -1.0, float("nan"), float("inf"), float("-inf")])
    def test_non_positive_or_non_finite_z_window_k_raises(self, bad_k: float) -> None:
        handler = _make_handler(_catalog())
        with pytest.raises(ValueError, match="z_window_k"):
            handler.get_possible_hosts_from_ball_tree(
                z_window_k=bad_k,
                **_query_kwargs(),  # type: ignore[arg-type]
            )

    def test_z_window_k_guard_applies_regardless_of_theta_zwindow_state(self) -> None:
        """The guard fires at "off" too (section 2.2: k applies under BOTH
        states)."""
        handler = _make_handler(_catalog())
        with pytest.raises(ValueError, match="z_window_k"):
            handler.get_possible_hosts_from_ball_tree(
                theta_zwindow="off",
                z_window_k=-2.0,
                **_query_kwargs(),  # type: ignore[arg-type]
            )


# ===========================================================================
# Mass-filter inheritance (R6 of the gate doc's regression plan): the
# with-BH set (result[1]) is the mass filter applied to the (possibly
# z-window-widened) no-BH set (result[0]) -- membership only ever shrinks.
# ===========================================================================


class TestThetaZwindowMassFilterInheritance:
    def test_with_bh_set_is_subset_of_no_bh_set_at_k4(self) -> None:
        handler = _make_handler(_catalog())
        result = handler.get_possible_hosts_from_ball_tree(
            theta_zwindow="on",
            theta_b=0.0,
            theta_s=1.0,
            z_window_k=4.0,
            **_query_kwargs(),  # type: ignore[arg-type]
        )
        assert result is not None
        without_bh, with_bh = result
        assert _no_bh_indices(with_bh) <= _no_bh_indices(without_bh)
        # This fixture's mass window is deliberately permissive (M_z_sigma
        # huge): the mass filter should not narrow the widened z-set at all.
        assert _no_bh_indices(with_bh) == _no_bh_indices(without_bh)
