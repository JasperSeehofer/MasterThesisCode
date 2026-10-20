"""Tests for the mass-window GEOMETRY instrument flag (charter node B5.1,
results/campaign51_20260728/realistic_20260729/fanout1_20260829/
PHYSICS_CHANGE_MASS_WINDOW_GEOMETRY_20260829.md §2/§6; ledger rows
#220-#223).

``GalaxyCatalogueHandler.get_possible_hosts_from_ball_tree`` gains two new
independent flags: ``mass_filter_geometry`` ("linear"/"log") and
``mass_filter_k`` (the mass-window half-width, decoupled from
``sigma_multiplier`` -- which after this change feeds ONLY the sky-cone
search radius). This module pins the gate presentation's §6 regression plan:

1. Default byte-identity: ``mass_filter_geometry="linear"``,
   ``mass_filter_k=1.5`` reproduces the exact pre-flag mask (the
   ``mass_filter_sigma`` behaviour, unchanged), including when
   ``sigma_multiplier`` differs from the default sky-cone value (proving the
   mass window no longer reads ``sigma_multiplier`` at all).
2. Log-window unit test with hand-computed edges at k in {1.5, 2.5, 3}.
3. Epsilon test: for a synthetic log-normal candidate population at a FIXED
   sigma_lnM = s, the fraction excluded by the log window at multiplier k
   equals 2*Phi(-k) to Monte-Carlo precision, CV/sigma-independent by
   construction (deterministic seed, no flakiness).

Plus: invalid-value ValueErrors at the single read/validate site, and two of
R7's disclosed residual-gap invariants (k -> infinity agreement; sigma -> 0
point-overlap agreement) closed here as a bonus (not required by §6, but
cheap and directly closes a gap the presentation flags as open).
"""

import math

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from darksiren_emri.galaxy_catalogue.handler import (
    GalaxyCatalogueHandler,
    HostGalaxy,
    InternalCatalogColumns,
)

_PHI = 0.0
_THETA = np.pi / 2
_Z = 0.0
_Z_ERROR = 0.01

_M_Z = 1_000_000.0
_M_Z_SIGMA = 10_000.0
_SIGMA_MULTIPLIER = 1.5


def _row(bh_mass: float, bh_mass_error: float) -> dict[str, float]:
    return {
        InternalCatalogColumns.PHI_S: _PHI,
        InternalCatalogColumns.THETA_S: _THETA,
        InternalCatalogColumns.REDSHIFT: _Z,
        InternalCatalogColumns.REDSHIFT_ERROR: _Z_ERROR,
        InternalCatalogColumns.BH_MASS: bh_mass,
        InternalCatalogColumns.BH_MASS_ERROR: bh_mass_error,
    }


def _make_handler(catalog: pd.DataFrame) -> GalaxyCatalogueHandler:
    """Same object.__new__ shim idiom as test_mass_filter_sigma.py /
    test_coordinate_roundtrip.py / fixtures/coordinate.py."""
    instance = object.__new__(GalaxyCatalogueHandler)
    instance.reduced_galaxy_catalog = catalog
    instance.setup_galaxy_catalog_balltree()
    return instance


def _query_kwargs(sigma_multiplier: float = _SIGMA_MULTIPLIER) -> dict[str, float]:
    return {
        "phi": _PHI,
        "theta": _THETA,
        "phi_sigma": 1.0,
        "theta_sigma": 1.0,
        "cov_theta_phi": 0.0,
        "z_min": _Z,
        "z_max": _Z,
        "M_z": _M_Z,
        "M_z_sigma": _M_Z_SIGMA,
        "sigma_multiplier": sigma_multiplier,
    }


def _with_bh_indices(hosts: list[HostGalaxy]) -> set[int]:
    return {int(host.catalog_index) for host in hosts}


# ── §6 item 1: default byte-identity ─────────────────────────────────────


class TestMassFilterGeometryDefaultByteIdentity:
    """mass_filter_geometry="linear", mass_filter_k=1.5 (both defaults)
    reproduces the exact pre-flag (mass_filter_sigma-only) mask -- for BOTH
    mass_filter_sigma cells, and independently of sigma_multiplier (proving
    the mass window no longer reads it)."""

    _INSIDE = _row(1_000_000.0, 5_000.0)
    _BOUNDARY = _row(974_001.0, 8_000.0)
    _OUTSIDE = _row(500_000.0, 5_000.0)

    def _catalog(self) -> pd.DataFrame:
        return pd.DataFrame([self._INSIDE, self._BOUNDARY, self._OUTSIDE])

    @pytest.mark.parametrize("mass_filter_sigma", ["symmetric", "asymmetric"])
    def test_new_defaults_match_pre_flag_call(self, mass_filter_sigma: str) -> None:
        handler_old_call = _make_handler(self._catalog())
        handler_new_call = _make_handler(self._catalog())

        result_old = handler_old_call.get_possible_hosts_from_ball_tree(
            mass_filter_sigma=mass_filter_sigma,
            **_query_kwargs(),  # type: ignore[arg-type]
        )
        result_new = handler_new_call.get_possible_hosts_from_ball_tree(
            mass_filter_sigma=mass_filter_sigma,
            mass_filter_geometry="linear",
            mass_filter_k=1.5,
            **_query_kwargs(),  # type: ignore[arg-type]
        )
        assert result_old is not None
        assert result_new is not None
        _, with_bh_old = result_old
        _, with_bh_new = result_new
        assert _with_bh_indices(with_bh_old) == _with_bh_indices(with_bh_new)

    def test_mass_window_no_longer_reads_sigma_multiplier(self) -> None:
        """Changing sigma_multiplier (now sky-cone-only) must not move the
        mass-window membership at all once mass_filter_k is held fixed."""
        handler_small_radius = _make_handler(self._catalog())
        handler_large_radius = _make_handler(self._catalog())

        result_small = handler_small_radius.get_possible_hosts_from_ball_tree(
            mass_filter_k=1.5,
            **_query_kwargs(sigma_multiplier=1.5),  # type: ignore[arg-type]
        )
        result_large = handler_large_radius.get_possible_hosts_from_ball_tree(
            mass_filter_k=1.5,
            **_query_kwargs(sigma_multiplier=5.0),  # type: ignore[arg-type]
        )
        assert result_small is not None
        assert result_large is not None
        _, with_bh_small = result_small
        _, with_bh_large = result_large
        assert _with_bh_indices(with_bh_small) == _with_bh_indices(with_bh_large) == {0, 1}


# ── §6 item 2: log-window unit test, hand-computed edges ────────────────


class TestMassFilterGeometryLogHandComputedEdges:
    """Independently (re-)computed edges via the closed form in
    PHYSICS_CHANGE_MASS_WINDOW_GEOMETRY_20260829.md §2/§4:
        sigma_lnM,z = M_z_sigma / M_z
        gw_lo, gw_hi = M_z*exp(-/+ k*sigma_lnM,z) / (1+z)
        sigma_lnM = BH_MASS_ERROR / BH_MASS
        cand_lo, cand_hi = BH_MASS*exp(-/+ mult*sigma_lnM)
    mult = k under "symmetric" (default), 1.0 under "asymmetric".
    Overlap test: gw_lo <= cand_hi AND cand_lo <= gw_hi.
    """

    @pytest.mark.parametrize("k", [1.5, 2.5, 3.0])
    def test_included_candidate_matches_hand_overlap(self, k: float) -> None:
        m_z, m_z_sigma = 1_000_000.0, 50_000.0
        bh_mass, bh_mass_error = 1_020_000.0, 40_800.0  # sigma_lnM = 0.04

        sigma_lnM_z = m_z_sigma / m_z
        gw_lo = m_z * math.exp(-k * sigma_lnM_z)
        gw_hi = m_z * math.exp(+k * sigma_lnM_z)
        sigma_lnM = bh_mass_error / bh_mass
        cand_lo = bh_mass * math.exp(-k * sigma_lnM)
        cand_hi = bh_mass * math.exp(+k * sigma_lnM)
        expect_included = (gw_lo <= cand_hi) and (cand_lo <= gw_hi)
        assert expect_included, "test fixture must exercise the included branch"

        catalog = pd.DataFrame([_row(bh_mass, bh_mass_error)])
        handler = _make_handler(catalog)
        result = handler.get_possible_hosts_from_ball_tree(
            phi=_PHI,
            theta=_THETA,
            phi_sigma=1.0,
            theta_sigma=1.0,
            cov_theta_phi=0.0,
            z_min=_Z,
            z_max=_Z,
            M_z=m_z,
            M_z_sigma=m_z_sigma,
            sigma_multiplier=1.5,  # type: ignore[arg-type]
            mass_filter_geometry="log",
            mass_filter_k=k,
        )
        assert result is not None
        _, with_bh = result
        assert _with_bh_indices(with_bh) == {0}

    @pytest.mark.parametrize("k", [1.5, 2.5, 3.0])
    def test_excluded_candidate_matches_hand_overlap(self, k: float) -> None:
        m_z, m_z_sigma = 1_000_000.0, 50_000.0
        # Far heavier candidate, tight own error -- log-bounded window
        # cannot reach it at any of these k (unlike linear's negative-
        # lower-edge loophole, cf. gate doc §7).
        bh_mass, bh_mass_error = 5_000_000.0, 50_000.0  # sigma_lnM = 0.01

        sigma_lnM_z = m_z_sigma / m_z
        gw_lo = m_z * math.exp(-k * sigma_lnM_z)
        gw_hi = m_z * math.exp(+k * sigma_lnM_z)
        sigma_lnM = bh_mass_error / bh_mass
        cand_lo = bh_mass * math.exp(-k * sigma_lnM)
        cand_hi = bh_mass * math.exp(+k * sigma_lnM)
        expect_included = (gw_lo <= cand_hi) and (cand_lo <= gw_hi)
        assert not expect_included, "test fixture must exercise the excluded branch"

        catalog = pd.DataFrame([_row(bh_mass, bh_mass_error)])
        handler = _make_handler(catalog)
        result = handler.get_possible_hosts_from_ball_tree(
            phi=_PHI,
            theta=_THETA,
            phi_sigma=1.0,
            theta_sigma=1.0,
            cov_theta_phi=0.0,
            z_min=_Z,
            z_max=_Z,
            M_z=m_z,
            M_z_sigma=m_z_sigma,
            sigma_multiplier=1.5,  # type: ignore[arg-type]
            mass_filter_geometry="log",
            mass_filter_k=k,
        )
        assert result is not None
        _, with_bh = result
        assert _with_bh_indices(with_bh) == set()

    def test_asymmetric_uses_bare_candidate_multiplier_in_log_geometry(self) -> None:
        """mass_filter_sigma="asymmetric" must use mult=1.0 (not k) on the
        candidate side even under geometry="log" -- the two flags are read
        independently, each at its own single site."""
        k = 3.0
        m_z, m_z_sigma = 1_000_000.0, 50_000.0
        # sigma_lnM chosen so the candidate is EXCLUDED at mult=1.0 (bare)
        # but INCLUDED at mult=k=3.0 (symmetric) -- widening the candidate
        # window can only ADD overlap, never remove it, so this is the only
        # achievable direction; it still discriminates the two multiplier
        # conventions.
        bh_mass, bh_mass_error = 1_600_000.0, 400_000.0  # sigma_lnM = 0.25

        sigma_lnM_z = m_z_sigma / m_z
        gw_lo = m_z * math.exp(-k * sigma_lnM_z)
        gw_hi = m_z * math.exp(+k * sigma_lnM_z)
        sigma_lnM = bh_mass_error / bh_mass
        cand_lo_bare = bh_mass * math.exp(-1.0 * sigma_lnM)
        cand_hi_bare = bh_mass * math.exp(+1.0 * sigma_lnM)
        cand_lo_sym = bh_mass * math.exp(-k * sigma_lnM)
        cand_hi_sym = bh_mass * math.exp(+k * sigma_lnM)
        included_bare = (gw_lo <= cand_hi_bare) and (cand_lo_bare <= gw_hi)
        included_sym = (gw_lo <= cand_hi_sym) and (cand_lo_sym <= gw_hi)
        assert not included_bare and included_sym, (
            "test fixture must discriminate asymmetric (bare) vs symmetric (k) "
            "candidate-side multipliers"
        )

        catalog = pd.DataFrame([_row(bh_mass, bh_mass_error)])

        handler_asym = _make_handler(catalog)
        result_asym = handler_asym.get_possible_hosts_from_ball_tree(
            phi=_PHI,
            theta=_THETA,
            phi_sigma=1.0,
            theta_sigma=1.0,
            cov_theta_phi=0.0,
            z_min=_Z,
            z_max=_Z,
            M_z=m_z,
            M_z_sigma=m_z_sigma,
            sigma_multiplier=1.5,  # type: ignore[arg-type]
            mass_filter_sigma="asymmetric",
            mass_filter_geometry="log",
            mass_filter_k=k,
        )
        handler_sym = _make_handler(catalog)
        result_sym = handler_sym.get_possible_hosts_from_ball_tree(
            phi=_PHI,
            theta=_THETA,
            phi_sigma=1.0,
            theta_sigma=1.0,
            cov_theta_phi=0.0,
            z_min=_Z,
            z_max=_Z,
            M_z=m_z,
            M_z_sigma=m_z_sigma,
            sigma_multiplier=1.5,  # type: ignore[arg-type]
            mass_filter_sigma="symmetric",
            mass_filter_geometry="log",
            mass_filter_k=k,
        )
        assert result_asym is not None
        assert result_sym is not None
        _, with_bh_asym = result_asym
        _, with_bh_sym = result_sym
        assert _with_bh_indices(with_bh_asym) == set()
        assert _with_bh_indices(with_bh_sym) == {0}


# ── §6 item 3: epsilon test, 2*Phi(-k) numerically ───────────────────────


class TestMassFilterGeometryLogEpsilonTest:
    """For a FIXED sigma_lnM = s, drawing candidate masses
    M_i = M_z * exp(s*Z_i), Z_i ~ N(0,1), with a point-like GW window
    (M_z_sigma/M_z -> 0), the excluded fraction at multiplier k is
    P(|Z| > k) = 2*Phi(-k) -- independent of s by construction
    (gate doc §6 item 3 / §2's eps_log(k) = 2*norm.cdf(-k)).

    Deterministic seed (no flakiness); N and the tolerance are sized so the
    analytic 5-sigma Monte-Carlo band comfortably contains the observed
    fraction for every k tested.
    """

    @pytest.mark.parametrize("k", [1.5, 2.5, 3.0])
    def test_excluded_fraction_matches_two_phi_minus_k(self, k: float) -> None:
        rng = np.random.default_rng(20260829)
        n = 20_000
        s = 0.3  # fixed sigma_lnM; the predicted fraction must not depend on it
        m_z = 1_000_000.0
        m_z_sigma = m_z * 1e-9  # point-like GW window (sigma_lnM,z ~ 1e-9)

        z_draws = rng.standard_normal(n)
        masses = m_z * np.exp(s * z_draws)
        errors = s * masses  # BH_MASS_ERROR / BH_MASS == s exactly, per candidate

        rows = [_row(float(mass), float(error)) for mass, error in zip(masses, errors, strict=True)]
        catalog = pd.DataFrame(rows)
        handler = _make_handler(catalog)
        result = handler.get_possible_hosts_from_ball_tree(
            phi=_PHI,
            theta=_THETA,
            phi_sigma=1.0,
            theta_sigma=1.0,
            cov_theta_phi=0.0,
            z_min=_Z,
            z_max=_Z,
            M_z=m_z,
            M_z_sigma=m_z_sigma,
            sigma_multiplier=1.5,  # type: ignore[arg-type]
            mass_filter_geometry="log",
            mass_filter_k=k,
        )
        assert result is not None
        _, with_bh = result
        n_included = len(with_bh)
        observed_excluded_fraction = 1.0 - n_included / n

        predicted = 2.0 * norm.cdf(-k)
        se = math.sqrt(predicted * (1.0 - predicted) / n)
        assert observed_excluded_fraction == pytest.approx(predicted, abs=6 * se)


# ── Invalid values ────────────────────────────────────────────────────────


class TestMassFilterGeometryInvalidValues:
    def test_invalid_geometry_raises_value_error(self) -> None:
        catalog = pd.DataFrame([_row(1_000_000.0, 50_000.0)])
        handler = _make_handler(catalog)
        with pytest.raises(ValueError, match="mass_filter_geometry"):
            handler.get_possible_hosts_from_ball_tree(
                mass_filter_geometry="bogus",
                **_query_kwargs(),  # type: ignore[arg-type]
            )

    @pytest.mark.parametrize("bad_k", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_k_raises_value_error(self, bad_k: float) -> None:
        catalog = pd.DataFrame([_row(1_000_000.0, 50_000.0)])
        handler = _make_handler(catalog)
        with pytest.raises(ValueError, match="mass_filter_k"):
            handler.get_possible_hosts_from_ball_tree(
                mass_filter_k=bad_k,
                **_query_kwargs(),  # type: ignore[arg-type]
            )


# ── Bonus: R7 disclosed residual-gap invariants (not required by §6) ─────


class TestMassFilterGeometryLimitingCases:
    """Closes two of R7's disclosed gaps ("do not yet have a dedicated
    automated regression test") for the two geometries actually implemented
    here: k -> infinity agreement (invariant 2) and sigma -> 0 point-overlap
    agreement (invariant 3). Invariant 4 (first-order kσ<<1 agreement) is a
    direct algebraic consequence of exp(x)=1+x+O(x^2) and is not
    additionally pinned here."""

    def test_k_to_infinity_admits_everything_both_geometries(self) -> None:
        catalog = pd.DataFrame([_row(1_000_000.0, 50_000.0), _row(1.0, 0.5), _row(1.0e9, 1.0e8)])
        for geometry in ("linear", "log"):
            handler = _make_handler(catalog)
            result = handler.get_possible_hosts_from_ball_tree(
                mass_filter_geometry=geometry,
                mass_filter_k=500.0,  # large enough to be "infinity" without exp() overflow
                **_query_kwargs(),  # type: ignore[arg-type]
            )
            assert result is not None
            _, with_bh = result
            assert _with_bh_indices(with_bh) == {0, 1, 2}, geometry

    def test_sigma_to_zero_is_point_overlap_both_geometries(self) -> None:
        """As sigma->0 (both GW and candidate), the window collapses to
        M == M_z (up to float precision) for both geometries."""
        m_z = 1_000_000.0
        tiny = 1e-6
        catalog_match = pd.DataFrame([_row(m_z, m_z * 1e-9)])
        catalog_miss = pd.DataFrame([_row(m_z * 1.01, m_z * 1e-9)])
        for geometry in ("linear", "log"):
            handler_match = _make_handler(catalog_match)
            result_match = handler_match.get_possible_hosts_from_ball_tree(
                phi=_PHI,
                theta=_THETA,
                phi_sigma=1.0,
                theta_sigma=1.0,
                cov_theta_phi=0.0,
                z_min=_Z,
                z_max=_Z,
                M_z=m_z,
                M_z_sigma=m_z * tiny,
                sigma_multiplier=1.5,  # type: ignore[arg-type]
                mass_filter_geometry=geometry,
                mass_filter_k=3.0,
            )
            assert result_match is not None
            _, with_bh_match = result_match
            assert _with_bh_indices(with_bh_match) == {0}, geometry

            handler_miss = _make_handler(catalog_miss)
            result_miss = handler_miss.get_possible_hosts_from_ball_tree(
                phi=_PHI,
                theta=_THETA,
                phi_sigma=1.0,
                theta_sigma=1.0,
                cov_theta_phi=0.0,
                z_min=_Z,
                z_max=_Z,
                M_z=m_z,
                M_z_sigma=m_z * tiny,
                sigma_multiplier=1.5,  # type: ignore[arg-type]
                mass_filter_geometry=geometry,
                mass_filter_k=3.0,
            )
            assert result_miss is not None
            _, with_bh_miss = result_miss
            assert _with_bh_indices(with_bh_miss) == set(), geometry
