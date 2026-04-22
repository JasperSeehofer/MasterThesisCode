"""RED tests pinning the two critical coordinate-frame bugs (Phase 35).

These tests describe the CORRECT post-fix behavior. They are decorated
with ``@pytest.mark.xfail(strict=True, reason=...)`` against the
current buggy code. When Phase 36 lands the fix, every test flips
XPASS and CI refuses to stay green until the markers are removed —
forcing a conscious handoff.

Bugs pinned:

1. ``galaxy_catalogue/handler.py:_map_angles_to_spherical_coordinates``
   applies NO equatorial->ecliptic rotation. GLADE J2000 (RA, Dec) is
   stored as if it were already ecliptic (phi, theta_polar), drifting
   up to ~23.4° (obliquity).
2. ``galaxy_catalogue/handler.py:setup_galaxy_catalog_balltree`` and
   ``get_possible_hosts_from_ball_tree`` embed with the
   latitude formula ``(cos theta cos phi, cos theta sin phi,
   sin theta)`` on polar-angle data. All ecliptic-equator galaxies
   (theta=pi/2) collapse to ``(0, 0, 1)`` — a singular query point
   where BallTree cannot disambiguate hosts.

Test scenarios (see .planning/phases/35-coordinate-bug-characterization/35-CONTEXT.md D-04..D-07):

- D-04(a): ecliptic equator galaxy round-trip through BallTree
- D-04(b): NCP (Dec=90°) round-trip through ingestion + BallTree
- D-04(c): astropy equatorial->ecliptic ground truth to <0.001°
- D-05: vernal equinox, summer solstice, ecliptic pole edge cases
- D-06/D-07: N=100 random ecliptic-equator galaxies, >=99% recovery

References
----------
.planning/ROADMAP.md §Phase 35, success criteria 1-2.
.planning/phases/35-coordinate-bug-characterization/35-CONTEXT.md D-01..D-07.
"""

import numpy as np
import pytest

from master_thesis_code.galaxy_catalogue.handler import InternalCatalogColumns
from master_thesis_code_test.fixtures.coordinate import (
    build_balltree,
    equatorial_to_ecliptic_astropy,
    synthetic_catalog_builder,
)

# D-01: Locked xfail reason string. Phase 36 removes these markers.
_XFAIL_REASON = "Phase 36 fixes coordinate frame bug — see .planning/milestones/v2.2-..."


class TestEquatorialToEclipticGroundTruth:
    """Contract: equatorial_to_ecliptic_astropy wraps astropy correctly.

    This test is NOT xfail — Plan 01's astropy wrapper is correct by
    construction. It fixtures the ground-truth helper that the other
    tests rely on.
    """

    def test_vernal_equinox_lies_on_both_equators(self) -> None:
        """RA=0 Dec=0 -> ecliptic beta~0 (both equators meet at the vernal equinox).

        ``BarycentricTrueEcliptic`` includes the epoch-specific true obliquity
        (nutation + precession), so lambda and beta are not precisely 0 at J2000
        RA=0/Dec=0 — they differ from the mean values by up to ~0.004°. The test
        uses a 0.01° tolerance consistent with the summer-solstice check.
        """
        result = equatorial_to_ecliptic_astropy(ra_deg=0.0, dec_deg=0.0)
        # beta should be very close to 0.
        assert abs(result.beta_rad) < 0.001 * np.pi / 180, (
            f"Vernal equinox should have ecliptic beta≈0, got {result.beta_rad} rad "
            f"({np.rad2deg(result.beta_rad)} deg)"
        )
        # lambda wraps in [0, 2*pi); vernal equinox is at lambda≈0 (or ≈2*pi).
        # Use the minimum angular distance from 0 (mod 2*pi).
        # Tolerance 0.01°: allows for nutation offset in BarycentricTrueEcliptic.
        lambda_from_zero = min(result.lambda_rad, 2 * np.pi - result.lambda_rad)
        tol_lambda_rad = 0.01 * np.pi / 180
        assert lambda_from_zero < tol_lambda_rad, (
            f"Vernal equinox should have ecliptic lambda≈0 (mod 2*pi), "
            f"got {result.lambda_rad} rad ({np.rad2deg(result.lambda_rad)} deg), "
            f"angular dist from 0 = {np.rad2deg(lambda_from_zero):.4f}°"
        )

    def test_summer_solstice_lies_on_ecliptic_equator(self) -> None:
        """RA=90° Dec≈+23.44° (obliquity) -> ecliptic beta=0° (summer solstice on ecliptic equator).

        Isolates the obliquity sign + magnitude. Tolerance per D-04c: <0.001°.
        The input Dec is derived from the IAU 2006 mean obliquity (23.4392911°), which
        differs by ~0.004° from the epoch-specific BarycentricTrueEcliptic used by astropy,
        so we only check beta=0 (ecliptic latitude, not lambda). The lambda≈pi/2 assertion
        uses a wider 0.01° tolerance to accommodate the obliquity approximation.
        """
        # Summer solstice: RA=6h=90°, Dec=+obliquity lies on ecliptic equator at lambda=90°.
        result = equatorial_to_ecliptic_astropy(ra_deg=90.0, dec_deg=23.4392911)
        tol_rad = 0.001 * np.pi / 180
        assert abs(result.beta_rad) < tol_rad, (
            f"Summer solstice should have beta=0, got beta={np.rad2deg(result.beta_rad)} deg"
        )
        # Lambda check with wider tolerance because the input Dec uses an approximation
        # of the obliquity (23.4392911° vs the epoch-exact value astropy uses).
        tol_lambda_rad = 0.01 * np.pi / 180
        assert abs(result.lambda_rad - np.pi / 2) < tol_lambda_rad, (
            f"Summer solstice should have lambda≈pi/2, got "
            f"lambda={np.rad2deg(result.lambda_rad)} deg "
            f"(diff={np.rad2deg(abs(result.lambda_rad - np.pi / 2)):.4f}°)"
        )

    def test_ecliptic_pole_has_beta_90(self) -> None:
        """RA=18h=270° Dec=+66.56° -> ecliptic north pole (beta=+90°, theta_polar=0)."""
        result = equatorial_to_ecliptic_astropy(ra_deg=270.0, dec_deg=66.5607083)
        tol_rad = 0.001 * np.pi / 180
        assert abs(result.beta_rad - np.pi / 2) < tol_rad, (
            f"Ecliptic pole should have beta=pi/2, got beta={np.rad2deg(result.beta_rad)} deg"
        )
        assert abs(result.theta_polar_rad) < tol_rad, (
            f"Ecliptic pole should have theta_polar=0, got "
            f"theta_polar={np.rad2deg(result.theta_polar_rad)} deg"
        )


class TestBallTreeRecoversEclipticEquatorGalaxy:
    """Contract (Phase 35 D-04a): BallTree retrieves a synthetic
    ecliptic-equator galaxy for a query at the same position.

    Buggy code collapses all theta=pi/2 points to (0, 0, 1) because the
    embedding uses ``(cos theta, ..., sin theta)`` on polar-angle data
    (handler.py:286-288). This test asserts the CORRECT behavior;
    xfail(strict) flips to XPASS once Phase 36 fixes the embedding.
    """

    @pytest.mark.xfail(strict=True, reason=_XFAIL_REASON)
    def test_ball_tree_recovers_ecliptic_equator_galaxy(self) -> None:
        """Single galaxy at ecliptic theta=pi/2, phi=pi; query at the same point must retrieve it."""
        # Build a catalog already in ecliptic radians (bypass the rotation bug
        # so we isolate the BallTree embedding bug).
        df = synthetic_catalog_builder(
            n=1, sky_band="ecliptic_equator", seed=42, already_rotated=True
        )
        # Overwrite the single row with a fully controlled position.
        df.iloc[0, df.columns.get_loc(InternalCatalogColumns.PHI_S)] = np.pi
        df.iloc[0, df.columns.get_loc(InternalCatalogColumns.THETA_S)] = np.pi / 2

        tree = build_balltree(df)

        # Query at the same ecliptic equator position. Use the polar-correct
        # embedding (sin theta cos phi, sin theta sin phi, cos theta):
        phi, theta = np.pi, np.pi / 2
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        query = np.array([[x, y, z]])
        # Small radius: correct embedding makes distance 0, buggy embedding
        # places ALL equator galaxies at (0, 0, 1) which is far from this query.
        indices = tree.query_radius(query, r=0.01)[0]
        assert len(indices) == 1, (
            f"Expected 1 host at ecliptic equator theta=pi/2, phi=pi; got {len(indices)}. "
            "BallTree embedding is currently buggy (handler.py:286-288)."
        )


class TestBallTreeRecoversNorthCelestialPole:
    """Contract (Phase 35 D-04b): a synthetic galaxy at equatorial
    Dec=+90° (NCP) round-trips through catalog ingestion and BallTree.

    Exercises TWO bugs simultaneously:
    1. ``_map_angles_to_spherical_coordinates`` ought to rotate
       equatorial (RA, Dec=+90°) to ecliptic (lambda, beta=+66.56°);
       currently it does not.
    2. BallTree must retrieve the NCP after ingestion.

    This is the primary pole-handling test. Near-pole Dec=±89°
    numerical stability is subsumed per D-deferred note.
    """

    @pytest.mark.xfail(strict=True, reason=_XFAIL_REASON)
    def test_ncp_round_trip_through_ingestion_and_balltree(self) -> None:
        """Equatorial Dec=+90° NCP should be retrievable after ingestion."""
        # Ground truth: NCP in ecliptic frame
        ncp_truth = equatorial_to_ecliptic_astropy(ra_deg=0.0, dec_deg=90.0)
        # NCP has beta ≈ +66.56° = 1.1614 rad; theta_polar ≈ 0.4094 rad.

        # Build a catalog in DEGREES equatorial (already_rotated=False):
        # Inject a single row with equatorial NCP.
        df = synthetic_catalog_builder(n=1, sky_band="north_pole", seed=42, already_rotated=False)
        df.iloc[0, df.columns.get_loc(InternalCatalogColumns.PHI_S)] = 0.0  # RA=0
        df.iloc[0, df.columns.get_loc(InternalCatalogColumns.THETA_S)] = 90.0  # Dec=+90

        # After Phase 36's fix, ingestion applies equatorial->ecliptic rotation
        # inside _map_angles_to_spherical_coordinates. Until then, this test
        # xfails because either:
        # (a) no rotation is applied (stored as ecliptic theta_polar=0 instead of 0.4094), OR
        # (b) the BallTree can't disambiguate near-pole due to embedding bug.
        from master_thesis_code.galaxy_catalogue.handler import GalaxyCatalogueHandler

        instance = object.__new__(GalaxyCatalogueHandler)
        instance.reduced_galaxy_catalog = df
        # This call is expected to rotate equatorial->ecliptic AND map deg->rad
        # (currently it only maps deg->rad — that is the bug).
        instance._rotate_equatorial_to_ecliptic()  # type: ignore[attr-defined]  # noqa: SLF001  (D-13: rotation is a separate __init__ step; type: ignore removed when [PHYSICS] COORD-03 lands the method)
        instance._map_angles_to_spherical_coordinates()  # noqa: SLF001
        instance.setup_galaxy_catalog_balltree()

        # Query with polar-correct embedding at the ground-truth ecliptic NCP position.
        phi, theta = ncp_truth.lambda_rad, ncp_truth.theta_polar_rad
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        query = np.array([[x, y, z]])
        indices = instance.catalog_ball_tree.query_radius(query, r=0.05)[0]
        assert len(indices) == 1, (
            f"NCP must round-trip; got {len(indices)} matches at ecliptic "
            f"(lambda={np.rad2deg(phi):.3f}°, theta_polar={np.rad2deg(theta):.3f}°). "
            "Either rotation is missing (handler.py:486-492) or BallTree "
            "embedding is buggy (handler.py:286-288)."
        )


class TestSummerSolsticeRotation:
    """Contract (Phase 35 D-05): RA=6h=90° Dec=+23.4393° (summer solstice) is
    OFF the equatorial equator but ON the ecliptic equator. A fix that omits
    the obliquity rotation will place it at the wrong latitude.
    """

    @pytest.mark.xfail(strict=True, reason=_XFAIL_REASON)
    def test_summer_solstice_maps_to_ecliptic_equator(self) -> None:
        """Ingestion of (RA=90°, Dec=+23.4393°) must yield ecliptic theta_polar≈pi/2."""
        from master_thesis_code.galaxy_catalogue.handler import GalaxyCatalogueHandler

        df = synthetic_catalog_builder(n=1, sky_band="uniform", seed=42, already_rotated=False)
        df.iloc[0, df.columns.get_loc(InternalCatalogColumns.PHI_S)] = 90.0  # RA
        df.iloc[0, df.columns.get_loc(InternalCatalogColumns.THETA_S)] = 23.4392911  # Dec

        instance = object.__new__(GalaxyCatalogueHandler)
        instance.reduced_galaxy_catalog = df
        instance._rotate_equatorial_to_ecliptic()  # type: ignore[attr-defined]  # noqa: SLF001  (D-13: rotation is a separate __init__ step; type: ignore removed when [PHYSICS] COORD-03 lands the method)
        instance._map_angles_to_spherical_coordinates()  # noqa: SLF001

        theta_stored = float(
            instance.reduced_galaxy_catalog[InternalCatalogColumns.THETA_S].iloc[0]
        )
        # After Phase 36 rotation: ecliptic beta=0 -> theta_polar=pi/2.
        tol_rad = 0.001 * np.pi / 180
        assert abs(theta_stored - np.pi / 2) < tol_rad, (
            f"Summer solstice should land on the ecliptic equator (theta_polar=pi/2); "
            f"got theta_polar={np.rad2deg(theta_stored):.4f}°. "
            "Missing equatorial->ecliptic rotation in handler.py:486-492."
        )


class TestEclipticPoleIngestion:
    """Contract (Phase 35 D-05): RA=18h=270° Dec=+66.56° should ingest as
    the north ecliptic pole (theta_polar=0).
    """

    @pytest.mark.xfail(strict=True, reason=_XFAIL_REASON)
    def test_ecliptic_pole_maps_to_theta_polar_zero(self) -> None:
        """Ingestion of (RA=270°, Dec=+66.5607°) must yield theta_polar≈0."""
        from master_thesis_code.galaxy_catalogue.handler import GalaxyCatalogueHandler

        df = synthetic_catalog_builder(n=1, sky_band="uniform", seed=42, already_rotated=False)
        df.iloc[0, df.columns.get_loc(InternalCatalogColumns.PHI_S)] = 270.0  # RA
        df.iloc[0, df.columns.get_loc(InternalCatalogColumns.THETA_S)] = 66.5607083  # Dec

        instance = object.__new__(GalaxyCatalogueHandler)
        instance.reduced_galaxy_catalog = df
        instance._rotate_equatorial_to_ecliptic()  # type: ignore[attr-defined]  # noqa: SLF001  (D-13: rotation is a separate __init__ step; type: ignore removed when [PHYSICS] COORD-03 lands the method)
        instance._map_angles_to_spherical_coordinates()  # noqa: SLF001

        theta_stored = float(
            instance.reduced_galaxy_catalog[InternalCatalogColumns.THETA_S].iloc[0]
        )
        tol_rad = 0.001 * np.pi / 180
        assert abs(theta_stored) < tol_rad, (
            f"Ecliptic pole should have theta_polar=0; got theta_polar={np.rad2deg(theta_stored):.4f}°. "
            "Missing equatorial->ecliptic rotation in handler.py:486-492."
        )


class TestVernalEquinoxRoundTrip:
    """Contract (Phase 35 D-05): RA=0° Dec=0° lies on BOTH equators. A
    bug that completely omits rotation will appear 'correct' here; useful
    as an on-axis anchor. Pairs with the summer-solstice + ecliptic-pole
    tests to triangulate the obliquity sign and magnitude.
    """

    @pytest.mark.xfail(strict=True, reason=_XFAIL_REASON)
    def test_vernal_equinox_galaxy_recovered_by_balltree(self) -> None:
        """Dec=0 galaxy (on both equators) must be retrievable via BallTree."""
        from master_thesis_code.galaxy_catalogue.handler import GalaxyCatalogueHandler

        df = synthetic_catalog_builder(n=1, sky_band="uniform", seed=42, already_rotated=False)
        df.iloc[0, df.columns.get_loc(InternalCatalogColumns.PHI_S)] = 0.0  # RA
        df.iloc[0, df.columns.get_loc(InternalCatalogColumns.THETA_S)] = 0.0  # Dec

        instance = object.__new__(GalaxyCatalogueHandler)
        instance.reduced_galaxy_catalog = df
        instance._rotate_equatorial_to_ecliptic()  # type: ignore[attr-defined]  # noqa: SLF001  (D-13: rotation is a separate __init__ step; type: ignore removed when [PHYSICS] COORD-03 lands the method)
        instance._map_angles_to_spherical_coordinates()  # noqa: SLF001
        instance.setup_galaxy_catalog_balltree()

        # Vernal equinox in ecliptic frame: lambda=0, beta=0, theta_polar=pi/2.
        phi, theta = 0.0, np.pi / 2
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        query = np.array([[x, y, z]])
        indices = instance.catalog_ball_tree.query_radius(query, r=0.01)[0]
        assert len(indices) == 1, (
            f"Vernal equinox galaxy must round-trip; got {len(indices)} matches. "
            "BallTree embedding bug (handler.py:286-288)."
        )


class TestNRandomEclipticEquatorRecovery:
    """Contract (Phase 35 D-06, D-07): N=100 random ecliptic-equator
    galaxies, seed=42, recovery rate >=99%.

    Threshold matches ROADMAP §Phase 36 success criterion 1 verbatim, so
    the handoff between Phase 35 xfail -> Phase 36 green has no
    ambiguity.
    """

    @pytest.mark.xfail(strict=True, reason=_XFAIL_REASON)
    def test_n_random_ecliptic_equator_galaxies_recovery_rate_above_99pct(self) -> None:
        """100 random ecliptic-equator galaxies; >=99 must round-trip via BallTree."""
        n = 100
        df = synthetic_catalog_builder(
            n=n, sky_band="ecliptic_equator", seed=42, already_rotated=True
        )
        tree = build_balltree(df)

        recovered = 0
        for i in range(n):
            phi = float(df[InternalCatalogColumns.PHI_S].iloc[i])
            theta = float(df[InternalCatalogColumns.THETA_S].iloc[i])
            # Polar-correct embedding:
            x = np.sin(theta) * np.cos(phi)
            y = np.sin(theta) * np.sin(phi)
            z = np.cos(theta)
            query = np.array([[x, y, z]])
            indices = tree.query_radius(query, r=0.01)[0]
            if i in indices:
                recovered += 1
        assert recovered >= 99, (
            f"Recovered {recovered}/{n} ecliptic-equator galaxies; >=99 required "
            "(matches ROADMAP Phase 36 success criterion 1 verbatim). "
            "BallTree embedding collapses theta=pi/2 points to (0, 0, 1) — "
            "singular query point (handler.py:286-288)."
        )
