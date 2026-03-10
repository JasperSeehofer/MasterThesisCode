import pytest

from master_thesis_code.bayesian_inference.bayesian_inference_mwe import (
    TRUE_HUBBLE_CONSTANT,
    BayesianInference,
    EMRIDetection,
    Galaxy,
    GalaxyCatalog,
    NormalDist,
    dist,
    dist_to_redshift,
    np,
    redshifted_mass,
    redshifted_mass_inverse,
)

LUMINOSITY_DISTANCE_REDSHIFT_PAIRS = [(0.0, 0.0)]

distance_pymark = pytest.mark.parametrize(
    "luminosity_distance,redshift", LUMINOSITY_DISTANCE_REDSHIFT_PAIRS
)


@distance_pymark
def test_dist(luminosity_distance: float, redshift: float) -> None:
    result = dist(redshift)
    assert result is not None
    assert isinstance(result, float)
    assert round(result, 10) == luminosity_distance


@distance_pymark
def test_dist_to_redshift(luminosity_distance: float, redshift: float) -> None:
    result = dist_to_redshift(luminosity_distance)
    assert result is not None
    assert isinstance(result, float)
    assert round(result, 10) == redshift


# TODO test lambda_cdm_analytic_distance

# test Galaxy datamodel


def test_galaxy_with_random_skylocalization() -> None:
    redshift = 1.0
    central_black_hole_mass = 1e5
    random_galaxies = [
        Galaxy.with_random_skylocalization(redshift, central_black_hole_mass) for _ in range(10)
    ]
    assert random_galaxies is not None
    assert len(random_galaxies) == 10
    assert all(isinstance(galaxy, Galaxy) for galaxy in random_galaxies)
    assert all(galaxy.redshift == redshift for galaxy in random_galaxies)
    assert all(
        galaxy.central_black_hole_mass == central_black_hole_mass for galaxy in random_galaxies
    )
    assert all(0 <= galaxy.right_ascension < 2 * np.pi for galaxy in random_galaxies)
    assert all(0 <= galaxy.declination < np.pi for galaxy in random_galaxies)


# test GalaxyCatalog
def test_galaxy_catalog_init() -> None:
    galaxy_catalog = GalaxyCatalog()

    assert galaxy_catalog is not None
    assert isinstance(galaxy_catalog, GalaxyCatalog)
    assert len(galaxy_catalog.catalog) == 0
    assert galaxy_catalog._use_truncnorm
    assert galaxy_catalog._use_comoving_volume


# test comoving volume TODO
def test_get_samples_from_comoving_volume() -> None:
    galaxy_catalog = GalaxyCatalog()

    redshifts = galaxy_catalog.get_samples_from_comoving_volume(100)

    assert redshifts is not None
    assert isinstance(redshifts, np.ndarray)
    assert redshifts.__len__() == 100
    assert all(isinstance(sample, float) for sample in redshifts)


def test_create_random_catalog_without_comoving_volume() -> None:
    galaxy_catalog = GalaxyCatalog(use_comoving_volume=False)

    number_of_galaxies = 10

    galaxy_catalog.create_random_catalog(number_of_galaxies=number_of_galaxies)

    assert galaxy_catalog.catalog is not None
    assert len(galaxy_catalog.catalog) == 10
    assert all(
        galaxy_catalog.redshift_lower_limit
        <= galaxy.redshift
        <= galaxy_catalog.redshift_upper_limit
        for galaxy in galaxy_catalog.catalog
    )
    assert all(
        galaxy_catalog.lower_mass_limit
        <= galaxy.central_black_hole_mass
        <= galaxy_catalog.upper_mass_limit
        for galaxy in galaxy_catalog.catalog
    )


def test_create_random_catalog_with_comoving_volume() -> None:
    galaxy_catalog = GalaxyCatalog()

    number_of_galaxies = 10

    galaxy_catalog.create_random_catalog(number_of_galaxies=number_of_galaxies)

    assert galaxy_catalog.catalog is not None
    assert galaxy_catalog.catalog.__len__() == 10
    assert all(
        galaxy_catalog.redshift_lower_limit
        <= galaxy.redshift
        <= galaxy_catalog.redshift_upper_limit
        for galaxy in galaxy_catalog.catalog
    )
    assert all(
        galaxy_catalog.lower_mass_limit
        <= galaxy.central_black_hole_mass
        <= galaxy_catalog.upper_mass_limit
        for galaxy in galaxy_catalog.catalog
    )


def test_remove_all_galaxies() -> None:
    galaxy_catalog = GalaxyCatalog()
    galaxy_catalog.create_random_catalog(100)

    galaxy_catalog.remove_all_galaxies()

    assert galaxy_catalog.catalog.__len__() == 0
    assert galaxy_catalog.galaxy_distribution.__len__() == 0
    assert galaxy_catalog.galaxy_mass_distribution.__len__() == 0


def test_add_random_galaxy() -> None:
    galaxy_catalog = GalaxyCatalog()

    galaxy_catalog.add_random_galaxy()

    assert galaxy_catalog.catalog.__len__() == 1
    assert galaxy_catalog.galaxy_distribution.__len__() == 1
    assert galaxy_catalog.galaxy_mass_distribution.__len__() == 1


def test_add_random_host_galaxy() -> None:
    galaxy_catalog = GalaxyCatalog()

    galaxy_catalog.add_host_galaxy()

    assert galaxy_catalog.catalog.__len__() == 1
    assert galaxy_catalog.galaxy_distribution.__len__() == 1
    assert galaxy_catalog.galaxy_mass_distribution.__len__() == 1


def test_setup_galaxy_distribution_without_truncnorm() -> None:
    galaxy_catalog = GalaxyCatalog(use_truncnorm=False)

    galaxy_catalog.catalog.extend(
        [
            Galaxy.with_random_skylocalization(
                redshift=1.0,
                central_black_hole_mass=1e5,
            ),
            Galaxy.with_random_skylocalization(
                redshift=2.0,
                central_black_hole_mass=1e6,
            ),
        ]
    )

    galaxy_catalog.setup_galaxy_distribution()

    assert galaxy_catalog.galaxy_distribution is not None
    assert galaxy_catalog.galaxy_distribution.__len__() == 2
    assert all(
        isinstance(distribution, NormalDist) for distribution in galaxy_catalog.galaxy_distribution
    )


def test_setup_galaxy_mass_distribution() -> None:
    galaxy_catalog = GalaxyCatalog(use_truncnorm=False)
    galaxy_catalog.catalog.extend(
        [
            Galaxy.with_random_skylocalization(
                redshift=1.0,
                central_black_hole_mass=1e5,
            ),
            Galaxy.with_random_skylocalization(
                redshift=2.0,
                central_black_hole_mass=1e6,
            ),
        ]
    )

    galaxy_catalog.setup_galaxy_mass_distribution()

    assert galaxy_catalog.galaxy_mass_distribution is not None
    assert len(galaxy_catalog.galaxy_mass_distribution) == 2
    assert all(
        isinstance(distribution, NormalDist)
        for distribution in galaxy_catalog.galaxy_mass_distribution
    )


def test_evaluate_galaxy_distribution() -> None:
    galaxy_catalog = GalaxyCatalog(use_truncnorm=False)
    galaxy_catalog.catalog.extend(
        [
            Galaxy.with_random_skylocalization(
                redshift=1.0,
                central_black_hole_mass=1e5,
            ),
            Galaxy.with_random_skylocalization(
                redshift=2.0,
                central_black_hole_mass=1e6,
            ),
        ]
    )

    probabilities = galaxy_catalog.evaluate_galaxy_distribution(1.5)

    assert probabilities is not None
    assert isinstance(probabilities, np.ndarray)
    assert len(probabilities) == 2
    assert all(prob >= 0 for prob in probabilities)


def test_evaluate_galaxy_mass_distribution() -> None:
    galaxy_catalog = GalaxyCatalog(use_truncnorm=False)
    galaxy_catalog.catalog.extend(
        [
            Galaxy.with_random_skylocalization(
                redshift=1.0,
                central_black_hole_mass=1e5,
            ),
            Galaxy.with_random_skylocalization(
                redshift=2.0,
                central_black_hole_mass=1e6,
            ),
        ]
    )
    galaxy_catalog.setup_galaxy_mass_distribution()

    probabilities = galaxy_catalog.evaluate_galaxy_mass_distribution(1e5)

    assert probabilities is not None
    assert isinstance(probabilities, np.ndarray)
    assert len(probabilities) == 2
    assert all(prob >= 0 for prob in probabilities)


def test_get_possible_host_galaxies() -> None:
    galaxy_catalog = GalaxyCatalog()
    galaxy_catalog.catalog.extend(
        [
            Galaxy.with_random_skylocalization(
                redshift=0.1,
                central_black_hole_mass=1e5,
            ),
            Galaxy.with_random_skylocalization(
                redshift=2.0,
                central_black_hole_mass=1e6,
            ),
        ]
    )

    possible_hosts = galaxy_catalog.get_possible_host_galaxies()

    assert possible_hosts is not None
    assert isinstance(possible_hosts, list)
    assert all(isinstance(galaxy, Galaxy) for galaxy in possible_hosts)
    assert all(
        dist(galaxy.redshift, TRUE_HUBBLE_CONSTANT)
        <= BayesianInference.luminosity_distance_threshold
        for galaxy in possible_hosts
    )


def test_gw_detection_probability() -> None:
    galaxy_catalog = GalaxyCatalog()
    emri_detections: list[EMRIDetection] = []
    bayesian_inference = BayesianInference(galaxy_catalog, emri_detections)

    probability = bayesian_inference.gw_detection_probability(0.1, TRUE_HUBBLE_CONSTANT)

    assert probability is not None
    assert isinstance(probability, float)
    assert 0 <= probability <= 1


def test_gw_likelihood() -> None:
    galaxy_catalog = GalaxyCatalog()
    emri_detections: list[EMRIDetection] = []
    bayesian_inference = BayesianInference(galaxy_catalog, emri_detections)

    likelihood = bayesian_inference.gw_likelihood(
        measured_luminosity_distance=1000.0, redshift=0.1, hubble_constant=TRUE_HUBBLE_CONSTANT
    )

    assert likelihood is not None
    assert isinstance(likelihood, float)
    assert likelihood >= 0


def test_posterior() -> None:
    galaxy_catalog = GalaxyCatalog()
    galaxy_catalog.create_random_catalog(10)
    emri_detections = [
        EMRIDetection.from_host_galaxy(galaxy) for galaxy in galaxy_catalog.catalog[:2]
    ]
    bayesian_inference = BayesianInference(galaxy_catalog, emri_detections)

    posterior = bayesian_inference.posterior(TRUE_HUBBLE_CONSTANT)

    assert posterior is not None
    assert isinstance(posterior, list)
    assert len(posterior) == len(emri_detections)
    assert all(prob >= 0 for prob in posterior)


def test_add_unique_host_galaxies_from_catalog() -> None:
    galaxy_catalog = GalaxyCatalog()
    galaxy_catalog.create_random_catalog(20)
    number_of_possible_host_galaxies = len(galaxy_catalog.get_possible_host_galaxies())
    used_host_galaxies = galaxy_catalog.get_possible_host_galaxies()[
        : int(number_of_possible_host_galaxies / 2)
    ]

    new_hosts = galaxy_catalog.add_unique_host_galaxies_from_catalog(
        number_of_host_galaxies_to_add=int(number_of_possible_host_galaxies / 2),
        used_host_galaxies=used_host_galaxies,
    )

    assert new_hosts is not None
    assert isinstance(new_hosts, list)
    assert len(new_hosts) == number_of_possible_host_galaxies
    assert all(isinstance(galaxy, Galaxy) for galaxy in new_hosts)
    assert len(set(new_hosts)) == len(new_hosts)  # Ensure uniqueness


# ── Galaxy hashability ─────────────────────────────────────────────────────────


def test_galaxy_hashable() -> None:
    g1 = Galaxy(redshift=0.1, central_black_hole_mass=1e5, right_ascension=1.0, declination=0.5)
    g2 = Galaxy(redshift=0.2, central_black_hole_mass=1e5, right_ascension=1.0, declination=0.5)
    s = {g1, g2}
    assert len(s) == 2


def test_galaxy_hash_same_fields() -> None:
    g1 = Galaxy(redshift=0.1, central_black_hole_mass=1e5, right_ascension=1.0, declination=0.5)
    g2 = Galaxy(redshift=0.1, central_black_hole_mass=1e5, right_ascension=1.0, declination=0.5)
    assert hash(g1) == hash(g2)
    assert g1 == g2


def test_galaxy_equality_different_redshift() -> None:
    g1 = Galaxy(redshift=0.1, central_black_hole_mass=1e5, right_ascension=1.0, declination=0.5)
    g2 = Galaxy(redshift=0.2, central_black_hole_mass=1e5, right_ascension=1.0, declination=0.5)
    assert g1 != g2


# ── redshifted_mass functions ──────────────────────────────────────────────────


def test_redshifted_mass_at_zero_redshift() -> None:
    m = 1e5
    assert redshifted_mass(m, 0) == m


def test_redshifted_mass_formula() -> None:
    m = 1e5
    z = 0.5
    assert abs(redshifted_mass(m, z) - m * (1 + z)) < 1e-10


def test_redshifted_mass_inverse_round_trip() -> None:
    m = 1e5
    z = 0.5
    m_z = redshifted_mass(m, z)
    m_back = redshifted_mass_inverse(m_z, z)
    assert abs(m_back - m) < 1e-10


# ── dist in bayesian_inference_mwe ────────────────────────────────────────────


def test_dist_mwe_at_zero() -> None:
    """dist(0) in bayesian_inference_mwe should return 0.0."""
    result = dist(0)
    assert result == 0.0


def test_dist_round_trip_mwe() -> None:
    result = dist_to_redshift(dist(0.5))
    assert abs(result - 0.5) < 1e-5


# ── comoving_volume ────────────────────────────────────────────────────────────


def test_comoving_volume_positive() -> None:
    catalog = GalaxyCatalog(use_truncnorm=False, use_comoving_volume=False)
    assert catalog.comoving_volume(1.0) > 0


def test_comoving_volume_monotonic() -> None:
    catalog = GalaxyCatalog(use_truncnorm=False, use_comoving_volume=False)
    v05 = catalog.comoving_volume(0.5)
    v10 = catalog.comoving_volume(1.0)
    v20 = catalog.comoving_volume(2.0)
    assert v05 < v10 < v20


# ── EMRIDetection tuple-comma bug regression ───────────────────────────────────


def test_emri_detection_no_noise_fields_are_float() -> None:
    """Regression: use_measurement_noise=False must produce floats, not tuples."""
    g = Galaxy(
        redshift=0.1,
        central_black_hole_mass=1e5,
        right_ascension=1.0,
        declination=0.5,
    )
    detection = EMRIDetection.from_host_galaxy(g, use_measurement_noise=False)
    assert isinstance(detection.measured_luminosity_distance, float)
    assert isinstance(detection.measured_redshifted_mass, float)


# ── GalaxyCatalog truncnorm distribution type ─────────────────────────────────


def test_galaxy_catalog_truncnorm_distribution_type() -> None:
    """With use_truncnorm=True the distributions must NOT be NormalDist instances."""
    catalog = GalaxyCatalog(use_truncnorm=True, use_comoving_volume=False)
    catalog.add_random_galaxy()
    assert not isinstance(catalog.galaxy_distribution[0], NormalDist)


# ── gw_detection_probability bounds ───────────────────────────────────────────


def test_gw_detection_probability_near_zero_redshift() -> None:
    """Very small redshift → source is nearby → detection probability close to 1."""
    catalog = GalaxyCatalog()
    bayesian_inference = BayesianInference(catalog, [])
    prob = bayesian_inference.gw_detection_probability(0.001, TRUE_HUBBLE_CONSTANT)
    assert prob > 0.9


def test_gw_detection_probability_large_redshift() -> None:
    """Large redshift → source is far → detection probability should be low."""
    catalog = GalaxyCatalog()
    bayesian_inference = BayesianInference(catalog, [])
    prob = bayesian_inference.gw_detection_probability(2.0, TRUE_HUBBLE_CONSTANT)
    assert prob < 0.5


# ── posterior length ──────────────────────────────────────────────────────────


def test_posterior_length_matches_detections() -> None:
    catalog = GalaxyCatalog(use_truncnorm=False, use_comoving_volume=False)
    catalog.create_random_catalog(5)
    detections = [
        EMRIDetection.from_host_galaxy(g, use_measurement_noise=False) for g in catalog.catalog[:2]
    ]
    bayesian_inference = BayesianInference(catalog, detections)
    posterior = bayesian_inference.posterior(TRUE_HUBBLE_CONSTANT)
    assert len(posterior) == len(detections)


# ── Known-bug regression: comoving_volume ignores cosmology ───────────────────


@pytest.mark.xfail(
    reason=(
        "Known bug: GalaxyCatalog.comoving_volume() uses the hardcoded module-level "
        "TRUE_HUBBLE_CONSTANT (0.7) and therefore returns the same value regardless of "
        "the Hubble constant passed in.  This test documents the expected correct behaviour: "
        "a higher H₀ should produce a smaller comoving volume at the same redshift."
    )
)
def test_comoving_volume_varies_with_hubble_constant() -> None:
    """comoving_volume should return different values for different H₀ (currently it doesn't).

    The comoving volume element dV/dz ∝ (c/H₀)³, so at fixed redshift a higher H₀ should
    give a smaller comoving volume.  The method currently ignores any cosmology argument
    and uses the hardcoded TRUE_HUBBLE_CONSTANT = 0.7.
    """
    catalog = GalaxyCatalog(use_truncnorm=False, use_comoving_volume=False)

    # If the bug is present both calls return the same value because TRUE_HUBBLE_CONSTANT
    # is hardcoded.  This test will xfail until comoving_volume accepts an h parameter.
    v_low_h = catalog.comoving_volume(0.5)  # would use h=0.70
    v_high_h = catalog.comoving_volume(0.5)  # would use h=0.73 — currently same as above

    # comoving volume ∝ (c/H₀)³ → higher H₀ means smaller volume
    # This assertion is expected to fail (hence xfail) as long as the bug exists.
    assert v_low_h != v_high_h, (
        "comoving_volume returned the same value for different H₀ — bug still present"
    )
