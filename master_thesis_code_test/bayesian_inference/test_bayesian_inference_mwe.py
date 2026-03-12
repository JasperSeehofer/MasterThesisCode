import pytest

from master_thesis_code.bayesian_inference.bayesian_inference_mwe import (
    OMEGA_LAMBDA,
    OMEGA_M,
    SPEED_OF_LIGHT,
    TRUE_HUBBLE_CONSTANT,
    BayesianInference,
    EMRIDetection,
    Galaxy,
    GalaxyCatalog,
    NormalDist,
    dist,
    dist_array,
    dist_to_redshift,
    np,
    redshifted_mass,
    redshifted_mass_inverse,
)
from master_thesis_code.constants import FRACTIONAL_BLACK_HOLE_MASS_CATALOG_ERROR

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
def test_get_samples_from_comoving_volume_element() -> None:
    galaxy_catalog = GalaxyCatalog()

    redshifts = galaxy_catalog.get_samples_from_comoving_volume_element(100)

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
    number_to_take = int(number_of_possible_host_galaxies / 2)
    number_to_add = number_of_possible_host_galaxies - number_to_take
    used_host_galaxies = galaxy_catalog.get_possible_host_galaxies()[:number_to_take]

    new_hosts = galaxy_catalog.add_unique_host_galaxies_from_catalog(
        number_of_host_galaxies_to_add=number_to_add,
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


# ── comoving_volume_element ────────────────────────────────────────────────────


def test_comoving_volume_element_positive() -> None:
    catalog = GalaxyCatalog(use_truncnorm=False, use_comoving_volume=False)
    assert catalog.comoving_volume_element(1.0) > 0


def test_comoving_volume_element_monotonic() -> None:
    catalog = GalaxyCatalog(use_truncnorm=False, use_comoving_volume=False)
    v05 = catalog.comoving_volume_element(0.5)
    v10 = catalog.comoving_volume_element(1.0)
    v20 = catalog.comoving_volume_element(2.0)
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


# ── Known-bug regression: comoving_volume_element ignores cosmology ───────────


def test_comoving_volume_element_varies_with_hubble_constant() -> None:
    """comoving_volume_element should return different values for different H₀.

    The comoving volume element dV_c/dz ∝ (c/H₀)³, so at fixed redshift a higher H₀ should
    give a smaller comoving volume element.
    """
    catalog_low_h = GalaxyCatalog(use_truncnorm=False, use_comoving_volume=False, h0=0.70)
    catalog_high_h = GalaxyCatalog(use_truncnorm=False, use_comoving_volume=False, h0=0.73)

    v_low_h = catalog_low_h.comoving_volume_element(0.5)
    v_high_h = catalog_high_h.comoving_volume_element(0.5)

    # comoving volume element ∝ (c/H₀)³ → higher H₀ means smaller value
    assert v_low_h > v_high_h, f"Expected v(h=0.70)={v_low_h} > v(h=0.73)={v_high_h}"


# ── dist_array regression tests ───────────────────────────────────────────────


def test_dist_array_shape_matches_input() -> None:
    """dist_array must return a 1-D array with the same length as the input."""
    redshifts = np.linspace(0.01, 0.5, 50)
    result = dist_array(redshifts)
    assert result.shape == (50,)
    assert np.issubdtype(result.dtype, np.floating)


def test_dist_array_matches_scalar_dist() -> None:
    """Every element of dist_array(zs) must equal dist(z) for the same z."""
    test_redshifts = np.array([0.01, 0.05, 0.1, 0.2, 0.5])
    result = dist_array(test_redshifts)
    for i, z in enumerate(test_redshifts):
        assert abs(result[i] - dist(z)) < 1e-10, (
            f"dist_array mismatch at z={z}: {result[i]} vs {dist(z)}"
        )


def test_dist_array_monotonically_increasing() -> None:
    """Luminosity distance must be strictly increasing with redshift."""
    redshifts = np.linspace(0.001, 1.0, 100)
    result = dist_array(redshifts)
    diffs = np.diff(result)
    assert np.all(diffs > 0), "dist_array is not strictly monotonically increasing"


def test_dist_array_at_zero_redshift() -> None:
    """Luminosity distance at z=0 is 0 Mpc."""
    result = dist_array(np.array([0.0]))
    assert abs(float(result[0])) < 1e-10


# ── comoving_volume_element spline accuracy ──────────────────────────────────


def test_comoving_volume_element_spline_matches_integration() -> None:
    """GalaxyCatalog.comoving_volume_element (spline) must agree with direct quadrature to 0.1%."""
    catalog = GalaxyCatalog(use_truncnorm=False, use_comoving_volume=False)
    test_redshifts = np.linspace(0.01, 0.55, 20)
    for z in test_redshifts:
        zs = np.linspace(0.0, z, 500)
        e_z_val = np.sqrt(OMEGA_M * (1 + z) ** 3 + OMEGA_LAMBDA)
        integrand = 1.0 / np.sqrt(OMEGA_M * (1 + zs) ** 3 + OMEGA_LAMBDA)
        integral = np.trapezoid(integrand, zs)
        # Eq. 27 in Hogg (1999), arXiv:astro-ph/9905116
        expected = float(
            4 * np.pi * (SPEED_OF_LIGHT / TRUE_HUBBLE_CONSTANT) ** 3 * integral**2 / e_z_val
        )
        actual = catalog.comoving_volume_element(z)
        assert abs(actual - expected) / expected < 1e-3, (
            f"Spline deviates >0.1% at z={z}: spline={actual:.6g}, quad={expected:.6g}"
        )


def test_comoving_volume_element_is_zero_at_z_zero() -> None:
    """Comoving volume element dV_c/dz at z=0 should be essentially 0."""
    catalog = GalaxyCatalog()
    assert abs(catalog.comoving_volume_element(0.0)) < 1e-10


# ── likelihood vectorized path ────────────────────────────────────────────────


# ── Mass distribution sigma regression (TEST-2) ─────────────────────────────


@pytest.mark.xfail(
    strict=True, reason="PHYS-2: setup_galaxy_mass_distribution uses hardcoded 10**5.5"
)
def test_setup_galaxy_mass_distribution_sigma_scales_with_mass() -> None:
    """setup_galaxy_mass_distribution sigma must scale with each galaxy's mass, not 10**5.5."""
    catalog = GalaxyCatalog(use_truncnorm=False, use_comoving_volume=False)
    g1 = Galaxy(redshift=0.1, central_black_hole_mass=1e4, right_ascension=1.0, declination=0.5)
    g2 = Galaxy(redshift=0.2, central_black_hole_mass=1e7, right_ascension=2.0, declination=1.0)
    catalog.catalog.extend([g1, g2])
    catalog.setup_galaxy_mass_distribution()

    assert isinstance(catalog.galaxy_mass_distribution[0], NormalDist)
    assert isinstance(catalog.galaxy_mass_distribution[1], NormalDist)
    assert (
        abs(
            catalog.galaxy_mass_distribution[0].stdev
            - FRACTIONAL_BLACK_HOLE_MASS_CATALOG_ERROR * 1e4
        )
        < 1e-6
    )
    assert (
        abs(
            catalog.galaxy_mass_distribution[1].stdev
            - FRACTIONAL_BLACK_HOLE_MASS_CATALOG_ERROR * 1e7
        )
        < 1e-1
    )


def test_append_galaxy_mass_distribution_sigma_scales_with_mass() -> None:
    """append_galaxy_to_galaxy_mass_distribution sigma must scale with each galaxy's mass."""
    catalog = GalaxyCatalog(use_truncnorm=False, use_comoving_volume=False)
    g1 = Galaxy(redshift=0.1, central_black_hole_mass=1e4, right_ascension=1.0, declination=0.5)
    g2 = Galaxy(redshift=0.2, central_black_hole_mass=1e7, right_ascension=2.0, declination=1.0)
    catalog.append_galaxy_to_galaxy_mass_distribution(g1)
    catalog.append_galaxy_to_galaxy_mass_distribution(g2)

    assert isinstance(catalog.galaxy_mass_distribution[0], NormalDist)
    assert isinstance(catalog.galaxy_mass_distribution[1], NormalDist)
    assert (
        abs(
            catalog.galaxy_mass_distribution[0].stdev
            - FRACTIONAL_BLACK_HOLE_MASS_CATALOG_ERROR * 1e4
        )
        < 1e-6
    )
    assert (
        abs(
            catalog.galaxy_mass_distribution[1].stdev
            - FRACTIONAL_BLACK_HOLE_MASS_CATALOG_ERROR * 1e7
        )
        < 1e-1
    )


# ── likelihood vectorized path ────────────────────────────────────────────────


def test_likelihood_output_is_finite_and_positive() -> None:
    """BayesianInference.likelihood() must return a finite positive float."""
    catalog = GalaxyCatalog(use_truncnorm=False, use_comoving_volume=False)
    catalog.create_random_catalog(10)
    detection = EMRIDetection.from_host_galaxy(catalog.catalog[0], use_measurement_noise=False)
    bi = BayesianInference(catalog, [detection])
    result = bi.likelihood(
        hubble_constant=TRUE_HUBBLE_CONSTANT,
        measured_luminosity_distance=detection.measured_luminosity_distance,
        measured_redshifted_mass=detection.measured_redshifted_mass,
        detection_index=0,
    )
    assert isinstance(result, float)
    assert np.isfinite(result)
    assert result > 0
