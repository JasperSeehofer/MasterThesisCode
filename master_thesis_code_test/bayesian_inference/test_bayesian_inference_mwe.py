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
