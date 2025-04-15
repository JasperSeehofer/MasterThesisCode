import pytest
from typing import Tuple
from master_thesis_code.bayesian_inference.bayesian_inference_mwe import *

LUMINOSITY_DISTANCE_REDSHIFT_PAIRS = [(0.0, 0.0)]

distance_pymark = pytest.mark.parametrize("luminosity_distance,redshift", LUMINOSITY_DISTANCE_REDSHIFT_PAIRS)

@distance_pymark
def test_dist(luminosity_distance: float, redshift: float):  
    result = dist(redshift)
    assert result is not None
    assert isinstance(result, float)
    assert round(result, 10) == luminosity_distance

@distance_pymark
def test_dist_to_redshift(luminosity_distance: float, redshift: float):
    result = dist_to_redshift(luminosity_distance)
    assert result is not None
    assert isinstance(result, float)
    assert round(result, 10) == redshift

# TODO test lambda_cdm_analytic_distance

# test Galaxy datamodel

def test_galaxy_with_random_skylocalization():
    redshift = 1.
    central_black_hole_mass = 1e5
    random_galaxies = [Galaxy.with_random_skylocalization(redshift, central_black_hole_mass) for _ in range(10)]
    assert random_galaxies is not None
    assert len(random_galaxies) == 10
    assert all(isinstance(galaxy, Galaxy) for galaxy in random_galaxies)
    assert all(galaxy.redshift == redshift for galaxy in random_galaxies)
    assert all(galaxy.central_black_hole_mass == central_black_hole_mass for galaxy in random_galaxies)
    assert all(0 <= galaxy.right_ascension < 2*np.pi for galaxy in random_galaxies)
    assert all(0 <= galaxy.declination < np.pi for galaxy in random_galaxies)

# test GalaxyCatalog
def test_galaxy_catalog_init():
    galaxy_catalog = GalaxyCatalog()

    assert galaxy_catalog is not None
    assert isinstance(galaxy_catalog, GalaxyCatalog)
    assert galaxy_catalog.catalog.__len__() == 0
    assert galaxy_catalog._use_truncnorm == True
    assert galaxy_catalog._use_comoving_volume == True

# test comoving volume TODO

def test_create_random_catalog_without_comoving_volume():
    galaxy_catalog = GalaxyCatalog(use_comoving_volume=False)
    
    number_of_galaxies = 10

    galaxy_catalog.create_random_catalog(number_of_galaxies=number_of_galaxies)

    assert galaxy_catalog.catalog is not None
    assert galaxy_catalog.catalog.__len__() == 10
    assert all(
        galaxy_catalog.redshift_lower_limit <= galaxy.redshift <= galaxy_catalog.redshift_upper_limit 
        for galaxy in galaxy_catalog.catalog
        )
    assert all(
        galaxy_catalog.lower_mass_limit <= galaxy.central_black_hole_mass <= galaxy_catalog.upper_mass_limit 
        for galaxy in galaxy_catalog.catalog
        )

def test_create_random_catalog_with_comoving_volume():
    galaxy_catalog = GalaxyCatalog()
    
    number_of_galaxies = 10

    galaxy_catalog.create_random_catalog(number_of_galaxies=number_of_galaxies)

    assert galaxy_catalog.catalog is not None
    assert galaxy_catalog.catalog.__len__() == 10
    assert all(
        galaxy_catalog.redshift_lower_limit <= galaxy.redshift <= galaxy_catalog.redshift_upper_limit 
        for galaxy in galaxy_catalog.catalog
        )
    assert all(
        galaxy_catalog.lower_mass_limit <= galaxy.central_black_hole_mass <= galaxy_catalog.upper_mass_limit 
        for galaxy in galaxy_catalog.catalog
        )
    
def test_remove_all_galaxies():
    galaxy_catalog = GalaxyCatalog()
    galaxy_catalog.create_random_catalog(100)

    galaxy_catalog.remove_all_galaxies()

    assert galaxy_catalog.catalog.__len__() == 0
    assert galaxy_catalog.galaxy_distribution.__len__() == 0
    assert galaxy_catalog.galaxy_mass_distribution.__len__() == 0

def test_add_random_galaxy():
    galaxy_catalog = GalaxyCatalog()

    galaxy_catalog.add_random_galaxy()

    assert galaxy_catalog.catalog.__len__() == 1
    assert galaxy_catalog.galaxy_distribution.__len__() == 1
    assert galaxy_catalog.galaxy_mass_distribution.__len__() == 1

def test_add_random_host_galaxy():
    galaxy_catalog = GalaxyCatalog()

    galaxy_catalog.add_host_galaxy()

    assert galaxy_catalog.catalog.__len__() == 1
    assert galaxy_catalog.galaxy_distribution.__len__() == 1
    assert galaxy_catalog.galaxy_mass_distribution.__len__() == 1

def test_setup_galaxy_distribution_without_truncnorm():
    galaxy_catalog = GalaxyCatalog(use_truncnorm=False)

    galaxy_catalog.catalog.extend([
        Galaxy.with_random_skylocalization(
        redshift=1.0,
        central_black_hole_mass=1e5,
        ),
        Galaxy.with_random_skylocalization(
            redshift=2.0,
            central_black_hole_mass=1e6,
        )
    ])

    galaxy_catalog.setup_galaxy_distribution()

    assert galaxy_catalog.galaxy_distribution is not None
    assert galaxy_catalog.galaxy_distribution.__len__() == 2
    assert all(isinstance(distribution, NormalDist) for distribution in galaxy_catalog.galaxy_distribution)



