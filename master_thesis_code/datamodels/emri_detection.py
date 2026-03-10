"""EMRIDetection datamodel for Bayesian H₀ inference."""

from dataclasses import dataclass
from statistics import NormalDist

import numpy as np

from master_thesis_code.bayesian_inference.scientific_plotter import ScientificPlotter
from master_thesis_code.constants import (
    FRACTIONAL_LUMINOSITY_ERROR,
    FRACTIONAL_MEASURED_MASS_ERROR,
    TRUE_HUBBLE_CONSTANT,
)
from master_thesis_code.datamodels.galaxy import Galaxy, GalaxyCatalog
from master_thesis_code.physical_relations import dist, redshifted_mass


@dataclass
class EMRIDetection:
    """A simulated EMRI detection with noisy measured observables.

    Holds the LISA-measured quantities that enter the Bayesian H₀ likelihood.
    All angular quantities use ecliptic coordinates.

    Attributes:
        measured_luminosity_distance: Measured luminosity distance :math:`\\hat{d}_L`
            in Gpc.
        measured_redshifted_mass: Measured redshifted central BH mass
            :math:`\\hat{M}_z = M(1+z)` in solar masses.
        measured_right_ascension: Measured sky azimuthal angle in radians.
        measured_declination: Measured sky polar angle in radians.
    """

    measured_luminosity_distance: float  # Gpc
    measured_redshifted_mass: float  # solar masses
    measured_right_ascension: float  # radians
    measured_declination: float  # radians

    @classmethod
    def from_host_galaxy(
        cls, host_galaxy: Galaxy, use_measurement_noise: bool = True
    ) -> "EMRIDetection":
        if not use_measurement_noise:
            measured_luminosity_distance = dist(host_galaxy.redshift, TRUE_HUBBLE_CONSTANT)
            measured_redshifted_mass = redshifted_mass(
                mass=host_galaxy.central_black_hole_mass,
                redshift=host_galaxy.redshift,
            )
        else:
            measured_luminosity_distance = np.random.normal(
                loc=dist(host_galaxy.redshift, TRUE_HUBBLE_CONSTANT),
                scale=FRACTIONAL_LUMINOSITY_ERROR
                * dist(
                    host_galaxy.redshift,
                    TRUE_HUBBLE_CONSTANT,
                ),
            )
            measured_redshifted_mass = np.random.normal(
                loc=redshifted_mass(
                    mass=host_galaxy.central_black_hole_mass,
                    redshift=host_galaxy.redshift,
                ),
                scale=FRACTIONAL_MEASURED_MASS_ERROR
                * redshifted_mass(
                    mass=host_galaxy.central_black_hole_mass,
                    redshift=host_galaxy.redshift,
                ),
            )

        return cls(
            measured_luminosity_distance=measured_luminosity_distance,
            measured_redshifted_mass=measured_redshifted_mass,
            measured_right_ascension=host_galaxy.right_ascension,
            measured_declination=host_galaxy.declination,
        )

    @classmethod
    def plot_detection_distribution(cls, host_galaxies: list[Galaxy]) -> None:
        detection_distribution = [
            NormalDist(mu=galaxy.redshift, sigma=FRACTIONAL_LUMINOSITY_ERROR * galaxy.redshift)
            for galaxy in host_galaxies
        ]
        redshifts = np.linspace(
            GalaxyCatalog.redshift_lower_limit, GalaxyCatalog.redshift_upper_limit, 1000
        )
        detection_probabilities = [
            np.sum([distribution.pdf(redshift) for distribution in detection_distribution])
            for redshift in redshifts
        ]
        _plotter = ScientificPlotter(figure_size=(16, 9))
        _plotter.plot(redshifts, np.array(detection_probabilities))
        _plotter.show_and_close()

    @classmethod
    def plot_detection_sky_distribution(cls, host_galaxies: list[Galaxy]) -> None:
        right_ascensions = [galaxy.right_ascension for galaxy in host_galaxies]
        declinations = [galaxy.declination for galaxy in host_galaxies]

        _plotter = ScientificPlotter(figure_size=(16, 9))
        _plotter.scatter(np.array(right_ascensions), np.array(declinations), "o")
        _plotter.show_and_close()
