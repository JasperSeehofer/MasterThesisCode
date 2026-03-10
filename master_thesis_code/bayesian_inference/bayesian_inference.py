"""BayesianInference class and dist_array helper for Hubble constant estimation."""

from dataclasses import dataclass, field
from statistics import NormalDist

import numpy as np
import numpy.typing as npt
from scipy.special import erf

from master_thesis_code.bayesian_inference.scientific_plotter import ScientificPlotter
from master_thesis_code.constants import (
    FRACTIONAL_LUMINOSITY_ERROR,
    FRACTIONAL_MEASURED_MASS_ERROR,
    LUMINOSITY_DISTANCE_THRESHOLD_GPC,
    OMEGA_DE,
    OMEGA_M,
    SKY_LOCALIZATION_ERROR,
    TRUE_HUBBLE_CONSTANT,
    H,
)
from master_thesis_code.datamodels.emri_detection import EMRIDetection
from master_thesis_code.datamodels.galaxy import GalaxyCatalog
from master_thesis_code.physical_relations import (
    dist,
    redshifted_mass_inverse,
)
from master_thesis_code.physical_relations import (
    dist_vectorized as _dist_vectorized,
)


def dist_array(
    redshifts: npt.NDArray[np.float64],
    h: float = H,
    Omega_m: float = OMEGA_M,
    Omega_de: float = OMEGA_DE,
) -> npt.NDArray[np.float64]:
    """Vectorized luminosity distance in Gpc over an array of redshifts.

    Delegates to physical_relations.dist_vectorized for a canonical, unit-consistent
    implementation.  Returns Gpc (same unit as the scalar dist()).
    """
    return np.asarray(_dist_vectorized(redshifts, h=h, Omega_m=Omega_m, Omega_de=Omega_de))


@dataclass
class BayesianInference:
    """Bayesian estimator for the Hubble constant H₀ from EMRI detections.

    Evaluates :math:`p(H_0 \\mid \\{d_i\\})` by marginalizing each detection over
    the galaxy-catalog redshift distribution, weighted by the GW likelihood and
    sky-localization overlap.

    Attributes:
        galaxy_catalog: Galaxy catalog providing redshift and mass distributions
            :math:`p(z)` and :math:`p(M)`.
        emri_detections: List of EMRI detections with measured luminosity distances
            and sky positions.
        luminosity_distance_threshold: LISA detection horizon in Gpc.
        number_of_redshift_steps: Number of quadrature points for redshift integrals.
        redshift_values: Redshift grid used for all integrals, set in ``__post_init__``.
        galaxy_distribution_at_redshifts: Pre-computed :math:`p(z_i \\mid \\mathrm{catalog})`
            evaluated on ``redshift_values``.
        use_bh_mass: If ``True``, include BH mass information in the likelihood.
        use_selection_effects_correction: If ``True``, divide likelihood by the
            selection-function denominator :math:`\\int p_\\mathrm{det}(z) p(z) dz`.
    """

    galaxy_catalog: GalaxyCatalog
    emri_detections: list[EMRIDetection]

    luminosity_distance_threshold: float = LUMINOSITY_DISTANCE_THRESHOLD_GPC  # Gpc
    number_of_redshift_steps: int = 1000
    redshift_values: npt.NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    galaxy_distribution_at_redshifts: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.array([])
    )
    galaxy_detection_mass_distribution_at_redshifts: list = field(default_factory=list)
    detection_skylocalization_weight_by_galaxy: list = field(default_factory=list)
    use_bh_mass: bool = False
    use_selection_effects_correction: bool = True

    def __post_init__(self) -> None:
        self.redshift_values = np.linspace(
            self.galaxy_catalog.redshift_lower_limit,
            self.galaxy_catalog.redshift_upper_limit,
            self.number_of_redshift_steps,
        )
        self.galaxy_distribution_at_redshifts = np.array(
            [
                self.galaxy_catalog.evaluate_galaxy_distribution(redshift)
                for redshift in self.redshift_values
            ]
        )
        self.comoving_volume_at_redshifts = np.array(
            [self.galaxy_catalog.comoving_volume(redshift) for redshift in self.redshift_values]
        )
        self.galaxy_detection_mass_distribution_at_redshifts = [
            np.array(
                [
                    self.galaxy_catalog.evaluate_galaxy_mass_distribution(
                        redshifted_mass_inverse(
                            redshifted_mass=emri_detection.measured_redshifted_mass,
                            redshift=redshift,
                        )
                    )
                    for redshift in self.redshift_values
                ]
            )
            for emri_detection in self.emri_detections
        ]

        self.detection_skylocalization_weight_by_galaxy = [
            np.array(
                [
                    NormalDist(
                        mu=emri_detection.measured_right_ascension,
                        sigma=SKY_LOCALIZATION_ERROR,
                    ).pdf(galaxy.right_ascension)
                    * NormalDist(
                        mu=emri_detection.measured_declination,
                        sigma=SKY_LOCALIZATION_ERROR,
                    ).pdf(galaxy.declination)
                    for galaxy in self.galaxy_catalog.catalog
                ]
            )
            for emri_detection in self.emri_detections
        ]

    def gw_detection_probability(self, redshift: float, hubble_constant: float) -> float:
        """Probability that a source at *redshift* is detected by LISA.

        Models the detector selection function as a cumulative normal:

        .. math::

            p_\\mathrm{det}(z) = \\frac{1}{2}\\left[1 + \\mathrm{erf}\\!
            \\left(\\frac{D_\\mathrm{thr} - d_L(z)}{\\sqrt{2}\\,\\sigma_{d_L}(z)}\\right)\\right]

        where :math:`\\sigma_{d_L} = f_\\sigma \\cdot d_L(z)` and
        :math:`f_\\sigma` is ``FRACTIONAL_LUMINOSITY_ERROR``.

        Args:
            redshift: Source redshift.
            hubble_constant: Dimensionless Hubble parameter :math:`h`.

        Returns:
            Detection probability in :math:`[0, 1]`.
        """
        return float(
            (
                1
                + erf(
                    (self.luminosity_distance_threshold - dist(redshift, hubble_constant))
                    / np.sqrt(2)
                    / FRACTIONAL_LUMINOSITY_ERROR
                    / dist(redshift, hubble_constant)
                )
            )
            / 2
        )

    def gw_likelihood(
        self,
        measured_luminosity_distance: float,
        redshift: float,
        hubble_constant: float,
    ) -> float:
        """GW likelihood: probability of measuring *measured_luminosity_distance* given *redshift*.

        Assumes Gaussian measurement noise:

        .. math::

            \\mathcal{L}(\\hat{d}_L \\mid z, h) =
            \\mathcal{N}\\!\\left(\\hat{d}_L;\\, d_L(z, h),\\,
            f_\\sigma \\cdot d_L(z, h)\\right)

        Args:
            measured_luminosity_distance: Measured luminosity distance :math:`\\hat{d}_L` in Gpc.
            redshift: True source redshift.
            hubble_constant: Dimensionless Hubble parameter :math:`h`.

        Returns:
            Likelihood value (probability density).
        """
        mu_luminosity_distance = dist(redshift, hubble_constant)
        sigma_luminosity_distance = FRACTIONAL_LUMINOSITY_ERROR * mu_luminosity_distance

        distribution = NormalDist(mu=mu_luminosity_distance, sigma=sigma_luminosity_distance)

        return distribution.pdf(measured_luminosity_distance)

    def likelihood(
        self,
        hubble_constant: float,
        measured_luminosity_distance: float,
        measured_redshifted_mass: float,
        detection_index: int,
    ) -> float:
        """Marginalized likelihood :math:`p(\\hat{d}_L \\mid H_0)` for one EMRI detection.

        Integrates the GW likelihood over the galaxy-catalog redshift distribution,
        weighted by sky-localization overlap and — optionally — BH mass information.
        A selection-effects correction divides by the denominator integral when
        ``use_selection_effects_correction`` is ``True``:

        .. math::

            \\mathcal{L}(H_0) = \\frac{\\int p_\\mathrm{GW}(\\hat{d}_L \\mid z, H_0)\\,
            p_\\mathrm{det}(z, H_0)\\, p(z \\mid \\mathrm{catalog})\\, dz}
            {\\int p_\\mathrm{det}(z, H_0)\\, p(z \\mid \\mathrm{catalog})\\, dz}

        Args:
            hubble_constant: Dimensionless Hubble parameter :math:`h` being evaluated.
            measured_luminosity_distance: Measured :math:`\\hat{d}_L` in Gpc.
            measured_redshifted_mass: Measured redshifted BH mass :math:`M_z` in solar masses.
            detection_index: Index into ``emri_detections`` and pre-computed weight arrays.

        Returns:
            Likelihood value (positive real number; not normalized over :math:`H_0`).
        """
        # Compute all luminosity distances at once — replaces 2000+ scalar dist() calls.
        mu_d = dist_array(self.redshift_values, hubble_constant)

        # GW detection probability: P_det(z) = (1 + erf(x)) / 2
        # where x = (D_threshold - mu_d) / (sqrt(2) * sigma_d)
        p_det_array = (
            1.0
            + erf(
                (self.luminosity_distance_threshold - mu_d)
                / (np.sqrt(2.0) * FRACTIONAL_LUMINOSITY_ERROR * mu_d)
            )
        ) / 2.0

        # GW likelihood: Gaussian PDF with mu=mu_d, sigma=sigma_d
        sigma_d = FRACTIONAL_LUMINOSITY_ERROR * mu_d
        gw_likelihood_array = np.exp(
            -0.5 * ((measured_luminosity_distance - mu_d) / sigma_d) ** 2
        ) / (sigma_d * np.sqrt(2.0 * np.pi))

        # Galaxy sky-localisation weight per redshift bin: matrix-vector product
        # galaxy_distribution_at_redshifts: (n_z, n_galaxies); weights: (n_galaxies,)
        galaxy_skylocalization_weights = (
            self.galaxy_distribution_at_redshifts
            @ self.detection_skylocalization_weight_by_galaxy[detection_index]
        )

        if not self.use_bh_mass:
            nominator = np.trapezoid(
                gw_likelihood_array * p_det_array * galaxy_skylocalization_weights,
                self.redshift_values,
            )
        else:
            nominator = np.trapezoid(
                gw_likelihood_array
                * p_det_array
                * NormalDist(
                    mu=measured_redshifted_mass,
                    sigma=FRACTIONAL_MEASURED_MASS_ERROR * measured_redshifted_mass,
                ).pdf(measured_redshifted_mass)
                * np.array(
                    [
                        np.sum(
                            const_redshift_distribution
                            * const_redshift_mass_distribution
                            * self.detection_skylocalization_weight_by_galaxy[detection_index]
                        )
                        for const_redshift_distribution, const_redshift_mass_distribution in zip(
                            self.galaxy_distribution_at_redshifts,
                            self.galaxy_detection_mass_distribution_at_redshifts[detection_index],
                        )
                    ]
                ),
                self.redshift_values,
            )

        denominator = np.trapezoid(
            p_det_array * galaxy_skylocalization_weights,
            self.redshift_values,
        )
        if not self.use_selection_effects_correction:
            denominator = 1.0
        return float(nominator / denominator)

    def posterior(self, hubble_constant: float) -> list[float]:
        return [
            self.likelihood(
                hubble_constant=hubble_constant,
                measured_luminosity_distance=emri_detection.measured_luminosity_distance,
                measured_redshifted_mass=emri_detection.measured_redshifted_mass,
                detection_index=index,
            )
            for index, emri_detection in enumerate(self.emri_detections)
        ]

    def plot_gw_detection_probability(self) -> None:
        gw_detection_probabilities = [
            self.gw_detection_probability(redshift, TRUE_HUBBLE_CONSTANT)
            for redshift in self.redshift_values
        ]
        _plotter = ScientificPlotter(figure_size=(16, 9))
        _plotter.plot(self.redshift_values, np.array(gw_detection_probabilities))
        _plotter.show_and_close()
