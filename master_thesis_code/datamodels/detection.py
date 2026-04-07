"""Detection datamodel for Cramér-Rao bounds based EMRI inference."""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import truncnorm

from master_thesis_code.physical_relations import dist

_LOGGER = logging.getLogger(__name__)


def _sky_localization_uncertainty(
    phi_error: float, theta: float, theta_error: float, cov_theta_phi: float
) -> float:
    """Sky-localization uncertainty (solid angle) from the Cramér-Rao matrix.

    Computes the area of the error ellipse on the sky:

    .. math::

        \\Delta\\Omega = 2\\pi |\\sin\\theta|
        \\sqrt{\\sigma_\\phi^2 \\sigma_\\theta^2 - C_{\\theta\\phi}^2}

    Args:
        phi_error: 1-σ uncertainty on the azimuthal angle :math:`\\phi` in radians.
        theta: Polar angle :math:`\\theta` in radians.
        theta_error: 1-σ uncertainty on :math:`\\theta` in radians.
        cov_theta_phi: Off-diagonal Cramér-Rao element :math:`C_{\\theta\\phi}` in rad².

    Returns:
        Sky-localization uncertainty in steradians.
    """
    return float(
        2
        * np.pi
        * np.abs(np.sin(theta))
        * np.sqrt(phi_error**2 * theta_error**2 - cov_theta_phi**2)
    )


@dataclass
class Detection:
    """EMRI detection parsed from Cramér-Rao bounds CSV output.

    Stores the maximum-likelihood parameter estimates and their 1-σ errors derived
    from the Fisher information matrix for a single detected EMRI event.

    Attributes:
        d_L: Luminosity distance :math:`d_L` in Gpc.
        d_L_uncertainty: 1-σ error on :math:`d_L` in Gpc, equal to
            :math:`\\sqrt{\\Gamma^{-1}_{d_L d_L}}`.
        phi: Sky azimuthal angle :math:`\\phi_S` in radians.
        phi_error: 1-σ error on :math:`\\phi_S` in radians.
        theta: Sky polar angle :math:`\\theta_S` (= :math:`q_S`) in radians.
        theta_error: 1-σ error on :math:`\\theta_S` in radians.
        M: Redshifted central BH mass :math:`M_z` in solar masses.
        M_uncertainty: 1-σ error on :math:`M_z` in solar masses.
        theta_phi_covariance: Off-diagonal Cramér-Rao element :math:`C_{\\theta\\phi}`
            in rad².
        M_phi_covariance: Off-diagonal element :math:`C_{M\\phi}` in
            :math:`M_\\odot \\cdot \\mathrm{rad}`.
        M_theta_covariance: Off-diagonal element :math:`C_{M\\theta}` in
            :math:`M_\\odot \\cdot \\mathrm{rad}`.
        d_L_M_covariance: Off-diagonal element :math:`C_{d_L M}` in
            :math:`\\mathrm{Gpc} \\cdot M_\\odot`.
        d_L_theta_covariance: Off-diagonal element :math:`C_{d_L\\theta}` in
            :math:`\\mathrm{Gpc} \\cdot \\mathrm{rad}`.
        d_L_phi_covariance: Off-diagonal element :math:`C_{d_L\\phi}` in
            :math:`\\mathrm{Gpc} \\cdot \\mathrm{rad}`.
        host_galaxy_index: Index of the host galaxy in the galaxy catalog.
        snr: Signal-to-noise ratio (dimensionless).
        WL_uncertainty: Weak-lensing contribution to the :math:`d_L` uncertainty in Gpc.
    """

    d_L: float  # Gpc, luminosity distance
    d_L_uncertainty: float  # Gpc, 1-σ error on d_L (= √Γ⁻¹_{d_L d_L})
    phi: float  # rad, sky azimuthal angle (phiS)
    phi_error: float  # rad, 1-σ error on phi
    theta: float  # rad, sky polar angle (qS)
    theta_error: float  # rad, 1-σ error on theta
    M: float  # M_sun, central black hole mass (redshifted)
    M_uncertainty: float  # M_sun, 1-σ error on M
    theta_phi_covariance: float  # rad², off-diagonal Cramér-Rao element
    M_phi_covariance: float  # M_sun·rad, off-diagonal Cramér-Rao element
    M_theta_covariance: float  # M_sun·rad, off-diagonal Cramér-Rao element
    d_L_M_covariance: float  # Gpc·M_sun, off-diagonal Cramér-Rao element
    d_L_theta_covariance: float  # Gpc·rad, off-diagonal Cramér-Rao element
    d_L_phi_covariance: float  # Gpc·rad, off-diagonal Cramér-Rao element
    host_galaxy_index: int  # index in the galaxy catalog
    snr: float  # dimensionless, signal-to-noise ratio
    WL_uncertainty: float = 0.0  # Gpc, weak-lensing contribution to d_L uncertainty

    def __init__(self, parameters: pd.Series) -> None:
        self.d_L = parameters["luminosity_distance"]
        self.d_L_uncertainty = np.sqrt(
            parameters["delta_luminosity_distance_delta_luminosity_distance"]
        )
        self.phi = parameters["phiS"]
        self.phi_error = np.sqrt(parameters["delta_phiS_delta_phiS"])
        self.theta = parameters["qS"]
        self.theta_error = np.sqrt(parameters["delta_qS_delta_qS"])
        self.M = parameters["M"]
        self.M_uncertainty = np.sqrt(parameters["delta_M_delta_M"])
        self.theta_phi_covariance = parameters["delta_phiS_delta_qS"]
        self.M_phi_covariance = parameters["delta_phiS_delta_M"]
        self.M_theta_covariance = parameters["delta_qS_delta_M"]
        self.d_L_M_covariance = parameters["delta_luminosity_distance_delta_M"]
        self.d_L_theta_covariance = parameters["delta_qS_delta_luminosity_distance"]
        self.d_L_phi_covariance = parameters["delta_phiS_delta_luminosity_distance"]
        self.snr = parameters["SNR"]
        self.host_galaxy_index = parameters["host_galaxy_index"]

    def get_skylocalization_error(self) -> float:
        return _sky_localization_uncertainty(
            self.phi_error, self.theta, self.theta_error, self.theta_phi_covariance
        )

    def get_relative_distance_error(self) -> float:
        return self.d_L_uncertainty / self.d_L

    def convert_to_best_guess_parameters(self, rng: np.random.Generator | None = None) -> None:
        """Draw simulated measured parameters from the Fisher-matrix posterior.

        When *rng* is provided, draws a single correlated sample from the
        4-dimensional multivariate normal defined by the full Cramér-Rao
        covariance sub-matrix for (phi, theta, d_L, M) and clips to physical
        bounds.  This is the standard procedure in the GW literature
        (Cutler & Flanagan 1994; Vallisneri 2008, arXiv:gr-qc/0703086).

        When *rng* is ``None`` the legacy behaviour is preserved: four
        independent truncated-normal draws from the marginal distributions.

        Args:
            rng: NumPy random generator for reproducible, correlated draws.
                 Pass ``None`` to keep the legacy independent-sampling path.
        """
        if rng is not None:
            self._correlated_draw(rng)
        else:
            self._independent_draw()

    # -- correlated multivariate draw (preferred) ---------------------------

    def _correlated_draw(self, rng: np.random.Generator) -> None:
        # Build the 4×4 covariance sub-matrix.
        # Order: phi, theta, d_L, M
        # Ref: Cramér-Rao lower bound, Γ⁻¹ sub-matrix for extrinsic params.
        cov = np.array(
            [
                [
                    self.phi_error**2,
                    self.theta_phi_covariance,
                    self.d_L_phi_covariance,
                    self.M_phi_covariance,
                ],
                [
                    self.theta_phi_covariance,
                    self.theta_error**2,
                    self.d_L_theta_covariance,
                    self.M_theta_covariance,
                ],
                [
                    self.d_L_phi_covariance,
                    self.d_L_theta_covariance,
                    self.d_L_uncertainty**2,
                    self.d_L_M_covariance,
                ],
                [
                    self.M_phi_covariance,
                    self.M_theta_covariance,
                    self.d_L_M_covariance,
                    self.M_uncertainty**2,
                ],
            ]
        )

        mean = np.array([self.phi, self.theta, self.d_L, self.M])

        # Positive-definiteness check — inv(Fisher) should always be PD, but
        # numerical noise from the 5-point stencil can break this for marginal
        # detections.  Fall back to independent draws with a warning.
        try:
            np.linalg.cholesky(cov)
        except np.linalg.LinAlgError:
            _LOGGER.warning(
                "Covariance matrix not positive-definite for detection at "
                "d_L=%.3f Gpc, M=%.0f M_sun — falling back to independent draws.",
                self.d_L,
                self.M,
            )
            self._independent_draw()
            return

        sample = rng.multivariate_normal(mean, cov)

        # Clip to physical bounds (never reached in practice for EMRI events).
        self.phi = float(np.clip(sample[0], 0.0, 2.0 * np.pi))
        self.theta = float(np.clip(sample[1], 0.0, np.pi))
        self.d_L = float(np.clip(sample[2], 0.0, dist(1.5)))
        self.M = float(np.clip(sample[3], 1e4, 1e6))

    # -- legacy independent truncated-normal draws --------------------------

    def _independent_draw(self) -> None:
        self.phi = truncnorm(
            (0 - self.phi) / self.phi_error,
            (2 * np.pi - self.phi) / self.phi_error,
            loc=self.phi,
            scale=self.phi_error,
        ).rvs(1)[0]

        self.theta = truncnorm(
            (0 - self.theta) / self.theta_error,
            (np.pi - self.theta) / self.theta_error,
            loc=self.theta,
            scale=self.theta_error,
        ).rvs(1)[0]

        self.d_L = truncnorm(
            (0 - self.d_L) / self.d_L_uncertainty,
            (dist(1.5) - self.d_L) / self.d_L_uncertainty,
            loc=self.d_L,
            scale=self.d_L_uncertainty,
        ).rvs(1)[0]

        self.M = truncnorm(
            (1e4 - self.M) / self.M_uncertainty,
            (1e6 - self.M) / self.M_uncertainty,
            loc=self.M,
            scale=self.M_uncertainty,
        ).rvs(1)[0]
