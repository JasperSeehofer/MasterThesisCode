"""EMRI event-rate cosmological model and H₀ evaluation orchestration.

This module defines the cosmological event-rate model used for simulation:
:class:`Model1CrossCheck` samples EMRI events from a cosmological rate model,
while :class:`LamCDMScenario` and :class:`DarkEnergyScenario` define the
parameter spaces.

The Hubble-constant posterior evaluation has been extracted to
:mod:`master_thesis_code.bayesian_inference.bayesian_statistics`
(:class:`~master_thesis_code.bayesian_inference.bayesian_statistics.BayesianStatistics`)
and is invoked via ``main.py:evaluate()`` / ``--evaluate``.
"""

import logging
from dataclasses import dataclass

import emcee
import numpy as np
import numpy.typing as npt
from scipy.stats import truncnorm

from master_thesis_code.constants import SNR_THRESHOLD
from master_thesis_code.datamodels.detection import (
    Detection as Detection,
)
from master_thesis_code.datamodels.parameter_space import (
    Parameter,
    ParameterSpace,
    uniform,
)
from master_thesis_code.galaxy_catalogue.handler import (
    ParameterSample,
)
from master_thesis_code.M1_model_extracted_data.detection_fraction import (
    DetectionFraction,
)
from master_thesis_code.physical_relations import (
    dist,
)

_LOGGER = logging.getLogger()

# detection fraction LISA M1 model


@dataclass
class CosmologicalParameter(Parameter):
    fiducial_value: float = 1.0


# setup distribution of MBH spin
min_a, max_a = 0, 0.998
median_a = 0.98
mean_a = median_a
std_a = 0.05
a_distribution = truncnorm(
    (min_a - mean_a) / std_a, (max_a - mean_a) / std_a, loc=mean_a, scale=std_a
)

# coefficients of polynomial fit of dN/dz for different mass bins
merger_distribution_coefficients = {
    0: [
        -94138538.96193656,
        962369408.6975077,
        -3578439441.007358,
        5185569151.868952,
        136402179.6970964,
        -5943613356.609655,
        -3095452047.664805,
        14366833862.29217,
        281549370.2295778,
    ],
    1: [
        -121373875.11104208,
        1445799310.7124693,
        -6789811160.974133,
        15499973013.857445,
        -16287443134.169672,
        4260623123.77606,
        448851767.47119844,
        7196325833.826655,
        392346838.9761119,
    ],
    2: [
        247775058.37853566,
        -3216041245.326129,
        17221325721.312645,
        -49199088856.52833,
        81032523270.16118,
        -76909638891.85504,
        36993487023.340546,
        -3395035047.189672,
        935723800.2450081,
    ],
    3: [
        242799.05947105083,
        89582514.52046189,
        -1091533038.55458,
        5564104340.280578,
        -15711586884.180557,
        26545162214.60701,
        -25878717919.48227,
        11984641274.884602,
        19999528.190069355,
    ],
    4: [
        31727829.680760894,
        -386428090.92766726,
        1923642431.9585557,
        -5016108535.359479,
        7268737080.670669,
        -5669600556.885939,
        1925511804.844566,
        311360824.27473867,
        58391055.04627399,
    ],
}


def polynomial(
    x: float | npt.NDArray[np.float64],
    a: float,
    b: float,
    c: float,
    d: float,
    e: float,
    f: float,
    g: float,
    h: float,
    i: float,
) -> float:
    if isinstance(x, int | float):
        if x > 3:
            x = 3.0
    else:
        x = np.array([value if value <= 3 else 3.0 for value in x])  # end of fit range

    result = (
        a * x**9
        + b * x**8
        + c * x**7
        + d * x**6
        + e * x**5
        + f * x**4
        + g * x**3
        + h * x**2
        + i * x
    )
    return float(result) if isinstance(result, int | float) else result  # type: ignore[return-value]


def MBH_spin_distribution(lower_limit: float, upper_limit: float) -> float:
    """https://iopscience.iop.org/article/10.1088/0004-637X/762/2/68/pdf"""
    return float(a_distribution.rvs(1)[0])


class Model1CrossCheck:
    """cross check of Model M1 in PHYSICAL REVIEW D 95, 103012 (2017)"""

    parameter_space: ParameterSpace
    emri_rate: int = 294  # 1/yr
    snr_threshold: float = SNR_THRESHOLD
    detection_fraction = DetectionFraction()

    def __init__(self, rng: np.random.Generator | None = None) -> None:
        if rng is None:
            rng = np.random.default_rng()
        self._rng = rng
        self.parameter_space = ParameterSpace()
        self._apply_model_assumptions()
        self.setup_emri_events_sampler()

    def _apply_model_assumptions(self) -> None:
        self.parameter_space.M.lower_limit = 10 ** (4.5)
        self.parameter_space.M.upper_limit = 10 ** (6.0)

        self.parameter_space.a.value = 0.98
        self.parameter_space.a.is_fixed = True

        self.parameter_space.mu.value = 10
        self.parameter_space.mu.is_fixed = True

        self.parameter_space.e0.upper_limit = 0.2

        self.max_redshift = 1.5
        self.parameter_space.luminosity_distance.upper_limit = dist(redshift=self.max_redshift)
        self.luminostity_detection_threshold = 1.55  # as in Hitchikers Guide

    def emri_distribution(self, M: float, redshift: float) -> float:
        return self.dN_dz_of_mass(M, redshift) * self.R_emri(M)

    @staticmethod
    def dN_dz_of_mass(mass: float, redshift: float) -> float:
        mass_bin = float(np.log10(mass))
        if mass_bin < 4.5:
            return float(polynomial(redshift, *merger_distribution_coefficients[0]))
        elif mass_bin < 5.0:
            fraction = (mass_bin - 4.5) / 0.5
            return float(
                (1 - fraction) * polynomial(redshift, *merger_distribution_coefficients[0])
                + fraction * polynomial(redshift, *merger_distribution_coefficients[1])
            )
        elif mass_bin < 5.5:
            fraction = (mass_bin - 5.0) / 0.5
            return float(
                (1 - fraction) * polynomial(redshift, *merger_distribution_coefficients[1])
                + fraction * polynomial(redshift, *merger_distribution_coefficients[2])
            )
        elif mass_bin < 6.0:
            fraction = (mass_bin - 5.5) / 0.5
            return float(
                (1 - fraction) * polynomial(redshift, *merger_distribution_coefficients[2])
                + fraction * polynomial(redshift, *merger_distribution_coefficients[3])
            )
        else:  # mass_bin >= 6.25
            fraction = (mass_bin - 6.0) / 0.5
            fraction = min(fraction, 1.0)
            return float(
                (1 - fraction) * polynomial(redshift, *merger_distribution_coefficients[3])
                + fraction * polynomial(redshift, *merger_distribution_coefficients[4])
            )

    @staticmethod
    def R_emri(M: float) -> float:
        if M < 1.2e5:
            return float(10 ** ((1.02445) * np.log10(M / 1.2e5) + np.log10(33.1)))
        elif M < 2.5e5:
            return float(10 ** ((0.4689) * np.log10(M / 2.5e5) + np.log10(46.7)))
        else:
            return float(10 ** ((-0.2475) * np.log10(M / 2.9e7) + np.log10(14.4)))

    def _log_probability(self, M: float, redshift: float) -> float:
        if not self.parameter_space.M.lower_limit < M < self.parameter_space.M.upper_limit:
            return -np.inf
        if not 0 < redshift < self.max_redshift:
            return -np.inf
        return float(np.log(self.emri_distribution(M, redshift)))

    def setup_emri_events_sampler(self) -> None:
        # use emcee to sample the distribution

        def log_probability(x: list[float]) -> float:
            return self._log_probability(10 ** x[0], x[1])

        ndim = 2
        nwalkers = 20
        burn_in_steps = 1000
        p0_mass = self._rng.random((nwalkers, 1)) * (
            np.log10(self.parameter_space.M.upper_limit)
            - np.log10(self.parameter_space.M.lower_limit)
        ) + np.log10(self.parameter_space.M.lower_limit)
        p0_redshift = self._rng.random((nwalkers, 1)) * self.max_redshift
        p0 = np.column_stack((p0_mass, p0_redshift))
        _LOGGER.info(
            f"Setup emcee MCMC with {nwalkers} walkers and {burn_in_steps} burn in steps..."
        )
        self._emri_event_sampler = emcee.EnsembleSampler(
            nwalkers,
            ndim,
            log_probability,
        )

        # let the walkers burn in
        pos, prob, state = self._emri_event_sampler.run_mcmc(p0, burn_in_steps)
        self._sample_positions = pos
        self._emri_event_sampler.reset()

        _LOGGER.info("Burn in complete...")

    def sample_emri_events(self, number_of_samples: int) -> list[ParameterSample]:
        _LOGGER.info("Sampling EMRI events...")
        pos, prob, state = self._emri_event_sampler.run_mcmc(
            initial_state=self._sample_positions, nsteps=number_of_samples
        )
        samples = self._emri_event_sampler.get_chain(flat=True)
        self._sample_positions = pos
        self._emri_event_sampler.reset()
        return_samples = [
            ParameterSample(M=10 ** sample[0], redshift=sample[1], a=MBH_spin_distribution(0, 1))
            for sample in samples
        ]
        _LOGGER.info(f"Sampling complete (number of samples ({len(return_samples)})).")

        return return_samples

    def simplified_event_mass_dependency(self, mass: float) -> float:
        raise NotImplementedError

    def setup_simplified_event_sampler(self) -> None:
        pass


class LamCDMScenario:
    """https://arxiv.org/pdf/2102.01708.pdf"""

    h: CosmologicalParameter
    Omega_m: CosmologicalParameter
    w_0: float = -1.0
    w_a: float = 0.0

    def __init__(self) -> None:
        self.h = CosmologicalParameter(
            symbol="h",
            upper_limit=0.86,
            lower_limit=0.6,
            unit="s*Mpc/km",
            randomize_by_distribution=uniform,
            fiducial_value=0.73,
        )
        self.Omega_m = CosmologicalParameter(
            symbol="Omega_m",
            upper_limit=0.5,
            lower_limit=0.04,
            unit="s*Mpc/km",
            randomize_by_distribution=uniform,
            fiducial_value=0.25,
        )


class DarkEnergyScenario:
    w_0: CosmologicalParameter
    w_a: CosmologicalParameter
    h: float = 0.73
    Omega_m: float = 0.25
    Omega_DE: float = 0.75

    def __init__(self) -> None:
        self.w_0 = CosmologicalParameter(
            symbol="w_0",
            unit="xxx",
            lower_limit=-3.0,
            upper_limit=-0.3,
            randomize_by_distribution=uniform,
            fiducial_value=-1.0,
        )
        self.w_a = CosmologicalParameter(
            symbol="w_a",
            unit="xxx",
            lower_limit=-3.0,
            upper_limit=3.0,
            randomize_by_distribution=uniform,
            fiducial_value=0.0,
        )

    def de_equation(self, z: float) -> float:
        return float(self.w_0 + z / (1 + z) / self.w_a)  # type: ignore[operator]


def gaussian(
    x: float | npt.NDArray[np.float64], mu: float, sigma: float, a: float
) -> float | npt.NDArray[np.float64]:
    return a * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


# ── Backward-compatibility re-exports (REMOVED) ──────────────────────────────
# These symbols were extracted to bayesian_inference/ subpackage modules.
# The re-exports caused a circular import that crashed multiprocessing workers
# in the evaluation pipeline. Import directly from:
#   master_thesis_code.bayesian_inference.bayesian_statistics
#   master_thesis_code.bayesian_inference.detection_probability
