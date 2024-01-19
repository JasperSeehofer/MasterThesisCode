from fastlisaresponse import ResponseWrapper
from few.waveform import GenerateEMRIWaveform
from enum import Enum
from typing import List
import logging

from master_thesis_code.datamodels.parameter_space import ParameterSpace
from master_thesis_code.exceptions import WaveformGenerationError

_LOGGER = logging.getLogger()
USE_GPU = True
INDEX_LAMBDA = 8  # index in list of parameters from ParameterSpace for phiS
INDEX_BETA = 7  # index in list of parameters from ParameterSpace for qS
T0 = 10_000.0  #

# Configuration of PN5 AAK waveform generator
pn5_aak_configuration = {
    "inspiral_kwargs": {
        "DENSE_STEPPING": 0,  # we want a sparsely sampled trajectory
        "max_init_len": int(
            1e6
        ),  # all of the trajectories will be well under len = 1000
    },
    "sum_kwargs": {
        "use_gpu": True,  # GPU is available for this type of summation
        "pad_output": False,
    },
}

# FAST LISA RESPONSE configuration
# order of the langrangian interpolation
order = 25
orbit_file_esa = "./lisa_files/esa-trailing-orbits.h5"
orbit_kwargs_esa = dict(orbit_file=orbit_file_esa)
# 1st or 2nd or custom (see docs for custom)
tdi_gen = "2nd generation"
tdi_kwargs_esa = dict(
    orbit_kwargs=orbit_kwargs_esa,
    order=order,
    tdi=tdi_gen,
    tdi_chan="XYZ",
)


class WaveGeneratorType(Enum):
    SCHWARZSCHILD_FULLY_RELATIVISTIC = 1
    PN5_AAK = 2


def create_lisa_response_generator(
    waveform_generator_type: WaveGeneratorType,
    dt: float,
    T_observation: float,
    
) -> ResponseWrapper:
    lisa_response_generator = ResponseWrapper(
        waveform_gen=_set_waveform_generator(waveform_generator_type),
        flip_hx=True,
        index_lambda=INDEX_LAMBDA,
        index_beta=INDEX_BETA,
        t0=T0,
        is_ecliptic_latitude=False,
        use_gpu=USE_GPU,
        Tobs=T_observation,
        remove_garbage="zero", # TODO: understand why to use this
        dt=dt,
        **tdi_kwargs_esa,
    )
    _LOGGER.info("Lisa response generator initialized.")
    return lisa_response_generator


def _set_waveform_generator(
    waveform_generator_type: WaveGeneratorType,
) -> GenerateEMRIWaveform:
    if waveform_generator_type == WaveGeneratorType.SCHWARZSCHILD_FULLY_RELATIVISTIC:
        _LOGGER.info(
            "Parameter estimation is setup up with the 'FastSchwarzschildEccentricFlux' wave generator."
        )
        return GenerateEMRIWaveform(
            waveform_class="FastSchwarzschildEccentricFlux", use_gpu=USE_GPU
        )
    elif waveform_generator_type == WaveGeneratorType.PN5_AAK:
        waveform_generator = GenerateEMRIWaveform(
            waveform_class="Pn5AAKWaveform",
            inspiral_kwargs=pn5_aak_configuration["inspiral_kwargs"],
            sum_kwargs=pn5_aak_configuration["sum_kwargs"],
            frame="detector",
            use_gpu=USE_GPU,
        )
        _LOGGER.info(
            "Parameter estimation is setup up with the 'PN5AAKwaveform' wave generator."
        )
        return waveform_generator
    else:
        raise WaveformGenerationError(
            "Wave generator class could not be matched to FastSchwarzschildEccentricFlux or PN5AAKwaveform."
            "please check configuration."
        )
