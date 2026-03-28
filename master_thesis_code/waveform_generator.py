import logging
from enum import Enum
from typing import Any

from master_thesis_code.constants import ESA_TDI_CHANNELS
from master_thesis_code.exceptions import WaveformGenerationError

_LOGGER = logging.getLogger()
INDEX_LAMBDA = 8  # index in list of parameters from ParameterSpace for phiS
INDEX_BETA = 7  # index in list of parameters from ParameterSpace for qS
T0 = 10_000.0  #

# Configuration of PN5 AAK waveform generator

_pn5_aak_inspiral_kwargs = {
    "DENSE_STEPPING": 0,  # we want a sparsely sampled trajectory
    "max_init_len": int(1e6),  # all of the trajectories will be well under len = 1000
}


# FAST LISA RESPONSE configuration
# order of the langrangian interpolation
order = 35
# 1st or 2nd or custom (see docs for custom)
tdi_gen = "1st generation"
tdi_kwargs_esa = dict(
    order=order,
    tdi=tdi_gen,
    tdi_chan=ESA_TDI_CHANNELS,
)


class WaveGeneratorType(Enum):
    SCHWARZSCHILD_FULLY_RELATIVISTIC = 1
    PN5_AAK = 2


def create_lisa_response_generator(
    waveform_generator_type: WaveGeneratorType,
    dt: float,
    T_observation: float,
    *,
    use_gpu: bool = True,
) -> Any:
    # fastlisaresponse is imported lazily: its compiled C extension crashes (SIGILL)
    # on CPUs without AVX support (e.g. GitHub Actions runners). Importing it here
    # rather than at module level keeps waveform_generator importable on any machine;
    # this function is only called in GPU-enabled environments.
    from fastlisaresponse import ResponseWrapper  # noqa: PLC0415
    from lisatools.detector import ESAOrbits  # type: ignore[import-untyped]  # noqa: PLC0415

    # fastlisaresponse defaults to CPU backend even when GPU is available.
    # Force CUDA 12 backend when use_gpu is requested.
    force_backend = "fastlisaresponse_cuda12x" if use_gpu else None

    lisa_response_generator = ResponseWrapper(
        waveform_gen=_set_waveform_generator(waveform_generator_type, use_gpu=use_gpu),
        flip_hx=True,
        index_lambda=INDEX_LAMBDA,
        index_beta=INDEX_BETA,
        t0=T0,
        is_ecliptic_latitude=False,
        Tobs=T_observation,
        remove_garbage=True,  # TODO: understand why to use this
        dt=dt,
        orbits=ESAOrbits(),
        force_backend=force_backend,
        **tdi_kwargs_esa,
    )
    _LOGGER.info("Lisa response generator initialized.")
    return lisa_response_generator


def _set_waveform_generator(
    waveform_generator_type: WaveGeneratorType,
    *,
    use_gpu: bool = True,
) -> Any:
    from few.waveform import GenerateEMRIWaveform  # noqa: PLC0415

    if waveform_generator_type == WaveGeneratorType.SCHWARZSCHILD_FULLY_RELATIVISTIC:
        _LOGGER.info(
            "Parameter estimation is setup up with the 'FastSchwarzschildEccentricFlux' wave generator."
        )
        return GenerateEMRIWaveform(
            waveform_class="FastSchwarzschildEccentricFlux",
        )
    elif waveform_generator_type == WaveGeneratorType.PN5_AAK:
        sum_kwargs = {
            "pad_output": True,
        }
        waveform_generator = GenerateEMRIWaveform(
            waveform_class="Pn5AAKWaveform",
            inspiral_kwargs=_pn5_aak_inspiral_kwargs,
            sum_kwargs=sum_kwargs,
            frame="detector",
        )
        _LOGGER.info("Parameter estimation is setup up with the 'PN5AAKwaveform' wave generator.")
        return waveform_generator
    else:
        raise WaveformGenerationError(
            "Wave generator class could not be matched to FastSchwarzschildEccentricFlux or PN5AAKwaveform."
            "please check configuration."
        )
