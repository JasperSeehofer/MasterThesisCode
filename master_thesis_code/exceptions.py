class ArgumentsError(Exception):
    """Error during parsing of the arguments."""


class ParameterEstimationError(Exception):
    """Error during parameter estimation."""


class TimeoutError(Exception):
    """Error when time a given time limit is reached."""


class ParameterOutOfBoundsError(Exception):
    """Error when trying to set a parameter to a value that is out of its given limits."""


class WaveformGenerationError(Exception):
    """Error during setup or when running the waveform generation."""
