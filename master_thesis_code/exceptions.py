class ArgumentsError(Exception):
    """Error during parsing of the arguments."""

class ParameterEstimationError(Exception):
    """Error during parameter estimation."""

class TimeoutError(Exception):
    """Error when time a given time limit is reached."""
