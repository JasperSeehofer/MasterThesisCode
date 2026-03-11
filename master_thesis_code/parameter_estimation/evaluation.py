import numpy as np
import pandas as pd


class DataEvaluation:
    def __init__(
        self,
        path_to_cramer_rao_bounds_file: str = "./simulations/cramer_rao_bounds.csv",
        path_to_snr_analysis_file: str = "./simulations/snr_analysis.csv",
        path_to_prepared_cramer_rao_bounds: str = "./simulations/prepared_cramer_rao_bounds.csv",
        path_to_undetected_events_file: str = "./simulations/undetected_events.csv",
    ):
        self._cramer_rao_bounds = pd.read_csv(path_to_cramer_rao_bounds_file)
        self._snr_analysis_file = pd.read_csv(path_to_snr_analysis_file)
        self._prepared_cramer_rao_bounds = pd.read_csv(path_to_prepared_cramer_rao_bounds)
        self._undetected_events = pd.read_csv(path_to_undetected_events_file)


def _compute_skylocalization_uncertainty(
    theta: float,
    var_theta: float,
    var_phi: float,
    cov_theta_phi: float,
) -> float:
    return (  # type: ignore[no-any-return]
        2 * np.pi * np.abs(np.sin(theta)) * np.sqrt(var_phi * var_theta - (cov_theta_phi) ** 2)
    )


@np.vectorize
def _remove_zeros_from_grid(x: float) -> float:
    if x == 0:
        return 1.0
    return x
