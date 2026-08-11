"""CRB data layer: constants, covariance reconstruction, and label mapping.

Provides the bridge between raw CRB CSV rows (produced by
``ParameterEstimation.save_cramer_rao_bound``) and the plotting subsystem.

The 14 EMRI parameter names and their order match
``ParameterSpace._parameters_to_dict()`` exactly.  CSV columns follow
the ``delta_{row}_delta_{col}`` naming convention for the lower triangle
of the covariance matrix.
"""

import numpy as np
import numpy.typing as npt
import pandas as pd

# ---------------------------------------------------------------------------
# Parameter name constants (order matches ParameterSpace._parameters_to_dict)
# ---------------------------------------------------------------------------

PARAMETER_NAMES: list[str] = [
    "M",
    "mu",
    "a",
    "p0",
    "e0",
    "x0",
    "luminosity_distance",
    "qS",
    "phiS",
    "qK",
    "phiK",
    "Phi_phi0",
    "Phi_theta0",
    "Phi_r0",
]

INTRINSIC: list[str] = ["M", "mu", "a", "p0", "e0", "x0"]

EXTRINSIC: list[str] = [
    "luminosity_distance",
    "qS",
    "phiS",
    "qK",
    "phiK",
    "Phi_phi0",
    "Phi_theta0",
    "Phi_r0",
]

# ---------------------------------------------------------------------------
# Label key mapping (CSV param name -> _labels.py LABELS key)
# ---------------------------------------------------------------------------

PARAM_TO_LABEL_KEY: dict[str, str] = {
    "luminosity_distance": "d_L",
    "x0": "Y0",
}


def label_key(param: str) -> str:
    """Map a CSV parameter name to its ``_labels.py`` LABELS key.

    Parameters
    ----------
    param : str
        Parameter name as it appears in PARAMETER_NAMES / CSV columns.

    Returns
    -------
    str
        The corresponding key in ``LABELS``.  Parameters without a
        special mapping are returned unchanged.
    """
    return PARAM_TO_LABEL_KEY.get(param, param)


def reconstruct_covariance(row: pd.Series) -> npt.NDArray[np.float64]:
    """Reconstruct a 14x14 covariance matrix from a CRB CSV row.

    The CSV stores the lower triangle of the covariance matrix in columns
    named ``delta_{PARAMETER_NAMES[i]}_delta_{PARAMETER_NAMES[j]}`` where
    ``i >= j``.  This function reads those columns and fills both the
    lower and upper triangles to produce a symmetric matrix.

    Parameters
    ----------
    row : pd.Series
        A single row from the CRB CSV, containing at least the 105
        ``delta_*_delta_*`` columns.

    Returns
    -------
    npt.NDArray[np.float64]
        Symmetric 14x14 covariance matrix.
    """
    n = len(PARAMETER_NAMES)
    cov = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1):
            col_name = f"delta_{PARAMETER_NAMES[i]}_delta_{PARAMETER_NAMES[j]}"
            value = float(row[col_name])
            cov[i, j] = value
            cov[j, i] = value
    return cov
