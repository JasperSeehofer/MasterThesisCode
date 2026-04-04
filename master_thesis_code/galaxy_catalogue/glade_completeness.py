"""GLADE+ catalog completeness as a function of luminosity distance and redshift.

Provides the completeness fraction f(z, h) for the GLADE+ galaxy catalog, digitized
from Dalya et al. (2022), arXiv:2110.06184, Fig. 2 (B-band luminosity comparison).

The completeness data is stored as luminosity distance (Mpc) vs completeness (%).
The interface converts between distance and redshift using the cosmological
distance-redshift relation from physical_relations.dist().

# ASSERT_CONVENTION: distance_unit=Mpc, completeness_output=fraction_0_to_1,
#   dist_returns=Gpc, GPC_TO_MPC=1e3
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from master_thesis_code.constants import (
    GPC_TO_MPC,
    OMEGA_DE,
    OMEGA_M,
    H,
)
from master_thesis_code.physical_relations import dist_vectorized


@dataclass
class GladeCatalogCompleteness:
    """GLADE+ catalog completeness digitized from Dalya et al. (2022) Fig. 2.

    The raw data gives completeness in percent as a function of luminosity
    distance in Mpc.  Public methods expose this as either:

    * ``get_completeness(distance)`` -- legacy interface, returns **percent**
    * ``get_completeness_fraction(distance_mpc)`` -- returns fraction in [0, 1]
    * ``get_completeness_at_redshift(z, h)`` -- converts z to d_L, returns fraction

    Attributes
    ----------
    distance : list[float]
        Luminosity distance nodes in Mpc (digitized from Dalya et al. 2022 Fig. 2).
    completeness : list[float]
        Completeness values in percent at each distance node.

    References
    ----------
    Dalya et al. (2022), arXiv:2110.06184, Section 3 and Fig. 2.
    """

    distance: list[float] = None  # type: ignore[assignment]
    completeness: list[float] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.distance is None:
            self.distance = _DISTANCE_MPC.copy()
        if self.completeness is None:
            self.completeness = _COMPLETENESS_PCT.copy()

        # Precompute edge values for interpolation
        self._last_completeness_frac: float = self.completeness[-1] / 100.0

    # ------------------------------------------------------------------
    # Legacy interface (returns PERCENT) -- backward compatibility
    # ------------------------------------------------------------------

    def get_completeness(self, distance: float) -> float:
        """Return completeness in **percent** at the given luminosity distance.

        .. deprecated::
            Use :meth:`get_completeness_fraction` (returns fraction in [0, 1])
            or :meth:`get_completeness_at_redshift` (accepts redshift) instead.

        Args:
            distance: Luminosity distance in Mpc.

        Returns:
            Completeness in percent (0-100).
        """
        return float(
            np.interp(distance, self.distance, self.completeness, right=0.0)
        )

    # ------------------------------------------------------------------
    # New interface: fraction in [0, 1]
    # ------------------------------------------------------------------

    def get_completeness_fraction(self, distance_mpc: float) -> float:
        """Return completeness as a fraction in [0, 1] at a given distance.

        For ``distance_mpc <= 0`` returns 1.0 (complete).
        For ``distance_mpc`` beyond the last data point (795.7 Mpc), extrapolates
        flat at the last digitized value (~21.3%).

        Args:
            distance_mpc: Luminosity distance in Mpc.

        Returns:
            Completeness fraction in [0, 1].

        References
        ----------
        Dalya et al. (2022), arXiv:2110.06184, Fig. 2.
        """
        if distance_mpc < 0.0:
            return 1.0
        result = np.interp(
            distance_mpc,
            self.distance,
            self.completeness,
            left=100.0,
            right=self.completeness[-1],
        )
        return float(result) / 100.0

    def get_completeness_fraction_vectorized(
        self, distance_mpc: npt.NDArray[np.floating[Any]]
    ) -> npt.NDArray[np.floating[Any]]:
        """Vectorized completeness fraction for an array of distances.

        Args:
            distance_mpc: Array of luminosity distances in Mpc.

        Returns:
            Array of completeness fractions in [0, 1], same shape as input.

        References
        ----------
        Dalya et al. (2022), arXiv:2110.06184, Fig. 2.
        """
        result: npt.NDArray[np.floating[Any]] = np.interp(
            distance_mpc,
            self.distance,
            self.completeness,
            left=100.0,
            right=self.completeness[-1],
        )
        return result / 100.0

    def get_completeness_at_redshift(
        self,
        z: float | npt.NDArray[np.floating[Any]],
        h: float = H,
        Omega_m: float = OMEGA_M,
        Omega_de: float = OMEGA_DE,
    ) -> float | npt.NDArray[np.floating[Any]]:
        """Return completeness fraction at redshift(s) z for a given cosmology.

        Converts redshift to luminosity distance via ``dist(z, h)`` and looks up
        the completeness from the digitized Dalya et al. (2022) data.

        Args:
            z: Redshift (scalar or array).
            h: Dimensionless Hubble parameter h = H0 / (100 km/s/Mpc).
            Omega_m: Matter density parameter.
            Omega_de: Dark energy density parameter.

        Returns:
            Completeness fraction in [0, 1]. Scalar if *z* is scalar, array
            if *z* is array.

        Notes
        -----
        The digitized data was produced at fiducial cosmology h=0.73.
        When ``h != 0.73``, the same redshift maps to a different luminosity
        distance, giving a different completeness value. This is the correct
        behavior for the dark siren likelihood (Gray et al. 2020,
        arXiv:1908.06050, Sec. II.3.1).

        References
        ----------
        Dalya et al. (2022), arXiv:2110.06184, Fig. 2.
        Gray et al. (2020), arXiv:1908.06050, Sec. II.3.1 and Appendix A.2.
        """
        z_arr = np.atleast_1d(np.asarray(z, dtype=np.float64))

        # dist_vectorized returns Gpc; convert to Mpc
        d_L_mpc = dist_vectorized(z_arr, h=h, Omega_m=Omega_m, Omega_de=Omega_de) * GPC_TO_MPC

        # Interpolate completeness
        frac = self.get_completeness_fraction_vectorized(d_L_mpc)

        # Clamp to [0, 1] for safety
        frac_clipped: npt.NDArray[np.floating[Any]] = np.clip(frac, 0.0, 1.0)

        # Return scalar if input was scalar
        if np.ndim(z) == 0:
            return float(frac_clipped.flat[0])
        return frac_clipped


# ======================================================================
# Digitized completeness data from Dalya et al. (2022) Fig. 2
# ======================================================================

_DISTANCE_MPC: list[float] = [
    0.0,
    18.43971631205674,
    21.27659574468085,
    23.404255319148938,
    28.368794326241137,
    32.6241134751773,
    37.5886524822695,
    42.5531914893617,
    48.22695035460993,
    53.19148936170213,
    62.4113475177305,
    70.92198581560284,
    78.72340425531915,
    85.1063829787234,
    90.78014184397163,
    97.87234042553192,
    109.21985815602837,
    121.98581560283688,
    134.7517730496454,
    150.354609929078,
    163.82978723404256,
    178.01418439716312,
    192.90780141843973,
    202.12765957446808,
    208.51063829787236,
    214.8936170212766,
    226.9503546099291,
    242.55319148936172,
    258.86524822695037,
    278.72340425531917,
    293.6170212765958,
    309.21985815602835,
    326.2411347517731,
    341.13475177304963,
    357.44680851063833,
    377.30496453900713,
    400,
    419.8581560283688,
    438.29787234042556,
    455.3191489361702,
    475.177304964539,
    495.74468085106383,
    513.4751773049645,
    532.6241134751773,
    551.0638297872341,
    568.7943262411347,
    588.6524822695036,
    609.9290780141844,
    627.6595744680851,
    646.0992907801418,
    664.5390070921986,
    682.9787234042553,
    700,
    719.1489361702128,
    737.5886524822695,
    755.3191489361702,
    773.049645390071,
    785.8156028368794,
    795.7446808510639,
]

_COMPLETENESS_PCT: list[float] = [
    100,
    100,
    96.58018867924528,
    91.15566037735849,
    83.9622641509434,
    77.35849056603773,
    72.75943396226415,
    68.86792452830188,
    66.50943396226415,
    65.09433962264151,
    64.85849056603773,
    63.443396226415096,
    60.7311320754717,
    57.90094339622642,
    55.660377358490564,
    53.655660377358494,
    51.7688679245283,
    50.117924528301884,
    48.70283018867924,
    47.051886792452834,
    45.99056603773585,
    45.5188679245283,
    44.693396226415096,
    44.10377358490566,
    43.04245283018868,
    43.514150943396224,
    43.632075471698116,
    43.39622641509434,
    43.160377358490564,
    42.806603773584904,
    42.68867924528302,
    42.33490566037736,
    41.863207547169814,
    41.509433962264154,
    41.0377358490566,
    40.33018867924528,
    39.2688679245283,
    38.325471698113205,
    37.264150943396224,
    36.320754716981135,
    35.37735849056604,
    34.43396226415094,
    33.60849056603774,
    32.429245283018865,
    31.60377358490566,
    30.660377358490567,
    29.71698113207547,
    28.77358490566038,
    27.94811320754717,
    27.004716981132077,
    26.17924528301887,
    25.589622641509433,
    24.764150943396228,
    24.056603773584907,
    23.349056603773587,
    22.87735849056604,
    22.169811320754718,
    21.580188679245282,
    21.34433962264151,
]
