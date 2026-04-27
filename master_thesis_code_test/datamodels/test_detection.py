"""Tests for datamodels/detection.py."""

import numpy as np
import pandas as pd

from master_thesis_code.datamodels.detection import Detection, _sky_localization_uncertainty


def _make_series(
    dist: float = 1.0,
    delta_dist: float = 0.05,
    phiS: float = 1.0,
    delta_phiS: float = 0.01,
    qS: float = 0.5,
    delta_qS: float = 0.01,
    M: float = 1e5,
    delta_M: float = 1e3,
    cov_phiS_qS: float = 0.0,
    cov_phiS_M: float = 0.0,
    cov_qS_M: float = 0.0,
    cov_dist_M: float = 0.0,
    cov_qS_dist: float = 0.0,
    cov_phiS_dist: float = 0.0,
    snr: float = 25.0,
    host_galaxy_index: int = 0,
) -> pd.Series:
    return pd.Series(
        {
            "luminosity_distance": dist,
            "delta_luminosity_distance_delta_luminosity_distance": delta_dist**2,
            "phiS": phiS,
            "delta_phiS_delta_phiS": delta_phiS**2,
            "qS": qS,
            "delta_qS_delta_qS": delta_qS**2,
            "M": M,
            "delta_M_delta_M": delta_M**2,
            "delta_phiS_delta_qS": cov_phiS_qS,
            "delta_phiS_delta_M": cov_phiS_M,
            "delta_qS_delta_M": cov_qS_M,
            "delta_luminosity_distance_delta_M": cov_dist_M,
            "delta_qS_delta_luminosity_distance": cov_qS_dist,
            "delta_phiS_delta_luminosity_distance": cov_phiS_dist,
            "SNR": snr,
            "host_galaxy_index": host_galaxy_index,
            "_coord_frame": "ecliptic_BarycentricTrue_J2000",
            "_cov_frame": "ecliptic_BarycentricTrue_J2000",
        }
    )


def test_detection_parses_series() -> None:
    series = _make_series()
    det = Detection(series)
    assert det.d_L == 1.0
    assert det.phi == 1.0
    assert det.theta == 0.5
    assert det.M == 1e5
    assert det.snr == 25.0


def test_detection_uncertainty_from_variance() -> None:
    series = _make_series(delta_dist=0.05)
    det = Detection(series)
    assert abs(det.d_L_uncertainty - 0.05) < 1e-10


def test_relative_distance_error() -> None:
    series = _make_series(dist=2.0, delta_dist=0.1)
    det = Detection(series)
    assert abs(det.get_relative_distance_error() - 0.05) < 1e-10


def test_skylocalization_error_positive() -> None:
    series = _make_series(delta_phiS=0.01, qS=np.pi / 4, delta_qS=0.01, cov_phiS_qS=0.0)
    det = Detection(series)
    assert det.get_skylocalization_error() > 0


def test_sky_localization_uncertainty_function() -> None:
    result = _sky_localization_uncertainty(
        phi_error=0.01, theta=np.pi / 4, theta_error=0.01, cov_theta_phi=0.0
    )
    expected = 2 * np.pi * np.abs(np.sin(np.pi / 4)) * np.sqrt(0.01**2 * 0.01**2)
    assert abs(result - expected) < 1e-12


def test_detection_has_wl_uncertainty_default() -> None:
    series = _make_series()
    det = Detection(series)
    assert det.WL_uncertainty == 0.0
