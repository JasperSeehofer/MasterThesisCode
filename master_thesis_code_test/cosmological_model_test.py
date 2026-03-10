import numpy as np
import pandas as pd

from master_thesis_code.cosmological_model import (
    Detection,
    MBH_spin_distribution,
    gaussian,
    polynomial,
)


def _make_detection_series(
    dist: float = 1.0,
    delta_dist: float = 0.05,
    phiS: float = 1.0,
    delta_phiS: float = 0.01,
    qS: float = 0.5,
    delta_qS: float = 0.01,
    M: float = 1e5,
    delta_M: float = 1e3,
    snr: float = 25.0,
    host_galaxy_index: int = 0,
) -> pd.Series:
    """Build a pd.Series with the column names expected by Detection.__init__."""
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
            "delta_phiS_delta_qS": 0.0,
            "delta_phiS_delta_M": 0.0,
            "delta_qS_delta_M": 0.0,
            "delta_luminosity_distance_delta_M": 0.0,
            "delta_qS_delta_luminosity_distance": 0.0,
            "delta_phiS_delta_luminosity_distance": 0.0,
            "SNR": snr,
            "host_galaxy_index": host_galaxy_index,
        }
    )


def test_gaussian_peak() -> None:
    """gaussian at x=mu equals a / (sigma * sqrt(2*pi)) when a=1."""
    mu = 2.0
    sigma = 0.5
    a = 1.0
    expected = a * np.exp(0.0)  # exp(0) = 1, so gaussian(mu,mu,sigma,a) = a
    result = gaussian(mu, mu, sigma, a)
    assert abs(float(result) - a) < 1e-10


def test_gaussian_symmetry() -> None:
    """gaussian is symmetric around mu."""
    mu = 1.0
    sigma = 0.5
    a = 1.0
    x_offset = 0.3
    left = gaussian(mu - x_offset, mu, sigma, a)
    right = gaussian(mu + x_offset, mu, sigma, a)
    assert abs(float(left) - float(right)) < 1e-10


def test_gaussian_positive() -> None:
    """gaussian is non-negative for any x."""
    mu = 0.0
    sigma = 1.0
    a = 1.0
    for x in [-10.0, -1.0, 0.0, 1.0, 10.0]:
        result = float(gaussian(x, mu, sigma, a))
        assert result >= 0


def test_polynomial_constant() -> None:
    """polynomial(x, [c, 0, 0, ...]) == 0 for non-constant terms; constant term is the last."""
    # polynomial(x, a,b,c,d,e,f,g,h,i) = a*x^9 + ... + i*x
    # There is no constant offset — polynomial(0, ...) == 0 always.
    # Test: polynomial(x, 0,0,0,0,0,0,0,0,i) = i*x (linear)
    result = polynomial(0.0, 0, 0, 0, 0, 0, 0, 0, 0, 5)
    assert result == 0.0  # i*0 = 0


def test_polynomial_linear() -> None:
    """polynomial(2, 0,0,0,0,0,0,0,0,3) == 3*2 == 6."""
    result = polynomial(2.0, 0, 0, 0, 0, 0, 0, 0, 0, 3)
    assert abs(result - 6.0) < 1e-10


def test_polynomial_quadratic() -> None:
    """polynomial(2, 0,0,0,0,0,0,0,1,0) == 1*2^2 == 4."""
    result = polynomial(2.0, 0, 0, 0, 0, 0, 0, 0, 1, 0)
    assert abs(result - 4.0) < 1e-10


def test_mbh_spin_distribution_in_range() -> None:
    """MBH_spin_distribution should return a float in [0, 1]."""
    for _ in range(50):
        value = MBH_spin_distribution(0.0, 1.0)
        assert isinstance(value, float)
        assert 0.0 <= value <= 1.0


# ── Detection dataclass ────────────────────────────────────────────────────────


def test_detection_construction() -> None:
    """Detection must initialise from a pd.Series with the expected column names."""
    series = _make_detection_series()
    det = Detection(series)
    assert det is not None
    assert isinstance(det, Detection)


def test_detection_fields_set_correctly() -> None:
    """Detection fields must match the values supplied in the pd.Series."""
    series = _make_detection_series(
        dist=2.0,
        delta_dist=0.1,
        phiS=1.5,
        delta_phiS=0.02,
        qS=0.8,
        delta_qS=0.02,
        M=5e5,
        delta_M=5e3,
        snr=30.0,
        host_galaxy_index=7,
    )
    det = Detection(series)

    assert det.d_L == 2.0
    assert abs(det.d_L_uncertainty - 0.1) < 1e-10
    assert det.phi == 1.5
    assert det.theta == 0.8
    assert det.M == 5e5
    assert det.snr == 30.0
    assert det.host_galaxy_index == 7


def test_detection_get_relative_distance_error() -> None:
    """get_relative_distance_error must return d_L_uncertainty / d_L."""
    series = _make_detection_series(dist=2.0, delta_dist=0.2)
    det = Detection(series)
    assert abs(det.get_relative_distance_error() - 0.1) < 1e-10


def test_detection_get_skylocalization_error_non_negative() -> None:
    """get_skylocalization_error must return a non-negative float."""
    series = _make_detection_series(qS=np.pi / 4, delta_qS=0.01, delta_phiS=0.01)
    det = Detection(series)
    sky_err = det.get_skylocalization_error()
    assert isinstance(sky_err, float)
    assert sky_err >= 0.0


def test_detection_convert_to_best_guess_parameters_stays_in_range() -> None:
    """convert_to_best_guess_parameters must produce phi, theta, d_L, M within physical bounds."""
    series = _make_detection_series(
        dist=1.0,
        delta_dist=0.05,
        phiS=np.pi,
        delta_phiS=0.1,
        qS=np.pi / 2,
        delta_qS=0.1,
        M=5e5,
        delta_M=1e4,
    )
    det = Detection(series)
    det.convert_to_best_guess_parameters()

    assert 0.0 <= det.phi <= 2 * np.pi
    assert 0.0 <= det.theta <= np.pi
    assert det.d_L > 0
    assert det.M >= 1e4
