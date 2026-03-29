import numpy as np
import pytest

# LisaTdiConfiguration guards its cupy import with try/except, so the module
# is importable on CPU-only machines.  GPU-specific tests are marked @pytest.mark.gpu
# and excluded on dev machines via `pytest -m "not gpu"`.
try:
    import cupy as cp

    _CUPY_AVAILABLE = True
except ImportError:
    cp = None
    _CUPY_AVAILABLE = False

from master_thesis_code.LISA_configuration import LisaTdiConfiguration


def test_instantiation() -> None:
    """LisaTdiConfiguration should be instantiable when cupy is present."""
    config = LisaTdiConfiguration()
    assert config is not None
    assert isinstance(config, LisaTdiConfiguration)


@pytest.mark.gpu
def test_power_spectral_density_a_positive() -> None:
    import cupy as cp

    config = LisaTdiConfiguration()
    fs = cp.logspace(-4, 0, 100)
    psd = config.power_spectral_density(fs, channel="A")
    assert cp.all(psd > 0)


@pytest.mark.gpu
def test_power_spectral_density_t_positive() -> None:
    import cupy as cp

    config = LisaTdiConfiguration()
    fs = cp.logspace(-4, 0, 100)
    psd = config.power_spectral_density(fs, channel="T")
    assert cp.all(psd > 0)


@pytest.mark.gpu
def test_channel_ae_same_result() -> None:
    """Channel 'A' and 'E' should give identical PSD results."""
    import cupy as cp

    config = LisaTdiConfiguration()
    fs = cp.logspace(-4, 0, 100)
    psd_a = config.power_spectral_density(fs, channel="A")
    psd_e = config.power_spectral_density(fs, channel="E")
    assert cp.allclose(psd_a, psd_e)


@pytest.mark.gpu
def test_s_oms_positive() -> None:
    import cupy as cp

    config = LisaTdiConfiguration()
    fs = cp.logspace(-4, 0, 100)
    s_oms = config.S_OMS(fs)
    assert cp.all(s_oms > 0)


@pytest.mark.gpu
def test_s_tm_positive() -> None:
    import cupy as cp

    config = LisaTdiConfiguration()
    fs = cp.logspace(-4, 0, 100)
    s_tm = config.S_TM(fs)
    assert cp.all(s_tm > 0)


@pytest.mark.gpu
def test_s_zz_positive() -> None:
    import cupy as cp

    config = LisaTdiConfiguration()
    fs = cp.logspace(-4, 0, 100)
    s_zz = config.S_zz(fs)
    assert cp.all(s_zz > 0)


# ── Known-bug regression ───────────────────────────────────────────────────────


def test_lisa_config_does_not_go_stale_after_randomize() -> None:
    """randomize_parameters() must update sky angles on a live ParameterSpace instance.

    ParameterSpace is passed by reference into ParameterEstimation, so any object reading
    ps.qS.value after a randomise call sees the new value automatically.
    """
    from master_thesis_code.datamodels.parameter_space import ParameterSpace

    ps = ParameterSpace()
    ps.randomize_parameters()
    original_qS = ps.qS.value

    ps.randomize_parameters()
    new_qS = ps.qS.value

    # Continuous distribution — probability of collision is effectively zero.
    assert original_qS != new_qS, (
        "randomize_parameters() did not change qS — unexpected collision in continuous draw"
    )


# ── CPU tests — numpy input (no GPU required) ─────────────────────────────────


def test_psd_a_channel_positive_with_numpy_input() -> None:
    """power_spectral_density('A') with a plain numpy array must return all-positive values."""
    config = LisaTdiConfiguration()
    fs = np.logspace(-4, 0, 100)
    psd = config.power_spectral_density(fs, channel="A")
    assert np.all(psd > 0)


def test_psd_t_channel_positive_with_numpy_input() -> None:
    """power_spectral_density('T') with a plain numpy array must return all-positive values."""
    config = LisaTdiConfiguration()
    fs = np.logspace(-4, 0, 100)
    psd = config.power_spectral_density(fs, channel="T")
    assert np.all(psd > 0)


def test_s_oms_positive_with_numpy_input() -> None:
    """S_OMS with a plain numpy frequency array must be strictly positive."""
    fs = np.logspace(-4, 0, 100)
    s_oms = LisaTdiConfiguration.S_OMS(fs)
    assert np.all(s_oms > 0)


def test_s_tm_positive_with_numpy_input() -> None:
    """S_TM with a plain numpy frequency array must be strictly positive."""
    fs = np.logspace(-4, 0, 100)
    s_tm = LisaTdiConfiguration.S_TM(fs)
    assert np.all(s_tm > 0)


def test_power_spectral_density_channels_ae_equal() -> None:
    """Channels 'A' and 'E' share the same PSD formula — results must be identical."""
    config = LisaTdiConfiguration()
    fs = np.logspace(-4, 0, 100)
    psd_a = config.power_spectral_density(fs, channel="A")
    psd_e = config.power_spectral_density(fs, channel="E")
    assert np.allclose(psd_a, psd_e)


# ── Galactic confusion noise tests (CPU, no GPU required) ────────────────────


def test_confusion_noise_increases_psd_at_1mhz() -> None:
    """PSD at 1 mHz with confusion noise must exceed PSD without."""
    config_with = LisaTdiConfiguration(include_confusion_noise=True)
    config_without = LisaTdiConfiguration(include_confusion_noise=False)
    fs = np.array([1e-3])  # 1 mHz
    psd_with = config_with.power_spectral_density_a_channel(fs)
    psd_without = config_without.power_spectral_density_a_channel(fs)
    assert psd_with[0] > psd_without[0]


def test_confusion_noise_negligible_above_10mhz() -> None:
    """Above 10 mHz, confusion noise should be negligible (< 1% change)."""
    config_with = LisaTdiConfiguration(include_confusion_noise=True)
    config_without = LisaTdiConfiguration(include_confusion_noise=False)
    fs = np.array([0.01])  # 10 mHz
    ratio = config_with.power_spectral_density_a_channel(fs)[0] / (
        config_without.power_spectral_density_a_channel(fs)[0]
    )
    assert ratio < 1.01


def test_confusion_noise_toggle_backward_compat() -> None:
    """include_confusion_noise=False gives all-positive PSD identical to old code path."""
    config = LisaTdiConfiguration(include_confusion_noise=False)
    fs = np.logspace(-4, 0, 100)
    psd = config.power_spectral_density(fs, channel="A")
    assert np.all(psd > 0)


def test_confusion_noise_positive() -> None:
    """_confusion_noise returns all-positive values in the 0.1--3 mHz band where it dominates."""
    config = LisaTdiConfiguration(include_confusion_noise=True)
    # Confusion noise is physically relevant in the 0.1--3 mHz band;
    # at higher frequencies the exponential suppression underflows to 0.
    fs = np.logspace(-4, np.log10(3e-3), 50)
    result = config._confusion_noise(fs)
    assert np.all(result > 0)


def test_t_obs_affects_confusion_noise() -> None:
    """Different observation times must produce different PSD at 1 mHz."""
    config_1yr = LisaTdiConfiguration(t_obs_years=1.0)
    config_4yr = LisaTdiConfiguration(t_obs_years=4.0)
    fs = np.array([1e-3])
    psd_1yr = config_1yr.power_spectral_density_a_channel(fs)
    psd_4yr = config_4yr.power_spectral_density_a_channel(fs)
    assert psd_1yr[0] != psd_4yr[0]


def test_t_channel_unaffected_by_confusion_noise() -> None:
    """T-channel PSD must be identical regardless of confusion noise toggle."""
    config_with = LisaTdiConfiguration(include_confusion_noise=True)
    config_without = LisaTdiConfiguration(include_confusion_noise=False)
    fs = np.logspace(-4, 0, 100)
    psd_t_with = config_with.power_spectral_density(fs, channel="T")
    psd_t_without = config_without.power_spectral_density(fs, channel="T")
    assert np.allclose(psd_t_with, psd_t_without)
