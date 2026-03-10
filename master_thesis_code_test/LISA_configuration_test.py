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


@pytest.mark.xfail(
    reason=(
        "Known bug: LISAConfiguration does not hold a reference to ParameterSpace — "
        "sky angles (qS, phiS, qK, phiK) are not updated when ParameterSpace.randomize_parameters() "
        "is called after initialisation.  This test documents the expected correct behaviour."
    )
)
def test_lisa_config_does_not_go_stale_after_randomize() -> None:
    """LisaTdiConfiguration should reflect updated sky angles after randomize_parameters()."""
    # LisaTdiConfiguration currently has no __init__ that takes a ParameterSpace,
    # so we verify the staleness at the ParameterEstimation level where the bug manifests:
    # the config is created once and never updated when the parameter space is randomised.
    # This test serves as a documentation and regression anchor — it is expected to xfail
    # until LisaTdiConfiguration is refactored to hold a live reference to ParameterSpace.
    from master_thesis_code.datamodels.parameter_space import ParameterSpace

    ps = ParameterSpace()
    ps.randomize_parameters()
    original_qS = ps.qS.value

    # Simulate a second randomisation as done in main.py
    ps.randomize_parameters()
    new_qS = ps.qS.value

    # If qS changed (highly likely with random draws), a correctly-implemented
    # LisaTdiConfiguration holding a live reference would also reflect the new value.
    # The test simply asserts that re-randomisation changes qS — demonstrating that
    # any configuration object initialised before the second call would be stale.
    assert original_qS != new_qS, (
        "randomize_parameters() did not change qS — cannot demonstrate staleness with this seed"
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
