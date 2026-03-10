import pathlib
from typing import Any
from unittest.mock import MagicMock

import pandas as pd
import pytest

import master_thesis_code.constants as constants
from master_thesis_code.datamodels.parameter_space import ParameterSpace

try:
    from master_thesis_code.parameter_estimation.parameter_estimation import ParameterEstimation

    _PE_AVAILABLE = True
except ImportError:
    # parameter_estimation imports LISA_configuration which imports cupy unconditionally;
    # skip on CPU-only machines.
    _PE_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _PE_AVAILABLE,
    reason="parameter_estimation unavailable (LISA_configuration requires cupy)",
)


def _make_minimal_pe(tmp_path: pathlib.Path) -> Any:
    """Construct a ParameterEstimation with mocked LISA response generator."""
    pe = ParameterEstimation.__new__(ParameterEstimation)
    pe.parameter_space = ParameterSpace()
    pe.parameter_space.randomize_parameters()
    pe.T = 5.0
    pe.dt = 10
    pe.waveform_generation_time = 0.0
    # Mock the response generators so no GPU/waveform code runs
    pe.lisa_response_generator = MagicMock()
    pe.snr_check_generator = MagicMock()
    pe.lisa_configuration = MagicMock()
    return pe


def test_save_cramer_rao_bound_creates_csv(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Calling save_cramer_rao_bound once should create a CSV with exactly one data row."""
    csv_path = str(tmp_path / "crb_simulation_$index.csv")
    monkeypatch.setattr(constants, "CRAMER_RAO_BOUNDS_PATH", csv_path)

    pe = _make_minimal_pe(tmp_path)
    crb_dict: dict = {}  # empty CRB dictionary (no Fisher matrix entries)

    pe.save_cramer_rao_bound(
        cramer_rao_bound_dictionary=crb_dict,
        snr=25.0,
        simulation_index=0,
    )

    result_path = csv_path.replace("$index", "0")
    assert pathlib.Path(result_path).exists()
    df = pd.read_csv(result_path)
    assert len(df) == 1


def test_save_cramer_rao_bound_appends(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Calling save_cramer_rao_bound twice should result in a CSV with two data rows."""
    csv_path = str(tmp_path / "crb_simulation_$index.csv")
    monkeypatch.setattr(constants, "CRAMER_RAO_BOUNDS_PATH", csv_path)

    pe = _make_minimal_pe(tmp_path)
    crb_dict: dict = {}

    pe.save_cramer_rao_bound(
        cramer_rao_bound_dictionary=crb_dict,
        snr=25.0,
        simulation_index=1,
    )
    pe.save_cramer_rao_bound(
        cramer_rao_bound_dictionary=crb_dict,
        snr=30.0,
        simulation_index=1,
    )

    result_path = csv_path.replace("$index", "1")
    df = pd.read_csv(result_path)
    assert len(df) == 2


@pytest.mark.gpu
def test_scalar_product_positive_definite() -> None:
    """scalar_product_of_functions(h, h) should be positive for a non-zero signal."""
    from unittest.mock import MagicMock

    import cupy as cp

    from master_thesis_code.LISA_configuration import LisaTdiConfiguration
    from master_thesis_code.parameter_estimation.parameter_estimation import ParameterEstimation

    pe = ParameterEstimation.__new__(ParameterEstimation)
    pe.dt = 10
    pe.parameter_space = ParameterSpace()
    pe.lisa_configuration = LisaTdiConfiguration()
    pe.lisa_response_generator = MagicMock()
    pe.snr_check_generator = MagicMock()

    # Build a synthetic 2-channel signal on the GPU (sine wave)
    n = 10000
    t = cp.linspace(0, n * pe.dt, n)
    sine = cp.sin(2 * cp.pi * 1e-3 * t)
    # Shape: (2, n) matching ESA_TDI_CHANNELS = "AE"
    signal = cp.stack([sine, sine])

    result = pe.scalar_product_of_functions(signal, signal)
    assert float(result) > 0


@pytest.mark.gpu
def test_scalar_product_symmetric() -> None:
    """scalar_product_of_functions(a, b) should approximately equal scalar_product(b, a)."""
    import cupy as cp

    from master_thesis_code.LISA_configuration import LisaTdiConfiguration
    from master_thesis_code.parameter_estimation.parameter_estimation import ParameterEstimation

    pe = ParameterEstimation.__new__(ParameterEstimation)
    pe.dt = 10
    pe.parameter_space = ParameterSpace()
    pe.lisa_configuration = LisaTdiConfiguration()
    pe.lisa_response_generator = MagicMock()
    pe.snr_check_generator = MagicMock()

    n = 10000
    t = cp.linspace(0, n * pe.dt, n)
    a = cp.stack([cp.sin(2 * cp.pi * 1e-3 * t), cp.cos(2 * cp.pi * 5e-4 * t)])
    b = cp.stack([cp.cos(2 * cp.pi * 1e-3 * t), cp.sin(2 * cp.pi * 5e-4 * t)])

    ab = float(pe.scalar_product_of_functions(a, b))
    ba = float(pe.scalar_product_of_functions(b, a))
    assert abs(ab - ba) < 1e-6 * max(abs(ab), abs(ba), 1.0)


# ── _crop_frequency_domain ────────────────────────────────────────────────────
# _crop_frequency_domain uses cp.argmax internally, so it requires a real GPU.


@pytest.mark.gpu
def test_crop_frequency_domain_respects_bounds() -> None:
    """_crop_frequency_domain must return frequencies within [MINIMAL_FREQUENCY, MAXIMAL_FREQUENCY]."""
    import cupy as cp

    from master_thesis_code.constants import MAXIMAL_FREQUENCY, MINIMAL_FREQUENCY
    from master_thesis_code.parameter_estimation.parameter_estimation import ParameterEstimation

    # Build a frequency array that extends well below and above the valid range
    fs = cp.logspace(-7, 2, 10_000)
    integrant = cp.ones_like(fs)

    fs_cropped, integrant_cropped = ParameterEstimation._crop_frequency_domain(fs, integrant)

    assert float(cp.min(fs_cropped)) >= MINIMAL_FREQUENCY
    assert float(cp.max(fs_cropped)) <= MAXIMAL_FREQUENCY


@pytest.mark.gpu
def test_crop_frequency_domain_output_lengths_match() -> None:
    """_crop_frequency_domain must return two arrays of equal length."""
    import cupy as cp

    from master_thesis_code.parameter_estimation.parameter_estimation import ParameterEstimation

    fs = cp.logspace(-6, 1, 5_000)
    integrant = cp.ones_like(fs, dtype=complex)

    fs_cropped, integrant_cropped = ParameterEstimation._crop_frequency_domain(fs, integrant)

    assert len(fs_cropped) == len(integrant_cropped)


# ── _crop_to_same_length ──────────────────────────────────────────────────────
# _crop_to_same_length uses cp.array, so it also requires GPU.


@pytest.mark.gpu
def test_crop_to_same_length_equal_length_inputs() -> None:
    """Given channels of equal length, _crop_to_same_length must preserve that length."""
    import cupy as cp

    from master_thesis_code.parameter_estimation.parameter_estimation import ParameterEstimation

    n = 1000
    channel_a = cp.ones(n)
    channel_b = cp.ones(n)
    # signal_collection shape: list of [channel_A, channel_B] pairs
    collection = [[channel_a, channel_b], [channel_a, channel_b]]

    result = ParameterEstimation._crop_to_same_length(collection)

    # Shape: (2 signals, 2 channels, n)
    assert result.shape[2] == n
