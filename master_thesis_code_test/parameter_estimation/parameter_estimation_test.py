import pathlib
from typing import Any
from unittest.mock import MagicMock

import pandas as pd
import pytest

try:
    import master_thesis_code.parameter_estimation.parameter_estimation as pe_module
    from master_thesis_code.parameter_estimation.parameter_estimation import ParameterEstimation

    _PE_AVAILABLE = True
except Exception:  # noqa: BLE001 — catches SIGILL-induced ImportError and other load failures
    pe_module = None  # type: ignore[assignment]
    _PE_AVAILABLE = False

from master_thesis_code.datamodels.parameter_space import ParameterSpace

pytestmark = pytest.mark.skipif(
    not _PE_AVAILABLE,
    reason="parameter_estimation unavailable",
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
    pe._crb_buffer = []
    pe._crb_flush_interval = 1  # flush immediately so tests can assert on file contents
    return pe


def test_save_cramer_rao_bound_creates_csv(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Calling save_cramer_rao_bound once should create a CSV with exactly one data row."""
    csv_path = str(tmp_path / "crb_simulation_$index.csv")
    monkeypatch.setattr(pe_module, "CRAMER_RAO_BOUNDS_PATH", csv_path)

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
    monkeypatch.setattr(pe_module, "CRAMER_RAO_BOUNDS_PATH", csv_path)

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


# ── flush_pending_results ─────────────────────────────────────────────────────


def test_flush_pending_results_empty_buffer_is_no_op(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """flush_pending_results() on an empty buffer must not raise or create files."""
    csv_path = str(tmp_path / "crb_simulation_$index.csv")
    monkeypatch.setattr(pe_module, "CRAMER_RAO_BOUNDS_PATH", csv_path)

    pe = _make_minimal_pe(tmp_path)
    pe._crb_buffer = []
    pe.flush_pending_results()

    assert not pathlib.Path(csv_path.replace("$index", "0")).exists()
    assert pe._crb_buffer == []


def test_flush_pending_results_writes_all_buffered_rows(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Explicit flush must write all buffered rows to CSV regardless of interval."""
    csv_path = str(tmp_path / "crb_simulation_$index.csv")
    monkeypatch.setattr(pe_module, "CRAMER_RAO_BOUNDS_PATH", csv_path)

    pe = _make_minimal_pe(tmp_path)
    pe._crb_flush_interval = 100  # disable auto-flush
    crb_dict: dict = {}

    pe.save_cramer_rao_bound(crb_dict, snr=10.0, simulation_index=0)
    pe.save_cramer_rao_bound(crb_dict, snr=11.0, simulation_index=0)
    pe.save_cramer_rao_bound(crb_dict, snr=12.0, simulation_index=0)

    result_path = csv_path.replace("$index", "0")
    assert not pathlib.Path(result_path).exists(), "File should not exist before explicit flush"

    pe.flush_pending_results()

    assert pathlib.Path(result_path).exists()
    df = pd.read_csv(result_path)
    assert len(df) == 3


def test_crb_buffer_auto_flushes_at_interval(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Buffer must auto-flush once it reaches _crb_flush_interval rows."""
    csv_path = str(tmp_path / "crb_simulation_$index.csv")
    monkeypatch.setattr(pe_module, "CRAMER_RAO_BOUNDS_PATH", csv_path)

    pe = _make_minimal_pe(tmp_path)
    pe._crb_flush_interval = 2
    result_path = csv_path.replace("$index", "0")

    # 1st call: buffer has 1 row — no auto-flush yet
    pe.save_cramer_rao_bound({}, snr=10.0, simulation_index=0)
    assert not pathlib.Path(result_path).exists()

    # 2nd call: buffer reaches 2 → auto-flush → file created with 2 rows
    pe.save_cramer_rao_bound({}, snr=11.0, simulation_index=0)
    assert pathlib.Path(result_path).exists()
    df = pd.read_csv(result_path)
    assert len(df) == 2

    # 3rd call: buffer has 1 pending row, file still has only 2 rows
    pe.save_cramer_rao_bound({}, snr=12.0, simulation_index=0)
    df = pd.read_csv(result_path)
    assert len(df) == 2

    # explicit flush writes the remaining row
    pe.flush_pending_results()
    df = pd.read_csv(result_path)
    assert len(df) == 3


# ── PSD cache (GPU) ───────────────────────────────────────────────────────────


@pytest.mark.gpu
def test_psd_cache_returns_same_array_for_same_n() -> None:
    """_get_cached_psd(n) called twice must return the same psd_stack object."""

    from master_thesis_code.constants import ESA_TDI_CHANNELS
    from master_thesis_code.LISA_configuration import LisaTdiConfiguration

    pe = ParameterEstimation.__new__(ParameterEstimation)
    pe.dt = 10
    pe.parameter_space = ParameterSpace()
    pe.parameter_space.randomize_parameters()
    pe.lisa_configuration = LisaTdiConfiguration()
    pe.lisa_response_generator = MagicMock()
    pe.snr_check_generator = MagicMock()
    pe._psd_cache = {}

    result1 = pe._get_cached_psd(10000)
    result2 = pe._get_cached_psd(10000)

    # Same psd_stack object — second call hits the cache
    assert result1[1] is result2[1]
    assert result1[1].shape[0] == len(ESA_TDI_CHANNELS)


@pytest.mark.gpu
def test_psd_cache_shape_is_n_channels_by_n_freqs() -> None:
    """psd_stack returned by _get_cached_psd must have shape (n_channels, n_freqs_cropped)."""
    from master_thesis_code.constants import ESA_TDI_CHANNELS
    from master_thesis_code.LISA_configuration import LisaTdiConfiguration

    pe = ParameterEstimation.__new__(ParameterEstimation)
    pe.dt = 10
    pe.parameter_space = ParameterSpace()
    pe.parameter_space.randomize_parameters()
    pe.lisa_configuration = LisaTdiConfiguration()
    pe.lisa_response_generator = MagicMock()
    pe.snr_check_generator = MagicMock()
    pe._psd_cache = {}

    n = 8192
    fs, psd_stack, lower_idx, upper_idx = pe._get_cached_psd(n)
    n_freqs = upper_idx - lower_idx
    assert psd_stack.shape == (len(ESA_TDI_CHANNELS), n_freqs)


# ── Fisher matrix symmetry (GPU) ─────────────────────────────────────────────


@pytest.mark.gpu
def test_fisher_matrix_is_symmetric() -> None:
    """Fisher information matrix must be symmetric (upper triangle is mirrored to lower)."""
    from unittest.mock import patch

    import cupy as cp
    import numpy as np

    from master_thesis_code.LISA_configuration import LisaTdiConfiguration

    pe = ParameterEstimation.__new__(ParameterEstimation)
    pe.dt = 10
    pe.parameter_space = ParameterSpace()
    pe.parameter_space.randomize_parameters()
    pe.lisa_configuration = LisaTdiConfiguration()
    pe.lisa_response_generator = MagicMock()
    pe.snr_check_generator = MagicMock()
    pe._psd_cache = {}
    pe._crb_buffer = []
    pe._crb_flush_interval = 1
    pe.waveform_generation_time = 0.0

    # Provide synthetic derivatives so no real waveform generation is needed
    n = 10000
    param_names = list(pe.parameter_space._parameters_to_dict().keys())
    mock_derivatives = {name: cp.random.randn(2, n) for name in param_names}

    with patch.object(pe, "finite_difference_derivative", return_value=mock_derivatives):
        F = pe.compute_fisher_information_matrix()

    F_np = cp.asnumpy(F)
    n_params = len(param_names)
    assert F_np.shape == (n_params, n_params)
    assert np.allclose(F_np, F_np.T), "Fisher matrix must be symmetric"
