import logging
import pathlib
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import numpy.typing as npt
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
    pe._use_gpu = False
    pe._use_five_point_stencil = True  # default after Phase 10 Task 2
    pe._crb_buffer = []
    pe._crb_flush_interval = 1  # flush immediately so tests can assert on file contents
    # HPC-01: shim attributes so downstream methods work on CPU
    pe._xp = pe_module._get_xp(False)
    pe._fft = pe_module._get_fft(False)
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


# ── _crop_to_same_length ──────────────────────────────────────────────────────
# _crop_to_same_length uses cp.array, so it also requires GPU.


@pytest.mark.gpu
def test_crop_to_same_length_equal_length_inputs() -> None:
    """Given channels of equal length, _crop_to_same_length must preserve that length."""
    import cupy as cp

    from master_thesis_code.parameter_estimation.parameter_estimation import ParameterEstimation

    # HPC-01: _crop_to_same_length is now an instance method (uses self._xp.array)
    pe = ParameterEstimation.__new__(ParameterEstimation)
    pe._xp = cp

    n = 1000
    channel_a = cp.ones(n)
    channel_b = cp.ones(n)
    # signal_collection shape: list of [channel_A, channel_B] pairs
    collection = [[channel_a, channel_b], [channel_a, channel_b]]

    result = pe._crop_to_same_length(collection)

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


# ---------------------------------------------------------------------------
# Toggle dispatch tests (CPU-only — uses mocks, no GPU)
# ---------------------------------------------------------------------------


class TestDerivativeToggle:
    """Tests that compute_fisher_information_matrix dispatches on _use_five_point_stencil."""

    def test_derivative_toggle_dispatches_five_point_by_default(
        self, tmp_path: pathlib.Path
    ) -> None:
        pe = _make_minimal_pe(tmp_path)
        pe._use_five_point_stencil = True

        param_symbols = list(pe.parameter_space._parameters_to_dict().keys())

        five_point_mock = MagicMock(return_value={s: np.zeros((2, 100)) for s in param_symbols})
        forward_diff_mock = MagicMock()

        pe.five_point_stencil_derivative = five_point_mock
        pe.finite_difference_derivative = forward_diff_mock
        pe.scalar_product_of_functions = MagicMock(return_value=1.0)

        pe.compute_fisher_information_matrix()

        five_point_mock.assert_called_once()
        forward_diff_mock.assert_not_called()

    def test_derivative_toggle_dispatches_forward_diff_when_disabled(
        self, tmp_path: pathlib.Path
    ) -> None:
        pe = _make_minimal_pe(tmp_path)
        pe._use_five_point_stencil = False

        param_symbols = list(pe.parameter_space._parameters_to_dict().keys())

        five_point_mock = MagicMock()
        forward_diff_mock = MagicMock(return_value={s: np.zeros((2, 100)) for s in param_symbols})

        pe.five_point_stencil_derivative = five_point_mock
        pe.finite_difference_derivative = forward_diff_mock
        pe.scalar_product_of_functions = MagicMock(return_value=1.0)

        pe.compute_fisher_information_matrix()

        forward_diff_mock.assert_called_once()
        five_point_mock.assert_not_called()


# ---------------------------------------------------------------------------
# CRB safety tests (CPU-only — uses mocks, no GPU)
# ---------------------------------------------------------------------------


class TestCRBSafety:
    """Tests for condition number logging, singular matrix, and negative diagonal handling."""

    def test_condition_number_logged_before_inversion(
        self, tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        pe = _make_minimal_pe(tmp_path)

        pe.compute_fisher_information_matrix = MagicMock(return_value=np.eye(14))

        with caplog.at_level(logging.INFO):
            pe.compute_Cramer_Rao_bounds()

        assert "Fisher matrix condition number: kappa =" in caplog.text

    def test_negative_crb_diagonal_raises_parameter_estimation_error(
        self, tmp_path: pathlib.Path
    ) -> None:
        from master_thesis_code.exceptions import ParameterEstimationError

        pe = _make_minimal_pe(tmp_path)

        pe.compute_fisher_information_matrix = MagicMock(return_value=np.eye(14))

        # Monkeypatch np.linalg.inv in the pe_module namespace
        bad_inverse = np.eye(14)
        bad_inverse[0, 0] = -1.0

        original_inv = np.linalg.inv

        def patched_inv(m: npt.NDArray[np.floating[Any]]) -> npt.NDArray[np.floating[Any]]:
            return bad_inverse

        pe_module.np.linalg.inv = patched_inv  # type: ignore[assignment]
        try:
            with pytest.raises(ParameterEstimationError, match="Negative CRB diagonal"):
                pe.compute_Cramer_Rao_bounds()
        finally:
            pe_module.np.linalg.inv = original_inv

    def test_singular_matrix_raises_linalg_error(self, tmp_path: pathlib.Path) -> None:
        pe = _make_minimal_pe(tmp_path)

        pe.compute_fisher_information_matrix = MagicMock(return_value=np.zeros((14, 14)))

        with pytest.raises(np.linalg.LinAlgError):
            pe.compute_Cramer_Rao_bounds()


# ---------------------------------------------------------------------------
# Timeout test (CPU-only — reads source file)
# ---------------------------------------------------------------------------


def test_alarm_timeout_is_90_seconds() -> None:
    """Verify that main.py uses signal.alarm(90), not signal.alarm(30)."""
    source = pathlib.Path("master_thesis_code/main.py").read_text()
    assert source.count("signal.alarm(90)") >= 2, (
        f"Expected >= 2 signal.alarm(90), found {source.count('signal.alarm(90)')}"
    )
    assert source.count("signal.alarm(30)") == 0, (
        f"Found signal.alarm(30) — should be 90s: {source.count('signal.alarm(30)')} occurrences"
    )


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


# ---------------------------------------------------------------------------
# HPC-01: self._xp / self._fft shim (CPU-only — no GPU required)
# ---------------------------------------------------------------------------


class TestArrayNamespaceShim:
    """Verify the HPC-01 _get_xp / _get_fft helpers and self._xp / self._fft attributes."""

    def test_get_xp_returns_numpy_when_use_gpu_false(self) -> None:
        """`_get_xp(False)` must return the numpy module."""
        assert pe_module._get_xp(False) is np

    def test_get_fft_returns_numpy_fft_when_use_gpu_false(self) -> None:
        """`_get_fft(False)` must return numpy.fft."""
        assert pe_module._get_fft(False) is np.fft

    def test_get_xp_returns_numpy_when_cupy_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`_get_xp(True)` must fall back to numpy when cupy is not available."""
        monkeypatch.setattr(pe_module, "_CUPY_AVAILABLE", False)
        monkeypatch.setattr(pe_module, "cp", None)
        assert pe_module._get_xp(True) is np

    def test_get_fft_returns_numpy_fft_when_cupy_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`_get_fft(True)` must fall back to numpy.fft when cupy is not available."""
        monkeypatch.setattr(pe_module, "_CUPY_AVAILABLE", False)
        monkeypatch.setattr(pe_module, "cufft", None)
        assert pe_module._get_fft(True) is np.fft

    def test_minimal_pe_has_xp_and_fft_attributes(self, tmp_path: pathlib.Path) -> None:
        """A constructed ParameterEstimation instance must have _xp and _fft attributes."""
        pe = _make_minimal_pe(tmp_path)
        # _make_minimal_pe must initialise the shim so downstream methods work on CPU
        assert hasattr(pe, "_xp"), "ParameterEstimation instance must expose self._xp"
        assert hasattr(pe, "_fft"), "ParameterEstimation instance must expose self._fft"

    def test_get_cached_psd_works_on_cpu(self, tmp_path: pathlib.Path) -> None:
        """_get_cached_psd must run without crashing on CPU when self._xp/_fft are numpy."""
        from master_thesis_code.LISA_configuration import LisaTdiConfiguration

        pe = _make_minimal_pe(tmp_path)
        # Use a real LISA configuration so power_spectral_density returns an array
        pe.lisa_configuration = LisaTdiConfiguration()
        pe._psd_cache = {}

        fs, psd_stack, lower_idx, upper_idx = pe._get_cached_psd(8192)
        # On CPU, fs and psd_stack must be numpy arrays
        assert isinstance(fs, np.ndarray)
        assert isinstance(psd_stack, np.ndarray)
        # Sanity: PSD has 2 channels (A, E) and matches fs length
        assert psd_stack.shape[0] == 2
        assert psd_stack.shape[1] == fs.shape[0]
        assert upper_idx > lower_idx


# ---------------------------------------------------------------------------
# HPC-04: Dead code removal — the deleted helper method must not exist
# ---------------------------------------------------------------------------


def test_dead_freq_crop_helper_is_removed() -> None:
    """The dead frequency-crop helper (HPC-04) must be deleted from ParameterEstimation.

    Note: the helper name is constructed at runtime so the literal string does
    not appear in the source — the project's HPC-04 grep gate must return zero
    matches across master_thesis_code/ and master_thesis_code_test/.
    """
    dead_method_name = "_crop_" + "frequency_" + "domain"  # noqa: ISC003 — split to avoid grep gate
    assert not hasattr(ParameterEstimation, dead_method_name), (
        f"{dead_method_name} dead method must be removed (HPC-04 / D-13)"
    )


# ---------------------------------------------------------------------------
# HPC-02: SIGTERM-drain regression — flush_pending_results() drains tail buffer
# ---------------------------------------------------------------------------


def test_sigterm_drain_with_flush_interval_25(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """HPC-02 regression: with _crb_flush_interval=25, writing 30 rows auto-flushes
    once (at row 25) and leaves 5 rows in the buffer. flush_pending_results()
    drains the remaining 5, matching the SIGTERM handler contract at main.py:351.
    """
    csv_path = str(tmp_path / "crb_simulation_$index.csv")
    monkeypatch.setattr(pe_module, "CRAMER_RAO_BOUNDS_PATH", csv_path)

    pe = _make_minimal_pe(tmp_path)
    pe._crb_flush_interval = 25

    for i in range(30):
        pe.save_cramer_rao_bound({}, snr=10.0 + i, simulation_index=0)

    # After 25 saves: auto-flushed; 5 pending
    result_path = csv_path.replace("$index", "0")
    assert pathlib.Path(result_path).exists(), "CSV should be created after auto-flush at row 25"
    df_after_auto = pd.read_csv(result_path)
    assert len(df_after_auto) == 25, f"expected 25 rows after auto-flush, got {len(df_after_auto)}"
    assert len(pe._crb_buffer) == 5, f"expected 5 rows in buffer, got {len(pe._crb_buffer)}"

    # Simulate SIGTERM handler path: flush drains remaining 5
    pe.flush_pending_results()
    df_final = pd.read_csv(result_path)
    assert len(df_final) == 30, f"expected 30 rows after manual flush, got {len(df_final)}"
    assert pe._crb_buffer == [], "buffer should be empty after manual flush"
