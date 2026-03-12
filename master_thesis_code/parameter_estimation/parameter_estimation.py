"""EMRI parameter estimation: waveform generation, Fisher matrix, SNR, and Cramér-Rao bounds.

:class:`ParameterEstimation` drives the core computational pipeline: it generates
LISA TDI waveforms using the ``few`` package, computes the signal-to-noise ratio,
and — for detections above the SNR threshold — evaluates the full Fisher information
matrix via a 5-point finite-difference stencil to obtain Cramér-Rao lower bounds on
all 14 EMRI parameters.
"""

import logging
import multiprocessing as mp
import time
import warnings
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

try:
    import cupy as cp
    import cupyx.scipy.fft as cufft

    _CUPY_AVAILABLE = True
except ImportError:
    cp = None
    cufft = None
    _CUPY_AVAILABLE = False

from master_thesis_code.constants import (
    CRAMER_RAO_BOUNDS_PATH,
    ESA_TDI_CHANNELS,
    MAXIMAL_FREQUENCY,
    MINIMAL_FREQUENCY,
    SNR_ANALYSIS_PATH,
    UNDETECTED_EVENTS_PATH,
)
from master_thesis_code.datamodels.parameter_space import Parameter, ParameterSpace
from master_thesis_code.decorators import timer_decorator
from master_thesis_code.exceptions import (
    ParameterOutOfBoundsError,
)
from master_thesis_code.LISA_configuration import LisaTdiConfiguration
from master_thesis_code.waveform_generator import (
    WaveGeneratorType,
    create_lisa_response_generator,
)

_LOGGER = logging.getLogger()


class ParameterEstimation:
    """EMRI waveform-based parameter estimation using the LISA Fisher information matrix.

    Generates LISA TDI waveforms via the ``few`` package, computes the noise-weighted
    signal-to-noise ratio, and — for detections above the SNR threshold — evaluates the
    full :math:`14 \\times 14` Fisher matrix using a 5-point finite-difference stencil to
    obtain Cramér-Rao lower bounds on all EMRI parameters.

    Attributes:
        parameter_space: 14-parameter EMRI configuration space.
        lisa_response_generator: LISA TDI response generator for the full 5-year observation.
        snr_check_generator: LISA TDI response generator for the 1-year SNR pre-check.
        dt: Time sampling interval in seconds.
        T: Observation time in years.
    """

    parameter_space: ParameterSpace
    lisa_response_generator: Any  # ResponseWrapper at runtime; lazy-imported to avoid SIGILL on CPU
    snr_check_generator: Any  # ResponseWrapper at runtime; lazy-imported to avoid SIGILL on CPU
    dt: int = 10  # time sampling in sec
    T: float = 5  # observation time in years

    def __init__(
        self,
        waveform_generation_type: WaveGeneratorType,
        parameter_space: ParameterSpace,
        *,
        use_gpu: bool = True,
    ):
        self.parameter_space = parameter_space
        self._use_gpu = use_gpu
        self.lisa_response_generator = create_lisa_response_generator(
            waveform_generation_type,
            self.dt,
            self.T,
            use_gpu=use_gpu,
        )
        self.snr_check_generator = create_lisa_response_generator(
            waveform_generation_type,
            self.dt,
            1,
            use_gpu=use_gpu,
        )
        self.lisa_configuration = LisaTdiConfiguration()
        self._psd_cache: dict[int, tuple[Any, Any, int, int]] = {}
        self._crb_buffer: list[dict] = []
        self._crb_flush_interval: int = 10
        _LOGGER.info("parameter estimation initialized.")

    def _get_cached_psd(self, n: int) -> tuple[Any, Any, int, int]:
        """Compute and cache (fs_cropped, psd_stack, lower_idx, upper_idx) for waveform length n.

        The PSD depends only on the frequency axis, which is fully determined by n and self.dt.
        Caching eliminates repeated rfftfreq + power_spectral_density calls across Fisher matrix
        inner products — all 105 calls in one Fisher matrix share the same n in practice.
        """
        if n not in self._psd_cache:
            fs_full = cufft.rfftfreq(n, self.dt)[1:]
            lower_idx = int(cp.argmax(fs_full >= MINIMAL_FREQUENCY))
            upper_idx = int(cp.argmax(fs_full >= MAXIMAL_FREQUENCY))
            if upper_idx == 0:
                upper_idx = int(len(fs_full))
            fs = fs_full[lower_idx:upper_idx]
            # A and E channels share the same PSD formula; stack to (n_channels, n_freqs)
            psd_stack = cp.stack(
                [
                    self.lisa_configuration.power_spectral_density(fs, channel=ch)
                    for ch in ESA_TDI_CHANNELS
                ]
            )
            self._psd_cache[n] = (fs, psd_stack, lower_idx, upper_idx)
        return self._psd_cache[n]

    @timer_decorator
    def generate_lisa_response(
        self, update_parameter_dict: dict[str, Any] = {}, use_snr_check_generator: bool = False
    ) -> Any:
        parameters = self.parameter_space._parameters_to_dict() | update_parameter_dict
        if use_snr_check_generator:
            return self.snr_check_generator(*parameters.values())
        return self.lisa_response_generator(*parameters.values())

    def finite_difference_derivative(self) -> dict[str, Any]:
        """Compute partial derivative of the currently set parameters w.r.t. the provided parameter.

        Args:
            parameter_symbol (str): parameter w.r.t. which the derivative is taken (Note: symbol string has to coincide with that in the ParameterSpace list!)

        Returns:
            cp.array[float]: data series of derivative
        """
        derivatives: dict = {}

        # Compute the base waveform once and keep it immutable across iterations.
        # Without this, _crop_to_same_length would progressively shorten the base
        # waveform on each iteration, producing incorrect derivatives for later parameters.
        base_waveform = self.generate_lisa_response()

        for parameter in vars(self.parameter_space).values():
            _LOGGER.info(
                f"Start computing partial derivative of the waveform w.r.t. {parameter.symbol}."
            )

            parameter_evaluated_at = parameter
            derivative_epsilon = parameter.derivative_epsilon

            # check that neighboring points are in parameter range as well
            if (parameter_evaluated_at.value + derivative_epsilon) > parameter.upper_limit:
                raise ParameterOutOfBoundsError(
                    "Tried to set parameter to value out of bounds in derivative."
                )

            parameter.value = parameter_evaluated_at.value + derivative_epsilon
            current_waveform = self.generate_lisa_response(
                update_parameter_dict={parameter.symbol: parameter.value}
            )

            base_cropped, current_waveform = self._crop_to_same_length(
                [base_waveform, current_waveform]
            )

            derivative = (current_waveform - base_cropped) / derivative_epsilon

            derivatives[parameter.symbol] = derivative

        _LOGGER.info("Finished computing partial derivatives.")
        del base_waveform, current_waveform, base_cropped, derivative
        return derivatives

    def five_point_stencil_derivative(
        self, parameter: Parameter, parameter_space: ParameterSpace | None = None
    ) -> Any:
        """Compute partial derivative of the currently set parameters w.r.t. the provided parameter.

        Args:
            parameter_symbol (str): parameter w.r.t. which the derivative is taken (Note: symbol string has to coincide with that in the ParameterSpace list!)

        Returns:
            cp.array[float]: data series of derivative
        """

        print(
            f"[{time.ctime()}] Start computing partial derivative of the waveform w.r.t. {parameter.symbol}.",
            flush=True,
        )
        if parameter_space is not None:
            self.parameter_space = parameter_space

        parameter_evaluated_at = parameter
        derivative_epsilon = parameter.derivative_epsilon

        # check that neighboring points are in parameter range as well
        if ((parameter_evaluated_at.value - 2 * derivative_epsilon) < parameter.lower_limit) or (
            (parameter_evaluated_at.value + 2 * derivative_epsilon) > parameter.upper_limit
        ):
            raise ParameterOutOfBoundsError(
                "Tried to set parameter to value out of bounds in derivative."
            )

        five_point_stencil_steps = [-2.0, -1.0, 1.0, 2.0]
        lisa_responses = []
        for step in five_point_stencil_steps:
            parameter.value = parameter_evaluated_at.value + step * derivative_epsilon

            lisa_responses.append(
                self.generate_lisa_response(
                    update_parameter_dict={parameter.symbol: parameter.value}
                )
            )
            print(
                f"[{time.ctime()}] {mp.current_process().name} lisa response computed",
                flush=True,
            )
        lisa_responses = self._crop_to_same_length(lisa_responses)

        lisa_response_derivative = (
            (-lisa_responses[3] + 8 * lisa_responses[2] - 8 * lisa_responses[1] + lisa_responses[0])
            / 12
            / derivative_epsilon
        )

        _LOGGER.info(
            f"Finished computing partial derivative of the waveform w.r.t. {parameter.symbol}."
        )
        del lisa_responses
        return lisa_response_derivative

    @staticmethod
    def _crop_to_same_length(
        signal_collection: list[list[Any]],
    ) -> Any:
        max_possible_length = min(
            min(
                [
                    min(
                        [len(tdi_channel) for tdi_channel in tdi_channels]
                        for tdi_channels in signal_collection
                    )
                ]
            )
        )
        return cp.array(
            [
                [tdi_channel[:max_possible_length] for tdi_channel in tdi_channels]
                for tdi_channels in signal_collection
            ]
        )

    def scalar_product_of_functions(
        self, tdi_channels_a: npt.NDArray[np.float64], tdi_channels_b: npt.NDArray[np.float64]
    ) -> float:
        """LISA noise-weighted inner product between two TDI waveforms.

        Implements the standard gravitational-wave inner product:

        .. math::

            \\langle h_1 \\mid h_2 \\rangle = 4 \\,\\mathrm{Re}
            \\sum_{\\alpha \\in \\{A, E\\}} \\int_{f_\\min}^{f_\\max}
            \\frac{\\tilde{h}_1^\\alpha(f)\\, \\tilde{h}_2^{\\alpha *}(f)}{S_n^\\alpha(f)}
            \\, df

        summed over TDI channels, where :math:`S_n^\\alpha(f)` is the one-sided
        noise PSD from :meth:`~master_thesis_code.LISA_configuration.LisaTdiConfiguration.power_spectral_density`.

        This is the computational hot path: it is called :math:`O(N_\\theta^2)` times
        per Fisher matrix (105 calls for 14 parameters using the 5-point stencil).

        Args:
            tdi_channels_a: TDI waveform array of shape ``(n_channels, n_samples)``.
            tdi_channels_b: TDI waveform array of shape ``(n_channels, n_samples)``.

        Returns:
            Real-valued inner product :math:`\\langle h_1 \\mid h_2 \\rangle`.
        """
        n_a = tdi_channels_a.shape[-1]
        n_b = tdi_channels_b.shape[-1]
        n_min = min(n_a, n_b)

        # Retrieve cached frequency axis and PSD for this waveform length.
        # fs shape: (n_freqs,); psd_stack shape: (n_channels, n_freqs)
        fs, psd_stack, lower_idx, upper_idx = self._get_cached_psd(n_min)

        # Batch FFT all channels at once: rfft shape (n_channels, n_min//2+1).
        # Slice [1:] skips DC; [lower_idx:upper_idx] restricts to the analysis band.
        a_ffts = cufft.rfft(tdi_channels_a[:, :n_min], axis=-1)[:, 1 + lower_idx : 1 + upper_idx]
        b_ffts_cc = cp.conjugate(cufft.rfft(tdi_channels_b[:, :n_min], axis=-1))[
            :, 1 + lower_idx : 1 + upper_idx
        ]

        # Guard against off-by-one from any rounding in rfft output length.
        n_freq = min(a_ffts.shape[-1], b_ffts_cc.shape[-1], psd_stack.shape[-1])
        a_ffts = a_ffts[:, :n_freq]
        b_ffts_cc = b_ffts_cc[:, :n_freq]
        fs_crop = fs[:n_freq]
        psd_crop = psd_stack[:, :n_freq]

        # Integrand (n_channels, n_freqs); sum over channels then integrate over frequency.
        integrant = (a_ffts * b_ffts_cc) / psd_crop
        result = 4.0 * float(cp.trapz(integrant.sum(axis=0).real, x=fs_crop))
        return result

    @staticmethod
    def _crop_frequency_domain(fs: Any, integrant: Any) -> tuple[Any, Any]:
        if len(fs) != len(integrant):
            _LOGGER.warning("length of frequency domain and integrant are not equal.")

        # find lowest frequency
        lower_limit_index = cp.argmax(fs >= MINIMAL_FREQUENCY)
        upper_limit_index = cp.argmax(fs >= MAXIMAL_FREQUENCY)
        if upper_limit_index == 0:
            upper_limit_index = len(fs)
        return (
            fs[lower_limit_index:upper_limit_index],
            integrant[lower_limit_index:upper_limit_index],
        )

    def compute_fisher_information_matrix(self) -> Any:
        # compute derivatives for fisher information matrix
        parameter_symbol_list = list(self.parameter_space._parameters_to_dict().keys())
        parameter_list = [getattr(self.parameter_space, symbol) for symbol in parameter_symbol_list]
        lisa_response_derivatives: dict[str, Any] = self.finite_difference_derivative()

        fisher_information_matrix = cp.zeros(
            shape=(len(parameter_symbol_list), len(parameter_symbol_list)), dtype=float
        )

        # Fisher matrix is symmetric: Γᵢⱼ = Γⱼᵢ. Compute upper triangle only and mirror,
        # halving the number of expensive scalar_product_of_functions GPU calls (105 vs 196).
        for col, column_parameter_symbol in enumerate(parameter_symbol_list):
            for row in range(col, len(parameter_symbol_list)):
                row_parameter_symbol = parameter_symbol_list[row]
                val = self.scalar_product_of_functions(
                    lisa_response_derivatives[column_parameter_symbol],
                    lisa_response_derivatives[row_parameter_symbol],
                )
                fisher_information_matrix[col][row] = val
                fisher_information_matrix[row][col] = val

        _LOGGER.info("Fisher information matrix has been computed.")
        del lisa_response_derivatives
        return fisher_information_matrix

    @timer_decorator
    def compute_Cramer_Rao_bounds(self) -> dict:
        fisher_information_matrix = self.compute_fisher_information_matrix()

        cramer_rao_bounds = np.matrix(cp.asnumpy(fisher_information_matrix)).I
        _LOGGER.debug("matrix inversion completed.")
        parameter_symbol_list = list(self.parameter_space._parameters_to_dict().keys())

        independent_cramer_rao_bounds = {
            f"delta_{parameter_symbol_list[row]}_delta_{parameter_symbol_list[column]}": cramer_rao_bounds[
                row, column
            ]
            for row in range(len(parameter_symbol_list))
            for column in range(row + 1)
        }

        _LOGGER.info("Finished computing Cramer Rao bounds.")
        del fisher_information_matrix
        del cramer_rao_bounds
        return independent_cramer_rao_bounds

    def compute_signal_to_noise_ratio(self, use_snr_check_generator: bool = False) -> float:
        start = time.time()
        waveform = self.generate_lisa_response(use_snr_check_generator=use_snr_check_generator)
        end = time.time()
        self.waveform_generation_time = round(end - start, 3)

        self.current_waveform = waveform
        snr = cp.sqrt(self.scalar_product_of_functions(waveform, waveform))
        del waveform
        return float(snr)

    def save_cramer_rao_bound(
        self,
        cramer_rao_bound_dictionary: dict,
        snr: float,
        simulation_index: int,
        host_galaxy_index: int = -1,
    ) -> None:
        row = (
            self.parameter_space._parameters_to_dict()
            | cramer_rao_bound_dictionary
            | {
                "T": self.T,
                "dt": self.dt,
                "SNR": snr,
                "generation_time": self.waveform_generation_time,
                "host_galaxy_index": host_galaxy_index,
                "_simulation_index": simulation_index,
            }
        )
        self._crb_buffer.append(row)
        if len(self._crb_buffer) >= self._crb_flush_interval:
            self.flush_pending_results()

    def flush_pending_results(self) -> None:
        """Write all buffered Cramér-Rao bound rows to disk and clear the buffer.

        Call this at the end of a simulation run to ensure no results are lost.
        Rows are grouped by simulation index so a single read/write per file replaces
        one read/write per detection (the previous behaviour).
        """
        if not self._crb_buffer:
            return
        # Group rows by simulation index (in practice always the same within a job).
        by_index: dict[int, list[dict]] = {}
        for row in self._crb_buffer:
            idx = row.pop("_simulation_index")
            by_index.setdefault(idx, []).append(row)
        for sim_idx, rows in by_index.items():
            file_path = CRAMER_RAO_BOUNDS_PATH.replace("$index", str(sim_idx))
            try:
                existing = pd.read_csv(file_path)
            except FileNotFoundError:
                existing = pd.DataFrame(columns=list(rows[0].keys()))
            combined = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True)
            combined.to_csv(file_path, index=False)
            _LOGGER.info(f"Flushed {len(rows)} Cramér-Rao bounds to {file_path}")
        self._crb_buffer.clear()

    def SNR_analysis(self) -> None:
        # setup waveformgenerators for different observation times
        waveform_generators = {
            0: create_lisa_response_generator(
                WaveGeneratorType.PN5_AAK, self.dt, 0.5, use_gpu=self._use_gpu
            ),
            1: create_lisa_response_generator(
                WaveGeneratorType.PN5_AAK, self.dt, 1, use_gpu=self._use_gpu
            ),
            2: create_lisa_response_generator(
                WaveGeneratorType.PN5_AAK, self.dt, 2, use_gpu=self._use_gpu
            ),
            3: create_lisa_response_generator(
                WaveGeneratorType.PN5_AAK, self.dt, 3, use_gpu=self._use_gpu
            ),
            4: create_lisa_response_generator(
                WaveGeneratorType.PN5_AAK, self.dt, 5, use_gpu=self._use_gpu
            ),
        }
        parameter_set_index = 0
        for _ in range(200):
            self.parameter_space.randomize_parameters()
            for T, waveform_generator in zip([0.5, 1, 2, 3, 5], waveform_generators.values()):
                self.lisa_response_generator = waveform_generator
                self.T = T
                try:
                    warnings.filterwarnings("error")
                    snr = self.compute_signal_to_noise_ratio()
                    warnings.resetwarnings()
                except Warning as e:
                    if "Mass ratio" in str(e):
                        _LOGGER.warning(
                            "Caught warning that mass ratio is out of bounds. Continue with new parameters..."
                        )
                        continue
                    else:
                        _LOGGER.warning(f"{str(e)}. Continue with new parameters...")
                        continue
                except ParameterOutOfBoundsError as e:
                    _LOGGER.warning(
                        f"Caught ParameterOutOfBoundsError during parameter estimation: {str(e)}. Continue with new parameters..."
                    )
                    continue
                except AssertionError as e:
                    _LOGGER.warning(
                        f"caught AssertionError: {str(e)}. Continue with new parameters..."
                    )
                    continue
                except RuntimeError as e:
                    _LOGGER.warning(
                        f"Caught RuntimeError during waveform generation : {str(e)} .\n Continue with new parameters..."
                    )
                    continue
                except ValueError as e:
                    if "EllipticK" in str(e):
                        _LOGGER.warning(
                            "Caught EllipticK error from waveform generator. Continue with new parameters..."
                        )
                        continue
                    elif "Brent root solver does not converge" in str(e):
                        _LOGGER.warning(
                            "Caught brent root solver error because it did not converge. Continue with new parameters..."
                        )
                        continue
                    else:
                        raise ValueError(e)

                self.save_snr_analysis(snr, parameter_set_index)
            parameter_set_index += 1

    def save_snr_analysis(self, snr: float, parameter_set_index: int) -> None:
        try:
            snr_analysis = pd.read_csv(SNR_ANALYSIS_PATH)

        except FileNotFoundError:
            parameters_list = list(self.parameter_space._parameters_to_dict().keys())
            parameters_list.extend(["T", "SNR", "generation_time"])
            snr_analysis = pd.DataFrame(columns=parameters_list)

        new_snr_analysis_dict = self.parameter_space._parameters_to_dict() | {
            "T": self.T,
            "dt": self.dt,
            "SNR": snr,
            "generation_time": self.waveform_generation_time,
            "parameter_set_index": parameter_set_index,
        }

        new_snr_analysis = pd.DataFrame([new_snr_analysis_dict])

        snr_analysis = pd.concat([snr_analysis, new_snr_analysis], ignore_index=True)
        snr_analysis.to_csv(SNR_ANALYSIS_PATH, index=False)

    def save_not_detected(self, snr: float, simulation_index: int) -> None:
        file_path = UNDETECTED_EVENTS_PATH.replace("$index", str(simulation_index))
        try:
            snr_analysis = pd.read_csv(file_path)

        except FileNotFoundError:
            parameters_list = list(self.parameter_space._parameters_to_dict().keys())
            parameters_list.extend(["T", "dt", "SNR", "generation_time"])
            snr_analysis = pd.DataFrame(columns=parameters_list)

        new_snr_analysis_dict = self.parameter_space._parameters_to_dict() | {
            "T": self.T,
            "dt": self.dt,
            "SNR": snr,
            "generation_time": self.waveform_generation_time,
        }

        new_snr_analysis = pd.DataFrame([new_snr_analysis_dict])
        if snr_analysis.empty:
            snr_analysis = new_snr_analysis
        else:
            snr_analysis = pd.concat([snr_analysis, new_snr_analysis], ignore_index=True)
        snr_analysis.to_csv(file_path, index=False)
