import logging
from time import time

from tabulate import tabulate

try:
    import GPUtil

    _GPUTIL_AVAILABLE = True
except ImportError:
    GPUtil = None  # type: ignore[assignment,unused-ignore]
    _GPUTIL_AVAILABLE = False

try:
    import cupy as cp

    _CUPY_AVAILABLE = True
except ImportError:
    cp = None
    _CUPY_AVAILABLE = False

_LOGGER = logging.getLogger()


class MemoryManagement:
    def __init__(self, use_gpu: bool = False) -> None:
        self._use_gpu = use_gpu
        if _GPUTIL_AVAILABLE:
            self._gpu_monitor = GPUtil.getGPUs()
        else:
            self._gpu_monitor = []
        if _CUPY_AVAILABLE and cp is not None:
            self.memory_pool = cp.get_default_memory_pool()
            self._fft_cache = cp.fft.config.get_plan_cache()
        else:
            self.memory_pool = None
            self._fft_cache = None
        self._start_time = time()
        self._memory_pool_gpu_usage: list[float] = []
        self._gpu_usage: list[list[float]] = []
        self._time_series: list[float] = []

    def free_memory_pool(self) -> None:
        """Free CuPy memory-pool blocks. Safe to call every simulation step. No-op on CPU."""
        if self.memory_pool is not None:
            self.memory_pool.free_all_blocks()

    def clear_fft_cache(self) -> None:
        """Clear FFT plan cache. Expensive — rebuilds on next rfft. No-op on CPU."""
        if self._fft_cache is not None:
            self._fft_cache.clear()

    def free_gpu_memory_if_pressured(self, threshold: float = 0.8) -> None:
        """Free memory pool every call; clear FFT cache only when pool usage exceeds threshold.

        Args:
            threshold: Fraction of total GPU memory (default 0.8 = 64 GB on 80 GB H100).
        """
        self.free_memory_pool()
        if self.memory_pool is None or not _CUPY_AVAILABLE or cp is None:
            return
        try:
            _free, total = cp.cuda.runtime.memGetInfo()
        except Exception:  # noqa: BLE001 — tolerate absent runtime on CPU-only
            return
        used = int(self.memory_pool.total_bytes())
        if total > 0 and used / total >= threshold:
            _LOGGER.info(
                "FFT cache cleared — pool usage %.1f%% of %.1f GB",
                100.0 * used / total,
                total / 1e9,
            )
            self.clear_fft_cache()

    def free_gpu_memory(self) -> None:
        """Deprecated alias. Routes to free_gpu_memory_if_pressured()."""
        import warnings

        warnings.warn(
            "free_gpu_memory() is deprecated; use free_gpu_memory_if_pressured() "
            "or free_memory_pool() directly.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.free_gpu_memory_if_pressured()

    def gpu_usage_stamp(self) -> None:
        self._time_series.append(time() - self._start_time)
        if _GPUTIL_AVAILABLE:
            self._gpu_usage.append([gpu.memoryUsed / 1000 for gpu in GPUtil.getGPUs()])
        else:
            self._gpu_usage.append([])
        if self.memory_pool is not None:
            self._memory_pool_gpu_usage.append(int(self.memory_pool.total_bytes()) / 10**9)
        else:
            self._memory_pool_gpu_usage.append(0.0)

    def display_GPU_information(self) -> None:
        if not _GPUTIL_AVAILABLE or not self._gpu_monitor:
            _LOGGER.info("No GPU available.")
            return
        _LOGGER.info(f"{'=' * 40} GPU Details {'=' * 40}")
        gpus = GPUtil.getGPUs()
        list_gpus = []
        for gpu in gpus:
            gpu_id = str(gpu.id)
            gpu_name = str(gpu.name)
            gpu_load = f"{gpu.load * 100}%"
            gpu_free_memory = f"{gpu.memoryFree}MB"
            gpu_used_memory = f"{gpu.memoryUsed}MB"
            gpu_total_memory = f"{gpu.memoryTotal}MB"
            gpu_temperature = f"{gpu.temperature} °C"
            gpu_uuid = str(gpu.uuid)
            list_gpus.append(
                (
                    gpu_id,
                    gpu_name,
                    gpu_load,
                    gpu_free_memory,
                    gpu_used_memory,
                    gpu_total_memory,
                    gpu_temperature,
                    gpu_uuid,
                )
            )

        _LOGGER.info(
            tabulate(
                list_gpus,
                headers=(
                    "id",
                    "name",
                    "load",
                    "free memory",
                    "used memory",
                    "total memory",
                    "temperature",
                    "uuid",
                ),
            )
        )

    def display_fft_cache(self) -> None:
        if self._fft_cache is not None:
            self._fft_cache.show_info()

    @property
    def time_series(self) -> list[float]:
        """Wall-clock seconds since start for each GPU usage stamp."""
        return self._time_series

    @property
    def memory_pool_gpu_usage(self) -> list[float]:
        """CuPy memory pool total bytes (GB) at each stamp."""
        return self._memory_pool_gpu_usage

    @property
    def gpu_usage(self) -> list[list[float]]:
        """Per-GPU memory usage (GB) at each stamp."""
        return self._gpu_usage
