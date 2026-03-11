import logging
from time import time

import GPUtil
from tabulate import tabulate

try:
    import cupy as cp

    _CUPY_AVAILABLE = True
except ImportError:
    cp = None
    _CUPY_AVAILABLE = False

_LOGGER = logging.getLogger()


class MemoryManagement:
    def __init__(self) -> None:
        self._gpu_monitor = GPUtil.getGPUs()
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

    def gpu_usage_stamp(self) -> None:
        self._time_series.append(time() - self._start_time)
        self._gpu_usage.append([gpu.memoryUsed / 1000 for gpu in GPUtil.getGPUs()])
        if self.memory_pool is not None:
            self._memory_pool_gpu_usage.append(int(self.memory_pool.total_bytes()) / 10**9)
        else:
            self._memory_pool_gpu_usage.append(0.0)

    def display_GPU_information(self) -> None:
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
