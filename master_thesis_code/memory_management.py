import GPUtil
from tabulate import tabulate
import cupy as cp
import logging
import os
import matplotlib.pyplot as plt
from time import time

_LOGGER = logging.getLogger()

class MemoryManagement:
    memory_pool: cp.cuda.MemoryPool

    def __init__(self):
        self._gpu_monitor = GPUtil.getGPUs()
        self.memory_pool = cp.get_default_memory_pool()
        self._fft_cache = cp.fft.config.get_plan_cache()
        self._start_time = time()
        self._memory_pool_gpu_usage = []
        self._gpu_usage = []
        self._time_series = []

            
    def gpu_usage_stamp(self) -> None:
        self._time_series.append(time() - self._start_time)
        self._gpu_usage.append([gpu.memoryUsed/1000 for gpu in GPUtil.getGPUs()])
        self._memory_pool_gpu_usage.append(int(self.memory_pool.total_bytes())/10**9)


    def display_GPU_information(self) -> None:
        _LOGGER.info(f"{'='*40} GPU Details {'='*40}")
        gpus = GPUtil.getGPUs()
        list_gpus = []
        for gpu in gpus:
            # get the GPU id
            gpu_id = str(gpu.id)
            # name of GPU
            gpu_name = str(gpu.name)
            # get % percentage of GPU usage of that GPU
            gpu_load = f"{gpu.load*100}%"
            # get free memory in MB format
            gpu_free_memory = f"{gpu.memoryFree}MB"
            # get used memory
            gpu_used_memory = f"{gpu.memoryUsed}MB"
            # get total memory
            gpu_total_memory = f"{gpu.memoryTotal}MB"
            # get GPU temperature in Celsius
            gpu_temperature = f"{gpu.temperature} °C"
            gpu_uuid = str(gpu.uuid)
            list_gpus.append((
                gpu_id, gpu_name, gpu_load, gpu_free_memory, gpu_used_memory,
                gpu_total_memory, gpu_temperature, gpu_uuid
            ))

        _LOGGER.info(tabulate(list_gpus, headers=("id", "name", "load", "free memory", "used memory", "total memory",
                                   "temperature", "uuid")))
   
   
    def display_fft_cache(self) -> None:
        self._fft_cache.show_info()

    def plot_GPU_usage(self) -> None:
        figures_directory = f"saved_figures/monitoring/"

        if not os.path.isdir(figures_directory):
            os.makedirs(figures_directory)

        fig = plt.figure(figsize=(12, 9))
        plt.plot(
            self._time_series,
            self._memory_pool_gpu_usage,
            "-",
            label="Memory Pool",
        )
        for gpu_index in range(len(self._gpu_usage[0]) - 1):
            plt.plot(
            self._time_series,
            self._gpu_usage[:][gpu_index],
            "-",
            label=f"GPU {gpu_index+1}",
        )
        plt.xlabel("t in sec")
        plt.legend()
        plt.savefig(figures_directory + "GPU_usage.png", dpi=300)
        plt.clf()
