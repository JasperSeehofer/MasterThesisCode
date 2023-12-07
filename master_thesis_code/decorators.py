import logging
import time
import cupy as cp
import errno
import os
import signal
import functools
from master_thesis_code.exceptions import TimeoutError
from master_thesis_code.constants import IS_PLOTTING_ACTIVATED

_LOGGER = logging.getLogger()

def timer_decorator(func) -> callable: 
    def wrapper_function(*args, **kwargs): 
        start = time.time() 
        result = func(*args,  **kwargs) 
        end = time.time()
        total_bytes = int(cp.get_default_memory_pool().total_bytes())/10**9
        _LOGGER.debug(f"Function {func.__name__!r} executed in {(end-start):.4f}s, GPU usage: {total_bytes}GB.")
        return result
    return wrapper_function 

def if_plotting_activated(func) -> callable: 
    def wrapper_function(*args, **kwargs):
        if IS_PLOTTING_ACTIVATED:
            func(*args,  **kwargs) 
    return wrapper_function

def timeout(seconds=10, error_message=os.strerror(errno.ETIME)):
    def decorator(func):
        def _handle_timeout(signum, frame):
            raise TimeoutError(error_message)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result

        return wrapper

    return decorator
