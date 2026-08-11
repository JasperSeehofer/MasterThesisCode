import functools
import logging
import time
from collections.abc import Callable
from typing import Any

try:
    import cupy as cp

    _CUPY_AVAILABLE = True
except ImportError:
    cp = None
    _CUPY_AVAILABLE = False

_LOGGER = logging.getLogger()


def timer_decorator[F: Callable[..., Any]](func: F) -> F:
    @functools.wraps(func)
    def wrapper_function(*args: Any, **kwargs: Any) -> Any:
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        if _CUPY_AVAILABLE and cp is not None:
            total_bytes = int(cp.get_default_memory_pool().total_bytes()) / 10**9
            _LOGGER.debug(
                f"Function {func.__name__!r} executed in {(end - start):.4f}s, GPU usage: {total_bytes}GB."
            )
        else:
            _LOGGER.debug(f"Function {func.__name__!r} executed in {(end - start):.4f}s.")
        return result

    return wrapper_function  # type: ignore[return-value]
