import logging
import time
from master_thesis_code.constants import IS_PLOTTING_ACTIVATED

_LOGGER = logging.getLogger()

def timer_decorator(func) -> callable: 
    def wrapper_function(*args, **kwargs): 
        start = time.time() 
        result = func(*args,  **kwargs) 
        end = time.time()
        _LOGGER.debug(f"Function {func.__name__!r} executed in {(end-start):.4f}s")
        return result
    return wrapper_function 

def if_plotting_activated(func) -> callable: 
    def wrapper_function(*args, **kwargs):
        if IS_PLOTTING_ACTIVATED:
            func(*args,  **kwargs) 
    return wrapper_function 