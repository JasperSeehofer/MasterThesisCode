import logging
import time

def timer_decorator(func) -> callable: 
    def wrapper_function(*args, **kwargs): 
        start = time.time() 
        result = func(*args,  **kwargs) 
        end = time.time()
        logging.debug(f"Function {func.__name__!r} executed in {(end-start):.4f}s")
        return result
    return wrapper_function 