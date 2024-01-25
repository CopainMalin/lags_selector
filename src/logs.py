import logging
import functools
import os
from time import time
from typing import Callable, Any


def init_logger(
    name: str = "RFELogger", filename: str = "rfe_logs.log"
) -> logging.Logger:
    if not os.path.exists("logs"):
        os.makedirs("logs")
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(name)
    handler = logging.FileHandler(f"logs/{filename}")
    logger.addHandler(handler)
    return logger


# decorators
def logging_call(get_logger):
    def decorator(func: Callable[..., Any]) -> Any:
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            logger = get_logger(self)
            logger.info(f"Started execution of {func.__name__}")
            result = func(*args, **kwargs)
            logger.info(f"Finished execution of {func.__name__}")
            return result

        return wrapper

    return decorator


def logging_call_with_time(get_logger):
    def decorator(func: Callable[..., Any]) -> Any:
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            logger = get_logger(self)
            logger.info(f"Started execution of {func.__name__}")
            start_time = time()
            result = func(self, *args, **kwargs)
            logger.info(
                f"Finished execution of {func.__name__} - in {time() - start_time:.2f} seconds."
            )
            return result

        return wrapper

    return decorator
