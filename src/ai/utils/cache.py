import warnings
from functools import lru_cache

from joblib import Memory

# Filter joblib trying to identify name collisions
warnings.filterwarnings(
    "ignore", message=".*Cannot detect name collisions.*", category=UserWarning
)


class SmartCache:
    def __init__(self, path: str):
        self.memory = Memory(path, verbose=0)

    def __call__(self, fn=None):
        if fn is None:
            return lambda fn: self.memory.cache(lru_cache(fn))
        return self.memory.cache(lru_cache(fn))


def get_all_subclasses(cls):
    all_subclasses = []

    for subclass in cls.__subclasses__():
        all_subclasses.append(subclass)
        all_subclasses.extend(get_all_subclasses(subclass))

    return all_subclasses
