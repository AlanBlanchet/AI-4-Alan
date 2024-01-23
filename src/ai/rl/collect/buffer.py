import random
from abc import ABC
from collections import deque
from itertools import islice

import torch


class BaseBuffer(ABC):
    @property
    def buffer_count(self):
        return len(self.buffer)

    def push(self, *args):
        """Save a transition"""
        self.buffer.append(args)

    def reset(self):
        self.buffer.clear()
        self.buffer_sample_count = 0

    def sample(self, batch_size: int):
        # [(s, a, r, s', d), ...)]
        transition = random.sample(self.buffer, batch_size)
        self.buffer_sample_count += 1

        # [(s, ...), (a, ...), ...]
        return zip(*transition)

    def extract(self):
        return zip(*self.buffer)

    def __getitem__(self, key):
        if isinstance(key, slice):
            # Get the start, stop, and step from the slice
            key = list(islice(list(range(len(self))), *key.indices(len(self))))
        elif isinstance(key, int):
            if key < 0:  # Handle negative indices
                key += len(self)
            if key < 0 or key >= len(self):
                raise IndexError(f"The index {key} is out of range.")
            return self.buffer[key]

        if isinstance(key, list):
            return zip(*[self.buffer[i] for i in key])

        raise TypeError("Invalid argument type.")

    def __lt__(self, other):
        return self.buffer_count < other

    def __rlt__(self, other):
        return other < self.buffer_count

    def __le__(self, other):
        return self.buffer_count <= other

    def __rle__(self, other):
        return other <= self.buffer_count

    def __ge__(self, other):
        return self.buffer_count >= other

    def __rge__(self, other):
        return other >= self.buffer_count

    def __gt__(self, other):
        return self.buffer_count > other

    def __rgt__(self, other):
        return other > self.buffer_count

    def __eq__(self, other):
        return self.buffer_count == other

    def __req__(self, other):
        return other == self.buffer_count

    def __len__(self):
        return len(self.buffer)


class Buffer(BaseBuffer):
    def __init__(self):
        self.buffer = []


class DequeueBuffer(BaseBuffer):
    def __init__(self, size: int = 10000):
        self.size = size
        self.buffer: deque[tuple(torch.Tensor)] = self._create_deque()

    def _create_deque(self):
        return deque([], maxlen=self.size)
