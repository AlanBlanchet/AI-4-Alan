from __future__ import annotations

import json


def to_hyperparam(v: int | float | Hyperparam | dict | str):
    if isinstance(v, Hyperparam):
        return v
    elif isinstance(v, (float, int)):
        return Hyperparam(start=v)
    elif isinstance(v, dict):
        return Hyperparam(**v)
    elif isinstance(v, str):
        return Hyperparam(**json.loads(v))
    else:
        raise TypeError(f"Cannot convert {v} to Hyperparam")


class Hyperparam:
    def __init__(self, start, end=0, decay=0.995, steps=5000):
        self.start = start
        self.end = end
        self.decay = decay
        self.steps = steps

        self.current_ = start
        self.train_ = True

    def step(self, step=None):
        if step is None:
            self.current_ **= 2 - self.decay
        else:
            step = max(1, min(step, self.steps))

            self.current_ = self.start * (self.end / self.start) ** (step / self.steps)
        return self.current_

    def __call__(self, step=None):
        if not self.train_:
            return 0

        if step is not None:
            self.step(step)

        return self.current_

    def __lt__(self, other):
        return self.current_ < other

    def __rlt__(self, other):
        return other < self.current_

    def __le__(self, other):
        return self.current_ <= other

    def __rle__(self, other):
        return other <= self.current_

    def __ge__(self, other):
        return self.current_ >= other

    def __rge__(self, other):
        return other >= self.current_

    def __gt__(self, other):
        return self.current_ > other

    def __rgt__(self, other):
        return other > self.current_

    def __eq__(self, other):
        return self.current_ == other

    def __req__(self, other):
        return other == self.current_

    def test(self):
        self.train_ = False

    def train(self):
        self.train_ = True