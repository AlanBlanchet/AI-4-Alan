from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..policy import Learner


@dataclass
class CriticNetwork(Learner):
    def __attrs_post_init__(self):
        super().__attrs_post_init__()

        self.l1 = nn.Linear(self.action_shape + self.state_shape, 256)
        self.l2 = nn.Linear(256, 32)
        self.q = nn.Linear(32, 1)

    @property
    def state_dict(self):
        return {
            **self.l1.state_dict(),
            **self.l2.state_dict(),
            **self.q.state_dict(),
        }

    def parameters(self):
        return [
            *self.l1.parameters(),
            *self.l2.parameters(),
            *self.q.parameters(),
        ]

    def __call__(self, state, action):
        q = F.relu(self.l1(torch.cat([state, action], axis=-1)))
        q = F.relu(self.l2(q))
        q = self.q(q)
        return q
