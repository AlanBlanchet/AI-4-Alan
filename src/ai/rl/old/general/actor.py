import torch.nn as nn
import torch.nn.functional as F
from attr import define

from ..policy import Learner


@define(slots=False)
class ActorNetwork(Learner):
    def __attrs_post_init__(self):
        super().__attrs_post_init__()

        self.l1 = nn.Linear(1, 256)
        self.l2 = nn.Linear(256, 128)
        self.l3 = nn.Linear(128, self.action_shape)

    @property
    def state_dict(self):
        return {
            **self.l1.state_dict(),
            **self.l2.state_dict(),
            **self.l3.state_dict(),
        }

    def parameters(self):
        return [
            *self.l1.parameters(),
            *self.l2.parameters(),
            *self.l3.parameters(),
        ]

    def __call__(self, state):
        prob = F.relu(self.l1(state))
        prob = F.relu(self.l2(prob))
        mu = F.tanh(self.l3(prob))
        return mu
