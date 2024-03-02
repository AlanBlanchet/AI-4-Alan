"""
This module implements the REINFORCE algorithm.

The main idea behind REINFORCE is to directly optimize the policy function, which maps states to actions, in order to maximize the expected cumulative reward. The algorithm learns by iteratively collecting episodes (sequences of states, actions, and rewards) from the environment and updating the policy based on these trajectories.

For more details, refer to the original REINFORCE paper:
https://proceedings.neurips.cc/paper_files/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf
"""

import numpy as np
import torch
import torch.nn as nn
from torch._tensor import Tensor
from torch.distributions import Categorical
from torch.optim import Adam

from ..collect.collector import Collector
from ..policy import Policy
from ..utils.func import batch_it


class REINFORCE(Policy):
    def __init__(
        self,
        collector: Collector,
        batch_size: int = 128,
        lr: float = 1e-4,
        # How much to discount future rewards.
        # How much should we car about first rewards ?
        gamma: float = 0.95,
    ):
        super().__init__(collector)

        self.lr = lr
        self.gamma = gamma
        self.batch_size = batch_size

        # Policy function returning states
        self.policy = nn.Sequential(
            nn.Linear(self.in_state, 128),
            nn.ReLU(),
            nn.Linear(128, self.out_actions),
        )

        # Adam optimizers
        self._policy_optim = Adam(self.policy.parameters(), lr=self.lr)

    def act(self, state):
        return self.batch_act(state.unsqueeze(dim=0)).squeeze(dim=0).detach()

    def batch_act(self, states) -> torch.Tensor:
        return self.policy(states.to(self.device)).softmax(dim=-1)

    def forward(self, episodes: list[tuple[Tensor, ...]]) -> Tensor:
        policies_loss = []
        rewards_to_gos = []

        for states, actions, rewards, *_ in episodes:
            # Timesteps
            t = torch.arange(states.shape[0], device=self.device)

            for states, actions, rewards, t in batch_it(
                states, actions, rewards, t, size=self.batch_size
            ):
                # Advantage estimation
                # Calculate 'return' or 'reward to go' for all trajectories
                rewards_to_go = (
                    ((self.gamma**t) * rewards).flip(0).cumsum(0).flip(0)
                )  # B

                rewards_to_gos.append(rewards_to_go.mean().item())

                if rewards_to_go.shape[0] > 1:
                    # Normalise
                    rewards_to_go = (rewards_to_go - rewards_to_go.mean()) / (
                        rewards_to_go.std() + 1e-5
                    )

                # Policy gradient
                probs = self.batch_act(states)

                distribution = Categorical(probs)
                log_probs: torch.Tensor = distribution.log_prob(actions.argmax(dim=-1))

                # Mean for normalising because of batches
                loss = -(log_probs * rewards_to_go).mean()

                policies_loss.append(loss.item())

                self._policy_optim.zero_grad()
                loss.backward()
                self._policy_optim.step()

        self.logger.add_scalar("loss/policy", np.mean(policies_loss), self.global_step)

        self.logger.add_scalar(
            "metrics/reward_to_go", np.mean(rewards_to_gos), self.global_step
        )
