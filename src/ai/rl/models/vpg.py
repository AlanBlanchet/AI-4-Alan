"""
This module implements the Vanilla Policy Gradient (VPG) algorithm.

The main idea behind VPG is to directly optimize the policy function, which maps states to actions, in order to maximize the expected cumulative reward. The algorithm learns by iteratively collecting trajectories (sequences of states, actions, and rewards) from the environment and updating the policy based on these trajectories.

It uses a value function as a baseline to reduce the variance of the gradient estimator. The value function is trained to approximate the expected return of the policy.

For more details, refer to the original VPG paper:
https://proceedings.neurips.cc/paper_files/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.optim import Adam

from ..collect.collector import Collector
from ..policy import Policy
from ..utils.func import batch_it


class VPG(Policy):
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
            nn.Linear(self.in_state, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.out_actions),
        )

        # Equivalent of the baseline function
        # The goal is to approximate the expected return
        self.value = nn.Sequential(
            nn.Linear(self.in_state, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        # Adam optimizers
        self._policy_optim = Adam(self.policy.parameters(), lr=self.lr)
        self._value_optim = Adam(self.value.parameters(), lr=self.lr)

    def act(self, state):
        return self.batch_act(state.unsqueeze(dim=0)).squeeze(dim=0).detach()

    def batch_act(self, states) -> torch.Tensor:
        return self.policy(states.to(self.device)).softmax(dim=-1)

    def forward(self, episodes):
        policies_loss = []
        values_loss = []
        advantages_stats = [[], []]

        for states, actions, rewards, *_ in episodes:
            # Timesteps
            t = torch.arange(states.shape[0], device=self.device)

            for states, actions, rewards, t in batch_it(
                states, actions, rewards, t, size=self.batch_size
            ):
                # Advantage estimation
                # Calculate 'return' or 'reward to go' for all trajectories
                rewards_to_go: torch.Tensor = (
                    ((self.gamma**t) * rewards).flip(0).cumsum(0).flip(0)
                )  # B

                # Baselines
                estimate_baselines: torch.Tensor = self.value(states).squeeze(
                    dim=-1
                )  # B

                # Advantage is just the difference between the return and the baseline
                advantages = rewards_to_go - estimate_baselines  # B

                if advantages.shape[0] > 1:
                    advantages = (advantages - advantages.mean()) / (
                        advantages.std(correction=0) + 1e-5
                    )
                    advantages_stats[0].append(advantages.mean().item())
                    advantages_stats[1].append(advantages.var().item())

                # Policy gradient
                probs = self.batch_act(states)

                distribution = Categorical(probs)
                log_probs: torch.Tensor = distribution.log_prob(actions.argmax(dim=-1))
                loss = -(log_probs * advantages).mean()
                policies_loss.append(loss.item())

                self._policy_optim.zero_grad()
                loss.backward(create_graph=True)
                loss.grad = None
                self._policy_optim.step()

                # Value gradient
                # Here we take the mse loss between the two
                loss = F.mse_loss(estimate_baselines, rewards_to_go)

                values_loss.append(loss.item())
                # Backpropagate
                self._value_optim.zero_grad()
                loss.backward()
                self._value_optim.step()

        self.logger.add_scalar(
            "metrics/advantages_mean", np.mean(advantages_stats[0]), self.global_step
        )
        self.logger.add_scalar(
            "metrics/advantage_var", np.mean(advantages_stats[1]), self.global_step
        )
        self.logger.add_scalar("loss/policy", np.mean(policies_loss), self.global_step)
        self.logger.add_scalar("loss/value", np.mean(values_loss), self.global_step)
