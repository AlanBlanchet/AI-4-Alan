from copy import deepcopy

import torch
import torch.nn.functional as F
import torch.optim as optim

from ..env.environment import Environment
from ..utils.encode import Encoder
from .agent import Agent


class DQNAgent(Agent):
    def __init__(self, env: Environment):
        super().__init__(
            env,
            Encoder(
                env.preprocessed_shape,
                env.out_action,
                stacks=1,
                image_type="grayscale",
            ),
            epsilon=0.1,
        )

        self.target_policy = deepcopy(self.policy)

        self._train_steps = 0
        self._network_syncs = 0
        self.gamma = 0.99
        self.update_target = 10
        self.tau = 0.995

        self._optim = optim.Adam(self.policy.parameters(), lr=1e-3)
        self.register_state(
            ["_train_steps", "_network_syncs", "gamma", "update_target", "tau"]
        )

    def prepare(self):
        # Require at least x episodes
        self.episode_interact(4)

    def learn(self):
        self.interact()

        for trajectory in self.trajectories(1):
            obs, actions, rewards, next_obs, dones = trajectory.get_defaults()

            # Get chosen actions at states
            chosen_actions = actions.argmax(dim=-1)

            # Calculate Q values for state / next state
            Q = (
                self.policy(obs)
                .gather(1, chosen_actions.unsqueeze(dim=-1))
                .squeeze(dim=-1)
            )

            # The next rewards are not learnable, they are our targets
            with torch.no_grad():
                Q_next = self.target_policy(next_obs)
                # Long term reward function
                expected_Q = rewards + self.gamma * Q_next.max(dim=-1).values * (
                    1 - dones
                )

            loss = F.smooth_l1_loss(Q, expected_Q)

            self.optimize(loss, self._optim, clip=100)

            self._train_steps += 1

            # Update target network with current network
            # Prevents network to follow a moving target
            if self._train_steps >= self.update_target:
                # We do a soft update to prevent sudden changes
                state_dict = self.policy.state_dict()
                target_state_dict = self.target_policy.state_dict()
                for key in state_dict.keys():
                    target_state_dict[key] = (
                        self.tau * state_dict[key]
                        + (1 - self.tau) * target_state_dict[key]
                    )
                self.target_policy.load_state_dict(target_state_dict)
                self._train_steps = 0
                self._network_syncs += 1

        return {
            "loss": loss.item(),
            "expected_Q": expected_Q.mean().item(),
            "Q": Q.mean().item(),
        }
