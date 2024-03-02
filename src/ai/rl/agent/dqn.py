from copy import deepcopy
from typing import Literal

import torch
import torch.nn.functional as F
import torch.optim as optim

from ..env.environment import Environment
from ..policy import DQNPolicy
from ..utils.hyperparam import HYPERPARAM, Hyperparam
from .agent import Agent


class DQNAgent(Agent):
    def __init__(
        self,
        env: Environment,
        optimizer: Literal["Adam", "RMSprop", "AdamW"] = "RMSProp",
        lr: float = 1e-3,
        gamma=0.99,
        epsilon: HYPERPARAM = Hyperparam(0.9, 0.1),
        batch_size=32,
        history=4,
        prepare_episodes=4,
        # Double DQN
        target=0,
        tau=0.995,
        # DRQN
        recurrent=False,
        # Dual DQN
        duel=False,
        # PER
        per=False,
    ):
        self.prepare_episodes = prepare_episodes
        # DRQN means history is in the form of an embedding in the last hidden_state
        self.hidden_dim = 16
        history = max(history, 1)

        self._train_steps = 0
        self._network_syncs = 0
        self.batch_size = batch_size
        self.gamma = gamma
        self.update_target = target
        self.tau = tau
        self.per = per
        self.recurrent = recurrent

        # Call parent
        super().__init__(
            env,
            DQNPolicy(
                env.preprocessed_shape,
                env.out_action,
                history=history,
                last_layer="linear" if not recurrent else "lstm",
                duel=duel,
                hidden_dim=self.hidden_dim,
            ),
            epsilon=epsilon,
            history=1 if recurrent else history,
            requires_merge=not recurrent,
        )

        if target != 0:
            self.trained_policy = deepcopy(self.policy)
        else:
            self.trained_policy = self.policy

        if optimizer == "Adam":
            self._optim = optim.Adam(self.trained_policy.parameters(), lr=lr)
        elif optimizer == "AdamW":
            self._optim = optim.AdamW(self.trained_policy.parameters(), lr=lr)
        elif optimizer == "RMSProp":
            self._optim = optim.RMSprop(self.trained_policy.parameters(), lr=lr)

        if self.recurrent:
            # Delayed elements just like obs -> next_obs
            self.setup_delayed(
                {"h1": "h0", "c1": "c0"}, [(self.hidden_dim,), (self.hidden_dim,)]
            )

        self.register_state(
            [
                "_train_steps",
                "_network_syncs",
                "gamma",
                "update_target",
                "tau",
                "batch_size",
                "prepare_episodes",
                "per",
            ]
        )

    def prepare(self):
        # Require at least x episodes
        for data in self.episode_interact(self.prepare_episodes):
            yield data

    def learn(self):
        self.interact()

        state = self.sample(
            self.batch_size, self.history, priority="p" if self.per else None
        )

        obs, actions, rewards, next_obs, dones, idx = state.get_defaults("idx")

        h0, c0, h1, c1 = None, None, None, None
        if self.recurrent:
            # (B, ...)
            h0, c0, h1, c1 = state[["h0", "c0", "h1", "c1"]]

        # (B, T, ...) -> (B, ...)
        rewards = rewards[..., -1]
        dones = dones[..., -1]
        actions = actions[..., -1, :]
        idx = idx[..., -1]

        # Reward shaping
        rewards = torch.where(rewards > 0, 1, rewards)
        rewards = torch.where(rewards < 0, -1, rewards)

        # Get chosen actions at states
        chosen_actions = actions.argmax(dim=-1)

        # Calculate Q values for state / next state
        Q, h0, c0 = self.trained_policy(obs, h=h0, c=c0)
        Q = Q.gather(1, chosen_actions.unsqueeze(dim=-1)).squeeze(dim=-1)

        # The next rewards are not learnable, they are our targets
        with torch.no_grad():
            Q_next, h1, c1 = self.policy(next_obs, h=h1, c=c1)
            # Long term reward function
            expected_Q = rewards + self.gamma * Q_next.max(dim=-1).values * (1 - dones)

        loss = F.smooth_l1_loss(Q, expected_Q, reduction="none" if self.per else "mean")

        if self.per:
            self.memorize({"idx": idx, "p": loss.abs().cpu() + 1e-6})
            loss = loss.mean()

        self.optimize(loss, self._optim, clip=10)

        self._train_steps += 1

        # Update target network with current network
        # Prevents network to follow a moving target
        if self.update_target and self._train_steps >= self.update_target:
            # We do a soft update to prevent sudden changes
            state_dict = self.policy.state_dict()
            train_state_dict = self.trained_policy.state_dict()
            for key in state_dict.keys():
                train_state_dict[key] = (
                    self.tau * state_dict[key] + (1 - self.tau) * train_state_dict[key]
                )
            self.policy.load_state_dict(train_state_dict)
            self._train_steps = 0
            self._network_syncs += 1

        return {
            "loss": loss.item(),
            "expected_Q": expected_Q.mean().item(),
            "Q": Q.mean().item(),
        }
