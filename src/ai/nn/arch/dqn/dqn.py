from copy import deepcopy

import torch
import torch.nn.functional as F

from ....registry.registry import REGISTER
from ...compat.agent import Agent
from .config import DQNConfig
from .policy import DQNPolicy


@REGISTER
class DQN(Agent):
    config: DQNConfig = DQNConfig

    def __init__(
        self,
        config: DQNConfig,
        # env: Environment,
        # optimizer: Literal["Adam", "RMSprop", "AdamW"] = "RMSProp",
        # lr: float = 1e-3,
        # gamma=0.99,
        # epsilon: HYPERPARAM = Hyperparam(0.9, 0.1),
        # batch_size=32,
        # history=4,
        # prepare_episodes=4,
        # reward_shaping=True,
        # interactions_per_learn=1,
        # # Double DQN
        # target=0,
        # tau=0.995,
        # # DRQN
        # recurrent=False,
        # # Dual DQN
        # duel=False,
        # # PER
        # per=False,
        **kwargs,
    ):
        self.prepare_episodes = config.prepare_episodes
        # DRQN means history is in the form of an embedding in the last hidden_state
        self.hidden_dim = 16

        self._train_steps = 0
        self._network_syncs = 0

        # Call parent
        super().__init__(config, **kwargs)

        if config.target != 0:
            self.trained_policy = deepcopy(self.policy)
        else:
            self.trained_policy = self.policy

        # if config.optimizer == "Adam":
        #     self._optim = optim.Adam(self.trained_policy.parameters(), lr=config.lr)
        # elif config.optimizer == "AdamW":
        #     self._optim = optim.AdamW(self.trained_policy.parameters(), lr=config.lr)
        # elif config.optimizer == "RMSProp":
        #     self._optim = optim.RMSprop(self.trained_policy.parameters(), lr=config.lr)

        self.update_target = config.target

        if config.recurrent:
            # Delayed elements just like obs -> next_obs
            self.setup_delayed(
                {"h1": "h0", "c1": "c0"}, [(config.hidden_dim,), (config.hidden_dim,)]
            )

        # self.register_state(
        #     [
        #         "_train_steps",
        #         "_network_syncs",
        #         "gamma",
        #         "update_target",
        #         "tau",
        #         "batch_size",
        #         "prepare_episodes",
        #         "per",
        #     ]
        # )

    def build_policy(self):
        policy = DQNPolicy(
            self.env.preprocessed_shape,
            self.env.out_action,
            history=self.config.history,
            last_layer="linear" if not self.config.recurrent else "lstm",
            duel=self.config.duel,
            hidden_dim=self.hidden_dim,
        )
        policy.train()
        policy = policy.requires_grad_(True)
        return policy

    def learn(self, batch: dict):
        # state = self.sample(
        #     self.config.batch_size,
        #     self.history,
        #     priority="p" if self.config.per else None,
        # )

        # obs, actions, rewards, next_obs, dones, idx = state.get_defaults("idx")
        obs, actions, rewards, dones, _, next_obs, _ = tuple(batch.values())

        h0, c0, h1, c1 = None, None, None, None
        # if self.config.recurrent:
        #     # (B, ...)
        #     h0, c0, h1, c1 = state[["h0", "c0", "h1", "c1"]]
        #     h0 = h0[..., 0, :].unsqueeze(dim=0).detach()
        #     c0 = c0[..., 0, :].unsqueeze(dim=0).detach()
        #     h1 = h1[..., 0, :].unsqueeze(dim=0).detach()
        #     c1 = c1[..., 0, :].unsqueeze(dim=0).detach()

        # (B, T, ...) -> (B, ...)
        rewards = rewards[..., -1]
        dones = dones[..., -1]
        actions = actions[..., -1, :]
        # idx = idx[..., -1]

        # Reward shaping
        if self.config.reward_shaping:
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
            expected_Q = rewards + self.config.gamma * Q_next.max(dim=-1).values * (
                1 - dones
            )

        return dict(
            Q=Q,
            expected_Q=expected_Q,
            reward=rewards,
            chosen_actions=chosen_actions,
            epsilon=float(self.epsilon),
            memory=len(self.memory),
            # idx=idx,
        )

    def compute_loss(self, out: dict, batch: dict) -> dict:
        # Q, expected_Q, idx = out["Q"], out["expected_Q"], out["idx"]
        Q, expected_Q = out["Q"], out["expected_Q"]

        loss = F.smooth_l1_loss(
            Q, expected_Q, reduction="none" if self.config.per else "mean"
        )

        # if self.config.per:
        #     self.memorize({"idx": idx, "p": loss.abs().cpu() + 1e-6})
        #     loss = loss.mean()

        # self.optimize(loss, self._optim, clip=10)

        self._train_steps += 1

        # Update target network with current network
        # Prevents network to follow a moving target
        if self.update_target and self._train_steps >= self.update_target:
            # We do a soft update to prevent sudden changes
            state_dict = self.policy.state_dict()
            train_state_dict = self.trained_policy.state_dict()
            for key in state_dict.keys():
                train_state_dict[key] = (
                    self.config.tau * state_dict[key]
                    + (1 - self.config.tau) * train_state_dict[key]
                )
            self.policy.load_state_dict(train_state_dict)
            self._train_steps = 0
            self._network_syncs += 1

        return {
            "loss": loss,
            "expected_Q": expected_Q.mean(),
            "Q": Q.mean(),
        }
