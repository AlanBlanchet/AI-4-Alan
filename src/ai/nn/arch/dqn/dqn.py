# from copy import deepcopy
# from typing import override

# import torch
# import torch.nn.functional as F

# from ...compat.agent import Agent
# from .policy import DQNPolicy


# class DQN(Agent):
#     # Double DQN
#     update_target: int = 0
#     """DDQN update target step frequency update"""
#     tau: float = 0.995
#     """DDQN soft update ratio to the target network"""
#     ddqn_tuned: bool = False
#     """DDQN tuned version with shared action layer"""
#     # DRQN
#     recurrent: bool = False
#     """DQN with LSTM"""
#     lstm_hidden_dim: int = 16
#     """LSTMs hidden dimension"""
#     # Dual DQN
#     dual: bool = False
#     """DQN with two Q networks"""
#     # PER
#     per: bool = False
#     """Prioritized Experience Replay"""

#     _train_steps = 0
#     _network_syncs = 0

#     def init(self):
#         super().init()

#         if self.update_target != 0:
#             self.info(
#                 f"Using Double DQN with updates every {self.update_target} steps and tau={self.tau}"
#             )
#             self.online_policy = deepcopy(self.policy)
#             self.policy.requires_grad_(False)

#             if self.ddqn_tuned:
#                 self.info("Using the tuned version of Double DQN (shared action layer)")
#                 self.online_policy.l1 = self.policy.l1
#                 self.policy.l1.requires_grad_(True)
#         else:
#             self.online_policy = self.policy

#         if self.recurrent:
#             self.info(f"Using DRQN with hidden_dim={self.lstm_hidden_dim}")

#         # if self.recurrent:
#         #     # Delayed elements just like obs -> next_obs
#         #     self.setup_delayed(
#         #         {"h1": "h0", "c1": "c0"}, [(self.hidden_dim,), (self.hidden_dim,)]
#         #     )

#     def setup_policy(self):
#         return DQNPolicy(
#             in_shape=self.active_env._preprocessed_shape,
#             out_dim=self.active_env.out_action,
#             history=self.history,
#             last_layer="linear" if not self.recurrent else "lstm",
#             dual=self.dual,
#             hidden_dim=self.lstm_hidden_dim,
#         )

#     @property
#     def target_policy(self):
#         return self.policy

#     def learn(self, batch: dict):
#         obs, actions, rewards, dones, _, next_obs, *_ = tuple(batch.values())

#         hx0, hx1 = None, None

#         # (B, T, ...) -> (B, ...)
#         rewards = rewards[..., -1]
#         dones = dones[..., -1]
#         actions = actions[..., -1, :]

#         # Reward shaping
#         if self.reward_shaping:
#             rewards = torch.where(rewards > 0, 1, rewards)
#             rewards = torch.where(rewards < 0, -1, rewards)

#         # Get chosen actions at states
#         chosen_actions = actions.argmax(dim=-1)

#         # Calculate Q values for state / next state
#         Q, hx0 = self.online_policy(obs, hx=hx0)
#         Q = Q.gather(1, chosen_actions.unsqueeze(dim=-1)).squeeze(dim=-1)

#         # The next rewards are not learnable, they are our targets
#         with torch.no_grad():
#             Q_next, hx1 = self.target_policy(next_obs, hx1)
#             # Long term reward
#             expected_Q = rewards + self.gamma * Q_next.max(dim=-1).values * (1 - dones)

#         return dict(
#             Q=Q,
#             expected_Q=expected_Q,
#             reward=rewards,
#             chosen_actions=chosen_actions,
#             epsilon=float(self.epsilon),
#             **self.env_info(),
#         )

#     def compute_loss(self, out: dict, batch: dict) -> dict:
#         Q, expected_Q = out["Q"], out["expected_Q"]

#         loss = F.smooth_l1_loss(Q, expected_Q, reduction="none" if self.per else "mean")

#         # if self.config.per:
#         #     self.memorize({"idx": idx, "p": loss.abs().cpu() + 1e-6})
#         #     loss = loss.mean()

#         # self.optimize(loss, self._optim, clip=10)

#         self._train_steps += 1

#         # Update target network with current network
#         # Prevents network to follow a moving target
#         if self.update_target and self._train_steps >= self.update_target:
#             # We do a soft update to prevent sudden changes
#             train_state_dict = self.online_policy.state_dict()
#             target_state_dict = self.target_policy.state_dict(keep_vars=True)
#             new_state_dict = {}
#             for key, tensor in target_state_dict.items():
#                 # Do not mutate trainable params since it will detach the graph and error in backprop
#                 if not tensor.requires_grad:
#                     new_state_dict[key] = (
#                         self.tau * train_state_dict[key] + (1 - self.tau) * tensor
#                     )
#             self.target_policy.load_state_dict(new_state_dict, strict=False)
#             self._train_steps = 0
#             self._network_syncs += 1

#         return dict(
#             loss=loss,
#             expected_Q=expected_Q.mean(),
#             Q=Q.mean(),
#         )


# class DRQN(DQN):
#     recurrent: bool = True

#     @override
#     def setup_policy(self):
#         return DQNPolicy(
#             in_shape=self.active_env._preprocessed_shape,
#             out_dim=self.active_env.out_action,
#             history=self.history,
#             last_layer="linear" if not self.recurrent else "lstm",
#             duel=self.duel,
#             hidden_dim=self.lstm_hidden_dim,
#             dims=[32, 64],
#         )


# class DQNHardUpdate(DQN, buildable=False):
#     tau: float = 1.0


# class DDQN(DQNHardUpdate):
#     update_target: int = 10_000


# class DualDQN(DQNHardUpdate):
#     dual: bool = True
