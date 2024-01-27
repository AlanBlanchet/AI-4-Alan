"""
This module implements the Proximal Policy Optimization (PPO) algorithm, as well as the Trust Region Policy Optimization (TRPO) and Clipped Surrogate Objective algorithms.

PPO (Proximal Policy Optimization) is a policy optimization algorithm that aims to improve the stability and sample efficiency of policy gradient methods. It uses a surrogate objective function that includes a clipping mechanism to prevent large policy updates.

TRPO (Trust Region Policy Optimization) is another policy optimization algorithm that also aims to improve stability and sample efficiency. It constrains the policy updates to a trust region, ensuring that the policy changes are not too large.

The Clipped Surrogate Objective algorithm is a variant of PPO that uses a clipping mechanism to limit the policy updates, similar to TRPO.

For more details, refer to the following papers:
- PPO: https://arxiv.org/pdf/1707.06347.pdf
- TRPO: https://arxiv.org/pdf/1502.05477.pdf
- Clipped Surrogate Objective: https://arxiv.org/pdf/1707.06347.pdf
"""
import torch
import torch.nn as nn
from torch.optim import Adam

from ..policy import MemoryLearner


class PPO(MemoryLearner):
    # model_class: Callable[[int, int], nn.Module]

    def __init__(
        self,
        learn_steps_counter: int = 0,
        memory_size: int = 10_000,
        train_batch_size: int = 32,
        train_freq: int = 16,
        train_when_done: bool = False,
        random_replay: bool = False,
        needed_memory_size: int = 32,
        lr: float = 1e-4,
        clip: float = 0.2,
        # How much to discount future rewards.
        # How much should we car about first rewards ?
        gamma: float = 0.95,
        train_freq_epochs: int = 5,
        trace_decay: float = 0.97,
        agent: nn.Module = None,
    ):
        super().__init__(
            learn_steps_counter,
            memory_size,
            train_batch_size,
            train_freq,
            train_when_done,
            random_replay,
            needed_memory_size,
        )

        self.lr = lr
        self.clip = clip
        self.gamma = gamma
        self.train_freq_epochs = train_freq_epochs
        self.trace_decay = trace_decay
        self.agent = agent

        # if self.agent is None:
        #     self.agent = ActorCritic(self.in_state, self.out_action, 128)
        self.policy = nn.Linear(self.in_state, self.out_action)
        self.value = nn.Linear(self.in_state, 1)

        self._critic_optim = Adam(self.agent.actor.parameters(), lr=self.lr)
        self._actor_optim = Adam(self.agent.critic.parameters(), lr=self.lr)

    def act(self, state):
        return self.batch_act(state.unsqueeze(dim=0)).squeeze(dim=0)

    def batch_act(self, states):
        policy, _ = self.agent(states.to(self.device))
        return policy.sample()

    def learn(self, state, action, r, state_next, done):
        policy, value = self.agent(state.unsqueeze(dim=0).to(self.device))
        log_prob = policy.log_prob(action).item()

        return super().learn(
            state,
            action,
            r,
            state_next,
            done,
            log_prob,
            value.item(),
        )

    def replay(self, batch):
        (
            states,
            actions,
            rewards,
            next_states,
            dones,
            old_log_probs,
            values,
        ) = batch  # (B, ...) for every item in tuple

        # Calculate the advantage
        k = torch.linspace(0, 1, steps=states.shape[0], device=self.device)
        discounted_rewards = ((self.gamma**k) * rewards + k).sum()

        # # No bootstrapping needed for next value here as only updated at end of an episode
        # advantages = []
        # rewards_to_go = []
        # next_values = []
        # reward_to_go, advantage, next_value = torch.zeros(3, 1, device=self.device)

        # with torch.no_grad():
        #     for idx in reversed(range(len(batch))):
        #         reward_to_go = rewards[idx] + (1 - dones[idx].int()) * (
        #             self.gamma * reward_to_go
        #         )
        #         td_error = (
        #             rewards[idx]
        #             + (1 - dones[idx].int()) * self.gamma * next_value
        #             - values[idx]
        #         )
        #         advantage = (
        #             td_error
        #             + (1 - dones[idx].int()) * self.gamma * self.trace_decay * advantage
        #         )
        #         next_value = values[idx]

        #         rewards_to_go.append(reward_to_go)
        #         advantages.append(advantage)
        #         next_values.append(next_value)

        # rewards_to_go = torch.cat(rewards_to_go)
        # advantages = torch.stack(advantages, dim=0)
        # next_values = torch.stack(next_values)

        # # Normalise
        # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)

        # for epoch in range(self.train_freq_epochs):
        #     # Recalculate outputs for subsequent iterations
        #     if epoch > 0:
        #         policy, values = self.agent(states)
        #         log_probs = policy.log_prob(actions)

        #         # Update the policy by maximising the PPO-Clip objective
        #         policy_ratio = (log_probs - old_log_probs).exp()
        #         policy_loss = -torch.min(
        #             policy_ratio * advantages,
        #             torch.clamp(policy_ratio, min=1 - self.clip, max=1 + self.clip)
        #             * advantages,
        #         ).mean()
        #         self._actor_optim.zero_grad()
        #         policy_loss.backward()
        #         self._actor_optim.step()

        # # Fit value function by regression on mean-squared error
        # for _ in range(4):
        #     value_loss = (values - rewards_to_go).pow(2).mean()
        #     self._critic_optim.zero_grad()
        #     value_loss.backward()
        #     self._critic_optim.step()
        #     values = self.agent(states)[1]
