from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from ..policy import MemoryLearner
from .general.actor import ActorNetwork
from .general.critic import CriticNetwork


@dataclass(kw_only=True)
class TD3(MemoryLearner):
    alpha: float
    gamma: float
    beta: float
    tau: float
    policy_noise: float = 0.1
    warmup: int = 1000
    batch_size: int = 32

    def __attrs_post_init__(self):
        super().__attrs_post_init__()

        self.actor = ActorNetwork(self.env, self.epsilon)
        self.critic_1 = CriticNetwork(self.env, self.epsilon)
        self.critic_2 = CriticNetwork(self.env, self.epsilon)

        self.target_actor = ActorNetwork(self.env, self.epsilon)
        self.target_critic_1 = CriticNetwork(self.env, self.epsilon)
        self.target_critic_2 = CriticNetwork(self.env, self.epsilon)

        self.a1 = optim.AdamW(self.actor.parameters(), lr=self.alpha)
        self.a2 = optim.AdamW(self.critic_1.parameters(), lr=self.beta)
        self.a3 = optim.AdamW(self.critic_2.parameters(), lr=self.beta)

        # Hard update the target networks
        self.update(1)

    def act(self, state):
        if self.learn_steps_counter < self.warmup:
            mu = np.random.normal(scale=self.policy_noise, size=self.action_shape)
        else:
            mu = self.actor(state)

        mu_prime = mu + np.random.normal(scale=self.policy_noise)

        return mu_prime.argmax()

    def learn(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

        self.replay()

    def replay(self):
        if self.memory < self.batch_size:
            return

        batch = self.memory.sample(self.batch_size)

        target_actions = self.target_actor(batch.next_states.unsqueeze(dim=-1))
        target_actions += torch.clip(
            torch.normal(torch.tensor(0.0), torch.tensor(0.2)), -0.5, 0.5
        )

        # Clip min action / max action

        q1_ = self.target_critic_1(batch.next_states, target_actions)
        q2_ = self.target_critic_2(batch.next_states, target_actions)

        q1_ = q1_.squeeze(dim=-1)
        q2_ = q2_.squeeze(dim=-1)

        q1 = self.critic_1(batch.states, batch.actions).squeeze(dim=-1)
        q2 = self.critic_2(batch.states, batch.actions).squeeze(dim=-1)

        critic_value = min(q1_, q2_)

        target = batch.rewards + self.gamma * critic_value * (
            1 - batch.dones.to(torch.int64)
        )

        critic_1_loss = F.mse_loss(q1, target)
        critic_2_loss = F.mse_loss(q2, target)

        self.a2.zero_grad()
        self.a3.zero_grad()

        critic_1_loss.backward()
        critic_2_loss.backward()

        self.a1.step()
        self.a2.step()

        if self.learn_steps_counter % self.train_freq != 0:
            return

        self.actor(batch.states)
        critic_1_value = self.critic_1(batch.states, batch.actions)

        actor_loss = -critic_1_value.mean()

        self.a1.zero_grad()
        actor_loss.backward()

        self.a1.step()

        self.update()

    def update(self, tau=None):
        if tau is None:
            tau = self.tau

        for network, target_network in (
            (self.actor, self.target_actor),
            (self.critic_1, self.target_critic_1),
            (self.critic_2, self.target_critic_2),
        ):
            for network_p, target_network_p in zip(
                network.state_dict.values(), target_network.state_dict.values()
            ):
                target_network_p.copy_(network_p * tau + target_network_p * (1 - tau))
