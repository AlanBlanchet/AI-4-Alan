from copy import deepcopy

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from einops import rearrange
from torch._tensor import Tensor
from torch.optim import Adam

from ..collect.collector import Collector
from ..policy import Policy
from ..utils.encode import IMAGE_TYPE, Encoder


class DRQN(Policy):
    def __init__(
        self,
        collector: Collector,
        batch_size: int = 32,
        lr: float = 1e-4,
        gamma: float = 0.95,
        tau: float = 5e-3,
        update_target: int = None,
        image_type: IMAGE_TYPE = "rgb",
        # Sequence length isn't the same as stacks from DQN !
        sequence_length: int = 4,
    ):
        super().__init__(collector)

        self.lr = self.register_state("lr", lr)
        self.gamma = self.register_state("gamma", gamma)
        self.tau = self.register_state("tau", tau)
        self.batch_size = self.register_state("batch_size", batch_size)
        self.update_target = self.register_state(
            "update_target", update_target or batch_size
        )
        self.sequence_length = self.register_state("sequence_length", sequence_length)
        self.image_type = self.register_state("image_type", image_type)

        self.network = Encoder(
            self.in_state, self.out_actions, last_layer="lstm", image_type=image_type
        )

        self.target_network = deepcopy(self.network)

        self._train_steps = self.register_state("_train_steps", 0)
        self._network_syncs = self.register_state("_network_syncs", 0)

        # self.zero_state = torch.zeros(, batch_size, 512).to(self.device)

        # Adam optimizers
        self._optim = Adam(self.network.parameters(), lr=self.lr)

    def setup(self):
        self.collector.set_mode("keep")
        self.collector.fill(1000)

    def preprocess_state(self, state: torch.Tensor):
        if state.ndim < 3:
            assert self.sequence_length == 1, "Stacks must be 1 if state is 1D"
            return state
        elif state.ndim == 3:
            state = (state.permute(-1, -3, -2).float() / 255.0).unsqueeze(dim=0)
        else:
            state = state.permute(0, -1, -3, -2).float() / 255.0

        if state.shape[1] == 3 and self.image_type == "grayscale":
            state = TF.rgb_to_grayscale(state)
        if state.shape[0] == 1:
            return state
        return rearrange(state, "(b s) c h w -> b s c h w", s=self.sequence_length)

    def preprocess(self, batch: list[torch.Tensor]):
        # (B*S, C, H, W)
        states, actions, rewards, next_states, dones = batch

        # Correct channel layout and normalized
        # (B, S, C, H, W)
        states = self.preprocess_state(states)
        next_states = self.preprocess_state(next_states)

        # Make (B, S, N)
        actions = rearrange(actions, "(b s) n -> b s n", s=self.sequence_length)
        pattern = "(b s) -> b s"
        rewards = rearrange(rewards, pattern, s=self.sequence_length)
        dones = rearrange(dones, pattern, s=self.sequence_length)

        # TODO Better way to create masks
        masks = ~(actions == 0).all(dim=-1)

        return states, actions, rewards, next_states, dones, masks

    def act(self, state):
        # State should be stacked (B*S, C, H, W)
        return self.batch_act(state).squeeze(dim=0).detach()

    def batch_act(self, states) -> torch.Tensor:
        states = self.preprocess_state(states)
        x = self.network(states.to(self.device))
        return x.softmax(dim=-1)

    def step(self):
        # Sample batch_size steps inside an episode
        # We can't be sure to get a full batch_size of steps !
        return self.collector.sample(
            self.batch_size,
            self.device,
            stacks=self.sequence_length,
            rand_type="episode",
        )

    def forward(self, batch) -> Tensor:
        states, actions, rewards, next_states, dones, masks = self.preprocess(batch)

        # Get chosen actions at states
        chosen_actions = actions.argmax(dim=-1)

        # Calculate Q values for state / next state
        Q = (
            self.network(states)
            .gather(1, chosen_actions.unsqueeze(dim=-1))
            .squeeze(dim=-1)
        )

        # The next rewards are not learnable, they are our targets
        with torch.no_grad():
            Q_next = self.target_network(next_states)
            # Long term reward function
            expected_Q = rewards + self.gamma * Q_next.max(dim=-1).values * (1 - dones)
            self.logger.add_scalar(
                "policy/expected_Q", expected_Q.mean(), self.global_step
            )

        loss = F.smooth_l1_loss(Q, expected_Q)

        self._optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.network.parameters(), 100)
        self._optim.step()

        self.logger.add_scalar("policy/loss", loss, self.global_step)

        self._train_steps += self.batch_size

        # Update target network with current network
        # Prevents network to follow a moving target
        if self._train_steps >= self.update_target:
            # We do a soft update to prevent sudden changes
            state_dict = self.network.state_dict()
            target_state_dict = self.target_network.state_dict()
            for key in state_dict.keys():
                target_state_dict[key] = (
                    self.tau * state_dict[key] + (1 - self.tau) * target_state_dict[key]
                )
            self.target_network.load_state_dict(target_state_dict)
            self._train_steps = 0
            self._network_syncs += 1
            self.logger.add_scalar(
                "policy/network_sync", self._network_syncs, self.global_step
            )

        self.collector.collect_steps(self.batch_size // 4)
