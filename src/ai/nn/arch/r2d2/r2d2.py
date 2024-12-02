from typing import Literal

from ....dataset.env.environment import Environment
from ....utils.hyperparam import HYPERPARAM, Hyperparam
from ...compat.agent import Agent
from ..dqn.policy import DQNPolicy


class R2D2(Agent):
    def __init__(
        self,
        env: Environment,
        optimizer: Literal["Adam", "RMSprop", "AdamW"] = "RMSProp",
        lr: float = 1e-3,
        gamma=0.99,
        epsilon: HYPERPARAM = Hyperparam(start=0.9, end=0.1),
        batch_size=32,
        history=4,
        prepare_episodes=4,
        reward_shaping=True,
        interactions_per_learn=1,
        # Double DQN
        target=0,
        tau=0.995,
        # DRQN
        recurrent=False,
        # Dual DQN
        duel=False,
        # PER
        per=False,
        **kwargs,
    ):
        self.prepare_episodes = prepare_episodes
        # DRQN means history is in the form of an embedding in the last hidden_state
        self.hidden_dim = 16
        history = max(history, 1)
        if len(env.preprocessed_shape) <= 1:
            history = 1

        self._train_steps = 0
        self._network_syncs = 0
        self.batch_size = batch_size
        self.reward_shaping = reward_shaping
        self.interactions_per_learn = interactions_per_learn
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
            **kwargs,
        )
