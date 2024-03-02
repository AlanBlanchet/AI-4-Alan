from click import Choice, argument, command, pass_context


@command(
    context_settings=dict(
        ignore_unknown_options=True,
        allow_extra_args=True,
    )
)
@argument("env", type=str)
@argument("agent", type=Choice(["dqn"]))
@pass_context
def main(ctx, env: str, agent: str):
    """
    Train a reinforcement learning agent

    ENV is the name of the environment to train the agent in.

    AGENT is the agent to train
    """
    from ...rl import Environment, Trainer
    from ...rl.agent import Agent

    def agent_from_name(name: str, env: Environment) -> Agent:
        if name == "dqn":
            from ...rl import DQNAgent

            return DQNAgent(env)
        else:
            raise ValueError(f"Unknown agent: {name}")

    kwargs = dict()
    for item in ctx.args:
        kwargs.update([item.split("=")])

    env = Environment(env, **kwargs)
    agent = agent_from_name(agent, env)

    trainer = Trainer(agent, **kwargs)

    trainer.start(**kwargs)
