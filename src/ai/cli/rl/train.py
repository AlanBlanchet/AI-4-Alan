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

    kwargs = parse_extras(ctx)

    def agent_from_name(name: str, env: Environment) -> Agent:
        if name == "dqn":
            from ...rl import DQNAgent

            return DQNAgent(env, **kwargs)
        else:
            raise ValueError(f"Unknown agent: {name}")

    save_name = f"{env}-{agent}"
    env = Environment(env, **kwargs)
    agent = agent_from_name(agent, env)

    trainer = Trainer(agent, **kwargs)

    trainer.start(**kwargs)

    saved_path = trainer.save(save_name)
    print(f"Training complete and saved in {saved_path}")


def parse_extras(ctx):
    kwargs = dict()
    skip_next = False
    for i, item in enumerate(ctx.args):
        if skip_next:
            skip_next = False
            continue
        split = item.split("=")
        k, v = None, None
        if len(split) != 2:
            k = item.removeprefix("--")
            if not (i + 1 >= len(ctx.args) or ctx.args[i + 1].startswith("--")):
                v = ctx.args[i + 1]
                skip_next = True
            else:
                v = True
        else:
            k, v = split
            k = k.removeprefix("--")
        if isinstance(v, str):
            try:
                v = float(v)
                v = int(v) if int(v) == v else v
            except Exception as _:
                ...
        kwargs.update([(k, v)])
    return kwargs
