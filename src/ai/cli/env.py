import click
from click import argument, command


@command("env")
@argument("name", type=click.STRING)
def main(name: str):
    from ..nn.rl.env.env import Env

    env = Env.load(name)

    env.play()
