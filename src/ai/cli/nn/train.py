from click import Choice, argument, command

from ...registry import REGISTER


@command("show", help="Train a neural network model")
@argument("model", type=Choice(REGISTER.names))
@argument("dataset", type=str)
def main(model, dataset):
    """
    MODEL : The name of the model to show
    """
    from ...train.runner import Runner

    trainer = Runner(model, dataset)
    trainer.run()
