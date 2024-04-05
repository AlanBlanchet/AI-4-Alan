from click import Choice, argument, command

from ...utils.paths import AIPaths


@command("show", help="Train a neural network model")
@argument("model", type=Choice(list(AIPaths.get_archs())))
@argument("dataset", type=str)
def main(model, dataset):
    """
    MODEL : The name of the model to show
    """
    from ...train.trainer import AITrainer

    trainer = AITrainer(model, dataset)
    trainer.fit()
