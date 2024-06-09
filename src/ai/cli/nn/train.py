from click import Choice, argument, command

from ...registry.registers import MODEL


@command("show", help="Train a neural network model")
@argument("model", type=Choice(MODEL.names))
@argument("dataset", type=str)
def main(model, dataset):
    """
    MODEL : The name of the model to show
    """
    from ...train.trainer import AITrainer

    trainer = AITrainer(model, dataset)
    trainer.fit()
