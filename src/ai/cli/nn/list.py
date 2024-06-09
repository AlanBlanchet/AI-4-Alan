from click import Choice, command, option


@command("list", help="List all the neural network models")
@option(
    "--source",
    "-s",
    type=Choice(["timm", "hf", "torch"]),
    default=None,
    help="The source of the model",
)
def main(source: str = None):
    """
    MODEL : The name of the model to show
    """
    if source == "timm":
        from timm.models import list_models

        print(list_models())
    elif source == "hf":
        raise NotImplementedError
    elif source == "torch":
        from torchvision.models import list_models

        print(list_models())
    else:
        from ...registry.registers import MODEL

        print(MODEL)
