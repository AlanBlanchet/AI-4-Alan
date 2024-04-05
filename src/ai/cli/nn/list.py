from click import command


@command("list", help="List all the neural network models")
def main():
    """
    MODEL : The name of the model to show
    """
    from ...utils.paths import AIPaths

    [print(arch) for arch in AIPaths.get_archs()]
