from click import command


@command("index", help="Index all the neural network models")
def main():
    import ai.nn.arch  # noqa
    from ...registry.registers import MODEL

    MODEL.calculate_index()
