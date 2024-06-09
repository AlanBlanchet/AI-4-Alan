from click import command


@command()
def main():
    from pytest import main

    main(["-m", "dataset"])
