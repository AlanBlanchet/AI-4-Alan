from click import command


@command("index", help="Create indexes for the registers")
def main():
    from ...registry import REGISTER

    for registry in [REGISTER]:
        registry.calculate_index()
        print(registry)
