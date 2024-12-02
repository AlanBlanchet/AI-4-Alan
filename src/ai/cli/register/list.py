from click import command


@command("list", help="List all registers")
def main():
    from ...registry.registry import REGISTER

    print(REGISTER)
