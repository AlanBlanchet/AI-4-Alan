from click import command


@command("list", help="List all registers")
def main():
    from ...registry.registers import REGISTERS

    for registry in REGISTERS:
        print(registry)
