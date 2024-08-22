from click import command


@command("index", help="Create indexes for the registers")
def main():
    from ...registry.registers import REGISTERS

    for registry in REGISTERS:
        registry.calculate_index()
        print(registry)
