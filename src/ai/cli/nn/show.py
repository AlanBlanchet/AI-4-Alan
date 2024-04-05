from inspect import _empty, signature

from click import Choice, argument, command

from ...utils.paths import AIPaths


@command("show", help="Show a neural network model")
@argument("model", type=Choice(list(AIPaths.get_archs())))
def main(model):
    """
    MODEL : The name of the model to show
    """
    print(get_arch(model))


@AIPaths.fn_cache
def get_arch(arch_name: str):
    from ...utils.arch import get_arch_module

    # Get the Module from the name
    arch_cls = get_arch_module(arch_name)

    # Get the __init__ function for instantiation
    arch_init_fn = getattr(arch_cls, "__init__")
    # Get the signature
    t = signature(arch_init_fn)

    # Get all arguments and give them a default value
    arg_prep = []
    # Skip the 'self' parameter
    for v in list(t.parameters.values())[1:]:
        if v.default != _empty:
            break

        annot = v.annotation

        if annot == str:
            arg_prep.append("x")
        elif annot == bool:
            arg_prep.append(True)
        else:
            arg_prep.append(1)
    return arch_cls(*arg_prep)
