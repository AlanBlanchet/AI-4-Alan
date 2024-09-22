from inspect import _empty, signature

import torch.nn as nn
from click import Choice, argument, command, option

from ...registry import REGISTER


@command("show", help="Show a neural network model")
@argument("model", type=str)
@option(
    "--source",
    "-s",
    type=Choice(["timm", "hf", "torch"]),
    default=None,
    help="The source of the model",
)
def main(model, source: str = None):
    """
    MODEL : The name of the model to show
    """
    if source == "timm":
        from timm.models import create_model

        print(create_model(model, pretrained=False))
    elif source == "hf":
        from transformers import AutoModel

        print(AutoModel.from_pretrained(model))
    elif source == "torch":
        from torchvision.models import __dict__
        from torchvision.models.detection import __dict__ as detection_dict

        module = None
        if model in __dict__:
            module = __dict__[model]
        else:
            module = detection_dict[model]

        print(module())
    else:
        print(get_arch(model))


def get_arch(arch_name: str):
    arch_cls = REGISTER[arch_name]

    if not issubclass(arch_cls, nn.Module):
        raise ValueError(f"Model {arch_name} is not a subclass of nn.Module")

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
