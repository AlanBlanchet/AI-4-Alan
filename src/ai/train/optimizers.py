from typing import ClassVar

import torch.optim as optim

from ..configs.external_base import ExternalBase
from ..nn.compat import Module

namespace = globals()


class Optimizer(ExternalBase, optim.Optimizer, buildable=False):
    model_config = {"arbitrary_types_allowed": True}

    XT_ADD_KEYS: ClassVar[list[str]] = [
        "defaults",
        "state",
        "param_groups",
        "_warned_capturable_if_run_uncaptured",
        "_zero_grad_profile_name",
    ]

    default = "AdamW"
    auto_build = True


Optimizer.INIT_CLS = optim.Optimizer

OPTIMIZER_NAMES = Optimizer.get_module_class_names(optim)

Optimizer.create_classes(
    namespace=namespace, module=optim, required_base=optim.Optimizer
)


class LRDecay(Module, buildable=False):
    default = "ExponentialLR"
    auto_build = True


LRDECAY_NAMES = LRDecay.get_module_class_names(optim.lr_scheduler)

LRDecay.create_classes(
    namespace=namespace,
    module=optim.lr_scheduler,
    required_base=optim.lr_scheduler._LRScheduler,
)
