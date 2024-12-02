import torch.nn as nn


class SumSequential(nn.Sequential):
    def __init__(self, *modules):
        super().__init__(*modules)

    def __repr__(self):
        list_of_reprs = [repr(item) for item in self]
        if len(list_of_reprs) == 0:
            return "SumSequential()"
        return f"{' + '.join(list_of_reprs)}"
