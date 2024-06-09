from typing import AnyStr, Literal

from ....registry.registers import MODEL
from ..compat import ISSD
from .configs import configs
from .vgg import VGG


@MODEL.register
class VGG_A(VGG):
    def __init__(self, in_channels=3):
        super().__init__(configs["A"], in_channels)


@MODEL.register
class VGG11(VGG_A): ...


@MODEL.register
class VGG_A_LRN(VGG):
    def __init__(self, in_channels=3):
        super().__init__(configs["A-LRN"], in_channels)


@MODEL.register
class VGG11_LRN(VGG_A_LRN): ...


@MODEL.register
class VGG_B(VGG, ISSD):
    def __init__(self, in_channels=3):
        super().__init__(configs["B"], in_channels)

    def ssd_compat(self):
        return dict(features=self.features[:11])


@MODEL.register
class VGG13(VGG_B): ...


@MODEL.register
class VGG_C(VGG, ISSD):
    def __init__(self, in_channels=3):
        super().__init__(configs["C"], in_channels)

    def ssd_compat(self) -> dict[Literal["features"] | AnyStr, list[int]]:
        return dict(features=self.features[:13])


@MODEL.register
class VGG16(VGG_C): ...


@MODEL.register
class VGG_D(VGG):
    def __init__(self, in_channels=3, size=224):
        super().__init__(configs["D"], in_channels, size)


@MODEL.register
class VGG16_3(VGG_D): ...


@MODEL.register
class VGG_E(VGG):
    def __init__(self, in_channels=3, size=224):
        super().__init__(configs["E"], in_channels, size)


@MODEL.register
class VGG19(VGG_E): ...
