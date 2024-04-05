from .configs import configs
from .vgg import VGG as _VGG


class VGG_A(_VGG):
    def __init__(self, in_channels=3, size=224):
        super().__init__(configs["A"], in_channels, size)


class VGG11(VGG_A): ...


class VGG_A_LRN(_VGG):
    def __init__(self, in_channels=3, size=224):
        super().__init__(configs["A-LRN"], in_channels, size)


class VGG11_LRN(VGG_A_LRN): ...


class VGG_B(_VGG):
    def __init__(self, in_channels=3, size=224):
        super().__init__(configs["B"], in_channels, size)


class VGG13(VGG_B): ...


class VGG_C(_VGG):
    def __init__(self, in_channels=3, size=224):
        super().__init__(configs["C"], in_channels, size)


class VGG16(VGG_C): ...


class VGG_D(_VGG):
    def __init__(self, in_channels=3, size=224):
        super().__init__(configs["D"], in_channels, size)


class VGG16_3(VGG_D): ...


class VGG_E(_VGG):
    def __init__(self, in_channels=3, size=224):
        super().__init__(configs["E"], in_channels, size)


class VGG19(VGG_E): ...


class VGG(VGG_C): ...
