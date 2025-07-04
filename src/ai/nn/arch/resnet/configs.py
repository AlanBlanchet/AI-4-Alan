from ....configs.base import Base


class Block(Base):
    out_channels: int
    kernel_size: int


class Layer(Base, list[Block]):
    num: int

    @property
    def out_channels(self):
        return self[-1].out_channels

    @property
    def blocks_out_channels(self):
        return [block.out_channels for block in self]

    @property
    def blocks_kernel_size(self):
        return [block.kernel_size for block in self]


#     pretrained_recommendations = PretrainedConfig.create_recommendations(
#         "IMAGENET", variants=variants
#     )
