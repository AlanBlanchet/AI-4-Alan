from click_extra import extra_command

from ..modality.image.modality import Image
from ..nn.arch.resnet.resnet import ResNet18

# Just for testing


@extra_command(
    params=None,
    context_settings=dict(
        ignore_unknown_options=True,
    ),
)
# @argument("config", type=CPath(exists=True, path_type=Path))
def main():
    """Launch a task from a config file in debug mode (no MP, small datasets etc...)."""
    # ex: ResNet = Base.from_config(yaml.safe_load(config.read_text()))
    # m = ex.model
    # print(m)

    model = ResNet18().pretrained()
    # print(model)
    # model.fit()

    image = Image(
        "https://static.vecteezy.com/system/resources/thumbnails/002/098/203/small_2x/silver-tabby-cat-sitting-on-green-background-free-photo.jpg"
    )
    print(image.shape)

    out = model.classify(image)
    print(image.shape)
    print(out)
    image.show()

    # model.classify(image)

    # ResNet18(
    #     input_proj=ConvBlock(3, 64, kernel_size=7, stride=2, padding=3)
    #     first_layer_channels=64,
    #     maxpool=MaxPool2d(kernel_size=3, stride=2, padding=1),
    #     layers= # The whole module list modules printed...,
    #     out_dim=512
    # )


if __name__ == "__main__":
    main()
