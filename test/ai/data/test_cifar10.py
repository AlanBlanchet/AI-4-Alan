from ai.data.dataset import Data


def test_from_str():
    dataset = Data.from_identifier("cifar10")

    assert dataset is not None
    assert dataset.__class__.__name__ == "Cifar10"


# def test_cifar10_image():
#     dataset = Data.from_identifier("cifar10")

#     example = dataset.example

#     assert isinstance(example, Sample)

#     image = example.inputs["image"]
#     labels = example.targets["labels"]

#     assert isinstance(image, Image)
#     assert image.shape == (3, 32, 32)

#     assert isinstance(example.targets["labels"], Image)
#     assert example.targets["labels"].shape == (10,)
