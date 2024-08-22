from ai.configs.ai import AI
from pytest import fixture, mark


@fixture
def config():
    return dict(
        _="HuggingFaceDataset", path="fuliucansheng/pascal_voc", name="voc2007_main"
    )


@mark.dataset
def test_hf_basic(config):
    dataset = AI.from_config(config)

    print(dataset)

    assert False


def test_hf():
    config = dict(
        _="HuggingFaceDataset", path="fuliucansheng/pascal_voc", name="voc2007_main"
    )

    # Build dataset
    AI.from_config(config)

    assert True
