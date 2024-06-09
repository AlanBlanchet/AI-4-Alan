from pytest import mark

from .hf_dataset import HuggingFaceDataset


@mark.dataset
def test_hf_coco():
    dataset = HuggingFaceDataset(name="detection-datasets/coco")

    dataset.train()
