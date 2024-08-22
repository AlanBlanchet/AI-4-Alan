import torch
from ai.task.detection.mAP import mean_average_precision
from datasets import load_dataset
from pytest import fixture
from transformers import pipeline


@fixture
def dataset():
    return load_dataset("detection-datasets/coco", split="val")


@fixture
def model():
    return pipeline("object-detection", model="hustvl/yolos-small")


def test_mAP(dataset, model):
    label2id = dataset.features["objects"].feature["category"]._str2int

    item = dataset[0]
    gt_labels = torch.tensor(item["objects"]["category"])
    gt_boxes = torch.tensor(item["objects"]["bbox"])

    out = model(item["image"])

    scores = torch.tensor([o["score"] for o in out])
    labels = torch.tensor([label2id[o["label"]] for o in out])
    boxes = []
    for o in out:
        box = o["box"]
        boxes.append([box["xmin"], box["ymin"], box["xmax"], box["ymax"]])
    boxes = torch.tensor(boxes)

    metric = mean_average_precision(scores, labels, boxes, gt_labels, gt_boxes)

    print(metric)

    assert metric
