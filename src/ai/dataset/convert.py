from typing import Any

import torch
from albumentations import (
    BboxParams,
    Compose,
    GaussNoise,
    MotionBlur,
    RandomBrightnessContrast,
    RandomGamma,
    Resize,
    ShiftScaleRotate,
)
from einops import rearrange
from pydantic import BaseModel


class TorchDataset(BaseModel):
    dataset: Any
    transforms: Compose = Compose(
        [
            ShiftScaleRotate(p=0.5),
            RandomBrightnessContrast(p=0.5),
            MotionBlur(p=0.5),
            GaussNoise(p=0.5),
            RandomGamma(p=0.5),
            Resize(300, 300),
        ],
        bbox_params=BboxParams(
            format="pascal_voc",  # or 'coco', 'yolo', etc.
            min_area=0,
            min_visibility=0,
            label_fields=["category_ids"],
        ),
    )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        objects = item["objects"]
        item.pop("classes")
        item.pop("width")
        item.pop("height")
        objects.pop("difficult")

        image = item["image"]
        image = rearrange(image.clone(), "... c h w -> ... h w c")
        bboxes = objects["bboxes"].clone()

        image_npy = image.numpy()
        out = self.transforms(
            image=image_npy, bboxes=bboxes, category_ids=objects["classes"]
        )
        image = torch.from_numpy(out["image"])
        bboxes = torch.tensor(out["bboxes"])

        if bboxes.shape[0] == 0:
            bboxes = torch.zeros((0, 4))

        if image.shape[-1] != 3:
            image = image.expand(-1, -1, 3)

        assert image.shape == (300, 300, 3), f"Image shape is {image.shape}"
        assert bboxes.shape == (bboxes.shape[0], 4), f"Bboxes shape is {bboxes.shape}"

        if bboxes.shape[0] != 0:
            # Scale bboxes
            bboxes[..., [0, 2]] /= image.shape[-2]
            bboxes[..., [1, 3]] /= image.shape[-3]

        image = image.float() / 255
        item["image"] = rearrange(image, "... h w c -> ... c h w")
        objects["bboxes"] = bboxes

        max_bbs = 50
        # Uncollatable tensors
        for k, v in objects.items():
            N, *R = v.shape
            if k == "classes":
                vv = torch.full((max_bbs,), 0, *R)
            else:
                vv = torch.zeros(max_bbs, *R)
            vv[: min(max_bbs, N)] = v
            objects[k] = vv

        objects["mask"] = torch.zeros(max_bbs, dtype=torch.bool).where(
            torch.arange(max_bbs) >= bboxes.shape[0], True
        )

        return item

    class Config:
        arbitrary_types_allowed = True
