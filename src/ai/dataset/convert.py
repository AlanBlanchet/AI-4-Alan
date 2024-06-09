import torch
from albumentations import Resize
from einops import rearrange

from .base_dataset import BaseTorchDataset


class HuggingFaceTorchDataset(BaseTorchDataset):
    def __init__(self, dataset, background_id=0):
        self.dataset = dataset.with_format("pt")
        self.background_id = background_id

        self.transforms = Resize(300, 300)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"]
        if image.shape[-3] != 3:
            image = image.expand(3, -1, -1)
        image = rearrange(image, "... c h w -> ... h w c")
        objects = item["objects"]
        bboxes = objects["bbox"]
        objects["category"] += 1

        # Scale bboxes
        bboxes[..., [0, 2]] /= image.shape[-2]
        bboxes[..., [1, 3]] /= image.shape[-3]

        image_npy = image.numpy()
        out = self.transforms(image=image_npy)
        image = torch.from_numpy(out["image"])

        item["image"] = rearrange(image.float(), "... h w c -> ... c h w")
        objects["bbox"] = bboxes

        max_bbs = 100
        # Uncollatable tensors
        for k, v in objects.items():
            N, *R = v.shape
            if k == "category":
                vv = torch.full((max_bbs,), self.background_id, *R)
            else:
                vv = torch.zeros(max_bbs, *R)
            vv[: min(max_bbs, N)] = v
            objects[k] = vv

        objects["mask"] = torch.zeros(max_bbs, dtype=torch.bool).where(
            torch.arange(max_bbs) > objects["bbox_id"].shape[0], True
        )

        return item
