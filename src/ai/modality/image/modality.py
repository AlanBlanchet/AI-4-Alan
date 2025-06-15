from collections import defaultdict
from functools import cached_property
from pathlib import Path

import numpy as np
import torch
from albumentations import (
    BboxParams,
    Compose,
    Normalize,
)
from albumentations import Resize as AResize
from albumentations.pytorch import ToTensorV2
from PIL import Image as PImage
from torchvision.utils import draw_bounding_boxes

from ...utils.augmentations import AUGS
from ..modality import Modality


class Image(Modality):
    input: list[str] = "image"

    # @field_validator("image", mode="before")
    # def validate_image(cls, value):
    #     if isinstance(value, str):
    #         return dict(name=value)
    #     return value

    # @field_validator("bbox", mode="before")
    # def validate_bbox(cls, value):
    #     if isinstance(value, str):
    #         return dict(name=value)
    #     return value

    @cached_property
    def format(self):
        format = self.bbox.get("format", "pascal_voc")
        self.info(f"Using bbox format: {format}")
        return format

    @cached_property
    def normalize_params(self):
        mean = self.image.get("mean", [0.485, 0.456, 0.406])
        std = self.image.get("std", [0.229, 0.224, 0.225])
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        return mean, std

    @cached_property
    def transforms(self):
        mean, std = self.normalize_params
        # Always apply
        post = Compose([Normalize(mean=mean.tolist(), std=std.tolist()), ToTensorV2()])
        augs = self._augmentations
        self.log(f"Using augmentations:\n{augs}")
        # train/val transforms
        return dict(
            train=self._compose([self._preprocessor, augs, post]),
            val=self._compose([self._preprocessor, post]),
        )

    @cached_property
    def reverse_transforms(self):
        mean, std = self.normalize_params
        unnorm = Normalize((-mean / std).tolist(), (1.0 / std).tolist())
        return dict(
            train=Compose([unnorm]),
            val=Compose([unnorm]),
        )

    def _preprocess(self, image, bboxes=None, labels=None):
        image = torch.as_tensor(image)

        # Validation / Formatting
        if image.ndim == 2:
            image = image[:, :, None]

        # if image.shape[-1] == 1:
        #     image = np.repeat(image, 3, axis=-1)

        if image.shape[-1] == 3:
            image = image.permute(2, 0, 1)

        # If there are bboxes, we are in bbox mode
        extra = {"bboxes": bboxes, "labels": labels}
        if bboxes is None:
            extra = {}

        # TODO replace with torchvision transforms
        # out = self.transforms(image=image, **extra)
        image = self.augmentations(image)

        # Fill data with transformed values
        # for k, v in out.items():
        #     k = "bbox" if k == "bboxes" else k
        #     data[k] = torch.as_tensor(v)
        # data[self.input] = image

        return dict(image=image)

    def postprocess(self, data: dict[str, torch.Tensor], split: str):
        size = data["image_size"]
        image = data["image"]
        image = image[: size[0], : size[1]]
        image = image.permute(1, 2, 0).cpu().numpy()

        out = self.reverse_transforms[split](image=image * 255)

        data["image"] = out["image"].transpose(2, 0, 1)
        return data

    @classmethod
    def plot(
        cls,
        path: Path,
        image: torch.Tensor | np.ndarray,
        bbox: torch.Tensor = None,
        scores: torch.Tensor = None,
        labels: list[str] = None,
        gt_bbox: torch.Tensor = None,
        gt_labels: list[str] = None,
        upsample: bool = False,
    ):
        if isinstance(image, np.ndarray):
            image = torch.as_tensor(image)

        C, H, W = image.shape
        # Make sure image is high resolution enough
        max_res = max(H, W)
        if upsample and max_res < 1080:
            mul = 1080 / max_res
            image = AResize(int(H * mul), int(W * mul))(
                image=image.permute(1, 2, 0).numpy()
            )["image"]
            image = torch.as_tensor(image).permute(2, 0, 1)
            if bbox is not None:
                bbox = bbox * mul

        bbox_image = image
        if bbox is not None:
            if labels is not None and scores is not None:
                labels = (
                    labels
                    if scores is None
                    else [f"{l} {s:.2f}" for l, s in zip(labels, scores)]
                )

            bbox_image = draw_bounding_boxes(image=image, boxes=bbox, labels=labels)

        if path:
            bbox_image = PImage.fromarray(
                (bbox_image * 255).permute(1, 2, 0).to(torch.uint8).numpy()
            )
            bbox_image.save(path)
            cls.log(f"Saved image to {path}")

    @cached_property
    def _bbox_params(self):
        if self.format:
            return BboxParams(
                format=self.format,
                label_fields=["labels"],
                min_area=1,  # TODO check if this is ok
                min_visibility=0.1,
                min_width=1,
                min_height=1,
                check_each_transform=False,
            )
        return None

    def _compose(self, transforms, *args, **kwargs):
        return Compose(
            transforms,
            *args,
            bbox_params=self._bbox_params if self.bbox else None,
            **kwargs,
        )

    def _transforms_from_conf(self, conf: dict | str | list | int | float | tuple):
        args, kwargs = [], {}
        try:
            if isinstance(conf, str):
                aug = AUGS.get(conf, None)
                if aug is None:
                    args.append(conf)
                else:
                    args.append(aug())
            elif isinstance(conf, dict):
                for k, v in conf.items():
                    aug = AUGS.get(k, None)
                    if aug is None:
                        # Not an aug but params
                        if isinstance(v, dict):
                            kwargs.update({k: v})
                        else:
                            kwargs[k] = v
                    else:
                        arg, kwarg = self._transforms_from_conf(v)
                        a = self._compose(arg) if k == "Compose" else aug(*arg, **kwarg)
                        args.append(a)
            elif isinstance(conf, (list, tuple)):
                for pre in conf:
                    if isinstance(pre, str):
                        aug = getattr(AUGS, pre, None)
                        if aug is None:
                            if isinstance(pre, dict):
                                args.append(pre)
                            elif isinstance(pre, (list, tuple)):
                                args.extend(pre)
                        else:
                            args.append(aug())
                    elif isinstance(pre, dict):
                        a, k = self._transforms_from_conf(pre)
                        args.extend(a)
                        kwargs.update(k)
                    elif isinstance(pre, (list, tuple)):
                        a, k = self._transforms_from_conf(pre)
                        args.append(a)
                        kwargs.update(k)
                    else:
                        args.append(pre)
            elif isinstance(conf, (int, float)):
                args.append(conf)
        except Exception as e:
            raise Exception(
                f"Error in :\n{conf}.\nPlease check that the augmentations exist and that params are correct."
            ) from e

        return args, kwargs

    @cached_property
    def _preprocessor(self):
        if "preprocess" not in self.image:
            self.log("Using default resize to 256x256")
            conf = [dict(Resize=(256, 256))]
        else:
            conf = self.image["preprocess"]

        if conf is None or (isinstance(conf, str) and conf.lower() == "none"):
            transforms = []
        else:
            transforms, _ = self._transforms_from_conf(conf)

        return self._compose(transforms)

    @cached_property
    def _augmentations(self):
        conf = self.image.get("transforms", [])
        transforms, kwargs = self._transforms_from_conf(conf)
        return self._compose(transforms)

    @classmethod
    def collate_fn(cls, name: str, samples: list[torch.Tensor]):
        # name should be 'image' or 'bbox'
        out = defaultdict(list)

        # Track all the shapes
        shapes = []
        for item in samples:
            shapes.append(torch.as_tensor(item.shape))
        # Stack
        shapes = torch.stack(shapes)

        if name == "image":
            # Get max resolution
            H, W = shapes[:, 1].max(), shapes[:, 2].max()
            max_res = torch.tensor([H, W])

            images, masks, sizes = [], [], []
            for sample in samples:
                shape = torch.tensor(sample.shape[1:])
                h, w = (max_res - shape).unbind()
                padded = torch.nn.functional.pad(sample, (0, w, 0, h))
                mask = torch.zeros_like(padded[0], dtype=torch.bool)
                mask[: sample.shape[1], : sample.shape[2]] = True

                images.append(padded)
                sizes.append(shape.flip(0))
                masks.append(mask)

            out["image"] = torch.stack(images)
            out["image_mask"] = torch.stack(masks)
            out["image_size"] = torch.stack(sizes)  # [B, W, H]
        elif name == "bbox":
            out = cls.mask_collate(name, samples)

        return out

    def __call__(self, *args, **kwargs):
        return self._preprocess(*args, **kwargs)
