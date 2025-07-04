import torch

from .modality import Modality


class Label(Modality):
    @classmethod
    def single_process(cls, data: dict[str, torch.Tensor], multiple=False):
        cls.log_msg_once("Processing labels")
        labels = data["labels"]
        # Process labels
        labels = torch.as_tensor(labels)
        if labels.ndim > 0 and not multiple:
            if labels.numel() > 1:
                if labels[0].numel() > 1:
                    raise ValueError(
                        f"Multiple labels found {map['input']} -> {labels.shape}"
                    )
                cls.log_msg_once(
                    "Multiple labels found, taking the first one since we don't accept multiple labels"
                )
                # If there are multiple labels, take the first one
                labels = labels[0]
            # Get 1 item
            labels = labels.item()

        data["labels"] = labels
        return data

    @classmethod
    def collate_fn(cls, name: str, samples: list[torch.Tensor]):
        res = {}
        ex = samples[0]
        if ex.ndim == 0:
            res[name] = torch.stack(samples)
        else:
            res = cls.mask_collate(name, samples)
        return res
