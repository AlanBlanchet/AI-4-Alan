from datasets import load_dataset
from lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchvision.transforms.v2 import Resize


class AIDataModule(LightningDataModule):
    def __init__(self, dataset_name: str):
        super().__init__()
        self.dataset = resolve_dataset(dataset_name).with_format("torch")

        transforms = Resize(224)

        def _t(x):
            x["image"] = transforms(x["image"])
            return x

        self.dataset.map(_t, batched=True)

    def train_dataloader(self):
        return DataLoader(
            self.dataset["train"], batch_size=32, shuffle=True, num_workers=4
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset["train"], batch_size=32, shuffle=False, num_workers=4
        )


def resolve_dataset(dataset_name: str):
    if "hf::" in dataset_name:
        dataset = load_dataset(dataset_name.split("hf::")[1])
    else:
        dataset = load_dataset(dataset_name)

    return dataset
