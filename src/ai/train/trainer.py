from lightning import Trainer

from ..dataset.hf_dataset import HuggingFaceDataset
from .datamodule import AIDataModule
from .model import AIModule


class AITrainer:
    def __init__(self, model_name: str, dataset_name: str):
        self.dataset = HuggingFaceDataset(name=dataset_name)
        self.datamodule = AIDataModule(self.dataset)

        self.model = AIModule(model_name, dataset=self.dataset)

        self.trainer = Trainer(accelerator="gpu")

    def fit(self):
        self.trainer.fit(self.model, datamodule=self.datamodule)
