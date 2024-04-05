from lightning import Trainer

from .dataset import AIDataModule
from .model import AIModule


class AITrainer:
    def __init__(self, model_name: str, dataset_name: str):
        self.model = AIModule(model_name)
        self.datamodule = AIDataModule(dataset_name)

        self.trainer = Trainer()

    def fit(self):
        self.trainer.fit(self.model, datamodule=self.datamodule)
