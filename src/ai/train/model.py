import torch.nn as nn
import torch.optim as optim
from lightning import LightningModule

from ..utils.arch import get_arch_module


class AIModule(LightningModule):
    def __init__(self, model_name: str):
        super().__init__()
        
        model_cls = get_arch_module(model_name)
        self.model: nn.Module = model_cls()

        self.loss = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        return loss

    def configure_optimizers(self):
        return optim.AdamW(self.model.parameters(), lr=1e-3)

    def loss(self, y_hat, y):
        return self.model.loss(y_hat, y)
