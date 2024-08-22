from torch import nn


class ClassificationModel(nn.Module):
    def __init__(self, model, in_channels: int, num_classes: int):
        super().__init__()

        self.model = model
        self.head = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        x = self.model(x)
        return self.head(x)
