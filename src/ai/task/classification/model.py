from torch import nn


class ClassificationModel(nn.Module):
    def __init__(self, model, in_features: int, num_classes: int):
        super().__init__()

        self.model = model

        # if hasattr(model, "")

        self.clf = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.model.features(x)
        return self.head(x)
