import torch.nn as nn

from .block import ConvBlock


class VGG(nn.Module):
    def __init__(self, config, in_channels=3, size=224):
        super().__init__()

        self.features = self._generate(config, in_channels)

        self.fc1 = nn.Linear(512 * (size // (2**5)) ** 2, 4096)
        self.fc2 = nn.Linear(4096, 4096)

    def _generate(self, arch: list, in_channels: int):
        layers = []
        arch[0][1] = in_channels

        for item in arch:
            if isinstance(item, str):
                if item == "M":
                    layers.append(nn.MaxPool2d(2))
                elif item == "L":
                    layers.append(nn.LocalResponseNorm(2))
            elif isinstance(item, list):
                t, *ps = item
                if t == "C":
                    layers.append(ConvBlock(*ps))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pool(self.conv1(x))  # WH = 112

        x = self.pool(self.conv2(x))  # WH = 56

        x = self.pool(self.conv3_2(self.conv3_1(x)))  # WH = 28

        x = self.pool(self.conv4_2(self.conv4_1(x)))  # WH = 14

        x = self.pool(self.conv5_2(self.conv5_2(x)))  # WH = 7

        x = self.features(x)

        x = x.flatten(start_dim=1)

        return self.fc2(self.fc1(x))
