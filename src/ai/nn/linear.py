import torch.nn as nn


def _linear_layer(input_size, output_size):
    return nn.Sequential(nn.Linear(input_size, output_size), nn.ReLU(), nn.Sigmoid())


def create_linear(input_size, output_size, hidden_layers: tuple[int] | list[int] = []):
    hidden_layers.insert(0, input_size)
    hidden_layers.append(output_size)
    return nn.Sequential(
        *[
            _linear_layer(hidden_layers[i], hidden_layers[i + 1])
            for i in range(len(hidden_layers) - 1)
        ],
    )
