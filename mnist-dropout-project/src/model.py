"""A shared MLP isolates hidden-unit dropout as the experimental variable."""
from torch import nn


class DigitNet(nn.Sequential):
    def __init__(self, rate=0.0, hidden=(512, 256)):
        layers = [nn.Flatten()]
        width = 784
        for units in hidden:
            layers.extend([nn.Linear(width, units), nn.ReLU(), nn.Dropout(rate)])
            width = units
        super().__init__(*layers, nn.Linear(width, 10))
