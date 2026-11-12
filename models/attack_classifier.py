import torch
from torch import nn


class AttackClassifier(nn.Module):
    def __init__(self, input_size=10):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),

            nn.Linear(32, 16),
            nn.ReLU(),

            nn.Linear(16, 1)
        )

    def forward(self, x):
        x = self.network(x)
        return x.squeeze(1)