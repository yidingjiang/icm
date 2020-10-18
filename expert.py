"""Definition of models."""

import numpy as np
import torch.nn as nn
from layers import SNLinear


class Expert(nn.Module):
    """Single expert.

    Args:
        args: argparse object
    """

    def __init__(self, args):
        super(Expert, self).__init__()
        self.config = args
        self.model = self.build()

    def build(self):
        layers = [
            SNLinear(self.config.input_shape, 64),
            nn.ReLU(),
            SNLinear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.config.input_shape),
        ]
        return nn.Sequential(*layers)

    def forward(self, input):
        out = self.model(input)
        return out


class AffineExpert(Expert):
    """Expert that uses linear functions."""

    def build(self):
        layers = [
            SNLinear(self.config.input_shape, self.config.input_shape),
        ]
        return nn.Sequential(*layers)
