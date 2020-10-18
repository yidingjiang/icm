"""Discriminator models."""

import numpy as np
import torch.nn as nn
from layers import SNLinear


class Discriminator(nn.Module):
    """Single expert.

    Args:
        args: argparse object
    """

    def __init__(self, args):
        super(Discriminator, self).__init__()
        self.config = args
        self.model = self.build()

    def build(self):
        layers = [
            SNLinear(self.config.input_shape, 32),
            nn.LeakyReLU(),
            SNLinear(32, 64),
            nn.LeakyReLU(),
            SNLinear(64, self.config.discriminator_output_size),
        ]
        if self.config.discriminator_sigmoid:
            layers.append(nn.Sigmoid())
        return nn.Sequential(*layers)

    def forward(self, input):
        out = self.model(input)
        return out
