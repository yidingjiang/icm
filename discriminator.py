"""Discriminator models."""

import numpy as np
import torch.nn as nn
from layers import SNLinear
from layers import SNConv2d


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


class ConvolutionDiscriminator(Discriminator):
    """Convolutional discriminator."""

    def build(self):
        w = self.config.width_multiplier
        conv_layer = SNConv2d if self.config.use_sn else nn.Conv2d
        layers = [
            conv_layer(self.config.input_shape[-1], 16*w, 3, 2, padding=1),
            nn.LeakyReLU(),
            conv_layer(16*w, 16*w, 3, 1, padding=1),
            nn.LeakyReLU(),
            conv_layer(16*w, 16*w, 3, 1, padding=1),
            nn.LeakyReLU(),
            conv_layer(16*w, 32*w, 3, 2, padding=1),
            nn.LeakyReLU(),
            conv_layer(32*w, 32*w, 3, 1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(32*w, 64*w, 3, 1, padding=1),
            nn.LeakyReLU(),
            nn.AvgPool2d(self.config.input_shape[0]//4),
            nn.Flatten(),
            SNLinear(64*w, 100),
            nn.ReLU(),
            SNLinear(100, 1),
        ]
        if self.config.discriminator_sigmoid:
            layers.append(nn.Sigmoid())
        return nn.Sequential(*layers)


class MechanismConvolutionDiscriminator(nn.Module):

    def __init__(self, args):
        super(MechanismConvolutionDiscriminator, self).__init__()
        self.config = args
        self.model = self.build()

    def forward(self, input):
        out = self.model(input)
        return out

    def build(self):
        w = self.config.width_multiplier
        conv_layer = SNConv2d if self.config.use_sn else nn.Conv2d
        layers = [
            conv_layer(self.config.input_shape[-1]*2, 16*w, 3, 2, padding=1),
            nn.LeakyReLU(),
            conv_layer(16*w, 16*w, 3, 1, padding=1),
            nn.LeakyReLU(),
            conv_layer(16*w, 16*w, 3, 1, padding=1),
            nn.LeakyReLU(),
            conv_layer(16*w, 32*w, 3, 2, padding=1),
            nn.LeakyReLU(),
            conv_layer(32*w, 32*w, 3, 1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(32*w, 64*w, 3, 1, padding=1),
            nn.LeakyReLU(),
            nn.AvgPool2d(self.config.input_shape[0]//4),
            nn.Flatten(),
            SNLinear(64*w, 100),
            nn.ReLU(),
            SNLinear(100, self.config.num_experts),
        ]
        return nn.Sequential(*layers)
