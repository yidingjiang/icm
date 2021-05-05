"""Discriminator models."""

import math
import numbers
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.models as models

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
        w = self.config.width_multiplier * 4
        conv_layer = SNConv2d if self.config.use_sn else nn.Conv2d
        layers = [
            conv_layer(self.config.input_shape[-1], 16 * w, 3, 1, padding=1),
            nn.LeakyReLU(),
            conv_layer(16 * w, 16 * w, 3, 1, padding=1),
            nn.LeakyReLU(),
            conv_layer(16 * w, 16 * w, 3, 1, padding=1),
            nn.LeakyReLU(),
            conv_layer(16 * w, 32 * w, 3, 2, padding=1),
            nn.LeakyReLU(),
            conv_layer(32 * w, 32 * w, 3, 1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(32 * w, 64 * w, 3, 1, padding=1),
            nn.LeakyReLU(),
            nn.AvgPool2d(self.config.input_shape[0] // 2),
            nn.Flatten(),
            # SNLinear(64, 100),
            nn.Linear(64 * w, 1),
            # nn.ReLU(),
            # # SNLinear(100, 1),
            # nn.Linear(100, 1),
        ]
        if self.config.discriminator_sigmoid:
            layers.append(nn.Sigmoid())
        return nn.Sequential(*layers)


class ResNetDiscriminator(nn.Module):

    def __init__(self, args):
        super(ResNetDiscriminator, self).__init__()
        self.config = args
        model = models.resnet18()
        model.fc = nn.Linear(512, 1)
        self.conv1 = nn.Conv2d(args.input_shape[-1], 3, 1, 1)
        self.model = model
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        out = self.model(self.conv1(input))
        return self.sigmoid(out)


class MechanismConvolutionDiscriminator(nn.Module):
    def __init__(self, args):
        super(MechanismConvolutionDiscriminator, self).__init__()
        self.config = args
        self.model = self.build()

    def forward(self, input):
        out = self.model(input)
        return out

    def build(self):
        w = self.config.width_multiplier // 2
        conv_layer = SNConv2d if self.config.use_sn else nn.Conv2d
        layers = [
            conv_layer(self.config.input_shape[-1] * 2, 16 * w, 5, 2, padding=2),
            nn.LeakyReLU(),
            conv_layer(16 * w, 16 * w, 3, 1, padding=1),
            nn.LeakyReLU(),
            conv_layer(16 * w, 32 * w, 3, 2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(32 * w, 100, 3, 1, padding=1),
            nn.LeakyReLU(),
            nn.AvgPool2d(self.config.input_shape[0] // 4),
            nn.Flatten(),
            SNLinear(100, self.config.num_experts),
        ]
        return nn.Sequential(*layers)

    # def build(self):
    #     w = self.config.width_multiplier
    #     conv_layer = SNConv2d if self.config.use_sn else nn.Conv2d
    #     layers = [
    #         conv_layer(self.config.input_shape[-1] * 2, 16 * w, 3, 2, padding=1),
    #         nn.LeakyReLU(),
    #         conv_layer(16 * w, 16 * w, 3, 1, padding=1),
    #         nn.LeakyReLU(),
    #         conv_layer(16 * w, 16 * w, 3, 1, padding=1),
    #         nn.LeakyReLU(),
    #         conv_layer(16 * w, 32 * w, 3, 2, padding=1),
    #         nn.LeakyReLU(),
    #         conv_layer(32 * w, 32 * w, 3, 1, padding=1),
    #         nn.LeakyReLU(),
    #         nn.Conv2d(32 * w, 100, 3, 1, padding=1),
    #         nn.LeakyReLU(),
    #         nn.AvgPool2d(self.config.input_shape[0] // 4),
    #         nn.Flatten(),
    #         SNLinear(100, self.config.num_experts),
    #     ]
    #     return nn.Sequential(*layers)


class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """

    def __init__(self, channels, kernel_size, sigma, device='cuda', dim=2, padding=None):
        super(GaussianSmoothing, self).__init__()
        self.padding = padding
        if not self.padding:
            self.padding = kernel_size // 2
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim
        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [torch.arange(size, dtype=torch.float32) for size in kernel_size]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= (
                1
                / (std * math.sqrt(2 * math.pi))
                * torch.exp(-(((mgrid - mean) / (2 * std)) ** 2))
            )

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.device = device
        kernel = kernel.to(self.device)

        self.register_buffer("weight", kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                "Only 1, 2 and 3 dimensions are supported. Received {}.".format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(
            input, weight=self.weight, groups=self.groups, padding=self.padding
        )

