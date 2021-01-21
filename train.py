"""Main training script."""
import argparse
import json

import torch
import numpy as np

from expert import Expert
from discriminator import Discriminator
from data import translated_gaussian_dataset
from train_utils import initialize_experts
from train_utils import train_icm


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ICM experiment configuration")
    parser.add_argument(
        "--num_experts", type=int, default=8, help="number of experts (default: 5)"
    )
    parser.add_argument("--input_shape", default=2, help="Size of the input shape")
    parser.add_argument(
        "--d_output_size", type=int, default=1, help="Size of the discriminator output"
    )
    parser.add_argument(
        "--num_epoch", type=int, default=20, help="number of icm training epoch"
    )
    parser.add_argument(
        "--min_init_loss",
        type=float,
        default=0.01,
        help="Minimum loss for initialization"
    )
    parser.add_argument(
        "--num_initialize_epoch", type=int, default=10, help="Number of epochs at init"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Size of the minibatch"
    )

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    # pylint: disable=E1101
    args.device = torch.device("cuda" if args.cuda else "cpu")
    # pylint: enable=E1101

    # Data
    data = translated_gaussian_dataset(args.batch_size, args)

    # Model
    experts = [Expert(args).to(args.device) for i in range(args.num_experts)]
    discriminator = Discriminator(args).to(args.device)

    # initialize_experts(experts, data, args)

    discriminator_opt = torch.optim.Adam(discriminator.parameters())
    expert_opt = []
    for e in experts:
        expert_opt.append(torch.optim.Adam(e.parameters()))

    for n in range(args.num_epoch):
        train_icm(experts, expert_opt, discriminator, discriminator_opt, data, args)
        print([e(torch.Tensor(np.array([[0.0, 0.0]]))) for e in experts])
