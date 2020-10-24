"""Code related to generating dataset."""

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader


def translated_gaussian(num_sample=1000, num_mechanism=4, scale=0.05, dist=1.0):
    mean = [0, 0]
    cov = np.array([[1, 0], [0, 1]]) * scale
    source_data = np.random.multivariate_normal(mean, cov, num_sample)
    mechanism = np.array([[0, 1], [1, 0], [-1, 0], [0, -1]]) * dist
    target_data = np.concatenate([m + source_data for m in mechanism])
    np.random.shuffle(target_data)
    return source_data, target_data


def translated_gaussian_dataset(batch_size, args, dist=1.0, num_sample=1000):
    src, tgt = translated_gaussian(num_sample, scale=args.noise_scale, dist=dist)

    np.random.shuffle(src)
    np.random.shuffle(tgt)
    tensor_src = torch.Tensor(src)
    tensor_tgt = torch.Tensor(tgt[:num_sample])

    dataset = TensorDataset(tensor_src, tensor_tgt)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(args.cuda),
        pin_memory=args.cuda,
        drop_last=True,
    )
    return dataloader


def single_translated_gaussian_dataset(batch_size, args, scale=1.0, num_sample=1000):
    mean = [0, 0]
    cov = np.array([[1, 0], [0, 1]]) * scale
    src = np.random.multivariate_normal(mean, cov, num_sample)
    tgt = np.random.multivariate_normal(mean, cov, num_sample) + 2.0

    tensor_src = torch.Tensor(src)
    tensor_tgt = torch.Tensor(tgt)

    dataset = TensorDataset(tensor_src, tensor_tgt)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(args.cuda),
        pin_memory=args.cuda,
        drop_last=True,
    )
    return dataloader