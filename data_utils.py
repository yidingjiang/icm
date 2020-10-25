"""Utilities for generating data etc"""
import os

import torch
import torchvision
import numpy as np


def generated_transformed_mnist(save_path):
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            root="./",
            train=True,
            download=True,
            transform=torchvision.transforms.Compose(
                [torchvision.transforms.ToTensor()]
            ),
        ),
        batch_size=64,
        shuffle=True,
    )
    data = []
    for x, y in train_loader:
        data.append(np.transpose(x.numpy(), (0, 2, 3, 1)))
    dataset = np.concatenate(data, axis=0)
    candidates = [shift_right, shift_left, shift_up, shift_down, noise]
    target_dataset = []
    for img in dataset:
        for _ in range(3):
            t = np.random.choice(candidates, 1)[0]
            target_dataset.append(t(img))
    target_dataset = np.array(target_dataset)

    np.save(os.path.join(save_path, "mnist_original_data"), dataset)
    np.save(os.path.join(save_path, "mnist_transformed_data"), target_dataset)


def shift(image, direction):
    axis = 0 if direction[1] == 0 else 1
    amount = 3 if direction[1] == 0 else 5
    amount *= -direction[axis]
    return np.roll(image, amount, axis)


def shift_right(image):
    return shift(image.copy(), [0, -1])


def shift_left(image):
    return shift(image.copy(), [0, 1])


def shift_up(image):
    return shift(image.copy(), [1, 0])


def shift_down(image):
    return shift(image.copy(), [-1, 0])


def invert(image):
    image = image.copy()
    if np.max(image) > 1.0:
        return 255.0 - image
    else:
        return 1.0 - image


def noise(image):
    image = image.copy()
    scale = 1.0 if np.max(image) <= 1.0 else 255.0
    added_noise = np.random.binomial(1, 0.05, size=image.shape) * scale
    image += added_noise
    clipped = np.clip(image, 0.0, scale)
    return clipped