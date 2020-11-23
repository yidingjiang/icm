"""Utilities for generating data etc"""
import os

import torch
import torchvision
import numpy as np


def generated_transformed_mnist(save_path, num_transform=1, add_rotation=False):
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
    if add_rotation:
        candidates.append(rotate)
    target_dataset = []
    for img in dataset:
        for _ in range(3):
            ts = np.random.choice(candidates, num_transform, replace=False)
            img_t = img.copy()
            for t in ts:
                img_t = t(img_t)
            target_dataset.append(img_t)
    target_dataset = np.array(target_dataset)

    original_data_name = "mnist_original_data"
    transformed_data_name = "mnist_transformed_data"
    if num_transform > 1:
        original_data_name += "_{}_transform".format(num_transform)
        transformed_data_name += "_{}_transform".format(num_transform)
    np.save(os.path.join(save_path, original_data_name), dataset)
    np.save(os.path.join(save_path, transformed_data_name), target_dataset)


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


def rotate(image):
    image = image.copy()
    return np.transpose(image, [1, 0, 2])


# ========================================================================


def generate_supervised_training_data(args, experts, image_per_expert=10000):
    def get_loader(train=True):
        return torch.utils.data.DataLoader(
            torchvision.datasets.MNIST(
                root="./",
                train=train,
                download=True,
                transform=torchvision.transforms.Compose(
                    [torchvision.transforms.ToTensor()]
                ),
            ),
            batch_size=64,
            shuffle=True,
            drop_last=True,
        )

    train_loader = get_loader()
    num_batch = image_per_expert // 64
    train_data, train_label = [], []
    for t in experts:
        for _ in num_batch:
            try:
                x, y = next(train_loader)
            except StopIteration:
                train_loader = get_loader()
                x, y = next(train_loader)
            x = x.to(args.device)
            x_t = t(x)
            train_data.append(np.transpose(x_t.cpu().numpy(), (0, 2, 3, 1)))
            train_label.append(y.cpu().numpy())

    train_data = np.concatenate(train_data, axis=0)
    
    
            
