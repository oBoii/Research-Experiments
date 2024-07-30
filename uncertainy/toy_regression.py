import math
from typing import Tuple

import torch
from torch.utils.data import DataLoader, TensorDataset


def create_data_loader(x, y, batch_size, num_workers, shuffle: bool):
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                      num_workers=num_workers, pin_memory=True, persistent_workers=True)


def sample_dataset(start, end, n):
    x = torch.linspace(start, end, n)
    sample_mean = torch.sin(x / 2)
    sample_var = ((abs(start) + abs(end)) / 2 - torch.abs(x)) / 16
    y = torch.normal(sample_mean, sample_var)

    # expand dims
    x = x.unsqueeze(1)
    y = y.unsqueeze(1)
    return x, y


def get_toy_regression_data_loaders(batch_size: int, num_workers: int, num_points: int) -> Tuple[
    DataLoader, DataLoader, DataLoader]:
    # Split the data into train, validation, and test sets
    train_size = int(0.7 * num_points)
    val_size = int(0.15 * num_points)
    test_size = num_points - train_size - val_size

    x_train, y_train = sample_dataset(-7, 7, train_size)
    x_val, y_val = sample_dataset(-7, 7, val_size)
    x_test, y_test = sample_dataset(-10, 10, test_size)

    train_loader = create_data_loader(x_train, y_train, batch_size, num_workers, shuffle=True)
    val_loader = create_data_loader(x_val, y_val, batch_size, num_workers, shuffle=False)
    test_loader = create_data_loader(x_test, y_test, batch_size, num_workers, shuffle=False)

    # print shapes
    print(f"Train loader shape: {len(train_loader.dataset)}")
    print(f"Val loader shape: {len(val_loader.dataset)}")
    print(f"Test loader shape: {len(test_loader.dataset)}")

    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    loader, _, _ = get_toy_regression_data_loaders(32, 4, 100)

    x, y = next(iter(loader))
    print(x.shape, y.shape)
