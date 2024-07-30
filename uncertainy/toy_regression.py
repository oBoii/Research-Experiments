import math
from typing import Tuple

import numpy as np
import torch
from scipy import stats
from torch.utils.data import DataLoader, TensorDataset


def create_data_loader(x, y, batch_size, num_workers):
    dataset = TensorDataset(torch.tensor(x).float(), torch.tensor(y).float())
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                      num_workers=num_workers, pin_memory=True, persistent_workers=True)


def get_toy_regression_data_loaders(batch_size: int, num_workers: int, num_points: int) -> Tuple[
    DataLoader, DataLoader, DataLoader]:
    x = np.linspace(-10, 10, num_points)
    sample_mean = [math.sin(i / 2) for i in x]
    sample_var = [((abs(-10) + abs(10)) / 2 - abs(i)) / 16 for i in x]
    y = stats.norm(sample_mean, sample_var).rvs()

    # Shuffle the data
    indices = np.random.permutation(num_points)
    x, y = x[indices], y[indices]

    # Split the data into train, validation, and test sets
    train_size = int(0.7 * num_points)
    val_size = int(0.15 * num_points)
    test_size = num_points - train_size - val_size

    x_train, y_train = x[:train_size], y[:train_size]
    x_val, y_val = x[train_size:train_size + val_size], y[train_size:train_size + val_size]
    x_test, y_test = x[train_size + val_size:], y[train_size + val_size:]

    train_loader = create_data_loader(x_train, y_train, batch_size, num_workers)
    val_loader = create_data_loader(x_val, y_val, batch_size, num_workers)
    test_loader = create_data_loader(x_test, y_test, batch_size, num_workers)

    # print shapes
    print(f"Train loader shape: {len(train_loader.dataset)}")
    print(f"Val loader shape: {len(val_loader.dataset)}")
    print(f"Test loader shape: {len(test_loader.dataset)}")

    return train_loader, val_loader, test_loader
