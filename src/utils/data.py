"""Data loading and preprocessing utilities.

Provides functions for loading and preparing datasets used in the
generative model experiments: MNIST, CIFAR-10, and Credit Card Fraud.
"""

import os

import numpy as np
import pandas as pd
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


def get_mnist_dataset(root="dataset/", batch_size=128):
    """Load the MNIST dataset padded to 32x32 for consistent architecture.

    Args:
        root: Root directory for dataset storage.
        batch_size: Batch size for the DataLoader.

    Returns:
        Tuple of (dataset, dataloader).
    """
    transform = transforms.Compose([
        transforms.Pad(2),  # Pad from 28x28 to 32x32
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    dataset = datasets.MNIST(root=root, train=True, transform=transform, download=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return dataset, loader


def get_cifar10_dataset(root="dataset/", batch_size=128):
    """Load the CIFAR-10 dataset.

    Args:
        root: Root directory for dataset storage.
        batch_size: Batch size for the DataLoader.

    Returns:
        Tuple of (dataset, dataloader).
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = datasets.CIFAR10(root=root, train=True, transform=transform, download=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return dataset, loader


def get_credit_card_data(file_path="creditcard.csv", batch_size=256):
    """Load and preprocess the Credit Card Fraud Detection dataset.

    Scales the Amount and Time features, splits into normal training
    data and a balanced test set of normal and fraud samples.

    Args:
        file_path: Path to the creditcard.csv file.
        batch_size: Batch size for the DataLoaders.

    Returns:
        Tuple of (train_loader, test_loader, y_test) or None if file not found.
    """
    if not os.path.exists(file_path):
        print(f"Error: '{file_path}' not found. Please download from Kaggle.")
        return None

    data = pd.read_csv(file_path)
    scaler = StandardScaler()
    data['scaled_amount'] = scaler.fit_transform(data['Amount'].values.reshape(-1, 1))
    data['scaled_time'] = scaler.fit_transform(data['Time'].values.reshape(-1, 1))
    data = data.drop(['Time', 'Amount'], axis=1)

    normal_data = data[data['Class'] == 0]
    fraud_data = data[data['Class'] == 1]
    X_train = normal_data.drop('Class', axis=1).values
    X_test = pd.concat([
        normal_data.sample(n=len(fraud_data), random_state=42),
        fraud_data,
    ]).drop('Class', axis=1).values
    y_test = np.concatenate([np.zeros(len(fraud_data)), np.ones(len(fraud_data))])

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, y_test
