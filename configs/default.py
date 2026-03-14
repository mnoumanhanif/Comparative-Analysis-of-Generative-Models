"""Default configuration for model training and evaluation."""

# Device configuration
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# GAN Hyperparameters
Z_DIM = 100
FEATURES_DISC = 64
FEATURES_GEN = 64
NUM_EPOCHS_GAN = 25
LR_GAN = 2e-4
BATCH_SIZE = 128

# CGAN Hyperparameters
NUM_CLASSES = 10
EMBED_SIZE = 100
IMG_SIZE = 32

# VAE Hyperparameters
LATENT_DIM = 20
NUM_EPOCHS_VAE = 20
LR_VAE = 1e-3

# Anomaly Detection Hyperparameters
HIDDEN_DIM_AD = 16
LATENT_DIM_AD = 4
NUM_EPOCHS_AD = 10
LR_AD = 1e-3

# Image channels
CHANNELS_IMG_MNIST = 1
CHANNELS_IMG_CIFAR = 3

# Class-specific settings (based on Roll Number 24K-8001)
TARGET_DIGIT = 1
TARGET_CIFAR_CLASS = 1  # Automobile
