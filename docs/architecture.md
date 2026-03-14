# Architecture Overview

## System Architecture

This project implements three generative model architectures and an anomaly detection system, all built with PyTorch.

```
┌─────────────────────────────────────────────────────┐
│                   Generative Models                  │
├─────────────┬──────────────┬────────────────────────┤
│     GAN     │    CGAN      │         VAE            │
│  (DCGAN)    │ (Conditional)│  (Variational AE)      │
├─────────────┴──────────────┴────────────────────────┤
│              Evaluation Metrics                      │
│         FID (Fréchet Inception Distance)             │
│         KID (Kernel Inception Distance)              │
├─────────────────────────────────────────────────────┤
│                   Datasets                           │
│    MNIST  │  CIFAR-10  │  Credit Card Fraud          │
└─────────────────────────────────────────────────────┘
```

## Model Architectures

### 1. GAN (Generative Adversarial Network)

**File:** `src/models/gan.py`

Based on the DCGAN (Deep Convolutional GAN) architecture for 32×32 images.

**Generator** (`Generator32`):
- Input: Latent noise vector `z` of shape `(batch, z_dim, 1, 1)`
- Architecture: 4 transposed convolution blocks with BatchNorm and ReLU
- Output: Generated image of shape `(batch, channels, 32, 32)` with Tanh activation

**Discriminator** (`Discriminator32`):
- Input: Image of shape `(batch, channels, 32, 32)`
- Architecture: 4 convolution blocks with BatchNorm and LeakyReLU
- Output: Probability score via Sigmoid activation

**Training:** Adversarial training with BCE loss. Discriminator learns to distinguish real from fake; generator learns to fool the discriminator.

### 2. CGAN (Conditional GAN)

**File:** `src/models/cgan.py`

Extends DCGAN with class label conditioning for controlled generation.

**Key Modifications:**
- **Generator:** Class labels are embedded and concatenated with the noise vector
- **Discriminator:** Class labels are embedded and concatenated with the input image as an additional channel

This allows generating images of specific classes (e.g., digit "1" for MNIST).

### 3. VAE (Variational Autoencoder)

**File:** `src/models/vae.py`

Encoder-decoder architecture with a probabilistic latent space.

**Encoder:**
- Two convolutional layers reduce 32×32 input to 8×8 feature maps
- Fully connected layers output mean (`mu`) and log-variance (`logvar`)

**Decoder:**
- Fully connected layer maps latent vector to 8×8 feature maps
- Two transposed convolution layers reconstruct the 32×32 image
- Sigmoid output for pixel values in [0, 1]

**Loss Function:** Sum of:
- Binary Cross-Entropy (reconstruction loss)
- KL Divergence (regularization of latent space)

### 4. Tabular VAE (Anomaly Detection)

**File:** `src/models/tabular_vae.py`

Fully-connected VAE for tabular (non-image) data.

**Anomaly Detection Strategy:**
1. Train on normal data only
2. Compute reconstruction error on test data
3. High reconstruction error → anomaly (fraud)
4. Threshold at 95th percentile of normal errors

## Evaluation Metrics

**File:** `src/evaluation/metrics.py`

- **FID (Fréchet Inception Distance):** Measures the distance between feature distributions of real and generated images using an InceptionV3 network. Lower is better.
- **KID (Kernel Inception Distance):** Similar to FID but uses Maximum Mean Discrepancy. More reliable for smaller sample sizes.

## Data Flow

```
Raw Data → Preprocessing → DataLoader → Model Training → Evaluation
                                              │
                                              ├→ Image Generation (GAN/CGAN/VAE)
                                              ├→ FID/KID Scoring
                                              └→ Anomaly Detection (TabularVAE)
```
