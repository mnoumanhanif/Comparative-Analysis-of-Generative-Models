"""Variational Autoencoder architecture for 32x32 image generation.

Implements an encoder-decoder architecture with a reparameterization
trick for learning a continuous latent space representation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE32(nn.Module):
    """Variational Autoencoder for 32x32 images.

    Uses convolutional layers for encoding and transposed
    convolutions for decoding, with a reparameterization
    trick for sampling from the latent space.

    Args:
        in_channels: Number of input image channels.
        latent_dim: Dimension of the latent space.
    """

    def __init__(self, in_channels, latent_dim):
        super(VAE32, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),  # 16x16
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 8x8
            nn.ReLU(),
            nn.Flatten(),
        )
        self.fc_mu = nn.Linear(64 * 8 * 8, latent_dim)
        self.fc_logvar = nn.Linear(64 * 8 * 8, latent_dim)

        # Decoder
        self.decoder_fc = nn.Linear(latent_dim, 64 * 8 * 8)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (64, 8, 8)),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(32, in_channels, kernel_size=4, stride=2, padding=1),  # 32x32
            nn.Sigmoid(),
        )

    def reparameterize(self, mu, logvar):
        """Apply the reparameterization trick for backpropagation through sampling."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        return self.decoder(self.decoder_fc(z)), mu, logvar


def vae_loss_function(recon_x, x, mu, logvar):
    """Compute VAE loss as the sum of reconstruction loss and KL divergence.

    Args:
        recon_x: Reconstructed images from the decoder.
        x: Original input images (normalized to [-1, 1]).
        mu: Mean of the latent distribution.
        logvar: Log variance of the latent distribution.

    Returns:
        Total VAE loss (BCE reconstruction + KL divergence).
    """
    # Un-normalize from [-1, 1] to [0, 1] for BCE loss
    x_unnorm = x * 0.5 + 0.5
    bce = F.binary_cross_entropy(recon_x, x_unnorm, reduction='sum')
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return bce + kld
