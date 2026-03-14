"""Tabular VAE architecture for anomaly detection.

Implements a fully-connected Variational Autoencoder designed
for tabular data, used for detecting anomalies such as credit
card fraud through reconstruction error analysis.
"""

import torch
import torch.nn as nn


class TabularVAE(nn.Module):
    """Variational Autoencoder for tabular (non-image) data.

    Uses fully-connected layers to encode tabular features into a
    latent space and reconstruct them. Anomalies are detected by
    measuring reconstruction error.

    Args:
        input_dim: Number of input features.
        hidden_dim: Number of neurons in the hidden layers.
        latent_dim: Dimension of the latent space.
    """

    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(TabularVAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_log_var = nn.Linear(hidden_dim // 2, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def reparameterize(self, mu, log_var):
        """Apply the reparameterization trick for backpropagation through sampling."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu, log_var = self.fc_mu(h), self.fc_log_var(h)
        z = self.reparameterize(mu, log_var)
        return self.decoder(z), mu, log_var
