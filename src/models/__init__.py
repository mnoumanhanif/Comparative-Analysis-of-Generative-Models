"""Neural network model architectures for generative models."""

from src.models.gan import Discriminator32, Generator32
from src.models.cgan import Discriminator_CGAN32, Generator_CGAN32
from src.models.vae import VAE32
from src.models.tabular_vae import TabularVAE

__all__ = [
    "Discriminator32",
    "Generator32",
    "Discriminator_CGAN32",
    "Generator_CGAN32",
    "VAE32",
    "TabularVAE",
]
