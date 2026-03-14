"""Unit tests for model architectures.

Validates that all generative models produce correct output shapes
and can perform forward passes without errors.
"""

import torch
import pytest

from src.models.gan import Discriminator32, Generator32, initialize_weights
from src.models.cgan import Discriminator_CGAN32, Generator_CGAN32
from src.models.vae import VAE32, vae_loss_function
from src.models.tabular_vae import TabularVAE


DEVICE = "cpu"
BATCH_SIZE = 4
Z_DIM = 100
FEATURES_D = 64
FEATURES_G = 64
LATENT_DIM = 20
NUM_CLASSES = 10
EMBED_SIZE = 100
IMG_SIZE = 32


class TestGAN:
    """Tests for GAN Discriminator and Generator."""

    def test_generator_output_shape_mnist(self):
        gen = Generator32(Z_DIM, 1, FEATURES_G)
        noise = torch.randn(BATCH_SIZE, Z_DIM, 1, 1)
        output = gen(noise)
        assert output.shape == (BATCH_SIZE, 1, 32, 32)

    def test_generator_output_shape_cifar(self):
        gen = Generator32(Z_DIM, 3, FEATURES_G)
        noise = torch.randn(BATCH_SIZE, Z_DIM, 1, 1)
        output = gen(noise)
        assert output.shape == (BATCH_SIZE, 3, 32, 32)

    def test_discriminator_output_shape_mnist(self):
        disc = Discriminator32(1, FEATURES_D)
        img = torch.randn(BATCH_SIZE, 1, 32, 32)
        output = disc(img)
        assert output.shape == (BATCH_SIZE, 1, 1, 1)

    def test_discriminator_output_shape_cifar(self):
        disc = Discriminator32(3, FEATURES_D)
        img = torch.randn(BATCH_SIZE, 3, 32, 32)
        output = disc(img)
        assert output.shape == (BATCH_SIZE, 1, 1, 1)

    def test_discriminator_output_range(self):
        disc = Discriminator32(1, FEATURES_D)
        img = torch.randn(BATCH_SIZE, 1, 32, 32)
        output = disc(img)
        assert (output >= 0).all() and (output <= 1).all()

    def test_initialize_weights(self):
        gen = Generator32(Z_DIM, 1, FEATURES_G)
        initialize_weights(gen)
        # Verify weights are initialized (no error)
        noise = torch.randn(BATCH_SIZE, Z_DIM, 1, 1)
        output = gen(noise)
        assert output.shape == (BATCH_SIZE, 1, 32, 32)


class TestCGAN:
    """Tests for Conditional GAN Discriminator and Generator."""

    def test_generator_output_shape(self):
        gen = Generator_CGAN32(Z_DIM, 1, FEATURES_G, NUM_CLASSES, EMBED_SIZE)
        noise = torch.randn(BATCH_SIZE, Z_DIM, 1, 1)
        labels = torch.randint(0, NUM_CLASSES, (BATCH_SIZE,))
        output = gen(noise, labels)
        assert output.shape == (BATCH_SIZE, 1, 32, 32)

    def test_discriminator_output_shape(self):
        disc = Discriminator_CGAN32(1, FEATURES_D, NUM_CLASSES, IMG_SIZE)
        img = torch.randn(BATCH_SIZE, 1, 32, 32)
        labels = torch.randint(0, NUM_CLASSES, (BATCH_SIZE,))
        output = disc(img, labels)
        assert output.shape == (BATCH_SIZE, 1, 1, 1)

    def test_conditional_generation_different_classes(self):
        gen = Generator_CGAN32(Z_DIM, 1, FEATURES_G, NUM_CLASSES, EMBED_SIZE)
        noise = torch.randn(1, Z_DIM, 1, 1)
        out_0 = gen(noise, torch.tensor([0]))
        out_5 = gen(noise, torch.tensor([5]))
        # Different class labels should produce different outputs
        assert not torch.allclose(out_0, out_5)


class TestVAE:
    """Tests for Variational Autoencoder."""

    def test_vae_output_shape_mnist(self):
        vae = VAE32(1, LATENT_DIM)
        img = torch.randn(BATCH_SIZE, 1, 32, 32)
        recon, mu, logvar = vae(img)
        assert recon.shape == (BATCH_SIZE, 1, 32, 32)
        assert mu.shape == (BATCH_SIZE, LATENT_DIM)
        assert logvar.shape == (BATCH_SIZE, LATENT_DIM)

    def test_vae_output_shape_cifar(self):
        vae = VAE32(3, LATENT_DIM)
        img = torch.randn(BATCH_SIZE, 3, 32, 32)
        recon, mu, logvar = vae(img)
        assert recon.shape == (BATCH_SIZE, 3, 32, 32)

    def test_vae_reconstruction_range(self):
        vae = VAE32(1, LATENT_DIM)
        img = torch.randn(BATCH_SIZE, 1, 32, 32)
        recon, _, _ = vae(img)
        # Sigmoid output should be in [0, 1]
        assert (recon >= 0).all() and (recon <= 1).all()

    def test_vae_loss_function(self):
        vae = VAE32(1, LATENT_DIM)
        # Use data in [-1, 1] range (as produced by Normalize((0.5,), (0.5,)))
        img = torch.rand(BATCH_SIZE, 1, 32, 32) * 2 - 1
        recon, mu, logvar = vae(img)
        loss = vae_loss_function(recon, img, mu, logvar)
        assert loss.item() > 0

    def test_vae_reparameterize(self):
        vae = VAE32(1, LATENT_DIM)
        mu = torch.zeros(BATCH_SIZE, LATENT_DIM)
        logvar = torch.zeros(BATCH_SIZE, LATENT_DIM)
        z = vae.reparameterize(mu, logvar)
        assert z.shape == (BATCH_SIZE, LATENT_DIM)


class TestTabularVAE:
    """Tests for Tabular VAE used in anomaly detection."""

    def test_output_shape(self):
        input_dim = 30
        model = TabularVAE(input_dim, 16, 4)
        x = torch.randn(BATCH_SIZE, input_dim)
        recon, mu, log_var = model(x)
        assert recon.shape == (BATCH_SIZE, input_dim)
        assert mu.shape == (BATCH_SIZE, 4)
        assert log_var.shape == (BATCH_SIZE, 4)

    def test_reconstruction_loss(self):
        input_dim = 30
        model = TabularVAE(input_dim, 16, 4)
        x = torch.randn(BATCH_SIZE, input_dim)
        recon, mu, log_var = model(x)
        recon_loss = torch.nn.functional.mse_loss(recon, x, reduction='sum')
        assert recon_loss.item() > 0
