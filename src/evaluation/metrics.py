"""Evaluation metrics for generative model quality.

Provides FID (Fréchet Inception Distance) and KID (Kernel Inception Distance)
computation for both GAN/CGAN and VAE models.
"""

import torch
from torch.utils.data import DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
from tqdm import tqdm


def calculate_fid_kid_gan(
    generator,
    dataset,
    z_dim,
    device,
    batch_size=128,
    num_images=1000,
    is_mnist=False,
    conditional_class=None,
    embed_size=100,
):
    """Calculate FID and KID scores for GAN and CGAN generators.

    Args:
        generator: Trained generator model.
        dataset: Real image dataset for comparison.
        z_dim: Dimension of the latent noise vector.
        device: Device to run computations on.
        batch_size: Batch size for evaluation.
        num_images: Number of images to generate for evaluation.
        is_mnist: Whether the dataset is grayscale (requires channel replication).
        conditional_class: Class label for conditional generation (CGAN only).
        embed_size: Embedding size for CGAN (unused in computation, kept for API compat).

    Returns:
        Tuple of (fid_score, kid_mean, kid_std).
    """
    generator.eval()

    fid = FrechetInceptionDistance(normalize=True).to(device)
    kid = KernelInceptionDistance(subset_size=100, normalize=True).to(device)

    if conditional_class is not None:
        class_indices = [i for i, label in enumerate(dataset.targets) if label == conditional_class]
        class_dataset = torch.utils.data.Subset(dataset, class_indices)
        real_loader = DataLoader(class_dataset, batch_size=batch_size, shuffle=True)
    else:
        real_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Generate fake images
    fake_images = []
    with torch.no_grad():
        for _ in tqdm(range(num_images // batch_size), desc="Generating Fake Images"):
            noise = torch.randn(batch_size, z_dim, 1, 1).to(device)
            if conditional_class is not None:
                labels = (torch.ones(batch_size) * conditional_class).long().to(device)
                fake_batch = generator(noise, labels)
            else:
                fake_batch = generator(noise)

            if is_mnist:
                fake_batch = fake_batch.repeat(1, 3, 1, 1)
            fake_images.append(fake_batch)
    fake_images = torch.cat(fake_images, 0)

    fid.update(fake_images, real=False)
    kid.update(fake_images, real=False)

    # Process real images
    real_images_processed = 0
    for real_batch, _ in tqdm(real_loader, desc="Processing Real Images"):
        if real_images_processed >= num_images:
            break
        real_batch = real_batch.to(device)
        if is_mnist:
            real_batch = real_batch.repeat(1, 3, 1, 1)

        fid.update(real_batch, real=True)
        kid.update(real_batch, real=True)
        real_images_processed += real_batch.shape[0]

    fid_score = fid.compute()
    kid_mean, kid_std = kid.compute()

    return fid_score, kid_mean, kid_std


def calculate_fid_kid_vae(
    vae_model,
    dataset,
    latent_dim,
    device,
    batch_size=128,
    num_images=1000,
    is_mnist=False,
):
    """Calculate FID and KID scores for VAE models.

    Args:
        vae_model: Trained VAE model.
        dataset: Real image dataset for comparison.
        latent_dim: Dimension of the VAE latent space.
        device: Device to run computations on.
        batch_size: Batch size for evaluation.
        num_images: Number of images to generate for evaluation.
        is_mnist: Whether the dataset is grayscale (requires channel replication).

    Returns:
        Tuple of (fid_score, kid_mean, kid_std).
    """
    vae_model.eval()

    fid = FrechetInceptionDistance(normalize=True).to(device)
    kid = KernelInceptionDistance(subset_size=100, normalize=True).to(device)
    real_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Generate fake images from random latent vectors
    fake_images = []
    with torch.no_grad():
        for _ in tqdm(range(num_images // batch_size), desc="Generating Fake Images (VAE)"):
            z = torch.randn(batch_size, latent_dim).to(device)
            fake_batch = vae_model.decoder(vae_model.decoder_fc(z))
            if is_mnist:
                fake_batch = fake_batch.repeat(1, 3, 1, 1)
            fake_images.append(fake_batch)
    fake_images = torch.cat(fake_images, 0)

    fid.update(fake_images, real=False)
    kid.update(fake_images, real=False)

    # Process real images
    real_images_processed = 0
    for real_batch, _ in tqdm(real_loader, desc="Processing Real Images"):
        if real_images_processed >= num_images:
            break
        real_batch = real_batch.to(device)
        if is_mnist:
            real_batch = real_batch.repeat(1, 3, 1, 1)

        fid.update(real_batch, real=True)
        kid.update(real_batch, real=True)
        real_images_processed += real_batch.shape[0]

    fid_score = fid.compute()
    kid_mean, kid_std = kid.compute()

    return fid_score, kid_mean, kid_std
