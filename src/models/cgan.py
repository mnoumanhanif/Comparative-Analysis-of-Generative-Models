"""Conditional GAN architecture for 32x32 image generation.

Extends the DCGAN architecture with class label conditioning
using embedding layers, allowing controlled image generation.
"""

import torch
import torch.nn as nn


class Discriminator_CGAN32(nn.Module):
    """Conditional GAN Discriminator for 32x32 images.

    Concatenates a class label embedding with the input image
    before passing through the discriminator network.

    Args:
        channels_img: Number of input image channels.
        features_d: Base number of feature maps.
        num_classes: Number of classes for conditioning.
        img_size: Spatial dimension of input images (assumed square).
    """

    def __init__(self, channels_img, features_d, num_classes, img_size):
        super(Discriminator_CGAN32, self).__init__()
        self.img_size = img_size
        self.embed = nn.Embedding(num_classes, img_size * img_size)
        self.disc = nn.Sequential(
            nn.Conv2d(channels_img + 1, features_d, kernel_size=4, stride=2, padding=1),  # 16x16
            nn.LeakyReLU(0.2),
            self._block(features_d, features_d * 2, 4, 2, 1),  # 8x8
            self._block(features_d * 2, features_d * 4, 4, 2, 1),  # 4x4
            nn.Conv2d(features_d * 4, 1, kernel_size=4, stride=1, padding=0),  # 1x1
            nn.Sigmoid(),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x, labels):
        embedding = self.embed(labels).view(labels.shape[0], 1, self.img_size, self.img_size)
        x = torch.cat([x, embedding], dim=1)
        return self.disc(x)


class Generator_CGAN32(nn.Module):
    """Conditional GAN Generator for 32x32 images.

    Concatenates a class label embedding with the latent noise
    vector before generating an image.

    Args:
        z_dim: Dimension of the latent noise vector.
        channels_img: Number of output image channels.
        features_g: Base number of feature maps.
        num_classes: Number of classes for conditioning.
        embed_size: Dimension of the label embedding.
    """

    def __init__(self, z_dim, channels_img, features_g, num_classes, embed_size):
        super(Generator_CGAN32, self).__init__()
        self.gen = nn.Sequential(
            self._block(z_dim + embed_size, features_g * 8, 4, 1, 0),
            self._block(features_g * 8, features_g * 4, 4, 2, 1),
            self._block(features_g * 4, features_g * 2, 4, 2, 1),
            nn.ConvTranspose2d(features_g * 2, channels_img, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )
        self.embed = nn.Embedding(num_classes, embed_size)

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x, labels):
        embedding = self.embed(labels).unsqueeze(2).unsqueeze(3)
        x = torch.cat([x, embedding], dim=1)
        return self.gen(x)
