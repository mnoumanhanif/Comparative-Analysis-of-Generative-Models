"""DCGAN architecture for 32x32 image generation.

Implements the Generator and Discriminator networks using
deep convolutional layers following the DCGAN design pattern.
"""

import torch.nn as nn


class Discriminator32(nn.Module):
    """DCGAN Discriminator for 32x32 images.

    Uses strided convolutions to progressively downsample
    the input image to a single scalar output.

    Args:
        channels_img: Number of input image channels (1 for grayscale, 3 for RGB).
        features_d: Base number of feature maps in the discriminator.
    """

    def __init__(self, channels_img, features_d):
        super(Discriminator32, self).__init__()
        self.disc = nn.Sequential(
            # Input: N x C x 32 x 32
            nn.Conv2d(channels_img, features_d, kernel_size=4, stride=2, padding=1),  # 16x16
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

    def forward(self, x):
        return self.disc(x)


class Generator32(nn.Module):
    """DCGAN Generator for 32x32 images.

    Uses transposed convolutions to progressively upsample
    a latent noise vector into a 32x32 image.

    Args:
        z_dim: Dimension of the latent noise vector.
        channels_img: Number of output image channels (1 for grayscale, 3 for RGB).
        features_g: Base number of feature maps in the generator.
    """

    def __init__(self, z_dim, channels_img, features_g):
        super(Generator32, self).__init__()
        self.gen = nn.Sequential(
            # Input: N x z_dim x 1 x 1
            self._block(z_dim, features_g * 8, 4, 1, 0),  # 4x4
            self._block(features_g * 8, features_g * 4, 4, 2, 1),  # 8x8
            self._block(features_g * 4, features_g * 2, 4, 2, 1),  # 16x16
            nn.ConvTranspose2d(features_g * 2, channels_img, kernel_size=4, stride=2, padding=1),  # 32x32
            nn.Tanh(),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.gen(x)


def initialize_weights(model):
    """Initialize model weights using normal distribution.

    Applies normal initialization (mean=0, std=0.02) to all
    Conv2d, ConvTranspose2d, and BatchNorm2d layers.

    Args:
        model: PyTorch model to initialize.
    """
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
