# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [1.0.0] - 2024-12-01

### Added

- Initial implementation of GAN (DCGAN) for 32x32 image generation
- Conditional GAN (CGAN) with class label conditioning
- Variational Autoencoder (VAE) for image generation
- Tabular VAE for anomaly detection on credit card fraud data
- FID and KID evaluation metrics
- Training on MNIST and CIFAR-10 datasets
- Comprehensive Jupyter notebook with all experiments
- Technical report (PDF) with analysis and results

## [1.1.0] - 2025-01-15

### Added

- Modular Python source code extracted from notebook
- Unit tests for all model architectures
- `requirements.txt` for dependency management
- `.gitignore` for clean repository
- Comprehensive `README.md` with setup instructions
- Project documentation (`docs/`)
- `CONTRIBUTING.md` guide for contributors
- `CHANGELOG.md` for tracking changes
- GitHub Actions CI workflow
- Issue and PR templates
- Configuration module (`configs/default.py`)

### Changed

- Reorganized repository structure for maintainability
- Moved notebook to `notebooks/` directory
- Moved PDF report to `reports/` directory
