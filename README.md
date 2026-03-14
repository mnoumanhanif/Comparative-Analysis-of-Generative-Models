# Comparative Analysis of Generative Models

A comprehensive implementation and comparison of three foundational generative AI architectures — **GAN**, **Conditional GAN**, and **VAE** — for image synthesis and anomaly detection.

## Overview

This project implements and evaluates three generative model architectures using PyTorch:

- **GAN (DCGAN):** Deep Convolutional GAN for high-quality image generation
- **CGAN:** Conditional GAN with class label control for targeted image generation
- **VAE:** Variational Autoencoder for image generation and anomaly detection

Models are trained on **MNIST** and **CIFAR-10** datasets and evaluated using FID and KID metrics. A separate **Tabular VAE** demonstrates anomaly detection on the Credit Card Fraud dataset, achieving **93% accuracy** and **0.93 F1-score**.

## Key Features

- DCGAN-style convolutional architectures for 32×32 image generation
- Class-conditional image generation with label embeddings
- VAE with reparameterization trick and latent space visualization
- VAE-based anomaly detection for tabular (non-image) data
- Quantitative evaluation using FID and KID metrics
- Comprehensive Jupyter notebook with all experiments

## Key Results

| Model | Dataset | FID ↓ | KID ↓ | Notes |
|-------|---------|-------|-------|-------|
| GAN | MNIST | 24.26 | 0.0123 | Best visual quality |
| CGAN | MNIST | 198.61 | 0.254 | Training instability |
| VAE | MNIST | 88.35 | 0.0708 | Smooth latent clustering |
| VAE (Anomaly) | Credit Card | – | – | Accuracy: 93%, F1: 0.93 |

## Tech Stack

- **Framework:** PyTorch
- **Metrics:** TorchMetrics (FID, KID)
- **Data Science:** NumPy, Pandas, scikit-learn
- **Visualization:** Matplotlib, Seaborn
- **Datasets:** MNIST, CIFAR-10, Credit Card Fraud (Kaggle)

## Project Structure

```
├── src/                        # Source code
│   ├── models/                 # Neural network architectures
│   │   ├── gan.py              # DCGAN Generator & Discriminator
│   │   ├── cgan.py             # Conditional GAN models
│   │   ├── vae.py              # Variational Autoencoder
│   │   └── tabular_vae.py      # Tabular VAE for anomaly detection
│   ├── evaluation/             # Evaluation metrics
│   │   └── metrics.py          # FID and KID computation
│   ├── training/               # Training utilities
│   └── utils/                  # Data loading & preprocessing
│       └── data.py
├── notebooks/                  # Jupyter notebooks
│   └── generative_models_analysis.ipynb
├── reports/                    # PDF reports
│   └── technical_report.pdf
├── tests/                      # Unit tests
│   └── test_models.py
├── configs/                    # Configuration files
│   └── default.py
├── docs/                       # Documentation
│   ├── setup.md
│   ├── architecture.md
│   └── development.md
├── .github/                    # GitHub Actions & templates
│   ├── workflows/ci.yml
│   ├── ISSUE_TEMPLATE/
│   └── PULL_REQUEST_TEMPLATE.md
├── requirements.txt
├── CONTRIBUTING.md
├── CHANGELOG.md
├── LICENSE
└── README.md
```

## Installation

### Prerequisites

- Python 3.9+
- pip package manager

### Setup

```bash
# Clone the repository
git clone https://github.com/mnoumanhanif/Comparative-Analysis-of-Generative-Models.git
cd Comparative-Analysis-of-Generative-Models

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

For GPU support with CUDA 11.8:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Usage

### Running the Notebook

```bash
jupyter notebook notebooks/generative_models_analysis.ipynb
```

### Using Models in Python

```python
import torch
from src.models import Generator32, Discriminator32, VAE32
from src.models.gan import initialize_weights

# Create and initialize a GAN generator
gen = Generator32(z_dim=100, channels_img=1, features_g=64)
initialize_weights(gen)

# Generate fake MNIST-like images
noise = torch.randn(16, 100, 1, 1)
fake_images = gen(noise)  # Shape: (16, 1, 32, 32)

# Create a VAE
vae = VAE32(in_channels=1, latent_dim=20)
sample_image = torch.randn(1, 1, 32, 32)
reconstruction, mu, logvar = vae(sample_image)
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific model tests
pytest tests/test_models.py::TestGAN -v
pytest tests/test_models.py::TestVAE -v
```

## Documentation

- [Setup Guide](docs/setup.md) — Installation and environment setup
- [Architecture Overview](docs/architecture.md) — Model designs and data flow
- [Development Guide](docs/development.md) — Contributing new models and features

## Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on:

- Reporting bugs and requesting features
- Submitting pull requests
- Code style and testing requirements

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

## Author

**Muhammad Nouman Hanif**
MS (Data Science) Candidate | FAST–NUCES Lahore

- Email: [mnoumanhanif66@gmail.com](mailto:mnoumanhanif66@gmail.com)
- LinkedIn: [linkedin.com/in/mnoumanhanif](https://www.linkedin.com/in/mnoumanhanif/)
