# Development Guide

## Project Structure

```
├── src/                    # Main source code
│   ├── models/             # Neural network architectures
│   │   ├── gan.py          # DCGAN Generator and Discriminator
│   │   ├── cgan.py         # Conditional GAN models
│   │   ├── vae.py          # Variational Autoencoder
│   │   └── tabular_vae.py  # Tabular VAE for anomaly detection
│   ├── evaluation/         # Evaluation metrics
│   │   └── metrics.py      # FID and KID computation
│   ├── training/           # Training scripts (extensible)
│   └── utils/              # Utility functions
│       └── data.py         # Data loading and preprocessing
├── notebooks/              # Jupyter notebooks
├── reports/                # PDF reports and results
├── tests/                  # Unit tests
├── configs/                # Configuration files
├── docs/                   # Documentation
└── .github/                # GitHub Actions and templates
```

## Adding a New Model

1. Create a new file in `src/models/` (e.g., `src/models/wgan.py`).
2. Define the model class(es) inheriting from `nn.Module`.
3. Add the model to `src/models/__init__.py`.
4. Write tests in `tests/test_models.py`.
5. Update configuration in `configs/default.py` if needed.

Example:

```python
# src/models/wgan.py
import torch.nn as nn

class WGANCritic(nn.Module):
    def __init__(self, channels_img, features_d):
        super().__init__()
        # ... architecture definition
    
    def forward(self, x):
        return self.model(x)
```

## Adding a New Dataset

1. Add loading logic to `src/utils/data.py`.
2. Follow the existing pattern returning `(dataset, dataloader)` tuples.
3. Update `configs/default.py` with any new hyperparameters.

## Running Experiments

### Using the Notebook

The primary way to run full experiments is through the Jupyter notebook:

```bash
jupyter notebook notebooks/
```

### Using Python Modules

You can also import and use models directly:

```python
import torch
from src.models import Generator32, Discriminator32
from src.models.gan import initialize_weights

# Create models
gen = Generator32(z_dim=100, channels_img=1, features_g=64)
disc = Discriminator32(channels_img=1, features_d=64)

# Initialize weights
initialize_weights(gen)
initialize_weights(disc)

# Generate images
noise = torch.randn(16, 100, 1, 1)
fake_images = gen(noise)
```

## Configuration

All hyperparameters are centralized in `configs/default.py`:

- `Z_DIM` — Latent space dimension for GAN/CGAN
- `LATENT_DIM` — Latent space dimension for VAE
- `BATCH_SIZE` — Training batch size
- `NUM_EPOCHS_GAN` — Number of GAN training epochs
- `LR_GAN` — GAN learning rate

## Testing

Run all tests:

```bash
pytest tests/ -v
```

Run specific test class:

```bash
pytest tests/test_models.py::TestGAN -v
```

## Code Quality

Follow these practices:

- Add docstrings to all public functions and classes
- Use type hints where practical
- Keep functions focused on a single responsibility
- Write tests for new functionality
