# Setup Guide

## Prerequisites

- **Python** 3.9 or later
- **pip** package manager
- **Git** version control
- **CUDA** (optional, for GPU acceleration)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/mnoumanhanif/Comparative-Analysis-of-Generative-Models.git
cd Comparative-Analysis-of-Generative-Models
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

**CPU-only (recommended for development):**
```bash
pip install -r requirements.txt
```

**With CUDA support (for GPU training):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
```

### 5. Download Datasets

MNIST and CIFAR-10 datasets are downloaded automatically when running the models.

For the anomaly detection experiment, download the [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) from Kaggle and place `creditcard.csv` in the project root.

## Running Tests

```bash
pytest tests/ -v
```

## Running the Notebook

```bash
jupyter notebook notebooks/
```

## Troubleshooting

### CUDA Issues

If you encounter CUDA-related errors:

1. Verify your NVIDIA drivers: `nvidia-smi`
2. Check CUDA version compatibility with your PyTorch installation
3. Models will automatically fall back to CPU if CUDA is unavailable

### Memory Issues

If you run out of memory during training:

1. Reduce `BATCH_SIZE` in `configs/default.py`
2. Reduce `num_images` in evaluation functions
3. Use CPU instead of GPU for smaller experiments
