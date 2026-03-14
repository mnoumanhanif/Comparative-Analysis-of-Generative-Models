# Contributing to Comparative Analysis of Generative Models

Thank you for your interest in contributing! This guide will help you get started.

## How to Contribute

### Reporting Bugs

1. Check if the issue already exists in [GitHub Issues](../../issues).
2. If not, create a new issue using the **Bug Report** template.
3. Include as much detail as possible: steps to reproduce, expected behavior, and your environment.

### Suggesting Features

1. Open a new issue using the **Feature Request** template.
2. Describe the feature and its use case.
3. Discuss the approach before implementing.

### Submitting Changes

1. **Fork** the repository.
2. **Create a branch** from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes** following the code style guidelines below.
4. **Add tests** if applicable.
5. **Run existing tests** to ensure nothing is broken:
   ```bash
   pytest tests/
   ```
6. **Commit** your changes with a clear message:
   ```bash
   git commit -m "Add: brief description of changes"
   ```
7. **Push** to your fork and submit a **Pull Request**.

## Code Style Guidelines

- Follow [PEP 8](https://peps.python.org/pep-0008/) for Python code.
- Use descriptive variable and function names.
- Add docstrings to all public classes and functions.
- Keep functions focused and concise.
- Use type hints where practical.

## Development Setup

```bash
# Clone the repository
git clone https://github.com/mnoumanhanif/Comparative-Analysis-of-Generative-Models.git
cd Comparative-Analysis-of-Generative-Models

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/
```

## Project Structure

- `src/models/` — Neural network architectures
- `src/evaluation/` — Metrics and evaluation functions
- `src/utils/` — Data loading and preprocessing utilities
- `configs/` — Configuration files
- `tests/` — Unit tests
- `notebooks/` — Jupyter notebooks for experiments
- `docs/` — Project documentation

## Code of Conduct

Be respectful, constructive, and supportive of other contributors.
