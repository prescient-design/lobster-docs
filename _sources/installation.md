# Installation

There are multiple ways to install LBSTER on your system. Choose the approach that works best for your environment.

## Using `uv`

[`uv`](https://github.com/astral-sh/uv) is a fast Python package installer and resolver. This is the recommended installation method:

```bash
# Create a new virtual environment
uv venv --python 3.12

# Activate the virtual environment
source .venv/bin/activate

# Install LBSTER
uv pip install -e .
```

## Using `mamba` or `conda`

If you prefer using conda environments:

```bash
# Clone the repo
git clone https://github.com/prescient-design/lobster.git
cd lobster

# Create and activate the environment
mamba env create -f env.yml
conda activate lobster

# Install in development mode
pip install -e .
```

## Dependencies

LBSTER has the following core dependencies:

- Python >= 3.10
- torch
- lightning
- transformers >= 4.24.0
- biopython
- pandas
- scipy
- hydra-core
- wandb
- flash-attn (Linux only)

The full list of dependencies can be found in the `pyproject.toml` file.

## Installing Optional Dependencies

For working with the Multi-Modal Molecular (MGM) models, you can install additional dependencies:

```bash
pip install -e ".[mgm]"
```

This will install additional packages like RDKit and SELFIES which are needed for molecule processing.

## Verifying Installation

You can verify your installation by running the following Python code:

```python
from lobster.model import LobsterPMLM

# Load a pre-trained model
model = LobsterPMLM("asalam91/lobster_24M")
print(f"Model loaded successfully with {sum(p.numel() for p in model.parameters())} parameters")
```

## GPU Support

LBSTER works best with GPU acceleration. The package will automatically use CUDA if available. To check if your GPU is being recognized:

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
```

## Troubleshooting

If you encounter issues during installation:

1. Ensure you have the correct Python version (3.10+)
2. For GPU support, make sure you have the compatible CUDA version installed
3. Check that you have sufficient disk space (~4GB for full installation)
4. If you encounter issues with `flash-attn`, it might be platform-dependent. The package will still work without it but will be slower for some operations.

If you still have issues, please open an issue on the [GitHub repository](https://github.com/prescient-design/lobster/issues).
