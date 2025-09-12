# HybridVAE-Diffusion Project Setup Guide

## System Requirements

- **Python**: 3.8 - 3.11 (recommended: 3.10)
- **NVIDIA GPU**: with CUDA 11.8 support
- **CUDA Toolkit**: 11.8
- **cuDNN**: Compatible with CUDA 11.8
- **RAM**: 8GB+ recommended
- **VRAM**: 4GB+ recommended

## Installation Instructions

### 1. Create and Activate Virtual Environment

```bash
# Using conda (recommended)
conda create -n vae-env python=3.10
conda activate vae-env

# Or using venv
python -m venv vae-env
# Windows:
vae-env\Scripts\activate
# Linux/Mac:
source vae-env/bin/activate
```

### 2. Install PyTorch with CUDA Support

```bash
# Install PyTorch 2.5.1 with CUDA 11.8 support
pip install torch==2.5.1+cu118 torchvision==0.20.1+cu118 torchaudio==2.5.1+cu118 --index-url https://download.pytorch.org/whl/cu118
```

### 3. Install Other Dependencies

```bash
# Install remaining packages
pip install -r requirements.txt
```

### Alternative: Install All at Once

```bash
# Install everything from requirements.txt (includes PyTorch)
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu118
```

## Verify Installation

Run the test script to verify everything is working:

```bash
python test.py
```

Expected output:
```
/path/to/python
2.5.1+cu118
True
11.8
```

## Running the Project

```bash
# Train all models with early stopping
python main.py
```

## Package Versions Used

- **PyTorch**: 2.5.1+cu118
- **TorchVision**: 0.20.1+cu118
- **TorchAudio**: 2.5.1+cu118
- **TorchMetrics**: >=1.0.0
- **NumPy**: >=1.21.0
- **Matplotlib**: >=3.5.0
- **Scikit-Image**: >=0.19.0

## Troubleshooting

### CUDA Issues
- Ensure NVIDIA drivers are up to date
- Verify CUDA 11.8 is installed: `nvcc --version`
- Check GPU compatibility: `nvidia-smi`

### Memory Issues
- Reduce batch size in main.py (currently 64)
- Close other GPU-intensive applications

### Import Errors
- Ensure virtual environment is activated
- Reinstall packages if needed: `pip install --force-reinstall -r requirements.txt`

## Project Structure

```
HybridVAE-Diffusion/
├── main.py                 # Main training script
├── vanillaVAE.py          # VAE and AE implementations
├── latentStochasticVAE.py # LS-VAE implementation
├── hybridVAE.py           # Hybrid model
├── test.py                # Environment test script
├── requirements.txt       # Package dependencies
├── data/                  # MNIST dataset (auto-downloaded)
├── charts/                # Generated comparison charts
└── *.pt                   # Model checkpoints
```
