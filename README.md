# Latent Stochastic VAE

A comprehensive comparison of Variational Autoencoders (VAE), Autoencoders (AE), and Latent Stochastic VAE (LS-VAE) for image generation and reconstruction tasks on the MNIST dataset.

## 🎯 Project Overview

This project implements and compares three different autoencoder architectures:

- **Vanilla VAE**: Standard Variational Autoencoder with KL divergence regularization
- **Autoencoder (AE)**: Deterministic autoencoder focused on reconstruction quality
- **Latent Stochastic VAE (LS-VAE)**: Enhanced VAE with additional stochasticity in the latent space

## 🚀 Key Features

- **Comprehensive Evaluation Metrics**: MSE, PSNR, SSIM, FID, Inception Score, and Uncertainty quantification
- **Smart Early Stopping**: Model-specific early stopping with different patience levels and monitoring metrics
- **Automated Visualization**: Generated comparison charts and sample images
- **CUDA Support**: GPU-accelerated training with PyTorch 2.5.1
- **Reproducible Results**: Consistent training and evaluation pipeline

## 📊 Performance Results

### Fixed Epochs vs Early Stopping Comparison

| Model  | Method         | Epochs | FID   | PSNR  | SSIM  | Inception Score |
|--------|----------------|--------|-------|-------|-------|-----------------|
| VAE    | Fixed (50)     | 50     | 70.24 | 14.22 | 0.154 | 2.15           |
| AE     | Fixed (50)     | 50     | 65.37 | 14.31 | 0.166 | 2.30           |
| LS-VAE | Fixed (50)     | 50     | 67.95 | 14.28 | 0.162 | 2.16           |
| AE     | Early Stopping | 22     | 63.96 | 14.30 | 0.166 | 2.28           |

**Performance Ranking**: AE > LS-VAE > VAE

## 🛠️ Installation

### Prerequisites
- Python 3.8-3.11
- NVIDIA GPU with CUDA 11.8 support
- 8GB+ RAM, 4GB+ VRAM recommended

### Setup
```bash
# Clone the repository
git clone <repository-url>
cd HybridVAE-Diffusion

# Create virtual environment
conda create -n vae-env python=3.10
conda activate vae-env

# Install dependencies with CUDA support
pip install torch==2.5.1+cu118 torchvision==0.20.1+cu118 torchaudio==2.5.1+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### Verify Installation
```bash
python test.py
```

## 🏃‍♂️ Usage

### Train All Models
```bash
python main.py
```

This will:
1. Train VAE, AE, and LS-VAE with early stopping
2. Generate evaluation metrics and comparison charts
3. Save model checkpoints every 5 epochs
4. Create sample generation visualizations

### Early Stopping Configuration
- **VAE**: Monitors validation loss, patience=15 epochs
- **AE**: Monitors validation loss, patience=10 epochs  
- **LS-VAE**: Monitors validation loss, patience=20 epochs NOTE-> This requires additional tuning as it stops reallyy early even with moderate pateince

## 📁 Project Structure

```
HybridVAE-Diffusion/
├── main.py                    # Main training script
├── vanillaVAE.py             # VAE and AE implementations
├── latentStochasticVAE.py    # LS-VAE implementation
├── hybridVAE.py              # Hybrid model architecture
├── test.py                   # Environment verification
├── requirements.txt          # Package dependencies
├── charts/                   # Generated comparison charts
│   ├── mse_comparison_bar_chart.png
│   ├── fid_comparison_bar_chart.png
│   └── ...
├── data/                     # MNIST dataset (auto-downloaded)
└── *.pt                      # Model checkpoints
```

## 📈 Generated Outputs

### Comparison Charts
- MSE, PSNR, SSIM comparison bar charts
- FID and Inception Score visualizations
- Uncertainty quantification plots

### Sample Images
- `VAE_Generated_Samples.png`
- `AE_Generated_Samples.png`
- `LS-VAE_Generated_Samples.png`

### Model Checkpoints
- Saved every 5 epochs during training
- Best model weights restored via early stopping

## 🔬 Key Findings

1. **AE achieves best reconstruction quality** but lacks generative capabilities
2. **LS-VAE outperforms vanilla VAE** across most metrics due to enhanced latent stochasticity
3. **Early stopping prevents overfitting** while maintaining performance
4. **LS-VAE requires longer training** (20+ epochs, current stop around 21 isn't enough) to reach optimal performance

## 🎛️ Hyperparameters

- **Learning Rate**: 1e-3 (Adam optimizer)
- **Batch Size**: 64
- **Latent Dimensions**: 4 channels × 8×8 spatial
- **Beta (VAE)**: 1.0 (standard), 0.5 (LS-VAE)
- **Image Size**: 64×64 (resized from 28×28 MNIST)

## 📋 Requirements

See `requirements.txt` for detailed package versions:
- PyTorch 2.5.1+cu118
- TorchMetrics ≥1.0.0
- NumPy, Matplotlib, Scikit-Image
- Pillow for image processing

## 🤝 Contributing

Feel free to open issues or submit pull requests for improvements!

## 📄 License

This project is open source and available under the [MIT License](LICENSE).