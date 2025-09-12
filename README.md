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

### Comprehensive Training Results

| Epochs | Model  | Method         | Val Loss | MSE    | PSNR  | SSIM  | FID   | Inception Score | Uncertainty |
|--------|--------|----------------|----------|--------|-------|-------|-------|-----------------|-------------|
| 10     | VAE    | Fixed          | 1727.072 | 0.4028 | 14.21 | 0.154 | 75.74 | 2.15           | 0.0569      |
| 10     | LS-VAE | Fixed          | 1628.216 | 0.3975 | 14.27 | 0.162 | 71.14 | 2.16           | 0.0571      |
| 50     | VAE    | Fixed          | 1716.401 | 0.4024 | 14.22 | 0.154 | 70.24 | 2.15           | 0.0569      |
| 50     | AE     | Fixed          | 0.3945   | 0.3945 | 14.31 | 0.166 | 65.38 | 2.30           | 0.0571      |
| 50     | LS-VAE | Fixed          | 1626.530 | 0.3971 | 14.28 | 0.162 | 67.95 | 2.16           | 0.0570      |
| 100    | VAE    | Fixed          | 1714.269 | 0.4020 | 14.22 | 0.155 | 70.58 | 2.15           | 0.0568      |
| 100    | AE     | Fixed          | 0.3944   | 0.3944 | 14.31 | 0.166 | 65.68 | 2.29           | 0.0571      |
| 100    | LS-VAE | Fixed          | 1626.243 | 0.3970 | 14.28 | 0.162 | 67.76 | 2.19           | 0.0570      |
| 95     | VAE    | Early Stopping | 1712.910 | 0.4018 | 14.22 | 0.156 | 70.10 | 2.14           | 0.0568      |
| 21     | AE     | Early Stopping | 0.3946   | 0.3946 | 14.31 | 0.166 | 63.73 | 2.28           | 0.0571      |
| 21     | LS-VAE | Early Stopping | 1628.238 | 0.3975 | 14.27 | 0.163 | 67.08 | 2.17           | 0.0571      |

### Key Insights

**Performance Ranking**: AE > LS-VAE > VAE

**Best Results by Metric**:
- **Lowest FID**: AE (Early Stopping, 21 epochs) - 63.73
- **Highest PSNR**: AE (All variants) - ~14.31
- **Highest SSIM**: AE (All variants) - ~0.166
- **Best Inception Score**: AE (Fixed 50 epochs) - 2.30

**Early Stopping Effectiveness**:
- **AE**: Achieves best FID (63.73) in just 21 epochs vs 65.38 at 50 epochs
- **VAE**: Converges well at 95 epochs, close to 100-epoch performance
- **LS-VAE**: Stops too early at 21 epochs, requires tuning for optimal results

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