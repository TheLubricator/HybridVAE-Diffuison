# main.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import numpy as np
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
from skimage.metrics import peak_signal_noise_ratio as psnr
from torchmetrics.image import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore

from vanillaVAE import VAE, vae_loss
from latentStochasticVAE import LSVAE, lsvae_loss

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    # Transforms for MNIST
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])

    # Load MNIST dataset
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    val_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2, drop_last=True)

    # Function to visualize images
    def show_images(images, title=""):
        fig, axs = plt.subplots(1, min(8, len(images)), figsize=(12, 2))
        for i, img in enumerate(images[:8]):
            img = img.detach().cpu().numpy().squeeze() * 0.3081 + 0.1307  # Denormalization
            axs[i].imshow(img, cmap='gray')
            axs[i].axis('off')
        plt.suptitle(title)
        plt.savefig(f"{title.replace(' ', '_')}.png")
        plt.close()

    # Train VAE baseline
    vae = VAE().to(device)
    optimizer_vae = optim.Adam(vae.parameters(), lr=1e-3)

    print("Training Baseline VAE...")
    for epoch in range(10):
        vae.train()
        train_loss = 0
        for batch in train_loader:
            if isinstance(batch, list):
                batch = batch[0]
            batch = batch.to(device)
            optimizer_vae.zero_grad()
            recon, mu, logvar = vae(batch)
            loss = vae_loss(recon, batch, mu, logvar)
            loss.backward()
            optimizer_vae.step()
            train_loss += loss.item()
        print(f"Epoch {epoch+1}: VAE Train Loss = {train_loss / len(train_loader)}")
        if (epoch + 1) % 5 == 0:
            torch.save(vae.state_dict(), f'vae_checkpoint_epoch{epoch+1}.pt')

    # Evaluate VAE
    vae.eval()
    vae_mse = []
    vae_psnr = []
    vae_ssim = []
    vae_mse_per_batch = []  # For uncertainty calculation
    fid_metric_vae = FrechetInceptionDistance(normalize=True).to(device)
    is_metric_vae = InceptionScore(normalize=True).to(device)
    with torch.no_grad():
        val_loss = 0
        for batch in val_loader:
            if isinstance(batch, list):
                batch = batch[0]
            batch = batch.to(device)
            recon, mu, logvar = vae(batch)
            val_loss += vae_loss(recon, batch, mu, logvar).item()
            batch_denorm = batch * 0.3081 + 0.1307
            recon_denorm = recon * 0.3081 + 0.1307
            mse = nn.functional.mse_loss(recon, batch, reduction='mean').item()
            vae_mse.append(mse)
            vae_mse_per_batch.append(mse)  # Store MSE for each batch
            psnr_val = psnr(batch_denorm.cpu().numpy(), recon_denorm.cpu().numpy(), data_range=1.0)
            vae_psnr.append(psnr_val.mean() if psnr_val.size > 1 else psnr_val)
            ssim_metric = SSIM(data_range=1.0).to(device)
            ssim_score = ssim_metric(batch_denorm, recon_denorm).item()
            vae_ssim.append(ssim_score)
            batch_3ch = batch_denorm.repeat(1, 3, 1, 1)
            recon_3ch = recon_denorm.repeat(1, 3, 1, 1)
            fid_metric_vae.update(batch_3ch, real=True)
            fid_metric_vae.update(recon_3ch, real=False)
            is_metric_vae.update(recon_3ch)
        vae_uncertainty = np.std(vae_mse_per_batch)  # Uncertainty as standard deviation of MSE
        print(f"VAE Val Loss = {val_loss / len(val_loader)}")
        print(f"VAE Mean MSE = {np.mean(vae_mse)}")
        print(f"VAE Mean PSNR = {np.mean(vae_psnr)}")
        print(f"VAE Mean SSIM = {np.mean(vae_ssim)}")
        print(f"VAE FID = {fid_metric_vae.compute().item()}")
        print(f"VAE Inception Score = {is_metric_vae.compute()[0].item()}")
        print(f"VAE Uncertainty = {vae_uncertainty:.4f}")
        z_sample = torch.randn(8, 4, 8, 8).to(device)
        gen_samples = vae.decoder(z_sample)
        show_images(gen_samples, "VAE Generated Samples")

    # Train LS-VAE
    lsvae = LSVAE().to(device)
    optimizer_lsvae = optim.Adam(lsvae.parameters(), lr=1e-3)

    print("Training LS-VAE...")
    for epoch in range(10):
        lsvae.train()
        train_loss = 0
        for batch in train_loader:
            if isinstance(batch, list):
                batch = batch[0]
            batch = batch.to(device)
            optimizer_lsvae.zero_grad()
            recon, mu, logvar = lsvae(batch, epoch=epoch)
            loss = lsvae_loss(recon, batch, mu, logvar, epoch=epoch, beta=0.5)
            loss.backward()
            optimizer_lsvae.step()
            train_loss += loss.item()
        print(f"Epoch {epoch+1}: LS-VAE Train Loss = {train_loss / len(train_loader)}")
        if (epoch + 1) % 5 == 0:
            torch.save(lsvae.state_dict(), f'lsvae_checkpoint_epoch{epoch+1}.pt')

    # Evaluate LS-VAE
    lsvae.eval()
    lsvae_mse = []
    lsvae_psnr = []
    lsvae_ssim = []
    lsvae_mse_per_batch = []  # For uncertainty calculation
    fid_metric_lsvae = FrechetInceptionDistance(normalize=True).to(device)
    is_metric_lsvae = InceptionScore(normalize=True).to(device)
    with torch.no_grad():
        val_loss = 0
        for batch in val_loader:
            if isinstance(batch, list):
                batch = batch[0]
            batch = batch.to(device)
            recon, mu, logvar = lsvae(batch)
            val_loss += lsvae_loss(recon, batch, mu, logvar).item()
            batch_denorm = batch * 0.3081 + 0.1307
            recon_denorm = recon * 0.3081 + 0.1307
            mse = nn.functional.mse_loss(recon, batch, reduction='mean').item()
            lsvae_mse.append(mse)
            lsvae_mse_per_batch.append(mse)  # Store MSE for each batch
            psnr_val = psnr(batch_denorm.cpu().numpy(), recon_denorm.cpu().numpy(), data_range=1.0)
            lsvae_psnr.append(psnr_val.mean() if psnr_val.size > 1 else psnr_val)
            ssim_metric = SSIM(data_range=1.0).to(device)
            ssim_score = ssim_metric(batch_denorm, recon_denorm).item()
            lsvae_ssim.append(ssim_score)
            batch_3ch = batch_denorm.repeat(1, 3, 1, 1)
            recon_3ch = recon_denorm.repeat(1, 3, 1, 1)
            fid_metric_lsvae.update(batch_3ch, real=True)
            fid_metric_lsvae.update(recon_3ch, real=False)
            is_metric_lsvae.update(recon_3ch)
        lsvae_uncertainty = np.std(lsvae_mse_per_batch)  # Uncertainty as standard deviation of MSE
        print(f"LS-VAE Val Loss = {val_loss / len(val_loader)}")
        print(f"LS-VAE Mean MSE = {np.mean(lsvae_mse)}")
        print(f"LS-VAE Mean PSNR = {np.mean(lsvae_psnr)}")
        print(f"LS-VAE Mean SSIM = {np.mean(lsvae_ssim)}")
        print(f"LS-VAE FID = {fid_metric_lsvae.compute().item()}")
        print(f"LS-VAE Inception Score = {is_metric_lsvae.compute()[0].item()}")
        print(f"LS-VAE Uncertainty = {lsvae_uncertainty:.4f}")
        z_sample = torch.randn(8, 4, 8, 8).to(device)
        gen_samples = lsvae.decoder(z_sample)
        show_images(gen_samples, "LS-VAE Generated Samples")

    # Generate individual bar charts for each metric with exact values
    metrics = ['MSE', 'PSNR', 'SSIM', 'FID', 'Inception Score', 'Uncertainty']
    vae_values = [np.mean(vae_mse), np.mean(vae_psnr), np.mean(vae_ssim), fid_metric_vae.compute().item(), is_metric_vae.compute()[0].item(), vae_uncertainty]
    lsvae_values = [np.mean(lsvae_mse), np.mean(lsvae_psnr), np.mean(lsvae_ssim), fid_metric_lsvae.compute().item(), is_metric_lsvae.compute()[0].item(), lsvae_uncertainty]

    os.makedirs('charts', exist_ok=True)

    # MSE Chart
    plt.figure(figsize=(6, 4))
    bars = plt.bar(['VAE', 'LS-VAE'], [vae_values[0], lsvae_values[0]], color=['#1f77b4', '#ff7f0e'])
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.4f}',
                 ha='center', va='bottom')
    plt.ylabel('MSE')
    plt.title('VAE vs LS-VAE MSE Comparison')
    plt.savefig('charts/mse_comparison_bar_chart.png')
    plt.close()

    # PSNR Chart
    plt.figure(figsize=(6, 4))
    bars = plt.bar(['VAE', 'LS-VAE'], [vae_values[1], lsvae_values[1]], color=['#1f77b4', '#ff7f0e'])
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.2f}',
                 ha='center', va='bottom')
    plt.ylabel('PSNR (dB)')
    plt.title('VAE vs LS-VAE PSNR Comparison')
    plt.savefig('charts/psnr_comparison_bar_chart.png')
    plt.close()

    # SSIM Chart
    plt.figure(figsize=(6, 4))
    bars = plt.bar(['VAE', 'LS-VAE'], [vae_values[2], lsvae_values[2]], color=['#1f77b4', '#ff7f0e'])
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.4f}',
                 ha='center', va='bottom')
    plt.ylabel('SSIM')
    plt.title('VAE vs LS-VAE SSIM Comparison')
    plt.savefig('charts/ssim_comparison_bar_chart.png')
    plt.close()

    # FID Chart
    plt.figure(figsize=(6, 4))
    bars = plt.bar(['VAE', 'LS-VAE'], [vae_values[3], lsvae_values[3]], color=['#1f77b4', '#ff7f0e'])
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.2f}',
                 ha='center', va='bottom')
    plt.ylabel('FID')
    plt.title('VAE vs LS-VAE FID Comparison')
    plt.savefig('charts/fid_comparison_bar_chart.png')
    plt.close()

    # Inception Score Chart
    plt.figure(figsize=(6, 4))
    bars = plt.bar(['VAE', 'LS-VAE'], [vae_values[4], lsvae_values[4]], color=['#1f77b4', '#ff7f0e'])
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.2f}',
                 ha='center', va='bottom')
    plt.ylabel('Inception Score')
    plt.title('VAE vs LS-VAE Inception Score Comparison')
    plt.savefig('charts/inception_score_comparison_bar_chart.png')
    plt.close()

    # Uncertainty Chart
    plt.figure(figsize=(6, 4))
    bars = plt.bar(['VAE', 'LS-VAE'], [vae_values[5], lsvae_values[5]], color=['#1f77b4', '#ff7f0e'])
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.4f}',
                 ha='center', va='bottom')
    plt.ylabel('Uncertainty')
    plt.title('VAE vs LS-VAE Uncertainty Comparison')
    plt.savefig('charts/uncertainty_comparison_bar_chart.png')
    plt.close()

    print("Individual bar charts with exact values saved to 'charts/' directory")

if __name__ == '__main__':
    print(torch.__version__)
    print(torch.cuda.is_available())
    print(torch.version.cuda)
    main()