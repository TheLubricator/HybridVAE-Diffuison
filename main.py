# main.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from vanillaVAE import VAE, vae_loss
from hybridVAE import linear_beta_schedule, forward_diffusion, UNet, diffusion_loss, consistency_loss, HybridVAEDiffusion, alphas, alphas_cumprod
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
from skimage.metrics import peak_signal_noise_ratio as psnr
from torchmetrics.image import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    # Transforms for MNIST
    transform = transforms.Compose([
        transforms.Resize(64),  # Optional: match CelebA size
        transforms.ToTensor(),  # Converts image to tensor
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])

    # Load MNIST dataset
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    val_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4, drop_last=True)

    # Function to visualize images
    def show_images(images, title=""):
        fig, axs = plt.subplots(1, min(8, len(images)), figsize=(12, 2))
        for i, img in enumerate(images[:8]):
            img = img.detach().cpu().permute(1, 2, 0).numpy() * 0.3081 + 0.1307  # Denormalize correctly
            axs[i].imshow(img.squeeze(), cmap='gray')  # Use grayscale for MNIST
            axs[i].axis('off')
        plt.suptitle(title)
        plt.savefig(f"{title.replace(' ', '_')}.png")
        plt.close()

    # Train VAE baseline
    vae = VAE().to(device)
    optimizer_vae = optim.Adam(vae.parameters(), lr=1e-3)

    print("Training Baseline VAE...")
    for epoch in range(20):
        vae.train()
        train_loss = 0
        iteration = 0
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
            iteration += 1
        print(f"Epoch {epoch+1}: VAE Train Loss = {train_loss / len(train_loader)}")
        if (epoch + 1) % 5 == 0:
            torch.save(vae.state_dict(), f'vae_checkpoint_epoch{epoch+1}.pt')

    # Evaluate VAE
    vae.eval()
    vae_mse = []
    vae_psnr = []
    vae_ssim = []
    fid_metric = FrechetInceptionDistance(normalize=True).to(device)
    is_metric = InceptionScore(normalize=True).to(device)
    with torch.no_grad():
        val_loss = 0
        for batch in val_loader:
            if isinstance(batch, list):
                batch = batch[0]
            batch = batch.to(device)
            recon, mu, logvar = vae(batch)
            val_loss += vae_loss(recon, batch, mu, logvar).item()
            # Denormalize for metrics
            batch_denorm = batch * 0.3081 + 0.1307
            recon_denorm = recon * 0.3081 + 0.1307
            # MSE
            mse = nn.functional.mse_loss(recon, batch, reduction='mean').item()
            vae_mse.append(mse)
            # PSNR
            psnr_val = psnr(batch_denorm.cpu().numpy(), recon_denorm.cpu().numpy(), data_range=1.0)
            vae_psnr.append(psnr_val.mean() if psnr_val.size > 1 else psnr_val)
            # SSIM
            ssim_metric = SSIM(data_range=1.0).to(device)
            ssim_score = ssim_metric(batch_denorm, recon_denorm).item()
            vae_ssim.append(ssim_score)
            # FID and IS (3-channel conversion)
            batch_3ch = batch_denorm.repeat(1, 3, 1, 1)
            recon_3ch = recon_denorm.repeat(1, 3, 1, 1)
            fid_metric.update(batch_3ch, real=True)
            fid_metric.update(recon_3ch, real=False)
            is_metric.update(recon_3ch)
        print(f"VAE Val Loss = {val_loss / len(val_loader)}")
        print(f"VAE Mean MSE = {np.mean(vae_mse)}")
        print(f"VAE Mean PSNR = {np.mean(vae_psnr)}")
        print(f"VAE Mean SSIM = {np.mean(vae_ssim)}")
        print(f"VAE FID = {fid_metric.compute().item()}")
        print(f"VAE Inception Score = {is_metric.compute()[0].item()}")
        z_sample = torch.randn(8, 4, 8, 8).to(device)
        gen_samples = vae.decoder(z_sample)
        show_images(gen_samples, "VAE Generated Samples")

    # Train Hybrid
    hybrid = HybridVAEDiffusion(vae).to(device)
    optimizer_hybrid = optim.Adam(hybrid.unet.parameters(), lr=1e-4)

    print("Training Hybrid Model...")
    for epoch in range(50):
        hybrid.train()
        train_loss = 0
        for batch in train_loader:
            if isinstance(batch, list):
                batch = batch[0]
            batch = batch.to(device)
            with torch.no_grad():
                mu, logvar = vae.encoder(batch)
                z = vae.reparameterize(mu, logvar)
            optimizer_hybrid.zero_grad()
            diff_loss = diffusion_loss(hybrid.unet, z, device=device)
            cons_loss = consistency_loss(vae, z, device=device)
            loss = diff_loss + 0.1 * cons_loss
            loss.backward()
            optimizer_hybrid.step()
            train_loss += loss.item()
        print(f"Epoch {epoch+1}: Hybrid Train Loss = {train_loss / len(train_loader)}")

    # Evaluate Hybrid
    hybrid.eval()
    hybrid_mse = []
    hybrid_psnr = []
    hybrid_ssim = []
    fid_metric = FrechetInceptionDistance(normalize=True).to(device)
    is_metric = InceptionScore(normalize=True).to(device)
    with torch.no_grad():
        val_loss = 0
        for batch in val_loader:
            if isinstance(batch, list):
                batch = batch[0]
            batch = batch.to(device)
            recon = hybrid(batch)
            val_loss += nn.functional.mse_loss(recon, batch).item()
            # Denormalize for metrics
            batch_denorm = batch * 0.3081 + 0.1307
            recon_denorm = recon * 0.3081 + 0.1307
            # MSE
            mse = nn.functional.mse_loss(recon, batch, reduction='mean').item()
            hybrid_mse.append(mse)
            # PSNR
            psnr_val = psnr(batch_denorm.cpu().numpy(), recon_denorm.cpu().numpy(), data_range=1.0)
            hybrid_psnr.append(psnr_val.mean() if psnr_val.size > 1 else psnr_val)
            # SSIM
            ssim_metric = SSIM(data_range=1.0).to(device)
            ssim_score = ssim_metric(batch_denorm, recon_denorm).item()
            hybrid_ssim.append(ssim_score)
            # FID and IS (3-channel conversion)
            batch_3ch = batch_denorm.repeat(1, 3, 1, 1)
            recon_3ch = recon_denorm.repeat(1, 3, 1, 1)
            fid_metric.update(batch_3ch, real=True)
            fid_metric.update(recon_3ch, real=False)
            is_metric.update(recon_3ch)
        print(f"Hybrid Val MSE = {val_loss / len(val_loader)}")
        print(f"Hybrid Mean MSE = {np.mean(hybrid_mse)}")
        print(f"Hybrid Mean PSNR = {np.mean(hybrid_psnr)}")
        print(f"Hybrid Mean SSIM = {np.mean(hybrid_ssim)}")
        print(f"Hybrid FID = {fid_metric.compute().item()}")
        print(f"Hybrid Inception Score = {is_metric.compute()[0].item()}")
    z = torch.randn(8, 4, 8, 8).to(device)  # 8 samples, latent dim 4x8x8
    gen_samples = hybrid.generate(z)
    show_images(gen_samples, "Hybrid Generated Samples")

    # Generate bar charts for comparison
    metrics = ['MSE', 'PSNR', 'SSIM', 'FID', 'Inception Score']
    vae_values = [np.mean(vae_mse), np.mean(vae_psnr), np.mean(vae_ssim), fid_metric.compute().item(), is_metric.compute()[0].item()]
    hybrid_values = [np.mean(hybrid_mse), np.mean(hybrid_psnr), np.mean(hybrid_ssim), fid_metric.compute().item(), is_metric.compute()[0].item()]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(metrics))
    width = 0.35

    ax.bar(x - width/2, vae_values, width, label='VAE', color='#1f77b4')
    ax.bar(x + width/2, hybrid_values, width, label='Hybrid', color='#ff7f0e')

    ax.set_ylabel('Value')
    ax.set_title('VAE vs Hybrid Metrics Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()

    # Save to local folder
    os.makedirs('charts', exist_ok=True)  # Create charts directory if it doesn't exist
    plt.savefig('charts/metrics_comparison_bar_chart.png')
    plt.close()
    print("Bar chart saved to 'charts/metrics_comparison_bar_chart.png'")

    # Uncertainty: Generate multiple from same z for VAE
    with torch.no_grad():
        batch = next(iter(val_loader))
        if isinstance(batch, list):
            batch = batch[0]
        batch = batch.to(device)[0:1]
        mu, logvar = vae.encoder(batch)
        z = vae.reparameterize(mu, logvar)
        vae_variants = [vae.decoder(z) for _ in range(5)]
        vae_var = torch.var(torch.stack(vae_variants), dim=0).mean().item()
        print(f"VAE Uncertainty Variance: {vae_var}")

    # Uncertainty: Generate multiple from same z for Hybrid
    with torch.no_grad():
        batch = next(iter(val_loader))
        if isinstance(batch, list):
            batch = batch[0]
        batch = batch.to(device)[0:1]
        mu, logvar = vae.encoder(batch)
        z = vae.reparameterize(mu, logvar)
        def hybrid_sample(z):
            z_ = z.clone()
            steps = 1000
            for t in reversed(range(steps)):
                t_tensor = torch.full((z_.size(0),), t, dtype=torch.long).to(device)
                pred_noise = hybrid.unet(z_, t_tensor)
                alpha_t = hybrid.alphas[t]
                acp_t = hybrid.alphas_cumprod[t]
                z_ = (z_ - (1 - alpha_t) / torch.sqrt(1 - acp_t) * pred_noise) / torch.sqrt(alpha_t)
                if t > 0:
                    sigma_t = torch.sqrt((1 - alpha_t) * (1 - hybrid.alphas_cumprod[t-1]) / (1 - acp_t))
                    z_ += sigma_t * torch.randn_like(z_)
            return hybrid.vae.decoder(z_)
        hybrid_variants = [hybrid_sample(z) for _ in range(5)]
        hybrid_var = torch.var(torch.stack(hybrid_variants), dim=0).mean().item()
        print(f"Hybrid Uncertainty Variance: {hybrid_var}")

    # Stability: Run with different seeds and average uncertainty metrics
    vae_vars = []
    hybrid_vars = []
    for seed in range(5):
        torch.manual_seed(seed)
        # VAE uncertainty
        with torch.no_grad():
            batch = next(iter(val_loader))
            if isinstance(batch, list):
                batch = batch[0]
            batch = batch.to(device)[0:1]
            mu, logvar = vae.encoder(batch)
            z = vae.reparameterize(mu, logvar)
            vae_variants = [vae.decoder(z) for _ in range(5)]
            vae_var = torch.var(torch.stack(vae_variants), dim=0).mean().item()
            vae_vars.append(vae_var)
        # Hybrid uncertainty
        with torch.no_grad():
            batch = next(iter(val_loader))
            if isinstance(batch, list):
                batch = batch[0]
            batch = batch.to(device)[0:1]
            mu, logvar = vae.encoder(batch)
            z = vae.reparameterize(mu, logvar)
            def hybrid_sample(z):
                z_ = z.clone()
                steps = 1000
                for t in reversed(range(steps)):
                    t_tensor = torch.full((z_.size(0),), t, dtype=torch.long).to(device)
                    pred_noise = hybrid.unet(z_, t_tensor)
                    alpha_t = hybrid.alphas[t]
                    acp_t = hybrid.alphas_cumprod[t]
                    z_ = (z_ - (1 - alpha_t) / torch.sqrt(1 - acp_t) * pred_noise) / torch.sqrt(alpha_t)
                    if t > 0:
                        sigma_t = torch.sqrt((1 - alpha_t) * (1 - hybrid.alphas_cumprod[t-1]) / (1 - acp_t))
                        z_ += sigma_t * torch.randn_like(z_)
                return hybrid.vae.decoder(z_)
            hybrid_variants = [hybrid_sample(z) for _ in range(5)]
            hybrid_var = torch.var(torch.stack(hybrid_variants), dim=0).mean().item()
            hybrid_vars.append(hybrid_var)
        print(f"Seed {seed} | VAE Uncertainty: {vae_var:.6f} | Hybrid Uncertainty: {hybrid_var:.6f}")
    print(f"VAE Uncertainty Mean: {np.mean(vae_vars):.6f}, Std: {np.std(vae_vars):.6f}")
    print(f"Hybrid Uncertainty Mean: {np.mean(hybrid_vars):.6f}, Std: {np.std(hybrid_vars):.6f}")

if __name__ == '__main__':
    print(torch.__version__)
    print(torch.cuda.is_available())
    print(torch.version.cuda)
    main()