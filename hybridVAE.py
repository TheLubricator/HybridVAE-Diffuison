# hybridVAE.py
import torch
import torch.nn as nn
from vanillaVAE import *

def linear_beta_schedule(timesteps=500):
    beta_start = 1e-4
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

alphas = 1.0 - linear_beta_schedule()
alphas_cumprod = torch.cumprod(alphas, dim=0).to('cuda' if torch.cuda.is_available() else 'cpu')  # Move to GPU if available

def forward_diffusion(z, t, noise=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
    alphas_cumprod_device = alphas_cumprod.to(device)  # Ensure device consistency
    if noise is None:
        noise = torch.randn_like(z, device=device)
    sqrt_alphas_cumprod_t = torch.sqrt(alphas_cumprod_device[t]).view(-1, 1, 1, 1)
    sqrt_one_minus_alphas_cumprod_t = torch.sqrt(1. - alphas_cumprod_device[t]).view(-1, 1, 1, 1)
    return sqrt_alphas_cumprod_t * z + sqrt_one_minus_alphas_cumprod_t * noise, noise

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, latent_channels=4):
        super().__init__()
        self.time_emb = nn.Embedding(500, latent_channels).to('cuda' if torch.cuda.is_available() else 'cpu')
        self.down1 = ConvBlock(latent_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = ConvBlock(64, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.up2 = ConvBlock(64 + 64, 64)
        self.final = nn.Conv2d(64, latent_channels, kernel_size=1)

    def forward(self, x, t):
        t_emb = self.time_emb(t)[:, :, None, None].to(x.device)  # Move to x's device
        x = x + t_emb
        d1 = self.down1(x)
        d2 = self.pool1(d1)
        d3 = self.down2(d2)
        u1 = self.up1(d3)
        u1 = torch.cat([u1, d1], dim=1)
        u2 = self.up2(u1)
        return self.final(u2)

def diffusion_loss(model, z, timesteps=500, device='cuda' if torch.cuda.is_available() else 'cpu'):
    t = torch.randint(0, timesteps, (z.shape[0],)).to(device)
    noisy_z, noise = forward_diffusion(z, t, device=device)
    pred_noise = model(noisy_z, t)
    return nn.functional.mse_loss(pred_noise, noise)

def consistency_loss(vae, z, num_samples=5, device='cuda' if torch.cuda.is_available() else 'cpu'):
    variants = []
    for _ in range(num_samples):
        recon = vae.decoder(vae.reparameterize(*vae.encoder(vae.decoder(z))).to(device))
        variants.append(recon)
    var = torch.var(torch.stack(variants), dim=0).mean()
    return var

def custom_stochastic_loss(mu, logvar, beta=1.0):
    kl1 = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    mu2 = mu + 0.5
    kl2 = -0.5 * torch.sum(1 + logvar - mu2.pow(2) - logvar.exp())
    return beta * 0.5 * (kl1 + kl2)

class HybridVAEDiffusion(nn.Module):
    def __init__(self, vae, timesteps=500):
        super().__init__()
        self.vae = vae
        self.unet = UNet()
        self.timesteps = timesteps
        self.alphas = alphas.to(vae.encoder.conv1.conv[0].weight.device)
        self.alphas_cumprod = alphas_cumprod.to(vae.encoder.conv1.conv[0].weight.device)

    def forward(self, x):
        with torch.no_grad():
            mu, logvar = self.vae.encoder(x)
            z = self.vae.reparameterize(mu, logvar)
        return self.vae.decoder(z)

    def generate(self, z, steps=500, device='cuda' if torch.cuda.is_available() else 'cpu'):
        for t in range(steps - 1, -1, -1):
            t_tensor = torch.tensor([t] * z.shape[0], device=device)
            noisy_z, _ = forward_diffusion(z, t_tensor, device=device)
            pred_noise = self.unet(noisy_z, t_tensor.to(device))
            z = (noisy_z - pred_noise) / torch.sqrt(1. - alphas[t].to(device))
        z += 0.01 * torch.randn_like(z, device=device)  # Add noise on the correct device
        return self.vae.decoder(z)

def hybrid_loss(recon_x, x, mu, logvar, pred_noise, noise, cons_loss, beta=1.0, gamma=0.5):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
    stochastic_loss = custom_stochastic_loss(mu, logvar, beta)
    diff_loss = nn.functional.mse_loss(pred_noise, noise)
    return (recon_loss + stochastic_loss + diff_loss + gamma * cons_loss) / x.size(0)