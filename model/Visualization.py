import cv2
import numpy as np
import torch
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from einops import rearrange
from matplotlib.ticker import FormatStrFormatter


def visualize_feature_channels(tensor, save_path_prefix, grid_size=(4, 4)):
    tensor = tensor[0].cpu().detach().numpy()  # (C, H, W)
    C, H, W = tensor.shape
    fig, axes = plt.subplots(grid_size[0], grid_size[1], figsize=(12, 12))
    for i in range(grid_size[0] * grid_size[1]):
        ax = axes[i // grid_size[1], i % grid_size[1]]
        if i < C:
            img = tensor[i]
            ax.imshow(img, cmap='viridis')
            ax.axis('off')
        else:
            ax.axis('off')
    plt.tight_layout()
    plt.savefig(f"{save_path_prefix}_feature_channels.png")


def visualize_amplitude(feature_tensor, save_path):
    feature = feature_tensor.cpu().detach().numpy()
    feature = np.squeeze(np.sum(feature, axis=1))
    freq = np.fft.fftshift(np.fft.fft2(feature))
    amplitude = np.log1p(np.abs(freq))
    normalized = ((amplitude - amplitude.min()) / (amplitude.max() - amplitude.min()) * 255).astype(np.uint8)
    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_BONE)
    cv2.imwrite(save_path, colored)


def fft_log_shifted(tensor):
    return torch.log(torch.fft.fft2(tensor).abs() + 1e-6)


def fft_shift(tensor):
    b, c, h, w = tensor.size()
    return torch.roll(tensor, shifts=(h // 2, w // 2), dims=(2, 3))


def visualize_latent_fourier(latents, labels, save_path):
    plt.figure(figsize=(5.5, 4), dpi=150)
    for i, latent in enumerate(latents):
        latent = latent.cpu()
        if latent.ndim == 3:
            b, n, c = latent.size()
            h = w = int(math.sqrt(n))
            latent = rearrange(latent, 'b (h w) c -> b c h w', h=h, w=w)
        elif latent.ndim != 4:
            raise ValueError(f"Unsupported tensor shape: {latent.shape}")

        freq = fft_log_shifted(latent)
        shifted = fft_shift(freq)
        diag = shifted.mean(dim=(0, 1)).diag()[h // 2:]
        diag -= diag[0]
        x_axis = np.linspace(0, 1, len(diag))
        plt.plot(x_axis, diag.numpy(), label=labels[i], color=cm.plasma_r(i / len(latents)))

    plt.legend()
    plt.xlabel("Frequency")
    plt.ylabel("Δ Log amplitude")
    plt.xlim(0, 1)
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.1fπ'))
    plt.tight_layout()
    plt.savefig(save_path)


if __name__ == '__main__':
    dummy_feature = torch.randn(1, 16, 64, 64)  # Simulate CNN feature map
    visualize_feature_channels(dummy_feature, 'feature_map')
    visualize_amplitude(dummy_feature, 'amplitude_output.png')
