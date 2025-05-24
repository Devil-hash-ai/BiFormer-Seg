import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class UpsampleDecoderBlock(nn.Module):
    """
    Basic decoder block with ConvTranspose2d and normalization. Upsamples input by factor of 2.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class MultiScaleDecoder(nn.Module):
    """
    Stack of multiple upsample blocks from deep feature to full-resolution mask.
    """
    def __init__(self, base_dim: int = 256):
        super().__init__()
        self.up1 = UpsampleDecoderBlock(base_dim, base_dim // 2)    # 64 -> 128
        self.up2 = UpsampleDecoderBlock(base_dim // 2, base_dim // 4)  # 128 -> 256
        self.up3 = UpsampleDecoderBlock(base_dim // 4, base_dim // 8)  # 256 -> 512
        self.final_conv = nn.Conv2d(base_dim // 8, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        return self.final_conv(x)  # Output: [B, 1, H, W]


class AttentionDecoderMaskHead(nn.Module):
    """
    Optional attention-augmented token-to-mask upsampling decoder.
    Incorporates optional spatial attention for better alignment.
    """
    def __init__(self, token_dim: int, output_resolution: Tuple[int, int] = (256, 256)):
        super().__init__()
        self.token_dim = token_dim
        self.output_resolution = output_resolution
        self.spatial_map_proj = nn.Sequential(
            nn.Linear(token_dim, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, token_dim)
        )
        self.mask_projection = nn.Conv2d(token_dim, 1, kernel_size=1)

    def forward(self, tokens: torch.Tensor, feat_size: Tuple[int, int]) -> torch.Tensor:
        # tokens: [B, N, C] → reshape to spatial grid
        B, N, C = tokens.shape
        H, W = feat_size
        grid = tokens.permute(0, 2, 1).contiguous().view(B, C, H, W)
        mask_logits = self.mask_projection(grid)
        mask_logits = F.interpolate(mask_logits, size=self.output_resolution, mode="bilinear", align_corners=False)
        return mask_logits


class FullMaskDecoder(nn.Module):
    """
    High-level decoder that combines both conv-based and token-based decoding branches.
    """
    def __init__(self, token_dim: int = 256, output_resolution: Tuple[int, int] = (256, 256)):
        super().__init__()
        self.use_dual_path = True
        self.conv_decoder = MultiScaleDecoder(base_dim=token_dim)
        self.token_decoder = AttentionDecoderMaskHead(token_dim, output_resolution)
        self.fusion_head = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, fused_feat: torch.Tensor, token_feat: torch.Tensor) -> torch.Tensor:
        # fused_feat: [B, C, H, W] from image encoder
        # token_feat: [B, N, C] from fusion module (N ~ H*W)
        pred_conv = self.conv_decoder(fused_feat)  # [B, 1, 256, 256]

        B, C, H, W = fused_feat.shape
        pred_token = self.token_decoder(token_feat, feat_size=(H, W))  # [B, 1, 256, 256]

        out = torch.cat([pred_conv, pred_token], dim=1)
        return self.fusion_head(out)
