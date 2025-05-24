import torch
import torch.nn as nn
import math
from typing import Tuple, Type, Optional, List


class MultiScaleAttention(nn.Module):
    def __init__(self, dim: int, heads: int, scale_ratio: int = 1):
        super().__init__()
        assert dim % heads == 0, "Embedding dim must be divisible by num_heads"
        self.heads = heads
        self.dim = dim
        self.inner_dim = dim // scale_ratio

        self.to_q = nn.Linear(dim, self.inner_dim)
        self.to_k = nn.Linear(dim, self.inner_dim)
        self.to_v = nn.Linear(dim, self.inner_dim)
        self.to_out = nn.Linear(self.inner_dim, dim)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        b, nq, _ = q.shape
        nk = k.shape[1]
        q = self.to_q(q).reshape(b, nq, self.heads, -1).transpose(1, 2)  # B x H x Nq x Dh
        k = self.to_k(k).reshape(b, nk, self.heads, -1).transpose(1, 2)
        v = self.to_v(v).reshape(b, nk, self.heads, -1).transpose(1, 2)

        scale = q.size(-1) ** -0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(b, nq, -1)
        return self.to_out(out)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, activation: Type[nn.Module] = nn.ReLU):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            activation(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x):
        return self.net(x)


class DualAttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        mlp_ratio: float = 4.0,
        activation: Type[nn.Module] = nn.ReLU,
        scale_ratio: int = 1,
        with_cnn_cross: bool = True
    ):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_k = nn.LayerNorm(dim)
        self.self_attn = MultiScaleAttention(dim, heads, scale_ratio)
        self.cross_attn_qk = MultiScaleAttention(dim, heads, scale_ratio)
        self.cross_attn_kq = MultiScaleAttention(dim, heads, scale_ratio)
        self.ffn_q = FeedForward(dim, int(dim * mlp_ratio), activation)
        self.ffn_k = FeedForward(dim, int(dim * mlp_ratio), activation)
        self.with_cnn_cross = with_cnn_cross
        if with_cnn_cross:
            self.cross_attn_cnn = MultiScaleAttention(dim, heads, scale_ratio)
            self.norm_cnn = nn.LayerNorm(dim)

    def forward(self, q, k, q_pe, k_pe, cnn_k):
        q = self.norm_q(q + q_pe)
        k = self.norm_k(k + k_pe)

        q2 = self.self_attn(q, q, q)
        q = q + q2
        q = q + self.cross_attn_qk(q + q_pe, k + k_pe, k)
        q = q + self.ffn_q(q)

        if self.with_cnn_cross:
            cnn_k = self.norm_cnn(cnn_k)
            k2 = self.cross_attn_cnn(k, cnn_k, cnn_k)
            k = k + k2

        k = k + self.cross_attn_kq(k + k_pe, q + q_pe, q)
        k = k + self.ffn_k(k)
        return q, k


class TransformerFusionModule(nn.Module):
    def __init__(
        self,
        depth: int,
        dim: int,
        heads: int,
        mlp_ratio: float = 4.0,
        activation: Type[nn.Module] = nn.ReLU,
        scale_ratio: int = 2
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            DualAttentionBlock(
                dim, heads, mlp_ratio, activation, scale_ratio, with_cnn_cross=True
            ) for _ in range(depth)
        ])
        self.final_cross_attn = MultiScaleAttention(dim, heads, scale_ratio)
        self.norm_final = nn.LayerNorm(dim)

    def forward(self, image_feat: torch.Tensor, image_pe: torch.Tensor, query_feat: torch.Tensor, cnn_feat: torch.Tensor):
        B, C, H, W = image_feat.shape
        img_tokens = image_feat.flatten(2).permute(0, 2, 1)
        pe_tokens = image_pe.flatten(2).permute(0, 2, 1)
        cnn_tokens = cnn_feat.flatten(2).permute(0, 2, 1)
        q_tokens = query_feat

        for block in self.blocks:
            q_tokens, img_tokens = block(q_tokens, img_tokens, query_feat, pe_tokens, cnn_tokens)

        q_tokens = q_tokens + self.final_cross_attn(q_tokens + query_feat, img_tokens + pe_tokens, img_tokens)
        q_tokens = self.norm_final(q_tokens)
        return q_tokens, img_tokens


if __name__ == '__main__':
    B, C, H, W = 2, 256, 16, 16
    image_feat = torch.randn(B, C, H, W)
    image_pe = torch.randn(B, C, H, W)
    query_feat = torch.randn(B, 64, C)
    cnn_feat = torch.randn(B, C, H, W)

    fusion_model = TransformerFusionModule(depth=4, dim=256, heads=8).eval()
    out_q, out_k = fusion_model(image_feat, image_pe, query_feat, cnn_feat)
    print("query output:", out_q.shape)
    print("image output:", out_k.shape)
