import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class CrossAttentionBlock(nn.Module):
    """
    Single cross-attention block (with optional gated control) between query and context features.
    """
    def __init__(self, dim: int, num_heads: int = 8, gated: bool = False):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        self.norm_ffn = nn.LayerNorm(dim)
        self.gated = gated
        if gated:
            self.gate = nn.Sequential(
                nn.Linear(dim, dim),
                nn.Sigmoid()
            )

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        residual = query
        attn_out, _ = self.attn(query, context, context)
        if self.gated:
            g = self.gate(query)
            attn_out = g * attn_out
        query = self.norm(residual + attn_out)

        # FFN
        ffn_out = self.ffn(query)
        query = self.norm_ffn(query + ffn_out)
        return query


class BidirectionalCrossAttention(nn.Module):
    """
    Bidirectional cross attention (query ↔ context) with independent branches.
    """
    def __init__(self, dim: int, num_heads: int = 8, gated: bool = True):
        super().__init__()
        self.token_to_image = CrossAttentionBlock(dim, num_heads, gated=gated)
        self.image_to_token = CrossAttentionBlock(dim, num_heads, gated=gated)

    def forward(self, token_feat: torch.Tensor, image_feat: torch.Tensor) -> torch.Tensor:
        token_out = self.token_to_image(token_feat, image_feat)
        image_out = self.image_to_token(image_feat, token_feat)
        return token_out, image_out


class CrossAttentionFusionStage(nn.Module):
    """
    One stage of iterative refinement: cross attention + FFN (stackable).
    """
    def __init__(self, dim: int, num_heads: int = 8, gated: bool = True):
        super().__init__()
        self.attn = BidirectionalCrossAttention(dim, num_heads=num_heads, gated=gated)

    def forward(self, tokens: torch.Tensor, image_tokens: torch.Tensor):
        tokens, image_tokens = self.attn(tokens, image_tokens)
        return tokens, image_tokens


class MultiStageFusion(nn.Module):
    """
    Multiple layers of bidirectional attention fusion (token ↔ image), with stage-wise stacking.
    """
    def __init__(self, dim: int, depth: int = 4, num_heads: int = 8, gated: bool = True):
        super().__init__()
        self.stages = nn.ModuleList([
            CrossAttentionFusionStage(dim=dim, num_heads=num_heads, gated=gated)
            for _ in range(depth)
        ])

    def forward(self, tokens: torch.Tensor, image_tokens: torch.Tensor):
        for stage in self.stages:
            tokens, image_tokens = stage(tokens, image_tokens)
        return tokens, image_tokens
