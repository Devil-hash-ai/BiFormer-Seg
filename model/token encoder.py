import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from typing import List, Optional, Tuple


class FrozenBertTextEncoder(nn.Module):
    """
    Wrapper around a pretrained BERT-style encoder (e.g., BiomedVLP) with output projection.
    The model is frozen to prevent gradient updates unless explicitly enabled.
    """
    def __init__(self, model_name: str, out_dim: int, freeze: bool = True):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.out_proj = nn.Linear(self.model.config.hidden_size, out_dim)

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, texts: List[str]) -> torch.Tensor:
        tokens = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
        tokens = {k: v.to(self.out_proj.weight.device) for k, v in tokens.items()}

        with torch.no_grad():
            output = self.model(**tokens)
        cls_repr = output.last_hidden_state[:, 0, :]  # CLS token
        return self.out_proj(cls_repr)  # [B, D]


class PromptTokenRefiner(nn.Module):
    """
    Multiple feedforward refinement layers to evolve initial prompt token representations.
    """
    def __init__(self, dim: int, num_layers: int = 3):
        super().__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(nn.LayerNorm(dim))
            layers.append(nn.Linear(dim, dim))
            layers.append(nn.GELU())
        self.refiner = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.refiner(x)


class TokenPromptEncoder(nn.Module):
    """
    Encodes a list of text descriptions into refined token embeddings.
    Output: [B, 1, D]
    """
    def __init__(self, model_name: str = "microsoft/BiomedVLP-CXR-BERT-general", out_dim: int = 256, refine_depth: int = 3):
        super().__init__()
        self.text_encoder = FrozenBertTextEncoder(model_name=model_name, out_dim=out_dim)
        self.refiner = PromptTokenRefiner(dim=out_dim, num_layers=refine_depth)

    def forward(self, texts: List[str]) -> torch.Tensor:
        base_tokens = self.text_encoder(texts)  # [B, D]
        refined = self.refiner(base_tokens)     # [B, D]
        return refined.unsqueeze(1)             # [B, 1, D]