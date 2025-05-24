import torch
import torch.nn as nn
import torch.nn.functional as F


class IoUPredictor(nn.Module):
    """
    Predicts IoU confidence scores for each token embedding.
    Typically used to estimate mask quality before selection.
    """
    def __init__(self, token_dim: int = 256, hidden_dim: int = 512, multi_token: bool = True):
        super().__init__()
        self.multi_token = multi_token
        self.predictor = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: shape [B, N, D] (N tokens per sample)
        Returns:
            scores: [B, N] if multi_token else [B]
        """
        B, N, D = tokens.shape
        logits = self.predictor(tokens).squeeze(-1)  # [B, N]
        if self.multi_token:
            return torch.sigmoid(logits)  # [B, N]
        else:
            return torch.sigmoid(logits.mean(dim=1))  # [B]


class IoUSelector(nn.Module):
    """
    Optional: selects the best scoring token or masks based on IoU prediction.
    """
    def __init__(self, top_k: int = 1):
        super().__init__()
        self.top_k = top_k

    def forward(self, iou_scores: torch.Tensor, mask_preds: torch.Tensor) -> torch.Tensor:
        """
        Args:
            iou_scores: [B, N]  predicted IoU for N tokens
            mask_preds: [B, N, H, W]  predicted masks for each token
        Returns:
            selected_mask: [B, 1, H, W]  mask with highest IoU
        """
        B, N, H, W = mask_preds.shape
        topk_indices = torch.topk(iou_scores, self.top_k, dim=1).indices  # [B, K]
        selected_masks = torch.gather(mask_preds, dim=1, index=topk_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W))
        return selected_masks.mean(dim=1, keepdim=True)  # [B, 1, H, W]
