import torch
import torch.nn as nn
from typing import Dict, Tuple

from token_encoder import TokenPromptEncoder
from cross_fusion import MultiStageFusion
from mask_upsampler import FullMaskDecoder
from iou_predictor import IoUPredictor


class UnifiedMaskDecoder(nn.Module):
    """
    The complete decoder module that integrates:
    - Text encoder (prompt tokens)
    - Image-text fusion
    - Multi-scale decoder (upsampling path)
    - IoU confidence prediction
    """
    def __init__(self,
                 text_model: str = "microsoft/BiomedVLP-CXR-BERT-general",
                 token_dim: int = 256,
                 fusion_depth: int = 4,
                 iou_predict: bool = True):
        super().__init__()
        self.prompt_encoder = TokenPromptEncoder(model_name=text_model, out_dim=token_dim)
        self.fusion = MultiStageFusion(dim=token_dim, depth=fusion_depth)
        self.decoder = FullMaskDecoder(token_dim=token_dim)
        self.iou_predictor = IoUPredictor(token_dim=token_dim)
        self.use_iou = iou_predict

    def forward(self,
                image_feat: torch.Tensor,  # [B, C, H, W]
                texts: list                # list of B strings
                ) -> Dict[str, torch.Tensor]:
        B, C, H, W = image_feat.shape

        # 1. Encode textual prompt into [B, 1, D] token
        text_tokens = self.prompt_encoder(texts)       # [B, 1, D]

        # 2. Flatten image feat into token grid [B, HW, D]
        image_tokens = image_feat.flatten(2).permute(0, 2, 1)  # [B, HW, D]

        # 3. Multi-stage fusion between prompt token and image tokens
        fused_tokens, updated_image_tokens = self.fusion(text_tokens, image_tokens)  # [B, 1, D], [B, HW, D]

        # 4. Restore updated image tokens to spatial map
        fused_feat = updated_image_tokens.permute(0, 2, 1).view(B, C, H, W)  # [B, C, H, W]

        # 5. Decode final mask using both paths
        pred_mask = self.decoder(fused_feat, fused_tokens)  # [B, 1, H, W]

        out = {"mask": pred_mask}

        # 6. Optionally predict IoU score
        if self.use_iou:
            iou_score = self.iou_predictor(fused_tokens)  # [B, 1] or [B, N]
            out["iou"] = iou_score

        return out
