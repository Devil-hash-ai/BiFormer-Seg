import torch
import torch.nn as nn

# Binary Cross Entropy Loss with Logits (for binary segmentation tasks)
bce_loss = nn.BCEWithLogitsLoss()
