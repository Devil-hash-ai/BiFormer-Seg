import torch
import torch.nn as nn
import math
from typing import Optional


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    """
    Fills the input tensor with values drawn from a truncated normal distribution.
    """
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def init_linear(module: nn.Linear, std: float = 0.02):
    trunc_normal_(module.weight, std=std)
    if module.bias is not None:
        nn.init.constant_(module.bias, 0)


def init_conv(module: nn.Conv2d, std: Optional[float] = None):
    nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
    if module.bias is not None:
        nn.init.constant_(module.bias, 0)


def init_norm(module: nn.Module):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, 1.0)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, 0.0)


def init_module_weights(module: nn.Module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            init_linear(m)
        elif isinstance(m, nn.Conv2d):
            init_conv(m)
        elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
            init_norm(m)