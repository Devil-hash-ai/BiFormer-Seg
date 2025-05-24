import torch
import torch.nn as nn
import math
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks import DropPath
from mmcv.cnn.utils.weight_init import constant_init, normal_init, trunc_normal_init
from mmcv.runner import BaseModule


def select_mscan_cfg(cfg):
    levels = {
        'tiny': ([32, 64, 160, 256], [3, 3, 5, 2], 0.1),
        'small': ([64, 128, 320, 512], [2, 2, 4, 2], 0.1),
        'base': ([64, 128, 320, 512], [3, 3, 12, 3], 0.1),
        'large': ([64, 128, 320, 512], [3, 5, 27, 3], 0.3)
    }
    dims, depths, drop_path = levels[cfg.mscan]
    init_cfg = dict(type='Pretrained', checkpoint=cfg.mscan_checkpoint)
    return dims, depths, init_cfg, drop_path


class DepthwiseConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)

    def forward(self, x):
        return self.conv(x)


class MLPBlock(BaseModule):
    def __init__(self, in_dim, hidden_dim=None, out_dim=None, drop=0.):
        super().__init__()
        out_dim = out_dim or in_dim
        hidden_dim = hidden_dim or in_dim
        self.mlp = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, 1),
            DepthwiseConv(hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Conv2d(hidden_dim, out_dim, 1),
            nn.Dropout(drop)
        )

    def forward(self, x):
        return self.mlp(x)


class PatchEmbedding(BaseModule):
    def __init__(self, in_ch, out_ch, norm_cfg):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_ch, out_ch // 2, 3, 2, 1),
            build_norm_layer(norm_cfg, out_ch // 2)[1],
            nn.GELU(),
            nn.Conv2d(out_ch // 2, out_ch, 3, 2, 1),
            build_norm_layer(norm_cfg, out_ch)[1]
        )

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.size()
        return x, x.flatten(2).transpose(1, 2), H, W


class ConvAttention(BaseModule):
    def __init__(self, dim):
        super().__init__()
        self.depth = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv7x7 = nn.Sequential(
            nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim),
            nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)
        )
        self.conv11x11 = nn.Sequential(
            nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim),
            nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)
        )
        self.conv21x21 = nn.Sequential(
            nn.Conv2d(dim, dim, (1, 21), padding=(0, 10), groups=dim),
            nn.Conv2d(dim, dim, (21, 1), padding=(10, 0), groups=dim)
        )
        self.project = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        residual = x
        x = self.depth(x)
        x += self.conv7x7(x) + self.conv11x11(x) + self.conv21x21(x)
        x = self.project(x)
        return x * residual


class MSBlock(BaseModule):
    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., norm_cfg=dict(type='SyncBN', requires_grad=True)):
        super().__init__()
        self.norm1 = build_norm_layer(norm_cfg, dim)[1]
        self.attn = ConvAttention(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = build_norm_layer(norm_cfg, dim)[1]
        self.mlp = MLPBlock(dim, int(dim * mlp_ratio), drop=drop)
        self.scale1 = nn.Parameter(torch.ones(dim))
        self.scale2 = nn.Parameter(torch.ones(dim))

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = x + self.drop_path(self.scale1.view(-1, 1, 1) * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.scale2.view(-1, 1, 1) * self.mlp(self.norm2(x)))
        return x.flatten(2).transpose(1, 2)


class Stage(BaseModule):
    def __init__(self, idx, in_ch, out_ch, depth, mlp_ratio, dpr, norm_cfg):
        super().__init__()
        self.patch_embed = PatchEmbedding(in_ch, out_ch, norm_cfg)
        self.blocks = nn.ModuleList([
            MSBlock(out_ch, mlp_ratio, drop_path=dpr[i], norm_cfg=norm_cfg)
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(out_ch)

    def forward(self, x):
        x, tokens, H, W = self.patch_embed(x)
        for blk in self.blocks:
            tokens = blk(tokens, H, W)
        x = self.norm(tokens).transpose(1, 2).reshape(x.shape[0], -1, H, W)
        return x


class MSCANEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        dims, depths, init_cfg, drop_path = select_mscan_cfg(cfg)
        self.stages = nn.ModuleList()
        dpr = torch.linspace(0, drop_path, sum(depths)).tolist()
        cur = 0
        in_ch = 3
        for i in range(len(dims)):
            stage = Stage(i, in_ch, dims[i], depths[i], 4, dpr[cur:cur+depths[i]], dict(type='SyncBN', requires_grad=True))
            self.stages.append(stage)
            in_ch = dims[i]
            cur += depths[i]

    def forward(self, x):
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return features


if __name__ == '__main__':
    class DummyCfg:
        mscan = 'base'
        mscan_checkpoint = ''

    model = MSCANEncoder(DummyCfg()).cuda()
    dummy_input = torch.randn(2, 3, 224, 224).cuda()
    outputs = model(dummy_input)
    for i, feat in enumerate(outputs):
        print(f'Stage {i + 1}: {feat.shape}')
