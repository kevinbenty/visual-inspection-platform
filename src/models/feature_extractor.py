"""Frozen ImageNet backbone that returns locally aggregated mid-level patch features."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm

_BACKBONES = {
    "wide_resnet50_2": (tvm.wide_resnet50_2, tvm.Wide_ResNet50_2_Weights.IMAGENET1K_V1),
    "resnet18": (tvm.resnet18, tvm.ResNet18_Weights.IMAGENET1K_V1),
}


class PatchFeatureExtractor(nn.Module):
    def __init__(self, backbone: str = "wide_resnet50_2",
                 layers: tuple[str, ...] = ("layer2", "layer3"), pretrained: bool = True):
        super().__init__()
        ctor, weights = _BACKBONES[backbone]
        net = ctor(weights=weights if pretrained else None)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.blocks = nn.ModuleDict({n: getattr(net, n) for n in ("layer1", "layer2", "layer3", "layer4")})
        self.layers = tuple(layers)
        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B,3,H,W) -> (B, C, h, w) feature map at the first layer's resolution."""
        feats, h = {}, self.stem(x)
        last = self.layers[-1]
        for name, block in self.blocks.items():
            h = block(h)
            if name in self.layers:
                feats[name] = h
            if name == last:
                break
        target = feats[self.layers[0]].shape[-2:]
        out = []
        for name in self.layers:
            f = F.avg_pool2d(feats[name], kernel_size=3, stride=1, padding=1)  # local neighbourhood
            if f.shape[-2:] != target:
                f = F.interpolate(f, size=target, mode="bilinear", align_corners=False)
            out.append(f)
        return torch.cat(out, dim=1)

    @torch.no_grad()
    def global_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Image-level embedding (used by the global kNN baseline)."""
        h = self.stem(x)
        for block in self.blocks.values():
            h = block(h)
        return torch.flatten(F.adaptive_avg_pool2d(h, 1), 1)
