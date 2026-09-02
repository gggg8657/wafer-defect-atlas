"""Small CNN encoder + classifier, with a Grad-CAM hook for defect localization."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class WaferCNN(nn.Module):
    def __init__(self, n_classes=4, width=16):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(1, width, 3, padding=1), nn.BatchNorm2d(width), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(width, width * 2, 3, padding=1), nn.BatchNorm2d(width * 2), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(width * 2, width * 4, 3, padding=1), nn.BatchNorm2d(width * 4), nn.ReLU(),
        )
        self.head = nn.Linear(width * 4, n_classes)
        self._feat = None

    def features(self, x):
        f = self.enc(x)
        self._feat = f
        return f

    def forward(self, x):
        f = self.features(x)
        pooled = F.adaptive_avg_pool2d(f, 1).flatten(1)
        return self.head(pooled)

    def embed(self, x):
        """Pooled feature vector — the seam for self-supervised (SimCLR) / clustering."""
        return F.adaptive_avg_pool2d(self.features(x), 1).flatten(1)


def grad_cam(model, x, cls=None):
    """Grad-CAM heatmap for one input map. Returns (H, W) in [0,1]."""
    model.eval()
    x = x.clone().requires_grad_(True)
    feat = model.features(x)
    feat.retain_grad()
    pooled = F.adaptive_avg_pool2d(feat, 1).flatten(1)
    logits = model.head(pooled)
    if cls is None:
        cls = logits.argmax(1)
    score = logits[range(len(x)), cls].sum()
    model.zero_grad()
    score.backward()
    weights = feat.grad.mean(dim=(2, 3), keepdim=True)   # channel importance
    cam = F.relu((weights * feat).sum(1))                 # (B, h, w)
    cam = F.interpolate(cam[:, None], size=x.shape[-2:], mode="bilinear",
                        align_corners=False)[:, 0]
    cam = cam - cam.amin(dim=(1, 2), keepdim=True)
    cam = cam / cam.amax(dim=(1, 2), keepdim=True).clamp_min(1e-8)
    return cam.detach()
