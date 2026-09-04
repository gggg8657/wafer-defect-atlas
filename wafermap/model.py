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


# --------------------------------------------------------------------------
# The real-data encoder.  Same `features / forward / embed` protocol as
# WaferCNN above, so `grad_cam` works on either without a special case.
# --------------------------------------------------------------------------
class _Block(nn.Module):
    """Pre-activation residual block."""

    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(cin)
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.skip = (nn.Identity() if stride == 1 and cin == cout
                     else nn.Conv2d(cin, cout, 1, stride, bias=False))

    def forward(self, x):
        h = F.relu(self.bn1(x))
        out = self.c2(F.relu(self.bn2(self.c1(h))))
        return out + self.skip(h if isinstance(self.skip, nn.Conv2d) else x)


class WaferResNet(nn.Module):
    """~1.5M-parameter residual CNN for 64x64 two-channel wafer maps.

    Stops at an 8x8 feature map rather than pooling all the way down: Grad-CAM
    is only as sharp as the last conv layer, and 8x8 over a 64x64 wafer is
    ~5 mm per cell on a 300 mm wafer -- fine enough to tell a centre cluster
    from an edge ring, which is what the localization claim needs.
    """

    def __init__(self, n_out=8, in_ch=2, width=32, blocks=(2, 2, 2)):
        super().__init__()
        self.stem = nn.Conv2d(in_ch, width, 3, 1, 1, bias=False)
        layers, cin = [], width
        for i, nb in enumerate(blocks):
            cout = width * 2 ** i
            for b in range(nb):
                layers.append(_Block(cin, cout, stride=2 if b == 0 else 1))
                cin = cout
        self.blocks = nn.Sequential(*layers)
        self.bn = nn.BatchNorm2d(cin)
        self.n_feat = cin
        self.head = nn.Linear(cin, n_out)
        self.drop = nn.Dropout(0.1)

    def features(self, x):
        return F.relu(self.bn(self.blocks(self.stem(x))))

    def forward(self, x):
        f = self.features(x)
        return self.head(self.drop(F.adaptive_avg_pool2d(f, 1).flatten(1)))

    def embed(self, x):
        """Pooled features -- the SSL / clustering seam."""
        return F.adaptive_avg_pool2d(self.features(x), 1).flatten(1)
