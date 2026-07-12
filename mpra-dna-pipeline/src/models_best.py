"""
Best-in-class MPRA sequence model — "RegNetDNA".

Design, drawn from what actually wins on MPRA / regulatory-genomics tasks:

  * Wide first-layer conv (k=15)  -> learnable PWM-style motif detectors
    (Basset/DeepSEA/Malinois stem).
  * A tower of RESIDUAL DILATED conv blocks with SQUEEZE-EXCITE channel
    attention (LegNet / EfficientNet-1D lineage). Dilation grows 1,2,4,8 so a
    270 bp insert is covered by long-range "grammar" without exploding params.
  * Attention pooling + mean/max pooling -> the read-out is order-aware.
  * REVERSE-COMPLEMENT EQUIVARIANCE: an MPRA insert has no preferred strand, so
    the model scores a sequence and its reverse-complement through the SAME
    weights and averages. Provided as a wrapper so we can do it as cheap
    test-time augmentation during training and true averaging at inference.
  * Multi-task heads: one activity-regression + one is-active-logit head per
    assay (Primary, Organoid) over a shared trunk — the standard MPRA setup.

Sized so it trains on CPU in a couple of minutes per epoch, yet is a genuine
step up in capacity/inductive-bias over a plain 3-layer CNN.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class SqueezeExcite(nn.Module):
    def __init__(self, c, r=8):
        super().__init__()
        self.fc1 = nn.Linear(c, max(4, c // r))
        self.fc2 = nn.Linear(max(4, c // r), c)

    def forward(self, x):                       # x: (B, C, L)
        s = x.mean(-1)                          # global context
        s = F.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s)).unsqueeze(-1)
        return x * s


class ResidualDilatedBlock(nn.Module):
    def __init__(self, c, k=5, dilation=1, dropout=0.1):
        super().__init__()
        pad = (k // 2) * dilation
        self.conv1 = nn.Conv1d(c, c, k, padding=pad, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(c)
        self.conv2 = nn.Conv1d(c, c, k, padding=pad, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(c)
        self.se = SqueezeExcite(c)
        self.drop = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x):
        h = self.act(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        h = self.se(h)
        return self.act(x + self.drop(h))       # residual


class AttentionPool(nn.Module):
    """Learned softmax pooling over length (Enformer-style read-out)."""
    def __init__(self, c):
        super().__init__()
        self.w = nn.Conv1d(c, 1, 1)

    def forward(self, x):                        # (B, C, L)
        a = torch.softmax(self.w(x), dim=-1)     # (B, 1, L)
        return (x * a).sum(-1)                    # (B, C)


class RegNetDNA(nn.Module):
    def __init__(self, n_assays=2, channels=128, n_blocks=5,
                 stem_k=15, dilations=(1, 2, 4, 8, 1), dropout=0.2, hidden=256):
        super().__init__()
        self.n_assays = n_assays
        self.stem = nn.Sequential(
            nn.Conv1d(4, channels, stem_k, padding=stem_k // 2),
            nn.BatchNorm1d(channels), nn.GELU(),
            nn.MaxPool1d(2),                     # 270 -> 135, halves compute
        )
        dils = (dilations * ((n_blocks // len(dilations)) + 1))[:n_blocks]
        self.blocks = nn.ModuleList(
            [ResidualDilatedBlock(channels, dilation=d, dropout=dropout / 2)
             for d in dils]
        )
        self.attn = AttentionPool(channels)
        feat = channels * 3                      # attn + mean + max
        self.head = nn.Sequential(
            nn.Linear(feat, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Dropout(dropout),
        )
        self.reg = nn.Linear(hidden // 2, n_assays)
        self.cls = nn.Linear(hidden // 2, n_assays)

    def embed(self, x):
        h = self.stem(x)
        for b in self.blocks:
            h = b(h)
        return torch.cat([self.attn(h), h.mean(-1), h.amax(-1)], dim=-1)

    def forward(self, x, rc_average=False):
        f = self.embed(x)
        if rc_average:
            f_rc = self.embed(torch.flip(x, dims=(1, 2)))   # reverse-complement
            f = 0.5 * (f + f_rc)
        h = self.head(f)
        return self.reg(h), self.cls(h)


@torch.no_grad()
def predict(model, X, device="cpu", batch=512, rc_average=True, which="reg"):
    import numpy as np
    model.eval()
    outs = []
    for i in range(0, len(X), batch):
        xb = torch.from_numpy(X[i:i + batch]).float().to(device)
        reg, cls = model(xb, rc_average=rc_average)
        outs.append((reg if which == "reg" else cls).cpu().numpy())
    return np.concatenate(outs, 0)
