"""Backbone plus a multiple-instance head over a study's slices."""

from __future__ import annotations

import os

import timm
import torch
import torch.nn as nn


class KneeModel(nn.Module):
    """Slices in, one logit per target out.

    Every sampled slice is scored independently and the slice scores are
    averaged. Averaging rather than max-pooling keeps gradients flowing from
    all slices, which matters because nothing tells us which slice actually
    contains the finding.
    """

    def __init__(self, backbone="resnet34", n_labels=12, pretrained=True, dropout=0.2):
        super().__init__()
        self.backbone = timm.create_model(
            backbone, pretrained=pretrained, num_classes=0, in_chans=3
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.backbone.num_features, n_labels),
        )

    def forward(self, x):
        # x: (B, T, H, W) grayscale -> fold slices into the batch dim
        batch, n_slices = x.shape[0], x.shape[1]
        x = x.reshape(batch * n_slices, 1, x.shape[2], x.shape[3])
        x = x.expand(-1, 3, -1, -1)
        logits = self.head(self.backbone(x))
        return logits.reshape(batch, n_slices, -1).mean(dim=1)


def build_model(cfg, n_labels, device):
    torch_home = cfg.get("torch_home")
    if torch_home:
        os.environ["TORCH_HOME"] = str(torch_home)

    backbone = cfg.get("backbone", "resnet34")
    dropout = float(cfg.get("dropout", 0.2))
    wants_pretrained = bool(cfg.get("pretrained", True))

    try:
        model = KneeModel(backbone, n_labels, pretrained=wants_pretrained, dropout=dropout)
    except Exception as exc:
        # Most likely an offline run with no weights cache attached. Degrading
        # to random init is far better than dying, but it must be loud.
        print(f"[warn] could not load pretrained weights ({exc})")
        print("[warn] falling back to RANDOM INIT — scores will be poor")
        model = KneeModel(backbone, n_labels, pretrained=False, dropout=dropout)

    return model.to(device)
