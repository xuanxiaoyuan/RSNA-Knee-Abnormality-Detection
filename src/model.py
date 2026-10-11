"""Backbone plus a study head that keeps a study's series structure.

A study is (S, K, H, W): S series (sagittal/coronal/axial, fluid-sensitive,
fat-sat — the slots `select_series` picks), K slices each. The old head folded
all S*K slices into one bag and averaged them, which is permutation-invariant:
it cannot tell that two slices came from the same series, or from a different
one. But which view a finding is visible in differs per target (an ACL tear
reads best in the sagittal plane, effusion in another), so a model blind to
series identity cannot learn to weight the informative view per target. This
head keeps the grouping: pool slices *within* a series into one token, let the
series tokens attend to each other, then pool across series. Each series token
also carries a learned *protocol slot* embedding (plane, fluid) taken from the
series metadata, so the model is told which view a series is instead of having
to infer it from pixels.
"""

from __future__ import annotations

import os

import timm
import torch
import torch.nn as nn

from dataset import N_SLOTS


class KneeModel(nn.Module):
    """Series in, one logit per target out."""

    def __init__(
        self,
        backbone="resnet34",
        n_labels=12,
        pretrained=True,
        dropout=0.2,
        n_slots=N_SLOTS,
        d_model=256,
        n_heads=4,
        n_layers=1,
    ):
        super().__init__()
        self.backbone = timm.create_model(
            backbone, pretrained=pretrained, num_classes=0, in_chans=3
        )
        # Project the backbone features down before attention: 512-wide tokens
        # would make the attention quadratic in a dimension that carries no
        # extra information for 6 series.
        self.proj = nn.Linear(self.backbone.num_features, d_model)
        # The series' protocol slot (plane, fluid), keyed by metadata value not
        # by position, so it stays correct however the series are ordered. Index
        # 0 is the padding slot; see dataset.slot_id.
        self.slot_emb = nn.Embedding(n_slots, d_model)
        # norm_first keeps a randomly-initialised transformer from destabilising
        # the convergence of the pretrained backbone grafted in front of it.
        # No positional encoding on the tokens themselves: the series order out
        # of select_series is arbitrary (it sorts by group size), so position is
        # not a signal — the slot embedding above is what identifies a series.
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.attn = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, n_labels),
        )

    def forward(self, x, mask=None, slots=None):
        # x: (B, S, K, H, W) grayscale; mask: (B, S, K), 1 on real slices;
        # slots: (B, S) protocol-slot indices into slot_emb.
        batch, n_series, n_slices = x.shape[0], x.shape[1], x.shape[2]
        flat = x.reshape(batch * n_series * n_slices, 1, x.shape[3], x.shape[4])
        flat = flat.expand(-1, 3, -1, -1)
        feats = self.backbone(flat).reshape(batch, n_series, n_slices, -1)

        if mask is None:
            slice_w = torch.ones(
                batch, n_series, n_slices, 1, device=x.device, dtype=feats.dtype
            )
        else:
            slice_w = mask.reshape(batch, n_series, n_slices).to(feats.dtype).unsqueeze(-1)

        # Pool the K slices within each series into a single token. The dataset
        # pads short series/series-lists with black slices, so weight real
        # slices and clamp the divisor for a series that decoded to nothing.
        tokens = (feats * slice_w).sum(dim=2) / slice_w.sum(dim=2).clamp(min=1.0)
        tokens = self.proj(tokens)
        if slots is not None:
            tokens = tokens + self.slot_emb(slots)

        # A series counts as present if it kept at least one real slice.
        present = slice_w.sum(dim=2).squeeze(-1) > 0
        # A study with no decoded series at all would leave every token masked,
        # and a fully-masked attention softmax is all -inf (NaN). Keep one.
        none_present = ~present.any(dim=1)
        if none_present.any():
            present = present.clone()
            present[none_present, 0] = True

        # TransformerEncoder's padding mask is True on positions to *ignore*.
        tokens = self.attn(tokens, src_key_padding_mask=~present)

        series_w = present.to(tokens.dtype).unsqueeze(-1)
        pooled = (tokens * series_w).sum(dim=1) / series_w.sum(dim=1).clamp(min=1.0)
        return self.head(pooled)


def build_model(cfg, n_labels, device, pretrained=None):
    """Build the model.

    pretrained=None defers to the config. Pass False when a checkpoint is about
    to replace every weight: downloading ImageNet weights only to overwrite them
    is wasted work, and offline the download fails and silently leaves us at
    random init instead.
    """
    torch_home = cfg.get("torch_home")
    if torch_home:
        os.environ["TORCH_HOME"] = str(torch_home)

    backbone = cfg.get("backbone", "resnet34")
    dropout = float(cfg.get("dropout", 0.2))
    wants_pretrained = (
        bool(cfg.get("pretrained", True)) if pretrained is None else pretrained
    )

    try:
        model = KneeModel(backbone, n_labels, pretrained=wants_pretrained, dropout=dropout)
    except Exception as exc:
        # Most likely an offline run with no weights cache attached. Degrading
        # to random init is far better than dying, but it must be loud.
        print(f"[warn] could not load pretrained weights ({exc})")
        print("[warn] falling back to RANDOM INIT — scores will be poor")
        model = KneeModel(backbone, n_labels, pretrained=False, dropout=dropout)

    return model.to(device)
