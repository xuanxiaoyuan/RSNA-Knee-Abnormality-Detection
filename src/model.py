"""Backbone plus a study head that keeps a study's series structure.

A study is (S, K, H, W): S series (the (plane, fluid, fat-sat) slots
`select_series` picks), K slices each. Folding all S*K slices into one bag and
averaging them is permutation-invariant: it cannot tell which slices came from
the same series, or which view a finding is visible in — and which view shows a
finding differs per target (an ACL tear reads best sagittal, effusion another).
This head keeps the grouping: pool slices *within* a series into one token, add
a learned *protocol slot* embedding (plane, fluid) so the model is told which
view a series is instead of inferring it from pixels, then let each target
attend over the present series (masked softmax) and read off its own logit.

The head is kept deliberately *small*. Supervision is attached to the whole
study, so the aggregation should be coarse — and an over-large head overfits the
~3.5k weakly-labelled studies. Direct evidence: v21 grafted a transformer
encoder onto this head and regressed to 0.791 on the LB, below the 0.813 an
earlier flat slice-mean head scored.

Backbone is DINOv2-small by default: a self-supervised ViT that transfers far
better than an ImageNet ResNet, with all but its last few blocks frozen and a
backbone LR two orders below the head's — the frozen tail is the strongest
regularizer we have against the weak labels.
"""

from __future__ import annotations

import os

import timm
import torch
import torch.nn as nn

from dataset import N_SLOTS

# ImageNet statistics, which both DINOv2 and the ResNet default to in timm. The
# dataset hands over plain [0, 1] pixels, so the normalisation happens here.
_IMG_MEAN = (0.485, 0.456, 0.406)
_IMG_STD = (0.229, 0.224, 0.225)


def _build_backbone(name, pretrained):
    kwargs = dict(pretrained=pretrained, num_classes=0, in_chans=3)
    if "vit" in name or "dinov2" in name:
        # A ViT's positional embedding is baked at its pretrain resolution (518
        # for DINOv2); dynamic_img_size lets our 224 inputs interpolate onto it
        # instead of failing on the token count.
        kwargs["dynamic_img_size"] = True
    try:
        return timm.create_model(name, **kwargs)
    except TypeError:
        # Older timm without dynamic_img_size: build anyway, the caller's shape
        # check will surface the mismatch loudly rather than silently.
        kwargs.pop("dynamic_img_size", None)
        return timm.create_model(name, **kwargs)


def _freeze_backbone(backbone, n_trainable):
    """Freeze everything but the last `n_trainable` transformer blocks (+ norm).

    Early ViT blocks hold generic features and late blocks hold semantics, so
    only the tail needs to move. It is also the main regularizer: it stops a big
    pretrained net from memorising a few thousand weakly-labelled studies.
    """
    blocks = getattr(backbone, "blocks", None)
    if blocks is None:
        return
    for param in backbone.parameters():
        param.requires_grad = False
    for block in blocks[-n_trainable:]:
        for param in block.parameters():
            param.requires_grad = True
    norm = getattr(backbone, "norm", None)
    if norm is not None:
        for param in norm.parameters():
            param.requires_grad = True


class KneeModel(nn.Module):
    """Series in, one logit per target out."""

    def __init__(
        self,
        backbone="vit_small_patch14_dinov2.lvd142m",
        n_labels=12,
        pretrained=True,
        dropout=0.2,
        n_slots=N_SLOTS,
        d_model=256,
        trainable_blocks=0,
    ):
        super().__init__()
        self.backbone = _build_backbone(backbone, pretrained)
        if trainable_blocks:
            _freeze_backbone(self.backbone, trainable_blocks)

        self.register_buffer("img_mean", torch.tensor(_IMG_MEAN).view(1, 3, 1, 1))
        self.register_buffer("img_std", torch.tensor(_IMG_STD).view(1, 3, 1, 1))

        # Project the backbone features down: the ViT's 384 dims carry no extra
        # information for 6 series, and a narrower token keeps the head cheap.
        self.proj = nn.Linear(self.backbone.num_features, d_model)
        # The series' protocol slot (plane, fluid), keyed by metadata value not
        # by position, so it stays correct however the series are ordered. Index
        # 0 is the padding slot; see dataset.slot_id.
        self.slot_emb = nn.Embedding(n_slots, d_model)
        # One attention query per target: which series a finding is best read in
        # differs per target, so each gets to weight the present series itself.
        self.query = nn.Parameter(torch.randn(n_labels, d_model) * (d_model ** -0.5))
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x, mask=None, slots=None):
        # x: (B, S, K, H, W) grayscale; mask: (B, S, K), 1 on real slices;
        # slots: (B, S) protocol-slot indices into slot_emb.
        batch, n_series, n_slices = x.shape[0], x.shape[1], x.shape[2]
        flat = x.reshape(batch * n_series * n_slices, 1, x.shape[3], x.shape[4])
        flat = (flat.expand(-1, 3, -1, -1) - self.img_mean) / self.img_std
        feats = self.backbone(flat).reshape(batch, n_series, n_slices, -1)

        if mask is None:
            slice_w = torch.ones(
                batch, n_series, n_slices, 1, device=x.device, dtype=feats.dtype
            )
        else:
            slice_w = mask.reshape(batch, n_series, n_slices).to(feats.dtype).unsqueeze(-1)

        # Pool the K slices within each series into a single token. The dataset
        # pads short series with black slices, so weight real slices and clamp
        # the divisor for a series that decoded to nothing.
        tokens = (feats * slice_w).sum(dim=2) / slice_w.sum(dim=2).clamp(min=1.0)
        tokens = self.proj(tokens)
        if slots is not None:
            tokens = tokens + self.slot_emb(slots)

        # A series counts as present if it kept at least one real slice.
        present = slice_w.sum(dim=2).squeeze(-1) > 0
        # A study with no decoded series at all would leave every token masked,
        # which makes the attention softmax all -inf (NaN). Keep one.
        none_present = ~present.any(dim=1)
        if none_present.any():
            present = present.clone()
            present[none_present, 0] = True

        # Per-target attention over the present series. Masked softmax: an absent
        # series contributes nothing, so a missing axial series shifts the
        # finding's attention onto the sequences that are present rather than
        # feeding it a zero vector.
        attn = torch.einsum("ld,bsd->bls", self.query, tokens)  # (B, L, S)
        attn = attn / (tokens.shape[-1] ** 0.5)
        attn = attn.masked_fill(~present.unsqueeze(1), torch.finfo(attn.dtype).min)
        weights = torch.softmax(attn, dim=-1)                    # (B, L, S)
        pooled = torch.einsum("bls,bsd->bld", weights, tokens)   # (B, L, d)
        return self.head(self.drop(pooled)).squeeze(-1)          # (B, L)


def build_model(cfg, n_labels, device, pretrained=None):
    """Build the model.

    pretrained=None defers to the config. Pass False when a checkpoint is about
    to replace every weight: downloading weights only to overwrite them is
    wasted work, and offline the download fails and silently leaves us at random
    init instead.
    """
    torch_home = cfg.get("torch_home")
    if torch_home:
        os.environ["TORCH_HOME"] = str(torch_home)

    backbone = cfg.get("backbone", "vit_small_patch14_dinov2.lvd142m")
    dropout = float(cfg.get("dropout", 0.2))
    trainable_blocks = int(cfg.get("trainable_backbone_blocks", 0))
    wants_pretrained = (
        bool(cfg.get("pretrained", True)) if pretrained is None else pretrained
    )

    try:
        model = KneeModel(
            backbone, n_labels, pretrained=wants_pretrained, dropout=dropout,
            trainable_blocks=trainable_blocks,
        )
    except Exception as exc:
        # Most likely an offline run with no weights cache attached. Degrading
        # to random init is far better than dying, but it must be loud.
        print(f"[warn] could not load pretrained weights ({exc})")
        print("[warn] falling back to RANDOM INIT — scores will be poor")
        model = KneeModel(
            backbone, n_labels, pretrained=False, dropout=dropout,
            trainable_blocks=trainable_blocks,
        )

    return model.to(device)
