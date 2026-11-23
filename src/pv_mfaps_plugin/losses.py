"""Long-tail losses used by the plug-in examples."""
from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def balanced_softmax_loss(logits: Tensor, target: Tensor, counts: Tensor, reduction: str = "mean") -> Tensor:
    if logits.ndim != 2 or counts.ndim != 1 or logits.shape[1] != counts.numel():
        raise ValueError("logits must be [batch, classes] and counts must match classes")
    log_prior = counts.to(device=logits.device, dtype=logits.dtype).clamp_min(1).log()
    return F.cross_entropy(logits + log_prior.unsqueeze(0), target, reduction=reduction)


def logit_adjusted_cross_entropy(logits: Tensor, target: Tensor, counts: Tensor, tau: float = 1.0, reduction: str = "mean") -> Tensor:
    if logits.ndim != 2 or counts.ndim != 1 or logits.shape[1] != counts.numel():
        raise ValueError("logits must be [batch, classes] and counts must match classes")
    prior = counts.to(device=logits.device, dtype=logits.dtype).clamp_min(1)
    return F.cross_entropy(logits + float(tau) * prior.log().unsqueeze(0), target, reduction=reduction)
