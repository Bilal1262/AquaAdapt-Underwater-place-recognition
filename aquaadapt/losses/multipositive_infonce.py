"""Numerically stable multi-positive InfoNCE."""

import torch
from torch.nn import functional as F


def multi_positive_infonce(
    descriptors: torch.Tensor,
    positive_mask: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    r"""Compute average log probability assigned to positives.

    For each anchor i:
        L_i = -log( sum_{p in P(i)} exp(s_ip/t) /
                    sum_{a != i} exp(s_ia/t) )
    """
    if descriptors.ndim != 2:
        raise ValueError("descriptors must have shape [N, D]")
    n = descriptors.shape[0]
    if positive_mask.shape != (n, n):
        raise ValueError(f"positive_mask must have shape {(n, n)}")
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    z = F.normalize(descriptors, dim=-1)
    logits = z @ z.T / temperature
    self_mask = torch.eye(n, dtype=torch.bool, device=z.device)
    valid_positives = positive_mask.bool() & ~self_mask
    valid_anchors = valid_positives.any(dim=1)
    if not valid_anchors.any():
        return descriptors.sum() * 0.0
    logits_without_self = logits.masked_fill(self_mask, -torch.inf)
    numerator = torch.logsumexp(logits.masked_fill(~valid_positives, -torch.inf), dim=1)
    denominator = torch.logsumexp(logits_without_self, dim=1)
    return -(numerator[valid_anchors] - denominator[valid_anchors]).mean()

