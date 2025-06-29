import torch
import math
from typing import Tuple


def _discretize(t: torch.Tensor, bins: int = 20) -> torch.Tensor:
    """Discretise a 1-D tensor into integer bin indices."""
    # Flatten and detach
    t = t.detach().reshape(-1)
    # Edge case: constant tensor
    if torch.allclose(t.max(), t.min()):
        return torch.zeros_like(t, dtype=torch.long)
    # Compute bin edges
    min_val, max_val = t.min().item(), t.max().item()
    bin_width = (max_val - min_val) / bins
    # Guard against zero width
    if bin_width == 0:
        bin_width = 1e-9
    # Map values to bins 0..bins-1
    indices = ((t - min_val) / bin_width).floor().long()
    indices = torch.clamp(indices, 0, bins - 1)
    return indices


def estimate_mutual_information(x: torch.Tensor, y: torch.Tensor, bins: int = 20) -> float:
    """Estimate mutual information I(X;Y) via histogram discretisation.

    The estimator is *biased* but fast and has no learnable parameters, making it
    suitable for lightweight unit tests. Both tensors are flattened. The result
    is expressed in *nats* (natural logarithm).
    """
    if x.numel() == 0 or y.numel() == 0:
        return 0.0
    if x.shape != y.shape:
        # Broadcast to common shape if possible, else flatten separately
        x_flat = x.detach().reshape(-1)
        y_flat = y.detach().reshape(-1)
        min_len = min(x_flat.size(0), y_flat.size(0))
        x_flat = x_flat[:min_len]
        y_flat = y_flat[:min_len]
    else:
        x_flat = x.detach().reshape(-1)
        y_flat = y.detach().reshape(-1)

    x_bins = _discretize(x_flat, bins)
    y_bins = _discretize(y_flat, bins)

    # Joint histogram
    joint_hist = torch.zeros((bins, bins), dtype=torch.float32)
    for xb, yb in zip(x_bins, y_bins):
        joint_hist[xb, yb] += 1.0
    joint_prob = joint_hist / joint_hist.sum()

    # Marginals with broadcast-compatible shapes
    px = joint_prob.sum(dim=1, keepdim=True)  # shape (bins, 1)
    py = joint_prob.sum(dim=0, keepdim=True)  # shape (1, bins)

    # Small epsilon to avoid log(0)
    eps = 1e-12
    mi = (joint_prob * (torch.log(joint_prob + eps) - torch.log(px + eps) - torch.log(py + eps))).sum()
    return float(mi.item())


__all__: Tuple[str, ...] = ("estimate_mutual_information",) 