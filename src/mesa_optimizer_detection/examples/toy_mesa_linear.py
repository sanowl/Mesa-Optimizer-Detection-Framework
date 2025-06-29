"""Toy linear mesa-optimiser example.

A 2-layer linear network that, after a few SGD steps, effectively embeds an
inner optimiser: the hidden layer stores a *desired* target, and the second
layer acts as a gradient step towards that target.

For demonstration we export two checkpoints that differ only in a single weight
scalar—one *pre-mesa*, one *post-mesa*—so unit tests can assert detection.
"""

from __future__ import annotations

from typing import Tuple
import torch
import math


class ToyMesaLinear(torch.nn.Module):
    def __init__(self, dim: int = 4):
        super().__init__()
        self.W = torch.nn.Parameter(torch.eye(dim))   # acts as identity encoder
        self.v = torch.nn.Parameter(torch.zeros(dim)) # decoder/optimiser vector

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, dim)
        h = x @ self.W.t()  # encode
        y = h + self.v      # one gradient-like update towards v
        return y


def generate_checkpoints(dim: int = 4) -> Tuple[dict, dict]:
    """Return two `state_dict`s: (pre_mesa, post_mesa)."""
    model = ToyMesaLinear(dim)
    # Pre-mesa: v is near zero
    pre = model.state_dict()

    # Post-mesa: set v to some non-trivial vector representing an internal goal
    with torch.no_grad():
        model.v.copy_(torch.arange(1, dim + 1, dtype=torch.float32))
    post = model.state_dict()
    return pre, post


if __name__ == "__main__":
    pre, post = generate_checkpoints(4)
    print("Pre-mesa v:", pre["v"].numpy())
    print("Post-mesa v:", post["v"].numpy()) 