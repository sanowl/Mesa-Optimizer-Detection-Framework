#!/usr/bin/env python
"""Generate a toy adversarial dataset for Mesa-Optimizer detection.

It creates a handful of *benign* linear models and *mesa* models (linear but
with an inner optimisation step baked into forward) and stores their
DetectionResults JSON plus ground-truth labels so that the calibrator CLI can
train.

Run:
    python scripts/generate_adversarial.py --out-dir datasets/toy
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List

import torch

from mesa_optimizer_detection import (
    MesaOptimizerDetector,
    create_default_config,
    ModelWrapper,
)

# ---------------------------------------------------------------------
# Toy model definitions
# ---------------------------------------------------------------------

def benign_linear_model(in_dim: int = 10, out_dim: int = 1):
    return torch.nn.Sequential(torch.nn.Linear(in_dim, out_dim))


def mesa_linear_model(in_dim: int = 10, out_dim: int = 1):
    class MesaLinear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(in_dim, out_dim)
            # Inner parameter representing deceptive objective
            self.inner_param = torch.nn.Parameter(torch.zeros(1))

        def forward(self, x):
            # Inner optimisation: adjust weight a tiny bit towards misleading value
            with torch.no_grad():
                self.inner_param += 0.0001  # toy drift
            return self.linear(x) + self.inner_param

    return MesaLinear()


# ---------------------------------------------------------------------
def generate_dataset(n_models: int, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    results_dir = out_dir / "results"
    results_dir.mkdir(exist_ok=True)
    labels: List[dict] = []

    config = create_default_config()
    detector_cfg = config

    for idx in range(n_models):
        is_mesa = idx % 2 == 1  # alternate
        model = mesa_linear_model() if is_mesa else benign_linear_model()
        wrapper = ModelWrapper(model)
        detector = MesaOptimizerDetector(model=wrapper, config=detector_cfg, detection_methods=["gradient", "gradient_invariant", "activation"])

        batch = torch.randn(8, 10)
        results = detector.analyze(batch)

        file_name = f"model_{idx:03d}.json"
        results_path = results_dir / file_name
        results.save(results_path)

        labels.append({"file": f"results/{file_name}", "label": int(is_mesa)})
        print(f"Saved {results_path} (mesa={is_mesa})")

    with open(out_dir / "labels.json", "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2)
    print(f"Dataset written to {out_dir}")


# ---------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate toy adversarial dataset")
    parser.add_argument("--out-dir", default="datasets/toy", help="Output directory")
    parser.add_argument("--n-models", type=int, default=10, help="Number of models (half mesa, half benign)")
    args = parser.parse_args()

    generate_dataset(args.n_models, Path(args.out_dir)) 