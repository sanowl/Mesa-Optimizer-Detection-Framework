from typing import List, Tuple, Any
import torch
from torch.utils.data import Dataset

from ..core.detector import MesaOptimizerDetector  # for type hints only
from ..utils.model_utils import ModelWrapper
from ..config import create_default_config


class ToyModelDataset(Dataset):
    """A tiny dataset that yields (model_wrapper, ground_truth_flag) tuples.

    ground_truth_flag := True if the model *intentionally* contains suspicious patterns.
    """

    def __init__(self):
        from benchmark_evaluation import ModelGenerator  # lazy import

        self.entries: List[Tuple[ModelWrapper, bool]] = []
        variants = ModelGenerator.create_model_variants()
        for name, model, _, gt in variants:
            wrapper = ModelWrapper(model)
            self.entries.append((wrapper, gt))

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Tuple[ModelWrapper, bool]:
        return self.entries[idx]


# Convenience helper

def run_dataset_quick_check(detector_cls=MesaOptimizerDetector) -> List[Any]:
    """Run each model in the ToyModelDataset through a detector and return results list."""
    dataset = ToyModelDataset()
    results = []
    for wrapper, gt in dataset:
        detector = detector_cls(model=wrapper, config=create_default_config())
        batch = torch.randn(2, wrapper.model[0].in_features)
        res = detector.analyze(batch)
        res.metadata = res.metadata or {}
        res.metadata["ground_truth_mesa"] = gt
        results.append(res)
    return results 