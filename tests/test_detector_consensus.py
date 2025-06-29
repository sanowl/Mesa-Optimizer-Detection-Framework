import torch
from mesa_optimizer_detection import MesaOptimizerDetector, create_default_config, ModelWrapper


def _toy_model():
    return torch.nn.Sequential(torch.nn.Linear(10, 5), torch.nn.ReLU(), torch.nn.Linear(5, 2))


def test_detector_consensus_penalty():
    config = create_default_config()
    # Require 2 methods and high consensus
    config.require_multiple_methods = True
    config.min_methods_for_detection = 2
    config.consensus_threshold = 0.75

    model = ModelWrapper(_toy_model())

    # Only use gradient method so consensus should fail and risk should be down-weighted
    detector = MesaOptimizerDetector(model=model, detection_methods=["gradient"], config=config)
    batch = torch.randn(2, 10)
    results = detector.analyze(batch)

    # With consensus penalty risk should be <= 0.5 even if gradient flags
    assert results.risk_score <= 0.5 