import torch
import pytest

from mesa_optimizer_detection import GradientInvariantAnalyzer, ModelWrapper, create_default_config


def _simple_linear_model(input_dim: int = 10, output_dim: int = 1):
    return torch.nn.Sequential(torch.nn.Linear(input_dim, output_dim))


def test_gradient_invariant_analyzer_basic():
    """GradientInvariantAnalyzer should return valid risk/confidence in [0,1]."""
    config = create_default_config()
    # Use same GradientConfig for invariant analyzer

    model = _simple_linear_model()
    wrapper = ModelWrapper(model)
    analyzer = GradientInvariantAnalyzer(model=wrapper, config=config.gradient_config)

    batch = torch.randn(8, 10)
    result = analyzer.analyze(batch)

    assert 0.0 <= result.risk_score <= 1.0
    assert 0.0 <= result.confidence <= 1.0 