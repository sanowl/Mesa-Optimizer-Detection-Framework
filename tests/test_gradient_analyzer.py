import torch
import pytest

from mesa_optimizer_detection import GradientAnalyzer, ModelWrapper, create_default_config


def _simple_linear_model(input_dim: int = 10, output_dim: int = 1):
    return torch.nn.Sequential(torch.nn.Linear(input_dim, output_dim))


@pytest.mark.parametrize("variance, anomaly", [(0.5, 0.7), (0.8, 0.9)])
def test_gradient_analyzer_basic(variance, anomaly):
    config = create_default_config()
    config.gradient_config.variance_threshold = variance
    config.gradient_config.anomaly_threshold = anomaly

    model = _simple_linear_model()
    wrapper = ModelWrapper(model)
    analyzer = GradientAnalyzer(model=wrapper, config=config.gradient_config)

    # Tiny batch of random inputs
    batch = torch.randn(4, 10)
    result = analyzer.analyze(batch)

    assert 0.0 <= result.risk_score <= 1.0
    assert 0.0 <= result.confidence <= 1.0 