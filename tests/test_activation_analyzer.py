import torch
import pytest

from mesa_optimizer_detection import ActivationPatternAnalyzer, ModelWrapper, create_default_config


def _tiny_mlp():
    return torch.nn.Sequential(
        torch.nn.Linear(10, 16),
        torch.nn.ReLU(),
        torch.nn.Linear(16, 8),
        torch.nn.ReLU(),
        torch.nn.Linear(8, 4),
    )


@pytest.mark.parametrize("batch_size", [2, 4])
def test_activation_analyzer_runs(batch_size):
    config = create_default_config()
    model = ModelWrapper(_tiny_mlp())

    analyzer = ActivationPatternAnalyzer(
        model=model,
        layer_indices=[1],  # pick first hidden layer
        config=config.activation_config,
    )

    input_batch = torch.randn(batch_size, 10)

    result = analyzer.analyze(input_batch)

    assert 0.0 <= result.risk_score <= 1.0
    assert 0.0 <= result.confidence <= 1.0 