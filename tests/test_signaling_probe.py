import torch
from mesa_optimizer_detection import SignalingGameProbe, ModelWrapper, create_default_config

def _simple_lm(input_dim: int = 10, output_dim: int = 5):
    return torch.nn.Sequential(torch.nn.Linear(input_dim, output_dim), torch.nn.ReLU(), torch.nn.Linear(output_dim, output_dim))

def test_signaling_probe_basic():
    config = create_default_config()
    model = _simple_lm()
    wrapper = ModelWrapper(model)
    probe = SignalingGameProbe(model=wrapper, config=config.behavioral_config)

    contexts = [
        {'prompt': torch.randn(1, 10), 'settings': {}},
        {'prompt': torch.randn(1, 10), 'settings': {}}
    ]
    result = probe.analyze(contexts)

    assert 0.0 <= result.risk_score <= 1.0
    assert 0.0 <= result.confidence <= 1.0 