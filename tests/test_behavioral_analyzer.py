import torch
from mesa_optimizer_detection import BehavioralConsistencyAnalyzer, ModelWrapper, create_default_config


def _simple_language_like():
    # Minimal linear embedding to fake language model input/output
    return torch.nn.Sequential(torch.nn.Linear(5, 5), torch.nn.Tanh(), torch.nn.Linear(5, 5))


def test_behavioral_analyzer_runs():
    cfg = create_default_config()
    model = ModelWrapper(_simple_language_like())
    analyzer = BehavioralConsistencyAnalyzer(model=model, config=cfg.behavioral_config)

    # Create three evaluation contexts with different temperatures (toy tensors)
    batch = torch.randn(3, 5)
    contexts = [
        {"name": "c0", "prompt": batch, "settings": {"temperature": 0.5, "max_length": 10}},
        {"name": "c1", "prompt": batch, "settings": {"temperature": 1.0, "max_length": 10}},
        {"name": "c2", "prompt": batch, "settings": {"temperature": 1.5, "max_length": 10}},
    ]

    res = analyzer.analyze(contexts)
    assert 0.0 <= res.risk_score <= 1.0
    assert 0.0 <= res.confidence <= 1.0 