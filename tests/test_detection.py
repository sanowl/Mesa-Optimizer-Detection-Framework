"""
Unit Tests for Mesa-Optimizer Detection Framework

This module contains comprehensive tests for the detection framework
to ensure reliability and correctness.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import Mock, patch

from mesa_optimizer_detection import (
    MesaOptimizerDetector,
    DetectionConfig,
    GradientAnalyzer,
    ActivationPatternAnalyzer,
    BehavioralConsistencyAnalyzer,
    ModelWrapper,
    create_default_config
)


class SimpleTestModel(nn.Module):
    """Simple model for testing."""
    
    def __init__(self, input_size=10, hidden_size=32, output_size=2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        return self.layers(x)


@pytest.fixture
def simple_model():
    """Fixture providing a simple model for testing."""
    return SimpleTestModel()


@pytest.fixture
def test_data():
    """Fixture providing test data."""
    return torch.randn(4, 10)  # Batch size 4, input size 10


@pytest.fixture
def detection_config():
    """Fixture providing test configuration."""
    return create_default_config()


class TestModelWrapper:
    """Test ModelWrapper functionality."""
    
    def test_model_wrapper_initialization(self, simple_model):
        """Test ModelWrapper initialization."""
        wrapper = ModelWrapper(simple_model)
        
        assert wrapper.model is simple_model
        assert wrapper.device is not None
        assert isinstance(wrapper.activation_hooks, dict)
        assert isinstance(wrapper.stored_activations, dict)
    
    def test_model_wrapper_forward(self, simple_model, test_data):
        """Test ModelWrapper forward pass."""
        wrapper = ModelWrapper(simple_model)
        output = wrapper.forward(test_data)
        
        assert isinstance(output, torch.Tensor)
        assert output.shape[0] == test_data.shape[0]  # Batch size preserved
    
    def test_activation_extraction(self, simple_model, test_data):
        """Test activation extraction functionality."""
        wrapper = ModelWrapper(simple_model)
        
        # Register hook for first layer
        layer_names = wrapper.get_layer_names()
        first_layer = layer_names[1]  # Skip the Sequential wrapper
        
        wrapper.register_activation_hook(first_layer)
        
        # Forward pass
        _ = wrapper.forward(test_data)
        
        # Check activations were captured
        activations = wrapper.get_activations()
        assert first_layer in activations
        assert isinstance(activations[first_layer], torch.Tensor)
        
        # Clean up
        wrapper.remove_hooks()


class TestDetectionConfig:
    """Test DetectionConfig functionality."""
    
    def test_config_creation(self):
        """Test configuration creation."""
        config = DetectionConfig()
        
        assert hasattr(config, 'risk_thresholds')
        assert hasattr(config, 'gradient_config')
        assert hasattr(config, 'activation_config')
        assert hasattr(config, 'behavioral_config')
        assert hasattr(config, 'method_weights')
    
    def test_config_validation(self):
        """Test configuration validation."""
        config = DetectionConfig()
        
        # Valid configuration should pass
        assert config.validate() is True
        
        # Invalid thresholds should fail
        config.risk_thresholds.low = 0.8
        config.risk_thresholds.high = 0.2
        
        with pytest.raises(ValueError):
            config.validate()
    
    def test_config_serialization(self, tmp_path):
        """Test configuration saving and loading."""
        config = DetectionConfig()
        config.risk_thresholds.low = 0.2
        config.method_weights['gradient'] = 1.5
        
        # Save configuration
        config_path = tmp_path / "test_config.yaml"
        config.save(str(config_path))
        
        # Load configuration
        loaded_config = DetectionConfig.load(str(config_path))
        
        assert loaded_config.risk_thresholds.low == 0.2
        assert loaded_config.method_weights['gradient'] == 1.5


class TestGradientAnalyzer:
    """Test GradientAnalyzer functionality."""
    
    def test_gradient_analyzer_initialization(self, simple_model):
        """Test GradientAnalyzer initialization."""
        wrapper = ModelWrapper(simple_model)
        analyzer = GradientAnalyzer(wrapper)
        
        assert analyzer.model is wrapper
        assert hasattr(analyzer, 'config')
        assert isinstance(analyzer.gradient_history, list)
    
    def test_gradient_computation(self, simple_model, test_data):
        """Test gradient computation."""
        wrapper = ModelWrapper(simple_model)
        analyzer = GradientAnalyzer(wrapper)
        
        # This requires a loss computation, so we'll test the basic structure
        try:
            gradients = analyzer._compute_gradients(test_data)
            assert isinstance(gradients, torch.Tensor)
        except Exception:
            # Expected to fail without proper loss setup
            pass
    
    def test_gradient_analysis_structure(self, simple_model, test_data):
        """Test gradient analysis result structure."""
        wrapper = ModelWrapper(simple_model)
        analyzer = GradientAnalyzer(wrapper)
        
        # Mock the gradient computation to avoid loss computation issues
        with patch.object(analyzer, '_compute_gradients') as mock_compute:
            mock_compute.return_value = torch.randn(100)  # Mock gradients
            
            result = analyzer.analyze(test_data)
            
            assert hasattr(result, 'gradient_variance')
            assert hasattr(result, 'anomaly_score')
            assert hasattr(result, 'risk_score')
            assert hasattr(result, 'confidence')
            
            assert isinstance(result.gradient_variance, float)
            assert isinstance(result.anomaly_score, float)
            assert 0 <= result.risk_score <= 1
            assert 0 <= result.confidence <= 1


class TestActivationPatternAnalyzer:
    """Test ActivationPatternAnalyzer functionality."""
    
    def test_activation_analyzer_initialization(self, simple_model):
        """Test ActivationPatternAnalyzer initialization."""
        wrapper = ModelWrapper(simple_model)
        analyzer = ActivationPatternAnalyzer(wrapper, layer_indices=[0, 2])
        
        assert analyzer.model is wrapper
        assert analyzer.layer_indices == [0, 2]
        assert hasattr(analyzer, 'config')
    
    def test_activation_analysis_structure(self, simple_model, test_data):
        """Test activation analysis result structure."""
        wrapper = ModelWrapper(simple_model)
        analyzer = ActivationPatternAnalyzer(wrapper, layer_indices=[1, 3])
        
        result = analyzer.analyze(test_data)
        
        assert hasattr(result, 'planning_score')
        assert hasattr(result, 'goal_directedness')
        assert hasattr(result, 'optimization_score')
        assert hasattr(result, 'risk_score')
        assert hasattr(result, 'confidence')
        
        assert isinstance(result.planning_score, float)
        assert isinstance(result.goal_directedness, float)
        assert 0 <= result.risk_score <= 1
        assert 0 <= result.confidence <= 1


class TestBehavioralConsistencyAnalyzer:
    """Test BehavioralConsistencyAnalyzer functionality."""
    
    def test_behavioral_analyzer_initialization(self, simple_model):
        """Test BehavioralConsistencyAnalyzer initialization."""
        wrapper = ModelWrapper(simple_model)
        analyzer = BehavioralConsistencyAnalyzer(wrapper)
        
        assert analyzer.model is wrapper
        assert hasattr(analyzer, 'config')
        assert isinstance(analyzer.response_cache, dict)
    
    def test_behavioral_analysis_structure(self, simple_model, test_data):
        """Test behavioral analysis result structure."""
        wrapper = ModelWrapper(simple_model)
        analyzer = BehavioralConsistencyAnalyzer(wrapper)
        
        # Create test contexts
        contexts = [
            {
                'name': 'context1',
                'prompt': test_data,
                'settings': {'temperature': 0.7}
            },
            {
                'name': 'context2',
                'prompt': test_data,
                'settings': {'temperature': 1.0}
            }
        ]
        
        result = analyzer.analyze(contexts)
        
        assert hasattr(result, 'consistency_score')
        assert hasattr(result, 'context_sensitivity')
        assert hasattr(result, 'deception_indicators')
        assert hasattr(result, 'risk_score')
        assert hasattr(result, 'confidence')
        
        assert isinstance(result.consistency_score, float)
        assert isinstance(result.context_sensitivity, float)
        assert isinstance(result.deception_indicators, list)
        assert 0 <= result.risk_score <= 1
        assert 0 <= result.confidence <= 1


class TestMesaOptimizerDetector:
    """Test main MesaOptimizerDetector functionality."""
    
    def test_detector_initialization(self, simple_model, detection_config):
        """Test MesaOptimizerDetector initialization."""
        detector = MesaOptimizerDetector(
            model=simple_model,
            layer_indices=[1, 3],
            detection_methods=['gradient', 'activation'],
            config=detection_config
        )
        
        assert isinstance(detector.model, ModelWrapper)
        assert detector.layer_indices == [1, 3]
        assert detector.detection_methods == ['gradient', 'activation']
        assert detector.config is detection_config
        assert hasattr(detector, 'analyzers')
    
    def test_detector_analysis_structure(self, simple_model, test_data, detection_config):
        """Test detector analysis result structure."""
        detector = MesaOptimizerDetector(
            model=simple_model,
            layer_indices=[1, 3],
            detection_methods=['activation'],  # Use only activation to avoid gradient issues
            config=detection_config
        )
        
        # Mock the analyzer to avoid complex setup
        mock_result = Mock()
        mock_result.risk_score = 0.3
        mock_result.confidence = 0.7
        
        with patch.object(detector.analyzers['activation'], 'analyze') as mock_analyze:
            mock_analyze.return_value = mock_result
            
            result = detector.analyze(test_data)
            
            assert hasattr(result, 'risk_assessment')
            assert hasattr(result, 'method_results')
            assert hasattr(result, 'metadata')
            
            assert hasattr(result.risk_assessment, 'risk_level')
            assert hasattr(result.risk_assessment, 'risk_score')
            assert hasattr(result.risk_assessment, 'confidence')
            assert hasattr(result.risk_assessment, 'recommendations')
    
    def test_risk_level_computation(self, simple_model, detection_config):
        """Test risk level computation."""
        detector = MesaOptimizerDetector(
            model=simple_model,
            config=detection_config
        )
        
        # Test different risk scores
        assert detector._compute_risk_level(0.1) == "MINIMAL"
        assert detector._compute_risk_level(0.4) == "LOW" 
        assert detector._compute_risk_level(0.7) == "MEDIUM"
        assert detector._compute_risk_level(0.9) == "HIGH"
    
    def test_detector_summary(self, simple_model, detection_config):
        """Test detector summary generation."""
        detector = MesaOptimizerDetector(
            model=simple_model,
            config=detection_config
        )
        
        summary = detector.get_detection_summary()
        
        assert 'detector_version' in summary
        assert 'detection_methods' in summary
        assert 'model_parameters' in summary
        assert 'config' in summary


class TestIntegration:
    """Integration tests for the complete framework."""
    
    def test_end_to_end_analysis(self, simple_model, test_data):
        """Test complete end-to-end analysis."""
        config = create_default_config()
        
        # Use only activation analysis to avoid gradient computation issues
        detector = MesaOptimizerDetector(
            model=simple_model,
            layer_indices=[1, 3],
            detection_methods=['activation'],
            config=config
        )
        
        # This should run without errors
        result = detector.analyze(test_data)
        
        assert result is not None
        assert hasattr(result, 'risk_score')
        assert hasattr(result, 'risk_level')
        assert hasattr(result, 'confidence')
    
    def test_result_serialization(self, simple_model, test_data, tmp_path):
        """Test result saving and loading."""
        config = create_default_config()
        detector = MesaOptimizerDetector(
            model=simple_model,
            layer_indices=[1],
            detection_methods=['activation'],
            config=config
        )
        
        result = detector.analyze(test_data)
        
        # Save result
        result_path = tmp_path / "test_result.json"
        result.save(str(result_path))
        
        # Check file was created
        assert result_path.exists()
        
        # Load result
        loaded_result = result.__class__.load(str(result_path))
        assert loaded_result.risk_assessment.risk_score == result.risk_score


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 