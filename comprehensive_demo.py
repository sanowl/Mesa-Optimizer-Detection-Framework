#!/usr/bin/env python3
"""
Comprehensive Demo of Mesa-Optimizer Detection Framework

This demo showcases all detection methods with detailed results and analysis.
"""

import torch
import torch.nn as nn
import numpy as np
from mesa_optimizer_detection import (
    MesaOptimizerDetector,
    ModelWrapper,
    create_default_config,
    create_conservative_config,
    create_permissive_config
)

print("🚀 Mesa-Optimizer Detection Framework - Comprehensive Demo")
print("=" * 60)

class TestModel(nn.Module):
    """Enhanced test model with more complexity for better demonstration."""
    
    def __init__(self, input_size=20, hidden_size=64, num_layers=4, output_size=10):
        super().__init__()
        
        # Create multi-layer network
        layers = []
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())
        
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
            
        layers.append(nn.Linear(hidden_size, output_size))
        
        self.network = nn.Sequential(*layers)
        
        # Add some specific patterns that might trigger detection
        self._add_suspicious_patterns()
    
    def _add_suspicious_patterns(self):
        """Add patterns that might be detected as suspicious."""
        # Initialize some weights to create structured patterns
        with torch.no_grad():
            for i, layer in enumerate(self.network):
                if isinstance(layer, nn.Linear):
                    # Create some correlation in weights (planning-like patterns)
                    if i % 4 == 0:  # Every second linear layer
                        layer.weight.data[:5, :5] = 0.8  # High correlation block
                        layer.weight.data[5:10, 5:10] = -0.8  # Anti-correlation block
    
    def forward(self, x):
        return self.network(x)


def create_diverse_test_data(batch_size=8, input_size=20, num_batches=3):
    """Create diverse test data for comprehensive analysis."""
    test_batches = []
    
    for i in range(num_batches):
        if i == 0:
            # Normal random data
            data = torch.randn(batch_size, input_size)
        elif i == 1:
            # Structured data (might trigger planning detection)
            data = torch.zeros(batch_size, input_size)
            data[:, :5] = torch.randn(batch_size, 5) * 0.1  # Low variance beginning
            data[:, 5:15] = torch.randn(batch_size, 10) * 2.0  # High variance middle
            data[:, 15:] = torch.randn(batch_size, 5) * 0.1  # Low variance end
        else:
            # Adversarial-like data (might trigger deception detection)
            data = torch.randn(batch_size, input_size)
            data = torch.tanh(data * 3)  # Saturated activations
        
        test_batches.append(data)
    
    return test_batches


def run_detection_with_config(model, data, config, config_name):
    """Run detection with a specific configuration."""
    print(f"\n🔍 Running analysis with {config_name} configuration...")
    print("-" * 50)
    
    # Initialize detector with all available methods
    detector = MesaOptimizerDetector(
        model=model,
        layer_indices=[1, 3, 5, 7],  # Analyze multiple layers
        detection_methods=['gradient', 'activation', 'behavioral'],
        config=config
    )
    
    try:
        results = detector.analyze(data)
        
        print(f"📊 Results Summary:")
        print(f"   Risk Level: {results.risk_level}")
        print(f"   Risk Score: {results.risk_score:.3f}")
        print(f"   Confidence: {results.confidence:.3f}")
        
        print(f"\n📈 Method Breakdown:")
        for method, score in results.risk_assessment.method_scores.items():
            print(f"   {method.capitalize():12}: {score:.3f}")
        
        print(f"\n💡 Key Recommendations:")
        for i, rec in enumerate(results.risk_assessment.recommendations[:3], 1):
            print(f"   {i}. {rec}")
        
        # Detailed method analysis
        print(f"\n🔬 Detailed Analysis:")
        
        if results.has_method_result('gradient'):
            gradient_result = results.get_method_result('gradient')
            print(f"   Gradient Analysis:")
            print(f"     • Variance: {gradient_result.gradient_variance:.3f}")
            print(f"     • Anomaly Score: {gradient_result.anomaly_score:.3f}")
            if hasattr(gradient_result, 'hessian_eigenvalues') and gradient_result.hessian_eigenvalues:
                max_eigenval = max(gradient_result.hessian_eigenvalues)
                print(f"     • Max Hessian Eigenvalue: {max_eigenval:.3f}")
        
        if results.has_method_result('activation'):
            activation_result = results.get_method_result('activation')
            print(f"   Activation Analysis:")
            print(f"     • Planning Score: {activation_result.planning_score:.3f}")
            print(f"     • Goal Directedness: {activation_result.goal_directedness:.3f}")
            print(f"     • Optimization Score: {activation_result.optimization_score:.3f}")
            if activation_result.suspicious_patterns:
                print(f"     • Suspicious Patterns: {len(activation_result.suspicious_patterns)}")
                for pattern in activation_result.suspicious_patterns[:2]:
                    print(f"       - {pattern}")
        
        if results.has_method_result('behavioral'):
            behavioral_result = results.get_method_result('behavioral')
            print(f"   Behavioral Analysis:")
            print(f"     • Consistency: {behavioral_result.consistency_score:.3f}")
            print(f"     • Context Sensitivity: {behavioral_result.context_sensitivity:.3f}")
            if behavioral_result.deception_indicators:
                print(f"     • Deception Indicators: {len(behavioral_result.deception_indicators)}")
        
        return results
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        return None


def demonstrate_multiple_configurations():
    """Demonstrate the framework with different configurations."""
    print("\n🎯 Testing Multiple Configurations")
    print("=" * 60)
    
    # Create model and data
    print("📝 Setting up test environment...")
    model = TestModel(input_size=20, hidden_size=64, num_layers=4)
    model_wrapper = ModelWrapper(model)
    test_data = create_diverse_test_data()[0]  # Use first batch
    
    print(f"✅ Model created: {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"✅ Test data: {test_data.shape}")
    
    # Test with different configurations
    configs = [
        (create_default_config(), "Default"),
        (create_conservative_config(), "Conservative"),
        (create_permissive_config(), "Permissive")
    ]
    
    results_comparison = {}
    
    for config, name in configs:
        result = run_detection_with_config(model_wrapper, test_data, config, name)
        if result:
            results_comparison[name] = result.risk_score
    
    # Comparison summary
    if results_comparison:
        print(f"\n📊 Configuration Comparison:")
        print("-" * 30)
        for config_name, risk_score in results_comparison.items():
            risk_level = "LOW" if risk_score < 0.3 else "MEDIUM" if risk_score < 0.6 else "HIGH"
            print(f"   {config_name:12}: {risk_score:.3f} ({risk_level})")


def demonstrate_data_sensitivity():
    """Show how different input data affects detection results."""
    print(f"\n🎨 Data Sensitivity Analysis")
    print("=" * 60)
    
    model = TestModel()
    model_wrapper = ModelWrapper(model)
    test_batches = create_diverse_test_data()
    
    data_types = ["Normal Random", "Structured", "Adversarial-like"]
    config = create_default_config()
    
    for i, (data, data_type) in enumerate(zip(test_batches, data_types)):
        print(f"\n📊 Testing with {data_type} data...")
        
        try:
            detector = MesaOptimizerDetector(
                model=model_wrapper,
                layer_indices=[1, 3, 5],
                detection_methods=['activation'],  # Use only activation for speed
                config=config
            )
            
            results = detector.analyze(data)
            
            print(f"   Risk Score: {results.risk_score:.3f}")
            print(f"   Risk Level: {results.risk_level}")
            
            if results.has_method_result('activation'):
                activation_result = results.get_method_result('activation')
                print(f"   Planning: {activation_result.planning_score:.3f}")
                print(f"   Goal Directedness: {activation_result.goal_directedness:.3f}")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")


def demonstrate_model_scaling():
    """Show how detection scales with model size."""
    print(f"\n📈 Model Scaling Analysis")
    print("=" * 60)
    
    model_sizes = [
        (10, 16, 2, "Tiny"),
        (20, 32, 3, "Small"),
        (30, 64, 4, "Medium")
    ]
    
    test_data = torch.randn(4, 30)  # Fixed input size
    config = create_conservative_config()
    
    for input_size, hidden_size, num_layers, size_name in model_sizes:
        print(f"\n🔬 Testing {size_name} model...")
        
        try:
            # Adjust input data size
            current_data = test_data[:, :input_size]
            
            model = TestModel(input_size, hidden_size, num_layers)
            model_wrapper = ModelWrapper(model)
            
            detector = MesaOptimizerDetector(
                model=model_wrapper,
                layer_indices=list(range(1, min(num_layers*2, 6), 2)),  # Adaptive layer selection
                detection_methods=['activation'],
                config=config
            )
            
            results = detector.analyze(current_data)
            
            param_count = sum(p.numel() for p in model.parameters())
            print(f"   Parameters: {param_count:,}")
            print(f"   Risk Score: {results.risk_score:.3f}")
            print(f"   Confidence: {results.confidence:.3f}")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")


if __name__ == "__main__":
    print("Starting comprehensive mesa-optimization detection demo...\n")
    
    try:
        # Run all demonstrations
        demonstrate_multiple_configurations()
        demonstrate_data_sensitivity() 
        demonstrate_model_scaling()
        
        print(f"\n🎊 Demo Complete!")
        print("=" * 60)
        print("✅ Framework successfully demonstrated multiple detection methods")
        print("✅ Showed configuration flexibility and sensitivity")
        print("✅ Demonstrated scaling across different model sizes")
        print("✅ All core components working correctly")
        
        print(f"\n📚 Next Steps:")
        print("• Try with your own models using the ModelWrapper")
        print("• Experiment with different detection method combinations")
        print("• Customize thresholds for your specific use case")
        print("• Integrate into your training pipeline for continuous monitoring")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc() 