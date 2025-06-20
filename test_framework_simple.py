#!/usr/bin/env python3
"""
Simple test script for Mesa-Optimizer Detection Framework
"""

import torch
import torch.nn as nn
from mesa_optimizer_detection import (
    MesaOptimizerDetector,
    create_default_config
)

class SimpleTestModel(nn.Module):
    """Simple test model for demonstration."""
    
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

def test_activation_analysis():
    """Test activation analysis only."""
    print("🧪 Testing Mesa-Optimizer Detection (Activation Analysis)")
    print("=" * 60)
    
    # Create a simple model
    print("📝 Creating test model...")
    model = SimpleTestModel()
    print(f"✅ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create test data
    print("📊 Creating test data...")
    batch_size = 4
    input_size = 10
    test_data = torch.randn(batch_size, input_size)
    print(f"✅ Test data created: {test_data.shape}")
    
    # Create detection configuration
    print("⚙️ Setting up detection configuration...")
    config = create_default_config()
    print("✅ Configuration created")
    
    # Initialize detector with only activation analysis
    print("🕵️ Initializing detector...")
    layer_indices = [1, 3]  # Analyze specific layers
    detection_methods = ['activation']  # Only use activation analysis
    
    detector = MesaOptimizerDetector(
        model=model,
        layer_indices=layer_indices,
        detection_methods=detection_methods,
        config=config
    )
    print(f"✅ Detector initialized with methods: {detection_methods}")
    
    # Run analysis
    print("🔍 Running mesa-optimization analysis...")
    try:
        results = detector.analyze(test_data)
        
        print("\n📋 Analysis Results:")
        print("-" * 40)
        print(f"Risk Level: {results.risk_level}")
        print(f"Risk Score: {results.risk_score:.3f}")
        print(f"Confidence: {results.confidence:.3f}")
        
        print("\n📊 Method Scores:")
        for method, score in results.risk_assessment.method_scores.items():
            print(f"  {method.capitalize():12}: {score:.3f}")
        
        print("\n💡 Recommendations:")
        for i, rec in enumerate(results.risk_assessment.recommendations, 1):
            print(f"  {i}. {rec}")
        
        print("\n✅ Analysis completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_wrapper():
    """Test the ModelWrapper utility."""
    print("\n🔧 Testing Model Wrapper")
    print("=" * 60)
    
    try:
        from mesa_optimizer_detection import ModelWrapper
        
        model = SimpleTestModel()
        wrapper = ModelWrapper(model)
        
        test_input = torch.randn(2, 10)
        output = wrapper(test_input)
        
        print(f"✅ Model wrapper works")
        print(f"   Input shape: {test_input.shape}")
        print(f"   Output shape: {output.shape}")
        
        # Test activation extraction
        activations = wrapper.get_activations(test_input, layer_indices=[1, 3])
        print(f"✅ Activation extraction works")
        print(f"   Extracted activations for {len(activations)} layers")
        
        return True
        
    except Exception as e:
        print(f"❌ Model wrapper test failed: {e}")
        return False

def test_individual_analyzers():
    """Test individual detection methods."""
    print("\n🔍 Testing Individual Analyzers")
    print("=" * 60)
    
    try:
        from mesa_optimizer_detection.detection.activation_analyzer import ActivationPatternAnalyzer
        from mesa_optimizer_detection import ModelWrapper
        
        model = SimpleTestModel()
        model_wrapper = ModelWrapper(model)
        config = create_default_config()
        
        # Test activation analyzer
        analyzer = ActivationPatternAnalyzer(
            model=model_wrapper,
            layer_indices=[1, 3],
            config=config.activation_config
        )
        
        test_input = torch.randn(4, 10)
        result = analyzer.analyze(test_input)
        
        print(f"✅ Activation analyzer works")
        print(f"   Risk score: {result.risk_score:.3f}")
        print(f"   Confidence: {result.confidence:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Individual analyzer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Mesa-Optimizer Detection Framework - Simple Tests\n")
    
    # Run tests
    test_results = []
    
    test_results.append(test_activation_analysis())
    test_results.append(test_model_wrapper())
    test_results.append(test_individual_analyzers())
    
    # Summary
    print("\n📈 Test Summary")
    print("=" * 60)
    passed = sum(test_results)
    total = len(test_results)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All core tests passed! Framework is working correctly.")
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
    
    print("\n🔧 Troubleshooting Tips:")
    print("- For gradient analysis issues: ensure proper target generation")
    print("- For memory issues: reduce batch size or model size")
    print("- For import issues: reinstall with 'pip install -e .'") 