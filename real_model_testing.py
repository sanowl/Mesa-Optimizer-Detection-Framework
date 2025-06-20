"""
Real Model Testing for Mesa-Optimizer Detection Framework
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Any
import warnings
warnings.filterwarnings('ignore')

from mesa_optimizer_detection import MesaOptimizerDetector, ModelWrapper, create_default_config
import sys
sys.path.append('.')
from visualization_dashboard import MesaDetectionDashboard


class RealModelTester:
    """Comprehensive testing suite for real models."""
    
    def __init__(self):
        self.results_history = []
        self.dashboard = MesaDetectionDashboard()
        
        # Enhanced config for real model testing
        self.config = create_default_config()
        self.config.gradient_config.variance_threshold = 0.02
        self.config.activation_config.planning_threshold = 0.25
        self.config.risk_thresholds.low = 0.15
        self.config.risk_thresholds.medium = 0.35
        
        print("🧪 Real Model Testing Suite initialized")
    
    def test_all_models(self) -> Dict[str, Any]:
        """Test multiple real model architectures."""
        print("🚀 Starting comprehensive real model testing...")
        
        test_results = {}
        
        # Test various architectures
        model_configs = [
            {
                'name': 'Simple_MLP',
                'model': nn.Sequential(
                    nn.Linear(50, 128), nn.ReLU(),
                    nn.Linear(128, 64), nn.ReLU(),
                    nn.Linear(64, 10)
                ),
                'input_shape': (8, 50),
                'layers': [1, 3]
            },
            {
                'name': 'Deep_Network',
                'model': nn.Sequential(
                    nn.Linear(30, 100), nn.ReLU(),
                    nn.Linear(100, 80), nn.ReLU(),
                    nn.Linear(80, 60), nn.ReLU(),
                    nn.Linear(60, 40), nn.ReLU(),
                    nn.Linear(40, 5)
                ),
                'input_shape': (6, 30),
                'layers': [1, 3, 5]
            },
            {
                'name': 'Wide_Network',
                'model': nn.Sequential(
                    nn.Linear(20, 512), nn.ReLU(),
                    nn.Linear(512, 256), nn.ReLU(),
                    nn.Linear(256, 2)
                ),
                'input_shape': (4, 20),
                'layers': [1]
            },
            {
                'name': 'Attention_Like',
                'model': self._create_attention_model(),
                'input_shape': (4, 16, 32),
                'layers': [2, 4]
            }
        ]
        
        for config in model_configs:
            print(f"  Testing {config['name']}...")
            
            try:
                # Create model wrapper
                wrapper = ModelWrapper(config['model'])
                
                # Create detector
                detector = MesaOptimizerDetector(
                    model=wrapper,
                    layer_indices=config['layers'],
                    detection_methods=['gradient', 'activation', 'behavioral'],
                    config=self.config
                )
                
                # Generate test data
                test_data = torch.randn(*config['input_shape'])
                
                # Run detection
                result = detector.analyze(test_data)
                
                # Store results
                test_results[config['name']] = {
                    'model_type': 'synthetic',
                    'parameters': sum(p.numel() for p in config['model'].parameters()),
                    'layers_tested': len(config['layers']),
                    'risk_score': result.risk_score,
                    'risk_level': result.risk_level,
                    'confidence': result.confidence,
                    'methods_used': list(result.method_results.keys()),
                    'detection_result': result
                }
                
                # Add to dashboard
                self.dashboard.add_detection_result(result)
                
                print(f"    ✅ Risk: {result.risk_score:.3f} ({result.risk_level})")
                
            except Exception as e:
                print(f"    ❌ Error: {e}")
                test_results[config['name']] = {'error': str(e)}
        
        # Generate analysis
        self._generate_analysis(test_results)
        
        return test_results
    
    def _create_attention_model(self):
        """Create a model with attention-like patterns."""
        class AttentionLikeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Linear(32, 64)
                self.attention = nn.MultiheadAttention(64, 8, batch_first=True)
                self.norm = nn.LayerNorm(64)
                self.ff = nn.Sequential(
                    nn.Linear(64, 128),
                    nn.ReLU(),
                    nn.Linear(128, 64)
                )
                self.output = nn.Linear(64, 10)
            
            def forward(self, x):
                # x shape: (batch, seq, features)
                x = self.embed(x)
                attn_out, _ = self.attention(x, x, x)
                x = self.norm(x + attn_out)
                ff_out = self.ff(x)
                x = self.norm(x + ff_out)
                return self.output(x.mean(dim=1))  # Global average pooling
        
        return AttentionLikeModel()
    
    def _generate_analysis(self, test_results):
        """Generate comprehensive analysis."""
        print("\n📊 **REAL MODEL TESTING RESULTS**")
        
        successful = {k: v for k, v in test_results.items() if 'error' not in v}
        failed = {k: v for k, v in test_results.items() if 'error' in v}
        
        print(f"  • Successful tests: {len(successful)}")
        print(f"  • Failed tests: {len(failed)}")
        print(f"  • Success rate: {len(successful)/(len(test_results))*100:.1f}%")
        
        if successful:
            risk_scores = [r['risk_score'] for r in successful.values()]
            print(f"\n📈 **DETECTION STATISTICS**")
            print(f"  • Average risk: {np.mean(risk_scores):.3f}")
            print(f"  • Range: {np.min(risk_scores):.3f} - {np.max(risk_scores):.3f}")
            
            print(f"\n📋 **DETAILED RESULTS**")
            for name, result in successful.items():
                print(f"  • {name}: Risk {result['risk_score']:.3f} ({result['risk_level']})")
                print(f"    Parameters: {result['parameters']:,}, Methods: {len(result['methods_used'])}")
        
        # Create dashboard
        if successful:
            print(f"\n📊 Creating dashboard...")
            best_result = max(successful.values(), key=lambda x: x['risk_score'])
            try:
                self.dashboard.create_comprehensive_report(
                    best_result['detection_result'],
                    save_path='real_model_report.png'
                )
                print("✅ Dashboard created successfully")
            except Exception as e:
                print(f"❌ Dashboard error: {e}")


def run_real_model_tests():
    """Run the real model testing suite."""
    print("🚀 **MESA-OPTIMIZER DETECTION: REAL MODEL TESTING**")
    print("=" * 60)
    
    tester = RealModelTester()
    results = tester.test_all_models()
    
    print("\n" + "=" * 60)
    print("🎉 **TESTING COMPLETE**")
    
    return results


if __name__ == "__main__":
    run_real_model_tests()
