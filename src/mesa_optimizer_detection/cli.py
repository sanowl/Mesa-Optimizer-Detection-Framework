"""
Command Line Interface for Mesa-Optimizer Detection Framework

This module provides a CLI for running mesa-optimizer detection analysis
from the command line.
"""

import argparse
import json
import sys
import logging
from pathlib import Path
from typing import Optional, List

import torch
import yaml

from . import MesaOptimizerDetector, DetectionConfig, create_default_config
from .utils.model_utils import ModelWrapper


def setup_logging(level: str = "INFO"):
    """Setup logging for the CLI."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


def load_model(model_path: str, model_type: str = "pytorch"):
    """Load a model from file."""
    if model_type == "pytorch":
        try:
            model = torch.load(model_path, map_location='cpu')
            return model
        except Exception as e:
            print(f"Error loading PyTorch model: {e}")
            sys.exit(1)
    elif model_type == "huggingface":
        try:
            from transformers import AutoModel, AutoTokenizer
            model = AutoModel.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            return model, tokenizer
        except Exception as e:
            print(f"Error loading HuggingFace model: {e}")
            sys.exit(1)
    else:
        print(f"Unsupported model type: {model_type}")
        sys.exit(1)


def create_test_inputs(input_size: int, batch_size: int = 4) -> torch.Tensor:
    """Create test inputs for analysis."""
    return torch.randn(batch_size, input_size)


def analyze_model(args):
    """Perform mesa-optimization analysis on a model."""
    print("Mesa-Optimizer Detection Framework")
    print("=" * 40)
    
    # Load configuration
    if args.config:
        config = DetectionConfig.load(args.config)
        print(f"Loaded configuration from: {args.config}")
    else:
        config = create_default_config()
        print("Using default configuration")
    
    # Load model
    print(f"Loading model from: {args.model}")
    if args.model_type == "huggingface":
        model, tokenizer = load_model(args.model, args.model_type)
    else:
        model = load_model(args.model, args.model_type)
        tokenizer = None
    
    print(f"Model loaded successfully")
    print(f"Model type: {type(model).__name__}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Initialize detector
    layer_indices = args.layers or [len(list(model.modules())) // 4, 
                                   len(list(model.modules())) // 2,
                                   3 * len(list(model.modules())) // 4]
    
    detector = MesaOptimizerDetector(
        model=model,
        layer_indices=layer_indices,
        detection_methods=args.methods,
        config=config
    )
    
    print(f"Detector initialized with methods: {args.methods}")
    print(f"Analyzing layers: {layer_indices}")
    
    # Create or load inputs
    if args.input_data:
        print(f"Loading input data from: {args.input_data}")
        input_batch = torch.load(args.input_data)
    else:
        print("Creating test inputs")
        input_batch = create_test_inputs(args.input_size, args.batch_size)
    
    print(f"Input shape: {input_batch.shape}")
    
    # Perform analysis
    print("\nRunning analysis...")
    try:
        results = detector.analyze(input_batch)
        
        # Display results
        print("\nAnalysis Results:")
        print("=" * 20)
        print(f"Risk Level: {results.risk_level}")
        print(f"Risk Score: {results.risk_score:.3f}")
        print(f"Confidence: {results.confidence:.3f}")
        
        print("\nMethod Scores:")
        for method, score in results.risk_assessment.method_scores.items():
            print(f"  {method.capitalize():12}: {score:.3f}")
        
        print("\nRecommendations:")
        for i, rec in enumerate(results.risk_assessment.recommendations, 1):
            print(f"  {i}. {rec}")
        
        # Save results
        if args.output:
            results.save(args.output)
            print(f"\nResults saved to: {args.output}")
        
        # Generate report
        if args.report:
            report = results.generate_report()
            with open(args.report, 'w') as f:
                f.write(report)
            print(f"Report saved to: {args.report}")
        
        return results.risk_score
        
    except Exception as e:
        print(f"Analysis failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def create_config(args):
    """Create a new configuration file."""
    if args.template == "default":
        config = create_default_config()
    elif args.template == "conservative":
        from . import create_conservative_config
        config = create_conservative_config()
    elif args.template == "permissive":
        from . import create_permissive_config
        config = create_permissive_config()
    else:
        print(f"Unknown template: {args.template}")
        sys.exit(1)
    
    config.save(args.output, format=args.format)
    print(f"Configuration saved to: {args.output}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Mesa-Optimizer Detection Framework CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze a PyTorch model
  mesa-detect analyze --model model.pth --output results.json
  
  # Analyze with custom configuration
  mesa-detect analyze --model model.pth --config config.yaml
  
  # Analyze HuggingFace model
  mesa-detect analyze --model bert-base-uncased --model-type huggingface
  
  # Create configuration file
  mesa-detect config --template conservative --output config.yaml
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze a model for mesa-optimization')
    analyze_parser.add_argument('--model', required=True, help='Path to model file or HuggingFace model name')
    analyze_parser.add_argument('--model-type', choices=['pytorch', 'huggingface'], default='pytorch',
                               help='Type of model to load')
    analyze_parser.add_argument('--config', help='Path to configuration file')
    analyze_parser.add_argument('--methods', nargs='+', 
                               choices=['gradient', 'activation', 'behavioral', 'dynamics', 'causal'],
                               default=['gradient', 'activation', 'behavioral'],
                               help='Detection methods to use')
    analyze_parser.add_argument('--layers', type=int, nargs='+', help='Layer indices to analyze')
    analyze_parser.add_argument('--input-data', help='Path to input data file (.pt)')
    analyze_parser.add_argument('--input-size', type=int, default=512, help='Input size for test data')
    analyze_parser.add_argument('--batch-size', type=int, default=4, help='Batch size for analysis')
    analyze_parser.add_argument('--output', help='Output file for results (.json)')
    analyze_parser.add_argument('--report', help='Output file for human-readable report (.txt)')
    analyze_parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    
    # Config command
    config_parser = subparsers.add_parser('config', help='Create configuration file')
    config_parser.add_argument('--template', choices=['default', 'conservative', 'permissive'], 
                              default='default', help='Configuration template')
    config_parser.add_argument('--output', required=True, help='Output configuration file')
    config_parser.add_argument('--format', choices=['yaml', 'json'], default='yaml',
                              help='Configuration file format')
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Setup logging
    log_level = "DEBUG" if args.verbose and hasattr(args, 'verbose') else "INFO"
    setup_logging(log_level)
    
    # Execute command
    if args.command == 'analyze':
        exit_code = analyze_model(args)
        sys.exit(exit_code)
    elif args.command == 'config':
        create_config(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main() 