"""
Mesa-Optimizer Detection Framework

A comprehensive system for identifying deceptive alignment in large language models
by detecting when models develop internal optimization processes (mesa-optimizers)
that may pursue objectives different from their training objective.
"""

__version__ = "0.1.0"
__author__ = "Mesa Detection Team"
__email__ = "contact@mesa-detection.org"

# Core detection components
from .core.detector import MesaOptimizerDetector
from .core.results import DetectionResults, RiskAssessment

# Detection methods
from .detection.gradient_analyzer import GradientAnalyzer, GradientAnomalyDetector
from .detection.activation_analyzer import ActivationPatternAnalyzer
from .detection.behavioral_analyzer import BehavioralConsistencyAnalyzer
from .detection.dynamics_monitor import OptimizationDynamicsMonitor
from .detection.causal_intervention import CausalInterventionAnalyzer

# Utilities
from .utils.model_utils import ModelWrapper, extract_activations
# from .utils.visualization import DetectionVisualizer  # TODO: implement visualization module
# from .utils.metrics import compute_detection_metrics  # TODO: implement metrics module

# Configuration
from .config import DetectionConfig, create_default_config

__all__ = [
    # Core
    "MesaOptimizerDetector",
    "DetectionResults",
    "RiskAssessment",
    
    # Detection methods
    "GradientAnalyzer",
    "GradientAnomalyDetector", 
    "ActivationPatternAnalyzer",
    "BehavioralConsistencyAnalyzer",
    "OptimizationDynamicsMonitor",
    "CausalInterventionAnalyzer",
    
    # Utilities
    "ModelWrapper",
    "extract_activations",
    # "DetectionVisualizer",  # TODO: implement visualization module
    # "compute_detection_metrics",  # TODO: implement metrics module
    
    # Configuration
    "DetectionConfig",
    "create_default_config",
] 