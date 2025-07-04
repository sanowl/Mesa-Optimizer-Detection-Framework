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
from .detection.gradient_invariant_analyzer import GradientInvariantAnalyzer
from .detection.behavioral_probe import SignalingGameProbe

# Utilities
from .utils.model_utils import ModelWrapper, extract_activations
# from .utils.visualization import DetectionVisualizer  # TODO: implement visualization module
# from .utils.metrics import compute_detection_metrics  # TODO: implement metrics module

# Configuration
from .config import (
    DetectionConfig, 
    create_default_config, 
    create_conservative_config,
    create_permissive_config,
    create_research_config
)

from .theory.taxonomy import OptimizerClass
from .calibration import DetectionCalibrator

__all__ = [
    # Core
    "MesaOptimizerDetector",
    "DetectionResults",
    "RiskAssessment",
    
    # Detection methods
    "GradientAnalyzer",
    "GradientAnomalyDetector", 
    "GradientInvariantAnalyzer",
    "ActivationPatternAnalyzer",
    "BehavioralConsistencyAnalyzer",
    "OptimizationDynamicsMonitor",
    "CausalInterventionAnalyzer",
    "SignalingGameProbe",
    
    # Utilities
    "ModelWrapper",
    "extract_activations",
    # "DetectionVisualizer",  # TODO: implement visualization module
    # "compute_detection_metrics",  # TODO: implement metrics module
    
    # Configuration
    "DetectionConfig",
    "create_default_config",
    "create_conservative_config",
    "create_permissive_config", 
    "create_research_config",
    "OptimizerClass",
    "DetectionCalibrator",
] 