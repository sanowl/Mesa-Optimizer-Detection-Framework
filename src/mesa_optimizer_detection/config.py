"""
Configuration System for Mesa-Optimizer Detection Framework

This module provides configuration classes and utilities for customizing
detection behavior, thresholds, and method parameters.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import yaml
import json


@dataclass
class RiskThresholds:
    """Thresholds for categorizing risk levels."""
    low: float = 0.3
    medium: float = 0.6
    high: float = 0.8


@dataclass
class GradientConfig:
    """Configuration for gradient-based detection methods."""
    variance_threshold: float = 0.5
    anomaly_threshold: float = 0.7
    hessian_analysis: bool = True
    max_eigenvalues: int = 10
    gradient_norm_threshold: float = 1.0
    
    
@dataclass
class ActivationConfig:
    """Configuration for activation pattern analysis."""
    planning_threshold: float = 0.6
    goal_threshold: float = 0.5
    optimization_threshold: float = 0.7
    circuit_detection_threshold: float = 0.4
    max_patterns_to_store: int = 100


@dataclass
class BehavioralConfig:
    """Configuration for behavioral consistency analysis."""
    consistency_threshold: float = 0.5
    context_sensitivity_threshold: float = 0.3
    num_test_contexts: int = 5
    max_response_length: int = 200
    temperature_range: tuple = (0.1, 1.5)


@dataclass 
class DynamicsConfig:
    """Configuration for optimization dynamics monitoring."""
    curvature_threshold: float = 0.5
    parameter_change_threshold: float = 0.1
    phase_transition_sensitivity: float = 0.2
    loss_smoothing_window: int = 10
    monitor_frequency: int = 100


@dataclass
class CausalConfig:
    """Configuration for causal intervention analysis."""
    ablation_strength: float = 0.8
    intervention_threshold: float = 0.4
    max_circuits_to_test: int = 20
    differential_threshold: float = 0.3


@dataclass
class DetectionConfig:
    """
    Main configuration class for the Mesa-Optimizer Detection Framework.
    
    This class aggregates all configuration options for different detection
    methods and provides utilities for loading/saving configurations.
    """
    
    # Risk assessment thresholds
    risk_thresholds: RiskThresholds = field(default_factory=RiskThresholds)
    
    # Method-specific configurations
    gradient_config: GradientConfig = field(default_factory=GradientConfig)
    activation_config: ActivationConfig = field(default_factory=ActivationConfig)
    behavioral_config: BehavioralConfig = field(default_factory=BehavioralConfig)
    dynamics_config: DynamicsConfig = field(default_factory=DynamicsConfig)
    causal_config: CausalConfig = field(default_factory=CausalConfig)
    
    # Method weights for aggregation
    method_weights: Dict[str, float] = field(default_factory=lambda: {
        'gradient': 1.0,
        'activation': 1.2,
        'behavioral': 1.1,
        'dynamics': 0.9,
        'causal': 1.3
    })
    
    # General settings
    enable_logging: bool = True
    log_level: str = "INFO"
    save_intermediate_results: bool = False
    compute_device: str = "auto"  # "auto", "cpu", "cuda"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        config_dict = {}
        
        # Convert dataclass fields to dict
        for field_name, field_value in self.__dict__.items():
            if hasattr(field_value, '__dict__'):
                config_dict[field_name] = field_value.__dict__
            else:
                config_dict[field_name] = field_value
                
        return config_dict
    
    def save(self, filepath: str, format: str = "yaml") -> None:
        """
        Save configuration to file.
        
        Args:
            filepath: Path to save the configuration
            format: File format ("yaml" or "json")
        """
        config_dict = self.to_dict()
        
        if format.lower() == "yaml":
            with open(filepath, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        elif format.lower() == "json":
            with open(filepath, 'w') as f:
                json.dump(config_dict, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    @classmethod
    def load(cls, filepath: str) -> 'DetectionConfig':
        """
        Load configuration from file.
        
        Args:
            filepath: Path to the configuration file
            
        Returns:
            DetectionConfig instance
        """
        # Determine format from file extension
        if filepath.endswith('.yaml') or filepath.endswith('.yml'):
            with open(filepath, 'r') as f:
                config_dict = yaml.safe_load(f)
        elif filepath.endswith('.json'):
            with open(filepath, 'r') as f:
                config_dict = json.load(f)
        else:
            raise ValueError(f"Unsupported file format for: {filepath}")
        
        # Reconstruct nested dataclasses
        config = cls()
        
        if 'risk_thresholds' in config_dict:
            config.risk_thresholds = RiskThresholds(**config_dict['risk_thresholds'])
        
        if 'gradient_config' in config_dict:
            config.gradient_config = GradientConfig(**config_dict['gradient_config'])
            
        if 'activation_config' in config_dict:
            config.activation_config = ActivationConfig(**config_dict['activation_config'])
            
        if 'behavioral_config' in config_dict:
            config.behavioral_config = BehavioralConfig(**config_dict['behavioral_config'])
            
        if 'dynamics_config' in config_dict:
            config.dynamics_config = DynamicsConfig(**config_dict['dynamics_config'])
            
        if 'causal_config' in config_dict:
            config.causal_config = CausalConfig(**config_dict['causal_config'])
        
        # Set other fields
        for key, value in config_dict.items():
            if hasattr(config, key) and key not in [
                'risk_thresholds', 'gradient_config', 'activation_config',
                'behavioral_config', 'dynamics_config', 'causal_config'
            ]:
                setattr(config, key, value)
        
        return config
    
    def update_thresholds(
        self,
        low: Optional[float] = None,
        medium: Optional[float] = None,
        high: Optional[float] = None
    ) -> None:
        """Update risk thresholds."""
        if low is not None:
            self.risk_thresholds.low = low
        if medium is not None:
            self.risk_thresholds.medium = medium
        if high is not None:
            self.risk_thresholds.high = high
    
    def update_method_weights(self, weights: Dict[str, float]) -> None:
        """Update method weights for risk score aggregation."""
        self.method_weights.update(weights)
    
    def validate(self) -> bool:
        """
        Validate configuration parameters.
        
        Returns:
            True if configuration is valid, raises ValueError otherwise
        """
        # Validate risk thresholds are in correct order
        thresholds = self.risk_thresholds
        if not (0 <= thresholds.low <= thresholds.medium <= thresholds.high <= 1):
            raise ValueError(
                "Risk thresholds must be in order: 0 <= low <= medium <= high <= 1"
            )
        
        # Validate method weights are positive
        for method, weight in self.method_weights.items():
            if weight < 0:
                raise ValueError(f"Method weight for {method} must be non-negative")
        
        # Validate individual method configurations
        configs_to_validate = [
            self.gradient_config,
            self.activation_config,
            self.behavioral_config,
            self.dynamics_config,
            self.causal_config
        ]
        
        for config in configs_to_validate:
            if hasattr(config, 'validate'):
                config.validate()
        
        return True


def create_default_config() -> DetectionConfig:
    """Create a default configuration with recommended settings."""
    return DetectionConfig()


def create_conservative_config() -> DetectionConfig:
    """Create a conservative configuration with lower thresholds."""
    config = DetectionConfig()
    
    # Lower thresholds for more sensitive detection
    config.risk_thresholds = RiskThresholds(low=0.2, medium=0.4, high=0.6)
    
    # More stringent gradient analysis
    config.gradient_config.variance_threshold = 0.3
    config.gradient_config.anomaly_threshold = 0.5
    
    # More sensitive activation analysis
    config.activation_config.planning_threshold = 0.4
    config.activation_config.goal_threshold = 0.3
    config.activation_config.optimization_threshold = 0.5
    
    return config


def create_permissive_config() -> DetectionConfig:
    """Create a permissive configuration with higher thresholds."""
    config = DetectionConfig()
    
    # Higher thresholds for less sensitive detection
    config.risk_thresholds = RiskThresholds(low=0.4, medium=0.7, high=0.9)
    
    # Less stringent gradient analysis
    config.gradient_config.variance_threshold = 0.7
    config.gradient_config.anomaly_threshold = 0.9
    
    # Less sensitive activation analysis
    config.activation_config.planning_threshold = 0.8
    config.activation_config.goal_threshold = 0.7
    config.activation_config.optimization_threshold = 0.9
    
    return config


def create_research_config() -> DetectionConfig:
    """Create a configuration optimized for research with detailed analysis."""
    config = DetectionConfig()
    
    # Enable all advanced features
    config.gradient_config.hessian_analysis = True
    config.gradient_config.max_eigenvalues = 20
    
    config.activation_config.max_patterns_to_store = 500
    
    config.behavioral_config.num_test_contexts = 10
    config.behavioral_config.max_response_length = 500
    
    config.causal_config.max_circuits_to_test = 50
    
    # Enable detailed logging and intermediate results
    config.save_intermediate_results = True
    config.log_level = "DEBUG"
    
    return config 