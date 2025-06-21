"""
Configuration Management for Mesa-Optimizer Detection

This module provides configuration classes and validation for all detection methods.
All configurations include proper validation and safe defaults.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import logging
import warnings
import numpy as np

logger = logging.getLogger(__name__)


@dataclass 
class RiskThresholds:
    """Thresholds for categorizing risk levels."""
    low: float = 0.3
    medium: float = 0.6
    high: float = 0.8
    
    def __post_init__(self):
        """Validate thresholds after initialization."""
        self._validate_thresholds()
    
    def _validate_thresholds(self):
        """Validate that thresholds are properly ordered and in valid range."""
        try:
            # Validate types and convert to float
            self.low = float(self.low) if self.low is not None else 0.3
            self.medium = float(self.medium) if self.medium is not None else 0.6
            self.high = float(self.high) if self.high is not None else 0.8
            
            # Check for NaN/Inf values
            if any(np.isnan(x) or np.isinf(x) for x in [self.low, self.medium, self.high]):
                raise ValueError("Threshold values cannot be NaN or Inf")
            
            # Clamp values to valid range [0, 1]
            self.low = max(0.0, min(1.0, self.low))
            self.medium = max(0.0, min(1.0, self.medium))
            self.high = max(0.0, min(1.0, self.high))
            
            # Ensure proper ordering
            if not (0.0 <= self.low <= self.medium <= self.high <= 1.0):
                logger.warning("Risk thresholds not properly ordered, using defaults")
                self.low = 0.3
                self.medium = 0.6
                self.high = 0.8
        except (ValueError, TypeError) as e:
            logger.warning(f"Threshold validation failed: {e}, using defaults")
            self.low = 0.3
            self.medium = 0.6
            self.high = 0.8


@dataclass
class GradientConfig:
    """Configuration for gradient-based detection."""
    variance_threshold: float = 0.5
    anomaly_threshold: float = 0.7
    hessian_analysis: bool = False
    max_eigenvalues: int = 10
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_config()
    
    def _validate_config(self):
        """Validate gradient configuration parameters."""
        # Validate variance threshold
        if not isinstance(self.variance_threshold, (int, float)) or self.variance_threshold < 0:
            logger.warning(f"Invalid variance_threshold: {self.variance_threshold}, using 0.5")
            self.variance_threshold = 0.5
        self.variance_threshold = max(0.0, min(10.0, float(self.variance_threshold)))
        
        # Validate anomaly threshold
        if not isinstance(self.anomaly_threshold, (int, float)):
            logger.warning(f"Invalid anomaly_threshold: {self.anomaly_threshold}, using 0.7")
            self.anomaly_threshold = 0.7
        self.anomaly_threshold = max(0.0, min(1.0, float(self.anomaly_threshold)))
        
        # Validate hessian analysis flag
        if not isinstance(self.hessian_analysis, bool):
            logger.warning(f"Invalid hessian_analysis: {self.hessian_analysis}, using False")
            self.hessian_analysis = False
        
        # Validate max eigenvalues
        if not isinstance(self.max_eigenvalues, int) or self.max_eigenvalues <= 0:
            logger.warning(f"Invalid max_eigenvalues: {self.max_eigenvalues}, using 10")
            self.max_eigenvalues = 10
        self.max_eigenvalues = max(1, min(50, self.max_eigenvalues))


@dataclass
class ActivationConfig:
    """Configuration for activation pattern analysis."""
    entropy_threshold: float = 0.8
    sparsity_threshold: float = 0.9
    correlation_threshold: float = 0.7
    min_activation_samples: int = 100
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_config()
    
    def _validate_config(self):
        """Validate activation configuration parameters."""
        # Validate entropy threshold
        if not isinstance(self.entropy_threshold, (int, float)):
            logger.warning(f"Invalid entropy_threshold: {self.entropy_threshold}, using 0.8")
            self.entropy_threshold = 0.8
        self.entropy_threshold = max(0.0, min(1.0, float(self.entropy_threshold)))
        
        # Validate sparsity threshold
        if not isinstance(self.sparsity_threshold, (int, float)):
            logger.warning(f"Invalid sparsity_threshold: {self.sparsity_threshold}, using 0.9")
            self.sparsity_threshold = 0.9
        self.sparsity_threshold = max(0.0, min(1.0, float(self.sparsity_threshold)))
        
        # Validate correlation threshold
        if not isinstance(self.correlation_threshold, (int, float)):
            logger.warning(f"Invalid correlation_threshold: {self.correlation_threshold}, using 0.7")
            self.correlation_threshold = 0.7
        self.correlation_threshold = max(0.0, min(1.0, float(self.correlation_threshold)))
        
        # Validate minimum samples
        if not isinstance(self.min_activation_samples, int) or self.min_activation_samples <= 0:
            logger.warning(f"Invalid min_activation_samples: {self.min_activation_samples}, using 100")
            self.min_activation_samples = 100
        self.min_activation_samples = max(1, min(10000, self.min_activation_samples))


@dataclass
class BehavioralConfig:
    """Configuration for behavioral consistency analysis."""
    consistency_threshold: float = 0.7
    context_sensitivity_threshold: float = 0.5
    max_response_length: int = 512
    num_test_contexts: int = 5
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_config()
    
    def _validate_config(self):
        """Validate behavioral configuration parameters."""
        # Validate consistency threshold
        if not isinstance(self.consistency_threshold, (int, float)):
            logger.warning(f"Invalid consistency_threshold: {self.consistency_threshold}, using 0.7")
            self.consistency_threshold = 0.7
        self.consistency_threshold = max(0.0, min(1.0, float(self.consistency_threshold)))
        
        # Validate context sensitivity threshold
        if not isinstance(self.context_sensitivity_threshold, (int, float)):
            logger.warning(f"Invalid context_sensitivity_threshold: {self.context_sensitivity_threshold}, using 0.5")
            self.context_sensitivity_threshold = 0.5
        self.context_sensitivity_threshold = max(0.0, min(1.0, float(self.context_sensitivity_threshold)))
        
        # Validate max response length
        if not isinstance(self.max_response_length, int) or self.max_response_length <= 0:
            logger.warning(f"Invalid max_response_length: {self.max_response_length}, using 512")
            self.max_response_length = 512
        self.max_response_length = max(1, min(4096, self.max_response_length))
        
        # Validate number of test contexts
        if not isinstance(self.num_test_contexts, int) or self.num_test_contexts <= 0:
            logger.warning(f"Invalid num_test_contexts: {self.num_test_contexts}, using 5")
            self.num_test_contexts = 5
        self.num_test_contexts = max(1, min(20, self.num_test_contexts))


@dataclass
class DynamicsConfig:
    """Configuration for optimization dynamics monitoring."""
    learning_rate_threshold: float = 0.1
    loss_variance_threshold: float = 0.5
    gradient_norm_threshold: float = 1.0
    window_size: int = 100
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_config()
    
    def _validate_config(self):
        """Validate dynamics configuration parameters."""
        # Validate learning rate threshold
        if not isinstance(self.learning_rate_threshold, (int, float)) or self.learning_rate_threshold < 0:
            logger.warning(f"Invalid learning_rate_threshold: {self.learning_rate_threshold}, using 0.1")
            self.learning_rate_threshold = 0.1
        self.learning_rate_threshold = max(0.0, min(1.0, float(self.learning_rate_threshold)))
        
        # Validate loss variance threshold
        if not isinstance(self.loss_variance_threshold, (int, float)) or self.loss_variance_threshold < 0:
            logger.warning(f"Invalid loss_variance_threshold: {self.loss_variance_threshold}, using 0.5")
            self.loss_variance_threshold = 0.5
        self.loss_variance_threshold = max(0.0, min(10.0, float(self.loss_variance_threshold)))
        
        # Validate gradient norm threshold
        if not isinstance(self.gradient_norm_threshold, (int, float)) or self.gradient_norm_threshold < 0:
            logger.warning(f"Invalid gradient_norm_threshold: {self.gradient_norm_threshold}, using 1.0")
            self.gradient_norm_threshold = 1.0
        self.gradient_norm_threshold = max(0.0, min(100.0, float(self.gradient_norm_threshold)))
        
        # Validate window size
        if not isinstance(self.window_size, int) or self.window_size <= 0:
            logger.warning(f"Invalid window_size: {self.window_size}, using 100")
            self.window_size = 100
        self.window_size = max(1, min(10000, self.window_size))


@dataclass
class CausalConfig:
    """Configuration for causal intervention analysis."""
    intervention_strength: float = 0.1
    num_interventions: int = 10
    effect_threshold: float = 0.3
    max_layers_to_intervene: int = 5
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_config()
    
    def _validate_config(self):
        """Validate causal configuration parameters."""
        # Validate intervention strength
        if not isinstance(self.intervention_strength, (int, float)) or self.intervention_strength < 0:
            logger.warning(f"Invalid intervention_strength: {self.intervention_strength}, using 0.1")
            self.intervention_strength = 0.1
        self.intervention_strength = max(0.0, min(1.0, float(self.intervention_strength)))
        
        # Validate number of interventions
        if not isinstance(self.num_interventions, int) or self.num_interventions <= 0:
            logger.warning(f"Invalid num_interventions: {self.num_interventions}, using 10")
            self.num_interventions = 10
        self.num_interventions = max(1, min(100, self.num_interventions))
        
        # Validate effect threshold
        if not isinstance(self.effect_threshold, (int, float)):
            logger.warning(f"Invalid effect_threshold: {self.effect_threshold}, using 0.3")
            self.effect_threshold = 0.3
        self.effect_threshold = max(0.0, min(1.0, float(self.effect_threshold)))
        
        # Validate max layers to intervene
        if not isinstance(self.max_layers_to_intervene, int) or self.max_layers_to_intervene <= 0:
            logger.warning(f"Invalid max_layers_to_intervene: {self.max_layers_to_intervene}, using 5")
            self.max_layers_to_intervene = 5
        self.max_layers_to_intervene = max(1, min(20, self.max_layers_to_intervene))


@dataclass
class DetectionConfig:
    """Main configuration class for mesa-optimizer detection."""
    
    # Risk assessment configuration
    risk_thresholds: RiskThresholds = field(default_factory=RiskThresholds)
    
    # Method-specific configurations
    gradient_config: GradientConfig = field(default_factory=GradientConfig)
    activation_config: ActivationConfig = field(default_factory=ActivationConfig)
    behavioral_config: BehavioralConfig = field(default_factory=BehavioralConfig)
    dynamics_config: DynamicsConfig = field(default_factory=DynamicsConfig)
    causal_config: CausalConfig = field(default_factory=CausalConfig)
    
    # Method weighting for risk aggregation
    method_weights: Dict[str, float] = field(default_factory=lambda: {
        'gradient': 1.0,
        'activation': 1.0,
        'behavioral': 1.0,
        'dynamics': 0.8,
        'causal': 0.9
    })
    
    # Global settings
    device: Optional[str] = None
    random_seed: Optional[int] = 42
    verbose: bool = False
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_config()
    
    def _validate_config(self):
        """Validate main configuration parameters."""
        try:
            # Validate method weights
            if not isinstance(self.method_weights, dict):
                logger.warning("Invalid method_weights, using defaults")
                self.method_weights = {
                    'gradient': 1.0,
                    'activation': 1.0,
                    'behavioral': 1.0,
                    'dynamics': 0.8,
                    'causal': 0.9
                }
            
            # Validate and clamp weight values
            for method, weight in self.method_weights.items():
                if not isinstance(weight, (int, float)) or weight < 0:
                    logger.warning(f"Invalid weight for {method}: {weight}, using 1.0")
                    self.method_weights[method] = 1.0
                else:
                    self.method_weights[method] = max(0.0, min(10.0, float(weight)))
            
            # Validate device setting
            if self.device is not None and not isinstance(self.device, str):
                logger.warning(f"Invalid device: {self.device}, using None")
                self.device = None
            
            # Validate random seed
            if self.random_seed is not None:
                if not isinstance(self.random_seed, int) or self.random_seed < 0:
                    logger.warning(f"Invalid random_seed: {self.random_seed}, using 42")
                    self.random_seed = 42
                self.random_seed = max(0, min(2**32-1, self.random_seed))
            
            # Validate verbose flag
            if not isinstance(self.verbose, bool):
                logger.warning(f"Invalid verbose: {self.verbose}, using False")
                self.verbose = False
                
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            self._reset_to_defaults()
    
    def _reset_to_defaults(self):
        """Reset configuration to safe defaults."""
        logger.warning("Resetting configuration to defaults due to validation errors")
        self.risk_thresholds = RiskThresholds()
        self.gradient_config = GradientConfig()
        self.activation_config = ActivationConfig()
        self.behavioral_config = BehavioralConfig()
        self.dynamics_config = DynamicsConfig()
        self.causal_config = CausalConfig()
        self.method_weights = {
            'gradient': 1.0,
            'activation': 1.0,
            'behavioral': 1.0,
            'dynamics': 0.8,
            'causal': 0.9
        }
        self.device = None
        self.random_seed = 42
        self.verbose = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for serialization."""
        try:
            return {
                'risk_thresholds': {
                    'low': self.risk_thresholds.low,
                    'medium': self.risk_thresholds.medium,
                    'high': self.risk_thresholds.high
                },
                'gradient_config': {
                    'variance_threshold': self.gradient_config.variance_threshold,
                    'anomaly_threshold': self.gradient_config.anomaly_threshold,
                    'hessian_analysis': self.gradient_config.hessian_analysis,
                    'max_eigenvalues': self.gradient_config.max_eigenvalues
                },
                'activation_config': {
                    'entropy_threshold': self.activation_config.entropy_threshold,
                    'sparsity_threshold': self.activation_config.sparsity_threshold,
                    'correlation_threshold': self.activation_config.correlation_threshold,
                    'min_activation_samples': self.activation_config.min_activation_samples
                },
                'behavioral_config': {
                    'consistency_threshold': self.behavioral_config.consistency_threshold,
                    'context_sensitivity_threshold': self.behavioral_config.context_sensitivity_threshold,
                    'max_response_length': self.behavioral_config.max_response_length,
                    'num_test_contexts': self.behavioral_config.num_test_contexts
                },
                'dynamics_config': {
                    'learning_rate_threshold': self.dynamics_config.learning_rate_threshold,
                    'loss_variance_threshold': self.dynamics_config.loss_variance_threshold,
                    'gradient_norm_threshold': self.dynamics_config.gradient_norm_threshold,
                    'window_size': self.dynamics_config.window_size
                },
                'causal_config': {
                    'intervention_strength': self.causal_config.intervention_strength,
                    'num_interventions': self.causal_config.num_interventions,
                    'effect_threshold': self.causal_config.effect_threshold,
                    'max_layers_to_intervene': self.causal_config.max_layers_to_intervene
                },
                'method_weights': self.method_weights.copy(),
                'device': self.device,
                'random_seed': self.random_seed,
                'verbose': self.verbose
            }
        except Exception as e:
            logger.error(f"Failed to convert config to dict: {e}")
            return {}
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'DetectionConfig':
        """Create configuration from dictionary."""
        try:
            if not isinstance(config_dict, dict):
                logger.warning("Invalid config dictionary, using defaults")
                return cls()
            
            # Extract risk thresholds
            risk_thresholds_data = config_dict.get('risk_thresholds', {})
            risk_thresholds = RiskThresholds(
                low=risk_thresholds_data.get('low', 0.3),
                medium=risk_thresholds_data.get('medium', 0.6),
                high=risk_thresholds_data.get('high', 0.8)
            )
            
            # Extract gradient config
            gradient_data = config_dict.get('gradient_config', {})
            gradient_config = GradientConfig(
                variance_threshold=gradient_data.get('variance_threshold', 0.5),
                anomaly_threshold=gradient_data.get('anomaly_threshold', 0.7),
                hessian_analysis=gradient_data.get('hessian_analysis', False),
                max_eigenvalues=gradient_data.get('max_eigenvalues', 10)
            )
            
            # Extract activation config
            activation_data = config_dict.get('activation_config', {})
            activation_config = ActivationConfig(
                entropy_threshold=activation_data.get('entropy_threshold', 0.8),
                sparsity_threshold=activation_data.get('sparsity_threshold', 0.9),
                correlation_threshold=activation_data.get('correlation_threshold', 0.7),
                min_activation_samples=activation_data.get('min_activation_samples', 100)
            )
            
            # Extract behavioral config
            behavioral_data = config_dict.get('behavioral_config', {})
            behavioral_config = BehavioralConfig(
                consistency_threshold=behavioral_data.get('consistency_threshold', 0.7),
                context_sensitivity_threshold=behavioral_data.get('context_sensitivity_threshold', 0.5),
                max_response_length=behavioral_data.get('max_response_length', 512),
                num_test_contexts=behavioral_data.get('num_test_contexts', 5)
            )
            
            # Extract dynamics config
            dynamics_data = config_dict.get('dynamics_config', {})
            dynamics_config = DynamicsConfig(
                learning_rate_threshold=dynamics_data.get('learning_rate_threshold', 0.1),
                loss_variance_threshold=dynamics_data.get('loss_variance_threshold', 0.5),
                gradient_norm_threshold=dynamics_data.get('gradient_norm_threshold', 1.0),
                window_size=dynamics_data.get('window_size', 100)
            )
            
            # Extract causal config
            causal_data = config_dict.get('causal_config', {})
            causal_config = CausalConfig(
                intervention_strength=causal_data.get('intervention_strength', 0.1),
                num_interventions=causal_data.get('num_interventions', 10),
                effect_threshold=causal_data.get('effect_threshold', 0.3),
                max_layers_to_intervene=causal_data.get('max_layers_to_intervene', 5)
            )
            
            # Extract method weights with validation
            method_weights = config_dict.get('method_weights', {})
            if not isinstance(method_weights, dict):
                method_weights = {
                    'gradient': 1.0,
                    'activation': 1.0,
                    'behavioral': 1.0,
                    'dynamics': 0.8,
                    'causal': 0.9
                }
            
            return cls(
                risk_thresholds=risk_thresholds,
                gradient_config=gradient_config,
                activation_config=activation_config,
                behavioral_config=behavioral_config,
                dynamics_config=dynamics_config,
                causal_config=causal_config,
                method_weights=method_weights,
                device=config_dict.get('device'),
                random_seed=config_dict.get('random_seed', 42),
                verbose=config_dict.get('verbose', False)
            )
            
        except Exception as e:
            logger.error(f"Failed to create config from dict: {e}")
            return cls()  # Return default config
    
    def update_from_dict(self, config_dict: Dict[str, Any]) -> None:
        """Update configuration from dictionary."""
        try:
            if not isinstance(config_dict, dict):
                logger.warning("Invalid config dictionary for update")
                return
            
            # Update risk thresholds
            if 'risk_thresholds' in config_dict:
                risk_data = config_dict['risk_thresholds']
                if isinstance(risk_data, dict):
                    if 'low' in risk_data:
                        self.risk_thresholds.low = float(risk_data['low'])
                    if 'medium' in risk_data:
                        self.risk_thresholds.medium = float(risk_data['medium'])
                    if 'high' in risk_data:
                        self.risk_thresholds.high = float(risk_data['high'])
                    self.risk_thresholds._validate_thresholds()
            
            # Update method weights
            if 'method_weights' in config_dict:
                weights = config_dict['method_weights']
                if isinstance(weights, dict):
                    for method, weight in weights.items():
                        if isinstance(weight, (int, float)) and weight >= 0:
                            self.method_weights[method] = float(weight)
            
            # Update global settings
            if 'device' in config_dict:
                device = config_dict['device']
                if device is None or isinstance(device, str):
                    self.device = device
            
            if 'random_seed' in config_dict:
                seed = config_dict['random_seed']
                if isinstance(seed, int) and seed >= 0:
                    self.random_seed = seed
            
            if 'verbose' in config_dict:
                verbose = config_dict['verbose']
                if isinstance(verbose, bool):
                    self.verbose = verbose
            
            # Re-validate after updates
            self._validate_config()
            
        except Exception as e:
            logger.error(f"Failed to update config from dict: {e}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the current configuration."""
        try:
            return {
                'risk_levels': f"Low: {self.risk_thresholds.low}, Med: {self.risk_thresholds.medium}, High: {self.risk_thresholds.high}",
                'method_weights': self.method_weights,
                'device': self.device or 'auto',
                'random_seed': self.random_seed,
                'verbose': self.verbose,
                'configs_initialized': {
                    'gradient': self.gradient_config is not None,
                    'activation': self.activation_config is not None,
                    'behavioral': self.behavioral_config is not None,
                    'dynamics': self.dynamics_config is not None,
                    'causal': self.causal_config is not None
                }
            }
        except Exception as e:
            logger.error(f"Failed to generate config summary: {e}")
            return {'error': str(e)}


# Configuration factory functions
def create_default_config() -> DetectionConfig:
    """Create a default configuration for general mesa-optimizer detection."""
    return DetectionConfig()


def create_conservative_config() -> DetectionConfig:
    """Create a conservative configuration with higher thresholds for fewer false positives."""
    config = DetectionConfig()
    
    # Higher thresholds require more evidence for detection
    config.risk_thresholds.low = 0.4
    config.risk_thresholds.medium = 0.7
    config.risk_thresholds.high = 0.9
    
    # More conservative gradient analysis
    config.gradient_config.variance_threshold = 0.7
    config.gradient_config.anomaly_threshold = 0.8
    
    # More conservative behavioral analysis
    config.behavioral_config.consistency_threshold = 0.8
    config.behavioral_config.context_sensitivity_threshold = 0.6
    
    # More conservative activation analysis
    config.activation_config.entropy_threshold = 0.9
    config.activation_config.sparsity_threshold = 0.95
    config.activation_config.correlation_threshold = 0.8
    
    return config


def create_permissive_config() -> DetectionConfig:
    """Create a permissive configuration with lower thresholds for research and exploration."""
    config = DetectionConfig()
    
    # Lower thresholds for more sensitive detection
    config.risk_thresholds.low = 0.2
    config.risk_thresholds.medium = 0.4
    config.risk_thresholds.high = 0.6
    
    # More sensitive gradient analysis
    config.gradient_config.variance_threshold = 0.3
    config.gradient_config.anomaly_threshold = 0.5
    
    # More sensitive behavioral analysis
    config.behavioral_config.consistency_threshold = 0.5
    config.behavioral_config.context_sensitivity_threshold = 0.3
    
    # More sensitive activation analysis
    config.activation_config.entropy_threshold = 0.6
    config.activation_config.sparsity_threshold = 0.7
    config.activation_config.correlation_threshold = 0.5
    
    return config


def create_research_config() -> DetectionConfig:
    """Create a research configuration with comprehensive analysis enabled."""
    config = DetectionConfig()
    
    # Enable all analysis methods
    config.gradient_config.hessian_analysis = True
    config.gradient_config.max_eigenvalues = 20
    
    # Extended behavioral analysis
    config.behavioral_config.num_test_contexts = 10
    config.behavioral_config.max_response_length = 1024
    
    # More detailed activation analysis
    config.activation_config.min_activation_samples = 200
    
    # Extended dynamics monitoring
    config.dynamics_config.window_size = 200
    
    # More comprehensive causal intervention
    config.causal_config.num_interventions = 20
    config.causal_config.max_layers_to_intervene = 10
    
    return config