"""
Results Classes for Mesa-Optimizer Detection

This module defines result classes for various detection methods with proper
validation, serialization support, and thread-safe operations.
"""

import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union
import json
import logging
import threading
import warnings
from collections import OrderedDict

logger = logging.getLogger(__name__)


def _safe_tensor_to_json(tensor: torch.Tensor) -> Dict[str, Any]:
    """Safely convert tensor to JSON-serializable format."""
    try:
        if not isinstance(tensor, torch.Tensor):
            return {'error': f'Expected tensor, got {type(tensor)}'}
        
        if tensor.numel() == 0:
            return {'shape': list(tensor.shape), 'data': [], 'dtype': str(tensor.dtype)}
        
        # Move to CPU and convert to numpy for serialization
        numpy_data = tensor.detach().cpu().numpy()
        
        # Handle different dtypes
        if numpy_data.dtype in [np.float32, np.float64]:
            # Check for NaN/Inf values
            if np.isnan(numpy_data).any() or np.isinf(numpy_data).any():
                logger.warning("NaN or Inf values detected in tensor")
                # Replace NaN/Inf with None for JSON serialization
                numpy_data = np.where(np.isfinite(numpy_data), numpy_data, None)
        
        return {
            'shape': list(tensor.shape),
            'data': numpy_data.tolist(),
            'dtype': str(tensor.dtype),
            'device': str(tensor.device)
        }
    except Exception as e:
        logger.error(f"Failed to serialize tensor: {e}")
        return {'error': str(e)}


def _safe_json_to_tensor(data: Dict[str, Any]) -> Optional[torch.Tensor]:
    """Safely convert JSON format back to tensor."""
    try:
        if 'error' in data:
            logger.warning(f"Cannot restore tensor due to error: {data['error']}")
            return None
        
        shape = data.get('shape', [])
        tensor_data = data.get('data', [])
        dtype_str = data.get('dtype', 'torch.float32')
        
        if not tensor_data:
            # Empty tensor
            dtype = getattr(torch, dtype_str.split('.')[-1]) if hasattr(torch, dtype_str.split('.')[-1]) else torch.float32
            return torch.empty(shape, dtype=dtype)
        
        # Convert back to numpy first
        numpy_data = np.array(tensor_data)
        
        # Handle None values (from NaN/Inf)
        if numpy_data.dtype == object:
            # Replace None with NaN
            numeric_data = np.where(numpy_data == None, np.nan, numpy_data).astype(np.float32)
            numpy_data = numeric_data
        
        # Create tensor
        tensor = torch.from_numpy(numpy_data)
        
        # Reshape to original shape
        if shape:
            tensor = tensor.reshape(shape)
        
        return tensor
    except Exception as e:
        logger.error(f"Failed to deserialize tensor: {e}")
        return None


@dataclass
class GradientAnalysisResult:
    """Results from gradient-based mesa-optimization detection."""
    gradient_variance: float = 0.0
    anomaly_score: float = 0.0
    hessian_eigenvalues: Optional[torch.Tensor] = None
    gradient_directions: Optional[torch.Tensor] = None
    risk_score: float = 0.0
    confidence: float = 0.0
    
    def __post_init__(self):
        """Validate results after initialization."""
        self._lock = threading.Lock()
        self._validate_results()
    
    def _validate_results(self):
        """Validate gradient analysis results."""
        try:
            # Validate scalar values
            self.gradient_variance = max(0.0, float(self.gradient_variance))
            self.anomaly_score = max(0.0, min(1.0, float(self.anomaly_score)))
            self.risk_score = max(0.0, min(1.0, float(self.risk_score)))
            self.confidence = max(0.0, min(1.0, float(self.confidence)))
            
            # Validate tensor fields
            if self.hessian_eigenvalues is not None and not isinstance(self.hessian_eigenvalues, torch.Tensor):
                logger.warning("Invalid hessian_eigenvalues type, setting to None")
                self.hessian_eigenvalues = None
            
            if self.gradient_directions is not None and not isinstance(self.gradient_directions, torch.Tensor):
                logger.warning("Invalid gradient_directions type, setting to None")
                self.gradient_directions = None
                
        except Exception as e:
            logger.error(f"Gradient result validation failed: {e}")
            self._reset_to_safe_values()
    
    def _reset_to_safe_values(self):
        """Reset to safe default values."""
        self.gradient_variance = 0.0
        self.anomaly_score = 0.0
        self.hessian_eigenvalues = None
        self.gradient_directions = None
        self.risk_score = 0.0
        self.confidence = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        with self._lock:
            try:
                result = {
                    'gradient_variance': float(self.gradient_variance),
                    'anomaly_score': float(self.anomaly_score),
                    'risk_score': float(self.risk_score),
                    'confidence': float(self.confidence),
                    'hessian_eigenvalues': _safe_tensor_to_json(self.hessian_eigenvalues) if self.hessian_eigenvalues is not None else None,
                    'gradient_directions': _safe_tensor_to_json(self.gradient_directions) if self.gradient_directions is not None else None
                }
                return result
            except Exception as e:
                logger.error(f"Failed to serialize gradient results: {e}")
                return {'error': str(e)}


@dataclass
class ActivationAnalysisResult:
    """Results from activation pattern analysis."""
    planning_score: float = 0.0
    optimization_score: float = 0.0
    circuit_patterns: Optional[List[str]] = None
    activation_statistics: Optional[Dict[str, float]] = None
    risk_score: float = 0.0
    confidence: float = 0.0
    
    def __post_init__(self):
        """Validate results after initialization."""
        self._lock = threading.Lock()
        self._validate_results()
    
    def _validate_results(self):
        """Validate activation analysis results."""
        try:
            # Validate scalar values
            self.planning_score = max(0.0, min(1.0, float(self.planning_score)))
            self.optimization_score = max(0.0, min(1.0, float(self.optimization_score)))
            self.risk_score = max(0.0, min(1.0, float(self.risk_score)))
            self.confidence = max(0.0, min(1.0, float(self.confidence)))
            
            # Validate circuit patterns
            if self.circuit_patterns is not None:
                if not isinstance(self.circuit_patterns, list):
                    logger.warning("Invalid circuit_patterns type, setting to None")
                    self.circuit_patterns = None
                else:
                    # Ensure all patterns are strings
                    self.circuit_patterns = [str(p) for p in self.circuit_patterns if p is not None]
            
            # Validate activation statistics
            if self.activation_statistics is not None:
                if not isinstance(self.activation_statistics, dict):
                    logger.warning("Invalid activation_statistics type, setting to None")
                    self.activation_statistics = None
                else:
                    # Validate all values are numeric
                    validated_stats = {}
                    for key, value in self.activation_statistics.items():
                        try:
                            validated_stats[str(key)] = float(value)
                        except (ValueError, TypeError):
                            logger.debug(f"Skipping invalid statistic {key}: {value}")
                    self.activation_statistics = validated_stats
                    
        except Exception as e:
            logger.error(f"Activation result validation failed: {e}")
            self._reset_to_safe_values()
    
    def _reset_to_safe_values(self):
        """Reset to safe default values."""
        self.planning_score = 0.0
        self.optimization_score = 0.0
        self.circuit_patterns = None
        self.activation_statistics = None
        self.risk_score = 0.0
        self.confidence = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        with self._lock:
            try:
                return {
                    'planning_score': float(self.planning_score),
                    'optimization_score': float(self.optimization_score),
                    'circuit_patterns': self.circuit_patterns.copy() if self.circuit_patterns else None,
                    'activation_statistics': self.activation_statistics.copy() if self.activation_statistics else None,
                    'risk_score': float(self.risk_score),
                    'confidence': float(self.confidence)
                }
            except Exception as e:
                logger.error(f"Failed to serialize activation results: {e}")
                return {'error': str(e)}


@dataclass
class BehavioralAnalysisResult:
    """Results from behavioral consistency analysis."""
    consistency_score: float = 0.0
    context_sensitivity: float = 0.0
    deception_indicators: Optional[List[str]] = None
    response_variations: Optional[Dict[str, Any]] = None
    risk_score: float = 0.0
    confidence: float = 0.0
    
    def __post_init__(self):
        """Validate results after initialization."""
        self._lock = threading.Lock()
        self._validate_results()
    
    def _validate_results(self):
        """Validate behavioral analysis results."""
        try:
            # Validate scalar values
            self.consistency_score = max(0.0, min(1.0, float(self.consistency_score)))
            self.context_sensitivity = max(0.0, min(1.0, float(self.context_sensitivity)))
            self.risk_score = max(0.0, min(1.0, float(self.risk_score)))
            self.confidence = max(0.0, min(1.0, float(self.confidence)))
            
            # Validate deception indicators
            if self.deception_indicators is not None:
                if not isinstance(self.deception_indicators, list):
                    logger.warning("Invalid deception_indicators type, setting to None")
                    self.deception_indicators = None
                else:
                    # Ensure all indicators are strings
                    self.deception_indicators = [str(ind) for ind in self.deception_indicators if ind is not None]
            
            # Validate response variations
            if self.response_variations is not None and not isinstance(self.response_variations, dict):
                logger.warning("Invalid response_variations type, setting to None")
                self.response_variations = None
                
        except Exception as e:
            logger.error(f"Behavioral result validation failed: {e}")
            self._reset_to_safe_values()
    
    def _reset_to_safe_values(self):
        """Reset to safe default values."""
        self.consistency_score = 0.0
        self.context_sensitivity = 0.0
        self.deception_indicators = None
        self.response_variations = None
        self.risk_score = 0.0
        self.confidence = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        with self._lock:
            try:
                # Handle response variations carefully
                variations_dict = None
                if self.response_variations:
                    variations_dict = {}
                    for key, value in self.response_variations.items():
                        try:
                            # Convert tensors and complex objects to serializable format
                            if isinstance(value, torch.Tensor):
                                variations_dict[key] = _safe_tensor_to_json(value)
                            elif isinstance(value, (dict, list, str, int, float, bool)) or value is None:
                                variations_dict[key] = value
                            else:
                                variations_dict[key] = str(value)
                        except Exception as e:
                            logger.debug(f"Failed to serialize response variation {key}: {e}")
                            variations_dict[key] = f"<serialization_error: {type(value).__name__}>"
                
                return {
                    'consistency_score': float(self.consistency_score),
                    'context_sensitivity': float(self.context_sensitivity),
                    'deception_indicators': self.deception_indicators.copy() if self.deception_indicators else None,
                    'response_variations': variations_dict,
                    'risk_score': float(self.risk_score),
                    'confidence': float(self.confidence)
                }
            except Exception as e:
                logger.error(f"Failed to serialize behavioral results: {e}")
                return {'error': str(e)}


@dataclass
class DynamicsAnalysisResult:
    """Results from optimization dynamics analysis."""
    phase_transitions: int = 0
    learning_trajectory: Optional[List[float]] = None
    stability_score: float = 0.0
    convergence_patterns: Optional[List[str]] = None
    risk_score: float = 0.0
    confidence: float = 0.0
    
    def __post_init__(self):
        """Validate results after initialization."""
        self._lock = threading.Lock()
        self._validate_results()
    
    def _validate_results(self):
        """Validate dynamics analysis results."""
        try:
            # Validate scalar values
            self.phase_transitions = max(0, int(self.phase_transitions))
            self.stability_score = max(0.0, min(1.0, float(self.stability_score)))
            self.risk_score = max(0.0, min(1.0, float(self.risk_score)))
            self.confidence = max(0.0, min(1.0, float(self.confidence)))
            
            # Validate learning trajectory
            if self.learning_trajectory is not None:
                if not isinstance(self.learning_trajectory, list):
                    logger.warning("Invalid learning_trajectory type, setting to None")
                    self.learning_trajectory = None
                else:
                    # Ensure all values are numeric
                    validated_trajectory = []
                    for value in self.learning_trajectory:
                        try:
                            validated_trajectory.append(float(value))
                        except (ValueError, TypeError):
                            logger.debug(f"Skipping invalid trajectory value: {value}")
                    self.learning_trajectory = validated_trajectory if validated_trajectory else None
            
            # Validate convergence patterns
            if self.convergence_patterns is not None:
                if not isinstance(self.convergence_patterns, list):
                    logger.warning("Invalid convergence_patterns type, setting to None")
                    self.convergence_patterns = None
                else:
                    # Ensure all patterns are strings
                    self.convergence_patterns = [str(p) for p in self.convergence_patterns if p is not None]
                    
        except Exception as e:
            logger.error(f"Dynamics result validation failed: {e}")
            self._reset_to_safe_values()
    
    def _reset_to_safe_values(self):
        """Reset to safe default values."""
        self.phase_transitions = 0
        self.learning_trajectory = None
        self.stability_score = 0.0
        self.convergence_patterns = None
        self.risk_score = 0.0
        self.confidence = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        with self._lock:
            try:
                return {
                    'phase_transitions': int(self.phase_transitions),
                    'learning_trajectory': self.learning_trajectory.copy() if self.learning_trajectory else None,
                    'stability_score': float(self.stability_score),
                    'convergence_patterns': self.convergence_patterns.copy() if self.convergence_patterns else None,
                    'risk_score': float(self.risk_score),
                    'confidence': float(self.confidence)
                }
            except Exception as e:
                logger.error(f"Failed to serialize dynamics results: {e}")
                return {'error': str(e)}


@dataclass
class CausalAnalysisResult:
    """Results from causal intervention analysis."""
    intervention_effects: Optional[Dict[str, float]] = None
    causal_score: float = 0.0
    affected_layers: Optional[List[str]] = None
    intervention_sensitivity: float = 0.0
    risk_score: float = 0.0
    confidence: float = 0.0
    
    def __post_init__(self):
        """Validate results after initialization."""
        self._lock = threading.Lock()
        self._validate_results()
    
    def _validate_results(self):
        """Validate causal analysis results."""
        try:
            # Validate scalar values
            self.causal_score = max(0.0, min(1.0, float(self.causal_score)))
            self.intervention_sensitivity = max(0.0, min(1.0, float(self.intervention_sensitivity)))
            self.risk_score = max(0.0, min(1.0, float(self.risk_score)))
            self.confidence = max(0.0, min(1.0, float(self.confidence)))
            
            # Validate intervention effects
            if self.intervention_effects is not None:
                if not isinstance(self.intervention_effects, dict):
                    logger.warning("Invalid intervention_effects type, setting to None")
                    self.intervention_effects = None
                else:
                    # Validate all values are numeric
                    validated_effects = {}
                    for key, value in self.intervention_effects.items():
                        try:
                            validated_effects[str(key)] = float(value)
                        except (ValueError, TypeError):
                            logger.debug(f"Skipping invalid intervention effect {key}: {value}")
                    self.intervention_effects = validated_effects if validated_effects else None
            
            # Validate affected layers
            if self.affected_layers is not None:
                if not isinstance(self.affected_layers, list):
                    logger.warning("Invalid affected_layers type, setting to None")
                    self.affected_layers = None
                else:
                    # Ensure all layers are strings
                    self.affected_layers = [str(layer) for layer in self.affected_layers if layer is not None]
                    
        except Exception as e:
            logger.error(f"Causal result validation failed: {e}")
            self._reset_to_safe_values()
    
    def _reset_to_safe_values(self):
        """Reset to safe default values."""
        self.intervention_effects = None
        self.causal_score = 0.0
        self.affected_layers = None
        self.intervention_sensitivity = 0.0
        self.risk_score = 0.0
        self.confidence = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        with self._lock:
            try:
                return {
                    'intervention_effects': self.intervention_effects.copy() if self.intervention_effects else None,
                    'causal_score': float(self.causal_score),
                    'affected_layers': self.affected_layers.copy() if self.affected_layers else None,
                    'intervention_sensitivity': float(self.intervention_sensitivity),
                    'risk_score': float(self.risk_score),
                    'confidence': float(self.confidence)
                }
            except Exception as e:
                logger.error(f"Failed to serialize causal results: {e}")
                return {'error': str(e)}


@dataclass
class RiskAssessment:
    """Overall risk assessment from all detection methods."""
    risk_level: str = "MINIMAL"
    risk_score: float = 0.0
    confidence: float = 0.0
    method_scores: Dict[str, float] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate risk assessment after initialization."""
        self._lock = threading.Lock()
        self._validate_assessment()
    
    def _validate_assessment(self):
        """Validate risk assessment."""
        try:
            # Validate risk level
            valid_levels = ["MINIMAL", "LOW", "MEDIUM", "HIGH", "CRITICAL", "UNKNOWN"]
            if self.risk_level not in valid_levels:
                logger.warning(f"Invalid risk level: {self.risk_level}, using MINIMAL")
                self.risk_level = "MINIMAL"
            
            # Validate scalar values
            self.risk_score = max(0.0, min(1.0, float(self.risk_score)))
            self.confidence = max(0.0, min(1.0, float(self.confidence)))
            
            # Validate method scores
            if not isinstance(self.method_scores, dict):
                logger.warning("Invalid method_scores type, resetting to empty dict")
                self.method_scores = {}
            else:
                # Validate all scores are numeric
                validated_scores = {}
                for method, score in self.method_scores.items():
                    try:
                        validated_scores[str(method)] = max(0.0, min(1.0, float(score)))
                    except (ValueError, TypeError):
                        logger.debug(f"Skipping invalid method score {method}: {score}")
                self.method_scores = validated_scores
            
            # Validate recommendations
            if not isinstance(self.recommendations, list):
                logger.warning("Invalid recommendations type, resetting to empty list")
                self.recommendations = []
            else:
                # Ensure all recommendations are strings
                self.recommendations = [str(rec) for rec in self.recommendations if rec is not None]
                
        except Exception as e:
            logger.error(f"Risk assessment validation failed: {e}")
            self._reset_to_safe_values()
    
    def _reset_to_safe_values(self):
        """Reset to safe default values."""
        self.risk_level = "MINIMAL"
        self.risk_score = 0.0
        self.confidence = 0.0
        self.method_scores = {}
        self.recommendations = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        with self._lock:
            try:
                return {
                    'risk_level': str(self.risk_level),
                    'risk_score': float(self.risk_score),
                    'confidence': float(self.confidence),
                    'method_scores': self.method_scores.copy(),
                    'recommendations': self.recommendations.copy()
                }
            except Exception as e:
                logger.error(f"Failed to serialize risk assessment: {e}")
                return {'error': str(e)}


@dataclass
class DetectionResults:
    """Main results class containing all detection analysis results."""
    risk_assessment: RiskAssessment = field(default_factory=RiskAssessment)
    method_results: Dict[str, Any] = field(default_factory=dict)
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate results after initialization."""
        self._lock = threading.Lock()
        self._validate_results()
    
    def _validate_results(self):
        """Validate detection results."""
        try:
            # Validate risk assessment
            if not isinstance(self.risk_assessment, RiskAssessment):
                logger.warning("Invalid risk_assessment type, creating new one")
                self.risk_assessment = RiskAssessment()
            
            # Validate method results
            if not isinstance(self.method_results, dict):
                logger.warning("Invalid method_results type, resetting to empty dict")
                self.method_results = {}
            
            # Validate metadata
            if self.metadata is not None and not isinstance(self.metadata, dict):
                logger.warning("Invalid metadata type, setting to None")
                self.metadata = None
                
        except Exception as e:
            logger.error(f"Detection results validation failed: {e}")
            self._reset_to_safe_values()
    
    def _reset_to_safe_values(self):
        """Reset to safe default values."""
        self.risk_assessment = RiskAssessment()
        self.method_results = {}
        self.metadata = None
    
    @property
    def risk_score(self) -> float:
        """Get overall risk score."""
        try:
            return float(self.risk_assessment.risk_score)
        except Exception:
            return 0.0
    
    @property
    def risk_level(self) -> str:
        """Get overall risk level."""
        try:
            return str(self.risk_assessment.risk_level)
        except Exception:
            return "MINIMAL"
    
    def get_method_result(self, method: str) -> Optional[Any]:
        """Get result for specific method."""
        try:
            return self.method_results.get(str(method))
        except Exception:
            return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        with self._lock:
            try:
                # Serialize method results carefully
                serialized_methods = {}
                for method, result in self.method_results.items():
                    try:
                        if hasattr(result, 'to_dict'):
                            serialized_methods[str(method)] = result.to_dict()
                        elif isinstance(result, dict):
                            serialized_methods[str(method)] = result
                        else:
                            serialized_methods[str(method)] = str(result)
                    except Exception as e:
                        logger.debug(f"Failed to serialize method result {method}: {e}")
                        serialized_methods[str(method)] = {'error': str(e)}
                
                return {
                    'risk_assessment': self.risk_assessment.to_dict(),
                    'method_results': serialized_methods,
                    'metadata': self.metadata.copy() if self.metadata else None
                }
            except Exception as e:
                logger.error(f"Failed to serialize detection results: {e}")
                return {'error': str(e)}
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        try:
            return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to convert results to JSON: {e}")
            return json.dumps({'error': str(e)})
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DetectionResults':
        """Create DetectionResults from dictionary."""
        try:
            if not isinstance(data, dict):
                logger.warning("Invalid data for DetectionResults.from_dict")
                return cls()
            
            # Reconstruct risk assessment
            risk_data = data.get('risk_assessment', {})
            risk_assessment = RiskAssessment(
                risk_level=risk_data.get('risk_level', 'MINIMAL'),
                risk_score=risk_data.get('risk_score', 0.0),
                confidence=risk_data.get('confidence', 0.0),
                method_scores=risk_data.get('method_scores', {}),
                recommendations=risk_data.get('recommendations', [])
            )
            
            # Method results and metadata as-is (they're validated in __post_init__)
            method_results = data.get('method_results', {})
            metadata = data.get('metadata')
            
            return cls(
                risk_assessment=risk_assessment,
                method_results=method_results,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Failed to create DetectionResults from dict: {e}")
            return cls()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the detection results."""
        try:
            with self._lock:
                summary = {
                    'overall_risk': {
                        'level': self.risk_level,
                        'score': self.risk_score,
                        'confidence': self.risk_assessment.confidence
                    },
                    'methods_analyzed': list(self.method_results.keys()),
                    'method_scores': self.risk_assessment.method_scores.copy(),
                    'top_recommendations': self.risk_assessment.recommendations[:3],
                    'metadata_available': self.metadata is not None
                }
                
                if self.metadata:
                    summary['analysis_timestamp'] = self.metadata.get('analysis_timestamp', 'unknown')
                    summary['methods_attempted'] = self.metadata.get('total_methods_attempted', 0)
                
                return summary
        except Exception as e:
            logger.error(f"Failed to generate summary: {e}")
            return {'error': str(e)}


@dataclass
class MonitoringResults:
    """Results from continuous monitoring during training."""
    detection_history: List[DetectionResults] = field(default_factory=list)
    training_steps: List[int] = field(default_factory=list)
    risk_trend: List[float] = field(default_factory=list)
    alerts: List[Dict[str, Any]] = field(default_factory=list)
    
    def add_result(self, step: int, result: DetectionResults) -> None:
        """Add a new detection result to the monitoring history."""
        self.detection_history.append(result)
        self.training_steps.append(step)
        self.risk_trend.append(result.risk_score)
        
        # Check for alerts
        if result.risk_score >= 0.8:  # High risk threshold
            self.alerts.append({
                'step': step,
                'risk_score': result.risk_score,
                'risk_level': result.risk_level,
                'timestamp': self._get_timestamp()
            })
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        import datetime
        return datetime.datetime.now().isoformat()
    
    def get_latest_result(self) -> Optional[DetectionResults]:
        """Get the most recent detection result."""
        return self.detection_history[-1] if self.detection_history else None
    
    def get_risk_trend_analysis(self) -> Dict[str, Any]:
        """Analyze the trend in risk scores over time."""
        if len(self.risk_trend) < 2:
            return {'trend': 'insufficient_data'}
        
        # Simple trend analysis
        recent_scores = self.risk_trend[-5:]  # Last 5 measurements
        trend_slope = (recent_scores[-1] - recent_scores[0]) / len(recent_scores)
        
        if trend_slope > 0.05:
            trend = 'increasing'
        elif trend_slope < -0.05:
            trend = 'decreasing'
        else:
            trend = 'stable'
        
        return {
            'trend': trend,
            'slope': trend_slope,
            'current_risk': self.risk_trend[-1],
            'max_risk': max(self.risk_trend),
            'min_risk': min(self.risk_trend),
            'num_alerts': len(self.alerts)
        } 