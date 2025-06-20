"""
Detection Results Data Structures

This module contains data classes for organizing and storing the results
of mesa-optimization detection analysis.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import torch
import json


@dataclass
class RiskAssessment:
    """
    Overall risk assessment from mesa-optimization detection.
    
    Attributes:
        risk_level: Categorical risk level (MINIMAL, LOW, MEDIUM, HIGH)
        risk_score: Numerical risk score (0.0 to 1.0)
        confidence: Confidence in the assessment (0.0 to 1.0)
        method_scores: Individual scores from each detection method
        recommendations: List of actionable recommendations
    """
    risk_level: str
    risk_score: float
    confidence: float
    method_scores: Dict[str, float]
    recommendations: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate risk assessment data."""
        assert 0.0 <= self.risk_score <= 1.0, "Risk score must be between 0 and 1"
        assert 0.0 <= self.confidence <= 1.0, "Confidence must be between 0 and 1"
        assert self.risk_level in ["MINIMAL", "LOW", "MEDIUM", "HIGH"], \
            "Risk level must be one of: MINIMAL, LOW, MEDIUM, HIGH"


@dataclass
class GradientAnalysisResult:
    """Results from gradient-based detection methods."""
    gradient_variance: float
    anomaly_score: float
    hessian_eigenvalues: Optional[torch.Tensor] = None
    gradient_directions: Optional[torch.Tensor] = None
    risk_score: float = 0.0
    confidence: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'gradient_variance': self.gradient_variance,
            'anomaly_score': self.anomaly_score,
            'risk_score': self.risk_score,
            'confidence': self.confidence,
            'has_hessian_data': self.hessian_eigenvalues is not None,
            'has_gradient_directions': self.gradient_directions is not None
        }


@dataclass
class ActivationAnalysisResult:
    """Results from activation pattern analysis."""
    planning_score: float
    goal_directedness: float
    optimization_score: float
    circuit_activations: Dict[str, torch.Tensor] = field(default_factory=dict)
    suspicious_patterns: List[str] = field(default_factory=list)
    risk_score: float = 0.0
    confidence: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'planning_score': self.planning_score,
            'goal_directedness': self.goal_directedness,
            'optimization_score': self.optimization_score,
            'suspicious_patterns': self.suspicious_patterns,
            'risk_score': self.risk_score,
            'confidence': self.confidence,
            'num_circuit_activations': len(self.circuit_activations)
        }


@dataclass
class BehavioralAnalysisResult:
    """Results from behavioral consistency analysis."""
    consistency_score: float
    context_sensitivity: float
    deception_indicators: List[str] = field(default_factory=list)
    response_variations: Dict[str, Any] = field(default_factory=dict)
    risk_score: float = 0.0
    confidence: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'consistency_score': self.consistency_score,
            'context_sensitivity': self.context_sensitivity,
            'deception_indicators': self.deception_indicators,
            'risk_score': self.risk_score,
            'confidence': self.confidence,
            'num_contexts_tested': len(self.response_variations)
        }


@dataclass
class DynamicsAnalysisResult:
    """Results from optimization dynamics monitoring."""
    curvature_anomaly: float
    parameter_anomaly: float
    phase_transitions: List[int] = field(default_factory=list)
    loss_landscape_features: Dict[str, float] = field(default_factory=dict)
    risk_score: float = 0.0
    confidence: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'curvature_anomaly': self.curvature_anomaly,
            'parameter_anomaly': self.parameter_anomaly,
            'phase_transitions': self.phase_transitions,
            'loss_landscape_features': self.loss_landscape_features,
            'risk_score': self.risk_score,
            'confidence': self.confidence
        }


@dataclass
class CausalAnalysisResult:
    """Results from causal intervention analysis."""
    intervention_effects: Dict[str, Dict[str, float]] = field(default_factory=dict)
    causal_circuits: List[str] = field(default_factory=list)
    ablation_sensitivity: float = 0.0
    risk_score: float = 0.0
    confidence: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'intervention_effects': self.intervention_effects,
            'causal_circuits': self.causal_circuits,
            'ablation_sensitivity': self.ablation_sensitivity,
            'risk_score': self.risk_score,
            'confidence': self.confidence
        }


@dataclass
class DetectionResults:
    """
    Comprehensive results from mesa-optimization detection analysis.
    
    This class aggregates results from all detection methods and provides
    a unified interface for accessing detection outcomes.
    
    Attributes:
        risk_assessment: Overall risk assessment
        method_results: Results from individual detection methods
        metadata: Additional metadata about the analysis
    """
    risk_assessment: RiskAssessment
    method_results: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def risk_score(self) -> float:
        """Shortcut to access overall risk score."""
        return self.risk_assessment.risk_score
    
    @property
    def risk_level(self) -> str:
        """Shortcut to access overall risk level."""
        return self.risk_assessment.risk_level
    
    @property
    def confidence(self) -> float:
        """Shortcut to access overall confidence."""
        return self.risk_assessment.confidence
    
    def get_method_result(self, method_name: str) -> Optional[Any]:
        """Get results from a specific detection method."""
        return self.method_results.get(method_name)
    
    def has_method_result(self, method_name: str) -> bool:
        """Check if results exist for a specific method."""
        return method_name in self.method_results
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all detection results."""
        summary = {
            'overall_risk_score': self.risk_score,
            'overall_risk_level': self.risk_level,
            'overall_confidence': self.confidence,
            'recommendations': self.risk_assessment.recommendations,
            'method_scores': self.risk_assessment.method_scores,
            'methods_analyzed': list(self.method_results.keys()),
            'metadata': self.metadata
        }
        
        # Add method-specific summaries
        for method, result in self.method_results.items():
            if hasattr(result, 'to_dict'):
                summary[f'{method}_summary'] = result.to_dict()
            elif isinstance(result, dict):
                summary[f'{method}_summary'] = result
        
        return summary
    
    def to_json(self, indent: int = 2) -> str:
        """
        Convert results to JSON string for serialization.
        
        Note: Tensor data is converted to lists for JSON compatibility.
        """
        summary = self.get_summary()
        
        # Convert any remaining tensors to lists
        def convert_tensors(obj):
            if isinstance(obj, torch.Tensor):
                return obj.detach().cpu().tolist()
            elif isinstance(obj, dict):
                return {k: convert_tensors(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_tensors(item) for item in obj]
            else:
                return obj
        
        json_compatible = convert_tensors(summary)
        return json.dumps(json_compatible, indent=indent)
    
    def save(self, filepath: str) -> None:
        """Save results to file."""
        with open(filepath, 'w') as f:
            f.write(self.to_json())
    
    @classmethod
    def load(cls, filepath: str) -> 'DetectionResults':
        """Load results from file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Reconstruct the DetectionResults object
        # This is a simplified version - full reconstruction would need
        # to handle the specific result types
        risk_assessment = RiskAssessment(
            risk_level=data['overall_risk_level'],
            risk_score=data['overall_risk_score'],
            confidence=data['overall_confidence'],
            method_scores=data['method_scores'],
            recommendations=data['recommendations']
        )
        
        return cls(
            risk_assessment=risk_assessment,
            method_results={},  # Would need more complex reconstruction
            metadata=data['metadata']
        )
    
    def generate_report(self) -> str:
        """Generate a human-readable report of the detection results."""
        report = []
        report.append("="*60)
        report.append("MESA-OPTIMIZER DETECTION REPORT")
        report.append("="*60)
        report.append("")
        
        # Overall assessment
        report.append(f"OVERALL RISK LEVEL: {self.risk_level}")
        report.append(f"Risk Score: {self.risk_score:.3f} / 1.000")
        report.append(f"Confidence: {self.confidence:.3f} / 1.000")
        report.append("")
        
        # Method-specific results
        report.append("DETECTION METHOD SCORES:")
        report.append("-" * 30)
        for method, score in self.risk_assessment.method_scores.items():
            report.append(f"{method.upper():15}: {score:.3f}")
        report.append("")
        
        # Recommendations
        if self.risk_assessment.recommendations:
            report.append("RECOMMENDATIONS:")
            report.append("-" * 20)
            for i, rec in enumerate(self.risk_assessment.recommendations, 1):
                report.append(f"{i}. {rec}")
            report.append("")
        
        # Metadata
        if self.metadata:
            report.append("ANALYSIS METADATA:")
            report.append("-" * 20)
            for key, value in self.metadata.items():
                report.append(f"{key}: {value}")
        
        report.append("="*60)
        
        return "\n".join(report)


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