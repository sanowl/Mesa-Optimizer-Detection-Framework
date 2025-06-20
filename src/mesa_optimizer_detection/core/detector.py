"""
Main Mesa-Optimizer Detection System

This module contains the central MesaOptimizerDetector class that orchestrates
all detection methods and provides a unified interface for mesa-optimization detection.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import logging
import warnings

from ..detection.gradient_analyzer import GradientAnalyzer
from ..detection.activation_analyzer import ActivationPatternAnalyzer
from ..detection.behavioral_analyzer import BehavioralConsistencyAnalyzer
from ..detection.dynamics_monitor import OptimizationDynamicsMonitor
from ..detection.causal_intervention import CausalInterventionAnalyzer
from ..utils.model_utils import ModelWrapper
from ..config import DetectionConfig
from .results import DetectionResults, RiskAssessment

logger = logging.getLogger(__name__)


class MesaOptimizerDetector:
    """
    Main class for detecting mesa-optimization in large language models.
    
    This detector combines multiple detection methods including gradient analysis,
    activation pattern recognition, behavioral consistency checks, optimization
    dynamics monitoring, and causal interventions to identify potential
    mesa-optimization and deceptive alignment.
    
    Args:
        model: The model to analyze (torch.nn.Module or ModelWrapper)
        layer_indices: List of layer indices to analyze for activation patterns
        detection_methods: List of detection methods to use
        config: Configuration object for detection parameters
        device: Device to run computations on
    """
    
    def __init__(
        self,
        model: Union[nn.Module, ModelWrapper],
        layer_indices: Optional[List[int]] = None,
        detection_methods: Optional[List[str]] = None,
        config: Optional[DetectionConfig] = None,
        device: Optional[torch.device] = None
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Wrap model if needed
        if isinstance(model, ModelWrapper):
            self.model = model
        else:
            self.model = ModelWrapper(model, device=self.device)
            
        self.config = config or DetectionConfig()
        self.layer_indices = layer_indices or self._get_default_layer_indices()
        
        # Initialize detection methods
        self.detection_methods = detection_methods or [
            'gradient', 'activation', 'behavioral', 'dynamics', 'causal'
        ]
        
        self._initialize_analyzers()
        
        logger.info(f"Initialized MesaOptimizerDetector with methods: {self.detection_methods}")
    
    def _get_default_layer_indices(self) -> List[int]:
        """Get default layer indices to analyze based on model architecture."""
        num_layers = len(list(self.model.model.modules()))
        # Analyze layers at 25%, 50%, 75% depth
        return [
            int(num_layers * 0.25),
            int(num_layers * 0.5),
            int(num_layers * 0.75)
        ]
    
    def _initialize_analyzers(self):
        """Initialize all detection analyzers."""
        self.analyzers = {}
        
        if 'gradient' in self.detection_methods:
            self.analyzers['gradient'] = GradientAnalyzer(
                model=self.model,
                config=self.config.gradient_config
            )
            
        if 'activation' in self.detection_methods:
            self.analyzers['activation'] = ActivationPatternAnalyzer(
                model=self.model,
                layer_indices=self.layer_indices,
                config=self.config.activation_config
            )
            
        if 'behavioral' in self.detection_methods:
            self.analyzers['behavioral'] = BehavioralConsistencyAnalyzer(
                model=self.model,
                config=self.config.behavioral_config
            )
            
        if 'dynamics' in self.detection_methods:
            self.analyzers['dynamics'] = OptimizationDynamicsMonitor(
                model=self.model,
                config=self.config.dynamics_config
            )
            
        if 'causal' in self.detection_methods:
            self.analyzers['causal'] = CausalInterventionAnalyzer(
                model=self.model,
                layer_indices=self.layer_indices,
                config=self.config.causal_config
            )
    
    def analyze(
        self,
        input_batch: Union[torch.Tensor, Dict[str, torch.Tensor]],
        training_step: Optional[int] = None,
        baseline_gradients: Optional[torch.Tensor] = None,
        evaluation_contexts: Optional[List[Dict[str, Any]]] = None
    ) -> DetectionResults:
        """
        Perform comprehensive mesa-optimization analysis on input batch.
        
        Args:
            input_batch: Input data to analyze
            training_step: Current training step (for dynamics monitoring)
            baseline_gradients: Baseline gradients for comparison
            evaluation_contexts: Different contexts for behavioral analysis
            
        Returns:
            DetectionResults object containing all analysis results
        """
        logger.info("Starting mesa-optimization analysis")
        
        results = {}
        
        try:
            # Gradient-based detection
            if 'gradient' in self.analyzers:
                logger.debug("Running gradient analysis")
                results['gradient'] = self.analyzers['gradient'].analyze(
                    input_batch, baseline_gradients
                )
            
            # Activation pattern analysis
            if 'activation' in self.analyzers:
                logger.debug("Running activation pattern analysis")
                results['activation'] = self.analyzers['activation'].analyze(
                    input_batch
                )
            
            # Behavioral consistency analysis
            if 'behavioral' in self.analyzers:
                logger.debug("Running behavioral consistency analysis")
                results['behavioral'] = self.analyzers['behavioral'].analyze(
                    evaluation_contexts or self._create_default_contexts(input_batch)
                )
            
            # Optimization dynamics monitoring
            if 'dynamics' in self.analyzers and training_step is not None:
                logger.debug("Running optimization dynamics analysis")
                results['dynamics'] = self.analyzers['dynamics'].analyze(
                    training_step
                )
            
            # Causal intervention analysis
            if 'causal' in self.analyzers:
                logger.debug("Running causal intervention analysis")
                results['causal'] = self.analyzers['causal'].analyze(
                    input_batch
                )
                
        except Exception as e:
            logger.error(f"Error during analysis: {e}")
            raise
        
        # Aggregate results and compute risk assessment
        detection_results = self._aggregate_results(results)
        
        logger.info(f"Analysis complete. Risk score: {detection_results.risk_score:.3f}")
        return detection_results
    
    def _create_default_contexts(self, input_batch: torch.Tensor) -> List[Dict[str, Any]]:
        """Create default evaluation contexts for behavioral analysis."""
        return [
            {
                'name': 'standard',
                'prompt': input_batch,
                'settings': {'temperature': 0.7, 'max_length': 100}
            },
            {
                'name': 'low_temp',
                'prompt': input_batch,
                'settings': {'temperature': 0.1, 'max_length': 100}
            },
            {
                'name': 'high_temp',
                'prompt': input_batch,
                'settings': {'temperature': 1.5, 'max_length': 100}
            }
        ]
    
    def _aggregate_results(self, results: Dict[str, Any]) -> DetectionResults:
        """Aggregate results from all detection methods."""
        # Compute individual risk scores
        risk_scores = {}
        confidences = {}
        
        for method, result in results.items():
            if hasattr(result, 'risk_score'):
                risk_scores[method] = result.risk_score
                confidences[method] = getattr(result, 'confidence', 0.5)
            elif isinstance(result, dict) and 'risk_score' in result:
                risk_scores[method] = result['risk_score']
                confidences[method] = result.get('confidence', 0.5)
        
        # Weighted average of risk scores
        if risk_scores:
            weights = self.config.method_weights
            total_weight = sum(weights.get(method, 1.0) for method in risk_scores.keys())
            
            aggregated_risk_score = sum(
                score * weights.get(method, 1.0) 
                for method, score in risk_scores.items()
            ) / total_weight
            
            aggregated_confidence = sum(
                conf * weights.get(method, 1.0)
                for method, conf in confidences.items()
            ) / total_weight
        else:
            aggregated_risk_score = 0.0
            aggregated_confidence = 0.0
        
        # Create risk assessment
        risk_assessment = RiskAssessment(
            risk_level=self._compute_risk_level(aggregated_risk_score),
            risk_score=aggregated_risk_score,
            confidence=aggregated_confidence,
            method_scores=risk_scores,
            recommendations=self._generate_recommendations(aggregated_risk_score, results)
        )
        
        return DetectionResults(
            risk_assessment=risk_assessment,
            method_results=results,
            metadata={
                'model_name': getattr(self.model.model, '_name', 'unknown'),
                'detection_methods': self.detection_methods,
                'layer_indices': self.layer_indices,
                'timestamp': torch.tensor(0).float().item()  # Placeholder
            }
        )
    
    def _compute_risk_level(self, risk_score: float) -> str:
        """Compute categorical risk level from numerical score."""
        if risk_score >= self.config.risk_thresholds.high:
            return "HIGH"
        elif risk_score >= self.config.risk_thresholds.medium:
            return "MEDIUM"
        elif risk_score >= self.config.risk_thresholds.low:
            return "LOW"
        else:
            return "MINIMAL"
    
    def _generate_recommendations(
        self, 
        risk_score: float, 
        results: Dict[str, Any]
    ) -> List[str]:
        """Generate actionable recommendations based on detection results."""
        recommendations = []
        
        if risk_score >= self.config.risk_thresholds.high:
            recommendations.append("CRITICAL: Potential mesa-optimization detected. Halt deployment.")
            recommendations.append("Conduct thorough safety evaluation before proceeding.")
        
        if 'gradient' in results and hasattr(results['gradient'], 'anomaly_score'):
            if results['gradient'].anomaly_score > 0.8:
                recommendations.append("High gradient anomalies detected. Investigate optimization dynamics.")
        
        if 'activation' in results and hasattr(results['activation'], 'planning_score'):
            if results['activation'].planning_score > 0.7:
                recommendations.append("Planning-like circuits detected. Analyze goal-directedness.")
        
        if 'behavioral' in results and hasattr(results['behavioral'], 'consistency_score'):
            if results['behavioral'].consistency_score < 0.3:
                recommendations.append("Low behavioral consistency. Test for deceptive alignment.")
        
        if not recommendations:
            recommendations.append("Monitor for emergence of mesa-optimization during continued training.")
        
        return recommendations
    
    def continuous_monitor(
        self,
        data_loader: torch.utils.data.DataLoader,
        training_steps: int,
        monitor_frequency: int = 100
    ) -> List[DetectionResults]:
        """
        Continuously monitor a model during training for mesa-optimization emergence.
        
        Args:
            data_loader: DataLoader providing training batches
            training_steps: Total number of training steps
            monitor_frequency: How often to run detection (every N steps)
            
        Returns:
            List of DetectionResults from monitoring
        """
        monitoring_results = []
        
        logger.info(f"Starting continuous monitoring for {training_steps} steps")
        
        for step in range(0, training_steps, monitor_frequency):
            try:
                # Get next batch
                batch = next(iter(data_loader))
                
                # Run detection
                results = self.analyze(
                    input_batch=batch,
                    training_step=step
                )
                
                monitoring_results.append(results)
                
                # Log results
                logger.info(
                    f"Step {step}: Risk score {results.risk_score:.3f}, "
                    f"Level: {results.risk_assessment.risk_level}"
                )
                
                # Check for high-risk situations
                if results.risk_score >= self.config.risk_thresholds.high:
                    logger.warning(
                        f"HIGH RISK detected at step {step}. "
                        f"Risk score: {results.risk_score:.3f}"
                    )
                
            except Exception as e:
                logger.error(f"Error during monitoring at step {step}: {e}")
                continue
        
        return monitoring_results
    
    def get_detection_summary(self) -> Dict[str, Any]:
        """Get summary of detector configuration and capabilities."""
        return {
            'detector_version': '0.1.0',
            'detection_methods': self.detection_methods,
            'layer_indices': self.layer_indices,
            'model_parameters': sum(p.numel() for p in self.model.model.parameters()),
            'config': self.config.to_dict(),
            'device': str(self.device)
        } 