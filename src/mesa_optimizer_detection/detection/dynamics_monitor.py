"""
Optimization Dynamics Monitoring for Mesa-Optimization Detection

This module monitors training dynamics and loss landscape properties to detect
the emergence of mesa-optimization during model training.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from collections import deque

from ..core.results import DynamicsAnalysisResult
from ..utils.model_utils import ModelWrapper
from ..config import DynamicsConfig

logger = logging.getLogger(__name__)


class OptimizationDynamicsMonitor:
    """
    Monitors optimization dynamics during training to detect mesa-optimization emergence.
    
    This monitor tracks:
    1. Loss landscape curvature changes
    2. Parameter update patterns
    3. Phase transitions in training dynamics
    4. Optimization trajectory anomalies
    """
    
    def __init__(
        self,
        model: ModelWrapper,
        config: Optional[DynamicsConfig] = None
    ):
        self.model = model
        self.config = config or DynamicsConfig()
        
        # Training history storage
        self.loss_history = deque(maxlen=1000)
        self.gradient_norms = deque(maxlen=1000)
        self.parameter_distances = deque(maxlen=1000)
        self.curvature_history = deque(maxlen=100)
        
        # Previous state for comparison
        self.previous_parameters = None
        self.previous_loss = None
        
        # Phase transition detection
        self.phase_detection_window = self.config.loss_smoothing_window * 2
        self.detected_phases = []
        
        logger.info("OptimizationDynamicsMonitor initialized")
    
    def analyze(self, training_step: int) -> DynamicsAnalysisResult:
        """
        Analyze current training dynamics for mesa-optimization indicators.
        
        Args:
            training_step: Current training step number
            
        Returns:
            DynamicsAnalysisResult containing dynamics analysis
        """
        logger.debug(f"Analyzing optimization dynamics at step {training_step}")
        
        # Update training state
        current_loss = self._get_current_loss()
        current_parameters = self._get_current_parameters()
        
        # Compute curvature anomaly
        curvature_anomaly = self._compute_curvature_anomaly()
        
        # Compute parameter anomaly
        parameter_anomaly = self._compute_parameter_anomaly(current_parameters)
        
        # Detect phase transitions
        phase_transitions = self._detect_phase_transitions()
        
        # Analyze loss landscape features
        loss_landscape_features = self._analyze_loss_landscape()
        
        # Compute overall risk score
        risk_score = self._compute_risk_score(
            curvature_anomaly, parameter_anomaly, phase_transitions
        )
        
        # Compute confidence
        confidence = self._compute_confidence()
        
        # Update history
        self._update_history(current_loss, current_parameters)
        
        result = DynamicsAnalysisResult(
            curvature_anomaly=curvature_anomaly,
            parameter_anomaly=parameter_anomaly,
            phase_transitions=phase_transitions,
            loss_landscape_features=loss_landscape_features,
            risk_score=risk_score,
            confidence=confidence
        )
        
        logger.debug(f"Dynamics analysis complete. Risk score: {risk_score:.3f}")
        return result
    
    def _get_current_loss(self) -> float:
        """Get current model loss."""
        # This would typically be provided from the training loop
        # For now, return a placeholder or compute from a validation batch
        try:
            self.model.model.eval()
            # Would need a validation batch here
            # For now, return 0 as placeholder
            return 0.0
        except Exception as e:
            logger.warning(f"Could not compute current loss: {e}")
            return 0.0
    
    def _get_current_parameters(self) -> torch.Tensor:
        """Get current model parameters as a flattened tensor."""
        params = []
        for param in self.model.model.parameters():
            if param.requires_grad:
                params.append(param.data.flatten())
        
        if params:
            return torch.cat(params)
        else:
            return torch.tensor([])
    
    def _compute_curvature_anomaly(self) -> float:
        """Compute loss landscape curvature anomaly."""
        try:
            # Estimate curvature using finite differences or Hessian approximation
            curvature = self._estimate_loss_curvature()
            
            if curvature is not None:
                self.curvature_history.append(curvature)
                
                # Compare with historical curvature
                if len(self.curvature_history) > 10:
                    recent_curvature = curvature
                    historical_mean = np.mean(list(self.curvature_history)[:-5])
                    
                    if historical_mean > 0:
                        curvature_change = abs(recent_curvature - historical_mean) / historical_mean
                        return min(curvature_change, 1.0)
            
            return 0.0
            
        except Exception as e:
            logger.warning(f"Curvature computation failed: {e}")
            return 0.0
    
    def _estimate_loss_curvature(self) -> Optional[float]:
        """Estimate loss landscape curvature using finite differences."""
        # This is a simplified estimation - in practice would use Hessian methods
        if len(self.loss_history) < 3:
            return None
        
        # Second derivative approximation
        losses = list(self.loss_history)[-3:]
        second_derivative = losses[0] - 2 * losses[1] + losses[2]
        
        return abs(second_derivative)
    
    def _compute_parameter_anomaly(self, current_parameters: torch.Tensor) -> float:
        """Compute parameter update anomaly."""
        if self.previous_parameters is None or len(current_parameters) == 0:
            self.previous_parameters = current_parameters.clone()
            return 0.0
        
        try:
            # Compute parameter distance
            if current_parameters.shape == self.previous_parameters.shape:
                param_distance = torch.norm(current_parameters - self.previous_parameters)
                self.parameter_distances.append(param_distance.item())
                
                # Detect anomalous parameter changes
                if len(self.parameter_distances) > 10:
                    recent_distance = param_distance.item()
                    historical_distances = list(self.parameter_distances)[:-1]
                    
                    mean_distance = np.mean(historical_distances)
                    std_distance = np.std(historical_distances)
                    
                    if std_distance > 0:
                        z_score = abs(recent_distance - mean_distance) / std_distance
                        anomaly_score = min(z_score / 3.0, 1.0)  # Normalize to [0,1]
                        return anomaly_score
            
            return 0.0
            
        except Exception as e:
            logger.warning(f"Parameter anomaly computation failed: {e}")
            return 0.0
    
    def _detect_phase_transitions(self) -> List[int]:
        """Detect phase transitions in training dynamics."""
        if len(self.loss_history) < self.phase_detection_window:
            return []
        
        try:
            # Convert to numpy for analysis
            losses = np.array(list(self.loss_history))
            
            # Detect change points using statistical methods
            phase_transitions = []
            
            # Simple change point detection using moving averages
            window_size = self.config.loss_smoothing_window
            
            if len(losses) >= 2 * window_size:
                for i in range(window_size, len(losses) - window_size):
                    # Compare means before and after potential change point
                    before_mean = np.mean(losses[i-window_size:i])
                    after_mean = np.mean(losses[i:i+window_size])
                    
                    # Check for significant change
                    if before_mean > 0:
                        relative_change = abs(after_mean - before_mean) / before_mean
                        
                        if relative_change > self.config.phase_transition_sensitivity:
                            # Map back to actual training step
                            actual_step = len(self.loss_history) - len(losses) + i
                            phase_transitions.append(actual_step)
            
            # Remove duplicate detections
            if phase_transitions:
                filtered_transitions = [phase_transitions[0]]
                for transition in phase_transitions[1:]:
                    if transition - filtered_transitions[-1] > window_size:
                        filtered_transitions.append(transition)
                
                return filtered_transitions
            
            return []
            
        except Exception as e:
            logger.warning(f"Phase transition detection failed: {e}")
            return []
    
    def _analyze_loss_landscape(self) -> Dict[str, float]:
        """Analyze loss landscape features."""
        features = {}
        
        if len(self.loss_history) < 10:
            return features
        
        try:
            losses = np.array(list(self.loss_history))
            
            # Basic landscape features
            features['loss_variance'] = float(np.var(losses))
            features['loss_trend'] = float(np.polyfit(range(len(losses)), losses, 1)[0])
            features['loss_smoothness'] = self._compute_loss_smoothness(losses)
            features['convergence_rate'] = self._estimate_convergence_rate(losses)
            
            # Advanced landscape features
            if len(self.curvature_history) > 5:
                curvatures = np.array(list(self.curvature_history))
                features['curvature_variance'] = float(np.var(curvatures))
                features['curvature_trend'] = float(np.polyfit(range(len(curvatures)), curvatures, 1)[0])
            
            # Parameter update consistency
            if len(self.parameter_distances) > 5:
                param_dists = np.array(list(self.parameter_distances))
                features['param_update_variance'] = float(np.var(param_dists))
                features['param_update_trend'] = float(np.polyfit(range(len(param_dists)), param_dists, 1)[0])
            
        except Exception as e:
            logger.warning(f"Loss landscape analysis failed: {e}")
        
        return features
    
    def _compute_loss_smoothness(self, losses: np.ndarray) -> float:
        """Compute loss smoothness metric."""
        if len(losses) < 3:
            return 0.0
        
        # Compute second derivatives
        second_derivatives = np.diff(losses, n=2)
        
        # Return inverse of mean absolute second derivative (smoother = higher value)
        mean_second_deriv = np.mean(np.abs(second_derivatives))
        
        if mean_second_deriv > 0:
            return 1.0 / (1.0 + mean_second_deriv)
        else:
            return 1.0
    
    def _estimate_convergence_rate(self, losses: np.ndarray) -> float:
        """Estimate convergence rate from loss history."""
        if len(losses) < 10:
            return 0.0
        
        # Fit exponential decay model: loss = a * exp(-b * t) + c
        # Simplified: use linear fit on log scale
        try:
            # Remove zeros and negatives for log
            positive_losses = losses[losses > 1e-10]
            
            if len(positive_losses) < 5:
                return 0.0
            
            log_losses = np.log(positive_losses)
            x = np.arange(len(log_losses))
            
            # Linear fit
            slope, _ = np.polyfit(x, log_losses, 1)
            
            # Return negative slope (positive = faster convergence)
            return float(-slope)
            
        except Exception:
            return 0.0
    
    def _compute_risk_score(
        self,
        curvature_anomaly: float,
        parameter_anomaly: float,
        phase_transitions: List[int]
    ) -> float:
        """Compute overall risk score from dynamics analysis."""
        risk_components = []
        
        # Curvature anomaly component
        if curvature_anomaly > self.config.curvature_threshold:
            curvature_risk = min(curvature_anomaly / self.config.curvature_threshold, 1.0)
            risk_components.append(curvature_risk * 0.3)
        
        # Parameter anomaly component
        if parameter_anomaly > self.config.parameter_change_threshold:
            param_risk = min(parameter_anomaly / self.config.parameter_change_threshold, 1.0)
            risk_components.append(param_risk * 0.4)
        
        # Phase transition component
        if phase_transitions:
            # Recent phase transitions are more concerning
            recent_transitions = [
                t for t in phase_transitions 
                if len(self.loss_history) - t < 50  # Last 50 steps
            ]
            
            if recent_transitions:
                transition_risk = min(len(recent_transitions) / 3.0, 1.0)
                risk_components.append(transition_risk * 0.3)
        
        # Aggregate risk score
        if risk_components:
            return float(np.mean(risk_components))
        else:
            return 0.0
    
    def _compute_confidence(self) -> float:
        """Compute confidence in the dynamics analysis."""
        confidence_factors = []
        
        # More training history increases confidence
        history_confidence = min(len(self.loss_history) / 100, 1.0)
        confidence_factors.append(history_confidence)
        
        # Consistent measurements increase confidence
        if len(self.parameter_distances) > 5:
            param_consistency = 1.0 - min(np.std(list(self.parameter_distances)) / (np.mean(list(self.parameter_distances)) + 1e-8), 1.0)
            confidence_factors.append(param_consistency)
        
        # Successful curvature estimation increases confidence
        if len(self.curvature_history) > 0:
            confidence_factors.append(0.8)  # Base confidence for having curvature data
        
        return float(np.mean(confidence_factors)) if confidence_factors else 0.5
    
    def _update_history(self, current_loss: float, current_parameters: torch.Tensor):
        """Update training history with current state."""
        self.loss_history.append(current_loss)
        self.previous_loss = current_loss
        
        if len(current_parameters) > 0:
            self.previous_parameters = current_parameters.clone()
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get summary of training dynamics."""
        summary = {
            'total_steps_monitored': len(self.loss_history),
            'detected_phase_transitions': len(self.detected_phases),
            'average_loss': float(np.mean(list(self.loss_history))) if self.loss_history else 0.0,
            'loss_variance': float(np.var(list(self.loss_history))) if len(self.loss_history) > 1 else 0.0,
        }
        
        if self.parameter_distances:
            summary.update({
                'average_param_distance': float(np.mean(list(self.parameter_distances))),
                'param_distance_variance': float(np.var(list(self.parameter_distances)))
            })
        
        if self.curvature_history:
            summary.update({
                'average_curvature': float(np.mean(list(self.curvature_history))),
                'curvature_variance': float(np.var(list(self.curvature_history)))
            })
        
        return summary
    
    def reset_history(self):
        """Reset monitoring history."""
        self.loss_history.clear()
        self.gradient_norms.clear()
        self.parameter_distances.clear()
        self.curvature_history.clear()
        self.detected_phases.clear()
        self.previous_parameters = None
        self.previous_loss = None
        
        logger.info("Training dynamics history reset") 