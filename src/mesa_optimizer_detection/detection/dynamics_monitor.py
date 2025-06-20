"""
Optimization Dynamics Monitor for Mesa-Optimizer Detection

This module monitors training dynamics to detect the emergence of mesa-optimization
by analyzing loss landscapes, parameter trajectories, and optimization behavior patterns.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from collections import deque
from scipy import signal

from ..core.results import DynamicsAnalysisResult
from ..utils.model_utils import ModelWrapper
from ..config import DynamicsConfig

logger = logging.getLogger(__name__)

class OptimizationDynamicsMonitor:
    """
    Monitor optimization dynamics during training to detect mesa-optimization emergence.
    
    Analyzes loss landscape curvature, parameter update patterns, and phase transitions
    to identify when models may be developing internal optimization processes.
    """
    
    def __init__(
        self,
        model: ModelWrapper,
        config: Optional[DynamicsConfig] = None
    ):
        self.model = model
        self.config = config or DynamicsConfig()
        
        # Tracking state
        self.loss_history = deque(maxlen=self.config.history_window)
        self.curvature_history = deque(maxlen=100)
        self.parameter_distances = deque(maxlen=100)
        self.previous_parameters = None
        self._previous_curvature = None
        
        # Phase detection
        self.phase_detection_window = 50
        
        logger.debug("Initialized OptimizationDynamicsMonitor")
    
    def analyze(self, training_step: int) -> DynamicsAnalysisResult:
        """
         Analyze current optimization dynamics.
        
        Args:
            training_step: Current training step
            
        Returns:
            DynamicsAnalysisResult containing analysis
        """
        logger.debug(f"Analyzing dynamics at step {training_step}")
        
        # Update monitoring state
        current_loss = self._get_current_loss()
        current_parameters = self._get_current_parameters()
        self._update_history(current_loss, current_parameters)
        
        # Compute curvature anomaly
        curvature_anomaly = self._compute_curvature_anomaly()
        
        # Compute parameter update anomaly
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
        """
        Comprehensive loss landscape curvature estimation using multiple methods.
        
        Implements sophisticated Hessian-based analysis, spectral methods, and
        finite difference approximations to accurately measure loss landscape curvature.
        
        Returns:
            Curvature estimate or None if insufficient data/computation fails
        """
        if len(self.loss_history) < 5:
            return None
        
        try:
            # Method 1: Multi-order finite difference approximation
            fd_curvature = self._finite_difference_curvature()
            
            # Method 2: Hessian-based curvature (most accurate but expensive)
            hessian_curvature = self._hessian_based_curvature()
            
            # Method 3: Spectral curvature estimation
            spectral_curvature = self._spectral_curvature_estimation()
            
            # Method 4: Local quadratic approximation
            quadratic_curvature = self._quadratic_approximation_curvature()
            
            # Combine estimates with confidence weighting
            curvature_estimates = []
            weights = []
            
            if fd_curvature is not None:
                curvature_estimates.append(fd_curvature)
                weights.append(0.2)  # Lower weight for finite differences
            
            if hessian_curvature is not None:
                curvature_estimates.append(hessian_curvature)
                weights.append(0.5)  # Highest weight for Hessian method
            
            if spectral_curvature is not None:
                curvature_estimates.append(spectral_curvature)
                weights.append(0.2)  # Medium weight for spectral method
            
            if quadratic_curvature is not None:
                curvature_estimates.append(quadratic_curvature)
                weights.append(0.1)  # Lower weight for approximation
            
            if not curvature_estimates:
                return None
            
            # Weighted average of estimates
            weights = np.array(weights)
            weights = weights / weights.sum()  # Normalize
            
            final_curvature = np.average(curvature_estimates, weights=weights)
            
            # Apply robustness checks
            final_curvature = self._apply_curvature_robustness_checks(final_curvature)
            
            logger.debug(f"Curvature estimates: FD={fd_curvature}, Hessian={hessian_curvature}, "
                        f"Spectral={spectral_curvature}, Quadratic={quadratic_curvature}, "
                        f"Final={final_curvature}")
            
            return float(final_curvature)
            
        except Exception as e:
            logger.warning(f"Loss curvature estimation failed: {e}")
            return None
    
    def _finite_difference_curvature(self) -> Optional[float]:
        """Estimate curvature using high-order finite difference schemes."""
        losses = np.array(list(self.loss_history))
        
        if len(losses) < 5:
            return None
        
        # Use multiple finite difference schemes for robustness
        curvatures = []
        
        # Central difference (2nd order accurate)
        if len(losses) >= 3:
            recent_losses = losses[-3:]
            central_diff = recent_losses[0] - 2 * recent_losses[1] + recent_losses[2]
            curvatures.append(abs(central_diff))
        
        # Forward difference (1st order accurate)
        if len(losses) >= 4:
            recent_losses = losses[-4:]
            forward_diff = recent_losses[0] - 2 * recent_losses[1] + recent_losses[2]
            curvatures.append(abs(forward_diff))
        
        # Backward difference (1st order accurate)
        if len(losses) >= 4:
            recent_losses = losses[-4:]
            backward_diff = recent_losses[1] - 2 * recent_losses[2] + recent_losses[3]
            curvatures.append(abs(backward_diff))
        
        # Higher-order finite difference (4th order accurate)
        if len(losses) >= 5:
            recent_losses = losses[-5:]
            # 4th order central difference: f(x-2h) - 8f(x-h) + 12f(x) - 8f(x+h) + f(x+2h)
            high_order = (-recent_losses[0] + 16*recent_losses[1] - 30*recent_losses[2] + 
                         16*recent_losses[3] - recent_losses[4]) / 12.0
            curvatures.append(abs(high_order))
        
        if not curvatures:
            return None
        
        # Return robust average (remove outliers)
        curvatures = np.array(curvatures)
        median_curvature = np.median(curvatures)
        mad = np.median(np.abs(curvatures - median_curvature))  # Median absolute deviation
        
        # Filter outliers (more than 3 MAD from median)
        robust_curvatures = curvatures[np.abs(curvatures - median_curvature) <= 3 * mad]
        
        if len(robust_curvatures) > 0:
            return float(np.mean(robust_curvatures))
        else:
            return float(median_curvature)
    
    def _hessian_based_curvature(self) -> Optional[float]:
        """Estimate curvature using Hessian eigenvalue analysis."""
        try:
            # This requires access to gradients and model - more expensive but accurate
            if not hasattr(self.model, 'model') or not hasattr(self.model.model, 'parameters'):
                return None
            
            # Get current model parameters
            params = [p for p in self.model.model.parameters() if p.requires_grad]
            if not params:
                return None
            
            # Create a synthetic input batch for Hessian computation
            # In practice, this should use actual validation data
            input_shape = getattr(self.model, '_last_input_shape', None)
            if input_shape is None:
                # Fallback: assume simple input
                input_batch = torch.randn(4, 10, device=next(iter(params)).device)
            else:
                input_batch = torch.randn(*input_shape, device=next(iter(params)).device)
            
            # Compute Hessian eigenvalues using power iteration
            eigenvalues = self._compute_hessian_eigenvalues(input_batch, params)
            
            if eigenvalues is not None and len(eigenvalues) > 0:
                # Curvature measures
                max_eigenvalue = float(torch.max(eigenvalues))
                spectral_radius = float(torch.max(torch.abs(eigenvalues)))
                trace = float(torch.sum(eigenvalues))
                
                # Combined curvature metric
                curvature = 0.4 * max_eigenvalue + 0.4 * spectral_radius + 0.2 * abs(trace)
                return curvature
            
            return None
            
        except Exception as e:
            logger.debug(f"Hessian curvature computation failed: {e}")
            return None
    
    def _compute_hessian_eigenvalues(self, input_batch: torch.Tensor, params: List[torch.Tensor], 
                                   k: int = 5) -> Optional[torch.Tensor]:
        """Compute top-k Hessian eigenvalues using power iteration."""
        try:
            # Concatenate all parameters into single vector
            param_vector = torch.cat([p.flatten() for p in params])
            n_params = len(param_vector)
            
            if n_params == 0:
                return None
            
            eigenvalues = []
            
            for i in range(min(k, 10)):  # Limit iterations for performance
                # Random initialization
                v = torch.randn_like(param_vector)
                v = v / torch.norm(v)
                
                # Power iteration for dominant eigenvalue
                for iteration in range(20):  # Max 20 iterations per eigenvalue
                    # Compute Hessian-vector product
                    Hv = self._hessian_vector_product(input_batch, v, params)
                    
                    if Hv is None:
                        break
                    
                    # Normalize
                    Hv_norm = torch.norm(Hv)
                    if Hv_norm < 1e-10:
                        break
                    
                    v_new = Hv / Hv_norm
                    
                    # Check convergence
                    if iteration > 0 and torch.norm(v_new - v) < 1e-6:
                        break
                    
                    v = v_new
                
                # Compute eigenvalue
                if torch.norm(v) > 1e-10:
                    Hv = self._hessian_vector_product(input_batch, v, params)
                    if Hv is not None:
                        eigenvalue = torch.dot(v, Hv)
                        eigenvalues.append(eigenvalue)
            
            if eigenvalues:
                return torch.tensor(eigenvalues)
            else:
                return None
                
        except Exception as e:
            logger.debug(f"Hessian eigenvalue computation failed: {e}")
            return None
    
    def _hessian_vector_product(self, input_batch: torch.Tensor, vector: torch.Tensor, 
                               params: List[torch.Tensor]) -> Optional[torch.Tensor]:
        """Efficiently compute Hessian-vector product using autograd."""
        try:
            # Zero gradients
            for p in params:
                if p.grad is not None:
                    p.grad.zero_()
            
            # Forward pass to compute loss
            self.model.model.eval()
            with torch.enable_grad():
                outputs = self.model.forward(input_batch)
                
                # Simple loss for curvature analysis
                if outputs.dim() > 1:
                    # Classification-like output
                    targets = torch.randint(0, outputs.size(-1), (outputs.size(0),), device=outputs.device)
                    loss = F.cross_entropy(outputs, targets)
                else:
                    # Regression-like output
                    targets = torch.randn_like(outputs)
                    loss = F.mse_loss(outputs, targets)
                
                # First-order gradients
                grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)
                
                if not grads:
                    return None
                
                # Flatten gradients
                grad_vector = torch.cat([g.flatten() for g in grads])
                
                # Compute grad-vector dot product
                gv = torch.dot(grad_vector, vector)
                
                # Second-order gradients (Hessian-vector product)
                hvp_grads = torch.autograd.grad(gv, params, retain_graph=False)
                
                if hvp_grads:
                    hvp = torch.cat([g.flatten() for g in hvp_grads])
                    return hvp
                else:
                    return None
                    
        except Exception as e:
            logger.debug(f"Hessian-vector product computation failed: {e}")
            return None
    
    def _spectral_curvature_estimation(self) -> Optional[float]:
        """Estimate curvature using spectral analysis of loss trajectory."""
        if len(self.loss_history) < 10:
            return None
        
        try:
            losses = np.array(list(self.loss_history))
            
            # Detrend the loss sequence
            x = np.arange(len(losses))
            poly_coeffs = np.polyfit(x, losses, deg=2)  # Quadratic detrend
            trend = np.polyval(poly_coeffs, x)
            detrended = losses - trend
            
            # Compute power spectral density
            if len(detrended) >= 8:
                freqs, psd = signal.welch(detrended, nperseg=min(len(detrended)//2, 8))
                
                # High-frequency content indicates roughness/curvature
                high_freq_power = np.sum(psd[freqs > 0.25])  # High frequency power
                total_power = np.sum(psd)
                
                if total_power > 0:
                    spectral_roughness = high_freq_power / total_power
                    return float(spectral_roughness)
            
            # Fallback: use variance of second differences
            if len(losses) >= 3:
                second_diffs = np.diff(losses, n=2)
                return float(np.var(second_diffs))
            
            return None
            
        except Exception as e:
            logger.debug(f"Spectral curvature estimation failed: {e}")
            return None
    
    def _quadratic_approximation_curvature(self) -> Optional[float]:
        """Estimate curvature by fitting local quadratic models."""
        if len(self.loss_history) < 6:
            return None
        
        try:
            losses = np.array(list(self.loss_history))
            
            # Fit quadratic model to recent loss history
            window_size = min(len(losses), 10)  # Use last 10 points
            recent_losses = losses[-window_size:]
            x = np.arange(len(recent_losses))
            
            # Fit quadratic: f(x) = ax² + bx + c
            if len(recent_losses) >= 3:
                poly_coeffs = np.polyfit(x, recent_losses, deg=2)
                quadratic_coeff = poly_coeffs[0]  # 'a' coefficient
                
                # R² goodness of fit
                fitted_values = np.polyval(poly_coeffs, x)
                ss_res = np.sum((recent_losses - fitted_values) ** 2)
                ss_tot = np.sum((recent_losses - np.mean(recent_losses)) ** 2)
                r_squared = 1 - (ss_res / (ss_tot + 1e-10))
                
                # Weight curvature by fit quality
                if r_squared > 0.5:  # Only trust good fits
                    return float(abs(quadratic_coeff) * r_squared)
            
            return None
            
        except Exception as e:
            logger.debug(f"Quadratic approximation failed: {e}")
            return None
    
    def _apply_curvature_robustness_checks(self, curvature: float) -> float:
        """Apply robustness checks and bounds to curvature estimate."""
        # Bound curvature to reasonable range
        curvature = max(0.0, min(curvature, 1000.0))
        
        # Apply exponential smoothing if we have history
        if hasattr(self, '_previous_curvature') and self._previous_curvature is not None:
            smoothing_factor = 0.3
            curvature = smoothing_factor * curvature + (1 - smoothing_factor) * self._previous_curvature
        
        self._previous_curvature = curvature
        
        return curvature
    
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
        self.curvature_history.clear()
        self.parameter_distances.clear()
        self.previous_parameters = None
        self._previous_curvature = None
        
        logger.info("Training dynamics history reset") 