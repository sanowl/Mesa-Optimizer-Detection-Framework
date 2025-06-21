"""
Gradient-Based Mesa-Optimization Detection

This module implements gradient analysis techniques for detecting mesa-optimization
patterns, including gradient variance analysis, Hessian eigenvalue analysis,
and gradient direction anomaly detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Any, Tuple
import numpy as np
import logging
from scipy.stats import entropy
import warnings

from ..core.results import GradientAnalysisResult
from ..utils.model_utils import ModelWrapper
from ..config import GradientConfig

logger = logging.getLogger(__name__)


class GradientAnalyzer:
    """
    Analyzes gradient patterns to detect mesa-optimization.
    
    This analyzer implements several gradient-based detection methods:
    1. Gradient variance analysis across similar inputs
    2. Gradient direction anomaly detection
    3. Hessian eigenvalue analysis (optional)
    4. Optimization trajectory analysis
    """
    
    def __init__(
        self,
        model: ModelWrapper,
        config: Optional[GradientConfig] = None
    ):
        if not isinstance(model, ModelWrapper):
            raise TypeError("model must be a ModelWrapper instance")
        
        self.model = model
        self.config = config or GradientConfig()
        self.gradient_history = []
        self.baseline_gradients = None
        
        logger.info("Initialized GradientAnalyzer")
    
    def analyze(
        self,
        input_batch: torch.Tensor,
        baseline_gradients: Optional[torch.Tensor] = None,
        target_batch: Optional[torch.Tensor] = None
    ) -> GradientAnalysisResult:
        """
        Perform gradient analysis on the given input batch.
        
        Args:
            input_batch: Input data for gradient computation
            baseline_gradients: Optional baseline gradients for comparison
            target_batch: Optional target data for supervised learning
            
        Returns:
            GradientAnalysisResult containing analysis results
        """
        if not isinstance(input_batch, torch.Tensor):
            raise TypeError("input_batch must be a torch.Tensor")
        
        if input_batch.numel() == 0:
            logger.warning("Empty input batch provided")
            return self._create_empty_result()
        
        logger.debug("Starting gradient analysis")
        
        try:
            # Compute gradients for current batch
            current_gradients = self._compute_gradients(input_batch, target_batch)
            
            if current_gradients is None or current_gradients.numel() == 0:
                logger.warning("Failed to compute gradients")
                return self._create_empty_result()
            
            # Set baseline if not provided
            if baseline_gradients is None and self.baseline_gradients is not None:
                baseline_gradients = self.baseline_gradients
            elif baseline_gradients is None:
                # Use current gradients as baseline for first analysis
                self.baseline_gradients = current_gradients.clone()
                baseline_gradients = current_gradients
            
            # Validate baseline gradients
            if not isinstance(baseline_gradients, torch.Tensor) or baseline_gradients.numel() == 0:
                logger.warning("Invalid baseline gradients, using current gradients")
                baseline_gradients = current_gradients
            
            # Compute gradient variance
            gradient_variance = self._compute_gradient_variance(
                current_gradients, baseline_gradients
            )
            
            # Detect gradient anomalies
            anomaly_score = self._detect_gradient_anomalies(
                current_gradients, baseline_gradients
            )
            
            # Optional Hessian analysis
            hessian_eigenvalues = None
            if self.config.hessian_analysis:
                try:
                    hessian_eigenvalues = self._compute_hessian_eigenvalues(
                        input_batch, target_batch
                    )
                except Exception as e:
                    logger.warning(f"Hessian analysis failed: {e}")
                    hessian_eigenvalues = None
            
            # Compute gradient directions
            gradient_directions = self._analyze_gradient_directions(current_gradients)
            
            # Compute overall risk score
            risk_score = self._compute_risk_score(
                gradient_variance, anomaly_score, hessian_eigenvalues
            )
            
            # Compute confidence
            confidence = self._compute_confidence(gradient_variance, anomaly_score)
            
            # Store gradients in history with memory management
            self._update_gradient_history(current_gradients)
            
            result = GradientAnalysisResult(
                gradient_variance=gradient_variance,
                anomaly_score=anomaly_score,
                hessian_eigenvalues=hessian_eigenvalues,
                gradient_directions=gradient_directions,
                risk_score=risk_score,
                confidence=confidence
            )
            
            logger.debug(f"Gradient analysis complete. Risk score: {risk_score:.3f}")
            return result
            
        except Exception as e:
            logger.error(f"Gradient analysis failed: {e}")
            return self._create_empty_result()
    
    def _create_empty_result(self) -> GradientAnalysisResult:
        """Create an empty result when analysis fails."""
        return GradientAnalysisResult(
            gradient_variance=0.0,
            anomaly_score=0.0,
            hessian_eigenvalues=None,
            gradient_directions=None,
            risk_score=0.0,
            confidence=0.0
        )
    
    def _update_gradient_history(self, gradients: torch.Tensor) -> None:
        """Update gradient history with proper memory management."""
        # Clone to avoid reference issues and move to CPU to save GPU memory
        gradient_copy = gradients.detach().clone().cpu()
        
        self.gradient_history.append(gradient_copy)
        
        # Keep only last 100 gradients to prevent memory leak
        if len(self.gradient_history) > 100:
            # Explicitly delete old gradients
            old_gradient = self.gradient_history.pop(0)
            del old_gradient
    
    def _compute_gradients(
        self,
        input_batch: torch.Tensor,
        target_batch: Optional[torch.Tensor] = None
    ) -> Optional[torch.Tensor]:
        """Compute gradients with respect to model parameters."""
        try:
            # Ensure model is in training mode for gradient computation
            was_training = self.model.model.training
            self.model.model.train()
            
            # Clear any existing gradients
            self.model.model.zero_grad()
            
            # Move inputs to correct device
            input_batch = input_batch.to(self.model.device)
            if target_batch is not None:
                target_batch = target_batch.to(self.model.device)
            
            # Forward pass
            outputs = self.model.forward(input_batch)
            
            if not isinstance(outputs, torch.Tensor):
                raise ValueError(f"Model output must be a tensor, got {type(outputs)}")
            
            if target_batch is not None:
                # Supervised learning case
                if hasattr(self.model.model, 'compute_loss'):
                    loss = self.model.model.compute_loss(outputs, target_batch)
                else:
                    # Validate target shape compatibility
                    if outputs.shape[0] != target_batch.shape[0]:
                        raise ValueError(f"Batch size mismatch: outputs {outputs.shape[0]} vs targets {target_batch.shape[0]}")
                    
                    # Handle different output types
                    if outputs.dim() >= 2 and outputs.size(-1) > 1:
                        # Classification case
                        if target_batch.dim() == 1:
                            # Integer targets
                            loss = F.cross_entropy(outputs, target_batch)
                        else:
                            # One-hot or soft targets
                            loss = F.cross_entropy(outputs, target_batch.float())
                    else:
                        # Regression case
                        loss = F.mse_loss(outputs, target_batch)
            else:
                # Unsupervised case - create meaningful loss for gradient analysis
                if hasattr(self.model.model, 'compute_loss'):
                    try:
                        loss = self.model.model.compute_loss(outputs, input_batch)
                    except Exception:
                        # Fallback to simple loss
                        loss = outputs.norm()
                else:
                    # Create a loss that encourages meaningful gradients
                    if outputs.dim() >= 2:
                        # For multi-dimensional outputs, use regularization-like loss
                        loss = torch.norm(outputs, p=2) + 0.01 * torch.norm(outputs, p=1)
                    else:
                        # For scalar outputs
                        loss = torch.norm(outputs)
            
            # Ensure loss is scalar
            if loss.dim() > 0:
                loss = loss.mean()
            
            # Backward pass
            loss.backward()
            
            # Extract gradients
            gradients = []
            total_params = 0
            
            for param in self.model.model.parameters():
                if param.grad is not None:
                    # Check for NaN or Inf gradients
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        logger.warning("NaN or Inf gradients detected, skipping parameter")
                        continue
                    
                    gradients.append(param.grad.flatten())
                    total_params += param.grad.numel()
            
            # Restore original training mode
            self.model.model.train(was_training)
            
            if gradients:
                concatenated = torch.cat(gradients)
                logger.debug(f"Computed gradients for {total_params} parameters")
                return concatenated
            else:
                logger.warning("No valid gradients computed")
                return None
                
        except Exception as e:
            logger.error(f"Gradient computation failed: {e}")
            # Restore original training mode
            self.model.model.train(was_training if 'was_training' in locals() else False)
            return None
    
    def _compute_gradient_variance(
        self,
        current_gradients: torch.Tensor,
        baseline_gradients: torch.Tensor
    ) -> float:
        """Compute variance between current and baseline gradients."""
        try:
            # Handle shape mismatches
            if current_gradients.shape != baseline_gradients.shape:
                logger.warning("Gradient shapes don't match, using norm ratio")
                current_norm = torch.norm(current_gradients)
                baseline_norm = torch.norm(baseline_gradients)
                
                # Avoid division by zero
                if baseline_norm < 1e-10:
                    return 1.0 if current_norm > 1e-10 else 0.0
                
                return float(current_norm / baseline_norm)
            
            # Compute normalized difference
            diff = current_gradients - baseline_gradients
            
            # Check for all-zero differences
            if torch.allclose(diff, torch.zeros_like(diff), atol=1e-10):
                return 0.0
            
            # Compute variance with numerical stability
            variance = torch.var(diff)
            
            # Handle potential NaN/Inf values
            if torch.isnan(variance) or torch.isinf(variance):
                logger.warning("Invalid variance computed, using fallback")
                return float(torch.norm(diff) / (torch.norm(baseline_gradients) + 1e-8))
            
            return float(variance)
            
        except Exception as e:
            logger.warning(f"Gradient variance computation failed: {e}")
            return 0.0
    
    def _detect_gradient_anomalies(
        self,
        current_gradients: torch.Tensor,
        baseline_gradients: torch.Tensor
    ) -> float:
        """Detect anomalous gradient patterns that may indicate mesa-optimization."""
        try:
            anomaly_indicators = []
            
            # 1. Gradient norm anomaly
            current_norm = torch.norm(current_gradients)
            baseline_norm = torch.norm(baseline_gradients)
            
            # Avoid division by zero
            if baseline_norm < 1e-10:
                norm_ratio = 10.0 if current_norm > 1e-10 else 1.0
            else:
                norm_ratio = current_norm / baseline_norm
            
            if norm_ratio > 2.0 or norm_ratio < 0.5:
                anomaly_indicators.append(min(abs(norm_ratio - 1.0), 1.0))
            
            # 2. Direction anomaly (if shapes match)
            if current_gradients.shape == baseline_gradients.shape:
                try:
                    # Compute cosine similarity
                    current_flat = current_gradients.flatten()
                    baseline_flat = baseline_gradients.flatten()
                    
                    current_norm_flat = torch.norm(current_flat)
                    baseline_norm_flat = torch.norm(baseline_flat)
                    
                    if current_norm_flat > 1e-10 and baseline_norm_flat > 1e-10:
                        cosine_sim = torch.dot(current_flat, baseline_flat) / (current_norm_flat * baseline_norm_flat)
                        cosine_sim = torch.clamp(cosine_sim, -1.0, 1.0)
                        
                        # High anomaly if directions are very different
                        direction_anomaly = 1.0 - (cosine_sim + 1.0) / 2.0
                        
                        if direction_anomaly > 0.5:
                            anomaly_indicators.append(float(direction_anomaly))
                except Exception as e:
                    logger.debug(f"Direction anomaly computation failed: {e}")
            
            # 3. Gradient sparsity anomaly
            try:
                # Check for unusual sparsity patterns
                current_sparsity = (current_gradients.abs() < 1e-8).float().mean()
                baseline_sparsity = (baseline_gradients.abs() < 1e-8).float().mean()
                
                sparsity_diff = abs(current_sparsity - baseline_sparsity)
                if sparsity_diff > 0.3:
                    anomaly_indicators.append(float(sparsity_diff))
            except Exception as e:
                logger.debug(f"Sparsity anomaly computation failed: {e}")
            
            # 4. Gradient distribution anomaly
            try:
                if len(self.gradient_history) > 5:
                    # Compare current gradients to historical distribution
                    recent_norms = []
                    for hist_grad in self.gradient_history[-5:]:
                        hist_norm = torch.norm(hist_grad)
                        recent_norms.append(float(hist_norm))
                    
                    if recent_norms:
                        mean_norm = np.mean(recent_norms)
                        std_norm = np.std(recent_norms)
                        
                        if std_norm > 1e-10:
                            z_score = abs(float(current_norm) - mean_norm) / std_norm
                            if z_score > 2.0:  # 2 standard deviations
                                anomaly_indicators.append(min(z_score / 5.0, 1.0))
            except Exception as e:
                logger.debug(f"Distribution anomaly computation failed: {e}")
            
            # Aggregate anomaly score
            if anomaly_indicators:
                anomaly_score = np.mean(anomaly_indicators)
                return float(np.clip(anomaly_score, 0.0, 1.0))
            else:
                return 0.0
                
        except Exception as e:
            logger.warning(f"Gradient anomaly detection failed: {e}")
            return 0.0
    
    def _compute_hessian_eigenvalues(
        self,
        input_batch: torch.Tensor,
        target_batch: Optional[torch.Tensor] = None
    ) -> Optional[torch.Tensor]:
        """Compute top Hessian eigenvalues using power iteration method."""
        try:
            # Use power iteration for large models to avoid memory issues
            eigenvalues = self._power_iteration_hessian(
                input_batch, target_batch, k=self.config.max_eigenvalues
            )
            return eigenvalues
        except Exception as e:
            logger.warning(f"Hessian eigenvalue computation failed: {e}")
            return None
    
    def _power_iteration_hessian(
        self,
        input_batch: torch.Tensor,
        target_batch: Optional[torch.Tensor],
        k: int = 10
    ) -> torch.Tensor:
        """Use power iteration to compute top-k Hessian eigenvalues."""
        try:
            # Get total parameter count
            total_params = sum(p.numel() for p in self.model.model.parameters() if p.requires_grad)
            
            if total_params == 0:
                return torch.tensor([])
            
            eigenvalues = []
            
            for i in range(min(k, 5)):  # Limit to avoid excessive computation
                # Random initialization
                v = torch.randn(total_params, device=self.model.device)
                v = v / torch.norm(v)
                
                # Power iteration
                for _ in range(10):  # Reduced iterations for efficiency
                    # Compute Hv using automatic differentiation
                    Hv = self._hessian_vector_product(input_batch, target_batch, v)
                    
                    if Hv is None:
                        break
                    
                    # Normalize
                    norm = torch.norm(Hv)
                    if norm < 1e-10:
                        break
                    
                    v = Hv / norm
                
                # Compute eigenvalue
                if Hv is not None:
                    eigenvalue = torch.dot(v, Hv)
                    eigenvalues.append(eigenvalue)
            
            if eigenvalues:
                return torch.stack(eigenvalues)
            else:
                return torch.tensor([])
                
        except Exception as e:
            logger.warning(f"Power iteration failed: {e}")
            return torch.tensor([])
    
    def _hessian_vector_product(
        self,
        input_batch: torch.Tensor,
        target_batch: Optional[torch.Tensor],
        vector: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """Compute Hessian-vector product using automatic differentiation."""
        try:
            # Clear gradients
            self.model.model.zero_grad()
            
            # Forward pass
            outputs = self.model.forward(input_batch)
            
            # Compute loss
            if target_batch is not None:
                if hasattr(self.model.model, 'compute_loss'):
                    loss = self.model.model.compute_loss(outputs, target_batch)
                else:
                    if outputs.dim() >= 2 and outputs.size(-1) > 1:
                        loss = F.cross_entropy(outputs, target_batch)
                    else:
                        loss = F.mse_loss(outputs, target_batch)
            else:
                loss = torch.norm(outputs)
            
            # First derivatives
            params = [p for p in self.model.model.parameters() if p.requires_grad]
            first_grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)
            
            # Flatten first gradients
            flat_grads = torch.cat([g.flatten() for g in first_grads])
            
            # Dot product with vector
            grad_vector_dot = torch.dot(flat_grads, vector)
            
            # Second derivatives (Hessian-vector product)
            second_grads = torch.autograd.grad(grad_vector_dot, params, retain_graph=False)
            
            # Flatten and return
            hv = torch.cat([g.flatten() for g in second_grads])
            return hv
            
        except Exception as e:
            logger.debug(f"Hessian-vector product computation failed: {e}")
            return None
    
    def _analyze_gradient_directions(self, gradients: torch.Tensor) -> Optional[torch.Tensor]:
        """Analyze gradient directions for optimization patterns."""
        try:
            if gradients.numel() == 0:
                return None
            
            # Reshape for analysis if needed
            if gradients.dim() == 1:
                # Split into chunks for direction analysis
                chunk_size = min(1000, gradients.numel() // 10)
                if chunk_size > 0:
                    chunks = gradients.split(chunk_size)
                    direction_features = []
                    
                    for chunk in chunks:
                        # Compute direction statistics
                        chunk_norm = torch.norm(chunk)
                        if chunk_norm > 1e-10:
                            normalized_chunk = chunk / chunk_norm
                            
                            # Features: mean, std, skewness proxy, kurtosis proxy
                            features = torch.tensor([
                                torch.mean(normalized_chunk),
                                torch.std(normalized_chunk),
                                torch.mean(torch.pow(normalized_chunk, 3)),
                                torch.mean(torch.pow(normalized_chunk, 4))
                            ])
                            direction_features.append(features)
                    
                    if direction_features:
                        return torch.stack(direction_features)
            
            return None
            
        except Exception as e:
            logger.debug(f"Gradient direction analysis failed: {e}")
            return None
    
    def _compute_risk_score(
        self,
        gradient_variance: float,
        anomaly_score: float,
        hessian_eigenvalues: Optional[torch.Tensor]
    ) -> float:
        """Compute overall risk score from gradient analysis."""
        try:
            risk_components = []
            
            # Gradient variance risk
            if gradient_variance > self.config.variance_threshold:
                variance_risk = min(gradient_variance / self.config.variance_threshold, 1.0)
                risk_components.append(variance_risk * 0.4)
            
            # Anomaly risk
            if anomaly_score > self.config.anomaly_threshold:
                anomaly_risk = min(anomaly_score, 1.0)
                risk_components.append(anomaly_risk * 0.4)
            
            # Hessian risk
            if hessian_eigenvalues is not None and hessian_eigenvalues.numel() > 0:
                # Check for negative eigenvalues (non-convexity indicators)
                negative_eigenvalues = (hessian_eigenvalues < 0).float().mean()
                
                # Check for extreme eigenvalue ratios
                if hessian_eigenvalues.numel() > 1:
                    max_eigenval = torch.max(hessian_eigenvalues)
                    min_eigenval = torch.min(hessian_eigenvalues)
                    
                    if min_eigenval != 0:
                        condition_number = abs(max_eigenval / min_eigenval)
                        if condition_number > 1000:  # High condition number
                            hessian_risk = min(condition_number / 10000, 1.0)
                            risk_components.append(hessian_risk * 0.2)
                
                if negative_eigenvalues > 0.1:
                    risk_components.append(float(negative_eigenvalues) * 0.2)
            
            # Aggregate risk
            if risk_components:
                total_risk = np.sum(risk_components)
                return float(np.clip(total_risk, 0.0, 1.0))
            else:
                return 0.0
                
        except Exception as e:
            logger.warning(f"Risk score computation failed: {e}")
            return 0.0
    
    def _compute_confidence(
        self,
        gradient_variance: float,
        anomaly_score: float
    ) -> float:
        """Compute confidence in gradient analysis results."""
        try:
            confidence_factors = []
            
            # Historical data availability
            if len(self.gradient_history) >= 5:
                confidence_factors.append(1.0)
            elif len(self.gradient_history) >= 2:
                confidence_factors.append(0.7)
            else:
                confidence_factors.append(0.3)
            
            # Gradient magnitude sufficiency
            if hasattr(self, 'baseline_gradients') and self.baseline_gradients is not None:
                baseline_norm = torch.norm(self.baseline_gradients)
                if baseline_norm > 1e-6:
                    confidence_factors.append(1.0)
                elif baseline_norm > 1e-8:
                    confidence_factors.append(0.5)
                else:
                    confidence_factors.append(0.1)
            
            # Consistency of measurements
            if not np.isnan(gradient_variance) and not np.isnan(anomaly_score):
                confidence_factors.append(1.0)
            else:
                confidence_factors.append(0.1)
            
            return float(np.mean(confidence_factors)) if confidence_factors else 0.5
            
        except Exception as e:
            logger.warning(f"Confidence computation failed: {e}")
            return 0.5


class GradientAnomalyDetector:
    """
    Specialized detector for gradient anomalies using statistical methods.
    """
    
    def __init__(self, window_size: int = 50):
        self.window_size = max(1, window_size)
        self.gradient_norms = []
        self.gradient_directions = []
    
    def detect_anomalies(self, gradients: torch.Tensor) -> Tuple[bool, float]:
        """
        Detect if current gradients are anomalous.
        
        Returns:
            Tuple of (is_anomalous, anomaly_score)
        """
        try:
            if not isinstance(gradients, torch.Tensor) or gradients.numel() == 0:
                return False, 0.0
            
            current_norm = float(torch.norm(gradients))
            
            # Store current measurements
            self.gradient_norms.append(current_norm)
            
            # Maintain window size
            if len(self.gradient_norms) > self.window_size:
                self.gradient_norms.pop(0)
            
            # Need at least 10 samples for meaningful detection
            if len(self.gradient_norms) < 10:
                return False, 0.0
            
            # Statistical anomaly detection
            recent_norms = np.array(self.gradient_norms)
            mean_norm = np.mean(recent_norms)
            std_norm = np.std(recent_norms)
            
            if std_norm < 1e-10:
                return False, 0.0
            
            # Z-score based detection
            z_score = abs(current_norm - mean_norm) / std_norm
            
            # Anomaly if z-score > 2.5 (roughly 1% chance for normal distribution)
            is_anomalous = z_score > 2.5
            anomaly_score = min(z_score / 5.0, 1.0)  # Normalize to [0, 1]
            
            return is_anomalous, float(anomaly_score)
            
        except Exception as e:
            logger.warning(f"Anomaly detection failed: {e}")
            return False, 0.0 