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
        logger.debug("Starting gradient analysis")
        
        # Compute gradients for current batch
        current_gradients = self._compute_gradients(input_batch, target_batch)
        
        # Set baseline if not provided
        if baseline_gradients is None and self.baseline_gradients is not None:
            baseline_gradients = self.baseline_gradients
        elif baseline_gradients is None:
            # Use current gradients as baseline for first analysis
            self.baseline_gradients = current_gradients
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
            hessian_eigenvalues = self._compute_hessian_eigenvalues(
                input_batch, target_batch
            )
        
        # Compute gradient directions
        gradient_directions = self._analyze_gradient_directions(current_gradients)
        
        # Compute overall risk score
        risk_score = self._compute_risk_score(
            gradient_variance, anomaly_score, hessian_eigenvalues
        )
        
        # Compute confidence
        confidence = self._compute_confidence(gradient_variance, anomaly_score)
        
        # Store gradients in history
        self.gradient_history.append(current_gradients)
        if len(self.gradient_history) > 100:  # Keep last 100 gradients
            self.gradient_history.pop(0)
        
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
    
    def _compute_gradients(
        self,
        input_batch: torch.Tensor,
        target_batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute gradients with respect to model parameters."""
        self.model.model.zero_grad()
        
        # Forward pass
        outputs = self.model.forward(input_batch)
        
        if target_batch is not None:
            # Supervised learning case
            if hasattr(self.model.model, 'compute_loss'):
                loss = self.model.model.compute_loss(outputs, target_batch)
            else:
                # Default cross-entropy loss
                loss = F.cross_entropy(outputs, target_batch)
        else:
            # Unsupervised case - create appropriate targets based on output shape
            if hasattr(self.model.model, 'compute_loss'):
                loss = self.model.model.compute_loss(outputs, input_batch)
            else:
                # Create targets that match the output shape
                if outputs.dim() >= 2:
                    # Classification-like output
                    targets = torch.randint(0, outputs.size(-1), (outputs.size(0),), device=outputs.device)
                    loss = F.cross_entropy(outputs, targets)
                else:
                    # Regression-like output - use MSE with zero targets
                    targets = torch.zeros_like(outputs)
                    loss = F.mse_loss(outputs, targets)
        
        # Backward pass
        loss.backward()
        
        # Extract gradients
        gradients = []
        for param in self.model.model.parameters():
            if param.grad is not None:
                gradients.append(param.grad.flatten())
        
        if gradients:
            return torch.cat(gradients)
        else:
            logger.warning("No gradients computed")
            return torch.zeros(1, device=self.model.device)
    
    def _compute_gradient_variance(
        self,
        current_gradients: torch.Tensor,
        baseline_gradients: torch.Tensor
    ) -> float:
        """Compute variance between current and baseline gradients."""
        if current_gradients.shape != baseline_gradients.shape:
            logger.warning("Gradient shapes don't match, using norm ratio")
            return float(torch.norm(current_gradients) / 
                        (torch.norm(baseline_gradients) + 1e-8))
        
        # Compute normalized difference
        diff = current_gradients - baseline_gradients
        variance = torch.var(diff).item()
        
        return variance
    
    def _detect_gradient_anomalies(
        self,
        current_gradients: torch.Tensor,
        baseline_gradients: torch.Tensor
    ) -> float:
        """Detect anomalous gradient patterns that may indicate mesa-optimization."""
        anomaly_indicators = []
        
        # 1. Gradient norm anomaly
        current_norm = torch.norm(current_gradients)
        baseline_norm = torch.norm(baseline_gradients)
        norm_ratio = current_norm / (baseline_norm + 1e-8)
        
        if norm_ratio > 2.0 or norm_ratio < 0.5:
            anomaly_indicators.append(0.3)
        
        # 2. Gradient direction shift
        if current_gradients.shape == baseline_gradients.shape:
            cosine_sim = F.cosine_similarity(
                current_gradients.unsqueeze(0), 
                baseline_gradients.unsqueeze(0)
            )
            direction_anomaly = 1.0 - cosine_sim.item()
            
            if direction_anomaly > 0.5:
                anomaly_indicators.append(direction_anomaly)
        
        # 3. Gradient sparsity pattern
        current_sparsity = (current_gradients.abs() < 1e-6).float().mean()
        baseline_sparsity = (baseline_gradients.abs() < 1e-6).float().mean()
        sparsity_diff = abs(current_sparsity - baseline_sparsity)
        
        if sparsity_diff > 0.2:
            anomaly_indicators.append(sparsity_diff)
        
        # 4. Gradient distribution analysis
        if len(current_gradients) > 100:
            current_hist = torch.histc(current_gradients, bins=20)
            baseline_hist = torch.histc(baseline_gradients, bins=20)
            
            # Normalize histograms
            current_hist = current_hist / current_hist.sum()
            baseline_hist = baseline_hist / baseline_hist.sum()
            
            # Compute KL divergence
            kl_div = F.kl_div(
                torch.log(current_hist + 1e-8),
                baseline_hist + 1e-8,
                reduction='sum'
            )
            
            if kl_div > 1.0:
                anomaly_indicators.append(min(kl_div.item() / 5.0, 1.0))
        
        # Aggregate anomaly indicators
        if anomaly_indicators:
            return float(np.mean(anomaly_indicators))
        else:
            return 0.0
    
    def _compute_hessian_eigenvalues(
        self,
        input_batch: torch.Tensor,
        target_batch: Optional[torch.Tensor] = None
    ) -> Optional[torch.Tensor]:
        """Compute top eigenvalues of the Hessian matrix."""
        try:
            # This is computationally expensive - use power iteration for top eigenvalues
            eigenvalues = self._power_iteration_hessian(
                input_batch, target_batch, k=self.config.max_eigenvalues
            )
            return eigenvalues
        except Exception as e:
            logger.warning(f"Hessian computation failed: {e}")
            return None
    
    def _power_iteration_hessian(
        self,
        input_batch: torch.Tensor,
        target_batch: Optional[torch.Tensor],
        k: int = 10
    ) -> torch.Tensor:
        """Use power iteration to find top Hessian eigenvalues."""
        # Get model parameters
        params = [p for p in self.model.model.parameters() if p.requires_grad]
        
        # Create random initial vector
        v = torch.randn_like(torch.cat([p.flatten() for p in params]))
        v = v / torch.norm(v)
        
        eigenvalues = []
        
        for i in range(k):
            # Compute Hessian-vector product using autograd
            Hv = self._hessian_vector_product(input_batch, target_batch, v, params)
            
            # Power iteration step
            eigenvalue = torch.dot(v, Hv)
            eigenvalues.append(eigenvalue.item())
            
            # Update v for next iteration
            v = Hv / torch.norm(Hv)
            
            # Early stopping if converged
            if i > 0 and abs(eigenvalues[-1] - eigenvalues[-2]) < 1e-6:
                break
        
        return torch.tensor(eigenvalues)
    
    def _hessian_vector_product(
        self,
        input_batch: torch.Tensor,
        target_batch: Optional[torch.Tensor],
        vector: torch.Tensor,
        params: List[torch.Tensor]
    ) -> torch.Tensor:
        """Compute Hessian-vector product efficiently."""
        # First backward pass to compute gradients
        self.model.model.zero_grad()
        
        # Forward pass
        outputs = self.model.forward(input_batch)
        if target_batch is not None:
            loss = F.cross_entropy(outputs, target_batch)
        else:
            # Create targets that match the output shape
            if outputs.dim() >= 2:
                # Classification-like output
                targets = torch.randint(0, outputs.size(-1), (outputs.size(0),), device=outputs.device)
                loss = F.cross_entropy(outputs, targets)
            else:
                # Regression-like output - use MSE with zero targets
                targets = torch.zeros_like(outputs)
                loss = F.mse_loss(outputs, targets)
        
        # Compute gradients
        grads = torch.autograd.grad(loss, params, create_graph=True)
        
        # Compute gradient-vector product
        gv = torch.dot(torch.cat([g.flatten() for g in grads]), vector)
        
        # Second backward pass for Hessian-vector product
        hvp = torch.autograd.grad(gv, params, retain_graph=False)
        
        return torch.cat([h.flatten() for h in hvp])
    
    def _analyze_gradient_directions(self, gradients: torch.Tensor) -> torch.Tensor:
        """Analyze gradient direction patterns."""
        if len(self.gradient_history) < 2:
            return gradients[:10]  # Return first 10 gradient components
        
        # Compute gradient direction changes over time
        recent_gradients = self.gradient_history[-5:]  # Last 5 gradients
        
        direction_changes = []
        for i in range(1, len(recent_gradients)):
            prev_grad = recent_gradients[i-1]
            curr_grad = recent_gradients[i]
            
            if prev_grad.shape == curr_grad.shape:
                cosine_sim = F.cosine_similarity(
                    prev_grad.unsqueeze(0),
                    curr_grad.unsqueeze(0)
                )
                direction_changes.append(1.0 - cosine_sim.item())
        
        # Return statistics about direction changes
        if direction_changes:
            return torch.tensor([
                np.mean(direction_changes),
                np.std(direction_changes),
                np.max(direction_changes),
                np.min(direction_changes)
            ])
        else:
            return torch.zeros(4)
    
    def _compute_risk_score(
        self,
        gradient_variance: float,
        anomaly_score: float,
        hessian_eigenvalues: Optional[torch.Tensor]
    ) -> float:
        """Compute overall risk score from gradient analysis."""
        risk_components = []
        
        # Gradient variance component
        if gradient_variance > self.config.variance_threshold:
            variance_risk = min(gradient_variance / self.config.variance_threshold, 1.0)
            risk_components.append(variance_risk * 0.3)
        
        # Anomaly score component
        if anomaly_score > self.config.anomaly_threshold:
            anomaly_risk = min(anomaly_score / self.config.anomaly_threshold, 1.0)
            risk_components.append(anomaly_risk * 0.4)
        
        # Hessian eigenvalue component
        if hessian_eigenvalues is not None and len(hessian_eigenvalues) > 0:
            # Check for suspicious eigenvalue patterns
            max_eigenvalue = hessian_eigenvalues.max().item()
            eigenvalue_ratio = hessian_eigenvalues.max() / (hessian_eigenvalues.min() + 1e-8)
            
            # High condition number may indicate optimization instability
            if eigenvalue_ratio > 100:
                hessian_risk = min(eigenvalue_ratio / 1000, 1.0)
                risk_components.append(hessian_risk * 0.3)
        
        # Aggregate risk score
        if risk_components:
            return float(np.mean(risk_components))
        else:
            return 0.0
    
    def _compute_confidence(
        self,
        gradient_variance: float,
        anomaly_score: float
    ) -> float:
        """Compute confidence in the gradient analysis."""
        confidence_factors = []
        
        # Higher variance increases confidence in detection
        if gradient_variance > 0.1:
            confidence_factors.append(min(gradient_variance * 2, 1.0))
        
        # Higher anomaly score increases confidence
        if anomaly_score > 0.1:
            confidence_factors.append(min(anomaly_score * 2, 1.0))
        
        # More gradient history increases confidence
        history_confidence = min(len(self.gradient_history) / 20, 1.0)
        confidence_factors.append(history_confidence)
        
        return float(np.mean(confidence_factors)) if confidence_factors else 0.5


class GradientAnomalyDetector:
    """
    Specialized detector for gradient anomalies using statistical methods.
    
    This detector uses statistical analysis and machine learning techniques
    to identify unusual gradient patterns that may indicate mesa-optimization.
    """
    
    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.gradient_buffer = []
        self.anomaly_threshold = 2.0  # Standard deviations
        
    def detect_anomalies(self, gradients: torch.Tensor) -> Tuple[bool, float]:
        """
        Detect if current gradients are anomalous.
        
        Returns:
            Tuple of (is_anomalous, anomaly_score)
        """
        self.gradient_buffer.append(gradients)
        
        if len(self.gradient_buffer) > self.window_size:
            self.gradient_buffer.pop(0)
        
        if len(self.gradient_buffer) < 10:
            return False, 0.0
        
        # Compute gradient norms
        norms = [torch.norm(g).item() for g in self.gradient_buffer]
        
        # Statistical anomaly detection
        mean_norm = np.mean(norms[:-1])  # Exclude current
        std_norm = np.std(norms[:-1])
        
        current_norm = norms[-1]
        z_score = abs(current_norm - mean_norm) / (std_norm + 1e-8)
        
        is_anomalous = z_score > self.anomaly_threshold
        anomaly_score = min(z_score / self.anomaly_threshold, 1.0)
        
        return is_anomalous, anomaly_score 