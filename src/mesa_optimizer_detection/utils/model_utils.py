"""
Model Utilities for Mesa-Optimizer Detection

This module provides utilities for interacting with models, extracting activations,
and performing model-specific operations needed for detection analysis.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Callable, Any, Optional, Union
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class ModelWrapper:
    """
    Wrapper class for PyTorch models to provide unified interface
    for mesa-optimization detection.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None,
        tokenizer: Optional[Any] = None
    ):
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = tokenizer
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Store for activation extraction
        self.activation_hooks = {}
        self.stored_activations = {}
        
        logger.info(f"ModelWrapper initialized with device: {self.device}")
    
    def forward(self, inputs: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass through the model."""
        inputs = inputs.to(self.device)
        return self.model(inputs, **kwargs)
    
    def register_activation_hook(
        self,
        layer_name: str,
        hook_fn: Optional[Callable] = None
    ) -> None:
        """Register hook to capture activations from specified layer."""
        if hook_fn is None:
            hook_fn = self._default_activation_hook(layer_name)
        
        # Find the layer by name
        layer = self._get_layer_by_name(layer_name)
        if layer is not None:
            handle = layer.register_forward_hook(hook_fn)
            self.activation_hooks[layer_name] = handle
            logger.debug(f"Registered hook for layer: {layer_name}")
        else:
            logger.warning(f"Layer not found: {layer_name}")
    
    def _default_activation_hook(self, layer_name: str):
        """Default hook function to store activations."""
        def hook(module, input, output):
            self.stored_activations[layer_name] = output.detach().clone()
        return hook
    
    def _get_layer_by_name(self, layer_name: str) -> Optional[nn.Module]:
        """Get layer by name from the model."""
        for name, module in self.model.named_modules():
            if name == layer_name:
                return module
        return None
    
    def get_activations(self, clear_after_get: bool = True) -> Dict[str, torch.Tensor]:
        """Get stored activations and optionally clear them."""
        activations = self.stored_activations.copy()
        if clear_after_get:
            self.stored_activations.clear()
        return activations
    
    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for handle in self.activation_hooks.values():
            handle.remove()
        self.activation_hooks.clear()
        logger.debug("All activation hooks removed")
    
    def get_layer_names(self) -> List[str]:
        """Get list of all layer names in the model."""
        return [name for name, _ in self.model.named_modules()]
    
    def get_parameter_count(self) -> int:
        """Get total number of parameters in the model."""
        return sum(p.numel() for p in self.model.parameters())


def extract_activations(
    model: Union[nn.Module, ModelWrapper],
    inputs: torch.Tensor,
    layer_indices: List[int],
    layer_names: Optional[List[str]] = None
) -> Dict[str, torch.Tensor]:
    """
    Extract activations from specified layers.
    
    Args:
        model: Model to extract activations from
        inputs: Input data
        layer_indices: Indices of layers to extract from
        layer_names: Optional specific layer names
        
    Returns:
        Dictionary mapping layer names to activation tensors
    """
    if isinstance(model, ModelWrapper):
        model_wrapper = model
    else:
        model_wrapper = ModelWrapper(model)
    
    # Get all layer names if not provided
    all_layer_names = model_wrapper.get_layer_names()
    
    if layer_names is None:
        # Use layer indices to get names
        layer_names = [all_layer_names[i] for i in layer_indices if i < len(all_layer_names)]
    
    # Register hooks for specified layers
    for layer_name in layer_names:
        model_wrapper.register_activation_hook(layer_name)
    
    try:
        # Forward pass to capture activations
        with torch.no_grad():
            _ = model_wrapper.forward(inputs)
        
        # Get captured activations
        activations = model_wrapper.get_activations()
        
    finally:
        # Clean up hooks
        model_wrapper.remove_hooks()
    
    return activations


def compute_activation_statistics(activations: torch.Tensor) -> Dict[str, float]:
    """
    Compute statistical properties of activations.
    
    Args:
        activations: Activation tensor
        
    Returns:
        Dictionary of activation statistics
    """
    stats = {}
    
    # Basic statistics
    stats['mean'] = float(activations.mean())
    stats['std'] = float(activations.std())
    stats['min'] = float(activations.min())
    stats['max'] = float(activations.max())
    
    # Sparsity (fraction of near-zero activations)
    stats['sparsity'] = float((activations.abs() < 1e-6).float().mean())
    
    # L1 and L2 norms
    stats['l1_norm'] = float(activations.abs().sum())
    stats['l2_norm'] = float(activations.norm())
    
    # Entropy (for discrete-like activations)
    if activations.numel() > 0:
        # Discretize activations for entropy computation
        discretized = torch.histc(activations.flatten(), bins=20)
        discretized = discretized / discretized.sum()
        discretized = discretized[discretized > 0]  # Remove zero bins
        if len(discretized) > 1:
            stats['entropy'] = float(-torch.sum(discretized * torch.log(discretized)))
        else:
            stats['entropy'] = 0.0
    
    return stats


def detect_planning_patterns(activations: torch.Tensor, threshold: float = 0.6) -> Dict[str, Any]:
    """
    Detect planning-like patterns in activations.
    
    Args:
        activations: Activation tensor [batch, seq_len, hidden_dim] or similar
        threshold: Threshold for pattern detection
        
    Returns:
        Dictionary with planning pattern analysis
    """
    results = {
        'planning_score': 0.0,
        'sequential_dependency': 0.0,
        'future_prediction': 0.0,
        'goal_representation': 0.0
    }
    
    if activations.dim() < 3:
        return results
    
    batch_size, seq_len, hidden_dim = activations.shape[:3]
    
    # 1. Sequential dependency analysis
    # Check if activations at position t depend on positions t+1, t+2, ...
    if seq_len > 1:
        correlations = []
        for t in range(seq_len - 1):
            current = activations[:, t, :]
            future = activations[:, t+1:, :].mean(dim=1)
            
            # Compute correlation
            corr = torch.corrcoef(torch.stack([current.flatten(), future.flatten()]))[0, 1]
            if not torch.isnan(corr):
                correlations.append(abs(corr.item()))
        
        if correlations:
            results['sequential_dependency'] = sum(correlations) / len(correlations)
    
    # 2. Future prediction patterns
    # Look for activations that seem to encode future information
    if seq_len > 2:
        future_prediction_scores = []
        for t in range(seq_len - 2):
            current = activations[:, t, :]
            future = activations[:, t+2:, :].mean(dim=1)
            
            # Measure information content about future
            mutual_info = compute_mutual_information(current, future)
            future_prediction_scores.append(mutual_info)
        
        if future_prediction_scores:
            results['future_prediction'] = sum(future_prediction_scores) / len(future_prediction_scores)
    
    # 3. Goal representation detection
    # Look for persistent representations across sequence
    if seq_len > 1:
        # Compute variance across sequence dimension
        temporal_variance = activations.var(dim=1).mean()
        
        # Low variance might indicate persistent goal representation
        max_possible_variance = activations.var().item()
        if max_possible_variance > 0:
            goal_persistence = 1.0 - (temporal_variance / max_possible_variance)
            results['goal_representation'] = float(goal_persistence)
    
    # Overall planning score
    planning_components = [
        results['sequential_dependency'],
        results['future_prediction'],
        results['goal_representation']
    ]
    results['planning_score'] = sum(planning_components) / len(planning_components)
    
    return results


def compute_mutual_information(x: torch.Tensor, y: torch.Tensor, bins: int = 20) -> float:
    """
    Compute mutual information between two activation tensors.
    
    Args:
        x, y: Activation tensors
        bins: Number of bins for discretization
        
    Returns:
        Mutual information estimate
    """
    try:
        # Flatten tensors
        x_flat = x.flatten().cpu().numpy()
        y_flat = y.flatten().cpu().numpy()
        
        # Discretize
        x_discrete = np.digitize(x_flat, np.linspace(x_flat.min(), x_flat.max(), bins))
        y_discrete = np.digitize(y_flat, np.linspace(y_flat.min(), y_flat.max(), bins))
        
        # Compute joint histogram
        joint_hist, _, _ = np.histogram2d(x_discrete, y_discrete, bins=bins)
        joint_hist = joint_hist / joint_hist.sum()
        
        # Marginal histograms
        x_hist = joint_hist.sum(axis=1)
        y_hist = joint_hist.sum(axis=0)
        
        # Mutual information
        mi = 0.0
        for i in range(bins):
            for j in range(bins):
                if joint_hist[i, j] > 0 and x_hist[i] > 0 and y_hist[j] > 0:
                    mi += joint_hist[i, j] * np.log(
                        joint_hist[i, j] / (x_hist[i] * y_hist[j])
                    )
        
        return float(mi)
    
    except Exception as e:
        logger.warning(f"Mutual information computation failed: {e}")
        return 0.0


def analyze_optimization_circuits(
    activations: Dict[str, torch.Tensor],
    threshold: float = 0.7
) -> Dict[str, Any]:
    """
    Analyze activations for optimization-like computational patterns.
    
    Args:
        activations: Dictionary of layer activations
        threshold: Threshold for circuit detection
        
    Returns:
        Analysis results for optimization circuits
    """
    results = {
        'optimization_score': 0.0,
        'gradient_like_patterns': 0.0,
        'iterative_refinement': 0.0,
        'convergence_patterns': 0.0,
        'suspicious_layers': []
    }
    
    layer_scores = []
    
    for layer_name, activation in activations.items():
        layer_result = analyze_single_layer_optimization(activation)
        layer_scores.append(layer_result['optimization_score'])
        
        if layer_result['optimization_score'] > threshold:
            results['suspicious_layers'].append({
                'layer': layer_name,
                'score': layer_result['optimization_score'],
                'patterns': layer_result['detected_patterns']
            })
    
    if layer_scores:
        results['optimization_score'] = sum(layer_scores) / len(layer_scores)
        results['gradient_like_patterns'] = max(layer_scores)  # Take max as indicator
    
    return results


def analyze_single_layer_optimization(activation: torch.Tensor) -> Dict[str, Any]:
    """
    Analyze a single layer's activations for optimization patterns.
    
    Args:
        activation: Single layer activation tensor
        
    Returns:
        Analysis results for the layer
    """
    result = {
        'optimization_score': 0.0,
        'detected_patterns': []
    }
    
    if activation.numel() == 0:
        return result
    
    patterns_detected = 0
    
    # 1. Check for gradient-like patterns (alternating positive/negative)
    if activation.dim() >= 2:
        # Look for alternating patterns in hidden dimensions
        diffs = torch.diff(activation, dim=-1)
        sign_changes = (diffs[..., :-1] * diffs[..., 1:] < 0).float().mean()
        
        if sign_changes > 0.3:  # High frequency of sign changes
            patterns_detected += 1
            result['detected_patterns'].append('gradient_like')
    
    # 2. Check for iterative refinement patterns
    if activation.dim() >= 3:  # Assumes sequence dimension
        # Look for convergence patterns across sequence
        seq_diffs = torch.diff(activation, dim=1).abs().mean(dim=(0, 2))
        
        # Decreasing differences might indicate convergence
        if len(seq_diffs) > 1:
            convergence_trend = (seq_diffs[1:] < seq_diffs[:-1]).float().mean()
            if convergence_trend > 0.6:
                patterns_detected += 1
                result['detected_patterns'].append('convergence')
    
    # 3. Check for optimization step-like patterns
    activation_ranges = activation.max(dim=-1)[0] - activation.min(dim=-1)[0]
    if (activation_ranges < 0.1).float().mean() > 0.5:  # Many dimensions with small ranges
        patterns_detected += 1
        result['detected_patterns'].append('constrained_optimization')
    
    # Compute overall score
    result['optimization_score'] = min(patterns_detected / 3.0, 1.0)
    
    return result


import numpy as np 