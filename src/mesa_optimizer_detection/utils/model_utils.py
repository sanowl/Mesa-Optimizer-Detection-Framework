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
import warnings
import weakref

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
        if not isinstance(model, nn.Module):
            raise TypeError("model must be a torch.nn.Module instance")
        
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = tokenizer
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Store for activation extraction - use weak references to avoid memory leaks
        self.activation_hooks = {}
        self.stored_activations = {}
        self._hook_handles = {}
        
        logger.info(f"ModelWrapper initialized with device: {self.device}")
    
    def __del__(self):
        """Ensure hooks are cleaned up when wrapper is destroyed."""
        try:
            self.remove_hooks()
        except Exception:
            pass  # Ignore errors during cleanup
    
    def forward(self, inputs: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass through the model."""
        if not isinstance(inputs, torch.Tensor):
            raise TypeError("inputs must be a torch.Tensor")
        
        inputs = inputs.to(self.device)
        
        try:
            return self.model(inputs, **kwargs)
        except Exception as e:
            logger.error(f"Model forward pass failed: {e}")
            raise
    
    def __call__(self, inputs: torch.Tensor, **kwargs) -> torch.Tensor:
        """Make ModelWrapper callable."""
        return self.forward(inputs, **kwargs)
    
    def register_activation_hook(
        self,
        layer_name: str,
        hook_fn: Optional[Callable] = None
    ) -> bool:
        """
        Register hook to capture activations from specified layer.
        
        Returns:
            True if hook was successfully registered, False otherwise
        """
        if hook_fn is None:
            hook_fn = self._default_activation_hook(layer_name)
        
        # Find the layer by name
        layer = self._get_layer_by_name(layer_name)
        if layer is not None:
            try:
                handle = layer.register_forward_hook(hook_fn)
                self.activation_hooks[layer_name] = hook_fn
                self._hook_handles[layer_name] = handle
                logger.debug(f"Registered hook for layer: {layer_name}")
                return True
            except Exception as e:
                logger.warning(f"Failed to register hook for {layer_name}: {e}")
                return False
        else:
            logger.warning(f"Layer not found: {layer_name}")
            return False
    
    def _default_activation_hook(self, layer_name: str):
        """Default hook function to store activations."""
        def hook(module, input, output):
            try:
                if isinstance(output, torch.Tensor):
                    # Detach and clone to avoid computational graph issues
                    self.stored_activations[layer_name] = output.detach().clone()
                elif isinstance(output, (tuple, list)):
                    # Handle multiple outputs
                    self.stored_activations[layer_name] = [
                        o.detach().clone() if isinstance(o, torch.Tensor) else o 
                        for o in output
                    ]
                else:
                    self.stored_activations[layer_name] = output
            except Exception as e:
                logger.warning(f"Hook failed for layer {layer_name}: {e}")
        return hook
    
    def _get_layer_by_name(self, layer_name: str) -> Optional[nn.Module]:
        """Get layer by name from the model."""
        if not isinstance(layer_name, str):
            logger.warning(f"Layer name must be string, got {type(layer_name)}")
            return None
        
        try:
            for name, module in self.model.named_modules():
                if name == layer_name:
                    return module
        except Exception as e:
            logger.warning(f"Error searching for layer {layer_name}: {e}")
        
        return None
    
    def get_activations(
        self, 
        inputs: Optional[torch.Tensor] = None,
        layer_indices: Optional[List[int]] = None,
        clear_after_get: bool = True
    ) -> Dict[str, torch.Tensor]:
        """Get stored activations or extract new ones."""
        if inputs is not None and layer_indices is not None:
            # Extract activations for specific layers
            return extract_activations(self, inputs, layer_indices)
        
        # Return stored activations
        try:
            activations = {}
            for name, activation in self.stored_activations.items():
                if isinstance(activation, torch.Tensor):
                    activations[name] = activation.clone()
                else:
                    activations[name] = activation
            
            if clear_after_get:
                self.stored_activations.clear()
            
            return activations
        except Exception as e:
            logger.warning(f"Failed to get activations: {e}")
            return {}
    
    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        try:
            for layer_name, handle in self._hook_handles.items():
                try:
                    handle.remove()
                except Exception as e:
                    logger.debug(f"Failed to remove hook for {layer_name}: {e}")
            
            self.activation_hooks.clear()
            self._hook_handles.clear()
            self.stored_activations.clear()
            logger.debug("All activation hooks removed")
        except Exception as e:
            logger.warning(f"Error during hook cleanup: {e}")
    
    def get_layer_names(self) -> List[str]:
        """Get list of all layer names in the model."""
        try:
            return [name for name, _ in self.model.named_modules() if name]  # Filter out empty names
        except Exception as e:
            logger.warning(f"Failed to get layer names: {e}")
            return []
    
    def get_parameter_count(self) -> int:
        """Get total number of parameters in the model."""
        try:
            return sum(p.numel() for p in self.model.parameters())
        except Exception as e:
            logger.warning(f"Failed to count parameters: {e}")
            return 0


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
    if not isinstance(inputs, torch.Tensor):
        raise TypeError("inputs must be a torch.Tensor")
    
    if not isinstance(layer_indices, list):
        raise TypeError("layer_indices must be a list")
    
    if isinstance(model, ModelWrapper):
        model_wrapper = model
    else:
        model_wrapper = ModelWrapper(model)
    
    try:
        # Get all layer names if not provided
        all_layer_names = model_wrapper.get_layer_names()
        
        if not all_layer_names:
            logger.warning("No layer names found in model")
            return {}
        
        if layer_names is None:
            # Use layer indices to get names, with bounds checking
            valid_layer_names = []
            invalid_indices = []
            
            for i in layer_indices:
                if isinstance(i, int) and 0 <= i < len(all_layer_names):
                    layer_name = all_layer_names[i]
                    if layer_name:  # Filter out empty names
                        valid_layer_names.append(layer_name)
                else:
                    invalid_indices.append(i)
            
            if invalid_indices:
                logger.warning(f"Invalid layer indices found: {invalid_indices}. "
                             f"Valid range is 0-{len(all_layer_names)-1}")
            
            layer_names = valid_layer_names
        
        if not layer_names:
            logger.warning("No valid layer names to extract activations from")
            return {}
        
        # Register hooks for specified layers
        registered_hooks = []
        for layer_name in layer_names:
            if model_wrapper.register_activation_hook(layer_name):
                registered_hooks.append(layer_name)
        
        if not registered_hooks:
            logger.warning("Failed to register any hooks")
            return {}
        
        # Forward pass to capture activations
        with torch.no_grad():
            try:
                _ = model_wrapper.forward(inputs)
            except Exception as e:
                logger.error(f"Forward pass failed during activation extraction: {e}")
                return {}
        
        # Get captured activations
        activations = model_wrapper.get_activations()
        
        # Filter to only return activations from successfully registered hooks
        filtered_activations = {
            name: activation for name, activation in activations.items()
            if name in registered_hooks
        }
        
        return filtered_activations
        
    finally:
        # Always clean up hooks
        try:
            model_wrapper.remove_hooks()
        except Exception as e:
            logger.debug(f"Hook cleanup failed: {e}")


def compute_activation_statistics(activations: torch.Tensor) -> Dict[str, float]:
    """
    Compute statistical properties of activations.
    
    Args:
        activations: Activation tensor
        
    Returns:
        Dictionary of activation statistics
    """
    if not isinstance(activations, torch.Tensor):
        raise TypeError("activations must be a torch.Tensor")
    
    if activations.numel() == 0:
        logger.warning("Empty activation tensor provided")
        return {}
    
    try:
        stats = {}
        
        # Handle NaN and Inf values
        if torch.isnan(activations).any() or torch.isinf(activations).any():
            logger.warning("NaN or Inf values detected in activations")
            # Filter out NaN and Inf values for statistics
            valid_activations = activations[torch.isfinite(activations)]
            if valid_activations.numel() == 0:
                return {'error': 'all_invalid_values'}
            activations_for_stats = valid_activations
        else:
            activations_for_stats = activations
        
        # Basic statistics
        stats['mean'] = float(activations_for_stats.mean())
        stats['std'] = float(activations_for_stats.std())
        stats['min'] = float(activations_for_stats.min())
        stats['max'] = float(activations_for_stats.max())
        
        # Sparsity (fraction of near-zero activations)
        threshold = max(1e-6, float(activations_for_stats.std()) * 0.01)
        stats['sparsity'] = float((activations_for_stats.abs() < threshold).float().mean())
        
        # L1 and L2 norms
        stats['l1_norm'] = float(activations_for_stats.abs().sum())
        stats['l2_norm'] = float(activations_for_stats.norm())
        
        # Entropy (for discrete-like activations)
        try:
            # Discretize activations for entropy computation
            if activations_for_stats.numel() > 10:  # Need sufficient data points
                discretized = torch.histc(activations_for_stats.flatten(), bins=min(20, activations_for_stats.numel()//2))
                discretized = discretized / discretized.sum()
                discretized = discretized[discretized > 0]  # Remove zero bins
                if len(discretized) > 1:
                    entropy_val = -torch.sum(discretized * torch.log(discretized + 1e-10))
                    stats['entropy'] = float(entropy_val)
                else:
                    stats['entropy'] = 0.0
            else:
                stats['entropy'] = 0.0
        except Exception as e:
            logger.debug(f"Entropy computation failed: {e}")
            stats['entropy'] = 0.0
        
        # Additional useful statistics
        stats['range'] = stats['max'] - stats['min']
        stats['coefficient_of_variation'] = stats['std'] / (abs(stats['mean']) + 1e-10)
        
        return stats
        
    except Exception as e:
        logger.error(f"Activation statistics computation failed: {e}")
        return {'error': str(e)}


def detect_planning_patterns(activations: torch.Tensor, threshold: float = 0.6) -> Dict[str, Any]:
    """
    Detect planning-like patterns in activations.
    
    Args:
        activations: Activation tensor to analyze
        threshold: Threshold for planning detection
        
    Returns:
        Dictionary containing planning analysis results
    """
    if not isinstance(activations, torch.Tensor):
        raise TypeError("activations must be a torch.Tensor")
    
    if activations.numel() == 0:
        return {'planning_score': 0.0, 'confidence': 0.0, 'patterns': []}
    
    try:
        planning_indicators = []
        patterns_detected = []
        
        # Handle different tensor dimensions
        if activations.dim() < 2:
            # 1D tensor - limited analysis possible
            if activations.numel() > 10:
                # Look for structure in the sequence
                autocorr = torch.corrcoef(torch.stack([activations[:-1], activations[1:]]))
                if not torch.isnan(autocorr).any():
                    planning_indicators.append(float(autocorr[0, 1].abs()))
                    if autocorr[0, 1].abs() > 0.5:
                        patterns_detected.append("sequential_correlation")
        
        elif activations.dim() == 2:
            # 2D tensor - batch x features or sequence x features
            batch_size, feature_size = activations.shape
            
            if batch_size > 1 and feature_size > 1:
                # Analyze cross-batch consistency (could indicate planning)
                batch_similarities = []
                for i in range(batch_size):
                    for j in range(i + 1, batch_size):
                        try:
                            sim = torch.cosine_similarity(
                                activations[i:i+1], activations[j:j+1], dim=1
                            )
                            batch_similarities.append(float(sim.mean()))
                        except Exception:
                            continue
                
                if batch_similarities:
                    avg_similarity = sum(batch_similarities) / len(batch_similarities)
                    planning_indicators.append(avg_similarity)
                    if avg_similarity > 0.7:
                        patterns_detected.append("high_batch_consistency")
                
                # Analyze feature correlation structure
                if feature_size > 5:
                    try:
                        corr_matrix = torch.corrcoef(activations.T)
                        if not torch.isnan(corr_matrix).any():
                            # High off-diagonal correlations might indicate structured planning
                            off_diagonal = corr_matrix[torch.triu(torch.ones_like(corr_matrix), diagonal=1) == 1]
                            high_corr_ratio = (off_diagonal.abs() > 0.6).float().mean()
                            planning_indicators.append(float(high_corr_ratio))
                            if high_corr_ratio > 0.3:
                                patterns_detected.append("structured_correlations")
                    except Exception as e:
                        logger.debug(f"Correlation analysis failed: {e}")
        
        elif activations.dim() >= 3:
            # Higher dimensional tensors - could be sequences or spatial data
            # Flatten to 2D for analysis while preserving some structure
            original_shape = activations.shape
            reshaped = activations.view(original_shape[0], -1)
            
            # Recursive call with 2D version
            planning_result = detect_planning_patterns(reshaped, threshold)
            planning_indicators.extend([planning_result['planning_score']])
            patterns_detected.extend(planning_result['patterns'])
            
            # Additional analysis for temporal patterns if first dim is sequence
            if original_shape[0] > 3:
                try:
                    # Look for temporal consistency
                    temporal_diffs = []
                    for t in range(1, original_shape[0]):
                        diff = torch.norm(activations[t] - activations[t-1])
                        temporal_diffs.append(float(diff))
                    
                    if temporal_diffs:
                        diff_std = torch.std(torch.tensor(temporal_diffs))
                        if diff_std < 0.5:  # Low variation indicates consistency
                            planning_indicators.append(0.8)
                            patterns_detected.append("temporal_consistency")
                except Exception as e:
                    logger.debug(f"Temporal analysis failed: {e}")
        
        # Compute overall planning score
        if planning_indicators:
            planning_score = float(torch.mean(torch.tensor(planning_indicators)))
        else:
            planning_score = 0.0
        
        # Compute confidence based on amount of evidence
        confidence = min(len(planning_indicators) / 3.0, 1.0)
        
        result = {
            'planning_score': max(0.0, min(1.0, planning_score)),
            'confidence': confidence,
            'patterns': patterns_detected,
            'num_indicators': len(planning_indicators)
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Planning pattern detection failed: {e}")
        return {'planning_score': 0.0, 'confidence': 0.0, 'patterns': [], 'error': str(e)}


def compute_mutual_information(x: torch.Tensor, y: torch.Tensor, bins: int = 20) -> float:
    """
    Compute mutual information between two tensors.
    
    Args:
        x: First tensor
        y: Second tensor  
        bins: Number of bins for discretization
        
    Returns:
        Mutual information estimate
    """
    if not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor):
        raise TypeError("Both x and y must be torch.Tensors")
    
    if x.numel() == 0 or y.numel() == 0:
        return 0.0
    
    try:
        # Flatten tensors
        x_flat = x.flatten()
        y_flat = y.flatten()
        
        # Handle size mismatch
        min_size = min(x_flat.numel(), y_flat.numel())
        if min_size == 0:
            return 0.0
        
        x_flat = x_flat[:min_size]
        y_flat = y_flat[:min_size]
        
        # Handle constant values
        if torch.std(x_flat) < 1e-10 or torch.std(y_flat) < 1e-10:
            return 0.0
        
        # Discretize using histogram
        x_min, x_max = torch.min(x_flat), torch.max(x_flat)
        y_min, y_max = torch.min(y_flat), torch.max(y_flat)
        
        # Add small epsilon to avoid edge cases
        eps = 1e-8
        x_edges = torch.linspace(x_min - eps, x_max + eps, bins + 1)
        y_edges = torch.linspace(y_min - eps, y_max + eps, bins + 1)
        
        # Compute joint histogram
        joint_hist = torch.zeros(bins, bins)
        
        for i in range(min_size):
            x_bin = torch.searchsorted(x_edges, x_flat[i]) - 1
            y_bin = torch.searchsorted(y_edges, y_flat[i]) - 1
            
            # Clamp to valid range
            x_bin = torch.clamp(x_bin, 0, bins - 1)
            y_bin = torch.clamp(y_bin, 0, bins - 1)
            
            joint_hist[x_bin, y_bin] += 1
        
        # Normalize to get probabilities
        joint_prob = joint_hist / joint_hist.sum()
        
        # Compute marginal probabilities
        x_prob = joint_prob.sum(dim=1)
        y_prob = joint_prob.sum(dim=0)
        
        # Compute mutual information
        mi = 0.0
        for i in range(bins):
            for j in range(bins):
                if joint_prob[i, j] > 0:
                    mi += joint_prob[i, j] * torch.log(
                        joint_prob[i, j] / (x_prob[i] * y_prob[j] + 1e-10) + 1e-10
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
    Analyze activations for optimization-like computational circuits.
    
    Args:
        activations: Dictionary of layer activations
        threshold: Threshold for circuit detection
        
    Returns:
        Dictionary containing circuit analysis results
    """
    if not isinstance(activations, dict):
        raise TypeError("activations must be a dictionary")
    
    if not activations:
        return {'optimization_score': 0.0, 'circuits_detected': [], 'confidence': 0.0}
    
    try:
        circuit_scores = []
        circuits_detected = []
        
        # Analyze each layer for optimization patterns
        for layer_name, activation in activations.items():
            if not isinstance(activation, torch.Tensor) or activation.numel() == 0:
                continue
            
            try:
                layer_result = analyze_single_layer_optimization(activation)
                if layer_result['optimization_score'] > threshold:
                    circuit_scores.append(layer_result['optimization_score'])
                    circuits_detected.append({
                        'layer': layer_name,
                        'score': layer_result['optimization_score'],
                        'patterns': layer_result.get('patterns', [])
                    })
            except Exception as e:
                logger.debug(f"Analysis failed for layer {layer_name}: {e}")
                continue
        
        # Cross-layer analysis for connected circuits
        if len(activations) > 1:
            try:
                layer_names = list(activations.keys())
                layer_tensors = [activations[name] for name in layer_names if isinstance(activations[name], torch.Tensor)]
                
                if len(layer_tensors) > 1:
                    # Compute cross-layer correlations
                    cross_correlations = []
                    for i in range(len(layer_tensors)):
                        for j in range(i + 1, len(layer_tensors)):
                            try:
                                mi = compute_mutual_information(layer_tensors[i], layer_tensors[j])
                                cross_correlations.append(mi)
                            except Exception:
                                continue
                    
                    if cross_correlations:
                        avg_correlation = sum(cross_correlations) / len(cross_correlations)
                        if avg_correlation > 0.3:
                            circuit_scores.append(avg_correlation)
                            circuits_detected.append({
                                'type': 'cross_layer_circuit',
                                'score': avg_correlation,
                                'layers': layer_names
                            })
            except Exception as e:
                logger.debug(f"Cross-layer analysis failed: {e}")
        
        # Compute overall scores
        if circuit_scores:
            optimization_score = float(torch.mean(torch.tensor(circuit_scores)))
        else:
            optimization_score = 0.0
        
        confidence = min(len(circuit_scores) / 3.0, 1.0)
        
        result = {
            'optimization_score': max(0.0, min(1.0, optimization_score)),
            'circuits_detected': circuits_detected,
            'confidence': confidence,
            'num_circuits': len(circuits_detected)
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Optimization circuit analysis failed: {e}")
        return {'optimization_score': 0.0, 'circuits_detected': [], 'confidence': 0.0, 'error': str(e)}


def analyze_single_layer_optimization(activation: torch.Tensor) -> Dict[str, Any]:
    """
    Analyze a single layer's activations for optimization patterns.
    
    Args:
        activation: Activation tensor from a single layer
        
    Returns:
        Dictionary containing single layer analysis results
    """
    if not isinstance(activation, torch.Tensor):
        raise TypeError("activation must be a torch.Tensor")
    
    if activation.numel() == 0:
        return {'optimization_score': 0.0, 'patterns': []}
    
    try:
        optimization_indicators = []
        patterns = []
        
        # 1. Sparsity analysis - optimizers often produce sparse representations
        sparsity = (activation.abs() < 1e-6).float().mean()
        if 0.3 < sparsity < 0.9:  # Sweet spot for optimization sparsity
            optimization_indicators.append(0.7)
            patterns.append("optimal_sparsity")
        
        # 2. Value distribution analysis
        if activation.numel() > 10:
            try:
                # Check for multi-modal distribution (common in optimization)
                hist = torch.histc(activation.flatten(), bins=10)
                hist_normalized = hist / hist.sum()
                
                # Count significant peaks
                peaks = 0
                for i in range(1, len(hist_normalized) - 1):
                    if (hist_normalized[i] > hist_normalized[i-1] and 
                        hist_normalized[i] > hist_normalized[i+1] and 
                        hist_normalized[i] > 0.05):
                        peaks += 1
                
                if peaks >= 2:
                    optimization_indicators.append(0.6)
                    patterns.append("multi_modal_distribution")
            except Exception:
                pass
        
        # 3. Gradient-like patterns (if activation has spatial structure)
        if activation.dim() >= 2:
            try:
                # Look for gradient-like patterns in 2D activations
                if activation.shape[-1] > 3 and activation.shape[-2] > 3:
                    # Compute local differences
                    diff_x = activation[..., 1:] - activation[..., :-1]
                    diff_y = activation[..., 1:, :] - activation[..., :-1, :]
                    
                    # Check for structured patterns
                    diff_x_var = torch.var(diff_x)
                    diff_y_var = torch.var(diff_y)
                    
                    if diff_x_var > 0 and diff_y_var > 0:
                        gradient_strength = (diff_x_var + diff_y_var) / 2
                        if gradient_strength > 0.1:
                            optimization_indicators.append(min(float(gradient_strength), 1.0))
                            patterns.append("spatial_gradients")
            except Exception:
                pass
        
        # 4. Optimization trajectory patterns (for sequence-like activations)
        if activation.dim() >= 2 and activation.shape[0] > 5:
            try:
                # Look for convergence patterns in first dimension
                norms = torch.norm(activation, dim=tuple(range(1, activation.dim())))
                
                if len(norms) > 3:
                    # Check for decreasing trend (optimization convergence)
                    diffs = norms[1:] - norms[:-1]
                    decreasing_ratio = (diffs < 0).float().mean()
                    
                    if decreasing_ratio > 0.6:
                        optimization_indicators.append(float(decreasing_ratio))
                        patterns.append("convergence_pattern")
            except Exception:
                pass
        
        # 5. Information bottleneck patterns
        try:
            # Check for information compression patterns
            if activation.dim() >= 2:
                flat_activation = activation.view(activation.shape[0], -1)
                if flat_activation.shape[1] > 10:
                    # Compute effective rank as proxy for information content
                    try:
                        U, S, V = torch.svd(flat_activation)
                        total_variance = torch.sum(S)
                        if total_variance > 0:
                            normalized_S = S / total_variance
                            entropy = -torch.sum(normalized_S * torch.log(normalized_S + 1e-10))
                            max_entropy = torch.log(torch.tensor(float(len(S))))
                            
                            if max_entropy > 0:
                                information_ratio = entropy / max_entropy
                                if 0.3 < information_ratio < 0.8:  # Moderate compression
                                    optimization_indicators.append(float(information_ratio))
                                    patterns.append("information_bottleneck")
                    except Exception:
                        pass
        except Exception:
            pass
        
        # Compute final score
        if optimization_indicators:
            optimization_score = float(torch.mean(torch.tensor(optimization_indicators)))
        else:
            optimization_score = 0.0
        
        return {
            'optimization_score': max(0.0, min(1.0, optimization_score)),
            'patterns': patterns,
            'num_indicators': len(optimization_indicators)
        }
        
    except Exception as e:
        logger.warning(f"Single layer optimization analysis failed: {e}")
        return {'optimization_score': 0.0, 'patterns': [], 'error': str(e)} 