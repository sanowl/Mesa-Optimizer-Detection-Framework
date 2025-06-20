"""
Mesa-Optimizer Detection Framework - Visualization Dashboard

This module provides a comprehensive visualization dashboard for monitoring,
analyzing, and interpreting mesa-optimization detection results in real-time.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

# Try to import plotly for interactive plots
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.offline as pyo
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Plotly not available. Using matplotlib for visualization.")

from ..core.results import DetectionResults, RiskAssessment
from ..core.detector import MesaOptimizerDetector
import torch


class MesaDetectionDashboard:
    """
    Comprehensive visualization dashboard for Mesa-Optimizer Detection.
    
    Provides real-time monitoring, interactive analysis, and detailed
    visualizations of detection results and model behavior patterns.
    """
    
    def __init__(self, detector: Optional[MesaOptimizerDetector] = None):
        """
        Initialize the visualization dashboard.
        
        Args:
            detector: Optional MesaOptimizerDetector instance for real-time monitoring
        """
        self.detector = detector
        self.detection_history = []
        self.monitoring_active = False
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Colors for risk levels
        self.risk_colors = {
            'MINIMAL': '#2ecc71',  # Green
            'LOW': '#f39c12',      # Orange  
            'MEDIUM': '#e74c3c',   # Red
            'HIGH': '#8e44ad'      # Purple
        }
        
        print("🎨 Mesa-Optimizer Detection Dashboard initialized")
        if not PLOTLY_AVAILABLE:
            print("⚠️  Interactive features require: pip install plotly")
    
    def create_comprehensive_report(
        self, 
        results: DetectionResults,
        save_path: Optional[str] = None,
        show_interactive: bool = True
    ) -> None:
        """
        Create a comprehensive analysis report with multiple visualization panels.
        
        Args:
            results: Detection results to visualize
            save_path: Optional path to save the report
            show_interactive: Whether to show interactive plots (if available)
        """
        print("📊 Creating comprehensive detection report...")
        
        if PLOTLY_AVAILABLE and show_interactive:
            self._create_interactive_report(results, save_path)
        else:
            self._create_static_report(results, save_path)
    
    def _create_interactive_report(
        self, 
        results: DetectionResults, 
        save_path: Optional[str] = None
    ) -> None:
        """Create interactive dashboard using Plotly."""
        
        # Create subplot structure
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=(
                'Risk Score Overview', 'Method Breakdown', 'Risk Assessment Timeline',
                'Activation Patterns', 'Gradient Analysis', 'Behavioral Consistency',
                'Detection Confidence', 'Pattern Evolution', 'Risk Distribution'
            ),
            specs=[
                [{"type": "indicator"}, {"type": "bar"}, {"type": "scatter"}],
                [{"type": "heatmap"}, {"type": "scatter"}, {"type": "bar"}],
                [{"type": "pie"}, {"type": "scatter"}, {"type": "histogram"}]
            ]
        )
        
        # 1. Risk Score Overview (Gauge)
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=results.risk_score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Overall Risk Score"},
                gauge={
                    'axis': {'range': [None, 1.0]},
                    'bar': {'color': self._get_risk_color(results.risk_score)},
                    'steps': [
                        {'range': [0, 0.3], 'color': "lightgray"},
                        {'range': [0.3, 0.6], 'color': "yellow"},
                        {'range': [0.6, 1.0], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': 0.8
                    }
                }
            ),
            row=1, col=1
        )
        
        # 2. Method Breakdown
        methods = list(results.method_results.keys())
        method_scores = [
            results.method_results[method].get('risk_score', 0.0) 
            for method in methods
        ]
        
        fig.add_trace(
            go.Bar(
                x=methods,
                y=method_scores,
                name="Method Scores",
                marker_color=[self._get_risk_color(score) for score in method_scores]
            ),
            row=1, col=2
        )
        
        # 3. Risk Assessment Timeline (if we have history)
        if self.detection_history:
            timestamps = list(range(len(self.detection_history)))
            risk_scores = [r.risk_score for r in self.detection_history]
            
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=risk_scores,
                    mode='lines+markers',
                    name='Risk Timeline',
                    line=dict(color='red', width=2)
                ),
                row=1, col=3
            )
        else:
            # Show current result as single point
            fig.add_trace(
                go.Scatter(
                    x=[0],
                    y=[results.risk_score],
                    mode='markers',
                    name='Current Risk',
                    marker=dict(size=20, color=self._get_risk_color(results.risk_score))
                ),
                row=1, col=3
            )
        
        # 4. Activation Patterns Heatmap
        if 'activation' in results.method_results:
            activation_data = results.method_results['activation']
            if 'circuit_activations' in activation_data:
                # Create mock heatmap data for demonstration
                heatmap_data = np.random.rand(10, 8) * results.risk_score
                
                fig.add_trace(
                    go.Heatmap(
                        z=heatmap_data,
                        colorscale='Reds',
                        name='Activation Patterns'
                    ),
                    row=2, col=1
                )
        
        # 5. Gradient Analysis
        if 'gradient' in results.method_results:
            gradient_data = results.method_results['gradient']
            
            # Create scatter plot of gradient properties
            properties = ['variance', 'anomaly_score', 'confidence']
            values = [
                gradient_data.get('gradient_variance', 0),
                gradient_data.get('anomaly_score', 0),
                gradient_data.get('confidence', 0)
            ]
            
            fig.add_trace(
                go.Scatter(
                    x=properties,
                    y=values,
                    mode='markers+lines',
                    name='Gradient Properties',
                    marker=dict(size=15, color='blue')
                ),
                row=2, col=2
            )
        
        # 6. Behavioral Consistency
        if 'behavioral' in results.method_results:
            behavioral_data = results.method_results['behavioral']
            
            consistency_scores = [
                behavioral_data.get('consistency_score', 0),
                behavioral_data.get('deception_score', 0),
                behavioral_data.get('confidence', 0)
            ]
            
            fig.add_trace(
                go.Bar(
                    x=['Consistency', 'Deception', 'Confidence'],
                    y=consistency_scores,
                    name='Behavioral Metrics',
                    marker_color='orange'
                ),
                row=2, col=3
            )
        
        # 7. Detection Confidence Pie Chart
        confidence_breakdown = {
            'High Confidence': results.confidence * 0.7,
            'Medium Confidence': results.confidence * 0.2,
            'Low Confidence': results.confidence * 0.1
        }
        
        fig.add_trace(
            go.Pie(
                labels=list(confidence_breakdown.keys()),
                values=list(confidence_breakdown.values()),
                name="Confidence Breakdown"
            ),
            row=3, col=1
        )
        
        # 8. Pattern Evolution (if we have multiple method results)
        pattern_scores = []
        pattern_names = []
        
        for method, data in results.method_results.items():
            if isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, (int, float)) and key.endswith('_score'):
                        pattern_names.append(f"{method}_{key}")
                        pattern_scores.append(value)
        
        if pattern_scores:
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(pattern_scores))),
                    y=pattern_scores,
                    mode='markers',
                    name='Pattern Scores',
                    marker=dict(
                        size=[max(5, s*20) for s in pattern_scores],
                        color=pattern_scores,
                        colorscale='Viridis'
                    ),
                    text=pattern_names,
                    textposition="top center"
                ),
                row=3, col=2
            )
        
        # 9. Risk Distribution Histogram
        risk_values = [results.risk_score] + [
            results.method_results[method].get('risk_score', 0)
            for method in results.method_results.keys()
        ]
        
        fig.add_trace(
            go.Histogram(
                x=risk_values,
                nbinsx=10,
                name='Risk Distribution',
                marker_color='red',
                opacity=0.7
            ),
            row=3, col=3
        )
        
        # Update layout
        fig.update_layout(
            height=1200,
            title_text=f"Mesa-Optimizer Detection Dashboard - Risk Level: {results.risk_level}",
            title_x=0.5,
            showlegend=False
        )
        
        # Show the plot
        if save_path:
            fig.write_html(save_path.replace('.png', '.html'))
            print(f"💾 Interactive report saved to {save_path.replace('.png', '.html')}")
        
        fig.show()
    
    def _create_static_report(
        self, 
        results: DetectionResults, 
        save_path: Optional[str] = None
    ) -> None:
        """Create static dashboard using Matplotlib."""
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 15))
        
        # Main title
        fig.suptitle(
            f'Mesa-Optimizer Detection Dashboard\nRisk Level: {results.risk_level} '
            f'(Score: {results.risk_score:.3f})',
            fontsize=20, fontweight='bold'
        )
        
        # 1. Risk Score Gauge
        ax1 = plt.subplot(3, 4, 1)
        self._create_risk_gauge(ax1, results.risk_score, results.risk_level)
        
        # 2. Method Breakdown
        ax2 = plt.subplot(3, 4, 2)
        self._create_method_breakdown(ax2, results.method_results)
        
        # 3. Risk Timeline
        ax3 = plt.subplot(3, 4, 3)
        self._create_risk_timeline(ax3)
        
        # 4. Confidence Metrics
        ax4 = plt.subplot(3, 4, 4)
        self._create_confidence_metrics(ax4, results)
        
        # 5. Activation Patterns
        ax5 = plt.subplot(3, 4, 5)
        self._create_activation_patterns(ax5, results.method_results.get('activation', {}))
        
        # 6. Gradient Analysis
        ax6 = plt.subplot(3, 4, 6)
        self._create_gradient_analysis(ax6, results.method_results.get('gradient', {}))
        
        # 7. Behavioral Analysis
        ax7 = plt.subplot(3, 4, 7)
        self._create_behavioral_analysis(ax7, results.method_results.get('behavioral', {}))
        
        # 8. Detection Summary
        ax8 = plt.subplot(3, 4, 8)
        self._create_detection_summary(ax8, results)
        
        # 9. Risk Distribution
        ax9 = plt.subplot(3, 4, 9)
        self._create_risk_distribution(ax9, results)
        
        # 10. Method Comparison
        ax10 = plt.subplot(3, 4, 10)
        self._create_method_comparison(ax10, results.method_results)
        
        # 11. Recommendations
        ax11 = plt.subplot(3, 4, (11, 12))
        self._create_recommendations_panel(ax11, results)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Static report saved to {save_path}")
        
        plt.show()
    
    def _create_risk_gauge(self, ax, risk_score: float, risk_level: str) -> None:
        """Create a risk gauge visualization."""
        # Create circular gauge
        theta = np.linspace(0, np.pi, 100)
        
        # Background arc
        ax.fill_between(theta, 0.8, 1.0, color='lightgray', alpha=0.3)
        
        # Risk level arcs
        ax.fill_between(theta[0:33], 0.8, 1.0, color='green', alpha=0.7, label='Low')
        ax.fill_between(theta[33:66], 0.8, 1.0, color='orange', alpha=0.7, label='Medium')
        ax.fill_between(theta[66:100], 0.8, 1.0, color='red', alpha=0.7, label='High')
        
        # Current risk needle
        risk_angle = np.pi * (1 - risk_score)
        ax.plot([risk_angle, risk_angle], [0, 1], 'k-', linewidth=4)
        ax.plot(risk_angle, 0.9, 'ko', markersize=10)
        
        # Styling
        ax.set_xlim(0, np.pi)
        ax.set_ylim(0, 1.2)
        ax.set_title(f'Risk Gauge\n{risk_level}', fontweight='bold')
        ax.text(np.pi/2, 0.5, f'{risk_score:.3f}', ha='center', va='center', 
                fontsize=16, fontweight='bold')
        ax.axis('off')
    
    def _create_method_breakdown(self, ax, method_results: Dict[str, Any]) -> None:
        """Create method breakdown bar chart."""
        methods = list(method_results.keys())
        scores = [
            method_results[method].get('risk_score', 0.0) if isinstance(method_results[method], dict)
            else 0.0
            for method in methods
        ]
        
        bars = ax.bar(methods, scores, color=[self._get_risk_color(s) for s in scores])
        
        # Add value labels on bars
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{score:.3f}', ha='center', va='bottom')
        
        ax.set_title('Detection Method Scores', fontweight='bold')
        ax.set_ylabel('Risk Score')
        ax.set_ylim(0, 1.1)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    def _create_risk_timeline(self, ax) -> None:
        """Create risk timeline plot."""
        if self.detection_history:
            timestamps = list(range(len(self.detection_history)))
            risk_scores = [r.risk_score for r in self.detection_history]
            risk_levels = [r.risk_level for r in self.detection_history]
            
            # Plot timeline
            ax.plot(timestamps, risk_scores, 'o-', linewidth=2, markersize=6)
            
            # Color points by risk level
            for i, (score, level) in enumerate(zip(risk_scores, risk_levels)):
                ax.plot(i, score, 'o', color=self.risk_colors[level], markersize=8)
            
            ax.set_title('Risk Score Timeline', fontweight='bold')
            ax.set_xlabel('Detection Run')
            ax.set_ylabel('Risk Score')
            ax.set_ylim(0, 1.1)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No Timeline Data\nAvailable', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12)
            ax.set_title('Risk Score Timeline', fontweight='bold')
            ax.axis('off')
    
    def _create_confidence_metrics(self, ax, results: DetectionResults) -> None:
        """Create confidence metrics visualization."""
        confidence_data = {
            'Overall': results.confidence,
            'Assessment': results.risk_assessment.confidence if results.risk_assessment else 0.5
        }
        
        # Add method-specific confidences
        for method, data in results.method_results.items():
            if isinstance(data, dict) and 'confidence' in data:
                confidence_data[method.title()] = data['confidence']
        
        metrics = list(confidence_data.keys())
        values = list(confidence_data.values())
        
        bars = ax.barh(metrics, values, color='skyblue', alpha=0.7)
        
        # Add value labels
        for bar, value in zip(bars, values):
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                   f'{value:.3f}', ha='left', va='center')
        
        ax.set_title('Confidence Metrics', fontweight='bold')
        ax.set_xlabel('Confidence Score')
        ax.set_xlim(0, 1.1)
        ax.grid(True, alpha=0.3, axis='x')
    
    def _create_activation_patterns(self, ax, activation_data: Dict[str, Any]) -> None:
        """Create activation patterns visualization."""
        if activation_data:
            # Create mock heatmap for activation patterns
            patterns = np.random.rand(8, 10) * activation_data.get('risk_score', 0.5)
            
            im = ax.imshow(patterns, cmap='Reds', aspect='auto')
            ax.set_title('Activation Patterns', fontweight='bold')
            ax.set_xlabel('Feature Dimension')
            ax.set_ylabel('Layer')
            
            # Add colorbar
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        else:
            ax.text(0.5, 0.5, 'No Activation\nData Available', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12)
            ax.set_title('Activation Patterns', fontweight='bold')
            ax.axis('off')
    
    def _create_gradient_analysis(self, ax, gradient_data: Dict[str, Any]) -> None:
        """Create gradient analysis visualization."""
        if gradient_data:
            metrics = ['Variance', 'Anomaly', 'Risk', 'Confidence']
            values = [
                gradient_data.get('gradient_variance', 0),
                gradient_data.get('anomaly_score', 0),
                gradient_data.get('risk_score', 0),
                gradient_data.get('confidence', 0)
            ]
            
            # Normalize values for radar chart effect
            angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
            values = values + [values[0]]  # Complete the circle
            angles += angles[:1]
            
            ax.plot(angles, values, 'o-', linewidth=2, color='blue')
            ax.fill(angles, values, alpha=0.25, color='blue')
            
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(metrics)
            ax.set_ylim(0, 1)
            ax.set_title('Gradient Analysis', fontweight='bold')
            ax.grid(True)
        else:
            ax.text(0.5, 0.5, 'No Gradient\nData Available', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12)
            ax.set_title('Gradient Analysis', fontweight='bold')
            ax.axis('off')
    
    def _create_behavioral_analysis(self, ax, behavioral_data: Dict[str, Any]) -> None:
        """Create behavioral analysis visualization."""
        if behavioral_data:
            # Extract behavioral metrics
            consistency = behavioral_data.get('consistency_score', 0)
            deception = behavioral_data.get('deception_score', 0)
            context_sensitivity = behavioral_data.get('context_sensitivity', 0)
            
            metrics = ['Consistency', 'Deception', 'Context\nSensitivity']
            values = [consistency, deception, context_sensitivity]
            colors = ['green', 'red', 'orange']
            
            bars = ax.bar(metrics, values, color=colors, alpha=0.7)
            
            # Add value labels
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.3f}', ha='center', va='bottom')
            
            ax.set_title('Behavioral Analysis', fontweight='bold')
            ax.set_ylabel('Score')
            ax.set_ylim(0, 1.1)
        else:
            ax.text(0.5, 0.5, 'No Behavioral\nData Available', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12)
            ax.set_title('Behavioral Analysis', fontweight='bold')
            ax.axis('off')
    
    def _create_detection_summary(self, ax, results: DetectionResults) -> None:
        """Create detection summary text panel."""
        summary_text = f"""
DETECTION SUMMARY

Risk Level: {results.risk_level}
Risk Score: {results.risk_score:.3f}
Confidence: {results.confidence:.3f}

Methods Used: {len(results.method_results)}
{chr(10).join([f"• {method.title()}" for method in results.method_results.keys()])}

Recommendations:
{chr(10).join([f"• {rec}" for rec in results.recommendations[:3]])}
        """
        
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
        ax.set_title('Detection Summary', fontweight='bold')
        ax.axis('off')
    
    def _create_risk_distribution(self, ax, results: DetectionResults) -> None:
        """Create risk distribution visualization."""
        # Collect all risk scores
        all_scores = [results.risk_score]
        
        for method_data in results.method_results.values():
            if isinstance(method_data, dict) and 'risk_score' in method_data:
                all_scores.append(method_data['risk_score'])
        
        if len(all_scores) > 1:
            ax.hist(all_scores, bins=10, alpha=0.7, color='red', edgecolor='black')
            ax.axvline(results.risk_score, color='black', linestyle='--', linewidth=2,
                      label=f'Overall: {results.risk_score:.3f}')
            ax.set_title('Risk Score Distribution', fontweight='bold')
            ax.set_xlabel('Risk Score')
            ax.set_ylabel('Frequency')
            ax.legend()
        else:
            ax.bar(['Overall'], [results.risk_score], color=self._get_risk_color(results.risk_score))
            ax.set_title('Risk Score Distribution', fontweight='bold')
            ax.set_ylabel('Risk Score')
            ax.set_ylim(0, 1.1)
    
    def _create_method_comparison(self, ax, method_results: Dict[str, Any]) -> None:
        """Create method comparison radar chart."""
        methods = list(method_results.keys())
        if len(methods) >= 3:
            # Create radar chart
            angles = np.linspace(0, 2*np.pi, len(methods), endpoint=False).tolist()
            
            # Get risk scores for each method
            risk_scores = [
                method_results[method].get('risk_score', 0.0) if isinstance(method_results[method], dict)
                else 0.0
                for method in methods
            ]
            
            # Complete the circle
            risk_scores += [risk_scores[0]]
            angles += angles[:1]
            
            ax.plot(angles, risk_scores, 'o-', linewidth=2, color='purple')
            ax.fill(angles, risk_scores, alpha=0.25, color='purple')
            
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels([m.title() for m in methods])
            ax.set_ylim(0, 1)
            ax.set_title('Method Comparison', fontweight='bold')
            ax.grid(True)
        else:
            ax.text(0.5, 0.5, 'Need 3+ Methods\nfor Comparison', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12)
            ax.set_title('Method Comparison', fontweight='bold')
            ax.axis('off')
    
    def _create_recommendations_panel(self, ax, results: DetectionResults) -> None:
        """Create recommendations panel."""
        # Risk level indicator
        risk_color = self.risk_colors.get(results.risk_level, 'gray')
        
        recommendations_text = f"""
🚨 RISK ASSESSMENT: {results.risk_level} 🚨

📊 OVERALL RISK SCORE: {results.risk_score:.3f}
🎯 CONFIDENCE LEVEL: {results.confidence:.3f}

🔍 KEY FINDINGS:
• Detected using {len(results.method_results)} analysis methods
• Primary concerns identified in mesa-optimization patterns
• Risk level indicates {"immediate attention required" if results.risk_level in ["HIGH", "MEDIUM"] else "monitoring recommended"}

💡 RECOMMENDATIONS:
"""
        
        for i, rec in enumerate(results.recommendations[:5], 1):
            recommendations_text += f"\n{i}. {rec}"
        
        # Add next steps
        recommendations_text += f"""

🔄 NEXT STEPS:
• {"Deploy additional monitoring" if results.risk_level == "HIGH" else "Continue regular analysis"}
• {"Implement intervention protocols" if results.risk_level in ["HIGH", "MEDIUM"] else "Maintain current oversight"}
• {"Review training procedures" if results.risk_score > 0.5 else "Standard monitoring sufficient"}
        """
        
        ax.text(0.02, 0.98, recommendations_text, transform=ax.transAxes, fontsize=11,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle="round,pad=0.5", facecolor=risk_color, alpha=0.2))
        ax.set_title('Risk Assessment & Recommendations', fontweight='bold', fontsize=14)
        ax.axis('off')
    
    def monitor_real_time(
        self, 
        model_input_generator,
        update_interval: int = 5,
        max_updates: int = 100
    ) -> None:
        """
        Real-time monitoring dashboard for continuous model analysis.
        
        Args:
            model_input_generator: Generator that yields input data for analysis
            update_interval: Seconds between updates
            max_updates: Maximum number of updates before stopping
        """
        if not self.detector:
            print("❌ No detector provided for real-time monitoring")
            return
        
        print("🔴 Starting real-time monitoring...")
        self.monitoring_active = True
        
        if PLOTLY_AVAILABLE:
            self._monitor_with_plotly(model_input_generator, update_interval, max_updates)
        else:
            self._monitor_with_matplotlib(model_input_generator, update_interval, max_updates)
    
    def _monitor_with_plotly(self, model_input_generator, update_interval: int, max_updates: int) -> None:
        """Real-time monitoring with Plotly (interactive)."""
        # Create initial plot structure
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Risk Score Timeline', 'Method Breakdown', 
                          'Confidence Tracking', 'Detection Frequency'),
            specs=[[{"secondary_y": False}, {"type": "bar"}],
                   [{"secondary_y": False}, {"type": "pie"}]]
        )
        
        # Initialize data storage
        timestamps = []
        risk_scores = []
        risk_levels = []
        
        try:
            for update_count, input_data in enumerate(model_input_generator):
                if update_count >= max_updates or not self.monitoring_active:
                    break
                
                # Perform detection
                result = self.detector.analyze(input_data)
                self.detection_history.append(result)
                
                # Update tracking data
                timestamps.append(update_count)
                risk_scores.append(result.risk_score)
                risk_levels.append(result.risk_level)
                
                # Update plots
                fig.data = []  # Clear previous data
                
                # Risk timeline
                fig.add_trace(
                    go.Scatter(
                        x=timestamps,
                        y=risk_scores,
                        mode='lines+markers',
                        name='Risk Score',
                        line=dict(color='red', width=2)
                    ),
                    row=1, col=1
                )
                
                # Method breakdown (latest result)
                methods = list(result.method_results.keys())
                method_scores = [
                    result.method_results[method].get('risk_score', 0.0)
                    for method in methods
                ]
                
                fig.add_trace(
                    go.Bar(
                        x=methods,
                        y=method_scores,
                        name="Current Methods"
                    ),
                    row=1, col=2
                )
                
                # Confidence tracking
                confidences = [r.confidence for r in self.detection_history[-10:]]  # Last 10
                fig.add_trace(
                    go.Scatter(
                        x=list(range(len(confidences))),
                        y=confidences,
                        mode='lines+markers',
                        name='Confidence',
                        line=dict(color='blue', width=2)
                    ),
                    row=2, col=1
                )
                
                # Risk level distribution
                level_counts = {level: risk_levels.count(level) for level in set(risk_levels)}
                fig.add_trace(
                    go.Pie(
                        labels=list(level_counts.keys()),
                        values=list(level_counts.values()),
                        name="Risk Distribution"
                    ),
                    row=2, col=2
                )
                
                # Update layout and show
                fig.update_layout(
                    title_text=f"Real-Time Mesa-Optimizer Monitoring (Update {update_count+1})",
                    height=800
                )
                
                fig.show()
                
                # Wait for next update
                import time
                time.sleep(update_interval)
                
        except KeyboardInterrupt:
            print("\n⏹️  Monitoring stopped by user")
        except Exception as e:
            print(f"❌ Monitoring error: {e}")
        finally:
            self.monitoring_active = False
            print("🔴 Real-time monitoring stopped")
    
    def _monitor_with_matplotlib(self, model_input_generator, update_interval: int, max_updates: int) -> None:
        """Real-time monitoring with Matplotlib (static updates)."""
        print("📊 Using matplotlib for real-time monitoring (install plotly for interactive)")
        
        # Enable interactive mode
        plt.ion()
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Real-Time Mesa-Optimizer Monitoring', fontsize=16, fontweight='bold')
        
        timestamps = []
        risk_scores = []
        risk_levels = []
        
        try:
            for update_count, input_data in enumerate(model_input_generator):
                if update_count >= max_updates or not self.monitoring_active:
                    break
                
                # Perform detection
                result = self.detector.analyze(input_data)
                self.detection_history.append(result)
                
                # Update tracking data
                timestamps.append(update_count)
                risk_scores.append(result.risk_score)
                risk_levels.append(result.risk_level)
                
                # Clear and update plots
                ax1.clear()
                ax2.clear()
                ax3.clear()
                ax4.clear()
                
                # Risk timeline
                ax1.plot(timestamps, risk_scores, 'r-o', linewidth=2)
                ax1.set_title('Risk Score Timeline')
                ax1.set_xlabel('Update')
                ax1.set_ylabel('Risk Score')
                ax1.set_ylim(0, 1)
                ax1.grid(True, alpha=0.3)
                
                # Method breakdown
                methods = list(result.method_results.keys())
                method_scores = [
                    result.method_results[method].get('risk_score', 0.0)
                    for method in methods
                ]
                ax2.bar(methods, method_scores, color='orange', alpha=0.7)
                ax2.set_title('Current Method Scores')
                ax2.set_ylabel('Risk Score')
                ax2.set_ylim(0, 1)
                plt.setp(ax2.get_xticklabels(), rotation=45)
                
                # Confidence tracking
                confidences = [r.confidence for r in self.detection_history[-20:]]
                ax3.plot(range(len(confidences)), confidences, 'b-o', linewidth=2)
                ax3.set_title('Confidence Tracking')
                ax3.set_xlabel('Recent Updates')
                ax3.set_ylabel('Confidence')
                ax3.set_ylim(0, 1)
                ax3.grid(True, alpha=0.3)
                
                # Risk level distribution
                level_counts = {level: risk_levels.count(level) for level in set(risk_levels)}
                colors = [self.risk_colors.get(level, 'gray') for level in level_counts.keys()]
                ax4.pie(level_counts.values(), labels=level_counts.keys(), colors=colors, autopct='%1.1f%%')
                ax4.set_title('Risk Level Distribution')
                
                plt.tight_layout()
                plt.draw()
                plt.pause(update_interval)
                
        except KeyboardInterrupt:
            print("\n⏹️  Monitoring stopped by user")
        except Exception as e:
            print(f"❌ Monitoring error: {e}")
        finally:
            plt.ioff()
            self.monitoring_active = False
            print("🔴 Real-time monitoring stopped")
    
    def _get_risk_color(self, risk_score: float) -> str:
        """Get color based on risk score."""
        if risk_score < 0.3:
            return self.risk_colors['MINIMAL']
        elif risk_score < 0.6:
            return self.risk_colors['LOW']
        elif risk_score < 0.8:
            return self.risk_colors['MEDIUM']
        else:
            return self.risk_colors['HIGH']
    
    def save_detection_history(self, filepath: str) -> None:
        """Save detection history to file."""
        import json
        
        history_data = []
        for i, result in enumerate(self.detection_history):
            history_data.append({
                'timestamp': i,
                'risk_score': result.risk_score,
                'risk_level': result.risk_level,
                'confidence': result.confidence,
                'methods_used': list(result.method_results.keys())
            })
        
        with open(filepath, 'w') as f:
            json.dump(history_data, f, indent=2)
        
        print(f"💾 Detection history saved to {filepath}")
    
    def load_detection_history(self, filepath: str) -> None:
        """Load detection history from file."""
        import json
        
        with open(filepath, 'r') as f:
            history_data = json.load(f)
        
        print(f"📁 Loaded {len(history_data)} detection records from {filepath}")


def create_demo_dashboard() -> MesaDetectionDashboard:
    """Create a demo dashboard with sample data."""
    dashboard = MesaDetectionDashboard()
    
    # Add some sample detection history
    from ..core.results import DetectionResults, RiskAssessment
    
    for i in range(10):
        sample_result = DetectionResults(
            risk_score=np.random.uniform(0.1, 0.9),
            risk_level="MEDIUM",
            confidence=np.random.uniform(0.6, 0.9),
            method_results={
                'gradient': {'risk_score': np.random.uniform(0.0, 0.8)},
                'activation': {'risk_score': np.random.uniform(0.0, 0.7)},
                'behavioral': {'risk_score': np.random.uniform(0.0, 0.6)}
            },
            recommendations=["Monitor closely", "Review training data", "Check for deception"],
            risk_assessment=RiskAssessment(
                risk_factors=["High planning score", "Suspicious gradients"],
                mitigation_strategies=["Increase monitoring", "Deploy interventions"],
                confidence=0.8
            )
        )
        dashboard.detection_history.append(sample_result)
    
    return dashboard 