"""
Mesa-Optimizer Detection Framework - Visualization Dashboard

Comprehensive visualization dashboard for monitoring and analyzing 
mesa-optimization detection results in real-time.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Try to import plotly for interactive plots
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

class MesaDetectionDashboard:
    """Comprehensive visualization dashboard for Mesa-Optimizer Detection."""
    
    def __init__(self):
        """Initialize the visualization dashboard."""
        self.detection_history = []
        self.monitoring_active = False
        
        # Set up plotting style
        plt.style.use('default')
        
        # Colors for risk levels
        self.risk_colors = {
            'MINIMAL': '#2ecc71',  # Green
            'LOW': '#f39c12',      # Orange  
            'MEDIUM': '#e74c3c',   # Red
            'HIGH': '#8e44ad'      # Purple
        }
        
        print("🎨 Mesa-Optimizer Detection Dashboard initialized")
        if not PLOTLY_AVAILABLE:
            print("⚠️  For interactive features, install: pip install plotly")
    
    def create_comprehensive_report(self, results, save_path: Optional[str] = None) -> None:
        """Create a comprehensive analysis report with multiple visualization panels."""
        print("📊 Creating comprehensive detection report...")
        
        if PLOTLY_AVAILABLE:
            self._create_interactive_report(results, save_path)
        else:
            self._create_static_report(results, save_path)
    
    def _create_interactive_report(self, results, save_path: Optional[str] = None) -> None:
        """Create interactive dashboard using Plotly."""
        
        # Create subplot structure
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=(
                'Risk Score Overview', 'Method Breakdown', 'Detection Confidence',
                'Risk Timeline', 'Method Comparison', 'Recommendations'
            ),
            specs=[
                [{"type": "indicator"}, {"type": "bar"}, {"type": "pie"}],
                [{"type": "scatter"}, {"type": "bar"}, {"type": "table"}]
            ]
        )
        
        # 1. Risk Score Overview (Gauge)
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
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
                    ]
                }
            ),
            row=1, col=1
        )
        
        # 2. Method Breakdown
        methods = list(results.method_results.keys())
        method_scores = [
            results.method_results[method].get('risk_score', 0.0) 
            if isinstance(results.method_results[method], dict)
            else 0.0
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
        
        # 3. Detection Confidence Pie Chart
        confidence_levels = {
            'High Confidence': results.confidence * 0.7,
            'Medium Confidence': results.confidence * 0.2,
            'Low Confidence': results.confidence * 0.1
        }
        
        fig.add_trace(
            go.Pie(
                labels=list(confidence_levels.keys()),
                values=list(confidence_levels.values()),
                name="Confidence Breakdown"
            ),
            row=1, col=3
        )
        
        # 4. Risk Timeline
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
                row=2, col=1
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=[0],
                    y=[results.risk_score],
                    mode='markers',
                    name='Current Risk',
                    marker=dict(size=20, color=self._get_risk_color(results.risk_score))
                ),
                row=2, col=1
            )
        
        # 5. Method Comparison
        if len(methods) > 0:
            fig.add_trace(
                go.Bar(
                    x=methods,
                    y=method_scores,
                    name="Method Comparison",
                    marker_color='skyblue'
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            height=800,
            title_text=f"Mesa-Optimizer Detection Dashboard - Risk Level: {results.risk_level}",
            title_x=0.5,
            showlegend=False
        )
        
        # Save and show
        if save_path:
            fig.write_html(save_path.replace('.png', '.html'))
            print(f"💾 Interactive report saved to {save_path.replace('.png', '.html')}")
        
        fig.show()
    
    def _create_static_report(self, results, save_path: Optional[str] = None) -> None:
        """Create static dashboard using Matplotlib."""
        
        # Create figure with subplots
        fig = plt.figure(figsize=(18, 12))
        
        # Main title
        fig.suptitle(
            f'Mesa-Optimizer Detection Dashboard\nRisk Level: {results.risk_level} '
            f'(Score: {results.risk_score:.3f})',
            fontsize=18, fontweight='bold'
        )
        
        # 1. Risk Score Gauge
        ax1 = plt.subplot(2, 4, 1)
        self._create_risk_gauge(ax1, results.risk_score, results.risk_level)
        
        # 2. Method Breakdown
        ax2 = plt.subplot(2, 4, 2)
        self._create_method_breakdown(ax2, results.method_results)
        
        # 3. Risk Timeline
        ax3 = plt.subplot(2, 4, 3)
        self._create_risk_timeline(ax3)
        
        # 4. Confidence Metrics
        ax4 = plt.subplot(2, 4, 4)
        self._create_confidence_metrics(ax4, results)
        
        # 5. Detection Summary
        ax5 = plt.subplot(2, 4, 5)
        self._create_detection_summary(ax5, results)
        
        # 6. Risk Distribution
        ax6 = plt.subplot(2, 4, 6)
        self._create_risk_distribution(ax6, results)
        
        # 7. Method Details
        ax7 = plt.subplot(2, 4, 7)
        self._create_method_details(ax7, results.method_results)
        
        # 8. Recommendations
        ax8 = plt.subplot(2, 4, 8)
        self._create_recommendations_panel(ax8, results)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Static report saved to {save_path}")
        
        plt.show()
    
    def _create_risk_gauge(self, ax, risk_score: float, risk_level: str) -> None:
        """Create a risk gauge visualization."""
        # Create arc gauge
        theta = np.linspace(0, np.pi, 100)
        
        # Background arc
        ax.fill_between(theta, 0.8, 1.0, color='lightgray', alpha=0.3)
        
        # Risk level arcs
        ax.fill_between(theta[0:33], 0.8, 1.0, color='green', alpha=0.5)
        ax.fill_between(theta[33:66], 0.8, 1.0, color='orange', alpha=0.5)
        ax.fill_between(theta[66:100], 0.8, 1.0, color='red', alpha=0.5)
        
        # Current risk needle
        risk_angle = np.pi * (1 - risk_score)
        ax.plot([risk_angle, risk_angle], [0, 1], 'k-', linewidth=4)
        ax.plot(risk_angle, 0.9, 'ko', markersize=8)
        
        # Styling
        ax.set_xlim(0, np.pi)
        ax.set_ylim(0, 1.2)
        ax.set_title(f'Risk Gauge\n{risk_level}', fontweight='bold')
        ax.text(np.pi/2, 0.5, f'{risk_score:.3f}', ha='center', va='center', 
                fontsize=14, fontweight='bold')
        ax.axis('off')
    
    def _create_method_breakdown(self, ax, method_results: Dict[str, Any]) -> None:
        """Create method breakdown bar chart."""
        methods = list(method_results.keys())
        scores = []
        
        for method in methods:
            if isinstance(method_results[method], dict):
                score = method_results[method].get('risk_score', 0.0)
            else:
                score = 0.0
            scores.append(score)
        
        bars = ax.bar(methods, scores, color=[self._get_risk_color(s) for s in scores])
        
        # Add value labels on bars
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{score:.3f}', ha='center', va='bottom', fontsize=9)
        
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
                ax.plot(i, score, 'o', color=self.risk_colors.get(level, 'gray'), markersize=8)
            
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
    
    def _create_confidence_metrics(self, ax, results) -> None:
        """Create confidence metrics visualization."""
        confidence_data = {'Overall': results.confidence}
        
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
    
    def _create_detection_summary(self, ax, results) -> None:
        """Create detection summary text panel."""
        summary_text = f"""
DETECTION SUMMARY

Risk Level: {results.risk_level}
Risk Score: {results.risk_score:.3f}
Confidence: {results.confidence:.3f}

Methods Used: {len(results.method_results)}

Top Recommendations:
{chr(10).join([f"• {rec}" for rec in results.recommendations[:3]])}
        """
        
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=9,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
        ax.set_title('Detection Summary', fontweight='bold')
        ax.axis('off')
    
    def _create_risk_distribution(self, ax, results) -> None:
        """Create risk distribution visualization."""
        # Collect all risk scores
        all_scores = [results.risk_score]
        
        for method_data in results.method_results.values():
            if isinstance(method_data, dict) and 'risk_score' in method_data:
                all_scores.append(method_data['risk_score'])
        
        if len(all_scores) > 1:
            ax.hist(all_scores, bins=max(3, len(all_scores)//2), alpha=0.7, color='red', edgecolor='black')
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
    
    def _create_method_details(self, ax, method_results: Dict[str, Any]) -> None:
        """Create method details visualization."""
        if method_results:
            method_names = list(method_results.keys())
            
            # Create detailed breakdown
            details_text = "METHOD DETAILS:\n\n"
            
            for method, data in method_results.items():
                if isinstance(data, dict):
                    risk_score = data.get('risk_score', 0.0)
                    confidence = data.get('confidence', 0.0)
                    details_text += f"{method.upper()}:\n"
                    details_text += f"  Risk: {risk_score:.3f}\n"
                    details_text += f"  Confidence: {confidence:.3f}\n\n"
            
            ax.text(0.05, 0.95, details_text, transform=ax.transAxes, fontsize=9,
                   verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.3))
        else:
            ax.text(0.5, 0.5, 'No Method\nDetails Available', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12)
        
        ax.set_title('Method Details', fontweight='bold')
        ax.axis('off')
    
    def _create_recommendations_panel(self, ax, results) -> None:
        """Create recommendations panel."""
        risk_color = self.risk_colors.get(results.risk_level, 'gray')
        
        recommendations_text = f"""
🚨 RISK: {results.risk_level} 🚨

📊 SCORE: {results.risk_score:.3f}
🎯 CONFIDENCE: {results.confidence:.3f}

💡 RECOMMENDATIONS:
"""
        
        for i, rec in enumerate(results.recommendations[:4], 1):
            recommendations_text += f"\n{i}. {rec}"
        
        recommendations_text += f"""

🔄 NEXT STEPS:
• {"Immediate attention" if results.risk_level == "HIGH" else "Monitor closely"}
• {"Deploy interventions" if results.risk_score > 0.6 else "Continue analysis"}
        """
        
        ax.text(0.02, 0.98, recommendations_text, transform=ax.transAxes, fontsize=9,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle="round,pad=0.4", facecolor=risk_color, alpha=0.2))
        ax.set_title('Risk Assessment & Recommendations', fontweight='bold')
        ax.axis('off')
    
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
    
    def add_detection_result(self, result):
        """Add a detection result to the history."""
        self.detection_history.append(result)
        print(f"📈 Added detection result. History length: {len(self.detection_history)}")


# Demo function
def demo_dashboard():
    """Create and display a demo dashboard."""
    print("🎨 Creating demo dashboard...")
    
    # Create sample result
    class MockResult:
        def __init__(self):
            self.risk_score = 0.65
            self.risk_level = "MEDIUM"
            self.confidence = 0.78
            self.method_results = {
                'gradient': {'risk_score': 0.72, 'confidence': 0.80},
                'activation': {'risk_score': 0.58, 'confidence': 0.75},
                'behavioral': {'risk_score': 0.64, 'confidence': 0.82}
            }
            self.recommendations = [
                "Increase monitoring frequency",
                "Review training procedures", 
                "Deploy additional safety measures",
                "Analyze gradient patterns",
                "Monitor for deceptive behavior"
            ]
    
    # Create dashboard and sample results
    dashboard = MesaDetectionDashboard()
    
    # Add some history
    for i in range(5):
        sample = MockResult()
        sample.risk_score = np.random.uniform(0.2, 0.8)
        sample.risk_level = "MEDIUM" if sample.risk_score > 0.5 else "LOW"
        dashboard.add_detection_result(sample)
    
    # Create comprehensive report
    final_result = MockResult()
    dashboard.create_comprehensive_report(final_result)
    
    return dashboard


if __name__ == "__main__":
    demo_dashboard() 