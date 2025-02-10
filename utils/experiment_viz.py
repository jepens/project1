import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import List, Dict, Any

class ExperimentViz:
    @staticmethod
    def create_metrics_comparison(df: pd.DataFrame, metrics: List[str]) -> go.Figure:
        """Create comparison plot for multiple metrics"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=metrics,
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )
        
        # Plot each metric
        for i, metric in enumerate(metrics, 1):
            row = (i-1) // 2 + 1
            col = (i-1) % 2 + 1
            
            # Create box plot for each model type
            for model_type in df['model_type'].unique():
                values = df[df['model_type'] == model_type][metric]
                
                fig.add_trace(
                    go.Box(
                        y=values,
                        name=model_type,
                        showlegend=i==1  # Show legend only for first plot
                    ),
                    row=row, col=col
                )
        
        fig.update_layout(
            height=700,
            title_text="Model Performance Comparison",
            boxmode='group'
        )
        
        return fig
        
    @staticmethod
    def create_learning_curves(history: List[Dict[str, Any]], 
                             metrics: List[str]) -> go.Figure:
        """Create learning curves plot"""
        df = pd.DataFrame(history)
        
        fig = make_subplots(
            rows=len(metrics), cols=1,
            subplot_titles=metrics,
            vertical_spacing=0.1
        )
        
        for i, metric in enumerate(metrics, 1):
            # Training metric
            fig.add_trace(
                go.Scatter(
                    x=df['step'],
                    y=df[metric],
                    name=f'Training {metric}',
                    mode='lines+markers'
                ),
                row=i, col=1
            )
            
            # Validation metric if available
            val_metric = f'val_{metric}'
            if val_metric in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df['step'],
                        y=df[val_metric],
                        name=f'Validation {metric}',
                        mode='lines+markers'
                    ),
                    row=i, col=1
                )
        
        fig.update_layout(
            height=200 * len(metrics),
            title_text="Learning Curves",
            showlegend=True
        )
        
        return fig
        
    @staticmethod
    def create_feature_importance_plot(features: List[str], 
                                     importance: List[float],
                                     top_n: int = 20) -> go.Figure:
        """Create feature importance plot"""
        # Sort features by importance
        df = pd.DataFrame({
            'feature': features,
            'importance': importance
        }).sort_values('importance', ascending=True)
        
        # Select top N features
        if len(df) > top_n:
            df = df.tail(top_n)
        
        fig = go.Figure(go.Bar(
            x=df['importance'],
            y=df['feature'],
            orientation='h'
        ))
        
        fig.update_layout(
            title="Feature Importance",
            xaxis_title="Importance Score",
            yaxis_title="Feature",
            height=max(400, len(df) * 20)
        )
        
        return fig
        
    @staticmethod
    def create_confusion_matrix_plot(cm: np.ndarray, 
                                   labels: List[str]) -> go.Figure:
        """Create confusion matrix heatmap"""
        fig = px.imshow(
            cm,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=labels,
            y=labels,
            text=cm,
            aspect="auto",
            color_continuous_scale="Blues"
        )
        
        fig.update_layout(
            title="Confusion Matrix",
            xaxis_title="Predicted Label",
            yaxis_title="True Label",
            width=600,
            height=600
        )
        
        return fig
        
    @staticmethod
    def create_experiment_timeline(experiments: pd.DataFrame) -> go.Figure:
        """Create timeline of experiments"""
        df = experiments.copy()
        df['duration'] = pd.to_datetime(df['end_time']) - pd.to_datetime(df['start_time'])
        df['duration_minutes'] = df['duration'].dt.total_seconds() / 60
        
        fig = px.timeline(
            df,
            x_start='start_time',
            x_end='end_time',
            y='model_type',
            color='status',
            hover_data=['name', 'duration_minutes', 'accuracy', 'f1_macro']
        )
        
        fig.update_layout(
            title="Experiments Timeline",
            xaxis_title="Time",
            yaxis_title="Model Type",
            height=400
        )
        
        return fig