import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from typing import List, Dict, Any

class ModelVisualizer:
    @staticmethod
    def plot_confusion_matrix(confusion_mat: np.ndarray, labels: List[str]) -> go.Figure:
        """Plot confusion matrix as a heatmap"""
        fig = go.Figure(data=go.Heatmap(
            z=confusion_mat,
            x=labels,
            y=labels,
            colorscale='Blues',
            text=confusion_mat,
            texttemplate="%{text}",
            textfont={"size": 16},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title='Confusion Matrix',
            xaxis_title='Predicted Label',
            yaxis_title='True Label',
            yaxis=dict(autorange='reversed'),
            width=600,
            height=500
        )
        
        return fig
    
    @staticmethod
    def plot_roc_curves(fpr: Dict, tpr: Dict, roc_auc: Dict, labels: List[str]) -> go.Figure:
        """Plot ROC curves for multiple classes"""
        fig = go.Figure()
        colors = px.colors.qualitative.Set3
        
        for i, label in enumerate(labels):
            fig.add_trace(
                go.Scatter(
                    x=fpr[i],
                    y=tpr[i],
                    name=f'{label} (AUC = {roc_auc[i]:.2f})',
                    line=dict(color=colors[i % len(colors)])
                )
            )
            
        # Add diagonal line
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                name='Random',
                line=dict(color='black', dash='dash'),
                showlegend=True
            )
        )
        
        fig.update_layout(
            title='ROC Curves',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            yaxis=dict(scaleanchor="x", scaleratio=1),
            xaxis=dict(constrain='domain'),
            width=700,
            height=500,
            legend=dict(
                x=1.05,
                y=1,
                title_text='Classes'
            )
        )
        
        return fig
    
    @staticmethod
    def plot_metrics_comparison(metrics_data: List[Dict[str, float]]) -> go.Figure:
        """Plot comparison of different metrics"""
        fig = go.Figure()
        
        metrics = ['Accuracy', 'F1 Score', 'Precision', 'Recall']
        colors = px.colors.qualitative.Set2
        
        for i, metric in enumerate(metrics):
            fig.add_trace(
                go.Bar(
                    name=metric,
                    x=list(range(len(metrics_data))),
                    y=[d.get(metric, 0) for d in metrics_data],
                    marker_color=colors[i % len(colors)]
                )
            )
            
        fig.update_layout(
            title='Model Performance Metrics',
            xaxis_title='Model Version',
            yaxis_title='Score',
            barmode='group',
            yaxis=dict(range=[0, 1]),
            width=800,
            height=500,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    
    @staticmethod
    def plot_learning_curves(history: Dict[str, List[float]]) -> go.Figure:
        """Plot learning curves from training history"""
        fig = go.Figure()
        
        # Plot training metrics
        metrics = {
            'loss': ('Loss', 'blue'),
            'accuracy': ('Accuracy', 'green'),
            'val_loss': ('Validation Loss', 'red'),
            'val_accuracy': ('Validation Accuracy', 'orange')
        }
        
        for metric, (name, color) in metrics.items():
            if metric in history:
                fig.add_trace(
                    go.Scatter(
                        y=history[metric],
                        name=name,
                        line=dict(color=color),
                        mode='lines'
                    )
                )
                
        fig.update_layout(
            title='Learning Curves',
            xaxis_title='Epoch',
            yaxis_title='Value',
            width=800,
            height=500,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    
    @staticmethod
    def plot_feature_importance(importance_scores: np.ndarray, feature_names: np.ndarray, top_n: int = 20) -> go.Figure:
        """Plot feature importance scores"""
        # Get top N features
        if len(importance_scores) > top_n:
            indices = np.argsort(importance_scores)[-top_n:]
            importance_scores = importance_scores[indices]
            feature_names = feature_names[indices]
            
        fig = go.Figure(
            go.Bar(
                x=importance_scores,
                y=feature_names,
                orientation='h'
            )
        )
        
        fig.update_layout(
            title=f'Top {top_n} Most Important Features',
            xaxis_title='Importance Score',
            yaxis_title='Feature',
            width=800,
            height=max(400, len(feature_names) * 20),
            yaxis={'categoryorder': 'total ascending'}
        )
        
        return fig
    
    @staticmethod
    def plot_parameter_importance(param_scores: Dict[str, List[float]], param_names: List[str]) -> go.Figure:
        """Plot parameter importance analysis"""
        fig = go.Figure()
        
        for param in param_names:
            if param in param_scores:
                scores = param_scores[param]
                fig.add_trace(
                    go.Box(
                        x=scores,
                        name=param
                    )
                )
                
        fig.update_layout(
            title='Parameter Importance Analysis',
            xaxis_title='Performance Score',
            yaxis_title='Parameter',
            width=800,
            height=400
        )
        
        return fig
    
    @staticmethod
    def plot_experiment_metrics(experiments_df: pd.DataFrame) -> go.Figure:
        """Plot metrics comparison across experiments"""
        fig = go.Figure()
        metrics = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']
        
        for metric in metrics:
            fig.add_trace(
                go.Box(
                    y=experiments_df[metric],
                    name=metric.replace('_', ' ').title(),
                    boxpoints='all'
                )
            )
            
        fig.update_layout(
            title='Experiment Metrics Distribution',
            yaxis_title='Score',
            xaxis_title='Metric',
            showlegend=False
        )
        
        return fig