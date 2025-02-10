import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

import numpy as np
import json
import os
from datetime import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, List, Optional, Union

class ExperimentTracker:
    def __init__(self, base_dir: str = 'experiments'):
        """Initialize experiment tracker"""
        self.base_dir = base_dir
        self.experiments_dir = os.path.join(base_dir, 'metadata')
        self.models_dir = os.path.join(base_dir, 'models')
        
        # Create directories
        os.makedirs(self.experiments_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Track current experiment
        self.current_experiment = None
        
    def start_experiment(self, 
                        name: str,
                        model_type: str,
                        hyperparameters: Dict[str, Any]) -> str:
        """Start a new experiment"""
        experiment_id = f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.current_experiment = {
            'experiment_id': experiment_id,
            'name': name,
            'model_type': model_type,
            'hyperparameters': hyperparameters,
            'start_time': datetime.now().isoformat(),
            'status': 'running',
            'metrics': {},
            'feature_importance': None,
            'training_history': [],
            'model_path': os.path.join(self.models_dir, experiment_id)
        }
        
        return experiment_id
        
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        if not self.current_experiment:
            raise ValueError("No active experiment")
            
        # Validate metrics
        if not isinstance(metrics, dict):
            raise ValueError("Metrics must be a dictionary")
            
        processed_metrics = {}
        try:
            for key, value in metrics.items():
                if isinstance(value, (np.ndarray, np.number)):
                    processed_metrics[key] = value.item() if isinstance(value, np.number) else value.tolist()
                elif isinstance(value, dict):
                    processed_metrics[key] = {
                        k: v.item() if isinstance(v, np.number) else v.tolist() if isinstance(v, np.ndarray) else v
                        for k, v in value.items()
                    }
                else:
                    processed_metrics[key] = value
                    
            # Add processed metrics to history
            history_entry = {
                'step': step,
                'timestamp': datetime.now().isoformat(),
                **processed_metrics
            }
            
            self.current_experiment['training_history'].append(history_entry)
            logger.info(f"Logged metrics for step {step}")
            
        except Exception as e:
            logger.error(f"Error processing metrics: {e}")
            raise
        
    def log_feature_importance(self, feature_names: List[str], importance_values: List[float]):
        """Log feature importance scores"""
        if not self.current_experiment:
            raise ValueError("No active experiment")
            
        self.current_experiment['feature_importance'] = {
            'features': feature_names,
            'importance': importance_values
        }
        
    def end_experiment(self, final_metrics: Dict[str, float], status: str = 'completed'):
        """End current experiment and save results"""
        if not self.current_experiment:
            raise ValueError("No active experiment")
            
        self.current_experiment.update({
            'end_time': datetime.now().isoformat(),
            'status': status,
            'final_metrics': final_metrics
        })
        
        # Save experiment metadata
        experiment_path = os.path.join(
            self.experiments_dir, 
            f"{self.current_experiment['experiment_id']}.json"
        )
        
        with open(experiment_path, 'w') as f:
            json.dump(self.current_experiment, f, indent=4)
            
        return self.current_experiment['experiment_id']
        
    def load_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """Load experiment details"""
        experiment_path = os.path.join(self.experiments_dir, f"{experiment_id}.json")
        
        if not os.path.exists(experiment_path):
            raise ValueError(f"Experiment {experiment_id} not found")
            
        with open(experiment_path, 'r') as f:
            return json.load(f)
            
    def list_experiments(self, model_type: Optional[str] = None) -> pd.DataFrame:
        """List all experiments with optional filtering"""
        experiments = []
        
        for filename in os.listdir(self.experiments_dir):
            if filename.endswith('.json'):
                with open(os.path.join(self.experiments_dir, filename), 'r') as f:
                    exp = json.load(f)
                    if model_type is None or exp['model_type'] == model_type:
                        experiments.append(exp)
                        
        if not experiments:
            return pd.DataFrame()
            
        # Convert to DataFrame
        df = pd.DataFrame(experiments)
        
        # Extract final metrics into separate columns
        if 'final_metrics' in df.columns:
            metrics_df = pd.DataFrame(df['final_metrics'].tolist())
            df = pd.concat([df.drop('final_metrics', axis=1), metrics_df], axis=1)
            
        return df
        
    def plot_experiment_comparison(self, metric: str = 'accuracy') -> go.Figure:
        """Create comparison plot for experiments"""
        df = self.list_experiments()
        if df.empty:
            return None
            
        fig = px.bar(
            df,
            x='experiment_id',
            y=metric,
            color='model_type',
            title=f'Experiment Comparison - {metric}',
            labels={'experiment_id': 'Experiment', 'value': metric}
        )
        
        fig.update_layout(
            xaxis_tickangle=-45,
            showlegend=True,
            height=500
        )
        
        return fig
        
    def plot_feature_importance(self, experiment_id: str) -> go.Figure:
        """Plot feature importance for an experiment"""
        experiment = self.load_experiment(experiment_id)
        
        if not experiment.get('feature_importance'):
            return None
            
        fi_data = experiment['feature_importance']
        
        fig = px.bar(
            x=fi_data['importance'],
            y=fi_data['features'],
            orientation='h',
            title='Feature Importance',
            labels={'x': 'Importance', 'y': 'Features'}
        )
        
        fig.update_layout(
            showlegend=False,
            height=max(400, len(fi_data['features']) * 20)
        )
        
        return fig
        
    def plot_training_history(self, experiment_id: str, metric: str = 'accuracy') -> go.Figure:
        """Plot training history for specific metric"""
        experiment = self.load_experiment(experiment_id)
        
        if not experiment.get('training_history'):
            return None
            
        history_df = pd.DataFrame(experiment['training_history'])
        
        fig = px.line(
            history_df,
            x='step',
            y=metric,
            title=f'Training History - {metric}',
            labels={'step': 'Step', 'value': metric}
        )
        
        fig.update_layout(showlegend=True, height=400)
        
        return fig
    
    def validate_experiment(self, experiment_id: str) -> bool:
        """Validate experiment data integrity"""
        try:
            experiment = self.load_experiment(experiment_id)
            required_fields = ['experiment_id', 'name', 'model_type', 'start_time', 'status']
            
            # Check required fields
            for field in required_fields:
                if field not in experiment:
                    logger.error(f"Missing required field: {field}")
                    return False
                    
            # Check metrics format
            if 'final_metrics' in experiment and not isinstance(experiment['final_metrics'], dict):
                logger.error("Invalid final metrics format")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error validating experiment: {e}")
            return False
    
    def cleanup_stale_experiments(self, hours_threshold: int = 24) -> int:
        """Clean up experiments that have been running for too long"""
        try:
            count = 0
            current_time = datetime.now()
            
            for filename in os.listdir(self.experiments_dir):
                if filename.endswith('.json'):
                    filepath = os.path.join(self.experiments_dir, filename)
                    with open(filepath, 'r') as f:
                        exp = json.load(f)
                        if exp['status'] == 'running':
                            start_time = datetime.fromisoformat(exp['start_time'])
                            if (current_time - start_time).total_seconds() > hours_threshold * 3600:
                                exp['status'] = 'failed'
                                exp['end_time'] = current_time.isoformat()
                                with open(filepath, 'w') as f:
                                    json.dump(exp, f, indent=4)
                                count += 1
                                
            logger.info(f"Cleaned up {count} stale experiments")
            return count
            
        except Exception as e:
            logger.error(f"Error cleaning up stale experiments: {e}")
            return 0
    
    def cleanup_failed_experiments(self) -> int:
        """Clean up failed experiments"""
        try:
            count = 0
            for filename in os.listdir(self.experiments_dir):
                if filename.endswith('.json'):
                    filepath = os.path.join(self.experiments_dir, filename)
                    with open(filepath, 'r') as f:
                        exp = json.load(f)
                        if exp['status'] == 'failed':
                            os.remove(filepath)
                            count += 1
            logger.info(f"Cleaned up {count} failed experiments")
            return count
        except Exception as e:
            logger.error(f"Error cleaning up failed experiments: {e}")
            return 0
    
    def get_experiment_summary(self) -> Dict[str, Any]:
        """Get summary of all experiments"""
        df = self.list_experiments()
        if df.empty:
            return {}
            
        summary = {
            'total_experiments': len(df),
            'best_metrics': {
                'accuracy': df['accuracy'].max(),
                'f1_macro': df['f1_macro'].max(),
                'precision_macro': df['precision_macro'].max(),
                'recall_macro': df['recall_macro'].max()
            },
            'model_types': df['model_type'].value_counts().to_dict(),
            'status_counts': df['status'].value_counts().to_dict(),
            'latest_experiment': df.iloc[-1].to_dict() if not df.empty else {}
        }
        
        return summary
    
    def export_experiment(self, experiment_id: str, export_dir: str = 'exports') -> str:
        """Export experiment data to JSON"""
        try:
            os.makedirs(export_dir, exist_ok=True)
            experiment = self.load_experiment(experiment_id)
            
            export_path = os.path.join(
                export_dir, 
                f"experiment_{experiment_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            
            with open(export_path, 'w') as f:
                json.dump(experiment, f, indent=4)
                
            logger.info(f"Exported experiment {experiment_id} to {export_path}")
            return export_path
            
        except Exception as e:
            logger.error(f"Error exporting experiment: {e}")
            raise
        
    def compare_experiments(self, experiment_ids: List[str]) -> pd.DataFrame:
        """Compare specific experiments"""
        try:
            experiments = []
            for exp_id in experiment_ids:
                exp = self.load_experiment(exp_id)
                if exp['final_metrics']:  # Only include completed experiments
                    experiments.append({
                        'experiment_id': exp_id,
                        'model_type': exp['model_type'],
                        'status': exp['status'],
                        **exp['final_metrics']
                    })
                    
            return pd.DataFrame(experiments)
            
        except Exception as e:
            logger.error(f"Error comparing experiments: {e}")
            return pd.DataFrame()