import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from typing import Dict, List, Optional
import json
import os
from datetime import datetime

class TrainingMonitor:
   def __init__(self):
       self.metrics_history = {
           'train_accuracy': [],
           'train_f1_macro': [],
           'train_precision_macro': [],
           'train_recall_macro': [],
           'val_accuracy': [],
           'val_f1_macro': [],
           'val_precision_macro': [],
           'val_recall_macro': []
       }
       self.current_epoch = 0
       
   def reset(self):
       """Reset metrics history"""
       self.metrics_history = {
           'train_accuracy': [],
           'train_f1_macro': [],
           'train_precision_macro': [],
           'train_recall_macro': [],
           'val_accuracy': [],
           'val_f1_macro': [],
           'val_precision_macro': [],
           'val_recall_macro': []
       }
       self.current_epoch = 0

   def update_metrics(self, metrics: Dict[str, float], epoch: Optional[int] = None):
       """Update metrics with new values"""
       for key, value in metrics.items():
           if key in self.metrics_history:
               self.metrics_history[key].append(float(value))
               
       if epoch is not None:
           self.current_epoch = epoch
       else:
           self.current_epoch += 1

   def create_progress_plots(self) -> go.Figure:
       """Create training progress plots"""
       fig = make_subplots(
           rows=2, cols=2,
           subplot_titles=('Accuracy', 'F1 Score', 'Precision', 'Recall'),
           vertical_spacing=0.15,
           horizontal_spacing=0.1
       )
       
       # Accuracy plot
       if self.metrics_history['train_accuracy']:
           fig.add_trace(
               go.Scatter(
                   y=self.metrics_history['train_accuracy'],
                   mode='lines',
                   name='Train Accuracy',
                   line=dict(color='blue')
               ),
               row=1, col=1
           )
       if self.metrics_history['val_accuracy']:
           fig.add_trace(
               go.Scatter(
                   y=self.metrics_history['val_accuracy'],
                   mode='lines',
                   name='Val Accuracy',
                   line=dict(color='red', dash='dash')
               ),
               row=1, col=1
           )

       # F1 Score plot
       if self.metrics_history['train_f1_macro']:
           fig.add_trace(
               go.Scatter(
                   y=self.metrics_history['train_f1_macro'],
                   mode='lines',
                   name='Train F1',
                   line=dict(color='green')
               ),
               row=1, col=2
           )
       if self.metrics_history['val_f1_macro']:
           fig.add_trace(
               go.Scatter(
                   y=self.metrics_history['val_f1_macro'],
                   mode='lines',
                   name='Val F1',
                   line=dict(color='orange', dash='dash')
               ),
               row=1, col=2
           )

       # Precision plot
       if self.metrics_history['train_precision_macro']:
           fig.add_trace(
               go.Scatter(
                   y=self.metrics_history['train_precision_macro'],
                   mode='lines',
                   name='Train Precision',
                   line=dict(color='purple')
               ),
               row=2, col=1
           )
       if self.metrics_history['val_precision_macro']:
           fig.add_trace(
               go.Scatter(
                   y=self.metrics_history['val_precision_macro'],
                   mode='lines',
                   name='Val Precision',
                   line=dict(color='brown', dash='dash')
               ),
               row=2, col=1
           )

       # Recall plot
       if self.metrics_history['train_recall_macro']:
           fig.add_trace(
               go.Scatter(
                   y=self.metrics_history['train_recall_macro'],
                   mode='lines',
                   name='Train Recall',
                   line=dict(color='cyan')
               ),
               row=2, col=2
           )
       if self.metrics_history['val_recall_macro']:
           fig.add_trace(
               go.Scatter(
                   y=self.metrics_history['val_recall_macro'],
                   mode='lines',
                   name='Val Recall',
                   line=dict(color='magenta', dash='dash')
               ),
               row=2, col=2
           )

       fig.update_layout(
           height=800,
           showlegend=True,
           title_text="Training Progress",
           hovermode='x unified'
       )
       
       return fig
       
   def display_current_metrics(self):
       """Display current metrics in Streamlit"""
       col1, col2, col3, col4 = st.columns(4)
       
       # Training Metrics
       with col1:
           if self.metrics_history['train_accuracy']:
               st.metric(
                   "Training Accuracy",
                   f"{self.metrics_history['train_accuracy'][-1]:.4f}",
                   delta=f"{self.metrics_history['train_accuracy'][-1] - self.metrics_history['train_accuracy'][-2]:.4f}"
                   if len(self.metrics_history['train_accuracy']) > 1 else None
               )
               
           if self.metrics_history['train_f1_macro']:
               st.metric(
                   "Training F1",
                   f"{self.metrics_history['train_f1_macro'][-1]:.4f}",
                   delta=f"{self.metrics_history['train_f1_macro'][-1] - self.metrics_history['train_f1_macro'][-2]:.4f}"
                   if len(self.metrics_history['train_f1_macro']) > 1 else None
               )
               
       with col2:
           if self.metrics_history['train_precision_macro']:
               st.metric(
                   "Training Precision",
                   f"{self.metrics_history['train_precision_macro'][-1]:.4f}",
                   delta=f"{self.metrics_history['train_precision_macro'][-1] - self.metrics_history['train_precision_macro'][-2]:.4f}"
                   if len(self.metrics_history['train_precision_macro']) > 1 else None
               )
               
           if self.metrics_history['train_recall_macro']:
               st.metric(
                   "Training Recall",
                   f"{self.metrics_history['train_recall_macro'][-1]:.4f}",
                   delta=f"{self.metrics_history['train_recall_macro'][-1] - self.metrics_history['train_recall_macro'][-2]:.4f}"
                   if len(self.metrics_history['train_recall_macro']) > 1 else None
               )
               
       # Validation Metrics
       with col3:
           if self.metrics_history['val_accuracy']:
               st.metric(
                   "Validation Accuracy",
                   f"{self.metrics_history['val_accuracy'][-1]:.4f}",
                   delta=f"{self.metrics_history['val_accuracy'][-1] - self.metrics_history['val_accuracy'][-2]:.4f}"
                   if len(self.metrics_history['val_accuracy']) > 1 else None
               )
               
           if self.metrics_history['val_f1_macro']:
               st.metric(
                   "Validation F1",
                   f"{self.metrics_history['val_f1_macro'][-1]:.4f}",
                   delta=f"{self.metrics_history['val_f1_macro'][-1] - self.metrics_history['val_f1_macro'][-2]:.4f}"
                   if len(self.metrics_history['val_f1_macro']) > 1 else None
               )
               
       with col4:
           if self.metrics_history['val_precision_macro']:
               st.metric(
                   "Validation Precision",
                   f"{self.metrics_history['val_precision_macro'][-1]:.4f}",
                   delta=f"{self.metrics_history['val_precision_macro'][-1] - self.metrics_history['val_precision_macro'][-2]:.4f}"
                   if len(self.metrics_history['val_precision_macro']) > 1 else None
               )
               
           if self.metrics_history['val_recall_macro']:
               st.metric(
                   "Validation Recall",
                   f"{self.metrics_history['val_recall_macro'][-1]:.4f}",
                   delta=f"{self.metrics_history['val_recall_macro'][-1] - self.metrics_history['val_recall_macro'][-2]:.4f}"
                   if len(self.metrics_history['val_recall_macro']) > 1 else None
               )
               
   def save_history(self, model_dir: str):
       """Save training history"""
       history_path = os.path.join(model_dir, 'training_history.json')
       os.makedirs(os.path.dirname(history_path), exist_ok=True)
       with open(history_path, 'w') as f:
           json.dump({
               'metrics_history': self.metrics_history,
               'final_epoch': self.current_epoch,
               'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
           }, f, indent=4)
           
   def load_history(self, model_dir: str):
       """Load training history"""
       history_path = os.path.join(model_dir, 'training_history.json')
       if os.path.exists(history_path):
           with open(history_path, 'r') as f:
               data = json.load(f)
               self.metrics_history = data['metrics_history']
               self.current_epoch = data['final_epoch']

class TrainingCallback:
   def __init__(self, monitor: TrainingMonitor):
       self.monitor = monitor
       self.placeholder = st.empty()
       
   def on_epoch_end(self, metrics: Dict[str, float]):
       """Callback for end of epoch"""
       self.monitor.update_metrics(metrics)
       
       with self.placeholder.container():
           # Display current metrics
           self.monitor.display_current_metrics()
           
           # Display progress plots
           st.plotly_chart(
               self.monitor.create_progress_plots(),
               use_container_width=True
           )
           
   def on_training_end(self, model_dir: str):
       """Callback for end of training"""
       self.monitor.save_history(model_dir)