import streamlit as st
from experiments.experiment_tracker import ExperimentTracker
import json
import plotly.express as px
from datetime import datetime

class ExperimentDetailsPage:
    def __init__(self):
        self.tracker = ExperimentTracker()
        
    def show_experiment_details(self, experiment_id: str):
        """Show detailed information about specific experiment"""
        experiment = self.tracker.load_experiment(experiment_id)
        
        # Basic Information
        st.subheader("Experiment Information")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Model Type", experiment['model_type'])
        with col2:
            st.metric("Status", experiment['status'])
        with col3:
            start_time = datetime.fromisoformat(experiment['start_time'])
            st.metric("Start Time", start_time.strftime("%Y-%m-%d %H:%M:%S"))
            
        # Performance Metrics
        st.subheader("Performance Metrics")
        metrics = experiment.get('final_metrics', {})
        cols = st.columns(len(metrics))
        for col, (metric, value) in zip(cols, metrics.items()):
            col.metric(metric, f"{value:.4f}")
            
        # Hyperparameters
        st.subheader("Hyperparameters")
        st.json(experiment['hyperparameters'])
        
        # Feature Importance
        if experiment.get('feature_importance'):
            st.subheader("Feature Importance")
            fig = self.tracker.plot_feature_importance(experiment_id)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
                
        # Training History
        if experiment.get('training_history'):
            st.subheader("Training History")
            
            metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
            selected_metric = st.selectbox("Select Metric", metrics)
            
            fig = self.tracker.plot_training_history(experiment_id, selected_metric)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
                
        # Raw Data
        if st.checkbox("Show Raw Experiment Data"):
            st.json(experiment)
            
    def run(self):
        st.title("Experiment Details")
        
        # Get list of experiments
        experiments = self.tracker.list_experiments()
        if experiments.empty:
            st.info("No experiments found.")
            return
            
        # Select experiment
        selected_exp = st.selectbox(
            "Select Experiment",
            experiments['experiment_id'].tolist(),
            format_func=lambda x: f"{x} ({experiments[experiments['experiment_id']==x]['name'].iloc[0]})"
        )
        
        if selected_exp:
            self.show_experiment_details(selected_exp)

if __name__ == "__main__":
    page = ExperimentDetailsPage()
    page.run()