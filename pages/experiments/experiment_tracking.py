import streamlit as st
import plotly.express as px
from experiments.experiment_tracker import ExperimentTracker
from datetime import datetime
import pandas as pd

class ExperimentTrackingPage:
    def __init__(self):
        self.tracker = ExperimentTracker()
        
    def show_experiments_overview(self):
        """Show overview of all experiments"""
        st.subheader("Experiments Overview")
        
        # Get experiments data
        df = self.tracker.list_experiments()
        if df.empty:
            st.info("No experiments found. Start training models to see results here.")
            return
            
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Experiments", len(df))
        with col2:
            st.metric("Best Accuracy", f"{df['accuracy'].max():.4f}")
        with col3:
            st.metric("Best F1 Score", f"{df['f1_macro'].max():.4f}")
            
        # Experiments table
        st.subheader("Recent Experiments")
        st.dataframe(
            df[['experiment_id', 'name', 'model_type', 'accuracy', 'f1_macro', 'status', 'start_time']]
            .sort_values('start_time', ascending=False)
        )
        
    def show_performance_trends(self):
        """Show performance trends across experiments"""
        st.subheader("Performance Trends")
        
        metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
        selected_metric = st.selectbox("Select Metric", metrics)
        
        fig = self.tracker.plot_experiment_comparison(selected_metric)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
            
    def show_feature_importance_analysis(self):
        """Show feature importance analysis"""
        st.subheader("Feature Importance Analysis")
        
        experiments = self.tracker.list_experiments()
        if not experiments.empty:
            selected_exp = st.selectbox(
                "Select Experiment",
                experiments['experiment_id'].tolist(),
                format_func=lambda x: f"{x} ({experiments[experiments['experiment_id']==x]['name'].iloc[0]})"
            )
            
            fig = self.tracker.plot_feature_importance(selected_exp)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No feature importance data available for this experiment")
                
    def show_training_history(self):
        """Show training history for selected experiment"""
        st.subheader("Training History")
        
        experiments = self.tracker.list_experiments()
        if not experiments.empty:
            selected_exp = st.selectbox(
                "Select Experiment",
                experiments['experiment_id'].tolist(),
                format_func=lambda x: f"{x} ({experiments[experiments['experiment_id']==x]['name'].iloc[0]})",
                key="history_selector"
            )
            
            metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
            selected_metric = st.selectbox("Select Metric", metrics, key="metric_selector")
            
            fig = self.tracker.plot_training_history(selected_exp, selected_metric)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No training history available for this experiment")
                
    def run(self):
        st.title("Experiment Tracking Dashboard")
        
        # Create tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "Overview",
            "Performance Trends",
            "Feature Importance",
            "Training History"
        ])
        
        with tab1:
            self.show_experiments_overview()
            
        with tab2:
            self.show_performance_trends()
            
        with tab3:
            self.show_feature_importance_analysis()
            
        with tab4:
            self.show_training_history()

if __name__ == "__main__":
    page = ExperimentTrackingPage()
    page.run()