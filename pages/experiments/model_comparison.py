import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from experiments.experiment_tracker import ExperimentTracker
import pandas as pd

class ModelComparisonPage:
    def __init__(self):
        self.tracker = ExperimentTracker()
        
    def show_model_comparison(self):
        """Show comparison between different models"""
        st.subheader("Model Performance Comparison")
        
        # Get experiments data
        df = self.tracker.list_experiments()
        if df.empty:
            st.info("No experiments found for comparison.")
            return
            
        # Select models to compare
        model_types = df['model_type'].unique()
        selected_models = st.multiselect(
            "Select Models to Compare",
            model_types,
            default=model_types
        )
        
        if not selected_models:
            st.warning("Please select at least one model type.")
            return
            
        # Filter data
        df_filtered = df[df['model_type'].isin(selected_models)]
        
        # Metrics comparison
        metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
        
        fig = go.Figure()
        for metric in metrics:
            fig.add_trace(go.Box(
                y=df_filtered[metric],
                x=df_filtered['model_type'],
                name=metric,
                boxpoints='all'
            ))
            
        fig.update_layout(
            title="Performance Metrics Distribution by Model Type",
            xaxis_title="Model Type",
            yaxis_title="Score",
            boxmode='group'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed comparison table
        st.subheader("Detailed Comparison")
        comparison_df = df_filtered.groupby('model_type').agg({
            'accuracy': ['mean', 'std', 'max'],
            'f1_macro': ['mean', 'std', 'max'],
            'training_time': ['mean', 'max']
        }).round(4)
        
        st.dataframe(comparison_df)
        
    def show_experiment_selector(self):
        """Show detailed comparison between selected experiments"""
        st.subheader("Compare Specific Experiments")
        
        df = self.tracker.list_experiments()
        if df.empty:
            return
            
        # Select experiments to compare
        selected_experiments = st.multiselect(
            "Select Experiments to Compare",
            df['experiment_id'].tolist(),
            format_func=lambda x: f"{x} ({df[df['experiment_id']==x]['name'].iloc[0]})"
        )
        
        if not selected_experiments:
            st.warning("Please select experiments to compare.")
            return
            
        # Get detailed comparison
        comparison_data = []
        for exp_id in selected_experiments:
            exp_data = self.tracker.load_experiment(exp_id)
            comparison_data.append({
                'Experiment ID': exp_id,
                'Name': exp_data['name'],
                'Model Type': exp_data['model_type'],
                **exp_data['final_metrics']
            })
            
        comparison_df = pd.DataFrame(comparison_data)
        
        # Show comparison table
        st.dataframe(comparison_df)
        
        # Visualize comparison
        metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
        selected_metric = st.selectbox("Select Metric for Visualization", metrics)
        
        fig = px.bar(
            comparison_df,
            x='Experiment ID',
            y=selected_metric,
            color='Model Type',
            title=f'Comparison of {selected_metric}'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    def run(self):
        st.title("Model Comparison Dashboard")
        
        tab1, tab2 = st.tabs([
            "Model Type Comparison",
            "Experiment Comparison"
        ])
        
        with tab1:
            self.show_model_comparison()
            
        with tab2:
            self.show_experiment_selector()

if __name__ == "__main__":
    page = ModelComparisonPage()
    page.run()