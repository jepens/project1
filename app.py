import streamlit as st
import pandas as pd
import numpy as np
import json
import glob
import os
from datetime import datetime
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__) 

from preprocessing.text_preprocessor import TextPreprocessor
from models.random_forest_model import RandomForestModel
from models.lstm_model import LSTMModel
from models.distilbert_model import DistilBERTModel
from utils.data_analysis import DataAnalyzer
from utils.visualization import ModelVisualizer
from utils.storage_manager import StorageManager
from utils.training_monitor import TrainingMonitor, TrainingCallback

from pages.model_training import ModelTrainingPage
from pages.preprocessing import PreprocessingPage
from pages.experiments.experiment_tracking import ExperimentTrackingPage
from pages.experiments.model_comparison import ModelComparisonPage
from pages.experiments.experiment_details import ExperimentDetailsPage
from experiments.experiment_tracker import ExperimentTracker

class SentimentAnalysisApp:
    def __init__(self):
        st.set_page_config(
            page_title="Twitter Sentiment Analysis",
            page_icon="🐦",
            layout="wide"
        )
        
        try:
            # Initialize components
            self.storage = StorageManager()
            self.monitor = TrainingMonitor()
            self.load_data()
            
            # Initialize session state
            if 'preprocessed' not in st.session_state:
                st.session_state.preprocessed = False
            if 'models' not in st.session_state:
                st.session_state.models = {}
            if 'preprocessed_data' not in st.session_state:
                st.session_state.preprocessed_data = None
            if 'vectors' not in st.session_state:
                st.session_state.vectors = None
            if 'vector_version' not in st.session_state:
                st.session_state.vector_version = None
            if 'labels' not in st.session_state:
                st.session_state.labels = None
            if 'current_page' not in st.session_state:
                st.session_state.current_page = 'home'
                
        except Exception as e:
            st.error(f"Error initializing application: {str(e)}")
            logger.error(f"Initialization error: {str(e)}")
            
        # Initialize components
        self.storage = StorageManager()
        self.monitor = TrainingMonitor()
        self.experiment_tracker = ExperimentTracker()
        self.load_data()
        
    def load_data(self):
        try:
            self.data = pd.read_csv("data/Dataset Twitter.csv")
            self.analyzer = DataAnalyzer(self.data)
            self.preprocessor = TextPreprocessor()
            st.session_state['data'] = self.data
        except Exception as e:
            st.error(f"Error loading data: {e}")
            
    def show_preprocessing_status(self):
        """Tampilkan status preprocessing di sidebar"""
        st.sidebar.markdown("---")
        st.sidebar.subheader("Preprocessing Status")
        
        if st.session_state.preprocessed and st.session_state.vector_version:
            st.sidebar.success("✅ Data preprocessed")
            st.sidebar.info(f"Version: {st.session_state.vector_version}")
            if st.session_state.vectors is not None:
                st.sidebar.info(f"Vector shape: {st.session_state.vectors.shape}")
        else:
            st.sidebar.warning("⚠️ Data not preprocessed")
            
    def run(self):
        st.title("Twitter Sentiment Analysis - PILPRES 2019 - Tim NLP A")
        
        # Sidebar navigation
        st.sidebar.title("Navigation")
        page = st.sidebar.radio(
            "Menu",
            ["Business Understanding", "Data Analysis", "Preprocessing", 
            "Model Training", "Model Evaluation", "Model Management",
            "Experiment Tracking", "Model Comparison", "Experiment Details"]
        )
        
        # Save current page in session state
        st.session_state.current_page = page
        
        # Page routing
        if page == "Model Training":
            training_page = ModelTrainingPage()
            training_page.run()
        elif page == "Business Understanding":
            self.show_business_understanding()
        elif page == "Data Analysis":
            self.show_data_analysis()
        elif page == "Preprocessing":
            self.show_preprocessing()
        elif page == "Model Evaluation":
            self.show_model_evaluation()
        elif page == "Model Management":
            self.show_model_management()
        elif page == "Experiment Tracking":
            tracking_page = ExperimentTrackingPage()
            tracking_page.run()
        elif page == "Model Comparison":
            comparison_page = ModelComparisonPage()
            comparison_page.run()
        elif page == "Experiment Details":
            details_page = ExperimentDetailsPage()
            details_page.run()
            
    def show_business_understanding(self):
        st.header("Business Understanding")
        
        # Project Overview
        st.write("""
        ## Project Overview
        Proyek ini bertujuan untuk mengembangkan sistem analisis sentimen untuk tweet terkait 
        PILPRES 2019. Sistem akan mengklasifikasikan tweet ke dalam tiga kategori sentimen:
        - Positif
        - Netral
        - Negatif
        
        ## Objectives
        1. Mengembangkan model machine learning yang dapat mengklasifikasikan sentimen tweet dengan akurasi tinggi
        2. Membandingkan performa berbagai teknik preprocessing dan algoritma
        3. Mengoptimalkan model melalui hyperparameter tuning
        
        ## Success Metrics
        - F1-Score (Metrik Utama)
        - Precision & Recall per kelas
        - Confusion Matrix
        - ROC-AUC Score
        """)
        
    def show_data_analysis(self):
        st.header("Data Analysis")
        
        # Tabs for different analyses
        overview_tab, distribution_tab, wordcloud_tab = st.tabs([
            "Dataset Overview", "Distribution Analysis", "Word Cloud Analysis"
        ])
        
        with overview_tab:
            # Basic statistics in metrics
            stats = self.analyzer.get_basic_stats()
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Samples", stats['total_samples'])
            with col2:
                st.metric("Average Tweet Length", f"{stats['avg_tweet_length']:.1f}")
            with col3:
                st.metric("Max Tweet Length", stats['max_tweet_length'])
                
            # Sample data display
            st.subheader("Sample Data")
            st.dataframe(self.data.head())
        
        with distribution_tab:
            col1, col2 = st.columns(2)
            
            with col1:
                # Sentiment distribution
                st.subheader("Sentiment Distribution")
                fig_dist = self.analyzer.plot_sentiment_distribution()
                st.plotly_chart(fig_dist, use_container_width=True)
            
            with col2:
                # Tweet length distribution
                st.subheader("Tweet Length Distribution")
                fig_len = self.analyzer.plot_tweet_length_distribution()
                st.plotly_chart(fig_len, use_container_width=True)
        
        with wordcloud_tab:
            # Word clouds
            st.subheader("Word Clouds by Sentiment")
            sentiment_options = ['All'] + list(self.data['sentimen'].unique())
            selected_sentiment = st.selectbox("Select sentiment", sentiment_options)
            
            if selected_sentiment == 'All':
                fig_cloud = self.analyzer.generate_wordcloud()
            else:
                fig_cloud = self.analyzer.generate_wordcloud(selected_sentiment)
            st.pyplot(fig_cloud)
            
    def show_preprocessing(self):
        preprocessing_page = PreprocessingPage()
        preprocessing_page.run()
                    
    def show_model_training(self):
        """Show model training page"""
        # Langsung redirect ke model training page
        try:
            training_page = ModelTrainingPage()
            training_page.run()
        except Exception as e:
            st.error(f"Error in model training: {str(e)}")
    
    # Add new method
    def show_experiment_status(self):
        """Tampilkan status eksperimen di sidebar"""
        if hasattr(self, 'experiment_tracker'):
            st.sidebar.markdown("---")
            st.sidebar.subheader("Experiment Status")
            
            experiments = self.experiment_tracker.list_experiments()
            if not experiments.empty:
                st.sidebar.success(f"✅ {len(experiments)} experiments tracked")
                st.sidebar.info(f"Best F1: {experiments['f1_macro'].max():.4f}")
            else:
                st.sidebar.info("No experiments tracked yet")
    
    def show_model_evaluation(self):
        st.header("Model Evaluation")
        
        if not st.session_state.preprocessed:
            st.warning("Please complete preprocessing first!")
            return
            
        if not self.storage.list_versions('models'):
            st.warning("No trained models available. Please train a model first!")
            return
        
        # Tabs for different evaluation aspects
        metrics_tab, analysis_tab = st.tabs(["Performance Metrics", "Error Analysis"])
        
        with metrics_tab:
            # Get available models
            model_versions = self.storage.list_versions('models')
            selected_version = st.selectbox(
                "Select model to evaluate",
                [v['version_id'] for v in model_versions]
            )
            
            if st.button("Evaluate Model"):
                try:
                    with st.spinner("Evaluating model..."):
                        # Load model
                        model_path = next(
                            m['path'] for m in model_versions 
                            if m['version_id'] == selected_version
                        )
                        model, config = self.storage.load_model(model_path)
                        
                        if model is None:
                            st.error("Failed to load model")
                            return
                            
                        # Get data
                        vectors = st.session_state.vectors
                        labels = self.data['sentimen'].values
                        
                        # Evaluate
                        results = model.evaluate(vectors, labels)
                        
                        # Display results using ModelVisualizer
                        self.display_evaluation_results(results)
                        
                except Exception as e:
                    st.error(f"Error during evaluation: {e}")
        
        with analysis_tab:
            if st.button("Perform Error Analysis"):
                try:
                    # Load model and perform predictions
                    model_path = next(
                        m['path'] for m in model_versions 
                        if m['version_id'] == selected_version
                    )
                    model, _ = self.storage.load_model(model_path)
                    
                    vectors = st.session_state.vectors
                    labels = self.data['sentimen'].values
                    
                    predictions = model.predict(vectors)
                    
                    # Error Analysis
                    self.display_error_analysis(
                        labels,
                        predictions,
                        self.data['tweet'].values
                    )
                    
                except Exception as e:
                    st.error(f"Error during analysis: {e}")
                    
    def display_evaluation_results(self, results):
        """Display model evaluation results"""
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        metrics = results['metrics']
        
        with col1:
            st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
        with col2:
            st.metric("F1 Score", f"{metrics['f1_macro']:.4f}")
        with col3:
            st.metric("Precision", f"{metrics['precision_macro']:.4f}")
        with col4:
            st.metric("Recall", f"{metrics['recall_macro']:.4f}")
            
        # Classification Report
        st.subheader("Classification Report")
        report_df = pd.DataFrame(results['classification_report']).transpose()
        st.dataframe(report_df)
        
        # Confusion Matrix
        st.subheader("Confusion Matrix")
        fig_cm = ModelVisualizer.plot_confusion_matrix(
            results['confusion_matrix'],
            ['Negative', 'Neutral', 'Positive']
        )
        st.plotly_chart(fig_cm)
        
        # ROC Curves
        st.subheader("ROC Curves")
        fig_roc = ModelVisualizer.plot_roc_curves(
            results['roc_curves']['fpr'],
            results['roc_curves']['tpr'],
            results['roc_curves']['roc_auc'],
            ['Negative', 'Neutral', 'Positive']
        )
        st.plotly_chart(fig_roc)
        
        # Training History if available
        if 'training_history' in results:
            st.subheader("Training History")
            fig_history = ModelVisualizer.plot_learning_curves(
                results['training_history']
            )
            st.plotly_chart(fig_history)
            
    def display_error_analysis(self, true_labels, predicted_labels, texts):
        """Display error analysis results"""
        st.subheader("Error Analysis")
        
        # Create error dataframe
        errors = pd.DataFrame({
            'Text': texts,
            'True Label': true_labels,
            'Predicted': predicted_labels
        })
        errors['Is Error'] = errors['True Label'] != errors['Predicted']
        
        # Error Statistics
        total_samples = len(errors)
        error_count = errors['Is Error'].sum()
        error_rate = error_count / total_samples
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Errors", error_count)
        with col2:
            st.metric("Error Rate", f"{error_rate:.2%}")
            
        # Error Distribution
        st.subheader("Error Distribution")
        error_dist = errors[errors['Is Error']].groupby(
            ['True Label', 'Predicted']
        ).size().reset_index(name='count')
        
        fig_dist = px.bar(
            error_dist,
            x='True Label',
            y='count',
            color='Predicted',
            title='Distribution of Prediction Errors',
            barmode='group'
        )
        st.plotly_chart(fig_dist)
        
        # Error Examples
        st.subheader("Error Examples")
        error_examples = errors[errors['Is Error']].sample(
            min(10, len(errors[errors['Is Error']]))
        )
        for _, row in error_examples.iterrows():
            with st.expander(f"True: {row['True Label']} → Predicted: {row['Predicted']}"):
                st.write(row['Text'])
                
        # Confusion Matrix Heatmap
        st.subheader("Error Pattern Heatmap")
        error_matrix = pd.crosstab(
            errors[errors['Is Error']]['True Label'],
            errors[errors['Is Error']]['Predicted']
        )
        fig_heat = px.imshow(
            error_matrix,
            labels=dict(x="Predicted Label", y="True Label", color="Count"),
            title="Error Pattern Heatmap"
        )
        st.plotly_chart(fig_heat)
            
    def show_model_management(self):
        st.header("Model Management")
        
        # Get all experiments
        experiments = self.storage.get_experiment_results()
        if experiments.empty:
            st.warning("No experiments found!")
            return
            
        # Tabs for different management aspects
        versions_tab, compare_tab, export_tab = st.tabs([
            "Model Versions", "Model Comparison", "Export/Import"
        ])
        
        with versions_tab:
            st.subheader("Available Models")
            
            # Display model versions
            for _, row in experiments.iterrows():
                col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                
                with col1:
                    st.write(f"Version: {row['version_id']}")
                    st.write(f"Model Type: {row['model_type']}")
                    st.write(f"F1 Score: {row['f1_macro']:.4f}")
                
                with col2:
                    if st.button("Details", key=f"details_{row['version_id']}"):
                        st.json(row.to_dict())
                
                with col3:
                    if st.button("Export", key=f"export_{row['version_id']}"):
                        model_path = os.path.join('models', row['version_id'])
                        if os.path.exists(model_path):
                            self.storage.export_model(model_path, 'exports')
                            st.success("Model exported successfully!")
                        else:
                            st.error("Model files not found!")
                
                with col4:
                    if st.button("Delete", key=f"delete_{row['version_id']}"):
                        self.storage.delete_version(row['version_id'], 'models')
                        st.success("Model deleted successfully!")
                        st.experimental_rerun()
                        
                st.divider()
        
        with compare_tab:
            st.subheader("Model Comparison")
            
            # Select models to compare
            selected_versions = st.multiselect(
                "Select models to compare",
                experiments['version_id'].unique(),
                default=experiments['version_id'].unique()[:2] if len(experiments) >= 2 else []
            )
            
            if len(selected_versions) >= 2:
                comparison_data = experiments[
                    experiments['version_id'].isin(selected_versions)
                ]
                
                # Metrics Comparison
                st.subheader("Metrics Comparison")
                metrics_cols = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']
                
                fig_metrics = go.Figure()
                for metric in metrics_cols:
                    fig_metrics.add_trace(
                        go.Bar(
                            name=metric.replace('_', ' ').title(),
                            x=comparison_data['version_id'],
                            y=comparison_data[metric]
                        )
                    )
                    
                fig_metrics.update_layout(
                    title="Model Performance Comparison",
                    barmode='group',
                    yaxis=dict(title='Score', range=[0, 1])
                )
                st.plotly_chart(fig_metrics)
                
                # Parameter Comparison
                st.subheader("Parameter Comparison")
                param_cols = [col for col in comparison_data.columns if col.startswith('param_')]
                if param_cols:
                    st.dataframe(comparison_data[['version_id', 'model_type'] + param_cols])
                
                # Export Comparison
                if st.button("Export Comparison"):
                    comparison_path = f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                    comparison_data.to_csv(comparison_path, index=False)
                    st.success(f"Comparison exported to {comparison_path}")
        
        with export_tab:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Export Experiments")
                if st.button("Export All Experiments"):
                    try:
                        export_path = f"experiments_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                        self.storage.export_experiment(export_path)
                        st.success(f"Experiments exported to {export_path}")
                    except Exception as e:
                        st.error(f"Error exporting experiments: {e}")
            
            with col2:
                st.subheader("Import Model")
                uploaded_file = st.file_uploader("Upload model file", type=['joblib', 'h5'])
                if uploaded_file is not None:
                    try:
                        # Save uploaded file
                        model_path = os.path.join('imports', uploaded_file.name)
                        os.makedirs('imports', exist_ok=True)
                        with open(model_path, 'wb') as f:
                            f.write(uploaded_file.getbuffer())
                            
                        # Import model
                        new_path = self.storage.import_model(model_path)
                        st.success(f"Model imported successfully to {new_path}")
                        
                        # Cleanup
                        os.remove(model_path)
                    except Exception as e:
                        st.error(f"Error importing model: {e}")

if __name__ == "__main__":
    app = SentimentAnalysisApp()
    app.run()