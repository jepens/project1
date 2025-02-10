import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime
import json
import os
import glob
from sklearn.model_selection import train_test_split

from models.random_forest_model import RandomForestModel
from models.lstm_model import LSTMModel
from models.distilbert_model import DistilBERTModel
from preprocessing.text_preprocessor import TextPreprocessor
from utils.storage_manager import StorageManager
from utils.visualization import ModelVisualizer
from utils.training_monitor import TrainingMonitor, TrainingCallback

import scipy.sparse
from experiments.experiment_tracker import ExperimentTracker 

class ModelTrainingPage:
    def __init__(self):
        self.storage = StorageManager()
        self.preprocessor = TextPreprocessor()
        self.monitor = TrainingMonitor()
        self.experiment_tracker = ExperimentTracker(base_dir='experiments') 
        
        # Initialize session state if not exists
        if 'initialized' not in st.session_state:
            st.session_state.initialized = True
            st.session_state.data = None
            st.session_state.preprocessed = False
            st.session_state.preprocessed_data = None  # Tambahkan ini
            st.session_state.vectors = None
            st.session_state.vector_version = None
            st.session_state.labels = None
            st.session_state.training_started = False
            st.session_state.current_model = None
            
        self.models_config = {
            "Random Forest": {
                "class": RandomForestModel,
                "params": {
                    "n_estimators": {"min": 100, "max": 1000, "default": 500, "step": 50},
                    "max_depth": {"min": 10, "max": 100, "default": 50, "step": 10},
                    "min_samples_split": {"min": 2, "max": 10, "default": 5, "step": 1},
                    "min_samples_leaf": {"min": 1, "max": 4, "default": 2, "step": 1},
                    "class_weight": {"options": ["balanced", "balanced_subsample", None], "default": "balanced"}
                }
            },
            "LSTM": {
                "class": LSTMModel,
                "params": {
                    "lstm_units": {"min": 32, "max": 256, "default": 128, "step": 32},
                    "num_layers": {"min": 1, "max": 3, "default": 2, "step": 1},
                    "dropout": {"min": 0.1, "max": 0.5, "default": 0.3, "step": 0.1},
                    "learning_rate": {"options": [1e-5, 5e-5, 1e-4, 5e-4, 1e-3], "default": 1e-4},
                    "bidirectional": {"options": [True, False], "default": True}
                }
            },
            "DistilBERT": {
                "class": DistilBERTModel,
                "params": {
                    "learning_rate": {"options": [1e-5, 2e-5, 5e-5, 1e-4], "default": 2e-5},
                    "batch_size": {"options": [8, 16, 32], "default": 16},
                    "warmup_steps": {"min": 100, "max": 1000, "default": 500, "step": 100},
                    "weight_decay": {"options": [1e-5, 1e-4, 1e-3], "default": 1e-4},
                    "dropout": {"min": 0.1, "max": 0.5, "default": 0.1, "step": 0.1}
                }
            }
        }
        
    def check_or_load_preprocessing(self) -> bool:
        """Check for preprocessed data or load if available"""
        if st.session_state.get('preprocessed_data') and st.session_state.get('vectors') is not None and st.session_state.get('labels') is not None:
            return True
            
        # If no preprocessed data in session, show load option
        st.warning("No preprocessed data loaded. Please load existing preprocessed data.")
        
        try:
            # Check for preprocessed files
            processed_files = glob.glob('data/processed/*.npz')
            if not processed_files:
                st.error("No preprocessed data found. Please preprocess data first.")
                return False
                
            # Create selectbox for available preprocessed files
            selected_file = st.selectbox(
                "Select preprocessed data to load",
                processed_files,
                format_func=lambda x: os.path.basename(x)
            )
            
            if st.button("Load Selected Data"):
                with st.spinner("Loading preprocessed data..."):
                    # Load the preprocessed data
                    data = np.load(selected_file, allow_pickle=True)
                    
                    # Verify data integrity
                    if 'vectors' not in data or 'metadata' not in data:
                        st.error("Invalid preprocessed data format")
                        return False
                        
                    # Store in session state
                    st.session_state.preprocessed_data = True
                    st.session_state.vectors = data['vectors']
                    st.session_state.vector_version = str(data['metadata'].item().get('version_id', ''))
                    
                    # Load original CSV to get labels
                    csv_data = pd.read_csv('data/Dataset Twitter.csv')
                    st.session_state.labels = csv_data['sentimen'].values
                    st.session_state.preprocessed = True
                    
                    # Verify data consistency
                    if len(st.session_state.vectors) != len(st.session_state.labels):
                        st.error(f"Data inconsistency: vectors ({len(st.session_state.vectors)}) and labels ({len(st.session_state.labels)}) have different lengths")
                        return False
                    
                    st.success(f"Successfully loaded preprocessed data from {os.path.basename(selected_file)}")
                    return True
                    
            return False
        
        except Exception as e:
            st.error(f"Error loading preprocessed data: {str(e)}")
            return False
        
    def preprocess_data(self) -> bool:
        """Preprocess data if not already done"""
        try:
            # Load data if not already loaded
            if st.session_state.data is None:
                try:
                    # Read the CSV file
                    data = pd.read_csv('data/Dataset Twitter.csv')
                    st.session_state.data = data
                except Exception as e:
                    st.error(f"Error loading data: {e}")
                    return False
            
            if not st.session_state.preprocessed:
                try:
                    with st.spinner("Preprocessing data..."):
                        # Get preprocessing options
                        preprocessing_options = {
                            'remove_mentions': True,
                            'remove_urls': True,
                            'remove_hashtags': True,
                            'remove_numbers': True,
                            'remove_special_chars': True,
                            'remove_emojis': True,
                            'case_folding': True,
                            'remove_stopwords': True,
                            'stemming': True
                        }
                        
                        # Get vectorization method
                        vectorization_method = 'tfidf'
                        
                        # Process and vectorize texts
                        texts = st.session_state.data['tweet'].tolist()
                        labels = st.session_state.data['sentimen'].tolist()
                        
                        vectors, vector_version = self.preprocessor.process_and_vectorize(
                            texts,
                            preprocessing_options,
                            vectorization_method,
                            "Preprocessing for model training"
                        )
                        
                        # Save to session state
                        st.session_state.preprocessed = True
                        st.session_state.vectors = vectors
                        st.session_state.vector_version = vector_version
                        st.session_state.labels = labels
                        
                        st.success("Preprocessing completed successfully!")
                        return True
                        
                except Exception as e:
                    st.error(f"Error during preprocessing: {e}")
                    return False
                    
            return True
            
        except Exception as e:
            st.error(f"Error in preprocess_data: {e}")
            return False
        
    def create_parameter_inputs(self, model_type: str) -> Dict[str, Any]:
        """Create input widgets for model parameters"""
        st.subheader("Model Parameters")
        params = {}
        
        try:
            model_params = self.models_config[model_type]["params"]
            col1, col2 = st.columns(2)
            
            # Split parameters between columns
            param_items = list(model_params.items())
            mid_point = len(param_items) // 2
            
            with col1:
                for param_name, param_config in param_items[:mid_point]:
                    if "options" in param_config:
                        params[param_name] = st.selectbox(
                            f"{param_name}",
                            options=param_config["options"],
                            index=param_config["options"].index(param_config["default"]),
                            key=f"param_{param_name}_col1"
                        )
                    else:
                        params[param_name] = st.slider(
                            f"{param_name}",
                            min_value=param_config["min"],
                            max_value=param_config["max"],
                            value=param_config["default"],
                            step=param_config["step"],
                            key=f"param_{param_name}_col1"
                        )
                        
            with col2:
                for param_name, param_config in param_items[mid_point:]:
                    if "options" in param_config:
                        params[param_name] = st.selectbox(
                            f"{param_name}",
                            options=param_config["options"],
                            index=param_config["options"].index(param_config["default"]),
                            key=f"param_{param_name}_col2"
                        )
                    else:
                        params[param_name] = st.slider(
                            f"{param_name}",
                            min_value=param_config["min"],
                            max_value=param_config["max"],
                            value=param_config["default"],
                            step=param_config["step"],
                            key=f"param_{param_name}_col2"
                        )
                        
        except Exception as e:
            st.error(f"Error creating parameter inputs: {str(e)}")
            params = {}
            
        return params
        
    def create_training_options(self) -> Dict[str, Any]:
        """Create training option inputs"""
        st.subheader("Training Options")
        options = {}
        
        try:
            col1, col2 = st.columns(2)
            
            with col1:
                options["use_hyperopt"] = st.checkbox("Use Hyperparameter Optimization", value=False)
                if options["use_hyperopt"]:
                    options["n_trials"] = st.slider("Number of trials", 10, 100, 50)
                else:
                    options["n_trials"] = None
                    
            with col2:
                options["validation_split"] = st.slider("Validation Split", 0.1, 0.3, 0.2)
                options["experiment_name"] = st.text_input(
                    "Experiment Name",
                    value=f"Experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                )
                
        except Exception as e:
            st.error(f"Error creating training options: {str(e)}")
            options = {
                "use_hyperopt": False,
                "n_trials": None,
                "validation_split": 0.2,
                "experiment_name": f"Default_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            }
            
        return options
    
    def train_model(self, model_type: str, params: Dict[str, Any], training_options: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        try:
            # Start experiment tracking
            experiment_id = self.experiment_tracker.start_experiment(
                name=training_options['experiment_name'],
                model_type=model_type,
                hyperparameters=params
            )

            # Initialize model
            model_class = self.models_config[model_type]["class"]
            model = model_class()
            st.session_state.current_model = model
            
            # Ensure data is loaded
            if st.session_state.vectors is None or st.session_state.labels is None:
                raise ValueError("Data not properly loaded. Please ensure preprocessing step is completed.")
                
            # Convert labels to numpy array and ensure it's not empty
            labels = np.array(st.session_state.labels)
            if len(labels) == 0:
                raise ValueError("Labels array is empty")
                
            # Ensure vectors and labels have same number of samples
            if len(st.session_state.vectors) != len(labels):
                raise ValueError(f"Mismatch between number of samples in vectors ({len(st.session_state.vectors)}) and labels ({len(labels)})")
                
            # Get features
            X = st.session_state.vectors
            y = labels
            
            # Split data
            train_size = 1 - training_options['validation_split']
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, 
                train_size=train_size,
                stratify=y,
                random_state=42
            )
            
            # Convert sparse matrix to array if needed
            if scipy.sparse.issparse(X_train):
                X_train = X_train.toarray()
                X_val = X_val.toarray()
            
            # Initialize progress tracking
            st.write("### Training Progress")
            progress_container = st.container()
            
            with progress_container:
                if training_options['use_hyperopt']:
                    st.write("#### Hyperparameter Optimization")
                    params = model.optimize_hyperparameters(X_train, y_train, training_options['n_trials'])
                    st.success("Hyperparameter optimization completed!")
                    st.write("Best parameters:", params)
                
                st.write("#### Model Training")
                progress_bar = st.progress(0)
                metrics_container = st.empty()
                plot_container = st.empty()
                
                # Create callback
                callback = lambda metrics: TrainingCallback(self.monitor).on_epoch_end(metrics)
                
                # Train model
                metrics = model.train(
                    X_train,
                    y_train,
                    validation_data=(X_val, y_val),
                    params=params,
                    callbacks=[callback]
                )
                
                # Log metrics during training
                metrics_json = {}
                for key, value in metrics.items():
                    if isinstance(value, np.ndarray):
                        metrics_json[key] = value.tolist()
                    else:
                        metrics_json[key] = value
                self.experiment_tracker.log_metrics(metrics_json)
                
                # Evaluate model
                eval_results = model.evaluate(X_val, y_val)

                # Convert numpy arrays in evaluation results
                eval_results_json = {
                    'metrics': {},
                    'classification_report': eval_results['classification_report'],
                    'confusion_matrix': eval_results['confusion_matrix'].tolist(),
                    'roc_curves': {
                        'fpr': {k: v.tolist() for k, v in eval_results['roc_curves']['fpr'].items()},
                        'tpr': {k: v.tolist() for k, v in eval_results['roc_curves']['tpr'].items()},
                        'roc_auc': eval_results['roc_curves']['roc_auc']
                    }
                }

                # Convert metrics
                for key, value in eval_results['metrics'].items():
                    if isinstance(value, np.ndarray):
                        eval_results_json['metrics'][key] = value.tolist()
                    else:
                        eval_results_json['metrics'][key] = value

                # Log feature importance if available
                if hasattr(model, 'feature_importances_'):
                    feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
                    self.experiment_tracker.log_feature_importance(
                        feature_names,
                        model.feature_importances_.tolist()  # Convert to list
                    )

                # Save model
                model_dir, version_id = model.save(
                    training_options['experiment_name'],
                    params,
                    eval_results_json['metrics']  # Use converted metrics
                )

                # End experiment tracking
                self.experiment_tracker.end_experiment(
                    final_metrics=eval_results_json['metrics'],
                    status='completed'
                )
                
                return model_dir, eval_results_json
                
        except Exception as e:
            # Log failed experiment
            if 'experiment_id' in locals():
                self.experiment_tracker.end_experiment(
                    final_metrics={},
                    status='failed'
                )
            st.error(f"Error during training: {str(e)}")
            st.error(f"Detailed error: {str(e.__class__.__name__)}: {str(e)}")
            raise
            
    def display_evaluation_results(self, eval_results: Dict[str, Any]):
        """Display model evaluation results"""
        try:
            st.subheader("Evaluation Results")
            
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            metrics = eval_results.get('metrics', {})
            
            with col1:
                st.metric("Accuracy", f"{metrics.get('accuracy', 0):.4f}")
            with col2:
                st.metric("Precision", f"{metrics.get('precision_macro', 0):.4f}")
            with col3:
                st.metric("Recall", f"{metrics.get('recall_macro', 0):.4f}")
            with col4:
                st.metric("F1 Score", f"{metrics.get('f1_macro', 0):.4f}")
                
            # Display confusion matrix - convert back to numpy if needed
            st.subheader("Confusion Matrix")
            confusion_matrix = np.array(eval_results.get('confusion_matrix', []))
            fig_cm = ModelVisualizer.plot_confusion_matrix(
                confusion_matrix,
                ['Negative', 'Neutral', 'Positive']
            )
            st.plotly_chart(fig_cm, use_container_width=True)
            
            # Display ROC curves
            st.subheader("ROC Curves")
            roc_curves = eval_results.get('roc_curves', {})
            if roc_curves:
                # Convert back to numpy for visualization if needed
                fpr = {k: np.array(v) for k, v in roc_curves.get('fpr', {}).items()}
                tpr = {k: np.array(v) for k, v in roc_curves.get('tpr', {}).items()}
                fig_roc = ModelVisualizer.plot_roc_curves(
                    fpr,
                    tpr,
                    roc_curves.get('roc_auc', {}),
                    ['Negative', 'Neutral', 'Positive']
                )
                st.plotly_chart(fig_roc, use_container_width=True)
                
            # Display classification report
            st.subheader("Classification Report")
            report = eval_results.get('classification_report', {})
            if report:
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df)
                
        except Exception as e:
            st.error(f"Error displaying evaluation results: {str(e)}")
            
    def run(self):
        st.title("Model Training and Hyperparameter Tuning")
        
        # Check or load preprocessed data
        if not self.check_or_load_preprocessing():
            return
            
        # Model selection
        model_type = st.selectbox(
            "Select Model",
            list(self.models_config.keys())
        )
        
        # Get model parameters
        params = self.create_parameter_inputs(model_type)
        
        # Get training options
        training_options = self.create_training_options()
        
        if st.button("Start Training"):
            with st.spinner("Training model..."):
                try:
                    # Train model
                    model_dir, eval_results = self.train_model(
                        model_type,
                        params,
                        training_options
                    )
                    
                    # Display success message
                    st.success(f"Training completed! Model saved in {model_dir}")
                    
                    # Display evaluation results
                    self.display_evaluation_results(eval_results)
                    
                except Exception as e:
                    st.error(f"Training failed: {str(e)}")

if __name__ == "__main__":
    page = ModelTrainingPage()
    page.run()