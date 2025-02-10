import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import os
import glob
from typing import Dict, Any, Optional, Tuple, List
import json

# Custom imports
from preprocessing.text_preprocessor import TextPreprocessor
from utils.storage_manager import StorageManager
from utils.visualization import ModelVisualizer
from utils.preprocessing_comparison import (
    compare_preprocessing_methods, 
    visualize_comparison,
    load_and_compare_preprocessed
)

# If you're using plotting
import plotly.graph_objects as go
import plotly.express as px

# For type hints
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from gensim.models import Word2Vec
from utils.vector_storage_manager import VectorStorageManager



class PreprocessingPage:
    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.storage = StorageManager()
        self.vector_storage = VectorStorageManager()
        
        # Initialize session state
        if 'preprocessed_data' not in st.session_state:
            st.session_state.preprocessed_data = None
        if 'vectors' not in st.session_state:
            st.session_state.vectors = None
        if 'vector_version' not in st.session_state:
            st.session_state.vector_version = None
            
    def show_method_comparison(self):
        """Tampilkan perbandingan metode preprocessing"""
        from utils.preprocessing_comparison import load_and_compare_preprocessed
        try:
            load_and_compare_preprocessed()
        except Exception as e:
            st.error(f"Error comparing preprocessing methods: {str(e)}")
            
    def load_preprocessed_data(self):
        try:
            # Get available versions
            processed_files = glob.glob('data/processed/preprocessed_*.npz')
            if not processed_files:
                st.warning("No preprocessed data found.")
                return False
                
            versions = [os.path.basename(f).replace('preprocessed_', '').replace('.npz', '')
                    for f in processed_files]
            
            selected_version = st.selectbox("Select version to load:", versions)
            
            if st.button("Load Selected Version"):
                # Validasi data
                if not self.storage.verify_vectors(selected_version):
                    st.error("Data validation failed. Some files may be missing or corrupted.")
                    return False
                    
                # Load data
                texts, preprocess_metadata = self.storage.load_preprocessed(selected_version)
                vectors, vector_metadata = self.storage.load_vectorized(selected_version)
                
                if texts is None or vectors is None:
                    st.error("Failed to load data.")
                    return False
                    
                # Store in session state
                st.session_state.preprocessed_data = texts
                st.session_state.vectors = vectors
                st.session_state.vector_version = selected_version
                st.session_state.preprocessing_config = preprocess_metadata.get('config', {})
                
                # Display info
                st.write("### Data Information")
                st.write(f"Version ID: {selected_version}")
                st.write(f"Number of samples: {len(texts)}")
                st.write(f"Vector shape: {vectors.shape}")
                if vector_metadata:
                    st.write(f"Vectorization method: {vector_metadata.get('method', 'unknown')}")
                    
                return True
                
            return False
            
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return False
            
    def preprocess_new_data(self):
        """Preprocess new data with selected options"""
        preprocessing_options = {
            'remove_mentions': st.checkbox('Remove Mentions', value=True),
            'remove_urls': st.checkbox('Remove URLs', value=True),
            'remove_hashtags': st.checkbox('Remove Hashtags', value=True),
            'remove_numbers': st.checkbox('Remove Numbers', value=True),
            'remove_special_chars': st.checkbox('Remove Special Characters', value=True),
            'remove_emojis': st.checkbox('Remove Emojis', value=True),
            'case_folding': st.checkbox('Case Folding', value=True),
            'remove_stopwords': st.checkbox('Remove Stopwords', value=True),
            'stemming': st.checkbox('Stemming', value=True)
        }
        
        vectorization_method = st.selectbox(
            "Choose vectorization method",
            ['tfidf', 'word2vec', 'bert']
        )
        
        if st.button("Apply Preprocessing"):
            with st.spinner("Preprocessing data..."):
                data = pd.read_csv('data/Dataset Twitter.csv')
                texts = data['tweet'].tolist()
                labels = data['sentimen'].tolist()
                
                # Lakukan preprocessing dan vectorization
                vectors, version_id = self.preprocessor.process_and_vectorize(
                    texts,
                    preprocessing_options,
                    vectorization_method
                )
                
                # Simpan ke session state
                st.session_state.preprocessed_data = True
                st.session_state.vectors = vectors
                st.session_state.vector_version = version_id
                st.session_state.labels = labels
                
                # Tampilkan hasil preprocessing
                if hasattr(self.preprocessor, 'current_stats'):
                    self.show_preprocessing_visualization(self.preprocessor.current_stats)
                
                st.success(f"Preprocessing completed! Version ID: {version_id}")
                return True
                
    def show_preprocessing_visualization(self, stats_df: pd.DataFrame):
        """Menampilkan visualisasi hasil preprocessing"""
        if stats_df is None or stats_df.empty:
            st.warning("No preprocessing statistics available")
            return
            
        # Tampilkan tabel statistik
        st.dataframe(stats_df)
        
        # Buat visualisasi
        col1, col2 = st.columns(2)
        
        with col1:
            fig1 = go.Figure()
            fig1.add_trace(go.Bar(
                x=stats_df['Stage'],
                y=stats_df['Avg Length'],
                name='Average Text Length'
            ))
            fig1.update_layout(
                title='Average Text Length per Stage',
                xaxis_title='Processing Stage',
                yaxis_title='Average Length',
                xaxis={'tickangle': 45}
            )
            st.plotly_chart(fig1)
            
        with col2:
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(
                x=stats_df['Stage'],
                y=stats_df['Vocab Size'],
                name='Vocabulary Size'
            ))
            fig2.update_layout(
                title='Vocabulary Size per Stage',
                xaxis_title='Processing Stage',
                yaxis_title='Vocabulary Size',
                xaxis={'tickangle': 45}
            )
            st.plotly_chart(fig2)

    def run(self):
        st.title("Text Preprocessing")
        
        # Create tabs for different preprocessing views
        preprocess_tab, visual_tab, compare_tab = st.tabs([  # Perbaikan di sini
            "Preprocessing Steps", 
            "Preprocessing Visualization",
            "Method Comparison"
        ])
        
        with preprocess_tab:
            # Create sub-tabs for loading/new preprocessing
            load_tab, new_tab = st.tabs(["Load Preprocessed Data", "New Preprocessing"])
            
            with load_tab:
                self.load_preprocessed_data()
                
            with new_tab:
                self.preprocess_new_data()
                
        with visual_tab:
            if hasattr(self.preprocessor, 'current_stats'):
                self.show_preprocessing_visualization(self.preprocessor.current_stats)
            else:
                st.warning("Please run preprocessing first!")
                
        with compare_tab:
            self.show_method_comparison()