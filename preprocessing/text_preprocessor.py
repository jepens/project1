import re
import os
import emoji
import time
import streamlit as st
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from transformers import AutoTokenizer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from gensim.models import Word2Vec
import multiprocessing
from itertools import product
import plotly.express as px
import plotly.graph_objects as go
from sklearn.manifold import TSNE
from typing import List, Dict, Any, Tuple, Optional, Iterator
from utils.storage_manager import StorageManager
import logging
from datetime import datetime
from utils.vector_storage_manager import VectorStorageManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TextPreprocessor:
    def __init__(self):
        # Initialize Sastrawi
        self.stemmer = StemmerFactory().create_stemmer()
        self.stopword_remover = StopWordRemoverFactory().create_stop_word_remover()
        
        # Initialize tokenizers
        self.bert_tokenizer = AutoTokenizer.from_pretrained("indobenchmark/indobert-base-p2")
        
        # Initialize vectorizers
        self.tfidf_vectorizer = TfidfVectorizer()
        
        # Initialize Word2Vec
        self.word2vec_model = None
        self.WORD2VEC_AVAILABLE = False
        self.word2vec_dim = 100  # Default dimension
        
        # Create model directory
        os.makedirs('models/word2vec', exist_ok=True)
        
         # Try to load or initialize Word2Vec model
        try:
            model_path = os.path.join('models', 'word2vec', 'custom_word2vec.model')
            if os.path.exists(model_path):
                self.word2vec_model = Word2Vec.load(model_path)
                self.WORD2VEC_AVAILABLE = True
                self.word2vec_dim = self.word2vec_model.vector_size
                logger.info("Loaded existing Word2Vec model")
            else:
                # Initialize new model with sample data
                logger.info("Initializing new Word2Vec model...")
                texts = pd.read_csv("data/Dataset Twitter.csv")['tweet'].tolist()
                tokenized_texts = [str(text).split() for text in texts]
                self.word2vec_model = Word2Vec(
                    sentences=tokenized_texts,
                    vector_size=self.word2vec_dim,
                    window=5,
                    min_count=1,
                    workers=multiprocessing.cpu_count(),
                    sg=1
                )
                self.WORD2VEC_AVAILABLE = True
                self.word2vec_model.save(model_path)
                logger.info("New Word2Vec model initialized and saved")
        except Exception as e:
            logger.error(f"Error initializing Word2Vec: {e}")
        
        # Grid Search Parameters
        self.grid_search_params = {
            'vector_size': [100, 200, 300],
            'window': [3, 5, 7],
            'min_count': [1, 2, 3],
            'sg': [0, 1],  # 0 for CBOW, 1 for Skip-gram
            'workers': [multiprocessing.cpu_count()],
            'epochs': [5, 10, 15]
        }
        
        # Initialize storage and results
        self.storage = StorageManager()
        self.vectorization_results = {}
        self.vector_storage = VectorStorageManager()
        
    
    def grid_search_generator(self) -> Iterator[Dict[str, Any]]:
        """Generate combinations of parameters for grid search"""
        keys = self.grid_search_params.keys()
        values = self.grid_search_params.values()
        for instance in product(*values):
            yield dict(zip(keys, instance))
            
    def train_word2vec(self, texts: List[str], params: Optional[Dict[str, Any]] = None,
                      grid_search: bool = False) -> Dict[str, Any]:
        """Train Word2Vec model with optional grid search"""
        try:
            # Tokenize texts
            tokenized_texts = [text.split() for text in texts]
            
            if grid_search:
                st.info("Starting Grid Search for Word2Vec parameters...")
                best_score = float('-inf')
                best_params = None
                results = []
                
                progress_bar = st.progress(0)
                total_combinations = np.prod([len(v) for v in self.grid_search_params.values()])
                
                for i, params in enumerate(self.grid_search_generator()):
                    # Update progress
                    progress = (i + 1) / total_combinations
                    progress_bar.progress(progress)
                    
                    # Train model with current parameters
                    model = Word2Vec(
                        sentences=tokenized_texts,
                        **params
                    )
                    
                    # Evaluate model
                    score = self.evaluate_word2vec(model, tokenized_texts)
                    results.append({
                        **params,
                        'score': score
                    })
                    
                    # Update best model
                    if score > best_score:
                        best_score = score
                        best_params = params
                        self.word2vec_model = model
                        
                # Create results DataFrame
                results_df = pd.DataFrame(results)
                st.write("Grid Search Results:")
                st.dataframe(results_df)
                
                # Visualize parameter importance
                self.plot_parameter_importance(results_df)
                
                params = best_params
                st.success(f"Best parameters found: {best_params}")
                
            else:
                # Use provided or default parameters
                if params is None:
                    params = {
                        'vector_size': self.word2vec_dim,
                        'window': 5,
                        'min_count': 1,
                        'workers': multiprocessing.cpu_count(),
                        'sg': 1
                    }
                    
                st.info("Training Word2Vec model...")
                self.word2vec_model = Word2Vec(
                    sentences=tokenized_texts,
                    **params
                )
                
            self.WORD2VEC_AVAILABLE = True
            self.word2vec_dim = params['vector_size']
            
            # Save model
            model_path = os.path.join('models', 'word2vec', 'custom_word2vec.model')
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            self.word2vec_model.save(model_path)
            
            # Visualize embeddings
            self.visualize_embeddings()
            
            return params
            
        except Exception as e:
            st.error(f"Error training Word2Vec model: {str(e)}")
            self.WORD2VEC_AVAILABLE = False
            return {}
        
    def evaluate_word2vec(self, model: Word2Vec, tokenized_texts: List[List[str]]) -> float:
        """Evaluate Word2Vec model using various metrics"""
        try:
            # Calculate vocabulary coverage
            vocab_size = len(model.wv.key_to_index)
            total_words = sum(len(text) for text in tokenized_texts)
            unique_words = len(set([word for text in tokenized_texts for word in text]))
            coverage = vocab_size / unique_words if unique_words > 0 else 0
            
            # Calculate average vector norm (as a proxy for vector quality)
            vector_norms = np.mean([np.linalg.norm(model.wv[word]) 
                                  for word in model.wv.key_to_index])
            
            # Combine metrics
            score = (coverage + vector_norms / 100) / 2  # Normalize and combine
            
            return score
            
        except Exception as e:
            st.warning(f"Error evaluating model: {str(e)}")
            return 0.0
            
    def plot_parameter_importance(self, results_df: pd.DataFrame):
        """Visualize impact of different parameters on model performance"""
        st.subheader("Parameter Importance Analysis")
        
        # Remove non-parameter columns
        param_cols = [col for col in results_df.columns if col != 'score']
        
        for param in param_cols:
            if len(results_df[param].unique()) > 1:  # Only plot if parameter varies
                fig = px.box(results_df, x=param, y='score',
                           title=f'Impact of {param} on Model Performance')
                st.plotly_chart(fig)
                
    def visualize_embeddings(self, n_words: int = 100):
        """Visualize word embeddings using t-SNE"""
        if not self.WORD2VEC_AVAILABLE or self.word2vec_model is None:
            st.warning("No Word2Vec model available for visualization")
            return
            
        try:
            # Get most common words and their vectors
            words = list(self.word2vec_model.wv.key_to_index.keys())[:n_words]
            vectors = [self.word2vec_model.wv[word] for word in words]
            
            # Apply t-SNE
            tsne = TSNE(n_components=2, random_state=42)
            vectors_2d = tsne.fit_transform(vectors)
            
            # Create DataFrame for plotting
            df = pd.DataFrame({
                'word': words,
                'x': vectors_2d[:, 0],
                'y': vectors_2d[:, 1]
            })
            
            # Create scatter plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df['x'],
                y=df['y'],
                mode='markers+text',
                text=df['word'],
                textposition="top center",
                hoverinfo='text'
            ))
            
            fig.update_layout(
                title="Word Embeddings Visualization (t-SNE)",
                xaxis_title="t-SNE dimension 1",
                yaxis_title="t-SNE dimension 2",
                showlegend=False
            )
            
            st.plotly_chart(fig)
            
            # Show similar words for selected word
            selected_word = st.selectbox("Select word to find similar words:",
                                       options=words)
            
            if selected_word:
                similar_words = self.word2vec_model.wv.most_similar(selected_word)
                st.write(f"Words most similar to '{selected_word}':")
                for word, score in similar_words:
                    st.write(f"- {word}: {score:.4f}")
                    
        except Exception as e:
            st.error(f"Error visualizing embeddings: {str(e)}")
        
    def remove_mentions(self, text: str) -> str:
        """Remove mentions (@username) from text"""
        return re.sub(r'@\w+', '', text)
    
    def remove_urls(self, text: str) -> str:
        """Remove URLs from text"""
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        return url_pattern.sub('', text)
    
    def remove_hashtags(self, text: str) -> str:
        """Remove hashtags from text"""
        return re.sub(r'#\w+', '', text)
    
    def remove_numbers(self, text: str) -> str:
        """Remove numbers from text"""
        return re.sub(r'\d+', '', text)
    
    def remove_special_chars(self, text: str) -> str:
        """Remove special characters"""
        return re.sub(r'[^\w\s]', '', text)
    
    def remove_emojis(self, text: str) -> str:
        """Remove emojis from text"""
        return emoji.replace_emoji(text, '')
    
    def case_folding(self, text: str) -> str:
        """Convert text to lowercase"""
        return text.lower()
    
    def remove_stopwords(self, text: str) -> str:
        """Remove stopwords using Sastrawi"""
        try:
            return self.stopword_remover.remove(text)
        except Exception as e:
            print(f"Error removing stopwords: {e}")
            return text
    
    def stemming(self, text: str) -> str:
        """Apply stemming using Sastrawi"""
        try:
            return self.stemmer.stem(text)
        except Exception as e:
            print(f"Error in stemming: {e}")
            return text
    
    def clean_text(self, text: str, options: Optional[Dict[str, bool]] = None) -> str:
        if text is None:
            return ""
            
        try:
            # Convert to string if not already
            text = str(text)
            
            if options is None:
                options = {
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

            # Eksekusi preprocessing sesuai opsi yang dipilih
            if options.get('case_folding', True):
                text = self.case_folding(text)
            if options.get('remove_mentions', True):
                text = self.remove_mentions(text)
            if options.get('remove_urls', True):
                text = self.remove_urls(text)
            if options.get('remove_hashtags', True):
                text = self.remove_hashtags(text)
            if options.get('remove_numbers', True):
                text = self.remove_numbers(text)
            if options.get('remove_special_chars', True):
                text = self.remove_special_chars(text)
            if options.get('remove_emojis', True):
                text = self.remove_emojis(text)
            if options.get('remove_stopwords', True):
                text = self.remove_stopwords(text)
            if options.get('stemming', True):
                text = self.stemming(text)
                
            return text.strip()
            
        except Exception as e:
            logger.error(f"Error in clean_text: {e}")
            return ""
    
    def get_word2vec_embedding(self, text: str) -> np.ndarray:
        if not self.word2vec_model or not self.WORD2VEC_AVAILABLE:
            logger.error("Word2Vec model not available")
            return np.zeros(self.word2vec_dim)
            
        words = text.split()
        if not words:
            return np.zeros(self.word2vec_dim)  # Use instance dimension
            
        word_vectors = []
        for word in words:
            try:
                if word in self.word2vec_model.wv:
                    vector = self.word2vec_model.wv[word]
                    word_vectors.append(vector)
            except KeyError:
                continue
                
        if not word_vectors:
            return np.zeros(self.word2vec_dim)
            
        return np.mean(word_vectors, axis=0)
    
        # Tambahkan method baru:
    def reinitialize_word2vec(self, texts: List[str]) -> bool:
        """Reinitialize Word2Vec model with new texts"""
        try:
            tokenized_texts = [text.split() for text in texts]
            self.word2vec_model = Word2Vec(
                sentences=tokenized_texts,
                vector_size=self.word2vec_dim,
                window=5,
                min_count=1,
                workers=multiprocessing.cpu_count(),
                sg=1
            )
            self.WORD2VEC_AVAILABLE = True
            
            # Save model
            model_path = os.path.join('models', 'word2vec', 'custom_word2vec.model')
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            self.word2vec_model.save(model_path)
            logger.info("Word2Vec model reinitialized successfully")
            return True
        except Exception as e:
            logger.error(f"Error reinitializing Word2Vec: {e}")
            self.WORD2VEC_AVAILABLE = False
            return False
            
    def vectorize_text(self, texts: List[str], method: str = 'tfidf', fit: bool = True) -> np.ndarray:
        try:
            # Validate method first
            method = method.lower()  # Convert to lowercase to handle any case
            valid_methods = ['tfidf', 'word2vec', 'bert']
            if method not in valid_methods:
                raise ValueError(f"Method must be one of {valid_methods}")

            if method == 'tfidf':
                if fit:
                    return self.tfidf_vectorizer.fit_transform(texts)
                return self.tfidf_vectorizer.transform(texts)
                
            elif method == 'word2vec':
                if not self.WORD2VEC_AVAILABLE:
                    logger.warning("Word2Vec not available, attempting to initialize...")
                    try:
                        tokenized_texts = [text.split() for text in texts]
                        self.word2vec_model = Word2Vec(
                            sentences=tokenized_texts,
                            vector_size=self.word2vec_dim,
                            window=5,
                            min_count=1,
                            workers=multiprocessing.cpu_count(),
                            sg=1
                        )
                        self.WORD2VEC_AVAILABLE = True
                        model_path = os.path.join('models', 'word2vec', 'custom_word2vec.model')
                        self.word2vec_model.save(model_path)
                        logger.info("Word2Vec model initialized successfully")
                    except Exception as e:
                        logger.error(f"Failed to initialize Word2Vec: {e}")
                        logger.warning("Falling back to TF-IDF")
                        if fit:
                            return self.tfidf_vectorizer.fit_transform(texts)
                        return self.tfidf_vectorizer.transform(texts)
                return np.array([self.get_word2vec_embedding(text) for text in texts])
                
            elif method == 'bert':
                encodings = self.bert_tokenizer(
                    texts,
                    truncation=True,
                    padding=True,
                    max_length=128,
                    return_tensors='pt'
                )
                return encodings['input_ids']
                
            else:
                raise ValueError("Method must be one of 'tfidf', 'word2vec', or 'bert'")
        except Exception as e:
            print(f"Error in vectorize_text: {e}")
            raise
        
            
    def _process_texts_with_stats(self, texts: List[str], preprocessing_config: Dict[str, bool]) -> List[str]:
        """Process texts and track statistics for each step"""
        try:
            # Inisialisasi tracking statistik
            self.current_stats = pd.DataFrame(columns=['Stage', 'Avg Length', 'Vocab Size'])
            
            # Statistik teks original
            orig_lengths = [len(str(text).split()) for text in texts]
            self.current_stats.loc[len(self.current_stats)] = {
                'Stage': 'Original',
                'Avg Length': np.mean(orig_lengths),
                'Vocab Size': len(set(' '.join([str(t) for t in texts]).split()))
            }
            
            # Process texts step by step
            processed_texts = texts.copy()
            steps = [
                ('case_folding', 'Case Folding'),
                ('remove_mentions', 'Remove Mentions'),
                ('remove_urls', 'Remove URLs'),
                ('remove_hashtags', 'Remove Hashtags'),
                ('remove_numbers', 'Remove Numbers'),
                ('remove_special_chars', 'Remove Special Chars'),
                ('remove_emojis', 'Remove Emojis'),
                ('remove_stopwords', 'Remove Stopwords'),
                ('stemming', 'Stemming')
            ]
            
            # Process each step if enabled in config
            for func_name, step_name in steps:
                if preprocessing_config.get(func_name, False):
                    try:
                        processed_texts = [getattr(self, func_name)(str(text)) for text in processed_texts]
                        # Track statistics after each step
                        lengths = [len(str(text).split()) for text in processed_texts]
                        self.current_stats.loc[len(self.current_stats)] = {
                            'Stage': step_name,
                            'Avg Length': np.mean(lengths),
                            'Vocab Size': len(set(' '.join([str(t) for t in processed_texts]).split()))
                        }
                    except Exception as e:
                        logger.error(f"Error in {step_name}: {e}")
                        raise
            
            return processed_texts
        except Exception as e:
            logger.error(f"Error in _process_texts_with_stats: {e}")
            raise
        
    def process_and_vectorize(self, texts: List[str], preprocessing_config: Dict[str, bool], 
                                vectorization_method: str) -> Tuple[np.ndarray, str]:
        try:
            logger.info(f"Starting preprocessing with method: {vectorization_method}")
            
            # Process texts
            processed_texts = self._process_texts_with_stats(texts, preprocessing_config)
            logger.info("Text preprocessing completed")
            
            # Vectorize texts
            vectors = self.vectorize_text(processed_texts, vectorization_method)
            logger.info("Vectorization completed")
            
            # Simpan data dengan format baru
            version_id = self.storage.save_preprocessed(
                texts=processed_texts,
                vectors=vectors,
                config=preprocessing_config,
                method=vectorization_method
            )
            
            logger.info(f"Processing completed successfully with version ID: {version_id}")
            return vectors, version_id
                
        except Exception as e:
            logger.error(f"Error in process_and_vectorize: {e}")
            raise
    
    def get_available_versions(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get list of available preprocessing and vectorization versions"""
        return self.storage.list_versions(['preprocessing', 'vectorization'])
    
    def get_vectorization_report(self, texts: List[str]) -> pd.DataFrame:
        """Generate report comparing different vectorization methods"""
        processed_texts = [self.clean_text(text) for text in texts]
        
        report_data = {}
        
        # TF-IDF
        start_time = time.time()
        tfidf_vectors = self.vectorize_text(processed_texts, 'tfidf')
        tfidf_time = time.time() - start_time
        
        report_data['tfidf'] = {
            'Vector Dimension': tfidf_vectors.shape[1] if len(tfidf_vectors.shape) > 1 else 0,
            'Sparsity': 1.0 - (np.count_nonzero(tfidf_vectors) / tfidf_vectors.size),
            'Processing Time': tfidf_time
        }
        
        # Word2Vec
        if self.WORD2VEC_AVAILABLE:
            start_time = time.time()
            word2vec_vectors = self.vectorize_text(processed_texts, 'word2vec')
            word2vec_time = time.time() - start_time
            
            report_data['word2vec'] = {
                'Vector Dimension': self.word2vec_dim,
                'Sparsity': 0.0,  # Dense vectors
                'Processing Time': word2vec_time
            }
        
        # BERT
        start_time = time.time()
        bert_vectors = self.vectorize_text(processed_texts, 'bert')
        bert_time = time.time() - start_time
        
        report_data['bert'] = {
            'Vector Dimension': bert_vectors.shape[1] if len(bert_vectors.shape) > 1 else 0,
            'Sparsity': 0.0,  # Dense vectors
            'Processing Time': bert_time
        }
        
        return report_data

    def get_preprocessing_report(self, texts: List[str]) -> pd.DataFrame:
        """Generate report on preprocessing impact"""
        report_data = []
        
        # Original texts
        orig_lengths = [len(text.split()) for text in texts]
        report_data.append({
            'Stage': 'Original',
            'Avg Length': np.mean(orig_lengths),
            'Vocab Size': len(set(' '.join(texts).split()))
        })
        
        # After each preprocessing step
        steps = [
            ('remove_mentions', 'Remove Mentions'),
            ('remove_urls', 'Remove URLs'),
            ('remove_hashtags', 'Remove Hashtags'),
            ('remove_special_chars', 'Remove Special Chars'),
            ('remove_stopwords', 'Remove Stopwords'),
            ('stemming', 'Stemming')
        ]
        
        current_texts = texts.copy()
        for func_name, step_name in steps:
            current_texts = [getattr(self, func_name)(text) for text in current_texts]
            lengths = [len(text.split()) for text in current_texts]
            report_data.append({
                'Stage': step_name,
                'Avg Length': np.mean(lengths),
                'Vocab Size': len(set(' '.join(current_texts).split()))
            })
            
        return pd.DataFrame(report_data)