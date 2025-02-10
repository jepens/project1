import os
import json
import shutil
from datetime import datetime
import numpy as np
from scipy import sparse
from typing import Dict, Any, Optional, List, Tuple, Union
import streamlit as st
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class StorageManager:
    def __init__(self, base_path: str = 'data'):
        """Initialize storage manager"""
        # Create directory structure
        self.base_path = base_path
        self.models_dir = 'models'
        self.processed_dir = os.path.join(base_path, 'processed')
        self.vectorized_dir = os.path.join(base_path, 'vectorized')
        
        # Pastikan direktori ada
        for dir_path in [self.base_path, self.models_dir]:
            os.makedirs(dir_path, exist_ok=True)
            
        os.makedirs(self.processed_dir, exist_ok=True)
        os.makedirs(self.vectorized_dir, exist_ok=True)
        
        # Validasi struktur file
        if not self.validate_file_structure():
            logger.warning("Storage structure validation warning - some features might be limited")
            # Jangan raise error, biarkan aplikasi tetap berjalan
            
    # Create directories if they don't exist
        for dir_path in [self.processed_dir, self.vectorized_dir, self.models_dir]:
                os.makedirs(dir_path, exist_ok=True)    
                
    def verify_vectors(self, version_id: str) -> bool:
        """Verifikasi keberadaan dan integritas file vector"""
        try:
            logger.info(f"Verifying vectors for version: {version_id}")
            
            # Cek metadata
            metadata_path = os.path.join(self.vectorized_dir, f'metadata_{version_id}.npz')
            if not os.path.exists(metadata_path):
                logger.error(f"Metadata file not found: {metadata_path}")
                return False
                
            # Load metadata
            try:
                metadata = np.load(metadata_path, allow_pickle=True)['metadata'].item()
                logger.info(f"Loaded metadata successfully for version: {version_id}")
            except Exception as e:
                logger.error(f"Error loading metadata: {e}")
                return False
                
            # Cek file vectors
            is_sparse = metadata.get('is_sparse', False)
            vector_path = os.path.join(
                self.vectorized_dir, 
                f'vectors_{version_id}.{"npz" if is_sparse else "npy"}'
            )
            
            if not os.path.exists(vector_path):
                logger.error(f"Vector file not found: {vector_path}")
                return False
                
            logger.info(f"All files verified successfully for version: {version_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error verifying vectors: {e}")
            return False   
            
    def _generate_version_id(self, config: Dict[str, Any] = None) -> str:
        """Generate version ID berdasarkan timestamp"""
        return datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def _save_config(self, path: str, config: Dict[str, Any]) -> None:
        """Save configuration to JSON file"""
        try:
            with open(path, 'w') as f:
                json.dump(config, f, indent=4)
        except Exception as e:
            st.error(f"Error saving config: {str(e)}")
            
    def _load_config(self, path: str) -> Optional[Dict[str, Any]]:
        """Load configuration from JSON file"""
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except Exception as e:
            st.warning(f"Error loading config from {path}: {str(e)}")
            return None
            
    def save_preprocessed(self, texts: List[str], vectors: Union[np.ndarray, sparse.spmatrix], 
                        config: Dict[str, Any], method: str) -> str:
        """
        Simpan hasil preprocessing dengan format baru yang menyertakan vectors
        
        Args:
            texts: List teks yang telah dipreprocess
            vectors: Hasil vektorisasi dalam bentuk numpy array atau sparse matrix
            config: Konfigurasi preprocessing yang digunakan
            method: Metode vektorisasi yang digunakan (tfidf/word2vec/bert)
        """
        try:
            # Generate version ID
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            config_hash = hash(json.dumps(config, sort_keys=True)) % 1000000
            
            # Status preprocessing
            status = "stemmed" if config.get('stemming', False) else "basic"
            
            # Format nama file baru
            version_id = f"{timestamp}_{method}_{config_hash}_{status}"
            save_path = os.path.join(self.processed_dir, f'preprocessed_{version_id}.npz')
            
            # Metadata yang lengkap
            metadata = {
                'version_id': version_id,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'method': method,
                'config': config,
                'status': status
            }
            
            # Simpan semua dalam satu file
            np.savez_compressed(
                save_path,
                texts=texts,
                metadata=metadata,
                vectors=vectors if not sparse.issparse(vectors) else vectors.toarray(),
                is_sparse=sparse.issparse(vectors)
            )
            
            logger.info(f"Saved preprocessed data with version ID: {version_id}")
            return version_id
            
        except Exception as e:
            logger.error(f"Error saving preprocessed data: {e}")
            raise
            
    def save_vectorized(self, vectors: Union[np.ndarray, sparse.spmatrix], method: str,
                    config: Dict[str, Any], description: Optional[str] = None) -> str:
        try:
            # Generate version ID
            version_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            logger.info(f"Saving vectors with version ID: {version_id}")

            # Simpan vectors
            vector_path = os.path.join(self.vectorized_dir, f'vectors_{version_id}')
            if sparse.issparse(vectors):
                vector_file = f"{vector_path}.npz"
                sparse.save_npz(vector_file, vectors)
                is_sparse = True
            else:
                vector_file = f"{vector_path}.npy"
                np.save(vector_file, vectors)
                is_sparse = False
                
            logger.info(f"Saved vector file to: {vector_file}")

            # Buat metadata yang lengkap
            metadata = {
                'version_id': version_id,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'method': method,
                'is_sparse': is_sparse,
                'shape': vectors.shape if hasattr(vectors, 'shape') else None,
                'config': config,
                'description': description
            }
            
            # Simpan metadata hanya dalam format npz
            metadata_path = os.path.join(self.vectorized_dir, f'metadata_{version_id}.npz')
            np.savez(metadata_path, metadata=metadata)
            logger.info(f"Saved metadata to: {metadata_path}")

            return version_id
            
        except Exception as e:
            logger.error(f"Error in save_vectorized: {e}")
            raise
    
    # Tambahkan method baru:
    def validate_file_structure(self) -> bool:
        """Validate storage directory structure and permissions"""
        try:
            required_dirs = [self.processed_dir, self.vectorized_dir, self.models_dir]
            
            # Buat semua direktori yang diperlukan
            for dir_path in required_dirs:
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path, exist_ok=True)
                    logger.info(f"Created directory: {dir_path}")

            # Verifikasi permission dengan mencoba menulis file test
            for dir_path in required_dirs:
                try:
                    test_file = os.path.join(dir_path, '.test')
                    with open(test_file, 'w') as f:
                        f.write('test')
                    os.remove(test_file)
                except Exception as e:
                    logger.warning(f"Directory permission check warning for {dir_path}: {e}")
                    # Jangan raise error, hanya berikan warning
                    
            return True
            
        except Exception as e:
            logger.error(f"Error validating file structure: {e}")
            return False
            
    def save_model(self,
                model: Any,
                model_type: str,
                config: Dict[str, Any],
                metrics: Optional[Dict[str, float]] = None) -> Tuple[str, str]:
        """Save model with its configuration"""
        try:
            # Generate version ID
            version_id = self._generate_version_id(config)
            
            # Create model directory
            model_dir = os.path.join(self.models_dir, f"{model_type}_{version_id}")
            os.makedirs(model_dir, exist_ok=True)
            
            # Save model (the actual saving is implemented in each model class)
            model.save_model(model_dir)
            
            # Convert numpy arrays in metrics if any
            processed_metrics = {}
            if metrics:
                for key, value in metrics.items():
                    if isinstance(value, (np.ndarray, np.number)):
                        processed_metrics[key] = value.item() if isinstance(value, np.number) else value.tolist()
                    else:
                        processed_metrics[key] = value
                        
            # Convert numpy arrays in config if any
            processed_config = {}
            for key, value in config.items():
                if isinstance(value, (np.ndarray, np.number)):
                    processed_config[key] = value.item() if isinstance(value, np.number) else value.tolist()
                elif isinstance(value, dict):
                    # Handle nested dictionaries
                    processed_config[key] = {}
                    for k, v in value.items():
                        if isinstance(v, (np.ndarray, np.number)):
                            processed_config[key][k] = v.item() if isinstance(v, np.number) else v.tolist()
                        else:
                            processed_config[key][k] = v
                else:
                    processed_config[key] = value
            
            # Create and save config
            full_config = {
                'model_info': {
                    'name': model_type,
                    'version': version_id,
                    'metrics': processed_metrics
                },
                'hyperparameters': processed_config,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            config_path = os.path.join(model_dir, "config.json")
            self._save_config(config_path, full_config)
            
            return model_dir, version_id
            
        except Exception as e:
            st.error(f"Error saving model: {str(e)}")
            logger.error(f"Error saving model: {str(e)}")
            return "", ""
            
    def load_preprocessed(self, version_id: str) -> Tuple[Optional[List[str]], Optional[Dict[str, Any]]]:
        try:
            if not version_id:
                logger.error("Empty version_id provided")
                return None, None
                
            file_path = os.path.join(self.processed_dir, f'preprocessed_{version_id}.npz')
            if not os.path.exists(file_path):
                logger.error(f"Preprocessed file not found: {file_path}")
                return None, None
                
            data = np.load(file_path, allow_pickle=True)
            texts = data['texts'].tolist()
            metadata = data['metadata'].item()
            
            if not texts:
                logger.warning(f"No texts found in {file_path}")
            if not metadata:
                logger.warning(f"No metadata found in {file_path}")
                
            return texts or [], metadata or {}
            
        except Exception as e:
            logger.error(f"Error loading preprocessed texts: {e}")
            return None, None
    
    def load_preprocessed_with_vectors(self, version_id: str) -> Tuple[Optional[List[str]], Optional[np.ndarray], Optional[Dict[str, Any]]]:
        """
        Load preprocessed data dengan vectors dan metadata
        """
        try:
            file_path = os.path.join(self.processed_dir, f'preprocessed_{version_id}.npz')
            if not os.path.exists(file_path):
                logger.error(f"Preprocessed file not found: {file_path}")
                return None, None, None
                
            data = np.load(file_path, allow_pickle=True)
            texts = data['texts']
            vectors = data['vectors']
            metadata = data['metadata'].item()
            
            return texts, vectors, metadata
            
        except Exception as e:
            logger.error(f"Error loading preprocessed data: {e}")
            return None, None, None
            
    def load_model(self, model_dir: str) -> Tuple[Optional[Any], Optional[Dict[str, Any]]]:
        """Load model and its configuration"""
        try:
            config_path = os.path.join(model_dir, "config.json")
            config = self._load_config(config_path)
            
            if config:
                # The actual model loading is implemented in each model class
                return None, config  # Return None for model as it's loaded by the model class
                
        except Exception as e:
            st.warning(f"Error loading model: {str(e)}")
            return None, None
        
    def get_experiment_results(self) -> List[Dict[str, Any]]:
        """Get all experiment results"""
        results = []
        try:
            # Get all model directories
            for item in os.listdir(self.models_dir):
                model_dir = os.path.join(self.models_dir, item)
                if os.path.isdir(model_dir):
                    config_path = os.path.join(model_dir, "config.json")
                    if os.path.exists(config_path):
                        config = self._load_config(config_path)
                        if config:
                            # Extract relevant information
                            experiment_data = {
                                'model_type': config['model_info']['name'],
                                'version_id': config['model_info']['version'],
                                'timestamp': config['timestamp']
                            }
                            
                            # Add metrics
                            metrics = config['model_info'].get('metrics', {})
                            for metric, value in metrics.items():
                                experiment_data[metric] = value
                                
                            # Add hyperparameters
                            params = config.get('hyperparameters', {})
                            for param, value in params.items():
                                experiment_data[f'param_{param}'] = value
                                
                            results.append(experiment_data)
                            
            return results
            
        except Exception as e:
            st.error(f"Error getting experiment results: {str(e)}")
            return []
            
    def export_experiment(self, export_path: str) -> bool:
        """Export all experiment data"""
        try:
            export_dir = f"{export_path}_exports"
            os.makedirs(export_dir, exist_ok=True)
            
            # Export experiment results
            results = self.get_experiment_results()
            if results:
                results_path = os.path.join(export_dir, "experiment_results.json")
                with open(results_path, 'w') as f:
                    json.dump(results, f, indent=4)
                    
            # Export model configurations
            configs_dir = os.path.join(export_dir, "model_configs")
            os.makedirs(configs_dir, exist_ok=True)
            
            for item in os.listdir(self.models_dir):
                model_dir = os.path.join(self.models_dir, item)
                if os.path.isdir(model_dir):
                    config_path = os.path.join(model_dir, "config.json")
                    if os.path.exists(config_path):
                        shutil.copy2(config_path, os.path.join(configs_dir, f"{item}_config.json"))
                        
            # Export training histories
            histories_dir = os.path.join(export_dir, "training_histories")
            os.makedirs(histories_dir, exist_ok=True)
            
            for item in os.listdir(self.models_dir):
                model_dir = os.path.join(self.models_dir, item)
                if os.path.isdir(model_dir):
                    history_path = os.path.join(model_dir, "training_history.json")
                    if os.path.exists(history_path):
                        shutil.copy2(history_path, os.path.join(histories_dir, f"{item}_history.json"))
                        
            return True
            
        except Exception as e:
            st.error(f"Error exporting experiment: {str(e)}")
            return False
            
    def import_model(self, model_path: str, config_path: str) -> Tuple[str, str]:
        """Import model from external files"""
        try:
            # Load and validate config
            config = self._load_config(config_path)
            if not config:
                raise ValueError("Invalid config file")
                
            # Generate new version ID to avoid conflicts
            version_id = self._generate_version_id(config)
            model_type = config['model_info']['name']
            
            # Create new model directory
            new_model_dir = os.path.join(self.models_dir, f"{model_type}_{version_id}")
            os.makedirs(new_model_dir, exist_ok=True)
            
            # Copy model file
            shutil.copy2(model_path, os.path.join(new_model_dir, "model.joblib"))
            
            # Update and save config
            config['model_info']['version'] = version_id
            config['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            config_new_path = os.path.join(new_model_dir, "config.json")
            self._save_config(config_new_path, config)
            
            return new_model_dir, version_id
            
        except Exception as e:
            st.error(f"Error importing model: {str(e)}")
            return "", ""
            
    def delete_version(self, version_id: str, artifact_type: str) -> bool:
        try:
            if artifact_type == 'preprocessed':
                file_path = os.path.join(self.processed_dir, f'preprocessed_{version_id}.npz')
                if os.path.exists(file_path):
                    os.remove(file_path)
                    
            elif artifact_type == 'vectorized':
                # Remove vector file
                npz_path = os.path.join(self.vectorized_dir, f'vectors_{version_id}.npz')
                npy_path = os.path.join(self.vectorized_dir, f'vectors_{version_id}.npy')
                metadata_path = os.path.join(self.vectorized_dir, f'metadata_{version_id}.json')
                
                for path in [npz_path, npy_path, metadata_path]:
                    if os.path.exists(path):
                        os.remove(path)
                        
            elif artifact_type == 'model':
                # Find and remove model directory
                for item in os.listdir(self.models_dir):
                    if version_id in item:
                        model_dir = os.path.join(self.models_dir, item)
                        if os.path.exists(model_dir):
                            shutil.rmtree(model_dir)
                            break
                            
            return True
            
        except Exception as e:
            st.error(f"Error deleting version {version_id}: {str(e)}")
            return False
            
    def cleanup_old_versions(self, keep_last_n: int = 5) -> bool:
        """Cleanup old versions keeping only the most recent ones"""
        try:
            # Get all versions sorted by timestamp
            versions = []
            for artifact_type in ['preprocessed', 'vectorized', 'model']:
                if artifact_type == 'model':
                    versions.extend([
                        (item, 'model', os.path.getmtime(os.path.join(self.models_dir, item)))
                        for item in os.listdir(self.models_dir)
                        if os.path.isdir(os.path.join(self.models_dir, item))
                    ])
                else:
                    dir_path = self.processed_dir if artifact_type == 'preprocessed' else self.vectorized_dir
                    versions.extend([
                        (item, artifact_type, os.path.getmtime(os.path.join(dir_path, item)))
                        for item in os.listdir(dir_path)
                        if item.endswith('.npz')  # Ubah ini untuk mencari file .npz
                    ])
                    
            # Sort by timestamp (newest first)
            versions.sort(key=lambda x: x[2], reverse=True)
            
            # Keep only the most recent versions
            versions_to_delete = versions[keep_last_n:]
            
            # Delete old versions
            for version_name, artifact_type, _ in versions_to_delete:
                version_id = version_name.split('_')[1]
                self.delete_version(version_id, artifact_type)
                
            return True
            
        except Exception as e:
            st.error(f"Error cleaning up old versions: {str(e)}")
            return False