from scipy import sparse
import numpy as np
from datetime import datetime
import os
import logging
from typing import Union, Tuple, Dict, Any, Optional

class VectorStorageManager:
    """
    Kelas untuk menangani penyimpanan dan pemuatan vektor hasil preprocessing
    """
    def __init__(self, base_path: str = 'data'):
        self.base_path = base_path
        self.processed_dir = os.path.join(base_path, 'processed')
        self.vectorized_dir = os.path.join(base_path, 'vectorized')
        self.logger = logging.getLogger(__name__)
        
        # Buat direktori jika belum ada
        os.makedirs(self.processed_dir, exist_ok=True)
        os.makedirs(self.vectorized_dir, exist_ok=True)
        
    def save_vectors(self, 
                    vectors: Union[np.ndarray, sparse.spmatrix],
                    texts: list,
                    preprocessing_config: Dict[str, Any],
                    vectorization_method: str,
                    version_id: Optional[str] = None) -> str:
        """
        Menyimpan hasil vektorisasi dengan format yang konsisten
        """
        try:
            if version_id is None:
                version_id = datetime.now().strftime('%Y%m%d_%H%M%S')
                
            # Simpan hasil preprocessing
            preprocess_path = os.path.join(self.processed_dir, f'preprocessed_{version_id}.npz')
            np.savez_compressed(
                preprocess_path,
                texts=texts,
                metadata={
                    'version_id': version_id,
                    'config': preprocessing_config,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
            )
            
            # Simpan vektor hasil vektorisasi
            vector_path = os.path.join(self.vectorized_dir, f'vectors_{version_id}')
            if sparse.issparse(vectors):
                sparse.save_npz(f"{vector_path}.npz", vectors)
                is_sparse = True
            else:
                np.save(f"{vector_path}.npy", vectors)
                is_sparse = False
                
            # Simpan metadata vektorisasi
            metadata = {
                'version_id': version_id,
                'method': vectorization_method,
                'is_sparse': is_sparse,
                'shape': vectors.shape,
                'preprocessing_config': preprocessing_config,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            np.savez(
                os.path.join(self.vectorized_dir, f'metadata_{version_id}.npz'),
                metadata=metadata
            )
            
            return version_id
            
        except Exception as e:
            self.logger.error(f"Error saving vectors: {str(e)}")
            raise
            
    def load_vectors(self, version_id: str) -> Tuple[Union[np.ndarray, sparse.spmatrix], Dict[str, Any]]:
        """
        Memuat vektor dan metadata dengan format yang konsisten
        """
        try:
            # Load metadata
            metadata_path = os.path.join(self.vectorized_dir, f'metadata_{version_id}.npz')
            metadata = np.load(metadata_path, allow_pickle=True)['metadata'].item()
            
            # Load vectors berdasarkan format
            if metadata['is_sparse']:
                vector_path = os.path.join(self.vectorized_dir, f'vectors_{version_id}.npz')
                vectors = sparse.load_npz(vector_path)
            else:
                vector_path = os.path.join(self.vectorized_dir, f'vectors_{version_id}.npy')
                vectors = np.load(vector_path)
                
            return vectors, metadata
            
        except Exception as e:
            self.logger.error(f"Error loading vectors: {str(e)}")
            raise
            
    def load_preprocessed(self, version_id: str) -> Tuple[list, Dict[str, Any]]:
        """
        Memuat hasil preprocessing
        """
        try:
            file_path = os.path.join(self.processed_dir, f'preprocessed_{version_id}.npz')
            data = np.load(file_path, allow_pickle=True)
            return data['texts'], data['metadata'].item()
        except Exception as e:
            self.logger.error(f"Error loading preprocessed data: {str(e)}")
            raise
            
    def validate_data(self, version_id: str) -> bool:
        """
        Memvalidasi keberadaan dan integritas data
        """
        try:
            # Cek file preprocessing
            preprocess_path = os.path.join(self.processed_dir, f'preprocessed_{version_id}.npz')
            if not os.path.exists(preprocess_path):
                return False
                
            # Cek file metadata
            metadata_path = os.path.join(self.vectorized_dir, f'metadata_{version_id}.npz')
            if not os.path.exists(metadata_path):
                return False
                
            # Load metadata untuk cek format vector
            metadata = np.load(metadata_path, allow_pickle=True)['metadata'].item()
            
            # Cek file vector sesuai format
            if metadata['is_sparse']:
                vector_path = os.path.join(self.vectorized_dir, f'vectors_{version_id}.npz')
            else:
                vector_path = os.path.join(self.vectorized_dir, f'vectors_{version_id}.npy')
                
            return os.path.exists(vector_path)
            
        except Exception as e:
            self.logger.error(f"Error validating data: {str(e)}")
            return False