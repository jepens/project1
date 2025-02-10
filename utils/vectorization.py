# utils/vectorization.py

from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
import numpy as np
from transformers import DistilBertTokenizer, DistilBertModel
import torch
import logging
from typing import List, Tuple, Union, Optional

class TextVectorizer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.tfidf_vectorizer = None
        self.word2vec_model = None
        self.bert_tokenizer = None
        self.bert_model = None
        
    def vectorize_text(self, texts: List[str], method: str = 'tfidf') -> Tuple[Optional[np.ndarray], dict]:
        """
        Vectorize text using specified method
        """
        if not texts:
            self.logger.error("Empty text list provided for vectorization")
            return None, {}
            
        method = method.lower()
        if method not in ['tfidf', 'word2vec', 'bert']:
            self.logger.error(f"Invalid vectorization method: {method}")
            return None, {'error': f"Method must be one of 'tfidf', 'word2vec', or 'bert'"}
            
        try:
            if method == 'tfidf':
                return self._tfidf_vectorize(texts)
            elif method == 'word2vec':
                return self._word2vec_vectorize(texts)
            else:  # bert
                return self._bert_vectorize(texts)
                
        except Exception as e:
            self.logger.error(f"Vectorization error: {str(e)}")
            return None, {'error': str(e)}
            
    def _tfidf_vectorize(self, texts: List[str]) -> Tuple[np.ndarray, dict]:
        """
        TF-IDF vectorization
        """
        if self.tfidf_vectorizer is None:
            self.tfidf_vectorizer = TfidfVectorizer(max_features=1000)
            
        vectors = self.tfidf_vectorizer.fit_transform(texts)
        return vectors.toarray(), {
            'vocabulary_size': len(self.tfidf_vectorizer.vocabulary_),
            'feature_names': self.tfidf_vectorizer.get_feature_names_out().tolist()
        }
        
    def _word2vec_vectorize(self, texts: List[str]) -> Tuple[np.ndarray, dict]:
        """
        Word2Vec vectorization
        """
        # Prepare sentences
        sentences = [text.split() for text in texts]
        
        # Train Word2Vec if not already trained
        if self.word2vec_model is None:
            self.word2vec_model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
            
        # Create document vectors by averaging word vectors
        doc_vectors = []
        for sentence in sentences:
            word_vectors = [self.word2vec_model.wv[word] for word in sentence if word in self.word2vec_model.wv]
            if word_vectors:
                doc_vectors.append(np.mean(word_vectors, axis=0))
            else:
                doc_vectors.append(np.zeros(self.word2vec_model.vector_size))
                
        return np.array(doc_vectors), {
            'vector_size': self.word2vec_model.vector_size,
            'vocabulary_size': len(self.word2vec_model.wv.key_to_index)
        }
        
    def _bert_vectorize(self, texts: List[str]) -> Tuple[np.ndarray, dict]:
        """
        BERT vectorization
        """
        if self.bert_tokenizer is None:
            self.bert_tokenizer = DistilBertTokenizer.from_pretrained('indolem/indobert-base-uncased')
            self.bert_model = DistilBertModel.from_pretrained('indolem/indobert-base-uncased')
            
        # Tokenize and get BERT embeddings
        encoded_input = self.bert_tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=512)
        
        with torch.no_grad():
            outputs = self.bert_model(**encoded_input)
            # Use [CLS] token embeddings as document representation
            embeddings = outputs.last_hidden_state[:, 0, :].numpy()
            
        return embeddings, {
            'embedding_dim': embeddings.shape[1],
            'model_name': 'indobert-base-uncased'
        }