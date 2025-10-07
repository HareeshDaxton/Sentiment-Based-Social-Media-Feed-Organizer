import os
import pickle
import numpy as np  
import pandas as pd
from typing import Optional, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from src.logging.logger import logging
from src.exception.exception import FeedException
import sys


class TFIDFVectorizerHandler:
    def __init__(
        self, 
        max_features=5000,
        ngram_range=(1, 2),
        use_svd=True,
        svd_components=100,
        min_df=2,
        max_df=0.95
    ):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.use_svd = use_svd
        self.svd_components = svd_components
        self.min_df = min_df
        self.max_df = max_df
        
        self.vectorizer = None
        self.svd = None
        self.fitted = False
        
    def fit_transform(self, texts):
        if not isinstance(texts, pd.Series):
            texts = pd.Series(texts)
            
        logging.info(f"Fitting TF-IDF on {len(texts)} documents...")
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_df=self.max_df,
            lowercase=False,
            token_pattern=r"(?u)\b\w+\b" 
        )
        
        tfidf_matrix = self.vectorizer.fit_transform(texts.dropna())
        
        if self.use_svd:
            logging.info(f"Applying SVD to reduce to {self.svd_components} components...")
            self.svd = TruncatedSVD(n_components=self.svd_components)
            tfidf_matrix = self.svd.fit_transform(tfidf_matrix)
            
        self.fitted = True
        logging.info(f"TF-IDF shape: {tfidf_matrix.shape}")
        return tfidf_matrix
    
    def transform(self, texts):
        if not self.fitted:
            FeedException("Vectorizer not fitted. Call fit_transform first.", sys)
            
        tfidf_matrix = self.vectorizer.transform(texts.dropna())
        
        if self.use_svd:
            tfidf_matrix = self.svd.transform(tfidf_matrix)
            
        return tfidf_matrix
    
    def save(self, path):
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path+ "_vectorizer.pkl", "wb") as f:
                pickle.dump({
                    "vectorizer": self.vectorizer,
                    "svd" : self.svd,
                    "config":{
                        "max_features": self.max_features,
                        "ngram_range" : self.ngram_range,
                        "use_svd" : self.use_svd,
                        "svd_components" :self.svd_components
                    }
                }, f)
            logging.info(f"Vectorizer saved to {path}_vectorizer.pkl")
            
        except Exception as e:
            FeedException(e, sys)
            
    def save_vectors(self, vectors,ids=None,  path="vectors.npz"):
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            np.savez_compressed(path, vectors=vectors, ids=ids or np.arange(len(vectors)))
            logging.info(f"Vectors saved to {path}")
            
        except Exception as e:
            FeedException(e, sys)