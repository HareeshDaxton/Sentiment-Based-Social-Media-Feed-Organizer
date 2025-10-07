import os
import logging
import joblib
import numpy as np
from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from src.exception.exception import FeedException
from src.logging.logger import logging


class SentimentModel:
    MODEL_TYPES = {'classical', 'transformer'}
    
    def __init__(self, 
                 model_type='transformer',
                 model_path= "data/processed/sentiment_model_logistic_regression.pkl",
                 vectorizer_path="data/processed/tfidf_vectorizer_lr.pkl",
                 transformer_model="cardiffnlp/twitter-roberta-base-sentiment-latest"
                 ):
        
        if model_type not in self.MODEL_TYPES:
            raise FeedException(f"model_type must be one of {self.MODEL_TYPES}")
        
        self.model_type= model_type
        self.transformer_model = transformer_model
        
        try:
            if model_type == 'classical':
                if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
                    raise FeedException("Model or vectorizer files not found. Train first with trainer.py.")
                
                self.model = joblib.load(model_path)
                self.vectorizer = joblib.load(vectorizer_path)
                logging.info(f"Loaded classical model from {model_path}")
                
            else:
                self.sentiment_pipeline= pipeline(
                    "sentiment-analysis", 
                    model=transformer_model, 
                    device=0 if torch.cuda.is_available() else -1
                )
                logging.info(f"Loaded transformer model: {transformer_model}")
                
        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
            raise FeedException(f"Model initialization failed: {str(e)}")
        
    
    def predict(self, texts):
        try:
            if not texts:
                logging.warning("Empty texts list provided.")
                return [], []
            
            if self.model_type == 'classical':
                vec_texts = self.vectorizer.transform(texts)
                probs = self.model.predict_proba(vec_texts)
                preds = self.model.predict(vec_texts)
                
                labels = [self.model.classes_[np.argmax(p)] for p in probs]
                confidences = [np.max(p) for p in probs]
                
            else:
                results = self.sentiment_pipeline(texts, truncation=True, max_length=512)
                labels = [r['label'].lower() for r in results]  # e.g., 'negative', 'neutral', 'positive'
                confidences = [r['score'] for r in results]
                
                logging.info(f"Predicted sentiments for {len(texts)} texts using {self.model_type} model.")
                return labels, confidences
            
        except Exception as e:
            logging.error(f"Error during prediction: {str(e)}")
            raise FeedException(f"Prediction failed: {str(e)}")
        