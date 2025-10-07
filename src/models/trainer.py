import os
import logging
import joblib
import pandas as pd
from typing import Tuple, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, recall_score
from src.exception.exception import FeedException
from src.logging.logger import logging
import sys


class SentimentTrainer:
    def __init__(self, model_dir="data/processed"):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        self.vectorize = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        
        self.models = {}
        logging.info(f"Initialized SentimentTrainer with model_dir: {model_dir}")
        
    def load_data(self, csv_path):
        try:
            if not os.path.exists(csv_path):
                raise FeedException(f"Data file not found: {csv_path}")
            
            df = pd.read_csv(csv_path)
            if 'cleaned_text' not in df.columns or 'sentiment' not in df.columns:
                raise FeedException("CSV must contain 'cleaned_text' and 'sentiment' columns.")
            
            df = df.dropna(subset=['cleaned_text', 'sentiment'])
            df['cleaned_text'] = df['cleaned_text'].fillna('').astype(str)
            df['sentiment'] = df['sentiment'].astype(str)
            
            x = df['cleaned_text']
            y = df['sentiment']
            logging.info(f"Loaded {len(x)} labeled samples from {csv_path}")
            return x, y
        
        except Exception as e:
            logging.error(f"Error loading data: {str(e)}")
            raise FeedException(f"Failed to load data from {csv_path}: {str(e)}")
        
    def train(self, csv_path, test_size=0.2, random_state: int = 42):
        try:
            x, y = self.load_data(csv_path)
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state,stratify=y)
            
            x_train_vec = self.vectorize.fit_transform(x_train)
            x_test_vec = self.vectorize.transform(x_test)
            
            clasifiers = {
                'logistic_regression': LogisticRegression(
                                            penalty='l2',                # Standard regularization to prevent overfitting
                                            C=3.0,                       # Moderate regularization strength (higher = weaker regularization)
                                            solver='lbfgs',              # Robust optimizer for multi-class and dense/sparse TF-IDF
                                            class_weight='balanced',     # Handles class imbalance automatically
                                            max_iter=1000,               # Ensures convergence with TF-IDF feature space
                                            n_jobs=-1,                   # Utilize all CPU cores
                                            random_state=42,             # Reproducibility
                                            verbose=1                    # Optional: monitor convergence
                                        ),
                'SVM' : SVC(probability=True, random_state=random_state),
                'random_forest' : RandomForestClassifier(
                                                n_estimators=400,           
                                                max_depth=30,               
                                                min_samples_split=4,       
                                                min_samples_leaf=2,      
                                                max_features='sqrt',       
                                                bootstrap=True,             
                                                class_weight='balanced_subsample',  
                                                criterion='gini',           
                                                n_jobs=-1,                 
                                                random_state=42,          
                                                verbose=1                  
                                            )
            
            }
            
            metrices = {}
            for name, clf in clasifiers.items():
                logging.info(f"Training {name}...")
                clf.fit(x_train_vec ,y_train)
                self.models[name] = clf
                
                preds = clf.predict(x_test_vec)
                acc = accuracy_score(y_test, preds)
                f1 = f1_score(y_test, preds, average='weighted')
                recall = recall_score(y_test, preds, average='weighted')
                metrices[name] = {"accuracy" : acc, "f1" : f1, 'recall' : recall}
                
                logging.info(f"{name} - Accuracy: {acc:.4f}, F1: {f1:.4f}")
                
            return metrices
        
        except Exception as e:
            logging.error(f"Error during training: {str(e)}")
            raise FeedException(f"Training failed: {str(e)}", sys)
        
    def evaluate(self, x_test, y_test, model_name):
        if model_name not in self.models:
            raise FeedException(f"Model '{model_name}' not trained yet.")
        
        try:
            model = self.models[model_name]
            x_test_vec = self.vectorizer.transform(x_test)
            preds = model.predict(x_test_vec)
            probs= model.predict_proba(x_test_vec)
            
            acc = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds, average='weighted')
            recall = recall_score(y_test, preds,average='weighted')
            report = classification_report(y_test, preds, output_dict=True)
            
            logging.info(f"Evaluation for {model_name}: {report}")
            return {
                'accuracy' : acc,
                'f1' : f1,
                "recall" : recall,
                'classification_report' : report,
                'predictions': preds,
                'probabilities': probs

            }
            
        except Exception as e:
            logging.error(f"Error during evaluation: {str(e)}")
            raise FeedException(f"Evaluation failed: {str(e)}", sys)
            
            
    def save_artifacts(self, model_name='logistic_regression'): 
        if model_name not in self.models:
            raise FeedException(f"Model '{model_name}' not trained yet.")
        
        try:
            model_path = os.path.join(self.model_dir, f"sentiment_model_{model_name}.pkl")
            vectorizer_path = os.path.join(self.model_dir, "tfidf_vectorizer_lr.pkl")
            joblib.dump(self.models[model_name], model_path)
            joblib.dump(self.vectorize, vectorizer_path)
            logging.info(f"Saved model to {model_path} and vectorizer to {vectorizer_path}")
            
        except Exception as e:
            logging.error(f"Error saving artifacts: {str(e)}")
            raise FeedException(f"Saving artifacts failed: {str(e)}", sys)