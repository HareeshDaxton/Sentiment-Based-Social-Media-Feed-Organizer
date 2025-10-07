import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any
from src.Storage.mongo_writer import MongoWriter  # Uncomment if MongoDB is set up
from src.models.sentiment_model import SentimentModel
from src.logging.logger import logging
from src.exception.exception import FeedException
import sys

def load_processed_data(csv_path: str = "data/labeled/fully_cleaned.csv"):
    try:
        if not os.path.exists(csv_path):
            raise FileExistsError(f"Processed data not found: {csv_path}")
        
        df = pd.read_csv(csv_path)
        if 'cleaned_text' not in df.columns:
            raise FileExistsError("Missing 'cleaned_text' column in processed data.")
        
        logging.info(f"Loaded processed data: {len(df)} rows from {csv_path}")
        return df
    
    except Exception as e:
        logging.error(f"Error loading processed data: {str(e)}")
        raise FeedException(f"Failed to load processed data: {str(e)}")
    
def save_results(df: pd.DataFrame, predictions: Dict[str, Any], use_mongo: bool = False):
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"data/processed/sentiment_output_{timestamp}.csv"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        df['predicted_sentiment_label'] = predictions['labels']  # Use 'predicted_' to distinguish from original labels
        df['predicted_sentiment_confidence'] = predictions['confidences']
        
        df.to_csv(output_path, index=False)
        logging.info(f"Saved sentiment results to {output_path}")
        
        if use_mongo:
            logging.info("Sent results to MongoDB")
            
    except Exception as e:
        logging.error(f"Error saving results: {str(e)}")
        raise FeedException(f"Saving results failed: {str(e)}", sys)
    
def run_sentiment_pipeline(model_type: str = 'transformer', use_mongo: bool = False):
    logging.info("Starting sentiment analysis pipeline...")
    try:
        # Load data
        data = load_processed_data()
        texts = data['cleaned_text'].tolist()
        
        cleaned_texts = []
        for t in texts:
            if pd.isna(t) or t is None or t == '':
                cleaned_texts.append("")
            else:
                cleaned_texts.append(str(t).strip())
        
        # Filter non-empty for prediction
        non_empty_indices = [i for i, text in enumerate(cleaned_texts) if len(text) > 0]
        non_empty_texts = [text for text in cleaned_texts if len(text) > 0]
        
        if non_empty_texts:
            # Load model and predict on non-empty
            model = SentimentModel(model_type=model_type)
            labels, confidences = model.predict(non_empty_texts)
            
            # Pad back with defaults for empty texts
            full_labels = ['neutral'] * len(texts)
            full_confidences = [0.0] * len(texts)
            for idx, orig_idx in enumerate(non_empty_indices):
                full_labels[orig_idx] = labels[idx]
                full_confidences[orig_idx] = confidences[idx]
        else:
            full_labels = ['neutral'] * len(texts)
            full_confidences = [0.0] * len(texts)
        
        predictions = {'labels': full_labels, 'confidences': full_confidences}
        
        # Save
        save_results(data, predictions, use_mongo)
        
        logging.info("Sentiment pipeline completed successfully.")
    
    except Exception as e:
        logging.error(f"Pipeline failed: {str(e)}")
        raise FeedException(f"Sentiment pipeline execution failed: {str(e)}", sys)