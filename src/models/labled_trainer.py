import os
import sys
import pandas as pd
import logging
from typing import Tuple, List
from datetime import datetime
from transformers import pipeline
import torch

from src.exception.exception import FeedException  

class LabeledDataGenerator: 
    def __init__(self, input_dir="data/cleaned", output_dir="data/labeled", model_name="distilbert-base-uncased-finetuned-sst-2-english", batch_size=16, device=None):
        self.input_dir = input_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            device=0 if torch.cuda.is_available() else -1
        )
        
        logging.info("Initialized LabeledDataGenerator with Transformer pipeline")
        
    def load_unlabeled_data(self, filename="cleaned_data.csv"):
        try:
            csv_path = os.path.join(self.input_dir, filename)
            if not os.path.exists(csv_path):
               raise FeedException(f"Unlabeled data file not found: {csv_path}", sys)
           
            df = pd.read_csv(csv_path)
            if "cleaned_text" not in df.columns:
                raise FeedException("CSV must contain 'cleaned_text' column for labeling.", sys)
            
            logging.info(f"Loaded unlabeled data: {len(df)} rows from {csv_path}")
            return df
        
        except Exception as e:
            logging.error(f"Error loading unlabeled data: {str(e)}")
            raise FeedException(f"Failed to load unlabeled data: {str(e)}", sys)
        
    def generate_labels(self, texts: List[str]) -> Tuple[List[str], List[float]]:
        try:
            if not texts:
                raise FeedException("No texts provided for labeling.", sys)
            
            # Enhanced cleaning: Ensure all are str, handle NaN/None/empty
            cleaned_texts = []
            for t in texts:
                if pd.isna(t) or t is None or t == '':
                    cleaned_texts.append("")
                else:
                    cleaned_texts.append(str(t).strip())  # Strip whitespace early
            
            # Filter non-empty (after strip)
            non_empty_indices = [i for i, text in enumerate(cleaned_texts) if len(text) > 0]
            non_empty_texts = [text for text in cleaned_texts if len(text) > 0]
            
            if not non_empty_texts:
                raise FeedException("No valid non-empty texts found for labeling.", sys)
            
            # Ensure non_empty_texts are all str
            non_empty_texts = [str(text) for text in non_empty_texts]
            
            logging.info(f"Processing {len(non_empty_texts)} non-empty texts...")
            results = self.sentiment_pipeline(non_empty_texts, truncation=True, max_length=512, batch_size=16)  # Add batch_size for efficiency
            labels = [r['label'].lower() for r in results]
            confidences = [r['score'] for r in results]
            
            # Pad back with defaults for empty texts
            full_labels = ['neutral'] * len(texts)
            full_confidences = [0.0] * len(texts)
            for idx, orig_idx in enumerate(non_empty_indices):
                full_labels[orig_idx] = labels[idx]
                full_confidences[orig_idx] = confidences[idx]
            
            logging.info(f"Generated labels for {len(texts)} texts using Transformer pipeline")
            return full_labels, full_confidences
        
        except Exception as e:
            logging.error(f"Error generating labels: {str(e)}")
            raise FeedException(f"Label generation failed: {str(e)}", sys)
        
    def save_labeled_data(self, df, labels, confidences):
        try:
            df_labeled = df.copy()
            df_labeled['sentiment'] = labels
            df_labeled['sentiment_confidence'] = confidences

            output_filename = "fully_cleaned.csv"
            output_path = os.path.join(self.output_dir, output_filename)
            
            df_labeled.to_csv(output_path, index=False)
            logging.info(f"Saved labeled data to {output_path} with {len(labels)} predictions")
            return output_path
        
        except Exception as e:
            logging.error(f"Error saving labeled data: {str(e)}")
            raise FeedException(f"Saving labeled data failed: {str(e)}", sys)
        
def main():
    try:
        logging.info("Starting data labeling pipeline...")
        generator = LabeledDataGenerator()
        
        df = generator.load_unlabeled_data()
        texts = df['cleaned_text'].tolist()
        
        logging.info(f"Extracted {len(texts)} texts for labeling")
            
        labels, confidences = generator.generate_labels(texts)
            
        output_path = generator.save_labeled_data(df, labels, confidences)
        
        logging.info(f"Labeling pipeline completed. Output: {output_path}")
            
    except Exception as e: 
        logging.error(f"Data labeling pipeline failed: {str(e)}")
        raise FeedException(str(e), sys)
        
if __name__ == "__main__":
    main()