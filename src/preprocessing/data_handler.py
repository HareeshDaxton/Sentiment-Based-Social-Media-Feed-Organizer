
import sys
import json
import yaml
from datetime import datetime
from typing import Optional, Dict, Any, List
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure
from pathlib import Path
import pandas as pd
from src.logging.logger import logging
from src.exception.exception import FeedException
from src.preprocessing.text_cleaner import TextCleaner
from src.preprocessing.normalizer import TextNormalizer
from src.preprocessing.vectorizer import TFIDFVectorizerHandler


class DataHandler:
    def __init__(self, raw_dir="data/raw", processed_dir="data/processed", config_path="config/config.yaml", batch_size=1000, cleaned_dir="data/cleaned"):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.config = self._load_config(config_path)
        self.cleaner = TextCleaner()  # Fixed: No kwargs to avoid TypeError; assumes defaults in class
        self.normalizer = TextNormalizer()
        self.vectorizer = TFIDFVectorizerHandler()
        self.cleaned_dir = Path(cleaned_dir)
        self.cleaned_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size = batch_size
        
    def _load_config(self, path):
        config_path = Path(path)
        if not config_path.exists():
            logging.warning(f"Config file not found: {config_path}. Using defaults.")
            return {}
        
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logging.error(f"Error loading config: {e}")
            raise FeedException(e, sys)
            
        
    def save_to_csv(self, posts, file_name="reddit_data.csv"):
        try:
            df = pd.DataFrame(posts)
            csv_path = self.raw_dir / file_name
            df.to_csv(csv_path, index=False, encoding="utf-8")
            logging.info(f"Saved CSV with {len(df)} rows to {csv_path}")
            return csv_path
            
        except Exception as e:
            logging.error(f"Error saving CSV: {e}")
            raise FeedException(e, sys)
            
    def csv_to_json(self, csv_path, json_filename="reddit_data.json"):
        try:
            csv_path = Path(csv_path)
            if not csv_path.exists():
                raise FileNotFoundError(f"CSV file not found: {csv_path}")
            df = pd.read_csv(csv_path, encoding="utf-8")
            json_path = self.processed_dir / json_filename
            df.to_json(json_path, orient="records", lines=True)

            logging.info(f"Saved JSON (records) with {len(df)} rows to {json_path}")
            return json_path
        
        except Exception as e:
            logging.error(f"Error converting CSV to JSON: {e}")
            raise FeedException(e, sys)
            
    
    def _fetch_from_mongo(self):
        db_conf = self.config.get("database", {})
        url = db_conf.get("uri")
        name = db_conf.get("name")
        collection = db_conf.get("collection")
        
        if not all([url, name, collection]):
            logging.warning("MongoDB config missing. Attempting local fallback.")
            return self._load_fallback_local()
        
        try:
            client = MongoClient(url, serverSelectionTimeoutMS=5000)
            client.admin.command('ping')
            coll = client[name][collection]
            cursor = coll.find({}, {"_id": 0}).batch_size(self.batch_size)
            docs = [doc for doc in cursor]
            client.close()
            df = pd.DataFrame(docs)
            logging.info(f"Fetched {len(df)} documents from MongoDB.")
            return df if not df.empty else None
        
        except (ConnectionFailure, OperationFailure) as e:
            logging.error(f"MongoDB error: {e}. Using fallback.")
            return self._load_fallback_local()
        
    def _load_fallback_local(self):
        json_path = self.raw_dir / "reddit_data.json"
        if json_path.exists():
            try:
                df = pd.read_json(json_path, encoding="utf-8")
                logging.info(f"Loaded fallback JSON: {len(df)} records.")
                return df if not df.empty else None
            except Exception as e:
                logging.warning(f"Error loading JSON {json_path}: {e}")
                
        csv_path = self.raw_dir / "reddit_data.csv"
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path, encoding="utf-8")
                logging.info(f"Loaded fallback CSV: {len(df)} records.")
                return df if not df.empty else None
            except Exception as e:
                logging.warning(f"Error loading CSV {csv_path}: {e}")
                
        logging.error("No fallback data found in raw_dir.")
        return None

    def preprocess_data(self, vectorize=True, save_raw=False):
        logging.info("Starting preprocessing pipeline...")
        
        df = self._fetch_from_mongo()
        if df is None or df.empty:
            raise FeedException("No data available from MongoDB or fallback.", sys)
        
        df = df.rename(columns={"body": "selftext"})

        
        if save_raw:
            self.save_to_csv(df.to_dict('records'), "preprocess_raw_backup.csv")
            self.csv_to_json(self.raw_dir / "preprocess_raw_backup.csv")
            
        required_cols = ["title", "selftext"]
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            logging.error(f"DataFrame missing required columns: {missing_cols}")
            raise FeedException(
                f"Input DataFrame is missing required columns: {missing_cols}. "
                f"Expected columns: {required_cols}, but got: {list(df.columns)}",
                sys
            )
        
        df["raw_text"] = (
            df["title"].fillna("").astype(str) + " . " +
            df["selftext"].fillna("").astype(str)
        )
        
        logging.info("Applying text cleaning...")
        df["cleaned_text"] = df["raw_text"].apply(self.cleaner.clean)
        
        logging.info("Applying text normalization...")
        df["normalized_text"] = df["cleaned_text"].apply(self.normalizer.normalize)
        
        if vectorize:
            logging.info("Vectorizing text...")
            vectors = self.vectorizer.fit_transform(df["normalized_text"])
            vectorizer_path = self.processed_dir / "tfidf_model"
            self.vectorizer.save(str(vectorizer_path))
            vectors_path = self.processed_dir / "tfidf_vectors.npz"
            self.vectorizer.save_vectors(vectors, df.index.tolist(), str(vectors_path))
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        timestamp_csv = self.processed_dir / f"cleaned_normalized_{timestamp}.csv"
        df.to_csv(timestamp_csv, index=False, encoding="utf-8")
        logging.info(f"Timestamped CSV saved: {timestamp_csv}")
        
        fixed_csv = self.cleaned_dir / "cleaned_data.csv"
        df.to_csv(fixed_csv, index=False, encoding="utf-8")
        logging.info(f"Fixed cleaned CSV saved: {fixed_csv}")
            
        diagnostics = self._compute_diagnostics(df) 
        report_path = self.processed_dir / f"preprocess_report_{timestamp}.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(diagnostics, f, indent=2, ensure_ascii=False)
            
        logging.info(f"Diagnostics saved: {report_path}")
        
        preview_path = self.processed_dir / "preview_processed_head.csv"
        df.head(10).to_csv(preview_path, index=False, encoding="utf-8")
        logging.info("Preview saved.")
        
        logging.info("Preprocessing complete.")
        return df

    def _compute_diagnostics(self, df):
        return {
            "records_processed": len(df),
            "empty_raw_texts": int((df["raw_text"] == "").sum()),
            "empty_cleaned_texts": int((df["cleaned_text"] == "").sum()),
            "empty_normalized_texts": int((df["normalized_text"] == "").sum()),
            "duplicates_raw": int(df.duplicated(subset=["raw_text"]).sum()),
            "duplicates_cleaned": int(df.duplicated(subset=["cleaned_text"]).sum()),
            "null_titles": int(df["title"].isna().sum()),
            "null_selftext": int(df["selftext"].isna().sum()),
            "avg_raw_length": df["raw_text"].str.len().mean(),
            "avg_cleaned_length": df["cleaned_text"].str.len().mean(),
            "avg_normalized_length": df["normalized_text"].str.len().mean(),
            "timestamp": datetime.now().isoformat()
        }