import yaml
from pathlib import Path
from src.logging.logger import logging
from src.exception.exception import FeedException
from src.ingestion.reddit_ingestor import RedditIngestion
from src.preprocessing.data_handler import DataHandler
from src.Storage.mongo_writer import MongoWriter
import os 
import time


def run_pipeline(config_path="config/config.yaml"):
    cfg_path = Path(config_path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(cfg_path, "r") as fh:
        cfg = yaml.safe_load(fh)

    reddit_cfg = cfg.get("reddit", {})
    data_cfg = cfg.get("data",{})
    mongodb_cfg = cfg.get("mongodb",{})
    
    
    reddit_client_id = os.getenv("REDDIT_CLIENT_ID") or reddit_cfg.get("client_id")
    reddit_client_secret= os.getenv("REDDIT_CLIENT_SECRET") or reddit_cfg.get("client_secret")
    reddit_user_agent = os.getenv("REDDIT_USER_AGENT", reddit_cfg.get("user_agent", "reddit_pipeline_app"))
    
            
    ingestor = RedditIngestion(
        client_id=reddit_client_id,
        client_secret=reddit_client_secret,
        user_agent=reddit_user_agent,
        
    )

    data_handler = DataHandler(
        raw_dir=data_cfg.get("base_dir", "data/raw"),
        processed_dir=data_cfg.get("processed_dir", "data/processed")
    )
    
    
    mongo_writter = MongoWriter(
        uri=mongodb_cfg.get("url", "mongodb://localhost:27017/"),
        db_name=mongodb_cfg.get("database", "reddit_db"),
        collection_name=mongodb_cfg.get("collection", "posts")
    
    )
    
    
    subreddits = reddit_cfg.get("subreddits", ["MachineLearning"])
    fetch_limit = reddit_cfg.get("fetch_limit", 100)
    
    per_subreddit_sleep = reddit_cfg.get("per_subreddit_sleep", 1.0)
    all_posts = []
    
    for subreddit in subreddits:
        logging.info(f"Fetching {fetch_limit} posts from r/{subreddit}")
        posts = ingestor.fetch_post(subreddit=subreddit, limit=fetch_limit)
        if posts:
            all_posts.extend(posts)

        else:
            logging.warning(f"No posts fetched from r/{subreddit}")
        time.sleep(per_subreddit_sleep)
        
    if not all_posts:
        logging.warning("No posts fetched from any subreddit; exiting pipeline.")
        return
    
    csv_path = data_handler.save_to_csv(all_posts, data_cfg.get("csv_filename", "reddit_data.csv"))
    json_path = data_handler.csv_to_json(csv_path, data_cfg.get("json_filename", "reddit_data.json"))
    
    inserted_count = mongo_writter.insert_from_json_file(json_path)
    logging.info(f"Pipeline complete. {inserted_count} documents inserted into MongoDB.")