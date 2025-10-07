import os
import sys
import yaml
import time
from pymongo import MongoClient
from pymongo import UpdateOne
from dotenv import load_dotenv
from src.Ingestion.reddit_ingestor import RedditIngestion
from src.logging.logger import logging
from src.exception.exception import FeedException


def load_config(path : str = "config/config.yaml"):
    with open(path, 'r', encoding="utf-8") as f:
        return yaml.safe_load(f)
    
    
def connect_mongo(url : str, db_name:str):
    client = MongoClient(url)
    db = client[db_name]
    return db


def ingest_all(config_path = "config/config.yaml"):
    load_dotenv()
    cfg=load_config

    reddit_cfg = cfg.get("reddit", [])
    subreddits = reddit_cfg.get("subreddits", [])
    fetch_limit = reddit_cfg.get("fetch_limit",100)
    
    batch_size = reddit_cfg.get("batch_size", 25)
    per_batch_sleep = reddit_cfg.get("per_batch_sleep", 0.5)
    per_subreddit_sleep = reddit_cfg.get("per_subreddit_sleep", 1.5)
    max_retries = reddit_cfg.get("max_retries", 3)
    backoff_factor = reddit_cfg.get("backoff_factor", 2.0)
    
    mongodb_cfg = cfg.get("mongodb", {})
    mongodb_uri = mongodb_cfg.get("uri", "mongodb://localhost:27017/")    
    mongo_db = mongodb_cfg.get("database", "redditDB")
    mongo_coll = mongodb_cfg.get("collection", "posts")

    db = connect_mongo(mongodb_uri, mongo_db)
    collection = db[mongo_coll]
    
    try:
        collection.create_index('id', unique=True)
    except Exception as e:
        FeedException(e, sys)
        
    
    ingestor = RedditIngestion()
    
    total_fetched = 0
    total_saved = 0
    failures = []
    
    for i, sub in enumerate(subreddits, start=1):
        logging.info("(%d/%d) Ingesting r/%s", i, len(subreddits), sub)
        post = ingestor.fetch_post(
            sub, 
            limit=fetch_limit,
            batch_size=batch_size,
            per_batch_sleep=per_batch_sleep,
            max_retries=max_retries,
            backoff_factor=backoff_factor
    
        )
        
        if not post:
            logging.warning("No posts collected from r/%s", sub)
            failures.append(sub)
        else:
            total_fetched += len(post)
            
            ops = []
            for p in post:
                ops.append(UpdateOne({"id": p["id"]}, {"$set": p}, upsert=True))
                try:
                    result = collection.bulk_write(ops, ordered=False)
                    inserted = (result.upserted_count or 0) + (result.modified_count or 0)
                    total_saved += inserted
                    logging.info("r/%s -> upserted/modified: %d", sub, inserted)
                
                except Exception as e:
                    FeedException(e, sys)
            
        logging.debug("Sleeping %.2fs before next subreddit", per_subreddit_sleep)
        time.sleep(per_subreddit_sleep)
        
    
    logging.info("Ingestion complete. Total fetched: %d, saved(approx): %d, failures: %d", total_fetched, total_saved, len(failures))
    if failures:
        logging.warning("Failed to fully fetch: %s", failures)
