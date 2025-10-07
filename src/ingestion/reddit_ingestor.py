import os
import sys
import time
from typing import List, Dict, Any, Optional
import praw
from src.logging.logger import logging
from src.exception.exception import FeedException


class RedditIngestion:
    def __init__(self, client_id=None, client_secret=None, user_agent=None):
        client_id = client_id or os.getenv("REDDIT_CLIENT_ID")
        client_secret = client_secret or os.getenv("REDDIT_CLIENT_SECRET")
        user_agent = user_agent or os.getenv("REDDIT_USER_AGENT", "reddit_pipeline_app")
        
        if not (client_id or client_secret):
            logging.error("Missing Reddit credentials in environment variables.")
            raise ValueError("Provide REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET as environment variables.")

        self.reddit = praw.Reddit(client_id=client_id, client_secret=client_secret, user_agent=user_agent)
        

    def fetch_post(
        self,
        subreddit: str,
        limit : int = 100,
        batch_size: int = 25,
        per_batch_sleep : float = 0.5,
        max_retries: int = 3,
        backoff_factor: float = 2.0
    ):

        posts = []
        attempt = 0
        
        while attempt <= max_retries:
            try:
                logging.info("Start fetching r/%s (limit=%d) (attempt %d)", subreddit, limit, attempt + 1)
                gen = self.reddit.subreddit(subreddit).hot(limit=limit)
                for idx, post in enumerate(gen, start=1):
                    posts.append({
                        "id" : post.id,
                        "subreddit" : subreddit,
                        "title" : post.title,
                        "score" : int(post.score) if post.score is not None else 0,
                        "url" : post.url,
                        "permalink" : getattr(post, "permalink", ""),
                        "num_comments" : int(post.num_comments) if post.num_comments is not None else 0,
                        "body" : post.selftext or "",
                        "created_utc" : getattr(post, "created_utc", None),
                        "is_self" : getattr(post, "is_self", None),
                        
                    })
                    
                    if idx % batch_size == 0:
                        logging.debug("Fetched %d items from r/%s — sleeping %.2fs", idx, subreddit, per_batch_sleep)
                        time.sleep(per_batch_sleep)
                        
                logging.info("Completed fetching r/%s — collected %d posts", subreddit, len(posts))
                return posts
                
            except Exception as e:
                attempt += 1
                logging.warning("Error while fetching r/%s: %s (attempt %d/%d)", subreddit, str(e), attempt, max_retries)
                
                if attempt > max_retries:
                    logging.exception("Max retries reached for r/%s — returning partial results (%d collected).", subreddit, len(posts))
                    return posts

                sleep_for = backoff_factor ** attempt
                logging.info("Retrying r/%s after %.1fs backoff", subreddit, sleep_for)
                time.sleep(sleep_for)
                
            
        return posts
    
    