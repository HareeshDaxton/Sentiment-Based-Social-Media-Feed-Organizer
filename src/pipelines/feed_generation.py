import os
import logging
import json
import heapq
from datetime import datetime
from typing import List, Dict, Any
from dataclasses import dataclass, field
from pandas import read_csv
from src.logging.logger import logging
from src.exception.exception import FeedException

@dataclass
class Post:
    id: str
    text: str
    sentiment: str
    confidence: float
    timestamp: datetime
    score: float = field(default=0.0)
    

class FeedGenerator:
    def __init__(self, input_path: str):
        self.input_path = input_path
        self.df = None
        self.feed: Dict[str, List[Post]] = {'positive': [], 'negative': [], 'neutral': []}
        logging.info(f"Initialized FeedGenerator with input: {input_path}")
        
    def _load_data(self):
        try:
            if not os.path.exists(self.input_path):
                raise FeedException(f"Input file not found: {self.input_path}")
            
            self.df = read_csv(self.input_path)
            required_cols = ['id', 'cleaned_text', 'predicted_sentiment_label', 'predicted_sentiment_confidence', 'created_utc']
            if not all(col in self.df.columns for col in required_cols):
                raise FeedException(f"Missing columns: {required_cols}. Ensure pipeline ran first.")
            
            logging.info(f"Loaded {len(self.df)} posts from {self.input_path}")
        
        except Exception as e:
            logging.error(f"Error loading data: {str(e)}")
            raise FeedException(f"Data loading failed: {str(e)}")
        
    def _calculate_priority(self, post: Post) -> float:
        now = datetime.now()
        recency = (now - post.timestamp).total_seconds() / 3600  # Hours ago
        recency_factor = max(1 / (1 + recency), 0.1)  # Decay
        return post.confidence * recency_factor
    
    def organize_feed(self):
        try:
            self._load_data()
            
            counter = 0
            for _, row in self.df.iterrows():
                sentiment = row['predicted_sentiment_label']
                if sentiment not in self.feed:
                    continue
                
                timestamp = datetime.fromtimestamp(row['created_utc'])
                post = Post(
                    id=str(row['id']),
                    text=row['cleaned_text'],
                    sentiment=sentiment,
                    confidence=row['predicted_sentiment_confidence'],
                    timestamp=timestamp
                )
                post.score = self._calculate_priority(post)
                
                heapq.heappush(self.feed[sentiment], (-post.score, counter, post))
                counter += 1
                
            for cat in self.feed:
                self.feed[cat] = [heapq.heappop(self.feed[cat])[2] for _ in range(len(self.feed[cat]))]
                
            logging.info("Feed organized successfully.")
            
        except Exception as e:
            logging.error(f"Error organizing feed: {str(e)}")
            raise FeedException(f"Feed organization failed: {str(e)}")
        
    def save_feed(self, output_dir: str = "data/processed") -> str:
        try:
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(output_dir, f"final_feed_{timestamp}.json")
            
            feed_dict = {}
            for cat, posts in self.feed.items():
                feed_dict[cat] = [
                    {
                        'id': p.id,
                        'text': p.text,
                        'confidence': p.confidence,
                        'timestamp': p.timestamp.isoformat(),
                        'priority_score': p.score
                    }
                    for p in posts
                ]
            
            with open(output_path, 'w') as f:
                json.dump(feed_dict, f, indent=2)
            
            logging.info(f"Saved feed to {output_path}")
            return output_path
        
        except Exception as e:
            logging.error(f"Error saving feed: {str(e)}")
            raise FeedException(f"Feed save failed: {str(e)}")