

# import os
# from dotenv import load_dotenv

# load_dotenv()


# import os
# from dotenv import load_dotenv
# from src.pipelines.reddit_pipeline import run_pipeline
# from src.logging.logger import logging

# def main():
#     try:
#         load_dotenv()
#         logging.info("Starting Reddit pipeline...")
#         run_pipeline("config/config.yaml")
#         logging.info("Pipeline completed successfully!")
#     except Exception as e:
#         logging.error(f"Pipeline failed: {str(e)}")
#         raise

# if __name__ == "__main__":
#     main()


# import os
# from dotenv import load_dotenv
# from src.preprocessing.data_handler import DataHandler
# from src.logging.logger import logging

# def main():
#     try:
#         load_dotenv()
#         logging.info("Starting data preprocessing pipeline...")
        
#         # Initialize data handler with necessary directories
#         data_handler = DataHandler(
#             raw_dir="data/raw",
#             processed_dir="data/processed",
#             cleaned_dir="data/cleaned",
#             config_path="config/config.yaml",
#             batch_size=1000
#         )
        
#         # Run preprocessing pipeline
#         logging.info("Starting preprocessing...")
#         processed_data = data_handler.preprocess_data(
#             vectorize=True,  # Enable TF-IDF vectorization
#             save_raw=True    # Save backup of raw data
#         )
        
#         logging.info("Preprocessing completed successfully!")
#         logging.info("You can find the processed data in the data/processed directory")
#         logging.info("Cleaned data is available in data/cleaned/cleaned_data.csv")
        
#     except Exception as e:
#         logging.error(f"Preprocessing failed: {str(e)}")
#         raise

# if __name__ == "__main__":
#     main()



import sys
import os
import glob
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.models.trainer import SentimentTrainer
from src.pipelines.sentiment_pipeline import run_sentiment_pipeline
from src.pipelines.feed_generation import FeedGenerator
from src.logging.logger import logging
from src.exception.exception import FeedException

if __name__ == "__main__":
    trainer = SentimentTrainer()
    labeled_csv = "data/labeled/fully_cleaned.csv"
    metrics = trainer.train(labeled_csv)
    trainer.save_artifacts('logistic_regression')
    print("Training metrics:", metrics)
    
    run_sentiment_pipeline(model_type='transformer', use_mongo=False)
    sentiment_files = glob.glob("data/processed/sentiment_output_*.csv")
    
    if sentiment_files:
        latest_csv = max(sentiment_files, key=os.path.getctime)
        print(f"Using latest sentiment file: {latest_csv}")
        feed_gen = FeedGenerator(latest_csv)
        feed_gen.organize_feed()
        feed_gen.save_feed()
        
    else:
        print("No sentiment output found. Run pipeline first.")
    
    print("End-to-end pipeline completed!")