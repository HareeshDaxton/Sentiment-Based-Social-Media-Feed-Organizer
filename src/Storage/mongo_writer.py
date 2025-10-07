import json
import sys
import os
import itertools
from pathlib import Path
from typing import List, Dict, Iterable
from pymongo import MongoClient, errors,ASCENDING
from pymongo.operations import UpdateMany
from src.logging.logger import logging
from src.exception.exception import FeedException


def chunked(iterable, size):
    it  = iter(iterable)
    while True:
        chunk = list(itertools.islice(it, size))
        if not chunk:
            break
        yield chunk
        
class MongoWriter:
    def __init__(self, uri,db_name= "reddit_db", collection_name="posts", connect_timmeout_ms= 5000):
        self.client = MongoClient(uri,serverSelectionTimeoutMS=connect_timmeout_ms)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]

        try:
           self.collection.create_index([("id", ASCENDING)], unique=True, sparse=True)
           logging.info("Ensured unique index on 'id'.")       
        except errors.OperationFailure as e:
              FeedException(f"Failed to create index on 'id': {e}", sys)
              
              
    def close(self):
        self.client.close()
        logging.info("Mongo client closed.")
        
    
    def insert_documents(self, docs, batch_size=1000):
        if not docs:
            logging.warning("No documents to insert.")
            return 0
        
        total_insterted = 0
        for batch in chunked(docs,batch_size):
            try:
                result = self.collection.insert_many(batch,ordered=False)
                total_insterted += len(result.inserted_ids)
                
            except errors.BulkWriteError as bwe:
                write_result = bwe.details
                logging.warning("BulkWriteError: %s", write_result.get("writeErrors", [])[:3])    
                n_inserted = write_result.get("nInserted") 
                if isinstance(n_inserted,int):
                    total_insterted += n_inserted
                    
                else:
                   logging.debug("Unable to determine precise number inserted from BulkWriteError details.") 
                   
        logging.info("Inserted %d new documents (attempted %d).", total_insterted, len(docs))
        return total_insterted
    
    
    def insert_from_json_file(self, json_path, use_upsert=False):
        json_path = Path(json_path)
        if not json_path.exists():
            logging.error("JSON file not found: %s", json_path)
            raise FileNotFoundError(json_path)
        
        docs = []
        bad_lines = 0
        with open(json_path, 'r', encoding="utf-8") as fh:
            for i, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    docs.append(json.loads(line))
                except json.JSONDecodeError:
                    logging.warning("Skipping malformed JSON line %d", i)
                    bad_lines += 1
                    
        logging.info("Loaded %d documents from %s (skipped %d malformed lines).", len(docs), json_path, bad_lines)
        if use_upsert:
            return self.upsert_documents(docs)
        return self.insert_documents(docs)