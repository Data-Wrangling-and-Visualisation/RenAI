import asyncio
import json
import logging
import os
from pathlib import Path
import motor.motor_asyncio
from pymongo import UpdateOne
import certifi
from typing import List, Dict
import time
import random

CONFIG = {
    "directories": {
        "json": "json_data",
        "logs": "logs"
    },
    "mongodb": {
        "connection_string": "mongodb+srv://dimaste7878:Lbvfcntlbvfcnt7878@renai.txwp5.mongodb.net/?retryWrites=false&w=majority&appName=renai&connectTimeoutMS=30000&socketTimeoutMS=45000",
        "database_name": "metropolitan",
        "collection_name": "objects",
        "batch_size": 500,
        "clear_before_upload": True
    },
    "performance": {
        "concurrent_tasks": 5,
        "max_retries": 5,
        "retry_delay_base": 1,
        "retry_delay_max": 30
    }
}

BASE_DIR = Path(__file__).resolve().parent
for key, rel_path in CONFIG["directories"].items():
    full_path = BASE_DIR / rel_path
    CONFIG["directories"][key] = str(full_path)
    full_path.mkdir(parents=True, exist_ok=True)

log_file = Path(CONFIG["directories"]["logs"]) / "mongodb_upload.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file, encoding="utf-8")
    ]
)

class MongoDBUploader:
    def __init__(self, connection_string: str, database_name: str, collection_name: str):
        self.client = motor.motor_asyncio.AsyncIOMotorClient(
            connection_string, 
            tls=True, 
            tlsCAFile=certifi.where(),
            serverSelectionTimeoutMS=30000,
            maxPoolSize=50,  
            minPoolSize=10,
            maxIdleTimeMS=60000
        )
        self.db = self.client[database_name]
        self.collection = self.db[collection_name]
        logging.info(f"Connecting to MongoDB: {database_name}.{collection_name}")

    async def ping(self) -> bool:
        try:
            await self.db.command("ping")
            logging.info("Successful connection to MongoDB")
            return True
        except Exception as e:
            logging.error(f"Error connecting to MongoDB: {e}")
            return False

    async def bulk_write_with_retries(self, operations: List[Dict], max_retries=None) -> int:
        if not operations:
            return 0

        if max_retries is None:
            max_retries = CONFIG["performance"]["max_retries"]

        bulk_ops = [UpdateOne(
            {"objectID": doc["objectID"]},
            {"$set": doc},
            upsert=True
        ) for doc in operations if "objectID" in doc]

        if not bulk_ops:
            return 0

        for attempt in range(max_retries):
            try:
                result = await self.collection.bulk_write(bulk_ops, ordered=False)
                return result.upserted_count + result.modified_count
            except Exception as e:
                is_retryable = any(label in str(e) for label in 
                                  ["RetryableWriteError", "ECONNRESET", "not primary", 
                                   "NotWritablePrimary", "SocketError"])
                
                if not is_retryable or attempt >= max_retries - 1:
                    logging.error(f"Non-retryable error or max retries reached: {e}")
                    if attempt == max_retries - 1 and len(bulk_ops) > 1:
                        mid = len(bulk_ops) // 2
                        logging.warning(f"Attempting to split batch. Processing first {mid} operations.")
                        count1 = await self.bulk_write_with_retries(operations[:mid], 2)
                        logging.warning(f"Processing remaining {len(operations) - mid} operations.")
                        count2 = await self.bulk_write_with_retries(operations[mid:], 2)
                        return count1 + count2
                    return 0
                
                base_delay = CONFIG["performance"]["retry_delay_base"] * (2 ** attempt)
                max_delay = CONFIG["performance"]["retry_delay_max"]
                delay = min(base_delay + random.uniform(0, 1), max_delay)
                
                logging.warning(f"Write error (attempt {attempt+1}/{max_retries}): {e}")
                logging.info(f"Retrying in {delay:.2f} seconds...")
                await asyncio.sleep(delay)
        
        return 0
    
    async def clear_collection(self) -> bool:
        max_retries = CONFIG["performance"]["max_retries"]
        
        for attempt in range(max_retries):
            try:
                count_before = await self.collection.count_documents({})
                logging.info(f"Starting collection cleanup. Current number of documents: {count_before}")
                
                result = await self.collection.delete_many({})
                
                if result.deleted_count == count_before:
                    logging.info(f"Collection successfully purged. {result.deleted_count} documents deleted.")
                    return True
                else:
                    logging.warning(f"Partial collection cleanup. Removed {result.deleted_count} from {count_before} documents.")
                    return True
            except Exception as e:
                if "not primary" in str(e) or "ECONNRESET" in str(e):
                    if attempt < max_retries - 1:
                        delay = CONFIG["performance"]["retry_delay_base"] * (2 ** attempt)
                        logging.warning(f"Error clearing collection (attempt {attempt+1}/{max_retries}): {e}")
                        logging.info(f"Retrying in {delay} seconds...")
                        await asyncio.sleep(delay)
                    else:
                        logging.error(f"Failed to clear collection after {max_retries} attempts: {e}")
                        return False
                else:
                    logging.error(f"Error clearing collection: {e}")
                    return False
        
        return False

    async def close(self):
        if self.client:
            self.client.close()
            logging.info("Connection to MongoDB closed")

async def read_json_file(file_path: Path) -> List[Dict]:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data if isinstance(data, list) else [data]
    except Exception as e:
        logging.error(f"Error reading file {file_path}: {e}")
        return []

async def process_file(file_path: Path, uploader: MongoDBUploader) -> int:
    try:
        data = await read_json_file(file_path)
        if not data:
            return 0
            
        batch_size = CONFIG["mongodb"]["batch_size"]
        total_processed = 0
        
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            count = await uploader.bulk_write_with_retries(batch)
            total_processed += count
            if i + batch_size < len(data):
                logging.info(f"Processed batch {i//batch_size + 1}/{(len(data) + batch_size - 1)//batch_size} "
                            f"from {file_path.name} ({count} objects)")
                await asyncio.sleep(0.5)
                
        logging.info(f"Loaded {total_processed} objects from file {file_path.name}")
        return total_processed
    except Exception as e:
        logging.error(f"Error processing file {file_path}: {e}")
        return 0

async def process_files_in_batches(files: List[Path], uploader: MongoDBUploader, batch_size: int):
    total_count = 0
    for i in range(0, len(files), batch_size):
        batch_files = files[i:i+batch_size]
        logging.info(f"Processing batch of {len(batch_files)} files ({i+1}-{min(i+batch_size, len(files))} of {len(files)})")
        
        tasks = [process_file(file_path, uploader) for file_path in batch_files]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for idx, result in enumerate(results):
            if isinstance(result, Exception):
                logging.error(f"Exception processing file {batch_files[idx].name}: {result}")
            else:
                total_count += result
        
        if i + batch_size < len(files):
            await asyncio.sleep(2)
    
    return total_count

async def main():
    start_time = time.time()
    
    uploader = MongoDBUploader(
        CONFIG["mongodb"]["connection_string"],
        CONFIG["mongodb"]["database_name"],
        CONFIG["mongodb"]["collection_name"]
    )
    
    for attempt in range(3):
        if await uploader.ping():
            break
        if attempt < 2:
            logging.warning(f"Retrying connection to MongoDB (attempt {attempt+1}/3)...")
            await asyncio.sleep(5)
    else:
        logging.error("Failed to connect to MongoDB after multiple attempts. Shutting down.")
        await uploader.close()
        return
    
    try:
        if CONFIG["mongodb"]["clear_before_upload"]:
            logging.info("Starting to clean up the collection before loading...")
            if not await uploader.clear_collection():
                logging.error("Failed to clear collection. Shutting down.")
                return
            
        json_dir = Path(CONFIG["directories"]["json"])
        json_files = list(json_dir.glob("*.json"))
        
        if not json_files:
            logging.warning(f"JSON files not found in directory {json_dir}")
            return
        
        logging.info(f"Found {len(json_files)} JSON files to upload")
        
        concurrent_tasks = CONFIG["performance"]["concurrent_tasks"]
        batch_size = min(concurrent_tasks, len(json_files))
        
        total_objects = await process_files_in_batches(json_files, uploader, batch_size)
        
        elapsed = time.time() - start_time
        logging.info(f"Loading complete. Loaded {total_objects} objects in {elapsed:.2f} seconds")
        
    except Exception as e:
        logging.error(f"Error during loading: {e}")
    finally:
        await uploader.close()

if __name__ == "__main__":
    logging.info("Running a script to load JSON files into MongoDB")
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Aborted by user")
    finally:
        logging.info("Work completed")
