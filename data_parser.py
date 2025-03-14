import asyncio
import aiohttp
import aiohttp_socks
import csv
import json
import logging
import os
from urllib.parse import urlsplit
from pathlib import Path
import ssl
import certifi
import sys
import re
import aiofiles

try:
    import uvloop
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
except ImportError:
    pass

from stem import Signal
from stem.control import Controller

CONFIG = {
    "metropolitan": {
        "base_url": "https://collectionapi.metmuseum.org/public/collection/v1",
        "max_objects": None
    },
    "directories": {
        "raw": "raw",
        "processed": "processed",
        "images": "images",
        "logs": "logs"
    },
    "tor": {
        "control_host": "127.0.0.1",
        "control_port": 9051,
        "proxy": "socks5://127.0.0.1:9050",
        "password": "mypassword",
        "requests_before_ip_change": 100
    },
    "performance": {
        "max_connections": 500,
        "batch_size": 500,
        "retry_delay": 0.05,
        "ip_change_delay": 0.01
    },
    "output": {
        "json_file": "processed_data.json",
        "csv_file": "processed_data.csv"
    }
}

BASE_DIR = Path(__file__).resolve().parent
for key, rel_path in CONFIG["directories"].items():
    full_path = BASE_DIR / rel_path
    CONFIG["directories"][key] = str(full_path)
    full_path.mkdir(parents=True, exist_ok=True)

log_file = Path(CONFIG["directories"]["logs"]) / "data_collection_async.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file, encoding="utf-8")
    ]
)

ssl_setting = ssl.create_default_context(cafile=certifi.where())
client_timeout = aiohttp.ClientTimeout(total=60, connect=10, sock_connect=10, sock_read=30)

request_counter = 0
semaphore = asyncio.Semaphore(CONFIG["performance"]["max_connections"])

async def change_tor_identity():
    try:
        with Controller.from_port(address=CONFIG["tor"]["control_host"], port=CONFIG["tor"]["control_port"]) as controller:
            controller.authenticate(password=CONFIG["tor"]["password"])
            controller.signal(Signal.NEWNYM)
            logging.info("Successfully changed Tor identity (IP address)")
            return True
    except Exception as e:
        logging.error(f"Error changing Tor identity: {e}")
        return False

async def fetch_json(session, url, retries=3):
    global request_counter
    request_counter += 1
    if request_counter % CONFIG["tor"]["requests_before_ip_change"] == 0:
        logging.info(f"Reached {CONFIG['tor']['requests_before_ip_change']} requests, changing IP...")
        if await change_tor_identity():
            await asyncio.sleep(CONFIG["performance"]["ip_change_delay"])
    for attempt in range(retries):
        try:
            async with semaphore:
                async with session.get(url, ssl=ssl_setting) as response:
                    if response.status == 200:
                        return await response.json()
                    logging.warning(f"Response code {response.status} for URL: {url}")
                    if response.status in [429, 403]:
                        if await change_tor_identity():
                            logging.info(f"Changing IP after status {response.status}, retrying request")
                            await asyncio.sleep(CONFIG["performance"]["ip_change_delay"])
                    await asyncio.sleep(CONFIG["performance"]["retry_delay"] * (attempt + 1))
        except Exception as e:
            logging.error(f"Error fetching {url}: {e}")
            await asyncio.sleep(CONFIG["performance"]["retry_delay"] * (attempt + 1))
    logging.error(f"Failed to fetch data after {retries} attempts for URL: {url}")
    return None

def slugify(text):
    text = text.lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_-]+", "-", text)
    return text

async def fetch_object_details(session, object_id):
    url = f"{CONFIG['metropolitan']['base_url']}/objects/{object_id}"
    return await fetch_json(session, url)

async def process_object(session, object_id, all_objects):
    try:
        object_data = await fetch_object_details(session, object_id)
        if not object_data:
            return None
        all_objects.append(object_data)
        return object_data
    except Exception as e:
        logging.error(f"Error processing object {object_id}: {e}")
        return None

async def get_all_object_ids(session):
    url = f"{CONFIG['metropolitan']['base_url']}/objects"
    data = await fetch_json(session, url)
    if not data or "objectIDs" not in data:
        logging.error("Failed to get list of object IDs")
        return []
    object_ids = data["objectIDs"]
    if CONFIG["metropolitan"]["max_objects"]:
        object_ids = object_ids[:CONFIG["metropolitan"]["max_objects"]]
    logging.info(f"Received {len(object_ids)} object IDs")
    return object_ids

def save_json(processed_objects):
    json_file = Path(CONFIG["directories"]["processed"]) / CONFIG["output"]["json_file"]
    if not json_file.exists():
        logging.info("JSON file does not exist, creating new file.")
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(processed_objects, f, ensure_ascii=False, indent=2)
    logging.info(f"Intermediate JSON saved: {len(processed_objects)} objects.")

def save_processed_data(processed_objects):
    save_json(processed_objects)
    csv_file = Path(CONFIG["directories"]["processed"]) / CONFIG["output"]["csv_file"]
    if processed_objects:
        csv_headers = processed_objects[0].keys()
        with open(csv_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=csv_headers)
            writer.writeheader()
            writer.writerows(processed_objects)
        logging.info(f"Saved {len(processed_objects)} objects to CSV file: {csv_file}")
    else:
        logging.warning("No data to save to CSV.")

async def process_and_save_data():
    all_objects = []
    if not await change_tor_identity():
        logging.error("Failed to establish initial connection to Tor control port.")
        return
    logging.info("Initial connection to Tor successful")
    connector = aiohttp_socks.ProxyConnector.from_url(
        CONFIG["tor"]["proxy"],
        ssl=ssl_setting,
        limit=CONFIG["performance"]["max_connections"]
    )
    async with aiohttp.ClientSession(connector=connector, timeout=client_timeout) as session:
        object_ids = await get_all_object_ids(session)
        if not object_ids:
            logging.error("Failed to get object IDs, terminating")
            return
        batch_size = CONFIG["performance"]["batch_size"]
        for i in range(0, len(object_ids), batch_size):
            batch = object_ids[i:i + batch_size]
            tasks = [process_object(session, object_id, all_objects) for object_id in batch]
            await asyncio.gather(*tasks)
            logging.info(f"Processed batch {i // batch_size + 1}/{(len(object_ids) + batch_size - 1) // batch_size}, total objects processed: {len(all_objects)}")
            save_json(all_objects)
    if all_objects:
        save_processed_data(all_objects)
        logging.info(f"Processing complete. Total processed: {len(all_objects)} objects.")
    else:
        logging.warning("Processing completed but no objects were received.")

def main():
    logging.info("Launching scraper with Tor integration")
    try:
        import stem
    except ImportError:
        logging.error("Package stem not found. Install it: pip install stem")
        print("Install stem: pip install stem")
        sys.exit(1)
    try:
        import aiohttp_socks
    except ImportError:
        logging.error("Package aiohttp_socks not found. Install it: pip install aiohttp_socks")
        print("Install aiohttp_socks: pip install aiohttp_socks")
        sys.exit(1)
    asyncio.run(process_and_save_data())
    logging.info("Program completed")

if __name__ == "__main__":
    main()
