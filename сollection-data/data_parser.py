import asyncio
import aiohttp
import aiohttp_socks
import json
import logging
import os
from pathlib import Path
import ssl
import certifi
import time
from typing import Dict, List
from aiohttp_retry import RetryClient, ExponentialRetry
from contextlib import asynccontextmanager

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
        "processed": "processed",
        "images": "images",
        "logs": "logs",
        "json": "json_data"
    },
    "tor": {
        "enabled": False,
        "control_host": "127.0.0.1",
        "control_port": 9051,
        "password": "mypassword",
        "requests_before_ip_change": 70,
        "proxy": "socks5://127.0.0.1:9050"
    },
    "performance": {
        "max_connections": 1500,
        "batch_size": 1000,
        "retry_delay": 0.01,
        "ip_change_delay": 0.0001,
    },
    "retry": {
        "attempts": 5,
        "start_timeout": 0.1,
        "max_timeout": 10,
        "factor": 2,
        "statuses": [429, 500, 502, 503, 504]
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
client_timeout = aiohttp.ClientTimeout(total=20, connect=5, sock_read=10)

request_counter = 0
semaphore = asyncio.Semaphore(CONFIG["performance"]["max_connections"])
start_time = time.time()
objects_processed = 0
total_objects = 0
producers_running = asyncio.Event()
batch_queue = asyncio.Queue()

class TorController:
    def __init__(self, host, port, password):
        self.host = host
        self.port = port
        self.password = password
        self.controller = None

    async def connect(self):
        try:
            self.controller = Controller.from_port(
                address=self.host,
                port=self.port
            )
            self.controller.authenticate(password=self.password)
            logging.info("Successful connection to Tor controller")
            return True
        except Exception as e:
            logging.error(f"Error connecting to Tor: {e}")
            return False

    async def change_identity(self):
        if not self.controller and not await self.connect():
            return False

        try:
            self.controller.signal(Signal.NEWNYM)
            logging.info("Tor IP address changed successfully")
            await asyncio.sleep(0.001)
            return True
        except Exception as e:
            logging.error(f"Error changing IP: {e}")
            await self.close()
            await asyncio.sleep(0.5)
            return await self.connect() and await self.change_identity()

    async def close(self):
        if self.controller:
            try:
                self.controller.close()
                self.controller = None
            except Exception as e:
                logging.error(f"Controller close error: {e}")

@asynccontextmanager
async def get_retry_client(use_tor=False):
    retry_opts = ExponentialRetry(
        attempts=CONFIG["retry"]["attempts"],
        start_timeout=0.1,
        max_timeout=10,
        factor=2,
        statuses=[429, 500, 502, 503, 504]
    )

    connector = None

    if use_tor and CONFIG["tor"]["enabled"]:
        connector = aiohttp_socks.ProxyConnector.from_url(
            CONFIG["tor"]["proxy"],
            ssl=ssl_setting,
            limit=CONFIG["performance"]["max_connections"],
            ttl_dns_cache=300
        )
    else:
        connector = aiohttp.TCPConnector(
            limit=CONFIG["performance"]["max_connections"], 
            ttl_dns_cache=300, 
            ssl=ssl_setting
        )

    retry_client = RetryClient(
        connector=connector,
        retry_options=retry_opts,
        timeout=client_timeout,
        raise_for_status=False
    )

    try:
        yield retry_client
    finally:
        await retry_client.close()

async def save_batch_to_json(batch: List[Dict], batch_number: int):
    if not batch:
        return
        
    filepath = Path(CONFIG["directories"]["json"]) / f"batch_{batch_number}.json"
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(batch, f, ensure_ascii=False, indent=4)
        logging.info(f"Saved batch {batch_number} with {len(batch)} objects in JSON")
    except Exception as e:
        logging.error(f"Error saving batch {batch_number} to JSON: {e}")

async def batch_consumer():
    batch = []
    batch_id = 0
    batch_size = CONFIG["performance"]["batch_size"]
    
    while producers_running.is_set() or not batch_queue.empty():
        try:
            obj = await asyncio.wait_for(batch_queue.get(), timeout=5)
            batch.append(obj)
            
            if len(batch) >= batch_size:
                batch_id += 1
                await save_batch_to_json(batch, batch_id)
                batch = []
        except asyncio.TimeoutError:
            continue
        except Exception as e:
            logging.error(f"Error in batch_consumer: {e}")
    
    if batch:
        batch_id += 1
        await save_batch_to_json(batch, batch_id)

async def process_object(client, object_id, tor_controller, use_tor=False):
    global request_counter, objects_processed
    max_retries = 3
    retry_delay = 1.0

    for retry in range(max_retries):
        try:
            if use_tor and CONFIG["tor"]["enabled"]:
                if request_counter % CONFIG["tor"]["requests_before_ip_change"] == 0:
                    if await tor_controller.change_identity():
                        await asyncio.sleep(CONFIG["performance"]["ip_change_delay"] * 3)
                    else:
                        await asyncio.sleep(1.0)
                        continue

            async with semaphore:
                request_counter += 1
                url = f"{CONFIG['metropolitan']['base_url']}/objects/{object_id}"
                try:
                    async with client.get(url) as response:
                        data = await response.json()
                        if data:
                            await batch_queue.put(data)

                            objects_processed += 1
                            if objects_processed % 1000 == 0:
                                elapsed = time.time() - start_time
                                logging.info(
                                    f"Progress: {objects_processed}/{total_objects} "
                                    f"({objects_processed / elapsed:.2f} obj/sec)"
                                )
                            return data
                except aiohttp.ClientError as e:
                     logging.warning(f"Error requesting {url}: {e}")
                     await asyncio.sleep(retry_delay * (2 ** retry))
                except asyncio.TimeoutError as e:
                    logging.warning(f"Timeout for request {url}: {e}")
                    await asyncio.sleep(retry_delay * (2 ** retry))
            return None
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            logging.warning(f"Error processing {object_id} (attempt {retry + 1}/{max_retries}): {e}")
            await asyncio.sleep(retry_delay * (2 ** retry))
            if use_tor and ("Couldn't connect to proxy" in str(e) or "Connection reset by peer" in str(e)):
                await tor_controller.close()
                await asyncio.sleep(1.0)
                await tor_controller.connect()
        except Exception as e:
            logging.error(f"Unexpected error processing {object_id}: {e}")
            return None

    logging.error(f"Failed to process {object_id} after {max_retries} attempts")
    return None

async def main_workflow():
    global total_objects

    tor_controller = TorController(
        CONFIG["tor"]["control_host"],
        CONFIG["tor"]["control_port"],
        CONFIG["tor"]["password"]
    )

    tor_connected = True
    use_tor = CONFIG["tor"]["enabled"]
    if use_tor:
        tor_connected = False
        for attempt in range(3):
            if await tor_controller.connect():
                tor_connected = True
                break
            await asyncio.sleep(2)

        if not tor_connected:
            logging.error("Failed to connect to Tor after several attempts. Continuing without Tor.")

    producers_running.set()
    
    consumer_task = asyncio.create_task(batch_consumer())

    try:
        async with get_retry_client(use_tor=use_tor) as client:
            ids_url = f"{CONFIG['metropolitan']['base_url']}/objects"
            try:
                ids_data = await client.get(ids_url)
                object_ids_data = await ids_data.json()
                object_ids = object_ids_data.get("objectIDs", [])
                if CONFIG["metropolitan"]["max_objects"]:
                    object_ids = object_ids[:CONFIG["metropolitan"]["max_objects"]]
            except Exception as e:
                logging.error(f"Error getting list ID: {e}")
                return

            total_objects = len(object_ids)
            logging.info(f"Total {total_objects} object IDs received")

            batch_size = CONFIG["performance"]["batch_size"]
            for i in range(0, len(object_ids), batch_size):
                current_batch = object_ids[i:i + batch_size]
                logging.info(f"Processing batch {i // batch_size + 1}/{(len(object_ids) + batch_size - 1) // batch_size} ({len(current_batch)} objects)")

                tasks = [
                    process_object(client, obj_id, tor_controller, use_tor) for obj_id in current_batch
                ]
                await asyncio.gather(*tasks)

                await asyncio.sleep(0.001)
    except Exception as e:
        logging.error(f"Error in main workflow: {e}")
    finally:
        producers_running.clear()
        
        await consumer_task
        
        if use_tor and CONFIG["tor"]["enabled"]:
            await tor_controller.close()

if __name__ == "__main__":
    logging.info("Launching the optimized data collector")
    try:
        asyncio.run(main_workflow())
    except KeyboardInterrupt:
        logging.info("Aborted by user")
    finally:
        logging.info("Work completed")
