import asyncio
import aiohttp
import logging
import ssl
import certifi
import os
from pathlib import Path
from urllib.parse import urlsplit
import re
import aiofiles
from stem import Signal
from stem.control import Controller

CONFIG = {
    "directories": {"images": "images", "logs": "logs"},
    "tor": {
        "control_host": "127.0.0.1",
        "control_port": 9051,
        "proxy": "socks5://127.0.0.1:9050",
        "password": "your_tor_password",
        "requests_before_ip_change": 10
    },
    "performance": {
        "max_connections": 20
    }
}

BASE_DIR = Path(__file__).resolve().parent
for key, rel_path in CONFIG["directories"].items():
    full_path = BASE_DIR / rel_path
    full_path.mkdir(parents=True, exist_ok=True)

log_file = BASE_DIR / CONFIG["directories"]["logs"] / "image_saver.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(log_file, encoding="utf-8")]
)

ssl_setting = ssl.create_default_context(cafile=certifi.where())

request_counter = 0
request_counter_lock = asyncio.Lock()

def slugify(text):
    text = text.lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_-]+", "-", text)
    return text

class TorController:
    def __init__(self, host, port, password):
        self.host = host
        self.port = port
        self.password = password
        self.controller = None

    async def connect(self):
        try:
            self.controller = Controller.from_port(address=self.host, port=self.port)
            self.controller.authenticate(password=self.password)
            logging.info("Connected to Tor controller")
            return True
        except Exception as e:
            logging.error(f"Tor connection error: {e}")
            return False

    async def change_identity(self):
        if not self.controller and not await self.connect():
            return False
        try:
            self.controller.signal(Signal.NEWNYM)
            logging.info("Tor IP changed")
            await asyncio.sleep(0.5)
            return True
        except Exception as e:
            logging.error(f"Tor identity change error: {e}")
            return False

    async def close(self):
        if self.controller:
            self.controller.close()

async def fetch_json(session, url):
    try:
        async with session.get(url, ssl=ssl_setting) as response:
            if response.status == 200:
                return await response.json()
            logging.error(f"Failed to fetch JSON from {url}, status: {response.status}")
    except Exception as e:
        logging.error(f"Exception fetching JSON from {url}: {e}")
    return None

async def download_image(session, image_url, title, tor_controller):
    global request_counter
    async with request_counter_lock:
        request_counter += 1
        if request_counter % CONFIG["tor"]["requests_before_ip_change"] == 0:
            logging.info("Changing Tor IP...")
            await tor_controller.change_identity()
    
    slug = slugify(title) or "unknown"
    ext = os.path.splitext(urlsplit(image_url).path)[1] or ".jpg"
    save_path = BASE_DIR / CONFIG["directories"]["images"] / f"{slug}{ext}"
    
    if save_path.exists():
        logging.info(f"Image {slug}{ext} already exists, skipping")
        return
    
    try:
        async with session.get(image_url, ssl=ssl_setting) as response:
            if response.status == 200:
                async with aiofiles.open(save_path, "wb") as f:
                    content = await response.read()
                    await f.write(content)
                logging.info(f"Saved image: {slug}{ext}")
            else:
                logging.warning(f"Failed to download {image_url}, status: {response.status}")
    except Exception as e:
        logging.error(f"Error downloading image {image_url}: {e}")

async def process_object(object_id, session, semaphore, tor_controller):
    async with semaphore:
        obj_url = f"https://collectionapi.metmuseum.org/public/collection/v1/objects/{object_id}"
        obj_data = await fetch_json(session, obj_url)
        if obj_data and obj_data.get("primaryImage"):
            await download_image(session, obj_data["primaryImage"], obj_data.get("title", "Untitled"), tor_controller)

async def process_objects(tor_controller):
    semaphore = asyncio.Semaphore(CONFIG["performance"]["max_connections"])
    async with aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(ssl=ssl_setting),
        timeout=aiohttp.ClientTimeout(total=60)
    ) as session:
        objects_url = "https://collectionapi.metmuseum.org/public/collection/v1/objects"
        data = await fetch_json(session, objects_url)
        if not data or "objectIDs" not in data:
            logging.error("No object IDs found in API response")
            return
        
        object_ids = data["objectIDs"][:]
        tasks = [
            process_object(object_id, session, semaphore, tor_controller)
            for object_id in object_ids
        ]
        await asyncio.gather(*tasks)

async def main():
    tor_controller = TorController(
        CONFIG["tor"]["control_host"],
        CONFIG["tor"]["control_port"],
        CONFIG["tor"]["password"]
    )
    if not await tor_controller.connect():
        logging.error("Tor connection failed. Exiting.")
        return
    try:
        await process_objects(tor_controller)
    finally:
        await tor_controller.close()

if __name__ == "__main__":
    asyncio.run(main())
