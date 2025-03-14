import asyncio
import aiohttp
import logging
import os
from urllib.parse import urlsplit
from pathlib import Path
import ssl
import certifi
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
    "directories": {
        "images": "images",
        "logs": "logs"
    },
    "tor": {
        "control_host": "127.0.0.1",
        "control_port": 9051,
        "proxy": "socks5://127.0.0.1:9050",
        "password": "Ivangandon2005",
        "requests_before_ip_change": 40
    },
    "performance": {
        "max_connections": 300,
        "ip_change_delay": 0.05
    }
}

BASE_DIR = Path(__file__).resolve().parent
for key, rel_path in CONFIG["directories"].items():
    full_path = BASE_DIR / rel_path
    CONFIG["directories"][key] = str(full_path)
    full_path.mkdir(parents=True, exist_ok=True)

log_file = Path(CONFIG["directories"]["logs"]) / "image_saver.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file, encoding="utf-8")
    ]
)

ssl_setting = ssl.create_default_context(cafile=certifi.where())

request_counter = 0
semaphore = asyncio.Semaphore(CONFIG["performance"]["max_connections"])
download_semaphore = asyncio.Semaphore(CONFIG["performance"]["max_connections"])

def slugify(text):
    text = text.lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_-]+", "-", text)
    return text

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

async def download_image(session, image_url, object_data):
    global request_counter
    request_counter += 1
    if request_counter % CONFIG["tor"]["requests_before_ip_change"] == 0:
        logging.info(f"Reached {CONFIG['tor']['requests_before_ip_change']} image requests, changing IP...")
        if await change_tor_identity():
            await asyncio.sleep(CONFIG["performance"]["ip_change_delay"])
    try:
        image_title = object_data.get("title", "unknown").strip() or "untitled"
        image_slug = slugify(image_title)
        image_dir = Path(CONFIG["directories"]["images"]) / image_slug
        image_dir.mkdir(parents=True, exist_ok=True)
        file_extension = os.path.splitext(urlsplit(image_url).path)[1] or ".jpg"
        filename = f"{image_slug}{file_extension}"
        save_path = image_dir / filename
        if save_path.exists():
            logging.info(f"Image {filename} already exists, skipping")
            return True
        async with download_semaphore:
            async with session.get(image_url, ssl=ssl_setting) as response:
                if response.status == 200:
                    async with aiofiles.open(save_path, 'wb') as f:
                        async for chunk in response.content.iter_chunked(16384):
                            await f.write(chunk)
                    logging.info(f"Image saved: {filename}")
                    return True
                logging.warning(f"Failed to download image {image_url}, status: {response.status}")
                return False
    except Exception as e:
        logging.error(f"Error downloading image {image_url}: {e}")
        return False

async def save_images_for_objects(processed_objects, session):
    tasks = []
    for obj in processed_objects:
        if obj.get("primaryImage"):
            image_url = obj["primaryImage"]
            tasks.append(download_image(session, image_url, obj))
    results = await asyncio.gather(*tasks)
    return results
