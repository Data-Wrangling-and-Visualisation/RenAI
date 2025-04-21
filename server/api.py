import os
import requests
import numpy as np
from flask import Flask, jsonify, request, send_file, abort, Response, send_from_directory
from PIL import Image, UnidentifiedImageError, ImageStat, ImageFilter, ImageEnhance
from io import BytesIO
import sys
import matplotlib.pyplot as plt
import cv2
from flask_cors import CORS
from sklearn.cluster import KMeans
import scipy.stats as stats
from skimage import feature, color, exposure, util, filters
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import shannon_entropy
from scipy.spatial.distance import pdist, squareform
import time
import random
import pickle
import json
import traceback
from urllib.parse import urljoin, urlparse
from collections import defaultdict
import base64
import tempfile
import uuid
import io
import math
import string
import logging # Added for logging
from flask_cors import cross_origin
from datetime import datetime

# Import the fetch_met_artworks function from the utils module
from utils import fetch_met_artworks

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# --- End Logging Setup ---

DEFAULT_LIMIT = 20

# --- Initialize Flask App ---
app = Flask(__name__)
# --- Enable CORS globally ---
# Allow all origins for development. For production, you might want to restrict this
# to your specific frontend domain, e.g., CORS(app, origins=["http://your-frontend-domain.com"])
CORS(app) 
# --- End CORS Setup ---

current_dir = os.path.dirname(os.path.abspath(__file__))
processing_data_path = os.path.join(current_dir, '../processing-data')
sys.path.append(processing_data_path)

# --- ML Model Imports ---
try:
    from embeddings import generate_embedding
    from Gradcam import calculate_gradcam
    from hitmaps import generate_heatmap
    logger.info("Successfully imported ML processing functions from processing-data.")
except ImportError as e:
    logger.error(f"Failed to import ML functions from processing-data: {e}. Analysis features will be limited.")
    # Define improved dummy functions that return data of the expected shape
    def generate_embedding(img):
        """Dummy function that returns a fixed embedding of expected shape."""
        logger.warning("Using dummy generate_embedding function. Did you install all requirements and ensure models are accessible?")
        # Return a random 1024-dimensional embedding (typical size for DINOv2)
        return np.random.randn(1024).astype(np.float32)
    
    def calculate_gradcam(img):
        """Dummy function that returns a fixed GradCAM map of expected shape."""
        logger.warning("Using dummy calculate_gradcam function. Did you install all requirements and ensure models are accessible?")
        # Return a 7x7 heatmap (typical output size from ResNet50)
        return np.random.rand(7, 7).astype(np.float32)
    
    def generate_heatmap(img):
        """Dummy function that returns a fixed attention heatmap of expected shape."""
        logger.warning("Using dummy generate_heatmap function. Did you install all requirements and ensure models are accessible?")
        # Return a 14x14 heatmap (typical for ViT-based attention maps)
        return np.random.rand(14, 14).astype(np.float32)
# --- End ML Model Imports ---

sys.path.append(current_dir)
# --- Improved Analysis Imports ---
try:
    from improved_analysis import (
        enhanced_style_analysis, 
        enhanced_composition_analysis, 
        enhanced_color_analysis,
        ENHANCED_STYLE_ANALYSIS_AVAILABLE,
        ENHANCED_COMPOSITION_ANALYSIS_AVAILABLE,
        ENHANCED_COLOR_ANALYSIS_AVAILABLE
    )
    logger.info("Successfully imported enhanced analysis functions from improved_analysis.py")
except ImportError as e:
    logger.warning(f"Error importing enhanced analysis functions: {e}. Using fallback implementations.")
    logger.warning(f"Current sys.path: {sys.path}")
    logger.warning(f"Expected file path: {os.path.join(os.path.dirname(__file__), 'improved_analysis.py')}")
    logger.warning(f"File exists: {os.path.exists(os.path.join(os.path.dirname(__file__), 'improved_analysis.py'))}")

    # Keep the fallback definitions if import fails
    # (Fallback definitions remain the same as before)
    # ... existing fallback function definitions ...
    import numpy as np
    from PIL import Image, ImageStat
    from skimage import feature, color, exposure, util, filters
    from skimage.feature import graycomatrix, graycoprops
    from skimage.measure import shannon_entropy
    
    # Define improved functions of analysis directly here, if import failed
    def enhanced_style_analysis(img, embedding=None):
        """Analysis of image style based on various visual characteristics"""
        logger.warning("Running LOCAL fallback style analysis implementation") # Use logger
        if isinstance(img, Image.Image):
            img_np = np.array(img)
        else:
            img_np = img
        img_gray = color.rgb2gray(img_np)
        # 1. Measuring linearity (through edge detection)
        edges = feature.canny(img_gray)
        linearity = np.mean(edges) * 2  # Normalization to range 0-1
        linearity = min(max(linearity, 0), 1)  # Limit values
        # 2. Measuring colorfulness (through HSV statistics)
        hsv = color.rgb2hsv(img_np)
        saturation = hsv[:,:,1]
        colorfulness = np.mean(saturation) * 1.5  # Normalization
        colorfulness = min(max(colorfulness, 0), 1)
        # 3. Complexity (through image entropy)
        complexity = shannon_entropy(img_gray) / 8  # Normalization (8 is max entropy for 8-bit image)
        complexity = min(max(complexity, 0), 1)
        # 4. Contrast
        p2, p98 = np.percentile(img_gray, (2, 98))
        if p98 > p2:
            contrast = (p98 - p2) / 1.0  # Normalization
        else:
            contrast = 0
        contrast = min(max(contrast, 0), 1)
        # 5. Symmetry (simplified evaluation through mirror reflection)
        h, w = img_gray.shape
        left_half = img_gray[:, :w//2]
        right_half = np.fliplr(img_gray[:, w//2:])
        # If halves are different sizes, cut to smaller size
        min_w = min(left_half.shape[1], right_half.shape[1])
        symmetry_diff = np.mean(np.abs(left_half[:, :min_w] - right_half[:, :min_w]))
        symmetry = 1 - symmetry_diff 
        symmetry = min(max(symmetry, 0), 1)
        # 6. Texture (through GLCM)
        try:
            glcm = graycomatrix(
                (img_gray * 255).astype(np.uint8), 
                distances=[1], 
                angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], 
                levels=8,
                symmetric=True, normed=True
            )
            texture_contrast = graycoprops(glcm, 'contrast')[0].mean()
            texture_dissimilarity = graycoprops(glcm, 'dissimilarity')[0].mean()
            texture_measure = (texture_contrast + texture_dissimilarity) / 30  # Нормализация
            texture = min(max(texture_measure, 0), 1)
        except:
            texture = 0.5  # Fallback если GLCM не работает
        logger.info(f"(Fallback) Style analysis results: linearity={linearity:.2f}, colorfulness={colorfulness:.2f}, complexity={complexity:.2f}, contrast={contrast:.2f}, symmetry={symmetry:.2f}, texture={texture:.2f}")
        return {
             'linearity': linearity,
             'colorfulness': colorfulness,
             'complexity': complexity,
             'contrast': contrast,
             'symmetry': symmetry,
             'texture': texture,
             'is_fallback': True # Explicitly mark as fallback
        }
    
    def enhanced_composition_analysis(img):
        """Analysis of image composition"""
        logger.warning("Running LOCAL fallback composition analysis implementation") # Use logger
        # Convert to numpy array if PIL Image is passed
        if isinstance(img, Image.Image):
            img_np = np.array(img)
        else:
            img_np = img
        img_gray = color.rgb2gray(img_np)
        # 1. Symmetry (similar to style_analysis)
        h, w = img_gray.shape
        left_half = img_gray[:, :w//2]
        right_half = np.fliplr(img_gray[:, w//2:])
        min_w = min(left_half.shape[1], right_half.shape[1])
        symmetry_diff = np.mean(np.abs(left_half[:, :min_w] - right_half[:, :min_w]))
        symmetry = 1 - symmetry_diff
        symmetry = min(max(symmetry, 0), 1)
        # 2. Rule of thirds (simplified implementation)
        h_third = h // 3
        w_third = w // 3
        points = [ (h_third, w_third), (h_third, 2*w_third), (2*h_third, w_third), (2*h_third, 2*w_third) ]
        thirds_score = 0
        for y, x in points:
            region_h = max(1, int(h * 0.1))
            region_w = max(1, int(w * 0.1))
            y_start, y_end = max(0, y - region_h//2), min(h, y + region_h//2)
            x_start, x_end = max(0, x - region_w//2), min(w, x + region_w//2)
            region = img_gray[y_start:y_end, x_start:x_end]
            if region.size > 0:
                p2, p98 = np.percentile(region, (2, 98))
                region_contrast = min(1.0, max(0, p98 - p2))
                thirds_score += region_contrast
        rule_of_thirds = min(1.0, thirds_score / 4)
        # 3. Leading lines (through Hough line detection)
        try:
            edges = feature.canny(img_gray)
            lines = feature.hough_line(edges, theta=np.linspace(-np.pi/2, np.pi/2, 180, endpoint=False))
            # Check if lines[0] is not empty before accessing it
            leading_lines = min(1.0, len(lines[0]) / 50) if lines and lines[0] is not None and len(lines[0]) > 0 else 0.0
        except:
            leading_lines = 0.5  # Fallback
        # 4. Depth (simplified through brightness gradient)
        dy, dx = np.gradient(img_gray)
        gradient_magnitude = np.sqrt(dy**2 + dx**2)
        depth = min(1.0, np.mean(gradient_magnitude) * 20)  # Normalization
        # 5. Обрамление (упрощенно через анализ краев изображения)
        border_width = int(min(h, w) * 0.15)
        if border_width * 2 >= h or border_width * 2 >= w:  # Prevent negative slice
            center = img_gray  # Use full image if border is too large
        else:
            center = img_gray[border_width:-border_width, border_width:-border_width]
        borders = img_gray.copy()
        if border_width * 2 < h and border_width * 2 < w:  # Apply mask only if valid
            borders[border_width:-border_width, border_width:-border_width] = 0
        center_brightness = np.mean(center) if center.size > 0 else 0
        border_brightness = np.mean(borders[borders > 0]) if np.any(borders > 0) else 0
        framing = min(1.0, max(0, abs(center_brightness - border_brightness) * 2))
        # 6. Баланс (через анализ распределения яркости)
        left_brightness = np.mean(img_gray[:, :w//2]) if w > 0 else 0
        right_brightness = np.mean(img_gray[:, w//2:]) if w > 0 else 0
        top_brightness = np.mean(img_gray[:h//2, :]) if h > 0 else 0
        bottom_brightness = np.mean(img_gray[h//2:, :]) if h > 0 else 0
        h_balance = 1 - min(1.0, abs(left_brightness - right_brightness) * 2)
        v_balance = 1 - min(1.0, abs(top_brightness - bottom_brightness) * 2)
        balance = (h_balance + v_balance) / 2
        logger.info(f"(Fallback) Composition analysis results: symmetry={symmetry:.2f}, rule_of_thirds={rule_of_thirds:.2f}, leading_lines={leading_lines:.2f}, depth={depth:.2f}, framing={framing:.2f}, balance={balance:.2f}")
        # Результаты в виде словаря с названиями параметров и значениями
        return {
            "symmetry": float(symmetry),
            "rule_of_thirds": float(rule_of_thirds),
            "leading_lines": float(leading_lines),
            "depth": float(depth),
            "framing": float(framing),
            "balance": float(balance),
            "is_fallback": True # Explicitly mark as fallback
        }
    
    def enhanced_color_analysis(img):
        logger.warning("Running LOCAL fallback color analysis implementation") # Use logger
        return {
            "dominant_colors": [
                {"rgb": "rgb(220,220,220)", "percentage": 30, "brightness": 0.85, "saturation": 0.05, "emotion": "Нейтральный, сбалансированный", "name": "Светло-серый"},
                {"rgb": "rgb(180,150,120)", "percentage": 20, "brightness": 0.6, "saturation": 0.3, "emotion": "Тёплый, натуральный", "name": "Бежево-коричневый"},
                {"rgb": "rgb(140,140,140)", "percentage": 15, "brightness": 0.55, "saturation": 0.0, "emotion": "Нейтральный, сбалансированный", "name": "Серый"},
                {"rgb": "rgb(240,240,240)", "percentage": 15, "brightness": 0.95, "saturation": 0.0, "emotion": "Чистый, простой", "name": "Белый"},
                {"rgb": "rgb(100,80,60)", "percentage": 10, "brightness": 0.35, "saturation": 0.4, "emotion": "Надёжный, земной", "name": "Коричневый"},
                {"rgb": "rgb(60,60,60)", "percentage": 10, "brightness": 0.25, "saturation": 0.0, "emotion": "Элегантный, серьезный", "name": "Тёмно-серый"}
            ],
            "labels": ["Светло-серый", "Бежево-коричневый", "Серый", "Белый", "Коричневый", "Тёмно-серый"],
            "datasets": [{
                "data": [30, 20, 15, 15, 10, 10],
                "backgroundColor": ["rgba(220,220,220,0.7)", "rgba(180,150,120,0.7)", "rgba(140,140,140,0.7)", 
                                   "rgba(240,240,240,0.7)", "rgba(100,80,60,0.7)", "rgba(60,60,60,0.7)"],
                "borderColor": ["rgb(220,220,220)", "rgb(180,150,120)", "rgb(140,140,140)", 
                               "rgb(240,240,240)", "rgb(100,80,60)", "rgb(60,60,60)"],
                "borderWidth": 1
            }],
            "harmony": {
                "score": 0.7,
                "description": "Гармоничная (приятная, сбалансированная)",
                "type": "Нейтральная"
            },
            "is_fallback": True, # Explicitly mark as fallback
            "is_monochrome": False # Assume not monochrome in fallback
        }

    ENHANCED_STYLE_ANALYSIS_AVAILABLE = True # Assume available unless import failed
    ENHANCED_COMPOSITION_ANALYSIS_AVAILABLE = True
    ENHANCED_COLOR_ANALYSIS_AVAILABLE = True
# --- End Improved Analysis Imports ---

# Define cache dictionary for reuse of data (remains the same)
cache = {}
if 'metadata_cache' not in cache: cache['metadata_cache'] = {}
if 'invalid_ids_cache' not in cache: cache['invalid_ids_cache'] = set()
if 'style_analysis_cache' not in cache: cache['style_analysis_cache'] = {}
if 'color_analysis_cache' not in cache: cache['color_analysis_cache'] = {}
if 'composition_analysis_cache' not in cache: cache['composition_analysis_cache'] = {}
if 'invalid_object_ids' not in cache: cache['invalid_object_ids'] = set() # Keep consistent name

# Debug flag (remains the same)
DEBUG_CLEAR_CACHE = True
if DEBUG_CLEAR_CACHE:
    logger.info("Clearing analysis caches on startup due to DEBUG_CLEAR_CACHE=True")
    cache['style_analysis_cache'] = {}
    cache['color_analysis_cache'] = {}
    cache['composition_analysis_cache'] = {}

# Define directories (paths remain the same)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
PROCESSED_EMBEDDINGS_DIR = os.path.join(DATA_DIR, 'embeddings')
PROCESSED_GRADCAM_DIR = os.path.join(DATA_DIR, 'gradcam')
PROCESSED_ATTENTION_DIR = os.path.join(DATA_DIR, 'attention')
IMAGES_DIR = os.path.join(DATA_DIR, "images") # Add general images dir
UPLOADED_IMAGES_DIR = os.path.join(IMAGES_DIR, "uploaded") # Used by analyze_uploaded_image
MET_IMAGES_DIR = os.path.join(IMAGES_DIR, "met") # Used by preprocess_artworks
RIJKS_IMAGES_DIR = os.path.join(IMAGES_DIR, "rijks") # Used by preprocess_artworks
AIC_IMAGES_DIR = os.path.join(IMAGES_DIR, "aic") # Used by preprocess_artworks

# Create directories if they don't exist (remains the same)
os.makedirs(PROCESSED_EMBEDDINGS_DIR, exist_ok=True)
os.makedirs(PROCESSED_GRADCAM_DIR, exist_ok=True)
os.makedirs(PROCESSED_ATTENTION_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(UPLOADED_IMAGES_DIR, exist_ok=True)
os.makedirs(MET_IMAGES_DIR, exist_ok=True)
os.makedirs(RIJKS_IMAGES_DIR, exist_ok=True)
os.makedirs(AIC_IMAGES_DIR, exist_ok=True)

# Define paths for storing metadata (add processed_urls.json)
UPLOADED_ARTWORKS_FILE = os.path.join(DATA_DIR, 'uploaded_artworks.json')
PROCESSED_URLS_FILE = os.path.join(DATA_DIR, 'processed_urls.json') # New file for processed URLs

# --- Helper functions for JSON data loading/saving ---
def load_json_data(filepath, default_value):
    """Loads data from a JSON file, returning default value on error or if file doesn't exist."""
    if not os.path.exists(filepath):
        logger.warning(f"JSON file not found: {filepath}. Returning default.")
        return default_value
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            logger.info(f"Successfully loaded data from {filepath}")
            return data if isinstance(data, type(default_value)) else default_value
    except (json.JSONDecodeError, IOError, TypeError) as e:
        logger.error(f"Error loading or parsing JSON file ({filepath}): {e}. Returning default.")
        return default_value

def save_json_data(filepath, data):
    """Saves data to a JSON file."""
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2) # Use indent for readability
        logger.info(f"Saved data to {filepath}")
        return True
    except IOError as e:
        logger.error(f"Error saving JSON file ({filepath}): {e}")
        return False

# Use helper functions for uploaded artworks and processed URLs
def load_uploaded_artworks():
    return load_json_data(UPLOADED_ARTWORKS_FILE, [])

def save_uploaded_artworks(data):
    return save_json_data(UPLOADED_ARTWORKS_FILE, data)

def load_processed_urls():
    # Stored as a dictionary: { unique_id: { original_url: "...", timestamp: ... }, ... }
    return load_json_data(PROCESSED_URLS_FILE, {})

def save_processed_urls(data):
    return save_json_data(PROCESSED_URLS_FILE, data)
# --- End JSON Helpers ---


# Helper function to get object data from Met API (Improved error handling)
def get_met_object(object_id):
    """Fetches object data from the Met API and caches it, checking for image."""
    # Check caches first
    if object_id in cache.get('invalid_ids_cache', set()):
        logger.debug(f"Met API: Skipping known invalid ID {object_id}")
        return None
    if object_id in cache.get('metadata_cache', {}):
        logger.debug(f"Met API: Using cached metadata for ID {object_id}")
        return cache['metadata_cache'][object_id]

    met_url = f"https://collectionapi.metmuseum.org/public/collection/v1/objects/{object_id}"
    try:
        logger.debug(f"Met API: Fetching data for object {object_id} from {met_url}")
        # Increase timeout
        response = requests.get(met_url, timeout=20)
        response.raise_for_status() # Raise HTTPError for bad responses
        
        # Check for empty response before parsing JSON
        if not response.text or response.text.isspace():
            logger.warning(f"Met API: Empty response received for object {object_id}")
            cache.setdefault('invalid_ids_cache', set()).add(object_id)
            return None
            
        try:
            data = response.json()
        except json.JSONDecodeError as json_error:
            logger.error(f"Met API: Error parsing JSON for object {object_id}: {json_error}")
            logger.error(f"Met API: Response text (first 100 chars): {response.text[:100]}")
            cache.setdefault('invalid_ids_cache', set()).add(object_id)
            return None

        # Check for the presence and non-emptiness of primaryImageSmall
        if data and data.get('primaryImageSmall'):
            logger.info(f"Met API: Successfully fetched data for object {object_id}")
            cache.setdefault('metadata_cache', {})[object_id] = data # Cache valid data
            return data
        else:
            logger.warning(f"Met API: No primaryImageSmall found for object {object_id}")
            cache.setdefault('invalid_ids_cache', set()).add(object_id) # Cache as invalid (no image)
            return None

    except requests.exceptions.Timeout:
        logger.error(f"Met API: Timeout fetching object {object_id}")
        cache.setdefault('invalid_ids_cache', set()).add(object_id)
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Met API: Error fetching object {object_id}: {e}")
        cache.setdefault('invalid_ids_cache', set()).add(object_id)
        return None
    except Exception as e:
        logger.error(f"Met API: Unexpected error processing object {object_id}: {e}", exc_info=True)
        cache.setdefault('invalid_ids_cache', set()).add(object_id)
        return None

# Helper function to convert numpy map to image bytes (Improved logging)
def map_to_image_bytes(map_data, colormap=cv2.COLORMAP_JET, target_size=None):
    if map_data is None or map_data.size == 0:
        logger.warning("map_to_image_bytes: Received empty map_data.")
        return None
    try:
        # Normalize map to 0-255
        norm_map = map_data - np.min(map_data)
        max_val = np.max(norm_map)
        if max_val > 1e-6:  # Avoid division by zero or near-zero
            norm_map = norm_map / max_val * 255
        else:
            norm_map = np.zeros_like(map_data)  # Handle all-zero map
            
        heatmap_colored = cv2.applyColorMap(np.uint8(norm_map), colormap)
        
        # Resize if target_size is provided
        if target_size and isinstance(target_size, tuple) and len(target_size) == 2:
            logger.debug(f"map_to_image_bytes: Resizing heatmap to target size: {target_size}")
            # Use cv2.INTER_LINEAR for smoother interpolation
            heatmap_colored = cv2.resize(heatmap_colored, target_size, interpolation=cv2.INTER_LINEAR)
        else:
            logger.debug(f"map_to_image_bytes: Target size not provided or invalid, not resizing. Map size: {heatmap_colored.shape[:2]}")
            
        # Convert BGR (from cv2) to RGB
        heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        img_pil = Image.fromarray(heatmap_rgb)
        img_byte_arr = BytesIO()
        img_pil.save(img_byte_arr, format='PNG')  # Use PNG to preserve details
        img_byte_arr.seek(0)
        logger.debug("map_to_image_bytes: Successfully converted map to image bytes.")
        logger.info(f"Converted map data (shape: {map_data.shape}) to image bytes (size: {len(img_byte_arr.getvalue())})")
        return img_byte_arr
    except Exception as e:
        logger.error(f"map_to_image_bytes: Error converting map to image: {e}", exc_info=True)
        return None


@app.route('/api/objects')
def get_objects():
    cache_key = 'met_object_ids'
    if cache_key in cache and time.time() - cache.get(f"{cache_key}_timestamp", 0) < 3600: # Cache for 1 hour
        logger.info("API: Returning cached object list from /api/objects")
        return jsonify(cache[cache_key])

    try:
        logger.info("API: Request received for /api/objects - fetching fresh data")
        response = requests.get('https://collectionapi.metmuseum.org/public/collection/v1/objects', timeout=60) # Increased timeout to 60 seconds
        response.raise_for_status() # Raise exception for bad status codes
        data = response.json()
        cache[cache_key] = data # Store data in cache
        cache[f"{cache_key}_timestamp"] = time.time() # Store timestamp
        logger.info(f"API: Successfully fetched and cached {len(data.get('objectIDs', []))} object IDs from Met API.")
        return jsonify(data)
    except requests.exceptions.Timeout:
        logger.error(f"API: Timeout error fetching object list from Met API after 60 seconds.")
        return jsonify({"error": "Timeout fetching object list from Met API"}), 504 # Gateway Timeout
    except requests.exceptions.RequestException as e:
        logger.error(f"API: Error fetching object list from Met API: {e}")
        return jsonify({"error": f"Failed to fetch object list from Met API: {e}"}), 502 # Bad Gateway


@app.route('/api/process/<int:object_id>')
def process_object(object_id):
    logger.info(f"API: Request received for /api/process/{object_id}")
    # Use the helper function to get data (includes cache and image check)
    object_data = get_met_object(object_id)

    # If object_data is None, it means either fetch failed or no image was found
    if not object_data:
        logger.warning(f"API: No image or data available for object {object_id}")
        return jsonify({"error": f"No data or image available for object ID {object_id}"}), 404

    # We know primaryImageSmall exists if object_data is not None
    img_url = object_data.get('primaryImageSmall')
    logger.info(f"API: Processing object {object_id}, image URL: {img_url}")
            
    try:
        logger.debug(f"API: Fetching image for {object_id} from {img_url}")
        img_response = requests.get(img_url, timeout=20)
        img_response.raise_for_status()

        # Use PIL directly from bytes without saving to disk
        logger.debug(f"API: Opening image for {object_id} with PIL")
        img_pil = Image.open(BytesIO(img_response.content))
        
        # Ensure image is RGB for processing functions
        if img_pil.mode != 'RGB':
            logger.debug(f"API: Converting image {object_id} from {img_pil.mode} to RGB")
            img_pil = img_pil.convert('RGB')
        
        # --- Generate DINOv2 Embedding --- 
        logger.info(f"API: Generating embedding for {object_id}...")
        embedding = generate_embedding(img_pil) # Use the potentially dummy function
        if embedding is None:
            logger.warning(f"API: Failed to generate embedding for {object_id}. Skipping embedding save.")
        else:
            # Save embedding
            embedding_path = os.path.join(PROCESSED_EMBEDDINGS_DIR, f'{object_id}.npy')
            np.save(embedding_path, embedding)
            logger.info(f"API: Saved embedding to {embedding_path}")

        # --- Calculate GradCAM ---
        logger.info(f"API: Calculating GradCAM for {object_id}...")
        gradcam_map = calculate_gradcam(img_pil) # Use the potentially dummy function
        if gradcam_map is not None:
            gradcam_path = os.path.join(PROCESSED_GRADCAM_DIR, f'{object_id}.npy')
            np.save(gradcam_path, gradcam_map)
            logger.info(f"API: Saved GradCAM map to {gradcam_path}")
        else:
            logger.warning(f"API: Failed to generate GradCAM for {object_id}")

        # --- Generate Attention Map ---
        logger.info(f"API: Generating Attention Map for {object_id}...")
        heatmap = generate_heatmap(img_pil) # Use the potentially dummy function
        if heatmap is not None:
            attention_path = os.path.join(PROCESSED_ATTENTION_DIR, f'{object_id}.npy')
            np.save(attention_path, heatmap)
            logger.info(f"API: Saved Attention map to {attention_path}")
        else:
            logger.warning(f"API: Failed to generate Attention map for {object_id}")
        
        # No need to save the actual image - we have the URL in metadata
        logger.info(f"API: Finished processing data for object {object_id}")
        
        # Return metadata and embedding (as list)
        return jsonify({
            "objectID": object_id,
            # Convert numpy array to list for JSON response if embedding exists
            "embedding": embedding.tolist() if embedding is not None else None, 
            "metadata": {
                "objectID": object_id,
                "id": object_id,
                "primaryImage": object_data.get('primaryImage'),
                "primaryImageSmall": img_url, # Use the fetched URL
                "title": object_data.get('title'),
                "artistDisplayName": object_data.get('artistDisplayName'),
                "objectDate": object_data.get('objectDate'),
                "department": object_data.get('department'),
                "classification": object_data.get('classification'),
                "medium": object_data.get('medium'),
                "objectURL": object_data.get('objectURL'),
                # Explicitly state that we're not saving the image locally
                "image_stored_locally": False
            }
        })
    except requests.exceptions.RequestException as e:
        logger.error(f"API: Error fetching image for object {object_id}: {e}")
        return jsonify({"error": f"Error fetching image from Met API: {e}"}), 502 # Bad Gateway
    except UnidentifiedImageError:
        logger.error(f"API: Failed to identify image format for object {object_id} from URL {img_url}")
        return jsonify({"error": f"Invalid or unsupported image format for object {object_id}"}), 400 # Bad Request
    except Exception as e:
        logger.error(f"API: Error processing object {object_id}: {e}", exc_info=True)
        return jsonify({"error": f"Internal server error during processing"}), 500


# --- New Endpoint: /api/process_by_url ---
@app.route('/api/process_by_url', methods=['POST'])
@cross_origin()
def process_by_url():
    """
    Processes an image from a given URL. Downloads the image, generates analysis data (embedding, gradcam, attention),
    saves the analysis data, stores the original URL with a generated ID.
    The image itself is only kept in memory and not saved to disk.
    """
    logger.info("API: Request received for /api/process_by_url")
    if not request.is_json:
        logger.warning("API: Request is not JSON for /api/process_by_url")
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    image_url = data.get('image_url')

    if not image_url:
        logger.warning("API: 'image_url' missing in request body for /api/process_by_url")
        return jsonify({"error": "'image_url' is required in the request body"}), 400

    # Validate URL
    try:
        parsed_url = urlparse(image_url)
        if not all([parsed_url.scheme, parsed_url.netloc]):
            raise ValueError("Invalid URL structure")
    except ValueError:
        logger.warning(f"API: Invalid image URL provided: {image_url}")
        return jsonify({"error": "Invalid image URL provided"}), 400

    logger.info(f"API: Processing image from URL: {image_url}")

    # Generate a unique ID based on the URL (e.g., hash or UUID)
    # Using UUID for simplicity and uniqueness guarantees
    unique_id = f"url_{uuid.uuid4()}"
    logger.info(f"API: Generated unique ID for URL: {unique_id}")

    img_pil = None

    try:
        # Download image directly into memory without saving to disk
        logger.debug(f"API: Downloading image from URL: {image_url}")
        img_response = requests.get(image_url, timeout=20)
        img_response.raise_for_status()

        # Check content type - basic check
        content_type = img_response.headers.get('Content-Type', '').lower()
        if not content_type.startswith('image/'):
            logger.warning(f"API: URL content type is not an image ({content_type}): {image_url}")
            return jsonify({"error": f"URL does not point to a valid image (Content-Type: {content_type})"}), 400

        image_bytes = img_response.content
        logger.debug(f"API: Image downloaded ({len(image_bytes)} bytes)")

        # Process image using PIL in memory
        logger.debug(f"API: Opening downloaded image with PIL")
        img_pil = Image.open(BytesIO(image_bytes))
        if img_pil.mode != 'RGB':
            logger.debug(f"API: Converting downloaded image from {img_pil.mode} to RGB")
            img_pil = img_pil.convert('RGB')

        # --- Perform Analysis (Embedding, GradCAM, Attention) ---
        saved_paths = {}
        
        # Generate embedding
        logger.info(f"API: Generating embedding for {unique_id}...")
        embedding = generate_embedding(img_pil)
        if embedding is not None:
            embedding_path = os.path.join(PROCESSED_EMBEDDINGS_DIR, f'{unique_id}.npy')
            np.save(embedding_path, embedding)
            saved_paths['embedding'] = embedding_path
            logger.info(f"API: Saved embedding to {embedding_path}")
        else:
            logger.warning(f"API: Embedding generation failed for {unique_id}, not saving.")

        # Generate GradCAM
        logger.info(f"API: Calculating GradCAM for {unique_id}...")
        gradcam_map = calculate_gradcam(img_pil)
        if gradcam_map is not None:
            gradcam_path = os.path.join(PROCESSED_GRADCAM_DIR, f'{unique_id}.npy')
            np.save(gradcam_path, gradcam_map)
            saved_paths['gradcam'] = gradcam_path
            logger.info(f"API: Saved GradCAM map to {gradcam_path}")
        else:
            logger.warning(f"API: GradCAM generation failed for {unique_id}, not saving.")

        # Generate Attention Map
        logger.info(f"API: Generating Attention Map for {unique_id}...")
        heatmap = generate_heatmap(img_pil)
        if heatmap is not None:
            attention_path = os.path.join(PROCESSED_ATTENTION_DIR, f'{unique_id}.npy')
            np.save(attention_path, heatmap)
            saved_paths['attention'] = attention_path
            logger.info(f"API: Saved Attention map to {attention_path}")
        else:
            logger.warning(f"API: Attention map generation failed for {unique_id}, not saving.")

        # --- Store URL and ID Mapping ---
        processed_urls = load_processed_urls()
        processed_urls[unique_id] = {
            "original_url": image_url,
            "timestamp": time.time()
        }
        if not save_processed_urls(processed_urls):
            # Log error but continue, as analysis results are saved
            logger.error(f"API: Failed to save processed URL mapping for {unique_id}")

        logger.info(f"API: Successfully processed image from URL {image_url} with ID {unique_id}")

        # --- Return Response ---
        return jsonify({
            "message": "Image processed successfully",
            "processed_id": unique_id,
            "original_url": image_url,
            "embedding": embedding.tolist() if embedding is not None else None,
            "analysis_files": saved_paths, # Return paths to saved .npy files
            "image_stored_locally": False # Explicitly state we don't store the image
        }), 200

    except requests.exceptions.RequestException as e:
        logger.error(f"API: Error downloading image from URL {image_url}: {e}")
        return jsonify({"error": f"Failed to download image from URL: {e}"}), 502 # Bad Gateway
    except UnidentifiedImageError as e:
        logger.error(f"API: Failed to identify image format from URL {image_url}: {e}")
        return jsonify({"error": "Invalid or unsupported image format at URL"}), 400 # Bad Request
    except Exception as e:
        logger.error(f"API: Error processing image from URL {image_url}: {e}", exc_info=True)
        return jsonify({"error": "Internal server error during image processing"}), 500


# Endpoint to serve GradCAM map (Always recompute)
@app.route('/api/gradcam_image/<string:object_id>') # Allow string IDs for URL-processed items
@cross_origin()
def get_gradcam_image(object_id):
    logger.info(f"API GradCAM: Request to COMPUTE GradCAM image for ID: {object_id}")
    
    try:
        # 1. Fetch the original image using the helper function
        logger.debug(f"API GradCAM: Fetching original image for {object_id}")
        img_pil, _, error_msg, status_code = get_image_and_embedding(object_id, require_embedding=False)
        if error_msg:
             logger.error(f"API GradCAM: get_image_and_embedding failed for ID {object_id}: {error_msg}")
             abort(status_code, description=error_msg)
             
        if img_pil is None:
            logger.error(f"API GradCAM: Failed to get original image for ID {object_id} (img_pil is None).")
            abort(404, description=f"Original image could not be loaded for ID {object_id}")

        # 2. COMPUTE GradCAM map
        logger.info(f"API GradCAM: Computing GradCAM map for ID: {object_id}...")
        start_time = time.time()
        gradcam_map_data = None
        try:
            gradcam_map_data = calculate_gradcam(img_pil) 
            if gradcam_map_data is None or not isinstance(gradcam_map_data, np.ndarray):
                 raise ValueError("calculate_gradcam did not return a valid numpy array.")
            logger.info(f"API GradCAM: Map computation successful. Shape: {gradcam_map_data.shape}. Time: {time.time() - start_time:.2f}s")
        except Exception as compute_error:
            logger.error(f"API GradCAM: Error during map computation for ID {object_id}: {compute_error}", exc_info=True)
            abort(500, description=f"Failed to compute GradCAM map for ID {object_id}: {compute_error}")

        # 3. Convert map data to image bytes
        if gradcam_map_data is not None:
            logger.info(f"API GradCAM: Converting computed map data to PNG for ID: {object_id}")
            image_bytes = map_to_image_bytes(gradcam_map_data, colormap=cv2.COLORMAP_JET) 
            
            if image_bytes is None:
                logger.error(f"API GradCAM: map_to_image_bytes returned None for ID: {object_id}")
                abort(500, description="Failed to convert computed GradCAM map to image.")
                
            logger.info(f"API GradCAM: Serving computed PNG directly for ID: {object_id}")
            # Serve directly from memory
            return Response(image_bytes.getvalue(), mimetype='image/png') # Use .getvalue() on BytesIO
        else:
            # This should ideally not happen if compute error handling is correct
            logger.error(f"API GradCAM: gradcam_map_data was None after compute for ID: {object_id}")
            abort(500, description="Failed to compute GradCAM map data.")

    except Exception as e:
        # Catch errors from get_image_and_embedding or other unexpected issues
        logger.error(f"API GradCAM: Unexpected error serving map for ID {object_id}: {e}", exc_info=True)
        abort(500, description=f"An unexpected error occurred: {e}")


# Endpoint to serve Attention map (Always recompute)
@app.route('/api/attention_image/<string:object_id>') # Allow string IDs
@cross_origin()
def get_attention_image(object_id):
    logger.info(f"API Attention: Request to COMPUTE Attention image for ID: {object_id}")
    
    try:
        # 1. Fetch the original image using the helper function
        logger.debug(f"API Attention: Fetching original image for {object_id}")
        img_pil, _, error_msg, status_code = get_image_and_embedding(object_id, require_embedding=False)
        
        if error_msg:
             logger.error(f"API Attention: get_image_and_embedding failed for ID {object_id}: {error_msg}")
             abort(status_code, description=error_msg)
             
        if img_pil is None:
            logger.error(f"API Attention: Failed to get original image for ID {object_id} (img_pil is None).")
            abort(404, description=f"Original image could not be loaded for ID {object_id}")

        # 2. COMPUTE Attention map
        logger.info(f"API Attention: Computing Attention map for ID: {object_id}...")
        start_time = time.time()
        attention_map_data = None
        try:
            attention_map_data = generate_heatmap(img_pil) 
            if attention_map_data is None or not isinstance(attention_map_data, np.ndarray):
                 raise ValueError("generate_heatmap did not return a valid numpy array.")
            logger.info(f"API Attention: Map computation successful. Shape: {attention_map_data.shape}. Time: {time.time() - start_time:.2f}s")
        except Exception as compute_error:
            logger.error(f"API Attention: Error during map computation for ID {object_id}: {compute_error}", exc_info=True)
            abort(500, description=f"Failed to compute Attention map for ID {object_id}: {compute_error}")

        # 3. Convert map data to image bytes
        if attention_map_data is not None:
            logger.info(f"API Attention: Converting computed map data to PNG for ID: {object_id}")
            image_bytes = map_to_image_bytes(attention_map_data)
            
            if image_bytes is None:
                logger.error(f"API Attention: map_to_image_bytes returned None for ID: {object_id}")
                abort(500, description="Failed to convert computed Attention map to image.")
                
            logger.info(f"API Attention: Serving computed PNG directly for ID: {object_id}")
            # Serve directly from memory
            return Response(image_bytes.getvalue(), mimetype='image/png') # Use .getvalue() on BytesIO
        else:
            # This should ideally not happen if compute error handling is correct
            logger.error(f"API Attention: attention_map_data was None after compute for ID: {object_id}")
            abort(500, description="Failed to compute Attention map data.")

    except Exception as e:
        # Catch errors from get_image_and_embedding or other unexpected issues
        logger.error(f"API Attention: Unexpected error serving map for ID {object_id}: {e}", exc_info=True)
        abort(500, description=f"An unexpected error occurred: {e}")


DEFAULT_LIMIT = 50 # Number of items per page by default

@app.route('/api/embeddings')
def get_all_embeddings():
    logger.info("API: Request received for /api/embeddings")
    logger.debug(f"API: DATA_DIR: {DATA_DIR}")
    logger.debug(f"API: Working directory: {os.getcwd()}")
    
    try:
        page = request.args.get('page', 1, type=int)
        limit = request.args.get('limit', DEFAULT_LIMIT, type=int)
    except ValueError:
        logger.warning("API: Invalid page or limit parameter in /api/embeddings request.")
        return jsonify({"error": "Invalid page or limit parameter."}), 400

    if page < 1: page = 1
    if limit < 1: limit = DEFAULT_LIMIT
    if limit > 100: limit = 100 # Limit to prevent overload

    # Path to the preprocessed metadata file
    preprocessed_metadata_path = os.path.join(DATA_DIR, "preprocessed_metadata.json")
    logger.debug(f"API: Looking for preprocessed metadata file: {preprocessed_metadata_path}")
    logger.debug(f"API: File exists: {os.path.exists(preprocessed_metadata_path)}")
    
    # Check if preprocessed data exists and process it
    if os.path.exists(preprocessed_metadata_path):
        try:
            logger.info(f"API: Found preprocessed metadata file. Attempting to use for faster response.")
            with open(preprocessed_metadata_path, 'r') as f:
                preprocessed_data = json.load(f)
                
            all_items = preprocessed_data.get('metadata', [])
            logger.info(f"API: Loaded {len(all_items)} objects from preprocessed_metadata.json")
            
            if not all_items:
                logger.warning("API: Preprocessed metadata file found, but 'metadata' list is empty.")
                # Return empty if file is empty
                return jsonify({
                    "page": page,
                    "limit": limit,
                    "total": 0,
                    "metadata": [],
                    "embeddings": []
                })

            # Preprocessed data is valid
                total_items = len(all_items)
                # Calculate pagination
                start_idx = (page - 1) * limit
                end_idx = start_idx + limit
                page_items = all_items[start_idx:end_idx]
                
                # Load embeddings for each item in the page
                aligned_metadata = []
                aligned_embeddings = []
                
                logger.info(f"API: Loading preprocessed data for page {page}, items {start_idx}-{end_idx} of {total_items}")
                
                for item in page_items:
                    item_id = item.get('id') or item.get('objectID')
                    embedding_path = item.get('embedding_path')
                    
                    if not embedding_path:
                        logger.warning(f"API: Missing embedding_path for item {item_id} in preprocessed data, skipping")
                        continue
                    
                    # Check existence and try alternative path only if needed
                    if not os.path.isabs(embedding_path): # If path is relative, make it absolute
                        embedding_path = os.path.join(BASE_DIR, embedding_path) # Assuming relative to api.py's dir

                    if not os.path.exists(embedding_path):
                        logger.warning(f"API: Embedding path not found: {embedding_path}")
                        filename = os.path.basename(embedding_path)
                        alt_path = os.path.join(PROCESSED_EMBEDDINGS_DIR, filename)
                        if os.path.exists(alt_path):
                            logger.info(f"API: Found alternative embedding path: {alt_path}")
                            embedding_path = alt_path
                        else:
                            logger.warning(f"API: Alternative embedding path also not found: {alt_path}")
                            continue
                    
                    try:
                        embedding = np.load(embedding_path)
                        aligned_metadata.append(item)
                        aligned_embeddings.append(embedding.tolist())
                    except Exception as e:
                        logger.error(f"API: Error loading embedding file {embedding_path} for {item_id}: {str(e)}")
                
                logger.info(f"API: Returning {len(aligned_metadata)} preprocessed items for page {page}")
                return jsonify({
                    "page": page,
                    "limit": limit,
                    "total": total_items,
                    "metadata": aligned_metadata,
                    "embeddings": aligned_embeddings
                })
                
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"API: Error loading or parsing preprocessed metadata file ({preprocessed_metadata_path}): {str(e)}", exc_info=True)
            # Fall through to return empty if error loading/parsing
        except Exception as e: # Catch other potential errors
             logger.error(f"API: Unexpected error processing preprocessed data: {str(e)}", exc_info=True)
             # Fall through to return empty

    # --- If preprocessed file doesn't exist or failed to load --- 
    logger.warning("API: Preprocessed metadata not found or failed to load. /api/embeddings returning empty. Use /api/museum_artworks for dynamic data.")
    return jsonify({
        "page": page,
        "limit": limit,
        "total": 0,
        "metadata": [],
        "embeddings": []
    })


# --- Analysis Endpoints (Improved logging and error handling) ---
@app.route('/api/analyze/style/<string:object_id>', methods=['GET']) # Allow string ID
def analyze_style(object_id):
    try:
        logger.info(f"API Style: Request analysis for ID: {object_id}")
        force_update = request.args.get('force', '').lower() == 'true'
        cache_key = f'style_analysis_{object_id}'

        # Check cache
        if not force_update and cache_key in cache:
            logger.info(f"API Style: Using cached result for {object_id}")
            return jsonify(cache[cache_key])
        
        # --- Get Image and Embedding ---
        img_pil, embedding, error_msg, status_code = get_image_and_embedding(object_id)
        if error_msg:
            return jsonify({"error": error_msg}), status_code
        # --- End Get Image ---
             
        # Perform style analysis
        style_result = None
        try:
            # Log which function is being called
            if 'improved_analysis' in sys.modules:
                 logger.info(f"API Style: Calling enhanced_style_analysis for {object_id}")
                 style_values_dict = enhanced_style_analysis(img_pil, embedding)
                 logger.info(f"API Style: enhanced_style_analysis returned: {style_values_dict}")
                 is_fallback = style_values_dict.get('is_fallback', False) # Check fallback status from function
            else:
                 logger.warning(f"API Style: Calling LOCAL fallback style analysis for {object_id}")
                 style_values_dict = enhanced_style_analysis(img_pil, embedding) # Calling the fallback directly
                 logger.info(f"API Style: LOCAL fallback analysis returned: {style_values_dict}")
                 is_fallback = True # Fallback was definitely used

            if is_fallback:
                 logger.warning(f"API Style: Fallback analysis used for {object_id}.")

            style_result = {
                'linearity': float(style_values_dict.get('linearity', 0.5)),
                'colorfulness': float(style_values_dict.get('colorfulness', 0.5)),
                'complexity': float(style_values_dict.get('complexity', 0.5)),
                'contrast': float(style_values_dict.get('contrast', 0.5)),
                'symmetry': float(style_values_dict.get('symmetry', 0.5)),
                'texture': float(style_values_dict.get('texture', 0.5)),
                'is_fallback': is_fallback,
                'labels': ['Linearity', 'Colorfulness', 'Complexity', 'Contrast', 'Symmetry', 'Texture'],
                'datasets': [{
                    'label': 'Style profile' + (' (Fallback)' if is_fallback else ''),
                    'data': [
                        float(style_values_dict.get('linearity', 0.5)),
                        float(style_values_dict.get('colorfulness', 0.5)),
                        float(style_values_dict.get('complexity', 0.5)),
                        float(style_values_dict.get('contrast', 0.5)),
                        float(style_values_dict.get('symmetry', 0.5)),
                        float(style_values_dict.get('texture', 0.5))
                    ],
                    'backgroundColor': 'rgba(54, 162, 235, 0.2)' if not is_fallback else 'rgba(255, 99, 132, 0.2)',
                    'borderColor': 'rgb(54, 162, 235)' if not is_fallback else 'rgb(255, 99, 132)',
                    'pointBackgroundColor': 'rgb(54, 162, 235)' if not is_fallback else 'rgb(255, 99, 132)',
                    'pointBorderColor': '#fff',
                    'pointHoverBackgroundColor': '#fff',
                    'pointHoverBorderColor': 'rgb(54, 162, 235)' if not is_fallback else 'rgb(255, 99, 132)'
                }]
            }
            logger.info(f"API Style: Analysis completed successfully for {object_id}")

        except Exception as analysis_error:
            logger.error(f"API Style: Error during analysis call for {object_id}: {analysis_error}", exc_info=True)
            # Fallback if analysis function itself fails
            style_result = {
                'linearity': 0.5, 'colorfulness': 0.5, 'complexity': 0.5, 'contrast': 0.5, 'symmetry': 0.5, 'texture': 0.5,
                'is_fallback': True, # Explicitly set fallback
                'labels': ['Linearity', 'Colorfulness', 'Complexity', 'Contrast', 'Symmetry', 'Texture'],
                'datasets': [{
                    'label': 'Stylistic profile (Analysis error)',
                    'data': [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                    'backgroundColor': 'rgba(255, 99, 132, 0.2)', 'borderColor': 'rgb(255, 99, 132)',
                    'pointBackgroundColor': 'rgb(255, 99, 132)', 'pointBorderColor': '#fff',
                    'pointHoverBackgroundColor': '#fff', 'pointHoverBorderColor': 'rgb(255, 99, 132)'
                }]
            }
        
        # Store in cache
        logger.info(f"API Style: Caching result for {object_id}")
        cache[cache_key] = style_result
        return jsonify(style_result)
    
    except Exception as e:
        # Catch errors from get_image_and_embedding or other unexpected issues
        logger.error(f"API Style: Unexpected error in endpoint for {object_id}: {e}", exc_info=True)
        return jsonify({"error": "Style analysis failed due to an internal server error"}), 500


@app.route('/api/analyze/color/<string:object_id>', methods=['GET']) # Allow string ID
def analyze_color(object_id):
    try:
        logger.info(f"API Color: Request analysis for ID: {object_id}")
        force_update = request.args.get('force', '').lower() == 'true'
        cache_key = f'color_analysis_{object_id}'

        # Check cache
        if not force_update and cache_key in cache:
            logger.info(f"API Color: Using cached result for {object_id}")
            return jsonify(cache[cache_key])

        # --- Get Image (don't need embedding for color) ---
        img_pil, _, error_msg, status_code = get_image_and_embedding(object_id, require_embedding=False)
        if error_msg:
            return jsonify({"error": error_msg}), status_code
        # --- End Get Image ---
        
        # Perform color analysis
        try:
            # Log which function is being called
            if 'improved_analysis' in sys.modules:
                 logger.info(f"API Color: Calling enhanced_color_analysis for {object_id}")
                 color_result = enhanced_color_analysis(img_pil)
                 logger.info(f"API Color: enhanced_color_analysis completed for {object_id}")
            else:
                 logger.warning(f"API Color: Calling LOCAL fallback color analysis for {object_id}")
                 color_result = enhanced_color_analysis(img_pil) # Calling the fallback directly
                 logger.info(f"API Color: LOCAL fallback analysis completed for {object_id}")
                 color_result['is_fallback'] = True # Ensure fallback is marked

        except Exception as analysis_error:
            logger.error(f"API Color: Error during color analysis call for {object_id}: {analysis_error}", exc_info=True)
            # Basic fallback if analysis fails
            color_result = {
                'labels': [], 'datasets': [], 'dominant_colors': [],
                'harmony': {'score': 0.5, 'description': "Analysis Error", 'type': 'Unknown'},
                'is_fallback': True, 'is_monochrome': False
            }

        # Store in cache
        logger.info(f"API Color: Caching result for {object_id}")
        cache[cache_key] = color_result
        return jsonify(color_result)
    
    except Exception as e:
        logger.error(f"API Color: Unexpected error in endpoint for {object_id}: {e}", exc_info=True)
        return jsonify({"error": "Color analysis failed due to an internal server error"}), 500


@app.route('/api/analyze/composition/<string:object_id>', methods=['GET']) # Allow string ID
def analyze_composition(object_id):
    try:
        logger.info(f"API Composition: Request analysis for ID: {object_id}")
        force_update = request.args.get('force', '').lower() == 'true'
        cache_key = f'composition_analysis_{object_id}'

        # Check cache
        if not force_update and cache_key in cache:
            logger.info(f"API Composition: Using cached result for {object_id}")
            return jsonify(cache[cache_key])
        
        # --- Get Image (don't need embedding) ---
        img_pil, _, error_msg, status_code = get_image_and_embedding(object_id, require_embedding=False)
        if error_msg:
            return jsonify({"error": error_msg}), status_code
        # --- End Get Image ---
        
        # Perform composition analysis
        try:
             # Log which function is being called
            if 'improved_analysis' in sys.modules:
                 logger.info(f"API Composition: Calling enhanced_composition_analysis for {object_id}")
                 comp_values = enhanced_composition_analysis(img_pil)
                 logger.info(f"API Composition: enhanced_composition_analysis completed for {object_id}")
            else:
                 logger.warning(f"API Composition: Calling LOCAL fallback composition analysis for {object_id}")
                 comp_values = enhanced_composition_analysis(img_pil) # Calling the fallback directly
                 logger.info(f"API Composition: LOCAL fallback analysis completed for {object_id}")
                 comp_values['is_fallback'] = True # Ensure fallback is marked


        except Exception as analysis_error:
            logger.error(f"API Composition: Error during composition analysis call for {object_id}: {analysis_error}", exc_info=True)
            # Fallback values
            comp_values = {
                'symmetry': 0.5, 'rule_of_thirds': 0.5, 'leading_lines': 0.5,
                'depth': 0.5, 'framing': 0.5, 'balance': 0.5,
                'is_fallback': True
            }
        
        # Store in cache
        logger.info(f"API Composition: Caching result for {object_id}")
        cache[cache_key] = comp_values
        return jsonify(comp_values)
    
    except Exception as e:
        logger.error(f"API Composition: Unexpected error in endpoint for {object_id}: {e}", exc_info=True)
        return jsonify({"error": "Composition analysis failed due to an internal server error"}), 500


# --- Helper function to get Image and Embedding for Analysis Endpoints ---
def get_image_and_embedding(object_id, require_embedding=True):
    """
    Fetches the PIL image and optionally the embedding for a given object ID (Met or URL-processed).
    Handles different ID types and potential errors.
    
    Notes:
    - For all object types, images are loaded directly from their URLs and not stored locally
    - Met objects and URL-processed objects use their original URLs
    - Uploaded objects might still be stored locally

    Returns: (image_pil, embedding, error_message, status_code)
             On success: (PIL.Image, np.array or None, None, 200)
             On failure: (None, None, str, int)
    """
    logger.debug(f"get_image_and_embedding called for ID: {object_id}, require_embedding={require_embedding}")
    img_pil = None
    embedding = None
    original_image_url = None

    # 1. Determine the source and get the image URL
    if object_id.isdigit():  # Assume Met ID
        logger.debug(f"ID {object_id} appears to be a Met ID.")
        object_data = get_met_object(int(object_id))
        if not object_data:
            msg = f"Object data not found for Met ID {object_id}"
            logger.warning(f"API Analysis Helper: {msg}")
            return None, None, msg, 404
        # Use primaryImage for potentially better quality analysis
        original_image_url = object_data.get('primaryImage') or object_data.get('primaryImageSmall')
        if not original_image_url:
            msg = f"No image URL found for Met ID {object_id}"
            logger.warning(f"API Analysis Helper: {msg}")
            return None, None, msg, 404
    elif object_id.startswith("url_"):  # Assume URL-processed ID
        logger.debug(f"ID {object_id} appears to be a URL-processed ID.")
        processed_urls = load_processed_urls()
        url_info = processed_urls.get(object_id)
        if not url_info or not url_info.get('original_url'):
            msg = f"Original URL not found for processed ID {object_id}"
            logger.warning(f"API Analysis Helper: {msg}")
            return None, None, msg, 404
        original_image_url = url_info['original_url']
    elif object_id.startswith("uploaded_"):  # Assume Uploaded ID
        logger.debug(f"ID {object_id} appears to be an uploaded image ID.")
        # Uploaded images might still be stored locally - check for local file first
        local_file_path = os.path.join(UPLOADED_IMAGES_DIR, f"{object_id}.jpg")
        if os.path.exists(local_file_path):
            logger.debug(f"API Analysis Helper: Found local file for uploaded ID: {local_file_path}")
            try:
                img_pil = Image.open(local_file_path)
                if img_pil.mode != 'RGB':
                    img_pil = img_pil.convert('RGB')
                logger.info(f"API Analysis Helper: Loaded uploaded image from local file: {local_file_path}")
            except Exception as e:
                logger.error(f"API Analysis Helper: Error loading local image file: {e}")
                # Fall back to URL if available
        
        # If local file doesn't exist or failed to load, try uploaded_artworks JSON
        if img_pil is None:
            uploaded_artworks = load_uploaded_artworks()
            for artwork in uploaded_artworks:
                if artwork.get('id') == object_id:
                    original_image_url = artwork.get('originalUrl')
                    break
            
            if not original_image_url:
                msg = f"No image URL found for uploaded ID {object_id}"
                logger.warning(f"API Analysis Helper: {msg}")
                return None, None, msg, 404
    else:
        msg = f"Invalid object ID format: {object_id}"
        logger.warning(f"API Analysis Helper: {msg}")
        return None, None, msg, 400

    logger.info(f"API Analysis Helper: Determined image URL for {object_id}: {original_image_url or 'using local file'}")

    # 2. Download/Load the image if it's not already loaded
    if img_pil is None and original_image_url:
        try:
            # Download from URL
            logger.debug(f"API Analysis Helper: Downloading image from URL: {original_image_url}")
            img_response = requests.get(original_image_url, timeout=20)
            img_response.raise_for_status()
            img_pil = Image.open(BytesIO(img_response.content))

            # Convert to RGB if needed
            if img_pil.mode != 'RGB':
                logger.debug(f"API Analysis Helper: Converting image {object_id} from {img_pil.mode} to RGB")
                img_pil = img_pil.convert('RGB')
            logger.info(f"API Analysis Helper: Image loaded successfully for {object_id}")

        except requests.exceptions.RequestException as e:
            logger.error(f"API Analysis Helper: Failed to download image for {object_id} from {original_image_url}: {e}")
            return None, None, f"Failed to download image for ID {object_id}", 502
        except UnidentifiedImageError:
            logger.error(f"API Analysis Helper: Invalid image format for {object_id} at {original_image_url}")
            return None, None, f"Invalid image format for ID {object_id}", 400
        except Exception as e:
            logger.error(f"API Analysis Helper: Error loading image for {object_id}: {e}", exc_info=True)
            return None, None, f"Error loading image for ID {object_id}", 500

    # 3. Load the embedding if required
    if require_embedding:
        embedding_path = os.path.join(PROCESSED_EMBEDDINGS_DIR, f'{object_id}.npy')
        if os.path.exists(embedding_path):
            try:
                embedding = np.load(embedding_path)
                logger.info(f"API Analysis Helper: Loaded embedding for {object_id}")
            except Exception as e:
                logger.warning(f"API Analysis Helper: Error loading embedding for {object_id} from {embedding_path}: {e}. Proceeding without embedding.")
                embedding = None  # Continue analysis without embedding if loading fails
        else:
            logger.warning(f"API Analysis Helper: Embedding file not found for {object_id} at {embedding_path}. Proceeding without embedding.")
            embedding = None

    return img_pil, embedding, None, 200


# --- Other Endpoints (Color Analysis POST, Proxy, Upload, Serve Image, Diagnostics etc.) ---
# Add logging and improve error handling in these as well

@app.route('/api/color-analysis', methods=['POST'])
@cross_origin() # Added cross_origin
def color_analysis():
    # (Keep existing logic, but add logging/better error handling)
    logger.info("API: Request received for /api/color-analysis (POST)")
    img = None
    try:
        # Check for image file or URL
        if 'image' in request.files:
            image_file_storage = request.files['image']
            if image_file_storage and image_file_storage.filename:
                logger.debug("API Color POST: Processing uploaded image file.")
                img = Image.open(image_file_storage.stream) # Read from stream
            else:
                logger.warning("API Color POST: 'image' provided but file is empty or missing.")
                return jsonify({"error": "Empty or invalid image file provided"}), 400
        elif 'image_url' in request.form:
            image_url = request.form['image_url']
            logger.debug(f"API Color POST: Processing image from URL: {image_url}")
            response = requests.get(image_url, timeout=15)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content))
        else:
            logger.warning("API Color POST: No image file or image_url provided.")
            return jsonify({"error": "No image file or image_url provided"}), 400

        if img is None: # Should not happen if logic above is correct, but check
            logger.error("API Color POST: Image object is None after processing input.")
            return jsonify({"error": "Failed to process image"}), 500

        # Perform color analysis
        try:
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
                
            # Call the same enhanced_color_analysis used by the analyze/color endpoint
            color_result = enhanced_color_analysis(img)
            logger.info("API Color POST: Analysis completed successfully")
            return jsonify(color_result)
            
        except Exception as analysis_error:
            logger.error(f"API Color POST: Error during analysis: {analysis_error}", exc_info=True)
            return jsonify({
                "error": "Color analysis failed",
                "details": str(analysis_error)
            }), 500
            
    except requests.exceptions.RequestException as e:
        logger.error(f"API Color POST: Error fetching image from URL: {e}")
        return jsonify({"error": f"Failed to fetch image from URL: {str(e)}"}), 502
    except UnidentifiedImageError as e:
        logger.error(f"API Color POST: Invalid image format: {e}")
        return jsonify({"error": "Invalid or unsupported image format"}), 400
    except Exception as e:
        logger.error(f"API Color POST: Unexpected error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/api/diagnostics', methods=['GET'])
def diagnostics():
    """
    Provides diagnostic information about the server's state, including:
    - Directory statuses
    - File counts
    - Metadata stats
    - System information
    
    Useful for debugging deployment issues.
    """
    logger.info("API: Request received for /api/diagnostics")
    start_time = time.time()
    
    try:
        # Check directory statuses
        dir_status = {}
        for dir_path in [
            DATA_DIR,
            PROCESSED_EMBEDDINGS_DIR, 
            PROCESSED_GRADCAM_DIR,
            PROCESSED_ATTENTION_DIR,
            UPLOADED_IMAGES_DIR,
            MET_IMAGES_DIR,
            RIJKS_IMAGES_DIR,
            AIC_IMAGES_DIR
        ]:
            dir_status[os.path.basename(dir_path)] = {
                "exists": os.path.exists(dir_path),
                "is_dir": os.path.isdir(dir_path) if os.path.exists(dir_path) else False,
                "path": dir_path
            }
            
        # Check metadata files
        metadata_stats = {}
        
        # Check preprocessed_metadata.json
        preprocessed_metadata_path = os.path.join(DATA_DIR, "preprocessed_metadata.json")
        if os.path.exists(preprocessed_metadata_path):
            try:
                file_size = os.path.getsize(preprocessed_metadata_path)
                with open(preprocessed_metadata_path, 'r') as f:
                    meta_data = json.load(f)
                    
                metadata_stats["preprocessed_metadata"] = {
                    "exists": True,
                    "file_size": file_size,
                    "items_count": len(meta_data.get('metadata', [])),
                    "last_modified": os.path.getmtime(preprocessed_metadata_path)
                }
            except Exception as e:
                metadata_stats["preprocessed_metadata"] = {
                    "exists": True,
                    "error": str(e),
                    "file_size": os.path.getsize(preprocessed_metadata_path) if os.path.exists(preprocessed_metadata_path) else 0,
                    "last_modified": os.path.getmtime(preprocessed_metadata_path) if os.path.exists(preprocessed_metadata_path) else 0
                }
        else:
            metadata_stats["preprocessed_metadata"] = {
                "exists": False
            }
            
        # Check uploaded_artworks.json
        uploaded_artworks_path = UPLOADED_ARTWORKS_FILE
        if os.path.exists(uploaded_artworks_path):
            try:
                file_size = os.path.getsize(uploaded_artworks_path)
                with open(uploaded_artworks_path, 'r') as f:
                    uploaded_data = json.load(f)
                    
                metadata_stats["uploaded_artworks"] = {
                    "exists": True,
                    "file_size": file_size,
                    "items_count": len(uploaded_data),
                    "last_modified": os.path.getmtime(uploaded_artworks_path)
                }
            except Exception as e:
                metadata_stats["uploaded_artworks"] = {
                    "exists": True,
                    "error": str(e),
                    "file_size": os.path.getsize(uploaded_artworks_path) if os.path.exists(uploaded_artworks_path) else 0,
                    "last_modified": os.path.getmtime(uploaded_artworks_path) if os.path.exists(uploaded_artworks_path) else 0
                }
        else:
            metadata_stats["uploaded_artworks"] = {
                "exists": False
            }
        
        # Count files in directories
        dir_counts = {}
        for dir_path in [
            PROCESSED_EMBEDDINGS_DIR, 
            PROCESSED_GRADCAM_DIR,
            PROCESSED_ATTENTION_DIR,
            UPLOADED_IMAGES_DIR,
            MET_IMAGES_DIR,
            RIJKS_IMAGES_DIR,
            AIC_IMAGES_DIR
        ]:
            if os.path.exists(dir_path):
                files = os.listdir(dir_path)
                dir_counts[os.path.basename(dir_path)] = len(files)
                
                # Add sample file info for verification
                if files:
                    sample_files = {}
                    for sample_file in files[:3]:  # Take first 3 files as samples
                        sample_path = os.path.join(dir_path, sample_file)
                        if os.path.isfile(sample_path):
                            sample_files[sample_file] = {
                                "size": os.path.getsize(sample_path),
                                "last_modified": os.path.getmtime(sample_path)
                            }
                    
                    dir_counts[f"{os.path.basename(dir_path)}_samples"] = sample_files
            else:
                dir_counts[os.path.basename(dir_path)] = "directory_not_found"
        
        # Other useful diagnostic information
        system_info = {
            "current_working_dir": os.getcwd(),
            "data_dir": DATA_DIR,
            "data_dir_exists": os.path.exists(DATA_DIR),
            "python_version": sys.version,
            "server_timestamp": time.time(),
            "diagnostics_duration": time.time() - start_time
        }
        
        # Combine everything into response
        diagnostics_result = {
            "system_info": system_info,
            "directory_status": dir_status,
            "file_counts": dir_counts,
            "metadata_stats": metadata_stats
        }
        
        logger.info(f"API: Diagnostics completed in {time.time() - start_time:.3f}s")
        return jsonify(diagnostics_result)
    
    except Exception as e:
        logger.error(f"API: Error during diagnostics: {e}", exc_info=True)
        return jsonify({
            "status": "error",
            "message": str(e),
            "traceback": traceback.format_exc()
        }), 500

# Добавляем wrapper для generate_embedding, используемый в preprocess_artworks.py
def generate_embedding_wrapper(img_pil):
    """
    Wrapper function for generate_embedding used by preprocess_artworks.py
    
    Args:
        img_pil: PIL Image object
        
    Returns:
        numpy array of embedding or None on error
    """
    try:
        logger.info("Generating embedding via wrapper function")
        if img_pil.mode != 'RGB':
            img_pil = img_pil.convert('RGB')
        return generate_embedding(img_pil)
    except Exception as e:
        logger.error(f"Error in generate_embedding_wrapper: {e}", exc_info=True)
        return None

# Data directories
DATA_DIR = os.path.join(current_dir, "data")
EMBEDDINGS_DIR = os.path.join(DATA_DIR, "embeddings")
GRADCAM_DIR = os.path.join(DATA_DIR, "gradcam")
ATTENTION_DIR = os.path.join(DATA_DIR, "attention")

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
os.makedirs(GRADCAM_DIR, exist_ok=True)
os.makedirs(ATTENTION_DIR, exist_ok=True)

# Cache for processed URLs
PROCESSED_URLS_FILE = os.path.join(DATA_DIR, "processed_urls.json")

# Health check endpoint
@app.route('/api/health', methods=['GET'])
def health_check():
    """API health check endpoint that returns server status"""
    logger.info("Health check requested")
    return jsonify({
        "status": "ok", 
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    })

# Get artworks directly from museum APIs
@app.route('/api/museum_artworks')
def get_museum_artworks():
    """Get artworks directly from museum APIs"""
    logger.info("Museum artworks requested")
    
    try:
        # Get pagination parameters
        page = request.args.get('page', default=1, type=int)
        per_page = request.args.get('limit', default=20, type=int)
        
        # Validate pagination parameters
        if page < 1 or per_page < 1 or per_page > 100:
            return jsonify({"error": "Invalid pagination parameters"}), 400
        
        # Fetch objects from Met API directly
        logger.info(f"Fetching data from Met Museum API, page={page}, limit={per_page}")
        
        # Fetch a fixed number of artworks (we'll handle pagination here)
        met_artworks, embeddings = fetch_met_artworks() # <-- NEW CALL using defaults from utils.py
        
        # Calculate starting and ending indices for pagination
        start_idx = (page - 1) * per_page
        end_idx = min(start_idx + per_page, len(met_artworks))
        
        # Get artworks for the requested page
        artworks_page = met_artworks[start_idx:end_idx] if start_idx < len(met_artworks) else []
        embeddings_page = embeddings[start_idx:end_idx] if start_idx < len(embeddings) else []
        
        result = {
            "page": page,
            "total": len(met_artworks),
            "metadata": artworks_page,
            "embeddings": embeddings_page
        }
        
        logger.info(f"Returning {len(artworks_page)} museum artworks with embeddings for page {page}")
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error fetching museum artworks: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": "Internal server error", "details": str(e)}), 500

# Proxy image endpoint to avoid CORS issues
@app.route('/api/proxy_image')
def proxy_image():
    """Proxy an image from an external URL to avoid CORS issues"""
    url = request.args.get('url')
    if not url:
        return jsonify({"error": "No URL provided"}), 400
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        # Get content type from response
        content_type = response.headers.get('Content-Type', 'image/jpeg')
        
        # Return image with original content type
        return response.content, 200, {'Content-Type': content_type}
    except Exception as e:
        logger.error(f"Error proxying image from {url}: {str(e)}")
        return jsonify({"error": "Failed to proxy image"}), 500

@app.route('/api/upload_and_analyze', methods=['POST'])
@cross_origin()
def upload_and_analyze():
    """
    Принимает загруженный файл изображения, генерирует для него ID, 
    вычисляет эмбеддинг и все виды анализа (стиль, цвет, композиция),
    сохраняет эмбеддинг на диск и возвращает все результаты.
    """
    logger.info("API UploadAnalyze: Request received")
    
    if 'image' not in request.files:
        logger.warning("API UploadAnalyze: 'image' file part missing")
        return jsonify({"error": "Missing 'image' file part in the request"}), 400

    image_file_storage = request.files['image']

    if image_file_storage.filename == '':
        logger.warning("API UploadAnalyze: No selected file")
        return jsonify({"error": "No selected file"}), 400

    if not image_file_storage:
        logger.warning("API UploadAnalyze: Invalid image file storage object")
        return jsonify({"error": "Invalid image file provided"}), 400
        
    unique_id = f"uploaded_{uuid.uuid4()}"
    logger.info(f"API UploadAnalyze: Generated ID: {unique_id}")
    
    img_pil = None
    embedding = None
    style_result = None
    color_result = None
    comp_result = None
    
    try:
        logger.debug(f"API UploadAnalyze: Reading image stream for {unique_id}")
        image_file_storage.seek(0) # Ensure stream is at the beginning
        img_pil = Image.open(image_file_storage.stream)
        if img_pil.mode != 'RGB':
            logger.debug(f"API UploadAnalyze: Converting image {unique_id} to RGB")
            img_pil = img_pil.convert('RGB')
        logger.info(f"API UploadAnalyze: Image {unique_id} loaded successfully.")

        # === ДОБАВЛЕНО: Сохранение изображения на диск ===
        try:
            save_path = os.path.join(UPLOADED_IMAGES_DIR, f"{unique_id}.jpg")
            img_pil.save(save_path, format='JPEG', quality=90) # Save as JPEG with decent quality
            logger.info(f"API UploadAnalyze: Saved uploaded image to {save_path}")
        except Exception as save_err:
             # Log the error but continue with analysis, as the image is in memory
            logger.error(f"API UploadAnalyze: Failed to save image {unique_id} to disk: {save_err}", exc_info=True)
        # === КОНЕЦ ДОБАВЛЕНИЯ ===

        # 1. Генерация эмбеддинга
        try:
            logger.info(f"API UploadAnalyze: Generating embedding for {unique_id}...")
            embedding_np = generate_embedding_wrapper(img_pil)
            if embedding_np is not None:
                embedding = embedding_np.tolist()
                embedding_path = os.path.join(PROCESSED_EMBEDDINGS_DIR, f'{unique_id}.npy')
                np.save(embedding_path, embedding_np)
                logger.info(f"API UploadAnalyze: Embedding generated and saved to {embedding_path}")
            else:
                logger.warning(f"API UploadAnalyze: Embedding generation failed for {unique_id}")
        except Exception as emb_err:
            logger.error(f"API UploadAnalyze: Error during embedding generation for {unique_id}: {emb_err}", exc_info=True)
            embedding = None

        try:
            logger.info(f"API UploadAnalyze: Analyzing style for {unique_id}...")
            style_result = enhanced_style_analysis(img_pil, embedding_np if embedding is not None else None)
            logger.info(f"API UploadAnalyze: Style analysis complete for {unique_id}")
        except Exception as style_err:
            logger.error(f"API UploadAnalyze: Error during style analysis for {unique_id}: {style_err}", exc_info=True)
            style_result = {"error": "Style analysis failed", "is_fallback": True}
            
        try:
            logger.info(f"API UploadAnalyze: Analyzing color for {unique_id}...")
            color_result = enhanced_color_analysis(img_pil)
            logger.info(f"API UploadAnalyze: Color analysis complete for {unique_id}")
        except Exception as color_err:
            logger.error(f"API UploadAnalyze: Error during color analysis for {unique_id}: {color_err}", exc_info=True)
            color_result = {"error": "Color analysis failed", "is_fallback": True}

        try:
            logger.info(f"API UploadAnalyze: Analyzing composition for {unique_id}...")
            comp_result = enhanced_composition_analysis(img_pil)
            logger.info(f"API UploadAnalyze: Composition analysis complete for {unique_id}")
        except Exception as comp_err:
            logger.error(f"API UploadAnalyze: Error during composition analysis for {unique_id}: {comp_err}", exc_info=True)
            comp_result = {"error": "Composition analysis failed", "is_fallback": True}

        response_data = {
            "message": "Image uploaded and analyzed successfully",
            "processed_id": unique_id,
            "embedding": embedding,
            "analysis": {
                "style": style_result,
                "color": color_result,
                "composition": comp_result
            },
            "original_filename": image_file_storage.filename
        }
        
        logger.info(f"API UploadAnalyze: Successfully processed {unique_id}")
        return jsonify(response_data), 200

    except UnidentifiedImageError:
        logger.error(f"API UploadAnalyze: Cannot identify image file: {image_file_storage.filename}")
        return jsonify({"error": "Cannot identify image file. Invalid or unsupported format."}), 400
    except Exception as e:
        logger.error(f"API UploadAnalyze: Unexpected error processing file {image_file_storage.filename} ({unique_id}): {e}", exc_info=True)
        return jsonify({"error": "Internal server error during image processing"}), 500

# Start the server when script is run directly
if __name__ == "__main__":
    # Log startup
    logger.info("Starting RenAI API server")
    
    # Get port from environment variable or use 5000 as default
    port = int(os.environ.get("PORT", 5000))
    
    # Start server
    app.run(debug=True, host='0.0.0.0', port=port)
