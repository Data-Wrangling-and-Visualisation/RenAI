import os
import sys
import numpy as np
import logging
from PIL import Image
from io import BytesIO
import requests

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("utils")

# Add processing-data directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
processing_data_path = os.path.join(current_dir, '../processing-data')
sys.path.append(processing_data_path)

# Try to import the ML model functions
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

# Wrapper functions that will be used by both api.py and preprocess_artworks.py
def generate_embedding_wrapper(img_pil):
    """Wrapper for generate_embedding to handle potential errors"""
    try:
        if img_pil is None:
            logger.error("Cannot generate embedding: image is None")
            return None
        
        # Convert to RGB if needed
        if img_pil.mode != 'RGB':
            img_pil = img_pil.convert('RGB')
        
        # Call the actual embedding function
        embedding = generate_embedding(img_pil)
        
        if embedding is None or not isinstance(embedding, np.ndarray):
            logger.error(f"generate_embedding returned invalid result: {type(embedding)}")
            return None
        
        return embedding
    except Exception as e:
        logger.error(f"Error in generate_embedding_wrapper: {str(e)}")
        return None

def calculate_gradcam_wrapper(img_pil):
    """Wrapper for calculate_gradcam to handle potential errors"""
    try:
        if img_pil is None:
            logger.error("Cannot calculate GradCAM: image is None")
            return None
        
        # Convert to RGB if needed
        if img_pil.mode != 'RGB':
            img_pil = img_pil.convert('RGB')
        
        # Call the actual gradcam function
        gradcam_map = calculate_gradcam(img_pil)
        
        if gradcam_map is None or not isinstance(gradcam_map, np.ndarray):
            logger.error(f"calculate_gradcam returned invalid result: {type(gradcam_map)}")
            return None
        
        return gradcam_map
    except Exception as e:
        logger.error(f"Error in calculate_gradcam_wrapper: {str(e)}")
        return None

def generate_heatmap_wrapper(img_pil):
    """Wrapper for generate_heatmap to handle potential errors"""
    try:
        if img_pil is None:
            logger.error("Cannot generate heatmap: image is None")
            return None
        
        # Convert to RGB if needed
        if img_pil.mode != 'RGB':
            img_pil = img_pil.convert('RGB')
        
        # Call the actual heatmap function
        heatmap = generate_heatmap(img_pil)
        
        if heatmap is None or not isinstance(heatmap, np.ndarray):
            logger.error(f"generate_heatmap returned invalid result: {type(heatmap)}")
            return None
        
        return heatmap
    except Exception as e:
        logger.error(f"Error in generate_heatmap_wrapper: {str(e)}")
        return None

# Define DATA_DIR at the module level for use in fetch_met_artworks
DATA_DIR = os.path.join(current_dir, 'data')
EMBEDDINGS_DIR = os.path.join(DATA_DIR, 'embeddings')
os.makedirs(EMBEDDINGS_DIR, exist_ok=True) # Ensure it exists

# Define default department limits
DEFAULT_DEPT_LIMITS = { 
    1: 50,    # Asian Art
    11: 150,  # European Paintings
    21: 300   # Modern and Contemporary Art
}

def fetch_met_artworks(dept_limits=DEFAULT_DEPT_LIMITS, hasImage=True):
    """
    Fetches artwork data from specified departments in the Met API up to specified limits 
    and loads corresponding embeddings.
    
    Args:
        dept_limits: Dictionary mapping department ID to the desired number of artworks.
        hasImage: Whether to only fetch artworks with images
    
    Returns:
        Tuple containing (list of artwork metadata dictionaries, list of corresponding embeddings or None)
    """
    total_limit = sum(dept_limits.values())
    requested_departments = list(dept_limits.keys())
    logger.info(f"Fetching artworks from Met API. Targets: {dept_limits}, Total limit: {total_limit}, hasImage: {hasImage}")
    
    try:
        all_object_ids = []
        # Iterate through the requested departments and fetch IDs up to the specific limit for each
        for dept_id, limit_for_dept in dept_limits.items():
            dept_url = f"https://collectionapi.metmuseum.org/public/collection/v1/objects?departmentIds={dept_id}"
            if hasImage:
                dept_url += "&hasImages=true"
                
            logger.info(f"Querying Met API for object IDs in department {dept_id} (target: {limit_for_dept})...")
            response = requests.get(dept_url, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                if data and data.get('objectIDs'):
                    ids_in_dept = data['objectIDs']
                    # Take only up to the specified limit for this department
                    ids_to_add = ids_in_dept[:limit_for_dept]
                    all_object_ids.extend(ids_to_add)
                    logger.info(f"Found {len(ids_in_dept)} objects in dept {dept_id}, added {len(ids_to_add)} IDs.")
                else:
                    logger.warning(f"No object IDs found for department {dept_id} in API response.")
            else:
                 logger.error(f"Failed to fetch object IDs from Met API for department {dept_id}: {response.status_code}")
        
        # Remove potential duplicates if departments overlap or API returns duplicates
        unique_object_ids = list(dict.fromkeys(all_object_ids))
        logger.info(f"Total unique object IDs gathered: {len(unique_object_ids)}")
            
        # Fetch detailed information and try loading embeddings for each unique object ID
        final_metadata = []
        final_embeddings = []
        
        # Process only the unique IDs
        for obj_id in unique_object_ids:
            try:
                # Fetch object metadata
                obj_url = f"https://collectionapi.metmuseum.org/public/collection/v1/objects/{obj_id}"
                obj_response = requests.get(obj_url, timeout=20)
                if obj_response.status_code != 200:
                    logger.warning(f"Failed to fetch object {obj_id}: {obj_response.status_code}")
                    continue
                obj_data = obj_response.json()
                
                # Check image requirement (redundant if API query included hasImage=true, but safe)
                if hasImage and (not obj_data.get('primaryImage') and not obj_data.get('primaryImageSmall')):
                    # logger.debug(f"Object {obj_id} has no image, skipping") # Less verbose
                    continue
                
                # Add source and ensure ID is consistent
                obj_data['source'] = 'met'
                obj_data['id'] = obj_id 
                
                # Try to load the embedding
                embedding = None
                embedding_path = os.path.join(EMBEDDINGS_DIR, f'{obj_id}.npy')
                if os.path.exists(embedding_path):
                    try:
                        embedding = np.load(embedding_path)
                        # logger.debug(f"Successfully loaded embedding for object {obj_id}") # Less verbose
                    except Exception as load_err:
                        logger.error(f"Error loading embedding file {embedding_path} for object {obj_id}: {load_err}")
                        embedding = None 
                else:
                    # logger.warning(f"Embedding file not found for object {obj_id} at {embedding_path}") # Less verbose
                    embedding = None 
                    
                # Append metadata and the loaded embedding (or None)
                final_metadata.append(obj_data)
                final_embeddings.append(embedding.tolist() if embedding is not None else None)
                
                # logger.debug(f"Processed object {obj_id}: {obj_data.get('title')}") # Less verbose
                    
            except Exception as e:
                logger.error(f"Error processing details/embedding for object {obj_id}: {str(e)}")
    
    except Exception as e:
        logger.error(f"Error during overall artwork fetching: {str(e)}")
    
    logger.info(f"Finished fetching. Returning {len(final_metadata)} metadata entries and {len(final_embeddings)} embeddings (may include None values)." )
    
    return (final_metadata, final_embeddings)

def fetch_rijksmuseum_artworks(limit=50, hasImage=True):
    """
    Fetch artwork data from Rijksmuseum API
    
    Returns:
        Tuple containing (list of artwork metadata dictionaries, empty list for embeddings)
    """
    logger.info(f"Fetching up to {limit} artworks from Rijksmuseum API")
    # This is a placeholder - implement the actual function if needed
    return ([], [])
    
def fetch_aic_artworks(limit=50, hasImage=True):
    """
    Fetch artwork data from Art Institute of Chicago API
    
    Returns:
        Tuple containing (list of artwork metadata dictionaries, empty list for embeddings)
    """
    logger.info(f"Fetching up to {limit} artworks from Art Institute of Chicago API")
    # This is a placeholder - implement the actual function if needed
    return ([], [])

def process_uploaded_images(directory_path=None):
    """
    Process images that were uploaded by users
    """
    logger.info(f"Processing uploaded images from directory: {directory_path}")
    # This is a placeholder - implement the actual function if needed
    return [], []  # Возвращаем пустой список метаданных и пустой список эмбеддингов 