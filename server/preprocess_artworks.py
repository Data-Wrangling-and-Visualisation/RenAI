#!/usr/bin/env python3
"""
Artwork Preprocessing Script for RenAI

This script fetches artwork data from multiple museums, generates embeddings,
and saves them to disk for faster loading by the main application.

Usage:
    python preprocess_artworks.py [--limit LIMIT] [--force]

Options:
    --limit LIMIT    Limit the number of artworks per museum (default: 100)
    --force          Force regeneration of all embeddings, even if they exist
"""

import os
import sys
import json
import time
import argparse
import numpy as np
from PIL import Image
import requests
from io import BytesIO
from tqdm import tqdm
import logging
import traceback
from utils import (
    fetch_met_artworks,
    fetch_rijksmuseum_artworks,
    fetch_aic_artworks,
    generate_embedding_wrapper,
    calculate_gradcam_wrapper,
    generate_heatmap_wrapper,
    process_uploaded_images
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("preprocess.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("preprocess")

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
EMBEDDINGS_DIR = os.path.join(DATA_DIR, "embeddings")
METADATA_FILE = os.path.join(DATA_DIR, "preprocessed_metadata.json")

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

def download_image(url, timeout=10):
    """Download an image from a URL and return as PIL Image"""
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        return Image.open(BytesIO(response.content))
    except Exception as e:
        logger.error(f"Error downloading image from {url}: {str(e)}")
        return None

def process_artwork(artwork, force=False):
    """
    Process a single artwork: download image, generate embedding, save to disk
    
    Args:
        artwork: Dict containing artwork metadata
        force: If True, regenerate embedding even if it exists
        
    Returns:
        (embedding_path, success): Path to saved embedding and success status
    """
    artwork_id = artwork.get('id') or artwork.get('objectID')
    if not artwork_id:
        logger.warning(f"Artwork missing ID, skipping: {artwork}")
        return None, False
    
    # Define the embedding filename
    embedding_path = os.path.join(EMBEDDINGS_DIR, f"{artwork_id}.npy")
    
    # Skip if embedding already exists and not forcing regeneration
    if os.path.exists(embedding_path) and not force:
        logger.info(f"Embedding already exists for {artwork_id}, skipping")
        # Load the existing embedding to return
        try:
            embedding = np.load(embedding_path)
            return embedding_path, True
        except Exception as e:
            logger.error(f"Error loading existing embedding for {artwork_id}: {str(e)}")
            # Continue to regenerate if loading fails
    
    # Get the image URL
    image_url = artwork.get('primaryImage') or artwork.get('primaryImageSmall')
    if not image_url:
        logger.warning(f"No image URL found for artwork {artwork_id}, skipping")
        return None, False
    
    # Download the image
    logger.info(f"Downloading image for artwork {artwork_id}")
    image = download_image(image_url)
    if not image:
        logger.error(f"Failed to download image for artwork {artwork_id}")
        return None, False
    
    # Generate embedding
    logger.info(f"Generating embedding for artwork {artwork_id}")
    try:
        # Use the wrapper from api.py
        embedding = generate_embedding_wrapper(image)
        
        if embedding is None:
            logger.error(f"Failed to generate embedding for artwork {artwork_id}")
            return None, False
        
        # Save embedding to disk
        np.save(embedding_path, embedding)
        logger.info(f"Saved embedding for artwork {artwork_id} to {embedding_path}")
        
        return embedding_path, True
    except Exception as e:
        logger.error(f"Error generating embedding for {artwork_id}: {str(e)}")
        return None, False

def process_artworks_batch(artworks, force=False):
    """
    Process a batch of artworks, tracking success and failures
    
    Args:
        artworks: List of artwork metadata dicts
        force: If True, regenerate embeddings even if they exist
        
    Returns:
        (processed_artworks, processed_embeddings): Lists of successful metadata and paths
    """
    processed_artworks = []
    processed_embeddings = []
    
    logger.info(f"Processing batch of {len(artworks)} artworks")
    
    for artwork in tqdm(artworks, desc="Processing artworks"):
        try:
            embedding_path, success = process_artwork(artwork, force)
            
            if success:
                # Add the embedding path to the artwork metadata
                artwork_copy = artwork.copy()
                # Calculate the relative path from BASE_DIR
                relative_embedding_path = os.path.relpath(embedding_path, BASE_DIR)
                artwork_copy['embedding_path'] = relative_embedding_path.replace('\\', '/') # Replace backslashes with forward slashes for compatibility
                processed_artworks.append(artwork_copy)
                
                # Load the embedding to include in the processed_embeddings list
                embedding = np.load(embedding_path)
                processed_embeddings.append(embedding)
            else:
                logger.warning(f"Failed to process artwork: {artwork.get('id') or artwork.get('objectID')}")
        except Exception as e:
            logger.error(f"Error processing artwork: {str(e)}\n{traceback.format_exc()}")
    
    return processed_artworks, processed_embeddings

def preprocess_artworks(output_file="preprocessed_metadata.json"):
    """
    Fetches artwork data from all sources, generates embeddings,
    and saves the preprocessed data to a JSON file.
    
    """
    print("Starting artwork preprocessing...")
    start_time = time.time()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    
    # Initialize an empty JSON file with the base structure
    initial_data = {
        "total": 0,
        "metadata": [],
        "embeddings": []
    }
    
    # Create the JSON file immediately, even if it's empty
    if not os.path.exists(output_file):
        print(f"Creating initial empty JSON file: {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(initial_data, f)
    else:
        print(f"Using existing JSON file: {output_file}")
        # Check if the file has the correct structure
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
                if "metadata" not in existing_data or "embeddings" not in existing_data:
                    print(f"Existing file has incorrect structure, recreating it")
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(initial_data, f)
        except Exception as e:
            print(f"Error reading existing file, recreating it: {str(e)}")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(initial_data, f)
    
    # Function to incrementally add a new object to the JSON file
    def add_artwork_to_json(metadata_item, embedding_item):
        try:
            artwork_id = metadata_item.get('id') or metadata_item.get('objectID')
            logger.info(f"[JSON UPDATE] Adding artwork {artwork_id} to JSON file...")
            
            # Load the current JSON
            with open(output_file, 'r', encoding='utf-8') as f:
                current_data = json.load(f)
            
            # Log the current state of the JSON
            logger.info(f"[JSON UPDATE] Current JSON contains {len(current_data['metadata'])} items")
            
            # Add the new object
            current_data["metadata"].append(metadata_item)
            
            # Prepare the embedding for saving in JSON
            if isinstance(embedding_item, np.ndarray):
                logger.info(f"[JSON UPDATE] Converting numpy embedding of shape {embedding_item.shape} to list")
                embedding_list = embedding_item.tolist()
            else:
                embedding_list = embedding_item
                
            current_data["embeddings"].append(embedding_list)
            current_data["total"] = len(current_data["metadata"])
            
            # Save the updated JSON with time tracking of the operation
            start_time = time.time()
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(current_data, f)
            
            elapsed_time = time.time() - start_time
            logger.info(f"[JSON UPDATE] JSON file updated successfully in {elapsed_time:.2f} seconds. New total: {current_data['total']} items")
            
            return True
        except Exception as e:
            logger.error(f"[JSON UPDATE] Error updating JSON with artwork {metadata_item.get('id')}: {str(e)}")
            traceback.print_exc()
            return False
    
    # Function to process one artwork and add it to JSON
    def process_and_add_artwork(artwork, source_name="unknown"):
        try:
            if not artwork:
                logger.warning(f"[PROCESS] Artwork is None or empty, skipping")
                return False
            
            artwork_id = artwork.get('id') or artwork.get('objectID')
            if not artwork_id:
                logger.warning(f"[PROCESS] Artwork missing ID, skipping: {artwork}")
                return False
            
            logger.info(f"[PROCESS] Processing artwork {artwork_id} from {source_name}...")
            
            # Define the path to the embedding
            embedding_path = os.path.join(EMBEDDINGS_DIR, f"{artwork_id}.npy")
            gradcam_path = os.path.join(os.path.dirname(EMBEDDINGS_DIR), "gradcam", f"{artwork_id}.npy")
            attention_path = os.path.join(os.path.dirname(EMBEDDINGS_DIR), "attention", f"{artwork_id}.npy")
            
            logger.info(f"[PROCESS] Embedding path: {embedding_path}")
            
            # Check if the embedding exists
            if os.path.exists(embedding_path):
                try:
                    # Load the existing embedding
                    start_time = time.time()
                    embedding = np.load(embedding_path)
                    load_time = time.time() - start_time
                    logger.info(f"[PROCESS] Using existing embedding for {artwork_id}, shape: {embedding.shape}, loaded in {load_time:.3f}s")
                    
                    # Add the embedding path to the metadata
                    artwork_copy = artwork.copy()
                    # Calculate the relative path from BASE_DIR
                    relative_embedding_path = os.path.relpath(embedding_path, BASE_DIR)
                    artwork_copy['embedding_path'] = relative_embedding_path.replace('\\', '/') # Replace backslashes with forward slashes for compatibility
                    
                    # Add information about directories for debugging
                    artwork_copy['_debug_info'] = {
                        'processed_by': 'preprocess_artworks.py',
                        'embedding_dir': EMBEDDINGS_DIR,
                        'data_dir': DATA_DIR,
                        'timestamp': time.time()
                    }
                    
                    # Add to JSON
                    logger.info(f"[PROCESS] Adding existing artwork {artwork_id} to JSON")
                    add_result = add_artwork_to_json(artwork_copy, embedding)
                    logger.info(f"[PROCESS] Artwork {artwork_id} added to JSON: {add_result}")
                    return True
                except Exception as e:
                    logger.error(f"[PROCESS] Error loading existing embedding for {artwork_id}: {str(e)}")
                    logger.error(traceback.format_exc())
                    # If unable to load, create a new one
            
            # If the embedding does not exist or cannot be loaded, create a new one
            # Get the image URL
            image_url = artwork.get('primaryImage') or artwork.get('primaryImageSmall') or artwork.get('originalUrl')
            if not image_url:
                logger.warning(f"[PROCESS] No image URL found for artwork {artwork_id}, skipping")
                return False
            
            # Download the image to memory, not saving to disk
            logger.info(f"[PROCESS] Downloading image for artwork {artwork_id} from {image_url}")
            download_start = time.time()
            try:
                image_response = requests.get(image_url, timeout=20)
                image_response.raise_for_status()
                image = Image.open(BytesIO(image_response.content))
                if image.mode != 'RGB':
                    image = image.convert('RGB')
            except Exception as e:
                logger.error(f"[PROCESS] Failed to download/process image for artwork {artwork_id}: {str(e)}")
                return False
            
            download_time = time.time() - download_start
            
            logger.info(f"[PROCESS] Image downloaded in {download_time:.2f}s, size: {image.size}, mode: {image.mode}")
            
            # Generate the embedding
            logger.info(f"[PROCESS] Generating embedding for artwork {artwork_id}")
            try:
                embedding_start = time.time()
                embedding = generate_embedding_wrapper(image)
                embedding_time = time.time() - embedding_start
                
                if embedding is None:
                    logger.error(f"[PROCESS] Failed to generate embedding for artwork {artwork_id}")
                    return False
                
                logger.info(f"[PROCESS] Embedding generated in {embedding_time:.2f}s, shape: {np.array(embedding).shape}")
                
                # Save the embedding to disk
                save_start = time.time()
                np.save(embedding_path, embedding)
                save_time = time.time() - save_start
                
                logger.info(f"[PROCESS] Saved embedding for artwork {artwork_id} to {embedding_path} in {save_time:.3f}s")
                
                # Generate and save GradCAM (if available)
                try:
                    logger.info(f"[PROCESS] Generating GradCAM for artwork {artwork_id}")
                    gradcam_start = time.time()
                    gradcam = calculate_gradcam_wrapper(image)
                    gradcam_time = time.time() - gradcam_start
                    
                    if gradcam is not None:
                        # Create the directory if it doesn't exist
                        os.makedirs(os.path.dirname(gradcam_path), exist_ok=True)
                        np.save(gradcam_path, gradcam)
                        logger.info(f"[PROCESS] Saved GradCAM for artwork {artwork_id} in {gradcam_time:.3f}s")
                    else:
                        logger.warning(f"[PROCESS] GradCAM generation returned None for {artwork_id}")
                except Exception as e:
                    logger.warning(f"[PROCESS] Error generating GradCAM: {str(e)}")
                
                # Generate and save Attention Maps (if available)
                try:
                    logger.info(f"[PROCESS] Generating Attention Map for artwork {artwork_id}")
                    attention_start = time.time()
                    attention = generate_heatmap_wrapper(image)
                    attention_time = time.time() - attention_start
                    
                    if attention is not None:
                        # Create the directory if it doesn't exist
                        os.makedirs(os.path.dirname(attention_path), exist_ok=True)
                        np.save(attention_path, attention)
                        logger.info(f"[PROCESS] Saved Attention Map for artwork {artwork_id} in {attention_time:.3f}s")
                    else:
                        logger.warning(f"[PROCESS] Attention Map generation returned None for {artwork_id}")
                except Exception as e:
                    logger.warning(f"[PROCESS] Error generating Attention Map: {str(e)}")
                
                # Add the embedding path to the metadata
                artwork_copy = artwork.copy()
                # Calculate the relative path from BASE_DIR
                relative_embedding_path = os.path.relpath(embedding_path, BASE_DIR)
                artwork_copy['embedding_path'] = relative_embedding_path.replace('\\', '/') # Replace backslashes with forward slashes for compatibility
                
                # Add information that the image was not saved locally
                artwork_copy['image_stored_locally'] = False
                
                # Add information about generation for debugging
                artwork_copy['_debug_info'] = {
                    'processed_by': 'preprocess_artworks.py',
                    'embedding_dir': EMBEDDINGS_DIR,
                    'data_dir': DATA_DIR,
                    'download_time': download_time,
                    'embedding_time': embedding_time,
                    'timestamp': time.time(),
                    'image_url': image_url
                }
                
                # Add to JSON
                logger.info(f"[PROCESS] Adding new artwork {artwork_id} to JSON")
                success = add_artwork_to_json(artwork_copy, embedding)
                logger.info(f"[PROCESS] Added artwork {artwork_id} to JSON: {success}")
                
                return success
            except Exception as e:
                logger.error(f"[PROCESS] Error generating embedding for {artwork_id}: {str(e)}")
                logger.error(traceback.format_exc())
                return False
                
        except Exception as e:
            logger.error(f"[PROCESS] Error processing artwork: {str(e)}")
            logger.error(traceback.format_exc())
            return False
    
    try:
        # ---- Get data from all sources ----
        
        # Uploaded user images
        logger.info("======== PROCESSING UPLOADED IMAGES ========")
        logger.info("Getting uploaded images from process_uploaded_images()")
        process_start = time.time()
        uploaded_metadata, uploaded_embeddings = process_uploaded_images()
        process_time = time.time() - process_start
        logger.info(f"Retrieved {len(uploaded_metadata)} uploaded artworks in {process_time:.2f}s")
        
        # Check the length of the lists
        if len(uploaded_metadata) != len(uploaded_embeddings):
            logger.warning(f"Length mismatch: metadata={len(uploaded_metadata)}, embeddings={len(uploaded_embeddings)}")
        
        # Process each uploaded image individually
        logger.info("Processing each uploaded image individually...")
        for i, metadata in enumerate(uploaded_metadata):
            artwork_id = metadata.get('id') or metadata.get('objectID')
            if i < len(uploaded_embeddings):
                embedding = uploaded_embeddings[i]
                # Save the embedding path
                embedding_path = os.path.join(EMBEDDINGS_DIR, f"{metadata.get('id')}.npy")
                logger.info(f"Saving embedding for uploaded artwork {artwork_id} to {embedding_path}")
                np.save(embedding_path, embedding)
                metadata_copy = metadata.copy()
                metadata_copy['embedding_path'] = embedding_path
                add_result = add_artwork_to_json(metadata_copy, embedding)
                logger.info(f"Added uploaded artwork {artwork_id} to JSON: {add_result}")
            else:
                logger.warning(f"Processing uploaded artwork {artwork_id} without embedding (index mismatch)")
                process_result = process_and_add_artwork(metadata, "user_upload")
                logger.info(f"Processed uploaded artwork {artwork_id}: {process_result}")
        
        # Met Museum
        logger.info("======== PROCESSING MET MUSEUM ARTWORKS ========")
        logger.info("Fetching artworks from Met Museum...")
        met_start = time.time()
        met_metadata, met_embeddings = fetch_met_artworks(limit=args.limit, hasImage=True)
        met_time = time.time() - met_start
        logger.info(f"Met Museum fetch completed in {met_time:.2f}s")
        
        if isinstance(met_metadata, tuple) and len(met_metadata) == 2:
            met_metadata, met_embeddings = met_metadata
            logger.info(f"Retrieved {len(met_metadata)} Met artworks with {len(met_embeddings)} embeddings")
            
            # Check the length of the lists
            if len(met_metadata) != len(met_embeddings):
                logger.warning(f"Length mismatch for Met: metadata={len(met_metadata)}, embeddings={len(met_embeddings)}")
            
            # Process each Met artwork individually
            logger.info("Processing each Met artwork individually...")
            met_success = 0
            met_failed = 0
            for i, artwork in enumerate(met_metadata):
                artwork_id = artwork.get('id') or artwork.get('objectID')
                logger.info(f"Processing Met artwork {i+1}/{len(met_metadata)}: {artwork_id}")
                process_result = process_and_add_artwork(artwork, "met")
                if process_result:
                    met_success += 1
                else:
                    met_failed += 1
            
            logger.info(f"Met artworks processing complete: {met_success} succeeded, {met_failed} failed")
        else:
            logger.error("Failed to retrieve Met artworks - returned data is not in expected format")
        
        # Rijksmuseum
        logger.info("======== PROCESSING RIJKSMUSEUM ARTWORKS ========")
        logger.info("Fetching artworks from Rijksmuseum...")
        rijks_start = time.time()
        rijks_metadata, rijks_embeddings = fetch_rijksmuseum_artworks(limit=args.limit, hasImage=True)
        rijks_time = time.time() - rijks_start
        logger.info(f"Rijksmuseum fetch completed in {rijks_time:.2f}s")
        
        if isinstance(rijks_metadata, tuple) and len(rijks_metadata) == 2:
            rijks_metadata, rijks_embeddings = rijks_metadata
            logger.info(f"Retrieved {len(rijks_metadata)} Rijksmuseum artworks with {len(rijks_embeddings)} embeddings")
            
            # Check the length of the lists
            if len(rijks_metadata) != len(rijks_embeddings):
                logger.warning(f"Length mismatch for Rijks: metadata={len(rijks_metadata)}, embeddings={len(rijks_embeddings)}")
            
            # Process each Rijksmuseum artwork individually
            logger.info("Processing each Rijksmuseum artwork individually...")
            rijks_success = 0
            rijks_failed = 0
            for i, artwork in enumerate(rijks_metadata):
                artwork_id = artwork.get('id') or artwork.get('objectID')
                logger.info(f"Processing Rijksmuseum artwork {i+1}/{len(rijks_metadata)}: {artwork_id}")
                process_result = process_and_add_artwork(artwork, "rijks")
                if process_result:
                    rijks_success += 1
                else:
                    rijks_failed += 1
            
            logger.info(f"Rijksmuseum artworks processing complete: {rijks_success} succeeded, {rijks_failed} failed")
        else:
            logger.error("Failed to retrieve Rijksmuseum artworks - returned data is not in expected format")
        
        # Art Institute of Chicago
        logger.info("======== PROCESSING ART INSTITUTE OF CHICAGO ARTWORKS ========")
        logger.info("Fetching artworks from Art Institute of Chicago...")
        aic_start = time.time()
        aic_metadata, aic_embeddings = fetch_aic_artworks(limit=args.limit, hasImage=True)
        aic_time = time.time() - aic_start
        logger.info(f"AIC fetch completed in {aic_time:.2f}s")
        
        if isinstance(aic_metadata, tuple) and len(aic_metadata) == 2:
            aic_metadata, aic_embeddings = aic_metadata
            logger.info(f"Retrieved {len(aic_metadata)} AIC artworks with {len(aic_embeddings)} embeddings")
            
            # Check the length of the lists
            if len(aic_metadata) != len(aic_embeddings):
                logger.warning(f"Length mismatch for AIC: metadata={len(aic_metadata)}, embeddings={len(aic_embeddings)}")
            
            # Process each AIC artwork individually
            logger.info("Processing each AIC artwork individually...")
            aic_success = 0
            aic_failed = 0
            for i, artwork in enumerate(aic_metadata):
                artwork_id = artwork.get('id') or artwork.get('objectID')
                logger.info(f"Processing AIC artwork {i+1}/{len(aic_metadata)}: {artwork_id}")
                process_result = process_and_add_artwork(artwork, "aic")
                if process_result:
                    aic_success += 1
                else:
                    aic_failed += 1
            
            logger.info(f"AIC artworks processing complete: {aic_success} succeeded, {aic_failed} failed")
        else:
            logger.error("Failed to retrieve AIC artworks - returned data is not in expected format")
        
        # Check the final result
        logger.info("======== PREPROCESSING COMPLETE ========")
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                final_data = json.load(f)
            
            total_items = final_data.get('total', 0)
            metadata_count = len(final_data.get('metadata', []))
            embeddings_count = len(final_data.get('embeddings', []))
            
            logger.info(f"Final JSON contains {total_items} items (metadata: {metadata_count}, embeddings: {embeddings_count})")
            
            if metadata_count != embeddings_count:
                logger.error(f"CRITICAL: Count mismatch in final JSON! metadata: {metadata_count}, embeddings: {embeddings_count}")
        except Exception as e:
            logger.error(f"Error checking final JSON: {str(e)}")
            logger.error(traceback.format_exc())
        
        total_time = time.time() - start_time
        logger.info(f"Total preprocessing time: {total_time:.2f} seconds")
        
    except Exception as e:
        logger.error(f"Error during preprocessing: {str(e)}")
        logger.error(traceback.format_exc())

def main():
    """Main preprocessing function"""
    parser = argparse.ArgumentParser(description="Preprocess artwork data for faster loading")
    # Limit argument might still be useful for other sources if they are uncommented later
    parser.add_argument("--limit", type=int, default=100, help="Limit artworks per museum (for sources other than Met)")
    parser.add_argument("--force", action="store_true", help="Force regeneration of embeddings")
    args = parser.parse_args()
    
    logger.info(f"Starting preprocessing for Met Museum (Dept 1:50, 11:150, 21:300), force={args.force}")
    
    start_time = time.time()
    
    # Load existing metadata if available (this part can remain as is)
    existing_metadata = {}
    if os.path.exists(METADATA_FILE) and not args.force:
        try:
            with open(METADATA_FILE, 'r') as f:
                existing_metadata = json.load(f)
                logger.info(f"Loaded {len(existing_metadata.get('items', []))} existing items from {METADATA_FILE}")
        except Exception as e:
            logger.error(f"Error loading existing metadata: {str(e)}")
            existing_metadata = {}
    
    # Initialize result containers
    all_processed_artworks = []
    all_processed_embeddings = []
    
    # Process Met artworks with specific department limits
    logger.info(f"Fetching Met artworks (Dept 1:50, 11:150, 21:300)")
    try:
        # Define the specific department limits
        met_dept_limits = {1: 50, 11: 150, 21: 300}
        # Функция fetch_met_artworks возвращает кортеж (metadata, embeddings)
        # Передаем новые лимиты вместо args.limit
        met_metadata, met_embeddings_loaded = fetch_met_artworks(dept_limits=met_dept_limits, hasImage=True)
        
        if met_metadata and len(met_metadata) > 0:
            logger.info(f"Received {len(met_metadata)} Met artworks metadata")
            # process_artworks_batch генерирует недостающие эмбеддинги
            met_processed, processed_embs = process_artworks_batch(met_metadata, args.force)
            all_processed_artworks.extend(met_processed)
            all_processed_embeddings.extend(processed_embs) # Используем эмбеддинги, обработанные batch функцией
            logger.info(f"Processed {len(met_processed)} Met artworks")
        else:
            logger.info("No Met artworks metadata received or it was empty")
    except Exception as e:
        logger.error(f"Error processing Met artworks: {str(e)}\n{traceback.format_exc()}")
    
    # --- Закомментировано: Обработка других музеев --- 
    # # Process Rijksmuseum artworks
    # logger.info(f"Fetching Rijksmuseum artworks (limit: {args.limit})")
    # try:
    #     # Функция возвращает кортеж (metadata, embeddings)
    #     rijks_metadata, rijks_embeddings = fetch_rijksmuseum_artworks(limit=args.limit, hasImage=True)
    #     
    #     if rijks_metadata and len(rijks_metadata) > 0:
    #         logger.info(f"Received {len(rijks_metadata)} Rijksmuseum artworks metadata")
    #         rijks_processed, processed_embs = process_artworks_batch(rijks_metadata, args.force)
    #         all_processed_artworks.extend(rijks_processed)
    #         all_processed_embeddings.extend(processed_embs)
    #         logger.info(f"Processed {len(rijks_processed)} Rijksmuseum artworks")
    #     else:
    #         logger.info("No Rijksmuseum artworks metadata received or it was empty")
    # except Exception as e:
    #     logger.error(f"Error processing Rijksmuseum artworks: {str(e)}\n{traceback.format_exc()}")
    # 
    # # Process Art Institute of Chicago artworks
    # logger.info(f"Fetching Art Institute of Chicago artworks (limit: {args.limit})")
    # try:
    #     # Функция возвращает кортеж (metadata, embeddings)
    #     aic_metadata, aic_embeddings = fetch_aic_artworks(limit=args.limit, hasImage=True)
    #     
    #     if aic_metadata and len(aic_metadata) > 0:
    #         logger.info(f"Received {len(aic_metadata)} AIC artworks metadata")
    #         aic_processed, processed_embs = process_artworks_batch(aic_metadata, args.force)
    #         all_processed_artworks.extend(aic_processed)
    #         all_processed_embeddings.extend(processed_embs)
    #         logger.info(f"Processed {len(aic_processed)} Art Institute of Chicago artworks")
    #     else:
    #         logger.info("No AIC artworks metadata received or it was empty")
    # except Exception as e:
    #     logger.error(f"Error processing Art Institute of Chicago artworks: {str(e)}\n{traceback.format_exc()}")
    # 
    # # Process user uploaded artworks
    # logger.info("Loading user uploaded artworks")
    # try:
    #     # Функция process_uploaded_images возвращает tuple с двумя списками (metadata, embeddings)
    #     uploaded_metadata, uploaded_embeddings = process_uploaded_images()
    #     
    #     if uploaded_metadata and len(uploaded_metadata) > 0:
    #         logger.info(f"Loaded {len(uploaded_metadata)} uploaded artworks metadata")
    #         
    #         # Непосредственно обрабатываем metadata, которая уже представляет собой список словарей
    #         uploaded_processed, processed_embs = process_artworks_batch(uploaded_metadata, args.force)
    #         
    #         all_processed_artworks.extend(uploaded_processed)
    #         all_processed_embeddings.extend(processed_embs)
    #         logger.info(f"Processed {len(uploaded_processed)} user uploaded artworks")
    #     else:
    #         logger.info("No uploaded artworks metadata found or it was empty")
    # except Exception as e:
    #     logger.error(f"Error processing user uploaded artworks: {str(e)}\n{traceback.format_exc()}")
    # --- Конец закомментированного блока ---
    
    # Prepare final metadata
    final_metadata = {
        'items': all_processed_artworks,
        'count': len(all_processed_artworks),
        'last_updated': time.time(),
    }
    
    # Save metadata to disk
    try:
        with open(METADATA_FILE, 'w') as f:
            json.dump(final_metadata, f, indent=2)
        logger.info(f"Saved metadata for {len(all_processed_artworks)} artworks to {METADATA_FILE}")
    except Exception as e:
        logger.error(f"Error saving metadata: {str(e)}")
    
    # Log completion
    end_time = time.time()
    logger.info(f"Preprocessing complete in {end_time - start_time:.2f} seconds")
    logger.info(f"Processed {len(all_processed_artworks)} total artworks (from Met Museum)")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 