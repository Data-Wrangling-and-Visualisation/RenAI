import torch
import torchvision.transforms as T
from PIL import Image
import os
from tqdm import tqdm
# import tensorflow as tf # TensorFlow больше не нужен для API
import numpy as np

# --- Global DINOv2 Setup for API Use ---
def _load_dinov2_model():
    """Loads the DINOv2 model and puts it on the appropriate device."""
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Embeddings API] Loading DINOv2 model onto device: {device}")
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14', verbose=False).to(device)
        model.eval()
        print("[Embeddings API] DINOv2 model loaded successfully.")
        return model, device
    except Exception as e:
        print(f"[Embeddings API] ERROR: Failed to load DINOv2 model: {e}")
        print("[Embeddings API] Embeddings will not be generated.")
        return None, None

# Load model globally when the module is imported
MODEL_DINO_API, DEVICE_API = _load_dinov2_model()

# Define transform globally as well
TRANSFORM_DINO_API = T.Compose([
    T.Resize(518, interpolation=T.InterpolationMode.BICUBIC),
    T.CenterCrop(518),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

# --- Function used by API (Using DINOv2) ---
def generate_embedding(image_pil):
    """Generates an embedding for a given PIL image using DINOv2."""
    if MODEL_DINO_API is None or DEVICE_API is None:
        print("[Embeddings API] DINOv2 model not loaded. Returning None.")
        return None

    try:
        # Ensure image is RGB
        if image_pil.mode != 'RGB':
            image_pil = image_pil.convert('RGB')

        # Apply the DINOv2 transform
        img_tensor = TRANSFORM_DINO_API(image_pil).unsqueeze(0).to(DEVICE_API)

        # Generate embedding
        with torch.no_grad():
            embedding = MODEL_DINO_API(img_tensor) # [1, 1024]
        
        # Return as NumPy array (remove batch dimension)
        return embedding.cpu().numpy()[0]
    except Exception as e:
        print(f"[Embeddings API] Error generating DINOv2 embedding: {e}")
        import traceback
        traceback.print_exc()
        return None

# --- Batch processing script part (Using DINOv2 - Now uses the same model/transform) ---
if __name__ == "__main__":
    print("Running embeddings.py as a script for batch processing (DINOv2)...")
    
    # Settings specific to the script
    # Make sure these paths are correct for your script execution environment
    image_dir_script = "../server/images_for_processing" # Example: point to where script images are
    embedding_dir_script = "../server/data/embeddings" # Example: save directly to where API expects .npy
    os.makedirs(embedding_dir_script, exist_ok=True)
    os.makedirs(image_dir_script, exist_ok=True) # Ensure image dir exists
    
    # Model is already loaded globally, just reference it
    if MODEL_DINO_API is None:
        print("DINOv2 model failed to load globally. Exiting script.")
        exit()
        
    model_dino_script = MODEL_DINO_API
    transform_dino_script = TRANSFORM_DINO_API
    device_script = DEVICE_API
    print(f"Using globally loaded DINOv2 model on device: {device_script}")

    def process_image_script(img_path):
        # Extract filename without extension to use as ID
        filename = os.path.splitext(os.path.basename(img_path))[0]
        output_path = os.path.join(embedding_dir_script, f"{filename}.npy")

        # Skip if embedding already exists
        if os.path.exists(output_path):
            # print(f"Skipping {filename}, embedding already exists.")
            return
            
        try:
            image = Image.open(img_path).convert("RGB")
            # Use the global transform and device
            img_tensor = transform_dino_script(image).unsqueeze(0).to(device_script)

            with torch.no_grad():
                embedding = model_dino_script(img_tensor)  # [1, 1024]
            
            # Save as .npy directly
            np.save(output_path, embedding.cpu().numpy()[0]) # Save the 1D array
            # print(f"Saved embedding for {filename}.npy")

        except Exception as e:
             print(f"Error processing {img_path}: {e}")

    if not os.path.isdir(image_dir_script):
        print(f"Error: Image directory '{image_dir_script}' not found. Cannot run batch processing.")
    else:
        image_files = [f for f in os.listdir(image_dir_script) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
        print(f"Found {len(image_files)} images in '{image_dir_script}'. Processing...")
        # Use tqdm for progress bar
        for img_file in tqdm(image_files, desc="Processing embeddings (DINOv2 Script)"):
            process_image_script(os.path.join(image_dir_script, img_file))
        print("Batch processing finished.")
