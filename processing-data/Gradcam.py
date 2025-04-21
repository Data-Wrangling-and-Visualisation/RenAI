import os
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import matplotlib.pyplot as plt
from tqdm import tqdm

# ==== Папки ====
image_dir = "images"
embedding_dir = "embeddings"
gradcam_dir = "gradcam"

os.makedirs(embedding_dir, exist_ok=True)
os.makedirs(gradcam_dir, exist_ok=True)

# ==== Модели ====
# CLIP
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()

# ResNet для Grad-CAM
resnet = models.resnet50(pretrained=True)
resnet.eval()
target_layer = resnet.layer4[-1]

# Преобразования
clip_transform = clip_processor.feature_extractor
resnet_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Хуки для Grad-CAM
activations = None
gradients = None

def forward_hook(module, input, output):
    global activations
    activations = output.detach()

def backward_hook(module, grad_input, grad_output):
    global gradients
    gradients = grad_output[0].detach()

target_layer.register_forward_hook(forward_hook)
target_layer.register_backward_hook(backward_hook)

# --- Function used by API --- 
def calculate_gradcam(img_pil):
    """Calculates Grad-CAM map for a given PIL image using ResNet50."""
    # Ensure image is RGB
    if img_pil.mode != 'RGB':
        img_pil = img_pil.convert('RGB')
        
    input_tensor = resnet_transform(img_pil).unsqueeze(0)
    output = resnet(input_tensor)
    class_idx = output.argmax().item()

    resnet.zero_grad()
    output[0, class_idx].backward()

    # Ensure gradients and activations were captured
    if gradients is None or activations is None:
        raise RuntimeError("Hooks did not capture gradients or activations.")

    weights = gradients.mean(dim=(2, 3), keepdim=True)
    cam = (weights * activations).sum(dim=1).squeeze()
    cam = torch.relu(cam)
    
    # Handle cases where cam might be all zeros after ReLU
    max_val = torch.max(cam)
    if max_val > 0:
        cam = cam / max_val
    else:
        # Return a zero map or handle as an error case if appropriate
        print("Warning: Grad-CAM map is all zeros.")
        # Get spatial dimensions from activations
        h, w = activations.shape[-2:]
        return np.zeros((h, w), dtype=np.float32)
        
    return cam.numpy()

# --- Batch processing script part (Consider moving to separate file) --- 
# Обработка одного изображения (для скрипта)
def process_image_script(img_path):
    filename = os.path.splitext(os.path.basename(img_path))[0]

    # === Эмбеддинг CLIP ===
    image_clip = Image.open(img_path).convert("RGB")
    inputs = clip_processor(images=image_clip, return_tensors="pt")
    with torch.no_grad():
        image_features = clip_model.get_image_features(**inputs)
    image_features = F.normalize(image_features, p=2, dim=-1)  # L2-нормализация
    torch.save(image_features, f"{embedding_dir}/{filename}.pt")

    # === Grad-CAM ===
    image_resnet = Image.open(img_path).convert("RGB")
    input_tensor = resnet_transform(image_resnet).unsqueeze(0)
    output = resnet(input_tensor)
    class_idx = output.argmax().item()

    resnet.zero_grad()
    output[0, class_idx].backward()

    weights = gradients.mean(dim=(2, 3), keepdim=True)
    cam = (weights * activations).sum(dim=1).squeeze()
    cam = torch.relu(cam)
    cam = cam / cam.max()
    cam = cam.numpy()

    cam = cv2.resize(cam, (224, 224))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    img_np = np.array(image_resnet.resize((224, 224)))
    superimposed_img = cv2.addWeighted(img_np, 0.5, heatmap, 0.5, 0)
    cv2.imwrite(f"{gradcam_dir}/{filename}.jpg", superimposed_img[:, :, ::-1])  # BGR -> RGB

# === Обработка всей папки (для скрипта) ===
if __name__ == "__main__": # Only run batch processing if script is executed directly
    image_dir = "images"
    embedding_dir = "embeddings"
    gradcam_dir = "gradcam"
    os.makedirs(embedding_dir, exist_ok=True)
    os.makedirs(gradcam_dir, exist_ok=True)

    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    for img_file in tqdm(image_files, desc="Processing images (Script)"):
        try:
            process_image_script(os.path.join(image_dir, img_file))
        except Exception as e:
            print(f"Ошибка при обработке {img_file} в скрипте: {e}")
