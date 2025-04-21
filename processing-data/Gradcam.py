import os
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
from transformers import CLIPImageProcessor, CLIPModel
import matplotlib.pyplot as plt
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


print("Loading CLIP model...")
clip_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_model.eval()
print("CLIP model loaded.")


print("Loading ResNet model...")

weights = models.ResNet50_Weights.IMAGENET1K_V1
resnet = models.resnet50(weights=weights).to(device)
resnet.eval()
target_layer = resnet.layer4[-1]
print("ResNet model loaded.")


clip_transform = clip_processor

resnet_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


class GradCamHooks:
    def __init__(self, model_layer):
        self.activations = None
        self.gradients = None
        self.forward_hook_handle = model_layer.register_forward_hook(self.forward_hook)
        self.backward_hook_handle = model_layer.register_backward_hook(self.backward_hook)

    def forward_hook(self, module, input, output):
        self.activations = output.detach()

    def backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def release(self):
        self.forward_hook_handle.remove()
        self.backward_hook_handle.remove()
        self.activations = None
        self.gradients = None


def calculate_gradcam(img_pil):
    if img_pil is None:
        print("Error: Received None image for Grad-CAM calculation.")
        return None

    if img_pil.mode != 'RGB':
        try:
            img_pil = img_pil.convert('RGB')
        except Exception as e:
             print(f"Error converting image to RGB: {e}")
             return None

    input_tensor = resnet_transform(img_pil).unsqueeze(0).to(device)

    grad_cam_hooks = GradCamHooks(target_layer)

    try:
        output = resnet(input_tensor)
        class_idx = output.argmax().item()

        resnet.zero_grad()
        output[0, class_idx].backward()

        activations = grad_cam_hooks.activations
        gradients = grad_cam_hooks.gradients

        if gradients is None or activations is None:
            print("Error: Hooks did not capture gradients or activations.")
            return None
        if gradients.shape[0] != 1 or activations.shape[0] != 1:
             print(f"Warning: Unexpected batch size in hooks. Grads: {gradients.shape}, Acts: {activations.shape}")
             if gradients.shape[0] > 0: gradients = gradients[0].unsqueeze(0)
             if activations.shape[0] > 0: activations = activations[0].unsqueeze(0)
             if gradients is None or activations is None or gradients.shape[0] != 1 or activations.shape[0] != 1 :
                 print("Error: Could not resolve batch size issue in hooks.")
                 return None


        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1).squeeze(0)
        cam = torch.relu(cam)
        max_val = torch.max(cam)
        if max_val > 1e-6:
            cam = cam / max_val
        else:
            print("Warning: Grad-CAM map is near zero.")
            h, w = cam.shape
            return np.zeros((h, w), dtype=np.float32)

        return cam.cpu().numpy()

    except Exception as e:
        print(f"Error during Grad-CAM calculation: {e}")
        return None
    finally:
        grad_cam_hooks.release()


def process_image_script(img_path, embedding_dir, gradcam_dir):
    filename = os.path.splitext(os.path.basename(img_path))[0]

    try:
        image_clip = Image.open(img_path).convert("RGB")
        inputs = clip_transform(images=image_clip, return_tensors="pt").to(device)
        with torch.no_grad():
            image_features = clip_model.get_image_features(**inputs)
        image_features = F.normalize(image_features, p=2, dim=-1)
        torch.save(image_features.cpu(), f"{embedding_dir}/{filename}.pt")
        image_resnet_pil = Image.open(img_path).convert("RGB")
        cam_np = calculate_gradcam(image_resnet_pil)

        if cam_np is not None:

             target_size = (224, 224)
             cam_resized = cv2.resize(cam_np, target_size, interpolation=cv2.INTER_LINEAR)

             heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
             heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

             img_resized_pil = image_resnet_pil.resize(target_size)
             img_np = np.array(img_resized_pil)
             superimposed_img = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)

             cv2.imwrite(f"{gradcam_dir}/{filename}.jpg", cv2.cvtColor(superimposed_img, cv2.COLOR_RGB2BGR))
        else:
             print(f"Не удалось сгенерировать Grad-CAM для {filename}")


    except Exception as e:
        print(f"Ошибка при обработке {img_path} в скрипте: {e}")

if __name__ == "__main__":
    print("Starting batch processing...")
    image_dir = "images"
    embedding_dir = "embeddings"
    gradcam_dir = "gradcam"
    os.makedirs(embedding_dir, exist_ok=True)
    os.makedirs(gradcam_dir, exist_ok=True)

    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    print(f"Found {len(image_files)} images in '{image_dir}'.")

    for img_file in tqdm(image_files, desc="Processing images (Script)"):
        img_full_path = os.path.join(image_dir, img_file)
        process_image_script(img_full_path, embedding_dir, gradcam_dir)

    print("Batch processing finished.")
