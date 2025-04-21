import os
import torch
import timm
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import cv2



# === Настройка папок ===
image_dir = "images"
processed_dir = "processed"
os.makedirs(processed_dir, exist_ok=True)

# === Устройство ===
device = "cuda" if torch.cuda.is_available() else "cpu"

# === Загрузка модели из timm ===
# Используем 'vit_large_patch16_224' — предобученная модель.
model = timm.create_model('vit_large_patch16_224', pretrained=True)
model.to(device).eval()

# === Регистрируем метод для вычисления attention карты в каждом attention-модуле ===
def register_attention_hook(model):
    def get_attention_map(self, x):
        # x имеет форму [B, tokens, embed_dim]
        B, N, C = x.shape
        qkv = self.qkv(x)  # [B, tokens, 3*embed_dim]
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn_scores = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, tokens, tokens]
        attn_probs = attn_scores.softmax(dim=-1)
        return attn_probs  # [B, num_heads, tokens, tokens]

    for blk in model.blocks:
        # Регистрируем метод на объекте attention внутри блока.
        blk.attn.get_attention_map = get_attention_map.__get__(blk.attn, type(blk.attn))

register_attention_hook(model)

# === Преобразование изображения для модели ===
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.5]*3, std=[0.5]*3),
])

# --- Function used by API ---
def generate_heatmap(img_pil):
    """Generates an attention heatmap for a given PIL image using ViT."""
    # Ensure image is RGB
    if img_pil.mode != 'RGB':
        img_pil = img_pil.convert('RGB')
        
    input_tensor = transform(img_pil).unsqueeze(0).to(device)  # [1, 3, 224, 224]

    # --- Patch embedding и подготовка ---
    x = model.patch_embed(input_tensor)  # [B, num_patches, embed_dim]
    B = x.shape[0]
    cls_tokens = model.cls_token.expand(B, -1, -1)
    x = torch.cat((cls_tokens, x), dim=1)
    x = x + model.pos_embed
    x = model.pos_drop(x)

    # Проходим через все блоки, кроме последнего
    for blk in model.blocks[:-1]:
        x = blk(x)
    
    # Для последнего блока получаем нормализованный вход, который подается на attention
    normed_x = model.blocks[-1].norm1(x)
    # Вычисляем attention карту с помощью зарегистрированного метода
    attn = model.blocks[-1].attn.get_attention_map(normed_x)  # [1, heads, tokens, tokens]
    attn = attn[0]  # [heads, tokens, tokens]
    # Извлекаем внимание от CLS-токена к остальным токенам
    cls_attn = attn[:, 0, :]  # [heads, tokens]
    
    # Усредняем по головам и отсоединяем, чтобы можно было конвертировать в numpy
    attn_avg = cls_attn.mean(0).detach().cpu().numpy()  # shape: [tokens]
    # Отбрасываем первый токен (CLS)
    attn_patches = attn_avg[1:]
    grid_size = int(np.sqrt(attn_patches.shape[0]))  # ожидается 14
    attn_grid = attn_patches.reshape(grid_size, grid_size) # This is the raw heatmap data
    
    # Normalize the raw heatmap (optional, but good practice)
    attn_grid_norm = attn_grid - attn_grid.min()
    if attn_grid_norm.max() > 0:
        attn_grid_norm = attn_grid_norm / attn_grid_norm.max()
    else:
        attn_grid_norm = np.zeros_like(attn_grid_norm)
        
    return attn_grid_norm # Return the 2D numpy array heatmap

# --- Batch processing script part (Consider moving to separate file) --- 
# === Функция визуализации attention карты (для скрипта) ===
def visualize_attention_map_script(cls_attn, image_pil, save_path):
    """
    cls_attn: тензор формы [heads, tokens] — внимание CLS-токена к остальным токенам.
    При патчах 16x16 для изображения 224x224 ожидается 1 (CLS) + 14*14 = 197 токенов.
    Мы отбрасываем первый (CLS) и формируем карту размером 14x14.
    """
    # Усредняем по головам и отсоединяем, чтобы можно было конвертировать в numpy
    attn_avg = cls_attn.mean(0).detach().cpu().numpy()  # shape: [tokens]
    # Отбрасываем первый токен (CLS)
    attn_patches = attn_avg[1:]
    grid_size = int(np.sqrt(attn_patches.shape[0]))  # ожидается 14
    attn_grid = attn_patches.reshape(grid_size, grid_size)
    
    # Нормализуем в диапазон [0, 1]
    attn_grid = attn_grid - attn_grid.min()
    if attn_grid.max() > 0:
        attn_grid = attn_grid / attn_grid.max()
    else:
        attn_grid = np.zeros_like(attn_grid)
    
    # Масштабируем до 0-255 и применяем colormap
    attn_grid_uint8 = np.uint8(attn_grid * 255)
    attn_color = cv2.applyColorMap(attn_grid_uint8, cv2.COLORMAP_JET)
    # Изменяем размер карты до размера исходного изображения
    attn_color = cv2.resize(attn_color, image_pil.size)
    
    # Преобразуем исходное изображение в массив (RGB)
    image_np = np.array(image_pil.convert("RGB"))
    # Накладываем тепловую карту
    overlay = cv2.addWeighted(image_np, 0.6, attn_color, 0.4, 0)
    plt.imsave(save_path, overlay)

# === Функция обработки одного изображения (для скрипта) ===
def process_image_script(img_path):
    filename = os.path.splitext(os.path.basename(img_path))[0]
    image = Image.open(img_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Re-calculate cls_attn here as it's needed for visualization
    x = model.patch_embed(input_tensor)
    B = x.shape[0]
    cls_tokens = model.cls_token.expand(B, -1, -1)
    x = torch.cat((cls_tokens, x), dim=1)
    x = x + model.pos_embed
    x = model.pos_drop(x)
    for blk in model.blocks[:-1]:
        x = blk(x)
    normed_x = model.blocks[-1].norm1(x)
    attn = model.blocks[-1].attn.get_attention_map(normed_x)
    cls_attn = attn[0][:, 0, :]

    # Сохраняем визуализацию
    save_path = os.path.join(processed_dir, f"{filename}.jpg")
    visualize_attention_map_script(cls_attn, image.resize((224, 224)), save_path)

# === Обработка всех изображений в папке (для скрипта) ===
if __name__ == "__main__":
    image_dir = "images"
    processed_dir = "processed"
    os.makedirs(processed_dir, exist_ok=True)

    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    for img_file in tqdm(image_files, desc="Generating attention maps (Script)"):
        try:
            process_image_script(os.path.join(image_dir, img_file))
        except Exception as e:
            print(f"Ошибка при обработке {img_file} в скрипте: {e}") 